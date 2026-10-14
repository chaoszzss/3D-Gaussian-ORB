#!/usr/bin/env python3
"""Fixed 32-ray CPU image diagnostic; no model fitting or depth-GT claim."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
from collections import Counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'reports/surface_quality_v2_joint015_final_aa_floor.json'
DATA = ROOT / 'data/garage_surface_v1/block_015'
OUT = ROOT / 'reports/floor_depth_photometric_diagnostic.json'
OFFSETS = np.array([(x, y) for y in range(-3, 4) for x in range(-3, 4)], dtype=float)
METHODS = ('center_patch', 'parallel_plane_warp')
DEPTHS = ('fixed_old_lidar', 'final_native_expected')
TEXTURE_STD_U8 = 3.0
PINS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def state(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def pin(path):
    path = path.resolve()
    require(ROOT in path.parents, 'Input outside project')
    if str(path) in PINS:
        require(state(path) == PINS[str(path)]['state'], 'Input changed')
        return path
    before = state(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    require(state(path) == before, 'Input changed while hashing')
    PINS[str(path)] = {'sha256': h.hexdigest(), 'state': before}
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def camera(row):
    return {**{k: row[k] for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h')},
            'camera_to_worlds_ns': np.asarray(row['transform_matrix'], float)[:3].tolist()}


def basis(cam):
    pose = np.asarray(cam['camera_to_worlds_ns'], float)
    return pose[:, :3] @ np.diag([1., -1., -1.]), pose[:, 3]


def world_rays(cam, pixels):
    r, _ = basis(cam)
    p = np.asarray(pixels, float).reshape(-1, 2)
    cv = np.column_stack(((p[:, 0] + .5 - cam['cx']) / cam['fl_x'],
                          (p[:, 1] + .5 - cam['cy']) / cam['fl_y'], np.ones(len(p))))
    return cv @ r.T


def project(cam, points):
    r, center = basis(cam)
    cv = (np.asarray(points).reshape(-1, 3) - center) @ np.linalg.inv(r).T
    z = cv[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        xy = np.column_stack((cam['fl_x'] * cv[:, 0] / z + cam['cx'] - .5,
                              cam['fl_y'] * cv[:, 1] / z + cam['cy'] - .5))
    return xy, z


def load_image(row):
    with Image.open(pin(DATA / row['file_path'])) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float64)
    require(rgb.shape == (row['h'], row['w'], 3), 'Image shape differs')
    gray = rgb @ np.array([.299, .587, .114])
    with Image.open(pin(DATA / row['mask_path'])) as image:
        mask = np.asarray(image.convert('L')) > 0
    require(mask.shape == gray.shape, 'Mask shape differs')
    return gray, mask


def sample(gray, mask, xy):
    h, w = gray.shape
    if not np.isfinite(xy).all() or not ((xy[:, 0] >= 0) & (xy[:, 0] < w-1)
                                            & (xy[:, 1] >= 0) & (xy[:, 1] < h-1)).all():
        return None, 'outside_view'
    x, y = np.floor(xy).astype(int).T
    if not (mask[y, x] & mask[y, x+1] & mask[y+1, x] & mask[y+1, x+1]).all():
        return None, 'environment_mask_excluded'
    dx, dy = (xy - np.column_stack((x, y))).T
    values = (1-dy)*((1-dx)*gray[y, x]+dx*gray[y, x+1]) + dy*((1-dx)*gray[y+1, x]+dx*gray[y+1, x+1])
    return values, None


def compare(reference, candidate, invalid):
    result = {'status': 'unknown', 'reason': invalid, 'ncc': None,
              'target_std_u8': float(np.std(reference)) if reference is not None else None,
              'neighbor_std_u8': float(np.std(candidate)) if candidate is not None else None}
    if reference is None or candidate is None:
        return result
    residual = candidate - reference
    centered = residual - np.median(residual)
    abs_centered = np.abs(centered)
    result.update(median_abs_error_u8=float(np.median(np.abs(residual))),
                  median_abs_offset_corrected_error_u8=float(np.median(abs_centered)),
                  clipped_mean_offset_corrected_error_u8=float(np.mean(np.minimum(abs_centered, 20.))))
    if min(result['target_std_u8'], result['neighbor_std_u8']) < TEXTURE_STD_U8:
        result['reason'] = 'low_texture'
        return result
    a, b = reference-reference.mean(), candidate-candidate.mean()
    result.update(status='valid', reason=None, ncc=float(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))
    return result


def paired_vote(scores):
    old, final = (scores[name] for name in DEPTHS)
    if any(x['status'] != 'valid' for x in (old, final)):
        return {'vote': 'unknown', 'reason': 'both hypotheses must have valid textured patches'}
    delta = final['ncc'] - old['ncc']
    residual_advantage = old['median_abs_offset_corrected_error_u8'] - final['median_abs_offset_corrected_error_u8']
    winner = 'inconclusive'
    if delta >= .15 and final['ncc'] >= .6 and residual_advantage >= 1.:
        winner = 'final_native_expected'
    if delta <= -.15 and old['ncc'] >= .6 and residual_advantage <= -1.:
        winner = 'fixed_old_lidar'
    return {'vote': winner, 'ncc_final_minus_old': delta,
            'offset_corrected_median_error_old_minus_final_u8': residual_advantage}


def analytic_checks():
    cam = {'fl_x': 320., 'fl_y': 320., 'cx': 320., 'cy': 270., 'w': 640, 'h': 540,
           'camera_to_worlds_ns': np.column_stack((np.eye(3), np.zeros(3))).tolist()}
    pixels = np.array([[100., 200.], [400., 450.]])
    points = world_rays(cam, pixels) * 4.
    np.testing.assert_allclose(project(cam, points)[0], pixels, atol=1e-10)
    neighbor = dict(cam)
    pose = np.array(cam['camera_to_worlds_ns']); pose[0, 3] = .5
    neighbor['camera_to_worlds_ns'] = pose.tolist()
    np.testing.assert_allclose(project(neighbor, points)[0], pixels + [-40., 0.], atol=1e-10)
    values, reason = sample(np.arange(100.).reshape(10, 10), np.ones((10, 10), bool), np.array([[2.5, 3.5]]))
    require(reason is None and abs(values[0]-37.5) < 1e-10, 'Bilinear analytic check failed')
    score = compare(np.arange(49.), np.arange(49.)+15, None)
    require(score['ncc'] > .999999 and score['median_abs_offset_corrected_error_u8'] == 0, 'Brightness check failed')
    return {'status': 'passed', 'cases': ['NS_CV_pixel_closure', 'known_baseline_disparity',
                                         'bilinear_sampling', 'brightness_offset_invariance']}


def main():
    started, cpu_started = time.monotonic(), time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    signal.alarm(120)
    available = sorted(os.sched_getaffinity(0))
    preferred = [i for i in (20, 21, 22) if i in available]
    os.sched_setaffinity(0, preferred if len(preferred) == 3 else available[:3])
    require(not OUT.exists(), 'Output already exists; refusing overwrite')
    checks = analytic_checks()
    pin(Path(__file__))
    frozen = json.loads(pin(FROZEN).read_text())
    metadata_path = pin(DATA / 'transforms.json')
    require(PINS[str(metadata_path)]['sha256'] == frozen['inputs'][str(metadata_path)]['sha256'], 'Frozen source metadata drift')
    metadata = json.loads(metadata_path.read_text())
    require(metadata['pixel_center_convention'] == 'half_integer; principal point +0.5 from OpenCV rectification', 'Unknown pixel convention')
    require(frozen['final_native_validation']['status'] == 'passed' and frozen['final_native_validation']['rays'] == 32, 'Native expected-depth validation missing')
    rows = {(str(r['group_id']), r['camera_id']): r for r in metadata['frames']}
    trains, devs, tests = (set(metadata[k]) for k in ('train_filenames', 'val_filenames', 'test_filenames'))
    require(not (trains & devs or trains & tests or devs & tests), 'Split overlap')
    selected = frozen['selection']['frames']
    require(len(selected) == 4 and sum(len(s['rays']) for s in selected) == 32, 'Frozen ray count differs')

    # Freeze all neighbors using poses and timestamps BEFORE reading candidate
    # depths, loading any patches or computing a depth-dependent score.
    neighbors, selection = {}, []
    for s in selected:
        key = tuple(s['frame_key'])
        target = rows[key]
        require(target['file_path'] in devs, 'Target is not DEV')
        require((target['w'], target['h']) == (640, 540), 'Expected frozen 640 images')
        cam = camera(target)
        require(np.allclose(cam['camera_to_worlds_ns'], s['camera']['camera_to_worlds_ns'], atol=4e-6, rtol=0), 'Frozen target pose drift')
        require(all(abs(cam[k]-s['camera'][k]) < 1e-6 for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h')), 'Frozen target intrinsics drift')
        center = basis(cam)[1]
        eligible = []
        for n in metadata['frames']:
            if n['file_path'] not in trains or n['camera_id'] != target['camera_id'] or n['view_id'] != target['view_id']:
                continue
            baseline = float(np.linalg.norm(basis(camera(n))[1] - center))
            dt = (n['timestamp_ns'] - target['timestamp_ns']) / 1e9
            if .15 <= baseline <= .75:
                eligible.append((abs(dt), str(n['group_id']), baseline, dt, n))
        chosen = sorted(eligible, key=lambda item: (item[0], item[1]))[:4]
        neighbors[key] = chosen
        selection.append({'frame_key': list(key), 'target_file': target['file_path'],
            'target_camera_used': s['camera'], 'target_source_pose_ns': target['transform_matrix'],
            'eligible_neighbors': len(eligible), 'neighbors': [
                {'file_path': n['file_path'], 'group_id': n['group_id'], 'camera_id': n['camera_id'],
                 'timestamp_ns': n['timestamp_ns'], 'time_delta_s': dt, 'baseline_m': baseline,
                 'camera': camera(n), 'split': 'TRAIN'} for _, _, baseline, dt, n in chosen]})
    selection_sha = digest(selection)
    current = {tuple(f['frame_key']): f for f in frozen['current']['frames']}
    results, projection_errors = [], []
    for s in selected:
        key, cam = tuple(s['frame_key']), s['camera']
        target = rows[key]
        target_gray, target_mask = load_image(target)
        images = {n['file_path']: load_image(n) for *_, n in neighbors[key]}
        actual = {tuple(r['pixel_xy']): r for r in current[key]['rays']}
        require(set(actual) == {tuple(r['pixel_xy']) for r in s['rays']}, 'Current/selection rays differ')
        _, center = basis(cam)
        for ray in s['rays']:
            pixel = np.array(ray['pixel_xy'], float)
            native = actual[tuple(pixel)]
            require(native['reference_camera_z_m'] == ray['reference_camera_z_m'], 'Frozen reference depth mismatch')
            depths = {'fixed_old_lidar': ray['reference_camera_z_m'], 'final_native_expected': native['expected_camera_z_m']}
            require(all(np.isfinite(d) and d > 0 for d in depths.values()), 'Invalid frozen depth')
            target_xy = pixel + OFFSETS
            target_patch, target_invalid = sample(target_gray, target_mask, target_xy)
            direction = world_rays(cam, [pixel])[0]
            normal = np.asarray(ray['reference_normal'], float)
            require(np.linalg.norm(normal) > 1e-6, 'Invalid frozen normal')
            normal /= np.linalg.norm(normal)
            observations = []
            for _, _, baseline, dt, n in neighbors[key]:
                neighbor_cam = camera(n)
                gray, mask = images[n['file_path']]
                observation = {'train_file': n['file_path'], 'time_delta_s': dt, 'baseline_m': baseline, 'hypotheses': {}}
                for name, depth in depths.items():
                    world = center + depth * direction
                    check, _ = project(cam, [world])
                    projection_errors.append(float(np.max(np.abs(check[0] - pixel))))
                    xy, z = project(neighbor_cam, [world])
                    entry = {'camera_z_m': depth, 'world_xyz_m': world.tolist(), 'train_center_xy_integer_pixel_coordinates': xy[0].tolist(),
                             'train_camera_z_m': float(z[0]), 'scores': {}}
                    for method in METHODS:
                        invalid = target_invalid
                        if method == 'center_patch':
                            coords, patch_z = xy[0] + OFFSETS, z
                        else:
                            rays = world_rays(cam, target_xy)
                            denom = rays @ normal
                            if np.any(np.abs(denom) < 1e-8):
                                entry['scores'][method] = compare(target_patch, None, 'plane_parallel_ray')
                                continue
                            t = float(np.dot(world-center, normal)) / denom
                            coords, patch_z = project(neighbor_cam, center + t[:, None]*rays)
                            if np.any(t <= 0):
                                invalid = 'plane_behind_target'
                        if np.any(patch_z <= 0):
                            invalid = 'behind_neighbor'
                        values, reason = (None, invalid) if invalid else sample(gray, mask, coords)
                        entry['scores'][method] = compare(target_patch, values, reason)
                        entry['scores'][method]['projected_patch_xy'] = coords.tolist()
                    observation['hypotheses'][name] = entry
                observation['paired'] = {method: paired_vote({name: observation['hypotheses'][name]['scores'][method] for name in DEPTHS}) for method in METHODS}
                a, b = [np.array(observation['hypotheses'][name]['train_center_xy_integer_pixel_coordinates']) for name in DEPTHS]
                observation['hypothesis_center_separation_px'] = float(np.linalg.norm(a-b))
                observations.append(observation)
            method_support = {}
            for method in METHODS:
                votes = Counter(o['paired'][method]['vote'] for o in observations)
                decision = 'unknown' if votes['unknown'] == len(observations) else 'inconclusive'
                for name in DEPTHS:
                    other = next(d for d in DEPTHS if d != name)
                    if votes[name] >= 2 and votes[other] == 0:
                        decision = name
                method_support[method] = {'decision': decision, 'votes': dict(votes)}
            choices = [method_support[m]['decision'] for m in METHODS]
            joint = choices[0] if choices[0] == choices[1] else 'inconclusive_method_disagreement'
            results.append({'frame_key': list(key), 'pixel_xy': ray['pixel_xy'], 'flat_index': ray['flat_index'],
                'reference_normal_world': ray['reference_normal'], 'depths_camera_z_m': depths,
                'signed_depth_difference_m': depths[DEPTHS[1]]-depths[DEPTHS[0]],
                'frozen_conditioned_depth_class': native['conditioned_depth_class'],
                'target_patch_std_u8': float(np.std(target_patch)) if target_patch is not None else None,
                'observations': observations, 'method_support': method_support, 'image_support_decision': joint})
    require(len(results) == 32 and max(projection_errors) < 1e-6, 'Ray/pixel closure failed')
    def summarize(rays):
        return {'rays': len(rays), 'joint_decisions': dict(Counter(r['image_support_decision'] for r in rays)),
            'per_method': {m: dict(Counter(r['method_support'][m]['decision'] for r in rays)) for m in METHODS},
            'per_neighbor_votes': {m: dict(Counter(o['paired'][m]['vote'] for r in rays for o in r['observations'])) for m in METHODS},
            'patch_status': {m: {d: dict(Counter(o['hypotheses'][d]['scores'][m]['reason'] or 'valid'
                            for r in rays for o in r['observations'])) for d in DEPTHS} for m in METHODS}}
    wrong = [r for r in results if abs(r['signed_depth_difference_m']) > .15]
    require(all(state(Path(p)) == info['state'] for p, info in PINS.items()), 'Inputs changed during diagnostic')
    report = {'schema_version': 1, 'status': 'completed_fixed_32_ray_image_diagnostic', 'gpu_used': False,
        'cpu_threads_limit': 3, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'cpu_budget_seconds': 120,
        'high_fidelity_accepted': False, 'depth_ground_truth_established': False,
        'protocol': {'patch_size': 7, 'patch_samples': 49, 'intensity': '0.299R+0.587G+0.114B, 0..255 image intensities; bilinear sampling',
            'pixel_convention': 'NS/OpenGL poses; unproject x+.5,y+.5 with supplied NS K; project to image array coordinates subtract .5',
            'neighbors': 'same physical camera and center view, TRAIN only; baseline inclusive 0.15 to 0.75m; sorted absolute time then group ID; first up to4; fixed before reading current depths/scores',
            'selection_sha256': selection_sha, 'frozen_selection_sha256': digest(frozen['selection']),
            'center_patch': '7x7 translation-only patch around depth-projected center; no adaptive search',
            'parallel_plane_warp': 'same frozen reference normal for BOTH hypotheses; plane through each depth ray point; exact ray/plane intersection and reprojection of target 7x7 grid; no plane fitting',
            'low_texture_threshold_std_u8': TEXTURE_STD_U8,
            'paired_vote': 'both hypotheses valid; winner NCC>=.6, NCC advantage>=.15, median absolute offset-corrected residual advantage>=1 intensity unit; otherwise inconclusive',
            'ray_support': '>=2 neighbor votes for one depth and zero opposing votes; joint requires agreement of both fixed methods',
            'wrong_rays': 'abs(final-old)>0.15m only for post-hoc grouped reporting; all32 evaluated',
            'native_depth_source': 'EXACT current.expected_camera_z_m in frozen floor report; its native validation maxabs difference '+str(frozen['final_native_validation']['max_depth_abs_error_m'])+'m'},
        'selection': selection, 'rays': results, 'summary': {'all': summarize(results), 'wrong_depth': summarize(wrong),
            'near_wrong_depth': summarize([r for r in wrong if r['signed_depth_difference_m'] < 0]),
            'far_wrong_depth': summarize([r for r in wrong if r['signed_depth_difference_m'] > 0])},
        'analytic_checks': checks, 'self_projection_max_error_px': max(projection_errors), 'source_sha256': PINS,
        'limitations': ['NCC/photometric support is not depth ground truth; occlusion, repetitive texture, motion, lighting and pose/calibration error can confound it.',
            'Images are separate evidence from LiDAR depth values, but TRAIN RGB informed the model; this is not a wholly independent held-out training-data experiment.',
            'Translation-only patches ignore projective shape change; the fixed-normal warp shares LiDAR normal uncertainty and does not model occlusion.',
            'No model/plane fit, pixel search, candidate-error filtering, extra ray selection or threshold retuning was performed.',
            'No measured-map neighborhood or occlusion query was performed; Euclid geometry audit remains separate.'],
        'elapsed_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu_started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with OUT.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(OUT), 'summary': report['summary'], 'elapsed_seconds': report['elapsed_seconds'], 'cpu_seconds': report['cpu_seconds']}, indent=2))


if __name__ == '__main__':
    main()

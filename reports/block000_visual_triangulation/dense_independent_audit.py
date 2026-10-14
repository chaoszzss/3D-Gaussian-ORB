"""Read-only audit of saved dense v2 observations. No matcher/producer imports.

Run from repository root with three-CPU affinity and single-thread BLAS.
The sole result is a new JSON; the producer and all its inputs stay untouched.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/garage_fidelity_depthv3/block_000'
SOURCE = ROOT / 'reports/block000_dense_stereo_v2'
PAIRS = ((0, 1), (1, 2), (0, 2))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    v = np.asarray(values).ravel()
    return dict(count=len(v), min=float(v.min()), median=float(np.median(v)),
                p95=float(np.quantile(v, .95)), max=float(v.max())) if len(v) else dict(count=0)


def cam(frame):
    # Construct each axis explicitly instead of using the producer camera helper.
    T = np.array(frame['transform_matrix'], dtype=np.float64)
    T[:, 1:3] *= -1
    W = np.linalg.inv(T)
    K = np.array([[frame['fl_x'], 0, frame['cx'] - .5],
                  [0, frame['fl_y'], frame['cy'] - .5], [0, 0, 1]], dtype=np.float64)
    return dict(T=T, W=W, K=K, C=T[:3, 3])


def projections(xyz, cameras):
    h = np.column_stack((xyz, np.ones(len(xyz))))
    uv, depths = [], []
    for c in cameras:
        q = h @ c['W'][:3].T
        p = q @ c['K'].T
        uv.append(p[:, :2] / p[:, 2:3])
        depths.append(q[:, 2])
    return np.stack(uv, 1), np.stack(depths, 1)


def angles(xyz, cameras):
    rays = [xyz - c['C'] for c in cameras]
    return np.column_stack([np.degrees(np.arctan2(
        np.linalg.norm(np.cross(rays[a], rays[b]), axis=1),
        np.einsum('ij,ij->i', rays[a], rays[b]))) for a, b in PAIRS])


def dlt(uv, cameras, normalized):
    # Independent per-point SVD, with optional pixel-coordinate camera matrix.
    result, gaps = [], []
    for observations in uv:
        rows = []
        for xy, c in zip(observations, cameras):
            q = np.r_[xy, 1.]
            if normalized:
                q = np.linalg.solve(c['K'], q)
                P = c['W'][:3]
            else:
                P = c['K'] @ c['W'][:3]
            rows.extend((q[0] * P[2] - q[2] * P[0], q[1] * P[2] - q[2] * P[1]))
        _, s, vh = np.linalg.svd(np.asarray(rows))
        result.append(vh[-1, :3] / vh[-1, 3])
        gaps.append(s[-2] / max(s[-1], 1e-15))
    return np.array(result), np.array(gaps)


def sample(array, uv, mode=cv2.INTER_LINEAR):
    # At most 89 saved observations per triplet; never calls a matcher.
    p = np.asarray(uv, dtype=np.float32)
    out = cv2.remap(array, p[:, 0].reshape(-1, 1), p[:, 1].reshape(-1, 1), mode,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape((len(p),) + array.shape[2:])


def patch_metrics(a, b, u, v):
    # Higher precision independent calculation of the same 25-pixel stencil.
    offsets = [(dx, dy) for dy in (-4, -2, 0, 2, 4) for dx in (-4, -2, 0, 2, 4)]
    x = np.stack([sample(a, u + q) for q in offsets], 1).astype(np.float64)
    y = np.stack([sample(b, v + q) for q in offsets], 1).astype(np.float64)
    sx, sy = x.std(1), y.std(1)
    x -= x.mean(1)[:, None]
    y -= y.mean(1)[:, None]
    return (x * y).mean(1) / np.maximum(sx * sy, 1e-8), sx, sy


def q_world(left, disparity, Q, R1, c):
    h = np.column_stack((left, disparity, np.ones(len(left)))) @ Q.T
    rect = h[:, :3] / h[:, 3:]
    # Row vectors: inverse rectification R1.T, followed by camera-to-world.
    original = rect @ R1
    return original @ c['T'][:3, :3].T + c['C']


def run(output):
    if output.exists():
        raise FileExistsError('Refuse to overwrite an audit result')
    cv2.setNumThreads(1)
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:3]))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 2800 * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (min(hard, cap) if hard != resource.RLIM_INFINITY else cap, hard))
    start = time.monotonic()
    producer = json.loads((SOURCE / 'report.json').read_text())
    plan = json.loads((SOURCE / 'plan.json').read_text())
    meta = json.loads((DATA / 'transforms.json').read_text())
    prep = json.loads((DATA / 'preparation.json').read_text())
    with np.load(SOURCE / 'tracks.npz', allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    n = len(a['xyz'])
    frames = plan['train_frame_table']
    checks = {}
    failed_tracks = {}
    def check(name, value):
        checks[name] = bool(value)
    def point_check(name, good, ids=None):
        bad = np.flatnonzero(~np.asarray(good, dtype=bool))
        if len(bad):
            failed_tracks.setdefault(name, []).extend((bad if ids is None else ids[bad]).tolist())

    before = {}
    expected = dict(producer['source_sha256'])
    expected.update(producer['train_asset_sha256'])
    expected.update({str(SOURCE / k): v for k, v in producer['outputs_sha256'].items()})
    expected[prep['calibration']] = prep['calibration_sha256']
    expected[prep['pose_csv']] = prep['pose_sha256']
    hashes = []
    for path, digest in expected.items():
        actual = sha(path)
        before[path] = actual
        hashes.append(dict(path=path, expected=digest, actual=actual, matches=actual == digest))
    for path in (SOURCE / 'report.json', ROOT / 'scripts/triangulate_training_tracks.py', Path(__file__)):
        before[str(path)] = sha(path)
    check('source_and_output_hashes_match', all(h['matches'] for h in hashes))
    check('plan_source_pins_match_report', plan['source_sha256'] == producer['source_sha256'])
    check('producer_script_hash_matches', sha(ROOT / 'runtime/dense_train_stereo_probe.py') == producer['script_sha256'])
    check('336_points', n == 336)
    check('finite_numeric_arrays', all(np.isfinite(v).all() for v in a.values()))
    check('expected_shapes', all(a[k].shape == s for k, s in {
        'xyz': (n, 3), 'xyz_first_two': (n, 3), 'uv': (n, 3, 2),
        'train_frame_indices': (n, 3), 'rgb': (n, 3), 'triplet_id': (n,)}.items()))
    check('no_normal_claims', not a['normal_valid'].any())
    sets = [set(meta[k]) for k in ('train_filenames', 'val_filenames', 'test_filenames')]
    check('split_names_disjoint', all(not (sets[i] & sets[j]) for i, j in PAIRS))
    check('all_frames_accounted_for', set(f['file_path'] for f in meta['frames']) == set.union(*sets))
    groups = [set(f['group_id'] for f in meta['frames'] if f['file_path'] in s) for s in sets]
    check('split_timestamp_groups_disjoint', all(not (groups[i] & groups[j]) for i, j in PAIRS))
    actual_train = [f for f in meta['frames'] if f['file_path'] in sets[0]]
    check('complete_train_table_matches_source', frames == actual_train)
    check('pixel_center_half_integer_documented', meta['pixel_center_convention'].startswith('half_integer'))
    check('no_input_distortion', all(not np.any(f.get(k, meta.get(k, 0))) for f in frames
                                   for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2', 'distortion_params')))
    parent = json.loads((ROOT / 'reports/block000_visual_triangulation/triplet_plan.json').read_text())
    intended = [row for cid in ('camera_2', 'camera_5')
                for j, row in enumerate(t for t in parent['triplets'] if t['camera_id'] == cid) if j in (1, 3, 5, 7)]
    check('preselected_pose_triplets_match', plan['plan'] == intended)
    check('triplet_ids_complete', set(a['triplet_id']) == set(range(len(plan['plan']))) == set(range(8)))
    used = sorted(set(a['train_frame_indices'].ravel().tolist()))
    expected_assets = {str(DATA / frames[i][key]) for i in used for key in ('file_path', 'mask_path')}
    check('asset_hash_inventory_exact', expected_assets == set(producer['train_asset_sha256']))
    prepared = {f['file_path']: f for f in prep['resolution_pilot']['frames']}
    check('native_image_and_mask_pins_match_preparation', all(
        before[str(DATA / frames[i]['file_path'])] == prepared[frames[i]['file_path']]['image_sha256'] and
        before[str(DATA / frames[i]['mask_path'])] == prepared[frames[i]['file_path']]['mask_sha256'] and
        prepared[frames[i]['file_path']]['split'] == 'train_filenames' for i in used))
    ply = np.loadtxt(SOURCE / 'points.ply', skiprows=10)
    check('ply_matches_npz_exactly', np.array_equal(ply[:, :3], a['xyz']) and np.array_equal(ply[:, 3:], a['rgb']))

    all_cameras = [cam(f) for f in frames]
    rows = []
    per_point = []
    aggregate = {key: [] for key in ('pair_dlt_delta_m', 'pre_third_px', 'stored_pair_pre_third_px',
        'final_reprojection_px', 'final_parallax_deg', 'camera_depth_m', 'three_dlt_delta_m',
        'Q_delta_m', 'Q_relative_delta', 'inverse_grid_error_px', 'remapped_uv_error_px',
        'zncc', 'patch_std', 'mask_value', 'singular_gap')}
    for tid, triplet in enumerate(plan['plan']):
        idx = np.flatnonzero(a['triplet_id'] == tid)
        fi = triplet['train_indices']
        fs = [frames[i] for i in fi]
        cams = [all_cameras[i] for i in fi]
        uv = a['uv'][idx].astype(np.float64)
        xyz = a['xyz'][idx]
        check(f'triplet_{tid}_indices_match', np.array_equal(a['train_frame_indices'][idx], np.tile(fi, (len(idx), 1))))
        check(f'triplet_{tid}_same_camera_time_order', len(set(f['camera_id'] for f in fs)) == 1 and
              fs[0]['timestamp_ns'] < fs[1]['timestamp_ns'] < fs[2]['timestamp_ns'])
        baseline = np.array([np.linalg.norm(cams[i]['C'] - cams[j]['C']) for i, j in PAIRS])
        check(f'triplet_{tid}_baseline', (baseline >= .3).all() and (baseline <= 2).all() and
              np.allclose(baseline, triplet['baseline_m'], atol=1e-9, rtol=0))
        pair_xyz, _ = dlt(uv[:, :2], cams[:2], normalized=False)
        pair_delta = np.linalg.norm(pair_xyz - a['xyz_first_two'][idx], axis=1)
        independent_pair_uv, independent_pair_depth = projections(pair_xyz, cams)
        independent_pair_err = np.linalg.norm(independent_pair_uv - uv, axis=2)
        stored_pair_uv, stored_pair_depth = projections(a['xyz_first_two'][idx].astype(float), cams)
        stored_pair_err = np.linalg.norm(stored_pair_uv - uv, axis=2)
        final_uv, final_depth = projections(xyz, cams)
        final_err = np.linalg.norm(final_uv - uv, axis=2)
        final_angles = angles(xyz, cams)
        final_dlt, singular_gap = dlt(uv, cams, normalized=True)
        final_delta = np.linalg.norm(final_dlt - xyz, axis=1)
        point_check('first_two_DLT_float32_consistency', pair_delta <= 5e-5, idx)
        point_check('independent_pre_refine_all3_error_1p5', (independent_pair_err <= 1.5).all(1), idx)
        point_check('stored_pair_pre_refine_all3_error_1p5', (stored_pair_err <= 1.5).all(1), idx)
        point_check('pair_depth_bounds', ((independent_pair_depth >= .2) & (independent_pair_depth <= 50)).all(1), idx)
        point_check('pair_geometry_all_pair_parallax', (angles(pair_xyz, cams) >= 1).all(1), idx)
        point_check('final_reprojection_1p5', (final_err <= 1.5).all(1), idx)
        point_check('final_depth_bounds', ((final_depth >= .2) & (final_depth <= 50)).all(1), idx)
        point_check('final_geometry_all_pair_parallax_1deg', (final_angles >= 1).all(1), idx)
        point_check('final_3view_DLT_consistency', final_delta <= 1e-7, idx)
        point_check('three_view_singular_gap_10', singular_gap >= 10, idx)
        check(f'triplet_{tid}_saved_metric_consistency',
              np.allclose(stored_pair_err, a['first_two_solution_reprojection_px'][idx], atol=1e-8, rtol=0) and
              np.allclose(final_err, a['reprojection_px'][idx], atol=1e-8, rtol=0) and
              np.allclose(final_depth, a['depths_m'][idx], atol=1e-8, rtol=0) and
              np.allclose(final_angles, a['geometric_parallax_deg'][idx], atol=1e-8, rtol=0))
        images, masks, eroded = [], [], []
        for f in fs:
            assert f['file_path'] in sets[0] and f['file_path'] not in sets[1] | sets[2]
            im = cv2.imread(str(DATA / f['file_path']), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(DATA / f['mask_path']), cv2.IMREAD_GRAYSCALE)
            check(f'frame_{f["file_path"]}_native_shape', im.shape == (1080, 1280, 3) and mask.shape == (1080, 1280))
            images.append(im)
            masks.append(mask)
            eroded.append(cv2.erode(mask, np.ones((31, 31), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0))
        raw_values = np.column_stack([sample(m, uv[:, j]) for j, m in enumerate(masks)])
        eroded_values = np.column_stack([sample(m, uv[:, j]) for j, m in enumerate(eroded)])
        point_check('all_observations_original_mask', (raw_values > 254).all(1), idx)
        point_check('all_observations_eroded31_mask', (eroded_values > 254).all(1), idx)
        point_check('third_initial_prediction_eroded_mask', sample(eroded[2], stored_pair_uv[:, 2]) > 254, idx)
        colors = np.median(np.stack([sample(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), uv[:, j])
                                    for j, im in enumerate(images)], 1), axis=1).astype(np.uint8)
        point_check('TRAIN_RGB_three_view_median', (colors == a['rgb'][idx]).all(1), idx)

        # Derive A->B transform independently using full homogeneous transforms.
        rel = cams[1]['W'] @ cams[0]['T']
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cams[0]['K'], None, cams[1]['K'], None,
            (1280, 1080), rel[:3, :3], rel[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        check(f'triplet_{tid}_horizontal_rectification', abs(P2[0, 3]) > abs(P2[1, 3]))
        ur = cv2.undistortPoints(uv[:, 0].reshape(-1, 1, 2), cams[0]['K'], None, R=R1, P=P1[:, :3]).reshape(-1, 2)
        vr = cv2.undistortPoints(uv[:, 1].reshape(-1, 1, 2), cams[1]['K'], None, R=R2, P=P2[:, :3]).reshape(-1, 2)
        # Infer original raster samples, then verify by reproducing the inverse maps.
        grid = 16 + 4 * np.rint((ur - 16) / 4)
        disparity = np.rint((ur[:, 0] - vr[:, 0]) * 16) / 16
        right = grid.copy()
        right[:, 0] -= disparity
        grid_error = np.maximum(np.linalg.norm(ur - grid, axis=1), np.linalg.norm(vr - right, axis=1))
        maps = [cv2.initUndistortRectifyMap(c['K'], None, rr, pp[:, :3], (1280, 1080), cv2.CV_32FC1)
                for c, rr, pp in zip(cams[:2], (R1, R2), (P1, P2))]
        back_uv = np.stack([np.column_stack([sample(m, coords) for m in mm]) for mm, coords in zip(maps, (grid, right))], 1)
        back_error = np.linalg.norm(back_uv - uv[:, :2], axis=2)
        qxyz = q_world(grid, disparity, Q, R1, cams[0])
        qdelta = np.linalg.norm(qxyz - pair_xyz, axis=1)
        qrelative = qdelta / np.linalg.norm(pair_xyz - cams[0]['C'], axis=1)
        qproject, qdepth = projections(qxyz, cams[:2])
        qerror = np.linalg.norm(qproject - uv[:, :2], axis=2)
        # Independent analytic horizontal disparity depth check of Q convention/sign.
        tx = P2[0, 3] / P2[0, 0]
        zrect = -P1[0, 0] * tx / (disparity - (P1[0, 2] - P2[0, 2]))
        rectxyz = np.column_stack(((grid[:, 0] - P1[0, 2]) * zrect / P1[0, 0],
                                  (grid[:, 1] - P1[1, 2]) * zrect / P1[1, 1], zrect))
        analytic_xyz = (rectxyz @ R1) @ cams[0]['T'][:3, :3].T + cams[0]['C']
        md = 0 if P2[0, 3] < 0 else -256
        point_check('rectified_grid_and_disparity_recovered', grid_error <= .005, idx)
        point_check('inverse_map_saved_observations_consistency', (back_error <= .001).all(1), idx)
        point_check('inferred_disparity_range', (disparity > md) & (disparity < md + 255), idx)
        point_check('Q_backprojection_saved_observation_consistency', (qerror <= .001).all(1), idx)
        point_check('Q_backprojection_xyz_numerical_consistency', (qdelta <= .003) & (qrelative <= .0005), idx)
        point_check('Q_analytic_horizontal_formula_consistency', np.linalg.norm(qxyz - analytic_xyz, axis=1) <= 1e-8, idx)
        point_check('Q_positive_camera_depth', (qdepth > 0).all(1), idx)
        rect = [cv2.remap(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), *mm, cv2.INTER_LINEAR)
                for im, mm in zip(images, maps)]
        rect_masks = [cv2.remap(m, *mm, cv2.INTER_NEAREST) for m, mm in zip(eroded, maps)]
        rect_mask_values = np.column_stack((sample(rect_masks[0], grid), sample(rect_masks[1], right)))
        point_check('rectified_observations_eroded_mask', (rect_mask_values > 254).all(1), idx)
        zncc, s1, s2 = patch_metrics(rect[0], rect[1], grid, right)
        point_check('recomputed_stereo_zncc_texture', (zncc >= .85) & (s1 >= 4) & (s2 >= 4), idx)
        row = dict(triplet_id=tid, camera=triplet['camera_id'], tracks=len(idx), source_train_indices=fi,
            source_train_files=[f['file_path'] for f in fs], baseline_m=baseline.tolist(),
            pair_dlt_float32_difference_m=stats(pair_delta), first_two_pre_refine_third_error_px=stats(independent_pair_err[:, 2]),
            saved_pair_recomputed_pre_refine_third_error_px=stats(stored_pair_err[:, 2]),
            final_error_px=stats(final_err), final_xyz_parallax_deg=stats(final_angles), final_depth_m=stats(final_depth),
            three_view_dlt_difference_m=stats(final_delta), singular_gap=stats(singular_gap),
            original_mask_min=int(raw_values.min()), eroded_mask_min=int(eroded_values.min()),
            rectified_mask_min=int(rect_mask_values.min()),
            rectification=dict(R1=R1.tolist(), R2=R2.tolist(), P1=P1.tolist(), P2=P2.tolist(), Q=Q.tolist(),
                inverse_grid_error_px=stats(grid_error), inverse_map_error_px=stats(back_error),
                Q_world_difference_m=stats(qdelta), Q_relative_difference=stats(qrelative),
                Q_reprojection_error_px=stats(qerror), disparity_px=stats(disparity),
                zncc=stats(zncc), left_patch_std=stats(s1), right_patch_std=stats(s2)))
        rows.append(row)
        values = (pair_delta, independent_pair_err[:, 2], stored_pair_err[:, 2], final_err, final_angles,
                  final_depth, final_delta, qdelta, qrelative, grid_error, back_error, zncc,
                  np.column_stack((s1, s2)), eroded_values, singular_gap)
        for key, value in zip(aggregate, values):
            aggregate[key].extend(np.asarray(value).ravel().tolist())
        for j, track_id in enumerate(idx):
            per_point.append(dict(track_id=int(track_id), triplet_id=tid,
                pre_refine_third_error_px=float(independent_pair_err[j, 2]), final_max_error_px=float(final_err[j].max()),
                final_min_geometric_angle_deg=float(final_angles[j].min()), min_eroded_mask_value=int(eroded_values[j].min()),
                Q_world_difference_m=float(qdelta[j]), stereo_zncc=float(zncc[j]),
                inferred_rectified_left_uv=grid[j].tolist(), inferred_disparity_px=float(disparity[j])))
        print(json.dumps(dict(triplet_id=tid, tracks=len(idx), third_max=float(independent_pair_err[:, 2].max()),
                              min_angle=float(final_angles.min()), Q_max_m=float(qdelta.max()))), flush=True)

    centers = np.array([c['C'] for c in all_cameras])
    distances = np.linalg.norm(a['xyz'][:, None, :2] - centers[None, :, :2], axis=2)
    nearest = distances.argmin(1)
    xy = distances[np.arange(n), nearest]
    dz = a['xyz'][:, 2] - centers[nearest, 2]
    band_z = (dz >= -3) & (dz <= -.4)
    near3 = band_z & (xy <= 3)
    near8 = band_z & (xy <= 8)
    check('floor_band_diagnostic_recomputed', np.array_equal(near3, a['geometric_floor_band']) and
          np.array_equal(nearest, a['nearest_train_center_index']) and
          np.allclose(xy, a['nearest_train_center_xy_distance_m'], atol=1e-10, rtol=0) and
          np.allclose(dz, a['relative_world_z_m'], atol=1e-10, rtol=0))
    check('floor_band_counts_match_report', int(near3.sum()) == producer['floor_band_points'] == 1 and
          int(near8.sum()) == producer['band_within8m'] == 28)
    saved_only = {}
    for key, cap in (('third_fb_px', .75), ('third_photometric_l1', 20), ('third_cycle_px', .75)):
        saved_only[key] = dict(values=stats(a[key]), numerical_threshold=cap,
            stored_scalar_violations=int(((a[key] < 0) | (a[key] > cap)).sum()),
            independently_replayed=False)
    check('saved_only_LK_scalar_bounds', all(r['stored_scalar_violations'] == 0 for r in saved_only.values()))
    check('no_geometry_mask_color_violations', not failed_tracks)
    unchanged = {p: sha(p) == digest for p, digest in before.items()}
    check('all_read_inputs_unchanged_after_audit', all(unchanged.values()))
    out = dict(status='PASS_SAVED_GEOMETRY_MASK_TRAIN_SOURCE_WITH_MATCHING_PROVENANCE_LIMITS' if all(checks.values()) else 'FAIL_INDEPENDENT_AUDIT',
        tracks=n, accepted_for_integration=False, semantic_floor_claim=False,
        checks=checks, failed_track_ids_by_check=failed_tracks, statistics={k: stats(v) for k, v in aggregate.items()},
        geometric_height_band=dict(definition='nearest TRAIN camera center by XY; relative world z in [-3,-0.4] m; diagnostic, not semantic floor',
            within3m=int(near3.sum()), within8m=int(near8.sum()), track_ids_within3m=np.flatnonzero(near3).tolist(),
            track_ids_within8m=np.flatnonzero(near8).tolist()),
        source_hash_audit=hashes, current_imported_helper_sha256=before[str(ROOT / 'scripts/triangulate_training_tracks.py')],
        all_audited_input_sha256=before, all_audited_inputs_unchanged=unchanged,
        input_array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in a.items()},
        train_only=dict(train_frames=len(frames), used_train_frames=len(used), dev_frames=len(sets[1]), test_frames=len(sets[2]),
            dev_images_read=0, test_images_read=0, timestamp_groups_disjoint=checks['split_timestamp_groups_disjoint']),
        numerical_comparison_tolerances=dict(pair_float32_xyz_m=5e-5, final_float64_DLT_xyz_m=1e-7,
            inverse_grid_px=.005, inverse_map_px=.001, Q_reprojection_px=.001, Q_xyz_m=.003, Q_xyz_relative=.0005,
            rationale='Comparison tolerances account for float32 saved pair XYZ and sampled float32 inverse maps. Actual acceptance gates remain error<=1.5 px, all-pair final XYZ angle>=1 deg and camera z in [0.2,50] m with no relaxation.'),
        rectification_method='Independent metadata C2W inverse; regenerate stereoRectify; invert saved uv; infer 4px left raster and 1/16px disparity; verify inverse-map recovery, analytic horizontal depth, Q->R1.T->camera-to-world, and original-camera projections. Recovered raster/disparity are inferred, not saved matcher outputs.',
        third_view_validation='Recompute first-two-only pixel-DLT without using third uv, then project into third image and compare saved mean third observation before normalized three-view DLT. Separately project stored float32 pair XYZ. The third correspondence itself remains initialization-biased.',
        independently_recomputed=['TRAIN membership and camera-time groups', 'all producer source/output/asset SHA256 pins plus preparation calibration/pose/image/mask pins',
            'first-two DLT XYZ and third pre-refinement residual', 'three-view DLT XYZ and singular gap', 'all-view depths/reprojection and all-pair final XYZ angles',
            'rectification inverse mapping, Q backprojection and analytic disparity depth', 'all saved observations original/eroded masks and predicted-third eroded mask',
            'rectified image texture/ZNCC at recovered correspondences', 'TRAIN-only sampled RGB median', 'PLY/NPZ equality', 'geometric height-band counts'],
        saved_only_scalar_checks=saved_only,
        not_independently_replayable=['SGBM left-right consistency: original left/right disparity maps absent',
            'LK forward/backward status and displacement: backward endpoints and statuses absent',
            'Third pair-cycle: separate camera0->2 and camera1->2 endpoints absent; only mean uv and scalar cycle saved',
            'Per-branch third mask and L1: individual endpoints and LK pyramids absent',
            'Historical producer imported helper identity: producer pins its own script but not the imported helper; current hash is recorded only'],
        limitations=['Third-view LK initializes both forward branches with the first-two geometry prediction and initializes backward flow with the original source observation. The pre-refinement geometric test is independently recomputed here, but the correspondence acquisition is not statistically independent of pair geometry.',
            'Dense SGBM pixels, overlapping patches and neighboring optical-flow tracks are correlated; 336 points are not 336 independent surface measurements.',
            'Stored scalar gate compliance cannot certify underlying matches when intermediate measurements are absent; no new matching was authorized or run.',
            'Passing saved-data consistency does not validate inherited calibration/poses, resolve repeated texture/specular ambiguity, establish semantic ground, or prove reconstruction quality.',
            'No integration or modification of model, dataset, LiDAR reference, v3 depth teacher, old sparse/dense artifacts, or training/evaluation sources.'],
        triplets=rows, per_track=sorted(per_point, key=lambda x: x['track_id']),
        resources=dict(cpu_affinity=sorted(os.sched_getaffinity(0)), opencv_threads=cv2.getNumThreads(), blas_threads=1,
            virtual_memory_cap_mib=2800, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            elapsed_s=time.monotonic()-start, gpu_used=False, new_matching=False),
        reproduce_command="CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 4-6 nice -n 10 .venv-train/bin/python reports/block000_visual_triangulation/dense_independent_audit.py --output reports/block000_dense_stereo_v2_independent_repeat.json")
    with output.open('x') as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=out['status'], failed_checks=[k for k, v in checks.items() if not v],
                         failed_track_ids=failed_tracks, resources=out['resources'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/block000_dense_stereo_v2_independent.json')
    args = parser.parse_args()
    run(args.output.resolve())

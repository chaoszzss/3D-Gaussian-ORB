#!/usr/bin/env python3
"""Append strictly TRAIN-confirmed measured surfaces to one original block PLY.

Independent single-preparation experiment; the prior joint floor seeder is unchanged.
Normals-valid actual reference rows only, >=3 distinct TRAIN groups, positive Z,
ego mask and known v3 depth within 10 cm. Equal group-weight stored-sRGB color.
No interpolation, inferred planes, silent count cap, GPU or adoption claim.
"""
import sys
sys.dont_write_bytecode = True
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import re
import resource
import shutil
import time
import zipfile
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from assemble_core_gaussians import (
    ROOT, AssemblyError, InputGuard, inspect_ply, local_path, resource_preflight,
    sha256, write_json,
)
from seed_measured_floor_gaussians import train_groups, stable_voxel_rows

MAX_REFERENCE = MAX_SEEDS = 400_000
MEMORY_BUDGET = 2048 * 1024**2  # conservative preflight; RSS must stay <3 GiB
RSS_LIMIT = 3 * 1024**3
CHUNK_BYTES = 8 * 1024**2
POLICY = dict(minimum_unique_train_groups=3, minimum_camera_z_m=0.,
              depth_agreement_m=.10, seed_voxel_m=.02, minimum_source_center_distance_m=.02,
              tangent_sigma_m=.02, normal_thickness_m=.001, opacity=.9, maximum_seeds=MAX_SEEDS)
CLASS_NAMES = {1: 'floor_proxy', 2: 'ceiling_proxy', 3: 'mixed_horizontal',
               4: 'horizontal_unresolved', 5: 'wall_normal', 6: 'sloped_normal'}


def valid_reference_rows(xyz, normals, valid):
    xyz, normals, valid = map(np.asarray, (xyz, normals, valid))
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape
            or valid.shape != (len(xyz),) or valid.dtype != np.bool_):
        raise AssemblyError('Reference XYZ/normal/valid shapes or normal_valid dtype differ')
    length = np.linalg.norm(normals.astype(np.float64), axis=1)
    good = valid & np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1) & (length > 1e-12)
    unit = np.zeros(normals.shape, dtype=np.float64)
    unit[good] = normals[good] / length[good, None]
    return np.flatnonzero(good), unit


def normal_quaternions(normals):
    """Shortest deterministic +local-Z -> signed measured normal, wxyz order."""
    n = np.array(normals, dtype=np.float64, copy=True)
    if n.ndim != 2 or n.shape[1] != 3 or not np.isfinite(n).all():
        raise AssemblyError('Invalid measured normals')
    length = np.linalg.norm(n, axis=1)
    if np.any(length <= 1e-12):
        raise AssemblyError('Zero measured normal')
    n /= length[:, None]
    radial = np.hypot(n[:, 0], n[:, 1])
    angle = .5 * np.arctan2(radial, n[:, 2])
    q = np.zeros((len(n), 4), dtype=np.float64)
    q[:, 0] = np.cos(angle)
    nonpolar = radial > 0
    q[nonpolar, 1] = -n[nonpolar, 1] / radial[nonpolar] * np.sin(angle[nonpolar])
    q[nonpolar, 2] = n[nonpolar, 0] / radial[nonpolar] * np.sin(angle[nonpolar])
    q[(~nonpolar) & (n[:, 2] < 0)] = [0, 1, 0, 0]
    return q.astype('<f4'), n.astype('<f4')


def eligible_observations(xyz, frame, depth, depth_mask, ego_mask):
    """All surface orientations, no height gate; nearest integer measured pixel."""
    h, w = int(frame['h']), int(frame['w'])
    pose = np.asarray(frame['transform_matrix'], dtype=np.float64)
    if (min(h, w) <= 0 or h*w > 4_194_304 or pose.shape != (4, 4)
            or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
        raise AssemblyError('Invalid or oversized TRAIN camera')
    if any(np.asarray(a).shape != (h, w) for a in (depth, depth_mask, ego_mask)):
        raise AssemblyError('Depth and grayscale masks must match TRAIN H/W')
    if not all(np.isfinite(frame[k]) for k in ('fl_x', 'fl_y', 'cx', 'cy')) or min(frame['fl_x'], frame['fl_y']) <= 0:
        raise AssemblyError('Invalid TRAIN intrinsics')
    p = (np.asarray(xyz) - pose[:3, 3]) @ (pose[:3, :3] @ np.diag([1., -1., -1.]))
    with np.errstate(divide='ignore', invalid='ignore'):
        u = frame['fl_x']*p[:, 0]/p[:, 2] + frame['cx'] - .5
        v = frame['fl_y']*p[:, 1]/p[:, 2] + frame['cy'] - .5
    inside = ((p[:, 2] > 0) & np.isfinite(u) & np.isfinite(v)
              & (u >= -.5) & (u < w-.5) & (v >= -.5) & (v < h-.5))
    ids = np.flatnonzero(inside)
    ui, vi = np.floor(u[ids]+.5).astype(int), np.floor(v[ids]+.5).astype(int)
    target = depth[vi, ui]
    keep = ((depth_mask[vi, ui] > 0) & (ego_mask[vi, ui] > 0) & np.isfinite(target)
            & (target > 0) & (np.abs(p[ids, 2]-target) <= POLICY['depth_agreement_m']))
    return ids[keep], ui[keep], vi[keep]


def enforce_seed_limit(count, reference_count):
    limit = min(MAX_SEEDS, reference_count)
    if count > limit:
        raise AssemblyError(f'{count} seeds exceeds {limit}; refusing to truncate')


def seed_rows(names, xyz, normals, rgb):
    allowed = {'x', 'y', 'z', 'nx', 'ny', 'nz', 'opacity',
               *[f'scale_{i}' for i in range(3)], *[f'rot_{i}' for i in range(4)],
               *[f'f_dc_{i}' for i in range(3)], *[f'f_rest_{i}' for i in range(45)]}
    if not set(names) <= allowed:
        raise AssemblyError('Unsupported seed PLY fields')
    if rgb.shape != np.asarray(xyz).shape or np.any(~np.isfinite(rgb)) or np.any(rgb < 0) or np.any(rgb > 1):
        raise AssemblyError('Seed RGB must be measured sRGB bounded in [0,1]')
    rows = np.zeros((len(xyz), len(names)), dtype='<f4')
    q, n = normal_quaternions(normals)
    for j, key in enumerate(('x', 'y', 'z')):
        rows[:, names.index(key)] = xyz[:, j]
    for j, key in enumerate(('nx', 'ny', 'nz')):
        if key in names:
            rows[:, names.index(key)] = n[:, j]
    for j in range(4):
        rows[:, names.index(f'rot_{j}')] = q[:, j]
    for j in range(3):
        rows[:, names.index(f'f_dc_{j}')] = (rgb[:, j]-.5)/.28209479177387814
        rows[:, names.index(f'scale_{j}')] = np.log(.02 if j < 2 else .001)
    rows[:, names.index('opacity')] = np.log(.9/.1)
    return rows, q, n


def quantiles(values):
    values = np.asarray(values)
    return {'count': len(values), **({k: float(v) for k, v in zip(
        ('min', 'p05', 'p50', 'p95', 'max'), np.quantile(values, [0, .05, .5, .95, 1]))} if len(values) else {})}


def orientation_classes(unit):
    z = np.abs(unit[:, 2])
    return np.where(z >= np.cos(np.deg2rad(30)), 4, np.where(z <= .5, 5, 6)).astype(np.uint8)


def geometry_classes(unit, below_groups, above_groups):
    c = orientation_classes(unit)
    horizontal = c == 4
    c[horizontal & (below_groups >= 3) & (above_groups == 0)] = 1
    c[horizontal & (above_groups >= 3) & (below_groups == 0)] = 2
    c[horizontal & (below_groups > 0) & (above_groups > 0)] = 3
    return c


def class_summary(classes, distance):
    return {name: {'count': int((classes == code).sum()),
                   'original_center_distance_m': quantiles(distance[classes == code])}
            for code, name in CLASS_NAMES.items()}


def validate_single_preparation(meta, prep, guard, project):
    """Single-resolution schema has per-frame asset proofs, not joint generated_hashes."""
    if (meta.get('orientation_override', 'none') != 'none' or prep.get('world_frame') != 'camera_init'
            or prep.get('units') not in ('m', 'metres')):
        raise AssemblyError('Require original shared camera_init coordinates in meters')
    pilot, teacher = prep['resolution_pilot'], prep['teacher_provenance']
    policy = teacher['policy']
    if (policy.get('version') != 'surfel_depth_v3' or policy.get('ground_support') is not True
            or policy.get('world_up') != [0., 0., 1.]
            or not 0 <= policy.get('foreground_occlusion_tolerance_m', 1) <= .1):
        raise AssemblyError('Require pinned v3 known-foreground teacher with world Z up')
    if (pilot.get('world_recentered') is not False or pilot.get('physical_optics_unchanged') is not True
            or pilot.get('exact_source_poses_splits_timestamps') is not True
            or pilot['pixel_center']['offset'] != .5):
        raise AssemblyError('Single-preparation coordinate/pixel convention differs')
    source = local_path(project, Path(pilot['source_dataset'])/'transforms.json')
    source_meta = guard.json(source)
    if guard.files[source]['sha256'] != pilot['source_hashes']['transforms.json']:
        raise AssemblyError('Original camera metadata SHA mismatch')
    for split in ('train_filenames', 'val_filenames', 'test_filenames'):
        if meta[split] != source_meta[split]:
            raise AssemblyError('Single-preparation original split mismatch')
    by_name = {f['file_path']: f for f in source_meta['frames']}
    if len(by_name) != len(source_meta['frames']) or set(by_name) != {f['file_path'] for f in meta['frames']}:
        raise AssemblyError('Original camera frame set mismatch')
    for f in meta['frames']:
        original = by_name[f['file_path']]
        for key in ('transform_matrix', 'group_id', 'timestamp_ns', 'camera_id', 'source_file_path'):
            if f[key] != original[key]:
                raise AssemblyError('Original pose/group/timestamp mismatch')
        sx, sy = f['w']/original['w'], f['h']/original['h']
        if not np.allclose([f['fl_x'], f['fl_y'], f['cx'], f['cy']],
                           [original['fl_x']*sx, original['fl_y']*sy, original['cx']*sx, original['cy']*sy],
                           atol=1e-8, rtol=0):
            raise AssemblyError('Original optical intrinsics mismatch')
    assets = {f['file_path']: f for f in pilot['frames']}
    teachers = {f['file_path']: f for f in teacher['frames']}
    if len(assets) != len(pilot['frames']) or len(teachers) != len(teacher['frames']):
        raise AssemblyError('Duplicate per-frame preparation proof')
    return assets, teachers


def checked_image(path, size, mode):
    with Image.open(path) as im:
        if im.size != size or im.mode != mode:
            raise AssemblyError('TRAIN image/mask format or shape mismatch')
        return np.array(im)


def seed_measured_surfaces(source_ply, data, output, *, source_audit, threads=3, project=ROOT):
    with threadpool_limits(limits=1):
        return _seed_measured_surfaces(source_ply, data, output, source_audit=source_audit,
                                      threads=threads, project=project)


def _seed_measured_surfaces(source_ply, data, output, *, source_audit, threads, project):
    start = time.monotonic()
    project = Path(project).resolve()
    if project != ROOT:
        local_path(ROOT, project)
    if type(threads) is not int or not 1 <= threads <= 3:
        raise AssemblyError('CPU thread budget must be 1..3')
    source, base, destination = (local_path(project, p) for p in (source_ply, data, output))
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError('Output must be a NEW directory with an existing parent')
    with source.open('rb') as f:
        spec = inspect_ply(f)
    if spec['count'] > 1_000_000 or source.stat().st_size != len(spec['header'])+spec['count']*spec['stride']:
        raise AssemblyError('Source count/size outside bounded single-block contract')
    preflight = resource_preflight(destination,
        disk_bytes=source.stat().st_size+MAX_SEEDS*spec['stride']+128*1024**2, memory_bytes=MEMORY_BUDGET)
    guard = InputGuard(project)
    for path in (Path(__file__), ROOT/'scripts/seed_measured_floor_gaussians.py'):
        if project == ROOT:
            guard.snapshot(path)
    audit = guard.json(local_path(project, source_audit))
    guard.snapshot(source)
    meta = guard.json(base/'transforms.json')
    prep = guard.json(base/'preparation.json')
    matched = [s for s in audit['sources'] if local_path(project, s['path']) == source]
    if (len(matched) != 1 or matched[0]['source_sha256'] != guard.files[source]['sha256']
            or matched[0]['exported_gaussians'] != spec['count'] or matched[0]['id'] != prep['block_id']):
        raise AssemblyError('Original trained source audit SHA/count/block mismatch')
    groups = train_groups(meta)
    frame_count = sum(len(frames) for _, frames in groups)
    if frame_count > 4096:
        raise AssemblyError('TRAIN evidence exceeds bounded frame budget')
    assets, teachers = validate_single_preparation(meta, prep, guard, project)
    teacher = prep['teacher_provenance']
    refpath = base/'lidar_reference.npz'
    guard.snapshot(refpath)
    if guard.files[refpath]['sha256'] != teacher['reference_sha256']:
        raise AssemblyError('Reference SHA differs from v3 teacher')
    with zipfile.ZipFile(refpath) as archive:
        if sum(v.file_size for v in archive.infolist()) > 64*1024**2:
            raise AssemblyError('Reference exceeds decompressed memory budget')
    with np.load(refpath, allow_pickle=False) as archive:
        reference = {k: archive[k] for k in ('xyz', 'normals', 'normal_valid')}
    xyz0 = reference['xyz']
    if len(xyz0) > MAX_REFERENCE or xyz0.dtype.kind != 'f' or xyz0.dtype.itemsize != 4:
        raise AssemblyError('Require at most 400k float32 actual reference XYZ')
    if len(xyz0) != teacher['reference_points']:
        raise AssemblyError('Teacher reference count mismatch')
    refids, unit = valid_reference_rows(xyz0, reference['normals'], reference['normal_valid'])
    initial_valid = len(refids)
    mm = np.memmap(source, mode='r', dtype='<f4', offset=len(spec['header']),
                   shape=(spec['count'], len(spec['names'])))
    centers = np.asarray(mm[:, [spec['names'].index(k) for k in ('x', 'y', 'z')]], dtype=np.float64)
    del mm
    if not len(centers) or not np.isfinite(centers).all():
        raise AssemblyError('Missing or nonfinite original centers')
    tree = cKDTree(centers)
    distance, nearest = tree.query(xyz0[refids].astype(np.float64), workers=threads)
    baseline_distances = {'all_valid_reference': quantiles(distance),
                          'by_normal_orientation': class_summary(orientation_classes(unit[refids]), distance)}
    gap = distance >= POLICY['minimum_source_center_distance_m']
    refids, distance, nearest = refids[gap], distance[gap], nearest[gap]
    del tree, centers
    xyz = xyz0[refids].astype(np.float64)
    count = len(refids)
    estimated_peak = (spec['count']*64 + len(xyz0)*80
        + count*(128+(frame_count+7)//8+(len(groups)+7)//8) + MAX_SEEDS*spec['stride']*3 + 256*1024**2)
    if estimated_peak > MEMORY_BUDGET:
        raise AssemblyError('Evidence arrays exceed bounded peak memory estimate')
    support_groups = np.zeros(count, dtype='<u2')
    support_frames = np.zeros(count, dtype='<u2')
    below_groups = np.zeros(count, dtype='<u2')
    above_groups = np.zeros(count, dtype='<u2')
    group_rgb_sum = np.zeros((count, 3), dtype=np.float64)
    frame_bits = np.zeros((count, (frame_count+7)//8), np.uint8)
    group_bits = np.zeros((count, (len(groups)+7)//8), np.uint8)
    frame_table, frame_index = [], 0
    print(json.dumps({'valid_reference_rows': initial_valid, 'uncovered_2cm': count,
                      'train_groups': len(groups), 'train_frames': frame_count}), flush=True)
    for gi, (group, frames) in enumerate(groups):
        this_rgb = np.zeros((count, 3), dtype=np.float64)
        this_frames = np.zeros(count, dtype=np.uint16)
        this_below, this_above = np.zeros(count, bool), np.zeros(count, bool)
        for frame in frames:
            if not 0 < int(frame['h'])*int(frame['w']) <= 4_194_304:
                raise AssemblyError('TRAIN resolution exceeds bounded image memory')
            proof, asset = teachers[frame['file_path']], assets[frame['file_path']]
            if asset['split'] != 'train_filenames':
                raise AssemblyError('TRAIN asset split proof mismatch')
            paths = {k: local_path(project, base/frame[k]) for k in
                     ('file_path', 'mask_path', 'lidar_depth_path', 'lidar_depth_mask_path')}
            expected = {'file_path': asset['image_sha256'], 'mask_path': asset['mask_sha256'],
                        'lidar_depth_path': proof['depth_sha256'], 'lidar_depth_mask_path': proof['support_sha256']}
            for key, path in paths.items():
                guard.snapshot(path)
                if guard.files[path]['sha256'] != expected[key]:
                    raise AssemblyError('TRAIN asset SHA mismatch: '+str(path))
            depth = np.load(paths['lidar_depth_path'], allow_pickle=False, mmap_mode='r')
            size = (frame['w'], frame['h'])
            known = checked_image(paths['lidar_depth_mask_path'], size, 'L')
            ego = checked_image(paths['mask_path'], size, 'L')
            ids, ui, vi = eligible_observations(xyz, frame, depth, known, ego)
            rgb = checked_image(paths['file_path'], size, 'RGB')[vi, ui].astype(np.float64)
            this_rgb[ids] += rgb
            this_frames[ids] += 1
            support_frames[ids] += 1
            dz = frame['transform_matrix'][2][3]-xyz[ids, 2]
            this_below[ids[dz >= .3]] = True
            this_above[ids[dz <= -.3]] = True
            frame_bits[ids, frame_index//8] |= np.uint8(1 << (frame_index % 8))
            frame_table.append({'index': frame_index, 'group_index': gi, 'group_id': group,
                'file_path': frame['file_path'], 'camera_id': frame['camera_id'],
                'eligible_candidate_observations': len(ids), 'input_sha256': expected})
            frame_index += 1
        good = this_frames > 0
        support_groups += good
        below_groups += this_below
        above_groups += this_above
        group_rgb_sum[good] += this_rgb[good]/this_frames[good, None]
        group_bits[good, gi//8] |= np.uint8(1 << (gi % 8))
        if (gi+1) % 20 == 0 or gi+1 == len(groups):
            print(json.dumps({'train_groups_done': gi+1,
                              'supported_at_least_3': int((support_groups >= 3).sum())}), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= RSS_LIMIT:
            raise AssemblyError('CPU RSS reached 3 GiB bound; refusing publication')
    eligible = np.flatnonzero(support_groups >= 3)
    chosen = eligible[stable_voxel_rows(xyz[eligible], refids[eligible])]
    enforce_seed_limit(len(chosen), len(xyz0))
    chosen_ref = refids[chosen]
    seed_rgb = (group_rgb_sum[chosen]/support_groups[chosen, None]/255.).astype('<f4')
    appended, q, normals = seed_rows(spec['names'], xyz0[chosen_ref], unit[chosen_ref], seed_rgb)
    classes = geometry_classes(unit[refids], below_groups, above_groups)
    destination.mkdir()
    try:
        target = destination/'gaussians.ply'
        header, changed = re.subn(rb'(?m)^(element[ \t]+vertex[ \t]+)\d+([ \t]*\r?$)',
            lambda m: m[1]+str(spec['count']+len(chosen)).encode()+m[2], spec['header'])
        if changed != 1:
            raise AssemblyError('Cannot update only the PLY vertex count')
        original_payload = hashlib.sha256()
        with source.open('rb') as src, target.open('xb') as dst:
            if inspect_ply(src) != spec:
                raise AssemblyError('Original PLY schema changed')
            dst.write(header)
            for raw in iter(lambda: src.read(CHUNK_BYTES), b''):
                original_payload.update(raw)
                dst.write(raw)
            dst.write(appended.tobytes())
        with source.open('rb') as src, target.open('rb') as dst:
            inspect_ply(src)
            out_spec = inspect_ply(dst)
            if out_spec['count'] != spec['count']+len(chosen) or out_spec['names'] != spec['names']:
                raise AssemblyError('Output schema/count mismatch')
            remaining = spec['count']*spec['stride']
            while remaining:
                size = min(CHUNK_BYTES, remaining)
                a, b = src.read(size), dst.read(size)
                if len(a) != size or a != b:
                    raise AssemblyError('Original trained payload changed')
                remaining -= size
            if dst.read() != appended.tobytes():
                raise AssemblyError('On-disk seed payload differs')
        proofpath = destination/'seed_provenance.npz'
        with proofpath.open('xb') as stream:
            np.savez_compressed(stream, reference_row=chosen_ref.astype('<u4'), reference_xyz=xyz0[chosen_ref],
                reference_normal=reference['normals'][chosen_ref], normal_world=normals, quaternion_wxyz=q,
                output_row_index=np.arange(spec['count'], spec['count']+len(chosen), dtype='<u8'),
                nearest_original_center_m=distance[chosen], nearest_original_row=nearest[chosen].astype('<u4'),
                geometry_class=classes[chosen], below_camera_train_groups=below_groups[chosen],
                above_camera_train_groups=above_groups[chosen], support_train_groups=support_groups[chosen],
                support_train_frames=support_frames[chosen], color_support_groups=support_groups[chosen],
                color_support_frames=support_frames[chosen], rgb_srgb=seed_rgb,
                sum_of_group_mean_rgb_u8=group_rgb_sum[chosen], train_group_support_bits=group_bits[chosen],
                train_frame_support_bits=frame_bits[chosen])
        guard.check()
        record = {'schema_version': 1, 'status': 'measured_surface_seed_candidate_pending_teacher_audit_and_quality_evaluation',
            'cpu_only': True, 'gpu_used': False, 'quality_accepted': False, 'default_changed': False,
            'block_id': prep['block_id'], 'world_frame': 'camera_init', 'units': 'm',
            'preflight': preflight, 'estimated_peak_bytes': estimated_peak, 'threads_maximum': threads, 'blas_threads': 1,
            'policy': POLICY, 'existing_teacher_policy': teacher['policy'],
            'source_audit_entry': matched[0],
            'source_input': {'path': source.relative_to(project).as_posix(), 'sha256': guard.files[source]['sha256'],
                             'gaussians': spec['count'], 'payload_sha256': original_payload.hexdigest()},
            'output': {'path': target.relative_to(project).as_posix(), 'sha256': sha256(target),
                       'gaussians': spec['count']+len(chosen), 'file_bytes': target.stat().st_size},
            'seed_provenance': {'path': proofpath.relative_to(project).as_posix(), 'sha256': sha256(proofpath),
                'seed_count': len(chosen), 'reference_path': refpath.relative_to(project).as_posix(),
                'reference_sha256': guard.files[refpath]['sha256'],
                'bit_order': 'little: bit i of byte i//8; recorded TRAIN frame/group tables only'},
            'counts': {'reference_points': len(xyz0), 'valid_normal_reference_rows': initial_valid,
                'excluded_by_original_center_lt_2cm': initial_valid-count, 'uncovered_original_center_ge_2cm': count,
                'supported_at_least_3_groups_before_voxel': len(eligible),
                'voxel_duplicates_removed': len(eligible)-len(chosen), 'seeds_appended': len(chosen),
                'original_records_deleted': 0},
            'baseline_center_distances_m': baseline_distances,
            'gap_candidates_by_orientation': class_summary(orientation_classes(unit[refids]), distance),
            'supported_before_voxel_by_geometry': class_summary(classes[eligible], distance[eligible]),
            'seeds_by_geometry': class_summary(classes[chosen], distance[chosen]),
            'seed_group_support_quantiles': quantiles(support_groups[chosen]),
            'geometry_class_codes': CLASS_NAMES,
            'geometry_class_policy': 'Diagnostic proxies, not semantic GT: horizontal abs(nz)>=cos30; '
                'wall abs(nz)<=0.5; remaining sloped. Floor/ceiling require >=3 eligible TRAIN groups '
                'with camera Z above/below point by >=0.3m respectively and zero opposite groups. '
                'Any opposite support gives mixed_horizontal; other horizontal unresolved. '
                'No surface selection depends on these labels or normal sign.',
            'original_payload_all_bytes_compared': True, 'original_order_preserved': True,
            'seed_xyz_actual_reference_rows': True, 'plane_extension': False,
            'train_only_evidence_and_color': True, 'heldout_used_for_seed_geometry_or_color': False,
            'heldout_metadata_read_only_for_split_validation': True,
            'train_group_ids': [g for g, _ in groups], 'train_frames': frame_table,
            'color_policy': 'Nearest eligible TRAIN pixel in stored 8-bit sRGB, no gamma conversion. '
                'Average eligible cameras within each group, then equally average supported group means. '
                'RGB bounded [0,1], DC=(RGB-.5)/C0, SH rest zero. Bitsets enumerate all contributing views.',
            'coverage_scope': 'Center distance is a coverage proxy, not correct-surface or rendered-hole proof. '
                'Prepared v3 teacher is conditioned measured evidence, not independent ground truth. '
                'Original parameters preserved; appended fixed-opacity seeds are untrained and quality unassessed.',
            'inputs': guard.report(), 'seconds': time.monotonic()-start,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if record['peak_rss_kib']*1024 >= RSS_LIMIT:
            raise AssemblyError('CPU RSS reached 3 GiB bound; refusing publication')
        write_json(destination/'surface_seed_audit.json', record)
        return record
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-ply', 'data', 'output', 'source-audit'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        r = seed_measured_surfaces(args.source_ply, args.data, args.output,
                                  source_audit=args.source_audit, threads=args.threads)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(json.dumps({'status': 'incomplete', 'error': str(e)}), file=sys.stderr)
        return 2
    print(json.dumps({k: r[k] for k in ('status', 'output', 'counts', 'seconds', 'peak_rss_kib')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

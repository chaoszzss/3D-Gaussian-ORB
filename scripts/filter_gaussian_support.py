#!/usr/bin/env python3
"""Create traced, unmodified-record subsets of trained Gaussians near measured LiDAR.

This is a candidate selection experiment, not a newly trained or accepted map.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from orbvi_gaussian.training import confined_environment
os.environ.update(confined_environment())
import numpy as np
from scipy.spatial import cKDTree
from package_garage import inspect_ply, ply_header


def local(path):
    path = (ROOT / path).resolve()
    if path == ROOT or ROOT not in path.parents:
        raise ValueError('Path must be inside the reconstruction project')
    return path


def selected_rows(rows, names, reference, max_distance_m):
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError('Measured support radius must be positive and finite')
    if rows.ndim != 2 or rows.shape[1] != len(names) or not np.isfinite(rows).all():
        raise ValueError('Invalid trained Gaussian payload')
    if reference.ndim != 2 or reference.shape[1] != 3 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError('Reference must contain finite measured XYZ')
    xyz = rows[:, [names.index(k) for k in ('x', 'y', 'z')]]
    distances = cKDTree(reference).query(xyz, workers=4)[0]
    indices = np.flatnonzero(distances <= max_distance_m)
    if not len(indices):
        raise ValueError('Support filter would remove the entire source block')
    return indices, distances


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def train_positions(metadata):
    filenames = set(metadata['train_filenames'])
    # Camera3 gives a consistent physical rig reference; never use held-out photos.
    result = [np.asarray(f['transform_matrix'], dtype=np.float64)[:3, 3]
              for f in metadata['frames'] if f['file_path'] in filenames and f['camera_id'] == 'camera_3']
    if not result:
        raise ValueError('No actual training camera3 positions')
    return np.asarray(result)


def source_domain_mask(xyz, local_positions, global_positions, max_excess_m):
    if not np.isfinite(max_excess_m) or max_excess_m <= 0:
        raise ValueError('Source-domain excess must be positive and finite')
    own_distance = cKDTree(local_positions).query(xyz, workers=4)[0]
    best_distance = cKDTree(global_positions).query(xyz, workers=4)[0]
    # Full XYZ preserves distinct heights across nearby stacked floors.
    return own_distance - best_distance <= max_excess_m, own_distance, best_distance


def build_ownership(metadata_by_id):
    positions, memberships, spans = {}, {}, {}
    for bid, metadata in metadata_by_id.items():
        train = set(metadata['train_filenames'])
        groups = []
        for frame in metadata['frames']:
            if frame['file_path'] not in train or frame['camera_id'] != 'camera_3':
                continue
            group = int(frame['group_id'])
            xyz = np.asarray(frame['transform_matrix'], dtype=np.float64)[:3, 3]
            if group in positions and not np.allclose(positions[group], xyz, atol=1e-7, rtol=0):
                raise ValueError('Same source training pose differs between blocks')
            positions[group] = xyz
            memberships.setdefault(group, []).append(bid)
            groups.append(group)
        if not groups:
            raise ValueError('No training groups for ownership')
        spans[bid] = groups
    groups = sorted(positions)
    rank = {g: i for i, g in enumerate(groups)}
    spans = {bid: (min(rank[g] for g in gs), max(rank[g] for g in gs)) for bid, gs in spans.items()}
    owners = []
    for group in groups:
        def score(bid):
            first, last = spans[bid]
            return (abs(rank[group] - .5 * (first + last)) / (last - first + 1), bid)
        owners.append(min(memberships[group], key=score))
    return np.asarray([positions[g] for g in groups]), np.asarray(owners), groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block-ids', nargs='+', required=True)
    p.add_argument('--max-distance-m', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-domain-excess-m', type=float, default=0.,
                   help='Optional candidate gate: source training trajectory may be at most this far worse than global nearest trajectory')
    p.add_argument('--source-ownership', action='store_true',
                   help='Candidate spatial partition by nearest train pose, choosing source block closest to temporal center in overlaps')
    args = p.parse_args()
    if len(set(args.block_ids)) != len(args.block_ids):
        raise ValueError('Duplicate block IDs')
    manifest = json.loads((ROOT / 'maps/garage_manifest.json').read_text())
    by_id = {b['id']: b for b in manifest['blocks']}
    if any(bid not in by_id for bid in args.block_ids):
        raise ValueError('Unknown source block ID')
    domains = {}
    if args.source_domain_excess_m or args.source_ownership:
        metadata_by_id = {}
        for bid in by_id:
            meta = json.loads((ROOT / 'data/garage_surface_v1' / bid / 'transforms.json').read_text())
            metadata_by_id[bid] = meta
            domains[bid] = train_positions(meta)
        global_positions = np.unique(np.concatenate(list(domains.values())), axis=0)
        if args.source_ownership:
            owner_positions, owner_ids, owner_groups = build_ownership(metadata_by_id)
            owner_tree = cKDTree(owner_positions)
    out = local(args.output)
    if out.exists():
        raise FileExistsError('Candidate output already exists')
    out.mkdir(parents=True)
    sources = []
    for bid in args.block_ids:
        original = by_id[bid]
        if original['training']['status'] != 'trained':
            raise ValueError('Source block is not trained: ' + bid)
        source = local(Path('runs/garage_surface_v1') / bid / 'eval_dev/gaussians.ply')
        reference_path = local(Path('data/garage_surface_v1') / bid / 'lidar_reference.npz')
        source_hash = digest(source)
        reference_hash = digest(reference_path)
        with source.open('rb') as handle:
            spec = inspect_ply(handle)
            raw = handle.read()
        if len(raw) != spec['count'] * spec['stride']:
            raise ValueError('Truncated or trailing source PLY records')
        rows = np.frombuffer(raw, dtype='<f4').reshape(spec['count'], len(spec['names']))
        with np.load(reference_path, allow_pickle=False) as data:
            reference = data['xyz']
        indices, distances = selected_rows(rows, spec['names'], reference, args.max_distance_m)
        surface_count = len(indices)
        domain_report = None
        if args.source_domain_excess_m:
            xyz_all = rows[:, [spec['names'].index(k) for k in ('x', 'y', 'z')]]
            domain_keep, own_distance, best_distance = source_domain_mask(
                xyz_all, domains[bid], global_positions, args.source_domain_excess_m)
            indices = indices[domain_keep[indices]]
            if not len(indices):
                raise ValueError('Source domain gate would remove entire block')
            domain_report = {'max_excess_m': args.source_domain_excess_m,
                             'training_positions_only': True, 'distance_axes': 'world XYZ',
                             'own_distance_quantiles_m': np.quantile(own_distance, [.5, .9, .99]).tolist(),
                             'best_distance_quantiles_m': np.quantile(best_distance, [.5, .9, .99]).tolist()}
        ownership_report = None
        if args.source_ownership:
            xyz_all = rows[:, [spec['names'].index(k) for k in ('x', 'y', 'z')]]
            _, nearest_pose = owner_tree.query(xyz_all, workers=4)
            indices = indices[owner_ids[nearest_pose[indices]] == bid]
            if not len(indices):
                raise ValueError('Ownership would remove entire block')
            ownership_report = {'rule': 'nearest world XYZ training camera3 pose; temporal-center tie resolution',
                                'training_positions_only': True, 'training_pose_groups': len(owner_groups),
                                'independent_geometry_acceptance': False}
        selected = np.frombuffer(raw, dtype=np.uint8).reshape(spec['count'], spec['stride'])[indices]
        directory = out / bid
        directory.mkdir()
        destination = directory / 'gaussians.ply'
        with destination.open('xb') as handle:
            handle.write(ply_header(spec['names'], len(indices)))
            handle.write(selected.tobytes())
        np.save(directory / 'source_record_indices.npy', indices, allow_pickle=False)
        xyz = rows[indices][:, [spec['names'].index(k) for k in ('x', 'y', 'z')]]
        item = json.loads(json.dumps(original))
        item.pop('error', None)
        item['path'] = str(destination.relative_to(ROOT))
        item['bbox_m'] = {'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()}
        item['training']['gaussians'] = len(indices)
        item['training']['method'] = 'Unmodified subset of trained source Gaussian records'
        item['eval'] = {'status': 'not_evaluated', 'source_status': 'evaluated',
                        'note': 'Candidate subset requires real held-out evaluation'}
        report = {'id': bid, 'source': str(source.relative_to(ROOT)), 'source_sha256': source_hash,
                  'measured_reference': str(reference_path.relative_to(ROOT)),
                  'measured_reference_sha256': reference_hash, 'original_gaussians': len(rows),
                  'retained_gaussians': len(indices), 'max_distance_m': args.max_distance_m,
                  'surface_supported_gaussians': surface_count, 'source_domain_gate': domain_report,
                  'source_ownership': ownership_report,
                  'distance_quantiles_m': np.quantile(distances, [.5, .9, .95, .99, 1]).tolist(),
                  'source_indices': str((directory / 'source_record_indices.npy').relative_to(ROOT)),
                  'records_modified': False, 'geometry_protocol': 'Same input LiDAR, not independent GT',
                  'output_sha256': digest(destination)}
        if digest(source) != source_hash or digest(reference_path) != reference_hash:
            raise RuntimeError('Source changed during candidate filtering')
        (directory / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
        manifest_block = item
        sources.append((manifest_block, report))
        print(bid, len(rows), '->', len(indices), flush=True)
    manifest['blocks'] = [x[0] for x in sources]
    manifest['status'] = 'candidate_support_filtered_not_accepted'
    manifest['title'] = 'Measured-support Gaussian candidate; not quality accepted'
    manifest['coverage'] = {'source_block_ids': args.block_ids, 'block_count': len(args.block_ids),
                            'scope': 'candidate subset of real trained blocks',
                            'unobserved_building_complete': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (out / 'selection.json').write_text(json.dumps({'status': 'candidate', 'sources': [x[1] for x in sources],
        'high_fidelity_acceptance': 'not_assessed', 'records_modified': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""CPU-only evidence for the viewer adapter. Writes only reports/viewer_parity_v2*."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from direction_support import filter_direction_indices


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    meta_path = ROOT / 'maps/garage_surface_all28_noownership_sides_v2_full_20261009/metadata.json'
    meta = json.loads(meta_path.read_text())
    ply_path, flags_path = (ROOT / meta[k]['path'] for k in ('ply', 'side_flags'))
    assert sha(ply_path) == meta['ply']['sha256']
    assert sha(flags_path) == meta['side_flags']['sha256']
    flags = np.load(flags_path, mmap_mode='r', allow_pickle=False)
    count = meta['ply']['gaussians']
    assert flags.dtype == np.uint8 and flags.shape == (count,)
    counts = np.bincount(flags, minlength=4)
    assert len(counts) == 4 and dict(zip(map(str, range(4)), map(int, counts))) == meta['side_counts']
    with ply_path.open('rb') as stream:
        lines = []
        while True:
            line = stream.readline().decode('ascii').strip()
            lines.append(line)
            if line == 'end_header':
                break
        offset = stream.tell()
    fields = [line.split()[-1] for line in lines if line.startswith('property float ')]
    assert len(fields) == 62
    records = np.memmap(ply_path, dtype='<f4', mode='r', offset=offset, shape=(count, len(fields)))
    indices = np.linspace(0, count - 1, 2048).astype(np.int64)
    sample = records[indices]
    means = sample[:, [fields.index(k) for k in ('x', 'y', 'z')]]
    quats = sample[:, [fields.index('rot_' + str(i)) for i in range(4)]]
    sample_flags = flags[indices]
    metric_path = ROOT / 'runs/fidelity_v2/original_noownership015/block_015/metrics.json'
    metrics = json.loads(metric_path.read_text())
    frame = next(row for row in metrics['frames'] if row['dataset_index'] == 8)
    camera = np.asarray(frame['camera']['camera_to_worlds_ns'])[:, 3]
    cameras = [camera, camera + [10, 0, 0], camera + [0, -10, 0], camera + [0, 0, 10]]
    expected = []
    for point in cameras:
        keep, info = filter_direction_indices({'means': means, 'quats': quats},
            {'flags': sample_flags, 'grazing': meta['render_grazing_cosine']}, point, np.arange(len(indices)))
        visible = np.zeros(len(indices), dtype=bool)
        visible[keep] = True
        expected.append({'camera_world': point.tolist(), 'native_visible': visible.tolist(), 'info': info})
    native_float = metric_path.parent / '008_float.npz'
    with np.load(native_float, allow_pickle=False) as data:
        float_info = {key: {'dtype': str(data[key].dtype), 'shape': list(data[key].shape)} for key in data.files}
    report = {'status': 'cpu_inputs_verified_no_browser_render', 'gpu_used': False,
        'metadata_path': str(meta_path.relative_to(ROOT)), 'metadata_sha256': sha(meta_path),
        'ply_sha256': meta['ply']['sha256'], 'side_flags_sha256': meta['side_flags']['sha256'],
        'gaussians': count, 'side_counts': dict(zip(map(str, range(4)), map(int, counts))),
        'normal_convention': meta['normal_convention'], 'observed_cosine_min': meta['observed_cosine_min'],
        'render_grazing_cosine': meta['render_grazing_cosine'],
        'indices': indices.tolist(), 'means': means.tolist(), 'quaternions_wxyz': quats.tolist(),
        'flags': sample_flags.tolist(), 'native_direction_checks': expected,
        'native_frame_reference': {'metrics_path': str(metric_path.relative_to(ROOT)), 'metrics_sha256': sha(metric_path),
            'float_path': str(native_float.relative_to(ROOT)), 'float_sha256': sha(native_float), 'arrays': float_info,
            'source_ply_sha256': metrics['source_ply_sha256'], 'camera': frame['camera'],
            'dataset_index': frame['dataset_index'], 'group_id': frame['group_id'], 'camera_id': frame['camera_id'],
            'inference_sh_degree': metrics['inference_sh_degree'], 'acceptance_mask_path': frame['acceptance_mask_path'],
            'direction_support_sha256': None,
            'background_rgb': [0.1490, 0.1647, 0.2157],
            'background_evidence': 'config background_color=random; installed Nerfstudio eval default; verify or record actual tensor before formal parity',
            'browser_render_performed': False},
        'native_parity_psnr_delta_db': None, 'native_parity_ssim_delta': None,
        'limitations': ['This native reference was rendered WITHOUT side gating. It cannot be compared to the new gated bundle as renderer parity.',
            'Direction decisions are CPU helper results, not shader or full-image comparisons.',
            'Full 8.9M PLY exceeds current viewer budget; use correctly remapped static partition sidecars or explicitly scheduled full-scene loader.']}
    output = ROOT / 'reports/viewer_parity_v2_cpu_samples.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'gaussians': count, 'sampled': len(indices),
        'camera_positions': len(cameras), 'output': str(output.relative_to(ROOT))}))


if __name__ == '__main__':
    main()

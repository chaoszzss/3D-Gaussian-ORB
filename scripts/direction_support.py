"""Inference-only, fixed TRAIN observation-side support for a static 2DGS bundle.

Unknown and observed-both-side surfaces remain two-sided. No camera block is
selected at inference. Source PLY and side array are immutable and hash-bound.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(raw)
    return digest.hexdigest()


def asset_state(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verify_direction_support_unchanged(support):
    for path, state in support['asset_states'].items():
        if asset_state(path) != state or sha256(path) != support['asset_digests'][path]:
            raise ValueError('Direction-support asset changed during evaluation: ' + path)


def local_asset(path, base):
    path = Path(path)
    candidates = [path] if path.is_absolute() else [base / path, ROOT / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if ROOT in resolved.parents and resolved.is_file():
            return resolved
    raise ValueError('Missing project-local direction-support asset: ' + str(path))


def load_direction_support(path, candidate, parameters):
    path = local_asset(path, ROOT)
    states = {str(path): asset_state(path)}
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError('Direction-support metadata exceeds 16 MiB')
    metadata = json.loads(path.read_text())
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported direction-support schema')
    ply = metadata['ply']
    count = candidate['gaussians']
    if ply['sha256'] != candidate['sha256'] or ply['gaussians'] != count:
        raise ValueError('Direction support belongs to a different complete PLY')
    if local_asset(ply['path'], path.parent) != Path(candidate['path']).resolve():
        raise ValueError('Direction-support PLY path mismatch')
    if metadata['normal_convention'] != 'quaternion_local_z':
        raise ValueError('Require quaternion local +Z support convention')
    descriptor = metadata['side_flags']
    if descriptor.get('dtype') != 'uint8' or descriptor.get('count') != count:
        raise ValueError('Direction-support descriptor does not match source rows')
    flags_path = local_asset(descriptor['path'], path.parent)
    states[str(flags_path)] = asset_state(flags_path)
    if flags_path.stat().st_size > count + 65536 or sha256(flags_path) != descriptor['sha256']:
        raise ValueError('Direction-support array size/digest mismatch')
    flags = np.load(flags_path, mmap_mode='r', allow_pickle=False)
    if flags.shape != (count,) or flags.dtype != np.uint8 or np.any(flags > 3):
        raise ValueError('Direction flags must be uint8[N], 0..3')
    evidence = metadata.get('training_evidence', {})
    if (metadata.get('policy') != 'strong_unilateral_train_group_support'
            or evidence.get('split') != 'train' or evidence.get('train_only') is not True
            or evidence.get('heldout_used') is not False
            or evidence.get('minimum_unique_groups') != 3
            or evidence.get('opposite_side_required_count') != 0):
        raise ValueError('Require fixed TRAIN-only observation evidence')
    counts = []
    for key in ('positive_train_groups', 'negative_train_groups'):
        desc = evidence[key]
        asset = local_asset(desc['path'], path.parent)
        states[str(asset)] = asset_state(asset)
        if (desc.get('dtype') != 'uint16' or desc.get('count') != count
                or asset.stat().st_size > 2 * count + 65536 or sha256(asset) != desc['sha256']):
            raise ValueError('Direction TRAIN evidence size/digest mismatch')
        values = np.load(asset, mmap_mode='r', allow_pickle=False)
        if values.shape != (count,) or values.dtype != np.uint16:
            raise ValueError('Direction TRAIN evidence must be uint16[N]')
        counts.append(values)
    for start in range(0, count, 200000):
        pos, neg = (values[start:start+200000] for values in counts)
        expected = np.zeros(len(pos), dtype=np.uint8)
        expected[(pos >= 3) & (neg == 0)] = 1
        expected[(neg >= 3) & (pos == 0)] = 2
        expected[(pos > 0) & (neg > 0)] = 3
        if not np.array_equal(flags[start:start+200000], expected):
            raise ValueError('Direction flags disagree with frozen TRAIN group counts')
    grazing = metadata['render_grazing_cosine']
    if type(grazing) not in (int, float) or not np.isfinite(grazing) or not 0 <= grazing <= .2:
        raise ValueError('Invalid frozen grazing threshold')
    if len(parameters['means']) != count or len(parameters['quats']) != count:
        raise ValueError('Direction support requires every immutable source record')
    info = {'path': str(path), 'sha256': sha256(path), 'side_flags_sha256': descriptor['sha256'],
            'source_ply_sha256': candidate['sha256'], 'source_records': count,
            'normal_convention': metadata['normal_convention'], 'render_grazing_cosine': grazing,
            'model_kind': 'static_2dgs_with_train_observation_side_support',
            'policy': '0/3 retain both sides; unilateral support rejects only opposite non-grazing viewing direction'}
    digests = {str(path): info['sha256'], str(flags_path): descriptor['sha256']}
    for key in ('positive_train_groups', 'negative_train_groups'):
        desc = evidence[key]
        digests[str(local_asset(desc['path'], path.parent))] = desc['sha256']
    result = {'flags': flags, 'grazing': float(grazing), 'info': info, 'asset_states': states,
              'asset_digests': digests}
    verify_direction_support_unchanged(result)
    return result


def filter_direction_indices(parameters, support, camera_position, indices):
    """Preserve exact source order, including equal-depth tie order."""
    indices = np.asarray(indices, dtype=np.int64)
    flags = support['flags'][indices]
    unilateral = (flags == 1) | (flags == 2)
    selected = indices[unilateral]
    keep = np.ones(len(indices), dtype=bool)
    if len(selected):
        q = parameters['quats'][selected].astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        normal = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
        view = np.asarray(camera_position, dtype=np.float64).reshape(1, 3) - parameters['means'][selected]
        length = np.linalg.norm(view, axis=1)
        cosine = np.divide(np.einsum('ij,ij->i', normal, view), length,
                           out=np.zeros(len(view)), where=length > 1e-10)
        sides = flags[unilateral]
        rejected = ((sides == 1) & (cosine < -support['grazing'])) | ((sides == 2) & (cosine > support['grazing']))
        keep[unilateral] = ~rejected
    return indices[keep], {'native_active_records': len(indices), 'side_rejected_records': int((~keep).sum()),
                           'retained_records': int(keep.sum()),
                           'retained_index_sha256': hashlib.sha256(indices[keep].astype('<i8').tobytes()).hexdigest()}

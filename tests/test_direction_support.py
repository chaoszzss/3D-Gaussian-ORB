import importlib.util
from pathlib import Path
import numpy as np
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('direction_support', ROOT / 'scripts/direction_support.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_opposite_side_only_with_unknown_and_bilateral_preserved():
    params = {'means': np.zeros((4, 3)), 'quats': np.tile([1., 0., 0., 0.], (4, 1))}
    support = {'flags': np.array([0, 1, 2, 3], np.uint8), 'grazing': .05}
    ids, info = M.filter_direction_indices(params, support, [0, 0, 2], np.arange(4))
    np.testing.assert_array_equal(ids, [0, 1, 3])
    assert info['side_rejected_records'] == 1
    ids, _ = M.filter_direction_indices(params, support, [0, 0, -2], np.arange(4))
    np.testing.assert_array_equal(ids, [0, 2, 3])


def test_grazing_and_camera_at_center_preserve_both_sides():
    params = {'means': np.zeros((2, 3)), 'quats': np.tile([1., 0., 0., 0.], (2, 1))}
    support = {'flags': np.array([1, 2], np.uint8), 'grazing': .05}
    for camera in ([1., 0., .001], [0., 0., 0.]):
        ids, _ = M.filter_direction_indices(params, support, camera, [0, 1])
        np.testing.assert_array_equal(ids, [0, 1])


def test_world_rotated_normal_and_source_order_are_used():
    q = np.array([2**-.5, 0., 2**-.5, 0.])
    params = {'means': np.zeros((3, 3)), 'quats': np.tile(q, (3, 1))}
    support = {'flags': np.array([1, 2, 3], np.uint8), 'grazing': .05}
    ids, _ = M.filter_direction_indices(params, support, [-1., 0., 0.], [2, 0, 1])
    np.testing.assert_array_equal(ids, [2, 1])


def support_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'ROOT', tmp_path)
    ply = tmp_path / 'model.ply'
    ply.write_bytes(b'fixture model content')
    candidate = {'path': str(ply), 'sha256': M.sha256(ply), 'gaussians': 4}
    def array(name, value, dtype):
        path = tmp_path / (name + '.npy')
        np.save(path, np.array(value, dtype=dtype))
        return {'path': str(path), 'sha256': M.sha256(path), 'count': 4, 'dtype': dtype}
    evidence = dict(split='train', train_only=True, heldout_used=False, minimum_unique_groups=3,
                    opposite_side_required_count=0,
                    positive_train_groups=array('positive', [0, 3, 0, 1], 'uint16'),
                    negative_train_groups=array('negative', [0, 0, 3, 1], 'uint16'))
    metadata = dict(schema_version=1, ply=candidate, normal_convention='quaternion_local_z',
                    policy='strong_unilateral_train_group_support', render_grazing_cosine=.05,
                    side_flags=array('flags', [0, 1, 2, 3], 'uint8'), training_evidence=evidence)
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(metadata))
    params = {'means': np.zeros((4, 3)), 'quats': np.tile([1., 0., 0., 0.], (4, 1))}
    return path, metadata, candidate, params


def test_bound_train_evidence_loads_and_detects_mutation(tmp_path, monkeypatch):
    path, meta, candidate, params = support_fixture(tmp_path, monkeypatch)
    result = M.load_direction_support(path, candidate, params)
    M.verify_direction_support_unchanged(result)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        M.verify_direction_support_unchanged(result)


@pytest.mark.parametrize('change', ['heldout', 'wrong_counts', 'wrong_ply', 'wrong_flag_digest'])
def test_invalid_observation_side_evidence_rejected(tmp_path, monkeypatch, change):
    path, meta, candidate, params = support_fixture(tmp_path, monkeypatch)
    if change == 'heldout':
        meta['training_evidence']['heldout_used'] = True
    elif change == 'wrong_counts':
        desc = meta['training_evidence']['positive_train_groups']
        np.save(desc['path'], np.array([0, 1, 0, 1], np.uint16))
        desc['sha256'] = M.sha256(desc['path'])
    elif change == 'wrong_ply':
        meta['ply']['sha256'] = 'other'
        candidate = dict(candidate, sha256='original')
    else:
        meta['side_flags']['sha256'] = 'other'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        M.load_direction_support(path, candidate, params)

"""Single-block observed surface seeds: no invented geometry, leakage or lost prefix."""
import sys
sys.dont_write_bytecode = True
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import seed_measured_surface_gaussians as M
from assemble_core_gaussians import inspect_ply, ply_header, sha256


@pytest.fixture
def scene():
    with tempfile.TemporaryDirectory(prefix='surfaceseed_fixture_v2_', dir=ROOT/'reports') as tmp:
        root = Path(tmp)
        for path in ('warm', 'data', 'original', 'data/images', 'data/masks', 'data/depth', 'data/depth_masks'):
            (root/path).mkdir()
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz']+[f'f_dc_{i}' for i in range(3)]
        names += [f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        raw = (np.arange(len(names), dtype='<f4')/100)[None]
        raw[0, :3] = [1.005, 2, 1]
        raw[0, names.index('f_dc_0')] = -0.
        source = root/'warm/gaussians.ply'
        source.write_bytes(ply_header(names, 1)+raw.tobytes())
        audit = {'sources': [{'id': 'block_000', 'path': 'warm/gaussians.ply',
            'source_sha256': sha256(source), 'exported_gaussians': 1}]}
        (root/'source_audit.json').write_text(json.dumps(audit))
        # floor, duplicate floor, wall, ceiling, near old center, unknown, ego,
        # invalid normal, foreground mismatch, behind, four-camera/one-group only.
        xyz = np.array([[0, 2, 0], [.005, 2, 0], [.5, 2, 1], [0, 2, 2], [1, 2, 1],
                        [-1, 2, 1], [0, 2, .5], [1, 2, 0], [0, 2.4, 1], [0, -2, 1],
                        [1.5, 2, 1]], dtype='<f4')
        normals = np.tile([0, 0, 1.], (len(xyz), 1)).astype('<f4')
        normals[2] = [1, 0, 0]
        normals[3] = [0, 0, -1]
        valid = np.ones(len(xyz), bool)
        valid[7] = False
        np.savez_compressed(root/'data/lidar_reference.npz', xyz=xyz, normals=normals, normal_valid=valid)
        frames, teachers, assets = [], [], []
        for i, group in enumerate([10, 10, 10, 10, 20, 30]):
            pose = np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 1], [0, 0, 0, 1]])
            f = {'file_path': f'images/{i}.png', 'mask_path': f'masks/{i}.png',
                'lidar_depth_path': f'depth/{i}.npy', 'lidar_depth_mask_path': f'depth_masks/{i}.png',
                'group_id': str(group), 'timestamp_ns': group, 'camera_id': f'camera_{i%4}',
                'source_file_path': f'raw/{i}.jpg', 'w': 16, 'h': 16,
                'fl_x': 4., 'fl_y': 4., 'cx': 8.5, 'cy': 8.5, 'transform_matrix': pose.tolist()}
            rgb = np.zeros((16, 16, 3), np.uint8)
            rgb[:, :, 0 if group == 10 else 1 if group == 20 else 2] = 255
            known = np.full((16, 16), 255, np.uint8)
            ego = known.copy()
            known[8, 6] = 0
            ego[9, 8] = 0
            if group != 10:
                known[8, 11] = 0
            for array, key in ((rgb, 'file_path'), (ego, 'mask_path'), (known, 'lidar_depth_mask_path')):
                Image.fromarray(array).save(root/'data'/f[key])
            np.save(root/'data'/f['lidar_depth_path'], np.full((16, 16), 2, np.float32))
            assets.append({'file_path': f['file_path'], 'image_sha256': sha256(root/'data'/f['file_path']),
                'mask_sha256': sha256(root/'data'/f['mask_path']), 'split': 'train_filenames'})
            teachers.append({'file_path': f['file_path'], 'depth_sha256': sha256(root/'data'/f['lidar_depth_path']),
                'support_sha256': sha256(root/'data'/f['lidar_depth_mask_path'])})
            frames.append(f)
        held = copy.deepcopy(frames[0])
        held.update(file_path='images/heldout_NOT_PRESENT.png', group_id='40', timestamp_ns=40)
        meta = {'orientation_override': 'none', 'frames': frames+[held],
            'train_filenames': [f['file_path'] for f in frames],
            'val_filenames': [held['file_path']], 'test_filenames': []}
        for path in ('data/transforms.json', 'original/transforms.json'):
            (root/path).write_text(json.dumps(meta))
        prep = {'block_id': 'block_000', 'world_frame': 'camera_init', 'units': 'metres',
            'resolution_pilot': {'source_dataset': 'original',
                'source_hashes': {'transforms.json': sha256(root/'original/transforms.json')},
                'world_recentered': False, 'physical_optics_unchanged': True,
                'exact_source_poses_splits_timestamps': True, 'pixel_center': {'offset': .5}, 'frames': assets},
            'teacher_provenance': {'policy': {'version': 'surfel_depth_v3', 'ground_support': True,
                'world_up': [0, 0, 1], 'foreground_occlusion_tolerance_m': .1},
                'reference_sha256': sha256(root/'data/lidar_reference.npz'), 'reference_points': len(xyz),
                'frames': teachers}}
        (root/'data/preparation.json').write_text(json.dumps(prep))
        yield root, names, raw, xyz, meta, prep


def build(scene, output='candidate', **kwargs):
    return M.seed_measured_surfaces('warm/gaussians.ply', 'data', output, source_audit='source_audit.json',
                                    project=scene[0], threads=kwargs.pop('threads', 1), **kwargs)


def test_floor_wall_ceiling_reference_support_balanced_color_and_exact_original(scene):
    root, names, raw, xyz, meta, prep = scene
    original = (root/'warm/gaussians.ply').read_bytes()
    report = build(scene)
    assert report['counts']['seeds_appended'] == 3
    assert report['counts']['supported_at_least_3_groups_before_voxel'] == 4
    assert report['counts']['excluded_by_original_center_lt_2cm'] == 1
    with np.load(root/'candidate/seed_provenance.npz') as p:
        assert p['reference_row'].tolist() == [0, 2, 3]
        np.testing.assert_array_equal(p['reference_xyz'], xyz[[0, 2, 3]])
        assert p['geometry_class'].tolist() == [1, 5, 2]
        assert p['support_train_groups'].tolist() == [3, 3, 3]
        assert p['support_train_frames'].tolist() == [6, 6, 6]
        np.testing.assert_allclose(p['rgb_srgb'], np.full((3, 3), 1/3), atol=2e-8, rtol=0)
        assert np.unpackbits(p['train_group_support_bits'], axis=1, bitorder='little').sum(axis=1).tolist() == [3]*3
        assert np.unpackbits(p['train_frame_support_bits'], axis=1, bitorder='little').sum(axis=1).tolist() == [6]*3
        assert np.all(p['nearest_original_center_m'] >= .02)
        np.testing.assert_allclose(p['nearest_original_center_m'], np.linalg.norm(
            p['reference_xyz'].astype(float)-raw[0, :3].astype(float), axis=1), atol=1e-12, rtol=0)
        np.testing.assert_allclose(p['normal_world'], [[0, 0, 1], [1, 0, 0], [0, 0, -1]])
    with (root/'candidate/gaussians.ply').open('rb') as f:
        spec = inspect_ply(f)
        payload = f.read()
    assert spec['count'] == 4
    assert payload[:raw.nbytes] == raw.tobytes()
    out = np.frombuffer(payload, dtype='<f4').reshape(4, -1)
    np.testing.assert_array_equal(out[1:, :3], xyz[[0, 2, 3]])
    np.testing.assert_allclose(np.exp(out[1:, [names.index('scale_0'), names.index('scale_1')]]), .02, atol=1e-8)
    np.testing.assert_allclose(np.exp(out[1:, names.index('scale_2')]), .001, atol=1e-9)
    np.testing.assert_allclose(1/(1+np.exp(-out[1:, names.index('opacity')])), .9, atol=1e-7)
    assert np.all(out[1:, [names.index(f'f_rest_{i}') for i in range(45)]] == 0)
    assert (root/'warm/gaussians.ply').read_bytes() == original
    assert report['heldout_used_for_seed_geometry_or_color'] is False
    assert not any('heldout_NOT_PRESENT' in k for k in report['inputs'])
    assert report['original_payload_all_bytes_compared'] is True
    assert report['seeds_by_geometry']['floor_proxy']['count'] == 1
    assert report['seeds_by_geometry']['ceiling_proxy']['count'] == 1
    assert report['seeds_by_geometry']['wall_normal']['count'] == 1
    assert report['baseline_center_distances_m']['all_valid_reference']['count'] == len(xyz)-1
    assert json.loads((root/'candidate/surface_seed_audit.json').read_text())['output'] == report['output']


def test_unknown_ego_occlusion_behind_and_no_height_gate(scene):
    frame = scene[4]['frames'][0]
    depth = np.full((16, 16), 2., np.float32)
    known, ego = np.ones((16, 16), np.uint8), np.ones((16, 16), np.uint8)
    known[8, 6] = 0
    ego[9, 8] = 0
    ids, _, _ = M.eligible_observations(scene[3], frame, depth, known, ego)
    assert {0, 2, 3} <= set(ids)
    assert not {5, 6, 8, 9} & set(ids)
    assert len(M.eligible_observations(scene[3], frame, depth, known*0, ego)[0]) == 0
    depth[:] = np.nan
    assert len(M.eligible_observations(scene[3], frame, depth, known, ego)[0]) == 0


def test_normals_signed_wall_and_antipodal_quaternions():
    n = np.array([[0, 0, 1], [0, 0, -2], [1, 0, 0], [-1, 0, 0],
                  [0, 1, 0], [0, -1, 0], [.3, .2, -.99999], [1e-12, 0, -1]])
    q, normalized = M.normal_quaternions(n)
    w, x, y, z = q.T
    derived = np.column_stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    np.testing.assert_allclose(derived, normalized, atol=2e-7, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7)
    with pytest.raises(M.AssemblyError, match='Zero'):
        M.normal_quaternions(np.zeros((1, 3)))


def test_invalid_normal_never_seeded_and_walls_retained():
    xyz = np.zeros((5, 3), np.float32)
    normals = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0], [np.nan, 0, 1], [0, 0, -1]])
    ids, _ = M.valid_reference_rows(xyz, normals, np.array([1, 1, 1, 1, 0], bool))
    assert ids.tolist() == [0, 1]


def test_four_cameras_cannot_fake_three_train_groups_and_split_leakage(scene):
    report = build(scene)
    with np.load(scene[0]/'candidate/seed_provenance.npz') as p:
        assert 10 not in p['reference_row']
    meta = copy.deepcopy(scene[4])
    meta['frames'][-1]['group_id'] = '10'
    with pytest.raises(M.AssemblyError, match='group leakage'):
        M.train_groups(meta)
    assert len(report['train_group_ids']) == 3


def test_stable_actual_row_voxel_never_centroid_and_no_silent_cap():
    xyz = np.array([[.001, 0, 0], [.019, 0, 0], [.02, 0, 0], [-.001, 0, 0]])
    rows = np.array([9, 4, 8, 2])
    chosen = M.stable_voxel_rows(xyz, rows)
    assert rows[chosen].tolist() == [2, 4, 8]
    np.testing.assert_array_equal(xyz[chosen], xyz[[3, 1, 2]])
    M.enforce_seed_limit(400000, 400000)
    with pytest.raises(M.AssemblyError, match='refusing to truncate'):
        M.enforce_seed_limit(400001, 500000)
    with pytest.raises(M.AssemblyError, match='refusing to truncate'):
        M.enforce_seed_limit(5, 4)


def test_geometry_labels_use_observed_height_not_normal_sign():
    n = np.tile([0, 0, -1.], (5, 1))
    c = M.geometry_classes(n, np.array([3, 0, 1, 0, 2]), np.array([0, 3, 1, 0, 0]))
    assert c.tolist() == [1, 2, 3, 4, 4]


def test_existing_output_thread_resource_fail_before_writing(scene, monkeypatch):
    root = scene[0]
    (root/'old').mkdir()
    (root/'old/sentinel').write_bytes(b'preserve')
    with pytest.raises(M.AssemblyError, match='NEW directory'):
        build(scene, 'old')
    assert (root/'old/sentinel').read_bytes() == b'preserve'
    with pytest.raises(M.AssemblyError, match='1..3'):
        build(scene, 'invalid_threads', threads=4)
    def fail(*args, **kwargs):
        raise M.AssemblyError('CPU memory preflight failed')
    monkeypatch.setattr(M, 'resource_preflight', fail)
    with pytest.raises(M.AssemblyError, match='memory preflight'):
        build(scene)
    assert not (root/'candidate').exists()


@pytest.mark.parametrize('changed', ['depth', 'mask', 'image', 'source', 'pose', 'split_proof', 'teacher_version'])
def test_stale_or_mismatched_proofs_rejected(scene, changed):
    root, _, _, _, meta, prep = scene
    expected = 'TRAIN asset SHA mismatch'
    if changed == 'depth':
        np.save(root/'data'/meta['frames'][0]['lidar_depth_path'], np.zeros((16, 16), np.float32))
    elif changed in ('mask', 'image'):
        key = 'mask_path' if changed == 'mask' else 'file_path'
        with (root/'data'/meta['frames'][0][key]).open('ab') as f:
            f.write(b'changed')
    elif changed == 'source':
        with (root/'warm/gaussians.ply').open('r+b') as f:
            f.seek(-4, 2)
            f.write(np.float32(5).tobytes())
        expected = 'source audit'
    elif changed == 'pose':
        meta['frames'][0]['transform_matrix'][0][3] += .01
        (root/'data/transforms.json').write_text(json.dumps(meta))
        expected = 'Original pose'
    elif changed == 'split_proof':
        prep['resolution_pilot']['frames'][0]['split'] = 'val_filenames'
        (root/'data/preparation.json').write_text(json.dumps(prep))
        expected = 'split proof'
    elif changed == 'teacher_version':
        prep['teacher_provenance']['policy']['version'] = 'surfel_depth_v2'
        (root/'data/preparation.json').write_text(json.dumps(prep))
        expected = 'v3'
    with pytest.raises(M.AssemblyError, match=expected):
        build(scene)
    assert not (root/'candidate').exists()

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('support_filter', ROOT / 'scripts/filter_gaussian_support.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_support_is_metric_xyz_and_retains_original_record_bits():
    # Deliberately reordered XYZ; source attributes include signed zero/SH data.
    names = ('z', 'opacity', 'x', 'f_dc_0', 'y')
    rows = np.array([[0., -0., 0., -7., 0.], [.3, 20., 0., 2., 0.],
                     [0., -2., .1, 9., .1]], dtype='<f4')
    ids, distance = module.selected_rows(rows, names, np.zeros((1, 3)), .2)
    assert ids.tolist() == [0, 2]  # Unsupported height cannot be accepted by XY proximity.
    assert distance[1] == pytest.approx(.3)
    chosen = rows.view('u1').reshape(len(rows), -1)[ids]
    assert chosen[0].tobytes() == rows[0].tobytes()
    assert chosen[1].tobytes() == rows[2].tobytes()
    assert np.signbit(rows[ids][0, 1])


@pytest.mark.parametrize('radius', [0., -1., np.inf, np.nan])
def test_invalid_metric_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        module.selected_rows(np.zeros((1, 3)), ('x', 'y', 'z'), np.zeros((1, 3)), radius)


def test_empty_or_nonfinite_measured_reference_is_not_support():
    for reference in [np.empty((0, 3)), np.array([[0., np.nan, 0.]])]:
        with pytest.raises(ValueError):
            module.selected_rows(np.zeros((1, 3)), ('x', 'y', 'z'), reference, .2)


def test_refuses_empty_candidate_instead_of_publishing_completed_block():
    with pytest.raises(ValueError, match='entire source block'):
        module.selected_rows(np.ones((1, 3)), ('x', 'y', 'z'), np.zeros((1, 3)), .2)


def test_source_domain_preserves_height_and_rejects_foreign_floor():
    xyz = np.array([[3., 0., 0.], [0., 0., 3.7]])
    local = np.array([[0., 0., 0.]])
    global_positions = np.array([[0., 0., 0.], [0., 0., 4.]])
    keep, own, best = module.source_domain_mask(xyz, local, global_positions, 2.)
    assert keep.tolist() == [True, False]
    assert own[1] == pytest.approx(3.7)
    assert best[1] == pytest.approx(.3)


def test_training_domain_excludes_heldout_pose_and_other_camera():
    def frame(name, z, camera='camera_3'):
        T = np.eye(4); T[2, 3] = z
        return {'file_path': name, 'camera_id': camera, 'transform_matrix': T.tolist()}
    metadata = {'train_filenames': ['train.png', 'side.png'],
                'frames': [frame('train.png', 1.), frame('heldout.png', 99.),
                           frame('side.png', 77., 'camera_2')]}
    assert module.train_positions(metadata).tolist() == [[0., 0., 1.]]


def test_ownership_resolves_actual_overlap_and_preserves_z():
    def metadata(groups, height):
        frames = []
        for group in groups:
            T = np.eye(4); T[0, 3] = group; T[2, 3] = height
            frames.append({'group_id': str(group), 'camera_id': 'camera_3',
                           'file_path': str(group)+'.png', 'transform_matrix': T.tolist()})
        return {'frames': frames, 'train_filenames': [f['file_path'] for f in frames]}
    points, owners, groups = module.build_ownership({'lower_a': metadata([0, 1, 2], 0.),
        'lower_b': metadata([2, 3, 4], 0.), 'upper': metadata([5, 6, 7], 4.)})
    assert groups == list(range(8))
    assert owners[2] == 'lower_a'  # Equal center distance uses stable source ID.
    assert owners[3] == 'lower_b'
    assert points[6, 2] == 4.

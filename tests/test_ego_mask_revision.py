import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ego_revision', ROOT/'scripts/apply_ego_mask_revision.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_ego_mask_only_removes_eligibility_and_matching_depth():
    old = np.array([[255, 0], [255, 255]], np.uint8)
    extra = np.array([[0, 255], [255, 255]], np.uint8)
    depth = np.array([[3., 4.], [np.nan, 6.]], np.float32)
    support = np.full((2,2), 255, np.uint8)
    eligible, new_depth, known = module.tighten_support(old, extra, depth, support)
    assert eligible.tolist() == [[0, 0], [255, 255]]
    assert new_depth.tolist() == [[0., 0.], [0., 6.]]
    assert known.tolist() == [[0, 0], [0, 255]]
    assert old.tolist() == [[255, 0], [255, 255]]  # Source array remains unchanged.


def test_mask_revision_rejects_changed_camera_geometry_before_warp():
    base = {'virtual_views': [], 'cameras': {'camera_3': {'intrinsics': {'fx': 1.}}}}
    changed = {'virtual_views': [], 'cameras': {'camera_3': {'intrinsics': {'fx': 2.}}}}
    with pytest.raises(ValueError, match='changed camera'):
        module.revision_masks(base, changed)


def test_mask_revision_rejects_resizing_support_planes():
    with pytest.raises(ValueError, match='source dimensions'):
        module.tighten_support(np.ones((2,2)), np.ones((1,2)), np.ones((2,2)), np.ones((2,2)))


def test_full_copy_publishes_new_manifest_without_overwriting_parent(tmp_path, monkeypatch):
    # Actual file-copy/publication regression for the interrupted real run.
    import cv2
    source = tmp_path/'source'; source.mkdir()
    for folder in ('images','masks','depth','depth_masks'): (source/folder).mkdir()
    cv2.imwrite(str(source/'images/frame.png'), np.full((2,2,3), 120, np.uint8))
    cv2.imwrite(str(source/'masks/frame.png'), np.full((2,2), 255, np.uint8))
    cv2.imwrite(str(source/'depth_masks/frame.png'), np.full((2,2), 255, np.uint8))
    np.save(source/'depth/frame.npy', np.full((2,2), 7., np.float32))
    calibration = tmp_path/'calibration.yaml'; calibration.write_text('fixture')
    revision = tmp_path/'revision.yaml'; revision.write_text('fixture revision')
    frame = {'file_path':'images/frame.png', 'mask_path':'masks/frame.png',
             'lidar_depth_path':'depth/frame.npy', 'lidar_depth_mask_path':'depth_masks/frame.png',
             'camera_id':'camera_3','view_id':'center','w':2,'h':2,
             'fl_x':1.,'fl_y':1.,'cx':1.,'cy':1.}
    metadata = {'frames':[frame], 'train_filenames':['images/frame.png'],
                'val_filenames':[], 'test_filenames':[]}
    (source/'transforms.json').write_text(json.dumps(metadata))
    prep = {'method':'real multi-block reassociation of consistent prepared sources',
            'calibration':str(calibration), 'calibration_sha256':module.sha(calibration),
            'export':{'paths':{}},'teacher_provenance':{'frames':[{'file_path':'images/frame.png'}]},
            'copied_asset_hashes':{'masks/frame.png':module.sha(source/'masks/frame.png')}}
    (source/'preparation.json').write_text(json.dumps(prep))
    parent_hash = module.sha(source/'preparation.json')
    image_hash = module.sha(source/'images/frame.png')
    monkeypatch.setattr(module, 'load_calibration', lambda *a, **kw: {})
    mask = np.array([[255,0],[255,255]],np.uint8)
    monkeypatch.setattr(module, 'revision_masks', lambda *a: (
        {('camera_3','center'):mask},{}, {('camera_3','center'):(2,2,1.,1.,.5,.5)}))
    output = tmp_path/'published'
    result = module.apply_revision(source, output, revision)
    assert (output/'preparation.json').is_file() and not (output/'INCOMPLETE.json').exists()
    assert module.sha(source/'preparation.json') == parent_hash
    assert module.sha(output/'images/frame.png') == image_hash
    assert module.sha(output/'transforms.json') == module.sha(source/'transforms.json')
    assert np.load(output/'depth/frame.npy').tolist() == [[7.,0.],[7.,7.]]
    assert 'masks/frame.png' not in result['copied_asset_hashes']
    teacher = result['teacher_provenance']['frames'][0]
    assert teacher['depth_sha256'] == module.sha(output/'depth/frame.npy')
    assert json.loads((output/'preparation.json').read_text())['block_id'] == 'published'

#!/usr/bin/env python3
"""Copy a joint dataset with conservative source-camera ego masks reprojected.

RGB, poses, world geometry, timestamps and split membership remain source bytes.
This changes only pixel eligibility and its matching LiDAR support, not optics.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbvi_gaussian.training import confined_environment
os.environ.update(confined_environment())
import cv2
import numpy as np
from orbvi_gaussian.io import new_output_dir, output_path, write_json
from orbvi_gaussian.calibration import load_calibration
from orbvi_gaussian.workflow import _rectify


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''): h.update(b)
    return h.hexdigest()


def tighten_support(old_mask, revision_mask, depth, support):
    if old_mask.shape != revision_mask.shape or old_mask.shape != depth.shape or depth.shape != support.shape:
        raise ValueError('All pixel planes must retain source dimensions')
    valid = (old_mask > 0) & (revision_mask > 0)
    known = valid & (support > 0) & np.isfinite(depth) & (depth > 0)
    return (valid.astype(np.uint8) * 255, np.where(known, depth, 0).astype(depth.dtype),
            known.astype(np.uint8) * 255)


def revision_masks(base, revision):
    if base['virtual_views'] != revision['virtual_views']:
        raise ValueError('Mask revision must not change virtual camera rays')
    views = {v['name']: v for v in base['virtual_views']}
    masks, origins, intrinsics = {}, {}, {}
    for cid, camera in base['cameras'].items():
        changed = revision['cameras'][cid]
        for key in set(camera) | set(changed):
            if key != 'source_mask_path' and camera.get(key) != changed.get(key):
                raise ValueError('Mask revision changed camera %s.%s' % (cid, key))
        raw_path = output_path(changed['source_mask_path'])
        origins[cid] = {'path': str(raw_path.relative_to(ROOT)), 'sha256': sha(raw_path)}
        placeholder = np.zeros((camera['height'], camera['width'], 3), np.uint8)
        for name, view in views.items():
            _, valid, _, K = _rectify(placeholder, changed, view)
            masks[(cid, name)] = valid
            intrinsics[(cid, name)] = K
    return masks, origins, intrinsics


def apply_revision(dataset, output, revision_path):
    dataset, output, revision_path = map(output_path, (dataset, output, revision_path))
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError('Use a separate new output directory')
    if (dataset / 'INCOMPLETE.json').exists():
        raise ValueError('Joint source is not ready')
    prep = json.loads((dataset / 'preparation.json').read_text())
    if prep.get('method') != 'real multi-block reassociation of consistent prepared sources':
        raise ValueError('Mask revision requires owned, prepared joint input; do not alter original sources')
    meta = json.loads((dataset / 'transforms.json').read_text())
    base_path = output_path(prep['calibration'])
    if sha(base_path) != prep['calibration_sha256']:
        raise ValueError('Source optical calibration changed')
    base = load_calibration(base_path, require_verified=False)
    revision = load_calibration(revision_path, require_verified=False)
    masks, origins, intrinsics = revision_masks(base, revision)
    out = new_output_dir(output)
    write_json(out / 'INCOMPLETE.json', {'status': 'mask revision in progress'})
    for p in dataset.rglob('*'):
        relative = p.relative_to(dataset)
        if p.is_symlink(): raise ValueError('Do not copy symbolic source links')
        if p.is_dir(): (out / relative).mkdir(exist_ok=True)
        elif relative not in (Path('INCOMPLETE.json'), Path('preparation.json')):
            shutil.copy2(p, out / relative)
    records, assets = [], {}
    for frame in meta['frames']:
        key = (frame['camera_id'], frame['view_id'])
        mask = masks[key]
        w,h,fx,fy,cx,cy = intrinsics[key]
        actual = np.array([frame[k] for k in ('w','h','fl_x','fl_y','cx','cy')])
        if not np.allclose(actual, [w,h,fx,fy,cx+.5,cy+.5], atol=1e-7, rtol=0):
            raise ValueError('Mask rays differ from exported NS half-pixel camera')
        def asset(name):
            path = output_path(out / frame[name])
            if out not in path.parents: raise ValueError('Frame asset escaped copied joint dataset')
            return path
        mp,dp,sp = (asset(k) for k in ('mask_path','lidar_depth_path','lidar_depth_mask_path'))
        before = {str(p.relative_to(out)): sha(p) for p in (mp,dp,sp)}
        old = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        support = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        depth = np.load(dp, allow_pickle=False)
        if old is None or support is None: raise ValueError('Source mask/support decoding failed')
        eligible, depth, support = tighten_support(old, mask, depth, support)
        if not eligible.any(): raise ValueError('Revision removed all environment pixels')
        if not cv2.imwrite(str(mp), eligible) or not cv2.imwrite(str(sp), support):
            raise RuntimeError('Cannot write revised masks')
        with dp.open('wb') as f: np.save(f, depth, allow_pickle=False)
        after = {str(p.relative_to(out)): sha(p) for p in (mp,dp,sp)}
        assets.update(after)
        records.append({'file_path': frame['file_path'], 'camera_id': frame['camera_id'],
                        'original_valid_pixels': int((old>0).sum()),
                        'environment_valid_pixels': int((eligible>0).sum()),
                        'before_sha256': before, 'after_sha256': after})
    assert sha(out / 'transforms.json') == sha(dataset / 'transforms.json')
    prep['block_id'] = out.name
    prep['ego_mask_revision'] = {'revision_config': str(revision_path.relative_to(ROOT)),
        'revision_sha256': sha(revision_path), 'raw_masks': origins, 'frames': records,
        'source_joint_dataset': str(dataset.relative_to(ROOT)),
        'calibration_pose_rgb_splits_changed': False,
        'protocol': 'White=environment eligible; source body polygons from training photos only; no RGB background filling'}
    prep['revised_asset_hashes'] = assets
    # Historical sources stay attributed to the unchanged parent preparation;
    # current destination checksums must describe the revised eligibility.
    prep['ego_mask_revision']['parent_preparation_sha256'] = sha(dataset/'preparation.json')
    for name in assets: prep.get('copied_asset_hashes', {}).pop(name, None)
    teachers = prep.get('teacher_provenance', {})
    teachers['postprocess'] = 'Ego/environment eligibility intersected; excluded pixels become unknown zero'
    for row in teachers.get('frames', []):
        fp = next((f for f in meta['frames'] if f['file_path'] == row.get('file_path')), None)
        if fp:
            row['depth_sha256'] = assets[fp['lidar_depth_path']]
            row['support_sha256'] = assets[fp['lidar_depth_mask_path']]
    prep['initialization_mask_caveat'] = 'Initializer and warm PLY retain original training-color source provenance; this revision changes training/evaluation pixel eligibility only'
    prep['export']['paths'] = {'transforms': str(out/'transforms.json'), 'point_cloud': str(out/'points3D.ply')}
    write_json(out / 'preparation.json', prep)
    (out / 'INCOMPLETE.json').unlink()
    return prep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', type=Path, default=ROOT/'configs/direct_lio_refined.ego_v2.experimental.yaml')
    args = p.parse_args()
    result = apply_revision(args.dataset, args.output, args.revision)
    print('Prepared ego-mask revision:', len(result['ego_mask_revision']['frames']), 'real frames')


if __name__ == '__main__': main()

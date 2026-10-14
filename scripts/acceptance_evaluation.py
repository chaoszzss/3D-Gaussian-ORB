"""Fixed held-out selection and three pixel eligibility protocols for v2."""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def acceptance_selection(path, block_id, split, actual_images, by_image, transforms):
    from build_acceptance_plan import load_plan, validate_frame_source
    plan = load_plan(path, verify_assets=False)
    if plan.get('split', 'dev') != split:
        raise ValueError('Acceptance split differs from requested evaluation')
    frames = plan['frames']
    keys = [(str(f['group_id']), f['camera_id']) for f in frames]
    if len(keys) != len(set(keys)):
        raise ValueError('Acceptance plan contains duplicate group/camera keys')
    wanted = [f for f in frames if f['block_id'] == block_id]
    if not wanted:
        raise ValueError('No acceptance frames for this block')
    lookup = {str(p.resolve()): i for i, p in enumerate(actual_images)}
    selected = {}
    transforms_sha = file_sha256(transforms)
    for row in wanted:
        if Path(row['transforms']).resolve() != Path(transforms).resolve() or row['transforms_sha256'] != transforms_sha:
            raise ValueError('Acceptance transforms changed or wrong source dataset')
        index = lookup.get(str(Path(row['source_image_path']).resolve()))
        if index is None:
            raise ValueError('Acceptance image missing from actual split dataset')
        frame = by_image[actual_images[index]]
        validate_frame_source(row, frame, index)
        for key, digest in row['asset_sha256'].items():
            if file_sha256(row[key]) != digest:
                raise ValueError('Frozen acceptance asset changed: ' + key)
        if (str(frame['group_id']), frame['camera_id'], frame['file_path']) != (str(row['group_id']), row['camera_id'], row['file_path']):
            raise ValueError('Acceptance source identity mismatch')
        if index in selected:
            raise ValueError('Duplicate loaded acceptance index')
        selected[index] = row
    return selected, {'path': str(Path(path).resolve()), 'sha256': file_sha256(path),
                      'plan_id': plan.get('plan_id'), 'block_id': block_id,
                      'canonical_frames': len(frames), 'block_frames': len(wanted),
                      'primary_protocol': 'ego_v2'}


def protocol_masks(legacy, row, cv2):
    mask = cv2.imread(str(row['mask_path']), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != legacy.shape:
        raise ValueError('Acceptance ego mask missing or wrong dimensions')
    expected = row.get('asset_sha256', {}).get('mask_path')
    if expected is not None and file_sha256(row['mask_path']) != expected:
        raise ValueError('Acceptance ego mask changed')
    masks = {'raw': np.ones_like(legacy, dtype=bool), 'legacy': legacy,
             'ego_v2': legacy & (mask > 0)}
    if not masks['ego_v2'].any():
        raise ValueError('Empty acceptance environment mask')
    return masks


def summarize_protocols(records):
    result = {}
    for protocol in ('raw', 'legacy', 'ego_v2'):
        rows = [r['protocols'][protocol] for r in records]
        pixels = sum(r['rgb_valid_pixels'] for r in rows)
        ssim_pixels = sum(r['ssim_compared_pixels'] for r in rows)
        depth_pixels = sum(r.get('map_depth_eligible_pixels', 0) for r in rows)
        result[protocol] = {
            'images': len(rows), 'psnr_db_mean': float(np.mean([r['psnr_db'] for r in rows])),
            'rgb_valid_pixels': pixels,
            'rgb_mse_pixel_mean': sum(r['rgb_mse'] * r['rgb_valid_pixels'] for r in rows) / pixels,
            'ssim_valid_mean': float(np.mean([r['ssim_valid_pixels'] for r in rows if r['ssim_valid_pixels'] is not None])),
            'ssim_pixel_mean': sum(r['ssim_valid_pixels'] * r['ssim_compared_pixels'] for r in rows if r['ssim_valid_pixels'] is not None) / ssim_pixels if ssim_pixels else None,
            'alpha_gt_0_5_valid_pixel_fraction': sum(r['alpha_gt_0_5_valid_pixels'] for r in rows) / pixels,
            'map_depth_eligible_pixels': depth_pixels,
            'map_depth_coverage_fraction': sum(r.get('map_depth_compared_pixels', 0) for r in rows) / depth_pixels if depth_pixels else None,
        }
    return result

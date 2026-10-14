"""Descriptive, fixed Sobel-magnitude SSIM for the saved full-resolution pilot."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / 'runs/fidelity_v2' / n / 'metrics.json' for n in (
    'original015_resolution_depthv3', 'train_1280_depthv3_block015_resolution')]
reports = [json.loads(p.read_text()) for p in PATHS]
assert reports[0]['acceptance_plan_id'] == reports[1]['acceptance_plan_id']
assert len(reports[0]['frames']) == len(reports[1]['frames']) == 20


def gradient(rgb):
    # Fixed luminance and theoretical Sobel bound; no per-image normalization.
    gray = np.clip(rgb, 0, 1).astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.25, borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.25, borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(dx, dy) / np.float32(np.sqrt(2))


rows = []
for index, (a, b) in enumerate(zip(reports[0]['frames'], reports[1]['frames'])):
    for key in ('group_id', 'camera_id', 'source_image_path', 'source_mask_path', 'acceptance_mask_path'):
        assert a[key] == b[key], key
    gt = cv2.cvtColor(cv2.imread(a['source_image_path']), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    valid = (cv2.imread(a['source_mask_path'], 0) > 0) & (cv2.imread(a['acceptance_mask_path'], 0) > 0)
    # 3x3 Sobel followed by 7x7 SSIM needs a 9x9 valid support window.
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    scores = []
    ggt = gradient(gt)
    for path in PATHS:
        with np.load(path.parent / f'{index:03d}_float.npz', allow_pickle=False) as data:
            pred = data['rgb']
            assert pred.shape == gt.shape and np.isfinite(pred).all()
            _, score = structural_similarity(ggt, gradient(pred), data_range=1., win_size=7, full=True)
        scores.append(float(score[valid].mean()))
    rows.append(dict(index=index, group_id=a['group_id'], camera_id=a['camera_id'], pixels=int(valid.sum()),
        baseline=scores[0], candidate=scores[1], difference=scores[1] - scores[0]))
result = dict(status='measured', scope='block015 fixed20 full1280 DEV; descriptive new protocol',
    acceptance_plan_id=reports[0]['acceptance_plan_id'], high_fidelity_accepted=False,
    protocol='sRGB numeric RGB clipped0..1; Rec709 coefficients; Sobel3/4 magnitude/sqrt2; SSIM7 data_range1; ego mask eroded9 zero border; frame mean',
    protocol_limit='Original plan said gradient SSIM without a formula; this definition was made after RGB results, so it is not a retroactively frozen acceptance test.',
    baseline_mean=float(np.mean([r['baseline'] for r in rows])),
    candidate_mean=float(np.mean([r['candidate'] for r in rows])),
    paired_delta=float(np.mean([r['difference'] for r in rows])),
    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in PATHS],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), frames=rows)
out = ROOT / 'reports/resolution_gradient_ssim_015_descriptive.json'
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(result, indent=2) + '\n')
print({k: result[k] for k in ('baseline_mean', 'candidate_mean', 'paired_delta')})

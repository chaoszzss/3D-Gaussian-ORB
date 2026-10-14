"""Read-only CPU loss units audit; does not measure Gaussian parameter gradients."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from pytorch_msssim import SSIM
from orbvi_gaussian.lidar_splatfacto import compute_lidar_losses
from orbvi_gaussian.surfels import project_surfel_depth, SurfelDepthPolicy

torch.set_num_threads(4)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def masks(path):
    return np.asarray(Image.open(path).convert('L')) > 0


def camera(frame):
    pose = np.asarray(frame['transform_matrix']) @ np.diag([1., -1., -1., 1.])
    K = np.array([[frame['fl_x'], 0, frame['cx'] - .5],
                  [0, frame['fl_y'], frame['cy'] - .5], [0, 0, 1.]])
    return pose, K


def actual_loss(depth, alpha, target, support):
    depth = torch.as_tensor(depth).squeeze().clone().requires_grad_()
    alpha = torch.as_tensor(alpha).squeeze().clone().requires_grad_()
    target, valid = torch.as_tensor(target), torch.as_tensor(support)
    terms = compute_lidar_losses(dict(depth=depth, accumulation=alpha), target, valid, .5, .1)
    depth_grad, alpha_grad = torch.autograd.grad(sum(terms.values()), (depth, alpha), retain_graph=True)
    delta = (depth[valid] - target[valid])
    metric = .05 * F.huber_loss(delta, torch.zeros_like(delta), delta=.15)
    metric_grad, = torch.autograd.grad(metric, (depth,))
    error = delta.detach().numpy()
    n = int(valid.sum())
    return dict(support_pixels=n, **{k: float(v.detach()) for k, v in terms.items()},
                proposed_additive_metric_huber_weight_0_05_delta_m_0_15=float(metric.detach()),
                output_depth_gradient_l1=float(depth_grad.abs().sum()),
                output_depth_gradient_l2=float(depth_grad.norm()),
                output_alpha_gradient_l1=float(alpha_grad.abs().sum()),
                proposed_metric_output_depth_gradient_l1=float(metric_grad.abs().sum()),
                relative_quadratic_fraction=float((delta.detach().abs() < .1 * target[valid]).float().mean()),
                depth_abs_error_median_m=float(np.median(np.abs(error))),
                depth_wrong_over_0_15m_fraction=float((np.abs(error) > .15).mean()),
                depth_too_near_over_0_15m_fraction=float((error < -.15).mean()),
                support_alpha_mean=float(alpha[valid].detach().mean()),
                support_target_median_m=float(target[valid].median()))


def nearest_labels(reference, frame):
    pose, K = camera(frame)
    p = (reference['xyz'].astype(float) - pose[:3, 3]) @ pose[:3, :3]
    ids = np.flatnonzero(np.isfinite(p).all(1) & (p[:, 2] > .1))
    uv = p[ids] @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    w, h = frame['w'], frame['h']
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= -.5) & (uv[:, 0] < w-.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5)
    ids, uv = ids[inside], uv[inside]
    pix = np.floor(uv + .5).astype(int)
    flat = pix[:, 1] * w + pix[:, 0]
    order = np.lexsort((ids, p[ids, 2], flat))
    flat, ids = flat[order], ids[order]
    first = np.r_[True, np.diff(flat) != 0]
    flat, ids = flat[first], ids[first]
    normal = reference['normals'][ids]
    length = np.linalg.norm(normal, axis=1)
    dz = reference['xyz'][ids, 2] - pose[2, 3]
    floor = reference['normal_valid'][ids].astype(bool) & (length > 1e-8)
    floor &= np.abs(normal[:, 2]) / np.maximum(length, 1e-8) >= np.cos(np.deg2rad(30))
    floor &= (dz <= -.3) & (dz >= -3.)
    image = np.zeros((h, w), bool)
    image.ravel()[flat[floor]] = True
    return image


def main():
    root = ROOT / 'data/garage_fidelity_640_v2/block_015'
    meta = json.loads((root / 'transforms.json').read_text())
    by_file = {f['file_path']: f for f in meta['frames']}
    reference = dict(np.load(root / 'lidar_reference.npz', allow_pickle=False))
    rendered = ROOT / 'runs/fidelity_v2/train_640_block015_acceptance'
    metrics = json.loads((rendered / 'metrics.json').read_text())
    samples = []
    for z in (3., 10., 25.):
        for error in (.15, 1., 5.):
            pred = torch.tensor([[z+error]], dtype=torch.float64, requires_grad=True)
            target = torch.tensor([[z]], dtype=torch.float64)
            loss = compute_lidar_losses(dict(depth=pred, accumulation=torch.ones_like(pred)),
                                       target, torch.ones_like(pred, dtype=torch.bool), .5, .1)['lidar_depth_loss']
            grad, = torch.autograd.grad(loss, (pred,))
            candidate = .05 * F.huber_loss(pred-target, torch.zeros_like(pred), delta=.15)
            candidate_grad, = torch.autograd.grad(candidate, (pred,))
            samples.append(dict(target_m=z, error_m=error, relative_loss=float(loss.detach()),
                                derivative_wrt_rendered_depth=float(grad),
                                metric_0_05_loss=float(candidate.detach()), metric_0_05_derivative=float(candidate_grad)))
    records = []
    ssim = SSIM(data_range=1., size_average=True, channel=3)
    for i, row in enumerate(metrics['frames']):
        frame = by_file[row['file_path']]
        assert frame['file_path'] in meta['val_filenames'] and frame['file_path'] not in meta['train_filenames']
        image_mask = masks(root / frame['mask_path'])
        target_old = np.load(root / frame['lidar_depth_path'], allow_pickle=False)
        valid_old = masks(root / frame['lidar_depth_mask_path']) & image_mask & (target_old > 0)
        pose, K = camera(frame)
        target_v3, valid_v3 = project_surfel_depth(
            reference['xyz'], reference['normals'], reference['normal_valid'], pose, K,
            frame['w'], frame['h'], policy=SurfelDepthPolicy.trusted_ground_v3())
        valid_v3 &= image_mask
        blob = np.load(rendered / f'{i:03d}_float.npz')
        gt = torch.from_numpy(np.asarray(Image.open(root / frame['file_path']).convert('RGB')).copy()).float()/255
        pred = torch.from_numpy(blob['rgb']).clone().requires_grad_()
        mask = torch.from_numpy(image_mask)[..., None]
        l1 = ((pred-gt)*mask).abs().mean()
        similarity = 1-ssim((pred*mask).permute(2,0,1)[None], (gt*mask).permute(2,0,1)[None])
        main_loss = .8*l1+.2*similarity
        rgb_grad, = torch.autograd.grad(main_loss, (pred,))
        rec = dict(file_path=frame['file_path'], split='DEV', rgb_main_loss=float(main_loss.detach()),
                   rgb_output_gradient_l1=float(rgb_grad.abs().sum()),
                   legacy_dataset_v2_teacher=actual_loss(blob['depth'], blob['alpha'], target_old, valid_old),
                   strict_v3_teacher=actual_loss(blob['depth'], blob['alpha'], target_v3, valid_v3),
                   float_sha256=sha(rendered / f'{i:03d}_float.npz'))
        records.append(rec)
        print('measured DEV', i, flush=True)
    # Independent TRAIN-only selection, before any candidate render or gradient.
    train = sorted(meta['train_filenames'])
    probes = [train[i] for i in np.linspace(0, len(train)-1, 28).astype(int)]
    ranked = []
    for name in probes:
        frame = by_file[name]
        floor = nearest_labels(reference, frame)
        valid = masks(root/frame['lidar_depth_mask_path']) & masks(root/frame['mask_path'])
        ranked.append(dict(file_path=name, camera_id=frame['camera_id'], group_id=frame['group_id'],
                           supported_floor_pixels=int((floor & valid).sum()),
                           raw_nearest_floor_pixels=int(floor.sum())))
    selected, used_groups = [], set()
    for item in sorted(ranked, key=lambda r: (-r['supported_floor_pixels'], r['file_path'])):
        if item['group_id'] not in used_groups:
            selected.append(item); used_groups.add(item['group_id'])
        if len(selected)==3:
            break
    source_files = [Path(__file__), ROOT/'src/orbvi_gaussian/lidar_splatfacto.py',
                    ROOT/'src/orbvi_gaussian/surfels.py', root/'transforms.json', root/'lidar_reference.npz',
                    rendered/'metrics.json']
    report = dict(status='CPU_OUTPUT_LOSS_MEASURED_PARAMETER_GRADIENT_PENDING_GPU_LOCK', gpu_used=False,
                  scope='DEV float output gradients are not Gaussian parameter gradients. TRAIN frame selection uses raw measured floor support only; source-conditioned geometry is not independent ground truth.',
                  unit_samples=samples, dev_records=records, selected_train_frames=selected,
                  train_selection=dict(candidate_count=len(train), deterministic_probes=28,
                    criterion='highest known nearest-floor support; 3 distinct TRAIN groups; no render-dependent selection'),
                  summary={arm: {key:float(np.mean([r[arm][key] for r in records]))
                      for key in ('lidar_depth_loss','lidar_alpha_loss','output_depth_gradient_l1',
                                  'proposed_additive_metric_huber_weight_0_05_delta_m_0_15',
                                  'proposed_metric_output_depth_gradient_l1','support_alpha_mean',
                                  'depth_wrong_over_0_15m_fraction','depth_too_near_over_0_15m_fraction')}
                      for arm in ('legacy_dataset_v2_teacher','strict_v3_teacher')},
                  rgb_main_loss_mean=float(np.mean([r['rgb_main_loss'] for r in records])),
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
                  commands=["CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-train/bin/python runtime/lidar_loss_units_v3_audit.py"],
                  proposed_ablation=dict(enabled_by_default=False, formula='existing relative .5/alpha .1 plus weight*mean(Huber(D-Z,delta_m=.15))',
                    first_weight=.05, second_weight_only_after_grad_audit=.5,
                    delta_reason='matches fixed 0.15m evaluation tolerance; no claim of LIO absolute accuracy',
                    comparisons='same strict v3 teacher/original warm/init/seed/screen schedule; metrics and opacity reported without changing priors'))
    out = ROOT/'reports/lidar_loss_units_v3_cpu.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(out), summary=report['summary'], selected_train_frames=selected)))


if __name__=='__main__':
    main()

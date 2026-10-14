#!/usr/bin/env python3
"""Read-only depth-interface sidecar; writes artifacts only under runtime.

No training/model setup, no dependency installation, <=50k splats / <=64px GPU
render, <=400k reference points, <=12 sampled training images, two CPU threads.
For old checkpoints, explicitly select their principal-point convention. Their
depth is compared ONLY to a newly projected map on the historical render rays,
never to the corrected dataset's saved depth raster.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from importlib.metadata import version

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
for key, value in {
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2", "MAX_JOBS": "2",
    "PATH": str(ROOT / '.venv-train/bin') + os.pathsep + os.environ.get('PATH', ''),
    "TMPDIR": str(RUNTIME / "tmp"), "TMP": str(RUNTIME / "tmp"),
    "TEMP": str(RUNTIME / "tmp"), "XDG_CACHE_HOME": str(RUNTIME / "cache/xdg"),
    "TORCH_HOME": str(RUNTIME / "cache/torch"),
    "TORCH_EXTENSIONS_DIR": str(RUNTIME / "cache/torch_extensions"),
    "TORCHINDUCTOR_CACHE_DIR": str(RUNTIME / "cache/torchinductor"),
    "TRITON_CACHE_DIR": str(RUNTIME / "cache/triton"),
    "CUDA_CACHE_PATH": str(RUNTIME / "cache/cuda"),
    "HF_HOME": str(RUNTIME / "cache/huggingface"),
    "MPLCONFIGDIR": str(RUNTIME / "cache/matplotlib"),
    "NUMBA_CACHE_DIR": str(RUNTIME / "cache/numba"),
    "WANDB_DIR": str(RUNTIME / "cache/wandb"),
    "WANDB_CONFIG_DIR": str(RUNTIME / "cache/wandb_config"),
}.items():
    os.environ[key] = value
(RUNTIME / "tmp").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from orbvi_gaussian.geometry import opencv_c2w_to_opengl
from orbvi_gaussian.points import project_depth
from orbvi_gaussian.surfels import project_surfel_depth

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
cv2.setNumThreads(2)


def stats(values):
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {"count": len(x), "median": float(np.median(x)),
            "median_abs": float(np.median(np.abs(x))),
            "p95_abs": float(np.percentile(np.abs(x), 95)),
            "max_abs": float(np.max(np.abs(x))),
            "rmse": float(np.sqrt(np.mean(x * x)))}


def read_cv_intrinsics(data):
    cameras = {}
    for line in (data / "sparse/0/cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            cid, model, w, h, fx, fy, cx, cy = line.split()
            assert model == "PINHOLE"
            cameras[int(cid)] = np.array([[float(fx), 0, float(cx)],
                                          [0, float(fy), float(cy)], [0, 0, 1.]])
    return cameras


def frame_K(frame, cv_cameras):
    matches = [k for k in cv_cameras.values()
               if abs(k[0, 0] - frame['fl_x']) < 1e-7
               and abs(k[1, 1] - frame['fl_y']) < 1e-7
               and min(abs(k[0, 2] - frame['cx']), abs(k[0, 2] + .5 - frame['cx'])) < 1e-7
               and min(abs(k[1, 2] - frame['cy']), abs(k[1, 2] + .5 - frame['cy'])) < 1e-7]
    assert len(matches) == 1, "ambiguous original CV intrinsics"
    return matches[0].copy()


def loader_audit(data, meta):
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from orbvi_gaussian.lidar_splatfacto import LidarFullImageDatamanagerConfig, resize_lidar_targets
    parser = NerfstudioDataParserConfig(data=data, auto_scale_poses=False,
        center_method="none", orientation_method="none", scale_factor=1.,
        downscale_factor=1, load_3D_points=False)
    cfg = LidarFullImageDatamanagerConfig(data=data, dataparser=parser,
        cache_images="cpu", cache_images_type="uint8", lidar_depth_cache_max_bytes=0)
    with contextlib.redirect_stdout(io.StringIO()):
        dm = cfg.setup(device="cpu")
        dm._initialize_lidar()
    lookup = {(data / f['file_path']).resolve(): f for f in meta['frames']}
    cv_cameras = read_cv_intrinsics(data)
    mismatches = []
    for i, filename in enumerate(dm.train_dataset.image_filenames):
        f = lookup[filename.resolve()]
        if (dm._lidar_frames[i][0] != (data / f['lidar_depth_path']).resolve()
            or dm._lidar_frames[i][1] != (data / f['lidar_depth_mask_path']).resolve()):
            mismatches.append(i)
        np.testing.assert_allclose(dm.train_dataset.cameras.camera_to_worlds[i].numpy(),
                                   np.array(f['transform_matrix'])[:3], atol=3e-6)
    # Exercise the actual next_train chain, while bypassing ONLY its whole-dataset
    # memory cache. Zero distortion means _load_images returns get_data unchanged.
    ids = sorted(set([0, 1, 2, 3, len(dm.train_dataset)//2,
                      len(dm.train_dataset)//2+1, len(dm.train_dataset)-4,
                      len(dm.train_dataset)-3, len(dm.train_dataset)-2,
                      len(dm.train_dataset)-1]))
    dm.__dict__['cached_train'] = {
        i: dm.train_dataset.get_data(i, image_type="uint8") for i in ids}
    dm.train_cameras = dm.train_dataset.cameras
    dm.train_unseen_cameras = ids.copy()
    rows = []
    for expected_i in ids:
        camera, batch = dm.next_train(0)
        i = int(batch['image_idx'])
        assert i == expected_i and camera.metadata['cam_idx'] == i
        f = lookup[dm.train_dataset.image_filenames[i].resolve()]
        image = np.array(Image.open(data / f['file_path']))
        depth = np.load(data / f['lidar_depth_path'], allow_pickle=False)
        support = np.array(Image.open(data / f['lidar_depth_mask_path'])) > 0
        mask = np.array(Image.open(data / f['mask_path'])) > 0
        assert np.array_equal(image, batch['image'].numpy())
        assert np.array_equal(mask, batch['mask'].numpy()[..., 0])
        expected = support & mask & np.isfinite(depth) & (depth > 0)
        assert np.array_equal(expected, batch['lidar_depth_mask'].numpy()[..., 0])
        np.testing.assert_array_equal(batch['lidar_depth'].numpy()[..., 0], np.where(expected, depth, 0))
        k = frame_K(f, cv_cameras)
        target2, valid2 = resize_lidar_targets(batch['lidar_depth'], batch['lidar_depth_mask'],
                                              (f['h']//2, f['w']//2))
        assert np.array_equal(valid2.numpy()[..., 0], expected[::2, ::2])
        rows.append({"idx": i, "file": f['file_path'], "support": int(expected.sum()),
            "support_after_2x": int(valid2.sum()),
            "NS_minus_CV_cxy": [f['cx']-k[0, 2], f['cy']-k[1, 2]]})
    count_nonpositive_valid = count_positive_invalid = 0
    for f in meta['frames']:  # Sequential streaming, never retain all rasters.
        d = np.load(data / f['lidar_depth_path'], allow_pickle=False)
        m = np.array(Image.open(data / f['lidar_depth_mask_path'])) > 0
        count_positive_invalid += int(np.sum((d > 0) & ~m))
        count_nonpositive_valid += int(np.sum(m & (~np.isfinite(d) | (d <= 0))))
    return {"train": len(dm.train_dataset), "dev": len(dm.eval_dataset),
            "test_declared": len(meta['test_filenames']), "frame_mapping_mismatches": mismatches,
            "dataparser_scale": dm.train_dataparser_outputs.dataparser_scale,
            "dataparser_transform": dm.train_dataparser_outputs.dataparser_transform.tolist(),
            "distortion_all_zero": bool(torch.all(dm.train_cameras.distortion_params == 0)),
            "positive_depth_outside_support": count_positive_invalid,
            "invalid_depth_inside_support": count_nonpositive_valid, "sampled_next_train": rows}


def synthetic():
    from gsplat.rendering import rasterization, rasterization_2dgs
    from nerfstudio.models.splatfacto import get_viewmat, resize_image, SplatfactoModel
    from orbvi_gaussian.lidar_splatfacto import resize_lidar_targets
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    torch.cuda.reset_peak_memory_stats()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [17., -21., 33.], degrees=True).as_matrix()
    pose[:3, 3] = [2., -3., .8]
    # Algorithm fixture only: these are NOT estimated real-sensor calibrations.
    K = np.array([[48., 0, 31.5], [0, 48., 31.5], [0, 0, 1.]])
    Kns = K.copy(); Kns[:2, 2] += .5
    gl = torch.as_tensor(opencv_c2w_to_opengl(pose)[None, :3], device='cuda', dtype=torch.float32)
    view = get_viewmat(gl)
    view_error = float(np.max(np.abs(view.cpu().numpy()[0] - np.linalg.inv(pose))))
    yy, xx = np.mgrid[:64, :64]
    rows = []
    for name, normal in [('front', [0., 0., 1.]), ('oblique', [.45, -.2, 1.])]:
        n = np.array(normal); n /= np.linalg.norm(n)
        ray = np.stack([(xx-K[0, 2])/48, (yy-K[1, 2])/48, np.ones_like(xx)], -1)
        analytic = 5*n[2] / (ray @ n)
        p = (ray * analytic[..., None]).reshape(-1, 3)
        xyz = p @ pose[:3, :3].T + pose[:3, 3]
        normals = np.tile(n @ pose[:3, :3].T, (len(p), 1))
        gt, valid = project_surfel_depth(xyz, normals, np.ones(len(p), bool), pose, K, 64, 64)
        raw, raw_valid = project_depth(xyz, pose, K, 64, 64)
        guide = np.array([0., 1., 0.]); tangent = np.cross(guide, n); tangent /= np.linalg.norm(tangent)
        rotation = pose[:3, :3] @ np.column_stack([tangent, np.cross(n, tangent), n])
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].copy()
        args = dict(means=torch.tensor(xyz, device='cuda', dtype=torch.float32),
            quats=torch.tensor(np.tile(quat, (len(p), 1)), device='cuda', dtype=torch.float32),
            scales=torch.tensor([.075, .075, .004], device='cuda').expand(len(p), -1).contiguous(),
            opacities=torch.full((len(p),), .35, device='cuda'),
            colors=torch.full((len(p), 3), .6, device='cuda'), viewmats=view,
            Ks=torch.tensor(Kns[None], device='cuda', dtype=torch.float32),
            width=64, height=64, packed=False, near_plane=.01, far_plane=1e10)
        with torch.inference_mode():
            ed, alpha, info = rasterization(**args, render_mode='RGB+ED')
            accum, alpha2, _ = rasterization(**args, render_mode='RGB+D')
            two = rasterization_2dgs(**args, render_mode='RGB+ED')
        pred = ed[0, ..., 3].cpu().numpy(); a = alpha[0, ..., 0].cpu().numpy()
        interior = valid & (a > .1); interior[:4] = False; interior[-4:] = False
        interior[:, :4] = False; interior[:, -4:] = False
        norm_error = float(torch.max(torch.abs(ed[..., 3:4] - accum[..., 3:4]/alpha.clamp_min(1e-10))))
        target2, v2 = resize_lidar_targets(torch.tensor(gt), torch.tensor(valid), (32, 32))
        y2, x2 = np.mgrid[:32, :32]
        ray2 = np.stack([(2*x2+.5-K[0, 2])/48, (2*y2+.5-K[1, 2])/48, np.ones_like(x2)], -1)
        analytic2 = 5*n[2]/(ray2@n)
        rows.append({"plane": name, "splats": len(p), "matrix_inverse_max_error": view_error,
            "GT_vs_analytic": stats((gt-analytic)[valid]),
            "raw_vs_analytic": stats((raw-analytic)[raw_valid]),
            "ED_vs_surfel_GT": stats((pred-gt)[interior]),
            "alpha_range_compared": [float(a[interior].min()), float(a[interior].max())],
            "ED_equals_D_div_alpha_max_error": norm_error,
            "2DGS_tuple_length": len(two),
            "2DGS_ED_vs_surfel_GT": stats((two[0][0, ..., 3].cpu().numpy()-gt)[interior&(two[1][0, ..., 0].cpu().numpy()>.1)]),
            "2DGS_meta_means2d_shape": list(two[-1]['means2d'].shape),
            "metadata_z_vs_cv_z": stats(info['depths'].cpu().numpy().ravel()-p[:, 2]),
            "initial_2x_nearest_GT_vs_actual_render_ray": stats((target2.numpy()[..., 0]-analytic2)[v2.numpy()[..., 0]])})
    # Exercise parent loss itself on uint8 image / bool mask, using a tiny shell
    # model (no population, strategies, optimizers or training). Exact GT should
    # give zero RGB loss at each active resolution.
    from types import SimpleNamespace
    shell = SplatfactoModel.__new__(SplatfactoModel); torch.nn.Module.__init__(shell)
    shell.device_indicator_param = torch.nn.Parameter(torch.empty(0))
    shell.gauss_params = torch.nn.ParameterDict({'means': torch.nn.Parameter(torch.zeros(1, 3))})
    shell.config = SimpleNamespace(num_downscales=1, resolution_schedule=1500,
        use_scale_regularization=False, ssim_lambda=.2, use_bilateral_grid=False)
    shell.camera_optimizer = SimpleNamespace(get_loss_dict=lambda x: None)
    shell.ssim = StructuralSimilarityIndexMeasure(data_range=1.)
    image = (torch.arange(32*32*3).reshape(32, 32, 3) % 255).to(torch.uint8)
    mask = torch.ones(32, 32, 1, dtype=torch.bool); mask[:3] = False
    losses = []
    for step in [0, 1500]:
        shell.step = step
        gt = shell.get_gt_img(image)
        out = {'rgb': gt.clone(), 'background': torch.zeros(3)}
        loss = shell.get_loss_dict(out, {'image': image, 'mask': mask})
        losses.append({'step': step, 'shape': list(gt.shape),
                       'main_loss': float(loss['main_loss']),
                       'image_unchanged': bool(torch.equal(image, (torch.arange(32*32*3).reshape(32, 32, 3)%255).to(torch.uint8))),
                       'outputs_rgb_unchanged': bool(torch.equal(out['rgb'], gt))})
    # Two visible layers can match a uniform RGB perfectly and differ by metres
    # from the first measured surface, even with correct alpha normalization.
    layer_means = torch.tensor([[5*.5/48, 5*.5/48, 5.],
                                [15*.5/48, 15*.5/48, 15.]], device='cuda')
    with torch.inference_mode():
        layer, la, _ = rasterization(means=layer_means,
            quats=torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]], device='cuda'),
            scales=torch.tensor([[.2, .2, .01], [.6, .6, .01]], device='cuda'),
            opacities=torch.tensor([.4, .4], device='cuda'),
            colors=torch.full((2, 3), .6, device='cuda'),
            viewmats=torch.eye(4, device='cuda')[None],
            Ks=torch.tensor(Kns[None], device='cuda', dtype=torch.float32),
            width=64, height=64, packed=False, render_mode='RGB+ED')
    mixed = {'front_z_m': 5., 'back_z_m': 15.,
        'front_and_back_alpha': [.4, .4], 'accumulation_at_center': float(la[0, 32, 32, 0]),
        'expected_z_at_center_m': float(layer[0, 32, 32, 3]),
        'analytic_expected_z_m': (.4*5 + .6*.4*15)/(.4+.6*.4),
        'RGB_error_with_matching_06_background': float(torch.max(torch.abs(layer[0, ..., :3]+(1-la[0])*.6-.6)))}
    # One large tilted surfel discriminates ray-plane depth from a constant
    # center-z channel, even if dense small surfels happen to approximate a plane.
    n = np.array([.8, 0., .6]); tangent = np.cross([0., 1., 0.], n)
    quat = Rotation.from_matrix(np.column_stack([tangent, np.cross(n, tangent), n])).as_quat()[[3, 0, 1, 2]].copy()
    with torch.inference_mode():
        single = rasterization_2dgs(means=torch.tensor([[0., 0., 5.]], device='cuda'),
            quats=torch.tensor(quat[None], device='cuda', dtype=torch.float32),
            scales=torch.tensor([[2., 2., .01]], device='cuda'),
            opacities=torch.tensor([.8], device='cuda'), colors=torch.full((1, 3), .6, device='cuda'),
            viewmats=torch.eye(4, device='cuda')[None], Ks=torch.tensor(Kns[None], device='cuda', dtype=torch.float32),
            width=64, height=64, packed=False, render_mode='RGB+ED')
    one_z = single[0][0, ..., 3].cpu().numpy()
    one_a = single[1][0, ..., 0].cpu().numpy()
    ray = np.stack([(xx-K[0, 2])/48, (yy-K[1, 2])/48, np.ones_like(xx)], -1)
    plane_z = 5*n[2]/(ray@n)
    mask = (one_a>.1)&(plane_z>0)&(plane_z<30)
    single_check = {'tuple_length': len(single), 'compared_pixels': int(mask.sum()),
        'ED_range_m': [float(one_z[mask].min()), float(one_z[mask].max())],
        'plane_ray_hit_range_m': [float(plane_z[mask].min()), float(plane_z[mask].max())],
        'ED_minus_ray_plane_hit': stats((one_z-plane_z)[mask])}
    # Verify the strategy gradient interface with one actual backward. 2DGS's
    # densification proxy is gradient_2dgs; means2d need not carry absgrad.
    grad_args = dict(means=torch.tensor([[0., 0., 5.]], device='cuda', requires_grad=True),
        quats=torch.tensor(quat[None], device='cuda', dtype=torch.float32, requires_grad=True),
        scales=torch.tensor([[2., 2., .01]], device='cuda', requires_grad=True),
        opacities=torch.tensor([.8], device='cuda', requires_grad=True),
        colors=torch.full((1, 3), .6, device='cuda', requires_grad=True),
        viewmats=torch.eye(4, device='cuda')[None], Ks=torch.tensor(Kns[None], device='cuda', dtype=torch.float32),
        width=64, height=64, packed=False, render_mode='RGB+ED', absgrad=True)
    back = rasterization_2dgs(**grad_args)
    back[-1]['means2d'].retain_grad(); back[-1]['gradient_2dgs'].retain_grad()
    (back[0][0, :, 20:40, :3].sum()+back[0][0, :, 20:40, 3].sum()*.01).backward()
    single_check['backward'] = {key: {'grad_present': back[-1][key].grad is not None,
        'absgrad_present': hasattr(back[-1][key], 'absgrad'),
        'absgrad_finite': bool(torch.isfinite(back[-1][key].absgrad).all()) if hasattr(back[-1][key], 'absgrad') else None}
        for key in ['means2d', 'gradient_2dgs']}
    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(absgrad=True)
    strategy_state = strategy.initialize_state(scene_scale=1.)
    strategy._update_state({'means': grad_args['means']}, strategy_state, back[-1], packed=False)
    single_check['DefaultStrategy_absgrad_means2d_update'] = {
        'passed': True, 'grad2d_finite': bool(torch.isfinite(strategy_state['grad2d']).all())}
    return {'planes': rows, 'two_layer_RGB_depth_counterexample': mixed,
            '2DGS_single_tilted_surfel': single_check,
            'parent_actual_loss_perfect_GT': losses,
            'gpu_peak_bytes': torch.cuda.max_memory_allocated()}


def real_frame(data, meta, file_path, checkpoint, convention):
    frame = next(f for f in meta['frames'] if f['file_path'] == file_path)
    K = frame_K(frame, read_cv_intrinsics(data))
    pose = np.array(frame['transform_matrix']) @ np.diag([1., -1., -1., 1.])
    # Check the unchanged source pose against saved COLMAP w2c as an additional
    # provenance check before rendering a legacy checkpoint.
    source_w2c = None
    for line in (data/'sparse/0/images.txt').read_text().splitlines():
        fields = line.split()
        if len(fields)==10 and not line.startswith('#') and fields[-1]==Path(file_path).name:
            wxyz = np.array(fields[1:5], float)
            source_w2c = np.eye(4)
            source_w2c[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
            source_w2c[:3, 3] = np.array(fields[5:8], float)
            break
    assert source_w2c is not None
    pose_provenance_error = float(np.max(np.abs(source_w2c-np.linalg.inv(pose))))
    assert pose_provenance_error<1e-7
    with np.load(data / 'lidar_reference.npz', allow_pickle=False) as ref:
        xyz, normals, normal_valid = ref['xyz'], ref['normals'], ref['normal_valid']
    assert len(xyz) <= 400000
    saved = np.load(data / frame['lidar_depth_path'], allow_pickle=False)
    valid_saved = np.array(Image.open(data / frame['lidar_depth_mask_path'])) > 0
    fresh, valid_fresh = project_surfel_depth(xyz, normals, normal_valid, pose, K, frame['w'], frame['h'])
    raw, valid_raw = project_depth(xyz, pose, K, frame['w'], frame['h'])
    image_mask = np.array(Image.open(data / frame['mask_path'])) > 0
    common = valid_saved & valid_fresh
    raw_common = valid_saved & valid_raw
    p = (xyz.astype(float) - pose[:3, 3]) @ pose[:3, :3]
    n = normals.astype(float) @ pose[:3, :3]
    good = normal_valid & (p[:, 2] > .1)
    p, n = p[good], n[good]
    uv = (p @ K.T); uv = uv[:, :2]/uv[:, 2:3]
    inside = (uv[:, 0] >= -.5)&(uv[:, 0]<frame['w']-.5)&(uv[:, 1]>=-.5)&(uv[:, 1]<frame['h']-.5)
    p, n, uv = p[inside], n[inside], uv[inside]
    pixels = np.floor(uv+.5)
    ray = np.column_stack([(pixels[:, 0]-K[0, 2])/K[0, 0],
                           (pixels[:, 1]-K[1, 2])/K[1, 1], np.ones(len(p))])
    denom = np.sum(ray*n, 1); cosine = np.abs(denom)/np.linalg.norm(ray, axis=1)
    z = np.divide(np.sum(p*n, 1), denom, out=np.full(len(p), np.nan), where=np.abs(denom)>1e-8)
    accepted = (cosine>.1)&np.isfinite(z)&(z>.1)&(np.abs(z-p[:, 2])<.1+.01*p[:, 2])
    ray_error = np.linalg.norm((uv-pixels)/[K[0, 0], K[1, 1]], axis=1)
    result = {'frame': file_path, 'reference_points': len(xyz),
        'source_COLMAP_vs_current_CV_w2c_max_error': pose_provenance_error,
        'fresh_surfel_vs_saved': stats((fresh-saved)[common]),
        'support_mismatches_inside_image_mask': int(np.sum((valid_fresh != valid_saved)&image_mask)),
        'raw_zbuffer_vs_saved_surfel': stats((raw-saved)[raw_common]),
        'accepted_point_pixel_ray_error_rad_approx': stats(ray_error[accepted]),
        'accepted_tangent_z_correction_m': stats((z-p[:, 2])[accepted]),
        'rejected_grazing_points': int(np.sum(cosine<=.1)),
        'accepted_points': int(accepted.sum())}
    # Identify the RAW z-buffer winner, rather than attributing changes between
    # different points/surfaces to one point's subpixel quantization.
    pa = (xyz.astype(float)-pose[:3, 3])@pose[:3, :3]
    na = normals.astype(float)@pose[:3, :3]
    use = np.flatnonzero(np.isfinite(pa).all(1)&(pa[:, 2]>0))
    ua = pa[use]@K.T; ua = ua[:, :2]/ua[:, 2:3]
    ins = (ua[:, 0]>=0)&(ua[:, 0]<frame['w'])&(ua[:, 1]>=0)&(ua[:, 1]<frame['h'])
    use, ua = use[ins], ua[ins]
    pix = np.floor(ua+.5).astype(int)
    ins = (pix[:, 0]<frame['w'])&(pix[:, 1]<frame['h'])
    use, pix = use[ins], pix[ins]
    linear = pix[:, 1]*frame['w']+pix[:, 0]
    order = np.lexsort((pa[use, 2], linear))
    first = np.r_[True, np.diff(linear[order])!=0]
    use, linear = use[order[first]], linear[order[first]]
    raw_ids = np.full(frame['h']*frame['w'], -1, int); raw_ids[linear] = use
    raw_ids = raw_ids.reshape(saved.shape)
    bad = raw_common & (np.abs(raw-saved)>1.)
    winners = raw_ids[bad]; assert np.all(winners>=0)
    yp, xp = np.nonzero(bad)
    wr = np.column_stack([(xp-K[0, 2])/K[0, 0], (yp-K[1, 2])/K[1, 1], np.ones(len(xp))])
    wd = (wr*na[winners]).sum(1)
    wz = np.divide((pa[winners]*na[winners]).sum(1), wd,
                   out=np.full(len(winners), np.nan), where=np.abs(wd)>1e-8)
    wc = np.abs(wd)/np.linalg.norm(wr, axis=1)
    wn = normal_valid[winners]
    accepted_winner = wn&(wc>.1)&np.isfinite(wz)&(wz>.1)&(np.abs(wz-pa[winners, 2])<.1+.01*pa[winners, 2])
    result['raw_vs_surfel_gt_gt1m_winner_classification'] = {
        'pixels': len(winners), 'raw_closer_than_GT': int(np.sum(raw[bad]<saved[bad])),
        'normal_invalid': int(np.sum(~wn)),
        'normal_valid_but_grazing': int(np.sum(wn&(wc<=.1))),
        'normal_valid_nongrazing_rejected_intersection': int(np.sum(wn&(wc>.1)&~accepted_winner)),
        'accepted_winner': int(accepted_winner.sum())}
    if checkpoint is None:
        return result
    assert convention is not None, "checkpoint convention must be explicit"
    assert checkpoint.stat().st_size < 512*1024**2, "bounded checkpoint CPU load"
    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
    from nerfstudio.models.splatfacto import get_viewmat
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)['pipeline']
    params = {k.rsplit('.', 1)[-1]: v for k, v in state.items() if '.gauss_params.' in k}
    means = params['means'].numpy()
    # Entire model remains on CPU. Exact local projected bbox selection determines
    # all splats whose gsplat support can touch this 64x64 crop.
    cp = (means.astype(float)-pose[:3, 3])@pose[:3, :3]
    q = params['quats'].numpy(); q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    scale = np.exp(params['scales'].numpy().astype(float))
    Kold = K.copy(); Kold[:2, 2] += .5 if convention == 'half' else 0.
    uv = cp[:, :2]/np.maximum(cp[:, 2:3], 1e-6)*[Kold[0, 0], Kold[1, 1]]+Kold[:2, 2]
    # Screen covariance eigenvalue radius, matching gsplat's eps2d=.3 and 3sigma.
    rcam = np.einsum('ij,njk->nik', pose[:3, :3].T, rot)*scale[:, None, :]
    safe_z = np.maximum(cp[:, 2], .01)
    tx = np.clip(cp[:, 0]/safe_z, -(Kold[0, 2]/Kold[0, 0]+.3*frame['w']/Kold[0, 0]),
                 (frame['w']-Kold[0, 2])/Kold[0, 0]+.3*frame['w']/Kold[0, 0])
    ty = np.clip(cp[:, 1]/safe_z, -(Kold[1, 2]/Kold[1, 1]+.3*frame['h']/Kold[1, 1]),
                 (frame['h']-Kold[1, 2])/Kold[1, 1]+.3*frame['h']/Kold[1, 1])
    jac = np.zeros((len(cp), 2, 3)); jac[:, 0, 0] = Kold[0, 0]/safe_z
    jac[:, 1, 1] = Kold[1, 1]/safe_z; jac[:, 0, 2] = -Kold[0, 0]*tx/safe_z
    jac[:, 1, 2] = -Kold[1, 1]*ty/safe_z
    b = jac@rcam; cov = b@b.transpose(0, 2, 1)+np.eye(2)*.3
    radius = np.ceil(3*np.sqrt(np.linalg.eigvalsh(cov)[:, -1]))+2
    # Choose crop with high reference support, fixed bounded 4x4 candidate grid.
    x0, y0 = max(0, frame['w']//2-32), max(0, frame['h']//2-32)
    candidates = []
    for y in np.linspace(32, frame['h']-96, 4, dtype=int):
        for x in np.linspace(32, frame['w']-96, 4, dtype=int):
            candidates.append((int(valid_saved[y:y+64, x:x+64].sum()), int(x), int(y)))
    _, x0, y0 = max(candidates)
    select = (cp[:, 2]>.01)&(uv[:, 0]+radius>=x0)&(uv[:, 0]-radius<=x0+64)&(uv[:, 1]+radius>=y0)&(uv[:, 1]-radius<=y0+64)
    ids = np.flatnonzero(select)
    result['model_crop'] = {'xywh': [x0, y0, 64, 64], 'candidate_splats': len(ids),
                            'total_splats_CPU_only': len(cp), 'checkpoint_convention': convention}
    if len(ids)>50000:
        result['model_crop']['skipped'] = 'exceeds 50000 Gaussian budget; no subsampling'
        return result
    # Preserve the original full-camera projection/covariance clamp, then shift
    # projected screen coordinates and rasterize ONLY a 64x64 tile crop. This
    # avoids changing the projection model when cropping the camera intrinsics.
    view = get_viewmat(torch.tensor(opencv_c2w_to_opengl(pose)[None, :3], device='cuda', dtype=torch.float32))
    distance, nearest = cKDTree(xyz).query(means[ids], workers=2)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        radii, means2d, depths, conics, _ = fully_fused_projection(params['means'][ids].to('cuda'), None,
            params['quats'][ids].to('cuda'), params['scales'][ids].exp().to('cuda'), view,
            torch.tensor(Kold[None], device='cuda', dtype=torch.float32), frame['w'], frame['h'],
            packed=False, near_plane=.01, far_plane=1e10)
        means2d = means2d-torch.tensor([x0, y0], device='cuda')
        _, intersections, flatten = isect_tiles(means2d, radii, depths, 16, 4, 4, packed=False, n_cameras=1)
        offsets = isect_offset_encode(intersections, 1, 4, 4)
        channels = torch.stack([depths, depths.square(),
            torch.tensor((distance>.25)[None], device='cuda', dtype=torch.float32),
            torch.tensor(((distance>=.5)|~normal_valid[nearest])[None], device='cuda', dtype=torch.float32)], -1)
        rendered, alpha = rasterize_to_pixels(means2d, conics, channels,
            params['opacities'][ids].sigmoid().squeeze(-1).to('cuda')[None],
            64, 64, 16, offsets, flatten, packed=False)
        expectation = rendered/alpha.clamp_min(1e-10)
    # Map is reprojected on OLD physical rays. Never compare ED to saved depth!
    rayK = Kold.copy(); rayK[:2, 2] -= .5
    oldref, oldvalid = project_surfel_depth(xyz, normals, normal_valid, pose, rayK, frame['w'], frame['h'])
    oldraw, oldraw_valid = project_depth(xyz, pose, rayK, frame['w'], frame['h'])
    sl = np.s_[y0:y0+64, x0:x0+64]
    d = expectation[0, ..., 0].cpu().numpy(); a = alpha[0, ..., 0].cpu().numpy()
    sigma = (expectation[0, ..., 1]-expectation[0, ..., 0].square()).clamp_min(0).sqrt().cpu().numpy()
    good = oldvalid[sl]&image_mask[sl]&(a>.5)
    raw_common_old = good&oldraw_valid[sl]
    nearer_raw = raw_common_old&(oldraw[sl]+1.<oldref[sl])
    result['model_crop'].update({'GT': 'same-source map reprojected on historical renderer rays; NOT corrected saved depth',
        'expected_z_minus_historical_ray_surfel': stats((d-oldref[sl])[good]),
        'reference_z': stats(oldref[sl][good]), 'expected_z': stats(d[good]),
        'alpha': stats(a[good]),
        'alpha_gt_095_residual': stats((d-oldref[sl])[good&(a>.95)]),
        'depth_weighted_std_m': stats(sigma[good]),
        'alpha_weight_fraction_centers_gt025m_from_map': stats(expectation[0, ..., 2].cpu().numpy()[good]),
        'alpha_weight_fraction_outside_hard_projection_gate': stats(expectation[0, ..., 3].cpu().numpy()[good]),
        'raw_nearer_gt1m_pixels': int(nearer_raw.sum()),
        'ED_vs_raw_on_historical_rays': stats((d-oldraw[sl])[raw_common_old]),
        'ED_vs_surfel_excluding_raw_nearer_gt1m': stats((d-oldref[sl])[good&~nearer_raw]),
        'ED_vs_raw_where_raw_nearer_gt1m': stats((d-oldraw[sl])[nearer_raw]),
        'crop_projection': 'original full-camera CUDA projection then shifted means2d; actual gsplat 64x64 raster kernels',
        'GPU_peak_bytes': torch.cuda.max_memory_allocated()})
    return result


def median_probe():
    """One surfel, 64px; test median output and its isolated backward."""
    from gsplat.rendering import rasterization_2dgs
    normal = np.array([.8, 0., .6]); tangent = np.cross([0., 1., 0.], normal)
    quat = Rotation.from_matrix(np.column_stack([tangent, np.cross(normal, tangent), normal])).as_quat()[[3, 0, 1, 2]].copy()
    params = dict(means=torch.tensor([[0., 0., 5.]], device='cuda', requires_grad=True),
        quats=torch.tensor(quat[None], device='cuda', dtype=torch.float32, requires_grad=True),
        scales=torch.tensor([[2., 2., .01]], device='cuda', requires_grad=True),
        opacities=torch.tensor([.8], device='cuda', requires_grad=True),
        colors=torch.full((1, 3), .6, device='cuda', requires_grad=True))
    torch.cuda.reset_peak_memory_stats()
    out = rasterization_2dgs(**params, viewmats=torch.eye(4, device='cuda')[None],
        Ks=torch.tensor([[[48., 0., 32.], [0., 48., 32.], [0., 0., 1.]]], device='cuda'),
        width=64, height=64, packed=False, render_mode='RGB+ED', depth_mode='median', absgrad=False)
    yy, xx = np.mgrid[:64, :64]
    ray = np.stack([(xx-31.5)/48, (yy-31.5)/48, np.ones_like(xx)], -1)
    plane = 5*normal[2]/(ray@normal)
    median = out[5][0, ..., 0]
    alpha = out[1][0, ..., 0].detach().cpu().numpy()
    mask = (alpha>.1)&(plane>0)&(plane<30)
    values = median.detach().cpu().numpy()
    result = {'primitive': {'mean': [0, 0, 5], 'normal': normal.tolist(),
        'scales': [2, 2, .01], 'opacity': .8}, 'depth_mode': 'median',
        'compared_pixels': int(mask.sum()),
        'render_median_range_m': [float(values[mask].min()), float(values[mask].max())],
        'analytic_plane_range_m': [float(plane[mask].min()), float(plane[mask].max())],
        'median_minus_plane_hit': stats((values-plane)[mask]),
        'median_minus_center_z': stats(values[mask]-5),
        'alpha_division_applied': False}
    out[-1]['means2d'].retain_grad(); out[-1]['gradient_2dgs'].retain_grad()
    # Isolated median supervision on supported pixels; gradients must not be
    # inferred from the RGB/ED channels of the other backward probe.
    target = torch.tensor(plane, device='cuda', dtype=torch.float32)
    known = torch.tensor(mask, device='cuda')
    median_loss = torch.nn.functional.smooth_l1_loss(median[known], target[known])
    median_loss.backward()
    result['median_only_loss'] = float(median_loss.detach())
    result['median_only_backward'] = {key: {'grad_present': value.grad is not None,
        'grad_finite': bool(torch.isfinite(value.grad).all()) if value.grad is not None else None,
        'grad_norm': float(value.grad.norm()) if value.grad is not None else None}
        for key, value in params.items()}
    result['strategy_gradient_backward'] = {key: {'grad_present': out[-1][key].grad is not None,
        'grad_finite': bool(torch.isfinite(out[-1][key].grad).all()) if out[-1][key].grad is not None else None,
        'grad_norm': float(out[-1][key].grad.norm()) if out[-1][key].grad is not None else None}
        for key in ['means2d', 'gradient_2dgs']}
    result['GPU_peak_bytes'] = torch.cuda.max_memory_allocated()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/block000_surfel_reference')
    parser.add_argument('--output', type=Path, default=RUNTIME/'depth_pipeline_diagnostic.json')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--median-only', action='store_true', help='Append a one-surfel median probe to the existing runtime JSON')
    parser.add_argument('--frame', default='images/1720267884750228828_camera_5_center.png')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-pixel-center', choices=['legacy-cv', 'half'])
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(RUNTIME.resolve()), 'output must stay inside runtime'
    if args.median_only:
        result = json.loads(args.output.read_text())
        result['synthetic']['2DGS_single_tilted_surfel']['render_median_probe'] = median_probe()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result['synthetic']['2DGS_single_tilted_surfel']['render_median_probe'], indent=2))
        return
    assert not args.checkpoint or (args.gpu and args.checkpoint_pixel_center), 'explicit checkpoint version and --gpu required'
    data = args.data.resolve(); meta = json.loads((data/'transforms.json').read_text())
    result = {'protocol': 'read-only same-source consistency diagnosis; no independent geometry GT',
              'versions': {'torch': torch.__version__, 'gsplat': version('gsplat'),
                           'nerfstudio': version('nerfstudio')},
              'pixel_center_convention': meta.get('pixel_center_convention'),
              'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [data/'transforms.json', ROOT/'src/orbvi_gaussian/lidar_splatfacto.py',
                            ROOT/'src/orbvi_gaussian/surfels.py', ROOT/'scripts/evaluate_reconstruction.py']}}
    result['loader'] = loader_audit(data, meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if args.gpu:
        result['synthetic'] = synthetic()
    result['real_frame'] = real_frame(data, meta, args.frame, args.checkpoint, args.checkpoint_pixel_center)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

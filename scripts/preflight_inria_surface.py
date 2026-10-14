#!/usr/bin/env python3
"""Bounded 64x64 original-2DGS oracle/backward probe; never starts training."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
# Set cache and temporary locations before importing Torch or any GPU module.
locations = {
    "XDG_CACHE_HOME": "runtime/cache/inria2dgs/xdg",
    "XDG_CONFIG_HOME": "runtime/config/inria2dgs",
    "XDG_DATA_HOME": "runtime/share/inria2dgs",
    "XDG_STATE_HOME": "runtime/state/inria2dgs",
    "TORCH_HOME": "runtime/cache/inria2dgs/torch",
    "TORCH_EXTENSIONS_DIR": "runtime/cache/inria2dgs/torch_extensions",
    "CUDA_CACHE_PATH": "runtime/cache/inria2dgs/cuda",
    "PIP_CACHE_DIR": "runtime/cache/inria2dgs/pip",
    "UV_CACHE_DIR": "runtime/cache/inria2dgs/uv",
    "UV_PYTHON_INSTALL_DIR": "runtime/python",
    "UV_PYTHON_BIN_DIR": "runtime/bin",
    "HF_HOME": "runtime/cache/inria2dgs/huggingface",
    "MPLCONFIGDIR": "runtime/cache/inria2dgs/matplotlib",
    "NUMBA_CACHE_DIR": "runtime/cache/inria2dgs/numba",
    "WANDB_DIR": "runtime/cache/inria2dgs/wandb",
    "WANDB_CACHE_DIR": "runtime/cache/inria2dgs/wandb_cache",
    "WANDB_CONFIG_DIR": "runtime/config/inria2dgs/wandb",
    "IPYTHONDIR": "runtime/config/inria2dgs/ipython",
    "JUPYTER_CONFIG_DIR": "runtime/config/inria2dgs/jupyter",
    "JUPYTER_DATA_DIR": "runtime/share/inria2dgs/jupyter",
    "TMPDIR": "runtime/inria2dgs_tmp", "TMP": "runtime/inria2dgs_tmp",
    "TEMP": "runtime/inria2dgs_tmp",
}
for key, relative in locations.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                  PIP_CONFIG_FILE="/dev/null", PIP_DISABLE_PIP_VERSION_CHECK="1",
                  UV_NO_CONFIG="1")
sys.path.insert(0, str(ROOT / "src"))

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import subprocess
import traceback

import torch
from orbvi_gaussian.inria_surfel_backend import (
    nerfstudio_c2w_to_cv_w2c, rasterization_inria_surface, render_surfel,
)


def qmul(a, b):
    w, x, y, z = a.unbind(-1)
    v, i, j, k = b.unbind(-1)
    return torch.stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v), -1)


def rotation(q):
    w, x, y, z = q
    return torch.stack((
        torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y))),
        torch.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x))),
        torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))),
    ))


def fixture(posed=False, cameras=1, requires_grad=False):
    device = torch.device("cuda:0")
    angle = math.atan2(.8, .6)
    qt = torch.tensor([math.cos(angle/2), 0., math.sin(angle/2), 0.], device=device)
    camera_q = torch.tensor([1., 0., 0., 0.], device=device)
    translation = torch.zeros(3, device=device)
    center = torch.tensor([0., 0., 5.], device=device)
    if posed:
        qz = torch.tensor([math.cos(.31), 0., 0., math.sin(.31)], device=device)
        qx = torch.tensor([math.cos(.17), math.sin(.17), 0., 0.], device=device)
        camera_q = qmul(qz, qx)
        translation = torch.tensor([2., -3., .8], device=device)
        center = torch.tensor([.2, -.1, 5.], device=device)
    r = rotation(camera_q)
    cv_c2w = torch.eye(4, device=device)
    cv_c2w[:3, :3] = r
    cv_c2w[:3, 3] = translation
    ns_c2w = cv_c2w.clone()
    ns_c2w[:3, :3] *= ns_c2w.new_tensor([1., -1., -1.])
    view = nerfstudio_c2w_to_cv_w2c(ns_c2w[None, :3])
    torch.testing.assert_close(view[0], torch.linalg.inv(cv_c2w), atol=1e-6, rtol=1e-6)
    k = torch.tensor([[48., 0., 32.], [0., 48., 32.], [0., 0., 1.]], device=device)
    if posed:
        k[0, 2], k[1, 2], k[1, 1] = 29.2, 35.3, 51.
    ks = k[None].repeat(cameras, 1, 1)
    if cameras > 1:
        ks[1, 0, 2] -= 2.3
        ks[1, 1, 2] += 1.7
    params = dict(means=(r @ center + translation)[None],
                  quats=qmul(camera_q, qt)[None],
                  scales=torch.tensor([[2., 2., .01]], device=device),
                  opacities=torch.tensor([.8], device=device),
                  colors=torch.tensor([[.6, .4, .2]], device=device))
    params = {k: v.detach().clone().requires_grad_(requires_grad) for k, v in params.items()}
    camera = dict(viewmats=view.repeat(cameras, 1, 1), Ks=ks, width=64, height=64)
    return params, camera, center, r


def plane_oracle(camera, center, pixel_center=.5):
    y, x = torch.meshgrid(torch.arange(64, device="cuda"), torch.arange(64, device="cuda"), indexing="ij")
    k = camera['Ks']
    rays = torch.stack(((x[None]+pixel_center-k[:, None, None, 0, 2])/k[:, None, None, 0, 0],
                        (y[None]+pixel_center-k[:, None, None, 1, 2])/k[:, None, None, 1, 1],
                        torch.ones((len(k), 64, 64), device="cuda")), -1)
    normal = rays.new_tensor([.8, 0., .6])
    return (normal @ center) / (rays @ normal)


def depth_case(posed=False, cameras=1, pixel_center=.5):
    params, camera, center, r = fixture(posed, cameras)
    oracle = plane_oracle(camera, center, pixel_center)
    with torch.no_grad():
        out = render_surfel(**params, **camera, pixel_center=pixel_center)
        expected, median = out.depth_expected[..., 0], out.depth_median[..., 0]
        mask = (out.alpha[..., 0] > .1) & (oracle > .2) & (oracle < 30)
        assert int(mask.sum()) > 200
        errors = [(value[mask]-oracle[mask]).abs().max().item() for value in (expected, median)]
        assert max(errors) < 3e-4, errors
        assert float(expected[mask].max()-expected[mask].min()) > 2.
        weighted = out.allmap[:, 0]
        torch.testing.assert_close(weighted[mask], expected[mask]*out.alpha[..., 0][mask], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(median, out.allmap[:, 5], atol=0, rtol=0)
        world_normal = -(r @ r.new_tensor([.8, 0., .6]))
        normal_error = (out.normals_world[mask]-world_normal).abs().max().item()
        assert normal_error < 2e-5, normal_error
        expected_weighted_normal = world_normal * out.alpha[mask]
        weighted_normal_error = (out.normals_world_weighted[mask]-expected_weighted_normal).abs().max().item()
        assert weighted_normal_error < 2e-5, weighted_normal_error
        # Opposite normal faces yield the same geometry, and front-facing normals.
        sample_y = 32
        samples = []
        for sample_x in (24, 32, 40):
            samples.append(dict(pixel=[sample_x, sample_y], alpha=float(out.alpha[0, sample_y, sample_x, 0]),
                                oracle_z_m=float(oracle[0, sample_y, sample_x]),
                                expected_z_m=float(expected[0, sample_y, sample_x]),
                                median_z_m=float(median[0, sample_y, sample_x])))
        return dict(cameras=cameras, posed=posed, pixel_center=pixel_center,
                    compared_pixels=int(mask.sum()), expected_range_m=[float(expected[mask].min()), float(expected[mask].max())],
                    oracle_range_m=[float(oracle[mask].min()), float(oracle[mask].max())],
                    expected_max_error_m=errors[0], median_max_error_m=errors[1],
                    normals_world_max_error=normal_error, weighted_normals_max_error=weighted_normal_error,
                    native_allmap_shape=list(out.allmap.shape), radii=out.info['radii'].tolist(), samples=samples)


def backward_case(mode):
    # Strategy import is safe: its CUDA operators are lazy. No gsplat kernel is invoked.
    from gsplat.strategy import DefaultStrategy
    params, camera, center, _ = fixture(requires_grad=True)
    oracle = plane_oracle(camera, center)
    proxy = torch.zeros((1, 1, 2), device='cuda', requires_grad=True)
    image, alpha, info = rasterization_inria_surface(**params, **camera, depth_mode=mode, means2d=proxy)
    assert info['means2d'] is proxy and info['gradient_2dgs'] is proxy
    mask = ((alpha[..., 0] > .1) & (oracle > .2) & (oracle < 30)).detach()
    strategy = DefaultStrategy(absgrad=False, key_for_gradient='means2d')
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, {}, state, 1, info)
    target = (oracle + .3).detach()
    depth_loss = (image[..., 3][mask]-target[mask]).square().mean()
    depth_loss.backward(retain_graph=True)
    depth_grads = {k: dict(norm=float(v.grad.norm()), finite=bool(torch.isfinite(v.grad).all()))
                   for k, v in params.items()}
    assert depth_grads['means']['norm'] > 1e-4 and depth_grads['quats']['norm'] > 1e-4
    assert all(v['finite'] for v in depth_grads.values())
    analytic_mean_z_grad = float(params['means'].grad[0, 2])
    epsilon = 1e-3
    scalar_losses = []
    with torch.no_grad():
        for offset in (-epsilon, epsilon):
            shifted = {k: v.detach().clone() for k, v in params.items()}
            shifted['means'][0, 2] += offset
            value, _, _ = rasterization_inria_surface(**shifted, **camera, depth_mode=mode)
            scalar_losses.append(float((value[..., 3][mask]-target[mask]).square().mean()))
    finite_difference = (scalar_losses[1]-scalar_losses[0]) / (2*epsilon)
    assert math.isclose(analytic_mean_z_grad, finite_difference, rel_tol=.01, abs_tol=.002), (analytic_mean_z_grad, finite_difference)
    for v in params.values():
        v.grad = None
    proxy.grad = None
    native = info['native_means2d'][0]
    native.grad = None
    ramp = torch.linspace(.5, 1.5, 64, device='cuda')[None, None, :, None]
    loss = depth_loss + (image[..., :3]*ramp).mean() + .2*alpha.mean()
    loss.backward()
    grads = {k: dict(norm=float(v.grad.norm()), finite=bool(torch.isfinite(v.grad).all()))
             for k, v in params.items()}
    assert all(v['finite'] and v['norm'] > 0 for v in grads.values()), grads
    assert proxy.grad is not None and bool(torch.isfinite(proxy.grad).all()) and float(proxy.grad.norm()) > 0
    torch.testing.assert_close(proxy.grad[0], native.grad[:, :2] / proxy.new_tensor([32., 32.]), atol=1e-6, rtol=1e-5)
    strategy.step_post_backward(params, {}, state, 1, info, packed=False)
    assert float(state['grad2d'][0]) > 0 and float(state['count'][0]) == 1
    assert 'gsplat.cuda._backend' not in sys.modules, 'Unexpected gsplat CUDA backend import'
    return dict(depth_mode=mode, depth_only_gradients=depth_grads, composite_gradients=grads,
                depth_loss=float(depth_loss), mean_z_autograd=analytic_mean_z_grad,
                mean_z_finite_difference=finite_difference, means2d_shape=list(proxy.shape),
                means2d_pixel_gradient=proxy.grad.tolist(), native_means2d_ndc_gradient=native.grad.tolist(),
                strategy=dict(absgrad=False, key_for_gradient='means2d', grad2d=state['grad2d'].tolist(), count=state['count'].tolist()),
                gsplat_cuda_backend_imported=False)


def additional_contracts():
    params, camera, _, _ = fixture()
    with torch.no_grad():
        rgb_out = render_surfel(**params, **camera)
        sh_params = dict(params)
        sh_params['colors'] = ((params['colors']-.5) / .28209479177387814)[:, None, :]
        sh_out = render_surfel(**sh_params, **camera, sh_degree=0)
        torch.testing.assert_close(rgb_out.rgb, sh_out.rgb, atol=2e-6, rtol=2e-6)
        zero_params = dict(params, opacities=torch.zeros_like(params['opacities']))
        blank = render_surfel(**zero_params, **camera, backgrounds=params['colors'][0])
        assert not bool(blank.alpha.any()) and not bool(blank.depth_expected.any()) and not bool(blank.depth_median.any())
        torch.testing.assert_close(blank.rgb, params['colors'].expand_as(blank.rgb), atol=0, rtol=0)
        empty_params = {k: v[:0] for k, v in params.items()}
        empty = render_surfel(**empty_params, **camera, backgrounds=params['colors'][0])
        torch.testing.assert_close(empty.rgb, blank.rgb, atol=0, rtol=0)
    rejected = []
    for key, value in [('absgrad', True), ('packed', True), ('sparse_grad', True), ('rasterize_mode', 'antialiased')]:
        try:
            rasterization_inria_surface(**params, **camera, **{key: value})
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError(f'{key} was silently accepted')
    return dict(sh_degree_0_rgb_match=True, zero_alpha_depth_finite_zero=True,
                empty_scene_background_correct=True, unsupported_modes_rejected=rejected)


def run():
    import diff_surfel_rasterization as upstream
    before = json.loads((ROOT / 'runtime/inria2dgs_before.json').read_text())
    versions = {name: metadata.version(name) for name in ('torch', 'nerfstudio', 'gsplat', 'diff-surfel-rasterization')}
    assert versions['torch'] == '2.4.0+cu124' and versions['nerfstudio'] == '1.1.5' and versions['gsplat'] == '1.4.0'
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    evidence = dict(versions=versions, torch_cuda_runtime=torch.version.cuda,
                    cuda_compiler=subprocess.check_output(['/usr/local/cuda-12.8/bin/nvcc', '--version'], text=True).strip(),
                    gpu=torch.cuda.get_device_name(), extension=upstream._C.__file__,
                    provenance=json.loads((ROOT / 'runtime/inria2dgs_provenance.json').read_text()))
    evidence['tilted_identity'] = depth_case()
    evidence['tilted_pose_offcentre'] = depth_case(posed=True)
    evidence['two_cameras'] = depth_case(cameras=2)
    evidence['integer_centre_cv'] = depth_case(pixel_center=0.)
    evidence['backward'] = [backward_case(mode) for mode in ('expected', 'median')]
    evidence['contracts'] = additional_contracts()
    torch.cuda.synchronize()
    evidence['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    evidence['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    evidence['memory_limit_bytes'] = 50*1024*1024
    assert evidence['peak_reserved_bytes'] < evidence['memory_limit_bytes']
    current = {p.metadata['Name']: p.version for p in metadata.distributions()}
    changed = {name: [v, current.get(name)] for name, v in before['packages'].items() if current.get(name) != v}
    added = {name: v for name, v in current.items() if name not in before['packages']}
    assert not changed, changed
    assert set(added) == {'diff_surfel_rasterization'}, added
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in before['gsplat_shared_objects']}
    assert hashes == before['gsplat_shared_objects']
    evidence['isolation'] = dict(changed_existing_packages=changed, added_packages=added,
                                gsplat_shared_object_hashes_unchanged=hashes, environment={k: os.environ[k] for k in locations},
                                gsplat_cuda_backend_imported='gsplat.cuda._backend' in sys.modules,
                                training_started=False)
    evidence['passed'] = True
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/inria2dgs_preflight.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('all output must remain inside 3D-Gaussian')
    try:
        evidence = run()
    except Exception as exc:
        evidence = dict(passed=False, error=str(exc), traceback=traceback.format_exc())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2)+'\n')
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2)+'\n')
    summary = {k: evidence[k] for k in ('passed', 'versions', 'tilted_identity', 'backward', 'peak_allocated_bytes', 'peak_reserved_bytes', 'isolation')}
    summary['isolation'] = {k: v for k, v in summary['isolation'].items() if k != 'environment'}
    summary['output'] = str(output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Original hbb1 2DGS renderer with Nerfstudio/CV camera and gradient adapters.

Upstream: hbb1/diff-surfel-rasterization, e0ed0207b3e0669960cfad70852200a4a5847f61.
The upstream implementation retains its Inria/MPII research/evaluation license;
see runtime/inria2dgs_src/LICENSE.md and reports/INRIA_SURFACE_BACKEND.md.
This module imports neither the gsplat CUDA backend nor the training model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

InriaBackend = Literal['original', 'aa-depth-vjp']
_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_evidence(root: Path) -> dict:
    """Refuse unvalidated/rebuilt binaries or a changed forward definition."""
    manifest_path = root / 'aa_depth_vjp_provenance.json'
    validation_path = root / 'aa_depth_vjp_validation.json'
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    report_path = _ROOT / validation['report']
    report = json.loads(report_path.read_text())
    if (_sha(report_path) != validation['report_sha256']
            or validation['status'] != 'LOCAL_CUDA_DEPTH_VJP_FINITE_DIFFERENCE_PASS'
            or report.get('candidate_status') != 'PASS'
            or not report.get('forward_arrays_bitwise_equal')
            or not report.get('rgb_alpha_parameter_gradients_bitwise_equal')):
        raise ValueError('AA depth VJP requires pinned passing CUDA/forward validation')
    for relative, digest in manifest['candidate_source_sha256'].items():
        if _sha(root / relative) != digest:
            raise ValueError(f'AA depth VJP source drift: {relative}')
    forward = root / 'cuda_rasterizer/forward.cu'
    original = _ROOT / 'runtime/inria2dgs_src/cuda_rasterizer/forward.cu'
    if (_sha(forward) != validation['forward_source_sha256']
            or forward.read_bytes() != original.read_bytes()):
        raise ValueError('AA depth VJP forward must remain byte-identical to original')
    package = root / 'diff_surfel_rasterization_aa_depth_vjp'
    binaries = list(package.glob('_C*.so'))
    if len(binaries) != 1 or _sha(binaries[0]) != validation['candidate_extension_sha256']:
        raise ValueError('AA depth VJP binary changed; rerun native finite differences')
    if (validation['candidate_extension_sha256'] != report['candidate']['extension_sha256']
            or _sha(package / '__init__.py') != _sha(_ROOT / 'runtime/inria2dgs_src/diff_surfel_rasterization/__init__.py')):
        raise ValueError('AA depth VJP module differs from validated interface')
    return dict(validation_path=str(validation_path), validation_sha256=_sha(validation_path),
        finite_difference_report=str(report_path), finite_difference_report_sha256=_sha(report_path),
        build_manifest_path=str(manifest_path), build_manifest_sha256=_sha(manifest_path),
        forward_source_byte_identical=True, tested_forward_arrays_bitwise_equal=True,
        tested_rgb_alpha_parameter_gradients_bitwise_equal=True,
        original_extension_sha256=validation['original_extension_sha256'])


@lru_cache(maxsize=2)
def _load_backend(backend: InriaBackend = 'original'):
    if backend == 'original':
        return importlib.import_module('diff_surfel_rasterization')
    if backend != 'aa-depth-vjp':
        raise ValueError(f'Unknown INRIA backend: {backend}')
    root = _ROOT / 'runtime/inria2dgs_aa_depth_vjp_src'
    _candidate_evidence(root)
    # Explicit runtime-only opt-in; never install, alias, or replace original.
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module('diff_surfel_rasterization_aa_depth_vjp')
    finally:
        sys.path.remove(str(root))
    if Path(module.__file__).resolve() != root / 'diff_surfel_rasterization_aa_depth_vjp/__init__.py':
        raise ValueError('AA depth VJP imported from an unexpected package path')
    validation = json.loads((root / 'aa_depth_vjp_validation.json').read_text())
    if _sha(Path(module._C.__file__)) != validation['candidate_extension_sha256']:
        raise ValueError('Loaded AA depth VJP extension differs from validated binary')
    return module


def inria_backend_provenance(backend: InriaBackend = 'original') -> dict:
    """Actual imported module/binary identity; performs no CUDA operations."""
    module = _load_backend(backend)
    source = _ROOT / 'runtime' / ('inria2dgs_src' if backend == 'original' else 'inria2dgs_aa_depth_vjp_src')
    paths = ('cuda_rasterizer/forward.cu', 'cuda_rasterizer/backward.cu',
             'cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/auxiliary.h',
             'rasterize_points.cu', 'rasterize_points.h', 'ext.cpp', 'setup.py')
    return dict(schema_version=1, variant=backend, module=module.__name__,
        adapter_path=__file__, adapter_sha256=_sha(Path(__file__)),
        wrapper_path=module.__file__, wrapper_sha256=_sha(Path(module.__file__)),
        extension_path=module._C.__file__, extension_sha256=_sha(Path(module._C.__file__)),
        source_directory=str(source), source_sha256={p:_sha(source/p) for p in paths if (source/p).is_file()},
        forward_contract='original ray-plane depth and low-pass coverage; no center-depth fallback',
        validation=_candidate_evidence(source) if backend == 'aa-depth-vjp' else None)


def model_inria_backend_provenance(model_config) -> dict | None:
    if not getattr(model_config, 'use_inria_surfel_renderer', False):
        return None
    return inria_backend_provenance(getattr(model_config, 'inria_backend', 'original'))


@dataclass
class SurfelRender:
    """Images use [C,H,W,D]; allmap preserves the native [C,7,H,W] layout."""

    rgb: Tensor
    alpha: Tensor
    depth_expected: Tensor
    depth_median: Tensor
    normals_world: Tensor
    normals_world_weighted: Tensor
    distortion: Tensor
    allmap: Tensor
    info: dict[str, Any]


def nerfstudio_c2w_to_cv_w2c(camera_to_worlds: Tensor) -> Tensor:
    """NS/OpenGL [C,3,4] or [C,4,4] poses -> positive-Z CV [C,4,4].

    NS camera axes are right/up/back; CV axes are right/down/forward.
    Input and output translations retain the supplied world units (metres).
    """
    if camera_to_worlds.ndim != 3 or camera_to_worlds.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("camera_to_worlds must have shape [C,3,4] or [C,4,4]")
    c2w = camera_to_worlds
    rotation = c2w[:, :3, :3] * c2w.new_tensor([1.0, -1.0, -1.0])
    inverse_rotation = rotation.transpose(-1, -2)
    translation = -(inverse_rotation @ c2w[:, :3, 3:4])
    bottom = c2w.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(len(c2w), 1, 4)
    return torch.cat((torch.cat((inverse_rotation, translation), -1), bottom), -2)


def _projection(K: Tensor, width: int, height: int, pixel_center: float) -> Tensor:
    # Native ndc2pix adds (size-1)/2 and evaluates rays at integer pixel IDs.
    # A NS intrinsic matrix instead describes centres at (x+.5,y+.5).
    # Retain arbitrary principal points, including rectified/cropped cameras.
    projection = K.new_zeros((4, 4))
    projection[0, 0] = 2.0 * K[0, 0] / width
    projection[1, 1] = 2.0 * K[1, 1] / height
    projection[0, 2] = 2.0 * (K[0, 2] - pixel_center - (width - 1) / 2.0) / width
    projection[1, 2] = 2.0 * (K[1, 2] - pixel_center - (height - 1) / 2.0) / height
    # Original kernel's intrinsic depth/culling constants, not scene rescaling.
    near, far = 0.2, 100.0
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -far * near / (far - near)
    projection[3, 2] = 1.0  # homogeneous W = CV camera Z
    return projection


def render_surfel(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    *,
    sh_degree: int | None = None,
    backgrounds: Tensor | None = None,
    pixel_center: float = 0.5,
    means2d: Tensor | None = None,
    backend: InriaBackend = 'original',
) -> SurfelRender:
    """Render original ray-plane Z and expose a gsplat densification proxy.

    means [N,3] are world metres; quats [N,4] are wxyz; scales [N,2/3]
    are activated tangent scales in metres (the third scale is ignored).
    opacities [N] or [N,1] are activated probabilities. colors are RGB
    [N,3], or SH [N,K,3] when sh_degree is supplied. CV viewmats [C,4,4]
    use right/down/forward axes, as returned by NS get_viewmat. Ks [C,3,3]
    default to NS half-pixel centres; set pixel_center=0 for integer-centre CV K.

    Expected depth divides native weighted depth by alpha exactly once.
    Median is the native transmittance-threshold depth, without division.
    World normals are unit vectors; normals_world_weighted preserves native
    alpha-weighted normals for the original normal consistency loss.

    info['means2d'] (also 'gradient_2dgs') is a [C,N,2] gradient proxy,
    not projected coordinates. Its gradient has pixel units, suitable for
    DefaultStrategy(absgrad=False, key_for_gradient='means2d'), packed=False.
    info['native_means2d'] retains each original [N,3] input and its NDC grad.
    Native means2D is only a backward signal; its values do not affect forward.

    Cameras/intrinsics/backgrounds have no supported native backward. The
    original kernel culls at Z<=.2m and uses .2/100m for distortion; arbitrary
    clip planes, distortion models, absgrad, and antialiasing are unsupported.
    """
    if means.ndim != 2 or means.shape[1] != 3 or means.dtype != torch.float32 or not means.is_cuda:
        raise ValueError("means must be CUDA float32 [N,3]")
    n = len(means)
    if quats.shape != (n, 4) or scales.ndim != 2 or scales.shape[0] != n or scales.shape[1] not in (2, 3):
        raise ValueError("quats/scales must have shapes [N,4] and [N,2/3]")
    if opacities.shape not in ((n,), (n, 1)):
        raise ValueError("opacities must have shape [N] or [N,1]")
    if viewmats.ndim != 3 or viewmats.shape[-2:] != (4, 4) or Ks.shape != (len(viewmats), 3, 3):
        raise ValueError("viewmats/Ks must have shapes [C,4,4] and [C,3,3]")
    c = len(viewmats)
    if c < 1 or width < 1 or height < 1 or pixel_center not in (0.0, 0.5):
        raise ValueError("positive camera/image counts and pixel_center=0 or .5 are required")
    if sh_degree is None:
        if colors.shape != (n, 3):
            raise ValueError("precomputed RGB must have shape [N,3]")
    elif (not 0 <= sh_degree <= 3 or colors.ndim != 3 or colors.shape[0] != n
          or colors.shape[2] != 3 or colors.shape[1] < (sh_degree + 1) ** 2):
        raise ValueError("SH must have shape [N,K,3] with enough coefficients for degree 0..3")
    tensors = (quats, scales, opacities, colors, viewmats, Ks)
    if any(t.dtype != means.dtype or t.device != means.device for t in tensors):
        raise ValueError("all inputs must share means' CUDA device and float32 dtype")
    if viewmats.requires_grad or Ks.requires_grad:
        raise ValueError("original 2DGS does not implement camera/intrinsic gradients")
    if not bool(torch.all(Ks[:, (0, 1), (0, 1)] > 0)):
        raise ValueError("focal lengths must be positive")
    if not bool(torch.all(Ks[:, (0, 1), (1, 0)] == 0)) or not bool(torch.all(Ks[:, 2] == Ks.new_tensor([0., 0., 1.]))):
        raise ValueError("only zero-skew, undistorted pinhole intrinsics are supported")
    if not bool(torch.all(torch.isfinite(quats))) or not bool(torch.all(quats.square().sum(-1) > 0)):
        raise ValueError("quaternions must be finite and nonzero")
    if not bool(torch.all(torch.isfinite(scales[:, :2]) & (scales[:, :2] > 0))):
        raise ValueError("tangent scales must be finite and positive")
    if not bool(torch.all(torch.isfinite(opacities) & (opacities >= 0) & (opacities <= 1))):
        raise ValueError("opacities must be finite activated probabilities in [0,1]")
    if backgrounds is None:
        backgrounds = means.new_zeros((c, 3))
    if backgrounds.shape == (3,):
        backgrounds = backgrounds.expand(c, 3)
    if backgrounds.shape != (c, 3) or backgrounds.device != means.device or backgrounds.dtype != means.dtype:
        raise ValueError("backgrounds must be CUDA float32 [3] or [C,3]")
    if backgrounds.requires_grad:
        raise ValueError("original 2DGS does not implement background gradients")
    if means2d is None:
        means2d = torch.zeros((c, n, 2), device=means.device, dtype=means.dtype,
                              requires_grad=torch.is_grad_enabled())
    if means2d.shape != (c, n, 2) or means2d.device != means.device or means2d.dtype != means.dtype:
        raise ValueError("means2d must share means' device/dtype and have shape [C,N,2]")
    if torch.is_grad_enabled() and not means2d.requires_grad:
        raise ValueError("means2d must require grad when gradient recording is enabled")
    if means2d.requires_grad:
        means2d.retain_grad()

    module = _load_backend(backend)
    GaussianRasterizationSettings, GaussianRasterizer = module.GaussianRasterizationSettings, module.GaussianRasterizer

    images, maps, radii, native_means2d = [], [], [], []
    grad_scale = means.new_tensor([width / 2.0, height / 2.0])
    # The upstream Python API allocates missing arguments on current CUDA device.
    with torch.cuda.device(means.device):
        for cid in range(c):
            # Upstream backward already multiplies W/2,H/2 to report NDC grads.
            # This autograd bridge converts them to pixel grads, without hooks.
            native = F.pad(means2d[cid] / grad_scale, (0, 1))
            if native.requires_grad:
                native.retain_grad()
            native_means2d.append(native)
            w2c = viewmats[cid]
            projection = _projection(Ks[cid], width, height, pixel_center)
            camera_center = -(w2c[:3, :3].T @ w2c[:3, 3])
            settings = GaussianRasterizationSettings(
                image_height=height, image_width=width,
                tanfovx=width / (2.0 * float(Ks[cid, 0, 0])),
                tanfovy=height / (2.0 * float(Ks[cid, 1, 1])),
                bg=backgrounds[cid].contiguous(), scale_modifier=1.0,
                viewmatrix=w2c.T.contiguous(),
                projmatrix=(projection @ w2c).T.contiguous(),
                sh_degree=0 if sh_degree is None else sh_degree,
                campos=camera_center.contiguous(), prefiltered=False, debug=False,
            )
            image, radius, allmap = GaussianRasterizer(settings)(
                means3D=means.contiguous(), means2D=native,
                opacities=opacities.reshape(n, 1).contiguous(),
                shs=None if sh_degree is None else colors.contiguous(),
                colors_precomp=colors.contiguous() if sh_degree is None else None,
                scales=scales[:, :2].contiguous(),
                rotations=F.normalize(quats, dim=-1).contiguous(), cov3D_precomp=None,
            )
            # Native P=0 returns black even with a nonzero background.
            if n == 0:
                image = image + backgrounds[cid, :, None, None]
            images.append(image.permute(1, 2, 0))
            maps.append(allmap)
            radii.append(radius)
    raw = torch.stack(maps)
    attributes = raw.permute(0, 2, 3, 1)
    alpha = attributes[..., 1:2]
    # Avoid an invalid division even on the unselected side of torch.where.
    expected = attributes[..., 0:1] / torch.where(alpha > 0, alpha, torch.ones_like(alpha))
    expected = torch.where(alpha > 0, expected, torch.zeros_like(expected))
    median = attributes[..., 5:6]
    camera_normals_weighted = attributes[..., 2:5]
    # For row vectors n_camera @ R_w2c is n_world.
    world_weighted = torch.einsum("chwi,cij->chwj", camera_normals_weighted, viewmats[:, :3, :3])
    world_unit = F.normalize(world_weighted, dim=-1, eps=1e-12)
    info = {
        "width": width, "height": height, "n_cameras": c,
        "radii": torch.stack(radii), "gaussian_ids": None,
        "means2d": means2d, "gradient_2dgs": means2d,
        "native_means2d": tuple(native_means2d),
        "normals_rend": world_weighted, "normals_world": world_unit,
        "render_distloss": attributes[..., 6:7],
        "render_distort": attributes[..., 6:7],
        "depth_expected": expected, "depth_median": median,
        "allmap": raw, "backend": "hbb1-original-2dgs" if backend == 'original' else "hbb1-aa-depth-vjp-2dgs", "absgrad": False,
        "inria_backend": backend,
        "pixel_center": pixel_center, "gradient_units": "pixels",
    }
    return SurfelRender(torch.stack(images), alpha, expected, median, world_unit,
                        world_weighted, attributes[..., 6:7], raw, info)


def rasterization_inria_surface(
    means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int, *,
    render_mode: Literal["RGB", "RGB+ED"] = "RGB+ED",
    depth_mode: Literal["expected", "median"] = "expected",
    sh_degree: int | None = None, backgrounds: Tensor | None = None,
    packed: bool = False, absgrad: bool = False, sparse_grad: bool = False,
    rasterize_mode: str = "classic", pixel_center: float = 0.5,
    means2d: Tensor | None = None,
    backend: InriaBackend = 'original',
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """NS-shaped (render, alpha, info); RGB+ED contains normalized camera Z.

    Clip planes are deliberately not accepted: upstream culling/distortion
    constants are fixed. Parent integration must explicitly use those semantics.
    No module monkeypatching or model/training changes are performed here.
    """
    if packed or absgrad or sparse_grad or rasterize_mode != "classic":
        raise ValueError("original 2DGS requires packed=False, absgrad=False, sparse_grad=False, classic mode")
    if render_mode not in ("RGB", "RGB+ED") or depth_mode not in ("expected", "median"):
        raise ValueError("supported modes: RGB/RGB+ED and expected/median")
    result = render_surfel(means, quats, scales, opacities, colors, viewmats, Ks,
                           width, height, sh_degree=sh_degree, backgrounds=backgrounds,
                           pixel_center=pixel_center, means2d=means2d, backend=backend)
    depth = result.depth_expected if depth_mode == "expected" else result.depth_median
    image = result.rgb if render_mode == "RGB" else torch.cat((result.rgb, depth), -1)
    return image, result.alpha, result.info

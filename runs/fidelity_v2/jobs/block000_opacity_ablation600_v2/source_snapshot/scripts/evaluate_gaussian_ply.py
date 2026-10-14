#!/usr/bin/env python3
"""Evaluate an actual trained world-space GS PLY on fixed DEV or TEST frames.

Select candidates on DEV; evaluate the selected candidate on TEST once. This
script never fits parameters or changes source cameras, masks, or depth targets.
Schema/attribute helpers need only NumPy; the rendering imports are lazy.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from package_garage import (
    MAX_CHUNK_BYTES, MAX_JSON_BYTES, VIEWER_MAX_BYTES, VIEWER_MAX_GAUSSIANS,
    _state, inspect_ply, read_json, scan_ply,
)

MAX_GAUSSIANS = VIEWER_MAX_GAUSSIANS
MAX_PLY_BYTES = VIEWER_MAX_BYTES
MAX_IMAGE_PIXELS = 16_000_000
PARAMETER_WIDTHS = {"means": 3, "features_dc": 3, "opacities": 1,
                    "scales": 3, "quats": 4}


def confined_path(value, *, file=False):
    path = Path(value).expanduser().resolve()
    if ROOT not in path.parents:
        raise ValueError("All input/output paths must stay inside project: %s" % path)
    if file and not path.is_file():
        raise FileNotFoundError("Missing source file: %s" % path)
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with confined_path(path, file=True).open("rb") as stream:
        for raw in iter(lambda: stream.read(MAX_CHUNK_BYTES), b""):
            digest.update(raw)
    return digest.hexdigest()


def read_candidate(path, *, max_gaussians=None, max_ply_bytes=None, extended_cpu_budget=False):
    """Read every record without filtering; bound allocation before reading.

    Official f_rest fields are [R coefficients, G coefficients, B coefficients],
    exported by features_rest.transpose(1,2).flatten(start_dim=1). Store the
    inverse as [N, SH coefficients, RGB], without exponentiating/logisticizing.
    """
    path = confined_path(path, file=True)
    max_gaussians = MAX_GAUSSIANS if max_gaussians is None else max_gaussians
    max_ply_bytes = MAX_PLY_BYTES if max_ply_bytes is None else max_ply_bytes
    hard_count, hard_bytes = (12_000_000,3*1024**3) if extended_cpu_budget else (6_000_000,2*1024**3)
    if (type(max_gaussians) is not int or not 0 < max_gaussians <= hard_count
            or type(max_ply_bytes) is not int or not 0 < max_ply_bytes <= hard_bytes):
        raise ValueError('Explicit inference budgets require 1..%d records and 1..%d source bytes' % (hard_count,hard_bytes))
    state = _state(path)
    if state[2] > max_ply_bytes:
        raise ValueError("PLY exceeds %d-byte budget" % max_ply_bytes)
    with path.open("rb") as stream:
        spec = inspect_ply(stream)
    count = spec["count"]
    if count > max_gaussians:
        raise ValueError("PLY exceeds %d-Gaussian budget" % max_gaussians)
    if state[2] != len(spec["header"]) + count * spec["stride"]:
        raise ValueError("Incomplete or trailing PLY payload")
    names = spec["names"]
    # The packaging reader permits extra confidence fields. Reject misspelled
    # parameter-family fields instead of silently ignoring an invalid schema.
    families = {"f_dc_": 3, "f_rest_": 3 * ((spec["sh_degree"] + 1) ** 2 - 1),
                "scale_": 3, "rot_": 4}
    for prefix, width in families.items():
        if {n for n in names if n.startswith(prefix)} != {prefix + str(i) for i in range(width)}:
            raise ValueError("Invalid %s schema" % prefix)
    rest_count = (spec["sh_degree"] + 1) ** 2 - 1
    parameters = {key: np.empty((count, width), np.float32)
                  for key, width in PARAMETER_WIDTHS.items()}
    parameters["features_rest"] = np.empty((count, rest_count, 3), np.float32)
    columns = {"means": [names.index(n) for n in ("x", "y", "z")],
               "features_dc": [names.index("f_dc_%d" % i) for i in range(3)],
               "opacities": [names.index("opacity")],
               "scales": [names.index("scale_%d" % i) for i in range(3)],
               "quats": [names.index("rot_%d" % i) for i in range(4)]}
    rest_columns = [names.index("f_rest_%d" % i) for i in range(3 * rest_count)]
    cursor = 0

    def consume(raw):
        nonlocal cursor
        rows = np.frombuffer(raw, dtype="<f4").reshape(-1, len(names))
        stop = cursor + len(rows)
        for key, ids in columns.items():
            parameters[key][cursor:stop] = rows[:, ids]
        parameters["features_rest"][cursor:stop] = rows[:, rest_columns].reshape(
            len(rows), 3, rest_count).transpose(0, 2, 1)
        norms = np.linalg.norm(rows[:, columns["quats"]].astype(np.float64), axis=1)
        if not np.allclose(norms, 1., rtol=0., atol=1e-4):
            raise ValueError("Candidate requires unit wxyz quaternions")
        # Float32 exp is what the model actually passes to the renderer.
        with np.errstate(over="ignore", under="ignore"):
            activated = np.exp(rows[:, columns["scales"]])
        if not (np.isfinite(activated).all() and np.all(activated > 0)):
            raise ValueError("Log scales must activate to finite positive float32 scales")
        cursor = stop

    info = scan_ply({"path": path, "spec": spec, "state": state},
                    MAX_CHUNK_BYTES, full=consume)
    info.update(path=str(path), gaussians=count, sh_degree=spec["sh_degree"],
                fields=list(names), size_bytes=state[2],
                format="binary_little_endian_float32", coordinates="camera_init",
                attribute_convention="world XYZ; log scales; opacity logits; unit wxyz; channel-major PLY SH")
    return parameters, info


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def require_identity_parser(outputs):
    transform = as_numpy(outputs.dataparser_transform)
    scale = float(outputs.dataparser_scale)
    if (transform.shape != (3, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform, np.eye(4)[:3], atol=1e-7, rtol=0.)
            or not np.isfinite(scale) or abs(scale - 1.) > 1e-7):
        raise ValueError("Candidate world coordinates require identity parser transform and scale 1")
    return transform, scale


def select_frame_indices(count, frame_indices=None, max_images=0):
    """Indices address the loaded split dataset, not transforms.json ordering."""
    if type(max_images) is not int or max_images < 0:
        raise ValueError("max-images must be a nonnegative integer (0 means all)")
    indices = list(range(count)) if frame_indices is None else list(frame_indices)
    if not indices or any(type(i) is not int or not 0 <= i < count for i in indices):
        raise ValueError("frame-indices must be nonempty, nonnegative indices within the selected split")
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate frame-indices")
    return indices[:max_images] if max_images else indices


def source_frames(data_root, metadata, split):
    """Validate all dataset asset paths before Nerfstudio opens them."""
    frames = {}
    for key in ("ply_file_path", "point_cloud_path"):
        if metadata.get(key):
            confined_path(data_root / metadata[key], file=True)
    for frame in metadata["frames"]:
        if not all(key in frame for key in ("camera_id", "group_id", "timestamp_ns")):
            raise ValueError("Source frame requires physical camera, group and timestamp identity")
        image = confined_path(data_root / frame["file_path"], file=True)
        if image in frames:
            raise ValueError("Duplicate source image: %s" % image)
        for key in ("mask_path", "depth_file_path", "lidar_depth_path", "lidar_depth_mask_path"):
            if frame.get(key):
                confined_path(data_root / frame[key], file=True)
        if bool(frame.get("lidar_depth_path")) != bool(frame.get("lidar_depth_mask_path")):
            raise ValueError("LiDAR depth and support mask must be supplied together")
        frames[image] = frame
    memberships, groups = {}, {}
    selected = "val" if split == "dev" else "test"
    for name in ("train", "val", "test"):
        key = name + "_filenames"
        if name in ("train", selected) and not metadata.get(key):
            raise ValueError("Require explicit nonempty %s" % key)
        members = [confined_path(data_root / value, file=True) for value in metadata.get(key, [])]
        if len(members) != len(set(members)) or any(image not in frames for image in members):
            raise ValueError("Duplicate or unknown image in %s" % key)
        memberships[name] = set(members)
        groups[name] = {str(frames[image].get("group_id", frames[image].get("timestamp_ns")))
                        for image in members}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if memberships[a] & memberships[b] or groups[a] & groups[b]:
            raise ValueError("Source splits overlap in images or camera groups: %s/%s" % (a, b))
    return frames, memberships[selected]


def validate_config(config, candidate_degree, split):
    """eval_setup callback: preflight source/checkpoint paths before setup."""
    manager = config.pipeline.datamanager
    parser = manager.dataparser
    model = config.pipeline.model
    for value in (config.output_dir, config.get_checkpoint_dir(), manager.data, parser.data,
                  config.data, config.load_checkpoint, config.load_config, config.load_dir,
                  getattr(model, "lidar_reference_path", None)):
        if value is not None:
            confined_path(value)
    if not getattr(model, "use_inria_surfel_renderer", False) or model.rasterize_mode != "classic":
        raise ValueError("Require actual original INRIA surfel backend config in classic mode")
    if candidate_degree != model.sh_degree or model.sh_degree < 1:
        raise ValueError("Candidate SH degree must match the checkpoint's SH model (degree 1..3)")
    if (parser.auto_scale_poses or parser.orientation_method != "none"
            or parser.center_method != "none" or parser.scale_factor != 1.
            or parser.downscale_factor != 1 or manager.camera_res_scale_factor != 1.):
        raise ValueError("Require fixed world/source cameras: no parser transform, scaling, or image downscale")
    data_path = confined_path(manager.data if manager.data is not None else parser.data)
    transforms = confined_path(data_path if data_path.suffix == ".json" else data_path / "transforms.json", file=True)
    metadata = read_json(transforms)
    if metadata.get("orientation_override", "none") != "none":
        raise ValueError("Dataset overrides source camera orientation")
    require_identity_parser(type("SourceParser", (), {
        "dataparser_transform": metadata.get("applied_transform", np.eye(4)[:3]),
        "dataparser_scale": metadata.get("applied_scale", 1.)})())
    if parser.load_3D_points and not metadata.get("ply_file_path"):
        raise ValueError("Require existing source initialization PLY; parser must not create or rewrite dataset assets")
    source_frames(transforms.parent, metadata, split)
    checkpoint_dir = confined_path(config.get_checkpoint_dir())
    if config.load_step is None:
        checkpoints = list(checkpoint_dir.glob("step-*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError("No actual trained checkpoint: %s" % checkpoint_dir)
        for path in checkpoints:
            confined_path(path, file=True)
    else:
        confined_path(checkpoint_dir / ("step-%09d.ckpt" % config.load_step), file=True)
    return config


def replace_gauss_params(model, parameters, degree, torch):
    """Preserve the loaded model, backend and fixed learned rig correction."""
    if model.training or not model.config.use_inria_surfel_renderer:
        raise ValueError("Require loaded inference model using the original INRIA backend")
    if degree != model.config.sh_degree or degree < 1:
        raise ValueError("Candidate SH degree differs from the loaded SH model")
    if set(parameters) != set(model.gauss_params):
        raise ValueError("Candidate/model Gaussian parameter schemas differ")
    device = model.means.device
    replacements = {key: torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32, device=device),
                                            requires_grad=False)
                    for key, value in parameters.items()}
    for key, value in replacements.items():
        model.gauss_params[key] = value


def fixed_camera_identity(camera, frame, metadata):
    """Verify pose/intrinsics/size and preserve all other camera metadata."""
    expected_pose = np.asarray(frame["transform_matrix"], dtype=float)[:3]
    actual_pose = as_numpy(camera.camera_to_worlds).reshape(3, 4)
    if not np.isfinite(expected_pose).all() or not np.allclose(actual_pose, expected_pose, atol=1e-5, rtol=1e-6):
        raise ValueError("Loaded camera differs from fixed source pose")
    identity = {"camera_to_worlds_ns": actual_pose.tolist()}
    for key, attr in (("fl_x", "fx"), ("fl_y", "fy"), ("cx", "cx"), ("cy", "cy"),
                      ("w", "width"), ("h", "height")):
        actual = float(as_numpy(getattr(camera, attr)).reshape(-1)[0])
        expected = float(metadata[key] if key in metadata else frame[key])
        if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-5, rtol=1e-6):
            raise ValueError("Loaded camera differs from fixed source %s" % key)
        identity[key] = int(actual) if key in ("w", "h") else actual
    if min(identity["w"], identity["h"]) < 1 or identity["w"] * identity["h"] > MAX_IMAGE_PIXELS:
        raise ValueError("Source image exceeds bounded pixel budget")
    camera_id = frame["camera_id"]
    if camera_id not in ("camera_2", "camera_3", "camera_4", "camera_5"):
        raise ValueError("Require actual physical camera_2..camera_5 identity")
    physical_id = int(camera_id.rsplit("_", 1)[1])
    camera.metadata = dict(camera.metadata or {})
    previous = camera.metadata.get("physical_camera_id", physical_id)
    if not np.array_equal(as_numpy(previous).reshape(-1), [physical_id]):
        raise ValueError("Camera physical_camera_id conflicts with source frame")
    camera.metadata["physical_camera_id"] = physical_id
    identity["physical_camera_id"] = physical_id
    return identity


def plane(value, shape, label):
    value = as_numpy(value)
    if value.shape == (*shape, 1):
        value = value[..., 0]
    if value.shape != shape:
        raise ValueError("%s differs from fixed source H/W" % label)
    return value


def score_rgb(gt, pred, valid, ssim_map=None):
    """Same masked PSNR and 7x7 valid-window SSIM as evaluate_reconstruction."""
    import cv2
    from skimage.metrics import structural_similarity

    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError("RGB outputs differ from fixed source H/W/3")
    valid = plane(valid, gt.shape[:2], "RGB mask").astype(bool)
    if not valid.any() or not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("Require finite RGB and nonempty valid source mask")
    pred = pred.clip(0., 1.)
    mse = float(np.mean((gt[valid] - pred[valid]) ** 2))
    result = {"psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
              "rgb_mse": mse, "valid_fraction": float(valid.mean()),
              "rgb_valid_pixels": int(valid.sum()), "ssim_valid_pixels": None,
              "ssim_compared_pixels": 0}
    if min(gt.shape[:2]) >= 7:
        if ssim_map is None:
            _, ssim_map = structural_similarity(gt, pred, channel_axis=-1, data_range=1., full=True)
        # Preserve the existing evaluator's erosion/border and float32 metric
        # conventions exactly for original single-export baseline comparisons.
        eroded = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        result["ssim_compared_pixels"] = int(eroded.sum())
        if eroded.any():
            result["ssim_valid_pixels"] = float(ssim_map.mean(axis=2)[eroded].mean())
    return result


def score_depth(reference, support, depth, alpha, valid):
    """Coverage-conditioned diagnostics plus a fixed-denominator depth score.

    Missing predictions have unit relative error (equivalent to zero depth).
    The selection score clips all relative absolute errors at one, so removing
    a covered prediction can never improve it. Conditional metric values are
    null with zero comparisons, never zero. Source support is never reprojected.
    """
    shape = valid.shape
    reference = plane(reference, shape, "LiDAR depth")
    support = plane(support, shape, "LiDAR support")
    depth = plane(depth, shape, "Rendered depth")
    alpha = plane(alpha, shape, "Accumulation")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1 + 1e-6)):
        raise ValueError("Renderer alpha must be finite probabilities")
    eligible = valid & np.isfinite(reference) & (reference > 0) & np.isfinite(support) & (support > 0)
    compared = eligible & (alpha > .5) & np.isfinite(depth) & (depth > 0)
    denominator, count = int(eligible.sum()), int(compared.sum())
    result = {"map_depth_eligible_pixels": denominator, "map_depth_compared_pixels": count,
              "map_depth_missing_pixels": denominator - count,
              "map_depth_coverage_fraction": count / denominator if denominator else None,
              "map_projected_depth_median_abs_m": None, "map_projected_depth_rmse_m": None,
              "map_depth_capped_relative_error_all_supported": None}
    if denominator:
        capped = np.ones(denominator, np.float64)
        if count:
            error = depth[compared] - reference[compared]
            result["map_projected_depth_median_abs_m"] = float(np.median(np.abs(error)))
            result["map_projected_depth_rmse_m"] = float(np.sqrt(np.mean(error ** 2)))
            capped[compared[eligible]] = np.minimum(np.abs(error) / reference[compared], 1.)
        result["map_depth_capped_relative_error_all_supported"] = float(capped.mean())
    return result


def alpha_coverage(alpha, valid):
    alpha = plane(alpha, valid.shape, "Accumulation")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1 + 1e-6)):
        raise ValueError("Renderer alpha must be finite probabilities")
    covered = alpha > .5
    return {"alpha_gt_0_5_pixel_fraction": float(covered.mean()),
            "alpha_gt_0_5_valid_pixel_fraction": float(covered[valid].mean()),
            "alpha_gt_0_5_valid_pixels": int((covered & valid).sum())}


def save_panels(out, index, gt, pred, valid, cv2):
    """Exclusive files: REAL GT / actual candidate / absolute RGB error."""
    displayed = [np.where(valid[..., None], rgb, 0.) for rgb in (gt, pred)]
    panels = [cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
              for rgb in displayed]
    error = np.abs(displayed[0] - displayed[1]).mean(axis=2)
    colored = cv2.applyColorMap(np.clip(error * 500, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    names = []
    for label, panel in zip(("gt", "candidate", "error"), [*panels, colored]):
        name = "%03d_%s.png" % (index, label)
        ok, encoded = cv2.imencode(".png", panel)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        with confined_path(out / name).open("xb") as stream:
            stream.write(encoded.tobytes())
        names.append(name)
    comparison = [p.copy() for p in [*panels, colored]]
    for label, panel in zip(("REAL HELD-OUT", "CANDIDATE GAUSSIAN", "ABS ERROR"), comparison):
        cv2.putText(panel, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
    name = "%03d_comparison.jpg" % index
    ok, encoded = cv2.imencode(".jpg", np.hstack(comparison))
    if not ok:
        raise RuntimeError("Comparison encoding failed")
    with confined_path(out / name).open("xb") as stream:
        stream.write(encoded.tobytes())
    return dict(zip(("gt_png", "candidate_png", "error_png", "comparison_jpg"), [*names, name]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Actual saved Nerfstudio config.yml")
    parser.add_argument("--ply", type=Path, help="Trained GS candidate; omit to evaluate actual checkpoint parameters")
    parser.add_argument('--dataset-override', type=Path, help='Explicit fixed evaluation dataset; never modifies checkpoint cameras or parameters')
    parser.add_argument('--export-ply', action='store_true', help='Export evaluated checkpoint parameters')
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", required=True, type=Path, help="New project-confined directory")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all selected frames")
    parser.add_argument("--frame-indices", nargs="+", help="Split dataset indices, space/comma separated (0-based)")
    parser.add_argument('--max-gaussians', type=int, default=MAX_GAUSSIANS,
                        help='Explicit full-map inference allocation budget, at most 6000000')
    parser.add_argument('--max-ply-bytes', type=int, default=MAX_PLY_BYTES,
                        help='Explicit full-map source byte budget, at most 2GiB')
    parser.add_argument('--native-visibility', action='store_true',
                        help='Exact native geometry prepass over all CPU source records before per-camera inference')
    parser.add_argument('--active-gaussian-budget', type=int, default=1500000)
    parser.add_argument('--geometry-chunk-records', type=int, default=200000)
    parser.add_argument('--acceptance-plan', type=Path)
    parser.add_argument('--block-id')
    parser.add_argument('--save-float-arrays', action='store_true', help='Preserve unquantized RGB/alpha/depth for exact rescoring')
    parser.add_argument('--direction-support', type=Path, help='Hash-bound TRAIN observation-side metadata; explicitly changes the model representation')
    parser.add_argument("--renderer-integration-proof", type=Path,
                        help="Explicit project-local real-model integration evidence for this evaluator revision")
    args = parser.parse_args(argv)
    config_path = confined_path(args.config, file=True)
    if config_path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Saved config exceeds bounded read budget")
    out_path = confined_path(args.output)
    if out_path.exists():
        raise FileExistsError("Output already exists; choose a new directory: %s" % out_path)
    indices = None
    if args.frame_indices is not None:
        indices = [int(value) for token in args.frame_indices for value in token.split(",")]
    if args.max_images < 0:
        raise ValueError("max-images must be nonnegative")
    if not 1 <= args.active_gaussian_budget <= 6000000:
        raise ValueError('Active inference budget must contain 1..6000000 records')
    if args.ply is None and args.native_visibility:
        raise ValueError('Native whole-PLY visibility requires --ply')
    if args.ply is not None:
        parameters, candidate = read_candidate(args.ply, max_gaussians=args.max_gaussians,
            max_ply_bytes=args.max_ply_bytes, extended_cpu_budget=args.native_visibility)
        candidate_state = _state(Path(candidate['path']))
    else:
        parameters, candidate, candidate_state = None, {'sh_degree': 3}, None
    config_sha = sha256_file(config_path)
    direction_support = None
    if args.direction_support:
        if args.ply is None or not args.native_visibility:
            raise ValueError('Direction support requires an immutable PLY and native visibility prepass')
        from direction_support import load_direction_support
        direction_support = load_direction_support(args.direction_support, candidate, parameters)

    # Reuse the existing evaluator's actual eval_setup/rendering stack. All
    # implicit caches are confined before either Torch or Nerfstudio is imported.
    from orbvi_gaussian.training import confined_environment
    os.environ.update(confined_environment())
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"]
    import evaluate_reconstruction as existing
    from nerfstudio.cameras.cameras import CameraType
    torch, cv2 = existing.torch, existing.cv2

    setup_sources = {}

    def preflight_config(config):
        if args.dataset_override:
            override = confined_path(args.dataset_override)
            config.pipeline.datamanager.data = override
            config.pipeline.datamanager.dataparser.data = override
        validate_config(config, candidate["sh_degree"], args.split)
        manager = config.pipeline.datamanager
        data = confined_path(manager.data if manager.data is not None else manager.dataparser.data)
        transforms = data if data.suffix == ".json" else data / "transforms.json"
        setup_sources["transforms_sha256"] = sha256_file(transforms)
        return config

    config, pipeline, checkpoint, step = existing.eval_setup(
        config_path, test_mode="val" if args.split == "dev" else "test",
        update_config_callback=preflight_config)
    checkpoint = confined_path(checkpoint, file=True)
    backend_default_load_step = int(pipeline.model.step)
    pipeline.model.step = int(step)
    if sha256_file(config_path) != config_sha:
        raise ValueError("Source config changed during setup")
    dataset = pipeline.datamanager.eval_dataset
    if args.native_visibility and getattr(pipeline.model.config,'optimize_shared_rig',False):
        raise ValueError('Exact native prepass currently requires the fixed source rig')
    transform, scale = require_identity_parser(pipeline.datamanager.train_dataparser_outputs)
    require_identity_parser(dataset._dataparser_outputs)
    data_path = confined_path(config.pipeline.datamanager.data)
    transforms_path = confined_path(data_path if data_path.suffix == ".json" else data_path / "transforms.json", file=True)
    metadata = read_json(transforms_path)
    if sha256_file(transforms_path) != setup_sources["transforms_sha256"]:
        raise ValueError("Source transforms changed during setup")
    data_root = transforms_path.parent
    by_image, expected_images = source_frames(data_root, metadata, args.split)
    actual_images = [confined_path(p, file=True) for p in dataset.image_filenames]
    if len(actual_images) != len(set(actual_images)) or set(actual_images) != expected_images:
        raise ValueError("Loaded evaluation dataset differs from the explicit source split")
    if (np.any(as_numpy(dataset.cameras.camera_type) != CameraType.PERSPECTIVE.value)
            or (dataset.cameras.distortion_params is not None
                and np.any(as_numpy(dataset.cameras.distortion_params) != 0))):
        raise ValueError("Require actual rectified source perspective cameras")
    selected = select_frame_indices(len(dataset), indices, args.max_images)
    acceptance_rows, acceptance_info = {}, None
    if args.acceptance_plan:
        from acceptance_evaluation import acceptance_selection
        if not args.block_id or indices is not None:
            raise ValueError('Acceptance evaluation needs block-id and forbids manual frame-indices')
        acceptance_rows, acceptance_info = acceptance_selection(confined_path(args.acceptance_plan, file=True),
            args.block_id, args.split, actual_images, by_image, transforms_path)
        selected = list(acceptance_rows)
        if args.max_images:
            selected = selected[:args.max_images]
    original_count = len(pipeline.model.means)
    if args.ply is None:
        candidate = {'gaussians': original_count, 'sh_degree': pipeline.model.config.sh_degree,
                     'path': str(checkpoint), 'sha256': sha256_file(checkpoint), 'kind': 'trained_checkpoint'}
    if args.ply is not None and not args.native_visibility:
        replace_gauss_params(pipeline.model, parameters, candidate["sh_degree"], torch)
    if pipeline.model.means.device.type != "cuda":
        raise ValueError("Original INRIA candidate evaluation requires CUDA")
    out = existing.new_output_dir(out_path)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for index in selected:
        frame = by_image[actual_images[index]]
        batch = dataset[index]
        gt = as_numpy(batch["image"])
        gt = gt.astype(np.float32) / 255 if gt.dtype == np.uint8 else gt.astype(np.float32)
        valid = np.ones(gt.shape[:2], bool)
        if "mask" in batch:
            mask = plane(batch["mask"], gt.shape[:2], "Source RGB mask")
            valid = np.isfinite(mask) & (mask > 0)
        masks = dataset._dataparser_outputs.mask_filenames
        source_mask = confined_path(data_root / frame["mask_path"], file=True) if frame.get("mask_path") else None
        if (confined_path(masks[index], file=True) if masks else None) != source_mask:
            raise ValueError("Loaded RGB mask differs from the fixed source mask")
        masks_by_protocol = None
        if acceptance_info:
            from acceptance_evaluation import protocol_masks
            masks_by_protocol = protocol_masks(valid, acceptance_rows[index], cv2)
            valid = masks_by_protocol['ego_v2']
        camera = dataset.cameras[index:index + 1].to(pipeline.device)
        identity = fixed_camera_identity(camera, frame, metadata)
        if gt.shape != (identity["h"], identity["w"], 3):
            raise ValueError("Source RGB differs from the fixed camera H/W/3")
        start = time.monotonic()
        visibility = None
        if args.native_visibility:
            from native_gaussian_visibility import native_visible_indices
            active, visibility = native_visible_indices(parameters,camera,pipeline.device,args.geometry_chunk_records,
                backend=getattr(pipeline.model.config,'inria_backend','original'))
            if direction_support is not None:
                from direction_support import filter_direction_indices
                active, side_info = filter_direction_indices(parameters, direction_support,
                    as_numpy(camera.camera_to_worlds)[0, :3, 3], active)
                visibility['direction_visibility'] = side_info
            if len(active)>args.active_gaussian_budget:
                raise ValueError('Exact visible records exceed GPU budget; do not silently sample')
            subset = {key:value[active] for key,value in parameters.items()}
            replace_gauss_params(pipeline.model,subset,candidate['sh_degree'],torch)
        with torch.inference_mode():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        pred = as_numpy(outputs["rgb"])
        ssim_map = None
        if masks_by_protocol is not None:
            from skimage.metrics import structural_similarity
            _, ssim_map = structural_similarity(gt, pred.clip(0., 1.), channel_axis=-1, data_range=1., full=True)
        metrics = score_rgb(gt, pred, valid, ssim_map)
        metrics.update(alpha_coverage(outputs["accumulation"], valid))
        row = {"dataset_index": index, "file_path": frame["file_path"],
               "group_id": frame["group_id"], "camera_id": frame["camera_id"],
               "timestamp_ns": frame["timestamp_ns"], "camera": identity,
               "source_image_path": str(actual_images[index]),
               "source_mask_path": str(source_mask) if source_mask else None,
               "render_seconds": seconds, **metrics}
        if visibility is not None: row['native_visibility'] = visibility
        for key in ("view_id", "source_file_path", "source_header_ns", "source_bag_ns"):
            if key in frame:
                row[key] = frame[key]
        if frame.get("lidar_depth_path"):
            depth_path = confined_path(data_root / frame["lidar_depth_path"], file=True)
            support_path = confined_path(data_root / frame["lidar_depth_mask_path"], file=True)
            reference = np.load(depth_path, allow_pickle=False)
            if reference.dtype.kind != "f":
                raise ValueError("LiDAR depth must be floating camera-z metres")
            support = cv2.imread(str(support_path), cv2.IMREAD_GRAYSCALE)
            if support is None:
                raise ValueError("Cannot decode fixed source LiDAR support mask")
            if outputs.get("depth") is None:
                raise ValueError("Original renderer omitted required LiDAR comparison depth")
            row.update(score_depth(reference, support, as_numpy(outputs["depth"]),
                                   as_numpy(outputs["accumulation"]), valid))
            row.update(source_lidar_depth_path=str(depth_path), source_lidar_support_path=str(support_path))
        else:
            row["depth_status"] = "source_frame_has_no_lidar_reference"
        if masks_by_protocol is not None:
            row['protocols'] = {}
            for name, eligible in masks_by_protocol.items():
                values = score_rgb(gt, pred, eligible, ssim_map)
                values.update(alpha_coverage(outputs['accumulation'], eligible))
                if frame.get('lidar_depth_path'):
                    values.update(score_depth(reference, support, as_numpy(outputs['depth']),
                                              as_numpy(outputs['accumulation']), eligible))
                row['protocols'][name] = values
            row['acceptance_mask_path'] = acceptance_rows[index]['mask_path']
            row['block_id'] = args.block_id
        if args.save_float_arrays:
            np.savez(out / ('%03d_float.npz' % index), rgb=pred,
                                alpha=as_numpy(outputs['accumulation']), depth=as_numpy(outputs['depth']))
        row["artifacts"] = save_panels(out, index, gt, pred.clip(0., 1.), valid, cv2)
        records.append(row)
        print(json.dumps({"dataset_index": index, "psnr_db": row["psnr_db"],
                          "alpha_coverage": row["alpha_gt_0_5_valid_pixel_fraction"]}), flush=True)
    ssim_values = [r["ssim_valid_pixels"] for r in records if r["ssim_valid_pixels"] is not None]
    denominator = sum(r.get("map_depth_eligible_pixels", 0) for r in records)
    compared = sum(r.get("map_depth_compared_pixels", 0) for r in records)
    depth_score = sum(r["map_depth_capped_relative_error_all_supported"] * r["map_depth_eligible_pixels"]
                      for r in records if r.get("map_depth_eligible_pixels", 0))
    from orbvi_gaussian.inria_surfel_backend import model_inria_backend_provenance
    renderer_provenance=model_inria_backend_provenance(pipeline.model.config)
    report = {"status": "evaluated", "split": args.split, "config": str(config_path),
              "renderer_provenance": renderer_provenance,
              "config_sha256": config_sha, "checkpoint": str(checkpoint), "step": int(step),
              "checkpoint_is_template": args.ply is not None,
              "parameter_origin": "All Gaussian parameters replaced by source_ply; template checkpoint supplies model/backend and fixed source camera setup" if args.ply is not None else 'trained_checkpoint',
              "source_ply": candidate if args.ply is not None else None,
              "source_ply_sha256": candidate["sha256"] if args.ply is not None else None,
              "checkpoint_sha256": sha256_file(checkpoint),
              "transforms": str(transforms_path), "transforms_sha256": setup_sources["transforms_sha256"],
              "images": len(records), "split_images": len(dataset), "frame_indices": selected,
              "frames": records, "original_checkpoint_gaussians": original_count,
              "gaussians": candidate["gaussians"], "backend": "hbb1-original-2dgs" if getattr(pipeline.model.config,'inria_backend','original')=='original' else 'hbb1-aa-depth-vjp-2dgs',
              "inference_model_step": int(pipeline.model.step),
              "backend_default_load_step": backend_default_load_step,
              "inference_step_protocol": "Actual template checkpoint step; assembled PLY training provenance is separate",
              "inference_sh_degree": min(pipeline.model.step // pipeline.model.config.sh_degree_interval,
                                          pipeline.model.config.sh_degree),
              "psnr_db_mean": float(np.mean([r["psnr_db"] for r in records])),
              "ssim_valid_mean": float(np.mean(ssim_values)) if ssim_values else None,
              "ssim_scored_images": len(ssim_values),
              "alpha_gt_0_5_pixel_fraction_mean": float(np.mean([r["alpha_gt_0_5_pixel_fraction"] for r in records])),
              "map_depth_eligible_pixels": denominator, "map_depth_compared_pixels": compared,
              "map_depth_missing_pixels": denominator - compared,
              "map_depth_coverage_fraction": compared / denominator if denominator else None,
              "map_depth_capped_relative_error_all_supported": depth_score / denominator if denominator else None,
              "world_parser_transform": transform.tolist(), "world_parser_scale": scale,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "selection_protocol": "Select on DEV, then final TEST once; no fitting or tuning in this evaluator",
              "depth_protocol": "Fixed source camera-z metres/support/mask; alpha>0.5 and finite positive predictions; "
                                "conditional errors plus fixed-support mean min(abs(error)/reference,1); missing=1",
              "geometry_protocol": "Conditioned on same LIO/map used by initialization; not independent measurement GT",
              "bounds": {"max_gaussians": args.max_gaussians, "max_ply_bytes": args.max_ply_bytes,
                         "native_visibility":args.native_visibility,"active_gaussian_budget":args.active_gaussian_budget,
                         "max_image_pixels": MAX_IMAGE_PIXELS}}
    if ((args.ply is not None and _state(Path(candidate["path"])) != candidate_state) or sha256_file(config_path) != config_sha
            or sha256_file(transforms_path) != setup_sources["transforms_sha256"]):
        raise ValueError("Source PLY/config/transforms changed during evaluation")
    if getattr(pipeline.model.config, "optimize_shared_rig", False):
        report["shared_rig_correction_ns_local"] = as_numpy(pipeline.model.shared_rig.correction_matrices()).tolist()
        report["shared_rig_effective_delta_rotfirst"] = as_numpy(pipeline.model.shared_rig.effective_delta()).tolist()
        report["camera_protocol"] = "Fixed checkpoint rig correction applied by loaded backend; source depth/support unchanged"
    if args.renderer_integration_proof is not None:
        proof_path = confined_path(args.renderer_integration_proof, file=True)
        if report['backend'] != 'hbb1-aa-depth-vjp-2dgs':
            raise ValueError('Explicit integration proof requires the AA depth VJP backend')
        report['renderer_integration_proof'] = {
            'path': str(proof_path), 'sha256': sha256_file(proof_path)}
        from measure_surface_quality import validate_renderer_provenance
        validate_renderer_provenance(report)
    preparation = confined_path(data_root / "preparation.json")
    if acceptance_info:
        from acceptance_evaluation import summarize_protocols
        report['acceptance'] = acceptance_info
        report['acceptance_plan_id'] = acceptance_info['plan_id']
        report['metric_domain'] = 'float_renderer'
        report['protocols'] = summarize_protocols(records)
        report['primary_protocol'] = 'ego_v2'
    if direction_support is not None:
        from direction_support import verify_direction_support_unchanged
        verify_direction_support_unchanged(direction_support)
        report['direction_support'] = direction_support['info']
        report['representation_id'] = hashlib.sha256((candidate['sha256'] + ':' + direction_support['info']['sha256']).encode()).hexdigest()
        report['model_representation'] = 'PLY plus fixed TRAIN observation-side support; original native 2DGS kernels'
    if preparation.is_file():
        report["calibration_status"] = read_json(preparation).get("calibration_verified")
    if args.export_ply:
        if args.ply is not None:
            raise ValueError('Export requires actual checkpoint mode, not substituted PLY')
        report['export'] = existing.export_ply(pipeline.model, out / 'gaussians.ply')
    existing.write_json(out / "metrics.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, allow_nan=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)

#!/usr/bin/env python3
"""Bounded CPU measurements of native float renders and complete Gaussian PLYs.

Example (DEV only; creates a new report, never rewrites inputs)::
  .venv-train/bin/python -B scripts/measure_surface_quality.py \
    --metrics runs/fidelity_v2/pilot015_original_smoke/metrics.json \
    --reference data/garage_surface_v1/block_015/lidar_reference.npz \
    --acceptance-plan reports/acceptance_plan_v1.json \
    --output reports/surface_quality_v2_pilot015.json

Add --baseline-metrics for paired, fixed-support comparisons. --ply alone also
works. --checkpoint-proof accepts the CPU verifier's exact checkpoint/export
evidence for native trained renders. Geometry is consistency with the initialization LiDAR, NOT independent
ground truth. Floor/wall labels are geometric proxies, not verified semantics.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from package_garage import inspect_ply, scan_ply, _state, read_json

MIB = 1024**2
LABELS = ("floor", "wall", "other", "unknown")
COUNTS = ("denominator", "correct", "missing_alpha", "invalid_depth", "wrong_depth", "unknown_prediction")
DIRECTION_COUNTS = ("too_near", "too_far")
DEFAULT_SETTINGS = {"pixel_stride": 1, "depth_tolerance_m": .15, "reference_match_max_m": .20,
                    "render_pixel_center": .5, "reference_pixel_center": .5,
                    "floor_abs_normal_z_min": float(np.cos(np.deg2rad(30))), "wall_abs_normal_z_max": .30,
                    "floor_below_camera_min_m": .30, "floor_below_camera_max_m": 3., "wall_height_radius_m": 3.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in iter(lambda: stream.read(8*MIB), b""):
            digest.update(raw)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def memory_check(estimate, memory_mib, label):
    require(estimate <= memory_mib*MIB,
            f"{label} conservative working-memory estimate {estimate/MIB:.1f} MiB exceeds {memory_mib} MiB; reduce scope or explicitly raise --memory-mib")


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0, "p50_m": None, "p95_m": None, "max_m": None}
    require(np.isfinite(values).all(), "Nonfinite distance statistics")
    return {"count": int(values.size), "p50_m": float(np.quantile(values, .5)),
            "p95_m": float(np.quantile(values, .95)), "max_m": float(values.max())}


def signed_stats(values):
    values = np.asarray(values)
    require(np.isfinite(values).all(), "Nonfinite signed depth statistics")
    return {"count": int(values.size), **{name: float(np.quantile(values, q)) if values.size else None
            for name, q in (("p05_m", .05), ("p50_m", .5), ("p95_m", .95))},
            "min_m": float(values.min()) if values.size else None, "max_m": float(values.max()) if values.size else None}


def exact_voxel_counts(cells):
    """In-place lexicographic sort: exact occupied-voxel counts, no hash collisions."""
    require(cells.dtype == np.int32 and cells.ndim == 2 and cells.shape[1] == 3,
            "Expected int32 XYZ cells")
    structured = cells.view(np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])).reshape(-1)
    structured.sort(kind="quicksort", order=("x", "y", "z"))
    starts = np.flatnonzero(np.r_[True, np.any(cells[1:] != cells[:-1], axis=1)])
    return np.diff(np.r_[starts, len(cells)])


def measure_ply(path, *, memory_mib=1024, max_gaussians=12_000_000, max_ply_bytes=4*1024**3):
    """Read every record; retain only opacity, first two scale axes and voxel IDs."""
    path = Path(path).resolve()
    state = _state(path)
    require(state[2] <= max_ply_bytes, "PLY exceeds --max-ply-bytes")
    with path.open("rb") as stream:
        spec = inspect_ply(stream)
    count, names = spec["count"], spec["names"]
    require(count <= max_gaussians, "PLY exceeds --max-gaussians; no implicit sampling")
    require(state[2] == len(spec["header"]) + count*spec["stride"], "Incomplete/trailing PLY payload")
    estimate = 96*count + 96*MIB
    memory_check(estimate, memory_mib, "PLY")
    opacity = np.empty(count, np.float32)
    tangent = np.empty((count, 2), np.float64)
    cells = np.empty((count, 3), np.int32)
    cursor, below_half = 0, 0
    xyz_cols = [names.index(k) for k in ("x", "y", "z")]
    scale_cols = [names.index("scale_%d" % i) for i in range(3)]
    rot_cols = [names.index("rot_%d" % i) for i in range(4)]

    def consume(raw):
        nonlocal cursor, below_half
        rows = np.frombuffer(raw, dtype="<f4").reshape(-1, len(names))
        end = cursor + len(rows)
        require(np.allclose(np.linalg.norm(rows[:, rot_cols].astype(np.float64), axis=1), 1, atol=1e-4, rtol=0),
                "PLY rotation must be a unit quaternion")
        with np.errstate(over="ignore", under="ignore"):
            scales = np.exp(rows[:, scale_cols])
        require(np.isfinite(scales).all() and (scales > 0).all(), "Invalid activated scale")
        # Stable logistic; logits themselves may be any finite float32 value.
        logits = rows[:, names.index("opacity")].astype(np.float64)
        below_half += int(np.count_nonzero(logits < 0))
        opacity[cursor:end] = np.exp(-np.logaddexp(0, -logits))
        # Measure the mathematical radius from the stored log. CPU NumPy and
        # CUDA float32 exp can differ by an ULP at a hard cap (e.g. 0.12 m).
        # Preserve float64 here so that rounding is not reported as an overshoot.
        tangent[cursor:end] = np.exp(rows[:, scale_cols[:2]].astype(np.float64))
        voxel = np.floor(rows[:, xyz_cols].astype(np.float64)/.02)
        require((voxel >= np.iinfo(np.int32).min).all() and (voxel <= np.iinfo(np.int32).max).all(),
                "2cm voxel integer coordinates exceed int32")
        cells[cursor:end] = voxel
        cursor = end

    scan = scan_ply({"path": path, "spec": spec, "state": state}, 16*MIB, full=consume)
    major = tangent.max(axis=1)
    result = {"status": "known", "path": str(path), "sha256": scan["sha256"], "gaussians": count,
              "opacity_median": float(np.median(opacity)), "opacity_mean": float(opacity.mean(dtype=np.float64)),
              "opacity_frac_lt_0_5": float(below_half/count),
              "tangential_radius_p95_m": float(np.quantile(major, .95)),
              "tangential_radius_max_m": float(major.max()),
              "tangential_first_two_axes_p95_m": np.quantile(tangent, .95, axis=0).tolist(),
              "tangential_first_two_axes_max_m": tangent.max(axis=0).tolist(),
              "tangential_pooled_axes_p95_m": float(np.quantile(tangent, .95)),
              "working_memory_estimate_bytes": estimate,
              "definitions": {"opacity": "sigmoid(opacity logit), all PLY records, no pruning",
                 "tangential_radius": "Per Gaussian max(exp(float64(stored_float32(scale_0))),exp(float64(stored_float32(scale_1)))); p95 across Gaussians. Mathematical radius, not 3-sigma footprint or diameter; avoids CPU/CUDA float32 exp ULP mismatch at a cap. Axis 2 is never substituted by sorting.",
                 "voxel_multiplicity": "floor(world_xyz/0.02), origin=(0,0,0); exact number of Gaussian CENTRES per occupied voxel, each occupied voxel weighted once. No empty voxels, splat-footprint rasterization or Gaussian weighting."}}
    del tangent, major, opacity
    counts = exact_voxel_counts(cells)
    result.update(voxel_size_m=.02, occupied_voxels=int(len(counts)),
                  voxel_2cm_multiplicity_median=float(np.median(counts)),
                  voxel_2cm_multiplicity_mean=float(count/len(counts)),
                  voxel_2cm_multiplicity_p95=float(np.quantile(counts, .95)),
                  voxel_2cm_multiplicity_max=int(counts.max()),
                  voxel_2cm_singleton_fraction=float(np.mean(counts == 1)))
    return result


def bounded_npz(path, max_bytes):
    """Check decompressed bytes AND declared NPY shapes before allocating arrays."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len({e.filename for e in entries}) == len(entries), "Duplicate NPZ member")
        require(sum(e.file_size for e in entries) <= max_bytes, "NPZ exceeds decompressed byte budget")
        for entry in entries:
            require(entry.filename.endswith(".npy"), "Unexpected NPZ member")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "Unsupported NPY version")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                size = int(dtype.itemsize)
                for dim in shape:
                    require(type(dim) is int and dim >= 0, "Invalid NPY dimension")
                    size *= dim
                require(not dtype.hasobject and size <= max_bytes and size + stream.tell() == entry.file_size,
                        "NPY shape/payload exceeds bounded allocation or contains objects")
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def check_camera(camera):
    pose = np.asarray(camera["camera_to_worlds_ns"], dtype=np.float64)
    require(pose.shape == (3, 4) and np.isfinite(pose).all(), "Expected finite NS camera-to-world 3x4")
    require(np.allclose(pose[:, :3].T@pose[:, :3], np.eye(3), atol=2e-5, rtol=0)
            and np.isclose(np.linalg.det(pose[:, :3]), 1, atol=2e-5), "Camera pose must be rigid; no scale")
    require(all(np.isfinite(camera[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
            and min(camera["fl_x"], camera["fl_y"]) > 0, "Invalid camera intrinsics")
    require(all(type(camera[k]) is int and camera[k] > 0 for k in ("h", "w")), "Invalid camera dimensions")
    return pose


def unproject(camera, flat_indices, depths, pixel_center):
    """CV positive camera-z metres -> NS right/up/back -> world camera_init."""
    pose = check_camera(camera)
    y, x = np.divmod(np.asarray(flat_indices), camera["w"])
    rays = np.column_stack(((x+pixel_center-camera["cx"])/camera["fl_x"],
                            -(y+pixel_center-camera["cy"])/camera["fl_y"], -np.ones(len(x))))
    return (rays * np.asarray(depths)[:, None])@pose[:, :3].T + pose[:, 3]


def same_camera(a, b):
    check_camera(a)
    check_camera(b)
    return (all(a[k] == b[k] for k in ("w", "h", "physical_camera_id"))
            and all(np.allclose(a[k], b[k], atol=2e-5, rtol=1e-6)
                    for k in ("fl_x", "fl_y", "cx", "cy", "camera_to_worlds_ns")))


def frame_key(row):
    return str(row["group_id"]), row["camera_id"]


def validate_renderer_provenance(report, pin=None):
    """Read/hash artifacts only: never import torch, a CUDA extension or its adapter."""
    backend, provenance = report.get("backend"), report.get("renderer_provenance")
    if backend == "hbb1-original-2dgs":
        require(provenance is None or provenance.get("variant") == "original",
                "Original backend mislabeled with a different renderer variant")
        return {"status": "original_backend", "backend": backend, "assets": {}}
    require(backend == "hbb1-aa-depth-vjp-2dgs", "Require supported native float-render evaluator")
    require(isinstance(provenance, dict) and provenance.get("schema_version") == 1
            and provenance.get("variant") == "aa-depth-vjp"
            and provenance.get("module") == "diff_surfel_rasterization_aa_depth_vjp",
            "AA depth backend requires explicit schema-1 renderer provenance")
    require(provenance.get("forward_contract") == "original ray-plane depth and low-pass coverage; no center-depth fallback",
            "AA depth backend changed forward measurement contract")
    assets, snapshots = {}, {}
    def local(value):
        require(isinstance(value, str) and value, "Renderer asset path missing")
        return (ROOT/Path(value)).resolve()
    def verify(value, digest=None):
        path = local(value)
        if pin:
            pin(path)
        snapshots.setdefault(path, _state(path))
        actual = file_sha256(path)
        require(digest is None or actual == digest, "Renderer artifact SHA changed: " + str(path))
        assets[str(path)] = actual
        return path
    validation = provenance.get("validation")
    require(isinstance(validation, dict) and all(validation.get(k) is True for k in (
        "forward_source_byte_identical", "tested_forward_arrays_bitwise_equal",
        "tested_rgb_alpha_parameter_gradients_bitwise_equal")), "AA depth forward/gradient validation missing")
    for name in ("adapter", "wrapper", "extension"):
        require(isinstance(provenance.get(name+"_sha256"), str), "Renderer artifact SHA missing")
        verify(provenance[name+"_path"], provenance[name+"_sha256"])
    chain = {}
    for name, path_key, sha_key in (("validation", "validation_path", "validation_sha256"),
            ("fd", "finite_difference_report", "finite_difference_report_sha256"),
            ("manifest", "build_manifest_path", "build_manifest_sha256")):
        require(isinstance(validation.get(sha_key), str), "Renderer validation chain SHA missing")
        chain[name] = read_json(verify(validation[path_key], validation[sha_key]))
    v, fd, manifest = (chain[k] for k in ("validation", "fd", "manifest"))
    require(v.get("status") == "LOCAL_CUDA_DEPTH_VJP_FINITE_DIFFERENCE_PASS"
            and v.get("forward_bitwise_equal") is True and v.get("rgb_alpha_parameter_gradients_bitwise_equal") is True
            and fd.get("candidate_status") == "PASS" and fd.get("gpu_used") is True
            and fd.get("forward_arrays_bitwise_equal") is True and fd.get("rgb_alpha_parameter_gradients_bitwise_equal") is True,
            "AA depth native finite-difference/forward evidence did not pass")
    require(local(v.get("report")) == local(validation["finite_difference_report"])
            and v.get("report_sha256") == validation["finite_difference_report_sha256"], "Renderer validation/FD report differs")
    require(v.get("candidate_extension_sha256") == provenance["extension_sha256"] ==
            fd.get("candidate", {}).get("extension_sha256") == manifest.get("candidate_extension_sha256")
            and local(manifest.get("candidate_extension")) == local(provenance["extension_path"]),
            "Renderer validated binary identity differs")
    require(local(fd["candidate"].get("extension_path")) == local(provenance["extension_path"])
            and fd["candidate"].get("module") == provenance["module"], "Renderer FD candidate path/module differs")
    require(v.get("original_extension_sha256") == validation.get("original_extension_sha256") ==
            fd.get("original", {}).get("extension_sha256"), "Renderer original control binary differs")
    verify(fd["original"]["extension_path"], v["original_extension_sha256"])
    verify(fd["original"]["wrapper_path"], provenance["wrapper_sha256"])
    required = {"cuda_rasterizer/forward.cu", "cuda_rasterizer/backward.cu", "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/auxiliary.h", "rasterize_points.cu", "rasterize_points.h", "ext.cpp", "setup.py"}
    declared, built = provenance.get("source_sha256"), manifest.get("candidate_source_sha256")
    require(isinstance(declared, dict) and isinstance(built, dict) and required <= set(declared) <= set(built)
            and len(built) <= 32 and all(built[k] == value for k, value in declared.items()),
            "Renderer source manifest incomplete or mismatched")
    directory = local(provenance.get("source_directory"))
    require(manifest.get("package") == provenance["module"] and manifest.get("forward_byte_identical") is True
            and manifest.get("installed_original_replaced") is False, "Renderer build contract differs")
    for name, digest in built.items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory), "Renderer source path escapes directory")
        verify(str(path), digest)
    forward_sha = declared["cuda_rasterizer/forward.cu"]
    require(forward_sha == v.get("forward_source_sha256") == manifest.get("original_source_sha256", {}).get("cuda_rasterizer/forward.cu"),
            "Renderer forward source identity differs")
    verify(str(ROOT/"runtime/inria2dgs_src/cuda_rasterizer/forward.cu"), forward_sha)
    # Frozen real model integration evidence also binds the adapter, model and
    # evaluators, beyond the standalone rasterizer finite-difference fixture.
    if "renderer_integration_proof" in report:
        proof = report["renderer_integration_proof"]
        require(isinstance(proof, dict) and isinstance(proof.get("path"), str) and proof["path"]
                and isinstance(proof.get("sha256"), str) and len(proof["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in proof["sha256"]),
                "Renderer integration proof requires path and SHA256")
        chosen = local(proof["path"])
        require(chosen != ROOT.resolve() and chosen.is_relative_to(ROOT.resolve()),
                "Renderer integration proof must remain project-local")
        integration_path = verify(str(chosen), proof["sha256"])
    else:
        integration_path = verify(str(ROOT/"reports/inria_backend_integration_gpu_v1.json"))
    integration = read_json(integration_path)
    require(integration.get("status") == "PASS" and integration.get("gpu_used") is True and
            all(integration.get(k) is True for k in ("parameters_restored_bitwise", "forward_model_images_bitwise_equal",
                                                     "rgb_alpha_parameter_gradients_bitwise_equal")),
            "Renderer actual model integration evidence incomplete")
    candidates = [c for c in integration.get("cases", []) if c.get("variant") == "aa-depth-vjp"]
    require(len(candidates) == 1 and candidates[0].get("renderer_provenance") == provenance,
            "Renderer differs from validated model integration identity")
    integration_sources = integration.get("source_sha256")
    require(isinstance(integration_sources, dict) and {"src/orbvi_gaussian/inria_surfel_backend.py",
            "src/orbvi_gaussian/scale_annealing.py",
            "src/orbvi_gaussian/lidar_splatfacto.py", "scripts/evaluate_reconstruction.py", "scripts/evaluate_gaussian_ply.py"}
            <= set(integration_sources) and len(integration_sources) <= 32, "Renderer integration source hashes incomplete")
    for name, digest in integration_sources.items():
        verify(name, digest)
    require(all(_state(p) == state for p, state in snapshots.items()), "Renderer evidence changed during read")
    return {"status": "verified", "backend": backend, "variant": "aa-depth-vjp",
            "integration_proof": {"path": str(integration_path), "sha256": assets[str(integration_path)]},
            "forward_contract": provenance["forward_contract"], "assets": assets,
            "scope": "Pinned native forward/derivative and real model integration evidence; CPU hashes only, no new GPU test or scene quality claim"}


def load_metrics(path, pin=None):
    path = Path(path).resolve()
    report = read_json(path)
    require(report.get("split") == "dev", "Only DEV is allowed; this measurement must not read TEST")
    require(report.get("status") == "evaluated" and report.get("frames"), "Require evaluated native frames")
    report["_renderer_validation"] = validate_renderer_provenance(report, pin)
    require(np.allclose(report.get("world_parser_transform", []), np.eye(4)[:3], atol=1e-7, rtol=0)
            and report.get("world_parser_scale") == 1, "Require identity parser and metre world coordinates")
    keys = [frame_key(row) for row in report["frames"]]
    require(len(keys) == len(set(keys)), "Duplicate group/camera in metrics")
    transforms = Path(report["transforms"])
    require(file_sha256(transforms) == report["transforms_sha256"], "Source transforms changed")
    metadata = read_json(transforms)
    source_rows = {r["file_path"]: r for r in metadata["frames"]}
    require(metadata.get("pixel_center_convention") == "half_integer; principal point +0.5 from OpenCV rectification",
            "Unverified source pixel convention")
    require(report.get("images") == len(report["frames"]), "Metrics image count mismatch")
    for row in report["frames"]:
        check_camera(row["camera"])
        require(type(row["dataset_index"]) is int and row["dataset_index"] >= 0, "Invalid dataset_index")
        require(row["file_path"] in metadata["val_filenames"], "Frame is not in source DEV split")
        require(metadata["val_filenames"][row["dataset_index"]] == row["file_path"], "DEV dataset index drift")
        source = source_rows[row["file_path"]]
        expected = {k: metadata.get(k, source.get(k)) for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
        expected.update(camera_to_worlds_ns=source["transform_matrix"][:3], physical_camera_id=int(source["camera_id"].rsplit("_", 1)[1]))
        require(frame_key(row) == frame_key(source) and same_camera(row["camera"], expected), "Native source frame/camera drift")
        for key, source_key in (("source_image_path", "file_path"), ("source_mask_path", "mask_path"),
                                ("source_lidar_depth_path", "lidar_depth_path"), ("source_lidar_support_path", "lidar_depth_mask_path")):
            require(not source.get(source_key) or Path(row[key]).resolve() == (transforms.parent/source[source_key]).resolve(), "Native source asset path drift: " + key)
    return report


def mask_array(path, shape):
    if path is None:
        return np.ones(shape, dtype=bool)
    with Image.open(path) as image:
        require(image.size == (shape[1], shape[0]), "Mask dimensions differ from fixed camera")
        return np.asarray(image.convert("L")) > 0


def classify_reference(reference, nearest, distances, camera_z, settings):
    """Labels depend only on nearest FIXED reference point/normal and height."""
    xyz = reference["xyz"][nearest]
    normals = reference["normals"][nearest].astype(np.float64)
    length = np.linalg.norm(normals, axis=1)
    trusted = (distances <= settings["reference_match_max_m"]) & reference["normal_valid"][nearest].astype(bool) & (length > 1e-8)
    nz = np.abs(normals[:, 2])/np.maximum(length, 1e-8)
    dz = xyz[:, 2] - camera_z
    labels = np.full(len(nearest), 3, np.uint8)
    labels[trusted] = 2
    floor = trusted & (nz >= settings["floor_abs_normal_z_min"]) & (dz <= -settings["floor_below_camera_min_m"]) & (dz >= -settings["floor_below_camera_max_m"])
    wall = trusted & (nz <= settings["wall_abs_normal_z_max"]) & (np.abs(dz) <= settings["wall_height_radius_m"])
    labels[floor], labels[wall] = 0, 1
    return labels


def count_coverage(labels, reference_depth, alpha, depth, tolerance):
    """All categories share a fixed denominator; no candidate-dependent deletion."""
    if alpha is None or depth is None:
        unknown = np.ones(len(labels), bool)
        missing = invalid = wrong = correct = near = far = np.zeros(len(labels), bool)
    else:
        unknown = ~np.isfinite(alpha) | (alpha < 0) | (alpha > 1)
        missing = ~unknown & (alpha <= .5)
        invalid = ~unknown & ~missing & (~np.isfinite(depth) | (depth <= 0))
        predicted = ~unknown & ~missing & ~invalid
        wrong = predicted & (np.abs(depth-reference_depth) > tolerance)
        near = predicted & (depth-reference_depth < -tolerance)
        far = predicted & (depth-reference_depth > tolerance)
        correct = predicted & ~wrong
    result = {}
    for i, label in enumerate(LABELS):
        use = labels == i
        result[label] = {"denominator": int(use.sum()), **{name: int((use & value).sum()) for name, value in
                         (("correct", correct), ("missing_alpha", missing), ("invalid_depth", invalid),
                          ("wrong_depth", wrong), ("unknown_prediction", unknown), ("too_near", near), ("too_far", far))}}
    return result


def finalize_coverage(counts):
    result = {}
    for label, values in counts.items():
        n = values["denominator"]
        unknown = values["unknown_prediction"]
        result[label] = {**values, "status": "known" if n and not unknown and label != "unknown" else "unknown",
                        "hole_rate": (1-values["correct"]/n) if n else None,
                        "correct_depth_coverage": values["correct"]/n if n else None,
                        "alpha_gt_0_5_coverage": (n-values["missing_alpha"]-unknown)/n if n and not unknown else None}
        for key in COUNTS[2:] + DIRECTION_COUNTS:
            result[label][key+"_rate"] = values[key]/n if n else None
        for key in DIRECTION_COUNTS:
            result[label][key+"_given_wrong_fraction"] = values[key]/values["wrong_depth"] if values["wrong_depth"] else None
    return result


def representation_sources(report, pin):
    """Bind an inference sidecar without loading a GPU model or recasting PLY identity."""
    side = report.get("direction_support")
    if not side:
        return {"kind": "checkpoint" if report.get("checkpoint_sha256") and not report.get("source_ply_sha256") else "ply",
                "representation_id": report.get("source_ply_sha256") or report.get("checkpoint_sha256"), "assets": {}}
    path = Path(side["path"]).resolve()
    pin(path)
    require(file_sha256(path) == side["sha256"], "Direction metadata changed")
    # The sidecar includes TRAIN group attribution for all 28 source blocks;
    # use the same 16 MiB cap as direction_support.load_direction_support.
    with path.open("rb") as stream:
        raw = stream.read(16*MIB+1)
    require(len(raw) <= 16*MIB, "Direction metadata exceeds 16 MiB")
    metadata = json.loads(raw)
    require(side["source_ply_sha256"] == report["source_ply_sha256"] == metadata["ply"]["sha256"], "Direction source PLY mismatch")
    require(side["source_records"] == report["gaussians"] == metadata["ply"]["gaussians"], "Direction source record count mismatch")
    expected = hashlib.sha256((report["source_ply_sha256"]+":"+side["sha256"]).encode()).hexdigest()
    require(report.get("representation_id") == expected, "Direction representation identity mismatch")
    require(metadata["side_flags"]["sha256"] == side["side_flags_sha256"], "Direction flag descriptor mismatch")
    assets = {"metadata": {"path": str(path), "sha256": side["sha256"]}}
    evidence = metadata["training_evidence"]
    require(evidence.get("train_only") is True and evidence.get("heldout_used") is False, "Direction support requires TRAIN-only evidence")
    for name, desc in (("side_flags", metadata["side_flags"]), ("positive_train_groups", evidence["positive_train_groups"]),
                       ("negative_train_groups", evidence["negative_train_groups"])):
        asset = Path(desc["path"])
        if not asset.is_absolute():
            asset = (ROOT/asset).resolve() if (ROOT/asset).is_file() else (path.parent/asset).resolve()
        pin(asset)
        require(file_sha256(asset) == desc["sha256"], "Direction support asset changed: " + name)
        assets[name] = {"path": str(asset), "sha256": desc["sha256"]}
    return {"kind": "ply_plus_train_direction_sidecar", "representation_id": expected,
            "description": report.get("model_representation"), "direction_support": side, "assets": assets}


def validate_checkpoint_proof(proof_path, metrics_path, report, ply, pin):
    """Rehash CPU exact-export evidence; never claim to rerun tensor equality here.

    Schema 2 belongs to verify_training_checkpoint.py. Its source table must
    bind the actual evaluator report, checkpoint, exported PLY, config and
    transforms. Every listed source is rehashed and watched through measurement.
    Native checkpoint identity remains distinct from its opacity-filtered PLY.
    """
    require(ply is not None, "Checkpoint proof requires a measured exported PLY")
    proof_path, metrics_path = Path(proof_path).resolve(), Path(metrics_path).resolve()
    pin(proof_path)
    proof = read_json(proof_path)
    require(proof.get("schema_version") == 2 and proof.get("status") == "checkpoint_verified"
            and proof.get("incomplete_reasons") == [], "Checkpoint proof is incomplete or unsupported")
    require(proof.get("export_verified") is True and proof.get("all_tensors_finite") is True
            and proof.get("verification_scope") == "checkpoint_optimizer_and_export_authenticity_only",
            "Checkpoint proof lacks finite-tensor/exact-export verification")
    for origin in (proof, report):
        require(origin.get("parameter_origin") == "trained_checkpoint"
                and origin.get("checkpoint_is_template") is False,
                "Checkpoint proof requires actual trained_checkpoint parameter origin")
    require(report.get("source_ply") is None and report.get("source_ply_sha256") is None
            and not report.get("direction_support"), "Checkpoint proof cannot bind replaced/sidecar parameters")
    step = proof.get("step")
    require(type(step) is int and step >= 0 and all(type(o.get(k)) is int and o[k] == step
            for o, k in ((proof, "inference_model_step"), (report, "step"), (report, "inference_model_step"))),
            "Checkpoint proof/evaluator step mismatch")
    require(proof.get("completed_iterations") == step+1 == proof.get("total_iterations")
            and proof.get("remaining_iterations") == 0, "Checkpoint proof iteration evidence incomplete")
    count = proof.get("gaussians")
    require(type(count) is int and count > 0 and report.get("gaussians") == count,
            "Checkpoint proof native Gaussian count mismatch")
    exported = proof.get("export")
    require(isinstance(exported, dict) and exported.get("verification") ==
            "all field bytes exact; native opacity cutoff -5.5373; normalized wxyz; actual normals",
            "Checkpoint proof lacks schema-2 exact export field verification")
    require(exported.get("gaussians") == ply["gaussians"] == (report.get("export") or {}).get("gaussians")
            and 0 < ply["gaussians"] <= count, "Checkpoint proof export Gaussian count mismatch")
    def local_path(value):
        require(isinstance(value, str) and value, "Checkpoint proof source path missing")
        return (ROOT/Path(value).expanduser()).resolve()
    require(local_path(exported.get("path")) == Path(ply["path"]).resolve(), "Checkpoint proof export PLY path mismatch")
    for key in ("checkpoint", "config"):
        require(local_path(proof.get(key)) == local_path(report.get(key)), "Checkpoint proof " + key + " path mismatch")
    require(local_path(proof["checkpoint"]).name == f"step-{step:09d}.ckpt", "Checkpoint proof filename/step mismatch")
    require(proof.get("block_id") == local_path(report.get("transforms")).parent.name,
            "Checkpoint proof block differs from source camera dataset")
    source_hashes = proof.get("source_sha256")
    require(isinstance(source_hashes, dict) and 5 <= len(source_hashes) <= 64,
            "Checkpoint proof source SHA table missing or exceeds bound")
    hashes = {}
    for source, digest in source_hashes.items():
        path = local_path(source)
        require(path not in hashes and isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Checkpoint proof invalid/duplicate source SHA")
        pin(path)
        require(file_sha256(path) == digest, "Checkpoint proof source SHA changed: " + str(path))
        hashes[path] = digest
    assets = {"metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)}}
    for key, source, digest in (("checkpoint", report["checkpoint"], report.get("checkpoint_sha256")),
                                ("export", ply["path"], ply["sha256"]),
                                ("config", report["config"], report.get("config_sha256")),
                                ("transforms", report["transforms"], report.get("transforms_sha256"))):
        path = local_path(source)
        require(digest is not None and hashes.get(path) == digest, "Checkpoint proof required source SHA mismatch: " + key)
        assets[key] = {"path": str(path), "sha256": digest}
    require(hashes.get(metrics_path) == assets["metrics"]["sha256"], "Checkpoint proof evaluator report SHA/path mismatch")
    require(proof.get("checkpoint_sha256") == assets["checkpoint"]["sha256"]
            and exported.get("sha256") == ply["sha256"], "Checkpoint proof checkpoint/export SHA mismatch")
    assets["export"].update(gaussians=ply["gaussians"], verification=exported["verification"])
    return {"status": "verified", "path": str(proof_path), "sha256": file_sha256(proof_path),
            "proof_schema_version": 2, "export_verified": True, "all_tensors_finite": True,
            "parameter_origin": "trained_checkpoint", "checkpoint_is_template": False,
            "step": step, "inference_model_step": step, "native_gaussians": count, **assets,
            "source_sha256": {str(p): sha for p, sha in hashes.items()},
            "tensor_equality_verified_by": "scripts/verify_training_checkpoint.py",
            "verification_scope": "Rehashed complete CPU exact-export proof and source assets; tensor equality was verified by the separate CPU verifier, not recomputed here; no quality approval"}


def read_render(metrics_path, row, max_pixels):
    path = Path(metrics_path).parent / ("%03d_float.npz" % row["dataset_index"])
    if not path.is_file():
        return None, {"path": str(path), "status": "unknown", "reason": "missing native float archive"}
    arrays = bounded_npz(path, max_pixels*24 + 65536)
    if not all(key in arrays for key in ("rgb", "alpha", "depth")):
        return None, {"path": str(path), "status": "unknown", "reason": "missing rgb/alpha/depth array"}
    h, w = row["camera"]["h"], row["camera"]["w"]
    require(arrays["rgb"].shape == (h, w, 3), "RGB shape differs from camera")
    for key in ("rgb", "alpha", "depth"):
        require(arrays[key].dtype == np.float32, "Native arrays must be float32, never PNG-derived")
    for key in ("alpha", "depth"):
        require(arrays[key].shape in ((h, w), (h, w, 1)), key+" shape differs from camera")
        arrays[key] = arrays[key].reshape(-1)
    return arrays, {"path": str(path), "sha256": file_sha256(path), "status": "known"}


def measure_renders(metrics_path, reference_path, settings, *, baseline_metrics=None,
                    acceptance_plan=None, max_frames=0, memory_mib=1024,
                    max_reference_points=2_000_000, max_total_pixels=16_000_000,
                    checkpoint_proof=None, candidate_ply=None):
    snapshots = {}
    def pin(path):
        if path is not None:
            path = Path(path).resolve()
            snapshots.setdefault(path, _state(path))
    for path in (metrics_path, reference_path, baseline_metrics, acceptance_plan):
        pin(path)
    candidate = load_metrics(metrics_path, pin)
    checkpoint_evidence = validate_checkpoint_proof(checkpoint_proof, metrics_path, candidate, candidate_ply, pin) if checkpoint_proof else None
    baseline = load_metrics(baseline_metrics, pin) if baseline_metrics else None
    fixed = baseline or candidate
    selected = fixed["frames"][:max_frames] if max_frames else fixed["frames"]
    maps = {"candidate": {frame_key(row): row for row in candidate["frames"]}}
    paths = {"candidate": Path(metrics_path).resolve()}
    if baseline:
        maps["baseline"] = {frame_key(row): row for row in baseline["frames"]}
        paths["baseline"] = Path(baseline_metrics).resolve()
        require(set(maps["candidate"]) == set(maps["baseline"]), "Paired metrics require exactly the same DEV frame keys")
    representations = {name: representation_sources(report, pin) for name, report in
                       (("candidate", candidate), ("baseline", baseline)) if report is not None}
    all_pixels = sum(r["camera"]["w"]*r["camera"]["h"] for r in selected)
    largest = max(r["camera"]["w"]*r["camera"]["h"] for r in selected)
    require(largest <= 4_000_000 and all_pixels <= max_total_pixels, "Render pixel budget exceeded; use explicit --max-frames")
    reference_path = Path(reference_path).resolve()
    reference = bounded_npz(reference_path, max_reference_points*32+65536)
    require(all(k in reference for k in ("xyz", "normals", "normal_valid")), "Require LiDAR xyz/normals/normal_valid")
    nref = len(reference["xyz"])
    require(0 < nref <= max_reference_points and reference["xyz"].shape == (nref, 3)
            and reference["normals"].shape == (nref, 3) and reference["normal_valid"].shape == (nref,), "Invalid/budget-exceeding LiDAR shapes")
    require(np.isfinite(reference["xyz"]).all() and np.isfinite(reference["normals"]).all(), "Nonfinite reference geometry")
    estimate = nref*160 + largest*200 + all_pixels*24*len(paths) + 64*MIB
    memory_check(estimate, memory_mib, "Render/reference")
    reference_sha = file_sha256(reference_path)
    for report in (candidate, baseline):
        if report is not None:
            original_reference = Path(report["transforms"]).parent/"lidar_reference.npz"
            require(original_reference.is_file() and file_sha256(original_reference) == reference_sha,
                    "Reference differs from native dataset LiDAR; paired measurements require one shared reference")
    tree = cKDTree(reference["xyz"])
    plan = None
    plan_rows = {}
    if acceptance_plan:
        from build_acceptance_plan import load_plan
        plan = load_plan(Path(acceptance_plan), verify_assets=True)
        require(plan.get("split") == "dev", "Require DEV acceptance plan")
        plan_rows = {frame_key(r): r for r in plan["frames"]}
    totals = {name: {label: dict.fromkeys(COUNTS + DIRECTION_COUNTS, 0) for label in LABELS} for name in paths}
    distances = {name: {key: [] for key in ("render_to_reference", "reference_to_render", "floor_height_abs_residual")} for name in paths}
    signed_errors = {name: {label: [] for label in ("floor", "wall")} for name in paths}
    frame_reports, source_fingerprints = [], []
    unsupported_pixels, masked_pixels, support_pixels, unmatched_reference_pixels = 0, 0, 0, 0
    for fixed_row in selected:
        key = frame_key(fixed_row)
        camera = fixed_row["camera"]
        h, w = camera["h"], camera["w"]
        frozen = plan_rows.get(key)
        if plan is not None:
            require(frozen is not None and same_camera(camera, frozen["camera"]), "Frame/camera differs from frozen acceptance plan")
        for name, mapping in maps.items():
            require(key in mapping and same_camera(camera, mapping[key]["camera"]), "Paired source cameras differ")
            require(file_sha256(mapping[key]["source_image_path"]) == file_sha256(fixed_row["source_image_path"]), "Paired source images differ")
        source = frozen or fixed_row
        depth_path, support_path = source.get("source_lidar_depth_path"), source.get("source_lidar_support_path")
        require(depth_path and support_path, "Fixed LiDAR depth/support unavailable; cannot define denominator")
        mask_path = source.get("mask_path") if frozen else source.get("acceptance_mask_path", source.get("source_mask_path"))
        for path in (depth_path, support_path, mask_path, source.get("source_mask_path")):
            pin(path)
        depth_ref = np.load(depth_path, allow_pickle=False, mmap_mode="r")
        require(depth_ref.shape == (h, w) and depth_ref.dtype.kind == "f", "Invalid fixed camera-z depth")
        valid = mask_array(mask_path, (h, w))
        valid &= mask_array(source.get("source_mask_path"), (h, w))
        support = mask_array(support_path, (h, w)) & np.isfinite(depth_ref) & (depth_ref > 0)
        sampled = np.zeros((h, w), bool)
        sampled[::settings["pixel_stride"], ::settings["pixel_stride"]] = True
        eligible = support & valid & sampled
        indices = np.flatnonzero(eligible)
        ref_depth = depth_ref.reshape(-1)[indices]
        ref_world = unproject(camera, indices, ref_depth, settings["reference_pixel_center"])
        nearest_dist, nearest = tree.query(ref_world, workers=4)
        labels = classify_reference(reference, nearest, nearest_dist, camera["camera_to_worlds_ns"][2][3], settings)
        matched = nearest_dist <= settings["reference_match_max_m"]
        # Reverse distance covers only reference points witnessed by these fixed rays.
        target_ids = np.unique(nearest[matched])
        observed_reference = reference["xyz"][target_ids]
        source_pins = {k: file_sha256(v) for k, v in (("reference_depth", depth_path), ("reference_support", support_path),
                      ("mask", mask_path), ("legacy_mask", source.get("source_mask_path"))) if v}
        source_fingerprints.append({"key": list(key), "camera": camera, "assets": source_pins,
                                    "eligible_pixels_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                                    "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()})
        counts = {"sampled_unmasked_pixels": int((valid & sampled).sum()), "fixed_support_pixels": len(indices),
                  "unsupported_unknown_pixels": int((valid & sampled & ~support).sum()),
                  "unmatched_reference_pixels": int((~matched).sum()), "visible_reference_points": len(target_ids)}
        support_pixels += len(indices)
        unsupported_pixels += counts["unsupported_unknown_pixels"]
        unmatched_reference_pixels += counts["unmatched_reference_pixels"]
        masked_pixels += counts["sampled_unmasked_pixels"]
        result = {"frame_key": list(key), "dataset_index": fixed_row["dataset_index"], "camera": camera,
                  "fixed_support": counts, "source_sha256": source_pins, "results": {}}
        for name, mapping in maps.items():
            render_path = paths[name].parent/("%03d_float.npz" % mapping[key]["dataset_index"])
            if render_path.is_file():
                pin(render_path)
            arrays, artifact = read_render(paths[name], mapping[key], largest)
            alpha, depth = (arrays["alpha"][indices], arrays["depth"][indices]) if arrays is not None else (None, None)
            coverage = count_coverage(labels, ref_depth, alpha, depth, settings["depth_tolerance_m"])
            for label in LABELS:
                for count in COUNTS + DIRECTION_COUNTS:
                    totals[name][label][count] += coverage[label][count]
            direction = {"render_to_reference": np.empty(0), "reference_to_render": np.empty(0), "floor_height_abs_residual": np.empty(0)}
            errors = {label: np.empty(0, np.float32) for label in ("floor", "wall")}
            if arrays is not None:
                covered = np.isfinite(alpha) & (alpha > .5) & (alpha <= 1) & np.isfinite(depth) & (depth > 0) & matched
                world = unproject(camera, indices[covered], depth[covered], settings["render_pixel_center"])
                if len(world):
                    direction["render_to_reference"] = tree.query(world, workers=4)[0]
                    if len(observed_reference):
                        direction["reference_to_render"] = cKDTree(world).query(observed_reference, workers=4)[0]
                    floor = labels[covered] == 0
                    direction["floor_height_abs_residual"] = np.abs(world[floor, 2]-reference["xyz"][nearest[covered][floor], 2])
                for i, label in enumerate(("floor", "wall")):
                    mask = covered & (labels == i)
                    errors[label] = depth[mask]-ref_depth[mask]
            for label, values in errors.items():
                signed_errors[name][label].append(np.asarray(values, np.float32))
            for k, values in direction.items():
                distances[name][k].append(np.asarray(values, dtype=np.float32))
            result["results"][name] = {"artifact": artifact, "coverage": finalize_coverage(coverage),
                                      "signed_camera_z_error": {k: signed_stats(v) for k, v in errors.items()},
                                      "conditioned_geometry": {k: stats(v) for k, v in direction.items()}}
        frame_reports.append(result)
        print(f"Measured DEV {key[0]}/{key[1]}: {len(indices)} fixed supported rays", file=sys.stderr, flush=True)
    summaries = {}
    for name in paths:
        measured = {k: stats(np.concatenate(v)) for k, v in distances[name].items()}
        a, b = measured["render_to_reference"]["p50_m"], measured["reference_to_render"]["p50_m"]
        unknown = any(f["results"][name]["artifact"]["status"] != "known" or
                      sum(v["unknown_prediction"] for v in f["results"][name]["coverage"].values()) > 0
                      or (f["fixed_support"]["visible_reference_points"] > 0 and
                          f["results"][name]["conditioned_geometry"]["render_to_reference"]["count"] == 0)
                      for f in frame_reports)
        summaries[name] = {"coverage": finalize_coverage(totals[name]),
            "signed_camera_z_error": {k: signed_stats(np.concatenate(v)) for k, v in signed_errors[name].items()}, "conditioned_geometry": {
            **measured, "status": "known" if a is not None and b is not None and not unknown else "unknown",
            "bidirectional_p50_m": (a+b)/2 if a is not None and b is not None and not unknown else None,
            "independent_ground_truth": False,
            "floor_front_surface_z_span_p95_m": None,
            "floor_front_surface_z_span_status": "unknown: one expected-depth value cannot recover multilayer/front-surface thickness"}}
    comparison = {"status": "unknown", "reason": "No paired baseline renders supplied"}
    if baseline:
        comparison = {"status": "known", "fixed_denominator": True, "coverage": {}}
        for label in ("floor", "wall"):
            b, c = (summaries[k]["coverage"][label] for k in ("baseline", "candidate"))
            reduction = 1-c["hole_rate"]/b["hole_rate"] if b["status"] == c["status"] == "known" and b["hole_rate"] > 0 else None
            comparison["coverage"][label] = {"baseline_hole_rate": b["hole_rate"], "candidate_hole_rate": c["hole_rate"],
                                             "relative_reduction": reduction, "status": "known" if reduction is not None else "unknown",
                                             "eligible_pixels": b["denominator"], "candidate_hole_pixels": c["denominator"]-c["correct"],
                                             "baseline_hole_pixels": b["denominator"]-b["correct"]}
        b, c = (summaries[k]["conditioned_geometry"]["bidirectional_p50_m"] for k in ("baseline", "candidate"))
        comparison["conditioned_geometry_ratio"] = c/b if b is not None and b > 0 and c is not None else None
    protocol = {**settings, "reference_sha256": reference_sha,
                "mask_protocol": "ego_v2" if plan else ("reported_acceptance_mask" if any(r.get("acceptance_mask_path") for r in selected) else "legacy"),
                "denominator_policy": "baseline frozen camera-z support and mask, or candidate support for unpaired diagnostic; never candidate alpha/depth",
                "floor_definition": "nearest fixed reference has valid normal, abs(nz)>=threshold and lies below camera within frozen height band; geometric proxy includes ramps, cannot certify floor semantics/level identity",
                "wall_definition": "nearest fixed reference has valid normal, abs(nz)<=threshold and lies within fixed camera-height radius; geometric proxy may include columns/vehicles",
                "up_axis": "camera_init +Z assumed vertical; not independently surveyed",
                "hole_rate": "(missing_alpha + invalid_depth + wrong_depth + unknown_prediction)/fixed class support; not merely 1-alpha coverage",
                "wrong_depth": "alpha>0.5, finite positive depth, abs(predicted camera-z - fixed reference camera-z)>depth_tolerance_m; catches wrong layers even if opaque",
                "unknown": "unsupported pixels and unmatched/invalid-normal class labels are reported, never imputed as complete; unknown predictions remain in denominator and prohibit known/pass status",
                "bidirectional_p50": "0.5*(median(render->full LiDAR nearest distance)+median(visible fixed LiDAR->render nearest distance)); NOT median of pooled directions, NOT global building Chamfer",
                "frame_aggregation": "Pixel-weighted render distances pooled across frames; reverse targets deduplicated within each frame then pooled across frames (the same reference point can be observed in several frames). No cross-frame render merging or distance sampling.",
                "geometry_scope": "covered fixed supported rays whose reference backprojection matches LiDAR within reference_match_max_m; reverse targets are unique nearest LiDAR records of all such fixed rays, including missing predictions",
                "pixel_centres": "NS/OpenGL pose and supplied NS intrinsics; both reference and render use x+.5,y+.5. This equals teacher CV integer pixels with cx_cv=cx_ns-.5,cy_cv=cy_ns-.5. Nondefault overrides are explicit diagnostics."}
    denominator_id = digest_json({"protocol": protocol, "frames": source_fingerprints})
    provenance = {name: {"metric_domain": report.get("metric_domain"),
                         "backend": report.get("backend"),
                         "renderer_provenance": report.get("renderer_provenance"),
                         "renderer_validation": report["_renderer_validation"],
                         "acceptance_plan_id": report.get("acceptance_plan_id"),
                         "source_ply_sha256": report.get("source_ply_sha256"),
                         "checkpoint_sha256": report.get("checkpoint_sha256"),
                         "parameter_origin": report.get("parameter_origin"),
                         "inference_model_step": report.get("inference_model_step")}
                  for name, report in (("candidate", candidate), ("baseline", baseline)) if report is not None}
    if checkpoint_evidence:
        provenance["candidate"]["checkpoint_export_proof"] = checkpoint_evidence
    provenance_reasons = []
    if not plan:
        provenance_reasons.append("missing frozen acceptance plan")
    for role, origin in provenance.items():
        if origin["metric_domain"] != "float_renderer":
            provenance_reasons.append(role + ": missing float_renderer domain")
        if plan and origin["acceptance_plan_id"] != plan["plan_id"]:
            provenance_reasons.append(role + ": evaluator plan differs")
        if not origin["source_ply_sha256"] and not origin.get("checkpoint_export_proof"):
            provenance_reasons.append(role + ": no source PLY SHA or validated exact checkpoint/export proof")
    gate_provenance = not provenance_reasons
    for path, state in snapshots.items():
        require(_state(path) == state, "Measurement input changed during read: " + str(path))
    return {"status": "measured", "acceptance_plan_id": plan.get("plan_id") if plan else None,
            "fixed_denominator_id": denominator_id, "protocol_id": digest_json(protocol), "protocol": protocol,
            "gate_compatible_settings": settings == DEFAULT_SETTINGS,
            "gate_compatible_render_provenance": gate_provenance,
            "render_provenance_unknown_reasons": provenance_reasons,
            "gate_compatible_representation": all(r["kind"] != "ply_plus_train_direction_sidecar" for r in representations.values()),
            "representations": representations,
            "signed_error_definition": "Predicted camera-z minus fixed reference camera-z, metres: negative nearer camera, positive farther. Conditional quantiles on finite positive depth with alpha>0.5 in fixed floor/wall classes. too_near/too_far use configured depth_tolerance_m (default +/-0.15m); rates use fixed class denominator; given_wrong fractions use wrong_depth count. Not world-height residual.",
            "source_evaluator_provenance": provenance,
            "scope": {"frames_measured": len(selected), "frames_in_metrics": len(fixed["frames"]),
                      "canonical_plan_frames": len(plan["frames"]) if plan else None,
                      "complete_acceptance_plan": bool(plan and {frame_key(r) for r in selected} == set(plan_rows)),
                      "frame_keys": [{"group_id": frame_key(r)[0], "camera_id": frame_key(r)[1]} for r in selected], "test_read": False,
                      "block_ids": sorted({plan_rows[frame_key(r)]["block_id"] for r in selected}) if plan else [Path(fixed["transforms"]).parent.name],
                      "pixel_stride": settings["pixel_stride"], "fixed_support_pixels": support_pixels,
                      "unmasked_sampled_pixels": masked_pixels, "unsupported_unknown_pixels": unsupported_pixels,
                      "unmatched_reference_pixels": unmatched_reference_pixels,
                      "support_fraction": support_pixels/masked_pixels if masked_pixels else None,
                      "world_frame": "camera_init", "units": "m", "independent_ground_truth": False},
            "sources": {"reference": {"path": str(reference_path), "sha256": reference_sha, "points": nref},
                        **({"candidate_checkpoint_proof": {"path": checkpoint_evidence["path"], "sha256": checkpoint_evidence["sha256"]}} if checkpoint_evidence else {}),
                        **{role+"_direction_"+key: value for role, rep in representations.items() for key, value in rep["assets"].items()},
                        **{k: {"path": str(v), "sha256": file_sha256(v)} for k, v in paths.items()}},
            "working_memory_estimate_bytes": estimate, "summaries": summaries, "comparison": comparison, "frames": frame_reports}


def gate_evidence(ply, render):
    """Measurement handoff only: acceptance_gate owns thresholds and scope checks."""
    evidence = {"G3": {"status": "unknown"}, "G4": {"status": "unknown"},
                "G5": {"status": "unknown"}, "G7": {"status": "unknown"}}
    if ply:
        evidence["G3"] = {"status": "known", **{k: ply[k] for k in
            ("opacity_median", "opacity_frac_lt_0_5", "voxel_2cm_multiplicity_median", "voxel_size_m")}}
        evidence["G7"] = {"status": "known", **{k: ply[k] for k in
            ("tangential_radius_p95_m", "tangential_radius_max_m")}, "seam_gap_db": None,
            "seam_status": "unknown: requires separate fixed boundary photometric evaluation"}
    if render:
        summaries, comparison = render["summaries"], render["comparison"]
        ratio = comparison.get("conditioned_geometry_ratio")
        evidence["G4"] = {"status": "known" if ratio is not None else "unknown",
            "candidate_p50_m": summaries["candidate"]["conditioned_geometry"]["bidirectional_p50_m"],
            "baseline_p50_m": summaries.get("baseline", {}).get("conditioned_geometry", {}).get("bidirectional_p50_m"),
            "ratio": ratio, "independent_ground_truth": False,
            "reference_sha256": render["sources"]["reference"]["sha256"], "shared_reference": True,
            "candidate_points": summaries["candidate"]["conditioned_geometry"]["render_to_reference"]["count"],
            "baseline_points": summaries.get("baseline", {}).get("conditioned_geometry", {}).get("render_to_reference", {}).get("count", 0)}
        coverage = comparison.get("coverage", {})
        reductions = {label+"_relative_reduction": coverage.get(label, {}).get("relative_reduction") for label in ("floor", "wall")}
        evidence["G5"] = {"status": "known" if all(v is not None for v in reductions.values()) else "unknown", **reductions,
                          "reference_sha256": render["sources"]["reference"]["sha256"], "shared_reference": True,
                          "fixed_denominator": True,
                          "fixed_denominator_id": render["fixed_denominator_id"], "floor": coverage.get("floor"), "wall": coverage.get("wall")}
        if not render["gate_compatible_settings"] or not render["gate_compatible_render_provenance"] or not render.get("gate_compatible_representation", True):
            reasons = ([] if render["gate_compatible_settings"] else ["nondefault geometry settings"])
            if not render["gate_compatible_render_provenance"]:
                reasons.extend(render.get("render_provenance_unknown_reasons") or ["missing plan/float/source provenance"])
            if not render.get("gate_compatible_representation", True):
                reasons.append("directional-sidecar acceptance not integrated")
            for key in ("G4", "G5"):
                evidence[key].update(status="unknown", reason="; ".join(reasons) + "; measured values retained")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=Path, help="Defaults to metrics.source_ply.path")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--reference", type=Path, help="Same-source lidar_reference.npz")
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--acceptance-plan", type=Path)
    parser.add_argument("--checkpoint-proof", type=Path, help="Optional candidate schema-2 verify_training_checkpoint.py exact-export JSON; all bound files are rehashed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-mib", type=int, default=1024)
    parser.add_argument("--max-gaussians", type=int, default=12_000_000)
    parser.add_argument("--max-ply-bytes", type=int, default=4*1024**3)
    parser.add_argument("--max-reference-points", type=int, default=2_000_000)
    parser.add_argument("--max-total-pixels", type=int, default=16_000_000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--depth-tolerance-m", type=float, default=.15)
    parser.add_argument("--reference-match-max-m", type=float, default=.20)
    parser.add_argument("--render-pixel-center", type=float, choices=(0., .5), default=.5)
    parser.add_argument("--reference-pixel-center", type=float, choices=(0., .5), default=.5)
    parser.add_argument("--floor-abs-normal-z-min", type=float, default=float(np.cos(np.deg2rad(30))))
    parser.add_argument("--wall-abs-normal-z-max", type=float, default=.30)
    parser.add_argument("--floor-below-camera-min-m", type=float, default=.30)
    parser.add_argument("--floor-below-camera-max-m", type=float, default=3.)
    parser.add_argument("--wall-height-radius-m", type=float, default=3.)
    args = parser.parse_args(argv)
    require(args.ply or args.metrics, "Supply --ply and/or --metrics")
    require(not args.output.exists(), "Output exists; preserve original and choose a new report path")
    require(64 <= args.memory_mib <= 8192 and args.max_frames >= 0 and args.pixel_stride > 0, "Invalid budgets/stride")
    require(all(getattr(args, k) > 0 for k in ("max_gaussians", "max_ply_bytes", "max_reference_points", "max_total_pixels")), "Budgets must be positive")
    require(all(np.isfinite(getattr(args, k)) and getattr(args, k) > 0 for k in
                ("depth_tolerance_m", "reference_match_max_m", "floor_below_camera_min_m", "floor_below_camera_max_m", "wall_height_radius_m")), "Thresholds must be finite positive")
    require(0 <= args.wall_abs_normal_z_max < args.floor_abs_normal_z_min <= 1
            and args.floor_below_camera_min_m < args.floor_below_camera_max_m, "Invalid frozen classification thresholds")
    require(not (args.baseline_metrics or args.reference or args.acceptance_plan or args.checkpoint_proof) or args.metrics, "Render options require --metrics")
    if args.metrics:
        meta = load_metrics(args.metrics)
        if args.ply is None:
            source = meta.get("source_ply") or {}
            if source.get("path"):
                args.ply = Path(source["path"])
            elif meta.get("export"):
                args.ply = args.metrics.parent/"gaussians.ply"
        require(args.reference, "--metrics requires explicit --reference")
    started = time.monotonic()
    script_sha = file_sha256(__file__)
    ply = measure_ply(args.ply, memory_mib=args.memory_mib, max_gaussians=args.max_gaussians,
                      max_ply_bytes=args.max_ply_bytes) if args.ply else None
    if args.metrics and ply and meta.get("source_ply_sha256"):
        require(ply["sha256"] == meta["source_ply_sha256"], "Measured PLY differs from native render source")
    binding = {"status": "ply_only"}
    if args.metrics and ply:
        binding = {"status": "source_ply_sha_matched" if meta.get("source_ply_sha256") else "checkpoint_export_count_only",
                   "parameter_origin": meta.get("parameter_origin"), "inference_model_step": meta.get("inference_model_step"),
                   "checkpoint_sha256": meta.get("checkpoint_sha256"), "candidate_ply_sha256": ply["sha256"],
                   "checkpoint_export_tensor_equality_verified_here": False}
        if meta.get("export") and not meta.get("source_ply_sha256"):
            require(meta["export"]["gaussians"] == ply["gaussians"], "Checkpoint export count mismatch")
            expected_sha = meta["export"].get("sha256")
            if expected_sha:
                require(expected_sha == ply["sha256"], "Checkpoint export SHA mismatch")
                binding["status"] = "checkpoint_export_sha_matched"
    settings = {k: getattr(args, k) for k in ("pixel_stride", "depth_tolerance_m", "reference_match_max_m",
                "render_pixel_center", "reference_pixel_center", "floor_abs_normal_z_min", "wall_abs_normal_z_max",
                "floor_below_camera_min_m", "floor_below_camera_max_m", "wall_height_radius_m")}
    render = measure_renders(args.metrics, args.reference, settings, baseline_metrics=args.baseline_metrics,
                  acceptance_plan=args.acceptance_plan, max_frames=args.max_frames, memory_mib=args.memory_mib,
                  max_reference_points=args.max_reference_points, max_total_pixels=args.max_total_pixels,
                  checkpoint_proof=args.checkpoint_proof, candidate_ply=ply) if args.metrics else None
    if render and args.checkpoint_proof:
        proof = render["source_evaluator_provenance"]["candidate"]["checkpoint_export_proof"]
        binding.update(status="checkpoint_export_exact_proof_verified", checkpoint_export_tensor_equality_verified_by_cpu_proof=True,
                       checkpoint_proof={"path": proof["path"], "sha256": proof["sha256"]})
    result = {"schema_version": 2, "report_type": "surface_quality_v2", "status": "measured",
              "metric_domain": "conditioned_geometry",
              "candidate_model_sha256": ply["sha256"] if ply else (meta.get("source_ply_sha256") if args.metrics else None),
              "candidate_representation_id": render["representations"]["candidate"]["representation_id"] if render else None,
              "baseline_representation_id": render["representations"].get("baseline", {}).get("representation_id") if render else None,
              "candidate_ply_render_binding": binding,
              "baseline_model_sha256": load_metrics(args.baseline_metrics).get("source_ply_sha256") if args.baseline_metrics else None,
              "block_id": render["scope"]["block_ids"][0] if render and len(render["scope"]["block_ids"]) == 1 else None,
              "high_fidelity_pass": False, "independent_ground_truth": False,
              "acceptance_plan_id": render["acceptance_plan_id"] if render else None,
              "scope": render["scope"] if render else {"ply_only": True, "complete_acceptance_plan": False},
              "ply": ply, "render_measurement": render, "gate_evidence": gate_evidence(ply, render),
              "script_sha256": script_sha, "elapsed_seconds": time.monotonic()-started,
              "resource_usage": {"process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                                 "rss_platform": "Linux ru_maxrss KiB converted to bytes", "query_threads": 4,
                                 "working_memory_budget_bytes": args.memory_mib*MIB},
              "resource_policy": "CPU only, <=4 query/BLAS threads; exact PLY counts; explicitly bounded allocations; no reference/model sampling. Sequential one-frame loading; only distance vectors retained.",
              "limitations": ["Same initialization LiDAR and registration; never independent accuracy certification",
                              "Geometric floor/wall proxies, unverified floor labels and vertical axis",
                              "Unsupported and uncertain regions remain unknown; coverage is not whole-building completeness",
                              "Expected depth cannot establish front-layer thickness or absence of multiple layers",
                              "No gate decisions are made here; gate owner must verify frame/plan/protocol scope"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(file_sha256(__file__) == script_sha, "Measurement script changed during execution")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output.resolve()), "gate_evidence": result["gate_evidence"], "scope": result["scope"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)

#!/usr/bin/env python3
"""Read-only CPU audit of the 28-block first-delivery Gaussian map.

No renderer, trainer or Nerfstudio imports. NumPy and PyYAML are required;
torch is imported only for --verify-checkpoints, one checkpoint at a time.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import codecs
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = tuple(f"block_{i:03d}" for i in range(28))
FLOOR_SCHEDULE = Path("/home/chaoszzss/orbvi_proj_ws/src/"
                      "orbvi_omni_livo2_lioslam_fusion_CZY_v6/config/czy_all_floor_schedule.json")
SPLIT_KEYS = {"train": "train_filenames", "dev": "val_filenames", "test": "test_filenames"}
CONDITIONED = "same-source conditioned; not independent ground truth"
LIMITATIONS = [
    "Calibration is not independently verified by this audit; false/unknown flags are preserved.",
    "Six floor/ramp assignments are estimated temporal labels, not independently surveyed floors.",
    "RGB pixel, projected depth and map geometry metrics are same-source conditioned, not independent GT.",
    "Completeness covers captured LIO trajectory and observed/visible regions only; it does not prove coverage of every unobserved building corner.",
    "This audit does not grant high fidelity or absolute geometric accuracy acceptance.",
    "Evaluation metrics are read and cross-checked against per-frame reports, not re-rendered on CPU.",
]


class AuditError(ValueError):
    pass


class IncompleteError(AuditError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def project_path(value, root=ROOT, base=None):
    """Resolve symlinks before accepting a project input/output path."""
    require(isinstance(value, (str, Path)) and str(value), "missing path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or root) / path
    path = path.resolve()
    require(path.is_relative_to(root.resolve()), f"path outside project: {path}")
    return path


def output_path(value, root=ROOT):
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        requested = root / requested
    require(not requested.is_symlink(), f"refusing existing output symlink: {requested}")
    path = project_path(value, root)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    require(path.parent.is_dir(), f"output parent must already exist: {path.parent}")
    return path


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=lambda s: (_ for _ in ()).throw(AuditError(f"nonfinite JSON: {s}")))
    require(isinstance(value, dict), "JSON report must be an object")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    # BaseLoader parses Python-tagged YAML as inert strings/lists/maps. It never
    # constructs Python objects or imports the CUDA training configuration.
    import yaml
    class InertLoader(yaml.BaseLoader):
        pass
    InertLoader.add_multi_constructor("tag:yaml.org,2002:python/name:",
                                     lambda loader, suffix, node: suffix)
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=InertLoader)
    require(isinstance(value, dict), "config must be a mapping")
    return value


def config_path(value):
    if isinstance(value, list):
        return str(Path(*value))
    require(isinstance(value, str) and value not in ("", "null"), "missing config path")
    return value


def flag(value, expected):
    return str(value).lower() == str(expected).lower()


def identity_transform(value, scale):
    matrix = np.asarray(value, dtype=float)
    require(matrix.shape in ((3, 4), (4, 4)), "invalid world transform shape")
    require(np.isfinite(matrix).all() and np.allclose(matrix, np.eye(4)[:len(matrix)], atol=1e-7, rtol=0),
            "world transform changes global coordinates (centering/orientation)")
    require(math.isfinite(float(scale)) and abs(float(scale) - 1) < 1e-7,
            "world scale changes metres (autoscale)")


def validate_config(config, dataset, root, target):
    require(int(config["max_num_iterations"]) == target, "config target does not match requested delivery target")
    manager = config["pipeline"]["datamanager"]
    parser = manager["dataparser"]
    model = config["pipeline"]["model"]
    for section in (manager, parser):
        require(project_path(config_path(section["data"]), root) == dataset,
                "config consumes a different prepared block")
    require(manager.get("_target") == "orbvi_gaussian.lidar_splatfacto.LidarFullImageDatamanager", "stock datamanager does not consume LiDAR targets")
    require(model.get("_target") == "orbvi_gaussian.lidar_splatfacto.LidarSplatfactoModel", "model is not the LiDAR Gaussian model")
    require(parser.get("center_method") == "none" and parser.get("orientation_method") == "none",
            "per-block centering/orientation enabled")
    require(flag(parser.get("auto_scale_poses"), False) and float(parser["scale_factor"]) == 1,
            "parser autoscale enabled or metric scale changed")
    require(flag(parser.get("load_3D_points"), True) and flag(model.get("random_init"), False),
            "model lacks actual prepared LiDAR initialization")
    require(flag(model.get("surface_splats"), True) and flag(model.get("use_inria_surfel_renderer"), True),
            "actual training config does not use LiDAR surfels and Inria surfel renderer")
    require(not flag(model.get("use_2dgs_renderer", False), True), "ambiguous competing surfel renderer enabled")
    require(flag(model.get("output_depth_during_training"), True) and float(model["lidar_depth_loss_weight"]) > 0,
            "LiDAR depth supervision disabled")
    require(flag(manager.get("allow_depth_missing"), False) and flag(model.get("allow_depth_missing"), False),
            "missing depth silently permitted")
    require(flag(model["camera_optimizer"].get("mode"), "off"), "per-image camera optimizer changes prepared poses")
    require(not flag(model.get("optimize_shared_rig", False), True),
            "shared rig corrections need an explicit additional pose/provenance audit")
    require(project_path(config_path(model["lidar_reference_path"]), root) == dataset / "lidar_reference.npz",
            "renderer surface reference comes from another dataset")


PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def ply_header(stream):
    raw = bytearray()
    elements, properties, fmt = [], [], None
    for _ in range(2048):
        line = stream.readline(4097)
        require(line and len(line) <= 4096, "truncated/oversized PLY header")
        raw.extend(line)
        parts = line.decode("ascii").strip().split()
        if len(raw) == len(line):
            require(parts == ["ply"], "not a PLY file")
        elif parts and parts[0] == "format":
            require(len(parts) == 3 and parts[2] == "1.0" and fmt is None, "invalid PLY format")
            fmt = parts[1]
        elif parts and parts[0] == "element":
            require(len(parts) == 3, "invalid PLY element")
            elements.append((parts[1], int(parts[2])))
        elif parts and parts[0] == "property":
            require(elements and elements[-1][0] == "vertex", "unsupported non-vertex PLY property")
            require(len(parts) == 3 and parts[1] in PLY_TYPES, "unsupported PLY list/type")
            properties.append((parts[2], PLY_TYPES[parts[1]]))
        elif parts == ["end_header"]:
            break
        elif parts and parts[0] not in ("comment", "obj_info"):
            raise AuditError("unsupported PLY header entry")
    else:
        raise AuditError("PLY header exceeds bound")
    require(elements and elements[0][0] == "vertex" and elements[0][1] > 0,
            "PLY has no Gaussian vertices")
    require(all(n == 0 for _, n in elements[1:]), "unsupported extra PLY payload elements")
    require(fmt in ("binary_little_endian", "binary_big_endian", "ascii"), "unsupported PLY encoding")
    names = [n for n, _ in properties]
    require(len(set(names)) == len(names), "duplicate PLY properties")
    return bytes(raw), fmt, elements[0][1], properties


def scan_ply(stream, chunk_bytes=8 * 1024 * 1024, gaussian=True, compare=None):
    """Check EVERY scalar and quaternion, in bounded chunks, including the tail."""
    header, fmt, count, properties = ply_header(stream)
    names = [n for n, _ in properties]
    required = ["x", "y", "z"]
    degree = None
    if gaussian:
        required += ["nx", "ny", "nz", "opacity"] + [f"f_dc_{i}" for i in range(3)]
        required += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        rest = [n for n in names if n.startswith("f_rest_")]
        require(set(rest) == {f"f_rest_{i}" for i in range(len(rest))}, "invalid SH channel ordering")
        degree = math.isqrt(len(rest) // 3 + 1) - 1
        require(len(rest) == 3 * ((degree + 1) ** 2 - 1) and 0 <= degree <= 4, "invalid SH coefficient count")
    require(set(required).issubset(names), "PLY lacks real Gaussian attributes" if gaussian else "PLY lacks xyz")
    if gaussian:
        require(all(t.startswith("f") for n, t in properties if n in required or n.startswith("f_rest_")),
                "Gaussian attributes must be floating point")
    endian = ">" if fmt == "binary_big_endian" else "<"
    dtype = np.dtype([(n, endian + t) for n, t in properties])
    require(dtype.itemsize > 0 and dtype.itemsize <= chunk_bytes, "PLY row exceeds memory bound")
    rows_per_chunk = max(1, chunk_bytes // dtype.itemsize)
    digest = hashlib.sha256(header)
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    qmin, qmax, seen = math.inf, 0., 0
    while seen < count:
        nrows = min(rows_per_chunk, count - seen)
        if fmt == "ascii":
            values = []
            for _ in range(nrows):
                line = stream.readline(65537)
                require(line and len(line) <= 65536, "truncated/oversized ASCII PLY payload")
                digest.update(line)
                fields = line.split()
                require(len(fields) == len(properties), "ASCII PLY row width mismatch")
                values.append(tuple(float(v) for v in fields))
            rows = np.array(values, dtype=dtype)
        else:
            payload = stream.read(nrows * dtype.itemsize)
            require(len(payload) == nrows * dtype.itemsize, "truncated PLY payload")
            digest.update(payload)
            rows = np.frombuffer(payload, dtype=dtype)
        require(all(np.isfinite(rows[n]).all() for n in names), f"nonfinite PLY payload at chunk beginning {seen}")
        xyz = np.column_stack([rows[n] for n in ("x", "y", "z")])
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
        if gaussian:
            quat = np.column_stack([rows[f"rot_{i}"].astype(float) for i in range(4)])
            norms = np.linalg.norm(quat, axis=1)
            require(np.all(np.abs(norms - 1) <= 1e-3), f"non-unit/zero PLY quaternion at chunk beginning {seen}")
            qmin, qmax = min(qmin, float(norms.min())), max(qmax, float(norms.max()))
            require(all(np.all((rows[f"scale_{i}"] > -80) & (rows[f"scale_{i}"] < 80)) for i in range(3)),
                    "PLY log scale cannot yield usable finite positive float32 axes")
        if compare is not None:
            compare(rows, seen)
        seen += nrows
    tail = stream.read(1)
    require(not tail, "undeclared trailing PLY payload")
    result = {"vertices": seen, "properties": names, "sh_degree": degree, "sha256": digest.hexdigest(),
              "payload_all_finite": True, "all_vertices_checked": True,
              "bbox_m": {"min": minimum.tolist(), "max": maximum.tolist()}, "encoding": fmt}
    if gaussian:
        result["quat_norm_range"] = [qmin, qmax]
    return result


def check_ply(path, **kwargs):
    before = path.stat()
    with path.open("rb") as stream:
        result = scan_ply(stream, **kwargs)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "PLY changed during audit; rerun")
    return result


def check_surface_reference(path):
    # One prepared reference at a time (the current producer caps at 400k).
    with zipfile.ZipFile(path) as archive:
        require(sum(v.file_size for v in archive.infolist()) <= 256*1024*1024,
                "LiDAR reference exceeds 256 MiB decoded CPU bound")
    with np.load(path, allow_pickle=False) as data:
        xyz, normals, valid = data["xyz"], data["normals"], data["normal_valid"]
        require(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0, "invalid LiDAR reference shape")
        require(normals.shape == xyz.shape and valid.shape == (len(xyz),), "LiDAR normal support shape mismatch")
        for start in range(0, len(xyz), 32768):
            end = start + 32768
            require(np.isfinite(xyz[start:end]).all() and np.isfinite(normals[start:end]).all()
                    and np.isfinite(valid[start:end]).all(), "nonfinite LiDAR reference")
            require(np.isin(valid[start:end], [0, 1]).all(), "invalid LiDAR support flags")
            norms = np.linalg.norm(normals[start:end][valid[start:end].astype(bool)], axis=1)
            require(np.all(np.abs(norms - 1) < .01), "invalid measured surface normals")
        require(np.any(valid), "no measured surfel normal support")
        return {"points": len(xyz), "normal_supported_points": int(np.count_nonzero(valid)), "all_finite": True}


def validate_depth(depth, height, width):
    require(depth.shape == (height, width) and np.issubdtype(depth.dtype, np.floating), "prepared depth shape/type mismatch")
    for start in range(0, height, 128):
        part = depth[start:start+128]
        require(np.isfinite(part).all() and np.all(part >= 0), "nonfinite/negative prepared camera-z depth")
    require(np.any(depth > 0), "prepared image has no actual LiDAR depth support")


def quaternion_matrix(quat):
    x, y, z, w = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_trajectory(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) >= 2, "trajectory requires at least two poses")
    require({r["frame_id"] for r in rows} == {"camera_init"}, "actual CSV world frame is not camera_init")
    require({r["child_frame_id"] for r in rows} == {"aft_mapped"}, "actual CSV body frame is not aft_mapped")
    stamps = np.array([int(r["header_ns"]) for r in rows], dtype=np.int64)
    values = np.array([[float(r[k]) for k in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")] for r in rows])
    require(np.all(np.diff(stamps) > 0) and np.isfinite(values).all(), "nonfinite/non-monotonic trajectory")
    require(np.all(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1) < .001), "invalid trajectory quaternion")
    distances = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
    require(np.all(distances < 5), "trajectory jumps >5m")
    return stamps, values


def pose_at(stamps, values, stamp):
    require(stamps[0] <= stamp <= stamps[-1], "prepared pose timestamp outside trajectory")
    i = max(0, min(bisect.bisect_right(stamps, stamp) - 1, len(stamps) - 2))
    gap = int(stamps[i+1]) - int(stamps[i])
    require(gap <= 250000000, "trajectory interpolation gap >250ms")
    alpha = (stamp - int(stamps[i])) / gap
    q0, q1 = values[i, 3:].copy(), values[i+1, 3:].copy()
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        dot, q1 = -dot, -q1
    if dot > .9995:
        q = (1-alpha)*q0 + alpha*q1
    else:
        theta = math.acos(np.clip(dot, -1, 1))
        q = (math.sin((1-alpha)*theta)*q0 + math.sin(alpha*theta)*q1) / math.sin(theta)
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(q)
    result[:3, 3] = (1-alpha)*values[i, :3] + alpha*values[i+1, :3]
    return result


def expected_camera_pose(stamps, values, frame, calibration):
    camera = calibration["cameras"][frame["camera_id"]]
    stamp = int(frame["source_header_ns"]) + int(camera["time_offset_ns"])
    require(stamp == int(frame["timestamp_ns"]), "prepared camera timestamp not derived from source/calibration")
    views = [v for v in calibration["virtual_views"] if v["name"] == frame["view_id"]]
    require(len(views) == 1, "unknown prepared virtual view")
    angle = math.radians(float(views[0].get("yaw_deg", 0)))
    virtual = np.eye(4)
    virtual[:3, :3] = [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    return pose_at(stamps, values, stamp) @ np.asarray(camera["T_body_camera"], float) @ virtual @ np.diag([1., -1., -1., 1.])


def global_partition(index):
    return "train" if index % 8 else ("dev" if index % 16 == 8 else "test")


def validate_splits(meta, source_rows, group_indices, source_root, root, registry, digest_cache):
    frames = meta["frames"]
    require(frames, "no prepared frames")
    byname = {f["file_path"]: f for f in frames}
    require(len(byname) == len(frames), "duplicate prepared image path")
    partitions = {}
    for split, key in SPLIT_KEYS.items():
        paths = meta[key]
        require(paths and len(set(paths)) == len(paths), f"empty/duplicated {split} split")
        for name in paths:
            require(name in byname and name not in partitions, "split overlap or unknown split image")
            partitions[name] = split
    require(set(partitions) == set(byname), "unassigned prepared frames")
    groups, sources = {}, {}
    for frame in frames:
        split = partitions[frame["file_path"]]
        group = str(frame["group_id"])
        require(group in group_indices and split == global_partition(group_indices[group]),
                "group split differs from actual global source index")
        require(group not in groups or groups[group] == split, "one camera group leaks across splits")
        groups[group] = split
        camera, name = frame["camera_id"], frame["source_file_path"]
        row = source_rows.get((camera, name))
        require(row is not None and int(row["header_ns"]) == int(frame["source_header_ns"])
                and int(row["bag_ns"]) == int(frame["source_bag_ns"]), "source photo/timestamps not present in original index")
        source = project_path(name, root, source_root)
        require(source.is_file(), "missing original source photograph")
        if source not in digest_cache:
            digest_cache[source] = sha256(source)
        identities = [("source", str(source)), ("photo", digest_cache[source]),
                      ("header", camera, int(frame["source_header_ns"]))]
        for identity in identities:
            require(identity not in registry or registry[identity] == split,
                    "source photograph leaks across train/dev/test (including overlapping blocks)")
            registry[identity] = split
        sources[(camera, name)] = split
    return partitions, {"groups": len(groups), "source_photos": len(sources),
                        "split_frames": dict(Counter(partitions.values())),
                        "protocol": "global camera_2 index: train i%8!=0; dev i%16==8; test i%16==0",
                        "source_content_hashes_checked": True}


def interval_gaps(intervals, start, end):
    cursor, gaps = start, []
    for left, right in sorted(intervals):
        if right < cursor or left > end:
            continue
        if left > cursor:
            gaps.append([cursor, min(left, end)])
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append([cursor, end])
    return gaps


def validate_metrics(report, split, meta, partitions, config, checkpoint, target, root):
    require(report.get("status") == "evaluated" and report.get("split") == split, "evaluation status/split mismatch")
    require(project_path(report["config"], root) == config and project_path(report["checkpoint"], root) == checkpoint,
            "evaluation consumes another model/config/checkpoint")
    require(int(report["step"]) == target - 1, "evaluation predates final target checkpoint")
    identity_transform(report["world_parser_transform"], report["world_parser_scale"])
    records = report["frames"]
    expected = {n for n, part in partitions.items() if part == split}
    actual = [r["file_path"] for r in records]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "evaluation omits/duplicates held-out images")
    require(int(report["images"]) == len(records), "evaluation image count mismatch")
    byname = {f["file_path"]: f for f in meta["frames"]}
    psnr, ssim, depth = [], [], []
    for record in records:
        frame = byname[record["file_path"]]
        require(str(record["group_id"]) == str(frame["group_id"]) and record["camera_id"] == frame["camera_id"],
                "evaluation photo/group identity mismatch")
        p, s, valid = float(record["psnr_db"]), float(record["ssim_valid_pixels"]), float(record["valid_fraction"])
        require(math.isfinite(p) and math.isfinite(s) and -1 <= s <= 1 and 0 < valid <= 1, "invalid actual per-frame metrics")
        psnr.append(p)
        ssim.append(s)
        if "map_depth_compared_pixels" in record:
            n = int(record["map_depth_compared_pixels"])
            rmse = float(record["map_projected_depth_rmse_m"])
            median = float(record["map_projected_depth_median_abs_m"])
            require(n > 0 and math.isfinite(rmse) and math.isfinite(median) and rmse >= 0 and median >= 0,
                    "invalid conditioned depth metric")
            depth.append((n, rmse, median))
    p, s = float(np.mean(psnr)), float(np.mean(ssim))
    require(abs(p - float(report["psnr_db_mean"])) < 1e-5 and abs(s - float(report["ssim_valid_mean"])) < 1e-5,
            "reported PSNR/SSIM means disagree with actual frame rows")
    geometry = {}
    for name in ("gaussian_to_initializer_median_m", "gaussian_to_initializer_p95_m",
                 "gaussian_to_measured_map_median_m", "gaussian_to_measured_map_p95_m"):
        value = float(report[name])
        require(math.isfinite(value) and value >= 0, "invalid conditioned geometry metric")
        geometry[name] = value
    require(depth, "missing conditioned depth measurements")
    return {"images": len(records), "psnr_db_frame_mean": p, "ssim_valid_frame_mean": s,
            "protocol": CONDITIONED, "calibration_status_recorded": report.get("calibration_status", "unknown"),
            "source_geometry_protocol_recorded": report.get("geometry_protocol", "unknown"),
            "geometry": geometry, "depth_compared_pixels": sum(n for n, _, _ in depth),
            "depth_rmse_m_pixel_weighted": math.sqrt(sum(n*r*r for n, r, _ in depth) / sum(n for n, _, _ in depth)),
            "depth_median_abs_m_frame_mean": float(np.mean([m for _, _, m in depth]))}


def checkpoint_archive(path):
    require(path.stat().st_size > 0, "empty checkpoint")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(any(n.endswith("/data.pkl") for n in names) and any("/data/" in n for n in names),
                "checkpoint is not a tensor archive")
    return {"path": str(path), "bytes": path.stat().st_size, "archive_valid": True,
            "payload_status": "unknown (use --verify-checkpoints)", "step_verified": False}


def verify_checkpoint(path, ply, target, chunk_bytes):
    """Restricted CPU load, all tensor values, plus actual exported GS payload."""
    import torch
    torch.set_num_threads(1)
    # Nerfstudio 1.1.5 records NumPy scalar scheduler values. Allow only these
    # fixed numeric types, never arbitrary config/trainer objects or functions.
    safe = [np.core.multiarray.scalar, np.dtype, codecs.encode]
    safe += list({type(np.dtype(t)) for t in (np.int32, np.int64, np.float32, np.float64)})
    state = None
    before = path.stat()
    try:
        previous_safe = torch.serialization.get_safe_globals()
        torch.serialization.add_safe_globals(safe)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous_safe)
        require(isinstance(state, dict) and {"step", "pipeline", "optimizers", "schedulers", "scalers"} <= set(state),
                "checkpoint lacks actual training/model state")
        step = int(state["step"])
        match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
        require(match and int(match[1]) == step, "checkpoint payload step differs from filename")
        if step < target - 1:
            raise IncompleteError(f"checkpoint payload step {step}; target last zero-based step {target-1}")
        require(step == target - 1, "checkpoint payload step exceeds saved delivery target")
        count = 0

        def visit(value):
            nonlocal count
            if isinstance(value, torch.Tensor):
                require(value.device.type == "cpu", "checkpoint tensor escaped CPU")
                flat = value.reshape(-1)
                for start in range(0, flat.numel(), 262144):
                    require(bool(torch.isfinite(flat[start:start+262144]).all()), "nonfinite checkpoint tensor")
                count += 1
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, (float, np.floating)):
                require(math.isfinite(float(value)), "nonfinite checkpoint scalar")
        visit(state)
        params = {}
        for name in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
            candidates = [v for k, v in state["pipeline"].items() if k.endswith("gauss_params." + name)]
            require(len(candidates) == 1 and isinstance(candidates[0], torch.Tensor), "checkpoint lacks Gaussian " + name)
            params[name] = candidates[0].detach().numpy()
        n = len(params["means"])
        require(n > 0 and params["means"].shape == (n, 3) and params["scales"].shape == (n, 3)
                and params["quats"].shape == (n, 4) and params["features_dc"].shape == (n, 3)
                and params["opacities"].shape == (n, 1) and params["features_rest"].ndim == 3
                and params["features_rest"].shape[0] == n and params["features_rest"].shape[2] == 3,
                "invalid actual Gaussian tensor shapes")
        norms = np.linalg.norm(params["quats"].astype(float), axis=1)
        require(np.isfinite(norms).all() and np.all(norms > 1e-12), "zero/invalid checkpoint quaternion")
        ids = np.flatnonzero(params["opacities"][:, 0] >= -5.5373)
        require(len(ids) > 0, "checkpoint exports no Gaussian payload")

        def compare(rows, offset):
            selected = ids[offset:offset+len(rows)]
            require(len(selected) == len(rows), "PLY contains more Gaussians than actual checkpoint export")
            expected = {"opacity": params["opacities"][selected, 0]}
            for i, name in enumerate(("x", "y", "z")):
                expected[name] = params["means"][selected, i]
                expected[("nx", "ny", "nz")[i]] = np.zeros(len(selected))
                expected[f"scale_{i}"] = params["scales"][selected, i]
                expected[f"f_dc_{i}"] = params["features_dc"][selected, i]
            rest = params["features_rest"][selected].transpose(0, 2, 1).reshape(len(selected), -1)
            for i in range(rest.shape[1]):
                expected[f"f_rest_{i}"] = rest[:, i]
            for i in range(4):
                expected[f"rot_{i}"] = params["quats"][selected, i] / norms[selected]
            require(set(expected) == set(rows.dtype.names), "PLY schema differs from actual model export")
            for key, value in expected.items():
                require(np.allclose(rows[key], value, atol=1e-6, rtol=1e-5), "PLY differs from checkpoint Gaussian " + key)
        scanned = check_ply(ply, chunk_bytes=chunk_bytes, compare=compare)
        require(scanned["vertices"] == len(ids), "PLY omits actual exported checkpoint Gaussians")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "checkpoint changed during audit; rerun")
        return {"step_verified": True, "payload_status": "verified_cpu", "step": step,
                "tensor_count": count, "all_tensors_finite": True, "model_gaussians": n,
                "ply_matches_checkpoint": True}, scanned
    finally:
        del state
        gc.collect()


class Auditor:
    def __init__(self, root=ROOT, target=12000, verify=False, chunk_bytes=8*1024*1024, floor_schedule=FLOOR_SCHEDULE):
        self.root, self.target, self.verify = root.resolve(), target, verify
        self.chunk_bytes, self.floor_schedule = chunk_bytes, Path(floor_schedule).resolve()
        self.issues, self.registry, self.digest_cache, self.hash_cache = [], {}, {}, {}
        self.common_calibration_hash = None

    def issue(self, block, stage, category, detail):
        self.issues.append({"block_id": block, "stage": stage, "category": category, "detail": str(detail)})

    def attempt(self, block, stage, function):
        try:
            return function()
        except FileNotFoundError as exc:
            self.issue(block, stage, "missing", exc)
        except IncompleteError as exc:
            self.issue(block, stage, "incomplete", exc)
        except (ImportError, ModuleNotFoundError) as exc:
            self.issue(block, stage, "unknown", exc)
        except Exception as exc:
            self.issue(block, stage, "invalid", f"{type(exc).__name__}: {exc}")
        return None

    def hashed(self, path):
        if path not in self.hash_cache:
            self.hash_cache[path] = sha256(path)
        return self.hash_cache[path]

    def run(self):
        root = self.root
        manifest = self.attempt(None, "manifest", lambda: read_json(project_path("maps/garage_manifest.json", root))) or {}
        plan = self.attempt(None, "candidate_plan", lambda: read_json(project_path("reports/lio_candidate_blocks.json", root))) or {}
        def records(value):
            require(isinstance(value, list) and all(isinstance(b, dict) and isinstance(b.get("id"), str) for b in value),
                    "blocks must be a list of objects with string IDs")
            return value
        planned = self.attempt(None, "candidate_block_records", lambda: records(plan.get("blocks", []))) or []
        items = self.attempt(None, "manifest_block_records", lambda: records(manifest.get("blocks", []))) or []
        self.attempt(None, "expected_ids", lambda: require(
            len(planned) == 28 and {b["id"] for b in planned} == set(EXPECTED_IDS)
            and len(items) == 28 and {b["id"] for b in items} == set(EXPECTED_IDS),
            "candidate plan and manifest must each contain exactly all 28 unique expected IDs"))
        self.attempt(None, "manifest_frame", lambda: require(
            manifest.get("coordinates") == "camera_init" and manifest.get("units") == "m"
            and manifest.get("dataset_kind") == "real", "manifest global metric/real metadata invalid"))
        run_id = manifest.get("run_id", "garage_surface_v1")
        self.attempt(None, "run_id", lambda: require(run_id == "garage_surface_v1", "audit is scoped to garage_surface_v1"))
        # Always inspect the required run, even if manifest tries to redirect it.
        byplan = {b.get("id"): b for b in planned}
        byitem = {b.get("id"): b for b in items}
        trajectory = self.attempt(None, "trajectory", lambda: self.trajectory(plan))
        source = self.attempt(None, "source_index", self.source_index)
        schedule = self.attempt(None, "estimated_floor_schedule", lambda: read_json(self.floor_schedule))
        blocks = []
        for block_id in EXPECTED_IDS:
            blocks.append(self.block(block_id, byplan.get(block_id), byitem.get(block_id), trajectory, source))
        coverage = self.attempt(None, "coverage", lambda: self.coverage(planned, blocks, trajectory, source, schedule))
        all_complete = all(b["status"] == "complete" for b in blocks) and not self.issues
        aggregate = {}
        for split in ("dev", "test"):
            available = [b["metrics"][split] for b in blocks if split in b["metrics"]]
            images = sum(m["images"] for m in available)
            aggregate[split] = {"reported_blocks": len(available), "frame_observations": images,
                "psnr_db_frame_weighted_mean": sum(m["images"]*m["psnr_db_frame_mean"] for m in available)/images if images else None,
                "ssim_valid_frame_weighted_mean": sum(m["images"]*m["ssim_valid_frame_mean"] for m in available)/images if images else None,
                "note": "Means of reported frame metrics; overlapping blocks can repeat held-out photos. Not pooled-pixel PSNR or an independent-GT score.",
                "protocol": CONDITIONED}
        return {"schema_version": 1, "audit_kind": "complete_first_delivery_real_gs_cpu",
                "audit_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
                "status": "complete_first_delivery_with_limitations" if all_complete else "partial",
                "complete": all_complete, "high_fidelity_acceptance": "not_assessed",
                "target_iterations": self.target, "expected_blocks": 28,
                "checkpoint_payload_validation": self.verify,
                "block_status_counts": dict(Counter(b["status"] for b in blocks)),
                "issue_category_counts": dict(Counter(i["category"] for i in self.issues)),
                "calibration_verified_recorded": manifest.get("calibration_verified", "unknown"),
                "floor_assignment_recorded": manifest.get("floor_assignment", "unknown"),
                "coverage": coverage, "metrics_summary": aggregate, "blocks": blocks,
                "issues": self.issues, "limitations": LIMITATIONS,
                "snapshot_note": "Read-only live snapshot; files modified by training/evaluation may require a rerun."}

    def trajectory(self, plan):
        path = project_path(plan["pose_csv"], self.root)
        require(path == self.root / "data/all_lio_candidate/odometry.csv", "candidate trajectory source changed")
        stamps, values = read_trajectory(path)
        quality = plan["pose_quality"]
        require(int(quality["start_header_ns"]) == int(stamps[0]) and int(quality["end_header_ns"]) == int(stamps[-1]),
                "plan trajectory endpoints disagree with actual CSV")
        return path, self.hashed(path), stamps, values

    def source_index(self):
        source_root = project_path("data/keyframes_2hz", self.root)
        meta = read_json(source_root / "manifest.json")
        require(meta.get("status") == "complete" and not (source_root / "INCOMPLETE.json").exists(), "source extraction incomplete")
        path = source_root / "indexes/images.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        keyed = {(r["camera_id"], r["file_path"]): r for r in rows}
        require(len(keyed) == len(rows), "duplicate original source photo index")
        c2 = sorted([r for r in rows if r["camera_id"] == "camera_2"], key=lambda r: int(r["header_ns"]))
        require(c2 and len({r["header_ns"] for r in c2}) == len(c2), "invalid global camera_2 source groups")
        return source_root, self.hashed(path), keyed, c2

    def preparation(self, block_id, dataset, planned, trajectory, source):
        prep = read_json(dataset / "preparation.json")
        meta = read_json(dataset / "transforms.json")
        require(prep.get("status") in ("prepared", "prepared_experimental") and not (dataset / "INCOMPLETE.json").exists(),
                "preparation incomplete")
        require(prep.get("block_id") == block_id, "preparation belongs to another block")
        require(prep.get("world_frame") == "camera_init" and prep.get("units") in ("m", "metres", "meters"),
                "prepared world is not global camera_init metres")
        require(trajectory is not None and source is not None, "cannot verify preparation without actual trajectory/source index")
        pose_path, pose_hash, stamps, values = trajectory
        source_root, index_hash, source_rows, c2 = source
        require(project_path(prep["pose_csv"], self.root) == pose_path and prep["pose_sha256"] == pose_hash,
                "prepared trajectory differs from actual common CSV")
        require(project_path(prep["source_extraction"], self.root) == source_root
                and prep["split_reference_images_sha256"] == index_hash, "prepared global split source/hash changed")
        cal_path = project_path(prep["calibration"], self.root)
        require(prep["calibration_sha256"] == self.hashed(cal_path), "calibration file differs from preparation")
        cal = read_config(cal_path)
        cal_hash = self.hashed(cal_path)
        require(self.common_calibration_hash in (None, cal_hash), "blocks use inconsistent prepared calibrations")
        self.common_calibration_hash = cal_hash
        require(cal["pose"]["world_frame"] == "camera_init" and cal["pose"]["body_frame"] == "aft_mapped",
                "calibration pose semantics differ from common trajectory")
        indices = {str(int(r["header_ns"]) + int(cal["cameras"]["camera_2"]["time_offset_ns"])): i for i, r in enumerate(c2)}
        partitions, split_report = validate_splits(meta, source_rows, indices, source_root, self.root, self.registry, self.digest_cache)
        require(planned is not None, "missing planned block")
        begin, end = int(planned["start_header_ns"]), int(planned["end_header_ns"])
        expected = {g for g in indices if begin <= int(g) <= end and stamps[0] <= int(g) <= stamps[-1]}
        actual = {str(f["group_id"]) for f in meta["frames"]}
        require(actual == expected, f"prepared source group coverage incomplete: missing={len(expected-actual)}, extra={len(actual-expected)}")
        expected_views = {(camera, view["name"]) for camera in cal["cameras"] for view in cal["virtual_views"]}
        group_views = {}
        for frame in meta["frames"]:
            group_views.setdefault(str(frame["group_id"]), []).append((frame["camera_id"], frame["view_id"]))
        require(all(len(views) == len(expected_views) and set(views) == expected_views for views in group_views.values()),
                "prepared groups omit/duplicate physical cameras or virtual views")
        max_error = 0.
        for frame in meta["frames"]:
            T = np.asarray(frame["transform_matrix"], float)
            expected_T = expected_camera_pose(stamps, values, frame, cal)
            require(T.shape == (4, 4) and np.isfinite(T).all(), "invalid actual prepared camera transform")
            error = float(np.max(np.abs(T - expected_T)))
            require(error < 1e-5, "prepared camera transform differs from actual LIO/calibration; centering/autoscale suspected")
            max_error = max(error, max_error)
            require(abs(int(frame["timestamp_ns"]) - int(frame["group_id"])) <= 50000000,
                    "prepared frame assigned to a different source time group (>50ms)")
            for key in ("file_path", "mask_path", "lidar_depth_path", "lidar_depth_mask_path"):
                asset = project_path(frame[key], self.root, dataset)
                require(asset.is_file() and asset.stat().st_size > 0, "missing actual prepared asset: " + str(asset))
            depth = np.load(project_path(frame["lidar_depth_path"], self.root, dataset), mmap_mode="r", allow_pickle=False)
            validate_depth(depth, int(frame["h"]), int(frame["w"]))
            del depth
        require(int(prep["frames"]) == len(meta["frames"]) and int(prep["groups"]) == len(actual), "preparation counts disagree")
        initializer = project_path(meta["ply_file_path"], self.root, dataset)
        init_report = check_ply(initializer, chunk_bytes=self.chunk_bytes, gaussian=False)
        require({"red", "green", "blue"} <= set(init_report["properties"]), "actual initializer lacks observed image color channels")
        require(init_report["vertices"] == int(prep["export"]["point_count"]), "initializer actual point count mismatch")
        require(prep["map"]["world_frame"] == "camera_init", "LiDAR source frame mismatch")
        require(project_path(prep["lidar_reference_path"], self.root, dataset) == dataset / "lidar_reference.npz",
                "prepared LiDAR reference redirection")
        ref = check_surface_reference(dataset / "lidar_reference.npz")
        return prep, meta, partitions, {"status": "verified", "pose_sha256": pose_hash,
            "source_index_sha256": index_hash, "calibration_verified_recorded": prep.get("calibration_verified", "unknown"),
            "calibration_config_verified_recorded": cal.get("verified", "unknown"), "calibration_sha256": cal_hash,
            "lidar_source_map_recorded": prep["map"].get("source", "unknown"),
            "lidar_alignment_operator_flag_recorded": prep["map"].get("alignment_verified_by_operator", "unknown"),
            "pose_max_element_error": max_error, "actual_group_range_ns": [min(map(int, actual)), max(map(int, actual))],
            "global_split": split_report, "initializer": init_report, "surface_reference": ref}

    def block(self, block_id, planned, item, trajectory, source):
        root = self.root
        output = project_path(f"runs/garage_surface_v1/{block_id}", root)
        dataset = project_path(f"data/garage_surface_v1/{block_id}", root)
        result = {"id": block_id, "status": "missing", "metrics": {}, "floor_id_recorded": (item or {}).get("floor_id", "unknown")}
        result["floor_verified_recorded"] = (item or {}).get("floor_verified", "unknown")
        result["manifest_training_status_recorded"] = (item or {}).get("training", {}).get("status", "unknown")
        start_issue = len(self.issues)
        prepared = self.attempt(block_id, "preparation", lambda: self.preparation(block_id, dataset, planned, trajectory, source))
        if prepared:
            result["preparation"] = prepared[3]
        training_path = output / "training_result.json"
        trained = self.attempt(block_id, "training_result", lambda: read_json(training_path))
        configs = []
        if trained:
            self.attempt(block_id, "training_target", lambda: require(
                trained.get("status") == "trained" and int(trained["iterations_requested"]) == self.target,
                "training_result not trained to requested target"))
            configs = trained.get("configs", [])
        if not configs and output.exists():
            configs = list(output.rglob("config.yml"))
        config = self.attempt(block_id, "config_path", lambda: self.single_config(configs))
        checkpoint = None
        if config:
            saved = self.attempt(block_id, "config", lambda: read_config(config))
            if saved:
                self.attempt(block_id, "renderer_and_world_config", lambda: validate_config(saved, dataset, root, self.target))
            parser_world = self.attempt(block_id, "dataparser_transforms", lambda: read_json(config.parent / "dataparser_transforms.json"))
            if parser_world:
                self.attempt(block_id, "global_world", lambda: identity_transform(parser_world["transform"], parser_world["scale"]))
            checkpoint = self.attempt(block_id, "checkpoint", lambda: self.last_checkpoint(config))
            result["config"] = str(config.relative_to(root))
        ply = self.attempt(block_id, "map_ply_path", lambda: self.map_ply(item, output))
        if checkpoint:
            archive = self.attempt(block_id, "checkpoint_archive", lambda: checkpoint_archive(checkpoint))
            if archive:
                result["checkpoint"] = archive
            step = int(re.fullmatch(r"step-(\d+)\.ckpt", checkpoint.name)[1])
            result["checkpoint_filename_step_unverified"] = step
            if step != self.target - 1:
                self.issue(block_id, "training_steps", "incomplete", f"saved filename step {step}; target last zero-based step {self.target-1}")
            if self.verify and ply:
                verified = self.attempt(block_id, "checkpoint_payload_and_ply", lambda: verify_checkpoint(checkpoint, ply, self.target, self.chunk_bytes))
                if verified:
                    result.setdefault("checkpoint", {}).update(verified[0])
                    result["ply"] = verified[1]
            else:
                self.issue(block_id, "checkpoint_payload", "unknown", "actual step/tensors/export association require --verify-checkpoints")
        if ply and "ply" not in result:
            scanned = self.attempt(block_id, "ply_payload", lambda: check_ply(ply, chunk_bytes=self.chunk_bytes))
            if scanned:
                result["ply"] = scanned
        for split in ("dev", "test"):
            metric_path = output / f"eval_{split}/metrics.json"
            report = self.attempt(block_id, f"{split}_report", lambda p=metric_path: read_json(p))
            if report and prepared and config and checkpoint:
                m = self.attempt(block_id, f"{split}_metrics", lambda r=report, s=split: validate_metrics(
                    r, s, prepared[1], prepared[2], config, checkpoint, self.target, root))
                if m:
                    result["metrics"][split] = m
                if split == "dev" and "ply" in result:
                    self.attempt(block_id, "export_provenance", lambda: self.export_provenance(report, result["ply"], item))
        issues = self.issues[start_issue:]
        categories = {i["category"] for i in issues}
        result["status"] = ("invalid" if "invalid" in categories else "incomplete" if "incomplete" in categories
                            else "prepared_or_in_progress" if "missing" in categories and prepared
                            else "missing" if "missing" in categories else "unknown" if categories else "complete")
        result["issues"] = issues
        return result

    def single_config(self, configs):
        if not configs:
            raise FileNotFoundError("missing saved config")
        require(len(configs) == 1, "expected exactly one actual config (including reused source_run)")
        path = project_path(configs[0], self.root)
        if not path.is_file():
            raise FileNotFoundError(f"missing config: {path}")
        return path

    def last_checkpoint(self, config):
        paths = list((config.parent / "nerfstudio_models").glob("step-*.ckpt"))
        if not paths:
            raise FileNotFoundError("missing actual checkpoint")
        for path in paths:
            require(re.fullmatch(r"step-(\d+)\.ckpt", path.name), "invalid checkpoint name")
        return project_path(max(paths, key=lambda p: int(p.stem.split("-")[-1])), self.root)

    def map_ply(self, item, output):
        require(item is not None and item.get("coordinates") == "camera_init", "missing/wrong manifest block world")
        path = project_path(item["path"], self.root)
        require(path == output / "eval_dev/gaussians.ply", "map PLY does not come from this block actual dev export")
        return path

    def export_provenance(self, report, ply, item):
        export = report["export"]
        require(export["coordinates"] == "camera_init" and int(export["gaussians"]) == ply["vertices"]
                and int(export["sh_degree"]) == ply["sh_degree"], "actual PLY export metadata mismatch")
        for bound in ("min", "max"):
            require(np.allclose(export["bbox_m"][bound], ply["bbox_m"][bound], atol=1e-4, rtol=0), "PLY/export bbox mismatch")
            if not np.allclose(item["bbox_m"][bound], ply["bbox_m"][bound], atol=1e-4, rtol=0):
                if item.get("training", {}).get("status") != "trained":
                    raise IncompleteError("actual model exported but manifest bbox not yet published; rerun after publisher catches up")
                raise AuditError("manifest/actual PLY bbox mismatch")

    def coverage(self, planned, blocks, trajectory, source, schedule):
        require(trajectory is not None and source is not None and schedule is not None, "coverage cannot be verified")
        _, _, stamps, values = trajectory
        intervals = [(int(b["start_header_ns"]), int(b["end_header_ns"])) for b in planned]
        require(all(a <= b for a, b in intervals), "reversed candidate interval")
        gaps = interval_gaps(intervals, int(stamps[0]), int(stamps[-1]))
        require(not gaps, f"planned LIO interval gaps: {gaps}")
        mask = np.zeros(len(stamps), bool)
        for begin, end in intervals:
            mask |= (stamps >= begin) & (stamps <= end)
        require(mask.all(), "planned blocks omit actual LIO poses")
        prepared_ids = {b["id"] for b in blocks if "preparation" in b}
        completed_ids = {b["id"] for b in blocks if b["status"] == "complete"}
        prepared_intervals = [v for b, v in zip(planned, intervals) if b["id"] in prepared_ids]
        completed_intervals = [v for b, v in zip(planned, intervals) if b["id"] in completed_ids]
        prepared_gaps = interval_gaps(prepared_intervals, int(stamps[0]), int(stamps[-1]))
        require(not prepared_gaps, f"actual prepared trajectory has omitted intervals: {prepared_gaps}")
        floors, ramps = schedule["floors"], schedule["ramps"]
        require(len(floors) == 6 and {int(f["floor_id"]) for f in floors} == set(range(1, 7))
                and {(int(r["lower_floor_id"]), int(r["upper_floor_id"])) for r in ramps} == {(i, i+1) for i in range(1, 6)},
                "estimated schedule lacks six floors/five adjacent ramps")
        for block in blocks:
            candidate = next((b for b in planned if b["id"] == block["id"]), None)
            if candidate is None:
                continue
            begin, end = candidate["start_header_ns"]/1e9, candidate["end_header_ns"]/1e9
            span, label = max(.001, end-begin), None
            for floor in floors:
                overlap = max(0, min(end, floor["end_stamp"])-max(begin, floor["start_stamp"]))
                if overlap/span > .7:
                    label = f"F{floor['floor_id']}"
                    break
            if label is None:
                for ramp in ramps:
                    if begin <= ramp["end_stamp"] and end >= ramp["start_stamp"]:
                        label = f"坡道 F{ramp['lower_floor_id']}-F{ramp['upper_floor_id']}"
                        break
            require(block["floor_id_recorded"] == label, "manifest floor/ramp label disagrees with estimated schedule")
        visits = []
        all_groups = sorted({int(f["group_id"]) for b in blocks if "preparation" in b
                             for f in read_json(self.root / f"data/garage_surface_v1/{b['id']}/transforms.json")["frames"]})
        for entry in floors + ramps:
            begin, end = round(float(entry["start_stamp"]) * 1e9), round(float(entry["end_stamp"]) * 1e9)
            selected = (stamps >= begin) & (stamps <= end)
            observed = [g for g in all_groups if begin <= g <= end]
            require(selected.any() and observed, "estimated floor/ramp lacks actual poses or prepared camera groups")
            label = f"F{entry['floor_id']}" if "floor_id" in entry else f"ramp_F{entry['lower_floor_id']}_F{entry['upper_floor_id']}"
            visits.append({"label": label, "estimated_only": True, "actual_pose_count": int(selected.sum()),
                "actual_prepared_groups": len(observed), "actual_z_range_m": [float(values[selected, 2].min()), float(values[selected, 2].max())]})
        source_headers = [int(r["header_ns"]) for r in source[3]]
        return {"actual_lio_poses": len(stamps), "start_header_ns": int(stamps[0]), "end_header_ns": int(stamps[-1]),
                "source_camera_2_header_range_ns": [min(source_headers), max(source_headers)],
                "source_camera_2_groups_outside_retained_lio": int(sum(t < stamps[0] or t > stamps[-1] for t in source_headers)),
                "pose_coverage_denominator": "actual retained/exported LIO CSV, not every original bag message or building area",
                "planned_pose_coverage": 1., "prepared_interval_gaps_ns": prepared_gaps,
                "completed_interval_gaps_ns": interval_gaps(completed_intervals, int(stamps[0]), int(stamps[-1])),
                "actual_unique_prepared_groups": len(all_groups), "estimated_floor_ramp_visits": visits,
                "scope": "captured trajectory and visible regions; unobserved building areas excluded"}


def markdown_report(report):
    def metric(block, split, key, precision):
        value = block["metrics"].get(split, {}).get(key)
        return f"{value:.{precision}f}" if value is not None else "—"
    lines = ["# 全 28 块完整首版交付审计", "", f"状态：**{report['status']}**；目标 {report['target_iterations']} 步。",
             f"完整：{report['complete']}；high fidelity acceptance：**not assessed**。", "",
             "所有像素、深度与地图几何分数为 **same-source conditioned**，不是独立 GT。", "",
             "| Block | 状态 | dev PSNR | dev SSIM | test PSNR | test SSIM | GS |", "|---|---|---:|---:|---:|---:|---:|"]
    for block in report["blocks"]:
        lines.append(f"| {block['id']} | {block['status']} | {metric(block, 'dev', 'psnr_db_frame_mean', 3)} | "
                     f"{metric(block, 'dev', 'ssim_valid_frame_mean', 4)} | {metric(block, 'test', 'psnr_db_frame_mean', 3)} | "
                     f"{metric(block, 'test', 'ssim_valid_frame_mean', 4)} | {block.get('ply', {}).get('vertices', '—')} |")
    lines += ["", "## 同源条件深度/地图几何", "", "以下均为 same-source conditioned；没有独立 GT。",
              "", "| Block | split | depth pixels | pixel-weighted depth RMSE m | initializer p95 m | measured-map p95 m |",
              "|---|---|---:|---:|---:|---:|"]
    for block in report["blocks"]:
        for split, m in block["metrics"].items():
            lines.append(f"| {block['id']} | {split} | {m['depth_compared_pixels']} | {m['depth_rmse_m_pixel_weighted']:.4f} | "
                         f"{m['geometry']['gaussian_to_initializer_p95_m']:.4f} | {m['geometry']['gaussian_to_measured_map_p95_m']:.4f} |")
    lines += ["", "## 汇总", ""]
    for split, summary in report["metrics_summary"].items():
        lines.append(f"- {split}: {json.dumps(summary, ensure_ascii=False)}")
    lines += ["", "## 覆盖", "", json.dumps(report["coverage"], ensure_ascii=False), "", "## 未完成/错误", ""]
    for issue in report["issues"]:
        detail = issue["detail"].replace("\n", " ")
        lines.append(f"- [{issue['category']}] {issue['block_id'] or 'global'}/{issue['stage']}: {detail}")
    lines += ["", "## 限制", ""] + ["- " + s for s in report["limitations"]]
    lines += ["", "详细深度/几何指标及原始标定状态见 JSON 的各块 metrics/preparation。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-iterations", type=int, default=12000)
    parser.add_argument("--verify-checkpoints", action="store_true", help="restricted torch.load on CPU; check step, all tensors and exact PLY source")
    parser.add_argument("--require-complete", action="store_true", help="exit 1 unless all 28 blocks and coverage pass (payload verification required)")
    parser.add_argument("--chunk-mib", type=int, default=8, help="bounded PLY chunk size, 1..64 MiB")
    parser.add_argument("--floor-schedule", type=Path, default=FLOOR_SCHEDULE, help="read-only estimated schedule; external input allowed")
    parser.add_argument("--json", "--output-json", dest="json_output", type=Path, help="new project file only; stdout if omitted")
    parser.add_argument("--md", "--output-md", dest="md_output", type=Path, help="new project file only")
    args = parser.parse_args(argv)
    if args.target_iterations <= 0 or not 1 <= args.chunk_mib <= 64:
        parser.error("positive target and 1..64 chunk-mib required")
    # Validate ALL destinations before reading expensive data or writing anything.
    try:
        destinations = [output_path(p) if p else None for p in (args.json_output, args.md_output)]
        require(not destinations[0] or destinations[0] != destinations[1], "JSON and Markdown outputs must differ")
    except AuditError as exc:
        parser.error(str(exc))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    report = Auditor(target=args.target_iterations, verify=args.verify_checkpoints,
                     chunk_bytes=args.chunk_mib*1024*1024, floor_schedule=args.floor_schedule).run()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destinations[0]:
        with destinations[0].open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")
    if destinations[1]:
        with destinations[1].open("x", encoding="utf-8") as stream:
            stream.write(markdown_report(report))
    print(f"audit: {report['status']}; {report['block_status_counts']}", file=sys.stderr)
    return 1 if args.require_complete and not report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

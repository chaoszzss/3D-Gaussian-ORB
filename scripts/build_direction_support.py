#!/usr/bin/env python3
"""Build an opt-in, PLY-bound TRAIN hemisphere sidecar on CPU; no PLY edits."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from collections import defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from assemble_core_gaussians import (ROOT, AssemblyError, InputGuard, inspect_ply, local_path,
    resource_preflight, sha256, write_json)
from assemble_core_gaussians_lidar_ownership import axes_from_quaternion


def side_flags(positive, negative, minimum_groups=3):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if positive.shape != negative.shape or minimum_groups < 1:
        raise AssemblyError("Invalid direction evidence dimensions/minimum")
    flags = np.zeros(positive.shape, dtype=np.uint8)
    flags[(positive >= minimum_groups) & (negative == 0)] = 1
    flags[(negative >= minimum_groups) & (positive == 0)] = 2
    flags[(positive > 0) & (negative > 0)] = 3
    return flags


def frame_evidence(xyz, normals, frame, depth, depth_mask, image_mask, observed_cosine_min=.1):
    """Single TRAIN image evidence; no extrapolation where depth is unknown."""
    pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
    h, w = int(frame["h"]), int(frame["w"])
    if any(a.shape != (h, w) for a in (depth, depth_mask, image_mask)):
        raise AssemblyError("Evidence depth/masks must match prepared grayscale H/W")
    p = (xyz-pose[:3, 3]) @ (pose[:3, :3] @ np.diag([1., -1., -1.]))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = frame["fl_x"]*p[:, 0]/p[:, 2]+frame["cx"]-.5
        v = frame["fl_y"]*p[:, 1]/p[:, 2]+frame["cy"]-.5
    inside = (p[:, 2]>.2) & np.isfinite(u) & np.isfinite(v) & (u>=-.5) & (u<w-.5) & (v>=-.5) & (v<h-.5)
    ids = np.flatnonzero(inside)
    ui, vi = np.floor(u[ids]+.5).astype(np.int64), np.floor(v[ids]+.5).astype(np.int64)
    target = depth[vi, ui]
    known = (image_mask[vi, ui]>0) & (depth_mask[vi, ui]>0) & np.isfinite(target) & (target>.2)
    ids, target = ids[known], target[known]
    residual = p[ids, 2]-target
    foreground = np.abs(residual)<=np.minimum(.25, .1+.01*target)
    ids = ids[foreground]
    direction = pose[:3, 3]-xyz[ids]
    cosine = np.sum(direction*normals[ids], axis=1)/np.maximum(np.linalg.norm(direction, axis=1), 1e-12)
    return ids[cosine>=observed_cosine_min], ids[cosine<=-observed_cosine_min]


def source_evidence(project, source, original_indices, guard):
    start_time = time.monotonic()
    path = local_path(project, source["path"])
    guard.snapshot(path)
    if guard.files[path]["sha256"] != source["source_sha256"]:
        raise AssemblyError("Original source changed since assembly")
    with path.open("rb") as handle:
        spec = inspect_ply(handle)
    rows = np.memmap(path, mode="r", dtype="<f4", offset=len(spec["header"]), shape=(spec["count"], len(spec["names"])))
    columns = [spec["names"].index(n) for n in ("x", "y", "z")]
    xyz = np.asarray(rows[np.ix_(original_indices, columns)], dtype=np.float64)
    qcols = [spec["names"].index(f"rot_{i}") for i in range(4)]
    quats = np.asarray(rows[np.ix_(original_indices, qcols)], dtype=np.float64)
    del rows
    norms = np.linalg.norm(quats, axis=1)
    if not np.allclose(norms, 1., atol=1e-4, rtol=0):
        raise AssemblyError("Expected unchanged unit export quaternions")
    _, _, normals = axes_from_quaternion(quats)
    if len(source["core_ids"]) != 1:
        raise AssemblyError("Pilot evidence requires atomic sources")
    base = local_path(project, f"data/garage_surface_v1/{source['core_ids'][0]}")
    meta = guard.json(base/"transforms.json")
    train = set(meta["train_filenames"])
    heldout = set(meta.get("val_filenames", [])) | set(meta.get("test_filenames", []))
    if train & heldout:
        raise AssemblyError("TRAIN/held-out leakage")
    frames = defaultdict(list)
    for frame in meta["frames"]:
        if frame["file_path"] in train:
            frames[str(frame["group_id"])].append(frame)
    if not frames or len(frames)>65535:
        raise AssemblyError("Invalid TRAIN group count")
    guard.snapshot(local_path(project, source["measured_reference"]))
    positive, negative = np.zeros(len(xyz), dtype="<u2"), np.zeros(len(xyz), dtype="<u2")
    for number, (group, group_frames) in enumerate(sorted(frames.items())):
        plus, minus = np.zeros(len(xyz), dtype=bool), np.zeros(len(xyz), dtype=bool)
        for frame in group_frames:
            paths = {key: local_path(project, base/frame[key]) for key in ("lidar_depth_path", "lidar_depth_mask_path", "mask_path")}
            for path in paths.values():
                guard.snapshot(path)
            depth = np.load(paths["lidar_depth_path"], allow_pickle=False)
            with Image.open(paths["lidar_depth_mask_path"]) as image:
                depth_mask = np.asarray(image)
            with Image.open(paths["mask_path"]) as image:
                image_mask = np.asarray(image)
            pos, neg = frame_evidence(xyz, normals, frame, depth, depth_mask, image_mask)
            plus[pos] = True; minus[neg] = True
        positive += plus; negative += minus
        if (number+1)%20 == 0:
            print(json.dumps({"source": source["id"], "train_groups_done": number+1, "train_groups": len(frames)}), flush=True)
    flags = side_flags(positive, negative)
    report = {"id": source["id"], "source_sha256": source["source_sha256"], "retained_records": len(xyz),
              "train_groups": len(frames), "train_frames": sum(len(v) for v in frames.values()),
              "positive": int((flags==1).sum()), "negative": int((flags==2).sum()),
              "both": int((flags==3).sum()), "unknown": int((flags==0).sum()),
              "seconds": time.monotonic()-start_time,
              "normal": "PLY unit wxyz quaternion third axis; aligned with native surfel disk, not PLY nx/ny/nz",
              "distinct_train_group_ids": sorted(frames)}
    return positive, negative, report


def build(assembly, output, source_ids, *, project=ROOT):
    project = Path(project).resolve()
    destination = local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError("Output must be a NEW directory with an existing parent")
    guard = InputGuard(project)
    a = guard.json(local_path(project, assembly))
    if a["processing"]["ownership_mode"] != "none":
        raise AssemblyError("Initial pilot requires noownership assembly")
    sources = a["sources"]
    if source_ids is None:
        source_ids = [s["id"] for s in sources]
    if not source_ids or len(set(source_ids)) != len(source_ids) or set(source_ids)-{s["id"] for s in sources}:
        raise AssemblyError("Source IDs must be nonempty distinct source identifiers")
    count = a["retained_gaussians"]
    preflight = resource_preflight(destination, disk_bytes=count*5+32*1024**2, memory_bytes=2*1024**3)
    ply = local_path(project, a["full"]["path"])
    trace_path = local_path(project, a["full"]["provenance"]["path"])
    for path, expected in ((ply, a["full"]["sha256"]), (trace_path, a["full"]["provenance"]["sha256"])):
        guard.snapshot(path)
        if guard.files[path]["sha256"] != expected:
            raise AssemblyError("Model/provenance SHA changed since assembly")
    trace = np.load(trace_path, allow_pickle=False, mmap_mode="r")
    if len(trace) != count:
        raise AssemblyError("Model/provenance row mismatch")
    destination.mkdir()
    try:
        side_path = destination/"support_side_mask.npy"
        positive_path, negative_path = destination/"positive_train_groups.npy", destination/"negative_train_groups.npy"
        flags = np.lib.format.open_memmap(side_path, mode="w+", dtype="u1", shape=(count,))
        positive = np.lib.format.open_memmap(positive_path, mode="w+", dtype="<u2", shape=(count,))
        negative = np.lib.format.open_memmap(negative_path, mode="w+", dtype="<u2", shape=(count,))
        flags[:] = 0; positive[:] = 0; negative[:] = 0
        records = []
        for source in sources:
            if source["id"] not in source_ids:
                continue
            indices = np.flatnonzero(trace["source_index"]==source["source_index"])
            original = trace["source_record_index"][indices].astype(np.int64)
            pos, neg, report = source_evidence(project, source, original, guard)
            positive[indices], negative[indices] = pos, neg
            flags[indices] = side_flags(pos, neg)
            records.append(report)
            print(json.dumps(report), flush=True)
        for array in (flags, positive, negative):
            array.flush()
        distribution = {str(i): int(np.count_nonzero(flags==i)) for i in range(4)}
        del flags, positive, negative
        guard.check({ply})
        metadata = {"schema_version": 1,
                    "ply": {"path": a["full"]["path"], "sha256": a["full"]["sha256"], "gaussians": count},
                    "side_flags": {"path": side_path.relative_to(project).as_posix(), "sha256": sha256(side_path), "dtype": "uint8", "count": count},
                    "normal_convention": "quaternion_local_z", "observed_cosine_min": .1,
                    "render_grazing_cosine": .05, "policy": "strong_unilateral_train_group_support",
                    "status": "experimental_train_supported_sides_pending_paired_native_evaluation",
                    "coordinates": "camera_init", "preflight": preflight, "cpu_only": True,
                    "side_counts": distribution, "selected_source_ids": source_ids,
                    "unaltered_source_ids": [s["id"] for s in sources if s["id"] not in source_ids],
                    "training_evidence": {"split": "train", "train_only": True, "minimum_unique_groups": 3,
                        "opposite_side_required_count": 0, "mixed_nonzero_evidence": "both even if 1/2 observations",
                        "unknown_policy": "retain two-sided", "hemisphere": "dot(unit_quaternion_local_z,camera_world-mean_world)",
                        "pixel_convention": "prepared NS cx/cy minus .5 -> integer-centered projection; floor(u+.5)",
                        "depth_tolerance_m": "min(0.25, 0.1+0.01*prepared_depth)",
                        "foreground_evidence": "center projection with existing prepared LiDAR depth support and TRAIN image mask; no depth filling",
                        "reference_protocol": "conditioned_sparse_legacy_front_surface; prepared depth predates S1 multi-cluster rejection",
                        "native_transmittance_measured": False, "heldout_used": False,
                        "positive_train_groups": {"path": positive_path.relative_to(project).as_posix(), "sha256": sha256(positive_path), "dtype": "uint16", "count": count},
                        "negative_train_groups": {"path": negative_path.relative_to(project).as_posix(), "sha256": sha256(negative_path), "dtype": "uint16", "count": count},
                        "sources": records, "input_proofs": guard.report()},
                    "caveat": "Sidecar changes inference visibility semantics; a generic double-sided PLY viewer does not apply it. "
                        "Sparse center-depth evidence is not exact rasterized contribution or independently surveyed geometry. "
                        "Unknown support is preserved; no scene completeness/high-fidelity claim."}
        write_json(destination/"metadata.json", metadata)
        return metadata
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-ids", nargs="+", help="Default all sources; unspecified sources remain unknown")
    args = parser.parse_args(argv)
    try:
        report = build(args.assembly, args.output, args.source_ids)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"output": args.output, "side_counts": report["side_counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

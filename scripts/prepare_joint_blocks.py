#!/usr/bin/env python3
"""Prepare at most three real neighboring blocks in their existing metric world.

No colorization, pose correction, training or rendering. Joint depth teachers
are regenerated from merged measured surfels; existing datasets are read-only.
"""
import argparse
import copy
from dataclasses import asdict, fields
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import cv2
from PIL import Image

from package_garage import _state, read_json
from orbvi_gaussian.io import new_output_dir, output_path, write_json
from orbvi_gaussian.points import read_point_cloud
from orbvi_gaussian.geometry import validate_transform
from orbvi_gaussian.dataset import _write_ply
from orbvi_gaussian.surfels import SurfelDepthPolicy, project_surfel_depth

SOURCE_ROOT = ROOT / "data/garage_surface_v1"
MAX_BLOCKS = 3
MAX_FRAMES = 1600
MAX_IMAGE_PIXELS = 16_000_000
MAX_ASSET_BYTES = 128 * 1024 * 1024
MAX_COPY_BYTES = 8 * 1024 ** 3
MAX_REFERENCE_BYTES = 128 * 1024 * 1024
MAX_REFERENCE_POINTS = 1_500_000
MAX_INITIALIZER_INPUT_POINTS = 200_000  # Per source; no implicit reader sampling.
MAX_INITIALIZER_POINTS = 200_000
VOXEL_M = 0.05
SPLITS = ("train_filenames", "val_filenames", "test_filenames")
ASSETS = ("file_path", "mask_path", "lidar_depth_path", "lidar_depth_mask_path")
COMMON_PREPARATION = (
    "calibration_verified", "candidate_experiment_authorized", "independent_test_groups",
    "global_source_split", "global_split_protocol", "split_reference_images_sha256",
    "source_extraction", "calibration", "calibration_sha256", "calibration_id",
    "pose_csv", "pose_sha256", "pose_source", "world_frame", "body_frame", "units")
COMMON_METADATA = ("camera_model", "orientation_override", "pixel_center_convention",
                   "point_cloud_has_rgb", "point_cloud_has_observations", "holdout_every")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_depth_approval(path, hashes):
    """Use prepare_resolution_pilot's explicit approved_policy/S0/v3 contract.

    Its validation is inline in generate_depth, which must not be invoked on
    joint inputs. Here the same contract is checked before writes, additionally
    verifying the frozen evidence bytes. An explicit hash-bound archive may
    relocate historical evidence; never guess a replacement report by name.
    """
    requested = output_path(path)
    requested_sha = snapshot(requested, hashes)
    gate = read_json(requested)
    approval, relocations = requested, {}
    if "approval_file" in gate:
        archive = gate
        approval = output_path(ROOT / archive["approval_file"])
        if snapshot(approval, hashes) != archive["approval_sha256"]:
            raise ValueError("Evidence archive belongs to a different approval SHA256")
        for row in archive["historical_evidence"]:
            name = row["source_at_approval"]
            if name in relocations:
                raise ValueError("Duplicate historical evidence relocation")
            relocations[name] = row
        gate = read_json(approval)
    approved = gate.get("approved_policy")
    # Requiring EVERY dataclass field avoids silently inheriting future defaults.
    if not isinstance(approved, dict) or set(approved) != {f.name for f in fields(SurfelDepthPolicy)}:
        raise ValueError("Require a complete explicit approved_policy; no policy defaults")
    if not str(approved["version"]).startswith("surfel_depth_v3") or type(approved["ground_support"]) is not bool:
        raise ValueError("Require an explicit v3 depth policy")
    policy = SurfelDepthPolicy(**approved)
    s0, v3 = gate.get("s0", {}), gate.get("v3", {})
    if (s0.get("status") != "PASS" or integer(s0.get("passed"), "S0 passed") < 7
            or s0.get("wrong_index_count") != 0 or s0.get("foreground_array_equal_fraction") != 1
            or gate.get("s1", {}).get("status") != "ready"
            or v3.get("status") != "PASS" or v3.get("independent_safety_status") != "PASS"
            or v3.get("unknown_foreground_gate") != "PASS"
            or v3.get("unknown_foreground_DEV_support_pixels") != 0
            or integer(v3.get("dev_frames_checked"), "DEV frames") < 60):
        raise ValueError("Require frozen S0 closure and v3 independent safety/unknown-foreground PASS")
    implementation = {}
    for name in ("surfels.py", "points.py"):
        key = "src/orbvi_gaussian/" + name
        implementation[name] = snapshot(ROOT / key, hashes)
        if gate.get("implementation_sha256", {}).get(key) != implementation[name]:
            raise ValueError("Geometry implementation differs from approved S0/safety evidence")
    evidence = gate.get("evidence", {})
    if not evidence or not set(relocations) <= set(evidence):
        raise ValueError("Missing frozen evidence or archive contains unrelated evidence")
    verified, kinds = [], []
    for name, descriptor in evidence.items():
        relocation = relocations.get(name)
        if relocation and relocation["sha256"] != descriptor["sha256"]:
            raise ValueError("Archive evidence digest disagrees with approval")
        actual = output_path(ROOT / (relocation["immutable_evidence_copy"] if relocation else name))
        if snapshot(actual, hashes) != descriptor["sha256"]:
            raise ValueError("Frozen approval evidence SHA256 differs: " + name)
        proof = read_json(actual)
        code = proof.get("source_sha256", proof.get("implementation_sha256", {}))
        for file, digest in implementation.items():
            if code.get("src/orbvi_gaussian/" + file, code.get(file)) != digest:
                raise ValueError("Frozen evidence refers to a different projection implementation")
        if "wrong_index_count" in proof:
            kind = "s0"
            if (proof.get("status") != "passed" or proof["wrong_index_count"] != 0
                    or proof.get("foreground_array_equal_fraction") != 1
                    or integer(proof.get("tested_index_count"), "S0 tested indices") < 1
                    or proof.get("pytest_returncode") != 0 or not proof.get("commands")):
                raise ValueError("Frozen S0 pixel closure evidence did not pass")
        elif "safety_cases" in proof:
            kind = "geometry_safety"
            cases = proof["safety_cases"]
            if (canonical(proof.get("policy")) != canonical(approved)
                    or proof.get("status") != "passed" or proof.get("overall") != "PASS"
                    or proof.get("pytest_returncode") != 0 or not proof.get("commands")
                    or proof.get("sources_unchanged_during_tests") is not True
                    or not cases or proof.get("passed_cases") != len(cases)
                    or len(cases) != v3.get("geometry_safety_cases")
                    or any(c.get("status") != "PASS" for c in cases)):
                raise ValueError("Frozen geometry safety evidence/policy did not pass")
        elif "candidate_policy" in proof:
            kind = "independent_safety_dev"
            safety, results = proof.get("safety", {}), proof.get("results", [])
            cases = safety.get("cases", [])
            arm = "min%d" % policy.front_cluster_min_points
            if (canonical(proof["candidate_policy"]) != canonical(approved)
                    or proof.get("status") != "DEV_SUPPORT_AND_INDEPENDENT_SAFETY_PASS"
                    or safety.get("status") != "PASS" or safety.get("failed_cases") != []
                    or not cases or safety.get("case_count") != len(cases)
                    or len(cases) != v3.get("independent_safety_cases")
                    or any(c.get("results", {}).get(arm, {}).get("passes_expected") is not True for c in cases)
                    or sum(integer(r.get("frames"), "DEV result frames") for r in results) != v3["dev_frames_checked"]
                    or any(r.get("split") != "DEV" or r.get("potential_unknown_foreground_support", {}).get(arm) != 0
                           for r in results)):
                raise ValueError("Frozen independent safety/DEV evidence or policy did not pass")
        else:
            raise ValueError("Unrecognized frozen depth-policy evidence")
        kinds.append(kind)
        verified.append(dict(kind=kind, source_at_approval=name, path=str(actual), sha256=descriptor["sha256"]))
    if sorted(kinds) != sorted(("s0", "geometry_safety", "independent_safety_dev")):
        raise ValueError("Require exactly one frozen S0, geometry safety and independent safety/DEV report")
    return policy, dict(policy=asdict(policy), policy_sha256=hashlib.sha256(canonical(approved).encode()).hexdigest(),
        approval_path=str(approval), approval_sha256=hashes[approval]["sha256"],
        requested_approval_path=str(requested), requested_approval_sha256=requested_sha,
        implementation_sha256=implementation, frozen_evidence=verified)


def block_ids(values):
    result = []
    for value in values:
        match = re.fullmatch(r"(?:block_)?(\d{1,3})", str(value))
        if match is None or not 0 <= int(match[1]) < 28:
            raise ValueError("Expected garage block IDs 000..027")
        result.append("block_%03d" % int(match[1]))
    if not 1 <= len(result) <= MAX_BLOCKS or len(set(result)) != len(result):
        raise ValueError("Require one to three distinct source blocks")
    return sorted(result)


def asset(root, value):
    rel = PurePosixPath(value) if isinstance(value, str) else None
    if (rel is None or not value or rel.is_absolute() or ".." in rel.parts
            or "\\" in value or "\x00" in value):
        raise ValueError("Source asset must be a safe relative project path")
    path = output_path(root / str(rel))
    if root not in path.parents or not path.is_file():
        raise ValueError("Missing or escaping source asset: %s" % path)
    return path


def snapshot(path, sources):
    """Hash exact bytes in bounded chunks, remembering the source file state."""
    path = output_path(path)
    state = _state(path)
    if path in sources:
        if state != sources[path]["state"]:
            raise ValueError("Source changed during preparation: %s" % path)
        return sources[path]["sha256"]
    if state[2] > MAX_ASSET_BYTES:
        raise ValueError("Source exceeds byte budget: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(raw)
    if _state(path) != state:
        raise ValueError("Source changed while hashing: %s" % path)
    sources[path] = {"state": state, "sha256": digest.hexdigest()}
    return digest.hexdigest()


def integer(value, name):
    if type(value) is not int or not -(2 ** 63) <= value < 2 ** 63:
        raise ValueError("%s must be an integer fitting int64" % name)
    return value


def load_sources(ids, source_root, hashes):
    result = []
    for bid in ids:
        root = output_path(source_root / bid)
        if (root / "INCOMPLETE.json").exists():
            raise ValueError("Source preparation is incomplete: %s" % bid)
        prep_path, meta_path = root / "preparation.json", root / "transforms.json"
        snapshot(prep_path, hashes)
        snapshot(meta_path, hashes)
        prep, meta = read_json(prep_path), read_json(meta_path)
        if (prep.get("status") != "prepared_experimental" or prep.get("block_id") != bid
                or prep.get("calibration_verified") is not False
                or prep.get("candidate_experiment_authorized") is not True
                or prep.get("global_source_split") is not True
                or prep.get("independent_test_groups") is not True
                or prep.get("world_frame") != "camera_init" or prep.get("units") != "metres"):
            raise ValueError("Require actual experimental metric sources with global train/dev/test splits")
        required = set(COMMON_PREPARATION)
        if not required.issubset(prep) or not set(COMMON_METADATA).issubset(meta):
            raise ValueError("Missing source calibration/split/initializer provenance")
        if (meta["camera_model"] != "OPENCV" or meta["orientation_override"] != "none"
                or meta["point_cloud_has_rgb"] is not True
                or meta["point_cloud_has_observations"] is not False
                or prep.get("map", {}).get("color_source") != "calibrated_training_images_only"
                or prep["map"].get("has_measured_rgb") is not True
                or prep["map"].get("alignment_verified_by_operator") is not True
                or prep["map"].get("world_frame") != prep["world_frame"]):
            raise ValueError("Require measured geometry and explicitly training-only measured-color initializer")
        if (not np.array_equal(np.asarray(meta.get("applied_transform", np.eye(4)[:3])), np.eye(4)[:3])
                or meta.get("applied_scale", 1.) != 1.):
            raise ValueError("Require identity world transform and scale 1")
        if not meta.get("frames") or len(meta["frames"]) > MAX_FRAMES:
            raise ValueError("Source frame count exceeds budget or is empty")
        # These are actual preparation inputs; map.source is retained only as
        # historical provenance and is never opened or copied from outside ROOT.
        for key, sha_key in (("calibration", "calibration_sha256"), ("pose_csv", "pose_sha256")):
            if snapshot(output_path(prep[key]), hashes) != prep[sha_key]:
                raise ValueError("Source %s SHA256 disagrees with preparation" % key)
        index_path = output_path(Path(prep["source_extraction"]) / "indexes/images.csv")
        if snapshot(index_path, hashes) != prep["split_reference_images_sha256"]:
            raise ValueError("Global source split reference SHA256 disagrees with preparation")
        cloud = asset(root, meta["ply_file_path"])
        if (meta.get("point_cloud_path") != meta["ply_file_path"]
                or Path(prep["export"]["paths"]["point_cloud"]).resolve() != cloud
                or prep["export"]["point_count"] != prep["map"]["colorized_points"]):
            raise ValueError("Source initializer provenance/counts disagree")
        reference = asset(root, prep["lidar_reference_path"])
        snapshot(cloud, hashes)
        snapshot(reference, hashes)
        if result:
            first = result[0]
            if any(prep[k] != first["preparation"][k] for k in COMMON_PREPARATION):
                raise ValueError("Neighbor world/calibration/global split provenance differs")
            if any(meta[k] != first["metadata"][k] for k in COMMON_METADATA):
                raise ValueError("Neighbor parser/initializer flags differ")
            for key in ("source", "color_source", "input_has_rgb_field", "map_rgb_verified_by_operator"):
                if prep["map"].get(key) != first["preparation"]["map"].get(key):
                    raise ValueError("Neighbor measured-color initializer provenance differs")
        result.append(dict(id=bid, root=root, metadata=meta, preparation=prep,
                           initializer=cloud, reference=reference))
    return result


def validate_source_depth(sources, policy_info, hashes):
    """Reject v3->legacy fallback and source teachers outside the frozen policy."""
    for source in sources:
        prep = source["preparation"]
        teacher = prep.get("teacher_provenance", {})
        if policy_info is None:
            if (teacher.get("policy") is not None or "surfel_depth_v3" in prep["map"].get("depth_protocol", "")
                    or prep.get("resolution_pilot", {}).get("depth_status", "").startswith("ready_surfel_depth_v3")):
                raise ValueError("Policy-bearing sources require explicit --depth-policy-approval; no legacy fallback")
            continue
        if (canonical(teacher.get("policy")) != canonical(policy_info["policy"])
                or teacher.get("s0_report_sha256") != policy_info["approval_sha256"]
                or output_path(ROOT / teacher.get("s0_report", "")) != Path(policy_info["approval_path"])
                or teacher.get("implementation_sha256") != policy_info["implementation_sha256"]
                or teacher.get("reference_sha256") != hashes[source["reference"]]["sha256"]):
            raise ValueError("Source teacher policy/approval/implementation/reference differs: " + source["id"])
        rows = teacher.get("frames", [])
        by_name = {row["file_path"]: row for row in rows}
        frames = source["metadata"]["frames"]
        if len(rows) != len(by_name) or set(by_name) != {f["file_path"] for f in frames}:
            raise ValueError("Incomplete source teacher provenance")
        for frame in frames:
            row = by_name[frame["file_path"]]
            for key, sha_key in (("lidar_depth_path", "depth_sha256"), ("lidar_depth_mask_path", "support_sha256")):
                if snapshot(asset(source["root"], frame[key]), hashes) != row.get(sha_key):
                    raise ValueError("Source teacher bytes differ from approved provenance")
            if "policy" in row.get("diagnostics", {}) and canonical(row["diagnostics"]["policy"]) != canonical(policy_info["policy"]):
                raise ValueError("Source teacher diagnostics used a different policy")


def validate_frame_dimensions(records):
    dimensions, optics = set(), {}
    for record in records:
        frame = record["frame"]
        size = (frame["w"], frame["h"])
        dimensions.add(size)
        key = (frame["camera_id"], frame.get("view_id"))
        intrinsics = tuple(frame[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy"))
        if key in optics and optics[key] != intrinsics:
            raise ValueError("Mixed source resolution/intrinsics for a fixed physical camera/view")
        optics[key] = intrinsics
        for field in ("file_path", "mask_path", "lidar_depth_mask_path"):
            with Image.open(record["assets"][field]) as image:
                if image.size != size:
                    raise ValueError("Source asset resolution differs from frame metadata")
        depth = np.load(record["assets"]["lidar_depth_path"], mmap_mode="r", allow_pickle=False)
        if depth.shape != (size[1], size[0]) or depth.dtype != np.float32:
            raise ValueError("Source depth dimensions/dtype differ from frame metadata")
    if len(dimensions) != 1:
        raise ValueError("Mixed source resolutions are not supported for joint preparation")
    return list(next(iter(dimensions)))


def merge_frames(sources, hashes):
    """Deduplicate camera+group identities only after exact evidence comparison."""
    by_identity, by_filename, group_split, timestamp_split = {}, {}, {}, {}
    for source in sources:
        meta, root = source["metadata"], source["root"]
        memberships = {}
        for split in SPLITS:
            names = meta.get(split)
            if not isinstance(names, list) or not names or len(names) != len(set(names)):
                raise ValueError("Require explicit nonempty, unique source %s" % split)
            for name in names:
                if name in memberships:
                    raise ValueError("Source split leakage: %s" % name)
                memberships[name] = split
        seen = set()
        for original in meta["frames"]:
            frame = copy.deepcopy(original)
            name = frame["file_path"]
            if name in seen or name not in memberships or PurePosixPath(name).parts[0] != "images":
                raise ValueError("Duplicate, unassigned or invalid source image: %s" % name)
            seen.add(name)
            for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "distortion_params"):
                if key in meta:
                    frame[key] = meta[key]
            for key in ("timestamp_ns", "source_header_ns", "source_bag_ns"):
                if key in frame:
                    integer(frame[key], key)
            stamp = integer(frame["timestamp_ns"], "timestamp_ns")
            group = frame["group_id"]
            if type(group) not in (str, int) or str(group) == "":
                raise ValueError("Require actual source group identity")
            if frame["camera_id"] not in ("camera_2", "camera_3", "camera_4", "camera_5"):
                raise ValueError("Require physical source camera_2..camera_5")
            validate_transform(frame["transform_matrix"])  # Already NS/OpenGL; do not convert again.
            intrinsics = np.asarray([frame[k] for k in ("fl_x", "fl_y", "cx", "cy")], float)
            if not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
                raise ValueError("Invalid source intrinsics")
            if any(type(frame[k]) is not int or frame[k] < 1 for k in ("w", "h")):
                raise ValueError("Invalid source image dimensions")
            if frame["w"] * frame["h"] > MAX_IMAGE_PIXELS:
                raise ValueError("Source image exceeds pixel budget")
            if any(np.any(np.asarray(frame.get(k, 0)) != 0) for k in
                   ("k1", "k2", "k3", "k4", "p1", "p2", "distortion_params")):
                raise ValueError("Joint teachers require rectified source pinhole cameras")
            split = memberships[name]
            for table, key in ((group_split, str(group)), (timestamp_split, stamp)):
                if key in table and table[key] != split:
                    raise ValueError("Global camera group/timestamp split leakage")
                table[key] = split
            identity = (frame["camera_id"], str(group))
            if name in by_filename and by_filename[name] != identity:
                raise ValueError("Conflicting duplicate image filename: %s" % name)
            by_filename[name] = identity
            assets = {key: asset(root, frame[key]) for key in ASSETS}
            if frame.get("depth_file_path"):
                assets["depth_file_path"] = asset(root, frame["depth_file_path"])
            digests = {key: snapshot(path, hashes) for key, path in assets.items()}
            teacher = dict(source_block_id=source["id"],
                           depth_path=str(assets["lidar_depth_path"]), depth_sha256=digests["lidar_depth_path"],
                           support_path=str(assets["lidar_depth_mask_path"]), support_sha256=digests["lidar_depth_mask_path"])
            observation = {k: v for k, v in frame.items() if k not in assets}
            if identity in by_identity:
                previous = by_identity[identity]
                if (observation != previous["observation"]
                        or any(digests[k] != previous["digests"][k] for k in ("file_path", "mask_path"))):
                    changed = [key for key in ("file_path", "mask_path") if digests[key] != previous["digests"].get(key)]
                    raise ValueError("Conflicting duplicate %s/%s between %s and %s: %s" % (
                        *identity, previous["source_ids"][0], source["id"], ",".join(changed) or "C2W/K/frame identity"))
                if split != previous["split"]:
                    raise ValueError("Duplicate frame split leakage")
                previous["source_ids"].append(source["id"])
                previous["source_images"].append(str(assets["file_path"]))
                previous["teacher_sources"].append(teacher)
            else:
                by_identity[identity] = dict(frame=frame, assets=assets, digests=digests, observation=observation,
                                             split=split, source_ids=[source["id"]], source_images=[str(assets["file_path"])],
                                             teacher_sources=[teacher])
        if seen != set(memberships):
            raise ValueError("Source split refers to unknown image")
    records = sorted(by_identity.values(), key=lambda r: (r["frame"]["timestamp_ns"], r["frame"]["file_path"]))
    if len(records) > MAX_FRAMES or sum(path.stat().st_size for r in records for path in r["assets"].values()) > MAX_COPY_BYTES:
        raise ValueError("Joint frames/assets exceed bounded copy budget")
    return records


def merge_references(sources, *, preserve_visibility=False):
    arrays, blocks, rows = [], [], []
    total = 0
    for index, source in enumerate(sources):
        with zipfile.ZipFile(source["reference"]) as archive:
            if sum(info.file_size for info in archive.infolist()) > MAX_REFERENCE_BYTES:
                raise ValueError("Expanded reference exceeds byte budget")
        with np.load(source["reference"], allow_pickle=False) as archive:
            xyz, normals, valid = (archive[key] for key in ("xyz", "normals", "normal_valid"))
        count = len(xyz)
        total += count
        if (not count or total > MAX_REFERENCE_POINTS or xyz.shape != (count, 3)
                or normals.shape != xyz.shape or valid.shape != (count,)
                or xyz.dtype.kind != "f" or normals.dtype.kind != "f" or valid.dtype.kind != "b"
                or not np.isfinite(xyz).all() or not np.isfinite(normals).all()):
            raise ValueError("Invalid or oversized measured XYZ/normals/normal_valid reference")
        lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
        if np.any(valid & (lengths == 0)) or np.any((lengths > 0) & ~np.isclose(lengths, 1., rtol=0, atol=1e-4)):
            raise ValueError("Measured normals must be unit; zero is allowed only for invalid normals")
        arrays.append((xyz, normals, valid))
        blocks.append(np.full(count, index, np.uint8))
        rows.append(np.arange(count, dtype=np.uint32))
    xyz, normals, valid = (np.concatenate([a[i] for a in arrays]) for i in range(3))
    blocks, rows = np.concatenate(blocks), np.concatenate(rows)
    if preserve_visibility:
        # A 5cm voxel's trusted record must not replace a nearer unknown-normal
        # occluder. Keep all measured records, including exact overlap, for v3.
        return dict(xyz=xyz, normals=normals, normal_valid=valid,
                    source_block_index=blocks, source_row_index=rows)
    cell_values = np.floor(xyz.astype(np.float64) / VOXEL_M)
    if not np.isfinite(cell_values).all() or np.any(np.abs(cell_values) >= 2 ** 63):
        raise ValueError("Measured coordinates exceed voxel integer range")
    cells = cell_values.astype(np.int64)
    order = np.lexsort((rows, blocks, ~valid, cells[:, 2], cells[:, 1], cells[:, 0]))
    ordered_cells = cells[order]
    first = np.r_[True, np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1)]
    chosen = order[first]
    return dict(xyz=xyz[chosen], normals=normals[chosen], normal_valid=valid[chosen],
                source_block_index=blocks[chosen], source_row_index=rows[chosen])


def merge_initializer(sources):
    points, colors, blocks, rows = [], [], [], []
    for index, source in enumerate(sources):
        xyz, rgb = read_point_cloud(source["initializer"], max_points=MAX_INITIALIZER_INPUT_POINTS,
                                    max_bytes=MAX_ASSET_BYTES)
        if (rgb is None or len(xyz) != source["preparation"]["export"]["point_count"]
                or not len(xyz) or not np.isfinite(xyz).all()):
            raise ValueError("Initializer must retain every actual measured-color source row")
        points.append(xyz); colors.append(rgb)
        blocks.append(np.full(len(xyz), index, np.uint8))
        rows.append(np.arange(len(xyz), dtype=np.uint32))
    xyz, rgb = np.concatenate(points), np.concatenate(colors)
    origin_blocks, origin_rows = np.concatenate(blocks), np.concatenate(rows)
    # Evenly spaced actual input row IDs, stable under CLI source-order changes.
    count = min(len(xyz), MAX_INITIALIZER_POINTS)
    chosen = np.arange(count, dtype=np.int64) * len(xyz) // count
    return xyz[chosen], rgb[chosen], dict(source_block_index=origin_blocks[chosen], source_row_index=origin_rows[chosen])


def copy_asset(source, target, hashes):
    target = output_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _state(source) != hashes[source]["state"]:
        raise ValueError("Source changed before copy: %s" % source)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    shutil.copystat(source, target)  # copy2 metadata semantics, exclusive creation.
    if snapshot(target, {}) != hashes[source]["sha256"]:
        raise ValueError("Copied bytes differ from source: %s" % source)


def regenerate_teacher(reference, frame, image_mask, *, policy=None):
    """Use the producer's measured-surfel and unknown-foreground visibility rules."""
    shape = (frame["h"], frame["w"])
    if image_mask.shape != shape or not np.all(np.isin(image_mask, (0, 255))):
        raise ValueError("Source RGB mask must be same-size binary 0/255")
    pose_cv = np.asarray(frame["transform_matrix"], float) @ np.diag([1., -1., -1., 1.])
    K = np.array([[frame["fl_x"], 0, frame["cx"] - .5],
                  [0, frame["fl_y"], frame["cy"] - .5], [0, 0, 1.]])
    depth, supported = project_surfel_depth(reference["xyz"], reference["normals"],
        reference["normal_valid"], pose_cv, K, frame["w"], frame["h"], policy=policy)
    supported &= image_mask > 0
    depth[~supported] = 0
    return depth.astype(np.float32, copy=False), supported


def write_teacher(out, record, reference, *, policy=None):
    frame = record["frame"]
    image_mask = cv2.imread(str(record["assets"]["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if image_mask is None:
        raise ValueError("Cannot decode source RGB mask")
    depth, supported = regenerate_teacher(reference, frame, image_mask, policy=policy)
    # Preserve the source filenames, with entirely new joint geometry teachers.
    depth_path, mask_path = out / frame["lidar_depth_path"], out / frame["lidar_depth_mask_path"]
    for path in (depth_path, mask_path):
        output_path(path).parent.mkdir(parents=True, exist_ok=True)
    with depth_path.open("xb") as stream:
        np.save(stream, depth, allow_pickle=False)
    ok, encoded = cv2.imencode(".png", supported.astype(np.uint8) * 255)
    if not ok:
        raise ValueError("Cannot encode regenerated support mask")
    with mask_path.open("xb") as stream:
        stream.write(encoded.tobytes())
    # Optional NS depth_file_path must refer to the regenerated joint teacher.
    if "depth_file_path" in frame:
        frame["depth_file_path"] = frame["lidar_depth_path"]
    return dict(file_path=frame["file_path"], supported_pixels=int(supported.sum()),
                sources=record["teacher_sources"],
                source_teachers_differ=len({(s["depth_sha256"], s["support_sha256"])
                                           for s in record["teacher_sources"]}) > 1,
                depth_sha256=snapshot(depth_path, {}), support_sha256=snapshot(mask_path, {}))


def prepare_joint(ids, output=None, *, source_root=SOURCE_ROOT, depth_policy_approval=None, preflight_only=False):
    """Preflight never creates outputs or merges reference/initializer geometry.

    Approval may be the original resolution approval or an explicit evidence
    archive binding that same approval SHA to frozen copies. Neighborhood/floor
    adjacency remains a caller choice, not a conclusion of this input preflight.
    """
    ids = block_ids(ids)
    source_root = output_path(source_root)
    if not preflight_only:
        if output is None:
            raise ValueError("--output is required unless --preflight-only")
        output = output_path(output)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Refusing to overwrite output: %s" % output)
        for bid in ids:
            root = output_path(source_root / bid)
            if root in output.parents or output in root.parents or output == root:
                raise ValueError("Output must be separate from read-only source datasets")
    hashes = {}
    policy, policy_info = None, None
    if depth_policy_approval is not None:
        policy, policy_info = validate_depth_approval(depth_policy_approval, hashes)
    sources = load_sources(ids, source_root, hashes)
    validate_source_depth(sources, policy_info, hashes)
    records = merge_frames(sources, hashes)  # All shared-frame conflicts fail before any output.
    resolution = validate_frame_dimensions(records)
    if preflight_only:
        for path, info in hashes.items():
            if _state(path) != info["state"]:
                raise ValueError("Source changed during preflight: %s" % path)
        return dict(status="source_preflight_passed", source_root=str(source_root), source_block_ids=ids,
                    frames=len(records), resolution=resolution, depth_policy=policy_info,
                    depth_policy_mode="approved_explicit" if policy_info else "legacy_default",
                    files_verified=len(hashes),
                    source_manifest_sha256=hashlib.sha256(canonical({str(p): v["sha256"] for p, v in hashes.items()}).encode()).hexdigest(),
                    reference_merge_performed=False, initializer_merge_performed=False, outputs_created=False,
                    neighborhood_verified=False, gpu_used=False,
                    scope="Source identity, frozen policy/evidence and asset dimensions only; no geometry merge or depth generation")
    reference = merge_references(sources, preserve_visibility=policy_info is not None)
    xyz, rgb, origins = merge_initializer(sources)
    out = new_output_dir(output)
    write_json(out / "INCOMPLETE.json", {"status": "not_ready; incomplete until successful publication"})
    destinations = {}
    for record in records:
        for key, source in record["assets"].items():
            if key not in ("file_path", "mask_path"):
                continue
            target = out / record["frame"][key]
            if target in destinations:
                if destinations[target] != record["digests"][key]:
                    raise ValueError("Conflicting output asset path")
            else:
                copy_asset(source, target, hashes)
                destinations[target] = record["digests"][key]
    with (out / "lidar_reference.npz").open("xb") as stream:
        np.savez_compressed(stream, **reference)
    with (out / "point_origins.npz").open("xb") as stream:
        np.savez_compressed(stream, **origins)
    with (out / "points3D.ply").open("x", encoding="ascii") as stream:
        _write_ply(stream, xyz, rgb)
    teachers = []
    for index, record in enumerate(records):
        teachers.append(write_teacher(out, record, reference, policy=policy))
        if (index + 1) % 20 == 0:
            print("Regenerated joint teachers: %d/%d" % (index + 1, len(records)), flush=True)
    # NS poses are already exported; calling export_dataset would convert their
    # axes again and recompute splits. Retain metadata and explicit membership.
    metadata = {key: copy.deepcopy(value) for key, value in sources[0]["metadata"].items()
                if key != "frames" and key not in SPLITS and key not in ("w", "h", "fl_x", "fl_y", "cx", "cy",
                   "k1", "k2", "k3", "k4", "p1", "p2", "distortion_params")}
    metadata.update(frames=[r["frame"] for r in records], ply_file_path="points3D.ply", point_cloud_path="points3D.ply",
                    split_group_count=len({str(r["frame"]["group_id"]) for r in records}))
    for split in SPLITS:
        metadata[split] = [r["frame"]["file_path"] for r in records if r["split"] == split]
    write_json(out / "transforms.json", metadata)
    for path, info in hashes.items():
        if _state(path) != info["state"]:
            raise ValueError("Source changed during joint preparation: %s" % path)
    preparation = {key: sources[0]["preparation"][key] for key in COMMON_PREPARATION}
    preparation.update(status="prepared_experimental", calibration_verified=False, block_id=out.name,
        source_block_ids=ids, source_root=str(source_root), resolution=resolution,
        depth_policy_mode="approved_explicit" if policy_info else "legacy_default",
        frames=len(records), groups=metadata["split_group_count"],
        method="real multi-block reassociation of consistent prepared sources",
        note="Experimental joint training input; not a new final map or an accepted reconstruction",
        lidar_reference_path="lidar_reference.npz", initializer_origins_path="point_origins.npz",
        reference_selection=("All measured source records in sorted block order; no thinning of unknown-normal occluders"
                             if policy_info else "0.05m XYZ voxels; normal_valid first, then sorted source block ID and source row; original records only"),
        reference_voxel_m=None if policy_info else VOXEL_M, reference_points=len(reference["xyz"]), initializer_points=len(xyz),
        origin_source_order=ids, initializer_selection="Stable evenly spaced actual training-color source rows, at most 200000",
        depth="Regenerated camera-z metres from merged measured surfels; conditioned on source LIO/map, not independent GT",
        teacher_provenance=dict(regenerated_all_frames=True, helper="orbvi_gaussian.surfels.project_surfel_depth",
            camera_conversion="NS/OpenGL c2w @ diag(1,-1,-1,1); K uses cx-0.5,cy-0.5",
            visibility="Unknown remains zero; grazing/large-correction/unknown foreground occlusion rejected by existing helper",
            reference_path="lidar_reference.npz", reference_sha256=snapshot(out / "lidar_reference.npz", {}), frames=teachers),
        map=dict(world_frame="camera_init", alignment_verified_by_operator=True, has_measured_rgb=True,
                 color_source="calibrated_training_images_only", colorized_points=len(xyz),
                 depth_reference_points=len(reference["xyz"]), sources=[s["preparation"]["map"] for s in sources]),
        export=dict(frame_count=len(records), group_count=metadata["split_group_count"], point_count=len(xyz),
                    train_count=len(metadata[SPLITS[0]]), val_count=len(metadata[SPLITS[1]]), test_count=len(metadata[SPLITS[2]]),
                    paths=dict(transforms=str(out / "transforms.json"), point_cloud=str(out / "points3D.ply"))),
        shared_frames=[dict(file_path=r["frame"]["file_path"], source_block_ids=r["source_ids"],
                            source_image_paths=r["source_images"]) for r in records if len(r["source_ids"]) > 1],
        source_hashes={str(path.relative_to(ROOT)): info["sha256"] for path, info in sorted(hashes.items())},
        copied_asset_hashes={str(path.relative_to(out)): sha for path, sha in sorted(destinations.items())},
        generated_hashes={name: snapshot(out / name, {}) for name in
                          ("transforms.json", "points3D.ply", "point_origins.npz", "lidar_reference.npz")})
    if policy_info is not None:
        preparation["teacher_provenance"].update(policy_info)
        preparation["map"]["depth_protocol"] = policy.version + "; exact frozen approved_policy on merged reference"
    write_json(out / "preparation.json", preparation)  # Readiness record is published last.
    (out / "INCOMPLETE.json").unlink()
    return preparation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-ids", nargs="+", required=True, help="e.g. block_014 block_015 block_016")
    parser.add_argument("--output", type=Path, help="New directory inside this project; required except preflight")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT, help="Prepared block source directory (default old640)")
    parser.add_argument("--depth-policy-approval", type=Path,
                        help="Explicit resolution v3 approval, or its hash-bound frozen evidence archive")
    parser.add_argument("--preflight-only", "--dry-run", action="store_true",
                        help="Validate inputs only; no outputs, geometry merge, teacher generation or training")
    args = parser.parse_args(argv)
    from orbvi_gaussian.training import confined_environment
    os.environ.update(confined_environment())
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"]
    cv2.setNumThreads(4)
    result = prepare_joint(args.block_ids, args.output, source_root=args.source_root,
                           depth_policy_approval=args.depth_policy_approval, preflight_only=args.preflight_only)
    if args.preflight_only:
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    print("Prepared %d real frames, %d measured reference records, %d training-color initializer points" % (
        result["frames"], result["reference_points"], result["initializer_points"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)

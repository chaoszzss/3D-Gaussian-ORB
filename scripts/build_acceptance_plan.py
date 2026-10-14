#!/usr/bin/env python3
"""Freeze actual unique DEV cameras/assets and ego_v2 rev2 eligibility on CPU.

No renderer, Torch, GPU allocation, training or changes to source datasets.
The plan is immutable: reruns must produce the same bytes, or use a new path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOLS = ("raw", "legacy", "ego_v2")
ASSET_FIELDS = ("source_image_path", "source_mask_path", "source_lidar_depth_path",
                "source_lidar_support_path", "mask_path")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def frame_key(frame):
    value = frame.get("group_id")
    require(type(value) is int or (isinstance(value, str) and re.fullmatch(r"\d+", value)),
            "Noninteger group_id")
    require(frame.get("camera_id") in ("camera_2", "camera_3", "camera_4", "camera_5"), "Unknown camera_id")
    return str(int(value)), frame["camera_id"]


def local_file(root, value):
    path = (root / value).resolve()
    require(root == path or root in path.parents, "Asset escapes project: " + str(path))
    require(path.is_file(), "Missing asset: " + str(path))
    return path


def immutable_write(path, content):
    path = Path(path)
    require(not path.is_symlink(), "Symlink output forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, "Frozen output differs; choose a new path: " + str(path))
    else:
        with path.open("xb") as handle:
            handle.write(content)


def write_plan(path, plan):
    immutable_write(path, (json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())


def load_config(path):
    import yaml
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    require(config.get("schema_version") == 1 and config.get("split") == "dev", "Require DEV acceptance config v1")
    require(set(config.get("protocols", {})) == set(PROTOCOLS) and config.get("primary_protocol") == "ego_v2",
            "Require three fixed protocols and ego_v2 primary")
    require(config.get("mask_revision") == "ego_v2_rev2", "Require frozen ego_v2 rev2")
    return config


def validate_plan(plan, verify_assets=True):
    """Evaluator API: verify digest, unique exact frame counts and fixed file bytes."""
    require(plan.get("schema_version") == 1 and plan.get("split") == "dev", "Invalid acceptance plan")
    require(plan.get("plan_id") == object_sha({k: v for k, v in plan.items() if k != "plan_id"}),
            "Acceptance plan digest mismatch")
    frames = plan.get("frames", [])
    require(len(frames) == plan.get("expected_unique_frames") == plan.get("unique_frames") and frames,
            "Missing or extra acceptance frames")
    require(len({frame_key(f) for f in frames}) == len(frames), "Duplicate acceptance frame identity")
    require(set(plan.get("protocols", {})) == set(PROTOCOLS) and plan.get("primary_protocol") == "ego_v2",
            "Wrong acceptance protocols")
    require(plan.get("mask_revision") == "ego_v2_rev2", "Wrong mask revision")
    jobs = plan["jobs"]
    require(len({j["block_id"] for j in jobs}) == len(jobs) == plan["expected_blocks"], "Missing/duplicate block job")
    indexed = {(j["block_id"], i) for j in jobs for i in j["frame_indices"]}
    require(indexed == {(f["block_id"], f["dataset_index"]) for f in frames}, "Job/frame coverage differs")
    if verify_assets:
        for name, digest in plan["inputs"].items():
            require(Path(name).is_file() and sha256_file(name) == digest, "Pinned input changed/missing: " + name)
    return plan


def load_plan(path, verify_assets=True):
    with Path(path).open() as handle:
        return validate_plan(json.load(handle), verify_assets)


def frames_for_block(plan, block_id):
    require(block_id in {j["block_id"] for j in plan["jobs"]}, "Unknown acceptance block")
    return [f for f in plan["frames"] if f["block_id"] == block_id]


def validate_frame_source(frozen, source, dataset_index=None):
    """Compare a transforms.json frame with the frozen original camera identity."""
    require(frame_key(frozen) == frame_key(source) and frozen["file_path"] == source["file_path"], "Source frame identity changed")
    if dataset_index is not None:
        require(frozen["dataset_index"] == dataset_index, "Source DEV ordering changed")
    for key in ("fl_x", "fl_y", "cx", "cy", "w", "h"):
        require(source.get(key) == frozen["camera"][key], "Source camera changed: " + key)
    require(source["transform_matrix"][:3] == frozen["camera"]["camera_to_worlds_ns"], "Source pose changed")


def build_plan(project=ROOT, config_path=None):
    import cv2
    import numpy as np
    from orbvi_gaussian.calibration import load_calibration
    from orbvi_gaussian.workflow import _rectify

    root = Path(project).resolve()
    config_path = local_file(root, config_path or "configs/acceptance.experimental.yaml")
    cfg = load_config(config_path)
    inputs = {}
    def pin(value):
        path = local_file(root, value)
        if str(path) not in inputs:
            inputs[str(path)] = sha256_file(path)
        return str(path)
    pin(config_path)
    revision_path = pin(cfg["revision_config"])
    require(inputs[revision_path] == cfg["revision_sha256"], "Wrong ego revision config digest")
    revision = load_calibration(revision_path, require_verified=False)
    masks, optics, raw_masks = {}, {}, {}
    for cid, camera in revision["cameras"].items():
        raw = pin(camera["source_mask_path"])
        require(inputs[raw] == cfg["raw_mask_sha256"][cid], "Wrong ego rev2 raw mask: " + cid)
        raw_masks[cid] = {"path": raw, "sha256": inputs[raw]}
        blank = np.zeros((camera["height"], camera["width"], 3), np.uint8)
        for view in revision["virtual_views"]:
            _, mask, _, K = _rectify(blank, camera, view)
            masks[cid, view["name"]] = mask > 0
            w, h, fx, fy, cx, cy = K
            optics[cid, view["name"]] = [w, h, fx, fy, cx + .5, cy + .5]
    floor_map = {f"block_{i:03d}": label for label, ids in cfg["floor_labels"].items() for i in ids}
    blocks = cfg.get('block_ids', [f"block_{i:03d}" for i in range(cfg["expected_blocks"])])
    require(isinstance(blocks, list) and len(blocks) == cfg['expected_blocks']
            and all(isinstance(b, str) and b in floor_map for b in blocks)
            and blocks == sorted(set(blocks)), 'Explicit blocks must be unique sorted known block IDs')
    actual_blocks = sorted(p.parent.name for p in (root / cfg["dataset_root"]).glob("block_*/transforms.json"))
    require(actual_blocks == blocks, "Missing or unexpected source blocks")
    frames, seen, duplicate_observations, jobs, other_splits = [], {}, [], [], set()
    mask_pixels = {}
    observation_count = 0
    for bid in blocks:
        base = root / cfg["dataset_root"] / bid
        transforms = pin(base / "transforms.json")
        metadata = json.loads(Path(transforms).read_text())
        prep_path = pin(base / "preparation.json")
        prep = json.loads(Path(prep_path).read_text())
        original_cal_path = pin(prep["calibration"])
        require(inputs[original_cal_path] == prep["calibration_sha256"], "Source calibration changed")
        original_cal = load_calibration(original_cal_path, require_verified=False)
        require(original_cal["virtual_views"] == revision["virtual_views"], "Revision changed virtual views")
        for cid, camera in original_cal["cameras"].items():
            require({k: v for k, v in camera.items() if k != "source_mask_path"} ==
                    {k: v for k, v in revision["cameras"][cid].items() if k != "source_mask_path"},
                    "Revision changed camera optics")
        by_name = {f["file_path"]: f for f in metadata["frames"]}
        require(len(by_name) == len(metadata["frames"]), "Duplicate file_path inside source")
        names = metadata["val_filenames"]
        require(names and len(set(names)) == len(names), "Empty/duplicate DEV filename")
        keys = [frame_key(by_name[n]) for n in names]
        require(len(set(keys)) == len(keys), "Duplicate group/camera inside source block")
        for split in ("train_filenames", "test_filenames"):
            require(len(set(metadata[split])) == len(metadata[split]), "Duplicate source split filename")
            other_splits.update(frame_key(by_name[n]) for n in metadata[split])
        job = {"block_id": bid, "transforms": transforms, "transforms_sha256": inputs[transforms],
               "split_images": len(names), "frame_indices": []}
        for index, name in enumerate(names):
            observation_count += 1
            f, key = by_name[name], keys[index]
            assets = {target: pin(base / f[source]) for target, source in (
                ("source_image_path", "file_path"), ("source_mask_path", "mask_path"),
                ("source_lidar_depth_path", "lidar_depth_path"), ("source_lidar_support_path", "lidar_depth_mask_path"))}
            lens_key = (f["camera_id"], f["view_id"])
            require(np.allclose([f[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")],
                                optics[lens_key], rtol=0, atol=1e-7), "Revision mask rays differ from source camera")
            source_mask = cv2.imread(assets["source_mask_path"], cv2.IMREAD_GRAYSCALE)
            require(source_mask is not None and source_mask.shape == masks[lens_key].shape,
                    "Invalid source mask shape")
            require(np.isin(source_mask, [0, 255]).all(), "Nonbinary source mask")
            valid = (source_mask > 0) & masks[lens_key]
            require(valid.any(), "Ego mask excludes every pixel")
            # Content-derived suffix allows a stricter per-frame legacy mask.
            suffix = hashlib.sha256(valid.tobytes()).hexdigest()[:16]
            mask_path = (root / cfg["mask_output"] / ("_".join(lens_key) + "_" + suffix + ".png")).resolve()
            require(root in mask_path.parents, "Output mask escapes project")
            ok, encoded = cv2.imencode(".png", valid.astype(np.uint8) * 255)
            require(ok, "Mask encoding failed")
            immutable_write(mask_path, encoded.tobytes())
            assets["mask_path"] = pin(mask_path)
            mask_pixels[str(mask_path)] = int(valid.sum())
            camera = {k: f.get(k, metadata.get(k)) for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
            camera.update(physical_camera_id=int(f["camera_id"][-1]), camera_to_worlds_ns=f["transform_matrix"][:3])
            row = {"block_id": bid, "dataset_index": index, "group_id": key[0], "camera_id": key[1],
                   "file_path": name, "floor_id": floor_map[bid], "floor_verified": False,
                   "view_id": f["view_id"], "timestamp_ns": f["timestamp_ns"], "camera": camera,
                   "transforms": transforms, "transforms_sha256": inputs[transforms], **assets,
                   "asset_sha256": {k: inputs[v] for k, v in assets.items()}}
            if key in seen:
                canonical = seen[key]
                require(canonical["camera"] == row["camera"] and canonical["timestamp_ns"] == row["timestamp_ns"]
                        and all(canonical["asset_sha256"][k] == row["asset_sha256"][k]
                                for k in ("source_image_path", "source_mask_path", "mask_path")),
                        "Conflicting duplicate camera/RGB/mask identity")
                duplicate_observations.append({"group_id": key[0], "camera_id": key[1], "block_id": bid,
                                               "dataset_index": index, "canonical_block_id": canonical["block_id"],
                                               "depth_teacher_sha256": row["asset_sha256"]["source_lidar_depth_path"]})
                continue
            seen[key] = row
            frames.append(row)
            job["frame_indices"].append(index)
        jobs.append(job)
    require(not set(seen) & other_splits, "DEV identity leaks into TRAIN/TEST")
    require(len(frames) == cfg["expected_unique_frames"], "Actual DEV count differs from expected; do not truncate or pad")
    plan = {"schema_version": 1, "split": "dev", "expected_blocks": cfg["expected_blocks"],
            "expected_unique_frames": cfg["expected_unique_frames"], "unique_frames": len(frames),
            "source_observations": observation_count, "duplicate_observation_count": len(duplicate_observations),
            "canonical_policy": "First block in ascending block ID; original val_filenames order; never average depth teachers",
            "mask_revision": cfg["mask_revision"], "revision_config": revision_path,
            "revision_sha256": inputs[revision_path], "raw_masks": raw_masks, "mask_valid_pixels": mask_pixels,
            "protocols": cfg["protocols"], "primary_protocol": cfg["primary_protocol"],
            "floor_verified": False, "geometry_reference_independent": False,
            "frames": frames, "jobs": jobs, "duplicate_observations": duplicate_observations,
            "inputs": inputs, "gate_thresholds": cfg["gates"],
            "evaluation_contract": {"metric_domain": "float_renderer", "report_frame_protocol_key": "protocols",
                                    "required_protocols": list(PROTOCOLS), "colour_fit": False,
                                    "missing_measurement": "BLOCKED", "incomplete_frame_set": "FAIL"}}
    plan["plan_id"] = object_sha(plan)
    return validate_plan(plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/acceptance.experimental.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/acceptance_plan_v1.json")
    parser.add_argument("--verify", type=Path, help="Read and verify an existing frozen plan only")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.verify) if args.verify else build_plan(ROOT, args.config)
        if not args.verify:
            write_plan(args.output, plan)
        print(json.dumps({k: plan[k] for k in ("plan_id", "unique_frames", "source_observations", "duplicate_observation_count")}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

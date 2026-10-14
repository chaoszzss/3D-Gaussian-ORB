#!/usr/bin/env python3
"""Rectify resolution pilots from raw JPGs while preserving measured world/splits.

Images are published first with INCOMPLETE.json; S1 depth is a separate, gated
step. Source files are read-only and every copied asset has independent storage.
No map manifest, poses, train colors, or source depth is modified.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from orbvi_gaussian.calibration import load_calibration, virtual_parameters
from orbvi_gaussian.workflow import _rectify

SPLITS = ("train_filenames", "val_filenames", "test_filenames")
BASE_CONFIG = ROOT / "configs/direct_lio_refined.ego_v2.experimental.yaml"
FULL_CONFIG = ROOT / "configs/direct_lio_refined.ego_v3.experimental.yaml"
METHOD = "raw JPG resolution pilot; exact source poses and splits"
SUPPORTED_BLOCKS = ("block_000", "block_001", "block_002", "block_003", "block_004",
                    "block_014", "block_015", "block_016")
MIN_FREE_DISK_BYTES = 150 * 1024 ** 3


def require_disk_reserve(path, upcoming_bytes=0):
    """Keep the shared training volume above the requested free-space floor."""
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free - upcoming_bytes <= MIN_FREE_DISK_BYTES:
        raise OSError("Preparation requires more than 150 GiB free after the next write")


def limits():
    cv2.setNumThreads(2)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    # Hard per-process address-space cap is stricter than a 4 GB RSS budget.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    budget = 3750 * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        budget = min(budget, hard)
    resource.setrlimit(resource.RLIMIT_AS, (budget, hard))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def asset(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Asset must be a relative POSIX path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Asset escapes dataset")
    path = root / name
    if root.resolve() not in path.resolve().parents:
        raise ValueError("Asset escapes dataset")
    if path.is_symlink():
        raise ValueError("Symlink assets are not permitted")
    return path


def independent_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    if os.path.samefile(source, destination) or sha(source) != sha(destination):
        raise ValueError("Copy must have identical bytes and independent storage")


def calibration_for_width(width, config=None):
    if width not in (640, 1280):
        raise ValueError("Only 640x540 control and 1280x1080 pilot are supported")
    base = load_calibration(BASE_CONFIG, require_verified=False)
    path = Path(config or (FULL_CONFIG if width == 1280 else BASE_CONFIG)).resolve()
    selected = load_calibration(path, require_verified=False)
    expected = copy.deepcopy(base)
    for view in expected["virtual_views"]:
        view.update(width=width, height=width * 27 // 32)
    if selected != expected:
        raise ValueError("Resolution config may change only virtual width/height; optics, ego masks and poses must equal ego_v2")
    return path, selected, base


def split_membership(meta):
    memberships, group_splits = {}, {}
    for split in SPLITS:
        for name in meta[split]:
            if name in memberships:
                raise ValueError("Duplicate/split-leaking image")
            memberships[name] = split
    if len(meta["frames"]) != len(memberships):
        raise ValueError("Split/frame counts differ")
    for frame in meta["frames"]:
        split = memberships[frame["file_path"]]
        group = str(frame["group_id"])
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Time group leaks across splits")
        group_splits[group] = split
    return memberships


def validate_source(source, raw_root, base):
    prep = read_json(source / "preparation.json")
    meta = read_json(source / "transforms.json")
    if (source / "INCOMPLETE.json").exists() or prep.get("status") != "prepared_experimental":
        raise ValueError("Source is not prepared")
    if prep.get("world_frame") != "camera_init" or prep.get("units") != "metres":
        raise ValueError("Expected original metric camera_init world")
    if prep.get("map", {}).get("color_source") != "calibrated_training_images_only":
        raise ValueError("Initializer provenance must explicitly use training images only")
    old_config = load_calibration(prep["calibration"], require_verified=False)
    if sha(prep["calibration"]) != prep["calibration_sha256"]:
        raise ValueError("Source calibration hash changed")
    for cid, camera in base["cameras"].items():
        for key in ("width", "height", "model", "intrinsics", "distortion", "T_body_camera", "time_offset_ns"):
            if camera[key] != old_config["cameras"][cid][key]:
                raise ValueError("Source optics/pose calibration differs from ego_v2")
    if old_config["virtual_views"] != base["virtual_views"]:
        raise ValueError("Source virtual orientation differs")
    if not meta.get("pixel_center_convention", "").startswith("half_integer"):
        raise ValueError("Expected explicit half-integer training convention")
    split_membership(meta)
    views = {v["name"]: v for v in base["virtual_views"]}
    for frame in meta["frames"]:
        _, w, h, fx, fy, cx, cy = virtual_parameters(views[frame["view_id"]])
        actual = [frame[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")]
        if not np.allclose(actual, [w, h, fx, fy, cx + .5, cy + .5], atol=1e-9, rtol=0):
            raise ValueError("Source principal-point +0.5 convention does not close")
        raw = asset(raw_root, frame["source_file_path"])
        if not raw.is_file() or raw.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError("Missing original compressed JPG: " + str(raw))
        for key in ("file_path", "mask_path", "lidar_depth_path", "lidar_depth_mask_path"):
            asset(source, frame[key])
    with zipfile.ZipFile(source / "lidar_reference.npz") as archive:
        if sum(x.file_size for x in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("Expanded LiDAR reference exceeds memory budget")
    return prep, meta


def prepare_images(source, destination, raw_root, width=1280, config=None):
    source, destination, raw_root = map(lambda p: Path(p).resolve(), (source, destination, raw_root))
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Use a separate new destination")
    if destination.exists():
        raise FileExistsError("Never overwrite an existing dataset: " + str(destination))
    config_path, selected, base = calibration_for_width(width, config)
    prep, original = validate_source(source, raw_root, base)
    input_hashes = {name: sha(source / name) for name in
                    ("transforms.json", "preparation.json", "points3D.ply", "lidar_reference.npz")}
    selected_hash = sha(config_path)
    require_disk_reserve(destination, len(original["frames"]) * width * (width * 27 // 32) * 10)
    destination.mkdir(parents=True)
    write_json(destination / "INCOMPLETE.json", {"status": "rectifying_images", "method": METHOD})
    meta = copy.deepcopy(original)
    views = {v["name"]: v for v in selected["virtual_views"]}
    memberships = split_membership(original)
    records = []
    masks = {}
    start = time.monotonic()
    for index, frame in enumerate(meta["frames"]):
        require_disk_reserve(destination, width * (width * 27 // 32) * 5)
        camera = selected["cameras"][frame["camera_id"]]
        raw_path = asset(raw_root, frame["source_file_path"])
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None or raw.shape != (1080, 1280, 3):
            raise ValueError("Input must be original 1280x1080 JPG, never enlarged rectified pixels")
        image, mask, _, parameters = _rectify(raw, camera, views[frame["view_id"]])
        w, h, fx, fy, cx, cy = parameters
        frame.update(w=w, h=h, fl_x=fx, fl_y=fy, cx=cx + .5, cy=cy + .5)
        for key, array in (("file_path", image), ("mask_path", mask)):
            path = asset(destination, frame[key]); path.parent.mkdir(exist_ok=True, parents=True)
            if path.exists() or not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise RuntimeError("Failed exclusive image/mask creation: " + str(path))
        masks[frame["camera_id"]] = {"valid_pixels": int((mask > 0).sum()), "pixels": w * h}
        records.append({"file_path": frame["file_path"], "source_file_path": frame["source_file_path"],
                        "raw_sha256": sha(raw_path), "image_sha256": sha(destination / frame["file_path"]),
                        "mask_sha256": sha(destination / frame["mask_path"]), "split": memberships[frame["file_path"]]})
        if (index + 1) % 40 == 0 or index + 1 == len(meta["frames"]):
            print(f"{destination.name}: images {index + 1}/{len(meta['frames'])}", flush=True)
    for name in ("points3D.ply", "lidar_reference.npz", "sparse/0/images.txt", "sparse/0/points3D.txt"):
        independent_copy(source / name, destination / name)
    # Same camera IDs/exact poses; only the virtual PINHOLE intrinsics change.
    lines = []
    for line in (source / "sparse/0/cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            lines.append(line); continue
        values = line.split()
        if values[1] != "PINHOLE" or len(values) != 8:
            raise ValueError("Expected PINHOLE source camera")
        scale = width / int(values[2])
        intrinsics = [float(v) for v in values[4:]]
        revised = [intrinsics[0] * scale, intrinsics[1] * scale,
                   (intrinsics[2] + .5) * scale - .5, (intrinsics[3] + .5) * scale - .5]
        lines.append(f"{values[0]} PINHOLE {width} {width * 27 // 32} " + " ".join(format(v, '.17g') for v in revised))
    (destination / "sparse/0/cameras.txt").write_text("\n".join(lines) + "\n")
    if sha(config_path) != selected_hash or any(sha(source / k) != v for k, v in input_hashes.items()):
        raise ValueError("Source changed during preparation")
    for old, new in zip(original["frames"], meta["frames"]):
        for key in old:
            if key not in ("w", "h", "fl_x", "fl_y", "cx", "cy") and old[key] != new[key]:
                raise AssertionError("Frame identity/pose changed")
    write_json(destination / "transforms.json", meta)
    prep = copy.deepcopy(prep)
    prep.update(status="images_prepared_depth_pending", method=METHOD,
                calibration=str(config_path), calibration_sha256=selected_hash,
                calibration_id=selected["calibration_id"], source_extraction=str(raw_root))
    prep["map"]["depth_protocol"] = "PENDING_S0_AND_S1; no legacy depth copied or resized"
    prep["export"]["paths"] = {"transforms": str(destination / "transforms.json"),
                               "point_cloud": str(destination / "points3D.ply"),
                               "cameras": str(destination / "sparse/0/cameras.txt"),
                               "images": str(destination / "sparse/0/images.txt"),
                               "points3D": str(destination / "sparse/0/points3D.txt")}
    prep["resolution_pilot"] = {
        "source_dataset": str(source), "source_hashes": input_hashes,
        "width": width, "height": width * 27 // 32, "physical_optics_unchanged": True,
        "exact_source_poses_splits_timestamps": True, "world_recentered": False,
        "raw_image_shape": [1080, 1280, 3], "resized_old_images": False,
        "pixel_center": {"opencv_cx": (width - 1) / 2, "opencv_cy": (width * 27 // 32 - 1) / 2,
                         "nerfstudio_cx": width / 2, "nerfstudio_cy": width * 27 / 64, "offset": .5},
        "ego_masks": {cid: {"path": c["source_mask_path"], "sha256": sha(c["source_mask_path"])}
                      for cid, c in selected["cameras"].items()},
        "initializer": {"method": "byte-for-byte independent copy of source train initializer",
                        "source": str(source / "points3D.ply"), "sha256": input_hashes["points3D.ply"],
                        "color_source": "calibrated_training_images_only", "recolorized": False,
                        "caveat": "Retains original source-resolution/train-mask color provenance; not recolored using ego_v2 or high-resolution images",
                        "train_filenames_sha256": hashlib.sha256(json.dumps(original["train_filenames"]).encode()).hexdigest()},
        "mask_coverage": masks, "frames": records, "depth_status": "pending_S0_S1",
        "cpu_threads_max": 2, "gpu_used": False, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "image_elapsed_s": time.monotonic() - start}
    write_json(destination / "preparation.json", prep)
    write_json(destination / "INCOMPLETE.json", {"status": "images_ready_depth_pending", "method": METHOD,
                                               "next": "Run --stage depth with the geometry S0 PASS report"})
    return prep


def copy_prepared_images(source, destination):
    """Copy owned pilot inputs byte-for-byte into a new, depth-pending dataset."""
    source, destination = (Path(p).resolve() for p in (source, destination))
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Use a separate new destination")
    if destination.exists():
        raise FileExistsError("Never overwrite an existing dataset: " + str(destination))
    prep = read_json(source / "preparation.json")
    meta = read_json(source / "transforms.json")
    if (prep.get("method") != METHOD or prep.get("status") != "prepared_experimental"
            or (source / "INCOMPLETE.json").exists()):
        raise ValueError("Copy source must be a complete owned resolution pilot")
    split_membership(meta)
    records = prep["resolution_pilot"]["frames"]
    if len(records) != len(meta["frames"]):
        raise ValueError("Source frame manifest length changed")
    source_prep_sha = sha(source / "preparation.json")
    for frame, record in zip(meta["frames"], records):
        if frame["file_path"] != record["file_path"]:
            raise ValueError("Source image ordering changed")
        for key, digest in (("file_path", "image_sha256"), ("mask_path", "mask_sha256")):
            if sha(asset(source, frame[key])) != record[digest]:
                raise ValueError("Source image/mask differs from its frozen provenance")
    copied_files = [frame[key] for frame in meta["frames"] for key in ("file_path", "mask_path")]
    copied_files += ["transforms.json", "points3D.ply", "lidar_reference.npz",
                     "sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"]
    depth_budget = sum(frame["w"] * frame["h"] * 6 for frame in meta["frames"])
    require_disk_reserve(destination, sum((source / name).stat().st_size for name in copied_files) + depth_budget)
    destination.mkdir(parents=True)
    write_json(destination / "INCOMPLETE.json", {"status": "copying_frozen_images_depth_pending", "method": METHOD})
    for frame in meta["frames"]:
        for key in ("file_path", "mask_path"):
            require_disk_reserve(destination, asset(source, frame[key]).stat().st_size)
            independent_copy(asset(source, frame[key]), asset(destination, frame[key]))
    for name in ("transforms.json", "points3D.ply", "lidar_reference.npz",
                 "sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"):
        require_disk_reserve(destination, (source / name).stat().st_size)
        independent_copy(source / name, destination / name)
    if sha(source / "preparation.json") != source_prep_sha:
        raise ValueError("Source preparation changed during copy")
    prep["status"] = "images_prepared_depth_pending"
    prep.pop("teacher_provenance", None)
    prep["map"]["depth_protocol"] = "PENDING_NEW_POLICY_GATE; no previous depth copied"
    prep["resolution_pilot"].update(depth_status="pending_new_policy", images_copied_from=str(source),
        images_copied_from_preparation_sha256=source_prep_sha,
        image_copy_method="independent byte-for-byte copy; exact transforms/RGB/mask/init/reference; no hardlinks")
    prep["export"]["paths"] = {key: str(destination / Path(value).relative_to(source))
                               for key, value in prep["export"]["paths"].items()}
    write_json(destination / "preparation.json", prep)
    write_json(destination / "INCOMPLETE.json", {"status": "images_ready_depth_pending", "method": METHOD,
                                               "next": "Run --stage depth with a fresh policy validation report"})
    return prep


def generate_depth(destination, gate_path):
    """Generate independent S1 teachers only after documented S0 closure."""
    from orbvi_gaussian.surfels import SurfelDepthPolicy, project_surfel_depth
    destination = Path(destination).resolve()
    gate_path = Path(gate_path).resolve()
    gate = read_json(gate_path)
    s0 = gate.get("s0", {})
    if s0.get("status") != "PASS" or s0.get("passed", 0) < 7:
        raise ValueError("S1 requires geometry S0 PASS with at least seven closure tests")
    if gate.get("s1", {}).get("status") != "ready":
        raise ValueError("Geometry S1 policy is not ready")
    approved = gate.get("approved_policy")
    if approved is not None:
        validation = gate.get("v3", {})
        if (not str(approved.get("version", "")).startswith("surfel_depth_v3")
                or validation.get("status") != "PASS"
                or validation.get("independent_safety_status") != "PASS"
                or validation.get("unknown_foreground_gate") != "PASS"
                or validation.get("dev_frames_checked", 0) < 60):
            raise ValueError("V3 requires independent safety/unknown-foreground PASS and all 60 DEV frames")
    prep = read_json(destination / "preparation.json")
    if prep.get("method") != METHOD or prep.get("status") != "images_prepared_depth_pending":
        raise ValueError("Depth generation requires this script's fresh pending dataset")
    meta = read_json(destination / "transforms.json")
    expected = prep["resolution_pilot"]
    if sha(destination / "lidar_reference.npz") != expected["source_hashes"]["lidar_reference.npz"]:
        raise ValueError("Dense reference is no longer the exact source reference")
    if sha(destination / "points3D.ply") != expected["initializer"]["sha256"]:
        raise ValueError("Train-only initializer changed")
    for frame, record in zip(meta["frames"], expected["frames"]):
        for key, digest_key in (("file_path", "image_sha256"), ("mask_path", "mask_sha256")):
            if sha(asset(destination, frame[key])) != record[digest_key]:
                raise ValueError("Prepared image/mask changed")
        for key in ("lidar_depth_path", "lidar_depth_mask_path"):
            if asset(destination, frame[key]).exists():
                raise FileExistsError("Refuse to overwrite existing depth; use a new pilot dataset")
    policy = SurfelDepthPolicy(**approved) if approved is not None else SurfelDepthPolicy.trusted_ground()
    if policy.version.startswith("surfel_depth_v3") and approved is None:
        raise ValueError("V3 requires an explicit approved_policy with independent safety and DEV validation")
    # Hash both projection sides. A concurrently changed implementation must
    # not be published as one homogeneous depth protocol.
    implementation = {name: sha(ROOT / "src/orbvi_gaussian" / name)
                      for name in ("surfels.py", "points.py")}
    for name, digest in implementation.items():
        attested = gate.get("implementation_sha256", {}).get("src/orbvi_gaussian/" + name)
        if attested != digest:
            raise ValueError("Geometry code differs from the S0/S1 gate attestation")
    reference = destination / "lidar_reference.npz"
    with np.load(reference, allow_pickle=False) as archive:
        xyz, normals, normal_valid = (archive[k] for k in ("xyz", "normals", "normal_valid"))
    if len(xyz) > 1_500_000:
        raise ValueError("Reference exceeds bounded per-block point budget")
    rows = []
    start = time.monotonic()
    for index, frame in enumerate(meta["frames"]):
        require_disk_reserve(destination, frame["w"] * frame["h"] * 6)
        T = np.asarray(frame["transform_matrix"], dtype=np.float64) @ np.diag([1., -1., -1., 1.])
        K = np.array([[frame["fl_x"], 0, frame["cx"] - .5],
                      [0, frame["fl_y"], frame["cy"] - .5], [0, 0, 1.]])
        diagnostic = {}
        depth, valid = project_surfel_depth(xyz, normals, normal_valid, T, K, frame["w"], frame["h"],
                                           policy=policy, diagnostics=diagnostic)
        mask = cv2.imread(str(destination / frame["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if depth.shape != mask.shape or valid.shape != mask.shape:
            raise ValueError("S1 arrays and ego mask dimensions differ")
        valid &= (mask > 0) & np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, 0).astype(np.float32)
        dp, mp = (asset(destination, frame[k]) for k in ("lidar_depth_path", "lidar_depth_mask_path"))
        dp.parent.mkdir(parents=True, exist_ok=True); mp.parent.mkdir(parents=True, exist_ok=True)
        with dp.open("xb") as stream:
            np.save(stream, depth, allow_pickle=False)
        if not cv2.imwrite(str(mp), valid.astype(np.uint8) * 255):
            raise RuntimeError("Could not write S1 support mask")
        rows.append({"file_path": frame["file_path"], "depth_sha256": sha(dp), "support_sha256": sha(mp),
                     "valid_environment_pixels": int((mask > 0).sum()),
                     "supported_environment_pixels": int(valid.sum()), "diagnostics": diagnostic})
        if (index + 1) % 40 == 0 or index + 1 == len(meta["frames"]):
            print(f"{destination.name}: S1 depth {index + 1}/{len(meta['frames'])}", flush=True)
    if any(sha(ROOT / "src/orbvi_gaussian" / name) != digest for name, digest in implementation.items()):
        raise ValueError("Geometry implementation changed during depth generation; dataset remains incomplete")
    prep["status"] = "prepared_experimental"
    prep["map"]["depth_protocol"] = (f"{policy.version} trusted_ground; S0 closed; measured front cluster "
                                     f"min_points={policy.front_cluster_min_points}; ego_v2 masked")
    prep["resolution_pilot"]["depth_status"] = f"ready_{policy.version}_front{policy.front_cluster_min_points}"
    prep["resolution_pilot"]["peak_rss_mb"] = max(prep["resolution_pilot"].get("peak_rss_mb", 0),
                                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    prep["teacher_provenance"] = {"protocol": prep["map"]["depth_protocol"], "policy": asdict(policy),
        "s0_report": str(gate_path), "s0_report_sha256": sha(gate_path), "implementation_sha256": implementation,
        "reference_sha256": sha(reference), "reference_points": len(xyz),
        "train_only_color_initialization": True, "depth_elapsed_s": time.monotonic() - start,
        "frames": rows, "supported_environment_pixels": sum(r["supported_environment_pixels"] for r in rows),
        "valid_environment_pixels": sum(r["valid_environment_pixels"] for r in rows)}
    write_json(destination / "preparation.json", prep)
    (destination / "INCOMPLETE.json").unlink()
    return prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", default=["block_000", "block_015"])
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/garage_surface_v1")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/keyframes_2hz")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/garage_fidelity_v2")
    parser.add_argument("--width", type=int, choices=[640, 1280], default=1280)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--copy-images-from-root", type=Path,
                        help="Independently copy a complete pilot's frozen image inputs; recompute depth separately")
    parser.add_argument("--stage", choices=["images", "depth", "all"], default="images")
    parser.add_argument("--s0-report", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/resolution_pilot_v2.json")
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks) or any(b not in SUPPORTED_BLOCKS for b in args.blocks):
        parser.error("Only distinct pilot blocks 000..004/014..016 are supported")
    if args.stage != "images" and not args.s0_report:
        parser.error("--stage depth/all requires --s0-report")
    if args.width == 640 and args.output_root.resolve() == (ROOT / "data/garage_fidelity_v2").resolve():
        parser.error("Use an explicit separate --output-root for the 640 control")
    limits()
    results = []
    for block in args.blocks:
        destination = args.output_root / block
        if args.stage in ("images", "all"):
            if args.copy_images_from_root:
                prep = copy_prepared_images(args.copy_images_from_root / block, destination)
            else:
                prep = prepare_images(args.source_root / block, destination, args.raw_root, args.width, args.config)
        if args.stage in ("depth", "all"):
            prep = generate_depth(destination, args.s0_report)
        results.append({"block": block, "dataset": str(destination.resolve()), "status": prep["status"],
                        "frames": prep["frames"], "width": prep["resolution_pilot"]["width"],
                        "depth_status": prep["resolution_pilot"]["depth_status"],
                        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, {"method": METHOD, "blocks": results, "cpu_threads_max": 2, "gpu_used": False})
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()

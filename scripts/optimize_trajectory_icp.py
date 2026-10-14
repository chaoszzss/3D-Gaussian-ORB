#!/usr/bin/env python3
"""Bounded CPU-only scan-to-map ICP pilot; never replace the production trajectory.

First run --extract-only with the ROS system Python; then --samples with the
Open3D Python. Output consists of evidence and local correction candidates.
The same-run map cannot establish independent trajectory accuracy.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from orbvi_gaussian.geometry import validate_transform, matrix_to_quat_xyzw
from orbvi_gaussian.io import new_output_dir, write_json
from orbvi_gaussian.workflow import load_trajectory, read_rows


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def block_context(block):
    block = Path(block).resolve()
    preparation = json.loads((block / "preparation.json").read_text())
    transforms = json.loads((block / "transforms.json").read_text())
    frames = transforms["frames"]
    if not frames or preparation.get("units") != "metres":
        raise ValueError("Expected nonempty block in metres")
    return preparation, transforms, min(int(f["source_header_ns"]) for f in frames), max(
        int(f["source_header_ns"]) for f in frames)


def motion_window(trajectory, stamp_ns, half_window_ns=500000000,
                  max_displacement_m=.025, max_rotation_deg=.3):
    """Conservative one-second pose stationarity screen, not a scene-static proof."""
    stamp_ns = int(stamp_ns)
    times = np.linspace(-half_window_ns, half_window_ns, 11).astype(np.int64) + stamp_ns
    poses = [trajectory.at(int(t)) for t in times]
    if any(p is None for p in poses):
        return {"static_pose_candidate": False, "reason": "pose_gap_or_out_of_range"}
    p = np.array(poses)
    displacement = np.linalg.norm(p[:, None, :3, 3] - p[None, :, :3, 3], axis=-1).max()
    rotations = Rotation.from_matrix(p[:, :3, :3])
    angle = max(float(np.rad2deg((rotations[i].inv() * rotations).magnitude()).max())
                for i in range(len(p)))
    return {"static_pose_candidate": bool(displacement <= max_displacement_m and angle <= max_rotation_deg),
            "max_pair_displacement_m": float(displacement), "max_pair_rotation_deg": angle,
            "half_window_ns": int(half_window_ns),
            "thresholds": {"max_displacement_m": max_displacement_m, "max_rotation_deg": max_rotation_deg}}


def extract_samples(block, pose_csv, output, windows=3, window_frames=5, mapping_groups=6):
    if not 1 <= windows <= 5 or not 3 <= window_frames <= 10 or not 0 <= mapping_groups <= 12:
        raise ValueError("Pilot limit: 1..5 windows, 3..10 scans/window, 0..12 mapping groups")
    preparation, transforms, lo, hi = block_context(block)
    root = Path(preparation["source_extraction"])
    manifest = json.loads((root / "manifest.json").read_text())
    trajectory = load_trajectory(pose_csv)
    if preparation["pose_sha256"] != sha256(pose_csv):
        raise ValueError("Block pose provenance does not match the selected CSV")
    # Full lightweight message index; clouds.csv was only sampled at 0.5 Hz.
    with (root / "indexes/messages.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["topic"] == "/orbvi/lidar/custom"
                and lo <= int(r["header_ns"]) <= hi]
    rows.sort(key=lambda r: int(r["header_ns"]))
    if len(rows) < windows * window_frames:
        raise ValueError("Insufficient indexed raw scans in this block")
    windows_out, selected = [], {}
    for index, center in enumerate(np.linspace(window_frames, len(rows) - window_frames - 1, windows).astype(int)):
        members = rows[center - window_frames // 2:center - window_frames // 2 + window_frames]
        stamps = [int(r["header_ns"]) for r in members]
        if np.max(np.diff(stamps)) > 150000000:
            raise ValueError("Non-contiguous scan window")
        windows_out.append({"id": index, "scan_header_ns": stamps, "fit_header_ns": stamps[:-1],
                            "holdout_header_ns": stamps[-1:]})
        selected.update({int(r["header_ns"]): r for r in members})
    train = set(transforms["train_filenames"])
    by_stamp = {}
    for f in transforms["frames"]:
        if f["file_path"] in train:
            by_stamp.setdefault(int(f["source_header_ns"]), {})[f["camera_id"]] = f
    candidates = []
    for stamp, frames in sorted(by_stamp.items()):
        motion = motion_window(trajectory, stamp)
        if len(frames) == 4 and motion["static_pose_candidate"]:
            candidates.append((stamp, frames, motion))
    mapping, last = [], None
    for stamp, frames, motion in candidates:
        if last is not None and stamp - last < 2000000000:
            continue
        row = min(rows, key=lambda r: abs(int(r["header_ns"]) + 50000000 - stamp))
        if abs(int(row["header_ns"]) + 50000000 - stamp) > 60000000:
            continue
        mapping.append({"image_header_ns": stamp, "scan_header_ns": int(row["header_ns"]),
                        "images": {c: f["source_file_path"] for c, f in frames.items()},
                        "source_split": "train", "motion": motion})
        selected[int(row["header_ns"])] = row
        last = stamp
        if len(mapping) >= mapping_groups:
            break
    if not mapping_groups:
        mapping = []
        selected = {int(r["header_ns"]): r for r in rows if any(
            int(r["header_ns"]) in w["scan_header_ns"] for w in windows_out)}
    out = new_output_dir(output)
    # Do not invoke io.open_bag: its global TMPDIR would escape this task's write set.
    import tempfile
    temp = out / "ros_tmp"
    temp.mkdir()
    tempfile.tempdir = str(temp)
    import rosbag
    import genpy
    from orbvi_gaussian.points import decode_livox
    from orbvi_gaussian.io import header_ns
    scans = []
    with rosbag.Bag(manifest["source"], "r") as bag:
        for ns, row in sorted(selected.items()):
            bag_ns = int(row["bag_ns"])
            matches = list(bag.read_messages(topics=["/orbvi/lidar/custom"],
                start_time=genpy.Time(bag_ns // 10**9, bag_ns % 10**9),
                end_time=genpy.Time((bag_ns + 1) // 10**9, (bag_ns + 1) % 10**9)))
            matches = [(msg, rec) for _, msg, rec in matches if header_ns(msg) == ns]
            if len(matches) != 1:
                raise ValueError("Indexed bag seek did not return exactly one matching scan")
            msg, rec = matches[0]
            cloud = decode_livox(msg, max_points=100000)
            cloud.update(header_ns=np.int64(ns), bag_ns=np.int64(rec.to_nsec()))
            name = "livox_%d.npz" % ns
            np.savez_compressed(out / name, **cloud)
            scans.append({"header_ns": ns, "file": name, "sha256": sha256(out / name),
                          "frame_id": cloud["frame_id"], "points": len(cloud["xyz"]),
                          "timebase_minus_header_ns": int(cloud["timebase_ns"]) - ns})
    report = {"status": "bounded_raw_sample", "block": str(Path(block).resolve()), "units": "metres",
              "block_range_header_ns": [lo, hi], "source_bag": manifest["source"],
              "bag_size_bytes": Path(manifest["source"]).stat().st_size,
              "pose_csv": str(Path(pose_csv).resolve()), "pose_sha256": sha256(pose_csv),
              "source_extraction": str(root), "message_index_sha256": sha256(root / "indexes/messages.csv"),
              "windows": windows_out, "scans": scans, "mapping_groups": mapping,
              "static_training_groups_found": len(candidates),
              "time_basis": "device header/timebase nanoseconds; bag_ns used only for indexed seeks",
              "parameters": {"windows": windows, "window_frames": window_frames, "mapping_groups": mapping_groups}}
    write_json(out / "samples.json", report)
    return report


def load_scan(samples, ns):
    root = Path(samples)
    metadata = json.loads((root / "samples.json").read_text())
    entry = next(s for s in metadata["scans"] if int(s["header_ns"]) == int(ns))
    path = root / entry["file"]
    if sha256(path) != entry["sha256"]:
        raise ValueError("Sample checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    if str(result["frame_id"]) != "livox_frame":
        raise ValueError("Expected native livox_frame scan, not a registered map cloud")
    return result


def deskew(scan, trajectory, T_body_lidar, lidar_offset_ns=0, max_points=6000):
    """1 ms deskew; return deterministic bounded valid world points and reflectance."""
    xyz = scan["xyz"]
    ranges = np.linalg.norm(xyz, axis=1)
    ids = np.flatnonzero(np.isfinite(xyz).all(1) & (scan["tag"] == 0) & (ranges > 1) & (ranges < 20))
    if len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points).astype(int)]
    xyz = xyz[ids]
    bins = (scan["offset_ns"][ids].astype(np.int64) + 500000) // 1000000
    timebase = int(scan["timebase_ns"]) + int(lidar_offset_ns)
    result, valid = np.zeros_like(xyz), np.zeros(len(xyz), bool)
    for slot in np.unique(bins):
        chosen = bins == slot
        T = trajectory.at(timebase + int(slot) * 1000000)
        if T is not None:
            T = T @ validate_transform(T_body_lidar)
            result[chosen] = xyz[chosen] @ T[:3, :3].T + T[:3, 3]
            valid[chosen] = True
    return result[valid], scan["reflectivity"][ids][valid], {
        "input_points": len(scan["xyz"]), "sampled_points": len(ids), "deskew_valid_points": int(valid.sum()),
        "deskew_valid_fraction": float(valid.mean()) if len(ids) else 0.0,
        "bin_ns": 1000000, "tag_rule": "tag == 0", "range_m": [1, 20]}


def prove_map_subset(reference, source_pcd, samples=128):
    """Exact bounded-memory provenance probe against this release's binary XYZ/RGB PCD.

    A matching filename/bbox is insufficient. This reads the original map once in
    chunks and tests exact membership of evenly selected NPZ coordinates. It is
    explicitly a subset probe, not a claim that every NPZ point was validated.
    """
    reference = np.asarray(reference)
    chosen = reference[np.linspace(0, len(reference) - 1, min(samples, len(reference))).astype(int)]
    if not np.array_equal(chosen.astype(np.float32).astype(float), chosen):
        return {"passed": False, "reason": "reference coordinates are not exact source float32 values"}
    with Path(source_pcd).open("rb") as f:
        fields = {}
        for _ in range(64):
            line = f.readline(4096).decode("ascii").strip()
            if line and not line.startswith("#"):
                key, *values = line.split()
                fields[key] = values
                if key == "DATA":
                    break
        offset = f.tell()
    if (fields.get("DATA") != ["binary"] or fields.get("FIELDS") != ["x", "y", "z", "rgb"]
            or fields.get("SIZE") != ["4"] * 4 or fields.get("TYPE") != ["F", "F", "F", "U"]
            or fields.get("COUNT") != ["1"] * 4):
        return {"passed": False, "reason": "source PCD layout unsupported by exact provenance probe"}
    count = int(fields["POINTS"][0])
    if count * 16 + offset != Path(source_pcd).stat().st_size:
        raise ValueError("PCD size mismatch")
    data = np.memmap(source_pcd, dtype="<f4", mode="r", offset=offset, shape=(count, 4))
    # Compare binary XYZ triplets; no giant KD tree or decompressed full map.
    keys = np.ascontiguousarray(chosen, dtype="<f4").view("V12").ravel()
    found = np.zeros(len(keys), bool)
    for begin in range(0, count, 262144):
        chunk = np.ascontiguousarray(data[begin:begin+262144, :3]).view("V12").ravel()
        found |= np.isin(keys, chunk)
        if found.all():
            break
    return {"passed": bool(found.all()), "sampled": len(keys), "exact_matches": int(found.sum()),
            "source": str(source_pcd), "source_points": count,
            "note": "exact subset provenance only; same-run reference is not independent ground truth"}


def correction_size(T):
    T = validate_transform(T)
    return {"translation_m": float(np.linalg.norm(T[:3, 3])),
            "vertical_m": float(abs(T[2, 3])),
            "rotation_deg": float(np.rad2deg(Rotation.from_matrix(T[:3, :3]).magnitude()))}


def within_bounds(T, max_translation_m=.15, max_vertical_m=.05, max_rotation_deg=1.):
    size = correction_size(T)
    return (size["translation_m"] <= max_translation_m and size["vertical_m"] <= max_vertical_m
            and size["rotation_deg"] <= max_rotation_deg)


def residuals(points, target, normals, tree, T, max_distance=.3):
    points = points @ T[:3, :3].T + T[:3, 3]
    distances, ids = tree.query(points, k=1, workers=1)
    plane = np.abs(np.sum((points - target[ids]) * normals[ids], axis=1))
    return {"points": len(points), "fitness": float(np.mean(distances <= max_distance)),
            "nn_p50_m": float(np.median(distances)), "nn_p95_m": float(np.percentile(distances, 95)),
            # Penalize unmatched points rather than improving RMSE by losing support.
            "capped_plane_rmse_m": float(np.sqrt(np.mean(np.where(
                distances <= max_distance, np.minimum(plane, max_distance)**2, max_distance**2))))}


def observability(points, target, normals, tree, T, max_distance=.3):
    points = points @ T[:3, :3].T + T[:3, 3]
    d, ids = tree.query(points, workers=1)
    keep = d <= max_distance
    if keep.sum() < 100:
        return {"rank": 0, "condition": None, "correspondences": int(keep.sum())}
    p, n = points[keep], normals[ids[keep]]
    # Normalize rotational columns by a fixed 10 m lever arm for meaningful rank.
    J = np.column_stack([np.cross(p, n) / 10., n])
    s = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(s > s[0] * 1e-5))
    return {"rank": rank, "condition": float(s[0] / s[-1]) if rank == 6 else None,
            "singular_values": s.tolist(), "correspondences": int(keep.sum()), "rotation_scale_m": 10.}


def accept_correction(T, before, after, obs, bounded=True, minimum_improvement=.02):
    reasons = []
    keys = ["fitness", "capped_plane_rmse_m", "nn_p50_m", "nn_p95_m"]
    if not all(np.isfinite(report.get(k, np.nan)) for report in [before, after] for k in keys):
        return False, ["nonfinite_or_missing_residual_metrics"]
    if not bounded or not within_bounds(T):
        reasons.append("trust_region_exceeded")
    if obs["rank"] != 6 or obs["condition"] is None or not np.isfinite(obs["condition"]) or obs["condition"] > 1000:
        reasons.append("unobservable_or_ill_conditioned")
    if before["fitness"] < .5 or after["fitness"] < max(.5, before["fitness"] - .01):
        reasons.append("insufficient_or_lost_correspondence_support")
    if after["capped_plane_rmse_m"] >= before["capped_plane_rmse_m"] * (1 - minimum_improvement):
        reasons.append("holdout_plane_improvement_below_two_percent")
    if after["nn_p50_m"] > before["nn_p50_m"] * 1.01 or after["nn_p95_m"] > before["nn_p95_m"] * 1.01:
        reasons.append("holdout_nearest_neighbor_regression")
    return not reasons, reasons


def bounded_icp(fit, holdout, target, max_iterations=25):
    if not 1 <= max_iterations <= 50:
        raise ValueError("Pilot max_iterations must be 1..50")
    if min(len(fit), len(holdout), len(target)) < 100:
        raise ValueError("At least 100 fit, holdout and target points required")
    import open3d as o3d
    registration = o3d.pipelines.registration
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(target, float)))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.6, max_nn=30))
    normals = np.asarray(cloud.normals)
    tree = cKDTree(target)
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(fit, float)))
    estimator = registration.TransformationEstimationPointToPlane(registration.HuberLoss(k=.1))
    T = np.eye(4)
    bounded, history = True, []
    for _ in range(max_iterations):
        result = registration.registration_icp(source, cloud, .3, T, estimator,
            registration.ICPConvergenceCriteria(relative_fitness=0., relative_rmse=0., max_iteration=1))
        proposal = validate_transform(result.transformation)
        history.append(correction_size(proposal))
        if not within_bounds(proposal):
            bounded = False
            break  # reject the entire proposal; no clipping a bad unconstrained solution
        delta = np.linalg.norm(proposal - T)
        T = proposal
        if delta < 1e-8:
            break
    before = residuals(holdout, target, normals, tree, np.eye(4))
    after = residuals(holdout, target, normals, tree, T)
    obs = observability(fit, target, normals, tree, T)
    accepted, reasons = accept_correction(T, before, after, obs, bounded)
    return {"accepted_local_candidate": accepted, "rejection_reasons": reasons,
            "correction_local": T.tolist(), "size": correction_size(T), "before_holdout": before,
            "after_holdout": after, "observability": obs, "iterations": len(history),
            "iteration_sizes": history, "backend": "Open3D %s classic CPU point-to-plane" % o3d.__version__,
            "verified": False}


def centered_to_world(local, pivot):
    """Rotate around the window anchor, not the remote world origin."""
    result = validate_transform(local)
    pivot = np.asarray(pivot, float)
    result[:3, 3] += pivot - result[:3, :3] @ pivot
    return result


def common_support(points, target_z_half_width_m=2., correspondence_margin_m=.3):
    """Fixed source ROI inside the target crop; identical before/after selection.

    Exclude rays going outside the evaluated floor slab *before* registration.
    Otherwise unmatched distant surfaces dominate the clipped residual and mimic
    pose error. The 0.3 m border keeps correspondence search inside the map crop.
    """
    points = np.asarray(points, float)
    return points[np.abs(points[:, 2]) <= target_z_half_width_m - correspondence_margin_m]


def run_pilot(samples, pose_csv, calibration, reference, output, max_iterations=25):
    samples = Path(samples)
    meta = json.loads((samples / "samples.json").read_text())
    prep, _, _, _ = block_context(meta["block"])
    if meta["pose_sha256"] != sha256(pose_csv):
        raise ValueError("Samples were selected using a different trajectory")
    cal = yaml.safe_load(Path(calibration).read_text())
    if cal["pose"].get("timestamp_basis") != "header_ns" or meta.get("units") != "metres":
        raise ValueError("Pilot requires device header nanoseconds and metric scan coordinates")
    trajectory = load_trajectory(pose_csv)
    for key, expected in [("frame_id", cal["pose"]["world_frame"]), ("child_frame_id", cal["pose"]["body_frame"])]:
        if trajectory.message_frame_labels.get(key) != expected:
            raise ValueError("Calibration/trajectory frame mismatch")
    if prep["world_frame"] != cal["pose"]["world_frame"] or prep["map"]["world_frame"] != prep["world_frame"]:
        raise ValueError("Block/map coordinate mismatch")
    source_map = Path(prep["map"]["source"]).resolve()
    if source_map != Path(cal["provenance"]["capture_map"]).resolve():
        raise ValueError("Calibration and block refer to different maps")
    pose_provenance = json.loads((Path(pose_csv).parent / "provenance.json").read_text())
    if Path(pose_provenance["source_bag"]).resolve() != Path(meta["source_bag"]).resolve():
        raise ValueError("Raw scans and LIO trajectory belong to different captures")
    if Path(pose_provenance["pose_bag"]).resolve().parent != source_map.parent:
        raise ValueError("Pose and map release mismatch")
    with np.load(reference, allow_pickle=False) as data:
        ref = np.asarray(data["xyz"], float)
    if ref.shape[1:] != (3,) or not np.isfinite(ref).all() or len(ref) > 1000000:
        raise ValueError("Expected finite bounded reference map XYZ")
    proof = prove_map_subset(ref, source_map)
    if not proof["passed"]:
        raise ValueError("NPZ map provenance probe failed: %s" % proof)
    out = new_output_dir(output)
    windows = []
    for window in meta["windows"]:
        clouds, scan_stats = {}, []
        for ns in window["scan_header_ns"]:
            scan = load_scan(samples, ns)
            xyz, _, stats = deskew(scan, trajectory, cal["lidar"]["T_body_lidar"], cal["lidar"].get("time_offset_ns", 0))
            if stats["deskew_valid_fraction"] < .99:
                raise ValueError("Insufficient trajectory time coverage for scan deskew")
            clouds[ns] = xyz
            scan_stats.append(dict(stats, header_ns=ns))
        center_ns = int(window["scan_header_ns"][len(window["scan_header_ns"]) // 2]) + 50000000
        pivot = trajectory.at(center_ns)[:3, 3]
        fit = np.concatenate([clouds[t] for t in window["fit_header_ns"]]) - pivot
        holdout = np.concatenate([clouds[t] for t in window["holdout_header_ns"]]) - pivot
        raw_counts = {"fit": len(fit), "holdout": len(holdout)}
        fit, holdout = common_support(fit), common_support(holdout)
        keep = (np.linalg.norm(ref[:, :2] - pivot[:2], axis=1) < 22) & (np.abs(ref[:, 2] - pivot[2]) < 2.)
        target = ref[keep] - pivot  # floor-aware crop; reject correspondences > 0.3 m
        if len(target) > 60000:
            target = target[np.linspace(0, len(target)-1, 60000).astype(int)]
        result = bounded_icp(fit, holdout, target, max_iterations)
        D = centered_to_world(result["correction_local"], pivot)
        result.update(window_id=window["id"], center_header_ns=center_ns, pivot_world_m=pivot.tolist(),
                      scan_header_ns=window["scan_header_ns"], fit_header_ns=window["fit_header_ns"],
                      holdout_header_ns=window["holdout_header_ns"], scan_stats=scan_stats,
                      target_points=len(target), correction_world=D.tolist())
        result["common_support"] = {"source_counts_before_roi": raw_counts,
                                    "source_counts_after_roi": {"fit": len(fit), "holdout": len(holdout)},
                                    "source_z_half_width_m": 1.7, "target_z_half_width_m": 2.,
                                    "selection_frozen_before_registration": True}
        # Only accepted windows get a local CSV. Never interpolate across disjoint
        # windows or export a full CSV with discontinuous piecewise corrections.
        if result["accepted_local_candidate"]:
            low, high = min(window["scan_header_ns"]), max(window["scan_header_ns"]) + 100000000
            selected = [r for r in read_rows(pose_csv) if low <= int(r["header_ns"]) <= high]
            corrected = []
            for row in selected:
                original = trajectory.at(int(row["header_ns"]))
                T = D @ original
                change = T[:3, 3] - original[:3, 3]
                if np.linalg.norm(change) > .15 or abs(change[2]) > .05:
                    result["accepted_local_candidate"] = False
                    result["rejection_reasons"].append("pose_translation_exceeds_bound_away_from_pivot")
                    break
                row.update(dict(zip(["tx", "ty", "tz"], map(str, T[:3, 3]))))
                row.update(dict(zip(["qx", "qy", "qz", "qw"], map(str, matrix_to_quat_xyzw(T[:3, :3])))))
                corrected.append(row)
            if result["accepted_local_candidate"] and corrected:
                name = "window_%02d_candidate.csv" % window["id"]
                with (out / name).open("x", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(corrected[0]))
                    writer.writeheader()
                    writer.writerows(corrected)
                result["local_candidate_csv"] = name
        windows.append(result)
    report = {"status": "local_icp_candidates_unverified", "verified": False, "gpu_used": False,
              "promotion_allowed": False, "block": meta["block"], "windows": windows,
              "map_provenance_probe": proof, "actual_lio_quality": trajectory.quality_report,
              "device_vio_is_separate_source": True, "pose_source_topic": pose_provenance["pose_topic"],
              "same_run_reference": True, "independent_geometry_validation": False,
              "full_trajectory_exported": False, "production_inputs_modified": False,
              "coordinate_contract": "metres; T_world_body @ T_body_lidar; device integer nanoseconds; world camera_init",
              "parameters": {"max_iterations": max_iterations, "max_correspondence_m": .3, "normal_radius_m": .6,
                             "normal_max_neighbors": 30, "robust_kernel": "Huber", "huber_k_m": .1,
                             "max_translation_m": .15, "max_vertical_m": .05, "max_rotation_deg": 1.,
                             "map_z_half_width_m": 2., "source_z_half_width_m": 1.7, "points_per_scan_cap": 6000,
                             "holdout": "last complete scan in each window"},
              "environment": {"python": sys.version, "numpy": np.__version__,
                              "backend": "Open3D classic CPU, deterministic point selection"},
              "inputs_sha256": {str(Path(p).resolve()): sha256(p) for p in [pose_csv, calibration, reference, samples / "samples.json", __file__]},
              "limitations": ["Map contains observations from this trajectory; held-out scans are not held out of map construction.",
                              "Local candidates require continuity/loop validation and paired block015 full-parameter training >= +0.5 dB before promotion.",
                              "No independent survey, absolute calibration or global loop-closure claim."]}
    write_json(out / "quality.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", type=Path, default=ROOT / "data/garage_surface_v1/block_015")
    p.add_argument("--pose-csv", type=Path, default=ROOT / "data/all_lio_candidate/odometry.csv")
    p.add_argument("--calibration", type=Path, default=ROOT / "configs/direct_lio_refined.ego_v2.experimental.yaml")
    p.add_argument("--reference", type=Path, default=ROOT / "data/reference_map_voxel01.npz")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--samples", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--window-frames", type=int, default=5)
    p.add_argument("--mapping-groups", type=int, default=6)
    p.add_argument("--max-iterations", type=int, default=25)
    args = p.parse_args()
    if args.extract_only:
        result = extract_samples(args.block, args.pose_csv, args.output, args.windows, args.window_frames, args.mapping_groups)
        summary = {"scans": len(result["scans"]), "mapping_groups": len(result["mapping_groups"]),
                   "static_training_groups_found": result["static_training_groups_found"]}
    else:
        if not args.samples:
            p.error("--samples required unless --extract-only")
        result = run_pilot(args.samples, args.pose_csv, args.calibration, args.reference, args.output, args.max_iterations)
        summary = {"accepted_local_candidates": sum(w["accepted_local_candidate"] for w in result["windows"]),
                   "windows": len(result["windows"]), "promotion_allowed": False}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

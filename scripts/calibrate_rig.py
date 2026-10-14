#!/usr/bin/env python3
"""Recover the direct LIO/LiDAR/raw-camera chain and test measured projections."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import yaml
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt
from orbvi_gaussian.geometry import triple_sphere_project, validate_transform
from orbvi_gaussian.io import RAW_TOPICS, new_output_dir, output_path, write_json
from orbvi_gaussian.workflow import load_trajectory, read_rows

DEFAULT_FAST = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final')
DEFAULT_RELEASE = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release')
DEFAULT_BOARD = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v6/maps/board-calib-backup/camera_config.yaml')
FAST_RAW_MAPPING = {"camera_2": "cam1", "camera_3": "cam0", "camera_4": "cam2", "camera_5": "cam3"}


def rigid_from_rounded(R, t):
    R = np.asarray(R, float).reshape(3, 3)
    error = float(np.abs(R.T @ R - np.eye(3)).max())
    if error > 3e-6 or np.linalg.det(R) <= 0:
        raise ValueError("Calibration rotation exceeds six-decimal roundoff tolerance")
    u, _, vt = np.linalg.svd(R)
    proper = u @ vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = proper, np.asarray(t, float)
    return validate_transform(T), {"input_orthogonality_error": error,
                                 "max_roundoff_correction": float(np.abs(proper - R).max())}


def build_candidate(variant="qr", camera_mapping=None):
    mapping = camera_mapping or FAST_RAW_MAPPING
    runtime = DEFAULT_RELEASE / "runtime_parameters.yaml"
    parameters = yaml.safe_load(runtime.read_text())
    extrinsic = parameters["extrin_calib"]
    T_body_lidar = np.eye(4)
    T_body_lidar[:3, :3] = np.asarray(extrinsic["extrinsic_R"]).reshape(3, 3)
    T_body_lidar[:3, 3] = extrinsic["extrinsic_T"]
    validate_transform(T_body_lidar)
    qr = yaml.safe_load((DEFAULT_FAST / "qr_params.yaml").read_text())
    board = cv2.FileStorage(str(DEFAULT_BOARD), cv2.FILE_STORAGE_READ)
    sources = [runtime, DEFAULT_FAST / "qr_params.yaml", DEFAULT_BOARD]
    data = {"schema_version": 1, "verified": False,
            "calibration_id": "direct-lio-fast-%s-candidate" % variant,
            "provenance": {"chain": "T_world_camera = T_world_body @ T_body_lidar @ inv(T_camera_lidar)",
                           "body_semantics": "LIO _state emitted by /aft_mapped_to_init; extR/extT used by pointBodyToWorld",
                           "physical_cross_imu_calibration": "inherited LIO assumption; not newly measured",
                           "capture_map": str(DEFAULT_RELEASE / "czy_joint_raw_map.pcd"),
                           "status": "source chain recovered, raw-topic mapping and reprojection pending validation"},
            "pose": {"source": "external_csv", "timestamp_basis": "header_ns", "world_frame": "camera_init",
                     "body_frame": "aft_mapped", "frame_semantics_verified": True},
            "lidar": {"T_body_lidar": T_body_lidar.tolist(), "time_offset_ns": 0},
            "cameras": {}, "virtual_views": [{"name": "center", "width": 640, "height": 540,
                                               "fov_horizontal_deg": 90.0, "yaw_deg": 0.0}]}
    try:
        for camera, topic in RAW_TOPICS.items():
            fast_id = mapping[camera]
            path = DEFAULT_FAST / (fast_id + "_multi_calib_result.txt")
            ext = yaml.safe_load(path.read_text())
            T_camera_lidar, repair = rigid_from_rounded(ext["Rcl"], ext["Pcl"])
            if variant == "qr":
                intr = qr["cameras"][fast_id]
                values = [intr[k] for k in ["fx", "fy", "cx", "cy", "xi", "lamda", "alpha", "b", "c"]]
            elif variant == "board":
                values = board.getNode(fast_id).mat().ravel().tolist()
            else:
                raise ValueError("Unsupported candidate")
            data["cameras"][camera] = {"topic": topic, "source_camera_id": fast_id, "topic_mapping_verified": False,
                "model": "triple_sphere", "width": 1280, "height": 1080,
                "intrinsics": dict(zip(["fx", "fy", "cx", "cy"], values[:4])), "distortion": values[4:],
                "T_body_camera": (T_body_lidar @ np.linalg.inv(T_camera_lidar)).tolist(), "time_offset_ns": 0,
                "T_camera_lidar_source": T_camera_lidar.tolist(), "rotation_roundoff_repair": repair}
            sources.append(path)
    finally:
        board.release()
    data["provenance"]["sources_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return data


def deskew_world(scan, trajectory, T_body_lidar):
    xyz, offsets = scan["xyz"], scan["offset_ns"].astype(np.int64)
    bins = (offsets + 500000) // 1000000
    world = np.empty_like(xyz)
    valid = np.zeros(len(xyz), bool)
    timebase = int(scan["timebase_ns"])
    for slot in np.unique(bins):
        ids = np.flatnonzero(bins == slot)
        T = trajectory.at(timebase + int(slot) * 1000000)
        if T is None:
            continue
        T_world_lidar = T @ T_body_lidar
        world[ids] = xyz[ids] @ T_world_lidar[:3, :3].T + T_world_lidar[:3, 3]
        valid[ids] = True
    return world, valid


def projection(world, image, T_world_camera, camera, reflectivity):
    xyz = (world - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    intr = camera["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    params = dict(zip(["xi", "lambda", "alpha", "b", "c"], camera["distortion"]))
    uv, valid = triple_sphere_project(xyz, K, params)
    h, w = image.shape[:2]
    valid &= (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < w-1) & (uv[:, 1] < h-1)
    distance = np.linalg.norm(xyz, axis=1)
    valid &= (distance > 0.5) & (distance < 30)
    ids = np.flatnonzero(valid)
    if len(ids) < 100:
        raise ValueError("Insufficient points projected into camera")
    u, v = np.floor(uv[ids] + .5).astype(int).T
    # Fisheye visibility uses ray range, including valid rays behind the z=0 plane.
    pixels = v * w + u
    order = np.lexsort((distance[ids], pixels))
    order = order[np.r_[True, pixels[order][1:] != pixels[order][:-1]]]
    ids, u, v = ids[order], u[order], v[order]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    photo = gray[v, u]
    refl = reflectivity[ids].astype(float)
    rho = float(spearmanr(photo, refl)[0])
    if not np.isfinite(rho):
        rho = 0.0
    histogram = np.histogram2d(photo, refl, bins=32, range=[[0, 256], [0, 256]])[0]
    joint = histogram / histogram.sum()
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    ent = lambda p: -float(np.sum(p[p>0] * np.log(p[p>0])))
    hx, hy, hxy = ent(px), ent(py), ent(joint)
    nmi = (hx + hy - hxy) / max(1e-12, min(hx, hy))
    overlay = image.copy()
    colors = cv2.applyColorMap(np.clip(reflectivity[ids] * 1.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    for x, y, c in zip(u, v, colors):
        cv2.circle(overlay, (int(x), int(y)), 1, tuple(int(a) for a in c), -1)
    return {"visible_points": len(ids), "reflectance_luma_spearman": rho, "normalized_mutual_information": nmi,
            "metric_note": "Weak cross-modality alignment score; not pixel correspondence or absolute calibration accuracy"}, overlay


def evaluate(candidate, extracted, pose_csv, output, split="dev"):
    out = new_output_dir(output)
    with (out / "candidate.yaml").open("x") as f:
        yaml.safe_dump(candidate, f, sort_keys=False)
    root = Path(extracted).resolve()
    trajectory = load_trajectory(pose_csv)
    image_rows = read_rows(root / "indexes/images.csv")
    camera_rows = {c: [r for r in image_rows if r["camera_id"] == c] for c in RAW_TOPICS}
    cloud_paths = sorted((root / "auxiliary/clouds").glob("livox*.npz"))
    reports, mosaics = [], []
    # Separate scan windows; never use test scores to select the model.
    chosen = [p for i, p in enumerate(cloud_paths) if i >= 1 and i % 2 == (1 if split == "dev" else 0)]
    T_body_lidar = np.asarray(candidate["lidar"]["T_body_lidar"])
    for cloud in chosen:
        scan = np.load(str(cloud))
        world, valid = deskew_world(scan, trajectory, T_body_lidar)
        valid &= scan["tag"] == 0
        stamp = int(scan["timebase_ns"]) + 50000000
        panels = []
        for c in RAW_TOPICS:
            row = min(camera_rows[c], key=lambda r: abs(int(r["header_ns"]) - stamp))
            T_body = trajectory.at(int(row["header_ns"]))
            if T_body is None:
                continue
            image = cv2.imread(str(root / row["file_path"]))
            T_world_camera = T_body @ np.asarray(candidate["cameras"][c]["T_body_camera"])
            score, overlay = projection(world[valid], image, T_world_camera, candidate["cameras"][c], scan["reflectivity"][valid])
            score.update(camera_id=c, image_header_ns=int(row["header_ns"]), scan_header_ns=int(scan["header_ns"]),
                         source_image=row["file_path"], source_scan=cloud.name)
            reports.append(score)
            label = "%s %s MI %.3f rho %.3f" % (c, candidate["calibration_id"], score["normalized_mutual_information"], score["reflectance_luma_spearman"])
            overlay = cv2.resize(overlay, (640, 540))
            cv2.putText(overlay, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,255,255), 1)
            cv2.imwrite(str(out / ("%s_%d.png" % (c, stamp))), overlay)
            panels.append(overlay)
        if len(panels) == 4:
            mosaic = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            cv2.imwrite(str(out / ("overlay_%d.jpg" % stamp)), mosaic)
            mosaics.append("overlay_%d.jpg" % stamp)
    report = {"status": "projection_diagnostic", "split": split, "frames": reports,
              "mean_nmi": float(np.mean([r["normalized_mutual_information"] for r in reports])),
              "mean_spearman": float(np.mean([r["reflectance_luma_spearman"] for r in reports])),
              "mosaics": mosaics, "pose_csv": str(Path(pose_csv).resolve()),
              "time_assumption": "inherited common device time; per-point 1ms deskew bins; image timestamps unchanged"}
    write_json(out / "scores.json", report)
    return report


def mapping_candidates(candidate):
    """Enumerate all 4! physical-model/topic assignments, keeping raw masks fixed.

    A raw-camera mask describes that image's lens/ego support. It must not follow
    the permuted camera model and accidentally score a different image region.
    """
    cameras = sorted(RAW_TOPICS)
    if set(candidate["cameras"]) != set(cameras):
        raise ValueError("Mapping probe requires exactly four raw cameras")
    source_ids = [candidate["cameras"][c]["source_camera_id"] for c in cameras]
    if len(set(source_ids)) != len(cameras):
        raise ValueError("Candidate camera models must have distinct physical source IDs")
    result = []
    for permutation in itertools.permutations(cameras):
        value = copy.deepcopy(candidate)
        value["verified"] = False
        value["topic_mapping_verified"] = False
        mapping = {}
        for raw, model in zip(cameras, permutation):
            value["cameras"][raw] = copy.deepcopy(candidate["cameras"][model])
            value["cameras"][raw]["topic"] = RAW_TOPICS[raw]
            value["cameras"][raw]["topic_mapping_verified"] = False
            value["cameras"][raw].pop("source_mask_path", None)
            if candidate["cameras"][raw].get("source_mask_path"):
                value["cameras"][raw]["source_mask_path"] = candidate["cameras"][raw]["source_mask_path"]
            mapping[raw] = candidate["cameras"][model]["source_camera_id"]
        result.append({"id": "mapping_%02d" % len(result), "models": dict(zip(cameras, permutation)),
                       "mapping": mapping, "candidate": value})
    return result


def image_stationarity(image, neighbor, mask=None):
    """Texture-track check: pose stationarity alone does not prove static images."""
    if image is None or neighbor is None or image.shape != neighbor.shape:
        return {"passed": False, "reason": "missing_or_different_sized_image"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    other = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=.02, minDistance=10, mask=mask)
    if pts is None or len(pts) < 40:
        return {"passed": False, "reason": "insufficient_texture_tracks"}
    moved, valid, _ = cv2.calcOpticalFlowPyrLK(gray, other, pts, None)
    if moved is None:
        return {"passed": False, "reason": "optical_flow_failed"}
    reverse, good, _ = cv2.calcOpticalFlowPyrLK(other, gray, moved, None)
    if reverse is None:
        return {"passed": False, "reason": "reverse_optical_flow_failed"}
    keep = valid.ravel().astype(bool) & good.ravel().astype(bool)
    keep &= np.linalg.norm((reverse - pts).reshape(-1, 2), axis=1) < 1.
    distance = np.linalg.norm((moved - pts).reshape(-1, 2), axis=1)[keep]
    if len(distance) < 40:
        return {"passed": False, "reason": "insufficient_consistent_tracks", "tracks": len(distance)}
    median, p95 = np.percentile(distance, [50, 95])
    return {"passed": bool(median <= 1.5 and p95 <= 4.), "tracks": len(distance),
            "median_flow_px": float(median), "p95_flow_px": float(p95),
            "thresholds_px": {"median": 1.5, "p95": 4.},
            "note": "static image screen; does not establish absolute calibration or remove every dynamic object"}


def mapping_projection_score(world, reflectivity, image, T_world_camera, camera, mask=None, shift_px=0):
    """Weak cross-modal scores, including a LiDAR reflectance-edge residual.

    Image edges are not surveyed 2D/3D correspondences. Never use these scores
    alone to set topic_mapping_verified or calibration verified.
    """
    from scipy.spatial import cKDTree
    xyz = (world - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    intr = camera["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    if camera["model"] != "triple_sphere":
        raise ValueError("Mapping projection currently requires the declared triple_sphere model")
    uv, valid = triple_sphere_project(xyz, K, dict(zip(["xi", "lambda", "alpha", "b", "c"], camera["distortion"])))
    uv[:, 0] += shift_px
    h, w = image.shape[:2]
    if (w, h) != (int(camera["width"]), int(camera["height"])):
        raise ValueError("Raw image size differs from calibration; do not silently resize")
    valid &= np.isfinite(uv).all(1) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < w-1) & (uv[:, 1] < h-1)
    ranges = np.linalg.norm(xyz, axis=1)
    valid &= (ranges > 1.) & (ranges < 20.)
    ids = np.flatnonzero(valid)
    u, v = np.floor(uv[ids] + .5).astype(int).T
    if mask is not None:
        keep = mask[v, u] > 0
        ids, u, v = ids[keep], u[keep], v[keep]
    pixels = v*w + u
    order = np.lexsort((ranges[ids], pixels))
    if len(order):
        order = order[np.r_[True, pixels[order][1:] != pixels[order][:-1]]]
    ids, u, v = ids[order], u[order], v[order]
    if len(ids) < 100:
        return {"valid": False, "visible_points": len(ids), "reason": "insufficient_visible_support"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    photo, refl = gray[v, u].astype(float), reflectivity[ids].astype(float)
    histogram = np.histogram2d(photo, refl, bins=32, range=[[0, 256], [0, 256]])[0]
    joint = histogram / histogram.sum()
    ent = lambda p: -float(np.sum(p[p > 0]*np.log(p[p > 0])))
    hx, hy = ent(joint.sum(1)), ent(joint.sum(0))
    nmi = (hx + hy - ent(joint)) / max(1e-12, min(hx, hy))
    # Sparse LiDAR edges: large reflectance contrast within a small projected
    # neighborhood and with comparable range, to avoid joining depth layers.
    pix = np.column_stack([u, v])
    distance, neighbors = cKDTree(pix).query(pix, k=min(5, len(pix)), workers=1)
    edge = ((np.abs(refl[:, None] - refl[neighbors]) >= 25) & (distance < 12.)
            & (np.abs(ranges[ids, None] - ranges[ids][neighbors]) < .25)).any(1)
    image_edges = cv2.Canny(gray, 50, 120) > 0
    if mask is not None:
        image_edges &= mask > 0
    edge_dt = distance_transform_edt(~image_edges)
    residual = edge_dt[v[edge], u[edge]]
    return {"valid": True, "visible_points": len(ids), "normalized_mutual_information": float(nmi),
            "reflectance_edge_points": int(edge.sum()),
            "edge_distance_median_px": float(np.median(residual)) if len(residual) else None,
            "edge_distance_p95_px": float(np.percentile(residual, 95)) if len(residual) else None,
            "metric_note": "Unmatched reflectance/image edges and NMI are diagnostic proxies, not true reprojection error."}


def mapping_evidence_gate(scores, static_groups, negative_controls, independent_locations=0):
    """Fail closed: comparative proxies can support a hypothesis, not verify it."""
    reasons = []
    signatures = {tuple(sorted(s["mapping"].items())) for s in scores}
    if len(signatures) < 2:
        reasons.append("fewer_than_two_distinct_mapping_candidates")
    if static_groups < 3:
        reasons.append("fewer_than_three_static_groups")
    if independent_locations < 2:
        reasons.append("fewer_than_two_independent_static_locations")
    if not negative_controls or not all(n.get("valid", False) and np.isfinite(n.get("nmi", np.nan)) for n in negative_controls):
        reasons.append("missing_valid_negative_controls")
    eligible = [s for s in scores if s.get("train_nmi") is not None and s.get("validation_nmi") is not None
                and np.isfinite(s["train_nmi"]) and np.isfinite(s["validation_nmi"])]
    winner = max(eligible, key=lambda s: s["train_nmi"]) if eligible else None
    if winner is None:
        reasons.append("missing_disjoint_train_validation_scores")
    else:
        runners = [s for s in eligible if s["id"] != winner["id"]]
        if not runners or winner["validation_nmi"] <= max(s["validation_nmi"] for s in runners) + .01:
            reasons.append("validation_mapping_margin_below_0_01_nmi")
        controls = [n for n in negative_controls if n.get("valid") and n.get("candidate_id") == winner["id"]]
        if len(controls) < 2 or any(n["nmi"] >= winner["validation_nmi"] - .01 for n in controls):
            reasons.append("validation_negative_control_margin_insufficient")
    comparative_pass = not reasons
    reasons.append("no_independent_2d_3d_correspondence_or_capture_binding_evidence")
    return {"verified": False, "topic_mapping_verified": False,
            "status": "comparatively_supported_unverified" if comparative_pass else "insufficient_evidence",
            "selected_on_train": winner["id"] if winner else None, "reasons": reasons,
            "absolute_pixel_reprojection_error_available": False}


def check_mapping(candidate, samples, pose_csv, output):
    from optimize_trajectory_icp import load_scan, deskew, motion_window, sha256, block_context
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    samples = Path(samples)
    meta = json.loads((samples / "samples.json").read_text())
    if candidate["pose"].get("timestamp_basis") != "header_ns" or meta.get("units") != "metres":
        raise ValueError("Mapping probe requires device header nanoseconds and metric scan coordinates")
    if sha256(pose_csv) != meta["pose_sha256"]:
        raise ValueError("Mapping samples and trajectory provenance differ")
    trajectory = load_trajectory(pose_csv)
    if (trajectory.message_frame_labels.get("frame_id") != candidate["pose"]["world_frame"] or
            trajectory.message_frame_labels.get("child_frame_id") != candidate["pose"]["body_frame"]):
        raise ValueError("Mapping candidate and trajectory frames differ")
    _, transforms, _, _ = block_context(meta["block"])
    if len(meta["mapping_groups"]) > 12:
        raise ValueError("Mapping pilot supports at most 12 stationary groups")
    train_files = set(transforms["train_filenames"])
    train_frames = [f for f in transforms["frames"] if f["file_path"] in train_files]
    root = Path(meta["source_extraction"])
    candidates = mapping_candidates(candidate)
    groups, table, controls = [], [], []
    out = new_output_dir(output)
    for group in meta["mapping_groups"]:
        stamp = int(group["image_header_ns"])
        motion = motion_window(trajectory, stamp)
        body_at_image = trajectory.at(stamp)
        record = dict(group, motion_rechecked=motion, image_checks={},
                      body_position_world_m=body_at_image[:3, 3].tolist() if body_at_image is not None else None)
        if not motion["static_pose_candidate"] or group["source_split"] != "train":
            record["eligible"] = False
            groups.append(record)
            continue
        scan = load_scan(samples, group["scan_header_ns"])
        world, refl, stats = deskew(scan, trajectory, candidate["lidar"]["T_body_lidar"],
                                   candidate["lidar"].get("time_offset_ns", 0))
        midpoint = int(scan["timebase_ns"]) + int(np.median(scan["offset_ns"]))
        record.update(scan_image_midpoint_delta_ns=midpoint-stamp, deskew=stats)
        images, masks = {}, {}
        for camera, path in sorted(group["images"].items()):
            images[camera] = cv2.imread(str(root / path))
            model = candidate["cameras"][camera]
            mask = cv2.imread(model["source_mask_path"], cv2.IMREAD_GRAYSCALE) if model.get("source_mask_path") else None
            if model.get("source_mask_path") and mask is None:
                raise ValueError("Declared image mask missing")
            if images[camera] is None or (mask is not None and mask.shape != images[camera].shape[:2]):
                raise ValueError("Missing raw image or wrong mask shape")
            masks[camera] = mask
            nearby = [f for f in train_frames if f["camera_id"] == camera
                      and 350000000 <= abs(int(f["source_header_ns"]) - stamp) <= 800000000]
            neighbor = min(nearby, key=lambda f: abs(int(f["source_header_ns"])-stamp)) if nearby else None
            check = image_stationarity(images[camera], cv2.imread(str(root / neighbor["source_file_path"])) if neighbor else None, mask)
            check.update(image_path=str(root / path), image_sha256=sha256(root / path),
                         neighbor_path=str(root / neighbor["source_file_path"]) if neighbor else None,
                         neighbor_header_ns=int(neighbor["source_header_ns"]) if neighbor else None)
            record["image_checks"][camera] = check
        record["eligible"] = bool(len(images) == 4 and all(c["passed"] for c in record["image_checks"].values())
                                   and abs(midpoint-stamp) <= 60000000 and stats["deskew_valid_fraction"] >= .99)
        groups.append(record)
        if not record["eligible"]:
            continue
        # Alternate chronological stationary windows (>=2s apart), all from the
        # existing TRAIN split. This is not independent scene holdout; the
        # location-diversity gate below separately prevents that claim.
        split = "train" if sum(g["eligible"] for g in groups) % 2 else "validation"
        record["mapping_split"] = split
        cache, negative = {}, {}
        for camera, image in images.items():
            for model_id, model in candidate["cameras"].items():
                camera_stamp = stamp + int(model.get("time_offset_ns", 0))
                body = trajectory.at(camera_stamp)
                if body is None:
                    raise ValueError("Camera timestamp outside valid trajectory")
                T = body @ validate_transform(model["T_body_camera"])
                cache[camera, model_id] = mapping_projection_score(world, refl, image, T, model, masks[camera])
                if split == "validation":
                    for shift in [-120, 120]:
                        negative[camera, model_id, shift] = mapping_projection_score(world, refl, image, T, model, masks[camera], shift)
        for item in candidates:
            values = [cache[c, m] for c, m in item["models"].items()]
            table.append({"candidate_id": item["id"], "image_header_ns": stamp, "split": split,
                          "valid": all(s["valid"] for s in values),
                          "per_camera": {c: cache[c, m] for c, m in item["models"].items()}})
            if split == "validation":
                for shift in [-120, 120]:
                    vals = [negative[c, m, shift] for c, m in item["models"].items()]
                    controls.append({"candidate_id": item["id"], "shift_px": shift, "image_header_ns": stamp,
                                     "valid": all(v["valid"] for v in vals),
                                     "nmi": float(np.mean([v["normalized_mutual_information"] for v in vals]))
                                     if all(v["valid"] for v in vals) else None})
    scores = []
    for item in candidates:
        score = {"id": item["id"], "mapping": item["mapping"]}
        for split in ["train", "validation"]:
            rows = [r for r in table if r["candidate_id"] == item["id"] and r["split"] == split]
            # Do not cherry-pick visible cameras/frames; missing support invalidates the candidate score.
            values = [c["normalized_mutual_information"] for r in rows if r["valid"] for c in r["per_camera"].values()]
            score[split + "_nmi"] = float(np.mean(values)) if rows and all(r["valid"] for r in rows) else None
            score[split + "_groups"] = len(rows)
        scores.append(score)
    locations = []
    for g in groups:
        if g["eligible"]:
            position = np.asarray(g["body_position_world_m"])
            if all(np.linalg.norm(position - p) >= .5 for p in locations):
                locations.append(position)
    gate = mapping_evidence_gate(scores, sum(g["eligible"] for g in groups), controls, len(locations))
    if gate["selected_on_train"] is not None:
        selected = next(i for i in candidates if i["id"] == gate["selected_on_train"])
        value = selected["candidate"]
        value.setdefault("provenance", {})["mapping_probe"] = {
            "status": gate["status"], "report": str(out / "mapping_quality.json"),
            "selection": "train NMI only; disjoint static timestamps used as diagnostic validation; scene diversity separately gated",
            "absolute_accuracy": "unverified; projection proxy scores do not certify mapping"}
        with (out / "selected_mapping.unverified.yaml").open("x") as f:
            yaml.safe_dump(value, f, sort_keys=False)
    report = {"status": gate["status"], "evidence_gate": gate, "verified": False, "topic_mapping_verified": False,
              "candidate_count": len(candidates), "groups": groups, "candidate_scores": scores,
              "projection_scores": table, "negative_controls": controls, "gpu_used": False,
              "source_static_training_groups": meta["static_training_groups_found"],
              "independent_static_locations": len(locations), "static_location_min_separation_m": .5,
              "environment": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
                              "opencv_threads": 1, "opencv_rng_seed": 0},
              "parameters": {"negative_controls_shift_px": [-120, 120], "min_static_groups": 3,
                             "min_separation_ns": 2000000000, "max_scan_image_delta_ns": 60000000,
                             "nmi_margin": .01, "raw_resolution": [1280, 1080]},
              "inputs_sha256": {str(Path(p).resolve()): sha256(p) for p in [pose_csv, samples / "samples.json", __file__]},
              "candidate_sha256": hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest(),
              "limitations": ["All 24 permutations tested; wrong permutations and +/-120px shifts are diagnostic controls.",
                              "NMI/edge distances are cross-modal proxies, not labeled feature reprojection residuals.",
                              "No capture-time physical channel metadata or surveyed 2D/3D correspondences; verified stays false."]}
    write_json(out / "mapping_quality.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["qr", "board"], default="qr")
    p.add_argument("--mapping", help="JSON dictionary raw camera_id -> FAST camN; diagnostic permutation")
    p.add_argument("--extracted", type=Path, default=ROOT / "data/pilot_20s")
    p.add_argument("--pose-csv", type=Path, default=ROOT / "data/all_lio_candidate/odometry.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--calibration", type=Path, help="Evaluate a source-supported refined candidate without rebuilding it")
    p.add_argument("--check-mapping", action="store_true", help="Bounded all-permutation static-image evidence probe; never certifies calibration")
    p.add_argument("--samples", type=Path, help="Raw sample directory produced by optimize_trajectory_icp.py --extract-only")
    args = p.parse_args()
    candidate = yaml.safe_load(args.calibration.read_text()) if args.calibration else build_candidate(args.variant, json.loads(args.mapping) if args.mapping else None)
    if args.check_mapping:
        if not args.samples:
            p.error("--check-mapping requires --samples")
        result = check_mapping(candidate, args.samples, args.pose_csv, args.output)
        print(json.dumps({"status": result["status"], "gate": result["evidence_gate"],
                          "candidate_count": result["candidate_count"]}, ensure_ascii=False, indent=2))
        return
    result = evaluate(candidate,
                      args.extracted, args.pose_csv, args.output, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

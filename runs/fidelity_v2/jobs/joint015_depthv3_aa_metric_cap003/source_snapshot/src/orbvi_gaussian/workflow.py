"""Compose calibrated images, poses, map initialization and provisional blocks."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .calibration import camera_matrix, load_calibration, virtual_parameters
from .io import RAW_TOPICS, new_output_dir, write_json


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_extraction(path):
    root = Path(path).expanduser().resolve()
    with (root / "manifest.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("status") != "complete" or (root / "INCOMPLETE.json").exists():
        raise ValueError("Extraction is incomplete")
    return root, meta


def load_trajectory(path, max_gap_ns=250000000, reorder_tolerance_ns=1000000, require_plausible=True):
    from .geometry import PoseTrajectory, quat_xyzw_to_matrix
    rows = read_rows(path)
    if not rows:
        raise ValueError("Pose CSV is empty")
    frame_labels = {}
    for key in ["frame_id", "child_frame_id"]:
        if key in rows[0]:
            labels = {r[key] for r in rows}
            if len(labels) != 1 or not next(iter(labels)):
                raise ValueError("Pose CSV changes or omits %s; do not interpolate between frames" % key)
            frame_labels[key] = next(iter(labels))
    stamps, matrices = [], []
    for row in rows:
        ns = int(row["header_ns"])
        T = np.eye(4)
        T[:3, :3] = quat_xyzw_to_matrix([float(row[k]) for k in ["qx", "qy", "qz", "qw"]])
        T[:3, 3] = [float(row[k]) for k in ["tx", "ty", "tz"]]
        stamps.append(ns)
        matrices.append(T)
    stamps = np.asarray(stamps, dtype=np.int64)
    matrices = np.asarray(matrices)
    from .trajectory import trajectory_quality
    quality = trajectory_quality(stamps, matrices, reorder_tolerance_ns=reorder_tolerance_ns)
    if require_plausible and quality["status"] == "invalid_for_reconstruction":
        raise ValueError("Trajectory invalid for reconstruction: first bad motion at %.3fs, max step %.3gm. "
                         "Use matching reliable LIO full poses; run trajectory-qa for details." %
                         (quality.get("first_bad_motion_relative_s") or 0.0, quality.get("max_step_m", float("inf"))))
    differences = np.diff(stamps)
    lateness = np.maximum.accumulate(stamps) - stamps
    if require_plausible and (lateness > reorder_tolerance_ns).any():
        raise ValueError("Pose clock moved backwards by more than 1 ms; investigate reset/time domains")
    # Real bag has one 23 us out-of-order VIO pair. Sort observations, never edit timestamps.
    order = np.argsort(stamps, kind="stable")
    stamps, matrices = stamps[order], matrices[order]
    repeated = np.flatnonzero(np.diff(stamps) == 0)
    for i in repeated:
        if not np.allclose(matrices[i], matrices[i + 1], rtol=0, atol=1e-10):
            raise ValueError("Conflicting poses at the exact same header timestamp")
    if len(repeated):
        keep = np.r_[True, np.diff(stamps) != 0]
        stamps, matrices = stamps[keep], matrices[keep]
    result = PoseTrajectory(stamps, matrices, max_gap_ns=max_gap_ns)
    result.quality_report = quality
    result.message_frame_labels = frame_labels
    result.input_time_reordering = {"local_backwards_pairs": int((differences < 0).sum()),
                                   "identical_duplicate_poses_removed": len(repeated),
                                   "reorder_tolerance_ns": reorder_tolerance_ns,
                                   "original_csv_preserved": True}
    return result


def _rectify(image, camera, view):
    from .geometry import make_virtual_view
    T, w, h, fx, fy, cx, cy = virtual_parameters(view)
    kw = {"model": camera["model"]}
    D = camera["distortion"]
    if camera["model"] == "triple_sphere":
        kw["model_parameters"] = dict(zip(["xi", "lambda", "alpha", "b", "c"], D))
        D = None
    if camera.get("source_mask_path"):
        mask = cv2.imread(str(Path(camera["source_mask_path"]).expanduser()), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2] or not np.isin(mask, [0, 255]).all():
            raise ValueError("source_mask_path must refer to a same-size 0/255 lens/device mask")
        kw["source_mask"] = mask
    result, valid = make_virtual_view(image, camera_matrix(camera), D, T, w, h, fx, fy, cx, cy, **kw)
    return result, valid, T, (w, h, fx, fy, cx, cy)


def _input_image(root, row):
    path = (root / row["file_path"]).resolve()
    if root not in path.parents:
        raise ValueError("Image path escapes extraction directory")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode %s" % path)
    return image


def rectify_preview(extracted, calibration, output):
    """Diagnostic intrinsic-only views. They are deliberately not a training dataset."""
    root, meta = load_extraction(extracted)
    cal = load_calibration(calibration, require_verified=False)
    rows = read_rows(root / "indexes/images.csv")
    out = new_output_dir(output)
    result = []
    for camera in RAW_TOPICS:
        row = next((r for r in rows if r["camera_id"] == camera), None)
        if row is None:
            raise ValueError("No image found for %s" % camera)
        image = _input_image(root, row)
        c = cal["cameras"][camera]
        if image.shape[:2] != (c["height"], c["width"]):
            raise ValueError("Calibration image dimensions disagree for %s" % camera)
        for view in cal["virtual_views"]:
            rect, valid, _, _ = _rectify(image, c, view)
            name = camera + "_" + view["name"]
            for suffix, data in [(".png", rect), ("_mask.png", valid)]:
                if not cv2.imwrite(str(out / (name + suffix)), data):
                    raise IOError("Preview image write failed")
            result.append({"camera_id": camera, "view": view["name"], "source": row["file_path"],
                           "header_ns": int(row["header_ns"]), "valid_fraction": float((valid > 0).mean())})
    report = {"status": "diagnostic_only", "calibration_verified": cal["verified"],
              "topic_mapping_verified": all(c["topic_mapping_verified"] for c in cal["cameras"].values()),
              "note": "Intrinsic-model preview; this does not validate extrinsics, camera mapping or capture version",
              "source_bag": meta["source"], "views": result}
    write_json(out / "preview.json", report)
    return report


def plan_blocks(extracted, output, pose_csv=None, length_m=30.0, overlap_m=5.0, max_duration_s=90.0):
    root, manifest = load_extraction(extracted)
    if not (np.isfinite([length_m, overlap_m, max_duration_s]).all()
            and length_m > overlap_m >= 0 and max_duration_s > 0):
        raise ValueError("Require length > overlap >= 0, duration > 0, all finite")
    pose_path = Path(pose_csv).expanduser().resolve() if pose_csv else root / "indexes/odometry.csv"
    trajectory = load_trajectory(pose_path)
    stamps = trajectory.stamps_ns
    positions = trajectory.T_world_body[:, :3, 3]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    blocks = []
    start = 0
    while start < len(stamps):
        length_end = np.searchsorted(distance, distance[start] + length_m, side="right")
        time_end = np.searchsorted(stamps, int(stamps[start]) + int(max_duration_s * 1e9), side="right")
        end = int(min(len(stamps), max(start + 1, min(length_end, time_end))))
        p = positions[start:end]
        blocks.append({"id": "block_%03d" % len(blocks), "start_header_ns": int(stamps[start]),
                       "end_header_ns": int(stamps[end - 1]), "floor_id": None,
                       "arc_length_m": float(distance[end - 1] - distance[start]),
                       "bbox_min_m": p.min(axis=0).tolist(), "bbox_max_m": p.max(axis=0).tolist(),
                       "z_span_m": float(np.ptp(p[:, 2])), "poses": end - start})
        if end == len(stamps):
            break
        # Bounded overlap by arclength. Stationary regions still make temporal progress.
        overlap_start = int(np.searchsorted(distance, distance[end - 1] - overlap_m, side="left"))
        time_overlap_start = int(np.searchsorted(stamps, int(stamps[end - 1])
                                                - int(max_duration_s * overlap_m / length_m * 1e9)))
        overlap_start = max(overlap_start, time_overlap_start)
        start = max(start + 1, min(end - 1, overlap_start))
    result = {"status": "provisional_trajectory_blocks", "pose_csv": str(pose_path),
              "pose_source": "external" if pose_csv else "device_vio_unverified",
              "header_time_basis": True, "anchor_header_ns": manifest["anchor_header_ns"],
              "pose_time_ordering": trajectory.input_time_reordering,
              "pose_quality": trajectory.quality_report,
              "length_m": length_m, "overlap_m": overlap_m, "max_duration_s": max_duration_s,
              "note": "No floor identity inferred from height. Review against globally optimized trajectory, "
                      "floor controls and ramp windows; never flatten z or reset each block origin.",
              "blocks": blocks}
    write_json(output, result)
    return result


def prepare_dataset(extracted, calibration, output, pose_csv=None, sync_ms=10.0,
                    max_gap_ms=250.0, holdout_every=8, max_groups=160,
                    min_sharpness=0.0, block_plan=None, block_id=None,
                    initialization_map=None, map_world_frame=None, map_alignment_verified=False,
                    map_voxel_m=0.1, max_points=100000, save_depth=False,
                    map_rgb_verified=False, colorize_from_images=False,
                    allow_candidate_calibration=False, separate_test=False, global_split=False):
    from .dataset import export_dataset
    from .geometry import camera_world_pose
    root, manifest = load_extraction(extracted)
    cal_path = Path(calibration).expanduser().resolve()
    # Validate calibration and all expensive input contracts before creating output.
    cal = load_calibration(cal_path, require_verified=not allow_candidate_calibration)
    if allow_candidate_calibration:
        from .geometry import validate_transform
        for camera in cal["cameras"].values():
            validate_transform(camera["T_body_camera"])
        if cal["pose"]["source"] != "external_csv" or not cal["pose"]["frame_semantics_verified"]:
            raise ValueError("Candidate experiments still require an explicit, source-supported external body trajectory")
    if (not np.isfinite([sync_ms, max_gap_ms, min_sharpness, map_voxel_m]).all()
            or sync_ms <= 0 or max_gap_ms <= 0 or min_sharpness < 0 or map_voxel_m <= 0
            or max_groups < 0 or max_points <= 0 or holdout_every < 2):
        raise ValueError("Invalid sampling, pose, split or map controls")
    if cal["pose"]["source"] == "external_csv" and pose_csv is None:
        raise ValueError("external_csv pose source requires --pose-csv")
    if cal["pose"]["source"] == "device_vio" and pose_csv is not None:
        raise ValueError("Change calibration.pose.source to external_csv before replacing the trajectory")
    pose_path = Path(pose_csv).expanduser().resolve() if pose_csv else root / "indexes/odometry.csv"
    trajectory = load_trajectory(pose_path, max_gap_ns=int(max_gap_ms * 1e6))
    if trajectory.message_frame_labels.get("frame_id", cal["pose"]["world_frame"]) != cal["pose"]["world_frame"]:
        raise ValueError("Pose CSV world frame differs from the verified calibration world frame")
    begin, end = int(trajectory.stamps_ns[0]), int(trajectory.stamps_ns[-1])
    if block_plan or block_id:
        if not (block_plan and block_id):
            raise ValueError("--block-plan and --block-id must be provided together")
        with Path(block_plan).expanduser().open(encoding="utf-8") as f:
            blocks = json.load(f)["blocks"]
        found = [b for b in blocks if b["id"] == block_id]
        if len(found) != 1:
            raise ValueError("Block ID is missing or duplicated")
        begin = max(begin, found[0]["start_header_ns"])
        end = min(end, found[0]["end_header_ns"])
    if initialization_map:
        if not map_alignment_verified or map_world_frame != cal["pose"]["world_frame"]:
            raise ValueError("Map must be verified in the exact pose world frame; matching frame names alone is insufficient")
    elif save_depth or map_rgb_verified or colorize_from_images:
        raise ValueError("Depth/map color options require an aligned initialization map")
    candidates = {c: [] for c in RAW_TOPICS}
    for row in read_rows(root / "indexes/images.csv"):
        camera = row["camera_id"]
        if camera not in candidates:
            continue
        row["pose_ns"] = int(row["header_ns"]) + cal["cameras"][camera]["time_offset_ns"]
        candidates[camera].append(row)
    for camera in candidates:
        candidates[camera].sort(key=lambda r: r["pose_ns"])
        if not candidates[camera]:
            raise ValueError("Missing extracted images for %s" % camera)
    stamps = {c: np.array([r["pose_ns"] for r in rows], dtype=np.int64) for c, rows in candidates.items()}
    groups, skipped, used_sources = [], Counter(), set()
    for ref in candidates["camera_2"]:
        ns = ref["pose_ns"]
        if not begin <= ns <= end:
            continue
        group = []
        for camera, rows in candidates.items():
            i = int(np.searchsorted(stamps[camera], ns))
            neighbours = [j for j in [i - 1, i] if 0 <= j < len(rows)]
            j = min(neighbours, key=lambda k: abs(int(stamps[camera][k]) - ns))
            row = rows[j]
            if (camera, row["file_path"]) in used_sources:
                skipped["reused_source_frame"] += 1
                break
            if abs(row["pose_ns"] - ns) > sync_ms * 1e6:
                skipped["camera_sync"] += 1
                break
            if float(row["laplacian_variance"]) < min_sharpness:
                skipped["blurred_group"] += 1
                break
            T = trajectory.at(row["pose_ns"])
            if T is None:
                skipped["pose_gap"] += 1
                break
            group.append((camera, row, T))
        if len(group) == len(RAW_TOPICS):
            groups.append((ns, group))
            used_sources.update((c, r["file_path"]) for c, r, _ in group)
            if max_groups and len(groups) >= max_groups:
                break
    if len(groups) < 2:
        raise ValueError("Fewer than two valid camera groups after synchronization/pose checks: %s" % dict(skipped))
    xyz, rgb, map_report, dense_xyz = None, None, None, None
    dense_normals,normal_valid=None,None
    if initialization_map:
        from .points import read_point_cloud, voxel_downsample
        centers = np.array([T[:3, 3] for _, g in groups for _, _, T in g])
        lower = centers.min(axis=0) - [18.0, 18.0, 5.0]
        upper = centers.max(axis=0) + [18.0, 18.0, 5.0]
        # Bounded streaming avoids loading a many-million-point garage into RAM.
        xyz, rgb = read_point_cloud(initialization_map, voxel_size=map_voxel_m,
                                    max_points=min(500000, max(max_points * 4, 50000)), bounds=(lower, upper))
        input_has_rgb = rgb is not None
        if not map_rgb_verified:
            rgb = None  # A visualization RGB field is not evidence of measured texture.
        keep = np.isfinite(xyz).all(axis=1) & ((xyz >= lower) & (xyz <= upper)).all(axis=1)
        xyz, rgb = xyz[keep], None if rgb is None else rgb[keep]
        dense_xyz = xyz.copy()
        xyz, rgb = voxel_downsample(xyz, rgb, map_voxel_m, max_points)
        if not len(xyz):
            raise ValueError("Aligned map crop contains no finite points")
        map_report = {"source": str(Path(initialization_map).expanduser().resolve()),
                      "world_frame": map_world_frame, "alignment_verified_by_operator": True,
                      "points": len(xyz), "has_measured_rgb": rgb is not None,
                      "input_has_rgb_field": input_has_rgb, "map_rgb_verified_by_operator": map_rgb_verified,
                      "crop_min": lower.tolist(), "crop_max": upper.tolist(), "voxel_m": map_voxel_m}
        map_report['depth_reference_points'] = len(dense_xyz)
        map_report['depth_reference_density'] = 'bounded map crop before final initializer cap; no fabricated depth targets'
        if save_depth:
            from .surfels import estimate_normals
            dense_normals,normal_valid=estimate_normals(dense_xyz)
            map_report['depth_protocol']='Measured tangent-plane-supported pixel rays; grazing/large-correction references rejected'
    out = new_output_dir(output)
    for directory in ["images", "masks"] + (["depth", "depth_masks"] if save_depth else []):
        (out / directory).mkdir()
    if save_depth:
        np.savez_compressed(out/'lidar_reference.npz',xyz=dense_xyz.astype('float32'),normals=dense_normals,normal_valid=normal_valid)
    frames, residuals = [], []
    global_group_index = {str(r['pose_ns']): i for i, r in enumerate(candidates['camera_2'])}
    try:
        for ns, group in groups:
            for camera, row, T_world_body in group:
                c = cal["cameras"][camera]
                image = _input_image(root, row)
                if image.shape[:2] != (c["height"], c["width"]):
                    raise ValueError("Calibration resolution mismatch for %s" % camera)
                residuals.append(abs(row["pose_ns"] - ns) / 1e6)
                for view in cal["virtual_views"]:
                    rect, mask, T_camera_virtual, intr = _rectify(image, c, view)
                    if not (mask > 0).any():
                        raise ValueError("Virtual view has no valid pixels: %s/%s" % (camera, view["name"]))
                    T_world_camera = camera_world_pose(T_world_body, c["T_body_camera"], T_camera_virtual)
                    name = "%d_%s_%s" % (ns, camera, view["name"])
                    image_rel, mask_rel = "images/%s.png" % name, "masks/%s.png" % name
                    if not cv2.imwrite(str(out / image_rel), rect) or not cv2.imwrite(str(out / mask_rel), mask):
                        raise IOError("Cannot save rectified images/masks")
                    w, h, fx, fy, cx, cy = intr
                    frame = {"file_path": image_rel, "mask_path": mask_rel,
                             "w": w, "h": h, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
                             "transform_matrix": T_world_camera.tolist(),
                             "timestamp_ns": row["pose_ns"], "group_id": str(ns),
                             "source_header_ns": int(row["header_ns"]), "source_bag_ns": int(row["bag_ns"]),
                             "source_file_path": row["file_path"], "camera_id": camera, "view_id": view["name"]}
                    if save_depth:
                        from .surfels import project_surfel_depth
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        depth, depth_valid = project_surfel_depth(dense_xyz,dense_normals,normal_valid,T_world_camera,K,w,h)
                        depth_valid = (depth_valid > 0) & (mask > 0)
                        depth[~depth_valid] = 0
                        np.save(str(out / ("depth/%s.npy" % name)), depth.astype(np.float32))
                        cv2.imwrite(str(out / ("depth_masks/%s.png" % name)), depth_valid.astype(np.uint8) * 255)
                        frame["lidar_depth_path"] = "depth/%s.npy" % name
                        frame["lidar_depth_mask_path"] = "depth_masks/%s.png" % name
                    frames.append(frame)
        def intended_partition(items):
            if not global_split:
                from .dataset import _split_groups
                train, val, _ = _split_groups(items, holdout_every)
                test = []
                if separate_test:
                    val_set = set(val)
                    held = sorted({f['group_id'] for f in items if f['file_path'] in val_set}, key=int)
                    test_groups = set(held[1::2])
                    test = [f['file_path'] for f in items if f['group_id'] in test_groups]
                    test_set = set(test)
                    val = [p for p in val if p not in test_set]
                return train, val, test
            train, val, test = [], [], []
            for f in items:
                i = global_group_index[f['group_id']]
                # Reserve every 16th source group as test. In the calibration
                # protocol these early-floor groups belong to reserved pairs,
                # while groups 8 mod 16 can have calibration-training support.
                target = train if i % holdout_every else (test if separate_test and not (i // holdout_every) % 2 else val)
                target.append(f['file_path'])
            return train, val, test
        intended_train, intended_val, intended_test = intended_partition(frames)
        if not intended_train or not intended_val or (separate_test and not intended_test):
            raise ValueError('Requested block has no disjoint global train/dev/test camera groups')
        export_xyz, export_rgb = xyz, rgb
        if colorize_from_images:
            from .colors import colorize_from_frames
            train_paths = set(intended_train)
            colorized, supported = colorize_from_frames(xyz, [f for f in frames if f["file_path"] in train_paths], out,
                                                       visibility_xyz=dense_xyz)
            export_xyz, export_rgb = xyz[supported], colorized[supported]
            if not supported.any():
                raise ValueError("No map points have valid training-image color support")
            map_report.update(color_source="calibrated_training_images_only", colorized_points=int(supported.sum()),
                              uncolored_points=int((~supported).sum()), has_measured_rgb=True)
        export = export_dataset(out, frames, points_xyz=export_xyz, points_rgb=export_rgb, holdout_every=holdout_every)
        if separate_test or global_split:
            meta_path = out / "transforms.json"
            meta = json.loads(meta_path.read_text())
            meta['train_filenames'] = intended_train
            meta['val_filenames'] = intended_val
            meta['test_filenames'] = intended_test
            tmp = out / "transforms.json.tmp"
            with tmp.open("x") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False)
            tmp.replace(meta_path)
            export['train_count'], export["val_count"], export["test_count"] = len(intended_train),len(intended_val),len(intended_test)
        if not export["train_count"] or not export["val_count"]:
            raise ValueError("Time grouping produced no disjoint training/validation sets")
        report = {"status": "prepared_experimental" if not cal["verified"] else "prepared", "groups": len(groups), "frames": len(frames),
                  "calibration_verified": cal["verified"], "candidate_experiment_authorized": allow_candidate_calibration,
                  "independent_test_groups": separate_test,
                  "global_source_split": global_split,
                  "global_split_protocol": 'train: index%8!=0; dev: index%16==8; test: index%16==0' if global_split and separate_test else None,
                  "split_reference_images_sha256": hashlib.sha256((root/'indexes/images.csv').read_bytes()).hexdigest(),
                  "source_extraction": str(root), "calibration": str(cal_path),
                  "calibration_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
                  "calibration_id": cal["calibration_id"], "pose_csv": str(pose_path),
                  "pose_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                  "pose_source": cal["pose"]["source"], "world_frame": cal["pose"]["world_frame"],
                  "pose_time_ordering": trajectory.input_time_reordering,
                  "pose_quality": trajectory.quality_report,
                  "body_frame": cal["pose"]["body_frame"], "units": "metres", "block_id": block_id,
                  "sync_max_ms": float(max(residuals)), "skipped_groups": dict(skipped),
                  "map": map_report, "export": export,
                  "depth": "sparse map camera-z metres; requires custom loss, stock Splatfacto does not consume this" if save_depth else None,
                  "lidar_reference_path": 'lidar_reference.npz' if save_depth else None,
                  "note": "Prepared input, not a trained reconstruction. Preserve metric coordinates in the training parser."}
        write_json(out / "preparation.json", report)
        return report
    except Exception:
        write_json(out / "INCOMPLETE.json", {"status": "failed; do not use for training"})
        raise

"""Detect unusable trajectories before trusting finite-looking pose matrices."""

import csv
from pathlib import Path

import numpy as np

from .io import header_ns, new_output_dir, open_bag, stamp_ns, write_json


def trajectory_quality(stamps_ns, matrices, max_speed_m_s=20.0, max_step_m=5.0,
                       max_displacement_m=2000.0, reorder_tolerance_ns=1000000):
    t, T = np.asarray(stamps_ns, np.int64), np.asarray(matrices, float)
    if len(t) == 0 or T.shape != (len(t), 4, 4):
        raise ValueError("Trajectory needs nonempty timestamps and matching SE3 matrices")
    backwards = np.diff(t)
    lateness = np.maximum.accumulate(t) - t
    ordering_ok = not (lateness > reorder_tolerance_ns).any()
    order = np.argsort(t, kind="stable")
    t, p = t[order], T[order, :3, 3]
    finite = np.isfinite(T).all()
    result = {"poses": len(t), "start_header_ns": int(t[0]), "end_header_ns": int(t[-1]),
              "duration_s": float((t[-1] - t[0]) / 1e9), "all_finite": bool(finite),
              "local_nonincreasing_pairs": int((backwards <= 0).sum()),
              "clock_order_within_tolerance": bool(ordering_ok),
              "max_header_lateness_ns": int(lateness.max()),
              "thresholds": {"max_speed_m_s": max_speed_m_s, "max_step_m": max_step_m,
                             "max_displacement_m": max_displacement_m, "reorder_tolerance_ns": reorder_tolerance_ns},
              "note": "Plausibility screening is not pose accuracy or external ground truth. "
                      "Speed checks omit intervals <10ms so timestamp jitter cannot masquerade as large motion."}
    if not finite:
        result.update(status="invalid_for_reconstruction", reason="nonfinite poses")
        return result
    delta = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.diff(t).astype(float) / 1e9
    speed = np.divide(delta, dt, out=np.zeros_like(delta), where=dt >= 0.01)
    displacement = np.linalg.norm(p - p[0], axis=1)
    bad = (delta > max_step_m) | (speed > max_speed_m_s)
    bad_ids = np.flatnonzero(bad)
    passed = ordering_ok and not bad.any() and float(displacement.max()) <= max_displacement_m
    result.update(status="motion_plausible_unverified" if passed else "invalid_for_reconstruction",
                  arc_length_m=float(delta.sum()), max_step_m=float(delta.max()) if len(delta) else 0.0,
                  max_speed_m_s=float(speed.max()) if len(speed) else 0.0,
                  max_displacement_m=float(displacement.max()),
                  bbox_min_m=p.min(axis=0).tolist(), bbox_max_m=p.max(axis=0).tolist(),
                  last_position_m=p[-1].tolist(), bad_motion_intervals=len(bad_ids),
                  first_bad_motion_header_ns=int(t[bad_ids[0] + 1]) if len(bad_ids) else None,
                  first_bad_motion_relative_s=float((t[bad_ids[0] + 1] - t[0]) / 1e9) if len(bad_ids) else None)
    return result


def export_pose_bag(pose_bag, pose_topic, extracted, output):
    """Read a matching native Odometry topic into the external pose CSV contract.

    Selection does not transform poses, establish rig extrinsics, or prove map
    alignment. Source topic, frames and time coverage are retained for review.
    """
    from .workflow import load_extraction, read_rows
    root, manifest = load_extraction(extracted)
    source_rows = read_rows(root / "indexes/odometry.csv")
    begin = min(int(r["header_ns"]) for r in source_rows)
    end = max(int(r["header_ns"]) for r in source_rows)
    rows, world_frames, body_frames = [], set(), set()
    with open_bag(pose_bag) as bag:
        info = bag.get_type_and_topic_info().topics
        if pose_topic not in info or info[pose_topic].msg_type != "nav_msgs/Odometry":
            raise ValueError("Pose topic must exist and have nav_msgs/Odometry type")
        for _, msg, recorded in bag.read_messages(topics=[pose_topic]):
            ns = header_ns(msg)
            if not begin <= ns <= end:
                continue
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            rows.append([ns, stamp_ns(recorded), msg.header.frame_id, msg.child_frame_id,
                         p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            world_frames.add(msg.header.frame_id)
            body_frames.add(msg.child_frame_id)
    if len(rows) < 2:
        raise ValueError("Pose bag has no usable timestamp overlap with this extraction")
    if len(world_frames) != 1 or len(body_frames) != 1:
        raise ValueError("Pose topic changes world/body frames; cannot concatenate without registration")
    out = new_output_dir(output)
    with (out / "odometry.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header_ns", "bag_ns", "frame_id", "child_frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writerows(rows)
    from .workflow import load_trajectory
    trajectory = load_trajectory(out / "odometry.csv")
    coverage = (int(trajectory.stamps_ns[-1]) - int(trajectory.stamps_ns[0])) / max(1, end - begin)
    result = {"status": "pose_csv_exported_frames_and_map_alignment_unverified", "pose_bag": str(Path(pose_bag).resolve()),
              "pose_topic": pose_topic, "source_extraction": str(root), "source_bag": manifest["source"],
              "world_frame_from_message": next(iter(world_frames)), "body_frame_from_message": next(iter(body_frames)),
              "poses": len(rows), "time_span_coverage_fraction": coverage,
              "quality": trajectory.quality_report,
              "note": "Timestamp overlap alone does not prove same capture, sensor rig or map world frame"}
    write_json(out / "provenance.json", result)
    return result

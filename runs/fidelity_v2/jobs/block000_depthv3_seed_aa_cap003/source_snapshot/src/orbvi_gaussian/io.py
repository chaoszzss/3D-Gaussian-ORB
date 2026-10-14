"""Read ROS1 bags without running ROS nodes; write only inside this project."""

import csv
import json
import math
import os
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TOPICS = {"camera_%d" % i: "/orbvi/raw/camera_%d/image/compressed" % i
              for i in range(2, 6)}
ODOM_TOPIC = "/orbvi/vio/odometry"
IMU_TOPIC = "/orbvi/imu"
IPM_TOPIC = "/orbvi/ipm/image"
PATH_TOPIC = "/orbvi/vio/path"
DEPTH_TOPIC = "/orbvi/depth/points"
LIDAR_TOPIC = "/orbvi/lidar/custom"


def output_path(path):
    """Reject traversal and symlink escapes before any write."""
    candidate = Path(path).expanduser().resolve()
    if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
        raise ValueError("All output must stay inside %s" % PROJECT_ROOT)
    return candidate


def new_output_dir(path):
    target = output_path(path)
    if target.exists():
        raise FileExistsError("Output already exists; select a new run directory: %s" % target)
    target.mkdir(parents=True)
    return target


def write_json(path, data):
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        f.write(encoded)


def open_bag(path):
    # ROS generates Python classes for unknown bag message types using tempfile.
    # Confine those otherwise implicit writes to the requested project as well.
    temp = output_path(PROJECT_ROOT / "runtime/tmp")
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp)
    tempfile.tempdir = str(temp)
    try:
        import rosbag
        return rosbag.Bag(str(Path(path).expanduser().resolve()), "r")
    except ImportError as exc:
        raise RuntimeError("Use PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 (ROS Noetic); "
                           "the conda base Python cannot load ROS1 dependencies.") from exc


def stamp_ns(stamp):
    return int(stamp.secs) * 1000000000 + int(stamp.nsecs)


def header_ns(message):
    value = stamp_ns(message.header.stamp)
    if value <= 0:
        raise ValueError("Zero/missing device header timestamp; do not silently use bag time")
    return value


def serialized_header(data, offset=0):
    """Read the first ROS Header without deserializing a large cloud/path."""
    if len(data) < offset + 16:
        raise ValueError("Truncated ROS Header")
    _, secs, nsecs, size = struct.unpack_from("<4I", data, offset)
    end = offset + 16 + size
    if end > len(data) or nsecs >= 1000000000:
        raise ValueError("Invalid ROS Header")
    ns = secs * 1000000000 + nsecs
    if ns <= 0:
        raise ValueError("Zero device timestamp; do not fall back to bag time")
    return ns, data[offset + 16:end].decode("utf-8"), end


def deserialize_raw(raw):
    msg = raw[4]()
    msg.deserialize(raw[1])
    return msg


def image_decode(message):
    if hasattr(message, "format"):
        data = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Compressed image cannot be decoded")
        return image
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if message.encoding not in channels:
        raise ValueError("Unsupported image encoding: %s" % message.encoding)
    c = channels[message.encoding]
    row_bytes = int(message.width) * c
    if int(message.step) < row_bytes or len(message.data) < int(message.height) * int(message.step):
        raise ValueError("Truncated raw image or invalid row stride")
    image = np.frombuffer(message.data, np.uint8).reshape(int(message.height), int(message.step))
    image = image[:, :row_bytes].reshape(int(message.height), int(message.width), c)
    if c == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if message.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if message.encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if message.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def inspect_bag(path):
    source = Path(path).expanduser().resolve()
    with open_bag(source) as bag:
        info = bag.get_type_and_topic_info()
        topics = {}
        for topic, meta in sorted(info.topics.items()):
            first = next(bag.read_messages(topics=[topic]), None)
            row = {"type": meta.msg_type, "messages": meta.message_count,
                   "mean_hz_rosbag": meta.frequency}
            if first:
                _, msg, recorded = first
                row.update(first_header_ns=header_ns(msg), first_bag_ns=stamp_ns(recorded),
                           frame_id=msg.header.frame_id)
                row["bag_minus_header_s"] = (row["first_bag_ns"] - row["first_header_ns"]) / 1e9
                if hasattr(msg, "child_frame_id"):
                    row["child_frame_id"] = msg.child_frame_id
                if hasattr(msg, "format") or hasattr(msg, "encoding"):
                    image = image_decode(msg)
                    row.update(width=image.shape[1], height=image.shape[0])
                if hasattr(msg, "fields"):
                    row["fields"] = [{"name": f.name, "offset": f.offset,
                                      "datatype": f.datatype, "count": f.count} for f in msg.fields]
                    row.update(point_step=msg.point_step, row_step=msg.row_step)
            topics[topic] = row
        return {"source": str(source), "size_bytes": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns,
                "bag_start_s": bag.get_start_time(), "bag_end_s": bag.get_end_time(),
                "duration_s": bag.get_end_time() - bag.get_start_time(), "topics": topics,
                "time_basis": "header_ns; bag_ns is recording provenance only"}


def _write_image(path, message, decoded):
    if hasattr(message, "format") and path.suffix == ".jpg":
        # The actual magic bytes, rather than format text, decide whether JPEG is preserved.
        with path.open("xb") as f:
            f.write(bytes(message.data))
    elif not cv2.imwrite(str(path), decoded):
        raise IOError("Cannot write image %s" % path)


def extract_bag(path, output, camera_hz=2.0, ipm_hz=0.2, cloud_hz=0.5,
                start_s=0.0, duration_s=None, include_clouds=False):
    """Stream all topics, retaining full time indexes and sampled heavy payloads.

    Windows use device header time relative to the earliest first sensor header.
    Each camera retains the nearest exposure to a common scheduled device time;
    prepare still checks the actual inter-camera timestamp residual.
    No default pose/extrinsic or frame-name based transformation is applied.
    """
    if camera_hz <= 0 or ipm_hz < 0 or cloud_hz <= 0 or start_s < 0:
        raise ValueError("Rates/window must be positive (IPM rate may be zero)")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if not all(math.isfinite(v) for v in [camera_hz, ipm_hz, cloud_hz, start_s]
               + ([] if duration_s is None else [duration_s])):
        raise ValueError("Rates/window must be finite")
    meta = inspect_bag(path)
    required = list(RAW_TOPICS.values()) + [ODOM_TOPIC]
    missing = [t for t in required if t not in meta["topics"]]
    if missing:
        raise ValueError("Missing required topics: %s" % missing)
    anchor_ns = min(x["first_header_ns"] for x in meta["topics"].values())
    begin_ns = anchor_ns + int(round(start_s * 1e9))
    end_ns = None if duration_s is None else begin_ns + int(round(duration_s * 1e9))
    out = new_output_dir(output)
    for directory in ["indexes", "images", "auxiliary/ipm", "auxiliary/clouds"]:
        (out / directory).mkdir(parents=True, exist_ok=True)
    for camera in RAW_TOPICS:
        (out / "images" / camera).mkdir()
    write_json(out / "source.json", meta)
    image_topic_to_camera = {v: k for k, v in RAW_TOPICS.items()}
    counts, saved = Counter(), Counter()
    seqs = Counter()
    next_save, previous, image_candidates = {}, {}, {}
    issues = Counter()
    last_path = None
    files = []

    def writer(name, columns):
        f = (out / "indexes" / (name + ".csv")).open("x", newline="", encoding="utf-8")
        files.append(f)
        w = csv.writer(f)
        w.writerow(columns)
        return w

    messages = writer("messages", ["topic", "index", "header_ns", "bag_ns", "frame_id"])
    images = writer("images", ["camera_id", "index", "header_ns", "bag_ns", "frame_id",
                               "file_path", "width", "height", "laplacian_variance", "mean_luma"])
    odom = writer("odometry", ["header_ns", "bag_ns", "frame_id", "child_frame_id", "tx", "ty", "tz",
                              "qx", "qy", "qz", "qw"])
    imu = writer("imu", ["header_ns", "bag_ns", "frame_id", "qx", "qy", "qz", "qw", "wx", "wy", "wz",
                         "ax", "ay", "az", "orientation_covariance_0"])
    clouds = writer("clouds", ["topic", "header_ns", "bag_ns", "frame_id", "file_path", "points",
                              "timebase_ns", "motion_compensated"])
    paths = writer("path", ["header_ns", "bag_ns", "frame_id", "poses", "first_pose_ns", "last_pose_ns"])

    def save_candidate(topic, candidate):
        _, _, raw, ns, bns, index = candidate
        msg = deserialize_raw(raw)
        image = image_decode(msg)
        camera = image_topic_to_camera.get(topic, "ipm")
        jpeg = hasattr(msg, "format") and bytes(msg.data[:2]) == b"\xff\xd8"
        name = "%06d_%d%s" % (index, ns, ".jpg" if jpeg else ".png")
        rel = Path("images") / camera / name if camera != "ipm" else Path("auxiliary/ipm") / name
        _write_image(out / rel, msg, image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.writerow([camera, index, ns, bns, msg.header.frame_id, rel.as_posix(),
                         image.shape[1], image.shape[0],
                         float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())])
        saved[topic] += 1
    try:
        with open_bag(path) as bag:
            for topic, raw, bt in bag.read_messages(raw=True):
                ns, frame_id, payload_offset = serialized_header(raw[1])
                bns = stamp_ns(bt)
                index = seqs[topic]
                seqs[topic] += 1
                if ns < begin_ns or (end_ns is not None and ns > end_ns):
                    continue
                # Never break on a device timestamp: out-of-order topics may follow.
                counts[topic] += 1
                if topic in previous and ns <= previous[topic]:
                    issues["nonincreasing:" + topic] += 1
                previous[topic] = ns
                messages.writerow([topic, index, ns, bns, frame_id])
                if topic == ODOM_TOPIC:
                    msg = deserialize_raw(raw)
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    odom.writerow([ns, bns, msg.header.frame_id, msg.child_frame_id,
                                   p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                elif topic == IMU_TOPIC:
                    msg = deserialize_raw(raw)
                    q, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                    imu.writerow([ns, bns, msg.header.frame_id, q.x, q.y, q.z, q.w,
                                  w.x, w.y, w.z, a.x, a.y, a.z, msg.orientation_covariance[0]])
                elif topic == PATH_TOPIC:
                    size = struct.unpack_from("<I", raw[1], payload_offset)[0]
                    path_first = serialized_header(raw[1], payload_offset + 4)[0] if size else 0
                    # Last pose is variable-length; do not assume a fixed frame_id stride.
                    paths.writerow([ns, bns, frame_id, size, path_first, ""])
                    last_path = raw
                elif topic in image_topic_to_camera or topic == IPM_TOPIC:
                    rate = camera_hz if topic in image_topic_to_camera else ipm_hz
                    if rate == 0:
                        continue
                    period = max(1, int(round(1e9 / rate)))
                    slot = int((ns - begin_ns + period // 2) // period)
                    distance = abs(ns - (begin_ns + slot * period))
                    old = image_candidates.get(topic)
                    if old is None or slot > old[0]:
                        if old is not None:
                            save_candidate(topic, old)
                        image_candidates[topic] = (slot, distance, raw, ns, bns, index)
                    elif slot == old[0] and distance < old[1]:
                        image_candidates[topic] = (slot, distance, raw, ns, bns, index)
                elif topic in [DEPTH_TOPIC, LIDAR_TOPIC]:
                    if ns < next_save.get(topic, 0):
                        continue
                    next_save[topic] = ns + max(1, int(round(1e9 / cloud_hz)))
                    if topic == LIDAR_TOPIC:
                        timebase, count = struct.unpack_from("<QI", raw[1], payload_offset)
                    else:
                        height, width = struct.unpack_from("<II", raw[1], payload_offset)
                        count, timebase = height * width, ""
                    rel = ""
                    if include_clouds:
                        from .points import decode_livox, decode_pointcloud2
                        msg = deserialize_raw(raw)
                        rel = "auxiliary/clouds/%s_%d.npz" % ("livox" if topic == LIDAR_TOPIC else "depth", ns)
                        if topic == LIDAR_TOPIC:
                            values = decode_livox(msg)
                        else:
                            xyz, rgb = decode_pointcloud2(msg)
                            values = {"xyz": xyz}
                            if rgb is not None:
                                values["rgb"] = rgb
                        values.update(header_ns=np.int64(ns), frame_id=np.array(msg.header.frame_id))
                        np.savez_compressed(str(out / rel), **values)
                        saved[topic] += 1
                    clouds.writerow([topic, ns, bns, frame_id, rel, count, timebase, "unverified"])
                if sum(counts.values()) % 25000 == 0:
                    print("Read %d messages; saved %d images/clouds" %
                          (sum(counts.values()), sum(saved.values())), file=sys.stderr, flush=True)
        for topic, candidate in image_candidates.items():
            save_candidate(topic, candidate)
        if last_path is not None:
            last_path = deserialize_raw(last_path)
            pathw = writer("last_path_poses", ["header_ns", "frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for pose in last_path.poses:
                p, q = pose.pose.position, pose.pose.orientation
                pathw.writerow([stamp_ns(pose.header.stamp), pose.header.frame_id,
                                p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    except Exception:
        write_json(out / "INCOMPLETE.json", {"observed": dict(counts), "saved": dict(saved),
                                             "status": "failed; do not use for training"})
        raise
    finally:
        for f in files:
            f.close()
    if not counts[ODOM_TOPIC] or any(not saved[t] for t in RAW_TOPICS.values()):
        write_json(out / "INCOMPLETE.json", {"status": "empty required streams in requested window"})
        raise ValueError("Requested window contains no odometry or no images for one of the cameras")
    manifest = {"status": "complete", "schema_version": 1, "source": meta["source"],
                "source_mtime_ns": meta["source_mtime_ns"], "source_size_bytes": meta["size_bytes"],
                "anchor_header_ns": anchor_ns, "window_start_ns": begin_ns, "window_end_ns": end_ns,
                "time_basis": "header_ns", "observed_counts": dict(counts), "saved_counts": dict(saved),
                "camera_hz": camera_hz, "ipm_hz": ipm_hz, "cloud_hz": cloud_hz,
                "sampling": "nearest exposure to shared header-time schedule",
                "cloud_payloads_saved": include_clouds, "issues": dict(issues),
                "poses": "unmodified device VIO; body/world semantics require confirmation",
                "clouds": "native frames; not automatically world registered or motion compensated"}
    write_json(out / "manifest.json", manifest)
    return manifest

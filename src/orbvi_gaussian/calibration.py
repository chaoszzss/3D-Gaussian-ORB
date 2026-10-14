"""Explicit camera/pose contracts. Never guess optical or body reference frames."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from .io import RAW_TOPICS, output_path


def load_calibration(path, require_verified=True):
    with Path(path).expanduser().open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate_calibration(data, require_verified=require_verified)
    return data


def validate_calibration(data, require_verified=True):
    from .geometry import validate_transform
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Calibration schema_version must be 1")
    if require_verified:
        if data.get("verified") is not True:
            raise ValueError("Calibration is unverified. Confirm model, camera IDs, extrinsics, "
                             "units and time basis using the real calibration; do not invent values.")
        if not data.get("calibration_id") or not data.get("provenance"):
            raise ValueError("Verified calibration requires calibration_id and provenance")
        pose = data.get("pose", {})
        if pose.get("frame_semantics_verified") is not True or not pose.get("body_frame") or not pose.get("world_frame"):
            raise ValueError("Pose body/world semantics must be explicitly verified; world/world in bag is ambiguous")
        if pose.get("timestamp_basis") != "header_ns":
            raise ValueError("Pose timestamp_basis must be header_ns")
        if pose.get("source") not in ["device_vio", "external_csv"]:
            raise ValueError("Pose source must be device_vio or external_csv")
    cameras = data.get("cameras", {})
    for camera, topic in RAW_TOPICS.items():
        c = cameras.get(camera)
        if not isinstance(c, dict) or c.get("topic") != topic:
            raise ValueError("Missing/wrong topic for %s" % camera)
        model = c.get("model")
        length = {"opencv_fisheye": 4, "equidistant": 4, "kb4": 4, "triple_sphere": 5}.get(model)
        if length is None:
            raise ValueError("Unsupported model %r for %s; actual ORBVI candidates use triple_sphere" % (model, camera))
        try:
            values = np.asarray([c["intrinsics"][k] for k in ["fx", "fy", "cx", "cy"]], dtype=float)
            distortion = np.asarray(c["distortion"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing intrinsics/distortion for %s" % camera) from exc
        if (not np.isfinite(values).all() or (values[:2] <= 0).any()
                or distortion.shape != (length,) or not np.isfinite(distortion).all()):
            raise ValueError("Invalid intrinsics/distortion for %s" % camera)
        if model == "triple_sphere" and not 0 < distortion[2] < 1:
            raise ValueError("TripleSphere alpha must be in (0, 1)")
        for key in ["width", "height"]:
            if not isinstance(c.get(key), int) or isinstance(c[key], bool) or c[key] <= 0:
                raise ValueError("%s.%s must be a positive integer" % (camera, key))
        if not isinstance(c.get("time_offset_ns"), int) or isinstance(c["time_offset_ns"], bool):
            raise ValueError("%s.time_offset_ns must be explicit integer; camera header + offset = pose time" % camera)
        if require_verified:
            if c.get("topic_mapping_verified") is not True:
                raise ValueError("Raw camera ID mapping is not verified for %s" % camera)
            if c.get("T_body_camera") is None:
                raise ValueError("Missing T_body_camera (camera -> exact pose body) for %s" % camera)
            validate_transform(c["T_body_camera"])
    views = data.get("virtual_views")
    if not isinstance(views, list) or not views:
        raise ValueError("virtual_views must contain at least one view")
    names = set()
    for v in views:
        name = v.get("name", "")
        if not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("Virtual view names must be unique safe identifiers")
        names.add(name)
        if any(not isinstance(v.get(k), int) or isinstance(v[k], bool) or v[k] < 2 for k in ["width", "height"]):
            raise ValueError("Virtual dimensions must be positive integers >= 2")
        if not np.isfinite(float(v.get("yaw_deg", 0))):
            raise ValueError("Invalid virtual yaw")
        if not 10 <= float(v.get("fov_horizontal_deg", 0)) <= 140:
            raise ValueError("Virtual horizontal FOV must be between 10 and 140 degrees")
    return data


def camera_matrix(camera):
    i = camera["intrinsics"]
    return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], dtype=float)


def virtual_parameters(view):
    width, height = view["width"], view["height"]
    focal = width / (2 * np.tan(np.deg2rad(view["fov_horizontal_deg"]) / 2))
    angle = np.deg2rad(view.get("yaw_deg", 0))
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return T, width, height, focal, focal, (width - 1) / 2, (height - 1) / 2


def import_opencv_candidate(source, target):
    """Import known TS parameters without asserting topic/body/version correctness."""
    source = Path(source).expanduser().resolve()
    fs = cv2.FileStorage(str(source), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError("Cannot read OpenCV camera_config.yaml: %s" % source)
    data = {"schema_version": 1, "verified": False, "calibration_id": "board-candidate-unverified",
            "provenance": {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "status": "candidate; model coefficients imported, topic/body/frame/version unverified"},
            "pose": {"source": "device_vio", "timestamp_basis": "header_ns", "world_frame": "world",
                     "body_frame": None, "frame_semantics_verified": False},
            "cameras": {}, "lidar": {"T_body_lidar": None, "time_offset_ns": None},
            "virtual_views": [{"name": "center", "width": 640, "height": 540,
                               "fov_horizontal_deg": 90.0, "yaw_deg": 0.0}]}
    try:
        for index, camera in enumerate(RAW_TOPICS):
            vector = fs.getNode("cam%d" % index).mat()
            relative = fs.getNode("Twc%d" % index).mat()
            if vector is None or vector.size != 9 or relative is None or relative.shape != (3, 4):
                raise ValueError("Expected cam0..3 TS nine-vector and Twc 3x4")
            p = vector.ravel().tolist()
            T = np.eye(4)
            T[:3, :] = relative
            T[:3, 3] /= 1000.0  # The board format uses mm, unlike FAST-Calib m.
            data["cameras"][camera] = {"topic": RAW_TOPICS[camera], "source_camera_id": "cam%d" % index,
                "topic_mapping_verified": False, "model": "triple_sphere", "width": 1280, "height": 1080,
                "intrinsics": dict(zip(["fx", "fy", "cx", "cy"], p[:4])), "distortion": p[4:],
                "time_offset_ns": 0, "T_body_camera": None,
                "candidate_T_rawcam0_camera": T.tolist()}
    finally:
        fs.release()
    validate_calibration(data, require_verified=False)
    path = output_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return data

"""Export already-written pinhole images and measured poses for training.

No images, points, features, correspondences, or calibration are synthesized.
COLMAP's required RGB/error fields have documented zero placeholders when
these measurements are unavailable; point observation tracks are always empty.
"""

import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Dict, List

import cv2
import numpy as np

from .geometry import (opencv_c2w_to_colmap, opencv_c2w_to_opengl,
                       validate_transform)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("{} must be an integer".format(name))
    return int(value)


def _asset_path(value, name, root):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{} must be a nonempty POSIX relative path".format(name))
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(c.isspace() for c in value):
        raise ValueError("{} must be relative, without traversal or whitespace".format(name))
    if value.startswith("#") or "\x00" in value:
        raise ValueError("{} is unsafe for training text formats".format(name))
    full = root / str(path)
    try:
        full.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("{} resolves outside output_dir".format(name)) from None
    if not full.is_file():
        raise ValueError("{} must reference an existing file: {}".format(name, full))
    return str(path)


def _prepare_frames(frames, root):
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("frames must be a nonempty list of frame dictionaries")
    result, seen = [], set()
    required = {"file_path", "w", "h", "fl_x", "fl_y", "cx", "cy",
                "transform_matrix", "timestamp_ns"}
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or not required.issubset(frame):
            raise ValueError("frame {} needs fields {}".format(index, sorted(required)))
        file_path = _asset_path(frame["file_path"], "file_path", root)
        parts = PurePosixPath(file_path).parts
        if len(parts) < 2 or parts[0] != "images":
            raise ValueError("file_path must live under images/ for COLMAP")
        if file_path in seen:
            raise ValueError("duplicate file_path: {}".format(file_path))
        seen.add(file_path)
        w, h = _integer(frame["w"], "w"), _integer(frame["h"], "h")
        if min(w, h) <= 0:
            raise ValueError("image dimensions must be positive")
        intrinsics = np.asarray([frame[k] for k in ("fl_x", "fl_y", "cx", "cy")],
                                dtype=np.float64)
        if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:2] <= 0):
            raise ValueError("frame intrinsics must be finite, focal lengths positive")
        image = cv2.imread(str(root / file_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (h, w):
            raise ValueError("image is unreadable or has different dimensions: " + file_path)
        stamp = _integer(frame["timestamp_ns"], "timestamp_ns")
        if not np.iinfo(np.int64).min <= stamp <= np.iinfo(np.int64).max:
            raise ValueError("timestamp_ns must fit int64")
        prepared = dict(file_path=file_path, w=w, h=h,
                        fl_x=float(intrinsics[0]), fl_y=float(intrinsics[1]),
                        cx=float(intrinsics[2]), cy=float(intrinsics[3]),
                        timestamp_ns=stamp,
                        transform_matrix=validate_transform(frame["transform_matrix"]))
        # Keep provenance explicit and JSON-safe; these are audit metadata,
        # never reinterpret them as Nerfstudio's depth_file_path supervision.
        for name in ("source_header_ns", "source_bag_ns"):
            if name in frame:
                value = _integer(frame[name], name)
                if not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
                    raise ValueError("{} must fit int64".format(name))
                prepared[name] = value
        for name in ("camera_id", "view_id", "source_file_path"):
            if name in frame:
                value = frame[name]
                if not isinstance(value, str) or not value or "\x00" in value:
                    raise ValueError("{} must be a nonempty string".format(name))
                # source_file_path lives in the source dataset; don't look it
                # up under output_dir or pretend the source asset was copied.
                prepared[name] = value
        for name in ("lidar_depth_path", "lidar_depth_mask_path"):
            if name in frame:
                prepared[name] = _asset_path(frame[name], name, root)
        if "group_id" in frame:
            group = frame["group_id"]
            if isinstance(group, (bool, np.bool_)):
                raise ValueError("group_id must be a nonempty string or integer")
            if isinstance(group, Integral):
                group = int(group)
            elif not isinstance(group, str) or not group:
                raise ValueError("group_id must be a nonempty string or integer")
            prepared["group_id"] = group
        if "mask_path" in frame:
            mask_path = _asset_path(frame["mask_path"], "mask_path", root)
            mask = cv2.imread(str(root / mask_path), cv2.IMREAD_UNCHANGED)
            if (mask is None or mask.shape != (h, w)
                    or not np.all(np.isin(mask, [0, 255]))):
                raise ValueError("mask must be same-size single-channel 0/255: " + mask_path)
            prepared["mask_path"] = mask_path
        result.append(prepared)
    if any("mask_path" in f for f in result) and not all("mask_path" in f for f in result):
        raise ValueError("Nerfstudio requires masks for all frames when any mask is used")
    return result


def _split_groups(frames, holdout_every):
    """Union matching timestamps and explicit group IDs, then sort in time."""
    count = len(frames)
    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    by_stamp, by_group = {}, {}
    for index, frame in enumerate(frames):
        keys = [(by_stamp, frame["timestamp_ns"])]
        if "group_id" in frame:
            group = frame["group_id"]
            keys.append((by_group, (type(group), group)))
        for table, key in keys:
            if key in table:
                parents[find(index)] = find(table[key])
            else:
                table[key] = index
    groups = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    ordered = sorted(groups.values(), key=lambda indices: (
        min(frames[i]["timestamp_ns"] for i in indices),
        min(frames[i]["file_path"] for i in indices)))
    held_out = set()
    # A one-group smoke test retains training data. Zero disables holdout.
    if holdout_every and len(ordered) > 1:
        for indices in ordered[::holdout_every]:
            held_out.update(indices)
    train = [f["file_path"] for i, f in enumerate(frames) if i not in held_out]
    val = [f["file_path"] for i, f in enumerate(frames) if i in held_out]
    return train, val, len(ordered)


def _prepare_points(points_xyz, points_rgb):
    if points_xyz is None:
        if points_rgb is not None:
            raise ValueError("points_rgb requires points_xyz")
        return None, None
    xyz = np.asarray(points_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("points_xyz must be a finite (N,3) array in the pose world frame")
    rgb = None
    if points_rgb is not None:
        raw = np.asarray(points_rgb)
        if (raw.shape != xyz.shape or raw.dtype.kind not in "iuf"
                or not np.all(np.isfinite(raw)) or np.any(raw < 0)
                or np.any(raw > 255) or np.any(raw != np.floor(raw))):
            raise ValueError("points_rgb must match XYZ with integer RGB values in [0,255]")
        rgb = raw.astype(np.uint8)
    return xyz, rgb


def _write_ply(stream, xyz, rgb):
    stream.write("ply\nformat ascii 1.0\ncomment measured points in pose world frame\n")
    stream.write("element vertex {}\n".format(len(xyz)))
    stream.write("property double x\nproperty double y\nproperty double z\n")
    if rgb is not None:
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    stream.write("end_header\n")
    for index, point in enumerate(xyz):
        line = " ".join(format(float(v), ".17g") for v in point)
        if rgb is not None:
            line += " " + " ".join(str(int(v)) for v in rgb[index])
        stream.write(line + "\n")


def export_dataset(output_dir, frames: List[dict], points_xyz=None,
                   points_rgb=None, holdout_every: int = 8) -> Dict:
    """Write transforms.json and COLMAP sparse/0/*.txt without overwriting.

    Frames require file_path (under images/), w,h,fl_x,fl_y,cx,cy,
    transform_matrix (OpenCV c2w), and integer timestamp_ns. Images must already
    exist under output_dir. Optional mask_path must reference a binary 0/255
    single-channel image; masks must be supplied for all frames or none.
    Optional source_header_ns,source_bag_ns,source_file_path,camera_id,view_id,
    lidar_depth_path,lidar_depth_mask_path survive as audit metadata. LiDAR
    paths must exist in output_dir; source_file_path is a provenance string
    referring to the original dataset. These depth fields do not request
    Nerfstudio depth supervision and are not rewritten as depth_file_path.

    Identical timestamps and optional group_id connect frames into inseparable
    time groups (group_id can connect slightly asynchronous physical cameras).
    Every holdout_every-th group, starting with the earliest, is validation;
    one group stays in training. 0 disables holdout; 1 is rejected. Input order
    never changes membership. Explicit train/val/test filenames prevent an
    image-count split; test is empty because no independent test set is given.

    Real points_xyz are optional (same world frame and units as poses). No
    points are invented if absent. Real RGB uses byte values [0,255]. Unknown
    COLMAP RGB and reprojection ERROR have zero placeholders, documented in
    comments and metadata; no image features or observation tracks are made
    up. Uncolored PLY is exported but not automatically offered to Nerfstudio's
    colored-point initializer. Colored PLY adds ply_file_path to transforms.

    Returns counts and output paths. Validates before writing, preflights every
    target, opens with exclusive creation, and rolls back its new files on
    failure. Existing image assets and unrelated files are left in place.
    """
    holdout_every = _integer(holdout_every, "holdout_every")
    if holdout_every < 0 or holdout_every == 1:
        raise ValueError("holdout_every must be 0 (disabled) or at least 2")
    root = Path(output_dir)
    if root.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    sparse = root / "sparse" / "0"
    for path in (root, root / "sparse", sparse):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("output directories must be real directories: {}".format(path))
    targets = {"transforms": root / "transforms.json",
               "cameras": sparse / "cameras.txt", "images": sparse / "images.txt",
               "points3D": sparse / "points3D.txt"}
    if points_xyz is not None:
        targets["point_cloud"] = root / "points3D.ply"
    # Also reject dangling symlinks, which Path.exists() alone misses.
    for path in targets.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError("refusing to overwrite existing output: {}".format(path))
    prepared = _prepare_frames(frames, root)
    xyz, rgb = _prepare_points(points_xyz, points_rgb)
    train, val, group_count = _split_groups(prepared, holdout_every)
    camera_ids, cameras, ns_frames = {}, [], []
    for frame in prepared:
        key = tuple(frame[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy"))
        if key not in camera_ids:
            camera_ids[key] = len(cameras) + 1
            cameras.append(key)
        ns_frame = {k: v for k, v in frame.items() if k != "transform_matrix"}
        # Rectification/projection use integer OpenCV pixel centers. Nerfstudio
        # and gsplat sample at (column+.5,row+.5); preserve the same physical ray.
        ns_frame['cx'] += .5
        ns_frame['cy'] += .5
        ns_frame["transform_matrix"] = opencv_c2w_to_opengl(frame["transform_matrix"]).tolist()
        ns_frames.append(ns_frame)
    metadata = dict(camera_model="OPENCV", frames=ns_frames,
                    train_filenames=train, val_filenames=val, test_filenames=[],
                    orientation_override="none",
                    point_cloud_has_rgb=rgb is not None,
                    point_cloud_has_observations=False,
                    split_group_count=group_count, holdout_every=holdout_every)
    metadata['pixel_center_convention']='half_integer; principal point +0.5 from OpenCV rectification'
    if xyz is not None:
        metadata["point_cloud_path"] = "points3D.ply"
        if rgb is not None and len(xyz):
            metadata["ply_file_path"] = "points3D.ply"
    sparse.mkdir(parents=True, exist_ok=True)
    opened = {}
    try:
        for key, path in targets.items():
            opened[key] = path.open("x", encoding="utf-8", newline="\n")
        json.dump(metadata, opened["transforms"], indent=2, ensure_ascii=False, allow_nan=False)
        opened["transforms"].write("\n")
        camera_stream = opened["cameras"]
        camera_stream.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        camera_stream.write("# Number of cameras: {}\n".format(len(cameras)))
        for camera_id, key in enumerate(cameras, 1):
            w, h, fx, fy, cx, cy = key
            params = " ".join(format(v, ".17g") for v in (fx, fy, cx, cy))
            camera_stream.write("{} PINHOLE {} {} {}\n".format(camera_id, w, h, params))
        image_stream = opened["images"]
        image_stream.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        image_stream.write("# POINTS2D[] (empty: no measured feature observations supplied)\n")
        image_stream.write("# Number of images: {}, mean observations per image: 0\n".format(len(prepared)))
        for image_id, frame in enumerate(prepared, 1):
            q, t = opencv_c2w_to_colmap(frame["transform_matrix"])
            pose = " ".join(format(float(v), ".17g") for v in np.concatenate([q, t]))
            key = tuple(frame[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy"))
            name = str(PurePosixPath(frame["file_path"]).relative_to("images"))
            image_stream.write("{} {} {} {}\n\n".format(image_id, pose, camera_ids[key], name))
        point_stream = opened["points3D"]
        point_stream.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        point_stream.write("# Measured XYZ only; empty TRACK: no observations supplied.\n")
        point_stream.write("# ERROR=0 is an unavailable-error placeholder, not a measured reprojection error.\n")
        if rgb is None:
            point_stream.write("# RGB=0,0,0 is an unavailable-color placeholder, not a measured color.\n")
        point_count = 0 if xyz is None else len(xyz)
        point_stream.write("# Number of points: {}, mean track length: 0\n".format(point_count))
        if xyz is not None:
            for index, point in enumerate(xyz):
                coords = " ".join(format(float(v), ".17g") for v in point)
                colors = rgb[index] if rgb is not None else (0, 0, 0)
                colors = " ".join(str(int(v)) for v in colors)
                point_stream.write("{} {} {} 0\n".format(index + 1, coords, colors))
            _write_ply(opened["point_cloud"], xyz, rgb)
        # Flush/close errors must also trigger rollback of this writer's files.
        for stream in opened.values():
            stream.close()
    except BaseException:
        for key, stream in opened.items():
            try:
                stream.close()
            finally:
                targets[key].unlink(missing_ok=True)
        raise
    return dict(frame_count=len(prepared), camera_count=len(cameras),
                point_count=0 if xyz is None else len(xyz), group_count=group_count,
                train_count=len(train), val_count=len(val), test_count=0,
                paths={key: str(path) for key, path in targets.items()})

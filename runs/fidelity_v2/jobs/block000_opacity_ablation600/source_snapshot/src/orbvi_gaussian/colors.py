"""Image-supported RGB for measured map geometry; no intensity pseudocoloring."""

from pathlib import Path

import cv2
import numpy as np

from .geometry import validate_transform
from .points import project_depth


def colorize_from_frames(xyz, frames, image_root, depth_tolerance_m=0.05, visibility_xyz=None):
    """Keep the closest supported view color; caller supplies TRAINING frames only.

    Visibility is a sparse point z-buffer approximation. Unobserved points are
    returned with supported=False and must not be advertised as measured RGB.
    This does not establish dense free-space visibility or remove moving objects.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Colorization needs finite world-frame XYZ")
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0:
        raise ValueError("Depth tolerance must be finite and nonnegative")
    root = Path(image_root).resolve()
    rgb = np.zeros((len(xyz), 3), np.uint8)
    scores = np.full(len(xyz), np.inf)
    for frame in frames:
        T = validate_transform(frame["transform_matrix"])
        camera = (xyz - T[:3, 3]) @ T[:3, :3]
        positive = camera[:, 2] > 0.05
        ids = np.flatnonzero(positive)
        q = camera[ids]
        K = np.array([[frame["fl_x"], 0, frame["cx"]], [0, frame["fl_y"], frame["cy"]], [0, 0, 1]])
        w, h = frame["w"], frame["h"]
        u = q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2]
        v = q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2]
        inside = np.isfinite(u) & np.isfinite(v) & (u >= -0.5) & (v >= -0.5) & (u < w - 0.5) & (v < h - 0.5)
        ids, q = ids[inside], q[inside]
        u, v = np.floor(u[inside] + 0.5).astype(int), np.floor(v[inside] + 0.5).astype(int)
        if not len(ids):
            continue
        depth, valid = project_depth(xyz if visibility_xyz is None else visibility_xyz, T, K, w, h)
        if visibility_xyz is not None:
            # Visibility only: conservative nearby measured occluders can reject
            # an initializer point. This does not create synthetic depth targets.
            buffer = np.where(valid, depth, np.float32(1e8))
            depth = cv2.erode(buffer, np.ones((3, 3), np.uint8))
            valid = depth < 1e8
        visible = (valid[v, u] > 0) & (np.abs(q[:, 2] - depth[v, u]) <= np.maximum(depth_tolerance_m, q[:,2]*.005))
        image_path = (root / frame["file_path"]).resolve()
        if root not in image_path.parents:
            raise ValueError("Colorization image escapes dataset")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (h, w):
            raise ValueError("Invalid color image")
        if frame.get("mask_path"):
            mask_path = (root / frame["mask_path"]).resolve()
            if root not in mask_path.parents:
                raise ValueError("Colorization mask escapes dataset")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (h, w):
                raise ValueError("Invalid color mask")
            visible &= mask[v, u] > 0
        # Favor large projected pixel area, rather than blending exposure seams.
        score = np.sum(q * q, axis=1) / (K[0, 0] * K[1, 1])
        visible &= score < scores[ids]
        chosen = ids[visible]
        rgb[chosen] = image[v[visible], u[visible], ::-1]  # BGR -> measured RGB
        scores[chosen] = score[visible]
    return rgb, np.isfinite(scores)

"""Calibrated camera geometry. All T_parent_child map child into parent.

Images and camera poses use OpenCV axes (right, down, forward). No
calibration, extrinsic direction, or distortion model is inferred.
"""

from numbers import Integral
from collections.abc import Mapping
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# Equations checked against local ORBVI/Vikit sources, independently expressed
# as vectorized sphere projections/intersections (not copied C++ implementation).
# Primary publication: Xie et al., OmniVidar, CVPR 2023, pp. 21529-21538.
# https://openaccess.thecvf.com/content/CVPR2023/html/Xie_OmniVidar_Omnidirectional_Depth_Estimation_From_Multi-Fisheye_Images_CVPR_2023_paper.html
# Author calibration equations/parameter order:
# https://github.com/imuncle/TSCM_Calib/blob/master/TS.cpp
# https://github.com/imuncle/TSCM_Calib/blob/master/TS.h
# /home/chaoszzss/orbvi_proj_ws/src/rpg_vikit/vikit_common/src/triple_sphere_camera.cpp
TRIPLE_SPHERE_SOURCE_SHA256 = "149a4b0d3cf9fc075f8570c5d90366f54843fc03375454343adb46db68228b64"
# include/vikit/triple_sphere_camera.h constructor: fx,fy,cx,cy,xi,lambda,alpha,b,c
TRIPLE_SPHERE_HEADER_SHA256 = "ee11453ec162f4f9625a7547ec87872d0b0ea4af5b8a3b2722aba09066e29a1c"
TRIPLE_SPHERE_PARAMETER_ORDER = ("fx", "fy", "cx", "cy", "xi", "lambda", "alpha", "b", "c")


def quat_xyzw_to_matrix(q) -> np.ndarray:
    """Return a rotation from a finite, nonzero Hamilton quaternion x,y,z,w.

    Explicitly normalizes quaternions, including very large/small magnitudes.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must have four finite xyzw components")
    scale = np.max(np.abs(q))
    if scale == 0:
        raise ValueError("zero quaternion has no rotation")
    q = q / scale
    q = q / np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def validate_transform(T) -> np.ndarray:
    """Return a copy of a finite 4x4 rigid transform; reject reflections/scale.

    Rotations are checked to absolute tolerance 1e-6, the homogeneous row to
    1e-9. Invalid rotations are never silently orthogonalized.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], rtol=0, atol=1e-9):
        raise ValueError("transform must have homogeneous row [0,0,0,1]")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-6):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, rtol=0, atol=1e-6):
        raise ValueError("transform rotation must have determinant +1")
    return T.copy()


def matrix_to_quat_xyzw(R) -> np.ndarray:
    """Convert a proper 3x3 rotation to a normalized xyzw quaternion."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    T = np.eye(4)
    T[:3, :3] = R
    validate_transform(T)
    q = Rotation.from_matrix(R).as_quat()
    if q[3] < 0:
        q = -q
    return q


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("{} must be an integer".format(name))
    return int(value)


class PoseTrajectory:
    """Strictly increasing integer-nanosecond body poses in a fixed world.

    Translation is linear; rotation follows shortest-path SLERP. Interpolation
    across an interval longer than max_gap_ns and extrapolation return None.
    Exact measured poses remain available even beside a rejected interval.
    The default maximum gap is 200 ms and must be chosen for the actual sensor.
    """

    def __init__(self, stamps_ns: np.ndarray, T_world_body: np.ndarray,
                 max_gap_ns: int = 200_000_000):
        raw = np.asarray(stamps_ns)
        if raw.ndim != 1 or len(raw) == 0 or raw.dtype.kind not in "iu":
            raise ValueError("stamps_ns must be a nonempty 1D integer array")
        if np.any(raw > np.iinfo(np.int64).max):
            raise ValueError("timestamps must fit signed int64")
        self.stamps_ns = raw.astype(np.int64, copy=True)
        if np.any(self.stamps_ns[1:] <= self.stamps_ns[:-1]):
            raise ValueError("timestamps must be strictly increasing, no duplicates")
        self.max_gap_ns = _integer(max_gap_ns, "max_gap_ns")
        if self.max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        poses = np.asarray(T_world_body, dtype=np.float64)
        if poses.shape != (len(raw), 4, 4):
            raise ValueError("T_world_body must have shape (N,4,4)")
        self.T_world_body = np.stack([validate_transform(T) for T in poses])
        self.stamps_ns.setflags(write=False)
        self.T_world_body.setflags(write=False)

    def at(self, stamp_ns: int) -> Optional[np.ndarray]:
        stamp = _integer(stamp_ns, "stamp_ns")
        if stamp < int(self.stamps_ns[0]) or stamp > int(self.stamps_ns[-1]):
            return None
        index = int(np.searchsorted(self.stamps_ns, stamp, side="left"))
        if int(self.stamps_ns[index]) == stamp:
            return self.T_world_body[index].copy()
        left, right = index - 1, index
        gap = int(self.stamps_ns[right]) - int(self.stamps_ns[left])
        if gap > self.max_gap_ns:
            return None
        # Subtract integers before conversion: epoch nanoseconds exceed 2**53.
        alpha = (stamp - int(self.stamps_ns[left])) / gap
        rotations = Rotation.from_matrix(self.T_world_body[[left, right], :3, :3])
        T = np.eye(4)
        T[:3, :3] = Slerp([0.0, 1.0], rotations)(alpha).as_matrix()
        T[:3, 3] = ((1 - alpha) * self.T_world_body[left, :3, 3]
                    + alpha * self.T_world_body[right, :3, 3])
        return T


def camera_world_pose(T_world_body, T_body_camera,
                      T_camera_virtual) -> np.ndarray:
    """Compose OpenCV c2w: T_world_body @ T_body_camera @ T_camera_virtual."""
    return (validate_transform(T_world_body) @ validate_transform(T_body_camera)
            @ validate_transform(T_camera_virtual))


def opencv_c2w_to_colmap(T_world_camera) -> Tuple[np.ndarray, np.ndarray]:
    """Return COLMAP world-to-camera (quaternion wxyz, translation xyz)."""
    T = validate_transform(T_world_camera)
    R = T[:3, :3].T
    t = -R @ T[:3, 3]
    q = matrix_to_quat_xyzw(R)
    return q[[3, 0, 1, 2]], t


def opencv_c2w_to_opengl(T_world_camera) -> np.ndarray:
    """Nerfstudio c2w: flip camera Y/Z columns, preserving the world frame."""
    return validate_transform(T_world_camera) @ np.diag([1., -1., -1., 1.])


def opengl_c2w_to_opencv(T_world_camera) -> np.ndarray:
    """Inverse of opencv_c2w_to_opengl (the axis flip is self-inverse)."""
    return opencv_c2w_to_opengl(T_world_camera)


def _validate_fisheye(K, D, model):
    supported = {"equidistant", "kb4", "opencv_fisheye"}
    if not isinstance(model, str) or model.lower() not in supported:
        raise ValueError("explicit supported model required: equidistant, kb4, "
                         "opencv_fisheye; omni/scaramuzza need another adapter")
    if K is None or D is None:
        raise ValueError("explicit calibrated K and D are required")
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("calibrated K must be a finite 3x3 matrix")
    if (K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12)
            or not np.isclose(K[1, 0], 0, rtol=0, atol=1e-12)
            or not np.isclose(K[0, 1], 0, rtol=0, atol=1e-12)):
        raise ValueError("K needs positive focal lengths and zero skew")
    if D.shape not in ((4,), (4, 1), (1, 4)) or not np.all(np.isfinite(D)):
        raise ValueError("OpenCV equidistant/KB4 requires exactly four finite coefficients")
    return K, D.reshape(4, 1)


def _validate_triple_sphere(K, model_parameters):
    if K is None or not isinstance(model_parameters, Mapping):
        raise ValueError("TripleSphere requires explicit K and named model_parameters")
    names = {"xi", "lambda", "alpha", "b", "c"}
    if set(model_parameters) != names:
        raise ValueError("TripleSphere model_parameters must contain exactly xi,lambda,alpha,b,c")
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.all(np.isfinite(K))
            or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12)
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose([K[0, 1], K[1, 0]], 0, rtol=0, atol=1e-12)):
        raise ValueError("TripleSphere K needs finite positive fx/fy; b,c belong in model_parameters")
    values = np.asarray([model_parameters[k] for k in ("xi", "lambda", "alpha", "b", "c")],
                        dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("TripleSphere parameters must be finite scalars")
    xi, lam, alpha, b, c = values
    if not 0 <= alpha < 1:
        raise ValueError("TripleSphere alpha must satisfy 0 <= alpha < 1")
    A = np.array([[K[0, 0], b], [c, K[1, 1]]])
    if abs(np.linalg.det(A)) <= 1e-12 * max(1., np.linalg.norm(A) ** 2):
        raise ValueError("TripleSphere affine pixel matrix is singular")
    return K, A, xi, lam, alpha


def _triple_unproject(pixels, K, A, xi, lam, alpha):
    """Intersect the image ray and undo the three sphere shifts."""
    flat = pixels.reshape(-1, 2)
    m = (flat - K[:2, 2]) @ np.linalg.inv(A).T
    radius2 = np.sum(m * m, axis=1)
    kappa = alpha / (1 - alpha)
    radicand = 1 + (1 - kappa * kappa) * radius2
    valid = np.all(np.isfinite(flat), axis=1) & (radicand >= -1e-10)
    scale = (kappa + np.sqrt(np.maximum(radicand, 0))) / (1 + radius2)
    rays = np.column_stack([scale[:, None] * m, scale - kappa])
    for offset in (lam, xi):
        radicand = 1 - offset * offset * (1 - rays[:, 2] * rays[:, 2])
        distance = offset * rays[:, 2] + np.sqrt(np.maximum(radicand, 0))
        valid &= (radicand >= -1e-10) & (distance > 1e-12)
        rays *= distance[:, None]
        rays[:, 2] -= offset
    norm = np.linalg.norm(rays, axis=1)
    valid &= np.all(np.isfinite(rays), axis=1) & (norm > 1e-12)
    rays /= np.where(norm > 1e-12, norm, 1)[:, None]
    rays[~valid] = np.nan
    return rays.reshape(pixels.shape[:-1] + (3,)), valid.reshape(pixels.shape[:-1])


def triple_sphere_unproject(pixels, K, model_parameters):
    """Return (unit source rays, bool domain mask) for (...,2) pixels.

    Parameter order in the original nine-vector is fx,fy,cx,cy,xi,lambda,
    alpha,b,c. Here fx/fy/cx/cy are in K and all five other values are required
    by name; lambda is the second sphere shift, not a KB4 coefficient. Invalid
    rays are NaN. Validity concerns the mathematical domain, not image bounds.
    """
    calibration = _validate_triple_sphere(K, model_parameters)
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim < 1 or pixels.shape[-1] != 2:
        raise ValueError("pixels must have shape (...,2)")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _triple_unproject(pixels, *calibration)


def triple_sphere_project(rays, K, model_parameters):
    """Return (pixels, bool domain mask) for (...,3) source-camera rays.

    Supports legal rays beyond 90 degrees. Rejects nonpositive denominators,
    singular/zero rays and projections whose positive-root sphere inverse
    selects another direction. This round-trip check excludes folded branches
    without imposing the inappropriate OpenCV pinhole z>0 condition.
    """
    calibration = _validate_triple_sphere(K, model_parameters)
    K, A, xi, lam, alpha = calibration
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim < 1 or rays.shape[-1] != 3:
        raise ValueError("rays must have shape (...,3)")
    flat = rays.reshape(-1, 3)
    # Unit normalization makes the validity threshold independent of ray scale.
    largest = np.max(np.abs(flat), axis=1)
    valid = np.all(np.isfinite(flat), axis=1) & (largest > 0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        unit = flat / np.where(valid, largest, 1)[:, None]
        unit /= np.where(valid, np.linalg.norm(unit, axis=1), 1)[:, None]
        radial2 = np.sum(unit[:, :2] ** 2, axis=1)
        z = unit[:, 2] + xi
        z += lam * np.sqrt(radial2 + z * z)
        denominator = z + alpha / (1 - alpha) * np.sqrt(radial2 + z * z)
        valid &= np.isfinite(denominator) & (denominator > 1e-12)
        normalized = unit[:, :2] / np.where(valid, denominator, 1)[:, None]
        pixels = normalized @ A.T + K[:2, 2]
        inverse, inverse_valid = _triple_unproject(pixels, *calibration)
        valid &= inverse_valid & (np.linalg.norm(inverse - unit, axis=1) < 1e-7)
    pixels[~valid] = np.nan
    return pixels.reshape(rays.shape[:-1] + (2,)), valid.reshape(rays.shape[:-1])


def make_virtual_view(image, K, D, T_camera_virtual, width, height,
                      fx, fy, cx, cy, *, model=None, source_mask=None,
                      model_parameters=None):
    """Rectify calibrated OpenCV equidistant/KB4 or TripleSphere to pinhole.

    model must explicitly be equidistant, kb4, opencv_fisheye or triple_sphere.
    TripleSphere uses D=None and model_parameters={xi,lambda,alpha,b,c}; no
    parameters may be omitted. Its legal >90 degree source rays are retained.
    The first three model names
    describe theta*(1+k1*theta**2+...+k4*theta**8), not arbitrary fisheye models.
    T_camera_virtual maps virtual rays to source camera rays and must be a
    pure rotation: a translated viewpoint needs scene depth. Intrinsics and
    dimensions are explicit; no default/fabricated calibration is supplied.

    Returns (rectified_image, uint8_mask), mask values 0 or 255. Validity checks
    the model domain, map bounds, and complete bilinear sampling support.
    Optional source_mask is a same-sized single-channel 0/1 or 0/255 validity
    mask, needed for circular lens borders and known unusable source regions.
    Pixel values (including black) are never used to guess validity.
    """
    is_triple = isinstance(model, str) and model.lower() == "triple_sphere"
    if is_triple:
        if D is not None:
            raise ValueError("TripleSphere uses model_parameters, never a KB4 D vector")
        _validate_triple_sphere(K, model_parameters)
    else:
        if model_parameters is not None:
            raise ValueError("KB4/equidistant uses D, not model_parameters")
        K, D = _validate_fisheye(K, D, model)
    width = _integer(width, "width")
    height = _integer(height, "height")
    if width <= 0 or height <= 0:
        raise ValueError("virtual width and height must be positive")
    params = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.all(np.isfinite(params)) or fx <= 0 or fy <= 0:
        raise ValueError("virtual intrinsics must be finite with positive fx/fy")
    T = validate_transform(T_camera_virtual)
    if not np.allclose(T[:3, 3], 0, rtol=0, atol=1e-12):
        raise ValueError("virtual camera must share source center (pure rotation)")
    if (not isinstance(image, np.ndarray) or image.ndim not in (2, 3)
            or min(image.shape[:2]) <= 0
            or (image.ndim == 3 and image.shape[2] not in (1, 3, 4))):
        raise ValueError("image must be a nonempty 2D or 1/3/4-channel array")
    if image.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        raise ValueError("image dtype is unsupported by OpenCV remap")
    source_h, source_w = image.shape[:2]
    source_valid = np.ones((source_h, source_w), dtype=np.float32)
    if source_mask is not None:
        mask = np.asarray(source_mask)
        if (mask.shape != (source_h, source_w)
                or not np.all(np.isin(mask, [0, 1, 255]))):
            raise ValueError("source_mask must match image and contain only 0/1/255")
        source_valid = (mask != 0).astype(np.float32)
    virtual_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    yy, xx = np.indices((height, width), dtype=np.float64)
    if is_triple:
        virtual_rays = np.stack([(xx - cx) / fx, (yy - cy) / fy,
                                 np.ones_like(xx)], axis=-1)
        source_rays = virtual_rays @ T[:3, :3].T
        pixels, domain_valid = triple_sphere_project(source_rays, K, model_parameters)
        map_x, map_y = pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32)
    else:
        # OpenCV R maps source to rectified, opposite to T_camera_virtual.
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K, D, T[:3, :3].T, virtual_K, (width, height), cv2.CV_32FC1)
        source_z = (T[2, 0] * (xx - cx) / fx
                    + T[2, 1] * (yy - cy) / fy + T[2, 2])
        domain_valid = source_z > 1e-12
    valid = (np.isfinite(map_x) & np.isfinite(map_y) & domain_valid
             & (map_x >= 0) & (map_x <= source_w - 1)
             & (map_y >= 0) & (map_y <= source_h - 1))
    safe_x = np.where(valid, map_x, -1).astype(np.float32)
    safe_y = np.where(valid, map_y, -1).astype(np.float32)
    sampled_mask = cv2.remap(source_valid, safe_x, safe_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid &= sampled_mask == 1.0
    # Check mathematical neighbors too: remap quantizes interpolation to 1/32.
    x0 = np.clip(np.floor(safe_x), 0, source_w - 1).astype(np.intp)
    x1 = np.clip(np.ceil(safe_x), 0, source_w - 1).astype(np.intp)
    y0 = np.clip(np.floor(safe_y), 0, source_h - 1).astype(np.intp)
    y1 = np.clip(np.ceil(safe_y), 0, source_h - 1).astype(np.intp)
    valid &= ((source_valid[y0, x0] != 0) & (source_valid[y0, x1] != 0)
              & (source_valid[y1, x0] != 0) & (source_valid[y1, x1] != 0))
    rectified = cv2.remap(image, safe_x, safe_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if image.ndim == 3 and image.shape[2] == 1:
        rectified = rectified[..., None]
    rectified[~valid] = 0
    return rectified, valid.astype(np.uint8) * 255

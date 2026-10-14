"""Metric, timestamp-shared four-sensor extrinsic corrections (torch only).

The six-vector is ROTATION FIRST: [wx, wy, wz, vx, vy, vz]. Rotation is
axis-angle in radians; v is an SE(3) translation tangent in meters. Both use
the fixed sensor's Nerfstudio/OpenGL axes: right/up/backward. The exponential
has rotation Exp(w) and translation J_left(w) @ v, NOT an uncoupled offset v.
For column-vector poses, correction is T_world_camera @ Exp(delta).

Inputs must already be metric NS/OpenGL c2w, with the same local camera basis
for each sensor at every timestamp. Distinct virtual-view orientations need
conjugation into their view basis by the caller; see the integration report.
No image-index, timestamp, FAST-calibration index, or viewer pose is an ID.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import Tensor, nn


def _check_vector(value: Tensor, size: int) -> None:
    if not isinstance(value, Tensor) or value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"Expected floating tensor with shape [...,{size}]")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64 for rigid transforms")


def _so3_terms(omega: Tensor):
    _check_vector(omega, 3)
    x, y, z = omega.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(omega.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(skew)
    theta2 = omega.square().sum(-1)
    # Both branches must have finite derivatives: torch.where still evaluates
    # its unused branch. Polynomial coefficients avoid float32 cancellation.
    safe2 = theta2.clamp_min(1e-4)
    theta = safe2.sqrt()
    small = theta2 < 1e-4
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120 - theta2**3 / 5040,
                    theta.sin() / theta)
    b = torch.where(small, .5 - theta2 / 24 + theta2.square() / 720 - theta2**3 / 40320,
                    (1 - theta.cos()) / safe2)
    c = torch.where(small, 1 / 6 - theta2 / 120 + theta2.square() / 5040 - theta2**3 / 362880,
                    (theta - theta.sin()) / (safe2 * theta))
    return skew, eye, a[..., None, None], b[..., None, None], c[..., None, None]


def so3_exp(omega: Tensor) -> Tensor:
    """Differentiable SO(3) exponential of [...,3] axis-angle radians."""
    skew, eye, a, b, _ = _so3_terms(omega)
    return eye + a * skew + b * (skew @ skew)


def se3_exp(delta: Tensor) -> Tensor:
    """True SE(3) matrix exponential; [...,6] [omega,v] -> [...,4,4].

    At zero, dR/domega is the skew generator and dt/dv is identity.
    v is metric body-frame tangent translation; actual local offset is J @ v.
    """
    _check_vector(delta, 6)
    skew, eye, a, b, c = _so3_terms(delta[..., :3])
    skew2 = skew @ skew
    rotation = eye + a * skew + b * skew2
    jacobian = eye + b * skew + c * skew2
    translation = jacobian @ delta[..., 3:, None]
    top = torch.cat((rotation, translation), -1)
    bottom = delta.new_tensor([0., 0., 0., 1.]).expand(delta.shape[:-1] + (1, 4))
    return torch.cat((top, bottom), -2)


class SharedRigOptimizer(nn.Module):
    """Exactly four corrections, shared across timestamps and train/eval.

    Default row-to-ID mapping is [2,3,4,5] (raw physical camera IDs).
    physical_camera_id_offset=0 explicitly selects [0,1,2,3], in that SAME
    raw-sensor order; these compact IDs are not FAST cam0..3 indices.
    Never mix the two conventions: the overlapping IDs 2/3 cannot be inferred.

    Forward corrections obey radial bounds even before project_() is called.
    delta itself is an ordinary optimizable parameter: call project_() after
    each optimizer step to keep its stored value in the trust region. Radial
    clipping is differentiable away from the boundary; regularization on raw
    delta still pulls an overshooting parameter inward. All ID, prior, and
    bound buffers are persistent, so state_dict includes their semantics.
    """

    def __init__(
        self,
        *,
        physical_camera_id_offset: int = 2,
        rotation_prior_rad: float = math.radians(2),
        translation_prior_m: float = .05,
        max_rotation_rad: float = math.radians(12),
        max_translation_m: float = .2,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if (isinstance(physical_camera_id_offset, bool)
                or not isinstance(physical_camera_id_offset, Integral)
                or physical_camera_id_offset not in (0, 2)):
            raise ValueError("physical_camera_id_offset must be 0 or 2")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 for rigid transforms")
        values = dict(rotation_prior_rad=rotation_prior_rad, translation_prior_m=translation_prior_m,
                      max_rotation_rad=max_rotation_rad, max_translation_m=max_translation_m)
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            self.register_buffer(name, torch.tensor(value, device=device, dtype=dtype))
        if max_rotation_rad >= math.pi:
            raise ValueError("max_rotation_rad must be below pi for an unambiguous local correction")
        self.register_buffer("physical_camera_ids", torch.arange(4, device=device, dtype=torch.long)
                             + physical_camera_id_offset)
        self.delta = nn.Parameter(torch.zeros(4, 6, device=device, dtype=dtype))

    def effective_delta(self) -> Tensor:
        """Return rotation/translation-tangent vectors within their L2 balls.

        J_left has operator norm <= 1, so bounding ||v|| also bounds the actual
        camera-center offset ||J_left(w)v||, independent of world rotation.
        """
        def bounded(vector, limit):
            norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
            return vector * (limit / norm.clamp_min(limit))

        return torch.cat((bounded(self.delta[:, :3], self.max_rotation_rad),
                          bounded(self.delta[:, 3:], self.max_translation_m)), -1)

    def correction_matrices(self) -> Tensor:
        """The four bounded local SE(3) matrices, in physical_camera_ids order."""
        return se3_exp(self.effective_delta())

    def correct(self, camera_or_c2w, physical_camera_id=None) -> Tensor:
        """Return corrected c2w, preserving [...,3,4] or [...,4,4] shape.

        Accept a torch c2w tensor or a Cameras-like object with camera_to_worlds.
        Explicit integer IDs override camera.metadata['physical_camera_id'].
        IDs may be scalar, broadcastable to camera batch dimensions, or have
        an extra singleton final dimension as in NS camera metadata.
        Missing IDs and integer IDs outside this rig are unchanged. In
        particular, metadata['cam_idx'] is NEVER used. No object is mutated.
        train()/eval() do not change this behavior: apply learned corrections
        to explicitly identified held-out cameras without fitting on them.
        """
        if isinstance(camera_or_c2w, Tensor):
            c2w = camera_or_c2w
        else:
            c2w = getattr(camera_or_c2w, "camera_to_worlds", None)
            metadata = getattr(camera_or_c2w, "metadata", None)
            if physical_camera_id is None and isinstance(metadata, Mapping):
                physical_camera_id = metadata.get("physical_camera_id")
        if not isinstance(c2w, Tensor) or c2w.ndim < 2 or c2w.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError("c2w must be a tensor with shape [...,3,4] or [...,4,4]")
        if c2w.dtype not in (torch.float32, torch.float64):
            raise ValueError("c2w must be float32 or float64 in metric NS/OpenGL coordinates")
        if physical_camera_id is None:
            return c2w
        if c2w.dtype != self.delta.dtype or c2w.device != self.delta.device:
            raise ValueError("Move rig to the c2w dtype/device before applying a correction")
        ids = torch.as_tensor(physical_camera_id, device=c2w.device)
        if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("physical_camera_id must contain integers, not bool/float or cam_idx")
        batch_shape = c2w.shape[:-2]
        if ids.numel() == 1:
            ids = ids.reshape(())
        elif ids.ndim == len(batch_shape) + 1 and ids.shape[-1] == 1:
            ids = ids.squeeze(-1)
        try:
            ids = torch.broadcast_to(ids, batch_shape)
        except RuntimeError as error:
            raise ValueError("physical_camera_id shape must broadcast to c2w batch dimensions") from error
        matches = ids[..., None] == self.physical_camera_ids
        known = matches.any(-1)
        rows = matches.to(torch.long).argmax(-1)
        corrected = c2w @ self.correction_matrices()[rows]
        return torch.where(known[..., None, None], corrected, c2w)

    def forward(self, camera_or_c2w, physical_camera_id=None) -> Tensor:
        return self.correct(camera_or_c2w, physical_camera_id)

    def regularization(self) -> dict[str, Tensor]:
        """Named dimensionless Gaussian negative-log-priors, averaged over 4.

        Each is 0.5 * mean_sensor(||vector / sigma||^2). sigma is an isotropic
        per-axis standard deviation (2 degrees, .05 m by default), NOT a hard
        bound. These are raw-tangent priors; external loss weights are a main
        training choice. Penalizing raw delta avoids a saturated prior outside
        the forward trust region. Does not read any images or held-out data.
        """
        return {
            "rig_rotation_prior": .5 * (self.delta[:, :3] / self.rotation_prior_rad).square().sum(-1).mean(),
            "rig_translation_prior": .5 * (self.delta[:, 3:] / self.translation_prior_m).square().sum(-1).mean(),
        }

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Merge into main model's groups; configure its separate 'rig' optimizer."""
        return {"rig": [self.delta]}

    @torch.no_grad()
    def project_(self) -> SharedRigOptimizer:
        """Project the stored parameter after optimizer.step(); keeps its identity."""
        self.delta.copy_(self.effective_delta())
        return self

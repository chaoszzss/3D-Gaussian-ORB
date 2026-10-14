# Shared four-camera rig correction sidecar

Scope: only `src/orbvi_gaussian/rig_optimizer.py`, `tests/test_rig_optimizer.py`,
and this report. Main model, managers, scripts, installed packages, and
third-party code are untouched by this sidecar. The parent owns photometric
calibration and the six-floor reconstruction. No GPU work is performed here.
Parent-reported context is first-block PSNR 22.13 / SSIM 0.73 and median mean
surface distance 0.033 m. These capture-specific results were not rerun here.

## API and exact transform convention

```python
from orbvi_gaussian.rig_optimizer import SharedRigOptimizer

rig = SharedRigOptimizer(device=c2w.device, dtype=c2w.dtype)
corrected_c2w = rig.correct(camera, physical_camera_id=2)
# Or: rig.correct(c2w, physical_camera_id=ids)
# Or: rig.correct(camera) with metadata['physical_camera_id'] populated.
groups.update(rig.get_param_groups())  # {'rig': [rig.delta]}
losses.update(rig.regularization())    # two named dimensionless priors
# After the separate rig optimizer's step:
rig.project_()
```

`delta` is exactly one trainable tensor of shape `[4,6]`, 24 parameters total.
Ordering is **rotation first, translation last**: `[wx,wy,wz,vx,vy,vz]`.
Angles are radians; translation tangents are meters. Camera-local axes are
Nerfstudio/OpenGL **+x right, +y up, +z backward**, using column-vector c2w
transforms. The module implements the literal SE(3) exponential:

```text
D = Exp_SE3([omega, v]) = [Exp_SO3(omega), J_left(omega) v; 0 0 0 1]
T_new = T_original @ D
R_new = R_original @ Exp_SO3(omega)
t_new = t_original + R_original @ J_left(omega) @ v
```

`v` is an SE(3) tangent, not the final decoupled translation. For omega=0,
`J_left=I`, so v is the actual local offset. `so3_exp` and `se3_exp` are public
torch functions, differentiable at zero through protected Taylor expansions.
`correct()` returns a tensor and preserves `[...,3,4]` or `[...,4,4]` shape;
it does not mutate Cameras, metadata, input poses, images, or Gaussian means.
Metric c2w and mean coordinates must already share the existing world frame;
keep NS parser `auto_scale_poses=False`, `scale_factor=1`,
`orientation_method='none'`, and `center_method='none'`. Intrinsics/resolution
continue to follow the existing pixel convention. No normalized scene units
or world-frame reorientation are introduced by this module.

## Physical IDs and timestamp sharing

Default `physical_camera_id_offset=2` maps rows to raw sensors `[2,3,4,5]`.
Explicit `physical_camera_id_offset=0` maps compact `[0,1,2,3]` to those SAME
four raw sensors in order. These compact IDs are **not** FAST model indices:
source evidence maps raw 2/3/4/5 to FAST 1/0/2/3. IDs 2/3 overlap the two
conventions, so detection or mixing is impossible and intentionally absent.

Supply integer IDs: scalar, matching/broadcastable camera batch shape, or
NS metadata's additional trailing singleton dimension. Explicit IDs override
metadata. `cam_idx`, dataset frame position, timestamp, and `view_id` are never
interpreted as physical sensor IDs. Missing and out-of-rig integer IDs produce
unchanged poses, including arbitrary viewer/new cameras. Malformed IDs raise.
The same delta row is applied at every timestamp, in both train and eval modes.

Main must resolve each training sample's physical ID by matching the **actual
selected training filename** to its declared transforms frame, including
NS sorting, split selection, and `images_N` downscale redirection. The existing
LiDAR manager uses this filename matching pattern in `_initialize_lidar`.
Parse the frame's source `camera_id` (for example `camera_2`) via an explicit
source mapping and add `camera.metadata['physical_camera_id']` as an integer.
Do not use the training subset's `image_idx`/`cam_idx` as a rig row.

Stock `FullImageDatamanager.next_eval_image` does not attach physical IDs.
Main must likewise resolve the **actual eval filename** against its held-out
frame and pass an explicit physical ID, or attach that metadata before render.
Apply the frozen training-learned rig to identified dev/test frames. Do not
optimize deltas, appearance, or Gaussian parameters using those images.
`rig.eval()` does not disable corrections or freeze parameters by itself:
main must use `no_grad()` and prevent optimizer steps on held-out images.

## Integration into the source-supported differentiable path

The installed versions were read: torch **2.4.0+cu124**, gsplat **1.4.0**,
Nerfstudio **1.1.5**. The 3D CUDA wrapper's `_FullyFusedProjection.backward`
requests `viewmats_requires_grad` and returns `v_viewmats`; `_WorldToCam`
likewise returns a camera-matrix gradient. This establishes the source
contract, not a new runtime CUDA result. See the pinned
[gsplat 1.4.0 wrapper source](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/cuda/_wrapper.py#L714-L754)
and [3D projection backward](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/cuda/_wrapper.py#L832-L869).

NS's [get_viewmat](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/splatfacto.py#L128-L145)
flips c2w rotation columns by `[1,-1,-1]` and analytically inverts to gsplat's
OpenCV world-to-camera (right/down/forward) convention. Keep the differentiable
chain `rig.correct -> get_viewmat -> gsplat.rendering.rasterization` intact;
no NumPy conversion, detach, or reconstructing corrected c2w with `torch.tensor`.
The current Original2DGS adapter explicitly rejects camera gradients in
`inria_surfel_backend.py` at its `viewmats.requires_grad` guard. Run the bounded
extrinsic subproblem with gsplat **3D**. The unchanged adapter can render
frozen corrected poses after calibration, with its camera tensors detached.

NS [Splatfacto get_outputs](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/splatfacto.py#L529-L547)
uses stock pose optimization only while training. Main therefore needs an
explicit render integration applying the shared rig to both identified training
and held-out sensor views. Leave stock per-view `camera_optimizer.mode='off'`;
register this module as a model child, merge its `rig` group with Gaussian
groups, and configure that group's optimizer separately. This report specifies
those integration points; it does not implement them in the parent files.

During extrinsic updates, temporarily omit stale projected map-depth and
map-alpha targets/terms, which were prepared using the original extrinsics.
Retain world-space metric mean/surface geometry constraints. Use training RGB
and masks for the photometric objective. A depth weight of zero alone does
not avoid the current model's depth-target validation; main must explicitly
bypass the stale target path. After freezing an accepted rig, regenerate
projected depth/support from the unchanged metric reference and corrected
poses before enabling such targets. Scene metric scale, LiDAR reference,
physical mean constraints, trajectory, intrinsics, and timing stay fixed for
this shared extrinsic subproblem. Main selects which appearance/geometry
parameters participate; this module introduces only the four rig corrections.

## Bounds, priors, checkpointing, and source calibration export

`effective_delta()` applies separate radial L2 limits of **12 degrees** and
**0.2 m** per sensor. It is identity inside the trust region. Since
`||J_left(omega)||_2 <= 1`, the tangent translation bound also bounds the
actual local/world camera-center displacement. `project_()` bounds the stored
parameter after steps without replacing its Parameter object. Clipping is
nonsmooth at the boundary; use a conservative separate rig learning rate and
inspect saturation. Trust limits are engineering choices, not measured errors.

`regularization()` returns `rig_rotation_prior` and `rig_translation_prior`:
each is `0.5 * mean_sensor(sum_axis((raw_delta/sigma)**2))`, with isotropic
per-axis sigma **2 degrees** and **0.05 m**. These dimensionless Gaussian
priors act on the unprojected parameter so overshoot still receives an inward
gradient. Main may explicitly weight both in its photometric loss; no loss
weight or learning rate has been validated on the capture. `state_dict()`
stores delta, physical ID mapping, priors, and bounds. Attach the module to
the model before loading a full checkpoint; optimizer moments remain the
main trainer's responsibility. A loaded checkpoint's ID buffer is authoritative.

Frame basis matters when exporting a correction to the source calibration.
The repository's actual source chain is
`T_WV_cv = T_WB @ T_BC_cv @ T_CV_cv`, then dataset export applies
`F=diag(1,-1,-1,1)` to obtain `T_WV_ns = T_WV_cv @ F`.
For the learned center-view NS correction D, let `V=T_CV_cv @ F`. Then:

```text
T_BC_cv_new = T_BC_cv @ V @ D @ inverse(V)
```

This updates the real local lever arm, using the bounded matrix translation
`J_left(omega)v` rather than treating v as a world offset. Do not simply append
D to an OpenCV calibration or swap FAST/raw IDs. Current candidate configs
declare a single `center` view. If another virtual orientation A (mapping that
view's NS frame into the trained center's NS frame) is introduced, main must
apply `D_view=inverse(A) @ D @ A`, still using the same physical-sensor row.
Passing differently oriented virtual poses directly to `correct` would express
the correction in inconsistent axes. No calibration files are written here.

The source/raw-channel evidence and existing experimental optical transfers
are documented in [RIG_SOURCE_EVIDENCE.md](RIG_SOURCE_EVIDENCE.md),
`reports/rig_source_evidence.json`, `geometry.camera_world_pose`, and
`dataset.py`'s OpenCV-to-OpenGL export. This module does not establish a physical
calibration. Only after actual held-out **development** improvement and stable
metric geometry may main derive a source/calibration JSON candidate recording
original provenance, ID/basis mapping, priors, bounds, and learned matrices.
Do not use reserved test images for fitting/selection or change
`verified=false` based solely on this sidecar's tests or a photometric score.

## Actual validation scope

Command (run from the repository root, without installing packages):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/runtime" \
.venv-train/bin/python -B -m unittest discover \
  -s tests -p test_rig_optimizer.py -v
```

**Result: all 15 CPU tests passed in 1.723 s**, including a real NS Cameras
metadata/slicing check and strict ID constructor validation.
Coverage includes identity/shape/homogeneous row and no
mutation; proper SO(3) determinants and an independent `torch.matrix_exp`
oracle for SO(3)/SE(3); zero analytic local generators; central finite
differences through view-matrix/pinhole projection at zero and nonzero delta;
input-pose gradients; NS axes and timestamp sharing; four-sensor and compact
ID dispatch; unknown/viewer no-ops; separate optimizer-group updates; radial
bounds/projection; exact prior values/gradients; in-memory checkpoint restore
of corrections and their ID/prior/bound semantics; float32 tiny-angle finite
backward; and explicit invalid-contract failures. The suite also asserts that
CUDA was never initialized.

The tests exercise algebra and autograd through a source-equivalent analytic
NS view-matrix and pinhole projection; they do not call a gsplat CUDA renderer.
No photometric refinement, held-out scoring, six-floor rendering, or GPU
verification is claimed by this sidecar.

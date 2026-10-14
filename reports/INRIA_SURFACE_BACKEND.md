# Original 2DGS surface backend: installed and validated

Validated locally on 2026-10-06 in `/home/chaoszzss/dataset/rosbag/3D-Gaussian`.
`diff_surfel_rasterization==0.0.1` is installed in `.venv-train`; the original
CUDA source compiled without changes and completed actual GPU forward/backward.
The 64×64 single tilted surfel yields camera-Z depth **2.666666–8.000001 m**
over 2,780 supported pixels. Both expected and median depth match analytical
ray–plane intersections. This validates the renderer interface and local
geometry; it does not establish reconstruction quality for the six-floor garage.

## Exact source, build, and attribution

| Component | Revision/version |
| --- | --- |
| Official renderer | [hbb1/diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization/tree/e0ed0207b3e0669960cfad70852200a4a5847f61), `e0ed0207b3e0669960cfad70852200a4a5847f61` |
| GLM submodule | [g-truc/glm](https://github.com/g-truc/glm/tree/5c46b9c07008ae65cb81ab79cd677ecc1934b903), `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |
| Local source | `runtime/inria2dgs_src` |
| Installed distribution | `diff_surfel_rasterization==0.0.1` |
| Python / Torch / CUDA runtime | Python 3.10 / Torch `2.4.0+cu124` / CUDA 12.4 |
| Nerfstudio / gsplat | `1.1.5` / `1.4.0`, unchanged |
| CUDA compiler | `/usr/local/cuda-12.8/bin/nvcc`, `12.8.61` |
| C++ compiler / ABI | GCC `9.4.0`, `_GLIBCXX_USE_CXX11_ABI=0` |
| Build limits | `MAX_JOBS=2`, `TORCH_CUDA_ARCH_LIST=8.9` |
| Test device | NVIDIA GeForce RTX 4060 Laptop GPU |

Wheel: `runtime/inria2dgs_wheels/diff_surfel_rasterization-0.0.1-cp310-cp310-linux_x86_64.whl`.
SHA256: `495912381146ef4904078a6911a3acba488ef668655edd9fd8921abe09fddfc0`.
Installed extension: `.venv-train/lib/python3.10/site-packages/diff_surfel_rasterization/_C.cpython-310-x86_64-linux-gnu.so`.
SHA256: `d3b9040e301cb88f84f9c62e3a571a7295fe4a9e95526e37d97cd34f18e53603`.
Detailed source hashes and build settings: [provenance](../runtime/inria2dgs_provenance.json).

The original source identifies Inria's GRAPHDECO group and retains its copyright
notices. Its [Inria/MPII license](https://github.com/hbb1/diff-surfel-rasterization/blob/e0ed0207b3e0669960cfad70852200a4a5847f61/LICENSE.md)
permits non-commercial research/evaluation and requires preserving the license
and notices when redistributing. The complete license remains at
`runtime/inria2dgs_src/LICENSE.md` and in the installed package's `.dist-info/LICENSE.md`;
their hashes match: `cd5c95b3cfff3acc1bd412420c770f88809331c3db6872df11a970147aa8e81f`.
GLM's license remains at `runtime/inria2dgs_src/third_party/glm/copying.txt`.
The upstream README requests attribution to Huang, Yu, Chen, Geiger, and Gao,
*2D Gaussian Splatting for Geometrically Accurate Radiance Fields*, SIGGRAPH 2024,
DOI `10.1145/3641519.3657428`, and Kerbl, Kopanas, Leimkühler, and Drettakis,
*3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 42(4), 2023.
[Upstream citation metadata](https://github.com/hbb1/diff-surfel-rasterization/blob/e0ed0207b3e0669960cfad70852200a4a5847f61/README.md).

## Native API and depth semantics

The installed Python API is
`GaussianRasterizer(GaussianRasterizationSettings(...))`, called with
`means3D`, `means2D`, `opacities`, `scales`, `rotations`, and exactly one of
`shs`/`colors_precomp`. It returns `(rgb_CHW, radii_N, allmap_7HW)`.
`means2D` supplies a backward/densification signal and does not change forward
projection; the kernel derives its actual projection from world geometry.
[Pinned Python API](https://github.com/hbb1/diff-surfel-rasterization/blob/e0ed0207b3e0669960cfad70852200a4a5847f61/diff_surfel_rasterization/__init__.py).

For pixel `(x,y)`, the kernel intersects the two local homogeneous planes
`k=x*Tw-Tu`, `l=y*Tw-Tv`. With `p=cross(k,l)`, the local intersection is
`s=(p.x/p.z,p.y/p.z)`. The kernel's exact depth expression is:

```cpp
float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
```

This is `dot((s.x,s.y,1),Tw)`: the intersection's CV camera Z in the original
world units, rather than Euclidean ray length. Depth sorting still uses
surfel-centre Z; per-pixel evaluation computes the intersection depth. Filtering,
sorting and transmittance compositing remain the upstream algorithms.
[Pinned forward kernel](https://github.com/hbb1/diff-surfel-rasterization/blob/e0ed0207b3e0669960cfad70852200a4a5847f61/cuda_rasterizer/forward.cu).

| Native allmap channel | Meaning | Adapter treatment |
| --- | --- | --- |
| 0 | Sum of intersection Z × compositing weight | Divide by channel 1 once for expected Z |
| 1 | Accumulated alpha | Preserve |
| 2–4 | Alpha-weighted camera normal | Rotate to world, expose weighted and unit versions |
| 5 | Native median depth | Preserve; **never divide by alpha** |
| 6 | Native depth distortion | Preserve |

Native median follows upstream's `T>0.5` contributor rule; it is a discrete
transmittance-threshold statistic. With a single contributing surfel, it returns
that ray's intersection Z even where total alpha is below 0.5. Expected and
median gradients flow through native autograd. Empty/zero-alpha output is finite
zero depth; the adapter also preserves the requested empty-scene background.

## Adapter contract

[Implementation](../src/orbvi_gaussian/inria_surfel_backend.py):

```python
render_surfel(means, quats, scales, opacities, colors,
              viewmats, Ks, width, height,
              sh_degree=None, backgrounds=None,
              pixel_center=0.5, means2d=None) -> SurfelRender
```

All native inputs are CUDA float32 on the same device. Shapes: world means
`[N,3]` in metres; quaternions `[N,4]` in **wxyz** order; activated tangent scales
`[N,2]` or `[N,3]`; activated opacity `[N]` or `[N,1]`; RGB `[N,3]`, or SH
`[N,K,3]` with a degree in 0–3. The third Gaussian scale is ignored, so its
gradient is zero. No scene rescaling is introduced.

`viewmats=[C,4,4]` are conventional column-vector world-to-CV-camera matrices,
with right/down/forward axes and **positive camera Z**. A view matrix from NS
`get_viewmat` is already in this convention: pass it directly. The renderer
only transposes/contiguates it for the original API's storage layout; there is
no second camera-axis flip. The separate helper
`nerfstudio_c2w_to_cv_w2c([C,3,4] or [C,4,4])` converts a raw NS/OpenGL c2w pose
when that conversion has not already happened.

`Ks=[C,3,3]` supports positive focal lengths, arbitrary principal points,
zero skew, and undistorted pinhole images. Default `pixel_center=.5` means rays
`((x+.5-cx)/fx,(y+.5-cy)/fy,1)`, matching NS intrinsics. For integer-centre CV K,
set `pixel_center=0`. Internally the projection principal-point terms compensate
for the original `ndc2pix` offset `(W-1)/2,(H-1)/2`; homogeneous W is positive CV Z.
Full projection storage is `(P @ w2c).T.contiguous()`.

Returned fields: `rgb`, `alpha`, `depth_expected`, `depth_median`,
`normals_world` (unit), `normals_world_weighted`, `distortion`, `allmap`, `info`.
Images are `[C,H,W,D]`; native allmap is `[C,7,H,W]`. Native two-sided surfel
normals face the camera. World row-vector normals are `n_camera @ R_w2c`.

`rasterization_inria_surface(...)` returns `(render, alpha, info)` for NS parent
integration, with `render_mode="RGB"` or `"RGB+ED"` and
`depth_mode="expected"` or `"median"`. In RGB+ED mode channel 3 already contains
the chosen **normalized Z**. Parent code must not divide it by alpha again.
For median, the historical RGB+ED interface name is retained but the selected
depth statistic is explicitly median. Default background is black; NS can add
its usual `(1-alpha)*background`, or pass a background directly, once.

## Densification gradient and parent integration

`info["means2d"]` and `info["gradient_2dgs"]` alias the same `[C,N,2] tensor.
If no external proxy is supplied, the adapter creates a leaf when grad recording
is enabled. This tensor is a gradient proxy, not projected coordinates.
`info["native_means2d"]` retains the original `[N,3] input tensors and their
native gradients for each camera. `radii` has shape `[C,N]`;
`width`, `height`, `n_cameras`, and `gaussian_ids=None` complete the unpacked
gsplat strategy contract.

Upstream backward multiplies its densification signal by `W/2,H/2` to report
NDC gradients. A differentiable padding/scaling bridge divides that signal by
`W/2,H/2` before it reaches the adapter's pixel proxy. This lets gsplat 1.4
`DefaultStrategy` apply its own normalization exactly once. No absolute-gradient
attribute is fabricated. [Pinned upstream backward kernel](https://github.com/hbb1/diff-surfel-rasterization/blob/e0ed0207b3e0669960cfad70852200a4a5847f61/cuda_rasterizer/backward.cu).

```python
from orbvi_gaussian.inria_surfel_backend import rasterization_inria_surface

strategy.absgrad = False
strategy.key_for_gradient = "means2d"  # "gradient_2dgs" is the same proxy
render, alpha, info = rasterization_inria_surface(
    means=means, quats=quats, scales=scales.exp(),
    opacities=opacities.sigmoid().reshape(-1), colors=colors,
    viewmats=cv_w2c, Ks=ns_K, width=W, height=H,
    sh_degree=active_sh_degree, render_mode="RGB+ED", depth_mode="median",
    packed=False, absgrad=False, sparse_grad=False, rasterize_mode="classic",
)
strategy.step_pre_backward(params, optimizers, state, step, info)
loss.backward()
strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
```

Metadata also exposes `normals_rend` (weighted world normal), `normals_world`
(unit world normal), and both `render_distloss`/`render_distort` aliases for
native distortion. It does not generate a depth-derived `normals_surf` map.
The parent's requested approximately 500K Gaussian cap belongs in its strategy
growth logic; it is not implemented by this bounded renderer sidecar. A correct
gradient key/scale does not itself impose a population or memory cap.

The native kernel culls surfel centres at `Z<=0.2m` and intersections below
`0.2m`; its distortion constants are near `0.2m`, far `100m`. The far constant
is not a hard visibility cutoff. Arbitrary clip planes are **not accepted** by
this adapter, so the parent must explicitly remove its stock `near_plane=.01`
and `far_plane=1e10` arguments for this backend. The kernel's own low-pass
filter remains active; optional gsplat antialiased mode, packed/sparse gradients
and absgrad are unsupported and rejected. Native camera, intrinsic, and
background gradients are unsupported and rejected. Parent camera optimization
must be disabled or handled separately. No training model was modified.

## Actual GPU evidence

Machine-readable evidence: [runtime/inria2dgs_preflight.json](../runtime/inria2dgs_preflight.json).
Build log: [runtime/inria2dgs_build.log](../runtime/inria2dgs_build.log).
Probe log: [runtime/inria2dgs_preflight.log](../runtime/inria2dgs_preflight.log).

Single primitive: centre `(0,0,5)m`, normal `(.8,0,.6)`, tangent scales `(2,2)m`,
opacity `.8`, K with fx=fy=48, cx=cy=32, NS half-pixel centres. Analytical depth:
`Z=3/(.8*(x+.5-32)/48+.6)`.

| Probe | Pixels | Expected max error (m) | Median max error (m) |
| --- | ---: | ---: | ---: |
| Single tilted surfel, identity camera | 2780 | 1.91e-6 | 9.54e-7 |
| Rotated/translated camera, off-centre K | 2896 | 3.34e-6 | 1.91e-6 |
| Two-camera batch, different K | 5703 | 2.86e-6 | 9.54e-7 |
| Integer-centre CV K | 2751 | 2.86e-6 | 9.54e-7 |

At row 32, columns 24/32/40, both expected and median match the analytic
values `6.315789 / 4.931507 / 4.044944m`. World normal maximum component error
is `1.19e-7`, including the rotated-camera probe.

Both expected and median depth-only backward tests produce finite, nonzero
mean/quaternion gradients. Mean-Z autograd `-0.509018` agrees with finite
difference `-0.508942` (epsilon `.001m`). The composite RGB/alpha/depth loss
also yields nonzero finite tangent-scale, opacity, and colour gradients.
The strategy's pre/post-backward bookkeeping records one visible Gaussian and
finite nonzero grad2d; no optimization or Gaussian growth is run. Degree-zero
SH matches precomputed RGB, and zero-opacity/empty outputs remain correct.

Peak Torch allocation **9,809,920 bytes (9.36 MiB)**; peak reserved
**23,068,672 bytes (22 MiB)**, below the requested 50 MiB tensor-memory budget.
These are Torch allocator statistics, excluding CUDA context/library overhead.
Existing package versions are unchanged; only `diff_surfel_rasterization` was
added. The existing `runtime/cache/torch_extensions/gsplat_cuda/gsplat_cuda.so`
SHA256 remains `b6433feeb66f7271b3dc7a20d23d04cda65ec5a7ba2025f071e51c3deff40461`.
The probe did not import the gsplat CUDA backend or invoke its kernels.

## Reproduce inside ROOT

The extension is already installed. Rerun the bounded probe:

```bash
cd /home/chaoszzss/dataset/rosbag/3D-Gaussian
eval "$(bash scripts/bootstrap_training.sh --print-env)"
.venv-train/bin/python scripts/preflight_inria_surface.py
```

The probe sets its cache/temp environment inside ROOT before importing Torch,
rejects output paths outside ROOT, and checks package/gsplat-library isolation.
It never starts training. Source, GLM, the wheel, caches, temporary build files,
and reports remain inside ROOT. No global package, driver, or publication action
was performed.

The recorded build used the project's existing environment with the following
additional settings; pip used **no dependency resolution or build isolation**:

```bash
export TMPDIR="$PWD/runtime/inria2dgs_tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TORCH_EXTENSIONS_DIR="$PWD/runtime/cache/inria2dgs/torch_extensions"
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
.venv-train/bin/python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir runtime/inria2dgs_wheels runtime/inria2dgs_src
.venv-train/bin/python -m pip install --no-deps --no-index \
  runtime/inria2dgs_wheels/diff_surfel_rasterization-0.0.1-cp310-cp310-linux_x86_64.whl
```

Upstream clone and its GLM submodule are checked out at the exact revisions
above. `runtime/inria2dgs_provenance.json` confirms no tracked source changes.

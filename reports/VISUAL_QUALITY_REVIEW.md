# First-block visual quality review

**Decision: block_000 is recognizable, but does not meet the requested high-fidelity real six-floor environment.** Missing arrows, degraded parking text, merged pipes and floor haze are fidelity failures. At the 2026-10-06 21:06 +08:00 log snapshot, `runtime/garage_inria_training.log` recorded blocks 000/001 exported and block 002 next; the batch remains active. `CURRENT_RECONSTRUCTION_STATE.json` describes an earlier phase. This audit does not accept other blocks or floors.

Only this report and [visual_quality_annotations.json](visual_quality_annotations.json) were written. No GPU, training, dependency/source/calibration changes, new scene generation, or external-original writes were performed.

## Fixed evidence and scope

Before opening images, freeze **eight** Original2DGS GT/render pairs: middle dev group `1720267900750125641` (008–011, camera2/3/4/5), and last test group `1720267912749981248` (012–015, camera2/3/4/5). Keep all pairs regardless of quality. This balances sensors and samples middle/later viewpoints, not the entire scene. Test serves final failure documentation only; the next candidate follows dev, RGB baseline and input evidence.

Visually inspected with `view_image` at original resolution: eight comparisons, four corresponding `stock_refined_global_7500/eval_dev/008–011` comparisons, and four original fisheye photos `data/keyframes_2hz/images/camera_{2,3,4,5}/000397_1720267900750125641.jpg`. Comparisons show **GT | render | absolute error**, each panel 640×540. Black lower cutouts are shared validity masks, not holes. Raw 1280×1080 photos confirm input arrows, lettering, pipe separations and ego vehicle. Scores assess losses relative to rectified GT.

| Existing evaluation | Frames | PSNR dB | Valid SSIM |
| --- | ---: | ---: | ---: |
| Original2DGS dev | 20 | 24.166 | 0.7768 |
| Original2DGS test | 16 | 22.133 | 0.7298 |
| RGB-only stock dev | 20 | 24.652 | 0.7794 |

Source: the three run `metrics.json` files. Test is lower by 2.033 dB / 0.0469 SSIM. All-frame camera3 is weakest: dev 22.274/0.7541, test 21.567/0.7105. This is a cohort diagnostic, not a causal calibration measurement.

## Visual scores and defects

Rubric: **3** close to GT, **2** recognizable/detail loss, **1** major degradation, **0** erased/destructive mismatch; **—** not assessable. Subjective ordinal scores, not geometry accuracy. F = fine lines/parking text; G = pillar/roof/pipes; C = colours/materials; L = light/glare; A = contours/completeness; E = ego exclusion from a static environment.

| Pair | Camera | F | G | C | L | A | E | Concrete observation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [dev 008](../runs/inria_surface_block000_12000/eval_dev/008_comparison.jpg) | 2 | 2 | 2 | 2 | — | 2 | — | Service doors/wall retained; small fittings soften and overhead pipes merge. Parking text outside crop. |
| [dev 009](../runs/inria_surface_block000_12000/eval_dev/009_comparison.jpg) | 3 | 1 | 2 | 1 | 1 | 1 | 0 | Clear GT left-turn arrow disappears; road becomes broad grey haze; roof edges smear; hood region becomes streaks. |
| [dev 010](../runs/inria_surface_block000_12000/eval_dev/010_comparison.jpg) | 4 | 1 | 2 | 1 | 2 | 1 | 0 | Small floor arrow loses shape; ceiling pipes fuse; hood/road boundary becomes dark longitudinal texture. |
| [dev 011](../runs/inria_surface_block000_12000/eval_dev/011_comparison.jpg) | 5 | 1 | 2 | 2 | 1 | 1 | — | Main pillar and parked-car silhouette survive; left floor arrow/centre line are lost or displaced into isolated streaks; parking text erodes. |
| [test 012](../runs/inria_surface_block000_12000/eval_test/012_comparison.jpg) | 2 | 1 | 2 | 1 | 1 | 1 | — | GT parking number becomes smeared fragments; charging-pillar edges broaden; bright lamp gains diffuse haze. |
| [test 013](../runs/inria_surface_block000_12000/eval_test/013_comparison.jpg) | 3 | 1 | 1 | 0 | 0 | 0 | 0 | Large opaque-looking road/wall haze; broken overhead pipe contours; GT pink-white wall illumination loses shape; hood replaced by fog. |
| [test 014](../runs/inria_surface_block000_12000/eval_test/014_comparison.jpg) | 4 | 1 | 2 | 1 | 1 | 1 | 0 | Foreground road dash disappears; lane edges/roof pipes smear; hood area becomes road streaking. |
| [test 015](../runs/inria_surface_block000_12000/eval_test/015_comparison.jpg) | 5 | 1 | 2 | 1 | 1 | 1 | — | Foreground pillar remains plausible; right-side parking lines, reflections and far objects dissolve into streaks/haze. |

Wall/ceiling colours remain identifiable; floor gloss and reflections lose structure. GT/raw white-pink lamps are already saturated; added spreading and missing illumination shapes are renderer failures. Thickened/duplicated contours and erased features do not establish physical mesh holes. Still images and parked cars do not establish moving-actor leakage.

**RGB-only:** dev 008 has rougher roof/door edges; dev 009 also erases the arrow but retains more far-wall illumination. Dev 010 retains an arrow fragment and hood smear. Dev 011 preserves its left arrow/centre line substantially better. Shared failures implicate registration/masking/coverage; dev 011 also reveals an appearance tradeoff. Different iterations, initialization/depth preparation and pixel conventions prevent a renderer-only causal comparison.

## What the evidence supports

- **Confirmed contamination:** camera3's dev mask is entirely valid; camera3/4 hood ROI `[100,450,430,490]` is 100% valid. Render streaks coincide with this region. Masks feed initial colouring (`colors.py`) and LiDAR targets (`lidar_splatfacto.py`); exclusion must also cover RGB training. Ego causation of every haze pixel remains unproven.
- **Alignment candidate:** [RIG_SOURCE_EVIDENCE.md](RIG_SOURCE_EVIDENCE.md) documents board/FAST optical-version differences. Used `direct_lio_refined.experimental.yaml` remains `verified=false`, with rotation–principal-point correlations 0.988–0.997 and all cameras locally ill-conditioned. Microrefinement/cheirality do not establish absolute alignment; `mapping_cheirality_dev.json` diagonal values span 98.95–99.88%, not uniformly 99.8%. Exposure, specularity, motion blur and uneven coverage remain unisolated alternatives.
- **Geometry boundary:** [INRIA_SURFACE_BACKEND.md](INRIA_SURFACE_BACKEND.md) reports true ray–plane depth, worst tiny-plane error 3.34e-6 m. Actual centre-to-map NN is median **0.03336 m**, p95 **2.10691 m**; fixed dev image-depth median errors are **2.21–4.24 m**. Centres do not certify footprints/first-visible surfaces. Initialization/supervision share the map: no independent target GT. The current visibility-fix artifact already rejects unreliable near occluders; the older diagnostic's filtering bug is not evidence of an unfixed current bug.

## Strongest next candidate for the coordinator

**Hypothesis:** residual camera-to-map alignment of the mixed optical candidate averages static arrows, pillar edges and pipes across views; valid ego pixels amplify foreground floor haze. Correct alignment on uncontaminated training evidence should restore these features on fixed dev views without moving the measured world geometry.

Correct raw ego exclusion from training photos; regenerate masks/initial colours/targets. Evaluate one bounded **shared four-camera rig correction** on training-only static edges/patches: floor paint, near/far pillars and roof pipes, excluding saturated/specular patches and vehicles. Fix trajectory, metric map and geometry during alignment. Validate the JSON's dev 008–011 ROIs and check other dev groups for regressions. Test stays final evidence.

Proposed falsifiable dev gate: **≥25% lower median static-edge reprojection discrepancy; recover dev 009's arrow and dev 011's arrow/centre-line continuity; no camera2/4 pillar/pipe regression.** Compare mask-only versus mask+rig on identical ROIs. No edge improvement, bound saturation, or hood-only improvement rejects this rig hypothesis. These are proposed criteria, not outcomes.

Risks in [RIG_OPTIMIZER_INTEGRATION.md](RIG_OPTIMIZER_INTEGRATION.md): Original2DGS rejects camera gradients; calibrate through the supported 3D differentiable path, then freeze. Resolve physical sensors by filename/source ID; apply corrections in eval and regenerate projected depth/support. Stale depth/alpha targets or per-timestamp fitting can hide errors. Radius-dependent residuals would reject rig-only correction and require an explicit optical-version candidate, retaining provenance and `verified=false`.

Continue the active batch, but withhold high-fidelity/full-map acceptance until actual models for **every floor, ramps and block overlaps** receive fixed real-photo/render review and coverage assessment. This report signs off only the first-block audit.

#!/usr/bin/env python3
"""CPU-only block000 fixed DEV teacher/denominator crosscheck, never a gate."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from orbvi_gaussian.surfels import SurfelDepthPolicy, project_surfel_depth

NAMES = ("floor", "wall", "other", "unknown")
COUNTS = ("denominator", "correct", "missing_alpha", "invalid_depth", "unknown_prediction", "too_near", "too_far")
INPUTS = {}
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve()
    assert p.is_relative_to(ROOT)
    if str(p) not in INPUTS:
        st = p.stat()
        INPUTS[str(p)] = dict(sha256=sha(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return p


def read(path):
    return json.loads(pin(path).read_text())


def bounded():
    assert time.monotonic()-START < 600, "10-minute CPU bound"
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2*1024**2, "2GiB RSS bound"


def key(row):
    return str(row["group_id"]), row["camera_id"]


def mask(path, shape):
    with Image.open(pin(path)) as im:
        a = np.asarray(im.convert("L")) > 0
    assert a.shape == shape
    return a.ravel()


def quantiles(v):
    v = np.asarray(v)
    if not len(v): return dict(count=0)
    assert np.isfinite(v).all()
    return dict(count=len(v), mean=float(v.mean(dtype=np.float64)),
        p05=float(np.quantile(v, .05)), p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95)),
        min=float(v.min()), max=float(v.max()))


def class_counts(labels):
    return {n: int((labels == i).sum()) for i, n in enumerate(NAMES)}


def classify(ref, indices, dist, camera_z, settings):
    normal = ref["normals"][indices].astype(np.float64)
    length = np.linalg.norm(normal, axis=1)
    normal_ok = ref["normal_valid"][indices] & (length > 1e-8)
    matched = dist <= settings["reference_match_max_m"]
    trusted = matched & normal_ok
    nz = abs(normal[:, 2])/np.maximum(length, 1e-8)
    below = camera_z-ref["xyz"][indices, 2]
    horizontal = nz >= settings["floor_abs_normal_z_min"]
    height = (below >= settings["floor_below_camera_min_m"]) & (below <= settings["floor_below_camera_max_m"])
    labels = np.full(len(indices), 3, np.uint8)
    labels[trusted] = 2
    labels[trusted & (nz <= settings["wall_abs_normal_z_max"]) & (abs(below) <= settings["wall_height_radius_m"])] = 1
    labels[trusted & horizontal & height] = 0
    reasons = dict(total=len(indices), correspondence_over_20cm=int((~matched).sum()),
        matched_normal_invalid=int((matched & ~normal_ok).sum()),
        trusted_normal_not_horizontal=int((trusted & ~horizontal).sum()),
        trusted_horizontal_less_than_30cm_below_or_above_camera=int((trusted & horizontal & (below < .3)).sum()),
        trusted_horizontal_more_than_3m_below_camera=int((trusted & horizontal & (below > 3.)).sum()),
        floor=int((labels == 0).sum()))
    assert sum(v for k, v in reasons.items() if k != "total") == len(indices)
    reasons["nonexclusive_height_band_points"] = int(height.sum())
    reasons["nonexclusive_horizontal_points_even_invalid_normals"] = int(horizontal.sum())
    reasons["horizontal_in_height_band_but_normal_invalid"] = int((horizontal & height & ~normal_ok).sum())
    return labels, reasons, below[trusted & horizontal]


def backproject(camera, indices, depth):
    pose = np.asarray(camera["camera_to_worlds_ns"])
    y, x = np.divmod(indices, camera["w"])
    rays = np.column_stack([(x+.5-camera["cx"])/camera["fl_x"], -(y+.5-camera["cy"])/camera["fl_y"], -np.ones(len(indices))])
    return (rays*depth[:, None])@pose[:, :3].T+pose[:, 3]


def teacher_data(row, planrow):
    camera = row["camera"]
    shape = camera["h"], camera["w"]
    assert all(planrow["camera"][k] == camera[k] for k in ("w", "h", "physical_camera_id"))
    assert all(np.allclose(planrow["camera"][k], camera[k], atol=2e-5, rtol=1e-6)
        for k in ("camera_to_worlds_ns", "fl_x", "fl_y", "cx", "cy"))
    depth = np.load(pin(planrow["source_lidar_depth_path"]), allow_pickle=False).ravel()
    assert depth.size == shape[0]*shape[1] and depth.dtype == np.float32
    legacy = mask(planrow["source_mask_path"], shape)
    eligible = legacy & mask(planrow["mask_path"], shape)
    support = mask(planrow["source_lidar_support_path"], shape) & np.isfinite(depth) & (depth > 0)
    return depth, support, legacy, eligible


def back_labels(camera, depth, support, ref, tree, settings):
    ids = np.flatnonzero(support)
    distance, nearest = tree.query(backproject(camera, ids, depth[ids]), workers=3)
    labels, reasons, below = classify(ref, nearest, distance, camera["camera_to_worlds_ns"][2][3], settings)
    dense = np.full(depth.size, 255, np.uint8); dense[ids] = labels
    return dense, reasons, below


def projected_labels(camera, ref, eligible, settings):
    pose = np.asarray(camera["camera_to_worlds_ns"])
    p = (ref["xyz"].astype(np.float64)-pose[:, 3])@pose[:, :3]
    z = -p[:, 2]
    good = np.isfinite(p).all(1) & (z > 0)
    indices = np.flatnonzero(good); p, z = p[good], z[good]
    uv = np.column_stack([camera["fl_x"]*p[:, 0]/z+camera["cx"]-.5, -camera["fl_y"]*p[:, 1]/z+camera["cy"]-.5])
    good = (uv[:, 0] >= -.5) & (uv[:, 0] < camera["w"]-.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < camera["h"]-.5)
    uv, indices, z = uv[good], indices[good], z[good]
    x, y = np.floor(uv+.5).astype(np.int64).T
    pix = y*camera["w"]+x
    order = np.lexsort((indices, z, pix)); sp = pix[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sp))+1] if len(sp) else np.empty(0, int)
    pixels, nearest = sp[starts], indices[order[starts]]
    take = eligible[pixels]; pixels, nearest = pixels[take], nearest[take]
    labels, reasons, below = classify(ref, nearest, np.zeros(len(nearest)), pose[2, 3], settings)
    dense = np.full(eligible.size, 255, np.uint8); dense[pixels] = labels
    return dense, reasons, below


def replay(camera, ref, policy):
    T = np.eye(4); T[:3] = camera["camera_to_worlds_ns"]
    T = T@np.diag([1., -1., -1., 1.])
    K = np.array([[camera["fl_x"], 0., camera["cx"]-.5], [0., camera["fl_y"], camera["cy"]-.5], [0., 0., 1.]])
    diag = {}
    d, s = project_surfel_depth(ref["xyz"], ref["normals"], ref["normal_valid"], T, K, camera["w"], camera["h"], policy=policy, diagnostics=diag)
    return d.ravel(), s.ravel(), diag


def render(metric_path, row):
    p = pin(metric_path.parent/("%03d_float.npz" % row["dataset_index"]))
    with np.load(p, allow_pickle=False) as f:
        a, d = f["alpha"], f["depth"]
    assert a.dtype == d.dtype == np.float32 and a.shape == d.shape == (row["camera"]["h"], row["camera"]["w"], 1)
    return a.ravel(), d.ravel()


class Accumulator:
    def __init__(self):
        self.counts = defaultdict(lambda: dict.fromkeys(COUNTS, 0))
        self.errors = defaultdict(list)

    def add(self, name, labels, support, target, alpha, depth):
        ids = np.flatnonzero(support)
        a, d, t, lab = alpha[ids], depth[ids], target[ids], labels[ids]
        unknown = ~np.isfinite(a) | (a < 0) | (a > 1)
        missing = ~unknown & (a <= .5)
        invalid = ~unknown & ~missing & (~np.isfinite(d) | (d <= 0))
        covered = ~(unknown | missing | invalid)
        error = d-t
        near, far = covered & (error < -.15), covered & (error > .15)
        correct = covered & ~near & ~far
        parts = dict(denominator=np.ones(len(ids), bool), correct=correct, missing_alpha=missing,
            invalid_depth=invalid, unknown_prediction=unknown, too_near=near, too_far=far)
        for class_name, use in [("all", np.ones(len(ids), bool))]+[(n, lab == i) for i, n in enumerate(NAMES)]:
            key = name+"/"+class_name
            for k, v in parts.items(): self.counts[key][k] += int((v & use).sum())
            self.errors[key].append(error[covered & use].astype(np.float32))

    def finish(self):
        result = {}
        for name, c in self.counts.items():
            e = np.concatenate(self.errors[name]); den = c["denominator"]
            result[name] = dict(**c, wrong_depth=c["too_near"]+c["too_far"],
                correct_depth_coverage=c["correct"]/den if den else None,
                hole_rate=1-c["correct"]/den if den else None,
                missing_alpha_rate=c["missing_alpha"]/den if den else None,
                signed_camera_z_error_m=quantiles(e), abs_camera_z_error_m=quantiles(abs(e)))
        return result


def delta_summary(parts):
    e = np.concatenate(parts) if parts else np.empty(0)
    return dict(signed_v3_minus_old_camera_z_m=quantiles(e), absolute_difference_m=quantiles(abs(e)),
        abs_gt_15cm=int((abs(e) > .15).sum()), abs_gt_1m=int((abs(e) > 1).sum()),
        v3_nearer_gt_15cm=int((e < -.15).sum()), v3_farther_gt_15cm=int((e > .15).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/block000_depth_teacher_crosscheck.json")
    args = parser.parse_args()
    pin(Path(__file__))
    gate = read(ROOT/"reports/surface_quality_v2_block000_seed_aa_cap003.json")
    delivery = read(ROOT/"reports/resolution_pilot_v2_depthv3_delivery.json")
    oldplan = read(ROOT/"reports/acceptance_plan_v1.json")
    newplan = read(ROOT/"reports/acceptance_resolution_depthv3_plan.json")
    settings = delivery["classification"]["settings"]
    paths = {n: ROOT/"runs/fidelity_v2"/p/"metrics.json" for n, p in dict(
        candidate640="block000_seed_aa_cap003_acceptance", candidate1280="block000_seed_aa_cap003_resolution",
        original640="original000_refreshed_v2", original1280="original000_resolution_depthv3",
        gate_baseline640="old_all28_block000_refreshed_v2").items()}
    metrics = {n: read(p) for n, p in paths.items()}
    for m in metrics.values(): assert m["split"] == "dev" and len(m["frames"]) == 20 and m["metric_domain"] == "float_renderer"
    keys = [key(r) for r in metrics["candidate640"]["frames"]]
    assert len(set(keys)) == 20 and all([key(r) for r in m["frames"]] == keys for m in metrics.values())
    assert metrics["candidate640"]["checkpoint_sha256"] == metrics["candidate1280"]["checkpoint_sha256"]
    assert metrics["original640"]["source_ply_sha256"] == metrics["original1280"]["source_ply_sha256"]
    oldrows, newrows = {key(r): r for r in oldplan["frames"]}, {key(r): r for r in newplan["frames"]}
    oldref = ROOT/"data/garage_surface_v1/block_000/lidar_reference.npz"
    newref = ROOT/"data/garage_fidelity_depthv3/block_000/lidar_reference.npz"
    assert sha(pin(oldref)) == sha(pin(newref)), "Teachers must share exact measured reference"
    with np.load(oldref, allow_pickle=False) as f: ref = {k: f[k] for k in ("xyz", "normals", "normal_valid")}
    assert len(ref["xyz"]) == 400000
    tree = cKDTree(ref["xyz"])
    prep = read(newref.with_name("preparation.json")); policy = SurfelDepthPolicy(**prep["teacher_provenance"]["policy"])
    for file in ("src/orbvi_gaussian/surfels.py", "src/orbvi_gaussian/points.py"): pin(ROOT/file)
    accum = Accumulator(); frames = []
    counts = defaultdict(int); reasons_total = defaultdict(lambda: defaultdict(int)); heights = defaultdict(list)
    bridges = defaultdict(lambda: defaultdict(int))
    differences = defaultdict(list); switches = defaultdict(lambda: defaultdict(int))
    for fi, k in enumerate(keys):
        bounded()
        rows = {n: m["frames"][fi] for n, m in metrics.items()}
        c0, c1 = rows["candidate640"]["camera"], rows["candidate1280"]["camera"]
        assert all(rows[n]["camera"] == c0 for n in ("original640", "gate_baseline640"))
        assert rows["original1280"]["camera"] == c1
        assert c0["camera_to_worlds_ns"] == c1["camera_to_worlds_ns"]
        assert all(c1[n] == 2*c0[n] for n in ("w", "h", "fl_x", "fl_y", "cx", "cy"))
        d0, s0, legacy0, eligible0 = teacher_data(rows["candidate640"], oldrows[k]); s0 &= eligible0
        d1, s1raw, legacy1, eligible1 = teacher_data(rows["candidate1280"], newrows[k]); s1 = s1raw & eligible1
        raw0, whyraw0, bhraw0 = projected_labels(c0, ref, eligible0, settings)
        raw1, whyraw1, bhraw1 = projected_labels(newrows[k]["camera"], ref, eligible1, settings)
        raw1source, whysource, bhsource = projected_labels(newrows[k]["camera"], ref, legacy1, settings)
        lab0, why0, bh0 = back_labels(c0, d0, s0, ref, tree, settings)
        lab1, why1, bh1 = back_labels(c1, d1, s1, ref, tree, settings)
        # Same-ray normalization isolates teacher policy from resolution. Only in-memory diagnostic.
        dv, sv, diag = replay(c0, ref, policy); sv &= eligible0
        labv, whyv, bhv = back_labels(c0, dv, sv, ref, tree, settings)
        for arm, cam, depth, sup, direct, back in (("old640", c0, d0, s0, raw0, lab0),
                ("strict1280", c1, d1, s1, raw1, lab1)):
            rawfloor = direct == 0
            selected = np.flatnonzero(rawfloor & sup)
            dd, ni = tree.query(backproject(cam, selected, depth[selected]), workers=3)
            _, why_subset, _ = classify(ref, ni, dd, cam["camera_to_worlds_ns"][2][3], settings)
            b = bridges[arm]
            for name, value in dict(direct_projected_floor=int(rawfloor.sum()),
                    direct_floor_without_teacher_depth=int((rawfloor & ~sup).sum()),
                    direct_floor_with_teacher_depth=len(selected),
                    direct_floor_retained_as_backproject_floor=int((rawfloor & (back == 0)).sum()),
                    backproject_floor_entered_from_other_direct_classes=int(((back == 0) & ~rawfloor).sum()),
                    final_backproject_floor=int((back == 0).sum())).items(): b[name] += value
            for name, value in why_subset.items(): b["supported_direct_floor/"+name] += value
        native_replay = None
        if fi == 0:
            replay1, support1, _ = replay(newrows[k]["camera"], ref, policy); support1 &= legacy1
            replay1 = np.where(support1, replay1, 0).astype(np.float32)
            assert np.array_equal(support1, s1raw & legacy1) and np.array_equal(replay1, d1)
            native_replay = dict(first_DEV_stored_native1280_depth_and_support_exact=True)
        row_counts = dict(eligible640=int(eligible0.sum()), eligible1280=int(eligible1.sum()), legacy_eligible1280=int(legacy1.sum()),
            old640_supported=int(s0.sum()), strictv3_640_replay_supported=int(sv.sum()), strictv3_1280_supported=int(s1.sum()),
            old640_floor=int((lab0 == 0).sum()), strictv3_640_backproject_floor=int((labv == 0).sum()),
            strictv3_1280_backproject_floor=int((lab1 == 0).sum()),
            old640_raw_projected_floor=int((raw0 == 0).sum()), strict1280_raw_projected_floor=int((raw1 == 0).sum()),
            strict1280_raw_floor_supported=int(((raw1 == 0) & s1).sum()),
            historical1280_source_mask_raw_floor=int((raw1source == 0).sum()),
            historical1280_source_mask_raw_floor_supported=int(((raw1source == 0) & s1raw & legacy1).sum()),
            old640_raw_measured_eligible=int((raw0 != 255).sum()), strict1280_raw_measured_eligible=int((raw1 != 255).sum()),
            strict1280_unobserved_eligible=int((eligible1 & (raw1 == 255)).sum()),
            overlap_same640=int((s0 & sv).sum()), only_old640=int((s0 & ~sv).sum()), only_v3_640=int((sv & ~s0).sum()),
            old_floor_same640_overlap=int(((lab0 == 0) & sv).sum()), old_floor_without_strict640_support=int(((lab0 == 0) & ~sv).sum()))
        for name, value in row_counts.items(): counts[name] += value
        reasons = dict(old640_frozen_backprojection=why0, old640_raw_projected=whyraw0,
            strict1280_stored_backprojection=why1, strict1280_raw_projected=whyraw1,
            strict1280_historical_source_mask_raw_projected=whysource, strict640_replay_backprojection=whyv)
        for name, reason in reasons.items():
            for reason_name, value in reason.items(): reasons_total[name][reason_name] += value
        for name, values in dict(old640_frozen_backprojection=bh0, old640_raw_projected=bhraw0,
                strict1280_stored_backprojection=bh1, strict1280_raw_projected=bhraw1, strict640_replay_backprojection=bhv).items(): heights[name].append(values)
        allrender = {n: render(paths[n], r) for n, r in rows.items()}
        for name in ("candidate640", "original640", "gate_baseline640"):
            a, d = allrender[name]
            accum.add("A_old640_frozen/"+name, lab0, s0, d0, a, d)
            accum.add("C_strict640_same_ray_replay/"+name, labv, sv, dv, a, d)
            accum.add("D_same640_overlap_old_teacher/"+name, lab0, s0 & sv, d0, a, d)
            accum.add("D_same640_overlap_v3_teacher_old_labels/"+name, lab0, s0 & sv, dv, a, d)
            covered = np.isfinite(a) & (a > .5) & (a <= 1) & np.isfinite(d) & (d > 0)
            for group, use in (("all", s0 & sv), ("old_floor", (lab0 == 0) & sv), ("old_wall", (lab0 == 1) & sv)):
                use &= covered
                correct_old, correct_v3 = abs(d-d0) <= .15, abs(d-dv) <= .15
                for cls, choose in dict(both_correct=correct_old & correct_v3, old_only_correct=correct_old & ~correct_v3,
                        v3_only_correct=~correct_old & correct_v3, neither_correct=~correct_old & ~correct_v3).items():
                    switches[name+"/"+group][cls] += int((use & choose).sum())
        for name in ("candidate1280", "original1280"):
            a, d = allrender[name]
            accum.add("B_strict1280_stored/"+name, lab1, s1, d1, a, d)
            accum.add("B_input_diagnostic1280_direct_labels/"+name, raw1, s1 & (raw1 != 255), d1, a, d)
        common = s0 & sv
        for group, use in (("all", common), ("old_floor", common & (lab0 == 0)), ("old_wall", common & (lab0 == 1)),
                ("both_floor", common & (lab0 == 0) & (labv == 0))):
            differences["same640_ray/"+group].append((dv[use]-d0[use]).astype(np.float32))
        transitions = np.zeros((4, 4), dtype=np.int64)
        np.add.at(transitions, (lab0[common], labv[common]), 1)
        # Stored native grids have no identical pixel-centre rays: report all four children, never interpolation.
        oldids = np.flatnonzero(s0); y, x = np.divmod(oldids, c0["w"])
        childids = np.stack([(2*y+dy)*c1["w"]+2*x+dx for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))], axis=1)
        childsupport = s1[childids]
        childdelta = d1[childids]-d0[oldids, None]
        differences["stored1280_four_children_minus_old640/all_supported_pairs"].append(childdelta[childsupport])
        differences["stored1280_four_children_minus_old640/old_floor_supported_pairs"].append(childdelta[childsupport & (lab0[oldids, None] == 0)])
        childcounts = dict(old_supported_parents_any_v3child=int(childsupport.any(1).sum()),
            old_supported_parents_all4_v3children=int(childsupport.all(1).sum()),
            old_floor_parents_any_v3child=int((childsupport.any(1) & (lab0[oldids] == 0)).sum()),
            old_floor_parents_all4_v3children=int((childsupport.all(1) & (lab0[oldids] == 0)).sum()))
        for name, value in childcounts.items(): counts[name] += value
        frames.append(dict(frame_key=list(k), dataset_index=rows["candidate640"]["dataset_index"], camera_z_m=c0["camera_to_worlds_ns"][2][3],
            counts=row_counts, floor_rejection_reasons=reasons, same640_class_transition_matrix_old_to_v3=transitions.tolist(),
            stored1280_child_correspondence=childcounts, strict640_replay_generation_diagnostics=diag, stored1280_replay=native_replay,
            strict1280_raw_normal_trusted_horizontal_below_camera_m=quantiles(bhraw1),
            native_float32_pose_max_abs_difference_from_teacher_pose=float(np.max(abs(np.asarray(c1["camera_to_worlds_ns"])-np.asarray(newrows[k]["camera"]["camera_to_worlds_ns"]))))))
        print(f"DEV {fi+1}/20 {k}: oldfloor={row_counts['old640_floor']} strictfloor={row_counts['strictv3_1280_backproject_floor']}", flush=True)
    results = accum.finish()
    existing = gate["render_measurement"]["summaries"]
    for ours, theirs in (("candidate640", "candidate"), ("gate_baseline640", "baseline")):
        for label in NAMES:
            check = results["A_old640_frozen/"+ours+"/"+label]
            for name in COUNTS:
                assert check[name] == existing[theirs]["coverage"][label][name], (ours, label, name)
    historic = next(x for x in delivery["dev_teacher_floor_support_vs_corrected_legacy"] if x["source_dataset"].endswith("block_000"))
    assert counts["historical1280_source_mask_raw_floor"] == historic["fixed_floor_denominator"] == 1613
    assert counts["historical1280_source_mask_raw_floor_supported"] == historic["v3_teacher_supported"] == 1329
    diff_results = {k: delta_summary(v) for k, v in differences.items()}
    own = results["B_strict1280_stored/candidate1280/floor"]
    original = results["B_strict1280_stored/original1280/floor"]
    bounded()
    for path, info in INPUTS.items():
        st = Path(path).stat(); assert st.st_size == info["bytes"] and st.st_mtime_ns == info["mtime_ns"], "Input changed"
    report = dict(schema_version=1, status="bounded_DEV_teacher_crosscheck_complete", block_id="block_000", frames=frames,
        cpu_only=True, threads=3, gpu_used=False, test_read=False, independent_ground_truth=False,
        original_gate_or_reference_modified=False, evaluator_or_core_modified=False, protocol_settings=settings,
        exact20_DEV_frame_pose_and_scaled_intrinsics_match=True, same_measured_reference_sha256=sha(oldref),
        strictv3_policy=prep["teacher_provenance"]["policy"], historical_8239_percent_definition=historic,
        definitions=dict(A_old640_frozen="Exact old gate support: old depth/support intersect frozen ego+legacy mask, labels from fixed nearest reference within20cm. Existing candidate and all28 baseline coverage counts reproduced exactly; single original is a separate model.",
            B_strict1280_stored="Actual stored strictv3 training teacher at1280, frozen full-resolution ego+source masks; same nearest-reference classifier on supported backprojected depth. Conditional diagnostic, not a replacement gate.",
            B_input_diagnostic1280_direct_labels="Reproduce historical support statistic: nearest camera-z measured point per native pixel, ties by original row, direct match distance0, same normal/height classifier. Teacher availability uses1613 denominator; model depth error uses only supported measured-class pixels (1329 floor before any ego difference). Unsupported teacher is unknown, not a model failure/pass.",
            C_strict640_same_ray_replay="Strictv3 frozen policy replayed only in RAM at exact old640 camera rays; isolates depth-policy changes from resolution. First1280 replay equals stored teacher bytewise. This640 teacher is diagnostic, not a training dataset or gate.",
            D_same640_overlap="Common old/strict support, fixed old labels. Both actual native640 models compared with each teacher on identical rays. Also count which tolerance judgments switch.",
            stored_cross_resolution="All four supported1280 child pixels paired with old640 parent. Rays differ by +/-0.25 old640 pixel in each axis; no interpolation/averaging. These differences mix sampling and teacher policy and are not exact same-ray disagreement.",
            sign="Depth error=render camera-z minus teacher; teacher difference=v3 minus old; negative means nearer.",
            missing="alpha<=.5 is missing model coverage; opaque error>15cm is wrong depth; invalid prediction, unknown labels and unsupported teacher remain explicit.",
            normal_height_rejections="Exclusive cascade: correspondence>20cm, invalid nearest normal, trusted nonhorizontal, trusted horizontal too high (<.3m below), trusted horizontal too low (>3m below), floor. Extra nonexclusive counts are labeled.",
            reference_scope="Shared accumulated LIO/map XYZ and estimated normals, with camera_init+Z assumed up. Neither teacher establishes independent physical ground truth or semantic floor."),
        totals=dict(counts), floor_rejection_reasons={k: dict(v) for k, v in reasons_total.items()},
        floor_denominator_bridges={k: dict(v) for k, v in bridges.items()},
        trusted_horizontal_reference_below_camera_m={k: quantiles(np.concatenate(v)) for k, v in heights.items()},
        teacher_camera_z_discrepancies=diff_results,
        model_depth_results=results, identical_ray_covered_prediction_tolerance_switches={k: dict(v) for k, v in switches.items()},
        native_metric_sources={n: dict(path=str(paths[n]), psnr_db_mean=m["psnr_db_mean"],
            alpha_gt_0_5_pixel_fraction_mean=m["alpha_gt_0_5_pixel_fraction_mean"], protocols=m.get("protocols"),
            checkpoint_sha256=m.get("checkpoint_sha256"), source_ply_sha256=m.get("source_ply_sha256")) for n, m in metrics.items()},
        checks=dict(old_gate_all_class_coverage_counts_exact=True, historical1613_1329_reproduced=True, stored1280_firstDEV_replay_exact=True),
        findings=dict(
            denominator_explanation="82.393% is1329 supported teacher pixels /1613 directly projected measured floor pixels at1280, not image-floor coverage. Old725 is supported old640 teacher pixels classified after nearest-map correspondence. Native1280 backprojection gives1290 floor pixels; direct versus backproject labels are distinct fixed geometric queries.",
            old_floor_fraction_of_mask_eligible_pixels=counts["old640_floor"]/counts["eligible640"],
            v3_direct_floor_fraction_of_mask_eligible_pixels=counts["strict1280_raw_projected_floor"]/counts["eligible1280"],
            v3_measured_pixel_fraction_of_mask_eligible_pixels=counts["strict1280_raw_measured_eligible"]/counts["eligible1280"],
            sparse_floor_explanation="Most supported trusted normals are nonhorizontal, and almost all trusted horizontal matches lie above the camera or less than30cm below it under the unchanged+Z/height proxy. Correspondence failure and invalid nearest normals remove relatively few already-supported teacher pixels. These counts do not identify all physical ground pixels or prove absent ground geometry.",
            same_ray_teacher_conflict="On718 common old-floor640 rays, no old-versus-strictv3 depth difference exceeds15cm and neither model changes its covered floor correct/wrong judgment. Across591223 common rays only107 differences exceed15cm. Teacher-version inconsistency alone is not supported as the main explanation of this pilot's depth failure.",
            native_resolution_caution="Stored1280 child pixels are different rays:17.15% of all supported child-parent pairs and7.03% of old-floor pairs differ by>15cm. This mixes sparse point occupancy, occlusion/layer selection and changed sampling with policy; it cannot be called exact same-ray teacher disagreement.",
            candidate_own_teacher_floor=dict(denominator=own["denominator"], missing_alpha=own["missing_alpha"],
                missing_alpha_rate=own["missing_alpha_rate"], wrong_depth=own["wrong_depth"],
                correct=own["correct"], signed_error_p50_m=own["signed_camera_z_error_m"]["p50"]),
            original_own_teacher_floor=dict(denominator=original["denominator"], missing_alpha=original["missing_alpha"],
                wrong_depth=original["wrong_depth"], correct=original["correct"], signed_error_p50_m=original["signed_camera_z_error_m"]["p50"]),
            decision="Candidate still has both measurable missing alpha and opaque wrong depth against its actual strict1280 teacher. This is not merely a penalty from the old evaluation teacher. These conditional failures do not distinguish optimization/capacity limitations from errors common to the map/calibration/teacher, and are not proof of independent physical holes. Keep the original gate unchanged."),
        limitations=["Support percentages are conditional on sparse measured pixels, never broad image-floor completeness.",
            "A small geometric floor proxy cannot diagnose all physical ground holes or correct calibration/vertical assumptions.",
            "Model errors against its own teacher can show that evaluation mismatch alone is insufficient; they do not establish why optimization failed or prove an independent true surface.",
            "Frozen640 all28 gate baseline is distinct from the original single-block model used for1280 PSNR comparison.",
            "No new training, native rendering, occlusion-mask work, acceptance decision or renderer provenance audit."],
        inputs=INPUTS, elapsed_seconds=time.monotonic()-START, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    with args.output.open("x") as f: json.dump(report, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(dict(counts), indent=2)); print(f"saved={args.output} elapsed={report['elapsed_seconds']:.2f}s RSS={report['peak_rss_mib']:.1f}MiB")


if __name__ == "__main__":
    with threadpool_limits(limits=3): main()

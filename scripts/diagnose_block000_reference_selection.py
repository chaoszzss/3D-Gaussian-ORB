#!/usr/bin/env python3
"""Trace measured block000 reference crop/voxel/cap without changing geometry."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from orbvi_gaussian.points import voxel_downsample

START = time.monotonic()
INPUTS = {}
EDGES = np.array([-1000, -5, -3, -2.8, -2.5, -1.3, -.4, 0, .3, 1, 2, 3, 4, 5, 1000.])


def pin(path):
    p = Path(path).resolve()
    if str(p) not in INPUTS:
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1024**2), b""): digest.update(b)
        st = p.stat()
        INPUTS[str(p)] = dict(sha256=digest.hexdigest(), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return p


def read(path): return json.loads(pin(path).read_text())


def check():
    assert time.monotonic()-START < 600
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2*1024**2


def quant(v):
    if not len(v): return dict(count=0)
    return dict(count=len(v), min=float(np.min(v)), p05=float(np.quantile(v, .05)),
        p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95)), max=float(np.max(v)))


def stats(xyz, camera, normals=None, valid=None):
    delta = xyz.astype(np.float64)-camera
    radius = np.linalg.norm(delta[:, :2], axis=1)
    result = dict(points=len(xyz), bbox_min=xyz.min(0).tolist() if len(xyz) else None,
        bbox_max=xyz.max(0).tolist() if len(xyz) else None,
        absolute_z_histogram=np.histogram(xyz[:, 2], EDGES)[0].tolist())
    for r in (3., 6.):
        use = radius < r
        q = xyz[use]
        band = (q[:, 2] > -3) & (q[:, 2] < -.4)
        below = camera[2]-q[:, 2]
        entry = dict(points=len(q), z_m=quant(q[:, 2]), absolute_z_histogram=np.histogram(q[:, 2], EDGES)[0].tolist(),
            absolute_z_minus3_to_minus0_4=int(band.sum()),
            fixed_relative_height_band_0_3_to_3m_below_camera=int(((below >= .3) & (below <= 3)).sum()),
            absolute_z_lt_minus0_4=int((q[:, 2] < -.4).sum()), absolute_z_lt_minus3=int((q[:, 2] < -3).sum()),
            low_absolute_band_xy_1m_occupied_cells=len(np.unique(np.floor(q[band, :2]).astype(int), axis=0)))
        if normals is not None:
            n = normals[use].astype(float); v = valid[use]
            nz = abs(n[:, 2])/np.maximum(np.linalg.norm(n, axis=1), 1e-8)
            horizontal = nz >= np.cos(np.deg2rad(30))
            entry.update(low_band_normal_abs_z=quant(nz[band]), low_band_normal_valid=int((v & band).sum()),
                low_band_valid_horizontal=int((v & band & horizontal).sum()), low_band_valid_vertical=int((v & band & (nz <= .3)).sum()),
                low_band_invalid_normal=int((band & ~v).sum()),
                trusted_horizontal_z_m=quant(q[v & horizontal, 2]))
        result[f"first_DEV_camera_xy_radius_lt_{int(r)}m"] = entry
    return result


def normal_comparison(pre_xyz, ref, camera):
    # Exactly the same k16 PCA validity equations, but before the400k hash cap.
    use = (np.linalg.norm(ref["xyz"][:, :2]-camera[:2], axis=1) < 6) & (ref["xyz"][:, 2] > -3) & (ref["xyz"][:, 2] < -.4)
    ids = np.flatnonzero(use)
    assert len(ids) <= 20000
    tree = cKDTree(pre_xyz)
    distance, neighbor = tree.query(ref["xyz"][ids], k=16, workers=3)
    cloud = pre_xyz[neighbor].astype(np.float64)
    centered = cloud-cloud.mean(1, keepdims=True)
    eigen, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
    n = vectors[:, :, 0]
    v = (eigen[:, 0] < .25*np.maximum(eigen[:, 1], 1e-12)) & (eigen[:, 1] > .01*np.maximum(eigen[:, 2], 1e-12))
    old = ref["normals"][ids].astype(float); old /= np.maximum(np.linalg.norm(old, axis=1, keepdims=True), 1e-8)
    saved_v = ref["normal_valid"][ids]
    old_horizontal, new_horizontal = abs(old[:, 2]) >= np.cos(np.deg2rad(30)), abs(n[:, 2]) >= np.cos(np.deg2rad(30))
    old_vertical, new_vertical = abs(old[:, 2]) <= .3, abs(n[:, 2]) <= .3
    categories = dict(saved_valid_horizontal=saved_v & old_horizontal, saved_valid_vertical=saved_v & old_vertical,
        uncapped_valid_horizontal=v & new_horizontal, uncapped_valid_vertical=v & new_vertical,
        saved_vertical_becomes_uncapped_horizontal=saved_v & old_vertical & v & new_horizontal,
        saved_nonhorizontal_becomes_uncapped_valid_horizontal=~old_horizontal & v & new_horizontal,
        saved_invalid_becomes_uncapped_valid_horizontal=~saved_v & v & new_horizontal)
    return dict(fixed_reference_indices=ids.tolist(), population="All stored reference points within6m XY and absolute Z(-3,-.4); no model values used",
        saved_abs_normal_z=quant(abs(old[:, 2])), uncapped_abs_normal_z=quant(abs(n[:, 2])),
        uncapped_k16_max_neighbor_distance_m=quant(distance[:, -1]),
        saved_vs_uncapped_abs_normal_dot=quant(abs(np.sum(old*n, axis=1))),
        counts={name: int(value.sum()) for name, value in categories.items()},
        uncapped_horizontal_examples=[dict(reference_index=int(ids[i]), xyz=ref["xyz"][ids[i]].tolist(),
            saved_normal=old[i].tolist(), uncapped_normal=n[i].tolist(), saved_valid=bool(saved_v[i]),
            uncapped_k16_radius_m=float(distance[i, -1])) for i in np.flatnonzero(v & new_horizontal)[:20]],
        limitation="Changing neighborhood density is a diagnostic only. PCA normal agreement is same-map evidence, not independent wall/floor semantics; no stored normal is changed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"reports/block000_reference_selection_audit.json")
    args = p.parse_args()
    pin(Path(__file__))
    prep = read(ROOT/"data/garage_surface_v1/block_000/preparation.json")
    meta = read(ROOT/"runs/fidelity_v2/block000_seed_aa_cap003_acceptance/metrics.json")
    first = meta["frames"][0]; camera = np.asarray(first["camera"]["camera_to_worlds_ns"])[:, 3]
    source = pin(prep["map"]["source"])
    filegate = read(source.with_name("release_file_gate.json"))
    assert INPUTS[str(source)]["sha256"] == filegate["raw_sha256"]
    fusion = read(source.with_name("czy_joint_fusion_report.json"))
    with np.load(pin(ROOT/"data/garage_surface_v1/block_000/lidar_reference.npz"), allow_pickle=False) as f:
        ref = {k: f[k] for k in ("xyz", "normals", "normal_valid")}
    lower, upper = np.asarray(prep["map"]["crop_min"]), np.asarray(prep["map"]["crop_max"])
    voxel = prep["map"]["voxel_m"]
    cap = prep["map"]["depth_reference_points"]
    for file in ("src/orbvi_gaussian/workflow.py", "src/orbvi_gaussian/points.py", "src/orbvi_gaussian/surfels.py"): pin(ROOT/file)
    header = {}; crops, cropids, local6, localids = [], [], [], []
    raw_min, raw_max = np.full(3, np.inf), np.full(3, -np.inf)
    total = finite = in_crop = 0
    reject = {axis+side: 0 for axis in "xyz" for side in ("_below_min", "_above_max")}
    with source.open("rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if not line or line.startswith("#"): continue
            k, *v = line.split(); header[k] = v
            if k == "DATA": break
        offset = f.tell()
        assert header["FIELDS"] == ["x", "y", "z", "rgb"] and header["SIZE"] == ["4"]*4
        assert header["TYPE"] == ["F", "F", "F", "U"] and header["DATA"] == ["binary"]
        count = int(header["POINTS"][0]); dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        assert source.stat().st_size == offset+count*16
        for start in range(0, count, 262144):
            check()
            records = np.fromfile(f, dtype=dtype, count=min(262144, count-start))
            xyz = np.column_stack([records[n] for n in "xyz"])
            ok = np.isfinite(xyz).all(1); total += len(xyz); finite += int(ok.sum())
            if ok.any(): raw_min = np.minimum(raw_min, xyz[ok].min(0)); raw_max = np.maximum(raw_max, xyz[ok].max(0))
            within = ok & (xyz >= lower).all(1) & (xyz <= upper).all(1)
            in_crop += int(within.sum()); crops.append(xyz[within]); cropids.append(start+np.flatnonzero(within))
            nearby = ok & (np.linalg.norm(xyz[:, :2]-camera[:2], axis=1) < 6)
            local6.append(xyz[nearby]); localids.append(start+np.flatnonzero(nearby))
            for axis, letter in enumerate("xyz"):
                reject[letter+"_below_min"] += int((ok & (xyz[:, axis] < lower[axis])).sum())
                reject[letter+"_above_max"] += int((ok & (xyz[:, axis] > upper[axis])).sum())
    xyz = np.concatenate(crops); source_ids = np.concatenate(cropids)
    nearby = np.concatenate(local6); nearby_ids = np.concatenate(localids)
    del crops, cropids, local6, localids
    print(f"Raw scan complete: total={total} crop={in_crop} local6={len(nearby)}", flush=True)
    raw_local = stats(nearby, camera)
    raw_crop = stats(xyz, camera)
    # Integer cells match the reader's float64 calculation exactly.
    cells = np.floor(xyz.astype(np.float64)/voxel).astype(np.int32)
    _, firstindex = np.unique(cells, axis=0, return_index=True)
    del cells
    firstindex.sort()  # Preserve source order, and the exact first actual point per voxel.
    uncapped = xyz[firstindex]
    uncapped_source_ids = source_ids[firstindex]
    uncapped_stats = stats(uncapped, camera)
    print(f"Exact pre-cap occupied voxels={len(uncapped)}; replaying{cap} deterministic hash cap", flush=True)
    selected, _ = voxel_downsample(uncapped, voxel_size=voxel, max_points=cap)
    exact = np.array_equal(selected.astype(np.float32), ref["xyz"])
    assert exact, "Current source/selection does not exactly reproduce stored XYZ: cannot attribute selection losses"
    selected_stats = stats(ref["xyz"], camera, ref["normals"], ref["normal_valid"])
    normal = normal_comparison(uncapped, ref, camera)
    reference_tree = cKDTree(ref["xyz"])
    # Whole local below band: nearest preserved point measures sampling gaps without model outputs.
    low = (nearby[:, 2] > -3) & (nearby[:, 2] < -.4)
    dist, nearest = reference_tree.query(nearby[low], workers=3)
    local_low_examples = []
    for index in np.flatnonzero(low)[np.linspace(0, int(low.sum())-1, min(12, int(low.sum())), dtype=int)] if low.any() else []:
        local_low_examples.append(dict(raw_pcd_row_index=int(nearby_ids[index]), xyz=nearby[index].tolist()))
    perstage = dict(raw_source_within6m_xy_before_crop=raw_local, raw_crop=raw_crop,
        voxel6cm_before400kcap=uncapped_stats, stored_reference_after400kcap=selected_stats)
    retention = {}
    for rad in (3, 6):
        name=f"first_DEV_camera_xy_radius_lt_{rad}m"
        rawcount=raw_local[name]["absolute_z_minus3_to_minus0_4"]
        cropcount=raw_crop[name]["absolute_z_minus3_to_minus0_4"]
        voxelcount=uncapped_stats[name]["absolute_z_minus3_to_minus0_4"]
        savedcount=selected_stats[name]["absolute_z_minus3_to_minus0_4"]
        retention[name]=dict(raw_below_points=rawcount, after_crop_below_points=cropcount,
            before_cap_below_voxels=voxelcount, after_cap_below_reference_points=savedcount,
            crop_removed_below_points=rawcount-cropcount, cap_retained_below_voxel_fraction=savedcount/voxelcount if voxelcount else None)
    check()
    for path, saved in INPUTS.items():
        st=Path(path).stat(); assert st.st_size==saved["bytes"] and st.st_mtime_ns==saved["mtime_ns"]
    report = dict(schema_version=1, status="bounded_reference_selection_trace_complete", cpu_only=True, gpu_used=False,
        threads=3, test_images_read=False, model_outputs_used_for_point_selection=False,
        existing_reference_normals_teacher_gates_worldup_modified=False,
        first_DEV_frame=dict(frame_key=[first["group_id"],first["camera_id"]], camera_xyz=camera.tolist()),
        source_pcd=dict(path=str(source), header=header, bytes=source.stat().st_size, sha256=INPUTS[str(source)]["sha256"],
            release_SHA_matches=True, finite_points=finite, total_points=total, bbox_min=raw_min.tolist(), bbox_max=raw_max.tolist()),
        source_metadata={k:fusion.get(k) for k in ("frame_id","raw_clouds_seen","raw_points_seen","raw_points_written","ground_depth_filter_input_points","ground_depth_filter_accepted_points","sensor_height")},
        metadata_limitations="sensor_height2.65 is a configured prior, not measured mounting height. Semantic ground-filter counters are not evidence that this raw PCD removed those points. Upstream gravity/axis interpretation remains a separate task.",
        construction=dict(crop_min=lower.tolist(), crop_max=upper.tolist(), voxel_m=voxel, reference_cap=cap,
            crop_finite_points=in_crop, pre_cap_occupied_voxels=len(uncapped), cap_retention_fraction=cap/len(uncapped),
            stored_reference_xyz_exact_replay=exact,
            no_z_sign_change_or_rotation_in_reference_selection=True,
            crop_rule="Axis-aligned body trajectory bounds expanded18m XY and5m Z; inclusive bounds. Uses measured XYZ unchanged.",
            voxel_rule="First actual input point per6cm voxel;400k smallest deterministic integer-tuple hash ranks, no Z ordering or ground-removal predicate.",
            normal_rule="k16 PCA only after400k cap, valid if lambda0<.25lambda1 and lambda1>.01lambda2; all points and invalid-normal flags are saved. No XYZ deletion from normals.",
            initializer_rule="Later100k initializer cap and train-color visibility selection occur after dense reference copy; they do not delete lidar_reference XYZ.",
            per_axis_crop_rejections_nonexclusive=reject),
        histogram_absolute_z_edges_m=EDGES.tolist(), stages=perstage, low_band_retention=retention,
        low_raw6m_to_saved_reference_nearest_distance_m=quant(dist), low_raw6m_examples=local_low_examples,
        uncapped_vs_saved_normal_diagnostic=normal,
        interpretation=dict(
            decision_scope="Separates absence from the supplied raw PCD, crop/cap thinning, and neighborhood-dependent normal classification in first-camera XY cylinders. Absolute low band(-3,-.4) matches main's query; configured gate-relative band is reported separately.",
            physical_limits="Absence of returns in this accumulated raw source is not independent proof that a physical floor is absent. Dense positive-Z structure must not be relabeled ground or used as a mounting-height measurement.",
            no_new_protocol="All findings are diagnostics with unchanged+Z and height rules; no reconstruction/training, v3 teacher, gate, or world-frame modification."),
        inputs=INPUTS, elapsed_seconds=time.monotonic()-START, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    with args.output.open("x") as f: json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(retention,indent=2));print(json.dumps(normal["counts"],indent=2))
    print(f"Saved{args.output} elapsed={report['elapsed_seconds']:.2f}s RSS={report['peak_rss_mib']:.1f}MiB")


if __name__ == "__main__":
    with threadpool_limits(limits=3): main()

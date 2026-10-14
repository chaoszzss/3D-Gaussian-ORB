#!/usr/bin/env python3
"""Fixed-ray, measured-reference-only finite patch occlusion decision diagnostic.

No model/backend imports, training, GPU, depth replacement, or gate changes.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "reports/surface_quality_v2_joint015_fixed32_measured_patch_occlusion"
POLICY = dict(radius_min_m=.02, radius_max_m=.05, spacing_neighbors=6,
    spacing_knn_limit=65, duplicate_epsilon_m=1e-5,
    radius_rule="clip(median(first 6 distinct nonzero Euclidean neighbor distances), .02, .05)",
    local_neighborhood_radius_multiple=2., local_neighborhood_max_m=.1,
    local_plane_residual_max_m=.01, normal_agreement_degrees=20.,
    coherent_fraction_min=.75, min_coherent_other_points=3,
    min_tangent_second_singular_value_m=.002, min_incidence_abs_cos=.1,
    teacher_foreground_margin_camera_z_m=.15,
    hull_rule="Ray-plane hit lies inside convex hull of coherent local measured neighbor tangent coordinates, including source center; no hull extrapolation.")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write_new(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def geometry(camera, pixel):
    pose = np.asarray(camera["camera_to_worlds_ns"], dtype=np.float64)
    x, y = pixel
    d = np.linalg.solve(pose[:, :3].T, [(x+.5-camera["cx"])/camera["fl_x"],
        -(y+.5-camera["cy"])/camera["fl_y"], -1.])
    return pose[:, 3], d, d/np.linalg.norm(d)


def local_patch(index, xyz, normals, valid, tree):
    p, n = xyz[index], normals[index]
    dist, ids = tree.query(p, k=POLICY["spacing_knn_limit"], workers=3)
    # Exact overlap between source blocks must not count as denser support.
    distinct, seen = [], set()
    for distance, i in zip(dist, ids):
        key = tuple(np.round(xyz[i]/POLICY["duplicate_epsilon_m"]).astype(np.int64))
        if distance > POLICY["duplicate_epsilon_m"] and key not in seen:
            distinct.append((float(distance), int(i)))
            seen.add(key)
    spacing = float(np.median([d for d, _ in distinct[:6]])) if len(distinct) >= 6 else None
    radius = float(np.clip(spacing, .02, .05)) if spacing is not None else None
    result = dict(point_index=int(index), radius_m=radius, spacing_m=spacing,
        spacing_neighbor_indices=[i for _, i in distinct[:6]], spacing_neighbor_distances_m=[d for d, _ in distinct[:6]],
        neighborhood_indices=[], coherent_other_indices=[], neighborhood_distinct_other_points=0,
        coherent_fraction=None, plane_residual_p95_m=None, coherent_plane_residual_max_m=None,
        tangent_second_singular_value_m=None, local_support_reasons=[])
    if radius is None:
        result["local_support_reasons"].append("insufficient_distinct_spacing_neighbors_within_knn_budget")
        return result, None, None
    ids = tree.query_ball_point(p, 2*radius, workers=3)
    unique, seen = [], {tuple(np.round(p/1e-5).astype(np.int64))}
    for i in sorted(ids):
        key = tuple(np.round(xyz[i]/1e-5).astype(np.int64))
        if key not in seen:
            unique.append(i)
            seen.add(key)
    ids = np.asarray(unique, dtype=np.int64)
    delta = xyz[ids]-p
    residual = np.abs(delta@n)
    coherent = valid[ids] & (np.abs(normals[ids]@n) >= np.cos(np.deg2rad(20))) & (residual <= .01)
    support = ids[coherent]
    result.update(neighborhood_indices=ids.tolist(), coherent_other_indices=support.tolist(),
        neighborhood_distinct_other_points=len(ids),
        coherent_fraction=float(coherent.mean()) if len(ids) else 0.,
        plane_residual_p95_m=float(np.quantile(residual, .95)) if len(ids) else None,
        coherent_plane_residual_max_m=float(residual[coherent].max()) if coherent.any() else None)
    reasons = result["local_support_reasons"]
    if len(support) < 3:
        reasons.append("fewer_than_3_coherent_distinct_other_measured_points")
    if not len(ids) or coherent.mean() < .75:
        reasons.append("local_neighbors_not_75_percent_normal_valid_coherent")
    axis = np.eye(3)[np.argmin(abs(n))]
    u = np.cross(n, axis); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    basis = np.stack([u, v], axis=1)
    xy = np.vstack([np.zeros((1, 2)), (xyz[support]-p)@basis])
    singular = np.linalg.svd(xy-xy.mean(0), compute_uv=False)
    second = float(singular[1]) if len(singular) > 1 else 0.
    result["tangent_second_singular_value_m"] = second
    if second < .002:
        reasons.append("local_support_nearly_collinear")
    hull = None
    if len(xy) >= 3 and second >= .002:
        try:
            hull = ConvexHull(xy).equations
        except QhullError:
            reasons.append("local_tangent_hull_degenerate")
    return result, basis, hull


def point_info(index, ref, xyz, normals, valid, block_order):
    return dict(point_index=int(index), world_xyz=xyz[index].tolist(), normal=normals[index].tolist(),
        normal_valid=bool(valid[index]), source_block_id=block_order[int(ref["source_block_index"][index])],
        source_row_index=int(ref["source_row_index"][index]))


def measured_frame(frame, ref, xyz, normals, valid, tree, cache, block_order):
    camera = frame["camera"]
    pose = np.asarray(camera["camera_to_worlds_ns"])
    result = []
    # Candidate collection only examines fixed camera, fixed teacher and measured xyz/normals.
    for ray in frame["rays"]:
        center, direction, unit = geometry(camera, ray["pixel_xy"])
        target = ray["reference_camera_z_m"]
        candidates, nearest, nearest_trusted = [], None, None
        same640, same1280 = None, None
        invalid_tube = []
        for start in range(0, len(xyz), 65536):
            p = xyz[start:start+65536]-center
            projection = p@unit
            perpendicular = np.linalg.norm(p-projection[:, None]*unit, axis=1)
            zproj = projection/np.linalg.norm(direction)
            segment = (zproj >= .2) & (zproj <= target+.15)
            for trusted_only in (False, True):
                ids = np.flatnonzero(segment & (valid[start:start+len(p)] if trusted_only else True))
                if len(ids):
                    i = int(ids[np.argmin(perpendicular[ids])]); entry = (float(perpendicular[i]), start+i, float(zproj[i]))
                    prior = nearest_trusted if trusted_only else nearest
                    if prior is None or entry < prior:
                        if trusted_only: nearest_trusted = entry
                        else: nearest = entry
            cv = p@pose[:, :3]
            cz = -cv[:, 2]
            front = cz > .2
            uv = np.column_stack([camera["fl_x"]*cv[:, 0]/np.where(front, cz, 1)+camera["cx"]-.5,
                -camera["fl_y"]*cv[:, 1]/np.where(front, cz, 1)+camera["cy"]-.5])
            bins = np.floor(uv+.5).astype(np.int64)
            bins1280 = np.floor(2*uv+1).astype(np.int64)
            px, py = ray["pixel_xy"]
            for scale in (1, 2):
                ok = front & ((bins[:, 0] == px) & (bins[:, 1] == py) if scale == 1 else
                    ((bins1280[:, 0] >= 2*px) & (bins1280[:, 0] <= 2*px+1) &
                     (bins1280[:, 1] >= 2*py) & (bins1280[:, 1] <= 2*py+1)))
                ids = np.flatnonzero(ok)
                if len(ids):
                    i = int(ids[np.argmin(cz[ids])]); entry = (float(cz[i]), start+i)
                    old = same640 if scale == 1 else same1280
                    if old is None or entry < old:
                        if scale == 1: same640 = entry
                        else: same1280 = entry
            tube = segment & (perpendicular <= .05+1e-10)
            for i in np.flatnonzero(tube & ~valid[start:start+len(p)]):
                if zproj[i] < target-.15:
                    invalid_tube.append(dict(point_index=int(start+i), perpendicular_ray_distance_m=float(perpendicular[i]),
                        closest_ray_camera_z_m=float(zproj[i])))
            ids = np.flatnonzero(tube & valid[start:start+len(p)])
            if len(ids):
                den = normals[start+ids]@direction
                z = np.divide(np.sum(normals[start+ids]*p[ids], axis=1), den,
                    out=np.full(len(ids), np.nan), where=np.abs(den) > 1e-10)
                radial = np.linalg.norm(z[:, None]*direction-p[ids], axis=1)
                for j in np.flatnonzero(np.isfinite(z) & (z >= .2) & (z <= target+.15) & (radial <= .05+1e-10)):
                    candidates.append((start+int(ids[j]), float(z[j]), float(radial[j]), float(abs(den[j])/np.linalg.norm(direction))))
        hits, excluded = [], []
        for index, z, radial, incidence in candidates:
            if index not in cache:
                cache[index] = local_patch(index, xyz, normals, valid, tree)
            patch, basis, hull = cache[index]
            radius = patch["radius_m"]
            if radius is None or radial > radius+1e-10:
                excluded.append(dict(point_index=index, plane_camera_z_m=z, radial_distance_m=radial,
                    spacing_m=patch["spacing_m"], radius_m=radius))
                continue
            hit = center+z*direction
            nn_distance, nn = tree.query(hit, workers=3)
            inside = False
            if hull is not None:
                h = (hit-xyz[index])@basis
                inside = bool(np.all(hull[:, :2]@h+hull[:, 2] <= 1e-9))
            reasons = list(patch["local_support_reasons"])
            if incidence < .1: reasons.append("grazing_incidence_cos_below_0_1")
            if not inside: reasons.append("intersection_outside_local_measured_tangent_hull")
            info = point_info(index, ref, xyz, normals, valid, block_order)
            cp = (xyz[index]-center)@pose[:, :3]
            uv = np.array([camera["fl_x"]*cp[0]/-cp[2]+camera["cx"]-.5,
                -camera["fl_y"]*cp[1]/-cp[2]+camera["cy"]-.5])
            info.update(plane_camera_z_m=z, teacher_minus_patch_camera_z_m=target-z,
                foreground_before_fixed_teacher=bool(z < target-.15), hit_world_xyz=hit.tolist(),
                radius_m=radius, local_spacing_m=patch["spacing_m"], source_to_hit_distance_m=radial,
                incidence_abs_cos=incidence, normal_valid_measured_local_patch=True,
                local_measured_hull_contains_hit=inside, corroborated_local_intersection=not reasons,
                ambiguous_reasons=reasons, nearest_support_distance_m=float(nn_distance),
                nearest_support=point_info(int(nn), ref, xyz, normals, valid, block_order),
                source_projected_pixel_float_640=uv.tolist(), source_point_pixel640=np.floor(uv+.5).astype(int).tolist(),
                source_point_pixel1280=np.floor(2*uv+1).astype(int).tolist())
            hits.append(info)
        hits.sort(key=lambda h: (h["plane_camera_z_m"], h["point_index"]))
        foreground = [h for h in hits if h["foreground_before_fixed_teacher"]]
        corroborated = [h for h in foreground if h["corroborated_local_intersection"]]
        if corroborated: status = "corroborated_measured_foreground_intersection"
        elif foreground: status = "ambiguous_foreground_patch_intersection"
        else: status = "unknown_no_supported_foreground_patch_in_2_to_5cm_scope"
        def nearest_info(item):
            if item is None: return None
            distance, index, z = item
            return dict(**point_info(index, ref, xyz, normals, valid, block_order),
                perpendicular_ray_distance_m=distance, closest_ray_camera_z_m=z)
        def pixel_info(item):
            return None if item is None else dict(**point_info(item[1], ref, xyz, normals, valid, block_order), camera_z_m=item[0])
        result.append(dict(frame_key=frame["frame_key"], pixel_xy=ray["pixel_xy"], reference_camera_z_m=target,
            decision_status=status, finite_patch_intersections=hits, foreground_patch_count=len(foreground),
            corroborated_foreground_count=len(corroborated), nearest_foreground_patch=foreground[0] if foreground else None,
            nearest_corroborated_foreground_patch=corroborated[0] if corroborated else None,
            nearest_measured_point_to_ray_segment=nearest_info(nearest), nearest_normal_valid_point_to_ray_segment=nearest_info(nearest_trusted),
            nearest_raw_point_same_640_pixel=pixel_info(same640), nearest_raw_point_in_corresponding_1280_four_pixels=pixel_info(same1280),
            invalid_normal_points_in_5cm_foreground_tube=invalid_tube,
            rejected_outside_spacing_adaptive_radius=excluded,
            absence_is_not_visibility_or_empty_space_proof=True))
    return result


def groups(rays):
    out = {}
    for name, subset in (("all", rays), ("correct", [r for r in rays if r["model_comparison"]["conditioned_depth_class"] == "within_0_15m"]),
            ("wrong", [r for r in rays if r["model_comparison"]["conditioned_depth_class"] != "within_0_15m"]),
            ("too_near", [r for r in rays if r["model_comparison"]["conditioned_depth_class"] == "too_near"]),
            ("too_far", [r for r in rays if r["model_comparison"]["conditioned_depth_class"] == "too_far"])):
        states = {s: sum(r["decision_status"] == s for r in subset) for s in sorted({r["decision_status"] for r in rays})}
        out[name] = dict(rays=len(subset), decision_counts=states,
            patch_z_within_15cm_model_expected=sum(bool(r["model_comparison"]["any_corroborated_foreground_patch_within_15cm_expected"]) for r in subset))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=PREFIX)
    args = parser.parse_args()
    started = time.monotonic()
    selection_path = ROOT/"reports/surface_quality_v2_joint015_intermediate_aa_floor_selection.json"
    report_path = ROOT/"reports/surface_quality_v2_joint015_final_aa_floor.json"
    reference_path = ROOT/"data/garage_fidelity_depthv3_joint/core_015/lidar_reference.npz"
    prep_path = reference_path.with_name("preparation.json")
    inputs = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in (selection_path, report_path, reference_path, prep_path, Path(__file__))}
    selection = json.loads(selection_path.read_text())
    prep = json.loads(prep_path.read_text())
    assert selection["rays"] == 32 and len(selection["frames"]) == 4
    with np.load(reference_path, allow_pickle=False) as f:
        ref = {k: f[k] for k in f.files}
    xyz = ref["xyz"].astype(np.float64); normals = ref["normals"].astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    valid = ref["normal_valid"] & (lengths > 1e-8)
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 1e-8)
    assert len(xyz) == 1_200_000 and np.isfinite(xyz).all() and np.isfinite(normals).all()
    tree = cKDTree(xyz)
    rays, cache = [], {}
    for f in selection["frames"]:
        rays.extend(measured_frame(f, ref, xyz, normals, valid, tree, cache, prep["origin_source_order"]))
        assert time.monotonic()-started < 600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2*1024**2
        print(f"measured-only completed {f['frame_key'][1]}: {len(rays)}/32 rays", flush=True)
    # Only now read candidate outputs; they cannot influence measured sources, radii, support, or decisions.
    prior = json.loads(report_path.read_text())
    assert prior["selection"] == selection
    lookup = {(tuple(f["frame_key"]), tuple(r["pixel_xy"])): r for f in prior["current"]["frames"] for r in f["rays"]}
    for ray in rays:
        old = lookup[(tuple(ray["frame_key"]), tuple(ray["pixel_xy"]))]
        hits = [h for h in ray["finite_patch_intersections"] if h["foreground_before_fixed_teacher"] and h["corroborated_local_intersection"]]
        ray["model_comparison"] = dict(conditioned_depth_class=old["conditioned_depth_class"], expected_camera_z_m=old["expected_camera_z_m"],
            signed_error_m=old["signed_error_m"], AA_weight_share=old["AA"]["share_of_composited_alpha"],
            any_corroborated_foreground_patch_within_15cm_expected=any(abs(h["plane_camera_z_m"]-old["expected_camera_z_m"]) <= .15 for h in hits),
            nearest_corroborated_foreground_patch_minus_expected_z_m=(hits[0]["plane_camera_z_m"]-old["expected_camera_z_m"]) if hits else None)
    summary = groups(rays)
    for path, pinned in inputs.items():
        assert Path(path).stat().st_size == pinned["bytes"] and sha(path) == pinned["sha256"], "Diagnostic input changed"
    result = dict(schema_version=1, status="bounded_fixed32_measured_patch_diagnostic_complete", cpu_only=True, gpu_used=False,
        threads=3, training_modified=False, acceptance_protocol_modified=False, old_gated_denominator_modified=False,
        independent_ground_truth=False, creates_new_valid_depth=False, patch_policy=POLICY,
        selection_path=str(selection_path), source_reference_path=str(reference_path), source_reference_points=len(xyz),
        source_origin_block_order=prep["origin_source_order"], ray_count=32,
        candidate_output_read_after_all_measured_intersections=True,
        ray_interval="camera-z .2m through fixed teacher+.15m; all measured points scanned; finite discs centered on actual normal-valid reference points",
        local_patches={str(i): p[0] for i, p in sorted(cache.items())}, rays=rays, summaries=summary,
        limitations=["All evidence is conditioned on the same accumulated LIO map and its measured-normal estimates; no independent physical-surface truth or temporal visibility proof.",
            "A finite tangent disc is an explicit interpolation hypothesis; coherent measured-neighbor hull support is stronger evidence, not ground truth.",
            "Neighborhood and incidence rules were fixed before this diagnostic; they are uncalibrated conservative decision descriptors, not acceptance criteria.",
            "Unknown/ambiguous cases cannot become a pass or new valid depth. No intersection within the 2-5cm support cap does not prove foreground absence.",
            "Only the frozen 32 floor rays. Correct/wrong labels come from the previous fixed teacher and are postmeasurement group labels, not final truth.",
            "Repeated/overlapping block records are deduplicated to 10 micrometres for local spacing/support counts; original actual point indices are retained.",
            "This report does not modify training, references, ego masks, existing reports, renderer provenance validation or gate denominators."],
        inputs=inputs, elapsed_seconds=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_new(Path(str(args.output_prefix)+".json"), result)
    compact = {k: v for k, v in result.items() if k not in ("local_patches", "rays")}
    compact["rays"] = [{k: v for k, v in r.items() if k not in ("finite_patch_intersections", "invalid_normal_points_in_5cm_foreground_tube", "rejected_outside_spacing_adaptive_radius")} for r in rays]
    compact["full_report_path"] = str(args.output_prefix)+".json"
    write_new(Path(str(args.output_prefix)+"_summary.json"), compact)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"elapsed={result['elapsed_seconds']:.2f}s peak_rss={result['peak_rss_mib']:.1f}MiB", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()

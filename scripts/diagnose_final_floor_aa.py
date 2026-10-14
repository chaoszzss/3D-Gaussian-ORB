#!/usr/bin/env python3
"""Final-checkpoint CPU reuse of frozen 32 floor rays; no training/GPU changes."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import diagnose_floor_aa_contributions as D
import measure_surface_quality as S
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from verify_training_checkpoint import read_config
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "3"


def local(path):
    result = (ROOT/Path(path)).resolve()
    D.require(result.is_relative_to(ROOT), "Final diagnostic source must remain project-local")
    return result


def plain_json(path, guard):
    return S.read_json(guard.pin(path))


def plane_decomposition(mu, normal, nearest_point, nearest_normal, origin, ray, depth, fixed_z, bound):
    """Camera-z ray has CV z=1, so a unit-ray cosine alone is not the divisor."""
    norm = float(np.linalg.norm(nearest_normal))
    D.require(norm > 0, "Cannot use a zero reference normal")
    n = nearest_normal/norm
    signed = float((mu-nearest_point)@n)
    den = float(n@ray)
    unit = ray/np.linalg.norm(ray)
    result = dict(signed_center_to_reference_plane_m=signed,
        signed_center_to_raw_training_normal_m=float((mu-nearest_point)@nearest_normal),
        reference_normal_norm=norm, incidence_abs_cos=float(abs(n@unit)),
        surfel_incidence_abs_cos=float(abs(normal@unit)), normal_alignment_abs_cos=float(abs(normal@n)),
        camera_z_ray_norm=float(np.linalg.norm(ray)), n_dot_camera_z_ray=den,
        hit_signed_residual_to_center_reference_plane_m=float((origin+depth*ray-nearest_point)@n),
        decomposition=None)
    if abs(den) > 1e-8:
        refz = float(n@(nearest_point-origin)/den)
        parallel = float(n@(mu-origin)/den)
        offset = signed/den
        tilt = depth-parallel
        mismatch = refz-fixed_z
        residual = depth-fixed_z-offset-tilt-mismatch
        D.require(abs(residual) < 1e-7, "Plane depth decomposition failed")
        result["decomposition"] = dict(reference_plane_intersection_z_m=refz,
            parallel_plane_through_gaussian_z_m=parallel,
            signed_offset_z_component_m=offset, normal_direction_z_component_m=tilt,
            reference_plane_vs_fixed_teacher_z_component_m=mismatch,
            actual_contributor_z_error_m=depth-fixed_z, identity_residual_m=residual,
            allowed_bound_max_abs_z_shift_m=bound/abs(den), reference_plane_forward=refz >= .2)
    return result


def recover_and_measure(checkpoint, expected_sha, selection, guard):
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    D.require(not torch.cuda.is_initialized(), "CPU diagnostic unexpectedly initialized CUDA")
    fd = os.open(checkpoint, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        digest = D.fd_hash(fd)
        D.require(digest == expected_sha, "Final checkpoint differs from training completion SHA")
        state = torch.load(f"/proc/self/fd/{fd}", mmap=True, map_location="cpu", weights_only=False)
        D.require(state["step"] == 14999, "Requires actual final step 14999")
        params = {}
        for name, value in state["pipeline"].items():
            match = re.fullmatch(r"(?:module\.)?_model\.(?:module\.)?gauss_params\.(means|scales|quats|opacities)", name)
            if match:
                params[match.group(1)] = value
        del state, value
        D.require(set(params) == {"means", "scales", "quats", "opacities"}, "Incomplete final geometry tensors")
        n = len(params["means"])
        D.require(0 < n <= 3_000_000 and all(v.device.type == "cpu" and v.dtype == torch.float32 and len(v) == n for v in params.values()), "Invalid final tensor dtype/count")
        D.require(all(params[k].shape == (n, size) for k, size in (("means", 3), ("scales", 3), ("quats", 4), ("opacities", 1))), "Unexpected geometry shape")
        def chunks():
            for start in range(0, n, D.CHUNK):
                yield (start, *[params[k][start:start+D.CHUNK].detach().numpy() for k in ("means", "scales", "quats", "opacities")])
        measured = D.evaluate(chunks(), selection, guard, "final14999")
        ids = sorted({c["gaussian_index"] for f in measured["frames"] for r in f["rays"] for c in r["contributors"]})
        selected = {k: v[ids].detach().numpy().astype(np.float64) for k, v in params.items()}
        del params
        offsets = {gid: i for i, gid in enumerate(ids)}
        for frame in measured["frames"]:
            for ray in frame["rays"]:
                for c in ray["contributors"]:
                    i = offsets[c["gaussian_index"]]
                    c.update(world_xyz=selected["means"][i].tolist(), quaternion_wxyz=selected["quats"][i].tolist(),
                             log_scales=selected["scales"][i].tolist(), opacity_logit=float(selected["opacities"][i, 0]))
        after_sha = D.fd_hash(fd)
        after = os.fstat(fd)
        D.require(after_sha == digest and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                  (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Final checkpoint changed while held open")
        D.require(not torch.cuda.is_initialized(), "CPU diagnostic initialized CUDA")
        measured["checkpoint"] = dict(path=str(checkpoint), actual_step=14999, sha256_before=digest,
            sha256_after=after_sha, inode=before.st_ino, bytes=before.st_size, held_fd=True, mmap=True,
            optimizer_inspected=False, export_performed=False, retained_unique_gaussians=len(ids))
        return measured, ids, selected
    finally:
        os.close(fd)


def add_planes(measured, ids, selected, selection, reference_path, bound, guard):
    reference = S.bounded_npz(guard.pin(reference_path), 64*1024**2)
    xyz, normals, valid = (reference[k] for k in ("xyz", "normals", "normal_valid"))
    D.require(0 < len(xyz) <= 1_500_000 and xyz.shape == normals.shape and np.isfinite(xyz).all() and np.isfinite(normals).all(), "Invalid joint reference")
    tree = cKDTree(xyz)
    distance, nearest = tree.query(selected["means"], workers=3)
    offsets = {gid: i for i, gid in enumerate(ids)}
    q = selected["quats"]/np.linalg.norm(selected["quats"], axis=1, keepdims=True)
    w, x, y, z = q.T
    gaussian_normal = np.column_stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    hits, flattened = [], []
    for fi, frame in enumerate(measured["frames"]):
        camera = selection["frames"][fi]["camera"]
        pose = np.asarray(camera["camera_to_worlds_ns"])
        for ray in frame["rays"]:
            px, py = ray["pixel_xy"]
            # surfel_projection uses (world-C) @ saved_R. Invert that exact
            # float32-saved matrix for analytic plane closure; R @ ray assumes
            # exact orthogonality and amplifies tiny roundoff at grazing angles.
            direction = np.linalg.solve(pose[:, :3].T, np.array([(px+.5-camera["cx"])/camera["fl_x"], -(py+.5-camera["cy"])/camera["fl_y"], -1.]))
            for c in ray["contributors"]:
                i = offsets[c["gaussian_index"]]
                nn = int(nearest[i])
                mu, gn = selected["means"][i], gaussian_normal[i]
                cnz = float(-(mu-pose[:, 3])@pose[:, 2])
                D.require(abs(cnz-c["center_camera_z_m"]) < 1e-8, "Retained Gaussian center disagrees with forward contributor")
                surfel_den = float(gn@direction)
                recovered_z = float(gn@(mu-pose[:, 3])/surfel_den) if abs(surfel_den) > 1e-10 else None
                if recovered_z is not None:
                    D.require(abs(recovered_z-c["ray_plane_camera_z_m"]) < 1e-5, "Retained Gaussian plane differs from forward depth")
                normal = np.asarray(normals[nn], np.float64)
                geometry = plane_decomposition(mu, gn, np.asarray(xyz[nn], np.float64), normal, pose[:, 3], direction,
                    c["ray_plane_camera_z_m"], ray["reference_camera_z_m"], bound) if np.linalg.norm(normal) > 0 else dict(decomposition=None)
                trusted = bool(valid[nn] and distance[i] < .5 and np.linalg.norm(normal) > 0)
                geometry.update(fresh_nearest_trusted=trusted, nearest_reference_index=nn,
                    nearest_reference_xyz=xyz[nn].tolist(), nearest_reference_normal=normal.tolist(),
                    nearest_normal_valid=bool(valid[nn]), gaussian_nearest_distance_m=float(distance[i]),
                    gaussian_normal=gn.tolist(), nearest_reference_height_minus_fixed_floor_m=float(xyz[nn, 2]-ray["reference_xyz"][2]),
                    nearest_reference_abs_normal_z=float(abs(normal[2])/np.linalg.norm(normal)) if np.linalg.norm(normal) else None,
                    within_fresh_normal_bound=bool(abs(geometry["signed_center_to_raw_training_normal_m"]) <= bound+1e-6) if trusted else None,
                    recovered_surfel_plane_z_m=recovered_z)
                c["joint_reference_geometry"] = geometry
                hits.append(pose[:, 3]+direction*c["ray_plane_camera_z_m"])
                flattened.append((ray, c))
    hit_distance, hit_nearest = tree.query(np.asarray(hits), workers=3)
    for (ray, c), distance, nn in zip(flattened, hit_distance, hit_nearest):
        nn = int(nn)
        hn = normals[nn]
        hn_norm = np.linalg.norm(hn)
        c["joint_reference_geometry"].update(hit_nearest_reference_distance_m=float(distance), hit_nearest_reference_index=nn,
            hit_nearest_reference_xyz=xyz[nn].tolist(), hit_nearest_reference_normal_valid=bool(valid[nn]),
            hit_nearest_reference_abs_normal_z=float(abs(hn[2])/hn_norm) if hn_norm else None,
            hit_nearest_reference_height_minus_fixed_floor_m=float(xyz[nn, 2]-ray["reference_xyz"][2]))
    guard.check()
    return dict(path=str(reference_path), sha256=S.file_sha256(reference_path), points=len(xyz), bound_m=bound)


def weighted(values, weights):
    if not len(values) or sum(weights) <= 0:
        return None
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])/weights.sum()
    return dict(mean=float(values@weights/weights.sum()), p50=float(values[order[np.searchsorted(cumulative, .5)]]),
                p95=float(values[order[min(len(order)-1, np.searchsorted(cumulative, .95))]]), max=float(values.max()))


def summarize_planes(measured):
    groups = {}
    all_rays = [r for f in measured["frames"] for r in f["rays"]]
    for group in ("correct", "wrong", "missing_or_unknown"):
        rays = [r for r in all_rays if ("correct" if r["conditioned_depth_class"] == "within_0_15m" else
                "wrong" if r["conditioned_depth_class"] in ("too_near", "too_far") else "missing_or_unknown") == group]
        by_branch = {}
        for branch in ("AA", "nonAA", "all"):
            records = [c for r in rays for c in r["contributors"] if branch == "all" or c["AA"] == (branch == "AA")]
            weights = [c["weight"] for c in records]
            total = sum(weights)
            trusted = [c for c in records if c["joint_reference_geometry"]["fresh_nearest_trusted"]]
            decomposed = [c for c in trusted if c["joint_reference_geometry"]["decomposition"] is not None]
            def share(subset):
                return sum(c["weight"] for c in subset)/total if total else None
            def metric(key, subset=records, absolute=False):
                subset = [c for c in subset if c["joint_reference_geometry"].get(key) is not None]
                return weighted([(abs if absolute else float)(c["joint_reference_geometry"][key]) for c in subset], [c["weight"] for c in subset])
            components = {}
            for key in ("signed_offset_z_component_m", "normal_direction_z_component_m", "reference_plane_vs_fixed_teacher_z_component_m"):
                components[key] = weighted([abs(c["joint_reference_geometry"]["decomposition"][key]) for c in decomposed], [c["weight"] for c in decomposed])
            by_branch[branch] = dict(contributor_occurrences=len(records), unique_gaussians=len({c["gaussian_index"] for c in records}),
                composited_weight=total, fraction_of_group_alpha=total/sum(r["composited_alpha"] for r in rays) if rays else None,
                fresh_trusted_occurrences=len(trusted), fresh_trusted_weight_fraction=share(trusted),
                within_3cm_trusted_weight_fraction=share([c for c in trusted if c["joint_reference_geometry"]["within_fresh_normal_bound"]]),
                incidence_abs_cos=metric("incidence_abs_cos"), trusted_incidence_abs_cos=metric("incidence_abs_cos", trusted),
                trusted_abs_normal_offset_m=metric("signed_center_to_reference_plane_m", trusted, True),
                center_nearest_distance_m=metric("gaussian_nearest_distance_m"), hit_nearest_distance_m=metric("hit_nearest_reference_distance_m"),
                normal_alignment_abs_cos=metric("normal_alignment_abs_cos"),
                fresh_trusted_components_absolute_weighted_m=components,
                allowed_3cm_capacity_gt_1m_weight_fraction=share([c for c in decomposed if c["joint_reference_geometry"]["decomposition"]["allowed_bound_max_abs_z_shift_m"] > 1]),
                actual_offset_amplification_gt_1m_weight_fraction=share([c for c in decomposed if abs(c["joint_reference_geometry"]["decomposition"]["signed_offset_z_component_m"]) > 1]),
                center_nearest_normal_plane_mismatch_gt_1m_weight_fraction=share([c for c in decomposed if abs(c["joint_reference_geometry"]["decomposition"]["reference_plane_vs_fixed_teacher_z_component_m"]) > 1]),
                hit_within_20cm_joint_reference_weight_fraction=share([c for c in records if c["joint_reference_geometry"]["hit_nearest_reference_distance_m"] <= .2]),
                hit_near_joint_reference_with_height_difference_gt_1m_weight_fraction=share([c for c in records if c["joint_reference_geometry"]["hit_nearest_reference_distance_m"] <= .2
                    and abs(c["joint_reference_geometry"]["hit_nearest_reference_height_minus_fixed_floor_m"]) > 1]))
        groups[group] = dict(rays=len(rays), branches=by_branch)
    return dict(groups=groups, weighting="Native alpha*T per retained contributor occurrence across fixed rays; weighted quantiles use cumulative weight nearest order statistic. Unique Gaussian counts are reported separately.",
        note="Trusted statistics use a fresh nearest-reference query, not the uncheckpointed historical cache. Component absolute means are descriptive; signed components can cancel. A local plane extension mismatch or a height difference does not by itself prove a different physical surface.")


def validate_final_native(measured, selection, path, guard, wait_seconds):
    deadline = time.monotonic()+wait_seconds
    while not path.is_file() and time.monotonic() < deadline:
        guard.check()
        print("Waiting for fixed final native metrics (CPU idle)", flush=True)
        time.sleep(min(5, max(0, deadline-time.monotonic())))
    if not path.is_file():
        return dict(status="pending", reason="Final native metrics not yet available; no GPU launched")
    guard.pin(path)
    meta = S.load_metrics(path, guard.pin)
    D.require(meta.get("checkpoint_sha256") == measured["checkpoint"]["sha256_before"] and meta.get("step") == meta.get("inference_model_step") == 14999
        and meta.get("parameter_origin") == "trained_checkpoint" and meta.get("checkpoint_is_template") is False and meta.get("source_ply") is None,
        "Native final metrics are not the same actual checkpoint")
    D.require(meta["gaussians"] == measured["gaussians_scanned"] and meta.get("acceptance_plan_id") == selection["plan_id"]
        and meta.get("metric_domain") == "float_renderer", "Final native model/plan/domain differs")
    mapping = {tuple(S.frame_key(f)): f for f in meta["frames"]}
    errors = []
    for frame, fixed in zip(measured["frames"], selection["frames"]):
        row = mapping[tuple(frame["frame_key"])]
        D.require(S.same_camera(row["camera"], fixed["camera"]), "Native final fixed camera differs")
        arrays = S.bounded_npz(guard.pin(path.parent/("%03d_float.npz" % row["dataset_index"])), 16*1024**2)
        D.require(all(arrays[k].dtype == np.float32 for k in ("rgb", "depth", "alpha")), "Native float32 arrays required")
        for ray in frame["rays"]:
            x, y = ray["pixel_xy"]
            z, alpha = float(arrays["depth"][y,x].item()), float(arrays["alpha"][y,x].item())
            error_z = abs(ray["expected_camera_z_m"]-z) if ray["expected_camera_z_m"] is not None else None
            error_a = abs(ray["composited_alpha"]-alpha)
            passed = error_z is not None and error_z <= D.DEPTH_ATOL+D.DEPTH_RTOL*abs(z) and error_a <= D.ALPHA_ATOL
            record = dict(native_expected_z_m=z, native_alpha=alpha, depth_abs_error_m=error_z, alpha_abs_error=error_a, passed=bool(passed))
            ray["final_native_parity"] = record
            errors.append(record)
    return dict(status="passed" if all(e["passed"] for e in errors) else "failed", rays=len(errors),
        depth_atol_m=D.DEPTH_ATOL, depth_rtol=D.DEPTH_RTOL, alpha_atol=D.ALPHA_ATOL,
        max_depth_abs_error_m=max(e["depth_abs_error_m"] for e in errors if e["depth_abs_error_m"] is not None),
        max_alpha_abs_error=max(e["alpha_abs_error"] for e in errors), metrics_path=str(path), metrics_sha256=S.file_sha256(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, default=ROOT/"runs/fidelity_v2/train_joint015_depthv3_aa_metric_cap003")
    p.add_argument("--native-metrics", type=Path, default=ROOT/"runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/metrics.json")
    p.add_argument("--selection", type=Path, default=ROOT/"reports/surface_quality_v2_joint015_intermediate_aa_floor_selection.json")
    p.add_argument("--previous-report", type=Path, default=ROOT/"reports/surface_quality_v2_joint015_intermediate_aa_floor.json")
    p.add_argument("--output", type=Path, default=ROOT/"reports/surface_quality_v2_joint015_final_aa_floor.json")
    p.add_argument("--native-wait-seconds", type=int, default=90)
    args = p.parse_args()
    D.require(not args.output.exists() and 0 <= args.native_wait_seconds <= 120, "New output and bounded native wait required")
    guard = D.Guard(600)
    completion = plain_json(args.training_dir/"training_result.json", guard)  # never open checkpoint before this condition
    D.require(completion.get("status") == "trained" and completion.get("checkpoint_step") == 14999
        and completion.get("iterations_completed") == completion.get("iterations_requested") == 15000, "Training must first complete all 15000 iterations")
    old = plain_json(args.previous_report, guard)
    selection = plain_json(args.selection, guard)
    D.require(selection == old["selection"] and selection["rays"] == 32 and old["original_native_parity"]["passed"] is True,
        "Require unchanged frozen 32 rays and prior original-native oracle validation")
    for path, desc in old["inputs"].items():
        # Preserve the fixed geometric domain, not the previous measurement
        # program's hash (the optional integration-pointer patch was authorized).
        if path.endswith("_selection.json") or "/data/garage_surface_v1/" in path or "/runtime/acceptance_v2/masks/" in path or path.endswith("acceptance_plan_v1.json"):
            D.require(S.file_sha256(guard.pin(path)) == desc["sha256"], "Frozen source domain changed: "+path)
    for path in (__file__, D.__file__, S.__file__, ROOT/"scripts/assemble_core_gaussians_lidar_ownership.py"):
        guard.pin(path)
    for path in (D.__file__, ROOT/"scripts/assemble_core_gaussians_lidar_ownership.py"):
        D.require(S.file_sha256(path) == old["inputs"][str(Path(path).resolve())]["sha256"], "CPU forward equations changed since original native validation")
    cfg_path = local(completion["configs"][0])
    guard.pin(cfg_path)
    cfg = read_config(cfg_path)
    model, parser = cfg["pipeline"]["model"], cfg["pipeline"]["datamanager"]["dataparser"]
    world = plain_json(cfg_path.parent/"dataparser_transforms.json", guard)
    D.require(world["transform"] == [[1,0,0,0],[0,1,0,0],[0,0,1,0]] and world["scale"] == 1
        and model["camera_optimizer"]["mode"] == "off" and not model.get("optimize_shared_rig")
        and parser["orientation_method"] == parser["center_method"] == "none" and not parser["auto_scale_poses"] and parser["scale_factor"] == 1,
        "Final checkpoint/cameras must share original metre-world coordinates")
    bound = model["surface_max_normal_displacement_m"]
    D.require(bound == .03 and model["surface_splats"] is True, "Expected specified 3cm normal bound")
    launch = plain_json(args.training_dir/"launch.json", guard)
    D.require(model["inria_backend"] == completion["inria_backend"] == launch["inria_backend"] == "aa-depth-vjp"
        and launch["renderer_provenance"] == completion["renderer_provenance"], "Final backend provenance differs")
    renderer = S.validate_renderer_provenance(dict(backend="hbb1-aa-depth-vjp-2dgs", renderer_provenance=launch["renderer_provenance"]), guard.pin)
    checkpoint = local(completion["checkpoint"])
    D.require(checkpoint.name == "step-000014999.ckpt", "Expected final checkpoint filename")
    measured, ids, selected = recover_and_measure(checkpoint, completion["checkpoint_sha256"], selection, guard)
    D.require(measured["gaussians_scanned"] == completion["gaussians"], "Final completion Gaussian count differs")
    reference_path = local(model["lidar_reference_path"])
    reference = add_planes(measured, ids, selected, selection, reference_path, bound, guard)
    del selected
    validation = validate_final_native(measured, selection, args.native_metrics, guard, args.native_wait_seconds)
    guard.check()
    summary = dict(final=D.summarize(measured), intermediate6000=old["summaries"]["current"],
        original015=old["summaries"]["original"], planes=summarize_planes(measured))
    report = dict(schema_version=1, status="FINAL_CPU_FIXED_FLOOR_AA_DIAGNOSTIC", actual_step=14999,
        cpu_only=True, gpu_used=False, threads=3, chunk_gaussians=D.CHUNK, training_modified=False,
        acceptance_protocol_modified=False, high_fidelity_accepted=False, selection=selection,
        original_oracle_validation_reused=old["original_native_parity"], final_native_validation=validation,
        current=measured, joint_reference=reference, renderer_validation=renderer, summaries=summary,
        definitions=dict(trusted="Fresh nearest joint reference has valid normal and Gaussian-center Euclidean distance<0.5m; do not search past an invalid nearest point.",
            normal_bound="Compare raw-normal signed offset with configured .03m. Normal-plane geometry uses normalized normals.",
            amplification="delta(camera-z) = signed normal-plane offset / dot(reference unit normal, camera-z=1 world ray). The allowed bound produces a capacity, not proof of actual error causation.",
            decomposition="Contributor z-fixed teacher z = normal offset z component + surfel-vs-reference normal direction z component + local reference-plane intersection minus fixed teacher z.",
            physical_surface="Nearest-hit distance, normal and height are conditional map evidence; a nearby reference plane or a height difference alone cannot label a physical surface or prove missing points."),
        limitations=["Only the same fixed 32 floor rays, not whole-frame or whole-building completeness; correct/wrong groups are postmeasurement summaries, never adaptive selection.",
            "Same-source LIO, not independent ground truth. Final CPU/native parity status is explicit; pending/failed parity is never treated as passed.",
            "Historical nearest-normal cache is not checkpointed and refreshes every 250 steps. Fresh nearest queries may not match the exact training target.",
            "Normal clamp runs before training forward; saved post-optimizer parameters can move afterward. A fresh residual above 3cm is not by itself proof the clamp failed.",
            "Local reference plane extrapolation need not meet the same physical surface as the fixed teacher ray. Signed components can cancel.",
            "AA center-depth replacement is a named measurement-sensitivity diagnostic only; never an improvement, native protocol or acceptance change."],
        inputs=guard.assets, elapsed_seconds=time.monotonic()-guard.started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    D.write_new(args.output, report)
    summary_path = args.output.with_name(args.output.stem+"_summary.json")
    D.write_new(summary_path, dict(schema_version=1, status=report["status"], actual_step=14999,
        source_report=dict(path=str(args.output.resolve()), sha256=S.file_sha256(args.output)),
        final_native_validation=validation, summaries=summary, resource_usage=dict(seconds=report["elapsed_seconds"], peak_rss_mib=report["peak_rss_mib"], threads=3),
        limitations=report["limitations"], high_fidelity_accepted=False))
    print(json.dumps(dict(report=str(args.output), native_parity=validation, rays=summary["final"],
        seconds=report["elapsed_seconds"], rss_mib=report["peak_rss_mib"]), indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()

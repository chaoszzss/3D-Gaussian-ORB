#!/usr/bin/env python3
"""Bounded CPU fixed-reference floor rays: native AA exposure, not acceptance.

Select rays before accessing model outputs. Validate the original PLY against
existing native float arrays, then inspect one mmap checkpoint through an open
file descriptor. No export, optimizer operation, model import, or CUDA work.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import re
import resource
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np
from scipy.special import expit
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from assemble_core_gaussians_lidar_ownership import surfel_projection, ray_alpha
from package_garage import inspect_ply
import measure_surface_quality as S
from build_acceptance_plan import load_plan
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "3"

CHUNK = 65536
DEPTH_ATOL, DEPTH_RTOL, ALPHA_ATOL = .001, .0001, .0002


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Guard:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds
        self.assets = {}

    def pin(self, path):
        path = Path(path).resolve()
        st = path.stat()
        if str(path) not in self.assets:
            self.assets[str(path)] = dict(sha256=S.file_sha256(path), size=st.st_size,
                state=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns])
        return path

    def check(self):
        require(time.monotonic()-self.started < self.seconds, "CPU diagnostic time budget exhausted")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 2048*1024**2,
                "CPU diagnostic exceeded 2 GiB RSS bound")
        for name, desc in self.assets.items():
            st = Path(name).stat()
            require(desc["state"] == [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns],
                    "Pinned input changed: "+name)


def select_floor_rays(metrics, reference_path, plan_path, guard):
    reference = S.bounded_npz(guard.pin(reference_path), 64*1024**2)
    tree = cKDTree(reference["xyz"])
    plan = load_plan(guard.pin(plan_path), verify_assets=True)
    require(plan["plan_id"] == metrics["acceptance_plan_id"], "Plan identity differs")
    frozen = {S.frame_key(row): row for row in plan["frames"]}
    selected = []
    for row in metrics["frames"][:4]:
        fixed = frozen[S.frame_key(row)]
        camera = row["camera"]
        require(S.same_camera(camera, fixed["camera"]), "Frozen camera differs")
        depth_path = guard.pin(fixed["source_lidar_depth_path"])
        support_path = guard.pin(fixed["source_lidar_support_path"])
        mask_path = guard.pin(fixed["mask_path"])
        legacy_path = guard.pin(fixed["source_mask_path"])
        shape = (camera["h"], camera["w"])
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        valid = S.mask_array(mask_path, shape) & S.mask_array(legacy_path, shape)
        valid &= S.mask_array(support_path, shape) & np.isfinite(depth) & (depth > 0)
        ids = np.flatnonzero(valid)
        z = depth.reshape(-1)[ids]
        world = S.unproject(camera, ids, z, .5)
        distance, nearest = tree.query(world, workers=3)
        labels = S.classify_reference(reference, nearest, distance, camera["camera_to_worlds_ns"][2][3], S.DEFAULT_SETTINGS)
        floor = np.flatnonzero(labels == 0)
        require(len(floor) >= 8, "Insufficient fixed floor support in first four frames")
        ranks = np.floor((np.arange(8)+.5)*len(floor)/8).astype(int)
        positions = floor[ranks]
        rays = []
        for rank, position in zip(ranks, positions):
            y, x = divmod(int(ids[position]), camera["w"])
            ni = int(nearest[position])
            rays.append(dict(pixel_xy=[x, y], reference_camera_z_m=float(z[position]),
                selection_floor_rank=int(rank), flat_index=int(ids[position]), reference_nearest_index=ni,
                reference_nearest_distance_m=float(distance[position]), reference_xyz=reference["xyz"][ni].tolist(),
                reference_normal=reference["normals"][ni].tolist()))
        selected.append(dict(dataset_index=row["dataset_index"], frame_key=list(S.frame_key(row)), camera=camera,
            fixed_floor_pixels=len(floor), fixed_supported_pixels=len(ids), rays=rays,
            fixed_floor_indices_sha256=hashlib.sha256(ids[floor].astype("<i8").tobytes()).hexdigest()))
    require(len(selected) == 4 and len({f["frame_key"][0] for f in selected}) == 1, "Require first synchronized four DEV camera frames")
    return dict(frames=selected, rays=32, selection_policy="8 midpoint quantiles of row-major fixed-floor support per each first four original015 DEV frames; no model values, native outputs, candidate errors or AA exposure used for selection",
        floor_settings=S.DEFAULT_SETTINGS, plan_id=plan["plan_id"], reference_sha256=S.file_sha256(reference_path))


def ply_chunks(path):
    with Path(path).open("rb") as stream:
        spec = inspect_ply(stream)
        names = spec["names"]
        for start in range(0, spec["count"], CHUNK):
            n = min(CHUNK, spec["count"]-start)
            rows = np.frombuffer(stream.read(n*spec["stride"]), dtype="<f4").reshape(n, -1)
            yield start, rows[:, [names.index(k) for k in ("x", "y", "z")]], rows[:, [names.index(f"scale_{i}") for i in range(3)]], rows[:, [names.index(f"rot_{i}") for i in range(4)]], rows[:, names.index("opacity")]


def branch_stats(weight, depth, center, aa, ref, mask):
    w, d, c = weight[mask], depth[mask], center[mask]
    total, alpha = float(w.sum()), float(weight.sum())
    return dict(contributors=int(mask.sum()), weight=total, share_of_composited_alpha=total/alpha if alpha else None,
        weighted_mean_depth_m=float(w@d/total) if total else None,
        weighted_mean_center_z_m=float(w@c/total) if total else None,
        depth_minus_center_weighted_mean_m=float(w@(d-c)/total) if total else None,
        depth_minus_center_min_m=float(np.min(d-c)) if len(d) else None,
        depth_minus_center_max_m=float(np.max(d-c)) if len(d) else None,
        expected_depth_error_contribution_m=float(w@(d-ref)/alpha) if alpha else None,
        weighted_absolute_depth_error_contribution_m=float(w@np.abs(d-ref)/alpha) if alpha else None,
        nearer_than_reference_weight=float(w[d < ref-.15].sum()), farther_than_reference_weight=float(w[d > ref+.15].sum()))


def compose(chunks, ref):
    if chunks:
        ids, alpha, depth, center, aa = [np.concatenate([v[k] for v in chunks]) for k in range(5)]
    else:
        ids, alpha, depth, center, aa = np.empty(0, int), np.empty(0), np.empty(0), np.empty(0), np.empty(0, bool)
    order = np.lexsort((ids, center))
    ids, alpha, depth, center, aa = [v[order] for v in (ids, alpha, depth, center, aa)]
    after = np.cumprod(1.-alpha)
    stop = np.flatnonzero(after < 1e-4)
    limit = int(stop[0]) if len(stop) else len(ids)
    weight = alpha[:limit]*np.r_[1., after[:limit-1]] if limit else np.empty(0)
    depth, center, aa = depth[:limit], center[:limit], aa[:limit]
    total = float(weight.sum())
    expected = float(weight@depth/total) if total else None
    sensitivity = float(weight@np.where(aa, center, depth)/total) if total else None
    classify = lambda z: "unknown_depth" if z is None else ("missing_alpha" if total <= .5 else ("too_near" if z < ref-.15 else ("too_far" if z > ref+.15 else "within_0_15m")))
    return dict(precomposite_candidates=len(ids), composited_contributors=limit, early_stop_record_excluded=bool(len(stop)),
        composited_alpha=total, expected_camera_z_m=expected, signed_error_m=expected-ref if expected is not None else None,
        conditioned_depth_class=classify(expected),
        AA=branch_stats(weight, depth, center, aa, ref, aa), nonAA=branch_stats(weight, depth, center, aa, ref, ~aa),
        AA_center_depth_only_measurement_sensitivity=dict(expected_camera_z_m=sensitivity,
            delta_from_native_formula_m=sensitivity-expected if total else None,
            signed_error_m=sensitivity-ref if total else None, diagnostic_depth_class=classify(sensitivity),
            definition="Replace only AA contributors' depth by center camera-z after native alpha/culling/order/stop; all weights unchanged. Measurement sensitivity only, not a valid native renderer, improvement, acceptance or new protocol."),
        contributors=[dict(gaussian_index=int(ids[i]), AA=bool(aa[i]), weight=float(weight[i]), alpha=float(alpha[i]),
            ray_plane_camera_z_m=float(depth[i]), center_camera_z_m=float(center[i]), depth_minus_center_m=float(depth[i]-center[i])) for i in range(limit)])


def evaluate(chunks, selection, guard, label):
    selected = [[[] for _ in frame["rays"]] for frame in selection["frames"]]
    occurrence_count, n = 0, 0
    for start, xyz, logs, quats, logits in chunks:
        guard.check()
        require(all(np.isfinite(v).all() for v in (xyz, logs, quats, logits)), "Nonfinite geometry parameters")
        xyz, logs, quats = [np.asarray(v, np.float64) for v in (xyz, logs, quats)]
        require((np.linalg.norm(quats, axis=1) > 0).all(), "Zero quaternion")
        scales, opacity = np.exp(logs), expit(np.asarray(logits, np.float64).reshape(-1))
        require(np.isfinite(scales).all() and (scales > 0).all(), "Invalid scales")
        n += len(xyz)
        for fi, frame in enumerate(selection["frames"]):
            projection = surfel_projection(xyz, scales, quats, frame["camera"])
            for ri, ray in enumerate(frame["rays"]):
                pixel = ray["pixel_xy"]
                ids, alpha, depth = ray_alpha(projection, opacity, pixel)
                if not len(ids):
                    continue
                tu, tv, tw, image = [v[ids] for v in projection[:4]]
                cross = np.cross(pixel[0]*tw-tu, pixel[1]*tw-tv)
                uv = cross[:, :2]/cross[:, 2:3]
                aa = 2*np.sum((image-pixel)**2, axis=1) < np.sum(uv**2, axis=1)
                selected[fi][ri].append((ids+start, alpha, depth, tw[:, 2], aa))
                occurrence_count += len(ids)
                require(occurrence_count <= 1_000_000, "Selected contributor memory budget exceeded")
        if start//CHUNK % 8 == 0:
            print(f"{label}: scanned {n} actual Gaussians; {occurrence_count} ray candidates", flush=True)
    frames = []
    for fi, frame in enumerate(selection["frames"]):
        frames.append(dict(dataset_index=frame["dataset_index"], frame_key=frame["frame_key"],
            rays=[{**ray, **compose(selected[fi][ri], ray["reference_camera_z_m"])} for ri, ray in enumerate(frame["rays"])]))
    return dict(gaussians_scanned=n, selected_candidate_occurrences=occurrence_count, frames=frames)


def fd_hash(fd):
    h, offset = hashlib.sha256(), 0
    while True:
        block = os.pread(fd, 8*1024**2, offset)
        if not block:
            return h.hexdigest()
        h.update(block)
        offset += len(block)


def checkpoint_evaluate(directory, selection, guard):
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    paths = sorted(directory.glob("step-*.ckpt"), key=lambda p: int(p.stem[5:]), reverse=True)
    require(paths, "No intermediate checkpoint available")
    fd = None
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
            break
        except FileNotFoundError:
            continue
    require(fd is not None, "Checkpoint removed before opening")
    try:
        before = os.fstat(fd)
        digest = fd_hash(fd)
        state = torch.load(f"/proc/self/fd/{fd}", map_location="cpu", weights_only=False, mmap=True)
        step, pipeline = state["step"], state["pipeline"]
        require(type(step) is int and step == int(path.stem[5:]), "Actual checkpoint step differs from filename")
        parameters = {}
        for key, value in pipeline.items():
            match = re.fullmatch(r"(?:module\.)?_model\.(?:module\.)?gauss_params\.(means|scales|quats|opacities)", key)
            if match:
                parameters[match.group(1)] = value
        del pipeline, state  # optimizer entries not inspected, instantiated or modified
        require(set(parameters) == {"means", "scales", "quats", "opacities"}, "Missing actual geometry tensors")
        n = len(parameters["means"])
        require(0 < n <= 3_000_000 and all(v.device.type == "cpu" and v.dtype == torch.float32 and len(v) == n for v in parameters.values()), "Invalid/budget-exceeding checkpoint tensors")
        require(parameters["means"].shape == (n, 3) and parameters["scales"].shape == (n, 3)
                and parameters["quats"].shape == (n, 4) and parameters["opacities"].shape == (n, 1), "Unexpected Gaussian tensor shapes")
        def chunks():
            for i in range(0, n, CHUNK):
                yield (i, *[parameters[k][i:i+CHUNK].detach().numpy() for k in ("means", "scales", "quats", "opacities")])
        print(f"Opened actual step {step}, inode {before.st_ino}, {n} Gaussians, sha {digest}", flush=True)
        report = evaluate(chunks(), selection, guard, f"checkpoint{step}")
        require(fd_hash(fd) == digest, "Open checkpoint bytes changed during inspection")
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Open checkpoint inode/size/mtime changed")
        require(not torch.cuda.is_initialized(), "CPU diagnostic initialized CUDA")
        report["checkpoint"] = dict(path=str(path.resolve()), actual_step=step, sha256_before=digest, sha256_after=digest,
            inode=before.st_ino, bytes=before.st_size, nlink_before=before.st_nlink, nlink_after=after.st_nlink,
            mmap_via_held_fd=True, original_path_exists_at_end=path.exists(), export_performed=False,
            geometry="All actual means/log-scales/wxyz-quaternions/opacity-logits; no opacity export cutoff, no point pruning. SH is irrelevant to alpha/depth and was not read.",
            optimizer_inspected_or_modified=False)
        return report
    finally:
        os.close(fd)


def summarize(result):
    rays = [ray for frame in result["frames"] for ray in frame["rays"]]
    wrong = [r for r in rays if r["conditioned_depth_class"] in ("too_near", "too_far")]
    def aggregate(items):
        total = sum(r["composited_alpha"] for r in items)
        aa = sum(r["AA"]["weight"] for r in items)
        return dict(rays=len(items), class_counts={k:sum(r["conditioned_depth_class"] == k for r in items) for k in
                    ("within_0_15m", "too_near", "too_far", "missing_alpha", "unknown_depth")},
            rays_with_composited_AA=sum(r["AA"]["weight"] > 0 for r in items), AA_weight_sum=aa,
            AA_share_of_composited_alpha=aa/total if total else None,
            AA_depth_minus_center_weighted_mean_m=sum((r["AA"]["depth_minus_center_weighted_mean_m"] or 0)*r["AA"]["weight"] for r in items)/aa if aa else None,
            signed_error_p50_m=float(np.median([r["signed_error_m"] for r in items if r["signed_error_m"] is not None])) if items else None,
            AA_center_sensitivity_changes_class=sum(r["conditioned_depth_class"] != r["AA_center_depth_only_measurement_sensitivity"]["diagnostic_depth_class"] for r in items))
    return dict(all_fixed_rays=aggregate(rays), postmeasurement_wrong_depth_subset=aggregate(wrong),
        note="Wrong subset is a postmeasurement breakdown of the same fixed 32 rays; it never changes ray selection. Equal 8-ray camera strata, not an area-weighted building statistic.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=ROOT/"runs/fidelity_v2/original_single_blocks/block_015/metrics.json")
    p.add_argument("--reference", type=Path, default=ROOT/"data/garage_surface_v1/block_015/lidar_reference.npz")
    p.add_argument("--acceptance-plan", type=Path, default=ROOT/"reports/acceptance_plan_v1.json")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=750)
    args = p.parse_args()
    require(not args.output.exists() and 0 < args.seconds <= 900, "New report and <=15 minute budget required")
    guard = Guard(args.seconds)
    for path in (__file__, ROOT/"scripts/assemble_core_gaussians_lidar_ownership.py", ROOT/"scripts/measure_surface_quality.py"):
        guard.pin(path)
    metrics = S.load_metrics(guard.pin(args.metrics))
    guard.pin(metrics["transforms"])
    selection = select_floor_rays(metrics, args.reference, args.acceptance_plan, guard)
    selection_path = args.output.with_name(args.output.stem+"_selection.json")
    write_new(selection_path, selection)  # frozen before any PLY/native float/checkpoint read
    guard.pin(selection_path)
    print("Fixed reference selection saved:", selection_path, flush=True)
    ply = guard.pin(metrics["source_ply"]["path"])
    require(S.file_sha256(ply) == metrics["source_ply_sha256"], "Original PLY/render identity differs")
    original = evaluate(ply_chunks(ply), selection, guard, "original015")
    parity = []
    for frame in original["frames"]:
        native = S.bounded_npz(guard.pin(args.metrics.parent/("%03d_float.npz" % frame["dataset_index"])), 16*1024**2)
        for ray in frame["rays"]:
            x, y = ray["pixel_xy"]
            depth, alpha = float(native["depth"][y,x].item()), float(native["alpha"][y,x].item())
            require(ray["expected_camera_z_m"] is not None, "Original oracle has missing depth")
            err_d, err_a = abs(depth-ray["expected_camera_z_m"]), abs(alpha-ray["composited_alpha"])
            ray["native_parity"] = dict(native_expected_camera_z_m=depth, native_alpha=alpha,
                expected_depth_abs_error_m=err_d, alpha_abs_error=err_a,
                passed=bool(err_d <= DEPTH_ATOL+DEPTH_RTOL*abs(depth) and err_a <= ALPHA_ATOL))
            parity.append(ray["native_parity"])
    validation = dict(passed=all(r["passed"] for r in parity), rays=len(parity),
        depth_atol_m=DEPTH_ATOL, depth_rtol=DEPTH_RTOL, alpha_atol=ALPHA_ATOL,
        max_depth_abs_error_m=max(r["expected_depth_abs_error_m"] for r in parity), max_alpha_abs_error=max(r["alpha_abs_error"] for r in parity))
    print("Original native parity:", json.dumps(validation), flush=True)
    current = checkpoint_evaluate(args.checkpoint_dir, selection, guard) if validation["passed"] else None
    guard.check()
    report = dict(schema_version=1, status="CPU_FIXED_FLOOR_AA_DIAGNOSTIC" if current else "CPU_ORACLE_NATIVE_PARITY_FAILED",
        cpu_only=True, gpu_used=False, threads=3, chunk_gaussians=CHUNK, training_modified=False,
        acceptance_protocol_modified=False, high_fidelity_accepted=False, selection=selection,
        original_native_parity=validation, original=original, current=current,
        summaries={"original":summarize(original), **({"current":summarize(current)} if current else {})},
        conventions=dict(AA="rho2d < rho3d; both branches retain ray-plane depth", sort="center camera-z ascending, original Gaussian index stable tie break",
            stop="Exclude first record whose updated transmittance would be below 1e-4", expected_depth="sum(weight * ray-plane depth) / sum(weight)",
            error="camera-z expected depth minus fixed camera-z reference; alpha<=.5 reported separately; wrong depth if abs(error)>.15m",
            decomposition="AA.expected_depth_error_contribution_m + nonAA.expected_depth_error_contribution_m equals total signed error; opposite signs can cancel"),
        limitations=["32 deterministic floor rays in the first 4 block015 DEV camera frames, not whole-frame or whole-building completeness.",
            "Same-source LiDAR conditioned consistency, not independent truth. Current checkpoint is intermediate and is not a training outcome.",
            "Original CPU float64 equations checked against native float32 arrays on these exact rays; current checkpoint has no independent GPU render here.",
            "AA center-depth replacement keeps native weights/order/culling: measurement sensitivity only, never improvement, acceptance or a new native depth protocol.",
            "Forward contribution attribution does not measure backward gradient causality or training history."],
        inputs=guard.assets, elapsed_seconds=time.monotonic()-guard.started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_new(args.output, report)
    print(json.dumps(dict(report=str(args.output), parity=validation, summaries=report["summaries"],
        step=current["checkpoint"]["actual_step"] if current else None, seconds=report["elapsed_seconds"], rss_mib=report["peak_rss_mib"]), indent=2), flush=True)
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        sys.exit(main())

#!/usr/bin/env python3
"""Bounded, TRAIN-only, fixed-pose three-view visual triangulation.

No plane fitting, pose/calibration fitting, LiDAR filling, GPU, image resizing,
or dataset writes. A third observation must validate the first-two-view point
before it participates in refinement. Output points retain their observations.
"""
import argparse
from collections import Counter, OrderedDict
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
GATES = dict(baseline_min_m=.3, baseline_max_m=2., max_span_s=8.,
             max_view_axis_angle_deg=25., min_parallax_deg=1.,
             max_reprojection_px=1.5, min_depth_m=.2, max_depth_m=50.,
             max_forward_backward_px=.75, max_triangle_cycle_px=.75,
             max_lk_photometric_l1=20., min_dlt_nullspace_gap=10.,
             max_corners=1600, corner_quality=.01, corner_min_distance_px=8.,
             mask_patch_radius_px=15, low_track_warning_threshold=100)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_asset(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or root not in path.resolve().parents:
        raise ValueError("Asset must be an independent file inside the dataset")
    return path


def camera(frame):
    """NS/OpenGL C2W -> OpenCV C2W; OpenCV integer centres are NS minus .5."""
    T = np.asarray(frame["transform_matrix"], np.float64) @ np.diag([1., -1., -1., 1.])
    K = np.array([[frame["fl_x"], 0., frame["cx"] - .5],
                  [0., frame["fl_y"], frame["cy"] - .5], [0., 0., 1.]])
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.isfinite(K).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)
            or min(K[0, 0], K[1, 1]) <= 0):
        raise ValueError("Invalid fixed camera geometry")
    P = np.column_stack((T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]))
    return T, K, P


def load_training(dataset):
    dataset = Path(dataset).resolve()
    meta = json.loads((dataset / "transforms.json").read_text())
    prep = json.loads((dataset / "preparation.json").read_text())
    if prep.get("status") != "prepared_experimental" or (dataset / "INCOMPLETE.json").exists():
        raise ValueError("Require a complete prepared input")
    if not meta.get("pixel_center_convention", "").startswith("half_integer"):
        raise ValueError("Require explicit NS half-integer pixel convention")
    sets = [set(meta[k]) for k in ("train_filenames", "val_filenames", "test_filenames")]
    if any(len(meta[k]) != len(names) for k, names in zip(("train_filenames", "val_filenames", "test_filenames"), sets)):
        raise ValueError("Duplicate split image")
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("TRAIN/DEV/TEST overlap")
    all_names = [f["file_path"] for f in meta["frames"]]
    if len(set(all_names)) != len(all_names) or set(all_names) != set.union(*sets):
        raise ValueError("Split/frame identity mismatch")
    groups = {}
    for f in meta["frames"]:
        split = next(i for i, names in enumerate(sets) if f["file_path"] in names)
        g = str(f["group_id"])
        if g in groups and groups[g] != split:
            raise ValueError("Camera-time group crosses heldout boundary")
        groups[g] = split
    frames = [f for f in meta["frames"] if f["file_path"] in sets[0]]
    for f in frames:
        if any(np.any(np.asarray(f.get(key, meta.get(key, 0))) != 0) for key in
               ("k1", "k2", "k3", "k4", "p1", "p2", "distortion_params")):
            raise ValueError("Require already rectified pinhole images")
        if (f["w"], f["h"]) != (1280, 1080):
            raise ValueError("Require native1280x1080 TRAIN input; no image upscale")
        if type(f["timestamp_ns"]) is not int:
            raise ValueError("Source timestamps must retain integer nanoseconds")
        camera(f)
        for key in ("file_path", "mask_path"):
            if not safe_asset(dataset, f[key]).is_file():
                raise ValueError("Missing TRAIN image/mask")
    return frames, meta, prep


def select_triplets(frames, per_camera=8, max_triplets=32):
    """Choose using poses/time ONLY, before looking at any image or feature."""
    if not 1 <= per_camera <= 8 or not 1 <= max_triplets <= 32:
        raise ValueError("At most8 triplets per camera and32 total")
    cams = [camera(f)[0] for f in frames]
    selected, statistics = [], {}
    for cid in sorted({f["camera_id"] for f in frames}):
        ids = sorted((i for i, f in enumerate(frames) if f["camera_id"] == cid),
                     key=lambda i: (frames[i]["timestamp_ns"], frames[i]["file_path"]))
        candidates = []
        for ia, ib, ic in itertools.combinations(ids, 3):
            times = [frames[i]["timestamp_ns"] for i in (ia, ib, ic)]
            if not times[0] < times[1] < times[2] or (times[2] - times[0]) / 1e9 > GATES["max_span_s"]:
                continue
            Ts = [cams[i] for i in (ia, ib, ic)]
            baselines = [float(np.linalg.norm(Ts[a][:3, 3] - Ts[b][:3, 3])) for a, b in ((0, 1), (1, 2), (0, 2))]
            if min(baselines) < GATES["baseline_min_m"] or max(baselines) > GATES["baseline_max_m"]:
                continue
            angles = [float(np.degrees(np.arccos(np.clip(Ts[a][:3, 2] @ Ts[b][:3, 2], -1, 1)))) for a, b in ((0, 1), (1, 2), (0, 2))]
            if max(angles) > GATES["max_view_axis_angle_deg"]:
                continue
            score = abs(baselines[0] - .65) + abs(baselines[1] - .65) + .01 * max(angles)
            candidates.append(dict(camera_id=cid, train_indices=[ia, ib, ic], baseline_m=baselines,
                                   view_axis_angles_deg=angles, time_span_s=(times[2] - times[0]) / 1e9,
                                   middle_time_ns=times[1], pose_selection_score=score))
        chosen, uses = [], Counter()
        anchors = np.linspace(frames[ids[0]]["timestamp_ns"], frames[ids[-1]]["timestamp_ns"], per_camera) if ids else []
        for anchor in anchors:
            eligible = [c for c in candidates if c not in chosen and all(uses[i] < 2 for i in c["train_indices"])]
            if not eligible:
                break
            best = min(eligible, key=lambda c: (abs(c["middle_time_ns"] - anchor) / 1e9 + c["pose_selection_score"], c["train_indices"]))
            chosen.append(best); uses.update(best["train_indices"])
        statistics[cid] = dict(train_frames=len(ids), eligible_pose_triplets=len(candidates), selected=len(chosen))
        selected.extend(chosen)
    # Round-robin by camera if a smaller total budget is requested.
    by_cam = [[s for s in selected if s["camera_id"] == cid] for cid in sorted(statistics)]
    selected = [x for row in itertools.zip_longest(*by_cam) for x in row if x is not None][:max_triplets]
    return selected, statistics


def pixels_in_mask(uv, mask):
    uv = np.asarray(uv)
    good = np.isfinite(uv).all(axis=1)
    rounded = np.floor(np.where(np.isfinite(uv), uv, -1) + .5).astype(int)
    h, w = mask.shape
    good &= (rounded[:, 0] >= 0) & (rounded[:, 0] < w) & (rounded[:, 1] >= 0) & (rounded[:, 1] < h)
    ids = np.flatnonzero(good)
    good[ids] &= mask[rounded[ids, 1], rounded[ids, 0]] > 0
    return good


class ImageCache:
    def __init__(self, dataset, frames):
        self.dataset, self.frames, self.cache, self.hashes = dataset, frames, OrderedDict(), {}

    def get(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        frame = self.frames[index]
        paths = [safe_asset(self.dataset, frame[k]) for k in ("file_path", "mask_path")]
        rgb_bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or rgb_bgr.shape != (1080, 1280, 3) or mask is None or mask.shape != (1080, 1280):
            raise ValueError("Actual image/mask dimensions differ from native1280 calibration")
        for p in paths:
            digest = sha(p)
            if p in self.hashes and self.hashes[p] != digest:
                raise ValueError("Read-only TRAIN asset changed")
            self.hashes[p] = digest
        r = GATES["mask_patch_radius_px"]
        interior = cv2.erode(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        value = (cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), interior)
        self.cache[index] = value
        if len(self.cache) > 6:
            self.cache.popitem(last=False)
        return value


def corners(gray, mask):
    found = cv2.goodFeaturesToTrack(gray, maxCorners=4000, qualityLevel=GATES["corner_quality"],
                                  minDistance=GATES["corner_min_distance_px"], mask=mask, blockSize=7)
    if found is None:
        return np.empty((0, 2), np.float32)
    uv = found.reshape(-1, 2)
    # Spatial balancing uses the entire valid image; it is not a floor label.
    used, chosen = Counter(), []
    for i, (u, v) in enumerate(uv):
        cell = (min(7, int(u * 8 / gray.shape[1])), min(5, int(v * 6 / gray.shape[0])))
        if used[cell] < 40:
            chosen.append(i); used[cell] += 1
        if len(chosen) >= GATES["max_corners"]:
            break
    uv = uv[chosen].copy().reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, uv, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
    uv = uv.reshape(-1, 2)
    return uv[pixels_in_mask(uv, mask)]


def flow_pair(gray_a, gray_b, uv_a, mask_b):
    if not len(uv_a):
        return np.empty((0, 2), np.float32), np.empty(0, bool), np.empty(0), np.empty(0)
    args = dict(winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01), minEigThreshold=1e-4)
    uv_b, forward, error = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, uv_a.reshape(-1, 1, 2), None, **args)
    finite = np.isfinite(uv_b).all(axis=(1, 2))
    safe = np.where(np.isfinite(uv_b), uv_b, 0).astype(np.float32)
    back, backward, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, safe, None, **args)
    fb = np.linalg.norm(back.reshape(-1, 2) - uv_a, axis=1)
    ok = finite & (forward.ravel() > 0) & (backward.ravel() > 0) & np.isfinite(fb)
    ok &= (fb <= GATES["max_forward_backward_px"]) & (error.ravel() <= GATES["max_lk_photometric_l1"])
    uv_b = uv_b.reshape(-1, 2)
    ok &= pixels_in_mask(uv_b, mask_b)
    return uv_b, ok, fb, error.ravel()


def track_triplet(images):
    a, b, c = images
    uv0 = corners(a[0], a[2]); stats = dict(corners=len(uv0))
    uv1, good01, fb01, err01 = flow_pair(a[0], b[0], uv0, b[2])
    idx = np.flatnonzero(good01); stats["pair01_forward_backward"] = len(idx)
    if not len(idx):
        return [], stats
    uv0, uv1 = uv0[idx], uv1[idx]
    uv2, good02, fb02, err02 = flow_pair(a[0], c[0], uv0, c[2])
    via2, good12, fb12, err12 = flow_pair(b[0], c[0], uv1, c[2])
    cycle = np.linalg.norm(uv2 - via2, axis=1)
    good = good02 & good12 & np.isfinite(cycle) & (cycle <= GATES["max_triangle_cycle_px"])
    stats.update(pair02_forward_backward=int(good02.sum()), pair12_forward_backward=int(good12.sum()), triangle_cycle_consistent=int(good.sum()))
    tracks = []
    for j in np.flatnonzero(good):
        tracks.append(dict(uv=np.array([uv0[j], uv1[j], uv2[j]], np.float64),
                           forward_backward_px=[float(fb01[idx[j]]), float(fb02[j]), float(fb12[j])],
                           photometric_l1=[float(err01[idx[j]]), float(err02[j]), float(err12[j])],
                           cycle_px=float(cycle[j])))
    return tracks, stats


def dlt_point(uv, cameras):
    rows = []
    for xy, (_, K, P) in zip(uv, cameras):
        ray = np.linalg.solve(K, np.r_[xy, 1.])
        rows.extend([ray[0] * P[2] - P[0], ray[1] * P[2] - P[1]])
    _, s, vh = np.linalg.svd(np.asarray(rows))
    if s[-2] < 1e-10 or abs(vh[-1, 3]) < 1e-10:
        return None, 0.
    return vh[-1, :3] / vh[-1, 3], float(s[-2] / max(s[-1], 1e-15))


def project_point(xyz, cameras):
    pixels, depth = [], []
    for _, K, P in cameras:
        p = P @ np.r_[xyz, 1.]
        depth.append(p[2])
        q = K @ p
        pixels.append(q[:2] / q[2] if abs(q[2]) > 1e-12 else [np.nan, np.nan])
    return np.asarray(pixels), np.asarray(depth)


def refine_three(initial, uv, cameras):
    def residual(x):
        return (project_point(x, cameras)[0] - uv).ravel()
    result = least_squares(residual, initial, method="lm", max_nfev=50, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return result.x, bool(result.success)


def point_parallax(xyz, cameras):
    rays = np.asarray([xyz - T[:3, 3] for T, _, _ in cameras])
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return [float(np.degrees(np.arccos(np.clip(rays[a] @ rays[b], -1, 1))))
            for a, b in ((0, 1), (1, 2), (0, 2))]


def triangulate_track(uv, cameras):
    uv = np.asarray(uv, np.float64)
    if uv.shape != (3, 2) or not np.isfinite(uv).all():
        return None, "nonfinite_observation"
    initial, gap = dlt_point(uv[:2], cameras[:2])
    if initial is None or not np.isfinite(initial).all() or gap < GATES["min_dlt_nullspace_gap"]:
        return None, "degenerate_first_pair"
    projected, depth = project_point(initial, cameras)
    error = np.linalg.norm(projected - uv, axis=1)
    if not np.isfinite(depth).all() or np.any(depth < GATES["min_depth_m"]) or np.any(depth > GATES["max_depth_m"]):
        return None, "first_pair_cheirality_or_depth"
    if not np.isfinite(error).all() or np.any(error[:2] > GATES["max_reprojection_px"]):
        return None, "first_pair_reprojection"
    # The third image has not been used in the triangulation or refinement.
    if error[2] > GATES["max_reprojection_px"]:
        return None, "independent_third_view_reprojection"
    rays = []
    for xy, (T, K, _) in zip(uv, cameras):
        ray = T[:3, :3] @ np.linalg.solve(K, np.r_[xy, 1.])
        rays.append(ray / np.linalg.norm(ray))
    parallax = [float(np.degrees(np.arccos(np.clip(rays[a] @ rays[b], -1, 1)))) for a, b in ((0, 1), (1, 2), (0, 2))]
    if min(parallax) < GATES["min_parallax_deg"]:
        return None, "low_parallax"
    refined, converged = refine_three(initial, uv, cameras)
    if not converged or not np.isfinite(refined).all():
        return None, "refinement_failed"
    final_uv, final_depth = project_point(refined, cameras)
    final_error = np.linalg.norm(final_uv - uv, axis=1)
    if np.any(final_depth < GATES["min_depth_m"]) or np.any(final_depth > GATES["max_depth_m"]) or not np.isfinite(final_depth).all():
        return None, "refined_cheirality_or_depth"
    if not np.isfinite(final_error).all() or np.any(final_error > GATES["max_reprojection_px"]):
        return None, "refined_reprojection"
    geometric_parallax = point_parallax(refined, cameras)
    if min(geometric_parallax) < GATES["min_parallax_deg"]:
        return None, "refined_geometric_low_parallax"
    return dict(xyz=refined.tolist(), xyz_first_two=initial.tolist(), first_two_solution_reprojection_px=error.tolist(),
                reprojection_px=final_error.tolist(), parallax_deg=parallax, depths_m=final_depth.tolist(),
                geometric_parallax_deg=geometric_parallax,
                first_pair_dlt_nullspace_gap=gap), None


def deduplicate_observations(tracks):
    """Keep exact triangulations; suppress near-identical repeated image tracks."""
    kept, observations = [], {}
    order = sorted(range(len(tracks)), key=lambda i: (max(tracks[i]["reprojection_px"]), -min(tracks[i]["parallax_deg"]), i))
    for i in order:
        t = tracks[i]; repeated = False
        for index, xy in zip(t["train_indices"], t["uv"]):
            cell = np.floor(np.asarray(xy) / 2.).astype(int)
            for dx, dy in itertools.product((-1, 0, 1), repeat=2):
                if any(np.linalg.norm(np.asarray(xy) - prev) < 2. for prev in observations.get((index, cell[0] + dx, cell[1] + dy), [])):
                    repeated = True; break
            if repeated:
                break
        if repeated:
            continue
        kept.append(t)
        for index, xy in zip(t["train_indices"], t["uv"]):
            cell = np.floor(np.asarray(xy) / 2.).astype(int)
            observations.setdefault((index, cell[0], cell[1]), []).append(np.asarray(xy))
    return kept


def floor_band(xyz, centers):
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    if not len(xyz):
        return np.empty(0), np.empty(0), np.empty(0, bool), np.empty(0, np.int64)
    distance, nearest = cKDTree(np.asarray(centers)[:, :2]).query(xyz[:, :2], workers=1)
    relative_z = xyz[:, 2] - np.asarray(centers)[nearest, 2]
    inside = (distance <= 3.) & (relative_z >= -3.) & (relative_z <= -.4)
    return distance, relative_z, inside, nearest


def write_ply(path, xyz, colors):
    # Float64 round-trip text preserves actual optimized triangulations.
    with Path(path).open("x") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(xyz))
        stream.write("property double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, colors):
            stream.write(" ".join(format(float(x), ".17g") for x in p) + " " + " ".join(str(int(x)) for x in c) + "\n")


def run(dataset, output, threads=3, max_triplets=32):
    if not 1 <= threads <= 3:
        raise ValueError("At most3 CPU threads")
    cv2.setNumThreads(threads); cv2.setRNGSeed(0)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:threads]))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 2800 * 1024 ** 2
    resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard))
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Output must be independent from read-only dataset")
    if output.exists():
        raise FileExistsError("Never overwrite a triangulation experiment")
    frames, meta, prep = load_training(dataset)
    source_hashes = {str(dataset / name): sha(dataset / name) for name in ("preparation.json", "transforms.json", "lidar_reference.npz", "points3D.ply")}
    calibration = Path(prep["calibration"])
    if sha(calibration) != prep["calibration_sha256"]:
        raise ValueError("Frozen source calibration changed")
    source_hashes[str(calibration)] = sha(calibration)
    script_sha = sha(__file__)
    plan, selection = select_triplets(frames, max_triplets=max_triplets)
    output.mkdir(parents=True)
    write_json(output / "INCOMPLETE.json", {"status": "running_CPU_TRAIN_triangulation"})
    write_json(output / "triplet_plan.json", {"selection_before_matching": True, "gates": GATES, "camera_selection": selection,
               "triplets": plan, "train_frame_table": frames, "heldout_image_reads": 0})
    start = time.monotonic(); cache = ImageCache(dataset, frames); candidates, triplet_rows = [], []
    for ti, triplet in enumerate(plan):
        indices = triplet["train_indices"]; images = [cache.get(i) for i in indices]
        matches, stats = track_triplet(images); rejects = Counter()
        cameras = [camera(frames[i]) for i in indices]
        for match in matches:
            geometric, reason = triangulate_track(match["uv"], cameras)
            if reason:
                rejects[reason] += 1; continue
            sampled = [image[1][int(np.floor(xy[1] + .5)), int(np.floor(xy[0] + .5))].tolist() for image, xy in zip(images, match["uv"])]
            candidates.append(dict(geometric, triplet_id=ti, camera_id=triplet["camera_id"], train_indices=indices,
                                   train_file_ids=[frames[i]["file_path"] for i in indices],
                                   source_timestamp_ns=[frames[i]["timestamp_ns"] for i in indices],
                                   uv=match["uv"].tolist(), forward_backward_px=match["forward_backward_px"],
                                   photometric_l1=match["photometric_l1"], cycle_px=match["cycle_px"],
                                   sampled_train_rgb=sampled, rgb=np.median(sampled, axis=0).astype(np.uint8).tolist()))
        accepted = len(matches) - sum(rejects.values())
        row = dict(triplet, triplet_id=ti, matching=stats, geometry_rejections=dict(rejects), accepted_before_deduplication=accepted)
        triplet_rows.append(row)
        print(json.dumps({"triplet": ti + 1, "total": len(plan), "camera": triplet["camera_id"], "tracks": len(matches), "accepted": accepted, "rejects": dict(rejects)}), flush=True)
    tracks = deduplicate_observations(candidates)
    xyz = np.array([t["xyz"] for t in tracks], np.float64).reshape(-1, 3)
    colors = np.array([t["rgb"] for t in tracks], np.uint8).reshape(-1, 3)
    centers = np.array([camera(f)[0][:3, 3] for f in frames])
    xy, dz, band, nearest = floor_band(xyz, centers)
    for i, t in enumerate(tracks):
        t.update(track_id=i, nearest_train_center_index=int(nearest[i]), nearest_train_center_xy_distance_m=float(xy[i]),
                 relative_world_z_m=float(dz[i]), geometric_floor_band=bool(band[i]))
    write_ply(output / "points.ply", xyz, colors)
    np.savez_compressed(output / "tracks.npz", xyz=xyz, rgb=colors,
        colors=colors, train_file_ids=np.asarray([t["train_file_ids"] for t in tracks], dtype=str).reshape(-1, 3),
        xyz_first_two=np.array([t["xyz_first_two"] for t in tracks], np.float64).reshape(-1, 3),
        uv=np.array([t["uv"] for t in tracks], np.float64).reshape(-1, 3, 2),
        train_frame_indices=np.array([t["train_indices"] for t in tracks], np.int32).reshape(-1, 3),
        reprojection_px=np.array([t["reprojection_px"] for t in tracks], np.float64).reshape(-1, 3),
        first_two_solution_reprojection_px=np.array([t["first_two_solution_reprojection_px"] for t in tracks], np.float64).reshape(-1, 3),
        parallax_deg=np.array([t["parallax_deg"] for t in tracks], np.float64).reshape(-1, 3),
        geometric_parallax_deg=np.array([t["geometric_parallax_deg"] for t in tracks], np.float64).reshape(-1, 3),
        depths_m=np.array([t["depths_m"] for t in tracks], np.float64).reshape(-1, 3),
        source_timestamp_ns=np.array([t["source_timestamp_ns"] for t in tracks], np.int64).reshape(-1, 3),
        nearest_train_center_index=nearest, nearest_train_center_xy_distance_m=xy, relative_world_z_m=dz,
        geometric_floor_band=band, normal_valid=np.zeros(len(tracks), bool))
    write_json(output / "tracks.json", {"tracks": tracks, "train_frame_table": frames,
               "coordinate_system": "Original metric camera_init world; fixed source poses; no centering/scaling", "normal_estimation": "not performed; normal_valid=false"})
    for path, digest in dict(source_hashes, **{str(p): h for p, h in cache.hashes.items()}).items():
        if sha(path) != digest:
            raise ValueError("Source changed during triangulation; output remains incomplete")
    if sha(__file__) != script_sha:
        raise ValueError("Implementation changed while running")
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if high_water >= 3 * 1024:
        raise RuntimeError("RSS exceeded3GiB")
    warnings = []
    if len(tracks) < GATES["low_track_warning_threshold"]:
        warnings.append("LOW_TRACK_COUNT; gates were not weakened")
    if not band.any():
        warnings.append("NO_NEAR_TRAJECTORY_GEOMETRIC_FLOOR_BAND_POINTS")
    elif int(band.sum()) < GATES["low_track_warning_threshold"]:
        warnings.append("LOW_NEAR_TRAJECTORY_GEOMETRIC_FLOOR_BAND_COUNT; no sufficiency claim")
    if len(plan) < max_triplets:
        warnings.append("FEWER_POSE_ELIGIBLE_TRIPLETS_THAN_BUDGET")
    def quantiles(array):
        return dict(zip(("min", "median", "p95", "max"), [float(x) for x in np.percentile(array, [0, 50, 95, 100])])) if np.size(array) else None
    report = dict(status="COMPLETED_EXPERIMENTAL_TRAIN_VISUAL_POINTS_QUALITY_UNASSESSED", warnings=warnings,
        source_dataset=str(dataset), source_sha256=source_hashes, train_asset_sha256={str(p): h for p, h in cache.hashes.items()},
        script_sha256=script_sha, method="Native-resolution ShiTomasi/subpixel + bidirectional LK on all3 pairs + cycle consistency; fixed-pose pair DLT; independent third reprojection; three-view nonlinear refinement",
        gates=GATES, selection=selection, triplets=triplet_rows, triplets_processed=len(plan),
        candidates_before_observation_deduplication=len(candidates), exact_triangular_tracks=len(tracks),
        train_images_read=len({i for t in plan for i in t["train_indices"]}), dev_images_read=0, test_images_read=0,
        normals_estimated=False, plane_fabrication=False, pose_or_calibration_fitted=False,
        geometric_floor_band=dict(definition="Closest TRAIN camera centre in worldXY, XYdistance<=3m and worldZ-cameraZ in[-3,-0.4]m. Geometric diagnostic, not a semantic floor label.",
            points=int(band.sum()), points_all_depths=len(tracks), nearest_xy_le3m=int((xy <= 3).sum()),
            band_within8m=int(((xy <= 8) & (dz >= -3) & (dz <= -.4)).sum()),
            band_3to8m=int(((xy > 3) & (xy <= 8) & (dz >= -3) & (dz <= -.4)).sum()),
            by_camera={cid:sum(t["geometric_floor_band"] for t in tracks if t["camera_id"] == cid) for cid in sorted(selection)}),
        reprojection_px=quantiles([v for t in tracks for v in t["reprojection_px"]]),
        independent_third_reprojection_px=quantiles([t["first_two_solution_reprojection_px"][2] for t in tracks]),
        minimum_pairwise_parallax_deg=quantiles([min(t["parallax_deg"]) for t in tracks]),
        resources=dict(cpu_threads_max=threads, gpu_used=False, peak_rss_mb=high_water, elapsed_s=time.monotonic()-start),
        outputs_sha256={name:sha(output/name) for name in ("triplet_plan.json", "tracks.json", "tracks.npz", "points.ply")},
        limitations=["No absolute3D ground truth; inherited calibration and pose errors remain. Photometric tracks can still follow moving or repeated structures.",
                    "Three-view geometric validation is independent of third-view refinement, not a heldoutDEV/TEST evaluation.",
                    "Unknown surfaces are not filled; no semantic floor claim, planar projection, spatial voxel averaging or fabricated normals.",
                    "No existing DEV triangulations were loaded. No model, reference, calibration, gate or input dataset was modified.",
                    "Integration or training is a main-agent decision, not authorized by this artifact."])
    write_json(output / "report.json", report)
    (output / "INCOMPLETE.json").unlink()
    print(json.dumps({"status":report["status"], "tracks":len(tracks), "floor_band":int(band.sum()), "warnings":warnings, "report":str(output/'report.json')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/garage_fidelity_depthv3/block_000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/block000_visual_triangulation")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--max-triplets", type=int, default=32)
    args = parser.parse_args()
    run(args.dataset, args.output, args.threads, args.max_triplets)


if __name__ == "__main__":
    main()

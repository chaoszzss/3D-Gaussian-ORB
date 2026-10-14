#!/usr/bin/env python3
"""Fixed all28 reference/TRAIN-camera3 spatial screen; no images or model work."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
import csv
import datetime
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile
from collections import Counter, defaultdict

LIMIT = 3 * 1024**3
_soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (min(LIMIT, _soft) if _soft > 0 else LIMIT, _hard))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "reports/all28_low_surface_reference_coverage"
RADII = (3, 6, 8, 12)
Z_BAND = (-3.3, -2.0)
NORMAL_MIN = .85
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024**2), b""): h.update(part)
    return h.hexdigest()


def pin(path):
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    INPUTS[str(path.relative_to(ROOT))] = dict(sha256=digest, bytes=after.st_size)
    return digest


def bounds(points):
    return {axis: [float(points[:, i].min()), float(points[:, i].max())] for i, axis in enumerate("xyz")} if len(points) else None


def stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                max=float(a.max()), mean=float(a.mean())) if len(a) else dict(count=0, min=None, median=None, p95=None, max=None, mean=None)


def support(rows):
    return {str(r): dict(groups_with_support=sum(x[f"points_xy_le_{r}m"] > 0 for x in rows),
                        groups_without_support=sum(x[f"points_xy_le_{r}m"] == 0 for x in rows),
                        with_support_fraction=sum(x[f"points_xy_le_{r}m"] > 0 for x in rows) / len(rows),
                        without_support_fraction=sum(x[f"points_xy_le_{r}m"] == 0 for x in rows) / len(rows),
                        point_count_per_group=stats([x[f"points_xy_le_{r}m"] for x in rows])) for r in RADII}


def main():
    start = time.monotonic()
    outputs = [PREFIX.with_suffix(ext) for ext in (".csv", ".json", ".md")]
    assert not any(p.exists() for p in outputs), "Refuse to overwrite an earlier screen"
    script_sha = pin(Path(__file__).resolve())
    rows, blocks = [], []
    for b in range(28):
        bid = f"block_{b:03d}"
        directory = ROOT / "data/garage_surface_v1" / bid
        tp, rp = directory / "transforms.json", directory / "lidar_reference.npz"
        tsha, rsha = pin(tp), pin(rp)
        meta = json.loads(tp.read_text())
        assert meta.get("orientation_override") == "none"
        index = {f["file_path"]: (i, f) for i, f in enumerate(meta["frames"])}
        assert len(index) == len(meta["frames"])
        train = meta["train_filenames"]
        assert len(train) == len(set(train))
        assert not set(train) & (set(meta["val_filenames"]) | set(meta["test_filenames"]))
        groups = defaultdict(list)
        for ti, path in enumerate(train):
            fi, frame = index[path]
            groups[str(frame["group_id"])].append((ti, fi, frame))
        c3 = []
        for group, members in groups.items():
            selected = [m for m in members if m[2]["camera_id"] == "camera_3"]
            assert len(selected) == 1, (bid, group, "Missing/duplicate TRAIN camera3")
            c3.append((*selected[0], members))
        c3.sort(key=lambda m: m[0])
        with zipfile.ZipFile(rp) as archive:
            assert sum(i.file_size for i in archive.infolist()) < 256 * 1024**2, "Reference exceeds bounded read budget"
        with np.load(rp, allow_pickle=False) as ref:
            xyz = ref["xyz"].astype(np.float64)
            normals = ref["normals"].astype(np.float64)
            valid = ref["normal_valid"]
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and normals.shape == xyz.shape and valid.shape == (len(xyz),)
        assert valid.dtype == np.bool_ or np.all((valid == 0) | (valid == 1))
        finite = np.isfinite(xyz).all(axis=1)
        length = np.linalg.norm(normals, axis=1)
        trusted = finite & valid.astype(bool) & np.isfinite(normals).all(axis=1) & (length > 1e-8)
        unit_nz = np.divide(np.abs(normals[:, 2]), length, out=np.zeros(len(xyz)), where=length > 1e-8)
        selected = trusted & (unit_nz >= NORMAL_MIN)
        ids, points = np.flatnonzero(selected), xyz[selected]
        band_union = np.zeros(len(points), bool)
        block_rows, poses = [], []
        for order, (ti, fi, frame, members) in enumerate(c3):
            pose = np.asarray(frame["transform_matrix"], np.float64)
            assert pose.shape == (4, 4) and np.isfinite(pose).all() and np.array_equal(pose[3], [0, 0, 0, 1])
            camera = pose[:3, 3]; poses.append(camera)
            dz = points[:, 2] - camera[2]
            mask = (dz >= Z_BAND[0]) & (dz <= Z_BAND[1])
            band_union |= mask
            local = points[mask]
            local_ids = ids[mask]
            dxy2 = np.sum((local[:, :2] - camera[:2]) ** 2, axis=1)
            dz_local = local[:, 2] - camera[2]
            nearest = int(np.argmin(dxy2)) if len(local) else None
            timestamps = [int(m[2]["timestamp_ns"]) for m in members]
            group = str(frame["group_id"])
            row = dict(block_id=bid, floor_id="unassigned", floor_verified=False,
                camera_id="camera_3", view_id=frame.get("view_id"), train_camera3_order=order,
                train_dataset_index=ti, source_frame_index=fi, group_id=group,
                camera_timestamp_ns=str(frame["timestamp_ns"]), source_header_ns=str(frame["source_header_ns"]),
                source_bag_ns=str(frame["source_bag_ns"]), camera_timestamp_minus_group_ns=int(frame["timestamp_ns"])-int(group),
                group_train_frames=len(members), group_train_camera_ids=";".join(sorted({m[2]["camera_id"] for m in members})),
                group_anchor_matching_camera_ids=";".join(sorted({m[2]["camera_id"] for m in members if int(m[2]["timestamp_ns"]) == int(group)})),
                group_train_timestamp_min_ns=str(min(timestamps)), group_train_timestamp_max_ns=str(max(timestamps)),
                group_train_timestamp_span_ns=max(timestamps)-min(timestamps),
                file_path=frame["file_path"], source_file_path=frame.get("source_file_path"),
                camera_x_m=float(camera[0]), camera_y_m=float(camera[1]), camera_z_m=float(camera[2]),
                height_band_world_z_min_m=float(camera[2]+Z_BAND[0]), height_band_world_z_max_m=float(camera[2]+Z_BAND[1]),
                eligible_trusted_horizontal_height_band_points=len(local),
                height_band_points_any_normal=int((finite & (xyz[:,2]-camera[2] >= Z_BAND[0]) & (xyz[:,2]-camera[2] <= Z_BAND[1])).sum()),
                **{f"points_xy_le_{r}m": int((dxy2 <= r*r).sum()) for r in RADII},
                nearest_eligible_xy_m=float(np.sqrt(dxy2[nearest])) if nearest is not None else None,
                nearest_eligible_3d_m=float(np.sqrt(np.min(dxy2+dz_local**2))) if len(local) else None,
                nearest_xy_point_reference_index=int(local_ids[nearest]) if nearest is not None else None,
                **{f"nearest_xy_point_{axis}_m": float(local[nearest, i]) if nearest is not None else None for i, axis in enumerate("xyz")},
                nearest_xy_point_delta_z_m=float(dz_local[nearest]) if nearest is not None else None,
                nearest_xy_point_3d_m=float(np.sqrt(dxy2[nearest]+dz_local[nearest]**2)) if nearest is not None else None,
                reference_sha256=rsha, transforms_sha256=tsha)
            assert all(row[f"points_xy_le_{a}m"] <= row[f"points_xy_le_{b}m"] for a, b in zip(RADII[:-1], RADII[1:]))
            if nearest is not None:
                assert Z_BAND[0] <= row["nearest_xy_point_delta_z_m"] <= Z_BAND[1]
            block_rows.append(row)
        rows.extend(block_rows)
        blocks.append(dict(block_id=bid, floor_id=None, floor_verified=False,
            reference_path=str(rp.relative_to(ROOT)), reference_sha256=rsha, transforms_path=str(tp.relative_to(ROOT)), transforms_sha256=tsha,
            reference_points=len(xyz), finite_reference_points=int(finite.sum()), normal_valid_flag_points=int(valid.sum()),
            trusted_finite_normal_points=int(trusted.sum()), trusted_horizontal_points=len(points),
            reference_xyz_range_m=bounds(xyz[finite]), trusted_horizontal_xyz_range_m=bounds(points),
            eligible_for_any_train_camera3_unique_reference_points=int(band_union.sum()),
            eligible_union_xyz_range_m=bounds(points[band_union]),
            train_frames=len(train), train_groups=len(groups), train_camera3_groups=len(block_rows),
            train_camera3_xyz_range_m=bounds(np.asarray(poses)),
            camera3_timestamp_ns_range=[str(min(int(r["camera_timestamp_ns"]) for r in block_rows)), str(max(int(r["camera_timestamp_ns"]) for r in block_rows))],
            group_id_range=[str(min(int(r["group_id"]) for r in block_rows)), str(max(int(r["group_id"]) for r in block_rows))],
            camera3_timestamp_minus_group_ns=stats([r["camera_timestamp_minus_group_ns"] for r in block_rows]),
            camera3_timestamp_differs_from_group_rows=sum(r["camera_timestamp_minus_group_ns"] != 0 for r in block_rows),
            groups_without_any_eligible_point=sum(r["eligible_trusted_horizontal_height_band_points"] == 0 for r in block_rows),
            radius_support=support(block_rows), nearest_eligible_xy_m=stats([r["nearest_eligible_xy_m"] for r in block_rows]),
            nearest_eligible_3d_m=stats([r["nearest_eligible_3d_m"] for r in block_rows])))
        print(bid, len(block_rows), "TRAIN groups; with support", {str(r): blocks[-1]["radius_support"][str(r)]["groups_with_support"] for r in RADII}, flush=True)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < LIMIT
        assert time.monotonic()-start < 600, "Bounded screen exceeded10minutes"
    unique = defaultdict(list)
    for row in rows: unique[row["group_id"]].append(row)
    repeated_pose_max = max(float(np.ptp(np.asarray([[r["camera_x_m"],r["camera_y_m"],r["camera_z_m"]] for r in members]), axis=0).max()) for members in unique.values())
    radius_blocks = {str(r): {"all_groups_without_support": [b["block_id"] for b in blocks if b["radius_support"][str(r)]["groups_with_support"] == 0],
        "some_groups_without_support": [b["block_id"] for b in blocks if 0 < b["radius_support"][str(r)]["groups_without_support"] < b["train_groups"]],
        "all_groups_with_support": [b["block_id"] for b in blocks if b["radius_support"][str(r)]["groups_without_support"] == 0]} for r in RADII}
    for name, info in INPUTS.items(): assert sha(ROOT/name) == info["sha256"], "Input changed during screen"
    with outputs[0].open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    overall = dict(blocks=28, block_group_rows=len(rows), unique_camera3_timegroups=len(unique),
        repeated_block_group_rows=len(rows)-len(unique), max_repeated_group_camera_xyz_spread_m=repeated_pose_max,
        radius_support_block_group_weighted=support(rows), block_categories_by_radius=radius_blocks,
        nearest_eligible_xy_m=stats([r["nearest_eligible_xy_m"] for r in rows]),
        camera3_timestamp_differs_from_group_rows=sum(r["camera_timestamp_minus_group_ns"] != 0 for r in rows),
        camera3_timestamp_minus_group_ns=stats([r["camera_timestamp_minus_group_ns"] for r in rows]),
        source_header_equals_camera_timestamp_all=all(r["source_header_ns"] == r["camera_timestamp_ns"] for r in rows),
        camera3_z_range_m=[min(r["camera_z_m"] for r in rows),max(r["camera_z_m"] for r in rows)],
        reference_rows_total_not_unique=sum(b["reference_points"] for b in blocks))
    report = dict(schema_version=1, status="BOUNDED_REFERENCE_COVERAGE_SCREEN_COMPLETE",
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), high_fidelity_accepted=False, geometric_acceptance=False,
        method=dict(camera="camera_3", split="TRAIN", normal_valid_required=True, normal_normalized=True,
            abs_unit_normal_z_min=NORMAL_MIN, relative_world_z_band_m=list(Z_BAND), xy_radii_m=list(RADII), boundaries="inclusive",
            distance="nearest_eligible_xy_m is exact XY distance across all eligible reference rows in that block; nearest_eligible_3d_m is a separate minimum. No eligible point -> null, not zero.",
            points="existing lidar_reference.npz only; all stored rows examined; no extra sampling; no image/PCD/bag reads",
            time_relation="group_id is the recorded grouping anchor; camera_timestamp_ns/source_header_ns/source_bag_ns are retained separately. Each group's sole TRAIN camera3 pose is used, never an interpolated group-anchor pose.",
            height_hint=dict(value_m=2.65, measured=False, sensor="unspecified LiDAR or camera", used_as_fixed_height=False),
            floor_assignment="unassigned; world Z bands and block IDs are not measured floor labels"),
        overall=overall, blocks=blocks, csv=dict(path=str(outputs[0].relative_to(ROOT)), sha256=sha(outputs[0]), rows=len(rows)),
        input_files=INPUTS, implementation=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=script_sha),
        execution=dict(cpu_thread_limit=3, cpu_affinity=sorted(os.sched_getaffinity(0)), address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, elapsed_seconds=time.monotonic()-start, gpu_used=False),
        limitations=["This is spatial point presence in prepared, cropped/capped references (observed400000rows/block), not original map or bag completeness. Sampling/cropping and invalid normals can remove otherwise real support.",
            "Normal_valid comes from prepared local PCA confidence; abs(unit_nz)>=.85 describes orientation, not semantic ground or an independently verified normal.",
            "World+Z and metres are inherited camera_init coordinates; a[-3.3,-2.0]m band below each camera3 is a wide screening condition, not a measured mounting height or floor plane.",
            "Ramps, multiple levels, nearby platforms/vehicles and changing camera Z can affect membership; no true floor assignment or six-floor acceptance is inferred.",
            "XY proximity does not prove visibility, line of sight, density sufficiency, contemporaneous LiDAR capture, or recoverable image texture. One point counts as presence only.",
            "Overlapping blocks repeat groups and may repeat reference points; overall fractions use block-group observations, not unique samples or area coverage.",
            "The earlier first-six-scan angles/15.8m conditional ray intersection/21.6m observed returns describe only block000 start and were not recomputed or generalized here."])
    with outputs[1].open("x") as f: json.dump(report, f, indent=2, allow_nan=False); f.write("\n")
    lines = ["# 全28块低位近水平参考点覆盖筛查", "", "仅检查已准备的 LiDAR reference 与 TRAIN camera_3 位姿；不是语义地面、几何验收或六层完整性证明。", "",
        "条件：normal_valid 且有限、归一化法线 |nz|≥0.85；每条实际 TRAIN camera_3 位姿的世界 Z 下方2.0–3.3m；XY半径3/6/8/12m，边界均包含。最近距离不限于12m。", "",
        f"共28块、{len(rows)}条块内时间组记录、{len(unique)}个去重时间组；重叠重复{len(rows)-len(unique)}条。每块reference均有400,000行，不能将其加总当作唯一原始点数。", "",
        f"camera_3 时间戳与group_id锚点不同的记录：{overall['camera3_timestamp_differs_from_group_rows']}条；差值范围{overall['camera3_timestamp_minus_group_ns']['min']/1e6:.3f}至{overall['camera3_timestamp_minus_group_ns']['max']/1e6:.3f}ms。CSV保留实际时间、源header/bag时间、TRAIN索引及组内成员；纳秒列应按字符串/整数解析。", "",
        "下表百分比是有≥1个符合条件参考点的TRAIN组比例，不是表面面积覆盖率或质量合格率。楼层归属全部未核实。", "",
        "|block|TRAIN组|3m有点|6m有点|8m有点|12m有点|最近XY中位(m)|camera3世界Z范围(m)|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for b in blocks:
        percentages = [f"{100*b['radius_support'][str(r)]['with_support_fraction']:.1f}%" for r in RADII]
        median = b["nearest_eligible_xy_m"]["median"]; z = b["train_camera3_xyz_range_m"]["z"]
        lines.append("|"+"|".join([b["block_id"],str(b["train_groups"]),*percentages, f"{median:.3f}" if median is not None else "无符合点",f"{z[0]:.3f}–{z[1]:.3f}"])+"|")
    lines.extend(["", "按半径分组（全部沿用block编号，不推测楼层）：", ""])
    for r in RADII:
        cats = radius_blocks[str(r)]
        lines.append(f"- {r}m：全组无点的块为{', '.join(cats['all_groups_without_support']) or '无'}；部分组无点{len(cats['some_groups_without_support'])}块；全组有点{len(cats['all_groups_with_support'])}块。")
    lines.extend(["", "点XYZ范围、候选点范围、相机轨迹XYZ范围、每个半径的有/无比例及最近距离分位数详见JSON；全部逐组计数和实际相机时间关系详见CSV。", "",
        "限制：400k裁剪/采样参考可能漏点；宽Z带可能跨坡道或其他低位结构；可信法线也不是地面语义。空缺只说明该参考和筛查条件下未找到点。约2.65m高度未测且传感器未指定，本筛查不拟合或锁定高度。最初6扫描的射线/return距离不向其他block外推。", "",
        f"执行：CPU≤3线程，地址空间上限3GiB，RSS峰值{report['execution']['process_peak_rss_bytes']/1024**2:.1f}MiB，耗时{report['execution']['elapsed_seconds']:.2f}s；未读图片、PCD或bag，未调用GPU，未修改输入和工程阈值。", "",
        "复现命令：`PYTHONDONTWRITEBYTECODE=1 .venv-train/bin/python -B scripts/screen_low_surface_reference_coverage.py`（已存在输出时拒绝覆盖）。", ""])
    with outputs[2].open("x") as f: f.write("\n".join(lines))
    print(json.dumps({"outputs": [str(p.relative_to(ROOT)) for p in outputs], "rows": len(rows), "distinct_groups": len(unique), "resources": report["execution"]}), flush=True)


if __name__ == "__main__": main()

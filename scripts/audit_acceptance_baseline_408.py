#!/usr/bin/env python3
"""Read-only CPU audit of three existing 408-frame baseline evaluations.

Writes only the requested new JSON/Markdown reports. Does not run a renderer,
train, change the frozen plan, or evaluate a new reconstruction candidate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
import cv2
import numpy as np

from build_acceptance_plan import ROOT, PROTOCOLS, frame_key, load_plan, object_sha, sha256_file
from acceptance_gate import read_reports, validate_render_reports, paired

DATASETS = {"original_single": "original_single_blocks", "old_full": "baseline_old_all28",
            "old_full_repeat": "baseline_old_all28_repeat"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack("!d", a) == struct.pack("!d", b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(bits_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(bits_equal(x, y) for x, y in zip(a, b))
    return a == b


def distribution(values):
    v = np.asarray(values, dtype=np.float64)
    require(len(v) > 0 and np.isfinite(v).all(), "Empty/nonfinite distribution")
    return {"count": len(values), "mean": float(v.mean()), "median": float(np.median(v)),
            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}


def scores(rows, protocol):
    values = [r["protocols"][protocol] for r in rows]
    pixels = sum(v["rgb_valid_pixels"] for v in values)
    ssim_pixels = sum(v["ssim_compared_pixels"] for v in values)
    depth_pixels = sum(v["map_depth_eligible_pixels"] for v in values)
    return {"images": len(values), "psnr_db_mean": float(np.mean([v["psnr_db"] for v in values])),
            "rgb_valid_pixels": pixels,
            "rgb_mse_pixel_mean": sum(v["rgb_mse"] * v["rgb_valid_pixels"] for v in values) / pixels,
            "ssim_valid_mean": float(np.mean([v["ssim_valid_pixels"] for v in values])),
            "ssim_pixel_mean": sum(v["ssim_valid_pixels"] * v["ssim_compared_pixels"] for v in values) / ssim_pixels,
            "alpha_gt_0_5_valid_pixel_fraction": sum(v["alpha_gt_0_5_valid_pixels"] for v in values) / pixels,
            "map_depth_eligible_pixels": depth_pixels,
            "map_depth_coverage_fraction": sum(v["map_depth_compared_pixels"] for v in values) / depth_pixels}


def audit(output_json, output_md):
    plan_path = ROOT / "reports/acceptance_plan_v1.json"
    plan = load_plan(plan_path, verify_assets=True)
    require(plan["unique_frames"] == 408, "This audit requires exactly the frozen 408 frames")
    source_hashes = {str(plan_path): sha256_file(plan_path), **plan["inputs"]}
    states = {}
    def state(path):
        s = Path(path).stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    def pin(path):
        path = Path(path).resolve()
        before = state(path)
        digest = sha256_file(path)
        require(state(path) == before, "Input changed while hashing: " + str(path))
        source_hashes[str(path)] = digest; states[str(path)] = before
        return digest
    def read(path):
        pin(path)
        return json.loads(Path(path).read_text())

    for path in plan["inputs"]:
        states[path] = state(path)

    manifest = {role: [str(ROOT / "runs/fidelity_v2" / directory / "summary.json")]
                for role, directory in (("single_block", DATASETS["original_single"]),
                                        ("baseline", DATASETS["old_full"]), ("repeat", DATASETS["old_full_repeat"]))}
    loaded, hashes = read_reports(manifest)
    source_hashes.update(hashes)
    for path, digest in hashes.items():
        states[path] = state(path)
        require(sha256_file(path) == digest, "Evidence changed during initial loading")
    for script in ("scripts/audit_acceptance_baseline_408.py", "scripts/build_acceptance_plan.py", "scripts/acceptance_gate.py"):
        pin(ROOT / script)
    reports = {"original_single": loaded["single_block"], "old_full": loaded["baseline"], "old_full_repeat": loaded["repeat"]}
    data, locations, aggregate = {}, {}, {}
    for role, children in reports.items():
        checked = validate_render_reports(plan, children, whole_model=role != "original_single")
        data[role] = checked
        locations[role] = {frame_key(row): child for child in children for row in child["frames"]}
        summary_path = Path(manifest[{"original_single": "single_block", "old_full": "baseline", "old_full_repeat": "repeat"}[role]][0])
        summary = read(summary_path)
        require(summary.get("complete") is True and summary.get("status") == "evaluated"
                and summary.get("images") == summary.get("expected_images") == 408, "Incomplete scene summary")
        canonical_rows = [checked["rows"][frame_key(f)] for f in plan["frames"]]
        require(bits_equal(summary["frames"], canonical_rows), "Summary rows differ from exact canonical child records")
        require(summary["identity"]["plan_sha256"] == source_hashes[str(plan_path)], "Summary plan file differs")
        aggregate[role] = {p: scores(canonical_rows, p) for p in PROTOCOLS}
        for protocol in PROTOCOLS:
            for key, value in aggregate[role][protocol].items():
                require(math.isclose(value, summary["protocols"][protocol][key], rel_tol=0, abs_tol=1e-12),
                        "Scene summary metric does not reproduce: " + role + "/" + protocol + "/" + key)
    paired(data["original_single"], data["old_full"])
    paired(data["old_full"], data["old_full_repeat"])
    require(data["old_full"]["models"] == data["old_full_repeat"]["models"], "Repeat model differs")

    historical_delivery = read(ROOT / "reports/CURRENT_ROUND_DELIVERY_20261007.json")
    historical = {}
    historical_observations = 0
    for bid in [j["block_id"] for j in plan["jobs"]]:
        report = read(ROOT / "runs/full_scene_core_all_v1_final_dev" / (bid + "_candidate") / "metrics.json")
        rows = {frame_key(r): r for r in report["frames"]}
        require(len(rows) == report["images"] == len(report["frames"]), "Duplicate historical block record")
        historical_observations += len(rows)
        for key, row in rows.items():
            historical.setdefault(key, row)
    require(set(historical) == set(data["old_full"]["rows"]), "Historical unique frame set differs")

    frame_results = []
    float_fields = {name: {"arrays_compared": 0, "bitexact_arrays": 0, "numeric_array_equal": 0,
                          "finite_arrays": 0, "values_per_run": 0, "max_abs_difference": 0.}
                    for name in ("rgb", "alpha", "depth")}
    archives_identical = distinct_inodes = timings_different = total_float_bytes = 0
    mask_cache = {}
    recomputed_rgb_max_mse_diff = recomputed_rgb_max_psnr_diff = 0.
    for position, frozen in enumerate(plan["frames"]):
        key = frame_key(frozen)
        rows = {role: data[role]["rows"][key] for role in DATASETS}
        old, repeat = rows["old_full"], rows["old_full_repeat"]
        comparison = {p: {"psnr_delta_old_full_minus_single_db": old["protocols"][p]["psnr_db"] - rows["original_single"]["protocols"][p]["psnr_db"],
                          "repeat_psnr_abs_difference_db": abs(old["protocols"][p]["psnr_db"] - repeat["protocols"][p]["psnr_db"]),
                          "repeat_all_metrics_bitexact": bits_equal(old["protocols"][p], repeat["protocols"][p])}
                      for p in PROTOCOLS}
        paths = {role: Path(locations[role][key]["_evidence_path"]).parent / ("%03d_float.npz" % frozen["dataset_index"])
                 for role in ("old_full", "old_full_repeat")}
        for path in paths.values():
            require(not path.is_symlink(), "Repeat float artifact is a symlink")
        checksums = {role: pin(path) for role, path in paths.items()}
        total_float_bytes += sum(path.stat().st_size for path in paths.values())
        archives_identical += checksums["old_full"] == checksums["old_full_repeat"]
        distinct_inodes += state(paths["old_full"])[:2] != state(paths["old_full_repeat"])[:2]
        timings_different += not bits_equal(old["render_seconds"], repeat["render_seconds"])
        require(paths["old_full"] != paths["old_full_repeat"], "Repeat reuses same float file")
        field_proof = {}
        with np.load(paths["old_full"], allow_pickle=False) as first, np.load(paths["old_full_repeat"], allow_pickle=False) as second:
            require(set(first.files) == set(second.files) == set(float_fields), "Missing/extra float output field")
            for name in float_fields:
                x, y = first[name], second[name]
                expected = (frozen["camera"]["h"], frozen["camera"]["w"])
                require(x.shape == y.shape and (x.shape == (*expected, 3) if name == "rgb" else x.shape in (expected, (*expected, 1))),
                        "Float array dimensions differ")
                require(x.dtype == y.dtype and x.dtype == np.float32, "Expected original float32 renderer output")
                finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
                require(finite, "Nonfinite float renderer array")
                sha_a, sha_b = (hashlib.sha256(z.tobytes(order="C")).hexdigest() for z in (x, y))
                byte_equal = sha_a == sha_b
                numeric_equal = bool(np.array_equal(x, y))
                difference = 0. if byte_equal else float(np.max(np.abs(x.astype(np.float64) - y)))
                stats = float_fields[name]
                stats["arrays_compared"] += 1; stats["bitexact_arrays"] += byte_equal
                stats["numeric_array_equal"] += numeric_equal; stats["finite_arrays"] += finite
                stats["values_per_run"] += int(x.size)
                stats["max_abs_difference"] = max(stats["max_abs_difference"], difference)
                field_proof[name] = {"dtype": x.dtype.str, "shape": list(x.shape), "old_full_array_sha256": sha_a,
                                     "repeat_array_sha256": sha_b, "bitexact": byte_equal, "max_abs_difference": difference}
            # Confirm stored metrics really score these unquantized RGB/alpha
            # arrays against frozen GT/masks; SSIM is compared across runs.
            gt = cv2.imread(frozen["source_image_path"], cv2.IMREAD_COLOR)
            require(gt is not None, "Cannot decode frozen GT")
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            pred, alpha = first["rgb"].clip(0., 1.), first["alpha"].reshape(gt.shape[:2])
            masks = {"raw": np.ones(gt.shape[:2], bool)}
            for protocol, field in (("legacy", "source_mask_path"), ("ego_v2", "mask_path")):
                path = frozen[field]
                mask_key = plan["inputs"][path]
                if mask_key not in mask_cache:
                    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    require(mask is not None, "Cannot decode fixed RGB mask")
                    mask_cache[mask_key] = mask > 0
                masks[protocol] = mask_cache[mask_key]
            for protocol, valid in masks.items():
                mse = float(np.mean((gt[valid] - pred[valid]) ** 2))
                psnr = float(-10 * np.log10(max(mse, 1e-12)))
                recorded = old["protocols"][protocol]
                mse_diff, psnr_diff = abs(mse-recorded["rgb_mse"]), abs(psnr-recorded["psnr_db"])
                recomputed_rgb_max_mse_diff = max(recomputed_rgb_max_mse_diff, mse_diff)
                recomputed_rgb_max_psnr_diff = max(recomputed_rgb_max_psnr_diff, psnr_diff)
                require(mse_diff <= 1e-12 and psnr_diff < 1e-6, "Metrics are not scores of the stored float RGB")
                require(int(valid.sum()) == recorded["rgb_valid_pixels"] and
                        int(((alpha > .5) & valid).sum()) == recorded["alpha_gt_0_5_valid_pixels"], "Stored float alpha/mask counts differ")
        frame_results.append({"group_id": key[0], "camera_id": key[1], "block_id": frozen["block_id"],
            "floor_id": frozen["floor_id"], "floor_verified": False, "dataset_index": frozen["dataset_index"],
            "metrics": {role: row["protocols"] for role, row in rows.items()}, "comparison": comparison,
            "native_visibility_bitexact": bits_equal(old["native_visibility"], repeat["native_visibility"]),
            "legacy_historical_psnr_abs_difference_db": abs(old["protocols"]["legacy"]["psnr_db"] - historical[key]["psnr_db"]),
            "float_artifacts": {role: {"path": str(path), "sha256": checksums[role], "stat": states[str(path)]}
                                for role, path in paths.items()}, "float_array_comparison": field_proof})
        if (position + 1) % 32 == 0 or position + 1 == len(plan["frames"]):
            print(json.dumps({"audited_float_frame_pairs": position + 1, "expected": 408}), flush=True)

    def grouped(field):
        result = []
        for name in dict.fromkeys(f[field] for f in frame_results):
            items = [f for f in frame_results if f[field] == name]
            measures = {}
            for p in PROTOCOLS:
                deltas = [f["comparison"][p]["psnr_delta_old_full_minus_single_db"] for f in items]
                measures[p] = {"original_single_psnr_db_mean": statistics.mean(f["metrics"]["original_single"][p]["psnr_db"] for f in items),
                               "old_full_psnr_db_mean": statistics.mean(f["metrics"]["old_full"][p]["psnr_db"] for f in items),
                               "old_full_minus_single_db": distribution(deltas),
                               "loss_single_minus_old_full_db": -statistics.mean(deltas),
                               "g6_old_baseline_diagnostic": "PASS" if -statistics.mean(deltas) <= 1. else "FAIL"}
            result.append({field: name, "images": len(items), "floor_verified": False, "protocols": measures})
        return result
    blocks, floors = grouped("block_id"), grouped("floor_id")
    failed = [b["block_id"] for b in blocks if b["protocols"]["ego_v2"]["g6_old_baseline_diagnostic"] == "FAIL"]
    f1_ids = plan["gate_thresholds"]["G2"]["f1_blocks"]
    f1_failed = [bid for bid in failed if bid in f1_ids]
    repeats = {p: {"frames": 408,
                   "all_metric_fields_bitexact_frames": sum(f["comparison"][p]["repeat_all_metrics_bitexact"] for f in frame_results),
                   "max_abs_psnr_difference_db": max(f["comparison"][p]["repeat_psnr_abs_difference_db"] for f in frame_results)}
               for p in PROTOCOLS}
    repeat_pass = all(r["max_abs_psnr_difference_db"] < 1e-6 for r in repeats.values())
    closure_path = ROOT / "reports/pixel_closure_v2.json"
    closure = read(closure_path)
    closure_sources = {}
    for path, sha in closure["source_sha256"].items():
        actual = pin(ROOT / path); closure_sources[path] = actual == sha
    closure_pass = (closure.get("status") == "passed" and closure.get("pytest_returncode") == 0
                    and closure.get("wrong_index_count") == 0 and closure.get("foreground_array_equal_fraction") == 1.
                    and closure.get("tested_index_count", 0) > 0 and bool(closure_sources) and all(closure_sources.values()))
    stale_closure_sources = [{"path": str(ROOT / path), "recorded_sha256": closure["source_sha256"][path],
                             "actual_sha256": source_hashes[str(ROOT / path)]}
                            for path, match in closure_sources.items() if not match]
    closure_status = "PASS" if closure_pass else "STALE_SOURCE" if stale_closure_sources else "FAIL"
    historical_mean = historical_delivery["evaluation_unique_images"]["dev"]["psnr_db_frame_mean"]
    result = {"schema_version": 1, "report_type": "acceptance_baseline_v2_408_cpu_audit", "audit_status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "baseline only; no new candidate; not a complete rollout gate report",
        "high_fidelity_accepted": False, "rollout_allowed": False, "new_candidate_evaluated": False,
        "full_gate_evaluated": False, "gpu_used_by_audit": False, "test_split_read": False,
        "acceptance_plan_id": plan["plan_id"], "acceptance_plan_sha256": source_hashes[str(plan_path)],
        "identity_audit": {"unique_frames_per_dataset": 408, "blocks_per_dataset": 28,
            "duplicate_frames": 0, "missing_frames": 0, "extra_frames": 0, "three_protocol_fixed_denominators_paired": True,
            "source_assets_sha256_verified": True, "summary_rows_exactly_match_canonical_children": True,
            "primary_protocol": "ego_v2", "floor_labels": "historical collection grouping; floor_verified=false",
            "whole_ply_gaussians": 6780816, "whole_ply_sha256": next(iter(data["old_full"]["models"].values()))},
        "dataset_protocols": aggregate,
        "G1_baseline": {"status": "FAIL" if not repeat_pass else "PASS" if closure_pass else "BLOCKED",
            "reproducibility_status": "PASS" if repeat_pass else "FAIL", "repeat_protocols": repeats,
            "status_explanation": "Renderer reproducibility is evaluated separately; stale pixel-closure source binding cannot pass the broader G1 contract.",
            "strict_psnr_tolerance_db": 1e-6, "pixel_closure_status": closure_status,
            "pixel_closure_report": str(closure_path), "pixel_closure_source_hash_matches": closure_sources,
            "pixel_closure_stale_sources": stale_closure_sources,
            "native_visibility_bitexact_frames": sum(f["native_visibility_bitexact"] for f in frame_results),
            "float_array_fields": float_fields, "bitidentical_npz_archives": archives_identical,
            "npz_file_pairs_with_distinct_inodes": distinct_inodes, "frames_with_distinct_recorded_render_times": timings_different,
            "float_archive_bytes_read_and_hashed": total_float_bytes,
            "rgb_recomputed_from_float": {"frames": 408, "protocols_per_frame": 3,
                "max_abs_mse_difference": recomputed_rgb_max_mse_diff, "max_abs_psnr_difference_db": recomputed_rgb_max_psnr_diff,
                "alpha_valid_pixel_counts_exact": True, "ssim_recomputed": False},
            "execution_provenance": "Two separately recorded native evaluator runs, distinct output directories/inodes and actual per-frame render timings; CPU audit reuses saved float32 outputs, not PNG rescoring or a third GPU render.",
            "bitexact_definition": "All parsed scalar metrics compared with type-aware float64 IEEE bits; rgb/alpha/depth compared by dtype,shape,C-order payload SHA256 and array_equal. JSON report files may differ in timing/path metadata."},
        "historical_legacy_recovery": {"historical_observations": historical_observations, "historical_unique_frames": len(historical),
            "delivery_reported_legacy_psnr_db": historical_mean, "fresh_legacy_psnr_db": aggregate["old_full"]["legacy"]["psnr_db_mean"],
            "mean_abs_difference_db": abs(aggregate["old_full"]["legacy"]["psnr_db_mean"] - historical_mean),
            "max_per_frame_psnr_difference_db": max(f["legacy_historical_psnr_abs_difference_db"] for f in frame_results),
            "correction": "Historical 18.1037500788 dB already uses 408 unique frames; 492 is observation count, not the denominator of that mean."},
        "G6_old_baseline_diagnostic": {"status": "FAIL" if failed else "PASS", "loss_limit_db": 1.,
            "all28_failed_blocks": failed, "f1_failed_blocks": f1_failed,
            "all_frame_ego_psnr_loss_db": aggregate["original_single"]["ego_v2"]["psnr_db_mean"] - aggregate["old_full"]["ego_v2"]["psnr_db_mean"],
            "ego_block_mean_loss_distribution_db": distribution([b["protocols"]["ego_v2"]["loss_single_minus_old_full_db"] for b in blocks]),
            "worst_ego_block": max(blocks, key=lambda b: b["protocols"]["ego_v2"]["loss_single_minus_old_full_db"])["block_id"],
            "scope_limit": "Old assembled core_joint map vs original garage_surface single blocks; both source training and assembly differ. Paired loss is observed, not an isolated causal ownership ablation or a new candidate G6 result."},
        "by_block": blocks, "by_floor": floors, "frames": frame_results,
        "training_provenance_note": "source_ply is a legitimate exported-model evaluation origin and does not imply untrained. This audit does not re-verify checkpoint training/export authenticity or assess G8.",
        "source_sha256": source_hashes,
        "reproduce_command": ".venv-train/bin/python -B scripts/audit_acceptance_baseline_408.py --output-json reports/acceptance_baseline_v2_408.json --output-md reports/acceptance_baseline_v2_408.md"}
    for path, before in states.items():
        require(state(path) == before, "Input changed during baseline audit: " + path)
    require(sha256_file(plan_path) == result["acceptance_plan_sha256"], "Frozen plan changed during audit")
    result["audit_id"] = object_sha(result)
    json_text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    md_text = markdown(result)
    for path in (output_json, output_md):
        require(not path.exists(), "Refuse overwrite: " + str(path))
        require(ROOT in path.resolve().parents, "Output escapes project")
    output_json.write_text(json_text); output_md.write_text(md_text)
    return result


def markdown(d):
    lines = ["# 固定408帧基线审计", "", "**仅基线审计；没有评估新候选，也没有完成 G1–G8 全部放量验收。**",
             "", "三套数据均完整覆盖固定 408 张 DEV、28 块，重复/缺失/额外帧均为0；相机、掩码、资产 SHA 和三口径分母严格配对。",
             "", "| 口径 | 原单块 PSNR | 旧整图 PSNR | 旧整图重复 PSNR | 旧整图相对单块 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for p in PROTOCOLS:
        a, b, c = [d["dataset_protocols"][r][p]["psnr_db_mean"] for r in DATASETS]
        lines.append(f"| {p} | {a:.9f} | {b:.9f} | {c:.9f} | {b-a:+.9f} dB |")
    g1 = d["G1_baseline"]
    lines += ["", f"G1 的渲染复现子项：**{g1['reproducibility_status']}**；包含当前源码 S0 证据的完整 G1 状态：**{g1['status']}**。三口径逐帧 PSNR 最大差均为 " +
              ", ".join(f"{p}={g1['repeat_protocols'][p]['max_abs_psnr_difference_db']:.12g}" for p in PROTOCOLS) + " dB（要求严格小于1e−6）。",
              f"408对 RGB/alpha/depth 浮点输出逐数组比较；各字段位一致数组数：" +
              ", ".join(f"{k}={v['bitexact_arrays']}/408" for k,v in g1['float_array_fields'].items()) + "。",
              f"NPZ 文件本身 SHA 相同 {g1['bitidentical_npz_archives']}/408；独立 inode {g1['npz_file_pairs_with_distinct_inodes']}/408；记录的渲染耗时不同 {g1['frames_with_distinct_recorded_render_times']}/408。",
              "审计还从全部408张 float32 RGB 与冻结 GT/掩码重新计算三口径 MSE、PSNR，并核对 alpha 像素计数；未重算 SSIM，但逐帧完整指标（含 SSIM/深度）已做位比较。",
              f"CPU 像素闭合证据及源码哈希核验：{g1['pixel_closure_status']}。" +
              ("绑定已变化的文件为 " + ", ".join("`" + s["path"] + "`" for s in g1["pixel_closure_stale_sources"]) + "；不能沿用旧 S0 PASS，详细预期/实际 SHA 见 JSON。" if g1["pixel_closure_stale_sources"] else "") +
              "本次只读取两个既有原生渲染的浮点结果，没有用8-bit PNG代替重复渲染，也没有启动第三次 GPU 渲染。",
              "", "历史 legacy 的 **18.10375007882831 dB 本来就已去重为408帧**；492是跨块观测条数。新 legacy 均值与历史报告差 " +
              f"{d['historical_legacy_recovery']['mean_abs_difference_db']:.3g} dB，逐帧 PSNR 最大差 {d['historical_legacy_recovery']['max_per_frame_psnr_difference_db']:.3g} dB。",
              "", "## G6：旧整图相对原单块", "", f"ego_v2 按408帧平均损失 **{d['G6_old_baseline_diagnostic']['all_frame_ego_psnr_loss_db']:.6f} dB**；" +
              f"28块中 {len(d['G6_old_baseline_diagnostic']['all28_failed_blocks'])} 块平均损失超过1 dB，F1五块中 {len(d['G6_old_baseline_diagnostic']['f1_failed_blocks'])} 块超限。",
              "这是旧整图的 G6 诊断。原单块与整图的训练来源及装配同时不同，不能把全部差值归因于 ownership，也不能当作新候选放量结果。",
              "", "| 块 | 帧数 | 单块 ego PSNR | 旧整图 ego PSNR | 损失 dB | ≤1 dB |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in d["by_block"]:
        v=row["protocols"]["ego_v2"]
        lines.append(f"| {row['block_id']} | {row['images']} | {v['original_single_psnr_db_mean']:.6f} | {v['old_full_psnr_db_mean']:.6f} | {v['loss_single_minus_old_full_db']:.6f} | {v['g6_old_baseline_diagnostic']} |")
    lines += ["", "楼层/坡道仍使用历史采集标签，`floor_verified=false`；下表是分组统计，不表示真实楼层身份独立核验通过。", "",
              "| 历史区域 | 帧数 | 单块 ego PSNR | 旧整图 ego PSNR | 损失 dB |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in d["by_floor"]:
        v=row["protocols"]["ego_v2"]
        lines.append(f"| {row['floor_id']} | {row['images']} | {v['original_single_psnr_db_mean']:.6f} | {v['old_full_psnr_db_mean']:.6f} | {v['loss_single_minus_old_full_db']:.6f} |")
    lines += ["", "全部逐帧三口径指标、配对差值、数组哈希、按块/楼层三口径结果和源文件 SHA 均在同名 JSON。",
              "`source_ply` 是合法的导出模型评估来源，本审计不据此认定未训练，也未重新核验 G8。几何深度仍基于同源 LIO 条件参考，不能冒充独立真值。",
              "", "## 复现", "", "```bash", d["reproduce_command"], "```", "",
              "输出采用新文件写入，重跑须换输出路径。只用 CPU，逐帧加载浮点数组；没有修改冻结计划、模型、数据集或完整闸门实现。",
              "", f"审计 ID：`{d['audit_id']}`。", f"计划 ID：`{d['acceptance_plan_id']}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports/acceptance_baseline_v2_408.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports/acceptance_baseline_v2_408.md")
    args = parser.parse_args()
    for path in (args.output_json, args.output_md):
        require(not path.exists(), "Refuse existing report output")
    result = audit(args.output_json.resolve(), args.output_md.resolve())
    print(json.dumps({"audit_status": result["audit_status"], "G1_baseline": result["G1_baseline"]["status"],
                      "G6_old_baseline_diagnostic": result["G6_old_baseline_diagnostic"]["status"],
                      "ego_psnr_loss_db": result["G6_old_baseline_diagnostic"]["all_frame_ego_psnr_loss_db"]}))


if __name__ == "__main__":
    main()

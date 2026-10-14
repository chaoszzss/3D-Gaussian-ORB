#!/usr/bin/env python3
"""Reproduce the bounded block000 comparison on CPU; creates three new reports.

Run from the project: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=3
OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 .venv-train/bin/python -B
reports/block000_seed_aa_cap003_audit.py
"""
import csv
import datetime
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "3"
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
import numpy as np
import cv2
import acceptance_gate as G
from build_acceptance_plan import validate_frame_source
import measure_surface_quality as S
from skimage.metrics import structural_similarity
from threadpoolctl import threadpool_limits, threadpool_info
import torch

# The shared geometry module's import sets environment defaults to four.
# Restore three and constrain already-loaded pools before doing audit work.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "3"
threadpool_limits(limits=3)
torch.set_num_threads(3)
torch.set_num_interop_threads(1)
cv2.setNumThreads(3)

PREFIX = ROOT / "reports/block000_seed_aa_cap003"
RUN = ROOT / "runs/fidelity_v2/train_block000_depthv3_seed_aa_cap003"
JOB = ROOT / "runs/fidelity_v2/jobs/block000_depthv3_seed_aa_cap003"
PROTOCOLS = ("raw", "legacy", "ego_v2")
CACHE = {}
SNAPSHOTS = {}


def state(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def pin(path, expected=None):
    path = (ROOT / path).resolve()
    before = state(path)
    SNAPSHOTS.setdefault(str(path), before)
    assert SNAPSHOTS[str(path)] == before, "Asset changed during audit: " + str(path)
    sha = expected or CACHE.get("_pins", {}).get(str(path)) or G.sha256_file(path)
    G.verify_artifact(path, sha, CACHE)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha}


def read(path):
    descriptor = pin(path)
    return json.loads((ROOT / descriptor["path"]).read_text()), descriptor


def near(actual, expected, label, tolerance=1e-7):
    assert actual is not None and math.isfinite(actual)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), (label, actual, expected)


def model_and_frame_identity(report, plan, whole):
    """Audit saved inputs even when the current live-code backend check is stale.

    This is deliberately separate from gate validation; no stale-code failure
    is converted into a gate PASS or a substituted integration proof.
    """
    from verify_training_checkpoint import read_config
    for name in ("config", "checkpoint", "transforms"):
        pin(report[name], report[name + "_sha256"])
    key = ("audit_checkpoint", report["checkpoint_sha256"])
    if key not in CACHE: CACHE[key] = G.checkpoint_metadata(Path(report["checkpoint"]))
    stored = CACHE[key]
    assert report["step"] == report["inference_model_step"] == stored["step"]
    model = read_config(report["config"])["pipeline"]["model"]
    assert report["inference_sh_degree"] == min(stored["step"] // model["sh_degree_interval"], model["sh_degree"])
    assert report["backend"] == {"original": "hbb1-original-2dgs", "aa-depth-vjp": "hbb1-aa-depth-vjp-2dgs"}[model.get("inria_backend", "original")]
    assert report["original_checkpoint_gaussians"] == stored["gaussians"]
    if report["source_ply"] is None:
        assert report["gaussians"] == stored["gaussians"] and report["parameter_origin"] == "trained_checkpoint"
        assert report["checkpoint_is_template"] is False and not whole
    else:
        assert report["checkpoint_is_template"] is True
    identity = G.validate_representation(report, CACHE)
    assert report["status"] == "evaluated" and report["split"] == "dev" and report["metric_domain"] == "float_renderer"
    assert report["acceptance_plan_id"] == plan["plan_id"] and report["world_parser_scale"] == 1.
    assert np.array_equal(np.array(report["world_parser_transform"]), np.eye(4)[:3])
    frozen = [f for f in plan["frames"] if f["block_id"] == "block_000"]
    assert len(report["frames"]) == report["images"] == len(frozen) == 20
    for row, fixed in zip(report["frames"], frozen):
        assert G.frame_key(row) == G.frame_key(fixed) and row["render_seconds"] > 0
        for name in ("dataset_index", "file_path", "source_image_path", "source_mask_path", "source_lidar_depth_path", "source_lidar_support_path"):
            assert row[name] == fixed[name], name
        assert row["acceptance_mask_path"] == fixed["mask_path"]
        assert set(row["protocols"]) == set(PROTOCOLS)
        for name, value in fixed["camera"].items():
            assert np.allclose(row["camera"][name], value, rtol=1e-6, atol=1e-5), name
        if whole:
            v = row["native_visibility"]
            assert v["protocol"] == G.NATIVE_PROTOCOL and v["source_records"] == report["gaussians"]
            assert 0 <= v["active_records"] <= report["gaussians"]
            G.digest(v["active_index_sha256"])
    return {"rows": {G.frame_key(row): row for row in report["frames"]},
            "models": {G.frame_key(row): identity for row in report["frames"]}}


def rescore(path, report, frozen):
    """Independent arithmetic from saved native floats, never PNG predictions."""
    results, maxima = [], {"rgb_mse": 0., "psnr_db": 0., "ssim": 0., "depth_capped_error": 0.}
    for row, fixed in zip(report["frames"], frozen):
        array_path = path.parent / ("%03d_float.npz" % row["dataset_index"])
        desc = pin(array_path)
        with np.load(array_path, allow_pickle=False) as saved:
            assert set(saved.files) == {"rgb", "alpha", "depth"}
            pred = saved["rgb"].clip(0., 1.)
            alpha, depth = (saved[k].squeeze() for k in ("alpha", "depth"))
        assert pred.dtype == np.float32 and pred.shape == (fixed["camera"]["h"], fixed["camera"]["w"], 3)
        assert np.isfinite(pred).all() and np.isfinite(alpha).all()
        gt = cv2.cvtColor(cv2.imread(row["source_image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.)
        legacy = cv2.imread(row["source_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        ego = cv2.imread(row["acceptance_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        masks = {"raw": np.ones(legacy.shape, bool), "legacy": legacy, "ego_v2": legacy & ego}
        reference = np.load(row["source_lidar_depth_path"], allow_pickle=False).squeeze()
        support = cv2.imread(row["source_lidar_support_path"], cv2.IMREAD_GRAYSCALE) > 0
        _, ssim_map = structural_similarity(gt, pred, channel_axis=-1, data_range=1., full=True)
        ssim_map = ssim_map.mean(axis=2)
        for name, mask in masks.items():
            score = row["protocols"][name]
            mse = float(np.mean((gt[mask] - pred[mask]) ** 2))
            psnr = float(-10 * np.log10(max(mse, 1e-12)))
            eroded = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            ssim = float(ssim_map[eroded].mean())
            eligible = mask & support & np.isfinite(reference) & (reference > 0)
            compared = eligible & (alpha > .5) & np.isfinite(depth) & (depth > 0)
            capped = np.ones(int(eligible.sum()), np.float64)
            capped[compared[eligible]] = np.minimum(np.abs(depth[compared] - reference[compared]) / reference[compared], 1.)
            depth_error = float(capped.mean())
            for actual, expected, key in ((mse, score["rgb_mse"], "rgb_mse"), (psnr, score["psnr_db"], "psnr_db"),
                    (ssim, score["ssim_valid_pixels"], "ssim"),
                    (depth_error, score["map_depth_capped_relative_error_all_supported"], "depth_capped_error")):
                near(actual, expected, name + ":" + key)
                maxima[key] = max(maxima[key], abs(actual - expected))
            for actual, key in ((mask.sum(), "rgb_valid_pixels"), (eroded.sum(), "ssim_compared_pixels"),
                    (eligible.sum(), "map_depth_eligible_pixels"), (compared.sum(), "map_depth_compared_pixels"),
                    ((eligible & ~compared).sum(), "map_depth_missing_pixels"), ((mask & (alpha > .5)).sum(), "alpha_gt_0_5_valid_pixels")):
                assert int(actual) == score[key], (name, key, int(actual), score[key])
        results.append({**desc, "group_id": str(row["group_id"]), "camera_id": row["camera_id"],
                        "dataset_index": row["dataset_index"], "shape": list(pred.shape), "dtype": str(pred.dtype)})
    return {"status": "VERIFIED_FROM_SAVED_NATIVE_FLOATS", "frames": len(results),
        "protocol_frame_scores": len(results) * 3, "max_absolute_metric_difference": maxima,
        "checked": ["RGB MSE/PSNR", "7x7 valid-window SSIM", "RGB/SSIM/fixed-depth denominators",
                    "alpha coverage counts", "fixed-support capped relative depth error; missing=1"],
        "new_renderer_execution": False, "new_repeat_run": False, "float_artifacts": results}


def compare(left, right, label, size, records):
    a, b = records[left], records[right]
    G.paired(a["validated"], b["validated"])
    rows, summaries, cameras = [], {}, {}
    for protocol in PROTOCOLS:
        selected = []
        for ca, cb in zip(a["report"]["frames"], b["report"]["frames"]):
            assert G.frame_key(ca) == G.frame_key(cb)
            for field in ("camera", "dataset_index", "file_path", "source_image_path", "source_mask_path",
                          "source_lidar_depth_path", "source_lidar_support_path", "acceptance_mask_path"):
                assert ca[field] == cb[field], field
            x, y = ca["protocols"][protocol], cb["protocols"][protocol]
            selected.append(dict(comparison=label, width=size, height=ca["camera"]["h"], protocol=protocol,
                group_id=str(ca["group_id"]), camera_id=ca["camera_id"], dataset_index=ca["dataset_index"],
                candidate_psnr_db=x["psnr_db"], baseline_psnr_db=y["psnr_db"], paired_delta_db=x["psnr_db"]-y["psnr_db"],
                candidate_ssim=x["ssim_valid_pixels"], baseline_ssim=y["ssim_valid_pixels"],
                paired_ssim_delta=x["ssim_valid_pixels"]-y["ssim_valid_pixels"],
                rgb_valid_pixels=x["rgb_valid_pixels"], ssim_compared_pixels=x["ssim_compared_pixels"],
                depth_eligible_pixels=x["map_depth_eligible_pixels"], candidate_depth_compared_pixels=x["map_depth_compared_pixels"],
                baseline_depth_compared_pixels=y["map_depth_compared_pixels"],
                candidate_depth_capped_error=x["map_depth_capped_relative_error_all_supported"],
                baseline_depth_capped_error=y["map_depth_capped_relative_error_all_supported"]))
        delta = [r["paired_delta_db"] for r in selected]
        summaries[protocol] = dict(images=len(selected), candidate_psnr_db=statistics.mean(r["candidate_psnr_db"] for r in selected),
            baseline_psnr_db=statistics.mean(r["baseline_psnr_db"] for r in selected), paired_delta_db_mean=statistics.mean(delta),
            paired_delta_db_median=statistics.median(delta), paired_delta_db_min=min(delta), paired_delta_db_max=max(delta),
            frames_improved=sum(d > 0 for d in delta), frames_below_minus_1db=sum(d < -1 for d in delta),
            candidate_ssim_mean=statistics.mean(r["candidate_ssim"] for r in selected),
            baseline_ssim_mean=statistics.mean(r["baseline_ssim"] for r in selected),
            paired_ssim_delta_mean=statistics.mean(r["paired_ssim_delta"] for r in selected),
            fixed_rgb_pixels=sum(r["rgb_valid_pixels"] for r in selected), fixed_depth_pixels=sum(r["depth_eligible_pixels"] for r in selected),
            candidate_depth_coverage=sum(r["candidate_depth_compared_pixels"] for r in selected)/sum(r["depth_eligible_pixels"] for r in selected),
            baseline_depth_coverage=sum(r["baseline_depth_compared_pixels"] for r in selected)/sum(r["depth_eligible_pixels"] for r in selected),
            candidate_depth_capped_error_pixel_mean=sum(r["candidate_depth_capped_error"]*r["depth_eligible_pixels"] for r in selected)/sum(r["depth_eligible_pixels"] for r in selected),
            baseline_depth_capped_error_pixel_mean=sum(r["baseline_depth_capped_error"]*r["depth_eligible_pixels"] for r in selected)/sum(r["depth_eligible_pixels"] for r in selected))
        cameras[protocol] = {camera: {"frames": sum(r["camera_id"] == camera for r in selected),
            "paired_delta_db_mean": statistics.mean(r["paired_delta_db"] for r in selected if r["camera_id"] == camera)}
            for camera in sorted({r["camera_id"] for r in selected})}
        rows.extend(selected)
    return {"comparison": label, "width": size, "images": len(a["report"]["frames"]),
        "candidate": a["descriptor"], "baseline": b["descriptor"], "acceptance_plan_id": a["report"]["acceptance_plan_id"],
        "primary_protocol": "ego_v2", **summaries["ego_v2"], "protocols": summaries, "by_camera": cameras,
        "pairing": "Exact ordered frozen block000 DEV group/camera/index; all source assets/cameras/masks/teacher and scoring denominators identical within each resolution",
        "rows": rows}


def main():
    start = time.monotonic()
    outputs = [Path(str(PREFIX) + suffix) for suffix in ("_paired_comparison.json", "_paired_comparison.csv", "_decision.json")]
    assert not any(p.exists() for p in outputs), "Refuse to overwrite completed audit reports"
    for name in ("scripts/acceptance_gate.py", "scripts/build_acceptance_plan.py", "scripts/measure_surface_quality.py",
                 "scripts/verify_training_checkpoint.py", "scripts/evaluate_gaussian_ply.py", "runtime/finish_block000_depthv3_seeded.py", __file__):
        pin(name)
    plan_paths = {640: "reports/acceptance_plan_v1.json", 1280: "reports/acceptance_resolution_depthv3_plan.json"}
    plans = {size: G.load_plan(ROOT/path, verify_assets=False) for size, path in plan_paths.items()}
    plan_audits = {}
    for size, plan in plans.items():
        frozen = [f for f in plan["frames"] if f["block_id"] == "block_000"]
        assert len(frozen) == 20 and len({G.frame_key(f) for f in frozen}) == 20
        transforms, td = read(frozen[0]["transforms"])
        index = {f["file_path"]: f for f in transforms["frames"]}
        for frame in frozen:
            assert td["sha256"] == frame["transforms_sha256"]
            assert transforms["val_filenames"][frame["dataset_index"]] == frame["file_path"]
            validate_frame_source(frame, index[frame["file_path"]], frame["dataset_index"])
            for field, sha in frame["asset_sha256"].items(): pin(frame[field], sha)
        plan_audits[str(size)] = {**pin(plan_paths[size]), "plan_id": plan["plan_id"], "plan_frames": len(plan["frames"]),
            "audited_frames": 20, "mask_revision": plan["mask_revision"], "floor_id_verified": False,
            "status": "VERIFIED_PLAN_DIGEST_AND_BLOCK000_SOURCE_ASSETS", "other_block_assets_rehashed": False}
    records = {}
    sources = [("candidate_640", "block000_seed_aa_cap003_acceptance", 640, False),
        ("original_640", "original000_refreshed_v2", 640, False), ("oldwhole_640", "old_all28_block000_refreshed_v2", 640, True),
        ("candidate_1280", "block000_seed_aa_cap003_resolution", 1280, False),
        ("original_1280", "original000_resolution_depthv3", 1280, False)]
    for label, directory, size, whole in sources:
        path = ROOT / "runs/fidelity_v2" / directory / "metrics.json"
        report, descriptor = read(path)
        assert report["acceptance"]["sha256"] == plan_audits[str(size)]["sha256"]
        validated = model_and_frame_identity(report, plans[size], whole)
        try:
            G.validate_render_reports(plans[size], [report], ["block_000"], whole_model=whole, asset_cache=CACHE)
            current_gate = {"status": "PASS"}
        except ValueError as exc:
            current_gate = {"status": "FAIL_CURRENT_LIVE_SOURCE_PROVENANCE", "reason": str(exc)}
        fixed = [f for f in plans[size]["frames"] if f["block_id"] == "block_000"]
        assert [G.frame_key(f) for f in report["frames"]] == [G.frame_key(f) for f in fixed]
        audit = rescore(path, report, fixed)
        for protocol in PROTOCOLS:
            near(statistics.mean(r["protocols"][protocol]["psnr_db"] for r in report["frames"]), report["protocols"][protocol]["psnr_db_mean"], "aggregate PSNR")
        records[label] = {"report": report, "descriptor": descriptor, "validated": validated, "audit": audit, "current_gate": current_gate}
        print(label, "verified", audit["frames"], "native float frames", flush=True)
    candidate = records["candidate_640"]["report"]
    assert candidate["checkpoint_sha256"] == records["candidate_1280"]["report"]["checkpoint_sha256"]
    assert candidate["step"] == candidate["inference_model_step"] == 14999
    expected_original = pin("runs/garage_surface_v1/block_000/eval_dev/gaussians.ply")
    expected_whole = pin("maps/garage_core_all_v1/gaussians_full.ply")
    assert records["original_640"]["report"]["source_ply_sha256"] == records["original_1280"]["report"]["source_ply_sha256"] == expected_original["sha256"]
    assert records["oldwhole_640"]["report"]["source_ply_sha256"] == expected_whole["sha256"]
    comparisons = [compare("candidate_640", "original_640", "block_000_640_vs_original_single_refreshed", 640, records),
        compare("candidate_640", "oldwhole_640", "block_000_640_vs_old_whole_scene_refreshed", 640, records),
        compare("candidate_1280", "original_1280", "block_000_1280_vs_original_single", 1280, records)]
    proof, proof_desc = read("reports/block000_seed_aa_cap003_checkpoint.json")
    geometry, geometry_desc = read("reports/surface_quality_v2_block000_seed_aa_cap003.json")
    try:
        G.validate_geometry_protocol(geometry, plans[640]["plan_id"], CACHE)
        geometry_current = {"status": "PASS"}
    except ValueError as exc:
        geometry_current = {"status": "FAIL_CURRENT_LIVE_SOURCE_PROVENANCE", "reason": str(exc)}
    reference = geometry["render_measurement"]["protocol"]["reference_sha256"]
    denominator = geometry["render_measurement"]["fixed_denominator_id"]
    for key, value in G.GEOMETRY_PROTOCOL.items():
        near(geometry["render_measurement"]["protocol"][key], value, key, 1e-12)
    assert geometry["render_measurement"]["protocol"]["mask_protocol"] == "ego_v2"
    assert geometry["acceptance_plan_id"] == plans[640]["plan_id"]
    for value in geometry["render_measurement"]["sources"].values(): pin(value["path"], value["sha256"])
    for path, sha in proof["source_sha256"].items(): pin(path, sha)
    assert proof["status"] == "checkpoint_verified" and proof["export_verified"] is True
    assert proof["export"]["verification"] == G.EXACT_EXPORT_METHOD
    assert geometry["candidate_model_sha256"] == proof["export"]["sha256"]
    assert geometry["candidate_representation_id"] == proof["checkpoint_sha256"] == candidate["checkpoint_sha256"]
    assert {G.frame_key(f) for f in geometry["scope"]["frame_keys"]} == set(records["candidate_640"]["validated"]["rows"])
    for role, record in (("candidate", "candidate_640"), ("baseline", "oldwhole_640")):
        assert geometry["render_measurement"]["sources"][role]["sha256"] == records[record]["descriptor"]["sha256"]
    try:
        G.validate_parameter_change(proof)
        parameter_contract = {"status": "ACCEPTED"}
    except ValueError as exc:
        parameter_contract = {"status": "REJECTED_BY_CURRENT_GATE_HELPER", "reason": str(exc),
            "explanation": "Direct measured initialization legitimately supplies topology/moments evidence with no saved initial tensor hash; the gate currently requires parameter-hash-change evidence. This is a contract mismatch, not evidence of no training. No helper was changed."}
    training, training_desc = read(RUN / "training_result.json")
    launch, launch_desc = read(RUN / "launch.json")
    snapshot, snapshot_desc = read(JOB / "input_snapshot.json")
    training_status, training_status_desc = read(JOB / "status.json")
    evaluation_status, evaluation_status_desc = read(JOB / "evaluation_status.json")
    assert training["status"] == "trained" and training["iterations_completed"] == 15000
    assert training["checkpoint_sha256"] == proof["checkpoint_sha256"] == candidate["checkpoint_sha256"]
    assert training["gaussians"] == proof["gaussians"] == candidate["gaussians"]
    assert evaluation_status["stage"] == "evaluated_pending_main_review"
    assert training["renderer_provenance"] == launch["renderer_provenance"] == proof["renderer_provenance"] == candidate["renderer_provenance"]
    source_drift = []
    for path, sha in snapshot["sources"].items():
        actual = pin(path)
        if actual["sha256"] != sha:
            source_drift.append({"path": path, "launch_sha256": sha, "current_sha256": actual["sha256"]})
        archived = JOB / "source_snapshot" / path
        if archived.is_file(): pin(archived, sha)
    for path, info in snapshot["inputs"].items():
        pin(path, info["sha256"])
        assert (ROOT/path).stat().st_size == info["bytes"]
    log_evidence = {p.name: pin(p) for p in JOB.glob("*.log")}
    seed_evidence = {name: read(name) for name in ("reports/resolution_pilot_v2_depthv3_block_000_audit.json",
        "reports/resolution_pilot_v2_support_1280_block_000.json") if (ROOT/name).is_file()}
    render_audits = {name: {**record["descriptor"], "status": "VERIFIED_SAVED_MODEL_FRAME_AND_FLOAT_ASSETS",
        "current_gate_identity_check": record["current_gate"], "model_identity": next(iter(record["validated"]["models"].values())),
        "backend": record["report"]["backend"], "step": record["report"]["step"],
        "checkpoint_sha256": record["report"]["checkpoint_sha256"], "source_ply_sha256": record["report"]["source_ply_sha256"],
        "gaussians": record["report"]["gaussians"], "metric_domain": "float_renderer", "saved_float_rescore": record["audit"],
        "render_seconds_sum": sum(f["render_seconds"] for f in record["report"]["frames"]),
        "gpu_peak_allocated_bytes_recorded": record["report"].get("gpu_peak_allocated_bytes")}
        for name, record in records.items()}
    known = geometry["gate_evidence"]
    training_evidence = {"proof": proof_desc, "status": proof["status"], "actual_checkpoint_step": candidate["step"],
        "completed_iterations": proof["completed_iterations"], "initial_gaussians": proof["initial_gaussians"],
        "final_gaussians": proof["gaussians"], "refine_events": proof["refine_events"],
        "all_tensors_finite_recorded": proof["all_tensors_finite"], "optimizer_moments_recorded": proof["optimizer_moments"],
        "parameter_change_evidence": proof["parameter_change_evidence"], "gate_parameter_change_contract": parameter_contract,
        "export": proof["export"], "pinned_proof_inputs_verified": len(proof["source_sha256"]),
        "independent_export_tensor_equality_reexecuted": False,
        "independent_audit": "Verified saved exact-export proof identity/bindings and every source SHA; read actual checkpoint step/shapes on CPU. Existing verifier's tensor equality/finite/moment assertions are attributed to that pinned proof. Current live model source differs from historical integration_v2 and that separate check fails closed.",
        "run_result": training_desc, "launch": launch_desc, "initialization": training["initialization"],
        "input_snapshot": snapshot_desc, "training_input_assets_verified": len(snapshot["inputs"]),
        "training_input_bytes_verified": sum(v["bytes"] for v in snapshot["inputs"].values()),
        "launch_source_drift_at_audit": source_drift, "training_status": training_status_desc,
        "evaluation_status": evaluation_status_desc, "status_note": "training status remains trained_pending_evaluation; later evaluation_status and all five verified reports establish followup completion. Source files were not rewritten."}
    resource_evidence = {"training": {k: training[k] for k in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "gpu_peak_protocol")},
        "geometry_recorded": {"usage": geometry["resource_usage"], "policy": geometry["resource_policy"], "elapsed_seconds": geometry["elapsed_seconds"]},
        "training_proof_recorded_cpu_threads": proof["cpu_threads"], "gpu_device_total_peak": None,
        "limitations": "Allocator peaks belong to recorded training/evaluation processes, not total device usage; previous geometry/proof used 4 CPU threads. This new audit uses <=3 and no GPU."}
    report = {"schema_version": 2, "status": "PAIRED_DESCRIPTIVE_COMPARISON", "block_id": "block_000",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "high_fidelity_accepted": False, "rollout_allowed": False, "scope": "20 fixed DEV views per resolution, raw/legacy/ego_v2; local checkpoint pilot, not assembled scene or independent six-floor fidelity acceptance",
        "comparisons": comparisons, "identity_audit": {"plans": plan_audits, "renders": render_audits,
            "fresh_controls": {"old_original_export": expected_original, "old_whole_export": expected_whole,
                "status": "ALL_NEW_REPORTS_AND_FLOAT_ARRAYS_VERIFIED", "script": pin("runtime/finish_block000_depthv3_seeded.py"), "logs": log_evidence,
                "inference_steps": [11999], "repeated_render_reproducibility": "NOT_MEASURED; these are controls, not a second same-model repeat"}},
        "protocol_definition": {"raw": "All image pixels", "legacy": "Frozen source mask", "ego_v2": "Frozen source mask AND ego_v2_rev2 mask",
            "psnr_aggregation": "Arithmetic mean of per-frame PSNR; paired mean is mean of exact per-frame differences",
            "resolution_comparability": "Comparisons are within each frozen resolution/teacher plan. No 640-to-1280 difference is attributed to training. 1280 legacy and ego_v2 coincide here because the source already uses the revised mask.",
            "test_metrics_evaluated": False},
        "geometry": {"evidence": geometry_desc, "status": "VERIFIED_SAVED_FIXED_SCOPE_AND_NATIVE_CHECKPOINT_EXPORT_BINDING",
            "current_gate_provenance_check": geometry_current,
            "reference_sha256": reference, "fixed_denominator_id": denominator, "scope": geometry["render_measurement"]["scope"],
            "gate_evidence": known, "summaries": geometry["render_measurement"]["summaries"], "ply": geometry["ply"],
            "geometry_1280": "NOT_MEASURED", "independent_ground_truth": False, "limitations": geometry["limitations"]},
        "training_evidence": training_evidence, "resource_evidence": resource_evidence,
        "dataset_preparation_evidence": {name: {"artifact": desc, "recorded": value} for name, (value, desc) in seed_evidence.items()},
        "audit_execution": {"helper": pin(__file__), "cpu_thread_limit": 3, "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "threadpools": [{k: p.get(k) for k in ("internal_api", "prefix", "num_threads")} for p in threadpool_info()],
            "gpu_used": False, "cuda_initialized": torch.cuda.is_initialized(), "elapsed_seconds": time.monotonic()-start,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
        "evidence_files": CACHE["_pins"]}
    # Keep both historical and current identities; v3 never substitutes for v2.
    change_evidence = {name: read(name) for name in ("reports/warm_read_budget_cpu.json", "reports/inria_backend_integration_gpu_v3.json") if (ROOT/name).is_file()}
    report["live_source_change"] = {"historical_validation_before_change": {
        "status": "PASS_OBSERVED_DURING_INITIAL_CPU_PROBE_IN_THIS_AUDIT",
        "checks": "All5 render identity validators and geometry protocol/checkpoint-export binding passed before the concurrent live model/CLI edit; CUDA uninitialized",
        "standalone_receipt_saved": False},
        "current_historical_v2_validation": "FAIL_STALE_LIVE_SOURCE; retained, not waived",
        "launch_sources_changed": source_drift,
        "change_context": "Main confirmed warm-read-budget-only change (fixed1.5M limit to config count and2GiB) after completed block000 inference; no automatic rerender requested",
        "current_change_evidence": {name: {"artifact": desc, "recorded_status": value.get("status"),
            "forward_model_images_bitwise_equal": value.get("forward_model_images_bitwise_equal"),
            "rgb_alpha_parameter_gradients_bitwise_equal": value.get("rgb_alpha_parameter_gradients_bitwise_equal")}
            for name, (value, desc) in change_evidence.items()},
        "integration_v2_rewritten": False, "integration_v3_substituted_into_historical_reports": False,
        "frozen_historical_comparisons_remain_descriptive": True}
    assert not torch.cuda.is_initialized() and all(p["num_threads"] <= 3 for p in threadpool_info())
    # All source hashes are checked once; detect concurrent replacements at end.
    for path, sha in list(CACHE["_pins"].items()): G.verify_artifact(path, sha, CACHE)
    for path, before in SNAPSHOTS.items(): assert state(Path(path)) == before, "Asset changed during audit"
    report["report_id"] = G.object_sha(report)
    outputs[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = [row for comparison in comparisons for row in comparison["rows"]]
    with outputs[1].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    decision = {"schema_version": 2, "status": "PILOT_NOT_PROMOTED", "high_fidelity_accepted": False, "rollout_allowed": False,
        "training_complete": True, "checkpoint_verified": True, "completed_iterations": 15000, "gaussians": proof["gaussians"],
        "paired_appearance_evidence": {"path": str(outputs[0].relative_to(ROOT)), "sha256": G.sha256_file(outputs[0]), "report_id": report["report_id"]},
        "csv_evidence": {"path": str(outputs[1].relative_to(ROOT)), "sha256": G.sha256_file(outputs[1]), "rows": len(rows)},
        "geometric_evidence": geometry_desc, "checkpoint_evidence": proof_desc,
        "current_live_provenance": report["live_source_change"],
        "reasons": ["640 ego PSNR loses %.6f dB to refreshed original single-block control" % -comparisons[0]["paired_delta_db_mean"],
            "1280 ego PSNR loses %.6f dB to original single-block control" % -comparisons[2]["paired_delta_db_mean"],
            "640 improves %.6f dB over weak old whole-scene control; this does not recover original single-block fidelity" % comparisons[1]["paired_delta_db_mean"],
            "Conditioned p50 %.6fm exceeds .15m and is worse than baseline %.6fm" % (known["G4"]["candidate_p50_m"], known["G4"]["baseline_p50_m"]),
            "Floor hole reduction %.4f%% and wall reduction %.4f%% both miss 30%%; floor support is only725pixels" % (100*known["G5"]["floor_relative_reduction"],100*known["G5"]["wall_relative_reduction"]),
            "2cm occupied-voxel multiplicity median1 is below2; seam metric, fresh repeat, full F1 and 408-view candidate are absent"],
        "appearance": [{k: v for k, v in c.items() if k not in ("rows", "by_camera", "pairing")} for c in comparisons],
        "gate_evidence": known,
        "acceptance_audit": {"full_gate_run": False, "full_goal_accepted": False, "thresholds_modified": False,
            "G1": {"status": "NOT_MEASURED", "reason": "No new same-model repeat and no complete408candidate"},
            "G2": {"status": "INCOMPLETE_SCOPE", "local_block000_gain_db": comparisons[1]["paired_delta_db_mean"], "reason": "Only one of five required F1blocks"},
            "G3": {"local_status": "FAIL", "reason": "Multiplicity1<2; opacity subchecks pass"},
            "G4": {"local_status": "FAIL", "reason": "p50 exceeds absolute and baseline-ratio targets"},
            "G5": {"local_status": "FAIL", "reason": "Neither floor nor wall achieves30%reduction"},
            "G6": {"status": "ASSEMBLY_NOT_EVALUATED", "local_single_model_regression_db": -comparisons[0]["paired_delta_db_mean"], "reason": "Native checkpoint candidate is not a new full-scene assembly; original-single quality regression is descriptive"},
            "G7": {"status": "INCOMPLETE_EVIDENCE", "radius_subchecks": "PASS", "seam_gap_db": None},
            "G8": {"status": "INCOMPLETE_SCOPE_AND_CONTRACT_MISMATCH", "training_authenticity": "VERIFIED_PINNED_PROOF",
                "parameter_change_helper": parameter_contract, "reason": "One directly seeded block; no initial tensor hash was saved. No fabricated L2/hash or five-block training claim."}},
        "resource_evidence": resource_evidence,
        "engineering_interpretation": "The directly measured initialization plus AA/cap003/fullres combined recipe does not preserve this block's original appearance and does not improve both fixed floor/wall geometry. High alpha coverage is not correct depth: wall wrong-depth pixels remain dominant. Current evidence does not isolate initialization, cap, backend, loss weights or calibration as the cause.",
        "next_action": "Keep as an unpromoted local diagnostic; inspect existing signed-depth/visibility and measured-support evidence before choosing a controlled engineering change. Do not spend on full408 candidate rendering based on this pilot.",
        "limitations": ["Floor label is a geometric proxy with725supportedpixels; do not generalize to the whole floor", "Only9.494% of masked640pixels have teacher support", "1280 conditioned geometry and viewer parity were not measured", "Same-LIO reference is not independent truth", "Historical source reports were preserved; old training status was not rewritten"]}
    decision["report_id"] = G.object_sha(decision)
    outputs[2].write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"files": [str(p.relative_to(ROOT)) for p in outputs], "rows": len(rows),
        "identities_verified": len(records), "float_frames_rescored": sum(r["audit"]["frames"] for r in records.values()),
        "comparison_report_id": report["report_id"], "decision": decision["status"]}), flush=True)


if __name__ == "__main__":
    main()

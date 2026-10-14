#!/usr/bin/env python3
"""Bounded block000 audit. Default: one-shot metadata precheck, no image reads.

Preparation: .venv-train/bin/python -B reports/block000_measuredseed_anneal_v1_audit.py --precheck --write-precheck
After controller completion, explicitly request --run to rescore saved floats.
Exit codes: 0 ready/completed, 3 pending inputs, 2 invalid evidence/error.
Pending/invalid evidence never writes paired reports or a failed-quality decision.
No rendering, training, polling, or modification of frozen inputs is performed.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEM = "block000_measuredseed_anneal_v1"
PREFIX = ROOT / "reports" / STEM
RUN = ROOT / "runs/fidelity_v2" / ("train_" + STEM)
JOB = ROOT / "runs/fidelity_v2/jobs" / STEM
RECOVERY = JOB / "recovery_allocator_v1"
HELPER = ROOT / "reports/block000_seed_aa_cap003_audit.py"
HELPER_SHA = "33ef882adf3825d97d28c66fc47a0f09e10712d7a4bc5116d75a29136dab15b9"
PLANS = {
    640: ("reports/acceptance_plan_v1.json", "150ef4619734eebcedc64162e5e4d63eb853d86e1e8d67bd5a484397be769101"),
    1280: ("reports/acceptance_resolution_depthv3_plan.json", "00cc0522b5fb713a73d9fd0860340c93acb0ec47d1cf209057e9c9070039f1d7"),
}
SOURCES = (
    ("candidate_640", STEM, 640, False),
    ("candidate_1280", STEM + "_resolution", 1280, False),
    ("original_640", "original000_refreshed_v2", 640, False),
    ("original_1280", "original000_resolution_depthv3", 1280, False),
    ("oldwhole_640", "old_all28_block000_refreshed_v2", 640, True),
    ("directseed_640", "block000_seed_aa_cap003_acceptance", 640, False),
    ("directseed_1280", "block000_seed_aa_cap003_resolution", 1280, False),
)
PROOF = ROOT / "reports" / (STEM + "_checkpoint.json")
GEOMETRY = ROOT / "reports" / ("surface_quality_v2_" + STEM + ".json")
PRECHECK = Path(__file__).with_suffix(".precheck.json")
OUTPUTS = [Path(str(PREFIX) + s) for s in ("_paired_comparison.json", "_paired_comparison.csv", "_decision.json")]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short(path):
    return str(Path(path).resolve().relative_to(ROOT))


def precheck():
    """Inventory only; readiness is not acceptance or hash/float validation."""
    missing, waiting, invalid, inventory = [], [], [], []

    def exists(path):
        path = ROOT / path
        present = path.is_file() and path.stat().st_size > 0
        if not present:
            missing.append(short(path))
        return present

    def read_small(path):
        if not exists(path):
            return None
        try:
            return json.loads((ROOT / path).read_text())
        except (ValueError, OSError) as exc:
            invalid.append({"path": short(ROOT / path), "reason": str(exc)})
            return None

    if exists(HELPER) and sha256(HELPER) != HELPER_SHA:
        invalid.append({"path": short(HELPER), "reason": "Reviewed helper SHA changed"})
    plans = {}
    for size, (path, expected) in PLANS.items():
        plan = read_small(path)
        if plan is not None:
            if sha256(ROOT / path) != expected:
                invalid.append({"path": path, "reason": "Frozen plan SHA changed"})
            plans[size] = [f for f in plan["frames"] if f["block_id"] == "block_000"]
            if len(plans[size]) != 20:
                invalid.append({"path": path, "reason": "Expected exactly 20 frozen block000 frames"})
    for label, directory, size, _ in SOURCES:
        path = ROOT / "runs/fidelity_v2" / directory / "metrics.json"
        report = read_small(path)
        found = sum(exists(path.parent / ("%03d_float.npz" % f["dataset_index"])) for f in plans.get(size, []))
        if report is not None:
            if report.get("status") != "evaluated":
                waiting.append(label + ": report not evaluated")
            elif report.get("images") != 20 or len(report.get("frames", [])) != 20:
                invalid.append({"path": short(path), "reason": "Completed report does not contain 20 frames"})
            for field in ("config", "checkpoint", "transforms", "source_ply"):
                if report.get(field):
                    asset = report[field]
                    exists(asset["path"] if isinstance(asset, dict) else asset)
        inventory.append({"label": label, "metrics_present": report is not None, "float_files_present": found, "expected_float_files": 20})
    required = [PROOF, GEOMETRY, RUN / "launch.json", RUN / "refinement.json", RUN / "resume_events.jsonl",
        JOB / "input_snapshot.json", JOB / "warm_source_lineage.json", RECOVERY / "input_snapshot.json",
        RECOVERY / "pre_resume_launch.json", RECOVERY / "pre_resume_refinement.json",
        ROOT / "reports/inria_backend_integration_gpu_v4.json",
        ROOT / "reports/block000_measuredseed_anneal_resume_midcheck.json",
        ROOT / "reports/surface_quality_v2_block000_seed_aa_cap003.json",
        ROOT / "reports/block000_seed_aa_cap003_checkpoint.json",
        ROOT / "reports/block000_seed_aa_cap003_paired_comparison.json",
        ROOT / "runs/fidelity_v2" / STEM / "gaussians.ply"]
    for path in required:
        exists(path)
    status = read_small(RECOVERY / "status.json")
    if status and status.get("stage") != "evaluated_pending_main_review":
        waiting.append("Recovery controller stage: " + str(status.get("stage")))
    training = read_small(RUN / "training_result.json")
    if training:
        if training.get("status") != "trained" or training.get("checkpoint_step") != 14999:
            waiting.append("Final trained step14999 result not ready")
        if training.get("checkpoint"):
            exists(training["checkpoint"])
    recovery = read_small(RECOVERY / "input_snapshot.json")
    if recovery:
        exists(recovery["resume_checkpoint"])
        for path in recovery["recovery_sources"]:
            exists(RECOVERY / "source_snapshot" / path)
    state = "INVALID_INPUT_METADATA" if invalid else ("PENDING_INPUTS" if missing or waiting else "READY_FOR_EXPLICIT_AUDIT")
    return {"status": state, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "missing": sorted(set(missing)), "waiting": waiting,
        "invalid": invalid, "render_inventory": inventory, "recovery_stage": (status or {}).get("stage"),
        "quality_assessment": None, "final_reports_written": False, "gpu_used": False,
        "scope": "One block000 pilot, 20 DEV views per resolution; no full28 or six-floor acceptance",
        "precheck_only": True, "images_or_float_arrays_read": False, "large_checkpoints_hashed": False,
        "next_action": "Run --run once all inputs are complete; no automatic wait or retry"}


def check_result(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
        return {"status": "PASS"}
    except ValueError as exc:
        return {"status": "REJECTED_BY_CURRENT_GATE_HELPER", "reason": str(exc)}


def audit_plans(H):
    plans, evidence, group_splits = {}, {}, {}
    for size, (path, expected) in PLANS.items():
        descriptor = H.pin(path, expected)
        plan = H.G.load_plan(ROOT / path, verify_assets=False)
        fixed = [f for f in plan["frames"] if f["block_id"] == "block_000"]
        assert len(fixed) == len({H.G.frame_key(f) for f in fixed}) == 20
        assert plan["mask_revision"] == "ego_v2_rev2" and set(plan["protocols"]) == set(H.PROTOCOLS)
        transforms, td = H.read(fixed[0]["transforms"])
        index = {f["file_path"]: f for f in transforms["frames"]}
        assert len(index) == len(transforms["frames"])
        splits = {k: transforms[k + "_filenames"] for k in ("train", "val", "test")}
        for values in splits.values():
            assert len(values) == len(set(values)) and set(values) <= set(index)
        groups = {k: {str(index[v]["group_id"]) for v in values} for k, values in splits.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not set(splits[a]) & set(splits[b]) and not groups[a] & groups[b]
        for frame in fixed:
            assert td["sha256"] == frame["transforms_sha256"]
            assert splits["val"][frame["dataset_index"]] == frame["file_path"]
            H.validate_frame_source(frame, index[frame["file_path"]], frame["dataset_index"])
            for field, sha in frame["asset_sha256"].items():
                H.pin(frame[field], sha)
        group_splits[size] = groups
        plans[size] = plan
        evidence[str(size)] = {**descriptor, "plan_id": plan["plan_id"], "audited_frames": 20,
            "canonical_plan_frames": len(plan["frames"]), "mask_revision": plan["mask_revision"],
            "transforms": td, "split_image_counts": {k: len(v) for k, v in splits.items()},
            "split_group_counts": {k: len(v) for k, v in groups.items()}, "splits_disjoint": True,
            "test_image_pixels_read": False, "other_block_assets_rehashed": False, "floor_id_verified": False}
    assert group_splits[640] == group_splits[1280]
    assert [H.G.frame_key(f) for f in plans[640]["frames"] if f["block_id"] == "block_000"] == [H.G.frame_key(f) for f in plans[1280]["frames"] if f["block_id"] == "block_000"]
    return plans, evidence


def audit_resume(H, candidate, proof):
    training, td = H.read(RUN / "training_result.json")
    launch, ld = H.read(RUN / "launch.json")
    snapshot, sd = H.read(JOB / "input_snapshot.json")
    recovery, rd = H.read(RECOVERY / "input_snapshot.json")
    status, status_desc = H.read(RECOVERY / "status.json")
    assert status["stage"] == "evaluated_pending_main_review" and status["checkpoint_step"] == 14999
    assert training["status"] == "trained" and training["iterations_completed"] == training["iterations_requested"] == 15000
    assert training["checkpoint_step"] == candidate["step"] == candidate["inference_model_step"] == 14999
    assert training["checkpoint_sha256"] == proof["checkpoint_sha256"] == candidate["checkpoint_sha256"]
    assert training["gaussians"] == proof["gaussians"] == candidate["gaussians"] <= snapshot["final_count_budget"]
    assert training["renderer_provenance"] == launch["renderer_provenance"] == proof["renderer_provenance"] == candidate["renderer_provenance"]
    assert recovery["original_job_input_sha256"] == sd["sha256"]
    backup = H.pin(recovery["resume_checkpoint"], recovery["resume_checkpoint_sha256"])
    stored = H.G.checkpoint_metadata(ROOT / backup["path"])
    resume = training["resumed_from"]
    restored = resume["restored_state"]
    assert stored["step"] == resume["checkpoint_step"] == restored["checkpoint_step"] == restored["model_step"] == 2000
    assert resume["sha256"] == backup["sha256"]
    assert restored["next_training_step"] == 2001 and restored["scheduler_state_restored"] is True
    assert resume["target_iterations"] == 15000 and resume["remaining_iterations"] == 12999
    assert set(restored["optimizer_binding"]) == H.G.PARAMETER_GROUPS
    assert all(v["live_parameter_identity"] is True for v in restored["optimizer_binding"].values())
    event_desc = H.pin(RUN / "resume_events.jsonl")
    events = [json.loads(line) for line in (ROOT / event_desc["path"]).read_text().splitlines() if line.strip()]
    assert len(events) == 1 and events[0]["checkpoint"] == resume
    assert training["cuda_allocator_config"] == recovery["allocator"] == events[0]["cuda_allocator_config"]
    drift = []
    for archive, sources in ((JOB, snapshot["sources"]), (RECOVERY, recovery["recovery_sources"])):
        for path, expected in sources.items():
            H.pin(archive / "source_snapshot" / path, expected)
            current = H.pin(path)
            if current["sha256"] != expected:
                drift.append({"path": path, "archived_sha256": expected, "current_sha256": current["sha256"]})
    for path, info in snapshot["inputs"].items():
        H.pin(path, info["sha256"])
        assert (ROOT / path).stat().st_size == info["bytes"]
    lineage, lineage_desc = H.read(JOB / "warm_source_lineage.json")
    assert lineage["original_trained_gaussians"] + lineage["new_untrained_measured_surface_gaussians"] == lineage["total"] == snapshot["initial_gaussians"] == proof["initial_gaussians"]
    H.pin(lineage["initialization_ply"], training["initialization"]["initialization_sha256"])
    midcheck, mid_desc = H.read("reports/block000_measuredseed_anneal_resume_midcheck.json")
    assert midcheck["checkpoints"][0]["checkpoint_sha256"] == backup["sha256"]
    return training, {"training_result": td, "launch": ld, "input_snapshot": sd, "recovery_snapshot": rd,
        "recovery_status": status_desc, "resume_events": event_desc, "resume_checkpoint": backup,
        "actual_resume_checkpoint_step": stored["step"], "first_resumed_step": 2001, "final_checkpoint_step": 14999,
        "completed_iterations_total": 15000, "remaining_iterations_at_resume": 12999,
        "restored_state_recorded": restored, "bitwise_continuous_training": False,
        "resume_scope": "Model/Adam/scheduler/scaler restoration is recorded; image-sampling RNG is reseeded and uncheckpointed strategy accumulators restart. No uninterrupted-run or repeated-render equivalence claim.",
        "source_snapshots_verified": True, "current_source_drift": drift,
        "training_input_assets_verified": len(snapshot["inputs"]), "lineage": {"artifact": lineage_desc, "recorded": lineage},
        "midrun_progress": {"artifact": mid_desc, "recorded": midcheck,
            "scope": "Historical step2000-to4000 parameter/Adam check; removed intermediate4000 checkpoint is not required or freshly reverified"},
        "pre_resume_launch": H.pin(RECOVERY / "pre_resume_launch.json"),
        "pre_resume_refinement": H.pin(RECOVERY / "pre_resume_refinement.json")}


def geometry_evidence(H, path, proof_path, candidate_label, records, plan, strict):
    geometry, gd = H.read(path)
    proof, pd = H.read(proof_path)
    candidate = records[candidate_label]["report"]
    assert proof["status"] == "checkpoint_verified" and proof["export_verified"] is True
    assert proof["export"]["verification"] == H.G.EXACT_EXPORT_METHOD
    H.pin(proof["checkpoint"], proof["checkpoint_sha256"])
    H.pin(proof["export"]["path"], proof["export"]["sha256"])
    assert geometry["candidate_model_sha256"] == proof["export"]["sha256"]
    assert geometry["candidate_representation_id"] == proof["checkpoint_sha256"] == candidate["checkpoint_sha256"]
    assert proof["completed_iterations"] == candidate["step"] + 1 == 15000
    assert geometry["acceptance_plan_id"] == plan["plan_id"]
    assert {H.G.frame_key(f) for f in geometry["scope"]["frame_keys"]} == set(records[candidate_label]["validated"]["rows"])
    measured = geometry["render_measurement"]
    for key, value in H.G.GEOMETRY_PROTOCOL.items():
        H.near(measured["protocol"][key], value, key, 1e-12)
    assert measured["protocol"]["mask_protocol"] == "ego_v2"
    for desc in measured["sources"].values():
        H.pin(desc["path"], desc["sha256"])
    for role, label in (("candidate", candidate_label), ("baseline", "oldwhole_640")):
        assert measured["sources"][role]["sha256"] == records[label]["descriptor"]["sha256"]
    current = check_result(H.G.validate_geometry_protocol, geometry, plan["plan_id"], H.CACHE)
    parameter = check_result(H.G.validate_parameter_change, proof)
    if strict:
        assert current["status"] == "PASS", current
        for source, sha in proof["source_sha256"].items():
            H.pin(source, sha)
    return proof, {"artifact": gd, "checkpoint_proof": pd, "current_gate_provenance": current,
        "parameter_change_contract": parameter, "fixed_denominator_id": measured["fixed_denominator_id"],
        "reference_sha256": measured["protocol"]["reference_sha256"], "scope": geometry["scope"],
        "gate_evidence": geometry["gate_evidence"], "summaries": measured["summaries"], "ply": geometry["ply"],
        "geometry_1280": "NOT_MEASURED", "independent_ground_truth": False, "limitations": geometry["limitations"],
        "proof_recorded": proof, "export_tensor_equality_reexecuted": False,
        "resource_usage_recorded": geometry["resource_usage"], "resource_policy_recorded": geometry["resource_policy"]}


def run_audit():
    assert not any(p.exists() for p in OUTPUTS), "Refuse to overwrite completed audit reports"
    assert sha256(HELPER) == HELPER_SHA, "Reviewed helper changed"
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("reviewed_block000_audit", HELPER)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    start = time.monotonic()
    for path in (__file__, HELPER, "scripts/acceptance_gate.py", "scripts/measure_surface_quality.py",
                 "scripts/build_acceptance_plan.py", "scripts/verify_training_checkpoint.py"):
        H.pin(path)
    plans, plan_audits = audit_plans(H)
    records = {}
    for label, directory, size, whole in SOURCES:
        path = ROOT / "runs/fidelity_v2" / directory / "metrics.json"
        report, descriptor = H.read(path)
        assert report["acceptance"]["sha256"] == plan_audits[str(size)]["sha256"]
        validated = H.model_and_frame_identity(report, plans[size], whole)
        current = check_result(H.G.validate_render_reports, plans[size], [report], ["block_000"], whole_model=whole, asset_cache=H.CACHE)
        if label.startswith("candidate"):
            assert current["status"] == "PASS", current
            assert report["source_ply"] is None and report["step"] == 14999
            assert report["backend"] == "hbb1-aa-depth-vjp-2dgs"
            integration = report["renderer_integration_proof"]
            assert (ROOT / integration["path"]).resolve() == ROOT / "reports/inria_backend_integration_gpu_v4.json"
            H.pin(integration["path"], integration["sha256"])
        fixed = [f for f in plans[size]["frames"] if f["block_id"] == "block_000"]
        rescored = H.rescore(path, report, fixed)
        for protocol in H.PROTOCOLS:
            H.near(statistics.mean(r["protocols"][protocol]["psnr_db"] for r in report["frames"]), report["protocols"][protocol]["psnr_db_mean"], "aggregate PSNR")
        records[label] = dict(report=report, descriptor=descriptor, validated=validated, rescore=rescored, current_gate=current)
        print(label + ": verified20 saved float frames", file=sys.stderr, flush=True)
    for label in ("candidate", "directseed"):
        assert records[label + "_640"]["report"]["checkpoint_sha256"] == records[label + "_1280"]["report"]["checkpoint_sha256"]
    original = H.pin("runs/garage_surface_v1/block_000/eval_dev/gaussians.ply")
    oldwhole = H.pin("maps/garage_core_all_v1/gaussians_full.ply")
    assert records["original_640"]["report"]["source_ply_sha256"] == records["original_1280"]["report"]["source_ply_sha256"] == original["sha256"]
    assert records["oldwhole_640"]["report"]["source_ply_sha256"] == oldwhole["sha256"]
    comparisons = [H.compare("candidate_" + str(size), baseline + "_" + str(size), "block000_%s_vs_%s" % (size, baseline), size, records)
        for size, baseline in ((640, "original"), (640, "oldwhole"), (640, "directseed"), (1280, "original"), (1280, "directseed"))]
    proof, geometry = geometry_evidence(H, GEOMETRY, PROOF, "candidate_640", records, plans[640], True)
    _, direct_geometry = geometry_evidence(H, "reports/surface_quality_v2_block000_seed_aa_cap003.json",
        "reports/block000_seed_aa_cap003_checkpoint.json", "directseed_640", records, plans[640], False)
    assert geometry["fixed_denominator_id"] == direct_geometry["fixed_denominator_id"]
    assert geometry["reference_sha256"] == direct_geometry["reference_sha256"]
    training, recovery = audit_resume(H, records["candidate_640"]["report"], proof)
    historical, historical_desc = H.read("reports/block000_seed_aa_cap003_paired_comparison.json")
    assert H.G.object_sha({k: v for k, v in historical.items() if k != "report_id"}) == historical["report_id"]
    for old_label, new_label in (("candidate_640", "directseed_640"), ("candidate_1280", "directseed_1280"),
                                  ("original_640", "original_640"), ("original_1280", "original_1280"), ("oldwhole_640", "oldwhole_640")):
        assert historical["identity_audit"]["renders"][old_label]["sha256"] == records[new_label]["descriptor"]["sha256"]
    logs = [H.pin(p) for folder in (JOB, RECOVERY) for p in sorted(folder.glob("*.log"))]
    resource_samples = []
    for folder in (JOB, RECOVERY):
        path = folder / "board_memory_samples.jsonl"
        if path.is_file():
            descriptor = H.pin(path)
            samples = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            valid = [float(s["values"].split(",")[0]) for s in samples if s.get("returncode") == 0 and s.get("values") and "\n" not in s["values"]]
            resource_samples.append({"artifact": descriptor, "samples": len(samples), "single_device_samples": len(valid),
                "recorded_device_memory_peak_MiB": max(valid) if valid else None,
                "scope": "Sampled whole-device usage; includes other processes, not a continuous or candidate-only peak"})
    gates = geometry["gate_evidence"]
    thresholds = plans[640]["gate_thresholds"]
    def local_status(keys, condition):
        if any(gates[k].get("status") != "known" for k in keys):
            return "UNKNOWN"
        return "PASS_LOCAL_SUBCHECK_ONLY" if condition() else "FAIL_LOCAL_SUBCHECK"
    local = {
        "G3": local_status(["G3"], lambda: gates["G3"]["opacity_median"] >= thresholds["G3"]["opacity_median_min"] and gates["G3"]["opacity_frac_lt_0_5"] <= thresholds["G3"]["opacity_below_half_fraction_max"] and gates["G3"]["voxel_2cm_multiplicity_median"] >= thresholds["G3"]["voxel_multiplicity_median_min"]),
        "G4": local_status(["G4"], lambda: gates["G4"]["candidate_p50_m"] <= thresholds["G4"]["chamfer_p50_m_max"] and gates["G4"]["candidate_p50_m"] <= thresholds["G4"]["chamfer_p50_ratio_max"] * gates["G4"]["baseline_p50_m"]),
        "G5": local_status(["G5"], lambda: all(gates["G5"][k]["candidate_hole_rate"] <= thresholds["G5"]["hole_rate_ratio_max"] * gates["G5"][k]["baseline_hole_rate"] for k in ("floor", "wall"))),
        "G7_radius_only": local_status(["G7"], lambda: gates["G7"]["tangential_radius_p95_m"] <= thresholds["G7"]["tangent_radius_p95_m_max"] + 1e-8 and gates["G7"]["tangential_radius_max_m"] <= thresholds["G7"]["tangent_radius_max_m_max"] + 1e-8),
    }
    resources = {"training_recorded": {k: training.get(k) for k in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "gpu_peak_protocol", "cuda_allocator_config")},
        "training_peak_scope": "Resumed segment only; not the uninterrupted whole run. Original OOM is retained in archived logs.", "sampled_device_memory": resource_samples, "logs": logs}
    render_audits = {k: {**r["descriptor"], "current_gate_identity_check": r["current_gate"], "saved_float_rescore": r["rescore"],
        "model_identity": next(iter(r["validated"]["models"].values())),
        **{f: r["report"].get(f) for f in ("checkpoint_sha256", "source_ply_sha256", "step", "inference_model_step", "backend", "gaussians", "metric_domain", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")}}
        for k, r in records.items()}
    report = {"schema_version": 1, "status": "PAIRED_DESCRIPTIVE_COMPARISON", "block_id": "block_000",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "high_fidelity_accepted": False,
        "rollout_allowed": False, "scope": "20 fixed DEV views per resolution, one native checkpoint pilot; not whole28 or six-floor acceptance",
        "comparisons": comparisons, "identity_audit": {"plans": plan_audits, "renders": render_audits,
            "controls": {"original_export": original, "oldwhole_export": oldwhole, "historical_comparison": historical_desc,
                "freshness": "Existing refreshed controls are bound to their exact saved report/model/float identities; this CPU audit launches no render or repeat"}},
        "protocol_definition": {"raw": "All pixels", "legacy": "Frozen source mask", "ego_v2": "Frozen source mask AND ego_v2_rev2",
            "aggregation": "Arithmetic mean of per-frame PSNR; exact paired per-frame deltas", "metric_tolerance": 1e-7,
            "resolution_scope": "Within-resolution matching only; no640-to1280 causal attribution", "test_image_pixels_read": False},
        "geometry": geometry, "directseed_geometry_same_fixed_denominator": direct_geometry,
        "training_recovery": recovery, "resource_evidence": resources,
        "local_geometry_threshold_checks": local, "unchanged_thresholds": thresholds,
        "historical_source_state": {"previous_audit": historical_desc,
            "historical_validation_before_change_recorded": historical.get("live_source_change", {}).get("historical_validation_before_change"),
            "scope": "Attributed historical statement, not a new PASS. Current per-report checks above retain stale-source failures; no v4 substitution into historical v2 reports."},
        "audit_execution": {"script": H.pin(__file__), "reviewed_helper": H.pin(HELPER, HELPER_SHA),
            "cpu_thread_limit": 3, "gpu_used": False, "cuda_initialized": H.torch.cuda.is_initialized(),
            "float_frames_rescored": 140, "protocol_frame_scores_recomputed": 420,
            "elapsed_seconds": time.monotonic() - start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "evidence_files": H.CACHE["_pins"]}
    assert not H.torch.cuda.is_initialized() and all(p["num_threads"] <= 3 for p in H.threadpool_info())
    for path, sha in list(H.CACHE["_pins"].items()):
        H.G.verify_artifact(path, sha, H.CACHE)
    for path, before in H.SNAPSHOTS.items():
        assert H.state(Path(path)) == before, "Asset changed during audit: " + path
    report["report_id"] = H.G.object_sha(report)
    rows = [r for c in comparisons for r in c["rows"]]
    assert len(rows) == 300
    decision = {"schema_version": 1, "status": "LOCAL_PILOT_REVIEW_ONLY", "high_fidelity_accepted": False,
        "rollout_allowed": False, "training_complete": True, "completed_iterations": 15000,
        "checkpoint_verified": True, "checkpoint_evidence": geometry["checkpoint_proof"],
        "appearance": [{k: v for k, v in c.items() if k not in ("rows", "by_camera", "pairing")} for c in comparisons],
        "local_geometry_threshold_checks": local, "gate_evidence": gates,
        "training_parameter_change_contract": geometry["parameter_change_contract"],
        "acceptance_scope": {"full_gate_run": False, "full_goal_accepted": False, "G1": "NOT_MEASURED: no same-model fresh repeat",
            "G2": "INCOMPLETE: onlyblock000 of fiveF1 blocks", "G3_G4_G5": "Only local recorded geometry subchecks above",
            "G6": "ASSEMBLY_NOT_EVALUATED: single-model original-control differences are descriptive",
            "G7": "INCOMPLETE: seam evidence missing even if radius subchecks pass",
            "G8": "LOCAL_CHECKPOINT_PROOF_ONLY: not five-block training acceptance"},
        "reasons": ["%s: ego_v2 paired mean delta %+.6f dB (%d/20 frames improved)" % (c["comparison"], c["paired_delta_db_mean"], c["frames_improved"]) for c in comparisons],
        "limitations": ["Same-LIO reference is not independent truth; fixed supported rays do not certify unsupported floor completeness",
            "One block000 cannot establish all28 or six-floor fidelity; no semantic floor identity is inferred",
            "1280 conditioned geometry, fresh repeat and whole-scene seam/viewer evidence absent",
            "Resumed RNG and strategy state are not bitwise continuous; combined recipe does not isolate individual engineering causes",
            "Historical controls retain their actual current provenance status; saved-float reproducibility is not renderer repeatability"],
        "resource_evidence": resources}
    # Serialize everything before creating outputs; missing/invalid evidence above cannot write a quality decision.
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    json.dumps(decision, allow_nan=False)
    with OUTPUTS[0].open("x") as stream:
        stream.write(report_text)
    with OUTPUTS[1].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    decision["paired_comparison"] = {"path": short(OUTPUTS[0]), "sha256": sha256(OUTPUTS[0]), "report_id": report["report_id"]}
    decision["csv"] = {"path": short(OUTPUTS[1]), "sha256": sha256(OUTPUTS[1]), "rows": len(rows)}
    decision["report_id"] = H.G.object_sha(decision)
    with OUTPUTS[2].open("x") as stream:
        stream.write(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": decision["status"], "files": [short(p) for p in OUTPUTS], "csv_rows": len(rows), "high_fidelity_accepted": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--precheck", action="store_true", help="One-shot metadata inventory (default); no arrays/checkpoint loads")
    mode.add_argument("--run", action="store_true", help="After ready only: independently rescore140 saved arrays and write300 paired CSV rows plus JSON/decision")
    parser.add_argument("--write-precheck", action="store_true", help="Save same-stem precheck JSON only; incompatible with --run")
    args = parser.parse_args()
    if args.run and args.write_precheck:
        parser.error("--write-precheck is a preparation-only option")
    if sys.flags.optimize:
        parser.error("Assertions are audit checks; do not use Python -O")
    sys.dont_write_bytecode = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "3"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
    try:
        result = precheck()
        if args.write_precheck:
            PRECHECK.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if result["status"] != "READY_FOR_EXPLICIT_AUDIT" or not args.run:
            print(json.dumps(result, indent=2, allow_nan=False))
            return {"READY_FOR_EXPLICIT_AUDIT": 0, "PENDING_INPUTS": 3, "INVALID_INPUT_METADATA": 2}[result["status"]]
        run_audit()
        return 0
    except Exception as exc:
        print(json.dumps({"status": "AUDIT_INPUT_VALIDATION_ERROR", "error": repr(exc), "quality_assessment": None,
            "scope": "Input/provenance/computation could not be validated; this is not a failed-quality measurement",
            "final_files_present": [short(p) for p in OUTPUTS if p.exists()]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

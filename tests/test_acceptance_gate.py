"""Counterexamples for misleading or incomplete rollout acceptance evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import acceptance_gate as G


def write_checkpoint(path, step=11999, count=100):
    import torch
    params = {k: torch.zeros((count, width)) for k, width in (
        ("means", 3), ("scales", 3), ("quats", 4), ("features_dc", 3), ("opacities", 1))}
    params["quats"][:, 0] = 1
    params["features_rest"] = torch.zeros((count, 15, 3))
    torch.save({"step": step, "pipeline": {"_model.gauss_params." + k: v for k, v in params.items()}}, path)


def write_ply(path, count=100, offset=0):
    from package_garage import REQUIRED, ply_header
    names = (*REQUIRED, *("f_rest_%d" % i for i in range(45)))
    records = np.zeros((count, len(names)), dtype="<f4")
    records[:, names.index("rot_0")] = 1
    records[:, 0] = offset
    path.write_bytes(ply_header(names, count) + records.tobytes())


@pytest.fixture
def evidence(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/acceptance.experimental.yaml").read_text())
    plan = {"plan_id": "a" * 64, "gate_thresholds": cfg["gates"], "frames": [], "mask_valid_pixels": {}}
    bundle = {role: [] for role in ("baseline", "repeat", "candidate", "single_block", "surface", "training")}
    model_file = tmp_path / "trained.ply"; write_ply(model_file)
    baseline_file = tmp_path / "baseline.ply"; write_ply(baseline_file, offset=1)
    baseline_sha = G.sha256_file(baseline_file)
    model_sha = G.sha256_file(model_file)
    checkpoint = tmp_path / "trained.ckpt"; write_checkpoint(checkpoint)
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"pipeline": {"model": {"inria_backend": "original",
        "use_inria_surfel_renderer": True, "sh_degree": 3, "sh_degree_interval": 1000}}}))
    transforms = tmp_path / "transforms.json"; transforms.write_text("{}")
    inputs = {name: str(path) for name, path in (("config", config), ("checkpoint", checkpoint), ("transforms", transforms))}
    inputs.update({name + "_sha256": G.sha256_file(path) for name, path in list(inputs.items())})
    for i, bid in enumerate(cfg["gates"]["G2"]["f1_blocks"]):
        camera = {"w": 8, "h": 8, "fl_x": 4., "fl_y": 4., "cx": 4., "cy": 4.,
                  "physical_camera_id": 3, "camera_to_worlds_ns": [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]}
        frozen = {"block_id": bid, "dataset_index": i, "group_id": str(i + 100), "camera_id": "camera_3",
                  "file_path": f"images/{i}.png", "mask_path": f"/ego/{i}.png", "camera": camera,
                  "transforms_sha256": inputs["transforms_sha256"], **{name: f"/{name}/{i}" for name in (
                      "source_image_path", "source_mask_path", "source_lidar_depth_path", "source_lidar_support_path")}}
        plan["frames"].append(frozen); plan["mask_valid_pixels"][frozen["mask_path"]] = 50
        for role in bundle:
            if role in ("surface", "training"): continue
            psnr = 21. if role in ("baseline", "repeat") else 24. if role == "candidate" else 24.5
            sha = baseline_sha if role in ("baseline", "repeat") else model_sha
            ply = baseline_file if role in ("baseline", "repeat") else model_file
            row = {**frozen, "acceptance_mask_path": frozen["mask_path"], "render_seconds": .2,
                   "protocols": {name: {"rgb_mse": 10 ** (-psnr / 10), "psnr_db": psnr,
                       "rgb_valid_pixels": 64 if name == "raw" else 50, "ssim_valid_pixels": .8,
                       "ssim_compared_pixels": 40, "map_depth_eligible_pixels": 20,
                       "map_depth_compared_pixels": 18, "map_depth_missing_pixels": 2,
                       "map_depth_capped_relative_error_all_supported": .2} for name in G.PROTOCOLS},
                   "native_visibility": {"protocol": G.NATIVE_PROTOCOL, "source_records": 100,
                                         "active_records": 80, "active_index_sha256": "d" * 64}}
            report = {"status": "evaluated", "split": "dev", "acceptance_plan_id": plan["plan_id"],
                      "metric_domain": "float_renderer", "block_id": bid, "images": 1, "frames": [row],
                      **inputs, "source_ply_sha256": sha,
                      "gaussians": 100, "source_ply": {"path": str(ply), "sha256": sha, "gaussians": 100},
                      "checkpoint_is_template": True, "parameter_origin": "All Gaussian parameters replaced by source_ply",
                      "world_parser_scale": 1., "world_parser_transform": camera["camera_to_worlds_ns"],
                      "step": 11999, "inference_model_step": 11999, "inference_sh_degree": 3,
                      "backend": "hbb1-original-2dgs", "original_checkpoint_gaussians": 100,
                      "_evidence_path": str(tmp_path / role / bid / "metrics.json")}
            saved = Path(report["_evidence_path"]); saved.parent.mkdir(parents=True)
            saved.write_text(json.dumps({k: v for k, v in report.items() if not k.startswith("_")}))
            bundle[role].append(report)
        bundle["surface"].append({"block_id": bid, "acceptance_plan_id": plan["plan_id"],
            "metric_domain": "conditioned_geometry", "scope": {"frame_keys": [frozen], "pixel_stride": 1},
            "candidate_model_sha256": model_sha, "baseline_model_sha256": baseline_sha,
            "render_measurement": {"acceptance_plan_id": plan["plan_id"],
                "gate_compatible_settings": True, "gate_compatible_render_provenance": True,
                "fixed_denominator_id": "f" * 64,
                "protocol": {**G.GEOMETRY_PROTOCOL, "mask_protocol": "ego_v2", "reference_sha256": "e" * 64},
                "sources": {role: {"path": bundle[role][-1]["_evidence_path"],
                    "sha256": G.sha256_file(bundle[role][-1]["_evidence_path"])} for role in ("candidate", "baseline")},
                "source_evaluator_provenance": {role: {"metric_domain": "float_renderer", "acceptance_plan_id": plan["plan_id"],
                    "backend": "hbb1-original-2dgs", "source_ply_sha256": sha}
                    for role, sha in (("candidate", model_sha), ("baseline", baseline_sha))}},
            "gate_evidence": {
                "G3": {"status": "known", "opacity_median": .8, "opacity_frac_lt_0_5": .1,
                       "voxel_size_m": .02, "voxel_2cm_multiplicity_median": 2},
                "G4": {"status": "known", "candidate_p50_m": .1, "baseline_p50_m": .2,
                       "reference_sha256": "e" * 64, "shared_reference": True, "independent_ground_truth": False,
                       "candidate_points": 200, "baseline_points": 100},
                "G5": {"status": "known", "reference_sha256": "e" * 64, "fixed_denominator": True, "fixed_denominator_id": "f" * 64,
                       **{name: {"eligible_pixels": 100, "candidate_hole_pixels": 20, "baseline_hole_pixels": 30}
                          for name in ("floor", "wall")}},
                "G7": {"status": "known", "tangential_radius_p95_m": .03, "tangential_radius_max_m": .1,
                       "seam_gap_db": .6, "seam_band_width_px": 64, "seam_compared_pixels": 300}}})
        bundle["training"].append({"block_id": bid, "status": "checkpoint_verified", "all_tensors_finite": True,
            "checkpoint_is_template": False, "completed_iterations": 12000, "total_iterations": 12000,
            "step": 11999, "inference_model_step": 11999, "remaining_iterations": 0, "refine_events": 50,
            "optimizer_state_verified": True, "parameter_update_l2": .1, "export_verified": True,
            "checkpoint": str(checkpoint), "checkpoint_sha256": G.sha256_file(checkpoint),
            "export": {"path": str(model_file), "sha256": model_sha}})
    bundle["pixel_closure"] = [{"status": "passed", "wrong_index_count": 0, "foreground_array_equal_fraction": 1.,
                               "tested_index_count": 1716, "pytest_returncode": 0,
                               "commands": [["pytest", "closure.py"]], "source_sha256": {"closure.py": "d" * 64}}]
    return plan, bundle


def test_all_measured_gates_pass_but_do_not_claim_six_floor_fidelity(evidence):
    plan, bundle = evidence
    result = G.evaluate_gate(plan, bundle)
    assert result["overall"] == "PASS", result
    assert result["rollout_allowed"] is True and result["high_fidelity_accepted"] is False
    # source_ply can be a genuinely trained export; both evaluation and training
    # proofs now bind the actual stored checkpoint inference step.
    assert result["gates"]["G8"]["evidence"]["source_ply_is_valid_trained_export"]


def test_every_missing_measurement_is_blocked(evidence):
    plan, _ = evidence
    result = G.evaluate_gate(plan, {})
    assert result["overall"] == "BLOCKED" and not result["rollout_allowed"]
    assert {v["status"] for v in result["gates"].values()} == {"BLOCKED"}


@pytest.mark.parametrize("fault", ["duplicate", "missing", "mask", "domain", "old_plan", "sample", "nan", "denominator", "camera"])
def test_renderer_identity_and_full_scope_faults_fail_closed(evidence, fault):
    plan, bundle = evidence
    report = bundle["candidate"][0]; row = report["frames"][0]
    if fault == "duplicate": bundle["candidate"].append(copy.deepcopy(report))
    elif fault == "missing": bundle["candidate"].pop()
    elif fault == "mask": row["acceptance_mask_path"] = "/old-mask.png"
    elif fault == "domain": report["metric_domain"] = "png_8bit_diagnostic"
    elif fault == "old_plan": report["acceptance_plan_id"] = "f" * 64
    elif fault == "sample": row["native_visibility"]["source_records"] = 80
    elif fault == "nan": row["protocols"]["ego_v2"]["psnr_db"] = float("nan")
    elif fault == "denominator": row["protocols"]["ego_v2"]["rgb_valid_pixels"] = 40
    else:
        row["camera"] = copy.deepcopy(row["camera"]); row["camera"]["cx"] += 1
    result = G.evaluate_gate(plan, bundle)
    assert result["overall"] == "FAIL" and result["gates"]["G2"]["status"] == "FAIL"


def test_repeat_must_not_be_same_report_reused(evidence):
    plan, bundle = evidence
    bundle["repeat"] = bundle["baseline"]
    assert G.evaluate_gate(plan, bundle)["gates"]["G1"]["status"] == "FAIL"


def test_repeat_is_strictly_less_than_one_micro_db(evidence):
    plan, bundle = evidence
    score = bundle["repeat"][0]["frames"][0]["protocols"]["ego_v2"]
    score["psnr_db"] += 1.1e-6; score["rgb_mse"] = 10 ** (-score["psnr_db"] / 10)
    assert G.evaluate_gate(plan, bundle)["gates"]["G1"]["status"] == "FAIL"


@pytest.mark.parametrize("gate,key,value", [("G3", "opacity_median", .69), ("G3", "opacity_frac_lt_0_5", .16),
    ("G3", "voxel_size_m", .03), ("G3", "voxel_2cm_multiplicity_median", 1.9),
    ("G4", "candidate_p50_m", .16), ("G4", "baseline_p50_m", 0),
    ("G4", "shared_reference", False), ("G4", "candidate_points", 0),
    ("G7", "tangential_radius_max_m", .121), ("G7", "seam_gap_db", .71)])
def test_failed_surface_target_blocks_everything(evidence, gate, key, value):
    plan, bundle = evidence
    bundle["surface"][0]["gate_evidence"][gate][key] = value
    result = G.evaluate_gate(plan, bundle)
    assert result["gates"][gate]["status"] == "FAIL" and not result["rollout_allowed"]


def test_missing_seam_cannot_inherit_radius_pass(evidence):
    plan, bundle = evidence
    del bundle["surface"][0]["gate_evidence"]["G7"]["seam_gap_db"]
    assert G.evaluate_gate(plan, bundle)["gates"]["G7"]["status"] == "BLOCKED"


@pytest.mark.parametrize("p95,maximum,status", [(.03, .12000000476837158, "PASS"),
    (.03, .12000002, "FAIL"), (.12000000476837158, .12000000476837158, "FAIL")])
def test_float32_radius_cap_roundoff_does_not_relax_real_geometry(evidence, p95, maximum, status):
    plan, bundle = evidence
    bundle["surface"][0]["gate_evidence"]["G7"].update(tangential_radius_p95_m=p95, tangential_radius_max_m=maximum)
    result = G.evaluate_gate(plan, bundle)["gates"]["G7"]
    assert result["status"] == status
    if status == "PASS": assert result["evidence"]["radius_roundoff_atol_m"] == 1e-8


def test_floor_improves_but_wall_fails(evidence):
    plan, bundle = evidence
    bundle["surface"][0]["gate_evidence"]["G5"]["wall"]["candidate_hole_pixels"] = 22
    assert G.evaluate_gate(plan, bundle)["gates"]["G5"]["status"] == "FAIL"


def set_psnr(report, psnr):
    for value in report["frames"][0]["protocols"].values():
        value.update(psnr_db=psnr, rgb_mse=10 ** (-psnr / 10))


def test_four_improvements_are_required_not_just_a_large_average(evidence):
    plan, bundle = evidence
    for report in bundle["candidate"][:2]: set_psnr(report, 22.)
    set_psnr(bundle["candidate"][-1], 60.)
    assert G.evaluate_gate(plan, bundle)["gates"]["G2"]["status"] == "FAIL"


def test_g6_worst_block_loss_cannot_be_hidden_by_other_blocks(evidence):
    plan, bundle = evidence
    set_psnr(bundle["single_block"][0], 25.1)
    for report in bundle["single_block"][1:]: set_psnr(report, 10.)
    assert G.evaluate_gate(plan, bundle)["gates"]["G6"]["status"] == "FAIL"


def test_unknown_geometry_and_partial_training_stay_blocked(evidence):
    plan, bundle = evidence
    bundle["surface"][0]["gate_evidence"]["G4"]["status"] = "unknown"
    bundle["training"].pop()
    result = G.evaluate_gate(plan, bundle)
    assert result["gates"]["G4"]["status"] == result["gates"]["G8"]["status"] == "BLOCKED"


@pytest.mark.parametrize("name,value", [("pixel_stride", 2), ("depth_tolerance_m", 1.),
    ("reference_match_max_m", 1.), ("render_pixel_center", 0.), ("wall_abs_normal_z_max", .9)])
def test_forged_compatible_flag_cannot_hide_changed_geometry_settings(evidence, name, value):
    plan, bundle = evidence
    bundle["surface"][0]["render_measurement"]["protocol"][name] = value
    result = G.evaluate_gate(plan, bundle)
    assert result["gates"]["G4"]["status"] == result["gates"]["G5"]["status"] == "FAIL"


def test_geometry_diagnostic_or_old_renderer_provenance_is_never_accepted(evidence):
    plan, bundle = evidence
    measurement = bundle["surface"][0]["render_measurement"]
    measurement["source_evaluator_provenance"]["baseline"]["metric_domain"] = None
    result = G.evaluate_gate(plan, bundle)
    assert result["gates"]["G4"]["status"] == result["gates"]["G5"]["status"] == "FAIL"


def test_hole_denominator_id_must_match_measurement(evidence):
    plan, bundle = evidence
    bundle["surface"][0]["gate_evidence"]["G5"]["fixed_denominator_id"] = "a" * 64
    assert G.evaluate_gate(plan, bundle)["gates"]["G5"]["status"] == "FAIL"


def test_same_source_fixed_denominator_is_required(evidence):
    plan, bundle = evidence
    bundle["single_block"][0]["frames"][0]["protocols"]["ego_v2"]["map_depth_eligible_pixels"] -= 1
    assert G.evaluate_gate(plan, bundle)["gates"]["G6"]["status"] == "FAIL"


@pytest.mark.parametrize("key,value", [("inference_model_step", 30000), ("completed_iterations", 1000),
    ("step", 12000), ("refine_events", 0), ("checkpoint_is_template", True),
    ("parameter_update_l2", 0.), ("optimizer_state_verified", False), ("export_verified", False)])
def test_training_authenticity_cannot_be_substituted_with_ply_render(evidence, key, value):
    plan, bundle = evidence
    bundle["training"][0][key] = value
    assert G.evaluate_gate(plan, bundle)["gates"]["G8"]["status"] == "FAIL"


def test_checkpoint_artifact_change_invalidates_provenance(evidence):
    plan, bundle = evidence
    Path(bundle["training"][0]["checkpoint"]).write_bytes(b"other checkpoint")
    assert G.evaluate_gate(plan, bundle)["gates"]["G8"]["status"] == "FAIL"


def test_controller_rejects_bare_pass_or_stale_evidence(evidence, tmp_path):
    plan, bundle = evidence
    result = G.evaluate_gate(plan, bundle)
    with pytest.raises(ValueError): G.require_rollout({"overall": "PASS"}, plan["plan_id"])
    proof = tmp_path / "proof.json"; proof.write_text("{}")
    result["evidence_files"] = {str(proof): G.sha256_file(proof), **result["verified_asset_files"]}
    result["report_id"] = G.object_sha(result)
    assert G.require_rollout(result, plan["plan_id"])
    proof.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed/missing"): G.require_rollout(result, plan["plan_id"])


def test_manifest_expands_actual_scene_children_and_checks_digest(tmp_path):
    child = tmp_path / "block_000/metrics.json"; child.parent.mkdir()
    child.write_text('{"status":"evaluated"}')
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"blocks": [{"path": str(child), "sha256": G.sha256_file(child)}]}))
    evidence, sources = G.read_reports({"candidate": [str(summary)]}, tmp_path)
    assert len(evidence["candidate"]) == 1 and len(sources) == 2
    child.write_text('{"status":"tampered"}')
    with pytest.raises(ValueError, match="digest differs"): G.read_reports({"candidate": str(summary)}, tmp_path)


def test_manifest_pins_model_and_closure_sources_not_just_json(tmp_path):
    source = tmp_path / "closure.py"; source.write_text("# CPU oracle\n")
    ply = tmp_path / "model.ply"; ply.write_text("synthetic fixture model")
    report = tmp_path / "metrics.json"
    report.write_text(json.dumps({"source_ply": {"path": str(ply), "sha256": G.sha256_file(ply)},
                                 "source_sha256": {str(source): G.sha256_file(source)}}))
    _, sources = G.read_reports({"candidate": str(report)}, tmp_path)
    assert set(sources) == {str(source), str(ply), str(report)}
    source.write_text("# changed oracle\n")
    with pytest.raises(ValueError, match="asset digest differs"): G.read_reports({"candidate": str(report)}, tmp_path)


@pytest.fixture
def checkpoint_origin(tmp_path):
    assets = {}
    for name in ("config", "checkpoint", "transforms", "export"):
        path = tmp_path / name; path.write_bytes(("actual " + name).encode())
        if name == "config":
            path.write_text(yaml.safe_dump({"pipeline": {"model": {"inria_backend": "original",
                "use_inria_surfel_renderer": True, "sh_degree": 3, "sh_degree_interval": 1000}}}))
        elif name == "checkpoint":
            write_checkpoint(path)
        elif name == "export":
            write_ply(path, count=95)
        assets[name] = {"path": str(path), "sha256": G.sha256_file(path)}
    assets["export"].update(gaussians=95, verification=G.EXACT_EXPORT_METHOD)
    plan_id = "a" * 64
    metrics = {"status": "evaluated", "metric_domain": "float_renderer", "acceptance_plan_id": plan_id,
               "parameter_origin": "trained_checkpoint", "checkpoint_is_template": False,
               "source_ply": None, "source_ply_sha256": None, "step": 11999, "inference_model_step": 11999,
               "gaussians": 100, "export": {"gaussians": 95}, "inference_sh_degree": 3, "backend": "hbb1-original-2dgs"}
    for name in ("config", "checkpoint", "transforms"):
        metrics[name] = assets[name]["path"]; metrics[name + "_sha256"] = assets[name]["sha256"]
    path = tmp_path / "metrics.json"; path.write_text(json.dumps(metrics))
    assets["metrics"] = {"path": str(path), "sha256": G.sha256_file(path)}
    proof = {"schema_version": 2, "status": "checkpoint_verified", "block_id": "block_015",
             "checkpoint_is_template": False, "parameter_origin": "trained_checkpoint",
             "all_tensors_finite": True, "export_verified": True, "step": 11999, "inference_model_step": 11999,
             "completed_iterations": 12000, "gaussians": 100, "config": metrics["config"],
             "checkpoint": metrics["checkpoint"], "checkpoint_sha256": metrics["checkpoint_sha256"],
             "export": assets["export"], "source_sha256": {a["path"]: a["sha256"] for a in assets.values()}}
    proof_path = tmp_path / "proof.json"; proof_path.write_text(json.dumps(proof))
    binding = {"path": str(proof_path), "sha256": G.sha256_file(proof_path), **assets}
    origin = {"metric_domain": "float_renderer", "acceptance_plan_id": plan_id,
              "source_ply_sha256": None, "checkpoint_export_proof": binding}
    report = {"block_id": "block_015", "candidate_model_sha256": assets["export"]["sha256"],
              "candidate_representation_id": assets["checkpoint"]["sha256"]}
    return origin, report, plan_id, proof


def test_native_checkpoint_and_filtered_export_have_distinct_valid_identities(checkpoint_origin):
    origin, report, plan_id, proof = checkpoint_origin
    assert G.verify_checkpoint_export_origin(origin, report, "candidate", plan_id) == proof
    assert proof["gaussians"] == 100 and proof["export"]["gaussians"] == 95
    assert origin["source_ply_sha256"] is None


@pytest.mark.parametrize("fault", ["count_only", "template", "proof_digest", "source_drift", "step", "wrong_metrics",
                                   "export_count", "unpinned_transforms", "representation"])
def test_checkpoint_geometry_rejects_incomplete_or_changed_proof(checkpoint_origin, fault):
    origin, report, plan_id, proof = checkpoint_origin
    binding = origin["checkpoint_export_proof"]
    if fault == "count_only": proof["export"]["verification"] = "checkpoint_export_count_only"
    elif fault == "template": proof["checkpoint_is_template"] = True
    elif fault == "proof_digest": binding["sha256"] = "f" * 64
    elif fault == "source_drift": Path(proof["checkpoint"]).write_bytes(b"changed checkpoint")
    elif fault == "step": proof["inference_model_step"] = 30000
    elif fault == "wrong_metrics": binding["metrics"]["sha256"] = "f" * 64
    elif fault == "export_count": proof["export"]["gaussians"] = 96
    elif fault == "unpinned_transforms": del proof["source_sha256"][binding["transforms"]["path"]]
    else: report["candidate_representation_id"] = report["candidate_model_sha256"]
    if fault not in ("proof_digest", "source_drift", "wrong_metrics", "representation"):
        Path(binding["path"]).write_text(json.dumps(proof))
        binding["sha256"] = G.sha256_file(binding["path"])
    with pytest.raises(ValueError): G.verify_checkpoint_export_origin(origin, report, "candidate", plan_id)


def test_read_reports_accepts_null_source_ply_and_pins_checkpoint_proof(checkpoint_origin, tmp_path):
    origin, report, plan_id, _ = checkpoint_origin
    surface = tmp_path / "surface.json"
    surface.write_text(json.dumps({"source_ply": None, "render_measurement": {
        "source_evaluator_provenance": {"candidate": origin}}}))
    _, sources = G.read_reports({"surface": str(surface)}, tmp_path)
    assert len(sources) == 7  # surface, proof, metrics, checkpoint, config, transforms, export


def v2_parameter_proof():
    return {"schema_version": 2, "optimizer_state_verified": True, "parameter_update_l2": None,
            "completed_iterations": 12000, "initial_gaussians": 70, "gaussians": 100,
            "config_launch_consistent": True, "source_sha256": {"config": "c" * 64},
            "parameter_change_evidence": {"schema_version": 1, "method": "optimizer_moments_and_parameter_hash_change",
                "verified": True, "parameter_hash_changed": True, "initial_parameter_sha256": "a" * 64,
                "final_parameter_sha256": "b" * 64, "initial_gaussians": 70, "final_gaussians": 100,
                "optimizer_nonzero_groups": sorted(G.PARAMETER_GROUPS)},
            "optimizer_moments": {name: {"exp_avg_nonzero": 1, "exp_avg_sq_nonzero": 1,
                "exp_avg_max_abs": .01, "exp_avg_sq_max_abs": .001, "adam_step": 12000.}
                for name in G.PARAMETER_GROUPS}}


def test_v2_parameter_change_does_not_invent_l2_without_persistent_ids():
    proof = v2_parameter_proof()
    G.validate_parameter_change(proof)
    assert proof["parameter_update_l2"] is None


@pytest.mark.parametrize("fault", ["same_hash", "zero_moments", "wrong_optimizer_step", "missing_group", "invalid_teacher", "changed_count"])
def test_v2_hash_change_needs_real_optimizer_and_provenance(fault):
    proof = v2_parameter_proof()
    change = proof["parameter_change_evidence"]
    if fault == "same_hash": change["final_parameter_sha256"] = change["initial_parameter_sha256"]
    elif fault == "zero_moments": proof["optimizer_moments"]["means"]["exp_avg_nonzero"] = 0
    elif fault == "wrong_optimizer_step": proof["optimizer_moments"]["means"]["adam_step"] = 1000.
    elif fault == "missing_group": change["optimizer_nonzero_groups"].remove("means")
    elif fault == "invalid_teacher": proof["diagnostic_only"] = True
    else: change["final_gaussians"] = 99
    with pytest.raises(ValueError): G.validate_parameter_change(proof)

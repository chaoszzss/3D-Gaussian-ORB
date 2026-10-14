"""Real CPU file counterexamples for representation/backend/evaluation binding.

Tiny PLYs/checkpoints and NumPy side arrays are real serialized artifacts. The
AA contract fixture is metadata only: no compiled extension or GPU is loaded.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_acceptance_gate import G, evidence, write_checkpoint
import direction_support as D
import measure_surface_quality as S
import test_measure_surface_quality as surface_tests


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def save_report(report):
    path = Path(report["_evidence_path"])
    write_json(path, {k: v for k, v in report.items() if not k.startswith("_")})
    return path


def seal_side(report, path, metadata):
    """Rehash even forged descriptors, so negative tests reach semantic checks."""
    write_json(path, metadata)
    report["direction_support"] = {
        "path": str(path), "sha256": G.sha256_file(path),
        "side_flags_sha256": metadata["side_flags"]["sha256"],
        "source_ply_sha256": report["source_ply_sha256"], "source_records": report["gaussians"],
        "normal_convention": metadata["normal_convention"], "render_grazing_cosine": metadata["render_grazing_cosine"],
        "model_kind": "static_2dgs_with_train_observation_side_support",
        "policy": "0/3 retain both sides; unilateral support rejects only opposite non-grazing viewing direction"}
    report["representation_id"] = hashlib.sha256((report["source_ply_sha256"] + ":" + G.sha256_file(path)).encode()).hexdigest()
    report["model_representation"] = "PLY plus fixed TRAIN observation-side support; original native 2DGS kernels"
    for row in report["frames"]:
        row["native_visibility"]["direction_visibility"] = {
            "native_active_records": 80, "side_rejected_records": 20,
            "retained_records": 60, "retained_index_sha256": "e" * 64}


@pytest.fixture
def side_evidence(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ROOT", tmp_path)
    plan, bundle = evidence
    def array(name, values, dtype):
        path = tmp_path / (name + ".npy")
        np.save(path, np.asarray(values * 25, dtype=dtype))
        return {"path": str(path), "sha256": G.sha256_file(path), "dtype": dtype, "count": 100}
    meta = {"schema_version": 1, "ply": bundle["candidate"][0]["source_ply"],
        "normal_convention": "quaternion_local_z", "policy": "strong_unilateral_train_group_support",
        "render_grazing_cosine": .05, "side_flags": array("flags", [0, 1, 2, 3], "uint8"),
        "training_evidence": {"split": "train", "train_only": True, "heldout_used": False,
            "minimum_unique_groups": 3, "opposite_side_required_count": 0,
            "positive_train_groups": array("positive", [0, 3, 0, 1], "uint16"),
            "negative_train_groups": array("negative", [0, 0, 3, 1], "uint16")}}
    path = tmp_path / "side.json"
    for report in bundle["candidate"]:
        seal_side(report, path, meta)
    return plan, bundle, path, meta


def test_plain_historical_identity_is_implicit_but_explicit_wrong_is_rejected(evidence):
    plan, bundle = evidence
    rendered = G.validate_render_reports(plan, bundle["candidate"])
    sha = bundle["candidate"][0]["source_ply_sha256"]
    assert set(rendered["models"].values()) == {sha}
    for r in bundle["candidate"]:
        r["representation_id"] = sha
    G.validate_render_reports(plan, bundle["candidate"])
    bundle["candidate"][0]["representation_id"] = "e" * 64
    with pytest.raises(ValueError, match="Explicit representation"):
        G.validate_render_reports(plan, bundle["candidate"])


def test_direction_identity_uses_complete_ply_and_sidecar_and_pins_all_train_arrays(side_evidence, tmp_path):
    plan, bundle, path, meta = side_evidence
    report = bundle["candidate"][0]
    cache = {}
    rendered = G.validate_render_reports(plan, bundle["candidate"], asset_cache=cache)
    assert set(rendered["models"].values()) == {report["representation_id"]}
    assert set(rendered["base_models"].values()) == {report["source_ply_sha256"]}
    assert rendered["models"] != rendered["base_models"]
    saved = save_report(report)
    _, pins = G.read_reports({"candidate": str(saved)}, tmp_path)
    side_paths = [path, Path(meta["side_flags"]["path"]),
                  *(Path(meta["training_evidence"][name]["path"]) for name in ("positive_train_groups", "negative_train_groups"))]
    assert all(pins[str(p)] == G.sha256_file(p) for p in side_paths)
    assert all(cache["_pins"][str(p)] == G.sha256_file(p) for p in side_paths)


@pytest.mark.parametrize("asset", ["metadata", "flags", "positive_train_groups", "negative_train_groups"])
def test_cached_sidecar_validation_still_detects_mutated_asset(side_evidence, asset):
    _, bundle, path, meta = side_evidence
    report, cache = bundle["candidate"][0], {}
    G.validate_representation(report, cache)
    target = path if asset == "metadata" else Path((meta["side_flags"] if asset == "flags" else meta["training_evidence"][asset])["path"])
    with target.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        G.validate_representation(report, cache)


@pytest.mark.parametrize("fault", ["heldout", "train_count", "flags", "dtype", "missing_count",
                                   "metadata_ply", "metadata_count", "reported_count", "identity",
                                   "partial_native", "retained_count", "drop_side", "grazing"])
def test_rehashed_sidecar_tampering_fails_semantic_validation(side_evidence, fault):
    plan, bundle, path, meta = side_evidence
    report = bundle["candidate"][0]
    if fault == "heldout": meta["training_evidence"]["heldout_used"] = True
    elif fault in ("train_count", "dtype"):
        desc = meta["training_evidence"]["positive_train_groups"]
        values = np.load(desc["path"]).copy()
        if fault == "train_count": values[1] = 2  # cannot justify flag 1 anymore
        else: values = values.astype(np.uint32)
        np.save(desc["path"], values); desc["sha256"] = G.sha256_file(desc["path"])
    elif fault == "flags":
        desc = meta["side_flags"]; values = np.load(desc["path"]).copy(); values[1] = 2
        np.save(desc["path"], values); desc["sha256"] = G.sha256_file(desc["path"])
    elif fault == "missing_count": Path(meta["training_evidence"]["negative_train_groups"]["path"]).unlink()
    elif fault == "metadata_ply": meta["ply"] = {**meta["ply"], "sha256": "b" * 64}
    elif fault == "metadata_count": meta["ply"] = {**meta["ply"], "gaussians": 99}
    elif fault == "grazing": meta["render_grazing_cosine"] = .3
    seal_side(report, path, meta)
    if fault == "reported_count": report["direction_support"]["source_records"] = 99
    elif fault == "identity": report["representation_id"] = report["source_ply_sha256"]
    elif fault == "partial_native": report["frames"][0]["native_visibility"]["source_records"] = 99
    elif fault == "retained_count": report["frames"][0]["native_visibility"]["direction_visibility"]["retained_records"] = 59
    elif fault == "drop_side":
        del report["direction_support"]; del report["representation_id"]
    with pytest.raises(ValueError):
        G.validate_render_reports(plan, [report], [report["block_id"]])


def test_direction_strip_cannot_rewrite_a_pinned_saved_report(side_evidence, tmp_path):
    plan, bundle, _, _ = side_evidence
    path = save_report(bundle["candidate"][0])
    loaded, _ = G.read_reports({"candidate": str(path)}, tmp_path)
    report = loaded["candidate"][0]
    for name in ("direction_support", "representation_id", "model_representation"):
        report.pop(name)
    report["frames"][0]["native_visibility"].pop("direction_visibility")
    with pytest.raises(ValueError, match="pinned saved evidence"):
        G.validate_render_reports(plan, [report], [report["block_id"]])


@pytest.mark.parametrize("variant", ["plain", "different_side"])
def test_whole_model_cannot_mix_direction_representations(side_evidence, tmp_path, variant):
    plan, bundle, _, meta = side_evidence
    report = bundle["candidate"][0]
    if variant == "plain":
        for name in ("direction_support", "representation_id", "model_representation"):
            report.pop(name)
        report["frames"][0]["native_visibility"].pop("direction_visibility")
    else:
        meta = copy.deepcopy(meta); meta["render_grazing_cosine"] = .10
        seal_side(report, tmp_path / "different.json", meta)
    with pytest.raises(ValueError, match="different complete representations"):
        G.validate_render_reports(plan, bundle["candidate"])


@pytest.mark.parametrize("fault", ["config_bytes", "checkpoint_bytes", "transforms_bytes", "config_digest",
    "checkpoint_digest", "declared_step", "synthetic_inference", "stored_step", "sh_degree", "backend_config"])
def test_evaluation_binding_checks_actual_assets_and_stored_step(evidence, fault):
    plan, bundle = evidence
    report = bundle["candidate"][0]
    if fault.endswith("_bytes"):
        with Path(report[fault[:-6]]).open("ab") as f: f.write(b"tampered")
    elif fault.endswith("_digest"): report[fault[:-7] + "_sha256"] = "b" * 64
    elif fault == "declared_step": report["step"] = report["inference_model_step"] = 30000
    elif fault == "synthetic_inference": report["inference_model_step"] = 30000
    elif fault == "stored_step":
        write_checkpoint(Path(report["checkpoint"]), step=999)
        report["checkpoint_sha256"] = G.sha256_file(report["checkpoint"])
    elif fault == "sh_degree": report["inference_sh_degree"] = 2
    else:
        path = Path(report["config"]); config = yaml.safe_load(path.read_text())
        config["pipeline"]["model"]["inria_backend"] = "aa-depth-vjp"
        path.write_text(yaml.safe_dump(config)); report["config_sha256"] = G.sha256_file(path)
    with pytest.raises(ValueError):
        G.validate_render_reports(plan, [report], [report["block_id"]])


def as_native(report):
    report["source_ply"] = report["source_ply_sha256"] = None
    report["checkpoint_is_template"] = False; report["parameter_origin"] = "trained_checkpoint"


def test_checkpoint_identity_remains_native_and_does_not_promote_to_whole_model(evidence):
    plan, bundle = evidence
    for report in bundle["single_block"]:
        as_native(report)
    rendered = G.validate_render_reports(plan, bundle["single_block"], whole_model=False)
    assert set(rendered["models"].values()) == {bundle["single_block"][0]["checkpoint_sha256"]}
    with pytest.raises(ValueError, match="Whole-scene report missing source PLY proof"):
        G.validate_render_reports(plan, bundle["single_block"])


@pytest.fixture
def aa_evidence(evidence, tmp_path, monkeypatch):
    # Reuse Euclid's bounded artifact-chain fixture, never its validation result.
    backend = surface_tests.SurfaceTests.renderer_fixture(SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(S, "ROOT", tmp_path)
    plan, bundle = evidence
    report = bundle["single_block"][0]
    report.update(backend); as_native(report)
    path = tmp_path / "aa-config.yml"
    config = yaml.safe_load(Path(report["config"]).read_text())
    config["pipeline"]["model"]["inria_backend"] = "aa-depth-vjp"
    path.write_text(yaml.safe_dump(config))
    report["config"] = str(path); report["config_sha256"] = G.sha256_file(path)
    return plan, report


def test_validated_aa_checkpoint_keeps_original_forward_contract_on_cpu(aa_evidence):
    import torch
    plan, report = aa_evidence
    assert not torch.cuda.is_initialized()
    cache = {}
    result = G.validate_render_reports(plan, [report], [report["block_id"]], whole_model=False, asset_cache=cache)
    assert set(result["models"].values()) == {report["checkpoint_sha256"]}
    assert len(cache["_pins"]) >= 23
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["no_provenance", "unknown_backend", "mislabel_original", "forward_contract",
    "validation_flag", "fd_digest", "extension_bytes", "source_bytes", "integration_bytes", "config_backend"])
def test_aa_report_requires_actual_same_helper_validation_chain(aa_evidence, fault):
    plan, report = aa_evidence
    p = report["renderer_provenance"]
    if fault == "no_provenance": report["renderer_provenance"] = None
    elif fault == "unknown_backend": report["backend"] = "approximate"
    elif fault == "mislabel_original": report["backend"] = "hbb1-original-2dgs"
    elif fault == "forward_contract": p["forward_contract"] = "center-depth fallback"
    elif fault == "validation_flag": p["validation"]["tested_forward_arrays_bitwise_equal"] = 1
    elif fault == "fd_digest": p["validation"]["finite_difference_report_sha256"] = "f" * 64
    elif fault in ("extension_bytes", "source_bytes"):
        path = Path(p["extension_path"]) if fault == "extension_bytes" else Path(p["source_directory"]) / "cuda_rasterizer/backward.cu"
        with path.open("ab") as f: f.write(b"different")
    elif fault == "integration_bytes":
        path = S.ROOT / "reports/inria_backend_integration_gpu_v1.json"
        value = json.loads(path.read_text()); value["forward_model_images_bitwise_equal"] = False; write_json(path, value)
    else:
        path = Path(report["config"]); value = yaml.safe_load(path.read_text())
        value["pipeline"]["model"]["inria_backend"] = "original"
        path.write_text(yaml.safe_dump(value)); report["config_sha256"] = G.sha256_file(path)
    with pytest.raises(ValueError):
        G.validate_render_reports(plan, [report], [report["block_id"]], whole_model=False)


def test_cached_aa_validation_rechecks_actual_extension_bytes(aa_evidence):
    _, report = aa_evidence
    cache = {}; G.validate_backend(report, cache)
    with Path(report["renderer_provenance"]["extension_path"]).open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        G.validate_backend(report, cache)


def test_geometry_uses_full_sidecar_identity_not_just_base_ply(side_evidence):
    plan, bundle, _, _ = side_evidence
    metrics, surface = bundle["candidate"][0], bundle["surface"][0]
    path = save_report(metrics)
    surface["render_measurement"]["sources"]["candidate"]["sha256"] = G.sha256_file(path)
    with pytest.raises(ValueError, match="complete representation"):
        G.validate_geometry_protocol(surface, plan["plan_id"])
    surface["candidate_representation_id"] = metrics["representation_id"]
    G.validate_geometry_protocol(surface, plan["plan_id"])


def test_geometry_cannot_relabel_its_pinned_renderer_backend(evidence):
    plan, bundle = evidence
    bundle["surface"][0]["render_measurement"]["source_evaluator_provenance"]["candidate"]["backend"] = "hbb1-aa-depth-vjp-2dgs"
    with pytest.raises(ValueError, match="provenance differs"):
        G.validate_geometry_protocol(bundle["surface"][0], plan["plan_id"])


def test_g1_same_psnr_with_changed_template_config_is_not_reproducibility(evidence, tmp_path):
    plan, bundle = evidence
    report = bundle["repeat"][0]
    path = tmp_path / "other-config.yml"
    path.write_text(Path(report["config"]).read_text() + "# different saved template\n")
    report["config"] = str(path); report["config_sha256"] = G.sha256_file(path)
    result = G.evaluate_gate(plan, bundle)
    assert result["gates"]["G1"]["status"] == "FAIL"
    assert "different renderer/config/checkpoint/inference" in result["gates"]["G1"]["reason"]


def test_rollout_cannot_drop_newly_verified_assets(evidence):
    plan, bundle = evidence
    result = G.evaluate_gate(plan, bundle)
    assert result["overall"] == "PASS"
    result["evidence_files"] = dict(result["verified_asset_files"])
    result["evidence_files"].pop(next(iter(result["evidence_files"])))
    result["report_id"] = G.object_sha(result)
    with pytest.raises(ValueError, match="omit or replace"):
        G.require_rollout(result, plan["plan_id"])

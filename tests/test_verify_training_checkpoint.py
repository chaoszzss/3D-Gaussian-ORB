"""CPU-only authenticity checks against tiny complete train_lidar_block artifacts."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_training_checkpoint as V
import evaluate_gaussian_ply as PLY


def native_export(parameters, path):
    # Execute the real export function without importing its GPU evaluator stack.
    source = ast.parse((ROOT / "scripts/evaluate_reconstruction.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "export_ply")
    module = ast.Module(body=[node], type_ignores=[])
    env = {"np": np}
    exec(compile(ast.fix_missing_locations(module), "native_export", "exec"), env)
    model = SimpleNamespace(config=SimpleNamespace(sh_degree=3), **{k: v.clone() for k, v in parameters.items()})
    return env["export_ply"](model, path)


def parameters(n):
    gen = torch.Generator().manual_seed(9)
    return {"means": torch.randn(n, 3, generator=gen), "scales": torch.ones(n, 3)*-3,
            "quats": torch.randn(n, 4, generator=gen), "features_dc": torch.randn(n, 3, generator=gen),
            "features_rest": torch.randn(n, 15, 3, generator=gen), "opacities": torch.zeros(n, 1)}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "ROOT", tmp_path)
    monkeypatch.setattr(PLY, "ROOT", tmp_path)
    directory = tmp_path / "run"
    saved = directory / "block_015/splatfacto/date"
    checkpoints = saved / "nerfstudio_models"
    checkpoints.mkdir(parents=True)
    dataset = tmp_path / "data"; dataset.mkdir()
    transforms = dataset / "transforms.json"; transforms.write_text('{"frames": []}')
    warm = tmp_path / "warm.ply"; native_export(parameters(3), warm)
    model = {"max_gaussians": 20, "sh_degree": 3, "enable_refinement": True,
             "warm_opacity_cap": 1., "gaussian_initialization_path": str(warm),
             "training_audit_path": str(directory / "refinement.json")}
    config = {"max_num_iterations": 12000, "experiment_name": "block_015", "output_dir": str(directory),
              "relative_model_dir": "nerfstudio_models", "pipeline": {"model": model,
              "datamanager": {"data": str(dataset), "dataparser": {"data": str(dataset),
              "auto_scale_poses": False, "orientation_method": "none", "center_method": "none", "scale_factor": 1.}}},
              "optimizers": {n: {"scheduler": {} if n == "means" else None} for n in V.GROUPS}}
    cfg = saved / "config.yml"; cfg.write_text(yaml.safe_dump(config))
    launch = {"iterations": 12000, "dataset": str(dataset), "world": "camera_init", "units": "m",
              "no_refine": False, "warm_ply": str(warm), "initialization_sha256": V.sha256(warm)}
    (directory / "launch.json").write_text(json.dumps(launch))
    pars = parameters(4)
    pars["opacities"][-1, 0] = -6  # Native export excludes this record but training has four.
    state = {"step": 11999, "pipeline": {"_model.gauss_params."+n: v for n, v in pars.items()},
             "optimizers": {}, "schedulers": {"means": {"last_epoch": 12000}}, "scalers": {}}
    for n, v in pars.items():
        state["optimizers"][n] = {"param_groups": [{"params": [0]}], "state": {0: {
            "step": torch.tensor(12000.), "exp_avg": torch.ones_like(v)*.01, "exp_avg_sq": torch.ones_like(v)*.001}}}
    checkpoint = checkpoints / "step-000011999.ckpt"; torch.save(state, checkpoint)
    audit = {"initial_gaussians": 3, "enabled": True, "final_gaussians": 4,
             "refine_events": 1, "last_callback_step": 11999,
             "events": [{"step": 600, "before": 3, "after": 4, "duplicated": 0, "split": 1, "pruned": 0}]}
    (directory / "refinement.json").write_text(json.dumps(audit))
    completion = {"status": "trained", "iterations_requested": 12000,
                  "configs": [str(cfg)], "checkpoint": str(checkpoint), "checkpoint_step": 11999,
                  "iterations_completed": 12000, "gaussians": 4, "checkpoint_sha256": V.sha256(checkpoint)}
    (directory / "training_result.json").write_text(json.dumps(completion))
    evaluation = tmp_path / "eval"; evaluation.mkdir()
    ply = evaluation / "gaussians.ply"
    export = native_export(pars, ply)
    report = {"status": "evaluated", "parameter_origin": "trained_checkpoint", "checkpoint_is_template": False,
              "source_ply": None, "config": str(cfg), "config_sha256": V.sha256(cfg),
              "checkpoint": str(checkpoint), "checkpoint_sha256": V.sha256(checkpoint), "step": 11999,
              "inference_model_step": 11999, "gaussians": 4, "export": export,
              "transforms": str(transforms), "transforms_sha256": V.sha256(transforms)}
    metrics = evaluation / "metrics.json"; metrics.write_text(json.dumps(report))
    return SimpleNamespace(directory=directory, state=state, checkpoint=checkpoint, metrics=metrics,
                           report=report, ply=ply, config=config, cfg=cfg, warm=warm, pars=pars)


def test_full_proof_checks_all_native_fields_and_never_invents_index_l2(run):
    result = V.verify_run(run.directory, run.metrics, threads=1)
    assert result["status"] == "checkpoint_verified", result
    assert result["export"]["gaussians"] == 3
    assert result["gaussians"] == 4 and result["initial_gaussians"] == 3
    assert result["parameter_update_l2"] is None
    change = result["parameter_change_evidence"]
    assert change["verified"] and change["topology_changed"] and change["parameter_hash_changed"]
    assert len(change["optimizer_nonzero_groups"]) == 6
    assert result["inference_model_step"] == 11999
    assert not torch.cuda.is_initialized()
    assert result["quality_status"] == "not_assessed"
    assert not result["high_fidelity_accepted"] and not result["promotion_allowed"]
    assert result["adoptable"] is None
    assert result["inria_backend"] == "original"
    assert result["renderer_verification"]["status"] == "historical_provenance_not_recorded"
    assert not result["renderer_verification"]["verified"]


def test_invalid_teacher_does_not_become_quality_accepted_when_authenticity_passes(run):
    path = run.directory / "experiment_disposition.json"
    disposition = {"status": "completed_diagnostic_only_invalid_depth_teacher", "adoptable": False,
                   "high_fidelity_accepted": False, "reason": "unknown-normal foreground admits background"}
    path.write_text(json.dumps(disposition))
    result = V.verify_run(run.directory, run.metrics, threads=1)
    assert result["status"] == "checkpoint_verified" and result["export_verified"]
    assert result["diagnostic_only"] and result["quality_status"] == "invalid_depth_teacher"
    assert result["adoptable"] is False and result["promotion_allowed"] is False
    assert result["high_fidelity_accepted"] is False and result["quality_acceptance_evaluated"] is False
    assert result["experiment_disposition"] == disposition
    assert result["source_sha256"][str(path)] == V.sha256(path)


def test_other_nonadoptable_disposition_is_preserved_without_claiming_teacher_failure(run):
    path = run.directory / "experiment_disposition.json"
    path.write_text(json.dumps({"status": "failed_geometry", "adoptable": False}))
    result = V.verify_run(run.directory, run.metrics, threads=1)
    assert result["quality_status"] == "failed_geometry"
    assert result["diagnostic_only"] and result["adoptable"] is False


@pytest.mark.parametrize("fault", ["missing_group", "moment_shape", "moment_nan", "negative_second_moment", "adam_step", "checkpoint_step", "scheduler"])
def test_corrupt_checkpoint_cannot_pass(run, fault):
    state = copy.deepcopy(run.state)
    if fault == "missing_group": del state["optimizers"]["means"]
    elif fault == "moment_shape": state["optimizers"]["means"]["state"][0]["exp_avg"] = torch.zeros(1, 3)
    elif fault == "moment_nan": state["optimizers"]["means"]["state"][0]["exp_avg"][0, 0] = float("nan")
    elif fault == "negative_second_moment": state["optimizers"]["means"]["state"][0]["exp_avg_sq"][0, 0] = -.01
    elif fault == "adam_step": state["optimizers"]["means"]["state"][0]["step"] = torch.tensor(12001.)
    elif fault == "checkpoint_step": state["step"] = 1000
    else: state["schedulers"] = {}
    torch.save(state, run.checkpoint)
    with pytest.raises(ValueError): V.verify_run(run.directory, run.metrics, threads=1)


def test_zero_optimizer_moments_are_not_evidence_of_training(run):
    state = copy.deepcopy(run.state)
    for v in state["optimizers"].values():
        for key in ["exp_avg", "exp_avg_sq"]: v["state"][0][key].zero_()
    pars, moments, _ = V.inspect_state(state, run.config["pipeline"]["model"], run.config, run.checkpoint, torch)
    assert all(s["exp_avg_nonzero"] == s["exp_avg_sq_nonzero"] == 0 for s in moments.values())


@pytest.mark.parametrize("field", ["nx", "f_dc_1", "f_rest_17", "opacity", "scale_2", "rot_3", "x"])
def test_each_export_parameter_family_is_compared_exactly(run, field):
    from package_garage import inspect_ply
    with run.ply.open("rb") as f: spec = inspect_ply(f)
    raw = bytearray(run.ply.read_bytes())
    offset = len(spec["header"]) + spec["names"].index(field)*4
    value = np.frombuffer(raw, dtype="<f4", count=1, offset=offset)
    value[0] += .25
    run.ply.write_bytes(raw)
    with pytest.raises(ValueError, match="field values"):
        V.verify_export(run.ply, run.pars)


@pytest.mark.parametrize("fault", ["synthetic_step", "template", "source_ply", "wrong_ckpt_sha", "wrong_config_sha"])
def test_eval_report_must_describe_the_actual_checkpoint(run, fault):
    report = copy.deepcopy(run.report)
    if fault == "synthetic_step": report["inference_model_step"] = 30000
    elif fault == "template": report["checkpoint_is_template"] = True
    elif fault == "source_ply": report["parameter_origin"] = "All Gaussian parameters replaced by source_ply"
    elif fault == "wrong_ckpt_sha": report["checkpoint_sha256"] = "0"*64
    else: report["config_sha256"] = "0"*64
    run.metrics.write_text(json.dumps(report))
    with pytest.raises(ValueError): V.verify_run(run.directory, run.metrics, threads=1)


def test_missing_export_stays_incomplete(run):
    result = V.verify_run(run.directory, threads=1)
    assert result["status"] == "incomplete_evidence"
    assert result["export_verified"] is False
    assert "evaluator_export_report_missing" in result["incomplete_reasons"]


def test_warm_and_pinned_config_drift_are_rejected(run):
    path = run.directory / "inputs.json"
    V.verify_run(run.directory, run.metrics, input_pin=path, threads=1)
    run.cfg.write_text(run.cfg.read_text()+"\n# changed\n")
    with pytest.raises(ValueError, match="Pinned"):
        V.verify_run(run.directory, run.metrics, input_pin=path, threads=1)
    run.warm.write_bytes(run.warm.read_bytes()+b"bad")
    with pytest.raises(ValueError, match="Warm PLY SHA"):
        V.verify_run(run.directory, run.metrics, threads=1)


@pytest.mark.parametrize("launch_key,config_key,value,changed", [
    ("metric_depth_weight", "lidar_metric_depth_loss_weight", .2, .1),
    ("metric_depth_delta_m", "lidar_metric_depth_huber_delta_m", .15, .3),
    ("stop_screen_size_at", "stop_screen_size_at", 12000, 4000),
    ("split_screen_size", "split_screen_size", .02, .05),
    ("vis", "vis", "tensorboard", "viewer"),
])
def test_quality_affecting_launch_settings_are_matched(run, launch_key, config_key, value, changed):
    launch = json.loads((run.directory / "launch.json").read_text())
    launch[launch_key] = value
    owner = run.config if launch_key == "vis" else run.config["pipeline"]["model"]
    owner[config_key] = value
    V.config_contract(run.config, launch, run.directory, run.cfg)
    owner[config_key] = changed
    with pytest.raises(ValueError, match="setting differs: " + launch_key):
        V.config_contract(run.config, launch, run.directory, run.cfg)
    owner.pop(config_key)
    # Explicit .15 is compatible with an old config omitting delta; the
    # remaining declared nondefault settings may not disappear from config.
    if launch_key != "metric_depth_delta_m":
        with pytest.raises(ValueError, match="setting differs: " + launch_key):
            V.config_contract(run.config, launch, run.directory, run.cfg)


@pytest.mark.parametrize("present", ["neither", "config", "launch", "both"])
def test_historical_metric_fields_use_fixed_disabled_and_point15_defaults(run, present):
    launch = json.loads((run.directory / "launch.json").read_text())
    if present in ("config", "both"):
        run.config["pipeline"]["model"].update(lidar_metric_depth_loss_weight=0., lidar_metric_depth_huber_delta_m=.15)
    if present in ("launch", "both"):
        launch.update(metric_depth_weight=0., metric_depth_delta_m=.15)
    V.config_contract(run.config, launch, run.directory, run.cfg)


@pytest.mark.parametrize("attr,value", [("lidar_metric_depth_loss_weight", .1),
                                       ("lidar_metric_depth_huber_delta_m", .2)])
def test_old_launch_cannot_hide_nondefault_metric_config(run, attr, value):
    launch = json.loads((run.directory / "launch.json").read_text())
    run.config["pipeline"]["model"][attr] = value
    with pytest.raises(ValueError, match="setting differs: metric_depth"):
        V.config_contract(run.config, launch, run.directory, run.cfg)


@pytest.mark.parametrize("launch_key,attr,value", [
    ("metric_depth_weight", "lidar_metric_depth_loss_weight", float("nan")),
    ("metric_depth_weight", "lidar_metric_depth_loss_weight", -1.),
    ("metric_depth_weight", "lidar_metric_depth_loss_weight", True),
    ("metric_depth_delta_m", "lidar_metric_depth_huber_delta_m", float("inf")),
    ("metric_depth_delta_m", "lidar_metric_depth_huber_delta_m", 0.),
])
def test_invalid_metric_settings_cannot_pass_even_if_launch_and_config_agree(run, launch_key, attr, value):
    launch = json.loads((run.directory / "launch.json").read_text())
    launch[launch_key] = run.config["pipeline"]["model"][attr] = value
    with pytest.raises(ValueError, match="Invalid metric depth setting"):
        V.config_contract(run.config, launch, run.directory, run.cfg)


def test_config_yaml_python_names_are_not_executed(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("!!python/object:nerfstudio.engine.trainer.TrainerConfig\nmax_num_iterations: 7\noutput_dir: !!python/object/apply:pathlib.PosixPath [/, tmp, run]\nevent: !!python/object/apply:nerfstudio.utils.writer.EventName [Train Rays / Sec]\n")
    result = V.read_config(path)
    assert result["max_num_iterations"] == 7 and result["output_dir"] == "/tmp/run"
    assert result["event"] == "Train Rays / Sec"
    path.write_text("!!python/object/apply:os.system ['echo unsafe']")
    with pytest.raises(yaml.constructor.ConstructorError): V.read_config(path)


@pytest.mark.parametrize("config_variant,launch_variant", [
    ("aa-depth-vjp", "original"), ("original", "aa-depth-vjp"),
    ("aa-depth-vjp", None), (None, "aa-depth-vjp"),
])
def test_backend_nondefault_or_missing_side_cannot_pass(run, config_variant, launch_variant):
    launch = json.loads((run.directory / "launch.json").read_text())
    if config_variant is not None:
        run.config["pipeline"]["model"]["inria_backend"] = config_variant
    if launch_variant is not None:
        launch["inria_backend"] = launch_variant
    with pytest.raises(ValueError, match="setting differs: inria_backend"):
        V.config_contract(run.config, launch, run.directory, run.cfg)


@pytest.mark.parametrize("variant", [None, "automatic", 0, ["original"]])
def test_backend_invalid_agreed_values_cannot_pass(variant):
    with pytest.raises(ValueError, match="Invalid inria_backend"):
        V.backend_variant_contract({"inria_backend": variant}, {"inria_backend": variant})


@pytest.mark.parametrize("model_enabled,launch_enabled", [(False, True), (True, False), (None, None)])
def test_aa_requires_inria_in_both_config_and_launch(model_enabled, launch_enabled):
    with pytest.raises(ValueError, match="requires enabled INRIA"):
        V.backend_variant_contract({"inria_backend": "aa-depth-vjp", "use_inria_surfel_renderer": model_enabled},
                                   {"inria_backend": "aa-depth-vjp", "renderer_inria": launch_enabled})


@pytest.fixture
def renderer_run(run, monkeypatch):
    """Tiny backend bytes; real helper is separately exercised on frozen artifacts."""
    root = V.ROOT
    source = root / "runtime/inria2dgs_aa_depth_vjp_src"
    def artifact(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    forward = artifact(source / "cuda_rasterizer/forward.cu", b"unchanged forward")
    wrapper = artifact(source / "diff_surfel_rasterization_aa_depth_vjp/__init__.py", b"wrapper")
    extension = artifact(wrapper.parent / "_C.so", b"validated binary")
    adapter = artifact(root / "src/adapter.py", b"adapter")
    artifact(root / "runtime/inria2dgs_src/cuda_rasterizer/forward.cu", forward.read_bytes())
    artifact(root / "runtime/inria2dgs_src/diff_surfel_rasterization/__init__.py", wrapper.read_bytes())
    sources = {"cuda_rasterizer/forward.cu": V.sha256(forward)}
    manifest = artifact(source / "manifest.json", json.dumps({"candidate_source_sha256": sources}).encode())
    validation = artifact(source / "validation.json", b'{"status":"PASS"}')
    fd_report = artifact(root / "reports/fd.json", b'{"candidate_status":"PASS"}')
    provenance = {"schema_version": 1, "variant": "aa-depth-vjp", "module": "aa_fixture",
                  "adapter_path": str(adapter), "adapter_sha256": V.sha256(adapter),
                  "wrapper_path": str(wrapper), "wrapper_sha256": V.sha256(wrapper),
                  "extension_path": str(extension), "extension_sha256": V.sha256(extension),
                  "source_directory": str(source), "source_sha256": sources,
                  "forward_contract": "unchanged", "validation": {
                      "validation_path": str(validation), "validation_sha256": V.sha256(validation),
                      "finite_difference_report": str(fd_report), "finite_difference_report_sha256": V.sha256(fd_report),
                      "build_manifest_path": str(manifest), "build_manifest_sha256": V.sha256(manifest),
                      "forward_source_byte_identical": True, "tested_forward_arrays_bitwise_equal": True}}
    frozen = copy.deepcopy(provenance)
    monkeypatch.setattr(V, "current_renderer_provenance", lambda variant: copy.deepcopy(frozen))
    run.config["pipeline"]["model"].update(inria_backend="aa-depth-vjp", use_inria_surfel_renderer=True)
    run.cfg.write_text(yaml.safe_dump(run.config))
    launch_path = run.directory / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch.update(inria_backend="aa-depth-vjp", renderer_inria=True, renderer_provenance=provenance)
    launch_path.write_text(json.dumps(launch))
    result_path = run.directory / "training_result.json"
    completion = json.loads(result_path.read_text())
    completion.update(inria_backend="aa-depth-vjp", renderer_provenance=provenance)
    result_path.write_text(json.dumps(completion))
    run.report.update(config_sha256=V.sha256(run.cfg), renderer_provenance=provenance, backend="hbb1-aa-depth-vjp-2dgs")
    run.metrics.write_text(json.dumps(run.report))
    run.renderer_records = {"launch": launch_path, "training_result": result_path, "evaluation": run.metrics}
    run.provenance = provenance
    return run


def test_aa_renderer_all_records_and_validation_bytes_are_pinned(renderer_run):
    run = renderer_run
    proof = V.verify_run(run.directory, run.metrics, threads=1)
    assert proof["status"] == "checkpoint_verified"
    renderer = proof["renderer_verification"]
    assert renderer["verified"] and renderer["records_verified"] == ["launch", "training_result", "evaluation"]
    assert proof["renderer_provenance"] == run.provenance
    for path in (run.provenance["extension_path"], run.provenance["validation"]["validation_path"],
                 run.provenance["validation"]["finite_difference_report"]):
        assert proof["source_sha256"][path] == V.sha256(path)
    assert not torch.cuda.is_initialized()


def test_joint_config_accepts_fixed_block015_evaluator_override_cli(renderer_run, monkeypatch):
    run = renderer_run
    training_data = V.ROOT / "data/depthv3_joint/core_015"
    training_data.mkdir(parents=True)
    (training_data / "transforms.json").write_text('{"frames": [], "fixture": "joint training"}')
    fixed_data = V.ROOT / "data/original/block_015"
    fixed_data.mkdir(parents=True)
    transforms = fixed_data / "transforms.json"
    transforms.write_text('{"frames": [], "fixture": "fixed block015 DEV"}')
    run.config["experiment_name"] = "core_015"
    manager = run.config["pipeline"]["datamanager"]
    manager["data"] = manager["dataparser"]["data"] = str(training_data)
    run.cfg.write_text(yaml.safe_dump(run.config))
    launch_path = run.directory / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["dataset"] = str(training_data)
    launch_path.write_text(json.dumps(launch))
    report = json.loads(run.metrics.read_text())
    report.update(config_sha256=V.sha256(run.cfg), transforms=str(transforms),
                  transforms_sha256=V.sha256(transforms))
    run.metrics.write_text(json.dumps(report))
    output = V.ROOT / "joint_proof.json"
    monkeypatch.setattr(sys, "argv", ["verify_training_checkpoint.py", "--training-dir", str(run.directory),
        "--evaluation-report", str(run.metrics), "--block-id", "block_015", "--threads", "1", "--output", str(output)])
    assert V.main() == 0
    proof = json.loads(output.read_text())
    assert proof["status"] == "checkpoint_verified" and proof["block_id"] == "block_015"
    assert V.read_config(proof["config"])["experiment_name"] == "core_015"
    assert proof["renderer_verification"]["verified"] and proof["inria_backend"] == "aa-depth-vjp"
    assert proof["source_sha256"][str(transforms)] == report["transforms_sha256"]
    assert not proof["gpu_used"] and not proof["high_fidelity_accepted"]
    # Dataset override is permitted, but its frozen source hash remains strict.
    transforms.write_text('{"frames": [], "fixture": "changed evaluation"}')
    with pytest.raises(ValueError, match="Evaluator dataset SHA differs"):
        V.verify_run(run.directory, run.metrics, block_id="block_015", threads=1)


@pytest.mark.parametrize("record", ["launch", "training_result", "evaluation"])
@pytest.mark.parametrize("fault", ["missing", "variant", "extension", "forward_validation"])
def test_renderer_record_mismatch_or_missing_is_rejected(renderer_run, record, fault):
    path = renderer_run.renderer_records[record]
    data = json.loads(path.read_text())
    if fault == "missing":
        data.pop("renderer_provenance")
    elif fault == "variant":
        data["renderer_provenance"]["variant"] = "original"
    elif fault == "extension":
        data["renderer_provenance"]["extension_sha256"] = "0" * 64
    else:
        data["renderer_provenance"]["validation"]["tested_forward_arrays_bitwise_equal"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="renderer_provenance"):
        V.verify_run(renderer_run.directory, renderer_run.metrics, threads=1)


@pytest.mark.parametrize("fault", ["training_variant", "missing_training_variant", "eval_label"])
def test_renderer_top_level_record_labels_cannot_disagree(renderer_run, fault):
    path = renderer_run.renderer_records["evaluation" if fault == "eval_label" else "training_result"]
    data = json.loads(path.read_text())
    if fault == "missing_training_variant":
        data.pop("inria_backend")
    else:
        data["backend" if fault == "eval_label" else "inria_backend"] = "original"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="backend.*differs"):
        V.verify_run(renderer_run.directory, renderer_run.metrics, threads=1)


@pytest.mark.parametrize("artifact", ["extension", "adapter", "wrapper", "forward", "validation", "fd_report"])
def test_agreed_provenance_cannot_hide_artifact_drift(renderer_run, artifact):
    p = renderer_run.provenance
    paths = {"extension": p["extension_path"], "adapter": p["adapter_path"], "wrapper": p["wrapper_path"],
             "forward": str(Path(p["source_directory"]) / "cuda_rasterizer/forward.cu"),
             "validation": p["validation"]["validation_path"], "fd_report": p["validation"]["finite_difference_report"]}
    Path(paths[artifact]).write_bytes(b"changed after recorded validation")
    with pytest.raises(ValueError, match="Renderer artifact SHA differs"):
        V.verify_run(renderer_run.directory, renderer_run.metrics, threads=1)


def test_agreed_forged_validation_still_requires_backend_helper_agreement(renderer_run):
    for path in renderer_run.renderer_records.values():
        record = json.loads(path.read_text())
        record["renderer_provenance"]["validation"]["tested_forward_arrays_bitwise_equal"] = False
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="current validated backend"):
        V.verify_run(renderer_run.directory, renderer_run.metrics, threads=1)


def test_renderer_bytes_changing_after_pin_are_rejected(renderer_run, monkeypatch):
    def changed(variant):
        Path(renderer_run.provenance["extension_path"]).write_bytes(b"replacement")
        return renderer_run.provenance
    monkeypatch.setattr(V, "current_renderer_provenance", changed)
    with pytest.raises(ValueError, match="Evidence changed"):
        V.verify_run(renderer_run.directory, renderer_run.metrics, threads=1)

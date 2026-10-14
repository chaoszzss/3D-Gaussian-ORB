#!/usr/bin/env python3
"""CPU verification of a train_lidar_block run and its actual-checkpoint export.

No renderer or Nerfstudio model is instantiated. Missing final/export evidence
is reported as incomplete; topology-changing training never gets an invented
indexwise parameter_update_l2. Run with the training Python (PyTorch required).
Authenticity is separate from reconstruction quality: experiment dispositions
are pinned and carried into the proof, and this verifier never authorizes rollout.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
GROUPS = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    path = (ROOT / Path(path).expanduser()).resolve()
    require(ROOT in path.parents, "Artifact must remain inside project: " + str(path))
    return path


def file_state(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def sha256(path):
    before = file_state(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(data)
    require(file_state(path) == before, "Artifact changed while hashing: " + str(path))
    return digest.hexdigest()


def read_config(path):
    """Decode saved YAML as plain data; never execute python/object constructors."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def object_fields(loader, suffix, node):
        value = loader.construct_mapping(node, deep=True)
        value["_serialized_class"] = suffix
        return value

    Loader.add_multi_constructor("tag:yaml.org,2002:python/object:", object_fields)
    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix)
    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node, deep=True))
    Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
                           lambda loader, node: str(Path(*loader.construct_sequence(node, deep=True))))
    Loader.add_constructor("tag:yaml.org,2002:python/object/apply:nerfstudio.utils.writer.EventName",
                           lambda loader, node: loader.construct_sequence(node, deep=True)[0])
    return yaml.load(Path(path).read_text(), Loader=Loader)


def backend_variant_contract(model, launch):
    # Fixed historical fallback; never inherit a future model/backend default.
    actual, declared = model.get("inria_backend", "original"), launch.get("inria_backend", "original")
    require(all(isinstance(v, str) and v in ("original", "aa-depth-vjp") for v in (actual, declared)),
            "Invalid inria_backend setting")
    require(actual == declared, "Config/launch setting differs: inria_backend")
    if actual != "original":
        require(model.get("use_inria_surfel_renderer") is True and launch.get("renderer_inria") is True,
                "AA backend requires enabled INRIA renderer in config and launch")
    return actual


def current_renderer_provenance(variant):
    """Import the existing identity/validation helper without a CUDA context."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    require(not torch.cuda.is_initialized(), "Renderer verification requires CPU-only process")
    from orbvi_gaussian.inria_surfel_backend import inria_backend_provenance
    provenance = inria_backend_provenance(variant)
    require(not torch.cuda.is_initialized(), "Renderer helper initialized CUDA")
    return provenance


def verify_renderer_contract(model, launch, result, report, pin):
    """Bind recorded identities to each other and current validated file bytes.

    Missing historical original provenance remains explicitly unverified. AA
    requires provenance on every available record; missing completion/evaluation
    records are handled by verify_run's existing incomplete-evidence reasons.
    """
    variant = backend_variant_contract(model, launch)
    records = {"launch": launch}
    if result is not None:
        records["training_result"] = result
    if report is not None:
        records["evaluation"] = report
    present = {}
    for name, record in records.items():
        if "inria_backend" in record or (name == "training_result" and variant != "original"):
            require(record.get("inria_backend") == variant, name + " inria_backend differs or missing")
        provenance = record.get("renderer_provenance")
        if variant != "original":
            require(isinstance(provenance, dict), name + " renderer_provenance missing for AA backend")
        if provenance is not None:
            require(isinstance(provenance, dict) and provenance.get("variant") == variant,
                    name + " renderer_provenance variant differs")
            require(model.get("use_inria_surfel_renderer") is True,
                    "Renderer provenance conflicts with disabled INRIA renderer")
            present[name] = provenance
    if report is not None and "backend" in report and model.get("use_inria_surfel_renderer") is True:
        require(report["backend"] == "hbb1-" + variant + "-2dgs", "Evaluator backend label differs")
    if not present:
        return {"variant": variant, "verified": False, "status": "historical_provenance_not_recorded",
                "records_verified": [], "provenance": None}
    expected = next(iter(present.values()))
    for name, provenance in present.items():
        require(provenance == expected, name + " renderer_provenance differs across records")

    def pin_digest(path, digest):
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Invalid renderer artifact SHA")
        require(sha256(pin(path)) == digest, "Renderer artifact SHA differs: " + str(path))

    for name in ("adapter", "wrapper", "extension"):
        pin_digest(expected[name + "_path"], expected[name + "_sha256"])
    source = local(expected["source_directory"])
    require(bool(expected["source_sha256"]), "Renderer source hashes missing")
    for relative, digest in expected["source_sha256"].items():
        pin_digest(source / relative, digest)
    if variant != "original":
        validation = expected.get("validation")
        require(isinstance(validation, dict), "AA forward validation missing")
        for path_key, sha_key in (("validation_path", "validation_sha256"),
                                  ("finite_difference_report", "finite_difference_report_sha256"),
                                  ("build_manifest_path", "build_manifest_sha256")):
            pin_digest(validation[path_key], validation[sha_key])
        manifest = json.loads(local(validation["build_manifest_path"]).read_text())
        for relative, digest in manifest["candidate_source_sha256"].items():
            pin_digest(source / relative, digest)
        # Also watch the original bytes used by the helper's forward/interface
        # equality checks; the pinned FD report supplies the tested baseline ID.
        pin_digest(ROOT / "runtime/inria2dgs_src/cuda_rasterizer/forward.cu",
                   expected["source_sha256"]["cuda_rasterizer/forward.cu"])
        pin_digest(ROOT / "runtime/inria2dgs_src/diff_surfel_rasterization/__init__.py",
                   expected["wrapper_sha256"])
    # The helper checks native FD PASS, forward/RGB-alpha bitwise equality,
    # source manifest and extension SHA. No rendering/compilation is performed.
    require(current_renderer_provenance(variant) == expected,
            "Recorded renderer provenance differs from current validated backend")
    return {"variant": variant, "verified": True, "status": "recorded_and_current_artifacts_verified",
            "records_verified": list(present), "provenance": expected}


def config_contract(config, launch, run, config_path):
    model, manager = config["pipeline"]["model"], config["pipeline"]["datamanager"]
    parser = manager["dataparser"]
    backend_variant_contract(model, launch)
    target = config["max_num_iterations"]
    require(type(target) is int and target > 0 and launch["iterations"] == target,
            "Config/launch iteration target differs")
    require(local(config["output_dir"]) == run and run in config_path.parents, "Config output location differs")
    require(local(manager["data"]) == local(parser["data"]) == local(launch["dataset"]),
            "Config/launch dataset differs")
    require(launch.get("world") == "camera_init" and launch.get("units") == "m", "Nonmetric or unknown world")
    require(parser["auto_scale_poses"] is False and parser["orientation_method"] == "none"
            and parser["center_method"] == "none" and parser["scale_factor"] == 1., "Parser changes metric world")
    pairs = {"max_gaussians": "max_gaussians", "warm_opacity_cap": "warm_opacity_cap",
             "refine_every": "refine_every", "stop_split_at": "stop_split_at", "warmup_length": "warmup_length",
             "stop_screen_size_at": "stop_screen_size_at", "split_screen_size": "split_screen_size",
             "gaussian_max_scale_m": "gaussian_max_scale_m", "num_downscales": "num_downscales",
             "depth_weight": "lidar_depth_loss_weight", "alpha_weight": "lidar_alpha_loss_weight",
             "surface_splats": "surface_splats", "renderer_inria": "use_inria_surfel_renderer",
             "renderer_2dgs": "use_2dgs_renderer", "optimize_shared_rig": "optimize_shared_rig",
             "reset_opacity": "reset_opacity_during_refinement", "tangent_scale_target_m": "tangent_scale_target_m",
             "tangent_scale_weight": "tangent_scale_loss_weight", "tangent_screen_weight": "tangent_screen_loss_weight",
             "tangent_max_diameter_px": "tangent_max_diameter_px"}
    for name, attr in pairs.items():
        if name in launch:
            require(model.get(attr) == launch[name], "Config/launch setting differs: " + name)
    # Historical configs/launch records predate this opt-in loss. Only those
    # absent fields get the documented disabled/default values, never today's
    # imported model defaults. A one-sided nondefault value must fail.
    for name, attr, default in (("metric_depth_weight", "lidar_metric_depth_loss_weight", 0.),
                                ("metric_depth_delta_m", "lidar_metric_depth_huber_delta_m", .15)):
        actual, declared = model.get(attr, default), launch.get(name, default)
        require(all(type(v) in (int, float) and math.isfinite(v) for v in (actual, declared)),
                "Invalid metric depth setting: " + name)
        require((actual >= 0 if name == "metric_depth_weight" else actual > 0),
                "Invalid metric depth setting: " + name)
        require(actual == declared, "Config/launch setting differs: " + name)
    if "vis" in launch:
        require(config.get("vis") == launch["vis"], "Config/launch setting differs: vis")
    require(model.get("enable_refinement") == (not launch["no_refine"]), "Config/launch refine flag differs")
    if "camera_optimizer_mode" in launch:
        require(model["camera_optimizer"]["mode"] == launch["camera_optimizer_mode"], "Camera optimizer setting differs")
    if "save_every" in launch:
        require(config["steps_per_save"] == launch["save_every"], "Checkpoint save schedule differs")
    warm = local(launch["warm_ply"]) if launch.get("warm_ply") else None
    require((local(model["gaussian_initialization_path"]) if model.get("gaussian_initialization_path") else None) == warm,
            "Config/launch warm source differs")
    if warm:
        require(sha256(warm) == launch.get("initialization_sha256"), "Warm PLY SHA differs from recorded initialization")
    require(local(model["training_audit_path"]) == run / "refinement.json", "Refinement audit path differs")
    return target, model, warm


def finite_state(value, torch):
    count = 0
    if torch.is_tensor(value):
        require(value.device.type == "cpu", "Verification must use CPU tensors")
        count += 1
        flat = value.reshape(-1)
        for start in range(0, flat.numel(), 1_000_000):
            require(torch.isfinite(flat[start:start+1_000_000]).all().item(), "Nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        count += sum(finite_state(v, torch) for v in value.values())
    elif isinstance(value, (list, tuple)):
        count += sum(finite_state(v, torch) for v in value)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite checkpoint scalar")
    return count


def inspect_state(state, model, config, checkpoint, torch):
    require({"step", "pipeline", "optimizers", "schedulers", "scalers"} <= set(state), "Missing checkpoint components")
    step = state["step"]
    require(type(step) is int and step >= 0 and int(checkpoint.stem[5:]) == step, "Checkpoint filename/stored step differs")
    parameters = {}
    for key, tensor in state["pipeline"].items():
        match = re.fullmatch(r"(?:module\.)?_model\.(?:module\.)?gauss_params\.(\w+)", key)
        if match:
            require(match[1] not in parameters, "Duplicate Gaussian tensor")
            parameters[match[1]] = tensor
    require(set(parameters) == set(GROUPS), "Missing or extra Gaussian parameter groups")
    count = len(parameters["means"])
    require(0 < count <= model["max_gaussians"], "Gaussian count outside configured budget")
    widths = {"means": 3, "scales": 3, "quats": 4, "features_dc": 3, "opacities": 1}
    for name, width in widths.items():
        require(tuple(parameters[name].shape) == (count, width), "Parameter shape mismatch: " + name)
    require(tuple(parameters["features_rest"].shape) == (count, (model["sh_degree"]+1)**2-1, 3), "SH shape mismatch")
    require(all(v.dtype == torch.float32 for v in parameters.values()), "Expected native float32 Gaussian parameters")
    require(torch.all(torch.linalg.vector_norm(parameters["quats"], dim=1) > 1e-12).item(), "Zero Gaussian quaternion")
    require(set(GROUPS) <= set(state["optimizers"]), "Missing six Gaussian Adam groups")
    expected_schedulers = {name for name in state["optimizers"] if config["optimizers"].get(name, {}).get("scheduler") is not None}
    require(set(state["schedulers"]) == expected_schedulers, "Checkpoint scheduler groups differ from enabled config")
    tensor_count = finite_state(state, torch)
    moments = {}
    for name in GROUPS:
        adam = state["optimizers"][name]
        ids = [pid for group in adam["param_groups"] for pid in group["params"]]
        require(len(ids) == 1 and set(adam["state"]) == set(ids), "Adam parameter ID/state mismatch: " + name)
        saved = adam["state"][ids[0]]
        adam_step = float(saved["step"])
        require(math.isfinite(adam_step) and adam_step.is_integer() and 0 < adam_step <= step+1, "Invalid Adam step: " + name)
        statistics = {}
        for moment in ["exp_avg", "exp_avg_sq"]:
            value = saved[moment]
            require(tuple(value.shape) == tuple(parameters[name].shape), "Adam moment shape mismatch: " + name)
            if moment == "exp_avg_sq":
                require(torch.all(value >= 0).item(), "Negative Adam second moment: " + name)
            statistics[moment + "_nonzero"] = int(torch.count_nonzero(value))
            statistics[moment + "_max_abs"] = float(value.abs().max()) if value.numel() else 0.
        moments[name] = dict(statistics, adam_step=adam_step)
    return parameters, moments, tensor_count


def inspect_refinement(audit, model, step, count):
    require(audit.get("enabled") == model["enable_refinement"], "Refinement enablement differs")
    initial = audit["initial_gaussians"]
    require(type(initial) is int and 0 < initial <= model["max_gaussians"], "Invalid initial Gaussian count")
    events = audit["events"]
    require(audit.get("refine_events") == len(events), "Refinement event count differs")
    last_step, previous = -1, initial
    for event in events:
        e = event["step"]
        require(type(e) is int and last_step < e <= step, "Refinement events newer than checkpoint or unordered")
        for key in ["before", "after", "duplicated", "split", "pruned"]:
            require(type(event[key]) is int and event[key] >= 0, "Invalid refinement event count")
        require(event["before"] == previous and event["after"] == event["before"] + event["duplicated"] + event["split"] - event["pruned"],
                "Refinement topology accounting differs")
        last_step, previous = e, event["after"]
    require(previous == count == audit["final_gaussians"], "Refinement final point count differs from checkpoint")
    require(audit.get("last_callback_step") == step, "Refinement audit progress differs from checkpoint")
    return len(events), initial


def parameter_digest(parameters):
    """Canonical ordered tensor digest, not a correspondence/movement metric."""
    import numpy as np
    digest = hashlib.sha256()
    for name in sorted(GROUPS):
        v = parameters[name]
        array = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        digest.update(json.dumps([name, list(array.shape)], separators=(",", ":")).encode())
        for start in range(0, len(array), 8192):
            digest.update(np.ascontiguousarray(array[start:start+8192], dtype="<f4").tobytes())
    return digest.hexdigest()


def verify_export(path, parameters):
    """Byte-exact native export, including SH order, normalized quats and normals."""
    import numpy as np
    from package_garage import inspect_ply
    rest = parameters["features_rest"].shape[1] * 3
    fields = (["x", "y", "z", "nx", "ny", "nz"] + ["f_dc_%d" % i for i in range(3)]
              + ["f_rest_%d" % i for i in range(rest)] + ["opacity"]
              + ["scale_%d" % i for i in range(3)] + ["rot_%d" % i for i in range(4)])
    before = file_state(path)
    digest, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        spec = inspect_ply(stream)
        require(list(spec["names"]) == fields, "Export fields/order differ from native evaluator")
        digest.update(spec["header"])
        for start in range(0, len(parameters["means"]), 8192):
            def array(name):
                return parameters[name][start:start+8192].detach().cpu().numpy()
            opacity, rotations = array("opacities"), array("quats").copy()
            rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
            w, x, y, z = rotations.T
            normals = np.column_stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
            values = np.concatenate([array("means"), normals, array("features_dc"),
                array("features_rest").transpose(0, 2, 1).reshape(len(opacity), rest), opacity,
                array("scales"), rotations], axis=1).astype("<f4")
            values = values[np.isfinite(values).all(1) & (opacity[:, 0] >= -5.5373)]
            raw = values.tobytes()
            require(stream.read(len(raw)) == raw, "Exported PLY field values differ from actual checkpoint")
            digest.update(raw)
            count += len(values)
        require(not stream.read(1) and spec["count"] == count, "Export record count/trailing bytes differ")
    require(file_state(path) == before, "Export changed during inspection")
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest(), "gaussians": count,
            "verification": "all field bytes exact; native opacity cutoff -5.5373; normalized wxyz; actual normals"}


def verify_run(training_dir, evaluation_report=None, export_ply=None, input_pin=None, block_id=None, threads=4):
    require(type(threads) is int and 1 <= threads <= 8, "CPU threads must be 1..8")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(threads)
    require(not torch.cuda.is_initialized(), "Verifier must not inherit an initialized CUDA context")
    run = local(training_dir)
    configs = sorted(run.rglob("config.yml"))
    require(len(configs) == 1, "Training directory must contain exactly one config.yml")
    config_path = local(configs[0])
    launch_path, audit_path = run / "launch.json", run / "refinement.json"
    watched, hashes = {}, {}
    def pin(path):
        path = local(path)
        if str(path) in watched:
            require(file_state(path) == watched[str(path)], "Previously pinned artifact changed: " + str(path))
            return path
        watched[str(path)] = file_state(path)
        hashes[str(path)] = sha256(path)
        return path
    for path in [config_path, launch_path, audit_path]:
        pin(path)
    config, launch = read_config(config_path), json.loads(launch_path.read_text())
    target, model, warm = config_contract(config, launch, run, config_path)
    if warm:
        pin(warm)
    fixed = {str(p): hashes[str(p)] for p in [config_path, launch_path] + ([warm] if warm else [])}
    if input_pin:
        pin_path = local(input_pin)
        if pin_path.exists():
            require(json.loads(pin_path.read_text())["source_sha256"] == fixed, "Pinned config/launch/warm SHA changed")
        else:
            pin_path.parent.mkdir(parents=True, exist_ok=True)
            with pin_path.open("x") as f:
                json.dump({"schema_version": 1, "source_sha256": fixed,
                           "note": "Pins bytes at verifier invocation; not retroactive proof of pretraining bytes"}, f, indent=2)
        pin(pin_path)
    checkpoint_dir = config_path.parent / config.get("relative_model_dir", "nerfstudio_models")
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"), key=lambda p: int(p.stem[5:]))
    require(checkpoints, "No saved checkpoint")
    checkpoint = pin(checkpoints[-1])
    # mmap keeps CPU RSS bounded; chunks below avoid full float64 moment copies.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    parameters, moments, tensor_count = inspect_state(state, model, config, checkpoint, torch)
    step, count = state["step"], len(parameters["means"])
    require(step < target, "Checkpoint exceeds requested iteration target")
    audit = json.loads(audit_path.read_text())
    refine_events, initial = inspect_refinement(audit, model, step, count)
    final_hash = parameter_digest(parameters)
    initial_hash = None
    if warm:
        from evaluate_gaussian_ply import read_candidate
        import numpy as np
        arrays, info = read_candidate(warm, max_gaussians=model["max_gaussians"], max_ply_bytes=2*1024**3)
        require(info["gaussians"] == initial, "Warm count differs from refinement initialization")
        cap = model["warm_opacity_cap"]
        require(0 < cap <= 1, "Invalid warm opacity cap")
        if cap < 1:
            arrays["opacities"] = np.minimum(arrays["opacities"], math.log(cap/(1-cap)))
        initial_hash = parameter_digest(arrays)
        del arrays
    active_groups = [n for n, s in moments.items() if s["exp_avg_nonzero"] > 0 and s["exp_avg_sq_nonzero"] > 0]
    topology_changed = any(e["duplicated"] or e["split"] or e["pruned"] for e in audit["events"])
    changed = (initial_hash != final_hash) if initial_hash else None
    change_verified = len(active_groups) == 6 and (changed is True or (initial_hash is None and topology_changed))
    change_evidence = {"schema_version": 1, "method": "optimizer_moments_and_parameter_hash_change" if warm else "optimizer_moments_and_refinement_topology",
                       "verified": bool(change_verified), "optimizer_nonzero_groups": active_groups,
                       "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash,
                       "parameter_hash_changed": changed, "topology_changed": bool(topology_changed),
                       "initial_gaussians": initial, "final_gaussians": count,
                       "correspondence": "none; persistent Gaussian IDs were not saved",
                       "l2_status": "not_measured_without_stable_correspondence; ordered digest is not displacement"}
    reasons = []
    if step+1 != target:
        reasons.append("training_iterations_incomplete")
    if not refine_events:
        reasons.append("no_observed_refinement_events")
    if not change_verified:
        reasons.append("insufficient_optimizer_and_parameter_change_evidence")
    result, report = None, None
    result_path = run / "training_result.json"
    if result_path.exists():
        result = json.loads(pin(result_path).read_text())
        require(result["status"] == "trained" and result["iterations_requested"] == target, "Training completion report differs")
        require([local(p) for p in result["configs"]] == [config_path], "Training report config differs")
        for key, expected in [("checkpoint_step", step), ("iterations_completed", step+1), ("gaussians", count),
                              ("checkpoint_sha256", hashes[str(checkpoint)])]:
            if key in result:
                require(result[key] == expected, "Training result differs: " + key)
        if result.get("checkpoint"):
            require(local(result["checkpoint"]) == checkpoint, "Training result checkpoint path differs")
        if "refinement" in result:
            require(result["refinement"] == audit, "Training result refinement snapshot differs")
    else:
        reasons.append("training_completion_report_missing")
    exported, inference_step = None, None
    if evaluation_report:
        report_path = pin(evaluation_report)
        report = json.loads(report_path.read_text())
        require(report.get("status") == "evaluated" and report.get("parameter_origin") == "trained_checkpoint"
                and report.get("checkpoint_is_template") is False and report.get("source_ply") is None,
                "Evaluator did not use actual checkpoint parameters")
        require(local(report["config"]) == config_path and local(report["checkpoint"]) == checkpoint,
                "Evaluator config/checkpoint identity differs")
        require(report["step"] == step == report["inference_model_step"], "Evaluator used a synthetic or different step")
        inference_step = report["inference_model_step"]
        require(report.get("config_sha256") == hashes[str(config_path)] and report.get("checkpoint_sha256") == hashes[str(checkpoint)],
                "Evaluator config/checkpoint SHA missing or different")
        require(report["gaussians"] == count, "Evaluator Gaussian count differs")
        require(report.get("export"), "Evaluator export metadata missing")
        ply = pin(export_ply if export_ply else report_path.parent / "gaussians.ply")
        exported = verify_export(ply, parameters)
        require(report["export"]["gaussians"] == exported["gaussians"], "Evaluator export count differs")
        for path_key, sha_key in [("transforms", "transforms_sha256")]:
            require(report.get(path_key) and report.get(sha_key), "Evaluator source dataset proof missing")
            path = pin(report[path_key])
            require(hashes[str(path)] == report[sha_key], "Evaluator dataset SHA differs")
    else:
        reasons.append("evaluator_export_report_missing")
    renderer = verify_renderer_contract(model, launch, result, report, pin)
    disposition_path = run / "experiment_disposition.json"
    disposition = None
    if disposition_path.exists():
        disposition = json.loads(pin(disposition_path).read_text())
        require(isinstance(disposition, dict) and isinstance(disposition.get("status"), str),
                "Malformed experiment disposition")
    disposition_status = disposition["status"] if disposition else "not_assessed"
    diagnostic_only = bool(disposition and (disposition.get("adoptable") is False
                                           or "diagnostic_only" in disposition_status
                                           or "invalid_depth_teacher" in disposition_status))
    require(all(file_state(p) == s for p, s in watched.items()), "Evidence changed during CPU verification; retry after training/evaluation finishes")
    bid = block_id or config.get("experiment_name")
    require(isinstance(bid, str) and bid, "Block ID missing")
    output = {"schema_version": 2, "status": "checkpoint_verified" if not reasons else "incomplete_evidence",
              "block_id": bid, "checkpoint_is_template": False, "parameter_origin": "trained_checkpoint",
              "config": str(config_path), "checkpoint": str(checkpoint), "checkpoint_sha256": hashes[str(checkpoint)],
              "step": step, "completed_iterations": step+1, "total_iterations": target,
              "remaining_iterations": target-step-1, "inference_model_step": inference_step,
              "gaussians": count, "initial_gaussians": initial, "refine_events": refine_events,
              "all_tensors_finite": True, "tensor_count": tensor_count, "optimizer_state_verified": True,
              "optimizer_groups": sorted(state["optimizers"]), "optimizer_moments": moments,
              "parameter_update_l2": None, "parameter_change_evidence": change_evidence,
              "export_verified": exported is not None, "export": exported,
              "verification_scope": "checkpoint_optimizer_and_export_authenticity_only",
              "quality_status": "invalid_depth_teacher" if "invalid_depth_teacher" in disposition_status else disposition_status,
              "quality_acceptance_evaluated": False, "high_fidelity_accepted": False,
              "promotion_allowed": False, "diagnostic_only": diagnostic_only,
              "experiment_disposition": disposition,
              "experiment_disposition_path": str(disposition_path) if disposition else None,
              "adoptable": False if diagnostic_only else None,
              "source_sha256": hashes, "gpu_used": False, "cpu_threads": threads,
              "incomplete_reasons": reasons, "g8_contract": "schema_v2_parameter_change_evidence; no indexwise L2 without IDs",
              "config_launch_consistent": True,
              "inria_backend": renderer["variant"], "renderer_verification": renderer,
              "renderer_provenance": renderer["provenance"],
              "provenance_limit": "Config/launch are pinned at inspection; warm SHA is checked against launch-time initialization SHA."}
    return output


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, required=True)
    p.add_argument("--evaluation-report", type=Path)
    p.add_argument("--export-ply", type=Path)
    p.add_argument("--input-pin", type=Path)
    p.add_argument("--block-id")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = local(args.output)
    require(not out.exists(), "Verification output already exists; choose a new path")
    try:
        report = verify_run(args.training_dir, args.evaluation_report, args.export_ply,
                            args.input_pin, args.block_id, args.threads)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        report = {"schema_version": 2, "status": "verification_failed", "gpu_used": False, "reason": str(exc)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ["status", "quality_status", "diagnostic_only", "promotion_allowed", "step", "refine_events", "export_verified", "incomplete_reasons", "reason"] if k in report}, indent=2))
    return 0 if report["status"] == "checkpoint_verified" else 2


if __name__ == "__main__":
    sys.exit(main())

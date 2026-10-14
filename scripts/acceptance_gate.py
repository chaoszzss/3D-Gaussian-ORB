#!/usr/bin/env python3
"""CPU-only G1--G8 rollout gate; absent evidence never passes.

Input is a manifest of report paths, NOT precomputed PASS flags. Full-scene PLY
rendering and checkpoint training provenance are deliberately separate evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
from build_acceptance_plan import (ROOT, PROTOCOLS, frame_key, load_plan, object_sha,
                                   sha256_file, write_plan)

NATIVE_PROTOCOL = "Exact original INRIA per-Gaussian radius>0; all source records considered"
# Float32 exp(log(.12)) can export .120000004768. This 10 nm arithmetic
# tolerance is not a geometric accuracy allowance and never relaxes p95 .03.
RADIUS_ROUNDOFF_ATOL_M = 1e-8
# Frozen independently of the measurement script's current CLI defaults. A
# diagnostic may change them, but cannot silently redefine acceptance holes.
GEOMETRY_PROTOCOL = {"pixel_stride": 1, "depth_tolerance_m": .15, "reference_match_max_m": .20,
                     "render_pixel_center": .5, "reference_pixel_center": .5,
                     "floor_abs_normal_z_min": .8660254037844387, "wall_abs_normal_z_max": .30,
                     "floor_below_camera_min_m": .30, "floor_below_camera_max_m": 3.,
                     "wall_height_radius_m": 3.}
EXACT_EXPORT_METHOD = "all field bytes exact; native opacity cutoff -5.5373; normalized wxyz; actual normals"
PARAMETER_GROUPS = {"means", "scales", "quats", "features_dc", "features_rest", "opacities"}


class MissingEvidence(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def field(value, key):
    if key not in value or value[key] is None:
        raise MissingEvidence("Missing measurement: " + key)
    return value[key]


def number(value, key, lo=None, hi=None):
    result = field(value, key)
    need(type(result) in (int, float) and math.isfinite(result), "Nonfinite/nonnumeric " + key)
    need((lo is None or result >= lo) and (hi is None or result <= hi), "Out of range " + key)
    return result


def integer(value, key, minimum=0):
    result = field(value, key)
    need(type(result) is int and result >= minimum, "Invalid integer " + key)
    return result


def digest(value):
    need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def read_reports(manifest, project=ROOT):
    """Resolve saved JSON artifacts, expanding the existing scene summary format."""
    project = Path(project).resolve()
    evidence, sources, asset_cache = {}, {}, {}
    def pin_artifact(path, expected):
        path = (project / path).resolve()
        need(project in path.parents, "Evidence asset escapes project")
        if not path.is_file():
            raise MissingEvidence("Missing evidence asset: " + str(path))
        if str(path) not in sources:
            sources[str(path)] = sha256_file(path)
        need(sources[str(path)] == digest(expected), "Evidence asset digest differs: " + str(path))
    def read(path):
        path = (project / path).resolve()
        need(project in path.parents, "Evidence escapes project")
        if not path.is_file():
            raise MissingEvidence("Missing report: " + str(path))
        sources[str(path)] = sha256_file(path)
        report = json.loads(path.read_text())
        report["_evidence_path"] = str(path)
        report["_evidence_sha256"] = sources[str(path)]
        source = report.get("source_ply") or {}
        if source.get("path"):
            pin_artifact(source["path"], field(source, "sha256"))
        # Pin representation and renderer dependencies, not merely the JSON
        # which asserts that those assets passed once in the past.
        if "backend" in report:
            validate_backend(report, asset_cache)
        if report.get("direction_support") is not None:
            validate_representation(report, asset_cache)
        for name in ("config", "checkpoint", "transforms"):
            if report.get("status") == "evaluated" and report.get(name):
                pin_artifact(report[name], field(report, name + "_sha256"))
        if report.get("status") == "checkpoint_verified":
            pin_artifact(field(report, "checkpoint"), field(report, "checkpoint_sha256"))
            exported = field(report, "export")
            pin_artifact(field(exported, "path"), field(exported, "sha256"))
        for source_path, sha in report.get("source_sha256", {}).items():
            pin_artifact(source_path, sha)
        for source in report.get("render_measurement", {}).get("sources", {}).values():
            pin_artifact(field(source, "path"), field(source, "sha256"))
        for origin in report.get("render_measurement", {}).get("source_evaluator_provenance", {}).values():
            binding = origin.get("checkpoint_export_proof")
            if binding:
                pin_artifact(field(binding, "path"), field(binding, "sha256"))
                proof = json.loads((project / binding["path"]).resolve().read_text())
                for source_path, sha in field(proof, "source_sha256").items():
                    pin_artifact(source_path, sha)
        return report
    for role in ("baseline", "repeat", "candidate", "single_block", "surface", "training", "pixel_closure"):
        values = manifest.get(role, [])
        if isinstance(values, str):
            values = [values]
        need(isinstance(values, list), "Evidence role requires report path(s): " + role)
        evidence[role] = []
        for value in values:
            try:
                report = read(value)
                if role in ("baseline", "repeat", "candidate", "single_block") and "blocks" in report:
                    for item in report["blocks"]:
                        child = read(item["path"])
                        need(sources[child["_evidence_path"]] == item["sha256"], "Scene child report digest differs")
                        evidence[role].append(child)
                else:
                    evidence[role].append(report)
            except MissingEvidence as exc:
                evidence[role].append({"_missing": str(exc)})
    sources.update(asset_cache.get("_pins", {}))
    return evidence, sources


def validate_render_reports(plan, reports, expected_blocks=None, whole_model=True, asset_cache=None):
    if not reports:
        raise MissingEvidence("Missing renderer reports")
    expected = {frame_key(f): f for f in plan["frames"]
                if expected_blocks is None or f["block_id"] in expected_blocks}
    all_frames = {frame_key(f): f for f in plan["frames"]}
    asset_cache = {} if asset_cache is None else asset_cache
    rows, models, base_models, bindings, run_paths = {}, {}, {}, {}, set()
    for report in reports:
        if report.get("_missing"):
            raise MissingEvidence(report["_missing"])
        need(report.get("status") == "evaluated" and report.get("split") == "dev", "Renderer incomplete or wrong split")
        need(report.get("acceptance_plan_id") == plan["plan_id"], "Wrong frozen plan identity")
        need(report.get("metric_domain") == "float_renderer", "Require actual float renderer; PNG/unknown is diagnostic only")
        need(report.get("primary_protocol", "ego_v2") == "ego_v2", "Wrong primary scoring protocol")
        # A caller may not strip/change a saved sidecar or renderer assertion
        # after read_reports has established its on-disk identity.
        if report.get("_evidence_sha256"):
            saved = verify_artifact(field(report, "_evidence_path"), report["_evidence_sha256"], asset_cache)
            need(json.loads(saved.read_text()) == {k: v for k, v in report.items() if not k.startswith("_")},
                 "Evaluator report differs from pinned saved evidence")
        binding = validate_evaluator_assets(report, asset_cache)
        representation = validate_representation(report, asset_cache)
        need(report.get("world_parser_scale") == 1.0 and report.get("world_parser_transform") in (
            [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            "Renderer world transform/scale changed")
        source = report.get("source_ply")
        if source:
            model_sha = digest(field(report, "source_ply_sha256"))
            need(source.get("sha256") == model_sha, "Source PLY SHA mismatch")
            count = integer(source, "gaussians", 1)
            need(report.get("gaussians") == count, "Partial Gaussian model")
        else:
            need(not whole_model, "Whole-scene report missing source PLY proof")
            need(report.get("checkpoint_is_template") is not True, "Template checkpoint is not a trained checkpoint evaluation")
            model_sha = digest(field(report, "checkpoint_sha256"))
        block = report.get("block_id", report.get("acceptance", {}).get("block_id"))
        declared = field(report, "frames")
        need(integer(report, "images", 1) == len(declared), "Report frame count mismatch")
        if report.get("_evidence_path"):
            run_paths.add(report["_evidence_path"])
        for row in declared:
            key = frame_key(row)
            need(key in all_frames and key not in rows, "Unexpected/duplicate evaluated frame")
            frozen = all_frames[key]
            need(number(row, "render_seconds", 0) > 0, "Missing actual renderer timing")
            need(block == frozen["block_id"], "Evaluated under noncanonical source block")
            need(row.get("dataset_index") == frozen["dataset_index"] and row.get("file_path") == frozen["file_path"],
                 "Wrong source val order or filename")
            need(report.get("transforms_sha256") == frozen["transforms_sha256"], "Wrong source transforms SHA")
            for asset in ("source_image_path", "source_mask_path", "source_lidar_depth_path", "source_lidar_support_path"):
                need(row.get(asset) == frozen[asset], "Wrong fixed source asset: " + asset)
            mask = row.get("acceptance_mask_path", row.get("mask_path"))
            need(mask == frozen["mask_path"], "Wrong fixed ego mask")
            camera = field(row, "camera")
            for name, value in frozen["camera"].items():
                actual = field(camera, name)
                flat = lambda v: [x for line in v for x in line] if isinstance(v, list) else [v]
                a, b = flat(actual), flat(value)
                need(len(a) == len(b) and all(type(x) in (int, float) and math.isfinite(x)
                     and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-5) for x, y in zip(a, b)), "Changed source camera")
            protocols = field(row, "protocols")
            need(set(protocols) == set(PROTOCOLS), "Missing/extra scoring protocol")
            for name, score in protocols.items():
                pixels = integer(score, "rgb_valid_pixels", 1)
                mse = number(score, "rgb_mse", 0, 1)
                psnr = number(score, "psnr_db", 0, 120)
                need(abs(psnr + 10 * math.log10(max(mse, 1e-12))) < 1e-4, "PSNR does not match MSE")
                if name == "ego_v2":
                    need(pixels == plan["mask_valid_pixels"][frozen["mask_path"]], "Ego pixel denominator changed")
                elif name == "raw":
                    need(pixels == frozen["camera"]["w"] * frozen["camera"]["h"], "Raw pixel denominator changed")
                number(score, "ssim_valid_pixels", -1, 1)
                need(integer(score, "ssim_compared_pixels", 1) <= pixels, "Invalid SSIM denominator")
                eligible = integer(score, "map_depth_eligible_pixels", 1)
                compared = integer(score, "map_depth_compared_pixels")
                need(compared <= eligible and integer(score, "map_depth_missing_pixels") == eligible - compared,
                     "Invalid fixed depth denominator/missing count")
                number(score, "map_depth_capped_relative_error_all_supported", 0, 1)
            if whole_model:
                visibility = field(row, "native_visibility")
                need(visibility.get("protocol") == NATIVE_PROTOCOL and visibility.get("source_records") == count,
                     "Missing complete source PLY visibility proof")
                need(integer(visibility, "active_records") <= count, "Active count exceeds source")
                digest(field(visibility, "active_index_sha256"))
            visibility = row.get("native_visibility") or {}
            directional = visibility.get("direction_visibility")
            if report.get("direction_support") is not None:
                need(visibility.get("protocol") == NATIVE_PROTOCOL and visibility.get("source_records") == count,
                     "Direction support requires complete native visibility")
                active = integer(visibility, "active_records")
                need(active <= count, "Direction native count exceeds complete PLY")
                digest(field(visibility, "active_index_sha256"))
                directional = field(visibility, "direction_visibility")
                need(integer(directional, "native_active_records") == active and
                     integer(directional, "retained_records") + integer(directional, "side_rejected_records") == active,
                     "Direction retained/rejected counts disagree with native visibility")
                digest(field(directional, "retained_index_sha256"))
            else:
                need(directional is None and row.get("direction_visibility") is None,
                     "Direction visibility has no bound sidecar")
            rows[key], models[key], base_models[key], bindings[key] = row, representation, model_sha, binding
    # A five-block pilot or the complete frozen dataset is allowed, never a
    # hand-picked subset of the five pilot blocks.
    need(set(rows) in (set(expected), set(all_frames)), "Missing/unexpected DEV frame set")
    if whole_model:
        need(len(set(models.values())) == 1, "Whole-scene blocks used different complete representations")
        need(len({v["renderer_id"] for v in bindings.values()}) == 1, "Whole-scene blocks mixed renderer backends")
    return {"rows": {k: rows[k] for k in expected}, "models": {k: models[k] for k in expected},
            "base_models": {k: base_models[k] for k in expected},
            "bindings": {k: bindings[k] for k in expected}, "run_paths": run_paths}


def paired(a, b):
    need(set(a["rows"]) == set(b["rows"]), "Paired frame sets differ")
    for key in a["rows"]:
        for protocol in PROTOCOLS:
            x, y = (r["rows"][key]["protocols"][protocol] for r in (a, b))
            for field_name in ("rgb_valid_pixels", "ssim_compared_pixels", "map_depth_eligible_pixels"):
                need(field(x, field_name) == field(y, field_name), "Paired fixed denominator changed: " + field_name)


def verify_artifact(path, expected, cache=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise MissingEvidence("Missing checkpoint proof asset: " + str(path))
    before = path.stat()
    key = (str(path.resolve()), digest(expected), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if cache is None or key not in cache:
        need(sha256_file(path) == expected, "Checkpoint proof asset changed: " + str(path))
        after = path.stat()
        need((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Checkpoint proof asset changed during read")
        if cache is not None:
            cache[key] = True
    if cache is not None:
        pins = cache.setdefault("_pins", {})
        need(pins.get(str(path.resolve()), expected) == expected, "One audit used conflicting asset versions")
        pins[str(path.resolve())] = expected
    return path.resolve()


def validate_backend(report, cache=None):
    """Use the geometry auditor's actual AA binary/source/FD/integration chain."""
    from measure_surface_quality import validate_renderer_provenance
    # Integration evidence may change independently of the renderer binary.
    # Include it in validation caching so a second report cannot bypass its
    # declared proof by sharing the same backend provenance with a first one.
    key = ("renderer", object_sha({k: report.get(k) for k in
                                  ("backend", "renderer_provenance", "renderer_integration_proof")}))
    verified = cache.get(key) if cache is not None else None
    if verified is None:
        verified = validate_renderer_provenance(report)
        if cache is not None:
            cache[key] = verified
    for path, sha in verified["assets"].items():
        verify_artifact(path, sha, cache)
    # Legacy original reports predate the redundant provenance object. Their
    # supported renderer contract remains the same original profile; adding
    # provenance to a later report must not invent a different render method.
    if verified.get("backend") == "hbb1-original-2dgs":
        return object_sha({"backend": "hbb1-original-2dgs"})
    return object_sha({k: report.get(k) for k in ("backend", "renderer_provenance")})


def checkpoint_metadata(path):
    """Read stored step/shapes on CPU without materializing optimizer arrays.

    Restricted weights loading allows only the NumPy scalar encoding used by
    historical Nerfstudio checkpoints; it never falls back to arbitrary pickle.
    This is an evaluation identity check, not G8's training authenticity proof.
    """
    import codecs
    import pickle
    import numpy as np
    import torch
    old_globals = torch.serialization.get_safe_globals()
    try:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype,
                                               type(np.dtype("float64")), codecs.encode])
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ValueError("Cannot read actual checkpoint metadata: " + str(exc)) from exc
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(old_globals)
    step = integer(state, "step")
    if re.fullmatch(r"step-\d+", path.stem):
        need(int(path.stem[5:]) == step, "Checkpoint filename/stored step differs")
    parameters = {}
    for name, value in field(state, "pipeline").items():
        match = re.fullmatch(r"(?:module\.)?_model\.(?:module\.)?gauss_params\.(\w+)", name)
        if match:
            need(match[1] not in parameters, "Duplicate checkpoint Gaussian group")
            parameters[match[1]] = value
    need(set(parameters) == PARAMETER_GROUPS, "Incomplete checkpoint Gaussian groups")
    count = len(parameters["means"])
    need(count > 0, "Empty checkpoint")
    for name, width in (("means", 3), ("scales", 3), ("quats", 4), ("features_dc", 3), ("opacities", 1)):
        need(tuple(parameters[name].shape) == (count, width), "Checkpoint Gaussian shape differs: " + name)
    need(all(v.device.type == "cpu" and v.dtype == torch.float32 for v in parameters.values()),
         "Require CPU float32 checkpoint parameters")
    return {"step": step, "gaussians": count, "features_rest_shape": tuple(parameters["features_rest"].shape)}


def validate_evaluator_assets(report, cache=None):
    """Bind evaluator backend and inference step to actual saved inputs."""
    from verify_training_checkpoint import read_config
    cache = {} if cache is None else cache
    assets = {name: verify_artifact(field(report, name), field(report, name + "_sha256"), cache)
              for name in ("config", "checkpoint", "transforms")}
    config_key = ("config", str(assets["config"]), report["config_sha256"])
    if config_key not in cache:
        cache[config_key] = read_config(assets["config"])
        verify_artifact(assets["config"], report["config_sha256"], cache)
    model = field(field(cache[config_key], "pipeline"), "model")
    variant = model.get("inria_backend", "original")  # fixed historical default
    need(variant in ("original", "aa-depth-vjp") and model.get("use_inria_surfel_renderer") is True,
         "Config does not enable supported native renderer")
    expected_backend = {"original": "hbb1-original-2dgs", "aa-depth-vjp": "hbb1-aa-depth-vjp-2dgs"}[variant]
    need(report.get("backend") == expected_backend, "Renderer backend differs from actual config")
    renderer_id = validate_backend(report, cache)
    ckpt_key = ("checkpoint", str(assets["checkpoint"]), report["checkpoint_sha256"])
    if ckpt_key not in cache:
        cache[ckpt_key] = checkpoint_metadata(assets["checkpoint"])
        verify_artifact(assets["checkpoint"], report["checkpoint_sha256"], cache)
    actual = cache[ckpt_key]
    need(integer(report, "step") == integer(report, "inference_model_step") == actual["step"],
         "Evaluator inference step differs from actual checkpoint")
    degree, interval = integer(model, "sh_degree"), integer(model, "sh_degree_interval", 1)
    need(actual["features_rest_shape"] == (actual["gaussians"], (degree + 1)**2 - 1, 3), "Config/checkpoint SH shape differs")
    need(integer(report, "inference_sh_degree") == min(actual["step"] // interval, degree),
         "Evaluator SH inference degree differs from actual config/checkpoint")
    if report.get("source_ply"):
        need(report.get("checkpoint_is_template") is True, "PLY evaluator must declare template checkpoint")
        need(integer(report, "original_checkpoint_gaussians", 1) == actual["gaussians"], "Template checkpoint count differs")
        if "sh_degree" in report["source_ply"]:
            need(report["source_ply"]["sh_degree"] == degree, "Source PLY/config SH degree differs")
    else:
        need(report.get("source_ply_sha256") is None and report.get("checkpoint_is_template") is False
             and report.get("parameter_origin") == "trained_checkpoint", "Missing native checkpoint parameter provenance")
        need(integer(report, "gaussians", 1) == actual["gaussians"], "Native checkpoint evaluation is partial")
    return {"renderer_id": renderer_id, **{name + "_sha256": report[name + "_sha256"] for name in assets},
            "step": actual["step"], "inference_sh_degree": report["inference_sh_degree"]}


def validate_representation(report, cache=None):
    """Complete PLY + all TRAIN side assets form one immutable representation.

    Plain historical reports need no redundant representation_id. An explicit
    identity is always checked, including plain PLY and native checkpoint cases.
    """
    from package_garage import inspect_ply
    cache = {} if cache is None else cache
    source = report.get("source_ply")
    side = report.get("direction_support")
    if source:
        base = digest(field(report, "source_ply_sha256"))
        need(field(source, "sha256") == base, "Source PLY SHA mismatch")
        path = verify_artifact(field(source, "path"), base, cache)
        key = ("ply_header", str(path), base)
        if key not in cache:
            with path.open("rb") as handle:
                cache[key] = inspect_ply(handle)
        spec = cache[key]
        count = integer(source, "gaussians", 1)
        need(spec["count"] == count == integer(report, "gaussians", 1)
             and path.stat().st_size == len(spec["header"]) + count * spec["stride"], "Incomplete source PLY records/payload")
        if "sh_degree" in source:
            need(source["sh_degree"] == spec["sh_degree"], "Source PLY/header SH degree differs")
    else:
        base = digest(field(report, "checkpoint_sha256"))
        need(side is None, "Direction support requires a complete source PLY")
    expected = base
    if side is not None:
        from direction_support import load_direction_support
        side_path = verify_artifact(field(side, "path"), field(side, "sha256"), cache)
        key = ("direction", str(side_path), side["sha256"], base)
        if key not in cache:
            # The loader only needs lengths of these two arrays. range supplies
            # the verified full PLY count without allocating all model tensors.
            support = load_direction_support(side_path, {**source, "path": str(path)},
                                             {"means": range(count), "quats": range(count)})
            cache[key] = {k: support[k] for k in ("info", "asset_digests")}
        support = cache[key]
        need(side == support["info"], "Direction-support report differs from validated TRAIN assets")
        for asset, sha in support["asset_digests"].items():
            verify_artifact(asset, sha, cache)
        expected = hashlib.sha256((base + ":" + side["sha256"]).encode()).hexdigest()
        need(report.get("representation_id") == expected, "Sidecar complete representation identity differs")
    else:
        need("TRAIN observation-side support" not in str(report.get("model_representation", "")),
             "Side representation lacks its sidecar")
    need(report.get("representation_id", expected) == expected, "Explicit representation identity differs")
    return expected


def verify_checkpoint_export_origin(origin, report, role, plan_id, cache=None):
    """Strict native checkpoint -> exact filtered PLY proof; count-only is invalid.

This checks the CPU inspector's recorded byte equality and every pinned input,
and reads native checkpoint step/shapes with restricted CPU mmap loading. It
does not pretend that a filtered PLY was rendered or re-prove all export bytes.
"""
    binding = field(origin, "checkpoint_export_proof")
    proof_path = verify_artifact(field(binding, "path"), field(binding, "sha256"), cache)
    proof = json.loads(proof_path.read_text())
    need(proof.get("schema_version") == 2 and proof.get("status") == "checkpoint_verified"
         and proof.get("checkpoint_is_template") is False and proof.get("parameter_origin") == "trained_checkpoint"
         and proof.get("all_tensors_finite") is True and proof.get("export_verified") is True,
         "Checkpoint export proof is incomplete/template/unverified")
    need(proof.get("block_id") == report["block_id"], "Checkpoint export proof block differs")
    exported = field(proof, "export")
    need(exported.get("verification") == EXACT_EXPORT_METHOD, "Export proof is count-only, not exact native field bytes")
    # Measurement reports use nested artifact identities; retain the initially
    # proposed flat shape for explicit callers, with identical checks.
    bound_export = field(binding["export"], "sha256") if "export" in binding else field(binding, "export_sha256")
    bound_checkpoint = field(binding["checkpoint"], "sha256") if "checkpoint" in binding else field(binding, "checkpoint_sha256")
    bound_metrics = binding["metrics"] if "metrics" in binding else {
        "path": field(binding, "metrics_path"), "sha256": field(binding, "metrics_sha256")}
    need(digest(field(exported, "sha256")) == report[role + "_model_sha256"] == bound_export,
         "Geometry exported PLY identity differs")
    need(digest(field(proof, "checkpoint_sha256")) == bound_checkpoint, "Proof checkpoint identity differs")
    representation = report.get(role + "_representation_id")
    need(representation == proof["checkpoint_sha256"], "Native checkpoint representation must be checkpoint SHA")
    sources = field(proof, "source_sha256")
    need(isinstance(sources, dict) and sources, "Proof has no pinned source assets")
    for path, sha in sources.items():
        verify_artifact(path, sha, cache)
    metrics_path = verify_artifact(field(bound_metrics, "path"), field(bound_metrics, "sha256"), cache)
    need(sources.get(str(metrics_path)) == bound_metrics["sha256"], "Proof does not bind actual evaluator metrics")
    metrics = json.loads(metrics_path.read_text())
    need(metrics.get("status") == "evaluated" and metrics.get("metric_domain") == "float_renderer"
         and metrics.get("acceptance_plan_id") == plan_id and metrics.get("parameter_origin") == "trained_checkpoint"
         and metrics.get("checkpoint_is_template") is False and metrics.get("source_ply") is None
         and metrics.get("source_ply_sha256") is None, "Geometry was not rendered from the claimed native checkpoint")
    validate_evaluator_assets(metrics, cache)
    need(validate_representation(metrics, cache) == representation, "Native checkpoint representation differs")
    step = integer(proof, "step")
    need(step == metrics.get("step") == metrics.get("inference_model_step") == proof.get("inference_model_step"),
         "Checkpoint proof/renderer step differs")
    need(integer(proof, "completed_iterations", 1) == step + 1, "Checkpoint proof iteration count differs")
    native_count = integer(proof, "gaussians", 1)
    export_count = integer(exported, "gaussians", 1)
    need(native_count == metrics.get("gaussians") and export_count <= native_count
         and export_count == field(metrics, "export").get("gaussians"), "Checkpoint/export counts differ from verified filtering")
    for field_name in ("config", "checkpoint", "transforms"):
        path = str(Path(field(metrics, field_name)).resolve())
        sha = field(metrics, field_name + "_sha256")
        need(sources.get(path) == digest(sha), "Proof lacks fixed evaluator asset: " + field_name)
        if field_name in ("config", "checkpoint"):
            need(str(Path(field(proof, field_name)).resolve()) == path, "Proof/evaluator path differs: " + field_name)
        if field_name in binding:
            need(str(Path(field(binding[field_name], "path")).resolve()) == path and
                 field(binding[field_name], "sha256") == sha, "Nested binding identity differs: " + field_name)
    need(metrics["checkpoint_sha256"] == proof["checkpoint_sha256"], "Native checkpoint SHA differs")
    need(sources.get(str(Path(exported["path"]).resolve())) == exported["sha256"], "Proof does not pin exact exported PLY")
    if "export" in binding:
        need(binding["export"] == exported, "Nested binding export proof differs")
    return proof


def validate_geometry_protocol(report, plan_id, asset_cache=None):
    """Do not accept a larger tolerance or sparser pixels as improved geometry."""
    measured = field(report, "render_measurement")
    need(field(measured, "gate_compatible_settings") is True, "Geometry settings are diagnostic only")
    need(field(measured, "gate_compatible_render_provenance") is True, "Geometry renderer provenance is diagnostic only")
    need(measured.get("acceptance_plan_id") == plan_id, "Geometry measurement plan differs")
    protocol = field(measured, "protocol")
    for name, expected in GEOMETRY_PROTOCOL.items():
        actual = number(protocol, name)
        need(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), "Changed geometry measurement setting: " + name)
    need(protocol.get("mask_protocol") == "ego_v2", "Geometry must use frozen ego mask")
    need(integer(field(report, "scope"), "pixel_stride", 1) == 1, "Geometry skipped fixed support pixels")
    digest(field(measured, "fixed_denominator_id"))
    reference_sha = digest(field(protocol, "reference_sha256"))
    provenance = field(measured, "source_evaluator_provenance")
    for role in ("candidate", "baseline"):
        origin = field(provenance, role)
        need(origin.get("metric_domain") == "float_renderer" and origin.get("acceptance_plan_id") == plan_id,
             "Missing float renderer/plan provenance for " + role)
        validate_geometry_origin(report, role, asset_cache)
        if origin.get("source_ply_sha256") is None:
            verify_checkpoint_export_origin(origin, report, role, plan_id, asset_cache)
        else:
            need(origin["source_ply_sha256"] == report[role + "_model_sha256"], "Geometry provenance model differs")
    return reference_sha, measured["fixed_denominator_id"]


def validate_geometry_origin(report, role, cache=None):
    """Bind surface summaries to the same actual representation and evaluator."""
    measured = field(report, "render_measurement")
    origin = field(field(measured, "source_evaluator_provenance"), role)
    source = field(field(measured, "sources"), role)
    path = verify_artifact(field(source, "path"), field(source, "sha256"), cache)
    metrics = json.loads(path.read_text())
    for key in ("metric_domain", "acceptance_plan_id", "backend", "renderer_provenance", "source_ply_sha256"):
        need(origin.get(key) == metrics.get(key), "Geometry evaluator provenance differs: " + key)
    need(metrics.get("status") == "evaluated" and metrics.get("metric_domain") == "float_renderer"
         and metrics.get("acceptance_plan_id") == report.get("acceptance_plan_id", measured.get("acceptance_plan_id")),
         "Geometry source is not the fixed float evaluation")
    for key in ("checkpoint_sha256", "parameter_origin", "inference_model_step"):
        if key in origin:
            need(origin[key] == metrics.get(key), "Geometry evaluator provenance differs: " + key)
    binding = validate_evaluator_assets(metrics, cache)
    identity = validate_representation(metrics, cache)
    declared = report.get(role + "_representation_id", metrics.get("source_ply_sha256") or metrics.get("checkpoint_sha256"))
    need(declared == identity, "Surface complete representation binding differs")
    representation = measured.get("representations", {}).get(role)
    if representation is not None:
        need(representation.get("representation_id") == identity, "Surface representation descriptor differs")
        if metrics.get("direction_support") is not None:
            need(representation.get("direction_support") == metrics["direction_support"], "Surface sidecar descriptor differs")
    return identity, binding


def validate_parameter_change(proof):
    need(proof.get("optimizer_state_verified") is True, "Optimizer state not verified")
    if proof.get("schema_version") != 2:
        need(number(proof, "parameter_update_l2", 0) > 0, "No actual parameter update evidence")
        return
    change = field(proof, "parameter_change_evidence")
    need(change.get("schema_version") == 1 and change.get("method") == "optimizer_moments_and_parameter_hash_change"
         and change.get("verified") is True and change.get("parameter_hash_changed") is True,
         "Missing actual optimizer/hash parameter change evidence")
    initial, final = (digest(field(change, k)) for k in ("initial_parameter_sha256", "final_parameter_sha256"))
    need(initial != final, "Initial/final parameter hashes are identical")
    need(integer(change, "initial_gaussians", 1) == integer(proof, "initial_gaussians", 1)
         and integer(change, "final_gaussians", 1) == integer(proof, "gaussians", 1), "Parameter-change counts differ")
    need(set(field(change, "optimizer_nonzero_groups")) == PARAMETER_GROUPS, "Missing optimizer parameter groups")
    moments = field(proof, "optimizer_moments")
    for group in PARAMETER_GROUPS:
        state = field(moments, group)
        integer(state, "exp_avg_nonzero", 1); integer(state, "exp_avg_sq_nonzero", 1)
        need(number(state, "exp_avg_max_abs", 0) > 0 and number(state, "exp_avg_sq_max_abs", 0) > 0,
             "Zero optimizer moment group: " + group)
        need(number(state, "adam_step", 0) == proof["completed_iterations"], "Optimizer step differs from actual training")
    need(proof.get("config_launch_consistent") is True and proof.get("source_sha256"), "Missing config/launch provenance")
    need(proof.get("diagnostic_only") is not True and proof.get("adoptable") is not False
         and proof.get("quality_status") != "invalid_depth_teacher", "Diagnostic/invalid teacher run is not adoptable")


def evaluate_gate(plan, evidence):
    """Return G1--G8 without treating absent/NaN/partial evidence as a pass."""
    thresholds = plan["gate_thresholds"]
    blocks = thresholds["G2"]["f1_blocks"]
    frozen = {frame_key(f): f for f in plan["frames"]}
    cache, results, asset_cache = {}, {}, {}
    def render(role, full=False):
        key = (role, full)
        if key not in cache:
            cache[key] = validate_render_reports(plan, evidence.get(role, []), None if full else blocks,
                                                 whole_model=role != "single_block", asset_cache=asset_cache)
        return cache[key]
    def surfaces():
        reports = evidence.get("surface", [])
        if not reports:
            raise MissingEvidence("Missing measured surface/geometry evidence")
        by_block = {}
        candidate, baseline = render("candidate"), render("baseline")
        for report in reports:
            bid = field(report, "block_id")
            need(bid in blocks and bid not in by_block, "Unexpected/duplicate surface block")
            need(report.get("acceptance_plan_id") == plan["plan_id"], "Surface report uses wrong plan")
            need(report.get("metric_domain") == "conditioned_geometry", "Unknown geometry protocol")
            keys = [frame_key(f) for f in field(field(report, "scope"), "frame_keys")]
            expected = {k for k, f in frozen.items() if f["block_id"] == bid}
            need(len(keys) == len(set(keys)) and set(keys) == expected, "Surface scope incomplete/duplicated")
            for role, rendered in (("candidate", candidate), ("baseline", baseline)):
                need(all(report.get(role + "_model_sha256") == rendered["base_models"][k] for k in expected),
                     "Surface model binding differs")
                identity, binding = validate_geometry_origin(report, role, asset_cache)
                need(all(identity == rendered["models"][k] and binding == rendered["bindings"][k] for k in expected),
                     "Surface evaluation binding differs from paired report")
            by_block[bid] = report
        if set(by_block) != set(blocks):
            raise MissingEvidence("Missing surface evidence for all F1 pilot blocks")
        return by_block
    def measured(report, gate):
        value = field(field(report, "gate_evidence"), gate)
        if value.get("status") in ("unknown", "missing", "unmeasured", "blocked"):
            raise MissingEvidence(gate + " measurement unknown")
        need(value.get("status") in ("known", "measured"), gate + " lacks measurement status")
        return value
    def g1():
        baseline, repeat = render("baseline", True), render("repeat", True)
        paired(baseline, repeat)
        need(baseline["models"] == repeat["models"], "Repeat used a different model")
        need(baseline["bindings"] == repeat["bindings"], "Repeat used different renderer/config/checkpoint/inference inputs")
        need(baseline["run_paths"] and repeat["run_paths"] and not baseline["run_paths"] & repeat["run_paths"],
             "Repeat must come from distinct actual renderer output files")
        difference = max(abs(baseline["rows"][k]["protocols"][p]["psnr_db"] - repeat["rows"][k]["protocols"][p]["psnr_db"])
                         for k in baseline["rows"] for p in PROTOCOLS)
        need(difference < thresholds["G1"]["max_repeat_psnr_difference_db"], "Renderer repeat difference exceeds strict tolerance")
        closure = evidence.get("pixel_closure", [])
        if not closure:
            raise MissingEvidence("Missing pixel-convention closure test result")
        need(len(closure) == 1, "Ambiguous closure evidence")
        proof = closure[0]
        need(integer(proof, "wrong_index_count") == 0 and number(proof, "foreground_array_equal_fraction", 0, 1) == 1,
             "Pixel-convention closure failed")
        need(proof.get("status") == "passed", "Pixel closure not passed")
        integer(proof, "tested_index_count", 1)
        need(proof.get("pytest_returncode") == 0 and proof.get("commands") and proof.get("source_sha256"),
             "Pixel closure lacks reproducible source/test evidence")
        return {"frames": len(baseline["rows"]), "max_repeat_psnr_difference_db": difference}
    def g2():
        candidate, baseline = render("candidate"), render("baseline")
        paired(candidate, baseline)
        means, gains = {}, {}
        for bid in blocks:
            keys = [k for k in candidate["rows"] if frozen[k]["block_id"] == bid]
            means[bid] = statistics.mean(candidate["rows"][k]["protocols"]["ego_v2"]["psnr_db"] for k in keys)
            gains[bid] = statistics.mean(candidate["rows"][k]["protocols"]["ego_v2"]["psnr_db"] -
                                        baseline["rows"][k]["protocols"]["ego_v2"]["psnr_db"] for k in keys)
        need(sum(g >= thresholds["G2"]["min_paired_gain_db"] for g in gains.values()) >= thresholds["G2"]["min_improved_blocks"],
             "Fewer than four F1 blocks improve by 2 dB")
        need(statistics.median(means.values()) >= thresholds["G2"]["min_block_psnr_median_db"], "F1 median PSNR below threshold")
        return {"block_psnr_db": means, "paired_gain_db": gains}
    def g3():
        for report in surfaces().values():
            v = measured(report, "G3")
            need(number(v, "opacity_median", 0, 1) >= thresholds["G3"]["opacity_median_min"], "Opacity median too low")
            need(number(v, "opacity_frac_lt_0_5", 0, 1) <= thresholds["G3"]["opacity_below_half_fraction_max"], "Too many transparent Gaussians")
            need(number(v, "voxel_size_m", 0) == thresholds["G3"]["voxel_size_m"], "Wrong voxel size")
            need(number(v, "voxel_2cm_multiplicity_median", 0) >= thresholds["G3"]["voxel_multiplicity_median_min"], "Voxel multiplicity too low")
        return {"blocks": blocks}
    def g4():
        for report in surfaces().values():
            v = measured(report, "G4")
            reference_sha, _ = validate_geometry_protocol(report, plan["plan_id"], asset_cache)
            need(digest(field(v, "reference_sha256")) == reference_sha, "Geometry reference SHA differs")
            need(v.get("shared_reference") is True and v.get("independent_ground_truth") is False,
                 "Require shared conditioned reference; cannot claim independent truth")
            integer(v, "candidate_points", 1); integer(v, "baseline_points", 1)
            baseline = number(v, "baseline_p50_m", 0)
            candidate = number(v, "candidate_p50_m", 0)
            need(baseline > 0, "Zero baseline does not define a reduction ratio")
            need(candidate <= baseline * thresholds["G4"]["chamfer_p50_ratio_max"] and
                 candidate <= thresholds["G4"]["chamfer_p50_m_max"], "Chamfer target not met")
        return {"blocks": blocks, "independent_ground_truth": False}
    def g5():
        for report in surfaces().values():
            v = measured(report, "G5")
            reference_sha, denominator = validate_geometry_protocol(report, plan["plan_id"], asset_cache)
            need(digest(field(v, "reference_sha256")) == reference_sha and
                 field(v, "fixed_denominator_id") == denominator, "Hole measurement reference/denominator differs")
            need(v.get("fixed_denominator") is True, "Hole rates do not use fixed reference denominator")
            for category in ("floor", "wall"):
                rates = field(v, category)
                eligible = integer(rates, "eligible_pixels", 1)
                baseline = integer(rates, "baseline_hole_pixels", 1)
                candidate = integer(rates, "candidate_hole_pixels")
                need(max(baseline, candidate) <= eligible, "Hole count exceeds fixed denominator")
                need(candidate <= baseline * thresholds["G5"]["hole_rate_ratio_max"], category + " holes not reduced 30%")
        return {"blocks": blocks}
    def g6():
        candidate, single = render("candidate"), render("single_block")
        paired(candidate, single)
        losses = {}
        for bid in blocks:
            keys = [k for k in candidate["rows"] if frozen[k]["block_id"] == bid]
            losses[bid] = statistics.mean(single["rows"][k]["protocols"]["ego_v2"]["psnr_db"] -
                                         candidate["rows"][k]["protocols"]["ego_v2"]["psnr_db"] for k in keys)
        need(max(losses.values()) <= thresholds["G6"]["max_block_loss_db"], "At least one block loses more than 1 dB on assembly")
        return {"block_loss_db": losses}
    def g7():
        for report in surfaces().values():
            v = measured(report, "G7")
            need(number(v, "tangential_radius_p95_m", 0) <= thresholds["G7"]["tangent_radius_p95_m_max"] + RADIUS_ROUNDOFF_ATOL_M,
                 "Tangential radius p95 too large")
            need(number(v, "tangential_radius_max_m", 0) <= thresholds["G7"]["tangent_radius_max_m_max"] + RADIUS_ROUNDOFF_ATOL_M,
                 "Tangential radius max too large")
            need(number(v, "seam_gap_db", 0) <= thresholds["G7"]["seam_gap_db_max"], "Seam gap too large")
            need(integer(v, "seam_band_width_px", 1) == 64 and integer(v, "seam_compared_pixels", 1) > 0,
                 "Missing fixed 64px seam measurement")
        return {"blocks": blocks, "radius_roundoff_atol_m": RADIUS_ROUNDOFF_ATOL_M}
    def g8():
        reports = evidence.get("training", [])
        if not reports:
            raise MissingEvidence("Missing checkpoint training/export proof")
        single = render("single_block")
        selected = {}
        for v in reports:
            bid = field(v, "block_id")
            need(bid in blocks and bid not in selected, "Unexpected/duplicate trained block")
            selected[bid] = v
            need(v.get("status") == "checkpoint_verified" and v.get("all_tensors_finite") is True,
                 "Checkpoint tensors not verified finite")
            need(v.get("checkpoint_is_template") is False, "Training proof references template checkpoint")
            count = integer(v, "completed_iterations", 1)
            need(count >= thresholds["G8"]["min_completed_iterations"], "Insufficient actually completed iterations")
            need(integer(v, "step") + 1 == count == integer(v, "total_iterations", 1), "Zero-based checkpoint step/iteration mismatch")
            need(integer(v, "remaining_iterations") == 0, "Incomplete training")
            need(integer(v, "inference_model_step") == v["step"], "Checkpoint inference step is synthetic")
            need(integer(v, "refine_events") >= thresholds["G8"]["min_refine_events"], "No observed refine events")
            validate_parameter_change(v)
            need(v.get("export_verified") is True, "PLY export not linked to checkpoint tensors")
            exported = field(v, "export")
            export_sha = digest(field(exported, "sha256"))
            checkpoint_sha = digest(field(v, "checkpoint_sha256"))
            for name, value in (("checkpoint", checkpoint_sha), ("export", export_sha)):
                path = Path(field(v, name) if name == "checkpoint" else field(exported, "path"))
                path = path if path.is_absolute() else ROOT / path
                if not path.is_file():
                    raise MissingEvidence("Missing training artifact: " + str(path))
                need(sha256_file(path) == value, "Training artifact bytes changed: " + name)
            for k, f in frozen.items():
                if f["block_id"] == bid:
                    need(single["base_models"][k] in (export_sha, checkpoint_sha), "Single-block evaluated model is not verified trained export/checkpoint")
        if set(selected) != set(blocks):
            raise MissingEvidence("Missing training proof for every pilot block")
        return {"blocks": blocks, "source_ply_is_valid_trained_export": True}
    for name, function in zip((f"G{i}" for i in range(1, 9)), (g1, g2, g3, g4, g5, g6, g7, g8)):
        try:
            results[name] = {"status": "PASS", "evidence": function()}
        except MissingEvidence as exc:
            results[name] = {"status": "BLOCKED", "reason": str(exc)}
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, RuntimeError, ImportError) as exc:
            results[name] = {"status": "FAIL", "reason": str(exc)}
    statuses = [r["status"] for r in results.values()]
    overall = "PASS" if all(s == "PASS" for s in statuses) else "FAIL" if "FAIL" in statuses else "BLOCKED"
    return {"schema_version": 1, "acceptance_plan_id": plan["plan_id"], "overall": overall,
            "rollout_allowed": overall == "PASS", "high_fidelity_accepted": False,
            "scope": "F1 pilot rollout gate; not six-floor independent 1:1 fidelity acceptance",
            "gates": results, "gate_thresholds": thresholds, "verified_asset_files": asset_cache.get("_pins", {})}


def require_rollout(report, plan_id):
    """Training-controller hook: reject a forged top-level PASS with missing gates."""
    need(report.get("acceptance_plan_id") == plan_id and report.get("overall") == "PASS"
         and report.get("rollout_allowed") is True, "Rollout gate is not PASS for this plan")
    need(set(report.get("gates", {})) == {f"G{i}" for i in range(1, 9)} and
         all(v.get("status") == "PASS" and v.get("evidence") for v in report["gates"].values()),
         "Every G1--G8 needs passing evidence")
    need(report.get("report_id") == object_sha({k: v for k, v in report.items() if k != "report_id"}), "Gate report digest changed")
    sources = field(report, "evidence_files")
    need(bool(sources), "Gate report has no source evidence files")
    need(all(sources.get(path) == sha for path, sha in report.get("verified_asset_files", {}).items()),
         "Gate evidence files omit or replace verified representation/renderer assets")
    for path, sha in sources.items():
        need(Path(path).is_file() and sha256_file(path) == sha, "Gate source changed/missing: " + path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "reports/acceptance_plan_v1.json")
    parser.add_argument("--evidence", required=True, type=Path, help="JSON manifest of report paths per evidence role")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        implementation = {str(ROOT / "scripts" / name): sha256_file(ROOT / "scripts" / name) for name in (
            "acceptance_gate.py", "build_acceptance_plan.py", "direction_support.py",
            "measure_surface_quality.py", "package_garage.py", "verify_training_checkpoint.py")}
        plan = load_plan(args.plan)
        manifest = json.loads(args.evidence.read_text())
        evidence, files = read_reports(manifest)
        result = evaluate_gate(plan, evidence)
        for path, sha in implementation.items():
            verify_artifact(path, sha)
        result["evidence_files"] = {str(args.plan.resolve()): sha256_file(args.plan),
                                    str(args.evidence.resolve()): sha256_file(args.evidence),
                                    **implementation,
                                    **plan["inputs"], **files, **result["verified_asset_files"]}
        result["report_id"] = object_sha(result)
        write_plan(args.output, result)
        print(json.dumps({"overall": result["overall"], "gates": {k: v["status"] for k, v in result["gates"].items()}}))
        return 0 if result["overall"] == "PASS" else 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, ImportError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

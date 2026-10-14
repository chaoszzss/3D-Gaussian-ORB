#!/usr/bin/env python3
"""Serial native whole-PLY evaluation on the original, fixed 28-core splits."""
import sys
sys.dont_write_bytecode = True
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def environment(project):
    env = os.environ.copy()
    for key, folder in {"XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
                        "TORCHINDUCTOR_CACHE_DIR": "torch_inductor", "TRITON_CACHE_DIR": "triton",
                        "HF_HOME": "huggingface", "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
                        "CUDA_CACHE_PATH": "cuda", "PIP_CACHE_DIR": "pip", "UV_CACHE_DIR": "uv",
                        "WANDB_DIR": "wandb", "WANDB_CONFIG_DIR": "wandb_config",
                        "__GL_SHADER_DISK_CACHE_PATH": "nvidia_gl"}.items():
        env[key] = str(project / "runtime/cache" / folder)
    env.update({key: str(project / "runtime/tmp") for key in ("TMPDIR", "TMP", "TEMP")})
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", __GL_SHADER_DISK_CACHE="0",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", MAX_JOBS="2")
    return env


os.environ.update(environment(ROOT))  # Before the helper's NumPy import; no Torch in this wrapper.
sys.path.insert(0, str(ROOT / "scripts"))
from package_garage import EXPECTED_IDS, _state, inspect_ply, local_path, read_json

PROTOCOL = "Exact original INRIA per-Gaussian radius>0; all source records considered"
BUDGETS = {"max_gaussians": 12_000_000, "max_ply_bytes": 3_221_225_472,
           "active_gaussian_budget": 4_000_000, "geometry_chunk_records": 200_000}
FALSE_FLAGS = {"high_fidelity_accepted": False, "floor_verified": False, "ramp_verified": False,
               "calibration_verified": False, "independent_geometry_acceptance": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path):
    before, digest = list(_state(path)), hashlib.sha256()
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(raw)
    require(before == list(_state(path)), "Input changed while hashing: " + str(path))
    return {"path": str(path), "stat": before, "sha256": digest.hexdigest()}


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    require(not path.is_symlink() and not temporary.is_symlink(), "Symlink output forbidden")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        directory = Path("/proc") / str(int(pid))
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "argv": [s.decode(errors="replace") for s in (directory / "cmdline").read_bytes().split(b"\0") if s]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved, current):
    return bool(saved and current and all(saved.get(k) == current.get(k) for k in ("pid", "start_ticks", "argv")))


def refuse_live_batch(project):
    status = project / "runs/core_joint_v1/status.json"
    if status.is_file():
        saved = read_json(local_path(project, status)).get("controller")
        require(not same_process(saved, process_identity(saved["pid"]) if saved else None),
                "Live core_joint_v1 controller; defer full-scene evaluation")
    # Scan as well as saved status: controller startup precedes its status write.
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            identity = process_identity(int(directory.name))
            tokens = identity["argv"] if identity else []
            absolute = str(project / "scripts/train_core_joint_batch.py") in tokens
            relative = "scripts/train_core_joint_batch.py" in tokens and (directory / "cwd").resolve() == project
            if identity and (absolute or relative):
                require("--inspect-job" in identity["argv"], "Live core_joint_v1 controller; defer full-scene evaluation")


@contextmanager
def lock(path, wait=False, project=None):
    require(not path.is_symlink(), "Symlink lock forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(wait, "Recorded controller/child still holds lock: " + str(path))
                refuse_live_batch(project)
                time.sleep(1)  # No timeout, cancellation never schedules a duplicate.
        yield handle
        # Close only: an inherited FD keeps the same lock alive after parent death.


def world_fixed(report):
    transform = report.get("world_parser_transform")
    identity = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
    return report.get("world_parser_scale") == 1. and transform in (identity, identity + [[0., 0., 0., 1.]])


def frame_key(frame):
    value = frame["group_id"]
    require(type(value) is int or (isinstance(value, str) and re.fullmatch(r"\d+", value)), "Noninteger group ID")
    return int(value), frame["camera_id"]


def build_plan(project, candidate_ply, baseline_ply=None, split="dev", max_images=0, budgets=None):
    require(split in ("dev", "test") and type(max_images) is int and max_images >= 0, "Invalid split/image limit")
    budgets = dict(BUDGETS if budgets is None else budgets)
    for key, ceiling in {**BUDGETS, "active_gaussian_budget": 6_000_000, "geometry_chunk_records": 500_000}.items():
        require(type(budgets.get(key)) is int and 0 < budgets[key] <= ceiling, "Invalid budget: " + key)
    inputs = {}
    def pin(value):
        path = local_path(project, value)
        if str(path) not in inputs:
            inputs[str(path)] = snapshot(path)
        return str(path)
    models = {}
    for role, value in (("candidate", candidate_ply), ("baseline", baseline_ply)):
        if value is not None:
            path = pin(value)
            with Path(path).open("rb") as handle:
                spec = inspect_ply(handle)
            require(spec["count"] <= budgets["max_gaussians"] and inputs[path]["stat"][2] <= budgets["max_ply_bytes"], "Whole PLY exceeds budget")
            require(inputs[path]["stat"][2] == len(spec["header"]) + spec["stride"] * spec["count"], "Invalid whole PLY payload size")
            models[role] = dict(inputs[path], gaussians=spec["count"])
    pin("scripts/evaluate_gaussian_ply.py"); pin("scripts/native_gaussian_visibility.py")
    jobs = []
    for bid in EXPECTED_IDS:
        source = pin(f"runs/garage_surface_v1/{bid}/eval_dev/metrics.json")
        original = read_json(Path(source))
        require(original.get("status") == "evaluated" and original.get("split") == "dev" and world_fixed(original), "Original template is not evaluated/fixed world: " + bid)
        config, checkpoint = pin(original["config"]), pin(original["checkpoint"])
        reference = source if split == "dev" else pin(f"runs/garage_surface_v1/{bid}/eval_test/metrics.json")
        reference_data = read_json(Path(reference))
        require(reference_data.get("status") == "evaluated" and reference_data.get("split") == split and world_fixed(reference_data)
                and local_path(project, reference_data["config"]) == Path(config)
                and local_path(project, reference_data["checkpoint"]) == Path(checkpoint)
                and reference_data["step"] == original["step"], "Original split/template identity mismatch")
        transforms = pin(f"data/garage_surface_v1/{bid}/transforms.json")
        metadata = read_json(Path(transforms))
        names = metadata["val_filenames" if split == "dev" else "test_filenames"]
        frames = {frame["file_path"]: frame for frame in metadata["frames"]}
        require(len(frames) == len(metadata["frames"]) and isinstance(names, list)
                and len(set(names)) == len(names), "Duplicate/malformed original source split")
        order = [r["file_path"] for r in reference_data["frames"]]
        require(names and len(set(order)) == len(order) == reference_data["images"] and set(order) == set(names), "Original split report does not cover exact source frames")
        expected = []
        for name in order:
            frame = frames[name]
            require(all(isinstance(frame.get(key), str) and frame[key] for key in
                        ("file_path", "mask_path", "lidar_depth_path", "lidar_depth_mask_path")), "Missing original V1 image/mask/depth/support reference")
            require(frame_key(frame) == frame_key(next(r for r in reference_data["frames"] if r["file_path"] == name)), "Original frame identity mismatch")
            assets = {key: pin(Path(transforms).parent / frame[key]) for key in
                      ("file_path", "mask_path", "lidar_depth_path", "lidar_depth_mask_path") if frame.get(key)}
            camera = {key: metadata.get(key, frame.get(key)) for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
            require(frame["camera_id"] in ("camera_2", "camera_3", "camera_4", "camera_5"), "Unknown physical camera")
            camera["physical_camera_id"] = int(frame["camera_id"].rsplit("_", 1)[1])
            camera["camera_to_worlds_ns"] = frame["transform_matrix"][:3]
            expected.append({"file_path": name, "group_id": frame["group_id"], "camera_id": frame["camera_id"],
                             "assets": assets, "camera": camera})
        require(len({frame_key(f) for f in expected}) == len(expected), "Duplicate group/camera inside core")
        jobs.append({"id": bid, "config": config, "checkpoint": checkpoint, "step": original["step"],
                     "template_report": source, "original_split_report": reference, "transforms": transforms, "frames": expected})
    return {"schema_version": 1, "planned_core_ids": list(EXPECTED_IDS), "split": split, "max_images": max_images,
            "models": models, "budgets": budgets, "inputs": inputs, "jobs": jobs, "coordinates": "camera_init", "units": "m"}


def check_inputs(plan):
    for path, item in plan["inputs"].items():
        require(list(_state(Path(path))) == item["stat"], "Pinned input changed: " + path)


def command(project, plan, job, role, output):
    return [str(project / ".venv-train/bin/python"), "-B", str(project / "scripts/evaluate_gaussian_ply.py"),
            "--config", job["config"], "--ply", plan["models"][role]["path"], "--split", plan["split"],
            "--output", str(output), "--max-images", str(plan["max_images"]), "--native-visibility",
            *[token for key, value in plan["budgets"].items() for token in ("--" + key.replace("_", "-"), str(value))]]


def validate_report(path, plan, job, role):
    report, model = read_json(path), plan["models"][role]
    require(report.get("status") == "evaluated" and report.get("split") == plan["split"] and world_fixed(report)
            and report.get("backend") == "hbb1-original-2dgs" and report.get("checkpoint_is_template") is True
            and report.get("parameter_origin", "").startswith("All Gaussian parameters replaced by source_ply"), "Report lacks native/template identity")
    for key in ("config", "checkpoint", "transforms", "step"):
        require(report.get(key) == job[key], "Wrong template/source identity: " + key)
    for key in ("config", "transforms"):
        require(report.get(key + "_sha256") == plan["inputs"][job[key]]["sha256"], "Wrong source SHA: " + key)
    source = report.get("source_ply", {})
    require(report.get("gaussians") == source.get("gaussians") == model["gaussians"]
            and source.get("path") == model["path"] and source.get("size_bytes") == model["stat"][2]
            and source.get("sha256") == report.get("source_ply_sha256") == model["sha256"], "Whole-model count/SHA/path mismatch")
    bounds = report.get("bounds", {})
    require(bounds.get("native_visibility") is True and all(bounds.get(k) == v for k, v in plan["budgets"].items()
                                                          if k != "geometry_chunk_records"), "Wrong native allocation budgets")
    expected = job["frames"][:plan["max_images"]] if plan["max_images"] else job["frames"]
    rows = report.get("frames", [])
    require(report.get("split_images") == len(job["frames"]) and report.get("images") == len(rows) == len(expected)
            and report.get("frame_indices") == list(range(len(expected))), "Incomplete selected split report")
    for index, (row, frame) in enumerate(zip(rows, expected)):
        require(row.get("dataset_index") == index and row.get("file_path") == frame["file_path"] and frame_key(row) == frame_key(frame), "Wrong split frame ID/order")
        for asset, field in {"file_path": "source_image_path", "mask_path": "source_mask_path",
                             "lidar_depth_path": "source_lidar_depth_path", "lidar_depth_mask_path": "source_lidar_support_path"}.items():
            require(row.get(field) == frame["assets"].get(asset), "Wrong fixed mask/depth/image reference")
        for key, value in frame["camera"].items():
            actual = row.get("camera", {}).get(key)
            flat = lambda v: [x for line in v for x in line] if isinstance(v, list) else [v]
            require(actual is not None and len(flat(actual)) == len(flat(value))
                    and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5) for a, b in zip(flat(actual), flat(value))), "Changed fixed camera")
        visibility = row.get("native_visibility", {})
        require(visibility.get("protocol") == PROTOCOL and visibility.get("source_records") == model["gaussians"]
                and type(visibility.get("active_records")) is int and 0 <= visibility["active_records"] <= min(model["gaussians"], plan["budgets"]["active_gaussian_budget"])
                and visibility.get("geometry_chunk_records") == plan["budgets"]["geometry_chunk_records"]
                and re.fullmatch(r"[0-9a-f]{64}", visibility.get("active_index_sha256", "")), "Missing whole-source visibility proof")
        require(type(row.get("rgb_valid_pixels")) is int and row["rgb_valid_pixels"] > 0
                and all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ("rgb_mse", "psnr_db"))
                and 0 <= row["rgb_mse"] <= 1, "Invalid RGB scores/denominator")
        ssim, pixels = row.get("ssim_valid_pixels"), row.get("ssim_compared_pixels")
        require(type(pixels) is int and 0 <= pixels <= row["rgb_valid_pixels"]
                and (ssim is None or (type(ssim) in (int, float) and math.isfinite(ssim) and -1 <= ssim <= 1 and pixels > 0)), "Invalid SSIM score/denominator")
        eligible, compared = row.get("map_depth_eligible_pixels", 0), row.get("map_depth_compared_pixels", 0)
        require(type(eligible) is int and type(compared) is int and 0 <= compared <= eligible, "Invalid fixed depth denominator")
        if "lidar_depth_path" in frame["assets"]:
            require("map_depth_eligible_pixels" in row and row.get("map_depth_missing_pixels") == eligible - compared, "Missing fixed-depth score")
        if eligible:
            require(isinstance(row.get("map_depth_capped_relative_error_all_supported"), (int, float))
                    and 0 <= row["map_depth_capped_relative_error_all_supported"] <= 1, "Invalid missing-penalized depth score")
    return report


def scores(rows):
    n, pixels = len(rows), sum(r["rgb_valid_pixels"] for r in rows)
    eligible, compared = (sum(r.get(key, 0) for r in rows) for key in ("map_depth_eligible_pixels", "map_depth_compared_pixels"))
    ssim_pixels = sum(r.get("ssim_compared_pixels", 0) for r in rows if r.get("ssim_valid_pixels") is not None)
    return {"frame_observations": n, "psnr_db_frame_mean": sum(r["psnr_db"] for r in rows) / n if n else None,
            "rgb_valid_pixels": pixels, "rgb_mse_pixel_mean": sum(r["rgb_mse"] * r["rgb_valid_pixels"] for r in rows) / pixels if pixels else None,
            "ssim_pixel_mean": sum(r["ssim_valid_pixels"] * r.get("ssim_compared_pixels", 0) for r in rows if r.get("ssim_valid_pixels") is not None) / ssim_pixels if ssim_pixels else None,
            "depth_fixed_eligible_pixels": eligible, "depth_compared_pixels": compared,
            "depth_coverage": compared / eligible if eligible else None,
            "depth_capped_relative_error_missing_penalized": sum(r["map_depth_capped_relative_error_all_supported"] * r["map_depth_eligible_pixels"]
                                                                for r in rows if r.get("map_depth_eligible_pixels", 0)) / eligible if eligible else None}


def aggregate(plan, reports):
    observations, unique, duplicates, comparisons = {}, {}, {}, []
    for role in plan["models"]:
        entries = [(job, reports[(job["id"], role)]) for job in plan["jobs"] if (job["id"], role) in reports]
        rows, chosen = [], {}
        for job, report in entries:
            for row in report["frames"]:
                rows.append(row); key = frame_key(row)
                if key not in chosen:
                    chosen[key] = row
                if role == "candidate":
                    frame = job["frames"][row["dataset_index"]]
                    assets = {k: plan["inputs"][v] for k, v in frame["assets"].items()}
                    duplicates.setdefault(str(key), []).append({"core_id": job["id"], "assets": assets,
                                                                 "depth_eligible_pixels": row.get("map_depth_eligible_pixels", 0)})
        observations[role], unique[role] = scores(rows), scores(list(chosen.values()))
        unique[role]["unique_image_count"] = len(chosen)
    for job in plan["jobs"]:
        candidate = reports.get((job["id"], "candidate"))
        if not candidate:
            continue
        old = read_json(Path(job["original_split_report"]))
        old_rows = {frame_key(row): row for row in old["frames"]}
        comparisons.append({"core_id": job["id"], "scope": "original single block vs whole candidate; not paired whole maps",
                            "frames": len(candidate["frames"]), "original_psnr_db_frame_mean": sum(old_rows[frame_key(r)]["psnr_db"] for r in candidate["frames"]) / len(candidate["frames"]),
                            "candidate_psnr_db_frame_mean": scores(candidate["frames"])["psnr_db_frame_mean"],
                            "depth_protocol_comparable": False})
        baseline = reports.get((job["id"], "baseline"))
        if baseline:
            for a, b in zip(candidate["frames"], baseline["frames"]):
                require(frame_key(a) == frame_key(b) and a["rgb_valid_pixels"] == b["rgb_valid_pixels"]
                        and a.get("map_depth_eligible_pixels", 0) == b.get("map_depth_eligible_pixels", 0), "Paired whole-map denominators differ")
    return {"frame_observations": observations, "unique_images": unique,
            "unique_policy": "canonical first stable core ID per (integer group_id,camera_id); depth teachers never averaged",
            "duplicates": {k: {"canonical_core_id": v[0]["core_id"], "observations": v,
                                "depth_teachers_differ": len({tuple((a, r["assets"].get(a, {}).get("path"), r["assets"].get(a, {}).get("sha256"))
                                                                  for a in ("lidar_depth_path", "lidar_depth_mask_path")) for r in v}) > 1}
                           for k, v in duplicates.items() if len(v) > 1}, "single_block_references": comparisons,
            "paired_whole_map_comparison": "same source cameras, masks and fixed denominators" if "baseline" in plan["models"] else None}


def evaluate(candidate_ply, output, *, baseline_ply=None, core_ids=None, split="dev", max_images=0,
             resume=False, budgets=None, project=ROOT):
    project = Path(project).resolve()
    require(project == ROOT or ROOT in project.parents, "Project must remain inside ROOT")
    refuse_live_batch(project)
    out = local_path(project, output)
    require(out.parent.is_dir() and (resume or not out.exists()), "Output must be new, or explicitly --resume")
    require(not resume or out.is_dir(), "Resume requires existing output")
    selected = list(EXPECTED_IDS) if core_ids is None else list(core_ids)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(EXPECTED_IDS), "Invalid selected core IDs")
    if not out.exists():
        out.mkdir()
    with lock(out / ".controller.lock") as controller_lock:
        plan_path, state_path = out / "plan.json", out / "state.json"
        state = read_json(state_path) if state_path.exists() else {"children": {}}
        saved = state.get("controller")
        require(not (saved and saved["pid"] != os.getpid() and same_process(saved, process_identity(saved["pid"]))), "Known live evaluation controller")
        for child in state["children"].values():
            identity = child.get("identity")
            require(not same_process(identity, process_identity(identity["pid"]) if identity else None), "Known live evaluation child; observe, do not duplicate")
        state.update(status="incomplete", complete=False, controller=process_identity(os.getpid()), requested_core_ids=selected, **FALSE_FLAGS)
        atomic_json(state_path, state)
        reports = {}
        try:
            plan = build_plan(project, candidate_ply, baseline_ply, split, max_images, budgets)
            if resume:
                require(read_json(plan_path) == plan, "Resume input/plan identity changed")
            else:
                atomic_json(plan_path, plan)
            for job in plan["jobs"]:
                for role in plan["models"]:
                    key, destination = job["id"] + ":" + role, out / (job["id"] + "_" + role)
                    child = state["children"].get(key, {})
                    argv = command(project, plan, job, role, destination)
                    require(not child.get("argv") or child["argv"] == argv, "Saved child command identity changed")
                    require(not same_process(child.get("identity"), process_identity(child["identity"]["pid"]) if child.get("identity") else None), "Known live evaluation child; observe, do not duplicate")
                    report_path = destination / "metrics.json"
                    if report_path.exists():
                        if child.get("validated_report"):
                            require(snapshot(report_path) == child["validated_report"], "Previously validated report changed")
                        reports[(job["id"], role)] = validate_report(report_path, plan, job, role)
                    elif job["id"] in selected:
                        require(not destination.exists() and not child, "Interrupted/incomplete child has no valid report; use a new output, never auto-restart")
                        check_inputs(plan); refuse_live_batch(project)
                        with lock(project / "runtime/core_joint_gpu.lock", wait=True, project=project) as gpu_lock:
                            refuse_live_batch(project)
                            with (out / (key.replace(":", "_") + ".log")).open("ab", buffering=0) as log:
                                process = subprocess.Popen(argv, cwd=str(project), env=environment(project), stdout=log,
                                                           stderr=subprocess.STDOUT, pass_fds=(controller_lock.fileno(), gpu_lock.fileno()))
                                identity = process_identity(process.pid)
                                require(identity is not None or process.poll() is not None, "Cannot identify live child PID/start ticks")
                                state["children"][key] = {"identity": identity, "launched_identity": identity,
                                                          "argv": argv, "report": str(report_path)}
                                atomic_json(state_path, state)
                                code = process.wait()  # No timeout/retry. Inherited locks protect orphaned children.
                                state["children"][key].update(identity=None, returncode=code)
                                atomic_json(state_path, state)
                                require(code == 0 or report_path.exists(), "Native child failed; retained log/partial state")
                            check_inputs(plan)
                            reports[(job["id"], role)] = validate_report(report_path, plan, job, role)
                    if (job["id"], role) in reports:
                        state["children"].setdefault(key, {}).update(validated_report=snapshot(report_path), identity=None, argv=argv)
                        atomic_json(state_path, state)
                state["validated_core_ids"] = [j["id"] for j in plan["jobs"] if all((j["id"], r) in reports for r in plan["models"])]
                atomic_json(state_path, state)
            check_inputs(plan)
            for child in state["children"].values():
                if child.get("validated_report"):
                    require(snapshot(Path(child["validated_report"]["path"])) == child["validated_report"], "Validated report changed before summary")
            complete = len(reports) == 28 * len(plan["models"]) and all(len(report["frames"]) == report["split_images"] for report in reports.values())
            result = {"status": "native_fullscope_evaluated_pending_quality_acceptance" if complete else "partial",
                      "complete": complete, "native_all28_assessed": complete, "planned_core_ids": list(EXPECTED_IDS),
                      "validated_core_ids": state["validated_core_ids"], "split": split, "models": plan["models"],
                      "geometry_protocol": "Conditioned on fixed original V1 LiDAR/depth/support masks; not independent measurement GT",
                      **FALSE_FLAGS, **aggregate(plan, reports), "coverage_scope": "all original collection cameras; floors/ramps/holes remain unverified"}
            atomic_json(out / "summary.json", result)
            state.update(status=result["status"], complete=complete, controller=None)
            atomic_json(state_path, state)
            return result
        except BaseException as exc:
            state.update(status="incomplete", complete=False, error=str(exc), controller=None)
            atomic_json(state_path, state)
            atomic_json(out / "summary.json", {"status": "incomplete", "complete": False, "error": str(exc),
                                               "validated_core_ids": state.get("validated_core_ids", []), **FALSE_FLAGS})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ply", required=True)
    parser.add_argument("--baseline-ply")
    parser.add_argument("--output", required=True)
    parser.add_argument("--core-ids", nargs="+", help="Canonical block_XXX IDs; progress subset only")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="TEST requires explicit --split test")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    for key, default in BUDGETS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.candidate_ply, args.output, baseline_ply=args.baseline_ply, core_ids=args.core_ids,
                          split=args.split, max_images=args.max_images, resume=args.resume,
                          budgets={key: getattr(args, key) for key in BUDGETS})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "incomplete", "complete": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("status", "complete", "validated_core_ids")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

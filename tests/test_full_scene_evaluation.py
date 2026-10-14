"""No GPU children: synthetic native reports and subprocess doubles inside ROOT."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_scene", ROOT / "scripts/evaluate_full_scene.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def put_json(path, value):
    put(path, json.dumps(value, allow_nan=False).encode())


@pytest.fixture
def scene(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="full_scene_test_", dir=ROOT / "runtime/tmp") as directory:
        project = Path(directory)
        for name, count in (("candidate", 8), ("baseline", 17)):
            names = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
            header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n" +
                      "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
            row = [0.] * len(names); row[names.index("rot_0")] = 1.
            put(project / (name + ".ply"), header + struct.pack("<14f", *row) * count)
        for script in ("evaluate_gaussian_ply.py", "native_gaussian_visibility.py"):
            put(project / "scripts" / script, b"fixture only; never executed")
        for i, bid in enumerate(M.EXPECTED_IDS):
            root = project / "data/garage_surface_v1" / bid
            frames = []
            for j in range(3):
                group = 1720268000000000001 if i in (0, 1) and j == 0 else 1720268000000010001 + 100 * i + j
                name = f"images/{j}.png"
                shared = i in (0, 1) and j == 0
                pose = [[1., 0., 0., 0. if shared else float(i)], [0., 1., 0., 0.], [0., 0., 1., 2.], [0., 0., 0., 1.]]
                frame = {"file_path": name, "camera_id": "camera_3" if j == 0 else "camera_2", "group_id": str(group),
                         "transform_matrix": pose, "fl_x": 10., "fl_y": 10., "cx": 5., "cy": 5., "w": 10, "h": 10,
                         "mask_path": f"masks/{j}.png", "lidar_depth_path": f"depth/{j}.npy", "lidar_depth_mask_path": f"support/{j}.png"}
                frames.append(frame)
                for asset in ("file_path", "mask_path", "lidar_depth_path", "lidar_depth_mask_path"):
                    put(root / frame[asset], f"{asset}:{0 if shared and asset in ('file_path', 'mask_path') else i}:{j}".encode())
            put_json(root / "transforms.json", {"frames": frames, "train_filenames": [frames[2]["file_path"]],
                                                "val_filenames": [f["file_path"] for f in frames[:2]], "test_filenames": [frames[1]["file_path"]]})
            config, checkpoint = project / f"runs/templates/{bid}/config.yml", project / f"runs/templates/{bid}/step.ckpt"
            put(config, b"fixed world config; fixture never parsed by native setup")
            put(checkpoint, b"template checkpoint")
            for split, selected in (("dev", frames[:2]), ("test", frames[1:2])):
                report = {"status": "evaluated", "split": split, "config": str(config), "checkpoint": str(checkpoint), "step": 7,
                          "world_parser_transform": [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]], "world_parser_scale": 1.,
                          "images": len(selected), "frames": [dict(f, psnr_db=20. + i) for f in selected]}
                put_json(project / f"runs/garage_surface_v1/{bid}/eval_{split}/metrics.json", report)
        monkeypatch.setattr(M, "refuse_live_batch", lambda project: None)
        scene = SimpleNamespace(project=project, candidate="candidate.ply", baseline="baseline.ply", calls=[])
        scene.plan = M.build_plan(project, scene.candidate, scene.baseline)
        yield scene


def native_report(plan, job, role):
    model = plan["models"][role]
    expected = job["frames"][:plan["max_images"]] if plan["max_images"] else job["frames"]
    rows = []
    core = M.EXPECTED_IDS.index(job["id"])
    for index, frame in enumerate(expected):
        pixels, eligible = 10 + core * 3 + index, 10 + core + index
        mse = (.1 if role == "candidate" else .2) + core * .001
        row = {"file_path": frame["file_path"], "group_id": frame["group_id"], "camera_id": frame["camera_id"],
               "dataset_index": index, "camera": frame["camera"], "rgb_valid_pixels": pixels,
               "rgb_mse": mse, "psnr_db": -10 * math.log10(mse), "ssim_valid_pixels": .7,
               "ssim_compared_pixels": pixels - 2, "map_depth_eligible_pixels": eligible,
               "map_depth_compared_pixels": eligible - 1, "map_depth_missing_pixels": 1,
               "map_depth_capped_relative_error_all_supported": .3 if role == "candidate" else .5,
               "native_visibility": {"protocol": M.PROTOCOL, "source_records": model["gaussians"], "active_records": 4,
                                     "geometry_chunk_records": plan["budgets"]["geometry_chunk_records"], "active_index_sha256": "a" * 64}}
        for asset, field in {"file_path": "source_image_path", "mask_path": "source_mask_path",
                             "lidar_depth_path": "source_lidar_depth_path", "lidar_depth_mask_path": "source_lidar_support_path"}.items():
            row[field] = frame["assets"].get(asset)
        rows.append(row)
    return {"status": "evaluated", "split": plan["split"], "config": job["config"], "checkpoint": job["checkpoint"],
            "transforms": job["transforms"], "step": job["step"], "config_sha256": plan["inputs"][job["config"]]["sha256"],
            "transforms_sha256": plan["inputs"][job["transforms"]]["sha256"], "backend": "hbb1-original-2dgs",
            "checkpoint_is_template": True, "parameter_origin": "All Gaussian parameters replaced by source_ply; fixed template",
            "world_parser_transform": [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]], "world_parser_scale": 1.,
            "gaussians": model["gaussians"], "source_ply": {"path": model["path"], "sha256": model["sha256"],
                                                            "size_bytes": model["stat"][2], "gaussians": model["gaussians"]},
            "source_ply_sha256": model["sha256"], "bounds": dict(plan["budgets"], native_visibility=True),
            "images": len(rows), "split_images": len(job["frames"]), "frames": rows, "frame_indices": list(range(len(rows)))}


@pytest.fixture
def simulator(scene, monkeypatch):
    def popen(argv, **kwargs):
        scene.calls.append((argv, kwargs))
        for fd, path in zip(kwargs["pass_fds"], (scene.project / "output/.controller.lock", scene.project / "runtime/core_joint_gpu.lock")):
            assert M.os.fstat(fd).st_ino == path.stat().st_ino
            with path.open("a+") as competitor:
                with pytest.raises(BlockingIOError):
                    M.fcntl.flock(competitor, M.fcntl.LOCK_EX | M.fcntl.LOCK_NB)
        args = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith("--") and argv[i] != "--native-visibility"}
        plan = M.build_plan(scene.project, scene.candidate, scene.baseline if args["--ply"].endswith("baseline.ply") or any(
            "baseline" in x for x in scene.plan["models"]) else None, split=args["--split"], max_images=int(args["--max-images"]))
        job = next(j for j in plan["jobs"] if j["config"] == args["--config"])
        role = "candidate" if args["--ply"].endswith("candidate.ply") else "baseline"
        put_json(Path(args["--output"]) / "metrics.json", native_report(plan, job, role))
        return SimpleNamespace(pid=99999999, poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(M.subprocess, "Popen", popen)
    return scene


def run(scene, **kwargs):
    values = {"project": scene.project, "baseline_ply": scene.baseline}
    values.update(kwargs)
    return M.evaluate(scene.candidate, "output", **values)


def test_global_plan_and_exact_cli_never_sample_whole_models(scene):
    assert scene.plan["planned_core_ids"] == list(M.EXPECTED_IDS)
    assert len(scene.plan["jobs"]) == 28
    for job in scene.plan["jobs"]:
        argv = M.command(scene.project, scene.plan, job, "baseline", scene.project / "output/job")
        for flag, value in (("--max-gaussians", "12000000"), ("--max-ply-bytes", "3221225472"),
                            ("--active-gaussian-budget", "4000000"), ("--geometry-chunk-records", "200000"), ("--max-images", "0")):
            assert argv[argv.index(flag) + 1] == value
        assert "--native-visibility" in argv and "--frame-indices" not in argv
        assert argv[argv.index("--ply") + 1] == scene.plan["models"]["baseline"]["path"]


def test_full_28_serial_reports_and_unique_teacher_aggregation(simulator):
    result = run(simulator)
    assert result["complete"] is result["native_all28_assessed"] is True
    assert result["status"] == "native_fullscope_evaluated_pending_quality_acceptance"
    assert all(result[k] is False for k in M.FALSE_FLAGS)
    assert "not independent measurement GT" in result["geometry_protocol"]
    assert len(simulator.calls) == 56
    assert result["frame_observations"]["candidate"]["frame_observations"] == 56
    assert result["unique_images"]["candidate"]["unique_image_count"] == 55
    duplicate = next(iter(result["duplicates"].values()))
    assert duplicate["canonical_core_id"] == "block_000"
    assert [r["core_id"] for r in duplicate["observations"]] == ["block_000", "block_001"]
    assert duplicate["depth_teachers_differ"] is True
    assert result["unique_images"]["candidate"]["depth_fixed_eligible_pixels"] == sum(10 + i + j for i in range(28) for j in range(2)) - 11
    assert result["unique_images"]["candidate"]["depth_capped_relative_error_missing_penalized"] == pytest.approx(.3)
    assert result["unique_images"]["baseline"]["depth_capped_relative_error_missing_penalized"] == pytest.approx(.5)
    expected_mse = sum((.1 + i * .001) * (10 + i * 3 + j) for i in range(28) for j in range(2))
    denominator = sum(10 + i * 3 + j for i in range(28) for j in range(2))
    assert result["frame_observations"]["candidate"]["rgb_mse_pixel_mean"] == pytest.approx(expected_mse / denominator)
    assert all("single block vs whole candidate" in r["scope"] for r in result["single_block_references"])
    for argv, options in simulator.calls:
        assert len(options["pass_fds"]) == 2 and options["cwd"] == str(simulator.project)
        assert options["env"]["TORCH_HOME"].startswith(str(simulator.project))
        assert options["env"]["TMPDIR"].startswith(str(simulator.project))
        assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "timeout" not in options


def test_subset_is_progress_only_then_resume_skips_actual_validated_reports(simulator):
    first = run(simulator, core_ids=["block_014"])
    assert first["complete"] is False and first["status"] == "partial"
    assert first["planned_core_ids"] == list(M.EXPECTED_IDS)
    assert len(simulator.calls) == 2
    second = run(simulator, resume=True)
    assert second["complete"] is True and len(simulator.calls) == 56
    run(simulator, resume=True)
    assert len(simulator.calls) == 56


def test_max_images_limit_cannot_claim_global_split_complete(simulator):
    result = run(simulator, max_images=1)
    assert len(result["validated_core_ids"]) == 28
    assert result["complete"] is False and result["native_all28_assessed"] is False


def test_test_split_only_when_explicit_and_pinned_to_same_template(simulator):
    result = run(simulator, split="test", core_ids=["block_000"])
    assert result["split"] == "test" and result["complete"] is False
    assert all(argv[argv.index("--split") + 1] == "test" for argv, _ in simulator.calls)
    assert M.build_plan(simulator.project, simulator.candidate)["split"] == "dev"


@pytest.mark.parametrize("fault", ["model_count", "model_sha", "model_path", "template_sha", "checkpoint", "world",
                                  "frame", "mask", "depth_teacher", "visibility_count", "visibility_missing", "sampling",
                                  "budget", "camera", "missing_frame"])
def test_wrong_whole_model_template_or_frame_proof_never_validates(scene, fault):
    job, role = scene.plan["jobs"][0], "candidate"
    report = native_report(scene.plan, job, role)
    row = report["frames"][0]
    if fault == "model_count": report["gaussians"] -= 1
    elif fault == "model_sha": report["source_ply_sha256"] = "0" * 64
    elif fault == "model_path": report["source_ply"]["path"] = scene.plan["models"]["baseline"]["path"]
    elif fault == "template_sha": report["config_sha256"] = "0" * 64
    elif fault == "checkpoint": report["checkpoint"] = str(scene.project / "other.ckpt")
    elif fault == "world": report["world_parser_scale"] = .5
    elif fault == "frame": row["group_id"] = "999"
    elif fault == "mask": row["source_mask_path"] = None
    elif fault == "depth_teacher": row["source_lidar_depth_path"] = None
    elif fault == "visibility_count": row["native_visibility"]["source_records"] -= 1
    elif fault == "visibility_missing": row.pop("native_visibility")
    elif fault == "sampling": row["native_visibility"]["protocol"] = "sampled spatial subset"
    elif fault == "budget": report["bounds"]["active_gaussian_budget"] = 100
    elif fault == "camera": row["camera"] = dict(row["camera"], fl_x=20.)
    else: report["frames"].pop()
    path = scene.project / "child/metrics.json"
    put_json(path, report)
    with pytest.raises(ValueError):
        M.validate_report(path, scene.plan, job, role)


def test_paired_whole_maps_require_same_frame_depth_and_mask_denominators(scene):
    job = scene.plan["jobs"][0]
    candidate, baseline = (native_report(scene.plan, job, role) for role in ("candidate", "baseline"))
    baseline["frames"][0]["map_depth_eligible_pixels"] += 1
    with pytest.raises(ValueError, match="denominators differ"):
        M.aggregate(scene.plan, {(job["id"], "candidate"): candidate, (job["id"], "baseline"): baseline})


def test_crash_after_native_report_recovers_finished_file_without_duplicate(simulator):
    run(simulator, core_ids=["block_000"])
    state_path = simulator.project / "output/state.json"
    state = json.loads(state_path.read_text())
    state["children"]["block_000:candidate"] = {"identity": {"pid": 99999998, "start_ticks": "42", "argv": ["native"]}}
    put_json(state_path, state)
    run(simulator, core_ids=["block_000"], resume=True)
    assert len(simulator.calls) == 2


def test_known_live_child_refused_even_if_report_exists_and_pid_reuse_is_safe(simulator, monkeypatch):
    run(simulator, core_ids=["block_000"])
    identity = {"pid": 99999998, "start_ticks": "42", "argv": ["native"]}
    state_path = simulator.project / "output/state.json"
    state = json.loads(state_path.read_text())
    state["children"]["block_000:candidate"]["identity"] = identity
    put_json(state_path, state)
    real = M.process_identity
    monkeypatch.setattr(M, "process_identity", lambda pid: identity if pid == identity["pid"] else real(pid))
    with pytest.raises(ValueError, match="Known live evaluation child"):
        run(simulator, resume=True, core_ids=["block_000"])
    assert len(simulator.calls) == 2
    monkeypatch.setattr(M, "process_identity", lambda pid: dict(identity, start_ticks="43") if pid == identity["pid"] else real(pid))
    run(simulator, resume=True, core_ids=["block_000"])
    assert len(simulator.calls) == 2


def test_interrupted_dead_child_without_metrics_never_restarts(simulator):
    run(simulator, core_ids=["block_000"])
    put(simulator.project / "output/block_001_candidate/unfinished.png", b"partial")
    with pytest.raises(ValueError, match="Interrupted/incomplete"):
        run(simulator, resume=True, core_ids=["block_001"])
    assert len(simulator.calls) == 2
    assert json.loads((simulator.project / "output/state.json").read_text())["complete"] is False


@pytest.mark.parametrize("changed", ["candidate.ply", "runs/templates/block_000/config.yml",
                                      "data/garage_surface_v1/block_000/masks/0.png", "scripts/native_gaussian_visibility.py"])
def test_resume_changed_inputs_invalidates_completion_without_new_children(simulator, changed):
    run(simulator)
    path = simulator.project / changed
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run(simulator, resume=True)
    assert len(simulator.calls) == 56
    assert json.loads((simulator.project / "output/summary.json").read_text())["complete"] is False


def test_tampered_completed_report_is_not_silently_retrusted(simulator):
    run(simulator)
    path = simulator.project / "output/block_000_candidate/metrics.json"
    report = json.loads(path.read_text()); report["frames"][0]["rgb_mse"] += .01
    put_json(path, report)
    with pytest.raises(ValueError, match="validated report changed"):
        run(simulator, resume=True)
    assert len(simulator.calls) == 56
    assert json.loads((simulator.project / "output/summary.json").read_text())["complete"] is False


def test_live_batch_refusal_checks_start_ticks_and_command_before_any_gpu(scene, monkeypatch):
    # Saved status catches a relative controller command during a CPU/GPU gap.
    saved = {"pid": 99999998, "start_ticks": "42", "argv": ["python", "scripts/train_core_joint_batch.py"]}
    put_json(scene.project / "runs/core_joint_v1/status.json", {"controller": saved})
    # Reload the production function that the fixture deliberately disabled.
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "process_identity", lambda pid: saved if pid == saved["pid"] else None)
    with pytest.raises(ValueError, match="Live core_joint_v1"):
        module.evaluate(scene.candidate, "blocked", project=scene.project)
    assert not (scene.project / "blocked").exists()
    monkeypatch.setattr(module, "process_identity", lambda pid: dict(saved, start_ticks="43") if pid == saved["pid"] else None)
    module.refuse_live_batch(scene.project)  # Reused PID is not that controller.


def test_output_lock_contention_refuses_duplicate_controller(scene):
    out = scene.project / "output"; out.mkdir()
    with M.lock(out / ".controller.lock"):
        with pytest.raises(ValueError, match="still holds lock"):
            run(scene, resume=True)


def test_timeout_never_retries_a_child_or_schedules_another(scene, monkeypatch):
    calls = []
    saved = {"pid": 99999998, "start_ticks": "42", "argv": ["native"]}
    def popen(argv, **options):
        calls.append(argv)
        put(Path(argv[argv.index("--output") + 1]) / "partial.png", b"unfinished")
        def wait(): raise M.subprocess.TimeoutExpired("native", 1)
        return SimpleNamespace(pid=saved["pid"], poll=lambda: None, wait=wait)
    real = M.process_identity
    monkeypatch.setattr(M.subprocess, "Popen", popen)
    monkeypatch.setattr(M, "process_identity", lambda pid: saved if pid == saved["pid"] else real(pid))
    with pytest.raises(M.subprocess.TimeoutExpired): run(scene)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Known live evaluation child"): run(scene, resume=True)
    assert len(calls) == 1


def test_busy_gpu_lock_waits_without_failing_or_restarting_other_work(simulator, monkeypatch):
    real, waits, blocked = M.fcntl.flock, [], []
    path = simulator.project / "runtime/core_joint_gpu.lock"
    def flock(handle, flags):
        fd = handle if isinstance(handle, int) else handle.fileno()
        if M.os.fstat(fd).st_ino == path.stat().st_ino and not blocked:
            blocked.append(True)
            raise BlockingIOError()
        return real(handle, flags)
    monkeypatch.setattr(M.fcntl, "flock", flock)
    monkeypatch.setattr(M.time, "sleep", lambda duration: waits.append(duration))
    # Create only the fixture's lock, never the real main controller's lock.
    path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    result = run(simulator, core_ids=["block_000"])
    assert result["complete"] is False and waits == [1] and len(simulator.calls) == 2


@pytest.mark.parametrize("field,value", [("ssim_valid_pixels", "bad"), ("ssim_compared_pixels", -1), ("rgb_mse", -1.)])
def test_invalid_metric_types_or_weights_are_rejected_before_aggregation(scene, field, value):
    job = scene.plan["jobs"][0]
    report = native_report(scene.plan, job, "candidate")
    report["frames"][0][field] = value
    path = scene.project / "bad/metrics.json"; put_json(path, report)
    with pytest.raises(ValueError): M.validate_report(path, scene.plan, job, "candidate")


@pytest.mark.parametrize("fault", ["empty_ply", "source_budget", "empty_split", "mask", "depth", "support", "missing_asset"])
def test_empty_scope_budget_and_missing_original_references_fail_before_any_child(scene, monkeypatch, fault):
    path = scene.project / "data/garage_surface_v1/block_027/transforms.json"
    data = json.loads(path.read_text())
    budgets = dict(M.BUDGETS)
    if fault == "empty_ply":
        candidate = scene.project / scene.candidate
        candidate.write_bytes(candidate.read_bytes().replace(b"element vertex 8", b"element vertex 0"))
    elif fault == "source_budget": budgets["max_gaussians"] = 7
    elif fault == "empty_split": data["val_filenames"] = []
    elif fault == "missing_asset": (path.parent / data["frames"][0]["mask_path"]).unlink()
    else:
        data["frames"][0].pop({"mask": "mask_path", "depth": "lidar_depth_path", "support": "lidar_depth_mask_path"}[fault])
    put_json(path, data)
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: pytest.fail("GPU child must never start"))
    with pytest.raises((ValueError, OSError)): run(scene, budgets=budgets)
    summary = json.loads((scene.project / "output/summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["complete"] is False


def test_candidate_only_has_no_implied_whole_map_baseline(simulator):
    result = run(simulator, baseline_ply=None, core_ids=["block_000"])
    assert result["paired_whole_map_comparison"] is None
    assert "baseline" not in result["models"]
    assert len(simulator.calls) == 1


def test_cli_defaults_explicit_test_and_refusal_status(monkeypatch, capsys):
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("Live core_joint_v1 controller")
    monkeypatch.setattr(M, "evaluate", evaluate)
    assert M.main(["--candidate-ply", "candidate.ply", "--output", "out"]) == 2
    assert calls[-1]["split"] == "dev" and calls[-1]["max_images"] == 0 and calls[-1]["budgets"] == M.BUDGETS
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "incomplete" and error["complete"] is False
    assert M.main(["--candidate-ply", "candidate.ply", "--output", "out", "--split", "test"]) == 2
    assert calls[-1]["split"] == "test"

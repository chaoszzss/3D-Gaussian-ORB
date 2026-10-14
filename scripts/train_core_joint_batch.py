#!/usr/bin/env python3
"""Plan/execute 28 overlapping core windows without changing the source world.

The controller is CPU only. All Torch inspection, training and rendering run in
confined children; GPU children are serialized, including across controllers.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RUN_ID = "core_joint_v2"
CORE_COUNT = 28
ITERATIONS = 15000
MAX_GAUSSIANS = 3_000_000
SUPPORT_RADIUS_M = .2
SPLITS = ("train_filenames", "val_filenames", "test_filenames")
SETTINGS = dict(iterations=ITERATIONS, depth_weight=.5, alpha_weight=.1,
                surface_splats=True, renderer_2dgs=False, renderer_inria=True,
                optimize_shared_rig=False, warm_opacity_cap=1.,
                max_gaussians=MAX_GAUSSIANS, no_refine=False, save_every=250,
                refine_every=100, stop_split_at=14000, warmup_length=500, reset_opacity=False,
                stop_screen_size_at=4000, split_screen_size=.05,
                gaussian_max_scale_m=.12, num_downscales=0, camera_optimizer_mode='off',
                tangent_scale_target_m=.03, tangent_scale_weight=.001,
                tangent_screen_weight=0., tangent_max_diameter_px=2.5,
                calibration_experimental=True, world="camera_init", units="m")


def local(value):
    path = (ROOT / Path(value).expanduser()).resolve()
    if ROOT not in path.parents:
        raise ValueError("Path must remain inside ROOT: %s" % path)
    return path


def relative(value):
    return str(local(value).relative_to(ROOT))


def within(directory, value):
    directory, path = local(directory), local(value)
    if directory not in path.parents:
        raise ValueError("Artifact escaped its owned directory: %s" % path)
    return path


def owned(value):
    """Generated destinations must not alias originals through a symlink."""
    lexical = ROOT / Path(value)
    path = local(lexical)
    if path != lexical.absolute():
        raise ValueError("Owned output path must not traverse symlinks: %s" % lexical)
    return path


def read_json(value):
    with local(value).open(encoding="utf-8") as handle:
        return json.load(handle)


def file_state(value):
    stat = local(value).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def sha256(value):
    path = local(value)
    before = file_state(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(raw)
    if file_state(path) != before:
        raise ValueError("File changed while hashing: %s" % path)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(value, record):
    path = owned(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate names also permit the two CPU preparation workers to publish.
    import threading
    temporary = owned(path.with_name(path.name + ".%s.%s.tmp" % (os.getpid(), threading.get_ident())))
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def core_ids(values=None):
    if values is None:
        return list(range(CORE_COUNT))
    result = []
    for value in values:
        for token in str(value).split(","):
            match = re.fullmatch(r"(?:(?:core|block)_)?(\d{1,3})", token)
            if not match or not 0 <= int(match[1]) < CORE_COUNT:
                raise ValueError("Core IDs must be 000..027")
            result.append(int(match[1]))
    if not result or len(set(result)) != len(result):
        raise ValueError("Core selection must be nonempty and unique")
    return sorted(result)


def neighbors(core):
    if type(core) is not int or not 0 <= core < CORE_COUNT:
        raise ValueError("Core must be an integer 0..27")
    return ["block_%03d" % i for i in range(max(0, core - 1), min(CORE_COUNT, core + 2))]


def default_python(kind):
    # System 3.8 loses its user-installed SciPy under PYTHONNOUSERSITE=1.
    if kind == "cpu":
        for path in (ROOT / ".venv-prepared/bin/python", ROOT / "runtime/.venv-prepared/bin/python"):
            if path.is_file():
                return str(path)
    return str(ROOT / ".venv-train/bin/python")


def mask_revision(value):
    if value is None:
        return None
    path = local(value)
    if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
        raise ValueError("ego-mask-revision must name an existing YAML file inside ROOT")
    import yaml
    config = yaml.safe_load(path.read_text())
    masks = {}
    for cid, camera in config["cameras"].items():
        raw = local(camera["source_mask_path"])
        masks[cid] = dict(path=relative(raw), sha256=sha256(raw))
    if set(masks) != {"camera_%d" % i for i in range(2, 6)}:
        raise ValueError("Mask revision must explicitly cover physical cameras 2..5")
    return dict(path=relative(path), sha256=sha256(path), raw_masks=masks)


def validate_revision(revision):
    if revision and mask_revision(revision["path"]) != revision:
        raise ValueError("Mask revision config or raw mask SHA changed")


def command(script, python, *args):
    # Retain the venv interpreter's lexical path, even when python is a symlink.
    return [str(python), str(local(ROOT / "scripts" / script))] + [str(a) for a in args]


def job_for(core, cpu_python, train_python, revision=None):
    cid = "core_%03d" % core
    data = owned("data/%s/%s" % (RUN_ID, cid))
    raw = owned("data/%s/%s_raw" % (RUN_ID, cid)) if revision else data
    run = owned("runs/%s/%s" % (RUN_ID, cid))
    stage = owned("runs/%s/jobs/%s" % (RUN_ID, cid))
    warm = stage / "warm_three.ply"
    window = neighbors(core)
    training = command("train_lidar_block.py", train_python, "--dataset", data, "--output", run,
        "--warm-ply", warm, "--warm-opacity-cap", "1.0", "--max-gaussians", MAX_GAUSSIANS,
        "--refine-every", SETTINGS['refine_every'], "--stop-split-at", SETTINGS['stop_split_at'],
        "--stop-screen-size-at", SETTINGS['stop_screen_size_at'], "--split-screen-size", SETTINGS['split_screen_size'],
        "--iterations", ITERATIONS, "--save-every", 250,
        "--depth-weight", ".5", "--alpha-weight", ".1", "--surface-splats",
        "--renderer-inria", "--allow-candidate-calibration")
    return dict(id=cid, core_index=core, core_ids=["block_%03d" % core],
        source_block_ids=window, dataset=relative(data), raw_dataset=relative(raw),
        ego_mask_revision=revision, run=relative(run), staging=relative(stage),
        warm_ply=relative(warm), path=relative(run / "eval_dev/gaussians.ply"),
        measured_reference=relative(data / "lidar_reference.npz"), settings=dict(SETTINGS),
        commands=dict(
            prepare=command("prepare_joint_blocks.py", cpu_python, "--block-ids", *window, "--output", raw),
            mask_revision=command("apply_ego_mask_revision.py", cpu_python, "--dataset", raw,
                "--output", data, "--revision", local(revision["path"])) if revision else None,
            filter=command("filter_gaussian_support.py", cpu_python, "--block-ids", *window,
                           "--max-distance-m", SUPPORT_RADIUS_M, "--output", stage / "filtered"),
            train=training,
            resume=command("resume_lidar_block.py", train_python, "--output", run),
            inspect_resume=command("resume_lidar_block.py", train_python, "--output", run, "--inspect"),
            evaluate_template=command("evaluate_reconstruction.py", train_python, "--config", "{config}",
                "--split", "dev", "--export-ply", "--max-images", 0, "--output", run / "eval_dev")))


def source_snapshot():
    """Small metadata plus full Gaussian/reference SHAs for ALL 28 originals."""
    manifest_path = local("maps/garage_manifest.json")
    manifest = read_json(manifest_path)
    blocks = manifest.get("blocks", [])
    by_id = {b["id"]: b for b in blocks}
    expected = {"block_%03d" % i for i in range(CORE_COUNT)}
    if len(blocks) != CORE_COUNT or set(by_id) != expected:
        raise ValueError("Original manifest must contain exactly blocks 000..027")
    files, common, sources = {relative(manifest_path): sha256(manifest_path)}, None, {}
    keys = ("world_frame", "units", "calibration", "calibration_sha256", "pose_csv", "pose_sha256",
            "source_extraction", "split_reference_images_sha256", "global_split_protocol")
    for bid in sorted(by_id):
        data = local("data/garage_surface_v1/" + bid)
        prep_path, meta_path = data / "preparation.json", data / "transforms.json"
        prep, meta = read_json(prep_path), read_json(meta_path)
        if (by_id[bid].get("training", {}).get("status") != "trained"
                or prep.get("status") != "prepared_experimental" or prep.get("block_id") != bid
                or prep.get("world_frame") != "camera_init" or prep.get("units") != "metres"
                or prep.get("global_source_split") is not True
                or prep.get("independent_test_groups") is not True
                or (data / "INCOMPLETE.json").exists()):
            raise ValueError("Require trained originals with fixed metric world/global splits: " + bid)
        provenance = {key: prep[key] for key in keys}
        if common is not None and common != provenance:
            raise ValueError("Original pose/calibration/world/split provenance differs")
        if common is None:
            common = provenance
            for key, digest_key in (("calibration", "calibration_sha256"), ("pose_csv", "pose_sha256")):
                path = local(prep[key])
                files[relative(path)] = sha256(path)
                if files[relative(path)] != prep[digest_key]:
                    raise ValueError("Original %s SHA disagrees" % key)
            index = local(Path(prep["source_extraction"]) / "indexes/images.csv")
            files[relative(index)] = sha256(index)
            if files[relative(index)] != prep["split_reference_images_sha256"]:
                raise ValueError("Original global image split reference SHA disagrees")
        if any(not meta.get(key) for key in SPLITS):
            raise ValueError("Original explicit train/dev/test membership missing")
        ply = local("runs/garage_surface_v1/%s/eval_dev/gaussians.ply" % bid)
        reference = local(data / "lidar_reference.npz")
        paths = (prep_path, meta_path, reference, ply, local(data / meta["ply_file_path"]))
        own = {relative(path): sha256(path) for path in paths}
        files.update(own)
        sources[bid] = dict(dataset=relative(data), path=relative(ply), measured_reference=relative(reference),
                            files=own, split_counts={key: len(meta[key]) for key in SPLITS})
    return dict(files=files, sources=sources, provenance=common)


def build_plan(cpu_python=None, train_python=None, revision=None):
    cpu_python, train_python = cpu_python or default_python("cpu"), train_python or default_python("train")
    plan = dict(schema_version=1, run_id=RUN_ID, planned_core_ids=list(range(CORE_COUNT)),
                planned_core_count=CORE_COUNT, world_frame="camera_init", units="m", world_scale=1.,
                camera_pose_calibration_and_global_splits="fixed original sources",
                max_cpu_prepare_workers=2, max_gpu_children=1, settings=dict(SETTINGS), ego_mask_revision=revision,
                sources=source_snapshot(), jobs=[job_for(i, cpu_python, train_python, revision) for i in range(CORE_COUNT)])
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def validate_plan(plan):
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if fingerprint(unsigned) != plan.get("plan_sha256"):
        raise ValueError("Saved plan identity/hash differs")
    if (plan.get("run_id") != RUN_ID or plan.get("settings") != SETTINGS
            or plan.get("planned_core_ids") != list(range(CORE_COUNT)) or len(plan.get("jobs", [])) != CORE_COUNT):
        raise ValueError("Saved plan is not this full 28-core contract")
    for core, job in enumerate(plan["jobs"]):
        cpu_python, train_python = job["commands"]["prepare"][0], job["commands"]["train"][0]
        if job != job_for(core, cpu_python, train_python, plan.get("ego_mask_revision")):
            raise ValueError("Saved window, path or command differs from core contract")


def validate_sources(plan, job):
    validate_revision(job.get("ego_mask_revision"))
    sources = plan["sources"]
    paths = set()
    for bid in job["source_block_ids"]:
        paths.update(sources["sources"][bid]["files"])
    for key in ("calibration", "pose_csv"):
        paths.add(relative(sources["provenance"][key]))
    paths.add(relative(Path(sources["provenance"]["source_extraction"]) / "indexes/images.csv"))
    for path in sorted(paths):
        if sha256(path) != sources["files"][path]:
            raise ValueError("Read-only original source SHA changed: " + path)


def process_identity(pid):
    try:
        directory = Path("/proc") / str(int(pid))
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        argv = (directory / "cmdline").read_bytes().split(b"\0")
        return dict(pid=int(pid), start_ticks=fields[19],
                    argv=[value.decode(errors="replace") for value in argv if value])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved, current):
    return bool(saved and current and saved.get("pid") == current.get("pid")
                and saved.get("start_ticks") == current.get("start_ticks")
                and saved.get("argv") == current.get("argv"))


def confined_child_environment(cpu=False):
    # Call this BEFORE importing subprocess, Torch, or process-pool machinery.
    from orbvi_gaussian.training import confined_environment
    env = confined_environment()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", MAX_JOBS="2")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


class FileLock:
    def __init__(self, path):
        self.path, self.handle = owned(path), None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            raise RuntimeError("A controller/child still holds %s; observe it, do not restart" % self.path)
        return self

    def __exit__(self, *unused):
        if self.handle:
            self.handle.close()


class Batch:
    def __init__(self, plan):
        import threading
        self.plan = plan
        self.base = owned("runs/" + RUN_ID)
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.batch_lock = None
        self.verified = set()  # Only evidence checked during this invocation counts.
        self.state = dict(schema_version=1, run_id=RUN_ID, plan_sha256=plan["plan_sha256"],
                          planned_core_ids=list(range(CORE_COUNT)), planned_core_count=CORE_COUNT,
                          status="planned", complete=False, controller=process_identity(os.getpid()),
                          cores={job["id"]: dict(status="planned", phase="planned", **{
                              key: job[key] for key in ("core_ids", "source_block_ids", "dataset", "run", "path", "measured_reference")})
                                 for job in plan["jobs"]})
        path = self.base / "status.json"
        if path.exists():
            saved = read_json(path)
            if saved.get("plan_sha256") != plan["plan_sha256"]:
                raise ValueError("Status belongs to a different plan")
            self.state = saved
            self.state["controller"] = process_identity(os.getpid())
            if set(self.state.get("cores", {})) != {job["id"] for job in plan["jobs"]}:
                raise ValueError("Saved status does not contain exactly the full 28-core plan")
            for core in self.state["cores"].values():
                if core["status"] == "trained_exported":
                    core["status"] = "requires_verification"
            self.state.update(complete=False, trained_exported_core_count=0, verified_core_ids=[])

    def publish(self):
        with self.mutex:
            ready = [i for i, job in enumerate(self.plan["jobs"])
                     if job["id"] in self.verified and self.state["cores"][job["id"]]["status"] == "trained_exported"]
            self.state.update(updated_epoch_s=time.time(), verified_core_ids=ready,
                trained_exported_core_count=len(ready), remaining_core_ids=[i for i in range(CORE_COUNT) if i not in ready],
                complete=len(ready) == CORE_COUNT)
            if self.state["complete"]:
                self.state["status"] = "complete"
            elif self.state["status"] == "complete":
                self.state["status"] = "incomplete"
            atomic_json(self.base / "status.json", self.state)
            # The assembler receives only verified windows here; plan.json always has all 28.
            sources = []
            for i in ready:
                core = self.state["cores"][self.plan["jobs"][i]["id"]]
                sources.append(dict(id=self.plan["jobs"][i]["id"], **{key: core[key] for key in (
                    "core_ids", "path", "measured_reference")}))
            atomic_json(self.base / "verified_sources.json", dict(run_id=RUN_ID,
                plan_sha256=self.plan["plan_sha256"], status=self.state["status"], complete=self.state["complete"],
                planned_core_ids=list(range(CORE_COUNT)), verified_core_ids=ready,
                coordinates="camera_init", units="m", sources=sources))
            assembly = self.base / "source_plan.json"
            if self.state["complete"]:
                atomic_json(assembly, dict(schema_version=1, coordinates="camera_init", units="m", sources=sources))
            elif assembly.exists():
                assembly.rename(local(assembly.with_name("source_plan.stale.%s.json" % time.time_ns())))

    def update(self, job, **values):
        with self.mutex:
            core = self.state["cores"][job["id"]]
            core.update(values, updated_epoch_s=time.time())
            if core["status"] != "trained_exported":
                self.verified.discard(job["id"])
            atomic_json(local(job["staging"]) / "status.json", core)
            self.publish()

    def child(self, job, phase, argv, gpu=False):
        env = confined_child_environment(cpu=not gpu)
        import subprocess
        if self.stop.is_set():
            raise InterruptedError("Batch stop requested")
        saved = self.state["cores"][job["id"]].get("child")
        if saved and same_process(saved, process_identity(saved["pid"])):
            raise RuntimeError("Recorded child is still alive; refusing a duplicate: %s" % saved["pid"])
        log = owned(local(job["staging"]) / (phase + ".log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock("runtime/core_joint_gpu.lock") if gpu else None
        if lock:
            lock.__enter__()
        child = None
        try:
            inherited = []
            if self.batch_lock:
                inherited.append(self.batch_lock.handle.fileno())
            if lock:
                inherited.append(lock.handle.fileno())
            with log.open("ab", buffering=0) as handle:
                child = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=tuple(inherited))
                identity = process_identity(child.pid)
                self.update(job, status="incomplete", phase=phase, child=identity,
                            child_argv=list(argv), log=relative(log), gpu_child=gpu)
                while child.poll() is None:
                    if self.stop.wait(2.):
                        raise InterruptedError("Batch stop requested")
                    checkpoints = sorted(local(job["run"]).rglob("step-*.ckpt")) if local(job["run"]).exists() else []
                    progress = {}
                    if checkpoints:
                        match = re.fullmatch(r"step-(\d{9})", checkpoints[-1].stem)
                        if match:
                            progress = dict(observed_checkpoint=relative(checkpoints[-1]),
                                            observed_checkpoint_step=int(match[1]), progress_cpu_verified=False)
                    self.update(job, **progress)
                code = child.returncode
            self.update(job, child=None, child_returncode=code, gpu_child=False)
            if code:
                raise RuntimeError("%s exited %s; retained log: %s" % (phase, code, relative(log)))
        except BaseException:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            if child is not None:
                self.update(job, child=None, child_returncode=child.returncode, gpu_child=False)
            raise
        finally:
            if lock:
                lock.__exit__()

    def job_record(self, job):
        if job != job_for(job["core_index"], job["commands"]["prepare"][0], job["commands"]["train"][0],
                          job.get("ego_mask_revision")):
            raise ValueError("Job identity differs from canonical window/path/command contract")
        path = local(job["staging"]) / "job.json"
        record = dict(job=job, plan_sha256=self.plan["plan_sha256"],
                      job_sha256=fingerprint(job), sources=self.plan["sources"])
        if path.exists():
            existing = read_json(path)
            if any(existing.get(k) != record[k] for k in record):
                raise ValueError("Saved job identity differs; refusing restart/resume")
        else:
            atomic_json(path, record)
        return path

    def inspect(self, job, refresh=False, verify_export=False):
        path = self.job_record(job)
        # Resolve and confine actual config/checkpoint paths before Torch or GPU work.
        config, checkpoint = latest_config_checkpoint(job)
        output = local(job["staging"]) / "checkpoint_inspection.json"
        if output.exists() and not refresh:
            report = read_json(output)
            states = report.get("file_states", {})
            if (report.get("job_sha256") == fingerprint(job) and states
                    and (not verify_export or report.get("export_verified") is True)
                    and all(local(p).is_file() and file_state(p) == s for p, s in states.items())):
                if report.get("config") == relative(config) and report.get("checkpoint") == relative(checkpoint):
                    return report
        argv = command("train_core_joint_batch.py", job["commands"]["train"][0],
                       "--inspect-job", path, "--inspection-output", output)
        if verify_export:
            argv.append("--verify-export")
        self.child(job, "inspect_checkpoint", argv)
        report = read_json(output)
        if report.get("job_sha256") != fingerprint(job):
            raise ValueError("CPU inspection report belongs to another job")
        return report

    def prepare(self, job):
        try:
            self.job_record(job)
            validate_sources(self.plan, job)
            raw = local(job["raw_dataset"])
            if raw.exists():
                if not (raw / "preparation.json").exists() or (raw / "INCOMPLETE.json").exists():
                    if local(job["run"]).exists():
                        raise ValueError("Interrupted training dataset is incomplete; preserve and inspect")
                    archive_partial(raw, job, "raw_dataset")
            if not raw.exists():
                self.child(job, "prepare", job["commands"]["prepare"])
            if job["ego_mask_revision"]:
                validate_prepared(job, self.plan, raw=True)
                data = local(job["dataset"])
                if data.exists() and (not (data / "preparation.json").exists() or (data / "INCOMPLETE.json").exists()):
                    if local(job["run"]).exists():
                        raise ValueError("Interrupted training mask revision is incomplete; preserve and inspect")
                    archive_partial(data, job, "masked_dataset")
                if not data.exists():
                    self.child(job, "mask_revision", job["commands"]["mask_revision"])
            prepared = validate_prepared(job, self.plan)
            filtered = local(job["staging"]) / "filtered"
            if filtered.exists() and not (filtered / "selection.json").exists():
                archive_partial(filtered, job, "filter")
            if not filtered.exists():
                self.child(job, "filter", job["commands"]["filter"])
            warm = assemble_warm(job, self.plan)
            self.update(job, status="prepared", phase="prepared", warm_count=warm["gaussians"],
                        measured_reference_points=prepared["reference_points"], warm_sha256=warm["sha256"])
        except BaseException as exc:
            self.update(job, status="incomplete", error=str(exc))
            raise

    def train_and_export(self, job):
        try:
            self.job_record(job)
            validate_sources(self.plan, job)
            validate_prepared(job, self.plan)
            warm = assemble_warm(job, self.plan)
            run = local(job["run"])
            if not run.exists():
                self.child(job, "train", job["commands"]["train"], gpu=True)
            inspection = self.inspect(job)
            result_path = run / "training_result.json"
            try:
                validate_training_result(job, inspection)
                result_valid = True
            except (ValueError, KeyError, OSError):
                result_valid = False
            if not result_valid:
                if result_path.exists():
                    archive_partial(result_path, job, "premature_training_result")
                if inspection["step"] < ITERATIONS - 1 and (run / "eval_dev").exists():
                    archive_partial(run / "eval_dev", job, "pre_final_eval")
                # Reuse the existing helper's CPU check and additional-iterations semantics.
                self.child(job, "inspect_resume", job["commands"]["inspect_resume"])
                self.child(job, "resume", job["commands"]["resume"], gpu=True)
                inspection = self.inspect(job, refresh=True)
            validate_training_result(job, inspection)
            evaluation = run / "eval_dev"
            if evaluation.exists() and not (evaluation / "metrics.json").exists():
                archive_partial(evaluation, job, "eval_dev")
            if not evaluation.exists():
                argv = [str(local(inspection["config"])) if item == "{config}" else item
                        for item in job["commands"]["evaluate_template"]]
                self.child(job, "evaluate_dev", argv, gpu=True)
            inspection = self.inspect(job, verify_export=True)
            self.record_verified(job, inspection, validate_export(job, inspection, warm["gaussians"]))
        except BaseException as exc:
            self.update(job, status="incomplete", error=str(exc))
            raise


    def record_verified(self, job, inspection, evidence):
        with self.mutex:
            self.verified.add(job["id"])
            self.update(job, status="trained_exported", phase="verified", error=None,
                path=job["path"], measured_reference=job["measured_reference"], evidence=evidence,
                training=dict(status="trained", iterations=ITERATIONS, checkpoint_step=inspection["step"],
                              checkpoint=inspection["checkpoint"], gaussians=inspection["gaussians"]),
                eval=dict(status="evaluated", split="dev", report=relative(local(job["run"]) / "eval_dev/metrics.json"),
                          images=evidence["dev_images"], exported_gaussians=evidence["exported_gaussians"]))

    def reconcile(self):
        """Recover actual finished exports even if the controller died before status."""
        for job in self.plan["jobs"]:
            if self.stop.is_set():
                raise InterruptedError("Batch stop requested")
            core = self.state["cores"][job["id"]]
            run = local(job["run"])
            has_evidence = (run / "training_result.json").exists() and (run / "eval_dev/metrics.json").exists()
            if not has_evidence and core["status"] not in ("trained_exported", "requires_verification"):
                continue
            try:
                self.job_record(job)
                validate_sources(self.plan, job)
                validate_prepared(job, self.plan)
                warm = assemble_warm(job, self.plan)
                inspection = self.inspect(job, verify_export=True)
                validate_training_result(job, inspection)
                self.record_verified(job, inspection, validate_export(job, inspection, warm["gaussians"]))
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                self.update(job, status="incomplete", error=str(exc))


def archive_partial(path, job, label):
    path = owned(path)
    allowed = [local(job[key]) for key in ("dataset", "raw_dataset", "run", "staging")]
    if not any(path == directory or directory in path.parents for directory in allowed):
        raise ValueError("Refusing to archive an original or unrelated source")
    destination = owned(local(job["staging"]) / "interrupted" / (label + "_" + str(time.time_ns())))
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)


def validate_prepared(job, plan, raw=False):
    data = local(job["raw_dataset"] if raw else job["dataset"])
    prep, meta = read_json(data / "preparation.json"), read_json(data / "transforms.json")
    if ((data / "INCOMPLETE.json").exists() or prep.get("status") != "prepared_experimental"
            or prep.get("source_block_ids") != job["source_block_ids"]
            or prep.get("world_frame") != "camera_init" or prep.get("units") != "metres"
            or prep.get("global_source_split") is not True
            or prep.get("teacher_provenance", {}).get("regenerated_all_frames") is not True):
        raise ValueError("Joint preparation/window/world/teacher provenance differs")
    for key, value in plan["sources"]["provenance"].items():
        if prep.get(key) != value:
            raise ValueError("Joint pose/calibration/global split changed: " + key)
    for name, digest in prep["generated_hashes"].items():
        if sha256(within(data, data / name)) != digest:
            raise ValueError("Joint generated asset SHA differs: " + name)
    # The preparation helper records every original image/pose/teacher byte.
    for name, digest in prep["source_hashes"].items():
        if sha256(name) != digest:
            raise ValueError("Joint source provenance changed: " + name)
        expected = plan["sources"]["files"].get(name)
        if expected is not None and expected != digest:
            raise ValueError("Prepared data came from another original source revision")
    for name, digest in prep["copied_asset_hashes"].items():
        if sha256(within(data, data / name)) != digest:
            raise ValueError("Joint copied RGB/mask changed")
    frames = {frame["file_path"]: frame for frame in meta["frames"]}
    teachers = prep["teacher_provenance"]["frames"]
    if (len(frames) != prep["frames"] or len(meta["frames"]) != len(frames) or len(teachers) != len(frames)
            or {row["file_path"] for row in teachers} != set(frames)):
        raise ValueError("Joint frame/teacher counts differ")
    for teacher in teachers:
        frame = frames[teacher["file_path"]]
        for key, sha_key in (("lidar_depth_path", "depth_sha256"), ("lidar_depth_mask_path", "support_sha256")):
            if sha256(within(data, data / frame[key])) != teacher[sha_key]:
                raise ValueError("Regenerated joint teacher bytes differ")
    original_metadata = [read_json(local(plan["sources"]["sources"][bid]["dataset"]) / "transforms.json")
                         for bid in job["source_block_ids"]]
    original_names = set()
    for source in original_metadata:
        for original in source["frames"]:
            expected_frame = dict(original)
            for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "distortion_params"):
                if key in source:
                    expected_frame[key] = source[key]
            if "depth_file_path" in expected_frame:
                expected_frame["depth_file_path"] = expected_frame["lidar_depth_path"]
            original_names.add(original["file_path"])
            if frames.get(original["file_path"]) != expected_frame:
                raise ValueError("Joint source pose/camera/timestamp/frame identity changed")
    if set(frames) != original_names:
        raise ValueError("Joint frames must include every original neighbor observation")
    for key in SPLITS:
        expected = set()
        for source in original_metadata:
            expected.update(source[key])
        if set(meta[key]) != expected or len(meta[key]) != len(expected):
            raise ValueError("Joint global image split was recomputed or truncated")
    import numpy as np
    if (meta.get("applied_scale", 1.) != 1.
            or not np.array_equal(np.asarray(meta.get("applied_transform", np.eye(4)[:3])), np.eye(4)[:3])):
        raise ValueError("Joint metadata changed world transform/scale")
    with np.load(data / "lidar_reference.npz", allow_pickle=False) as archive:
        xyz = archive["xyz"]
        if len(xyz) != prep["reference_points"] or not len(xyz) or xyz.shape != (len(xyz), 3) or not np.isfinite(xyz).all():
            raise ValueError("Measured joint NPZ actual count/coordinates differ")
    revision = job.get("ego_mask_revision")
    if revision and not raw:
        validate_revision(revision)
        parent = local(job["raw_dataset"])
        validate_prepared(job, plan, raw=True)
        info = prep.get("ego_mask_revision", {})
        if (info.get("revision_config") != revision["path"] or info.get("revision_sha256") != revision["sha256"]
                or info.get("raw_masks") != revision["raw_masks"]
                or info.get("source_joint_dataset") != job["raw_dataset"]
                or info.get("calibration_pose_rgb_splits_changed") is not False
                or info.get("parent_preparation_sha256") != sha256(parent / "preparation.json")
                or sha256(data / "transforms.json") != sha256(parent / "transforms.json")):
            raise ValueError("Masked dataset changed parent pose/split/optical provenance")
        assets = prep["revised_asset_hashes"]
        expected_assets = {frame[key] for frame in frames.values()
                           for key in ("mask_path", "lidar_depth_path", "lidar_depth_mask_path")}
        if set(assets) != expected_assets or len(info.get("frames", [])) != len(frames):
            raise ValueError("Masked dataset eligibility/teacher coverage is incomplete")
        for name, digest in assets.items():
            if sha256(within(data, data / name)) != digest:
                raise ValueError("Revised eligibility/teacher bytes changed")
        for row in info["frames"]:
            for name, digest in row["before_sha256"].items():
                if sha256(within(parent, parent / name)) != digest:
                    raise ValueError("Mask revision parent byte provenance differs")
            if any(assets.get(name) != digest for name, digest in row["after_sha256"].items()):
                raise ValueError("Mask revision current byte provenance differs")
    elif prep.get("ego_mask_revision"):
        raise ValueError("Unexpected mask revision in an unmasked job")
    return prep


def inspect_gaussian(path):
    from package_garage import inspect_ply, scan_ply
    import numpy as np
    path = local(path)
    before = file_state(path)
    with path.open("rb") as handle:
        spec = inspect_ply(handle)
    if spec["count"] > MAX_GAUSSIANS or path.stat().st_size != len(spec["header"]) + spec["count"] * spec["stride"]:
        raise ValueError("PLY exceeds count budget or has truncated/trailing records")
    if spec["sh_degree"] != 3:
        raise ValueError("Warm/trained Gaussian schema must use complete SH3")
    for prefix, width in (("f_dc_", 3), ("scale_", 3), ("rot_", 4)):
        if {name for name in spec["names"] if name.startswith(prefix)} != {prefix + str(i) for i in range(width)}:
            raise ValueError("Gaussian field family differs: " + prefix)
    def check_attributes(raw):
        rows = np.frombuffer(raw, dtype="<f4").reshape(-1, len(spec["names"]))
        quats = rows[:, [spec["names"].index("rot_%d" % i) for i in range(4)]].astype(np.float64)
        if not np.allclose(np.linalg.norm(quats, axis=1), 1., rtol=0., atol=1e-4):
            raise ValueError("PLY rotations must be unit wxyz")
        with np.errstate(over="ignore", under="ignore"):
            scales = np.exp(rows[:, [spec["names"].index("scale_%d" % i) for i in range(3)]])
        if not np.isfinite(scales).all() or not np.all(scales > 0):
            raise ValueError("PLY log scales must activate to finite positive float32 values")
    # scan_ply checks finite attributes and rotations, while preserving opaque bytes.
    record = scan_ply({"path": path, "spec": spec, "state": tuple(before)}, 8 * 1024 * 1024, full=check_attributes)
    return spec, record


def assemble_warm(job, plan):
    """Concatenate traced records byte-for-byte, with an aborting count guard."""
    import numpy as np
    from package_garage import ply_header
    filtered = local(job["staging"]) / "filtered"
    selection = read_json(filtered / "selection.json")
    reports = selection.get("sources", [])
    if selection.get("records_modified") is not False or [r["id"] for r in reports] != job["source_block_ids"]:
        raise ValueError("Filter output does not trace exactly this source window")
    inputs, fields, count = [], None, 0
    for report in reports:
        bid = report["id"]
        original = plan["sources"]["sources"][bid]
        path = local(filtered / bid / "gaussians.ply")
        spec, scanned = inspect_gaussian(path)
        if (report.get("source") != original["path"]
                or report.get("source_sha256") != original["files"][original["path"]]
                or report.get("measured_reference") != original["measured_reference"]
                or report.get("measured_reference_sha256") != original["files"][original["measured_reference"]]
                or report.get("max_distance_m") != SUPPORT_RADIUS_M
                or report.get("records_modified") is not False
                or report.get("source_domain_gate") is not None or report.get("source_ownership") is not None
                or report.get("retained_gaussians") != spec["count"]
                or report.get("output_sha256") != scanned["sha256"]):
            raise ValueError("Filtered source SHA/count/settings disagree: " + bid)
        source_path = local(original["path"])
        if sha256(source_path) != report["source_sha256"]:
            raise ValueError("Original Gaussian source changed: " + bid)
        source_spec, _ = inspect_gaussian(source_path)
        indices = np.load(local(report["source_indices"]), allow_pickle=False)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != spec["count"]
                or np.any(indices < 0) or np.any(indices >= source_spec["count"])
                or np.any(indices[1:] <= indices[:-1]) or source_spec["names"] != spec["names"]
                or report["original_gaussians"] != source_spec["count"]):
            raise ValueError("Invalid source record trace")
        original_bytes = np.memmap(source_path, mode="r", dtype=np.uint8, offset=len(source_spec["header"]),
                                   shape=(source_spec["count"], source_spec["stride"]))
        with path.open("rb") as handle:
            handle.seek(len(spec["header"]))
            for start in range(0, len(indices), 8192):
                chosen = original_bytes[indices[start:start + 8192]].tobytes()
                if handle.read(len(chosen)) != chosen:
                    raise ValueError("Filter modified original attributes/record bytes")
        del original_bytes
        if fields is not None and fields != spec["names"]:
            raise ValueError("Cannot concatenate incompatible Gaussian fields/order")
        fields = spec["names"]
        count += spec["count"]
        inputs.append(dict(path=relative(path), sha256=scanned["sha256"], count=spec["count"],
                           header_bytes=len(spec["header"]), source_sha256=report["source_sha256"]))
    if not count or count > MAX_GAUSSIANS:
        raise ValueError("Warm concatenation has %d records, limit %d; abort, never sample" % (count, MAX_GAUSSIANS))
    warm = owned(job["warm_ply"])
    manifest_path = owned(local(job["staging"]) / "warm_provenance.json")
    contract = dict(job_sha256=fingerprint(job), inputs=inputs, gaussians=count, fields=list(fields),
                    support_radius_m=SUPPORT_RADIUS_M, records_modified=False,
                    world_frame="camera_init", units="m", sampled=False)
    if warm.exists():
        manifest = read_json(manifest_path)
        if any(manifest.get(k) != v for k, v in contract.items()) or sha256(warm) != manifest["sha256"]:
            raise ValueError("Existing warm PLY/job/source identity differs")
        return manifest
    temporary = owned(warm.with_suffix(".partial"))
    if temporary.exists():
        archive_partial(temporary, job, "warm")
    payload = hashlib.sha256()
    with temporary.open("xb") as output:
        output.write(ply_header(fields, count))
        for source in inputs:
            with local(source["path"]).open("rb") as handle:
                handle.seek(source["header_bytes"])
                for raw in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    output.write(raw)
                    payload.update(raw)
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError("Filtered PLY changed while concatenating")
        output.flush()
        os.fsync(output.fileno())
    contract.update(sha256=sha256(temporary), payload_sha256=payload.hexdigest())
    # Provenance precedes final visibility so a crash cannot expose untraced warm data.
    atomic_json(manifest_path, contract)
    temporary.replace(warm)
    return contract


def latest_config_checkpoint(job):
    run = local(job["run"])
    configs = [within(run, p) for p in run.rglob("config.yml")]
    if len(configs) != 1:
        raise ValueError("Require exactly one real saved config; never start another run")
    config = configs[0]
    directory = within(config.parent, config.parent / "nerfstudio_models")
    checkpoints = [within(directory, p) for p in directory.glob("step-*.ckpt")]
    if not checkpoints or any(not re.fullmatch(r"step-\d{9}\.ckpt", p.name) for p in checkpoints):
        raise ValueError("Interrupted run has no valid saved checkpoint; preserve it for inspection")
    return config, local(max(checkpoints, key=lambda p: int(p.stem[5:])))


def expected_launch(job):
    return dict(SETTINGS, dataset=str(local(job["dataset"])), warm_ply=str(local(job["warm_ply"])))


def inspect_training(job_record, output, verify_export=False):
    """Dedicated training-Python child: validate actual config and checkpoint on CPU."""
    os.environ.update(confined_child_environment(cpu=True))
    record = read_json(job_record)
    job = record["job"]
    if (record["job_sha256"] != fingerprint(job)
            or job != job_for(job["core_index"], job["commands"]["prepare"][0], job["commands"]["train"][0],
                              job.get("ego_mask_revision"))):
        raise ValueError("Inspection job identity differs")
    within(job["staging"], output)
    within(job["staging"], job_record)
    launch_path = within(job["run"], local(job["run"]) / "launch.json")
    launch = read_json(launch_path)
    for key, expected in expected_launch(job).items():
        if launch.get(key) != expected:
            raise ValueError("Real launch settings differ: %s=%r (expected %r)" % (key, launch.get(key), expected))
    config_path, checkpoint = latest_config_checkpoint(job)
    states = {relative(p): file_state(p) for p in (launch_path, config_path, checkpoint)}
    import torch
    import yaml
    from importlib.metadata import version
    from nerfstudio.engine.trainer import TrainerConfig
    if version("nerfstudio") != "1.1.5":
        raise ValueError("Resume semantics require pinned Nerfstudio 1.1.5")
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    if not isinstance(config, TrainerConfig):
        raise ValueError("Saved config is not an actual TrainerConfig")
    manager, model = config.pipeline.datamanager, config.pipeline.model
    parser = manager.dataparser
    checks = dict(max_num_iterations=(config.max_num_iterations, ITERATIONS),
        steps_per_save=(config.steps_per_save, 250), seed=(config.machine.seed, 42),
        num_devices=(config.machine.num_devices, 1), num_machines=(config.machine.num_machines, 1),
        dataset=(local(manager.data), local(job["dataset"])), parser_dataset=(local(parser.data), local(job["dataset"])),
        output=(local(config.output_dir), local(job["run"])), base_dir=(local(config.get_base_dir()), config_path.parent),
        checkpoint_dir=(local(config.get_checkpoint_dir()), checkpoint.parent),
        load_scheduler=(config.load_scheduler, True),
        orientation=(parser.orientation_method, "none"), center=(parser.center_method, "none"),
        auto_scale=(parser.auto_scale_poses, False), scale=(parser.scale_factor, 1.),
        downscale=(parser.downscale_factor, 1), camera_optimizer=(model.camera_optimizer.mode, "off"),
        warm_ply=(local(model.gaussian_initialization_path), local(job["warm_ply"])),
        measured_reference=(local(model.lidar_reference_path), local(job["measured_reference"])),
        max_gaussians=(model.max_gaussians, MAX_GAUSSIANS), warm_opacity_cap=(model.warm_opacity_cap, 1.),
        refine_every=(model.refine_every, SETTINGS['refine_every']), stop_split_at=(model.stop_split_at, SETTINGS['stop_split_at']),
        enable_refinement=(model.enable_refinement, True), reset_opacity=(model.reset_opacity_during_refinement, False),
        scale_bound=(model.gaussian_max_scale_m, SETTINGS['gaussian_max_scale_m']),
        tangent_weight=(model.tangent_scale_loss_weight, SETTINGS['tangent_scale_weight']),
        tangent_target=(model.tangent_scale_target_m, SETTINGS['tangent_scale_target_m']),
        tangent_screen_weight=(model.tangent_screen_loss_weight, SETTINGS['tangent_screen_weight']),
        warmup_length=(model.warmup_length, SETTINGS['warmup_length']),
        stop_screen_size_at=(model.stop_screen_size_at, SETTINGS['stop_screen_size_at']),
        split_screen_size=(model.split_screen_size, SETTINGS['split_screen_size']),
        depth_weight=(model.lidar_depth_loss_weight, .5), alpha_weight=(model.lidar_alpha_loss_weight, .1),
        surface_splats=(model.surface_splats, True), renderer_inria=(model.use_inria_surfel_renderer, True),
        renderer_2dgs=(model.use_2dgs_renderer, False), optimize_rig=(model.optimize_shared_rig, False),
        random_init=(model.random_init, False), sh_degree=(model.sh_degree, 3),
        sh_interval=(model.sh_degree_interval, 1), num_downscales=(model.num_downscales, 0))
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError("Real config differs: %s=%r (expected %r)" % (name, actual, expected))
    for path in (config.load_dir, config.load_checkpoint, config.load_config):
        if path is not None:
            within(job["run"], path)
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"step", "pipeline", "optimizers", "schedulers", "scalers"}
    if not required.issubset(state):
        raise ValueError("Checkpoint lacks model/Adam/scheduler/scaler state")
    if type(state["step"]) is not int:
        raise ValueError("Checkpoint real step must be an integer")
    step = state["step"]
    if step != int(checkpoint.stem[5:]) or not 0 <= step < ITERATIONS:
        raise ValueError("Checkpoint real step differs from filename/total target")
    parameters = {name.rsplit(".", 1)[-1]: value for name, value in state["pipeline"].items()
                  if ".gauss_params." in name}
    widths = dict(means=3, features_dc=3, opacities=1, scales=3, quats=4)
    count = len(parameters["means"])
    warm = read_json(local(job["staging"]) / "warm_provenance.json")
    if not 0 < count <= MAX_GAUSSIANS:
        raise ValueError("Refined checkpoint count exceeds the actual growth budget")
    for name, width in widths.items():
        if tuple(parameters[name].shape) != (count, width):
            raise ValueError("Checkpoint Gaussian tensor shape differs: " + name)
    if tuple(parameters["features_rest"].shape) != (count, 15, 3):
        raise ValueError("Checkpoint SH3 tensor shape differs")
    groups = set(widths) | {"features_rest"}
    if set(state["optimizers"]) != groups:
        raise ValueError("Checkpoint lacks required Gaussian Adam groups")
    schedulers = {name for name in groups if config.optimizers[name]["scheduler"] is not None}
    if set(state["schedulers"]) != schedulers:
        raise ValueError("Checkpoint scheduler groups differ from enabled Gaussian config")
    for name in groups:
        adam = state["optimizers"][name]
        params = [pid for group in adam["param_groups"] for pid in group["params"]]
        if len(params) != 1 or set(adam["state"]) != set(params):
            raise ValueError("Checkpoint Adam parameter identity/state missing: " + name)
        saved = adam["state"][params[0]]
        if not 0 < float(saved["step"]) <= step + 1:
            raise ValueError("Checkpoint Adam step differs from real progress: " + name)
        for moment in ("exp_avg", "exp_avg_sq"):
            if tuple(saved[moment].shape) != tuple(parameters[name].shape):
                raise ValueError("Checkpoint Adam moment shape differs: " + name)
    tensors = 0

    def finite(value):
        nonlocal tensors
        if torch.is_tensor(value):
            tensors += 1
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), 4_000_000):
                if not torch.isfinite(flat[start:start + 4_000_000]).all().item():
                    raise ValueError("Nonfinite saved model/optimizer/scheduler tensor")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                finite(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite saved optimizer/scheduler scalar")
    finite(state)
    export = verify_checkpoint_export(job, parameters) if verify_export else None
    if verify_export:
        states[job["path"]] = export.pop("file_state")
    if any(file_state(p) != s for p, s in states.items()):
        raise ValueError("Training evidence changed during CPU inspection")
    atomic_json(output, dict(status="checkpoint_verified", job_sha256=fingerprint(job),
        config=relative(config_path), checkpoint=relative(checkpoint), checkpoint_sha256=digest,
        step=step, total_iterations=ITERATIONS, remaining_iterations=ITERATIONS - step - 1,
        gaussians=count, initial_gaussians=warm['gaussians'], gaussian_growth_ratio=count/warm['gaussians'],
        parameter_origin='trained_checkpoint', all_tensors_finite=True, tensor_count=tensors, file_states=states,
        optimizer_groups=sorted(groups), scheduler_groups=sorted(schedulers),
        export_verified=verify_export, export=export))


def verify_checkpoint_export(job, parameters):
    """Stream the evaluator's exact export convention against real saved tensors."""
    import numpy as np
    from package_garage import inspect_ply
    path = within(job["run"], local(job["path"]))
    before = file_state(path)
    fields = (["x", "y", "z", "nx", "ny", "nz"] + ["f_dc_%d" % i for i in range(3)]
              + ["f_rest_%d" % i for i in range(45)] + ["opacity"]
              + ["scale_%d" % i for i in range(3)] + ["rot_%d" % i for i in range(4)])
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        spec = inspect_ply(stream)
        if list(spec["names"]) != fields:
            raise ValueError("Exported fields/order differ from native evaluator")
        digest.update(spec["header"])
        for start in range(0, len(parameters["means"]), 8192):
            def array(name):
                return parameters[name][start:start + 8192].detach().cpu().numpy()
            means, opacity, rotations = array("means"), array("opacities"), array("quats").copy()
            rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
            rest = array("features_rest").transpose(0, 2, 1).reshape(len(means), 45)
            w,x,y,z=rotations.T
            normals=np.column_stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)))
            values = np.concatenate([means, normals, array("features_dc"), rest,
                                     opacity, array("scales"), rotations], axis=1).astype("<f4")
            values = values[(opacity[:, 0] >= -5.5373)]  # Native evaluator's documented export cull.
            raw = values.tobytes()
            if stream.read(len(raw)) != raw:
                raise ValueError("Exported PLY parameters differ from actual final checkpoint")
            digest.update(raw)
            count += len(values)
        if stream.read(1) or spec["count"] != count or file_state(path) != before:
            raise ValueError("Exported PLY actual records changed or differ from checkpoint count")
    return dict(path=relative(path), sha256=digest.hexdigest(), gaussians=count, file_state=before)


def validate_training_result(job, inspection):
    result = read_json(local(job["run"]) / "training_result.json")
    if (result.get("status") != "trained" or result.get("iterations_requested") != ITERATIONS
            or result.get("configs") != [inspection["config"]] or inspection["step"] != ITERATIONS - 1
            or inspection.get("all_tensors_finite") is not True):
        raise ValueError("Training result is not supported by final CPU-checked step %d" % (ITERATIONS-1))


def validate_export(job, inspection, warm_count):
    report_path = local(job["run"]) / "eval_dev/metrics.json"
    report = read_json(report_path)
    meta = read_json(local(job["dataset"]) / "transforms.json")
    frames = report.get("frames", [])
    expected = set(meta["val_filenames"])
    actual = [frame["file_path"] for frame in frames]
    if (report.get("status") != "evaluated" or report.get("split") != "dev"
            or local(report["config"]) != local(inspection["config"])
            or local(report["checkpoint"]) != local(inspection["checkpoint"])
            or report.get("step") != ITERATIONS - 1 or report.get("gaussians") != inspection['gaussians']
            or report.get('parameter_origin') != 'trained_checkpoint'
            or report.get('inference_model_step') != inspection['step']
            or report.get("images") != len(expected) or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError("DEV report is truncated or belongs to another run/checkpoint")
    by_name = {frame["file_path"]: frame for frame in meta["frames"]}
    for frame in frames:
        original = by_name[frame["file_path"]]
        if (frame.get("camera_id") != original["camera_id"] or frame.get("group_id") != original["group_id"]
                or any(not math.isfinite(frame[name]) for name in ("psnr_db", "ssim_valid_pixels"))):
            raise ValueError("DEV frame camera/group identity or metrics differ")
    identity = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
    transform = report.get("world_parser_transform", [])
    if (len(transform) != 3 or any(len(row) != 4 for row in transform)
            or any(not math.isfinite(v) or abs(v - identity[i][j]) > 1e-7
                   for i, row in enumerate(transform) for j, v in enumerate(row))
            or report.get("world_parser_scale") != 1.):
        raise ValueError("DEV parser must preserve camera_init world and scale 1")
    if report.get("gaussian_filter", {}).get("surface_distance_max_m", 0.) != 0. or report.get("gaussian_filter", {}).get("opacity_min", 0.) != 0.:
        raise ValueError("DEV must evaluate the complete unfiltered trained model")
    for name in ("psnr_db_mean", "ssim_valid_mean"):
        if not math.isfinite(report[name]):
            raise ValueError("DEV metrics are nonfinite")
    prep = read_json(local(job["dataset"]) / "preparation.json")
    if report.get("measured_reference_points") != prep["reference_points"]:
        raise ValueError("DEV measured reference differs from joint NPZ")
    spec, scanned = inspect_gaussian(job["path"])
    exported = report.get("export", {})
    if (exported.get("gaussians") != spec["count"] or not 0 < spec["count"] <= inspection['gaussians']
            or exported.get("coordinates") != "camera_init" or exported.get("sh_degree") != spec["sh_degree"]
            or exported.get("bbox_m") != scanned["bbox_m"]):
        raise ValueError("Actual exported PLY count/world/fields disagree with report")
    if (inspection.get("export_verified") is not True
            or inspection.get("export") != dict(path=job["path"], sha256=scanned["sha256"], gaussians=spec["count"])):
        raise ValueError("Actual PLY requires a CPU checkpoint-to-export parameter proof")
    return dict(checkpoint=inspection["checkpoint"], checkpoint_sha256=inspection["checkpoint_sha256"],
        checkpoint_step=inspection["step"], ply_sha256=scanned["sha256"], exported_gaussians=spec["count"],
        bbox_m=scanned["bbox_m"], dev_images=len(expected), report_sha256=sha256(report_path),
        measured_reference_sha256=sha256(job["measured_reference"]), world_parser_scale=1.,
        training_result_sha256=sha256(local(job["run"]) / "training_result.json"))


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Write full plan/status; start no helpers/GPU")
    mode.add_argument("--status-only", action="store_true", help="Revalidate saved completion evidence on CPU")
    parser.add_argument("--core-ids", nargs="+", help="e.g. 14,15,16; full 28-core plan is always retained")
    parser.add_argument("--batch-size", type=int, help="Maximum pending selected cores in this invocation")
    parser.add_argument("--prepare-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cpu-python", default=default_python("cpu"))
    parser.add_argument("--train-python", default=default_python("train"))
    parser.add_argument("--ego-mask-revision", type=Path,
                        help="ROOT YAML; prepare separate raw then masked joint data before training")
    parser.add_argument('--gate-report', type=Path,
                        help='Required for multi-core execution: acceptance report with overall PASS; single explicit core pilot is exempt')
    parser.add_argument('--acceptance-plan', type=Path, default=ROOT/'reports/acceptance_plan_v1.json',
                        help='Frozen external plan against which gate identity and pinned evidence are verified')
    parser.add_argument("--inspect-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inspection-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--verify-export", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_gate_report(path, expected_sha256=None, acceptance_plan=None):
    """Fail before launching a child; pin the actual PASS artifact on disk."""
    if path is None:
        raise ValueError('Multi-core execution requires --gate-report with overall PASS')
    def unique_object(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:
                raise ValueError('Duplicate key in acceptance gate report: '+key)
            obj[key]=value
        return obj
    path=local(path)
    before=file_state(path)
    report=json.loads(path.read_text(),object_pairs_hook=unique_object)
    if not isinstance(report,dict) or report.get('overall') != 'PASS':
        raise ValueError('Acceptance gate report must have overall PASS')
    plan_path=local(acceptance_plan or ROOT/'reports/acceptance_plan_v1.json')
    if not plan_path.is_file():
        raise ValueError('Acceptance gate requires an existing --acceptance-plan')
    from build_acceptance_plan import load_plan
    from acceptance_gate import require_rollout
    plan=load_plan(plan_path,verify_assets=True)
    require_rollout(report,plan['plan_id'])
    digest=sha256(path)
    if file_state(path)!=before or (expected_sha256 is not None and digest!=expected_sha256):
        raise ValueError('Acceptance gate report changed during batch execution')
    return dict(path=relative(path),sha256=digest,overall='PASS',
                acceptance_plan=relative(plan_path),acceptance_plan_id=plan['plan_id'])


def main(argv=None):
    args = parser_for_cli().parse_args(argv)
    os.environ.update(confined_child_environment())
    if args.inspect_job:
        if args.inspection_output is None:
            raise ValueError("Inspection output is required")
        inspect_training(args.inspect_job, args.inspection_output, args.verify_export)
        return 0
    selected = core_ids(args.core_ids)
    release_gate = None
    if not (args.plan_only or args.status_only):
        if args.gate_report is not None or len(selected)>1:
            release_gate=validate_gate_report(args.gate_report,acceptance_plan=args.acceptance_plan)
    revision = mask_revision(args.ego_mask_revision)
    if args.batch_size is not None and not 1 <= args.batch_size <= CORE_COUNT:
        raise ValueError("batch-size must be 1..28")
    base = owned("runs/" + RUN_ID)
    with FileLock(base / ".batch.lock") as lock:
        plan_path = base / "plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            validate_plan(plan)
            if plan.get("ego_mask_revision") != revision:
                raise ValueError("Mask revision/job identity differs from saved plan; do not mix runs")
            if (plan["jobs"][0]["commands"]["prepare"][0] != args.cpu_python
                    or plan["jobs"][0]["commands"]["train"][0] != args.train_python):
                raise ValueError("Interpreter/job identity differs from saved plan")
        else:
            plan = build_plan(args.cpu_python, args.train_python, revision)
            atomic_json(plan_path, plan)
        batch = Batch(plan)
        batch.batch_lock = lock
        batch.state.update(requested_core_ids=selected, prepare_workers=args.prepare_workers,
                           release_gate=release_gate,
                           batch_size=args.batch_size, status="planned" if args.plan_only else "incomplete")
        if args.plan_only:
            # Planning never trusts an old trained flag or claims actual completion.
            for core in batch.state["cores"].values():
                if core["status"] == "trained_exported":
                    core["status"] = "requires_verification"
            batch.publish()
            print(json.dumps(dict(status="planned", plan=relative(plan_path), planned_cores=CORE_COUNT,
                                  requested_core_ids=selected, gpu_started=False)))
            return 0
        old_handlers = {}

        def request_stop(signum, unused):
            batch.stop.set()
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        try:
            batch.reconcile()
            pending = [i for i in selected if batch.state["cores"][plan["jobs"][i]["id"]]["status"] != "trained_exported"]
            scheduled = pending[:args.batch_size] if args.batch_size else pending
            batch.state["scheduled_core_ids"] = scheduled
            batch.publish()
            if args.status_only:
                print(json.dumps(dict(status=batch.state["status"], complete=batch.state["complete"],
                                      verified_core_ids=batch.state["verified_core_ids"])))
                return 0
            from concurrent.futures import ThreadPoolExecutor
            # At most two CPU helper children; no GPU starts until all these finish.
            with ThreadPoolExecutor(max_workers=args.prepare_workers) as pool:
                futures = [pool.submit(batch.prepare, plan["jobs"][i]) for i in scheduled]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    batch.stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            for i in scheduled:
                if batch.stop.is_set():
                    raise InterruptedError("Batch stop requested")
                if release_gate is not None:
                    validate_gate_report(release_gate['path'],release_gate['sha256'],release_gate['acceptance_plan'])
                batch.train_and_export(plan["jobs"][i])
            batch.state["status"] = "complete" if batch.state["complete"] else "incomplete"
            batch.publish()
            print(json.dumps(dict(status=batch.state["status"], complete=batch.state["complete"],
                                  trained_exported_core_count=batch.state["trained_exported_core_count"],
                                  remaining_core_ids=batch.state["remaining_core_ids"])))
            return 0
        except BaseException as exc:
            batch.state.update(status="incomplete", complete=False, error=str(exc))
            batch.publish()
            print("ERROR: %s; progress/checkpoints retained in %s" % (exc, relative(base)), file=sys.stderr)
            return 130 if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or batch.stop.is_set() else 2
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)

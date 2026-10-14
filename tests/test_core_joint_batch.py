"""CPU fixtures for the real batch contracts; no model setup or GPU training."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("core_joint_batch", PROJECT / "scripts/train_core_joint_batch.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


def passing_gate_fixture(root):
    """Synthetic structurally valid gate/plan; never a reconstruction claim."""
    from build_acceptance_plan import PROTOCOLS, object_sha
    source=root/'reports/evidence.json';write_json(source,{'fixture':'CPU gate contract only'})
    plan=dict(schema_version=1,split='dev',expected_unique_frames=1,unique_frames=1,
        frames=[dict(block_id='block_015',dataset_index=0,group_id='1',camera_id='camera_3')],
        protocols={k:{} for k in PROTOCOLS},primary_protocol='ego_v2',mask_revision='ego_v2_rev2',
        jobs=[dict(block_id='block_015',frame_indices=[0])],expected_blocks=1,
        inputs={str(source):M.sha256(source)})
    plan['plan_id']=object_sha(plan)
    write_json(root/'reports/acceptance_plan_v1.json',plan)
    report=dict(overall='PASS',rollout_allowed=True,acceptance_plan_id=plan['plan_id'],
        gates={f'G{i}':dict(status='PASS',evidence={'fixture':True}) for i in range(1,9)},
        evidence_files={str(source):M.sha256(source)})
    report['report_id']=object_sha(report)
    path=root/'reports/gate.json';write_json(path,report)
    return path


def names():
    return (["x", "y", "z", "nx", "ny", "nz"] + ["f_dc_%d" % i for i in range(3)]
            + ["f_rest_%d" % i for i in range(45)] + ["opacity"]
            + ["scale_%d" % i for i in range(3)] + ["rot_%d" % i for i in range(4)] + ["confidence"])


def write_gaussians(path, values, fields=None):
    from package_garage import ply_header
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ply_header(fields or names(), len(values)) + values.astype("<f4").tobytes())


@pytest.fixture
def scene(monkeypatch):
    import orbvi_gaussian.io as io
    import orbvi_gaussian.training as training
    base = PROJECT / "runtime/tmp"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="core_batch_cpu_", dir=base) as temporary:
        root = Path(temporary)
        monkeypatch.setattr(M, "ROOT", root)
        monkeypatch.setattr(io, "PROJECT_ROOT", root)
        monkeypatch.setattr(training, "PROJECT_ROOT", root)
        # Confine cache/temp writes before importing checkpoint/Torch machinery.
        for key, value in M.confined_child_environment(cpu=True).items():
            monkeypatch.setenv(key, value)
        source = root / "data/garage_surface_v1"
        calibration = root / "configs/base.yaml"
        poses = root / "data/poses.csv"
        index = root / "data/extraction/indexes/images.csv"
        calibration.parent.mkdir(parents=True)
        calibration.write_text("world_frame: camera_init\nverified: false\n")
        poses.parent.mkdir(parents=True, exist_ok=True)
        poses.write_text("timestamp_ns,x,y,z\n1720268000000000001,100,-20,5\n")
        index.parent.mkdir(parents=True)
        index.write_text("original global train/dev/test groups\n")
        blocks = []
        for core in range(28):
            bid = "block_%03d" % core
            directory = source / bid
            for sub in ("images", "masks", "depth", "depth_masks"):
                (directory / sub).mkdir(parents=True)
            z = 5. + core * .5  # Separate floors/ramp heights remain in one metric world.
            frames = []
            for split in range(3):
                group = core * 3 + split + 1
                pose = np.diag([1., -1., -1., 1.])
                pose[:3, 3] = [100. + core, -20., z]
                stamp = 1720268000000000001 + group * 1_000_000_000
                frame = dict(file_path="images/%d_camera_3.png" % group,
                    mask_path="masks/%d_camera_3.png" % group,
                    lidar_depth_path="depth/%d_camera_3.npy" % group,
                    lidar_depth_mask_path="depth_masks/%d_camera_3.png" % group,
                    w=4, h=3, fl_x=4., fl_y=4., cx=1.5, cy=1.5,
                    camera_id="camera_3", group_id=str(group), view_id="center",
                    timestamp_ns=stamp, source_header_ns=stamp, source_bag_ns=stamp + 1,
                    transform_matrix=pose.tolist())
                frames.append(frame)
                assert cv2.imwrite(str(directory / frame["file_path"]), np.full((3, 4, 3), group, np.uint8))
                for key in ("mask_path", "lidar_depth_mask_path"):
                    assert cv2.imwrite(str(directory / frame[key]), np.full((3, 4), 255, np.uint8))
                np.save(directory / frame["lidar_depth_path"], np.full((3, 4), 2., np.float32))
            xyz = np.array([[100. + core, -20., z + 2], [100.4 + core, -20., z + 3]], np.float32)
            np.savez_compressed(directory / "lidar_reference.npz", xyz=xyz,
                                normals=np.array([[0., 0., -1.]] * 2, np.float32), normal_valid=np.ones(2, bool))
            from orbvi_gaussian.dataset import _write_ply
            with (directory / "points3D.ply").open("w") as stream:
                _write_ply(stream, xyz, np.full((2, 3), 128, np.uint8))
            meta = dict(camera_model="OPENCV", orientation_override="none", holdout_every=8,
                pixel_center_convention="half_integer", point_cloud_has_rgb=True,
                point_cloud_has_observations=False, ply_file_path="points3D.ply", point_cloud_path="points3D.ply",
                frames=frames, **{key: [frames[i]["file_path"]] for i, key in enumerate(M.SPLITS)})
            prep = dict(status="prepared_experimental", block_id=bid, calibration_verified=False,
                candidate_experiment_authorized=True, independent_test_groups=True, global_source_split=True,
                global_split_protocol="fixed original groups", split_reference_images_sha256=M.sha256(index),
                source_extraction=str(index.parents[1]), calibration=str(calibration),
                calibration_sha256=M.sha256(calibration), calibration_id="original",
                pose_csv=str(poses), pose_sha256=M.sha256(poses), pose_source="external_csv",
                world_frame="camera_init", body_frame="aft_mapped", units="metres",
                lidar_reference_path="lidar_reference.npz",
                map=dict(source="historical-map-only", color_source="calibrated_training_images_only",
                    has_measured_rgb=True, alignment_verified_by_operator=True, world_frame="camera_init",
                    colorized_points=2, input_has_rgb_field=True, map_rgb_verified_by_operator=False),
                export=dict(point_count=2, paths=dict(point_cloud=str(directory / "points3D.ply"))))
            write_json(directory / "transforms.json", meta)
            write_json(directory / "preparation.json", prep)
            values = np.zeros((4, len(names())), dtype="<f4")
            values[:, :3] = np.repeat(xyz, 2, axis=0)
            values[:, 6:54] = np.arange(48, dtype=np.float32) * .001
            values[:, names().index("opacity")] = 2.
            values[:, [names().index("scale_%d" % i) for i in range(3)]] = [-2., -3., -4.]
            values[:, names().index("rot_0")] = 1.
            values[:, names().index("confidence")] = [-0., .2, .3, .4]
            write_gaussians(root / "runs/garage_surface_v1" / bid / "eval_dev/gaussians.ply", values)
            blocks.append(dict(id=bid, training=dict(status="trained"), coordinates="camera_init", units="m"))
        write_json(root / "maps/garage_manifest.json", dict(schema_version=1, coordinates="camera_init", units="m", blocks=blocks))
        yield root


def plan():
    return M.build_plan(str(PROJECT / ".venv-train/bin/python"), str(PROJECT / ".venv-train/bin/python"))


def prepare_fixture(job, saved_plan):
    spec = importlib.util.spec_from_file_location("prepare_core_fixture", PROJECT / "scripts/prepare_joint_blocks.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = M.ROOT
    helper.prepare_joint(job["source_block_ids"], M.local(job["raw_dataset"]),
                         source_root=M.ROOT / "data/garage_surface_v1")


def filtered_fixture(job, saved_plan):
    from package_garage import inspect_ply
    root = M.local(job["staging"]) / "filtered"
    reports = []
    for bid in job["source_block_ids"]:
        original = saved_plan["sources"]["sources"][bid]
        path = M.local(original["path"])
        with path.open("rb") as stream:
            spec = inspect_ply(stream)
            values = np.frombuffer(stream.read(), dtype="<f4").reshape(spec["count"], -1)
        indices = np.array([0, 2], np.int64)
        out = root / bid / "gaussians.ply"
        write_gaussians(out, values[indices], list(spec["names"]))
        trace = out.parent / "source_record_indices.npy"
        np.save(trace, indices, allow_pickle=False)
        reports.append(dict(id=bid, source=original["path"], source_sha256=M.sha256(path),
            measured_reference=original["measured_reference"], measured_reference_sha256=M.sha256(original["measured_reference"]),
            max_distance_m=.2, records_modified=False, retained_gaussians=len(indices), original_gaussians=len(values),
            source_domain_gate=None, source_ownership=None, source_indices=M.relative(trace), output_sha256=M.sha256(out)))
    write_json(root / "selection.json", dict(records_modified=False, sources=reports))


def warm_fixture(job, saved_plan):
    prepare_fixture(job, saved_plan)
    filtered_fixture(job, saved_plan)
    return M.assemble_warm(job, saved_plan)


def checkpoint_fixture(job, batch, step=None):
    if step is None: step=M.ITERATIONS-1
    torch = pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("nerfstudio")
    from nerfstudio.configs.method_configs import method_configs
    from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModelConfig, LidarFullImageDatamanagerConfig
    c = copy.deepcopy(method_configs["splatfacto"])
    c.pipeline.model = LidarSplatfactoModelConfig()
    c.pipeline.datamanager = LidarFullImageDatamanagerConfig()
    m, manager = c.pipeline.model, c.pipeline.datamanager
    parser = manager.dataparser
    manager.data = parser.data = M.local(job["dataset"])
    parser.orientation_method = parser.center_method = "none"
    parser.auto_scale_poses = False
    parser.scale_factor = 1.
    parser.downscale_factor = 1
    m.camera_optimizer.mode = "off"
    m.gaussian_initialization_path = M.local(job["warm_ply"])
    m.lidar_reference_path = M.local(job["measured_reference"])
    m.max_gaussians = M.MAX_GAUSSIANS
    m.warm_opacity_cap = 1.
    m.refine_every = 100
    m.stop_split_at = 14000
    m.reset_opacity_during_refinement = False
    m.gaussian_max_scale_m = .12
    m.tangent_scale_loss_weight = .001
    m.lidar_depth_loss_weight = .5
    m.lidar_alpha_loss_weight = .1
    m.surface_splats = m.use_inria_surfel_renderer = True
    m.sh_degree_interval = 1
    m.num_downscales = 0
    c.output_dir = M.local(job["run"])
    c.experiment_name = job["id"]
    c.timestamp = "2026-10-07_000000"
    c.max_num_iterations = M.ITERATIONS
    c.steps_per_save = 250
    c.machine.seed = 42
    config = c.get_base_dir() / "config.yml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.dump(c))
    write_json(M.local(job["run"]) / "launch.json", M.expected_launch(job))
    count = M.read_json(M.local(job["staging"]) / "warm_provenance.json")["gaussians"]
    params = {name: torch.ones((count, width), dtype=torch.float32)
              for name, width in dict(means=3, features_dc=3, opacities=1, scales=3, quats=4).items()}
    params["features_rest"] = torch.ones((count, 15, 3), dtype=torch.float32)
    state = dict(step=step, pipeline={"_model.gauss_params." + name: tensor for name, tensor in params.items()},
                 optimizers={}, schedulers={}, scalers={})
    for name, tensor in params.items():
        state["optimizers"][name] = dict(state={0: dict(step=torch.tensor(float(step + 1)),
            exp_avg=torch.zeros_like(tensor), exp_avg_sq=torch.zeros_like(tensor))}, param_groups=[dict(params=[0], lr=.001)])
        if c.optimizers[name]["scheduler"] is not None:
            state["schedulers"][name] = dict(last_epoch=step + 1, _last_lr=[.001])
    checkpoint = c.get_checkpoint_dir() / ("step-%09d.ckpt" % step)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint)
    job_record = batch.job_record(job)
    output = M.local(job["staging"]) / "checkpoint_inspection.json"
    M.inspect_training(job_record, output)
    return c, state, M.read_json(output), checkpoint


def result_fixture(job, inspection):
    write_json(M.local(job["run"]) / "training_result.json", dict(status="trained",
               iterations_requested=M.ITERATIONS, configs=[inspection["config"]]))


def evaluation_fixture(job, inspection):
    import torch
    state = torch.load(M.local(inspection["checkpoint"]), map_location="cpu", weights_only=False)
    params = {name.rsplit(".", 1)[-1]: value.numpy() for name, value in state["pipeline"].items()}
    rotations = params["quats"].copy()
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    means = params["means"]
    w,x,y,z=rotations.T
    normals=np.column_stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)))
    values = np.concatenate([means, normals, params["features_dc"],
        params["features_rest"].transpose(0, 2, 1).reshape(len(means), 45),
        params["opacities"], params["scales"], rotations], axis=1).astype("<f4")
    values = values[params["opacities"][:, 0] >= -5.5373]
    target = M.local(job["path"])
    write_gaussians(target, values, names()[:-1])
    spec, scanned = M.inspect_gaussian(target)
    meta = M.read_json(M.local(job["dataset"]) / "transforms.json")
    frames = [dict(file_path=f["file_path"], group_id=f["group_id"], camera_id=f["camera_id"], psnr_db=20.,
                   ssim_valid_pixels=.8) for f in meta["frames"] if f["file_path"] in meta["val_filenames"]]
    report = dict(status="evaluated", split="dev", config=str(M.local(inspection["config"])),
        checkpoint=str(M.local(inspection["checkpoint"])), step=M.ITERATIONS-1, gaussians=inspection["gaussians"],
        parameter_origin="trained_checkpoint", inference_model_step=M.ITERATIONS-1,
        images=len(frames), frames=frames, world_parser_transform=np.eye(4)[:3].tolist(), world_parser_scale=1.,
        psnr_db_mean=20., ssim_valid_mean=.8,
        measured_reference_points=M.read_json(M.local(job["dataset"]) / "preparation.json")["reference_points"],
        export=dict(gaussians=spec["count"], coordinates="camera_init", sh_degree=3, bbox_m=scanned["bbox_m"]))
    write_json(target.parent / "metrics.json", report)
    return report


def final_inspection(job, batch):
    output = M.local(job["staging"]) / "checkpoint_inspection.json"
    M.inspect_training(batch.job_record(job), output, verify_export=True)
    return M.read_json(output)


def test_full_plan_has_all_ids_edge_windows_and_correct_commands(scene):
    p = plan()
    M.validate_plan(p)
    assert p["planned_core_ids"] == list(range(28))
    assert p["jobs"][0]["source_block_ids"] == ["block_000", "block_001"]
    assert p["jobs"][27]["source_block_ids"] == ["block_026", "block_027"]
    assert {bid for job in p["jobs"] for bid in job["core_ids"]} == {"block_%03d" % i for i in range(28)}
    for core, job in enumerate(p["jobs"]):
        assert job["source_block_ids"] == M.neighbors(core)
        assert job["core_ids"] == ["block_%03d" % core]
        for argv in job["commands"].values():
            if argv is not None:
                assert isinstance(argv, list) and all(isinstance(value, str) for value in argv)
        train = job["commands"]["train"]
        assert train[train.index("--max-gaussians") + 1] == "3000000"
        assert train[train.index("--iterations") + 1] == "15000"
        assert train[train.index("--save-every") + 1] == "250"
        assert "--no-refine" not in train
        assert train[train.index("--refine-every")+1] == "100"
        assert train[train.index("--stop-split-at")+1] == "14000"
        assert scene in M.local(job["path"]).parents  # Also works with system Python 3.8.
        assert job["measured_reference"] == job["dataset"] + "/lidar_reference.npz"


def test_plan_only_subset_starts_no_children_and_preserves_28(scene, monkeypatch):
    monkeypatch.setattr(M.Batch, "child", lambda *a, **k: pytest.fail("dry plan started a child"))
    assert M.main(["--plan-only", "--core-ids", "14,15,16", "--batch-size", "2"]) == 0
    saved = M.read_json(scene / "runs/core_joint_v2/plan.json")
    state = M.read_json(scene / "runs/core_joint_v2/status.json")
    assert len(saved["jobs"]) == 28 and state["requested_core_ids"] == [14, 15, 16]
    assert state["complete"] is False and state["trained_exported_core_count"] == 0
    assert not (scene / "runs/core_joint_v2/source_plan.json").exists()
    assert not (scene / "data/core_joint_v2").exists()


@pytest.mark.parametrize("values", [["28"], ["-1"], ["14", "014"], ["../14"], ["14,,15"]])
def test_bad_core_selection(values):
    with pytest.raises(ValueError):
        M.core_ids(values)


def test_owned_paths_reject_symlink_alias_to_originals(scene):
    with pytest.raises(ValueError):
        M.local(scene.parent / "outside")
    (scene / "data/core_joint_v2").symlink_to(scene / "data/garage_surface_v1", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        M.job_for(0, "cpu", "train")


def test_atomic_state_cannot_follow_symlink_into_read_only_original(scene):
    original = scene / "data/garage_surface_v1/block_000/preparation.json"
    before = original.read_bytes()
    alias = scene / "runs/core_joint_v2/status.json"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="symlinks"):
        M.atomic_json(alias, dict(status="fixture overwrite forbidden"))
    assert original.read_bytes() == before


def test_source_snapshot_guard_detects_original_revision(scene):
    p = plan()
    path = M.local(p["sources"]["sources"]["block_014"]["path"])
    path.write_bytes(path.read_bytes() + b"altered fixture")
    with pytest.raises(ValueError, match="original source SHA"):
        M.validate_sources(p, p["jobs"][15])


def test_prepared_frames_cannot_change_original_pose_even_with_rehashed_metadata(scene):
    p = plan()
    job = p["jobs"][15]
    prepare_fixture(job, p)
    path = M.local(job["dataset"]) / "transforms.json"
    changed = M.read_json(path)
    changed["frames"][0]["transform_matrix"][2][3] += 1.
    write_json(path, changed)
    prep_path = path.parent / "preparation.json"
    prep = M.read_json(prep_path)
    prep["generated_hashes"]["transforms.json"] = M.sha256(path)
    write_json(prep_path, prep)
    with pytest.raises(ValueError, match="source pose"):
        M.validate_prepared(job, p)


def test_warm_concat_preserves_all_attributes_and_signed_zero(scene):
    p = plan()
    job = p["jobs"][15]
    before = {name: M.sha256(name) for name in p["sources"]["files"]}
    warm = warm_fixture(job, p)
    assert warm["gaussians"] == 6 and warm["sampled"] is False
    from package_garage import inspect_ply
    with M.local(job["warm_ply"]).open("rb") as stream:
        spec = inspect_ply(stream)
        payload = stream.read()
    expected = b""
    for bid in job["source_block_ids"]:
        with (M.local(job["staging"]) / "filtered" / bid / "gaussians.ply").open("rb") as stream:
            inspect_ply(stream)
            expected += stream.read()
    assert payload == expected and list(spec["names"]) == names()
    assert np.signbit(np.frombuffer(payload, "<f4").reshape(6, -1)[0, -1])
    assert before == {name: M.sha256(name) for name in p["sources"]["files"]}
    M.validate_prepared(job, p)
    assert M.assemble_warm(job, p) == warm


def test_warm_oversize_aborts_without_sampling_or_output(scene, monkeypatch):
    p = plan()
    job = p["jobs"][15]
    prepare_fixture(job, p)
    filtered_fixture(job, p)
    monkeypatch.setattr(M, "MAX_GAUSSIANS", 5)
    with pytest.raises(ValueError, match="abort, never sample"):
        M.assemble_warm(job, p)
    assert not M.local(job["warm_ply"]).exists()


def test_filter_trace_and_output_hash_guard(scene):
    p = plan()
    job = p["jobs"][15]
    prepare_fixture(job, p)
    filtered_fixture(job, p)
    trace = M.local(job["staging"]) / "filtered/block_014/source_record_indices.npy"
    np.save(trace, np.array([1, 2], np.int64), allow_pickle=False)
    # First selected record differs in its exact confidence bits.
    with pytest.raises(ValueError, match="record bytes"):
        M.assemble_warm(job, p)


def test_cpu_config_checkpoint_step_and_real_scheduler_groups(scene):
    p = plan()
    job = p["jobs"][15]
    warm_fixture(job, p)
    batch = M.Batch(p)
    _, _, inspection, _ = checkpoint_fixture(job, batch, 749)
    assert inspection["remaining_iterations"] == M.ITERATIONS-750
    assert inspection["scheduler_groups"] == ["means"]  # quats has no configured scheduler.
    assert inspection["all_tensors_finite"] is True
    result_fixture(job, inspection)
    with pytest.raises(ValueError, match="step 14999"):
        M.validate_training_result(job, inspection)


@pytest.mark.parametrize("corruption", ["step", "optimizer_nan", "missing_scheduler", "config_world", "config_target"])
def test_checkpoint_rejects_real_corruption(scene, corruption):
    p = plan()
    job = p["jobs"][15]
    warm_fixture(job, p)
    batch = M.Batch(p)
    config, state, inspection, path = checkpoint_fixture(job, batch)
    import torch
    import yaml
    if corruption == "step":
        state["step"] = 998
    elif corruption == "optimizer_nan":
        state["optimizers"]["means"]["state"][0]["exp_avg"][0, 0] = float("nan")
    elif corruption == "missing_scheduler":
        state["schedulers"].clear()
    elif corruption == "config_world":
        config.pipeline.datamanager.dataparser.auto_scale_poses = True
    else:
        config.max_num_iterations = 1001
    torch.save(state, path)
    M.local(inspection["config"]).write_text(yaml.dump(config))
    with pytest.raises(ValueError):
        M.inspect_training(batch.job_record(job), M.local(job["staging"]) / "new_inspection.json")


def test_checkpoint_symlink_escape_is_rejected_before_torch(scene, monkeypatch):
    p = plan()
    job = p["jobs"][15]
    run = M.local(job["run"])
    folder = run / "one/splatfacto/timestamp"
    folder.mkdir(parents=True)
    (folder / "config.yml").write_text("fixture config")
    target = scene / "runs/garage_surface_v1/block_015"
    (folder / "nerfstudio_models").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="escaped"):
        M.latest_config_checkpoint(job)


def test_final_export_requires_full_dev_actual_count_and_scale_one(scene):
    p = plan()
    job = p["jobs"][15]
    warm = warm_fixture(job, p)
    batch = M.Batch(p)
    _, _, inspection, _ = checkpoint_fixture(job, batch)
    result_fixture(job, inspection)
    report = evaluation_fixture(job, inspection)
    inspection = final_inspection(job, batch)
    M.validate_training_result(job, inspection)
    evidence = M.validate_export(job, inspection, warm["gaussians"])
    batch.record_verified(job, inspection, evidence)
    assert batch.state["verified_core_ids"] == [15] and batch.state["complete"] is False
    assert not (batch.base / "source_plan.json").exists()
    assert M.read_json(batch.base / "verified_sources.json")["sources"][0]["core_ids"] == ["block_015"]
    for bad in (dict(world_parser_scale=.5), dict(images=1, frames=report["frames"][:1]),
                dict(export=dict(report["export"], gaussians=999))):
        write_json(M.local(job["path"]).parent / "metrics.json", dict(report, **bad))
        with pytest.raises(ValueError):
            M.validate_export(job, inspection, warm["gaussians"])


def test_saved_trained_flags_cannot_fake_28_complete(scene):
    p = plan()
    batch = M.Batch(p)
    for value in batch.state["cores"].values():
        value["status"] = "trained_exported"
    batch.publish()
    assert batch.state["complete"] is False and batch.state["trained_exported_core_count"] == 0
    restored = M.Batch(p)
    restored.reconcile()
    assert all(c["status"] == "incomplete" for c in restored.state["cores"].values())
    assert not (batch.base / "source_plan.json").exists()


def test_resume_never_starts_fresh_training_or_changes_job_identity(scene, monkeypatch):
    p = plan()
    job = p["jobs"][15]
    warm_fixture(job, p)
    batch = M.Batch(p)
    _, _, inspection, _ = checkpoint_fixture(job, batch, 749)
    result_fixture(job, inspection)  # A premature helper marker must be archived.
    calls = []
    def child(target, phase, argv, gpu=False):
        calls.append((phase, gpu))
        if phase == "resume":
            _, _, final, _ = checkpoint_fixture(target, batch, M.ITERATIONS-1)
            result_fixture(target, final)
        elif phase == "evaluate_dev":
            evaluation_fixture(target, M.read_json(M.local(target["staging"]) / "checkpoint_inspection.json"))
        elif phase == "inspect_checkpoint":
            M.inspect_training(batch.job_record(target), M.local(target["staging"]) / "checkpoint_inspection.json",
                               verify_export="--verify-export" in argv)
    monkeypatch.setattr(batch, "child", child)
    batch.train_and_export(job)
    assert [phase for phase, gpu in calls if gpu] == ["resume", "evaluate_dev"]
    assert "train" not in [phase for phase, _ in calls]
    assert batch.state["complete"] is False and batch.state["verified_core_ids"] == [15]
    assert list((M.local(job["staging"]) / "interrupted").glob("premature_training_result_*"))
    changed = copy.deepcopy(job)
    changed["source_block_ids"] = ["block_013", "block_014", "block_015"]
    with pytest.raises(ValueError, match="[Jj]ob identity"):
        batch.job_record(changed)


def test_no_checkpoint_does_not_restart_existing_output(scene, monkeypatch):
    p = plan()
    job = p["jobs"][15]
    warm_fixture(job, p)
    M.local(job["run"]).mkdir()
    batch = M.Batch(p)
    monkeypatch.setattr(batch, "child", lambda *a, **k: pytest.fail("started a child for missing checkpoint"))
    with pytest.raises(ValueError, match="real saved config"):
        batch.train_and_export(job)
    assert batch.state["cores"][job["id"]]["status"] == "incomplete"


def test_pid_reuse_and_live_child_do_not_trigger_duplicate(scene, monkeypatch):
    p = plan()
    batch = M.Batch(p)
    job = p["jobs"][15]
    identity = dict(pid=123, start_ticks="456", argv=["train", "--output", job["run"]])
    assert M.same_process(identity, dict(identity))
    assert not M.same_process(identity, dict(identity, start_ticks="789"))
    batch.state["cores"][job["id"]]["child"] = identity
    monkeypatch.setattr(M, "process_identity", lambda pid: identity)
    with pytest.raises(RuntimeError, match="still alive"):
        batch.child(job, "train", ["does-not-exist"], gpu=True)


def test_locks_reject_parallel_gpu_children_and_keep_checkpoint_on_stop(scene, monkeypatch):
    p = plan()
    job = p["jobs"][15]
    batch = M.Batch(p)
    with M.FileLock("runtime/core_joint_gpu.lock"):
        with pytest.raises(RuntimeError, match="still holds"):
            batch.child(job, "train", ["never-launch"], gpu=True)
    M.local(job["run"]).mkdir()
    kept = M.local(job["run"]) / "saved.ckpt"
    kept.write_bytes(b"saved fixture checkpoint")
    batch.stop.set()
    with pytest.raises(InterruptedError):
        batch.child(job, "train", ["never-launch"], gpu=True)
    assert kept.read_bytes() == b"saved fixture checkpoint"


def test_constructor_and_total_iteration_semantics_use_actual_helper(scene):
    pytest.importorskip("nerfstudio")
    import inspect
    from nerfstudio.engine.trainer import Trainer
    helper_spec = importlib.util.spec_from_file_location("resume_core_fixture", PROJECT / "scripts/resume_lidar_block.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    assert list(inspect.signature(Trainer.__init__).parameters) == ["self", "config", "local_rank", "world_size"]
    assert [helper.remaining_iterations(step, 1000) for step in (0, 249, 749, 999)] == [999, 750, 250, 0]
    assert "self._start_step + num_iterations" in inspect.getsource(Trainer.train)


def revision_fixture():
    raw_masks = {}
    for cid in ("camera_2", "camera_3", "camera_4", "camera_5"):
        path = M.ROOT / "configs/masks" / (cid + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.full((3, 4), 255, np.uint8))
        raw_masks[cid] = dict(source_mask_path=str(path))
    path = M.ROOT / "configs/revision.yaml"
    import yaml
    path.write_text(yaml.safe_dump(dict(cameras=raw_masks)))
    return M.mask_revision(path)


def masked_fixture(job):
    import shutil
    raw, data = M.local(job["raw_dataset"]), M.local(job["dataset"])
    shutil.copytree(raw, data)
    prep = M.read_json(data / "preparation.json")
    meta = M.read_json(data / "transforms.json")
    assets, records = {}, []
    for frame in meta["frames"]:
        planes = [frame[k] for k in ("mask_path", "lidar_depth_path", "lidar_depth_mask_path")]
        before = {name: M.sha256(raw / name) for name in planes}
        mask = cv2.imread(str(data / frame["mask_path"]), 0)
        mask[0, 0] = 0
        cv2.imwrite(str(data / frame["mask_path"]), mask)
        depth = np.load(data / frame["lidar_depth_path"], allow_pickle=False)
        depth[0, 0] = 0
        np.save(data / frame["lidar_depth_path"], depth)
        support = cv2.imread(str(data / frame["lidar_depth_mask_path"]), 0)
        support[0, 0] = 0
        cv2.imwrite(str(data / frame["lidar_depth_mask_path"]), support)
        after = {name: M.sha256(data / name) for name in planes}
        assets.update(after)
        records.append(dict(file_path=frame["file_path"], before_sha256=before, after_sha256=after))
        prep["copied_asset_hashes"].pop(frame["mask_path"], None)
    prep["block_id"] = data.name
    revision = job["ego_mask_revision"]
    prep["ego_mask_revision"] = dict(revision_config=revision["path"], revision_sha256=revision["sha256"],
        raw_masks=revision["raw_masks"], source_joint_dataset=job["raw_dataset"],
        calibration_pose_rgb_splits_changed=False, parent_preparation_sha256=M.sha256(raw / "preparation.json"), frames=records)
    prep["revised_asset_hashes"] = assets
    by_name = {f["file_path"]: f for f in meta["frames"]}
    for row in prep["teacher_provenance"]["frames"]:
        frame = by_name[row["file_path"]]
        row["depth_sha256"] = assets[frame["lidar_depth_path"]]
        row["support_sha256"] = assets[frame["lidar_depth_mask_path"]]
    write_json(data / "preparation.json", prep)


def test_optional_mask_step_is_separate_after_raw_prepare_and_fixes_pixel_hashes(scene, monkeypatch):
    revision = revision_fixture()
    p = M.build_plan("cpu", "train", revision)
    job = p["jobs"][15]
    batch = M.Batch(p)
    calls = []
    def child(target, phase, argv, gpu=False):
        assert not gpu
        calls.append(phase)
        if phase == "prepare":
            prepare_fixture(target, p)
        elif phase == "mask_revision":
            masked_fixture(target)
        elif phase == "filter":
            filtered_fixture(target, p)
    monkeypatch.setattr(batch, "child", child)
    batch.prepare(job)
    assert calls == ["prepare", "mask_revision", "filter"]
    assert job["raw_dataset"] != job["dataset"]
    assert M.sha256(M.local(job["dataset"]) / "transforms.json") == M.sha256(M.local(job["raw_dataset"]) / "transforms.json")
    M.validate_prepared(job, p)
    path = M.local(revision["raw_masks"]["camera_3"]["path"])
    path.write_bytes(path.read_bytes() + b"modified fixture mask")
    with pytest.raises(ValueError, match="revision config or raw mask"):
        M.validate_sources(p, job)


def test_plan_mask_identity_change_is_refused(scene):
    assert M.main(["--plan-only"]) == 0
    revision = revision_fixture()
    with pytest.raises(ValueError, match="Mask revision/job identity"):
        M.main(["--plan-only", "--ego-mask-revision", revision["path"]])


def test_prepare_worker_limit_and_bounded_selection_keep_full_plan(scene, monkeypatch):
    prepared, trained = [], []
    monkeypatch.setattr(M.Batch, "prepare", lambda batch, job: prepared.append(job["core_index"]))
    monkeypatch.setattr(M.Batch, "train_and_export", lambda batch, job: trained.append(job["core_index"]))
    gate = passing_gate_fixture(scene)
    assert M.main(["--core-ids", "14,15,16", "--batch-size", "2", "--prepare-workers", "2", "--gate-report", str(gate)]) == 0
    state = M.read_json(scene / "runs/core_joint_v2/status.json")
    assert sorted(prepared) == trained == [14, 15]
    assert state["scheduled_core_ids"] == [14, 15] and state["planned_core_count"] == 28
    assert state["complete"] is False
    with pytest.raises(SystemExit):
        M.parser_for_cli().parse_args(["--prepare-workers", "3"])


def test_all_28_verified_exports_publish_assembler_schema_and_reconcile_invalidated_one(scene, monkeypatch):
    p = plan()
    batch = M.Batch(p)
    for job in p["jobs"]:
        warm = warm_fixture(job, p)
        _, _, inspection, _ = checkpoint_fixture(job, batch)
        result_fixture(job, inspection)
        evaluation_fixture(job, inspection)
        inspection = final_inspection(job, batch)
        batch.record_verified(job, inspection, M.validate_export(job, inspection, warm["gaussians"]))
    assert batch.state["complete"] is True and batch.state["trained_exported_core_count"] == 28
    source_plan = M.read_json(batch.base / "source_plan.json")
    assert set(source_plan) == {"schema_version", "coordinates", "units", "sources"}
    assert len(source_plan["sources"]) == 28
    assert {s["core_ids"][0] for s in source_plan["sources"]} == {"block_%03d" % i for i in range(28)}
    assert all(set(s) == {"id", "path", "core_ids", "measured_reference"} for s in source_plan["sources"])
    # Recover from interrupted status without rerunning any training or export.
    restored = M.Batch(p)
    monkeypatch.setattr(restored, "child", lambda *a, **k: pytest.fail("unchanged complete evidence started a child"))
    restored.reconcile()
    assert restored.state["complete"] is True
    report_path = M.local(p["jobs"][0]["run"]) / "eval_dev/metrics.json"
    report = M.read_json(report_path)
    write_json(report_path, dict(report, images=0, frames=[]))
    restored.reconcile()
    assert restored.state["complete"] is False and restored.state["verified_core_ids"] == list(range(1, 28))
    assert not (batch.base / "source_plan.json").exists()
    assert list(batch.base.glob("source_plan.stale.*.json"))


def test_checkpoint_to_export_proof_rejects_wrong_parameter_bytes(scene):
    p = plan()
    job = p["jobs"][15]
    warm_fixture(job, p)
    batch = M.Batch(p)
    _, _, inspection, _ = checkpoint_fixture(job, batch)
    result_fixture(job, inspection)
    evaluation_fixture(job, inspection)
    from package_garage import inspect_ply
    path = M.local(job["path"])
    with path.open("r+b") as stream:
        spec = inspect_ply(stream)
        # A finite but different XYZ passes schema/count guards, fails parameter origin.
        stream.seek(len(spec["header"]))
        stream.write(np.asarray([123.], "<f4").tobytes())
    with pytest.raises(ValueError, match="parameters differ"):
        final_inspection(job, batch)


def test_cpu_environment_and_interpreter_fallback_are_confined(scene):
    env = M.confined_child_environment(cpu=True)
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTHONNOUSERSITE"] == "1"
    assert env["MAX_JOBS"] == "2" and env["OMP_NUM_THREADS"] == "2"
    for key in ("TMPDIR", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH"):
        assert scene in Path(env[key]).parents
    assert M.default_python("cpu") == str(scene / ".venv-train/bin/python")
    prepared = scene / ".venv-prepared/bin/python"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("synthetic interpreter fixture")
    assert M.default_python("cpu") == str(prepared)


def test_real_cpu_child_holds_global_gpu_lock_so_second_thread_cannot_launch(scene):
    import threading
    import time
    p = plan()
    batch = M.Batch(p)
    failures = []
    def first():
        try:
            # GPU scheduling flag exercised with a harmless CPU interpreter child.
            batch.child(p["jobs"][14], "lock_fixture", [sys.executable, "-c", "import time;time.sleep(.5)"], gpu=True)
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=first)
    thread.start()
    deadline = time.monotonic() + 5
    while not batch.state["cores"]["core_014"].get("child") and time.monotonic() < deadline:
        time.sleep(.005)
    try:
        assert batch.state["cores"]["core_014"].get("child")
        with pytest.raises(RuntimeError, match="still holds"):
            batch.child(p["jobs"][15], "lock_fixture", ["never-launch"], gpu=True)
    finally:
        thread.join(5)
    assert not thread.is_alive() and not failures


@pytest.mark.parametrize('report', [None, {}, {'overall':'FAIL'}, {'overall':True}, {'overall':'pass'}, {'overall':'PASS'}, []])
def test_multicore_gate_fails_closed_before_any_child_or_plan(scene,monkeypatch,report):
    monkeypatch.setattr(M.Batch,'child',lambda *a,**k: pytest.fail('gate failure launched a child'))
    argv=['--core-ids','14,15']
    if report is not None:
        path=scene/'reports/gate.json';write_json(path,report)
        argv+=['--gate-report',str(path)]
    with pytest.raises(ValueError,match='gate|PASS'):
        M.main(argv)
    assert not (scene/'runs/core_joint_v2/plan.json').exists()


def test_single_explicit_pilot_is_not_blocked_by_full_release_gate(scene,monkeypatch):
    calls=[]
    monkeypatch.setattr(M.Batch,'prepare',lambda self,job:calls.append(('prepare',job['core_index'])))
    monkeypatch.setattr(M.Batch,'train_and_export',lambda self,job:calls.append(('train',job['core_index'])))
    assert M.main(['--core-ids','15'])==0
    assert calls==[('prepare',15),('train',15)]


def test_gate_rejects_duplicate_status_and_changed_pass_artifact(scene):
    path=scene/'reports/gate.json';path.parent.mkdir()
    path.write_text('{"overall":"FAIL","overall":"PASS"}')
    with pytest.raises(ValueError,match='Duplicate'):
        M.validate_gate_report(path)
    passing_gate_fixture(scene)
    checked=M.validate_gate_report(path)
    report=M.read_json(path);report['experiment']='b';write_json(path,report)
    with pytest.raises(ValueError,match='changed'):
        M.validate_gate_report(path,checked['sha256'])


def test_gate_rejects_wrong_external_plan_and_changed_evidence(scene):
    from build_acceptance_plan import object_sha
    path=passing_gate_fixture(scene)
    report=M.read_json(path);report['acceptance_plan_id']='a'*64
    report['report_id']=object_sha({k:v for k,v in report.items() if k!='report_id'})
    write_json(path,report)
    with pytest.raises(ValueError,match='this plan'):
        M.validate_gate_report(path)
    passing_gate_fixture(scene)
    (scene/'reports/evidence.json').write_text('changed after PASS')
    with pytest.raises(ValueError,match='changed'):
        M.validate_gate_report(path)


def test_refined_checkpoint_and_export_may_exceed_warm_count(scene):
    import torch
    p=plan();job=p['jobs'][15];warm=warm_fixture(job,p);batch=M.Batch(p)
    _,state,inspection,path=checkpoint_fixture(job,batch)
    # Model and Adam both acquired actual duplicated records.
    for key,tensor in state['pipeline'].items():
        state['pipeline'][key]=torch.cat([tensor,tensor[:2]])
    for optimizer in state['optimizers'].values():
        for saved in optimizer['state'].values():
            for key in ('exp_avg','exp_avg_sq'):
                saved[key]=torch.cat([saved[key],saved[key][:2]])
    torch.save(state,path)
    output=M.local(job['staging'])/'checkpoint_inspection.json'
    M.inspect_training(batch.job_record(job),output)
    inspection=M.read_json(output)
    assert inspection['gaussians']==warm['gaussians']+2
    result_fixture(job,inspection);evaluation_fixture(job,inspection)
    inspection=final_inspection(job,batch)
    M.validate_export(job,inspection,warm['gaussians'])

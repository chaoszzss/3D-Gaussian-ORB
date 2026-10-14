"""Real file-boundary checks for fresh raw-image pilots and split provenance."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resolution_pilot", ROOT / "scripts/prepare_resolution_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def approved_v3_gate():
    from orbvi_gaussian.surfels import SurfelDepthPolicy
    return {"s0": {"status": "PASS", "passed": 7}, "s1": {"status": "ready"},
            "approved_policy": asdict(SurfelDepthPolicy.trusted_ground_v3()),
            "v3": {"status": "PASS", "independent_safety_status": "PASS",
                   "unknown_foreground_gate": "PASS", "dev_frames_checked": 60},
            "implementation_sha256": {"src/orbvi_gaussian/" + name: pilot.sha(ROOT / "src/orbvi_gaussian" / name)
                                      for name in ("surfels.py", "points.py")}}


def test_disk_reserve_rejects_write_that_crosses_shared_volume_floor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(pilot.shutil, "disk_usage", lambda path: SimpleNamespace(free=pilot.MIN_FREE_DISK_BYTES + 100))
    pilot.require_disk_reserve(tmp_path / "not_created", 99)
    with pytest.raises(OSError, match="150 GiB"):
        pilot.require_disk_reserve(tmp_path / "not_created", 100)


@pytest.fixture
def source_data(tmp_path, monkeypatch):
    # Tiny fixtures may live on a smaller /tmp filesystem than production data.
    # The real 150 GiB boundary is exercised separately above.
    monkeypatch.setattr(pilot, "MIN_FREE_DISK_BYTES", 1024 * 1024)
    source, raw = tmp_path / "source", tmp_path / "raw"
    source.mkdir(); raw.mkdir()
    original = {"pixel_center_convention": "half_integer; principal point +0.5 from OpenCV rectification",
                "camera_model": "OPENCV", "point_cloud_has_rgb": True, "ply_file_path": "points3D.ply"}
    frames = []
    for i in range(3):
        pose = np.eye(4); pose[:3, 3] = [i, 2 * i, -i]
        frame = {"w": 640, "h": 540, "fl_x": 320., "fl_y": 320., "cx": 320., "cy": 270.,
                 "source_header_ns": 1000 + i, "source_bag_ns": 2000 + i, "view_id": "center",
                 "transform_matrix": pose.tolist()}
        frame["camera_id"] = "camera_2"
        frame["group_id"] = str(100 + i)
        frame["timestamp_ns"] = 100 + i
        frame["source_file_path"] = f"raw_{i}.jpg"
        frame["file_path"] = f"images/{i}.png"
        frame["mask_path"] = f"masks/{i}.png"
        frame["lidar_depth_path"] = f"depth/{i}.npy"
        frame["lidar_depth_mask_path"] = f"depth_masks/{i}.png"
        frames.append(frame)
        # Alternating raw stripes distinguish direct rectification from old RGB.
        picture = np.zeros((1080, 1280, 3), np.uint8)
        picture[:, ::2, :] = 255
        assert cv2.imwrite(str(raw / frame["source_file_path"]), picture)
    meta = {k: v for k, v in original.items() if k not in (*pilot.SPLITS, "frames")}
    meta["frames"] = frames
    for i, split in enumerate(pilot.SPLITS):
        meta[split] = [frames[i]["file_path"]]
    (source / "transforms.json").write_text(json.dumps(meta))
    prep = {"status": "prepared_experimental", "world_frame": "camera_init", "units": "metres",
            "map": {"color_source": "calibrated_training_images_only"}, "export": {},
            "calibration": str(pilot.BASE_CONFIG), "calibration_sha256": pilot.sha(pilot.BASE_CONFIG),
            "frames": 3, "groups": 3}
    (source / "preparation.json").write_text(json.dumps(prep))
    (source / "points3D.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 1 20 30 40\n")
    np.savez(source / "lidar_reference.npz", xyz=np.array([[0, 0, 1]], np.float32),
             normals=np.array([[0, 0, 1]], np.float32), normal_valid=np.array([True]))
    sparse = source / "sparse/0"; sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 640 540 320 320 319.5 269.5\n")
    (sparse / "images.txt").write_text("# source poses preserved byte for byte\n")
    (sparse / "points3D.txt").write_text("# source training colors\n")
    return source, raw, meta


@pytest.mark.parametrize("width", [640, 1280])
def test_raw_rectification_preserves_world_splits_and_independent_assets(source_data, tmp_path, width):
    source, raw, original = source_data
    before = {str(p.relative_to(source)): pilot.sha(p) for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "pilot"
    prep = pilot.prepare_images(source, out, raw, width)
    meta = pilot.read_json(out / "transforms.json")
    assert (out / "INCOMPLETE.json").exists()
    assert prep["status"] == "images_prepared_depth_pending"
    assert prep["resolution_pilot"]["initializer"]["recolorized"] is False
    for split in pilot.SPLITS:
        assert meta[split] == original[split]
    for old, new in zip(original["frames"], meta["frames"]):
        for key in ("transform_matrix", "timestamp_ns", "source_header_ns", "source_bag_ns", "group_id"):
            assert old[key] == new[key]
        assert (new["cx"], new["cy"]) == (width / 2, width * 27 / 64)
        assert new["fl_x"] == pytest.approx(width / 2)
        mask = cv2.imread(str(out / new["mask_path"]), 0)
        image = cv2.imread(str(out / new["file_path"]))
        assert image.shape == (width * 27 // 32, width, 3)
        assert set(np.unique(mask)) <= {0, 255}
        assert image.max() > 0
        assert not (out / new["lidar_depth_path"]).exists()
    for name in ("points3D.ply", "lidar_reference.npz", "sparse/0/images.txt", "sparse/0/points3D.txt"):
        assert pilot.sha(source / name) == pilot.sha(out / name)
        assert (source / name).stat().st_ino != (out / name).stat().st_ino
    (out / "points3D.ply").write_text("edit copied initializer")
    assert before == {str(p.relative_to(source)): pilot.sha(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        pilot.prepare_images(source, out, raw, width)


def test_ego_v3_changes_only_virtual_size():
    _, selected, base = pilot.calibration_for_width(1280)
    assert selected["cameras"] == base["cameras"]
    assert selected["pose"] == base["pose"]
    assert selected["lidar"] == base["lidar"]
    T, width, height, fx, fy, cx, cy = pilot.virtual_parameters(selected["virtual_views"][0])
    assert (width, height, cx, cy) == (1280, 1080, 639.5, 539.5)
    assert (fx, fy) == pytest.approx((640, 640))


def test_rejects_half_pixel_mismatch_before_creating_destination(source_data, tmp_path):
    source, raw, meta = source_data
    meta["frames"][0]["cx"] -= .5
    (source / "transforms.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="principal-point"):
        pilot.prepare_images(source, tmp_path / "pilot", raw)
    assert not (tmp_path / "pilot").exists()


def test_rejects_split_leakage_and_nontrain_initializer(source_data, tmp_path):
    source, raw, meta = source_data
    meta["test_filenames"] = meta["train_filenames"]
    with pytest.raises(ValueError, match="split-leaking"):
        pilot.split_membership(meta)
    prep = pilot.read_json(source / "preparation.json")
    prep["map"]["color_source"] = "all_images"
    (source / "preparation.json").write_text(json.dumps(prep))
    with pytest.raises(ValueError, match="training images only"):
        pilot.prepare_images(source, tmp_path / "pilot", raw)


def test_rejects_old_640_image_as_raw(source_data, tmp_path):
    source, raw, meta = source_data
    cv2.imwrite(str(raw / meta["frames"][0]["source_file_path"]), np.zeros((540, 640, 3), np.uint8))
    with pytest.raises(ValueError, match="original 1280x1080"):
        pilot.prepare_images(source, tmp_path / "pilot", raw)


def test_depth_is_gated_masked_and_publishes_only_after_all_frames(source_data, tmp_path, monkeypatch):
    import orbvi_gaussian.surfels as surfels
    source, raw, original = source_data
    out = tmp_path / "pilot"
    pilot.prepare_images(source, out, raw, 640)
    report = tmp_path / "s0.json"
    gate = approved_v3_gate()
    gate["s0"] = {"status": "FAIL", "passed": 0}
    report.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="S0 PASS"):
        pilot.generate_depth(out, report)
    assert (out / "INCOMPLETE.json").exists() and not (out / "depth").exists()
    gate["s0"] = {"status": "PASS", "passed": 7}
    report.write_text(json.dumps(gate))
    calls = []

    def project(xyz, normals, normal_valid, T, K, width, height, *, policy, diagnostics):
        assert policy.ground_support and policy.multilayer_mode == "front_cluster"
        np.testing.assert_allclose(K, [[320, 0, 319.5], [0, 320, 269.5], [0, 0, 1]], atol=1e-12, rtol=0)
        assert (K[0, 2], K[1, 2]) == (319.5, 269.5)
        expected_pose = np.asarray(original["frames"][len(calls)]["transform_matrix"]) @ np.diag([1., -1., -1., 1.])
        np.testing.assert_array_equal(T, expected_pose)
        calls.append(T)
        depth = np.full((height, width), 3., np.float32)
        depth[1, 1] = np.nan
        diagnostics["valid_pixels"] = width * height
        return depth, np.ones((height, width), bool)

    monkeypatch.setattr(surfels, "project_surfel_depth", project)
    prep = pilot.generate_depth(out, report)
    assert len(calls) == 3 and not (out / "INCOMPLETE.json").exists()
    assert prep["status"] == "prepared_experimental"
    for frame in original["frames"]:
        depth = np.load(out / frame["lidar_depth_path"])
        support = cv2.imread(str(out / frame["lidar_depth_mask_path"]), 0) > 0
        mask = cv2.imread(str(out / frame["mask_path"]), 0) > 0
        assert np.isfinite(depth).all() and np.all(depth[~support] == 0)
        assert not np.any(support & ~mask) and not support[1, 1]
    with pytest.raises(ValueError, match="fresh pending dataset"):
        pilot.generate_depth(out, report)


def test_depth_rejects_geometry_changed_since_s0(source_data, tmp_path):
    source, raw, _ = source_data
    out = tmp_path / "pilot"
    pilot.prepare_images(source, out, raw, 640)
    gate = tmp_path / "s0.json"
    attestation = approved_v3_gate()
    attestation["implementation_sha256"]["src/orbvi_gaussian/surfels.py"] = "wrong"
    gate.write_text(json.dumps(attestation))
    with pytest.raises(ValueError, match="gate attestation"):
        pilot.generate_depth(out, gate)
    assert (out / "INCOMPLETE.json").exists() and not (out / "depth").exists()


def test_copy_images_keeps_exact_bytes_but_discards_old_teacher(source_data, tmp_path):
    source, raw, _ = source_data
    first, copied = tmp_path / "first", tmp_path / "copied"
    prep = pilot.prepare_images(source, first, raw, 640)
    prep["status"] = "prepared_experimental"
    prep["teacher_provenance"] = {"protocol": "obsolete teacher"}
    pilot.write_json(first / "preparation.json", prep)
    (first / "INCOMPLETE.json").unlink()
    new = pilot.copy_prepared_images(first, copied)
    assert new["status"] == "images_prepared_depth_pending"
    assert "teacher_provenance" not in new
    for output in new["export"]["paths"].values():
        assert copied.resolve() in Path(output).parents
        assert Path(output).is_file()
    assert (copied / "INCOMPLETE.json").exists()
    assert pilot.sha(first / "transforms.json") == pilot.sha(copied / "transforms.json")
    for frame in pilot.read_json(first / "transforms.json")["frames"]:
        for key in ("file_path", "mask_path"):
            assert pilot.sha(first / frame[key]) == pilot.sha(copied / frame[key])
            assert (first / frame[key]).stat().st_ino != (copied / frame[key]).stat().st_ino
        assert not (copied / frame["lidar_depth_path"]).exists()
    with pytest.raises(FileExistsError):
        pilot.copy_prepared_images(first, copied)


def test_v3_gate_requires_independent_foreground_and_dev_validation(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"s0": {"status": "PASS", "passed": 7}, "s1": {"status": "ready"},
                               "approved_policy": {"version": "surfel_depth_v3"}}))
    with pytest.raises(ValueError, match="V3 requires independent"):
        pilot.generate_depth(tmp_path / "no_dataset", gate)

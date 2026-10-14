import copy
import csv
import json

import cv2
import numpy as np
import pytest

from orbvi_gaussian.colors import colorize_from_frames
from orbvi_gaussian.io import write_json
from orbvi_gaussian.training import splatfacto_command
from orbvi_gaussian.trajectory import trajectory_quality
from orbvi_gaussian.workflow import load_trajectory


def test_colorization_respects_occlusion_mask_and_rgb_order(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    image = np.zeros((3, 3, 3), np.uint8)
    image[1, 1] = [10, 20, 200]
    cv2.imwrite(str(tmp_path / "images/rgb.png"), image)
    mask = np.ones((3, 3), np.uint8) * 255
    cv2.imwrite(str(tmp_path / "masks/valid.png"), mask)
    frame = {"file_path": "images/rgb.png", "mask_path": "masks/valid.png",
             "transform_matrix": np.eye(4), "w": 3, "h": 3,
             "fl_x": 1., "fl_y": 1., "cx": 1., "cy": 1.}
    xyz = np.array([[0, 0, 2.], [0, 0, 4.], [0, 0, -2.], [100, 0, 2.]])
    rgb, supported = colorize_from_frames(xyz, [frame], tmp_path)
    assert supported.tolist() == [True, False, False, False]
    assert rgb[0].tolist() == [200, 20, 10]
    mask[1, 1] = 0
    cv2.imwrite(str(tmp_path / "masks/valid.png"), mask)
    _, supported = colorize_from_frames(xyz, [frame], tmp_path)
    assert not supported.any()


def test_numerically_finite_divergence_is_rejected():
    T = np.repeat(np.eye(4)[None], 3, axis=0)
    T[:, 0, 3] = [0., .1, 1e10]
    report = trajectory_quality(np.array([1000000000, 1100000000, 1200000000]), T)
    assert report["all_finite"]
    assert report["status"] == "invalid_for_reconstruction"
    assert report["first_bad_motion_relative_s"] == .2


def test_clock_lateness_uses_running_maximum():
    stamps = np.array([10000000, 9500000, 9000000, 8500000, 8000000])
    T = np.repeat(np.eye(4)[None], len(stamps), axis=0)
    report = trajectory_quality(stamps, T)
    assert report["max_header_lateness_ns"] == 2000000
    assert report["status"] == "invalid_for_reconstruction"


def _write_poses(path, times, frames=None, positions=None):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header_ns", "frame_id", "child_frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for i, ns in enumerate(times):
            writer.writerow([ns, (frames or ["world"] * len(times))[i], "body",
                             (positions or [0.] * len(times))[i], 0, 0, 0, 0, 0, 1])


def test_small_clock_reordering_preserves_true_ns_and_rejects_frame_switch(tmp_path):
    path = tmp_path / "poses.csv"
    times = [1000000000, 1100023127, 1100000000, 1200000000]
    _write_poses(path, times)
    trajectory = load_trajectory(path)
    assert trajectory.stamps_ns.tolist() == sorted(times)
    assert trajectory.input_time_reordering["local_backwards_pairs"] == 1
    assert path.read_text().splitlines()[2].startswith("1100023127,")
    _write_poses(path, [1000000000, 1100000000], ["world", "different_world"])
    with pytest.raises(ValueError):
        load_trajectory(path)


def test_conflicting_same_timestamp_is_rejected(tmp_path):
    path = tmp_path / "poses.csv"
    _write_poses(path, [1000000000, 1000000000], positions=[0., 0.001])
    with pytest.raises(ValueError):
        load_trajectory(path)


def test_training_plan_preserves_metric_coordinates_and_color_gate(tmp_path):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    write_json(dataset / "preparation.json", {"status": "prepared"})
    transforms = {"frames": [{"transform_matrix": np.eye(4).tolist()}],
                  "train_filenames": ["images/a.png"], "val_filenames": ["images/b.png"],
                  "point_cloud_has_rgb": False}
    write_json(dataset / "transforms.json", transforms)
    out = tmp_path / "training_output"
    with pytest.raises(ValueError):
        splatfacto_command(dataset, out)
    command = splatfacto_command(dataset, out, allow_random_init=True)
    for flag, value in [("--auto-scale-poses", "False"), ("--scale-factor", "1.0"),
                        ("--center-method", "none"), ("--orientation-method", "none"),
                        ("--pipeline.datamanager.cache-images", "cpu")]:
        assert command[command.index(flag) + 1] == value
    assert not out.exists()
    transforms.update(point_cloud_has_rgb=True, ply_file_path="missing.ply")
    (dataset / "transforms.json").write_text(json.dumps(transforms))
    with pytest.raises(ValueError):
        splatfacto_command(dataset, out)
    transforms.update(point_cloud_has_rgb=False)
    (dataset / "transforms.json").write_text(json.dumps(transforms))
    with pytest.raises(ValueError):
        splatfacto_command(dataset, "/var/lib/orbvi-training", allow_random_init=True)

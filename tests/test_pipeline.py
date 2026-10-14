"""Cross-module regressions for clock domains, calibration gates and file boundaries."""

import copy
import csv
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from orbvi_gaussian.calibration import validate_calibration
from orbvi_gaussian.io import (RAW_TOPICS, extract_bag, image_decode, output_path,
                              serialized_header)
from orbvi_gaussian.workflow import plan_blocks, prepare_dataset, rectify_preview

BASE_NS = 1720267880000000000


def test_header_ns_precision_and_invalid_data():
    data = struct.pack("<4I", 7, 1720267880, 123456789, 5) + b"world" + b"payload"
    assert serialized_header(data) == (BASE_NS + 123456789, "world", 21)
    for raw in [data[:4], data[:19], struct.pack("<4I", 0, 0, 0, 0),
                struct.pack("<4I", 0, 1, 1000000000, 0)]:
        with pytest.raises(ValueError):
            serialized_header(raw)


def test_output_boundary_and_symlink_escape(tmp_path):
    assert output_path(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError):
        output_path("/var/lib/orbvi-new-output")
    link = tmp_path / "escape"
    link.symlink_to("/var/lib", target_is_directory=True)
    with pytest.raises(ValueError):
        output_path(link / "orbvi-new-output")


def test_image_row_padding():
    class Image:
        width, height, step, encoding = 2, 2, 8, "rgb8"
        data = bytes([255, 0, 0, 0, 255, 0, 91, 92, 0, 0, 255, 255, 255, 0, 91, 92])
    decoded = image_decode(Image())
    assert decoded.shape == (2, 2, 3)
    assert decoded[0, 0].tolist() == [0, 0, 255]
    assert decoded[1, 0].tolist() == [255, 0, 0]


def synthetic_calibration():
    return {"schema_version": 1, "verified": True, "calibration_id": "SYNTHETIC_TEST_ONLY",
            "provenance": "generated unit test; never real ORBVI calibration",
            "pose": {"source": "device_vio", "timestamp_basis": "header_ns",
                     "world_frame": "world", "body_frame": "body", "frame_semantics_verified": True},
            "cameras": {c: {"topic": t, "topic_mapping_verified": True, "model": "opencv_fisheye",
                            "width": 32, "height": 24,
                            "intrinsics": {"fx": 12.0, "fy": 12.0, "cx": 15.5, "cy": 11.5},
                            "distortion": [0.0] * 4, "time_offset_ns": 0,
                            "T_body_camera": np.eye(4).tolist()} for c, t in RAW_TOPICS.items()},
            "virtual_views": [{"name": "center", "width": 24, "height": 16,
                               "yaw_deg": 0.0, "fov_horizontal_deg": 75.0}]}


def test_missing_actual_model_and_pose_semantics_rejected():
    cal = synthetic_calibration()
    for mutate in [lambda c: c.update(verified=False),
                   lambda c: c["pose"].update(frame_semantics_verified=False),
                   lambda c: c["cameras"]["camera_2"].update(topic_mapping_verified=False),
                   lambda c: c["cameras"]["camera_2"].update(model="triple_sphere")]:
        candidate = copy.deepcopy(cal)
        mutate(candidate)
        with pytest.raises(ValueError):
            validate_calibration(candidate)


def make_test_bag(path):
    import rosbag
    from genpy import Time
    from nav_msgs.msg import Odometry, Path as RosPath
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CompressedImage, Imu, PointCloud2, PointField
    from std_msgs.msg import Header

    def time(ns):
        return Time(ns // 1000000000, ns % 1000000000)

    _, jpeg = cv2.imencode(".jpg", np.full((24, 32, 3), [25, 70, 140], np.uint8))
    poses = []
    with rosbag.Bag(str(path), "w") as bag:
        for index in range(25):
            ns = BASE_NS + index * 50000000
            recorded = time(ns + 65737401700000000)
            if index % 2 == 0:
                m = Odometry()
                m.header = Header(stamp=time(ns), frame_id="world")
                m.child_frame_id = "body"
                m.pose.pose.position.x = index * 0.01
                m.pose.pose.orientation.w = 1.0
                bag.write("/orbvi/vio/odometry", m, recorded)
                ps = PoseStamped(header=m.header, pose=m.pose.pose)
                poses.append(ps)
            imu = Imu()
            imu.header = Header(stamp=time(ns), frame_id="imu")
            imu.linear_acceleration.z = 9.81
            imu.orientation_covariance[0] = -1
            bag.write("/orbvi/imu", imu, recorded)
            for i, topic in enumerate(RAW_TOPICS.values()):
                msg = CompressedImage()
                msg.header = Header(stamp=time(ns + i * 1000000), frame_id="raw_%d" % i)
                msg.format = "bgr8; jpeg compressed bgr8"
                msg.data = jpeg.tobytes()
                bag.write(topic, msg, recorded)
            if index % 2 == 0:
                pc = PointCloud2()
                pc.header = Header(stamp=time(ns), frame_id="sensor")
                pc.height, pc.width = 1, 1
                pc.fields = [PointField(name=n, offset=i * 4, datatype=7, count=1)
                             for i, n in enumerate(["x", "y", "z"])]
                pc.point_step, pc.row_step = 12, 12
                pc.data = struct.pack("<3f", 1.0, 2.0, 3.0)
                bag.write("/orbvi/depth/points", pc, recorded)
        bag.write("/orbvi/vio/path", RosPath(header=Header(stamp=time(ns), frame_id="world"), poses=poses), recorded)


def test_real_ros_serialization_to_training_formats(tmp_path):
    # Real ROS serialization catches assumptions that mocked message objects miss.
    source = tmp_path / "synthetic.bag"
    make_test_bag(source)
    extracted = tmp_path / "extracted"
    manifest = extract_bag(source, extracted, camera_hz=2.0, ipm_hz=0.0, cloud_hz=1.0,
                           include_clouds=True)
    assert manifest["anchor_header_ns"] == BASE_NS
    assert manifest["observed_counts"]["/orbvi/vio/odometry"] == 13
    assert manifest["saved_counts"][RAW_TOPICS["camera_2"]] == 3
    rows = list(csv.DictReader((extracted / "indexes/images.csv").open()))
    assert int(rows[0]["bag_ns"]) - int(rows[0]["header_ns"]) > 65000000 * 1000000000
    cal = synthetic_calibration()
    cal_path = tmp_path / "synthetic_calibration.yaml"
    cal_path.write_text(yaml.safe_dump(cal))
    preview = rectify_preview(extracted, cal_path, tmp_path / "preview")
    assert len(preview["views"]) == 4
    prepared = tmp_path / "prepared"
    report = prepare_dataset(extracted, cal_path, prepared, holdout_every=2)
    assert report["frames"] == 12
    ns = json.loads((prepared / "transforms.json").read_text())
    assert ns["point_cloud_has_rgb"] is False
    assert not set(ns["train_filenames"]) & set(ns["val_filenames"])
    membership = {p: "train" for p in ns["train_filenames"]}
    membership.update({p: "val" for p in ns["val_filenames"]})
    by_group = {}
    for frame in ns["frames"]:
        assert frame["source_bag_ns"] > frame["source_header_ns"]
        by_group.setdefault(frame["group_id"], set()).add(membership[frame["file_path"]])
    assert all(len(v) == 1 for v in by_group.values())
    blocks = plan_blocks(extracted, tmp_path / "blocks.json", length_m=0.1,
                         overlap_m=0.02, max_duration_s=0.7)
    assert blocks["blocks"]
    assert all(b["floor_id"] is None for b in blocks["blocks"])
    # Holdout groups are red/blue; only the green training group may color map points.
    for row in rows:
        group = int(round((int(row["header_ns"]) - BASE_NS) / 500000000))
        color = [[0, 0, 255], [0, 255, 0], [255, 0, 0]][group]
        cv2.imwrite(str(extracted / row["file_path"]), np.full((24, 32, 3), color, np.uint8))
    point_map = tmp_path / "synthetic_map.ply"
    point_map.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                         "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                         "end_header\n0 0 2.5 255 0 0\n0.15 0 2.8 255 0 0\n0.4 0 3.0 255 0 0\n")
    colored = tmp_path / "color_prepared"
    colored_report = prepare_dataset(extracted, cal_path, colored, holdout_every=2,
                                      initialization_map=point_map, map_world_frame="world",
                                      map_alignment_verified=True, save_depth=True, colorize_from_images=True)
    from orbvi_gaussian.points import read_point_cloud
    _, measured_colors = read_point_cloud(colored / "points3D.ply")
    assert np.max(np.abs(measured_colors.astype(int) - [0, 255, 0])) <= 2  # JPEG quantization
    assert colored_report["map"]["map_rgb_verified_by_operator"] is False
    assert colored_report["map"]["color_source"] == "calibrated_training_images_only"
    ns_color = json.loads((colored / "transforms.json").read_text())
    assert ns_color["ply_file_path"] == "points3D.ply"
    assert all((colored / f["lidar_depth_path"]).exists() for f in ns_color["frames"])
    from orbvi_gaussian.trajectory import export_pose_bag
    external = tmp_path / "external_poses"
    provenance = export_pose_bag(source, "/orbvi/vio/odometry", extracted, external)
    assert provenance["poses"] == 13
    assert provenance["time_span_coverage_fraction"] == 1.0
    cal["verified"] = False
    cal_path.write_text(yaml.safe_dump(cal))
    failed_out = tmp_path / "must_not_exist"
    with pytest.raises(ValueError):
        prepare_dataset(extracted, cal_path, failed_out)
    assert not failed_out.exists()

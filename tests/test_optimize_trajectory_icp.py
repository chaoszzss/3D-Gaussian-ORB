import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SPEC = importlib.util.spec_from_file_location("trajectory_icp", Path(__file__).resolve().parents[1] / "scripts/optimize_trajectory_icp.py")
icp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(icp)


def test_local_correction_rotates_about_pivot_and_preserves_rigid_chain():
    local = np.eye(4)
    local[:3, :3] = Rotation.from_euler("z", .5, degrees=True).as_matrix()
    local[:3, 3] = [.01, -.02, .003]
    pivot = np.array([80., -50., 18.])
    world = icp.centered_to_world(local, pivot)
    assert np.allclose(world[:3, :3] @ pivot + world[:3, 3], pivot + local[:3, 3])
    assert icp.within_bounds(local)
    assert not icp.within_bounds(world)  # incorrect world-origin translation gate would fail
    assert np.allclose(world[:3, :3].T @ world[:3, :3], np.eye(3))


def test_bounds_reject_floor_jump_and_large_rotation():
    T = np.eye(4)
    T[2, 3] = .051
    assert not icp.within_bounds(T)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", 1.01, degrees=True).as_matrix()
    assert not icp.within_bounds(T)


def test_correction_gate_requires_holdout_improvement_support_and_observability():
    before = {"fitness": .95, "capped_plane_rmse_m": .05, "nn_p50_m": .1, "nn_p95_m": .2}
    after = dict(before, capped_plane_rmse_m=.04)
    obs = {"rank": 6, "condition": 30.}
    assert icp.accept_correction(np.eye(4), before, after, obs)[0]
    for values, geometry in [(dict(after, fitness=.8), obs),
                              (dict(after, nn_p95_m=.3), obs),
                              (before, obs), (after, {"rank": 3, "condition": None})]:
        assert not icp.accept_correction(np.eye(4), before, values, geometry)[0]
    assert not icp.accept_correction(np.eye(4), before, after, obs, bounded=False)[0]
    assert not icp.accept_correction(np.eye(4), before, dict(after, capped_plane_rmse_m=float("nan")), obs)[0]


def test_source_and_target_support_are_consistent_and_frozen_before_fit():
    points = np.array([[0, 0, 0], [0, 0, 1.7], [0, 0, 1.8], [0, 0, -3.]])
    assert np.array_equal(icp.common_support(points), points[:2])


def test_planar_scene_is_not_observable_in_six_degrees_of_freedom():
    x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    normals = np.tile([0., 0., 1.], (len(points), 1))
    report = icp.observability(points, points, normals, cKDTree(points), np.eye(4))
    assert report["rank"] == 3
    assert report["condition"] is None


def test_deskew_preserves_integer_epoch_ns_and_actual_lio_body_chain():
    from orbvi_gaussian.geometry import PoseTrajectory
    start = 1720268320733244035
    T = np.repeat(np.eye(4)[None], 3, axis=0)
    T[:, 0, 3] = [0., .1, .2]
    traj = PoseTrajectory(np.array([start, start+100000000, start+200000000]), T)
    scan = {"xyz": np.array([[2., 0., 0.]]*4), "tag": np.array([0, 0, 1, 0]),
            "offset_ns": np.array([0, 100000000, 0, 400000000], np.uint32),
            "timebase_ns": start, "reflectivity": np.array([10, 20, 30, 40])}
    extrinsic = np.diag([1., -1., -1., 1.])
    extrinsic[1, 3] = .05
    world, refl, stats = icp.deskew(scan, traj, extrinsic)
    assert np.allclose(world, [[2, .05, 0], [2.1, .05, 0]])
    assert refl.tolist() == [10, 20]
    assert stats["deskew_valid_fraction"] == pytest.approx(2/3)


def test_static_pose_gate_rejects_moving_platform():
    from orbvi_gaussian.geometry import PoseTrajectory
    t = np.arange(21, dtype=np.int64)*100000000 + 1000000000
    T = np.repeat(np.eye(4)[None], len(t), axis=0)
    still = PoseTrajectory(t, T)
    assert icp.motion_window(still, int(t[10]))["static_pose_candidate"]
    T[:, 0, 3] = np.arange(21)*.01
    moving = PoseTrajectory(t, T)
    assert not icp.motion_window(moving, int(t[10]))["static_pose_candidate"]
    assert not icp.motion_window(still, int(t[0]))["static_pose_candidate"]


def test_exact_map_provenance_probe_rejects_different_coordinates(tmp_path):
    points = np.array([[1., 2., 3., 0], [4., 5., 6., 0], [7., 8., 9., 0]], dtype="<f4")
    path = tmp_path / "map.pcd"
    path.write_bytes(b"FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nPOINTS 3\nDATA binary\n" + points.tobytes())
    assert icp.prove_map_subset(points[:, :3].astype(float), path)["passed"]
    assert not icp.prove_map_subset(points[:, :3].astype(float) + 1, path)["passed"]


def test_cpu_icp_recovers_known_small_rigid_error_and_rejects_large_one():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(17)
    planes = []
    for axis in range(3):
        points = rng.uniform(-2, 2, (1600, 3))
        points[:, axis] = 2
        planes.append(points)
    target = np.concatenate(planes)
    D = np.eye(4)
    D[:3, :3] = Rotation.from_rotvec([.003, -.004, .002]).as_matrix()
    D[:3, 3] = [.025, -.02, .015]
    source = (target - D[:3, 3]) @ D[:3, :3]
    fit, held = source[np.arange(len(source)) % 5 != 0], source[np.arange(len(source)) % 5 == 0]
    report = icp.bounded_icp(fit, held, target)
    assert report["accepted_local_candidate"], report
    assert np.allclose(report["correction_local"], D, atol=2e-3)
    assert report["verified"] is False
    rejected = icp.bounded_icp(fit - [0, 0, .15], held - [0, 0, .15], target)
    assert not rejected["accepted_local_candidate"]
    assert "trust_region_exceeded" in rejected["rejection_reasons"]

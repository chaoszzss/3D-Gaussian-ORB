"""Regression tests for pose conventions, calibrated projections and masks."""

import unittest

import cv2
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from orbvi_gaussian.geometry import (
    PoseTrajectory, camera_world_pose, make_virtual_view,
    matrix_to_quat_xyzw, opencv_c2w_to_colmap, opencv_c2w_to_opengl,
    opengl_c2w_to_opencv, quat_xyzw_to_matrix, triple_sphere_project,
    triple_sphere_unproject, validate_transform,
)


def pose(euler=(0, 0, 0), translation=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    T[:3, 3] = translation
    return T


class PoseGeometryTests(unittest.TestCase):
    def test_xyzw_rotation_and_normalization(self):
        q = np.array([0, 0, np.sqrt(.5), np.sqrt(.5)])
        expected = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        for factor in (1., -3., 1e-300, 1e300):
            assert_allclose(quat_xyzw_to_matrix(q * factor), expected, atol=1e-14)
        assert_allclose(matrix_to_quat_xyzw(quat_xyzw_to_matrix(q)), q, atol=1e-14)

    def test_bad_quaternions_rejected(self):
        for q in ([0] * 4, [0, 0, 1], [[0, 0, 0, 1]], [0, 0, np.nan, 1],
                  [0, 0, 0, np.inf]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                quat_xyzw_to_matrix(q)

    def test_rigid_transform_validation_and_copy(self):
        valid = pose((10, 20, 30), (1, 2, 3))
        out = validate_transform(valid)
        out[0, 3] = -9
        self.assertEqual(valid[0, 3], 1)
        invalid = []
        reflection = np.diag([-1, 1, 1, 1]); invalid.append(reflection)
        scaling = np.diag([1.01, 1, 1, 1]); invalid.append(scaling)
        row = np.eye(4); row[3, 0] = .001; invalid.append(row)
        bad = np.eye(4); bad[0, 3] = np.nan; invalid.append(bad)
        invalid += [np.eye(3), np.zeros((4, 4))]
        for T in invalid:
            with self.subTest(T=T), self.assertRaises(ValueError):
                validate_transform(T)

    def test_pose_chain_applies_lever_arm_in_rotated_body(self):
        world_body = pose((0, 0, 90), (10, 20, 30))
        body_camera = pose((0, 90, 0), (2, 0, 0))
        camera_virtual = pose((0, 0, 90))
        result = camera_world_pose(world_body, body_camera, camera_virtual)
        assert_allclose(result[:3, 3], [10, 22, 30], atol=1e-14)
        # A virtual forward ray becomes camera right, body down, world right.
        point = np.array([0, 0, 1, 1])
        assert_allclose(result @ point, [10, 23, 30, 1], atol=1e-14)

    def test_colmap_world_to_camera_and_opengl_roundtrip(self):
        T = pose((15, -40, 72), (3, -5, 10))
        q_wxyz, t = opencv_c2w_to_colmap(T)
        R = quat_xyzw_to_matrix(q_wxyz[[1, 2, 3, 0]])
        assert_allclose(R @ T[:3, 3] + t, 0, atol=1e-14)
        X_camera = np.array([.2, -.7, 4])
        X_world = T[:3, :3] @ X_camera + T[:3, 3]
        assert_allclose(R @ X_world + t, X_camera, atol=1e-14)
        gl = opencv_c2w_to_opengl(T)
        assert_allclose(gl[:3, 0], T[:3, 0])
        assert_allclose(gl[:3, 1:3], -T[:3, 1:3])
        assert_allclose(gl[:3, 3], T[:3, 3])
        assert_allclose(opengl_c2w_to_opencv(gl), T)

    def test_interpolation_epoch_ns_retains_single_ns_precision(self):
        base = 1_700_000_000_000_000_000
        trajectory = PoseTrajectory(np.array([base, base + 100], dtype=np.int64),
                                    np.stack([pose(), pose((0, 0, 90), (10, 0, 0))]),
                                    max_gap_ns=100)
        middle = trajectory.at(base + 50)
        assert_allclose(middle[:3, 3], [5, 0, 0])
        assert_allclose(middle[:3, :3], pose((0, 0, 45))[:3, :3], atol=1e-14)
        assert_allclose(trajectory.at(base + 1)[:3, 3], [.1, 0, 0])

    def test_shortest_slerp_path_through_180_degrees(self):
        trajectory = PoseTrajectory(np.array([0, 10]),
                                    np.stack([pose((0, 0, 170)), pose((0, 0, -170))]), 10)
        assert_allclose(trajectory.at(5)[:3, :3], pose((0, 0, 180))[:3, :3], atol=1e-14)

    def test_gap_rejection_exact_endpoints_and_no_extrapolation(self):
        poses = np.stack([pose(), pose(translation=(1, 2, 3))])
        trajectory = PoseTrajectory(np.array([10, 100]), poses, max_gap_ns=89)
        self.assertIsNone(trajectory.at(11))
        self.assertIsNone(trajectory.at(99))
        self.assertIsNone(trajectory.at(9))
        self.assertIsNone(trajectory.at(101))
        assert_array_equal(trajectory.at(10), poses[0])
        assert_array_equal(trajectory.at(100), poses[1])
        endpoint = trajectory.at(10); endpoint[0, 3] = -9
        assert_array_equal(trajectory.at(10), poses[0])
        single = PoseTrajectory(np.array([10]), poses[:1])
        assert_array_equal(single.at(10), poses[0])
        self.assertIsNone(single.at(11))

    def test_invalid_trajectory_input(self):
        for stamps in ([], [1., 2.], [2, 1], [1, 1], [[1, 2]],
                       np.array([2 ** 63], dtype=np.uint64)):
            with self.subTest(stamps=stamps), self.assertRaises(ValueError):
                PoseTrajectory(np.asarray(stamps), np.stack([pose(), pose()]))
        for gap in (0, -1, True, 2.5):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                PoseTrajectory(np.array([1]), np.stack([pose()]), gap)
        trajectory = PoseTrajectory(np.array([0]), np.stack([pose()]))
        with self.assertRaises(ValueError):
            trajectory.at(.1)


class RectificationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 100, 3), 200, dtype=np.uint8)
        self.K = np.array([[40., 0, 49.5], [0, 41., 39.5], [0, 0, 1]])
        self.D = np.array([.01, -.002, .0001, -.00001])

    def view(self, **kwargs):
        parameters = dict(image=self.image, K=self.K, D=self.D,
                          T_camera_virtual=pose(), width=20, height=16,
                          fx=40., fy=40., cx=9.5, cy=7.5, model="equidistant")
        parameters.update(kwargs)
        return make_virtual_view(**parameters)

    def test_full_mask_and_black_source_are_valid(self):
        out, mask = self.view()
        self.assertEqual(out.shape, (16, 20, 3))
        assert_array_equal(out, 200)
        assert_array_equal(mask, 255)
        out, mask = self.view(image=np.zeros_like(self.image))
        assert_array_equal(out, 0)
        assert_array_equal(mask, 255)

    def test_rotated_view_sampling_matches_forward_projection(self):
        yy, xx = np.indices(self.image.shape[:2], dtype=np.float32)
        source = np.stack([xx, yy, np.full_like(xx, 7)], axis=-1)
        T = pose((0, 25, 0))
        out, mask = self.view(image=source, T_camera_virtual=T)
        y, x = 8, 10
        ray = T[:3, :3] @ np.array([(x - 9.5) / 40., (y - 7.5) / 40., 1.])
        uv, _ = cv2.fisheye.projectPoints(ray.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                        self.K, self.D)
        self.assertEqual(mask[y, x], 255)
        assert_allclose(out[y, x, :2], uv[0, 0], atol=1 / 32.)
        other, _ = self.view(image=source, T_camera_virtual=pose((0, -25, 0)))
        self.assertGreater(float(out[y, x, 0]), float(other[y, x, 0]))

    def test_border_and_source_mask_bilinear_validity(self):
        mask = np.full(self.image.shape[:2], 255, dtype=np.uint8)
        mask[35:45, 45:55] = 0
        out, valid = self.view(source_mask=mask)
        self.assertEqual(valid[8, 10], 0)
        assert_array_equal(out[valid == 0], 0)
        out, valid = self.view(width=200, height=200, fx=10., fy=10., cx=99.5, cy=99.5)
        self.assertTrue(np.any(valid == 0))
        self.assertTrue(np.any(valid == 255))
        assert_array_equal(out[valid == 255], 200)
        assert_array_equal(out[valid == 0], 0)

    def test_back_facing_source_rays_rejected_for_kb4(self):
        out, mask = self.view(T_camera_virtual=pose((0, 180, 0)))
        assert_array_equal(mask, 0)
        assert_array_equal(out, 0)

    def test_no_implicit_calibration_or_arbitrary_fisheye_model(self):
        for model in (None, "omni", "scaramuzza", "fisheye", "triple_sphere"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.view(model=model)
        for D in (None, [], np.zeros(3), np.zeros(5), np.zeros((2, 2)), [0, 0, 0, np.nan]):
            with self.subTest(D=D), self.assertRaises(ValueError):
                self.view(D=D)
        with self.assertRaises(ValueError):
            self.view(K=None)

    def test_bad_virtual_camera_and_mask_rejected(self):
        for kwargs in (dict(T_camera_virtual=pose(translation=(1, 0, 0))),
                       dict(width=0), dict(width=2.5), dict(fx=-1), dict(fy=np.nan),
                       dict(source_mask=np.ones((1, 1))),
                       dict(source_mask=np.full((80, 100), 2))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.view(**kwargs)
        K = self.K.copy(); K[0, 1] = .5
        with self.assertRaises(ValueError):
            self.view(K=K)


class TripleSphereTests(unittest.TestCase):
    def setUp(self):
        # Golden ORBVI calibration used by the local Vikit C++ reference test.
        self.K = np.array([[617.38977302476292, 0, 644.02209060597841],
                           [0, 622.04838389900192, 552.24488765399883], [0, 0, 1]])
        self.parameters = {"xi": -.20389208692403207, "lambda": -.065891008932412801,
                           "alpha": .59248909223834112, "b": 0., "c": 0.}

    def test_golden_center_and_110_degree_ray(self):
        theta = np.deg2rad(110)
        rays = [[0, 0, 1], [np.sin(theta), 0, np.cos(theta)]]
        pixels, valid = triple_sphere_project(rays, self.K, self.parameters)
        assert_array_equal(valid, [True, True])
        assert_allclose(pixels[0], [644.02209060597841, 552.24488765399883], atol=1e-12)
        assert_allclose(pixels[1], [1214.301300, 552.244887654], atol=1e-6)
        recovered, valid = triple_sphere_unproject(pixels, self.K, self.parameters)
        assert_array_equal(valid, True)
        assert_allclose(recovered, rays, atol=1e-12)

    def test_hemisphere_roundtrip_affine_b_c_and_scale_invariance(self):
        rng = np.random.RandomState(13)
        theta = rng.uniform(0, np.deg2rad(120), size=400)
        phi = rng.uniform(-np.pi, np.pi, size=400)
        rays = np.c_[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        parameters = dict(self.parameters, b=3.25, c=-1.75)
        pixels, valid = triple_sphere_project(rays, self.K, parameters)
        self.assertTrue(np.all(valid))
        recovered, inverse_valid = triple_sphere_unproject(pixels, self.K, parameters)
        self.assertTrue(np.all(inverse_valid))
        assert_allclose(recovered, rays, atol=3e-13)
        scaled, valid_scaled = triple_sphere_project(rays * 1e250, self.K, parameters)
        assert_array_equal(valid_scaled, valid)
        assert_allclose(scaled, pixels, atol=1e-10)

    def test_folded_and_nonfinite_rays_are_invalid_not_mirrored(self):
        theta = np.deg2rad(150)
        rays = [[np.sin(theta), 0, np.cos(theta)], [0, 0, -1], [0, 0, 0], [np.nan, 0, 1]]
        pixels, valid = triple_sphere_project(rays, self.K, self.parameters)
        assert_array_equal(valid, False)
        self.assertTrue(np.all(np.isnan(pixels)))
        rays, valid = triple_sphere_unproject([[1e6, 1e6]], self.K, self.parameters)
        self.assertFalse(valid[0])
        self.assertTrue(np.all(np.isnan(rays)))

    def test_required_named_parameters_and_model_separation(self):
        for parameters in (None, [0, 0, .5, 0, 0], dict(self.parameters, alpha=1),
                           {k: v for k, v in self.parameters.items() if k != "lambda"}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                triple_sphere_project([[0, 0, 1]], self.K, parameters)
        with self.assertRaises(ValueError):
            make_virtual_view(np.zeros((1080, 1280), np.uint8), self.K, np.zeros(5),
                              pose(), 1, 1, 1, 1, 0, 0, model="triple_sphere",
                              model_parameters=self.parameters)

    def test_virtual_110_degree_view_valid_and_150_degree_fold_rejected(self):
        source = np.full((1080, 1280), 203, dtype=np.uint8)
        out, mask = make_virtual_view(source, self.K, None, pose((0, 110, 0)),
                                     5, 5, 100, 100, 2, 2, model="triple_sphere",
                                     model_parameters=self.parameters)
        assert_array_equal(mask, 255)
        assert_array_equal(out, 203)
        out, mask = make_virtual_view(source, self.K, None, pose((0, 150, 0)),
                                     5, 5, 100, 100, 2, 2, model="triple_sphere",
                                     model_parameters=self.parameters)
        assert_array_equal(mask, 0)
        assert_array_equal(out, 0)

    def test_current_v6_board_four_camera_parameters_and_rearward_domain(self):
        # Snapshot: current v6/maps/board-calib-backup/camera_config.yaml,
        # SHA256 1651d01dfd2d2d949888ecf99334d551b0a5675ef5249d5bea83b80e0eb7f55f.
        # This validates the model math, not the candidate calibration's truth.
        vectors = [
            [619.9727239894911, 624.4998924652682, 645.77827037008922,
             495.91837506770293, -.20175555452665106, -.066260533217913864,
             .59234047378840116, 0., 0.],
            [619.45163699117472, 623.83187937727325, 625.43841331882231,
             528.60666048376572, -.20534996288752175, -.062234901229899428,
             .5927662823683536, 0., 0.],
            [611.84744243535681, 616.05854792728314, 663.21070129594227,
             527.46845752491276, -.20584750961851611, -.066665673290309105,
             .59160036634795832, 0., 0.],
            [617.00288813992495, 621.26738066249618, 673.99092516701148,
             538.35654794942263, -.20503092348711907, -.064381057151794763,
             .59128823597164037, 0., 0.],
        ]
        theta = np.deg2rad([0., 45., 90., 110., 120.])
        rays = np.c_[np.sin(theta), np.zeros_like(theta), np.cos(theta)]
        for camera, v in enumerate(vectors):
            with self.subTest(camera=camera):
                K = np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]])
                p = dict(zip(["xi", "lambda", "alpha", "b", "c"], v[4:]))
                uv, valid = triple_sphere_project(rays, K, p)
                assert_array_equal(valid, True)
                assert_allclose(uv[0], v[2:4], atol=1e-12)
                back, domain = triple_sphere_unproject(uv, K, p)
                assert_array_equal(domain, True)
                assert_allclose(back, rays, atol=3e-13)
                source = np.full((1080, 1280), 183, dtype=np.uint8)
                output, mask = make_virtual_view(source, K, None, pose((0, 95, 0)),
                                                3, 3, 100, 100, 1, 1,
                                                model="triple_sphere", model_parameters=p)
                assert_array_equal(mask, 255)
                assert_array_equal(output, 183)


if __name__ == "__main__":
    unittest.main()

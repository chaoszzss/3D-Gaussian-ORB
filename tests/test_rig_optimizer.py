"""CPU-only rig algebra/gradient contracts; no rasterization or quality claims.

Run: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
     .venv-train/bin/python -B -m unittest discover -s tests -p test_rig_optimizer.py -v
"""

import io
import math
import unittest
from types import SimpleNamespace

import torch
from nerfstudio.cameras.cameras import Cameras

from orbvi_gaussian.rig_optimizer import SharedRigOptimizer, se3_exp, so3_exp


def tearDownModule():
    if torch.cuda.is_initialized():
        raise AssertionError("CPU rig suite must not initialize CUDA")


def hat(vector):
    """Independent algebra oracle for torch.matrix_exp."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)


def cv_viewmat(c2w):
    """NS 1.1.5 get_viewmat equations, without importing its renderer stack."""
    rotation = c2w[..., :3, :3] * c2w.new_tensor([1., -1., -1.])
    inverse = rotation.transpose(-1, -2)
    top = torch.cat((inverse, -inverse @ c2w[..., :3, 3:4]), -1)
    bottom = c2w.new_tensor([0., 0., 0., 1.]).expand(c2w.shape[:-2] + (1, 4))
    return torch.cat((top, bottom), -2)


class TestSharedRigOptimizer(unittest.TestCase):
    def setUp(self):
        self.rig = SharedRigOptimizer(dtype=torch.float64, device="cpu")
        self.eye = torch.eye(4, dtype=torch.float64)

    def set_delta(self, row, value):
        with torch.no_grad():
            self.rig.delta[row].copy_(torch.tensor(value, dtype=torch.float64))

    def test_zero_identity_shapes_homogeneous_row_and_no_mutation(self):
        pose = se3_exp(torch.tensor([.3, -.1, .2, 10., 2., -3.], dtype=torch.float64))
        original = pose.clone()
        for value in (pose, pose[:3], pose.repeat(4, 1, 1), pose[:3].repeat(4, 1, 1)):
            with self.subTest(shape=value.shape):
                output = self.rig.correct(value, 2)
                self.assertEqual(output.shape, value.shape)
                torch.testing.assert_close(output, value, rtol=0, atol=0)
        self.set_delta(0, [.02, -.01, .03, .02, .01, -.01])
        corrected = self.rig.correct(pose, 2)
        torch.testing.assert_close(corrected[3], self.eye[3], rtol=0, atol=0)
        torch.testing.assert_close(pose, original, rtol=0, atol=0)

    def test_so3_proper_rotations_and_matrix_exponential_oracle(self):
        omega = torch.tensor([[0., 0., 0.], [1e-12, -2e-12, 3e-12], [.009, 0., 0.],
                              [.011, 0., 0.], [.2, -.1, .3], [math.pi, 0., 0.]], dtype=torch.float64)
        rotations = so3_exp(omega)
        torch.testing.assert_close(rotations, torch.matrix_exp(hat(omega)), rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, self.eye[:3, :3].expand_as(rotations))
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(len(omega), dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(so3_exp, (omega.clone().requires_grad_(),)))

    def test_true_se3_exponential_translation_coupling(self):
        delta = torch.tensor([[0., 0., 0., .03, -.02, .01],
                              [.11, -.07, .04, .08, .03, -.04],
                              [1e-9, -1e-9, 2e-9, .01, .02, .03]], dtype=torch.float64)
        generator = torch.zeros(3, 4, 4, dtype=torch.float64)
        generator[:, :3, :3] = hat(delta[:, :3])
        generator[:, :3, 3] = delta[:, 3:]
        result = se3_exp(delta)
        torch.testing.assert_close(result, torch.matrix_exp(generator), rtol=1e-11, atol=1e-12)
        self.assertFalse(torch.allclose(result[1, :3, 3], delta[1, 3:]))
        self.assertTrue(torch.autograd.gradcheck(se3_exp, (delta.clone().requires_grad_(),)))
        self.assertTrue(torch.autograd.gradcheck(se3_exp, (torch.zeros(6, dtype=torch.float64, requires_grad=True),)))

    def test_zero_gradient_has_analytic_local_generators(self):
        pose = self.eye.clone()
        pose[:3, :3] = so3_exp(torch.tensor([.3, -.2, .1], dtype=torch.float64))
        pose[:3, 3] = torch.tensor([10., 20., 30.], dtype=torch.float64)
        weights = torch.arange(1., 17., dtype=torch.float64).reshape(4, 4)
        loss = (self.rig.correct(pose, 3) * weights).sum()
        loss.backward()
        expected = torch.zeros_like(self.rig.delta)
        for component in range(6):
            generator = torch.zeros_like(self.eye)
            if component < 3:
                generator[:3, :3] = hat(torch.eye(3, dtype=torch.float64)[component])[0]
            else:
                generator[component - 3, 3] = 1
            expected[1, component] = ((pose @ generator) * weights).sum()
        torch.testing.assert_close(self.rig.delta.grad, expected, rtol=1e-12, atol=1e-12)

    def test_viewmat_projection_gradient_matches_central_finite_difference(self):
        for value in ([0.] * 6, [.05, -.03, .02, .02, -.01, .03]):
            with self.subTest(delta=value):
                self.set_delta(0, value)
                self.rig.zero_grad(set_to_none=True)
                pose = self.eye.clone()
                pose[:3, :3] = so3_exp(torch.tensor([.2, .1, -.2], dtype=torch.float64))
                pose[:3, 3] = torch.tensor([1., 2., 3.], dtype=torch.float64)
                # Fixed world points/means stay in meters; pinhole projection
                # exercises delta -> c2w -> CV viewmat -> normalized image ray.
                points = pose @ torch.tensor([[.3, -.4], [.2, -.3], [-3., -4.], [1., 1.]], dtype=torch.float64)

                def objective():
                    camera_points = cv_viewmat(self.rig.correct(pose, 2)) @ points
                    uv = camera_points[:2] / camera_points[2:3]
                    return (uv * uv.new_tensor([[.7, -.2], [.3, .5]])).sum()

                objective().backward()
                analytic = self.rig.delta.grad[0].clone()
                numeric = torch.empty(6, dtype=torch.float64)
                epsilon = 1e-6
                with torch.no_grad():
                    for component in range(6):
                        original = self.rig.delta[0, component].item()
                        self.rig.delta[0, component] = original + epsilon
                        plus = objective().item()
                        self.rig.delta[0, component] = original - epsilon
                        minus = objective().item()
                        self.rig.delta[0, component] = original
                        numeric[component] = (plus - minus) / (2 * epsilon)
                torch.testing.assert_close(analytic, numeric, rtol=1e-6, atol=1e-8)
                self.assertTrue(torch.all(analytic.abs() > 1e-5))
                torch.testing.assert_close(self.rig.delta.grad[1:], torch.zeros(3, 6, dtype=torch.float64))

    def test_input_pose_gradient_is_preserved(self):
        self.set_delta(0, [.01, .02, -.03, .02, -.01, .03])
        pose = self.eye.clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda p: self.rig.correct(p, 2), (pose,)))

    def test_ns_axes_and_shared_correction_across_timestamps(self):
        poses = self.eye.repeat(3, 1, 1)
        poses[1, :3, :3] = so3_exp(torch.tensor([0., 0., math.pi / 2], dtype=torch.float64))
        poses[1, :3, 3] = torch.tensor([10., 20., 30.], dtype=torch.float64)
        poses[2, :3, 3] = torch.tensor([-2., 3., 1.], dtype=torch.float64)
        for axis in range(3):
            self.set_delta(0, [0.] * 6)
            with torch.no_grad():
                self.rig.delta[0, axis + 3] = .05
            corrected = self.rig.correct(poses, torch.tensor([[2], [2], [3]]))
            unit = torch.eye(3, dtype=torch.float64)[axis] * .05
            torch.testing.assert_close(corrected[:2, :3, 3] - poses[:2, :3, 3], poses[:2, :3, :3] @ unit)
            torch.testing.assert_close(corrected[2], poses[2])
            recovered = torch.linalg.inv(poses[:2]) @ corrected[:2]
            torch.testing.assert_close(recovered[0], recovered[1])
        # Positive NS local z is backward: it becomes negative CV forward.
        shifted = self.rig.correct(self.eye, 2)
        camera_point = cv_viewmat(shifted) @ torch.tensor([0., 0., -2., 1.], dtype=torch.float64)
        self.assertAlmostEqual(camera_point[2].item(), 2.05)
        self.set_delta(0, [0., 0., .1, 0., 0., 0.])
        output = self.rig.correct(poses[:2], 2)
        torch.testing.assert_close(output[:, :3, :3], poses[:2, :3, :3] @ so3_exp(self.rig.delta[0, :3]))
        torch.testing.assert_close(output[:, :3, 3], poses[:2, :3, 3])

    def test_metadata_ids_override_and_unknown_viewer(self):
        self.set_delta(0, [.01, .02, .03, .02, 0., 0.])
        pose = self.eye[:3].unsqueeze(0)
        for metadata in (None, {}, {"cam_idx": 2}, {"physical_camera_id": 99}):
            camera = SimpleNamespace(camera_to_worlds=pose, metadata=metadata)
            torch.testing.assert_close(self.rig.correct(camera), pose, rtol=0, atol=0)
        camera = SimpleNamespace(camera_to_worlds=pose, metadata={"physical_camera_id": torch.tensor([[2]])})
        torch.testing.assert_close(self.rig(camera), self.rig.correct(pose, 2))
        torch.testing.assert_close(self.rig.correct(camera, 3), pose)
        torch.testing.assert_close(camera.camera_to_worlds, pose)
        for training in (True, False):
            self.rig.train(training)
            torch.testing.assert_close(self.rig.correct(camera), self.rig.correct(pose, 2))
        unknown = self.rig.correct(self.eye, -1)
        unknown.sum().backward()
        torch.testing.assert_close(self.rig.delta.grad, torch.zeros_like(self.rig.delta))

    def test_four_physical_ids_compact_mapping_and_batch_broadcast(self):
        with torch.no_grad():
            self.rig.delta[:, 3] = torch.tensor([.01, .02, .03, .04], dtype=torch.float64)
        poses = self.eye.repeat(2, 4, 1, 1)
        corrected = self.rig.correct(poses, [2, 3, 4, 5])
        torch.testing.assert_close(corrected[..., 0, 3], self.rig.delta[:, 3].expand(2, 4))
        compact = SharedRigOptimizer(physical_camera_id_offset=0, dtype=torch.float64)
        with torch.no_grad():
            compact.delta.copy_(self.rig.delta)
        torch.testing.assert_close(compact.correct(poses, [0, 1, 2, 3]), corrected)
        torch.testing.assert_close(self.rig.correct(self.eye, 0), self.eye)
        torch.testing.assert_close(compact.correct(self.eye, 5), self.eye)

    def test_real_nerfstudio_camera_metadata_and_no_mutation(self):
        self.set_delta(0, [.01, -.02, .03, .02, -.01, .03])
        poses = self.eye[:3].repeat(2, 1, 1)
        cameras = Cameras(camera_to_worlds=poses, fx=100., fy=100., cx=50., cy=50.,
                          metadata={"physical_camera_id": torch.tensor([[2], [3]])})
        snapshot = cameras.camera_to_worlds.clone()
        torch.testing.assert_close(self.rig.correct(cameras), self.rig.correct(poses, [2, 3]))
        torch.testing.assert_close(self.rig.correct(cameras[:1]), self.rig.correct(poses[:1], 2))
        torch.testing.assert_close(cameras.camera_to_worlds, snapshot, rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bounds_project_and_regularization_exact_values_and_gradients(self):
        self.assertEqual(set(self.rig.regularization()), {"rig_rotation_prior", "rig_translation_prior"})
        self.assertTrue(all(loss.item() == 0 for loss in self.rig.regularization().values()))
        self.set_delta(0, [self.rig.rotation_prior_rad.item(), 0., 0., self.rig.translation_prior_m.item(), 0., 0.])
        losses = self.rig.regularization()
        for loss in losses.values():
            self.assertAlmostEqual(loss.item(), .125)
        sum(losses.values()).backward()
        self.assertAlmostEqual(self.rig.delta.grad[0, 0].item(), 1 / (4 * self.rig.rotation_prior_rad.item()))
        self.assertAlmostEqual(self.rig.delta.grad[0, 3].item(), 1 / (4 * self.rig.translation_prior_m.item()))
        with torch.no_grad():
            self.rig.delta.copy_(torch.tensor([[1., 2., -3., 1., -2., 3.]] * 4, dtype=torch.float64))
        effective = self.rig.effective_delta()
        torch.testing.assert_close(effective[:, :3].norm(dim=-1), self.rig.max_rotation_rad.expand(4))
        torch.testing.assert_close(effective[:, 3:].norm(dim=-1), self.rig.max_translation_m.expand(4))
        matrices = self.rig.correction_matrices()
        self.assertTrue(torch.all(matrices[:, :3, 3].norm(dim=-1) <= .2 + 1e-12))
        angles = torch.acos(((matrices[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        self.assertTrue(torch.all(angles <= math.radians(12) + 1e-12))
        self.rig.zero_grad(set_to_none=True)
        sum(self.rig.regularization().values()).backward()
        self.assertTrue(torch.all(self.rig.delta.grad * self.rig.delta > 0))
        parameter = self.rig.delta
        self.assertIs(self.rig.project_(), self.rig)
        self.assertIs(self.rig.delta, parameter)
        torch.testing.assert_close(self.rig.delta, effective)
        torch.testing.assert_close(self.rig.correction_matrices(), matrices)

    def test_named_optimizer_group_updates_only_shared_parameters(self):
        groups = self.rig.get_param_groups()
        self.assertEqual(list(groups), ["rig"])
        self.assertEqual(list(self.rig.named_parameters()), [("delta", self.rig.delta)])
        self.assertEqual(self.rig.delta.numel(), 24)
        optimizer = torch.optim.SGD(groups["rig"], lr=.001)
        output = self.rig.correct(self.eye, 4)
        ((output[0, 3] - .04).square() + sum(self.rig.regularization().values())).backward()
        optimizer.step()
        self.rig.project_()
        self.assertGreater(self.rig.delta[2, 3].item(), 0.)
        torch.testing.assert_close(self.rig.delta[[0, 1, 3]], torch.zeros(3, 6, dtype=torch.float64))

    def test_saved_state_restores_ids_priors_bounds_and_four_corrections(self):
        rig = SharedRigOptimizer(physical_camera_id_offset=0, rotation_prior_rad=.04,
                                 translation_prior_m=.06, max_rotation_rad=.15,
                                 max_translation_m=.12, dtype=torch.float64)
        with torch.no_grad():
            rig.delta.copy_(torch.arange(24, dtype=torch.float64).reshape(4, 6) * .002)
        stream = io.BytesIO()
        torch.save(rig.state_dict(), stream)
        stream.seek(0)
        restored = SharedRigOptimizer(dtype=torch.float64)
        restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        for name, value in rig.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        poses = self.eye.repeat(4, 1, 1)
        torch.testing.assert_close(restored.correct(poses, [0, 1, 2, 3]), rig.correct(poses, [0, 1, 2, 3]))
        torch.testing.assert_close(restored.correct(self.eye, 5), self.eye)

    def test_float32_zero_and_small_angle_backward_is_finite(self):
        rig = SharedRigOptimizer()
        for magnitude in (0., 1e-8, 1e-4, .009, .011):
            with torch.no_grad():
                rig.delta.fill_(magnitude)
            rig.zero_grad(set_to_none=True)
            output = rig.correct(torch.eye(4).repeat(4, 1, 1), [2, 3, 4, 5])
            (output * torch.arange(16.).reshape(4, 4)).sum().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(rig.delta.grad).all())

    def test_invalid_contracts_fail_explicitly(self):
        for kwargs in ({"physical_camera_id_offset": 1}, {"physical_camera_id_offset": True},
                       {"physical_camera_id_offset": 2.0},
                       {"rotation_prior_rad": 0}, {"translation_prior_m": -1},
                       {"max_rotation_rad": math.pi}, {"max_translation_m": float("nan")},
                       {"dtype": torch.float16}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SharedRigOptimizer(**kwargs)
        for ids in (True, 2., [2., 3.], [2, 3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.rig.correct(self.eye, ids)
        with self.assertRaises(ValueError):
            self.rig.correct(torch.eye(3), 2)
        with self.assertRaises(ValueError):
            self.rig.correct(torch.eye(4), 2)  # float32 pose with float64 rig
        for function, value in ((so3_exp, torch.zeros(2)), (se3_exp, torch.zeros(5)),
                                (se3_exp, torch.zeros(6, dtype=torch.long))):
            with self.assertRaises(ValueError):
                function(value)


if __name__ == "__main__":
    unittest.main()

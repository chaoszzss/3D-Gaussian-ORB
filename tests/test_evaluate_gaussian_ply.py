"""CPU fixtures for the real GS exporter/reader and fixed-source scoring rules."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_gaussian_ply", PROJECT / "scripts/evaluate_gaussian_ply.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import numpy as np


class ArrayTensor:
    """NumPy adapter for executing the existing exporter without Torch/NS."""
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()

    def transpose(self, a, b):
        return ArrayTensor(self.value.swapaxes(a, b))

    def flatten(self, start_dim):
        return ArrayTensor(self.value.reshape(*self.value.shape[:start_dim], -1))


def existing_export_ply():
    """Execute the unchanged production exporter, without its eager GPU imports."""
    tree = ast.parse((PROJECT / "scripts/evaluate_reconstruction.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "export_ply")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROJECT / "scripts/evaluate_reconstruction.py"), "exec"), scope)
    return scope["export_ply"]


def model_fixture(degree=3):
    count = (degree + 1) ** 2 - 1
    # Each color and coefficient has a different value; N=2 also catches a
    # reshape across Gaussian rows. Powers of two remain exact float32 values.
    rest = np.arange(2 * count * 3, dtype=np.float32).reshape(2, count, 3) / 128.
    return SimpleNamespace(
        means=ArrayTensor([[1., -2., 3.], [4., 5., -6.]]),
        features_dc=ArrayTensor([[.25, -.5, .75], [1., 1.25, -1.5]]),
        features_rest=ArrayTensor(rest), scales=ArrayTensor([[-2., -3., -4.], [-5., -6., -7.]]),
        opacities=ArrayTensor([[-1.5], [2.5]]),
        quats=ArrayTensor([[2., 0., 0., 0.], [.5, .5, .5, .5]]))


def standard_blob(names, rows, count=None, types=None):
    header = "ply\nformat binary_little_endian 1.0\ncomment CPU schema fixture\nelement vertex %d\n" % (
        len(rows) if count is None else count)
    header += "".join("property %s %s\n" % (types[i] if types else "float", name)
                      for i, name in enumerate(names))
    return (header + "end_header\n").encode("ascii") + np.asarray(rows, dtype="<f4").tobytes()


FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
ROW = [1., 2., 3., 0., 0., 0., .1, .2, .3, -9., -1., -2., -3., 1., 0., 0., 0.]


class CandidateCpuTests(unittest.TestCase):
    def setUp(self):
        local_tmp = PROJECT / "runtime/tmp"
        local_tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="candidate_cpu_", dir=local_tmp)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, blob, name="candidate.ply"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return path

    def exported(self, degree=3):
        model = model_fixture(degree)
        path = self.root / ("export_degree_%d.ply" % degree)
        existing_export_ply()(model, path)
        return path, model

    def test_existing_exporter_round_trip_all_sh_degrees_and_unactivated_attributes(self):
        for degree in range(4):
            with self.subTest(degree=degree):
                path, original = self.exported(degree)
                params, info = M.read_candidate(path)
                self.assertEqual(info["sh_degree"], degree)
                self.assertEqual(info["gaussians"], 2)
                self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                for key, attr in (("means", "means"), ("features_dc", "features_dc"),
                                  ("features_rest", "features_rest"), ("scales", "scales"),
                                  ("opacities", "opacities")):
                    np.testing.assert_array_equal(params[key], getattr(original, attr).value)
                    self.assertEqual(params[key].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(params["quats"], [[1., 0., 0., 0.], [.5, .5, .5, .5]])

    def test_numeric_sh_order_and_shuffled_columns(self):
        # Independently specified standard degree-3 PLY channel-major records.
        names = FIELDS + ["f_rest_%d" % i for i in range(45)] + ["confidence"]
        rows = [ROW + list(range(15)) + list(range(100, 115)) + list(range(200, 215)) + [.8]]
        order = list(range(len(names)))[::-1]
        path = self.put(standard_blob([names[i] for i in order], np.array(rows)[:, order]))
        params, info = M.read_candidate(path)
        np.testing.assert_array_equal(params["features_rest"][0, 2], [2., 102., 202.])
        np.testing.assert_array_equal(params["features_rest"][0, 10], [10., 110., 210.])
        np.testing.assert_array_equal(params["means"], [[1., 2., 3.]])
        self.assertEqual(params["opacities"].item(), -9.)  # No opacity threshold/filter.
        self.assertIn("confidence", info["fields"])

    def test_invalid_schema_rejected_by_real_bounded_parser(self):
        cases = [(FIELDS[:-1], ROW[:-1]), (FIELDS + ["x"], ROW + [1.]),
                 (FIELDS + ["f_rest_0"], ROW + [0.]),
                 (FIELDS + ["f_dc_3"], ROW + [0.]),
                 (FIELDS + ["scale_3"], ROW + [0.]),
                 (FIELDS + ["rot_4"], ROW + [0.])]
        for i, (names, row) in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                M.read_candidate(self.put(standard_blob(names, [row]), "%d.ply" % i))
        blob = standard_blob(FIELDS, [ROW], types=["double"] + ["float"] * (len(FIELDS) - 1))
        with self.assertRaisesRegex(ValueError, "float32"):
            M.read_candidate(self.put(blob, "double.ply"))
        for format_ in (b"ascii", b"binary_big_endian"):
            with self.subTest(format=format_), self.assertRaisesRegex(ValueError, "binary_little_endian"):
                M.read_candidate(self.put(standard_blob(FIELDS, [ROW]).replace(b"binary_little_endian", format_), "format.ply"))

    def test_nonfinite_required_and_extra_attributes_are_not_silently_removed(self):
        for field in ("x", "opacity", "scale_0", "rot_1", "confidence"):
            for invalid in (np.nan, np.inf, -np.inf):
                names, row = FIELDS + ["confidence"], ROW + [.7]
                row[names.index(field)] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaisesRegex(ValueError, "Nonfinite"):
                    M.read_candidate(self.put(standard_blob(names, [row])))

    def test_invalid_quaternions_and_float32_scale_activation(self):
        for quat in ([0., 0., 0., 0.], [2., 0., 0., 0.]):
            row = ROW.copy()
            row[-4:] = quat
            with self.subTest(quat=quat), self.assertRaisesRegex(ValueError, "rotation|unit wxyz"):
                M.read_candidate(self.put(standard_blob(FIELDS, [row])))
        for log_scale in (100., -200.):
            row = ROW.copy()
            row[FIELDS.index("scale_1")] = log_scale
            with self.subTest(log_scale=log_scale), self.assertRaisesRegex(ValueError, "Log scales"):
                M.read_candidate(self.put(standard_blob(FIELDS, [row])))

    def test_bounds_truncation_and_trailing_bytes(self):
        for blob in (standard_blob(FIELDS, [ROW])[:-1], standard_blob(FIELDS, [ROW]) + b"junk"):
            with self.assertRaisesRegex(ValueError, "payload"):
                M.read_candidate(self.put(blob))
        with patch.object(M, "MAX_GAUSSIANS", 1), self.assertRaisesRegex(ValueError, "Gaussian budget"):
            M.read_candidate(self.put(standard_blob(FIELDS, [ROW, ROW])))
        with patch.object(M, "MAX_PLY_BYTES", 16), self.assertRaisesRegex(ValueError, "byte budget"):
            M.read_candidate(self.put(standard_blob(FIELDS, [ROW])))
        with self.assertRaises(ValueError):
            M.read_candidate(self.put(standard_blob(FIELDS, [], count=0)))

    def test_explicit_full_map_budget_does_not_silently_subset_records(self):
        path = self.put(standard_blob(FIELDS, [ROW, ROW]))
        with patch.object(M, 'MAX_GAUSSIANS', 1):
            params, info = M.read_candidate(path, max_gaussians=2, max_ply_bytes=4096)
        self.assertEqual(info['gaussians'], 2)
        self.assertEqual(len(params['means']), 2)
        for bad in [0, -1, True, 6_000_001]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                M.read_candidate(path, max_gaussians=bad, max_ply_bytes=4096)

    def test_multichunk_reader_hashes_and_maps_all_records(self):
        path, original = self.exported()
        # Force one Gaussian per chunk through the real packaging scanner.
        with patch.object(M, "MAX_CHUNK_BYTES", 4 * 62):
            params, info = M.read_candidate(path)
        np.testing.assert_array_equal(params["features_rest"], original.features_rest.value)
        self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_reader_import_and_schema_validation_without_torch_nerfstudio_or_cv2(self):
        path, _ = self.exported()
        code = """
import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *a, **kw):
    if name.split('.')[0] in ('torch', 'nerfstudio', 'cv2', 'skimage'):
        raise AssertionError('eager rendering import: ' + name)
    return original(name, *a, **kw)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parameters, info = module.read_candidate(sys.argv[2])
assert info['gaussians'] == 2 and parameters['features_rest'].shape == (2,15,3)
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PROJECT / "scripts/evaluate_gaussian_ply.py"), str(path)],
                                capture_output=True, text=True, check=False, cwd=PROJECT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_path_escape_symlink_and_no_overwrite_before_render_imports(self):
        for path in (PROJECT.parent / "outside.ply", self.root / "../../../../outside"):
            with self.assertRaisesRegex(ValueError, "inside project"):
                M.confined_path(path)
        link = self.root / "escape.ply"
        link.symlink_to("/dev/null")
        with self.assertRaisesRegex(ValueError, "inside project"):
            M.read_candidate(link)
        config = self.put(b"actual-config-placeholder", "config.yml")
        ply = self.put(standard_blob(FIELDS, [ROW]))
        with self.assertRaises(FileExistsError):
            M.main(["--config", str(config), "--ply", str(ply), "--output", str(self.root)])
        self.assertEqual(config.read_bytes(), b"actual-config-placeholder")

    def test_frame_indices_are_split_local_ordered_bounded_and_unique(self):
        self.assertEqual(M.select_frame_indices(6, [4, 1, 5], 2), [4, 1])
        self.assertEqual(M.select_frame_indices(3), [0, 1, 2])
        for indices in ([], [-1], [6], [1, 1], [True], [1.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                M.select_frame_indices(6, indices)
        with self.assertRaises(ValueError):
            M.select_frame_indices(6, max_images=-1)

    def test_identity_world_parser_no_relative_tolerance_on_scale_or_rotation(self):
        output = SimpleNamespace(dataparser_transform=np.eye(4)[:3], dataparser_scale=1.)
        _, scale = M.require_identity_parser(output)
        self.assertEqual(scale, 1.)
        for value in (.5, 1.0001, float("nan")):
            with self.subTest(scale=value), self.assertRaisesRegex(ValueError, "identity parser"):
                M.require_identity_parser(SimpleNamespace(dataparser_transform=np.eye(4)[:3], dataparser_scale=value))
        transform = np.eye(4)[:3]
        transform[0, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, "identity parser"):
            M.require_identity_parser(SimpleNamespace(dataparser_transform=transform, dataparser_scale=1.))

    def test_physical_camera_identity_preserves_metadata_pose_and_intrinsics(self):
        camera = SimpleNamespace(camera_to_worlds=np.eye(4)[:3][None].copy(), fx=np.array([32.]), fy=np.array([31.]),
                                 cx=np.array([16.5]), cy=np.array([15.5]), width=np.array([33]), height=np.array([31]),
                                 metadata={"cam_idx": 17, "other": "keep"})
        frame = dict(transform_matrix=np.eye(4).tolist(), camera_id="camera_5", fl_x=32., fl_y=31.,
                     cx=16.5, cy=15.5, w=33, h=31)
        before = camera.camera_to_worlds.copy()
        identity = M.fixed_camera_identity(camera, frame, {})
        self.assertEqual(identity["physical_camera_id"], 5)
        self.assertEqual(camera.metadata, {"cam_idx": 17, "other": "keep", "physical_camera_id": 5})
        np.testing.assert_array_equal(camera.camera_to_worlds, before)
        camera.metadata["physical_camera_id"] = 2
        with self.assertRaisesRegex(ValueError, "conflicts"):
            M.fixed_camera_identity(camera, frame, {})
        camera.metadata["physical_camera_id"] = 5
        camera.cx += .5
        with self.assertRaisesRegex(ValueError, "source cx"):
            M.fixed_camera_identity(camera, frame, {})

    def test_explicit_split_asset_mapping_and_camera_group_leakage(self):
        frames = []
        for index in range(3):
            path = self.put(b"source-placeholder", "images/%d.png" % index)
            frames.append(dict(file_path=str(path.relative_to(self.root)), group_id=str(index),
                               timestamp_ns=index + 1, camera_id="camera_2"))
        meta = dict(frames=frames, train_filenames=[frames[2]["file_path"]],
                    val_filenames=[frames[0]["file_path"]], test_filenames=[frames[1]["file_path"]])
        mapped, members = M.source_frames(self.root, meta, "dev")
        self.assertEqual(members, {self.root / frames[0]["file_path"]})
        self.assertEqual(mapped[self.root / frames[0]["file_path"]]["group_id"], "0")
        frames[1]["group_id"] = "0"
        with self.assertRaisesRegex(ValueError, "camera groups"):
            M.source_frames(self.root, meta, "dev")
        frames[1]["group_id"] = "1"
        frames[0]["lidar_depth_path"] = "missing.npy"
        with self.assertRaises(FileNotFoundError):
            M.source_frames(self.root, meta, "dev")

    def test_depth_coverage_has_fixed_support_denominator_and_penalizes_holes(self):
        reference = np.array([[2., 4., 0.], [8., np.nan, 10.]])
        support = np.array([[1, 1, 1], [0, 1, 1]])
        valid = np.array([[True, True, True], [True, True, False]])
        depth = np.array([[3., 4., np.inf], [999., 100., 100.]])
        alpha = np.array([[.8, .2, 1.], [1., 1., 1.]])
        result = M.score_depth(reference, support, depth, alpha, valid)
        self.assertEqual(result["map_depth_eligible_pixels"], 2)
        self.assertEqual(result["map_depth_compared_pixels"], 1)
        self.assertEqual(result["map_depth_coverage_fraction"], .5)
        self.assertEqual(result["map_projected_depth_median_abs_m"], 1.)
        self.assertEqual(result["map_projected_depth_rmse_m"], 1.)
        self.assertEqual(result["map_depth_capped_relative_error_all_supported"], .75)
        alpha[0, 0] = .5  # Threshold is strictly greater than .5.
        missing = M.score_depth(reference, support, depth, alpha, valid)
        self.assertEqual(missing["map_depth_eligible_pixels"], 2)
        self.assertEqual(missing["map_depth_missing_pixels"], 2)
        self.assertEqual(missing["map_depth_capped_relative_error_all_supported"], 1.)
        self.assertIsNone(missing["map_projected_depth_rmse_m"])
        alpha[0, :2] = 1.
        depth[0, :2] = [np.nan, 0.]
        self.assertEqual(M.score_depth(reference, support, depth, alpha, valid), missing)

    def test_disappearing_large_depth_errors_cannot_improve_selection_score(self):
        target = np.array([[2., 4.]])
        valid = np.ones((1, 2), bool)
        depth = np.array([[100., 4.]])
        full = M.score_depth(target, valid, depth, np.ones_like(target), valid)
        hole = M.score_depth(target, valid, depth, np.array([[0., 1.]]), valid)
        self.assertEqual(full["map_depth_capped_relative_error_all_supported"], .5)
        self.assertEqual(hole["map_depth_capped_relative_error_all_supported"], .5)
        self.assertLess(hole["map_projected_depth_rmse_m"], full["map_projected_depth_rmse_m"])
        self.assertEqual(hole["map_depth_coverage_fraction"], .5)

    def test_empty_depth_support_is_null_and_invalid_alpha_is_error(self):
        zeros, ones = np.zeros((2, 2)), np.ones((2, 2))
        result = M.score_depth(zeros, ones, ones, ones, ones.astype(bool))
        self.assertEqual(result["map_depth_eligible_pixels"], 0)
        self.assertIsNone(result["map_depth_capped_relative_error_all_supported"])
        self.assertIsNone(result["map_depth_coverage_fraction"])
        for value in (np.nan, np.inf, -.1, 1.1):
            with self.subTest(alpha=value), self.assertRaisesRegex(ValueError, "alpha"):
                M.score_depth(ones, ones, ones, np.full((2, 2), value), ones.astype(bool))
        with self.assertRaisesRegex(ValueError, "H/W"):
            M.score_depth(ones, ones, ones[:1], ones, ones.astype(bool))

    @unittest.skipUnless(importlib.util.find_spec("skimage"), "skimage absent in this CPU interpreter")
    def test_masked_rgb_and_ssim_match_existing_evaluator_exactly(self):
        import cv2
        from skimage.metrics import structural_similarity
        gt = np.full((11, 13, 3), .25, np.float32)
        pred = gt + .1
        valid = np.ones((11, 13), bool)
        score = M.score_rgb(gt, pred, valid)
        self.assertAlmostEqual(score["psnr_db"], 20., places=5)
        self.assertEqual(score["ssim_compared_pixels"], 143)
        self.assertAlmostEqual(score["ssim_valid_pixels"],
                               structural_similarity(gt, pred, data_range=1., channel_axis=-1), places=6)
        valid[:, :2] = False
        pred[:, :2] = 1.
        masked = M.score_rgb(gt, pred, valid)
        self.assertAlmostEqual(masked["psnr_db"], 20., places=5)
        self.assertEqual(masked["ssim_compared_pixels"], 88)
        self.assertAlmostEqual(masked["ssim_valid_pixels"], score["ssim_valid_pixels"], places=6)
        # Execute the original metric statements, rather than reimplementing
        # their float precision, erosion and SSIM border behavior in the test.
        tree = ast.parse((PROJECT / "scripts/evaluate_reconstruction.py").read_text())
        original_statements = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("mse", "eroded", "ssim"):
                    original_statements.append(node)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
                if any(isinstance(name, ast.Name) and name.id == "ssim_map" for name in node.targets[0].elts):
                    original_statements.append(node)
        # ast.walk visits these sibling assignments in source order.
        scope = {"gt": gt, "pred": pred, "valid": valid, "np": np,
                 "cv2": cv2, "structural_similarity": structural_similarity}
        exec(compile(ast.Module(body=original_statements, type_ignores=[]), "existing_rgb_metrics", "exec"), scope)
        self.assertEqual(masked["rgb_mse"], scope["mse"])
        self.assertEqual(masked["ssim_valid_pixels"], scope["ssim"])
        self.assertEqual(masked["psnr_db"], float(-10 * np.log10(max(scope["mse"], 1e-12))))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            M.score_rgb(gt, pred, np.zeros_like(valid))

    @unittest.skipUnless(importlib.util.find_spec("skimage"), "skimage absent in this CPU interpreter")
    def test_tiny_valid_masks_report_unscorable_ssim_explicitly(self):
        gt = np.ones((8, 8, 3), np.float32) / 2
        valid = np.eye(8, dtype=bool)
        score = M.score_rgb(gt, gt, valid)
        self.assertEqual(score["psnr_db"], 120.)
        self.assertIsNone(score["ssim_valid_pixels"])
        self.assertEqual(score["ssim_compared_pixels"], 0)
        self.assertEqual(M.alpha_coverage(np.ones((8, 8, 1)), valid)["alpha_gt_0_5_valid_pixel_fraction"], 1.)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch absent; schema helpers remain usable")
    def test_cpu_replacement_on_real_backend_model_preserves_fixed_state(self):
        from orbvi_gaussian.training import confined_environment
        os.environ.update(confined_environment())
        tempfile.tempdir = os.environ["TMPDIR"]
        import torch
        from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModel
        # Construct only the real model's CPU parameter container; no module
        # initialization, GPU kernels, rendering or perceptual downloads.
        model = LidarSplatfactoModel.__new__(LidarSplatfactoModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(use_inria_surfel_renderer=True, sh_degree=3)
        model.gauss_params = torch.nn.ParameterDict({
            key: torch.nn.Parameter(torch.zeros((1, 15, 3) if key == "features_rest" else (1, width)))
            for key, width in {**M.PARAMETER_WIDTHS, "features_rest": 3}.items()})
        fixed_state = torch.nn.Parameter(torch.tensor([.125]), requires_grad=False)
        model.register_parameter("fixed_rig_fixture", fixed_state)
        model.eval()
        path, original = self.exported()
        parameters, _ = M.read_candidate(path)
        M.replace_gauss_params(model, parameters, 3, torch)
        self.assertIs(model.fixed_rig_fixture, fixed_state)
        self.assertFalse(model.training)
        for key, expected in parameters.items():
            np.testing.assert_array_equal(model.gauss_params[key].detach().numpy(), expected)
            self.assertFalse(model.gauss_params[key].requires_grad)
        np.testing.assert_array_equal(model.features_rest.detach().numpy(), original.features_rest.value)
        model.train()
        with self.assertRaisesRegex(ValueError, "inference"):
            M.replace_gauss_params(model, parameters, 3, torch)


if __name__ == "__main__":
    unittest.main()

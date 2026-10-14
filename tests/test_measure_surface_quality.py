"""CPU regression tests for fixed-support geometry and exact PLY statistics."""
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_surface_quality", ROOT/"scripts/measure_surface_quality.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="surface_measure_cpu_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = dict(w=3, h=2, fl_x=4., fl_y=4., cx=1.5, cy=1., physical_camera_id=2,
                           camera_to_worlds_ns=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 2.]])

    def ply(self, xyz, opacity, scales, name="model.ply"):
        names = list(M.inspect_ply.__globals__["REQUIRED"])
        rows = np.zeros((len(xyz), len(names)), dtype="<f4")
        for i, key in enumerate(("x", "y", "z")):
            rows[:, names.index(key)] = np.asarray(xyz)[:, i]
        rows[:, names.index("opacity")] = opacity
        rows[:, names.index("rot_0")] = 1
        for i in range(3):
            rows[:, names.index("scale_%d" % i)] = np.log(np.asarray(scales)[:, i])
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz) +
                  "".join("property float %s\n" % k for k in names)+"end_header\n").encode()
        path = self.root/name
        path.write_bytes(header+rows.tobytes())
        return path

    def fixture(self):
        dataset = self.root/"block_000"
        dataset.mkdir()
        for name in ("image", "mask", "support"):
            Image.fromarray(np.full((2, 3), 255, np.uint8)).save(dataset/(name+".png"))
        depth = np.full((2, 3), 2, np.float32)
        np.save(dataset/"depth.npy", depth)
        xyz = M.unproject(self.camera, np.arange(6), depth.ravel(), .5).astype(np.float32)
        normals = np.array([[0, 0, 1]]*3 + [[1, 0, 0]]*3, np.float32)
        np.savez(dataset/"lidar_reference.npz", xyz=xyz, normals=normals, normal_valid=np.ones(6, bool))
        frame = {k: v for k, v in self.camera.items() if k not in ("physical_camera_id", "camera_to_worlds_ns")}
        frame.update(file_path="image.png", mask_path="mask.png", lidar_depth_path="depth.npy",
                     lidar_depth_mask_path="support.png", camera_id="camera_2", group_id="123", timestamp_ns=123,
                     transform_matrix=[*self.camera["camera_to_worlds_ns"], [0, 0, 0, 1]])
        transforms = dataset/"transforms.json"
        transforms.write_text(json.dumps({"frames": [frame], "val_filenames": ["image.png"],
            "pixel_center_convention": "half_integer; principal point +0.5 from OpenCV rectification"}))
        row = dict(dataset_index=0, group_id="123", camera_id="camera_2", camera=self.camera,
                   file_path="image.png", source_image_path=str(dataset/"image.png"),
                   source_mask_path=str(dataset/"mask.png"), source_lidar_depth_path=str(dataset/"depth.npy"),
                   source_lidar_support_path=str(dataset/"support.png"))
        report = dict(status="evaluated", split="dev", images=1, frames=[row], backend="hbb1-original-2dgs",
                      transforms=str(transforms), transforms_sha256=M.file_sha256(transforms),
                      world_parser_transform=np.eye(4)[:3].tolist(), world_parser_scale=1.)
        for role in ("baseline", "candidate"):
            folder = self.root/role
            folder.mkdir()
            (folder/"metrics.json").write_text(json.dumps(report))
        self.save_render("baseline", alpha=[1, .4, 1, 1, 1, 1], depth=[2, 2, 3, 2, 3, 3])
        self.save_render("candidate", alpha=[1, .4, 1, 1, 1, 1], depth=[2, 2, 2, 2, 2, 2])
        return dataset/"lidar_reference.npz"

    def save_render(self, role, alpha, depth, dtype=np.float32):
        np.savez(self.root/role/"000_float.npz", rgb=np.ones((2, 3, 3), dtype),
                 alpha=np.asarray(alpha, dtype).reshape(2, 3, 1), depth=np.asarray(depth, dtype).reshape(2, 3, 1))

    def measure_fixture(self, reference, **kwargs):
        return M.measure_renders(self.root/"candidate/metrics.json", reference, M.DEFAULT_SETTINGS.copy(),
                                 baseline_metrics=self.root/"baseline/metrics.json", **kwargs)

    def test_exact_voxel_counts_are_not_gaussian_weighted_or_footprints(self):
        # 4, 1, and 2 records in three occupied voxels. Median must be 2, not 4.
        xyz = [[x, 0, 0] for x in (.001, .003, .005, .007, .03, -.003, -.007)]
        path = self.ply(xyz, [0, 1, 2, -1, -2, -1e-9, 0], [[.02, .04, .9]]*7)
        original = path.read_bytes()
        result = M.measure_ply(path)
        self.assertEqual(result["occupied_voxels"], 3)
        self.assertEqual(result["voxel_2cm_multiplicity_median"], 2)
        self.assertAlmostEqual(result["voxel_2cm_multiplicity_mean"], 7/3)
        self.assertAlmostEqual(result["opacity_frac_lt_0_5"], 3/7)  # tiny negative rounds sigmoid to .5
        self.assertAlmostEqual(result["opacity_median"], .5)
        self.assertAlmostEqual(result["tangential_radius_p95_m"], .04)
        self.assertAlmostEqual(result["tangential_radius_max_m"], .04)  # axis 2 MUST NOT be sorted in
        np.testing.assert_allclose(result["tangential_first_two_axes_p95_m"], [.02, .04], rtol=1e-6)
        self.assertEqual(path.read_bytes(), original)

    def test_voxel_translation_and_negative_coordinates_use_floor(self):
        cells = np.array([[0, 0, -1], [0, 0, -1], [-1, 0, 0], [0, 1, 0], [0, 1, 0]], np.int32)
        self.assertEqual(sorted(M.exact_voxel_counts(cells).tolist()), [1, 2, 2])

    def test_radius_cap_uses_stored_log_float64_exp_without_cpu_gpu_ulp_false_failure(self):
        path = self.ply([[0, 0, 0]], [1], [[.12, .12, .01]])
        report = M.measure_ply(path)
        expected = float(np.exp(np.float64(np.float32(np.log(.12)))))
        self.assertEqual(report["tangential_radius_max_m"], expected)
        self.assertLessEqual(report["tangential_radius_max_m"], .12)
        self.assertGreater(report["tangential_radius_p95_m"], .03)

    def test_ply_preflight_budget_and_corrupt_tail(self):
        path = self.ply([[0, 0, 0], [1, 1, 1]], [0, 0], [[.02]*3]*2)
        with self.assertRaisesRegex(ValueError, "max-gaussians"):
            M.measure_ply(path, max_gaussians=1)
        with self.assertRaisesRegex(ValueError, "memory"):
            M.measure_ply(path, memory_mib=1)
        path.write_bytes(path.read_bytes()+b"x")
        with self.assertRaisesRegex(ValueError, "payload"):
            M.measure_ply(path)

    def test_ns_half_pixel_and_cv_integer_are_the_same_ray(self):
        camera = self.camera
        points = M.unproject(camera, np.array([0, 4]), [2., 3.], .5)
        cv_x = np.array([0, 1])-(camera["cx"]-.5)
        cv_y = np.array([0, 1])-(camera["cy"]-.5)
        expected = np.column_stack((cv_x/camera["fl_x"]*[2, 3], -cv_y/camera["fl_y"]*[2, 3], [0, -1]))
        np.testing.assert_allclose(points, expected)
        rotated = dict(camera, camera_to_worlds_ns=[[0., -1., 0., 5.], [1., 0., 0., -2.], [0., 0., 1., 2.]])
        p = M.unproject(rotated, np.array([0, 4]), [2., 3.], .5)
        np.testing.assert_allclose(p[:, :2], np.column_stack((-expected[:, 1]+5, expected[:, 0]-2)))

    def test_labels_depend_on_reference_normal_and_height_not_image_position(self):
        ref = {"xyz": np.array([[0, 0, 0], [0, 0, 4], [0, 0, 1], [0, 0, -8], [0, 0, 0]], float),
               "normals": np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0, 1], [0, 0, 1]], float),
               "normal_valid": np.array([1, 1, 1, 1, 0], bool)}
        labels = M.classify_reference(ref, np.arange(5), np.zeros(5), 2., M.DEFAULT_SETTINGS)
        self.assertEqual(labels.tolist(), [0, 2, 1, 2, 3])  # floor, ceiling/other, wall, wrong level, unknown
        far = M.classify_reference(ref, np.arange(5), np.ones(5), 2., M.DEFAULT_SETTINGS)
        self.assertTrue(np.all(far == 3))

    def test_missing_alpha_wrong_layer_invalid_depth_and_unknown_are_distinct(self):
        labels = np.array([0]*6+[1], np.uint8)
        result = M.count_coverage(labels, np.ones(7)*2,
            np.array([1, .5, 1, 1, np.nan, 1, 1]), np.array([2, 2, 4, np.nan, 2, 0, 2]), .15)
        floor = M.finalize_coverage(result)["floor"]
        self.assertEqual([floor[k] for k in M.COUNTS], [6, 1, 1, 2, 1, 1])
        self.assertEqual(floor["status"], "unknown")
        self.assertAlmostEqual(floor["hole_rate"], 5/6)
        self.assertIsNone(floor["alpha_gt_0_5_coverage"])

    def test_all_opaque_wrong_surface_is_not_complete(self):
        counts = M.count_coverage(np.array([0, 1]), np.array([2, 2]), np.ones(2), np.array([10, 10]), .15)
        for label in ("floor", "wall"):
            report = M.finalize_coverage(counts)[label]
            self.assertEqual(report["alpha_gt_0_5_coverage"], 1)
            self.assertEqual(report["hole_rate"], 1)
            self.assertEqual(report["missing_alpha"], 0)
            self.assertEqual(report["wrong_depth"], 1)

    def test_signed_depth_direction_has_fixed_denominator_and_separate_conditional_quantiles(self):
        labels = np.zeros(5, np.uint8)
        counts = M.count_coverage(labels, np.full(5, 2.), np.array([1, 1, 1, 1, .4]),
                                  np.array([1., 2., 3., 4., 100.]), .15)
        floor = M.finalize_coverage(counts)["floor"]
        self.assertEqual((floor["too_near"], floor["too_far"], floor["wrong_depth"]), (1, 2, 3))
        self.assertEqual((floor["too_near_rate"], floor["too_far_rate"]), (.2, .4))
        self.assertAlmostEqual(floor["too_far_given_wrong_fraction"], 2/3)
        self.assertEqual(M.signed_stats([-1., 0., 1., 2.])["p50_m"], .5)
        self.assertIsNone(M.signed_stats([])["p50_m"])
        reference = self.fixture()
        report = self.measure_fixture(reference)
        self.assertEqual(report["summaries"]["baseline"]["signed_camera_z_error"]["floor"]["p50_m"], .5)
        self.assertEqual(report["summaries"]["candidate"]["signed_camera_z_error"]["floor"]["p50_m"], 0)

    def test_same_ply_sidecar_requires_separate_hash_bound_representation(self):
        descriptors = {}
        for name in ("side_flags", "positive_train_groups", "negative_train_groups"):
            p = self.root/(name+".npy")
            np.save(p, np.zeros(2, np.uint8))
            descriptors[name] = {"path": str(p), "sha256": M.file_sha256(p)}
        meta = {"ply": {"sha256": "a"*64, "gaussians": 2}, "side_flags": descriptors["side_flags"],
                "training_evidence": {"train_only": True, "heldout_used": False,
                                      **{k:v for k,v in descriptors.items() if k != "side_flags"}}}
        path = self.root/"sidecar.json"
        path.write_text(json.dumps(meta))
        sha = M.file_sha256(path)
        representation = M.hashlib.sha256(("a"*64+":"+sha).encode()).hexdigest()
        report = {"source_ply_sha256": "a"*64, "gaussians": 2, "representation_id": representation,
                  "direction_support": {"path": str(path), "sha256": sha, "source_ply_sha256": "a"*64,
                                        "source_records": 2, "side_flags_sha256": descriptors["side_flags"]["sha256"]}}
        result = M.representation_sources(report, lambda p: None)
        self.assertEqual(result["representation_id"], representation)
        self.assertNotEqual(result["representation_id"], report["source_ply_sha256"])
        self.assertEqual(len(result["assets"]), 4)
        changed = {**report, "representation_id": "b"*64}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            M.representation_sources(changed, lambda p: None)
        Path(descriptors["side_flags"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "asset changed"):
            M.representation_sources(report, lambda p: None)

    def test_paired_fixture_exact_counts_and_immutable_denominator(self):
        reference = self.fixture()
        before = {p: M.file_sha256(p) for p in self.root.rglob("*") if p.is_file()}
        report = self.measure_fixture(reference)
        self.assertEqual(report["scope"]["fixed_support_pixels"], 6)
        self.assertFalse(report["scope"]["test_read"])
        self.assertEqual(report["scope"]["frame_keys"], [{"group_id": "123", "camera_id": "camera_2"}])
        floor, wall = (report["comparison"]["coverage"][k] for k in ("floor", "wall"))
        self.assertEqual((floor["eligible_pixels"], floor["baseline_hole_pixels"], floor["candidate_hole_pixels"]), (3, 2, 1))
        self.assertAlmostEqual(floor["relative_reduction"], .5)
        self.assertEqual(wall["relative_reduction"], 1.)
        self.assertTrue(report["summaries"]["candidate"]["conditioned_geometry"]["bidirectional_p50_m"] >= 0)
        self.assertFalse(report["summaries"]["candidate"]["conditioned_geometry"]["independent_ground_truth"])
        self.assertIsNone(report["summaries"]["candidate"]["conditioned_geometry"]["floor_front_surface_z_span_p95_m"])
        after = {p: M.file_sha256(p) for p in before}
        self.assertEqual(before, after)
        self.save_render("candidate", [0]*6, [0]*6)
        erased = self.measure_fixture(reference)
        self.assertEqual(report["fixed_denominator_id"], erased["fixed_denominator_id"])
        self.assertEqual(erased["summaries"]["candidate"]["coverage"]["floor"]["hole_rate"], 1)
        self.assertIsNone(erased["summaries"]["candidate"]["conditioned_geometry"]["bidirectional_p50_m"])

    def test_missing_archive_never_passes_or_disappears_from_denominator(self):
        reference = self.fixture()
        (self.root/"candidate/000_float.npz").unlink()
        report = self.measure_fixture(reference)
        floor = report["summaries"]["candidate"]["coverage"]["floor"]
        self.assertEqual((floor["denominator"], floor["unknown_prediction"]), (3, 3))
        self.assertEqual(floor["status"], "unknown")
        self.assertIsNone(report["comparison"]["coverage"]["floor"]["relative_reduction"])
        self.assertEqual(M.gate_evidence(None, report)["G5"]["status"], "unknown")

    def test_unsupported_pixels_remain_unknown_and_are_not_holes_with_zero_depth(self):
        reference = self.fixture()
        support = np.array([[255, 255, 0], [255, 255, 255]], np.uint8)
        Image.fromarray(support).save(reference.parent/"support.png")
        report = self.measure_fixture(reference)
        self.assertEqual(report["scope"]["unsupported_unknown_pixels"], 1)
        self.assertEqual(report["scope"]["fixed_support_pixels"], 5)
        self.assertEqual(report["summaries"]["candidate"]["coverage"]["floor"]["denominator"], 2)

    def test_frozen_ego_mask_overrides_legacy_and_reported_mask(self):
        reference = self.fixture()
        ego = self.root/"ego.png"
        Image.fromarray(np.array([[0, 255, 255], [255, 0, 255]], np.uint8)).save(ego)
        metrics_path = self.root/"candidate/metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["frames"][0]["acceptance_mask_path"] = metrics["frames"][0]["source_mask_path"]
        metrics_path.write_text(json.dumps(metrics))
        frozen = {**metrics["frames"][0], "mask_path": str(ego), "block_id": "block_000"}
        plan = {"split": "dev", "plan_id": "a"*64, "frames": [frozen]}
        plan_path = self.root/"plan.json"
        plan_path.write_text(json.dumps(plan))
        # Plan integrity itself is owned/tested by the acceptance agent.
        with patch("build_acceptance_plan.load_plan", return_value=plan) as loader:
            report = self.measure_fixture(reference, acceptance_plan=plan_path)
        loader.assert_called_once_with(plan_path, verify_assets=True)
        self.assertEqual(report["scope"]["fixed_support_pixels"], 4)
        self.assertEqual(report["protocol"]["mask_protocol"], "ego_v2")
        self.assertEqual(report["summaries"]["candidate"]["coverage"]["floor"]["denominator"], 2)
        self.assertEqual(report["summaries"]["candidate"]["coverage"]["wall"]["denominator"], 2)

    def test_source_drift_during_measurement_rejected(self):
        reference = self.fixture()
        real_read = M.read_render
        def change_source(*args, **kwargs):
            result = real_read(*args, **kwargs)
            with reference.open("ab") as stream:
                stream.write(b"x")
            return result
        with patch.object(M, "read_render", side_effect=change_source):
            with self.assertRaisesRegex(ValueError, "changed during read"):
                self.measure_fixture(reference)

    def test_gate_fields_and_diagnostic_provenance_block(self):
        reference = self.fixture()
        report = self.measure_fixture(reference)
        evidence = M.gate_evidence(None, report)
        self.assertEqual(evidence["G4"]["status"], "unknown")  # no frozen plan/domain
        self.assertEqual(evidence["G5"]["status"], "unknown")
        self.assertTrue(evidence["G5"]["fixed_denominator"])
        self.assertEqual(evidence["G5"]["floor"]["eligible_pixels"], 3)
        self.assertTrue(evidence["G4"]["shared_reference"])
        self.assertGreater(evidence["G4"]["candidate_points"], 0)
        report["gate_compatible_render_provenance"] = True
        good = M.gate_evidence(None, report)
        self.assertEqual(good["G5"]["status"], "known")
        report["gate_compatible_settings"] = False
        self.assertEqual(M.gate_evidence(None, report)["G5"]["status"], "unknown")

    def test_reject_test_camera_drift_and_changed_reference(self):
        reference = self.fixture()
        path = self.root/"candidate/metrics.json"
        original = json.loads(path.read_text())
        for change in (dict(split="test"), dict(world_parser_scale=2)):
            path.write_text(json.dumps({**original, **change}))
            with self.assertRaises(ValueError):
                self.measure_fixture(reference)
        altered = json.loads(json.dumps(original))
        altered["frames"][0]["camera"]["cx"] += 1
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "drift"):
            self.measure_fixture(reference)
        path.write_text(json.dumps(original))
        alien = self.root/"alien.npz"
        np.savez(alien, xyz=np.zeros((1, 3), np.float32), normals=np.ones((1, 3), np.float32), normal_valid=[True])
        with self.assertRaisesRegex(ValueError, "Reference differs"):
            self.measure_fixture(alien)

    def test_nonfloat_and_budget_rejections_before_large_allocation(self):
        reference = self.fixture()
        self.save_render("candidate", [1]*6, [2]*6, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.measure_fixture(reference)
        with self.assertRaisesRegex(ValueError, "pixel budget"):
            self.measure_fixture(reference, max_total_pixels=2)
        with self.assertRaisesRegex(ValueError, "decompressed"):
            M.bounded_npz(reference, 2)

    def test_npz_declared_shape_bomb_and_object_arrays_rejected(self):
        import io
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"shape": (10**12,), "fortran_order": False, "descr": "<f4"})
        path = self.root/"bomb.npz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("depth.npy", header.getvalue())
        with self.assertRaisesRegex(ValueError, "shape/payload"):
            M.bounded_npz(path, 1024)
        np.savez(self.root/"object.npz", depth=np.array([{}], object))
        with self.assertRaisesRegex(ValueError, "objects"):
            M.bounded_npz(self.root/"object.npz", 1024)

    def test_zero_baseline_holes_is_unknown_relative_improvement(self):
        reference = self.fixture()
        self.save_render("baseline", [1]*6, [2]*6)
        report = self.measure_fixture(reference)
        self.assertIsNone(report["comparison"]["coverage"]["floor"]["relative_reduction"])
        self.assertIsNone(report["comparison"]["conditioned_geometry_ratio"])

    def test_cli_creates_serializable_report_and_refuses_overwrite(self):
        path = self.ply([[.001, 0, 0]], [1], [[.01]*3])
        output = self.root/"report.json"
        self.assertEqual(M.main(["--ply", str(path), "--output", str(output)]), 0)
        report = json.loads(output.read_text())
        self.assertFalse(report["high_fidelity_pass"])
        self.assertEqual(report["gate_evidence"]["G3"]["voxel_size_m"], .02)
        old = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "Output exists"):
            M.main(["--ply", str(path), "--output", str(output)])
        self.assertEqual(old, output.read_bytes())

    def test_checkpoint_metrics_with_null_source_ply_auto_discover_export(self):
        reference = self.fixture()
        path = self.root/"candidate/metrics.json"
        report = json.loads(path.read_text())
        report.update(source_ply=None, source_ply_sha256=None, checkpoint_sha256="a"*64,
                      parameter_origin="trained_checkpoint", inference_model_step=11999, export={"gaussians": 1})
        path.write_text(json.dumps(report))
        self.ply([[0, 0, 0]], [1], [[.01]*3], name="candidate/gaussians.ply")
        output = self.root/"native_checkpoint_surface.json"
        self.assertEqual(M.main(["--metrics", str(path), "--reference", str(reference), "--output", str(output)]), 0)
        measured = json.loads(output.read_text())
        self.assertEqual(measured["ply"]["gaussians"], 1)
        self.assertEqual(measured["candidate_ply_render_binding"]["status"], "checkpoint_export_count_only")
        self.assertEqual(measured["candidate_ply_render_binding"]["inference_model_step"], 11999)
        self.assertFalse(measured["candidate_ply_render_binding"]["checkpoint_export_tensor_equality_verified_here"])

    def checkpoint_fixture(self):
        """Synthetic verifier evidence tests binding only, not tensor verification."""
        reference = self.fixture()
        path = self.root/"candidate/metrics.json"
        model = self.ply([[0, 0, 0]], [1], [[.01]*3], name="candidate/gaussians.ply")
        checkpoint, config = self.root/"step-000011999.ckpt", self.root/"config.yml"
        checkpoint.write_bytes(b"unit-test checkpoint identity; no torch deserialization")
        config.write_text("test: binding_only\n")
        report = json.loads(path.read_text())
        report.update(source_ply=None, source_ply_sha256=None, checkpoint=str(checkpoint), config=str(config),
            checkpoint_sha256=M.file_sha256(checkpoint), config_sha256=M.file_sha256(config),
            parameter_origin="trained_checkpoint", checkpoint_is_template=False, step=11999,
            inference_model_step=11999, gaussians=2, export={"gaussians": 1},
            metric_domain="float_renderer", acceptance_plan_id="a"*64)
        path.write_text(json.dumps(report))
        baseline = self.root/"baseline/metrics.json"
        base = json.loads(baseline.read_text())
        base.update(metric_domain="float_renderer", acceptance_plan_id="a"*64, source_ply_sha256="b"*64)
        baseline.write_text(json.dumps(base))
        proof = dict(schema_version=2, status="checkpoint_verified", incomplete_reasons=[], block_id="block_000",
            export_verified=True, all_tensors_finite=True, verification_scope="checkpoint_optimizer_and_export_authenticity_only",
            parameter_origin="trained_checkpoint", checkpoint_is_template=False, step=11999, inference_model_step=11999,
            completed_iterations=12000, total_iterations=12000, remaining_iterations=0, gaussians=2,
            checkpoint=str(checkpoint), config=str(config), checkpoint_sha256=M.file_sha256(checkpoint),
            export=dict(path=str(model), sha256=M.file_sha256(model), gaussians=1,
                verification="all field bytes exact; native opacity cutoff -5.5373; normalized wxyz; actual normals"),
            source_sha256={str(p): M.file_sha256(p) for p in (path, checkpoint, model, config, Path(report["transforms"]))})
        proof_path = self.root/"proof.json"
        proof_path.write_text(json.dumps(proof))
        plan = {"split": "dev", "plan_id": "a"*64, "frames": [{**report["frames"][0],
                "mask_path": report["frames"][0]["source_mask_path"], "block_id": "block_000"}]}
        plan_path = self.root/"plan.json"
        plan_path.write_text(json.dumps(plan))
        return reference, M.measure_ply(model), proof_path, plan

    def test_exact_checkpoint_proof_clears_only_provenance_unknown(self):
        reference, ply, proof_path, plan = self.checkpoint_fixture()
        with patch("build_acceptance_plan.load_plan", return_value=plan):
            old = self.measure_fixture(reference, acceptance_plan=self.root/"plan.json")
            new = self.measure_fixture(reference, acceptance_plan=self.root/"plan.json",
                                       checkpoint_proof=proof_path, candidate_ply=ply)
            output = self.root/"rev2.json"
            M.main(["--metrics", str(self.root/"candidate/metrics.json"), "--reference", str(reference),
                    "--baseline-metrics", str(self.root/"baseline/metrics.json"),
                    "--acceptance-plan", str(self.root/"plan.json"), "--checkpoint-proof", str(proof_path),
                    "--output", str(output)])
        for gate in ("G4", "G5"):
            self.assertEqual(M.gate_evidence(ply, old)[gate]["status"], "unknown")
            self.assertEqual(M.gate_evidence(ply, new)[gate]["status"], "known")
        self.assertEqual(old["summaries"], new["summaries"])
        self.assertEqual(old["fixed_denominator_id"], new["fixed_denominator_id"])
        self.assertEqual(new["render_provenance_unknown_reasons"], [])
        origin = new["source_evaluator_provenance"]["candidate"]
        self.assertIsNone(origin["source_ply_sha256"])  # do not invent PLY-based inference
        self.assertEqual(origin["checkpoint_export_proof"]["export"]["sha256"], ply["sha256"])
        measured = json.loads(output.read_text())
        self.assertEqual(measured["candidate_model_sha256"], ply["sha256"])
        self.assertEqual(measured["candidate_representation_id"], origin["checkpoint_sha256"])
        binding = measured["candidate_ply_render_binding"]
        self.assertEqual(binding["status"], "checkpoint_export_exact_proof_verified")
        self.assertTrue(binding["checkpoint_export_tensor_equality_verified_by_cpu_proof"])
        self.assertFalse(binding["checkpoint_export_tensor_equality_verified_here"])
        self.assertFalse(measured["high_fidelity_pass"])

    def test_checkpoint_proof_rejects_incomplete_identity_step_and_count_evidence(self):
        reference, ply, proof_path, _ = self.checkpoint_fixture()
        original = json.loads(proof_path.read_text())
        report = json.loads((self.root/"candidate/metrics.json").read_text())
        bad_cases = [dict(schema_version=1), dict(status="incomplete_evidence"), dict(export_verified=False),
            dict(export_verified=1), dict(all_tensors_finite=False), dict(incomplete_reasons=["missing"]),
            dict(parameter_origin="source_ply"), dict(checkpoint_is_template=True), dict(step=12000),
            dict(inference_model_step=30000), dict(remaining_iterations=1), dict(gaussians=3),
            dict(checkpoint_sha256="f"*64), dict(checkpoint=str(self.root/"alien.ckpt")),
            dict(config=str(self.root/"alien.yml")), dict(block_id="block_099"),
            dict(export={**original["export"], "sha256": "f"*64}),
            dict(export={**original["export"], "path": str(self.root/"alien.ply")}),
            dict(export={**original["export"], "gaussians": 2}),
            dict(export={**original["export"], "verification": "count only"})]
        for change in bad_cases:
            with self.subTest(change=change):
                proof_path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    M.validate_checkpoint_proof(proof_path, self.root/"candidate/metrics.json", report, ply, lambda p: None)
        proof_path.write_text(json.dumps(original))
        for change in (dict(parameter_origin="source_ply"), dict(checkpoint_is_template=True), dict(step=1),
                       dict(inference_model_step=30000), dict(source_ply={"path": ply["path"]}),
                       dict(gaussians=1), dict(config_sha256="f"*64)):
            with self.subTest(report_change=change), self.assertRaises(ValueError):
                M.validate_checkpoint_proof(proof_path, self.root/"candidate/metrics.json", {**report, **change}, ply, lambda p: None)

    def test_checkpoint_proof_rejects_missing_required_source_hashes_and_tampering(self):
        _, ply, proof_path, _ = self.checkpoint_fixture()
        original = json.loads(proof_path.read_text())
        metrics_path = self.root/"candidate/metrics.json"
        report = json.loads(metrics_path.read_text())
        extra = self.root/"extra.txt"
        extra.write_text("supplementary source")
        sources = {**original["source_sha256"], str(extra): M.file_sha256(extra)}
        for source in original["source_sha256"]:
            with self.subTest(missing=source):
                proof_path.write_text(json.dumps({**original, "source_sha256": {p: h for p, h in sources.items() if p != source}}))
                with self.assertRaisesRegex(ValueError, "SHA"):
                    M.validate_checkpoint_proof(proof_path, metrics_path, report, ply, lambda p: None)
        proof_path.write_text(json.dumps({**original, "source_sha256": sources}))
        for source in sources:
            path = Path(source)
            raw = path.read_bytes()
            with self.subTest(tampered=source):
                path.write_bytes(raw+b" ")
                with self.assertRaisesRegex(ValueError, "SHA changed"):
                    M.validate_checkpoint_proof(proof_path, metrics_path, report, ply, lambda p: None)
                path.write_bytes(raw)

    def test_checkpoint_sources_are_watched_until_render_measurement_finishes(self):
        reference, ply, proof_path, _ = self.checkpoint_fixture()
        actual_read = M.read_render
        def tamper_after_proof(*args, **kwargs):
            result = actual_read(*args, **kwargs)
            with (self.root/"step-000011999.ckpt").open("ab") as stream:
                stream.write(b"changed during render read")
            return result
        with patch.object(M, "read_render", side_effect=tamper_after_proof):
            with self.assertRaisesRegex(ValueError, "changed during read"):
                self.measure_fixture(reference, checkpoint_proof=proof_path, candidate_ply=ply)

    def renderer_fixture(self):
        """Synthetic files test metadata integrity; they are never imported."""
        def asset(name, content):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return path
        def js(name, value):
            return asset(name, json.dumps(value))
        directory = self.root/"runtime/candidate"
        names = ["cuda_rasterizer/forward.cu", "cuda_rasterizer/backward.cu", "cuda_rasterizer/rasterizer_impl.cu",
                 "cuda_rasterizer/auxiliary.h", "rasterize_points.cu", "rasterize_points.h", "ext.cpp", "setup.py"]
        hashes = {n: M.file_sha256(asset("runtime/candidate/"+n, "test source "+n)) for n in names}
        asset("runtime/inria2dgs_src/cuda_rasterizer/forward.cu", "test source cuda_rasterizer/forward.cu")
        integration_sources = {name: M.file_sha256(asset(name, "test integration source "+name)) for name in (
            "src/orbvi_gaussian/inria_surfel_backend.py", "src/orbvi_gaussian/lidar_splatfacto.py",
            "src/orbvi_gaussian/scale_annealing.py",
            "scripts/evaluate_reconstruction.py", "scripts/evaluate_gaussian_ply.py")}
        wrapper = asset("runtime/candidate/package/__init__.py", "test wrapper")
        extension = asset("runtime/candidate/package/_C.so", "test candidate binary")
        original_wrapper = asset("runtime/original/__init__.py", "test wrapper")
        original_extension = asset("runtime/original/_C.so", "test original binary")
        fd = js("reports/fd.json", dict(candidate_status="PASS", gpu_used=True,
            forward_arrays_bitwise_equal=True, rgb_alpha_parameter_gradients_bitwise_equal=True,
            candidate=dict(module="diff_surfel_rasterization_aa_depth_vjp", extension_path=str(extension), extension_sha256=M.file_sha256(extension)),
            original=dict(extension_path=str(original_extension), wrapper_path=str(original_wrapper), extension_sha256=M.file_sha256(original_extension))))
        validation = js("runtime/candidate/validation.json", dict(status="LOCAL_CUDA_DEPTH_VJP_FINITE_DIFFERENCE_PASS",
            report=str(fd), report_sha256=M.file_sha256(fd), forward_bitwise_equal=True, rgb_alpha_parameter_gradients_bitwise_equal=True,
            candidate_extension_sha256=M.file_sha256(extension), original_extension_sha256=M.file_sha256(original_extension),
            forward_source_sha256=hashes["cuda_rasterizer/forward.cu"]))
        manifest = js("runtime/candidate/manifest.json", dict(package="diff_surfel_rasterization_aa_depth_vjp",
            candidate_extension=str(extension), candidate_extension_sha256=M.file_sha256(extension),
            candidate_source_sha256=hashes, original_source_sha256=hashes, forward_byte_identical=True, installed_original_replaced=False))
        adapter = self.root/"src/orbvi_gaussian/inria_surfel_backend.py"
        provenance = dict(schema_version=1, variant="aa-depth-vjp", module="diff_surfel_rasterization_aa_depth_vjp",
            adapter_path=str(adapter), adapter_sha256=M.file_sha256(adapter), wrapper_path=str(wrapper), wrapper_sha256=M.file_sha256(wrapper),
            extension_path=str(extension), extension_sha256=M.file_sha256(extension), source_directory=str(directory), source_sha256=hashes,
            forward_contract="original ray-plane depth and low-pass coverage; no center-depth fallback",
            validation=dict(validation_path=str(validation), validation_sha256=M.file_sha256(validation),
                finite_difference_report=str(fd), finite_difference_report_sha256=M.file_sha256(fd),
                build_manifest_path=str(manifest), build_manifest_sha256=M.file_sha256(manifest),
                forward_source_byte_identical=True, tested_forward_arrays_bitwise_equal=True,
                tested_rgb_alpha_parameter_gradients_bitwise_equal=True, original_extension_sha256=M.file_sha256(original_extension)))
        js("reports/inria_backend_integration_gpu_v1.json", dict(status="PASS", gpu_used=True,
            parameters_restored_bitwise=True, forward_model_images_bitwise_equal=True, rgb_alpha_parameter_gradients_bitwise_equal=True,
            cases=[dict(variant="aa-depth-vjp", renderer_provenance=provenance)], source_sha256=integration_sources))
        return dict(backend="hbb1-aa-depth-vjp-2dgs", renderer_provenance=provenance)

    def test_verified_backend_retains_identity_and_fixed_geometry_without_importing_torch(self):
        reference = self.fixture()
        original = self.measure_fixture(reference)
        candidate = self.renderer_fixture()
        path = self.root/"candidate/metrics.json"
        report = json.loads(path.read_text())
        report.update(candidate)
        path.write_text(json.dumps(report))
        had_torch = "torch" in sys.modules
        with patch.object(M, "ROOT", self.root):
            measured = self.measure_fixture(reference)
        self.assertEqual("torch" in sys.modules, had_torch)
        self.assertEqual(measured["summaries"], original["summaries"])
        self.assertEqual(measured["fixed_denominator_id"], original["fixed_denominator_id"])
        origin = measured["source_evaluator_provenance"]["candidate"]
        self.assertEqual(origin["backend"], "hbb1-aa-depth-vjp-2dgs")
        self.assertEqual(origin["renderer_provenance"], candidate["renderer_provenance"])
        self.assertEqual(origin["renderer_validation"]["status"], "verified")
        self.assertGreaterEqual(len(origin["renderer_validation"]["assets"]), 20)
        self.assertFalse(measured["gate_compatible_render_provenance"])  # other missing provenance remains unknown

    def test_backend_missing_contract_mislabeling_and_changed_artifacts_rejected(self):
        candidate = self.renderer_fixture()
        provenance = candidate["renderer_provenance"]
        with patch.object(M, "ROOT", self.root):
            for patch_report in [dict(backend="unknown"), dict(backend="hbb1-original-2dgs"), dict(renderer_provenance=None)]:
                with self.subTest(patch_report=patch_report), self.assertRaises(ValueError):
                    M.validate_renderer_provenance({**candidate, **patch_report})
            for change in [dict(schema_version=2), dict(variant="original"), dict(forward_contract="center-depth fallback"),
                dict(extension_sha256="f"*64), dict(source_sha256={}),
                dict(validation={**provenance["validation"], "forward_source_byte_identical": False}),
                dict(validation={**provenance["validation"], "tested_forward_arrays_bitwise_equal": 1}),
                dict(validation={**provenance["validation"], "finite_difference_report_sha256": "f"*64})]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    M.validate_renderer_provenance({**candidate, "renderer_provenance": {**provenance, **change}})
            valid = M.validate_renderer_provenance(candidate)
            for source in valid["assets"]:
                path = Path(source)
                raw = path.read_bytes()
                path.write_bytes(raw+b" ")
                with self.subTest(tampered=source):
                    # Integration JSON itself has no external SHA until measured;
                    # a semantic change is required to invalidate its assertion.
                    if path.name == "inria_backend_integration_gpu_v1.json":
                        d = json.loads(raw);d["status"] = "FAIL";path.write_text(json.dumps(d))
                    with self.assertRaises(ValueError):
                        M.validate_renderer_provenance(candidate)
                path.write_bytes(raw)

    def test_renderer_sources_watched_until_measurement_finishes(self):
        reference = self.fixture()
        candidate = self.renderer_fixture()
        path = self.root/"candidate/metrics.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **candidate}))
        original_read = M.read_render
        def mutate(*args, **kwargs):
            result = original_read(*args, **kwargs)
            with Path(candidate["renderer_provenance"]["extension_path"]).open("ab") as stream:
                stream.write(b"drift")
            return result
        with patch.object(M, "ROOT", self.root), patch.object(M, "read_render", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during read"):
                self.measure_fixture(reference)

    def test_optional_integration_proof_selects_hash_pinned_v2_and_preserves_legacy(self):
        candidate = self.renderer_fixture()
        old = self.root/"reports/inria_backend_integration_gpu_v1.json"
        newer = self.root/"reports/inria_backend_integration_gpu_v2.json"
        newer.write_bytes(old.read_bytes())
        with patch.object(M, "ROOT", self.root):
            legacy = M.validate_renderer_provenance(candidate)
            self.assertEqual(legacy["integration_proof"]["path"], str(old))
            # Explicit v1 must be identical to the existing absent-field path.
            explicit_old = {**candidate, "renderer_integration_proof": {"path": str(old), "sha256": M.file_sha256(old)}}
            self.assertEqual(M.validate_renderer_provenance(explicit_old), legacy)
            # Optional proof is top-level evaluator metadata, not a modified
            # renderer_provenance object or an implicit search for newer files.
            pinned = {**candidate, "renderer_integration_proof": {"path": str(newer.relative_to(self.root)), "sha256": M.file_sha256(newer)}}
            old.unlink()
            visited = []
            result = M.validate_renderer_provenance(pinned, visited.append)
            self.assertEqual(result["integration_proof"], {"path": str(newer), "sha256": M.file_sha256(newer)})
            self.assertIn(newer, visited)
            self.assertNotIn(str(old), result["assets"])
            with self.assertRaises(FileNotFoundError):
                M.validate_renderer_provenance(candidate)  # never auto-discover v2

    def test_optional_integration_proof_rejects_hash_status_module_and_source_mismatch(self):
        candidate = self.renderer_fixture()
        newer = self.root/"reports/inria_backend_integration_gpu_v2.json"
        original = json.loads((self.root/"reports/inria_backend_integration_gpu_v1.json").read_text())
        newer.write_text(json.dumps(original))
        proof = {"path": str(newer), "sha256": M.file_sha256(newer)}
        with patch.object(M, "ROOT", self.root):
            for malformed in (None, {}, {"path": str(newer)}, {"path": str(newer), "sha256": None},
                              {"path": str(newer), "sha256": "f"*64}):
                with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                    M.validate_renderer_provenance({**candidate, "renderer_integration_proof": malformed})
            # Rehash the bad reports so that content checks, not merely the
            # outer digest, must reject status/module/source inconsistencies.
            for key in ("status", "module", "source", "missing_source", "model_source_drift", "binary"):
                changed = json.loads(json.dumps(original))
                restore = None
                if key == "status":
                    changed["status"] = "FAIL"
                elif key in ("module", "binary"):
                    field = "module" if key == "module" else "extension_sha256"
                    changed["cases"][0]["renderer_provenance"][field] = "wrong" if key == "module" else "f"*64
                elif key == "source":
                    changed["source_sha256"]["src/orbvi_gaussian/lidar_splatfacto.py"] = "f"*64
                elif key == "missing_source":
                    del changed["source_sha256"]["src/orbvi_gaussian/lidar_splatfacto.py"]
                else:
                    path = self.root/"src/orbvi_gaussian/lidar_splatfacto.py"
                    restore = (path, path.read_bytes())
                    path.write_bytes(restore[1]+b"drift")
                newer.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    M.validate_renderer_provenance({**candidate, "renderer_integration_proof": {
                        **proof, "sha256": M.file_sha256(newer)}})
                if restore:
                    restore[0].write_bytes(restore[1])

    def test_scale_helper_required_rehashed_and_watched_in_integration_proof(self):
        candidate = self.renderer_fixture()
        relative = "src/orbvi_gaussian/scale_annealing.py"
        helper = self.root/relative
        helper_bytes = helper.read_bytes()
        proof_path = self.root/"reports/inria_backend_integration_gpu_v4.json"
        original = json.loads((self.root/"reports/inria_backend_integration_gpu_v1.json").read_text())
        with patch.object(M, "ROOT", self.root):
            for mode in ("valid", "missing", "wrong_hash", "source_drift"):
                changed = json.loads(json.dumps(original))
                helper.write_bytes(helper_bytes)
                if mode == "missing":
                    del changed["source_sha256"][relative]
                elif mode == "wrong_hash":
                    changed["source_sha256"][relative] = "f"*64
                elif mode == "source_drift":
                    helper.write_bytes(helper_bytes+b"changed")
                proof_path.write_text(json.dumps(changed))
                report = {**candidate, "renderer_integration_proof": {
                    "path": str(proof_path), "sha256": M.file_sha256(proof_path)}}
                with self.subTest(mode=mode):
                    if mode == "valid":
                        watched = []
                        result = M.validate_renderer_provenance(report, watched.append)
                        self.assertEqual(result["assets"][str(helper)], M.file_sha256(helper))
                        self.assertIn(helper, watched)
                    else:
                        with self.assertRaisesRegex(ValueError, "incomplete|SHA changed"):
                            M.validate_renderer_provenance(report)

    def test_probe_records_actual_scale_helper_hash_without_loading_cuda(self):
        path = ROOT/"scripts/probe_inria_backend_integration.py"
        spec = importlib.util.spec_from_file_location("integration_probe_sources_cpu", path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        relative = "src/orbvi_gaussian/scale_annealing.py"
        self.assertEqual(probe.integration_source_hashes()[relative], M.file_sha256(ROOT/relative))

    def test_optional_integration_proof_rejects_project_escape_and_is_watched(self):
        reference = self.fixture()
        candidate = self.renderer_fixture()
        newer = self.root/"reports/inria_backend_integration_gpu_v2.json"
        newer.write_bytes((self.root/"reports/inria_backend_integration_gpu_v1.json").read_bytes())
        proof = {"path": str(newer), "sha256": M.file_sha256(newer)}
        with tempfile.TemporaryDirectory(prefix="surface_external_proof_") as other:
            external = Path(other)/"proof.json"
            external.write_bytes(newer.read_bytes())
            link = self.root/"reports/external_link.json"
            link.symlink_to(external)
            with patch.object(M, "ROOT", self.root):
                for path in (str(external), str(Path("..")/external.parent.name/external.name), str(link)):
                    with self.subTest(path=path), self.assertRaisesRegex(ValueError, "project-local"):
                        M.validate_renderer_provenance({**candidate, "renderer_integration_proof": {**proof, "path": path}})
        metrics = self.root/"candidate/metrics.json"
        metrics.write_text(json.dumps({**json.loads(metrics.read_text()), **candidate, "renderer_integration_proof": proof}))
        actual_read = M.read_render
        def mutate(*args, **kwargs):
            result = actual_read(*args, **kwargs)
            newer.write_bytes(newer.read_bytes()+b" ")
            return result
        with patch.object(M, "ROOT", self.root), patch.object(M, "read_render", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during read"):
                self.measure_fixture(reference)


if __name__ == "__main__":
    unittest.main()

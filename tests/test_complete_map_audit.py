"""CPU audit regressions. Fixtures live in memory; no disk/GPU/training writes."""
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("complete_map_audit", Path(__file__).resolve().parents[1] / "scripts/audit_complete_map.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

NAMES = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
         + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])


def gaussian_bytes(count=5, change=None, names=None, encoding="binary_little_endian"):
    names = names or NAMES
    endian = ">" if encoding == "binary_big_endian" else "<"
    rows = np.zeros(count, dtype=[(n, endian+"f4") for n in names])
    if "rot_0" in names:
        rows["rot_0"] = 1
    rows["x"] = np.arange(count)
    if change:
        change(rows)
    header = "ply\nformat " + encoding + " 1.0\nelement vertex " + str(count) + "\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    payload = ("".join(" ".join(str(float(row[n])) for n in names) + "\n" for row in rows).encode()
               if encoding == "ascii" else rows.tobytes())
    return header.encode() + payload


def config_fixture():
    dataset = audit.ROOT / "data/garage_surface_v1/block_000"
    parser = {"data": str(dataset), "center_method": "none", "orientation_method": "none",
              "auto_scale_poses": "false", "scale_factor": "1.0", "load_3D_points": "true"}
    model = {"_target": "orbvi_gaussian.lidar_splatfacto.LidarSplatfactoModel", "random_init": "false",
             "surface_splats": "true", "use_inria_surfel_renderer": "true", "output_depth_during_training": "true",
             "lidar_depth_loss_weight": ".5", "allow_depth_missing": "false", "camera_optimizer": {"mode": "off"},
             "lidar_reference_path": str(dataset / "lidar_reference.npz")}
    return dataset, {"max_num_iterations": "12000", "pipeline": {"datamanager": {
        "data": str(dataset), "dataparser": parser, "allow_depth_missing": "false",
        "_target": "orbvi_gaussian.lidar_splatfacto.LidarFullImageDatamanager"}, "model": model}}


def split_fixture():
    frames, rows = [], {}
    parts = {key: [] for key in audit.SPLIT_KEYS.values()}
    indices = {}
    for index, group in ((0, "100"), (1, "101"), (8, "108")):
        name = "images/" + group + ".png"
        source = "images/camera_2/" + group + ".jpg"
        frame = {"file_path": name, "source_file_path": source, "group_id": group,
                 "camera_id": "camera_2", "source_header_ns": int(group), "source_bag_ns": int(group)+1000}
        frames.append(frame)
        rows[("camera_2", source)] = {"header_ns": group, "bag_ns": str(int(group)+1000)}
        indices[group] = index
        parts[audit.SPLIT_KEYS[audit.global_partition(index)]].append(name)
    return {"frames": frames, **parts}, rows, indices


class PlyTests(unittest.TestCase):
    def test_all_rows_and_sha_scanned_in_bounded_chunks(self):
        payload = gaussian_bytes(17)
        result = audit.scan_ply(io.BytesIO(payload), chunk_bytes=136)
        self.assertEqual(result["vertices"], 17)
        self.assertTrue(result["payload_all_finite"])
        self.assertEqual(result["bbox_m"]["max"][0], 16)
        import hashlib
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())

    def test_nonfinite_tail_in_any_property_fails(self):
        for property_name in ("x", "f_dc_1", "scale_2", "extra_property"):
            with self.subTest(property=property_name):
                payload = gaussian_bytes(17, lambda rows: rows[property_name].__setitem__(-1, np.nan),
                                         names=NAMES+["extra_property"])
                with self.assertRaisesRegex(audit.AuditError, "nonfinite PLY"):
                    audit.scan_ply(io.BytesIO(payload), chunk_bytes=144)

    def test_quaternion_normality_tail(self):
        for value in (0, 1.01, float("inf")):
            with self.subTest(value=value):
                payload = gaussian_bytes(17, lambda rows: rows["rot_0"].__setitem__(-1, value))
                with self.assertRaises(audit.AuditError):
                    audit.scan_ply(io.BytesIO(payload), chunk_bytes=136)

    def test_initializer_points_cannot_pass_as_gs(self):
        with self.assertRaisesRegex(audit.AuditError, "real Gaussian attributes"):
            audit.scan_ply(io.BytesIO(gaussian_bytes(names=["x", "y", "z"])))

    def test_truncation_and_undeclared_payload(self):
        payload = gaussian_bytes()
        for broken in (payload[:-1], payload+b"garbage"):
            with self.subTest(tail=broken[-8:]):
                with self.assertRaisesRegex(audit.AuditError, "truncated|trailing"):
                    audit.scan_ply(io.BytesIO(broken))

    def test_ascii_and_big_endian(self):
        for encoding in ("ascii", "binary_big_endian"):
            self.assertEqual(audit.scan_ply(io.BytesIO(gaussian_bytes(encoding=encoding)))["vertices"], 5)

    def test_invalid_sh_and_log_scale(self):
        with self.assertRaisesRegex(audit.AuditError, "SH coefficient"):
            audit.scan_ply(io.BytesIO(gaussian_bytes(names=NAMES+["f_rest_0"])))
        with self.assertRaisesRegex(audit.AuditError, "log scale"):
            audit.scan_ply(io.BytesIO(gaussian_bytes(change=lambda r: r["scale_0"].__setitem__(-1, 90))))


class WorldAndRendererTests(unittest.TestCase):
    def test_real_lidar_renderer_config(self):
        dataset, config = config_fixture()
        audit.validate_config(config, dataset, audit.ROOT, 12000)

    def test_bad_renderer_and_wrong_data_cannot_be_green(self):
        for section, key, value in (("model", "use_inria_surfel_renderer", "false"),
                                    ("model", "random_init", "true"),
                                    ("model", "surface_splats", "false"),
                                    ("datamanager", "_target", "StockImageDatamanager"),
                                    ("datamanager", "data", str(audit.ROOT/"data/wrong_block"))):
            with self.subTest(key=key):
                dataset, config = config_fixture()
                config["pipeline"][section][key] = value
                with self.assertRaises(audit.AuditError):
                    audit.validate_config(config, dataset, audit.ROOT, 12000)

    def test_centering_and_autoscale_rejected(self):
        audit.identity_transform(np.eye(4)[:3], 1)
        audit.identity_transform(np.eye(4), 1)
        for key, value in (("center_method", "poses"), ("orientation_method", "up"),
                           ("auto_scale_poses", "true"), ("scale_factor", "0.1")):
            dataset, config = config_fixture()
            config["pipeline"]["datamanager"]["dataparser"][key] = value
            with self.assertRaises(audit.AuditError):
                audit.validate_config(config, dataset, audit.ROOT, 12000)
        T = np.eye(4)[:3].copy()
        T[0, 3] = 5
        with self.assertRaisesRegex(audit.AuditError, "centering"):
            audit.identity_transform(T, 1)
        with self.assertRaisesRegex(audit.AuditError, "autoscale"):
            audit.identity_transform(np.eye(4)[:3], .1)

    def test_yaml_python_tags_are_inert_and_preserve_target_name(self):
        yaml_text = "!!python/object:missing.Trainer\n_target: !!python/name:missing.LidarSplatfactoModel ''\ndata: !!python/object/apply:pathlib.PosixPath\n- /\n- test\n"
        with mock.patch.object(Path, "open", return_value=io.StringIO(yaml_text)):
            config = audit.read_config(Path("unused"))
        self.assertEqual(config["_target"], "missing.LidarSplatfactoModel")
        self.assertEqual(config["data"], ["/", "test"])

    def test_camera_pose_reconstructed_from_global_trajectory(self):
        stamps = np.array([100, 200], dtype=np.int64)
        values = np.array([[30, -5, 2, 0, 0, 0, 1], [32, -5, 2, 0, 0, 0, 1]], float)
        cal = {"cameras": {"camera_2": {"time_offset_ns": "0", "T_body_camera": np.eye(4).tolist()}},
               "virtual_views": [{"name": "center", "yaw_deg": 0}]}
        frame = {"camera_id": "camera_2", "source_header_ns": 150, "timestamp_ns": 150, "view_id": "center"}
        T = audit.expected_camera_pose(stamps, values, frame, cal)
        np.testing.assert_allclose(T[:3, 3], [31, -5, 2])
        np.testing.assert_allclose(T[:3, :3], np.diag([1, -1, -1]))

    def test_prepared_depth_corruption_and_empty_support(self):
        audit.validate_depth(np.ones((2, 3), dtype=np.float32), 2, 3)
        for depth in (np.zeros((2, 3), np.float32), np.full((2, 3), np.nan, np.float32),
                      np.full((2, 3), -1, np.float32), np.ones((3, 2), np.float32)):
            with self.assertRaises(audit.AuditError):
                audit.validate_depth(depth, 2, 3)


class SplitTests(unittest.TestCase):
    def validate(self, meta, rows, indices, registry=None, hashes=None):
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(audit, "sha256", side_effect=lambda p: str(p) if hashes is None else hashes[str(p)]):
            return audit.validate_splits(meta, rows, indices, audit.ROOT/"data/keyframes_2hz", audit.ROOT,
                                         registry if registry is not None else {}, {})

    def test_source_index_split_passes(self):
        meta, rows, indices = split_fixture()
        _, report = self.validate(meta, rows, indices)
        self.assertEqual(report["split_frames"], {"test": 1, "train": 1, "dev": 1})

    def test_split_recomputed_not_preparation_boolean(self):
        meta, rows, indices = split_fixture()
        meta["train_filenames"], meta["val_filenames"] = meta["val_filenames"], meta["train_filenames"]
        with self.assertRaisesRegex(audit.AuditError, "actual global source index"):
            self.validate(meta, rows, indices)

    def test_same_source_content_leaks_across_groups(self):
        meta, rows, indices = split_fixture()
        hashes = {str((audit.ROOT/"data/keyframes_2hz"/f["source_file_path"]).resolve()): "same-photo" for f in meta["frames"]}
        with self.assertRaisesRegex(audit.AuditError, "photograph leaks"):
            self.validate(meta, rows, indices, hashes=hashes)

    def test_cross_block_source_leak_and_bad_original_timestamp(self):
        meta, rows, indices = split_fixture()
        registry = {("header", "camera_2", 101): "test"}
        with self.assertRaisesRegex(audit.AuditError, "photograph leaks"):
            self.validate(meta, rows, indices, registry=registry)
        rows[("camera_2", meta["frames"][0]["source_file_path"])]["header_ns"] = "999"
        with self.assertRaisesRegex(audit.AuditError, "original index"):
            self.validate(meta, rows, indices)

    def test_unassigned_duplicate_and_multi_view_leak(self):
        meta, rows, indices = split_fixture()
        extra = copy.deepcopy(meta["frames"][0])
        extra["file_path"] = "images/second_view.png"
        meta["frames"].append(extra)
        with self.assertRaisesRegex(audit.AuditError, "unassigned"):
            self.validate(meta, rows, indices)
        meta["train_filenames"].append(extra["file_path"])
        with self.assertRaises(audit.AuditError):
            self.validate(meta, rows, indices)


class MetricsAndCompletenessTests(unittest.TestCase):
    def metric_fixture(self):
        meta, _, _ = split_fixture()
        partitions = {f["file_path"]: audit.global_partition(i) for f, i in zip(meta["frames"], (0, 1, 8))}
        frame = meta["frames"][0]
        config = audit.ROOT/"runs/model/config.yml"
        checkpoint = audit.ROOT/"runs/model/nerfstudio_models/step-000011999.ckpt"
        report = {"status": "evaluated", "split": "test", "config": str(config), "checkpoint": str(checkpoint),
                  "step": 11999, "images": 1, "world_parser_transform": np.eye(4)[:3].tolist(), "world_parser_scale": 1,
                  "psnr_db_mean": 22., "ssim_valid_mean": .7, "calibration_status": False,
                  "frames": [{"file_path": frame["file_path"], "group_id": frame["group_id"], "camera_id": frame["camera_id"],
                              "psnr_db": 22., "ssim_valid_pixels": .7, "valid_fraction": .9,
                              "map_depth_compared_pixels": 10, "map_projected_depth_rmse_m": 2., "map_projected_depth_median_abs_m": 1.}],
                  **{k: .2 for k in ("gaussian_to_initializer_median_m", "gaussian_to_initializer_p95_m",
                                     "gaussian_to_measured_map_median_m", "gaussian_to_measured_map_p95_m")}}
        return report, meta, partitions, config, checkpoint

    def validate(self, fixture):
        return audit.validate_metrics(*fixture[:1], "test", *fixture[1:], 12000, audit.ROOT)

    def test_actual_metrics_and_unknown_calibration_preserved(self):
        fixture = self.metric_fixture()
        fixture[0].pop("calibration_status")
        m = self.validate(fixture)
        self.assertEqual(m["psnr_db_frame_mean"], 22)
        self.assertEqual(m["calibration_status_recorded"], "unknown")
        self.assertIn("not independent", m["protocol"])

    def test_fabricated_summary_old_step_wrong_checkpoint_or_missing_frame(self):
        for key, value in (("psnr_db_mean", 100), ("step", 999), ("frames", []),
                           ("checkpoint", str(audit.ROOT/"runs/other/step-000011999.ckpt"))):
            fixture = self.metric_fixture()
            fixture[0][key] = value
            with self.assertRaises(audit.AuditError):
                self.validate(fixture)

    def test_interval_coverage_handles_overlap_but_reports_omissions(self):
        self.assertEqual(audit.interval_gaps([(0, 40), (30, 70), (65, 100)], 0, 100), [])
        self.assertEqual(audit.interval_gaps([(0, 40), (50, 100)], 0, 100), [[40, 50]])
        self.assertEqual(audit.interval_gaps([(10, 90)], 0, 100), [[0, 10], [90, 100]])

    def test_full_coverage_json_serializes_and_keeps_source_boundary_omissions(self):
        stamps = np.arange(60, dtype=np.int64)*1000000000
        values = np.zeros((60, 7))
        values[:, 2] = np.arange(60)/10
        schedule = {"floors": [{"floor_id": i+1, "start_stamp": i*10, "end_stamp": i*10+9} for i in range(6)],
                    "ramps": [{"lower_floor_id": i+1, "upper_floor_id": i+2,
                               "start_stamp": i*10+9, "end_stamp": i*10+10} for i in range(5)]}
        planned = [{"id": "block_000", "start_header_ns": 0, "end_header_ns": int(stamps[-1])}]
        blocks = [{"id": "block_000", "preparation": {}, "status": "unknown", "floor_id_recorded": "坡道 F1-F2"}]
        source = (None, None, None, [{"header_ns": str(t)} for t in (-1, 0, int(stamps[-1]), int(stamps[-1])+1)])
        meta = {"frames": [{"group_id": str(int(t))} for t in stamps]}
        with mock.patch.object(audit, "read_json", return_value=meta):
            coverage = audit.Auditor().coverage(planned, blocks, (None, None, stamps, values), source, schedule)
        encoded = json.dumps(coverage, allow_nan=False)
        self.assertEqual(json.loads(encoded)["source_camera_2_groups_outside_retained_lio"], 2)
        self.assertEqual(len(coverage["estimated_floor_ramp_visits"]), 11)
        self.assertEqual(coverage["prepared_interval_gaps_ns"], [])

    def test_manifest_words_cannot_make_partial_data_complete(self):
        manifest = {"status": "complete", "coordinates": "camera_init", "units": "m", "dataset_kind": "real",
                    "run_id": "garage_surface_v1", "calibration_verified": "unknown",
                    "blocks": [{"id": i, "training": {"status": "trained", "iterations": 12000}} for i in audit.EXPECTED_IDS]}
        plan = {"blocks": [{"id": i} for i in audit.EXPECTED_IDS]}
        auditor = audit.Auditor()
        def partial(block_id, *args):
            return {"id": block_id, "status": "prepared_or_in_progress", "metrics": {}}
        with mock.patch.object(audit, "read_json", side_effect=lambda p: manifest if p.name == "garage_manifest.json" else plan), \
             mock.patch.object(auditor, "trajectory", return_value=None), mock.patch.object(auditor, "source_index", return_value=None), \
             mock.patch.object(auditor, "block", side_effect=partial), mock.patch.object(auditor, "coverage", return_value={}):
            report = auditor.run()
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["high_fidelity_acceptance"], "not_assessed")
        self.assertEqual(report["calibration_verified_recorded"], "unknown")
        self.assertEqual(len(report["blocks"]), 28)

    def test_require_complete_exit_and_read_only_stdout(self):
        for strict, complete, expected in ((True, False, 1), (False, False, 0), (True, True, 0)):
            report = {"complete": complete, "status": "partial" if not complete else "complete_first_delivery_with_limitations", "block_status_counts": {}}
            with mock.patch.object(audit.Auditor, "run", return_value=report), \
                 mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(audit.main(["--require-complete"] if strict else []), expected)

    def test_wrong_or_reduced_candidate_ids_block_completion(self):
        ids = list(audit.EXPECTED_IDS)
        plan = {"blocks": [{"id": i} for i in ids[:-1]]}
        manifest = {"coordinates": "camera_init", "units": "m", "dataset_kind": "real", "run_id": "garage_surface_v1",
                    "blocks": [{"id": i} for i in ids]}
        auditor = audit.Auditor()
        with mock.patch.object(audit, "read_json", side_effect=lambda p: manifest if p.name == "garage_manifest.json" else plan), \
             mock.patch.object(auditor, "trajectory", return_value=None), mock.patch.object(auditor, "source_index", return_value=None), \
             mock.patch.object(auditor, "block", side_effect=lambda block_id, *a: {"id": block_id, "status": "complete", "metrics": {}}), \
             mock.patch.object(auditor, "coverage", return_value={}):
            report = auditor.run()
        self.assertFalse(report["complete"])
        self.assertTrue(any(i["stage"] == "expected_ids" for i in report["issues"]))

    def test_publish_pending_bbox_is_incomplete_not_fake_model_damage(self):
        ply = {"vertices": 2, "sh_degree": 0, "bbox_m": {"min": [0, 0, 0], "max": [1, 1, 1]}}
        report = {"export": {"coordinates": "camera_init", "gaussians": 2, "sh_degree": 0, "bbox_m": ply["bbox_m"]}}
        item = {"bbox_m": {"min": [0, 0, 0], "max": [2, 2, 2]}, "training": {"status": "training"}}
        with self.assertRaises(audit.IncompleteError):
            audit.Auditor().export_provenance(report, ply, item)
        item["training"]["status"] = "trained"
        with self.assertRaises(audit.AuditError) as error:
            audit.Auditor().export_provenance(report, ply, item)
        self.assertNotIsInstance(error.exception, audit.IncompleteError)


class OutputBoundaryTests(unittest.TestCase):
    def test_absolute_relative_traversal_and_symlink_escape(self):
        for value in ("/tmp/audit.json", "../audit.json", "reports/../../escape.md"):
            with self.assertRaisesRegex(audit.AuditError, "outside project"):
                audit.output_path(value)
        with mock.patch.object(Path, "resolve", return_value=Path("/tmp/escaped")):
            # ROOT must stay fixed during this mocked resolution.
            root = mock.MagicMock(spec=Path)
            root.resolve.return_value = audit.ROOT
            with self.assertRaisesRegex(audit.AuditError, "outside project"):
                audit.project_path(Path("/absolute/symlink"), root)

    def test_existing_outputs_are_never_overwritten(self):
        with self.assertRaisesRegex(audit.AuditError, "overwrite"):
            audit.output_path("maps/garage_manifest.json")

    def test_output_rejected_before_audit(self):
        with mock.patch.object(audit.Auditor, "run") as run, mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                audit.main(["--json", "/tmp/forbidden.json"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_json_and_markdown_write_only_exclusive_requested_new_paths(self):
        report = {"complete": False, "status": "partial", "target_iterations": 12000,
                  "block_status_counts": {}, "blocks": [], "metrics_summary": {}, "coverage": None,
                  "issues": [], "limitations": audit.LIMITATIONS}
        outputs = [audit.ROOT/"reports/virtual_audit.json", audit.ROOT/"reports/virtual_audit.md"]
        class Retained(io.StringIO):
            def close(self):
                pass
        streams = {p: Retained() for p in outputs}
        def opening(path, mode, **kwargs):
            self.assertEqual(mode, "x")
            return streams[path]
        with mock.patch.object(audit, "output_path", side_effect=outputs), mock.patch.object(audit.Auditor, "run", return_value=report), \
             mock.patch.object(Path, "open", autospec=True, side_effect=opening), mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(audit.main(["--require-complete", "--json", "virtual.json", "--md", "virtual.md"]), 1)
        self.assertEqual(json.loads(streams[outputs[0]].getvalue())["status"], "partial")
        self.assertIn("not assessed", streams[outputs[1]].getvalue())
        self.assertIn("same-source conditioned", streams[outputs[1]].getvalue())


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU torch verification tests")
class CheckpointTests(unittest.TestCase):
    def fixture(self, step=11, corrupt=None, wrong_ply=False, filename_step=11):
        import torch
        params = {"means": torch.tensor([[0., 0, 0], [1., 0, 0]]), "features_dc": torch.zeros(2, 3),
                  "features_rest": torch.zeros(2, 0, 3), "opacities": torch.zeros(2, 1),
                  "scales": torch.zeros(2, 3), "quats": torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])}
        state = {"step": step, "pipeline": {"_model.gauss_params."+k: v for k, v in params.items()},
                 "optimizers": {"means": {"state": {"momentum": torch.zeros(2, 3)}}}, "schedulers": {}, "scalers": {}}
        if corrupt:
            corrupt(state)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        original_load = torch.load
        path = mock.MagicMock(spec=Path)
        path.name = f"step-{filename_step:09d}.ckpt"
        path.stat.return_value = SimpleNamespace(st_size=999, st_mtime_ns=100)
        payload = gaussian_bytes(2, change=(lambda rows: rows["x"].__setitem__(-1, 999)) if wrong_ply else None)
        def scan(path, **kwargs):
            return audit.scan_ply(io.BytesIO(payload), **kwargs)
        def load(path, **kwargs):
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            return original_load(buffer, **kwargs)
        with mock.patch.object(torch, "load", side_effect=load), mock.patch.object(audit, "check_ply", side_effect=scan):
            return audit.verify_checkpoint(path, Path("unused.ply"), 12, 136)

    def test_real_payload_step_and_ply_verified_cpu(self):
        state, ply = self.fixture()
        self.assertTrue(state["step_verified"] and state["ply_matches_checkpoint"] and state["all_tensors_finite"])
        self.assertEqual(ply["vertices"], 2)

    def test_filename_cannot_fake_checkpoint_target(self):
        with self.assertRaisesRegex(audit.AuditError, "payload step"):
            self.fixture(step=0)

    def test_real_intermediate_payload_is_incomplete(self):
        with self.assertRaises(audit.IncompleteError):
            self.fixture(step=8, filename_step=8)

    def test_nonfinite_optimizer_state_and_zero_model_quat(self):
        for corrupt in (lambda s: s["optimizers"]["means"]["state"]["momentum"].fill_(float("nan")),
                        lambda s: s["pipeline"]["_model.gauss_params.quats"].zero_()):
            with self.assertRaises(audit.AuditError):
                self.fixture(corrupt=corrupt)

    def test_unrelated_finite_ply_fails_model_association(self):
        with self.assertRaisesRegex(audit.AuditError, "differs from checkpoint"):
            self.fixture(wrong_ply=True)


if __name__ == "__main__":
    unittest.main()

"""Tiny CPU datasets exercising real copy, point I/O and surfel visibility."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_joint_blocks", PROJECT / "scripts/prepare_joint_blocks.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


class JointBlockCpuTests(unittest.TestCase):
    def setUp(self):
        local_tmp = PROJECT / "runtime/tmp"
        local_tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="joint_cpu_", dir=local_tmp)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_root = self.root / "sources"
        self.out = self.root / "joint"
        self.calibration = self.root / "calibration.yaml"
        self.poses = self.root / "poses.csv"
        self.index = self.root / "extraction/indexes/images.csv"
        self.index.parent.mkdir(parents=True)
        self.calibration.write_bytes(b"verified: false\nworld_frame: camera_init\n")
        self.poses.write_bytes(b"timestamp_ns,x,y,z\n1720268000000000001,100,-20,5\n")
        self.index.write_bytes(b"source groups: train 1,2,3; dev 8,24; test 16,32,48\n")
        self.ids = ["block_014", "block_015", "block_016"]
        self.original_points = {}
        for offset, (bid, groups) in enumerate(zip(self.ids, ((1, 8, 16), (2, 8, 32), (3, 24, 48)))):
            root = self.source_root / bid
            for subdir in ("images", "masks", "depth", "depth_masks"):
                (root / subdir).mkdir(parents=True)
            frames = []
            for group in groups:
                frame = self.frame(group)
                frames.append(frame)
                image = np.full((3, 4, 3), group, np.uint8)
                mask = np.full((3, 4), 255, np.uint8)
                mask[0, 0] = 0
                self.assertTrue(cv2.imwrite(str(root / frame["file_path"]), image))
                self.assertTrue(cv2.imwrite(str(root / frame["mask_path"]), mask))
                # Neighbor teachers deliberately differ even for shared RGB.
                np.save(root / frame["lidar_depth_path"], np.full((3, 4), 90. + offset, np.float32))
                self.assertTrue(cv2.imwrite(str(root / frame["lidar_depth_mask_path"]), np.full((3, 4), 255, np.uint8)))
            meta = dict(camera_model="OPENCV", orientation_override="none", holdout_every=8,
                        pixel_center_convention="half_integer; principal point +0.5 from OpenCV rectification",
                        point_cloud_has_rgb=True, point_cloud_has_observations=False,
                        ply_file_path="points3D.ply", point_cloud_path="points3D.ply", frames=frames,
                        train_filenames=[frames[0]["file_path"]], val_filenames=[frames[1]["file_path"]],
                        test_filenames=[frames[2]["file_path"]])
            xyz = np.array([[100.011 + .004 * offset, -20.011 - .004 * offset, 7.011 + .004 * offset],
                            [100.4 + offset, -20., 8.]], np.float32)
            normals = np.array([[0., 0., -1.], [0., 0., -1.]], np.float32)
            valid = np.array([offset > 0, True])
            if offset == 0:
                normals[0] = [1., 0., 0.]
            np.savez_compressed(root / "lidar_reference.npz", xyz=xyz, normals=normals, normal_valid=valid)
            init_xyz = np.array([[100. + offset, -20., 7.], [101. + offset, -20., 8.]], np.float64)
            init_rgb = np.array([[14 + offset, 20, 30], [40, 50 + offset, 60]], np.uint8)
            with (root / "points3D.ply").open("w") as stream:
                M._write_ply(stream, init_xyz, init_rgb)
            self.original_points[bid] = (init_xyz, init_rgb)
            prep = dict(status="prepared_experimental", block_id=bid, calibration_verified=False,
                candidate_experiment_authorized=True, independent_test_groups=True, global_source_split=True,
                global_split_protocol="train: index%8!=0; dev: index%16==8; test: index%16==0",
                split_reference_images_sha256=digest(self.index), source_extraction=str(self.index.parents[1]),
                calibration=str(self.calibration), calibration_sha256=digest(self.calibration), calibration_id="same-actual-calibration",
                pose_csv=str(self.poses), pose_sha256=digest(self.poses), pose_source="external_csv",
                world_frame="camera_init", body_frame="aft_mapped", units="metres",
                map=dict(source="historical/measured-map.pcd", world_frame="camera_init", alignment_verified_by_operator=True,
                         color_source="calibrated_training_images_only", has_measured_rgb=True, colorized_points=2,
                         input_has_rgb_field=True, map_rgb_verified_by_operator=False),
                export=dict(point_count=2, paths=dict(point_cloud=str(root / "points3D.ply"))),
                lidar_reference_path="lidar_reference.npz")
            write_json(root / "transforms.json", meta)
            write_json(root / "preparation.json", prep)

    def frame(self, group):
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = [100., -20., 5.]
        stamp = 1720268000000000001 + group * 1_000_000_000
        return dict(file_path="images/%d_camera_2.png" % group, mask_path="masks/%d_camera_2.png" % group,
                    lidar_depth_path="depth/%d_camera_2.npy" % group, lidar_depth_mask_path="depth_masks/%d_camera_2.png" % group,
                    w=4, h=3, fl_x=4., fl_y=4., cx=1.5, cy=1.5, transform_matrix=pose.tolist(),
                    timestamp_ns=stamp, source_header_ns=stamp, source_bag_ns=stamp + 1,
                    camera_id="camera_2", group_id=str(group), view_id="center", source_file_path="original/%d.jpg" % group)

    def merge(self, ids=None, out=None):
        return M.prepare_joint(ids or self.ids, out or self.out, source_root=self.source_root)

    def approve_v3_sources(self):
        """Synthetic evidence is confined to this tiny fixture, never real data."""
        self.policy = asdict(M.SurfelDepthPolicy.trusted_ground_v3())
        implementation = {"src/orbvi_gaussian/" + name: digest(PROJECT / "src/orbvi_gaussian" / name)
                          for name in ("surfels.py", "points.py")}
        self.evidence_paths = {kind: self.root / (kind + ".json") for kind in ("pixel", "geometry", "dev")}
        write_json(self.evidence_paths["pixel"], dict(status="passed", wrong_index_count=0,
            tested_index_count=12, foreground_array_equal_fraction=1., pytest_returncode=0,
            commands=["synthetic fixture"], source_sha256=implementation))
        write_json(self.evidence_paths["geometry"], dict(status="passed", overall="PASS", policy=self.policy,
            safety_cases=[dict(status="PASS")], passed_cases=1, sources_unchanged_during_tests=True,
            pytest_returncode=0, commands=["synthetic fixture"], source_sha256=implementation))
        write_json(self.evidence_paths["dev"], dict(status="DEV_SUPPORT_AND_INDEPENDENT_SAFETY_PASS",
            candidate_policy=self.policy, source_sha256=implementation,
            safety=dict(status="PASS", failed_cases=[], case_count=1,
                        cases=[dict(results=dict(min1=dict(passes_expected=True)))]),
            results=[dict(frames=60, split="DEV", potential_unknown_foreground_support=dict(min1=0))]))
        self.approval = self.root / "approval.json"
        write_json(self.approval, dict(
            s0=dict(status="PASS", passed=7, wrong_index_count=0, foreground_array_equal_fraction=1.),
            s1=dict(status="ready"), v3=dict(status="PASS", independent_safety_status="PASS",
                unknown_foreground_gate="PASS", unknown_foreground_DEV_support_pixels=0,
                dev_frames_checked=60, geometry_safety_cases=1, independent_safety_cases=1),
            approved_policy=self.policy, implementation_sha256=implementation,
            evidence={str(p.relative_to(PROJECT)): dict(sha256=digest(p)) for p in self.evidence_paths.values()}))
        for bid in self.ids:
            root = self.source_root / bid
            prep = json.loads((root / "preparation.json").read_text())
            prep["map"]["depth_protocol"] = "surfel_depth_v3 explicitly approved fixture"
            prep["teacher_provenance"] = dict(policy=self.policy, s0_report=str(self.approval),
                s0_report_sha256=digest(self.approval),
                implementation_sha256={Path(k).name: v for k, v in implementation.items()},
                reference_sha256=digest(root / "lidar_reference.npz"),
                frames=[dict(file_path=f["file_path"], depth_sha256=digest(root / f["lidar_depth_path"]),
                             support_sha256=digest(root / f["lidar_depth_mask_path"])) for f in self.meta(bid)["frames"]])
            write_json(root / "preparation.json", prep)

    def test_v3_uses_exact_approved_policy_on_all_merged_records_and_pins_evidence(self):
        self.approve_v3_sources()
        before = self.all_source_hashes()
        with patch.object(M, "project_surfel_depth", wraps=M.project_surfel_depth) as project:
            result = M.prepare_joint(self.ids, self.out, source_root=self.source_root, depth_policy_approval=self.approval)
        self.assertEqual(project.call_count, 8)
        self.assertTrue(all(M.canonical(asdict(c.kwargs["policy"])) == M.canonical(self.policy) for c in project.call_args_list))
        self.assertEqual(result["depth_policy_mode"], "approved_explicit")
        self.assertEqual(result["reference_points"], 6)  # No 5cm thinning of original v3 reference records.
        self.assertIsNone(result["reference_voxel_m"])
        proof = result["teacher_provenance"]
        self.assertEqual(proof["policy_sha256"], hashlib.sha256(M.canonical(self.policy).encode()).hexdigest())
        self.assertEqual(proof["approval_sha256"], digest(self.approval))
        self.assertEqual(len(proof["frozen_evidence"]), 3)
        with np.load(self.out / "lidar_reference.npz") as data:
            reference = {k: data[k] for k in ("xyz", "normals", "normal_valid")}
        for f in json.loads((self.out / "transforms.json").read_text())["frames"]:
            mask = cv2.imread(str(self.out / f["mask_path"]), 0)
            expected, valid = M.regenerate_teacher(reference, f, mask, policy=M.SurfelDepthPolicy(**self.policy))
            np.testing.assert_array_equal(np.load(self.out / f["lidar_depth_path"]), expected)
            np.testing.assert_array_equal(cv2.imread(str(self.out / f["lidar_depth_mask_path"]), 0) > 0, valid)
        self.assertEqual(before, self.all_source_hashes())

    def test_v3_preserves_nearer_unknown_occluder_inside_same_legacy_voxel(self):
        sources = []
        for index, (z, valid) in enumerate(((13., False), (13.02, True))):
            path = self.root / ("occluder_%d.npz" % index)
            np.savez(path, xyz=np.array([[100., -20., z]], np.float32),
                     normals=np.array([[0., 0., 1.]], np.float32), normal_valid=np.array([valid]))
            sources.append(dict(reference=path))
        legacy = M.merge_references(sources)
        preserved = M.merge_references(sources, preserve_visibility=True)
        self.assertEqual(len(legacy["xyz"]), 1)
        self.assertTrue(legacy["normal_valid"][0])
        self.assertEqual(len(preserved["xyz"]), 2)
        frame, mask = self.frame(1), np.full((3, 4), 255, np.uint8)
        policy = M.SurfelDepthPolicy.trusted_ground_v3()
        _, valid = M.regenerate_teacher(legacy, frame, mask, policy=policy)
        self.assertTrue(valid[1, 1])
        depth, valid = M.regenerate_teacher(preserved, frame, mask, policy=policy)
        self.assertFalse(valid[1, 1])
        self.assertEqual(depth[1, 1], 0.)

    def test_v3_cannot_silently_fall_back_to_default_or_legacy_sources(self):
        self.approve_v3_sources()
        with self.assertRaisesRegex(ValueError, "explicit --depth-policy-approval"):
            self.merge()
        p = self.source_root / "block_016/preparation.json"
        prep = json.loads(p.read_text()); prep.pop("teacher_provenance"); write_json(p, prep)
        with self.assertRaisesRegex(ValueError, "teacher policy/approval"):
            M.prepare_joint(self.ids, self.out, source_root=self.source_root, depth_policy_approval=self.approval)
        self.assertFalse(self.out.exists())

    def test_partial_policy_s0_safety_or_changed_implementation_fail_before_writes(self):
        self.approve_v3_sources()
        original = json.loads(self.approval.read_text())
        for kind in ("partial_policy", "s0", "unknown", "safety", "implementation"):
            changed = copy.deepcopy(original)
            if kind == "partial_policy": changed["approved_policy"].pop("front_cluster_min_points")
            elif kind == "s0": changed["s0"]["wrong_index_count"] = 1
            elif kind == "unknown": changed["v3"]["unknown_foreground_DEV_support_pixels"] = 1
            elif kind == "safety": changed["v3"]["independent_safety_status"] = "FAIL"
            else: changed["implementation_sha256"]["src/orbvi_gaussian/surfels.py"] = "0" * 64
            write_json(self.approval, changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                M.prepare_joint(self.ids, self.out, source_root=self.source_root, depth_policy_approval=self.approval)
            self.assertFalse(self.out.exists())

    def test_frozen_evidence_content_and_failed_cases_are_checked_beyond_summary_pass(self):
        self.approve_v3_sources()
        original = json.loads(self.approval.read_text())
        for kind, path in self.evidence_paths.items():
            raw = path.read_bytes()
            path.write_bytes(raw + b" ")
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "evidence SHA256"):
                M.validate_depth_approval(self.approval, {})
            proof = json.loads(raw)
            if kind == "pixel": proof["wrong_index_count"] = 1
            elif kind == "geometry": proof["safety_cases"][0]["status"] = "FAIL"
            else: proof["results"][0]["potential_unknown_foreground_support"]["min1"] = 1
            write_json(path, proof)
            changed = copy.deepcopy(original)
            changed["evidence"][str(path.relative_to(PROJECT))]["sha256"] = digest(path)
            write_json(self.approval, changed)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "did not pass"):
                M.validate_depth_approval(self.approval, {})
            path.write_bytes(raw); write_json(self.approval, original)

    def test_explicit_archive_relocation_preserves_exact_approval_hash(self):
        self.approve_v3_sources()
        source = self.evidence_paths["geometry"]
        frozen = self.root / "frozen_geometry.json"
        frozen.write_bytes(source.read_bytes())
        archive = self.root / "archive.json"
        value = dict(approval_file=str(self.approval), approval_sha256=digest(self.approval),
            historical_evidence=[dict(source_at_approval=str(source.relative_to(PROJECT)),
                sha256=digest(frozen), immutable_evidence_copy=str(frozen))])
        write_json(archive, value)
        source.write_bytes(b'{"unrelated_new_gate": true}')
        with self.assertRaisesRegex(ValueError, "evidence SHA256"):
            M.validate_depth_approval(self.approval, {})
        result = M.prepare_joint(self.ids, source_root=self.source_root, depth_policy_approval=archive, preflight_only=True)
        self.assertEqual(result["depth_policy"]["approval_sha256"], digest(self.approval))
        self.assertEqual(result["depth_policy"]["requested_approval_sha256"], digest(archive))
        frozen.write_bytes(b"changed frozen evidence")
        with self.assertRaisesRegex(ValueError, "evidence SHA256"):
            M.validate_depth_approval(archive, {})

    def test_preflight_never_merges_geometry_writes_outputs_or_regenerates_depth(self):
        self.approve_v3_sources()
        before = self.all_source_hashes()
        with patch.object(M, "merge_references", side_effect=AssertionError("No merge")), \
             patch.object(M, "merge_initializer", side_effect=AssertionError("No initializer merge")), \
             patch.object(M, "write_teacher", side_effect=AssertionError("No depth")), \
             patch.object(M, "new_output_dir", side_effect=AssertionError("No outputs")):
            result = M.prepare_joint(self.ids, self.out, source_root=self.source_root,
                                     depth_policy_approval=self.approval, preflight_only=True)
        self.assertEqual(result["status"], "source_preflight_passed")
        self.assertFalse(result["outputs_created"] or result["reference_merge_performed"] or result["gpu_used"])
        self.assertEqual(result["resolution"], [4, 3])
        self.assertFalse(self.out.exists())
        self.assertEqual(before, self.all_source_hashes())

    def test_mixed_resolution_unique_view_and_incomplete_sources_are_rejected(self):
        path = self.source_root / "block_016/transforms.json"
        meta = json.loads(path.read_text())
        for f in meta["frames"]:
            f["w"] = 8; f["camera_id"] = "camera_3"
            for field in ("file_path", "mask_path", "lidar_depth_mask_path"):
                cv2.imwrite(str(path.parent / f[field]), np.full((3, 8), 255, np.uint8))
            np.save(path.parent / f["lidar_depth_path"], np.ones((3, 8), np.float32))
        write_json(path, meta)
        with self.assertRaisesRegex(ValueError, "Mixed source resolutions"):
            self.merge()
        self.assertFalse(self.out.exists())
        write_json(path.parent / "INCOMPLETE.json", dict(status="pending"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.prepare_joint(self.ids, source_root=self.source_root, preflight_only=True)

    def test_approved_source_teacher_policy_and_actual_teacher_hash_must_match(self):
        self.approve_v3_sources()
        path = self.source_root / "block_015/preparation.json"
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed["teacher_provenance"]["policy"]["front_cluster_min_points"] = 2
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "teacher policy/approval"):
            M.prepare_joint(self.ids, self.out, source_root=self.source_root, depth_policy_approval=self.approval)
        write_json(path, original)
        f = self.meta()["frames"][0]
        np.save(path.parent / f["lidar_depth_path"], np.zeros((3, 4), np.float32))
        with self.assertRaisesRegex(ValueError, "teacher bytes"):
            M.prepare_joint(self.ids, self.out, source_root=self.source_root, depth_policy_approval=self.approval)
        self.assertFalse(self.out.exists())

    def meta(self, bid="block_015"):
        return json.loads((self.source_root / bid / "transforms.json").read_text())

    def all_source_hashes(self):
        return {str(p.relative_to(self.root)): digest(p) for p in self.root.rglob("*") if p.is_file() and self.out not in p.parents}

    def test_merge_preserves_bytes_world_integer_timestamps_and_explicit_splits(self):
        before = self.all_source_hashes()
        result = self.merge()
        meta = json.loads((self.out / "transforms.json").read_text())
        self.assertFalse((self.out / "INCOMPLETE.json").exists())
        self.assertEqual(result["status"], "prepared_experimental")
        self.assertIs(result["calibration_verified"], False)
        self.assertEqual(result["world_frame"], "camera_init")
        self.assertEqual(result["units"], "metres")
        self.assertEqual(len(meta["frames"]), 8)
        self.assertEqual([len(meta[k]) for k in M.SPLITS], [3, 2, 3])
        for split in M.SPLITS:
            expected = {name for bid in self.ids for name in self.meta(bid)[split]}
            self.assertEqual(set(meta[split]), expected)
        for frame in meta["frames"]:
            original = next(f for bid in self.ids for f in self.meta(bid)["frames"] if f["file_path"] == frame["file_path"])
            self.assertEqual(frame, original)
            self.assertIs(type(frame["timestamp_ns"]), int)
            self.assertGreater(frame["timestamp_ns"], 2 ** 53)
            for key in ("file_path", "mask_path"):
                first_source = next(self.source_root / bid / frame[key] for bid in self.ids if (self.source_root / bid / frame[key]).exists())
                self.assertEqual((self.out / frame[key]).read_bytes(), first_source.read_bytes())
                self.assertEqual(result["copied_asset_hashes"][frame[key]], digest(first_source))
        self.assertEqual(before, self.all_source_hashes())
        self.assertTrue(all(value == digest(PROJECT / path) for path, value in result["source_hashes"].items()))

    def test_shared_teacher_differences_are_documented_and_all_frames_regenerated(self):
        result = self.merge()
        provenance = result["teacher_provenance"]
        self.assertTrue(provenance["regenerated_all_frames"])
        self.assertEqual(len(provenance["frames"]), result["frames"])
        shared = next(r for r in provenance["frames"] if r["file_path"] == "images/8_camera_2.png")
        self.assertTrue(shared["source_teachers_differ"])
        self.assertEqual([r["source_block_id"] for r in shared["sources"]], ["block_014", "block_015"])
        self.assertNotEqual(shared["sources"][0]["depth_sha256"], shared["sources"][1]["depth_sha256"])
        with np.load(self.out / "lidar_reference.npz", allow_pickle=False) as archive:
            reference = {key: archive[key] for key in ("xyz", "normals", "normal_valid")}
        for frame in json.loads((self.out / "transforms.json").read_text())["frames"]:
            mask = cv2.imread(str(self.out / frame["mask_path"]), 0)
            expected, valid = M.regenerate_teacher(reference, frame, mask)
            actual = np.load(self.out / frame["lidar_depth_path"], allow_pickle=False)
            supported = cv2.imread(str(self.out / frame["lidar_depth_mask_path"]), 0) > 0
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(supported, valid)
            self.assertTrue(np.all(actual[~supported] == 0))
            self.assertLess(float(actual.max()), 10.)  # Source teachers contain 90+ metres.

    def test_reference_voxel_choice_is_existing_valid_record_then_stable_source_id(self):
        self.merge(ids=list(reversed(self.ids)))
        with np.load(self.out / "lidar_reference.npz", allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        self.assertEqual(len(values["xyz"]), 4)
        with np.load(self.source_root / "block_015/lidar_reference.npz", allow_pickle=False) as original:
            np.testing.assert_array_equal(values["xyz"][0], original["xyz"][0])
            np.testing.assert_array_equal(values["normals"][0], original["normals"][0])
        self.assertEqual(values["source_block_index"][0], 1)
        for i, (bid_index, row) in enumerate(zip(values["source_block_index"], values["source_row_index"])):
            with np.load(self.source_root / self.ids[int(bid_index)] / "lidar_reference.npz", allow_pickle=False) as original:
                for key in ("xyz", "normals", "normal_valid"):
                    np.testing.assert_array_equal(values[key][i], original[key][row])
        cells = np.floor(values["xyz"].astype(float) / .05).astype(np.int64)
        self.assertEqual(len({tuple(cell) for cell in cells}), len(cells))

    def test_capped_initializer_preserves_actual_training_color_rows_and_origin(self):
        with patch.object(M, "MAX_INITIALIZER_POINTS", 3), patch.object(M, "read_point_cloud", wraps=M.read_point_cloud) as reader:
            result = self.merge(ids=list(reversed(self.ids)))
        self.assertEqual(result["initializer_points"], 3)
        self.assertEqual(len(reader.call_args_list), 3)
        self.assertTrue(all(call.args[0].name == "points3D.ply" for call in reader.call_args_list))
        xyz, rgb = M.read_point_cloud(self.out / "points3D.ply")
        with np.load(self.out / "point_origins.npz", allow_pickle=False) as archive:
            blocks, rows = archive["source_block_index"], archive["source_row_index"]
        np.testing.assert_array_equal(blocks, [0, 1, 2])
        for i, (block, row) in enumerate(zip(blocks, rows)):
            original_xyz, original_rgb = self.original_points[self.ids[int(block)]]
            np.testing.assert_array_equal(xyz[i], original_xyz[row])
            np.testing.assert_array_equal(rgb[i], original_rgb[row])

    def test_shared_rgb_mask_pose_intrinsics_timestamp_and_filename_conflicts_fail_before_output(self):
        root = self.source_root / "block_015"
        path = root / "transforms.json"
        original_meta = self.meta()
        shared = original_meta["frames"][1]
        for key in ("image", "mask", "pose", "intrinsics", "timestamp", "filename"):
            with self.subTest(conflict=key):
                changed = copy.deepcopy(original_meta)
                files = {}
                if key in ("image", "mask"):
                    target = root / shared["file_path" if key == "image" else "mask_path"]
                    files[target] = target.read_bytes()
                    target.write_bytes(files[target] + b"different-source-bytes")
                elif key == "pose":
                    changed["frames"][1]["transform_matrix"][0][3] += .01
                elif key == "intrinsics":
                    changed["frames"][1]["cx"] += .5
                elif key == "timestamp":
                    changed["frames"][1]["timestamp_ns"] += 1
                else:
                    changed["frames"][1]["camera_id"] = "camera_3"
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                    self.merge()
                self.assertFalse(self.out.exists())
                write_json(path, original_meta)
                for target, raw in files.items():
                    target.write_bytes(raw)

    def test_global_group_and_timestamp_split_leakage_is_rejected(self):
        original = self.meta()
        meta = copy.deepcopy(original)
        meta["train_filenames"], meta["val_filenames"] = meta["val_filenames"], meta["train_filenames"]
        write_json(self.source_root / "block_015/transforms.json", meta)
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.merge()
        self.assertFalse(self.out.exists())
        meta = copy.deepcopy(original)
        meta["frames"][2]["timestamp_ns"] = self.meta("block_014")["frames"][0]["timestamp_ns"]
        write_json(self.source_root / "block_015/transforms.json", meta)
        with self.assertRaisesRegex(ValueError, "timestamp split leakage"):
            self.merge()
        self.assertFalse(self.out.exists())

    def test_source_symlink_escape_and_unsafe_block_ids_fail_before_writes(self):
        path = self.source_root / "block_015/images/8_camera_2.png"
        path.unlink()
        path.symlink_to("/dev/null")
        with self.assertRaisesRegex(ValueError, "inside"):
            self.merge()
        self.assertFalse(self.out.exists())
        for values in (("../014",), ("block_014", "014"), ("028",), ("014", "015", "016", "017")):
            with self.subTest(ids=values), self.assertRaises(ValueError):
                M.block_ids(values)

    def test_unverified_training_color_flags_world_scale_and_calibration_must_match(self):
        path = self.source_root / "block_015/preparation.json"
        prep = json.loads(path.read_text())
        changes = [("world_frame", "other-world"), ("calibration_sha256", "0" * 64),
                   ("calibration_verified", True), ("global_split_protocol", "new-local-split")]
        for key, value in changes:
            changed = copy.deepcopy(prep)
            changed[key] = value
            with self.subTest(key=key):
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    self.merge()
                self.assertFalse(self.out.exists())
        changed = copy.deepcopy(prep)
        changed["map"]["color_source"] = "all_images_including_test"
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "training-only"):
            self.merge()
        write_json(path, prep)
        meta = self.meta()
        meta["applied_scale"] = .5
        write_json(self.source_root / "block_015/transforms.json", meta)
        with self.assertRaisesRegex(ValueError, "identity world"):
            self.merge()

    def test_existing_helper_blocks_trusted_background_through_untrusted_foreground(self):
        frame = self.frame(8)
        translation = np.array([100., -20., 5.])
        reference = dict(xyz=np.array([[0., 0., 2.], [0., 0., 4.], [.8, 0., 2.]]) + translation,
                         normals=np.array([[1., 0., 0.], [0., 0., -1.], [0., 0., -1.]]),
                         normal_valid=np.array([False, True, True]))
        mask = np.full((3, 4), 255, np.uint8)
        depth, valid = M.regenerate_teacher(reference, frame, mask)
        self.assertFalse(valid[1, 1])
        self.assertEqual(depth[1, 1], 0.)
        self.assertTrue(valid[1, 3])
        self.assertEqual(depth[1, 3], 2.)  # Also catches GL/CV sign and half-pixel errors.
        self.assertEqual(int(valid.sum()), 1)
        mask[1, 3] = 0
        depth, valid = M.regenerate_teacher(reference, frame, mask)
        self.assertFalse(valid.any())
        self.assertFalse(depth.any())

    def test_bad_reference_normals_and_nonfinite_geometry_are_rejected_without_output(self):
        path = self.source_root / "block_015/lidar_reference.npz"
        with np.load(path, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        for key, value in (("xyz", np.nan), ("normals", np.inf), ("normals", 2.)):
            data = {key: array.copy() for key, array in original.items()}
            data[key][0, 0] = value
            np.savez_compressed(path, **data)
            with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, "reference|normals"):
                self.merge()
            self.assertFalse(self.out.exists())

    def test_no_overwrite_scope_and_incomplete_marker_on_regeneration_failure(self):
        before = self.all_source_hashes()
        with self.assertRaises(ValueError):
            self.merge(out=PROJECT.parent / "outside_joint")
        with self.assertRaises(ValueError):
            self.merge(out=self.source_root / "block_015/new_joint")
        self.out.mkdir()
        marker = self.out / "keep.txt"
        marker.write_bytes(b"never overwrite")
        with self.assertRaises(FileExistsError):
            self.merge()
        self.assertEqual(marker.read_bytes(), b"never overwrite")
        marker.unlink(); self.out.rmdir()
        with patch.object(M, "write_teacher", side_effect=RuntimeError("fixture projection failure")):
            with self.assertRaisesRegex(RuntimeError, "projection failure"):
                self.merge()
        self.assertTrue((self.out / "INCOMPLETE.json").is_file())
        self.assertFalse((self.out / "preparation.json").exists())
        self.assertEqual(before, self.all_source_hashes())

    def test_limits_and_source_file_changes_are_not_silently_sampled(self):
        with patch.object(M, "MAX_REFERENCE_POINTS", 3), self.assertRaisesRegex(ValueError, "oversized"):
            self.merge()
        self.assertFalse(self.out.exists())
        with patch.object(M, "MAX_INITIALIZER_INPUT_POINTS", 1), self.assertRaisesRegex(ValueError, "max_points"):
            self.merge()
        self.assertFalse(self.out.exists())
        hashes = {}
        path = self.source_root / "block_014/images/1_camera_2.png"
        M.snapshot(path, hashes)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            M.snapshot(path, hashes)

    @unittest.skipUnless(importlib.util.find_spec("nerfstudio"), "NS absent; pure CPU fixtures still run")
    def test_actual_ns_parser_and_lidar_manager_accept_regenerated_fixture(self):
        from orbvi_gaussian.training import confined_environment
        os.environ.update(confined_environment())
        tempfile.tempdir = os.environ["TMPDIR"]
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
        from orbvi_gaussian.lidar_splatfacto import LidarFullImageDatamanagerConfig
        self.merge()
        parser = NerfstudioDataParserConfig(data=self.out, downscale_factor=1, auto_scale_poses=False,
            scale_factor=1., orientation_method="none", center_method="none", load_3D_points=False)
        manager = LidarFullImageDatamanagerConfig(dataparser=parser, cache_images="cpu").setup(device="cpu", test_mode="val")
        np.testing.assert_array_equal(manager.train_dataparser_outputs.dataparser_transform.numpy(), np.eye(4)[:3])
        self.assertEqual(manager.train_dataparser_outputs.dataparser_scale, 1.)
        manager._initialize_lidar()
        depth, supported = manager._load_lidar_frame(0)
        self.assertEqual(depth.device.type, "cpu")
        self.assertEqual(supported.device.type, "cpu")
        self.assertEqual(tuple(depth.shape), (3, 4, 1))
        self.assertTrue(np.all(depth.numpy()[~supported.numpy()] == 0))
        for dataset, key in ((manager.train_dataset, "train_filenames"), (manager.eval_dataset, "val_filenames")):
            meta = json.loads((self.out / "transforms.json").read_text())
            self.assertEqual({str(p.relative_to(self.out)) for p in dataset.image_filenames}, set(meta[key]))


if __name__ == "__main__":
    unittest.main()

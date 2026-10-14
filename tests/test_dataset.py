"""Read exported formats back and test grouping and exclusive writer safety."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from orbvi_gaussian.dataset import export_dataset
from orbvi_gaussian.geometry import quat_xyzw_to_matrix


def data_lines(path):
    return [line for line in Path(path).read_text().splitlines()
            if line and not line.startswith("#")]


class ExportTests(unittest.TestCase):
    def setUp(self):
        # Keep implicit test exports inside the project even without TMPDIR.
        temporary_root = Path(__file__).resolve().parents[1] / "runtime" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="dataset_test_", dir=str(temporary_root))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "export"
        (self.root / "images").mkdir(parents=True)

    def frames(self, groups=10, views=3):
        frames = []
        for group in range(groups):
            for view in range(views):
                name = "images/{:02d}_{:02d}.png".format(group, view)
                self.assertTrue(cv2.imwrite(str(self.root / name), np.full((6, 8, 3), 30, np.uint8)))
                T = np.eye(4)
                T[:3, :3] = Rotation.from_euler("xyz", [view * 10, group, 30], degrees=True).as_matrix()
                T[:3, 3] = [group, view, 2.5]
                frames.append(dict(file_path=name, w=8, h=6, fl_x=7., fl_y=7.5,
                                   cx=3.5, cy=2.5, timestamp_ns=1_700_000_000_000_000_000 + group * 100,
                                   transform_matrix=T))
        return frames

    def metadata(self):
        return json.loads((self.root / "transforms.json").read_text())

    def test_nerfstudio_and_colmap_roundtrip_without_fake_points(self):
        frames = self.frames(4, 2)
        output = export_dataset(self.root, frames, holdout_every=2)
        self.assertEqual(output["frame_count"], 8)
        self.assertEqual(output["camera_count"], 1)
        self.assertEqual(output["point_count"], 0)
        metadata = self.metadata()
        self.assertEqual(metadata["camera_model"], "OPENCV")
        self.assertEqual(metadata["orientation_override"], "none")
        self.assertNotIn("ply_file_path", metadata)
        self.assertFalse((self.root / "points3D.ply").exists())
        assert_array_equal(data_lines(output["paths"]["points3D"]), [])
        for src, dst in zip(frames, metadata["frames"]):
            assert_allclose(np.asarray(dst["transform_matrix"]) @ np.diag([1, -1, -1, 1]),
                            src["transform_matrix"])
            self.assertEqual(dst["timestamp_ns"], src["timestamp_ns"])
        camera = data_lines(output["paths"]["cameras"])[0].split()
        self.assertEqual(camera[:4], ["1", "PINHOLE", "8", "6"])
        assert_allclose(list(map(float, camera[4:])), [7, 7.5, 3.5, 2.5])
        raw_images = Path(output["paths"]["images"]).read_text().splitlines()
        records = [line for line in raw_images if not line.startswith("#")]
        self.assertEqual(len(records), 2 * len(frames))
        for index, frame in enumerate(frames):
            self.assertEqual(records[index * 2 + 1], "")
            words = records[index * 2].split()
            self.assertEqual(len(words), 10)
            q = np.array(list(map(float, words[1:5])))
            R = quat_xyzw_to_matrix(q[[1, 2, 3, 0]])
            t = np.array(list(map(float, words[5:8])))
            assert_allclose(R @ frame["transform_matrix"][:3, :3], np.eye(3), atol=1e-14)
            assert_allclose(-R.T @ t, frame["transform_matrix"][:3, 3], atol=1e-14)
            self.assertEqual(words[9], Path(frame["file_path"]).name)

    def test_time_groups_holdout_without_virtual_view_leakage(self):
        frames = self.frames(10, 3)
        export_dataset(self.root, list(reversed(frames)))
        metadata = self.metadata()
        train, val = set(metadata["train_filenames"]), set(metadata["val_filenames"])
        self.assertFalse(train & val)
        self.assertEqual(len(val), 6)  # time groups 0 and 8, not every eighth image
        self.assertEqual(train | val, {f["file_path"] for f in frames})
        for group in range(10):
            paths = {f["file_path"] for f in frames[group * 3:(group + 1) * 3]}
            self.assertTrue(paths <= train or paths <= val)
        self.assertEqual(metadata["test_filenames"], [])

    def test_group_ids_join_asynchronous_cameras_and_equal_timestamps(self):
        frames = self.frames(4, 2)
        for index, frame in enumerate(frames):
            frame["group_id"] = "time{}".format(index // 2)
            frame["timestamp_ns"] += index % 2
        # Merge groups 0 and 1 through a shared timestamp despite different IDs.
        frames[2]["timestamp_ns"] = frames[1]["timestamp_ns"]
        output = export_dataset(self.root, frames, holdout_every=2)
        self.assertEqual(output["group_count"], 3)
        metadata = self.metadata()
        val = set(metadata["val_filenames"])
        self.assertTrue({f["file_path"] for f in frames[:4]} <= val)
        self.assertFalse({f["file_path"] for f in frames[:4]} & set(metadata["train_filenames"]))

    def test_single_group_or_disabled_holdout_retains_training(self):
        frames = self.frames(1, 3)
        output = export_dataset(self.root, frames)
        self.assertEqual(output["train_count"], 3)
        self.assertEqual(output["val_count"], 0)

    def test_disabled_holdout(self):
        frames = self.frames(3, 1)
        output = export_dataset(self.root, frames, holdout_every=0)
        self.assertEqual(output["train_count"], 3)
        self.assertEqual(output["val_count"], 0)

    def test_multiple_pinhole_intrinsics_get_distinct_camera_ids(self):
        frames = self.frames(1, 3)
        frames[1]["fl_x"] = 7.1
        output = export_dataset(self.root, frames)
        self.assertEqual(output["camera_count"], 2)
        lines = data_lines(output["paths"]["images"])
        self.assertEqual([line.split()[8] for line in lines], ["1", "2", "1"])

    def test_real_colored_points_exported_with_no_fabricated_observations(self):
        frames = self.frames(1, 1)
        xyz = np.array([[1.25, -3, 5], [-.001, 0, 20]])
        rgb = np.array([[10, 20, 255], [255, 0, 30]], dtype=np.uint8)
        output = export_dataset(self.root, frames, xyz, rgb)
        self.assertEqual(output["point_count"], 2)
        self.assertEqual(self.metadata()["ply_file_path"], "points3D.ply")
        lines = data_lines(output["paths"]["points3D"])
        for index, line in enumerate(lines):
            words = line.split()
            self.assertEqual(len(words), 8)  # fields only; no fabricated TRACK
            assert_allclose(list(map(float, words[1:4])), xyz[index])
            assert_array_equal(list(map(int, words[4:7])), rgb[index])
        ply = Path(output["paths"]["point_cloud"]).read_text()
        self.assertIn("element vertex 2\n", ply)
        self.assertIn("property uchar red", ply)
        rows = ply.split("end_header\n")[1].splitlines()
        assert_allclose([list(map(float, row.split()[:3])) for row in rows], xyz)
        self.assertFalse(self.metadata()["point_cloud_has_observations"])

    def test_uncolored_points_kept_without_claiming_measured_colors(self):
        frames = self.frames(1, 1)
        export_dataset(self.root, frames, [[2, 3, 4]])
        metadata = self.metadata()
        self.assertFalse(metadata["point_cloud_has_rgb"])
        self.assertNotIn("ply_file_path", metadata)
        text = (self.root / "sparse/0/points3D.txt").read_text()
        self.assertIn("unavailable-color placeholder", text)
        self.assertIn("unavailable-error placeholder", text)
        self.assertNotIn("property uchar", (self.root / "points3D.ply").read_text())

    def test_all_existing_output_targets_are_protected_before_any_write(self):
        frames = self.frames(1, 1)
        for relative in ("transforms.json", "sparse/0/cameras.txt", "sparse/0/images.txt",
                         "sparse/0/points3D.txt", "points3D.ply"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("user data")
            with self.subTest(relative=relative), self.assertRaises(FileExistsError):
                export_dataset(self.root, frames, [[1, 2, 3]])
            self.assertEqual(target.read_text(), "user data")
            target.unlink()
            self.assertFalse((self.root / "transforms.json").exists())

    def test_dangling_target_and_directory_symlinks_rejected(self):
        frames = self.frames(1, 1)
        link = self.root / "transforms.json"
        link.symlink_to(self.root / "nonexistent")
        with self.assertRaises(FileExistsError):
            export_dataset(self.root, frames)
        self.assertTrue(link.is_symlink())
        link.unlink()
        (self.root / "sparse").symlink_to(Path(self.temp.name))
        with self.assertRaises(ValueError):
            export_dataset(self.root, frames)

    def test_failed_serialization_rolls_back_only_new_outputs(self):
        frames = self.frames(1, 1)
        asset = self.root / frames[0]["file_path"]
        original = asset.read_bytes()
        with patch("orbvi_gaussian.dataset.json.dump", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                export_dataset(self.root, frames)
        self.assertFalse((self.root / "transforms.json").exists())
        self.assertEqual(list((self.root / "sparse/0").iterdir()), [])
        self.assertEqual(asset.read_bytes(), original)

    def test_exclusive_open_race_preserves_another_writers_file(self):
        frames = self.frames(1, 1)
        original_open = Path.open

        def racing_open(path, mode="r", *args, **kwargs):
            if path.name == "cameras.txt" and mode == "x":
                with original_open(path, "w") as stream:
                    stream.write("other writer")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", racing_open), self.assertRaises(FileExistsError):
            export_dataset(self.root, frames)
        self.assertFalse((self.root / "transforms.json").exists())
        self.assertEqual((self.root / "sparse/0/cameras.txt").read_text(), "other writer")

    def test_invalid_frames_fail_before_writing(self):
        frames = self.frames(1, 1)
        changes = [dict(timestamp_ns=1.5), dict(fl_x=0), dict(cy=np.nan), dict(w=9),
                   dict(file_path="../outside.png"), dict(file_path="/tmp/outside.png"),
                   dict(file_path="images/missing.png"), dict(group_id=[]),
                   dict(transform_matrix=np.diag([-1, 1, 1, 1]))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                export_dataset(self.root, [dict(frames[0], **change)])
            self.assertFalse((self.root / "transforms.json").exists())
        missing = dict(frames[0]); missing.pop("timestamp_ns")
        for bad_frames in ([], [missing], [frames[0], frames[0]]):
            with self.assertRaises(ValueError):
                export_dataset(self.root, bad_frames)

    def test_invalid_points_and_holdout_rejected_before_write(self):
        frames = self.frames(1, 1)
        for xyz, rgb in ((None, [[1, 2, 3]]), ([[np.nan, 1, 2]], None),
                         ([[1, 2]], None), ([[1, 2, 3]], [[256, 0, 0]]),
                         ([[1, 2, 3]], [[.5, 2, 3]]), ([[1, 2, 3]], [[1, 2]])):
            with self.subTest(xyz=xyz, rgb=rgb), self.assertRaises(ValueError):
                export_dataset(self.root, frames, xyz, rgb)
        for every in (1, -1, 2.5, True):
            with self.subTest(every=every), self.assertRaises(ValueError):
                export_dataset(self.root, frames, holdout_every=every)
        self.assertFalse((self.root / "transforms.json").exists())

    def test_masks_present_for_every_frame_and_binary(self):
        frames = self.frames(1, 2)
        (self.root / "masks").mkdir()
        for index, frame in enumerate(frames):
            path = "masks/{}.png".format(index)
            self.assertTrue(cv2.imwrite(str(self.root / path), np.full((6, 8), 255, np.uint8)))
            frame["mask_path"] = path
        with self.assertRaises(ValueError):
            export_dataset(self.root, [frames[0], {k: v for k, v in frames[1].items() if k != "mask_path"}])
        self.assertTrue(cv2.imwrite(str(self.root / frames[1]["mask_path"]), np.full((6, 8), 127, np.uint8)))
        with self.assertRaises(ValueError):
            export_dataset(self.root, frames)
        self.assertTrue(cv2.imwrite(str(self.root / frames[1]["mask_path"]), np.full((6, 8), 255, np.uint8)))
        export_dataset(self.root, frames)
        self.assertTrue(all("mask_path" in f for f in self.metadata()["frames"]))

    def test_source_provenance_and_lidar_paths_are_preserved(self):
        frames = self.frames(1, 1)
        (self.root / "depth").mkdir()
        np.save(str(self.root / "depth/0.npy"), np.full((6, 8), 2., dtype=np.float32))
        self.assertTrue(cv2.imwrite(str(self.root / "depth/0_mask.png"), np.full((6, 8), 255, np.uint8)))
        provenance = dict(source_header_ns=np.int64(1_720_267_880_900_293_988),
                          source_bag_ns=np.int64(1_786_005_282_650_293_988),
                          source_file_path="front_raw/000012.png", camera_id="front",
                          view_id="left", lidar_depth_path="depth/0.npy",
                          lidar_depth_mask_path="depth/0_mask.png")
        frames[0].update(provenance)
        export_dataset(self.root, frames)
        actual = self.metadata()["frames"][0]
        for name, expected in provenance.items():
            self.assertEqual(actual[name], expected)
        self.assertFalse((self.root / provenance["source_file_path"]).exists())
        self.assertNotIn("depth_file_path", actual)

    def test_provenance_types_and_depth_asset_existence_are_validated(self):
        frames = self.frames(1, 1)
        for field in (dict(source_header_ns=1.5), dict(source_bag_ns=True),
                      dict(camera_id=3), dict(view_id=[]), dict(source_file_path=""),
                      dict(lidar_depth_path="depth/missing.npy"),
                      dict(lidar_depth_mask_path="../outside.png")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                export_dataset(self.root, [dict(frames[0], **field)])
        self.assertFalse((self.root / "transforms.json").exists())


if __name__ == "__main__":
    unittest.main()

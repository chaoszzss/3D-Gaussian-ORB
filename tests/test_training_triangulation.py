"""Independent geometry, heldout exclusion, native-image and real-LK regressions."""
import copy
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_tri", ROOT / "scripts/triangulate_training_tracks.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
cv2.setNumThreads(1)


def frame(i, center, camera_id="camera_2", fx=640.):
    T = np.eye(4); T[:3, 3] = center
    ns = T @ np.diag([1., -1., -1., 1.])
    return dict(file_path=f"images/{i}.png", mask_path=f"masks/{i}.png",
                w=1280, h=1080, fl_x=fx, fl_y=fx, cx=640., cy=540.,
                camera_id=camera_id, view_id="center", group_id=str(i),
                timestamp_ns=1720000000000000000 + i * 500000000,
                source_header_ns=1720000000000000000 + i * 500000000,
                source_bag_ns=1780000000000000000 + i * 500000000,
                transform_matrix=ns.tolist())


def cameras(baseline=.5):
    return [M.camera(frame(i, [i * baseline, 0, 0])) for i in range(3)]


def test_half_pixel_and_fixed_pose_conversion():
    T, K, P = M.camera(frame(0, [2., 3., 4.]))
    np.testing.assert_array_equal(T[:3, :3], np.eye(3))
    np.testing.assert_array_equal(T[:3, 3], [2., 3., 4.])
    assert (K[0, 2], K[1, 2]) == (639.5, 539.5)
    np.testing.assert_array_equal(P[:, 3], [-2., -3., -4.])


def test_exact_point_retains_metric_world_without_plane_projection():
    cams = cameras(); point = np.array([.3, -1.34789123, 5.1234567])
    uv, _ = M.project_point(point, cams)
    result, reason = M.triangulate_track(uv, cams)
    assert reason is None
    np.testing.assert_allclose(result["xyz"], point, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result["xyz_first_two"], point, atol=1e-10, rtol=0)
    assert max(result["first_two_solution_reprojection_px"]) < 1e-9
    assert min(result["parallax_deg"]) > 1


def test_bad_independent_third_cannot_be_rescued_by_refinement(monkeypatch):
    cams = cameras(); uv, _ = M.project_point([.3, -.8, 5.], cams)
    uv[2, 0] += 2.
    def forbidden(*args):
        raise AssertionError("Third-view-invalid track reached all-three refinement")
    monkeypatch.setattr(M, "refine_three", forbidden)
    result, reason = M.triangulate_track(uv, cams)
    assert result is None and reason == "independent_third_view_reprojection"


def test_third_has_no_influence_on_initial_two_view_point():
    cams = cameras(); point = [.3, -.8, 5.]
    uv, _ = M.project_point(point, cams); uv[2, 0] += .7
    result, reason = M.triangulate_track(uv, cams)
    assert reason is None
    np.testing.assert_allclose(result["xyz_first_two"], point, atol=1e-10, rtol=0)
    assert result["first_two_solution_reprojection_px"][2] == pytest.approx(.7)
    assert max(result["reprojection_px"]) <= 1.5


@pytest.mark.parametrize("depth", [-5., .19, 50.01])
def test_cheirality_and_depth_gates(depth):
    cams = cameras(); uv, _ = M.project_point([.3, -.1, depth], cams)
    result, reason = M.triangulate_track(uv, cams)
    assert result is None and reason == "first_pair_cheirality_or_depth"


def test_low_parallax_rejected_even_when_projection_is_exact():
    cams = cameras(.01); uv, _ = M.project_point([.3, -.1, 10.], cams)
    result, reason = M.triangulate_track(uv, cams)
    assert result is None and reason == "low_parallax"


def test_refined_xyz_must_also_have_one_degree_parallax(monkeypatch):
    cams = cameras(.3); uv, _ = M.project_point([.3, 0, 16.5], cams)
    monkeypatch.setattr(M, "refine_three", lambda *args: (np.array([.3, 0, 17.3]), True))
    result, reason = M.triangulate_track(uv, cams)
    assert result is None and reason == "refined_geometric_low_parallax"


def test_degenerate_equal_centers_are_not_filled():
    cams = cameras(0.); uv, _ = M.project_point([.3, -.1, 5.], cams)
    result, reason = M.triangulate_track(uv, cams)
    assert result is None and reason == "degenerate_first_pair"


def test_pose_triplets_same_camera_time_and_baseline_limits():
    fs = [frame(i, [.4 * i, 0, 0], cid) for cid in ("camera_2", "camera_3") for i in range(15)]
    plan, stats = M.select_triplets(fs)
    assert 0 < len(plan) <= 16
    assert set(stats) == {"camera_2", "camera_3"}
    for row in plan:
        chosen = [fs[i] for i in row["train_indices"]]
        assert len({f["camera_id"] for f in chosen}) == 1
        assert chosen[0]["timestamp_ns"] < chosen[1]["timestamp_ns"] < chosen[2]["timestamp_ns"]
        assert min(row["baseline_m"]) >= .3 and max(row["baseline_m"]) <= 2
    no_baseline = [frame(i, [0, 0, 0]) for i in range(6)]
    assert M.select_triplets(no_baseline)[0] == []
    with pytest.raises(ValueError):
        M.select_triplets(fs, max_triplets=33)


@pytest.fixture
def dataset(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    fs = [frame(i, [.5 * i, 0, 0]) for i in range(5)]
    meta = dict(pixel_center_convention="half_integer; +.5 from OpenCV", frames=fs,
                train_filenames=[f["file_path"] for f in fs[:3]],
                val_filenames=[fs[3]["file_path"]], test_filenames=[fs[4]["file_path"]])
    (data / "images").mkdir(); (data / "masks").mkdir()
    for f in fs[:3]:
        assert cv2.imwrite(str(data/f["file_path"]), np.zeros((1080, 1280, 3), np.uint8))
        assert cv2.imwrite(str(data/f["mask_path"]), np.full((1080, 1280), 255, np.uint8))
    # DEV/TEST image files deliberately do not exist: opening them must fail.
    (data / "transforms.json").write_text(json.dumps(meta))
    (data / "preparation.json").write_text(json.dumps({"status":"prepared_experimental"}))
    return data, meta


def test_only_train_assets_loaded_heldout_files_never_required(dataset):
    data, meta = dataset
    fs, _, _ = M.load_training(data)
    assert len(fs) == 3 and {f["file_path"] for f in fs} == set(meta["train_filenames"])
    cache = M.ImageCache(data, fs)
    cache.get(0)
    assert len(cache.hashes) == 2
    assert all("/0.png" in str(p) for p in cache.hashes)


def test_group_leakage_is_rejected_before_matching(dataset):
    data, meta = dataset
    meta["frames"][3]["group_id"] = meta["frames"][0]["group_id"]
    (data / "transforms.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="heldout"):
        M.load_training(data)


def test_actual_half_resolution_image_is_rejected_not_enlarged(dataset):
    data, _ = dataset; fs, _, _ = M.load_training(data)
    cv2.imwrite(str(data/fs[0]["file_path"]), np.zeros((540, 640, 3), np.uint8))
    with pytest.raises(ValueError, match="native1280"):
        M.ImageCache(data, fs).get(0)


def test_mask_excludes_invalid_pixels_and_image_edges():
    mask = np.zeros((20, 20), np.uint8); mask[5:15, 5:15] = 255
    valid = M.pixels_in_mask(np.array([[10, 10], [0, 0], [-1, 10], [20, 10], [np.nan, 2]]), mask)
    assert valid.tolist() == [True, False, False, False, False]


def test_real_forward_backward_lk_and_cycle_on_translated_texture():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (256, 320), np.uint8)
    a = cv2.GaussianBlur(a, (3, 3), .7)
    mask = np.zeros_like(a); mask[25:-25, 25:-25] = 255
    images = []
    for shift in (0., -5., -10.):
        gray = cv2.warpAffine(a, np.array([[1., 0., shift], [0., 1., 0.]]), (320, 256))
        images.append((gray, np.repeat(gray[:, :, None], 3, axis=2), mask))
    tracks, stats = M.track_triplet(images)
    assert len(tracks) > 50
    displacements = np.array([t["uv"][2] - t["uv"][0] for t in tracks])
    np.testing.assert_allclose(np.median(displacements, axis=0), [-10, 0], atol=.08)
    assert all(max(t["forward_backward_px"]) <= .75 and t["cycle_px"] <= .75 for t in tracks)


def test_dedup_keeps_original_xyz_without_averaging():
    t = dict(train_indices=[0, 1, 2], uv=[[10, 20], [15, 20], [20, 20]],
             xyz=[.123456789, -.4321, 8.76543], reprojection_px=[.1, .2, .1], parallax_deg=[3, 4, 7])
    duplicate = copy.deepcopy(t); duplicate["xyz"] = [6, 7, 8]; duplicate["reprojection_px"] = [.3, .4, .3]
    assert M.deduplicate_observations([duplicate, t]) == [t]


def test_geometric_floor_band_uses_fixed_xyz_not_image_half():
    xyz = [[1, 0, -1.3], [4, 0, -1.3], [1, 0, .5], [0, 0, -3.1]]
    xy, dz, mask, nearest = M.floor_band(xyz, [[0, 0, 0]])
    assert mask.tolist() == [True, False, False, False]
    assert M.floor_band([], [[0, 0, 0]])[2].shape == (0,)


def test_point_ply_roundtrips_float64_and_does_not_overwrite(tmp_path):
    xyz = np.array([[.12345678912345678, -1.347891231231, 5.12345671234567]])
    out = tmp_path / "points.ply"
    M.write_ply(out, xyz, [[20, 30, 40]])
    row = out.read_text().split("end_header\n")[1].split()
    np.testing.assert_array_equal(np.array(row[:3], float), xyz[0])
    with pytest.raises(FileExistsError):
        M.write_ply(out, xyz, [[20, 30, 40]])

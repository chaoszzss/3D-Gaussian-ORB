"""CPU-only real-file fixtures for frozen DEV identity and ego mask contracts."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_acceptance_plan as M


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def scene(tmp_path, monkeypatch):
    import cv2
    import orbvi_gaussian.calibration as calibration
    import orbvi_gaussian.workflow as workflow
    cfg = yaml.safe_load((ROOT / "configs/acceptance.experimental.yaml").read_text())
    cfg.update(expected_blocks=2, expected_unique_frames=6, floor_labels={"F1": [0, 1]})
    base, revision = {"cameras": {}, "virtual_views": [{"name": "center"}]}, {"cameras": {}, "virtual_views": [{"name": "center"}]}
    for cid in ("camera_2", "camera_3"):
        raw = tmp_path / "raw" / (cid + ".png")
        raw.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(raw), np.ones((8, 8), np.uint8) * 255)
        base["cameras"][cid] = {"width": 8, "height": 8}
        revision["cameras"][cid] = {"width": 8, "height": 8, "source_mask_path": str(raw)}
        cfg["raw_mask_sha256"][cid] = M.sha256_file(raw)
    put(tmp_path / "base.json", base)
    put(tmp_path / cfg["revision_config"], revision)
    cfg["revision_sha256"] = M.sha256_file(tmp_path / cfg["revision_config"])
    config_path = tmp_path / "configs/acceptance.experimental.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(calibration, "load_calibration", lambda p, require_verified: json.loads(Path(p).read_text()))
    mask = np.ones((8, 8), np.uint8) * 255
    mask[-2:] = 0
    monkeypatch.setattr(workflow, "_rectify", lambda image, camera, view: (image, mask, None, (8, 8, 4., 4., 3.5, 3.5)))
    for bid, groups in (("block_000", [101, 102]), ("block_001", [102, 103])):
        folder = tmp_path / cfg["dataset_root"] / bid
        frames = []
        for group in groups:
            for cid in ("camera_2", "camera_3"):
                name = f"{group}_{cid}"
                frame = {"file_path": "images/" + name + ".png", "group_id": str(group), "camera_id": cid,
                         "view_id": "center", "timestamp_ns": group, "w": 8, "h": 8,
                         "fl_x": 4., "fl_y": 4., "cx": 4., "cy": 4.,
                         "transform_matrix": np.eye(4).tolist(), "mask_path": "masks/" + name + ".png",
                         "lidar_depth_path": "depth/" + name + ".npy", "lidar_depth_mask_path": "support/" + name + ".png"}
                for key in ("file_path", "mask_path", "lidar_depth_mask_path"):
                    p = folder / frame[key]; p.parent.mkdir(parents=True, exist_ok=True)
                    value = np.ones((8, 8), np.uint8) * 255
                    if key == "mask_path": value[0] = 0
                    assert cv2.imwrite(str(p), value)
                depth = folder / frame["lidar_depth_path"]; depth.parent.mkdir(exist_ok=True)
                np.save(depth, np.ones((8, 8), np.float32) * (1 if bid == "block_000" else 2))
                frames.append(frame)
        put(folder / "transforms.json", {"frames": frames, "val_filenames": [f["file_path"] for f in frames],
                                          "train_filenames": [], "test_filenames": []})
        put(folder / "preparation.json", {"calibration": str(tmp_path / "base.json"),
                                           "calibration_sha256": M.sha256_file(tmp_path / "base.json")})
    return tmp_path, config_path


def test_actual_dedup_and_fixed_masks_are_reproducible(scene):
    root, config = scene
    plan = M.build_plan(root, config)
    assert plan["source_observations"] == 8 and plan["unique_frames"] == 6
    assert plan["duplicate_observation_count"] == 2
    assert [j["frame_indices"] for j in plan["jobs"]] == [[0, 1, 2, 3], [2, 3]]
    assert all(n == 40 for n in plan["mask_valid_pixels"].values())
    assert len(plan["mask_valid_pixels"]) == 2
    assert M.build_plan(root, config) == plan
    path = root / "plan.json"
    M.write_plan(path, plan); M.write_plan(path, plan)
    assert M.load_plan(path) == plan
    with pytest.raises(ValueError, match="Frozen output differs"):
        M.write_plan(path, {"wrong": True})


def test_explicit_discontiguous_pilot_blocks_stay_strict(scene):
    root, config = scene
    old = root / 'data/garage_surface_v1/block_001'
    old.rename(old.with_name('block_015'))
    cfg = yaml.safe_load(config.read_text())
    cfg.update(block_ids=['block_000', 'block_015'], floor_labels={'F1': [0], 'ramp': [15]})
    config.write_text(yaml.safe_dump(cfg))
    plan = M.build_plan(root, config)
    assert [j['block_id'] for j in plan['jobs']] == ['block_000', 'block_015']
    cfg['block_ids'] = ['block_000', 'block_000']
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='Explicit blocks'):
        M.build_plan(root, config)


@pytest.mark.parametrize("fault", ["missing_asset", "duplicate", "leak", "conflict", "missing_block", "count"])
def test_bad_actual_dataset_rejected(scene, fault):
    root, config = scene
    path = root / "data/garage_surface_v1/block_001/transforms.json"
    meta = json.loads(path.read_text())
    if fault == "missing_asset":
        (path.parent / meta["frames"][0]["file_path"]).unlink()
    elif fault == "duplicate":
        meta["val_filenames"].append(meta["val_filenames"][0]); put(path, meta)
    elif fault == "leak":
        meta["train_filenames"] = [meta["val_filenames"][0]]; put(path, meta)
    elif fault == "conflict":
        meta["frames"][0]["transform_matrix"][0][3] = 2; put(path, meta)
    elif fault == "missing_block":
        path.unlink()
    else:
        cfg = yaml.safe_load(config.read_text()); cfg["expected_unique_frames"] = 408; config.write_text(yaml.safe_dump(cfg))
    with pytest.raises((ValueError, FileNotFoundError)):
        M.build_plan(root, config)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "job", "digest", "mask_asset", "raw_revision"])
def test_plan_integrity_and_assets_fail_closed(scene, fault):
    root, config = scene
    plan = M.build_plan(root, config)
    if fault == "missing": plan["frames"].pop()
    elif fault == "duplicate": plan["frames"][-1] = copy.deepcopy(plan["frames"][0])
    elif fault == "job": plan["jobs"][0]["frame_indices"].pop()
    elif fault == "digest": plan["plan_id"] = "0" * 64
    elif fault == "mask_asset": Path(plan["frames"][0]["mask_path"]).write_bytes(b"changed")
    else: Path(plan["raw_masks"]["camera_2"]["path"]).unlink()
    if fault != "digest": plan["plan_id"] = M.object_sha({k: v for k, v in plan.items() if k != "plan_id"})
    with pytest.raises(ValueError): M.validate_plan(plan)


def test_mask_revision_is_content_pinned_not_a_label(scene):
    root, config = scene
    revision = json.loads((root / "configs/direct_lio_refined.ego_v2.experimental.yaml").read_text())
    Path(revision["cameras"]["camera_3"]["source_mask_path"]).write_bytes(b"not rev2")
    with pytest.raises(ValueError, match="rev2 raw mask"): M.build_plan(root, config)


def test_frame_source_checks_pose_and_dataset_order(scene):
    root, config = scene
    plan = M.build_plan(root, config)
    f = plan["frames"][0]
    source = json.loads(Path(f["transforms"]).read_text())["frames"][0]
    M.validate_frame_source(f, source, 0)
    with pytest.raises(ValueError, match="ordering"): M.validate_frame_source(f, source, 2)
    source["cx"] += .5
    with pytest.raises(ValueError, match="camera changed"): M.validate_frame_source(f, source)

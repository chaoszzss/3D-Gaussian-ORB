"""CPU hemisphere evidence tests; no CUDA imports or renderer execution."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_direction_support as M
from assemble_core_gaussians import InputGuard, ply_header, sha256


def camera():
    return {"transform_matrix": np.diag([1., -1., -1., 1.]).tolist(),
            "w": 9, "h": 9, "fl_x": 8., "fl_y": 8., "cx": 4.5, "cy": 4.5}


def test_weak_opposite_side_is_never_silently_discarded():
    positive = np.array([3, 0, 3, 1, 2, 1, 0])
    negative = np.array([0, 3, 1, 1, 0, 0, 0])
    assert M.side_flags(positive, negative).tolist() == [1, 2, 3, 3, 0, 0, 0]
    assert M.side_flags(positive, negative).dtype == np.uint8


def test_ns_halfpixel_projection_normal_sign_masks_and_unknown_depth():
    frame = camera()
    xyz = np.array([[0., 0., 3.], [0., 0., 3.]])
    normals = np.array([[0., 0., 1.], [0., 0., -1.]])
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[4, 4] = 3.
    mask = (depth>0).astype(np.uint8)*255
    pos, neg = M.frame_evidence(xyz, normals, frame, depth, mask, mask)
    assert pos.tolist() == [1] and neg.tolist() == [0]
    for invalid in (np.zeros_like(mask),):
        p, n = M.frame_evidence(xyz, normals, frame, depth, mask, invalid)
        assert not len(p) and not len(n)
    depth[4, 4] = 5.
    p, n = M.frame_evidence(xyz, normals, frame, depth, mask, mask)
    assert not len(p) and not len(n)  # No foreground support cannot imply one side.


def test_grazing_evidence_does_not_choose_a_hemisphere():
    xyz = np.array([[0., 0., 3.]])
    normal = np.array([[1., 0., 0.]])
    depth = np.full((9, 9), 3., np.float32)
    mask = np.ones((9, 9), np.uint8)
    p, n = M.frame_evidence(xyz, normal, camera(), depth, mask, mask)
    assert not len(p) and not len(n)


def test_train_groups_are_deduplicated_and_heldout_never_counted():
    with tempfile.TemporaryDirectory(prefix="direction_groups_v2_", dir=ROOT/"reports") as directory:
        root = Path(directory)
        base = root/"data/garage_surface_v1/block_018"
        base.mkdir(parents=True)
        names = ["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"]
        names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        rows = np.zeros((1, len(names)), dtype="<f4")
        rows[0, 2] = 3.; rows[0, names.index("rot_0")] = 1.
        path = root/"source.ply"
        path.write_bytes(ply_header(names, 1)+rows.tobytes())
        frames = []
        for group, split in ((1, "train"), (2, "train"), (3, "val")):
            for cam in (2, 3):
                stem = f"{group}_{cam}"
                frame = dict(camera(), file_path=f"{stem}.png", group_id=str(group), camera_id=f"camera_{cam}",
                             lidar_depth_path=f"{stem}.npy", lidar_depth_mask_path=f"{stem}_depth.png", mask_path=f"{stem}_mask.png")
                frames.append(frame)
                np.save(base/frame["lidar_depth_path"], np.full((9, 9), 3., dtype=np.float32))
                for key in ("lidar_depth_mask_path", "mask_path"):
                    Image.fromarray(np.full((9, 9), 255, np.uint8)).save(base/frame[key])
        meta = {"frames": frames, "train_filenames": [f["file_path"] for f in frames if f["group_id"]!="3"],
                "val_filenames": [f["file_path"] for f in frames if f["group_id"]=="3"], "test_filenames": []}
        (base/"transforms.json").write_text(json.dumps(meta))
        ref = base/"lidar_reference.npz"
        np.savez(ref, xyz=np.array([[0., 0., 3.]]))
        source = {"id": "block_018", "core_ids": ["block_018"], "path": "source.ply", "source_sha256": sha256(path),
                  "measured_reference": ref.relative_to(root).as_posix()}
        positive, negative, report = M.source_evidence(root, source, np.array([0]), InputGuard(root))
        assert positive.tolist() == [0] and negative.tolist() == [2]
        assert report["train_groups"] == 2 and report["train_frames"] == 4
        assert report["unknown"] == 1  # Four images from two groups cannot pass minimum three.

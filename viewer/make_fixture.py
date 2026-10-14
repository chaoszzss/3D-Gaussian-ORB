#!/usr/bin/env python3
"""Create small, explicitly SYNTHETIC Gaussian fixtures. NOT a real garage map."""
import sys
sys.dont_write_bytecode = True
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
C0 = 0.28209479177387814


def fixture(offset, degree):
    """Colored anisotropic ellipsoids with varying wxyz rotation, log scales, logits."""
    vertices = []
    nrest = 3 * ((degree + 1) ** 2 - 1)
    for ix in range(21):
        for iy in range(21):
            x, y = (ix - 10) * 0.22, (iy - 10) * 0.22
            z = 0.45 * math.cos(x * 1.6) * math.cos(y * 1.6)
            angle = (ix + iy) * 0.11
            rgb = [0.3 + 0.6 * ix / 20, 0.3 + 0.6 * iy / 20, 0.8]
            dc = [(c - 0.5) / C0 for c in rgb]
            sh = [0.0] * nrest
            if sh:
                sh[0] = 0.22
                sh[(degree + 1) ** 2 - 1] = -0.15
                sh[-1] = 0.12
            row = [x + offset[0], y + offset[1], z + offset[2], 0, 0, 1] + dc + sh
            row += [math.log(0.85 / 0.15), math.log(0.16), math.log(0.065), math.log(0.035),
                    math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
            vertices.append(row)
    # A vertical band gives each view direction a visibly different silhouette.
    for iz in range(21):
        for ix in range(21):
            x, z = (ix - 10) * 0.17, iz * 0.12
            dc = [(c - 0.5) / C0 for c in (0.9, 0.5, 0.22)]
            vertices.append([x + offset[0], 0.5 + offset[1], z + offset[2], 0, 1, 0] + dc + [0.08] * nrest +
                            [2.1, math.log(0.11), math.log(0.035), math.log(0.09), 0.9238795, 0.3826834, 0, 0])
    return vertices


def write_ply(path, vertices, degree, ascii_format=False):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += ["f_rest_" + str(i) for i in range(3 * ((degree + 1)**2 - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ["ply", "format " + ("ascii" if ascii_format else "binary_little_endian") + " 1.0",
              "comment SYNTHETIC SELF-TEST ONLY - NOT REAL GARAGE", "element vertex " + str(len(vertices))]
    header += ["property float " + name for name in names] + ["end_header"]
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for row in vertices:
            handle.write((" ".join(format(v, ".8g") for v in row) + "\n").encode("ascii") if ascii_format else struct.pack("<" + "f" * len(row), *row))


def main():
    out = HERE / "fixtures"
    out.mkdir(exist_ok=True)
    blocks = []
    specs = [("synthetic_a", "selftest_lower", [0, 0, 0], 3, False),
             ("synthetic_b", "selftest_lower", [8, 0, 0], 1, False),
             ("synthetic_c", "selftest_upper", [0, 0, 5], 2, False),
             ("synthetic_d", None, [8, 0, 5], 0, False)]
    for block_id, floor, offset, degree, ascii_format in specs:
        vertices = fixture(offset, degree)
        write_ply(out / (block_id + ".ply"), vertices, degree, ascii_format)
        b = {"id": block_id, "label": "合成自检 " + block_id[-1].upper(), "floor_id": floor,
             "floor_verified": False, "path": "viewer/fixtures/" + block_id + ".ply",
             "coordinates": "camera_init", "bbox_m": {"min": [offset[0] - 2.4, offset[1] - 2.4, offset[2] - 0.7],
             "max": [offset[0] + 2.4, offset[1] + 2.4, offset[2] + 2.7]},
             "training": {"status": "synthetic", "method": "procedural_test_fixture", "not_real_data": True},
             "eval": {"status": "not_evaluated", "note": "Rendering self-test only, not reconstruction accuracy"}}
        if block_id == "synthetic_a":
            b["lidar"] = {"path": "viewer/fixtures/synthetic_lidar.ply", "coordinates": "camera_init",
                          "alignment_verified": True, "source": "SYNTHETIC geometric test points, not measured LiDAR"}
        blocks.append(b)
    lidar = out / "synthetic_lidar.ply"
    # Explicit negative compatibility fixture; not registered as a renderable block.
    write_ply(out / "synthetic_ascii_unsupported.ply", fixture([0, 0, 0], 1), 1, True)
    with lidar.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\ncomment SYNTHETIC LIDAR AUXILIARY ONLY\nelement vertex 81\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for ix in range(9):
            for iy in range(9):
                f.write("{} {} {}\n".format((ix - 4) * 0.5, (iy - 4) * 0.5, 0.9))
    manifest = {"schema_version": 1, "title": "合成 Gaussian 自检 · 非实景", "dataset_kind": "synthetic",
                "coordinates": "camera_init", "units": "m", "camera_up": [0, 0, 1], "blocks": blocks}
    (out / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Created 4 SYNTHETIC Gaussian blocks (SH0/1/2/3), 882 splats each; NOT REAL GARAGE.")


if __name__ == "__main__":
    main()

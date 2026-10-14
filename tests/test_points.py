"""Point schema and projection QA; all fixtures live in memory, no temp files."""

import importlib.util
import io
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/orbvi_gaussian/points.py"
_SPEC = importlib.util.spec_from_file_location("points_under_test", _MODULE_PATH)
points = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(points)


def _field(name, offset, datatype, count=1):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count)


def _pc2(endian="<", rgb_type=6):
    # Unordered fields, mixed scalar datatypes, intra-record padding, and eight
    # bytes of row padding filled with invalid values rather than real points.
    fields = [_field("y", 0, 3), _field("rgb", 12, rgb_type),
              _field("z", 16, 7), _field("x", 4, 8), _field("intensity", 20, 2)]
    xyz = [(1.25, -2, 3), (float("nan"), 4, 5), (5.5, 6, 7), (8.5, -10, 11)]
    colour = [0x00123456, 0x00ABCDEF, 0x00010203, 0x00FEFDFC]
    payload = bytearray([255] * 112)
    for index, (coord, packed) in enumerate(zip(xyz, colour)):
        offset = (index // 2) * 56 + (index % 2) * 24
        struct.pack_into(endian + "h", payload, offset, coord[1])
        struct.pack_into(endian + "d", payload, offset + 4, coord[0])
        struct.pack_into(endian + "I", payload, offset + 12, packed)
        struct.pack_into(endian + "f", payload, offset + 16, coord[2])
        payload[offset + 20] = 200
    return SimpleNamespace(width=2, height=2, point_step=24, row_step=56,
                           is_bigendian=endian == ">", fields=fields, data=payload,
                           is_dense=True)


def _read(monkeypatch, suffix, payload, **kwargs):
    # Mock a read-only path; do not create fixtures, pytest tmpdir, or /tmp files.
    path = Path(__file__).with_suffix(suffix)

    def open_memory(candidate, mode):
        assert Path(candidate) == path
        assert mode == "rb"
        return io.BytesIO(payload)

    monkeypatch.setattr("builtins.open", open_memory)
    return points.read_point_cloud(path, **kwargs)


def _pcd_header(mode, count, names="x y z rgb", sizes="4 4 4 4", types="F F F U", counts=None):
    text = "# fixture\nVERSION .7\nFIELDS {}\nSIZE {}\nTYPE {}\n".format(names, sizes, types)
    if counts is not None:
        text += "COUNT {}\n".format(counts)
    text += "WIDTH {}\nHEIGHT 1\nPOINTS {}\nDATA {}\n".format(count, count, mode)
    return text.encode("ascii")


def _lzf_literals(raw):
    return b"".join(bytes([len(raw[i:i + 32]) - 1]) + raw[i:i + 32]
                    for i in range(0, len(raw), 32))


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("rgb_type", [6, 7])
def test_pointcloud2_padding_endian_and_floatpacked(monkeypatch, endian, rgb_type):
    monkeypatch.setattr(points, "_CHUNK_POINTS", 1)
    xyz, rgb = points.decode_pointcloud2(_pc2(endian, rgb_type))
    np.testing.assert_array_equal(xyz, [[1.25, -2, 3], [5.5, 6, 7], [8.5, -10, 11]])
    np.testing.assert_array_equal(rgb, [[18, 52, 86], [1, 2, 3], [254, 253, 252]])
    assert xyz.dtype == np.float64 and rgb.dtype == np.uint8


@pytest.mark.parametrize("code,fmt,value", [(1, "b", -4), (2, "B", 4), (3, "h", -400),
                                           (4, "H", 400), (5, "i", -400000),
                                           (6, "I", 400000), (7, "f", 1.25), (8, "d", 1.25)])
def test_pointcloud2_all_ros_datatypes(code, fmt, value):
    size = struct.calcsize(fmt)
    msg = {"width": 1, "height": 1, "point_step": size * 3, "row_step": size * 3,
           "fields": [_field(name, i * size, code) for i, name in enumerate(("x", "y", "z"))],
           "data": struct.pack("<" + fmt * 3, value, value, value)}
    xyz, rgb = points.decode_pointcloud2(msg)
    np.testing.assert_array_equal(xyz, [[value] * 3])
    assert rgb is None


@pytest.mark.parametrize("endian", ["<", ">"])
def test_pointcloud2_rgba_nan_bits_are_colour(endian):
    msg = {"width": 1, "height": 1, "point_step": 16, "row_step": 16,
           "is_bigendian": endian == ">",
           "fields": [_field(name, i * 4, 7) for i, name in enumerate(("x", "y", "z", "rgba"))],
           "data": struct.pack(endian + "fffI", 1, 2, 3, 0xFFFF8040)}
    xyz, rgb = points.decode_pointcloud2(msg)
    np.testing.assert_array_equal(xyz, [[1, 2, 3]])
    np.testing.assert_array_equal(rgb, [[255, 128, 64]])


def test_pointcloud2_explicit_channels_and_multicount():
    msg = {"width": 1, "height": 1, "point_step": 28, "row_step": 28,
           "fields": [_field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7),
                      _field("red", 12, 4), _field("green", 14, 4), _field("blue", 16, 4),
                      _field("descriptor", 20, 7, 2)],
           "data": struct.pack("<fffHHH2xff", 1, 2, 3, 10, 20, 30, 99, 100)}
    xyz, rgb = points.decode_pointcloud2(msg)
    np.testing.assert_array_equal(rgb, [[10, 20, 30]])
    np.testing.assert_array_equal(xyz, [[1, 2, 3]])
    msg["fields"][0].count = 2
    with pytest.raises(ValueError, match="scalar"):
        points.decode_pointcloud2(msg)


def test_pointcloud2_intensity_is_never_rgb():
    msg = _pc2()
    msg.fields = [field for field in msg.fields if field.name != "rgb"]
    assert points.decode_pointcloud2(msg)[1] is None


@pytest.mark.parametrize("mutation,error", [
    (lambda m: setattr(m, "row_step", 47), "row_step"),
    (lambda m: setattr(m, "data", m.data[:-1]), "truncated"),
    (lambda m: setattr(m.fields[0], "offset", 24), "point_step"),
    (lambda m: setattr(m.fields[0], "datatype", 99), "datatype"),
    (lambda m: m.fields.append(m.fields[0]), "duplicate"),
    (lambda m: setattr(m, "width", -1), "width"),
])
def test_pointcloud2_rejects_bad_schema(mutation, error):
    msg = _pc2()
    mutation(msg)
    with pytest.raises(ValueError, match=error):
        points.decode_pointcloud2(msg)


def _livox_point(x, y=2, z=3, offset=0, tag=0x10):
    return SimpleNamespace(x=x, y=y, z=z, offset_time=offset,
                           reflectivity=237, tag=tag, line=5)


def test_livox_exact_integer_time_and_npz_roundtrip():
    cloud = [_livox_point(1, offset=2**32 - 1, tag=0xFF), _livox_point(float("inf")),
             _livox_point(0, 0, 0), _livox_point(-4, offset=17)]
    msg = SimpleNamespace(point_num=4, timebase=2**64 - 123, points=cloud,
                          header=SimpleNamespace(frame_id="livox_sensor"))
    decoded = points.decode_livox(msg)
    np.testing.assert_array_equal(decoded["xyz"], [[1, 2, 3], [-4, 2, 3]])
    np.testing.assert_array_equal(decoded["offset_ns"], [2**32 - 1, 17])
    np.testing.assert_array_equal(decoded["tag"], [0xFF, 0x10])
    assert decoded["timebase_ns"] == msg.timebase
    assert decoded["offset_ns"].dtype == np.uint32
    assert decoded["reflectivity"].dtype == np.uint8
    assert "rgb" not in decoded
    assert decoded["deskewed"] is False and decoded["is_ground_truth"] is False
    stream = io.BytesIO()
    np.savez_compressed(stream, **decoded)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as loaded:
        assert int(loaded["timebase_ns"]) == 2**64 - 123
        assert loaded["frame_id"].item() == "livox_sensor"
        assert loaded["deskewed"].item() is False


def test_livox_rejects_count_and_invalid_timestamps():
    msg = {"points": [_livox_point(1)], "point_num": 2, "timebase": 1}
    with pytest.raises(ValueError, match="point_num"):
        points.decode_livox(msg)
    msg["point_num"] = 1
    msg["timebase"] = 1.5
    with pytest.raises(ValueError, match="integer"):
        points.decode_livox(msg)
    msg["timebase"] = 1
    msg["points"][0].offset_time = 2**32
    with pytest.raises(ValueError, match="offset_time"):
        points.decode_livox(msg)


@pytest.mark.parametrize("mode", ["ascii", "binary", "binary_compressed"])
@pytest.mark.parametrize("colour_type", ["F", "U"])
def test_pcd_all_storage_modes_float_or_uint_rgb(monkeypatch, mode, colour_type):
    monkeypatch.setattr(points, "_CHUNK_POINTS", 1)
    coord = np.array([[1.25, 2, 3], [4, 5, 6], [np.nan, 7, 8]], dtype="<f4")
    colours = np.array([0x00112233, 0x00A0B0C0, 0x00010203], dtype="<u4")
    colour_field = colours.view("<f4") if colour_type == "F" else colours
    header = _pcd_header(mode, 3, types="F F F " + colour_type)
    if mode == "ascii":
        rows = ["{} {} {} {}".format(*row, format(float(rgb), ".9g") if colour_type == "F" else str(int(rgb)))
                for row, rgb in zip(coord, colour_field)]
        body = ("\n".join(rows) + "\n").encode("ascii")
    elif mode == "binary":
        body = b"".join(struct.pack("<fffI", *row, int(rgb)) for row, rgb in zip(coord, colours))
    else:
        raw = b"".join(coord[:, i].tobytes() for i in range(3)) + colours.tobytes()
        packed = _lzf_literals(raw)
        body = struct.pack("<II", len(packed), len(raw)) + packed
    xyz, rgb = _read(monkeypatch, ".pcd", header + body)
    np.testing.assert_array_equal(xyz, coord[:2])
    np.testing.assert_array_equal(rgb, [[17, 34, 51], [160, 176, 192]])


@pytest.mark.parametrize("mode", ["binary", "binary_compressed", "ascii"])
def test_pcd_count_fields_and_padding(monkeypatch, mode):
    header = _pcd_header(mode, 2, names="x _ descriptor z y rgb", sizes="4 1 4 8 2 4",
                         types="F U F F I U", counts="1 3 2 1 1 1")
    if mode == "binary":
        body = struct.pack("<f3B2fdhI", 1, 255, 254, 253, 100, 200, 3, 2, 0x00FF0000)
        body += struct.pack("<f3B2fdhI", 4, 255, 254, 253, 101, 201, 6, 5, 0x0000FF00)
    elif mode == "ascii":
        body = b"1 255 254 253 100 200 3 2 16711680\n4 255 254 253 101 201 6 5 65280\n"
    else:
        raw = (struct.pack("<2f", 1, 4) + struct.pack("<4f", 100, 200, 101, 201)
               + struct.pack("<2d", 3, 6) + struct.pack("<2h", 2, 5)
               + struct.pack("<2I", 0x00FF0000, 0x0000FF00))
        compressed = _lzf_literals(raw)
        body = struct.pack("<II", len(compressed), len(raw)) + compressed
    xyz, rgb = _read(monkeypatch, ".pcd", header + body)
    np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(rgb, [[255, 0, 0], [0, 255, 0]])


def test_compressed_pcd_overlapping_extended_lzf_match(monkeypatch):
    # One literal zero followed by a 255-byte match at distance one. The match
    # must copy already-produced bytes as it expands, not an uninitialised slice.
    packed = b"\x00\x00\xe0\xf6\x00"
    body = struct.pack("<II", len(packed), 256) + packed
    xyz, rgb = _read(monkeypatch, ".pcd", _pcd_header("binary_compressed", 16) + body)
    np.testing.assert_array_equal(xyz, np.zeros((16, 3)))
    np.testing.assert_array_equal(rgb, np.zeros((16, 3), dtype=np.uint8))


@pytest.mark.parametrize("packed,expected,error", [
    (b"\x01A", 2, "literal"), (b"\x20\x00", 3, "backreference"),
    (b"\xe0", 3, "extended"), (b"\x20", 3, "backreference"),
    (b"\x00A", 2, "output size"), (b"\x00A", 0, "literal"),
])
def test_lzf_corruption_rejected(packed, expected, error):
    with pytest.raises(ValueError, match=error):
        points._lzf_decompress(packed, expected)


def test_pcd_wrong_compressed_size_rejected_before_decode(monkeypatch):
    header = _pcd_header("binary_compressed", 2)
    with pytest.raises(ValueError, match="schema"):
        _read(monkeypatch, ".pcd", header + struct.pack("<II", 1, 999) + b"x")
    with pytest.raises(ValueError, match="truncated compressed"):
        _read(monkeypatch, ".pcd", header + struct.pack("<II", 100, 32) + b"x")


@pytest.mark.parametrize("mode,endian", [("ascii", "<"), ("binary_little_endian", "<"),
                                        ("binary_big_endian", ">")])
def test_ply_storage_modes_named_channels_and_preceding_lists(monkeypatch, mode, endian):
    header = ("ply\nformat {} 1.0\ncomment test\nelement face 1\n"
              "property list uchar int vertex_indices\nelement vertex 3\n"
              "property uchar blue\nproperty double z\nproperty short y\n"
              "property float x\nproperty uchar red\nproperty uchar green\n"
              "element face_after 1\nproperty list uchar int vertex_indices\nend_header\n").format(mode).encode()
    rows = [(30, 3, -2, 1.25, 10, 20), (60, 6, 5, 4, 40, 50), (90, 9, 8, np.nan, 70, 80)]
    if mode == "ascii":
        body = b"3 0 1 2\n" + ("\n".join(" ".join(map(str, row)) for row in rows) + "\n").encode()
    else:
        body = struct.pack(endian + "B3i", 3, 0, 1, 2)
        body += b"".join(struct.pack(endian + "BdhfBB", *row) for row in rows)
    xyz, rgb = _read(monkeypatch, ".ply", header + body)
    np.testing.assert_array_equal(xyz, [[1.25, -2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(rgb, [[10, 20, 30], [40, 50, 60]])


@pytest.mark.parametrize("mode", ["ascii", "binary_little_endian"])
def test_ply_vertex_lists_and_reflectivity_not_rgb(monkeypatch, mode):
    header = ("ply\nformat {} 1.0\nelement vertex 2\nproperty float x\n"
              "property list uchar float descriptor\nproperty float y\n"
              "property float z\nproperty uchar reflectivity\nend_header\n").format(mode).encode()
    body = b"1 2 100 200 2 3 255\n4 0 5 6 80\n" if mode == "ascii" else (
        struct.pack("<fB4fB", 1, 2, 100, 200, 2, 3, 255)
        + struct.pack("<fB2fB", 4, 0, 5, 6, 80))
    xyz, rgb = _read(monkeypatch, ".ply", header + body)
    np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
    assert rgb is None


def test_file_limits_truncation_and_unknown_format(monkeypatch):
    with pytest.raises(ValueError, match="max_points"):
        _read(monkeypatch, ".pcd", _pcd_header("binary", 1000), max_points=10)
    with pytest.raises(ValueError, match="max_bytes"):
        _read(monkeypatch, ".pcd", _pcd_header("binary", 1), max_bytes=10)
    with pytest.raises(ValueError, match="truncated"):
        _read(monkeypatch, ".pcd", _pcd_header("binary", 1) + b"x")
    with pytest.raises(ValueError, match="extension"):
        points.read_point_cloud("map.xyz")
    with pytest.raises(ValueError, match="POINTS"):
        header = _pcd_header("binary", 1).replace(b"POINTS 1", b"POINTS 2")
        _read(monkeypatch, ".pcd", header)


def test_empty_file_clouds_preserve_colour_shape(monkeypatch):
    xyz, rgb = _read(monkeypatch, ".pcd", _pcd_header("binary", 0))
    assert xyz.shape == rgb.shape == (0, 3)
    payload = b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    xyz, rgb = _read(monkeypatch, ".ply", payload)
    assert xyz.shape == (0, 3) and rgb is None


def test_voxels_sparse_extreme_range_and_observed_representatives(monkeypatch):
    monkeypatch.setattr(points, "_CHUNK_POINTS", 2)
    xyz = np.array([[0.1, 0.1, 1], [0.2, 0.2, 1], [-0.1, 0, 1],
                    [1e12, -1e12, 1], [np.nan, 1, 1]])
    rgb = np.arange(15, dtype=np.uint8).reshape(5, 3)
    sampled, colours = points.voxel_downsample(xyz, rgb, 1, 10)
    np.testing.assert_array_equal(sampled, xyz[[0, 2, 3]])
    np.testing.assert_array_equal(colours, rgb[[0, 2, 3]])


def test_voxel_cap_is_global_deterministic_and_chunk_independent(monkeypatch):
    xyz = np.column_stack((np.arange(100), np.zeros(100), np.ones(100)))
    rgb = np.tile(np.arange(100, dtype=np.uint8)[:, None], (1, 3))
    expected = sorted(range(100), key=lambda i: (hash((i, 0, 1)) & ((1 << 64) - 1), (i, 0, 1)))[:7]
    expected.sort()
    for chunk in (1, 9, 100):
        monkeypatch.setattr(points, "_CHUNK_POINTS", chunk)
        sampled, colours = points.voxel_downsample(xyz, rgb, 1, 7)
        np.testing.assert_array_equal(sampled, xyz[expected])
        np.testing.assert_array_equal(colours, rgb[expected])
    assert max(expected) >= 50


@pytest.mark.parametrize("mode", ["ascii", "binary", "binary_compressed"])
def test_stream_reader_downsamples_without_full_output(monkeypatch, mode):
    xyz = np.array([[0.1, 0, 1], [0.2, 0, 1], [1.1, 0, 1], [2.1, 0, 1], [3.1, 0, 1]], dtype="<f4")
    colours = np.arange(5, dtype="<u4") * 0x010101
    header = _pcd_header(mode, len(xyz))
    if mode == "ascii":
        body = ("\n".join("{} {} {} {}".format(*row, int(rgb)) for row, rgb in zip(xyz, colours)) + "\n").encode()
    elif mode == "binary":
        body = b"".join(struct.pack("<fffI", *row, int(rgb)) for row, rgb in zip(xyz, colours))
    else:
        raw = b"".join(xyz[:, i].tobytes() for i in range(3)) + colours.tobytes()
        compressed = _lzf_literals(raw)
        body = struct.pack("<II", len(compressed), len(raw)) + compressed

    class ForbidFullOutput:
        def __init__(self, *args):
            raise AssertionError("streaming mode allocated full declared point output")

    monkeypatch.setattr(points, "_Output", ForbidFullOutput)
    monkeypatch.setattr(points, "_CHUNK_POINTS", 2)
    sampled, rgb = _read(monkeypatch, ".pcd", header + body, voxel_size=1, max_points=2)
    expected_xyz, expected_rgb = points.voxel_downsample(xyz, np.tile(np.arange(5)[:, None], (1, 3)), 1, 2)
    np.testing.assert_allclose(sampled, expected_xyz)
    np.testing.assert_array_equal(rgb, expected_rgb)


def test_streaming_ply_and_voxel_memory_budget(monkeypatch):
    payload = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0.1 0 1\n0.2 0 1\n2 0 1\n"
    xyz, rgb = _read(monkeypatch, ".ply", payload, voxel_size=1, max_points=2)
    np.testing.assert_allclose(xyz, [[0.1, 0, 1], [2, 0, 1]])
    assert rgb is None
    with pytest.raises(ValueError, match="sparse voxel memory budget"):
        _read(monkeypatch, ".ply", payload, voxel_size=1, max_points=2, max_bytes=1024)


def test_project_depth_occlusion_negative_invalid_and_sparse_mask(monkeypatch):
    monkeypatch.setattr(points, "_CHUNK_POINTS", 2)
    xyz = np.array([[0, 0, 4], [0, 0, 2], [0, 0, 3], [2, 0, 2],
                    [0, 0, -1], [0, 0, 0], [np.nan, 0, 2], [0, np.inf, 2],
                    [100, 0, 1], [-100, 0, 1]])
    K = np.array([[2, 0, 2], [0, 2, 1], [0, 0, 1]])
    depth, valid = points.project_depth(xyz, np.eye(4), K, 5, 3)
    expected = np.zeros((3, 5), dtype=np.float32)
    expected[1, 2] = 2
    expected[1, 4] = 2
    np.testing.assert_array_equal(depth, expected)
    np.testing.assert_array_equal(valid, expected > 0)
    assert depth.dtype == np.float32 and valid.dtype == np.bool_


def test_project_depth_uses_c2w_with_rotation_and_translation():
    pose = np.array([[0, -1, 0, 10], [1, 0, 0, -4], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=float)
    camera = np.array([[0, 0, 2], [1, 0, 4], [0, 1, 2]])
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    K = np.array([[4, 0, 2], [0, 4, 1], [0, 0, 1]])
    depth, mask = points.project_depth(world, pose, K, 6, 5)
    assert depth[1, 2] == 2 and depth[1, 3] == 4 and depth[3, 2] == 2
    assert int(mask.sum()) == 3


def test_project_depth_pixel_boundaries_and_skew():
    K = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    # Negative subpixels/outside rounded centres must not be clamped to borders.
    xyz = np.array([[-0.1, 0, 1], [2.6, 0, 1], [0, 1, 1], [0.49, 0, 1], [0.51, 0, 1]])
    depth, mask = points.project_depth(xyz, np.eye(4), K, 3, 3)
    assert mask[0, 0] and mask[0, 1] and mask[1, 1]
    assert mask.sum() == 3 and not mask[0, 2]
    np.testing.assert_array_equal(depth[~mask], 0)


def test_project_depth_extreme_depth_float32_and_empty_cloud():
    xyz = np.array([[0, 0, 1e100], [0, 0, 1e-100]])
    depth, valid = points.project_depth(xyz, np.eye(4), np.eye(3), 2, 2)
    assert not valid.any() and not depth.any()
    empty, valid = points.project_depth(np.empty((0, 3)), np.eye(4), np.eye(3), 2, 2)
    assert not valid.any() and not empty.any()


@pytest.mark.parametrize("kwargs,error", [
    ({"width": 0}, "width"), ({"height": 2.5}, "height"),
    ({"T_world_camera": np.diag([2, 1, 1, 1])}, "orthonormal"),
    ({"T_world_camera": np.eye(3)}, "4x4"),
    ({"K": np.diag([-1, 1, 1])}, "intrinsic"),
    ({"xyz": np.zeros((2, 4))}, "shape"),
])
def test_projection_rejects_invalid_contract(kwargs, error):
    args = {"xyz": np.zeros((1, 3)), "T_world_camera": np.eye(4), "K": np.eye(3), "width": 3, "height": 3}
    args.update(kwargs)
    with pytest.raises(ValueError, match=error):
        points.project_depth(**args)

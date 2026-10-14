"""Bounded point-cloud decoding and sparse OpenCV depth projection.

XYZ is returned as float64 and RGB, when explicitly present, as uint8 RGB.
Non-finite coordinates are removed with their associated metadata. Reflectivity
and intensity are never interpreted as colour. No reader infers a world frame,
applies a PCD VIEWPOINT, deskews a scan, or establishes ground truth. The caller
must declare/gate a map's world frame and supply trusted, aligned geometry.

File readers enforce declared point and byte limits before allocating outputs,
stream ASCII/binary records in chunks, and bound LZF input/output sizes. These
limits raise ValueError rather than silently sampling a map. Compressed PCD
requires a full decompressed buffer within the configured limit; use offline
tiling for maps exceeding it. Only NumPy and the Python standard library are
required. Public functions accept ordinary ROS objects or mapping equivalents.
"""

import heapq
import operator
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np


__all__ = ["decode_pointcloud2", "decode_livox", "read_point_cloud",
           "voxel_downsample", "project_depth"]

_MAX_POINTS = 5_000_000
_MAX_BYTES = 512 * 1024 * 1024
_CHUNK_POINTS = 65_536
_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_LINE = 1024 * 1024
_ROS_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4",
              6: "u4", 7: "f4", 8: "f8"}
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "int64": "i8", "uint64": "u8",
}


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _integer(value, name, minimum=0):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError("{} must be an integer".format(name)) from exc
    if result < minimum:
        raise ValueError("{} must be >= {}".format(name, minimum))
    return result


def _limits(count, raw_bytes, max_points, max_bytes):
    max_points = _integer(max_points, "max_points", 1)
    max_bytes = _integer(max_bytes, "max_bytes", 1)
    if count > max_points:
        raise ValueError("point count {} exceeds max_points {}; tile/downsample the map first"
                         .format(count, max_points))
    if raw_bytes > max_bytes or count * 27 > max_bytes:
        raise ValueError("point-cloud bytes exceed max_bytes {}; tile/downsample the map first"
                         .format(max_bytes))


def _require_xyz(columns):
    for name in ("x", "y", "z"):
        if name not in columns or columns[name].ndim != 1:
            raise ValueError("point cloud requires scalar x, y, z fields")


def _channel_rgb(columns):
    channels = np.column_stack(columns)
    if not (np.isfinite(channels).all() and (channels >= 0).all()
            and (channels <= 255).all() and (channels == np.floor(channels)).all()):
        raise ValueError("explicit RGB channels must contain integers in [0, 255]")
    return channels.astype(np.uint8)


def _rgb(columns):
    for name in ("rgb", "rgba"):
        if name in columns:
            packed = columns[name]
            if packed.ndim != 1 or packed.dtype.itemsize != 4 or packed.dtype.kind not in "fu":
                raise ValueError("packed RGB/RGBA must be scalar FLOAT32 or UINT32")
            # Reinterpret float bits, including NaN bit patterns in RGBA, with
            # the field's byte order. Numeric float-to-int conversion is wrong.
            bits = packed.view(np.dtype("u4").newbyteorder(packed.dtype.byteorder))
            return np.column_stack(((bits >> 16) & 255, (bits >> 8) & 255,
                                    bits & 255)).astype(np.uint8)
    for names in (("red", "green", "blue"), ("r", "g", "b")):
        if any(name in columns for name in names):
            if not all(name in columns and columns[name].ndim == 1 for name in names):
                raise ValueError("incomplete or non-scalar RGB channels")
            return _channel_rgb([columns[name] for name in names])
    return None


def _has_rgb(names):
    return any(name in names for name in ("rgb", "rgba", "red", "green", "blue", "r", "g", "b"))


class _VoxelOutput:
    """Bounded deterministic selection shared by file streaming and arrays."""

    def __init__(self, voxel_size, max_points, has_rgb, bounds=None):
        self.size = float(voxel_size)
        if not np.isfinite(self.size) or self.size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        self.limit = _integer(max_points, "max_points", 1)
        self.has_rgb = has_rgb
        self.bounds = bounds
        self.kept, self.heap, self.seen = {}, [], 0

    def add_xyz(self, xyz, rgb):
        valid = np.isfinite(xyz).all(axis=1)
        if self.bounds is not None:
            valid &= ((xyz >= self.bounds[0]) & (xyz <= self.bounds[1])).all(axis=1)
        good = np.flatnonzero(valid)
        with np.errstate(over="ignore", invalid="ignore"):
            keys = np.floor(xyz[good] / self.size)
        if not np.isfinite(keys).all():
            raise ValueError("voxel coordinates overflow; increase voxel_size")
        if rgb is not None:
            rgb = _channel_rgb([rgb[good, channel] for channel in range(3)])
        for local, row in enumerate(good):
            key = tuple(int(value) for value in keys[local])
            if key in self.kept:
                continue
            # Integer tuple hashes are independent of PYTHONHASHSEED. Negating
            # all components makes heapq retain the largest rank at its root.
            rank = (hash(key) & ((1 << 64) - 1),) + key
            entry = tuple(-value for value in rank)
            if len(self.kept) == self.limit:
                if entry <= self.heap[0]:
                    continue
                worst = heapq.heapreplace(self.heap, entry)
                del self.kept[tuple(-value for value in worst[1:])]
            else:
                heapq.heappush(self.heap, entry)
            self.kept[key] = (self.seen + int(row), xyz[row].copy(),
                              None if rgb is None else rgb[local].copy())
        self.seen += len(xyz)

    def add(self, columns):
        _require_xyz(columns)
        xyz = np.column_stack([columns[name] for name in ("x", "y", "z")]).astype(np.float64)
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        rgb = _rgb({name: values[valid] for name, values in columns.items()})
        self.add_xyz(xyz, rgb)

    def finish(self):
        selected = sorted(self.kept.values(), key=lambda item: item[0])
        xyz = np.asarray([item[1] for item in selected], dtype=np.float64).reshape(-1, 3)
        rgb = None if not self.has_rgb else np.asarray(
            [item[2] for item in selected], dtype=np.uint8).reshape(-1, 3)
        return xyz, rgb


def _file_output(count, raw_bytes, names, max_points, max_bytes, voxel_size, bounds=None):
    if voxel_size is None:
        _limits(count, raw_bytes, max_points, max_bytes)
        return _Output(count, _has_rgb(names), bounds=bounds)
    _limits(0, raw_bytes, max_points, max_bytes)
    # A Python sparse key/dictionary/heap uses more memory than packed XYZ.
    # Reserve a conservative amount per retained voxel before accepting limits.
    if max_points * 768 > max_bytes:
        raise ValueError("streaming max_points exceeds sparse voxel memory budget; "
                         "reduce max_points or increase max_bytes")
    return _VoxelOutput(voxel_size, max_points, _has_rgb(names), bounds=bounds)


class _Output:
    """Preallocate bounded outputs; compact valid records while consuming chunks."""

    def __init__(self, count, has_rgb, bounds=None):
        self.xyz = np.empty((count, 3), dtype=np.float64)
        self.rgb = np.empty((count, 3), dtype=np.uint8) if has_rgb else None
        self.used = 0
        self.bounds = bounds

    def add(self, columns):
        _require_xyz(columns)
        xyz = np.column_stack([columns[name] for name in ("x", "y", "z")]).astype(np.float64)
        valid = np.isfinite(xyz).all(axis=1)
        if self.bounds is not None:
            valid &= ((xyz >= self.bounds[0]) & (xyz <= self.bounds[1])).all(axis=1)
        colours = _rgb({name: values[valid] for name, values in columns.items()})
        end = self.used + int(valid.sum())
        self.xyz[self.used:end] = xyz[valid]
        if self.rgb is not None:
            self.rgb[self.used:end] = colours
        self.used = end

    def finish(self):
        return self.xyz[:self.used], None if self.rgb is None else self.rgb[:self.used]


def decode_pointcloud2(msg, *, max_points=_MAX_POINTS, max_bytes=_MAX_BYTES):
    """Decode a ROS PointCloud2 using field offsets, datatypes and row strides.

    Returns (N, 3) finite XYZ and optional explicit RGB in input order. Supports
    all eight ROS PointField primitive types, arbitrary field ordering, point
    padding, row padding, and both byte orders. XYZ/RGB fields must be scalar.
    """
    width = _integer(_get(msg, "width"), "width")
    height = _integer(_get(msg, "height"), "height")
    step = _integer(_get(msg, "point_step"), "point_step", 1)
    row_step = _integer(_get(msg, "row_step"), "row_step")
    if row_step < width * step:
        raise ValueError("row_step is smaller than width * point_step")
    count = width * height
    _limits(count, row_step * height, max_points, max_bytes)
    endian = ">" if _get(msg, "is_bigendian", False) else "<"
    names, formats, offsets = [], [], []
    for field in _get(msg, "fields", []):
        name = str(_get(field, "name"))
        offset = _integer(_get(field, "offset"), "field offset")
        size = _integer(_get(field, "count", 1), "field count", 1)
        datatype = _get(field, "datatype")
        if datatype not in _ROS_TYPES:
            raise ValueError("unsupported PointField datatype {!r}".format(datatype))
        dtype = np.dtype(endian + _ROS_TYPES[datatype])
        if offset + size * dtype.itemsize > step:
            raise ValueError("field {!r} extends beyond point_step".format(name))
        if name in names:
            raise ValueError("duplicate PointField {!r}".format(name))
        names.append(name)
        formats.append(dtype if size == 1 else (dtype, (size,)))
        offsets.append(offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": step})
    _require_xyz({name: np.empty(0, dtype=dtype.fields[name][0]) for name in names})
    try:
        data = memoryview(_get(msg, "data")).cast("B")
    except (TypeError, ValueError) as exc:
        raise ValueError("PointCloud2 data must be a contiguous byte buffer") from exc
    if data.nbytes < row_step * height:
        raise ValueError("truncated PointCloud2 data, including row padding")
    output = _Output(count, _has_rgb(names))
    chunk_size = max(1, min(_CHUNK_POINTS, _CHUNK_BYTES // step))
    for row in range(height):
        for start in range(0, width, chunk_size):
            n = min(chunk_size, width - start)
            records = np.ndarray((n,), dtype=dtype, buffer=data,
                                 offset=row * row_step + start * step)
            output.add({name: records[name] for name in names})
    return output.finish()


def decode_livox(msg, *, max_points=_MAX_POINTS, max_bytes=_MAX_BYTES):
    """Decode Livox CustomMsg; keep finite, nonzero sensor-frame returns.

    offset_ns (uint32) and timebase_ns (Python integer) are preserved exactly,
    never converted to seconds, added together, or replaced by header time.
    reflectivity/tag/line remain raw uint8 metadata; caller applies tag-specific
    confidence gating. This does not deskew or provide ground-truth geometry.
    """
    points = _get(msg, "points", [])
    count = _integer(_get(msg, "point_num", len(points)), "point_num")
    if count != len(points):
        raise ValueError("Livox point_num does not match points length")
    _limits(count, count * 31, max_points, max_bytes)
    timebase = _integer(_get(msg, "timebase", _get(msg, "timebase_ns")), "timebase", 0)
    if timebase > (1 << 64) - 1:
        raise ValueError("Livox timebase exceeds uint64")
    xyz = np.empty((count, 3), dtype=np.float64)
    offset = np.empty(count, dtype=np.uint32)
    reflectivity = np.empty(count, dtype=np.uint8)
    tag = np.empty(count, dtype=np.uint8)
    line = np.empty(count, dtype=np.uint8)
    used = 0
    for point in points:
        coord = np.asarray([_get(point, name) for name in ("x", "y", "z")], dtype=np.float64)
        if not np.isfinite(coord).all() or not np.any(coord != 0):
            continue
        values = []
        for name, upper in (("offset_time", (1 << 32) - 1), ("reflectivity", 255),
                            ("tag", 255), ("line", 255)):
            value = _integer(_get(point, name), name)
            if value > upper:
                raise ValueError("Livox {} exceeds its declared integer range".format(name))
            values.append(value)
        xyz[used] = coord
        offset[used], reflectivity[used], tag[used], line[used] = values
        used += 1
    return {"xyz": xyz[:used], "reflectivity": reflectivity[:used],
            "offset_ns": offset[:used], "tag": tag[:used], "line": line[:used],
            "timebase_ns": timebase, "frame_id": _get(_get(msg, "header", {}), "frame_id", ""),
            "deskewed": False, "is_ground_truth": False}


def _line(fp):
    raw = fp.readline(_MAX_LINE + 1)
    if len(raw) > _MAX_LINE:
        raise ValueError("point-cloud line exceeds 1 MiB")
    if not raw:
        raise ValueError("truncated point-cloud header or ASCII records")
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("point-cloud text must be ASCII") from exc


def _header_size(fp):
    if fp.tell() > _MAX_LINE:
        raise ValueError("point-cloud header exceeds 1 MiB")


def _read_exact(fp, size):
    data = fp.read(size)
    if len(data) != size:
        raise ValueError("truncated point-cloud binary data")
    return data


def _number(token, dtype):
    if dtype.kind in "iu":
        value = int(token, 10)
        bounds = np.iinfo(dtype)
        if not bounds.min <= value <= bounds.max:
            raise ValueError("ASCII integer outside field datatype range")
        return value
    with np.errstate(over="ignore", invalid="ignore"):
        return dtype.type(token)


def _data_line(fp):
    while True:
        text = _line(fp).split("#", 1)[0].strip()
        if text:
            return text.split()


def _read_records(fp, count, dtype, output, ascii_mode=False):
    chunk_size = max(1, min(_CHUNK_POINTS, _CHUNK_BYTES // max(1, dtype.itemsize)))
    for start in range(0, count, chunk_size):
        n = min(chunk_size, count - start)
        if ascii_mode:
            records = np.empty(n, dtype=dtype)
            columns = []
            for name in dtype.names:
                field = dtype.fields[name][0]
                base, shape = field.subdtype if field.subdtype else (field, ())
                columns.append((name, base, int(np.prod(shape)) if shape else 1))
            total = sum(column[2] for column in columns)
            for row in range(n):
                tokens = _data_line(fp)
                if len(tokens) != total:
                    raise ValueError("ASCII point record has wrong field count")
                cursor = 0
                for name, base, length in columns:
                    values = [_number(token, base) for token in tokens[cursor:cursor + length]]
                    records[name][row] = values if dtype.fields[name][0].subdtype else values[0]
                    cursor += length
        else:
            records = np.frombuffer(_read_exact(fp, n * dtype.itemsize), dtype=dtype, count=n)
        output.add({name: records[name] for name in dtype.names})


def _lzf_decompress(data, expected):
    """LZF literals/backreferences, including overlapping matches, without deps."""
    output = bytearray(expected)
    src = dst = 0
    while src < len(data):
        control = data[src]
        src += 1
        if control < 32:
            length = control + 1
            if src + length > len(data) or dst + length > expected:
                raise ValueError("invalid LZF literal length")
            output[dst:dst + length] = data[src:src + length]
            src += length
        else:
            length = control >> 5
            distance = (control & 31) << 8
            if length == 7:
                if src >= len(data):
                    raise ValueError("truncated LZF extended match")
                length += data[src]
                src += 1
            if src >= len(data):
                raise ValueError("truncated LZF backreference")
            distance += data[src] + 1
            src += 1
            length += 2
            reference = dst - distance
            if reference < 0 or dst + length > expected:
                raise ValueError("invalid LZF backreference or output size")
            if length <= distance:
                output[dst:dst + length] = output[reference:reference + length]
            else:
                seed = output[reference:dst]
                output[dst:dst + length] = (seed * ((length + distance - 1) // distance))[:length]
        dst += length
    if dst != expected:
        raise ValueError("LZF output size disagrees with PCD uncompressed size")
    return output


def _read_pcd(fp, file_bytes, max_points, max_bytes, voxel_size, bounds=None):
    header = {}
    while True:
        tokens = _line(fp).split("#", 1)[0].split()
        _header_size(fp)
        if not tokens:
            continue
        name = tokens[0].upper()
        if name in header:
            raise ValueError("duplicate PCD header entry {}".format(name))
        header[name] = tokens[1:]
        if name == "DATA":
            break
    if "ENDIAN" in header or "ENDIANNESS" in header:
        raise ValueError("PCD endianness extensions are unsupported; binary PCD must be little endian")
    try:
        names = header.get("FIELDS", header.get("FIELD"))
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(names))]
        mode, = header["DATA"]
        height = int(header.get("HEIGHT", ["1"])[0])
        width = int(header.get("WIDTH", header.get("POINTS"))[0])
        count = int(header.get("POINTS", [str(width * height)])[0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("incomplete or malformed PCD header") from exc
    if not names or not len(names) == len(sizes) == len(types) == len(counts):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths disagree")
    if width < 0 or height < 1 or count < 0 or width * height != count:
        raise ValueError("PCD WIDTH * HEIGHT must equal POINTS")
    internal, formats, offset, offsets = [], [], 0, []
    for index, (name, size, kind, length) in enumerate(zip(names, sizes, types, counts)):
        if length < 1 or (kind not in ("F", "I", "U") or size not in (1, 2, 4, 8)
                          or (kind == "F" and size not in (4, 8))):
            raise ValueError("unsupported PCD field datatype or COUNT")
        field_name = "__padding_{}".format(index) if name == "_" else name
        if field_name in internal:
            raise ValueError("duplicate PCD field {!r}".format(name))
        base = np.dtype("<{}{}".format(kind.lower(), size))
        internal.append(field_name)
        formats.append(base if length == 1 else (base, (length,)))
        offsets.append(offset)
        offset += size * length
    compressed_size = sum(size * length for name, size, length in zip(names, sizes, counts) if name != "_")
    _limits(0, count * offset, max_points, max_bytes)
    if offset > max_bytes:
        raise ValueError("PCD point record exceeds max_bytes")
    dtype = np.dtype({"names": internal, "formats": formats, "offsets": offsets, "itemsize": offset})
    _require_xyz({name: np.empty(0, dtype=dtype.fields[name][0]) for name in internal})
    output = _file_output(count, count * offset, internal, max_points, max_bytes, voxel_size, bounds)
    if mode in ("ascii", "binary"):
        if mode == "binary" and file_bytes - fp.tell() < count * offset:
            raise ValueError("truncated PCD binary data")
        _read_records(fp, count, dtype, output, ascii_mode=mode == "ascii")
    elif mode == "binary_compressed":
        stored, expanded = struct.unpack("<II", _read_exact(fp, 8))
        if expanded != count * compressed_size:
            raise ValueError("PCD uncompressed size disagrees with field schema")
        if stored > max_bytes or expanded > max_bytes:
            raise ValueError("compressed PCD exceeds max_bytes")
        if stored > file_bytes - fp.tell():
            raise ValueError("truncated compressed PCD data")
        raw = _lzf_decompress(_read_exact(fp, stored), expanded)
        columns, cursor = {}, 0
        for name, original, size, length in zip(internal, names, sizes, counts):
            if original == "_":
                continue
            field = dtype.fields[name][0]
            base = field.subdtype[0] if field.subdtype else field
            values = np.frombuffer(raw, dtype=base, count=count * length, offset=cursor)
            columns[name] = values.reshape(count, length) if length != 1 else values
            cursor += count * size * length
        for start in range(0, count, _CHUNK_POINTS):
            output.add({name: values[start:start + _CHUNK_POINTS] for name, values in columns.items()})
    else:
        raise ValueError("unsupported PCD DATA {!r}".format(mode))
    return output.finish()


def _ply_type(name, endian):
    if name not in _PLY_TYPES:
        raise ValueError("unsupported PLY property type {!r}".format(name))
    return np.dtype(endian + _PLY_TYPES[name])


def _ply_variable_record(fp, properties, ascii_mode, file_bytes):
    tokens = _data_line(fp) if ascii_mode else None
    cursor, values = 0, {}
    for name, dtype, item_dtype in properties:
        if ascii_mode:
            if cursor >= len(tokens):
                raise ValueError("truncated PLY ASCII record")
            value = _number(tokens[cursor], dtype)
            cursor += 1
        else:
            value = np.frombuffer(_read_exact(fp, dtype.itemsize), dtype=dtype, count=1)[0]
        if item_dtype is None:
            values[name] = value
        else:
            length = int(value)
            if length < 0:
                raise ValueError("negative PLY list length")
            if ascii_mode:
                if cursor + length > len(tokens):
                    raise ValueError("truncated PLY ASCII list")
                cursor += length
            else:
                skip = length * item_dtype.itemsize
                if skip > file_bytes - fp.tell():
                    raise ValueError("truncated PLY binary list")
                fp.seek(skip, 1)
    if ascii_mode and cursor != len(tokens):
        raise ValueError("PLY ASCII record has extra fields")
    return values


def _read_ply(fp, file_bytes, max_points, max_bytes, voxel_size, bounds=None):
    if _line(fp) != "ply":
        raise ValueError("missing PLY magic")
    mode, endian, elements, seen = None, "<", [], set()
    while True:
        tokens = _line(fp).split()
        _header_size(fp)
        if not tokens or tokens[0] in ("comment", "obj_info"):
            continue
        if tokens[0] == "end_header":
            break
        if tokens[0] == "format" and len(tokens) == 3 and mode is None:
            mode = tokens[1]
            if tokens[2] != "1.0" or mode not in ("ascii", "binary_little_endian", "binary_big_endian"):
                raise ValueError("unsupported PLY format/version")
            endian = ">" if mode == "binary_big_endian" else "<"
        elif tokens[0] == "element" and len(tokens) == 3 and mode is not None:
            count = int(tokens[2])
            if count < 0 or tokens[1] in seen:
                raise ValueError("negative or duplicate PLY element")
            seen.add(tokens[1])
            elements.append((tokens[1], count, []))
        elif tokens[0] == "property" and elements:
            if len(tokens) == 3:
                prop = (tokens[2], _ply_type(tokens[1], endian), None)
            elif len(tokens) == 5 and tokens[1] == "list":
                prop = (tokens[4], _ply_type(tokens[2], endian), _ply_type(tokens[3], endian))
                if prop[1].kind not in "iu":
                    raise ValueError("PLY list count type must be integral")
            else:
                raise ValueError("malformed PLY property")
            if any(p[0] == prop[0] for p in elements[-1][2]):
                raise ValueError("duplicate PLY property")
            elements[-1][2].append(prop)
        else:
            raise ValueError("malformed PLY header entry")
    if mode is None or "vertex" not in seen:
        raise ValueError("PLY requires format and vertex element")
    ascii_mode = mode == "ascii"
    for name, count, properties in elements:
        scalar = [(key, dtype) for key, dtype, item in properties if item is None]
        variable = any(item is not None for key, dtype, item in properties)
        if count and not properties:
            raise ValueError("PLY element has no properties")
        dtype = np.dtype(scalar)
        if name == "vertex":
            _require_xyz({key: np.empty(0, dtype=dt) for key, dt in scalar})
            output = _file_output(count, count * dtype.itemsize, dtype.names,
                                  max_points, max_bytes, voxel_size, bounds)
            if not variable:
                if not ascii_mode and count * dtype.itemsize > file_bytes - fp.tell():
                    raise ValueError("truncated PLY binary vertices")
                _read_records(fp, count, dtype, output, ascii_mode)
            else:
                for start in range(0, count, _CHUNK_POINTS):
                    records = np.empty(min(_CHUNK_POINTS, count - start), dtype=dtype)
                    for row in range(len(records)):
                        values = _ply_variable_record(fp, properties, ascii_mode, file_bytes)
                        for key in dtype.names:
                            records[key][row] = values[key]
                    output.add({key: records[key] for key in dtype.names})
            return output.finish()  # Unrelated later faces/materials are not loaded.
        if not variable and not ascii_mode:
            skip = count * dtype.itemsize
            if skip > file_bytes - fp.tell():
                raise ValueError("truncated PLY element before vertices")
            fp.seek(skip, 1)
        else:
            # Even preceding list elements have bounded text/binary reads.
            if count > max_bytes:
                raise ValueError("PLY preceding element count exceeds max_bytes")
            for _ in range(count):
                _ply_variable_record(fp, properties, ascii_mode, file_bytes)
    raise ValueError("missing PLY vertices")


def read_point_cloud(path, *, max_points=_MAX_POINTS, max_bytes=_MAX_BYTES, voxel_size=None, bounds=None):
    """Read ASCII/binary PLY or ASCII/binary/LZF binary_compressed PCD.

    PLY supports both binary byte orders and skips non-vertex/list properties.
    Binary PCD uses the conventional little-endian PCL layout, including field
    COUNT and compressed field-major ordering (padding fields '_' omitted).
    Limits default to 5 million declared points and 512 MiB per file/raw buffer
    and output; peak memory also includes bounded chunks or compressed buffers.
    With voxel_size specified, stream into a sparse bounded voxel selector;
    max_points bounds the result instead of the declared input count. For large
    maps, use e.g. voxel_size=0.1, max_points=300_000. The sparse selector reserves
    768 bytes per output point against max_bytes for Python container overhead.
    Coordinates are returned verbatim; the caller must verify units/world frame.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".ply", ".pcd"):
        raise ValueError("unsupported point-cloud extension {}; expected .ply or .pcd".format(suffix))
    max_points = _integer(max_points, "max_points", 1)
    max_bytes = _integer(max_bytes, "max_bytes", 1)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[0] > bounds[1]).any():
            raise ValueError("bounds must be finite (min_xyz,max_xyz)")
    with open(path, "rb") as fp:
        fp.seek(0, 2)
        size = fp.tell()
        fp.seek(0)
        if size > max_bytes:
            raise ValueError("point-cloud file exceeds max_bytes {}; tile the map first".format(max_bytes))
        reader = _read_ply if suffix == ".ply" else _read_pcd
        return reader(fp, size, max_points, max_bytes, voxel_size, bounds)


def _xyz_input(xyz):
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu":
        raise ValueError("xyz must be a real numeric array of shape (N, 3)")
    return points


def voxel_downsample(xyz, rgb=None, voxel_size=0.1, max_points=1_000_000):
    """Keep one actual input point per voxel, with matching explicit RGB.

    Uses sparse integer voxel keys, chunks and a bounded dictionary/heap rather
    than allocating a dense spatial grid or sorting all input points. If there
    are more occupied voxels than max_points, keep the smallest deterministic
    hash ranks across the whole map. Each retained voxel uses its first input
    point; outputs follow input order. This preserves observed geometry instead
    of introducing centroids for depth QA. Memory is O(max_points + chunk).
    """
    points = _xyz_input(xyz)
    colours = None if rgb is None else np.asarray(rgb)
    if colours is not None and (colours.shape != points.shape or colours.dtype.kind not in "fiu"):
        raise ValueError("rgb must be a real numeric array matching xyz shape")
    output = _VoxelOutput(voxel_size, max_points, colours is not None)
    for start in range(0, len(points), _CHUNK_POINTS):
        chunk = points[start:start + _CHUNK_POINTS].astype(np.float64)
        output.add_xyz(chunk, None if colours is None else colours[start:start + _CHUNK_POINTS])
    return output.finish()


def project_depth(xyz, T_world_camera, K, width, height):
    """Project trusted world XYZ using the supplied OpenCV camera-to-world pose.

    T_world_camera is explicitly c2w: world_xyz = R @ camera_xyz + t, with
    positive camera Z forward, X right and Y down. K applies to the rectified
    image (no distortion parameters). Rasterization uses nearest pixel centres
    floor(uv + .5); out-of-image, non-finite and nonpositive depths are rejected.
    A z-buffer keeps the nearest positive camera-Z depth in metres, float32.
    Return (depth, boolean mask); unknown pixels have depth 0. No dilation,
    interpolation or hole filling. Mask validity is geometric, not a guarantee
    of sensor confidence, time alignment, deskewing, or ground-truth status.
    """
    points = _xyz_input(xyz)
    width = _integer(width, "width", 1)
    height = _integer(height, "height", 1)
    if width * height * 5 > _MAX_BYTES:
        raise ValueError("depth image exceeds 512 MiB allocation limit")
    pose = np.asarray(T_world_camera, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("T_world_camera must be a finite homogeneous 4x4 c2w transform")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
        raise ValueError("T_world_camera rotation must be proper and orthonormal")
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("K must be a finite OpenCV 3x3 intrinsic matrix with positive focal lengths")
    depth = np.full((height, width), np.inf, dtype=np.float32)
    for start in range(0, len(points), _CHUNK_POINTS):
        chunk = points[start:start + _CHUNK_POINTS].astype(np.float64)
        chunk = chunk[np.isfinite(chunk).all(axis=1)]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            camera = (chunk - pose[:3, 3]) @ rotation
            camera = camera[np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)]
            projected = camera @ intrinsics.T
            uv = projected[:, :2] / projected[:, 2:3]
            z = camera[:, 2].astype(np.float32)
        good = (np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
                # Integer-centre CV pixels own [i-.5,i+.5), matching the
                # surfel/native projection. A tiny negative roundoff at pixel
                # zero must not discard the nearest measured foreground.
                & (uv[:, 0] >= -.5) & (uv[:, 0] < width - .5)
                & (uv[:, 1] >= -.5) & (uv[:, 1] < height - .5))
        pixels = np.floor(uv[good] + 0.5).astype(np.int64)
        z = z[good]
        inside = (pixels[:, 0] < width) & (pixels[:, 1] < height)
        pixels = pixels[inside]
        np.minimum.at(depth.ravel(), pixels[:, 1] * width + pixels[:, 0], z[inside])
    valid = np.isfinite(depth)
    depth[~valid] = 0
    return depth, valid

#!/usr/bin/env python3
"""Read-only local Gaussian viewer server; Python 3.8+, standard library only."""
import sys
sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

PROJECT = Path(__file__).resolve().parents[1]
ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
MAX_MANIFEST = 4 * 1024 * 1024
LIMITS = {"max_loaded_blocks": 3, "max_splats": 1500000,
          "max_file_bytes": 512 * 1024 * 1024, "max_lidar_points": 300000,
          "max_lidar_bytes": 32 * 1024 * 1024}
SIZES = {"float": (4, "f"), "float32": (4, "f"), "double": (8, "d"),
         "float64": (8, "d"), "uchar": (1, "B"), "uint8": (1, "B"),
         "char": (1, "b"), "int8": (1, "b"), "short": (2, "h"),
         "ushort": (2, "H"), "int": (4, "i"), "uint": (4, "I")}
STATIC = {"/": "viewer/index.html", "/index.html": "viewer/index.html",
          "/app.js": "viewer/app.js", "/style.css": "viewer/style.css",
          "/gaussian_sh_restore.js": "viewer/gaussian_sh_restore.js",
          "/gaussian_direction_support.js": "viewer/gaussian_direction_support.js",
          "/vendor/spark.js": "viewer/node_modules/@sparkjsdev/spark/dist/spark.module.js",
          "/vendor/three/build/three.module.js": "viewer/node_modules/three/build/three.module.js",
          "/vendor/three/build/three.core.js": "viewer/node_modules/three/build/three.core.js",
          "/vendor/three/addons/controls/OrbitControls.js": "viewer/node_modules/three/examples/jsm/controls/OrbitControls.js",
          "/vendor/three/addons/loaders/PLYLoader.js": "viewer/node_modules/three/examples/jsm/loaders/PLYLoader.js",
          "/vendor/three/addons/postprocessing/Pass.js": "viewer/node_modules/three/examples/jsm/postprocessing/Pass.js"}


class MapError(ValueError):
    pass


def configured_limits(overrides=None):
    """Explicit per-process budgets; a manifest cannot raise them."""
    result = dict(LIMITS)
    ceilings = {"max_loaded_blocks": 28, "max_splats": 6000000,
                "max_file_bytes": 2 * 1024**3}
    for name, value in (overrides or {}).items():
        if name not in ceilings or type(value) is not int or not 1 <= value <= ceilings[name]:
            raise MapError("无效的查看器进程资源预算: " + str(name))
        result[name] = value
    return result


def reference_views(data):
    """Accept actual world camera poses; never transform the model itself."""
    if data.get('schema_version') != 1 or data.get('coordinates') != 'camera_init' or data.get('units') != 'm':
        raise MapError('区域视角坐标不匹配')
    if data.get('axes') != 'OpenGL/Nerfstudio camera +X right +Y up -Z forward':
        raise MapError('区域视角方向未声明')
    result = {}
    for bid, block in data.get('blocks', {}).items():
        if not isinstance(bid, str) or not ID.fullmatch(bid):
            raise MapError('区域视角编号无效')
        views = block.get('views', [])
        if not isinstance(views, list) or len(views) > 4:
            raise MapError('区域视角数量无效')
        for view in views:
            matrix = view.get('camera_to_world')
            if not isinstance(matrix, list) or len(matrix) != 4:
                raise MapError('区域视角姿态无效')
            if any(not isinstance(row, list) or len(row) != 4 for row in matrix):
                raise MapError('区域视角姿态无效')
            if any(type(x) not in (int, float) or not math.isfinite(x) for row in matrix for x in row):
                raise MapError('区域视角姿态含非有限数值')
            if matrix[3] != [0, 0, 0, 1]:
                raise MapError('区域视角姿态末行无效')
            r = [row[:3] for row in matrix[:3]]
            for i in range(3):
                for j in range(3):
                    if abs(sum(r[i][k]*r[j][k] for k in range(3)) - int(i == j)) > 1e-4:
                        raise MapError('区域视角旋转不是刚体')
            determinant = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                         - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                         + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
            if abs(determinant - 1) > 1e-4:
                raise MapError('区域视角旋转手性无效')
            fov = view.get('fov_y_deg')
            if type(fov) not in (int, float) or not math.isfinite(fov) or not 5 < fov < 150:
                raise MapError('区域视角视场无效')
            if not isinstance(view.get('label'), str) or len(view['label']) > 32:
                raise MapError('区域视角标签无效')
        result[bid] = views
    return result


def parts_for(path):
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or "\x00" in path:
        raise MapError("路径不合法")
    parts = PurePosixPath(path).parts
    if path.startswith("/") or any(p in ("..", ".") for p in path.split("/")):
        raise MapError("路径必须位于本工程内")
    if not parts or any(p.startswith(".") for p in parts):
        raise MapError("不允许访问隐藏文件")
    return parts


def open_local(root, path):
    """Walk with dir_fd + O_NOFOLLOW: reject symlinks, including swap races."""
    parts = parts_for(path)
    fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        result = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            if not stat.S_ISREG(os.fstat(result).st_mode):
                raise MapError("仅允许普通文件")
            return os.fdopen(result, "rb")
        except Exception:
            os.close(result)
            raise
    finally:
        os.close(fd)


def vector(value, label):
    if not isinstance(value, list) or len(value) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise MapError(label + " 必须是三个有限数值")
    return value


def no_duplicate_keys(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise MapError("清单含重复字段")
        result[k] = v
    return result


def reject_constant(value):
    raise MapError("清单含非有限数值")


def validate_manifest(data):
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise MapError("清单版本应为 1")
    if data.get("coordinates") != "camera_init" or data.get("units") != "m":
        raise MapError("清单应使用 camera_init 米制坐标")
    kind = data.get("dataset_kind", "real")
    if kind not in ("real", "synthetic"):
        raise MapError("数据类型不合法")
    up = vector(data.get("camera_up", [0, 0, 1]), "观察上方向")
    if sum(x*x for x in up) < 1e-12:
        raise MapError("观察上方向不能为零")
    data["camera_up"] = up
    data["dataset_kind"] = kind
    if not isinstance(data.get("blocks"), list) or len(data["blocks"]) > 10000:
        raise MapError("清单缺少有效区块数组")
    seen = set()
    for block in data["blocks"]:
        if not isinstance(block, dict) or not isinstance(block.get("id"), str) or not ID.fullmatch(block["id"]) or block["id"] in seen:
            raise MapError("区块编号无效或重复")
        seen.add(block["id"])
        floor = block.get("floor_id")
        if floor is not None and (not isinstance(floor, str) or not floor or len(floor) > 96):
            raise MapError("楼层编号应为空或字符串")
        block["floor_id"] = floor
        if block.get("coordinates") != "camera_init":
            raise MapError("所有区块必须使用 camera_init")
        parts_for(block.get("path"))
        if not block["path"].lower().endswith(".ply"):
            raise MapError("Gaussian 模型必须为标准 PLY")
        support = block.get('direction_support')
        if support is not None:
            if not isinstance(support, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(support.get('sha256', ''))):
                raise MapError('方向支持元数据缺少有效哈希')
            parts_for(support.get('path'))
            if not support['path'].endswith('.json'):
                raise MapError('方向支持元数据须为JSON')
        bbox = block.get("bbox_m")
        if not isinstance(bbox, dict):
            raise MapError("区块缺少米制包围盒")
        low, high = vector(bbox.get("min"), "包围盒下界"), vector(bbox.get("max"), "包围盒上界")
        if any(a > b for a, b in zip(low, high)):
            raise MapError("包围盒上下界顺序错误")
        for info in ("training", "eval"):
            if info not in block or not isinstance(block[info], dict):
                raise MapError("区块缺少 training / eval 信息对象")
        lidar = block.get("lidar")
        if lidar is not None:
            if not isinstance(lidar, dict) or lidar.get("coordinates") != "camera_init":
                raise MapError("LiDAR 辅层必须声明同一坐标系")
            parts_for(lidar.get("path"))
            if not lidar["path"].lower().endswith(".ply"):
                raise MapError("LiDAR 辅层当前接受 PLY 点云")
            if lidar.get("alignment_verified") is not True:
                raise MapError("LiDAR 辅层必须明确确认配准")
    return data


def inspect_ply(handle, gaussian=True):
    """Bounded structural preflight, not an accuracy/whole-payload certification."""
    handle.seek(0)
    raw = bytearray()
    while len(raw) < 65536:
        line = handle.readline(4096)
        if not line:
            raise MapError("PLY 文件头不完整")
        raw.extend(line)
        if line.strip() == b"end_header":
            break
    else:
        raise MapError("PLY 文件头过大")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        raise MapError("PLY 文件头编码不支持")
    if not lines or lines[0].strip() != "ply":
        raise MapError("文件不是 PLY")
    fmt, count, props, active = None, None, [], False
    for line in lines[1:]:
        tokens = line.split()
        if not tokens or tokens[0] in ("comment", "obj_info", "end_header"):
            continue
        if tokens[0] == "format":
            if len(tokens) != 3 or tokens[2] != "1.0" or fmt is not None:
                raise MapError("PLY 格式声明无效")
            fmt = tokens[1]
        elif tokens[0] == "element":
            if len(tokens) != 3 or tokens[1] != "vertex" or count is not None:
                raise MapError("仅接受标准 vertex PLY，不接受压缩或网格格式")
            try:
                count = int(tokens[2])
            except ValueError:
                raise MapError("PLY 数量无效")
            active = True
        elif tokens[0] == "property":
            if not active or len(tokens) != 3 or tokens[1] not in SIZES:
                raise MapError("PLY 属性类型不支持")
            if tokens[2] in [p[1] for p in props]:
                raise MapError("PLY 属性重复")
            props.append((tokens[1], tokens[2]))
        else:
            raise MapError("PLY 文件头声明不支持")
    if fmt not in ("binary_little_endian", "ascii") or count is None or count <= 0:
        raise MapError("需要非空 ASCII 或小端二进制 PLY")
    names = [p[1] for p in props]
    required = ["x", "y", "z"]
    if gaussian:
        required += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                     "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    if any(p not in names for p in required):
        raise MapError("缺少 Gaussian 尺度、旋转、不透明度或 SH；普通点云不能作为 Gaussian" if gaussian else "LiDAR 点云缺少 XYZ")
    if gaussian and fmt != "binary_little_endian":
        raise MapError("Gaussian 渲染器需要小端二进制 PLY，请以此格式导出训练模型")
    if gaussian and any(t not in ("float", "float32") for t, n in props if n in required or n.startswith("f_rest_")):
        raise MapError("标准 Gaussian 属性应为 float32")
    rest = [n for n in names if n.startswith("f_rest_")]
    if gaussian and (len(rest) not in (0, 9, 24, 45) or set(rest) != {"f_rest_" + str(i) for i in range(len(rest))}):
        raise MapError("SH 系数应完整连续，支持 0 到 3 阶")
    stride = sum(SIZES[t][0] for t, _ in props)
    size = os.fstat(handle.fileno()).st_size
    if fmt == "binary_little_endian":
        if size != len(raw) + count * stride:
            raise MapError("PLY 数据长度与 vertex 声明不符")
        samples = struct.unpack("<" + "".join(SIZES[t][1] for t, _ in props), handle.read(stride))
    else:
        try:
            samples = tuple(float(v) for v in handle.readline(65536).split())
        except ValueError:
            raise MapError("PLY 数据不是数值")
        if len(samples) != len(props):
            raise MapError("PLY 首条数据属性数量不符")
    if any(not math.isfinite(v) for v in samples):
        raise MapError("PLY 首条数据包含非有限数值")
    if gaussian and sum(samples[names.index("rot_" + str(i))] ** 2 for i in range(4)) < 1e-12:
        raise MapError("PLY 首条旋转无效")
    return {"vertex_count": count, "file_bytes": size, "sh_degree": {0: 0, 9: 1, 24: 2, 45: 3}.get(len(rest), 0),
            "format": fmt, "preflight": "header_and_first_vertex"}


class MapStore:
    def __init__(self, root, manifest, demo=False, limits=None):
        self.root = Path(root)
        self.manifest = manifest
        self.demo = demo
        self.limits = configured_limits(limits)

    def asset_sha256(self, path):
        with open_local(self.root, path) as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            for raw in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(raw)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise MapError('资产在读取时发生变化')
            return digest.hexdigest()

    def read(self):
        try:
            with open_local(self.root, self.manifest) as f:
                payload = f.read(MAX_MANIFEST + 1)
        except FileNotFoundError:
            if self.demo:
                raise MapError("合成自检数据尚未生成")
            return {"schema_version": 1, "coordinates": "camera_init", "units": "m",
                    "dataset_kind": "real", "camera_up": [0, 0, 1], "blocks": [],
                    "title": "车库 Gaussian 地图", "manifest_missing": True}
        if len(payload) > MAX_MANIFEST:
            raise MapError("清单文件过大")
        try:
            data = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise MapError("清单 JSON 格式不合法")
        return validate_manifest(data)

    def preflight(self, path, gaussian=True):
        with open_local(self.root, path) as f:
            size = os.fstat(f.fileno()).st_size
            if size > self.limits["max_file_bytes" if gaussian else "max_lidar_bytes"]:
                raise MapError("单块文件超出本地加载预算，请进一步分块")
            info = inspect_ply(f, gaussian)
            if info["vertex_count"] > self.limits["max_splats" if gaussian else "max_lidar_points"]:
                raise MapError("单块数量超出本地加载预算，请进一步分块")
            return info

    def public(self):
        data = copy.deepcopy(self.read())
        data["limits"] = dict(self.limits)
        views = {}
        try:
            with open_local(self.root, 'maps/garage_reference_views.json') as handle:
                raw = handle.read(MAX_MANIFEST + 1)
            if len(raw) > MAX_MANIFEST:
                raise MapError('区域视角文件过大')
            views = reference_views(json.loads(raw, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant))
        except (OSError, MapError, ValueError, TypeError, AttributeError):
            pass  # Optional navigation metadata must not block actual models.
        for b in data["blocks"]:
            b['camera_views'] = views.get(b['id'], []) if data['dataset_kind'] == 'real' else []
            if b.get('direction_support'):
                b['direction_support']['url'] = '/api/blocks/' + b['id'] + '/direction_support'
            for target, key, gaussian in ((b, "model", True), (b.get("lidar"), "lidar", False)):
                if target is None:
                    continue
                try:
                    info = self.preflight(target["path"], gaussian)
                    target["asset"] = dict(info, available=True, url="/api/blocks/" + b["id"] + "/" + key)
                except FileNotFoundError:
                    target["asset"] = {"available": False, "reason": "等待训练模型" if gaussian else "辅层文件尚未就绪"}
                except (OSError, MapError) as exc:
                    target["asset"] = {"available": False, "reason": str(exc) if isinstance(exc, MapError) else "文件不可访问或含符号链接"}
        return data

    def asset(self, block_id, layer):
        for block in self.read()["blocks"]:
            if block["id"] == block_id:
                target = block if layer == "model" else block.get("lidar")
                if target is None:
                    break
                self.preflight(target["path"], layer == "model")
                return target["path"]
        raise FileNotFoundError("asset not listed")

    def direction_asset(self, block_id, array=False):
        for block in self.read()['blocks']:
            if block['id'] != block_id or not block.get('direction_support'):
                continue
            support = block['direction_support']
            with open_local(self.root, support['path']) as stream:
                raw = stream.read(MAX_MANIFEST + 1)
            if len(raw) > MAX_MANIFEST or hashlib.sha256(raw).hexdigest() != support['sha256']:
                raise MapError('方向支持元数据哈希不匹配')
            meta = json.loads(raw, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)
            if meta.get('schema_version') != 1 or meta.get('ply', {}).get('path') != block['path']:
                raise MapError('方向支持未绑定本区块模型')
            info = self.preflight(block['path'])
            if meta['ply'].get('gaussians') != info['vertex_count']:
                raise MapError('方向支持记录数与PLY不一致')
            if meta['ply'].get('sha256') != self.asset_sha256(block['path']):
                raise MapError('方向支持PLY哈希不匹配')
            if not array:
                return support['path']
            descriptor = meta['side_flags']
            path = descriptor['path']
            parts_for(path)
            if not path.endswith('.npy') or descriptor.get('count') != info['vertex_count'] or descriptor.get('dtype') != 'uint8':
                raise MapError('方向支持数组格式不匹配')
            with open_local(self.root, path) as stream:
                if os.fstat(stream.fileno()).st_size > info['vertex_count'] + 65536:
                    raise MapError('方向支持数组过大')
                digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != descriptor['sha256']:
                raise MapError('方向支持数组哈希不匹配')
            return path
        raise FileNotFoundError('direction support not listed')


class Handler(BaseHTTPRequestHandler):
    server_version = "GarageLocal/1"

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; worker-src 'self' blob:; connect-src 'self' blob:; img-src 'self' data:; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def respond_json(self, data, status=200, head=False):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def do_HEAD(self):
        self.route(True)

    def do_GET(self):
        self.route(False)

    def route(self, head):
        # Reject DNS rebinding and remote browser subresource requests.
        if self.headers.get("Host", "").split(":")[0] not in self.server.allowed_hosts:
            self.respond_json({"error": "本地服务器拒绝此主机名"}, 403, head)
            return
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.respond_json({"error": "仅允许本地页面访问"}, 403, head)
            return
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
            if "\\" in path or "\x00" in path or any(p in ("..", ".") for p in path.split("/")):
                raise MapError("路径不合法")
            if path == "/api/manifest":
                self.respond_json(self.server.store.public(), head=head)
                return
            if path in STATIC:
                local = STATIC[path]
            else:
                support_match = re.fullmatch(r'/api/blocks/([A-Za-z0-9_-]{1,96})/direction_support(/side_flags)?', path)
                if support_match:
                    local = self.server.store.direction_asset(support_match.group(1), bool(support_match.group(2)))
                    with open_local(self.server.store.root, local) as handle:
                        self.send_file(handle, local, head)
                    return
                match = re.fullmatch(r"/api/blocks/([A-Za-z0-9_-]{1,96})/(model|lidar)", path)
                if not match:
                    raise FileNotFoundError()
                local = self.server.store.asset(*match.groups())
            with open_local(self.server.store.root, local) as handle:
                self.send_file(handle, local, head)
        except FileNotFoundError:
            self.respond_json({"error": "文件尚未就绪或未在清单中登记"}, 404, head)
        except (MapError, UnicodeError, ValueError):
            self.respond_json({"error": "清单或模型不符合查看协议，请检查工程文档"}, 422, head)
        except OSError:
            self.respond_json({"error": "文件不可访问"}, 403, head)

    def send_file(self, handle, local, head):
        size = os.fstat(handle.fileno()).st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if not match or not any(match.groups()):
                self.range_error(size, head)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
            if start > end or start >= size or size == 0:
                self.range_error(size, head)
                return
            status = 206
        self.send_response(status)
        mime = "text/javascript" if local.endswith(".js") else mimetypes.guess_type(local)[0] or "application/octet-stream"
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        if status == 206:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        self.end_headers()
        if head:
            return
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            payload = handle.read(min(256 * 1024, remaining))
            if not payload:
                break
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(payload)

    def range_error(self, size, head):
        self.send_response(416)
        self.send_header("Content-Range", "bytes */" + str(size))
        self.send_header("Content-Length", "0")
        self.end_headers()


def make_server(root=PROJECT, manifest="maps/garage_manifest.json", host="127.0.0.1", port=8765, demo=False, limits=None):
    parts_for(manifest)
    store = MapStore(root, manifest, demo, limits)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.store = store
    server.allowed_hosts = {"localhost", "127.0.0.1", host}
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Default loopback only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--manifest", default="maps/garage_manifest.json", help="Project-relative manifest path")
    parser.add_argument("--demo", action="store_true", help="Explicit synthetic fixture mode, never real map")
    parser.add_argument("--max-splats", type=int, help="Explicit total loaded Gaussian budget (up to 6000000)")
    parser.add_argument("--max-file-bytes", type=int, help="Explicit per-model file budget (up to 2 GiB)")
    parser.add_argument("--max-loaded-blocks", type=int, help="Explicit simultaneous block budget (up to 28)")
    args = parser.parse_args()
    if args.demo:
        if args.manifest != "maps/garage_manifest.json":
            parser.error("--demo cannot be combined with --manifest")
        args.manifest = "viewer/fixtures/synthetic_manifest.json"
    try:
        limits = {name: getattr(args, name) for name in ("max_splats", "max_file_bytes", "max_loaded_blocks")
                  if getattr(args, name) is not None}
        server = make_server(manifest=args.manifest, host=args.host, port=args.port, demo=args.demo, limits=limits)
    except (MapError, OSError) as exc:
        parser.error(str(exc))
    mode = "SYNTHETIC SELF-TEST ONLY" if args.demo else "real map manifest (may be pending)"
    print("Local viewer http://{}:{}/ [{}]".format(args.host, server.server_port, mode), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

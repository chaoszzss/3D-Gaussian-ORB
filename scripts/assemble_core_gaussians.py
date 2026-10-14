#!/usr/bin/env python3
"""CPU assembly of all 28 canonical cores, retaining exact trained PLY records."""
import sys

sys.dont_write_bytecode = True

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Confine caches BEFORE importing numpy, scipy, or the existing support module.
_cache = ROOT / "runtime/cache"
for _key, _sub in {
    "XDG_CACHE_HOME": "xdg", "PIP_CACHE_DIR": "pip", "UV_CACHE_DIR": "uv",
    "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torch_inductor", "TRITON_CACHE_DIR": "triton",
    "HF_HOME": "huggingface", "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
    "WANDB_DIR": "wandb", "WANDB_CONFIG_DIR": "wandb_config", "CUDA_CACHE_PATH": "cuda",
    "__GL_SHADER_DISK_CACHE_PATH": "nvidia_gl",
}.items():
    os.environ[_key] = str(_cache / _sub)
for _key in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_key] = str(ROOT / "runtime/tmp")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["__GL_SHADER_DISK_CACHE"] = "0"
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy.spatial import cKDTree

from filter_gaussian_support import build_ownership
from package_garage import (
    EXPECTED_IDS, MAX_CHUNK_BYTES, VIEWER_MAX_BYTES, PackageError, _state,
    inspect_ply, local_path, merge_bounds as _merge_bounds, ply_header, positive_int,
    quotas_for, read_json, sample_indices,
)

STATUS = "candidate_pending_native_fullscope_evaluation"
OWNERSHIP_MODES = ("legacy-camera", "none", "lidar-support")
OVERVIEW_LIMIT = 1_200_000
CORE_FILE_LIMIT = 1_500_000
MAX_CORE_SHARDS = 10_000
MAX_REFERENCE_BYTES = 256 * 1024**2
DEFAULT_REFERENCE_POINTS = 2_000_000
FOOTPRINT_CELL_LIMIT = 20_000
PROVENANCE = np.dtype([
    ("source_index", "<u4"), ("source_record_index", "<u8"),
    ("assigned_core", "<u2"), ("nearest_train_group", "<i8"),
    ("measured_support_checked", "?"), ("measured_support_distance_m", "<f8"),
])
FALSE_FLAGS = {
    "floor_verified": False, "calibration_verified": False,
    "independent_geometry_acceptance": False, "high_fidelity_accepted": False,
    "native_fullscope_evaluated": False, "unobserved_building_complete": False,
}


class AssemblyError(PackageError):
    """Failure publishes no candidate or coverage-complete marker."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(raw)
    return digest.hexdigest()


class InputGuard:
    def __init__(self, project):
        self.project, self.files = project, {}

    def remember(self, path, state, digest):
        if _state(path) != state:
            raise AssemblyError("Source changed during read: " + str(path))
        value = {"state": state, "sha256": digest}
        if path in self.files and self.files[path] != value:
            raise AssemblyError("Source changed between passes: " + str(path))
        self.files[path] = value

    def json(self, path):
        state, hashes = _state(path), {}
        data = read_json(path, hashes)
        self.remember(path, state, hashes[path])
        return data

    def snapshot(self, path):
        state = _state(path)
        self.remember(path, state, sha256(path))

    def check(self, ply_paths=()):
        # PLY hashes were compared during both complete streaming passes.
        for path, value in self.files.items():
            local_path(self.project, path)
            if _state(path) != value["state"]:
                raise AssemblyError("Source changed before publication: " + str(path))
            if path not in ply_paths and sha256(path) != value["sha256"]:
                raise AssemblyError("Source SHA changed before publication: " + str(path))
            if _state(path) != value["state"]:
                raise AssemblyError("Source changed while verifying: " + str(path))

    def report(self):
        return {p.relative_to(self.project).as_posix(): {
            "sha256": value["sha256"],
            "stat": dict(zip(("device", "inode", "bytes", "mtime_ns", "ctime_ns"), value["state"])),
        } for p, value in sorted(self.files.items())}


def default_source_plan():
    return {"schema_version": 1, "coordinates": "camera_init", "units": "m", "sources": [
        {"id": bid, "path": f"runs/garage_surface_v1/{bid}/eval_dev/gaussians.ply",
         "core_ids": [bid], "measured_reference": f"data/garage_surface_v1/{bid}/lidar_reference.npz"}
        for bid in EXPECTED_IDS]}


def initial_source_plan():
    """Proposed full-scope baseline: originals except the trained 014..016 pool."""
    plan = default_source_plan()
    cores = ["block_014", "block_015", "block_016"]
    plan["sources"] = [source for source in plan["sources"] if source["id"] not in cores]
    plan["sources"].append({"id": "joint_overlap_v1", "path": "runs/joint_overlap_v1/eval_dev/gaussians.ply",
                            "core_ids": cores, "measured_reference": "data/joint_overlap_v1/lidar_reference.npz"})
    return plan


def resource_preflight(destination, *, disk_bytes, memory_bytes):
    """Bounded CPU jobs fail before allocation/publication; respect cgroup RAM."""
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    if available is None:
        raise AssemblyError("Cannot establish available memory")
    # Resolve this process's cgroup and every parent, not just the host limit.
    cgroup = Path("/proc/self/cgroup")
    if cgroup.exists():
        for line in cgroup.read_text().splitlines():
            if line.startswith("0::"):
                base = Path("/sys/fs/cgroup")
                current = base / line[3:].lstrip("/")
                while current == base or base in current.parents:
                    limit, usage = current / "memory.max", current / "memory.current"
                    if limit.exists() and usage.exists() and limit.read_text().strip() != "max":
                        available = min(available, max(0, int(limit.read_text()) - int(usage.read_text())))
                    if current == base:
                        break
                    current = current.parent
    parent = Path(destination).parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    report = {"cpu_only": True, "disk_free_bytes": free, "memory_available_bytes": available,
              "estimated_disk_bytes": int(disk_bytes), "estimated_peak_memory_bytes": int(memory_bytes),
              "memory_safety_fraction": .8, "disk_reserve_bytes": 512 * 1024**2}
    if memory_bytes > .8 * available:
        raise AssemblyError("CPU memory preflight failed: " + json.dumps(report))
    if disk_bytes + 512 * 1024**2 > free:
        raise AssemblyError("Disk preflight failed: " + json.dumps(report))
    return report


def select_cores(sources, core_ids):
    selected = list(EXPECTED_IDS) if core_ids is None else sorted(core_ids)
    if (not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(EXPECTED_IDS)):
        raise AssemblyError("core_ids must be distinct canonical IDs")
    result = []
    for source in sources:
        overlap = set(source["core_ids"]) & set(selected)
        if overlap and overlap != set(source["core_ids"]):
            raise AssemblyError("Subset cannot split a joint source; select all its core_ids")
        if overlap:
            result.append(source)
    return result, selected


def parse_plan(project, data):
    if (type(data.get("schema_version", 1)) is not int or data.get("schema_version", 1) != 1
            or data.get("coordinates", "camera_init") != "camera_init"
            or data.get("units", "m") != "m"):
        raise AssemblyError("Source plan must use schema 1 and shared camera_init/m")
    if set(data) != {"schema_version", "coordinates", "units", "sources"}:
        raise AssemblyError("Source plan requires exactly schema_version, coordinates, units, sources")
    entries = data.get("sources")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 28:
        raise AssemblyError("Source plan requires 1..28 sources covering all 28 cores")
    sources, seen_ids, seen_files, covered = [], set(), set(), set()
    for item in entries:
        if not isinstance(item, dict) or set(item) - {"id", "path", "core_ids", "measured_reference"}:
            raise AssemblyError("Invalid source-plan entry/unknown field")
        sid = item.get("id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", sid) or sid in seen_ids:
            raise AssemblyError("Invalid or duplicate source ID")
        cores = item.get("core_ids")
        if (not isinstance(cores, list) or not cores or any(not isinstance(c, str) or c not in EXPECTED_IDS for c in cores)
                or len(set(cores)) != len(cores)):
            raise AssemblyError("Source requires distinct canonical core_ids")
        if covered.intersection(cores):
            raise AssemblyError("Duplicate core coverage: " + ", ".join(sorted(covered.intersection(cores))))
        path = local_path(project, item.get("path", ""))
        state = _state(path)
        if path.suffix.lower() != ".ply" or state[:2] in seen_files:
            raise AssemblyError("Source must be a distinct standard PLY file")
        reference = item.get("measured_reference")
        if reference is not None:
            reference = local_path(project, reference)
            if reference.suffix.lower() != ".npz":
                raise AssemblyError("Measured reference must be an NPZ")
        sources.append({"id": sid, "path": path, "state": state, "core_ids": sorted(cores),
                        "measured_reference": reference})
        seen_ids.add(sid); seen_files.add(state[:2]); covered.update(cores)
    if covered != set(EXPECTED_IDS):
        raise AssemblyError("Missing core coverage: " + ", ".join(sorted(set(EXPECTED_IDS) - covered)))
    return sorted(sources, key=lambda source: source["id"])


def integer_group(value):
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        value = int(value)
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise AssemblyError("Training group_id must be a stable int64 integer, never a float")
    return value


def canonical_ownership(project, guard):
    metadata_by_id, group_splits = {}, {}
    for bid in EXPECTED_IDS:
        meta = guard.json(local_path(project, f"data/garage_surface_v1/{bid}/transforms.json"))
        if meta.get("orientation_override", "none") != "none":
            raise AssemblyError("Ownership requires unchanged world poses")
        frames = meta.get("frames")
        if not isinstance(frames, list):
            raise AssemblyError("Missing source frames: " + bid)
        membership = {}
        for key in ("train_filenames", "val_filenames", "test_filenames"):
            names = meta.get(key, [])
            if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
                    or len(set(names)) != len(names)):
                raise AssemblyError("Invalid split filenames: " + bid)
            for name in names:
                if name in membership:
                    raise AssemblyError("Training/held-out split leakage: " + bid)
                membership[name] = key
        seen, train_frames = set(), []
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("file_path"), str):
                raise AssemblyError("Invalid source frame: " + bid)
            name = frame["file_path"]
            if name in seen or name not in membership:
                raise AssemblyError("Duplicate or unsplit source frame: " + bid)
            seen.add(name)
            group = integer_group(frame.get("group_id"))
            split = membership[name]
            if group in group_splits and group_splits[group] != split:
                raise AssemblyError("Global training/held-out group leakage")
            group_splits[group] = split
            if split != "train_filenames" or frame.get("camera_id") != "camera_3":
                continue
            transform = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)):
                raise AssemblyError("Invalid training camera3 world transform")
            train_frames.append(dict(frame, group_id=group))
        if seen != set(membership) or not train_frames:
            raise AssemblyError("Missing actual TRAIN camera3 groups: " + bid)
        # Only training camera3 frames enter the existing resolver. The global
        # rank and normalized temporal-center score remain exactly its rule.
        metadata_by_id[bid] = {"frames": train_frames, "train_filenames": [f["file_path"] for f in train_frames]}
    positions, owners, groups = build_ownership(metadata_by_id)
    # Coincident poses from distinct groups tie by smallest integer group.
    _, first = np.unique(positions, axis=0, return_index=True)
    first.sort()
    return Ownership(positions[first], owners[first], np.asarray(groups, dtype="<i8")[first]), {
        "rule": "nearest world XYZ TRAIN camera3 pose; canonical build_ownership temporal-center resolution",
        "training_positions_only": True, "held_out_poses_used": False,
        "training_group_count": len(groups), "distinct_world_positions": len(first),
        "nearest_pose_distance_tie": "smallest integer training group_id",
        "group_ids": groups, "group_owner_core_ids": owners.tolist(), "group_positions_m": positions.tolist(),
    }


class Ownership:
    def __init__(self, positions, owners, groups):
        self.positions, self.groups = positions, groups
        self.core_indices = np.asarray([EXPECTED_IDS.index(owner) for owner in owners], dtype="<u2")
        self.tree = cKDTree(positions)

    def assign(self, xyz, workers):
        if len(self.positions) == 1:
            nearest = np.zeros(len(xyz), dtype=np.int64)
        else:
            distances, pairs = self.tree.query(xyz, k=2, workers=workers)
            if not np.isfinite(distances).all():
                raise AssemblyError("Nonfinite distance to world training poses")
            nearest = pairs[:, 0].copy()
            for row in np.flatnonzero(distances[:, 0] == distances[:, 1]):
                choices = self.tree.query_ball_point(xyz[row], np.nextafter(distances[row, 0], np.inf))
                offsets = self.positions[choices] - xyz[row]
                squared = np.sum(offsets * offsets, axis=1)
                tied = np.asarray(choices)[squared == squared.min()]
                nearest[row] = tied[np.argmin(self.groups[tied])]
        return self.core_indices[nearest], self.groups[nearest]


def reference_pool(project, source, guard, max_points):
    explicit = source["measured_reference"]
    paths = [explicit] if explicit is not None else [
        local_path(project, f"data/garage_surface_v1/{bid}/lidar_reference.npz") for bid in source["core_ids"]]
    arrays, total, expanded = [], 0, 0
    for path in paths:
        guard.snapshot(path)
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 32 or len({info.filename for info in infos}) != len(infos):
                raise AssemblyError("Invalid measured reference archive")
            size = sum(info.file_size for info in infos)
            expanded += size
            if size > MAX_REFERENCE_BYTES or expanded > 2 * MAX_REFERENCE_BYTES:
                raise AssemblyError("Expanded measured reference exceeds byte budget")
            info = archive.getinfo("xyz.npy")
            with archive.open(info) as handle:
                version = np.lib.format.read_magic(handle)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise AssemblyError("Unsupported measured XYZ NPY header")
                shape, _, dtype = readers[version](handle)
                if (len(shape) != 2 or shape[1] != 3 or not 0 < shape[0] <= max_points - total
                        or dtype.kind != "f" or dtype.itemsize not in (4, 8)
                        or info.file_size != handle.tell() + shape[0] * 3 * dtype.itemsize):
                    raise AssemblyError("Invalid or oversized measured XYZ reference")
        with np.load(path, allow_pickle=False) as archive:
            xyz = archive["xyz"]
        if not np.isfinite(xyz).all():
            raise AssemblyError("Nonfinite measured XYZ reference")
        arrays.append(xyz.astype(np.float64, copy=False)); total += len(xyz)
        if _state(path) != guard.files[path]["state"]:
            raise AssemblyError("Measured reference changed while loading")
    # Exact duplicate XYZ rows only: no voxel average, interpolation, or new points.
    xyz = np.unique(np.concatenate(arrays), axis=0)
    return cKDTree(xyz), {
        "paths": [p.relative_to(project).as_posix() for p in paths],
        "input_rows": total, "unique_xyz_rows": len(xyz),
        "deduplication": "exact existing world XYZ rows; no synthesized measurements",
        "explicit_joint_reference": explicit is not None,
        "independent_ground_truth": False,
    }


def bounds(xyz):
    if not len(xyz):
        return None
    return {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()}


def merge_bounds(a, b):
    # A chunk may contain no offered/retained records even after earlier chunks
    # contributed a bbox. The packaging helper expects a nonempty second bbox.
    return a if b is None else _merge_bounds(a, b)


class Footprint:
    """Bounded world XYZ occupancy disclosure, never a surface-coverage test."""
    def __init__(self):
        self.cells, self.truncated = set(), False

    def add(self, xyz):
        if self.truncated or not len(xyz):
            return
        cells = np.floor(xyz.astype(np.float64))  # Fixed 1 m world XYZ cells.
        if np.any(np.abs(cells) >= 2**63):
            raise AssemblyError("XYZ exceeds footprint integer range")
        for cell in np.unique(cells.astype(np.int64), axis=0):
            key = tuple(int(v) for v in cell)
            if key not in self.cells and len(self.cells) >= FOOTPRINT_CELL_LIMIT:
                self.truncated = True
                break
            self.cells.add(key)


def scan_source(source, ownership, support, radius, chunk_bytes, workers, consume):
    if _state(source["path"]) != source["state"]:
        raise AssemblyError("Source changed before read")
    digest = hashlib.sha256()
    with source["path"].open("rb") as handle:
        spec = inspect_ply(handle)
        if "spec" in source and spec != source["spec"]:
            raise AssemblyError("Source PLY schema changed between passes")
        source["spec"] = spec
        names, stride = spec["names"], spec["stride"]
        if source["state"][2] != len(spec["header"]) + spec["count"] * stride:
            raise AssemblyError("Incomplete or trailing PLY payload")
        for prefix, width in {"f_dc_": 3, "scale_": 3, "rot_": 4}.items():
            if {n for n in names if n.startswith(prefix)} != {prefix + str(i) for i in range(width)}:
                raise AssemblyError("Invalid standard GS attribute family: " + prefix)
        per_chunk = chunk_bytes // stride
        if not per_chunk:
            raise AssemblyError("Chunk budget cannot hold one PLY record")
        xyz_columns = [names.index(n) for n in ("x", "y", "z")]
        rot_columns = [names.index(f"rot_{i}") for i in range(4)]
        covered = np.asarray([EXPECTED_IDS.index(bid) for bid in source["core_ids"]])
        digest.update(spec["header"])
        for start in range(0, spec["count"], per_chunk):
            count = min(per_chunk, spec["count"] - start)
            raw = handle.read(count * stride)
            if len(raw) != count * stride:
                raise AssemblyError("Incomplete PLY payload")
            digest.update(raw)
            rows = np.frombuffer(raw, dtype="<f4").reshape(count, len(names))
            if not np.isfinite(rows).all():
                raise AssemblyError("Nonfinite PLY attribute in payload")
            norms = np.linalg.norm(rows[:, rot_columns].astype(np.float64), axis=1)
            if not np.allclose(norms, 1., atol=1e-4, rtol=0):
                raise AssemblyError("Candidate requires unit wxyz quaternions; no normalization is performed")
            xyz = rows[:, xyz_columns]
            owners, groups = ownership.assign(xyz, workers)
            if source.get("ownership_mode", "legacy-camera") == "none":
                # Preserve source membership. Assigning every record to a global
                # nearest camera would silently migrate structure across floors.
                if len(covered) == 1:
                    owners = np.full(count, covered[0], dtype="<u2")
                else:
                    owners, groups = source["partition"].assign(xyz, workers)
                offered = np.ones(count, dtype=bool)
            elif source.get("ownership_mode") == "lidar-support":
                owners = np.full(count, covered[0], dtype="<u2")
                offered = source["ownership_keep"][start:start+count]
            else:
                offered = np.isin(owners, covered)
            distances = np.full(count, -1., dtype=np.float64)
            if support is not None:
                distances[offered] = support.query(xyz[offered], workers=workers)[0]
                if not np.isfinite(distances).all():
                    raise AssemblyError("Nonfinite measured support distance")
            keep = offered if support is None else offered & (distances <= radius)
            consume(spec, start, raw, xyz, owners, groups, offered, keep, distances)
        if handle.read(1):
            raise AssemblyError("Trailing PLY payload")
    if _state(source["path"]) != source["state"]:
        raise AssemblyError("Source changed during streaming")
    return digest.hexdigest()


class Artifact:
    def __init__(self, path, count, names, stack, provenance=False):
        self.path, self.count, self.written, self.bbox = path, count, 0, None
        self.digest = hashlib.sha256()
        self.handle = stack.enter_context(path.open("xb"))
        if provenance:
            np.lib.format.write_array_header_1_0(self.handle, {
                "descr": PROVENANCE.descr, "fortran_order": False, "shape": (count,)})
            # Include the actual NPY header in the checksum.
            self.handle.flush()
            with path.open("rb") as header:
                self.digest.update(header.read(self.handle.tell()))
        else:
            self.write_bytes(ply_header(names, count))
        self.expected_bytes = self.handle.tell() + count * (PROVENANCE.itemsize if provenance else 4 * len(names))

    def write_bytes(self, raw):
        if self.handle.write(raw) != len(raw):
            raise OSError("Short artifact write: " + str(self.path))
        self.digest.update(raw)

    def append(self, raw, count, xyz=None):
        self.write_bytes(raw); self.written += count
        if xyz is not None:
            self.bbox = merge_bounds(self.bbox, bounds(xyz))
        if self.written > self.count:
            raise AssemblyError("Artifact count exceeded declared header")

    def finish(self, project):
        if self.written != self.count:
            raise AssemblyError("Artifact count does not match declared header")
        self.handle.flush()
        if self.path.stat().st_size != self.expected_bytes:
            raise AssemblyError("Artifact byte count does not match declared payload")
        return {"path": self.path.relative_to(project).as_posix(), "gaussians": self.count,
                "file_bytes": self.path.stat().st_size, "sha256": self.digest.hexdigest(), "bbox_m": self.bbox}


def write_json(path, data):
    with path.open("xb") as handle:
        handle.write((json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())


def assemble(output, *, source_plan=None, manifest="maps/garage_manifest.json", support_radius_m=.2,
             chunk_bytes=8 * 1024**2, overview_max_gaussians=OVERVIEW_LIMIT,
             core_file_max_gaussians=CORE_FILE_LIMIT, reference_max_points=DEFAULT_REFERENCE_POINTS,
             workers=1, project=ROOT, ownership_mode="legacy-camera", core_ids=None,
             preflight_only=False, ownership_selection=None):
    project = Path(project).resolve()
    if project != ROOT:
        local_path(ROOT, project)  # Synthetic test roots must also stay inside ROOT.
    destination = local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError("Output must be a NEW directory with an existing parent")
    for value, label, ceiling in ((chunk_bytes, "chunk_bytes", MAX_CHUNK_BYTES),
                                  (overview_max_gaussians, "overview_max_gaussians", OVERVIEW_LIMIT),
                                  (core_file_max_gaussians, "core_file_max_gaussians", CORE_FILE_LIMIT),
                                  (reference_max_points, "reference_max_points", 8_000_000),
                                  (workers, "workers", 8)):
        positive_int(value, label)
        if value > ceiling:
            raise AssemblyError(label + " exceeds bounded budget")
    if support_radius_m is not None and (not np.isfinite(support_radius_m) or support_radius_m <= 0):
        raise AssemblyError("Measured support radius must be positive and finite")
    if ownership_mode not in OWNERSHIP_MODES:
        raise AssemblyError("Unknown ownership_mode")
    if ownership_mode == "lidar-support" and ownership_selection is None:
        raise AssemblyError("lidar-support requires the audited planner in assemble_core_gaussians_lidar_ownership.py")
    guard = InputGuard(project)
    ledger = guard.json(local_path(project, manifest))
    if (type(ledger.get("schema_version")) is not int or ledger.get("schema_version") != 1
            or ledger.get("coordinates") != "camera_init"
            or ledger.get("units") != "m" or ledger.get("dataset_kind", "real") not in ("real", "synthetic")):
        raise AssemblyError("Original manifest must use shared camera_init/m schema 1")
    blocks = ledger.get("blocks")
    if (not isinstance(blocks, list) or len(blocks) != 28 or any(not isinstance(b, dict) for b in blocks)
            or any(not isinstance(b.get("id"), str) for b in blocks)
            or {b.get("id") for b in blocks} != set(EXPECTED_IDS)):
        raise AssemblyError("Original manifest must contain all 28 distinct canonical core IDs")
    if any(b.get("coordinates") != "camera_init" or b.get("units", "m") != "m"
           or not isinstance(b.get("training"), dict) or b["training"].get("status") != "trained" for b in blocks):
        raise AssemblyError("All 28 originals must be trained in shared camera_init/m")
    up = ledger.get("camera_up", [0, 0, 1])
    if (not isinstance(up, list) or len(up) != 3 or any(type(v) not in (int, float) for v in up)
            or not np.isfinite(up).all() or np.linalg.norm(up) == 0):
        raise AssemblyError("camera_up must be a finite nonzero 3-vector")
    plan = default_source_plan() if source_plan is None else guard.json(local_path(project, source_plan))
    sources, selected = select_cores(parse_plan(project, plan), core_ids)
    selected_indices = [EXPECTED_IDS.index(bid) for bid in selected]
    input_bytes = sum(s["state"][2] for s in sources)
    preflight = resource_preflight(destination, disk_bytes=3 * input_bytes + 512 * 1024**2,
                                   memory_bytes=20 * chunk_bytes + 8 * MAX_REFERENCE_BYTES)
    if preflight_only:
        return {"status": "preflight_passed", "preflight": preflight, "selected_core_ids": selected,
                "ownership_mode": ownership_mode, "assembly_started": False}
    ownership, ownership_report = canonical_ownership(project, guard)
    ownership_report.update(mode=ownership_mode, compatibility_default="legacy-camera",
                            default_changed=False, duplicate_records_removed=0)
    blocks_by_id = {block["id"]: block for block in blocks}
    if ownership_mode == "lidar-support":
        ownership_selection["guard"].check()
        guard.files.update(ownership_selection["guard"].files)
        ownership_report.update(ownership_selection["report"])
        for source in sources:
            keep = ownership_selection["masks"][source["id"]]
            with source["path"].open("rb") as handle:
                size = inspect_ply(handle)["count"]
            if len(source["core_ids"]) != 1 or keep.dtype != np.bool_ or keep.shape != (size,):
                raise AssemblyError("Invalid audited LiDAR ownership selection")
            keep.setflags(write=False)
            source["ownership_keep"] = keep
    if ownership_mode == "none":
        ownership_report.update(
            rule="retain source records in original core membership; same-floor joint partition only",
            exclusion_enabled=False, cross_source_reassignment=False,
            nearest_train_group_role="diagnostic only for single-core sources; not an exclusion rule",
            floor_policy="preserve source floor labels and world XYZ; reject ambiguous cross-floor joint sources",
            floor_geometry_verified=False)
    else:
        ownership_report["exclusion_enabled"] = True
    for source in sources:
        source["ownership_mode"] = ownership_mode
        if ownership_mode == "none" and len(source["core_ids"]) > 1:
            floors = {blocks_by_id[bid].get("floor_id") for bid in source["core_ids"]}
            if len(floors) != 1 or next(iter(floors)) in (None, "unverified", "unknown"):
                raise AssemblyError("No-ownership joint source needs one known source floor; cross-floor partition is ambiguous")
            eligible = np.isin(ownership.core_indices, [EXPECTED_IDS.index(bid) for bid in source["core_ids"]])
            owners = [EXPECTED_IDS[i] for i in ownership.core_indices[eligible]]
            if set(owners) != set(source["core_ids"]):
                raise AssemblyError("No training positions for a joint source core")
            source["partition"] = Ownership(ownership.positions[eligible], owners, ownership.groups[eligible])
    counts = np.zeros(28, dtype=np.int64)
    cores = [{"id": bid, "offered_gaussians": 0, "offered_bbox_m": None, "retained_bbox_m": None,
              "offered_footprint": Footprint(), "retained_footprint": Footprint()} for bid in EXPECTED_IDS]
    candidate_bounds, names = None, None
    for source in sources:
        tree, reference = (None, None) if support_radius_m is None else reference_pool(project, source, guard, reference_max_points)
        info = {"candidate_gaussians": 0, "offered_gaussians": 0, "retained_gaussians": 0,
                "candidate_bbox_m": None, "retained_bbox_m": None, "measured_support": reference,
                "assigned_core_counts": {bid: 0 for bid in EXPECTED_IDS},
                "retained_core_counts": {bid: 0 for bid in source["core_ids"]}}
        def count_chunk(spec, start, raw, xyz, owners, groups, offered, keep, distances):
            nonlocal names
            if names is not None and spec["names"] != names:
                raise AssemblyError("Sources have different PLY schema/property order")
            names = spec["names"]
            info["candidate_gaussians"] += len(xyz)
            info["offered_gaussians"] += int(offered.sum())
            info["retained_gaussians"] += int(keep.sum())
            info["candidate_bbox_m"] = merge_bounds(info["candidate_bbox_m"], bounds(xyz))
            info["retained_bbox_m"] = merge_bounds(info["retained_bbox_m"], bounds(xyz[keep]))
            for core in np.unique(owners):
                info["assigned_core_counts"][EXPECTED_IDS[core]] += int(np.count_nonzero(owners == core))
            for core in np.unique(owners[offered]):
                offered_xyz, kept_xyz = xyz[offered & (owners == core)], xyz[keep & (owners == core)]
                item = cores[core]
                item["offered_gaussians"] += len(offered_xyz); counts[core] += len(kept_xyz)
                info["retained_core_counts"][EXPECTED_IDS[core]] += len(kept_xyz)
                item["offered_bbox_m"] = merge_bounds(item["offered_bbox_m"], bounds(offered_xyz))
                item["retained_bbox_m"] = merge_bounds(item["retained_bbox_m"], bounds(kept_xyz))
                item["offered_footprint"].add(offered_xyz); item["retained_footprint"].add(kept_xyz)
        source_hash = scan_source(source, ownership, tree, support_radius_m, chunk_bytes, workers, count_chunk)
        guard.remember(source["path"], source["state"], source_hash)
        info["ownership_rejected_gaussians"] = info["candidate_gaussians"] - info["offered_gaussians"]
        info["support_rejected_gaussians"] = info["offered_gaussians"] - info["retained_gaussians"]
        source["selection"] = info
        candidate_bounds = merge_bounds(candidate_bounds, info["candidate_bbox_m"])
        del tree
    empty = [EXPECTED_IDS[i] for i in selected_indices if counts[i] == 0]
    if empty:
        raise AssemblyError("Incomplete: zero retained records in core(s): " + ", ".join(empty))
    stride, total = 4 * len(names), int(counts.sum())
    overview_budget = min(overview_max_gaussians,
                          (VIEWER_MAX_BYTES - len(ply_header(names, overview_max_gaussians))) // stride)
    quotas = [0] * len(EXPECTED_IDS)
    for i, quota in zip(selected_indices, quotas_for(counts[selected_indices].tolist(), overview_budget)):
        quotas[i] = quota
    shard_limit = min(core_file_max_gaussians,
                      (VIEWER_MAX_BYTES - len(ply_header(names, core_file_max_gaussians))) // stride)
    if shard_limit < 1:
        raise AssemblyError("PLY schema cannot fit a viewer record")
    if sum((int(count) + shard_limit - 1) // shard_limit for count in counts) > MAX_CORE_SHARDS:
        raise AssemblyError("Per-core shard count exceeds viewer manifest budget")
    guard.check({s["path"] for s in sources})
    local_path(project, destination)
    # Recheck free disk after the read pass, just before creating any outputs.
    preflight = resource_preflight(destination, disk_bytes=2 * total * (stride + PROVENANCE.itemsize)
                                   + sum(quotas) * (stride + PROVENANCE.itemsize) + 16 * 1024**2,
                                   memory_bytes=20 * chunk_bytes + 8 * MAX_REFERENCE_BYTES)
    destination.mkdir(exist_ok=False)
    try:
        with ExitStack() as stack:
            full = Artifact(destination / "gaussians_full.ply", total, names, stack)
            provenance = Artifact(destination / "full_provenance.npy", total, names, stack, True)
            overview = Artifact(destination / "gaussians_overview.ply", sum(quotas), names, stack)
            overview_provenance = Artifact(destination / "overview_provenance.npy", sum(quotas), names, stack, True)
            # At most 28 active shard pairs. Complete shards close immediately.
            shards, active, core_offsets = [[] for _ in EXPECTED_IDS], {}, np.zeros(28, dtype=np.int64)
            def core_append(core, records, origins, xyz):
                cursor = 0
                while cursor < len(records):
                    if core not in active:
                        directory = destination / EXPECTED_IDS[core]
                        directory.mkdir(exist_ok=True)
                        number = len(shards[core])
                        size = min(shard_limit, int(counts[core] - core_offsets[core]))
                        stem = "gaussians" if counts[core] <= shard_limit else f"gaussians_{number:03d}"
                        active[core] = (Artifact(directory / (stem + ".ply"), size, names, stack),
                                        Artifact(directory / (stem + "_provenance.npy"), size, names, stack, True))
                    model, trace = active[core]
                    take = min(model.count - model.written, len(records) - cursor)
                    stop = cursor + take
                    model.append(records[cursor:stop].tobytes(), take, xyz[cursor:stop])
                    trace.append(origins[cursor:stop].tobytes(), take)
                    core_offsets[core] += take; cursor = stop
                    if model.written == model.count:
                        item = model.finish(project)
                        item["provenance"] = trace.finish(project)
                        shards[core].append(item)
                        model.handle.close(); trace.handle.close(); del active[core]
            for source_index, source in enumerate(sources):
                tree, reference = (None, None) if support_radius_m is None else reference_pool(project, source, guard, reference_max_points)
                if reference != source["selection"]["measured_support"]:
                    raise AssemblyError("Measured support changed between passes")
                def copy_chunk(spec, start, raw, xyz, owners, groups, offered, keep, distances):
                    chosen = np.flatnonzero(keep)
                    if not len(chosen):
                        return
                    records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, stride)[chosen]
                    kept_xyz, kept_owners = xyz[chosen], owners[chosen]
                    origins = np.empty(len(chosen), dtype=PROVENANCE)
                    origins["source_index"] = source_index
                    origins["source_record_index"] = start + chosen
                    origins["assigned_core"] = kept_owners
                    origins["nearest_train_group"] = groups[chosen]
                    origins["measured_support_checked"] = support_radius_m is not None
                    origins["measured_support_distance_m"] = distances[chosen]
                    full.append(records.tobytes(), len(chosen), kept_xyz)
                    provenance.append(origins.tobytes(), len(chosen))
                    overview_ids = []
                    for core in np.unique(kept_owners):
                        indices = np.flatnonzero(kept_owners == core)
                        start_core = int(core_offsets[core])
                        sampled = sample_indices(int(counts[core]), quotas[core], start_core, start_core + len(indices))
                        overview_ids.append(indices[sampled])
                        core_append(core, records[indices], origins[indices], kept_xyz[indices])
                    # Emit sampled records in source-record order, not core order
                    # within a chunk. Interleaved joint rows now give the same
                    # PLY and provenance bytes for every chunk boundary.
                    sampled = np.sort(np.concatenate(overview_ids))
                    if len(sampled):
                        overview.append(records[sampled].tobytes(), len(sampled), kept_xyz[sampled])
                        overview_provenance.append(origins[sampled].tobytes(), len(sampled))
                source_hash = scan_source(source, ownership, tree, support_radius_m, chunk_bytes, workers, copy_chunk)
                guard.remember(source["path"], source["state"], source_hash)
                del tree
            if active or not np.array_equal(core_offsets, counts):
                raise AssemblyError("Per-core artifacts are incomplete")
            full_info, overview_info = full.finish(project), overview.finish(project)
            full_info["provenance"] = provenance.finish(project)
            overview_info["provenance"] = overview_provenance.finish(project)
        guard.check({s["path"] for s in sources})
        core_reports, viewer_blocks = [], []
        for i, item in enumerate(cores):
            if i not in selected_indices:
                continue
            before, after = item.pop("offered_footprint"), item.pop("retained_footprint")
            exact = not (before.truncated or after.truncated)
            item.update(retained_gaussians=int(counts[i]), overview_gaussians=quotas[i], artifacts=shards[i],
                        support_rejected_gaussians=item["offered_gaussians"] - int(counts[i]),
                        footprint={"axes": "world XYZ", "cell_m": 1., "cell_budget_per_set": FOOTPRINT_CELL_LIMIT,
                                   "offered_occupied_cells": len(before.cells), "retained_occupied_cells": len(after.cells),
                                   "counts_exact": exact, "lost_cells": len(before.cells - after.cells) if exact else None,
                                   "offered_truncated": before.truncated, "retained_truncated": after.truncated,
                                   "surface_coverage_verified": False})
            core_reports.append(item)
            for part, artifact in enumerate(shards[i]):
                bid = item["id"] if len(shards[i]) == 1 else item["id"] + f"_part_{part:03d}"
                block = viewer_block(bid, artifact, [item["id"]])
                if ownership_mode == "none":
                    floor = blocks_by_id[item["id"]].get("floor_id")
                    block["floor_id"] = floor if floor not in ("unverified", "unknown") else None
                    block["floor_assignment"] = "source ledger label, not independently verified"
                viewer_blocks.append(block)
        normalized_plan = {"schema_version": 1, "coordinates": "camera_init", "units": "m", "sources": [
            dict(id=s["id"], path=s["path"].relative_to(project).as_posix(), core_ids=s["core_ids"],
                 measured_reference=s["measured_reference"].relative_to(project).as_posix()
                 if s["measured_reference"] is not None else None) for s in sources]}
        coverage = {"artifact_coverage_complete": len(selected) == 28, "expected_core_ids": list(EXPECTED_IDS),
                    "selected_core_ids": selected, "selected_core_coverage_complete": True,
                    "nonzero_core_count": len(selected), "surface_coverage_verified": False, "holes_assessed": False,
                    "note": "Nonzero selected-core artifacts only; subset trials are not all-28 delivery or surface completeness."}
        record = {"schema_version": 1, "status": STATUS, "coordinates": "camera_init", "units": "m",
                  **FALSE_FLAGS, "coverage": coverage, "ownership": ownership_report,
                  "processing": {"cpu_only": True, "chunk_bytes": chunk_bytes, "workers": workers,
                                 "resource_preflight": preflight, "ownership_mode": ownership_mode,
                                 "deduplication": "none; overlap retained without unsupported deletion",
                                 "reference_max_points": reference_max_points, "support_radius_m": support_radius_m,
                                 "measured_support_checked": support_radius_m is not None,
                                 "attributes_modified": False, "origin_reset": False, "per_block_centering": False,
                                 "floor_flattening": False, "surfaces_synthesized": False},
                  "properties": [{"name": n, "type": "float32"} for n in names],
                  "candidate_gaussians": sum(s["spec"]["count"] for s in sources),
                  "candidate_full_bbox_m": candidate_bounds, "retained_full_bbox_m": full_info["bbox_m"],
                  "retained_gaussians": total, "sources": [dict(
                      source_index=i, **normalized_plan["sources"][i], **s["selection"],
                      source_sha256=guard.files[s["path"]]["sha256"], source_stat=guard.report()[
                          s["path"].relative_to(project).as_posix()]["stat"]) for i, s in enumerate(sources)],
                  "inputs": guard.report(), "cores": core_reports, "full": full_info, "overview": overview_info,
                  "provenance": {"core_index_to_id": list(EXPECTED_IDS), "source_index_to_id": [s["id"] for s in sources],
                                 "row_alignment": "Each NPY row traces the corresponding PLY record, zero-based source index",
                                 "unchecked_support_distance_m": -1., "records_modified": False,
                                 "geometry_protocol": "Same input LiDAR, not independent measurement GT"}}
        common = {"schema_version": 1, "coordinates": "camera_init", "units": "m", "camera_up": up,
                  "dataset_kind": ledger.get("dataset_kind", "real"), "status": STATUS,
                  **FALSE_FLAGS, "coverage": coverage}
        # Keep the all-28 input contract even in a subset trial. Active sources
        # and their local provenance indices live in assembly.json.
        write_json(destination / "source_plan.json", plan if len(selected) != 28 else normalized_plan)
        scope_title = "All 28 cores" if len(selected) == 28 else f"{len(selected)} selected cores (subset trial)"
        write_json(destination / "manifest.json", dict(common, title=scope_title + " · candidate", blocks=viewer_blocks))
        write_json(destination / "overview_manifest.json", dict(common, title=scope_title + " · balanced candidate overview",
                   blocks=[viewer_block("garage_overview", overview_info, selected)]))
        guard.check({s["path"] for s in sources})
        # This is the last marker; every failure removes the exclusively owned directory.
        write_json(destination / "assembly.json", record)
        return record
    except BaseException:
        shutil.rmtree(destination)
        raise


def viewer_block(bid, artifact, cores):
    return {"id": bid, "label": bid, "floor_id": None, **FALSE_FLAGS,
            "coordinates": "camera_init", "units": "m", "path": artifact["path"], "bbox_m": artifact["bbox_m"],
            "core_ids": cores, "source_block_ids": cores, "provenance_path": artifact["provenance"]["path"],
            "training": {"status": "trained", "gaussians": artifact["gaussians"], "calibration_verified": False,
                         "method": "Unmodified records selected from configured trained sources; no new training"},
            "eval": {"status": "not_evaluated", "native_fullscope_evaluated": False,
                     "note": "Candidate selection and overview require native full-scope evaluation"}}


def audit_original_sources(output, project=ROOT):
    """Verify each original 12k checkpoint against every exported PLY attribute on CPU.

    Checkpoints are weights-only loaded with a small NumPy scalar allowlist.
    No pipeline/model/backend or CUDA context is constructed.
    """
    import codecs
    import gc
    project = Path(project).resolve()
    destination = local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError("Audit output must be a NEW directory with an existing parent")
    guard = InputGuard(project)
    plan = default_source_plan()
    sources = parse_plan(project, plan)
    details, largest = [], 0
    for source in sources:
        metrics = guard.json(source["path"].parent / "metrics.json")
        trained = guard.json(source["path"].parent.parent / "training_result.json")
        checkpoint = local_path(project, metrics["checkpoint"])
        config = local_path(project, metrics["config"])
        if (metrics.get("status") != "evaluated" or metrics.get("step") != 11999
                or trained.get("status") != "trained" or trained.get("iterations_requested") != 12000
                or metrics.get("source_ply") is not None or "source_ply" in metrics.get("parameter_origin", "")
                or metrics.get("world_parser_scale") != 1.
                or metrics.get("world_parser_transform") not in (np.eye(4).tolist(), np.eye(4)[:3].tolist())):
            raise AssemblyError("Original source provenance mismatch: " + source["id"])
        guard.snapshot(config)
        if not re.search(r"(?m)^max_num_iterations: 12000\s*$", config.read_text()):
            raise AssemblyError("Original source config is not 12000 iterations")
        if config.relative_to(project).as_posix() not in trained.get("configs", []):
            raise AssemblyError("Training result does not identify export config")
        filtering = metrics.get("gaussian_filter", {})
        if filtering.get("surface_distance_max_m", 0) != 0 or filtering.get("opacity_min", 0) != 0:
            raise AssemblyError("Audit only supports the original unfiltered export")
        largest = max(largest, checkpoint.stat().st_size + source["state"][2])
        details.append((source, metrics, checkpoint, config))
    preflight = resource_preflight(destination, disk_bytes=8 * 1024**2,
                                   memory_bytes=3 * largest + 1024**3)
    import torch
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype,
                                           type(np.dtype("float64")), codecs.encode])
    records = []
    for source, metrics, checkpoint, config in details:
        guard.snapshot(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if type(state.get("step")) is not int or state["step"] != 11999:
            raise AssemblyError("Actual checkpoint step must be 11999")
        params = {key.rsplit(".", 1)[-1]: tensor.detach().numpy()
                  for key, tensor in state["pipeline"].items() if key.startswith("_model.gauss_params.")}
        count = len(params["means"])
        with source["path"].open("rb") as handle:
            spec = inspect_ply(handle)
            digest = hashlib.sha256(spec["header"])
            rows_checked = 0
            max_error = 0.
            for start in range(0, count, 32768):
                stop = min(count, start + 32768)
                q = params["quats"][start:stop].copy()
                q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
                xyz = params["means"][start:stop]
                values = np.concatenate([xyz, np.zeros_like(xyz), params["features_dc"][start:stop].reshape(-1, 3),
                    params["features_rest"][start:stop].transpose(0, 2, 1).reshape(stop - start, -1),
                    params["opacities"][start:stop].reshape(-1, 1), params["scales"][start:stop], q], axis=1).astype("<f4")
                names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                names += [f"f_rest_{i}" for i in range(values.shape[1] - 17)]
                names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
                if list(spec["names"]) != names:
                    raise AssemblyError("Original export schema differs from checkpoint convention")
                valid = np.isfinite(values).all(axis=1) & (params["opacities"][start:stop].ravel() >= -5.5373)
                expected = values[valid]
                raw = handle.read(len(expected) * spec["stride"])
                digest.update(raw)
                if len(raw) != expected.nbytes:
                    raise AssemblyError("Original export payload truncated")
                actual = np.frombuffer(raw, dtype="<f4").reshape(expected.shape)
                error = float(np.max(np.abs(actual - expected))) if len(expected) else 0.
                max_error = max(max_error, error)
                # Native export normalizes quaternions in float32; permit only
                # float32 rounding there, demand exact other learned parameters.
                if (not np.array_equal(actual[:, :-4], expected[:, :-4])
                        or not np.allclose(actual[:, -4:], expected[:, -4:], atol=2e-7, rtol=0)):
                    raise AssemblyError("PLY does not match actual trained checkpoint: " + source["id"])
                rows_checked += len(expected)
            if handle.read(1) or rows_checked != spec["count"] or rows_checked != metrics["export"]["gaussians"]:
                raise AssemblyError("Original export record count mismatch")
        guard.remember(source["path"], source["state"], digest.hexdigest())
        # Bind the support file without loading or treating it as independent GT.
        guard.snapshot(source["measured_reference"])
        records.append({"id": source["id"], "path": source["path"].relative_to(project).as_posix(),
                        "checkpoint": checkpoint.relative_to(project).as_posix(), "checkpoint_step": state["step"],
                        "iterations_completed": state["step"] + 1, "checkpoint_gaussians": count,
                        "exported_gaussians": rows_checked, "export_opacity_logit_min": -5.5373,
                        "checkpoint_export_max_abs_error": max_error,
                        "all_export_records_compared": True, "nonrotation_parameters_exact": True,
                        "source_ply_override": False, "source_sha256": digest.hexdigest(),
                        "old_dev_psnr_db_mean": metrics.get("psnr_db_mean")})
        print(json.dumps({"source_audited": source["id"], "step": state["step"], "records": rows_checked}), flush=True)
        del params, state, values, actual, expected
        gc.collect()
    guard.check({s["path"] for s in sources})
    report = {"schema_version": 1, "status": "verified_original_12000_step_exports", "cpu_only": True,
              "preflight": preflight, "sources": records, "inputs": guard.report(),
              "total_exported_gaussians": sum(r["exported_gaussians"] for r in records),
              "quality_accepted": False, "same_mask_paired_evaluation_pending": True}
    destination.mkdir()
    try:
        write_json(destination / "source_plan.json", plan)
        write_json(destination / "source_audit.json", report)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report


def support_surface_labels(normals, valid, relative_z, lateral_distance):
    """Frozen diagnostic labels, independent of opacity/selection/normal sign.

    Ground/ceiling are only trajectory-relative proxies, never surveyed floors.
    """
    normals = np.asarray(normals, dtype=np.float64)
    length = np.linalg.norm(normals, axis=1)
    trusted = np.asarray(valid, dtype=bool) & np.isfinite(normals).all(axis=1) & (length > 1e-6)
    nz = np.abs(normals[:, 2]) / np.maximum(length, 1e-12)
    labels = np.full(len(normals), 0, dtype=np.uint8)  # unknown normal
    labels[trusted] = 1  # inclined
    labels[trusted & (nz <= .3)] = 2  # wall-like
    horizontal = trusted & (nz >= .9)
    labels[horizontal] = 3  # horizontal, floor/ceiling ambiguous
    near = lateral_distance <= 10.
    labels[horizontal & near & (relative_z >= -2.5) & (relative_z <= -.5)] = 4
    labels[horizontal & near & (relative_z >= .5) & (relative_z <= 2.5)] = 5
    return labels


def audit_support_rejections(output, *, source_plan=None, core_ids=None, radius=.2, workers=1, project=ROOT):
    """CPU-only fixed-normal, distance and plane-offset census of original sources."""
    project = Path(project).resolve()
    destination = local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError("Support report must be a NEW file with an existing parent")
    if not np.isfinite(radius) or radius <= 0:
        raise AssemblyError("Support radius must be positive finite")
    positive_int(workers, "workers")
    if workers > 8:
        raise AssemblyError("workers exceeds bounded budget")
    guard = InputGuard(project)
    plan = default_source_plan() if source_plan is None else guard.json(local_path(project, source_plan))
    sources, selected = select_cores(parse_plan(project, plan), core_ids)
    if any(len(s["core_ids"]) != 1 for s in sources):
        raise AssemblyError("Fixed-normal census requires single-core original sources")
    # At most three float64 vectors and one uint8 label per Gaussian, plus
    # temporary grouped/quantile arrays and one source's bounded reference tree.
    source_bytes = sum(s["state"][2] for s in sources)
    preflight = resource_preflight(destination, disk_bytes=16 * 1024**2,
                                   memory_bytes=2 * source_bytes + 1024**3)
    canonical_ownership(project, guard)  # Validate actual TRAIN/held-out topology.
    categories = ["normal_unknown", "inclined", "wall_like", "horizontal_ambiguous",
                  "floor_like_below_train_camera", "ceiling_like_above_train_camera"]
    aggregate = [[] for _ in categories]

    def stats(values):
        if not len(values):
            return {"count": 0, "rejected": 0, "rejected_fraction": None}
        distance = values[:, 0]
        rejected = distance > radius
        def quantiles(x):
            x = x[np.isfinite(x)]
            return dict(zip(("p50", "p75", "p90", "p95", "p99"), np.quantile(x, [.5, .75, .9, .95, .99]).tolist())) if len(x) else None
        return {"count": len(distance), "rejected": int(rejected.sum()), "rejected_fraction": float(rejected.mean()),
                "distance_m": quantiles(distance), "rejected_distance_m": quantiles(distance[rejected]),
                "rejected_abs_plane_offset_m": quantiles(values[rejected, 1]),
                "rejected_tangent_offset_m": quantiles(values[rejected, 2]),
                "fraction_beyond_radius": {str(t): float((distance > t).mean()) for t in (.1, .2, .3, .5)},
                "rejected_plane_offset_le_0_05m": int(np.count_nonzero(rejected & (values[:, 1] <= .05)))}

    per_source = []
    for source in sources:
        tree, reference = reference_pool(project, source, guard, DEFAULT_REFERENCE_POINTS)
        if len(reference["paths"]) != 1:
            raise AssemblyError("Fixed normals require one reference file")
        ref_path = local_path(project, reference["paths"][0])
        with np.load(ref_path, allow_pickle=False) as archive:
            raw_xyz, normals, valid = archive["xyz"], archive["normals"], archive["normal_valid"]
        if normals.shape != raw_xyz.shape or valid.shape != (len(raw_xyz),) or valid.dtype.kind != "b":
            raise AssemblyError("Invalid fixed reference normals")
        xyz, first = np.unique(raw_xyz.astype(np.float64), axis=0, return_index=True)
        if not np.array_equal(xyz, tree.data):
            raise AssemblyError("Fixed normal reference differs from selection reference")
        normals, valid = normals[first], valid[first]
        meta = guard.json(local_path(project, f"data/garage_surface_v1/{source['core_ids'][0]}/transforms.json"))
        training = set(meta["train_filenames"])
        poses = np.asarray([f["transform_matrix"] for f in meta["frames"]
                            if f["file_path"] in training and f["camera_id"] == "camera_3"])[:, :3, 3]
        _, nearest = cKDTree(poses).query(xyz, workers=workers)
        offset = xyz - poses[nearest]
        ref_labels = support_surface_labels(normals, valid, offset[:, 2], np.linalg.norm(offset[:, :2], axis=1))
        normalized = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        chunks = [[] for _ in categories]
        digest = hashlib.sha256()
        with source["path"].open("rb") as handle:
            spec = inspect_ply(handle)
            if source["state"][2] != len(spec["header"]) + spec["count"] * spec["stride"]:
                raise AssemblyError("Incomplete or trailing source payload")
            digest.update(spec["header"])
            columns = [spec["names"].index(n) for n in ("x", "y", "z")]
            for start in range(0, spec["count"], 32768):
                count = min(32768, spec["count"] - start)
                raw = handle.read(count * spec["stride"])
                if len(raw) != count * spec["stride"]:
                    raise AssemblyError("Incomplete source payload")
                digest.update(raw)
                rows = np.frombuffer(raw, dtype="<f4").reshape(count, -1)
                if not np.isfinite(rows).all():
                    raise AssemblyError("Nonfinite source attribute")
                points = rows[:, columns]
                if len(xyz) > 1:
                    distances, nn = tree.query(points, k=2, workers=workers)
                else:
                    d, n = tree.query(points, workers=workers)
                    distances = np.column_stack([d, np.full(count, np.inf)])
                    nn = np.column_stack([n, n])
                distance, index = distances[:, 0], nn[:, 0]
                labels = ref_labels[index].copy()
                # Equidistant references with disagreeing labels cannot supply
                # a reliable surface class. Distances remain exact regardless.
                ambiguous = np.isclose(distances[:, 0], distances[:, 1], atol=1e-9, rtol=0) & (labels != ref_labels[nn[:, 1]])
                labels[ambiguous] = 0
                normal_offset = np.abs(np.sum((points - xyz[index]) * normalized[index], axis=1))
                normal_offset[labels == 0] = np.nan
                tangent = np.sqrt(np.maximum(0., distance**2 - normal_offset**2))
                values = np.column_stack([distance, normal_offset, tangent])
                for i in range(len(categories)):
                    chunks[i].append(values[labels == i])
        guard.remember(source["path"], source["state"], digest.hexdigest())
        classes = {}
        for i, name in enumerate(categories):
            values = np.concatenate(chunks[i])
            aggregate[i].append(values)
            classes[name] = stats(values)
        per_source.append({"id": source["id"], "count": spec["count"], "source_sha256": digest.hexdigest(),
                           "reference": reference, "classes": classes})
        print(json.dumps({"support_census": source["id"], "count": spec["count"],
                          "rejected": sum(s["rejected"] for s in classes.values())}), flush=True)
        del tree
    summary = {name: stats(np.concatenate(aggregate[i])) for i, name in enumerate(categories)}
    horizontal = stats(np.concatenate([v for i in (3, 4, 5) for v in aggregate[i]]))
    guard.check({s["path"] for s in sources})
    report = {"schema_version": 1, "status": "cpu_fixed_reference_normal_support_census", "preflight": preflight,
              "cpu_only": True, "selected_core_ids": selected, "support_radius_m": radius,
              "gaussians": sum(s["count"] for s in per_source),
              "support_rejected": sum(s["rejected"] for s in summary.values()),
              "protocol": {"normals": "unchanged measured-reference PCA normals; ignore eigenvector sign",
                           "unit_normal_abs_z_horizontal_min": .9, "unit_normal_abs_z_wall_max": .3,
                           "camera_poses": "same-source TRAIN camera3 only; nearest world XYZ to reference point",
                           "floor_like_reference_minus_camera_z_m": [-2.5, -.5],
                           "ceiling_like_reference_minus_camera_z_m": [.5, 2.5], "max_lateral_camera_distance_m": 10.,
                           "duplicates": "exact reference XYZ only; retain first normal in input order",
                           "equidistant_disagreeing_classes": "normal_unknown",
                           "weights": "per Gaussian, not per pixel, area, voxel or independent GT",
                           "floor_labels_verified": False, "classification_extrapolated_for_far_gaussians": True,
                           "all_thresholds_fixed_before_counting": True},
              "classes": summary, "all_horizontal": horizontal, "sources": per_source, "inputs": guard.report(),
              "limitation": "Distance to sparse sampled reference includes tangential sampling gaps. "
                  "Plane-offset at a distant nearest point is only a local proxy. Floor-like includes car roofs/ramps/platforms; "
                  "unoriented normals cannot establish floor versus ceiling. No threshold/default changed."}
    write_json(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New directory inside this project; parent must exist")
    parser.add_argument("--source-plan", help="JSON object containing sources [{id,path,core_ids,measured_reference?}]")
    parser.add_argument("--manifest", default="maps/garage_manifest.json", help="Read-only original 28-core ledger")
    helpers = parser.add_mutually_exclusive_group()
    helpers.add_argument("--write-default-source-plan", help="Write only a new proposed originals plan; no assembly/validation")
    helpers.add_argument("--write-initial-source-plan", help="Write only a new proposed mixed 014..016 joint plan; no assembly/validation")
    helpers.add_argument("--audit-original-sources", help="NEW report directory: CPU-check all 28 original exports against 12000-step checkpoints")
    helpers.add_argument("--audit-support-rejections", help="NEW JSON report: fixed LiDAR normal/distance census, no assembly")
    ownership_options = parser.add_mutually_exclusive_group()
    ownership_options.add_argument("--ownership-mode", choices=OWNERSHIP_MODES, default="legacy-camera")
    ownership_options.add_argument("--no-ownership", action="store_true", help="Alias for --ownership-mode none; preserve source membership")
    parser.add_argument("--core-ids", nargs="+", choices=EXPECTED_IDS, help="CPU subset trial; input plan still covers all 28")
    parser.add_argument("--preflight-only", action="store_true", help="Check memory/disk budgets; create no output")
    support = parser.add_mutually_exclusive_group()
    support.add_argument("--support-radius-m", type=float, default=.2)
    support.add_argument("--no-measured-support", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--overview-max-gaussians", type=int, default=OVERVIEW_LIMIT)
    parser.add_argument("--core-file-max-gaussians", type=int, default=CORE_FILE_LIMIT)
    parser.add_argument("--reference-max-points", type=int, default=DEFAULT_REFERENCE_POINTS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.audit_support_rejections:
            if args.output or args.no_measured_support:
                raise AssemblyError("Support census cannot combine with assembly/no-measured-support")
            record = audit_support_rejections(args.audit_support_rejections, source_plan=args.source_plan,
                core_ids=args.core_ids, radius=args.support_radius_m, workers=args.workers, project=ROOT)
            print(json.dumps({"status": record["status"], "output": args.audit_support_rejections,
                              "support_rejected": record["support_rejected"]}))
            return 0
        if args.audit_original_sources:
            if args.output or args.source_plan or args.core_ids:
                raise AssemblyError("Original audit cannot be combined with assembly options")
            record = audit_original_sources(args.audit_original_sources, project=ROOT)
            print(json.dumps({"status": record["status"], "output": args.audit_original_sources,
                              "sources": len(record["sources"])}))
            return 0
        helper = args.write_default_source_plan or args.write_initial_source_plan
        if helper:
            if args.output or args.source_plan:
                raise AssemblyError("Plan helper cannot be combined with assembly/source-plan")
            plan = initial_source_plan() if args.write_initial_source_plan else default_source_plan()
            write_json(local_path(ROOT, helper), plan)
            print(json.dumps({"status": "proposed_source_plan_written", "path": helper,
                              "inputs_verified": False, "assembly_started": False}))
            return 0
        if not args.output:
            raise AssemblyError("--output is required for assembly")
        record = assemble(args.output, source_plan=args.source_plan, manifest=args.manifest,
                          support_radius_m=None if args.no_measured_support else args.support_radius_m,
                          chunk_bytes=args.chunk_mib * 1024**2, overview_max_gaussians=args.overview_max_gaussians,
                          core_file_max_gaussians=args.core_file_max_gaussians,
                          reference_max_points=args.reference_max_points, workers=args.workers,
                          ownership_mode="none" if args.no_ownership else args.ownership_mode,
                          core_ids=args.core_ids, preflight_only=args.preflight_only, project=ROOT)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "incomplete", "artifact_coverage_complete": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": record["status"], "output": args.output,
                      "nonzero_core_count": record.get("coverage", {}).get("nonzero_core_count", 0),
                      "retained_gaussians": record.get("retained_gaussians"),
                      "preflight": record.get("preflight")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

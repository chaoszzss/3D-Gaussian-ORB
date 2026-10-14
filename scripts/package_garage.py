#!/usr/bin/env python3
"""CPU-only, lossless packaging of all 28 collected garage trajectory blocks."""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = tuple("block_%03d" % i for i in range(28))
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 65536
MAX_CHUNK_BYTES = 64 * 1024 * 1024
VIEWER_MAX_GAUSSIANS = 1_500_000
VIEWER_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_OVERVIEW_GAUSSIANS = 1_200_000
REQUIRED = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
COVERAGE_NOTE = ("Complete means all 28 planned collection-trajectory blocks have trained, "
                 "evaluated exports. It does not certify every observed surface, unobserved "
                 "building areas, floor identities, calibration, or reconstruction quality.")


class PackageError(ValueError):
    pass


def local_path(project, value):
    """Use only visible project paths; reject symlinks, including internal ones."""
    raw = str(value)
    if not raw or any(c in raw for c in ("\\", ":", "\x00")):
        raise PackageError("Invalid project path: %r" % raw)
    path = Path(raw)
    if ".." in path.parts:
        raise PackageError("Path traversal is forbidden: " + raw)
    if not path.is_absolute():
        path = project / path
    try:
        relative = path.relative_to(project)
    except ValueError:
        raise PackageError("Path must stay inside project: " + raw)
    if not relative.parts or any(p.startswith(".") for p in relative.parts):
        raise PackageError("Path must be a visible child of project: " + raw)
    cursor = project
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PackageError("Symlink paths are forbidden: " + raw)
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackageError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise PackageError("Nonfinite JSON value: " + value)


def read_json(path, snapshots=None):
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise PackageError("JSON exceeds bounded read budget: " + str(path))
    try:
        result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError("Invalid JSON %s: %s" % (path, exc))
    if not isinstance(result, dict):
        raise PackageError("JSON must be an object: " + str(path))
    if snapshots is not None:
        snapshots[path] = hashlib.sha256(raw).hexdigest()
    return result


def positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise PackageError(label + " must be a positive integer")
    return value


def index_blocks(data, label):
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        raise PackageError(label + " requires blocks")
    result = {}
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("id"), str):
            raise PackageError(label + " has an invalid block ID")
        if block["id"] in result:
            raise PackageError(label + " has duplicate ID: " + block["id"])
        result[block["id"]] = block
    missing, extra = set(EXPECTED_IDS) - result.keys(), result.keys() - set(EXPECTED_IDS)
    if missing or extra:
        raise PackageError("%s must contain all 28 predetermined IDs; missing=%s extra=%s" %
                           (label, sorted(missing), sorted(extra)))
    return result


def _state(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise PackageError("Expected a regular source file: " + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def inspect_ply(handle):
    """Parse a bounded header; the caller must scan the ENTIRE payload."""
    raw = bytearray()
    while len(raw) < MAX_HEADER_BYTES:
        line = handle.readline(min(4096, MAX_HEADER_BYTES - len(raw)))
        if not line or not line.endswith(b"\n"):
            raise PackageError("Incomplete or oversized PLY header")
        raw.extend(line)
        if line.strip() == b"end_header":
            break
    else:
        raise PackageError("Oversized PLY header")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        raise PackageError("PLY header must be ASCII")
    if not lines or lines[0] != "ply":
        raise PackageError("Not a standard PLY")
    formatted, count, names = False, None, []
    for line in lines[1:]:
        tokens = line.split()
        if not tokens or tokens[0] in ("comment", "obj_info", "end_header"):
            continue
        if tokens[0] == "format":
            if formatted or count is not None or tokens != ["format", "binary_little_endian", "1.0"]:
                raise PackageError("Need standard binary_little_endian PLY 1.0")
            formatted = True
        elif tokens[0] == "element":
            if not formatted or count is not None or len(tokens) != 3 or tokens[1] != "vertex":
                raise PackageError("Only one vertex element is allowed")
            try:
                count = positive_int(int(tokens[2]), "PLY vertex count")
            except ValueError as exc:
                raise PackageError("Invalid PLY vertex count: " + str(exc))
        elif tokens[0] == "property":
            if count is None or len(tokens) != 3 or tokens[1] not in ("float", "float32"):
                raise PackageError("Every PLY property must be scalar float32")
            if tokens[2] in names:
                raise PackageError("Duplicate PLY property: " + tokens[2])
            names.append(tokens[2])
        else:
            raise PackageError("Unsupported PLY declaration: " + line)
    if not formatted or count is None or not set(REQUIRED).issubset(names):
        raise PackageError("PLY is missing standard GS XYZ/SH/opacity/scale/rotation fields")
    rest = [n for n in names if n.startswith("f_rest_")]
    if len(rest) not in (0, 9, 24, 45) or set(rest) != {"f_rest_%d" % i for i in range(len(rest))}:
        raise PackageError("Incomplete SH schema; require consecutive degree 0..3 coefficients")
    return {"header": bytes(raw), "count": count, "names": tuple(names),
            "stride": 4 * len(names), "sh_degree": {0: 0, 9: 1, 24: 2, 45: 3}[len(rest)]}


def ply_header(names, count):
    return ("ply\nformat binary_little_endian 1.0\n"
            "comment original trained GS records; shared camera_init metres\n"
            "element vertex %d\n" % count +
            "".join("property float %s\n" % name for name in names) + "end_header\n").encode("ascii")


def _bounds(values):
    return {"min": values.min(axis=0).astype(float).tolist(),
            "max": values.max(axis=0).astype(float).tolist()}


def merge_bounds(a, b):
    if a is None:
        return b
    return {"min": [min(x, y) for x, y in zip(a["min"], b["min"])],
            "max": [max(x, y) for x, y in zip(a["max"], b["max"])]}


def sample_indices(count, quota, start, end):
    """One midpoint per equal record-index stratum; only this chunk's indices."""
    def ceil_div(a, b):
        return -(-a // b)
    first = max(0, ceil_div(2 * quota * start - count, 2 * count))
    stop = min(quota, ceil_div(2 * quota * end - count, 2 * count))
    return np.fromiter((((2 * j + 1) * count) // (2 * quota) - start
                        for j in range(first, stop)), dtype=np.int64, count=stop - first)


def scan_ply(source, chunk_bytes, full=None, overview=None, quota=0):
    path, spec = source["path"], source["spec"]
    if _state(path) != source["state"]:
        raise PackageError("Source changed before read: " + str(path))
    stride = spec["stride"]
    rows_per_chunk = chunk_bytes // stride
    if rows_per_chunk < 1:
        raise PackageError("Chunk byte budget cannot hold one complete record")
    digest, bounds, sampled_bounds, sampled = hashlib.sha256(), None, None, 0
    xyz = [spec["names"].index(n) for n in ("x", "y", "z")]
    rot = [spec["names"].index("rot_%d" % i) for i in range(4)]
    with path.open("rb") as handle:
        header = inspect_ply(handle)
        if header != spec:
            raise PackageError("Source PLY header changed: " + str(path))
        digest.update(spec["header"])
        for start in range(0, spec["count"], rows_per_chunk):
            size = min(rows_per_chunk, spec["count"] - start)
            raw = handle.read(size * stride)
            if len(raw) != size * stride:
                raise PackageError("Incomplete PLY payload: " + str(path))
            rows = np.frombuffer(raw, dtype="<f4").reshape(size, len(spec["names"]))
            if not np.isfinite(rows).all():
                raise PackageError("Nonfinite PLY attribute in payload: " + str(path))
            quats = rows[:, rot].astype(np.float64)
            if np.any(np.sum(quats * quats, axis=1) < 1e-12):
                raise PackageError("Invalid zero rotation in PLY payload: " + str(path))
            bounds = merge_bounds(bounds, _bounds(rows[:, xyz]))
            digest.update(raw)
            if full is not None:
                full(raw)
            if quota:
                indices = sample_indices(spec["count"], quota, start, start + size)
                if len(indices):
                    # Select opaque record bytes, preserving even signed zero bits.
                    overview(np.frombuffer(raw, dtype=np.uint8).reshape(size, stride)[indices].tobytes())
                    sampled_bounds = merge_bounds(sampled_bounds, _bounds(rows[indices][:, xyz]))
                    sampled += len(indices)
        if handle.read(1):
            raise PackageError("Trailing bytes after PLY payload: " + str(path))
    if _state(path) != source["state"]:
        raise PackageError("Source changed during read: " + str(path))
    if sampled != quota:
        raise PackageError("Overview contribution does not match quota")
    return {"sha256": digest.hexdigest(), "bbox_m": bounds,
            "overview_count": sampled, "overview_bbox_m": sampled_bounds}


def quotas_for(counts, budget):
    """Balanced block strata with deterministic ID-order remainder allocation."""
    budget = min(budget, sum(counts))
    if budget < len(counts):
        raise PackageError("Overview budgets must permit at least one Gaussian from each of 28 blocks")
    quotas = [0] * len(counts)
    active = list(range(len(counts)))
    while budget:
        share = budget // len(active)
        if share == 0:
            for i in active[:budget]:
                quotas[i] += 1
            break
        for i in active:
            take = min(share, counts[i] - quotas[i])
            quotas[i] += take
            budget -= take
        active = [i for i in active if quotas[i] < counts[i]]
    return quotas


def _check_snapshots(project, snapshots):
    for path, sha in snapshots.items():
        current = {}
        read_json(local_path(project, path), current)
        if current[path] != sha:
            raise PackageError("Source JSON changed during packaging: " + str(path))


def prepare(project, manifest, candidates, chunk_bytes):
    snapshots = {}
    manifest_path, candidate_path = local_path(project, manifest), local_path(project, candidates)
    data, plan = read_json(manifest_path, snapshots), read_json(candidate_path, snapshots)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise PackageError("Manifest schema_version must be 1")
    if data.get("coordinates") != "camera_init" or data.get("units") != "m":
        raise PackageError("Manifest must use camera_init metre coordinates")
    if data.get("dataset_kind", "real") not in ("real", "synthetic"):
        raise PackageError("Manifest dataset_kind must be real or synthetic")
    up = data.get("camera_up", [0, 0, 1])
    if (not isinstance(up, list) or len(up) != 3 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in up) or
            sum(x * x for x in up) < 1e-12):
        raise PackageError("camera_up must be a finite nonzero 3-vector")
    blocks, _ = index_blocks(data, "manifest"), index_blocks(plan, "candidate plan")
    for bid, block in blocks.items():
        if not isinstance(block.get("training"), dict) or not isinstance(block.get("eval"), dict):
            raise PackageError(bid + " requires training and eval objects")
    unfinished = [bid for bid in EXPECTED_IDS if
                  blocks[bid].get("training", {}).get("status") != "trained" or
                  blocks[bid].get("eval", {}).get("status") != "evaluated"]
    if unfinished:
        raise PackageError("Refusing complete package: blocks are not trained AND evaluated: " + ", ".join(unfinished))
    sources, schema, seen_files = [], None, set()
    for bid in EXPECTED_IDS:
        block = blocks[bid]
        if block.get("coordinates") != "camera_init" or block.get("units", "m") != "m":
            raise PackageError(bid + " does not use shared camera_init metre coordinates")
        path = local_path(project, block.get("path", ""))
        if path.suffix.lower() != ".ply":
            raise PackageError(bid + " must point to a standard GS PLY")
        state = _state(path)
        identity = state[:2]
        if identity in seen_files:
            raise PackageError(bid + " reuses another block's source PLY")
        seen_files.add(identity)
        with path.open("rb") as handle:
            spec = inspect_ply(handle)
        if state[2] != len(spec["header"]) + spec["count"] * spec["stride"]:
            raise PackageError(bid + " has an incomplete or trailing PLY payload")
        if schema is not None and spec["names"] != schema:
            raise PackageError(bid + " has a different PLY schema/property order")
        schema = spec["names"]
        if "gaussians" in block["training"] and (type(block["training"]["gaussians"]) is not int or
                                                   block["training"]["gaussians"] != spec["count"]):
            raise PackageError(bid + " manifest Gaussian count differs from payload")
        # Existing reconstruction contract: training_result + export/dev metrics + held-out metrics.
        training_path = local_path(project, block["training"].get("report", path.parent.parent / "training_result.json"))
        export_path = local_path(project, path.parent / "metrics.json")
        eval_path = local_path(project, block["eval"].get("report", block["eval"].get("report_path", "")))
        trained = read_json(training_path, snapshots)
        exported = read_json(export_path, snapshots)
        evaluated = read_json(eval_path, snapshots)
        target = positive_int(block["training"].get("iterations"), bid + " training iterations")
        if trained.get("status") != "trained" or trained.get("iterations_requested") != target:
            raise PackageError(bid + " actual training result is incomplete/inconsistent")
        for report in (exported, evaluated):
            if report.get("status") != "evaluated" or type(report.get("images")) is not int or report["images"] <= 0:
                raise PackageError(bid + " actual evaluation report is incomplete")
            if type(report.get("step")) is not int or report["step"] != target - 1:
                raise PackageError(bid + " evaluation checkpoint step does not match completed training")
            # Pinned Nerfstudio exports its unchanged rigid transform as 3x4;
            # accept the equivalent homogeneous 4x4 representation as well.
            if report.get("world_parser_scale") != 1.0 or report.get("world_parser_transform") not in (
                    np.eye(4)[:3].tolist(), np.eye(4).tolist()):
                raise PackageError(bid + " evaluation does not establish unchanged metric coordinates")
        if evaluated.get("split") != block["eval"].get("split"):
            raise PackageError(bid + " evaluation split differs from manifest")
        for key in ("config", "checkpoint"):
            if not exported.get(key) or not evaluated.get(key):
                raise PackageError(bid + " evaluation lacks " + key)
            if local_path(project, exported[key]) != local_path(project, evaluated[key]):
                raise PackageError(bid + " export and evaluation use different " + key)
            if _state(local_path(project, exported[key]))[2] <= 0:
                raise PackageError(bid + " completed training artifact is empty: " + key)
        configs = trained.get("configs")
        if not isinstance(configs, list) or local_path(project, exported["config"]) not in [local_path(project, c) for c in configs]:
            raise PackageError(bid + " export config does not match actual training result")
        export = exported.get("export", {})
        if (not isinstance(export, dict) or export.get("gaussians") != spec["count"] or
                export.get("coordinates") != "camera_init" or
                export.get("format") != "binary_little_endian_float32" or
                export.get("sh_degree") != spec["sh_degree"]):
            raise PackageError(bid + " actual export report does not match GS PLY")
        source = {"id": bid, "path": path, "spec": spec, "state": state,
                  "training_report": training_path, "export_report": export_path, "eval_report": eval_path}
        source["verified"] = scan_ply(source, chunk_bytes)
        sources.append(source)
    _check_snapshots(project, snapshots)
    return data, sources, snapshots


def package_garage(output=None, *, manifest="maps/garage_manifest.json",
                   candidates="reports/lio_candidate_blocks.json", check_only=False,
                   overview_max_gaussians=DEFAULT_OVERVIEW_GAUSSIANS,
                   overview_max_bytes=VIEWER_MAX_BYTES, chunk_bytes=8 * 1024 * 1024,
                   project=PROJECT):
    project = Path(project).resolve()
    try:
        project.relative_to(PROJECT)
    except ValueError:
        raise PackageError("Project root must stay inside " + str(PROJECT))
    positive_int(chunk_bytes, "chunk_bytes")
    if chunk_bytes > MAX_CHUNK_BYTES:
        raise PackageError("chunk_bytes exceeds 64 MiB")
    positive_int(overview_max_gaussians, "overview_max_gaussians")
    positive_int(overview_max_bytes, "overview_max_bytes")
    destination = None
    if not check_only:
        if output is None:
            raise PackageError("--output is required unless --check-only")
        destination = local_path(project, output)
        if destination.exists():
            raise PackageError("Output already exists; refusing overwrite: " + str(destination))
        if not destination.parent.is_dir():
            raise PackageError("Output parent must already exist: " + str(destination.parent))
    data, sources, snapshots = prepare(project, manifest, candidates, chunk_bytes)
    names = sources[0]["spec"]["names"]
    stride = sources[0]["spec"]["stride"]
    counts = [s["spec"]["count"] for s in sources]
    total = sum(counts)
    gaussian_limit = min(total, overview_max_gaussians, VIEWER_MAX_GAUSSIANS)
    byte_limit = min(overview_max_bytes, VIEWER_MAX_BYTES)
    # Reserve the header too, conservatively using the maximum possible count's digits.
    budget = min(gaussian_limit, max(0, (byte_limit - len(ply_header(names, gaussian_limit))) // stride))
    quotas = quotas_for(counts, budget)
    overview_count = sum(quotas)
    full_bounds = None
    records = []
    for source, quota in zip(sources, quotas):
        full_bounds = merge_bounds(full_bounds, source["verified"]["bbox_m"])
        records.append({"id": source["id"], "path": source["path"].relative_to(project).as_posix(),
                        "sha256": source["verified"]["sha256"], "file_bytes": source["state"][2],
                        "gaussians": source["spec"]["count"], "overview_gaussians": quota,
                        "bbox_m": source["verified"]["bbox_m"],
                        "floor_verified": False, "calibration_verified": False,
                        "reports": {key: source[key].relative_to(project).as_posix()
                                    for key in ("training_report", "export_report", "eval_report")}})
    record = {"schema_version": 1, "status": "validated_inputs" if check_only else "complete",
              "coordinates": "camera_init", "units": "m", "floor_verified": False,
              "calibration_verified": False,
              "coverage": {"scope": "collection_trajectory", "expected_block_ids": list(EXPECTED_IDS),
                           "block_count": 28, "unobserved_building_complete": False, "note": COVERAGE_NOTE},
              "overlap": {"duplicates_retained": True, "deduplicated": False,
                          "duplicate_count": None, "note": "All source records retained; overlap duplicates are unquantified."},
              "processing": {"cpu_only": True, "chunk_bytes": chunk_bytes, "attributes_modified": False,
                             "origin_reset": False, "cropping": False, "surfaces_synthesized": False},
              "properties": [{"name": n, "type": "float32"} for n in names],
              "sources": records,
              "source_json_sha256": {p.relative_to(project).as_posix(): sha for p, sha in snapshots.items()},
              "full": {"path": "gaussians_full.ply", "gaussians": total, "bbox_m": full_bounds,
                       "file_bytes": len(ply_header(names, total)) + total * stride},
              "overview": {"path": "gaussians_overview.ply", "gaussians": overview_count,
                           "file_bytes": len(ply_header(names, overview_count)) + overview_count * stride,
                           "gaussian_budget": gaussian_limit, "file_byte_budget": byte_limit,
                           "sampling": "balanced block quotas; one midpoint per equal record-index stratum",
                           "quality_evaluated": False}}
    if check_only:
        return record

    # Only after all 28 payloads pass do we create the exclusively owned output directory.
    destination = local_path(project, destination)
    destination.mkdir(exist_ok=False)
    created = []
    def write_json(name, value):
        path = destination / name
        with path.open("xb") as handle:
            created.append(path)
            handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))

    try:
        paths = [destination / (name + ".partial") for name in ("gaussians_full.ply", "gaussians_overview.ply")]
        digests = [hashlib.sha256(), hashlib.sha256()]
        with paths[0].open("xb") as full_handle:
            created.append(paths[0])
            with paths[1].open("xb") as overview_handle:
                created.append(paths[1])
                def full_write(raw):
                    full_handle.write(raw)
                    digests[0].update(raw)
                def overview_write(raw):
                    overview_handle.write(raw)
                    digests[1].update(raw)
                full_write(ply_header(names, total))
                overview_write(ply_header(names, overview_count))
                overview_bounds = None
                for source, block_record, quota in zip(sources, records, quotas):
                    local_path(project, source["path"])
                    copied = scan_ply(source, chunk_bytes, full_write, overview_write, quota)
                    if copied["sha256"] != source["verified"]["sha256"]:
                        raise PackageError("Source PLY changed between validation and copy: " + str(source["path"]))
                    block_record["overview_bbox_m"] = copied["overview_bbox_m"]
                    overview_bounds = merge_bounds(overview_bounds, copied["overview_bbox_m"])
        _check_snapshots(project, snapshots)
        for source in sources:
            if _state(source["path"]) != source["state"]:
                raise PackageError("Source changed before publication: " + str(source["path"]))
        for path, info, digest in zip(paths, (record["full"], record["overview"]), digests):
            if path.stat().st_size != info["file_bytes"]:
                raise PackageError("Output byte count does not match declared payload")
            info["sha256"] = digest.hexdigest()
            final = destination / info["path"]
            if final.exists() or final.is_symlink():
                raise PackageError("Refusing overwrite of unexpected output: " + str(final))
            path.rename(final)
            created.remove(path)
            created.append(final)
        record["overview"]["bbox_m"] = overview_bounds
        overview_manifest = {"schema_version": 1, "coordinates": "camera_init", "units": "m",
                             "camera_up": data.get("camera_up", [0, 0, 1]),
                             "dataset_kind": data.get("dataset_kind", "real"),
                             "title": "车库 · 全28块采集轨迹 overview（楼层与标定未核验）",
                             "status": "complete", "coverage": record["coverage"],
                             "floor_verified": False, "calibration_verified": False,
                             "overlap": record["overlap"], "blocks": [{
                                 "id": "garage_overview", "label": "全部28块 · 精简总览", "floor_id": None,
                                 "floor_verified": False, "calibration_verified": False,
                                 "coordinates": "camera_init", "bbox_m": overview_bounds,
                                 "path": (destination / record["overview"]["path"]).relative_to(project).as_posix(),
                                 "source_block_ids": list(EXPECTED_IDS),
                                 "training": {"status": "trained", "gaussians": overview_count,
                                              "calibration_verified": False,
                                              "method": "Unmodified records sampled from all 28 trained exports"},
                                 "eval": {"status": "not_evaluated", "source_status": "evaluated",
                                          "note": "All source blocks evaluated; combined sampled overview has no new quality evaluation."}}]}
        write_json("overview_manifest.json", overview_manifest)
        # Completion marker is last. Partial/failed work never publishes this marker.
        write_json("packaging.json", record)
        return record
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="maps/garage_manifest.json")
    parser.add_argument("--candidates", default="reports/lio_candidate_blocks.json")
    parser.add_argument("--output", help="New visible project directory; parent must exist")
    parser.add_argument("--check-only", action="store_true", help="Validate all sources, emit JSON, write nothing")
    parser.add_argument("--overview-max-gaussians", type=int, default=DEFAULT_OVERVIEW_GAUSSIANS)
    parser.add_argument("--overview-max-bytes", type=int, default=VIEWER_MAX_BYTES)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        record = package_garage(args.output, manifest=args.manifest, candidates=args.candidates,
                                check_only=args.check_only, overview_max_gaussians=args.overview_max_gaussians,
                                overview_max_bytes=args.overview_max_bytes, chunk_bytes=args.chunk_mib * 1024 * 1024)
    except (PackageError, OSError) as exc:
        print("package_garage: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Packaging regressions using only in-memory files; no tmpdir/cache/fixtures on disk."""
import sys

sys.dont_write_bytecode = True

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_script("package_garage")
VIEWER = load_script("serve_map")
ROOT = M.PROJECT / "tests" / "memory_garage"
FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2") + tuple(
    "f_rest_%d" % i for i in range(45)) + ("opacity", "scale_0", "scale_1", "scale_2",
                                        "rot_0", "rot_1", "rot_2", "rot_3", "extra_confidence")


class MemoryFiles:
    """Small Path/handle adapter, keeping production streaming and publication intact."""
    def __init__(self, monkeypatch):
        self.files, self.versions, self.dirs, self.links = {}, {}, {ROOT}, set()
        self.clock, self.writes, self.read_sizes, self.fds = 0, [], [], {}
        self.fail_write = None
        original_fstat = VIEWER.os.fstat
        monkeypatch.setattr(VIEWER.os, "fstat", lambda fd: self.info(self.fds[fd]) if fd in self.fds else original_fstat(fd))
        methods = {"open": self.open, "stat": self.info, "exists": self.exists,
                   "is_dir": lambda path: path in self.dirs,
                   "is_symlink": lambda path: path in self.links,
                   "mkdir": self.mkdir, "rename": self.rename,
                   "unlink": self.unlink, "rmdir": self.rmdir}
        for name, operation in methods.items():
            original = getattr(Path, name)
            def dispatch(path, *args, _operation=operation, _original=original, **kwargs):
                if path == ROOT or ROOT in path.parents:
                    return _operation(path, *args, **kwargs)
                return _original(path, *args, **kwargs)
            monkeypatch.setattr(Path, name, dispatch)

    def put(self, path, raw):
        path = Path(path)
        self.clock += 1
        self.files[path] = raw
        self.versions[path] = self.clock
        self.dirs.update(path.parents)

    def json(self, path, value):
        self.put(path, json.dumps(value, allow_nan=False).encode())

    def info(self, path, **kwargs):
        if path in self.files:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_dev=1, st_ino=hash(path),
                                   st_size=len(self.files[path]), st_mtime_ns=self.versions[path],
                                   st_ctime_ns=self.versions[path])
        if path in self.dirs:
            return SimpleNamespace(st_mode=stat.S_IFDIR)
        raise FileNotFoundError(path)

    def exists(self, path):
        return path in self.files or path in self.dirs

    def open(self, path, mode="r", *args, **kwargs):
        if mode not in ("rb", "xb"):
            raise AssertionError("Unexpected mode: " + mode)
        writing = mode == "xb"
        if writing:
            if self.exists(path):
                raise FileExistsError(path)
            if path.parent not in self.dirs:
                raise FileNotFoundError(path.parent)
            self.put(path, b"")
            self.writes.append(path)
        elif path not in self.files:
            raise FileNotFoundError(path)
        fs = self
        class Handle(io.BytesIO):
            def read(self, size=-1):
                assert size >= 0, "Unbounded reads are forbidden"
                fs.read_sizes.append((path, size))
                return super().read(size)

            def write(self, raw):
                if fs.fail_write == path.name:
                    raise OSError("simulated disk full")
                return super().write(raw)

            def fileno(self):
                fd = -id(self)
                fs.fds[fd] = path
                return fd

            def close(self):
                if writing and not self.closed:
                    fs.put(path, self.getvalue())
                super().close()
        return Handle(b"" if writing else self.files[path])

    def mkdir(self, path, parents=False, exist_ok=False):
        if self.exists(path):
            if exist_ok and path in self.dirs:
                return
            raise FileExistsError(path)
        if path.parent not in self.dirs:
            raise FileNotFoundError(path.parent)
        self.dirs.add(path)
        self.writes.append(path)

    def rename(self, path, target):
        if self.exists(target):
            raise AssertionError("Publication attempted overwrite")
        self.put(target, self.files.pop(path))
        return target

    def unlink(self, path, missing_ok=False):
        if path not in self.files and not missing_ok:
            raise FileNotFoundError(path)
        self.files.pop(path, None)

    def rmdir(self, path):
        if any(p.parent == path for p in self.files) or any(p.parent == path for p in self.dirs):
            raise OSError("directory not empty")
        self.dirs.remove(path)


def model_bytes(block_index, count, names=FIELDS, type_name="float"):
    values = np.arange(count * len(names), dtype=np.float32).reshape(count, len(names)) / 128
    for name, xyz in zip(("x", "y", "z"), (71.5 + block_index * 3, -53.25 + block_index, 18.25 + block_index / 4)):
        values[:, names.index(name)] = xyz + np.arange(count, dtype=np.float32) / 16
    values[:, names.index("rot_0")] = 2.0  # Deliberately unnormalized: packaging must never normalize.
    values[:, names.index("opacity")] = -8.0  # Never filter low-opacity records.
    values[:, names.index("scale_0")] = -4.5
    values[0, names.index("nx")] = -0.0
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % count +
              "".join("property %s %s\n" % (type_name, n) for n in names) + "end_header\n").encode()
    return header + values.astype("<f4").tobytes()


@pytest.fixture
def garage(monkeypatch):
    fs = MemoryFiles(monkeypatch)
    blocks, plan = [], []
    for i, bid in enumerate(M.EXPECTED_IDS):
        count = 5 + i % 4
        path = "runs/%s/eval_dev/gaussians.ply" % bid
        config = str(ROOT / ("runs/%s/config.yml" % bid))
        checkpoint = str(ROOT / ("runs/%s/step-000011999.ckpt" % bid))
        fs.put(Path(config), b"synthetic config; never loaded")
        fs.put(Path(checkpoint), b"synthetic checkpoint; never loaded")
        block = {"id": bid, "path": path, "floor_id": "F1", "floor_verified": False,
                 "coordinates": "camera_init", "bbox_m": {"min": [0, 0, 0], "max": [0, 0, 0]},
                 "training": {"status": "trained", "iterations": 12000, "gaussians": count},
                 "eval": {"status": "evaluated", "split": "test", "report": "runs/%s/eval_test/metrics.json" % bid}}
        blocks.append(block)
        plan.append({"id": bid})
        fs.put(ROOT / path, model_bytes(i, count))
        fs.json(ROOT / ("runs/%s/training_result.json" % bid),
                {"status": "trained", "iterations_requested": 12000, "configs": [config]})
        base = {"status": "evaluated", "images": 3, "step": 11999, "config": config,
                "checkpoint": checkpoint, "world_parser_scale": 1.0, "world_parser_transform": np.eye(4)[:3].tolist()}
        fs.json(ROOT / block["eval"]["report"], dict(base, split="test"))
        fs.json(ROOT / ("runs/%s/eval_dev/metrics.json" % bid), dict(base, split="dev", export={
            "gaussians": count, "sh_degree": 3, "coordinates": "camera_init", "format": "binary_little_endian_float32"}))
    manifest = {"schema_version": 1, "coordinates": "camera_init", "units": "m", "camera_up": [0, 0, 1],
                "dataset_kind": "synthetic", "status": "building", "blocks": blocks,
                "calibration_verified": False}
    fs.json(ROOT / "maps/garage_manifest.json", manifest)
    fs.json(ROOT / "reports/lio_candidate_blocks.json", {"blocks": plan})
    return SimpleNamespace(fs=fs, manifest=manifest, plan=plan, blocks=blocks)


def run(garage, output="package", **kwargs):
    return M.package_garage(output, project=ROOT, chunk_bytes=4 * 4 * len(FIELDS),
                            overview_max_gaussians=87, **kwargs)


def decode(raw):
    handle = io.BytesIO(raw)
    spec = M.inspect_ply(handle)
    offset = len(spec["header"])
    return spec, raw[offset:], np.frombuffer(raw[offset:], dtype="<f4").reshape(spec["count"], -1)


def publish(garage):
    garage.fs.json(ROOT / "maps/garage_manifest.json", garage.manifest)


def assert_no_package(garage):
    assert ROOT / "package" not in garage.fs.dirs
    assert not garage.fs.writes


def test_lossless_full_coordinates_fields_and_every_block_contribution(garage):
    record = run(garage)
    fs = garage.fs
    spec, full_payload, full = decode(fs.files[ROOT / "package/gaussians_full.ply"])
    _, overview_payload, overview = decode(fs.files[ROOT / "package/gaussians_overview.ply"])
    original_payloads, sampled_payloads, expected_bounds = [], [], None
    for b, source in zip(garage.blocks, record["sources"]):
        raw = fs.files[ROOT / b["path"]]
        original_spec, payload, rows = decode(raw)
        original_payloads.append(payload)
        quota = source["overview_gaussians"]
        assert quota > 0
        indices = [((2 * j + 1) * len(rows)) // (2 * quota) for j in range(quota)]
        opaque = np.frombuffer(payload, dtype=np.uint8).reshape(len(rows), original_spec["stride"])
        sampled_payloads.append(opaque[indices].tobytes())
        assert source["sha256"] == hashlib.sha256(raw).hexdigest()
        assert source["overview_bbox_m"] == M._bounds(rows[indices, :3])
        expected_bounds = M.merge_bounds(expected_bounds, M._bounds(rows[:, :3]))
    assert full_payload == b"".join(original_payloads)  # Every float bit, including SH/rot/logscale/negative zero.
    assert overview_payload == b"".join(sampled_payloads)
    assert spec["names"] == FIELDS
    assert len(full) == sum(b["training"]["gaussians"] for b in garage.blocks)
    assert len(overview) == 87
    assert np.all(full[:, FIELDS.index("opacity")] == -8)
    assert full[0, 0] == 71.5 and full[0, 1] == -53.25 and full[0, 2] == 18.25
    assert record["full"]["bbox_m"] == expected_bounds
    assert record["floor_verified"] is False and record["calibration_verified"] is False
    assert record["coverage"]["unobserved_building_complete"] is False
    assert record["overlap"]["duplicates_retained"] is True
    for name in ("full", "overview"):
        raw = fs.files[ROOT / "package" / record[name]["path"]]
        assert record[name]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record[name]["file_bytes"] == len(raw)
    assert fs.writes[-1].name == "packaging.json"
    assert all(size <= 4 * 4 * len(FIELDS) for path, size in fs.read_sizes if path.suffix == ".ply")


def test_overview_is_one_all_block_viewer_asset_under_both_budgets(garage, monkeypatch):
    stride = len(FIELDS) * 4
    byte_budget = len(M.ply_header(FIELDS, 87)) + stride * 42
    record = run(garage, overview_max_bytes=byte_budget)
    data = json.loads(garage.fs.files[ROOT / "package/overview_manifest.json"])
    VIEWER.validate_manifest(copy.deepcopy(data))
    assert len(data["blocks"]) == 1 <= VIEWER.LIMITS["max_loaded_blocks"]
    b = data["blocks"][0]
    assert b["source_block_ids"] == list(M.EXPECTED_IDS)
    assert b["floor_id"] is None and b["floor_verified"] is False
    assert b["eval"]["status"] == "not_evaluated"  # No inherited score claimed for resampled map.
    assert b["bbox_m"] == record["overview"]["bbox_m"]
    monkeypatch.setattr(VIEWER, "open_local", lambda root, path: (root / path).open("rb"))
    info = VIEWER.MapStore(ROOT, "package/overview_manifest.json").preflight(b["path"])
    assert info["vertex_count"] == 42
    assert info["file_bytes"] <= byte_budget <= VIEWER.LIMITS["max_file_bytes"]
    assert info["vertex_count"] <= VIEWER.LIMITS["max_splats"]
    assert all(s["overview_gaussians"] > 0 for s in record["sources"])
    assert M.VIEWER_MAX_BYTES == VIEWER.LIMITS["max_file_bytes"]
    assert M.VIEWER_MAX_GAUSSIANS == VIEWER.LIMITS["max_splats"]


def test_sampling_is_deterministic_independent_of_chunking_and_input_order(garage):
    run(garage, "first")
    garage.manifest["blocks"].reverse()
    publish(garage)
    M.package_garage("second", project=ROOT, chunk_bytes=len(FIELDS) * 4,
                     overview_max_gaussians=87)
    for name in ("gaussians_full.ply", "gaussians_overview.ply"):
        assert garage.fs.files[ROOT / "first" / name] == garage.fs.files[ROOT / "second" / name]


def test_check_only_and_top_level_building_does_not_override_actual_evidence(garage):
    result = run(garage, check_only=True)
    assert result["status"] == "validated_inputs"
    assert len(result["sources"]) == 28
    assert_no_package(garage)


@pytest.mark.parametrize("where", ["manifest", "plan"])
@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "wrong_28"])
def test_require_predetermined_ids_not_a_count_or_status(garage, where, change):
    garage.manifest["status"] = "complete"
    target = garage.blocks if where == "manifest" else garage.plan
    if change == "missing":
        target.pop()
    elif change == "extra":
        target.append({"id": "block_028"})
    elif change == "duplicate":
        target.append(copy.deepcopy(target[0]))
    else:
        target[-1]["id"] = "other_block"
    publish(garage)
    garage.fs.json(ROOT / "reports/lio_candidate_blocks.json", {"blocks": garage.plan})
    with pytest.raises(M.PackageError, match="28|duplicate"):
        run(garage)
    assert_no_package(garage)


@pytest.mark.parametrize("key,status", [("training", "training"), ("training", "prepared"),
                                       ("training", "failed"), ("eval", "not_evaluated")])
def test_live_incomplete_status_refuses_even_when_top_says_complete(garage, key, status):
    garage.manifest["status"] = "complete"
    garage.blocks[-1][key]["status"] = status
    publish(garage)
    with pytest.raises(M.PackageError, match="block_027"):
        run(garage)
    assert_no_package(garage)


def test_equivalent_homogeneous_parser_transform_is_accepted(garage):
    for bid in M.EXPECTED_IDS:
        for split in ('dev', 'test'):
            path = ROOT / ('runs/%s/eval_%s/metrics.json' % (bid, split))
            report = json.loads(garage.fs.files[path])
            report['world_parser_transform'] = np.eye(4).tolist()
            garage.fs.json(path, report)
    assert run(garage)['status'] == 'complete'


@pytest.mark.parametrize("kind", ["missing_ply", "missing_training", "missing_eval", "missing_config",
                                   "missing_checkpoint", "empty_checkpoint", "training_status",
                                   "training_iterations", "eval_status", "images", "step", "checkpoint",
                                   "config", "export_count", "export_frame", "parser_scale", "parser_transform"])
def test_actual_disk_evidence_not_only_manifest_flags(garage, kind):
    base = ROOT / "runs/block_027"
    if kind.startswith("missing"):
        name = {"missing_ply": "eval_dev/gaussians.ply", "missing_training": "training_result.json",
                "missing_eval": "eval_test/metrics.json", "missing_config": "config.yml",
                "missing_checkpoint": "step-000011999.ckpt"}[kind]
        garage.fs.files.pop(base / name)
    elif kind == "empty_checkpoint":
        garage.fs.put(base / "step-000011999.ckpt", b"")
    else:
        filename = "training_result.json" if kind.startswith("training") else (
            "eval_dev/metrics.json" if kind.startswith("export") else "eval_test/metrics.json")
        path = base / filename
        data = json.loads(garage.fs.files[path])
        if kind in ("training_status", "eval_status"):
            data["status"] = "failed"
        elif kind == "training_iterations":
            data["iterations_requested"] = 6000
        elif kind == "export_count":
            data["export"]["gaussians"] += 1
        elif kind == "export_frame":
            data["export"]["coordinates"] = "local"
        elif kind == "parser_scale":
            data["world_parser_scale"] = 1000
        elif kind == "parser_transform":
            data["world_parser_transform"][0][3] = -71.5
        else:
            data[kind] = {"images": 0, "step": 5999, "checkpoint": str(base / "other.ckpt"),
                          "config": str(base / "other.yml")}[kind]
        garage.fs.json(path, data)
    with pytest.raises((M.PackageError, FileNotFoundError)):
        run(garage)
    assert_no_package(garage)


@pytest.mark.parametrize("kind", ["truncated", "trailing", "ascii", "big_endian", "float64", "list",
                                   "missing_sh", "extra_element", "property_order", "extra_property",
                                   "duplicate_property", "ordinary_cloud", "nan_late", "zero_rot_late"])
def test_payload_and_schema_validation_scans_unsampled_tail(garage, kind):
    path = ROOT / garage.blocks[-1]["path"]
    raw = garage.fs.files[path]
    spec, payload, rows = decode(raw)
    if kind == "truncated":
        raw = raw[:-1]
    elif kind == "trailing":
        raw += b"\x00"
    elif kind == "ascii":
        raw = raw.replace(b"binary_little_endian", b"ascii", 1)
    elif kind == "big_endian":
        raw = raw.replace(b"binary_little_endian", b"binary_big_endian", 1)
    elif kind == "float64":
        raw = raw.replace(b"property float x", b"property double x", 1)
    elif kind == "list":
        raw = raw.replace(b"property float x", b"property list uchar float x", 1)
    elif kind == "extra_element":
        raw = raw.replace(b"end_header", b"element face 0\nend_header", 1)
    elif kind == "duplicate_property":
        raw = raw.replace(b"end_header", b"property float x\nend_header", 1)
    elif kind in ("missing_sh", "property_order", "extra_property", "ordinary_cloud"):
        names = list(FIELDS)
        if kind == "missing_sh":
            names.remove("f_rest_44")
        elif kind == "property_order":
            names[0], names[1] = names[1], names[0]
        elif kind == "extra_property":
            names.append("another_field")
        else:
            names = ["x", "y", "z", "nx"]
        if kind == "ordinary_cloud":
            raw = M.ply_header(names, 1) + np.zeros(len(names), dtype="<f4").tobytes()
        else:
            raw = model_bytes(27, len(rows), tuple(names))
    else:
        rows = rows.copy()
        if kind == "nan_late":
            rows[-1, FIELDS.index("extra_confidence")] = np.nan
        else:
            rows[-1, [FIELDS.index("rot_%d" % i) for i in range(4)]] = 0
        raw = spec["header"] + rows.tobytes()
    garage.fs.put(path, raw)
    with pytest.raises(M.PackageError):
        run(garage)
    assert_no_package(garage)


@pytest.mark.parametrize("mutate", [lambda m: m.update(coordinates="local"), lambda m: m.update(units="mm"),
                                    lambda m: m.update(camera_up=[0, 0, 0]),
                                    lambda m: m["blocks"][-1].update(coordinates="local"),
                                    lambda m: m["blocks"][-1].update(units="cm")])
def test_coordinate_contract_rejects_changes(garage, mutate):
    mutate(garage.manifest)
    publish(garage)
    with pytest.raises(M.PackageError):
        run(garage)
    assert_no_package(garage)


def test_existing_output_and_escape_symlinks_never_overwrite(garage):
    garage.fs.dirs.add(ROOT / "package")
    garage.fs.put(ROOT / "package/keep.txt", b"keep")
    with pytest.raises(M.PackageError, match="exists"):
        run(garage)
    assert garage.fs.files[ROOT / "package/keep.txt"] == b"keep"
    for output in ("../escape", "/tmp/escape", ".hidden/package"):
        with pytest.raises(M.PackageError):
            run(garage, output)
    garage.fs.links.add(ROOT / "link")
    with pytest.raises(M.PackageError, match="Symlink"):
        run(garage, "link/package")
    garage.fs.links.add(ROOT / "runs/block_027")
    with pytest.raises(M.PackageError, match="Symlink"):
        run(garage, "new_package")
    assert not garage.fs.writes


def test_reused_source_cannot_impersonate_a_second_trajectory_block(garage):
    garage.blocks[-1]["path"] = garage.blocks[0]["path"]
    publish(garage)
    with pytest.raises(M.PackageError, match="reuses"):
        run(garage)
    assert_no_package(garage)


@pytest.mark.parametrize("key", ["training", "eval"])
def test_malformed_per_block_objects_refuse_without_writes(garage, key):
    garage.blocks[-1][key] = None
    publish(garage)
    with pytest.raises(M.PackageError, match="objects"):
        run(garage)
    assert_no_package(garage)


@pytest.mark.parametrize("raw", [b'{"blocks": [], "blocks": []}', b'{"bad": NaN}',
                                 b'{"bad": Infinity}', b'[]', b'{broken'])
def test_invalid_json_refuses_without_writes(garage, raw):
    garage.fs.put(ROOT / "maps/garage_manifest.json", raw)
    with pytest.raises(M.PackageError):
        run(garage)
    assert_no_package(garage)


def test_api_cannot_change_project_root_to_write_elsewhere(garage):
    with pytest.raises(M.PackageError, match="Project root"):
        M.package_garage("package", project="/tmp")
    assert_no_package(garage)


@pytest.mark.parametrize("chunk", [0, 4, M.MAX_CHUNK_BYTES + 1])
def test_invalid_or_too_small_chunk_budget_refuses_without_writes(garage, chunk):
    with pytest.raises(M.PackageError, match="chunk|Chunk|64 MiB"):
        M.package_garage("package", project=ROOT, chunk_bytes=chunk)
    assert_no_package(garage)


@pytest.mark.parametrize("budget", ["gaussians", "bytes"])
def test_insufficient_budget_for_all_blocks_refuses(garage, budget):
    args = {"overview_max_gaussians": 27} if budget == "gaussians" else {"overview_max_bytes": 100}
    with pytest.raises(M.PackageError, match="28 blocks"):
        M.package_garage("package", project=ROOT, **args)
    assert_no_package(garage)


def test_source_digest_change_during_copy_leaves_no_completion(garage, monkeypatch):
    original_scan = M.scan_ply
    def mutate_then_scan(source, chunk_bytes, full=None, overview=None, quota=0):
        if full is not None and source["id"] == "block_027":
            path = source["path"]
            raw = bytearray(garage.fs.files[path])
            raw[-4:] = np.array([42], dtype="<f4").tobytes()
            # Simulate a same-size same-stat mutation; the second-pass SHA must catch it.
            garage.fs.files[path] = bytes(raw)
        return original_scan(source, chunk_bytes, full, overview, quota)
    monkeypatch.setattr(M, "scan_ply", mutate_then_scan)
    with pytest.raises(M.PackageError, match="changed between"):
        run(garage)
    assert ROOT / "package" not in garage.fs.dirs
    assert not any(ROOT / "package" in path.parents for path in garage.fs.files)


def test_source_json_change_during_copy_refuses_publication(garage, monkeypatch):
    original_scan = M.scan_ply
    def change_json(source, chunk_bytes, full=None, overview=None, quota=0):
        result = original_scan(source, chunk_bytes, full, overview, quota)
        if full is not None and source["id"] == "block_027":
            garage.manifest["status"] = "changed"
            publish(garage)
        return result
    monkeypatch.setattr(M, "scan_ply", change_json)
    with pytest.raises(M.PackageError, match="JSON changed"):
        run(garage)
    assert ROOT / "package" not in garage.fs.dirs


def test_failed_write_does_not_publish_complete(garage):
    garage.fs.fail_write = "gaussians_overview.ply.partial"
    with pytest.raises(OSError, match="disk full"):
        run(garage)
    assert ROOT / "package" not in garage.fs.dirs


def test_float32_alias_and_full_overview_when_under_budget(garage):
    for i, b in enumerate(garage.blocks):
        garage.fs.put(ROOT / b["path"], model_bytes(i, b["training"]["gaussians"], type_name="float32"))
    record = M.package_garage("package", project=ROOT)
    assert record["overview"]["gaussians"] == record["full"]["gaussians"]
    assert decode(garage.fs.files[ROOT / "package/gaussians_full.ply"])[1] == decode(
        garage.fs.files[ROOT / "package/gaussians_overview.ply"])[1]


def test_balanced_allocation_handles_tiny_blocks_and_exact_stratum_samples():
    counts = [1, 2, 3] + [100] * 25
    quotas = M.quotas_for(counts, 80)
    assert sum(quotas) == 80 and all(0 < q <= n for q, n in zip(quotas, counts))
    for n in range(1, 35):
        for q in range(1, n + 1):
            for chunk in (1, 3, 8):
                actual = [i + start for start in range(0, n, chunk)
                          for i in M.sample_indices(n, q, start, min(n, start + chunk)).tolist()]
                assert actual == [((2 * j + 1) * n) // (2 * q) for j in range(q)]


def test_cli_refusal_is_nonzero_without_writes(garage, monkeypatch, capsys):
    garage.blocks[-1]["training"]["status"] = "training"
    publish(garage)
    original = M.package_garage
    monkeypatch.setattr(M, "package_garage", lambda *a, **kw: original(*a, project=ROOT, **kw))
    assert M.main(["--check-only"]) == 2
    assert "block_027" in capsys.readouterr().err
    assert_no_package(garage)

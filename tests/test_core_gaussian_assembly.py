"""CPU regression fixtures use new reports/*v2* directories, removed afterward."""
import sys

sys.dont_write_bytecode = True

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Loading the assembler confines caches before either scientific import below.
spec = importlib.util.spec_from_file_location("core_assembly", ROOT / "scripts/assemble_core_gaussians.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import numpy as np
import pytest

viewer_spec = importlib.util.spec_from_file_location("core_viewer", ROOT / "scripts/serve_map.py")
VIEWER = importlib.util.module_from_spec(viewer_spec)
viewer_spec.loader.exec_module(VIEWER)

FIELDS = ("z", "nx", "opacity", "x", "f_dc_0", "y", "f_dc_1", "f_dc_2") + tuple(
    f"f_rest_{i}" for i in range(45)) + ("scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
                                         "rot_2", "rot_3", "confidence")
STRIDE = 4 * len(FIELDS)
BASE_GROUP = 1_720_268_000_000_000_001  # Exact int64, deliberately above float64 integer precision.


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def frame(name, group, center, camera="camera_3"):
    transform = np.eye(4)
    transform[:3, 3] = center
    return {"file_path": name, "group_id": str(group), "camera_id": camera,
            "transform_matrix": transform.tolist()}


def gaussian_rows(xyz, tag):
    rows = (np.arange(len(xyz) * len(FIELDS), dtype=np.float32).reshape(len(xyz), -1) + tag) / 64
    for axis, name in enumerate(("x", "y", "z")):
        rows[:, FIELDS.index(name)] = xyz[:, axis]
    for i in range(4):
        rows[:, FIELDS.index(f"rot_{i}")] = 1. if i == 0 else 0.
    rows[:, FIELDS.index("rot_2")] = -0.
    rows[:, FIELDS.index("f_dc_1")] = -0.
    rows[:, FIELDS.index("opacity")] = -20.  # Opacity must never become a selection gate.
    for i in range(3):
        rows[:, FIELDS.index(f"scale_{i}")] = -4.5 - i
    return rows.astype("<f4")


def put_ply(path, rows, names=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.ply_header(names, len(rows)) + rows.astype("<f4").tobytes())


def decode(path):
    handle = io.BytesIO(path.read_bytes())
    spec = M.inspect_ply(handle)
    raw = handle.read()
    return spec, raw, np.frombuffer(raw, dtype="<f4").reshape(spec["count"], -1)


@pytest.fixture
def garage():
    with tempfile.TemporaryDirectory(prefix="core_assembly_test_v2_", dir=ROOT / "reports") as directory:
        project = Path(directory)
        centers = np.asarray([[100.25 + 10 * (i % 7), -60.5 + 3 * (i % 7), -3.25 + 5 * (i // 7)]
                              for i in range(28)], dtype=np.float64)
        blocks, rows_by_id, paths, counts = [], {}, {}, []
        for i, bid in enumerate(M.EXPECTED_IDS):
            group, count = BASE_GROUP + 100 * i, 2 + i % 4
            dataset = project / "data/garage_surface_v1" / bid
            metadata = {"orientation_override": "none", "frames": [
                frame("train.png", group, centers[i]),
                frame("side.png", group, [999, 888, 777], "camera_2"),
                frame("dev.png", group + 10, centers[(i + 1) % 28]),
                frame("test.png", group + 20, centers[(i + 2) % 28])],
                "train_filenames": ["train.png", "side.png"],
                "val_filenames": ["dev.png"], "test_filenames": ["test.png"]}
            put_json(dataset / "transforms.json", metadata)
            np.savez(dataset / "lidar_reference.npz", xyz=np.stack([centers[i], centers[i]]))
            good = centers[i] + np.arange(count)[:, None] * np.asarray([.02, .01, .005])
            xyz = np.vstack([good, centers[i] + [2.5, 0, 0], centers[(i + 1) % 28]])
            rows = gaussian_rows(xyz, 1000 * i)
            relative = f"runs/garage_surface_v1/{bid}/eval_dev/gaussians.ply"
            put_ply(project / relative, rows)
            rows_by_id[bid], paths[bid] = rows, relative
            counts.append(count)
            blocks.append({"id": bid, "coordinates": "camera_init", "floor_id": "unverified",
                           "floor_verified": False, "path": relative,
                           "training": {"status": "trained"}, "eval": {"status": "evaluated", "psnr_db": 99}})
        ledger = {"schema_version": 1, "coordinates": "camera_init", "units": "m", "camera_up": [0, 0, 1],
                  "dataset_kind": "synthetic", "calibration_verified": False, "blocks": blocks}
        put_json(project / "maps/garage_manifest.json", ledger)
        plan = M.default_source_plan()
        put_json(project / "plan.json", plan)
        yield SimpleNamespace(project=project, centers=centers, rows=rows_by_id, paths=paths,
                              counts=counts, ledger=ledger, plan=plan)


def run(garage, output="candidate", **kwargs):
    defaults = {"project": garage.project, "chunk_bytes": STRIDE * 2 + 11,
                "overview_max_gaussians": 57, "source_plan": "plan.json"}
    defaults.update(kwargs)
    return M.assemble(output, **defaults)


def install_joint(garage, explicit=False):
    cores = [f"block_{i:03d}" for i in (14, 15, 16)]
    # Deliberately interleave descending owners and unequal core row counts.
    rows = []
    for index in range(max(garage.counts[14:17])):
        for bid in reversed(cores):
            if index < garage.counts[M.EXPECTED_IDS.index(bid)]:
                rows.append(garage.rows[bid][index])
    rows = np.vstack([rows, garage.rows[cores[0]][-2:]])
    relative = "runs/joint_overlap_v1/eval_dev/gaussians.ply"
    put_ply(garage.project / relative, rows)
    entry = {"id": "joint_014_016", "path": relative, "core_ids": cores}
    if explicit:
        entry["measured_reference"] = "data/joint_overlap_v1/lidar_reference.npz"
        reference = garage.project / entry["measured_reference"]
        reference.parent.mkdir(parents=True)
        np.savez(reference, xyz=garage.centers[14:17])
        # An explicit joint reference must not require these original NPZs.
        for bid in cores:
            (garage.project / f"data/garage_surface_v1/{bid}/lidar_reference.npz").unlink()
    garage.plan["sources"] = [s for s in garage.plan["sources"] if s["id"] not in cores] + [entry]
    put_json(garage.project / "plan.json", garage.plan)
    return entry, rows


def verify_trace(garage, report, artifact):
    spec, raw, rows = decode(garage.project / artifact["path"])
    trace = np.load(garage.project / artifact["provenance"]["path"], allow_pickle=False)
    assert spec["names"] == FIELDS
    assert len(trace) == len(rows) == artifact["gaussians"]
    assert trace.dtype == M.PROVENANCE
    for index, origin in enumerate(trace):
        source = report["sources"][int(origin["source_index"])]
        source_spec, source_raw, _ = decode(garage.project / source["path"])
        start = int(origin["source_record_index"]) * source_spec["stride"]
        assert raw[index * STRIDE:(index + 1) * STRIDE] == source_raw[start:start + STRIDE]
        assert M.EXPECTED_IDS[int(origin["assigned_core"])] in source["core_ids"]
        assert int(origin["nearest_train_group"]) in report["ownership"]["group_ids"]
    for value in (artifact, artifact["provenance"]):
        path = garage.project / value["path"]
        assert value["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert value["file_bytes"] == path.stat().st_size
    return trace, raw


def test_complete_default_sources_preserve_every_selected_byte_and_report_losses(garage):
    before = {p: p.read_bytes() for p in garage.project.rglob("*") if p.is_file()}
    report = run(garage, source_plan=None, chunk_bytes=STRIDE, core_file_max_gaussians=2)
    assert report["status"] == M.STATUS
    assert report["coverage"]["artifact_coverage_complete"] is True
    assert report["coverage"]["nonzero_core_count"] == 28
    assert report["coverage"]["holes_assessed"] is False
    assert all(report[key] is False for key in M.FALSE_FLAGS)
    assert report["retained_gaussians"] == sum(garage.counts)
    assert report["candidate_gaussians"] == sum(garage.counts) + 56
    expected = b"".join(garage.rows[bid][:garage.counts[i]].tobytes() for i, bid in enumerate(M.EXPECTED_IDS))
    trace, raw = verify_trace(garage, report, report["full"])
    assert raw == expected
    assert trace["measured_support_checked"].all()
    assert np.all(trace["measured_support_distance_m"] <= .2)
    assert np.unique(trace["assigned_core"]).tolist() == list(range(28))
    assert len({(int(r["source_index"]), int(r["source_record_index"])) for r in trace}) == len(trace)
    overview_trace, _ = verify_trace(garage, report, report["overview"])
    assert len(overview_trace) == 57
    for core, item in enumerate(report["cores"]):
        assert item["retained_gaussians"] == garage.counts[core]
        assert item["offered_gaussians"] == garage.counts[core] + 1
        assert item["support_rejected_gaussians"] == 1
        assert item["footprint"]["lost_cells"] == 1
        assert item["footprint"]["axes"] == "world XYZ"
        assert item["offered_bbox_m"]["max"][0] > item["retained_bbox_m"]["max"][0]
        assert np.count_nonzero(overview_trace["assigned_core"] == core) == item["overview_gaussians"] > 0
        parts = []
        for artifact in item["artifacts"]:
            core_trace, part = verify_trace(garage, report, artifact)
            assert np.all(core_trace["assigned_core"] == core)
            assert artifact["gaussians"] <= 2
            parts.append(part)
        assert b"".join(parts) == garage.rows[item["id"]][:garage.counts[core]].tobytes()
    assert all(p.read_bytes() == raw for p, raw in before.items())
    for name in ("manifest.json", "overview_manifest.json"):
        manifest = VIEWER.validate_manifest(json.loads((garage.project / "candidate" / name).read_text()))
        assert manifest["status"] == M.STATUS
        for block in manifest["blocks"]:
            assert block["floor_id"] is None
            assert block["floor_verified"] is block["calibration_verified"] is False
            assert block["eval"]["status"] == "not_evaluated"
            assert "psnr_db" not in block["eval"]
            assert block["training"]["gaussians"] <= M.CORE_FILE_LIMIT


def test_same_xy_different_z_and_heldout_poses_do_not_steal_ownership(garage):
    ownership, report = M.canonical_ownership(garage.project, M.InputGuard(garage.project))
    assert np.array_equal(garage.centers[0, :2], garage.centers[7, :2])
    cores, groups = ownership.assign(garage.centers[[0, 7]], 1)
    assert cores.tolist() == [0, 7]
    assert groups.tolist() == [BASE_GROUP, BASE_GROUP + 700]
    assert report["group_ids"] == [BASE_GROUP + 100 * i for i in range(28)]
    assert report["held_out_poses_used"] is False
    assert not np.any(ownership.positions == 999)
    # Equidistant between these floors is stable across chunks and workers.
    midpoint = (garage.centers[0] + garage.centers[7]) / 2
    assert ownership.assign(midpoint[None], 2)[0].tolist() == [0]


def test_duplicate_groups_use_existing_global_temporal_center_resolver(garage):
    mapping = {}
    for i, groups in ((0, [0, 1, 2, 3, 4]), (1, [3, 4, 5, 6, 7])):
        frames = [frame(f"{g}.png", BASE_GROUP + g, [100 + g, 0, 2]) for g in groups]
        meta = {"frames": frames, "train_filenames": [f["file_path"] for f in frames]}
        bid = M.EXPECTED_IDS[i]
        mapping[bid] = meta
        put_json(garage.project / f"data/garage_surface_v1/{bid}/transforms.json", meta)
    for bid in M.EXPECTED_IDS[2:]:
        mapping[bid] = json.loads((garage.project / f"data/garage_surface_v1/{bid}/transforms.json").read_text())
    expected_positions, expected_owners, expected_groups = M.build_ownership(mapping)
    ownership, report = M.canonical_ownership(garage.project, M.InputGuard(garage.project))
    assert report["group_ids"] == expected_groups
    assert report["group_owner_core_ids"] == expected_owners.tolist()
    assert report["group_positions_m"] == expected_positions.tolist()
    owners, _ = ownership.assign(np.asarray([[103, 0, 2], [104, 0, 2]]), 1)
    assert owners.tolist() == [0, 1]  # Ownership changes inside the real overlap.


@pytest.mark.parametrize("explicit", [False, True])
def test_joint_pool_borrows_existing_records_into_each_covered_core(garage, explicit):
    entry, rows = install_joint(garage, explicit)
    report = run(garage)
    joint = next(s for s in report["sources"] if s["id"] == entry["id"])
    source_index = joint["source_index"]
    trace, _ = verify_trace(garage, report, report["full"])
    joint_trace = trace[trace["source_index"] == source_index]
    assert set(joint_trace["assigned_core"].tolist()) == {14, 15, 16}
    assert set(joint_trace["source_record_index"].tolist()) == set(range(sum(garage.counts[14:17]))) | {len(rows) - 1}
    # Last joint row was originally another block's Gaussian, now owned by 015.
    assert joint_trace[-1]["assigned_core"] == 15
    assert joint_trace[-1]["source_record_index"] == len(rows) - 1
    assert joint["retained_core_counts"]["block_015"] == garage.counts[15] + 1
    support = joint["measured_support"]
    assert support["explicit_joint_reference"] is explicit
    assert support["unique_xyz_rows"] == 3
    assert support["input_rows"] == (3 if explicit else 6)
    assert support["independent_ground_truth"] is False
    assert len(report["sources"]) == 26
    assert len({c for s in report["sources"] for c in s["core_ids"]}) == 28
    normalized = json.loads((garage.project / "candidate/source_plan.json").read_text())
    assert set(normalized) == {"schema_version", "coordinates", "units", "sources"}
    assert all(set(s) == {"id", "path", "core_ids", "measured_reference"} for s in normalized["sources"])


def test_chunk_sizes_and_plan_order_do_not_change_any_ply_or_provenance_bytes(garage):
    install_joint(garage)
    prior = None
    for number, chunk in enumerate((STRIDE, 2 * STRIDE + 17, 5 * STRIDE, 7 * STRIDE + 3)):
        garage.plan["sources"].reverse()
        put_json(garage.project / "plan.json", garage.plan)
        name = f"candidate_{number}"
        report = run(garage, output=name, chunk_bytes=chunk, core_file_max_gaussians=3, workers=1 + number % 2)
        files = {p.relative_to(garage.project / name).as_posix(): p.read_bytes()
                 for p in (garage.project / name).rglob("*") if p.suffix in (".ply", ".npy")}
        if prior is not None:
            assert files == prior
        prior = files
        overview_trace, _ = verify_trace(garage, report, report["overview"])
        full_trace, _ = verify_trace(garage, report, report["full"])
        expected = []
        for core, item in enumerate(report["cores"]):
            origins = full_trace[full_trace["assigned_core"] == core]
            n, q = len(origins), item["overview_gaussians"]
            expected.extend(origins[((2 * j + 1) * n) // (2 * q)] for j in range(q))
        expected = np.asarray(sorted(expected, key=lambda r: (r["source_index"], r["source_record_index"])), dtype=M.PROVENANCE)
        assert overview_trace.tobytes() == expected.tobytes()


@pytest.mark.parametrize("count,quota", [(1, 1), (7, 3), (31, 13), (53, 28), (2**54 + 17, 29)])
def test_integer_sampler_handles_unequal_chunks_without_float_rounding(count, quota):
    cuts = sorted({0, 1, count // 7, count // 2 + 1, count - 1, count})
    actual = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        actual.extend((M.sample_indices(count, quota, start, end) + start).tolist())
    assert actual == [((2 * j + 1) * count) // (2 * quota) for j in range(quota)]


@pytest.mark.parametrize("change,match", [
    ("missing", "Missing core"), ("overlap", "Duplicate core"), ("repeated_core", "distinct canonical"),
    ("unknown", "distinct canonical"), ("duplicate_id", "duplicate source ID"),
    ("duplicate_file", "distinct standard PLY"), ("outside", "inside project"),
    ("traversal", "traversal"), ("typo", "unknown field"), ("bad_world", "camera_init/m"),
])
def test_bad_plans_fail_closed_before_publication(garage, change, match):
    plan = garage.plan
    if change == "missing":
        plan["sources"].pop()
    elif change == "overlap":
        plan["sources"][0]["core_ids"].append("block_001")
    elif change == "repeated_core":
        plan["sources"][0]["core_ids"].append("block_000")
    elif change == "unknown":
        plan["sources"][0]["core_ids"] = ["block_028"]
    elif change == "duplicate_id":
        plan["sources"][1]["id"] = "block_000"
    elif change == "duplicate_file":
        plan["sources"][1]["path"] = plan["sources"][0]["path"]
    elif change == "outside":
        plan["sources"][0]["path"] = "/tmp/external_gaussians.ply"
    elif change == "traversal":
        plan["sources"][0]["path"] = "../other.ply"
    elif change == "typo":
        plan["sources"][0]["core_id"] = ["block_000"]
    else:
        plan["coordinates"] = "block_local"
    put_json(garage.project / "plan.json", plan)
    with pytest.raises(ValueError, match=match):
        run(garage)
    assert not (garage.project / "candidate").exists()


def test_zero_core_fails_even_if_joint_source_has_other_retained_cores(garage):
    entry, rows = install_joint(garage)
    xyz = rows[:, [FIELDS.index(n) for n in ("x", "y", "z")]]
    rows = rows[np.linalg.norm(xyz - garage.centers[16], axis=1) > .2]
    put_ply(garage.project / entry["path"], rows)
    with pytest.raises(M.AssemblyError, match="zero retained.*block_016"):
        run(garage)
    assert not (garage.project / "candidate").exists()


def test_partial_batch_progress_json_is_not_an_assembly_source_plan(garage):
    partial = {"status": "partial", "sources": garage.plan["sources"][:-1]}
    put_json(garage.project / "plan.json", partial)
    with pytest.raises(M.AssemblyError, match="requires exactly"):
        run(garage)
    assert not (garage.project / "candidate").exists()


def test_initial_mixed_plan_helper_uses_the_batch_contract(garage, monkeypatch, capsys):
    monkeypatch.setattr(M, "ROOT", garage.project)
    assert M.main(["--write-initial-source-plan", "proposed_initial.json"]) == 0
    data = json.loads((garage.project / "proposed_initial.json").read_text())
    assert set(data) == {"schema_version", "coordinates", "units", "sources"}
    assert len(data["sources"]) == 26
    assert all(set(s) == {"id", "path", "core_ids", "measured_reference"} for s in data["sources"])
    cores = [core for s in data["sources"] for core in s["core_ids"]]
    assert len(cores) == 28 and set(cores) == set(M.EXPECTED_IDS)
    joint = next(s for s in data["sources"] if len(s["core_ids"]) > 1)
    assert joint["path"] == "runs/joint_overlap_v1/eval_dev/gaussians.ply"
    assert joint["measured_reference"] == "data/joint_overlap_v1/lidar_reference.npz"
    assert joint["core_ids"] == ["block_014", "block_015", "block_016"]
    output = json.loads(capsys.readouterr().out)
    assert output["inputs_verified"] is output["assembly_started"] is False
    assert not (garage.project / "candidate").exists()


@pytest.mark.parametrize("column", ["x", "f_dc_0", "f_rest_44", "opacity", "scale_2", "confidence"])
def test_nonfinite_attributes_in_discarded_tail_are_still_rejected(garage, column):
    rows = garage.rows["block_027"].copy()
    rows[-1, FIELDS.index(column)] = np.nan
    put_ply(garage.project / garage.paths["block_027"], rows)
    with pytest.raises(M.AssemblyError, match="Nonfinite PLY"):
        run(garage, chunk_bytes=STRIDE)
    assert not (garage.project / "candidate").exists()


@pytest.mark.parametrize("kind", ["zero_rotation", "nonunit_rotation", "truncated", "trailing", "ascii",
                                  "missing_sh", "extra_dc", "order"])
def test_invalid_rotations_payload_and_schema_fail_closed(garage, kind):
    path = garage.project / garage.paths["block_027"]
    rows = garage.rows["block_027"].copy()
    if kind.endswith("rotation"):
        rows[-1, FIELDS.index("rot_0")] = 0. if kind == "zero_rotation" else 2.
        put_ply(path, rows)
    elif kind == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif kind == "trailing":
        path.write_bytes(path.read_bytes() + b"x")
    elif kind == "ascii":
        path.write_bytes(path.read_bytes().replace(b"binary_little_endian", b"ascii"))
    elif kind == "missing_sh":
        indices = [i for i, name in enumerate(FIELDS) if name != "f_rest_44"]
        put_ply(path, rows[:, indices], tuple(FIELDS[i] for i in indices))
    elif kind == "extra_dc":
        put_ply(path, np.column_stack([rows, np.zeros(len(rows))]), FIELDS + ("f_dc_3",))
    else:
        put_ply(path, rows[:, ::-1], FIELDS[::-1])
    with pytest.raises(ValueError):
        run(garage, chunk_bytes=STRIDE)
    assert not (garage.project / "candidate").exists()


@pytest.mark.parametrize("kind", ["empty", "nan", "shape", "object", "missing_xyz", "lying_header"])
def test_invalid_measured_reference_is_never_assumed_support(garage, kind):
    path = garage.project / "data/garage_surface_v1/block_000/lidar_reference.npz"
    if kind == "lying_header":
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False,
                                                      "shape": (M.DEFAULT_REFERENCE_POINTS + 1, 3)})
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xyz.npy", header.getvalue())
    elif kind == "missing_xyz":
        np.savez(path, wrong_name=np.zeros((1, 3)))
    else:
        xyz = {"empty": np.empty((0, 3)), "nan": np.full((1, 3), np.nan),
               "shape": np.zeros((2, 2)), "object": np.asarray([["0", "0", "0"]], dtype=object)}[kind]
        np.savez(path, xyz=xyz)
    with pytest.raises((ValueError, KeyError)):
        run(garage)
    assert not (garage.project / "candidate").exists()


@pytest.mark.parametrize("hide_stat", [False, True])
def test_source_stat_and_sha_both_guard_against_mutation_between_passes(garage, monkeypatch, hide_stat):
    path = garage.project / garage.paths["block_000"]
    original_state, original_scan, original_stat = M._state(path), M.scan_source, M._state
    calls = 0
    if hide_stat:
        monkeypatch.setattr(M, "_state", lambda p: original_state if p == path else original_stat(p))
    def scan(*args, **kwargs):
        nonlocal calls
        result = original_scan(*args, **kwargs)
        calls += 1
        if calls == 28:
            rows = garage.rows["block_000"].copy()
            rows[0, FIELDS.index("f_dc_0")] += 1.
            put_ply(path, rows)
        return result
    monkeypatch.setattr(M, "scan_source", scan)
    with pytest.raises(M.AssemblyError, match="Source changed"):
        run(garage)
    assert not (garage.project / "candidate").exists()
    assert calls > 28 if hide_stat else calls == 28


@pytest.mark.parametrize("target", ["plan.json", "maps/garage_manifest.json",
                                     "data/garage_surface_v1/block_000/transforms.json",
                                     "data/garage_surface_v1/block_000/lidar_reference.npz"])
def test_late_metadata_or_reference_changes_remove_already_written_manifests(garage, monkeypatch, target):
    original = M.write_json
    def write(path, value):
        original(path, value)
        if path.name == "overview_manifest.json":
            changed = garage.project / target
            changed.write_bytes(changed.read_bytes() + b" ")
    monkeypatch.setattr(M, "write_json", write)
    with pytest.raises(M.AssemblyError, match="Source changed"):
        run(garage)
    assert not (garage.project / "candidate").exists()


def test_disk_failure_removes_partial_artifacts_without_touching_sources(garage, monkeypatch):
    original = M.Artifact.append
    before = (garage.project / garage.paths["block_000"]).read_bytes()
    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.path.name == "gaussians_overview.ply":
            raise OSError("simulated disk full")
    monkeypatch.setattr(M.Artifact, "append", fail)
    with pytest.raises(OSError, match="disk full"):
        run(garage)
    assert not (garage.project / "candidate").exists()
    assert (garage.project / garage.paths["block_000"]).read_bytes() == before


def test_support_disabled_is_explicit_and_still_enforces_single_owner(garage):
    for path in (garage.project / "data").rglob("*.npz"):
        path.unlink()
    report = run(garage, support_radius_m=None)
    trace, _ = verify_trace(garage, report, report["full"])
    assert not trace["measured_support_checked"].any()
    assert np.all(trace["measured_support_distance_m"] == -1)
    assert report["processing"]["measured_support_checked"] is False
    assert all(s["ownership_rejected_gaussians"] == 1 for s in report["sources"])
    assert report["retained_gaussians"] == sum(garage.counts) + 28
    assert all(s["measured_support"] is None for s in report["sources"])
    assert all(report[key] is False for key in M.FALSE_FLAGS)


@pytest.mark.parametrize("group", [1.25, True, "1.2", 2**63])
def test_group_ids_never_pass_through_float_or_truncated_int_conversion(garage, group):
    path = garage.project / "data/garage_surface_v1/block_000/transforms.json"
    data = json.loads(path.read_text())
    data["frames"][0]["group_id"] = group
    put_json(path, data)
    with pytest.raises(M.AssemblyError, match="stable int64"):
        run(garage)


def test_global_heldout_group_leakage_is_rejected(garage):
    path = garage.project / "data/garage_surface_v1/block_001/transforms.json"
    data = json.loads(path.read_text())
    data["frames"][2]["group_id"] = str(BASE_GROUP)
    put_json(path, data)
    with pytest.raises(M.AssemblyError, match="held-out group leakage"):
        run(garage)


def test_footprint_truncation_is_disclosed_without_claiming_no_holes(garage, monkeypatch):
    monkeypatch.setattr(M, "FOOTPRINT_CELL_LIMIT", 1)
    report = run(garage)
    for core in report["cores"]:
        assert core["footprint"]["offered_truncated"] is True
        assert core["footprint"]["counts_exact"] is False
        assert core["footprint"]["lost_cells"] is None
        assert core["footprint"]["surface_coverage_verified"] is False


def test_bounded_ply_reads_use_only_complete_record_chunks(garage, monkeypatch):
    original_open, sizes = Path.open, []
    source_paths = {garage.project / path for path in garage.paths.values()}
    class Reader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def readline(self, size):
            assert 0 < size <= 4096
            return self.handle.readline(size)
        def read(self, size):
            assert size > 0
            sizes.append(size)
            return self.handle.read(size)
    def open_path(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        return Reader(handle) if path in source_paths and mode == "rb" else handle
    monkeypatch.setattr(Path, "open", open_path)
    run(garage, chunk_bytes=STRIDE * 3 + 19)
    assert sizes and max(sizes) == STRIDE * 3
    assert all(size == 1 or size % STRIDE == 0 for size in sizes)


def test_existing_output_and_plan_helper_are_exclusive_and_cli_fails_closed(garage, monkeypatch, capsys):
    existing = garage.project / "candidate"
    existing.mkdir()
    (existing / "sentinel").write_bytes(b"keep")
    with pytest.raises(M.AssemblyError, match="NEW directory"):
        run(garage)
    assert (existing / "sentinel").read_bytes() == b"keep"
    monkeypatch.setattr(M, "ROOT", garage.project)
    assert M.main(["--write-default-source-plan", "generated.json"]) == 0
    expected = (garage.project / "generated.json").read_bytes()
    assert json.loads(expected) == M.default_source_plan()
    assert M.main(["--write-default-source-plan", "generated.json"]) == 2
    assert (garage.project / "generated.json").read_bytes() == expected
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "incomplete" and error["artifact_coverage_complete"] is False


@pytest.mark.parametrize("kwargs", [{"overview_max_gaussians": 27}, {"overview_max_gaussians": 1_200_001},
                                    {"core_file_max_gaussians": 1_500_001}, {"chunk_bytes": STRIDE - 1},
                                    {"chunk_bytes": 64 * 1024**2 + 1}, {"support_radius_m": float("nan")},
                                    {"support_radius_m": 0.}, {"reference_max_points": 1}, {"workers": 9}])
def test_budget_and_radius_errors_publish_nothing(garage, kwargs):
    with pytest.raises(ValueError):
        run(garage, **kwargs)
    assert not (garage.project / "candidate").exists()


def test_actual_28_block_training_topology_matches_resolver_read_only():
    paths = [ROOT / f"data/garage_surface_v1/{bid}/transforms.json" for bid in M.EXPECTED_IDS]
    if not all(path.is_file() for path in paths):
        pytest.skip("Actual project metadata not present; synthetic topology tests still run")
    mapping = {bid: M.read_json(path) for bid, path in zip(M.EXPECTED_IDS, paths)}
    positions, owners, groups = M.build_ownership(mapping)
    guard = M.InputGuard(ROOT)
    ownership, report = M.canonical_ownership(ROOT, guard)
    assert report["group_ids"] == groups
    assert report["group_owner_core_ids"] == owners.tolist()
    assert report["group_positions_m"] == positions.tolist()
    assert set(owners) == set(M.EXPECTED_IDS)
    assert ownership.groups.dtype == np.dtype("<i8")
    assert report["held_out_poses_used"] is False
    guard.check()


def test_no_ownership_preserves_all_records_and_source_floor_membership(garage):
    for i, block in enumerate(garage.ledger["blocks"]):
        block["floor_id"] = f"F{i // 7 + 1}"
    put_json(garage.project / "maps/garage_manifest.json", garage.ledger)
    reports = [run(garage, output=f"none_{i}", ownership_mode="none", support_radius_m=None,
                   chunk_bytes=chunk) for i, chunk in enumerate((STRIDE, STRIDE * 7))]
    expected = b"".join(garage.rows[bid].tobytes() for bid in M.EXPECTED_IDS)
    for report in reports:
        trace, raw = verify_trace(garage, report, report["full"])
        assert raw == expected
        assert report["candidate_gaussians"] == report["retained_gaussians"]
        assert report["ownership"]["mode"] == "none"
        assert report["ownership"]["cross_source_reassignment"] is False
        assert all(s["ownership_rejected_gaussians"] == 0 for s in report["sources"])
        for source in report["sources"]:
            indices = trace[trace["source_index"] == source["source_index"]]
            assert indices["source_record_index"].tolist() == list(range(len(garage.rows[source["id"]])))
            assert np.all(indices["assigned_core"] == M.EXPECTED_IDS.index(source["id"]))
    for suffix in ("gaussians_full.ply", "full_provenance.npy", "gaussians_overview.ply"):
        assert (garage.project / "none_0" / suffix).read_bytes() == (garage.project / "none_1" / suffix).read_bytes()
    manifest = json.loads((garage.project / "none_0/manifest.json").read_text())
    assert [b["floor_id"] for b in manifest["blocks"]] == [b["floor_id"] for b in garage.ledger["blocks"]]


def test_no_ownership_support_filter_is_independent_and_explicit(garage):
    row = garage.rows["block_000"][-1]
    xyz = row[[FIELDS.index(n) for n in ("x", "y", "z")]]
    np.savez(garage.project / "data/garage_surface_v1/block_000/lidar_reference.npz",
             xyz=np.vstack([garage.centers[0], xyz]))
    old = run(garage, output="old")
    new = run(garage, output="new", ownership_mode="none")
    assert new["retained_gaussians"] == old["retained_gaussians"] + 1
    assert new["sources"][0]["support_rejected_gaussians"] == 1
    assert new["sources"][0]["ownership_rejected_gaussians"] == 0


def test_subset_is_not_advertised_as_full_coverage_and_keeps_reusable_plan(garage):
    selected = ["block_015", "block_016"]
    report = run(garage, ownership_mode="none", support_radius_m=None, core_ids=selected)
    assert report["coverage"]["artifact_coverage_complete"] is False
    assert report["coverage"]["selected_core_coverage_complete"] is True
    assert report["coverage"]["nonzero_core_count"] == 2
    assert [s["id"] for s in report["sources"]] == selected
    assert [c["id"] for c in report["cores"]] == selected
    verify_trace(garage, report, report["full"])
    M.parse_plan(garage.project, json.loads((garage.project / "candidate/source_plan.json").read_text()))


def test_no_ownership_rejects_cross_floor_joint_without_lidar_attribution(garage):
    install_joint(garage)
    for block in garage.ledger["blocks"]:
        block["floor_id"] = "F3" if block["id"] == "block_016" else "F2"
    put_json(garage.project / "maps/garage_manifest.json", garage.ledger)
    with pytest.raises(M.AssemblyError, match="cross-floor"):
        run(garage, ownership_mode="none")
    assert not (garage.project / "candidate").exists()


def test_no_ownership_same_floor_joint_retains_every_record(garage):
    install_joint(garage)
    for block in garage.ledger["blocks"]:
        block["floor_id"] = "F2"
    put_json(garage.project / "maps/garage_manifest.json", garage.ledger)
    report = run(garage, ownership_mode="none", support_radius_m=None)
    assert report["candidate_gaussians"] == report["retained_gaussians"]
    verify_trace(garage, report, report["full"])


def test_preflight_no_output_and_memory_disk_fail_closed(garage, monkeypatch):
    result = run(garage, preflight_only=True)
    assert result["status"] == "preflight_passed"
    assert not (garage.project / "candidate").exists()
    with pytest.raises(M.AssemblyError, match="CPU memory preflight failed"):
        M.resource_preflight(garage.project / "candidate", disk_bytes=1, memory_bytes=10**30)
    monkeypatch.setattr(M.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(M.AssemblyError, match="Disk preflight failed"):
        run(garage)
    assert not (garage.project / "candidate").exists()


def test_no_ownership_cli_alias_and_legacy_mode_are_explicit(garage, monkeypatch, capsys):
    monkeypatch.setattr(M, "ROOT", garage.project)
    assert M.main(["--output", "candidate", "--no-ownership", "--no-measured-support",
                   "--core-ids", "block_015", "block_016"]) == 0
    report = json.loads((garage.project / "candidate/assembly.json").read_text())
    assert report["ownership"]["mode"] == "none"
    assert report["retained_gaussians"] == len(garage.rows["block_015"]) + len(garage.rows["block_016"])
    assert json.loads(capsys.readouterr().out)["nonzero_core_count"] == 2


def test_fixed_surface_labels_ignore_normal_sign_and_keep_unknown_separate():
    normals = np.array([[0, 0, 2], [0, 0, -2], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1.]])
    z = np.array([-1., 1., -1., -1., -5., -1.])
    lateral = np.array([1., 1., 1., 1., 1., 12.])
    expected = [4, 5, 2, 0, 3, 3]
    for sign in (-1, 1):
        assert M.support_surface_labels(sign*normals, np.ones(6, bool), z, lateral).tolist() == expected


def test_support_census_counts_match_unowned_selection_with_same_reference(garage):
    selected = ["block_015", "block_016"]
    for bid in selected:
        path = garage.project / f"data/garage_surface_v1/{bid}/lidar_reference.npz"
        xyz = np.load(path)["xyz"]
        np.savez(path, xyz=xyz, normals=np.tile([0., 0., 1.], (len(xyz), 1)), normal_valid=np.ones(len(xyz), bool))
    report = run(garage, core_ids=selected, ownership_mode="none")
    census = M.audit_support_rejections("census.json", source_plan="plan.json", core_ids=selected, project=garage.project)
    assert census["support_rejected"] == sum(s["support_rejected_gaussians"] for s in report["sources"])
    assert census["gaussians"] == report["candidate_gaussians"]
    assert census["protocol"]["floor_labels_verified"] is False


def test_sparse_ray_two_sided_surfel_and_center_projection_agree():
    spec = importlib.util.spec_from_file_location("attribution", ROOT / "scripts/assemble_core_gaussians_lidar_ownership.py")
    attribution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attribution)
    camera = {"camera_to_worlds_ns": np.diag([1., -1., -1., 1.])[:3].tolist(),
              "fl_x": 20., "fl_y": 20., "cx": 10.5, "cy": 10.5, "w": 32, "h": 32}
    xyz = np.array([[0., 0., 3.], [0., 0., 3.]])
    scales = np.ones((2, 3))*.1
    q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    projection = attribution.surfel_projection(xyz, scales, q, camera)
    np.testing.assert_allclose(projection[3], [[10., 10.], [10., 10.]])
    ids, alpha, depth = attribution.ray_alpha(projection, np.array([.8, .8]), [10, 10])
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(alpha, [.8, .8])
    np.testing.assert_allclose(depth, [3., 3.])
    # Native dual-sided visibility: flipping normal hemisphere does not cull.
    np.testing.assert_allclose(projection[6][0], -projection[6][1])


def test_lidar_owner_requires_a_retained_same_floor_replacement(garage):
    spec = importlib.util.spec_from_file_location("lidar_owner", ROOT / "scripts/assemble_core_gaussians_lidar_ownership.py")
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    anchor = garage.centers[0]+[.15, 0, 0]
    for bid in ("block_000", "block_001"):
        rows = garage.rows[bid].copy()
        rows[0, [FIELDS.index(n) for n in ("x", "y", "z")]] = anchor
        put_ply(garage.project / garage.paths[bid], rows)
    reference = garage.project / "data/garage_surface_v1/block_001/lidar_reference.npz"
    np.savez(reference, xyz=np.vstack([garage.centers[1], anchor]))
    for block in garage.ledger['blocks']:
        block['floor_id'] = 'F1' if block['id'] in ('block_000', 'block_001') else 'unknown'
    guard = M.InputGuard(garage.project)
    sources = M.parse_plan(garage.project, garage.plan)
    selection = L.lidar_witness_plan(garage.project, sources, garage.ledger, guard, workers=1)
    assert not selection['masks']['block_000'][0]
    assert selection['masks']['block_001'][0]
    assert selection['report']['camera_distance_used_for_decisions'] is False
    assert selection['report']['unwitnessed_deletion'] is False
    report = run(garage, ownership_mode='lidar-support', ownership_selection=selection)
    trace, _ = verify_trace(garage, report, report['full'])
    assert np.any((trace['source_index']==1)&(trace['source_record_index']==0))
    # Same XYZ on an explicitly separate floor cannot steal its representation.
    garage.ledger['blocks'][1]['floor_id'] = 'F2'
    separate = L.lidar_witness_plan(garage.project, sources, garage.ledger, M.InputGuard(garage.project), workers=1)
    assert separate['masks']['block_000'][0]
    # A closer LiDAR return alone cannot justify deleting geometry.
    garage.ledger['blocks'][1]['floor_id'] = 'F1'
    put_ply(garage.project / garage.paths['block_001'], garage.rows['block_001'])
    sources = M.parse_plan(garage.project, garage.plan)
    unsupported = L.lidar_witness_plan(garage.project, sources, garage.ledger, M.InputGuard(garage.project), workers=1)
    assert unsupported['masks']['block_000'][0]


def test_train_support_partition_distinguishes_unknown_from_weak_and_mixed():
    spec = importlib.util.spec_from_file_location('train_support_partition', ROOT / 'scripts/assemble_core_gaussians_lidar_ownership.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A 1+1 observation must remain mixed, never become weak unilateral support.
    p = np.array([0, 1, 0, 2, 0, 3, 0, 1, 65535], dtype=np.uint16)
    n = np.array([0, 0, 1, 0, 2, 0, 3, 1, 65535], dtype=np.uint16)
    assert module.train_support_categories(p, n).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    with pytest.raises(module.AssemblyError):
        module.train_support_categories(np.array([-1]), np.array([0]))
    with pytest.raises(module.AssemblyError):
        module.train_support_categories(p[:-1], n)

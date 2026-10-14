"""CPU tests for explicit, auditable opacity experiments."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rescale_opacity", ROOT / "scripts/rescale_opacity_ply.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from assemble_core_gaussians import ply_header


@pytest.fixture
def ply():
    with tempfile.TemporaryDirectory(prefix="opacity_test_v2_", dir=ROOT / "reports") as directory:
        project = Path(directory)
        names = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)]
        names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        rows = np.zeros((7, len(names)), dtype="<f4")
        rows[:, 0] = np.arange(7)
        rows[:, 3] = [-1000., -20., -2., -0., 1., 20., 1000.]
        rows[:, 4] = -0.
        rows[:, -4] = 1.
        header = ply_header(names, len(rows))
        source = project / "input.ply"
        source.write_bytes(header + rows.tobytes())
        yield project, source, rows, header


@pytest.mark.parametrize("method", ["optical-depth", "alpha-multiply"])
def test_identity_keeps_header_and_all_payload_bytes(ply, method):
    project, source, rows, header = ply
    before = source.read_bytes()
    report = M.rescale(source, "identity", project=project, method=method, factor=1., chunk_bytes=rows.shape[1] * 4)
    assert (project / report["output"]).read_bytes() == before
    assert report["source_sha256"] == report["output_sha256"]
    assert report["changed_records"] == report["clipped_records"] == 0


def test_optical_depth_has_center_transmittance_law():
    logits = np.array([-2., -1., 0., 1., 2.], dtype=np.float64)
    new, clips = M.transform_opacity(logits, "optical-depth", 1.5)
    np.testing.assert_allclose(1. - expit(new), (1. - expit(logits)) ** 1.5, atol=1e-15)
    assert clips == 0


def test_alpha_multiplication_is_clipped_and_distinct_from_density_scaling():
    logits = np.array([-2., 0., 2.], dtype=np.float64)
    new, clips = M.transform_opacity(logits, "alpha-multiply", 2.13)
    np.testing.assert_allclose(expit(new), np.clip(expit(logits) * 2.13, 1e-6, 1 - 1e-6))
    assert clips == 2
    other, _ = M.transform_opacity(logits, "optical-depth", 2.13)
    assert not np.allclose(new, other)


def test_only_opacity_changes_and_all_original_colors_remain_exact(ply):
    project, source, rows, header = ply
    original = source.read_bytes()
    report = M.rescale(source, "experiment", project=project, method="alpha-multiply", factor=2.13)
    output = (project / report["output"]).read_bytes()
    assert output[:len(header)] == header
    after = np.frombuffer(output[len(header):], dtype="<f4").reshape(rows.shape)
    columns = [i for i in range(rows.shape[1]) if i != 3]
    assert after[:, columns].tobytes() == rows[:, columns].tobytes()
    assert np.isfinite(after).all()
    assert source.read_bytes() == original
    assert report["clipped_records"] > 0
    assert report["color_modified"] is report["trained_opacity_recovered"] is False
    assert report["pixel_alpha_coverage_measured"] is False
    assert json.loads((project / "experiment/opacity_experiment.json").read_text()) == report


@pytest.mark.parametrize("factor", [0., -1., float("nan"), float("inf")])
def test_invalid_factor_does_not_publish(ply, factor):
    project, source, rows, header = ply
    with pytest.raises(M.AssemblyError):
        M.rescale(source, "bad", project=project, method="alpha-multiply", factor=factor)
    assert not (project / "bad").exists()


def test_existing_output_never_overwritten_and_invalid_payload_cleaned(ply):
    project, source, rows, header = ply
    output = project / "old"
    output.mkdir()
    (output / "sentinel").write_bytes(b"keep")
    with pytest.raises(M.AssemblyError, match="NEW directory"):
        M.rescale(source, "old", project=project, method="optical-depth", factor=1.2)
    assert (output / "sentinel").read_bytes() == b"keep"
    rows[0, 4] = np.nan
    source.write_bytes(header + rows.tobytes())
    with pytest.raises(M.AssemblyError, match="Nonfinite PLY"):
        M.rescale(source, "bad", project=project, method="optical-depth", factor=1.2)
    assert not (project / "bad").exists()


def test_opacity_chunk_independent_and_preflight_only_writes_nothing(ply):
    project, source, rows, header = ply
    M.rescale(source, "preflight", project=project, method="optical-depth", factor=1.2, preflight_only=True)
    assert not (project / "preflight").exists()
    outputs = []
    for i, size in enumerate((rows.shape[1] * 4, rows.nbytes)):
        report = M.rescale(source, f"trial{i}", project=project, method="optical-depth", factor=1.2, chunk_bytes=size)
        outputs.append((project / report["output"]).read_bytes())
    assert outputs[0] == outputs[1]

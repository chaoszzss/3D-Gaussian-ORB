"""CPU guards for isolated CUDA finite-difference experiment."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_branch_unsaturated_fixtures():
    probe = load('probe_surfel_aa_depth_gradients')
    cases = probe.fixture_checks()
    assert [x['base']['branch'] for x in cases] == ['aa', 'aa', 'ray']
    assert all(x['checked_states'] == 31 for x in cases)
    assert all(x['minimum_branch_margin'] > .05 for x in cases)


def test_backward_patch_preserves_lowpass_and_other_branches():
    builder = load('build_surfel_aa_depth_vjp')
    source = (ROOT / 'runtime/inria2dgs_src/cuda_rasterizer/backward.cu').read_text()
    result = builder.patch_backward(source)
    prefix, tail = source.split('// // Propagate the gradients of depth', 1)
    _, suffix = tail.split('atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);', 1)
    assert result.startswith(prefix)
    assert result.endswith(suffix)
    patch = result[len(prefix):-len(suffix)]
    assert 'dL_dG' not in patch  # Coverage's existing low-pass chain stays above.
    assert 'dL_dz * Tw.x' in patch and 'dL_dz * Tw.y' in patch
    for i in range(9):
        assert f'global_id * 9 + {i}' in patch


def test_patch_rejects_drift_and_repatch():
    builder = load('build_surfel_aa_depth_vjp')
    source = (ROOT / 'runtime/inria2dgs_src/cuda_rasterizer/backward.cu').read_text()
    with pytest.raises(ValueError):
        builder.patch_backward('unrecognized source')
    with pytest.raises(ValueError):
        builder.patch_backward(builder.patch_backward(source))

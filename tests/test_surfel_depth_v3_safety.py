"""Reproduce the v3 unknown-foreground safety artifact on CPU."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from orbvi_gaussian.surfels import SurfelDepthPolicy, project_surfel_depth


def safety_measurements():
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    intrinsics = np.array([[320., 0, 2], [0, 320., 0], [0, 0, 1.]])
    cases = [
        dict(name=f'unknown_foreground_gap_{gap:g}', source_z=[13., 13. + gap],
             source_v=[32., 32.], known=[False, True], expected=False)
        for gap in (.00001, .05, .08, .1, .2)
    ]
    cases.extend([
        dict(name='unknown_foreground_close_source_own_correction',
             source_z=[13., 13.05], source_v=[32., 32.49],
             known=[False, True], expected=False),
        dict(name='unknown_distant_foreground', source_z=[25., 25.2],
             source_v=[32., 32.], known=[False, True], expected=False),
        dict(name='nearest_trusted_own_correction', source_z=[13.],
             source_v=[32.49], known=[True], expected=True,
             expected_depth=13. * 32.49 / 32.),
        dict(name='trusted_foreground_does_not_lend_plane_correction',
             source_z=[13., 13.05], source_v=[32., 32.49],
             known=[True, True], normals=[[1., 0, 0], [0, 0, 1.]], expected=False),
        dict(name='zero_tolerance_float32_nearest', source_z=[13.0000001],
             source_v=[32.], known=[True], expected=True,
             expected_depth=13.0000001, tolerance=0.),
    ])
    results = []
    for case in cases:
        for min_points in (1, 2):
            z = np.asarray(case['source_z'])
            v = np.asarray(case['source_v'])
            xyz = np.column_stack([np.zeros(len(z)), z * v / 320., z]) @ pose[:3, :3].T
            normals = case.get('normals', [[0., 0, 1.]] * len(z))
            policy = SurfelDepthPolicy.trusted_ground_v3(
                front_cluster_min_points=min_points,
                foreground_occlusion_tolerance_m=case.get('tolerance', .1))
            diagnostics = {}
            depth, valid = project_surfel_depth(
                xyz, normals, case['known'], pose, intrinsics, 5, 40,
                policy=policy, diagnostics=diagnostics)
            expected_depth = case.get('expected_depth', 0.)
            passed = (bool(valid[32, 2]) == case['expected']
                      and abs(float(depth[32, 2]) - expected_depth) <= 1e-6
                      and int(valid.sum()) == int(case['expected'])
                      and bool((depth[~valid] == 0).all()))
            results.append(dict(
                name=case['name'], front_cluster_min_points=min_points,
                source_z_m=case['source_z'], source_v=case['source_v'],
                normal_valid=case['known'], normals=normals,
                expected_valid=case['expected'], expected_depth_m=expected_depth,
                actual_valid=bool(valid[32, 2]), actual_depth_m=float(depth[32, 2]),
                status='PASS' if passed else 'FAIL', diagnostics=diagnostics))
    return results


def test_safety_report_measurements():
    results = safety_measurements()
    assert len(results) == 20
    assert all(case['status'] == 'PASS' for case in results), results


def write_report(path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    tests = ['tests/test_lidar_splatfacto.py', 'tests/test_core_joint_batch.py',
             'tests/test_surfels.py', 'tests/test_points.py',
             'tests/test_pixel_convention_closure.py', 'tests/test_prepare_joint_blocks.py',
             'tests/test_surfel_depth_v3_safety.py', 'tests/test_metric_depth_units.py',
             'tests/test_bounded_refinement_budget.py']
    sources = ['src/orbvi_gaussian/surfels.py', 'src/orbvi_gaussian/points.py',
               'src/orbvi_gaussian/lidar_splatfacto.py',
               'src/orbvi_gaussian/bounded_strategy.py',
               'scripts/train_lidar_block.py', 'scripts/train_core_joint_batch.py', *tests]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(root / 'src'))
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    measurements = safety_measurements()
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
                    for name, sha in hashes.items())
    passed = result.returncode == 0 and unchanged and all(c['status'] == 'PASS' for c in measurements)
    report = dict(schema_version=1, policy_version='surfel_depth_v3',
                  status='passed' if passed else 'failed', overall='PASS' if passed else 'FAIL',
                  policy=asdict(SurfelDepthPolicy.trusted_ground_v3()),
                  unknown_foreground_rule='source_float32_equals_nearest_zbuffer',
                  safety_cases=measurements,
                  passed_cases=sum(c['status'] == 'PASS' for c in measurements),
                  source_sha256=hashes, sources_unchanged_during_tests=unchanged,
                  commands=[command, [sys.executable, 'tests/test_surfel_depth_v3_safety.py',
                                      '--report', str(path)]],
                  environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH')},
                  pytest_returncode=result.returncode, pytest_stdout=result.stdout,
                  pytest_stderr=result.stderr, pixel_closure_report='reports/pixel_closure_v2.json',
                  scope='CPU safety and regression evidence only; requires independent DEV support/geometry checks before new training data. Not a rollout acceptance gate.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(result.stdout)
    print(json.dumps(dict(report=str(path), status=report['status'],
                          safety_cases=len(measurements), passed_cases=report['passed_cases'])))
    return 0 if passed else 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    raise SystemExit(write_report(parser.parse_args().report))

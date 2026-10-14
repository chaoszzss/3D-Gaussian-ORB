"""Complete fixed-reference evaluation of the directly seeded block000 pilot."""
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/fidelity_v2/train_block000_depthv3_seed_aa_cap003'
JOB = ROOT / 'runs/fidelity_v2/jobs/block000_depthv3_seed_aa_cap003'
PYTHON = str(ROOT / '.venv-train/bin/python')
OLD_CONFIG = 'runs/inria_surface_block000_12000/block_000/splatfacto/2026-10-06_203633/config.yml'
OLD_PLY = 'runs/garage_surface_v1/block_000/eval_dev/gaussians.ply'


def status(stage, **items):
    path = JOB / 'evaluation_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage, high_fidelity_accepted=False, **items), indent=2) + '\n')
    tmp.replace(path)


def execute(name, arguments, gpu=True):
    status('queued', operation=name)
    with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
        if gpu:
            fcntl.flock(lock, fcntl.LOCK_EX)
        result = json.loads((RUN / 'training_result.json').read_text())
        if result['status'] != 'trained' or result['checkpoint_step'] != 14999:
            raise ValueError('Requires actual completed 15000-step block000 pilot')
        config = str(ROOT / result['configs'][0])
        command = [PYTHON, '-B', *[config if v == '{config}' else v for v in arguments]]
        status('running', operation=name, command=command)
        with (JOB / (name + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS':'3',
                'OPENBLAS_NUM_THREADS':'3', 'MKL_NUM_THREADS':'3'}, check=True,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),) if gpu else ())


try:
    common = ['scripts/evaluate_gaussian_ply.py', '--block-id', 'block_000',
              '--max-images', '0', '--save-float-arrays']
    plan = ['--acceptance-plan', 'reports/acceptance_plan_v1.json',
            '--dataset-override', 'data/garage_surface_v1/block_000']
    resolution = ['--acceptance-plan', 'reports/acceptance_resolution_depthv3_plan.json',
                  '--dataset-override', 'data/garage_fidelity_depthv3/block_000']
    candidate = common + ['--config', '{config}', '--max-gaussians', '2500000',
                         '--renderer-integration-proof', 'reports/inria_backend_integration_gpu_v2.json']
    execute('candidate_640', candidate + plan + ['--export-ply', '--output',
        'runs/fidelity_v2/block000_seed_aa_cap003_acceptance'])
    execute('candidate_1280', candidate + resolution + ['--output',
        'runs/fidelity_v2/block000_seed_aa_cap003_resolution'])
    baseline = common + ['--config', OLD_CONFIG, '--ply', OLD_PLY]
    execute('original_640_refreshed', baseline + plan + ['--output',
        'runs/fidelity_v2/original000_refreshed_v2'])
    execute('original_1280', baseline + resolution + ['--output',
        'runs/fidelity_v2/original000_resolution_depthv3'])
    execute('old_full_640_refreshed', common + plan + ['--config', OLD_CONFIG,
        '--ply', 'maps/garage_core_all_v1/gaussians_full.ply', '--native-visibility',
        '--max-gaussians', '12000000', '--max-ply-bytes', '3221225472',
        '--active-gaussian-budget', '6000000', '--output',
        'runs/fidelity_v2/old_all28_block000_refreshed_v2'])
    execute('checkpoint_proof', ['scripts/verify_training_checkpoint.py',
        '--training-dir', str(RUN), '--block-id', 'block_000', '--evaluation-report',
        'runs/fidelity_v2/block000_seed_aa_cap003_acceptance/metrics.json',
        '--output', 'reports/block000_seed_aa_cap003_checkpoint.json'], gpu=False)
    execute('geometry_000', ['scripts/measure_surface_quality.py',
        '--ply', 'runs/fidelity_v2/block000_seed_aa_cap003_acceptance/gaussians.ply',
        '--metrics', 'runs/fidelity_v2/block000_seed_aa_cap003_acceptance/metrics.json',
        '--reference', 'data/garage_surface_v1/block_000/lidar_reference.npz',
        '--baseline-metrics', 'runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json',
        '--acceptance-plan', 'reports/acceptance_plan_v1.json',
        '--checkpoint-proof', 'reports/block000_seed_aa_cap003_checkpoint.json',
        '--output', 'reports/surface_quality_v2_block000_seed_aa_cap003.json'], gpu=False)
    status('evaluated_pending_main_review')
except Exception as exc:
    status('failed', error=repr(exc))
    raise

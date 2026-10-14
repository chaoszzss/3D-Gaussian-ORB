"""Evaluate the completed joint candidate on unchanged held-out references."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbvi_gaussian.training import confined_environment

run = ROOT / 'runs/fidelity_v2/train_joint015_depthv3_aa_metric_cap003'
job = ROOT / 'runs/fidelity_v2/jobs/joint015_depthv3_aa_metric_cap003'
python = str(ROOT / '.venv-train/bin/python')
env = confined_environment()
env.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')
status_path = job / 'evaluation_status.json'


def status(stage, **items):
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(stage=stage, high_fidelity_accepted=False, **items), indent=2) + '\n')
    temporary.replace(status_path)


def execute(name, arguments, gpu=True):
    status('queued', operation=name)
    with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
        if gpu:
            fcntl.flock(lock, fcntl.LOCK_EX)
        completed = json.loads((run / 'training_result.json').read_text())
        if completed['status'] != 'trained' or completed['checkpoint_step'] != 14999:
            raise ValueError('Requires completed 15000-step joint training')
        config = str(ROOT / completed['configs'][0])
        command = [python, '-B', *[config if item == '{config}' else item for item in arguments]]
        status('running', operation=name, command=command)
        with (job / (name + '.log')).open('xb') as log:
            subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log,
                stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),) if gpu else ())


try:
    common = ['scripts/evaluate_gaussian_ply.py', '--config', '{config}',
        '--max-images', '0', '--save-float-arrays', '--max-gaussians', '2500000']
    for bid in ('block_015', 'block_014', 'block_016'):
        command = common + ['--block-id', bid, '--dataset-override', 'data/garage_surface_v1/' + bid,
            '--acceptance-plan', 'reports/acceptance_plan_v1.json',
            '--output', 'runs/fidelity_v2/joint015_aa_cap003_acceptance/' + bid]
        if bid == 'block_015':
            command += ['--export-ply']
        execute('eval_' + bid, command)
    execute('eval_1280', common + ['--block-id', 'block_015',
        '--dataset-override', 'data/garage_fidelity_depthv3/block_015',
        '--acceptance-plan', 'reports/acceptance_resolution_depthv3_plan.json',
        '--output', 'runs/fidelity_v2/joint015_aa_cap003_resolution'])
    execute('checkpoint_proof', ['scripts/verify_training_checkpoint.py',
        '--training-dir', str(run), '--block-id', 'block_015',
        '--evaluation-report', 'runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/metrics.json',
        '--output', 'reports/joint015_aa_cap003_checkpoint.json'], gpu=False)
    execute('geometry_015', ['scripts/measure_surface_quality.py',
        '--ply', 'runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/gaussians.ply',
        '--metrics', 'runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/metrics.json',
        '--reference', 'data/garage_surface_v1/block_015/lidar_reference.npz',
        '--baseline-metrics', 'runs/fidelity_v2/baseline_old_all28/block_015/metrics.json',
        '--acceptance-plan', 'reports/acceptance_plan_v1.json',
        '--checkpoint-proof', 'reports/joint015_aa_cap003_checkpoint.json',
        '--output', 'reports/surface_quality_v2_joint015_aa_cap003.json'], gpu=False)
    status('evaluated_pending_main_review')
except Exception as exc:
    status('failed', error=repr(exc))
    raise

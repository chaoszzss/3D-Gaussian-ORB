"""Session-specific follow-up: immutable 015 checkpoint evaluation after GPU training."""
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbvi_gaussian.training import confined_environment

run = ROOT / 'runs/fidelity_v2/train_1280_depthv3_block015'
env = confined_environment()
env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
python = str(ROOT / '.venv-train/bin/python')
base = ROOT / 'runs/fidelity_v2'
status_path = base / 'depthv3_pilot015_followup.json'

def status(stage, **kwargs):
    status_path.write_text(json.dumps(dict(stage=stage, high_fidelity_accepted=False, **kwargs), indent=2)+'\n')

def execute(name, arguments, gpu=True):
    status('queued', operation=name)
    with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
        if gpu:
            fcntl.flock(lock, fcntl.LOCK_EX)
        complete = json.loads((run / 'training_result.json').read_text())
        if complete['status'] != 'trained' or complete['checkpoint_step'] != 11999:
            raise ValueError('Required completed 12000-step checkpoint unavailable')
        config = str(ROOT / complete['configs'][0])
        argv = [python, '-B', *[config if a == '{trained_config}' else a for a in arguments]]
        status('running', operation=name, argv=argv)
        with (base / 'logs' / (name + '.log')).open('xb') as log:
            subprocess.run(argv, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT,
                           pass_fds=(lock.fileno(),) if gpu else ())

try:
    common = ['scripts/evaluate_gaussian_ply.py', '--config', '{trained_config}', '--block-id', 'block_015',
              '--max-images', '0', '--save-float-arrays', '--max-gaussians', '2000000']
    execute('depthv3_pilot015_eval640', common + ['--dataset-override', 'data/garage_surface_v1/block_015',
            '--acceptance-plan', 'reports/acceptance_plan_v1.json', '--export-ply',
            '--output', 'runs/fidelity_v2/train_1280_depthv3_block015_acceptance'])
    execute('depthv3_pilot015_eval1280', common + ['--acceptance-plan', 'reports/acceptance_resolution_depthv3_plan.json',
            '--output', 'runs/fidelity_v2/train_1280_depthv3_block015_resolution'])
    original = json.loads((ROOT / 'runs/garage_surface_v1/block_015/eval_dev/metrics.json').read_text())
    execute('depthv3_original015_eval1280', ['scripts/evaluate_gaussian_ply.py', '--config', original['config'],
            '--ply', 'runs/garage_surface_v1/block_015/eval_dev/gaussians.ply',
            '--dataset-override', 'data/garage_fidelity_depthv3/block_015', '--block-id', 'block_015',
            '--max-images', '0', '--save-float-arrays', '--acceptance-plan', 'reports/acceptance_resolution_depthv3_plan.json',
            '--output', 'runs/fidelity_v2/original015_resolution_depthv3'])
    execute('depthv3_pilot015_verify', ['scripts/verify_training_checkpoint.py', '--training-dir', str(run),
            '--evaluation-report', 'runs/fidelity_v2/train_1280_depthv3_block015_acceptance/metrics.json',
            '--block-id', 'block_015', '--output', 'reports/depthv3_pilot015_checkpoint.json'], gpu=False)
    status('evaluations_completed_geometry_pending')
except Exception as exc:
    status('failed', error=repr(exc))
    raise

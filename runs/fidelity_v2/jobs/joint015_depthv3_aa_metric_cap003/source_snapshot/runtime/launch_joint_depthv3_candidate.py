"""Session experiment: snapshot actual sources/assets before launching joint training."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbvi_gaussian.training import confined_environment


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-ply', required=True, type=Path)
    args = parser.parse_args()
    warm = args.warm_ply.resolve()
    run = ROOT / 'runs/fidelity_v2/train_joint015_depthv3_aa_metric_cap003'
    job = ROOT / 'runs/fidelity_v2/jobs/joint015_depthv3_aa_metric_cap003'
    dataset = ROOT / 'data/garage_fidelity_depthv3_joint/core_015'
    if run.exists() or job.exists():
        raise FileExistsError('Use a new experiment identity; no overwrite/resume here')
    if not warm.is_file() or ROOT not in warm.parents:
        raise ValueError('Warm initialization must be a verified local file')
    for report in ('surfel_aa_depth_fd_gpu.json', 'inria_backend_integration_gpu_v1.json'):
        proof = json.loads((ROOT / 'reports' / report).read_text())
        if proof.get('candidate_status', proof.get('status')) != 'PASS':
            raise ValueError('Required CUDA proof has not passed: ' + report)
    job.mkdir(parents=True)
    status_path = job / 'status.json'

    def status(stage, **values):
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(),
            high_fidelity_accepted=False, **values), ensure_ascii=False, indent=2) + '\n')
        tmp.replace(status_path)

    status('snapshotting')
    sources = sorted((ROOT / 'src/orbvi_gaussian').rglob('*.py'))
    sources += [ROOT / p for p in ('scripts/train_lidar_block.py',
        'scripts/evaluate_gaussian_ply.py', 'scripts/evaluate_reconstruction.py',
        'scripts/native_gaussian_visibility.py', 'runtime/launch_joint_depthv3_candidate.py')]
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        dest = job / 'source_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        source_hashes[str(relative)] = digest(dest)
    inputs = {}
    for path in sorted(dataset.rglob('*')):
        if path.is_file():
            inputs[str(path.relative_to(ROOT))] = dict(sha256=digest(path), bytes=path.stat().st_size)
    inputs[str(warm.relative_to(ROOT))] = dict(sha256=digest(warm), bytes=warm.stat().st_size)
    command = [str(ROOT / '.venv-train/bin/python'), '-B', 'scripts/train_lidar_block.py',
        '--dataset', str(dataset), '--output', str(run), '--iterations', '15000',
        '--allow-candidate-calibration', '--surface-splats', '--renderer-inria',
        '--inria-backend', 'aa-depth-vjp', '--warm-ply', str(warm), '--warm-opacity-cap', '1',
        '--max-gaussians', '2500000', '--refine-every', '100', '--stop-split-at', '14000',
        '--stop-screen-size-at', '14000', '--split-screen-size', '.02',
        '--gaussian-max-scale-m', '.03', '--tangent-scale-target-m', '.03', '--tangent-scale-weight', '.001',
        '--depth-weight', '.5', '--alpha-weight', '.1', '--metric-depth-weight', '.05',
        '--metric-depth-delta-m', '.15', '--save-every', '2000', '--vis', 'tensorboard']
    manifest = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        run=str(run), command=command, sources=source_hashes, inputs=inputs,
        experiment_kind='compound engineering candidate; no single-variable causal attribution',
        selection='Three neighboring TRAIN blocks; fixed DEV/TEST split and fixed external scoring plans',
        radius_decision='3cm cap from step zero; static cap reduced floor wrong depth but created holes, so refine must recover observed coverage',
        budget_decision='2.5M points; prior 2M control reserved5.36GiB; actual peak monitored',
        high_fidelity_accepted=False)
    (job / 'input_snapshot.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    env = confined_environment()
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    status('queued')
    with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        changed = [p for p, sha in source_hashes.items() if digest(ROOT / p) != sha]
        if changed:
            status('blocked_source_changed', files=changed)
            raise ValueError('Source changed after snapshot: ' + repr(changed))
        status('training', controller_pid=os.getpid(), command=command)
        with (job / 'training.log').open('xb', buffering=0) as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
    changed = [p for p, sha in source_hashes.items() if digest(ROOT / p) != sha]
    status('trained_pending_evaluation' if result.returncode == 0 else 'training_failed',
        returncode=result.returncode, source_changed_during_run=changed)
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

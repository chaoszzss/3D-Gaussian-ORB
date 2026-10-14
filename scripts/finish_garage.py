#!/usr/bin/env python3
"""Wait for the verified batch, audit all 28 real blocks, then package the map.

This produces first-version artifacts. It never declares fidelity accepted.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbvi_gaussian.io import output_path
from orbvi_gaussian.training import confined_environment


def process_identity(pid):
    directory = Path('/proc') / str(pid)
    try:
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] == 'Z':
        return None
    if 'scripts/reconstruct_garage.py' not in command or 'garage_surface_v1' not in command:
        raise RuntimeError('PID belongs to another process; refusing to wait on it')
    return (command, fields[19])  # /proc stat field 22: process start ticks.


def manifest():
    value = json.loads((ROOT / 'maps/garage_manifest.json').read_text())
    expected = {'block_%03d' % i for i in range(28)}
    ids = [block['id'] for block in value['blocks']]
    if len(ids) != 28 or set(ids) != expected or value.get('run_id') != 'garage_surface_v1':
        raise RuntimeError('Manifest differs from the full planned run')
    return value


def publish(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--output', type=Path, default=ROOT / 'maps/garage_full_v1')
    parser.add_argument('--state', type=Path, default=ROOT / 'runtime/garage_completion_state.json')
    args = parser.parse_args()
    destination, state_path = output_path(args.output), output_path(args.state)
    if destination.exists():
        parser.error('Output already exists; inspect it or select another directory')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    record = {'status': 'waiting_for_training', 'training_pid': args.wait_for_pid,
              'output': str(destination.relative_to(ROOT)), 'started_epoch_s': time.time(),
              'high_fidelity_acceptance': 'not_assessed', 'viewer_review': 'pending'}
    def update(status):
        record.update(status=status, updated_epoch_s=time.time())
        publish(state_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    try:
        identity = process_identity(args.wait_for_pid) if args.wait_for_pid else None
        update('waiting_for_training')
        last = None
        while identity is not None:
            current = process_identity(args.wait_for_pid)
            if current is None:
                break
            if current != identity:
                raise RuntimeError('Training PID was reused; inspect actual job state')
            value = manifest()
            count = sum(b['training']['status'] == 'trained' for b in value['blocks'])
            if count != last:
                record['trained_blocks'] = count
                update('waiting_for_training')
                last = count
            time.sleep(15)
        value = manifest()
        if value['status'] == 'paused':
            update('paused_with_training')
            return 0
        unfinished = [b['id'] for b in value['blocks'] if
                      b['training']['status'] != 'trained' or b['eval']['status'] != 'evaluated']
        if unfinished:
            raise RuntimeError('Batch stopped with incomplete blocks: ' + ', '.join(unfinished))
        record['trained_blocks'] = sum(b['training']['status'] == 'trained' for b in value['blocks'])
        # All GPU training has stopped before this CPU-only checkpoint verification.
        env = confined_environment()
        env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        report_json = output_path(ROOT / 'reports' / ('completion_audit_' + tag + '.json'))
        report_md = report_json.with_suffix('.md')
        record['audit_json'], record['audit_md'] = str(report_json.relative_to(ROOT)), str(report_md.relative_to(ROOT))
        update('auditing_full_map')
        with (ROOT / 'runtime/garage_completion_audit.log').open('a') as log:
            result = subprocess.run([str(ROOT / '.venv-train/bin/python'), '-B',
                str(ROOT / 'scripts/audit_complete_map.py'), '--verify-checkpoints', '--require-complete',
                '--json', str(report_json), '--md', str(report_md)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Full-map audit failed; see ' + str(report_json.relative_to(ROOT)))
        if json.loads(report_json.read_text()).get('complete') is not True:
            raise RuntimeError('Audit did not establish all 28 block requirements')
        update('packaging_full_map')
        with (ROOT / 'runtime/garage_completion_package.log').open('a') as log:
            result = subprocess.run(['/usr/bin/python3', '-B', str(ROOT / 'scripts/package_garage.py'),
                '--output', str(destination)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (destination / 'packaging.json').is_file():
            raise RuntimeError('Packaging failed; see runtime/garage_completion_package.log')
        package = json.loads((destination / 'packaging.json').read_text())
        if package['status'] != 'complete' or package['coverage']['block_count'] != 28:
            raise RuntimeError('Package completion record is inconsistent')
        record['full_gaussians'] = package['full']['gaussians']
        record['overview_gaussians'] = package['overview']['gaussians']
        record['overview_manifest'] = str((destination / 'overview_manifest.json').relative_to(ROOT))
        record['coverage_scope'] = package['coverage']['scope']
        update('artifacts_ready_pending_viewer_review')
        return 0
    except Exception as exc:
        record['error'] = str(exc)
        update('failed_requires_review')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

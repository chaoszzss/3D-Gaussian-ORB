#!/usr/bin/env python3
"""Verify retained bytes and native environment after final-output cleanup."""
import json
import os
import subprocess
from cleanup_final_delivery import ROOT, RECEIPT, read, write, sha, now, usage


def main():
    assert (ROOT / (RECEIPT + '_execution.json')).is_file(), 'Cleanup has not executed'
    checked = []
    for row in read(RECEIPT + '_migration.json')['retained_hashes']:
        p = ROOT / row['path']
        assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == row['bytes']
        digest = sha(row['path'])
        assert digest == row['sha256'], row['path']
        checked.append(dict(row, verified=True))
    references = []
    for row in read('reports/cleanup_v2_retention_audit.json')['canonical_pointclouds_recommended']:
        assert sha(row['path']) == row['sha256'], row['path']
        references.append(row['path'])
    tracked = read(RECEIPT + '_tracked_before.json')
    for row in tracked:
        assert sha(row['path']) == row['sha256'], row['path']
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    code = '''import json,torch,sys
from orbvi_gaussian.inria_surfel_backend import inria_backend_provenance
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'backends':[inria_backend_provenance(x) for x in ['original','aa-depth-vjp']]}))
'''
    proc = subprocess.run([str(ROOT / '.venv-train/bin/python'), '-B', '-c', code], env=env,
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-4000:]
    native = json.loads(proc.stdout.strip().splitlines()[-1])
    baseline = read(RECEIPT + '_baseline.json')
    current = usage()
    result = {'status': 'PASS', 'verified_at': now(), 'retained_files': checked,
              'canonical_reference_files_verified': references,
              'unchanged_tracked_files': len(tracked), 'native_imports_cpu_only': native,
              'before': baseline, 'after': current,
              'project_allocated_bytes_reduced': baseline['project_allocated_bytes'] - current['project_allocated_bytes'],
              'filesystem_free_bytes_increase': current['filesystem_free_bytes'] - baseline['filesystem_free_bytes'],
              'limits': 'Integrity and dependency checks only; no new training, render-quality acceptance or raw bag rehash.'}
    write(RECEIPT + '_integrity.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'retained_files','canonical_reference_files_verified','native_imports_cpu_only'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()

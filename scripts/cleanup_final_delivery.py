#!/usr/bin/env python3
"""One-time, manifest-driven retention and cleanup of this garage experiment.

prepare retains/hash-checks deliverables, plan enumerates only approved output
trees, execute unlinks unchanged listed files. No recursive symlink traversal.
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path('/home/chaoszzss/dataset/rosbag/3D-Gaussian')
FINAL = 'maps/garage_fidelity_v2_full_preview'
RECEIPT = 'reports/cleanup_v2'
TARGETS = ['data', 'runs', 'maps', 'reports', 'runtime', 'viewer/runtime', 'experiments', '.pytest_cache']
SOURCE_SUFFIXES = {'.py', '.sh', '.patch', '.cu', '.cuh', '.cpp', '.h', '.hpp', '.js', '.mjs'}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads((ROOT / path).read_text())

def write(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.cleanup-tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, p)

def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def usage():
    return {'project_allocated_bytes': int(subprocess.check_output(['du', '-sx', '--block-size=1', str(ROOT)]).split()[0]),
            'filesystem_free_bytes': shutil.disk_usage(ROOT).free, 'at': now()}

def inside(path):
    p = ROOT / path
    assert not Path(path).is_absolute() and '..' not in Path(path).parts, path
    parent = p.parent
    while parent != ROOT:
        assert not parent.is_symlink(), path
        parent = parent.parent
    return p

def retain(src, dst, expected=None):
    s, d = inside(src), inside(dst)
    assert s.is_file() and not s.is_symlink(), src
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        os.link(s, d)
    assert not d.is_symlink() and d.stat().st_size == s.stat().st_size, dst
    digest = sha(dst)
    assert digest == (expected or sha(src)), dst
    return {'from': src, 'to': dst, 'bytes': d.stat().st_size, 'sha256': digest}

def prepare():
    assert not (ROOT / (RECEIPT + '_migration.json')).exists(), 'Already prepared'
    write(RECEIPT + '_baseline.json', usage())
    audit = read('reports/cleanup_v2_retention_audit.json')
    viewer = read('reports/cleanup_v2_viewer_dependencies.json')
    moved = []
    for row in viewer['migration']['mapping']:
        moved.append(retain(row['from'], row['to'], row['sha256']))
    for label, row in zip(['block_000', 'joint_014_016'], audit['latest_final_training_checkpoints']):
        dst = f'{FINAL}/checkpoints/{label}/step-000014999.ckpt'
        moved.append(retain(row['path'], dst, row['sha256']))
        moved.append(retain(str(Path(row['path']).parent.parent / 'config.yml'), f'{FINAL}/checkpoints/{label}/config.yml'))
        train_root = '/'.join(Path(row['path']).parts[:3])
        for name in ['training_result.json', 'launch.json', 'refinement.json', 'resume_events.jsonl']:
            p = f'{train_root}/{name}'
            if (ROOT / p).is_file():
                moved.append(retain(p, f'{FINAL}/checkpoints/{label}/{name}'))
    evaluations = ['block000_measuredseed_anneal_v1', 'block000_measuredseed_anneal_v1_resolution',
                   'joint015_aa_cap003_acceptance', 'joint015_aa_cap003_resolution']
    for name in evaluations:
        base = ROOT / 'runs/fidelity_v2' / name
        for p in sorted(base.rglob('*')):
            if p.is_file() and (p.suffix in {'.png', '.jpg'} or p.name == 'metrics.json'):
                dst = f'{FINAL}/evaluation/{name}/{p.relative_to(base)}'
                moved.append(retain(str(p.relative_to(ROOT)), dst))
    mapping = {x['from']: x['to'] for x in moved}
    # Retain exact old evidence before modifying any live manifest.
    manifests = [f'{FINAL}/manifest.json', f'{FINAL}/overview_manifest.json',
                 'maps/garage_fidelity_v2_block000_anneal_diagnostic/manifest.json',
                 'maps/garage_fidelity_v2_joint015_diagnostic/manifest.json', 'maps/garage_manifest.json']
    for name in manifests:
        retain(name, 'reports/cleanup_v2_history/' + name)
    def replace(value):
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, str):
            if value in mapping:
                return mapping[value]
            if value.startswith(str(ROOT) + '/') and value[len(str(ROOT)) + 1:] in mapping:
                return str(ROOT / mapping[value[len(str(ROOT)) + 1:]])
        return value
    for name in manifests[:-1]:
        write(name, replace(read(name)))
    write('maps/garage_manifest.json', read(f'{FINAL}/overview_manifest.json'))
    # Old gallery depended on failed/obsolete renders. Keep its URL useful.
    gallery = ROOT / 'reports/fidelity_v2_visual_review.html'
    if gallery.exists():
        retain(str(gallery.relative_to(ROOT)), 'reports/cleanup_v2_history/reports/fidelity_v2_visual_review.html')
    gallery_tmp = gallery.with_suffix('.cleanup-tmp')
    gallery_tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=full_preview_v2/"><title>最终重建成果</title><p><a href="full_preview_v2/">打开最终全场重建预览</a></p></html>\n')
    os.replace(gallery_tmp, gallery)
    hashes = [{'path': x['to'], 'sha256': x['sha256'], 'bytes': x['bytes']} for x in moved]
    for row in audit['full_and_overview']:
        assert sha(row['path']) == row['sha256'], row['path']
        hashes.append({k: row[k] for k in ['path', 'sha256', 'bytes']})
    write(RECEIPT + '_migration.json', {'created_at': now(), 'mapping': moved, 'retained_hashes': hashes,
          'historical_evidence_policy': 'assembly and previous audits retain original source paths/hashes; this receipt records relocation. Removed images/depth/intermediates prevent replay without rebuilding input data.'})
    print(json.dumps({'prepared': len(moved), 'final_model_assets': 26, 'final_checkpoints': 2}))

def policy():
    audit = read('reports/cleanup_v2_retention_audit.json')
    migration = read(RECEIPT + '_migration.json')
    moved = {x['from']: x['to'] for x in migration['mapping']}
    keep = set(audit['concrete_protected_files_for_main_crosscheck'])
    keep.update(x['path'] for x in audit['concrete_preserve_metadata_and_software_files'])
    keep.update(audit['pointcloud_recommended_paths'])
    keep = {moved.get(x, x) for x in keep}
    keep.update(x['path'] for x in read('reports/cleanup_v2_viewer_dependencies.json')['historical_provenance']['direct_small_json_files'] if x['path'] not in moved)
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
    keep.update(x for x in tracked if x)
    keep.add('reports/fidelity_v2_visual_review.html')
    prefixes = [FINAL + '/', 'reports/cleanup_v2', 'reports/full_preview_v2',
                'runtime/python/', 'runtime/inria2dgs_wheels/', 'runtime/training-sources/',
                'runtime/bin/', 'runtime/share/', 'runtime/config/']
    def protected(path):
        if path in keep or any(path.startswith(x) for x in prefixes):
            return True
        parts = Path(path).parts
        if len(parts) > 1 and parts[0] == 'runtime' and parts[1] in {'inria2dgs_src', 'inria2dgs_aa_depth_vjp_src'}:
            return 'build' not in parts and '__pycache__' not in parts
        if parts[0] == 'runtime' and len(parts) == 2 and Path(path).suffix in SOURCE_SUFFIXES:
            return True
        if parts[:2] == ('viewer', 'runtime') and len(parts) == 3 and Path(path).suffix in SOURCE_SUFFIXES:
            return True
        if parts[0] in {'reports', 'runs', 'experiments'} and Path(path).suffix in SOURCE_SUFFIXES:
            return True
        if parts[0] == 'experiments' and Path(path).suffix in {'.yaml', '.yml', '.json', '.md'}:
            return True
        return False
    return protected

def signature(p):
    s = p.lstat()
    return {'dev': s.st_dev, 'ino': s.st_ino, 'mode': s.st_mode, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'blocks': s.st_blocks, 'nlink': s.st_nlink}

def plan():
    protected = policy()
    groups = collections.defaultdict(lambda: {'files': 0, 'logical_bytes': 0})
    inodes = {}
    planfile = ROOT / (RECEIPT + '_delete_plan.jsonl.gz')
    with gzip.open(planfile, 'wt', encoding='utf-8') as out:
        for target in TARGETS:
            base = inside(target)
            assert not base.is_symlink(), target
            if not base.exists():
                continue
            for directory, dirs, files in os.walk(base, followlinks=False):
                for name in list(dirs):
                    if (Path(directory) / name).is_symlink():
                        files.append(name)
                        dirs.remove(name)
                for name in files:
                    p = Path(directory) / name
                    rel = str(p.relative_to(ROOT))
                    if protected(rel):
                        continue
                    s = signature(p)
                    assert stat.S_ISREG(s['mode']) or stat.S_ISLNK(s['mode']), rel
                    assert s['dev'] == ROOT.stat().st_dev, rel
                    out.write(json.dumps(dict(path=rel, **s)) + '\n')
                    groups[target]['files'] += 1
                    groups[target]['logical_bytes'] += s['bytes']
                    key = (s['dev'], s['ino'])
                    item = inodes.setdefault(key, dict(links_deleted=0, **s))
                    item['links_deleted'] += 1
    reclaim = sum(x['blocks'] * 512 for x in inodes.values() if x['links_deleted'] == x['nlink'])
    summary = {'created_at': now(), 'targets': TARGETS, 'groups': dict(groups),
               'files': sum(x['files'] for x in groups.values()),
               'estimated_reclaimed_allocated_bytes_excluding_directories': reclaim,
               'plan_sha256': sha(str(planfile.relative_to(ROOT))),
               'protected_tracked_files': True, 'follow_directory_symlinks': False}
    write(RECEIPT + '_plan_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))

def execute():
    assert not (ROOT / (RECEIPT + '_execution.json')).exists(), 'Already executed'
    planpath = RECEIPT + '_delete_plan.jsonl.gz'
    summary = read(RECEIPT + '_plan_summary.json')
    assert sha(planpath) == summary['plan_sha256']
    protected = policy()
    for row in read(RECEIPT + '_migration.json')['retained_hashes']:
        p = inside(row['path'])
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'], row['path']
    # Check the entire frozen plan before the first unlink.
    with gzip.open(ROOT / planpath, 'rt') as f:
        for line in f:
            row = json.loads(line)
            assert not protected(row['path']), row['path']
            assert any(row['path'].startswith(t + '/') for t in TARGETS), row['path']
            current = signature(inside(row['path']))
            assert all(current[k] == row[k] for k in ['dev', 'ino', 'mode', 'bytes', 'mtime_ns']), row['path']
    removed, skipped = 0, []
    with gzip.open(ROOT / planpath, 'rt') as f:
        for line in f:
            row = json.loads(line)
            p = inside(row['path'])
            current = signature(p)
            if any(current[k] != row[k] for k in ['dev', 'ino', 'mode', 'bytes', 'mtime_ns']):
                skipped.append(row['path'])
                continue
            p.unlink()
            removed += 1
            if removed % 25000 == 0:
                print(f'deleted {removed}', flush=True)
    dirs_removed = 0
    for target in TARGETS:
        for directory, dirs, files in os.walk(ROOT / target, topdown=False, followlinks=False):
            p = Path(directory)
            if p == ROOT / target or p.is_symlink():
                continue
            rel = str(p.relative_to(ROOT))
            if '.git' in p.parts or protected(rel) or protected(rel + '/'):
                continue
            try:
                p.rmdir()
                dirs_removed += 1
            except OSError:
                pass
    for path in ['runtime/tmp', 'runtime/cache', 'viewer/runtime', 'runs']:
        (ROOT / path).mkdir(parents=True, exist_ok=True)
    result = {'completed_at': now(), 'deleted_files_and_symlinks': removed, 'deleted_empty_directories': dirs_removed,
              'skipped_changed_files': skipped, 'plan_sha256': summary['plan_sha256'], 'after': usage()}
    write(RECEIPT + '_execution.json', result)
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'plan', 'execute'])
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    globals()[args.action]()

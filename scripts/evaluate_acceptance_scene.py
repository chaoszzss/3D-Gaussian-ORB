#!/usr/bin/env python3
"""Resumeable serial native evaluation on one immutable, deduplicated DEV plan."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from orbvi_gaussian.training import confined_environment
from acceptance_evaluation import file_sha256, summarize_protocols


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def validate_child_report(report, plan, block, plan_sha, max_images, ply_sha, direction_sha=None, model=None):
    expected = [r for r in plan['frames'] if r['block_id'] == block]
    if max_images:
        expected = expected[:max_images]
    rows = report.get('frames', [])
    if report.get('status') != 'evaluated' or report.get('metric_domain') != 'float_renderer':
        raise ValueError('Child lacks completed floating-point native metrics')
    if report.get('acceptance', {}).get('sha256') != plan_sha or report.get('acceptance_plan_id') != plan['plan_id']:
        raise ValueError('Child was evaluated with a different acceptance plan')
    if ply_sha and report.get('source_ply_sha256') != ply_sha:
        raise ValueError('Child used different PLY')
    actual_direction = report.get('direction_support') or {}
    if actual_direction.get('sha256') != direction_sha:
        raise ValueError('Child used different direction-support representation')
    if model is not None:
        for key in ('config', 'config_sha256', 'checkpoint', 'checkpoint_sha256',
                    'step', 'inference_model_step', 'transforms', 'transforms_sha256'):
            if report.get(key) != model[key]:
                raise ValueError('Child model/template identity differs: ' + key)
        source = report.get('source_ply') or {}
        if (report.get('checkpoint_is_template') is not True or report.get('split') != 'dev'
                or report.get('world_parser_scale') != 1.
                or report.get('world_parser_transform') not in (
                    [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]],
                    [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])):
            raise ValueError('Child is not the required fixed metric template evaluation')
        if any(source.get(k) != model['source_ply'][k] for k in ('path', 'sha256', 'gaussians')):
            raise ValueError('Child used different complete source PLY')
        if report.get('gaussians') != source['gaussians'] or report.get('source_ply_sha256') != source['sha256']:
            raise ValueError('Child source record count or SHA mismatch')
        expected_representation = source['sha256']
        if direction_sha:
            import hashlib
            expected_representation = hashlib.sha256((expected_representation + ':' + direction_sha).encode()).hexdigest()
        actual_representation = report.get('representation_id')
        if actual_representation is None and direction_sha is None:
            # Plain historical reports encode the complete identity in the
            # already-verified source PLY SHA, without a redundant field.
            actual_representation = source['sha256']
        if actual_representation != expected_representation:
            raise ValueError('Child complete representation identity differs')
    if len(rows) != len(expected) or report.get('images') != len(expected):
        raise ValueError('Child does not cover exactly the requested canonical frames')
    for row, frozen in zip(rows, expected):
        for key in ('dataset_index', 'file_path', 'camera_id', 'source_image_path', 'source_mask_path'):
            if row.get(key) != frozen[key]:
                raise ValueError('Child fixed source mismatch: ' + key)
        if str(row.get('group_id')) != str(frozen['group_id']) or row.get('acceptance_mask_path') != frozen['mask_path']:
            raise ValueError('Child group/ego mask mismatch')
        for key in ('source_lidar_depth_path', 'source_lidar_support_path'):
            if key in frozen and row.get(key) != frozen[key]:
                raise ValueError('Child fixed teacher mismatch: ' + key)
        if 'camera' in frozen:
            def flatten(value):
                if isinstance(value, list):
                    return [x for v in value for x in flatten(v)]
                return [value]
            for key, value in frozen['camera'].items():
                actual = row.get('camera', {}).get(key)
                a, b = flatten(actual), flatten(value)
                if len(a) != len(b) or not all(type(x) in (int, float) and math.isfinite(x)
                    and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-5) for x, y in zip(a, b)):
                    raise ValueError('Child fixed camera mismatch: ' + key)
        if set(row.get('protocols', {})) != {'raw', 'legacy', 'ego_v2'}:
            raise ValueError('Child lacks all three scoring protocols')
    return rows


def pin_model_inputs(plan, blocks, ply):
    """Pin actual templates and models at controller entry, including resume."""
    from package_garage import inspect_ply
    digests, sources, jobs = {}, {}, {}
    def sha(path):
        path = Path(path).resolve()
        if path not in digests:
            digests[path] = file_sha256(path)
        return digests[path]
    for block in blocks:
        reference = json.loads((ROOT / 'runs/garage_surface_v1' / block / 'eval_dev/metrics.json').read_text())
        source = ply.resolve() if ply else ROOT / 'runs/garage_surface_v1' / block / 'eval_dev/gaussians.ply'
        if source not in sources:
            with source.open('rb') as stream:
                spec = inspect_ply(stream)
            if source.stat().st_size != len(spec['header']) + spec['count'] * spec['stride']:
                raise ValueError('Malformed complete source PLY')
            sources[source] = dict(path=str(source), sha256=sha(source), gaussians=spec['count'])
        frame = next(f for f in plan['frames'] if f['block_id'] == block)
        config, checkpoint = (Path(reference[key]).resolve() for key in ('config', 'checkpoint'))
        jobs[block] = dict(config=str(config), config_sha256=sha(config),
            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), step=reference['step'],
            inference_model_step=reference['step'], transforms=frame['transforms'],
            transforms_sha256=frame['transforms_sha256'], source_ply=sources[source])
    return jobs


def evaluate(args):
    plan_path = args.plan.resolve()
    from build_acceptance_plan import load_plan
    plan = load_plan(plan_path, verify_assets=True)
    blocks = list(dict.fromkeys(f['block_id'] for f in plan['frames']))
    if args.blocks:
        if not set(args.blocks) <= set(blocks):
            raise ValueError('Unknown plan block')
        blocks = [b for b in blocks if b in args.blocks]
    out = args.output.resolve()
    if ROOT not in out.parents:
        raise ValueError('Output must be inside project')
    if out.exists() and not args.resume:
        raise FileExistsError('Use a new output or explicit --resume')
    out.mkdir(parents=True, exist_ok=True)
    env = confined_environment()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    identity = {'plan': str(plan_path), 'plan_sha256': file_sha256(plan_path),
                'ply': str(args.ply.resolve()) if args.ply else 'original_single_block_sources',
                'ply_sha256': file_sha256(args.ply) if args.ply else None,
                'direction_support': str(args.direction_support.resolve()) if args.direction_support else None,
                'direction_support_sha256': file_sha256(args.direction_support) if args.direction_support else None,
                'blocks': blocks, 'max_images': args.max_images}
    identity_path = out / 'identity.json'
    if identity_path.exists():
        previous_identity = json.loads(identity_path.read_text())
        previous_identity.setdefault('direction_support', None)
        previous_identity.setdefault('direction_support_sha256', None)
        if previous_identity != identity:
            raise ValueError('Resume identity changed')
    atomic_json(identity_path, identity)
    with (out / '.controller.lock').open('a+') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models = pin_model_inputs(plan, blocks, args.ply)
        pinned_path = out / 'model_inputs.json'
        pinned = dict(schema_version=1, models=models,
            scope='Assets verified on controller entry; not a retrospective launch-time proof for historical children')
        if pinned_path.exists() and json.loads(pinned_path.read_text()) != pinned:
            raise ValueError('Pinned model/template inputs changed since previous controller entry')
        atomic_json(pinned_path, pinned)
        support = None
        if args.direction_support:
            from direction_support import load_direction_support, verify_direction_support_unchanged
            source = models[blocks[0]]['source_ply']
            count = source['gaussians']
            support = load_direction_support(args.direction_support, source,
                {'means': range(count), 'quats': range(count)})
        rows, reports = [], []
        for block in blocks:
            dest = out / block
            report_path = dest / 'metrics.json'
            if not report_path.exists():
                if dest.exists():
                    raise ValueError('Incomplete child retained; inspect and choose new output: ' + str(dest))
                model = models[block]
                argv = [str(ROOT / '.venv-train/bin/python'), '-B', str(ROOT / 'scripts/evaluate_gaussian_ply.py'),
                        '--config', model['config'], '--ply', model['source_ply']['path'], '--output', str(dest),
                        '--dataset-override', str(Path(model['transforms']).parent),
                        '--acceptance-plan', str(plan_path), '--block-id', block,
                        '--max-images', str(args.max_images), '--save-float-arrays']
                if args.ply:
                    argv += ['--native-visibility', '--max-gaussians', '12000000', '--max-ply-bytes', str(3 * 1024**3),
                             '--active-gaussian-budget', str(args.active_gaussian_budget)]
                if args.direction_support:
                    argv += ['--direction-support', str(args.direction_support.resolve())]
                atomic_json(out / 'status.json', {'status': 'running', 'block': block, 'pid': os.getpid(),
                                                 'completed_blocks': [r['block_id'] for r in reports],
                                                 'updated_epoch_s': time.time(), 'argv': argv})
                with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    with (out / (block + '.log')).open('ab', buffering=0) as log:
                        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                pass_fds=(controller.fileno(), gpu_lock.fileno()))
                if result.returncode:
                    atomic_json(out / 'status.json', {'status': 'failed', 'block': block, 'returncode': result.returncode})
                    raise RuntimeError('Native evaluation failed: ' + str(out / (block + '.log')))
            report = json.loads(report_path.read_text())
            rows.extend(validate_child_report(report, plan, block, identity['plan_sha256'],
                                              args.max_images, models[block]['source_ply']['sha256'],
                                              identity['direction_support_sha256'], models[block]))
            reports.append({'block_id': block, 'path': str(report_path), 'sha256': file_sha256(report_path),
                            'psnr_db_mean': report['psnr_db_mean'], 'images': report['images']})
            expected = len(plan['frames'])
            keys = [(str(r['group_id']), r['camera_id']) for r in rows]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate scores across blocks')
            summary = {'status': 'evaluated' if len(rows) == expected else 'partial',
                       'complete': len(rows) == expected, 'high_fidelity_accepted': False,
                       'identity': identity, 'images': len(rows), 'expected_images': expected,
                       'model_inputs_path': str(pinned_path), 'model_inputs_sha256': file_sha256(pinned_path),
                       'acceptance_plan_id': plan['plan_id'], 'metric_domain': 'float_renderer',
                       'primary_protocol': 'ego_v2', 'protocols': summarize_protocols(rows),
                       'blocks': reports, 'frames': rows}
            atomic_json(out / 'summary.json', summary)
            print(block, report['psnr_db_mean'], 'unique frames:', len(rows), flush=True)
        if support is not None:
            verify_direction_support_unchanged(support)
        atomic_json(out / 'status.json', {'status': summary['status'], 'completed_blocks': blocks,
                                         'updated_epoch_s': time.time()})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=ROOT / 'reports/acceptance_plan_v1.json')
    p.add_argument('--ply', type=Path, help='Omit for original trained single-block baselines')
    p.add_argument('--direction-support', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--blocks', nargs='+')
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--active-gaussian-budget', type=int, default=6000000)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.max_images < 0 or not 1 <= args.active_gaussian_budget <= 6000000:
        p.error('Invalid image/allocation budget')
    if args.direction_support and not args.ply:
        p.error('Direction support requires --ply')
    evaluate(args)


if __name__ == '__main__':
    main()

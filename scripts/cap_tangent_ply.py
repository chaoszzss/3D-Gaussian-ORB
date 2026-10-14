#!/usr/bin/env python3
"""CPU static PLY ablation: cap only activated scale_0/1; retain every record."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time

from assemble_core_gaussians import (
    ROOT, MAX_CHUNK_BYTES, PROVENANCE, AssemblyError, InputGuard, _state,
    inspect_ply, local_path, positive_int, resource_preflight, sha256, write_json,
)
import numpy as np


def log_cap_float32(cap_m):
    if not np.isfinite(cap_m) or cap_m <= 0:
        raise AssemblyError('Tangent cap must be positive finite meters')
    exact = np.log(float(cap_m))
    value = np.float32(exact)
    # Round downward when needed, so exp(stored float32 log) never exceeds cap.
    if float(value) > exact:
        value = np.nextafter(value, np.float32(-np.inf))
    if not np.isfinite(value):
        raise AssemblyError('Tangent cap is not representable in float32 log space')
    return value


def radius_quantiles(log_scales):
    levels = [0., .5, .9, .95, .99, 1.]
    keys = ['min', 'p50', 'p90', 'p95', 'p99', 'max']
    result = {'method': 'exact nearest order statistic of exp(stored log scale); no sampling',
              'units': 'm', 'quantity': 'tangent Gaussian one-sigma scale, not screen radius'}
    for key, values in [('scale_0', log_scales[:, 0]), ('scale_1', log_scales[:, 1]),
                        ('max_tangent', np.max(log_scales, axis=1))]:
        q = np.exp(np.quantile(values, levels, method='nearest').astype(np.float64))
        if not np.isfinite(q).all():
            raise AssemblyError('Activated source scale is not finite')
        result[key] = dict(zip(keys, q.tolist()))
    return result


def verify_output(source, output, spec, columns, limit, step):
    """Read actual on-disk records; verify all non-target field bytes and cap law."""
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    untouched = [i for i in range(len(spec['names'])) if i not in columns]
    changed = [0, 0]
    with source.open('rb') as src, output.open('rb') as dst:
        if inspect_ply(src) != spec or inspect_ply(dst) != spec:
            raise AssemblyError('Source or output PLY header changed')
        source_hash.update(spec['header']); output_hash.update(spec['header'])
        for begin in range(0, spec['count'], step):
            n = min(step, spec['count']-begin)
            old, new = src.read(n*spec['stride']), dst.read(n*spec['stride'])
            if len(old) != n*spec['stride'] or len(new) != len(old):
                raise AssemblyError('Truncated verification payload')
            source_hash.update(old); output_hash.update(new)
            a = np.frombuffer(old, dtype='<f4').reshape(n, -1)
            b = np.frombuffer(new, dtype='<f4').reshape(n, -1)
            if not np.array_equal(a.view('<u4')[:, untouched], b.view('<u4')[:, untouched]):
                raise AssemblyError('A non-tangent attribute changed')
            for j, column in enumerate(columns):
                clipped = a[:, column] > limit
                if (not np.array_equal(a[:, column][~clipped].view('<u4'), b[:, column][~clipped].view('<u4'))
                        or not np.all(b[:, column][clipped] == limit)):
                    raise AssemblyError('Tangent cap law or unchanged small scales violated')
                changed[j] += int(clipped.sum())
        if src.read(1) or dst.read(1):
            raise AssemblyError('Trailing verification payload')
    return source_hash.hexdigest(), output_hash.hexdigest(), changed


def cap_assembly(assembly_path, output, *, cap_m, chunk_bytes=8*1024**2,
                 preflight_only=False, project=ROOT):
    start = time.monotonic()
    project = Path(project).resolve()
    if project != ROOT:
        local_path(ROOT, project)
    destination = local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError('Output must be a NEW directory with an existing parent')
    positive_int(chunk_bytes, 'chunk_bytes')
    if chunk_bytes > MAX_CHUNK_BYTES:
        raise AssemblyError('chunk_bytes exceeds bounded budget')
    limit = log_cap_float32(cap_m)
    guard = InputGuard(project)
    parent_path = local_path(project, assembly_path)
    parent = guard.json(parent_path)
    full = parent['full']
    source = local_path(project, full['path'])
    provenance = local_path(project, full['provenance']['path'])
    source_state = _state(source)
    with source.open('rb') as reader:
        spec = inspect_ply(reader)
    if (spec['count'] != full['gaussians'] or spec['count'] < 1
            or source_state[2] != len(spec['header'])+spec['count']*spec['stride']):
        raise AssemblyError('PLY count/size differs from parent assembly')
    if not all(name in spec['names'] for name in ('scale_0', 'scale_1', 'scale_2')) or chunk_bytes < spec['stride']:
        raise AssemblyError('Need all scale fields and at least one complete record per chunk')
    preflight = resource_preflight(destination, disk_bytes=source_state[2]+provenance.stat().st_size+4*1024**2,
                                   memory_bytes=1024**3+spec['count']*8+16*chunk_bytes)
    if preflight_only:
        return {'status': 'preflight_passed', 'preflight': preflight, 'gaussians': spec['count'],
                'cap_m': cap_m, 'activated_stored_cap_m': float(np.exp(float(limit)))}
    guard.snapshot(provenance)
    if guard.files[provenance]['sha256'] != full['provenance']['sha256']:
        raise AssemblyError('Provenance SHA differs from parent assembly')
    trace = np.load(provenance, mmap_mode='r', allow_pickle=False)
    if trace.dtype != PROVENANCE or trace.shape != (spec['count'],):
        raise AssemblyError('Provenance dtype/count/row alignment mismatch')
    source_ids = parent['provenance']['source_index_to_id']
    per_source = np.zeros((len(source_ids), 4), dtype=np.int64)
    columns = [spec['names'].index(f'scale_{i}') for i in range(2)]
    step = chunk_bytes // spec['stride']
    log_scales = np.empty((spec['count'], 2), dtype='<f4')
    destination.mkdir()
    try:
        result_path = destination/'gaussians_full.ply'
        source_hash, result_hash = hashlib.sha256(), hashlib.sha256()
        changed_axes = np.zeros(2, dtype=np.int64)
        changed_records = both = 0
        with source.open('rb') as reader, result_path.open('xb') as writer:
            if inspect_ply(reader) != spec:
                raise AssemblyError('Source schema changed before transform')
            writer.write(spec['header']); source_hash.update(spec['header']); result_hash.update(spec['header'])
            for begin in range(0, spec['count'], step):
                n = min(step, spec['count']-begin)
                raw = reader.read(n*spec['stride'])
                if len(raw) != n*spec['stride']:
                    raise AssemblyError('Incomplete source PLY payload')
                source_hash.update(raw)
                rows = np.frombuffer(raw, dtype='<f4').reshape(n, -1).copy()
                if not np.isfinite(rows).all():
                    raise AssemblyError('Nonfinite source PLY attribute')
                old = rows[:, columns]
                log_scales[begin:begin+n] = old
                clipped = old > limit
                union = np.any(clipped, axis=1)
                changed_axes += clipped.sum(axis=0)
                changed_records += int(union.sum()); both += int(np.all(clipped, axis=1).sum())
                origins = trace['source_index'][begin:begin+n]
                if np.any(origins >= len(source_ids)):
                    raise AssemblyError('Provenance references an unknown source')
                for j, mask in enumerate((np.ones(n, bool), clipped[:, 0], clipped[:, 1], union)):
                    per_source[:, j] += np.bincount(origins[mask], minlength=len(source_ids))
                rows[:, columns] = np.minimum(old, limit)
                encoded = rows.tobytes()
                if writer.write(encoded) != len(encoded):
                    raise OSError('Short PLY output write')
                result_hash.update(encoded)
            if reader.read(1):
                raise AssemblyError('Trailing source PLY payload')
        if source_hash.hexdigest() != full['sha256']:
            raise AssemblyError('Source PLY SHA differs from parent assembly')
        guard.remember(source, source_state, source_hash.hexdigest())
        output_trace = destination/'full_provenance.npy'
        with provenance.open('rb') as reader, output_trace.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=8*1024**2)
        output_trace_sha = sha256(output_trace)
        if output_trace_sha != full['provenance']['sha256']:
            raise AssemblyError('Copied provenance is not byte-identical')
        source_sha, result_sha, verified_axes = verify_output(source, result_path, spec, columns, limit, step)
        if (source_sha != full['sha256'] or result_sha != result_hash.hexdigest()
                or verified_axes != changed_axes.tolist()):
            raise AssemblyError('On-disk full-record verification failed')
        before = radius_quantiles(log_scales)
        np.minimum(log_scales, limit, out=log_scales)
        after = radius_quantiles(log_scales)
        guard.check({source})  # Source was fully rehashed during on-disk verification.
        report = {'schema_version': 1, 'status': 'tangent_cap_experiment_pending_paired_native_evaluation',
            'cpu_only': True, 'high_fidelity_accepted': False, 'default_changed': False,
            'preflight': preflight, 'cap_m': cap_m, 'stored_log_cap_float32': float(limit),
            'activated_stored_cap_m': float(np.exp(float(limit))),
            'parent_assembly': {'path': parent_path.relative_to(project).as_posix(),
                                'sha256': guard.files[parent_path]['sha256']},
            'source_ply': {'path': source.relative_to(project).as_posix(), 'sha256': source_sha},
            'full': {'path': result_path.relative_to(project).as_posix(), 'gaussians': spec['count'],
                'file_bytes': result_path.stat().st_size, 'sha256': result_sha,
                'provenance': {'path': output_trace.relative_to(project).as_posix(), 'gaussians': spec['count'],
                    'file_bytes': output_trace.stat().st_size, 'sha256': output_trace_sha}},
            'provenance': dict(parent['provenance'], records_modified=True,
                               modified_fields=['scale_0', 'scale_1'], row_provenance_unmodified=True),
            'records_deleted': 0, 'row_order_unchanged': True,
            'header_byte_identical': True, 'provenance_byte_identical': True,
            'changed_fields': ['scale_0', 'scale_1'], 'scale_2_byte_identical': True,
            'all_other_fields_byte_identical': True, 'on_disk_all_records_verified': spec['count'],
            'changed_records': changed_records, 'clipped_scale_0': int(changed_axes[0]),
            'clipped_scale_1': int(changed_axes[1]), 'both_axes_clipped_records': both,
            'before': before, 'after': after,
            'sources': [dict(source=sid, **dict(zip(('records', 'clipped_scale_0', 'clipped_scale_1', 'changed_records'),
                                                   per_source[i].tolist()))) for i, sid in enumerate(source_ids)],
            'physical_scope': 'Caps Gaussian tangent one-sigma scales in meters. Keeps centers, quaternions, third scale, '
                'peak opacity and SH unchanged. This reduces the local footprint; it does not conserve integrated opacity '
                'or guarantee ray-plane depth bounds. Smaller splats can create holes. Ordinary static double-sided PLY; '
                'no camera-conditioned source selection and no direction sidecar applied.',
            'inputs': guard.report(), 'seconds': time.monotonic()-start,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        write_json(destination/'tangent_cap_audit.json', report)
        return report
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', required=True)
    parser.add_argument('--output', required=True, help='NEW map directory')
    parser.add_argument('--cap-m', required=True, type=float)
    parser.add_argument('--chunk-mib', type=int, default=8)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = cap_assembly(args.assembly, args.output, cap_m=args.cap_m,
                              chunk_bytes=args.chunk_mib*1024**2, preflight_only=args.preflight_only)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'incomplete', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ('status', 'preflight', 'full', 'changed_records',
        'clipped_scale_0', 'clipped_scale_1', 'both_axes_clipped_records', 'before', 'after', 'seconds', 'peak_rss_kib') if k in result}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

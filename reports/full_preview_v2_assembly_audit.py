#!/usr/bin/env python3
"""Independently compare every full PLY byte and every midpoint overview record.

CPU only, <=3 threads. No assembler import, GPU, bag read, image read, or edits
to maps/data. Requires the separate completed source audit. No quality claim.
"""
import datetime
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
import full_preview_v2_source_audit as A
import numpy as np

ROOT = A.ROOT
BASE = ROOT / 'maps/garage_fidelity_v2_full_preview'
OUTPUT = ROOT / 'reports/full_preview_v2_assembly_audit.json'


def bbox(lower, upper):
    return {'min': lower.tolist(), 'max': upper.tolist()}


def run():
    started = time.monotonic()
    assert not OUTPUT.exists(), 'Refuse to overwrite completed audit'
    A.pin(__file__)
    A.pin('reports/full_preview_v2_source_audit.py')
    independent, independent_desc = A.read('reports/full_preview_v2_source_audit.json')
    content = {k: v for k, v in independent.items() if k != 'report_id'}
    assert hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == independent['report_id']
    assert independent['status'] == 'PASS_SOURCE_COVERAGE_AND_IDENTITY_ONLY'
    assembly, assembly_desc = A.read(BASE / 'assembly.json')
    manifest, manifest_desc = A.read(BASE / 'manifest.json')
    overview_manifest, overview_manifest_desc = A.read(BASE / 'overview_manifest.json')
    assert assembly['properties'] == A.PROPERTIES
    for m in (assembly, manifest, overview_manifest):
        assert m['coordinates'] == 'camera_init' and m['units'] == 'm' and m['camera_up'] == [0, 0, 1]
        assert m['high_fidelity_accepted'] is False and m['native_fullscope_evaluated'] is False
        assert m['summary']['full_scene_metrics'] is None
        assert m['coverage']['covered_source_blocks'] == A.EXPECTED
        assert m['coverage']['source_coverage_complete'] is True and m['coverage']['floor_labels_verified'] is False
    assets = independent['assets']
    sources = assembly['sources']
    assert len(sources) == len(assets) == len(manifest['blocks']) == 26
    covered = [b for s in sources for b in s['source_blocks']]
    assert len(covered) == len(set(covered)) == 28 and sorted(covered) == A.EXPECTED
    assert len({s['path'] for s in sources}) == 26
    for path, sha in assembly['input_sha256'].items():
        assert independent['input_sha256'][path] == sha, 'Assembly source differs from independent source selection'
        if not path.endswith('.ply'):
            A.pin(path, sha)
    full_path, overview_path = (ROOT / assembly[k]['path'] for k in ('full', 'overview'))
    full_before, overview_before = A.state(full_path), A.state(overview_path)
    results = []
    full_low, full_high = np.full(3, np.inf), np.full(3, -np.inf)
    sample_low, sample_high = np.full(3, np.inf), np.full(3, -np.inf)
    full_cursor = overview_cursor = 0
    with full_path.open('rb') as full, overview_path.open('rb') as overview:
        full_header, full_count, stride = A.ply_header(full)
        overview_header, overview_count, overview_stride = A.ply_header(overview)
        assert stride == overview_stride == 248
        for key, count, header, before in (('full', full_count, full_header, full_before), ('overview', overview_count, overview_header, overview_before)):
            expected = assembly[key]
            assert count == expected['gaussians'] and len(header) == expected['header_bytes']
            assert before[2] == expected['file_bytes'] == len(header) + count * stride
        assert full_count == independent['sum_source_ply_gaussians']
        full_hash, overview_hash = hashlib.sha256(full_header), hashlib.sha256(overview_header)
        per_region, remainder = divmod(overview_count, 28)
        region_quota = {b: per_region + (i < remainder) for i, b in enumerate(A.EXPECTED)}
        for source_index, (source, checked, entry) in enumerate(zip(sources, assets, manifest['blocks'])):
            assert source['source_index'] == source_index
            for field in ('id', 'path', 'sha256', 'source_blocks', 'gaussians', 'bbox_m'):
                assert source[field] == checked[field], 'Independent source mismatch: ' + field
            for field in ('id', 'path', 'sha256', 'source_blocks', 'bbox_m'):
                assert entry[field] == checked[field], 'Preview manifest mismatch: ' + field
            assert entry['training']['gaussians'] == checked['gaussians']
            path = ROOT / source['path']
            before = A.state(path)
            with path.open('rb') as original:
                header, count, source_stride = A.ply_header(original)
                assert source_stride == source['record_stride_bytes'] == stride
                assert len(header) == source['source_header_bytes'] == checked['source_header_bytes']
                assert count == source['gaussians']
                assert before[2] == checked['file_bytes'] == len(header) + count * stride
                assert source['full_row_start'] == full_cursor
                assert source['full_row_stop'] == full_cursor + count
                assert full.tell() == len(full_header) + full_cursor * stride
                quota = sum(region_quota[b] for b in source['source_blocks'])
                assert 0 < quota <= count
                assert source['overview_gaussians'] == quota
                assert source['overview_row_start'] == overview_cursor
                assert source['overview_row_stop'] == overview_cursor + quota
                assert overview.tell() == len(overview_header) + overview_cursor * stride
                # Exact integer arithmetic; no random draw, rounding, float conversion, or attributes rewritten.
                indices = ((2 * np.arange(quota, dtype=np.int64) + 1) * count) // (2 * quota)
                assert 0 <= indices[0] <= indices[-1] < count and np.all(np.diff(indices) > 0)
                source_hash = hashlib.sha256(header)
                source_payload_hash, copied_payload_hash = hashlib.sha256(), hashlib.sha256()
                expected_sample_hash, actual_sample_hash = hashlib.sha256(), hashlib.sha256()
                source_low, source_high = np.full(3, np.inf), np.full(3, -np.inf)
                local_sample_low, local_sample_high = np.full(3, np.inf), np.full(3, -np.inf)
                source_row = sample_row = 0
                while source_row < count:
                    rows = min(32768, count - source_row)
                    original_bytes = original.read(rows * stride)
                    full_bytes = full.read(rows * stride)
                    assert len(original_bytes) == len(full_bytes) == rows * stride
                    assert full_bytes == original_bytes, f'Full payload changed: {source["id"]}, row{source_row}'
                    source_hash.update(original_bytes)
                    source_payload_hash.update(original_bytes)
                    copied_payload_hash.update(full_bytes)
                    full_hash.update(full_bytes)
                    floats = np.frombuffer(full_bytes, '<f4').reshape(rows, len(A.PROPERTIES))
                    assert np.isfinite(floats).all()
                    source_low = np.minimum(source_low, floats[:, :3].min(axis=0))
                    source_high = np.maximum(source_high, floats[:, :3].max(axis=0))
                    stop = int(np.searchsorted(indices, source_row + rows, side='left'))
                    wanted = indices[sample_row:stop] - source_row
                    if len(wanted):
                        expected = np.frombuffer(original_bytes, np.uint8).reshape(rows, stride)[wanted].tobytes()
                        actual = overview.read(len(expected))
                        assert len(actual) == len(expected) and actual == expected, f'Overview attributes/indices changed: {source["id"]}, sample{sample_row}'
                        expected_sample_hash.update(expected)
                        actual_sample_hash.update(actual)
                        overview_hash.update(actual)
                        xyz = np.frombuffer(actual, '<f4').reshape(len(wanted), len(A.PROPERTIES))[:, :3]
                        local_sample_low = np.minimum(local_sample_low, xyz.min(axis=0))
                        local_sample_high = np.maximum(local_sample_high, xyz.max(axis=0))
                    source_row += rows
                    sample_row = stop
                assert not original.read(1) and source_row == count and sample_row == quota
                assert source_hash.hexdigest() == source['sha256']
                assert bbox(source_low, source_high) == source['bbox_m']
                assert source_payload_hash.digest() == copied_payload_hash.digest()
                assert expected_sample_hash.digest() == actual_sample_hash.digest()
            A.register(path, source_hash.hexdigest(), before)
            full_low, full_high = np.minimum(full_low, source_low), np.maximum(full_high, source_high)
            sample_low, sample_high = np.minimum(sample_low, local_sample_low), np.maximum(sample_high, local_sample_high)
            full_cursor += count
            overview_cursor += quota
            results.append({**source, 'status': 'PASS_ALL_FULL_AND_SAMPLED_RECORD_BYTES',
                'full_payload_bytes_compared': count * stride, 'full_payload_sha256': source_payload_hash.hexdigest(),
                'overview_records_compared': quota, 'overview_payload_bytes_compared': quota * stride,
                'overview_payload_sha256': actual_sample_hash.hexdigest(),
                'overview_source_indices_sha256_int64_le': hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest(),
                'overview_first_source_index': int(indices[0]), 'overview_last_source_index': int(indices[-1]),
                'sampled_bbox_m': bbox(local_sample_low, local_sample_high)})
            print(source['id'], 'full', count, 'overview', quota, 'all bytes exact', flush=True)
        assert full_cursor == full_count and overview_cursor == overview_count
        assert not full.read(1) and not overview.read(1), 'Unexpected trailing output payload'
    assert full_hash.hexdigest() == assembly['full']['sha256']
    assert overview_hash.hexdigest() == assembly['overview']['sha256']
    full_desc = A.register(full_path, full_hash.hexdigest(), full_before)
    overview_desc = A.register(overview_path, overview_hash.hexdigest(), overview_before)
    assert bbox(full_low, full_high) == assembly['full']['bbox_m'] == independent['bbox_m']
    assert bbox(sample_low, sample_high) == assembly['overview']['bbox_m']
    assert np.all(full_low <= sample_low) and np.all(sample_high <= full_high)
    assert len(overview_manifest['blocks']) == 1
    view = overview_manifest['blocks'][0]
    assert view['path'] == overview_desc['path'] and view['sha256'] == overview_desc['sha256']
    assert view['training']['gaussians'] == overview_count and view['source_blocks'] == A.EXPECTED
    assert view['bbox_m'] == assembly['full']['bbox_m']
    assert view['sampled_bbox_m'] == assembly['overview']['bbox_m']
    assert overview_manifest['full_artifact'] == assembly['full']
    for path, before in A.STATES.items():
        assert A.state(ROOT / path) == before, 'Input changed during audit: ' + path
    result = {'schema_version': 1, 'status': 'PASS_LOSSLESS_ASSEMBLY_AND_EXACT_OVERVIEW_SAMPLING',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'assembly': assembly_desc, 'independent_source_audit': independent_desc,
        'manifest': manifest_desc, 'overview_manifest': overview_manifest_desc,
        'assets_verified': len(results), 'source_blocks_verified': A.EXPECTED, 'source_coverage_duplicate_count': 0,
        'full': {**full_desc, 'gaussians': full_count, 'file_bytes': full_before[2], 'header_bytes': len(full_header),
            'payload_bytes_compared': full_count * stride, 'bbox_m': bbox(full_low, full_high),
            'exact_all_source_records_preserved': True, 'attributes_modified': False, 'cropping_or_deduplication': False},
        'overview': {**overview_desc, 'gaussians': overview_count, 'file_bytes': overview_before[2],
            'header_bytes': len(overview_header), 'payload_bytes_compared': overview_count * stride,
            'sampling_formula': 'source_index=floor((2*j+1)*count/(2*quota)), j=0..quota-1; integer arithmetic',
            'regional_quota': region_quota, 'all_selected_attributes_bitexact': True,
            'actual_sampled_bbox_m': bbox(sample_low, sample_high), 'manifest_fit_bbox_m': view['bbox_m'],
            'bbox_scope': 'Fit uses full-source bbox; actual sample bbox can shrink. No requirement that every extremal source record be sampled.'},
        'sources': results, 'input_sha256': A.PINS, 'high_fidelity_accepted': False, 'full_scene_metrics': None,
        'scope': 'Exact records and coverage of existing models only. No new training, native full-scene image evaluation, floor survey, calibration proof or unobserved-space completeness claim.',
        'attribute_scope': 'Every one of62 float32 fields compared as raw bytes, including XYZ/normals/SH colour coefficients/opacity/log scales/quaternions. No assembly-created or altered records found.',
        'historical_block006': independent['historical_block006_correction']['retained_statement'],
        'resource_usage': {'cpu_threads_max': 3, 'gpu_used': False, 'elapsed_seconds': time.monotonic()-started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'chunk_source_records': 32768},
        'command': "CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -B reports/full_preview_v2_assembly_audit.py"}
    result['report_id'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    with OUTPUT.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'full_count': full_count, 'overview_count': overview_count,
        'full_sha256': full_desc['sha256'], 'overview_sha256': overview_desc['sha256'], 'resource_usage': result['resource_usage']}))


if __name__ == '__main__':
    if sys.flags.optimize:
        raise SystemExit('Do not disable audit assertions with -O')
    run()

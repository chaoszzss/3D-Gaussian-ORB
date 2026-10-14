#!/usr/bin/env python3
"""CPU-only source coverage audit; reads PLYs and metadata, never images or bags."""
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '3'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'reports/full_preview_v2_source_audit'
EXPECTED = [f'block_{i:03d}' for i in range(28)]
MANIFESTS = ['maps/garage_fidelity_v2_viewer/manifest.json',
    'maps/garage_fidelity_v2_block000_anneal_diagnostic/manifest.json',
    'maps/garage_fidelity_v2_joint015_diagnostic/manifest.json']
PINS, STATES = {}, {}
PROPERTIES = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)] + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]


def state(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def digest(path):
    path = (ROOT / path).resolve()
    assert path.suffix != '.bag', 'Bag rehash explicitly forbidden'
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def register(path, sha, before):
    path = (ROOT / path).resolve()
    assert state(path) == before, 'Input changed: ' + str(path)
    key = str(path.relative_to(ROOT))
    assert key not in PINS or PINS[key] == sha
    PINS[key] = sha
    STATES[key] = before
    return {'path': key, 'sha256': sha}


def pin(path, expected=None):
    path = (ROOT / path).resolve()
    before = state(path)
    key = str(path.relative_to(ROOT))
    if key in PINS:
        assert before == STATES[key]
        sha = PINS[key]
    else:
        sha = digest(path)
    if expected is not None:
        assert sha == expected, 'SHA mismatch: ' + str(path)
    return register(path, sha, before)


def read(path, expected=None):
    desc = pin(path, expected)
    return json.loads((ROOT / desc['path']).read_text()), desc


def ply_header(stream):
    lines = []
    while True:
        b = stream.readline()
        assert b and sum(map(len, lines)) < 65536, 'Invalid PLY header'
        lines.append(b)
        if b.strip() == b'end_header':
            break
    header = b''.join(lines)
    words = [b.decode('ascii').strip().split() for b in lines]
    assert words[0] == ['ply'] and words[1] == ['format', 'binary_little_endian', '1.0']
    elements = [w for w in words if w[0] == 'element']
    assert len(elements) == 1 and elements[0][:2] == ['element', 'vertex']
    props = [w for w in words if w[0] == 'property']
    assert all(len(w) == 3 and w[1] == 'float' for w in props)
    assert [w[2] for w in props] == PROPERTIES
    return header, int(elements[0][2]), len(PROPERTIES) * 4


def scan_ply(path, expected_sha):
    path = ROOT / path
    before = state(path)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    with path.open('rb') as f:
        header, count, stride = ply_header(f)
        assert before[2] == len(header) + count * stride
        h = hashlib.sha256(header)
        remaining = count
        while remaining:
            take = min(32768, remaining)
            b = f.read(take * stride)
            assert len(b) == take * stride
            h.update(b)
            a = np.frombuffer(b, '<f4').reshape(take, len(PROPERTIES))
            assert np.isfinite(a).all(), 'Nonfinite source attribute'
            lower = np.minimum(lower, a[:, :3].min(axis=0))
            upper = np.maximum(upper, a[:, :3].max(axis=0))
            remaining -= take
        assert not f.read(1)
    assert h.hexdigest() == expected_sha, 'Source PLY SHA mismatch: ' + str(path)
    return {**register(path, h.hexdigest(), before), 'gaussians': count, 'file_bytes': before[2],
        'source_header_bytes': len(header), 'record_stride_bytes': stride,
        'bbox_m': {'min': lower.tolist(), 'max': upper.tolist()}, 'all_attributes_finite': True,
        'sh_degree': 3, 'coordinates': 'camera_init', 'units': 'm'}


def time_range(values):
    values = list(map(int, values))
    return {'min_ns': min(values), 'max_ns': max(values), 'span_s': (max(values)-min(values))/1e9}


def dataset(path, image_index):
    d, desc = read(path)
    assert d['orientation_override'] == 'none'
    frames = {f['file_path']: f for f in d['frames']}
    assert len(frames) == len(d['frames'])
    sets, groups, summary = {}, {}, {}
    assert len({f['source_file_path'] for f in frames.values()}) == len(frames)
    for f in frames.values():
        raw = image_index[f['source_file_path']]
        assert raw['camera_id'] == f['camera_id']
        assert int(raw['header_ns']) == f['source_header_ns'] == f['timestamp_ns']
        assert int(raw['bag_ns']) == f['source_bag_ns']
    for split in ('train', 'val', 'test'):
        files = d[split + '_filenames']
        assert len(files) == len(set(files)) and set(files) <= frames.keys()
        selected = [frames[f] for f in files]
        sets[split] = {f['source_file_path'] for f in selected}
        groups[split] = {str(f['group_id']) for f in selected}
        summary[split] = {'images': len(selected), 'groups': len(groups[split]),
            'group_time': time_range(groups[split]), 'actual_header_time': time_range(f['timestamp_ns'] for f in selected)}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not sets[a] & sets[b] and not groups[a] & groups[b]
    assert sum(len(v) for v in sets.values()) == len(frames)
    return {'artifact': desc, 'frames': len(frames), 'splits': summary, 'splits_disjoint': True,
        'source_image_index_time_binding': 'All camera/header/bag timestamps match saved image metadata'}, sets, {f['source_file_path']: f for f in frames.values()}


def main():
    started = time.monotonic()
    assert not PREFIX.with_suffix('.json').exists() and not PREFIX.with_suffix('.md').exists()
    sys.path.insert(0, str(ROOT / 'scripts'))
    from verify_training_checkpoint import read_config
    pin('scripts/verify_training_checkpoint.py')
    pin(__file__)
    manifests = [read(p)[0] for p in MANIFESTS]
    old, zero, joint = manifests
    for m in manifests:
        assert m['coordinates'] == 'camera_init' and m['units'] == 'm' and m['camera_up'] == [0, 0, 1]
    original = {b['id']: b for b in old['blocks']}
    assert sorted(original) == EXPECTED and len(old['blocks']) == 28
    inventory, _ = read(old['source_inventory']['path'], old['source_inventory']['sha256'])
    original_pins = {b['id']: b for b in inventory['sources']}
    images_path = ROOT / 'data/keyframes_2hz/indexes/images.csv'
    pin(images_path)
    images = list(csv.DictReader(images_path.open()))
    image_index = {r['file_path']: r for r in images}
    assert len(image_index) == len(images)
    camera_images = [r for r in images if r['camera_id'].startswith('camera_')]
    base, base_sets = {}, {}
    for bid in EXPECTED:
        desc, split_sets, _ = dataset(f'data/garage_surface_v1/{bid}/transforms.json', image_index)
        base[bid] = {**desc, 'block_id': bid, 'floor_label_unverified': original[bid]['floor_id']}
        base_sets[bid] = split_sets
    chosen = [(zero['blocks'][0], zero, ['block_000'])]
    chosen += [(original[b], old, [b]) for b in EXPECTED if b not in ('block_000', 'block_014', 'block_015', 'block_016')]
    assert len(joint['blocks']) == len(zero['blocks']) == 1
    chosen += [(joint['blocks'][0], joint, ['block_014', 'block_015', 'block_016'])]
    chosen.sort(key=lambda item: item[2][0])
    covered = [b for _, _, ids in chosen for b in ids]
    assert len(chosen) == 26 and len(covered) == len(set(covered)) == 28 and sorted(covered) == EXPECTED
    assets, inodes = [], set()
    active_splits = {s: set() for s in ('train', 'val', 'test')}
    for block, manifest, ids in chosen:
        bid = block['id']
        assert block.get('source_blocks', [bid]) == ids and block['floor_verified'] is False
        assert block['coordinates'] == 'camera_init'
        expected_sha = block.get('sha256') or original_pins[bid]['sha256']
        asset = scan_ply(block['path'], expected_sha)
        inode = state(ROOT / block['path'])[:2]
        assert inode not in inodes
        inodes.add(inode)
        assert asset['gaussians'] == block['training']['gaussians']
        assert asset['bbox_m'] == block['bbox_m']
        if manifest is old:
            assert block['path'] == original_pins[bid]['path'] and asset['gaussians'] == original_pins[bid]['vertex_count']
            metrics, metric_desc = read(Path(block['path']).parent / 'metrics.json')
            training_path = Path(block['path']).parent.parent / 'training_result.json'
            trained, training_desc = read(training_path)
        else:
            metrics, metric_desc = read(manifest['evaluation_report']['path'], manifest['evaluation_report']['sha256'])
            trained, training_desc = read(manifest['training_result']['path'], manifest['training_result']['sha256'])
            assert manifest['source_export']['sha256'] == asset['sha256']
            assert manifest['source_export']['gaussians'] == asset['gaussians']
        assert metrics['world_parser_scale'] == 1.0
        assert np.array_equal(np.asarray(metrics['world_parser_transform']), np.eye(4)[:3])
        assert metrics['export']['coordinates'] == 'camera_init' and metrics['export']['sh_degree'] == 3
        assert metrics['export']['gaussians'] == asset['gaussians'] and metrics['export']['bbox_m'] == asset['bbox_m']
        config_desc = pin(metrics['config'], metrics.get('config_sha256'))
        assert config_desc['path'] in [str((ROOT / p).resolve().relative_to(ROOT)) for p in trained['configs']]
        config = read_config(ROOT / config_desc['path'])
        dm = config['pipeline']['datamanager']; parser = dm['dataparser']
        assert parser['auto_scale_poses'] is False and parser['orientation_method'] == parser['center_method'] == 'none'
        assert parser['scale_factor'] == 1.0 and (ROOT / dm['data']).resolve() == (ROOT / parser['data']).resolve()
        ds, sets, _ = dataset(Path(parser['data']) / 'transforms.json', image_index)
        for split in active_splits:
            expected = set().union(*(base_sets[b][split] for b in ids))
            assert sets[split] == expected, 'Replacement source split differs: ' + bid + ':' + split
            active_splits[split] |= sets[split]
        assets.append({**asset, 'id': bid, 'source_blocks': ids, 'source_manifest': pin(MANIFESTS[manifests.index(manifest)]),
            'generation': 'historical' if manifest is old else 'current_round', 'floor_label_unverified': block['floor_id'],
            'evaluation_metadata': metric_desc, 'config': config_desc, 'training_result': training_desc,
            'training_dataset': ds, 'source_split_union_exact': True,
            'world_parser_scale': metrics['world_parser_scale'], 'world_parser_transform': metrics['world_parser_transform'],
            'export_filter_recorded': manifest.get('export_filter'),
            'quality_scope': 'Existing diagnostic/historical PLY; neither manifest inclusion nor source coverage certifies fidelity'})
        print(bid, asset['gaussians'], 'SHA/count/all-float-payload/bbox/parser/split verified', flush=True)
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not active_splits[a] & active_splits[b]
    all_prepared = set().union(*active_splits.values())
    full_bbox = {'min': np.min([a['bbox_m']['min'] for a in assets], axis=0).tolist(),
                 'max': np.max([a['bbox_m']['max'] for a in assets], axis=0).tolist()}
    bag_proof, bag_desc = read('reports/raw_bag_full_identity.json')
    assert bag_proof['status'] == 'PASS_FULL_FILE_SHA256_EQUAL' and len({r['sha256'] for r in bag_proof['files']}) == 1
    bag_stats = []
    for record in bag_proof['files']:
        s = Path(record['path']).stat()  # No bag open or payload read.
        assert (s.st_size, s.st_mtime_ns, s.st_ino) == (record['bytes'], record['mtime_ns'], record['inode'])
        bag_stats.append({**record, 'current_stat_matches_recorded_identity': True})
    extraction, extraction_desc = read('data/keyframes_2hz/manifest.json')
    source, source_desc = read('data/keyframes_2hz/source.json')
    assert extraction['source'] == source['source'] == bag_proof['files'][0]['path']
    assert extraction['source_size_bytes'] == source['size_bytes'] == bag_proof['files'][0]['bytes']
    assert extraction['source_mtime_ns'] == source['source_mtime_ns'] == bag_proof['files'][0]['mtime_ns']
    message_path = ROOT / 'data/keyframes_2hz/indexes/messages.csv'
    pin(message_path)
    topics = {}
    with message_path.open() as stream:
        for row in csv.DictReader(stream):
            item = topics.setdefault(row['topic'], {'count': 0, 'header_min_ns': None, 'header_max_ns': None, 'bag_min_ns': None, 'bag_max_ns': None})
            item['count'] += 1
            for field, key in (('header_ns', 'header'), ('bag_ns', 'bag')):
                if row[field]:
                    value = int(row[field])
                    for op in ('min', 'max'):
                        k = key + '_' + op + '_ns'
                        item[k] = value if item[k] is None else (min(item[k], value) if op == 'min' else max(item[k], value))
    assert {k: v['count'] for k, v in topics.items()} == extraction['observed_counts']
    correction, correction_desc = read('reports/historical_block006_resume_audit.json')
    pin(correction['training_result_path'], correction['training_result_sha256'])
    assert correction['checkpoint_step'] == 11999 and all(row['step'] == 11001 for rows in correction['optimizers'].values() for row in rows)
    split_summary = {s: {'unique_images': len(keys), 'header_time': time_range(image_index[k]['header_ns'] for k in keys),
        'bag_record_time': time_range(image_index[k]['bag_ns'] for k in keys)} for s, keys in active_splits.items()}
    omitted_keyframes = [r for r in camera_images if r['file_path'] not in all_prepared]
    report = {'schema_version': 1, 'status': 'PASS_SOURCE_COVERAGE_AND_IDENTITY_ONLY',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'asset_count': len(assets), 'source_blocks': sorted(covered), 'source_block_count': len(covered),
        'new_assets': 2, 'new_source_blocks': ['block_000', 'block_014', 'block_015', 'block_016'], 'historical_assets': 24,
        'sum_source_ply_gaussians': sum(a['gaussians'] for a in assets), 'bbox_m': full_bbox,
        'coordinates': 'camera_init', 'units': 'm', 'all_parsers_identity': True, 'source_properties': PROPERTIES,
        'assets': assets, 'prepared_blocks': list(base.values()),
        'bag_identity': {'evidence': bag_desc, 'recorded_method': bag_proof['method'], 'files': bag_stats,
            'bag_payload_reopened_or_rehashed': False, 'scope': 'Prior full-file hash evidence pinned; current size/mtime/inode checked only'},
        'time_and_split_coverage': {'extraction_manifest': extraction_desc, 'source_metadata': source_desc,
            'time_basis': 'Header time for capture; bag time for later recording. These clocks are not interchangeable.',
            'existing_message_index_topics': topics, 'saved_camera_images': len(camera_images), 'saved_ipm_images': len(images)-len(camera_images),
            'keyframe_camera_header_time': time_range(r['header_ns'] for r in camera_images),
            'prepared_unique_images': len(all_prepared), 'selected_actual_header_time': time_range(image_index[k]['header_ns'] for k in all_prepared),
            'splits': split_summary, 'global_splits_disjoint': True, 'replacement_split_unions_equal_historical_regions': True,
            'unused_saved_camera_keyframes': omitted_keyframes, 'unused_saved_camera_keyframe_count': len(omitted_keyframes),
            'scope': 'Declared training split membership verified from actual saved config/transforms, not proof of how often every image was sampled by SGD. No image pixels read.'},
        'historical_block006_correction': {'artifact': correction_desc, 'recorded': correction,
            'retained_statement': 'Loop step11999 / declared12000; six Adam counters11001. Do not claim12000 Adam updates or no training. Checkpoint optimizer tensors not reread in this coverage audit.'},
        'full_scene_metrics': None, 'high_fidelity_accepted': False, 'floor_labels_verified': False,
        'scope_limits': ['F1-F6 and ramps are inherited temporal labels, not independently validated physical floors.',
            'Complete means all28 prepared collection regions represented once; spatial overlap among distinct regions remains.',
            'Neither every bag message was used for training nor unobserved space proven complete.',
            'No scores from different blocks or generations averaged into a full-scene score.',
            'Source audit does not by itself verify assembler output bytes; separate assembly audit follows.'],
        'input_sha256': PINS, 'resource_usage': {'cpu_threads_max': 3, 'gpu_used': False,
            'elapsed_seconds': time.monotonic()-started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    for path, before in STATES.items():
        assert state(ROOT / path) == before, 'Input changed during audit'
    report['report_id'] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    PREFIX.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# 全场预览独立源覆盖核查', '', 'PASS仅指源覆盖和资产身份，不是高保真质量验收。', '',
        f"26个资产覆盖28个源区块，无重复归属：24个历史资产＋000新资产＋014/015/016联合资产。共{report['sum_source_ply_gaussians']:,}条导出PLY记录。",
        f"坐标camera_init，单位m；全部parser变换为单位矩阵、scale=1。实际源并集bbox：{full_bbox}。", '',
        '|资产|覆盖源区块|PLY记录数|源路径|', '|---|---|---:|---|']
    lines += [f"|{a['id']}|{', '.join(a['source_blocks'])}|{a['gaussians']}|`{a['path']}`|" for a in assets]
    lines += ['', '每个PLY完整流式SHA、全部float有限性、实际bbox/count/stride与清单及export元数据一致；全部训练config及split归属已核对。', '',
        f"已有相机keyframes共{len(camera_images)}张（另IPM{len(images)-len(camera_images)}张）；28块去重后{len(all_prepared)}张，TRAIN/DEV/TEST分别" + '/'.join(str(len(active_splits[s])) for s in ('train','val','test')) + '，全局互斥。替换模型的训练/DEV/TEST集合与被替换源区块的并集相同。',
        f"实际keyframe header范围：{report['time_and_split_coverage']['keyframe_camera_header_time']}；已准备图像范围：{report['time_and_split_coverage']['selected_actual_header_time']}。未进入prepared blocks的已保存相机帧{len(omitted_keyframes)}张，逐条见JSON。", '',
        f"用户bag完整SHA历史证据：`{bag_proof['files'][0]['sha256']}`，{bag_proof['files'][0]['bytes']} bytes。本次仅核对既有证据与当前stat，未打开或重新hash bag。images/messages CSV记录独立核对；header采集时间与bag写入时间分开报告。", '',
        '006更正保留：loop11999/声明12000，六组Adam计数11001；不写成12000次优化，也不写成未经训练。', '',
        'F1-F6＋坡道沿用未核实标签；完整仅指28个现有采集区块覆盖，不等于每条bag消息均训练或未观测空间完整。新000/joint现有导出分别省略32/5条checkpoint记录，未在本审计改写。',
        '各来源分数未作混合平均；全场分数为空。后续无损拼接核查同样不代表高保真。', '',
        '命令：`CUDA_VISIBLE_DEVICES=\'\' .venv-train/bin/python -B reports/full_preview_v2_source_audit.py`',
        f"CPU≤3线程、无GPU；耗时{report['resource_usage']['elapsed_seconds']:.2f}s，RSS峰值{report['resource_usage']['peak_rss_bytes']/2**20:.1f}MiB。", '']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps({k: report[k] for k in ('status','asset_count','source_block_count','sum_source_ply_gaussians','bbox_m','report_id')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

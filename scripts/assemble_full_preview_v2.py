#!/usr/bin/env python3
"""Lossless mixed-generation preview; never promotes a reconstruction candidate."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.dont_write_bytecode = True
from package_garage import (
    EXPECTED_IDS, _state, inspect_ply, local_path, merge_bounds,
    ply_header, quotas_for, read_json, scan_ply,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    'maps/garage_fidelity_v2_viewer/manifest.json',
    'maps/garage_fidelity_v2_block000_anneal_diagnostic/manifest.json',
    'maps/garage_fidelity_v2_joint015_diagnostic/manifest.json',
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class Output:
    def __init__(self, path, names, count):
        self.path, self.count = path, count
        self.stride = len(names) * 4
        self.h = hashlib.sha256()
        self.handle = path.open('xb')
        header = ply_header(names, count)
        self.header_bytes, self.payload_bytes = len(header), 0
        self.handle.write(header)
        self.h.update(header)

    def append(self, raw):
        assert len(raw) % self.stride == 0
        assert self.handle.write(raw) == len(raw)
        self.h.update(raw)
        self.payload_bytes += len(raw)

    def finish(self, bbox):
        self.handle.flush()
        self.handle.close()
        assert self.payload_bytes == self.count * self.stride
        assert self.path.stat().st_size == self.header_bytes + self.payload_bytes
        return dict(path=str(self.path.relative_to(ROOT)), gaussians=self.count,
                    file_bytes=self.path.stat().st_size, sha256=self.h.hexdigest(),
                    header_bytes=self.header_bytes, bbox_m=bbox)


def assemble(destination, overview_budget):
    start = time.monotonic()
    out = local_path(ROOT, destination)
    assert not out.exists(), 'Use a new output directory; existing models are never overwritten'
    assert out.parent.is_dir() and 28 <= overview_budget <= 1_500_000
    snapshots, pins = {}, {}

    def pinned_json(name):
        p = local_path(ROOT, name)
        snapshots[p] = _state(p)
        value = read_json(p, pins)
        assert _state(p) == snapshots[p]
        return value

    base, zero, joint = [pinned_json(p) for p in MANIFESTS]
    for m in (base, zero, joint):
        assert m['coordinates'] == 'camera_init' and m['units'] == 'm'
        assert m['dataset_kind'] == 'real' and m['camera_up'] == base['camera_up']
        assert m['renderer'] == {'schema_version': 1, 'mode': 'inria_2dgs_v3'}
    originals = {b['id']: b for b in base['blocks']}
    assert len(originals) == len(base['blocks']) == 28 and set(originals) == set(EXPECTED_IDS)
    assert len(zero['blocks']) == len(joint['blocks']) == 1
    assert zero['blocks'][0]['source_blocks'] == ['block_000']
    assert joint['blocks'][0]['source_blocks'] == ['block_014', 'block_015', 'block_016']
    inventory = pinned_json(base['source_inventory']['path'])
    assert pins[ROOT / base['source_inventory']['path']] == base['source_inventory']['sha256']
    old_sources = {v['id']: v for v in inventory['sources']}
    sources = []
    for bid in EXPECTED_IDS:
        if bid in ('block_014', 'block_016'):
            continue
        is_new = bid in ('block_000', 'block_015')
        block = copy.deepcopy(zero['blocks'][0] if bid == 'block_000' else
                              joint['blocks'][0] if bid == 'block_015' else originals[bid])
        cores = block.get('source_blocks', [bid])
        expected = block['sha256'] if is_new else old_sources[bid]['sha256']
        p = local_path(ROOT, block['path'])
        with p.open('rb') as f:
            spec = inspect_ply(f)
        assert spec['count'] == block['training']['gaussians']
        assert p.stat().st_size == len(spec['header']) + spec['count'] * spec['stride']
        snapshots[p] = _state(p)
        sources.append(dict(id=bid, core_ids=cores, block=block, path=p, spec=spec,
                            state=snapshots[p], expected_sha256=expected,
                            generation='current_round' if is_new else 'historical'))
    flattened = [bid for s in sources for bid in s['core_ids']]
    assert len(sources) == 26 and len(flattened) == 28 and set(flattened) == set(EXPECTED_IDS)
    names = sources[0]['spec']['names']
    assert all(s['spec']['names'] == names for s in sources)
    counts = {bid: s['spec']['count'] // len(s['core_ids']) for s in sources for bid in s['core_ids']}
    core_quotas = dict(zip(EXPECTED_IDS, quotas_for([counts[b] for b in EXPECTED_IDS], overview_budget)))
    total = sum(s['spec']['count'] for s in sources)
    overview_total = sum(core_quotas.values())
    assert shutil.disk_usage(out.parent).free > (total + overview_total) * len(names) * 4 + 1024**3
    out.mkdir()
    full = overview = None
    try:
        full = Output(out / 'gaussians_full.ply', names, total)
        overview = Output(out / 'gaussians_overview.ply', names, overview_total)
        full_bbox = overview_bbox = None
        offset = overview_offset = 0
        records, detail_blocks = [], []
        for index, source in enumerate(sources):
            quota = sum(core_quotas[bid] for bid in source['core_ids'])
            info = scan_ply(source, 8 * 1024**2, full=full.append,
                            overview=overview.append, quota=quota)
            assert info['sha256'] == source['expected_sha256'], 'Source PLY identity changed'
            pins[source['path']] = info['sha256']
            full_bbox = merge_bounds(full_bbox, info['bbox_m'])
            overview_bbox = merge_bounds(overview_bbox, info['overview_bbox_m'])
            record = dict(source_index=index, id=source['id'], source_blocks=source['core_ids'],
                          generation=source['generation'], path=str(source['path'].relative_to(ROOT)),
                          sha256=info['sha256'], gaussians=source['spec']['count'],
                          source_header_bytes=len(source['spec']['header']), record_stride_bytes=source['spec']['stride'],
                          full_row_start=offset, full_row_stop=offset+source['spec']['count'],
                          overview_row_start=overview_offset, overview_row_stop=overview_offset+quota,
                          overview_gaussians=quota, bbox_m=info['bbox_m'])
            records.append(record)
            block = source['block']
            block.update(path=record['path'], sha256=record['sha256'], bbox_m=record['bbox_m'],
                         source_blocks=record['source_blocks'], model_generation=source['generation'],
                         high_fidelity_accepted=False, floor_verified=False)
            block['label'] = ('014–016 · 本轮联合模型' if source['id'] == 'block_015' else
                              source['id'].replace('block_', '') + (' · 本轮新模型' if source['generation']=='current_round' else ' · 历史模型'))
            for key in ('lidar', 'direction_support'):
                block.pop(key, None)
            block['eval']['scope_note'] = 'Source-only score; not the mixed full-scene or overview score'
            if source['id'] == 'block_006':
                block['training']['actual_adam_updates_verified'] = 11001
                block['training']['historical_resume_correction'] = 'reports/historical_block006_resume_audit.json'
            detail_blocks.append(block)
            offset += source['spec']['count']
            overview_offset += quota
            print(json.dumps(dict(source=source['id'], covered=source['core_ids'], count=record['gaussians'])), flush=True)
        full_info = full.finish(full_bbox)
        overview_info = overview.finish(overview_bbox)
        assert offset == total and overview_offset == overview_total
        for p, previous in snapshots.items():
            assert _state(p) == previous, 'Input changed during packaging: ' + str(p)
        coverage = dict(expected_source_blocks=list(EXPECTED_IDS), covered_source_blocks=sorted(flattened),
                        covered_source_block_count=28, model_asset_count=26, source_coverage_complete=True,
                        new_source_blocks=['block_000', 'block_014', 'block_015', 'block_016'],
                        historical_source_block_count=24, floor_labels_verified=False,
                        scope='All 28 prepared collection regions, not every raw bag message or unobserved building surface')
        common = dict(schema_version=1, coordinates='camera_init', units='m', camera_up=base['camera_up'],
                      dataset_kind='real', renderer=base['renderer'], status='mixed_generation_preview',
                      high_fidelity_accepted=False, rollout_allowed=False, diagnostic_only=True,
                      calibration_verified=False, native_fullscope_evaluated=False,
                      unobserved_building_complete=False, floor_assignment='inherited_unverified',
                      lidar_overlay_enabled=False, direction_support_enabled=False, coverage=coverage,
                      summary=dict(covered_source_blocks=28, model_assets=26, full_gaussians=total,
                                   current_round_source_blocks=4, historical_source_blocks=24,
                                   high_fidelity_accepted=False, full_scene_metrics=None))
        detail = dict(common, title='六层全场 · 新旧模型拼接 · 原始分区细节',
                      description='覆盖全部28个采集区块；000及014–016为本轮模型，其余24块为历史模型。按需加载原始高斯；未通过高保真验收。',
                      blocks=detail_blocks)
        ov_block = dict(id='garage_overview', label='全部28区块 · 六层总览（精简显示）', floor_id=None,
                        coordinates='camera_init', path=overview_info['path'], sha256=overview_info['sha256'],
                        bbox_m=full_bbox, sampled_bbox_m=overview_bbox, source_blocks=list(EXPECTED_IDS),
                        training=dict(status='trained', gaussians=overview_total,
                                      method='Unmodified sampled records from 26 actual trained source exports; no new training'),
                        eval=dict(status='not_evaluated', scope='Sampled overview only; no whole-scene score claimed'))
        overview_manifest = dict(common, title='六层完整采集范围 · 全场总览',
                                 description='总览精简抽样显示全部28区块；原始完整模型和分区细节另存。新旧模型混合，未通过高保真验收。',
                                 blocks=[ov_block], full_artifact=full_info)
        write_json(out/'manifest.json', detail)
        write_json(out/'overview_manifest.json', overview_manifest)
        record = dict(common, created_at=datetime.datetime.now().astimezone().isoformat(),
                      full=full_info, overview=overview_info, sources=records,
                      processing=dict(cpu_only=True, attributes_modified=False, ownership_cropping=False,
                                      deduplication=False, coordinate_transform=False, opacity_rescaled=False,
                                      full_record_preservation=True, new_training=False,
                                      overview_sampling='Balanced quota per canonical region; three quotas combined for joint014-016; source index floor((2*j+1)*count/(2*quota)), j=0..quota-1'),
                      provenance=dict(full='Contiguous exact source payload bytes in sources[] order; row ranges are half-open',
                                      overview='Contiguous per-source midpoint samples; formula and source indices reconstruct every selected record'),
                      properties=list(names), input_sha256={str(p.relative_to(ROOT)):v for p,v in pins.items()},
                      implementation_sha256=sha(Path(__file__)),
                      resource_usage=dict(elapsed_seconds=time.monotonic()-start,
                                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        write_json(out/'assembly.json', record)
        return record
    except BaseException as exc:
        for stream in (full, overview):
            if stream is not None and not stream.handle.closed:
                stream.handle.close()
        write_json(out/'FAILED.json', dict(error=repr(exc), published=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='maps/garage_fidelity_v2_full_preview')
    parser.add_argument('--overview-gaussians', type=int, default=1200000)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Do not disable validation assertions with -O')
    record = assemble(args.output, args.overview_gaussians)
    print(json.dumps({k:record[k] for k in ('status','coverage','full','overview','resource_usage')}, ensure_ascii=False))

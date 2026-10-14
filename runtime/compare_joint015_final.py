"""Paired descriptive comparison of the completed joint experiment."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    data = path.read_bytes()
    return json.loads(data), dict(path=relative, sha256=hashlib.sha256(data).hexdigest())


def compare(candidate_path, baseline_path, label):
    candidate, cp = read(candidate_path)
    baseline, bp = read(baseline_path)
    assert candidate['status'] == baseline['status'] == 'evaluated'
    assert candidate['acceptance_plan_id'] == baseline['acceptance_plan_id']
    key = lambda r: (r['group_id'], r['camera_id'])
    left = {key(r): r for r in candidate['frames']}
    right = {key(r): r for r in baseline['frames']}
    assert len(left) == len(candidate['frames']) and left.keys() == right.keys()
    rows = []
    for identity, r in left.items():
        b = right[identity]
        assert r['camera'] == b['camera']
        for field in ('source_image_path', 'acceptance_mask_path', 'source_lidar_depth_path', 'source_lidar_support_path'):
            assert r[field] == b[field], field
        x, y = r['protocols']['ego_v2'], b['protocols']['ego_v2']
        assert x['rgb_valid_pixels'] == y['rgb_valid_pixels']
        rows.append(dict(comparison=label, group_id=identity[0], camera_id=identity[1],
            dataset_index=r['dataset_index'], candidate_psnr_db=x['psnr_db'],
            baseline_psnr_db=y['psnr_db'], paired_delta_db=x['psnr_db']-y['psnr_db'],
            candidate_ssim=x['ssim_valid_pixels'], baseline_ssim=y['ssim_valid_pixels']))
    delta = [r['paired_delta_db'] for r in rows]
    return dict(comparison=label, candidate=cp, baseline=bp, images=len(rows),
        acceptance_plan_id=candidate['acceptance_plan_id'],
        candidate_psnr_db=statistics.mean(r['candidate_psnr_db'] for r in rows),
        baseline_psnr_db=statistics.mean(r['baseline_psnr_db'] for r in rows),
        paired_delta_db_mean=statistics.mean(delta), paired_delta_db_median=statistics.median(delta),
        frames_below_minus_1db=sum(d < -1 for d in delta), frames_improved=sum(d > 0 for d in delta),
        paired_delta_db_min=min(delta), paired_delta_db_max=max(delta), rows=rows)


def main():
    comparisons = []
    for block in ('block_014', 'block_015', 'block_016'):
        for baseline, label in (('original_single_blocks', 'original_single'), ('baseline_old_all28', 'old_whole_scene')):
            comparisons.append(compare(f'runs/fidelity_v2/joint015_aa_cap003_acceptance/{block}/metrics.json',
                f'runs/fidelity_v2/{baseline}/{block}/metrics.json', f'{block}_vs_{label}'))
    comparisons.append(compare('runs/fidelity_v2/joint015_aa_cap003_resolution/metrics.json',
        'runs/fidelity_v2/original015_resolution_depthv3/metrics.json', 'block_015_1280_vs_original_single'))
    output = ROOT / 'reports/joint015_aa_cap003_paired_comparison.json'
    report = dict(status='PAIRED_DESCRIPTIVE_COMPARISON', high_fidelity_accepted=False,
        scope='Fixed DEV views and ego_v2 protocol; geometry/whole-scene acceptance separate', comparisons=comparisons)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    rows = [r for comparison in comparisons for r in comparison['rows']]
    with output.with_suffix('.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('rows', 'candidate', 'baseline')} for r in comparisons], indent=2))


if __name__ == '__main__':
    main()

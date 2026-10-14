"""Correct depth statistic labels using cached contributors, without PLY rescan."""
import sys
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from assemble_core_gaussians import InputGuard,write_json,resource_preflight

out=ROOT/'reports/core_assembly_redwall_aa_depth_rev2_v2_20261009.json'
if out.exists():
    raise FileExistsError(out)
preflight=resource_preflight(out,disk_bytes=4*1024**2,memory_bytes=128*1024**2)
guard=InputGuard(ROOT)
original_path=ROOT/'reports/core_assembly_redwall_aa_depth_v2_20261009.json'
earlier_expected=ROOT/'reports/core_assembly_redwall_aa_depth_expected_v2_20261009.json'
r=guard.json(original_path)
guard.snapshot(earlier_expected)
r['revision']=2
r['status']='nine_rays_forward_AA_exposure_expected_depth_comparison_rev2'
r['preflight']=preflight
r['correction']={
    'old_report':{'path':original_path.relative_to(ROOT).as_posix(),'sha256':guard.files[original_path]['sha256']},
    'earlier_expected_report':{'path':earlier_expected.relative_to(ROOT).as_posix(),'sha256':guard.files[earlier_expected]['sha256']},
    'issue':'Old max_native_depth_abs_error compared CPU median with native exported expected depth. That cross-statistic difference is NOT a depth parity measurement.',
    'fixed':'Recompute sum(weight * ray_plane_depth) / sum(weight) from every cached composited contributor and compare against SHA-pinned native float depth.',
    'median_policy':'Keep the CPU median solely as a separately named diagnostic; do not compare it as native parity.',
    'old_reports_modified':False,'ply_rescanned':False,'kernel_modified':False,'gpu_used':False}
r['convention']['native_depth_comparison']='Expected versus expected: sum(weight * ray_plane_depth) / sum(weight), meters; native RGB+ED model output.'
summary={}
for variant,v in r['variants'].items():
    base=('runs/fidelity_v2/original_noownership015/block_015' if variant=='noownership'
          else 'runs/fidelity_v2/directional015/block_015')
    fp=ROOT/base/'008_float.npz'
    guard.snapshot(fp)
    assert guard.files[fp]['sha256']==r['inputs'][fp.relative_to(ROOT).as_posix()]['sha256']
    with np.load(fp,allow_pickle=False) as f:
        native_depth=np.array(f['depth'])
    previous_cross_statistic=v.pop('max_native_depth_abs_error')
    for ray in v['rays']:
        records=ray['composited_records']
        weights=np.array([c['weight'] for c in records],dtype=np.float64)
        depth=np.array([c['ray_plane_depth_m'] for c in records],dtype=np.float64)
        assert weights.sum()>0
        expected=float(np.sum(weights*depth)/weights.sum())
        x,y=ray['pixel_xy']
        actual=float(native_depth[y,x,0])
        assert actual==ray.pop('native_exported_depth')
        median=ray.pop('cpu_native_rule_median_depth')
        ray.pop('depth_abs_error')
        ray['cpu_expected_depth_m']=expected
        ray['native_expected_depth_m']=actual
        ray['expected_depth_abs_error_m']=abs(expected-actual)
        ray['cpu_median_depth_m_diagnostic']=median
        ray['median_minus_native_expected_m_not_parity']=median-actual
        assert abs(weights.sum()-ray['composited_alpha'])<1e-12
    v['max_expected_depth_abs_error_m']=max(ray['expected_depth_abs_error_m'] for ray in v['rays'])
    v['legacy_max_cross_statistic_difference_m_not_parity']=previous_cross_statistic
    assert v['max_expected_depth_abs_error_m']<1e-3
    assert sum(c['AA'] for ray in v['rays'] for c in ray['composited_records'])==v['AA']['contributors']
    summary[variant]={'composited_contributor_occurrences':sum(ray['contributors'] for ray in v['rays']),
        'AA_contributor_occurrences':v['AA']['contributors'],'AA_weight':v['AA']['weight'],
        'max_expected_depth_abs_error_m':v['max_expected_depth_abs_error_m'],
        'max_RGB_abs_error_unchanged':v['max_rgb_abs_error'],
        'legacy_cross_statistic_difference_m_not_parity':previous_cross_statistic}
guard.check()
r['summary']=summary
r['inputs_verified_this_revision']=guard.report()
r['limitations'].append('Legacy median-versus-expected difference is retained only to document the correction; it is not renderer depth error.')
write_json(out,r)
print(json.dumps(summary,indent=2))

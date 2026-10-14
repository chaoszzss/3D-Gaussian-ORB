"""Fixed-camera local ablations; descriptive600-step geometry, no rollout claim."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
names=['block000_seed_aa_cap003_acceptance','block000_opacity600_control','block000_opacity600_soft09','block000_original_cap003_diagnostic','block000_measuredseed_cap003_diagnostic']
reports={s:json.load(open(ROOT/'runs/fidelity_v2'/s/'metrics.json')) for s in names}
base=reports[names[0]];keys=['group_id','camera_id','camera','source_image_path','acceptance_mask_path','source_lidar_depth_path','source_lidar_support_path']
for name,d in reports.items():
 assert d['acceptance_plan_id']==base['acceptance_plan_id'] and len(d['frames'])==20
 for f,b in zip(d['frames'],base['frames']):
  assert all(f[k]==b[k] for k in keys),(name,f['dataset_index'])
pairs=[('opacity_initialization','block000_opacity600_control','block000_opacity600_soft09'),('measured_seed_addition','block000_original_cap003_diagnostic','block000_measuredseed_cap003_diagnostic')]
rows=[];comparisons=[]
for label,a,b in pairs:
 values=[]
 for fa,fb in zip(reports[a]['frames'],reports[b]['frames']):
  delta=fb['protocols']['ego_v2']['psnr_db']-fa['protocols']['ego_v2']['psnr_db'];values.append(delta)
  rows.append({'comparison':label,'index':fa['dataset_index'],'group_id':fa['group_id'],'camera_id':fa['camera_id'],'control_psnr_db':fa['protocols']['ego_v2']['psnr_db'],'candidate_psnr_db':fb['protocols']['ego_v2']['psnr_db'],'paired_delta_db':delta})
 comparisons.append({'name':label,'control':a,'candidate':b,'paired_psnr_mean_delta_db':float(np.mean(values)),'paired_psnr_median_delta_db':float(np.median(values)),'frames_improved':int(np.sum(np.array(values)>0)),'alpha_coverage_delta':reports[b]['protocols']['ego_v2']['alpha_gt_0_5_valid_pixel_fraction']-reports[a]['protocols']['ego_v2']['alpha_gt_0_5_valid_pixel_fraction']})
configs=[next((ROOT/'runs/fidelity_v2'/('train_block000_opacity600_'+s)).rglob('config.yml')) for s in ['control','soft09']]
a,b=[yaml.load(p.read_text(),Loader=yaml.BaseLoader) for p in configs]
def diffs(x,y,p=''):
 if isinstance(x,dict) and isinstance(y,dict):
  for k in sorted(set(x)|set(y)):yield from diffs(x.get(k),y.get(k),p+'.'+k)
 elif x!=y:yield {'path':p,'control':x,'candidate':y}
diff=list(diffs(a,b));assert {d['path'] for d in diff}=={'.output_dir','.pipeline.model.training_audit_path','.pipeline.model.warm_opacity_cap','.timestamp'}
geometry={}
for name in names[1:]:
 path=ROOT/'reports'/('surface_quality_v2_'+name+('_diagnostic' if 'opacity600' in name else '')+'.json')
 g=json.load(open(path));geometry[name]={'source':str(path.relative_to(ROOT)),'source_sha256':sha(path),'gate_evidence':g['gate_evidence'],'representation_gate_compatible':g['render_measurement']['gate_compatible_representation'],'interpretation':'Known measured numbers retained even when strict provenance gate status is unknown. The independent600-step CPU export proof verified all parameters but is incomplete for refinement by design.' if 'opacity600' in name else 'Static addition/footprint probe; not trained.'}
artifacts={s:{'path':'runs/fidelity_v2/'+s+'/metrics.json','sha256':sha(ROOT/'runs/fidelity_v2'/s/'metrics.json'),'checkpoint_step':d['step'],'gaussians':d['gaussians'],'ego_v2':d['protocols']['ego_v2']} for s,d in reports.items()}
report={'status':'LOCAL_PAIRED_DIAGNOSTICS_MEASURED','high_fidelity_accepted':False,'promotion_allowed':False,'frames_per_arm':20,'acceptance_plan_id':base['acceptance_plan_id'],'frame_identity_fields_verified':keys,'comparisons':comparisons,'actual_training_config_differences':diff,'metrics':artifacts,'geometry':geometry,'limitations':['600iterations and zero refine are intentionally insufficient for full-training gate; strict no-refine proof rejection was preserved.','Same-LIO teacher, limited725floor proxy pixels; no independent floor truth.','Static seed comparison measures an untrained addition; no claim of training or complete ground.','No complete408candidate or boundary evaluation.']}
path=ROOT/'reports/block000_local_ablation_comparison.json';assert not path.exists();path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'reports/block000_local_ablation_comparison.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(comparisons,indent=2))

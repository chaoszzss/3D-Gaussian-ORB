"""Historical CPU checkpoint audit prompted by the verified resume binding bug."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import json,hashlib,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/garage_surface_v1/block_006';result=RUN/'training_result.json';r=json.load(open(result));check=sorted(RUN.rglob('step-*.ckpt'))[-1]
st=torch.load(check,map_location='cpu',weights_only=False);rows={}
for key,value in st['optimizers'].items():rows[key]=[dict(step=int(s['step']),exp_avg_shape=list(s['exp_avg'].shape)) for s in value['state'].values()]
last=[v['step'] for group in rows.values() for v in group]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
report=dict(status='HISTORICAL_RESUME_ADAM_COUNTER_AUDIT',block_id='block_006',training_result_path=str(result),training_result_sha256=sha(result),checkpoint=str(check),checkpoint_sha256=sha(check),checkpoint_step=int(st['step']),declared_iterations=r['iterations_requested'],resumed_from=r['resumed_from'],optimizers=rows,all_optimizer_steps_match_final_iteration=all(n==int(st['step'])+1 for n in last),all_optimizer_steps_unchanged_from_resume=all(n==r['resumed_from']['checkpoint_step']+1 for n in last),gpu_used=False,original_files_modified=False,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,script_sha256=sha(Path(__file__)),interpretation='Frozen historical appearance measurements and exported parameter equality remain valid. Completed loop count alone does not prove optimizer progress. Compare exact Adam counters with saved step; hard surface projection may still mutate means/quats without Adam updates.',scope='Only historical training_result.json files explicitly marking resumed_from were searched; one original block006 matched. No claim that every possible undocumented resume is detected.')
out=ROOT/'reports/historical_block006_resume_audit.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('checkpoint_step','declared_iterations','all_optimizer_steps_match_final_iteration','all_optimizer_steps_unchanged_from_resume','optimizers')}))

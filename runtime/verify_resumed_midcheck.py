"""Read-only CPU evidence that resumed live parameters/Adam advance past OOM."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import json,hashlib,gc,time,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
job=ROOT/'runs/fidelity_v2/jobs/block000_measuredseed_anneal_v1/recovery_allocator_v1';run=ROOT/'runs/fidelity_v2/train_block000_measuredseed_anneal_v1'
paths=[job/'step-000002000.ckpt',next(run.rglob('step-000004000.ckpt'))];rows=[]
def hash_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def digest_tensor(t):
 a=t.detach().contiguous().numpy();view=memoryview(a).cast('B');h=hashlib.sha256()
 for i in range(0,len(view),8*1024**2):h.update(view[i:i+8*1024**2])
 return h.hexdigest()
for expected,p in zip((2000,4000),paths):
 before=p.stat();st=torch.load(p,map_location='cpu',weights_only=False);assert st['step']==expected
 params={k.rsplit('.',1)[-1]:v for k,v in st['pipeline'].items() if '.gauss_params.' in k};opt={}
 for key,value in st['optimizers'].items():
  assert len(value['state'])==1
  state=next(iter(value['state'].values()));step=int(state['step']);assert step==expected+1,(key,step)
  assert state['exp_avg'].shape==params[key].shape and state['exp_avg_sq'].shape==params[key].shape
  opt[key]=dict(step=step,exp_avg_sha256=digest_tensor(state['exp_avg']),exp_avg_sq_sha256=digest_tensor(state['exp_avg_sq']))
 row=dict(checkpoint=str(p),step=expected,gaussians=len(params['means']),parameters={k:digest_tensor(v) for k,v in params.items()},optimizers=opt)
 del state,st,params;gc.collect();row['checkpoint_sha256']=hash_file(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);rows.append(row)
changed={k:rows[0]['parameters'][k]!=rows[1]['parameters'][k] for k in rows[0]['parameters']};assert all(changed.values())
report=dict(status='PASS_RESUMED_PARAMETER_AND_ADAM_PROGRESS_NOT_FIDELITY',checkpoints=rows,all_six_parameter_families_changed=changed,adam_steps_advance_2001_to_4001=True,restored_binding_log=str(run/'resume_events.jsonl'),source_script_sha256=hash_file(Path(__file__)),gpu_used=False,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,limitations=['Parameter order/count changes through refinement; hashes establish change, not corresponding-row geometric displacement.','Midrun checkpoint4000 may be removed by upstream latest-only checkpoint saving; hashes and scalar evidence retained.','This is recovery validation, not the final15k training/heldout fidelity acceptance.'])
out=ROOT/'reports/block000_measuredseed_anneal_resume_midcheck.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','all_six_parameter_families_changed','adam_steps_advance_2001_to_4001','peak_rss_mb')}))

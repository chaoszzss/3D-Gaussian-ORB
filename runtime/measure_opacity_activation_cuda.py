import fcntl,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(3)
rows=[]
with (ROOT/'runtime/core_joint_gpu.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.cuda.reset_peak_memory_stats()
 for name in ['train_block000_depthv3_seed_aa_cap003','train_block000_opacity600_control','train_block000_opacity600_soft09']:
  result=json.load(open(ROOT/'runs/fidelity_v2'/name/'training_result.json'));path=ROOT/result['checkpoint'];checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
  p=checkpoint['pipeline'];keys=[k for k in p if k.endswith('gauss_params.opacities')];assert len(keys)==1
  logits=p[keys[0]].to('cuda').detach().requires_grad_(True);alpha=logits.sigmoid();vjp,=torch.autograd.grad(alpha.sum(),logits)
  rows.append({'run':name,'step':int(checkpoint['step']),'recorded_checkpoint_sha256':result['checkpoint_sha256'],'n':logits.numel(),'exact_one_count':int((alpha==1).sum()),'zero_vjp_count':int((vjp==0).sum()),'vjp_median':float(vjp.median()),'opacity_median':float(alpha.median()),'opacity_gt_099_fraction':float((alpha>.99).float().mean()),'logit_median':float(logits.median())})
  del logits,alpha,vjp,checkpoint,p
 report={'status':'MEASURED_ACTUAL_CUDA_FLOAT32_ACTIVATION','device':torch.cuda.get_device_name(),'dtype':'float32','optimizer_steps':0,'parameters_modified':False,'scope':'Actual torch sigmoid/logit VJP on complete saved opacity arrays; not rasterized per-pixel visibility or proof opacity softening improves reconstruction','rows':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
path=ROOT/'reports/block000_opacity_activation_cuda.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

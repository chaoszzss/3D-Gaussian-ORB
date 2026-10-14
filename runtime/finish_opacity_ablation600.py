"""Finish the paired diagnostic; retain the no-refine verifier's incomplete gate status."""
import datetime,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from orbvi_gaussian.training import confined_environment
JOB=ROOT/'runs/fidelity_v2/jobs/block000_opacity_ablation600_v2';PYTHON=str(ROOT/'.venv-train/bin/python')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def status(stage,**kw):(JOB/'recovery_status.json').write_text(json.dumps(dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),high_fidelity_accepted=False,**kw),indent=2)+'\n')
pin=json.load(open(JOB/'input_snapshot.json'))
assert all(sha(ROOT/p)==v for p,v in pin['sources'].items())
env=confined_environment();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
def execute(label,cmd,gpu=True,proof=False):
 cmd=list(map(str,cmd));status('running',operation=label,command=cmd)
 with (ROOT/'runtime/core_joint_gpu.lock').open('a+') as lock:
  if gpu:fcntl.flock(lock,fcntl.LOCK_EX)
  with (JOB/(label+'.log')).open('x') as log:
   r=subprocess.run([PYTHON,'-B',*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),) if gpu else ())
 if proof and r.returncode==2:
  p=json.load(open(ROOT/cmd[-1]));assert p['incomplete_reasons']==['no_observed_refinement_events'] and p['export_verified'] and p['optimizer_state_verified'] and p['step']==599
 elif r.returncode:raise RuntimeError((label,r.returncode))
 assert all(sha(ROOT/p)==v for p,v in pin['sources'].items())
def geometry(name):
 stem='block000_opacity600_'+name;out='runs/fidelity_v2/'+stem
 # No promotion is possible: fixed topology is intentional for this mechanism probe.
 # Keep the strict checkpoint proof as incomplete, and compute descriptive geometry
 # without claiming its strict gate-compatible parameter/export binding.
 execute(name+'_geometry_diagnostic',['scripts/measure_surface_quality.py','--ply',out+'/gaussians.ply','--metrics',out+'/metrics.json','--reference','data/garage_surface_v1/block_000/lidar_reference.npz','--baseline-metrics','runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json','--acceptance-plan','reports/acceptance_plan_v1.json','--output','reports/surface_quality_v2_'+stem+'_diagnostic.json'],gpu=False)
try:
 control=json.load(open(ROOT/'reports/block000_opacity600_control_checkpoint.json'));assert control['step']==599 and control['export_verified'] and control['incomplete_reasons']==['no_observed_refinement_events']
 geometry('control')
 run='runs/fidelity_v2/train_block000_opacity600_soft09';out='runs/fidelity_v2/block000_opacity600_soft09';assert not (ROOT/run).exists()
 execute('soft09_train',['scripts/train_lidar_block.py','--dataset',ROOT/'data/garage_fidelity_depthv3/block_000','--output',ROOT/run,'--iterations',600,'--allow-candidate-calibration','--surface-splats','--renderer-inria','--inria-backend','aa-depth-vjp','--warm-ply',ROOT/'runs/fidelity_v2/block000_seed_aa_cap003_acceptance/gaussians.ply','--warm-opacity-cap',.9,'--max-gaussians',2500000,'--no-refine','--stop-split-at',0,'--gaussian-max-scale-m',.03,'--tangent-scale-target-m',.03,'--tangent-scale-weight',.001,'--depth-weight',.5,'--alpha-weight',.1,'--metric-depth-weight',.05,'--metric-depth-delta-m',.15,'--save-every',600,'--vis','tensorboard'])
 result=json.load(open(ROOT/run/'training_result.json'));assert result['checkpoint_step']==599 and result['gaussians']==2499616
 config=ROOT/result['configs'][0]
 execute('soft09_eval',['scripts/evaluate_gaussian_ply.py','--config',config,'--block-id','block_000','--dataset-override','data/garage_surface_v1/block_000','--acceptance-plan','reports/acceptance_plan_v1.json','--renderer-integration-proof','reports/inria_backend_integration_gpu_v3.json','--max-images',0,'--max-gaussians',2500000,'--export-ply','--save-float-arrays','--output',out])
 proof='reports/block000_opacity600_soft09_checkpoint.json'
 execute('soft09_proof',['scripts/verify_training_checkpoint.py','--training-dir',ROOT/run,'--block-id','block_000','--evaluation-report',out+'/metrics.json','--output',proof],gpu=False,proof=True)
 geometry('soft09');status('complete_pending_paired_review',strict_proofs='incomplete: intentionally no refinement; all exports and optimizer changes verified',promotion_allowed=False)
except Exception as e:status('failed',error=repr(e));raise

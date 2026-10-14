"""Paired600-step mechanism probe; not a rollout or full-fidelity candidate."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from orbvi_gaussian.training import confined_environment
PYTHON=str(ROOT/'.venv-train/bin/python');WARM=ROOT/'runs/fidelity_v2/block000_seed_aa_cap003_acceptance/gaussians.ply';DATA=ROOT/'data/garage_fidelity_depthv3/block_000'
JOB=ROOT/'runs/fidelity_v2/jobs/block000_opacity_ablation600'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(stage,**kw):write(JOB/'status.json',dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),high_fidelity_accepted=False,**kw))
def execute(label,cmd,gpu=True):
 status('running',operation=label,command=cmd)
 with (ROOT/'runtime/core_joint_gpu.lock').open('a+') as lock:
  if gpu:fcntl.flock(lock,fcntl.LOCK_EX)
  with (JOB/(label+'.log')).open('x') as log:
   subprocess.run([PYTHON,'-B',*map(str,cmd)],cwd=ROOT,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),) if gpu else ())
 assert all(sha(ROOT/p)==v for p,v in sources.items()),'Core changed during paired experiment'
if JOB.exists():raise FileExistsError(JOB)
JOB.mkdir(parents=True)
sources={}
for p in [*sorted((ROOT/'src/orbvi_gaussian').rglob('*.py')),*[ROOT/n for n in ['scripts/train_lidar_block.py','scripts/evaluate_gaussian_ply.py','scripts/evaluate_reconstruction.py','scripts/native_gaussian_visibility.py','runtime/run_block000_opacity_ablation.py']]]:
 rel=p.relative_to(ROOT);dest=JOB/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources[str(rel)]=sha(dest)
parent=json.load(open(ROOT/'runs/fidelity_v2/jobs/block000_depthv3_seed_aa_cap003/input_snapshot.json'))
inputs={p:v for p,v in parent['inputs'].items() if p.startswith('data/garage_fidelity_depthv3/block_000/')}
assert all(sha(ROOT/p)==v['sha256'] for p,v in inputs.items())
inputs[str(WARM.relative_to(ROOT))]={'sha256':sha(WARM),'bytes':WARM.stat().st_size}
write(JOB/'input_snapshot.json',{'sources':sources,'inputs':inputs,'paired_variable':'warm_opacity_cap 1.0 versus0.9; same checkpoint PLY, new Adam, seed42,600steps,no refinement; all other flags same','selection':'CPUfloat32 saturation plus true AA TRAIN gradient evidence; opacity cap is a hypothesis, not a proven correction','budget':2500000,'high_fidelity_accepted':False})
env=confined_environment();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
try:
 for name,cap in [('control',1.),('soft09',.9)]:
  stem='block000_opacity600_'+name;run=ROOT/'runs/fidelity_v2'/('train_'+stem);out='runs/fidelity_v2/'+stem
  cmd=['scripts/train_lidar_block.py','--dataset',DATA,'--output',run,'--iterations',600,'--allow-candidate-calibration','--surface-splats','--renderer-inria','--inria-backend','aa-depth-vjp','--warm-ply',WARM,'--warm-opacity-cap',cap,'--max-gaussians',2500000,'--no-refine','--stop-split-at',0,'--gaussian-max-scale-m',.03,'--tangent-scale-target-m',.03,'--tangent-scale-weight',.001,'--depth-weight',.5,'--alpha-weight',.1,'--metric-depth-weight',.05,'--metric-depth-delta-m',.15,'--save-every',600,'--vis','tensorboard']
  execute(name+'_train',cmd)
  result=json.load(open(run/'training_result.json'));assert result['checkpoint_step']==599 and result['gaussians']==2499616
  config=ROOT/result['configs'][0]
  execute(name+'_eval',['scripts/evaluate_gaussian_ply.py','--config',config,'--block-id','block_000','--dataset-override','data/garage_surface_v1/block_000','--acceptance-plan','reports/acceptance_plan_v1.json','--renderer-integration-proof','reports/inria_backend_integration_gpu_v3.json','--max-images',0,'--max-gaussians',2500000,'--export-ply','--save-float-arrays','--output',out])
  proof='reports/'+stem+'_checkpoint.json'
  execute(name+'_proof',['scripts/verify_training_checkpoint.py','--training-dir',run,'--block-id','block_000','--evaluation-report',out+'/metrics.json','--output',proof],gpu=False)
  execute(name+'_geometry',['scripts/measure_surface_quality.py','--ply',out+'/gaussians.ply','--metrics',out+'/metrics.json','--reference','data/garage_surface_v1/block_000/lidar_reference.npz','--baseline-metrics','runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json','--acceptance-plan','reports/acceptance_plan_v1.json','--checkpoint-proof',proof,'--output','reports/surface_quality_v2_'+stem+'.json'],gpu=False)
 status('complete_pending_paired_review',arms=['control','soft09'])
except Exception as exc:
 status('failed',error=repr(exc));raise

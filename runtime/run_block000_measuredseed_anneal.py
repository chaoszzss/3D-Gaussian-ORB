"""Supervised 15k measured-seed/annealed-scale pilot with fixed DEV evaluation."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from orbvi_gaussian.training import confined_environment
PYTHON=str(ROOT/'.venv-train/bin/python');STEM='block000_measuredseed_anneal_v1'
RUN=ROOT/'runs/fidelity_v2'/('train_'+STEM);JOB=ROOT/'runs/fidelity_v2/jobs'/STEM
DATA=ROOT/'data/garage_fidelity_depthv3/block_000';WARM=ROOT/'maps/garage_block000_measuredseed_v2_20261009/gaussians.ply'
PROOF=ROOT/'reports/inria_backend_integration_gpu_v4.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):write(JOB/'status.json',dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),high_fidelity_accepted=False,**kw))
if RUN.exists() or JOB.exists():raise FileExistsError('Preserve every previous experiment')
proof=json.load(open(PROOF));assert proof['status']=='PASS'
from probe_inria_backend_integration import integration_source_hashes
assert proof['source_sha256']==integration_source_hashes()
assert sha(WARM)=='c9914351c1fc5fed026d677f4bd1e06f71da3aff9dcaaa4c3cd6042c583147fc'
JOB.mkdir(parents=True);status('snapshotting')
sources={}
for p in [*sorted((ROOT/'src/orbvi_gaussian').rglob('*.py')),*[ROOT/n for n in ['scripts/train_lidar_block.py','scripts/evaluate_gaussian_ply.py','scripts/evaluate_reconstruction.py','scripts/native_gaussian_visibility.py','scripts/probe_inria_backend_integration.py','scripts/measure_surface_quality.py','scripts/verify_training_checkpoint.py','runtime/run_block000_measuredseed_anneal.py']]]:
 rel=p.relative_to(ROOT);dest=JOB/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources[str(rel)]=sha(dest)
parent=json.load(open(ROOT/'runs/fidelity_v2/jobs/block000_depthv3_seed_aa_cap003/input_snapshot.json'))
inputs={p:v for p,v in parent['inputs'].items() if p.startswith('data/garage_fidelity_depthv3/block_000/')}
assert all(sha(ROOT/p)==v['sha256'] for p,v in inputs.items())
for p in (WARM,PROOF,ROOT/'maps/garage_block000_measuredseed_v2_20261009/surface_seed_audit.json'):
 inputs[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
train=['scripts/train_lidar_block.py','--dataset',DATA,'--output',RUN,'--iterations',15000,'--allow-candidate-calibration','--surface-splats','--renderer-inria','--inria-backend','aa-depth-vjp','--warm-ply',WARM,'--warm-opacity-cap',1.,'--max-gaussians',2500000,'--refine-every',100,'--stop-split-at',14000,'--stop-screen-size-at',14000,'--split-screen-size',.02,'--gaussian-max-scale-m',.03,'--gaussian-initial-max-scale-m',.12,'--gaussian-scale-anneal-start',1000,'--gaussian-scale-anneal-end',10000,'--tangent-scale-target-m',.03,'--tangent-scale-weight',.001,'--depth-weight',.5,'--alpha-weight',.1,'--metric-depth-weight',.05,'--metric-depth-delta-m',.15,'--save-every',2000,'--vis','tensorboard']
write(JOB/'input_snapshot.json',dict(sources=sources,inputs=inputs,train_command=list(map(str,train)),initial_gaussians=558010,final_count_budget=2500000,selection='Measured surface additions improve matched static cap diagnostic +0.722449dB, +9.3086 percentage points alpha; gradual cap preserves initial support until refinement. Sparse TRAIN visual points excluded. Combined engineering pilot, not a factorial causal attribution.',validation='Fixed20 DEV at640 and1280; unchanged original LiDAR-reference geometry gates; no TEST fitting.',high_fidelity_accepted=False))
env=confined_environment();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
stop=threading.Event()
def monitor():
 with (JOB/'board_memory_samples.jsonl').open('x') as f:
  while not stop.is_set():
   r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),values=r.stdout.strip(),returncode=r.returncode))+'\n');f.flush();stop.wait(5)
def execute(label,args,gpu=True):
 assert all(sha(ROOT/p)==v for p,v in sources.items()),'Frozen source changed'
 args=list(map(str,args));status('queued',operation=label)
 with (ROOT/'runtime/core_joint_gpu.lock').open('a+') as lock:
  if gpu:fcntl.flock(lock,fcntl.LOCK_EX)
  status('running',operation=label,command=args,controller_pid=os.getpid())
  with (JOB/(label+'.log')).open('x') as log:
   subprocess.run([PYTHON,'-B',*args],cwd=ROOT,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),) if gpu else ())
 assert all(sha(ROOT/p)==v for p,v in sources.items()),'Frozen source changed'
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
 execute('train',train)
 result=json.load(open(RUN/'training_result.json'));assert result['checkpoint_step']==14999
 config=ROOT/result['configs'][0];out=ROOT/'runs/fidelity_v2'/STEM
 common=['scripts/evaluate_gaussian_ply.py','--config',config,'--block-id','block_000','--max-images',0,'--save-float-arrays','--max-gaussians',2500000,'--renderer-integration-proof',PROOF]
 execute('eval640',common+['--dataset-override','data/garage_surface_v1/block_000','--acceptance-plan','reports/acceptance_plan_v1.json','--export-ply','--output',out])
 execute('eval1280',common+['--dataset-override',DATA,'--acceptance-plan','reports/acceptance_resolution_depthv3_plan.json','--output',str(out)+'_resolution'])
 check=ROOT/'reports'/(STEM+'_checkpoint.json')
 execute('checkpoint_proof',['scripts/verify_training_checkpoint.py','--training-dir',RUN,'--block-id','block_000','--evaluation-report',out/'metrics.json','--output',check],gpu=False)
 execute('geometry',['scripts/measure_surface_quality.py','--ply',out/'gaussians.ply','--metrics',out/'metrics.json','--reference','data/garage_surface_v1/block_000/lidar_reference.npz','--baseline-metrics','runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json','--acceptance-plan','reports/acceptance_plan_v1.json','--checkpoint-proof',check,'--output',ROOT/'reports'/('surface_quality_v2_'+STEM+'.json')],gpu=False)
 status('evaluated_pending_main_review',run=str(RUN),checkpoint_step=14999)
except Exception as exc:status('failed',error=repr(exc));raise
finally:stop.set();thread.join(timeout=10)

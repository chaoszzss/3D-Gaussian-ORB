"""Resume the saved real model after OOM, then complete the frozen evaluations."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from orbvi_gaussian.training import confined_environment
from probe_inria_backend_integration import integration_source_hashes
PYTHON=str(ROOT/'.venv-train/bin/python');STEM='block000_measuredseed_anneal_v1'
RUN=ROOT/'runs/fidelity_v2'/('train_'+STEM);JOB=ROOT/'runs/fidelity_v2/jobs'/STEM;RECOVERY=JOB/'recovery_allocator_v1'
DATA=ROOT/'data/garage_fidelity_depthv3/block_000';PROOF=ROOT/'reports/inria_backend_integration_gpu_v4.json'
assert not RECOVERY.exists();RECOVERY.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):write(RECOVERY/'status.json',dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),high_fidelity_accepted=False,**kw))
pin=json.load(open(JOB/'input_snapshot.json'));sources=pin['sources'];assert all(sha(ROOT/p)==v for p,v in sources.items())
assert json.load(open(PROOF))['source_sha256']==integration_source_hashes()
cp=list(RUN.rglob('step-000002000.ckpt'));assert len(cp)==1
backup=RECOVERY/cp[0].name;shutil.copyfile(cp[0],backup);assert sha(backup)==sha(cp[0])
for name in ('refinement.json','launch.json'):shutil.copyfile(RUN/name,RECOVERY/('pre_resume_'+name))
extra={}
for path in ('scripts/resume_lidar_block.py','runtime/recover_block000_measuredseed_anneal.py','tests/test_resume_optimizer_binding.py'):
 src=ROOT/path;dst=RECOVERY/'source_snapshot'/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);extra[path]=sha(src)
write(RECOVERY/'input_snapshot.json',dict(original_job_input_sha256=sha(JOB/'input_snapshot.json'),resume_checkpoint=str(backup),resume_checkpoint_sha256=sha(backup),recovery_sources=extra,allocator='expandable_segments:True',reason='Split allocation OOM; preserve final count2.5M and original loss/scale schedule',resume_semantics='Restore live model, Adam, schedulers and scaler. Upstream does not store sampling RNG or strategy gradient accumulators; these restart. Real next step2001, total target15000. No claim bitwise-continuous training.'))
env=confined_environment();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
stop=threading.Event()
def monitor():
 with (RECOVERY/'board_memory_samples.jsonl').open('x') as f:
  while not stop.is_set():
   r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),values=r.stdout.strip(),returncode=r.returncode))+'\n');f.flush();stop.wait(5)
def execute(label,args,gpu=True):
 assert all(sha(ROOT/p)==v for p,v in {**sources,**extra}.items()),'Source changed after snapshot'
 args=list(map(str,args));status('queued',operation=label)
 with (ROOT/'runtime/core_joint_gpu.lock').open('a+') as lock:
  if gpu:fcntl.flock(lock,fcntl.LOCK_EX)
  status('running',operation=label,controller_pid=os.getpid(),command=args)
  with (RECOVERY/(label+'.log')).open('x') as log:subprocess.run([PYTHON,'-B',*args],cwd=ROOT,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),) if gpu else ())
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
 execute('resume',['scripts/resume_lidar_block.py','--output',RUN])
 result=json.load(open(RUN/'training_result.json'));assert result['checkpoint_step']==14999 and result['resumed_from']['restored_state']['next_training_step']==2001
 config=ROOT/result['configs'][0];out=ROOT/'runs/fidelity_v2'/STEM
 common=['scripts/evaluate_gaussian_ply.py','--config',config,'--block-id','block_000','--max-images',0,'--save-float-arrays','--max-gaussians',2500000,'--renderer-integration-proof',PROOF]
 execute('eval640',common+['--dataset-override','data/garage_surface_v1/block_000','--acceptance-plan','reports/acceptance_plan_v1.json','--export-ply','--output',out])
 execute('eval1280',common+['--dataset-override',DATA,'--acceptance-plan','reports/acceptance_resolution_depthv3_plan.json','--output',str(out)+'_resolution'])
 check=ROOT/'reports'/(STEM+'_checkpoint.json')
 execute('checkpoint_proof',['scripts/verify_training_checkpoint.py','--training-dir',RUN,'--block-id','block_000','--evaluation-report',out/'metrics.json','--output',check],False)
 execute('geometry',['scripts/measure_surface_quality.py','--ply',out/'gaussians.ply','--metrics',out/'metrics.json','--reference','data/garage_surface_v1/block_000/lidar_reference.npz','--baseline-metrics','runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json','--acceptance-plan','reports/acceptance_plan_v1.json','--checkpoint-proof',check,'--output',ROOT/'reports'/('surface_quality_v2_'+STEM+'.json')],False)
 assert all(sha(ROOT/p)==v for p,v in {**sources,**extra}.items())
 status('evaluated_pending_main_review',checkpoint_step=14999)
except Exception as exc:status('failed',error=repr(exc));raise
finally:stop.set();thread.join(timeout=10)

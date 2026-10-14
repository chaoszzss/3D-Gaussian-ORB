import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=str(ROOT/'.venv-train/bin/python')
for label in ['original','measuredseed']:
 stem='block000_'+label+'_cap003_diagnostic';out='reports/surface_quality_v2_'+stem+'.json'
 subprocess.run([P,'-B','scripts/measure_surface_quality.py','--ply','maps/garage_block000_'+label+'_cap003_diagnostic/gaussians.ply','--metrics','runs/fidelity_v2/'+stem+'/metrics.json','--reference','data/garage_surface_v1/block_000/lidar_reference.npz','--baseline-metrics','runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json','--acceptance-plan','reports/acceptance_plan_v1.json','--output',out],cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'3','MKL_NUM_THREADS':'3'},stdout=subprocess.DEVNULL,check=True)
 print(label,json.load(open(ROOT/out))['gate_evidence'],flush=True)

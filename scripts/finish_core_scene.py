#!/usr/bin/env python3
"""Follow one verified batch PID; assemble/evaluate all28, never certify fidelity."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from orbvi_gaussian.training import confined_environment
from orbvi_gaussian.io import output_path


def identity(pid):
    directory = Path('/proc')/str(pid)
    try:
        state = (directory/'stat').read_text().rsplit(')',1)[1].split()
        if state[0]=='Z': return None
        return {'pid':pid,'start_ticks':state[19],
                'argv':[x.decode(errors='replace') for x in (directory/'cmdline').read_bytes().split(b'\0') if x]}
    except FileNotFoundError: return None


def require_full_batch(status, plan):
    ids = ['core_%03d'%i for i in range(28)]
    if (status.get('complete') is not True or status.get('trained_exported_core_count')!=28
            or status.get('planned_core_ids')!=list(range(28))
            or any(status.get('cores',{}).get(k,{}).get('status')!='trained_exported' for k in ids)):
        raise ValueError('All28 verified trained exports are required; do not assemble a partial batch')
    covered = [k for source in plan.get('sources',[]) for k in source.get('core_ids',[])]
    if sorted(covered)!=['block_%03d'%i for i in range(28)]:
        raise ValueError('Actual source plan does not cover each28 core exactly once')


def atomic_record(path, data):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-for-pid',required=True,type=int)
    p.add_argument('--output',type=Path,default=ROOT/'maps/garage_core_all_v1')
    p.add_argument('--evaluation-output',type=Path,default=ROOT/'runs/full_scene_core_all_v1')
    p.add_argument('--state',type=Path,default=ROOT/'runtime/finish_core_scene_state.json')
    args=p.parse_args();destination=output_path(args.output);evaluation=output_path(args.evaluation_output);state=output_path(args.state)
    env=confined_environment();os.environ.update(env)
    initial=identity(args.wait_for_pid)
    if not initial or not any('train_core_joint_batch.py' in x for x in initial['argv']):
        raise ValueError('Follower requires the actual live core-joint controller')
    record={'status':'waiting_for_full28_batch','batch_identity':initial,'full_map':str(destination.relative_to(ROOT)),
            'evaluation':str(evaluation.relative_to(ROOT)),'started_local':datetime.now().isoformat(),
            'high_fidelity_accepted':False,'visual_review':'pending'}
    def update(status,**extra):
        record.update(status=status,updated_epoch_s=time.time(),**extra);atomic_record(state,record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    update('waiting_for_full28_batch')
    try:
        last=None
        while True:
            current=identity(args.wait_for_pid)
            if current is None:break
            if current!=initial:raise RuntimeError('Controller PID reused; inspect existing job')
            status=json.loads((ROOT/'runs/core_joint_v1/status.json').read_text())
            count=status['trained_exported_core_count']
            if count!=last:update('waiting_for_full28_batch',trained_exported_cores=count);last=count
            time.sleep(15)
        progress=json.loads((ROOT/'reports/CURRENT_RECONSTRUCTION_STATE.json').read_text())
        if progress.get('paused') is True:
            update('paused_by_user');return 0
        status=json.loads((ROOT/'runs/core_joint_v1/status.json').read_text())
        source_plan=ROOT/'runs/core_joint_v1/source_plan.json'
        if not source_plan.exists():raise RuntimeError('Batch ended without actual full source plan; inspect batch errors')
        require_full_batch(status,json.loads(source_plan.read_text()))
        record['trained_exported_cores']=28
        py=str(ROOT/'.venv-train/bin/python')
        update('assembling_full28')
        with (ROOT/'runtime/finish_core_scene_assembly.log').open('a') as log:
            subprocess.run([py,'-B','scripts/assemble_core_gaussians.py','--source-plan',str(source_plan),
                '--output',str(destination),'--support-radius-m','.2','--workers','1'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        assembly=json.loads((destination/'assembly.json').read_text())
        if assembly['coverage']['nonzero_core_count']!=28:raise RuntimeError('Assembly lacks nonzero artifacts for all28')
        record['full_gaussians']=assembly['full']['gaussians']
        update('evaluating_complete_scene_DEV')
        with (ROOT/'runtime/finish_core_scene_evaluation.log').open('a') as log:
            subprocess.run([py,'-B','scripts/evaluate_full_scene.py','--candidate-ply',str(destination/'gaussians_full.ply'),
                '--baseline-ply',str(ROOT/'maps/garage_full_v1/gaussians_full.ply'),'--output',str(evaluation),
                '--split','dev'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        update('full_source_DEV_ready_pending_visual_quality_review')
        return 0
    except Exception as error:
        update('incomplete_requires_review',error=str(error));return 1


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Resumeable real-garage block preparation, training, evaluation and map export."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from orbvi_gaussian.io import output_path
from orbvi_gaussian.training import confined_environment

MAP=Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release/czy_joint_raw_map.pcd')
FLOORS=Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v6/config/czy_all_floor_schedule.json')


def atomic_json(path,obj):
    path=output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def execute(cmd,log,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        result=subprocess.run([str(x) for x in cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Command failed (%d); see %s'%(result.returncode,log))


def floor_label(block,schedule):
    start,end=block['start_header_ns']/1e9,block['end_header_ns']/1e9
    span=max(.001,end-start)
    for floor in schedule['floors']:
        overlap=max(0,min(end,floor['end_stamp'])-max(start,floor['start_stamp']))
        if overlap/span>.7:return 'F%d'%floor['floor_id']
    for ramp in schedule['ramps']:
        if start<=ramp['end_stamp'] and end>=ramp['start_stamp']:
            return '坡道 F%d-F%d'%(ramp['lower_floor_id'],ramp['upper_floor_id'])
    return None


def prepare(block,args,env):
    out=ROOT/'data'/args.run_id/block['id']
    if (out/'preparation.json').exists():return out
    if out.exists():raise RuntimeError('Incomplete prepare output exists: %s; inspect before retry'%out)
    cmd=['/usr/bin/python3',ROOT/'scripts/orbvi_pipeline.py','prepare',
         '--extracted',ROOT/'data/keyframes_2hz','--calibration',args.calibration,
         '--pose-csv',ROOT/'data/all_lio_candidate/odometry.csv',
         '--block-plan',ROOT/'reports/lio_candidate_blocks.json','--block-id',block['id'],
         '--output',out,'--max-groups','0','--initialization-map',MAP,
         '--map-world-frame','camera_init','--map-alignment-verified','--map-voxel-m','0.06',
         '--max-points',str(args.initial_points),'--colorize-from-images','--save-depth',
         '--allow-candidate-calibration','--separate-test','--global-split']
    prep_env=env.copy()
    # ROS/system Python's existing SciPy is in the read-only user site; keep
    # training Python isolated while permitting its own existing dependencies.
    prep_env.pop('PYTHONNOUSERSITE',None)
    execute(cmd,ROOT/'runs'/args.run_id/'logs'/(block['id']+'_prepare.log'),prep_env)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',default='garage_lidar_v1')
    p.add_argument('--calibration',type=Path,default=ROOT/'configs/direct_lio_refined.experimental.yaml')
    p.add_argument('--iterations',type=int,default=10000)
    p.add_argument('--initial-points',type=int,default=100000)
    p.add_argument('--prepare-workers',type=int,default=2)
    p.add_argument('--blocks',help='comma-separated IDs; default all 28')
    p.add_argument('--depth-weight',type=float,default=.2)
    p.add_argument('--alpha-weight',type=float,default=.05)
    p.add_argument('--surface-splats',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--renderer-inria',action='store_true')
    args=p.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():p.error('Unsafe run-id')
    args.calibration=args.calibration.resolve()
    run=output_path(ROOT/'runs'/args.run_id);run.mkdir(parents=True,exist_ok=True)
    blocks=json.loads((ROOT/'reports/lio_candidate_blocks.json').read_text())['blocks']
    if args.blocks:
        ids=set(args.blocks.split(','));blocks=[b for b in blocks if b['id'] in ids]
        if len(blocks)!=len(ids):p.error('Unknown block ID')
    schedule=json.loads(FLOORS.read_text())
    env=confined_environment();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    manifest={'schema_version':1,'coordinates':'camera_init','units':'m','camera_up':[0,0,1],
              'dataset_kind':'real','title':'六层车库 · LiDAR约束Gaussian地图','status':'building',
              'run_id':args.run_id,'calibration_verified':False,
              'floor_assignment':'Estimated temporal floor schedule; not independently surveyed',
              'blocks':[]}
    previous=run/'map_manifest.json'
    if previous.exists():manifest=json.loads(previous.read_text())
    manifest['status']='preparing' if args.prepare_only else 'building'
    manifest.pop('pause',None)
    byid={b['id']:b for b in manifest['blocks']}
    for block in blocks:
        if block['id'] not in byid:
            b={'id':block['id'],'floor_id':floor_label(block,schedule),'floor_verified':False,
               'coordinates':'camera_init','bbox_m':{'min':block['bbox_min_m'],'max':block['bbox_max_m']},
               'path':str((run/block['id']/'eval_dev/gaussians.ply').relative_to(ROOT)),
               'training':{'status':'pending'},'eval':{'status':'not_evaluated'}}
            manifest['blocks'].append(b);byid[block['id']]=b
    def publish():
        manifest['updated_at_epoch_s']=time.time()
        atomic_json(previous,manifest);atomic_json(ROOT/'maps/garage_manifest.json',manifest)
    publish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.prepare_workers) as pool:
        futures={block['id']:pool.submit(prepare,block,args,env) for block in blocks}
        for index,block in enumerate(blocks):
            item=byid[block['id']]
            try:
                print('[%d/%d] %s: wait preparation'%(index+1,len(blocks),block['id']),flush=True)
                dataset=futures[block['id']].result()
                if args.prepare_only:
                    item['training']={'status':'prepared','dataset':str(dataset.relative_to(ROOT))};publish()
                    print('%s: prepared'%block['id'],flush=True)
                    continue
                train_dir=run/block['id']
                if not (train_dir/'training_result.json').exists():
                    item['training']={'status':'training','iterations':args.iterations,'calibration_verified':False};publish()
                    if train_dir.exists():
                        cmd=[ROOT/'.venv-train/bin/python',ROOT/'scripts/resume_lidar_block.py',
                             '--output',train_dir]
                    else:
                        cmd=[ROOT/'.venv-train/bin/python',ROOT/'scripts/train_lidar_block.py',
                             '--dataset',dataset,'--output',train_dir,'--iterations',str(args.iterations),
                             '--depth-weight',str(args.depth_weight),'--alpha-weight',str(args.alpha_weight),
                             '--allow-candidate-calibration']
                        if args.surface_splats:cmd.append('--surface-splats')
                        if args.renderer_inria:cmd.append('--renderer-inria')
                    execute(cmd,run/'logs'/(block['id']+'_train.log'),env)
                result=json.loads((train_dir/'training_result.json').read_text())
                configs=[ROOT/p for p in result.get('configs',[])] or list(train_dir.rglob('config.yml'))
                if len(configs)!=1:raise RuntimeError('Expected one checkpoint config')
                for split in ['dev','test']:
                    dest=train_dir/('eval_'+split)
                    if not (dest/'metrics.json').exists():
                        cmd=[ROOT/'.venv-train/bin/python',ROOT/'scripts/evaluate_reconstruction.py',
                             '--config',configs[0],'--split',split,'--output',dest]
                        if split=='dev':cmd.append('--export-ply')
                        execute(cmd,run/'logs'/(block['id']+'_eval_'+split+'.log'),env)
                dev=json.loads((train_dir/'eval_dev/metrics.json').read_text())
                test=json.loads((train_dir/'eval_test/metrics.json').read_text())
                item['bbox_m']=dev['export']['bbox_m']
                item['training']={'status':'trained','iterations':args.iterations,'gaussians':dev['export']['gaussians'],
                                  'calibration_verified':False,'algorithm':'LiDAR-depth-and-surface-constrained Splatfacto'}
                item['eval']={'status':'evaluated','split':'test','psnr_db':test['psnr_db_mean'],
                              'ssim':test['ssim_valid_mean'],'initializer_surface_p95_m':test['gaussian_to_initializer_p95_m'],
                              'report':str((train_dir/'eval_test/metrics.json').relative_to(ROOT)),
                              'geometry_protocol':test['geometry_protocol']}
                publish()
                print('%s: trained/exported, held-out PSNR %.2f SSIM %.3f, %d GS'%
                      (block['id'],test['psnr_db_mean'],test['ssim_valid_mean'],dev['export']['gaussians']),flush=True)
            except Exception as exc:
                item['training']['status']='failed';item['error']=str(exc);publish()
                print('%s failed: %s'%(block['id'],exc),flush=True)
    finished=[b for b in manifest['blocks'] if b['training']['status']=='trained']
    manifest['status']='all_blocks_prepared' if args.prepare_only else ('complete_models_pending_quality_acceptance' if len(finished)==len(blocks) else 'partial_failed')
    if finished:
        manifest['summary']={'trained_blocks':len(finished),'requested_blocks':len(blocks),
                             'gaussians':sum(b['training']['gaussians'] for b in finished),
                             'test_psnr_db_mean':sum(b['eval']['psnr_db'] for b in finished)/len(finished),
                             'test_ssim_mean':sum(b['eval']['ssim'] for b in finished)/len(finished)}
    publish();print(json.dumps(manifest.get('summary',{}),indent=2),flush=True)


if __name__=='__main__':main()

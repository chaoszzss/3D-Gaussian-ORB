#!/usr/bin/env python3
"""Dump real fixed-camera RGB/alpha/native-depth to distinguish missing coverage.

Run after the active core training/evaluation controller is terminal. This
does not fit parameters, alter source inputs or grant reconstruction acceptance.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from orbvi_gaussian.training import confined_environment
os.environ.update(confined_environment())
import cv2
import numpy as np
import torch
import evaluate_gaussian_ply as candidate
from orbvi_gaussian.io import new_output_dir,write_json
from nerfstudio.utils.eval_utils import eval_setup
from evaluate_full_scene import refuse_live_batch,lock


def dump(args):
    refuse_live_batch(ROOT)
    config_path=candidate.confined_path(args.config,file=True)
    ply_path=candidate.confined_path(args.ply,file=True)
    outpath=candidate.confined_path(args.output)
    if outpath.exists():raise FileExistsError('Diagnostic output already exists')
    parameters,info=candidate.read_candidate(ply_path,max_gaussians=12000000,max_ply_bytes=3*1024**3,extended_cpu_budget=True)
    with lock(ROOT/'runtime/core_joint_gpu.lock',wait=False,project=ROOT):
        config,pipeline,checkpoint,step=eval_setup(config_path,test_mode='val',
            update_config_callback=lambda c:candidate.validate_config(c,info['sh_degree'],'dev'))
        if getattr(pipeline.model.config,'optimize_shared_rig',False):raise ValueError('Require fixed source rig')
        candidate.require_identity_parser(pipeline.datamanager.train_dataparser_outputs)
        dataset=pipeline.datamanager.eval_dataset;data=Path(config.pipeline.datamanager.data).resolve()
        meta=json.loads((data/'transforms.json').read_text());frames={candidate.confined_path(data/f['file_path']):f for f in meta['frames']}
        indices=candidate.select_frame_indices(len(dataset),args.frame_indices,0)
        out=new_output_dir(outpath);records=[]
        from native_gaussian_visibility import native_visible_indices
        for index in indices:
            frame=frames[dataset.image_filenames[index].resolve()]
            camera=dataset.cameras[index:index+1].to(pipeline.device)
            camera_info=candidate.fixed_camera_identity(camera,frame,meta)
            active,visibility=native_visible_indices(parameters,camera,pipeline.device,200000)
            if len(active)>4000000:raise ValueError('Exact active set exceeds diagnostic GPU budget')
            candidate.replace_gauss_params(pipeline.model,{k:v[active] for k,v in parameters.items()},info['sh_degree'],torch)
            with torch.inference_mode():outputs=pipeline.model.get_outputs_for_camera(camera)
            torch.cuda.synchronize()
            rgb=outputs['rgb'].detach().cpu().numpy().clip(0,1)
            alpha=outputs['accumulation'].detach().cpu().numpy().squeeze()
            expected=outputs['depth'].detach().cpu().numpy().squeeze()
            median=pipeline.model.info['depth_median'].detach().cpu().numpy().squeeze()
            source=dataset[index];valid=np.ones(alpha.shape,bool) if 'mask' not in source else source['mask'].cpu().numpy().squeeze()>0
            reference=np.load(data/frame['lidar_depth_path'],allow_pickle=False)
            support=cv2.imread(str(data/frame['lidar_depth_mask_path']),0)>0
            if any(a.shape!=alpha.shape for a in [valid,expected,median,reference,support]):raise ValueError('Diagnostic planes differ from source camera')
            if not np.isfinite(alpha).all() or not np.isfinite(expected).all() or not np.isfinite(median).all():raise ValueError('Nonfinite diagnostic render')
            prefix='%03d'%index
            for name,array in [('alpha',alpha),('depth_expected_m',expected),('depth_median_m',median),('source_depth_m',reference),('source_support',support),('source_valid',valid)]:
                with (out/(prefix+'_'+name+'.npy')).open('xb') as stream:np.save(stream,array,allow_pickle=False)
            cv2.imwrite(str(out/(prefix+'_rgb.png')),cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/(prefix+'_alpha.png')),(alpha.clip(0,1)*255).astype(np.uint8))
            eligible=valid&support&(reference>0)
            row={'dataset_index':index,'group_id':frame['group_id'],'camera_id':frame['camera_id'],
                'camera':camera_info,'visibility':visibility,'valid_pixels':int(valid.sum()),
                'low_alpha_valid_pixels':int((valid&(alpha<.5)).sum()),
                'known_depth_pixels':int(eligible.sum()),'low_alpha_known_depth_pixels':int((eligible&(alpha<.5)).sum()),
                'scope':'Diagnostic evidence only; source LiDAR conditioned, not independent GT'}
            records.append(row);print(json.dumps(row),flush=True)
        report={'status':'actual_native_diagnostics_dumped','candidate':info,'template_checkpoint':str(checkpoint),
            'template_step':int(step),'source_dataset':str(data),'split':'dev','records':records,'high_fidelity_accepted':False}
        write_json(out/'diagnostics.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--ply',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame-indices',nargs='+',type=int,required=True);dump(p.parse_args())


if __name__=='__main__':main()

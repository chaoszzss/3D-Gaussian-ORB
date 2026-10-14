#!/usr/bin/env python3
"""Read-only native TRAIN parameter gradients, under an inherited GPU lock.

Queue with fd9 held by flock and wrap this process in timeout 290s. Only the
final completed trained checkpoint is accepted. No optimizers are constructed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def verify_run_completion(run,expected_step=11999):
    path=run/'training_result.json'
    if not path.is_file():
        raise ValueError('Missing training_result.json; require completed actual training')
    result=json.loads(path.read_text())
    if (result.get('status')!='trained' or result.get('parameter_origin')!='trained_checkpoint'
            or result.get('checkpoint_step')!=expected_step
            or result.get('iterations_completed')!=expected_step+1):
        raise ValueError('Require completed trained checkpoint at expected final step')
    checkpoint=(ROOT/result['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run.resolve()) or not checkpoint.is_file():
        raise ValueError('Final checkpoint must exist within the completed run')
    if sha(checkpoint)!=result.get('checkpoint_sha256'):
        raise ValueError('Final checkpoint SHA differs from training_result')
    configs=result.get('configs',[])
    if len(configs)!=1:
        raise ValueError('Require exactly one recorded training config')
    config=(ROOT/configs[0]).resolve()
    if not config.is_relative_to(run.resolve()) or not config.is_file():
        raise ValueError('Recorded training config missing/outside run')
    return result,checkpoint,config


def fixed_train_frames(metadata,names):
    train=set(metadata.get('train_filenames',[]))
    held=set(metadata.get('val_filenames',[]))|set(metadata.get('test_filenames',[]))
    if train&held or len(names)!=len(set(names)) or not 1<=len(names)<=3:
        raise ValueError('Require one to three unique disjoint TRAIN frames')
    by_name={frame['file_path']:frame for frame in metadata['frames']}
    if any(name not in train or name in held or name not in by_name for name in names):
        raise ValueError('Requested frame is not exclusively TRAIN')
    return [by_name[name] for name in names]


def gradient_summary(gradient,visible,center_known):
    import numpy as np
    import torch
    n=len(visible)
    if gradient is None:
        norms=np.zeros(n,dtype=np.float64)
    else:
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite actual parameter gradient')
        norms=gradient.detach().double().reshape(n,-1).square().sum(1).sqrt().cpu().numpy()
    masks=dict(all=np.ones(n,bool),native_visible=visible,center_on_known_support=center_known)
    output={}
    for key,mask in masks.items():
        selected=norms[mask]
        output[key]=dict(gaussians=len(selected),nonzero_gaussians=int((selected>0).sum()),
            l2=float(np.linalg.norm(selected)),per_gaussian_norm_quantiles={
                str(q):float(np.quantile(selected,q)) if len(selected) else None for q in (0.,.5,.9,.95,.99,1.)})
    return dict(connected=gradient is not None,finite=True,**output)


def require_inherited_lock(fd):
    held=os.fstat(fd);expected=(ROOT/'runtime/core_joint_gpu.lock').stat()
    if (held.st_ino,held.st_dev)!=(expected.st_ino,expected.st_dev):
        raise ValueError('Inherited descriptor is not runtime/core_joint_gpu.lock')
    # Same inherited open-file description keeps ownership across the entire
    # timeout/Python lifetime. A direct invocation cannot silently bypass it.
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


def geometry_prior_scalars(parameters, reference, nearest, config, step):
    """Final-parameter diagnostics; nearest cache itself is not checkpointed."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    means=parameters['means']
    values={}
    with torch.no_grad():
        if config.lidar_surface_anchor_weight>0:
            target=torch.as_tensor(reference['xyz'][nearest],device=means.device,dtype=means.dtype)
            distance=torch.linalg.vector_norm(means-target,dim=-1)
            values['lidar_surface_anchor_loss']=float(config.lidar_surface_anchor_weight*F.smooth_l1_loss(
                distance/.1,torch.zeros_like(distance),beta=1.))
            if config.surface_splats:
                raw=reference['normals'].astype('float32').copy()
                raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
                normal=torch.as_tensor(raw[nearest],device=means.device,dtype=means.dtype)
                valid=torch.as_tensor(reference['normal_valid'][nearest],device=means.device).bool()
                q=F.normalize(parameters['quats'],dim=-1,eps=1e-12)
                w,x,y,z=q.unbind(-1)
                actual=torch.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],dim=-1)
                error=(1-(actual*normal).sum(-1).square()).clamp_min(0)
                values['lidar_surface_normal_loss']=float(config.surface_normal_weight*error[valid].mean()) if valid.any() else 0.
        if config.use_scale_regularization:
            scales=parameters['scales'].exp()
            active=.1*(scales.amax(-1)/scales.amin(-1)-config.max_gauss_ratio).clamp_min(0).mean()
            values['scale_reg_if_active_step']=float(active)
            values['scale_reg']=float(active) if step%10==0 else 0.
        else:
            values['scale_reg']=0.
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError('Nonfinite additional geometry prior scalar')
    return values


def audit(args):
    require_inherited_lock(args.lock_fd)
    started=time.monotonic()
    run=args.run.resolve()
    result,checkpoint,config_path=verify_run_completion(run,args.expected_step)
    if args.output.exists():
        raise FileExistsError(args.output)
    selection=json.loads(args.selection_report.read_text())
    names=[row['file_path'] for row in selection['selected_train_frames']][:args.max_frames]
    # Heavy imports and all GPU operations occur only after lock verification.
    from orbvi_gaussian.training import confined_environment
    os.environ.update(confined_environment())
    import numpy as np
    from PIL import Image
    import torch
    import yaml
    from pytorch_msssim import SSIM
    from scipy.spatial import cKDTree
    from orbvi_gaussian.inria_surfel_backend import render_surfel,nerfstudio_c2w_to_cv_w2c
    from orbvi_gaussian.lidar_splatfacto import compute_lidar_losses,tangent_scale_penalties
    sys.path.insert(0,str(ROOT/'scripts'))
    from measure_surface_quality import unproject,classify_reference,DEFAULT_SETTINGS

    torch.set_num_threads(4)
    config=yaml.load(config_path.read_text(),Loader=yaml.Loader)
    model_config=config.pipeline.model
    if (not model_config.use_inria_surfel_renderer or model_config.optimize_shared_rig
            or model_config.camera_optimizer.mode!='off' or model_config.use_bilateral_grid
            or model_config.num_downscales!=0):
        raise ValueError('Audit requires fixed native cameras/full resolution/no bilateral grid')
    parser_config=config.pipeline.datamanager.dataparser
    if (parser_config.orientation_method!='none' or parser_config.center_method!='none'
            or parser_config.auto_scale_poses or parser_config.scale_factor!=1.
            or parser_config.downscale_factor!=1
            or config.pipeline.datamanager.camera_res_scale_factor!=1.):
        raise ValueError('Audit requires identity metric parser and prepared resolution')
    data=Path(config.pipeline.datamanager.data).resolve()
    metadata=json.loads((data/'transforms.json').read_text())
    frames=fixed_train_frames(metadata,names)
    reference=dict(np.load(data/'lidar_reference.npz',allow_pickle=False))
    tree=cKDTree(reference['xyz'])
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if int(state['step'])!=args.expected_step:
        raise ValueError('Checkpoint payload actual step differs from completion record')
    source={key.rsplit('.',1)[-1]:value for key,value in state['pipeline'].items() if '.gauss_params.' in key}
    if len(source['means'])!=result['gaussians']:
        raise ValueError('Checkpoint Gaussian count differs from completion record')
    sh_degree=min(args.expected_step//model_config.sh_degree_interval,model_config.sh_degree)
    if sh_degree!=model_config.sh_degree:
        raise ValueError('Require full trained SH active at final step')
    parameter_names=('means','scales','opacities','quats')
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    parameters={key:torch.nn.Parameter(value.to(device),requires_grad=key in parameter_names)
                for key,value in source.items()}
    _,nearest_prior=tree.query(source['means'].numpy(),workers=4)
    prior_scalars=geometry_prior_scalars(parameters,reference,nearest_prior,model_config,args.expected_step)
    del nearest_prior
    report=dict(status='RUNNING_TRAIN_PARAMETER_GRADIENT_AUDIT',training_complete=True,
        audited_training_step=args.expected_step,optimizer_steps=0,parameters_changed=False,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=result['checkpoint_sha256'],
        config=str(config_path.relative_to(ROOT)),config_sha256=sha(config_path),
        selection_report=str(args.selection_report),selection_report_sha256=sha(args.selection_report),
        split='TRAIN',sh_degree=sh_degree,gaussians=len(source['means']),frames=[],
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__),ROOT/'src/orbvi_gaussian/inria_surfel_backend.py',ROOT/'src/orbvi_gaussian/lidar_splatfacto.py')},
        loss_settings=dict(relative_weight=model_config.lidar_depth_loss_weight,alpha_weight=model_config.lidar_alpha_loss_weight,
            diagnostic_metric_weight=.05,diagnostic_metric_delta_m=.15,
            checkpoint_metric_weight=model_config.lidar_metric_depth_loss_weight,
            checkpoint_metric_delta_m=model_config.lidar_metric_depth_huber_delta_m,
            tangent_weight=model_config.tangent_scale_loss_weight,tangent_target_m=model_config.tangent_scale_target_m),
        additional_prior_settings=dict(anchor_weight=model_config.lidar_surface_anchor_weight,
            normal_weight=model_config.surface_normal_weight,distortion_weight=model_config.surface_distortion_weight,
            use_scale_regularization=model_config.use_scale_regularization,max_gauss_ratio=model_config.max_gauss_ratio),
        additional_prior_protocol='Anchor/normal recomputed using fresh nearest references at final parameters: training nearest cache (updated every 250 steps) is not checkpointed, so these are diagnostics, not exact historical step11999 cached losses. Distortion uses actual native map. Scale regularization preserves config and step%10 activation. Scalars only; no extra VJP budget.',
        scope='Actual native autograd to raw checkpoint parameters on fixed TRAIN images. No optimizer/refine callbacks or parameter projection. Same RGB/known-depth masks; all-GS tangent prior. Per-GS centre-on-support is a diagnostic subset, not a footprint visibility proof. LIO-conditioned teacher is not independent GT.',
        gradient_coordinates={'means':'world metres','scales':'log tangent/third axes','opacities':'logit','quats':'raw wxyz before renderer normalization'},
        rendering_protocol='Original native full-GS renderer, complete SH, unmodified checkpoint values. Same stochastic training background seeded 42+frame index; no in-place training clamps or optimizer states.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ssim=SSIM(data_range=1.,size_average=True,channel=3).to(device)
    for frame_index,frame in enumerate(frames):
        if time.monotonic()-started>235 and report['frames']:
            report['stopped_before_next_frame_for_time_budget']=True
            break
        torch.manual_seed(42+frame_index)
        gt=torch.from_numpy(np.asarray(Image.open(data/frame['file_path']).convert('RGB')).copy()).to(device).float()/255
        rgb_mask=np.asarray(Image.open(data/frame['mask_path']).convert('L'))>0
        target=np.load(data/frame['lidar_depth_path'],allow_pickle=False)
        known=(np.asarray(Image.open(data/frame['lidar_depth_mask_path']).convert('L'))>0)&rgb_mask&np.isfinite(target)&(target>0)
        h,w=gt.shape[:2]
        assert known.shape==target.shape==(h,w)==(frame['h'],frame['w'])
        # Class evidence selects no render-dependent pixels.
        cam=dict(camera_to_worlds_ns=np.asarray(frame['transform_matrix'])[:3].tolist(),
                 fl_x=frame['fl_x'],fl_y=frame['fl_y'],cx=frame['cx'],cy=frame['cy'],w=w,h=h)
        ids=np.flatnonzero(known)
        world=unproject(cam,ids,target.ravel()[ids],.5)
        distance,nearest=tree.query(world,workers=4)
        labels=classify_reference(reference,nearest,distance,frame['transform_matrix'][2][3],DEFAULT_SETTINGS)
        # Block000 diagnostic selects a TRAIN group before reading render errors; all classes included.
        c2w=torch.tensor(frame['transform_matrix'],dtype=torch.float32,device=device)
        view=nerfstudio_c2w_to_cv_w2c(c2w[None])
        K=torch.tensor([[[frame['fl_x'],0,frame['cx']],[0,frame['fl_y'],frame['cy']],[0,0,1.]]],device=device)
        colors=torch.cat((parameters['features_dc'][:,None,:],parameters['features_rest']),dim=1)
        rendered=render_surfel(parameters['means'],parameters['quats'],parameters['scales'].exp(),
            parameters['opacities'].sigmoid(),colors,view,K,w,h,sh_degree=sh_degree,backend=getattr(model_config,'inria_backend','original'))
        alpha=rendered.alpha[0];depth=rendered.depth_expected[0]
        background=(torch.rand(3,device=device) if model_config.background_color=='random' else
                    torch.ones(3,device=device) if model_config.background_color=='white' else torch.zeros(3,device=device))
        rgb=(rendered.rgb[0]+(1-alpha)*background).clamp(0,1)
        mask=torch.from_numpy(rgb_mask).to(device)[...,None]
        losses=compute_lidar_losses(dict(depth=depth,accumulation=alpha),torch.from_numpy(target).to(device),
            torch.from_numpy(known).to(device),model_config.lidar_depth_loss_weight,
            model_config.lidar_alpha_loss_weight,wmetric=.05,metric_delta_m=.15)
        ll1=((rgb-gt)*mask).abs().mean()
        simloss=1-ssim((rgb*mask).permute(2,0,1)[None],(gt*mask).permute(2,0,1)[None])
        losses['main_loss']=(1-model_config.ssim_lambda)*ll1+model_config.ssim_lambda*simloss
        tangent,_=tangent_scale_penalties(parameters['scales'],model_config.tangent_scale_target_m)
        losses['tangent_scale_loss']=model_config.tangent_scale_loss_weight*tangent
        visible=rendered.info['radii'][0].detach().cpu().numpy()>0
        with torch.no_grad():
            camera_points=parameters['means']@view[0,:3,:3].T+view[0,:3,3]
            uv=camera_points@K[0].T;uv=uv[:,:2]/uv[:,2:3]
            inside=torch.isfinite(uv).all(1)&(camera_points[:,2]>.2)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            center_known=torch.zeros(len(visible),dtype=torch.bool,device=device)
            pixels=uv[inside].floor().long()
            center_known[inside]=torch.from_numpy(known).to(device)[pixels[:,1],pixels[:,0]]
            center_known=center_known.cpu().numpy()&visible
        del camera_points,uv,inside,pixels
        row=dict(file_path=frame['file_path'],group_id=frame['group_id'],camera_id=frame['camera_id'],
            rgb_valid_pixels=int(rgb_mask.sum()),known_depth_pixels=int(known.sum()),floor_support_pixels=int((labels==0).sum()),
            native_visible_gaussians=int(visible.sum()),center_on_known_support_gaussians=int(center_known.sum()),
            alpha_mean_on_known=float(alpha[...,0][torch.from_numpy(known).to(device)].detach().mean()),losses={})
        with torch.no_grad():
            distortion=float(model_config.surface_distortion_weight*rendered.distortion.mean())
        if not np.isfinite(distortion):
            raise ValueError('Nonfinite distortion scalar')
        row['additional_geometry_loss_scalars']={**prior_scalars,'surfel_distortion_loss':distortion}
        for name in ('main_loss','lidar_depth_loss','lidar_metric_depth_loss','lidar_alpha_loss','tangent_scale_loss'):
            loss=losses[name]
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss '+name)
            gradients=torch.autograd.grad(loss,[parameters[key] for key in parameter_names],
                retain_graph=True,allow_unused=True)
            row['losses'][name]=dict(value=float(loss.detach()),parameters={
                key:gradient_summary(gradient,visible,center_known) for key,gradient in zip(parameter_names,gradients)})
            del gradients
        if any(not torch.equal(parameters[key].detach().cpu(),source[key]) for key in parameters):
            raise RuntimeError('Audit changed raw checkpoint parameters')
        row['parameters_bitwise_equal_checkpoint']=True
        report['frames'].append(row)
        report['elapsed_seconds']=time.monotonic()-started
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(frame=frame['file_path'],floor_pixels=row['floor_support_pixels'],
                              losses={key:value['value'] for key,value in row['losses'].items()})),flush=True)
        del row,loss,losses,rendered,colors,rgb,alpha,depth,gt,tangent,ll1,simloss
        torch.cuda.empty_cache()
    if not report['frames']:
        raise RuntimeError('No completed actual TRAIN gradient audit frame')
    if sha(checkpoint)!=result['checkpoint_sha256']:
        raise RuntimeError('Checkpoint bytes changed during audit')
    if any(sha(ROOT/name)!=expected for name,expected in report['source_sha256'].items()):
        raise RuntimeError('Audit source changed during the diagnostic')
    report.update(status='ACTUAL_TRAIN_PARAMETER_GRADIENTS_MEASURED',frames_completed=len(report['frames']),
        requested_frames=len(frames),elapsed_seconds=time.monotonic()-started,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_sha256_verified_after_measurement=result['checkpoint_sha256'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],frames=report['frames_completed'],output=str(args.output))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--selection-report',type=Path,default=ROOT/'reports/lidar_loss_units_v3_cpu.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-step',type=int,default=11999)
    parser.add_argument('--max-frames',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--lock-fd',type=int,default=9)
    audit(parser.parse_args())


if __name__=='__main__':
    main()

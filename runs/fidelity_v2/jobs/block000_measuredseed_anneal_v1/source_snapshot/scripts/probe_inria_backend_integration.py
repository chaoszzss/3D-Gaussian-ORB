#!/usr/bin/env python3
"""Tiny real model/backend TRAIN VJP probe; no optimizer, under project GPU lock."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def integration_source_hashes():
    paths=(Path(__file__),ROOT/'src/orbvi_gaussian/inria_surfel_backend.py',
           ROOT/'src/orbvi_gaussian/lidar_splatfacto.py',ROOT/'scripts/train_lidar_block.py',
           ROOT/'src/orbvi_gaussian/scale_annealing.py',
           ROOT/'scripts/evaluate_reconstruction.py',ROOT/'scripts/evaluate_gaussian_ply.py',
           ROOT/'scripts/native_gaussian_visibility.py')
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--lock-fd',type=int,default=9)
    args=parser.parse_args()
    from audit_train_parameter_gradients import require_inherited_lock
    require_inherited_lock(args.lock_fd)
    if args.output.exists():raise FileExistsError(args.output)
    started=time.monotonic()
    from orbvi_gaussian.training import confined_environment
    os.environ.update(confined_environment())
    import torch
    from types import SimpleNamespace
    from pytorch_msssim import SSIM
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
    from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModel,LidarSplatfactoModelConfig
    from orbvi_gaussian.inria_surfel_backend import inria_backend_provenance
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    device='cuda'
    c2w=torch.tensor([[[1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,-1.,0.]]],device=device)
    camera=Cameras(camera_to_worlds=c2w,fx=64.,fy=64.,cx=16.5,cy=16.5,width=32,height=32).to(device)
    angle=math.radians(85)/2
    means=torch.tensor([[0.,0.,3.]],device=device)
    quats=torch.tensor([[math.cos(angle),0.,math.sin(angle),0.]],device=device)
    initial=dict(means=means,quats=quats,scales=torch.tensor([[.03,.03,.01]],device=device).log(),
        opacities=torch.logit(torch.tensor([[.45]],device=device)),
        features_dc=torch.tensor([[.1,.2,.3]],device=device),features_rest=torch.zeros(1,15,3,device=device))
    # Activate a nonzero high-order SH coefficient to exercise unchanged full SH.
    initial['features_rest'][0,14,0]=.1
    results=[];images=[];rgb_grads=[]
    def make_model(variant):
        model=LidarSplatfactoModel.__new__(LidarSplatfactoModel)
        torch.nn.Module.__init__(model)
        model.device_indicator_param=torch.nn.Parameter(torch.empty(0,device=device),requires_grad=False)
        model.config=LidarSplatfactoModelConfig(use_inria_surfel_renderer=True,inria_backend=variant,
            num_downscales=0,sh_degree=3,sh_degree_interval=1,background_color='black',
            surface_splats=True,lidar_surface_anchor_weight=0.,gaussian_max_scale_m=.12,
            lidar_depth_loss_weight=.5,lidar_alpha_loss_weight=.1,use_scale_regularization=False,
            tangent_scale_loss_weight=.001)
        model.gauss_params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.clone()) for k,v in initial.items()})
        model.step=11999;model.crop_box=None;model._nearest_surface=None
        model.background_color=torch.zeros(3,device=device)
        model.camera_optimizer=CameraOptimizerConfig(mode='off').setup(num_cameras=1,device=device)
        model.strategy=SimpleNamespace(absgrad=False,step_pre_backward=lambda *args:None)
        model.strategy_state={};model.optimizers={}
        model.ssim=SSIM(data_range=1.,size_average=True,channel=3).to(device)
        return model.train()
    target=torch.zeros(32,32,1,device=device);target[16,17]=3.
    known=target>0
    batch=dict(image=torch.full((32,32,3),.3,device=device),mask=torch.ones(32,32,1,device=device),
        lidar_depth=target,lidar_depth_mask=known)
    for variant in ('original','aa-depth-vjp'):
        model=make_model(variant)
        outputs=model.get_outputs(camera)
        assert model.info['inria_backend']==variant
        losses=model.get_loss_dict(outputs,batch)
        names=('means','scales','opacities','quats')
        params=[model.gauss_params[k] for k in names]
        gradients=torch.autograd.grad(losses['lidar_depth_loss'],params,retain_graph=True)
        color_gradients=torch.autograd.grad(losses['main_loss']+losses['lidar_alpha_loss'],params)
        assert all(torch.isfinite(g).all() for g in (*gradients,*color_gradients))
        rgb_grads.append([g.detach().clone() for g in color_gradients])
        images.append({k:outputs[k].detach().clone() for k in ('rgb','depth','accumulation')})
        scalar=float(losses['lidar_depth_loss'].detach())
        analytic=float(gradients[0][0,0]);epsilon=.001
        samples=[]
        with torch.no_grad():
            for sign in (-1,1):
                model.means[0,0]=sign*epsilon
                out=model.get_outputs(camera)
                samples.append(float(model.get_loss_dict(out,batch)['lidar_depth_loss']))
            model.means[0,0]=0.
        fd=(samples[1]-samples[0])/(2*epsilon)
        if variant=='aa-depth-vjp':assert abs(analytic-fd)<.002
        else:assert abs(analytic-fd)>.1
        assert all(torch.equal(model.gauss_params[k].detach(),value) for k,value in initial.items())
        results.append(dict(variant=variant,model_step=model.step,active_sh_degree=3,
            known_pixels=int(known.sum()),relative_depth_loss=scalar,means_x_autograd=analytic,
            means_x_finite_difference=fd,parameter_gradients_finite=True,parameters_restored_bitwise=True,
            parameter_gradients={k:g.detach().cpu().tolist() for k,g in zip(names,gradients)},
            renderer_provenance=inria_backend_provenance(variant)))
        del model,outputs,losses,gradients,color_gradients
    equal=all(torch.equal(images[0][k],images[1][k]) for k in images[0])
    grad_equal=all(torch.equal(x,y) for x,y in zip(*rgb_grads))
    assert equal and grad_equal
    report=dict(status='PASS',gpu_used=True,lock='runtime/core_joint_gpu.lock',
        command=[sys.executable,*sys.argv],optimizer_steps=0,training_iterations=0,
        parameters_restored_bitwise=True,forward_model_images_bitwise_equal=equal,
        rgb_alpha_parameter_gradients_bitwise_equal=grad_equal,cases=results,
        source_sha256=integration_source_hashes(),
        elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Real LidarSplatfactoModel.get_outputs/get_loss_dict with one Gaussian, full SH3, fixed tiny TRAIN camera. No optimizer/refinement steps. Local backend integration only, not reconstruction acceptance.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds','forward_model_images_bitwise_equal','rgb_alpha_parameter_gradients_bitwise_equal')}),flush=True)


if __name__=='__main__':main()

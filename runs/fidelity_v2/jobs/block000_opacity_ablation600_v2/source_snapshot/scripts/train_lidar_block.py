#!/usr/bin/env python3
"""Train Gaussian appearance with sparse LiDAR geometry in a fixed metric frame."""
import argparse
import copy
import dataclasses
import json
import hashlib
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from orbvi_gaussian.training import confined_environment, splatfacto_command
os.environ.update(confined_environment())
from orbvi_gaussian.io import new_output_dir
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.scripts.train import main as train_main
from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModelConfig,LidarFullImageDatamanagerConfig,warm_start_read_budget


def fields_of(obj, excluded):
    return {f.name:copy.deepcopy(getattr(obj,f.name)) for f in dataclasses.fields(obj) if f.name not in excluded}


def parser_for_cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--iterations',type=int,default=12000)
    p.add_argument('--depth-weight',type=float,default=.2)
    p.add_argument('--alpha-weight',type=float,default=.05)
    p.add_argument('--metric-depth-weight',type=float,default=0.,
                   help='Opt-in additive metre Huber on the same known depth pixels; zero preserves original losses')
    p.add_argument('--metric-depth-delta-m',type=float,default=.15,
                   help='Metre Huber transition; HuberLoss, not SmoothL1 normalized by delta')
    p.add_argument('--resolution-schedule',type=int,default=1500)
    p.add_argument('--allow-candidate-calibration',action='store_true')
    p.add_argument('--surface-splats',action='store_true',help='Thin Gaussian disks initialized along measured local surface normals')
    p.add_argument('--renderer-2dgs',action='store_true',help='Perspective-correct ray/surfel intersection rendering')
    p.add_argument('--renderer-inria',action='store_true',help='Original 2DGS with differentiable true ray-plane depth')
    p.add_argument('--inria-backend',choices=('original','aa-depth-vjp'),default='original',
                   help='Explicit native backward variant; aa-depth-vjp requires pinned CUDA validation and preserves original forward')
    p.add_argument('--optimize-shared-rig',action='store_true',help='Train four time-shared extrinsics with differentiable 3D renderer')
    p.add_argument('--warm-ply',type=Path,help='Actual trained Gaussian PLY used before new optimizers are created')
    p.add_argument('--warm-opacity-cap',type=float,default=1.,help='Optional initial opacity softening for joint optimization')
    p.add_argument('--max-gaussians',type=int,default=500000)
    p.add_argument('--no-refine',action='store_true',help='Keep warm Gaussian record count fixed during the joint trial')
    p.add_argument('--refine-every',type=int,default=100)
    p.add_argument('--stop-split-at',type=int,help='Exclusive refinement stop; default iterations-1000')
    p.add_argument('--stop-screen-size-at',type=int,default=4000,help='Exclusive stop for screen-size-driven split/cull, in real training steps')
    p.add_argument('--split-screen-size',type=float,default=.05,help='Split if native screen radius / max(width,height) exceeds this fraction')
    p.add_argument('--warmup-length',type=int,default=500)
    p.add_argument('--reset-opacity',action='store_true',help='Explicitly enable periodic low-alpha reset; warm pilot preserves opacity by default')
    p.add_argument('--gaussian-max-scale-m',type=float,default=.12)
    p.add_argument('--tangent-scale-target-m',type=float,default=.03)
    p.add_argument('--tangent-scale-weight',type=float,default=.001,help='Weak metric tangent prior, zero disables')
    p.add_argument('--tangent-screen-weight',type=float,default=0.,help='Opt-in screen footprint prior; disabled until an ablation justifies it')
    p.add_argument('--tangent-max-diameter-px',type=float,default=2.5)
    p.add_argument('--num-downscales',type=int,default=0,help='0 supervises prepared full resolution from step zero')
    p.add_argument('--camera-optimizer-mode',choices=('off','SO3xR3','SE3'),default='off')
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--viewer-port',type=int,help='Optional local viewer websocket port for this run')
    p.add_argument('--vis',choices=('viewer','tensorboard','viewer+tensorboard'),default='viewer',
                   help='Training visualization/logger; tensorboard records per-term training losses')
    return p


def validate_args(args):
    if args.warm_ply is not None:
        warm_start_read_budget(args.max_gaussians)
    if not (0 <= args.metric_depth_weight < float('inf') and 0 < args.metric_depth_delta_m < float('inf')):
        raise ValueError('Metric depth weight must be finite nonnegative and delta-m finite positive')
    if 'tensorboard' in args.vis:
        # Fail before creating a run if this optional local dependency is absent.
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    if args.save_every<1 or args.max_gaussians<1:raise ValueError('Checkpoint interval and Gaussian budget must be positive')
    if args.viewer_port is not None and not 1 <= args.viewer_port <= 65535:
        raise ValueError('Viewer port must be in 1..65535')
    if args.iterations < 1 or args.refine_every < 1 or args.warmup_length < 0:
        raise ValueError('Iterations/refine interval must be positive; warmup must be nonnegative')
    if args.stop_screen_size_at < 0 or not 0 < args.split_screen_size <= 1:
        raise ValueError('Screen-size stop must be nonnegative and split fraction in (0,1]')
    if not 0 < args.warm_opacity_cap <= 1 or not 0 < args.gaussian_max_scale_m < float('inf'):
        raise ValueError('Opacity cap must be in (0,1]; scale bound must be finite and positive')
    if args.num_downscales < 0 or args.resolution_schedule < 1:
        raise ValueError('Invalid resolution schedule')
    if not (0 < args.tangent_scale_target_m < float('inf') and 0 < args.tangent_max_diameter_px < float('inf')
            and 0 <= args.tangent_scale_weight < float('inf') and 0 <= args.tangent_screen_weight < float('inf')):
        raise ValueError('Tangent targets must be finite positive; loss weights finite nonnegative')
    if args.tangent_screen_weight and not (args.renderer_2dgs or args.renderer_inria):
        raise ValueError('Screen tangent constraint requires a surface renderer')
    if args.renderer_2dgs and args.renderer_inria:
        raise ValueError('Choose exactly one surface renderer')
    if args.inria_backend != 'original' and not args.renderer_inria:
        raise ValueError('--inria-backend aa-depth-vjp requires --renderer-inria')
    if args.renderer_inria and (args.camera_optimizer_mode != 'off' or args.optimize_shared_rig):
        raise ValueError('Original 2DGS ABI has no camera gradients; use --camera-optimizer-mode off')
    if args.optimize_shared_rig and (args.renderer_2dgs or args.camera_optimizer_mode != 'off'):
        raise ValueError('Shared rig optimization requires the 3D renderer and camera optimizer off')
    if args.stop_split_at is None:
        args.stop_split_at = max(0, args.iterations - 1000)
    if not 0 <= args.stop_split_at <= args.iterations:
        raise ValueError('stop-split-at must be within total iterations')
    first = (args.warmup_length // args.refine_every + 1) * args.refine_every
    if not args.no_refine and first >= args.stop_split_at:
        raise ValueError('Refinement schedule has zero events; increase iterations/stop-split-at or explicitly use --no-refine')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args=parser_for_cli().parse_args(argv)
    validate_args(args)
    from orbvi_gaussian.inria_surfel_backend import inria_backend_provenance
    backend_provenance=inria_backend_provenance(args.inria_backend) if args.renderer_inria else None
    splatfacto_command(args.dataset,args.output,iterations=args.iterations,
                       allow_candidate_calibration=args.allow_candidate_calibration)
    out=new_output_dir(args.output)
    c=copy.deepcopy(method_configs['splatfacto'])
    c.pipeline.model=LidarSplatfactoModelConfig(**fields_of(c.pipeline.model,{'_target','output_depth_during_training'}),
                      lidar_depth_loss_weight=args.depth_weight,lidar_alpha_loss_weight=args.alpha_weight,
                      lidar_metric_depth_loss_weight=args.metric_depth_weight,
                      lidar_metric_depth_huber_delta_m=args.metric_depth_delta_m)
    c.pipeline.datamanager=LidarFullImageDatamanagerConfig(**fields_of(c.pipeline.datamanager,{'_target'}))
    c.pipeline.datamanager.data=args.dataset.resolve()
    c.pipeline.datamanager.cache_images='cpu'
    c.pipeline.datamanager.cache_images_type='uint8'
    c.pipeline.datamanager.max_thread_workers=4
    parser=c.pipeline.datamanager.dataparser
    parser.data=args.dataset.resolve();parser.orientation_method='none';parser.center_method='none'
    parser.auto_scale_poses=False;parser.scale_factor=1.;parser.downscale_factor=1;parser.load_3D_points=True
    parser.scene_scale=200.
    m=c.pipeline.model;m.random_init=False;m.camera_optimizer.mode=args.camera_optimizer_mode;m.use_scale_regularization=True
    m.num_downscales=args.num_downscales;m.resolution_schedule=args.resolution_schedule
    m.gaussian_max_scale_m=args.gaussian_max_scale_m
    m.tangent_scale_target_m=args.tangent_scale_target_m
    m.tangent_scale_loss_weight=args.tangent_scale_weight
    m.tangent_screen_loss_weight=args.tangent_screen_weight
    m.tangent_max_diameter_px=args.tangent_max_diameter_px
    m.enable_refinement=not args.no_refine
    m.refine_every=args.refine_every
    m.stop_screen_size_at=args.stop_screen_size_at
    m.split_screen_size=args.split_screen_size
    m.warmup_length=args.warmup_length
    m.reset_opacity_during_refinement=args.reset_opacity
    m.training_audit_path=out/'refinement.json'
    m.training_iterations=args.iterations
    m.surface_splats=args.surface_splats
    m.use_2dgs_renderer=args.renderer_2dgs
    m.use_inria_surfel_renderer=args.renderer_inria
    m.inria_backend=args.inria_backend
    m.optimize_shared_rig=args.optimize_shared_rig
    m.max_gaussians=args.max_gaussians
    if args.warm_ply is not None:
        warm_path=args.warm_ply.resolve()
        if ROOT not in warm_path.parents or not warm_path.is_file():
            raise ValueError('Warm PLY must be an existing project file')
        m.gaussian_initialization_path=warm_path
        m.warm_opacity_cap=args.warm_opacity_cap
        # Warm SH3 coefficients must participate immediately, not remain
        # dormant behind the fresh-run SH/resolution schedules.
        m.sh_degree_interval=1
        m.num_downscales=args.num_downscales
    if args.optimize_shared_rig:
        if args.renderer_inria or args.renderer_2dgs:raise ValueError('Rig gradients require gsplat 3D in this calibration subproblem')
        from nerfstudio.engine.optimizers import AdamOptimizerConfig
        c.optimizers['rig']={'optimizer':AdamOptimizerConfig(lr=.0003,eps=1e-15),'scheduler':None}
    if (args.dataset/'lidar_reference.npz').exists():m.lidar_reference_path=(args.dataset/'lidar_reference.npz').resolve()
    m.stop_split_at=args.stop_split_at
    c.output_dir=out;c.max_num_iterations=args.iterations;c.vis=args.vis
    c.viewer.websocket_host='127.0.0.1';c.viewer.quit_on_train_completion=True
    if args.viewer_port is not None:c.viewer.websocket_port=args.viewer_port
    c.steps_per_save=args.save_every;c.steps_per_eval_image=500;c.steps_per_eval_all_images=2000
    c.machine.seed=42
    launch={'dataset':str(args.dataset.resolve()),'iterations':args.iterations,
            'depth_weight':args.depth_weight,'alpha_weight':args.alpha_weight,
            'metric_depth_weight':args.metric_depth_weight,'metric_depth_delta_m':args.metric_depth_delta_m,
            'metric_depth_loss_formula':'additive weight * mean(Huber(camera_z_prediction_m - target_m, delta_m)) on identical known support; disabled at weight=0',
            'model':'Splatfacto + sparse camera-z depth and visibility','world':'camera_init','units':'m',
            'calibration_experimental':args.allow_candidate_calibration}
    launch['surface_splats']=args.surface_splats
    launch['renderer_2dgs']=args.renderer_2dgs
    launch['renderer_inria']=args.renderer_inria
    launch['inria_backend']=args.inria_backend
    launch['renderer_provenance']=backend_provenance
    launch['optimize_shared_rig']=args.optimize_shared_rig
    launch['warm_ply']=str(args.warm_ply.resolve()) if args.warm_ply is not None else None
    launch['warm_opacity_cap']=args.warm_opacity_cap
    launch['max_gaussians']=args.max_gaussians
    launch['no_refine']=args.no_refine
    launch['save_every']=args.save_every
    launch['viewer_port']=args.viewer_port
    launch['vis']=args.vis
    launch.update(refine_every=m.refine_every, stop_split_at=m.stop_split_at,
        stop_screen_size_at=m.stop_screen_size_at, split_screen_size=m.split_screen_size,
        warmup_length=m.warmup_length, reset_opacity=args.reset_opacity,
        gaussian_max_scale_m=m.gaussian_max_scale_m, num_downscales=m.num_downscales,
        tangent_scale_target_m=m.tangent_scale_target_m, tangent_scale_weight=m.tangent_scale_loss_weight,
        tangent_screen_weight=m.tangent_screen_loss_weight, tangent_max_diameter_px=m.tangent_max_diameter_px,
        camera_optimizer_mode=m.camera_optimizer.mode,
        initialization_origin='warm_trained_ply' if args.warm_ply else 'measured_points',
        initialization_sha256=file_sha256(args.warm_ply) if args.warm_ply else None,
        parameter_origin='optimizer_updates_to_initialized_gaussian_parameters')
    (out/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2))
    print('Training LiDAR-supervised block:',json.dumps(launch),flush=True)
    import torch
    torch.cuda.reset_peak_memory_stats()
    train_main(c)
    peak_allocated=torch.cuda.max_memory_allocated()
    peak_reserved=torch.cuda.max_memory_reserved()
    configs=[str(x.relative_to(ROOT)) for x in out.rglob('config.yml')]
    checkpoints=sorted(out.rglob('step-*.ckpt'))
    if len(configs) != 1 or not checkpoints:
        raise RuntimeError('Training returned without one config and a saved checkpoint')
    checkpoint=checkpoints[-1]
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    actual_step=state['step']
    if actual_step != args.iterations-1:
        raise RuntimeError(f'Training stopped at {actual_step}; expected {args.iterations-1}')
    parameters={k:v for k,v in state['pipeline'].items() if '.gauss_params.' in k}
    final_count=len(next(v for k,v in parameters.items() if k.endswith('.means')))
    result={'status':'trained','iterations_requested':args.iterations,'configs':configs,
            'checkpoint':str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256':file_sha256(checkpoint),
            'checkpoint_step':actual_step, 'iterations_completed':actual_step+1,
            'parameter_origin':'trained_checkpoint', 'gaussians':final_count,
            'inria_backend':args.inria_backend, 'renderer_provenance':backend_provenance,
            'gpu_peak_allocated_bytes':peak_allocated, 'gpu_peak_reserved_bytes':peak_reserved,
            'gpu_peak_protocol':'torch.cuda allocator counters reset immediately before train_main in this training process',
            'initialization':{k:launch[k] for k in ('initialization_origin','initialization_sha256','warm_ply','warm_opacity_cap')},
            'refinement':json.loads((out/'refinement.json').read_text()),
            'note':'Held-out appearance and conditioned geometry must be evaluated before acceptance'}
    (out/'training_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

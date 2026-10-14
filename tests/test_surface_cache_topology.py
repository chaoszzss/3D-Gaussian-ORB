"""Exercise production cache invalidation with actual CPU gsplat topology.

The control removes only the cache invalidation from a copy of production code.
Both versions use real model loss/callback methods and real gsplat operations.
No live source is changed; no CUDA renderer or training is run.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from pytorch_msssim import SSIM
from scipy.spatial import cKDTree
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

ROOT=Path(__file__).resolve().parents[1]
RELATIVE=Path('src/orbvi_gaussian/lidar_splatfacto.py')
CACHE_FIX='''            if duplicate or split or strategy.last_pruned:
                # Topology changes invalidate index-aligned references even when
                # growth and pruning cancel out to the same Gaussian count.
                self._nearest_surface = None
                self._nearest_normal = None
                self._nearest_normal_valid = None
                self._last_surface_update = -1
'''


@pytest.fixture(scope='module')
def versions(tmp_path_factory):
    scratch=tmp_path_factory.mktemp('surface_cache_patch')
    target=scratch/RELATIVE
    target.parent.mkdir(parents=True)
    production=(ROOT/RELATIVE).read_text()
    assert production.count(CACHE_FIX)==1
    original=production.replace(CACHE_FIX,'').encode()
    target.write_text(production)
    result={}
    for label,data in (('old',original),('patched',target.read_bytes())):
        name='orbvi_gaussian._surface_cache_'+label+'_isolated'
        spec=importlib.util.spec_from_loader(name,loader=None)
        module=importlib.util.module_from_spec(spec)
        module.__file__=str(target)
        sys.modules[name]=module
        exec(compile(data,str(target),'exec'),module.__dict__)
        result[label]=module
    yield result
    for module in result.values():sys.modules.pop(module.__name__,None)


def make_model(module,mode,step=10800):
    model=module.LidarSplatfactoModel.__new__(module.LidarSplatfactoModel)
    torch.nn.Module.__init__(model)
    model.device_indicator_param=torch.nn.Parameter(torch.empty(0),requires_grad=False)
    model.config=module.LidarSplatfactoModelConfig(surface_splats=True,gaussian_max_scale_m=0.,
        lidar_surface_anchor_weight=.02,surface_normal_weight=.03,use_scale_regularization=False,
        training_audit_path=None,reset_opacity_during_refinement=False)
    small=mode in ('duplicate_prune','duplicate_only')
    prune=mode in ('split_prune','duplicate_prune','prune_only')
    q=math.sqrt(.5)
    reference=np.array([[0.,0.,0.],[10.,0.,0.],[20.,0.,0.]],np.float32)
    model.gauss_params=torch.nn.ParameterDict(dict(
        means=torch.nn.Parameter(torch.from_numpy(reference.copy())),
        scales=torch.nn.Parameter(torch.full((3,3),math.log(.005 if small else .02))),
        quats=torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.],[q,0.,q,0.],[q,-q,0.,0.]])),
        opacities=torch.nn.Parameter(torch.logit(torch.tensor([[.9],[.001 if prune else .9],[.9]])))))
    model.optimizers={key:torch.optim.Adam([value],lr=0.) for key,value in model.gauss_params.items()}
    sum(p.square().sum() for p in model.gauss_params.values()).backward()
    for optimizer in model.optimizers.values():optimizer.step();optimizer.zero_grad()
    strategy=module.AuditedRefinementStrategy(refine_start_iter=500,refine_stop_iter=20000,
        refine_every=100,reset_every=50000,pause_refine_after_reset=0,
        grow_grad2d=.5,grow_scale3d=.01,refine_scale2d_stop_iter=0,
        prune_opa=.01,key_for_gradient='means2d')
    strategy.max_gaussians=10
    # Stale event counters must never invalidate cache on a non-event callback.
    strategy.last_growth=(7,5);strategy.last_pruned=9
    model.strategy=strategy;model.strategy_state=strategy.initialize_state(scene_scale=1.)
    proxy=torch.zeros(1,3,2,requires_grad=True)
    proxy.grad=torch.zeros_like(proxy)
    if mode in ('split_prune','duplicate_prune','split_only','duplicate_only'):proxy.grad[0,0,0]=1.
    model.info=dict(means2d=proxy,width=2,height=2,n_cameras=1,radii=torch.ones(1,3),gaussian_ids=None)
    model.refinement_audit=dict(events=[]);model._audit_resumed=False
    model._surface_reference_cpu=reference
    model._surface_tree=cKDTree(reference)
    model._surface_normals_cpu=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]],np.float32)
    model._surface_normals_valid_cpu=np.array([True,False,True])
    model._nearest_surface=None;model._nearest_normal=None;model._nearest_normal_valid=None
    model._last_surface_update=-1
    model.camera_optimizer=CameraOptimizerConfig(mode='off').setup(num_cameras=1,device='cpu')
    model.ssim=SSIM(data_range=1.,size_average=True,channel=3)
    model.step=10799
    outputs=dict(rgb=torch.zeros(16,16,3),depth=torch.ones(16,16,1),
                 accumulation=torch.ones(16,16,1),background=torch.zeros(3))
    batch=dict(image=torch.zeros(16,16,3),lidar_depth=torch.ones(16,16,1),
               lidar_depth_mask=torch.ones(16,16,1,dtype=torch.bool))
    model.get_loss_dict(outputs,batch)
    model.step=step
    return model,outputs,batch


@pytest.mark.parametrize('mode',['split_prune','duplicate_prune'])
def test_same_count_real_topology_old_fails_patched_passes(versions,mode,record_property):
    rows={}
    for name,module in versions.items():
        torch.manual_seed(17)
        model,outputs,batch=make_model(module,mode)
        old_means=model.means.detach().clone()
        cached=model._nearest_surface
        model.step_post_backward(10800)
        assert len(model.means)==3 and not torch.equal(model.means,old_means)
        assert model.refinement_audit['events'][-1]['pruned']==1
        assert model.refinement_audit['events'][-1]['split' if mode=='split_prune' else 'duplicated']==1
        if name=='patched':
            assert model._nearest_surface is model._nearest_normal is model._nearest_normal_valid is None
            assert model._last_surface_update==-1
        else:
            assert model._nearest_surface is cached and model._last_surface_update==10799
        model.step=10801
        losses=model.get_loss_dict(outputs,batch)
        _,ids=model._surface_tree.query(model.means.detach().numpy())
        reference=torch.from_numpy(model._surface_reference_cpu[ids])
        aligned=torch.equal(model._nearest_surface,reference)
        assert aligned==(name=='patched')
        if name=='patched':
            assert torch.equal(model._nearest_normal,torch.from_numpy(model._surface_normals_cpu[ids]))
            assert torch.equal(model._nearest_normal_valid,torch.from_numpy(model._surface_normals_valid_cpu[ids]))
            assert model._last_surface_update==10801
        for key,param in model.gauss_params.items():
            optimizer=model.optimizers[key]
            assert optimizer.param_groups[0]['params'][0] is param
            assert optimizer.state[param]['exp_avg'].shape==param.shape
            assert optimizer.state[param]['exp_avg_sq'].shape==param.shape
        rows[name]={key:float(losses[key].detach()) for key in ('lidar_surface_anchor_loss','lidar_surface_normal_loss')}
        record_property(name+'_anchor',rows[name]['lidar_surface_anchor_loss'])
        record_property(name+'_normal',rows[name]['lidar_surface_normal_loss'])
    assert rows['old']['lidar_surface_anchor_loss']>1.
    assert rows['patched']['lidar_surface_anchor_loss']<.001
    assert rows['old']['lidar_surface_normal_loss']>.01
    assert rows['patched']['lidar_surface_normal_loss']<1e-6


@pytest.mark.parametrize('mode,expected_count',[
    ('split_only',4),('duplicate_only',4),('prune_only',2)])
def test_any_actual_topology_invalidates_all_caches(versions,mode,expected_count):
    model,outputs,batch=make_model(versions['patched'],mode)
    model.step_post_backward(10800)
    assert len(model.means)==expected_count
    assert model._nearest_surface is model._nearest_normal is model._nearest_normal_valid is None
    model.step=10801;model.get_loss_dict(outputs,batch)
    assert len(model._nearest_surface)==expected_count
    assert model._last_surface_update==10801


@pytest.mark.parametrize('step',[10800,10801])
def test_no_topology_reuses_cache_and_preserves_periodic_refresh(versions,step):
    model,outputs,batch=make_model(versions['patched'],'none',step=step)
    cached=model._nearest_surface
    model.step_post_backward(step)
    assert model._nearest_surface is cached
    model.step=11048;model.get_loss_dict(outputs,batch)
    assert model._nearest_surface is cached and model._last_surface_update==10799
    model.step=11049;model.get_loss_dict(outputs,batch)
    assert model._nearest_surface is not cached and model._last_surface_update==11049


def test_regression_exercises_the_actual_production_class(versions):
    from orbvi_gaussian import lidar_splatfacto as live
    assert live.LidarSplatfactoModel.step_post_backward.__code__.co_code == versions['patched'].LidarSplatfactoModel.step_post_backward.__code__.co_code

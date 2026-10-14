"""CPU schedule, real training callbacks and serialized parameter constraints."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from gsplat.strategy import DefaultStrategy
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.models.splatfacto import SplatfactoModel

from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModel, LidarSplatfactoModelConfig
from orbvi_gaussian.scale_annealing import gaussian_scale_cap_m, validate_scale_annealing


@pytest.mark.parametrize('step,expected',[(0,.12),(999,.12),(1000,.12),(5500,.06),
                                          (10000,.03),(14999,.03),(30000,.03)])
def test_log_linear_boundaries_and_geometric_midpoint(step,expected):
    value=gaussian_scale_cap_m(step,.03,.12,1000,10000)
    if step==5500:
        assert value==pytest.approx(expected,rel=1e-15)
        assert value != pytest.approx((.12+.03)/2)
    else:
        assert value==expected


def test_monotone_and_constant_schedule():
    caps=[gaussian_scale_cap_m(step,.03,.12,1000,10000) for step in range(15000)]
    assert all(a>=b for a,b in zip(caps,caps[1:]))
    assert all(gaussian_scale_cap_m(s,.03,.03,1000,10000)==.03 for s in (0,5500,14999))


@pytest.mark.parametrize('target',[0.,.03,.12,.5])
def test_disabled_schedule_returns_exact_legacy_cap(target):
    for step in (0,5500,14999,30000):
        assert gaussian_scale_cap_m(step,target).hex()==target.hex()


@pytest.mark.parametrize('target,initial,start,end,iterations',[
    (.03,float('nan'),1000,10000,15000),(.03,float('inf'),1000,10000,15000),
    (.03,0.,1000,10000,15000),(.03,-.12,1000,10000,15000),
    (.03,.02,1000,10000,15000),(float('nan'),.12,1000,10000,15000),
    (float('inf'),.12,1000,10000,15000),(0.,.12,1000,10000,15000),
    (.03,.12,-1,10000,15000),(.03,.12,1000,1000,15000),
    (.03,.12,1001,1000,15000),(.03,.12,0,15000,15000),
    (.03,.12,0,16000,15000),(.03,.12,.5,10000,15000),
    (.03,.12,False,10000,15000),(.03,None,1000,0,15000),
    (.03,None,0,10000,15000)])
def test_invalid_schedule_rejected(target,initial,start,end,iterations):
    with pytest.raises(ValueError,match='Scale annealing'):
        validate_scale_annealing(target,initial,start,end,iterations)


def make_model(surface,enabled=True,dtype=torch.float32):
    model=LidarSplatfactoModel.__new__(LidarSplatfactoModel)
    torch.nn.Module.__init__(model)
    model.config=LidarSplatfactoModelConfig(gaussian_max_scale_m=.03,
        gaussian_initial_max_scale_m=.12 if enabled else None,
        gaussian_scale_anneal_start=1000 if enabled else 0,
        gaussian_scale_anneal_end=10000 if enabled else 0,
        training_iterations=15000,surface_splats=surface,surface_thickness_m=.015,
        enable_refinement=False,reset_opacity_during_refinement=False)
    model.gauss_params=torch.nn.ParameterDict(dict(
        means=torch.nn.Parameter(torch.zeros(2,3,dtype=dtype)),
        scales=torch.nn.Parameter(torch.tensor([[.2,.08,.02],[.001,.3,.5]],dtype=dtype).log()),
        quats=torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]],dtype=dtype)),
        opacities=torch.nn.Parameter(torch.tensor([[17.],[2.]],dtype=dtype))))
    model.strategy=DefaultStrategy()
    model.refinement_audit=dict(events=[])
    model._nearest_surface=None
    model.step=0
    return model


@pytest.mark.parametrize('surface',[False,True])
@pytest.mark.parametrize('step',[0,1000,5500,10000,14999])
def test_forward_and_actual_post_optimizer_callback_share_cap_and_save(tmp_path,monkeypatch,surface,step):
    model=make_model(surface)
    original_opacity=model.opacities.detach().clone()
    original_quats=model.quats.detach().clone()
    monkeypatch.setattr(SplatfactoModel,'get_outputs',lambda self,camera:dict(scales=self.scales.detach().clone()))
    optimizer=torch.optim.SGD([model.scales],lr=1.)
    attrs=TrainingCallbackAttributes(optimizers=SimpleNamespace(optimizers={'scales':optimizer}),
                                     grad_scaler=None,pipeline=None,trainer=None)
    callbacks=model.get_training_callbacks(attrs)
    for callback in callbacks:
        callback.run_callback_at_location(step,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
    rendered=model.get_outputs(None)['scales']
    cap=model._effective_scale_cap_m(step)
    assert float(rendered[:,:2].max())==torch.tensor(math.log(cap),dtype=rendered.dtype).item()
    if surface:
        assert float(rendered[:,2].max())==torch.tensor(math.log(.015),dtype=rendered.dtype).item()
    else:
        assert torch.equal(rendered[:,2].max(),rendered[:,:2].max())
    # Force a real optimizer update beyond the bound after forward.
    (-model.scales.sum()).backward();optimizer.step()
    assert model.scales.max()>math.log(cap)
    for callback in callbacks:
        callback.run_callback_at_location(step,TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
    assert torch.equal(model.opacities,original_opacity)
    assert torch.equal(model.quats,original_quats)
    assert model.refinement_audit['last_callback_step']==step
    path=tmp_path/'checkpoint.ckpt'
    torch.save(dict(step=step,pipeline=model.state_dict()),path)
    saved=torch.load(path,map_location='cpu',weights_only=True)['pipeline']['gauss_params.scales']
    assert saved.max().item()==torch.tensor(math.log(cap),dtype=saved.dtype).item()
    assert float(saved.exp().max())<=cap
    if step>=10000:
        assert float(saved.exp().max())<=.03
    if surface:
        assert float(saved[:,2].exp().max())<=.015


@pytest.mark.parametrize('surface',[False,True])
def test_default_forward_and_callback_bitwise_match_legacy_and_eval_never_clamps(monkeypatch,surface):
    model=make_model(surface,enabled=False)
    model.step=5500
    monkeypatch.setattr(SplatfactoModel,'get_outputs',lambda self,camera:self.scales.detach().clone())
    original=model.scales.detach().clone()
    expected=original.clamp(max=math.log(.03))
    if surface:expected[:,2].clamp_(max=math.log(.015))
    assert torch.equal(model.get_outputs(None),expected)
    with torch.no_grad():model.scales.copy_(original)
    model.step_post_backward(5500)
    assert torch.equal(model.scales,expected)
    model.config.gaussian_initial_max_scale_m=.12
    model.config.gaussian_scale_anneal_start=1000
    model.config.gaussian_scale_anneal_end=10000
    with torch.no_grad():model.scales.copy_(original)
    model.eval();model.step=14999
    assert torch.equal(model.get_outputs(None),original)
    assert torch.equal(model.scales,original)


def test_model_invalid_schedule_fails_before_parent_allocation(monkeypatch):
    model=make_model(False)
    model.config.gaussian_scale_anneal_end=15000
    monkeypatch.setattr(SplatfactoModel,'populate_modules',lambda self:pytest.fail('invalid config allocated parent'))
    with pytest.raises(ValueError,match='end must be less'):
        model.populate_modules()


@pytest.fixture
def cli():
    path=Path(__file__).resolve().parents[1]/'scripts/train_lidar_block.py'
    spec=importlib.util.spec_from_file_location('scale_anneal_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('flags',[
    ['--gaussian-initial-max-scale-m','nan'],['--gaussian-initial-max-scale-m','inf'],
    ['--gaussian-initial-max-scale-m','0'],['--gaussian-initial-max-scale-m','.02'],
    ['--gaussian-initial-max-scale-m','.12'],['--gaussian-scale-anneal-start','1000'],
    ['--gaussian-initial-max-scale-m','.12','--gaussian-scale-anneal-start','1000',
     '--gaussian-scale-anneal-end','1000'],
    ['--gaussian-initial-max-scale-m','.12','--gaussian-scale-anneal-end','15000']])
def test_cli_rejects_invalid_schedule(cli,flags):
    args=cli.parser_for_cli().parse_args(['--dataset','data/x','--output','runs/x',
        '--iterations','15000','--gaussian-max-scale-m','.03',*flags])
    with pytest.raises(ValueError,match='Scale annealing'):cli.validate_args(args)


@pytest.mark.parametrize('enabled',[False,True])
def test_cli_schedule_reaches_yaml_config_and_launch_without_training(tmp_path,monkeypatch,cli,enabled):
    monkeypatch.setattr(cli,'splatfacto_command',lambda *a,**kw:None)
    def create(path):path.mkdir();return path
    monkeypatch.setattr(cli,'new_output_dir',create)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    class StopBeforeTraining(Exception):pass
    def capture(config):
        import yaml
        loaded=yaml.load(yaml.dump(config),Loader=yaml.Loader)
        for m in (config.pipeline.model,loaded.pipeline.model):
            assert m.gaussian_max_scale_m==.03
            assert m.gaussian_initial_max_scale_m==(.12 if enabled else None)
            assert m.gaussian_scale_anneal_start==(1000 if enabled else 0)
            assert m.gaussian_scale_anneal_end==(10000 if enabled else 0)
            assert m.training_iterations==15000
            assert m.warm_opacity_cap==1. and not m.reset_opacity_during_refinement
        raise StopBeforeTraining
    monkeypatch.setattr(cli,'train_main',capture)
    out=tmp_path/'run'
    flags=['--gaussian-initial-max-scale-m','.12','--gaussian-scale-anneal-start','1000',
           '--gaussian-scale-anneal-end','10000'] if enabled else []
    with pytest.raises(StopBeforeTraining):
        cli.main(['--dataset',str(tmp_path/'data'),'--output',str(out),
                  '--iterations','15000','--gaussian-max-scale-m','.03',*flags])
    launch=json.loads((out/'launch.json').read_text())
    assert launch['gaussian_initial_max_scale_m']==(.12 if enabled else None)
    assert launch['gaussian_scale_anneal_start']==(1000 if enabled else 0)
    assert launch['gaussian_scale_anneal_end']==(10000 if enabled else 0)
    assert launch['gaussian_scale_anneal_interpolation']==('log-linear metres' if enabled else 'disabled')

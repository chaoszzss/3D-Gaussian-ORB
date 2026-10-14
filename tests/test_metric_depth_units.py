"""Physical-unit, unknown-pixel and configuration tests for additive depth Huber."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from orbvi_gaussian.lidar_splatfacto import compute_lidar_losses, LidarSplatfactoModelConfig


@pytest.mark.parametrize('target',[3.,10.,25.])
@pytest.mark.parametrize('error',[-.3,-.05,.05,.3])
def test_metric_loss_and_derivative_have_metre_units(target,error):
    p=torch.tensor([[target+error]],dtype=torch.float64,requires_grad=True)
    z=torch.tensor([[target]],dtype=torch.float64)
    losses=compute_lidar_losses(dict(depth=p,accumulation=torch.ones_like(p)),z,z>0,
        .5,.1,wmetric=.05,metric_delta_m=.15)
    metric=losses['lidar_metric_depth_loss']
    expected=.05*(.5*error**2 if abs(error)<.15 else .15*(abs(error)-.075))
    assert float(metric)==pytest.approx(expected,abs=2e-16)
    grad,=torch.autograd.grad(metric,(p,))
    assert float(grad)==pytest.approx(.05*max(-.15,min(.15,error)),abs=2e-16)
    scaled=compute_lidar_losses(dict(depth=p*10,accumulation=torch.ones_like(p)),z*10,z>0,
        .5,.1,wmetric=.05,metric_delta_m=1.5)
    torch.testing.assert_close(scaled['lidar_depth_loss'],losses['lidar_depth_loss'])
    torch.testing.assert_close(scaled['lidar_metric_depth_loss'],metric*100)


def test_disabled_metric_preserves_exact_original_entries_values_and_gradients():
    p=torch.tensor([[10.15,2.8],[1.,99.]],dtype=torch.float64,requires_grad=True)
    a=torch.full_like(p,.8,requires_grad=True)
    z=torch.tensor([[10.,3.],[0.,float('nan')]],dtype=torch.float64)
    valid=torch.tensor([[True,True],[False,False]])
    old=dict(lidar_depth_loss=.5*F.huber_loss((p[valid]-z[valid])/z[valid],torch.zeros_like(p[valid]),delta=.1),
             lidar_alpha_loss=.1*(1-a[valid]).mean())
    default=compute_lidar_losses(dict(depth=p,accumulation=a),z,valid,.5,.1)
    explicit=compute_lidar_losses(dict(depth=p,accumulation=a),z,valid,.5,.1,wmetric=0,metric_delta_m=.15)
    assert old.keys()==default.keys()==explicit.keys()
    for key in old:
        assert torch.equal(old[key],default[key]) and torch.equal(old[key],explicit[key])
    expected=torch.autograd.grad(sum(old.values()),(p,a),retain_graph=True)
    actual=torch.autograd.grad(sum(default.values()),(p,a),retain_graph=True)
    assert all(torch.equal(x,y) for x,y in zip(expected,actual))
    assert LidarSplatfactoModelConfig().lidar_metric_depth_loss_weight==0


def test_metric_unknown_pixels_empty_support_and_gradcheck():
    p=torch.tensor([[10.08,700.],[8.,9.]],dtype=torch.float64,requires_grad=True)
    a=torch.full_like(p,.4,requires_grad=True)
    target=torch.tensor([[10.,99.],[0.,float('nan')]],dtype=torch.float64)
    valid=torch.tensor([[True,False],[True,True]])
    def objective(depth,alpha):
        return sum(compute_lidar_losses(dict(depth=depth,accumulation=alpha),target,valid,
                                       wmetric=.05).values())
    assert torch.autograd.gradcheck(objective,(p,a))
    metric=compute_lidar_losses(dict(depth=p,accumulation=a),target,valid,wmetric=.05)['lidar_metric_depth_loss']
    grad,=torch.autograd.grad(metric,(p,))
    torch.testing.assert_close(grad,torch.tensor([[.004,0.],[0.,0.]],dtype=torch.float64))
    empty=compute_lidar_losses(dict(depth=p,accumulation=a),target,torch.zeros_like(valid),wmetric=.05)
    assert all(value.requires_grad and value.item()==0 for value in empty.values())
    pg,ag=torch.autograd.grad(sum(empty.values()),(p,a))
    assert torch.equal(pg,torch.zeros_like(p)) and torch.equal(ag,torch.zeros_like(a))


@pytest.mark.parametrize('key,value',[
    ('wmetric',-1.),('wmetric',float('nan')),('wmetric',float('inf')),
    ('metric_delta_m',0.),('metric_delta_m',-1.),
    ('metric_delta_m',float('nan')),('metric_delta_m',float('inf'))])
def test_metric_rejects_invalid_configuration(key,value):
    with pytest.raises(ValueError,match='metric'):
        compute_lidar_losses(dict(depth=torch.ones(1,1),accumulation=torch.ones(1,1)),
                             torch.ones(1,1),torch.ones(1,1),**{key:value})


@pytest.mark.parametrize('name',['depth','accumulation'])
@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_metric_rejects_nonfinite_renderer_even_with_no_support(name,value):
    outputs=dict(depth=torch.ones(1,1),accumulation=torch.ones(1,1))
    outputs[name][0,0]=value
    with pytest.raises(ValueError,match='Nonfinite'):
        compute_lidar_losses(outputs,torch.zeros(1,1),torch.zeros(1,1),wmetric=.05)


def test_weighted_metric_overflow_is_rejected():
    with pytest.raises(ValueError,match='Nonfinite weighted metric'):
        compute_lidar_losses(dict(depth=torch.full((1,1),3e38),accumulation=torch.ones(1,1)),
                             torch.ones(1,1),torch.ones(1,1),wmetric=1e38)


@pytest.mark.parametrize('weight',[0.,.05])
def test_cli_metric_parameters_reach_real_model_config_and_launch(tmp_path,monkeypatch,weight):
    path=Path(__file__).resolve().parents[1]/'scripts/train_lidar_block.py'
    spec=importlib.util.spec_from_file_location('metric_train_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'splatfacto_command',lambda *args,**kwargs:None)
    def output_dir(path):
        path.mkdir();return path
    monkeypatch.setattr(cli,'new_output_dir',output_dir)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    class StopBeforeTraining(Exception):
        pass
    def capture(config):
        import yaml
        reloaded=yaml.load(yaml.dump(config),Loader=yaml.Loader)
        for model in (config.pipeline.model,reloaded.pipeline.model):
            assert model.lidar_metric_depth_loss_weight==weight
            assert model.lidar_metric_depth_huber_delta_m==.15
            assert model.lidar_depth_loss_weight==.5 and model.lidar_alpha_loss_weight==.1
        raise StopBeforeTraining
    monkeypatch.setattr(cli,'train_main',capture)
    out=tmp_path/'run'
    with pytest.raises(StopBeforeTraining):
        cli.main(['--dataset',str(tmp_path/'data'),'--output',str(out),
                  '--depth-weight','.5','--alpha-weight','.1',
                  '--metric-depth-weight',str(weight),'--metric-depth-delta-m','.15'])
    launch=json.loads((out/'launch.json').read_text())
    assert launch['metric_depth_weight']==weight and launch['metric_depth_delta_m']==.15
    assert 'additive' in launch['metric_depth_loss_formula']


@pytest.mark.parametrize('flags',[
    ['--metric-depth-weight','nan'],['--metric-depth-weight','inf'],
    ['--metric-depth-weight','-1'],['--metric-depth-delta-m','0'],
    ['--metric-depth-delta-m','nan']])
def test_cli_metric_invalid_arguments_fail_before_training(flags):
    path=Path(__file__).resolve().parents[1]/'scripts/train_lidar_block.py'
    spec=importlib.util.spec_from_file_location('metric_train_cli_bad',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    args=cli.parser_for_cli().parse_args(['--dataset','data/x','--output','runs/x',*flags])
    with pytest.raises(ValueError,match='Metric depth'):
        cli.validate_args(args)

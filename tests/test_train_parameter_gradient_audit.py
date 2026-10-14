"""CPU safety/aggregation checks; CUDA gradients remain measured by the queued job."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def audit():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_train_parameter_gradients.py'
    spec=importlib.util.spec_from_file_location('train_grad_audit_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_missing_incomplete_or_changed_checkpoint_refused(audit,tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(ValueError,match='Missing training_result'):
        audit.verify_run_completion(run)
    checkpoint=run/'step-000011999.ckpt';checkpoint.write_bytes(b'actual bytes to pin')
    config=run/'config.yml';config.write_text('config test')
    result=dict(status='trained',parameter_origin='trained_checkpoint',checkpoint_step=11999,
        iterations_completed=12000,checkpoint='run/step-000011999.ckpt',
        checkpoint_sha256=audit.sha(checkpoint),configs=['run/config.yml'])
    result_path=run/'training_result.json';result_path.write_text(json.dumps(result))
    assert audit.verify_run_completion(run)[1]==checkpoint
    result['checkpoint_step']=4000;result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='expected final step'):
        audit.verify_run_completion(run)
    result['checkpoint_step']=11999;result_path.write_text(json.dumps(result))
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        audit.verify_run_completion(run)


def test_train_selection_cannot_read_dev_or_test(audit):
    metadata=dict(train_filenames=['a','b'],val_filenames=['v'],test_filenames=['t'],
                  frames=[dict(file_path=name) for name in ('a','b','v','t')])
    assert audit.fixed_train_frames(metadata,['a','b'])==metadata['frames'][:2]
    for names in (['v'],['t'],['unknown'],['a','a'],[]):
        with pytest.raises(ValueError,match='TRAIN'):
            audit.fixed_train_frames(metadata,names)
    metadata['val_filenames'].append('a')
    with pytest.raises(ValueError,match='disjoint TRAIN'):
        audit.fixed_train_frames(metadata,['a'])


def test_parameter_gradient_summary_preserves_zero_and_masked_norms(audit):
    p=torch.nn.Parameter(torch.tensor([[3.,4.],[0.,0.],[0.,2.]]))
    grad,=torch.autograd.grad(.5*p.square().sum(),(p,))
    row=audit.gradient_summary(grad,np.array([True,True,False]),np.array([True,False,False]))
    assert row['connected'] and row['finite']
    assert row['all']['l2']==pytest.approx(np.sqrt(29))
    assert row['all']['nonzero_gaussians']==2
    assert row['native_visible']['l2']==5
    assert row['native_visible']['per_gaussian_norm_quantiles']['0.5']==2.5
    assert row['center_on_known_support']['per_gaussian_norm_quantiles']['0.5']==5
    assert torch.equal(p.detach(),torch.tensor([[3.,4.],[0.,0.],[0.,2.]]))
    absent=audit.gradient_summary(None,np.array([True,False,False]),np.zeros(3,bool))
    assert absent['connected'] is False and absent['all']['l2']==0
    assert absent['center_on_known_support']['per_gaussian_norm_quantiles']['0.5'] is None
    grad[0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite actual parameter gradient'):
        audit.gradient_summary(grad,np.ones(3,bool),np.ones(3,bool))


def test_lock_descriptor_must_match_project_gpu_lock(audit,tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    (tmp_path/'runtime').mkdir()
    with (tmp_path/'runtime/core_joint_gpu.lock').open('w') as correct, (tmp_path/'wrong.lock').open('w') as wrong:
        with pytest.raises(ValueError,match='not runtime/core_joint_gpu.lock'):
            audit.require_inherited_lock(wrong.fileno())
        audit.require_inherited_lock(correct.fileno())


def test_extra_prior_scalars_preserve_units_normal_mask_and_step_schedule(audit):
    from types import SimpleNamespace
    config=SimpleNamespace(lidar_surface_anchor_weight=.02,surface_splats=True,
        surface_normal_weight=.03,use_scale_regularization=True,max_gauss_ratio=10.)
    params=dict(means=torch.tensor([[.1,0.,0.]]),quats=torch.tensor([[1.,0.,0.,0.]]),
        scales=torch.tensor([[.12,.02,.005]]).log())
    reference=dict(xyz=np.zeros((1,3),np.float32),normals=np.array([[1.,0.,0.]],np.float32),
        normal_valid=np.array([True]))
    scalars=audit.geometry_prior_scalars(params,reference,np.array([0]),config,11999)
    assert scalars['lidar_surface_anchor_loss']==pytest.approx(.01)
    assert scalars['lidar_surface_normal_loss']==pytest.approx(.03)
    assert scalars['scale_reg']==0.
    assert scalars['scale_reg_if_active_step']==pytest.approx(1.4)
    reference['normal_valid'][:]=False
    active=audit.geometry_prior_scalars(params,reference,np.array([0]),config,11990)
    assert active['lidar_surface_normal_loss']==0.
    assert active['scale_reg']==pytest.approx(1.4)

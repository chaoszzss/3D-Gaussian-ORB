"""Reject partial or stale native reports before resuming an acceptance run."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('acceptance_scene', ROOT / 'scripts/evaluate_acceptance_scene.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture():
    frame = dict(block_id='block_015', dataset_index=4, group_id='1000', camera_id='camera_3',
                 file_path='images/a.png', source_image_path='/image/a.png', source_mask_path='/mask/a.png',
                 mask_path='/frozen/ego.png')
    plan = dict(plan_id='frozen-plan', frames=[frame])
    row = dict(frame, acceptance_mask_path=frame['mask_path'], protocols=dict(raw={}, legacy={}, ego_v2={}))
    report = dict(status='evaluated', metric_domain='float_renderer', acceptance={'sha256': 'plan-sha'},
                  acceptance_plan_id='frozen-plan', source_ply_sha256='ply-sha', images=1, frames=[row])
    return plan, report


def test_complete_exact_source_report():
    plan, report = fixture()
    assert M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha') == report['frames']


def test_direction_bundle_cannot_resume_as_plain_ply_or_other_support():
    plan, report = fixture()
    report['direction_support'] = {'sha256': 'support-sha'}
    assert M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha', 'support-sha')
    for expected in (None, 'other-support-sha'):
        with pytest.raises(ValueError, match='direction-support'):
            M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha', expected)


@pytest.mark.parametrize('change', ['missing_frame', 'duplicate_frame', 'wrong_plan', 'wrong_model', 'quantized',
                                    'wrong_image', 'wrong_order', 'wrong_mask', 'missing_protocol'])
def test_rejects_incomplete_and_different_evidence(change):
    plan, report = fixture()
    report = copy.deepcopy(report)
    if change == 'missing_frame': report['frames'] = []
    elif change == 'duplicate_frame': report['frames'] *= 2
    elif change == 'wrong_plan': report['acceptance_plan_id'] = 'other'
    elif change == 'wrong_model': report['source_ply_sha256'] = 'other'
    elif change == 'quantized': report['metric_domain'] = '8bit_png'
    elif change == 'wrong_image': report['frames'][0]['source_image_path'] = '/other.png'
    elif change == 'wrong_order': report['frames'][0]['dataset_index'] = 5
    elif change == 'wrong_mask': report['frames'][0]['acceptance_mask_path'] = '/other.png'
    else: del report['frames'][0]['protocols']['raw']
    with pytest.raises(ValueError):
        M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha')


def pinned_fixture():
    plan, report = fixture()
    source = dict(path='/model/a.ply', sha256='ply-sha', gaussians=100)
    model = dict(config='/config.yml', config_sha256='config-sha', checkpoint='/step.ckpt',
        checkpoint_sha256='checkpoint-sha', step=11999, inference_model_step=11999,
        transforms='/transforms.json', transforms_sha256='transforms-sha', source_ply=source)
    report.update(model, checkpoint_is_template=True, split='dev', world_parser_scale=1.,
        world_parser_transform=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]],
        gaussians=100, representation_id='ply-sha')
    extra = dict(source_lidar_depth_path='/depth.npy', source_lidar_support_path='/support.png',
        camera={'w':640, 'fl_x':320., 'camera_to_worlds_ns':[[1.,0.,0.,2.],[0.,1.,0.,3.],[0.,0.,1.,4.]]})
    plan['frames'][0].update(copy.deepcopy(extra))
    report['frames'][0].update(copy.deepcopy(extra))
    return plan, report, model


def test_current_pins_accept_exact_historical_child():
    plan, report, model = pinned_fixture()
    assert M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha', model=model)
    del report['representation_id']
    assert M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha', model=model)


@pytest.mark.parametrize('change', ['config_sha', 'checkpoint_sha', 'inference_step', 'source_path',
    'source_count', 'representation', 'teacher', 'support', 'camera', 'world'])
def test_resume_rejects_changed_complete_model_or_fixed_reference(change):
    plan, report, model = pinned_fixture()
    report = copy.deepcopy(report)
    if change == 'config_sha': report['config_sha256'] = 'other'
    elif change == 'checkpoint_sha': report['checkpoint_sha256'] = 'other'
    elif change == 'inference_step': report['inference_model_step'] = 0
    elif change == 'source_path': report['source_ply']['path'] = '/model/other.ply'
    elif change == 'source_count': report['source_ply']['gaussians'] = 99
    elif change == 'representation': report['representation_id'] = 'different-sidecar'
    elif change == 'teacher': report['frames'][0]['source_lidar_depth_path'] = '/other.npy'
    elif change == 'support': report['frames'][0]['source_lidar_support_path'] = '/other.png'
    elif change == 'camera': report['frames'][0]['camera']['camera_to_worlds_ns'][0][3] += .1
    elif change == 'world': report['world_parser_scale'] = .1
    with pytest.raises(ValueError):
        M.validate_child_report(report, plan, 'block_015', 'plan-sha', 0, 'ply-sha', model=model)

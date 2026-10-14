"""Warm initialization must preserve trained records and feed actual optimizers."""
import sys
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from orbvi_gaussian.training import confined_environment
os.environ.update(confined_environment())
import torch
from nerfstudio.data.scene_box import SceneBox
from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModelConfig
from package_garage import ply_header


def source_ply(directory):
    names = ['x','y','z','nx','ny','nz'] + ['f_dc_%d'%i for i in range(3)]
    names += ['f_rest_%d'%i for i in range(45)] + ['opacity']
    names += ['scale_%d'%i for i in range(3)] + ['rot_%d'%i for i in range(4)]
    values = np.zeros((6, len(names)), dtype='<f4')
    values[:, :3] = np.arange(18, dtype=np.float32).reshape(6, 3) * .1 + 2
    values[:, 6:9] = [.1, .2, .3]
    values[:, 9:54] = np.arange(45, dtype=np.float32) * .001
    values[:, names.index('opacity')] = 9.
    for i in range(3): values[:, names.index('scale_%d'%i)] = -3. - i
    values[:, names.index('rot_0')] = 1.
    path = directory / 'actual_schema_synthetic_fixture.ply'
    path.write_bytes(ply_header(tuple(names), len(values)) + values.tobytes())
    return path, values, names


def setup_model(directory, budget=10, cap=1.):
    path, values, names = source_ply(directory)
    config = LidarSplatfactoModelConfig(gaussian_initialization_path=path,
                                       max_gaussians=budget, warm_opacity_cap=cap)
    config.camera_optimizer.mode = 'off'
    seed_xyz = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    model = config.setup(scene_box=SceneBox(aabb=torch.tensor([[-10.,-10.,-10.],[10.,10.,10.]])),
                         num_train_data=1, metadata={}, device='cpu',
                         seed_points=(seed_xyz, torch.ones((4,3)) * 128))
    return model, values, names


def test_loaded_parameters_are_the_parameters_updated_by_adam(tmp_path):
    model, values, names = setup_model(tmp_path)
    assert len(model.means) == 6  # Initializer had four points.
    assert torch.equal(model.means.detach(), torch.from_numpy(values[:, :3]))
    rest = values[:, 9:54].reshape(6,3,15).transpose(0,2,1).copy()
    assert torch.equal(model.features_rest.detach(), torch.from_numpy(rest))
    groups = model.get_param_groups()
    optimizer = torch.optim.Adam(groups['means'], lr=.01)
    assert optimizer.param_groups[0]['params'][0] is model.means
    before = model.means.detach().clone()
    model.means.square().sum().backward(); optimizer.step()
    assert not torch.equal(before, model.means.detach())
    assert torch.equal(model._fixed_lidar_reference, torch.tensor(
        [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]))


def test_softened_opacity_is_explicit_and_other_geometry_is_preserved(tmp_path):
    model, values, names = setup_model(tmp_path, cap=.9)
    assert torch.allclose(torch.sigmoid(model.opacities), torch.full((6,1), .9))
    assert torch.equal(model.means.detach(), torch.from_numpy(values[:, :3]))
    assert torch.equal(model.scales.detach(), torch.from_numpy(values[:, [names.index('scale_%d'%i) for i in range(3)]]))
    assert model.warm_start_info['initial_opacities_capped'] == 6


def test_rejects_warm_count_over_growth_budget(tmp_path):
    with pytest.raises(ValueError, match='budget'):
        setup_model(tmp_path, budget=5)

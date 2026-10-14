"""CPU-only warm read bounds; large count is forwarded without large allocation."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from gsplat.strategy import DefaultStrategy
from nerfstudio.models.splatfacto import SplatfactoModel
from orbvi_gaussian.lidar_splatfacto import (
    LidarSplatfactoModel,LidarSplatfactoModelConfig,warm_start_read_budget)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_gaussian_ply as reader
from package_garage import ply_header


def fixture_ply(path,count=2):
    names=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]
    names += [f'f_rest_{i}' for i in range(45)]+['opacity']
    names += [f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    values=np.arange(2*len(names),dtype=np.float32).reshape(2,-1)/100
    values[:,names.index('rot_0'):]=[1,0,0,0]
    for i in range(3):values[:,names.index(f'scale_{i}')]=-3-i
    path.write_bytes(ply_header(tuple(names),count)+values.astype('<f4').tobytes())
    return path


def bare_model(path,budget,monkeypatch):
    def parent(self):
        self.strategy=DefaultStrategy()
        self.gauss_params=torch.nn.ParameterDict({
            k:torch.nn.Parameter(torch.zeros(shape)) for k,shape in
            dict(means=(4,3),quats=(4,4),scales=(4,3),opacities=(4,1),
                 features_dc=(4,3),features_rest=(4,15,3)).items()})
    monkeypatch.setattr(SplatfactoModel,'populate_modules',parent)
    model=LidarSplatfactoModel.__new__(LidarSplatfactoModel)
    torch.nn.Module.__init__(model)
    model.config=LidarSplatfactoModelConfig(gaussian_initialization_path=path,max_gaussians=budget)
    return model


@pytest.mark.parametrize('budget',[1,2_500_000,6_000_000])
def test_explicit_bounded_warm_budget(budget):
    assert warm_start_read_budget(budget)==dict(max_gaussians=budget,max_ply_bytes=2*1024**3)


@pytest.mark.parametrize('budget',[0,-1,6_000_001,2_500_000.,True,None])
def test_invalid_model_budget_rejected_before_parent_allocation(tmp_path,monkeypatch,budget):
    model=bare_model(tmp_path/'warm.ply',budget,monkeypatch)
    def forbidden(self):raise AssertionError('parent allocation must not run')
    monkeypatch.setattr(SplatfactoModel,'populate_modules',forbidden)
    with pytest.raises(ValueError,match='integer in 1..6000000'):
        model.populate_modules()


def test_real_model_forwards_2point5m_count_and_2gib_to_reader(tmp_path,monkeypatch):
    model=bare_model(tmp_path/'header_2499616.ply',2_500_000,monkeypatch)
    class Captured(Exception):pass
    def inspect_call(path,**kwargs):
        assert path==model.config.gaussian_initialization_path
        assert kwargs==dict(max_gaussians=2_500_000,max_ply_bytes=2*1024**3)
        raise Captured  # No multi-million-row allocations.
    monkeypatch.setattr(reader,'read_candidate',inspect_call)
    with pytest.raises(Captured):model.populate_modules()


def test_small_file_parameters_identical_and_model_preserves_every_field(tmp_path,monkeypatch):
    monkeypatch.setattr(reader,'ROOT',tmp_path)
    path=fixture_ply(tmp_path/'small.ply')
    old,old_info=reader.read_candidate(path)
    new,new_info=reader.read_candidate(path,**warm_start_read_budget(2_500_000))
    assert old_info==new_info
    assert all(np.array_equal(old[k],new[k]) for k in old)
    model=bare_model(path,2_500_000,monkeypatch);model.populate_modules()
    for name,array in old.items():
        assert torch.equal(model.gauss_params[name].detach(),torch.from_numpy(array))


def test_header_count_rejected_before_payload_or_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(reader,'ROOT',tmp_path)
    path=fixture_ply(tmp_path/'over_count.ply',count=2_500_001)
    def forbidden(*args,**kwargs):raise AssertionError('must reject before array allocation')
    monkeypatch.setattr(reader.np,'empty',forbidden)
    with pytest.raises(ValueError,match='2500000-Gaussian budget'):
        reader.read_candidate(path,**warm_start_read_budget(2_500_000))


def test_file_byte_ceiling_rejected_before_header_or_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(reader,'ROOT',tmp_path)
    path=fixture_ply(tmp_path/'over_bytes.ply')
    stat=list(reader._state(path));stat[2]=2*1024**3+1
    monkeypatch.setattr(reader,'_state',lambda p:tuple(stat))
    monkeypatch.setattr(reader,'inspect_ply',lambda *args:pytest.fail('header must not be read'))
    with pytest.raises(ValueError,match='2147483648-byte budget'):
        reader.read_candidate(path,**warm_start_read_budget(2_500_000))


@pytest.mark.parametrize('budget',[0,-1,6_000_001,2_500_000])
def test_cli_preflight_matches_model_budget(budget):
    path=ROOT/'scripts/train_lidar_block.py'
    spec=importlib.util.spec_from_file_location('warm_budget_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    args=cli.parser_for_cli().parse_args(['--dataset','data/x','--output','runs/x',
        '--warm-ply','unused.ply','--max-gaussians',str(budget)])
    if budget==2_500_000:cli.validate_args(args)
    else:
        with pytest.raises(ValueError,match='Warm Gaussian growth budget'):
            cli.validate_args(args)

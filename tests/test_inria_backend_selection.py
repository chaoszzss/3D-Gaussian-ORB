"""CPU opt-in/provenance contracts; native integration has a separate locked probe."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from orbvi_gaussian import inria_surfel_backend as backend
from orbvi_gaussian.lidar_splatfacto import LidarSplatfactoModel, LidarSplatfactoModelConfig


def load_cli():
    path=Path(__file__).resolve().parents[1]/'scripts/train_lidar_block.py'
    spec=importlib.util.spec_from_file_location('inria_cli_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_original_defaults_and_legacy_config_without_new_field():
    config=LidarSplatfactoModelConfig()
    assert config.inria_backend=='original'
    del config.__dict__['inria_backend']
    assert config.inria_backend=='original'
    assert inspect.signature(backend.render_surfel).parameters['backend'].default=='original'
    assert inspect.signature(backend.rasterization_inria_surface).parameters['backend'].default=='original'
    cli=load_cli()
    args=cli.parser_for_cli().parse_args(['--dataset','data/x','--output','runs/x'])
    assert args.inria_backend=='original'


def test_candidate_import_is_explicit_and_keeps_original_and_sys_path():
    backend._load_backend.cache_clear()
    before=list(sys.path)
    original=backend._load_backend()
    candidate=backend._load_backend('aa-depth-vjp')
    assert sys.path==before
    assert backend._load_backend() is original
    assert candidate is not original
    proof=backend.inria_backend_provenance('aa-depth-vjp')
    assert proof['validation']['forward_source_byte_identical']
    assert proof['validation']['tested_forward_arrays_bitwise_equal']
    assert len(proof['extension_sha256'])==64
    assert proof['extension_sha256']!=backend.inria_backend_provenance()['extension_sha256']
    with pytest.raises(ValueError,match='Unknown INRIA backend'):
        backend._load_backend('automatic')


def test_candidate_validation_rejects_unpinned_report_and_source(monkeypatch):
    real_sha=backend._sha
    root=backend._ROOT/'runtime/inria2dgs_aa_depth_vjp_src'
    for suffix,match in (('surfel_aa_depth_fd_gpu.json','pinned passing'),
                         ('backward.cu','source drift')):
        def changed(path):
            return '0'*64 if str(path).endswith(suffix) else real_sha(path)
        monkeypatch.setattr(backend,'_sha',changed)
        with pytest.raises(ValueError,match=match):
            backend._candidate_evidence(root)
    monkeypatch.setattr(backend,'_sha',real_sha)


@pytest.mark.parametrize('variant',['original','aa-depth-vjp'])
def test_cli_variant_reaches_yaml_config_launch_and_hashes(tmp_path,monkeypatch,variant):
    cli=load_cli()
    monkeypatch.setattr(cli,'splatfacto_command',lambda *args,**kwargs:None)
    monkeypatch.setattr(cli,'new_output_dir',lambda p:(p.mkdir(),p)[1])
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    class Captured(Exception):pass
    def capture(config):
        restored=yaml.load(yaml.dump(config),Loader=yaml.Loader)
        assert restored.pipeline.model.inria_backend==variant
        assert restored.pipeline.model.use_inria_surfel_renderer
        raise Captured
    monkeypatch.setattr(cli,'train_main',capture)
    out=tmp_path/'run'
    with pytest.raises(Captured):
        cli.main(['--dataset',str(tmp_path/'data'),'--output',str(out),
                  '--renderer-inria','--inria-backend',variant])
    launch=json.loads((out/'launch.json').read_text())
    assert launch['inria_backend']==variant
    assert launch['renderer_provenance']['variant']==variant
    assert len(launch['renderer_provenance']['source_sha256']['cuda_rasterizer/forward.cu'])==64
    assert len(launch['renderer_provenance']['extension_sha256'])==64


def test_optin_cannot_silently_select_non_inria_renderer():
    cli=load_cli()
    args=cli.parser_for_cli().parse_args(['--dataset','data/x','--output','runs/x','--inria-backend','aa-depth-vjp'])
    with pytest.raises(ValueError,match='requires --renderer-inria'):
        cli.validate_args(args)


@pytest.mark.parametrize('variant',['original','aa-depth-vjp'])
def test_real_model_call_passes_backend_and_restores_parent_renderer(monkeypatch,variant):
    import nerfstudio.models.splatfacto as ns
    model=LidarSplatfactoModel.__new__(LidarSplatfactoModel)
    torch.nn.Module.__init__(model)
    model.config=LidarSplatfactoModelConfig(use_inria_surfel_renderer=True,inria_backend=variant)
    model.eval()
    previous=ns.rasterization
    class Captured(Exception):pass
    def fake_parent(self,camera):
        keys=('means','quats','scales','opacities','colors','viewmats','Ks','width','height','sh_degree')
        return ns.rasterization(**dict.fromkeys(keys,None))
    def capture(**kwargs):
        assert kwargs['backend']==variant
        raise Captured
    monkeypatch.setattr(ns.SplatfactoModel,'get_outputs',fake_parent)
    monkeypatch.setattr(backend,'render_surfel',capture)
    with pytest.raises(Captured):model.get_outputs(None)
    assert ns.rasterization is previous


def test_evaluation_provenance_uses_config_and_legacy_fallback():
    assert backend.model_inria_backend_provenance(SimpleNamespace()) is None
    old=SimpleNamespace(use_inria_surfel_renderer=True)
    assert backend.model_inria_backend_provenance(old)['variant']=='original'
    new=SimpleNamespace(use_inria_surfel_renderer=True,inria_backend='aa-depth-vjp')
    assert backend.model_inria_backend_provenance(new)['variant']=='aa-depth-vjp'

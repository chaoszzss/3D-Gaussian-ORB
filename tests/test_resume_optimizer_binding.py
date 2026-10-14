"""Actual NS parameter replacement + Adam restoration; no GPU or dataset access."""
import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.engine.optimizers import Optimizers,AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resume_binding_subject',ROOT/'scripts/resume_lidar_block.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def model(n):
 m=SplatfactoModel.__new__(SplatfactoModel);torch.nn.Module.__init__(m)
 m.device_indicator_param=torch.nn.Parameter(torch.empty(0),requires_grad=False)
 shapes={'means':(n,3),'quats':(n,4),'scales':(n,3),'opacities':(n,1),'features_dc':(n,3),'features_rest':(n,15,3)}
 m.gauss_params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.linspace(.1,.9,torch.tensor(s).prod().item()).reshape(s)) for k,s in shapes.items()});m.step=0
 return m
class Pipeline:
 def __init__(self,m):self.model=m
 def load_pipeline(self,state,step):self.model.load_state_dict(state)
 def get_param_groups(self):return {k:[v] for k,v in self.model.gauss_params.items()}
class Trainer:
 def __init__(self,n):
  self.pipeline=Pipeline(model(n));self.config=SimpleNamespace(load_scheduler=True);self._start_step=0
  self.grad_scaler=torch.amp.GradScaler('cuda',enabled=False);self.optimizers=self.setup_optimizers()
 def setup_optimizers(self):
  return Optimizers({k:dict(optimizer=AdamOptimizerConfig(lr=.01),scheduler=ExponentialDecaySchedulerConfig(lr_final=.001,max_steps=100) if k=='means' else None) for k in self.pipeline.get_param_groups()},self.pipeline.get_param_groups())
def update(t):
 sum(p.square().sum() for p in t.pipeline.model.gauss_params.values()).backward()
 for opt in t.optimizers.optimizers.values():opt.step();opt.zero_grad()
def saved():
 t=Trainer(3);update(t)
 for scheduler in t.optimizers.schedulers.values():scheduler.step()
 s=dict(step=2000,pipeline=t.pipeline.model.state_dict(),optimizers={k:o.state_dict() for k,o in t.optimizers.optimizers.items()},schedulers={k:o.state_dict() for k,o in t.optimizers.schedulers.items()},scalers=t.grad_scaler.state_dict())
 return t,copy.deepcopy(s)
def test_actual_upstream_replaces_parameters_and_sets_inference_step():
 source,s=saved();t=Trainer(2);old=t.optimizers.optimizers['means'].param_groups[0]['params'][0]
 t.pipeline.load_pipeline(s['pipeline'],s['step'])
 assert t.pipeline.model.step==30000
 assert old is not t.pipeline.model.gauss_params['means']
 assert old.shape==(2,3) and t.pipeline.model.gauss_params['means'].shape==(3,3)

def test_restored_adam_next_update_matches_uninterrupted_model_bitwise():
 source,s=saved();t=Trainer(2);r=M.restore_checkpoint_state(t,s)
 assert (t._start_step,t.pipeline.model.step)==(2001,2000)
 for k,p in t.pipeline.model.gauss_params.items():
  opt=t.optimizers.optimizers[k];assert opt.param_groups[0]['params'][0] is p
  for name in ('step','exp_avg','exp_avg_sq'):assert torch.equal(opt.state[p][name],s['optimizers'][k]['state'][0][name])
 assert t.optimizers.schedulers['means'].state_dict()==source.optimizers.schedulers['means'].state_dict()
 update(source);update(t)
 for k,p in t.pipeline.model.gauss_params.items():assert torch.equal(p,source.pipeline.model.gauss_params[k])
 assert len(r['optimizer_binding'])==6
@pytest.mark.parametrize('field',('exp_avg','exp_avg_sq'))
def test_wrong_adam_shape_is_rejected(field):
 _,s=saved();s['optimizers']['means']['state'][0][field]=torch.zeros(2,3)
 with pytest.raises(RuntimeError,match='Adam state shape'):M.restore_checkpoint_state(Trainer(2),s)
def test_missing_optimizer_group_rejected():
 _,s=saved();s['optimizers'].pop('means')
 with pytest.raises(RuntimeError,match='optimizer groups'):M.restore_checkpoint_state(Trainer(2),s)
def test_missing_scheduler_group_rejected():
 _,s=saved();s['schedulers'].clear()
 with pytest.raises(RuntimeError,match='scheduler groups'):M.restore_checkpoint_state(Trainer(2),s)
def test_remaining_iteration_count_uses_saved_completed_step():
 assert M.remaining_iterations(2000,15000)==12999
 assert M.remaining_iterations(14999,15000)==0

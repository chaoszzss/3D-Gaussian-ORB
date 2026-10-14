"""Real CPU topology/optimizer-state regression for the Gaussian budget."""
import math

import pytest
import torch

from orbvi_gaussian.bounded_strategy import BoundedDefaultStrategy


@pytest.mark.parametrize('scales,grads,radii,maximum,expected',[
    ([.005],[1.],[.03],2,(0,1)),  # simultaneous duplicate/screen split: split wins
    ([.005],[1.],[0.],2,(1,0)),
    ([.02],[1.],[0.],2,(0,1)),
    ([.005],[1.],[.03],1,(0,0)),  # full budget cannot allocate
    ([.005,.02,.005,.02],[1.,2.,3.,4.],[.03,0.,0.,.03],6,(1,1)),
    ([.005,.005],[0.,0.],[.03,.04],3,(0,1)),  # screen-only budget truncation
    ([.005,.02],[0.,0.],[0.,0.],10,(0,0)),
])
def test_budget_net_growth_and_optimizer_shapes(scales,grads,radii,maximum,expected):
    n=len(scales)
    params=torch.nn.ParameterDict({
        'means':torch.nn.Parameter(torch.ones(n,3)),
        'scales':torch.nn.Parameter(torch.tensor(scales).log()[:,None].repeat(1,3)),
        'quats':torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]]*n)),
        'opacities':torch.nn.Parameter(torch.full((n,1),3.)),
    })
    optimizers={key:torch.optim.Adam([value],lr=.001) for key,value in params.items()}
    sum(value.square().sum() for value in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step();optimizer.zero_grad()
    with torch.no_grad():
        params['scales'].copy_(torch.tensor(scales).log()[:,None].repeat(1,3))
    strategy=BoundedDefaultStrategy(grow_grad2d=.0008,grow_scale3d=.01,
        grow_scale2d=.02,refine_scale2d_stop_iter=11000)
    strategy.max_gaussians=maximum
    state=strategy.initialize_state(scene_scale=1.)
    state.update(grad2d=torch.tensor(grads),count=torch.ones(n),radii=torch.tensor(radii))
    before={key:value.detach().clone() for key,value in params.items()}
    duplicated,split=strategy._grow_gs(params,optimizers,state,600)
    assert (duplicated,split)==expected
    assert len(params['means'])==n+duplicated+split<=maximum
    for key,value in params.items():
        optimizer=optimizers[key]
        assert optimizer.param_groups[0]['params'][0] is value
        assert optimizer.state[value]['exp_avg'].shape==value.shape
        assert optimizer.state[value]['exp_avg_sq'].shape==value.shape
        assert torch.isfinite(value).all()
        if expected==(0,0):
            assert torch.equal(value,before[key])
    assert all(len(state[key])==len(params['means']) for key in ('grad2d','count','radii'))
    if n==1 and radii[0]>.02 and maximum==2:
        # The original overlapping parent is gone; both records are children.
        torch.testing.assert_close(params['scales'],
            torch.full((2,3),math.log(scales[0]/1.6)))
    # The remapped optimizer states must remain usable, not merely shaped right.
    sum(value.square().sum() for value in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()


def test_screen_stop_retains_small_gradient_duplicate_semantics():
    params=torch.nn.ParameterDict({
        'means':torch.nn.Parameter(torch.zeros(1,3)),
        'scales':torch.nn.Parameter(torch.full((1,3),math.log(.005))),
        'quats':torch.nn.Parameter(torch.tensor([[1.,0,0,0]])),
        'opacities':torch.nn.Parameter(torch.zeros(1,1)),
    })
    optimizers={key:torch.optim.Adam([value]) for key,value in params.items()}
    strategy=BoundedDefaultStrategy(grow_scale2d=.02,refine_scale2d_stop_iter=11000)
    strategy.max_gaussians=2
    state=strategy.initialize_state(scene_scale=1.)
    state.update(grad2d=torch.ones(1),count=torch.ones(1),radii=torch.tensor([.03]))
    assert strategy._grow_gs(params,optimizers,state,11000)==(1,0)
    assert len(params['means'])==2

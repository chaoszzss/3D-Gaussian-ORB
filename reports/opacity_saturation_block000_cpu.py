"""Bounded completed-run CPU activation/reset audit; no renderer or training."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from gsplat.strategy.ops import reset_opa
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):d.update(chunk)
    return d.hexdigest()


def quantiles(value):
    q=(0.,.1,.5,.9,.99,1.)
    return {str(p):float(v) for p,v in zip(q,torch.quantile(value.flatten(),torch.tensor(q)))}


def run(name):
    folder=ROOT/'runs/fidelity_v2'/name
    result=json.loads((folder/'training_result.json').read_text())
    launch=json.loads((folder/'launch.json').read_text())
    assert result['status']=='trained' and result['checkpoint_step']==14999
    path=ROOT/result['checkpoint'];stat=path.stat()
    assert sha(path)==result['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert state['step']==14999
    logits=next(v for k,v in state['pipeline'].items() if k.endswith('gauss_params.opacities'))
    scales=next(v for k,v in state['pipeline'].items() if k.endswith('gauss_params.scales'))
    p=torch.nn.Parameter(logits.clone())
    alpha=p.sigmoid();gradient,=torch.autograd.grad(alpha.sum(),p)
    count=len(p);optimizer=torch.optim.Adam([p],lr=.05)
    optimizer.state[p]=dict(step=torch.tensor(1.),exp_avg=torch.ones_like(p),exp_avg_sq=torch.ones_like(p))
    params={'opacities':p};optimizers={'opacities':optimizer}
    before=dict(exact_alpha_one_count=int((alpha==1).sum()),exact_zero_activation_vjp_count=int((gradient==0).sum()),
        fraction_exact_zero_activation_vjp=float((gradient==0).double().mean()),
        fraction_activation_vjp_lt_1e6=float((gradient<1e-6).double().mean()),
        logits_quantiles=quantiles(logits),activation_vjp_quantiles=quantiles(gradient),
        prunable_below_point1=int((alpha<.1).sum()))
    # Even ideally correctly-rounded sigmoid is indistinguishable from 1 above
    # this fp32 threshold. Exact CPU/CUDA implementations can differ near it.
    threshold=math.log((1.-2.**-25)/2.**-25)
    before['correctly_rounded_fp32_alpha_one_logit_threshold']=threshold
    before['fraction_logit_above_correct_rounding_threshold']=float((logits>threshold).double().mean())
    reset_opa(params,optimizers,{},value=.2)  # Same cull_alpha_thresh*.2/.1 ratio as current NS strategy.
    reset=params['opacities'];a=reset.sigmoid();g,=torch.autograd.grad(a.sum(),reset)
    after=dict(count=len(reset),max_alpha=float(a.max()),zero_activation_vjp_count=int((g==0).sum()),
        activation_vjp_quantiles=quantiles(g),prunable_below_point1=int((a<.1).sum()),
        optimizer_momentum_zero=bool((optimizer.state[reset]['exp_avg']==0).all()
            and (optimizer.state[reset]['exp_avg_sq']==0).all()),immediate_capacity_freed=count-len(reset))
    # A two-layer local example with equal RGB isolates an occlusion correction
    # direction. This is not a native raster or scene gradient measurement.
    toy=[]
    for label,value in [('checkpoint_median',float(logits.median())),('reset_point2',float(torch.logit(torch.tensor(.2))))]:
        x=torch.tensor(value,requires_grad=True);af=.8*x.sigmoid();ab=torch.tensor(.99)
        accum=af+(1-af)*ab
        depth=(af*3+(1-af)*ab*10)/accum
        metric=.05*torch.nn.functional.huber_loss(depth,torch.tensor(10.),delta=.15)
        grad,=torch.autograd.grad(metric,x)
        rgb=.5*accum
        toy.append(dict(state=label,expected_depth=float(depth.detach()),rgb=float(rgb.detach()),
            metric_loss=float(metric.detach()),d_metric_d_foreground_logit=float(grad)))
    event_file=next(folder.rglob('events.out.tfevents.*'))
    events=EventAccumulator(str(event_file),size_guidance={'scalars':0});events.Reload()
    history={}
    for tag in events.Tags()['scalars']:
        if not tag.startswith('Train Loss Dict/'):continue
        rows=events.Scalars(tag)
        history[tag]=[{ 'start':lo,'stop':hi,'logged_samples':len(v),
            'mean':float(np.mean(v))} for lo,hi in [(0,1000),(1000,5000),(5000,10000),(10000,15000)]
            for v in [[r.value for r in rows if lo<=r.step<hi]]]
    refine=result['refinement'];metadata=json.loads((Path(launch['dataset'])/'transforms.json').read_text())
    ntrain=len(metadata['train_filenames']);pause=ntrain+100
    assert (path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
    return dict(run=name,checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],
        checkpoint_step=state['step'],actual_count=count,training_reset_enabled=launch['reset_opacity'],
        initialization=launch['initialization_origin'],before=before,after_standard_reset_cpu=after,
        local_two_layer_example=toy,loss_history_logged_means=history,
        unused_normal_scale_m_quantiles=quantiles(scales[:,2].exp()),
        refinement={k:refine[k] for k in ('initial_gaussians','split_total','duplicated_total','pruned_total')},
        first_95percent_budget_event=next(e for e in refine['events'] if e['after']>.95*launch['max_gaussians']),
        train_frames=ntrain,standard_reset_period=3000,refine_pause_after_reset=pause,
        first_refine_after_step3000=3000+math.ceil(pause/100)*100)


if __name__=='__main__':
    started=time.monotonic();torch.set_num_threads(3)
    report=dict(status='CPU_ACTIVATION_SATURATION_AND_RESET_MEASURED',gpu_used=False,
        scope='Actual completed checkpoint logits, CPU float32 sigmoid VJP and actual gsplat reset_opa on disposable CPU clones. Not native scene parameter gradients, not evidence reset improves reconstruction or frees capacity.',
        runs=[run(n) for n in ('train_block000_depthv3_seed_aa_cap003','train_joint015_depthv3_aa_metric_cap003')],
        recommendation='One-variable fresh-seed reset ablation using existing --reset-opacity; do not change losses, teacher, split rules, or defaults. Initial opacity .1 is not reduced by step0 reset cap .2. Later reset permits opacity updates/pruning but does not itself remove any Gaussian.',
        code_sha256=sha(__file__),command=[sys.executable,*sys.argv],elapsed_seconds=time.monotonic()-started)
    out=ROOT/'reports/opacity_saturation_block000_cpu.json'
    with out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(out)

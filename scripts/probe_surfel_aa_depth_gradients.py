#!/usr/bin/env python3
"""Tiny native finite differences with fixed AA branch and unsaturated opacity.

Runs only under inherited project GPU lock. CPU mode checks fixture margins.
Original and optional isolated candidate are separately imported, never swapped.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np


def fixtures():
    result=[]
    for name,angle,scale,z,branch in (
        ('aa_grazing85',85.,.03,3.,'aa'),('aa_grazing89',89.,.03,10.,'aa'),
        ('ray_branch_control',30.,.1,3.,'ray')):
        half=math.radians(angle)/2
        result.append(dict(name=name,expected_branch=branch,means=[0.,0.,z],
            scales=[scale,scale],quats=[math.cos(half),0.,math.sin(half),0.],opacities=[.45]))
    return result


def branch_measurement(case):
    """Only forward fixture classification; main owns independent VJP proof."""
    w,x,y,z=np.asarray(case['quats'])/np.linalg.norm(case['quats'])
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    matrix=np.column_stack((rotation[:,:2]*np.asarray(case['scales']),case['means']))
    Tu,Tv,Tw=np.array([[64.,0,16.],[0,64.,16.],[0,0,1.]])@matrix
    p=np.cross(17*Tw-Tu,16*Tw-Tv)
    s=p[:2]/p[2]
    f=np.array([9.,9.,-1.]);f=f/np.dot(f,Tw*Tw)
    center=np.array([sum(f*Tu*Tw),sum(f*Tv*Tw)])
    rho3=float(s@s);rho2=float(2*np.sum((center-[17,16])**2))
    depth=float(Tw@np.r_[s,1.])
    alpha=float(case['opacities'][0]*np.exp(-.5*min(rho3,rho2)))
    return dict(branch='ray' if rho3<=rho2 else 'aa',rho3d=rho3,rho2d=rho2,
                branch_margin=abs(rho3-rho2),expected_depth=depth,alpha=alpha,
                above_near=depth>.2,opacity_unsaturated=.01<alpha<.8)


def perturb(case,key,index,amount):
    changed={k:(list(v) if isinstance(v,list) else v) for k,v in case.items()}
    changed[key][index]+=amount
    return changed


DIRECTIONS=(('means',0),('means',2),('scales',0),('quats',2),('opacities',0))


def epsilon_values(key):
    return (.001,.0005,.00025) if key=='quats' else (.002,.001,.0005)


def fixture_checks():
    records=[]
    for case in fixtures():
        measures=[branch_measurement(case)]
        for key,index in DIRECTIONS:
            for epsilon in epsilon_values(key):
                measures.extend(branch_measurement(perturb(case,key,index,sign*epsilon)) for sign in (-1,1))
        assert all(m['branch']==case['expected_branch'] and m['branch_margin']>.05
                   and m['above_near'] and m['opacity_unsaturated'] for m in measures)
        records.append(dict(fixture=case,base=measures[0],checked_states=len(measures),
                            minimum_branch_margin=min(m['branch_margin'] for m in measures)))
    return records


def run_module(module_name):
    import torch
    from torch.nn import functional as F
    from orbvi_gaussian.inria_surfel_backend import _projection
    module=importlib.import_module(module_name)
    device=torch.device('cuda')
    K=torch.tensor([[64.,0,16.5],[0,64.,16.5],[0,0,1.]],device=device)
    settings=module.GaussianRasterizationSettings(image_height=32,image_width=32,
        tanfovx=.25,tanfovy=.25,bg=torch.zeros(3,device=device),scale_modifier=1.,
        viewmatrix=torch.eye(4,device=device),projmatrix=_projection(K,32,32,.5).T.contiguous(),
        sh_degree=0,campos=torch.zeros(3,device=device),prefiltered=False,debug=False)
    def render(case,grad):
        params={key:torch.tensor([case[key]],device=device,dtype=torch.float32,requires_grad=grad)
                for key in ('means','scales','quats','opacities')}
        color,radii,raw=module.GaussianRasterizer(settings)(means3D=params['means'],
            means2D=torch.zeros(1,3,device=device,requires_grad=grad),
            opacities=params['opacities'],colors_precomp=torch.tensor([[.3,.5,.7]],device=device),
            scales=params['scales'],rotations=F.normalize(params['quats'],dim=-1))
        value=raw[0,16,17]/raw[1,16,17]
        return value,params,(color,radii,raw)
    records=[];forward_arrays=[];coverage_grads=[]
    for case in fixtures():
        value,params,outputs=render(case,True)
        gradients=torch.autograd.grad(value,list(params.values()))
        grads={key:g.detach().cpu().numpy()[0] for key,g in zip(params,gradients)}
        color,radii,raw=outputs
        base_alpha=float(raw[1,16,17].detach());base_depth=float(value.detach())
        assert .01<base_alpha<.8 and int(radii[0])>0
        oracle=branch_measurement(case)
        assert abs(base_depth-oracle['expected_depth'])<.0001
        forward_arrays.append((color.detach().cpu().numpy(),raw.detach().cpu().numpy()))
        directions=[]
        for key,index in DIRECTIONS:
            finite=[];perturbed_states=[]
            for epsilon in epsilon_values(key):
                values=[]
                for sign in (-1,1):
                    changed=perturb(case,key,index,sign*epsilon)
                    classification=branch_measurement(changed)
                    assert classification['branch']==case['expected_branch']
                    with torch.no_grad():
                        actual,_,(_,r,a)=render(changed,False)
                    assert .01<float(a[1,16,17])<.8 and int(r[0])>0 and float(actual)>.2
                    values.append(float(actual))
                    perturbed_states.append(dict(epsilon=sign*epsilon,actual_alpha=float(a[1,16,17]),
                        actual_depth=float(actual),**classification))
                # Match the actual float32 tensor perturbation, not a nominal
                # double epsilon that could differ by a float32 ULP.
                delta=float(np.float32(case[key][index]+epsilon)-np.float32(case[key][index]-epsilon))
                finite.append(dict(epsilon=epsilon,gradient=(values[1]-values[0])/delta))
            fd=float(np.median([f['gradient'] for f in finite]));analytic=float(grads[key][index])
            spread=float(np.ptp([f['gradient'] for f in finite]))
            tolerance=.005+.01*abs(fd)
            stable=spread<=.005+.02*abs(fd)
            directions.append(dict(parameter=key,index=index,autograd=analytic,finite_differences=finite,
                median_fd=fd,fd_spread=spread,fd_converged=stable,error_abs=abs(analytic-fd),
                comparison_tolerance=tolerance,passes=stable and abs(analytic-fd)<=tolerance,
                perturbations=perturbed_states))
        records.append(dict(name=case['name'],branch=case['expected_branch'],base_depth=base_depth,
            base_alpha=base_alpha,oracle=oracle,directions=directions,
            all_pass=all(d['passes'] for d in directions)))
        print(module_name,case['name'],[(r['parameter'],r['autograd'],r['median_fd'],r['passes']) for r in directions],flush=True)
        # A pure RGB+alpha objective must retain the existing low-pass VJP.
        _,p,images=render(case,True)
        c,_,a=images
        coverage_loss=c[:,16,17].sum()+.37*a[1,16,17]
        cg=torch.autograd.grad(coverage_loss,list(p.values()))
        coverage_grads.append([g.detach().cpu().numpy() for g in cg])
    return dict(module=module_name,wrapper_path=module.__file__,extension_path=module._C.__file__,
                extension_sha256=hashlib.sha256(Path(module._C.__file__).read_bytes()).hexdigest(),cases=records),forward_arrays,coverage_grads


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-root',type=Path)
    parser.add_argument('--lock-fd',type=int,default=9)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args();cpu=fixture_checks();started=time.monotonic()
    if args.cpu_only:
        args.output.write_text(json.dumps(dict(status='CPU_FIXTURE_MARGINS_VERIFIED',cases=cpu),indent=2)+'\n')
        print('CPU fixture checks passed');return
    from audit_train_parameter_gradients import require_inherited_lock
    require_inherited_lock(args.lock_fd)
    from orbvi_gaussian.training import confined_environment
    import os
    os.environ.update(confined_environment())
    import torch
    torch.cuda.reset_peak_memory_stats()
    original,base,original_coverage=run_module('diff_surfel_rasterization')
    controls=[r for r in original['cases'] if r['branch']=='ray']
    aa=[r for r in original['cases'] if r['branch']=='aa']
    confirmed=all(r['all_pass'] for r in controls) and all(
        any(d['parameter']=='means' and d['index']==0 and d['fd_converged'] and d['error_abs']>1 for d in r['directions']) for r in aa)
    report=dict(status='ORIGINAL_AA_DEPTH_VJP_MISMATCH_CONFIRMED' if confirmed else 'INCONCLUSIVE',
        gpu_used=True,original=original,cpu_fixtures=cpu,
        created_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),ROOT/'runtime/inria2dgs_src/cuda_rasterizer/forward.cu',ROOT/'runtime/inria2dgs_src/cuda_rasterizer/backward.cu')},
        comparison='Single-GS expected depth D/alpha at a fixed pixel; AA branch stable for every ±epsilon, no clipping/opacity saturation/transmittance early stop. Control uses ray branch. Gradients target actual means/scales/unit-normalized quaternion/activated opacity.',
        limitation='Confirms local derivative consistency only; does not prove reconstruction improvement.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if args.candidate_root is not None:
        if not confirmed:
            raise RuntimeError('Original local defect not confirmed; refuse candidate pass claim')
        sys.path.insert(0,str(args.candidate_root.resolve()))
        candidate,changed,candidate_coverage=run_module('diff_surfel_rasterization_aa_depth_vjp')
        equal=all(np.array_equal(x,y) for pair1,pair2 in zip(base,changed) for x,y in zip(pair1,pair2))
        coverage_equal=all(np.array_equal(x,y) for pair1,pair2 in zip(original_coverage,candidate_coverage) for x,y in zip(pair1,pair2))
        passed=equal and coverage_equal and all(row['all_pass'] for row in candidate['cases'])
        report.update(candidate=candidate,forward_arrays_bitwise_equal=equal,
                      rgb_alpha_parameter_gradients_bitwise_equal=coverage_equal,
                      candidate_status='PASS' if passed else 'FAIL')
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        if not passed:
            raise RuntimeError('Isolated candidate failed finite differences or forward parity')
    report.update(elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],flush=True)


if __name__=='__main__':
    main()

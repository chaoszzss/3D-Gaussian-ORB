#!/usr/bin/env python3
"""Break ray/pose ambiguity with fixed photo tracks and measured map surfaces."""
import sys,os,json,copy,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,yaml
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from orbvi_gaussian.geometry import triple_sphere_unproject
from orbvi_gaussian.workflow import load_trajectory
from orbvi_gaussian.io import new_output_dir


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--initial',type=Path,default=ROOT/'configs/direct_lio_refined.experimental.yaml')
    a=p.parse_args();out=new_output_dir(a.output)
    r=ROOT/'worktrees/rig-refine/reports/rig_refine'
    manifest=json.loads((r/'pairs_manifest.json').read_text())['pairs'];cache=np.load(r/'cache/correspondences.npz')
    trajectory=load_trajectory(ROOT/'data/all_lio_candidate/odometry.csv')
    reference=np.load(ROOT/'data/reference_map_voxel01.npz')['xyz'];tree=cKDTree(reference)
    cfg=yaml.safe_load(a.initial.read_text());reports={}
    for cam in cfg['cameras']:
        c=cfg['cameras'][cam];i=c['intrinsics'];K=np.array([[i['fx'],0,i['cx']],[0,i['fy'],i['cy']],[0,0,1]])
        pars=dict(zip(['xi','lambda','alpha','b','c'],c['distortion']));base=np.array(c['T_body_camera'])
        groups={'train':[],'dev':[]}
        for e in manifest:
            key=cam+'_%03d'%e['slot']
            if e['camera_id']!=cam or e['split'] not in groups or e['translation_m']<.5 or key+'_a' not in cache:continue
            u,v=cache[key+'_a'],cache[key+'_b'];r1,ok1=triple_sphere_unproject(u,K,pars);r2,ok2=triple_sphere_unproject(v,K,pars)
            T1=trajectory.at(e['header_ns'][0]);D=np.array(e['body_2_from_body_1']);groups[e['split']].append((r1[ok1&ok2],r2[ok1&ok2],T1,D,e['slot']))
        def points(group,x):
            T=base.copy();T[:3,:3]=base[:3,:3]@Rotation.from_rotvec(np.deg2rad(x)).as_matrix()
            result=[];gaps=[];positive=[]
            for r1,r2,T1,D,slot in group:
                C=np.linalg.inv(T)@np.linalg.inv(D)@T;b=r2@C[:3,:3].T;t=C[:3,3]
                dot=(r1*b).sum(1);den=np.maximum(1-dot*dot,1e-7);xx=r1@t;yy=b@t
                l1=(xx-dot*yy)/den;l2=(dot*xx-yy)/den
                valid=(den>1e-5)&(l1>1)&(l1<35)&(l2>1)&(l2<35)
                W=T1@T;world=(r1*l1[:,None])@W[:3,:3].T+W[:3,3]
                result.append(world);gaps.append(np.linalg.norm(r1*l1[:,None]-b*l2[:,None]-t,axis=1));positive.append(valid)
            return np.vstack(result),np.concatenate(gaps),np.concatenate(positive)
        # Freeze support once, using only training tracks and the initial model.
        initial,gap,valid=points(groups['train'],np.zeros(3));valid &= gap<.15
        _,neighbors=tree.query(initial,k=8,workers=4);near=reference[neighbors];center=near.mean(1)
        _,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',near-center[:,None],near-center[:,None]))
        normal=vec[:,:,0]
        def objective(x):
            q,g,p=points(groups['train'],x)
            _,ids=tree.query(q,workers=4)
            _,nb=tree.query(reference[ids],k=8,workers=4)
            local=reference[nb];ctr=local.mean(1)
            _,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',local-ctr[:,None],local-ctr[:,None]))
            signed=((q-ctr)*vectors[:,:,0]).sum(1)
            return np.r_[signed[valid]/.2,g[valid]/.1,x/5.]
        fit=least_squares(objective,np.zeros(3),bounds=(-12,12),loss='soft_l1',f_scale=1.,max_nfev=25,diff_step=.01)
        def score(group,x):
            q,g,v=points(group,x);d=tree.query(q,workers=4)[0];ok=v&(g<.15)
            return {'tracks':int(ok.sum()),'median_measured_surface_nn_m':float(np.median(d[ok])),
                    'p95_measured_surface_nn_m':float(np.percentile(d[ok],95)),'median_ray_gap_m':float(np.median(g[ok]))}
        report={'rotation_delta_deg':fit.x.tolist(),'train_before':score(groups['train'],np.zeros(3)),
                'train_after':score(groups['train'],fit.x),'dev_before':score(groups['dev'],np.zeros(3)),
                'dev_after':score(groups['dev'],fit.x),'nfev':fit.nfev}
        reports[cam]=report;print(cam,json.dumps(report),flush=True)
        T=base.copy();T[:3,:3]=base[:3,:3]@Rotation.from_rotvec(np.deg2rad(fit.x)).as_matrix();c['T_body_camera']=T.tolist()
    cfg['verified']=False;cfg['calibration_id']='measured-scene-surface-calibration-candidate'
    cfg['provenance']['scene_surface_refinement']={'report':str(out/'report.json'),'scope':'train/dev photographic triangles conditioned on LIO/map; no independent GT'}
    (out/'candidate.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    (out/'report.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()

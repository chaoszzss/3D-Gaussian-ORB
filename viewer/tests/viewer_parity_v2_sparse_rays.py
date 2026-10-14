#!/usr/bin/env python3
"""Bounded CPU sparse rays, frozen worst pixels. No CUDA, fit or source mutation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import cv2
from scipy.special import expit
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from assemble_core_gaussians_lidar_ownership import surfel_projection,ray_alpha,sh_rgb
from package_garage import inspect_ply
OUT=ROOT/'reports/viewer_parity_v2_sparse'
BG=np.array([.149,.1647,.2157],np.float32).astype(float)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def geometry(projection,pixel):
    tu,tv,tw,image=projection[:4];x,y=pixel
    cross=np.cross(x*tw-tu,y*tw-tv)
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=cross[:,:2]/cross[:,2:3]
        rho=(uv*uv).sum(1);rho2=2*((image-pixel)**2).sum(1)
        depth=(uv*tw[:,:2]).sum(1)+tw[:,2]
    return uv,rho,rho2,depth

def compose(ids,alpha,depth,rgb,z,native=True):
    order=np.lexsort((ids,z[ids]));ids=ids[order];a=alpha[order];d=depth[order]
    after=np.cumprod(1-a);stop=np.flatnonzero(after<1e-4)
    limit=int(stop[0]) if native and len(stop) else len(ids)
    ids,a,d=ids[:limit],a[:limit],d[:limit]
    weight=a*np.r_[1.,np.cumprod(1-a[:-1])] if limit else np.empty(0)
    raw=weight@rgb[ids]+(1-weight.sum())*BG
    return {'ids':ids,'alpha':a,'weight':weight,'depth':d,'rgb':np.clip(raw,0,1),'unclipped_rgb':raw,
        'accumulation':float(weight.sum()),'expected_depth':float(weight@d/max(weight.sum(),1e-12))}

def spark_rays(proj,opacity,rgb,camera,pixel,packing=False):
    uv,rho,rho2,depth=geometry(proj,pixel);tu,tv,tw=proj[:3];z=tw[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        x=(tu[:,2]/z-(camera['w']-1)/2)/(camera['w']/2)
        y=(tv[:,2]/z-(camera['h']-1)/2)/(camera['h']/2)
        a=opacity*np.exp(-.5*rho)
    center=(z>.2)&(z<100)&(abs(x)<=1.4)&(abs(y)<=1.4)
    keep=center&(depth>=.2)&(depth<=100)&(rho<=9)&(a>=1/255)&np.isfinite(a)
    ids=np.flatnonzero(keep)
    return compose(ids,a[ids],depth[ids],rgb,z,native=False),{'center':center,'rho':rho,'rho2':rho2,'depth':depth,'alpha':a,'keep':keep,'uv':uv}

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    mp=ROOT/'runs/fidelity_v2/original_single_blocks/block_015/metrics.json';metrics=json.loads(mp.read_text())
    cp=ROOT/'reports/viewer_parity_v2_hdr015_three/browser_capture.json';capture=json.loads(cp.read_text())
    ply=Path(metrics['source_ply']['path']);assert sha(ply)==metrics['source_ply']['sha256']
    with ply.open('rb') as f:spec=inspect_ply(f);rows=np.fromfile(f,dtype='<f4').reshape(spec['count'],-1).astype(float)
    names=spec['names'];xyz=rows[:,:3];scales=np.exp(rows[:,[names.index(f'scale_{i}') for i in range(3)]].astype(np.float32)).astype(float)
    quats=rows[:,[names.index(f'rot_{i}') for i in range(4)]];opacity=expit(rows[:,names.index('opacity')].astype(np.float32)).astype(float)
    freeze=OUT/'pixels.json';inputs=[]
    for idx in (5,9,8):
        row=next(x for x in metrics['frames'] if x['dataset_index']==idx)
        npz=mp.parent/f'{idx:03d}_float.npz';png=ROOT/f'reports/viewer_parity_v2_hdr015_three/{idx:03d}_browser.png'
        f=np.load(npz);b=cv2.imread(str(png))[:,:,::-1]/255;mask=cv2.imread(row['acceptance_mask_path'],0)>0
        error=((b-f['rgb'])**2).mean(2);valid=np.flatnonzero(mask);rank=np.lexsort((valid,-error.ravel()[valid]));flat=valid[rank[:10]]
        pixels=np.column_stack([flat%mask.shape[1],flat//mask.shape[1]]).tolist()
        inputs.append({'dataset_index':idx,'pixel_xy':pixels,'selection':'top10 masked RGB MSE, descending; row-major tie break; v2 capture frozen before diagnosis',
            'native_float_sha256':sha(npz),'browser_png_sha256':sha(png),'mask_sha256':sha(Path(row['acceptance_mask_path']))})
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save(freeze,inputs)
    report={'cpu_only':True,'source_ply_sha256':sha(ply),'frozen_pixels_sha256':sha(freeze),'rays':[]}
    cpu_cases=[]
    packedpath=OUT/'spark_decoded.bin';packed=None
    if packedpath.exists():
        decoded=json.loads((OUT/'spark_decoded.json').read_text())
        assert decoded['ply_sha256']==report['source_ply_sha256'] and decoded['count']==len(rows) and decoded['width']==62
        assert decoded['bin_sha256']==sha(packedpath)
        packed=np.fromfile(packedpath,dtype='<f4').reshape(len(rows),62).astype(float)
    for frozen in inputs:
        idx=frozen['dataset_index'];row=next(x for x in metrics['frames'] if x['dataset_index']==idx);camera=row['camera'];pose=np.array(camera['camera_to_worlds_ns']);proj=surfel_projection(xyz,scales,quats,camera)
        rgb=sh_rgb(rows,names,pose[:,3]);f=np.load(mp.parent/f'{idx:03d}_float.npz');b=cv2.imread(str(ROOT/f'reports/viewer_parity_v2_hdr015_three/{idx:03d}_browser.png'))[:,:,::-1]/255
        if packed is not None:
            pp=surfel_projection(packed[:,:3],packed[:,3:6],packed[:,6:10],camera)
            # packed file columns: means3, activated scales3, wxyz4, opacity1, dcRGB3, restored SH15x3.
            rr=rows.copy();rr[:,:3]=packed[:,:3]
            rr[:,[names.index(f'f_dc_{i}') for i in range(3)]]=(packed[:,11:14]-.5)/.28209479177387814
            rr[:,[names.index(f'f_rest_{i}') for i in range(45)]]=packed[:,14:59].reshape(-1,15,3).transpose(0,2,1).reshape(-1,45)
            prgb=sh_rgb(rr,names,pose[:,3]).astype(np.float16).astype(float) # accumulator half RGB after SH
        for pixel in frozen['pixel_xy']:
            x,y=pixel;ids,a,d=ray_alpha(proj,opacity,pixel);native=compose(ids,a,d,rgb,proj[2][:,2]);spark,detail=spark_rays(proj,opacity,rgb,camera,pixel)
            uv,rho,rho2,depth=geometry(proj,pixel)
            variants={}
            for label,alph,keep in [
              ('native_without_lowpass',np.minimum(.99,opacity[ids]*np.exp(-.5*rho[ids])),np.ones(len(ids),bool)),
              ('native_rho3sigma_only',a,rho[ids]<=9),
              ('native_spark_center_clip_only',a,detail['center'][ids]),
              ('native_spark_ray_far_clip_only',a,depth[ids]<=100),
              ('native_no_alpha_cap',opacity[ids]*np.exp(-.5*np.minimum(rho[ids],rho2[ids])),np.ones(len(ids),bool))]:
                keep=keep&(alph>=1/255);c=compose(ids[keep],alph[keep],d[keep],rgb,proj[2][:,2]);variants[label]=c['rgb'].tolist()
            contrib=[]
            for k in np.argsort(-native['weight'])[:12]:
                g=int(native['ids'][k]);contrib.append({'id':g,'weight':float(native['weight'][k]),'native_alpha':float(native['alpha'][k]),
                    'rgb':rgb[g].tolist(),'rho3d':float(rho[g]),'rho2d':float(rho2[g]),'uv':uv[g].tolist(),'AA':bool(rho2[g]<rho[g]),
                    'center_z':float(proj[2][g,2]),'ray_depth':float(depth[g]),'spark_center_keep':bool(detail['center'][g]),
                    'spark_ideal_keep':bool(detail['keep'][g]),'spark_ideal_alpha':float(detail['alpha'][g])})
            record={'dataset_index':idx,'pixel_xy':pixel,'native_rgb':f['rgb'][y,x].tolist(),'browser_v2_rgb':b[y,x].tolist(),
                'native_cpu_rgb':native['rgb'].tolist(),'native_cpu_rgb_maxerr':float(abs(native['rgb']-f['rgb'][y,x]).max()),
                'native_cpu_alpha_error':abs(native['accumulation']-float(f['alpha'][y,x,0])),
                'native_cpu_expected_depth_error':abs(native['expected_depth']-float(f['depth'][y,x,0])),
                'native_contributors':len(native['ids']),'native_AA_weight':float(native['weight'][rho2[native['ids']]<rho[native['ids']]].sum()),
                'spark_ideal_rgb':spark['rgb'].tolist(),'spark_ideal_contributors':len(spark['ids']),
                'variants':variants,'dominant_native':contrib}
            if packed is not None:
                ps,pd=spark_rays(pp,packed[:,10],prgb,camera,pixel);record['spark_packed_rgb']=ps['rgb'].tolist();record['spark_packed_maxerr']=float(abs(ps['rgb']-b[y,x]).max())
                vi,va,vd=ray_alpha(pp,packed[:,10],pixel)
                # v3 follows native coverage/rho and alpha cap but keeps Spark's
                # back-to-front compositing (no native early termination).
                v3=compose(vi,va,vd,prgb,pp[2][:,2],native=False)
                record['v3_cpu_predicted_rgb']=v3['rgb'].tolist()
                record['v3_cpu_native_maxerr']=float(abs(v3['rgb']-f['rgb'][y,x]).max())
                record['v3_cpu_contributors']=len(v3['ids'])
                native_all=compose(vi,va,vd,prgb,pp[2][:,2],native=True)
                record['v3_early_termination_only_delta_max']=float(abs(v3['rgb']-native_all['rgb']).max())
                # Exact reference T/AA alpha for CPU shader projection closure.
                samples=np.unique(np.r_[native['ids'],ps['ids'],v3['ids']])
                uvp,rhop,rho2p,dp=geometry(pp,pixel)
                for g in samples:
                    cpu_cases.append({'frame':idx,'camera':camera,'pixel':pixel,'id':int(g),
                        'center':packed[g,:3].tolist(),'scales':packed[g,3:6].tolist(),'quaternion_wxyz':packed[g,6:10].tolist(),'opacity':float(packed[g,10]),
                        'T':[pp[j][g].tolist() for j in range(3)],'aabb_image':pp[3][g].tolist(),'radius':float(pp[4][g]),
                        'native_active':bool(np.any(vi==g)), 'native_alpha':float(va[np.where(vi==g)[0][0]]) if np.any(vi==g) else 0,
                        'rho3d':float(rhop[g]),'rho2d':float(rho2p[g]),'ray_depth':float(dp[g])})
                for c in contrib:
                    g=c['id'];c.update(spark_packed_rgb=prgb[g].tolist(),spark_packed_rho=float(pd['rho'][g]),spark_packed_alpha=float(pd['alpha'][g]),spark_packed_keep=bool(pd['keep'][g]))
            report['rays'].append(record)
    report['seconds']=time.monotonic()-start
    save(OUT/'rays.json',report)
    if cpu_cases:save(OUT/'cpu_cases.json',cpu_cases)
    for idx in (5,9,8):
        rr=[r for r in report['rays'] if r['dataset_index']==idx]
        print(idx,'native max',max(r['native_cpu_rgb_maxerr'] for r in rr),'AAmean',np.mean([r['native_AA_weight'] for r in rr]),'packed max',max(r.get('spark_packed_maxerr',0) for r in rr))
        print('v3 CPU maxerr',max(r.get('v3_cpu_native_maxerr',0) for r in rr))
if __name__=='__main__':main()

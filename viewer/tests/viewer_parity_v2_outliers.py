#!/usr/bin/env python3
"""CPU-only stratified original015 parity diagnosis; no fitting or GPU imports."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import cv2
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_gaussian_ply import read_candidate

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def sh_rgb(p, camera):
    d = p['means'].astype(np.float64)-camera
    d /= np.linalg.norm(d,axis=1)[:,None]
    x,y,z=d.T; xx,yy,zz=x*x,y*y,z*z
    b=np.stack([np.full(len(x),.28209479177387814),-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x,
       1.0925484305920792*x*y,-1.0925484305920792*y*z,.31539156525252005*(2*zz-xx-yy),-1.0925484305920792*x*z,.5462742152960396*(xx-yy),
       -.5900435899266435*y*(3*xx-yy),2.890611442640554*x*y*z,-.4570457994644658*y*(4*zz-xx-yy),.3731763325901154*z*(2*zz-3*xx-3*yy),
       -.4570457994644658*x*(4*zz-xx-yy),1.445305721320277*z*(xx-yy),-.5900435899266435*x*(xx-3*yy)],axis=1)
    return .5+np.einsum('nk,nkc->nc',b,np.concatenate([p['features_dc'][:,None],p['features_rest']],axis=1))

def main():
    mp=ROOT/'runs/fidelity_v2/original_single_blocks/block_015/metrics.json'; m=json.loads(mp.read_text())
    p, info=read_candidate(Path(m['source_ply']['path'])); assert info['sha256']==m['source_ply']['sha256']
    rows=[]; out=ROOT/'reports/viewer_parity_v2_outliers';out.mkdir(exist_ok=True)
    for idx in (5,9,8):
        row=next(r for r in m['frames'] if r['dataset_index']==idx)
        npz=mp.parent/f'{idx:03d}_float.npz'; png=ROOT/f'reports/viewer_parity_v2_original015/{idx:03d}_browser.png'
        n=np.load(npz,allow_pickle=False);r=n['rgb'].astype(np.float64);a=n['alpha'].squeeze().astype(np.float64);depth=n['depth'].squeeze()
        b=cv2.imread(str(png))[:,:,::-1].astype(np.float64)/255
        valid=cv2.imread(row['acceptance_mask_path'],0)>0
        err=b-r; energy=np.mean(err**2,axis=2);total=energy[valid].sum()
        def stats(mask):
            t=mask&valid;count=int(t.sum());ys,xs=np.where(t)
            return {'pixels':count,'valid_fraction':count/int(valid.sum()),'squared_error_fraction':float(energy[t].sum()/total),
                'mse':float(energy[t].mean()) if count else None,'rgb_bias':err[t].mean(0).tolist() if count else None,
                'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if count else None}
        bg=np.array([.1490,.1647,.2157],np.float32).astype(np.float64)
        linear=np.where(bg<=.04045,bg/12.92,((bg+.055)/1.055)**2.4)
        encoded=np.where(bg<=.0031308,bg*12.92,1.055*bg**(1/2.4)-.055)
        opaque=valid&(a>=.99)
        hypotheses={}
        for name,candidate in [('background_srgb_decoded',linear),('background_srgb_encoded',encoded)]:
            predicted=(1-a[:,:,None])*(candidate-bg)
            # Fixed source-defined alternatives, never fitted to observed error.
            residual=err-predicted
            hypotheses[name]={'background_rgb':candidate.tolist(),'predicted_mse':float((predicted[valid]**2).mean()),
                'observed_minus_prediction_mse':float((residual[valid]**2).mean()),
                'opaque_predicted_max_abs':float(np.max(abs(predicted[opaque])))}
        c=row['camera'];pose=np.asarray(c['camera_to_worlds_ns']);cvrot=pose[:,:3]*[1,-1,-1]
        view=(p['means']-pose[:,3])@cvrot;rgb=sh_rgb(p,pose[:,3]);z=view[:,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            px=c['fl_x']*view[:,0]/z+c['cx'];py=c['fl_y']*view[:,1]/z+c['cy']
        ins=(z>.2)&(px>=0)&(px<c['w'])&(py>=0)&(py<c['h']);hdr=ins&(rgb.max(1)>1)
        grid=[]
        yy,xx=np.indices(a.shape)
        for gy in range(3):
            grid.append([stats((yy>=gy*180)&(yy<(gy+1)*180)&(xx>=gx*160)&(xx<(gx+1)*160)) for gx in range(4)])
        record={'dataset_index':idx,'camera_id':row['camera_id'],'native_float_sha256':sha(npz),'browser_png_sha256':sha(png),
          'all_valid':stats(valid),'native_rgb_min_max':[float(r.min()),float(r.max())],
          'alpha_bins':{label:stats((a>=lo)&(a<hi)) for label,lo,hi in [('<.5',0,.5),('.5-.9',.5,.9),('.9-.99',.9,.99),('>=.99',.99,1.01)]},
          'rgb_gt1':stats(r.max(2)>1),'rgb_eq1_any_channel':stats(r.max(2)>=1),
          'max_abs_error_gt_.05':stats(np.max(abs(err),axis=2)>.05),
          'depth_lt_.4':stats(depth<.4),'depth_gt100':stats(depth>100),
          'spatial_grid_3rows_4cols':grid,'fixed_background_hypotheses':hypotheses,
          'opaque_error_exceeds_any_background_only_change_bound':stats((a>=.99)&(np.max(abs(err),2)>(1-a)+1/255)),
          'source_gaussian_audit':{'camera_z_min_max':[float(z.min()),float(z.max())],
            'center_near_culled':int((z<=.2).sum()),'center_beyond100':int((z>100).sum()),'projected_centers_in_image':int(ins.sum()),
            'in_image_sh_gt1_count':int(hdr.sum()),'in_image_sh_max':float(rgb[ins].max()),
            'hdr_center_bbox_xyxy':[float(px[hdr].min()),float(py[hdr].min()),float(px[hdr].max()),float(py[hdr].max())]}}
        rows.append(record)
        # Analytic diagnostic panels only, no alteration to source evidence PNGs.
        heat=cv2.applyColorMap(np.uint8(np.minimum(np.max(abs(err),2)/.15,1)*255),cv2.COLORMAP_INFERNO)
        heat[~valid]=0
        cv2.imwrite(str(out/f'{idx:03d}_error.png'),heat)
        alpha=np.uint8(np.where(a<.5,32,np.where(a<.9,96,np.where(a<.99,160,255))))
        cv2.imwrite(str(out/f'{idx:03d}_alpha_bins.png'),alpha)
    report={'status':'cpu_diagnosed_no_color_fit','gpu_used':False,'source_ply_sha256':info['sha256'],'native_metrics_sha256':sha(mp),'rows':rows,
       'facts':['NPZ rgb is post model torch.clamp(0,1); pre-clamp composite RGB was not exported and cannot be counted from NPZ.',
       'Native CUDA computeColorFromSH applies max(rgb,0) only; Spark vertex also applies max(rgb,0), ExtSplats uses half RGB and permits >1.',
       'Current Spark composites back-to-front directly into unorm8 canvas. GL normalized color buffers clamp fragment colors and each blend result.',
       'Native composites float32 then adds eval background then clamps once. This is a concrete pipeline mismatch for HDR SH radiance, independent of any fit.',
       'Native near culls center z<=.2 and ray depth<.2; Spark center and rasterized quad near clipping are analogous, but low-pass and geometric clipping differ.',
       'Depth is weighted expectation; depth>100 incidence does not bound individual ray far clipping. No exact per-pixel attribution to ray clipping is claimed.'],
       'source_references':{'native_sh':'runtime/inria2dgs_src/cuda_rasterizer/forward.cu:20-72',
       'native_pixel':'runtime/inria2dgs_src/cuda_rasterizer/forward.cu:350-444',
       'native_final_clamp':'.venv-train/lib/python3.10/site-packages/nerfstudio/models/splatfacto.py:626-628',
       'spark':'viewer/node_modules/@sparkjsdev/spark/dist/spark.module.js.map (splatVertex, splatFragment, splatDefines)',
       'GL_spec':'https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf section 4.1.7/4.1.8'}}
    (ROOT/'reports/viewer_parity_v2_outliers_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in rows:print(json.dumps({k:r[k] for k in ('dataset_index','source_gaussian_audit','fixed_background_hypotheses','opaque_error_exceeds_any_background_only_change_bound')}))
if __name__=='__main__':main()

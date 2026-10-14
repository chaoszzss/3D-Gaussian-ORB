"""CPU audit of every contributor on the frozen nine rays; no model changes.

Uses original native forward behavior: AA selects rho2d but still uses ray-plane
depth. This counts forward exposure only, not backward gradients or training loss.
"""
import sys
from pathlib import Path
import json
import time
import resource
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from assemble_core_gaussians import InputGuard,inspect_ply,resource_preflight,write_json
from assemble_core_gaussians_lidar_ownership import surfel_projection,ray_alpha,sh_rgb

out=ROOT/'reports/core_assembly_redwall_aa_depth_expected_v2_20261009.json'
if out.exists():
    raise FileExistsError(out)
start=time.monotonic()
preflight=resource_preflight(out,disk_bytes=16*1024**2,memory_bytes=1536*1024**2)
guard=InputGuard(ROOT)
previous=guard.json(ROOT/'reports/core_assembly_redwall_unknown_split_v2_20261009/attribution.json')
assembly=guard.json(ROOT/previous['assembly'])
meta_path=ROOT/'maps/garage_surface_all28_noownership_sides_v2_full_20261009/metadata.json'
guard.snapshot(meta_path)
if meta_path.stat().st_size > 16*1024**2:
    raise ValueError('Frozen sidecar metadata exceeds this diagnostic read budget')
meta=json.loads(meta_path.read_text())
assert guard.files[ROOT/'maps/garage_surface_all28_noownership_sides_v2_full_20261009/metadata.json']['sha256']==previous['direction_support']['sha256']
for path in ('runtime/inria2dgs_src/cuda_rasterizer/forward.cu',
             'runtime/inria2dgs_src/cuda_rasterizer/backward.cu','reports/aa_depth_chain_rule_cpu.json'):
    guard.snapshot(ROOT/path)
full=ROOT/assembly['full']['path'];guard.snapshot(full)
assert guard.files[full]['sha256']==assembly['full']['sha256']==meta['ply']['sha256']
tp=ROOT/assembly['full']['provenance']['path'];guard.snapshot(tp)
assert guard.files[tp]['sha256']==assembly['full']['provenance']['sha256']
trace=np.load(tp,mmap_mode='r',allow_pickle=False)
arrays=[]
for d in (meta['side_flags'],meta['training_evidence']['positive_train_groups'],meta['training_evidence']['negative_train_groups']):
    p=ROOT/d['path'];guard.snapshot(p);assert guard.files[p]['sha256']==d['sha256']
    x=np.load(p,mmap_mode='r',allow_pickle=False);assert len(x)==meta['ply']['gaussians'];arrays.append(x)
flags,positive,negative=arrays
assert np.all((flags==1)==((positive>=3)&(negative==0)))
assert np.all((flags==2)==((negative>=3)&(positive==0)))
assert np.all((flags==3)==((positive>0)&(negative>0)))
assert np.all((flags==0)==(((positive==0)&(negative<3))|((negative==0)&(positive<3))))
camera=previous['camera'];camera_position=np.asarray(camera['camera_to_worlds_ns'])[:,3]
pixels=[r['pixel_xy'] for r in previous['rays']]
selected=[[] for _ in pixels]
with full.open('rb') as handle:
    spec=inspect_ply(handle);names=spec['names']
    for begin in range(0,spec['count'],65536):
        n=min(65536,spec['count']-begin)
        rows=np.frombuffer(handle.read(n*spec['stride']),dtype='<f4').reshape(n,-1).astype(float)
        xyz=rows[:,[names.index(k) for k in ('x','y','z')]]
        projection=surfel_projection(xyz,np.exp(rows[:,[names.index(f'scale_{i}') for i in range(3)]]),
            rows[:,[names.index(f'rot_{i}') for i in range(4)]],camera)
        direction=camera_position-xyz
        cosine=np.sum(projection[6]*direction,axis=1)/np.maximum(np.linalg.norm(direction,axis=1),1e-12)
        f=flags[begin:begin+n]
        rejected=((f==1)&(cosine < -meta['render_grazing_cosine']))|((f==2)&(cosine > meta['render_grazing_cosine']))
        opacity=expit(rows[:,names.index('opacity')])
        for i,pixel in enumerate(pixels):
            ids,alpha,depth=ray_alpha(projection,opacity,pixel)
            if not len(ids):
                continue
            tu,tv,tw,image=projection[:4]
            cross=np.cross(pixel[0]*tw[ids]-tu[ids],pixel[1]*tw[ids]-tv[ids])
            uv=cross[:,:2]/cross[:,2:3]
            rho3d=np.sum(uv*uv,axis=1)
            rho2d=2*np.sum((image[ids]-pixel)**2,axis=1)
            selected[i].append((ids+begin,alpha,depth,tw[ids,2],rho3d,rho2d,rejected[ids],sh_rgb(rows[ids],names,camera_position)))

def branch_summary(delta,weight,mask):
    d=delta[mask];w=weight[mask];total=float(w.sum())
    return {'contributors':int(mask.sum()),'weight':total,
            'depth_minus_center_min_m':float(d.min()) if len(d) else None,
            'depth_minus_center_max_m':float(d.max()) if len(d) else None,
            'max_absolute_delta_m':float(np.max(np.abs(d))) if len(d) else None,
            'weighted_mean_delta_m':float(w@d/total) if total else None,
            'weighted_mean_absolute_delta_m':float(w@np.abs(d)/total) if total else None,
            'weight_absolute_delta_gt_1m':float(w[np.abs(d)>1].sum()),
            'weight_depth_at_least_1m_nearer_than_center':float(w[d < -1].sum())}

results={}
for variant,base in [('noownership','runs/fidelity_v2/original_noownership015/block_015'),
                     ('directional','runs/fidelity_v2/directional015/block_015')]:
    m=guard.json(ROOT/base/'metrics.json')
    assert m['source_ply_sha256']==assembly['full']['sha256']
    indices=[i for i,f in enumerate(m['frames']) if f['file_path']==previous['frame']['file_path']]
    assert len(indices)==1
    i=indices[0]
    assert m['frames'][i]['camera']==camera
    fp=ROOT/base/f'{i:03d}_float.npz';guard.snapshot(fp)
    with np.load(fp,allow_pickle=False) as f:
        native_rgb=np.array(f['rgb']);native_depth=np.array(f['depth']);native_alpha=np.array(f['alpha'])
    rays=[];all_delta=[];all_weight=[];all_aa=[]
    for pixel,chunks in zip(pixels,selected):
        ids,alpha,depth,center,rho3d,rho2d,rejected,colors=[np.concatenate([c[j] for c in chunks]) for j in range(8)]
        keep=~rejected if variant=='directional' else np.ones(len(ids),bool)
        ids,alpha,depth,center,rho3d,rho2d,colors=[v[keep] for v in (ids,alpha,depth,center,rho3d,rho2d,colors)]
        order=np.lexsort((ids,center))
        ids,alpha,depth,center,rho3d,rho2d,colors=[v[order] for v in (ids,alpha,depth,center,rho3d,rho2d,colors)]
        after=np.cumprod(1-alpha);stop=np.flatnonzero(after<1e-4);limit=int(stop[0]) if len(stop) else len(ids)
        weight=alpha[:limit]*np.r_[1.,after[:limit-1]] if limit else np.empty(0)
        delta=depth[:limit]-center[:limit];aa=rho2d[:limit]<rho3d[:limit]
        rgb=np.clip(np.sum(weight[:,None]*colors[:limit],axis=0),0,1)
        trans_before=np.r_[1.,after[:limit-1]] if limit else np.empty(0)
        median_indices=np.flatnonzero(trans_before>.5)
        median_index=int(median_indices[-1]) if len(median_indices) else None
        md=float(depth[median_index]) if median_index is not None else 0.
        expected_depth=float(np.dot(weight,depth[:limit])/max(float(weight.sum()),1e-8))
        records=[]
        for j in range(limit):
            origin=trace[ids[j]];sid=int(origin['source_index'])
            records.append({'full_record_index':int(ids[j]),'source_id':assembly['sources'][sid]['id'],
                'source_record_index':int(origin['source_record_index']),'AA':bool(aa[j]),
                'rho3d':float(rho3d[j]),'rho2d':float(rho2d[j]),'ray_plane_depth_m':float(depth[j]),
                'center_camera_z_m':float(center[j]),'depth_minus_center_m':float(delta[j]),
                'alpha':float(alpha[j]),'weight':float(weight[j]),'flag':int(flags[ids[j]]),
                'positive_train_groups':int(positive[ids[j]]),'negative_train_groups':int(negative[ids[j]])})
        x,y=pixel
        active_aa=rho2d<rho3d
        rays.append({'pixel_xy':pixel,'precomposite_candidates':len(ids),'precomposite_AA_candidates':int(active_aa.sum()),
            'contributors':limit,'composited_alpha':float(weight.sum()),
            'AA':branch_summary(delta,weight,aa),'non_AA':branch_summary(delta,weight,~aa),
            'all':branch_summary(delta,weight,np.ones(limit,bool)),
            'cpu_rgb':rgb.tolist(),'native_rgb':native_rgb[y,x].tolist(),
            'rgb_max_abs_error':float(np.max(np.abs(rgb-native_rgb[y,x]))),
            'cpu_native_rule_median_depth':md,'native_exported_depth':float(native_depth[y,x,0]),
            'cpu_native_rule_expected_depth':expected_depth,
            'depth_abs_error':float(abs(expected_depth-native_depth[y,x,0])),
            'alpha_abs_error':float(abs(weight.sum()-native_alpha[y,x,0])),
            'median_contributor':records[median_index] if median_index is not None else None,
            'composited_records':records})
        all_delta.append(delta);all_weight.append(weight);all_aa.append(aa)
    d,w,aa=map(np.concatenate,(all_delta,all_weight,all_aa))
    results[variant]={'AA':branch_summary(d,w,aa),'non_AA':branch_summary(d,w,~aa),
        'AA_fraction_of_composited_weight':float(w[aa].sum()/w.sum()),
        'equal_ray_mean_AA_weight':float(w[aa].sum()/len(pixels)),
        'max_rgb_abs_error':max(ray['rgb_max_abs_error'] for ray in rays),
        'max_native_depth_abs_error':max(ray['depth_abs_error'] for ray in rays),
        'rays':rays}
guard.check({full})
write_json(out,{'schema_version':1,'status':'nine_rays_forward_AA_exposure_cpu_diagnostic',
    'cpu_only':True,'preflight':preflight,'model_changed':False,'kernel_changed':False,
    'source_ply':assembly['full'],'frame':previous['frame'],'variants':results,'inputs':guard.report(),
    'convention':{'AA':'rho2d < rho3d (strict); original ray-plane depth preserved',
       'delta':'ray_plane_depth - center_camera_z, meters; negative means nearer than mean',
       'weight':'alpha * accumulated transmittance before this Gaussian; exclude native early-termination record',
       'native_depth_comparison':'sum(weight*ray_plane_depth)/sum(weight), matching RGB+ED native float export; median retained as a separate diagnostic'},
    'limitations':['Nine targeted rays are not a full-frame or dataset-level exposure estimate.',
        'Forward contribution counting does not measure the backward chain-rule mismatch or training history.',
        'AA toy establishes a possible mechanism only; scene attribution requires observed AA exposure.',
        'No filter, train change, or GPU workload was started.'],
    'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
for key,v in results.items():
    print(json.dumps({'variant':key,**{k:v[k] for k in ('AA','non_AA','AA_fraction_of_composited_weight','max_rgb_abs_error','max_native_depth_abs_error')}}))
    for ray in v['rays']:
        print(json.dumps({'pixel':ray['pixel_xy'],'AA':ray['AA'],'all':ray['all'],
            'precomposite_AA':ray['precomposite_AA_candidates'],'depth_error':ray['depth_abs_error']}))

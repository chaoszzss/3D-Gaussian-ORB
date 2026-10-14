"""Audit directional residual means, surfel hit points, LiDAR and world heights."""
import sys
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from assemble_core_gaussians import InputGuard,inspect_ply,write_json,resource_preflight
from assemble_core_gaussians_lidar_ownership import surfel_projection
out=ROOT/'reports/core_assembly_redwall_geometry_v2_20261009.json'
if out.exists():
    raise FileExistsError(out)
preflight=resource_preflight(out,disk_bytes=8*1024**2,memory_bytes=1024**3)
guard=InputGuard(ROOT)
r=guard.json(ROOT/'reports/core_assembly_redwall_directional_attribution_v2_20261009/attribution.json')
a=guard.json(ROOT/r['assembly'])
ledger=guard.json(ROOT/'maps/garage_manifest.json')
floors={b['id']:b.get('floor_id') for b in ledger['blocks']}
camera=r['camera'];pose=np.asarray(camera['camera_to_worlds_ns']);center=pose[:,3]
rotation=pose[:,:3]@np.diag([1.,-1.,-1.])
refs={};sources={};specs={}
for source in a['sources']:
    sid=source['id']
    if not any(c['source_id']==sid for ray in r['rays'] for c in ray['top_contributors']) and sid!='block_015':
        continue
    p=ROOT/source['path'];guard.snapshot(p)
    assert guard.files[p]['sha256']==source['source_sha256']
    with p.open('rb') as h:
        spec=inspect_ply(h)
    specs[sid]=spec
    sources[sid]=np.memmap(p,mode='r',dtype='<f4',offset=len(spec['header']),shape=(spec['count'],len(spec['names'])))
    rp=ROOT/source['measured_reference'];guard.snapshot(rp)
    with np.load(rp,allow_pickle=False) as z:
        refs[sid]=cKDTree(np.asarray(z['xyz'],dtype=float))

native={}
for key,base in [('directional','runs/fidelity_v2/directional015/block_015'),
                 ('single015','runs/fidelity_v2/original_single_blocks/block_015'),
                 ('noownership','runs/fidelity_v2/original_noownership015/block_015')]:
    metrics=guard.json(ROOT/base/'metrics.json')
    indices=[i for i,f in enumerate(metrics['frames']) if f['file_path']==r['frame']['file_path']]
    assert len(indices)==1
    i=indices[0];p=ROOT/base/f'{i:03d}_float.npz';guard.snapshot(p)
    with np.load(p,allow_pickle=False) as z:
        native[key]={k:np.asarray(z[k]) for k in ('rgb','depth','alpha')}
    if key=='directional':
        frame=metrics['frames'][i]
        p=Path(frame['source_lidar_depth_path']);guard.snapshot(p);reference_depth=np.load(p)
        p=Path(frame['source_lidar_support_path']);guard.snapshot(p)
        with Image.open(p) as im:
            reference_mask=np.asarray(im)>0
        p=ROOT/base/frame['artifacts']['gt_png'];guard.snapshot(p)
        with Image.open(p) as im:
            gt=np.asarray(im).astype(float)/255.
rays=[]
for ray in r['rays']:
    x,y=ray['pixel_xy']
    dcv=np.array([(x-camera['cx']+.5)/camera['fl_x'],(y-camera['cy']+.5)/camera['fl_y'],1.])
    contributors=[]
    for c in ray['top_contributors']:
        sid=c['source_id'];names=specs[sid]['names']
        row=sources[sid][c['source_record_index']].astype(float)
        xyz=row[[names.index(k) for k in ('x','y','z')]]
        scales=np.exp(row[[names.index(f'scale_{i}') for i in range(3)]])
        q=row[[names.index(f'rot_{i}') for i in range(4)]]
        projection=surfel_projection(xyz[None],scales[None],q[None],camera)
        tu,tv,tw,image,radius,valid,normal=projection
        cross=np.cross(x*tw-tu,y*tw-tv)[0]
        s=cross[:2]/cross[2]
        rho3=float(s@s);rho2=float(2*np.sum((image[0]-[x,y])**2))
        hit=center+rotation@dcv*c['ray_depth_m']
        mean_d,mean_i=refs[sid].query(xyz)
        hit_d,hit_i=refs[sid].query(hit)
        target_d,target_i=refs['block_015'].query(xyz)
        contributors.append({**c,'source_floor_label':floors[sid],
            'mean_own_reference_nn_m':float(mean_d),'mean_own_reference_z_m':float(refs[sid].data[mean_i,2]),
            'mean_target015_reference_nn_m':float(target_d),'ray_hit_world':hit.tolist(),
            'ray_hit_own_reference_nn_m':float(hit_d),'ray_hit_own_reference_z_m':float(refs[sid].data[hit_i,2]),
            'ray_hit_minus_mean_m':float(np.linalg.norm(hit-xyz)),'ray_hit_minus_mean_z_m':float(hit[2]-xyz[2]),
            'mean_minus_query_camera_z_m':float(xyz[2]-center[2]),
            'tangent_sigma_m':scales[:2].tolist(),'native_bbox_radius_px':float(radius[0]),
            'rho3d':rho3,'rho2d':rho2,'lowpass_dominates':rho2<rho3})
    values={}
    for key,arrays in native.items():
        values[key]={'rgb':arrays['rgb'][y,x].tolist(),'depth':float(arrays['depth'][y,x,0]),
                     'alpha':float(arrays['alpha'][y,x,0])}
    rays.append({'pixel_xy':[x,y],'category_weights':ray['direction_category_weights'],
        'source_weights':ray['source_weights'],'native':values,'gt_rgb':gt[y,x,:3].tolist(),
        'prepared_reference_depth_m':float(reference_depth[y,x]),'prepared_reference_support':bool(reference_mask[y,x]),
        'audited_top_weight_sum':sum(c['weight'] for c in contributors),'contributors':contributors})
guard.check()
category={k:float(np.mean([ray['category_weights'].get(k,0) for ray in rays])) for k in
          ('unknown','positive_supported','negative_supported','both_observed')}
allc=[c for ray in rays for c in ray['contributors']]
summary={'equal_ray_mean_category_weight':category,'audited_top_contributors':len(allc),
         'audited_weight_sum':sum(c['weight'] for c in allc),
         'weight_of_hit_minus_mean_gt_1m':sum(c['weight'] for c in allc if c['ray_hit_minus_mean_m']>1),
         'weight_of_abs_hit_minus_mean_z_gt_1m':sum(c['weight'] for c in allc if abs(c['ray_hit_minus_mean_z_m'])>1),
         'weight_of_lowpass_dominated':sum(c['weight'] for c in allc if c['lowpass_dominates']),
         'weight_of_tangent_sigma_gt_0_12m':sum(c['weight'] for c in allc if max(c['tangent_sigma_m'])>.12),
         'mean_nn_max_m':max(c['mean_own_reference_nn_m'] for c in allc)}
write_json(out,{'status':'nine_targeted_rays_geometry_diagnostic','cpu_only':True,'preflight':preflight,
    'query_camera_world':center.tolist(),'source_floor_labels_independently_verified':False,
    'summary':summary,'rays':rays,'inputs':guard.report(),
    'limitations':['No independent LiDAR GT; nearest-center support is inherited from the .2m assembly gate.',
                  'Full sources are selected by training-path coverage, not exclusive physical floors.',
                  'Nine chosen rays and top20 contributors are not representative whole-frame statistics.',
                  'Native single015 depth is a model prediction, not metric ground truth.']})
print(json.dumps(summary))
for ray in rays:
    print(json.dumps({'pixel':ray['pixel_xy'],'native_depth':{k:v['depth'] for k,v in ray['native'].items()},
        'lidar':(ray['prepared_reference_depth_m'],ray['prepared_reference_support']),
        'top':[{'source':c['source_id'],'row':c['source_record_index'],'weight':c['weight'],
        'ownNN':c['mean_own_reference_nn_m'],'015NN':c['mean_target015_reference_nn_m'],
        'hitNN':c['ray_hit_own_reference_nn_m'],'meanZ':c['xyz'][2],'hit_delta_m':c['ray_hit_minus_mean_m'],
        'hit_delta_z':c['ray_hit_minus_mean_z_m'],'sigma':c['tangent_sigma_m']} for c in ray['contributors'][:3]]}))

"""CPU final-checkpoint self-source contribution at nearest supported TRAIN views.

Diagnostic target records are chosen from held-out residual attribution. No result
is fitted into a confidence mask, no held-out RGB is used for source confidence.
This is a targeted falsification probe, not an unbiased dataset-wide statistic.
"""
import sys
from pathlib import Path
import json
import time
import resource
import numpy as np
from scipy.special import expit
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from assemble_core_gaussians import InputGuard,inspect_ply,resource_preflight,write_json
from assemble_core_gaussians_lidar_ownership import training_evidence,surfel_projection,ray_alpha,sh_rgb

out=ROOT/'reports/core_assembly_train_visibility_probe_v2_20261009.json'
if out.exists():
    raise FileExistsError(out)
start=time.monotonic()
preflight=resource_preflight(out,disk_bytes=32*1024**2,memory_bytes=1024**3)
guard=InputGuard(ROOT)
report=guard.json(ROOT/'reports/core_assembly_redwall_directional_attribution_v2_20261009/attribution.json')
assembly=guard.json(ROOT/report['assembly'])
targets={'block_012':[143022,37202,240786], 'block_013':[9339,10382],
         'block_015':[186159], 'block_016':[158634,168544]}
query=np.asarray(report['camera']['camera_to_worlds_ns'])[:,3]
results=[]
for source in assembly['sources']:
    sid=source['id']
    if sid not in targets:
        continue
    path=ROOT/source['path'];guard.snapshot(path)
    assert guard.files[path]['sha256']==source['source_sha256']
    with path.open('rb') as h:
        spec=inspect_ply(h)
    names=spec['names']; n=spec['count']
    mm=np.memmap(path,dtype='<f4',mode='r',offset=len(spec['header']),shape=(n,len(names)))
    records=np.asarray(mm[targets[sid]],dtype=np.float64)
    evidence=training_evidence(ROOT,source,records,names,query,guard)
    base=ROOT/f'data/garage_surface_v1/{sid}'
    meta=guard.json(base/'transforms.json')
    frames={f['file_path']:f for f in meta['frames']}
    xyzcols=[names.index(k) for k in ('x','y','z')]
    for rid,target,ev in zip(targets[sid],records,evidence):
        frame=frames[ev['nearest_supported_training_frame']['file_path']]
        assert frame['file_path'] in meta['train_filenames']
        assert frame['file_path'] not in meta['val_filenames']+meta['test_filenames']
        pose=np.asarray(frame['transform_matrix'],dtype=float)
        camera={k:frame[k] for k in ('fl_x','fl_y','cx','cy','w','h')}
        camera['camera_to_worlds_ns']=pose[:3].tolist()
        p=(target[xyzcols]-pose[:3,3])@(pose[:3,:3]@np.diag([1.,-1.,-1.]))
        pixel=np.floor([camera['fl_x']*p[0]/p[2]+camera['cx'],camera['fl_y']*p[1]/p[2]+camera['cy']]).astype(int)
        chunks=[]
        for begin in range(0,n,32768):
            rows=np.asarray(mm[begin:begin+32768],dtype=float)
            xyz=rows[:,xyzcols]
            projection=surfel_projection(xyz,np.exp(rows[:,[names.index(f'scale_{i}') for i in range(3)]]),
                rows[:,[names.index(f'rot_{i}') for i in range(4)]],camera)
            ids,alpha,depth=ray_alpha(projection,expit(rows[:,names.index('opacity')]),pixel)
            if len(ids):
                chunks.append((ids+begin,alpha,depth,projection[2][ids,2],rows[ids]))
        ids,alpha,depth,sortz,rows=[np.concatenate([c[j] for c in chunks]) for j in range(5)]
        order=np.lexsort((ids,sortz))
        ids,alpha,depth,sortz,rows=[v[order] for v in (ids,alpha,depth,sortz,rows)]
        after=np.cumprod(1.-alpha)
        stop=np.flatnonzero(after<1e-4)
        limit=int(stop[0]) if len(stop) else len(ids)
        trans=np.r_[1.,after[:-1]]
        weights=alpha*trans
        weights[limit:]=0
        colors=sh_rgb(rows,names,pose[:3,3])
        rgb=np.clip(np.sum(colors*weights[:,None],axis=0),0,1)
        imagepath=base/frame['file_path'];guard.snapshot(imagepath)
        with Image.open(imagepath) as image:
            gt=np.asarray(image.convert('RGB'))[pixel[1],pixel[0]].astype(float)/255.
        match=np.flatnonzero(ids==rid)
        target_weight=float(weights[match[0]]) if len(match) else 0.
        details=None
        if len(match):
            j=int(match[0]);details={'alpha':float(alpha[j]),'transmittance_before':float(trans[j]),
                'center_depth_m':float(sortz[j]),'ray_depth_m':float(depth[j]),'sorted_rank':j,
                'terminated_before_target':bool(j>=limit),'own_SH_rgb':colors[j].tolist()}
        dev=[]
        for ray in report['rays']:
            for item in ray['top_contributors']:
                if item['source_id']==sid and item['source_record_index']==rid:
                    dev.append({'pixel_xy':ray['pixel_xy'],'weight':item['weight'],
                                'ray_depth_m':item['ray_depth_m'],'direction_evidence':item['direction_evidence']})
        top=[{'source_record_index':int(ids[j]),'weight':float(weights[j]),'ray_depth_m':float(depth[j]),
              'center_depth_m':float(sortz[j])} for j in np.argsort(-weights)[:8]]
        record={'source':sid,'source_record_index':rid,'nearest_supported_TRAIN':ev,
            'train_frame':{k:frame[k] for k in ('file_path','group_id','camera_id')},
            'train_pixel_xy':pixel.tolist(),'query_to_train_camera_distance_m':float(np.linalg.norm(query-pose[:3,3])),
            'own_source_train_pixel_target_weight':target_weight,'target_native_equation_details':details,
            'own_source_train_pixel_rgb':rgb.tolist(),'train_image_pixel_rgb':gt.tolist(),
            'pixel_max_RGB_error':float(np.max(np.abs(rgb-gt))),
            'own_source_train_pixel_top_contributors':top,'heldout_residual_attribution':dev}
        results.append(record)
        print(json.dumps({'source':sid,'record':rid,'alpha_T':target_weight,'details':details,
            'angle_deg':ev['nearest_supported_training_angle_deg'],'train_pixel_RGB_error':record['pixel_max_RGB_error']}),flush=True)
    del mm
guard.check()
write_json(out,{'schema_version':1,'status':'targeted_cpu_final_model_visibility_diagnostic',
    'cpu_only':True,'preflight':preflight,'source_attributes_modified':False,'confidence_filter_generated':False,
    'split_for_self_visibility':'train','heldout_used_for_target_selection':True,
    'heldout_pixels_used_to_construct_train_evidence':False,'results':results,'inputs':guard.report(),
    'limitations':['Nearest supported TRAIN ray is one targeted sample, not all-view visibility count.',
        'Final model alpha*T does not reconstruct optimizer history or prove never visible during training.',
        'CPU equations matched saved native DEV rays within 1/255; these exact TRAIN rays have not had a GPU parity check.',
        'LiDAR foreground evidence uses legacy prepared depth, not independent geometry GT.',
        'No filter threshold or reconstruction quality claim follows automatically.'],
    'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

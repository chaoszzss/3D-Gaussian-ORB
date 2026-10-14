#!/usr/bin/env python3
"""Append observed measured floor seeds to a byte-preserved warm PLY, CPU only.

Fixed experiment: actual reference XYZ, valid normals within 30 degrees of +/−Z,
2 cm voxel / warm-center gap, 3 distinct TRAIN groups, v3 depth agreement 10 cm.
No fitted plane, interpolation, held-out RGB, silent seed cap, or quality claim.
"""
import sys
sys.dont_write_bytecode = True
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from assemble_core_gaussians import (
    ROOT, AssemblyError, InputGuard, inspect_ply, local_path, resource_preflight,
    sha256, write_json,
)

MAX_SEEDS = 500_000
MAX_REFERENCE = 2_000_000
MEMORY_BUDGET = 1792*1024**2
CHUNK_BYTES = 8*1024**2
POLICY = {'normal_abs_z_min': float(np.cos(np.deg2rad(30))),
          'below_observing_camera_z_m': [.3, 3.], 'minimum_camera_z_m': 0.,
          'depth_agreement_m': .10, 'minimum_unique_train_groups': 3,
          'seed_voxel_m': .02, 'warm_center_exclusion_m': .02,
          'tangent_sigma_m': .02, 'normal_thickness_m': .001, 'opacity': .9,
          'maximum_seeds': MAX_SEEDS}


def horizontal_reference_rows(xyz, normals, valid):
    xyz, normals, valid = np.asarray(xyz), np.asarray(normals), np.asarray(valid)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape or valid.shape != (len(xyz),):
        raise AssemblyError('Reference XYZ/normal/valid shapes differ')
    lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    good = valid.astype(bool) & np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths>1e-12)
    unit = normals.astype(np.float64)/np.maximum(lengths[:, None], 1e-12)
    good &= np.abs(unit[:, 2]) >= POLICY['normal_abs_z_min']
    return np.flatnonzero(good), unit


def floor_quaternions(normals):
    n = np.asarray(normals, dtype=np.float64).copy()
    if n.ndim != 2 or n.shape[1] != 3 or not np.isfinite(n).all():
        raise AssemblyError('Invalid floor normals')
    length = np.linalg.norm(n, axis=1)
    if np.any(length<=1e-12):
        raise AssemblyError('Zero floor normal')
    n /= length[:, None]
    n[n[:, 2]<0] *= -1  # Canonical up-facing sign; reference position unchanged.
    q = np.column_stack((1+n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype('<f4'), n.astype('<f4')


def train_groups(meta):
    membership = {}
    for split in ('train_filenames', 'val_filenames', 'test_filenames'):
        values = meta.get(split, [])
        if not isinstance(values, list) or len(set(values)) != len(values):
            raise AssemblyError('Invalid or duplicate split filenames')
        for name in values:
            if name in membership:
                raise AssemblyError('TRAIN/held-out filename leakage')
            membership[name] = split
    groups, group_split, seen = defaultdict(list), {}, set()
    for frame in meta['frames']:
        name = frame['file_path']
        if name in seen or name not in membership:
            raise AssemblyError('Duplicate or unsplit frame')
        seen.add(name)
        value = str(frame['group_id'])
        if not value.isdigit():
            raise AssemblyError('TRAIN group ID must be an integer timestamp')
        group = str(int(value))
        split = membership[name]
        if group in group_split and group_split[group] != split:
            raise AssemblyError('TRAIN/held-out group leakage')
        group_split[group] = split
        if split == 'train_filenames':
            groups[group].append(frame)
    if seen != set(membership) or len(groups)<3 or len(groups)>65535:
        raise AssemblyError('Require complete splits and 3..65535 TRAIN groups')
    return [(g, sorted(frames, key=lambda f:f['file_path'])) for g,frames in sorted(groups.items(), key=lambda item:int(item[0]))]


def eligible_observations(xyz, frame, depth, depth_mask, ego_mask):
    """Nearest integer pixel; all geometric/known-depth gates use this TRAIN view."""
    h, w = int(frame['h']), int(frame['w'])
    pose = np.asarray(frame['transform_matrix'], dtype=np.float64)
    if (min(h,w)<=0 or h*w>4_194_304 or pose.shape!=(4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0,0,0,1])
            or not np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-5, rtol=0)):
        raise AssemblyError('Invalid or oversized TRAIN camera')
    if any(np.asarray(a).shape != (h,w) for a in (depth,depth_mask,ego_mask)):
        raise AssemblyError('Depth and grayscale masks must match TRAIN H/W')
    if not all(np.isfinite(frame[k]) for k in ('fl_x','fl_y','cx','cy')) or min(frame['fl_x'],frame['fl_y'])<=0:
        raise AssemblyError('Invalid TRAIN intrinsics')
    below = pose[2,3]-xyz[:,2]
    ids = np.flatnonzero((below>=.3)&(below<=3.))
    p = (xyz[ids]-pose[:3,3]) @ (pose[:3,:3]@np.diag([1.,-1.,-1.]))
    with np.errstate(divide='ignore',invalid='ignore'):
        u=frame['fl_x']*p[:,0]/p[:,2]+frame['cx']-.5
        v=frame['fl_y']*p[:,1]/p[:,2]+frame['cy']-.5
    inside=(p[:,2]>0)&np.isfinite(u)&np.isfinite(v)&(u>=-.5)&(u<w-.5)&(v>=-.5)&(v<h-.5)
    ids,p,u,v=ids[inside],p[inside],u[inside],v[inside]
    ui,vi=np.floor(u+.5).astype(int),np.floor(v+.5).astype(int)
    target=depth[vi,ui]
    known=(depth_mask[vi,ui]>0)&(ego_mask[vi,ui]>0)&np.isfinite(target)&(target>0)
    keep=known&(np.abs(p[:,2]-target)<=.10)
    return ids[keep],ui[keep],vi[keep]


def stable_voxel_rows(xyz, reference_rows, voxel=.02):
    """Keep the smallest actual reference row per occupied voxel; never average XYZ."""
    rows=np.asarray(reference_rows)
    order=np.argsort(rows,kind='stable')
    cells=np.floor(np.asarray(xyz)[order]/voxel).astype(np.int64)
    _,first=np.unique(cells,axis=0,return_index=True)
    selected=order[first]
    return selected[np.argsort(rows[selected],kind='stable')]


def enforce_seed_limit(count):
    if count>MAX_SEEDS:
        raise AssemblyError(f'{count} seeds exceeds fixed {MAX_SEEDS} limit; refusing to truncate')


def seed_rows(names, xyz, normals, rgb):
    allowed={'x','y','z','nx','ny','nz','opacity',*[f'scale_{i}' for i in range(3)],
             *[f'rot_{i}' for i in range(4)],*[f'f_dc_{i}' for i in range(3)],*[f'f_rest_{i}' for i in range(45)]}
    if not set(names)<=allowed:
        raise AssemblyError('Unsupported seed PLY fields')
    if np.any(~np.isfinite(rgb)) or np.any(rgb<0) or np.any(rgb>1):
        raise AssemblyError('Seed RGB must be bounded measured sRGB in [0,1]')
    rows=np.zeros((len(xyz),len(names)),dtype='<f4')
    q,n=floor_quaternions(normals)
    for j,key in enumerate(('x','y','z')):rows[:,names.index(key)]=xyz[:,j]
    for j,key in enumerate(('nx','ny','nz')):
        if key in names:rows[:,names.index(key)]=n[:,j]
    for j in range(4):rows[:,names.index(f'rot_{j}')]=q[:,j]
    for j in range(3):
        rows[:,names.index(f'f_dc_{j}')]=(rgb[:,j]-.5)/.28209479177387814
        rows[:,names.index(f'scale_{j}')]=np.log(.02 if j<2 else .001)
    rows[:,names.index('opacity')]=np.log(.9/.1)
    return rows,q,n


def seed_measured_floor(warm_ply, data, output, *, threads=3, project=ROOT):
    with threadpool_limits(limits=1):
        return _seed_measured_floor(warm_ply,data,output,threads=threads,project=project)


def _seed_measured_floor(warm_ply, data, output, *, threads, project):
    start=time.monotonic();project=Path(project).resolve()
    if project!=ROOT:local_path(ROOT,project)
    if type(threads) is not int or not 1<=threads<=3:
        raise AssemblyError('CPU thread budget must be 1..3')
    warm,base,destination=(local_path(project,p) for p in (warm_ply,data,output))
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError('Output must be a NEW directory with an existing parent')
    with warm.open('rb') as f:spec=inspect_ply(f)
    if spec['count']>2_500_000 or warm.stat().st_size!=len(spec['header'])+spec['count']*spec['stride']:
        raise AssemblyError('Warm count/size outside bounded input contract')
    preflight=resource_preflight(destination,disk_bytes=warm.stat().st_size+MAX_SEEDS*spec['stride']+128*1024**2,
        memory_bytes=MEMORY_BUDGET)
    guard=InputGuard(project)
    warm_manifest=guard.json(warm.parent/'manifest.json')
    guard.snapshot(warm)
    if warm_manifest.get('gaussians')!=spec['count'] or warm_manifest.get('sha256')!=guard.files[warm]['sha256']:
        raise AssemblyError('Warm manifest SHA/count mismatch')
    meta=guard.json(base/'transforms.json');prep=guard.json(base/'preparation.json')
    if prep['generated_hashes']['transforms.json']!=guard.files[base/'transforms.json']['sha256']:
        raise AssemblyError('Camera/split metadata differs from frozen preparation')
    if (meta.get('orientation_override','none')!='none' or prep.get('world_frame')!='camera_init'
            or prep.get('units') not in ('m','metres')):
        raise AssemblyError('Require original shared camera_init coordinates in meters')
    teacher=prep['teacher_provenance'];policy=teacher['policy']
    if (policy.get('version')!='surfel_depth_v3' or not policy.get('ground_support')
            or policy.get('world_up')!=[0.,0.,1.] or policy.get('foreground_occlusion_tolerance_m',1)>.1
            or not teacher.get('regenerated_all_frames')):
        raise AssemblyError('Require pinned v3 known-foreground teacher with world Z up')
    groups=train_groups(meta)
    frame_count=sum(len(frames) for _,frames in groups)
    if frame_count>4096:raise AssemblyError('TRAIN evidence exceeds bounded frame budget')
    teacher_frames={f['file_path']:f for f in teacher['frames']}
    if len(teacher_frames)!=len(teacher['frames']):raise AssemblyError('Duplicate teacher frame proof')
    refpath=base/'lidar_reference.npz';guard.snapshot(refpath)
    if guard.files[refpath]['sha256']!=teacher['reference_sha256']:
        raise AssemblyError('Reference SHA differs from v3 teacher')
    with zipfile.ZipFile(refpath) as archive:
        if sum(v.file_size for v in archive.infolist())>256*1024**2:
            raise AssemblyError('Reference exceeds decompressed memory budget')
    with np.load(refpath,allow_pickle=False) as archive:
        reference={k:archive[k] for k in ('xyz','normals','normal_valid','source_block_index','source_row_index')}
    xyz0=reference['xyz']
    if len(xyz0)>MAX_REFERENCE or xyz0.dtype.kind!='f' or xyz0.dtype.itemsize!=4:
        raise AssemblyError('Require bounded float32 measured XYZ, without coordinate resampling')
    refids,unit=horizontal_reference_rows(xyz0,reference['normals'],reference['normal_valid'])
    initial_horizontal=len(refids)
    mm=np.memmap(warm,mode='r',dtype='<f4',offset=len(spec['header']),shape=(spec['count'],len(spec['names'])))
    centers=np.asarray(mm[:,[spec['names'].index(k) for k in ('x','y','z')]],dtype=np.float64)
    del mm
    if not np.isfinite(centers).all():raise AssemblyError('Nonfinite warm center')
    tree=cKDTree(centers)
    distance,_=tree.query(xyz0[refids].astype(np.float64),workers=threads)
    gap=distance>.02
    refids,distance=refids[gap],distance[gap]
    del tree,centers
    xyz=xyz0[refids].astype(np.float64);count=len(refids)
    estimated_peak=(spec['count']*64+len(xyz0)*64+count*(96+(frame_count+7)//8+(len(groups)+7)//8)
                    +MAX_SEEDS*spec['stride']*2+256*1024**2)
    if estimated_peak>MEMORY_BUDGET:
        raise AssemblyError('Evidence arrays exceed bounded peak memory estimate')
    support_groups=np.zeros(count,dtype='<u2');support_frames=np.zeros(count,dtype='<u2')
    group_rgb_sum=np.zeros((count,3),dtype=np.float64)
    frame_bits=np.zeros((count,(frame_count+7)//8),np.uint8)
    group_bits=np.zeros((count,(len(groups)+7)//8),np.uint8)
    frame_table=[];frame_index=0
    print(json.dumps({'horizontal_reference_rows':initial_horizontal,'outside_warm_2cm':count,
                      'train_groups':len(groups),'train_frames':frame_count}),flush=True)
    for gi,(group,frames) in enumerate(groups):
        this_rgb=np.zeros((count,3),dtype=np.float64);this_frames=np.zeros(count,dtype=np.uint16)
        for frame in frames:
            if not 0<int(frame['h'])*int(frame['w'])<=4_194_304:
                raise AssemblyError('TRAIN resolution exceeds bounded image memory')
            proof=teacher_frames[frame['file_path']]
            paths={k:local_path(project,base/frame[k]) for k in ('file_path','mask_path','lidar_depth_path','lidar_depth_mask_path')}
            expected={'file_path':prep['copied_asset_hashes'][frame['file_path']],
                      'mask_path':prep['copied_asset_hashes'][frame['mask_path']],
                      'lidar_depth_path':proof['depth_sha256'],'lidar_depth_mask_path':proof['support_sha256']}
            for key,path in paths.items():
                guard.snapshot(path)
                if guard.files[path]['sha256']!=expected[key]:raise AssemblyError('TRAIN asset SHA mismatch: '+str(path))
            if paths['lidar_depth_path'].stat().st_size>frame['h']*frame['w']*8+1024**2:
                raise AssemblyError('TRAIN depth file exceeds bounded array budget')
            depth=np.load(paths['lidar_depth_path'],allow_pickle=False)
            with Image.open(paths['lidar_depth_mask_path']) as im:depth_mask=np.array(im)
            with Image.open(paths['mask_path']) as im:ego_mask=np.array(im)
            ids,ui,vi=eligible_observations(xyz,frame,depth,depth_mask,ego_mask)
            with Image.open(paths['file_path']) as im:
                if im.mode!='RGB' or im.size!=(frame['w'],frame['h']):raise AssemblyError('TRAIN RGB format/shape mismatch')
                rgb=np.asarray(im)[vi,ui].astype(np.float64)
            this_rgb[ids]+=rgb;this_frames[ids]+=1;support_frames[ids]+=1
            frame_bits[ids,frame_index//8] |= np.uint8(1<<(frame_index%8))
            frame_table.append({'index':frame_index,'group_index':gi,'group_id':group,
                'file_path':frame['file_path'],'camera_id':frame['camera_id'],
                'eligible_candidate_observations':len(ids),'input_sha256':expected})
            frame_index+=1
        good=this_frames>0
        support_groups+=good
        group_rgb_sum[good]+=this_rgb[good]/this_frames[good,None]
        group_bits[good,gi//8] |= np.uint8(1<<(gi%8))
        if (gi+1)%20==0 or gi+1==len(groups):
            print(json.dumps({'train_groups_done':gi+1,'supported_at_least_3':int((support_groups>=3).sum())}),flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=2*1024**3:
            raise AssemblyError('CPU RSS reached 2 GiB bound; aborting before publication')
    eligible=np.flatnonzero(support_groups>=3)
    unique=stable_voxel_rows(xyz[eligible],refids[eligible])
    chosen=eligible[unique]
    enforce_seed_limit(len(chosen))
    chosen_ref=refids[chosen]
    seed_rgb=(group_rgb_sum[chosen]/support_groups[chosen,None]/255.).astype('<f4')
    appended,q,normals=seed_rows(spec['names'],xyz0[chosen_ref],unit[chosen_ref],seed_rgb)
    if len(chosen) and (not np.array_equal(appended[:,[spec['names'].index(k) for k in ('x','y','z')]],xyz0[chosen_ref])
                        or not np.all(distance[chosen]>.02)):
        raise AssemblyError('Seed coordinates or warm-center exclusion failed')
    destination.mkdir()
    try:
        target=destination/'gaussians.ply'
        header,changed=re.subn(rb'(?m)^(element[ \t]+vertex[ \t]+)\d+([ \t]*\r?$)',
                              lambda m:m[1]+str(spec['count']+len(chosen)).encode()+m[2],spec['header'])
        if changed!=1:raise AssemblyError('Cannot update only the PLY vertex count')
        original_payload=hashlib.sha256()
        with warm.open('rb') as src,target.open('xb') as dst:
            if inspect_ply(src)!=spec:raise AssemblyError('Warm schema changed')
            dst.write(header)
            for raw in iter(lambda:src.read(CHUNK_BYTES),b''):
                original_payload.update(raw);dst.write(raw)
            dst.write(appended.tobytes())
        prefix_hash=hashlib.sha256()
        with warm.open('rb') as src,target.open('rb') as dst:
            inspect_ply(src);out_spec=inspect_ply(dst)
            if out_spec['count']!=spec['count']+len(chosen) or out_spec['names']!=spec['names']:
                raise AssemblyError('Output schema/count mismatch')
            remaining=spec['count']*spec['stride']
            while remaining:
                size=min(CHUNK_BYTES,remaining);a,b=src.read(size),dst.read(size)
                if len(a)!=size or a!=b:raise AssemblyError('Original warm payload changed')
                prefix_hash.update(b);remaining-=size
            if dst.read()!=appended.tobytes():raise AssemblyError('On-disk seed payload differs')
        proofpath=destination/'seed_provenance.npz'
        with proofpath.open('xb') as stream:
            np.savez_compressed(stream,reference_row=chosen_ref.astype('<u4'),reference_xyz=xyz0[chosen_ref],
                reference_normal=reference['normals'][chosen_ref],normal_world=normals,quaternion_wxyz=q,
                source_block_index=reference['source_block_index'][chosen_ref],source_row_index=reference['source_row_index'][chosen_ref],
                output_row_index=np.arange(spec['count'],spec['count']+len(chosen),dtype='<u8'),
                nearest_original_warm_center_m=distance[chosen],support_train_groups=support_groups[chosen],
                support_train_frames=support_frames[chosen],color_support_groups=support_groups[chosen],
                color_support_frames=support_frames[chosen],rgb_srgb=seed_rgb,
                sum_of_group_mean_rgb_u8=group_rgb_sum[chosen],train_group_support_bits=group_bits[chosen],
                train_frame_support_bits=frame_bits[chosen])
        guard.check()
        record={'schema_version':1,'status':'observed_floor_seed_warm_candidate_pending_geometry_and_training',
            'cpu_only':True,'gpu_used':False,'quality_accepted':False,'default_changed':False,
            'world_frame':'camera_init','units':'m','preflight':preflight,'threads_maximum':threads,'blas_threads':1,
            'policy':POLICY,'existing_teacher_policy':policy,'warm_manifest':warm_manifest,
            'warm_input':{'path':warm.relative_to(project).as_posix(),'sha256':guard.files[warm]['sha256'],
                'gaussians':spec['count'],'payload_sha256':original_payload.hexdigest()},
            'output':{'path':target.relative_to(project).as_posix(),'sha256':sha256(target),
                'gaussians':spec['count']+len(chosen),'file_bytes':target.stat().st_size},
            'seed_provenance':{'path':proofpath.relative_to(project).as_posix(),'sha256':sha256(proofpath),
                'seed_count':len(chosen),'bit_order':'little: bit i of byte i//8; only recorded TRAIN frame/group tables'},
            'counts':{'reference_points':len(xyz0),'trusted_horizontal_rows':initial_horizontal,
                'excluded_by_warm_center_2cm':initial_horizontal-count,'outside_warm_center_2cm':count,
                'supported_at_least_3_groups_before_voxel':len(eligible),'voxel_duplicates_removed':len(eligible)-len(chosen),
                'seeds_appended':len(chosen),'original_records_deleted':0},
            'original_payload_all_bytes_compared':True,'original_payload_prefix_sha256':prefix_hash.hexdigest(),
            'original_order_preserved':True,'seed_xyz_actual_reference_rows':True,'plane_extension':False,
            'train_only_evidence_and_color':True,'heldout_used_for_seed_geometry_or_color':False,
            'heldout_metadata_read_only_for_split_validation':True,
            'train_group_ids':[g for g,_ in groups],'train_frames':frame_table,
            'color_policy':'Nearest eligible TRAIN pixel in stored 8-bit sRGB, no gamma conversion. '
                'Average eligible cameras within each group, then equal-weight average supported groups; '
                'bounded [0,1], DC=(RGB-.5)/C0 and SH rest zero. Group/frame bitsets identify every contributing observation.',
            'coverage_scope':'2cm distance from existing warm centers is only a coverage proxy, not correct-surface or rendered-hole proof. '
                'Reference and strict prepared teacher are conditioned measurements, not independent ground truth. '
                'Seeds have fixed peak opacity and are untrained; no quality/adoption claim.',
            'inputs':guard.report(),'seconds':time.monotonic()-start,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        write_json(destination/'floor_seed_audit.json',record)
        return record
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-ply',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=3)
    args=parser.parse_args(argv)
    try:
        r=seed_measured_floor(args.warm_ply,args.data,args.output,threads=args.threads)
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(json.dumps({'status':'incomplete','error':str(e)}),file=sys.stderr);return 2
    print(json.dumps({k:r[k] for k in ('status','output','counts','seconds','peak_rss_kib')}))
    return 0


if __name__=='__main__':sys.exit(main())

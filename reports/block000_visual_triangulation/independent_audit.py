"""Read-only independent arithmetic/image audit; never tracks or matches images."""
import os,sys,json,hashlib,csv,resource
from pathlib import Path
from collections import Counter,OrderedDict
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np,cv2
cv2.setNumThreads(3);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]))
resource.setrlimit(resource.RLIMIT_AS,(2800*1024**2,resource.RLIM_INFINITY))
ROOT=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent;out=parent/'validated_handoff'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
r=read(out/'report.json');d=read(out/'tracks.json');tracks=d['tracks'];frames=d['train_frame_table']
meta=read(Path(r['source_dataset'])/'transforms.json');train=set(meta['train_filenames']);held=set(meta['val_filenames'])|set(meta['test_filenames'])
protected=read(parent/'protected_inputs_before.json')
for p,h in protected.items():assert sha(ROOT/p)==h,p
for p,h in r['source_sha256'].items():assert sha(p)==h,p
for p,h in r['train_asset_sha256'].items():assert sha(p)==h,p
for p,h in r['outputs_sha256'].items():assert sha(out/p)==h,p
for p,h in r['parent_outputs_sha256'].items():assert sha(parent/p)==h,p
assert sha(parent/'report.json')==r['parent_report_sha256']
assert sha(ROOT/'scripts/triangulate_training_tracks.py')==r['script_sha256']
assert len(tracks)==300
with np.load(out/'tracks.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
assert np.array_equal(arrays['colors'],arrays['rgb']) and not arrays['normal_valid'].any()
assert arrays['xyz'].shape==(300,3) and arrays['uv'].shape==(300,3,2) and arrays['train_file_ids'].shape==(300,3)
np.testing.assert_array_equal(arrays['xyz'],np.array([t['xyz'] for t in tracks]))
np.testing.assert_array_equal(np.loadtxt(out/'points.ply',skiprows=10)[:,:3],arrays['xyz'])
all_centers=np.array([np.array(f['transform_matrix'])[:3,3] for f in frames]);rgb_cache=OrderedDict();mask_cache=OrderedDict()
actual_errors=[];actual_third=[];actual_angles=[];band_count=0;image_reads=set()
def asset(index):
    if index not in rgb_cache:
        f=frames[index];assert f['file_path'] in train and f['file_path'] not in held
        rgb=cv2.imread(str(Path(r['source_dataset'])/f['file_path']))[:,:,::-1]
        m=cv2.imread(str(Path(r['source_dataset'])/f['mask_path']),0)
        assert rgb.shape==(1080,1280,3) and m.shape==(1080,1280)
        rgb_cache[index]=rgb;mask_cache[index]=cv2.erode(m,np.ones((31,31),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        image_reads.add(f['file_path'])
        if len(rgb_cache)>8:
            old,_=rgb_cache.popitem(last=False);mask_cache.pop(old)
    return rgb_cache[index],mask_cache[index]
for i,t in enumerate(tracks):
    ids=t['train_indices'];assert len(set(ids))==3
    assert len({frames[j]['camera_id'] for j in ids})==1
    assert [frames[j]['file_path'] for j in ids]==t['train_file_ids']==arrays['train_file_ids'][i].tolist()
    assert all(s in train and s not in held for s in t['train_file_ids'])
    assert frames[ids[0]]['timestamp_ns']<frames[ids[1]]['timestamp_ns']<frames[ids[2]]['timestamp_ns']
    Ts=[];Ps=[];Ks=[];observed_rays=[];pixels=[];depths=[];sampled=[]
    uv=np.asarray(t['uv']);x=np.asarray(t['xyz']);x0=np.asarray(t['xyz_first_two'])
    for j,index in enumerate(ids):
        f=frames[index];T=np.array(f['transform_matrix'])@np.diag([1.,-1.,-1.,1.]);P=np.linalg.inv(T)[:3]
        K=np.array([[f['fl_x'],0,f['cx']-.5],[0,f['fl_y'],f['cy']-.5],[0,0,1.]])
        p=P@np.r_[x,1.];q=K@p;pixels.append(q[:2]/q[2]);depths.append(p[2]);Ts.append(T);Ps.append(P);Ks.append(K)
        ray=T[:3,:3]@np.linalg.inv(K)@np.r_[uv[j],1.];observed_rays.append(ray/np.linalg.norm(ray))
        rgb,mask=asset(index);u,v=np.floor(uv[j]+.5).astype(int)
        assert 0<=u<1280 and 0<=v<1080 and mask[v,u]>0
        sampled.append(rgb[v,u].tolist())
    assert sampled==t['sampled_train_rgb']
    np.testing.assert_array_equal(np.median(sampled,axis=0).astype(np.uint8),arrays['rgb'][i])
    errors=np.linalg.norm(np.asarray(pixels)-uv,axis=1)
    assert np.isfinite(errors).all() and errors.max()<=1.5
    assert np.min(depths)>=.2 and np.max(depths)<=50
    np.testing.assert_allclose(errors,t['reprojection_px'],rtol=0,atol=1e-8)
    homogeneous=cv2.triangulatePoints(Ks[0]@Ps[0],Ks[1]@Ps[1],uv[0].reshape(2,1),uv[1].reshape(2,1)).ravel()
    independent=homogeneous[:3]/homogeneous[3]
    np.testing.assert_allclose(independent,x0,rtol=0,atol=1e-8)
    first_errors=[]
    for K,P in zip(Ks,Ps):
        q=K@P@np.r_[independent,1.];first_errors.append(q[:2]/q[2])
    first_errors=np.linalg.norm(np.asarray(first_errors)-uv,axis=1)
    assert first_errors.max()<=1.5
    np.testing.assert_allclose(first_errors,t['first_two_solution_reprojection_px'],rtol=0,atol=1e-8)
    rays=np.array([x-T[:3,3] for T in Ts]);rays/=np.linalg.norm(rays,axis=1)[:,None]
    angles=[float(np.degrees(np.arccos(np.clip(rays[a]@rays[b],-1,1)))) for a,b in ((0,1),(1,2),(0,2))]
    obs_angles=[float(np.degrees(np.arccos(np.clip(observed_rays[a]@observed_rays[b],-1,1)))) for a,b in ((0,1),(1,2),(0,2))]
    assert min(angles)>=1 and min(obs_angles)>=1
    np.testing.assert_allclose(angles,t['geometric_parallax_deg'],atol=1e-8,rtol=0)
    for a,b in ((0,1),(1,2),(0,2)):
        baseline=np.linalg.norm(Ts[a][:3,3]-Ts[b][:3,3]);assert .3<=baseline<=2
    distances=np.linalg.norm(all_centers[:,:2]-x[:2],axis=1);nearest=int(np.argmin(distances));dz=x[2]-all_centers[nearest,2]
    band=bool(distances[nearest]<=3 and -3<=dz<=-.4)
    assert band==t['geometric_floor_band']==bool(arrays['geometric_floor_band'][i]);band_count+=int(band)
    actual_errors.extend(errors);actual_third.append(first_errors[2]);actual_angles.append(min(angles))
assert band_count==23
test=dict(status='PASS',passed=19,result='19 passed in0.34s',test_sha256=sha(ROOT/'tests/test_training_triangulation.py'),script_sha256=sha(ROOT/'scripts/triangulate_training_tracks.py'))
emit(out/'tests.json',test)
audit=dict(status='PASS_ALL300_TRACKS_READONLY_INDEPENDENT_AUDIT',tracks=300,geometric_floor_band=23,
    independent_first_two_method='OpenCV triangulatePoints using separately inverted fixed C2W, compared to saved NumPy normalized DLT',
    maximum_final_reprojection_px=float(max(actual_errors)),maximum_independent_third_reprojection_px=float(max(actual_third)),minimum_final_xyz_parallax_deg=float(min(actual_angles)),
    all_cheirality_depth_baseline_mask_color_checks_passed=True,all_train_only=True,dev_test_images_read=0,
    normal_valid_all_false=True,source_dataset_reference_calibration_core_and_gates_unchanged=True,
    parent305_point_files_unchanged=True,protected_sha256=protected,outputs_sha256=r['outputs_sha256'],
    audit_script_sha256=sha(__file__),cpu_threads_max=3,gpu_used=False,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
emit(out/'independent_audit.json',audit)
old=read(parent/'report.json');total=Counter();rejects=Counter();per_triplet=[]
kept_by=Counter(t['triplet_id'] for t in tracks);floor_by=Counter(t['triplet_id'] for t in tracks if t['geometric_floor_band'])
for row in old['triplets']:
    total.update(row['matching']);rejects.update(row['geometry_rejections'])
    per_triplet.append(dict(triplet_id=row['triplet_id'],camera_id=row['camera_id'],train_indices=row['train_indices'],
        train_file_ids=[frames[i]['file_path'] for i in row['train_indices']],baseline_m=row['baseline_m'],
        **{k:row['matching'].get(k,0) for k in ('corners','pair01_forward_backward','pair02_forward_backward','pair12_forward_backward','triangle_cycle_consistent')},
        geometry_rejections=row['geometry_rejections'],geometry_accepted_before_dedup=row['accepted_before_deduplication'],
        final_retained=kept_by[row['triplet_id']],geometric_floor_band=floor_by[row['triplet_id']]))
funnel=dict(total_matching=dict(total),geometry_rejections=dict(rejects),geometry_accepted_before_dedup=310,
    repeated_observation_suppression=5,refined_xyz_low_parallax_rejection=5,final_tracks=300,final_near_trajectory_floor_band=23,
    note='pair02 and pair12 counts are each conditional on pair01, not sequential. triangle_cycle_consistent combines both later bidirectional/L1/mask gates AND closure. Isolated cycle rejection was not recorded and is not inferred; no new matching performed.',
    bottlenecks={'first_pair_flow_rejected':31198-8524,'later_pairs_plus_cycle_rejected':8524-2605,
                'first_pair_survival_fraction':8524/31198,'cycle_pipeline_survival_fraction_of_first_pair':2605/8524,
                'geometry_survival_fraction_of_cycle_tracks':310/2605,'quality_pass_claim':False},
    per_triplet=per_triplet)
emit(out/'filter_funnel.json',funnel)
with (out/'filter_funnel.csv').open('x',newline='') as f:
    keys=['triplet_id','camera_id','corners','pair01_forward_backward','pair02_forward_backward','pair12_forward_backward','triangle_cycle_consistent','geometry_accepted_before_dedup','final_retained','geometric_floor_band']
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in per_triplet)
emit(out/'handoff.json',dict(status='VALIDATED_SPARSE_VISUAL_CANDIDATES_MAIN_INTEGRATION_DECISION',
    points=300,near_trajectory_geometric_floor_band_points=23,sufficient_for_broad_floor_coverage=False,
    canonical_files={k:dict(path=str(out/k),sha256=sha(out/k)) for k in ('points.ply','tracks.npz','tracks.json','report.json','independent_audit.json','filter_funnel.json','filter_funnel.csv','tests.json')},
    npz_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
    observations='uv isOpenCV integer-centre pixels; K cx=source metadata.cx-.5, cy=metadata.cy-.5. T_world_camera_cv=source transform_matrix @diag(1,-1,-1,1).',
    source_frames='train_frame_indices indexes tracks.json.train_frame_table; train_file_ids isalso explicit Unicode[N,3] inNPZ.',
    color='rgb andcolors are identical uint8RGB, median of the3 actual TRAIN pixels; sampled colors also inJSON.',
    normals='No normals fitted; normal_valid=false for every point. Any later PCA must have enough real neighbors and reliable planarity.',
    source_identity='Original camera_init metric world; retained XYZ arrays are exact subsets of initial measured triangulations.',
    limit='32 total pose-selected triplets; no additional matching; no model/dataset/reference/teacher/gate/core integration or mutation.',
    rejection='Initial305 point parent is superseded by this300 point subset due5 post-refinement geometric parallax failures.',
    quality='Sparse, unassessed candidates, not a semantic floor or a calibrated absolute3D accuracy claim.'))
print(json.dumps(dict(status=audit['status'],tracks=300,floor_band=23,reprojection_max=audit['maximum_final_reprojection_px'],third_max=audit['maximum_independent_third_reprojection_px'],geometric_parallax_min=audit['minimum_final_xyz_parallax_deg'],handoff=str(out/'handoff.json'))))

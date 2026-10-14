"""Revalidate saved observations only; no images matched or new triplets added."""
import os, sys, json, copy, hashlib, importlib.util, resource, time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('tri',ROOT/'scripts/triangulate_training_tracks.py')
tri=importlib.util.module_from_spec(spec);spec.loader.exec_module(tri)
tri.cv2.setNumThreads(3)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]))
resource.setrlimit(resource.RLIMIT_AS,(2800*1024**2,resource.RLIM_INFINITY))
source=Path(__file__).resolve().parent;out=source/'validated_handoff'
if out.exists():raise FileExistsError('Do not overwrite a handoff')
old_report=json.loads((source/'report.json').read_text());old=json.loads((source/'tracks.json').read_text())
frames=old['train_frame_table'];dataset=Path(old_report['source_dataset'])
actual,meta,_=tri.load_training(dataset);assert actual==frames
for p,h in dict(old_report['source_sha256'],**old_report['train_asset_sha256']).items():assert tri.sha(p)==h,p
for p,h in old_report['outputs_sha256'].items():assert tri.sha(source/p)==h,p
keep=[];retained=[];rejected=[];start=time.monotonic()
for i,t in enumerate(old['tracks']):
    cameras=[tri.camera(frames[j]) for j in t['train_indices']]
    checked,reason=tri.triangulate_track(t['uv'],cameras)
    if reason:
        rejected.append(dict(parent_track_id=i,reason=reason,geometric_parallax_deg=tri.point_parallax(np.array(t['xyz']),cameras),geometric_floor_band=t['geometric_floor_band']))
        continue
    # Preserve the original measured triangulation exactly, not a replacement.
    np.testing.assert_allclose(checked['xyz'],t['xyz'],atol=1e-10,rtol=0)
    geometric=tri.point_parallax(np.array(t['xyz']),cameras)
    assert min(geometric)>=1.
    t=copy.deepcopy(t);t.update(parent_track_id=i,track_id=len(retained),geometric_parallax_deg=geometric)
    keep.append(i);retained.append(t)
assert len(rejected)==5 and {r['reason'] for r in rejected}=={'refined_geometric_low_parallax'}
keep=np.asarray(keep,dtype=np.int64)
with np.load(source/'tracks.npz',allow_pickle=False) as old_npz:
    arrays={k:old_npz[k][keep] for k in old_npz.files}
arrays.update(colors=arrays['rgb'].copy(),train_file_ids=np.asarray([t['train_file_ids'] for t in retained],dtype=str),
              geometric_parallax_deg=np.array([t['geometric_parallax_deg'] for t in retained]),parent_track_id=keep)
assert np.array_equal(arrays['xyz'],np.asarray([t['xyz'] for t in retained]))
assert not arrays['normal_valid'].any()
out.mkdir()
tri.write_json(out/'INCOMPLETE.json',dict(status='revalidating_existing_observations_no_new_matching'))
tri.write_ply(out/'points.ply',arrays['xyz'],arrays['rgb'])
np.savez_compressed(out/'tracks.npz',**arrays)
tri.write_json(out/'tracks.json',dict(tracks=retained,train_frame_table=frames,
    coordinate_system=old['coordinate_system'],normal_estimation=old['normal_estimation']))
tri.write_json(out/'rejected_parent_tracks.json',dict(rejected=rejected,source=str(source),
    rule='Both observed-bearing and finalXYZ geometric pairwise parallax must be>=1degree',
    additional_triplets=0,additional_matching=0,accepted_xyz_modified=False))
def quantiles(a):
    return dict(zip(('min','median','p95','max'),map(float,np.percentile(a,[0,50,95,100])))) if np.size(a) else None
report=copy.deepcopy(old_report)
report.update(status='VALIDATED_CONSERVATIVE_TRAIN_VISUAL_HANDOFF_QUALITY_UNASSESSED',
    parent_experiment=str(source),parent_report_sha256=tri.sha(source/'report.json'),
    parent_outputs_sha256=old_report['outputs_sha256'],parent_script_sha256=old_report['script_sha256'],
    script_sha256=tri.sha(ROOT/'scripts/triangulate_training_tracks.py'),
    revalidation_script_sha256=tri.sha(__file__),exact_triangular_tracks=len(retained),
    points_before_final_geometric_parallax_gate=len(old['tracks']),final_geometric_parallax_rejections=len(rejected),
    additional_triplets=0,additional_matching=0,accepted_xyz_modified=False,
    warnings=['LOW_NEAR_TRAJECTORY_GEOMETRIC_FLOOR_BAND_COUNT; sufficiency not established',
              'Initial ray-only parallax gate missed5 refinedXYZ points; this handoff removes them',
              'Original matching diagnostics refer to32 triplets; no suppressed duplicate was resurrected'],
    reprojection_px=quantiles(arrays['reprojection_px']),
    independent_third_reprojection_px=quantiles(arrays['first_two_solution_reprojection_px'][:,2]),
    minimum_pairwise_parallax_deg=quantiles(arrays['parallax_deg'].min(1)),
    minimum_geometric_parallax_deg=quantiles(arrays['geometric_parallax_deg'].min(1)),
    revalidation_resources=dict(cpu_threads_max=3,gpu_used=False,elapsed_s=time.monotonic()-start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
    outputs_sha256={p:tri.sha(out/p) for p in ('points.ply','tracks.npz','tracks.json','rejected_parent_tracks.json')})
xyz=arrays['xyz'];distance=arrays['nearest_train_center_xy_distance_m'];dz=arrays['relative_world_z_m'];band=arrays['geometric_floor_band']
report['geometric_floor_band'].update(points=int(band.sum()),points_all_depths=len(retained),
    nearest_xy_le3m=int((distance<=3).sum()),band_within8m=int(((distance<=8)&(dz>=-3)&(dz<=-.4)).sum()),
    band_3to8m=int(((distance>3)&(distance<=8)&(dz>=-3)&(dz<=-.4)).sum()),
    by_camera={c:sum(t['geometric_floor_band'] for t in retained if t['camera_id']==c) for c in report['selection']})
tri.write_json(out/'report.json',report)
(out/'INCOMPLETE.json').unlink()
tri.write_json(source/'SUPERSEDED.json',dict(status='SUPERSEDED_DO_NOT_INTEGRATE_PARENT305_POINTS',
    replacement=str(out),reason='Five observed-ray-valid points fail finalXYZ geometric parallax>=1degree',
    parent_files_preserved=True,replacement_report_sha256=tri.sha(out/'report.json')))
print(json.dumps(dict(handoff=str(out),tracks=len(retained),geometric_floor_band=int(band.sum()),
    final_geometric_parallax_min=float(arrays['geometric_parallax_deg'].min()),rejected=len(rejected))))

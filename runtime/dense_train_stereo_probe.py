"""CPU-only dense TRAIN stereo proposal; never modifies a reconstruction or teacher.
Side-facing, pose-preselected triplets; rectified SGBM with explicit LR check,
texture/patch checks, two bidirectional independent image tracks into third view,
then strict fixed-camera three-view geometry. Photometric ambiguity remains.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='3'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,resource,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import cv2,numpy as np
from triangulate_training_tracks import camera,sha,load_training,ImageCache,floor_band,write_ply
cv2.setNumThreads(3)
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[8:11]))
DATA=ROOT/'data/garage_fidelity_depthv3/block_000';OUT=ROOT/'reports/block000_dense_stereo_v2'
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir();(OUT/'INCOMPLETE.json').write_text('{"status":"running"}')
frames,meta,prep=load_training(DATA)
parent=ROOT/'reports/block000_visual_triangulation/triplet_plan.json';plan0=json.load(open(parent))
assert frames==plan0['train_frame_table']
plan=[]
for cid in ('camera_2','camera_5'):
 rows=[t for t in plan0['triplets'] if t['camera_id']==cid]
 plan.extend(rows[i] for i in (1,3,5,7))
policy=dict(pixel_stride=4,disparity_count=256,disparity_lr_error=.75,stereo_zncc=.85,patch_std_min=4.,
 third_forward_backward_px=.75,third_pair_cycle_px=.75,lk_photometric_l1=20.,third_initial_reprojection_px=1.5,
 final_reprojection_px=1.5,minimum_all_pair_parallax_deg=1.,depth_min_m=.2,depth_max_m=50.,maximum_output_points=250000)
pin={str(DATA/n):sha(DATA/n) for n in ('transforms.json','preparation.json','lidar_reference.npz','points3D.ply')}
pin[str(parent)]=sha(parent);pin[str(Path(__file__).resolve())]=sha(__file__)
(OUT/'plan.json').write_text(json.dumps(dict(plan=plan,train_frame_table=frames,policy=policy,source_sha256=pin,heldout_images_read=0),indent=2))
cache=ImageCache(DATA,frames);start=time.monotonic();stats=[];chunks=[]
def sample(a,uv):
 if not len(uv):return np.empty((0,)+a.shape[2:],a.dtype)
 values=[]
 for part in np.array_split(uv,max(1,(len(uv)+16383)//16384)):
  value=cv2.remap(a,np.asarray(part[:,0],np.float32).reshape(-1,1),np.asarray(part[:,1],np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  values.append(value.reshape((len(part),)+a.shape[2:]))
 return np.concatenate(values)
def zncc(a,b,u,v):
 x=[];y=[]
 for dy in (-4,-2,0,2,4):
  for dx in (-4,-2,0,2,4):
   x.append(sample(a,u+[dx,dy]));y.append(sample(b,v+[dx,dy]))
 x=np.asarray(x,dtype=np.float32).T;y=np.asarray(y,dtype=np.float32).T
 sx=x.std(1);sy=y.std(1);x-=x.mean(1)[:,None];y-=y.mean(1)[:,None]
 z=(x*y).mean(1)/np.maximum(sx*sy,1e-8)
 return z,sx,sy
def project(xyz,cameras):
 ps=np.asarray([(xyz-T[:3,3])@T[:3,:3] for T,K,P in cameras]).transpose(1,0,2)
 ks=np.asarray([K for T,K,P in cameras]);q=np.einsum('cij,ncj->nci',ks,ps)
 return q[:,:,:2]/q[:,:,2:],ps[:,:,2]
def parallax(xyz,cameras):
 rays=xyz[:,None,:]-np.asarray([T[:3,3] for T,K,P in cameras])[None,:,:]
 rays/=np.linalg.norm(rays,axis=2)[:,:,None]
 return np.stack([np.degrees(np.arccos(np.clip((rays[:,i]*rays[:,j]).sum(1),-1,1))) for i,j in ((0,1),(1,2),(0,2))],1)
def track(a,b,u,v,mask):
 args=dict(winSize=(31,31),maxLevel=1,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),minEigThreshold=1e-4,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
 u=np.float32(u).reshape(-1,1,2);init=np.float32(v).reshape(-1,1,2)
 q,s,e=cv2.calcOpticalFlowPyrLK(a,b,u,init.copy(),**args)
 back,sb,_=cv2.calcOpticalFlowPyrLK(b,a,q,u.copy(),**args)
 q=q.reshape(-1,2);fb=np.linalg.norm(back.reshape(-1,2)-u.reshape(-1,2),axis=1)
 good=(s.ravel()>0)&(sb.ravel()>0)&np.isfinite(q).all(1)&(fb<=.75)&(e.ravel()<=20.)&(sample(mask,q)>254)
 return q,good,fb,e.ravel()
for tid,t in enumerate(plan):
 cameras=[camera(frames[i]) for i in t['train_indices']];images=[cache.get(i) for i in t['train_indices']]
 (Ta,Ka,_),(Tb,Kb,_),_=cameras
 R=Tb[:3,:3].T@Ta[:3,:3];T=Tb[:3,:3].T@(Ta[:3,3]-Tb[:3,3])
 size=(1280,1080)
 R1,R2,P1,P2,Q,roi1,roi2=cv2.stereoRectify(Ka,None,Kb,None,size,R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
 assert abs(P2[0,3])>abs(P2[1,3]),'Horizontal lateral baseline required'
 maps=[cv2.initUndistortRectifyMap(K,None,RR,PP[:,:3],size,cv2.CV_32FC1) for K,RR,PP in ((Ka,R1,P1),(Kb,R2,P2))]
 rect=[cv2.remap(im[0],*m,cv2.INTER_LINEAR) for im,m in zip(images,maps)]
 masks=[cv2.remap(im[2],*m,cv2.INTER_NEAREST) for im,m in zip(images,maps)]
 md=0 if P2[0,3]<0 else -256;rev=-md-256
 def disp(a,b,lo):
  return cv2.StereoSGBM_create(minDisparity=lo,numDisparities=256,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=15,speckleWindowSize=100,speckleRange=1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY).compute(a,b).astype(np.float32)/16
 d=disp(rect[0],rect[1],md);dr=disp(rect[1],rect[0],rev)
 yy,xx=np.mgrid[16:1064:4,16:1264:4];u=np.column_stack((xx.ravel(),yy.ravel())).astype(float);dd=d[yy,xx].ravel()
 v=u.copy();v[:,0]-=dd;back=sample(dr,v)
 good=(dd>md)&(dd<md+255)&(back>rev)&(back<rev+255)&(abs(dd+back)<=.75)&(sample(masks[0],u)>254)&(sample(masks[1],v)>254)
 u,v=u[good],v[good];row=dict(triplet_id=tid,camera_id=t['camera_id'],train_indices=t['train_indices'],sgbm_lr_valid=len(u))
 z,s1,s2=zncc(rect[0],rect[1],u,v);good=(z>=.85)&(s1>=4)&(s2>=4);u,v=u[good],v[good];row['textured_correlated']=len(u)
 ua=np.column_stack([sample(m,u) for m in maps[0]]);ub=np.column_stack([sample(m,v) for m in maps[1]])
 H=cv2.triangulatePoints(Ka@cameras[0][2],Kb@cameras[1][2],ua.T,ub.T);xyz=(H[:3]/H[3]).T
 projected,depth=project(xyz,cameras);pa=parallax(xyz,cameras)
 good=np.isfinite(xyz).all(1)&(depth>=.2).all(1)&(depth<=50).all(1)&(pa>=1).all(1)&(sample(images[2][2],projected[:,2])>254)
 ua,ub,xyz,projected,depth,pa=[x[good] for x in (ua,ub,xyz,projected,depth,pa)];row['pair_geometry']=len(xyz)
 if not len(xyz):stats.append(row);print(json.dumps(row),flush=True);continue
 uc,g1,fb1,e1=track(images[0][0],images[2][0],ua,projected[:,2],images[2][2]);uc2,g2,fb2,e2=track(images[1][0],images[2][0],ub,projected[:,2],images[2][2]);cycle=np.linalg.norm(uc-uc2,axis=1)
 uv=np.stack((ua,ub,(uc+uc2)/2),1);err=np.linalg.norm(projected-uv,axis=2)
 good=g1&g2&(cycle<=.75)&(err<=1.5).all(1)
 uv,xyz,err,fb1,fb2,e1,e2,cycle=[x[good] for x in (uv,xyz,err,fb1,fb2,e1,e2,cycle)];row['third_view_verified']=len(xyz)
 if not len(xyz):stats.append(row);print(json.dumps(row),flush=True);continue
 # Linear 3-view DLT; first-two XYZ and independent third errors retained.
 A=[]
 for j,(TT,KK,PP) in enumerate(cameras):
  q=np.column_stack((uv[:,j],np.ones(len(uv))))@np.linalg.inv(KK).T
  A.extend((q[:,0,None]*PP[2]-PP[0],q[:,1,None]*PP[2]-PP[1]))
 _,sv,vh=np.linalg.svd(np.stack(A,1));xyz3=vh[:,-1,:3]/vh[:,-1,3:]
 pp,zz=project(xyz3,cameras);ee=np.linalg.norm(pp-uv,axis=2);angles=parallax(xyz3,cameras)
 good=np.isfinite(xyz3).all(1)&(zz>=.2).all(1)&(zz<=50).all(1)&(ee<=1.5).all(1)&(angles>=1).all(1)&(sv[:,-2]/np.maximum(sv[:,-1],1e-15)>=10)
 cols=np.stack([sample(im[1],uv[:,j]) for j,im in enumerate(images)],1);rgb=np.median(cols,axis=1).astype(np.uint8)
 out=dict(xyz=xyz3,xyz_first_two=xyz,uv=uv,reprojection_px=ee,first_two_solution_reprojection_px=err,geometric_parallax_deg=angles,depths_m=zz,rgb=rgb,third_fb_px=np.column_stack((fb1,fb2)),third_photometric_l1=np.column_stack((e1,e2)),third_cycle_px=cycle,train_frame_indices=np.tile(t['train_indices'],(len(xyz),1)),triplet_id=np.full(len(xyz),tid))
 chunks.append({k:v[good] for k,v in out.items()});row['accepted']=int(good.sum());stats.append(row);print(json.dumps(row),flush=True)
 assert sum(len(c['xyz']) for c in chunks)<=250000,'Bounded output budget exceeded'
assert chunks,'No validated dense visual samples'
a={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};centers=np.asarray([camera(f)[0][:3,3] for f in frames]);xy,dz,band,nearest=floor_band(a['xyz'],centers)
a.update(geometric_floor_band=band,nearest_train_center_xy_distance_m=xy,relative_world_z_m=dz,nearest_train_center_index=nearest,normal_valid=np.zeros(len(xy),bool))
np.savez_compressed(OUT/'tracks.npz',**a);write_ply(OUT/'points.ply',a['xyz'],a['rgb'])
assert all(sha(p)==v for p,v in pin.items());assert all(sha(p)==v for p,v in cache.hashes.items())
report=dict(status='COMPLETED_EXPERIMENTAL_DENSE_TRAIN_POINTS_QUALITY_UNASSESSED',tracks=len(xy),floor_band_points=int(band.sum()),band_within8m=int(((xy<=8)&(dz>=-3)&(dz<=-.4)).sum()),triplets=stats,policy=policy,source_sha256=pin,train_asset_sha256={str(p):v for p,v in cache.hashes.items()},dev_images_read=0,test_images_read=0,gpu_used=False,normal_estimation=False,pose_fit=False,plane_fill=False,script_sha256=sha(__file__),elapsed_s=time.monotonic()-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,outputs_sha256={n:sha(OUT/n) for n in ('tracks.npz','points.ply','plan.json')},limitations=['Correlated dense stereo pixels are not independent surface observations.','Third-image LK is initialized with pair geometry; strict photometric, backward, cycle and independent pre-refinement reprojection gates reduce but do not eliminate ambiguity.','Inherited fixed calibration/poses, image motion blur, specular floors and repeated structures may still bias geometry.','No integration, semantic-floor claim, new depth teacher or reconstruction-quality acceptance.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2));(OUT/'INCOMPLETE.json').unlink();print(json.dumps({k:report[k] for k in ('status','tracks','floor_band_points','band_within8m','elapsed_s','peak_rss_mb')}),flush=True)

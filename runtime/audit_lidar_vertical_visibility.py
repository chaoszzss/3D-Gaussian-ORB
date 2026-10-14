"""Read-only first-six-scan vertical-return audit with measured static IMU up."""
import json,hashlib,datetime,os
from pathlib import Path
import numpy as np,rosbag
ROOT=Path(__file__).resolve().parents[1];BAG=Path('/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_czy_all.bag');OUT=ROOT/'reports/lidar_vertical_visibility_height_hint.json'
assert not OUT.exists()
scans=[];acc=[];gyro=[];records=[];first=None
with rosbag.Bag(str(BAG)) as bag:
 for topic,m,record in bag.read_messages(topics=['/orbvi/lidar/custom','/orbvi/imu']):
  t=m.header.stamp.to_nsec()
  if topic=='/orbvi/lidar/custom':
   if first is None:first=t
   if len(scans)<6:
    scans.append(np.asarray([(p.x,p.y,p.z) for p in m.points],np.float64));records.append(dict(header_ns=t,record_ns=record.to_nsec(),frame_id=m.header.frame_id,points=len(m.points)))
  elif first is not None and first<=t<=first+3_000_000_000:
   a=m.linear_acceleration;g=m.angular_velocity;acc.append([a.x,a.y,a.z]);gyro.append([g.x,g.y,g.z])
  if first is not None and t>first+3_100_000_000 and len(scans)==6 and len(acc)>100:break
R=np.diag([1.,-1.,-1.]);am=np.mean(acc,axis=0);up=R.T@am;up/=np.linalg.norm(up)
p=np.concatenate(scans);finite=np.isfinite(p).all(1);ranges=np.linalg.norm(p,axis=1);keep=finite&(ranges>.2)&(ranges<80);p=p[keep]
height=p@up;horizontal=np.sqrt(np.maximum(0,(p*p).sum(1)-height*height));elevation=np.degrees(np.arctan2(height,horizontal))
negative=elevation<0
q=[0,.001,.01,.05,.5,.95,1]
def quant(x):return {str(a):float(b) for a,b in zip(q,np.quantile(x,q))} if len(x) else {}
scenarios=[]
for H in [2.35,2.65,2.95]:
 theoretical=H/np.tan(np.radians(-elevation[negative]))
 near_band=np.abs(height+H)<=.3
 scenarios.append(dict(assumed_sensor_height_m=H,assumed_horizontal_ground_intersection_radii_m=quant(theoretical),observed_returns_in_height_band_plusminus_0p3m=int(near_band.sum()),observed_band_horizontal_range_m=quant(horizontal[near_band])))
r=dict(status='READONLY_MEASURED_RETURN_ANGLE_AUDIT_CONDITIONAL_GROUND_RANGE',bag=str(BAG),bag_stat=dict(bytes=BAG.stat().st_size,mtime_ns=BAG.stat().st_mtime_ns),scans=records,raw_points=int(keep.size),valid_points=len(p),static_imu_samples=len(acc),static_mean_acc_mps2=am.tolist(),static_std_acc_mps2=np.std(acc,axis=0).tolist(),static_mean_gyro_rad_s=np.mean(gyro,axis=0).tolist(),configured_lidar_to_imu_rotation=R.tolist(),estimated_up_in_lidar=up.tolist(),elevation_deg_quantiles=quant(elevation),negative_elevation_deg_quantiles=quant(elevation[negative]),negative_returns=int(negative.sum()),height_scenarios=scenarios,user_hint=dict(approx_height_m=2.65,measured=False,sensor_not_specified=True),gpu_used=False,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations=['First-six-scan measured return angles are neither the device nominal field of view nor proof of all available beams.','Static accelerometer up uses existing configured LiDAR-to-IMU rotation, which is inherited calibration rather than independent ground truth.','Ground intersection radii assume an unobstructed horizontal plane below the same sensor; user approximate height may refer to a different sensor.','No geometry, plane, depth target, pose or dataset was changed; no full bag SHA256 or full-route completeness claim.'],written_at=datetime.datetime.now().astimezone().isoformat())
OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('valid_points','static_imu_samples','negative_elevation_deg_quantiles','height_scenarios')}))

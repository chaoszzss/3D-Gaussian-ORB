"""Read six raw Livox payloads from the requested bag; no map/trajectory mutation."""
import datetime,json,os
from pathlib import Path
import numpy as np
import rosbag
ROOT=Path(__file__).resolve().parents[1];bagpath=Path('/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_czy_all.bag')
old=Path('/home/chaoszzss/dataset/CZY_Dataset/all/orbvi_4vl_data_czy_all.bag')
rows=[];totals={f'{lo}_{hi}':0 for lo,hi in [(0,2),(2,4),(4,8),(8,12),(12,20),(20,50)]}
with rosbag.Bag(str(bagpath)) as bag:
 for _,msg,record in bag.read_messages(topics=['/orbvi/lidar/custom']):
  p=np.array([(v.x,v.y,v.z) for v in msg.points]);r=np.linalg.norm(p[:,:2],axis=1);mask=p[:,2]<-1.3
  counts={f'{lo}_{hi}':int(np.sum(mask&(r>=lo)&(r<hi))) for lo,hi in [(0,2),(2,4),(4,8),(8,12),(12,20),(20,50)]}
  for k,v in counts.items():totals[k]+=v
  rows.append({'header_ns':msg.header.stamp.to_nsec(),'record_ns':record.to_nsec(),'frame_id':msg.header.frame_id,'points':len(p),'negative_raw_z_below_minus1p3m_by_xy_radius':counts})
  if len(rows)==6:break
prior=ROOT.parent # no opaque source substitution
report={'status':'raw_first_six_scans_rechecked','bag':str(bagpath),'same_inode_as_historical_gravity_audit_bag':os.path.samefile(bagpath,old),'bag_bytes':bagpath.stat().st_size,'topic':'/orbvi/lidar/custom','scans':rows,'sum_negative_raw_z_below_minus1p3m_by_xy_radius':totals,'units':'metres in reported raw Livox frame','scope':'Finite start-of-recording raw returns; negative raw Z is not a universal semantic-floor label. Frame/world and scene interpretation supported separately by existing gravity audit. No points added, no global completeness claim.','gpu_used':False,'written_at':datetime.datetime.now().astimezone().isoformat()}
path=ROOT/'reports/nearfield_raw_returns_v2.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'same_source':report['same_inode_as_historical_gravity_audit_bag'],'scans':len(rows),'counts':totals}))

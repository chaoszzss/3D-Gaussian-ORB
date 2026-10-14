"""Native CPU helper reference for tiny/near/grazing fixed synthetic surfels."""
import sys,json
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from assemble_core_gaussians_lidar_ownership import surfel_projection,ray_alpha
camera={'camera_to_worlds_ns':[[1,0,0,0],[0,1,0,0],[0,0,1,0]],'fl_x':320.,'fl_y':320.,'cx':320.,'cy':270.,'w':640,'h':540}
out=[]
fixtures=[('tiny_positive',[0,0,-3],[.001,.001,0],[1,0,0,0],[320,270]),
 ('tiny_positive',[0,0,-3],[.001,.001,0],[1,0,0,0],[321,270]),
 ('extreme_grazing',[0,0,-3],[.08,.0001,0],[np.cos(np.deg2rad(89.99)/2),np.sin(np.deg2rad(89.99)/2),0,0],[319,269]),
 ('extreme_grazing',[0,0,-3],[.08,.0001,0],[np.cos(np.deg2rad(89.99)/2),np.sin(np.deg2rad(89.99)/2),0,0],[320,270]),
 ('near_center_rejected',[0,0,-.1],[.001,.001,0],[1,0,0,0],[320,270]),
 ('far_retained',[0,0,-120],[.001,.001,0],[1,0,0,0],[320,270]),
 ('opaque_center',[.5/320*3,-.5/320*3,-3],[.3,.3,0],[1,0,0,0],[320,270])]
for kind,center,scales,q,pixel in fixtures:
 proj=surfel_projection(np.array([center],float),np.array([scales],float),np.array([q],float),camera)
 ids,a,depth=ray_alpha(proj,np.array([.9999]),pixel)
 cr=np.cross(pixel[0]*proj[2]-proj[0],pixel[1]*proj[2]-proj[1]);uv=cr[:,:2]/cr[:,2:3]
 rho3=float((uv*uv).sum());rho2=float(2*((proj[3]-pixel)**2).sum())
 out.append({'kind':kind,'camera':camera,'center':center,'scales':scales,'quaternion_wxyz':list(map(float,q)),
 'pixel':pixel,'opacity':.9999,'T':[t[0].tolist() for t in proj[:3]],'native_active':bool(len(ids)),
 'native_alpha':float(a[0]) if len(a) else 0,'rho3d':rho3,'rho2d':rho2})
 print(kind,pixel,'active',bool(len(ids)),'rho3d',rho3,'rho2d',rho2,'depth',depth.tolist())
(ROOT/'reports/viewer_parity_v2_sparse/toy_cases.json').write_text(json.dumps(out,indent=2)+'\n')

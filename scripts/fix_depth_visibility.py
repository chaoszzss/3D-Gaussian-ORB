#!/usr/bin/env python3
"""Reject derived targets hidden by a closer untrusted measured foreground."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,cv2
from orbvi_gaussian.points import project_depth
from orbvi_gaussian.io import output_path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path)
    a=p.parse_args();base=output_path(a.data_root)
    for directory in sorted(base.glob('block_*')):
        marker=directory/'DEPTH_VISIBILITY_FIXED.json'
        if marker.exists():continue
        if not (directory/'preparation.json').exists():continue
        meta=json.loads((directory/'transforms.json').read_text());xyz=np.load(directory/'lidar_reference.npz')['xyz']
        before,after=0,0
        for f in meta['frames']:
            T=np.array(f['transform_matrix'])@np.diag([1.,-1.,-1.,1.])
            K=np.array([[f['fl_x'],0,f['cx']-.5],[0,f['fl_y'],f['cy']-.5],[0,0,1.]])
            foreground,valid=project_depth(xyz,T,K,f['w'],f['h'])
            path=directory/f['lidar_depth_path'];depth=np.load(path)
            mask=cv2.imread(str(directory/f['lidar_depth_mask_path']),0)>0
            before+=int(mask.sum());mask &= valid & (depth<=foreground+np.maximum(.1,foreground*.01));after+=int(mask.sum())
            depth[~mask]=0
            tmp=path.with_name(path.stem+'.visibility.npy');np.save(tmp,depth);tmp.replace(path)
            cv2.imwrite(str(directory/f['lidar_depth_mask_path']),mask.astype('uint8')*255)
        marker.write_text(json.dumps({'original_supported_pixels':before,'visible_supported_pixels':after,
                                     'protocol':'Nearer measured occluders rejected even if their normal is unreliable; no background supervision through unknown foreground'},indent=2))
        print(directory.name,before,after,flush=True)


if __name__=='__main__':main()

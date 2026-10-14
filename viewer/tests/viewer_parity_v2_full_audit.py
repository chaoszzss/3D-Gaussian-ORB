#!/usr/bin/env python3
"""CPU-only audit of frozen full20 v3 captures and their input identities."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
def read(p): return json.loads((ROOT/p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def main():
    result_path='reports/viewer_parity_v2_aa015_full_result.json'
    capture_path='reports/viewer_parity_v2_aa015_full/browser_capture.json'
    native_path='runs/fidelity_v2/original_single_blocks/block_015/metrics.json'
    r,c,m=map(read,(result_path,capture_path,native_path))
    old=read('reports/viewer_parity_v2_original015_result.json')
    oldc=read('reports/viewer_parity_v2_original015/browser_capture.json')
    three=read('reports/viewer_parity_v2_aa015_three_result.json')
    job=read('reports/viewer_parity_v2_aa015_full_job.json')
    assert r['frames']==len(r['rows'])==len(c['rows'])==len(m['frames'])==20
    assert r['same_frame_parity_all_abs_delta_le_1db'] and r['frames_abs_delta_le_1db']==20
    assert r['native_metrics_sha256']==sha(ROOT/native_path)==old['native_metrics_sha256']==three['native_metrics_sha256']
    assert sha(m['source_ply']['path'])==r['source_ply_sha256']==old['source_ply_sha256']==three['source_ply_sha256']
    assert r['source_sha256']==c['source_sha256']==three['source_sha256']
    assert r['native_step']==m['step']==m['inference_model_step']==11999 and r['sh_degree']==m['inference_sh_degree']==3
    for p,d in r['source_sha256'].items():assert sha(ROOT/p)==d
    assert 'RTX 4060' in r['gpu']['renderer'] and not c['errors']
    rows=[]
    for i,(result,actual,native,record) in enumerate(zip(r['rows'],c['rows'],m['frames'],job['frames'])):
        assert result['dataset_index']==actual['dataset_index']==native['dataset_index']==record['dataset_index']==i
        assert record['camera']==native['camera']
        s=actual['snapshot'];n=native['camera'];matrix=np.vstack([n['camera_to_worlds_ns'],[0,0,0,1]])
        assert s['camera_to_world_column_major']==matrix.flatten(order='F').tolist()
        near,far=.2,100.
        projection=np.array([[2*n['fl_x']/n['w'],0,1-2*n['cx']/n['w'],0],
            [0,2*n['fl_y']/n['h'],2*n['cy']/n['h']-1,0],
            [0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]])
        assert s['projection_column_major']==projection.flatten(order='F').tolist()
        prior=next(x['snapshot'] for x in oldc['rows'] if x['dataset_index']==i)
        assert s['camera_to_world_column_major']==prior['camera_to_world_column_major']
        assert s['projection_column_major']==prior['projection_column_major']
        assert s['viewport']==[n['w'],n['h']]==[640,540]
        assert s['frame']['background_rgb']==r['background_rgb']==prior['frame']['background_rgb']
        assert s['frame']['inference_sh_degree']==s['models'][0]['shRestore']['degree']==3
        assert len(s['models'])==1 and s['models'][0]['count']==285964 and s['models'][0]['visible']
        assert s['models'][0]['plySha256']==r['source_ply_sha256']
        assert s['frame']['direction_support_sha256'] is None and not s['models'][0]['surface']['directionSupport']
        assert s['renderer']['mode']=='inria_2dgs_v3' and s['renderer']==r['resolved_renderer']
        assert s['nativeAa']['passes']==1 and s['compositing']['target']=='RGBA32F'
        assert actual['stable'] and not s['sortDirty'] and not s['sorting']
        assert result['browser_png_sha256']==actual['sha256']==sha(ROOT/'reports/viewer_parity_v2_aa015_full'/actual['file'])
        assert result['native_float_sha256']==old['rows'][i]['native_float_sha256']==sha(ROOT/Path(native_path).parent/f'{i:03d}_float.npz')
        assert result['native_vs_gt']==old['rows'][i]['native_vs_gt']
        repeat=next((x for x in three['rows'] if x['dataset_index']==i),None)
        if repeat:assert repeat['browser_png_sha256']==result['browser_png_sha256']
        rows.append({'dataset_index':i,'camera_id':native['camera_id'],'group_id':native['group_id'],
          'native_pose_exact':True,'intrinsic_projection_exact':True,'historical_camera_projection_exact':True,
          'source_image_path':native['source_image_path'],'source_image_sha256':sha(native['source_image_path']),
          'acceptance_mask_path':native['acceptance_mask_path'],'acceptance_mask_sha256':sha(native['acceptance_mask_path']),
          'native_float_sha256':result['native_float_sha256'],'png_sha256':actual['sha256'],
          'repeat_three_png_byte_identical':True if repeat else None})
    assert not any(str(r['lock_identity']['owner_pid'])==line.split()[4] for line in Path('/proc/locks').read_text().splitlines() if len(line.split())>5)
    summary={'status':'full20_input_identity_audit_passed','gpu_used_by_audit':False,'renderer':'inria_2dgs_v3',
      'result_sha256':sha(ROOT/result_path),'capture_sha256':sha(ROOT/capture_path),'source_ply_sha256':r['source_ply_sha256'],
      'renderer_source_sha256':r['source_sha256'],'same_renderer_as_passing_three':True,'same_native_float_as_historical_twenty':True,
      'frames':20,'all_native_pose_exact':True,'all_intrinsic_projection_exact':True,'all_viewport_background_SH_model_match':True,
      'three_frame_repeated_pngs_byte_identical':3,'browser_lock_owner_no_longer_holds_lock':True,
      'ssim_delta_mean':float(np.mean([x['ssim_delta'] for x in r['rows']])),
      'direct_psnr_min_db':min(x['browser_vs_native']['psnr_db'] for x in r['rows']),
      'max_abs_delta_frame':max(r['rows'],key=lambda x:abs(x['psnr_delta_db']))['dataset_index'],'rows':rows}
    (ROOT/'reports/viewer_parity_v2_aa015_full_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
if __name__=='__main__':main()

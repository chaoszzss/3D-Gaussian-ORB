#!/usr/bin/env python3
"""Render disjoint real camera groups, score valid pixels, and export actual GS."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbvi_gaussian.training import confined_environment
os.environ.update(confined_environment())
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity
from nerfstudio.utils.eval_utils import eval_setup
from orbvi_gaussian.io import new_output_dir, write_json
from orbvi_gaussian.points import read_point_cloud


def export_ply(model, path):
    """Official 3DGS channel ordering, log-scale, opacity logits and wxyz."""
    means = model.means.detach().cpu().numpy()
    dc = model.features_dc.detach().cpu().numpy().reshape(len(means), 3)
    rest = model.features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
    opacity = model.opacities.detach().cpu().numpy().reshape(len(means), 1)
    scales = model.scales.detach().cpu().numpy()
    rotations = model.quats.detach().cpu().numpy()
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    # Surface orientation is encoded by wxyz, not by the historical placeholder
    # nx/ny/nz fields. Export its third local axis for interoperable diagnostics.
    w, x, y, z = rotations.T
    normals = np.column_stack((2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)))
    values = np.concatenate([means, normals, dc, rest, opacity, scales, rotations], axis=1).astype('<f4')
    valid = np.isfinite(values).all(axis=1) & (opacity[:, 0] >= -5.5373)
    values = values[valid]
    names = ['x','y','z','nx','ny','nz'] + ['f_dc_%d'%i for i in range(3)]
    names += ['f_rest_%d'%i for i in range(rest.shape[1])] + ['opacity']
    names += ['scale_%d'%i for i in range(3)] + ['rot_%d'%i for i in range(4)]
    with path.open('xb') as f:
        header = 'ply\nformat binary_little_endian 1.0\ncomment actual trained Gaussian model; camera_init metres\n'
        header += 'element vertex %d\n' % len(values)
        header += ''.join('property float %s\n'%n for n in names) + 'end_header\n'
        f.write(header.encode('ascii')); f.write(values.tobytes())
    return {'gaussians': len(values), 'bbox_m': {'min': values[:,:3].min(axis=0).tolist(),
                                               'max': values[:,:3].max(axis=0).tolist()},
            'format': 'binary_little_endian_float32', 'coordinates': 'camera_init',
            'sh_degree': int(round(np.sqrt(rest.shape[1]/3+1)-1))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--split', choices=['dev','test'], default='dev')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--export-ply', action='store_true')
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--surface-filter-m', type=float, default=0.,help='Reject unsupported Gaussian centers; compare dev before accepting')
    p.add_argument('--opacity-min', type=float, default=0.)
    args = p.parse_args()
    out = new_output_dir(args.output)
    torch.cuda.reset_peak_memory_stats()
    config, pipeline, checkpoint, step = eval_setup(args.config.resolve(), test_mode='val' if args.split=='dev' else 'test')
    backend_load_step = int(pipeline.model.step)
    pipeline.model.step = int(step)
    dataset = pipeline.datamanager.eval_dataset
    data_root = config.pipeline.datamanager.data.resolve()
    meta = json.loads((data_root/'transforms.json').read_text())
    frames = {(data_root/f['file_path']).resolve(): f for f in meta['frames']}
    filtered_from=len(pipeline.model.means)
    if args.surface_filter_m>0 or args.opacity_min>0:
        ref=np.load(data_root/'lidar_reference.npz')['xyz']
        model=pipeline.model
        distance=cKDTree(ref).query(model.means.detach().cpu().numpy(),workers=4)[0]
        opacity=torch.sigmoid(model.opacities).detach().cpu().numpy().ravel()
        keep=(distance<=args.surface_filter_m)&(opacity>=args.opacity_min)
        ids=torch.as_tensor(np.flatnonzero(keep),device=model.means.device)
        for key in list(model.gauss_params):
            model.gauss_params[key]=torch.nn.Parameter(model.gauss_params[key][ids].detach())
    parser_outputs = pipeline.datamanager.train_dataparser_outputs
    transform = parser_outputs.dataparser_transform.cpu().numpy()
    scale = float(parser_outputs.dataparser_scale)
    if not np.allclose(transform, np.eye(4)[:3], atol=1e-7) or abs(scale-1)>1e-7:
        raise RuntimeError('Training parser changed metric world coordinates; export must undo it explicitly')
    records = []
    count = len(dataset) if args.max_images==0 else min(len(dataset),args.max_images)
    for i in range(count):
        batch = dataset[i]
        gt = batch['image'].cpu().numpy()
        gt = gt.astype(np.float32)/255 if gt.dtype==np.uint8 else gt.astype(np.float32)
        mask = batch.get('mask')
        valid = np.ones(gt.shape[:2],bool) if mask is None else mask.cpu().numpy().reshape(gt.shape[:2])>0
        camera = dataset.cameras[i:i+1].to(pipeline.device)
        frame = frames[dataset.image_filenames[i].resolve()]
        if camera.metadata is None:camera.metadata={}
        camera.metadata['physical_camera_id']=int(frame['camera_id'].split('_')[-1])
        start = time.monotonic()
        with torch.inference_mode(): outputs = pipeline.model.get_outputs_for_camera(camera)
        torch.cuda.synchronize()
        pred = outputs['rgb'].cpu().numpy().clip(0,1)
        mse = float(np.mean((gt[valid]-pred[valid])**2))
        _, ssim_map = structural_similarity(gt,pred,channel_axis=-1,data_range=1.0,full=True)
        eroded = cv2.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        ssim = float(ssim_map.mean(axis=2)[eroded].mean())
        row = {'file_path':frame['file_path'],'group_id':frame['group_id'],'camera_id':frame['camera_id'],
               'psnr_db':float(-10*np.log10(max(mse,1e-12))),'ssim_valid_pixels':ssim,
               'valid_fraction':float(valid.mean()),'render_seconds':time.monotonic()-start}
        if frame.get('lidar_depth_path') and outputs.get('depth') is not None:
            reference = np.load(data_root/frame['lidar_depth_path'])
            if getattr(pipeline.model.config,'optimize_shared_rig',False):
                from orbvi_gaussian.surfels import project_surfel_depth
                measured=np.load(data_root/'lidar_reference.npz')
                corrected=pipeline.model.shared_rig.correct(camera).detach().cpu().numpy()[0]
                T=np.eye(4);T[:3]=corrected;T=T@np.diag([1.,-1.,-1.,1.])
                K=np.array([[frame['fl_x'],0,frame['cx']-.5],[0,frame['fl_y'],frame['cy']-.5],[0,0,1.]])
                reference,supported_corrected=project_surfel_depth(measured['xyz'],measured['normals'],measured['normal_valid'],T,K,frame['w'],frame['h'])
            depth = outputs['depth'].cpu().numpy().squeeze()
            alpha = outputs['accumulation'].cpu().numpy().squeeze()
            ok = valid & (reference>0) & (alpha>.5) & np.isfinite(depth)
            if frame.get('lidar_depth_mask_path'):
                supported=supported_corrected if getattr(pipeline.model.config,'optimize_shared_rig',False) else cv2.imread(str(data_root/frame['lidar_depth_mask_path']),cv2.IMREAD_GRAYSCALE)>0
                ok &= supported
            if ok.any():
                error = depth[ok]-reference[ok]
                row.update(map_projected_depth_median_abs_m=float(np.median(np.abs(error))),
                           map_projected_depth_rmse_m=float(np.sqrt(np.mean(error**2))),
                           map_depth_compared_pixels=int(ok.sum()))
        gt_display = gt.copy(); pred_display = pred.copy()
        gt_display[~valid]=0;pred_display[~valid]=0
        error = np.abs(gt_display-pred_display).mean(axis=2)
        colored = cv2.applyColorMap(np.clip(error*500,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
        panels=[cv2.cvtColor((gt_display*255).astype(np.uint8),cv2.COLOR_RGB2BGR),
                cv2.cvtColor((pred_display*255).astype(np.uint8),cv2.COLOR_RGB2BGR),colored]
        for panel,label in zip(panels,['REAL HELD-OUT','TRAINED GAUSSIAN','ABS ERROR']):
            cv2.putText(panel,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
        cv2.imwrite(str(out/('%03d_comparison.jpg'%i)),np.hstack(panels))
        cv2.imwrite(str(out/('%03d_render.png'%i)),panels[1])
        records.append(row)
    init_xyz,_=read_point_cloud(data_root/meta['ply_file_path'])
    means = pipeline.model.means.detach().cpu().numpy()
    displacement = cKDTree(init_xyz).query(means,workers=4)[0]
    reference_path=data_root/'lidar_reference.npz'
    if reference_path.exists():
        measured_xyz=np.load(reference_path)['xyz']
    elif (ROOT/'data/reference_map_voxel01.npz').exists():
        measured_xyz=np.load(ROOT/'data/reference_map_voxel01.npz')['xyz']
    else:measured_xyz=init_xyz
    measured_distance=cKDTree(measured_xyz).query(means,workers=4)[0]
    from orbvi_gaussian.inria_surfel_backend import model_inria_backend_provenance
    report={'status':'evaluated','split':args.split,'config':str(args.config.resolve()),'checkpoint':str(checkpoint),
            'renderer_provenance':model_inria_backend_provenance(pipeline.model.config),
            'step':int(step),'images':len(records),'frames':records,
            'inference_model_step':int(pipeline.model.step),'backend_default_load_step':backend_load_step,
            'parameter_origin':'trained_checkpoint','checkpoint_is_template':False,
            'gaussian_filter':{'original_count':filtered_from,'surface_distance_max_m':args.surface_filter_m,'opacity_min':args.opacity_min},
            'psnr_db_mean':float(np.mean([r['psnr_db'] for r in records])),
            'ssim_valid_mean':float(np.mean([r['ssim_valid_pixels'] for r in records])),
            'gaussians':len(means),'gaussian_to_initializer_median_m':float(np.median(displacement)),
            'gaussian_to_initializer_p95_m':float(np.percentile(displacement,95)),
            'gaussian_to_measured_map_median_m':float(np.median(measured_distance)),
            'gaussian_to_measured_map_p95_m':float(np.percentile(measured_distance,95)),
            'measured_reference_points':len(measured_xyz),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'world_parser_transform':transform.tolist(),'world_parser_scale':scale,
            'geometry_protocol':'Conditioned on same LIO/map used by initialization; not independent measurement GT',
            'calibration_status':json.loads((data_root/'preparation.json').read_text())['calibration_verified']}
    if args.export_ply:
        report['export']=export_ply(pipeline.model,out/'gaussians.ply')
    if getattr(pipeline.model.config,'optimize_shared_rig',False):
        rig=pipeline.model.shared_rig
        report['shared_rig_correction_ns_local']=rig.correction_matrices().detach().cpu().numpy().tolist()
        report['shared_rig_effective_delta_rotfirst']=rig.effective_delta().detach().cpu().numpy().tolist()
    write_json(out/'metrics.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()

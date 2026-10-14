"""Compare v3 teachers on a FIXED old-reference support/class domain, CPU only."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'scripts')]
import numpy as np
from scipy.spatial import cKDTree
from lidar_loss_units_v3_audit import camera,masks,sha,project_surfel_depth,SurfelDepthPolicy
from measure_surface_quality import unproject,classify_reference,DEFAULT_SETTINGS


def distribution(values):
    return dict(count=len(values), median_m=float(np.median(values)) if len(values) else None,
                abs_error_gt_0_15_count=int((np.abs(values)>.15).sum()),
                too_near_gt_0_15_count=int((values<-.15).sum()))


def main():
    rendered=ROOT/'runs/fidelity_v2/train_640_block015_acceptance'
    metrics=json.loads((rendered/'metrics.json').read_text())
    root=ROOT/'data/garage_fidelity_640_v2/block_015'
    meta=json.loads((root/'transforms.json').read_text())
    frames={f['file_path']:f for f in meta['frames']}
    reference=dict(np.load(root/'lidar_reference.npz'))
    old_reference=Path(metrics['transforms']).parent/'lidar_reference.npz'
    assert sha(old_reference)==sha(root/'lidar_reference.npz')
    tree=cKDTree(reference['xyz'])
    records=[]
    for index,row in enumerate(metrics['frames']):
        frame=frames[row['file_path']]
        pose,K=camera(frame)
        depth_new,valid_new=project_surfel_depth(reference['xyz'],reference['normals'],reference['normal_valid'],
            pose,K,frame['w'],frame['h'],policy=SurfelDepthPolicy.trusted_ground_v3())
        old=np.load(row['source_lidar_depth_path'])
        fixed=masks(row['source_lidar_support_path'])&masks(row['acceptance_mask_path'])&masks(row['source_mask_path'])
        fixed &= np.isfinite(old)&(old>0)
        ids=np.flatnonzero(fixed)
        old=old.ravel()[ids]
        world=unproject(row['camera'],ids,old,.5)
        distance,nearest=tree.query(world,workers=4)
        labels=classify_reference(reference,nearest,distance,row['camera']['camera_to_worlds_ns'][2][3],DEFAULT_SETTINGS)
        new=depth_new.ravel()[ids]; known=valid_new.ravel()[ids]
        blob=np.load(rendered/f'{index:03d}_float.npz')
        prediction=blob['depth'].reshape(-1)[ids]
        alpha=blob['alpha'].reshape(-1)[ids]
        classes={}
        for label,name in enumerate(('floor','wall','other','unknown')):
            use=labels==label; common=use&known
            classes[name]=dict(fixed_denominator=int(use.sum()),old_valid_to_v3_unknown=int((use&~known).sum()),
                old_farther_to_v3_nearer_gt_0_15=int((common&(new<old-.15)).sum()),
                old_nearer_to_v3_farther_gt_0_15=int((common&(new>old+.15)).sum()),
                both_valid_depth_within_0_15=int((common&(np.abs(new-old)<=.15)).sum()),
                old_signed_error_fixed_domain=distribution(prediction[use]-old[use]),
                old_signed_error_common_domain=distribution(prediction[common]-old[common]),
                v3_signed_error_common_domain=distribution(prediction[common]-new[common]),
                missing_alpha_fixed_domain=int((use&(alpha<=.5)).sum()))
        records.append(dict(file_path=frame['file_path'],camera_id=frame['camera_id'],group_id=frame['group_id'],classes=classes))
    totals={}
    keys=('fixed_denominator','old_valid_to_v3_unknown','old_farther_to_v3_nearer_gt_0_15',
          'old_nearer_to_v3_farther_gt_0_15','both_valid_depth_within_0_15','missing_alpha_fixed_domain')
    for camera_id in ('all','camera_2','camera_3','camera_4','camera_5'):
        selected=[r for r in records if camera_id=='all' or r['camera_id']==camera_id]
        totals[camera_id]={name:{key:sum(r['classes'][name][key] for r in selected) for key in keys}
                          for name in ('floor','wall')}
    report=dict(status='FIXED_REFERENCE_TEACHER_MIGRATION_MEASURED',gpu_used=False,records=records,totals=totals,
        scope='Fixed original depth+ego+legacy-mask support and classify_reference labels; v3 rejection stays unknown in the original denominator. Common-domain residuals are explicitly secondary, not a coverage gain. LIO-conditioned reference is not independent GT.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'src/orbvi_gaussian/surfels.py',rendered/'metrics.json',old_reference)},
        commands=["CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-train/bin/python -B runtime/lidar_teacher_migration_v3_audit.py"])
    out=ROOT/'reports/lidar_teacher_migration_v3_fixed_reference.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(out),totals=totals)))


if __name__=='__main__':
    main()

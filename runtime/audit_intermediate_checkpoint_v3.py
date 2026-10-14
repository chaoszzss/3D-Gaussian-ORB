"""CPU-only stable-checkpoint statistics; no GPU, export, or training mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from measure_surface_quality import exact_voxel_counts

torch.set_num_threads(4)


def stream_sha(stream):
    stream.seek(0)
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        result.update(block)
    return result.hexdigest()


def measure(path, run, output):
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        time.sleep(3)
        stable = os.fstat(stream.fileno())
        if (before.st_size,before.st_mtime_ns) != (stable.st_size,stable.st_mtime_ns):
            raise RuntimeError('Checkpoint still being written')
        first_sha = stream_sha(stream)
        again = os.fstat(stream.fileno())
        if (stable.st_size,stable.st_mtime_ns) != (again.st_size,again.st_mtime_ns):
            raise RuntimeError('Checkpoint changed while hashing')
        path_stat = path.stat()
        if path_stat.st_ino != again.st_ino:
            raise RuntimeError('Checkpoint path was replaced')
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        step = int(state['step'])
        tensors = {key.rsplit('.',1)[-1]:value for key,value in state['pipeline'].items() if '.gauss_params.' in key}
        means, logs = tensors['means'].numpy(), tensors['scales'].numpy()
        logits = tensors['opacities'].numpy().reshape(-1).astype(np.float64)
        assert means.shape == logs.shape and means.shape[1] == 3
        assert all(np.isfinite(a).all() for a in (means,logs,logits))
        tangent = np.exp(logs[:,:2].astype(np.float64)).max(axis=1)
        opacity = np.exp(-np.logaddexp(0,-logits))
        cells = np.floor(means.astype(np.float64)/.02).astype(np.int32)
        counts = exact_voxel_counts(cells)
        launch = json.loads((run/'launch.json').read_text())
        refinement_bytes = (run/'refinement.json').read_bytes()
        refinement = json.loads(refinement_bytes)
        events = [event for event in refinement['events'] if event['step'] <= step]
        result = dict(status='INTERMEDIATE_CHECKPOINT_CPU_MEASURED', training_complete=False,
            acceptance_candidate=False, gpu_used=False, actual_checkpoint_step=step,
            checkpoint_path=str(path.relative_to(ROOT)), checkpoint_sha256=first_sha,
            checkpoint_bytes=again.st_size, source_inode=again.st_ino,
            gaussians=len(means), max_gaussians=launch['max_gaussians'],
            budget_fraction=len(means)/launch['max_gaussians'],
            tangential_radius_p50_m=float(np.median(tangent)),
            tangential_radius_p95_m=float(np.quantile(tangent,.95)),
            tangential_radius_max_m=float(tangent.max()),
            tangent_fraction_gt_0_03=float((tangent>.03).mean()),
            tangent_fraction_gt_0_06=float((tangent>.06).mean()),
            tangent_fraction_ge_0_119=float((tangent>=.119).mean()),
            opacity_median=float(np.median(opacity)), opacity_mean=float(opacity.mean()),
            opacity_frac_lt_0_5=float((logits<0).mean()), opacity_frac_ge_0_99=float((opacity>=.99).mean()),
            voxel_size_m=.02, occupied_voxels=len(counts),
            voxel_2cm_multiplicity_median=float(np.median(counts)),
            voxel_2cm_multiplicity_mean=float(counts.mean()),
            voxel_2cm_multiplicity_p95=float(np.quantile(counts,.95)),
            voxel_2cm_multiplicity_max=int(counts.max()),
            voxel_2cm_singleton_fraction=float((counts==1).mean()),
            refinement_events_through_checkpoint=events,
            refinement_snapshot_last_step=refinement['last_callback_step'],
            refinement_snapshot_sha256=hashlib.sha256(refinement_bytes).hexdigest(),
            screen_threshold_fraction=launch['split_screen_size'],
            screen_stop_step=launch['stop_screen_size_at'],
            definitions=dict(radius='max(exp(float64(log_scale_0)), exp(float64(log_scale_1))); all checkpoint Gaussians, not a projected radius or PLY-pruned subset',
                voxel='Exact centre count per occupied floor(world_xyz/0.02) voxel, origin zero; equal occupied-voxel weighting. Reuses Euclid measure_surface_quality.exact_voxel_counts.',
                screen='Native integer 3-sigma bounding radius/max(image dimensions), not diameter. At width1280 threshold.02 requires radius>=26px; front-facing focal640,Z10m s=.12m has ceil(23.04)=24px, below threshold.'),
            protocol='3s stable size/mtime, full SHA before mmap CPU load, second SHA after measurements; no checkpoint/export edits',
            audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            command=['.venv-train/bin/python','runtime/audit_intermediate_checkpoint_v3.py','--run',str(run.relative_to(ROOT)), '--steps',str(step),'--watch-seconds','60'])
        second_sha = stream_sha(stream)
        after = os.fstat(stream.fileno())
        if first_sha != second_sha or (again.st_size,again.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError('Checkpoint bytes changed during measurement')
        result['checkpoint_sha256_verified_after_measurement'] = second_sha
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('actual_checkpoint_step','gaussians','budget_fraction','tangential_radius_p95_m','tangent_fraction_gt_0_03','opacity_median','voxel_2cm_multiplicity_median')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--steps',nargs='+',type=int,default=[2000,4000])
    parser.add_argument('--watch-seconds',type=int,default=900)
    args=parser.parse_args()
    run=args.run.resolve(); deadline=time.monotonic()+args.watch_seconds
    pending=set(args.steps)
    while pending and time.monotonic()<deadline:
        for step in sorted(pending):
            paths=list(run.rglob(f'step-{step:09d}.ckpt'))
            if len(paths)!=1:
                continue
            try:
                measure(paths[0],run,ROOT/f'reports/checkpoint_v3_block015_step{step:06d}_cpu.json')
            except (OSError,RuntimeError,EOFError) as error:
                print('retry',step,str(error),flush=True)
                continue
            pending.remove(step)
        if pending:
            time.sleep(5)
    if pending:
        print('Unobserved requested checkpoints:',sorted(pending),flush=True)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())

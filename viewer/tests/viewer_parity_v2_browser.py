#!/usr/bin/env python3
"""Schedule actual browser parity under the shared GPU lock, then score on CPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports/viewer_parity_v2_original015'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def score(a, b, mask):
    error = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    mse = float(np.mean(np.square(error)[mask]))
    _, full = structural_similarity(a.astype(np.float64), b.astype(np.float64), channel_axis=2, data_range=1., full=True)
    return {'mse': mse, 'psnr_db': -10 * np.log10(max(mse, 1e-20)), 'ssim_masked': float(full.mean(axis=2)[mask].mean())}


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['inria_2dgs_v1', 'inria_2dgs_v2', 'inria_2dgs_v3'], default='inria_2dgs_v1')
    parser.add_argument('--indices', default='')
    parser.add_argument('--score-only', action='store_true', help='Rescore preserved captures without launching GPU')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare manifest/job only; do not request GPU lock or launch browser')
    parser.add_argument('--tag', choices=['original015', 'hdr015_three', 'hdr015_full', 'aa015_three', 'aa015_full'], default='original015')
    args = parser.parse_args()
    if args.profile == 'inria_2dgs_v3':
        assert (args.tag == 'aa015_three' and args.indices == '5,9,8') or (args.tag == 'aa015_full' and not args.indices), 'v3 supports the authorized three-frame gate or complete original015 DEV'
    stem = 'viewer_parity_v2_' + args.tag
    OUTPUT = ROOT / 'reports' / stem
    OUTPUT.mkdir(exist_ok=True)
    metrics_path = ROOT / 'runs/fidelity_v2/original_single_blocks/block_015/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    if args.indices:
        indices = list(map(int, args.indices.split(',')))
        assert len(set(indices)) == len(indices)
        metrics['frames'] = [next(row for row in metrics['frames'] if row['dataset_index'] == idx) for idx in indices]
    model = metrics['source_ply']
    ply_path = Path(model['path'])
    assert sha(ply_path) == model['sha256']
    assert metrics['step'] == metrics['inference_model_step'] == 11999 and metrics['inference_sh_degree'] == 3
    assert metrics.get('direction_support') is None
    if args.tag == 'aa015_full':
        prerequisite = json.loads((ROOT / 'reports/viewer_parity_v2_aa015_three_result.json').read_text())
        assert args.profile == 'inria_2dgs_v3' and len(metrics['frames']) == 20
        assert prerequisite['same_frame_parity_all_abs_delta_le_1db'] and prerequisite['frames'] == 3
        assert prerequisite['source_ply_sha256'] == model['sha256'] and prerequisite['native_step'] == 11999
        for filename, expected in prerequisite['source_sha256'].items():
            assert sha(ROOT / filename) == expected, 'Full20 must retain the passing three-frame renderer code'
    config = Path(metrics['config'])
    assert 'background_color: random' in config.read_text()
    source = ROOT / '.venv-train/lib/python3.10/site-packages/nerfstudio/models/splatfacto.py'
    assert '[0.1490, 0.1647, 0.2157]' in source.read_text()
    background = np.array([.1490, .1647, .2157], dtype=np.float32).tolist()
    manifest = {'schema_version': 1, 'title': '原始015同帧诊断 · 非新地图交付', 'dataset_kind': 'real',
        'coordinates': 'camera_init', 'units': 'm', 'camera_up': [0, 0, 1],
        'renderer': {'schema_version': 1, 'mode': args.profile},
        'blocks': [{'id': 'block_015', 'label': '原始单块015 · native parity', 'floor_id': None,
            'path': str(ply_path.relative_to(ROOT)), 'coordinates': 'camera_init', 'bbox_m': model['bbox_m'],
            'training': {'status': 'trained', 'iterations': 12000},
            'eval': {'status': 'evaluated', 'note': 'original single-block fixed DEV; not full-scene fidelity'}}]}
    manifest_path = ROOT / 'reports' / (stem + '_manifest.json')
    write(manifest_path, manifest)
    frames = []
    for row in metrics['frames']:
        frames.append({'dataset_index': row['dataset_index'], 'camera': row['camera'],
            'source_ply_sha256': model['sha256'], 'direction_support_sha256': None,
            'inference_sh_degree': 3, 'background_rgb': background})
    job = {'manifest': str(manifest_path.relative_to(ROOT)), 'block_id': 'block_015',
        'output_directory': str(OUTPUT.relative_to(ROOT)), 'frames': frames,
        'direction_smoke': args.tag == 'original015'}
    job_path = ROOT / 'reports' / (stem + '_job.json')
    write(job_path, job)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared_no_gpu', 'job': str(job_path.relative_to(ROOT))}), flush=True)
        return
    if args.profile == 'inria_2dgs_v3' and not args.score_only:
        assert os.environ.get('GARAGE_PARITY_MAIN_GPU_RELEASE') == '1', 'v3 GPU queue waits for explicit main release after training'
    if not args.score_only:
        with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
            print('Waiting for exclusive runtime/core_joint_gpu.lock (no browser started).', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print('GPU lock acquired. Starting browser; lock retained until browser exits.', flush=True)
            env = dict(os.environ, GARAGE_PARITY_GPU_LOCK_HELD='1', PYTHONDONTWRITEBYTECODE='1')
            proc = subprocess.Popen(['node', 'viewer/tests/viewer_parity_v2_browser.mjs', str(job_path)], cwd=ROOT,
                env=env, start_new_session=True)
            try:
                returncode = proc.wait(timeout=300)
                if returncode:
                    raise RuntimeError('Browser capture failed: ' + str(returncode))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                fcntl.flock(lock, fcntl.LOCK_UN)
                print('Browser process ended; shared GPU lock released. Scoring on CPU.', flush=True)
    capture = json.loads((OUTPUT / 'browser_capture.json').read_text())
    assert len(capture['rows']) == len(metrics['frames'])
    assert capture['model_sha256'] == model['sha256']
    for actual in capture['rows']:
        assert actual['snapshot']['renderer']['mode'] == args.profile
        assert actual['snapshot']['frame']['background_rgb'] == background
        assert sha(OUTPUT / actual['file']) == actual['sha256']
    rows = []
    for row, actual in zip(metrics['frames'], capture['rows']):
        idx = row['dataset_index']
        assert idx == actual['dataset_index']
        browser = cv2.cvtColor(cv2.imread(str(OUTPUT / actual['file'])), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
        native_path = metrics_path.parent / ('%03d_float.npz' % idx)
        with np.load(native_path, allow_pickle=False) as native:
            native_rgb = native['rgb'].astype(np.float64)
        gt = cv2.cvtColor(cv2.imread(row['source_image_path']), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
        mask = cv2.imread(row['acceptance_mask_path'], cv2.IMREAD_GRAYSCALE) > 0
        assert browser.shape == native_rgb.shape == gt.shape
        native_score, browser_score, direct = score(native_rgb, gt, mask), score(browser, gt, mask), score(browser, native_rgb, mask)
        rows.append({'dataset_index': idx, 'camera_id': row['camera_id'], 'group_id': row['group_id'],
            'native_float_sha256': sha(native_path), 'browser_png_sha256': actual['sha256'],
            'native_vs_gt': native_score, 'browser_vs_gt': browser_score, 'browser_vs_native': direct,
            'psnr_delta_db': browser_score['psnr_db'] - native_score['psnr_db'],
            'ssim_delta': browser_score['ssim_masked'] - native_score['ssim_masked']})
    report = {'status': 'measured', 'model_kind': 'original_single_block015_no_direction_support',
        'source_ply_sha256': model['sha256'], 'native_metrics_sha256': sha(metrics_path),
        'native_step': 11999, 'sh_degree': 3, 'background_rgb': background,
        'background_basis': {'config_path': str(config), 'config_sha256': sha(config),
            'installed_splatfacto_sha256': sha(source), 'rule': 'random config in eval returns fixed model.background_color; no override'},
        'gpu': capture['gpu'], 'gpu_lock_held_entire_browser_lifecycle': True, 'browser_closed_before_scoring': True,
        'frames': len(rows), 'viewport': [640, 540], 'pixel_center': .5,
        'renderer': manifest['renderer'], 'metric_domain': 'browser_canvas_unorm8_vs_native_float32',
        'resolved_renderer': capture['rows'][0]['snapshot']['renderer'],
        'compositing': capture['rows'][0]['snapshot'].get('compositing'),
        'source_sha256': capture['source_sha256'], 'lock_identity': capture['lock_identity'],
        'native_vs_gt_psnr_db_mean': float(np.mean([r['native_vs_gt']['psnr_db'] for r in rows])),
        'browser_vs_gt_psnr_db_mean': float(np.mean([r['browser_vs_gt']['psnr_db'] for r in rows])),
        'frames_abs_delta_le_1db': int(sum(abs(r['psnr_delta_db']) <= 1 for r in rows)),
        'mask_protocol': 'fixed ego_v2', 'psnr_delta_db_mean': float(np.mean([r['psnr_delta_db'] for r in rows])),
        'psnr_delta_db_abs_max': float(np.max(np.abs([r['psnr_delta_db'] for r in rows]))),
        'browser_native_direct_psnr_db_mean': float(np.mean([r['browser_vs_native']['psnr_db'] for r in rows])),
        'same_frame_parity_all_abs_delta_le_1db': all(abs(r['psnr_delta_db']) <= 1 for r in rows),
        'ssim_definition': 'CPU skimage default 7x7 uniform SSIM map, masked per-channel mean; not evaluator torch SSIM',
        'rows': rows}
    write(ROOT / 'reports' / (stem + '_result.json'), report)
    print(json.dumps({k: report[k] for k in ('status', 'frames', 'psnr_delta_db_mean', 'psnr_delta_db_abs_max', 'same_frame_parity_all_abs_delta_le_1db')}), flush=True)


if __name__ == '__main__':
    main()

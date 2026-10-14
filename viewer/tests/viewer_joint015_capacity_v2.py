#!/usr/bin/env python3
"""One final joint015 frame, exclusive GPU ownership, bounded browser lifetime."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = 'viewer_joint015_capacity_v2'
OUT = ROOT / 'reports' / STEM
MANIFEST = 'maps/garage_fidelity_v2_joint015_diagnostic/manifest.json'
METRICS = 'runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/metrics.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def owned_processes(owner):
    records = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            records[int(p.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    owned = {owner}
    while True:
        extra = {pid for pid, (parent, _, _) in records.items() if parent in owned}
        if extra <= owned:
            break
        owned.update(extra)
    return {pid: records[pid][1] for pid in owned if pid in records}


def live_same_process(pid, start):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return fields[19] == start and fields[0] != 'Z'
    except (OSError, IndexError):
        return False


def sample_memory(elapsed):
    proc = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
    return {'elapsed_s': elapsed, 'returncode': proc.returncode, 'gpu_memory_csv': proc.stdout.strip(),
            'stderr': proc.stderr.strip()}


def psnr(a, b, mask):
    mse = float(np.square(a.astype(np.float64) - b.astype(np.float64))[mask].mean())
    return {'mse': mse, 'psnr_db': float(-10 * np.log10(max(mse, 1e-20)))}


def main():
    assert os.environ.get('GARAGE_PARITY_MAIN_GPU_RELEASE') == '1'
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'browser_capture.json').exists(), 'Do not repeat the authorized single capture.'
    manifest = json.loads((ROOT / MANIFEST).read_text())
    metrics = json.loads((ROOT / METRICS).read_text())
    frame = next(row for row in metrics['frames'] if row['dataset_index'] == 0)
    model = manifest['source_export']
    assert sha(ROOT / model['path']) == model['sha256']
    assert model['gaussians'] == 2498837 and metrics['gaussians'] == 2498842
    assert metrics['step'] == metrics['inference_model_step'] == 14999
    assert metrics['inference_sh_degree'] == 3 and not metrics.get('direction_support')
    native_path = (ROOT / METRICS).parent / '000_float.npz'
    assert native_path.is_file()
    config = Path(metrics['config'])
    source = ROOT / '.venv-train/lib/python3.10/site-packages/nerfstudio/models/splatfacto.py'
    assert 'background_color: random' in config.read_text()
    assert '[0.1490, 0.1647, 0.2157]' in source.read_text()
    background = np.asarray([.1490, .1647, .2157], dtype=np.float32).tolist()
    prior = json.loads((ROOT / 'reports/viewer_parity_v2_aa015_full/browser_capture.json').read_text())
    for filename, expected in prior['source_sha256'].items():
        assert sha(ROOT / filename) == expected, 'Keep the measured renderer frozen.'
    record = {'dataset_index': 0, 'camera': frame['camera'], 'source_ply_sha256': model['sha256'],
              'direction_support_sha256': None, 'inference_sh_degree': 3, 'background_rgb': background}
    job = {'manifest': MANIFEST, 'block_id': 'block_015', 'output_directory': str(OUT.relative_to(ROOT)),
           'direction_smoke': False, 'frames': [record], 'server_limits': manifest['recommended_server_limits']}
    job_path = ROOT / 'reports' / (STEM + '_job.json')
    write(job_path, job)
    samples, tracked = [], {}
    with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write(OUT / 'deferred.json', {'status': 'deferred_gpu_lock_busy', 'browser_started': False})
            print('DEFERRED: GPU lock busy; no browser started.', flush=True)
            return
        print('Exclusive GPU lock acquired; starting the one-frame 120s capacity check.', flush=True)
        start = time.monotonic()
        samples.append(sample_memory(0))
        proc = subprocess.Popen(['node', 'viewer/tests/viewer_parity_v2_browser.mjs', str(job_path)],
            cwd=ROOT, start_new_session=True,
            env=dict(os.environ, GARAGE_PARITY_GPU_LOCK_HELD='1', PYTHONDONTWRITEBYTECODE='1'))
        timed_out = False
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                tracked.update(owned_processes(proc.pid))
                if elapsed >= 110:
                    timed_out = True
                    proc.send_signal(signal.SIGTERM)
                    break
                samples.append(sample_memory(elapsed))
                time.sleep(.5)
            if timed_out:
                try:
                    proc.wait(timeout=max(.01, 118 - (time.monotonic() - start)))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Include detached Chromium descendants; never release the lock with survivors.
            tracked.update(owned_processes(proc.pid))
            survivors = {pid: stamp for pid, stamp in tracked.items() if live_same_process(pid, stamp)}
            for pid, stamp in survivors.items():
                if live_same_process(pid, stamp):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            proc.wait(timeout=3)
            for _ in range(30):
                if not any(live_same_process(pid, stamp) for pid, stamp in tracked.items()):
                    break
                time.sleep(.05)
            assert not any(live_same_process(pid, stamp) for pid, stamp in tracked.items()), 'GPU descendants survived cleanup'
            samples.append(sample_memory(time.monotonic() - start))
            write(OUT / 'memory_samples.json', {'samples': samples, 'interval_note': '~0.5s plus nvidia-smi latency; sampled board usage, not allocator peak',
                'timed_out': timed_out, 'browser_returncode': proc.returncode,
                'lifecycle_s': time.monotonic() - start, 'all_owned_processes_closed': True,
                'lock_owner_pid': os.getpid(), 'gpu_lock_held_until_browser_and_server_closed': True})
            fcntl.flock(lock, fcntl.LOCK_UN)
        assert not timed_out and proc.returncode == 0, 'Bounded browser attempt failed; see log and memory_samples.json'
    capture = json.loads((OUT / 'browser_capture.json').read_text())
    assert len(capture['rows']) == 1 and capture['errors'] == []
    row = capture['rows'][0]; snapshot = row['snapshot']
    assert snapshot['renderer']['mode'] == 'inria_2dgs_v3'
    assert snapshot['viewport'] == [640, 540] and snapshot['frame']['inference_sh_degree'] == 3
    assert snapshot['frame']['background_rgb'] == background
    assert len(snapshot['models']) == 1 and snapshot['models'][0]['count'] == 2498837
    assert snapshot['models'][0]['plySha256'] == model['sha256']
    pose = np.eye(4); pose[:3] = frame['camera']['camera_to_worlds_ns']
    assert np.array_equal(np.asarray(snapshot['camera_to_world_column_major']).reshape(4, 4).T, pose)
    camera = frame['camera']; near, far = .2, 100.
    projection = np.array([[2 * camera['fl_x'] / camera['w'], 0, 1 - 2 * camera['cx'] / camera['w'], 0],
        [0, 2 * camera['fl_y'] / camera['h'], 2 * camera['cy'] / camera['h'] - 1, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
    assert snapshot['projection_column_major'] == projection.flatten(order='F').tolist()
    assert sha(OUT / row['file']) == row['sha256']
    browser = cv2.cvtColor(cv2.imread(str(OUT / row['file'])), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    with np.load(native_path, allow_pickle=False) as data:
        native = data['rgb'].astype(np.float64)
    gt = cv2.cvtColor(cv2.imread(frame['source_image_path']), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    mask = cv2.imread(frame['acceptance_mask_path'], cv2.IMREAD_GRAYSCALE) > 0
    native_gt, browser_gt = psnr(native, gt, mask), psnr(browser, gt, mask)
    delta = browser_gt['psnr_db'] - native_gt['psnr_db']
    used_mib = [int(s['gpu_memory_csv'].splitlines()[0].split(',')[-1]) for s in samples if s['returncode'] == 0]
    result = {'status': 'CAPACITY_CAPTURE_PASS', 'high_fidelity_accepted': False, 'frames': 1,
        'manifest': MANIFEST, 'job': str(job_path.relative_to(ROOT)), 'gpu': capture['gpu'],
        'loaded_gaussians': 2498837, 'native_checkpoint_gaussians': 2498842,
        'export_difference': 'Existing exporter removed 5 below-opacity-threshold Gaussians; no new modification.',
        'source_ply_sha256': model['sha256'], 'native_metrics_sha256': sha(ROOT / METRICS),
        'native_float_sha256': sha(native_path), 'native_step': 14999, 'SH': 3,
        'matching_camera_source_SH_background': True, 'viewport': [640, 540],
        'js_shader_console_errors': capture['errors'], 'gl_error_scope': 'Browser page/console shader errors; gl.getError not separately instrumented.',
        'native_vs_gt': native_gt, 'browser_vs_gt': browser_gt,
        'browser_vs_native_masked': psnr(browser, native, mask),
        'browser_vs_native_full_image': psnr(browser, native, np.ones(mask.shape, dtype=bool)),
        'psnr_delta_db': delta, 'single_frame_abs_delta_le_1db': bool(abs(delta) <= 1),
        'sampled_board_memory_max_mib': max(used_mib), 'sampled_board_memory_start_mib': used_mib[0],
        'gpu_lock_held_entire_browser_lifecycle': True, 'browser_and_server_closed': True,
        'source_sha256': capture['source_sha256'],
        'scope': 'One final joint015 frame capacity/parity smoke, not full20/full-scene or geometric acceptance.'}
    write(ROOT / 'reports' / (STEM + '_result.json'), result)
    print(json.dumps({key: result[key] for key in ('status', 'loaded_gaussians', 'psnr_delta_db', 'sampled_board_memory_max_mib')}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bounded overview/F6 smoke; owns only its browser, never the user servers."""
import fcntl
import json
import os
import signal
import subprocess
import time
from urllib.request import urlopen

from viewer_joint015_capacity_v2 import ROOT, owned_processes, live_same_process, sample_memory, write

REPORT = ROOT / 'reports/full_preview_v2_viewer.json'


def main():
    assert os.environ.get('GARAGE_FULL_PREVIEW_GPU_READY') == '1'
    data = json.loads(REPORT.read_text())
    assert data['cpu']['status'] == 'PASS'
    assert data['gpu']['status'] == 'pending', 'Do not repeat a completed browser smoke.'
    samples, tracked = [], {}
    with (ROOT / 'runtime/core_joint_gpu.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('DEFERRED: GPU lock busy. No browser launched.', flush=True)
            return
        start = time.monotonic()
        print('Shared GPU lock held; connecting existing 8769/8770 user servers.', flush=True)
        samples.append(sample_memory(0))
        proc = subprocess.Popen(['node', 'viewer/tests/full_preview_v2_browser.mjs'], cwd=ROOT,
            start_new_session=True, env=dict(os.environ, GARAGE_FULL_PREVIEW_LOCK_HELD='1', PYTHONDONTWRITEBYTECODE='1'))
        timed_out = False
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                tracked.update(owned_processes(proc.pid))
                if elapsed > 110:
                    timed_out = True
                    proc.send_signal(signal.SIGTERM)
                    break
                samples.append(sample_memory(elapsed))
                time.sleep(.5)
            if timed_out:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            tracked.update(owned_processes(proc.pid))
            # Process identity checks protect unrelated/user Chromium instances.
            for pid, stamp in tracked.items():
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
            assert not any(live_same_process(pid, stamp) for pid, stamp in tracked.items())
            samples.append(sample_memory(time.monotonic() - start))
            data = json.loads(REPORT.read_text())
            data['gpu']['lifecycle'] = {
                'lock_owner_pid': os.getpid(), 'lock': 'runtime/core_joint_gpu.lock',
                'held_entire_browser_lifecycle': True, 'all_owned_processes_closed': True,
                'user_servers_started_or_stopped': False, 'existing_user_browsers_touched': False,
                'timed_out': timed_out, 'returncode': proc.returncode,
                'elapsed_s': time.monotonic() - start, 'memory_samples': samples,
                'memory_scope': 'Sampled whole-board usage includes any user browser; not allocator peak.'}
            if timed_out or proc.returncode:
                data['gpu']['status'] = 'FAILED'
            write(REPORT, data)
            fcntl.flock(lock, fcntl.LOCK_UN)
    data = json.loads(REPORT.read_text())
    data['gpu']['lifecycle']['lock_released'] = True
    data['user_servers_after_test'] = {str(p): urlopen(f'http://127.0.0.1:{p}/', timeout=10).status for p in (8767, 8768, 8769, 8770)}
    data['status'] = 'VIEWER_SMOKE_PASS' if data['gpu']['status'] == 'PASS' else 'VIEWER_SMOKE_FAILED'
    write(REPORT, data)
    print(json.dumps({'status': data['status'], 'lifecycle_s': data['gpu']['lifecycle']['elapsed_s'],
        'user_servers_after_test': data['user_servers_after_test']}), flush=True)
    assert data['status'] == 'VIEWER_SMOKE_PASS'


if __name__ == '__main__':
    main()

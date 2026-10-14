#!/usr/bin/env python3
"""CPU-only checks of the two explicit v3 delivery manifests and HTTP assets."""
import hashlib
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from serve_map import LIMITS, MapStore, make_server

ORIGINAL = 'maps/garage_fidelity_v2_viewer/manifest.json'
JOINT = 'maps/garage_fidelity_v2_joint015_diagnostic/manifest.json'
JOINT_LIMITS = {'max_splats': 2500000, 'max_file_bytes': 671088640, 'max_loaded_blocks': 1}


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_http(manifest, limits=None):
    server = make_server(root=ROOT, manifest=manifest, port=0, limits=limits)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:%d' % server.server_port
    try:
        with urlopen(base + '/api/manifest', timeout=10) as response:
            data = json.load(response)
        assert data['renderer']['mode'] == 'inria_2dgs_v3'
        assert data['high_fidelity_accepted'] is False
        rows = []
        for block in data['blocks']:
            asset = block['asset']
            assert asset['available'], (block['id'], asset)
            assert not block.get('lidar') and not block.get('direction_support')
            with urlopen(Request(base + asset['url'], method='HEAD'), timeout=10) as response:
                assert response.status == 200
                assert int(response.headers['Content-Length']) == asset['file_bytes']
            try:
                urlopen(base + '/api/blocks/' + block['id'] + '/lidar', timeout=10)
                raise AssertionError('LiDAR unexpectedly exposed')
            except HTTPError as error:
                assert error.code == 404
            rows.append({'id': block['id'], 'path': block['path'], 'asset': asset,
                         'camera_views': len(block['camera_views']), 'lidar_http': 404})
        for route in ('/', '/app.js', '/gaussian_direction_support.js', '/vendor/spark.js'):
            with urlopen(Request(base + route, method='HEAD'), timeout=10) as response:
                assert response.status == 200
        with urlopen(Request(base + '/api/blocks/block_015/model',
                             headers={'Range': 'bytes=0-31'}), timeout=10) as response:
            assert response.status == 206 and response.read().startswith(b'ply\n')
        return {'manifest': manifest, 'sha256': sha(manifest), 'limits': data['limits'],
                'available_models': len(rows), 'rows': rows,
                'static_routes_passed': True, 'model_range_206': True}
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def main():
    inventory = read('reports/viewer_parity_v2_delivery_source_inventory.json')
    original = read(inventory['source_manifest']['path'])
    assert sha(inventory['source_manifest']['path']) == inventory['source_manifest']['sha256']
    new = read(ORIGINAL)
    assert len(original['blocks']) == len(new['blocks']) == 28
    for old, delivered, source in zip(original['blocks'], new['blocks'], inventory['sources']):
        assert {k: delivered[k] for k in old} == old  # All old block fields, including TEST scores.
        assert sha(source['path']) == source['sha256']
    for key in ('coordinates', 'units', 'camera_up', 'floor_assignment', 'calibration_verified'):
        assert new[key] == original[key]
    assert not any(key.startswith('test_') for key in new['summary'])
    assert new['historical_summary']['values'] == original['summary']
    joint = read(JOINT)
    export = joint['source_export']
    assert sha(export['path']) == export['sha256']
    assert export['gaussians'] == 2498837
    assert joint['export_filter']['checkpoint_gaussians'] == 2498842
    default_limits = dict(LIMITS)
    assert not MapStore(ROOT, JOINT).public()['blocks'][0]['asset']['available']
    outputs = [check_http(ORIGINAL), check_http(JOINT, JOINT_LIMITS)]
    assert outputs[1]['rows'][0]['asset']['vertex_count'] == export['gaussians']
    assert LIMITS == default_limits == MapStore(ROOT, 'maps/garage_manifest.json').public()['limits']
    assert 'renderer' not in original
    evidence = read('reports/viewer_parity_v2_profile_adoption_proposal.json')
    for code in evidence['candidate_code']:
        assert sha(code['path']) == code['sha256']
    report = {'status': 'PASS', 'cpu_only': True, 'browser_started': False,
              'original_28_source_hashes_match_DEV408': True,
              'all_original_block_fields_preserved': True, 'legacy_manifest_unchanged': True,
              'legacy_budget_defaults_unchanged': True, 'joint_requires_explicit_budget': True,
              'render_code_matches_passed_original015_full20': True,
              'no_lidar_or_direction_sidecar_exposed': True, 'stores': outputs}
    (ROOT / 'reports/viewer_parity_v2_delivery_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('PASS: 28 original + 1 joint model, HTTP HEAD/Range/static routes; no GPU.')


if __name__ == '__main__':
    main()

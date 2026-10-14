#!/usr/bin/env python3
"""Bounded stdlib HTTP checks. Default is BEFORE deletion, never a GPU test.

python3 -B reports/cleanup_v2_postcheck.py --phase pre-delete
After main confirms deletion: use --phase post-delete. Both runs are retained.
Only cleanup_v2_viewer_postcheck.json is written; no model body SHA is computed.
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/cleanup_v2_viewer_dependencies.json'
OUTPUT = ROOT / 'reports/cleanup_v2_viewer_postcheck.json'
LANDING = 'http://127.0.0.1:8767/reports/full_preview_v2/'
GALLERY = 'http://127.0.0.1:8767/maps/garage_fidelity_v2_full_preview/evaluation/index.html'
MAX_SMALL = 4 * 1024**2


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def small_file(path):
    require(path.stat().st_size <= MAX_SMALL, 'Refusing large full read: ' + str(path))
    return path.read_bytes()


def request(url, method='GET', headers=None, limit=MAX_SMALL):
    parts = urlsplit(url)
    require(parts.hostname == '127.0.0.1' and parts.port in (8767, 8768, 8769, 8770), 'Unexpected origin: ' + url)
    try:
        response = urlopen(Request(url, method=method, headers=headers or {}), timeout=10)
    except URLError as exc:
        raise RuntimeError(f'{method} {url}: {exc}') from exc
    with response:
        meta = dict(response.headers)
        if method == 'HEAD':
            body = b''
        else:
            require(int(response.headers.get('Content-Length', limit + 1)) <= limit, 'Refusing large response: ' + url)
            body = response.read(limit + 1)
            require(len(body) <= limit, 'HTTP read exceeded bound')
        return response.status, meta, body


def static_disk_path(url):
    parts = urlsplit(url)
    require(parts.port == 8767, 'Not the static file service')
    path = ROOT / unquote(parts.path).lstrip('/')
    if parts.path.endswith('/'):
        path /= 'index.html'
    require(path.resolve().is_relative_to(ROOT), 'Static path escaped project')
    return path


def file_head(url):
    code, headers, _ = request(url, 'HEAD')
    require(code == 200, 'HEAD failed: ' + url)
    row = {'url': url, 'status': code, 'bytes': int(headers['Content-Length'])}
    if urlsplit(url).port == 8767:
        path = static_disk_path(url)
        require(path.is_file() and path.stat().st_size == row['bytes'], 'HTTP/disk size mismatch: ' + url)
        row['path'] = str(path.relative_to(ROOT))
    return row


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.images = [], []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src') and value:
                self.links.append(value)
            if tag == 'img' and name == 'src':
                self.images.append(value)


def html_page(url):
    code, _, raw = request(url)
    require(code == 200 and raw == small_file(static_disk_path(url)), 'HTML HTTP/disk mismatch')
    parser = Links()
    parser.feed(raw.decode('utf-8'))
    return parser, raw


def check_landing():
    parsed, raw = html_page(LANDING)
    urls = {urljoin(LANDING, link) for link in parsed.links}
    urls.update(urljoin(LANDING, name) for name in ('overview.png', 'overview_side.png', 'f6_block_027.png', '../full_preview_v2_viewer.json'))
    rows = [file_head(url) for url in sorted(urls)]
    for name in ('overview.png', 'overview_side.png', 'f6_block_027.png'):
        with (ROOT / 'reports/full_preview_v2' / name).open('rb') as stream:
            require(stream.read(8) == b'\x89PNG\r\n\x1a\n', 'Invalid screenshot header: ' + name)
    evidence = json.loads(request(urljoin(LANDING, '../full_preview_v2_viewer.json'))[2])
    require(evidence['gpu']['status'] == 'PASS', 'Historical viewer evidence unavailable')
    return {'links': rows, 'gallery_link_present': GALLERY in urls or urljoin(GALLERY, './') in urls, 'html_sha256': digest(raw),
            'historical_browser_evidence_only': True}


def check_gallery():
    parsed, raw = html_page(GALLERY)
    images = sorted({urljoin(GALLERY, link) for link in parsed.images})
    require(len(images) == 108, 'Expected 108 actual comparison JPGs')
    groups = Counter(urlsplit(url).path.split('/')[-2] for url in images)
    require(len(groups) == 6, 'Expected six final evaluation groups')
    rows = [file_head(url) for url in sorted({urljoin(GALLERY, link) for link in parsed.links})]
    for url in images:
        require(url.endswith('_comparison.jpg'), 'Unexpected gallery image: ' + url)
        with static_disk_path(url).open('rb') as stream:
            require(stream.read(3) == b'\xff\xd8\xff', 'Invalid JPEG header: ' + url)
    metrics_urls = [r['url'] for r in rows if r['url'].endswith('/metrics.json')]
    require(len(metrics_urls) == 6, 'Expected six gallery metrics files')
    for url in metrics_urls:
        require(isinstance(json.loads(request(url)[2]), dict), 'Invalid gallery metrics')
    return {'url': GALLERY, 'groups': dict(groups), 'images': len(images), 'links': rows,
            'html_sha256': digest(raw), 'image_check': 'HEAD/disk byte lengths and local JPEG magic; no full image/body SHA or quality scoring'}


def ply_header(path):
    require(path.is_file() and not path.is_symlink(), 'PLY unavailable: ' + str(path))
    with path.open('rb') as stream:
        header = bytearray()
        while len(header) < 65536:
            line = stream.readline(4096)
            require(bool(line), 'Incomplete PLY header')
            header.extend(line)
            if line.strip() == b'end_header':
                break
        else:
            raise AssertionError('PLY header exceeds bound')
        text = header.decode('ascii')
        require(text.startswith('ply\nformat binary_little_endian 1.0\n'), 'Unexpected PLY encoding')
        counts = re.findall(r'^element vertex (\d+)$', text, re.M)
        require(len(counts) == 1, 'Expected one PLY vertex element')
        properties = re.findall(r'^property (\S+) (\S+)$', text, re.M)
        require(len(properties) == 62 and all(t in ('float', 'float32') for t, _ in properties), 'Expected unchanged 62-float SH3 PLY')
        count, stride = int(counts[0]), 4 * len(properties)
        require(path.stat().st_size == len(header) + count * stride, 'PLY count/length mismatch')
        first = stream.read(stride)
        require(len(first) == stride, 'Missing first record')
    return count, bytes(header) + first, len(header)


def check_models(audit):
    references = json.loads(small_file(ROOT / 'maps/garage_reference_views.json'))['blocks']
    mapping = {m['id']: m for m in audit['migration']['mapping']}
    reports = []
    for service in audit['runtime']['live_services']:
        port, name = service['port'], service['manifest']['path']
        local_raw = small_file(ROOT / name)
        local = json.loads(local_raw)
        public = json.loads(request(f'http://127.0.0.1:{port}/api/manifest')[2])
        require(public['renderer']['mode'] == 'inria_2dgs_v3' and public['high_fidelity_accepted'] is False, 'Renderer/quality identity changed')
        require(public['limits'] == service['limits'], 'Server limits changed')
        expected_ids = ['garage_overview'] if port == 8769 else ['block_000'] if port == 8768 else list(mapping)
        require([b['id'] for b in public['blocks']] == expected_ids, 'Wrong manifest membership')
        require([b['id'] for b in local['blocks']] == expected_ids, 'Local/API membership mismatch')
        rows = []
        for b, disk in zip(public['blocks'], local['blocks']):
            for key in ('id', 'path', 'sha256', 'training', 'bbox_m'):
                require(b[key] == disk[key], 'API/local manifest mismatch: ' + key)
            expected = audit['runtime']['model_keep']['overview'] if port == 8769 else mapping[b['id']]
            path = expected['path'] if port == 8769 else expected['to']
            require(b['path'] == path and b['sha256'] == expected['sha256'], 'Moved model identity mismatch')
            require(b['asset']['available'] is True, 'Model not available')
            count, prefix, header_bytes = ply_header(ROOT / path)
            require(count == expected['gaussians'] == b['asset']['vertex_count'] == b['training']['gaussians'], 'Gaussian count mismatch')
            require(b['asset']['file_bytes'] == (ROOT / path).stat().st_size, 'API PLY size mismatch')
            url = f'http://127.0.0.1:{port}' + b['asset']['url']
            head = file_head(url)
            require(head['bytes'] == b['asset']['file_bytes'], 'HEAD size mismatch')
            end = len(prefix) - 1
            code, headers, body = request(url, headers={'Range': f'bytes=0-{end}'}, limit=len(prefix))
            require(code == 206 and headers['Content-Range'] == f'bytes 0-{end}/{head["bytes"]}' and body == prefix, 'HTTP/disk PLY header/first-record mismatch')
            views = b.get('camera_views', [])
            if port == 8769:
                require(views == [], 'Overview should have no invented reference camera')
            else:
                require(len(views) == 4 and views == references[b['id']]['views'], 'Four reference cameras differ')
                require({v['camera_id'] for v in views} == {'camera_2', 'camera_3', 'camera_4', 'camera_5'}, 'Four-camera membership mismatch')
            rows.append(dict(head, id=b['id'], path=path, gaussians=count, range_status=206,
                header_bytes=header_bytes, bytes_downloaded=len(body), header_and_first_record_sha256=digest(body),
                declared_ply_sha256=b['sha256'], full_ply_sha_recomputed=False, reference_views=len(views)))
        total = sum(row['gaussians'] for row in rows)
        require(total == {8768: 2495734, 8769: 1200000, 8770: 13681252}[port], 'Total Gaussian count mismatch')
        if port == 8770:
            require(sorted(x for b in public['blocks'] for x in b['source_blocks']) == [f'block_{i:03d}' for i in range(28)], '28-region coverage mismatch')
        reports.append({'port': port, 'manifest': name, 'manifest_sha256': digest(local_raw), 'models': rows, 'total_gaussians': total})
    return reports


def check_static(audit):
    pins = {r['path']: r for r in audit['runtime']['runtime_file_descriptors']}
    routes = {r['path']: urlsplit(r['url']).path for r in audit['runtime']['static_http']}
    require(len(routes) == 11, 'Expected 11 unique viewer static files')
    rows = []
    for path, route in routes.items():
        expected = pins[path]['sha256']
        require(digest(small_file(ROOT / path)) == expected, 'Static source changed: ' + path)
        for port in (8768, 8769, 8770):
            code, _, body = request(f'http://127.0.0.1:{port}' + route)
            require(code == 200 and digest(body) == expected, 'HTTP static source changed')
        rows.append({'path': path, 'sha256': expected, 'ports_passed': [8768, 8769, 8770]})
    for path in ('scripts/serve_map.py', 'maps/garage_reference_views.json'):
        require(digest(small_file(ROOT / path)) == pins[path]['sha256'], 'Server/reference source changed: ' + path)
    return {'files': rows, 'unique_static_files': 11, 'server_and_reference_source_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pre-delete', 'post-delete'), default='pre-delete')
    args = parser.parse_args()
    start = time.monotonic()
    audit_raw = small_file(AUDIT)
    audit = json.loads(audit_raw)
    run = {'phase': args.phase, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'deletion_completed_asserted_by_caller': args.phase == 'post-delete', 'checks': {}, 'errors': [],
           'audit_sha256': digest(audit_raw), 'script_sha256': digest(small_file(Path(__file__))),
           'scope': 'CPU/HTTP only; no GPU/browser/server lifecycle changes; no large-model download/full SHA; no new quality or native parity claim.'}
    for name, fn in [('landing', check_landing), ('gallery', check_gallery), ('models', lambda: check_models(audit)), ('static', lambda: check_static(audit))]:
        try:
            run['checks'][name] = {'status': 'PASS', 'result': fn()}
        except Exception as exc:
            run['errors'].append({'check': name, 'error': str(exc)})
            run['checks'][name] = {'status': 'FAIL'}
    prefix = 'PRE_DELETE_BASELINE' if args.phase == 'pre-delete' else 'POST_DELETE_HTTP'
    run['status'] = prefix + ('_FAIL' if run['errors'] else '_PASS')
    run['elapsed_s'] = time.monotonic() - start
    record = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {'schema_version': 1, 'runs': []}
    record['runs'].append(run)
    record['status'], record['latest_phase'] = run['status'], args.phase
    record['post_delete_verified'] = args.phase == 'post-delete' and not run['errors']
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': run['status'], 'elapsed_s': run['elapsed_s'], 'errors': run['errors']}, ensure_ascii=False))
    return bool(run['errors'])


if __name__ == '__main__':
    raise SystemExit(main())

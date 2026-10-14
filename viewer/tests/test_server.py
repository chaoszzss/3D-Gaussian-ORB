"""Protocol, PLY discrimination, confined access, and HTTP transfer tests."""
import sys
sys.dont_write_bytecode = True
import copy
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest

VIEWER = Path(__file__).resolve().parents[1]
PROJECT = VIEWER.parent
spec = importlib.util.spec_from_file_location("serve_map", PROJECT / "scripts/serve_map.py")
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
M = server_module


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        tmp = VIEWER / "runtime" / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="server-tests-", dir=str(tmp))
        self.root = Path(self.temp.name)
        self.data = json.loads((VIEWER / "fixtures/synthetic_manifest.json").read_text())
        for block in self.data["blocks"]:
            path = self.root / block["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((PROJECT / block["path"]).read_bytes())
        lidar = self.data["blocks"][0]["lidar"]["path"]
        (self.root / lidar).write_bytes((PROJECT / lidar).read_bytes())
        self.manifest = self.root / "manifest.json"
        self.publish()

    def tearDown(self):
        self.temp.cleanup()

    def publish(self):
        self.manifest.write_text(json.dumps(self.data))

    def store(self):
        return M.MapStore(self.root, "manifest.json")

    def test_real_missing_does_not_fall_back_to_synthetic(self):
        data = M.MapStore(self.root, "missing.json").public()
        self.assertEqual(data["dataset_kind"], "real")
        self.assertEqual(data["blocks"], [])
        self.assertTrue(data["manifest_missing"])

    def test_preflight_degrees_and_world_coordinates(self):
        data = self.store().public()
        self.assertEqual([b["asset"]["sh_degree"] for b in data["blocks"]], [3, 1, 2, 0])
        self.assertTrue(all(b["asset"]["available"] for b in data["blocks"]))
        self.assertEqual(data["blocks"][1]["bbox_m"]["min"][0], 5.6)
        self.assertEqual(data["blocks"][3]["floor_id"], None)
        self.assertEqual(data["blocks"][0]["lidar"]["asset"]["vertex_count"], 81)

    def test_process_budget_enforced_and_isolated_from_manifest_and_other_servers(self):
        path = self.data['blocks'][0]['path']
        info = self.store().preflight(path)
        self.data['limits'] = {'max_splats': 6000000, 'max_file_bytes': 2 * 1024**3}
        self.publish()
        limited = M.MapStore(self.root, 'manifest.json', limits={'max_splats': info['vertex_count'] - 1})
        self.assertFalse(limited.public()['blocks'][0]['asset']['available'])
        with self.assertRaisesRegex(M.MapError, '数量'):
            limited.asset(self.data['blocks'][0]['id'], 'model')
        allowed = M.MapStore(self.root, 'manifest.json', limits={
            'max_splats': info['vertex_count'], 'max_file_bytes': info['file_bytes'], 'max_loaded_blocks': 1})
        self.assertEqual(allowed.asset(self.data['blocks'][0]['id'], 'model'), path)
        public = allowed.public()
        self.assertEqual(public['limits']['max_loaded_blocks'], 1)
        public['limits']['max_splats'] = 0
        self.assertEqual(allowed.preflight(path), info)
        self.assertEqual(self.store().public()['limits'], M.LIMITS)
        self.assertEqual(M.LIMITS['max_splats'], 1500000)

    def test_explicit_file_budget_still_rejects_oversize_asset(self):
        path = self.data['blocks'][0]['path']
        info = self.store().preflight(path)
        limited = M.MapStore(self.root, 'manifest.json', limits={'max_file_bytes': info['file_bytes'] - 1})
        with self.assertRaisesRegex(M.MapError, '文件'):
            limited.preflight(path)

    def test_invalid_process_budgets_fail_before_server_bind(self):
        for limits in ({'max_splats': True}, {'max_splats': 0}, {'max_splats': 6000001},
                       {'max_file_bytes': 2 * 1024**3 + 1}, {'max_loaded_blocks': 0}, {'unknown': 1}):
            with self.subTest(limits=limits), self.assertRaises(M.MapError):
                M.make_server(self.root, 'manifest.json', port=0, limits=limits)

    def test_direction_support_is_bound_to_manifest_model_and_digest(self):
        block = self.data['blocks'][0]
        ply = self.root / block['path']
        info = self.store().preflight(block['path'])
        array = self.root / 'flags.npy'
        array.write_bytes(bytes(info['vertex_count']))
        meta = dict(schema_version=1,
                    ply=dict(path=block['path'], sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
                             gaussians=info['vertex_count']),
                    side_flags=dict(path='flags.npy', sha256=hashlib.sha256(array.read_bytes()).hexdigest(),
                                    dtype='uint8', count=info['vertex_count']))
        path = self.root / 'direction.json'
        path.write_text(json.dumps(meta))
        block['direction_support'] = dict(path='direction.json', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.publish()
        store = self.store()
        self.assertEqual(store.direction_asset(block['id']), 'direction.json')
        self.assertEqual(store.direction_asset(block['id'], True), 'flags.npy')
        self.assertEqual(store.public()['blocks'][0]['direction_support']['url'],
                         '/api/blocks/' + block['id'] + '/direction_support')
        original = ply.read_bytes()
        ply.write_bytes(original[:-4] + struct.pack('<f', 0.123))
        with self.assertRaisesRegex(M.MapError, 'PLY哈希'):
            store.direction_asset(block['id'])
        ply.write_bytes(original)
        array.write_bytes(array.read_bytes() + b'1')
        with self.assertRaisesRegex(M.MapError, '数组哈希'):
            store.direction_asset(block['id'], True)

    def test_pointcloud_cannot_be_gaussian(self):
        b = self.data["blocks"][0]
        b["path"] = b["lidar"]["path"]
        self.publish()
        data = self.store().public()
        self.assertFalse(data["blocks"][0]["asset"]["available"])
        self.assertIn("普通点云", data["blocks"][0]["asset"]["reason"])

    def test_gaussian_ascii_is_rejected_before_renderer(self):
        with (VIEWER / "fixtures/synthetic_ascii_unsupported.ply").open("rb") as f:
            with self.assertRaisesRegex(M.MapError, "小端二进制"):
                M.inspect_ply(f)

    def test_truncated_and_nonfinite_models(self):
        path = self.root / self.data["blocks"][0]["path"]
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        self.assertFalse(self.store().public()["blocks"][0]["asset"]["available"])
        offset = original.index(b"end_header\n") + len(b"end_header\n")
        path.write_bytes(original[:offset] + struct.pack("<f", float("nan")) + original[offset + 4:])
        self.assertFalse(self.store().public()["blocks"][0]["asset"]["available"])

    def test_schema_rejects_incompatible_spaces(self):
        for mutate in (lambda d: d.update(units="mm"), lambda d: d.update(coordinates="map"),
                       lambda d: d.update(camera_up=[0, 0, 0]),
                       lambda d: d["blocks"][0].update(coordinates="local"),
                       lambda d: d["blocks"][0].update(path="../outside.ply"),
                       lambda d: d["blocks"][0].update(path="/etc/passwd"),
                       lambda d: d["blocks"][0].update(id=123),
                       lambda d: d["blocks"][0]["lidar"].update(alignment_verified=False),
                       lambda d: d["blocks"].append(copy.deepcopy(d["blocks"][0]))):
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(M.MapError):
                M.validate_manifest(data)

    def test_symlinks_and_nonregular_files_denied(self):
        (self.root / "escape.ply").symlink_to(PROJECT / "README.md")
        with self.assertRaises(OSError):
            M.open_local(self.root, "escape.ply")
        (self.root / "escaped_directory").symlink_to(PROJECT, target_is_directory=True)
        with self.assertRaises(OSError):
            M.open_local(self.root, "escaped_directory/README.md")
        with self.assertRaises((OSError, M.MapError)):
            M.open_local(self.root, "viewer")

    def test_duplicate_json_keys_and_nan_denied(self):
        self.manifest.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(M.MapError):
            self.store().read()
        self.manifest.write_text('{"foo":NaN}')
        with self.assertRaises(M.MapError):
            self.store().read()

    def test_http_allowlist_ranges_head_and_host(self):
        server = M.make_server(self.root, "manifest.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        def request(path, method="GET", headers=None):
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            return result
        try:
            path = "/api/blocks/synthetic_a/model"
            original = (self.root / self.data["blocks"][0]["path"]).read_bytes()
            code, headers, body = request(path, headers={"Range": "bytes=0-31"})
            self.assertEqual(code, 206)
            self.assertEqual(body, original[:32])
            self.assertIn("Content-Range", headers)
            code, _, body = request(path, headers={"Range": "bytes=-16"})
            self.assertEqual((code, body), (206, original[-16:]))
            code, headers, body = request(path, method="HEAD")
            self.assertEqual((code, body), (200, b""))
            self.assertEqual(int(headers["Content-Length"]), len(original))
            self.assertEqual(request(path, headers={"Range": "bytes=999999999-"})[0], 416)
            self.assertEqual(request(path, headers={"Range": "bytes=0-1,3-4"})[0], 416)
            for bad in ("/README.md", "/viewer/runtime/file", "/%2e%2e/README.md", "/api/blocks/unlisted/model", "/%5cetc/passwd"):
                self.assertIn(request(bad)[0], (404, 422))
            self.assertEqual(request("/api/manifest", headers={"Host": "attacker.example"})[0], 403)
            self.assertEqual(request("/api/manifest", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
            self.assertEqual(request("/api/manifest")[0], 200)
            # Swap an allowlisted model for a symlink: neither manifest nor asset serves it.
            local = self.root / self.data["blocks"][0]["path"]
            local.unlink(); local.symlink_to(PROJECT / "README.md")
            self.assertEqual(request(path)[0], 403)
        finally:
            connection.close(); server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)

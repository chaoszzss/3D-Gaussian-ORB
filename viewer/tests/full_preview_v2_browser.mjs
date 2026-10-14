// Small real-UI smoke; connects existing services, never starts/stops servers.
import { chromium } from 'playwright';
import * as THREE from 'three';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const reportPath = path.join(root, 'reports/full_preview_v2_viewer.json');
const report = JSON.parse(fs.readFileSync(reportPath));
const output = path.join(root, 'reports/full_preview_v2');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
assert.equal(process.env.GARAGE_FULL_PREVIEW_LOCK_HELD, '1');
function verifyLock() {
  const inode = String(fs.statSync(path.join(root, 'runtime/core_joint_gpu.lock'), { bigint: true }).ino);
  assert.ok(fs.readFileSync('/proc/locks', 'utf8').split('\n').some(line => {
    const f = line.trim().split(/\s+/);
    return f[1] === 'FLOCK' && f[3] === 'WRITE' && f[4] === String(process.ppid) && f[5]?.endsWith(`:${inode}`);
  }), 'parent must actually hold shared GPU lock');
}
verifyLock();
let browser;
const errors = [], result = { status: 'pending', errors, screenshots: [], scope: 'Overview and one F6 asset load/UI smoke only; no native parity or geometry acceptance.' };
async function cleanup() { await browser?.close(); browser = undefined; }
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, async () => { await cleanup(); process.exit(130); });
async function settle(page) {
  await page.waitForFunction(() => {
    const s = window.__garageParity.snapshot();
    return !document.querySelector('#refresh').disabled && !s.sorting && !s.sortDirty;
  }, null, { timeout: 20000 });
  await page.waitForTimeout(700);
}
async function openViewer(url) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.route('**/*', route => {
    const u = new URL(route.request().url());
    return u.hostname === '127.0.0.1' && ['8769', '8770'].includes(u.port) ? route.continue() : route.abort();
  });
  await page.goto(`${url}?parity=1`);
  await page.waitForFunction(() => window.__garageParity, null, { timeout: 20000 });
  await page.locator('#helpers').uncheck();
  const gpu = await page.locator('canvas').evaluate(c => {
    const gl = c.getContext('webgl2'), e = gl.getExtension('WEBGL_debug_renderer_info');
    return { renderer: gl.getParameter(e ? e.UNMASKED_RENDERER_WEBGL : gl.RENDERER), version: gl.getParameter(gl.VERSION) };
  });
  assert.match(gpu.renderer, /NVIDIA|GeForce/i);
  assert.doesNotMatch(gpu.renderer, /swiftshader|llvmpipe|software/i);
  result.gpu = gpu;
  return page;
}
async function load(page, b) {
  await page.locator(`[data-block-id="${b.id}"]`).getByRole('button', { name: '加载', exact: true }).click();
  await page.waitForFunction(() => window.__garageParity.snapshot().models.length === 1 && !document.querySelector('#refresh').disabled,
    null, { timeout: 55000 });
  await settle(page);
  const s = await page.evaluate(() => window.__garageParity.snapshot());
  assert.equal(s.frame, null, 'Normal UI camera, no native frame injection');
  assert.equal(s.renderer.mode, 'inria_2dgs_v3');
  assert.equal(s.models[0].id, b.id);
  assert.equal(s.models[0].count, b.gaussians);
  assert.equal(s.models[0].plySha256, b.sha256);
  assert.equal(s.models[0].visible, true);
  return s;
}
function frustum(s, box) {
  const transform = new THREE.Matrix4().fromArray(s.projection_column_major)
    .multiply(new THREE.Matrix4().fromArray(s.camera_to_world_column_major).invert());
  const corners = [];
  for (const x of [box.min[0], box.max[0]]) for (const y of [box.min[1], box.max[1]]) for (const z of [box.min[2], box.max[2]]) {
    const p = new THREE.Vector4(x, y, z, 1).applyMatrix4(transform);
    assert.ok(p.w > 0);
    const ndc = [p.x / p.w, p.y / p.w, p.z / p.w];
    assert.ok(ndc.every(v => v >= -1 && v <= 1), 'Full bbox must fit actual camera frustum');
    corners.push(ndc);
  }
  return { all_full_bbox_corners_in_frustum: true, ndc_corners: corners,
    scope: 'Camera framing only; sampled points are not required to preserve full extrema.' };
}
async function capture(page, name) {
  await settle(page);
  const s = await page.evaluate(() => window.__garageParity.snapshot());
  const gl = await page.locator('canvas').evaluate(c => { const g = c.getContext('webgl2'); return { error: g.getError(), context_lost: g.isContextLost() }; });
  assert.equal(gl.error, 0); assert.equal(gl.context_lost, false);
  const bytes = await page.screenshot({ path: path.join(output, name) });
  result.screenshots.push({ path: `reports/full_preview_v2/${name}`, sha256: hash(bytes), snapshot: s, gl });
  console.log(`Captured ${name}: ${s.models[0]?.count} Gaussian records.`);
  return s;
}
try {
  browser = await chromium.launch({ headless: true,
    executablePath: path.join(root, 'viewer/runtime/browsers/chromium_headless_shell-1194/chrome-linux/headless_shell'),
    args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist', '--disable-gpu-sandbox'],
    env: { ...process.env, TMPDIR: output, XDG_CACHE_HOME: output, __GL_SHADER_DISK_CACHE: '0',
      __NV_PRIME_RENDER_OFFLOAD: '1', __GLX_VENDOR_LIBRARY_NAME: 'nvidia' } });
  result.browser_version = browser.version();
  let page = await openViewer(report.urls.overview);
  result.overview = await load(page, report.cpu.overview_model);
  await page.locator('[data-view="oblique"]').click();
  result.overview_frustum = frustum(await capture(page, 'overview.png'), report.cpu.full.bbox_m);
  await page.locator('[data-view="side"]').click();
  result.side_frustum = frustum(await capture(page, 'overview_side.png'), report.cpu.full.bbox_m);
  await page.close();
  page = await openViewer(report.urls.detail);
  result.floor_options = await page.locator('#floor option').evaluateAll(nodes => nodes.map(n => ({ value: n.value, label: n.textContent })));
  await page.locator('#floor').selectOption('floor:F6');
  result.f6_listed_ids = await page.locator('[data-block-id]').evaluateAll(nodes => nodes.map(n => n.dataset.blockId));
  assert.deepEqual(result.f6_listed_ids, ['block_025', 'block_026', 'block_027']);
  result.f6 = await load(page, report.cpu.f6_model);
  await page.locator('#reference').click();
  result.f6_reference = await capture(page, 'f6_block_027.png');
  result.f6_reference_selection = await page.locator('#reference-view').inputValue();
  assert.notDeepEqual(result.f6.camera_to_world_column_major, result.f6_reference.camera_to_world_column_major);
  assert.deepEqual(errors, []);
  result.status = 'PASS';
} catch (error) {
  result.status = 'FAILED'; result.failure = String(error); process.exitCode = 1;
  console.error(error);
} finally {
  await cleanup(); verifyLock();
  result.own_browser_closed_before_unlock = true;
  for (const [filename, expected] of Object.entries(report.cpu.renderer_sha256)) assert.equal(hash(fs.readFileSync(path.join(root, filename))), expected);
  report.gpu = result;
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
}

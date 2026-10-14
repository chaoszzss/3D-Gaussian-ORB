// Real browser/GPU-renderer test using only the clearly marked synthetic fixtures.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const viewer = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const project = path.dirname(viewer), output = path.join(viewer, 'runtime', 'verification');
await mkdir(output, { recursive: true });
const env = { ...process.env, PYTHONDONTWRITEBYTECODE: '1', TMPDIR: path.join(viewer, 'runtime', 'tmp'),
  PLAYWRIGHT_BROWSERS_PATH: path.join(viewer, 'runtime', 'browsers'),
  XDG_CACHE_HOME: path.join(viewer, 'runtime', 'cache'),
  CUDA_CACHE_PATH: path.join(viewer, 'runtime', 'cache', 'cuda'),
  __GL_SHADER_DISK_CACHE_PATH: path.join(viewer, 'runtime', 'cache', 'nvidia'), __GL_SHADER_DISK_CACHE: '0' };
process.env.PLAYWRIGHT_BROWSERS_PATH = env.PLAYWRIGHT_BROWSERS_PATH;
process.env.TMPDIR = env.TMPDIR;
const processes = [];
async function start(demo) {
  const proc = spawn('/usr/bin/python3', ['scripts/serve_map.py', '--port', '0', ...(demo ? ['--demo'] : [])], { cwd: project, env });
  processes.push(proc);
  let logs = '';
  proc.stderr.on('data', data => { logs += data.toString(); });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Server start timed out')), 10000);
    proc.once('error', reject);
    proc.stdout.on('data', data => { const match = data.toString().match(/http:\/\/127\.0\.0\.1:\d+\//); if (match) { clearTimeout(timer); resolve(match[0]); } });
    proc.once('exit', code => { clearTimeout(timer); reject(new Error(`Server exit ${code}: ${logs}`)); });
  });
  return { url, logs: () => logs };
}
let browser;
const errors = [], network = [];
try {
  const demo = await start(true), real = await start(false);
  browser = await chromium.launch({ headless: true, env, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('request', req => network.push(req.url()));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.hostname !== '127.0.0.1') return route.abort();
    return route.continue();
  });
  await page.goto(demo.url);
  await page.waitForFunction(() => window.__garageTest);
  await page.locator('#helpers').uncheck();
  await page.locator('#empty').evaluate(n => { n.style.visibility = 'hidden'; }); // Isolate rendered canvas from empty-state text.
  const snap = () => page.evaluate(() => window.__garageTest.snapshot());
  const rasterHash = async () => createHash('sha256').update(await page.locator('canvas').screenshot()).digest('hex');
  const card = id => page.locator(`[data-block-id="${id}"]`);
  const load = async id => {
    await card(id).getByRole('button', { name: '加载', exact: true }).click();
    await page.waitForFunction(id => window.__garageTest.snapshot().meshes.some(m => m.id === id), id, { timeout: 45000 });
    await card(id).getByRole('button', { name: '卸载', exact: true }).waitFor();
    await page.waitForFunction(() => !document.querySelector('#refresh').disabled);
  };
  assert.match(await page.locator('#banner').innerText(), /合成自检模型.*非实景/);
  assert.equal((await snap()).blocks, 0);
  assert.equal(network.filter(n => /\/api\/blocks\//.test(n)).length, 0, 'No automatic model download');
  const emptyRasterHash = await rasterHash();
  await load('synthetic_a');
  await page.waitForTimeout(1400); // Wait for the asynchronous Spark sorter to draw.
  const first = await snap();
  assert.equal(first.meshes[0].count, 882);
  assert.deepEqual(first.meshes[0].position, [0, 0, 0]);
  assert.deepEqual(first.meshes[0].quaternion, [0, 0, 0, 1]);
  assert.deepEqual(first.meshes[0].scale, [1, 1, 1]);
  assert.equal(first.meshes[0].encoding, 'ExtSplats');
  assert.equal(first.meshes[0].maxSh, 3);
  assert.equal(first.meshes[0].shDegree, 3, 'SH3 coefficients are retained by the actual renderer');
  assert.notEqual(await rasterHash(), emptyRasterHash, 'True splats must alter rendered canvas');
  const visibleRasterHash = await rasterHash();
  await page.screenshot({ path: path.join(output, 'synthetic-gaussian.png') });
  await card('synthetic_a').locator('input[data-action="visibility"]').uncheck();
  await page.waitForTimeout(700);
  assert.equal((await snap()).meshes[0].visible, false);
  assert.equal(await rasterHash(), emptyRasterHash, 'Hidden splats no longer render');
  assert.equal((await snap()).blocks, 1, 'Hiding retains allocation');
  await card('synthetic_a').locator('input[data-action="visibility"]').check();
  await load('synthetic_b'); // Binary PLY, SH1, distinct world-space offset.
  assert.equal((await snap()).meshes.find(m => m.id === 'synthetic_b').shDegree, 1);
  assert.ok(Math.abs((await snap()).meshes.find(m => m.id === 'synthetic_b').bbox[0] - 8) < 0.001, 'Decoded splat centers retain world offset');
  await load('synthetic_c'); // Binary SH2, upper floor.
  assert.equal((await snap()).meshes.find(m => m.id === 'synthetic_c').shDegree, 2);
  await card('synthetic_d').getByRole('button', { name: '加载', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('达到预算'));
  assert.equal((await snap()).blocks, 3, 'Fourth block is rejected before allocating');
  await page.selectOption('#floor', 'floor:selftest_lower');
  assert.equal((await snap()).meshes.find(m => m.id === 'synthetic_c').visible, false);
  assert.equal((await snap()).meshes.find(m => m.id === 'synthetic_b').visible, true);
  await page.getByRole('button', { name: '适配已显示区块' }).click();
  const target = (await snap()).target;
  assert.ok(target[0] > 3.5 && target[0] < 4.5, 'Distinct blocks stay in common metric coordinates');
  const angles = {};
  for (const [key, label] of [['top', '俯视'], ['front', '正视'], ['side', '侧视'], ['oblique', '斜视']]) {
    await page.getByRole('button', { name: label, exact: true }).click();
    await page.waitForTimeout(800); angles[key] = await rasterHash();
  }
  assert.notEqual(angles.top, angles.front);
  await card('synthetic_a').getByRole('button', { name: '加载辅层', exact: true }).click();
  await page.waitForFunction(() => window.__garageTest.snapshot().lidar === 1);
  assert.match(await card('synthetic_a').innerText(), /LiDAR 几何辅层/);
  await card('synthetic_a').getByRole('button', { name: '卸载', exact: true }).click();
  assert.equal((await snap()).blocks, 2);
  assert.equal((await snap()).lidar, 0);
  await page.selectOption('#near-count', '1');
  await page.getByRole('button', { name: '加载观察中心附近区块' }).click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('近邻区块'));
  assert.equal((await snap()).blocks, 1, 'Nearest selection replaces working set');
  await page.selectOption('#floor', 'unknown');
  assert.equal((await snap()).meshes.every(m => !m.visible), true);
  await page.selectOption('#floor', 'all');
  await page.getByRole('button', { name: '刷新清单' }).click();
  await page.waitForFunction(() => window.__garageTest.snapshot().blocks === 0);
  await load('synthetic_d'); // SH0, unverified floor remains explicit.
  assert.equal((await snap()).meshes[0].shDegree, 0);
  await page.setViewportSize({ width: 540, height: 920 });
  await page.screenshot({ path: path.join(output, 'synthetic-mobile.png') });
  await page.goto(real.url);
  await page.waitForFunction(() => document.querySelector('#status').textContent !== '正在读取地图清单…');
  assert.equal(await page.evaluate(() => Boolean(window.__garageTest)), false);
  assert.equal(await page.locator('#banner').innerText().then(t => t.includes('合成自检')), false);
  assert.equal(network.filter(n => !/^http:\/\/127\.0\.0\.1:/.test(n)).length, 0, 'Runtime stays offline');
  assert.deepEqual(errors, [], 'Browser has no JS/shader/network errors');
  const report = { status: 'passed', dataset: 'SYNTHETIC SELF-TEST ONLY; NOT REAL GARAGE',
    renderer: 'Spark 2.3.1 / Three.js 0.180.0 / WebGL2 (Chromium SwiftShader)',
    checks: ['SH0–3 binary Gaussian decode', 'actual framebuffer rendering', 'identity metric scene transforms',
      'multi-block world frame', 'floor visibility', 'budget guard', 'hide vs unload', 'four view angles',
      'nearest working set', 'LiDAR auxiliary layer', 'refresh unload', 'mobile layout', 'no auto-load', 'no external requests', 'real mode isolated from fixtures'],
    emptyRasterHash, visibleRasterHash, angles, first, requests: network.length, errors };
  await writeFile(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
} catch (e) {
  console.error('Browser verification failed:', e, errors);
  throw e;
} finally {
  await browser?.close();
  for (const proc of processes) proc.kill('SIGTERM');
}

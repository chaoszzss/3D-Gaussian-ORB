// Launched ONLY by a Python owner of the shared GPU lock (verified below).
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { directionGpuSmoke } from './direction_gpu_smoke.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
assert.equal(process.env.GARAGE_PARITY_GPU_LOCK_HELD, '1', 'use the lock-owning Python runner');
const job = JSON.parse(fs.readFileSync(process.argv[2]));
const serverArgs = ['-B', '-u', 'scripts/serve_map.py', '--port', '0', '--manifest', job.manifest];
if (job.server_limits !== undefined) {
  assert.ok(job.server_limits && typeof job.server_limits === 'object' && !Array.isArray(job.server_limits));
  const allowed = { max_splats: ['--max-splats', 6000000], max_file_bytes: ['--max-file-bytes', 2 * 1024 ** 3],
    max_loaded_blocks: ['--max-loaded-blocks', 28] };
  for (const [key, value] of Object.entries(job.server_limits)) {
    assert.ok(Object.hasOwn(allowed, key), `unsupported server limit: ${key}`);
    assert.ok(Number.isSafeInteger(value) && value > 0 && value <= allowed[key][1], `invalid server limit: ${key}`);
    serverArgs.push(allowed[key][0], String(value));
  }
}
const output = path.resolve(root, job.output_directory);
process.env.TMPDIR = output;
let browser, server, failure;
const errors = [];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const codeHashes = Object.fromEntries(['viewer/app.js', 'viewer/gaussian_direction_support.js',
  'viewer/gaussian_sh_restore.js', 'viewer/node_modules/@sparkjsdev/spark/dist/spark.module.js']
  .map(filename => [filename, sha(fs.readFileSync(path.join(root, filename)))]));
function verifyParentGpuLock() {
  const inode = String(fs.statSync(path.join(root, 'runtime/core_joint_gpu.lock'), { bigint: true }).ino);
  const held = fs.readFileSync('/proc/locks', 'utf8').split('\n').some(line => {
    const fields = line.trim().split(/\s+/);
    return fields[1] === 'FLOCK' && fields[3] === 'WRITE' && fields[4] === String(process.ppid) && fields[5]?.endsWith(`:${inode}`);
  });
  assert.ok(held, 'parent must actually hold the shared exclusive GPU flock');
  return { owner_pid: process.ppid, inode };
}
const gpuLock = verifyParentGpuLock();
async function cleanup() {
  await browser?.close(); browser = undefined;
  if (server && server.exitCode === null) {
    server.kill('SIGTERM');
    await new Promise(resolve => server.once('exit', resolve));
  }
}
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, async () => { await cleanup(); process.exit(130); });
try {
  server = spawn('/usr/bin/python3', serverArgs,
    { cwd: root, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('server timeout')), 15000);
    server.stdout.on('data', bytes => { const match = String(bytes).match(/http:\/\/127\.0\.0\.1:\d+\//); if (match) { clearTimeout(timer); resolve(match[0]); } });
    server.once('error', reject); server.once('exit', code => reject(new Error(`server exit ${code}`)));
  });
  browser = await chromium.launch({ headless: true, dumpio: true,
    executablePath: path.join(root, 'viewer/runtime/browsers/chromium_headless_shell-1194/chrome-linux/headless_shell'),
    args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--enable-logging=stderr'],
    env: { ...process.env, TMPDIR: output, XDG_CACHE_HOME: output, __GL_SHADER_DISK_CACHE: '0',
      __NV_PRIME_RENDER_OFFLOAD: '1', __GLX_VENDOR_LIBRARY_NAME: 'nvidia' } });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error' && !/404/.test(message.text())) errors.push(message.text()); });
  await page.route('**/*', route => new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort());
  await page.goto(`${url}?parity=1`);
  await page.waitForFunction(() => window.__garageParity, null, { timeout: 45000 });
  const gpu = await page.locator('canvas').evaluate(canvas => {
    const gl = canvas.getContext('webgl2'), extension = gl.getExtension('WEBGL_debug_renderer_info');
    return { renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      vendor: extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      version: gl.getParameter(gl.VERSION) };
  });
  console.log(JSON.stringify({ status: 'browser_gpu_started_under_lock', gpu }));
  assert.doesNotMatch(gpu.renderer, /swiftshader|llvmpipe|software/i, 'requires actual GPU renderer');
  assert.match(gpu.renderer, /NVIDIA|GeForce/i, 'requires the allocated NVIDIA GPU');
  const card = page.locator(`[data-block-id="${job.block_id}"]`);
  await card.getByRole('button', { name: '加载', exact: true }).click();
  await page.waitForFunction(() => window.__garageParity.snapshot().models.length === 1 && !document.querySelector('#refresh').disabled,
    null, { timeout: 120000 });
  const rows = [];
  for (const record of job.frames) {
    await page.evaluate(record => window.__garageParity.setNativeFrame(record), record);
    await page.waitForFunction(() => {
      const s = window.__garageParity.snapshot();
      const axis = s.camera_to_world_column_major.slice(8, 11), norm = Math.hypot(...axis);
      return !s.sorting && !s.sortDirty && s.sortedCenter.every((v, i) => Math.abs(v - s.camera_to_world_column_major[12 + i]) < 1e-5) &&
        s.sortedDirection.every((v, i) => Math.abs(v + axis[i] / norm) < 1e-5);
    }, null, { timeout: 30000 });
    // Two consecutive pixel-identical renders catch asynchronous accumulation/sort.
    let previous, stable = false, png;
    for (let attempt = 0; attempt < 40; attempt++) {
      await page.waitForTimeout(100);
      const data = await page.evaluate(() => window.__garageParity.capture());
      png = Buffer.from(data.split(',')[1], 'base64');
      const hash = sha(png);
      if (hash === previous) { stable = true; break; }
      previous = hash;
    }
    assert.ok(stable, 'render did not stabilize');
    assert.deepEqual(errors, [], 'no JS/shader errors');
    const basename = `${String(record.dataset_index).padStart(3, '0')}_browser.png`;
    fs.writeFileSync(path.join(output, basename), png);
    const snapshot = await page.evaluate(() => window.__garageParity.snapshot());
    rows.push({ dataset_index: record.dataset_index, file: basename, sha256: sha(png), stable, snapshot });
    console.log(JSON.stringify({ status: 'frame_captured', dataset_index: record.dataset_index }));
  }
  await page.screenshot({ path: path.join(output, 'browser_diagnostic_window.png') });
  // Save the actual camera snapshots even if an independent shader unit test fails.
  const capture = { status: 'captured_pending_scoring', gpu, gpu_lock: 'runtime/core_joint_gpu.lock',
    lock_identity: gpuLock, held_entire_browser_lifecycle: true, rows, errors,
    browser_version: browser.version(), model_sha256: job.frames[0].source_ply_sha256, source_sha256: codeHashes };
  fs.writeFileSync(path.join(output, 'browser_capture.json'), JSON.stringify(capture, null, 2) + '\n');
  if (job.direction_smoke !== false) {
  const directionSmoke = await directionGpuSmoke(page);
  for (const row of directionSmoke.rows) {
    const name = row.camera_z > 0 ? 'synthetic_direction_front.png' : 'synthetic_direction_back.png';
    fs.writeFileSync(path.join(output, name), Buffer.from(row.png.split(',')[1], 'base64'));
    delete row.png; row.file = name;
  }
  fs.writeFileSync(path.join(output, 'direction_gpu_smoke.json'), JSON.stringify(directionSmoke, null, 2) + '\n');
  console.log(JSON.stringify({ status: 'direction_gpu_smoke', result: directionSmoke.status }));
  assert.equal(directionSmoke.status, 'passed', 'world-direction shader must satisfy synthetic side policy');
  assert.deepEqual(errors, [], 'no shader errors including direction modifier');
  }
} catch (error) {
  failure = { status: 'browser_attempt_failed', error: String(error), console_errors: errors,
    gpu_lock: 'runtime/core_joint_gpu.lock', held_entire_browser_lifecycle: true };
  throw error;
} finally {
  await cleanup();
  verifyParentGpuLock(); // Browser is closed while the SAME flock is still owned.
  if (failure) fs.writeFileSync(path.join(output, `browser_failure_${process.pid}.json`),
    JSON.stringify({ ...failure, browser_and_server_closed: true }, null, 2) + '\n');
  console.log('Browser and local server closed before returning to GPU lock owner.');
}

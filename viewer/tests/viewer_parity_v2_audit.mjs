// CPU only. Run after viewer_parity_v2_audit.py. Real full-bundle artifact audit.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { sideVisible, buildSupportTextureData, parseSideFlagsNpy, validateDirectionMetadata } from '../gaussian_direction_support.js';
const root = new URL('../../', import.meta.url);
const samples = JSON.parse(fs.readFileSync(new URL('reports/viewer_parity_v2_cpu_samples.json', root)));
const metaBytes = fs.readFileSync(new URL(samples.metadata_path, root));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
assert.equal(sha(metaBytes), samples.metadata_sha256);
const meta = validateDirectionMetadata(JSON.parse(metaBytes), samples.gaussians);
const flagsBytes = fs.readFileSync(new URL(meta.side_flags.path, root));
assert.equal(sha(flagsBytes), samples.side_flags_sha256);
const fullFlags = parseSideFlagsNpy(flagsBytes, samples.gaussians), counts = [0, 0, 0, 0];
for (const flag of fullFlags) counts[flag]++;
assert.deepEqual(counts, Object.values(samples.side_counts));
const fields = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
const header = Buffer.from('ply\nformat binary_little_endian 1.0\n' + `element vertex ${samples.indices.length}\n` +
  fields.map(f => `property float ${f}\n`).join('') + 'end_header\n');
const body = Buffer.alloc(samples.indices.length * fields.length * 4);
samples.indices.forEach((index, i) => {
  assert.equal(fullFlags[index], samples.flags[i]);
  [...samples.means[i], ...samples.quaternions_wxyz[i]].forEach((v, j) => body.writeFloatLE(v, 4 * (i * fields.length + j)));
});
const packed = buildSupportTextureData(Buffer.concat([header, body]), Uint8Array.from(samples.flags));
let comparisons = 0, mismatch = 0, float32Mismatch = 0, minBoundaryDistance = Infinity;
const f = Math.fround;
for (const row of samples.native_direction_checks) samples.indices.forEach((_, i) => {
  const normal = [...packed.data.subarray(i * 4, i * 4 + 3)];
  const ray = row.camera_world.map((v, k) => v - samples.means[i][k]);
  const visible = sideVisible(samples.flags[i], normal, ray, meta.render_grazing_cosine);
  mismatch += Number(visible !== row.native_visible[i]); comparisons++;
  // Float32 arithmetic simulation bounds actual shader input risks; GPU dot/rsqrt
  // ordering is not guaranteed identical to this CPU implementation.
  const r = row.camera_world.map((v, k) => f(f(v) - f(samples.means[i][k])));
  const length = f(Math.sqrt(f(f(f(r[0] * r[0]) + f(r[1] * r[1])) + f(r[2] * r[2]))));
  const d = r.map(v => f(v / length));
  const dot = f(f(f(normal[0] * d[0]) + f(normal[1] * d[1])) + f(normal[2] * d[2]));
  const flag = samples.flags[i], grazing = f(meta.render_grazing_cosine);
  const simulated = !((flag === 1 && dot < -grazing) || (flag === 2 && dot > grazing));
  float32Mismatch += Number(simulated !== row.native_visible[i]);
  if (flag === 1 || flag === 2) minBoundaryDistance = Math.min(minBoundaryDistance, Math.abs(Math.abs(dot) - grazing));
});
assert.equal(mismatch, 0, 'JS source-normal decisions versus native Python helper');
assert.equal(float32Mismatch, 0, 'sampled float32 simulation versus native helper');
const sources = ['viewer/app.js', 'viewer/gaussian_direction_support.js', 'viewer/tests/direction_support.mjs',
  'viewer/tests/viewer_parity_v2_audit.py', 'viewer/tests/viewer_parity_v2_audit.mjs',
  'viewer/node_modules/@sparkjsdev/spark/dist/spark.module.js', 'scripts/direction_support.py'];
const report = { status: 'cpu_contract_verified_browser_parity_not_measured', gpu_used: false, webgl_context_created: false,
  spark_version: '2.3.1', three_version: '0.180.0', full_model_gaussians: samples.gaussians,
  metadata_sha256: samples.metadata_sha256, ply_sha256: samples.ply_sha256, side_flags_sha256: samples.side_flags_sha256,
  all_flags_decoded: fullFlags.length, full_flag_histogram: counts,
  sampled_gaussians: samples.indices.length, direction_comparisons: comparisons,
  native_python_vs_js_mismatches: mismatch, float32_simulation_mismatches: float32Mismatch,
  minimum_sampled_cosine_distance_to_grazing_boundary: minBoundaryDistance,
  normal_texture_sample_bytes: packed.data.byteLength,
  native_frame_reference: samples.native_frame_reference,
  browser_native_psnr_delta_db: null, browser_native_ssim_delta: null, parity_gate: 'NOT_MEASURED',
  source_sha256: Object.fromEntries(sources.map(p => [p, sha(fs.readFileSync(new URL(p, root)))])),
  limitations: [...samples.limitations, 'Float32 shader arithmetic near grazing can differ from double CPU normal tests.',
    'Spark projected-quad 2DGS lacks the native per-pixel low-pass fallback; depth sorting/tie order and cutoff can differ.',
    'Spark zero-scale 2DGS branch bypasses blurAmount, preBlurAmount and maxPixelRadius.',
    'This CPU audit does not compile GPU shaders or produce browser images; the separate original015 GPU experiment is reported independently.'] };
fs.writeFileSync(new URL('reports/viewer_parity_v2_cpu_audit.json', root), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ status: report.status, flags: fullFlags.length, comparisons, mismatch, float32Mismatch,
  report: 'reports/viewer_parity_v2_cpu_audit.json' }));

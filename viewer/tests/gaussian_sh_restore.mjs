// CPU only: actual installed Spark worker/WASM, meshes, SH words and textures.
// Run: node viewer/tests/gaussian_sh_restore.mjs (stdout only; no artifacts).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { restoreGaussianSh } from '../gaussian_sh_restore.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
Object.assign(process.env, { PYTHONDONTWRITEBYTECODE: '1', NODE_DISABLE_COMPILE_CACHE: '1',
  XDG_CACHE_HOME: path.join(root, 'viewer/runtime/cache'),
  TMPDIR: path.join(root, 'viewer/runtime/tmp'),
  NODE_COMPILE_CACHE: path.join(root, 'viewer/runtime/cache/node'),
  CUDA_CACHE_PATH: path.join(root, 'viewer/runtime/cache/cuda'),
  __GL_SHADER_DISK_CACHE: '0' });
const { ExtSplats, SplatMesh, dyno } = await import('../node_modules/@sparkjsdev/spark/dist/spark.module.js');
const THREE = await import('../node_modules/three/build/three.module.js');
const bundle = fs.readFileSync(path.join(root, 'viewer/node_modules/@sparkjsdev/spark/dist/spark.module.js'), 'utf8');
const workerDeclaration = bundle.match(/var jsContent = [^\n]+/);
const embeddedWasm = bundle.match(/var spark_rs_bg_default = [^\n]*?\("([A-Za-z0-9+/=]+)"\)/);
assert.ok(workerDeclaration && embeddedWasm, 'installed Spark worker/WASM must exist');
const workerText = vm.runInNewContext(`${workerDeclaration[0]}\njsContent`);
const handlers = new Set(), pending = new Map();
let serial = 0;
const self = {
  addEventListener: (name, callback) => handlers.add(callback),
  removeEventListener: (name, callback) => handlers.delete(callback),
  postMessage: ({ id, result, error, status }) => {
    if (status !== undefined) return;
    const request = pending.get(id);
    if (!request) return;
    pending.delete(id);
    error !== undefined ? request.reject(new Error(String(error))) : request.resolve(result);
  },
};
vm.runInNewContext(workerText, { self, console, WebAssembly, TextDecoder, TextEncoder,
  ReadableStream, Response, Request, Headers, URL, ArrayBuffer, Uint8Array, Uint16Array,
  Uint32Array, Int32Array, Float32Array, Float64Array, DataView, performance });
const send = data => { for (const callback of [...handlers]) callback({ data }); };
send({ name: 'init-wasm', module: await WebAssembly.compile(Buffer.from(embeddedWasm[1], 'base64')) });
async function importedMesh(bytes) {
  const decoded = await new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    send({ id, name: 'loadExtSplats', args: { fileBytes: new Uint8Array(bytes), fileType: 'ply', lod: false } });
  });
  const mesh = new SplatMesh({ extSplats: new ExtSplats(decoded),
    enableLod: false, raycastable: false, editable: false });
  await mesh.initialized;
  return mesh;
}
const bytesOf = array => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const coreSha = mesh => {
  const hash = createHash('sha256');
  for (const array of mesh.extSplats.extArrays) hash.update(bytesOf(array));
  return hash.digest('hex');
};
function snapshotSplat(mesh, i) {
  const s = mesh.extSplats.getSplat(i); // Copies required: Spark reuses result objects.
  return { position: s.center.toArray(), scales: s.scales.toArray(), rotation: s.quaternion.toArray(),
    opacity: s.opacity, dc: s.color.toArray() };
}
function frame(mesh) {
  mesh.frameUpdate({ renderer: null, object: mesh, time: 0, deltaTime: 0,
    viewToWorld: new THREE.Matrix4(), camera: new THREE.PerspectiveCamera(),
    renderSize: new THREE.Vector2(640, 540), globalEdits: [] });
}
function materializeShTextures(mesh) {
  // Public source API constructs the real cached textures; generator closures
  // alone are lazy and need not allocate them until a shader program is built.
  mesh.extSplats.fetchSplat({ index: dyno.dynoConst('int', 0),
    viewOrigin: dyno.dynoConst('vec3', [0, 0, 0]) });
}
const bands = ['sh1', 'sh2', 'sh3a', 'sh3b'];
const textureKeys = ['sh1Texture', 'sh2Texture', 'sh3TextureA', 'sh3TextureB'];
function schema(bytes) {
  const end = bytes.indexOf('end_header\n') + 'end_header\n'.length;
  assert.ok(end >= 'end_header\n'.length);
  const header = bytes.subarray(0, end).toString('ascii');
  const fields = [...header.matchAll(/property (float|uchar) (\S+)/g)];
  const offsets = new Map();
  let stride = 0;
  for (const [, type, name] of fields) { offsets.set(name, stride); stride += type === 'float' ? 4 : 1; }
  const count = Number(header.match(/element vertex (\d+)/)[1]);
  return { count, stride, offsets, end,
    view: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) };
}
function decodedCoefficients(mesh, i, perChannel) {
  const extra = mesh.extSplats.extra;
  return Array.from({ length: perChannel }, (_, k) => {
    const band = k < 4 ? 0 : k < 8 ? 1 : k < 12 ? 2 : 3;
    const word = extra[bands[band]][i * 4 + k % 4];
    const range = 2 ** ((word >>> 27) - 15), step = range / 255;
    const rgb = [0, 1, 2].map(c => ((word >>> (c * 8)) & 255) * step *
      (word & (1 << (24 + c)) ? -1 : 1));
    const rgbFloat32 = [0, 1, 2].map(c => Math.fround(((word >>> (c * 8)) & 255) *
      Math.fround(step)) * (word & (1 << (24 + c)) ? -1 : 1));
    return { rgb, rgbFloat32, range, step };
  });
}
function verifyCoefficients(mesh, bytes, samples) {
  const source = schema(bytes), perChannel = ((mesh.extSplats.getNumSh() + 1) ** 2 - 1);
  let maxError = 0, maxFloat32Error = 0, coefficients = 0;
  for (const i of samples) {
    const decoded = decodedCoefficients(mesh, i, perChannel);
    for (let k = 0; k < perChannel; k++) for (let c = 0; c < 3; c++) {
      const raw = source.view.getFloat32(source.end + i * source.stride +
        source.offsets.get(`f_rest_${c * perChannel + k}`), true);
      const { rgb, rgbFloat32, range, step } = decoded[k];
      const error = Math.abs(rgb[c] - raw), error32 = Math.abs(rgbFloat32[c] - raw);
      assert.ok(Math.abs(raw) <= range, 'source signed magnitude must fit chosen exponent');
      assert.ok(error <= step / 2 + 4 * range * Number.EPSILON, 'per-triplet quantization bound');
      assert.ok(error32 <= step / 2 + range * (2 ** -23 + 2 ** -48), 'GLSL float32 decode bound');
      if (Math.abs(raw) > step / 2) assert.equal(Math.sign(rgb[c]), Math.sign(raw));
      maxError = Math.max(maxError, error);
      maxFloat32Error = Math.max(maxFloat32Error, error32);
      coefficients++;
    }
  }
  return { checkedCoefficients: coefficients, checkedMaxError: maxError,
    checkedMaxFloat32DecodeError: maxFloat32Error };
}
function fixture(degree = 3, count = 2, reordered = false) {
  const perChannel = (degree + 1) ** 2 - 1;
  let names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
    ...Array.from({ length: perChannel * 3 }, (_, i) => `f_rest_${i}`),
    'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  if (reordered) names = names.reverse();
  const header = Buffer.from('ply\nformat binary_little_endian 1.0\ncomment no fixed property order\n' +
    `element vertex ${count}\n` + names.map(n => `property float ${n}\n`).join('') + 'end_header\n');
  const payload = Buffer.alloc(count * names.length * 4);
  for (let i = 0; i < count; i++) for (let j = 0; j < names.length; j++) {
    const name = names[j];
    let value = 0;
    if (name === 'x') value = i + .25;
    if (name === 'y') value = -2;
    if (name === 'z') value = 3;
    if (name === 'rot_0') value = 1;
    if (name.startsWith('scale_')) value = -3;
    if (name.startsWith('f_rest_')) {
      const index = Number(name.slice(7));
      value = (index + 1) * .001 * (index % 2 ? -1 : 1);
      if (i === 0 && index % perChannel === 0) value = [.001, -.016, .031][Math.floor(index / perChannel)];
      if (i === 1 && index % perChannel === 0) value = [65536, -65536, 2 ** -149][Math.floor(index / perChannel)];
    }
    payload.writeFloatLE(value, (i * names.length + j) * 4);
  }
  return Buffer.concat([header, payload]);
}
function failureWithoutMutation(mesh, bytes, pattern) {
  const core = coreSha(mesh), sh = bands.map(key => mesh.extSplats.extra[key] && sha(bytesOf(mesh.extSplats.extra[key])));
  const version = mesh.version, dirty = mesh.generatorDirty;
  assert.throws(() => restoreGaussianSh(mesh, bytes), pattern);
  assert.equal(coreSha(mesh), core);
  assert.deepEqual(bands.map(key => mesh.extSplats.extra[key] && sha(bytesOf(mesh.extSplats.extra[key]))), sh);
  assert.equal(mesh.version, version);
  assert.equal(mesh.generatorDirty, dirty);
}

// Marker + arbitrary property order + cold and already materialized SH textures.
for (const degree of [0, 1, 2, 3]) {
  const bytes = fixture(degree, 2, degree === 3), mesh = await importedMesh(bytes);
  const markerBefore = degree === 3 ? decodedCoefficients(mesh, 0, 15)[0].rgb : null;
  const core = coreSha(mesh), input = sha(bytes), degreeBefore = mesh.extSplats.getNumSh();
  const splatBefore = [snapshotSplat(mesh, 0), snapshotSplat(mesh, 1)];
  if (degree !== 1) {
    frame(mesh);
    materializeShTextures(mesh);
  } // SH1 deliberately exercises lazy texture creation.
  const arrays = bands.map(key => mesh.extSplats.extra[key]);
  const padding = arrays.map(a => a && sha(bytesOf(a.subarray(mesh.numSplats * 4))));
  const textures = textureKeys.map(key => mesh.extSplats.extra[key]?.value);
  const versions = textures.map(t => t?.version);
  const meshVersion = mesh.version;
  const maxSh = [mesh.maxSh, mesh.extSplats.maxSh];
  const stats = restoreGaussianSh(mesh, bytes);
  assert.equal(coreSha(mesh), core);
  assert.equal(sha(bytes), input);
  assert.deepEqual([snapshotSplat(mesh, 0), snapshotSplat(mesh, 1)], splatBefore);
  assert.equal(mesh.extSplats.getNumSh(), degreeBefore);
  assert.deepEqual([mesh.maxSh, mesh.extSplats.maxSh], maxSh);
  arrays.forEach((array, b) => {
    assert.equal(mesh.extSplats.extra[bands[b]], array);
    if (array) assert.equal(sha(bytesOf(array.subarray(mesh.numSplats * 4))), padding[b]);
    if (textures[b]) assert.equal(textures[b].version, versions[b] + Number(stats.changedWords > 0));
  });
  frame(mesh);
  if (degree === 1) materializeShTextures(mesh); // Consume corrected words through the lazy path.
  if (degree > 0) assert.ok(mesh.version > meshVersion, 'next frame must regenerate accumulator RGB');
  verifyCoefficients(mesh, bytes, [0, 1]);
  assert.equal(restoreGaussianSh(mesh, bytes).changedWords, 0, 'repeat restore is idempotent');
  if (degree === 3) {
    const markerAfter = decodedCoefficients(mesh, 0, 15)[0];
    assert.equal(markerBefore[2], .015625, 'actual installed decoder demonstrates saturation');
    console.log('marker', JSON.stringify({ source: [.001, -.016, .031],
      installedDecoderBefore: markerBefore, restored: markerAfter.rgb,
      tripletQuantizationErrorBound: markerAfter.step / 2, ...stats }));
    const s = schema(bytes), bad = Buffer.from(bytes);
    bad.writeFloatLE(NaN, s.end + s.stride + s.offsets.get('f_rest_44'));
    failureWithoutMutation(mesh, bad, /nonfinite/);
    bad.writeFloatLE(131072, s.end + s.stride + s.offsets.get('f_rest_44'));
    failureWithoutMutation(mesh, bad, /exponent range/);
    bad.writeFloatLE(999, s.end + s.offsets.get('x'));
    failureWithoutMutation(mesh, bad, /center\/order/);
    failureWithoutMutation(mesh, bytes.subarray(0, -1), /truncated/);
    failureWithoutMutation(mesh, Buffer.from(bytes.toString('latin1').replace('f_rest_44', 'f_rest_45'), 'latin1'), /contiguous/);
    // Correct byteOffset handling for an ArrayBuffer view containing unrelated bytes.
    const envelope = Buffer.concat([Buffer.alloc(23), bytes, Buffer.alloc(17)]);
    assert.equal(restoreGaussianSh(mesh, envelope.subarray(23, 23 + bytes.length)).changedWords, 0);
    // Header-derived offsets also handle a scalar pre-vertex element and CRLF.
    const header = bytes.subarray(0, s.end).toString().replace('element vertex',
      'element auxiliary 2\nproperty uchar code\nproperty ushort flags\nelement vertex').replaceAll('\n', '\r\n');
    const withPrelude = Buffer.concat([Buffer.from(header), Buffer.alloc(6), bytes.subarray(s.end)]);
    assert.equal(restoreGaussianSh(mesh, withPrelude).changedWords, 0);
    // Reject a stale SH texture rather than silently changing an unreferenced buffer.
    const t = mesh.extSplats.extra.sh1Texture.value, originalData = t.image.data;
    t.image.data = new Uint32Array(originalData.length);
    failureWithoutMutation(mesh, bytes, /does not alias/);
    t.image.data = originalData;
  }
  mesh.dispose();
}

// Full actual source restoration; check all 45 coefficients in 1,000 records per block.
for (const block of ['014', '015', '016']) {
  const filename = path.join(root, `runs/garage_surface_v1/block_${block}/eval_dev/gaussians.ply`);
  const bytes = fs.readFileSync(filename), mesh = await importedMesh(bytes);
  const source = schema(bytes), inputSha = sha(bytes), beforeSha = coreSha(mesh);
  const records = Array.from({ length: 1000 }, (_, j) => Math.floor(j * (source.count - 1) / 999));
  const before = records.map(i => snapshotSplat(mesh, i));
  frame(mesh);
  materializeShTextures(mesh); // Real SH texture objects/backing arrays; no WebGL.
  const versions = textureKeys.map(key => mesh.extSplats.extra[key].value.version);
  const stats = restoreGaussianSh(mesh, bytes);
  const checked = verifyCoefficients(mesh, bytes, records);
  const afterSha = coreSha(mesh);
  assert.equal(afterSha, beforeSha, 'all positions/scales/rotation/opacity/DC bytes must be unchanged');
  assert.equal(sha(bytes), inputSha, 'source bytes unchanged');
  assert.deepEqual(records.map(i => snapshotSplat(mesh, i)), before, 'actual getSplat roundtrip unchanged');
  assert.equal(mesh.extSplats.getNumSh(), 3);
  assert.equal(stats.coefficients, source.count * 45);
  textureKeys.forEach((key, j) => assert.equal(mesh.extSplats.extra[key].value.version, versions[j] + 1));
  console.log('real', JSON.stringify({ block, ...stats, ...checked,
    inputSha256: inputSha, nonShExtArraysSha256Before: beforeSha, nonShExtArraysSha256After: afterSha }));
  mesh.dispose();
}
console.log('PASS: CPU-only actual Spark SH restoration; no GPU/browser, files or artifacts written.');

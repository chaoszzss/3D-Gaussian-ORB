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

const filename = path.join(root, 'runs/garage_surface_v1/block_015/eval_dev/gaussians.ply');
const bytes = fs.readFileSync(filename), spec = schema(bytes), mesh = await importedMesh(bytes);
const report = {gpu_used:false, ply_sha256:sha(bytes), count:spec.count, dc_source_gt1:0, dc_source_lt0:0,
 dc_loaded_gt1:0, error_max:0, error_gt_half_tolerance:0, clamp01_match:0, worst:[]};
for (let i=0;i<spec.count;i++) {
 const source=[0,1,2].map(c=>.5+.28209479177387814*spec.view.getFloat32(spec.end+i*spec.stride+spec.offsets.get(`f_dc_${c}`),true));
 const loaded=mesh.extSplats.getSplat(i).color.toArray();
 for(let c=0;c<3;c++) {
  if(source[c]>1)report.dc_source_gt1++;
  if(source[c]<0)report.dc_source_lt0++;
  if(loaded[c]>1)report.dc_loaded_gt1++;
  const error=Math.abs(source[c]-loaded[c]);
  if(error>Math.max(.000001,Math.abs(source[c])*.001)) report.error_gt_half_tolerance++;
  if(Math.abs(loaded[c]-Math.max(0,Math.min(1,source[c])))<.001)report.clamp01_match++;
  if(error>report.error_max){report.error_max=error;report.worst.push({index:i,source,loaded:[...loaded],error});}
 }
}
report.worst=report.worst.slice(-10);
fs.writeFileSync(path.join(root,'reports/viewer_parity_v2_dc_cpu.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report)); mesh.dispose();

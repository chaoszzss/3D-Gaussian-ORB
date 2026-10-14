// CPU only. Uses installed Spark/Three objects and GLSL generation, no browser.
// node --test viewer/tests/direction_support.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { webcrypto } from 'node:crypto';
import * as THREE from '../node_modules/three/build/three.module.js';
import { ExtSplats, SplatMesh, dyno } from '../node_modules/@sparkjsdev/spark/dist/spark.module.js';
import { rendererConfig, parseSideFlagsNpy, validateDirectionMetadata, loadDirectionSupport,
  sha256, sideVisible, buildSupportTextureData, attachSurfaceModifiers, applyNativeCamera, createNativeCompositor } from '../gaussian_direction_support.js';
globalThis.crypto ??= webcrypto;
const hex = 'a'.repeat(64);
function metadata(count = 4) {
  return { schema_version: 1, ply: { path: 'maps/test/gaussians.ply', sha256: hex, gaussians: count },
    side_flags: { path: 'maps/test/support_side_mask.npy', sha256: hex, dtype: 'uint8', count },
    normal_convention: 'quaternion_local_z', observed_cosine_min: .1, render_grazing_cosine: .05,
    policy: 'strong_unilateral_train_group_support', training_evidence: { split: 'train', train_only: true,
      heldout_used: false, minimum_unique_groups: 3, opposite_side_required_count: 0 } };
}
function npy(flags, version = 1, overrides = {}) {
  const length = version === 1 ? 10 : 12;
  const shape = overrides.shape ?? `${flags.length},`;
  let header = `{'descr': '${overrides.descr ?? '|u1'}', 'fortran_order': ${overrides.fortran ?? 'False'}, 'shape': (${shape}), }`;
  header += ' '.repeat((64 - (length + header.length + 1) % 64) % 64) + '\n';
  const preamble = Buffer.alloc(length);
  preamble[0] = 147; preamble.write('NUMPY', 1); preamble[6] = version;
  version === 1 ? preamble.writeUInt16LE(header.length, 8) : preamble.writeUInt32LE(header.length, 8);
  return Buffer.concat([preamble, Buffer.from(header), Buffer.from(flags)]);
}
function ply(quats = [[1, 0, 0, 0], [1, 0, 0, 0], [Math.SQRT1_2, 0, Math.SQRT1_2, 0], [1, 0, 0, 0]], reversed = false) {
  let names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  if (reversed) names.reverse();
  const header = Buffer.from('ply\nformat binary_little_endian 1.0\n' + `element vertex ${quats.length}\n` +
    names.map(n => `property float ${n}\n`).join('') + 'end_header\n');
  const body = Buffer.alloc(quats.length * names.length * 4);
  quats.forEach((q, i) => names.forEach((n, j) => {
    const value = n.startsWith('rot_') ? q[Number(n.slice(4))] : n === 'z' ? -2 : n.startsWith('scale_') ? -3 : 0;
    body.writeFloatLE(value, (i * names.length + j) * 4);
  }));
  return Buffer.concat([header, body]);
}
async function meshOf(count = 4) {
  let splats = new ExtSplats({ maxSplats: count }); await splats.initialized;
  for (let i = 0; i < count; i++) splats.pushSplat(new THREE.Vector3(i, 0, -2), new THREE.Vector3(.1, .2, .015),
    new THREE.Quaternion(), .9, new THREE.Color(1, .5, .1));
  splats = new ExtSplats({ extArrays: splats.extArrays, numSplats: count });
  await splats.initialized;
  assert.equal(splats.textures[0].image.data.buffer, splats.extArrays[0].buffer);
  const mesh = new SplatMesh({ extSplats: splats, enableLod: false, editable: false, raycastable: false });
  await mesh.initialized; return mesh;
}
function frame(mesh, z = 0) {
  mesh.frameUpdate({ renderer: null, object: mesh, time: 0, deltaTime: 0,
    viewToWorld: new THREE.Matrix4().makeTranslation(0, 0, z), camera: new THREE.PerspectiveCamera(),
    renderSize: new THREE.Vector2(640, 540), globalEdits: [] });
}

test('no metadata keeps legacy; explicit renderer enforces 2DGS and Z sort', () => {
  assert.equal(rendererConfig({}), null);
  const profile = rendererConfig({ renderer: { schema_version: 1, mode: 'inria_2dgs_v1' } });
  assert.deepEqual([profile.spark.enable2DGS, profile.spark.sortRadial, profile.surfaceScaleAxis], [true, false, 2]);
  assert.equal(profile.spark.accumExtSplats, true);
  for (const r of [null, {}, { schema_version: 1, mode: '3d' }]) assert.throws(() => rendererConfig({ renderer: r }));
});
test('ablations are explicit bounded inputs; no invisible 2DGS disable', () => {
  const config = spark => ({ renderer: { schema_version: 1, mode: 'inria_2dgs_v1', spark } });
  assert.equal(rendererConfig(config({ sortRadial: true, minAlpha: 0 })).spark.sortRadial, true);
  for (const spark of [{ enable2DGS: false }, { sortRadial: 'false' }, { maxStdDev: -1 }, { minAlpha: NaN }, { blurAmount: -1 }])
    assert.throws(() => rendererConfig(config(spark)));
});
test('v2 compositor preserves image-domain background, camera and restores renderer state', () => {
  const profile = rendererConfig({renderer: {schema_version: 1, mode: 'inria_2dgs_v2'}});
  assert.equal(profile.compositing, 'image_domain_rgba32f_final_clamp');
  let target = null, color = new THREE.Color().setRGB(.149, .1647, .2157, THREE.SRGBColorSpace);
  const saved = color.clone(), world = new THREE.Scene(), camera = new THREE.Camera(), calls = [];
  const spark = {uniforms: {encodeLinear: {value: false}}, onBeforeRender() { this.uniforms.encodeLinear.value = true; }};
  const original = spark.onBeforeRender;
  const renderer = { extensions: {has: () => true}, getDrawingBufferSize: v => v.set(640, 540),
    getRenderTarget: () => target, setRenderTarget: t => {target = t;}, getClearAlpha: () => 1,
    getClearColor: c => c.copy(color), setClearColor: c => {color = c.clone();},
    render(scene, view) {
      if (scene === world) {
        spark.onBeforeRender(); assert.equal(spark.uniforms.encodeLinear.value, false);
        assert.equal(target.texture.type, THREE.FloatType); assert.equal(view, camera);
        assert.ok(Math.abs(color.r - .149) < 1e-7); assert.ok(Math.abs(color.b - .2157) < 1e-7);
      } else assert.equal(target, null);
      calls.push(scene);
    }};
  const compositor = createNativeCompositor(THREE, renderer, spark);
  compositor.render(world,camera,[.149,.1647,.2157]); assert.equal(calls.length,2); assert.deepEqual(color,saved);
  renderer.render = () => {throw Error('lost context');};
  assert.throws(() => compositor.render(world,camera)); assert.equal(target,null); assert.deepEqual(color,saved);
  compositor.dispose(); assert.equal(spark.onBeforeRender,original);
  renderer.extensions.has = () => false;
  assert.throws(() => createNativeCompositor(THREE,renderer,spark));
});
test('NPY v1 and v2 decode exact uint8 order including subarray offsets', () => {
  for (const version of [1, 2]) {
    const flags = [0, 1, 2, 3], data = npy(flags, version);
    const envelope = Buffer.concat([Buffer.alloc(13), data, Buffer.alloc(5)]);
    assert.deepEqual([...parseSideFlagsNpy(envelope.subarray(13, 13 + data.length), 4)], flags);
  }
});
test('NPY rejects dtype, Fortran, wrong shape, bad flags, truncation and extra bytes', () => {
  for (const data of [npy([0, 1], 1, { descr: '|i1' }), npy([0, 1], 1, { fortran: 'True' }),
    npy([0, 1], 1, { shape: '1,2' }), npy([0, 4]), npy([0, 1]).subarray(0, -1),
    Buffer.concat([npy([0, 1]), Buffer.from([0])])]) assert.throws(() => parseSideFlagsNpy(data, 2));
  assert.throws(() => parseSideFlagsNpy(npy([0, 1]), 3));
});
test('metadata keeps TRAIN and render thresholds separate, validates evidence', () => {
  const m = metadata(); assert.equal(validateDirectionMetadata(m, 4).render_grazing_cosine, .05);
  for (const edit of [m => m.ply.gaussians++, m => m.side_flags.count++, m => m.side_flags.sha256 = 'bad',
    m => m.normal_convention = 'ply_normals', m => m.render_grazing_cosine = .1,
    m => m.training_evidence.split = 'dev', m => m.training_evidence.heldout_used = true,
    m => m.training_evidence.minimum_unique_groups = 2, m => m.training_evidence.opposite_side_required_count = 1]) {
    const changed = metadata(); edit(changed); assert.throws(() => validateDirectionMetadata(changed, 4));
  }
});
test('side signs, both/unknown, grazing and camera translation obey contract', () => {
  const n = [0, 0, 1];
  assert.equal(sideVisible(1, n, [0, 0, 1], .05), true);
  assert.equal(sideVisible(1, n, [0, 0, -1], .05), false);
  assert.equal(sideVisible(2, n, [0, 0, 1], .05), false);
  assert.equal(sideVisible(2, n, [0, 0, -1], .05), true);
  for (const flag of [0, 3]) for (const z of [-1, 1]) assert.equal(sideVisible(flag, n, [0, 0, z], .05), true);
  for (const flag of [1, 2]) {
    assert.equal(sideVisible(flag, n, [1, 0, 0], .05), true);
    assert.equal(sideVisible(flag, n, [0, 0, 0], .05), true);
    assert.equal(sideVisible(flag, n, [1, 0, -.049], .05), true);
  }
  assert.equal(sideVisible(1, n, [1, 0, -.051], .05), false);
  assert.equal(sideVisible(1, n, [1, 0, -.075], .1), true); // Exposes accidental TRAIN threshold substitution.
  assert.equal(sideVisible(1, n, [1, 0, -.075], .05), false);
});
test('fixed original PLY wxyz normals, q/-q invariance and row order', () => {
  const bytes = ply(), flags = Uint8Array.from([0, 1, 2, 3]), before = Buffer.from(bytes);
  const packed = buildSupportTextureData(bytes, flags);
  assert.deepEqual(packed.histogram, [1, 1, 1, 1]);
  assert.deepEqual([...packed.data.subarray(4, 8)], [0, 0, 1, 1]);
  assert.ok(Math.abs(packed.data[8] - 1) < 1e-6); assert.equal(packed.data[11], 2);
  const qq = [[2, 0, 0, 0], [-2, 0, 0, 0]];
  const same = buildSupportTextureData(ply(qq, true), Uint8Array.from([1, 1])).data;
  assert.deepEqual([...same.subarray(0, 3)], [...same.subarray(4, 7)]);
  assert.deepEqual(bytes, before, 'immutable PLY');
});
test('invalid unilateral quaternion and GPU texture capacity fail before attachment', () => {
  assert.throws(() => buildSupportTextureData(ply([[0, 0, 0, 0]]), Uint8Array.of(1)), /quaternion/);
  assert.throws(() => buildSupportTextureData(ply([[NaN, 0, 0, 0]]), Uint8Array.of(2)), /quaternion/);
  assert.doesNotThrow(() => buildSupportTextureData(ply([[0, 0, 0, 0]]), Uint8Array.of(0)));
  assert.throws(() => buildSupportTextureData(ply(), Uint8Array.from([0, 1, 2, 3]), 1), /capacity/);
});
test('authenticated loader pins metadata, exact PLY and flags, uses only listed routes', async () => {
  const bytes = ply(), flags = npy([0, 1, 2, 3]), m = metadata();
  m.ply.sha256 = await sha256(bytes); m.side_flags.sha256 = await sha256(flags);
  const json = Buffer.from(JSON.stringify(m)), calls = [];
  const descriptor = { url: '/api/blocks/unit/direction_support', sha256: await sha256(json) };
  const fetchBytes = async url => { calls.push(url); return url.endsWith('/side_flags') ? flags : json; };
  const args = { descriptor, blockId: 'unit', plyBytes: bytes, count: 4, fetchBytes, baseURL: 'http://127.0.0.1:8765/' };
  const support = await loadDirectionSupport(args);
  assert.deepEqual([...support.flags], [0, 1, 2, 3]);
  assert.deepEqual(calls, ['http://127.0.0.1:8765/api/blocks/unit/direction_support', 'http://127.0.0.1:8765/api/blocks/unit/direction_support/side_flags']);
  await assert.rejects(loadDirectionSupport({ ...args, descriptor: { ...descriptor, sha256: hex } }), /metadata SHA/);
  await assert.rejects(loadDirectionSupport({ ...args, plyBytes: Buffer.from('wrong PLY') }), /PLY SHA/);
  await assert.rejects(loadDirectionSupport({ ...args, descriptor: { ...descriptor, url: 'https://external.example/sidecar' } }), /route/);
  await assert.rejects(loadDirectionSupport({ ...args, fetchBytes: async url => url.endsWith('/side_flags') ? npy([3, 2, 1, 0]) : json }), /side_flags SHA/);
});
test('real Spark dyno compiles zero Z + world alpha gate; SH0 camera motion regenerates', async () => {
  const mesh = await meshOf(), arrays = mesh.extSplats.extArrays.map(a => new Uint32Array(a));
  const oldModifier = new dyno.Dyno({ inTypes: { gsplat: dyno.Gsplat }, outTypes: { gsplat: dyno.Gsplat },
    statements: ({ inputs, outputs }) => [`${outputs.gsplat} = ${inputs.gsplat};`] });
  mesh.objectModifiers = [oldModifier];
  const attached = attachSurfaceModifiers({ THREE, dyno, mesh, plyBytes: ply(), maxTextureSize: 4096,
    support: { metadata: metadata(), flags: Uint8Array.from([0, 1, 2, 3]), metadataSha256: hex, plySha256: hex } });
  assert.equal(mesh.objectModifiers[0], oldModifier);
  assert.equal(mesh.enableViewToWorld, true);
  frame(mesh);
  const compilation = new dyno.Compilation();
  const shader = mesh.generator.compile({ inputs: { index: 'testIndex' }, outputs: { gsplat: 'result' }, compile: compilation }).join('\n');
  assert.match(shader, /\.scales\.z = 0\.0/);
  assert.match(shader, /texelFetch/); assert.match(shader, /\.index % supportWidth/);
  assert.match(shader, /\.rgba\.a = 0\.0/); assert.match(shader, /support\.w == 1\.0 && cosine < -/);
  assert.equal(Object.values(compilation.uniforms).filter(u => u.value?.isDataTexture && u.value.type === THREE.FloatType).length, 1);
  assert.ok(Object.values(compilation.uniforms).some(u => u === mesh.context.viewToWorld.translate.uniform));
  const version = mesh.version; frame(mesh, 5);
  assert.ok(mesh.version > version, 'SH0 needs camera-dependent accumulator regeneration');
  assert.equal(mesh.context.viewToWorld.translate.value.z, 5);
  arrays.forEach((a, i) => assert.deepEqual(mesh.extSplats.extArrays[i], a, 'original Gaussian attributes unmodified'));
  assert.equal(attached.diagnostics.renderGrazingCosine, .05);
  assert.equal(attached.diagnostics.nativeParityMeasured, false);
  attached.dispose(); assert.deepEqual(mesh.objectModifiers, [oldModifier]);
  assert.equal(mesh.enableViewToWorld, false); mesh.dispose();
});
test('surface-only renderer has zero Z and no direction culling', async () => {
  const mesh = await meshOf();
  const attachment = attachSurfaceModifiers({ THREE, dyno, mesh });
  frame(mesh); const c = new dyno.Compilation();
  const shader = mesh.generator.compile({ inputs: { index: 'idx' }, outputs: { gsplat: 'outSplat' }, compile: c }).join('\n');
  assert.match(shader, /\.scales\.z = 0\.0/); assert.doesNotMatch(shader, /supportWidth/);
  assert.equal(mesh.enableViewToWorld, false); attachment.dispose(); mesh.dispose();
});
test('reject transformed or LoD meshes before modifying generator', async () => {
  const mesh = await meshOf(); mesh.position.x = 1;
  assert.throws(() => attachSurfaceModifiers({ THREE, dyno, mesh }), /identity/);
  assert.equal(mesh.objectModifiers, undefined);
  mesh.position.x = 0; mesh.enableLod = true;
  assert.throws(() => attachSurfaceModifiers({ THREE, dyno, mesh }), /non-LoD/); mesh.dispose();
});
test('native camera preserves real off-axis intrinsics and exact NS pose at pixel centres', () => {
  const camera = new THREE.PerspectiveCamera();
  const record = { camera: { w: 640, h: 540, fl_x: 320, fl_y: 300, cx: 307.25, cy: 266.75,
    camera_to_worlds_ns: [[0, -1, 0, 5], [1, 0, 0, -3], [0, 0, 1, 9]] } };
  applyNativeCamera(THREE, camera, record);
  const c = record.camera;
  for (const [px, py] of [[.5, .5], [639.5, 539.5], [320.5, 270.5]]) {
    const point = new THREE.Vector3((px - c.cx) / c.fl_x * 7, -(py - c.cy) / c.fl_y * 7, -7).applyMatrix4(camera.matrixWorld);
    const ndc = point.project(camera);
    assert.ok(Math.abs((ndc.x + 1) * c.w / 2 - px) < 1e-10);
    assert.ok(Math.abs((1 - ndc.y) * c.h / 2 - py) < 1e-10);
  }
  assert.equal(camera.matrixAutoUpdate, false);
  assert.deepEqual(camera.matrixWorld.elements, [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 5, -3, 9, 1]);
});
test('actual native frame008 camera enters unchanged; bad poses rejected', () => {
  const path = new URL('../../runs/fidelity_v2/original_noownership015/block_015/metrics.json', import.meta.url);
  const report = JSON.parse(fs.readFileSync(path));
  const record = report.frames.find(f => f.dataset_index === 8), camera = new THREE.PerspectiveCamera();
  const info = applyNativeCamera(THREE, camera, record);
  assert.deepEqual([info.width, info.height, report.inference_sh_degree], [640, 540, 3]);
  const flattened = record.camera.camera_to_worlds_ns.flat().concat([0, 0, 0, 1]);
  assert.deepEqual(camera.matrixWorld.clone().transpose().elements, flattened);
  const bad = structuredClone(record); bad.camera.camera_to_worlds_ns[0][0] = 99;
  assert.throws(() => applyNativeCamera(THREE, new THREE.PerspectiveCamera(), bad), /rigid/);
});
test('installed 2DGS vertex branch bypasses blur and radius clamps: record limitation', () => {
  const sourceMap = JSON.parse(fs.readFileSync(new URL('../node_modules/@sparkjsdev/spark/dist/spark.module.js.map', import.meta.url)));
  const index = sourceMap.sources.findIndex(p => p.endsWith('shaders/splatVertex.glsl'));
  const shader = JSON.parse(sourceMap.sourcesContent[index].slice('export default '.length).replace(/;\s*$/, ''));
  const start = shader.indexOf('if (enable2DGS && any(zeroScales))');
  assert.ok(start > 0);
  const branch = shader.slice(start, shader.indexOf('mat3 RS', start));
  assert.match(branch, /return;/); assert.doesNotMatch(branch, /blurAmount|maxPixelRadius/);
});

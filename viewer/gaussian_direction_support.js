// Static support contract and explicitly versioned Spark 2.3.1 renderer adapters.
// Source PLY row order is immutable; camera motion changes only generated alpha.
const SHA256 = /^[a-f0-9]{64}$/;
function check(ok, message) {
  if (!ok) throw new Error(`Gaussian direction support: ${message}`);
}
function bytesOf(value) {
  const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : value;
  check(bytes instanceof Uint8Array, 'expected bytes');
  return bytes;
}
export async function sha256(value) {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytesOf(value));
  return Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
}

/** Absence is the ONLY legacy default. Bad explicit metadata never falls back. */
export function rendererConfig(manifest) {
  if (manifest.renderer === undefined) return null;
  const r = manifest.renderer;
  check(r && r.schema_version === 1 && ['inria_2dgs_v1', 'inria_2dgs_v2', 'inria_2dgs_v3'].includes(r.mode), 'unsupported renderer contract');
  // These are experiment settings, not a claim of native rasterizer equivalence.
  const spark = { enable2DGS: true, sortRadial: false, accumExtSplats: true,
    enableLod: false, maxStdDev: 3, minAlpha: 1 / 255, blurAmount: 0,
    preBlurAmount: 0, maxPixelRadius: 512 };
  if (r.spark !== undefined) {
    check(r.spark && typeof r.spark === 'object' && !Array.isArray(r.spark), 'invalid Spark ablation');
    for (const [key, value] of Object.entries(r.spark)) {
      check(['sortRadial', 'maxStdDev', 'minAlpha', 'blurAmount', 'preBlurAmount', 'maxPixelRadius'].includes(key), 'unsupported Spark ablation key');
      check(key === 'sortRadial' ? typeof value === 'boolean' : Number.isFinite(value), 'invalid Spark ablation value');
      if (key === 'maxStdDev') check(value > 0 && value <= 8, 'maxStdDev out of range');
      if (key === 'minAlpha') check(value >= 0 && value <= 1, 'minAlpha out of range');
      if (key.endsWith('BlurAmount') || key === 'blurAmount') check(value >= 0 && value <= 16, 'blur amount out of range');
      if (key === 'maxPixelRadius') check(value > 0 && value <= 16384, 'pixel radius out of range');
      spark[key] = value;
    }
  }
  if (r.mode === 'inria_2dgs_v3') check(r.spark === undefined, 'v3 freezes native AA constants; no ablations');
  return { mode: r.mode, spark, surfaceScaleAxis: 2,
    compositing: r.mode !== 'inria_2dgs_v1' ? 'image_domain_rgba32f_final_clamp' : 'canvas_unorm8',
    coverage: r.mode === 'inria_2dgs_v3' ? 'native_tiled_aabb_rho_union_v1' : 'spark_3sigma_quad' };
}

/** Spark 2.3.1 public shader hooks; one primitive and one fragment contribution.
 * Only the explicit v3 2DGS path changes. The native low-pass disk can lie far
 * outside the projected 3-sigma quad, so union coverage must be generated first.
 * T is computed from the same projected tangent axes used by the native kernel.
 */
export function nativeAaShaderSources(vertexShader, fragmentShader) {
  const replaceOnce = (text, anchor, replacement) => {
    check(text.split(anchor).length === 2, 'unexpected Spark shader revision');
    return text.replace(anchor, replacement);
  };
  const declarations = `
flat $IO int vNativeAa;
flat $IO vec3 vNativeTu;
flat $IO vec3 vNativeTv;
flat $IO vec3 vNativeTw;
flat $IO vec2 vNativeImage;
`;
  let vertex = replaceOnce(vertexShader, 'void main() {', declarations.replaceAll('$IO', 'out') + '\nvoid main() {\n    vNativeAa = 0;');
  // Keep native center near cull. Far and XY coarse culling belong to Spark's
  // 3D path; native AA may contribute even when these coarse bounds reject it.
  vertex = replaceOnce(vertex, 'if (abs(clipCenter.z) >= clipCenter.w) {',
    'if (!(enable2DGS && any(zeroScales)) && abs(clipCenter.z) >= clipCenter.w) {');
  vertex = replaceOnce(vertex, 'if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {',
    'if (!(enable2DGS && any(zeroScales)) && (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip)) {');
  const begin = '        if (enable2DGS && any(zeroScales)) {';
  const end = '\n        mat3 RS = scaleQuaternionToMatrix(scales, viewQuaternion);';
  check(vertex.split(begin).length === 2 && vertex.split(end).length === 2, 'unexpected 2DGS shader branch');
  const start = vertex.indexOf(begin), stop = vertex.indexOf(end, start);
  check(stop > start, 'invalid 2DGS shader branch ordering');
  vertex = vertex.slice(0, start) + `
        if (enable2DGS && any(zeroScales)) {
            if (viewCenter.z >= -0.2) return;
            // This profile explicitly requires local Z=0 from its modifier.
            vec3 axisU = quatVec(viewQuaternion, vec3(scales.x, 0.0, 0.0));
            vec3 axisV = quatVec(viewQuaternion, vec3(0.0, scales.y, 0.0));
            vec4 clipU = projectionMatrix * vec4(axisU, 0.0);
            vec4 clipV = projectionMatrix * vec4(axisV, 0.0);
            vec3 tw = vec3(clipU.w, clipV.w, clipCenter.w);
            vec3 tu = 0.5 * renderSize.x * vec3(clipU.x, clipV.x, clipCenter.x) + 0.5 * (renderSize.x - 1.0) * tw;
            vec3 tv = -0.5 * renderSize.y * vec3(clipU.y, clipV.y, clipCenter.y) + 0.5 * (renderSize.y - 1.0) * tw;
            vec3 cutoff = vec3(9.0, 9.0, -1.0);
            float denominator = dot(cutoff, tw * tw);
            if (denominator == 0.0) return;
            vec3 f = cutoff / denominator;
            vec2 image = vec2(dot(f, tu * tw), dot(f, tv * tw));
            vec2 extent = sqrt(max(vec2(0.0001), image * image - vec2(dot(f, tu * tu), dot(f, tv * tv))));
            float radius = ceil(max(max(extent.x, extent.y), 3.0 * 0.707106));
            vec2 grid = ceil(renderSize / 16.0);
            // Native getRect uses integer truncation towards zero, not floor.
            vec2 tileLo = clamp(trunc((image - radius) / 16.0), vec2(0.0), grid);
            vec2 tileHi = clamp(trunc((image + radius + 15.0) / 16.0), vec2(0.0), grid);
            if (any(lessThanEqual(tileHi, tileLo))) return;
            vNativeAa = 1;
            vNativeTu = tu; vNativeTv = tv; vNativeTw = tw; vNativeImage = image;
            vec2 edge = mix(tileLo * 16.0, tileHi * 16.0, position.xy * 0.5 + 0.5);
            // Screen-aligned native tile union. Per-ray near test is in fragment;
            // use neutral clip Z so hardware does not erase valid far ray hits.
            gl_Position = vec4(2.0 * edge.x / renderSize.x - 1.0, 1.0 - 2.0 * edge.y / renderSize.y, 0.0, 1.0);
            vNdc = gl_Position.xyz;
            return;
        }
` + vertex.slice(stop);
  let fragment = replaceOnce(fragmentShader, 'void main() {', declarations.replaceAll('$IO', 'in') + '\nuniform vec2 renderSize;\nvoid main() {');
  const oldRho = `    float z2 = dot(vSplatUv, vSplatUv);
    if (z2 > (adjustedStdDev * adjustedStdDev)) {
        discard;
    }`;
  fragment = replaceOnce(fragment, oldRho, `    float z2;
    if (vNativeAa == 1) {
        vec2 pixel = vec2(gl_FragCoord.x - 0.5, renderSize.y - gl_FragCoord.y - 0.5);
        vec3 crossRay = cross(pixel.x * vNativeTw - vNativeTu, pixel.y * vNativeTw - vNativeTv);
        if (crossRay.z == 0.0) discard;
        vec2 uv = crossRay.xy / crossRay.z;
        float rayDepth = dot(uv, vNativeTw.xy) + vNativeTw.z;
        if (rayDepth < 0.2) discard;
        vec2 delta = vNativeImage - pixel;
        z2 = min(dot(uv, uv), 2.0 * dot(delta, delta));
    } else {
        z2 = dot(vSplatUv, vSplatUv);
        if (z2 > adjustedStdDev * adjustedStdDev) discard;
    }`);
  fragment = replaceOnce(fragment, '    if (rgba.a < minAlpha) {',
    '    if (vNativeAa == 1) rgba.a = min(0.99, rgba.a);\n    if (rgba.a < minAlpha) {');
  return { vertexShader: vertex, fragmentShader: fragment };
}

export function installNativeAaShader(spark) {
  check(spark.material?.isShaderMaterial && spark.uniforms.renderSize, 'requires Spark shader material');
  const shaders = nativeAaShaderSources(spark.material.vertexShader, spark.material.fragmentShader);
  Object.assign(spark.material, shaders); spark.material.needsUpdate = true;
  return { mode: 'native_tiled_aabb_rho_union_v1', passes: 1, lowpassInverseVariance: 2,
    aabbCutoff: 3, tileSize: 16, centerNear: 0.2, rayNear: 0.2, rayFar: null,
    perFragmentAlphaCap: 0.99, sourceOpacityChanged: false,
    limitation: 'Spark back-to-front blending retains its ordering; native front-to-back early termination is not emulated' };
}

/** v2 keeps HDR SH radiance through blending, then clamps once as native does.
 * No camera, geometry, opacity or fragment falloff modification. Requires real
 * float blending; never silently substitutes a clipped or gamma-linear pipeline.
 */
export function createNativeCompositor(THREE, renderer, spark) {
  check(renderer.extensions.has('EXT_color_buffer_float') && renderer.extensions.has('EXT_float_blend'),
    'v2 requires RGBA32F render targets and float blending');
  const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType,
    format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    depthBuffer: true, stencilBuffer: false, generateMipmaps: false });
  target.texture.colorSpace = THREE.NoColorSpace;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const material = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3,
    uniforms: { source: { value: target.texture } }, depthTest: false, depthWrite: false,
    blending: THREE.NoBlending,
    vertexShader: 'precision highp float; in vec3 position; out vec2 uv; void main(){ uv=position.xy*.5+.5; gl_Position=vec4(position,1.); }',
    fragmentShader: 'precision highp float; uniform sampler2D source; in vec2 uv; out vec4 color; void main(){ color=vec4(clamp(texture(source,uv).rgb,0.,1.),1.); }' });
  const scene = new THREE.Scene(); scene.add(new THREE.Mesh(geometry, material));
  const camera = new THREE.Camera(), size = new THREE.Vector2(), savedColor = new THREE.Color();
  const original = spark.onBeforeRender;
  // Spark chooses linear-light conversion for non-XR render targets. The native
  // supervision/composition domain is image RGB, so override that public uniform
  // AFTER its normal callback has populated it. Do not change global color state.
  const before = function (...args) {
    original.apply(this, args);
    if (renderer.getRenderTarget() === target) this.uniforms.encodeLinear.value = false;
  };
  spark.onBeforeRender = before;
  return {
    render(world, view, backgroundRgb = null) {
      renderer.getDrawingBufferSize(size);
      if (target.width !== size.x || target.height !== size.y) target.setSize(size.x, size.y);
      const previousTarget = renderer.getRenderTarget(), alpha = renderer.getClearAlpha();
      renderer.getClearColor(savedColor);
      const rgb = backgroundRgb ? new THREE.Color().setRGB(...backgroundRgb) : savedColor.clone().convertLinearToSRGB();
      try {
        renderer.setRenderTarget(target);
        // Three clears ordinary render targets in its working linear space;
        // supply the exact image-domain background numbers without conversion.
        renderer.setClearColor(rgb, alpha);
        renderer.render(world, view);
        renderer.setRenderTarget(previousTarget);
        renderer.render(scene, camera);
      } finally {
        renderer.setRenderTarget(previousTarget); renderer.setClearColor(savedColor, alpha);
      }
    },
    diagnostics: { target: 'RGBA32F', radianceDomain: 'image_rgb', encodeLinear: false,
      clamp: 'once_after_blending', alphaModification: false },
    dispose() {
      if (spark.onBeforeRender === before) spark.onBeforeRender = original;
      target.dispose(); geometry.dispose(); material.dispose();
    },
  };
}

/** NumPy uint8 v1/v2, C order, exact payload. No eval/pickle or coercion. */
export function parseSideFlagsNpy(value, count) {
  const bytes = bytesOf(value);
  check(Number.isSafeInteger(count) && count > 0, 'invalid flag count');
  check(bytes.length >= 10 && bytes[0] === 147 &&
    new TextDecoder().decode(bytes.subarray(1, 6)) === 'NUMPY', 'not NPY');
  const version = bytes[6];
  check((version === 1 || version === 2) && bytes[7] === 0, 'unsupported NPY version');
  const offset = version === 1 ? 10 : 12;
  check(bytes.length >= offset, 'truncated NPY header');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  check(headerLength > 0 && headerLength <= 65536 && offset + headerLength <= bytes.length, 'invalid NPY header length');
  const header = new TextDecoder('ascii').decode(bytes.subarray(offset, offset + headerLength));
  const format = /^\s*\{\s*['"]descr['"]\s*:\s*['"]\|u1['"]\s*,\s*['"]fortran_order['"]\s*:\s*False\s*,\s*['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*\)\s*,?\s*\}\s*$/;
  const match = header.match(format);
  check(match && Number(match[1]) === count, 'NPY requires uint8 C-order shape (N,)');
  check(offset + headerLength + count === bytes.length, 'NPY payload length/count mismatch');
  const flags = bytes.subarray(offset + headerLength);
  for (const flag of flags) check(flag <= 3, 'invalid side flag');
  return flags;
}

export function validateDirectionMetadata(meta, count) {
  check(meta && meta.schema_version === 1, 'unsupported metadata schema');
  check(meta.ply && meta.ply.gaussians === count && SHA256.test(meta.ply.sha256), 'PLY identity/count missing');
  check(meta.side_flags && meta.side_flags.dtype === 'uint8' && meta.side_flags.count === count &&
    SHA256.test(meta.side_flags.sha256), 'side_flags identity/count missing');
  check(meta.normal_convention === 'quaternion_local_z', 'unsupported normal convention');
  check(meta.policy === 'strong_unilateral_train_group_support', 'unsupported support evidence policy');
  check(Number.isFinite(meta.observed_cosine_min) && meta.observed_cosine_min > 0 && meta.observed_cosine_min < 1,
    'invalid TRAIN observation threshold');
  check(Number.isFinite(meta.render_grazing_cosine) && meta.render_grazing_cosine >= 0 &&
    meta.render_grazing_cosine < meta.observed_cosine_min, 'invalid render grazing threshold');
  check(meta.training_evidence && typeof meta.training_evidence === 'object' && !Array.isArray(meta.training_evidence),
    'TRAIN evidence metadata missing');
  const evidence = meta.training_evidence;
  check(evidence.split === 'train' && (evidence.train_only === undefined || evidence.train_only === true) && evidence.heldout_used === false &&
    evidence.minimum_unique_groups >= 3 && Number.isInteger(evidence.minimum_unique_groups) &&
    evidence.opposite_side_required_count === 0, 'unsupported TRAIN evidence provenance');
  return meta;
}

/** Only manifest-listed routes supplied by serve_map; never turn file paths into URLs. */
export async function loadDirectionSupport({ descriptor, blockId, plyBytes, count, fetchBytes, baseURL }) {
  check(/^[A-Za-z0-9_-]{1,96}$/.test(blockId), 'invalid block ID');
  const route = `/api/blocks/${blockId}/direction_support`;
  check(descriptor && descriptor.url === route && SHA256.test(descriptor.sha256), 'missing pinned sidecar route');
  const url = new URL(route, baseURL);
  const raw = await fetchBytes(url.href, 1048576);
  check(await sha256(raw) === descriptor.sha256, 'metadata SHA mismatch');
  const meta = validateDirectionMetadata(JSON.parse(new TextDecoder().decode(raw)), count);
  const plySha = await sha256(plyBytes);
  check(plySha === meta.ply.sha256, 'PLY SHA mismatch');
  const npy = await fetchBytes(new URL(`${route}/side_flags`, baseURL).href, count + 65548);
  check(await sha256(npy) === meta.side_flags.sha256, 'side_flags SHA mismatch');
  const flags = parseSideFlagsNpy(npy, count);
  return { metadata: meta, metadataSha256: descriptor.sha256, plySha256: plySha, flags };
}

/** CPU reference: unsupported side strictly beyond grazing; zero-distance keeps. */
export function sideVisible(flag, normal, cameraMinusMean, grazing) {
  check(Number.isInteger(flag) && flag >= 0 && flag <= 3, 'invalid side flag');
  check(Number.isFinite(grazing) && grazing >= 0 && grazing < 1, 'invalid grazing threshold');
  check(normal.length === 3 && cameraMinusMean.length === 3 && [...normal, ...cameraMinusMean].every(Number.isFinite), 'invalid direction');
  if (flag === 0 || flag === 3) return true;
  const distance = Math.hypot(...cameraMinusMean);
  if (distance <= 1e-10) return true;
  const cosine = normal.reduce((sum, x, i) => sum + x * cameraMinusMean[i], 0) / distance;
  return !((flag === 1 && cosine < -grazing) || (flag === 2 && cosine > grazing));
}

// Standard exported PLY: vertex only, float32 fields, arbitrary property order.
// Read the original quaternion (not Spark's compressed quaternion) for side tests.
export function buildSupportTextureData(plyBytes, flags, maxTextureSize = 4096) {
  const bytes = bytesOf(plyBytes);
  const text = new TextDecoder('ascii').decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
  const match = text.match(/^ply\r?\nformat binary_little_endian 1\.0\r?\n([\s\S]*?)end_header\r?\n/);
  check(match, 'requires standard binary little-endian PLY');
  const fields = [], counts = [];
  for (const line of match[1].split(/\r?\n/).filter(Boolean)) {
    if (/^(comment|obj_info)\s/.test(line)) continue;
    const element = line.match(/^element vertex (\d+)$/);
    if (element) { counts.push(Number(element[1])); continue; }
    const property = line.match(/^property (?:float|float32) (\w+)$/);
    check(property && counts.length === 1 && !fields.includes(property[1]), 'unsupported PLY layout');
    fields.push(property[1]);
  }
  check(counts.length === 1 && counts[0] === flags.length, 'PLY flag row count mismatch');
  const offsets = [0, 1, 2, 3].map(i => fields.indexOf(`rot_${i}`));
  check(offsets.every(i => i >= 0), 'PLY quaternion missing');
  const stride = fields.length * 4, headerBytes = match[0].length;
  check(headerBytes + flags.length * stride === bytes.length, 'PLY payload length mismatch');
  check(Number.isInteger(maxTextureSize) && maxTextureSize > 0, 'invalid texture size');
  const width = Math.min(2048, maxTextureSize, flags.length), height = Math.ceil(flags.length / width);
  check(height <= maxTextureSize, 'direction texture exceeds device capacity');
  const data = new Float32Array(width * height * 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), histogram = [0, 0, 0, 0];
  for (let i = 0; i < flags.length; i++) {
    const flag = flags[i];
    check(flag <= 3, 'invalid side flag');
    histogram[flag]++;
    data[4 * i + 3] = flag;
    // Unknown/both cannot be culled, and need no normal.
    if (flag !== 1 && flag !== 2) continue;
    let [w, x, y, z] = offsets.map(offset => view.getFloat32(headerBytes + i * stride + offset * 4, true));
    const norm = Math.hypot(w, x, y, z);
    check(Number.isFinite(norm) && norm > 1e-12, 'invalid single-sided PLY quaternion');
    w /= norm; x /= norm; y /= norm; z /= norm;
    data[4 * i] = 2 * (x * z + w * y);
    data[4 * i + 1] = 2 * (y * z - w * x);
    data[4 * i + 2] = 1 - 2 * (x * x + y * y);
  }
  return { data, width, height, histogram };
}

/** Append modifiers; preserve existing modifier chains and immutable PLY arrays. */
export function attachSurfaceModifiers({ THREE, dyno, mesh, support = null, plyBytes, maxTextureSize }) {
  check(mesh.isInitialized && mesh.extSplats && mesh.splats === mesh.extSplats && !mesh.paged &&
    mesh.enableLod === false && !mesh.extSplats.lodSplats && !mesh.covSplats, 'requires initialized non-LoD ExtSplats');
  // Normals are in camera_init; this viewer always uses identity model transforms.
  mesh.updateMatrixWorld(true);
  check(mesh.matrixWorld.equals(new THREE.Matrix4()), 'direction metadata requires identity world transform');
  let texture, worldModifier, histogram;
  if (support) {
    validateDirectionMetadata(support.metadata, mesh.numSplats);
    check(support.flags.length === mesh.numSplats, 'side flag count mismatch');
    const packed = buildSupportTextureData(plyBytes, support.flags, maxTextureSize);
    histogram = packed.histogram;
    texture = new THREE.DataTexture(packed.data, packed.width, packed.height, THREE.RGBAFormat, THREE.FloatType);
    texture.internalFormat = 'RGBA32F'; texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter; texture.generateMipmaps = false;
    texture.flipY = false; texture.needsUpdate = true;
    const map = dyno.dynoSampler2D(texture), count = dyno.dynoInt(mesh.numSplats);
    const grazing = dyno.dynoFloat(support.metadata.render_grazing_cosine);
    worldModifier = dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => ({
      gsplat: new dyno.Dyno({
        inTypes: { gsplat: dyno.Gsplat, map: 'sampler2D', camera: 'vec3', count: 'int', grazing: 'float' },
        outTypes: { gsplat: dyno.Gsplat },
        inputs: { gsplat, map, camera: mesh.context.viewToWorld.translate, count, grazing },
        statements: ({ inputs: i, outputs: o }) => [
          `${o.gsplat} = ${i.gsplat};`,
          `if (${i.gsplat}.index >= 0 && ${i.gsplat}.index < ${i.count}) {`,
          `  int supportWidth = textureSize(${i.map}, 0).x;`,
          `  vec4 support = texelFetch(${i.map}, ivec2(${i.gsplat}.index % supportWidth, ${i.gsplat}.index / supportWidth), 0);`,
          `  vec3 ray = ${i.camera} - ${i.gsplat}.center;`,
          '  float rayLength = length(ray);',
          '  if (rayLength > 1e-10) {',
          '    float cosine = dot(support.xyz, ray / rayLength);',
          `    if ((support.w == 1.0 && cosine < -${i.grazing}) || (support.w == 2.0 && cosine > ${i.grazing})) ${o.gsplat}.rgba.a = 0.0;`,
          '  }', '}',
        ],
      }).outputs.gsplat,
    }));
  }
  const surfaceModifier = new dyno.Dyno({ inTypes: { gsplat: dyno.Gsplat }, outTypes: { gsplat: dyno.Gsplat },
    statements: ({ inputs: i, outputs: o }) => [`${o.gsplat} = ${i.gsplat};`, `${o.gsplat}.scales.z = 0.0;`] });
  mesh.objectModifiers = [...(mesh.objectModifiers || []), surfaceModifier];
  const wasViewDependent = mesh.enableViewToWorld;
  if (worldModifier) {
    mesh.worldModifiers = [...(mesh.worldModifiers || []), worldModifier];
    // Required even for SH0: moving camera must regenerate the alpha accumulator.
    mesh.enableViewToWorld = true;
  }
  mesh.updateGenerator();
  return {
    diagnostics: { mode: 'inria_2dgs_v1', zeroScaleAxis: 2, directionSupport: Boolean(support),
      normalSource: support ? 'raw_PLY_normalized_wxyz_local_z' : null,
      metadataSha256: support?.metadataSha256 ?? null, plySha256: support?.plySha256 ?? null,
      renderGrazingCosine: support?.metadata.render_grazing_cosine ?? null,
      observedCosineMin: support?.metadata.observed_cosine_min ?? null,
      histogram: histogram ?? null, textureBytes: texture?.image.data.byteLength ?? 0,
      nativeParityMeasured: false },
    dispose() {
      mesh.objectModifiers = mesh.objectModifiers?.filter(m => m !== surfaceModifier);
      if (worldModifier) {
        mesh.worldModifiers = mesh.worldModifiers?.filter(m => m !== worldModifier);
        mesh.enableViewToWorld = wasViewDependent;
      }
      texture?.dispose(); mesh.updateGenerator();
    },
  };
}

/** Exact NS/OpenGL camera pose + full off-axis K, with image-edge coordinates. */
export function applyNativeCamera(THREE, camera, record) {
  const c = record.camera;
  check(c && ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h'].every(k => Number.isFinite(c[k])), 'native camera intrinsics missing');
  check(c.fl_x > 0 && c.fl_y > 0 && Number.isInteger(c.w) && c.w > 0 && Number.isInteger(c.h) && c.h > 0,
    'invalid native camera dimensions/focal length');
  const p = c.camera_to_worlds_ns;
  check(Array.isArray(p) && p.length === 3 && p.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite)),
    'requires 3x4 NS/OpenGL camera_to_worlds_ns');
  const matrix = new THREE.Matrix4().set(...p.flat(), 0, 0, 0, 1);
  const rot = new THREE.Matrix3().setFromMatrix4(matrix);
  const gram = rot.clone().transpose().multiply(rot).elements;
  check(gram.every((v, i) => Math.abs(v - (i % 4 === 0 ? 1 : 0)) < 1e-4) && Math.abs(rot.determinant() - 1) < 1e-4,
    'native pose is not a rigid rotation');
  // Keep provided float32 matrix exactly, without Euler/lookAt reconstruction.
  camera.matrixAutoUpdate = false; camera.matrix.copy(matrix); camera.matrixWorld.copy(matrix);
  camera.matrixWorldInverse.copy(matrix).invert(); camera.matrixWorldNeedsUpdate = true;
  matrix.decompose(camera.position, camera.quaternion, camera.scale);
  camera.aspect = c.w / c.h; camera.fov = 2 * Math.atan(c.h / (2 * c.fl_y)) * 180 / Math.PI;
  // Native near-plane cull is 0.2m. Far culling remains a known backend difference.
  camera.near = 0.2; camera.far = 100;
  const n = camera.near, f = camera.far;
  camera.projectionMatrix.set(2 * c.fl_x / c.w, 0, 1 - 2 * c.cx / c.w, 0,
    0, 2 * c.fl_y / c.h, 2 * c.cy / c.h - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n), 0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  return { width: c.w, height: c.h, cameraConvention: 'NS/OpenGL right/up/back', pixelCenter: 0.5 };
}

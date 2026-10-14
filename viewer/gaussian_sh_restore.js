// Spark 2.3.1 ExtSplats SH repair; call restoreGaussianSh(mesh, fileBytes)
// after awaiting mesh.initialized, using the same unmodified PLY bytes.
// Uses public extra arrays, texture.needsUpdate and mesh.updateGenerator;
// no private methods, GPU access, source edits or geometry re-encoding.
const SCALAR_BYTES = new Map([
  ['char', 1], ['int8', 1], ['uchar', 1], ['uint8', 1],
  ['short', 2], ['int16', 2], ['ushort', 2], ['uint16', 2],
  ['int', 4], ['int32', 4], ['uint', 4], ['uint32', 4],
  ['float', 4], ['float32', 4], ['double', 8], ['float64', 8],
]);
const FLOAT32 = new Set(['float', 'float32']);
const BANDS = [
  ['sh1', 'sh1Texture'], ['sh2', 'sh2Texture'],
  ['sh3a', 'sh3TextureA'], ['sh3b', 'sh3TextureB'],
];
function require(condition, message) {
  if (!condition) throw new Error(`Gaussian SH restore: ${message}`);
}

function schema(fileBytes) {
  const bytes = fileBytes instanceof ArrayBuffer ? new Uint8Array(fileBytes) : fileBytes;
  require(bytes instanceof Uint8Array, 'expected ArrayBuffer or Uint8Array PLY bytes');
  const decoder = new TextDecoder('ascii');
  const lines = [];
  let start = 0, headerBytes = 0;
  for (let i = 0; i < Math.min(bytes.length, 1048576); i++) {
    if (bytes[i] !== 10) continue;
    const line = decoder.decode(bytes.subarray(start, i)).trim();
    lines.push(line);
    start = i + 1;
    if (line === 'end_header') { headerBytes = start; break; }
  }
  require(headerBytes > 0 && lines[0] === 'ply', 'missing/bounded PLY header');
  require(lines[1] === 'format binary_little_endian 1.0', 'requires binary_little_endian 1.0');
  const elements = [];
  let element;
  for (const line of lines.slice(2, -1)) {
    const tokens = line.split(/\s+/);
    if (!line || tokens[0] === 'comment' || tokens[0] === 'obj_info') continue;
    if (tokens[0] === 'element') {
      require(tokens.length === 3 && /^\d+$/.test(tokens[2]), 'invalid element/count');
      const count = Number(tokens[2]);
      require(Number.isSafeInteger(count), 'unsafe element count');
      require(!elements.some(e => e.name === tokens[1]), 'duplicate element');
      element = { name: tokens[1], count, stride: 0, fields: new Map() };
      elements.push(element);
    } else if (tokens[0] === 'property') {
      require(element && tokens.length === 3 && SCALAR_BYTES.has(tokens[1]),
        'only declared fixed-width scalar properties are supported (no lists)');
      require(!element.fields.has(tokens[2]), 'duplicate property');
      element.fields.set(tokens[2], { type: tokens[1], offset: element.stride });
      element.stride += SCALAR_BYTES.get(tokens[1]);
    } else require(false, `unsupported header directive ${tokens[0]}`);
  }
  const vertex = elements.find(e => e.name === 'vertex');
  require(vertex && vertex.count > 0 && vertex.stride > 0, 'missing/nonempty vertex schema');
  let payloadEnd = headerBytes;
  for (const e of elements) {
    e.dataOffset = payloadEnd;
    payloadEnd += e.count * e.stride;
    require(Number.isSafeInteger(payloadEnd) && payloadEnd <= bytes.byteLength,
      'truncated or unsafe declared payload');
  }
  require(payloadEnd === bytes.byteLength, 'bytes differ from declared scalar payload');
  const rest = [...vertex.fields.keys()].filter(name => name.startsWith('f_rest_'));
  const degree = new Map([[0, 0], [9, 1], [24, 2], [45, 3]]).get(rest.length);
  require(degree !== undefined, 'SH must declare 0/9/24/45 coefficients');
  const offsets = rest.map((_, i) => {
    const field = vertex.fields.get(`f_rest_${i}`);
    require(field && FLOAT32.has(field.type), 'requires contiguous float32 f_rest_ indices');
    return field.offset;
  });
  const xyz = ['x', 'y', 'z'].map(name => {
    const field = vertex.fields.get(name);
    require(field && FLOAT32.has(field.type), 'requires float32 centers for record identity');
    return field.offset;
  });
  return { ...vertex, bytes, degree, offsets, xyz,
    view: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) };
}

/**
 * Repack source float32 SH into initialized, non-LoD ExtSplats, preserving
 * arrays/padding, degree, maxSh, all extArrays fields and source bytes.
 * For each signed RGB triplet, R = 2^max(-15,ceil(log2(maxAbs))).
 * Ideal decode error per component <= R/510; GLSL float32 decode error
 * <= R/510 + R*(2^-23 + 2^-48). Reject nonfinite or |coefficient| > 65536;
 * this 5-bit exponent format cannot represent larger values safely.
 * All validation/encoding is staged before any mutation. Throws on mismatch.
 */
export function restoreGaussianSh(mesh, fileBytes) {
  const source = schema(fileBytes);
  const splats = mesh?.extSplats;
  require(mesh?.isInitialized === true && splats?.isInitialized === true,
    'await mesh.initialized before restoring SH');
  require(mesh.splats === splats && !mesh.paged && !splats.lodSplats && !splats.lod,
    'requires original non-LoD ExtSplats record order');
  require(typeof mesh.updateGenerator === 'function' && typeof splats.getNumSh === 'function',
    'unsupported public mesh/ExtSplats API');
  require(splats.numSplats === source.count && mesh.numSplats === source.count,
    'PLY count differs from initialized mesh');
  require(splats.getNumSh() === source.degree, 'source/generated SH degrees differ');
  require(Array.isArray(splats.extArrays) && splats.extArrays.length === 2 &&
    splats.extArrays.every(a => a instanceof Uint32Array && a.length >= source.count * 4),
  'unsupported ExtSplats geometry arrays');
  const bandCount = source.degree === 3 ? 4 : source.degree;
  const textures = new Set();
  const staged = BANDS.slice(0, bandCount).map(([key, textureKey]) => {
    const array = splats.extra[key];
    require(array instanceof Uint32Array && array.length >= source.count * 4,
      `unsupported/undersized ${key} word array`);
    require(array.buffer !== source.bytes.buffer &&
      splats.extArrays.every(a => a.buffer !== array.buffer), 'SH/geometry/source buffers alias');
    const wrapper = splats.extra[textureKey];
    if (wrapper !== undefined) {
      const texture = wrapper?.value, data = texture?.image?.data;
      require(texture?.isDataArrayTexture === true && ArrayBuffer.isView(data) &&
        data.buffer === array.buffer && data.byteOffset === array.byteOffset &&
        data.byteLength === array.byteLength, `cached ${textureKey} does not alias its SH array`);
      textures.add(texture);
    }
    return { array, words: array.slice(0, source.count * 4) };
  });
  require(new Set(staged.map(b => b.array.buffer)).size === staged.length,
    'SH band buffers alias');
  const perChannel = source.offsets.length / 3;
  const stats = { vertices: source.count, degree: source.degree,
    coefficients: source.offsets.length * source.count, changedWords: 0,
    maxAbsoluteError: 0, maxQuantizationErrorBound: 0, maxFloat32DecodeErrorBound: 0,
    texturesMarkedDirty: 0 };
  for (let i = 0; i < source.count; i++) {
    const base = source.dataOffset + i * source.stride;
    for (let axis = 0; axis < 3; axis++) require(
      source.view.getUint32(base + source.xyz[axis], true) === splats.extArrays[0][i * 4 + axis],
      `source center/order differs at vertex ${i}`);
    for (let k = 0; k < perChannel; k++) {
      const rgb = [0, 1, 2].map(c => source.view.getFloat32(
        base + source.offsets[c * perChannel + k], true));
      require(rgb.every(Number.isFinite), `nonfinite SH at vertex ${i}`);
      const maxAbs = Math.max(...rgb.map(Math.abs));
      require(maxAbs <= 65536, `SH exceeds ExtSplats exponent range at vertex ${i}`);
      const exponent = maxAbs === 0 ? -15 : Math.max(-15, Math.ceil(Math.log2(maxAbs)));
      const range = 2 ** exponent, step = range / 255;
      const magnitude = rgb.map(v => Math.round(Math.abs(v) / step));
      require(magnitude.every(v => v <= 255), 'unrepresentable SH magnitude');
      const signs = (rgb[0] < 0 ? 1 : 0) | (rgb[1] < 0 ? 2 : 0) | (rgb[2] < 0 ? 4 : 0);
      const word = (magnitude[0] | magnitude[1] << 8 | magnitude[2] << 16 |
        (((exponent + 15) << 3) | signs) << 24) >>> 0;
      const band = k < 4 ? 0 : k < 8 ? 1 : k < 12 ? 2 : 3;
      const offset = i * 4 + (k % 4);
      stats.changedWords += Number(staged[band].array[offset] !== word);
      staged[band].words[offset] = word;
      for (let c = 0; c < 3; c++) stats.maxAbsoluteError = Math.max(stats.maxAbsoluteError,
        Math.abs(magnitude[c] * step * (rgb[c] < 0 ? -1 : 1) - rgb[c]));
      stats.maxQuantizationErrorBound = Math.max(stats.maxQuantizationErrorBound, step / 2);
      stats.maxFloat32DecodeErrorBound = Math.max(stats.maxFloat32DecodeErrorBound,
        step / 2 + range * (2 ** -23 + 2 ** -48));
    }
  }
  if (stats.changedWords > 0) {
    for (const band of staged) band.array.set(band.words);
    for (const texture of textures) texture.needsUpdate = true;
    mesh.updateGenerator(); // Next frame increments mesh.version and regenerates accumulator RGB.
    stats.texturesMarkedDirty = textures.size;
  }
  return stats;
}

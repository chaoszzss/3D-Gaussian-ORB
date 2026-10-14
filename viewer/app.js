import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { SparkRenderer, SplatMesh, dyno } from '@sparkjsdev/spark';
import { restoreGaussianSh } from './gaussian_sh_restore.js';

window.__garageStarted = true;
const $ = (s) => document.querySelector(s);
const state = { manifest: null, loaded: new Map(), lidar: new Map(), busy: false, renderer: null,
  scene: null, camera: null, controls: null, spark: null, helpers: null, floor: 'all', focused: null, view: 'oblique',
  surfaceTools: null, rendererProfile: null, parityFrame: null };

function status(message, error = false) {
  $('#status').textContent = message;
  $('#status').classList.toggle('error', error);
}
function el(tag, text, className) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (className) n.className = className;
  return n;
}
function button(text, action, disabled = false) {
  const n = el('button', text);
  n.type = 'button'; n.disabled = disabled || state.busy;
  n.addEventListener('click', action);
  return n;
}
function floorKey(b) { return b.floor_id === null ? 'unknown' : 'floor:' + b.floor_id; }
function matches(b) { return state.floor === 'all' || floorKey(b) === state.floor; }
function visibleBlocks() { return (state.manifest?.blocks || []).filter(matches); }
function blockById(id) { return state.manifest.blocks.find(b => b.id === id); }
function usage() {
  let splats = 0, bytes = 0;
  for (const entry of state.loaded.values()) { splats += entry.count; bytes += entry.bytes; }
  return { blocks: state.loaded.size, splats, bytes };
}
function syncVisibility() {
  for (const [id, entry] of state.loaded) entry.mesh.visible = entry.visible && matches(blockById(id));
  for (const [id, entry] of state.lidar) entry.points.visible = entry.visible && matches(blockById(id));
  $('#empty').hidden = [...state.loaded.values()].some(x => x.mesh.visible) || [...state.lidar.values()].some(x => x.points.visible);
  $('#empty').style.display = $('#empty').hidden ? 'none' : 'flex';
}

function renderList() {
  if (!state.manifest) return;
  const list = $('#blocks'); list.replaceChildren();
  for (const b of visibleBlocks()) {
    const entry = state.loaded.get(b.id), asset = b.asset;
    const card = el('article', undefined, 'block-card' + (entry ? ' loaded' : ''));
    card.dataset.blockId = b.id;
    const head = el('div', undefined, 'block-head');
    head.append(el('span', b.label || b.id, 'block-title'), el('span', b.floor_id || '楼层未核验', 'badge'));
    card.append(head);
    const info = el('div', undefined, 'block-info');
    info.append(el('p', asset.available ? `${asset.vertex_count.toLocaleString()} 个高斯 · SH ${asset.sh_degree} 阶 · ${(asset.file_bytes / 1048576).toFixed(1)} MB` : asset.reason));
    const labels = { trained: '已训练', prepared: '数据已准备', training: '正在生成', paused: '已暂停', pending: '等待训练', failed: '训练失败', synthetic: '合成自检', evaluated: '已评估', whole_map_evaluated: '整图已评估', not_evaluated: '尚未评估' };
    info.append(el('p', `训练：${labels[b.training.status] || '未声明'}　评估：${labels[b.eval.status] || '未声明'}`));
    if (b.eval.status === 'whole_map_evaluated') info.append(el('p', b.eval.display_kind === 'overview' ? '评估对象：完整高斯模型；当前为精简总览' : '评估对象：完整高斯模型；当前为局部显示'));
    if (Number.isFinite(b.eval.psnr_db)) info.append(el('p', `留出 PSNR：${b.eval.psnr_db.toFixed(2)} dB（清单声明）`));
    if (b.floor_verified !== true && b.floor_id !== null) info.append(el('p', '楼层索引待核验'));
    if (state.rendererProfile) info.append(el('p', b.direction_support ? '表面模型 · 按训练观测方向显示' : '表面模型 · 双面显示'));
    card.append(info);
    const actions = el('div', undefined, 'block-actions');
    actions.append(button(entry ? '卸载' : '加载', () => transact(async () => {
      if (state.loaded.has(b.id)) { unload(b.id); status('区块已卸载'); }
      else { await loadBlock(b.id); focus(b); status('区块已加载，坐标与尺度保持原样'); }
    }), !entry && !asset.available));
    actions.append(button('定位', () => { focus(b); status('观察中心已定位到所选区块'); }));
    const label = el('label', undefined);
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = Boolean(entry?.visible);
    check.disabled = !entry || state.busy; check.dataset.action = 'visibility';
    check.addEventListener('change', () => { entry.visible = check.checked; syncVisibility(); renderList(); });
    label.append(check, document.createTextNode('显示')); actions.append(label); card.append(actions);
    if (b.lidar) {
      const l = state.lidar.get(b.id), bar = el('div', undefined, 'lidar-control');
      bar.append(el('span', 'LiDAR 几何辅层'));
      bar.append(button(l ? '卸载辅层' : '加载辅层', () => transact(async () => {
        if (l) unloadLidar(b.id); else await loadLidar(b);
        status(l ? 'LiDAR 辅层已卸载' : 'LiDAR 点云辅层已加载');
      }), !l && !b.lidar.asset.available));
      card.append(bar);
    }
    list.append(card);
  }
  if (!visibleBlocks().length) list.append(el('p', '当前索引下没有可列出的区块。', 'note'));
  const u = usage(), limits = state.manifest.limits;
  $('#budget').textContent = `已加载 ${u.blocks} / ${limits.max_loaded_blocks} 块 · ${(u.splats / 10000).toFixed(1)} / ${(limits.max_splats / 10000).toFixed(0)} 万高斯`;
  $('#nearby').disabled = state.busy || !state.renderer || !visibleBlocks().some(b => b.asset.available);
  $('#refresh').disabled = state.busy;
  $('#floor').disabled = state.busy;
  syncReference();
}

async function transact(action) {
  if (state.busy) return;
  state.busy = true; $('#refresh').disabled = true; renderList();
  try { await action(); }
  catch (e) { if (!e.userMessage) console.error(e); status(e.userMessage || '加载失败，请检查模型格式、文件完整性或浏览器图形能力。', true); }
  finally { state.busy = false; $('#refresh').disabled = false; syncVisibility(); renderList(); }
}
function fail(message) { const e = new Error(message); e.userMessage = message; throw e; }

async function fetchBytes(url, maxBytes) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) fail('模型尚未就绪或不符合清单协议，请刷新清单后重试。');
  const size = Number(response.headers.get('Content-Length'));
  if (!size || size > maxBytes) { await response.body?.cancel(); fail('文件超出本次加载预算，请先卸载区块或进一步分块。'); }
  // Bounded single allocation; do not keep an additional list of chunks.
  const data = new Uint8Array(size), reader = response.body.getReader();
  let offset = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (offset + value.length > size) fail('模型在读取时发生变化，请刷新后重试。');
      data.set(value, offset); offset += value.length;
      status(`正在读取所选区块… ${Math.floor(offset / size * 100)}%`);
    }
    if (offset !== size) fail('模型文件不完整，请等待模型导出完成。');
    return data.buffer;
  } catch (e) { await reader.cancel(); throw e; }
  finally { reader.releaseLock(); }
}

async function loadBlock(id) {
  if (!state.renderer) fail('浏览器图形环境不可用，无法加载三维模型。');
  if (state.loaded.has(id)) { state.loaded.get(id).visible = true; return; }
  // Refresh preflight before each allocation; trainers should publish atomically.
  const response = await fetch('/api/manifest', { cache: 'no-store' });
  if (!response.ok) fail('清单暂时不可用，请稍后刷新。');
  const fresh = await response.json(), b = fresh.blocks.find(x => x.id === id);
  if (!b?.asset.available) fail('所选区块尚未就绪，请刷新清单。');
  if (JSON.stringify(fresh.renderer) !== JSON.stringify(state.manifest.renderer)) fail('模型渲染约定已变化，请刷新清单。');
  if (b.direction_support !== undefined && !state.rendererProfile) fail('方向支持模型需要显式表面渲染约定，请刷新清单。');
  const u = usage(), limits = state.manifest.limits;
  if (u.blocks >= limits.max_loaded_blocks || u.splats + b.asset.vertex_count > limits.max_splats || u.bytes + b.asset.file_bytes > limits.max_file_bytes) {
    fail('当前区块总量达到预算，请先卸载区块。隐藏区块仍占内存。');
  }
  status('正在加载 Gaussian 区块…');
  const bytes = await fetchBytes(b.asset.url, Math.min(limits.max_file_bytes - u.bytes, b.asset.file_bytes));
  let mesh, surface;
  try {
    const support = b.direction_support === undefined ? null : await state.surfaceTools.loadDirectionSupport({
      descriptor: b.direction_support, blockId: id, plyBytes: bytes, count: b.asset.vertex_count,
      fetchBytes, baseURL: window.location.href,
    });
    // True Gaussian decoder/renderer. Identity world transform; float32 metric centers.
    mesh = new SplatMesh({ fileBytes: bytes, extSplats: true, lod: false, enableLod: false, raycastable: false, editable: false });
    await mesh.initialized;
    const count = mesh.splats?.numSplats ?? mesh.extSplats?.numSplats;
    if (count !== b.asset.vertex_count || u.splats + count > limits.max_splats) fail('模型在预检后发生变化，请刷新清单后重试。');
    const shRestore = restoreGaussianSh(mesh, bytes);
    if (state.rendererProfile) surface = state.surfaceTools.attachSurfaceModifiers({ THREE, dyno, mesh, support,
      plyBytes: bytes, maxTextureSize: state.renderer.capabilities.maxTextureSize });
    mesh.maxSh = 3; mesh.updateGenerator();
    state.scene.add(mesh);
    const plySha256 = support?.plySha256 ?? (state.rendererProfile ? await state.surfaceTools.sha256(bytes) : null);
    state.loaded.set(id, { mesh, count, bytes: b.asset.file_bytes, visible: true, shRestore, surface, plySha256 });
  } catch (e) { surface?.dispose(); mesh?.removeFromParent(); mesh?.dispose(); throw e; }
}
function unload(id) {
  const entry = state.loaded.get(id);
  if (entry) { entry.surface?.dispose(); entry.mesh.removeFromParent(); entry.mesh.dispose(); state.loaded.delete(id); }
  unloadLidar(id);
}
async function loadLidar(b) {
  if (!state.renderer) fail('浏览器图形环境不可用。');
  const limits = state.manifest.limits;
  let count = 0, bytes = 0;
  for (const e of state.lidar.values()) { count += e.count; bytes += e.bytes; }
  if (state.lidar.size >= limits.max_loaded_blocks || count + b.lidar.asset.vertex_count > limits.max_lidar_points || bytes + b.lidar.asset.file_bytes > limits.max_lidar_bytes) fail('LiDAR 辅层达到单独预算，请先卸载辅层。');
  const data = await fetchBytes(b.lidar.asset.url, Math.min(limits.max_lidar_bytes - bytes, b.lidar.asset.file_bytes));
  const geometry = new PLYLoader().parse(data);
  const position = geometry.getAttribute('position');
  if (!position || position.count !== b.lidar.asset.vertex_count || count + position.count > limits.max_lidar_points) { geometry.dispose(); fail('LiDAR 辅层数量与清单不符。'); }
  const material = new THREE.PointsMaterial({ color: '#82d4ff', size: 0.035, sizeAttenuation: true, transparent: true, opacity: 0.65, depthWrite: false });
  const points = new THREE.Points(geometry, material); points.name = 'LiDAR geometry auxiliary layer';
  state.scene.add(points);
  state.lidar.set(b.id, { points, visible: true, count: position.count, bytes: b.lidar.asset.file_bytes });
}
function unloadLidar(id) {
  const entry = state.lidar.get(id);
  if (entry) { entry.points.removeFromParent(); entry.points.geometry.dispose(); entry.points.material.dispose(); state.lidar.delete(id); }
}

function boxFor(blocks) {
  const box = new THREE.Box3();
  for (const b of blocks) box.union(new THREE.Box3(new THREE.Vector3(...b.bbox_m.min), new THREE.Vector3(...b.bbox_m.max)));
  return box;
}
function focus(b) {
  state.focused = b.id;
  if (state.renderer) setView(state.view, boxFor([b]));
  syncReference();
}
function syncReference() {
  const b = state.focused && blockById(state.focused);
  const enabled = b && state.loaded.get(b.id)?.mesh.visible && matches(b) && b.camera_views?.length && !state.busy;
  $('#reference').disabled = !enabled;
  $('#reference-view').disabled = !enabled;
}
function enterReference() {
  const b = state.focused && blockById(state.focused);
  const view = b?.camera_views?.[Number($('#reference-view').value)];
  if (!view || !state.loaded.has(b.id) || !state.renderer) return;
  exitParityFrame();
  const matrix = new THREE.Matrix4().set(...view.camera_to_world.flat());
  state.camera.position.setFromMatrixPosition(matrix);
  state.camera.up.setFromMatrixColumn(matrix, 1).normalize();
  const direction = new THREE.Vector3().setFromMatrixColumn(matrix, 2).negate().normalize();
  state.controls.target.copy(state.camera.position).addScaledVector(direction, 5);
  state.camera.fov = view.fov_y_deg; state.camera.updateProjectionMatrix();
  state.controls.update();
  status('已进入所选区域；可以旋转、平移和缩放观察');
}
function fitBox() {
  let blocks = visibleBlocks().filter(b => state.loaded.get(b.id)?.mesh.visible || state.lidar.get(b.id)?.points.visible);
  if (!blocks.length && state.focused) blocks = visibleBlocks().filter(b => b.id === state.focused);
  if (!blocks.length) blocks = visibleBlocks();
  return boxFor(blocks);
}
function setView(view, box = fitBox()) {
  if (!state.renderer) return;
  exitParityFrame();
  state.view = view;
  state.camera.fov = 50;
  if (box.isEmpty()) box = new THREE.Box3(new THREE.Vector3(-5, -5, -1), new THREE.Vector3(5, 5, 3));
  const center = box.getCenter(new THREE.Vector3()), radius = Math.max(1, box.getSize(new THREE.Vector3()).length() * 0.5);
  const up = new THREE.Vector3(...state.manifest.camera_up).normalize();
  const front = Math.abs(up.dot(new THREE.Vector3(0, -1, 0))) > 0.95 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, -1, 0);
  front.addScaledVector(up, -front.dot(up)).normalize();
  const side = new THREE.Vector3().crossVectors(front, up).normalize();
  const directions = { top: up.clone().addScaledVector(front, 0.0001), front, side,
    oblique: front.clone().addScaledVector(side, 0.8).addScaledVector(up, 0.7) };
  const fov = THREE.MathUtils.degToRad(state.camera.fov / 2), aspect = Math.min(1, state.camera.aspect);
  const distance = radius / Math.sin(Math.atan(Math.tan(fov) * aspect)) * 1.15;
  state.camera.position.copy(center).addScaledVector(directions[view].normalize(), distance);
  state.camera.up.copy(up); state.camera.far = Math.max(2000, distance * 20); state.camera.updateProjectionMatrix();
  state.controls.target.copy(center); state.controls.update();
}
async function nearest() {
  const target = state.controls.target;
  const nearestBlocks = visibleBlocks().filter(b => b.asset.available).sort((a, b) => {
    const da = boxFor([a]).distanceToPoint(target), db = boxFor([b]).distanceToPoint(target);
    return da - db || a.id.localeCompare(b.id);
  }).slice(0, Number($('#near-count').value));
  // This explicit action replaces the working set, retaining only nearby selections.
  for (const id of [...state.loaded.keys()]) if (!nearestBlocks.some(b => b.id === id)) unload(id);
  let total = 0;
  for (const b of nearestBlocks) {
    await loadBlock(b.id); total++;
  }
  syncVisibility(); setView(state.view);
  status(`已加载 ${total} 个近邻区块；距离按观察中心到包围盒计算`);
}

function initRenderer() {
  const host = $('#viewport');
  try {
    const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: false });
    state.renderer = renderer;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setClearColor('#101923');
    state.scene = new THREE.Scene();
    state.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2000);
    state.camera.up.set(...state.manifest.camera_up).normalize();
    host.append(renderer.domElement);
    state.controls = new OrbitControls(state.camera, renderer.domElement);
    state.controls.enableDamping = true; state.controls.dampingFactor = 0.12;
    state.controls.minDistance = 0.15; state.controls.maxDistance = 20000;
    state.spark = new SparkRenderer({ renderer, ...(state.rendererProfile?.spark || {}) });
    configureCompositor();
    state.scene.add(state.spark);
    state.helpers = new THREE.Group();
    const grid = new THREE.GridHelper(100, 100, '#426164', '#253b48');
    grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...state.manifest.camera_up).normalize());
    state.helpers.add(grid, new THREE.AxesHelper(2)); state.scene.add(state.helpers);
    new ResizeObserver(() => {
      if (state.parityFrame) return; // Exact native viewport must not follow CSS resizing.
      const w = host.clientWidth, h = host.clientHeight;
      renderer.setSize(w, h, false); state.camera.aspect = w / Math.max(1, h); state.camera.updateProjectionMatrix();
    }).observe(host);
    renderer.domElement.addEventListener('webglcontextlost', (event) => { event.preventDefault(); renderer.setAnimationLoop(null); status('图形上下文丢失，请刷新页面并减少加载区块。', true); });
    renderer.setAnimationLoop(() => {
      if (!state.parityFrame) state.controls.update();
      renderFrame();
    });
    // Set aspect before the first focus, including narrow viewports.
    state.camera.aspect = host.clientWidth / Math.max(1, host.clientHeight);
    setView('oblique');
  } catch (e) {
    console.error(e); state.renderer?.dispose(); state.renderer = null;
    status('无法启动三维渲染，请使用支持 WebGL2 的本地浏览器。', true);
  }
}

async function refresh(initial = false) {
  const response = await fetch('/api/manifest', { cache: 'no-store' });
  if (!response.ok) fail('模型清单格式无效，请检查工程文档中的清单协议。');
  const data = await response.json();
  // Lazy import leaves existing maps independent of the new sidecar routes.
  let rendererProfile = null;
  if (data.renderer !== undefined) {
    state.surfaceTools ??= await import('./gaussian_direction_support.js');
    rendererProfile = state.surfaceTools.rendererConfig(data);
  }
  if (!rendererProfile && data.blocks.some(b => b.direction_support !== undefined)) fail('方向支持模型缺少表面渲染约定。');
  // Release before accepting a new manifest so IDs never point to stale assets.
  for (const id of [...state.loaded.keys()]) unload(id);
  for (const id of [...state.lidar.keys()]) unloadLidar(id);
  state.manifest = data;
  state.rendererProfile = rendererProfile;
  const completed = data.blocks.filter(b => b.training.status === 'trained' && b.asset.available).length;
  const total = data.blocks.length, progress = $('#map-progress');
  progress.max = Math.max(1, total); progress.value = completed;
  $('#map-progress-text').textContent = data.dataset_kind === 'synthetic' ? '合成自检，不计入车库地图进度' :
    `可查看区域 ${completed} / ${total}${data.status === 'paused' ? ' · 已暂停' : completed < total ? ' · 地图生成中' : total ? ' · 全部区域已生成' : ''}`;
  const floor = $('#floor'); floor.replaceChildren(el('option', '全部楼层')); floor.firstChild.value = 'all';
  const ids = [...new Set(data.blocks.map(floorKey))].sort();
  for (const id of ids) { const n = el('option', id === 'unknown' ? '楼层未核验' : id.slice(6)); n.value = id; floor.append(n); }
  if (!ids.includes(state.floor)) state.floor = 'all'; floor.value = state.floor;
  const banner = $('#banner'), synthetic = data.dataset_kind === 'synthetic';
  banner.className = 'banner' + (synthetic ? ' synthetic' : data.blocks.some(b => b.asset.available) ? ' real' : '');
  banner.textContent = synthetic ? '合成自检模型 · 非实景 · 仅验证 Gaussian 渲染与交互，不代表车库重建结果。' : data.blocks.some(b => b.asset.available) ? '实景模型以清单声明为准；模型可读不代表几何、颜色或标定精度已验收。' : '当前没有训练好的实景模型；等待清单中的模型就绪。';
  $('#empty p').textContent = synthetic ? '此处仅展示显式生成的合成自检模型。选择区块后可验证三维交互。' : '实景训练模型就绪后，可按楼层选择和加载。';
  if (initial) initRenderer();
  else if (state.renderer) {
    state.compositor?.dispose(); state.compositor = null;
    state.spark.removeFromParent(); state.spark.dispose();
    state.spark = new SparkRenderer({ renderer: state.renderer, ...(rendererProfile?.spark || {}) });
    configureCompositor();
    state.scene.add(state.spark);
    state.camera.up.set(...data.camera_up).normalize();
    state.helpers.children[0].quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), state.camera.up);
    setView(state.view);
  }
  renderList(); syncVisibility();
  if (state.renderer) status(data.blocks.length ? '已读取区块索引；按需选择加载' : '尚无训练模型，当前仅显示米制参考网格');
  if (synthetic) {
    // Test-only introspection: no diagnostic controls in the product UI.
    window.__garageTest = {
      snapshot: () => ({ ...usage(), floor: state.floor, target: state.controls?.target.toArray(),
        meshes: [...state.loaded].map(([id, e]) => ({ id, count: e.count, visible: e.mesh.visible,
          position: e.mesh.position.toArray(), quaternion: e.mesh.quaternion.toArray(), scale: e.mesh.scale.toArray(),
          maxSh: e.mesh.maxSh, encoding: e.mesh.extSplats ? 'ExtSplats' : 'PackedSplats',
          shDegree: e.mesh.splats.getNumSh(), bbox: e.mesh.getBoundingBox().getCenter(new THREE.Vector3()).toArray() })),
        lidar: state.lidar.size, drawCalls: state.renderer?.info.render.calls })
    };
  } else delete window.__garageTest;
  installParityDiagnostics();
}

function exitParityFrame() {
  if (!state.parityFrame) return;
  state.parityFrame = null;
  state.camera.matrixAutoUpdate = true; state.controls.enabled = true;
  state.camera.near = 0.05; state.camera.far = 2000;
  for (const entry of state.loaded.values()) { entry.mesh.maxSh = 3; entry.mesh.updateGenerator(); }
  state.helpers.visible = $('#helpers').checked;
  state.renderer.outputColorSpace = THREE.SRGBColorSpace;
  state.renderer.toneMapping = THREE.NoToneMapping;
  state.renderer.setClearColor('#101923');
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  const host = $('#viewport');
  state.renderer.setSize(host.clientWidth, host.clientHeight, false);
  state.camera.aspect = host.clientWidth / Math.max(1, host.clientHeight);
  state.camera.updateProjectionMatrix();
}

// Only explicit local diagnostic sessions get this API. It fixes inputs for a
// future scheduled browser/native comparison; it does not certify pixel parity.
function configureCompositor() {
  state.nativeAa = state.rendererProfile?.mode === 'inria_2dgs_v3'
    ? state.surfaceTools.installNativeAaShader(state.spark) : null;
  state.compositor = ['inria_2dgs_v2', 'inria_2dgs_v3'].includes(state.rendererProfile?.mode)
    ? state.surfaceTools.createNativeCompositor(THREE, state.renderer, state.spark) : null;
}
function renderFrame() {
  if (state.compositor) state.compositor.render(state.scene, state.camera, state.parityFrame?.background_rgb);
  else state.renderer.render(state.scene, state.camera);
}
function installParityDiagnostics() {
  delete window.__garageParity;
  if (!state.renderer || !state.rendererProfile || !new URLSearchParams(location.search).has('parity')) return;
  window.__garageParity = {
    setNativeFrame(record) {
      const entries = [...state.loaded.values()];
      if (state.busy || entries.length !== 1 || !entries[0].mesh.visible || state.lidar.size) throw new Error('Parity requires one complete visible pinned PLY and no LiDAR layers');
      const entry = entries[0];
      if (entry.plySha256 !== record.source_ply_sha256) throw new Error('Native/browser PLY SHA mismatch');
      const supportSha = entry.surface?.diagnostics.metadataSha256 ?? null;
      if ((record.direction_support_sha256 ?? null) !== supportSha) throw new Error('Native/browser direction metadata SHA mismatch');
      const degree = record.inference_sh_degree;
      if (!Number.isInteger(degree) || degree < 0 || degree > entry.mesh.splats.getNumSh()) throw new Error('Native SH degree unavailable');
      const rgb = record.background_rgb;
      if (!Array.isArray(rgb) || rgb.length !== 3 || !rgb.every(x => Number.isFinite(x) && x >= 0 && x <= 1)) throw new Error('Native background RGB must be explicit');
      const cameraInfo = state.surfaceTools.applyNativeCamera(THREE, state.camera, record);
      entry.mesh.maxSh = degree; entry.mesh.updateGenerator();
      state.controls.enabled = false; state.helpers.visible = false;
      state.renderer.setPixelRatio(1); state.renderer.setSize(cameraInfo.width, cameraInfo.height, false);
      // Spark's RawShaderMaterial already writes supervised sRGB numbers. Its
      // encodeLinear flag becomes true for NON-sRGB targets, so keep an sRGB
      // canvas to avoid converting the splat colors before compositing.
      state.renderer.outputColorSpace = THREE.SRGBColorSpace;
      state.renderer.toneMapping = THREE.NoToneMapping;
      state.renderer.setClearColor(new THREE.Color().setRGB(...rgb, THREE.SRGBColorSpace), 1);
      state.parityFrame = { ...cameraInfo, source_ply_sha256: record.source_ply_sha256,
        direction_support_sha256: supportSha, inference_sh_degree: degree, background_rgb: [...rgb],
        metric_domain: 'browser_canvas_unorm8_pending_comparison', nativeParityMeasured: false };
      return this.snapshot();
    },
    snapshot: () => ({ frame: state.parityFrame, renderer: state.rendererProfile,
      compositing: state.compositor?.diagnostics ?? { target: 'canvas_unorm8' },
      nativeAa: state.nativeAa,
      sorting: state.spark.sorting, sortDirty: state.spark.sortDirty,
      sortedCenter: state.spark.sortedCenter.toArray(), sortedDirection: state.spark.sortedDir.toArray(),
      viewport: state.renderer.getDrawingBufferSize(new THREE.Vector2()).toArray(),
      camera_to_world_column_major: state.camera.matrixWorld.toArray(),
      projection_column_major: state.camera.projectionMatrix.toArray(),
      models: [...state.loaded].map(([id, e]) => ({ id, count: e.count, visible: e.mesh.visible,
        plySha256: e.plySha256, shRestore: e.shRestore, surface: e.surface?.diagnostics })) }),
    capture() {
      if (!state.parityFrame || state.spark.sorting || state.spark.sortDirty) throw new Error('Exact frame not ready');
      renderFrame();
      return state.renderer.domElement.toDataURL('image/png');
    },
    exit: exitParityFrame,
  };
}

$('#refresh').addEventListener('click', () => transact(() => refresh()));
$('#floor').addEventListener('change', () => { state.floor = $('#floor').value; syncVisibility(); renderList(); });
$('#nearby').addEventListener('click', () => transact(nearest));
$('#fit').addEventListener('click', () => setView(state.view));
$('#reference').addEventListener('click', enterReference);
$('#reference-view').addEventListener('change', enterReference);
for (const b of document.querySelectorAll('[data-view]')) b.addEventListener('click', () => setView(b.dataset.view));
$('#helpers').addEventListener('change', () => { if (state.helpers) state.helpers.visible = $('#helpers').checked; });
refresh(true).catch(e => { console.error(e); status(e.userMessage || '本地清单暂时不可用，请检查服务器。', true); });

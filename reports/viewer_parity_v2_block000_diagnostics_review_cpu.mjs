// Bounded CPU verification of the real offline page; no browser or GPU.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const reports = path.join(root, 'reports');
const prefix = 'viewer_parity_v2_block000_diagnostics_review_';
const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const before = read(path.join(reports, prefix + 'before.json'));
const links = read(path.join(reports, prefix + 'links.json'));
const htmlFile = path.join(reports, 'fidelity_v2_visual_review.html');
const html = fs.readFileSync(htmlFile, 'utf8');
assert.equal(sha(htmlFile), links.html_sha256);
for (const [name, expected] of Object.entries(before.protected_sha256)) {
  assert.equal(sha(path.join(root, name)), expected, name);
}
for (const asset of links.assets) assert.equal(sha(path.join(root, asset.path)), asset.sha256, asset.path);
const local = name => {
  assert.ok(!/^[a-z]+:/i.test(name), name);
  const filename = fs.realpathSync(path.resolve(reports, name));
  assert.ok(filename.startsWith(root + path.sep), name);
  assert.ok(fs.statSync(filename).isFile(), name);
};
for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) local(match[1]);

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.checked = false; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
}
const elements = new Map(['block', 'model', 'frame', 'panels', 'geometry', 'support', 'identity', 'error', 'prev', 'next', 'pending']
  .map(id => [`#${id}`, new Element(id)]));
const context = vm.createContext({ document: {
  querySelector: selector => { assert.ok(elements.has(selector), selector); return elements.get(selector); },
  createElement: tag => new Element(tag),
} });
vm.runInContext(html.match(/<script>([\s\S]+)<\/script>/)[1], context);
const data = JSON.parse(vm.runInContext('JSON.stringify(data)', context));
assert.deepEqual(data.blocks.map(b => b.id), ['block_015', 'block_000']);
assert.deepEqual(data.blocks.map(b => b.models.length), [5, 7]);
for (const old of before.old_blocks) {
  const actual = data.blocks.find(b => b.id === old.id);
  assert.deepEqual(actual.models.slice(0, old.models.length), old.models, 'old model options must be preserved');
  assert.deepEqual(actual.geometry.slice(0, old.geometry.length), old.geometry, 'old geometry must be preserved');
  assert.deepEqual(actual.denominator, old.denominator);
}
assert.equal(data.pending.length, 1);
assert.deepEqual(Object.keys(data.pending[0]).sort(), ['block_id', 'id', 'message', 'status']);
assert.equal(data.pending[0].id, 'block000_measuredseed_anneal_v1');
assert.equal(data.pending[0].status, 'pending');
assert.ok(!links.assets.some(asset => asset.path.includes(data.pending[0].id)), 'pending must not invent links');

const diagnostics = ['measuredseed_cap003_diagnostic', 'original_cap003_diagnostic', 'opacity600_control', 'opacity600_soft09'];
for (const [index, name] of diagnostics.entries()) {
  const model = data.blocks[1].models[3 + index], geo = data.blocks[1].geometry[3 + index];
  assert.equal(model.report, '../runs/fidelity_v2/block000_' + name + '/metrics.json');
  const geometryPath = 'surface_quality_v2_block000_' + name + (name.endsWith('_diagnostic') ? '' : '_diagnostic') + '.json';
  assert.equal(geo.report, geometryPath);
  const report = read(path.join(reports, geometryPath));
  assert.equal(geo.geometry_report_sha256, sha(path.join(reports, geometryPath)));
  assert.equal(geo.floor, report.gate_evidence.G5.floor.candidate_hole_rate);
  assert.equal(geo.wall, report.gate_evidence.G5.wall.candidate_hole_rate);
  assert.equal(geo.p50_m, report.gate_evidence.G4.candidate_p50_m);
  assert.equal(geo.diagnostic_only, true);
  assert.equal(geo.high_fidelity_pass, false);
  if (name.startsWith('opacity600')) {
    assert.deepEqual(geo.gate_status, { G4: 'unknown', G5: 'unknown' });
    assert.ok(geo.evidence_note.includes('unknown'));
    assert.ok(geo.evidence_reasons[0].includes('checkpoint/export proof'));
  }
}

let states = 0;
for (let block = 0; block < data.blocks.length; block++) {
  elements.get('#block').value = String(block);
  elements.get('#block').onchange();
  const expected = data.blocks[block];
  assert.equal(Number(elements.get('#frame').value), 0);
  assert.equal(elements.get('#model').value, 'all');
  assert.equal(elements.get('#frame').children.length, 20);
  assert.equal(elements.get('#model').children.length, expected.models.length + 1);
  assert.equal(elements.get('#geometry').children.length, expected.models.length);
  if (expected.id === 'block_000') {
    assert.ok(elements.get('#pending').textContent.includes('block000_measuredseed_anneal_v1 · pending'));
    assert.equal(elements.get('#pending').children.length, 0);
    assert.ok(elements.get('#geometry').children[5].children[5].textContent.includes('unknown'));
    assert.ok(elements.get('#geometry').children[6].children[5].textContent.includes('unknown'));
  } else assert.equal(elements.get('#pending').textContent, '');
  for (const selection of ['all', ...expected.models.map((_, i) => String(i))]) {
    elements.get('#model').value = selection;
    elements.get('#model').onchange();
    const models = selection === 'all' ? expected.models : [expected.models[Number(selection)]];
    for (let frame = 0; frame < 20; frame++) {
      elements.get('#frame').value = String(frame);
      elements.get('#frame').onchange();
      for (const error of [false, true]) {
        elements.get('#error').checked = error;
        elements.get('#error').onchange();
        const panels = elements.get('#panels').children;
        assert.equal(panels.length, models.length + 1);
        const sources = panels.map(panel => panel.children.find(child => child.tag === 'img').src);
        assert.equal(sources[0], expected.reference[frame].image);
        models.forEach((model, index) => assert.equal(sources[index + 1], model.frames[frame][error ? 'error' : 'image']));
        for (const source of sources) local(source);
        for (const panel of panels) {
          for (const element of panel.children[0].children) if (element.href) local(element.href);
        }
        assert.ok(elements.get('#identity').textContent.includes(expected.id));
        assert.ok(elements.get('#identity').textContent.includes(expected.reference[frame].group));
        states++;
      }
    }
    elements.get('#next').onclick(); assert.equal(Number(elements.get('#frame').value), 0);
    elements.get('#prev').onclick(); assert.equal(Number(elements.get('#frame').value), 19);
  }
}
elements.get('#block').value = '0'; elements.get('#block').onchange();
assert.equal(elements.get('#model').value, 'all');
assert.equal(Number(elements.get('#frame').value), 0);
assert.equal(elements.get('#pending').textContent, '');
const result = { status: 'PASS', cpu_only: true, browser_started: false, states_checked: states,
  all_links_and_hashes_valid: links.linked_files, block_frame_model_error_switching: true,
  frame_wraparound: true, old_options_unchanged: true, protected_hashes_unchanged: true,
  opacity_unknown_geometry_disclosed: true, pending_has_no_links_or_metrics: true,
  diagnostic_geometry_reports_individually_bound: true, html_sha256: sha(htmlFile) };
fs.writeFileSync(path.join(reports, prefix + 'cpu.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));

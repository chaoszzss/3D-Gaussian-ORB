// Execute the actual offline page's selection logic using a tiny CPU-only DOM.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const htmlFile = path.join(root, 'reports/fidelity_v2_visual_review.html');
const html = fs.readFileSync(htmlFile, 'utf8');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.checked = false; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
}
const elements = new Map(['block', 'frame', 'panels', 'geometry', 'support', 'identity', 'error', 'prev', 'next']
  .map(id => [`#${id}`, new Element(id)]));
const context = vm.createContext({ document: {
  querySelector: selector => { assert.ok(elements.has(selector), selector); return elements.get(selector); },
  createElement: tag => new Element(tag),
} });
vm.runInContext(html.match(/<script>([\s\S]+)<\/script>/)[1], context);
const data = vm.runInContext('data', context);
assert.deepEqual(Array.from(data.blocks, b => b.id), ['block_015', 'block_000']);
let states = 0;
for (let block = 0; block < data.blocks.length; block++) {
  elements.get('#block').value = String(block);
  elements.get('#block').onchange();
  const expected = data.blocks[block];
  assert.equal(Number(elements.get('#frame').value), 0);
  assert.equal(elements.get('#frame').children.length, 20);
  assert.equal(elements.get('#geometry').children.length, expected.models.length);
  for (let frame = 0; frame < 20; frame++) {
    elements.get('#frame').value = String(frame);
    elements.get('#frame').onchange();
    for (const error of [false, true]) {
      elements.get('#error').checked = error;
      elements.get('#error').onchange();
      const panels = elements.get('#panels').children;
      assert.equal(panels.length, expected.models.length + 1);
      const sources = panels.map(panel => panel.children.find(child => child.tag === 'img').src);
      assert.equal(sources[0], expected.reference[frame].image);
      expected.models.forEach((model, index) => assert.equal(sources[index + 1], model.frames[frame][error ? 'error' : 'image']));
      for (const source of sources) assert.ok(fs.statSync(path.resolve(path.dirname(htmlFile), source)).isFile());
      assert.ok(elements.get('#identity').textContent.includes(expected.id));
      assert.ok(elements.get('#identity').textContent.includes(expected.reference[frame].group));
      states++;
    }
  }
  elements.get('#next').onclick(); assert.equal(Number(elements.get('#frame').value), 0);
  elements.get('#prev').onclick(); assert.equal(Number(elements.get('#frame').value), 19);
}
const validationPath = path.join(root, 'reports/viewer_parity_v2_visual_review_validation.json');
const validation = JSON.parse(fs.readFileSync(validationPath));
validation.actual_page_JS_cpu = { status: 'PASS', states_checked: states, block_frame_error_switching: true,
  frame_wraparound: true, browser_started: false };
fs.writeFileSync(validationPath, JSON.stringify(validation, null, 2) + '\n');
console.log(`PASS: ${states} actual page states, both blocks, all 20 frames, RGB/error and wraparound; no browser/GPU.`);

// Run with: node vr/navigate-binocular/tests/page_client_smoke.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const inline = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).find(source => source.includes('function connect()'));
assert.ok(inline, 'navigation page script must be self-contained');

const painted = [];
const sockets = [];
const input = { value: '127.0.0.1:5000' };
const status = { textContent: '' };
const button = {
  textContent: 'Connect',
  addEventListener(_event, handler) { this.click = handler; },
};
class FakeWebSocket {
  static OPEN = 1;
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    sockets.push(this);
  }
  close() {
    this.readyState = 3;
    if (this.onclose) this.onclose();
  }
  receive(value) { this.onmessage({ data: JSON.stringify(value) }); }
}
const context = {
  VR: { enableConfigPopup() {}, paintHUD(grid) { painted.push(grid); } },
  document: { getElementById(id) {
    return { 'stream-address': input, connect: button, status }[id];
  } },
  localStorage: { getItem() { return null; }, setItem() {} },
  WebSocket: FakeWebSocket,
  URL,
  performance: { now() { return 100; } },
  setInterval() {}, setTimeout() { return 1; }, clearTimeout() {},
  console,
};
vm.runInNewContext(inline, context, { filename: 'index.html' });

assert.equal(sockets[0].url, 'ws://127.0.0.1:5000/');
assert.deepEqual(Array.from(painted.at(-1)[0]), [0, 0, 0, 0, 0]);
sockets[0].readyState = 1;
sockets[0].onopen();
const values = { type: 'hud-config', on: 0.3, off: 0.1,
  actionStop: [0.3, 0.3, 0.3, 0.3, 0.3],
  actionUnknown: [0.1, 0.1, 0.1, 0.1, 0.1] };
sockets[0].receive(values);
assert.deepEqual(Array.from(painted.at(-1)[0]), values.actionUnknown);
assert.deepEqual(Array.from(painted.at(-1)[3]), values.actionStop);
const grid = [values.actionStop, values.actionUnknown,
  values.actionUnknown, values.actionUnknown];
sockets[0].receive(grid);
assert.equal(status.textContent, 'Receiving navigation HUD');
assert.deepEqual(Array.from(painted.at(-1)[0]), values.actionStop);
sockets[0].close();
assert.deepEqual(Array.from(painted.at(-1)[0]), values.actionUnknown);

button.click(); // disconnect
input.value = '192.168.1.42:5000';
button.click(); // reconnect to the chosen laptop address
assert.equal(sockets[1].url, 'ws://192.168.1.42:5000/');
console.log('Static page WebSocket smoke test passed');

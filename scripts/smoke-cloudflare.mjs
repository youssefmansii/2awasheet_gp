/* global process, fetch, setTimeout, console */
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:net';
import { once } from 'node:events';
import assert from 'node:assert/strict';

// Local workerd + real SQLite only. No remote host, match captures or deployed keys.
const directory = await mkdtemp(join(tmpdir(), 'cv-cloudflare-test-'));
const probe = createServer(); probe.listen(0, '127.0.0.1'); await once(probe, 'listening');
const port = probe.address().port; await new Promise((resolve) => probe.close(resolve));
const base = `http://127.0.0.1:${port}`;
const key = randomBytes(32).toString('hex');
const headers = { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };
let child; let logs = '';
async function start() {
  logs = '';
  child = spawn(process.execPath, [resolve('apps/cloudflare/node_modules/wrangler/bin/wrangler.js'),
    'dev', '--local', '--config', 'apps/cloudflare/wrangler.jsonc', '--ip', '127.0.0.1',
    '--port', String(port), '--persist-to', directory, '--var', `CV_API_KEY:${key}`],
  { env: { ...process.env, WRANGLER_SEND_METRICS: 'false', BROWSER: 'none' }, stdio: ['pipe', 'pipe', 'pipe'] });
  const record = (chunk) => { logs = (logs + chunk.toString().replaceAll(key, '[test-key]')).slice(-12000); };
  child.stdout.on('data', record); child.stderr.on('data', record);
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Local Worker exited: ${logs}`);
    try { if ((await fetch(base + '/health')).ok) return; } catch { /* startup */ }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Local Worker startup timed out: ${logs}`);
}
async function stop() {
  if (child && child.exitCode === null) {
    const done = once(child, 'exit'); child.stdin.end(); child.kill(); await done;
  }
}
const frame = (i) => ({ frameNum: i, videoTimeMs: i * 40, period: 1,
  periodGameClockTime: i / 25, homePlayers: [{ jerseyNum: '10', trackId: 3, x: i / 10, y: 2,
    confidence: 'HIGH', visibility: 'VISIBLE' }], awayPlayers: [], balls: [], extra: { preserved: true } });
const post = (path, body) => fetch(base + path, { method: 'POST', headers, body: JSON.stringify(body) });
try {
  await start();
  assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST' })).status, 401);
  const created = await fetch(base + '/api/cv/sessions', { method: 'POST', headers });
  assert.equal(created.status, 201); const { sessionId } = await created.json();
  const path = `/api/cv/sessions/${sessionId}`;
  const body = { sequence: 0, frames: Array.from({ length: 5 }, (_, i) => frame(i)) };
  const accepted = await post(path + '/frames', body); assert.equal(accepted.status, 200);
  assert.equal((await accepted.json()).acceptedFrames, 5);
  assert.deepEqual((await (await fetch(base + path + '/latest', { headers })).json()).frame, frame(4));
  assert.equal((await (await post(path + '/frames', body)).json()).duplicate, true);
  assert.equal((await post(path + '/frames', { sequence: 0, frames: [frame(5)] })).status, 409);
  for (const field of ['homePlayers', 'awayPlayers', 'homePlayersSmoothed', 'awayPlayersSmoothed']) {
    assert.equal((await post(path + '/frames', { sequence: 1,
      frames: [{ ...frame(5), [field]: [{ trackId: 3, jerseyNum: null, x: 1, y: 2 }] }] })).status, 422);
  }
  assert.equal((await post(path + '/frames', { sequence: 1, frames: [frame(5)], padding: 'x'.repeat(1048576) })).status, 413);
  assert.equal((await fetch(base + path + '/frames', { method: 'POST', headers: { ...headers, 'Content-Type': 'text/plain' }, body: '{}' })).status, 415);
  assert.equal((await fetch(base + path + '/latest', { headers: { ...headers, Origin: 'https://evil.invalid' } })).status, 403);
  assert.equal((await fetch(base + path + '/frames?limit=91', { headers })).status, 400);
  const batch1 = { sequence: 1, frames: [frame(5)] };
  const concurrent = await Promise.all([post(path + '/frames', batch1), post(path + '/frames', batch1)]);
  for (let i = 0; i < concurrent.length; i++) {
    if (concurrent[i].status === 409) concurrent[i] = await post(path + '/frames', batch1);
  }
  assert.ok(concurrent.every((r) => r.status === 200));
  const receipts = await Promise.all(concurrent.map((r) => r.json()));
  assert.equal(receipts.filter((r) => r.duplicate).length, 1);
  await stop(); await start();
  const restored = await fetch(base + path + '/latest', { headers }); assert.equal(restored.status, 200);
  const snapshot = await restored.json(); assert.equal(snapshot.receivedFrames, 6); assert.deepEqual(snapshot.frame, frame(5));
  assert.equal((await (await post(path + '/frames', batch1)).json()).duplicate, true);
  assert.equal((await post(path + '/frames', { sequence: 2, frames: Array.from({ length: 120 }, (_, i) => frame(i + 6)) })).status, 200);
  const tail = await fetch(base + path + '/frames?limit=90', { headers });
  const data = await tail.json(); assert.equal(data.retainedFrames, 90); assert.equal(data.frames[0].frameNum, 36);
  assert.equal(tail.headers.get('Cache-Control'), 'no-store');
  assert.equal(tail.headers.get('X-Content-Type-Options'), 'nosniff');
  assert.equal((await fetch(base + '/api/live/sessions', { method: 'POST', headers })).status, 404);
  for (let i = 0; i < 3; i++) assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST', headers })).status, 201);
  assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST', headers })).status, 429);
  assert.equal((await fetch(base + path, { method: 'DELETE', headers })).status, 200);
  assert.equal((await fetch(base + path + '/latest', { headers })).status, 404);
  for (let i = 0; i < 30; i++) await fetch(base + path + '/latest');
  assert.equal((await fetch(base + path + '/latest')).status, 429);
  assert.equal((await fetch(base + '/health')).status, 200);
  console.info('Cloudflare runtime smoke passed: auth, flexible batches, validation, atomic retries, SQLite restart recovery, retention, deletion and limits.');
  console.info(`Local test state retained at ${directory}; no match captures or real credentials.`);
} finally { await stop(); }

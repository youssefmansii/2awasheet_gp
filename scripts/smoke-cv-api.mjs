/* global process, fetch, setTimeout, console */
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { createServer } from 'node:net';
import { once } from 'node:events';
import assert from 'node:assert/strict';

// Exercise the actual compiled standalone service, with a fresh local-only key.
// No private samples, external hosts, user credentials or persistent files.
const probe = createServer();
probe.listen(0, '127.0.0.1');
await once(probe, 'listening');
const port = probe.address().port;
await new Promise((resolve) => probe.close(resolve));
const key = randomBytes(32).toString('hex');
const child = spawn(process.execPath, ['apps/api/dist/server.mjs'], {
  env: { ...process.env, PORT: String(port), CV_API_KEY: key, NODE_ENV: 'production' },
  stdio: ['ignore', 'pipe', 'pipe']
});
let startup = '';
child.stdout.on('data', (chunk) => { startup = (startup + chunk).slice(-8192); });
child.stderr.on('data', (chunk) => { startup = (startup + chunk).slice(-8192); });
const base = `http://127.0.0.1:${port}`;
const headers = { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };

try {
  const deadline = Date.now() + 20000;
  while (true) {
    if (child.exitCode !== null) throw new Error(`API exited during startup: ${startup}`);
    try { if ((await fetch(base + '/health')).ok) break; } catch { /* Startup is still in progress. */ }
    if (Date.now() >= deadline) throw new Error('API startup timed out.');
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST' })).status, 401);
  assert.equal((await fetch(base + '/api/live/sessions', { method: 'POST', headers })).status, 404);
  const created = await fetch(base + '/api/cv/sessions', { method: 'POST', headers });
  assert.equal(created.status, 201);
  const { sessionId } = await created.json();
  const path = `${base}/api/cv/sessions/${sessionId}`;
  const frame = {
    frameNum: 100, videoTimeMs: 4000, period: 1,
    homePlayers: [{ trackId: 42, jerseyNum: '10', x: -12.3, y: 8.5 }],
    awayPlayers: [], balls: []
  };
  const body = JSON.stringify({ sequence: 0, frames: [frame] });
  assert.equal((await fetch(path + '/frames', { method: 'POST', headers, body })).status, 200);
  const latest = await (await fetch(path + '/latest', { headers })).json();
  assert.deepEqual(latest.frame, frame);
  assert.equal(latest.receivedFrames, 1);
  const duplicate = await (await fetch(path + '/frames', { method: 'POST', headers, body })).json();
  assert.equal(duplicate.duplicate, true);
  assert.equal(duplicate.receivedFrames, 1);
  const unknown = { ...frame, frameNum: 101, videoTimeMs: 4040,
    homePlayers: [{ trackId: 42, jerseyNum: null, x: -12.3, y: 8.5 }] };
  assert.equal((await fetch(path + '/frames', {
    method: 'POST', headers, body: JSON.stringify({ sequence: 1, frames: [unknown] })
  })).status, 422);
  assert.equal((await fetch(path, { method: 'DELETE', headers })).status, 200);
  assert.equal((await fetch(path + '/latest', { headers })).status, 404);
  console.info('Standalone CV API smoke passed: health, auth, JSON POST/GET round-trip, retry, deletion, legacy-route isolation.');
} finally {
  if (child.exitCode === null) {
    const exited = once(child, 'exit');
    child.kill();
    await exited;
  }
}

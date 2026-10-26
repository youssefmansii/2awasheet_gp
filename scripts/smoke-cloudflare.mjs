/* global process, fetch, setTimeout, console */
import { spawn, execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:net';
import { once } from 'node:events';
import assert from 'node:assert/strict';
import { chromium, expect } from '@playwright/test';

// Local workerd + real SQLite only. No remote host, match captures or deployed keys.
execFileSync(process.execPath, ['apps/cloudflare/scripts/prepare-assets.mjs'], { stdio: 'inherit' });
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
  const docs = await fetch(base + '/docs'); assert.equal(docs.status, 200);
  assert.ok((await docs.text()).includes('swagger-ui-bundle.js'));
  const spec = await fetch(base + '/openapi.json'); assert.equal(spec.status, 200);
  assert.equal((await spec.json()).openapi, '3.1.0');
  const bundle = await fetch(base + '/docs-assets/swagger-ui-bundle.js'); assert.equal(bundle.status, 200);
  assert.ok((await bundle.text()).includes('SwaggerUIBundle'));
  assert.equal((await fetch(base + '/docs/swagger-initializer.js')).status, 200);
  assert.equal((await fetch(base + '/docs-assets/.env.cloudflare')).status, 401);
  // Isolated browser; only the random local test key is entered through the real UI.
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    const errors = []; const external = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
    page.on('request', (request) => {
      if (request.url().startsWith('http') && !request.url().startsWith(base + '/')) external.push(request.url());
    });
    await page.goto(base + '/docs?url=https://example.invalid/ignored-spec.json');
    await expect(page.locator('.opblock')).toHaveCount(9);
    await page.screenshot({ path: 'test-results/swagger-docs.png', fullPage: true });
    await page.getByRole('button', { name: 'Authorize', exact: true }).click();
    const dialog = page.locator('.dialog-ux');
    await dialog.locator('input').fill(key);
    await dialog.locator('button').filter({ hasText: /^Authorize$/ }).click();
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    const create = page.locator('#operations-Sessions-createSession');
    await create.locator('.opblock-summary').click();
    await create.getByRole('button', { name: 'Try it out' }).click();
    const responsePromise = page.waitForResponse((r) => r.url() === base + '/api/cv/sessions' && r.request().method() === 'POST');
    await create.getByRole('button', { name: 'Execute', exact: true }).click();
    const response = await responsePromise; assert.equal(response.status(), 201);
    const browserSession = await response.json();
    assert.equal((await fetch(base + '/api/cv/sessions/' + browserSession.sessionId, { method: 'DELETE', headers })).status, 200);
    assert.deepEqual(external, []); assert.deepEqual(errors, []);
    await page.reload();
    await expect(page.locator('.opblock')).toHaveCount(9);
    await page.getByRole('button', { name: 'Authorize', exact: true }).click();
    await expect(page.locator('.dialog-ux input')).toHaveValue('');
    console.info('Swagger browser passed: nine endpoints, real Authorize/Try it out, no external requests, no console errors, no persisted authorization.');
  } finally { await browser.close(); }
  assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST' })).status, 401);
  const created = await fetch(base + '/api/cv/sessions', { method: 'POST', headers });
  assert.equal(created.status, 201); const { sessionId } = await created.json();
  const path = `/api/cv/sessions/${sessionId}`;
  const frontend = 'http://localhost:5173';
  const corsOptions = await fetch(base + path + '/frames', { method: 'OPTIONS', headers: {
    Origin: frontend, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'authorization'
  } });
  assert.equal(corsOptions.status, 204);
  assert.equal(corsOptions.headers.get('Access-Control-Allow-Origin'), frontend);
  const corsRead = await fetch(base + path + '/frames?limit=90', { headers: { ...headers, Origin: frontend } });
  assert.equal(corsRead.status, 200); assert.equal(corsRead.headers.get('Access-Control-Allow-Origin'), frontend);
  assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST', headers: { ...headers, Origin: frontend } })).status, 403);
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
  assert.equal((await post(path + '/frames', { sequence: 1, frames: [frame(5)], padding: 'x'.repeat(1572864) })).status, 413);
  assert.equal((await fetch(base + path + '/frames', { method: 'POST', headers: { ...headers, 'Content-Type': 'text/plain' }, body: '{}' })).status, 415);
  assert.equal((await fetch(base + path + '/latest', { headers: { ...headers, Origin: 'https://evil.invalid' } })).status, 403);
  assert.equal((await fetch(base + path + '/frames?limit=301', { headers })).status, 400);
  const batch1 = { sequence: 1, frames: [frame(5)] };
  const concurrent = await Promise.all([post(path + '/frames', batch1), post(path + '/frames', batch1)]);
  for (let i = 0; i < concurrent.length; i++) {
    if (concurrent[i].status === 409) concurrent[i] = await post(path + '/frames', batch1);
  }
  assert.ok(concurrent.every((r) => r.status === 200));
  const receipts = await Promise.all(concurrent.map((r) => r.json()));
  assert.equal(receipts.filter((r) => r.duplicate).length, 1);
  // Durable batch archive: every accepted batch is stored and read back one by one by sequence.
  const nextBatch = async (after) => (await fetch(base + path + `/batches/next?after=${after}`, { headers })).json();
  const b0 = await nextBatch(-1);
  assert.equal(b0.batch.sequence, 0); assert.equal(b0.batch.frameOffset, 0); assert.deepEqual(b0.batch.frames, body.frames);
  assert.equal(b0.storedBatches, 2);
  const b1 = await nextBatch(0);
  assert.equal(b1.batch.sequence, 1); assert.equal(b1.batch.frameOffset, 5); assert.deepEqual(b1.batch.frames, batch1.frames);
  assert.equal((await nextBatch(1)).batch, null);
  assert.deepEqual((await (await fetch(base + path + '/batches/1', { headers })).json()).batch.frames, batch1.frames);
  const missingBatch = await fetch(base + path + '/batches/7', { headers });
  assert.equal(missingBatch.status, 404); assert.equal((await missingBatch.json()).error.code, 'BATCH_NOT_FOUND');
  assert.equal((await fetch(base + path + '/batches/next?after=-2', { headers })).status, 400);
  assert.equal((await fetch(base + path + '/batches?limit=101', { headers })).status, 400);
  const corsBatch = await fetch(base + path + '/batches/next?after=0', { headers: { ...headers, Origin: frontend } });
  assert.equal(corsBatch.status, 200); assert.equal(corsBatch.headers.get('Access-Control-Allow-Origin'), frontend);
  await stop(); await start();
  // Archive survives restart.
  assert.deepEqual((await nextBatch(0)).batch.frames, batch1.frames);
  const restored = await fetch(base + path + '/latest', { headers }); assert.equal(restored.status, 200);
  const snapshot = await restored.json(); assert.equal(snapshot.receivedFrames, 6); assert.deepEqual(snapshot.frame, frame(5));
  assert.equal((await (await post(path + '/frames', batch1)).json()).duplicate, true);
  assert.equal((await post(path + '/frames', { sequence: 2, frames: Array.from({ length: 301 }, (_, i) => frame(i + 6)) })).status, 422);
  assert.equal((await post(path + '/frames', { sequence: 2, frames: Array.from({ length: 300 }, (_, i) => frame(i + 6)) })).status, 200);
  // Upload many batches back-to-back: the tail buffer overwrites, but the archive keeps every batch.
  for (let seq = 3; seq < 9; seq++) {
    assert.equal((await post(path + '/frames', { sequence: seq, frames: Array.from({ length: 100 }, (_, i) => frame(306 + (seq - 3) * 100 + i)) })).status, 200);
  }
  const listing = await (await fetch(base + path + '/batches?after=-1', { headers })).json();
  assert.deepEqual(listing.batches.map((b) => b.sequence), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.ok(listing.batches.every((b) => !('frames' in b)));
  let cursor = -1; let consumed = 0;
  for (;;) { const r = await nextBatch(cursor); if (!r.batch) break; assert.equal(r.batch.frameOffset, consumed); consumed += r.batch.frames.length; cursor = r.batch.sequence; }
  assert.equal(consumed, 906); assert.equal(cursor, 8);
  const tail = await fetch(base + path + '/frames?limit=300', { headers });
  const data = await tail.json(); assert.equal(data.retainedFrames, 300); assert.equal(data.frames[0].frameNum, 606);
  assert.equal(tail.headers.get('Cache-Control'), 'no-store');
  assert.equal(tail.headers.get('X-Content-Type-Options'), 'nosniff');
  assert.equal((await fetch(base + '/api/live/sessions', { method: 'POST', headers })).status, 404);
  for (let i = 0; i < 3; i++) assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST', headers })).status, 201);
  assert.equal((await fetch(base + '/api/cv/sessions', { method: 'POST', headers })).status, 429);
  assert.equal((await fetch(base + path, { method: 'DELETE', headers })).status, 200);
  assert.equal((await fetch(base + path + '/latest', { headers })).status, 404);
  assert.equal((await fetch(base + path + '/batches/0', { headers })).status, 404);
  for (let i = 0; i < 30; i++) await fetch(base + path + '/latest');
  assert.equal((await fetch(base + path + '/latest')).status, 429);
  assert.equal((await fetch(base + '/health')).status, 200);
  console.info('Cloudflare runtime smoke passed: auth, flexible batches, durable one-by-one batch archive, validation, atomic retries, SQLite restart recovery, retention, deletion and limits.');
  console.info(`Local test state retained at ${directory}; no match captures or real credentials.`);
} finally { await stop(); }

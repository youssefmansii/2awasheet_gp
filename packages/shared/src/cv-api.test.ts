import { afterEach, describe, expect, it } from 'vitest';
import { createServer, request } from 'node:http';
import { once } from 'node:events';
import { createCvService } from '../../../apps/api/src/cv-service';
import { CV_LIMITS } from '../../../apps/api/src/cv-types';

// Test credential only; not a deployed API key.
const KEY = 'test-only-cv-key-0123456789-abcdef0123456789';
const headers = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' };
const cleanups: Array<() => Promise<void>> = [];
afterEach(async () => { for (const cleanup of cleanups.splice(0)) await cleanup(); });

function cvFrame(frameNum = 0, videoTimeMs = frameNum * 40) {
  return {
    frameNum, videoTimeMs, period: 1, gameRefId: 'cv-test',
    homePlayers: [{ trackId: 42, jerseyNum: '10', x: -12.3, y: 8.5 }],
    awayPlayers: [], balls: [], providerExtra: { calibrationStatus: 'provisional' }
  };
}

async function fixture(options: Parameters<typeof createCvService>[0] = { apiKey: KEY }) {
  const service = createCvService(options);
  const server = createServer((req, res) => { void service.handle(req, res); });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Missing test address');
  const base = `http://127.0.0.1:${address.port}`;
  cleanups.push(async () => {
    service.close(); server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });
  const create = async () => {
    const response = await fetch(base + '/api/cv/sessions', { method: 'POST', headers });
    expect(response.status).toBe(201);
    return await response.json() as { sessionId: string; createdAt: string; expiresAt: string };
  };
  return { base, create, server };
}

describe('CV diagnostic API', () => {
  it('offers public health but requires header credentials for all data operations', async () => {
    const { base, create } = await fixture();
    expect((await fetch(base + '/health')).status).toBe(200);
    expect((await fetch(base + '/api/cv/sessions', { method: 'POST' })).status).toBe(401);
    const { sessionId } = await create();
    for (const [method, suffix] of [['GET', '/latest'], ['GET', '/frames'], ['POST', '/frames'], ['DELETE', '']] as const) {
      const response = await fetch(`${base}/api/cv/sessions/${sessionId}${suffix}`, { method });
      expect(response.status).toBe(401);
      expect(await response.json()).toMatchObject({ error: { code: 'UNAUTHORIZED' } });
      expect((await fetch(`${base}/api/cv/sessions/${sessionId}${suffix}`, {
        method, headers: { Authorization: 'Bearer wrong-key-with-the-same-character-count-123456' }
      })).status).toBe(401);
    }
    expect((await fetch(`${base}/api/cv/sessions/${sessionId}/latest?token=${KEY}`)).status).toBe(401);
    expect((await fetch(base + '/api/live/sessions', { method: 'POST', headers })).status).toBe(404);
  });

  it('creates an empty session, accepts CV JSON, and returns original fields with compatibility warnings', async () => {
    const { base, create } = await fixture();
    const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    const empty = await (await fetch(path + '/latest', { headers })).json();
    expect(empty).toMatchObject({ sessionId, frame: null, receivedFrames: 0, lastSequence: null, receivedAt: null });
    const frames = [cvFrame(), cvFrame(1)];
    const accepted = await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence: 0, frames }) });
    expect(accepted.status).toBe(200);
    expect(await accepted.json()).toMatchObject({ sessionId, sequence: 0, acceptedFrames: 2, receivedFrames: 2, duplicate: false });
    const response = await fetch(path + '/latest', { headers });
    const latest = await response.json();
    expect(latest).toMatchObject({ frame: frames[1], receivedFrames: 2, lastSequence: 0, retainedFrames: 2 });
    expect(latest.receivedAt).toEqual(expect.any(String));
    expect(latest.warnings).toEqual(expect.arrayContaining(['MISSING_MATCH_CLOCK', 'MISSING_PLAYER_QUALITY']));
    expect(latest.warnings).not.toContain('UNKNOWN_JERSEY');
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(response.headers.get('x-content-type-options')).toBe('nosniff');
    const tail = await (await fetch(path + '/frames?limit=1', { headers })).json();
    expect(tail.frames).toEqual([frames[1]]);
    expect((await fetch(path, { method: 'DELETE', headers })).status).toBe(200);
    expect((await fetch(path + '/latest', { headers })).status).toBe(404);
  });

  it('acknowledges retries without double counting and rejects missing, changed or overlapping batches', async () => {
    const { base, create } = await fixture();
    const { sessionId } = await create(); const path = `${base}/api/cv/sessions/${sessionId}`;
    const upload = (sequence: number, frames: unknown[]) => fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence, frames }) });
    const initial = await (await upload(0, [cvFrame()])).json();
    expect((await upload(2, [cvFrame(2)])).status).toBe(409);
    expect((await upload(0, [cvFrame(1)])).status).toBe(409);
    expect((await upload(1, [cvFrame()])).status).toBe(409);
    expect((await upload(1, [cvFrame(2), cvFrame(1)])).status).toBe(422);
    expect((await upload(1, [{ ...cvFrame(1), gameRefId: 'different' }])).status).toBe(409);
    expect((await upload(1, [cvFrame(1)])).status).toBe(200);
    const retried = await (await upload(0, [cvFrame()])).json();
    expect(retried).toEqual({ ...initial, duplicate: true });
    expect(await (await fetch(path + '/latest', { headers })).json()).toMatchObject({ receivedFrames: 2, lastSequence: 1 });
  });

  it.each([
    null, { ...cvFrame(), frameNum: -1 }, { ...cvFrame(), videoTimeMs: '0' },
    { ...cvFrame(), homePlayers: null }, { ...cvFrame(), period: '1' },
    { ...cvFrame(), periodGameClockTime: -5 }, { ...cvFrame(), balls: [{ x: 0 }] },
    { ...cvFrame(), ballsSmoothed: [] }, { ...cvFrame(), homePlayersSmoothed: {} },
    { ...cvFrame(), homePlayers: [{ x: 0, y: 0, jerseyNum: null }] },
    { ...cvFrame(), homePlayers: [{ trackId: 1, jerseyNum: '10', x: null, y: 0 }] },
    { ...cvFrame(), homePlayers: [{ trackId: 1, jerseyNum: '10', x: 0, y: 0, confidence: 'CERTAIN' }] },
    { ...cvFrame(), homePlayers: [{ trackId: 1, jerseyNum: '10', x: 0, y: 0, visibility: 'NOT_VISIBLE' }] },
    { ...cvFrame(), homePlayers: [{ trackId: 1, jerseyNum: '10', x: 0, y: 0 }, { trackId: 1, jerseyNum: '11', x: 1, y: 1 }] },
    { ...cvFrame(), homePlayers: [{ trackId: 1, jerseyNum: '10', x: 0, y: 0 }, { trackId: 2, jerseyNum: 10, x: 1, y: 1 }] },
    { ...cvFrame(), homePlayers: Array.from({ length: 16 }, (_, trackId) => ({ trackId, jerseyNum: trackId + 1, x: 0, y: 0 })) }
  ])('rejects invalid frame %# atomically', async (invalid) => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    const response = await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence: 0, frames: [cvFrame(), invalid] }) });
    expect(response.status).toBe(422);
    expect(await response.json()).toMatchObject({ error: { code: 'INVALID_FRAME' } });
    expect(await (await fetch(path + '/latest', { headers })).json()).toMatchObject({ frame: null, receivedFrames: 0, lastSequence: null });
  });

  it('rejects bad envelopes, media types, oversized bodies and excessively nested provider fields', async () => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}/frames`;
    for (const body of [null, [], {}, { sequence: -1, frames: [cvFrame()] }, { sequence: 0, frames: [] },
      { sequence: 0, frames: Array.from({ length: 301 }, (_, index) => cvFrame(index)) }]) {
      expect((await fetch(path, { method: 'POST', headers, body: JSON.stringify(body) })).status).toBe(422);
    }
    expect((await fetch(path, { method: 'POST', headers, body: '{broken' })).status).toBe(400);
    expect((await fetch(path, { method: 'POST', headers: { ...headers, 'Content-Type': 'text/plain' }, body: '{}' })).status).toBe(415);
    expect((await fetch(path, { method: 'POST', headers: { ...headers, 'Content-Encoding': 'gzip' }, body: '{}' })).status).toBe(415);
    expect((await fetch(path, { method: 'POST', headers, body: ' '.repeat(CV_LIMITS.bodyBytes + 1) })).status).toBe(413);
    const nested = { sequence: 0, frames: [{ ...cvFrame(), extra: JSON.parse('['.repeat(80) + '0' + ']'.repeat(80)) }] };
    expect((await fetch(path, { method: 'POST', headers, body: JSON.stringify(nested) })).status).toBe(422);
    expect((await fetch(path, { method: 'POST', headers, body: '{"sequence":0,"frames":[{"frameNum":0,"videoTimeMs":0,"homePlayers":[],"awayPlayers":[],"extra":1e999}]}' })).status).toBe(422);
  });

  for (const team of ['homePlayers', 'awayPlayers', 'homePlayersSmoothed', 'awayPlayersSmoothed']) {
    it.each([undefined, null, '', 0, -1, 256, 10.5, 'unknown'])(`requires a valid jersey in ${team}, even with trackId (%s)`, async (jerseyNum) => {
      const { base, create } = await fixture(); const { sessionId } = await create();
      const path = `${base}/api/cv/sessions/${sessionId}`;
      const invalid = { ...cvFrame(1), [team]: [{ trackId: 42, jerseyNum, x: 1, y: 2 }] };
      const response = await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence: 0, frames: [cvFrame(), invalid] }) });
      expect(response.status).toBe(422);
      expect(await response.json()).toMatchObject({ error: { code: 'INVALID_FRAME', message: expect.stringContaining('jerseyNum') } });
      expect(await (await fetch(path + '/latest', { headers })).json()).toMatchObject({ frame: null, receivedFrames: 0, lastSequence: null });
    });
  }

  it.each(['10', 10, '255', 255])('accepts jersey %s without trackId and permits the same jersey on opposing teams', async (jerseyNum) => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    const frame = { ...cvFrame(), homePlayers: [{ jerseyNum, x: 1, y: 2 }], awayPlayers: [{ jerseyNum, x: -1, y: -2 }] };
    expect((await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence: 0, frames: [frame] }) })).status).toBe(200);
    expect((await (await fetch(path + '/latest', { headers })).json()).frame).toEqual(frame);
  });

  it('bounds retained frames and receipts, preserves chronological tails and rejects invalid read limits', async () => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    for (let sequence = 0; sequence < 34; sequence += 1) {
      const frames = Array.from({ length: 10 }, (_, index) => cvFrame(sequence * 10 + index));
      expect((await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence, frames }) })).status).toBe(200);
    }
    const tail = await (await fetch(path + '/frames?limit=90', { headers })).json();
    expect(tail.receivedFrames).toBe(340); expect(tail.retainedFrames).toBe(300);
    expect(tail.frames[0].frameNum).toBe(250); expect(tail.frames.at(-1).frameNum).toBe(339);
    expect((await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence: 0, frames: [cvFrame()] }) })).status).toBe(409);
    for (const limit of ['0', '301', '1.5', 'bad']) expect((await fetch(path + '/frames?limit=' + limit, { headers })).status).toBe(400);
  });

  it('enforces a byte cap even before the retained-frame count cap is reached', async () => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    for (let sequence = 0; sequence < 2; sequence += 1) {
      const frames = Array.from({ length: 40 }, (_, index) => ({ ...cvFrame(sequence * 40 + index), extra: 'x'.repeat(20000) }));
      expect((await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence, frames }) })).status).toBe(200);
    }
    const tail = await (await fetch(path + '/frames?limit=90', { headers })).json();
    expect(tail.retainedFrames).toBeLessThan(80); expect(tail.frames.at(-1).frameNum).toBe(79);
    expect(Buffer.byteLength(JSON.stringify(tail.frames))).toBeLessThan(CV_LIMITS.retainedBytes + 1000);
  });

  it('expires idle sessions even when polled, caps sessions, and releases deleted capacity', async () => {
    let now = 1000000;
    const { base, create } = await fixture({ apiKey: KEY, now: () => now });
    const first = await create(); await create(); await create(); await create();
    expect((await fetch(base + '/api/cv/sessions', { method: 'POST', headers })).status).toBe(429);
    expect((await fetch(`${base}/api/cv/sessions/${first.sessionId}`, { method: 'DELETE', headers })).status).toBe(200);
    const next = await create(); const path = `${base}/api/cv/sessions/${next.sessionId}`;
    now += CV_LIMITS.idleMs - 1;
    expect((await fetch(path + '/latest', { headers })).status).toBe(200);
    now += 2;
    expect((await fetch(path + '/latest', { headers })).status).toBe(404);
  });

  it('enforces hard expiry even when fresh uploads keep a session active', async () => {
    let now = 1000000;
    const { base, create } = await fixture({ apiKey: KEY, now: () => now });
    const { sessionId } = await create(); const path = `${base}/api/cv/sessions/${sessionId}`;
    for (let sequence = 0; sequence < 12; sequence += 1) {
      expect((await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence, frames: [cvFrame(sequence)] }) })).status).toBe(200);
      now += 10 * 60 * 1000;
    }
    expect((await fetch(path + '/latest', { headers })).status).toBe(404);
  });

  it('serializes simultaneous uploads without duplicating their side effects', async () => {
    const { base, create, server } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    const body = JSON.stringify({ sequence: 0, frames: [cvFrame()] });
    const started = once(server, 'request');
    const first = request(path + '/frames', { method: 'POST', headers: { ...headers, 'Content-Length': Buffer.byteLength(body) } });
    const finished = new Promise<number>((resolve, reject) => {
      first.on('response', (response) => { response.resume(); response.on('end', () => resolve(response.statusCode!)); });
      first.on('error', reject);
    });
    first.write(body.slice(0, 10)); await started;
    expect((await fetch(path + '/frames', { method: 'POST', headers, body })).status).toBe(409);
    first.end(body.slice(10)); expect(await finished).toBe(200);
    expect(await (await fetch(path + '/latest', { headers })).json()).toMatchObject({ receivedFrames: 1 });
  });

  it('limits failed authentication and resets the fixed window without exposing credentials', async () => {
    let now = 1000000; const { base } = await fixture({ apiKey: KEY, now: () => now });
    for (let index = 0; index < CV_LIMITS.failedAuthPerMinute; index += 1) expect((await fetch(base + '/api/cv/sessions', { method: 'POST' })).status).toBe(401);
    const limited = await fetch(base + '/api/cv/sessions', { method: 'POST' });
    expect(limited.status).toBe(429); expect(limited.headers.get('retry-after')).toBe('60');
    now += 60001;
    expect((await fetch(base + '/api/cv/sessions', { method: 'POST' })).status).toBe(401);
    expect((await fetch(base + '/api/cv/sessions', { method: 'POST', headers: { ...headers, Origin: 'https://evil.invalid' } })).status).toBe(403);
  });

  it('bounds total non-health traffic while keeping public health available', async () => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}/latest`;
    for (let index = 1; index < CV_LIMITS.requestsPerMinute; index += 1) {
      const response = await fetch(path, { headers });
      expect(response.status).toBe(200); await response.arrayBuffer();
    }
    expect((await fetch(path, { headers })).status).toBe(429);
    expect((await fetch(base + '/health')).status).toBe(200);
  });

  it('times out a stalled body and releases the upload lock', async () => {
    const { base, create } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    const slow = request(path + '/frames', { method: 'POST', headers: { ...headers, 'Content-Length': '1000' } });
    const finished = new Promise<number>((resolve, reject) => {
      slow.on('response', (response) => { response.resume(); response.on('end', () => resolve(response.statusCode!)); });
      slow.on('error', reject);
    });
    slow.write('{'); expect(await finished).toBe(408);
    slow.destroy();
    const result = await fetch(path + '/frames', { method: 'POST', headers, body: JSON.stringify({ sequence: 0, frames: [cvFrame()] }) });
    expect(result.status).toBe(200);
  }, 15000);

  it('never commits an upload into a session deleted while its body is arriving', async () => {
    const { base, create, server } = await fixture(); const { sessionId } = await create();
    const path = `${base}/api/cv/sessions/${sessionId}`;
    const body = JSON.stringify({ sequence: 0, frames: [cvFrame()] });
    const started = once(server, 'request');
    const pending = request(path + '/frames', { method: 'POST', headers: { ...headers, 'Content-Length': Buffer.byteLength(body) } });
    const finished = new Promise<number>((resolve, reject) => {
      pending.on('response', (response) => { response.resume(); response.on('end', () => resolve(response.statusCode!)); }); pending.on('error', reject);
    });
    pending.write(body.slice(0, 10)); await started;
    expect((await fetch(path, { method: 'DELETE', headers })).status).toBe(200);
    pending.end(body.slice(10)); expect(await finished).toBe(404);
  });
});

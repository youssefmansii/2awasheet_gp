import { describe, expect, it } from 'vitest';
import { newSession, acceptBatch, metadata, expired, restoreSession, batchSummary } from '../../../apps/cloudflare/src/session';
import { validateCvBatch } from '../../../apps/api/src/cv-validation';
import { CV_LIMITS } from '../../../apps/api/src/cv-types';
import { readFileSync, existsSync } from 'node:fs';

const frame = (i: number) => ({ frameNum: i, videoTimeMs: i * 40, period: 1,
  periodGameClockTime: i / 25, homePlayers: [{ jerseyNum: '10', x: i, y: 2 }], awayPlayers: [], balls: [] });
const batch = (sequence: number, start: number, count: number) => validateCvBatch({ sequence,
  frames: Array.from({ length: count }, (_, i) => frame(start + i)) });

describe('Cloudflare durable session state', () => {
  it.each([1, 5, 25, 120, 300])('accepts flexible batches of %i original frames', (count) => {
    const session = newSession('a'.repeat(32), 1000);
    const result = acceptBatch(session, batch(0, 0, count), 2000);
    expect(result.acceptedFrames).toBe(count);
    expect(metadata(session).receivedFrames).toBe(count);
    expect(session.frames.at(-1)?.frame).toEqual(frame(count - 1));
    expect(session.frames.length).toBe(count);
  });
  it('survives JSON storage and safely replays a duplicate without extending expiry', () => {
    const session = newSession('b'.repeat(32), 1000);
    acceptBatch(session, batch(0, 0, 5), 2000);
    const restored = JSON.parse(JSON.stringify(session));
    expect(acceptBatch(restored, batch(0, 0, 5), 3000).duplicate).toBe(true);
    expect(restored.updatedAt).toBe(2000);
    expect(restored.receivedFrames).toBe(5);
    expect(() => acceptBatch(restored, batch(0, 5, 5), 3000)).toThrow('different content');
    expect(() => acceptBatch(restored, batch(2, 5, 5), 3000)).toThrow('next batch');
    expect(() => acceptBatch(restored, batch(1, 4, 5), 3000)).toThrow('overlaps');
  });
  it('bounds retained bytes and keeps receipt history to 32 batches', () => {
    const session = newSession('c'.repeat(32), 0);
    for (let i = 0; i < 40; i++) {
      const data = batch(i, i, 1);
      data.frames[0]!.extra = 'x'.repeat(100000);
      acceptBatch(session, data, i + 1);
    }
    expect(session.bytes).toBeLessThanOrEqual(CV_LIMITS.retainedBytes);
    expect(Buffer.byteLength(JSON.stringify(session))).toBeLessThan(2_000_000);
    expect(session.receipts).toHaveLength(32);
    expect(session.receivedFrames).toBe(40);
  });
  it('retains a 300-frame batch, rejects 301, and preserves a chronological bounded tail', () => {
    const session = newSession('f'.repeat(32), 0);
    acceptBatch(session, batch(0, 0, 300), 1);
    expect(() => batch(1, 300, 301)).toThrow('1–300');
    acceptBatch(session, batch(1, 300, 10), 2);
    expect(session.frames).toHaveLength(300);
    expect(session.frames[0]!.frame.frameNum).toBe(10);
    expect(session.frames.at(-1)!.frame.frameNum).toBe(309);
  });
  it('fits a full-size 300-frame batch in both the request cap and SQLite snapshot', () => {
    // Exercise the real capture locally without requiring private files in public checkouts.
    const path = 'samples/3816-10s.jsonl';
    const frames = existsSync(path)
      ? readFileSync(path, 'utf8').trim().split(/\r?\n/).slice(0, 300).map((line) => JSON.parse(line))
      : Array.from({ length: 300 }, (_, i) => ({ ...frame(i), diagnosticPadding: 'x'.repeat(4100) }));
    expect(frames).toHaveLength(300);
    const input = validateCvBatch({ sequence: 0, frames });
    expect(Buffer.byteLength(JSON.stringify(input))).toBeGreaterThan(1024 * 1024);
    expect(Buffer.byteLength(JSON.stringify(input))).toBeLessThan(CV_LIMITS.bodyBytes);
    const session = newSession('f'.repeat(32), 0); acceptBatch(session, input, 1);
    expect(session.frames).toHaveLength(300);
    expect(Buffer.byteLength(JSON.stringify(session))).toBeLessThan(2_000_000);
  });
  it('expires on idle and absolute limits, with no reads extending it', () => {
    const session = newSession('d'.repeat(32), 0);
    expect(expired(session, 899999)).toBe(false);
    metadata(session);
    expect(expired(session, 900000)).toBe(true);
    session.updatedAt = 7199999;
    expect(expired(session, 7200000)).toBe(true);
  });
  it('rejects a match switch atomically and requires jersey numbers in all arrays', () => {
    const session = newSession('e'.repeat(32), 0);
    const first = batch(0, 0, 5); first.frames[0]!.gameRefId = 'match-a';
    acceptBatch(session, first, 1);
    const next = batch(1, 5, 5); next.frames[0]!.gameRefId = 'match-b';
    expect(() => acceptBatch(session, next, 2)).toThrow('Match ID');
    expect(session.receivedFrames).toBe(5);
    for (const key of ['homePlayers', 'awayPlayers', 'homePlayersSmoothed', 'awayPlayersSmoothed']) {
      const f = { ...frame(5), [key]: [{ trackId: 1, jerseyNum: null, x: 1, y: 2 }] };
      expect(() => validateCvBatch({ sequence: 1, frames: [f] })).toThrow('jerseyNum is required');
    }
  });
  it('counts archived batches/bytes exactly, never for duplicates, and restores pre-archive sessions', () => {
    const session = newSession('9'.repeat(32), 0);
    const first = batch(0, 0, 3);
    acceptBatch(session, first, 1);
    expect(session.storedBatches).toBe(1);
    expect(session.archivedBytes).toBe(Buffer.byteLength(JSON.stringify(first.frames)));
    acceptBatch(session, batch(0, 0, 3), 2);
    expect(session.storedBatches).toBe(1);
    acceptBatch(session, batch(1, 3, 2), 3);
    expect(metadata(session)).toMatchObject({ storedBatches: 2, receivedFrames: 5 });
    const legacy = JSON.stringify({ ...session, storedBatches: undefined, archivedBytes: undefined });
    const { storedBatches, archivedBytes, ...old } = JSON.parse(legacy);
    expect(storedBatches).toBeUndefined(); expect(archivedBytes).toBeUndefined();
    expect(restoreSession(JSON.stringify(old))).toMatchObject({ storedBatches: 0, archivedBytes: 0, receivedFrames: 5 });
  });
  it('rejects uploads atomically once the per-session archive is full', () => {
    const session = newSession('8'.repeat(32), 0);
    session.archivedBytes = CV_LIMITS.archiveBytes - 10;
    expect(() => acceptBatch(session, batch(0, 0, 1), 1)).toThrow('stored-batch limit');
    expect(session.receivedFrames).toBe(0);
    expect(session.lastSequence).toBeNull();
  });
  it('summarizes a batch with its frame offset and source ranges', () => {
    expect(batchSummary(batch(4, 10, 3), 10, 99, '2026-10-18T12:00:00.000Z')).toEqual({ sequence: 4, frameOffset: 10, frameCount: 3,
      firstFrameNum: 10, lastFrameNum: 12, firstVideoTimeMs: 400, lastVideoTimeMs: 480, bytes: 99, receivedAt: '2026-10-18T12:00:00.000Z' });
  });
});

import { describe, expect, it } from 'vitest';
import { newSession, acceptBatch, metadata, expired } from '../../../apps/cloudflare/src/session';
import { validateCvBatch } from '../../../apps/api/src/cv-validation';

const frame = (i: number) => ({ frameNum: i, videoTimeMs: i * 40, period: 1,
  periodGameClockTime: i / 25, homePlayers: [{ jerseyNum: '10', x: i, y: 2 }], awayPlayers: [], balls: [] });
const batch = (sequence: number, start: number, count: number) => validateCvBatch({ sequence,
  frames: Array.from({ length: count }, (_, i) => frame(start + i)) });

describe('Cloudflare durable session state', () => {
  it.each([1, 5, 25, 120])('accepts flexible batches of %i original frames', (count) => {
    const session = newSession('a'.repeat(32), 1000);
    const result = acceptBatch(session, batch(0, 0, count), 2000);
    expect(result.acceptedFrames).toBe(count);
    expect(metadata(session).receivedFrames).toBe(count);
    expect(session.frames.at(-1)?.frame).toEqual(frame(count - 1));
    expect(session.frames.length).toBe(Math.min(count, 90));
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
    expect(session.bytes).toBeLessThanOrEqual(1024 * 1024);
    expect(session.receipts).toHaveLength(32);
    expect(session.receivedFrames).toBe(40);
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
});

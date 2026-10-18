import { describe, expect, it } from 'vitest';
import { CvReceiver, cvReadUrl } from '../../../apps/web/src/live/cv-receiver';
const id = 'a'.repeat(32);
const frame = (n: number) => ({ frameNum: n, videoTimeMs: n * 100, period: 1, periodGameClockTime: n / 10,
  homePlayers: [{ jerseyNum: '10', x: n * 0.5, y: 0 }], awayPlayers: [], balls: [] });
const snapshot = (start: number, count: number, receivedFrames = start + count) => ({
  sessionId: id, receivedFrames, retainedFrames: count, lastSequence: receivedFrames ? 0 : null,
  receivedAt: receivedFrames ? '2026-10-18T12:00:00Z' : null, warnings: [],
  frames: Array.from({ length: count }, (_, i) => frame(start + i))
});
describe('CV live receiver', () => {
  it('builds only the fixed API read URL and rejects invalid session IDs', () => {
    expect(cvReadUrl(id)).toBe(`https://tracking-cv-api.youssefmansi2.workers.dev/api/cv/sessions/${id}/frames?limit=300`);
    expect(() => cvReadUrl('../secret')).toThrow();
  });
  it('waits for empty sessions and deduplicates overlapping polling snapshots', () => {
    const r = new CvReceiver(id);
    expect(r.consume(snapshot(0, 0)).data).toBeUndefined();
    expect(r.consume(snapshot(0, 10)).processedFrames).toBe(10);
    expect(r.consume(snapshot(0, 10)).data).toBeUndefined();
    const result = r.consume(snapshot(5, 10));
    expect(result.processedFrames).toBe(15);
    r.advance(0); const played = r.advance(1500);
    expect(played.workloads![0]!.distanceM).toBeCloseTo(7);
    expect(played.data!.frameNum.at(-1)).toBe(14);
  });
  it('reports late-join and missed-buffer history without bridging missing frames', () => {
    const r = new CvReceiver(id);
    expect(r.consume(snapshot(100, 5)).missingFrames).toBe(100);
    const result = r.consume(snapshot(106, 5));
    expect(result.missingFrames).toBe(101);
    r.advance(0);
    expect(r.advance(1500).workloads![0]!.distanceM).toBeCloseTo(4);
  });
  it('keeps only a bounded pitch window while accumulating observed workload', () => {
    const r = new CvReceiver(id);
    r.consume(snapshot(0, 90)); const result = r.consume(snapshot(90, 90));
    r.advance(0); const played = r.advance(18000);
    expect(played.data!.frameCount).toBe(90);
    expect(result.processedFrames).toBe(180);
    expect(played.workloads![0]!.distanceM).toBeCloseTo(89.5);
  });
  it('rejects malformed or wrong-session data and never accepts unknown jerseys', () => {
    const r = new CvReceiver(id);
    expect(() => r.consume({ ...snapshot(0, 1), sessionId: 'b'.repeat(32) })).toThrow();
    expect(() => r.consume({ ...snapshot(0, 1), frames: [{ ...frame(0), homePlayers: [{ trackId: 5, x: 0, y: 0 }] }] })).toThrow();
    expect(() => r.consume({ ...snapshot(0, 1), receivedFrames: -1 })).toThrow();
    expect(() => r.consume({ ...snapshot(0, 1), retainedFrames: 2 })).toThrow();
    expect(r.consume(snapshot(0, 1)).processedFrames).toBe(1);
  });
  it('rejects counters/source time going backwards rather than mixing sessions', () => {
    const r = new CvReceiver(id); r.consume(snapshot(0, 5));
    expect(() => r.consume(snapshot(0, 4))).toThrow();
    expect(() => r.consume(snapshot(0, 3, 6))).toThrow();
  });
  it('does not invent match clocks when the provider omits them', () => {
    const r = new CvReceiver(id); const input = snapshot(0, 1);
    const withoutClock: Record<string, unknown> = { ...input.frames[0]! }; delete withoutClock.periodGameClockTime;
    r.consume({ ...input, frames: [withoutClock] });
    expect(r.advance(0).clockComplete).toBe(false);
  });
});

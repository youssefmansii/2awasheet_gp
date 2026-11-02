import { describe, expect, it } from 'vitest';
import { CvReceiver } from '../../../apps/web/src/live/cv-receiver';

const id = 'a'.repeat(32);
const snapshot = (start: number, size: number, receivedFrames = start + size) => ({
  sessionId: id, receivedFrames, retainedFrames: size, warnings: [], receivedAt: '2026-10-18T12:00:00Z',
  frames: Array.from({ length: size }, (_, i) => ({ frameNum: start + i, videoTimeMs: (start + i) * 1000 / 30,
    period: 1, periodGameClockTime: (start + i) / 30,
    homePlayers: [{ jerseyNum: '10', x: (start + i) / 10, y: 0 }], awayPlayers: [] }))
});

describe('buffered CV playback', () => {
  it('queues 300 frames without jumping ahead, and plays ten seconds at source speed', () => {
    const receiver = new CvReceiver(id);
    const received = receiver.consume(snapshot(0, 300));
    expect(received.playedFrames).toBe(0);
    expect(received.bufferedFrames).toBe(300);
    const first = receiver.advance(0);
    expect(first.data!.frameNum.at(-1)).toBe(0);
    expect(first.workloads![0]!.distanceM).toBe(0);
    const middle = receiver.advance(5000);
    expect(middle.playedFrames).toBe(151);
    expect(middle.bufferedFrames).toBe(149);
    expect(middle.data!.frameNum.at(-1)).toBe(150);
    expect(middle.workloads![0]!.distanceM).toBeCloseTo(15);
    const end = receiver.advance(10000);
    expect(end.playedFrames).toBe(300);
    expect(end.bufferedFrames).toBe(0);
    expect(end.workloads![0]!.distanceM).toBeCloseTo(29.9);
  });
  it('holds the final frame during minutes of waiting and resumes without fast-forwarding', () => {
    const receiver = new CvReceiver(id);
    receiver.consume(snapshot(0, 30)); receiver.advance(0); receiver.advance(1000);
    expect(receiver.advance(120000).data).toBeUndefined();
    receiver.consume(snapshot(30, 30));
    expect(receiver.advance(120000).playedFrames).toBe(31);
    expect(receiver.advance(120500).playedFrames).toBe(46);
    expect(receiver.advance(121000).playedFrames).toBe(60);
  });
  it('never counts repeated polls twice and does not let new batches skip queued frames', () => {
    const receiver = new CvReceiver(id);
    receiver.consume(snapshot(0, 60)); receiver.advance(0);
    receiver.consume(snapshot(0, 60));
    receiver.consume(snapshot(30, 60));
    const midway = receiver.advance(1000);
    expect(midway.processedFrames).toBe(90);
    expect(midway.playedFrames).toBe(31);
    expect(midway.data!.frameNum.at(-1)).toBe(30);
  });
  it('breaks workload continuity at a known missing segment, only when it is played', () => {
    const receiver = new CvReceiver(id);
    receiver.consume(snapshot(0, 3)); receiver.advance(0);
    receiver.consume(snapshot(5, 3));
    const end = receiver.advance(1000);
    expect(end.missingFrames).toBe(2);
    expect(end.playedFrames).toBe(6);
    expect(end.workloads![0]!.distanceM).toBeCloseTo(0.4);
  });
  it('stops explicitly on queue overflow without corrupting accepted playback', () => {
    const receiver = new CvReceiver(id);
    for (let i = 0; i < 12; i++) receiver.consume(snapshot(i * 300, 300));
    expect(() => receiver.consume(snapshot(3600, 1))).toThrow('Playback buffer is full');
    expect(receiver.advance(0).processedFrames).toBe(3600);
    expect(receiver.advance(0).playedFrames).toBe(1);
  });
});

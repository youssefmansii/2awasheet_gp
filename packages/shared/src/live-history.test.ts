import { describe, expect, it } from 'vitest';
import { CvReceiver } from '../../../apps/web/src/live/cv-receiver';
import { LiveHistory, stepHistoryPlayback } from '../../../apps/web/src/live/live-history';
import { packFrames, parseTrackingLine, type ParsedFrame, type TrackingData } from './index';

const id = 'a'.repeat(32);
/** 30 FPS frames; `period2From` switches to the second half (clock restarts at 45:00). */
const raw = (n: number, period2From = Infinity) => ({ frameNum: n, videoTimeMs: n * 1000 / 30,
  period: n >= period2From ? 2 : 1, periodGameClockTime: n >= period2From ? 2700 + (n - period2From) / 30 : n / 30,
  homePlayers: [{ jerseyNum: '10', x: n / 10, y: 0 }], awayPlayers: [] });
const chunk = (from: number, count: number, period2From = Infinity): TrackingData => {
  const frames = Array.from({ length: count }, (_, i) => {
    const parsed = parseTrackingLine(JSON.stringify(raw(from + i, period2From)));
    if (parsed.kind !== 'frame') throw new Error('bad test frame');
    return parsed.frame as ParsedFrame;
  });
  return packFrames(frames, { malformedLines: 0, blankLines: 0, overflowPlayers: 0 }, []);
};

describe('live DVR history', () => {
  it('keeps every played frame (growing past its initial capacity) as zero-copy TrackingData', () => {
    const history = new LiveHistory();
    for (let i = 0; i < 10; i++) history.append(chunk(i * 1000, 1000));
    expect(history.frameCount).toBe(10000);
    const view = history.view([]);
    expect(view.frameCount).toBe(10000);
    expect(view.frameNum[0]).toBe(0);
    expect(view.frameNum[9999]).toBe(9999);
    expect(view.smoothedPlayers[9999 * 60]).toBeCloseTo(999.9);
    expect(view.videoTimeMs.length).toBe(10000);
    expect(() => history.append(chunk(5, 1))).toThrow('increase in source time');
    history.clear();
    expect(history.frameCount).toBe(0);
  });

  it('finds any second and moves ±5/10 s like a video player', () => {
    const history = new LiveHistory(); history.append(chunk(0, 1800)); // one minute
    expect(history.indexAtOrBefore(30_000)).toBe(900);
    expect(history.indexAtOrAfter(30_010)).toBe(901);
    expect(history.shift(900, -5)).toBe(750);
    expect(history.shift(900, 10)).toBe(1200);
    expect(history.shift(100, -10)).toBe(0);
    expect(history.shift(1700, 10)).toBe(1799);
    expect(history.label(900, true)).toBe('00:30');
    expect(history.label(900, false)).toBe('Video 00:30');
  });

  it('labels the second half with the match clock and marks where it starts', () => {
    const history = new LiveHistory(); history.append(chunk(0, 60, 30)); history.append(chunk(60, 60, 30));
    expect(history.periodStarts()).toEqual([{ index: 30, period: 2 }]);
    expect(history.label(29, true)).toBe('00:00');
    expect(history.label(30, true)).toBe('45:00');
    expect(history.label(119, true)).toBe('45:02'); // 89 frames = 2.97 s
  });

  it('replays behind live at source speed, skips known gaps, and reports catching the live edge', () => {
    const history = new LiveHistory();
    history.append(chunk(0, 300));
    history.append(chunk(400, 30), Uint8Array.from({ length: 30 }, (_, i) => i === 0 ? 1 : 0));
    const live = history.frameCount - 1;
    const second = stepHistoryPlayback(history, 0, 1000, live);
    expect(second.index).toBe(30);
    expect(stepHistoryPlayback(history, 0, 10, live).index).toBe(0); // less than one frame: hold
    // Crossing the missing segment does not wait ~3.3 s of non-existent footage.
    expect(stepHistoryPlayback(history, 299, 0, live).index).toBe(300);
    const caught = stepHistoryPlayback(history, 300, 60_000, live);
    expect(caught).toEqual({ index: live, carryMs: 0 });
  });

  it('receives the receiver chunks in order so the full match stays seekable after the live edge moves on', () => {
    const receiver = new CvReceiver(id); const history = new LiveHistory();
    receiver.consume({ sessionId: id, receivedFrames: 300, retainedFrames: 300, warnings: [], receivedAt: '2026-11-02T00:00:00Z',
      frames: Array.from({ length: 300 }, (_, i) => raw(i)) });
    for (let t = 0; t <= 10_000; t += 16) {
      const update = receiver.advance(t);
      if (update.data) history.append(update.data, update.gapBefore);
    }
    expect(history.frameCount).toBe(300);
    const view = history.view([]);
    expect(Array.from(view.frameNum.subarray(0, 3))).toEqual([0, 1, 2]);
    expect(view.frameNum[299]).toBe(299);
  });
});

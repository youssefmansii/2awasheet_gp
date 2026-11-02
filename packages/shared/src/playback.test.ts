import { describe, expect, it } from 'vitest';
import { findFrameByTime } from './playback';

describe('findFrameByTime', () => {
  const times = new Float64Array([100, 133.3, 166.6, 200]);
  it('handles boundaries and values between frames', () => {
    expect(findFrameByTime(times, 0)).toBe(0);
    expect(findFrameByTime(times, 150)).toBe(1);
    expect(findFrameByTime(times, 200)).toBe(3);
    expect(findFrameByTime(times, 999)).toBe(3);
    expect(findFrameByTime(new Float64Array(), 1)).toBe(-1);
  });
});

import { describe, expect, it } from 'vitest';
import { parseTrackingLine } from './parser';

function frame(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    frameNum: 42,
    videoTimeMs: 1200.5,
    period: 1,
    periodGameClockTime: 4.25,
    homePlayers: [{ jerseyNum: '10', x: 1, y: 2, confidence: 'HIGH', visibility: 'VISIBLE' }],
    homePlayersSmoothed: [{ jerseyNum: '10', x: 1.5, y: 2.5, confidence: 'LOW', visibility: 'VISIBLE' }],
    awayPlayers: [],
    awayPlayersSmoothed: [],
    balls: [{ x: 3, y: 4, z: 0, visibility: 'VISIBLE' }],
    ballsSmoothed: { x: 3.5, y: 4.5, z: 0.2, visibility: 'VISIBLE' },
    ...overrides
  });
}

describe('parseTrackingLine', () => {
  it('parses normal frames and keeps string jersey numbers', () => {
    const result = parseTrackingLine(frame());
    expect(result.kind).toBe('frame');
    if (result.kind !== 'frame') return;
    expect(result.frame.rawJerseys[0]).toBe(10);
    expect(result.frame.smoothedPlayers[0]).toBe(1.5);
    expect(result.frame.smoothedFlags[0]).toBe(1);
  });

  it('accepts null metadata fields', () => {
    const result = parseTrackingLine(frame({ version: null, gameRefId: null, generatedTime: null }));
    expect(result.kind).toBe('frame');
    if (result.kind === 'frame') expect(result.frame.gameRefId).toBeNull();
  });

  it('falls back to raw ball when ballsSmoothed is missing', () => {
    const source = JSON.parse(frame()) as Record<string, unknown>;
    delete source.ballsSmoothed;
    const result = parseTrackingLine(JSON.stringify(source));
    if (result.kind !== 'frame') throw new Error('Expected frame');
    expect(Array.from(result.frame.smoothedBall)).toEqual([3, 4, 0]);
  });

  it('classifies empty and malformed lines without throwing', () => {
    expect(parseTrackingLine('   ').kind).toBe('blank');
    expect(parseTrackingLine('{nope').kind).toBe('malformed');
  });

  it('handles varying player counts', () => {
    const result = parseTrackingLine(frame({ homePlayers: [], homePlayersSmoothed: [] }));
    if (result.kind !== 'frame') throw new Error('Expected frame');
    expect(result.frame.rawJerseys.every((value) => value === 0)).toBe(true);
  });
});

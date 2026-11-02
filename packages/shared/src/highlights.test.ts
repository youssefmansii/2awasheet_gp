import { describe, expect, it } from 'vitest';
import { parseEventDataset } from './events';
import { advanceHighlightPlayback, buildHighlights, highlightProgress, highlightTimeAtProgress, nearestHighlightTime } from '../../../apps/web/src/replay/highlights';

const times = Float64Array.from({ length: 1001 }, (_, i) => i * 100);
const events = (records: Array<{ time: number; game?: string; possession?: string; outcome?: string; card?: string }>) =>
  parseEventDataset(JSON.stringify(records.map((r, i) => ({ gameEventId: i, eventTime: r.time,
    gameEvents: { gameEventType: r.game ?? 'OTB', cardType: r.card, playerName: 'Player', period: 1 },
    possessionEvents: { possessionEventType: r.possession, shotOutcomeType: r.outcome, formattedGameClock: '00:12' }
  })))).events;

describe('automatic local highlights', () => {
  it('selects shots, raw goal candidates, explicit cards and substitutions, not ordinary passes/fouls', () => {
    const clips = buildHighlights(events([{ time: 12, possession: 'SH', outcome: 'G' }, { time: 40, game: 'YC' },
      { time: 70, game: 'SUB' }, { time: 90, possession: 'PA' }, { time: 95, possession: 'FO' }]), times);
    expect(clips.map((c) => [c.startMs, c.endMs])).toEqual([[7000, 17000], [35000, 45000], [65000, 75000]]);
    expect(clips[0]!.events[0]!.label).toContain('Goal candidate');
    expect(clips[1]!.events[0]!.label).toContain('Card');
    expect(clips[2]!.events[0]!.label).toContain('Substitution');
  });
  it('merges overlapping clips, orders them and respects recording and half boundaries', () => {
    const clips = buildHighlights(events([{ time: 19, possession: 'SH' }, { time: 15, game: 'RC' },
      { time: 24, possession: 'SH' }, { time: 99, game: 'SUB' }]), times, [[0, 20000], [60000, 100000]]);
    expect(clips.map((c) => [c.startMs, c.endMs])).toEqual([[10000, 20000], [94000, 100000]]);
    expect(clips[0]!.events).toHaveLength(2);
    expect(clips[0]!.events[0]!.timeMs).toBe(15000);
  });
  it('does not fabricate clips for missing tracking, absent events or events outside the recording', () => {
    expect(buildHighlights([], times)).toEqual([]);
    expect(buildHighlights(events([{ time: 12, possession: 'SH' }]), new Float64Array())).toEqual([]);
    expect(buildHighlights(events([{ time: 120, possession: 'SH' }]), times)).toEqual([]);
    expect(buildHighlights(events([{ time: 50, possession: 'SH' }]), Float64Array.of(0, 10000, 90000, 100000))).toEqual([]);
  });
  it('recognizes card metadata without treating every foul as a card', () => {
    expect(buildHighlights(events([{ time: 40, possession: 'FO', card: 'YELLOW_CARD' }]), times)).toHaveLength(1);
  });
  it('automatically skips gaps, carries elapsed time and stops at the last highlight', () => {
    const clips = buildHighlights(events([{ time: 12, possession: 'SH' }, { time: 40, game: 'RC' }]), times);
    expect(nearestHighlightTime(clips, 0)).toBe(7000);
    expect(nearestHighlightTime(clips, 25000)).toBe(35000);
    expect(advanceHighlightPlayback(clips, 16000, 2000)).toEqual({ timeMs: 36000, ended: false });
    expect(advanceHighlightPlayback(clips, 16000, 30000)).toEqual({ timeMs: 45000, ended: true });
    expect(advanceHighlightPlayback([], 100, 10)).toEqual({ timeMs: 100, ended: true });
  });
  it('maps the highlights-only seek bar to source timestamps without including skipped match time', () => {
    const clips = buildHighlights(events([{ time: 12, possession: 'SH' }, { time: 40, game: 'RC' }]), times);
    expect(highlightProgress(clips, 40000)).toBe(15000);
    expect(highlightTimeAtProgress(clips, 15000)).toBe(40000);
    expect(highlightTimeAtProgress(clips, 10000)).toBe(35000);
    expect(highlightTimeAtProgress(clips, 99999)).toBe(45000);
  });
});

import { findFrameByTime, type MatchEvent } from '@replay/shared';

export interface HighlightClip {
  startMs: number;
  endMs: number;
  events: Array<{ id: string; timeMs: number; label: string }>;
}

const cardTypes = new Set(['YC', 'RC', '2YC', 'YELLOW', 'RED', 'SECONDYELLOW', 'YELLOWCARD', 'REDCARD', 'SECONDYELLOWCARD', 'CARD', 'BOOKING']);
const code = (value: unknown) => typeof value === 'string' ? value.toUpperCase().replace(/[\s_-]/g, '') : '';

function highlightKind(event: MatchEvent): string | null {
  if (event.possessionType === 'SH' || event.gameType === 'SHOT') return event.outcome === 'G' ? 'Goal candidate' : 'Shot';
  if (event.gameType === 'GOAL') return 'Goal candidate';
  if (event.gameType === 'SUB' || event.gameType === 'SUBSTITUTION') return 'Substitution';
  const game = event.payload.gameEvents;
  const card = game && typeof game === 'object' ? (game as Record<string, unknown>).cardType : null;
  return cardTypes.has(code(event.gameType)) || cardTypes.has(code(card)) ? 'Card' : null;
}

/** Five source seconds either side. Keep only recorded frames and merge overlapping action. */
export function buildHighlights(events: MatchEvent[], times: Float64Array, halfRanges: Array<[number, number]> = []): HighlightClip[] {
  if (!times.length) return [];
  const first = times[0]!; const last = times.at(-1)!;
  const clips: HighlightClip[] = [];
  for (const event of [...events].sort((a, b) => a.eventTimeMs - b.eventTimeMs)) {
    const kind = highlightKind(event); const time = event.eventTimeMs;
    if (!kind || !Number.isFinite(time) || time < first || time > last) continue;
    const half = halfRanges.length ? halfRanges.find(([start, end]) => time >= start && time <= end) : [first, last];
    if (!half) continue;
    const start = Math.max(first, half[0]!, time - 5000); const end = Math.min(last, half[1]!, time + 5000);
    let startIndex = findFrameByTime(times, start);
    if (times[startIndex]! < start) startIndex++;
    const endIndex = findFrameByTime(times, end);
    if (startIndex > endIndex || startIndex >= times.length) continue;
    const item = { id: event.id, timeMs: time,
      label: [kind, event.clock, event.actorPlayerName ?? event.playerName, event.teamName].filter(Boolean).join(' · ') };
    const previous = clips.at(-1);
    if (previous && times[startIndex]! <= previous.endMs) {
      previous.endMs = Math.max(previous.endMs, times[endIndex]!);
      if (!previous.events.some((e) => e.id === item.id)) previous.events.push(item);
    } else clips.push({ startMs: times[startIndex]!, endMs: times[endIndex]!, events: [item] });
  }
  return clips;
}

export function nearestHighlightTime(clips: HighlightClip[], timeMs: number): number {
  for (const clip of clips) {
    if (timeMs <= clip.endMs) return Math.max(timeMs, clip.startMs);
  }
  return clips.at(-1)?.endMs ?? timeMs;
}

export function advanceHighlightPlayback(clips: HighlightClip[], currentMs: number, elapsedMs: number): { timeMs: number; ended: boolean } {
  let timeMs = nearestHighlightTime(clips, currentMs); let remaining = Math.max(0, elapsedMs);
  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i]!;
    if (timeMs > clip.endMs) continue;
    timeMs = Math.max(timeMs, clip.startMs);
    const available = clip.endMs - timeMs;
    if (remaining < available) return { timeMs: timeMs + remaining, ended: false };
    remaining -= available; timeMs = clip.endMs;
    // Advance beyond this clip even when exactly on the boundary.
    const next = clips[i + 1];
    if (next) timeMs = next.startMs;
    else return { timeMs, ended: true };
  }
  return { timeMs, ended: true };
}

export function highlightProgress(clips: HighlightClip[], sourceMs: number): number {
  let progress = 0;
  for (const clip of clips) {
    if (sourceMs <= clip.endMs) return progress + Math.max(0, sourceMs - clip.startMs);
    progress += clip.endMs - clip.startMs;
  }
  return progress;
}

export function highlightTimeAtProgress(clips: HighlightClip[], progressMs: number): number {
  let remaining = Math.max(0, progressMs);
  for (const clip of clips) {
    const duration = clip.endMs - clip.startMs;
    if (remaining < duration) return clip.startMs + remaining;
    remaining -= duration;
  }
  return clips.at(-1)?.endMs ?? 0;
}

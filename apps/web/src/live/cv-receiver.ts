import { WorkloadAccumulator, packFrames, parseTrackingLine, type ParsedFrame, type PlayerWorkload, type TrackingData } from '@replay/shared';
import { CV_LIMITS } from '../../../api/src/cv-types';
import { validateCvBatch } from '../../../api/src/cv-validation';

export const CV_API_BASE = 'https://tracking-cv-api.youssefmansi2.workers.dev';
export function cvReadUrl(id: string): string {
  if (!/^[a-f0-9]{32}$/.test(id)) throw new Error('Enter the 32-character session ID returned by the API.');
  return `${CV_API_BASE}/api/cv/sessions/${id}/frames?limit=300`;
}
export interface CvViewUpdate {
  data?: TrackingData;
  workloads?: PlayerWorkload[];
  playedFrames: number; bufferedFrames: number; bufferedSeconds: number;
  processedFrames: number; missingFrames: number; receivedFrames: number;
  lastReceivedAt: string | null; clockComplete: boolean; warnings: string[];
}
const object = (v: unknown): v is Record<string, unknown> => Boolean(v) && typeof v === 'object' && !Array.isArray(v);
const count = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0;
const invalid = (): never => { throw new Error('The API returned invalid or inconsistent tracking data. Reconnect to a valid session.'); };

/** Receipt and playback are independent: only played frames contribute workload. */
export class CvReceiver {
  private readonly workload = new WorkloadAccumulator();
  private recent: ParsedFrame[] = [];
  private received = 0; private processed = 0; private missing = 0;
  private lastFrame = -1; private lastTime = -Infinity;
  private clockComplete = true;
  private queue: Array<{ frame: ParsedFrame; clock: boolean; gap: boolean }> = [];
  private played = 0;
  private due: number | null = null;
  private lastStatsTime = -Infinity;
  private lastReceivedAt: string | null = null;
  private warnings: string[] = [];
  // Two minutes at 30 FPS. Stop explicitly on overflow; never silently drop queued playback.
  static readonly maxQueuedFrames = 3600;
  constructor(private readonly id: string) { cvReadUrl(id); }
  consume(value: unknown): CvViewUpdate {
    if (!object(value) || value.sessionId !== this.id || !count(value.receivedFrames)
      || !Array.isArray(value.frames) || value.frames.length > CV_LIMITS.retainedFrames
      || value.retainedFrames !== value.frames.length || value.receivedFrames < value.frames.length
      || value.receivedFrames < this.received || !Array.isArray(value.warnings)
      || value.warnings.some((w) => !['MISSING_MATCH_CLOCK', 'MISSING_PLAYER_QUALITY'].includes(w))
      || !(value.receivedAt === null || (typeof value.receivedAt === 'string' && Number.isFinite(Date.parse(value.receivedAt))))) invalid();
    const input = value as Record<string, unknown> & { receivedFrames: number; frames: Record<string, unknown>[]; warnings: string[]; receivedAt: string | null };
    if (input.receivedFrames > 0 && !input.frames.length) invalid();
    if (input.frames.length) {
      try { validateCvBatch({ sequence: 0, frames: input.frames }); } catch { invalid(); }
    }
    const fresh = input.frames.filter((f) => (f.frameNum as number) > this.lastFrame && (f.videoTimeMs as number) > this.lastTime);
    const delta = input.receivedFrames - this.received;
    if (fresh.length > delta || (delta > 0 && fresh.length === 0)) invalid();
    const parsed = fresh.map((f) => {
      const result = parseTrackingLine(JSON.stringify(f));
      if (result.kind !== 'frame' || result.overflowPlayers) return invalid();
      return result.frame;
    });
    // All response validation precedes accumulator mutation.
    const missed = delta - fresh.length;
    if (this.queue.length + parsed.length > CvReceiver.maxQueuedFrames) {
      throw new Error('Playback buffer is full. Reception stopped. Reconnect with slower uploads; queued playback will reset.');
    }
    this.missing += missed;
    for (let i = 0; i < parsed.length; i++) {
      const frame = parsed[i]!; const original = fresh[i]!;
      this.queue.push({ frame, gap: i === 0 && missed > 0,
        clock: (original.period === 1 || original.period === 2) && typeof original.periodGameClockTime === 'number' });
      this.lastFrame = frame.frameNum; this.lastTime = frame.videoTimeMs;
    }
    this.processed += fresh.length; this.received = input.receivedFrames;
    this.lastReceivedAt = input.receivedAt; this.warnings = input.warnings;
    return this.snapshot();
  }

  private snapshot(): CvViewUpdate {
    return { processedFrames: this.processed, playedFrames: this.played, bufferedFrames: this.queue.length,
      bufferedSeconds: this.queue.length > 1 ? (this.queue.at(-1)!.frame.videoTimeMs - this.queue[0]!.frame.videoTimeMs) / 1000 : 0,
      missingFrames: this.missing, receivedFrames: this.received, lastReceivedAt: this.lastReceivedAt,
      clockComplete: this.clockComplete, warnings: this.warnings };
  }

  /** Monotonic playback clock, frozen on underrun. Arrival time never determines movement speed. */
  advance(nowMs: number): CvViewUpdate {
    if (!Number.isFinite(nowMs)) throw new Error('Invalid playback time.');
    if (this.queue.length && this.due === null) this.due = nowMs;
    let changed = false;
    while (this.queue.length && this.due !== null && this.due <= nowMs + 0.001) {
      const item = this.queue.shift()!;
      if (item.gap) { this.workload.breakContinuity(); this.recent = []; }
      this.clockComplete &&= item.clock;
      this.workload.add(item.frame); this.recent.push(item.frame); this.played++; changed = true;
      this.recent = this.recent.slice(-90);
      const next = this.queue[0];
      // Missing history / long source stoppages are not invented movement or minutes of blank playback.
      const delta = next ? next.frame.videoTimeMs - item.frame.videoTimeMs : 0;
      this.due = next ? this.due + (next.gap || delta > 1000 ? 0 : delta) : null;
    }
    const update = this.snapshot();
    if (changed) {
      update.data = packFrames(this.recent, { malformedLines: 0, blankLines: 0, overflowPlayers: 0 }, []);
      const sourceTime = this.recent.at(-1)!.videoTimeMs;
      // Send growing analytics timelines once per source second, not with every pitch frame.
      if (sourceTime - this.lastStatsTime >= 1000 || !this.queue.length) {
        update.workloads = this.workload.finish(); this.lastStatsTime = sourceTime;
      }
    }
    return update;
  }
}

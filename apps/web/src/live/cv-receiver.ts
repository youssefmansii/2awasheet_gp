import { WorkloadAccumulator, packFrames, parseTrackingLine, type ParsedFrame, type PlayerWorkload, type TrackingData } from '@replay/shared';
import { CV_LIMITS } from '../../../api/src/cv-types';
import { validateCvBatch } from '../../../api/src/cv-validation';

export const CV_API_BASE = 'https://tracking-cv-api.youssefmansi2.workers.dev';
export function cvReadUrl(id: string): string {
  if (!/^[a-f0-9]{32}$/.test(id)) throw new Error('Enter the 32-character session ID returned by the API.');
  return `${CV_API_BASE}/api/cv/sessions/${id}/frames?limit=300`;
}
/** Cursor read of the durable batch archive: the first stored batch after `after`. */
export function cvBatchUrl(id: string, after: number): string {
  cvReadUrl(id);
  if (!Number.isSafeInteger(after) || after < -1) throw new Error('Invalid batch cursor.');
  return `${CV_API_BASE}/api/cv/sessions/${id}/batches/next?after=${after}`;
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
  private lastSequence = -1;
  // Two minutes at 30 FPS. Stop explicitly on overflow; never silently drop queued playback.
  static readonly maxQueuedFrames = 3600;
  constructor(private readonly id: string) { cvReadUrl(id); }

  /** Sequence of the last consumed archived batch (-1 before the first): the `after` cursor for the next read. */
  get cursor(): number { return this.lastSequence; }
  /** Backpressure: only fetch another batch when a full-size one fits in the playback queue. */
  canAcceptBatch(): boolean { return this.queue.length + CV_LIMITS.batchFrames <= CvReceiver.maxQueuedFrames; }

  private validSession(value: unknown): value is Record<string, unknown> & { receivedFrames: number; warnings: string[]; receivedAt: string | null } {
    return object(value) && value.sessionId === this.id && count(value.receivedFrames)
      && value.receivedFrames >= this.received && Array.isArray(value.warnings)
      && !value.warnings.some((w) => !['MISSING_MATCH_CLOCK', 'MISSING_PLAYER_QUALITY'].includes(w))
      && (value.receivedAt === null || (typeof value.receivedAt === 'string' && Number.isFinite(Date.parse(value.receivedAt))));
  }

  private parse(frames: Record<string, unknown>[]): ParsedFrame[] {
    return frames.map((f) => {
      const result = parseTrackingLine(JSON.stringify(f));
      if (result.kind !== 'frame' || result.overflowPlayers) return invalid();
      return result.frame;
    });
  }

  private enqueue(fresh: Record<string, unknown>[], parsed: ParsedFrame[], missed: number) {
    this.missing += missed;
    for (let i = 0; i < parsed.length; i++) {
      const frame = parsed[i]!; const original = fresh[i]!;
      this.queue.push({ frame, gap: i === 0 && missed > 0,
        clock: (original.period === 1 || original.period === 2) && typeof original.periodGameClockTime === 'number' });
      this.lastFrame = frame.frameNum; this.lastTime = frame.videoTimeMs;
    }
    this.processed += fresh.length;
  }

  /**
   * Consume one response of GET /batches/next: exactly one archived batch (or batch=null when none is newer).
   * Batches are read one by one from the server's durable store, so nothing is overwritten between polls.
   */
  consumeBatch(value: unknown): CvViewUpdate {
    if (!this.validSession(value) || !('batch' in value)) return invalid();
    const batch = value.batch;
    if (batch !== null) {
      if (!object(batch) || !Number.isSafeInteger(batch.sequence) || (batch.sequence as number) <= this.lastSequence
        || !count(batch.frameOffset) || !Array.isArray(batch.frames) || batch.frames.length < 1
        || batch.frames.length > CV_LIMITS.batchFrames || batch.frameCount !== batch.frames.length
        || (batch.frameOffset as number) < this.processed + this.missing
        || (batch.frameOffset as number) + batch.frames.length > value.receivedFrames) invalid();
      const { sequence, frameOffset, frames } = batch as { sequence: number; frameOffset: number; frames: Record<string, unknown>[] };
      try { validateCvBatch({ sequence, frames }); } catch { invalid(); }
      // The archive is exact: a batch must start strictly after everything already consumed.
      if ((frames[0]!.frameNum as number) <= this.lastFrame || (frames[0]!.videoTimeMs as number) <= this.lastTime) invalid();
      const parsed = this.parse(frames);
      if (this.queue.length + parsed.length > CvReceiver.maxQueuedFrames) {
        throw new Error('Playback buffer is full. Reception stopped. Reconnect with slower uploads; queued playback will reset.');
      }
      // All response validation precedes accumulator mutation.
      this.enqueue(frames, parsed, frameOffset - (this.processed + this.missing));
      this.lastSequence = sequence;
    }
    this.received = value.receivedFrames; this.lastReceivedAt = value.receivedAt; this.warnings = value.warnings;
    return this.snapshot();
  }

  /** Legacy tail-snapshot mode (GET /frames?limit=300). Can miss frames if uploads outpace polling. */
  consume(value: unknown): CvViewUpdate {
    if (!this.validSession(value) || !Array.isArray(value.frames) || value.frames.length > CV_LIMITS.retainedFrames
      || value.retainedFrames !== value.frames.length || value.receivedFrames < value.frames.length) invalid();
    const input = value as Record<string, unknown> & { receivedFrames: number; frames: Record<string, unknown>[]; warnings: string[]; receivedAt: string | null };
    if (input.receivedFrames > 0 && !input.frames.length) invalid();
    if (input.frames.length) {
      try { validateCvBatch({ sequence: 0, frames: input.frames }); } catch { invalid(); }
    }
    const fresh = input.frames.filter((f) => (f.frameNum as number) > this.lastFrame && (f.videoTimeMs as number) > this.lastTime);
    const delta = input.receivedFrames - this.received;
    if (fresh.length > delta || (delta > 0 && fresh.length === 0)) invalid();
    const parsed = this.parse(fresh);
    // All response validation precedes accumulator mutation.
    const missed = delta - fresh.length;
    if (this.queue.length + parsed.length > CvReceiver.maxQueuedFrames) {
      throw new Error('Playback buffer is full. Reception stopped. Reconnect with slower uploads; queued playback will reset.');
    }
    this.enqueue(fresh, parsed, missed);
    this.received = input.receivedFrames;
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

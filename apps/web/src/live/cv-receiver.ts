import { WorkloadAccumulator, packFrames, parseTrackingLine, type ParsedFrame, type TrackingData } from '@replay/shared';
import { validateCvBatch } from '../../../api/src/cv-validation';

export const CV_API_BASE = 'https://tracking-cv-api.youssefmansi2.workers.dev';
export function cvReadUrl(id: string): string {
  if (!/^[a-f0-9]{32}$/.test(id)) throw new Error('Enter the 32-character session ID returned by the API.');
  return `${CV_API_BASE}/api/cv/sessions/${id}/frames?limit=90`;
}
export interface CvViewUpdate {
  data?: TrackingData;
  processedFrames: number; missingFrames: number; receivedFrames: number;
  lastReceivedAt: string | null; clockComplete: boolean; warnings: string[];
}
const object = (v: unknown): v is Record<string, unknown> => Boolean(v) && typeof v === 'object' && !Array.isArray(v);
const count = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0;
const invalid = (): never => { throw new Error('The API returned invalid or inconsistent tracking data. Reconnect to a valid session.'); };

/** Incremental observed-only analytics; snapshots overlap, so never recount a frame. */
export class CvReceiver {
  private readonly workload = new WorkloadAccumulator();
  private recent: ParsedFrame[] = [];
  private received = 0; private processed = 0; private missing = 0;
  private lastFrame = -1; private lastTime = -Infinity;
  private clockComplete = true;
  constructor(private readonly id: string) { cvReadUrl(id); }
  consume(value: unknown): CvViewUpdate {
    if (!object(value) || value.sessionId !== this.id || !count(value.receivedFrames)
      || !Array.isArray(value.frames) || value.frames.length > 90
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
    if (missed > 0) { this.missing += missed; this.workload.breakContinuity(); this.recent = []; }
    for (let i = 0; i < parsed.length; i++) {
      const frame = parsed[i]!; const original = fresh[i]!;
      this.clockComplete &&= (original.period === 1 || original.period === 2) && typeof original.periodGameClockTime === 'number';
      this.workload.add(frame); this.recent.push(frame);
      this.lastFrame = frame.frameNum; this.lastTime = frame.videoTimeMs;
    }
    this.recent = this.recent.slice(-90); this.processed += fresh.length; this.received = input.receivedFrames;
    const update: CvViewUpdate = { processedFrames: this.processed, missingFrames: this.missing, receivedFrames: this.received,
      lastReceivedAt: input.receivedAt, clockComplete: this.clockComplete, warnings: input.warnings };
    if (fresh.length) update.data = packFrames(this.recent, { malformedLines: 0, blankLines: 0, overflowPlayers: 0 }, this.workload.finish());
    return update;
  }
}

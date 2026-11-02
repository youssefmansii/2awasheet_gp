import { BALL_FLOATS, PLAYER_FLOATS, TOTAL_SLOTS, formatMatchClock, type PlayerWorkload, type TrackingData } from '@replay/shared';

/** Same rule as live playback: known missing history and long source stoppages are skipped, not waited through. */
export const LIVE_MAX_PLAYBACK_GAP_MS = 1000;

type Buffers = {
  frameNum: Int32Array; videoTimeMs: Float64Array; period: Uint8Array; periodGameClockTime: Float64Array;
  rawPlayers: Float32Array; smoothedPlayers: Float32Array; rawJerseys: Uint8Array; smoothedJerseys: Uint8Array;
  rawFlags: Uint8Array; smoothedFlags: Uint8Array; rawBall: Float32Array; smoothedBall: Float32Array;
  rawBallVisible: Uint8Array; smoothedBallVisible: Uint8Array; gapBefore: Uint8Array;
};
const PER_FRAME: Record<keyof Buffers, number> = {
  frameNum: 1, videoTimeMs: 1, period: 1, periodGameClockTime: 1,
  rawPlayers: PLAYER_FLOATS, smoothedPlayers: PLAYER_FLOATS, rawJerseys: TOTAL_SLOTS, smoothedJerseys: TOTAL_SLOTS,
  rawFlags: TOTAL_SLOTS, smoothedFlags: TOTAL_SLOTS, rawBall: BALL_FLOATS, smoothedBall: BALL_FLOATS,
  rawBallVisible: 1, smoothedBallVisible: 1, gapBefore: 1
};
const KEYS = Object.keys(PER_FRAME) as Array<keyof Buffers>;
type TypedArray = Buffers[keyof Buffers];

function allocate(capacity: number): Buffers {
  return {
    frameNum: new Int32Array(capacity), videoTimeMs: new Float64Array(capacity), period: new Uint8Array(capacity),
    periodGameClockTime: new Float64Array(capacity), rawPlayers: new Float32Array(capacity * PLAYER_FLOATS),
    smoothedPlayers: new Float32Array(capacity * PLAYER_FLOATS), rawJerseys: new Uint8Array(capacity * TOTAL_SLOTS),
    smoothedJerseys: new Uint8Array(capacity * TOTAL_SLOTS), rawFlags: new Uint8Array(capacity * TOTAL_SLOTS),
    smoothedFlags: new Uint8Array(capacity * TOTAL_SLOTS), rawBall: new Float32Array(capacity * BALL_FLOATS),
    smoothedBall: new Float32Array(capacity * BALL_FLOATS), rawBallVisible: new Uint8Array(capacity),
    smoothedBallVisible: new Uint8Array(capacity), gapBefore: new Uint8Array(capacity)
  };
}

/**
 * Every frame that has reached the live edge, kept for the whole session so the viewer can seek back
 * like a live video (DVR). Stored in the same packed layout as TrackingData (~0.65 KB per frame, about
 * 105 MB for 90 minutes at 30 FPS). Capacity doubles; views are zero-copy subarrays.
 */
export class LiveHistory {
  private buffers = allocate(0);
  private capacity = 0;
  private count = 0;
  private gameRefId: string | null = null;
  private starts: Array<{ index: number; period: number }> = [];

  get frameCount(): number { return this.count; }

  clear(): void { this.buffers = allocate(0); this.capacity = 0; this.count = 0; this.gameRefId = null; this.starts = []; }

  /** Appends frames in played order. `gapBefore[i] = 1` marks known missing history before chunk frame i. */
  append(chunk: TrackingData, gapBefore?: Uint8Array): void {
    const added = chunk.frameCount;
    if (!added) return;
    if (gapBefore && gapBefore.length !== added) throw new Error('Gap markers must match the appended frames.');
    const first = chunk.videoTimeMs[0] ?? 0;
    if (this.count && first <= (this.buffers.videoTimeMs[this.count - 1] ?? 0)) throw new Error('Live history must increase in source time.');
    this.reserve(this.count + added);
    for (const key of KEYS) {
      if (key === 'gapBefore') continue;
      const per = PER_FRAME[key];
      (this.buffers[key] as TypedArray).set((chunk[key] as TypedArray).subarray(0, added * per) as never, this.count * per);
    }
    if (gapBefore) this.buffers.gapBefore.set(gapBefore, this.count);
    for (let i = Math.max(1, this.count); i < this.count + added; i++) {
      const period = this.buffers.period[i] ?? 0;
      if (period && period !== this.buffers.period[i - 1]) this.starts.push({ index: i, period });
    }
    this.gameRefId ??= chunk.gameRefId;
    this.count += added;
  }

  /** Zero-copy TrackingData of everything received so far. */
  view(workloads: PlayerWorkload[]): TrackingData {
    const n = this.count; const b = this.buffers;
    const cut = <T extends TypedArray>(values: T, per: number): T => values.subarray(0, n * per) as T;
    return {
      frameCount: n, malformedLines: 0, blankLines: 0, overflowPlayers: 0, gameRefId: this.gameRefId,
      frameNum: cut(b.frameNum, 1), videoTimeMs: cut(b.videoTimeMs, 1), period: cut(b.period, 1),
      periodGameClockTime: cut(b.periodGameClockTime, 1), rawPlayers: cut(b.rawPlayers, PLAYER_FLOATS),
      smoothedPlayers: cut(b.smoothedPlayers, PLAYER_FLOATS), rawJerseys: cut(b.rawJerseys, TOTAL_SLOTS),
      smoothedJerseys: cut(b.smoothedJerseys, TOTAL_SLOTS), rawFlags: cut(b.rawFlags, TOTAL_SLOTS),
      smoothedFlags: cut(b.smoothedFlags, TOTAL_SLOTS), rawBall: cut(b.rawBall, BALL_FLOATS),
      smoothedBall: cut(b.smoothedBall, BALL_FLOATS), rawBallVisible: cut(b.rawBallVisible, 1),
      smoothedBallVisible: cut(b.smoothedBallVisible, 1), workloads
    };
  }

  videoTimeAt(index: number): number { return this.buffers.videoTimeMs[this.clamp(index)] ?? 0; }

  /** Source milliseconds to wait before showing `index` after `index - 1`; gaps are skipped. */
  waitBefore(index: number): number {
    if (index <= 0 || index >= this.count || this.buffers.gapBefore[index]) return 0;
    const delta = (this.buffers.videoTimeMs[index] ?? 0) - (this.buffers.videoTimeMs[index - 1] ?? 0);
    return delta > LIVE_MAX_PLAYBACK_GAP_MS ? 0 : delta;
  }

  /** Last frame at or before a source time (first frame if the time is earlier than all history). */
  indexAtOrBefore(videoTimeMs: number): number {
    const times = this.buffers.videoTimeMs; let low = 0; let high = this.count - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1;
      if ((times[middle] ?? 0) <= videoTimeMs) low = middle + 1; else high = middle - 1;
    }
    return Math.max(0, high);
  }

  /** First frame at or after a source time (last frame if the time is later than all history). */
  indexAtOrAfter(videoTimeMs: number): number {
    const times = this.buffers.videoTimeMs; let low = 0; let high = this.count - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1;
      if ((times[middle] ?? 0) < videoTimeMs) low = middle + 1; else high = middle - 1;
    }
    return Math.min(Math.max(0, this.count - 1), low);
  }

  /** Index `seconds` of source time away from `index`, used by the ±5 s / ±10 s controls. */
  shift(index: number, seconds: number): number {
    if (!this.count) return 0;
    const target = this.videoTimeAt(index) + seconds * 1000;
    return seconds < 0 ? this.indexAtOrBefore(target) : this.indexAtOrAfter(target);
  }

  /** First received frame of each period after the first, for seek-bar markers (e.g. half-time). */
  periodStarts(): ReadonlyArray<{ index: number; period: number }> { return this.starts; }

  /** Match clock (e.g. 47:05) when the CV feed supplies it, otherwise source video time. */
  label(index: number, clockComplete: boolean): string {
    if (!this.count) return '00:00';
    const i = this.clamp(index);
    if (clockComplete) return formatMatchClock(this.buffers.periodGameClockTime[i] ?? 0, this.buffers.period[i] ?? 0);
    const seconds = Math.max(0, Math.floor((this.buffers.videoTimeMs[i] ?? 0) / 1000));
    const hours = Math.floor(seconds / 3600); const minutes = Math.floor(seconds / 60) % 60; const rest = seconds % 60;
    const mmss = `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`;
    return `Video ${hours ? `${hours}:` : ''}${mmss}`;
  }

  private clamp(index: number): number { return Math.min(Math.max(0, Math.floor(index)), Math.max(0, this.count - 1)); }

  private reserve(required: number): void {
    if (required <= this.capacity) return;
    let next = Math.max(4096, this.capacity);
    while (next < required) next *= 2;
    const grown = allocate(next);
    for (const key of KEYS) (grown[key] as TypedArray).set((this.buffers[key] as TypedArray).subarray(0, this.count * PER_FRAME[key]) as never);
    this.buffers = grown; this.capacity = next;
  }
}

/**
 * Advances a behind-live viewer at source speed. Returns the new frame index and unused elapsed time.
 * Reaching `liveIndex` means the viewer has caught up with the live edge.
 */
export function stepHistoryPlayback(history: LiveHistory, index: number, carryMs: number, liveIndex: number): { index: number; carryMs: number } {
  let next = index; let carry = carryMs;
  while (next < liveIndex) {
    const wait = history.waitBefore(next + 1);
    if (carry + 0.001 < wait) break;
    carry -= wait; next++;
  }
  return { index: next, carryMs: next >= liveIndex ? 0 : carry };
}

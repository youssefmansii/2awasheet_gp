import {
  PLAYER_SLOTS, TOTAL_SLOTS, TrackingFlag, type ParsedFrame, type PlayerWorkload,
  type PlayerWorkloadSnapshot, type WorkloadBucket, type WorkloadConfig
} from './types';

type EffortState = -1 | 0 | 1;
export type OutputTrendStatus = 'Stable' | 'Watch' | 'High decline' | 'Building baseline' | 'Low data quality' | 'Not on pitch';
interface MutableWorkload extends Omit<PlayerWorkload, 'buckets' | 'timeline' | 'speedP95Mps'> {
  speedHistogram: Uint32Array;
  buckets: Map<number, WorkloadBucket>;
  lastSampleTimeMs: number;
  lastVideoTimeMs: number;
  lastMatchClockSeconds: number;
  timeline: { [K in keyof PlayerWorkload['timeline']]: number[] };
}
interface PreviousPoint {
  x: number; y: number; timeMs: number; speed: number;
  speedHistory: Array<{ timeMs: number; speed: number }>;
  sprintCandidateStartMs: number | null; sprintConfirmed: boolean;
  accelerationCandidate: EffortState; accelerationCandidateStartMs: number | null;
  accelerationState: EffortState;
}

export const DEFAULT_WORKLOAD_CONFIG: Readonly<WorkloadConfig> = Object.freeze({
  highSpeedThresholdMps: 5.5,
  sprintThresholdMps: 7,
  accelerationThresholdMps2: 2.5,
  maxValidSpeedMps: 12.5,
  sprintMinDurationSeconds: 1,
  accelerationMinDurationSeconds: 0.5,
  accelerationWindowSeconds: 0.5,
  accelerationResetMps2: 1,
  maxFrameGapSeconds: 0.2,
  checkpointIntervalMs: 1000,
  histogramStepMps: 0.1,
  minimumTrendQualityPercent: 80
});

export function resolveWorkloadConfig(input: Partial<WorkloadConfig> = {}): WorkloadConfig {
  const config = { ...DEFAULT_WORKLOAD_CONFIG, ...input };
  for (const [name, value] of Object.entries(config)) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be greater than zero.`);
  }
  if (config.highSpeedThresholdMps >= config.sprintThresholdMps) {
    throw new Error('The high-speed threshold must be below the sprint threshold.');
  }
  if (config.sprintThresholdMps >= config.maxValidSpeedMps) {
    throw new Error('The sprint threshold must be below the maximum valid speed.');
  }
  if (config.accelerationResetMps2 >= config.accelerationThresholdMps2) {
    throw new Error('The acceleration reset must be below the acceleration threshold.');
  }
  if (config.minimumTrendQualityPercent > 100) {
    throw new Error('The minimum trend quality cannot exceed 100%.');
  }
  return config;
}

export class WorkloadAccumulator {
  private readonly players = new Map<string, MutableWorkload>();
  private readonly previous = new Map<string, PreviousPoint>();
  readonly config: WorkloadConfig;
  private readonly histogramBins: number;

  constructor(config: Partial<WorkloadConfig> = {}) {
    this.config = resolveWorkloadConfig(config);
    this.histogramBins = Math.ceil(this.config.maxValidSpeedMps / this.config.histogramStepMps) + 1;
  }

  /** Preserve observed totals, but never bridge motion across a known missing-data interval. */
  breakContinuity(): void { this.previous.clear(); }

  private percentile95(workload: MutableWorkload): number {
    const count = workload.speedHistogram.reduce((sum, value) => sum + value, 0);
    if (!count) return 0;
    const target = Math.ceil(count * 0.95);
    let cumulative = 0;
    for (let index = 0; index < workload.speedHistogram.length; index += 1) {
      cumulative += workload.speedHistogram[index] ?? 0;
      if (cumulative >= target) return index * this.config.histogramStepMps;
    }
    return 0;
  }

  private capture(workload: MutableWorkload, force = false): void {
    if (!force && workload.lastSampleTimeMs >= 0
      && workload.lastVideoTimeMs - workload.lastSampleTimeMs < this.config.checkpointIntervalMs) return;
    if (force && workload.timeline.videoTimeMs.at(-1) === workload.lastVideoTimeMs) return;
    const timeline = workload.timeline;
    timeline.videoTimeMs.push(workload.lastVideoTimeMs);
    timeline.matchClockSeconds.push(workload.lastMatchClockSeconds);
    timeline.observedSeconds.push(workload.observedSeconds);
    timeline.distanceM.push(workload.distanceM);
    timeline.highSpeedDistanceM.push(workload.highSpeedDistanceM);
    timeline.sprintDistanceM.push(workload.sprintDistanceM);
    timeline.sprintCount.push(workload.sprintCount);
    timeline.accelerationCount.push(workload.accelerationCount);
    timeline.decelerationCount.push(workload.decelerationCount);
    timeline.maxSpeedMps.push(workload.maxSpeedMps);
    timeline.speedP95Mps.push(this.percentile95(workload));
    timeline.fadedObservations.push(workload.fadedObservations);
    timeline.totalObservations.push(workload.totalObservations);
    workload.lastSampleTimeMs = workload.lastVideoTimeMs;
  }

  private createWorkload(key: string, team: 0 | 1, jerseyNum: number, frame: ParsedFrame, matchClock: number): MutableWorkload {
    const workload: MutableWorkload = {
      key, team, jerseyNum, observedSeconds: 0, distanceM: 0, highSpeedDistanceM: 0,
      sprintDistanceM: 0, sprintCount: 0, accelerationCount: 0, decelerationCount: 0,
      maxSpeedMps: 0, fadedObservations: 0, totalObservations: 0,
      speedHistogram: new Uint32Array(this.histogramBins), buckets: new Map(),
      lastSampleTimeMs: -1, lastVideoTimeMs: frame.videoTimeMs, lastMatchClockSeconds: matchClock,
      timeline: {
        videoTimeMs: [], matchClockSeconds: [], observedSeconds: [], distanceM: [],
        highSpeedDistanceM: [], sprintDistanceM: [], sprintCount: [], accelerationCount: [],
        decelerationCount: [], maxSpeedMps: [], speedP95Mps: [], fadedObservations: [], totalObservations: []
      }
    };
    this.players.set(key, workload);
    return workload;
  }

  private updateSprint(previous: PreviousPoint, speed: number, segmentStartMs: number, timeMs: number, workload: MutableWorkload, bucket: WorkloadBucket): void {
    if (speed < this.config.sprintThresholdMps) {
      previous.sprintCandidateStartMs = null;
      previous.sprintConfirmed = false;
      return;
    }
    if (previous.sprintCandidateStartMs === null) previous.sprintCandidateStartMs = segmentStartMs;
    if (!previous.sprintConfirmed
      && timeMs - previous.sprintCandidateStartMs >= this.config.sprintMinDurationSeconds * 1000) {
      workload.sprintCount += 1;
      bucket.sprintCount += 1;
      previous.sprintConfirmed = true;
    }
  }

  private windowAcceleration(previous: PreviousPoint, timeMs: number, speed: number): number | null {
    previous.speedHistory.push({ timeMs, speed });
    const target = timeMs - this.config.accelerationWindowSeconds * 1000;
    while (previous.speedHistory.length > 1 && (previous.speedHistory[1]?.timeMs ?? Infinity) <= target) {
      previous.speedHistory.shift();
    }
    const start = previous.speedHistory[0];
    if (!start) return null;
    const seconds = (timeMs - start.timeMs) / 1000;
    if (seconds < this.config.accelerationWindowSeconds * 0.8) return null;
    return (speed - start.speed) / seconds;
  }

  private updateAcceleration(previous: PreviousPoint, acceleration: number | null, timeMs: number, workload: MutableWorkload, bucket: WorkloadBucket): void {
    if (acceleration === null) return;
    const desired: EffortState = acceleration >= this.config.accelerationThresholdMps2
      ? 1 : acceleration <= -this.config.accelerationThresholdMps2 ? -1 : 0;
    if (desired !== 0 && desired !== previous.accelerationState) {
      if (previous.accelerationCandidate !== desired) {
        previous.accelerationCandidate = desired;
        previous.accelerationCandidateStartMs = timeMs;
      } else if (previous.accelerationCandidateStartMs !== null
        && timeMs - previous.accelerationCandidateStartMs >= this.config.accelerationMinDurationSeconds * 1000) {
        if (desired === 1) { workload.accelerationCount += 1; bucket.accelerationCount += 1; }
        else { workload.decelerationCount += 1; bucket.decelerationCount += 1; }
        previous.accelerationState = desired;
        previous.accelerationCandidate = 0;
        previous.accelerationCandidateStartMs = null;
      }
      return;
    }
    previous.accelerationCandidate = 0;
    previous.accelerationCandidateStartMs = null;
    if (Math.abs(acceleration) < this.config.accelerationResetMps2) previous.accelerationState = 0;
  }

  add(frame: ParsedFrame): void {
    const matchClock = frame.period === 2 && frame.periodGameClockTime < 2700
      ? frame.periodGameClockTime + 2700 : frame.periodGameClockTime;
    const bucketNumber = Math.max(0, Math.floor(matchClock / 300));
    for (let slot = 0; slot < TOTAL_SLOTS; slot += 1) {
      const jerseyNum = frame.smoothedJerseys[slot] ?? 0;
      if (!jerseyNum) continue;
      const offset = slot * 2;
      const x = frame.smoothedPlayers[offset];
      const y = frame.smoothedPlayers[offset + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const team = (slot >= PLAYER_SLOTS ? 1 : 0) as 0 | 1;
      const key = `${team}:${jerseyNum}`;
      const workload = this.players.get(key) ?? this.createWorkload(key, team, jerseyNum, frame, matchClock);
      workload.totalObservations += 1;
      workload.lastVideoTimeMs = frame.videoTimeMs;
      workload.lastMatchClockSeconds = matchClock;
      if (frame.smoothedFlags[slot] === TrackingFlag.Faded) workload.fadedObservations += 1;
      let bucket = workload.buckets.get(bucketNumber);
      if (!bucket) {
        bucket = { bucket: bucketNumber, period: frame.period, observedSeconds: 0, distanceM: 0,
          highSpeedDistanceM: 0, sprintDistanceM: 0, sprintCount: 0, accelerationCount: 0, decelerationCount: 0 };
        workload.buckets.set(bucketNumber, bucket);
      }

      const previous = this.previous.get(key);
      if (previous) {
        const seconds = (frame.videoTimeMs - previous.timeMs) / 1000;
        if (seconds > 0 && seconds <= this.config.maxFrameGapSeconds) {
          const distance = Math.hypot(x! - previous.x, y! - previous.y);
          const speed = distance / seconds;
          if (Number.isFinite(speed) && speed <= this.config.maxValidSpeedMps) {
            workload.observedSeconds += seconds;
            workload.distanceM += distance;
            workload.maxSpeedMps = Math.max(workload.maxSpeedMps, speed);
            workload.speedHistogram[Math.min(this.histogramBins - 1, Math.floor(speed / this.config.histogramStepMps))]! += 1;
            bucket.observedSeconds += seconds;
            bucket.distanceM += distance;
            if (speed >= this.config.sprintThresholdMps) {
              workload.sprintDistanceM += distance; bucket.sprintDistanceM += distance;
            } else if (speed >= this.config.highSpeedThresholdMps) {
              workload.highSpeedDistanceM += distance; bucket.highSpeedDistanceM += distance;
            }
            this.updateSprint(previous, speed, previous.timeMs, frame.videoTimeMs, workload, bucket);
            this.updateAcceleration(previous, this.windowAcceleration(previous, frame.videoTimeMs, speed), frame.videoTimeMs, workload, bucket);
            previous.x = x!; previous.y = y!; previous.timeMs = frame.videoTimeMs; previous.speed = speed;
            this.capture(workload);
            continue;
          }
        }
      }
      this.previous.set(key, {
        x: x!, y: y!, timeMs: frame.videoTimeMs, speed: 0,
        speedHistory: [{ timeMs: frame.videoTimeMs, speed: 0 }],
        sprintCandidateStartMs: null, sprintConfirmed: false,
        accelerationCandidate: 0, accelerationCandidateStartMs: null, accelerationState: 0
      });
      this.capture(workload);
    }
  }

  finish(): PlayerWorkload[] {
    return [...this.players.values()].map((workload) => {
      this.capture(workload, true);
      const timeline = workload.timeline;
      return {
        key: workload.key, team: workload.team, jerseyNum: workload.jerseyNum,
        observedSeconds: workload.observedSeconds, distanceM: workload.distanceM,
        highSpeedDistanceM: workload.highSpeedDistanceM, sprintDistanceM: workload.sprintDistanceM,
        sprintCount: workload.sprintCount, accelerationCount: workload.accelerationCount,
        decelerationCount: workload.decelerationCount, maxSpeedMps: workload.maxSpeedMps,
        speedP95Mps: this.percentile95(workload), fadedObservations: workload.fadedObservations,
        totalObservations: workload.totalObservations,
        buckets: [...workload.buckets.values()].sort((a, b) => a.bucket - b.bucket),
        timeline: {
          videoTimeMs: Float64Array.from(timeline.videoTimeMs),
          matchClockSeconds: Float32Array.from(timeline.matchClockSeconds),
          observedSeconds: Float32Array.from(timeline.observedSeconds),
          distanceM: Float32Array.from(timeline.distanceM),
          highSpeedDistanceM: Float32Array.from(timeline.highSpeedDistanceM),
          sprintDistanceM: Float32Array.from(timeline.sprintDistanceM),
          sprintCount: Uint32Array.from(timeline.sprintCount),
          accelerationCount: Uint32Array.from(timeline.accelerationCount),
          decelerationCount: Uint32Array.from(timeline.decelerationCount),
          maxSpeedMps: Float32Array.from(timeline.maxSpeedMps),
          speedP95Mps: Float32Array.from(timeline.speedP95Mps),
          fadedObservations: Uint32Array.from(timeline.fadedObservations),
          totalObservations: Uint32Array.from(timeline.totalObservations)
        }
      };
    }).sort((a, b) => a.team - b.team || a.jerseyNum - b.jerseyNum);
  }
}

export function workloadTransferables(workloads: PlayerWorkload[]): Transferable[] {
  return workloads.flatMap((workload) => Object.values(workload.timeline).map((values) => values.buffer));
}

export function trackingQualityBetween(start: PlayerWorkloadSnapshot, end: PlayerWorkloadSnapshot): number {
  const observations = end.totalObservations - start.totalObservations;
  if (observations <= 0) return 0;
  const faded = end.fadedObservations - start.fadedObservations;
  return Math.max(0, Math.min(1, 1 - faded / observations));
}

export function classifyOutputTrend(input: {
  distanceChangePercent: number | null;
  highIntensityChangePercent: number | null;
  sprintChangePercent: number | null;
  onPitch: boolean;
  quality: number | null;
  minimumQuality: number;
}): OutputTrendStatus {
  if (!input.onPitch) return 'Not on pitch';
  if (input.quality !== null && input.quality < input.minimumQuality) return 'Low data quality';
  if (input.distanceChangePercent === null || input.highIntensityChangePercent === null) return 'Building baseline';
  if (input.distanceChangePercent <= -20
    && (input.highIntensityChangePercent <= -25 || (input.sprintChangePercent !== null && input.sprintChangePercent <= -30))) return 'High decline';
  if (input.distanceChangePercent <= -10 || input.highIntensityChangePercent <= -15
    || (input.sprintChangePercent !== null && input.sprintChangePercent <= -20)) return 'Watch';
  return 'Stable';
}

export function workloadSnapshotAt(workload: PlayerWorkload, videoTimeMs: number): PlayerWorkloadSnapshot | null {
  const times = workload.timeline.videoTimeMs;
  if (!times.length || videoTimeMs < (times[0] ?? 0)) return null;
  let low = 0; let high = times.length - 1;
  while (low <= high) {
    const middle = (low + high) >>> 1;
    if ((times[middle] ?? 0) <= videoTimeMs) low = middle + 1;
    else high = middle - 1;
  }
  const sampleIndex = Math.max(0, high);
  const timeline = workload.timeline;
  return {
    sampleIndex, videoTimeMs: timeline.videoTimeMs[sampleIndex] ?? 0,
    matchClockSeconds: timeline.matchClockSeconds[sampleIndex] ?? 0,
    observedSeconds: timeline.observedSeconds[sampleIndex] ?? 0,
    distanceM: timeline.distanceM[sampleIndex] ?? 0,
    highSpeedDistanceM: timeline.highSpeedDistanceM[sampleIndex] ?? 0,
    sprintDistanceM: timeline.sprintDistanceM[sampleIndex] ?? 0,
    sprintCount: timeline.sprintCount[sampleIndex] ?? 0,
    accelerationCount: timeline.accelerationCount[sampleIndex] ?? 0,
    decelerationCount: timeline.decelerationCount[sampleIndex] ?? 0,
    maxSpeedMps: timeline.maxSpeedMps[sampleIndex] ?? 0,
    speedP95Mps: timeline.speedP95Mps[sampleIndex] ?? 0,
    fadedObservations: timeline.fadedObservations[sampleIndex] ?? 0,
    totalObservations: timeline.totalObservations[sampleIndex] ?? 0
  };
}

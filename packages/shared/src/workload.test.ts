import { describe, expect, it } from 'vitest';
import { parseTrackingLine } from './parser';
import { WorkloadAccumulator, classifyOutputTrend, trackingQualityBetween, workloadSnapshotAt } from './workload';

function parsedFrame(time: number, x: number, period = 1, clock = time / 1000) {
  const result = parseTrackingLine(JSON.stringify({
    frameNum: time,
    videoTimeMs: time,
    period,
    periodGameClockTime: clock,
    homePlayers: [{ jerseyNum: '10', x, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }],
    homePlayersSmoothed: [{ jerseyNum: '10', x, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }],
    awayPlayers: [],
    awayPlayersSmoothed: [],
    balls: []
  }));
  if (result.kind !== 'frame') throw new Error('Expected frame');
  return result.frame;
}

describe('WorkloadAccumulator', () => {
  it('keeps high-speed and sprint distance in distinct zones', () => {
    const accumulator = new WorkloadAccumulator();
    accumulator.add(parsedFrame(0, 0));
    accumulator.add(parsedFrame(100, 0.6));
    accumulator.add(parsedFrame(200, 1.4));
    const workload = accumulator.finish()[0]!;
    expect(workload.distanceM).toBeCloseTo(1.4);
    expect(workload.highSpeedDistanceM).toBeCloseTo(0.6);
    expect(workload.sprintDistanceM).toBeCloseTo(0.8);
    expect(workload.sprintCount).toBe(0);
    expect(workload.accelerationCount).toBe(0);
    expect(workload.totalObservations).toBe(3);
    expect(workload.timeline.videoTimeMs.length).toBeGreaterThan(0);
    expect(workloadSnapshotAt(workload, 200)?.distanceM).toBeCloseTo(1.4);
  });

  it.each([
    [0.9, 0],
    [1.1, 1]
  ])('counts a continuous sprint lasting %.1f seconds as %i bout(s)', (duration, expected) => {
    const accumulator = new WorkloadAccumulator();
    for (let time = 0; time <= duration * 1000; time += 100) {
      accumulator.add(parsedFrame(time, time / 1000 * 8));
    }
    expect(accumulator.finish()[0]?.sprintCount).toBe(expected);
  });

  it('uses configurable thresholds and minimum sprint duration', () => {
    const accumulator = new WorkloadAccumulator({ sprintThresholdMps: 6, sprintMinDurationSeconds: 0.5 });
    for (let time = 0; time <= 600; time += 100) accumulator.add(parsedFrame(time, time / 1000 * 6.5));
    expect(accumulator.finish()[0]?.sprintCount).toBe(1);
  });

  it('counts acceleration only after the windowed threshold is sustained', () => {
    const accumulator = new WorkloadAccumulator();
    let x = 0;
    accumulator.add(parsedFrame(0, x));
    for (let time = 100; time <= 1600; time += 100) {
      const speed = 3 * time / 1000;
      x += speed * 0.1;
      accumulator.add(parsedFrame(time, x));
    }
    expect(accumulator.finish()[0]?.accelerationCount).toBe(1);
  });

  it('returns cumulative values only up to the requested live checkpoint', () => {
    const accumulator = new WorkloadAccumulator();
    for (let time = 0; time <= 2000; time += 100) {
      const x = time <= 1000 ? time / 1000 : 1 + (time - 1000) / 500;
      accumulator.add(parsedFrame(time, x));
    }
    const workload = accumulator.finish()[0]!;
    expect(workloadSnapshotAt(workload, 1000)?.distanceM).toBeCloseTo(1);
    expect(workloadSnapshotAt(workload, 2000)?.distanceM).toBeCloseTo(3);
  });

  it('calculates quality for the selected trend window rather than the full match', () => {
    const snapshot = (totalObservations: number, fadedObservations: number) => ({
      sampleIndex: 0, videoTimeMs: 0, matchClockSeconds: 0, observedSeconds: 0,
      distanceM: 0, highSpeedDistanceM: 0, sprintDistanceM: 0, sprintCount: 0,
      accelerationCount: 0, decelerationCount: 0, maxSpeedMps: 0, speedP95Mps: 0,
      totalObservations, fadedObservations
    });
    expect(trackingQualityBetween(snapshot(100, 50), snapshot(200, 70))).toBeCloseTo(0.8);
    expect(trackingQualityBetween(snapshot(100, 50), snapshot(100, 50))).toBe(0);
  });

  it('gates output status by on-pitch state and window quality', () => {
    const base = { distanceChangePercent: 0, highIntensityChangePercent: 0, sprintChangePercent: 0, onPitch: true, quality: 0.79, minimumQuality: 0.8 };
    expect(classifyOutputTrend(base)).toBe('Low data quality');
    expect(classifyOutputTrend({ ...base, onPitch: false, quality: 1 })).toBe('Not on pitch');
    expect(classifyOutputTrend({ ...base, quality: 0.8 })).toBe('Stable');
  });

  it('offsets a restarting period-two clock and rejects halftime movement', () => {
    const accumulator = new WorkloadAccumulator();
    accumulator.add(parsedFrame(0, 0, 1, 2699));
    accumulator.add(parsedFrame(100, 0.1, 1, 2699.1));
    accumulator.add(parsedFrame(1_000_000, 30, 2, 0));
    accumulator.add(parsedFrame(1_000_100, 30.1, 2, 0.1));
    const workload = accumulator.finish()[0]!;
    expect(workload.distanceM).toBeCloseTo(0.2);
    expect(Array.from(workload.timeline.matchClockSeconds)).toEqual([2699, 2700, 2700.10009765625]);
    expect(workload.buckets.map((bucket) => bucket.bucket)).toEqual([8, 9]);
  });
});

import {
  BALL_FLOATS,
  PLAYER_FLOATS,
  PLAYER_SLOTS,
  TOTAL_SLOTS,
  TrackingFlag,
  type BallInput,
  type FrameInput,
  type ParsedFrame,
  type PlayerInput,
  type TrackingData
} from './types';
import { WorkloadAccumulator } from './workload';

export type LineResult =
  | { kind: 'blank' }
  | { kind: 'malformed'; error: string }
  | { kind: 'frame'; frame: ParsedFrame; overflowPlayers: number };

function finite(value: unknown, fallback = Number.NaN): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function writePlayers(
  destination: Float32Array,
  jerseys: Uint8Array,
  flags: Uint8Array,
  players: PlayerInput[] | null | undefined,
  teamOffset: number
): number {
  if (!Array.isArray(players)) return 0;
  const count = Math.min(players.length, PLAYER_SLOTS);
  for (let index = 0; index < count; index += 1) {
    const player = players[index];
    if (!player) continue;
    const slot = teamOffset + index;
    destination[slot * 2] = finite(player.x);
    destination[slot * 2 + 1] = finite(player.y);
    const parsedJersey = Number.parseInt(player.jerseyNum, 10);
    jerseys[slot] = Number.isInteger(parsedJersey) && parsedJersey > 0 && parsedJersey <= 255
      ? parsedJersey
      : 0;
    flags[slot] = player.visibility === 'ESTIMATED' || player.confidence === 'LOW'
      ? TrackingFlag.Faded
      : TrackingFlag.None;
  }
  return Math.max(0, players.length - PLAYER_SLOTS);
}

function writeBall(ball: BallInput | null | undefined): {
  values: Float32Array;
  visible: number;
} {
  const values = new Float32Array(BALL_FLOATS);
  values.fill(Number.NaN);
  if (!ball) return { values, visible: 0 };
  values[0] = finite(ball.x);
  values[1] = finite(ball.y);
  values[2] = finite(ball.z, 0);
  return {
    values,
    visible: Number.isFinite(values[0]) && Number.isFinite(values[1]) && ball.visibility !== 'NOT_VISIBLE' ? 1 : 0
  };
}

function createPlayerArrays(): [Float32Array, Uint8Array, Uint8Array] {
  const positions = new Float32Array(PLAYER_FLOATS);
  positions.fill(Number.NaN);
  return [positions, new Uint8Array(TOTAL_SLOTS), new Uint8Array(TOTAL_SLOTS)];
}

export function parseTrackingLine(line: string): LineResult {
  if (!line.trim()) return { kind: 'blank' };
  let input: FrameInput;
  try {
    input = JSON.parse(line) as FrameInput;
  } catch (error) {
    return { kind: 'malformed', error: error instanceof Error ? error.message : 'Invalid JSON' };
  }
  if (!input || typeof input !== 'object' || !Number.isFinite(input.frameNum) || !Number.isFinite(input.videoTimeMs)) {
    return { kind: 'malformed', error: 'Frame is missing frameNum or videoTimeMs' };
  }

  const [rawPlayers, rawJerseys, rawFlags] = createPlayerArrays();
  const [smoothedPlayers, smoothedJerseys, smoothedFlags] = createPlayerArrays();
  let overflowPlayers = 0;
  overflowPlayers += writePlayers(rawPlayers, rawJerseys, rawFlags, input.homePlayers, 0);
  overflowPlayers += writePlayers(rawPlayers, rawJerseys, rawFlags, input.awayPlayers, PLAYER_SLOTS);
  overflowPlayers += writePlayers(
    smoothedPlayers,
    smoothedJerseys,
    smoothedFlags,
    input.homePlayersSmoothed ?? input.homePlayers,
    0
  );
  overflowPlayers += writePlayers(
    smoothedPlayers,
    smoothedJerseys,
    smoothedFlags,
    input.awayPlayersSmoothed ?? input.awayPlayers,
    PLAYER_SLOTS
  );
  const rawBall = writeBall(input.balls?.[0]);
  const smoothedBall = writeBall(input.ballsSmoothed ?? input.balls?.[0]);

  return {
    kind: 'frame',
    overflowPlayers,
    frame: {
      frameNum: input.frameNum,
      videoTimeMs: input.videoTimeMs,
      period: finite(input.period, 0),
      periodGameClockTime: finite(input.periodGameClockTime, 0),
      gameRefId: input.gameRefId == null ? null : String(input.gameRefId).replace(/\.0$/, ''),
      rawPlayers,
      smoothedPlayers,
      rawJerseys,
      smoothedJerseys,
      rawFlags,
      smoothedFlags,
      rawBall: rawBall.values,
      smoothedBall: smoothedBall.values,
      rawBallVisible: rawBall.visible,
      smoothedBallVisible: smoothedBall.visible
    }
  };
}

export function packFrames(
  frames: ParsedFrame[],
  stats: { malformedLines: number; blankLines: number; overflowPlayers: number },
  workloads?: TrackingData['workloads']
): TrackingData {
  const count = frames.length;
  const workload = new WorkloadAccumulator();
  if (!workloads) frames.forEach((frame) => workload.add(frame));
  const data: TrackingData = {
    frameCount: count,
    malformedLines: stats.malformedLines,
    blankLines: stats.blankLines,
    overflowPlayers: stats.overflowPlayers,
    gameRefId: frames.find((frame) => frame.gameRefId)?.gameRefId ?? null,
    frameNum: new Int32Array(count),
    videoTimeMs: new Float64Array(count),
    period: new Uint8Array(count),
    periodGameClockTime: new Float64Array(count),
    rawPlayers: new Float32Array(count * PLAYER_FLOATS),
    smoothedPlayers: new Float32Array(count * PLAYER_FLOATS),
    rawJerseys: new Uint8Array(count * TOTAL_SLOTS),
    smoothedJerseys: new Uint8Array(count * TOTAL_SLOTS),
    rawFlags: new Uint8Array(count * TOTAL_SLOTS),
    smoothedFlags: new Uint8Array(count * TOTAL_SLOTS),
    rawBall: new Float32Array(count * BALL_FLOATS),
    smoothedBall: new Float32Array(count * BALL_FLOATS),
    rawBallVisible: new Uint8Array(count),
    smoothedBallVisible: new Uint8Array(count),
    workloads: workloads ?? workload.finish()
  };

  frames.forEach((frame, index) => {
    data.frameNum[index] = frame.frameNum;
    data.videoTimeMs[index] = frame.videoTimeMs;
    data.period[index] = frame.period;
    data.periodGameClockTime[index] = frame.periodGameClockTime;
    data.rawPlayers.set(frame.rawPlayers, index * PLAYER_FLOATS);
    data.smoothedPlayers.set(frame.smoothedPlayers, index * PLAYER_FLOATS);
    data.rawJerseys.set(frame.rawJerseys, index * TOTAL_SLOTS);
    data.smoothedJerseys.set(frame.smoothedJerseys, index * TOTAL_SLOTS);
    data.rawFlags.set(frame.rawFlags, index * TOTAL_SLOTS);
    data.smoothedFlags.set(frame.smoothedFlags, index * TOTAL_SLOTS);
    data.rawBall.set(frame.rawBall, index * BALL_FLOATS);
    data.smoothedBall.set(frame.smoothedBall, index * BALL_FLOATS);
    data.rawBallVisible[index] = frame.rawBallVisible;
    data.smoothedBallVisible[index] = frame.smoothedBallVisible;
  });
  return data;
}

export function trackingTransferables(data: TrackingData): ArrayBuffer[] {
  const timelineBuffers = data.workloads.flatMap((workload) => Object.values(workload.timeline).map((values) => values.buffer));
  return [
    data.frameNum.buffer,
    data.videoTimeMs.buffer,
    data.period.buffer,
    data.periodGameClockTime.buffer,
    data.rawPlayers.buffer,
    data.smoothedPlayers.buffer,
    data.rawJerseys.buffer,
    data.smoothedJerseys.buffer,
    data.rawFlags.buffer,
    data.smoothedFlags.buffer,
    data.rawBall.buffer,
    data.smoothedBall.buffer,
    data.rawBallVisible.buffer,
    data.smoothedBallVisible.buffer,
    ...timelineBuffers
  ] as ArrayBuffer[];
}

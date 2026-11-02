export const PLAYER_SLOTS = 15;
export const TEAM_COUNT = 2;
export const TOTAL_SLOTS = PLAYER_SLOTS * TEAM_COUNT;
export const PLAYER_FLOATS = TOTAL_SLOTS * 2;
export const BALL_FLOATS = 3;

export const enum TrackingFlag {
  None = 0,
  Faded = 1
}

export interface PlayerInput {
  jerseyNum: string;
  confidence?: string | null;
  visibility?: string | null;
  x: number;
  y: number;
  [key: string]: unknown;
}

export interface BallInput {
  visibility?: string | null;
  x: number;
  y: number;
  z?: number | null;
  [key: string]: unknown;
}

export interface FrameInput {
  frameNum: number;
  videoTimeMs: number;
  period?: number | null;
  periodGameClockTime?: number | null;
  gameRefId?: string | number | null;
  homePlayers?: PlayerInput[] | null;
  awayPlayers?: PlayerInput[] | null;
  homePlayersSmoothed?: PlayerInput[] | null;
  awayPlayersSmoothed?: PlayerInput[] | null;
  balls?: BallInput[] | null;
  ballsSmoothed?: BallInput | null;
  game_event?: Record<string, unknown> | null;
  possession_event?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface ParsedFrame {
  frameNum: number;
  videoTimeMs: number;
  period: number;
  periodGameClockTime: number;
  gameRefId: string | null;
  rawPlayers: Float32Array;
  smoothedPlayers: Float32Array;
  rawJerseys: Uint8Array;
  smoothedJerseys: Uint8Array;
  rawFlags: Uint8Array;
  smoothedFlags: Uint8Array;
  rawBall: Float32Array;
  smoothedBall: Float32Array;
  rawBallVisible: number;
  smoothedBallVisible: number;
}

export interface TrackingData {
  frameCount: number;
  malformedLines: number;
  blankLines: number;
  overflowPlayers: number;
  gameRefId: string | null;
  frameNum: Int32Array;
  videoTimeMs: Float64Array;
  period: Uint8Array;
  periodGameClockTime: Float64Array;
  rawPlayers: Float32Array;
  smoothedPlayers: Float32Array;
  rawJerseys: Uint8Array;
  smoothedJerseys: Uint8Array;
  rawFlags: Uint8Array;
  smoothedFlags: Uint8Array;
  rawBall: Float32Array;
  smoothedBall: Float32Array;
  rawBallVisible: Uint8Array;
  smoothedBallVisible: Uint8Array;
  workloads: PlayerWorkload[];
}

export interface WorkloadConfig {
  highSpeedThresholdMps: number;
  sprintThresholdMps: number;
  accelerationThresholdMps2: number;
  maxValidSpeedMps: number;
  sprintMinDurationSeconds: number;
  accelerationMinDurationSeconds: number;
  accelerationWindowSeconds: number;
  accelerationResetMps2: number;
  maxFrameGapSeconds: number;
  checkpointIntervalMs: number;
  histogramStepMps: number;
  minimumTrendQualityPercent: number;
}

export interface WorkloadBucket {
  bucket: number;
  period: number;
  observedSeconds: number;
  distanceM: number;
  highSpeedDistanceM: number;
  sprintDistanceM: number;
  sprintCount: number;
  accelerationCount: number;
  decelerationCount: number;
}

export interface PlayerWorkload {
  key: string;
  team: 0 | 1;
  jerseyNum: number;
  observedSeconds: number;
  distanceM: number;
  highSpeedDistanceM: number;
  sprintDistanceM: number;
  sprintCount: number;
  accelerationCount: number;
  decelerationCount: number;
  maxSpeedMps: number;
  speedP95Mps: number;
  fadedObservations: number;
  totalObservations: number;
  buckets: WorkloadBucket[];
  timeline: WorkloadTimeline;
}

export interface WorkloadTimeline {
  videoTimeMs: Float64Array;
  matchClockSeconds: Float32Array;
  observedSeconds: Float32Array;
  distanceM: Float32Array;
  highSpeedDistanceM: Float32Array;
  sprintDistanceM: Float32Array;
  sprintCount: Uint32Array;
  accelerationCount: Uint32Array;
  decelerationCount: Uint32Array;
  maxSpeedMps: Float32Array;
  speedP95Mps: Float32Array;
  fadedObservations: Uint32Array;
  totalObservations: Uint32Array;
}

export interface PlayerWorkloadSnapshot {
  sampleIndex: number;
  videoTimeMs: number;
  matchClockSeconds: number;
  observedSeconds: number;
  distanceM: number;
  highSpeedDistanceM: number;
  sprintDistanceM: number;
  sprintCount: number;
  accelerationCount: number;
  decelerationCount: number;
  maxSpeedMps: number;
  speedP95Mps: number;
  fadedObservations: number;
  totalObservations: number;
}

export interface EventRosterPlayer {
  playerId: string;
  playerName: string;
  team: 0 | 1;
  teamId: string | null;
  teamName: string | null;
  jerseyNum: number;
  position: string | null;
  eventSpeedAverageMps: number;
  eventSpeedMaxMps: number;
  eventSpeedSamples: number;
}

export interface MatchEvent {
  id: string;
  eventTimeMs: number;
  startTimeMs: number;
  endTimeMs: number;
  clock: string;
  period: number;
  gameType: string | null;
  possessionType: string | null;
  teamId: string | null;
  teamName: string | null;
  homeTeam: boolean | null;
  homeBall: boolean | null;
  playerId: string | null;
  playerName: string | null;
  playerOffId: string | null;
  playerOffName: string | null;
  playerOnId: string | null;
  playerOnName: string | null;
  passerPlayerId: string | null;
  passerPlayerName: string | null;
  targetPlayerId: string | null;
  targetPlayerName: string | null;
  receiverPlayerId: string | null;
  receiverPlayerName: string | null;
  actorPlayerId: string | null;
  actorPlayerName: string | null;
  outcome: string | null;
  payload: Record<string, unknown>;
}

export interface PlayerTechnicalStats {
  playerId: string;
  playerName: string;
  team: 0 | 1;
  jerseyNum: number;
  position: string | null;
  involvements: number;
  passes: number;
  completedPasses: number;
  shots: number;
  goalOutcomes: number;
  crosses: number;
  challenges: number;
  clearances: number;
  rebounds: number;
  carries: number;
  touches: number;
  fouls: number;
}

export interface EventDataset {
  gameId: string | null;
  events: MatchEvent[];
  roster: EventRosterPlayer[];
  playerStats: PlayerTechnicalStats[];
  gameTypeCounts: Record<string, number>;
  possessionTypeCounts: Record<string, number>;
  teamEventCounts: Record<string, number>;
  goalCandidates: GoalCandidate[];
}

export interface GoalCandidate {
  eventId: string;
  eventTimeMs: number;
  clock: string;
  period: number;
  teamId: string | null;
  teamName: string | null;
  homeTeam: boolean | null;
  shooterName: string | null;
}

export interface ScoreValidation {
  status: 'matched' | 'mismatch' | 'unknown';
  expectedHome: number | null;
  expectedAway: number | null;
  observedHome: number;
  observedAway: number;
  goalCandidates: GoalCandidate[];
}

export interface TeamMetadata {
  id?: string;
  name?: string;
  shortName?: string;
}

export interface KitMetadata {
  primaryColor?: string;
  primaryTextColor?: string;
}

export interface MatchMetadata {
  id?: string;
  date?: string;
  competition?: { name?: string };
  stadium?: {
    name?: string;
    pitches?: Array<{ length?: number; width?: number }>;
  };
  homeTeam?: TeamMetadata;
  awayTeam?: TeamMetadata;
  homeTeamKit?: KitMetadata;
  awayTeamKit?: KitMetadata;
  homeTeamStartLeft?: boolean;
  fps?: number;
  startPeriod1?: number;
  endPeriod1?: number;
  startPeriod2?: number;
  endPeriod2?: number;
  [key: string]: unknown;
}

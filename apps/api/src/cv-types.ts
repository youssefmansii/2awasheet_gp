/** Diagnostic frames use team + mandatory jerseyNum, matching replay identity. */
export interface CvPlayer {
  x: number;
  y: number;
  trackId?: number | string;
  jerseyNum: string | number;
  confidence?: 'HIGH' | 'MEDIUM' | 'LOW';
  visibility?: 'VISIBLE' | 'ESTIMATED';
  [key: string]: unknown;
}

export interface CvFrame {
  frameNum: number;
  videoTimeMs: number;
  homePlayers: CvPlayer[];
  awayPlayers: CvPlayer[];
  [key: string]: unknown;
}

export interface CvBatch { sequence: number; frames: CvFrame[] }
export type CvWarning = 'MISSING_MATCH_CLOCK' | 'MISSING_PLAYER_QUALITY';
export interface CvReceipt {
  sessionId: string;
  sequence: number;
  acceptedFrames: number;
  receivedFrames: number;
  receivedAt: string;
  warnings: CvWarning[];
  duplicate: boolean;
}

export const CV_LIMITS = Object.freeze({
  bodyBytes: 1024 * 1024, batchFrames: 120, teamPlayers: 15,
  retainedFrames: 90, retainedBytes: 1024 * 1024, receipts: 32, sessions: 4,
  idleMs: 15 * 60 * 1000, lifetimeMs: 2 * 60 * 60 * 1000,
  bodyReadMs: 10000, requestsPerMinute: 1200, failedAuthPerMinute: 30
});

export class CvError extends Error {
  constructor(readonly status: number, readonly code: string, message: string) { super(message); }
}

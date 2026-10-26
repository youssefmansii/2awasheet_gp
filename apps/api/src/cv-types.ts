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
  // Retained data plus wrappers/32 bounded receipts must fit Cloudflare's 2 MB SQLite row.
  bodyBytes: 1536 * 1024, batchFrames: 300, teamPlayers: 15,
  retainedFrames: 300, retainedBytes: 1536 * 1024, receipts: 32, sessions: 4,
  idleMs: 15 * 60 * 1000, lifetimeMs: 2 * 60 * 60 * 1000,
  bodyReadMs: 10000, requestsPerMinute: 1200, failedAuthPerMinute: 30,
  // Durable batch archive (Cloudflare SQLite): every accepted batch is stored per session until
  // the session expires/is deleted. 4 sessions x 768 MiB stays under the 5 GB free SQLite quota.
  archiveBytes: 768 * 1024 * 1024, batchListLimit: 100
});

export class CvError extends Error {
  constructor(readonly status: number, readonly code: string, message: string) { super(message); }
}

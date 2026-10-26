import { createHash } from 'node:crypto';
import { CV_LIMITS, CvError, type CvBatch, type CvFrame, type CvReceipt, type CvWarning } from '../../api/src/cv-types';
import { frameMatchId } from '../../api/src/cv-validation';

export interface Session {
  id: string; createdAt: number; updatedAt: number; receivedAt: string | null;
  lastSequence: number | null; receivedFrames: number; bytes: number;
  frames: Array<{ frame: CvFrame; bytes: number }>;
  warnings: CvWarning[];
  receipts: Array<{ sequence: number; hash: string; response: CvReceipt }>;
  gameRefId: string | null;
  /** Durable archive counters: every accepted batch is also stored as its own SQLite row. */
  storedBatches: number; archivedBytes: number;
}

/** One archived batch as returned by the batch endpoints (frames omitted in listings). */
export interface StoredBatchSummary {
  sequence: number; frameOffset: number; frameCount: number;
  firstFrameNum: number; lastFrameNum: number; firstVideoTimeMs: number; lastVideoTimeMs: number;
  bytes: number; receivedAt: string;
}

/** Sessions saved before the archive existed lack its counters. */
export function restoreSession(data: string): Session {
  return { storedBatches: 0, archivedBytes: 0, ...(JSON.parse(data) as Partial<Session>) } as Session;
}

export function batchSummary(batch: CvBatch, frameOffset: number, bytes: number, receivedAt: string): StoredBatchSummary {
  const first = batch.frames[0]!; const last = batch.frames.at(-1)!;
  return { sequence: batch.sequence, frameOffset, frameCount: batch.frames.length,
    firstFrameNum: first.frameNum, lastFrameNum: last.frameNum,
    firstVideoTimeMs: first.videoTimeMs, lastVideoTimeMs: last.videoTimeMs, bytes, receivedAt };
}

export function newSession(id: string, now: number): Session {
  return { id, createdAt: now, updatedAt: now, receivedAt: null, lastSequence: null,
    receivedFrames: 0, bytes: 0, frames: [], warnings: [], receipts: [], gameRefId: null,
    storedBatches: 0, archivedBytes: 0 };
}

export function expired(s: Session, now: number): boolean {
  return now >= s.createdAt + CV_LIMITS.lifetimeMs || now >= s.updatedAt + CV_LIMITS.idleMs;
}

export function metadata(s: Session) {
  return { sessionId: s.id, createdAt: new Date(s.createdAt).toISOString(),
    expiresAt: new Date(s.createdAt + CV_LIMITS.lifetimeMs).toISOString(),
    idleExpiresAt: new Date(s.updatedAt + CV_LIMITS.idleMs).toISOString(),
    receivedAt: s.receivedAt, lastSequence: s.lastSequence, receivedFrames: s.receivedFrames,
    retainedFrames: s.frames.length, warnings: s.warnings,
    storedBatches: s.storedBatches, archivedBytes: s.archivedBytes };
}

/** Caller validates the batch, loads a fresh snapshot, and persists synchronously in one SQL transaction. */
export function acceptBatch(s: Session, batch: CvBatch, now: number): CvReceipt {
  const hash = createHash('sha256').update(JSON.stringify(batch)).digest('hex');
  const old = s.receipts.find((r) => r.sequence === batch.sequence);
  if (old) {
    if (old.hash !== hash) throw new CvError(409, 'SEQUENCE_CONFLICT', 'Sequence already accepted with different content.');
    return { ...old.response, duplicate: true };
  }
  if (batch.sequence !== (s.lastSequence ?? -1) + 1) throw new CvError(409, 'SEQUENCE_CONFLICT', 'Expected the next batch sequence; old receipts may have expired.');
  const previous = s.frames.at(-1)?.frame;
  if (previous && (batch.frames[0]!.frameNum <= previous.frameNum || batch.frames[0]!.videoTimeMs <= previous.videoTimeMs)) {
    throw new CvError(409, 'FRAME_OVERLAP', 'Batch overlaps previously accepted timestamps or frame numbers.');
  }
  let gameRefId = s.gameRefId;
  const warnings = new Set<CvWarning>();
  for (const frame of batch.frames) {
    const id = frameMatchId(frame);
    if (id !== null && gameRefId !== null && id !== gameRefId) throw new CvError(409, 'MATCH_CHANGED', 'Match ID cannot change within a session.');
    gameRefId ??= id;
    if (frame.periodGameClockTime == null) warnings.add('MISSING_MATCH_CLOCK');
    for (const key of ['homePlayers', 'awayPlayers', 'homePlayersSmoothed', 'awayPlayersSmoothed']) {
      for (const p of (frame[key] ?? []) as CvFrame['homePlayers']) {
        if (p.confidence == null || p.visibility == null) warnings.add('MISSING_PLAYER_QUALITY');
      }
    }
  }
  const retained = batch.frames.map((frame) => ({ frame, bytes: new TextEncoder().encode(JSON.stringify(frame)).byteLength }));
  // Exact UTF-8 size of JSON.stringify(batch.frames): frames + commas + brackets.
  const archiveBytes = retained.reduce((n, item) => n + item.bytes, 0) + retained.length + 1;
  if (s.archivedBytes + archiveBytes > CV_LIMITS.archiveBytes) {
    throw new CvError(507, 'ARCHIVE_FULL', 'This session has reached its stored-batch limit; create a new session.');
  }
  // No state mutations until the entire batch (including match consistency) passes.
  s.frames.push(...retained); s.bytes += retained.reduce((n, item) => n + item.bytes, 0);
  while (s.frames.length > CV_LIMITS.retainedFrames || s.bytes > CV_LIMITS.retainedBytes) s.bytes -= s.frames.shift()!.bytes;
  s.gameRefId = gameRefId; s.updatedAt = now; s.receivedAt = new Date(now).toISOString();
  s.lastSequence = batch.sequence; s.receivedFrames += batch.frames.length; s.warnings = [...warnings];
  s.storedBatches += 1; s.archivedBytes += archiveBytes;
  const response: CvReceipt = { sessionId: s.id, sequence: batch.sequence, acceptedFrames: batch.frames.length,
    receivedFrames: s.receivedFrames, receivedAt: s.receivedAt, warnings: s.warnings, duplicate: false };
  s.receipts.push({ sequence: batch.sequence, hash, response });
  if (s.receipts.length > CV_LIMITS.receipts) s.receipts.shift();
  return response;
}

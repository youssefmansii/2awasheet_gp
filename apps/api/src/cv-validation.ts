import { CV_LIMITS, CvError, type CvBatch, type CvFrame } from './cv-types';

const object = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const number = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const fail = (path: string, message: string): never => { throw new CvError(422, 'INVALID_FRAME', `${path}: ${message}`); };

function validateJsonTree(value: unknown, depth = 0): void {
  if (depth > 64) throw new CvError(422, 'INVALID_JSON_VALUE', 'JSON nesting exceeds 64 levels.');
  if (typeof value === 'number' && !Number.isFinite(value)) throw new CvError(422, 'INVALID_JSON_VALUE', 'All JSON numbers must be finite.');
  if (value && typeof value === 'object') for (const child of Object.values(value)) validateJsonTree(child, depth + 1);
}

function players(value: unknown, path: string): void {
  if (!Array.isArray(value) || value.length > CV_LIMITS.teamPlayers) fail(path, 'expected an array of at most 15 players.');
  const tracks = new Set<string>(); const jerseys = new Set<number>();
  for (const [index, player] of (value as unknown[]).entries()) {
    const label = `${path}[${index}]`;
    if (!object(player)) fail(label, 'expected a player object.');
    const p = player as Record<string, unknown>;
    if (!number(p.x) || !number(p.y)) fail(label, 'x/y must be finite numbers in pitch metres. Omit unavailable players.');
    if (p.trackId !== undefined) {
      const hasTrack = typeof p.trackId === 'string' ? p.trackId.length > 0 && p.trackId.length <= 128 && p.trackId.trim() === p.trackId
        : typeof p.trackId === 'number' && Number.isSafeInteger(p.trackId) && p.trackId >= 0;
      if (!hasTrack) fail(label, 'trackId must be a non-empty bounded string or non-negative safe integer.');
      // String and numeric representations of the same ID cannot coexist in one team/frame.
      const track = String(p.trackId);
      if (tracks.has(track)) fail(label, 'duplicate trackId within team/frame.');
      tracks.add(track);
    }
    const jersey = typeof p.jerseyNum === 'string' && /^\d{1,3}$/.test(p.jerseyNum) ? Number(p.jerseyNum) : p.jerseyNum;
    if (typeof jersey !== 'number' || !Number.isInteger(jersey) || jersey < 1 || jersey > 255) {
      fail(label, 'jerseyNum is required: use the recognized number 1–255 or its decimal string. Omit unknown players; trackId is not a substitute.');
    }
    if (jerseys.has(jersey as number)) fail(label, 'duplicate jerseyNum within team/frame.');
    jerseys.add(jersey as number);
    if (p.confidence !== undefined && !['HIGH', 'MEDIUM', 'LOW'].includes(p.confidence as string)) fail(label, 'confidence must be HIGH, MEDIUM or LOW.');
    if (p.visibility !== undefined && !['VISIBLE', 'ESTIMATED'].includes(p.visibility as string)) fail(label, 'visibility must be VISIBLE or ESTIMATED; omit unavailable players.');
  }
}

function ball(value: unknown, path: string): void {
  if (!object(value) || !number(value.x) || !number(value.y)) fail(path, 'ball x/y must be finite pitch-metre numbers.');
  const b = value as Record<string, unknown>;
  if (b.z !== undefined && (!number(b.z) || b.z < 0)) fail(path, 'z must be a non-negative height in metres, or omitted.');
  if (b.visibility !== undefined && !['VISIBLE', 'ESTIMATED', 'NOT_VISIBLE'].includes(b.visibility as string)) fail(path, 'invalid ball visibility.');
}

export function validateCvBatch(value: unknown): CvBatch {
  validateJsonTree(value);
  if (!object(value) || !Number.isSafeInteger(value.sequence) || (value.sequence as number) < 0
    || !Array.isArray(value.frames) || value.frames.length < 1 || value.frames.length > CV_LIMITS.batchFrames) {
    throw new CvError(422, 'INVALID_BATCH', 'Provide a non-negative integer sequence and 1–300 frames.');
  }
  let previousTime = -Infinity; let previousFrame = -1;
  for (const [index, frame] of value.frames.entries()) {
    const path = `frames[${index}]`;
    if (!object(frame)) fail(path, 'expected a frame object.');
    const f = frame as Record<string, unknown>;
    if (!Number.isInteger(f.frameNum) || (f.frameNum as number) < 0 || (f.frameNum as number) > 2147483647) fail(path, 'frameNum must be a non-negative 32-bit integer.');
    if (!number(f.videoTimeMs) || f.videoTimeMs < 0 || f.videoTimeMs > Number.MAX_SAFE_INTEGER) fail(path, 'videoTimeMs must be finite non-negative source milliseconds.');
    if ((f.videoTimeMs as number) <= previousTime || (f.frameNum as number) <= previousFrame) fail(path, 'source timestamps and frame numbers must strictly increase.');
    previousTime = f.videoTimeMs as number; previousFrame = f.frameNum as number;
    if (f.period !== undefined && ![0, 1, 2].includes(f.period as number)) fail(path, 'period must be 0, 1 or 2.');
    if (f.periodGameClockTime !== undefined && (!number(f.periodGameClockTime) || f.periodGameClockTime < 0)) fail(path, 'periodGameClockTime must be non-negative match-clock seconds.');
    if (f.gameRefId != null && !((typeof f.gameRefId === 'string' && f.gameRefId.length > 0 && f.gameRefId.length <= 128)
      || (typeof f.gameRefId === 'number' && Number.isSafeInteger(f.gameRefId) && f.gameRefId >= 0))) fail(path, 'invalid gameRefId.');
    players(f.homePlayers, `${path}.homePlayers`); players(f.awayPlayers, `${path}.awayPlayers`);
    for (const key of ['homePlayersSmoothed', 'awayPlayersSmoothed']) if (f[key] !== undefined) players(f[key], `${path}.${key}`);
    if (f.balls !== undefined) {
      if (!Array.isArray(f.balls) || f.balls.length > 1) fail(path, 'balls must contain zero or one ball.');
      for (const b of f.balls as unknown[]) ball(b, `${path}.balls`);
    }
    if (f.ballsSmoothed !== undefined && f.ballsSmoothed !== null) ball(f.ballsSmoothed, `${path}.ballsSmoothed`);
  }
  return value as unknown as CvBatch;
}

export function frameMatchId(frame: CvFrame): string | null {
  return frame.gameRefId == null ? null : String(frame.gameRefId);
}

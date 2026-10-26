import { DurableObject } from 'cloudflare:workers';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { CV_LIMITS, CvError } from '../../api/src/cv-types';
import { cvServerConfig } from '../../api/src/cv-config';
import { validateCvBatch } from '../../api/src/cv-validation';
import { acceptBatch, batchSummary, metadata, newSession, restoreSession, type Session, type StoredBatchSummary } from './session';
import { json, jsonText, failure, readJson, readBytes } from './http';
import { documentationResponse } from './docs';
import { allowedReadOrigin, preflight, withReadCors } from './cors';

interface Env { CV_API_KEY: string; CV_ALLOWED_ORIGINS?: string; CV_COORDINATOR: DurableObjectNamespace<CvCoordinator>; DOCS_ASSETS: Fetcher }
type SessionRow = { id: string; data: string; expires: number };
type RateRow = { start: number; requests: number; failures: number };
type BatchRow = { sequence: number; frame_offset: number; frame_count: number; first_frame: number; last_frame: number;
  first_time: number; last_time: number; bytes: number; received_at: string };
const batchColumns = 'sequence, frame_offset, frame_count, first_frame, last_frame, first_time, last_time, bytes, received_at';
const summary = (r: BatchRow): StoredBatchSummary => ({ sequence: r.sequence, frameOffset: r.frame_offset, frameCount: r.frame_count,
  firstFrameNum: r.first_frame, lastFrameNum: r.last_frame, firstVideoTimeMs: r.first_time, lastVideoTimeMs: r.last_time,
  bytes: r.bytes, receivedAt: r.received_at });
/** Strict decimal query integer, or the default when absent. */
function queryInt(url: URL, name: string, fallback: number, min: number, max: number): number {
  const text = url.searchParams.get(name);
  if (text === null) return fallback;
  const value = Number(text);
  if (!/^-?\d{1,16}$/.test(text) || !Number.isSafeInteger(value) || value < min || value > max) {
    throw new CvError(400, 'INVALID_QUERY', `${name} must be an integer from ${min} to ${max}.`);
  }
  return value;
}

/** One bounded test-team coordinator, not an arbitrary object per untrusted URL. */
export class CvCoordinator extends DurableObject<Env> {
  private readonly keyHash: Buffer;
  private readonly uploads = new Set<string>();
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const config = cvServerConfig({ CV_API_KEY: env.CV_API_KEY });
    this.keyHash = createHash('sha256').update(config.apiKey).digest();
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL, expires INTEGER NOT NULL)');
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS rate (id INTEGER PRIMARY KEY CHECK(id = 1), start INTEGER NOT NULL, requests INTEGER NOT NULL, failures INTEGER NOT NULL)');
    // Durable per-session batch archive: one row per accepted batch, read back one by one by sequence.
    ctx.storage.sql.exec(`CREATE TABLE IF NOT EXISTS batches (
      session_id TEXT NOT NULL, sequence INTEGER NOT NULL, frame_offset INTEGER NOT NULL, frame_count INTEGER NOT NULL,
      first_frame INTEGER NOT NULL, last_frame INTEGER NOT NULL, first_time REAL NOT NULL, last_time REAL NOT NULL,
      bytes INTEGER NOT NULL, received_at TEXT NOT NULL, frames TEXT NOT NULL,
      PRIMARY KEY (session_id, sequence)) WITHOUT ROWID`);
  }

  private prune(now: number) {
    // Look up expired IDs in the tiny sessions table first so batch deletes use the primary key, not a scan.
    const expired = this.ctx.storage.sql.exec<{ id: string }>('SELECT id FROM sessions WHERE expires <= ?', now).toArray();
    for (const { id } of expired) this.remove(id);
  }

  private remove(id: string) {
    this.ctx.storage.sql.exec('DELETE FROM batches WHERE session_id = ?', id);
    this.ctx.storage.sql.exec('DELETE FROM sessions WHERE id = ?', id);
  }

  private batchRow(id: string, where: string, ...params: number[]): { row: BatchRow; frames: string } | null {
    const row = this.ctx.storage.sql.exec<BatchRow & { frames: string }>(
      `SELECT ${batchColumns}, frames FROM batches WHERE session_id = ? AND ${where} ORDER BY sequence LIMIT 1`, id, ...params).toArray()[0];
    return row ? { row, frames: row.frames } : null;
  }

  /** Session metadata plus one batch; the stored frames JSON is spliced in without re-parsing it. */
  private batchResponse(session: Session, found: { row: BatchRow; frames: string } | null): Response {
    if (!found) return json(200, { ...metadata(session), batch: null });
    const head = JSON.stringify({ ...metadata(session), batch: summary(found.row) });
    // head ends with "}}" (closing batch, then the envelope).
    return jsonText(200, `${head.slice(0, -2)},"frames":${found.frames}}}`);
  }

  private save(session: Session) {
    this.ctx.storage.sql.exec('INSERT INTO sessions (id, data, expires) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data, expires = excluded.expires',
      session.id, JSON.stringify(session), Math.min(session.createdAt + CV_LIMITS.lifetimeMs, session.updatedAt + CV_LIMITS.idleMs));
  }

  private load(id: string, now: number): Session {
    const row = this.ctx.storage.sql.exec<SessionRow>('SELECT id, data, expires FROM sessions WHERE id = ?', id).toArray()[0];
    if (!row || row.expires <= now) throw new CvError(404, 'NOT_FOUND', 'Unknown or expired session/endpoint. Create a new session if needed.');
    return restoreSession(row.data);
  }

  private authorize(request: Request, now: number) {
    const authorization = request.headers.get('Authorization');
    const supplied = authorization?.startsWith('Bearer ') ? authorization.slice(7) : '';
    const authenticated = Boolean(supplied) && timingSafeEqual(this.keyHash, createHash('sha256').update(supplied).digest());
    // SQLite, not per-isolate counters: limits survive object eviction/restart.
    const old = this.ctx.storage.sql.exec<RateRow>('SELECT start, requests, failures FROM rate WHERE id = 1').toArray()[0];
    const rate = !old || now - old.start >= 60000 ? { start: now, requests: 0, failures: 0 } : old;
    rate.requests++; if (!authenticated) rate.failures = Math.min(rate.failures + 1, CV_LIMITS.failedAuthPerMinute + 1);
    // Stop rewriting the limiter after rejection, avoiding unbounded database writes.
    if (rate.requests <= CV_LIMITS.requestsPerMinute + 1) {
      this.ctx.storage.sql.exec('INSERT INTO rate (id, start, requests, failures) VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET start = excluded.start, requests = excluded.requests, failures = excluded.failures',
        rate.start, rate.requests, rate.failures);
    }
    if (rate.requests > CV_LIMITS.requestsPerMinute) throw new CvError(429, 'RATE_LIMITED', 'Request limit reached; retry later.');
    if (!authenticated) {
      if (rate.failures > CV_LIMITS.failedAuthPerMinute) throw new CvError(429, 'AUTH_RATE_LIMITED', 'Too many failed authentication attempts.');
      throw new CvError(401, 'UNAUTHORIZED', 'A valid Bearer API key is required.');
    }
    const url = new URL(request.url);
    const origin = request.headers.get('Origin');
    if (origin && origin !== url.origin && !allowedReadOrigin(request, this.env.CV_ALLOWED_ORIGINS)) {
      throw new CvError(403, 'ORIGIN_DENIED', 'This origin or method is not enabled for browser access.');
    }
  }

  async fetch(request: Request): Promise<Response> {
    try {
      const now = Date.now(); this.authorize(request, now); this.prune(now);
      const url = new URL(request.url);
      if (url.pathname === '/api/cv/sessions' && request.method === 'POST') {
        if (Number(request.headers.get('Content-Length') ?? 0) > 0 || (await readBytes(request)).byteLength > 0) {
          throw new CvError(400, 'UNEXPECTED_BODY', 'Session creation takes no body.');
        }
        // Set a cleanup alarm before committing any new data. Reads/uploads never move it.
        if (await this.ctx.storage.getAlarm() === null) await this.ctx.storage.setAlarm(now + 60000);
        const created = this.ctx.storage.transactionSync(() => {
          this.prune(Date.now());
          if (this.ctx.storage.sql.exec<{ n: number }>('SELECT count(*) AS n FROM sessions').one().n >= CV_LIMITS.sessions) {
            throw new CvError(429, 'SESSION_LIMIT', 'Four sessions already active; delete one or wait for expiry.');
          }
          const session = newSession(randomBytes(16).toString('hex'), Date.now()); this.save(session); return session;
        });
        return json(201, metadata(created));
      }
      const route = /^\/api\/cv\/sessions\/([a-f0-9]{32})(?:\/(latest|frames)|\/(batches)(?:\/(next|\d{1,16}))?)?$/.exec(url.pathname);
      if (!route) throw new CvError(404, 'NOT_FOUND', 'Unknown session/endpoint.');
      const id = route[1]!; const action = route[2] ?? route[3]; const batchTarget = route[4]; const session = this.load(id, now);
      if (request.method === 'DELETE' && !action) {
        this.ctx.storage.transactionSync(() => this.remove(id));
        return json(200, { sessionId: id, deleted: true });
      }
      if (request.method === 'GET' && action === 'batches') {
        if (batchTarget === undefined) {
          // Listing: summaries only (no frames), oldest first, for sequences after the cursor.
          const after = queryInt(url, 'after', -1, -1, Number.MAX_SAFE_INTEGER);
          const limit = queryInt(url, 'limit', CV_LIMITS.batchListLimit, 1, CV_LIMITS.batchListLimit);
          const rows = this.ctx.storage.sql.exec<BatchRow>(
            `SELECT ${batchColumns} FROM batches WHERE session_id = ? AND sequence > ? ORDER BY sequence LIMIT ?`, id, after, limit).toArray();
          return json(200, { ...metadata(session), batches: rows.map(summary) });
        }
        if (batchTarget === 'next') {
          // Cursor read: the first stored batch after `after` (default: the first batch), or batch=null if none yet.
          const after = queryInt(url, 'after', -1, -1, Number.MAX_SAFE_INTEGER);
          return this.batchResponse(session, this.batchRow(id, 'sequence > ?', after));
        }
        const sequence = Number(batchTarget);
        if (!Number.isSafeInteger(sequence)) throw new CvError(400, 'INVALID_QUERY', 'sequence must be a non-negative safe integer.');
        const found = this.batchRow(id, 'sequence = ?', sequence);
        if (!found) throw new CvError(404, 'BATCH_NOT_FOUND', 'No stored batch with this sequence in the session (not uploaded yet, or uploaded before archiving).');
        return this.batchResponse(session, found);
      }
      if (request.method === 'GET' && action === 'latest') {
        return json(200, { ...metadata(session), frame: session.frames.at(-1)?.frame ?? null });
      }
      if (request.method === 'GET' && action === 'frames') {
        const text = url.searchParams.get('limit') ?? '30'; const limit = Number(text);
        if (!/^\d+$/.test(text) || !Number.isInteger(limit) || limit < 1 || limit > CV_LIMITS.retainedFrames) throw new CvError(400, 'INVALID_LIMIT', 'limit must be an integer from 1 to 300.');
        return json(200, { ...metadata(session), frames: session.frames.slice(-limit).map((item) => item.frame) });
      }
      if (request.method === 'POST' && action === 'frames') {
        if (this.uploads.has(id)) throw new CvError(409, 'UPLOAD_IN_PROGRESS', 'Retry after the current upload completes.');
        this.uploads.add(id);
        try {
          const batch = validateCvBatch(await readJson(request));
          const receipt = this.ctx.storage.transactionSync(() => {
            // Reload after asynchronous body read: another request may have deleted/expired it.
            const current = this.load(id, Date.now());
            const frameOffset = current.receivedFrames;
            const response = acceptBatch(current, batch, Date.now());
            if (!response.duplicate) {
              // Same transaction: the archive row and session snapshot commit (or roll back) together.
              const frames = JSON.stringify(batch.frames);
              const s = batchSummary(batch, frameOffset, new TextEncoder().encode(frames).byteLength, response.receivedAt);
              this.ctx.storage.sql.exec(`INSERT INTO batches (session_id, ${batchColumns}, frames) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                id, s.sequence, s.frameOffset, s.frameCount, s.firstFrameNum, s.lastFrameNum, s.firstVideoTimeMs, s.lastVideoTimeMs,
                s.bytes, s.receivedAt, frames);
              this.save(current);
            }
            return response;
          });
          return json(200, receipt);
        } finally { this.uploads.delete(id); }
      }
      throw new CvError(405, 'METHOD_NOT_ALLOWED', 'Method not supported on this endpoint.');
    } catch (error) { return failure(error); }
  }

  async alarm() {
    this.ctx.storage.transactionSync(() => this.prune(Date.now()));
    if (this.ctx.storage.sql.exec<{ n: number }>('SELECT count(*) AS n FROM sessions').one().n > 0) {
      await this.ctx.storage.setAlarm(Date.now() + 60000);
    }
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const origin = allowedReadOrigin(request, env.CV_ALLOWED_ORIGINS);
    try {
      const options = preflight(request, env.CV_ALLOWED_ORIGINS);
      if (options) return options;
      const path = new URL(request.url).pathname;
      if (request.method === 'GET') {
        const docs = documentationResponse(path);
        if (docs) return docs;
        if (['/docs-assets/swagger-ui-bundle.js', '/docs-assets/swagger-ui.css', '/docs-assets/LICENSE'].includes(path)) {
          const asset = await env.DOCS_ASSETS.fetch(request);
          const headers = new Headers(asset.headers);
          headers.set('X-Content-Type-Options', 'nosniff');
          headers.set('Cache-Control', 'no-cache');
          return new Response(asset.body, { status: asset.status, headers });
        }
      }
      if (request.method === 'GET' && path === '/health') {
        cvServerConfig({ CV_API_KEY: env.CV_API_KEY });
        return json(200, { status: 'ok', service: 'cv-diagnostic-api', storage: 'durable-sqlite' });
      }
      const response = await env.CV_COORDINATOR.getByName('trusted-team-v1').fetch(request);
      return origin ? withReadCors(response, origin) : response;
    } catch (error) { const response = failure(error); return origin ? withReadCors(response, origin) : response; }
  }
} satisfies ExportedHandler<Env>;

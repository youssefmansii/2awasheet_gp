import { DurableObject } from 'cloudflare:workers';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { CV_LIMITS, CvError } from '../../api/src/cv-types';
import { cvServerConfig } from '../../api/src/cv-config';
import { validateCvBatch } from '../../api/src/cv-validation';
import { acceptBatch, metadata, newSession, type Session } from './session';
import { json, failure, readJson, readBytes } from './http';
import { documentationResponse } from './docs';
import { allowedReadOrigin, preflight, withReadCors } from './cors';

interface Env { CV_API_KEY: string; CV_ALLOWED_ORIGINS?: string; CV_COORDINATOR: DurableObjectNamespace<CvCoordinator>; DOCS_ASSETS: Fetcher }
type SessionRow = { id: string; data: string; expires: number };
type RateRow = { start: number; requests: number; failures: number };

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
  }

  private prune(now: number) {
    this.ctx.storage.sql.exec('DELETE FROM sessions WHERE expires <= ?', now);
  }

  private save(session: Session) {
    this.ctx.storage.sql.exec('INSERT INTO sessions (id, data, expires) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data, expires = excluded.expires',
      session.id, JSON.stringify(session), Math.min(session.createdAt + CV_LIMITS.lifetimeMs, session.updatedAt + CV_LIMITS.idleMs));
  }

  private load(id: string, now: number): Session {
    const row = this.ctx.storage.sql.exec<SessionRow>('SELECT id, data, expires FROM sessions WHERE id = ?', id).toArray()[0];
    if (!row || row.expires <= now) throw new CvError(404, 'NOT_FOUND', 'Unknown or expired session/endpoint. Create a new session if needed.');
    return JSON.parse(row.data) as Session;
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
      const route = /^\/api\/cv\/sessions\/([a-f0-9]{32})(?:\/(latest|frames))?$/.exec(url.pathname);
      if (!route) throw new CvError(404, 'NOT_FOUND', 'Unknown session/endpoint.');
      const id = route[1]!; const action = route[2]; const session = this.load(id, now);
      if (request.method === 'DELETE' && !action) {
        this.ctx.storage.sql.exec('DELETE FROM sessions WHERE id = ?', id);
        return json(200, { sessionId: id, deleted: true });
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
            const response = acceptBatch(current, batch, Date.now());
            if (!response.duplicate) this.save(current);
            return response;
          });
          return json(200, receipt);
        } finally { this.uploads.delete(id); }
      }
      throw new CvError(405, 'METHOD_NOT_ALLOWED', 'Method not supported on this endpoint.');
    } catch (error) { return failure(error); }
  }

  async alarm() {
    this.prune(Date.now());
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

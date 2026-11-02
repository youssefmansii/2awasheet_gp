import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { CV_LIMITS, CvError, type CvFrame, type CvReceipt, type CvWarning } from './cv-types';
import { frameMatchId, validateCvBatch } from './cv-validation';
import { cvServerConfig } from './cv-config';

interface Session {
  id: string;
  createdAt: number;
  updatedAt: number;
  receivedAt: string | null;
  lastSequence: number | null;
  receivedFrames: number;
  frames: Array<{ frame: CvFrame; bytes: number }>;
  bytes: number;
  warnings: CvWarning[];
  receipts: Map<number, { hash: string; response: CvReceipt }>;
  uploading: boolean;
  gameRefId: string | null;
}

export interface CvServiceOptions { apiKey: string; now?: () => number }

export function cvJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
    'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'",
    'Strict-Transport-Security': 'max-age=31536000',
    'Referrer-Policy': 'no-referrer', 'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
  });
  res.end(JSON.stringify(body));
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const mediaType = req.headers['content-type']?.split(';')[0]?.trim().toLowerCase();
  if (mediaType !== 'application/json') throw new CvError(415, 'UNSUPPORTED_MEDIA_TYPE', 'Send application/json.');
  if (req.headers['content-encoding'] && req.headers['content-encoding'] !== 'identity') {
    throw new CvError(415, 'UNSUPPORTED_ENCODING', 'Send uncompressed JSON to the CV endpoint.');
  }
  const chunks: Buffer[] = []; let bytes = 0;
  const buffer = await new Promise<Buffer>((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer); req.off('data', data); req.off('end', end); req.off('aborted', aborted); req.off('error', error);
    };
    const error = (reason: Error) => { cleanup(); req.pause(); reject(reason); };
    const aborted = () => error(new CvError(400, 'REQUEST_ABORTED', 'Upload was interrupted.'));
    const data = (chunk: Buffer) => {
      bytes += chunk.length;
      if (bytes > CV_LIMITS.bodyBytes) error(new CvError(413, 'BODY_TOO_LARGE', 'JSON body exceeds 1.5 MiB.'));
      else chunks.push(chunk);
    };
    const end = () => { cleanup(); resolve(Buffer.concat(chunks)); };
    const timer = setTimeout(() => error(new CvError(408, 'REQUEST_TIMEOUT', 'Upload body timed out.')), CV_LIMITS.bodyReadMs);
    req.on('data', data); req.once('end', end); req.once('aborted', aborted); req.once('error', error);
  });
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer)); }
  catch { throw new CvError(400, 'INVALID_JSON', 'Request body must contain valid UTF-8 JSON.'); }
}

function warningsFor(frames: CvFrame[]): CvWarning[] {
  const warnings = new Set<CvWarning>();
  for (const frame of frames) {
    if (frame.periodGameClockTime == null) warnings.add('MISSING_MATCH_CLOCK');
    for (const key of ['homePlayers', 'awayPlayers', 'homePlayersSmoothed', 'awayPlayersSmoothed']) {
      for (const player of (frame[key] ?? []) as CvFrame['homePlayers']) {
        if (player.confidence == null || player.visibility == null) warnings.add('MISSING_PLAYER_QUALITY');
      }
    }
  }
  return [...warnings];
}

export function createCvService(options: CvServiceOptions) {
  cvServerConfig({ CV_API_KEY: options.apiKey });
  const keyHash = createHash('sha256').update(options.apiKey).digest();
  const now = options.now ?? Date.now;
  const sessions = new Map<string, Session>();
  const expire = () => {
    for (const [id, session] of sessions) if (now() - session.createdAt >= CV_LIMITS.lifetimeMs
      || now() - session.updatedAt >= CV_LIMITS.idleMs) sessions.delete(id);
  };
  const timer = setInterval(expire, 30000); timer.unref();
  let windowStart = now(); let requests = 0; let failures = 0;
  const metadata = (s: Session) => ({
    sessionId: s.id, createdAt: new Date(s.createdAt).toISOString(),
    expiresAt: new Date(s.createdAt + CV_LIMITS.lifetimeMs).toISOString(),
    idleExpiresAt: new Date(s.updatedAt + CV_LIMITS.idleMs).toISOString(),
    receivedAt: s.receivedAt, lastSequence: s.lastSequence, receivedFrames: s.receivedFrames,
    retainedFrames: s.frames.length, warnings: s.warnings
  });
  const handle = async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      if (req.method === 'GET' && url.pathname === '/health') {
        cvJson(res, 200, { status: 'ok', service: 'cv-diagnostic-api', storage: 'memory' }); return;
      }
      if (now() - windowStart >= 60000) { windowStart = now(); requests = 0; failures = 0; }
      if (++requests > CV_LIMITS.requestsPerMinute) throw new CvError(429, 'RATE_LIMITED', 'Request limit reached; retry later.');
      const authorization = req.headers.authorization;
      const supplied = authorization?.startsWith('Bearer ') ? authorization.slice(7) : '';
      if (!supplied || !timingSafeEqual(keyHash, createHash('sha256').update(supplied).digest())) {
        if (++failures > CV_LIMITS.failedAuthPerMinute) throw new CvError(429, 'AUTH_RATE_LIMITED', 'Too many failed authentication attempts.');
        throw new CvError(401, 'UNAUTHORIZED', 'A valid Bearer API key is required.');
      }
      if (req.headers.origin && req.headers.origin !== `https://${req.headers.host}`
        && req.headers.origin !== `http://${req.headers.host}`) throw new CvError(403, 'ORIGIN_DENIED', 'Cross-origin browser access is not enabled.');
      expire();
      if (req.method === 'POST' && url.pathname === '/api/cv/sessions') {
        if (Number(req.headers['content-length'] ?? 0) > 0 || req.headers['transfer-encoding']) {
          throw new CvError(400, 'UNEXPECTED_BODY', 'Session creation takes no body; POST frames to the session frames endpoint.');
        }
        if (sessions.size >= CV_LIMITS.sessions) throw new CvError(429, 'SESSION_LIMIT', 'Four sessions already active; delete one or wait for expiry.');
        const session: Session = {
          id: randomBytes(16).toString('hex'), createdAt: now(), updatedAt: now(), receivedAt: null,
          lastSequence: null, receivedFrames: 0, frames: [], bytes: 0, warnings: [], receipts: new Map(), uploading: false, gameRefId: null
        };
        sessions.set(session.id, session); cvJson(res, 201, metadata(session)); return;
      }
      const route = /^\/api\/cv\/sessions\/([a-f0-9]{32})(?:\/(latest|frames))?$/.exec(url.pathname);
      const session = route ? sessions.get(route[1]!) : undefined;
      if (!route || !session) throw new CvError(404, 'NOT_FOUND', 'Unknown or expired session/endpoint. Create a new session if needed.');
      const action = route[2];
      if (req.method === 'DELETE' && !action) {
        sessions.delete(session.id); cvJson(res, 200, { sessionId: session.id, deleted: true }); return;
      }
      if (req.method === 'GET' && action === 'latest') {
        cvJson(res, 200, { ...metadata(session), frame: session.frames.at(-1)?.frame ?? null }); return;
      }
      if (req.method === 'GET' && action === 'frames') {
        const text = url.searchParams.get('limit') ?? '30';
        const limit = Number(text);
        if (!/^\d+$/.test(text) || !Number.isInteger(limit) || limit < 1 || limit > CV_LIMITS.retainedFrames) {
          throw new CvError(400, 'INVALID_LIMIT', 'limit must be an integer from 1 to 300.');
        }
        cvJson(res, 200, { ...metadata(session), frames: session.frames.slice(-limit).map((item) => item.frame) }); return;
      }
      if (req.method === 'POST' && action === 'frames') {
        if (session.uploading) throw new CvError(409, 'UPLOAD_IN_PROGRESS', 'Retry after the current upload completes.');
        session.uploading = true;
        try {
          const body = validateCvBatch(await readJson(req));
          const hash = createHash('sha256').update(JSON.stringify(body)).digest('hex');
          const receipt = session.receipts.get(body.sequence);
          if (receipt) {
            if (receipt.hash !== hash) throw new CvError(409, 'SEQUENCE_CONFLICT', 'Sequence already accepted with different content.');
            cvJson(res, 200, { ...receipt.response, duplicate: true }); return;
          }
          if (body.sequence !== (session.lastSequence ?? -1) + 1) throw new CvError(409, 'SEQUENCE_CONFLICT', 'Expected the next batch sequence; old receipts may have expired.');
          const previous = session.frames.at(-1)?.frame;
          if (previous && (body.frames[0]!.videoTimeMs <= previous.videoTimeMs || body.frames[0]!.frameNum <= previous.frameNum)) {
            throw new CvError(409, 'FRAME_OVERLAP', 'Batch overlaps previously accepted timestamps or frame numbers.');
          }
          let gameRefId = session.gameRefId;
          for (const frame of body.frames) {
            const id = frameMatchId(frame);
            if (id !== null && gameRefId !== null && id !== gameRefId) throw new CvError(409, 'MATCH_CHANGED', 'Match ID cannot change within a session.');
            gameRefId ??= id;
          }
          const warnings = warningsFor(body.frames);
          const retained = body.frames.map((frame) => ({ frame, bytes: Buffer.byteLength(JSON.stringify(frame)) }));
          // All parsing/validation precedes this synchronous commit; no await may split it.
          expire();
          if (!sessions.has(session.id)) throw new CvError(404, 'NOT_FOUND', 'Session expired or was deleted during upload.');
          session.frames.push(...retained); session.bytes += retained.reduce((sum, item) => sum + item.bytes, 0);
          while (session.frames.length > CV_LIMITS.retainedFrames || session.bytes > CV_LIMITS.retainedBytes) session.bytes -= session.frames.shift()!.bytes;
          session.updatedAt = now(); session.receivedAt = new Date(now()).toISOString();
          session.receivedFrames += body.frames.length; session.lastSequence = body.sequence; session.warnings = warnings;
          session.gameRefId = gameRefId;
          const response: CvReceipt = {
            sessionId: session.id, sequence: body.sequence, acceptedFrames: body.frames.length,
            receivedFrames: session.receivedFrames, receivedAt: session.receivedAt, warnings, duplicate: false
          };
          session.receipts.set(body.sequence, { hash, response });
          if (session.receipts.size > CV_LIMITS.receipts) session.receipts.delete(session.receipts.keys().next().value!);
          cvJson(res, 200, response);
        } finally { session.uploading = false; }
        return;
      }
      throw new CvError(405, 'METHOD_NOT_ALLOWED', 'Method not supported on this endpoint.');
    } catch (error) {
      if (!res.headersSent && !res.destroyed) {
        const failure = error instanceof CvError ? error : new CvError(500, 'INTERNAL_ERROR', 'Request processing failed.');
        if (failure.status === 429) res.setHeader('Retry-After', '60');
        if (failure.status === 401) res.setHeader('WWW-Authenticate', 'Bearer');
        // Do not keep an unread/failed upload connection open after rejection.
        if (!req.complete) {
          res.setHeader('Connection', 'close');
          res.once('finish', () => req.destroy());
        }
        cvJson(res, failure.status, { error: { code: failure.code, message: failure.message } });
      }
    }
  };
  return { handle, close() { clearInterval(timer); sessions.clear(); } };
}

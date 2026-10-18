import { CV_LIMITS } from '../../api/src/cv-types';

const ref = (name: string) => ({ $ref: `#/components/schemas/${name}` });
const sessionId = '0123456789abcdef0123456789abcdef';
const frameExample = { frameNum: 0, videoTimeMs: 0, period: 1, periodGameClockTime: 0,
  gameRefId: 'cv-demo', homePlayers: [{ jerseyNum: '10', trackId: 42, x: -12.3, y: 8.5,
    confidence: 'HIGH', visibility: 'VISIBLE' }], awayPlayers: [], balls: [] };
const sessionExample = { sessionId, createdAt: '2026-10-18T12:00:00.000Z',
  expiresAt: '2026-10-18T14:00:00.000Z', idleExpiresAt: '2026-10-18T12:15:00.000Z',
  receivedAt: null, lastSequence: null, receivedFrames: 0, retainedFrames: 0, warnings: [] };
const activeExample = { ...sessionExample, receivedAt: '2026-10-18T12:00:01.000Z',
  idleExpiresAt: '2026-10-18T12:15:01.000Z', lastSequence: 0, receivedFrames: 1, retainedFrames: 1 };
const receiptExample = { sessionId, sequence: 0, acceptedFrames: 1, receivedFrames: 1,
  receivedAt: '2026-10-18T12:00:01.000Z', warnings: [], duplicate: false };
const content = (schema: object, example: unknown) => ({ 'application/json': { schema, example } });
const ok = (name: string, description: string, example: unknown) => ({ description, content: content(ref(name), example) });
const errors: Record<number, [string, string]> = {
  400: ['INVALID_JSON', 'Malformed UTF-8 JSON, invalid limit or unexpected session-creation body.'],
  401: ['UNAUTHORIZED', 'A valid Bearer API key is required.'],
  403: ['ORIGIN_DENIED', 'This browser origin or method is not enabled. Only explicitly configured frontend GET reads are allowed cross-origin.'],
  404: ['NOT_FOUND', 'Unknown or expired session/endpoint. Create a new session if needed.'],
  408: ['REQUEST_TIMEOUT', 'Upload body timed out after 10 seconds.'],
  409: ['SEQUENCE_CONFLICT', 'Sequence already accepted with different content.'],
  413: ['BODY_TOO_LARGE', 'JSON body exceeds 1.5 MiB.'],
  415: ['UNSUPPORTED_MEDIA_TYPE', 'Send uncompressed application/json, not JSONL or multipart.'],
  422: ['INVALID_FRAME', 'frames[0].homePlayers[0]: jerseyNum is required.'],
  429: ['RATE_LIMITED', 'Request, failed-auth or four-active-session limit reached; retry later.'],
  500: ['INTERNAL_ERROR', 'Request processing failed.']
};
const errorResponses = (...statuses: number[]) => Object.fromEntries(statuses.map((status) => {
  const [code, message] = errors[status]!;
  const headers = status === 429 ? { 'Retry-After': { schema: { type: 'string' }, example: '60', description: 'Seconds before retrying.' } }
    : status === 401 ? { 'WWW-Authenticate': { schema: { type: 'string' }, example: 'Bearer' } } : {};
  return [String(status), { description: message, headers, content: content(ref('Error'), { error: { code, message } }) }];
}));
const commonErrors = () => errorResponses(401, 403, 404, 429, 500);
const sessionParameter = { name: 'sessionId', in: 'path', required: true,
  description: 'The 32-character lowercase hex ID returned by session creation. Share this ID between producer and viewer.',
  schema: { type: 'string', pattern: '^[a-f0-9]{32}$' }, example: sessionId };
const dateTime = { type: 'string', format: 'date-time' };
const nonnegative = { type: 'integer', minimum: 0, maximum: Number.MAX_SAFE_INTEGER };
const quality = { visibility: { type: 'string', enum: ['VISIBLE', 'ESTIMATED'] },
  confidence: { type: 'string', enum: ['HIGH', 'MEDIUM', 'LOW'] } };
const warnings = { type: 'array', items: { type: 'string', enum: ['MISSING_MATCH_CLOCK', 'MISSING_PLAYER_QUALITY'] },
  description: 'Compatibility warnings only; observations are not fabricated or repaired.' };
const playerArray = { type: 'array', maxItems: CV_LIMITS.teamPlayers, items: ref('Player'),
  description: 'Omit unavailable or unknown-jersey players. Jerseys and supplied track IDs must be unique within this team array.' };

export const openApi = {
  openapi: '3.1.0',
  info: { title: 'Tracking Studio — CV API', version: '1.1.0',
    description: 'Backend-only prototype for one trusted team. Click Authorize and paste the API key value WITHOUT the Bearer prefix. The owner shares the key privately; session creation returns an ID, not a key.\n\nWorkflow: create session (no body), upload batches starting at sequence 0, read latest/recent frames, then delete the session. Try it out makes REAL requests, including creation and deletion. Your key is not saved across reloads.\n\nBatch size is flexible: 1–300 frames and at most 1.5 MiB of UTF-8 JSON per request. Send all available players in each frame. No gzip, multipart or JSONL-file uploads on these endpoints.\n\nSessions retain only the latest 300 frames AND 1.5 MiB of serialized frame data (plus metadata/32 retry receipts), not a full match. Idle expiry: 15 minutes without a newly accepted upload; absolute expiry: 2 hours. Reads/retries do not extend expiry. Four active sessions maximum. Global limits: 1,200 non-documentation/non-health requests per minute; after 30 failed authentications per minute, further failures return 429. Cloudflare quotas also apply. All key holders can access/delete known sessions. Use consented diagnostic data only.' },
  servers: [{ url: '/', description: 'This API instance (same origin)' }],
  security: [{ bearerAuth: [] }],
  tags: [{ name: 'Status', description: 'Public health/configuration status.' },
    { name: 'Sessions', description: 'Create and delete shared diagnostic sessions.' },
    { name: 'Frames', description: 'Upload original tracking observations and inspect the live buffer.' }],
  paths: {
    '/health': { get: { tags: ['Status'], operationId: 'getHealth', summary: 'Check backend health', security: [],
      description: 'Public configuration/readiness check. Does not read the session database or return tracking data.',
      responses: { '200': ok('Health', 'Backend configuration is valid.', { status: 'ok', service: 'cv-diagnostic-api', storage: 'durable-sqlite' }), ...errorResponses(500) } } },
    '/api/cv/sessions': { post: { tags: ['Sessions'], operationId: 'createSession', summary: 'Create a tracking session',
      description: 'Send NO body. Returns a new sessionId; does not return the API key. Creation is not idempotent: retrying can create another session. Share the returned ID with the CV producer and viewer.',
      responses: { '201': ok('Session', 'New empty session.', sessionExample), ...errorResponses(400, 401, 403, 408, 413, 429, 500) } } },
    '/api/cv/sessions/{sessionId}': { delete: { tags: ['Sessions'], operationId: 'deleteSession', summary: 'Delete a session and its active buffer',
      description: 'Destructive: deletes the session, retained frames and retry receipts. Further reads return 404. Platform-managed recovery copies may follow Cloudflare retention policies.',
      parameters: [sessionParameter], responses: { '200': ok('Deletion', 'Session deleted.', { sessionId, deleted: true }), ...commonErrors() } } },
    '/api/cv/sessions/{sessionId}/latest': { get: { tags: ['Frames'], operationId: 'getLatestFrame', summary: 'Read the latest accepted frame',
      description: 'Returns original provider fields under frame; null before any upload. GET does not extend expiry. This is HTTP polling, not SSE or WebSocket streaming.',
      parameters: [sessionParameter], responses: { '200': ok('Latest', 'Latest frame, counters and expiry information.', { ...activeExample, frame: frameExample }), ...commonErrors() } } },
    '/api/cv/sessions/{sessionId}/frames': {
      post: { tags: ['Frames'], operationId: 'uploadFrames', summary: 'Upload a flexible batch of tracking frames',
        description: 'sequence is the BATCH number, not a frame number. Start at 0; increment for each newly accepted batch. Keep frameNum and videoTimeMs strictly increasing within/across new batches. After an uncertain response, retry the exact same sequence/content. The last 32 receipts support duplicate=true without double-counting. Parsed JSON property order is part of the retry hash; whitespace is not.\n\nAll included players require jerseyNum; trackId is optional metadata, never a replacement. Coordinates must be pitch metres, not pixels. Extra provider fields are preserved; all numbers must be finite and JSON nesting cannot exceed 64 levels. Match ID cannot change within one session.\n\n409 codes: SEQUENCE_CONFLICT (changed/skipped/old sequence), FRAME_OVERLAP, MATCH_CHANGED, UPLOAD_IN_PROGRESS. For an in-flight upload, retry the same batch later. Validation is atomic: no partial batch commits.',
        parameters: [sessionParameter], requestBody: { required: true, description: 'Uncompressed JSON envelope; 1–300 frames and at most 1.5 MiB total.',
          content: content(ref('Batch'), { sequence: 0, frames: [frameExample] }) },
        responses: { '200': ok('Receipt', 'Accepted new batch, or replayed receipt with duplicate=true.', receiptExample), ...commonErrors(), ...errorResponses(400, 408, 409, 413, 415, 422) } },
      get: { tags: ['Frames'], operationId: 'getRecentFrames', summary: 'Read recent retained frames',
        description: 'Returns a bounded tail, oldest first. Not full-match history. Fewer frames may be returned due to the byte cap. GET does not extend expiry.',
        parameters: [sessionParameter, { name: 'limit', in: 'query', required: false, description: 'Maximum frames to return.',
          schema: { type: 'integer', minimum: 1, maximum: CV_LIMITS.retainedFrames, default: 30 }, example: 30 }],
        responses: { '200': ok('Recent', 'Recent original frames and session metadata.', { ...activeExample, frames: [frameExample] }), ...commonErrors(), ...errorResponses(400) } }
    }
  },
  components: {
    securitySchemes: { bearerAuth: { type: 'http', scheme: 'bearer',
      description: 'Paste only your private CV_API_KEY value. Swagger adds Bearer automatically. This is a shared opaque key, not a JWT. Never publish it in examples or screenshots.' } },
    schemas: {
      Player: { type: 'object', required: ['jerseyNum', 'x', 'y'], additionalProperties: true,
        description: 'Identity is team + jersey number. No guessed identities, null jerseys, or trackId substitutions.',
        properties: { jerseyNum: { anyOf: [{ type: 'integer', minimum: 1, maximum: 255 },
          { type: 'string', pattern: '^(?:0{0,2}[1-9]|0?[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$' }], description: 'Required recognized jersey, 1–255. Prefer a decimal string such as "10".', examples: ['10'] },
        trackId: { anyOf: [nonnegative, { type: 'string', minLength: 1, maxLength: 128, pattern: '^\\S(?:[\\s\\S]*\\S)?$' }], description: 'Optional CV tracker metadata. Strings must have no leading/trailing whitespace.' },
        x: { type: 'number', description: 'Pitch x coordinate in metres, not image pixels.' },
        y: { type: 'number', description: 'Pitch y coordinate in metres, not image pixels.' }, ...quality }, examples: [frameExample.homePlayers[0]] },
      Ball: { type: 'object', required: ['x', 'y'], additionalProperties: true,
        properties: { x: { type: 'number' }, y: { type: 'number' }, z: { type: 'number', minimum: 0, description: 'Optional height in metres.' },
          visibility: { type: 'string', enum: ['VISIBLE', 'ESTIMATED', 'NOT_VISIBLE'] } }, description: 'Finite pitch-metre coordinates. Omit the ball if coordinates are unavailable.' },
      Frame: { type: 'object', required: ['frameNum', 'videoTimeMs', 'homePlayers', 'awayPlayers'], additionalProperties: true,
        properties: { frameNum: { type: 'integer', minimum: 0, maximum: 2147483647, description: 'Strictly increasing source frame number.' },
          videoTimeMs: { type: 'number', minimum: 0, maximum: Number.MAX_SAFE_INTEGER, description: 'Strictly increasing source video time in milliseconds; not upload/wall-clock time.' },
          period: { type: 'integer', enum: [0, 1, 2] }, periodGameClockTime: { type: 'number', minimum: 0, description: 'Match-clock seconds for the period, not milliseconds.' },
          gameRefId: { anyOf: [{ type: 'string', minLength: 1, maxLength: 128 }, nonnegative, { type: 'null' }], description: 'Optional match ID, stable within a session.' },
          homePlayers: playerArray, awayPlayers: playerArray, homePlayersSmoothed: playerArray, awayPlayersSmoothed: playerArray,
          balls: { type: 'array', maxItems: 1, items: ref('Ball') },
          ballsSmoothed: { anyOf: [ref('Ball'), { type: 'null' }], description: 'An object or null, NOT an array.' } }, examples: [frameExample] },
      Batch: { type: 'object', required: ['sequence', 'frames'], additionalProperties: true,
        properties: { sequence: { ...nonnegative, description: 'Batch sequence starting at 0. Reuse for identical retries.' },
          frames: { type: 'array', minItems: 1, maxItems: CV_LIMITS.batchFrames, items: ref('Frame') } },
        examples: [{ sequence: 0, frames: [frameExample] }] },
      Session: { type: 'object', required: ['sessionId', 'createdAt', 'expiresAt', 'idleExpiresAt', 'receivedAt', 'lastSequence', 'receivedFrames', 'retainedFrames', 'warnings'],
        properties: { sessionId: { type: 'string', pattern: '^[a-f0-9]{32}$' }, createdAt: dateTime, expiresAt: dateTime, idleExpiresAt: dateTime,
          receivedAt: { anyOf: [dateTime, { type: 'null' }] }, lastSequence: { anyOf: [nonnegative, { type: 'null' }] },
          receivedFrames: { ...nonnegative, description: 'Total accepted frames, including those evicted from the buffer; retries do not increase it.' },
          retainedFrames: { type: 'integer', minimum: 0, maximum: CV_LIMITS.retainedFrames }, warnings }, examples: [sessionExample] },
      Latest: { allOf: [ref('Session'), { type: 'object', required: ['frame'], properties: { frame: { anyOf: [ref('Frame'), { type: 'null' }] } } }] },
      Recent: { allOf: [ref('Session'), { type: 'object', required: ['frames'], properties: { frames: { type: 'array', maxItems: CV_LIMITS.retainedFrames, items: ref('Frame') } } }] },
      Receipt: { type: 'object', required: ['sessionId', 'sequence', 'acceptedFrames', 'receivedFrames', 'receivedAt', 'warnings', 'duplicate'],
        properties: { sessionId: { type: 'string', pattern: '^[a-f0-9]{32}$' }, sequence: nonnegative,
          acceptedFrames: { type: 'integer', minimum: 1, maximum: CV_LIMITS.batchFrames }, receivedFrames: nonnegative,
          receivedAt: dateTime, warnings, duplicate: { type: 'boolean', description: 'true means the original receipt was replayed; counts were not incremented.' } }, examples: [receiptExample] },
      Deletion: { type: 'object', required: ['sessionId', 'deleted'], properties: { sessionId: { type: 'string' }, deleted: { type: 'boolean', const: true } } },
      Health: { type: 'object', required: ['status', 'service', 'storage'], properties: {
        status: { type: 'string', const: 'ok' }, service: { type: 'string', const: 'cv-diagnostic-api' }, storage: { type: 'string', const: 'durable-sqlite' } } },
      Error: { type: 'object', required: ['error'], properties: { error: { type: 'object', required: ['code', 'message'],
        properties: { code: { type: 'string', description: 'Machine-readable code. Also possible: INVALID_BATCH, INVALID_JSON_VALUE, UNSUPPORTED_ENCODING, AUTH_RATE_LIMITED, SESSION_LIMIT, FRAME_OVERLAP, MATCH_CHANGED, UPLOAD_IN_PROGRESS, UNEXPECTED_BODY, INVALID_LIMIT, METHOD_NOT_ALLOWED.' }, message: { type: 'string' } } } } }
    }
  }
};

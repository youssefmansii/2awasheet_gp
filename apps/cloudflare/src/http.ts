import { CV_LIMITS, CvError } from '../../api/src/cv-types';

export function json(status: number, body: unknown): Response {
  const headers = new Headers({ 'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
    'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'",
    'Strict-Transport-Security': 'max-age=31536000', 'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()' });
  if (status === 401) headers.set('WWW-Authenticate', 'Bearer');
  if (status === 429) headers.set('Retry-After', '60');
  return new Response(JSON.stringify(body), { status, headers });
}

export function failure(error: unknown): Response {
  const e = error instanceof CvError ? error : new CvError(500, 'INTERNAL_ERROR', 'Request processing failed.');
  return json(e.status, { error: { code: e.code, message: e.message } });
}

export async function readJson(request: Request): Promise<unknown> {
  if (request.headers.get('Content-Type')?.split(';')[0]?.trim().toLowerCase() !== 'application/json') {
    throw new CvError(415, 'UNSUPPORTED_MEDIA_TYPE', 'Send application/json.');
  }
  const encoding = request.headers.get('Content-Encoding');
  if (encoding && encoding !== 'identity') throw new CvError(415, 'UNSUPPORTED_ENCODING', 'Send uncompressed JSON to the CV endpoint.');
  const buffer = await readBytes(request);
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true, ignoreBOM: false }).decode(buffer)); }
  catch { throw new CvError(400, 'INVALID_JSON', 'Request body must contain valid UTF-8 JSON.'); }
}

export async function readBytes(request: Request): Promise<Uint8Array> {
  if (Number(request.headers.get('Content-Length') ?? 0) > CV_LIMITS.bodyBytes) throw new CvError(413, 'BODY_TOO_LARGE', 'JSON body exceeds 1 MiB.');
  const reader = request.body?.getReader();
  if (!reader) return new Uint8Array();
  const chunks: Uint8Array[] = []; let bytes = 0; let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    const read = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        bytes += value.byteLength;
        if (bytes > CV_LIMITS.bodyBytes) throw new CvError(413, 'BODY_TOO_LARGE', 'JSON body exceeds 1 MiB.');
        chunks.push(value);
      }
    };
    await Promise.race([read(), new Promise<never>((_, reject) => {
      timeout = setTimeout(() => reject(new CvError(408, 'REQUEST_TIMEOUT', 'Upload body timed out.')), CV_LIMITS.bodyReadMs);
    })]);
    const buffer = new Uint8Array(bytes); let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
    return buffer;
  } finally {
    clearTimeout(timeout);
    void reader.cancel().catch(() => { /* The client may already have disconnected. */ });
  }
}

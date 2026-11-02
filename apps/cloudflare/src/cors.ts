import { CvError } from '../../api/src/cv-types';
import { failure } from './http';
const readPath = /^\/api\/cv\/sessions\/[a-f0-9]{32}\/(frames|latest)$/;

export function allowedReadOrigin(request: Request, configured = ''): string | null {
  if (!['GET', 'OPTIONS'].includes(request.method) || !readPath.test(new URL(request.url).pathname)) return null;
  const origin = request.headers.get('Origin');
  // Exact origin match; no regex, prefix matching, reflection or wildcard allowlist.
  return origin && configured.split(',').map((s) => s.trim()).filter((s) => s !== '*' && s !== 'null').includes(origin) ? origin : null;
}
export function withReadCors(response: Response, origin: string): Response {
  const headers = new Headers(response.headers);
  headers.set('Access-Control-Allow-Origin', origin);
  headers.set('Access-Control-Expose-Headers', 'Retry-After');
  headers.append('Vary', 'Origin');
  return new Response(response.body, { status: response.status, headers });
}
export function preflight(request: Request, configured = ''): Response | null {
  if (request.method !== 'OPTIONS') return null;
  const origin = allowedReadOrigin(request, configured);
  const headers = (request.headers.get('Access-Control-Request-Headers') ?? '').toLowerCase().split(',').map((s) => s.trim()).filter(Boolean);
  if (!origin || request.headers.get('Access-Control-Request-Method') !== 'GET'
    || headers.some((h) => !['authorization', 'content-type'].includes(h))) {
    return failure(new CvError(403, 'ORIGIN_DENIED', 'Only configured frontend GET requests are enabled.'));
  }
  return new Response(null, { status: 204, headers: { 'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET', 'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    'Access-Control-Max-Age': '600', 'Cache-Control': 'no-store', Vary: 'Origin, Access-Control-Request-Method, Access-Control-Request-Headers' } });
}

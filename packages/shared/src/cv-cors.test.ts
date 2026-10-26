import { describe, expect, it } from 'vitest';
import { allowedReadOrigin, preflight, withReadCors } from '../../../apps/cloudflare/src/cors';
const path = `https://api.example/api/cv/sessions/${'a'.repeat(32)}/frames?limit=90`;
const allowed = 'http://localhost:5173,http://127.0.0.1:5173';
describe('read-only frontend CORS', () => {
  it('allows only exact configured origins for session read endpoints', () => {
    expect(allowedReadOrigin(new Request(path, { headers: { Origin: 'http://localhost:5173' } }), allowed)).toBe('http://localhost:5173');
    for (const origin of ['https://evil.example', 'null', 'http://localhost:5174', 'http://localhost:5173.evil.example']) {
      expect(allowedReadOrigin(new Request(path, { headers: { Origin: origin } }), allowed)).toBeNull();
    }
    expect(allowedReadOrigin(new Request(path, { method: 'POST', headers: { Origin: 'http://localhost:5173' } }), allowed)).toBeNull();
    expect(allowedReadOrigin(new Request('https://api.example/api/cv/sessions', { headers: { Origin: 'http://localhost:5173' } }), allowed)).toBeNull();
  });
  it('allows Authorization preflight for GET only, never wildcard credentials or writes', () => {
    const request = (method: string) => new Request(path, { method: 'OPTIONS', headers: {
      Origin: 'http://localhost:5173', 'Access-Control-Request-Method': method, 'Access-Control-Request-Headers': 'authorization'
    } });
    expect(preflight(request('GET'), allowed)!.status).toBe(204);
    expect(preflight(request('POST'), allowed)!.status).toBe(403);
    expect(withReadCors(new Response('{}', { status: 401 }), 'http://localhost:5173').headers.get('Access-Control-Allow-Origin')).toBe('http://localhost:5173');
    expect(withReadCors(new Response('{}'), 'http://localhost:5173').headers.has('Access-Control-Allow-Credentials')).toBe(false);
  });
  it('allows the read-only batch archive endpoints, and nothing else under batches', () => {
    const origin = { Origin: 'http://localhost:5173' };
    const base = `https://api.example/api/cv/sessions/${'a'.repeat(32)}`;
    for (const tail of ['/batches', '/batches?after=3', '/batches/next?after=-1', '/batches/12']) {
      expect(allowedReadOrigin(new Request(base + tail, { headers: origin }), allowed)).toBe('http://localhost:5173');
    }
    for (const tail of ['/batches/x', '/batches/next/1', '/batches/']) {
      expect(allowedReadOrigin(new Request(base + tail, { headers: origin }), allowed)).toBeNull();
    }
  });
});

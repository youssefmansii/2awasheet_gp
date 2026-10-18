/// <reference lib="webworker" />
import { CvReceiver, cvReadUrl, type CvViewUpdate } from '../live/cv-receiver';

export type CvWorkerMessage = { type: 'update'; update: CvViewUpdate; state: string }
  | { type: 'status'; state: string; message: string } | { type: 'error'; message: string };
const send = (message: CvWorkerMessage) => self.postMessage(message);
let started = false;
const friendly: Record<number, string> = {
  401: 'API key rejected. Paste the key value without the Bearer prefix.',
  403: 'This frontend origin is not allowed by the API. Ask the owner to configure its exact origin.',
  404: 'Session not found or expired. Get a current session ID from the CV team.',
  400: 'The API rejected the session request. Check the session ID.'
};
async function readSnapshot(response: Response): Promise<unknown> {
  const reader = response.body?.getReader(); if (!reader) throw new Error('Empty API response.');
  let text = ''; let bytes = 0; const decoder = new TextDecoder('utf-8', { fatal: true });
  try {
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      bytes += value.byteLength;
      if (bytes > 2 * 1024 * 1024) throw new Error('API response exceeds the safe live-buffer limit.');
      text += decoder.decode(value, { stream: true });
    }
    return JSON.parse(text + decoder.decode());
  } finally { void reader.cancel().catch(() => undefined); }
}
self.onmessage = (event: MessageEvent<{ sessionId: string; key: string; intervalMs: number }>) => {
  if (started) return; started = true;
  void poll(event.data).catch(() => send({ type: 'error', message: 'Live receiver failed. Reconnect to try again.' }));
};
async function poll({ sessionId, key, intervalMs }: { sessionId: string; key: string; intervalMs: number }) {
  const url = cvReadUrl(sessionId);
  if (!/^[\x21-\x7e]{32,256}$/.test(key) || intervalMs < 250 || intervalMs > 5000) {
    send({ type: 'error', message: 'Enter a valid API key (without Bearer) and polling interval.' }); return;
  }
  const receiver = new CvReceiver(sessionId); let failures = 0;
  while (true) {
    let delay = intervalMs;
    try {
      const response = await fetch(url, { method: 'GET', headers: { Authorization: `Bearer ${key}` },
        cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(10000) });
      if (friendly[response.status]) { send({ type: 'error', message: friendly[response.status]! }); return; }
      if (!response.ok) {
        delay = response.status === 429 ? Math.max(1000, Math.min(60000, Number(response.headers.get('Retry-After')) * 1000 || 60000)) : Math.min(30000, 1000 * 2 ** Math.min(++failures, 5));
        send({ type: 'status', state: 'Reconnecting', message: response.status === 429 ? 'API rate limit reached; waiting before retrying.' : 'API unavailable; retrying with backoff.' });
      } else {
        let update: CvViewUpdate;
        try { update = receiver.consume(await readSnapshot(response)); }
        catch { send({ type: 'error', message: 'Invalid API tracking response. Check the CV output and session, then reconnect.' }); return; }
        failures = 0;
        const stale = update.lastReceivedAt === null || Date.now() - Date.parse(update.lastReceivedAt) > 5000;
        send({ type: 'update', update, state: stale ? 'Waiting for new frames' : 'Connected' });
      }
    } catch {
      delay = Math.min(30000, 1000 * 2 ** Math.min(++failures, 5));
      send({ type: 'status', state: 'Reconnecting', message: 'Network request failed or timed out. Retrying; check connectivity and frontend CORS configuration.' });
    }
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
}

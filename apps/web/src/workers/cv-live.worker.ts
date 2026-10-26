/// <reference lib="webworker" />
import { CvReceiver, cvBatchUrl, cvReadUrl, type CvViewUpdate } from '../live/cv-receiver';

export type CvWorkerMessage = { type: 'update'; update: CvViewUpdate; state: string }
  | { type: 'status'; state: string; message: string } | { type: 'error'; message: string };
const send = (message: CvWorkerMessage) => self.postMessage(message);
let started = false;
let paused = false;
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
self.onmessage = (event: MessageEvent<{ sessionId: string; key: string; intervalMs: number } | { type: 'pause'; paused: boolean }>) => {
  if ('type' in event.data && event.data.type === 'pause') { paused = event.data.paused; return; }
  if (started) return; started = true;
  if ('sessionId' in event.data) void poll(event.data).catch(() => send({ type: 'error', message: 'Live receiver failed. Reconnect to try again.' }));
};
async function poll({ sessionId, key, intervalMs }: { sessionId: string; key: string; intervalMs: number }) {
  cvReadUrl(sessionId);
  if (!/^[\x21-\x7e]{32,256}$/.test(key) || intervalMs < 250 || intervalMs > 60000) {
    send({ type: 'error', message: 'Enter a valid API key (without Bearer) and polling interval.' }); return;
  }
  const receiver = new CvReceiver(sessionId); let failures = 0;
  const playbackState = (update: CvViewUpdate) => paused ? 'Paused' : update.bufferedFrames ? 'Playing buffered frames' : 'Waiting for next batch';
  // While the playback queue is full, stop fetching: batches stay safely stored on the server.
  const backpressureMs = 250;
  let playbackNow = 0; let lastTick = performance.now();
  const timer = setInterval(() => {
    const now = performance.now();
    if (!paused) playbackNow += Math.min(100, now - lastTick);
    lastTick = now;
    if (paused) return;
    const update = receiver.advance(playbackNow);
    if (update.data) send({ type: 'update', update, state: playbackState(update) });
  }, 16);
  try {
  while (true) {
    let delay = intervalMs;
    if (!receiver.canAcceptBatch()) { await new Promise((resolve) => setTimeout(resolve, backpressureMs)); continue; }
    try {
      const response = await fetch(cvBatchUrl(sessionId, receiver.cursor), { method: 'GET', headers: { Authorization: `Bearer ${key}` },
        cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(10000) });
      if (friendly[response.status]) { send({ type: 'error', message: friendly[response.status]! }); return; }
      if (!response.ok) {
        delay = response.status === 429 ? Math.max(1000, Math.min(60000, Number(response.headers.get('Retry-After')) * 1000 || 60000)) : Math.min(30000, 1000 * 2 ** Math.min(++failures, 5));
        send({ type: 'status', state: 'Reconnecting', message: response.status === 429 ? 'API rate limit reached; waiting before retrying.' : 'API unavailable; retrying with backoff.' });
      } else {
        let update: CvViewUpdate; const before = receiver.cursor;
        try { update = receiver.consumeBatch(await readSnapshot(response)); }
        catch (error) { send({ type: 'error', message: error instanceof Error && error.message.startsWith('Playback buffer is full.')
          ? error.message : 'Invalid API tracking response. Check the CV output and session, then reconnect.' }); return; }
        failures = 0;
        // A batch arrived: ask for the next one right away to drain any stored backlog, one batch at a time.
        if (receiver.cursor !== before) delay = 0;
        send({ type: 'update', update, state: playbackState(update) });
      }
    } catch {
      delay = Math.min(30000, 1000 * 2 ** Math.min(++failures, 5));
      send({ type: 'status', state: 'Reconnecting', message: 'Network request failed or timed out. Retrying; check connectivity and frontend CORS configuration.' });
    }
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
  } finally { clearInterval(timer); }
}

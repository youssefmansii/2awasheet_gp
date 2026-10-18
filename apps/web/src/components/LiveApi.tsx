import { DEFAULT_WORKLOAD_CONFIG, formatMatchClock, type MatchMetadata, type PlayerWorkload, type TrackingData } from '@replay/shared';
import { useEffect, useRef, useState } from 'react';
import { CV_API_BASE, cvReadUrl, type CvViewUpdate } from '../live/cv-receiver';
import type { CvWorkerMessage } from '../workers/cv-live.worker';
import { useUiStore } from '../store';
import { AnalyticsPanel } from './AnalyticsPanel';
import { PitchCanvas, type PlaybackSnapshot } from './PitchCanvas';

export function LiveApi({ metadata }: { metadata: MatchMetadata | null }) {
  const [sessionId, setSessionId] = useState(''); const [key, setKey] = useState('');
  const [intervalMs, setIntervalMs] = useState(1000); const [active, setActive] = useState(false);
  const [state, setState] = useState('Not connected'); const [message, setMessage] = useState('');
  const [error, setError] = useState(''); const [update, setUpdate] = useState<CvViewUpdate>();
  const [data, setData] = useState<TrackingData>();
  const [paused, setPaused] = useState(false);
  const workloads = useRef<PlayerWorkload[]>([]);
  const worker = useRef<Worker>(); const generation = useRef(0);
  const playback = useRef<PlaybackSnapshot>({ index: 0, nextIndex: 0, alpha: 0 });
  const ui = useUiStore();
  const cleanup = () => { generation.current++; worker.current?.terminate(); worker.current = undefined; };
  useEffect(() => () => cleanup(), []);
  const connect = () => {
    cleanup(); setError('');
    const id = sessionId.trim(); const secret = key.trim();
    try { cvReadUrl(id); } catch { setError('Enter a valid 32-character session ID.'); return; }
    if (!/^[\x21-\x7e]{32,256}$/.test(secret)) { setError('Enter your API key without the Bearer prefix.'); return; }
    setData(undefined); setUpdate(undefined); setActive(true); setState('Connecting'); setMessage(''); setKey('');
    setPaused(false); workloads.current = [];
    ui.set({ selectedPlayer: null });
    const run = generation.current;
    const source = new Worker(new URL('../workers/cv-live.worker.ts', import.meta.url), { type: 'module' });
    worker.current = source;
    source.onmessage = (event: MessageEvent<CvWorkerMessage>) => {
      if (run !== generation.current) return;
      if (event.data.type === 'error') { setError(event.data.message); setState('Stopped'); setActive(false); cleanup(); return; }
      setState(event.data.state);
      if (event.data.type === 'status') setMessage(event.data.message);
      else {
        const next = event.data.update; setUpdate(next); setMessage('');
        if (next.workloads) workloads.current = next.workloads;
        if (next.data) {
          const index = next.data.frameCount - 1; playback.current = { index, nextIndex: index, alpha: 0 };
          setData({ ...next.data, workloads: workloads.current });
        }
      }
    };
    source.onerror = () => { if (run !== generation.current) return; setError('Live receiver worker failed. Reconnect to retry.'); setState('Stopped'); setActive(false); cleanup(); };
    source.postMessage({ sessionId: id, key: secret, intervalMs });
  };
  const index = data ? data.frameCount - 1 : 0;
  const pitch = metadata?.stadium?.pitches?.[0];
  const clock = data && update?.clockComplete ? formatMatchClock(data.periodGameClockTime[index] ?? 0, data.period[index] ?? 0)
    : data ? `Video ${((data.videoTimeMs[index] ?? 0) / 1000).toFixed(2)} s · match clock unavailable` : '';
  const homeName = metadata?.homeTeam?.shortName ?? 'HOME'; const awayName = metadata?.awayTeam?.shortName ?? 'AWAY';
  return <main className="mx-auto max-w-[1480px] space-y-5 px-4 py-6 sm:px-6 lg:px-8">
    <section className="space-y-4 rounded-3xl border border-line bg-panel p-5 sm:p-6" aria-labelledby="live-api-title">
      <p className="text-xs font-bold uppercase tracking-wider text-lime">Tracking-only · read-only API</p>
      <h2 id="live-api-title" className="text-2xl font-semibold">Receive live tracking</h2>
      <p className="max-w-3xl text-sm leading-6 text-muted">The CV team uploads batches to Cloudflare. This viewer plays received frames in order at their original speed, then waits for the next batch. CV processing may be delayed; this is buffered playback, not a claim of real-time capture.</p>
      <p className="break-all text-xs text-muted">API: {CV_API_BASE} · <a className="text-lime underline" href={`${CV_API_BASE}/docs`} target="_blank" rel="noreferrer">Swagger documentation</a></p>
      <form className="grid items-end gap-3 sm:grid-cols-2 lg:grid-cols-[1fr_1fr_auto]" onSubmit={(event) => { event.preventDefault(); connect(); }}>
        <label className="min-w-0 text-sm" htmlFor="cv-session">Session ID
          <input id="cv-session" value={sessionId} onChange={(e) => setSessionId(e.target.value)} disabled={active} autoComplete="off" spellCheck={false} className="mt-2 min-h-11 w-full rounded-xl border border-line bg-black/20 px-3 font-mono text-sm disabled:opacity-60" />
        </label>
        <label className="min-w-0 text-sm" htmlFor="cv-key">API key (without Bearer)
          <input id="cv-key" type="password" value={key} onChange={(e) => setKey(e.target.value)} disabled={active} autoComplete="off" spellCheck={false} className="mt-2 min-h-11 w-full rounded-xl border border-line bg-black/20 px-3 text-sm disabled:opacity-60" />
        </label>
        <label className="text-sm" htmlFor="cv-interval">GET interval
          <select id="cv-interval" value={intervalMs} disabled={active} onChange={(e) => setIntervalMs(Number(e.target.value))} className="mt-2 min-h-11 w-full rounded-xl border border-line bg-ink px-3">
            {[250, 500, 1000, 2000, 5000, 60000].map((ms) => <option value={ms} key={ms}>{ms === 60000 ? '1 min' : ms < 1000 ? `${ms} ms` : `${ms / 1000} s`}</option>)}
          </select>
        </label>
        <div className="flex flex-wrap gap-3 sm:col-span-2 lg:col-span-3">
          <button type="submit" disabled={active} className="min-h-11 rounded-xl bg-lime px-5 font-semibold text-ink disabled:opacity-40">Connect live API</button>
          <button type="button" disabled={!active} onClick={() => { cleanup(); setKey(''); setActive(false); setState('Stopped'); setMessage('Reception stopped. The producer session was not deleted.'); }} className="min-h-11 rounded-xl border border-line px-5 disabled:opacity-40">Stop receiving</button>
          <button type="button" disabled={!active || !data} onClick={() => {
            const value = !paused; setPaused(value); worker.current?.postMessage({ type: 'pause', paused: value });
            setState(value ? 'Paused' : 'Resuming playback');
          }} className="min-h-11 rounded-xl border border-line px-5 disabled:opacity-40">{paused ? 'Resume playback' : 'Pause playback'}</button>
        </div>
      </form>
      {intervalMs === 60000 ? <p className="text-xs leading-5 text-amber-200">Checks once per minute after the first immediate request. Frequent uploads may overwrite the server’s 300-frame / 1.5 MiB buffer before the next check, causing missed frames. Playback speed is unchanged.</p> : null}
      <p className="text-xs leading-5 text-muted">The key is used only in an Authorization header, not stored in browser storage or URLs, and cleared from the form on connection. Use Local replay to read a whole file on your computer. Default live pitch: {pitch?.length ?? 105} × {pitch?.width ?? 68} m; metadata selected in Local replay is reused here.</p>
      <div role="status" className="rounded-xl bg-black/20 p-4 text-sm leading-6">
        <p className="font-semibold">{state}</p>{message ? <p className="text-muted">{message}</p> : null}
        <p aria-live="off" className="text-muted">{update?.processedFrames ?? 0} frames observed · {update?.receivedFrames ?? 0} accepted by server · {update?.missingFrames ?? 0} earlier/missed frames unavailable</p>
        <p aria-live="off" className="text-muted">{update?.playedFrames ?? 0} frames played · {update?.bufferedFrames ?? 0} queued · approximately {(update?.bufferedSeconds ?? 0).toFixed(1)} s buffered</p>
        {update?.lastReceivedAt ? <p className="text-xs text-muted">Last producer upload: {new Date(update.lastReceivedAt).toLocaleTimeString()}</p> : null}
      </div>
      {error ? <p role="alert" className="rounded-xl border border-amber-400/40 p-3 text-sm text-amber-200">{error}</p> : null}
      {update?.missingFrames ? <p className="rounded-xl border border-amber-400/40 p-3 text-sm text-amber-200">Partial history: earlier frames were not available or were missed. Workload covers observed frames only, not the whole match. Movement is not bridged across known missing frames.</p> : null}
      {update?.warnings.includes('MISSING_PLAYER_QUALITY') ? <p className="text-sm text-amber-200">CV confidence/visibility fields are missing. Quality indicators cannot establish tracking reliability from absent quality metadata.</p> : null}
      <p className="text-xs leading-5 text-muted">Positions and statistics follow played frames, not upload time. Workload checkpoints update about once per source second and at the end of a batch. Repeated polls are deduplicated. Playback waits when empty and resumes automatically; Pause keeps receiving. The browser queues up to 3,600 frames, while the server retains at most 300 frames / 1.5 MiB, not a match archive. Connect before uploading consecutive batches so none are missed. Reconnecting resets playback and observed statistics.</p>
    </section>
    {data ? <>
      <section className="rounded-3xl border border-line bg-panel p-3" aria-label="Live API pitch">
        <div className="flex flex-wrap justify-between gap-3 p-3 text-sm"><span className="font-semibold text-lime">{active ? state : 'Stopped · last played frame'}</span><span className="font-mono" data-testid="live-playback-frame">{clock} · source frame #{data.frameNum[index]}</span></div>
        <PitchCanvas data={data} playback={playback} pitchLength={pitch?.length ?? 105} pitchWidth={pitch?.width ?? 68} positionMode={ui.positionMode} showNumbers={ui.showNumbers} showBall={ui.showBall} interpolate={false} flipSides={ui.flipSides} homeColor={ui.homeColor} homeTextColor={ui.homeTextColor} awayColor={ui.awayColor} awayTextColor={ui.awayTextColor} trailLength={0} selectedPlayer={ui.selectedPlayer} onSelectPlayer={(selectedPlayer) => ui.set({ selectedPlayer })} eventOverlay={null} />
      </section>
      {update?.clockComplete ? <AnalyticsPanel events={null} data={data} workloadConfig={DEFAULT_WORKLOAD_CONFIG} currentTimeMs={data.videoTimeMs[index] ?? 0} frameIndex={index} clockLabel={clock} homeName={homeName} awayName={awayName} observedOnly />
        : <p role="status" className="rounded-xl border border-line p-4 text-sm text-muted">Positions are available. Workload/trend display requires period 1/2 and periodGameClockTime in every received frame. Correct the CV clock fields, then reconnect; no match clock is invented.</p>}
    </> : null}
  </main>;
}

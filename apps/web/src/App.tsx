import {
  DEFAULT_WORKLOAD_CONFIG,
  findFrameByTime,
  formatMatchClock,
  parseMetadata,
  periodLabel,
  playableRanges,
  validateEventScore,
  type EventDataset,
  type MatchMetadata,
  type TrackingData,
  type WorkloadConfig
} from '@replay/shared';
import {
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  SkipBack,
  SkipForward,
  Sparkles
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { DropZone } from './components/DropZone';
import { HighlightControls } from './components/HighlightControls';
import { advanceHighlightPlayback, buildHighlights, nearestHighlightTime } from './replay/highlights';
import { LiveApi } from './components/LiveApi';
import { AnalyticsPanel } from './components/AnalyticsPanel';
import { WorkloadSettings } from './components/WorkloadSettings';
import { EventTimeline, type EventFilter } from './components/EventTimeline';
import { PitchCanvas, type EventOverlay, type PlaybackSnapshot } from './components/PitchCanvas';
import { useUiStore } from './store';
import type { EventWorkerResponse, WorkerResponse, WorkloadWorkerResponse } from './types';

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

function IconButton({ label, disabled, onClick, children }: { label: string; disabled?: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-line bg-white/[0.035] text-slate-100 transition-colors hover:border-slate-500 hover:bg-white/[0.08] active:bg-white/[0.12] disabled:cursor-not-allowed disabled:opacity-35"
    >
      {children}
    </button>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex min-h-11 cursor-pointer items-center justify-between gap-3 rounded-xl px-3 text-sm text-slate-200 transition-colors hover:bg-white/[0.035]">
      <span>{label}</span>
      <input className="h-5 w-5 accent-lime" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function nearestPlayableTime(target: number, ranges: Array<[number, number]>, includeEverything: boolean): number {
  if (includeEverything || ranges.length === 0) return target;
  for (const [start, end] of ranges) {
    if (target >= start && target <= end) return target;
    if (target < start) return start;
  }
  return ranges.at(-1)?.[1] ?? target;
}

function eventAtOrBefore(events: EventDataset['events'], timeMs: number) {
  let low = 0;
  let high = events.length - 1;
  while (low <= high) {
    const middle = (low + high) >>> 1;
    if ((events[middle]?.eventTimeMs ?? 0) <= timeMs) low = middle + 1;
    else high = middle - 1;
  }
  return high >= 0 ? events[high] ?? null : null;
}

export function App() {
  const [mode, setMode] = useState<'replay' | 'live'>('replay');
  const [playbackView, setPlaybackView] = useState<'timeline' | 'highlights'>('timeline');
  const [data, setData] = useState<TrackingData | null>(null);
  const [metadata, setMetadata] = useState<MatchMetadata | null>(null);
  const [events, setEvents] = useState<EventDataset | null>(null);
  const [trackingName, setTrackingName] = useState<string>();
  const [metadataName, setMetadataName] = useState<string>();
  const [eventName, setEventName] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [parsedFrames, setParsedFrames] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [metadataError, setMetadataError] = useState<string>();
  const [eventError, setEventError] = useState<string>();
  const [eventLoading, setEventLoading] = useState(false);
  const [eventStage, setEventStage] = useState('');
  const [eventProgress, setEventProgress] = useState(0);
  const [workloadConfig, setWorkloadConfig] = useState<WorkloadConfig>({ ...DEFAULT_WORKLOAD_CONFIG });
  const [workloadProgress, setWorkloadProgress] = useState(0);
  const [workloadFrames, setWorkloadFrames] = useState(0);
  const [workloadLoading, setWorkloadLoading] = useState(false);
  const [workloadError, setWorkloadError] = useState<string>();
  const [eventFilter, setEventFilter] = useState<EventFilter>('ALL');
  const [playing, setPlaying] = useState(false);
  const [displayIndex, setDisplayIndex] = useState(0);
  const [analyticsIndex, setAnalyticsIndex] = useState(0);
  const workerRef = useRef<Worker>();
  const eventWorkerRef = useRef<Worker>();
  const workloadWorkerRef = useRef<Worker>();
  const trackingFileRef = useRef<File>();
  const playbackRef = useRef<PlaybackSnapshot>({ index: 0, nextIndex: 0, alpha: 0 });
  const playTimeRef = useRef(0);
  const lastUiUpdateRef = useRef(0);
  const lastAnalyticsUpdateRef = useRef(0);

  const ui = useUiStore();
  const pitch = metadata?.stadium?.pitches?.[0];
  const pitchLength = pitch?.length ?? 105;
  const pitchWidth = pitch?.width ?? 68;
  const ranges = useMemo(() => playableRanges(metadata), [metadata]);
  const trackingEventMismatch = Boolean(events?.gameId && data?.gameRefId && events.gameId !== data.gameRefId);
  const highlightClips = useMemo(() => events && data && !trackingEventMismatch
    ? buildHighlights(events.events, data.videoTimeMs.subarray(0, data.frameCount), ranges) : [],
  [events, data, ranges, trackingEventMismatch]);

  const loadMetadata = useCallback(async (file: File) => {
    setMetadataName(file.name);
    setMetadataError(undefined);
    try {
      const next = parseMetadata(await file.text());
      setMetadata(next);
      useUiStore.getState().set({
        homeColor: next.homeTeamKit?.primaryColor ?? '#ff6b1a',
        homeTextColor: next.homeTeamKit?.primaryTextColor ?? '#111518',
        awayColor: next.awayTeamKit?.primaryColor ?? '#f7f8fa',
        awayTextColor: next.awayTeamKit?.primaryTextColor ?? '#0f172a'
      });
    } catch (reason) {
      setMetadata(null);
      setMetadataError(reason instanceof Error ? reason.message : 'Unable to read metadata.');
    }
  }, []);

  const loadTracking = useCallback((file: File) => {
    workerRef.current?.terminate();
    workloadWorkerRef.current?.terminate();
    trackingFileRef.current = file;
    setTrackingName(file.name);
    setLoading(true);
    setProgress(0);
    setParsedFrames(0);
    setError(undefined);
    setData(null);
    setPlaying(false);
    setPlaybackView('timeline');
    const worker = new Worker(new URL('./workers/tracking.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === 'progress') {
        setProgress(message.totalBytes > 0 ? message.bytesRead / message.totalBytes : 0);
        setParsedFrames(message.parsedFrames);
      } else if (message.type === 'complete') {
        setData(message.data);
        const firstTime = message.data.videoTimeMs[0] ?? 0;
        playTimeRef.current = firstTime;
        playbackRef.current = { index: 0, nextIndex: Math.min(1, message.data.frameCount - 1), alpha: 0 };
        setDisplayIndex(0);
        setAnalyticsIndex(0);
        setProgress(1);
        setLoading(false);
        worker.terminate();
      } else {
        setError(message.message);
        setLoading(false);
      }
    };
    worker.onerror = (event) => {
      setError(event.message || 'The parsing worker stopped unexpectedly.');
      setLoading(false);
    };
    worker.postMessage({ file, config: workloadConfig });
  }, [workloadConfig]);

  const loadEvents = useCallback((file: File) => {
    eventWorkerRef.current?.terminate();
    setPlaying(false);
    setPlaybackView('timeline');
    setEvents(null);
    setEventName(file.name);
    setEventError(undefined);
    setEventLoading(true);
    setEventProgress(0);
    const worker = new Worker(new URL('./workers/events.worker.ts', import.meta.url), { type: 'module' });
    eventWorkerRef.current = worker;
    worker.onmessage = (event: MessageEvent<EventWorkerResponse>) => {
      const message = event.data;
      if (message.type === 'progress') {
        setEventProgress(message.progress);
        setEventStage(message.stage);
      } else if (message.type === 'complete') {
        setEvents(message.data);
        setEventProgress(1);
        setEventStage('Ready');
        setEventLoading(false);
        worker.terminate();
      } else {
        setEventError(message.message);
        setEventLoading(false);
      }
    };
    worker.onerror = (event) => {
      setEventError(event.message || 'The events worker stopped unexpectedly.');
      setEventLoading(false);
    };
    worker.postMessage({ file });
  }, []);

  const recomputeWorkload = useCallback((config: WorkloadConfig) => {
    const file = trackingFileRef.current;
    if (!file) return;
    workloadWorkerRef.current?.terminate();
    setWorkloadLoading(true);
    setWorkloadProgress(0);
    setWorkloadFrames(0);
    setWorkloadError(undefined);
    const worker = new Worker(new URL('./workers/workload.worker.ts', import.meta.url), { type: 'module' });
    workloadWorkerRef.current = worker;
    worker.onmessage = (event: MessageEvent<WorkloadWorkerResponse>) => {
      const message = event.data;
      if (message.type === 'progress') {
        setWorkloadProgress(message.totalBytes > 0 ? message.bytesRead / message.totalBytes : 0);
        setWorkloadFrames(message.parsedFrames);
      } else if (message.type === 'complete') {
        setData((current) => current ? { ...current, workloads: message.workloads } : current);
        setWorkloadConfig(message.config);
        setWorkloadProgress(1);
        setWorkloadLoading(false);
        worker.terminate();
      } else {
        setWorkloadError(message.message);
        setWorkloadLoading(false);
      }
    };
    worker.onerror = (event) => {
      setWorkloadError(event.message || 'The workload worker stopped unexpectedly.');
      setWorkloadLoading(false);
    };
    worker.postMessage({ file, config });
  }, []);

  useEffect(() => () => {
    workerRef.current?.terminate();
    eventWorkerRef.current?.terminate();
    workloadWorkerRef.current?.terminate();
  }, []);

  const seekToTime = useCallback((target: number) => {
    if (!data?.frameCount) return;
    const first = data.videoTimeMs[0] ?? 0;
    const last = data.videoTimeMs[data.frameCount - 1] ?? first;
    const bounded = Math.max(first, Math.min(last, target));
    const playable = playbackView === 'highlights' && highlightClips.length
      ? nearestHighlightTime(highlightClips, bounded) : nearestPlayableTime(bounded, ranges, ui.includeEverything);
    const index = findFrameByTime(data.videoTimeMs.subarray(0, data.frameCount), playable);
    const nextIndex = Math.min(data.frameCount - 1, index + 1);
    const start = data.videoTimeMs[index] ?? playable;
    const end = data.videoTimeMs[nextIndex] ?? start;
    playTimeRef.current = playable;
    playbackRef.current = { index, nextIndex, alpha: end > start ? Math.max(0, Math.min(1, (playable - start) / (end - start))) : 0 };
    setDisplayIndex(index);
    setAnalyticsIndex(index);
  }, [data, ranges, ui.includeEverything, playbackView, highlightClips]);

  useEffect(() => {
    if (!data?.frameCount) return;
    let animation = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const elapsed = now - previous;
      previous = now;
      if (playing) {
        const lastTime = data.videoTimeMs[data.frameCount - 1] ?? 0;
        let target = playTimeRef.current + elapsed * ui.speed;
        let highlightEnded = false;
        let highlightJumped = false;
        if (playbackView === 'highlights') {
          const next = advanceHighlightPlayback(highlightClips, playTimeRef.current, elapsed * ui.speed);
          highlightJumped = Math.abs(next.timeMs - target) > 1;
          target = next.timeMs; highlightEnded = next.ended;
        } else if (!ui.includeEverything && ranges.length > 0) {
          for (let rangeIndex = 0; rangeIndex < ranges.length - 1; rangeIndex += 1) {
            const end = ranges[rangeIndex]?.[1] ?? 0;
            const nextStart = ranges[rangeIndex + 1]?.[0] ?? end;
            if (playTimeRef.current <= end && target > end) target = nextStart + (target - end);
          }
        }
        if (target >= lastTime || highlightEnded) {
          target = Math.min(target, lastTime);
          setPlaying(false);
        }
        playTimeRef.current = target;
        const times = data.videoTimeMs.subarray(0, data.frameCount);
        const index = findFrameByTime(times, target);
        const nextIndex = Math.min(data.frameCount - 1, index + 1);
        const start = times[index] ?? target;
        const end = times[nextIndex] ?? start;
        playbackRef.current = { index, nextIndex, alpha: end > start ? (target - start) / (end - start) : 0 };
        if (now - lastUiUpdateRef.current >= 100 || highlightJumped || highlightEnded) {
          lastUiUpdateRef.current = now;
          setDisplayIndex(index);
        }
        if (now - lastAnalyticsUpdateRef.current >= 1000 || highlightJumped || highlightEnded) {
          lastAnalyticsUpdateRef.current = now;
          setAnalyticsIndex(index);
        }
      }
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [data, playing, ranges, ui.includeEverything, ui.speed, playbackView, highlightClips]);

  const currentPeriod = data?.period[displayIndex] ?? 0;
  const currentClock = data?.periodGameClockTime[displayIndex] ?? 0;
  const homeName = metadata?.homeTeam?.shortName ?? metadata?.homeTeam?.name ?? 'HOME';
  const awayName = metadata?.awayTeam?.shortName ?? metadata?.awayTeam?.name ?? 'AWAY';
  const mismatch = Boolean(data?.gameRefId && metadata?.id && data.gameRefId !== String(metadata.id));
  const warningCount = (data?.malformedLines ?? 0) + (data?.blankLines ?? 0);
  const currentTimeMs = data?.videoTimeMs[displayIndex] ?? 0;
  const currentEvent = events ? eventAtOrBefore(events.events, currentTimeMs) : null;
  const eventMismatch = Boolean(events?.gameId && metadata?.id && events.gameId !== String(metadata.id));
  const scoreValidation = useMemo(() => events ? validateEventScore(events) : null, [events]);
  const verifiedLiveScore = useMemo(() => {
    if (scoreValidation?.status !== 'matched') return null;
    return scoreValidation.goalCandidates.reduce<[number, number]>((score, goal) => {
      if (goal.eventTimeMs <= currentTimeMs && goal.homeTeam !== null) score[goal.homeTeam ? 0 : 1] += 1;
      return score;
    }, [0, 0]);
  }, [currentTimeMs, scoreValidation]);
  const eventOverlay: EventOverlay | null = useMemo(() => {
    if (!events || !currentEvent || currentTimeMs - currentEvent.eventTimeMs > 1500) return null;
    const keyForPlayer = (playerId: string | null) => {
      const player = events.roster.find((item) => item.playerId === playerId);
      return player ? `${player.team}:${player.jerseyNum}` : null;
    };
    return {
      playerKey: keyForPlayer(currentEvent.passerPlayerId ?? currentEvent.playerId),
      targetKey: keyForPlayer(currentEvent.receiverPlayerId ?? currentEvent.targetPlayerId),
      type: currentEvent.possessionType ?? currentEvent.gameType ?? 'EVENT'
    };
  }, [currentEvent, currentTimeMs, events]);

  return (
    <div className="min-h-dvh bg-ink pb-[max(2rem,env(safe-area-inset-bottom))] text-slate-50">
      <header className="border-b border-line/80 bg-[#0c1013]/95">
        <div className="mx-auto flex max-w-[1480px] items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-lime">
              <Sparkles aria-hidden="true" className="h-4 w-4" />
              <span className="text-xs font-bold uppercase tracking-[0.2em]">Tracking studio</span>
            </div>
            <h1 className="mt-1 truncate text-lg font-semibold tracking-tight text-white sm:text-xl">Match Replay</h1>
          </div>
          {data && mode === 'replay' ? (
            <div className="flex items-center gap-4 rounded-2xl border border-line bg-panel px-4 py-2.5 shadow-panel sm:gap-7 sm:px-6" aria-live="polite">
                <div className="hidden text-right sm:block">
                  <div className="text-sm font-bold tracking-wide">{homeName} <span className="mx-2 font-normal text-muted">{verifiedLiveScore ? `${verifiedLiveScore[0]}–${verifiedLiveScore[1]}` : 'vs'}</span> {awayName}</div>
                  <div className="flex items-center justify-end gap-2 text-xs text-muted"><span>{periodLabel(currentPeriod)}</span>{currentEvent?.teamName ? <span className="rounded-full bg-white/[0.06] px-2 py-0.5">Last event: {currentEvent.teamName}</span> : null}</div>
              </div>
              <div className="font-mono text-xl font-bold tabular-nums text-white sm:text-2xl">{formatMatchClock(currentClock, currentPeriod)}</div>
            </div>
          ) : null}
        </div>
      </header>

      <nav aria-label="Tracking mode" className="mx-auto flex max-w-[1480px] gap-3 px-4 pt-5 sm:px-6 lg:px-8">
        {(['replay', 'live'] as const).map((option) => <button key={option} type="button" aria-pressed={mode === option} onClick={() => { setPlaying(false); setMode(option); }} className={`min-h-11 rounded-xl border px-4 text-sm font-semibold ${mode === option ? 'border-lime bg-lime/10 text-lime' : 'border-line text-muted'}`}>{option === 'replay' ? 'Local replay' : 'Live API'}</button>)}
      </nav>
      {mode === 'live' ? <LiveApi metadata={metadata} /> : <main className="mx-auto grid max-w-[1480px] gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:px-8 lg:py-8">
        <section className="min-w-0 space-y-4" aria-label="Replay viewer">
          {!data ? (
            <div className="overflow-hidden rounded-3xl border border-line bg-panel shadow-panel">
              <div className="border-b border-line p-6 sm:p-8">
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-lime">Client-only MVP</p>
                <h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight text-white sm:text-5xl">Turn tracking data into a match you can watch.</h2>
                <p className="mt-4 max-w-2xl text-base leading-7 text-muted">Files are streamed and parsed locally in a worker. Nothing is uploaded, and even full-match captures stay off the UI thread.</p>
              </div>
              <div className="grid gap-4 p-4 sm:grid-cols-2 sm:p-6 lg:grid-cols-3">
                <DropZone label="Tracking file" hint="JSONL, one frame per line. Large files are supported." accept=".json,.jsonl,application/json,application/x-ndjson" required fileName={trackingName} disabled={loading} onFile={loadTracking} />
                <DropZone label="Match metadata" hint="Optional team, kit, pitch, and half information." accept=".json,application/json" fileName={metadataName} disabled={loading} onFile={(file) => void loadMetadata(file)} />
                <DropZone label="Match events" hint="Optional event-array JSON for passes, shots, roster and stats." accept=".json,application/json" fileName={eventName} disabled={eventLoading} onFile={loadEvents} />
              </div>
              <p className="border-t border-line px-5 py-4 text-sm text-muted">Select local files above. Match data stays on this device and is never uploaded.</p>
              {loading ? (
                <div className="border-t border-line px-5 py-5" aria-live="polite">
                  <div className="mb-2 flex justify-between gap-4 text-sm"><span>Parsing off the main thread…</span><span className="font-mono tabular-nums text-lime">{Math.round(progress * 100)}%</span></div>
                  <div className="h-2 overflow-hidden rounded-full bg-black/40"><div className="h-full rounded-full bg-lime transition-[width] duration-200" style={{ width: `${progress * 100}%` }} /></div>
                  <p className="mt-2 text-xs text-muted">{parsedFrames.toLocaleString()} frames parsed</p>
                </div>
              ) : null}
              {eventLoading ? (
                <div className="border-t border-line px-5 py-5" aria-live="polite">
                  <div className="mb-2 flex justify-between gap-4 text-sm"><span>{eventStage}…</span><span className="font-mono tabular-nums text-lime">{Math.round(eventProgress * 100)}%</span></div>
                  <div className="h-2 overflow-hidden rounded-full bg-black/40"><div className="h-full rounded-full bg-sky-300 transition-[width] duration-200" style={{ width: `${eventProgress * 100}%` }} /></div>
                </div>
              ) : null}
            </div>
          ) : (
            <>
              <div className="rounded-3xl border border-line bg-panel p-2 shadow-panel sm:p-3">
                <PitchCanvas
                  data={data}
                  playback={playbackRef}
                  pitchLength={pitchLength}
                  pitchWidth={pitchWidth}
                  positionMode={ui.positionMode}
                  showNumbers={ui.showNumbers}
                  showBall={ui.showBall}
                  interpolate={ui.interpolate}
                  flipSides={ui.flipSides}
                  homeColor={ui.homeColor}
                  homeTextColor={ui.homeTextColor}
                  awayColor={ui.awayColor}
                  awayTextColor={ui.awayTextColor}
                  trailLength={ui.trailLength}
                  selectedPlayer={ui.selectedPlayer}
                  onSelectPlayer={(selectedPlayer) => ui.set({ selectedPlayer })}
                  eventOverlay={eventOverlay}
                />
              </div>
              <div className="rounded-3xl border border-line bg-panel p-4 shadow-panel sm:p-5">
                <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Local playback view">
                  {(['timeline', 'highlights'] as const).map((view) => <button key={view} type="button"
                    aria-pressed={playbackView === view} onClick={() => {
                      setPlaying(false); setPlaybackView(view);
                      if (view === 'highlights' && highlightClips.length) seekToTime(nearestHighlightTime(highlightClips, playTimeRef.current));
                    }} className={`min-h-11 rounded-xl border px-4 text-sm font-semibold ${playbackView === view ? 'border-lime bg-lime/10 text-lime' : 'border-line text-muted hover:text-white'}`}>
                    {view === 'timeline' ? 'Timeline' : 'Highlights'}
                  </button>)}
                </div>
                <div className="mb-3 flex items-center justify-between gap-4 text-xs text-muted">
                  <span>Frame <strong className="font-mono text-slate-200">{(displayIndex + 1).toLocaleString()}</strong> / {data.frameCount.toLocaleString()}</span>
                  <span className="font-mono tabular-nums" data-testid="local-source-frame">Source #{data.frameNum[displayIndex]}</span>
                </div>
                {playbackView === 'highlights' ? <HighlightControls clips={highlightClips} currentTimeMs={currentTimeMs}
                  playing={playing} onSeek={seekToTime} onPlayingChange={setPlaying}
                  emptyReason={eventLoading ? 'Loading match events…' : !events ? 'Load a matching Match events JSON file to generate automatic highlights.'
                    : trackingEventMismatch ? 'The events match ID does not match the tracking file. Load matching events to use Highlights.'
                      : 'No goals, shots, cards or substitutions have available tracking in this recording.'} /> : <>
                <label className="sr-only" htmlFor="timeline">Seek through match</label>
                <input id="timeline" className="block h-11 w-full cursor-pointer" type="range" min={0} max={Math.max(0, data.frameCount - 1)} value={displayIndex} onChange={(event) => { setPlaying(false); const index = Number(event.target.value); seekToTime(data.videoTimeMs[index] ?? 0); }} />
                <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
                  <IconButton label="Jump back 5 seconds" onClick={() => seekToTime(playTimeRef.current - 5000)}><SkipBack aria-hidden="true" className="h-5 w-5" /></IconButton>
                  <IconButton label="Previous frame" onClick={() => seekToTime(data.videoTimeMs[Math.max(0, playbackRef.current.index - 1)] ?? 0)}><ChevronLeft aria-hidden="true" className="h-5 w-5" /></IconButton>
                  <button type="button" aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={() => { if (displayIndex >= data.frameCount - 1) seekToTime(data.videoTimeMs[0] ?? 0); setPlaying((value) => !value); }} className="flex h-14 w-14 items-center justify-center rounded-2xl bg-lime text-[#152000] transition-colors hover:bg-[#d6ff72] active:bg-[#b8df50]">
                    {playing ? <Pause aria-hidden="true" className="h-6 w-6 fill-current" /> : <Play aria-hidden="true" className="ml-0.5 h-6 w-6 fill-current" />}
                  </button>
                  <IconButton label="Next frame" onClick={() => seekToTime(data.videoTimeMs[Math.min(data.frameCount - 1, playbackRef.current.index + 1)] ?? 0)}><ChevronRight aria-hidden="true" className="h-5 w-5" /></IconButton>
                  <IconButton label="Jump forward 5 seconds" onClick={() => seekToTime(playTimeRef.current + 5000)}><SkipForward aria-hidden="true" className="h-5 w-5" /></IconButton>
                </div>
                </>}
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
                  <div className="flex items-center gap-2">
                    <Gauge aria-hidden="true" className="h-4 w-4 text-muted" />
                    <label className="text-sm text-muted" htmlFor="speed">Speed</label>
                    <select id="speed" value={ui.speed} onChange={(event) => ui.set({ speed: Number(event.target.value) })} className="h-11 rounded-xl border border-line bg-[#0d1215] px-3 text-sm font-semibold text-white">
                      {SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}×</option>)}
                    </select>
                  </div>
                  {playbackView === 'timeline' ? <div className="flex gap-2">
                    <button type="button" onClick={() => seekToTime((metadata?.startPeriod1 ?? 0) * 1000)} className="min-h-11 rounded-xl border border-line px-4 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]">1st half</button>
                    <button type="button" onClick={() => seekToTime((metadata?.startPeriod2 ?? 0) * 1000)} className="min-h-11 rounded-xl border border-line px-4 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]">2nd half</button>
                  </div> : null}
                </div>
                {events && playbackView === 'timeline' ? (
                  <EventTimeline
                    events={events.events}
                    currentTimeMs={currentTimeMs}
                    startTimeMs={data.videoTimeMs[0] ?? 0}
                    endTimeMs={data.videoTimeMs[data.frameCount - 1] ?? 0}
                    filter={eventFilter}
                    onFilter={setEventFilter}
                    onSeek={(timeMs) => { setPlaying(false); seekToTime(timeMs); }}
                    homeName={homeName}
                    awayName={awayName}
                  />
                ) : null}
              </div>
            </>
          )}

          {(error || metadataError || eventError || mismatch || eventMismatch || warningCount > 0) ? (
            <div className="flex items-start gap-3 rounded-2xl border border-amber-500/35 bg-amber-500/10 p-4 text-sm text-amber-100" role="alert">
              <CircleAlert aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
              <div className="space-y-1">
                {error ? <p>{error}</p> : null}
                {metadataError ? <p>Metadata: {metadataError}</p> : null}
                {eventError ? <p>Events: {eventError}</p> : null}
                {mismatch ? <p>Metadata match ID {metadata?.id} does not match tracking game ID {data?.gameRefId}.</p> : null}
                {eventMismatch ? <p>Events game ID {events?.gameId} does not match metadata ID {metadata?.id}.</p> : null}
                {warningCount > 0 ? <p>Skipped {data?.malformedLines ?? 0} malformed and {data?.blankLines ?? 0} blank lines.</p> : null}
              </div>
            </div>
          ) : null}
          {data ? <AnalyticsPanel events={events} data={data} workloadConfig={workloadConfig} currentTimeMs={data.videoTimeMs[analyticsIndex] ?? currentTimeMs} frameIndex={analyticsIndex} clockLabel={formatMatchClock(data.periodGameClockTime[analyticsIndex] ?? currentClock, data.period[analyticsIndex] ?? currentPeriod)} homeName={homeName} awayName={awayName} /> : null}
        </section>

        <aside className="space-y-4" aria-label="Replay settings">
          <div className="rounded-3xl border border-line bg-panel p-4 shadow-panel">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-muted">Files</p>
                <h2 className="mt-1 font-semibold text-white">Match data</h2>
              </div>
              {data ? <button type="button" onClick={() => { setData(null); setPlaying(false); setProgress(0); }} className="flex min-h-11 items-center gap-2 rounded-xl px-3 text-sm text-muted hover:bg-white/[0.05] hover:text-white"><RotateCcw aria-hidden="true" className="h-4 w-4" /> Replace</button> : null}
            </div>
            {data ? (
              <div className="space-y-3 text-sm">
                <div className="rounded-xl bg-black/20 p-3"><p className="truncate font-medium text-slate-100">{trackingName}</p><p className="mt-1 text-xs text-muted">{data.frameCount.toLocaleString()} frames · game {data.gameRefId ?? 'unknown'}</p></div>
                <DropZone label="Metadata" hint="Optional JSON metadata." accept=".json,application/json" fileName={metadataName} onFile={(file) => void loadMetadata(file)} />
                <DropZone label="Events" hint="Optional JSON array with match events." accept=".json,application/json" fileName={eventName} disabled={eventLoading} onFile={loadEvents} />
                {eventLoading ? <p className="px-1 text-xs text-muted" aria-live="polite">{eventStage} · {Math.round(eventProgress * 100)}%</p> : null}
              </div>
            ) : <p className="text-sm leading-6 text-muted">Choose files in the viewer to start a local replay.</p>}
          </div>

          <div className={`rounded-3xl border border-line bg-panel p-3 shadow-panel ${data ? '' : 'pointer-events-none opacity-45'}`} aria-disabled={!data}>
            <div className="px-3 pb-2 pt-1">
              <p className="text-xs font-bold uppercase tracking-[0.16em] text-muted">Display</p>
              <h2 className="mt-1 font-semibold text-white">Pitch layers</h2>
            </div>
            <Toggle label="Jersey numbers" checked={ui.showNumbers} onChange={(showNumbers) => ui.set({ showNumbers })} />
            <Toggle label="Ball" checked={ui.showBall} onChange={(showBall) => ui.set({ showBall })} />
            <Toggle label="60 Hz interpolation" checked={ui.interpolate} onChange={(interpolate) => ui.set({ interpolate })} />
            <Toggle label="Flip sides" checked={ui.flipSides} onChange={(flipSides) => ui.set({ flipSides })} />
            <Toggle label="Include halftime gap" checked={ui.includeEverything} onChange={(includeEverything) => ui.set({ includeEverything })} />
            <div className="mx-3 my-2 border-t border-line" />
            <fieldset className="px-3 py-2">
              <legend className="mb-2 text-sm text-slate-200">Tracking positions</legend>
              <div className="grid grid-cols-2 gap-2">
                {(['smoothed', 'raw'] as const).map((mode) => (
                  <label key={mode} className={`flex min-h-11 cursor-pointer items-center justify-center rounded-xl border px-3 text-sm font-semibold capitalize ${ui.positionMode === mode ? 'border-lime bg-lime/10 text-lime' : 'border-line text-muted'}`}>
                    <input className="sr-only" type="radio" name="position-mode" value={mode} checked={ui.positionMode === mode} onChange={() => ui.set({ positionMode: mode })} />
                    {mode}
                  </label>
                ))}
              </div>
            </fieldset>
          </div>

          <div className={`rounded-3xl border border-line bg-panel p-4 shadow-panel ${data ? '' : 'pointer-events-none opacity-45'}`} aria-disabled={!data}>
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-muted">Teams</p>
            <h2 className="mt-1 font-semibold text-white">Kit colors</h2>
            <div className="mt-4 space-y-3">
              {[
                { label: homeName, color: ui.homeColor, key: 'homeColor' as const },
                { label: awayName, color: ui.awayColor, key: 'awayColor' as const }
              ].map((team) => (
                <label key={team.key} className="flex min-h-11 items-center justify-between rounded-xl bg-black/20 px-3 text-sm text-slate-200">
                  <span>{team.label}</span>
                  <input aria-label={`${team.label} shirt color`} className="h-8 w-8 cursor-pointer" type="color" value={team.color} onChange={(event) => ui.set({ [team.key]: event.target.value })} />
                </label>
              ))}
            </div>
            <div className="mt-4 border-t border-line pt-4">
              <label className="flex items-center justify-between gap-3 text-sm text-slate-200" htmlFor="trail">Selected trail <span className="font-mono text-xs text-muted">{ui.trailLength}f</span></label>
              <input id="trail" className="mt-2 h-10 w-full" type="range" min={0} max={90} step={5} value={ui.trailLength} onChange={(event) => ui.set({ trailLength: Number(event.target.value) })} />
              <p className="text-xs leading-5 text-muted">Click a shirt on the pitch, then increase the trail length.</p>
            </div>
          </div>
          <WorkloadSettings config={workloadConfig} available={Boolean(data)} busy={workloadLoading} progress={workloadProgress} frames={workloadFrames} error={workloadError} onApply={recomputeWorkload} />
        </aside>
      </main>}
    </div>
  );
}

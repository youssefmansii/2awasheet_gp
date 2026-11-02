import { periodLabel } from '@replay/shared';
import { Pause, Play, Radio } from 'lucide-react';
import { useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { LiveHistory } from '../live/live-history';

function behindLabel(ms: number, receiving: boolean): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  const edge = receiving ? 'behind live' : 'before the latest frame';
  if (seconds < 60) return `${seconds} s ${edge}`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, '0')} ${edge}`;
}

/** YouTube-style DVR controls for the Live API: seek anywhere in the received history, then return to live. */
export function LiveSeekBar({ history, liveIndex, index, following, playing, receiving, clockComplete, onSeek, onTogglePlay, onGoLive }: {
  history: LiveHistory; liveIndex: number; index: number; following: boolean; playing: boolean; receiving: boolean; clockComplete: boolean;
  onSeek: (index: number) => void; onTogglePlay: () => void; onGoLive: () => void;
}) {
  const [hover, setHover] = useState<{ percent: number; label: string } | null>(null);
  const atLive = following && playing;
  const span = Math.max(1, liveIndex);
  const percent = Math.min(100, index / span * 100);
  const current = history.label(index, clockComplete);
  const live = history.label(liveIndex, clockComplete);
  const behindMs = history.videoTimeAt(liveIndex) - history.videoTimeAt(index);
  const jump = (seconds: number) => onSeek(history.shift(index, seconds));
  const markers = history.periodStarts();

  const hoverAt = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / Math.max(1, rect.width)));
    setHover({ percent: fraction * 100, label: history.label(Math.round(fraction * liveIndex), clockComplete) });
  };
  // Like a video player: arrows move 5 s, Page Up/Down 10 s, Home = first received frame, End = live.
  const keys = (event: KeyboardEvent<HTMLInputElement>) => {
    const action: Record<string, () => void> = {
      ArrowLeft: () => jump(-5), ArrowDown: () => jump(-5), ArrowRight: () => jump(5), ArrowUp: () => jump(5),
      PageDown: () => jump(-10), PageUp: () => jump(10), Home: () => onSeek(0), End: onGoLive
    };
    const run = action[event.key];
    if (run) { event.preventDefault(); run(); }
  };
  const iconButton = 'flex h-11 w-11 items-center justify-center rounded-xl border border-line bg-white/[0.035] text-slate-100 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-35';

  return <section className="space-y-2 px-3 pb-2" aria-label="Live seek controls">
    <div className="relative" onPointerMove={hoverAt} onPointerLeave={() => setHover(null)}>
      {hover ? <span aria-hidden="true" className="pointer-events-none absolute -top-7 z-10 -translate-x-1/2 rounded-md bg-black/85 px-2 py-0.5 font-mono text-xs text-white"
        style={{ left: `clamp(2rem, ${hover.percent}%, calc(100% - 2rem))` }}>{hover.label}</span> : null}
      <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-full bg-white/15">
        <div className="h-full bg-lime" style={{ width: `${percent}%` }} />
      </div>
      {markers.map((marker) => <span key={marker.index} aria-hidden="true" title={`${periodLabel(marker.period)} starts`}
        className="pointer-events-none absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-white/70" style={{ left: `${marker.index / span * 100}%` }} />)}
      <label className="sr-only" htmlFor="live-seek">Seek through live history</label>
      <input id="live-seek" type="range" min={0} max={liveIndex} step={1} value={Math.min(index, liveIndex)} disabled={liveIndex <= 0}
        aria-valuetext={atLive ? `${current}, ${receiving ? 'live' : 'latest frame'}` : `${current}, ${behindLabel(behindMs, receiving)}`}
        onChange={(event) => onSeek(Number(event.target.value))} onKeyDown={keys}
        className="live-seek relative block h-8 w-full cursor-pointer appearance-none bg-transparent disabled:cursor-not-allowed" />
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className={iconButton} aria-label="Back 10 seconds" title="Back 10 s (Page Down)" disabled={index <= 0} onClick={() => jump(-10)}>
        <span aria-hidden="true" className="text-xs font-bold">−10</span></button>
      <button type="button" className={iconButton} aria-label="Back 5 seconds" title="Back 5 s (←)" disabled={index <= 0} onClick={() => jump(-5)}>
        <span aria-hidden="true" className="text-xs font-bold">−5</span></button>
      <button type="button" aria-label={playing ? 'Pause playback' : 'Resume playback'} onClick={onTogglePlay}
        className="flex h-12 w-12 items-center justify-center rounded-2xl bg-lime text-[#152000] hover:bg-[#d6ff72]">
        {playing ? <Pause aria-hidden="true" className="h-5 w-5" /> : <Play aria-hidden="true" className="h-5 w-5" />}</button>
      <button type="button" className={iconButton} aria-label="Forward 5 seconds" title="Forward 5 s (→)" disabled={index >= liveIndex} onClick={() => jump(5)}>
        <span aria-hidden="true" className="text-xs font-bold">+5</span></button>
      <button type="button" className={iconButton} aria-label="Forward 10 seconds" title="Forward 10 s (Page Up)" disabled={index >= liveIndex} onClick={() => jump(10)}>
        <span aria-hidden="true" className="text-xs font-bold">+10</span></button>
      <span className="font-mono text-sm text-slate-200" data-testid="live-seek-time">{current} <span className="text-muted">/ {live}</span></span>
      <span className="ml-auto flex items-center gap-3">
        {atLive ? null : <span className="text-xs text-muted">{behindLabel(behindMs, receiving)}</span>}
        <button type="button" onClick={onGoLive} aria-pressed={atLive} aria-label={atLive ? (receiving ? 'Watching live' : 'At latest frame') : receiving ? 'Go live' : 'Go to latest frame'}
          className={`flex min-h-11 items-center gap-2 rounded-xl border px-4 text-sm font-bold ${atLive && receiving ? 'border-red-500/60 bg-red-500/15 text-red-300' : 'border-line text-slate-200 hover:bg-white/[0.05]'}`}>
          <span aria-hidden="true" className={`h-2.5 w-2.5 rounded-full ${atLive && receiving ? 'bg-red-500' : 'bg-slate-500'}`} />
          {atLive ? (receiving ? 'LIVE' : 'Latest') : <><Radio aria-hidden="true" className="h-4 w-4" />{receiving ? 'Go live' : 'Latest'}</>}
        </button>
      </span>
    </div>
  </section>;
}

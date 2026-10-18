import { formatMatchClock } from '@replay/shared';
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { highlightProgress, highlightTimeAtProgress, nearestHighlightTime, type HighlightClip } from '../replay/highlights';

export function HighlightControls({ clips, currentTimeMs, playing, emptyReason, onSeek, onPlayingChange }: {
  clips: HighlightClip[]; currentTimeMs: number; playing: boolean; emptyReason: string;
  onSeek: (timeMs: number) => void; onPlayingChange: (playing: boolean) => void;
}) {
  if (!clips.length) return <p role="status" className="rounded-xl border border-line bg-black/20 p-4 text-sm leading-6 text-muted">{emptyReason}</p>;
  const found = clips.findIndex((clip) => currentTimeMs <= clip.endMs);
  const index = found < 0 ? clips.length - 1 : found;
  const selected = clips[index]!;
  const duration = clips.reduce((sum, clip) => sum + clip.endMs - clip.startMs, 0);
  const progress = highlightProgress(clips, currentTimeMs);
  const jump = (next: number) => { onPlayingChange(false); onSeek(clips[next]!.startMs); };
  const toggle = () => {
    if (!playing) {
      if (currentTimeMs >= clips.at(-1)!.endMs) onSeek(clips[0]!.startMs);
      else if (currentTimeMs < selected.startMs) onSeek(nearestHighlightTime(clips, currentTimeMs));
    }
    onPlayingChange(!playing);
  };
  const transport = [
    { label: 'First highlight', target: 0, disabled: index === 0, Icon: SkipBack },
    { label: 'Previous highlight', target: Math.max(0, index - 1), disabled: index === 0, Icon: ChevronLeft },
    { label: 'Next highlight', target: Math.min(clips.length - 1, index + 1), disabled: index === clips.length - 1, Icon: ChevronRight },
    { label: 'Last highlight', target: clips.length - 1, disabled: index === clips.length - 1, Icon: SkipForward }
  ];
  const button = (item: typeof transport[number]) => <button key={item.label} type="button" aria-label={item.label}
    title={item.label} disabled={item.disabled} onClick={() => jump(item.target)}
    className="flex h-11 w-11 items-center justify-center rounded-xl border border-line bg-white/[0.035] text-slate-100 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-35">
    <item.Icon aria-hidden="true" className="h-5 w-5" />
  </button>;
  return <section className="space-y-3" aria-label="Automatic highlights playback">
    <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
      <p role="status" className="font-semibold text-lime">Highlight {index + 1} / {clips.length}</p>
      <span className="font-mono">{formatMatchClock(progress / 1000, 1)} / {formatMatchClock(duration / 1000, 1)} highlights time</span>
    </div>
    <label className="sr-only" htmlFor="highlight-seek">Seek through highlights</label>
    <input id="highlight-seek" type="range" min={0} max={duration} step={1} value={progress} disabled={duration === 0}
      className="block h-11 w-full cursor-pointer" onChange={(e) => {
        onPlayingChange(false); onSeek(highlightTimeAtProgress(clips, Number(e.target.value)));
      }} />
    <div className="flex flex-wrap items-center justify-center gap-2">
      {transport.slice(0, 2).map(button)}
      <button type="button" aria-label={playing ? 'Pause highlights' : 'Play highlights'} onClick={toggle}
        className="flex h-14 w-14 items-center justify-center rounded-2xl bg-lime text-[#152000] hover:bg-[#d6ff72] active:bg-[#b8df50]">
        {playing ? <Pause aria-hidden="true" className="h-6 w-6 fill-current" /> : <Play aria-hidden="true" className="ml-0.5 h-6 w-6 fill-current" />}
      </button>
      {transport.slice(2).map(button)}
    </div>
    <div className="text-sm text-muted">
      <label htmlFor="highlight-choice">Choose highlight</label>
      <select id="highlight-choice" value={index} onChange={(e) => jump(Number(e.target.value))}
        className="mt-2 min-h-11 w-full min-w-0 rounded-xl border border-line bg-ink px-3 text-sm text-white">
        {clips.map((clip, i) => <option value={i} key={clip.startMs}>{i + 1}. {clip.events.map((e) => e.label).join(' / ')}</option>)}
      </select>
    </div>
    <p className="text-xs leading-5 text-muted">5 seconds before and after each event, clipped to available tracking. Nearby actions are merged. Play skips automatically to the next clip. Goal candidates preserve the provider’s raw outcome, not a verified score.</p>
  </section>;
}

import type { MatchEvent } from '@replay/shared';
import { useState } from 'react';

export type EventFilter = 'ALL' | 'PA' | 'SH' | 'CH' | 'CR' | 'FO' | 'SUB';

const FILTERS: Array<{ value: EventFilter; label: string }> = [
  { value: 'ALL', label: 'Highlights' },
  { value: 'PA', label: 'Passes' },
  { value: 'SH', label: 'Shots' },
  { value: 'CH', label: 'Challenges' },
  { value: 'CR', label: 'Crosses' },
  { value: 'FO', label: 'Fouls' },
  { value: 'SUB', label: 'Subs' }
];

function matches(event: MatchEvent, filter: EventFilter): boolean {
  if (filter !== 'ALL') return event.possessionType === filter || event.gameType === filter;
  return event.possessionType === 'SH'
    || event.possessionType === 'FO'
    || event.gameType === 'SUB'
    || event.gameType === 'FIRSTKICKOFF'
    || event.gameType === 'SECONDKICKOFF'
    || event.gameType === 'END';
}

function eventLabel(event: MatchEvent): string {
  const type = event.possessionType ?? event.gameType ?? 'Event';
  const actor = event.passerPlayerName ?? event.playerName ?? 'Unknown player';
  const target = event.targetPlayerName ? ` → ${event.targetPlayerName}` : '';
  const outcome = event.outcome ? ` · ${event.outcome}` : '';
  return `${event.clock} · ${type} · ${actor}${target}${outcome}`;
}

export function EventTimeline({
  events,
  currentTimeMs,
  startTimeMs,
  endTimeMs,
  filter,
  onFilter,
  onSeek,
  homeName,
  awayName
}: {
  events: MatchEvent[];
  currentTimeMs: number;
  startTimeMs: number;
  endTimeMs: number;
  filter: EventFilter;
  onFilter: (filter: EventFilter) => void;
  onSeek: (timeMs: number) => void;
  homeName: string;
  awayName: string;
}) {
  const [teamFilter, setTeamFilter] = useState<'ALL' | 'HOME' | 'AWAY'>('ALL');
  const filtered = events.filter((event) => matches(event, filter)
    && (teamFilter === 'ALL' || event.homeTeam === (teamFilter === 'HOME')));
  const duration = Math.max(1, endTimeMs - startTimeMs);
  const nearby = [...filtered]
    .sort((a, b) => Math.abs(a.eventTimeMs - currentTimeMs) - Math.abs(b.eventTimeMs - currentTimeMs))
    .slice(0, 6)
    .sort((a, b) => a.eventTimeMs - b.eventTimeMs);

  const seekNearest = (fraction: number) => {
    if (!filtered.length) return;
    const target = startTimeMs + duration * fraction;
    let nearest = filtered[0]!;
    for (const event of filtered) {
      if (Math.abs(event.eventTimeMs - target) < Math.abs(nearest.eventTimeMs - target)) nearest = event;
    }
    onSeek(nearest.eventTimeMs);
  };

  return (
    <section className="mt-5 border-t border-line pt-4" aria-labelledby="event-timeline-title">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 id="event-timeline-title" className="text-sm font-semibold text-white">Event timeline</h3>
          <p className="text-xs text-muted">{filtered.length.toLocaleString()} matching events</p>
        </div>
        <div className="flex max-w-full gap-2 overflow-x-auto pb-1" aria-label="Event type filters">
          {FILTERS.map((item) => (
            <button
              key={item.value}
              type="button"
              aria-pressed={filter === item.value}
              onClick={() => onFilter(item.value)}
              className={`min-h-11 shrink-0 rounded-xl border px-3 text-xs font-semibold transition-colors ${filter === item.value ? 'border-lime bg-lime/10 text-lime' : 'border-line text-muted hover:text-white'}`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-2 flex flex-wrap gap-2" aria-label="Event team filters">
        {([
          ['ALL', 'Both teams'],
          ['HOME', homeName],
          ['AWAY', awayName]
        ] as const).map(([value, label]) => (
          <button key={value} type="button" aria-pressed={teamFilter === value} onClick={() => setTeamFilter(value)} className={`min-h-11 rounded-xl px-3 text-xs font-semibold ${teamFilter === value ? 'bg-white/10 text-white' : 'text-muted hover:bg-white/[0.04] hover:text-white'}`}>{label}</button>
        ))}
      </div>

      <button
        type="button"
        className="relative mt-3 block h-11 w-full overflow-hidden rounded-xl border border-line bg-black/25 focus-visible:outline-lime"
        aria-label="Event timeline. Click to seek to the nearest visible event."
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          seekNearest((event.clientX - rect.left) / rect.width);
        }}
      >
        <span className="absolute inset-x-3 top-1/2 h-px bg-slate-600" aria-hidden="true" />
        {filtered.map((event) => {
          const left = Math.max(0, Math.min(100, ((event.eventTimeMs - startTimeMs) / duration) * 100));
          const active = Math.abs(event.eventTimeMs - currentTimeMs) < 750;
          return (
            <span
              key={event.id}
              className={`absolute top-1/2 h-5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full ${event.homeTeam === true ? 'bg-sky-300' : event.homeTeam === false ? 'bg-emerald-300' : 'bg-amber-300'} ${active ? 'ring-4 ring-white/40' : ''}`}
              style={{ left: `${left}%` }}
              aria-hidden="true"
            />
          );
        })}
      </button>

      <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3" aria-label="Events nearest the playhead">
        {nearby.map((event) => (
          <button
            key={event.id}
            type="button"
            onClick={() => onSeek(event.eventTimeMs)}
            className="min-h-11 rounded-xl border border-line bg-white/[0.025] px-3 py-2 text-left text-xs leading-5 text-slate-200 transition-colors hover:border-slate-500 hover:bg-white/[0.05]"
          >
            {eventLabel(event)}
          </button>
        ))}
      </div>
    </section>
  );
}

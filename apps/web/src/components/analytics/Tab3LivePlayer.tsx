import { Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useUiStore } from '../../store';
import {
  clockLabel,
  errorMessage,
  fmt,
  getJson,
  isAbort,
  pct,
  signed,
  withMatch,
  type LiveDetail,
  type LivePlayersResponse,
  type LiveStats,
  type LiveTimeline,
  type MatchInfo,
  type RankEntry,
  type Side,
  type SummaryLine
} from './api';
import { PlayerPitch, type PitchMode } from './PlayerPitch';
import { ErrorBox, Panel, Spinner, Stat } from './ui';

const AXIS = { fill: '#93a0a8', fontSize: 11 };
const TOOLTIP_STYLE = { background: '#14191d', border: '1px solid #2b3339', borderRadius: 8, fontSize: 12 };
const SPEEDS = [10, 30, 60] as const;

const MODES: { key: PitchMode; label: string; what: string }[] = [
  { key: 'now', label: 'Now on the pitch', what: 'Where all 22 players are at this moment. Click a player to follow him.' },
  { key: 'heat', label: 'Heat map', what: 'Where he spent his time so far: the redder the square, the more time there. White dot = his average position. Drawn as if his team always attacks to the right.' },
  { key: 'passes', label: 'Passes', what: 'Every pass so far. Green = completed, red = not completed, yellow = key pass (led to a shot), dashed = cross. The dot is where the pass started.' },
  { key: 'shots', label: 'Shots', what: 'Every shot so far. The bigger the circle, the better the chance (xG). Green = goal, yellow = saved, red = off target, grey = blocked, purple = goal ruled out.' },
  { key: 'defending', label: 'Defending', what: 'Blue diamonds = interceptions, blocks and clearances. Circles = duels (green won, red lost).' }
];

const METRICS: { key: string; label: string; unit: string; pick: (r: LiveTimeline['minutes'][number]) => number | null }[] = [
  { key: 'distance', label: 'Distance', unit: 'km', pick: (r) => r.distance_km },
  { key: 'passes', label: 'Passes', unit: '', pick: (r) => r.passes },
  { key: 'acc', label: 'Pass accuracy', unit: '%', pick: (r) => (r.passes >= 5 ? (100 * r.passes_ok) / r.passes : null) },
  { key: 'def', label: 'Defensive actions', unit: '', pick: (r) => r.defensive },
  { key: 'value', label: 'Value added', unit: '', pick: (r) => r.value },
  { key: 'hsr', label: 'High-speed running', unit: 'm', pick: (r) => r.hsr_m }
];

const TONE: Record<SummaryLine['tone'], { dot: string; text: string }> = {
  good: { dot: 'bg-emerald-400', text: 'text-slate-100' },
  ok: { dot: 'bg-amber-300', text: 'text-slate-100' },
  poor: { dot: 'bg-red-400', text: 'text-slate-100' },
  info: { dot: 'bg-slate-500', text: 'text-slate-300' }
};

const MOMENT_STYLE: Record<string, string> = {
  goal: 'border-emerald-400/60 text-emerald-300',
  assist: 'border-emerald-400/60 text-emerald-300',
  key: 'border-amber-300/60 text-amber-200',
  shot: 'border-orange-400/60 text-orange-300',
  save: 'border-sky-400/60 text-sky-300',
  def: 'border-sky-400/60 text-sky-300',
  card: 'border-yellow-300/70 text-yellow-200'
};

// ---------------------------------------------------------------------------
function SplitBar({ parts, empty }: { parts: { label: string; value: number; color: string }[]; empty: string }) {
  const total = parts.reduce((s, p) => s + p.value, 0);
  if (!total) return <p className="text-xs text-muted">{empty}</p>;
  return (
    <div>
      <div className="flex h-3 w-full overflow-hidden rounded bg-ink" role="img" aria-label={parts.map((p) => `${p.label} ${p.value}`).join(', ')}>
        {parts.map((p) => (p.value > 0 ? <div key={p.label} style={{ width: `${(100 * p.value) / total}%`, background: p.color }} title={`${p.label}: ${p.value}`} /> : null))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-300">
        {parts.map((p) => (
          <span key={p.label} className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: p.color }} aria-hidden="true" />
            {p.label} <span className="font-mono font-bold text-white">{p.value}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function ratio(a: number, b: number): string {
  return b > 0 ? `${a}/${b}` : '0';
}

function rankText(e: RankEntry | undefined): string {
  if (!e || e.rank === null) return '–';
  return `${e.rank} of ${e.n}`;
}

function fmtRank(key: string, v: number | null): string {
  if (v === null || !Number.isFinite(v)) return '–';
  if (key === 'pass_acc' || key === 'duel_win_pct') return pct(v);
  if (key === 'distance_m') return `${(v / 1000).toFixed(1)} km`;
  if (key === 'hsr_m') return `${v.toFixed(0)} m`;
  if (key === 'xg' || key === 'vaep_oa') return key === 'vaep_oa' ? signed(v, 2) : v.toFixed(2);
  return v.toFixed(v % 1 === 0 ? 0 : 1);
}

// ---------------------------------------------------------------------------
export function Tab3LivePlayer({ info }: { info: MatchInfo }) {
  const homeColor = useUiStore((s) => s.homeColor);
  const homeText = useUiStore((s) => s.homeTextColor);
  const awayColor = useUiStore((s) => s.awayColor);
  const awayText = useUiStore((s) => s.awayTextColor);

  const periods = info.periods;
  const durations = useMemo(() => periods.map((p) => Math.max(1, (p.last_frame - p.first_frame) / info.fps)), [periods, info.fps]);
  const total = useMemo(() => durations.reduce((a, b) => a + b, 0), [durations]);

  /** seconds from the first frame of the match -> the PFF frame number */
  const frameAt = useCallback(
    (sec: number): number => {
      let acc = 0;
      for (let i = 0; i < periods.length; i++) {
        const p = periods[i];
        const d = durations[i] ?? 0;
        if (!p) continue;
        if (sec <= acc + d || i === periods.length - 1) return Math.min(p.last_frame, p.first_frame + Math.round(Math.max(0, sec - acc) * info.fps));
        acc += d;
      }
      return periods[0]?.first_frame ?? 0;
    },
    [periods, durations, info.fps]
  );
  /** the PFF frame number -> seconds from the first frame of the match */
  const secOfFrame = useCallback(
    (frameNum: number): number => {
      let acc = 0;
      for (let i = 0; i < periods.length; i++) {
        const p = periods[i];
        const d = durations[i] ?? 0;
        if (!p) continue;
        if (frameNum <= p.last_frame || i === periods.length - 1) return acc + Math.min(d, Math.max(0, (frameNum - p.first_frame) / info.fps));
        acc += d;
      }
      return 0;
    },
    [periods, durations, info.fps]
  );

  const [sec, setSec] = useState<number>(total); // opens at full time: the whole match so far
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(30);
  const [side, setSide] = useState<Side>('home');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<PitchMode>('now');
  const [metric, setMetric] = useState<string>('distance');
  const [roster, setRoster] = useState<LivePlayersResponse | null>(null);
  const [detail, setDetail] = useState<LiveDetail | null>(null);
  const [timeline, setTimeline] = useState<LiveTimeline | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef(false);
  busyRef.current = busy;

  const frameNum = frameAt(sec);

  // ---- everybody at this moment ----
  useEffect(() => {
    const ctrl = new AbortController();
    const h = window.setTimeout(
      () => {
        setBusy(true);
        getJson<LivePlayersResponse>(withMatch(`/api/live/players?frame_id=${frameNum}`, info.match_id), ctrl.signal)
          .then((r) => {
            setRoster(r);
            setError(null);
            setBusy(false);
          })
          .catch((e: unknown) => {
            if (isAbort(e)) return;
            setError(errorMessage(e));
            setBusy(false);
          });
      },
      playing ? 0 : 120
    );
    return () => {
      window.clearTimeout(h);
      ctrl.abort();
    };
  }, [frameNum, info.match_id, playing]);

  // first load: follow the home player who was most involved on the ball
  useEffect(() => {
    if (selectedId !== null || !roster) return;
    const best = roster.players
      .filter((p) => p.side === 'home' && p.player_id)
      .sort((a, b) => b.on_ball_actions + b.minutes / 10 - (a.on_ball_actions + a.minutes / 10))[0];
    const first = best ?? roster.players.find((p) => p.player_id);
    if (first?.player_id) {
      setSelectedId(first.player_id);
      setSide(first.side);
    }
  }, [roster, selectedId]);

  // ---- the selected player in detail ----
  useEffect(() => {
    if (!selectedId) return;
    const ctrl = new AbortController();
    const h = window.setTimeout(
      () => {
        getJson<LiveDetail>(withMatch(`/api/live/player/${encodeURIComponent(selectedId)}?frame_id=${frameNum}`, info.match_id), ctrl.signal)
          .then((d) => setDetail(d))
          .catch((e: unknown) => {
            if (!isAbort(e)) setError(errorMessage(e));
          });
      },
      playing ? 0 : 120
    );
    return () => {
      window.clearTimeout(h);
      ctrl.abort();
    };
  }, [selectedId, frameNum, info.match_id, playing]);

  // ---- the whole match of the selected player (charts + key moments) ----
  useEffect(() => {
    if (!selectedId) return;
    const ctrl = new AbortController();
    setTimeline(null);
    getJson<LiveTimeline>(withMatch(`/api/live/player/${encodeURIComponent(selectedId)}/timeline`, info.match_id), ctrl.signal)
      .then(setTimeline)
      .catch((e: unknown) => {
        if (!isAbort(e)) setError(errorMessage(e));
      });
    return () => ctrl.abort();
  }, [selectedId, info.match_id]);

  // ---- play: the clock moves `speed` match-seconds every real second (waits if the server is still busy) ----
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      if (busyRef.current) return;
      setSec((s) => {
        const n = Math.min(total, s + speed);
        if (n >= total) setPlaying(false);
        return n;
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, [playing, speed, total]);

  const stats: LiveStats | null = detail && detail.stats.player_id === selectedId ? detail.stats : null;
  const sel = roster?.players.find((p) => p.player_id === selectedId) ?? null;
  const teamColor = (stats ?? sel)?.side === 'away' ? awayColor : homeColor;
  const homeName = info.teams.home.name ?? 'Home';
  const awayName = info.teams.away.name ?? 'Away';
  const clock = roster ? clockLabel(roster.period, roster.minute) : '–';
  const currentK = sec / 60;
  const seekFrame = (f: number) => {
    setPlaying(false);
    setSec(secOfFrame(f));
  };

  const periodEnds = useMemo(() => {
    const out: number[] = [];
    let acc = 0;
    for (const d of durations) {
      acc += d;
      out.push(acc);
    }
    return out;
  }, [durations]);

  const chartData = useMemo(() => {
    const m = METRICS.find((x) => x.key === metric) ?? METRICS[0];
    if (!timeline || !m) return [];
    return timeline.minutes.filter((r) => r.on_pitch).map((r) => ({ k: r.k, v: m.pick(r) }));
  }, [timeline, metric]);
  const metricInfo = METRICS.find((x) => x.key === metric) ?? METRICS[0];

  const picker = (roster?.players ?? []).filter((p) => p.side === side);
  const rankRows: [string, RankEntry][] = detail ? Object.entries(detail.team) : [];

  if (!periods.length) return <ErrorBox message="This match has no playable periods." />;

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------ what is this page */}
      <section className="rounded-2xl border border-line bg-panel px-4 py-3 text-sm leading-relaxed text-slate-200">
        <span className="font-bold uppercase tracking-[0.14em] text-white">Live player view</span> · Pick a player and move the clock (or press Play). Every number is what he had done <span className="text-white">up to that moment</span>:
        passes and accuracy, shots on and off target, duels, running, and a plain-English verdict. Counts (passes, shots, tackles) come straight from the event data; running comes from the tracking; xG and “value added” are model estimates.
      </section>

      {error ? <ErrorBox message={error} /> : null}

      <div className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
        {/* ------------------------------------------------ player picker */}
        <aside className="space-y-3 xl:sticky xl:top-3 xl:self-start">
          <div className="rounded-2xl border border-line bg-panel p-3">
            <div role="tablist" aria-label="Team" className="mb-2 flex gap-1 rounded-xl border border-line bg-ink/60 p-1">
              {([['home', homeName, homeColor], ['away', awayName, awayColor]] as const).map(([k, name, col]) => (
                <button
                  key={k}
                  type="button"
                  role="tab"
                  aria-selected={side === k}
                  onClick={() => setSide(k)}
                  className={`min-h-9 flex-1 truncate rounded-lg px-2 text-sm font-semibold ${side === k ? 'bg-white/10 text-white' : 'text-muted hover:text-white'}`}
                >
                  <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full align-middle" style={{ background: col }} aria-hidden="true" />
                  {name}
                </button>
              ))}
            </div>
            {!roster ? <Spinner label="Loading players…" /> : null}
            <ul className="max-h-[520px] space-y-1 overflow-y-auto pr-1" aria-label="Players">
              {picker.map((p) => {
                const active = p.player_id === selectedId;
                const played = p.minutes >= 0.5;
                return (
                  <li key={p.slot}>
                    <button
                      type="button"
                      disabled={!p.player_id}
                      onClick={() => p.player_id && setSelectedId(p.player_id)}
                      aria-pressed={active}
                      className={`flex min-h-11 w-full items-center gap-2 rounded-xl border px-2 py-1.5 text-left transition ${active ? 'border-lime bg-lime/10' : 'border-transparent hover:bg-white/[0.05]'} ${played ? '' : 'opacity-50'}`}
                    >
                      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-bold" style={{ background: p.side === 'home' ? homeColor : awayColor, color: p.side === 'home' ? homeText : awayText }}>
                        {p.jersey}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-semibold text-white">{p.name}</span>
                        <span className="block truncate text-[11px] text-muted">
                          {p.position ?? ''} · {played ? `${p.passes_ok}/${p.passes} passes · ${p.shots} shot${p.shots === 1 ? '' : 's'}` : 'not played yet'}
                          {played && !p.on_pitch ? ' · off the pitch' : ''}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        </aside>

        <div className="min-w-0 space-y-4">
          {/* ------------------------------------------------ clock */}
          <section className="rounded-2xl border border-line bg-panel p-3" aria-label="Match clock">
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setPlaying((v) => !v)} className="inline-flex min-h-10 min-w-24 items-center justify-center gap-2 rounded-lg bg-lime px-3 text-sm font-bold text-ink" aria-label={playing ? 'Pause' : 'Play'}>
                {playing ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
                {playing ? 'Pause' : 'Play'}
              </button>
              <button type="button" onClick={() => { setPlaying(false); setSec(0); }} className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-line px-3 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]">
                <SkipBack size={14} aria-hidden="true" /> Start
              </button>
              {periods.length > 1 ? (
                <button type="button" onClick={() => { setPlaying(false); setSec(periodEnds[0] ?? 0); }} className="min-h-10 rounded-lg border border-line px-3 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]">
                  Half time
                </button>
              ) : null}
              <button type="button" onClick={() => { setPlaying(false); setSec(total); }} className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-line px-3 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]">
                Full time <SkipForward size={14} aria-hidden="true" />
              </button>
              <div className="ml-auto flex items-center gap-1" role="group" aria-label="Play speed">
                <span className="mr-1 text-xs text-muted">Speed</span>
                {SPEEDS.map((s) => (
                  <button key={s} type="button" onClick={() => setSpeed(s)} aria-pressed={speed === s} className={`min-h-9 rounded-lg px-2.5 text-xs font-bold ${speed === s ? 'bg-white/15 text-white' : 'text-muted hover:text-white'}`}>
                    {s}×
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-3 flex items-center gap-3">
              <div className="w-20 shrink-0 text-center">
                <div className="font-mono text-2xl font-bold tabular-nums text-white">{clock}</div>
                <div className="text-[10px] uppercase tracking-wider text-muted">{busy ? 'updating…' : roster ? `period ${roster.period}` : ''}</div>
              </div>
              <div className="relative min-w-0 flex-1">
                <input
                  type="range"
                  min={0}
                  max={Math.round(total)}
                  step={1}
                  value={Math.round(sec)}
                  onChange={(e) => {
                    setPlaying(false);
                    setSec(Number(e.target.value));
                  }}
                  aria-label="Match time"
                  className="h-8 w-full accent-lime"
                />
                {periodEnds.slice(0, -1).map((end) => (
                  <span key={end} className="pointer-events-none absolute top-0 h-2 w-0.5 bg-slate-400" style={{ left: `${(100 * end) / total}%` }} aria-hidden="true" />
                ))}
              </div>
            </div>
            <p className="mt-1 text-[11px] text-muted">
              Opens at full time, so you see the whole match. Drag the slider back to judge a player at any earlier moment. Play runs {speed} match-seconds per real second.
            </p>
          </section>

          {/* ------------------------------------------------ headline + pitch */}
          <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
            <Panel
              title={stats ? `${stats.name} · #${stats.jersey}` : 'Choose a player'}
              subtitle={stats ? `${stats.team} · ${stats.position ?? ''} · ${stats.minutes.toFixed(0)} min played${stats.on_pitch ? '' : ' · not on the pitch right now'}` : 'Pick one from the list, or click a player on the pitch.'}
              right={detail?.now?.speed_kmh != null ? <span className="rounded-lg border border-line px-2 py-1 font-mono text-xs text-slate-200">{detail.now.speed_kmh.toFixed(1)} km/h now</span> : null}
            >
              {detail ? (
                <ul className="space-y-1.5" aria-label="Plain-English summary">
                  {detail.summary.map((l, i) => (
                    <li key={i} className={`flex gap-2 text-sm leading-snug ${TONE[l.tone].text}`}>
                      <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${TONE[l.tone].dot}`} aria-hidden="true" />
                      <span>{l.text}</span>
                    </li>
                  ))}
                </ul>
              ) : selectedId ? (
                <Spinner label="Loading his numbers…" />
              ) : null}
              {detail && !detail.has_events ? <p className="mt-2 text-xs text-amber-300">{detail.events_note}</p> : null}
              <p className="mt-3 text-[11px] text-muted">
                <span className="mr-3"><span className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-400 align-middle" />among the best in his team</span>
                <span className="mr-3"><span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-300 align-middle" />middle</span>
                <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-red-400 align-middle" />among the weakest</span>
              </p>
            </Panel>

            <Panel title="Pitch" subtitle={MODES.find((m) => m.key === mode)?.what ?? ''} right={null}>
              <div role="tablist" aria-label="Pitch view" className="mb-2 flex flex-wrap gap-1">
                {MODES.map((m) => (
                  <button key={m.key} type="button" role="tab" aria-selected={mode === m.key} onClick={() => setMode(m.key)} className={`min-h-9 rounded-lg px-3 text-xs font-semibold ${mode === m.key ? 'bg-lime text-ink' : 'border border-line text-slate-200 hover:bg-white/[0.05]'}`}>
                    {m.label}
                  </button>
                ))}
              </div>
              <PlayerPitch
                mode={mode}
                length={info.pitch.length}
                width={info.pitch.width}
                players={roster?.players ?? []}
                ball={roster?.ball ?? null}
                selectedSlot={sel?.slot ?? null}
                onSelect={(slot) => {
                  const p = roster?.players.find((x) => x.slot === slot);
                  if (p?.player_id) {
                    setSelectedId(p.player_id);
                    setSide(p.side);
                  }
                }}
                detail={detail}
                homeColor={homeColor}
                awayColor={awayColor}
                homeText={homeText}
                awayText={awayText}
                teamColor={teamColor}
              />
            </Panel>
          </div>

          {stats ? (
            <>
              {/* ------------------------------------------------ passing */}
              <Panel title="Passing" subtitle="How often his passes reach a team-mate, and how much they help the attack." right={null}>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="Passes completed" value={ratio(stats.passes_ok, stats.passes)} hint={`${pct(stats.pass_acc)} accuracy`} help="Completed passes divided by all passes he tried. Crosses are counted separately." />
                  <Stat label="Key passes" value={String(stats.key_passes)} hint={stats.assists ? `${stats.assists} assist${stats.assists === 1 ? '' : 's'}` : 'passes that led to a shot'} help="A completed pass whose receiver took a shot within 8 seconds." />
                  <Stat label="Progressive" value={String(stats.progressive_passes)} hint="move the ball ≥10 m closer to goal" help="Completed passes that bring the ball at least 10 metres closer to the opponent's goal line." />
                  <Stat label="Into final third" value={String(stats.final_third_passes)} hint={`${stats.box_passes} into the box`} help="Completed passes that start outside and end inside the attacking third (and the penalty box)." />
                  <Stat label="Crosses" value={ratio(stats.crosses_ok, stats.crosses)} hint={stats.crosses ? `${pct(stats.cross_acc)} accurate` : 'none yet'} />
                  <Stat label="Lines broken" value={String(stats.lines_broken)} hint="opponent lines passed through" help="How many of the opponent's lines (attack / midfield / defence) his passes went through, from the event data." />
                  <Stat label="Expected assists" value={stats.xa.toFixed(2)} hint="xA: chances he created" help="Sum of the xG of the shots that came from his key passes." />
                  <Stat label="Received under pressure" value={String(stats.receptions_pressured)} hint={`of ${stats.receptions} receptions`} help="Times he received the ball while an opponent was pressuring him (provider label)." />
                </div>
              </Panel>

              {/* ------------------------------------------------ shooting */}
              <Panel title="Shooting" subtitle="On target = goal or saved. xG is the chance each shot had of scoring, from where it was taken." right={null}>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="Shots" value={String(stats.shots)} hint={`${stats.shots_on_target} on target · ${stats.shots - stats.shots_on_target - stats.goals_ruled_out - stats.shots_other} off or blocked`} help="All attempts. A shot that is a goal or is saved counts as on target." />
                  <Stat label="On target" value={ratio(stats.shots_on_target, stats.shots)} hint={stats.shots ? `${pct(stats.shots_on_target / stats.shots)} of his shots` : 'no shots yet'} />
                  <Stat label="Goals" value={String(stats.goals)} hint={stats.goals_ruled_out ? `${stats.goals_ruled_out} ruled out (offside / VAR)` : 'counted only if play restarts with a kick-off'} help="A goal counts only when the next restart is a kick-off, so goals that were disallowed are not counted." />
                  <Stat label="xG" value={stats.xg.toFixed(2)} hint={stats.shots ? `${signed(stats.goals - stats.xg, 2)} goals vs expected` : 'expected goals'} help="Expected goals: the sum of the scoring chance of each shot (our distance-and-angle model; penalties 0.76)." />
                </div>
                <div className="mt-3">
                  <SplitBar
                    empty="No shots yet."
                    parts={[
                      { label: 'Goals', value: stats.goals, color: '#4ade80' },
                      { label: 'Saved', value: stats.shots_saved, color: '#facc15' },
                      { label: 'Off target', value: stats.shots_off, color: '#f87171' },
                      { label: 'Blocked', value: stats.shots_blocked, color: '#94a3b8' },
                      { label: 'Goal ruled out', value: stats.goals_ruled_out, color: '#c084fc' },
                      { label: 'Other', value: stats.shots_other, color: '#64748b' }
                    ]}
                  />
                </div>
              </Panel>

              {/* ------------------------------------------------ defending */}
              <Panel title="Defending and duels" subtitle="Winning the ball and winning one-against-one contests." right={null}>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="Duels won" value={ratio(stats.duels_won, stats.duels)} hint={stats.duels ? `${pct(stats.duel_win_pct)} won` : 'none yet'} help="Every one-against-one contest he took part in where the data says who won (tackles, aerials, dribbles, 50-50s)." />
                  <Stat label="Tackles won" value={ratio(stats.tackles_won, stats.tackles)} hint="when he challenged for the ball" />
                  <Stat label="Aerials won" value={ratio(stats.aerials_won, stats.aerials)} hint="headed duels" />
                  <Stat label="Dribbles won" value={ratio(stats.dribbles_won, stats.dribbles)} hint="beating a defender" />
                  <Stat label="Interceptions" value={String(stats.interceptions)} hint="passes he cut out" />
                  <Stat label="Blocks" value={String(stats.blocks)} hint="shots and passes" />
                  <Stat label="Clearances" value={String(stats.clearances)} hint="" />
                  <Stat label="Cautions" value={String(stats.cautions)} hint={stats.cautions ? 'yellow card' : 'none'} />
                </div>
              </Panel>

              {stats.position === 'GK' ? (
                <Panel title="Goalkeeping" subtitle="Shots on target that he faced." right={null}>
                  <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                    <Stat label="Saves" value={String(stats.saves)} hint={`of ${stats.shots_on_target_faced} on target`} />
                    <Stat label="Goals conceded" value={String(stats.goals_conceded)} hint="" />
                    <Stat label="Save %" value={pct(stats.save_pct)} hint="saves ÷ shots on target faced" />
                    <Stat label="Passes" value={ratio(stats.passes_ok, stats.passes)} hint={`${pct(stats.pass_acc)} accurate`} />
                  </div>
                </Panel>
              ) : null}

              {/* ------------------------------------------------ physical */}
              <Panel title="Running" subtitle="From the tracking data. Speeds above 36 km/h are treated as tracking errors and ignored." right={null}>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="Distance" value={`${(stats.distance_m / 1000).toFixed(2)} km`} hint={stats.minutes > 0.5 ? `${(stats.distance_m / stats.minutes).toFixed(0)} m per minute` : ''} help="Total distance run, from his tracked speed." />
                  <Stat label="High-speed running" value={`${stats.hsr_m.toFixed(0)} m`} hint="between 19.8 and 25.2 km/h" help="Distance covered at 19.8–25.2 km/h (5.5–7 m/s)." />
                  <Stat label="Sprints" value={String(stats.sprints)} hint={`${stats.sprint_m.toFixed(0)} m above 25.2 km/h`} help="Runs above 25.2 km/h (7 m/s) lasting at least one second." />
                  <Stat label="Top speed" value={`${stats.top_speed_kmh.toFixed(1)} km/h`} hint="best speed held for 1 second" help="The highest speed he kept for a full second. Tracking noise can still overstate it by a few km/h, so compare players with each other rather than with published records." />
                </div>
              </Panel>

              {/* ------------------------------------------------ value */}
              <Panel title="Value added" subtitle="Did his on-ball actions make his team more likely to score? (model estimate)" right={null}>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="VAEP above average" value={signed(stats.vaep_oa, 2)} hint={`over ${stats.value_actions} actions`} help="Sum of how much each action changed the team's chance of scoring, minus what an average player would have added in the same situation. Above 0 = better than average." />
                  <Stat label="Raw VAEP" value={signed(stats.vaep_raw, 2)} hint="before the average is removed" />
                  <Stat label="OBV (decisions)" value={signed(stats.obv, 2)} hint="ignores luck" help="Like VAEP but uses the expected result of each pass, so a good pass that was unluckily intercepted is not punished." />
                  <Stat label="Actions on the ball" value={String(stats.on_ball_actions)} hint="passes, shots, carries, tackles…" />
                </div>
              </Panel>

              {/* ------------------------------------------------ vs team-mates */}
              <Panel title="Compared with his team-mates" subtitle={`Ranked against ${detail?.team_pool.label ?? 'team-mates'} with at least 10 minutes played (${detail?.team_pool.size ?? 0} so far, himself included if he qualifies). Rank 1 = best.`} right={null}>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[420px] text-sm">
                    <thead>
                      <tr className="text-left text-[11px] uppercase tracking-wider text-muted">
                        <th className="py-1 pr-2 font-semibold">Measure</th>
                        <th className="py-1 pr-2 text-right font-semibold">Him</th>
                        <th className="py-1 pr-2 text-right font-semibold">Team average</th>
                        <th className="py-1 text-right font-semibold">Rank</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rankRows.map(([key, e]) => (
                        <tr key={key} className="border-t border-line">
                          <td className="py-1.5 pr-2 text-slate-200">{e.label}</td>
                          <td className="py-1.5 pr-2 text-right font-mono font-bold text-white">{fmtRank(key, e.value)}</td>
                          <td className="py-1.5 pr-2 text-right font-mono text-muted">{fmtRank(key, e.avg)}</td>
                          <td className="py-1.5 text-right font-mono text-slate-200">{rankText(e)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          ) : null}

          {/* ------------------------------------------------ whole match */}
          <Panel title="His whole match" subtitle="Cumulative total over the match. The green line is the clock above; click the chart to jump there." right={null}>
            <div className="mb-2 flex flex-wrap gap-1" role="group" aria-label="Chart measure">
              {METRICS.map((m) => (
                <button key={m.key} type="button" aria-pressed={metric === m.key} onClick={() => setMetric(m.key)} className={`min-h-9 rounded-lg px-3 text-xs font-semibold ${metric === m.key ? 'bg-lime text-ink' : 'border border-line text-slate-200 hover:bg-white/[0.05]'}`}>
                  {m.label}
                </button>
              ))}
            </div>
            {!timeline ? (
              <Spinner label="Loading his match…" />
            ) : (
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chartData}
                    margin={{ top: 6, right: 12, bottom: 16, left: 0 }}
                    onClick={(s: unknown) => {
                      const label = (s as { activeLabel?: number } | null)?.activeLabel;
                      if (typeof label === 'number') {
                        setPlaying(false);
                        setSec(Math.min(total, Math.max(0, label * 60)));
                      }
                    }}
                  >
                    <CartesianGrid stroke="#2b3339" strokeDasharray="3 3" />
                    <XAxis dataKey="k" type="number" domain={[0, Math.ceil(total / 60)]} tick={AXIS} stroke="#2b3339" tickFormatter={(v: number) => `${Math.round(v)}`} label={{ value: 'minutes of play', position: 'insideBottom', offset: -10, fill: '#93a0a8', fontSize: 11 }} />
                    <YAxis tick={AXIS} stroke="#2b3339" width={48} tickFormatter={(v: number) => (metricInfo?.key === 'value' ? v.toFixed(1) : v.toFixed(metricInfo?.key === 'distance' ? 1 : 0))} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(v) => `${Number(v).toFixed(0)} min of play`} formatter={(v) => (typeof v === 'number' ? `${fmt(v, 2)} ${metricInfo?.unit ?? ''}` : String(v))} />
                    {periodEnds.slice(0, -1).map((end) => (
                      <ReferenceLine key={end} x={end / 60} stroke="#475569" strokeDasharray="4 4" label={{ value: 'half time', fill: '#93a0a8', fontSize: 10, position: 'top' }} />
                    ))}
                    {metricInfo?.key === 'value' ? <ReferenceLine y={0} stroke="#93a0a8" strokeDasharray="4 4" /> : null}
                    <ReferenceLine x={currentK} stroke="#c8f45a" strokeWidth={2} />
                    <Line type="monotone" dataKey="v" name={metricInfo?.label ?? ''} stroke="#38bdf8" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {timeline && timeline.moments.length ? (
              <div className="mt-3">
                <div className="mb-1 text-xs font-semibold text-slate-200">Key moments (click to jump there)</div>
                <div className="flex flex-wrap gap-1.5">
                  {timeline.moments.map((m, i) => (
                    <button key={i} type="button" onClick={() => seekFrame(m.frame_num)} className={`min-h-8 rounded-lg border px-2 text-xs font-semibold hover:bg-white/[0.05] ${MOMENT_STYLE[m.tone] ?? 'border-line text-slate-300'}`}>
                      {m.clock} · {m.label}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </Panel>

          {/* ------------------------------------------------ both teams */}
          {roster ? (
            <Panel title="Both teams at this moment" subtitle="For context when you judge a player." right={null}>
              <div className="grid gap-2 sm:grid-cols-2">
                {(['home', 'away'] as const).map((k) => {
                  const t = roster.teams[k];
                  return (
                    <div key={k} className="rounded-xl border border-line bg-ink/60 p-3">
                      <div className="mb-1 flex items-center gap-2 text-sm font-bold text-white">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ background: k === 'home' ? homeColor : awayColor }} aria-hidden="true" />
                        {t.name ?? k}
                      </div>
                      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-slate-300">
                        <dt>Passes</dt>
                        <dd className="text-right font-mono text-white">{t.passes_ok}/{t.passes} ({pct(t.pass_acc)})</dd>
                        <dt>Shots (on target)</dt>
                        <dd className="text-right font-mono text-white">{t.shots} ({t.shots_on_target})</dd>
                        <dt>Goals · xG</dt>
                        <dd className="text-right font-mono text-white">{t.goals} · {t.xg.toFixed(2)}</dd>
                        <dt>Distance (team)</dt>
                        <dd className="text-right font-mono text-white">{(t.distance_m / 1000).toFixed(1)} km</dd>
                      </dl>
                    </div>
                  );
                })}
              </div>
            </Panel>
          ) : null}
        </div>
      </div>
    </div>
  );
}

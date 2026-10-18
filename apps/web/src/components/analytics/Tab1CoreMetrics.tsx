import { ChevronLeft, ChevronRight, FastForward, Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useUiStore } from '../../store';
import { AnalyticsPitch, type PitchLayers } from './AnalyticsPitch';
import {
  errorMessage,
  fmt,
  getJson,
  mmss,
  signed,
  withMatch,
  type ActionRow,
  type EpvPoint,
  type FramePayload,
  type LivePeriod,
  type MatchInfo,
  type PeriodInfo
} from './api';
import { GUIDE, MetricGuide, ObjectiveCard, type LayerKey } from './metricGuide';
import { ErrorBox, Panel, Spinner, Stat } from './ui';

const LAYERS = GUIDE.map((g) => ({ key: g.key, label: g.name, hint: g.question, dot: g.dot }));

/** playback speeds, as multiples of real time */
const SPEEDS = [0.5, 1, 2, 5, 10];
const WINDOW_S = 20;
/** a frame is drawn every BEAT_MS while playing; players glide between two frames so the motion looks continuous */
const BEAT_MS = 400;
/** the "Computing" badge appears only when a frame takes longer than this (a quick reply never flashes it) */
const SLOW_MS = 350;
const FRAME_CACHE = 160;

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const EpvChart = memo(function EpvChart({
  points,
  info,
  currentMin,
  onSeek
}: {
  points: EpvPoint[];
  info: MatchInfo;
  currentMin: number;
  onSeek: (periodIdx: number, t: number) => void;
}) {
  const offsets = useMemo(() => {
    const o = new Map<number, number>();
    let acc = 0;
    for (const p of info.periods) {
      o.set(p.period, acc);
      acc += p.duration_s;
    }
    return o;
  }, [info.periods]);
  const data = useMemo(
    () => points.map((p) => ({ x: ((offsets.get(p.period) ?? 0) + p.time_s) / 60, epv: p.epv_home })),
    [points, offsets]
  );
  const handleClick = (state: unknown) => {
    const label = (state as { activeLabel?: number } | null)?.activeLabel;
    if (typeof label !== 'number') return;
    const sec = label * 60;
    let acc = 0;
    for (let i = 0; i < info.periods.length; i++) {
      const p = info.periods[i];
      if (!p) continue;
      if (sec <= acc + p.duration_s || i === info.periods.length - 1) {
        onSeek(i, Math.max(0, Math.min(p.duration_s, sec - acc)));
        return;
      }
      acc += p.duration_s;
    }
  };
  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} onClick={handleClick} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#2b3339" strokeDasharray="3 3" />
          <XAxis dataKey="x" type="number" domain={[0, 'dataMax']} tick={{ fill: '#93a0a8', fontSize: 11 }} tickFormatter={(v: number) => `${Math.round(v)}'`} stroke="#2b3339" />
          <YAxis tick={{ fill: '#93a0a8', fontSize: 11 }} stroke="#2b3339" width={52} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ background: '#14191d', border: '1px solid #2b3339', borderRadius: 8, fontSize: 12 }}
            labelFormatter={(v) => `${Number(v).toFixed(1)} min played`}
            formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : String(v))}
          />
          <ReferenceLine y={0} stroke="#93a0a8" strokeDasharray="4 4" />
          <ReferenceLine x={currentMin} stroke="#c8f45a" strokeWidth={2} />
          <Line type="monotone" dataKey="epv" name="EPV (home)" stroke="#38bdf8" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});

function Inner({ info }: { info: MatchInfo }) {
  const homeColor = useUiStore((s) => s.homeColor);
  const homeText = useUiStore((s) => s.homeTextColor);
  const awayColor = useUiStore((s) => s.awayColor);
  const awayText = useUiStore((s) => s.awayTextColor);
  const showNumbers = useUiStore((s) => s.showNumbers);

  const [periodIdx, setPeriodIdx] = useState(0);
  const [t, setT] = useState(300);
  const [frame, setFrame] = useState<FramePayload | null>(null);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [skipDead, setSkipDead] = useState(true);
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({ pc: true, das: false, obso: false, epv: true, vaep: false, obv: false });
  const [allActions, setAllActions] = useState<ActionRow[] | null>(null);
  const [epv, setEpv] = useState<EpvPoint[]>([]);
  const [live, setLive] = useState<LivePeriod[] | null>(null);

  const period: PeriodInfo = info.periods[periodIdx] ?? (info.periods[0] as PeriodInfo);
  const maxT = Math.max(1, (period.last_frame - period.first_frame) / info.fps);
  const tClamped = Math.min(Math.max(0, t), maxT);
  const frameNum = Math.min(period.last_frame, period.first_frame + Math.round(tClamped * info.fps));
  const homeName = info.teams.home.name ?? 'Home';
  const awayName = info.teams.away.name ?? 'Away';
  const windowFrames = Math.round(WINDOW_S * info.fps);
  const tRef = useRef(t);
  tRef.current = t;
  const seekToRef = useRef<number | null>(null);

  // ---- live play = the stretches where the ball is tracked ----
  useEffect(() => {
    const ctrl = new AbortController();
    getJson<{ periods: LivePeriod[] }>(withMatch('/api/metrics/live-segments', info.match_id), ctrl.signal)
      .then((r) => setLive(r.periods))
      .catch(() => {
        if (!ctrl.signal.aborted) setLive([]); // older server without this endpoint: everything works, only the strip is missing
      });
    return () => ctrl.abort();
  }, [info.match_id]);
  const liveSegs = useMemo<[number, number][]>(() => live?.find((p) => p.period === period.period)?.segments ?? [], [live, period.period]);

  // first time the segments arrive: start on live play instead of a blank dead-ball frame
  const startedOnLive = useRef(false);
  useEffect(() => {
    if (startedOnLive.current || !live) return;
    startedOnLive.current = true;
    const segs = live.find((p) => p.period === period.period)?.segments ?? [];
    const now = tRef.current;
    if (segs.length && !segs.some(([a, z]) => now >= a && now <= z)) {
      const nx = segs.find(([a]) => a > now) ?? segs[0];
      if (nx) setT(nx[0]);
    }
  }, [live, period.period]);

  // ---- frames: one cache, one request per frame, newest wish wins (nothing queues up while scrubbing) ----
  const cacheRef = useRef(new Map<string, FramePayload>());
  const inflightRef = useRef(new Map<string, Promise<FramePayload>>());
  const fetchFrame = useCallback(
    (fn: number): Promise<FramePayload> => {
      const key = `${info.match_id}:${fn}`;
      const hit = cacheRef.current.get(key);
      if (hit) return Promise.resolve(hit);
      const running = inflightRef.current.get(key);
      if (running) return running;
      const pr = getJson<FramePayload>(withMatch(`/api/metrics/frame/${fn}`, info.match_id))
        .then((f) => {
          cacheRef.current.set(key, f);
          while (cacheRef.current.size > FRAME_CACHE) {
            const oldest = cacheRef.current.keys().next().value;
            if (oldest === undefined) break;
            cacheRef.current.delete(oldest);
          }
          return f;
        })
        .finally(() => {
          inflightRef.current.delete(key);
        });
      inflightRef.current.set(key, pr);
      return pr;
    },
    [info.match_id]
  );

  const wantRef = useRef<number | null>(null);
  const pumpingRef = useRef(false);
  const show = useCallback(
    async (fn: number) => {
      wantRef.current = fn;
      if (pumpingRef.current) return;
      pumpingRef.current = true;
      const timer = window.setTimeout(() => setSlow(true), SLOW_MS);
      try {
        while (wantRef.current !== null) {
          const target: number = wantRef.current;
          const f: FramePayload = await fetchFrame(target);
          if (wantRef.current === target) {
            wantRef.current = null;
            setFrame(f);
            setError(null);
          }
        }
      } catch (e) {
        wantRef.current = null;
        setError(errorMessage(e));
      } finally {
        window.clearTimeout(timer);
        pumpingRef.current = false;
        setSlow(false);
      }
    },
    [fetchFrame]
  );

  // paused / scrubbing / stepping: show the chosen frame
  useEffect(() => {
    if (playing) return;
    const id = window.setTimeout(() => void show(frameNum), 40);
    return () => window.clearTimeout(id);
  }, [frameNum, playing, show]);

  // ---- playing: a steady beat, frames fetched ahead of time, dead ball skipped on request ----
  useEffect(() => {
    if (!playing) return;
    let alive = true;
    const keyStep = Math.min(4, Math.max(0.25, 0.4 * speed)); // seconds of match between two drawn frames
    const frameAt = (tt: number) => Math.min(period.last_frame, period.first_frame + Math.round(tt * info.fps));
    const nextTime = (from: number) => {
      let nt = from + keyStep;
      if (skipDead && liveSegs.length && !liveSegs.some(([a, z]) => nt >= a && nt <= z)) {
        const nx = liveSegs.find(([a]) => a > nt);
        nt = nx ? nx[0] : maxT;
      }
      return nt;
    };
    (async () => {
      let tCur = tRef.current;
      let due = performance.now();
      try {
        while (alive) {
          if (seekToRef.current !== null) { // somebody clicked the timeline while we were playing
            tCur = seekToRef.current;
            seekToRef.current = null;
            due = performance.now();
          }
          const nt = nextTime(tCur);
          if (nt >= maxT) {
            setT(maxT);
            setPlaying(false);
            return;
          }
          const pending = fetchFrame(frameAt(nt));
          due += BEAT_MS;
          const wait = due - performance.now();
          if (wait > 0) await sleep(wait);
          const timer = window.setTimeout(() => alive && setSlow(true), SLOW_MS); // only counts the time we are really kept waiting
          let f: FramePayload;
          try {
            f = await pending;
          } finally {
            window.clearTimeout(timer);
            setSlow(false);
          }
          if (!alive) return;
          if (performance.now() - due > BEAT_MS * 2) due = performance.now(); // after a stall do not rush to catch up
          tCur = nt;
          setT(nt);
          setFrame(f);
          const n1 = nextTime(nt);
          const n2 = nextTime(n1);
          for (const n of [n1, n2]) if (n < maxT) fetchFrame(frameAt(n)).catch(() => undefined);
        }
      } catch (e) {
        if (alive) {
          setError(errorMessage(e));
          setPlaying(false);
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, [playing, speed, skipDead, liveSegs, period, maxT, info.fps, fetchFrame]);

  // ---- EPV timeline: once ----
  useEffect(() => {
    const ctrl = new AbortController();
    getJson<{ points: EpvPoint[] }>(withMatch('/api/metrics/epv-timeline?step_s=3', info.match_id), ctrl.signal)
      .then((r) => setEpv(r.points))
      .catch(() => undefined);
    return () => ctrl.abort();
  }, [info.match_id]);

  // ---- actions (VAEP / OBV): once, when first needed ----
  useEffect(() => {
    if (!(layers.vaep || layers.obv) || allActions !== null) return;
    const ctrl = new AbortController();
    getJson<{ actions: ActionRow[] }>(withMatch('/api/metrics/actions?limit=10000', info.match_id), ctrl.signal)
      .then((r) => setAllActions(r.actions))
      .catch(() => {
        if (!ctrl.signal.aborted) setError('Could not load the action values (VAEP / OBV).');
      });
    return () => ctrl.abort();
  }, [layers.vaep, layers.obv, allActions, info.match_id]);

  const displayedFrameNum = frame?.frame_num ?? frameNum;
  const recent = useMemo(
    () =>
      (allActions ?? [])
        .filter((a) => a.period === period.period && a.frame_num <= displayedFrameNum && displayedFrameNum - a.frame_num <= windowFrames)
        .sort((a, b) => b.frame_num - a.frame_num),
    [allActions, period.period, displayedFrameNum, windowFrames]
  );

  const offsetSec = useMemo(() => info.periods.slice(0, periodIdx).reduce((s, p) => s + p.duration_s, 0), [info.periods, periodIdx]);
  const currentMin = Math.round((offsetSec + tClamped) / 3) / 20; // marker position, moved every 3 s (like the chart's own samples)

  const pitchLayers = useMemo<PitchLayers>(() => ({ pc: layers.pc, das: layers.das, obso: layers.obso, vaep: layers.vaep }), [layers.pc, layers.das, layers.obso, layers.vaep]);
  const toggle = (k: LayerKey) => setLayers((prev) => ({ ...prev, [k]: !prev[k] }));
  const seek = useCallback((pi: number, tt: number) => {
    seekToRef.current = tt;
    setPeriodIdx(pi);
    setT(tt);
  }, []);
  const step = (dt: number) => {
    setPlaying(false);
    setT((prev) => Math.min(maxT, Math.max(0, prev + dt)));
  };
  const jumpNextLive = () => {
    setPlaying(false);
    const nx = liveSegs.find(([a]) => a > tClamped + 0.5);
    if (nx) {
      setT(nx[0]);
      return;
    }
    const nextIdx = periodIdx + 1;
    const nextP = info.periods[nextIdx];
    if (nextP) {
      const first = live?.find((p) => p.period === nextP.period)?.segments[0];
      seek(nextIdx, first ? first[0] : 0);
    }
  };

  // ---- what the numbers say, in words ----
  const ballOk = frame ? (frame.ball_tracked ?? frame.ball !== null) : true;
  const possKnown = frame ? frame.possession_known !== false : true;
  const attackingName = frame ? (frame.attacking_side === 'home' ? homeName : awayName) : '–';
  const attackDir = period.attack_dir;
  const topObso = frame
    ? [...frame.players]
        .filter((p) => p.obso !== undefined)
        .sort((a, b) => (b.obso ?? 0) - (a.obso ?? 0))
        .slice(0, 5)
    : [];
  const halfLen = info.pitch.length / 2;
  const attackerShare = frame ? (frame.attacking_side === 'home' ? frame.metrics.pc_home_share : 1 - frame.metrics.pc_home_share) : 0;
  const caption = (() => {
    if (!frame) return 'Loading the first frame…';
    if (!ballOk) {
      return `Dead ball: the ball is not tracked, so there is no one "in attack". Pitch control is still drawn from the player positions (${homeName} controls ${(frame.metrics.pc_home_share * 100).toFixed(0)}% of the pitch). DAS, OBSO and EPV need the ball and are blank.`;
    }
    if (!possKnown) return 'Ball in play before the first touch of the match, so nobody is attacking yet.';
    const parts = [`${attackingName} have the ball and control ${(attackerShare * 100).toFixed(0)}% of the pitch.`];
    if (frame.metrics.epv_ball !== null) parts.push(`The ball is worth about a ${(frame.metrics.epv_ball * 100).toFixed(1)}% chance of scoring next (EPV).`);
    if (frame.metrics.das_value !== null) parts.push(`${frame.metrics.das_value.toFixed(2)} m² of dangerous space is reachable by a pass (DAS).`);
    const top = topObso[0];
    if (top && (top.obso ?? 0) > 0) parts.push(`Most valuable free attacker: #${top.jersey} ${top.name} (OBSO ${(top.obso ?? 0).toFixed(3)}).`);
    return parts.join(' ');
  })();

  return (
    <div className="space-y-4">
      <ObjectiveCard />

      {/* metric toggles */}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Metric layers">
        {LAYERS.map((l) => (
          <button
            key={l.key}
            type="button"
            title={l.hint}
            aria-pressed={layers[l.key]}
            onClick={() => toggle(l.key)}
            className={`flex min-h-10 items-center gap-2 rounded-xl border px-3.5 text-sm font-semibold transition ${
              layers[l.key] ? 'border-lime bg-lime/15 text-white' : 'border-line bg-panel text-muted hover:text-white'
            }`}
          >
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: l.dot }} aria-hidden="true" />
            {l.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 space-y-3">
          {/* live / dead ball status (fixed height, so nothing jumps when it changes) */}
          <div className="flex h-8 items-center gap-2 text-xs" aria-live="polite">
            {frame ? (
              ballOk ? (
                <span className="flex items-center gap-1.5 rounded-lg bg-lime/15 px-2.5 py-1 font-semibold text-lime"><span className="h-2 w-2 rounded-full bg-lime" aria-hidden="true" />Live play · ball tracked</span>
              ) : (
                <span className="flex items-center gap-1.5 rounded-lg bg-amber-400/15 px-2.5 py-1 font-semibold text-amber-300"><span className="h-2 w-2 rounded-full bg-amber-300" aria-hidden="true" />Dead ball · ball not tracked, so DAS / OBSO / EPV are blank</span>
              )
            ) : null}
            <button
              type="button"
              onClick={jumpNextLive}
              disabled={!live || (!liveSegs.length && periodIdx >= info.periods.length - 1)}
              className="ml-auto flex h-7 items-center gap-1.5 rounded-lg border border-line px-2.5 font-semibold text-muted hover:text-white disabled:opacity-40"
            >
              <FastForward className="h-3.5 w-3.5" aria-hidden="true" />
              Next live play
            </button>
          </div>

          <div className="relative">
            <AnalyticsPitch
              frame={frame}
              layers={pitchLayers}
              actions={recent}
              windowFrames={windowFrames}
              homeColor={homeColor}
              awayColor={awayColor}
              homeText={homeText}
              awayText={awayText}
              showNumbers={showNumbers}
              animateMs={playing && speed <= 2 ? BEAT_MS : 0}
              length={info.pitch.length}
              width={info.pitch.width}
            />
            {slow ? (
              <div className="absolute right-3 top-3 rounded-lg bg-black/60 px-2.5 py-1.5">
                <Spinner label={playing ? 'Buffering…' : 'Computing…'} />
              </div>
            ) : null}
          </div>

          <p className="min-h-[3.25rem] rounded-xl border border-line bg-panel px-3 py-2 text-xs leading-relaxed text-slate-200" aria-live="polite">
            <span className="mr-1.5 font-bold uppercase tracking-wider text-lime">In words</span>
            {caption}
          </p>

          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
            <span>
              <span className="font-semibold" style={{ color: homeColor }}>{homeName}</span> attacks {attackDir.home > 0 ? '→' : '←'} ·{' '}
              <span className="font-semibold text-slate-100">{awayName}</span> attacks {attackDir.away > 0 ? '→' : '←'}
            </span>
            {frame ? <span>frame {frame.frame_num} · computed in {frame.compute_ms ?? '–'} ms</span> : null}
          </div>

          {/* transport + slider */}
          <div className="rounded-2xl border border-line bg-panel p-3">
            <div className="flex flex-wrap items-center gap-2">
              {info.periods.map((p, i) => (
                <button
                  key={p.period}
                  type="button"
                  aria-pressed={i === periodIdx}
                  onClick={() => seek(i, 0)}
                  className={`min-h-9 rounded-lg border px-3 text-xs font-bold ${i === periodIdx ? 'border-lime bg-lime/15 text-white' : 'border-line text-muted hover:text-white'}`}
                >
                  {p.period === 1 ? '1st half' : p.period === 2 ? '2nd half' : `Period ${p.period}`}
                </button>
              ))}
              <div className="ml-auto flex items-center gap-1">
                <button type="button" aria-label="Back 10 seconds" onClick={() => step(-10)} className="grid h-9 w-9 place-items-center rounded-lg border border-line text-muted hover:text-white"><SkipBack className="h-4 w-4" aria-hidden="true" /></button>
                <button type="button" aria-label="Back 1 second" onClick={() => step(-1)} className="grid h-9 w-9 place-items-center rounded-lg border border-line text-muted hover:text-white"><ChevronLeft className="h-4 w-4" aria-hidden="true" /></button>
                <button type="button" aria-label={playing ? 'Pause' : 'Play'} onClick={() => setPlaying((p) => !p)} className="grid h-9 w-11 place-items-center rounded-lg bg-lime text-ink">{playing ? <Pause className="h-4 w-4" aria-hidden="true" /> : <Play className="h-4 w-4" aria-hidden="true" />}</button>
                <button type="button" aria-label="Forward 1 second" onClick={() => step(1)} className="grid h-9 w-9 place-items-center rounded-lg border border-line text-muted hover:text-white"><ChevronRight className="h-4 w-4" aria-hidden="true" /></button>
                <button type="button" aria-label="Forward 10 seconds" onClick={() => step(10)} className="grid h-9 w-9 place-items-center rounded-lg border border-line text-muted hover:text-white"><SkipForward className="h-4 w-4" aria-hidden="true" /></button>
                <select aria-label="Playback speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="ml-1 h-9 rounded-lg border border-line bg-ink px-2 text-xs text-white">
                  {SPEEDS.map((s) => (<option key={s} value={s}>{s}× real time</option>))}
                </select>
              </div>
            </div>
            <div className="mt-3 flex items-center gap-3">
              <span className="w-12 font-mono text-sm tabular-nums text-white">{mmss(tClamped)}</span>
              <div className="min-w-0 flex-1">
                <input
                  type="range"
                  aria-label="Match time"
                  min={0}
                  max={Math.floor(maxT)}
                  step={1}
                  value={Math.round(tClamped)}
                  onChange={(e) => {
                    setPlaying(false);
                    setT(Number(e.target.value));
                  }}
                  className="h-2 w-full cursor-pointer accent-lime"
                />
                {/* green = ball tracked (live play), dark = dead ball */}
                <div className="relative mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white/10" aria-hidden="true" title="Green: live play (ball tracked). Dark: dead ball.">
                  {liveSegs.map(([a, z]) => (
                    <div key={`${a}-${z}`} className="absolute top-0 h-full bg-lime/70" style={{ left: `${(a / maxT) * 100}%`, width: `${Math.max(0.3, ((z - a) / maxT) * 100)}%` }} />
                  ))}
                  <div className="absolute top-0 h-full w-0.5 bg-white" style={{ left: `${(tClamped / maxT) * 100}%` }} />
                </div>
              </div>
              <span className="w-12 text-right font-mono text-sm tabular-nums text-muted">{mmss(maxT)}</span>
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted">
              <span>
                Green strip = live play{live ? ` · ${mmss(live.find((p) => p.period === period.period)?.live_s ?? 0)} of ${mmss(maxT)} in this half` : ''}
              </span>
              <label className="flex cursor-pointer items-center gap-1.5">
                <input type="checkbox" checked={skipDead} onChange={(e) => setSkipDead(e.target.checked)} className="accent-lime" />
                Skip dead ball while playing
              </label>
            </div>
          </div>

          {layers.epv ? (
            <Panel title="EPV progression" subtitle="Value of the ball location · above 0 = home has the more valuable possession · click to jump" right={null}>
              {epv.length ? <EpvChart points={epv} info={info} currentMin={currentMin} onSeek={seek} /> : <Spinner label="Loading EPV timeline…" />}
            </Panel>
          ) : null}
        </div>

        {/* side panel */}
        <aside className="space-y-3" aria-label="Frame metrics">
          {error ? <ErrorBox message={error} /> : null}
          <div className="grid grid-cols-2 gap-2">
            <Stat
              label="Possession"
              value={!frame ? '–' : !ballOk ? 'Dead ball' : possKnown ? attackingName : '–'}
              hint={!frame ? '' : !ballOk ? (possKnown ? `last on-ball action: ${attackingName}` : 'no on-ball action yet') : 'team of the last on-ball action'}
              help="Which team made the last on-ball action (pass, shot, tackle…). Shown as 'Dead ball' while the ball is not tracked."
            />
            <Stat
              label="Pitch control"
              value={frame ? `${(frame.metrics.pc_home_share * 100).toFixed(0)}%` : '–'}
              hint={`${homeName} share of the pitch`}
              help="Share of the pitch where this team would reach the ball first. Computed from player positions, so it exists even when the ball is not tracked."
            />
            <Stat
              label="DAS"
              value={ballOk ? fmt(frame?.metrics.das_value, 2) : '–'}
              hint={!ballOk ? 'needs the ball (dead ball)' : frame?.das_source && !frame.das_source.startsWith('accessible-space') ? `danger-weighted m² · ${frame.das_source}` : 'danger-weighted m² · accessible-space'}
              help="Dangerous accessible space: how much dangerous space the attackers can reach with a pass before the defenders. Higher = more dangerous passes are open."
            />
            <Stat
              label="OBSO"
              value={ballOk ? fmt(frame?.metrics.obso_total, 2) : '–'}
              hint={ballOk ? 'team total · C-OBSO' : 'needs the ball (dead ball)'}
              help="Off-ball scoring opportunity: how much scoring value the attackers' positions offer (not a probability; compare within the match)."
            />
            <Stat
              label="EPV at ball"
              value={ballOk ? fmt(frame?.metrics.epv_ball, 4) : '–'}
              hint={ballOk ? 'chance of scoring next from here' : 'needs the ball (dead ball)'}
              help="Expected possession value: roughly the chance that the team in possession scores next, given only where the ball is. 0.0100 = about 1 %."
            />
            <Stat
              label="Back line"
              value={frame && possKnown ? `${Math.max(0, halfLen - frame.metrics.defensive_line_x).toFixed(1)} m` : '–'}
              hint="defenders' second-last man, from their own goal"
              help="Distance of the defending team's offside line (second-last defender, goalkeeper included) from the goal they are defending. Small = deep defence, large = high line."
            />
          </div>

          {layers.obso ? (
            <Panel title="Top off-ball threats" subtitle="Attackers without the ball, ranked by the value of the spot they stand on" right={null}>
              {topObso.length ? (
                <ul className="space-y-1.5 text-sm">
                  {topObso.map((p) => (
                    <li key={p.slot} className="flex items-center justify-between gap-2">
                      <span className="truncate text-slate-100">#{p.jersey} {p.name}</span>
                      <span className="font-mono text-yellow-300">{fmt(p.obso, 3)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-muted">{ballOk ? 'No off-ball attacker data for this frame.' : 'Dead ball: OBSO needs the ball position. Press "Next live play".'}</p>
              )}
            </Panel>
          ) : null}

          {layers.vaep || layers.obv ? (
            <Panel title="Recent actions" subtitle={`last ${WINDOW_S} s · VAEP = what really happened, OBV = what was expected (luck removed)`} right={null}>
              {allActions === null ? (
                <Spinner label="Loading actions…" />
              ) : recent.length ? (
                <div className="max-h-72 overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-muted">
                      <tr><th className="py-1 pr-2 font-semibold">Player</th><th className="pr-2 font-semibold">Action</th>{layers.vaep ? <th className="pr-2 text-right font-semibold">VAEP</th> : null}{layers.obv ? <th className="text-right font-semibold">OBV</th> : null}</tr>
                    </thead>
                    <tbody>
                      {recent.slice(0, 12).map((a, i) => (
                        <tr key={`${a.frame_num}-${i}`} className="border-t border-line/60">
                          <td className="max-w-[110px] truncate py-1 pr-2 text-slate-100">{a.jersey ? `#${a.jersey} ` : ''}{a.player ?? '?'}</td>
                          <td className="pr-2 text-muted">{a.action}{a.success ? '' : ' ✗'}</td>
                          {layers.vaep ? <td className={`pr-2 text-right font-mono ${a.vaep >= 0 ? 'text-green-400' : 'text-red-400'}`}>{signed(a.vaep)}</td> : null}
                          {layers.obv ? <td className={`text-right font-mono ${a.obv >= 0 ? 'text-green-400' : 'text-red-400'}`}>{signed(a.obv)}</td> : null}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-xs text-muted">No on-ball actions in the last {WINDOW_S} seconds.</p>
              )}
            </Panel>
          ) : null}

          {layers.pc ? (
            <div className="flex items-center gap-2 text-[11px] text-muted">
              <span className="h-2.5 w-6 rounded" style={{ background: `linear-gradient(90deg, ${awayColor}, transparent, ${homeColor})` }} aria-hidden="true" />
              {awayName} control ← → {homeName} control
            </div>
          ) : null}

          <MetricGuide active={layers} />
        </aside>
      </div>
    </div>
  );
}

export function Tab1CoreMetrics({ info }: { info: MatchInfo }) {
  if (!info.periods.length) return <ErrorBox message="This match has no playable periods." />;
  return <Inner info={info} />;
}

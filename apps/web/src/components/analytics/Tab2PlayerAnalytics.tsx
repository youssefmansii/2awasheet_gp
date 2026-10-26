import { useEffect, useMemo, useRef, useState } from 'react';
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  errorMessage,
  fmt,
  getJson,
  isAbort,
  pct,
  signed,
  withMatch,
  type AnalyticsStatus,
  type Chemistry,
  type FatigueBin,
  type MatchInfo,
  type PlayerListItem,
  type PlayerReport,
  type PlayersResponse
} from './api';
import { ErrorBox, Panel, Spinner, Stat } from './ui';

const AXIS = { fill: '#93a0a8', fontSize: 11 };
const TOOLTIP_STYLE = { background: '#14191d', border: '1px solid #2b3339', borderRadius: 8, fontSize: 12 };
const tipFmt = (v: unknown) => (typeof v === 'number' ? v.toFixed(3) : String(v));


// ---------------------------------------------------------------------------
/** "What is this?" box shown at the top of every section, in plain English. */
function Guide({ what, read }: { what: string; read: string }) {
  return (
    <div className="mb-3 rounded-xl border border-line bg-ink/50 px-3 py-2 text-xs leading-relaxed text-slate-300">
      <p><span className="font-bold text-slate-100">What it is. </span>{what}</p>
      <p className="mt-1"><span className="font-bold text-slate-100">How to read it. </span>{read}</p>
    </div>
  );
}

/** One sentence that says what the numbers mean for THIS player. */
function InWords({ text, tone = 'info' }: { text: string; tone?: 'good' | 'poor' | 'info' }) {
  const dot = tone === 'good' ? 'bg-emerald-400' : tone === 'poor' ? 'bg-red-400' : 'bg-lime';
  return (
    <p className="mb-3 flex gap-2 rounded-xl border border-lime/30 bg-lime/5 px-3 py-2 text-sm leading-snug text-slate-100" role="note">
      <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${dot}`} aria-hidden="true" />
      <span><span className="font-bold text-white">In words: </span>{text}</span>
    </p>
  );
}

function pressWords(p: PlayerReport['press_resistance'] | undefined): { text: string; tone: 'good' | 'poor' | 'info' } {
  if (!p || !p.n_actions) return { text: 'He had no on-ball actions with a valid ball position, so pressure cannot be judged.', tone: 'info' };
  const r = p.radius_m ?? 3;
  if (!p.n_pressured) return { text: `None of his ${p.n_actions} actions came with an opponent within ${r} m, so he was never really closed down.`, tone: 'info' };
  const under = p.success_rate_under_pressure;
  const free = p.success_rate_free;
  const base = `An opponent was within ${r} m on ${p.n_pressured} of his ${p.n_actions} actions (${pct(p.pressured_share)}).`;
  if (under === null || under === undefined || free === null || free === undefined) return { text: base, tone: 'info' };
  const drop = (free - under) * 100;
  if (drop <= 3) return { text: `${base} He succeeded ${pct(under)} of the time under pressure and ${pct(free)} when free: pressure hardly affects him.`, tone: 'good' };
  if (drop >= 15) return { text: `${base} He succeeded ${pct(under)} of the time under pressure but ${pct(free)} when free: he loses a lot when closed down.`, tone: 'poor' };
  return { text: `${base} He succeeded ${pct(under)} of the time under pressure and ${pct(free)} when free: a normal drop of ${drop.toFixed(0)} points.`, tone: 'info' };
}

function decisionWords(s: PlayerReport['decision_efficiency']['summary']): { text: string; tone: 'good' | 'poor' | 'info' } {
  const n = s.n_scored ?? s.n_passes;
  if (!n || s.mean_decision_score === null) return { text: 'Too few passes with at least three team-mates to choose from, so his decisions cannot be scored.', tone: 'info' };
  const sc = s.mean_decision_score;
  const level = sc >= 0.7 ? 'usually picks one of the best options' : sc >= 0.5 ? 'usually picks a good option, but not always the best' : 'often picks a weaker option than the best one available';
  return {
    text: `Over ${n} scored passes he ${level} (score ${sc.toFixed(2)}: 1 = always the best option, 0.5 = typical, 0 = always the worst). He picked the very best option ${pct(s.optimal_choice_rate)} of the time.`,
    tone: sc >= 0.7 ? 'good' : sc < 0.45 ? 'poor' : 'info'
  };
}

function fatigueWords(bins: FatigueBin[]): { text: string; tone: 'good' | 'poor' | 'info' } {
  const solid = bins.filter((b) => b.radius_m !== null && b.minutes_on_pitch >= 10);
  const first = solid[0];
  const last = solid[solid.length - 1];
  if (!first || !last || first === last || first.radius_m === null || last.radius_m === null) return { text: 'He played fewer than two full 10-minute blocks in live play, so a trend cannot be shown.', tone: 'info' };
  const ch = (100 * (last.radius_m / first.radius_m - 1));
  if (ch <= -10) return { text: `In ${last.label} the area he controls was ${Math.abs(ch).toFixed(0)}% smaller than in ${first.label}: he covered less space late on, which can be a sign of tiring (or of a change in his job).`, tone: 'poor' };
  if (ch >= 10) return { text: `In ${last.label} the area he controls was ${ch.toFixed(0)}% larger than in ${first.label}: he covered more space late on, with no sign of fatigue.`, tone: 'good' };
  return { text: `The area he controls is about the same in ${last.label} as in ${first.label} (${signed(ch, 0)}%): no clear sign of fatigue.`, tone: 'good' };
}

function chemistryWords(c: Chemistry, slot: number): string {
  const mine = c.top_duos.filter((d) => d.passer_slot === slot || d.receiver_slot === slot);
  const d = mine[0];
  if (!d) return 'No pair with this player reached the minimum number of completed passes, so no link is shown.';
  return `His best link is ${d.passer} → ${d.receiver} (${d.passes} passes): passes between them add ${signed(d.uplift, 3)} of scoring chance on average compared with a typical pass from the same zone.`;
}

function decoyWords(d: PlayerReport['decoy_run_index'] | undefined): { text: string; tone: 'good' | 'poor' | 'info' } {
  if (!d || !d.n_decoy_runs) return { text: 'He made no decoy run (a run of at least one second at 19.8 km/h or more, without receiving the ball) while his team had the ball.', tone: 'info' };
  const tot = d.total_delta_das ?? 0;
  return {
    text: `He made ${d.n_decoy_runs} decoy run${d.n_decoy_runs === 1 ? '' : 's'}. After them the team had ${tot >= 0 ? 'more' : 'less'} dangerous passing space on average (${signed(d.mean_delta_das, 1)} m² per run), so his runs ${tot >= 0 ? 'helped to open space for team-mates' : 'did not create extra space'}.`,
    tone: tot > 0 ? 'good' : 'info'
  };
}

// ---------------------------------------------------------------------------
function FatigueCharts({ bins, baseline }: { bins: FatigueBin[]; baseline: string | null }) {
  const data = bins.map((b) => ({
    label: b.label,
    radius: b.radius_m,
    change: b.radius_change_pct,
    vaep: b.vaep_per_action,
    score: b.decision_score
  }));
  if (!data.length) return <p className="text-sm text-muted">No tracked minutes for this player.</p>;
  return (
    <div className="space-y-4">
      <div>
        <div className="mb-1 text-xs font-semibold text-slate-200">Tactical coverage: pitch-control radius (m) and change vs {baseline ?? 'first block'} (%)</div>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#2b3339" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={AXIS} stroke="#2b3339" />
              <YAxis yAxisId="r" tick={AXIS} stroke="#2b3339" width={40} domain={['auto', 'auto']} />
              <YAxis yAxisId="c" orientation="right" tick={AXIS} stroke="#2b3339" width={44} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={tipFmt} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar yAxisId="c" dataKey="change" name="Change vs baseline (%)" fill="#334155" isAnimationActive={false} />
              <Line yAxisId="r" type="monotone" dataKey="radius" name="Control radius (m)" stroke="#c8f45a" strokeWidth={2.5} dot={{ r: 4 }} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <div className="mb-1 text-xs font-semibold text-slate-200">Decision quality: VAEP above average per action, and decision score</div>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#2b3339" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={AXIS} stroke="#2b3339" />
              <YAxis yAxisId="v" tick={AXIS} stroke="#2b3339" width={48} tickFormatter={(v: number) => v.toFixed(2)} />
              <YAxis yAxisId="s" orientation="right" domain={[0, 1]} tick={AXIS} stroke="#2b3339" width={36} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={tipFmt} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar yAxisId="v" dataKey="vaep" name="VAEP above avg / action" fill="#38bdf8" isAnimationActive={false} />
              <Line yAxisId="s" type="monotone" dataKey="score" name="Decision score (0–1)" stroke="#f472b6" strokeWidth={2.5} dot={{ r: 4 }} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-xs">
          <thead className="text-muted">
            <tr>
              <th className="py-1 pr-3 font-semibold">Block</th>
              <th className="pr-3 text-right font-semibold">Min on pitch</th>
              <th className="pr-3 text-right font-semibold">Radius</th>
              <th className="pr-3 text-right font-semibold">Δ radius</th>
              <th className="pr-3 text-right font-semibold">Actions</th>
              <th className="pr-3 text-right font-semibold">Distance</th>
              <th className="text-right font-semibold">Avg speed</th>
            </tr>
          </thead>
          <tbody>
            {bins.map((b) => (
              <tr key={b.bin} className="border-t border-line/60">
                <td className="py-1 pr-3 font-semibold text-slate-100">{b.label}</td>
                <td className="pr-3 text-right font-mono">{b.minutes_on_pitch.toFixed(1)}</td>
                <td className="pr-3 text-right font-mono">{fmt(b.radius_m, 2)} m</td>
                <td className={`pr-3 text-right font-mono ${b.radius_change_pct !== null && b.radius_change_pct < -10 ? 'text-red-400' : ''}`}>{b.radius_change_pct === null ? '–' : `${signed(b.radius_change_pct, 1)}%`}</td>
                <td className="pr-3 text-right font-mono">{b.n_actions}</td>
                <td className="pr-3 text-right font-mono">{(b.distance_m / 1000).toFixed(2)} km</td>
                <td className="text-right font-mono">{fmt(b.mean_speed, 2)} m/s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
function ChemistryMatrix({ chem, selectedSlot }: { chem: Chemistry; selectedSlot: number }) {
  const n = chem.players.length;
  if (!n) {
    return <p className="text-sm text-muted">No player pair reached {chem.min_passes ?? 3} completed passes.</p>;
  }
  const SCALE = 0.04;
  const color = (u: number | null) => {
    if (u === null) return 'transparent';
    const t = Math.max(-1, Math.min(1, u / SCALE));
    return t >= 0 ? `rgba(74,222,128,${0.12 + 0.75 * t})` : `rgba(248,113,113,${0.12 + 0.75 * -t})`;
  };
  return (
    <div className="space-y-3">
      <div className="overflow-auto">
        <table className="border-separate border-spacing-0.5 text-[10px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-panel pr-2 text-right font-semibold text-muted">passer ↓ / receiver →</th>
              {chem.players.map((p) => (
                <th key={p.slot} title={p.name} className={`h-7 w-9 rounded font-bold ${p.slot === selectedSlot ? 'bg-lime/25 text-lime' : 'text-slate-300'}`}>
                  {p.jersey}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {chem.players.map((rowP, i) => (
              <tr key={rowP.slot}>
                <th className={`sticky left-0 max-w-[120px] truncate bg-panel pr-2 text-right font-semibold ${rowP.slot === selectedSlot ? 'text-lime' : 'text-slate-300'}`} title={rowP.name}>
                  {rowP.jersey} {rowP.name}
                </th>
                {chem.players.map((colP, j) => {
                  const u = chem.matrix[i]?.[j] ?? null;
                  const c = chem.counts[i]?.[j] ?? 0;
                  const hl = rowP.slot === selectedSlot || colP.slot === selectedSlot;
                  return (
                    <td
                      key={colP.slot}
                      title={u === null ? `${rowP.name} → ${colP.name}: fewer than ${chem.min_passes ?? 3} passes` : `${rowP.name} → ${colP.name}: ${c} passes, uplift ${u.toFixed(4)} EPV per pass`}
                      className={`h-7 w-9 rounded text-center font-mono ${hl ? 'outline outline-1 outline-lime/50' : ''}`}
                      style={{ background: color(u) }}
                    >
                      {u === null ? '' : (u * 1000).toFixed(0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
        <span className="h-2.5 w-24 rounded" style={{ background: 'linear-gradient(90deg, rgba(248,113,113,.85), transparent, rgba(74,222,128,.85))' }} aria-hidden="true" />
        cell = EPV uplift per pass × 1000 (green = better than a typical pass from that zone) · min {chem.min_passes ?? 3} passes
      </div>
      {chem.top_duos.length ? (
        <div>
          <div className="mb-1 text-xs font-semibold text-slate-200">Highest-synergy duos</div>
          <ol className="space-y-1 text-sm">
            {chem.top_duos.slice(0, 5).map((d, i) => (
              <li key={`${d.passer_slot}-${d.receiver_slot}`} className="flex items-center justify-between gap-2">
                <span className="truncate text-slate-100">{i + 1}. {d.passer} → {d.receiver} <span className="text-muted">({d.passes} passes)</span></span>
                <span className="font-mono text-green-400">{signed(d.uplift, 4)}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
function DecisionTable({ report }: { report: PlayerReport }) {
  const [onlySub, setOnlySub] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const { summary, passes } = report.decision_efficiency;
  const rows = useMemo(() => (onlySub ? passes.filter((p) => p.chosen_name !== p.best_name) : passes), [passes, onlySub]);
  const shown = showAll ? rows : rows.slice(0, 12);
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Decision score" value={fmt(summary.mean_decision_score, 2)} hint={`1 = best option, 0 = worst · ${summary.n_scored ?? summary.n_passes} scored passes`} />
        <Stat label="Best option picked" value={pct(summary.optimal_choice_rate)} hint={`${summary.n_passes} passes`} />
        <Stat label="Pass success" value={pct(summary.pass_success_rate)} hint="completed passes" />
        <Stat label="Result of his passes" value={signed(summary.mean_realised_depv, 4)} hint="average change in scoring chance per pass (ΔEPV)" />
        <Stat label="Mean regret" value={fmt(summary.mean_regret ?? null, 4)} hint="scoring chance given up vs. the best option (lower is better)" />
        <Stat label="Gap-based score" value={fmt(summary.mean_score_minmax ?? null, 2)} hint="like the score, but counts how far he was from the best" />
      </div>
      <label className="flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={onlySub} onChange={(e) => setOnlySub(e.target.checked)} className="accent-lime" />
        Only passes where a better option existed
      </label>
      <div className="max-h-80 overflow-auto rounded-xl border border-line">
        <table className="w-full min-w-[720px] text-left text-xs">
          <thead className="sticky top-0 bg-panel text-muted">
            <tr>
              <th className="px-2 py-1.5 font-semibold">Min</th>
              <th className="px-2 font-semibold">Chosen</th>
              <th className="px-2 font-semibold">Best option</th>
              <th className="px-2 text-right font-semibold">ΔEPV chosen</th>
              <th className="px-2 text-right font-semibold">ΔEPV best</th>
              <th className="px-2 font-semibold">Score</th>
              <th className="px-2 text-right font-semibold">Gap score</th>
              <th className="px-2 text-right font-semibold">Result</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((p, i) => (
              <tr key={`${p.frame_num}-${i}`} className="border-t border-line/60">
                <td className="px-2 py-1 font-mono">{p.match_minute.toFixed(1)}'</td>
                <td className="px-2 text-slate-100">{p.chosen_name}</td>
                <td className={`px-2 ${p.chosen_name === p.best_name ? 'text-green-400' : 'text-slate-300'}`}>{p.best_name}</td>
                <td className="px-2 text-right font-mono">{signed(p.chosen_exp_depv, 4)}</td>
                <td className="px-2 text-right font-mono">{signed(p.max_exp_depv, 4)}</td>
                <td className="px-2">
                  {p.decision_score === null ? (
                    <span className="text-muted" title="fewer than 3 teammate options">–</span>
                  ) : (
                    <div className="flex items-center gap-1.5">
                      <div className="h-1.5 w-16 overflow-hidden rounded bg-ink"><div className="h-full bg-lime" style={{ width: `${Math.round(p.decision_score * 100)}%` }} /></div>
                      <span className="font-mono">{p.decision_score.toFixed(2)}</span>
                    </div>
                  )}
                </td>
                <td className="px-2 text-right font-mono text-muted">{p.score_minmax === null ? '–' : p.score_minmax.toFixed(2)}</td>
                <td className={`px-2 text-right ${p.success ? 'text-green-400' : 'text-red-400'}`}>{p.success ? '✓' : '✗'} <span className="font-mono text-muted">{signed(p.realised_depv, 3)}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > 12 ? (
        <button type="button" onClick={() => setShowAll((s) => !s)} className="text-xs font-semibold text-lime hover:underline">
          {showAll ? 'Show fewer' : `Show all ${rows.length} passes`}
        </button>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
export function Tab2PlayerAnalytics({ info }: { info: MatchInfo }) {
  const [players, setPlayers] = useState<PlayersResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<PlayerReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<AnalyticsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cache = useRef(new Map<string, PlayerReport>());

  useEffect(() => {
    const ctrl = new AbortController();
    getJson<PlayersResponse>(withMatch('/api/players', info.match_id), ctrl.signal)
      .then((r) => {
        setPlayers(r);
        const all: PlayerListItem[] = [...r.home, ...r.away].filter((p) => p.player_id);
        const pick = all.find((p) => /messi/i.test(p.name)) ?? [...all].sort((a, b) => b.minutes_tracked - a.minutes_tracked)[0];
        if (pick?.player_id) setSelected(pick.player_id);
      })
      .catch((e: unknown) => {
        if (!isAbort(e)) setError(errorMessage(e));
      });
    return () => ctrl.abort();
  }, [info.match_id]);

  useEffect(() => {
    if (!selected) return;
    const hit = cache.current.get(selected);
    if (hit) {
      setReport(hit);
      setError(null);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    getJson<PlayerReport>(withMatch(`/api/analytics/player/${selected}`, info.match_id), ctrl.signal)
      .then((r) => {
        cache.current.set(selected, r);
        setReport(r);
      })
      .catch((e: unknown) => {
        if (!isAbort(e)) setError(errorMessage(e));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [selected, info.match_id]);

  // while the heavy first-time computation runs, show its progress
  useEffect(() => {
    if (!loading) return;
    const ctrl = new AbortController();
    const poll = () =>
      getJson<AnalyticsStatus>(withMatch('/api/analytics/status', info.match_id), ctrl.signal)
        .then(setStatus)
        .catch(() => undefined);
    void poll();
    const id = setInterval(() => void poll(), 4000);
    return () => {
      clearInterval(id);
      ctrl.abort();
    };
  }, [loading, info.match_id]);

  const groups = players
    ? [
        { label: players.teams.home.name ?? 'Home', list: players.home },
        { label: players.teams.away.name ?? 'Away', list: players.away }
      ]
    : [];
  const readyCount = status ? Object.values(status.ready).filter(Boolean).length : 0;
  const decoy = report?.decoy_run_index;
  const press = report?.press_resistance;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-line bg-panel p-3">
        <label htmlFor="player-select" className="text-xs font-bold uppercase tracking-wider text-muted">Player</label>
        <select
          id="player-select"
          value={selected ?? ''}
          onChange={(e) => setSelected(e.target.value || null)}
          className="min-h-10 min-w-[240px] rounded-xl border border-line bg-ink px-3 text-sm text-white"
        >
          {!players ? <option value="">Loading players…</option> : null}
          {groups.map((g) => (
            <optgroup key={g.label} label={g.label}>
              {g.list
                .filter((p) => p.player_id)
                .map((p) => (
                  <option key={p.slot} value={p.player_id ?? ''}>#{p.jersey} {p.name} · {Math.round(p.minutes_tracked)} min</option>
                ))}
            </optgroup>
          ))}
        </select>
        {report ? <span className="text-sm text-muted">{report.player.team}{report.player.position ? ` · ${report.player.position}` : ''}</span> : null}
      </div>

      {error ? <ErrorBox message={error} /> : null}
      {loading ? (
        <div className="rounded-2xl border border-line bg-panel p-4">
          <Spinner label="Computing match analytics…" />
          <p className="mt-2 text-xs text-muted">
            The first request after the API starts computes pitch control for the whole match (about 1–3 minutes, then cached on disk).
            {status ? ` Progress: ${readyCount}/4 parts ready (${status.state}).` : ''}
          </p>
        </div>
      ) : null}

      {report && !loading ? (
        <div className="space-y-4">
          <section className="rounded-2xl border border-line bg-panel px-4 py-3 text-sm leading-relaxed text-slate-200">
            <span className="font-bold uppercase tracking-[0.14em] text-white">Player analytics</span> · Five deeper questions about one player, answered from the tracking and event data:
            <span className="text-white"> does he choose good passes, does he cope when closed down, does he tire, who does he link with, and do his runs help others?</span> Each section starts with a short explanation and ends with the answer for this player in plain words. For simple counts (passes, shots, tackles) at any minute, use the <span className="text-white">Live player</span> tab.
          </section>

          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            <Stat label="Actions under pressure" value={`${press?.n_pressured ?? 0} / ${press?.n_actions ?? 0}`} hint={`an opponent within ${press?.radius_m ?? 3} m`} help="On-ball actions (passes, shots, carries…) made while an opponent was closer than 3 metres." />
            <Stat label="Success under pressure" value={pct(press?.success_rate_under_pressure)} hint={`when free: ${pct(press?.success_rate_free)}`} help="Share of his pressured actions that worked. Compare with the 'when free' number: a small gap means he copes well." />
            <Stat label="Decoy runs" value={String(decoy?.n_decoy_runs ?? 0)} hint="runs without receiving the ball" help="Fast runs (19.8 km/h or more for at least one second) while his team had the ball and the pass did not come to him." />
            <Stat label="Space created by runs" value={signed(decoy?.total_delta_das, 1)} hint={`m² of danger space · ${signed(decoy?.mean_delta_das, 1)} per run`} help="Change in the team's dangerous passing space (DAS) after his decoy runs. Positive = he pulled defenders away and opened room." />
          </div>

          <Panel title="Pressure resistance" subtitle="How well does he act when an opponent is close?" right={null}>
            <Guide
              what="For every on-ball action we measure the distance to the nearest opponent at that moment. Closer than 3 metres counts as 'under pressure' (this matches the pressure label of the data provider much better than our first, pitch-control based version)."
              read="Compare 'success under pressure' with 'success when free'. 'Value per 100 pressured actions' adds up how much his successful pressured actions raised his team's scoring chance (VAEP above average), per 100 actions."
            />
            <InWords {...pressWords(press)} />
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Stat label="Pressured share" value={pct(press?.pressured_share)} hint="of all his on-ball actions" />
              <Stat label="Nearest opponent" value={press?.mean_nearest_opp_m == null ? '–' : `${press.mean_nearest_opp_m.toFixed(1)} m`} hint="on average when he acts" />
              <Stat label="Value per 100 pressured" value={fmt(press?.rating, 2)} hint="VAEP gained by successful pressured actions" help="Higher is better. Needs a decent number of pressured actions to mean anything." />
              <Stat label="Pressured actions" value={String(press?.n_pressured ?? 0)} hint={press?.reliable === false ? 'few actions: rough guide only' : 'sample size'} />
            </div>
            {press?.pressure_levels && press.n_actions > 0 && (
              <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Stat label="Average pressure" value={press.mean_pressure == null ? '–' : press.mean_pressure.toFixed(2)} hint="0 = free, 1 = closed down" help="A smooth score for every action: an opponent 5 m or more away adds 0, one right next to him adds close to 1, and several close defenders add up (never above 1)." />
                {press.pressure_levels.map((l) => (
                  <Stat key={l.level} label={`Success · ${l.level} pressure`} value={l.n ? pct(l.success_rate) : '–'} hint={`${l.n} actions`} />
                ))}
              </div>
            )}
            {press?.success_rate_under_pressure_ci && (
              <p className="mt-2 text-xs text-slate-500">
                Success under pressure: {pct(press.success_rate_under_pressure)} (likely between {pct(press.success_rate_under_pressure_ci[0])} and {pct(press.success_rate_under_pressure_ci[1])}).
              </p>
            )}
          </Panel>

          <Panel title="Decision efficiency" subtitle="Did he pick the best pass available?" right={null}>
            <Guide
              what="At the moment of each pass the model estimates how much each team-mate he could have passed to would have improved his team's chance of scoring (expected value, EPV, times the chance the pass arrives). Then it ranks the pass he chose against all the others."
              read="Decision score 1.0 = he chose the best option, 0.5 = a middle one, 0.0 = the worst. 'Regret' is the scoring chance he gave up compared with the best option. It judges the CHOICE, not whether the pass happened to work."
            />
            <InWords {...decisionWords(report.decision_efficiency.summary)} />
            <DecisionTable report={report} />
          </Panel>

          <Panel title="Fatigue · 15-minute blocks" subtitle="Does he cover less space as the match goes on?" right={null}>
            <Guide
              what="Every 15 seconds of live play (ball in play) we measure the area of the pitch that only he would reach first. The 'control radius' is the radius of a circle with that area. We then average it per 15-minute block and compare with his first full block."
              read="A radius that shrinks block after block (red bars) means he is covering less ground or staying in a smaller zone, which is a sign of tiring. Blocks with under 10 minutes on the pitch are not compared. The table shows what he actually ran in each block."
            />
            <InWords {...fatigueWords(report.fatigue_decay.bins)} />
            <FatigueCharts bins={report.fatigue_decay.bins} baseline={report.fatigue_decay.baseline_bin} />
          </Panel>

          <Panel title="Pairwise chemistry" subtitle={`${report.player.team} · who links well with whom`} right={null}>
            <Guide
              what="For every pair of team-mates that completed at least 3 passes, we compare how much those passes improved the scoring chance against a typical pass from the same part of the pitch."
              read="Rows are the passers, columns the receivers. Green = their passes are better than typical (the number is the gain × 1000), red = worse. The player you picked is highlighted in yellow."
            />
            <InWords text={chemistryWords(report.chemistry, report.player.slot)} />
            <ChemistryMatrix chem={report.chemistry} selectedSlot={report.player.slot} />
          </Panel>

          {decoy && decoy.top_runs && decoy.top_runs.length ? (
            <Panel title="Decoy runs" subtitle="Do his runs without the ball help team-mates?" right={null}>
              <Guide
                what="A decoy run is a fast run (19.8 km/h or more, for at least a second) made while his team has the ball and the pass is NOT played to him. Defenders often follow him, which can open space for others."
                read="ΔDAS is the change in the team's dangerous passing space (square metres, DAS) across the run. Positive = the run opened up room. The best runs are listed with their minute so you can find them on the Core metrics tab."
              />
              <InWords {...decoyWords(decoy)} />
              <table className="w-full text-left text-xs">
                <thead className="text-muted"><tr><th className="py-1 font-semibold">Minute</th><th className="font-semibold">Duration</th><th className="font-semibold">Avg speed</th><th className="text-right font-semibold">ΔDAS (team, m²)</th></tr></thead>
                <tbody>
                  {decoy.top_runs.map((r) => (
                    <tr key={r.frame_num} className="border-t border-line/60">
                      <td className="py-1 font-mono">{r.match_minute.toFixed(1)}'</td>
                      <td className="font-mono">{r.duration_s.toFixed(1)} s</td>
                      <td className="font-mono">{r.mean_speed.toFixed(1)} m/s</td>
                      <td className={`text-right font-mono ${r.delta_das >= 0 ? 'text-green-400' : 'text-red-400'}`}>{signed(r.delta_das, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

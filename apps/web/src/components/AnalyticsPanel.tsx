import {
  PLAYER_SLOTS,
  TOTAL_SLOTS,
  classifyOutputTrend,
  playerOnPitchAt,
  trackingQualityBetween,
  validateEventScore,
  workloadSnapshotAt,
  type EventDataset,
  type EventRosterPlayer,
  type PlayerTechnicalStats,
  type PlayerWorkload,
  type PlayerWorkloadSnapshot,
  type OutputTrendStatus,
  type TrackingData,
  type WorkloadConfig
} from '@replay/shared';
import { Activity, BarChart3, Radio, ShieldCheck, TimerReset, TrendingDown, Zap } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useUiStore } from '../store';

type View = 'match' | 'player' | 'workload';
type SortKey = 'player' | 'speed' | 'distance' | 'workRate' | 'hsr' | 'sprintDistance' | 'sprints' | 'maxSpeed' | 'efforts' | 'observed' | 'quality' | 'workTrend' | 'intensityTrend' | 'sprintTrend' | 'status';

interface RateSet { distance: number; highSpeed: number; sprints: number; quality: number }
interface LiveWorkload {
  source: PlayerWorkload;
  snapshot: PlayerWorkloadSnapshot;
  speedMps: number;
  distanceRate: number;
  quality: number;
  trendQuality: number | null;
  workRateTrend: number | null;
  highSpeedTrend: number | null;
  sprintTrend: number | null;
  status: OutputTrendStatus;
  pitchStateLabel: string;
}

function percentage(value: number | null): string {
  return value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(0)}%`;
}

function change(baseline: number, recent: number): number | null {
  return baseline > 0 ? ((recent - baseline) / baseline) * 100 : null;
}

function ratesBetween(start: PlayerWorkloadSnapshot, end: PlayerWorkloadSnapshot): RateSet | null {
  const seconds = end.observedSeconds - start.observedSeconds;
  if (seconds < 120) return null;
  return {
    distance: (end.distanceM - start.distanceM) / seconds * 60,
    // HSR and sprint are stored as disjoint zones; sum them to preserve the prior
    // high-intensity trend definition (all distance at or above the HSR threshold).
    highSpeed: ((end.highSpeedDistanceM + end.sprintDistanceM)
      - (start.highSpeedDistanceM + start.sprintDistanceM)) / seconds * 60,
    sprints: (end.sprintCount - start.sprintCount) / seconds * 60,
    quality: trackingQualityBetween(start, end)
  };
}

function currentSpeed(data: TrackingData, frameIndex: number, team: 0 | 1, jerseyNum: number, config: WorkloadConfig): number {
  if (frameIndex <= 0) return 0;
  const findSlot = (index: number): number => {
    const start = team * PLAYER_SLOTS;
    const jerseyOffset = index * TOTAL_SLOTS;
    for (let slot = start; slot < start + PLAYER_SLOTS; slot += 1) {
      if (data.smoothedJerseys[jerseyOffset + slot] === jerseyNum) return slot;
    }
    return -1;
  };
  const slot = findSlot(frameIndex);
  const previousSlot = findSlot(frameIndex - 1);
  const seconds = ((data.videoTimeMs[frameIndex] ?? 0) - (data.videoTimeMs[frameIndex - 1] ?? 0)) / 1000;
  if (slot < 0 || previousSlot < 0 || seconds <= 0 || seconds > config.maxFrameGapSeconds) return 0;
  const offset = (frameIndex * TOTAL_SLOTS + slot) * 2;
  const previousOffset = ((frameIndex - 1) * TOTAL_SLOTS + previousSlot) * 2;
  const values = [data.smoothedPlayers[offset], data.smoothedPlayers[offset + 1], data.smoothedPlayers[previousOffset], data.smoothedPlayers[previousOffset + 1]];
  if (!values.every(Number.isFinite)) return 0;
  const speed = Math.hypot(values[0]! - values[2]!, values[1]! - values[3]!) / seconds;
  return Number.isFinite(speed) && speed <= config.maxValidSpeedMps ? speed : 0;
}

function buildLiveWorkload(data: TrackingData, dataset: EventDataset | null, workload: PlayerWorkload, timeMs: number, frameIndex: number, config: WorkloadConfig): LiveWorkload {
  const observed = workloadSnapshotAt(workload, timeMs);
  const clock = data.periodGameClockTime[frameIndex] ?? 0;
  const snapshot = observed ?? {
    sampleIndex: -1, videoTimeMs: timeMs, matchClockSeconds: (data.period[frameIndex] === 2 && clock < 2700) ? clock + 2700 : clock,
    observedSeconds: 0, distanceM: 0, highSpeedDistanceM: 0, sprintDistanceM: 0,
    sprintCount: 0, accelerationCount: 0, decelerationCount: 0, maxSpeedMps: 0,
    speedP95Mps: 0, fadedObservations: 0, totalObservations: 0
  };
  const identity = dataset ? identityFor(workload, dataset.roster) : null;
  const onPitch = observed !== null && (!dataset || !identity || playerOnPitchAt(dataset.events, identity.playerId, timeMs));
  const substitutions = identity && dataset
    ? dataset.events.filter((event) => event.gameType === 'SUB'
      && (event.playerOnId === identity.playerId || event.playerOffId === identity.playerId))
    : [];
  const lastChange = substitutions.filter((event) => event.eventTimeMs <= timeMs).at(-1);
  const futureEntry = substitutions.find((event) => event.eventTimeMs > timeMs && event.playerOnId === identity?.playerId);
  const pitchStateLabel = lastChange && lastChange.playerOffId === identity?.playerId
    ? `Subbed off at ${lastChange.clock}`
    : lastChange && lastChange.playerOnId === identity?.playerId
      ? `Subbed on at ${lastChange.clock}`
      : futureEntry ? `Not entered · subbed on at ${futureEntry.clock}` : 'On pitch';
  const recentStart = snapshotByClock(workload, snapshot.matchClockSeconds - 5 * 60);
  const baselineStart = snapshotByClock(workload, snapshot.matchClockSeconds - 20 * 60);
  const recent = recentStart ? ratesBetween(recentStart, snapshot) : null;
  const baseline = recentStart && baselineStart ? ratesBetween(baselineStart, recentStart) : null;
  const workRateTrend = recent && baseline ? change(baseline.distance, recent.distance) : null;
  const highSpeedTrend = recent && baseline ? change(baseline.highSpeed, recent.highSpeed) : null;
  const sprintTrend = recent && baseline ? change(baseline.sprints, recent.sprints) : null;
  const trendQuality = recent && baseline ? Math.min(recent.quality, baseline.quality) : null;
  const minimumQuality = config.minimumTrendQualityPercent / 100;
  const qualityPasses = trendQuality === null || trendQuality >= minimumQuality;
  return {
    source: workload,
    snapshot,
    speedMps: currentSpeed(data, frameIndex, workload.team, workload.jerseyNum, config),
    distanceRate: snapshot.distanceM / Math.max(1, snapshot.observedSeconds) * 60,
    quality: 1 - snapshot.fadedObservations / Math.max(1, snapshot.totalObservations),
    trendQuality,
    workRateTrend: qualityPasses ? workRateTrend : null,
    highSpeedTrend: qualityPasses ? highSpeedTrend : null,
    sprintTrend: qualityPasses ? sprintTrend : null,
    status: classifyOutputTrend({ distanceChangePercent: workRateTrend, highIntensityChangePercent: highSpeedTrend, sprintChangePercent: sprintTrend, onPitch, quality: trendQuality, minimumQuality }),
    pitchStateLabel
  };
}

function statusClass(status: OutputTrendStatus): string {
  if (status === 'High decline') return 'border-rose-400/35 bg-rose-400/10 text-rose-200';
  if (status === 'Watch') return 'border-amber-400/35 bg-amber-400/10 text-amber-200';
  if (status === 'Stable') return 'border-sky-300/35 bg-sky-300/10 text-sky-200';
  if (status === 'Low data quality') return 'border-violet-300/35 bg-violet-300/10 text-violet-200';
  return 'border-slate-500/40 bg-white/[0.04] text-muted';
}

function statusHint(player: LiveWorkload, minimumQuality: number): string {
  if (player.status === 'Not on pitch') return player.pitchStateLabel;
  if (player.status === 'Low data quality') return `Recent or baseline tracking quality is ${(player.trendQuality ?? 0) * 100 < minimumQuality ? 'below' : 'at'} the ${minimumQuality}% minimum.`;
  if (player.status === 'Building baseline') return 'At least 20 earlier minutes and 5 recent valid minutes are required.';
  if (player.status === 'High decline') return 'Work rate is down at least 20% and high-intensity rate is down 25% or sprint frequency is down 30%.';
  if (player.status === 'Watch') return 'Work rate is down 10%, high-intensity rate 15%, or sprint frequency 20%.';
  return 'None of the configured output-decline thresholds is currently crossed.';
}

function sortValue(player: LiveWorkload, key: SortKey): number | string {
  if (key === 'player') return player.source.jerseyNum;
  if (key === 'speed') return player.speedMps;
  if (key === 'distance') return player.snapshot.distanceM;
  if (key === 'workRate') return player.distanceRate;
  if (key === 'hsr') return player.snapshot.highSpeedDistanceM;
  if (key === 'sprintDistance') return player.snapshot.sprintDistanceM;
  if (key === 'sprints') return player.snapshot.sprintCount;
  if (key === 'maxSpeed') return player.snapshot.maxSpeedMps;
  if (key === 'efforts') return player.snapshot.accelerationCount + player.snapshot.decelerationCount;
  if (key === 'observed') return player.snapshot.observedSeconds;
  if (key === 'quality') return player.quality;
  if (key === 'workTrend') return player.workRateTrend ?? -Infinity;
  if (key === 'intensityTrend') return player.highSpeedTrend ?? -Infinity;
  if (key === 'sprintTrend') return player.sprintTrend ?? -Infinity;
  return player.status;
}

function Metric({ label, value, hint, icon }: { label: string; value: string; hint: string; icon: React.ReactNode }) {
  return <div className="rounded-2xl border border-line bg-black/20 p-4"><div className="flex items-center justify-between gap-3 text-muted"><span className="text-xs font-bold uppercase tracking-[0.12em]">{label}</span>{icon}</div><p className="mt-2 font-mono text-2xl font-bold tabular-nums text-white">{value}</p><p className="mt-1 text-xs leading-5 text-muted">{hint}</p></div>;
}

function emptyTechnical(player: EventRosterPlayer): PlayerTechnicalStats {
  return { playerId: player.playerId, playerName: player.playerName, team: player.team, jerseyNum: player.jerseyNum, position: player.position, involvements: 0, passes: 0, completedPasses: 0, shots: 0, goalOutcomes: 0, crosses: 0, challenges: 0, clearances: 0, rebounds: 0, carries: 0, touches: 0, fouls: 0 };
}

function liveEventStats(dataset: EventDataset | null, timeMs: number) {
  const gameCounts: Record<string, number> = {};
  const possessionCounts: Record<string, number> = {};
  const players = new Map((dataset?.roster ?? []).map((player) => [player.playerId, emptyTechnical(player)]));
  let count = 0;
  for (const event of dataset?.events ?? []) {
    if (event.eventTimeMs > timeMs) break;
    count += 1;
    if (event.gameType) gameCounts[event.gameType] = (gameCounts[event.gameType] ?? 0) + 1;
    if (event.possessionType) possessionCounts[event.possessionType] = (possessionCounts[event.possessionType] ?? 0) + 1;
    const involved = event.playerId ? players.get(event.playerId) : null;
    if (involved) involved.involvements += 1;
    const actor = event.actorPlayerId ? players.get(event.actorPlayerId) : null;
    if (!actor) continue;
    if (event.possessionType === 'PA') { actor.passes += 1; if (event.outcome === 'C') actor.completedPasses += 1; }
    else if (event.possessionType === 'SH') { actor.shots += 1; if (event.outcome === 'G') actor.goalOutcomes += 1; }
    else if (event.possessionType === 'CR') actor.crosses += 1;
    else if (event.possessionType === 'CH') actor.challenges += 1;
    else if (event.possessionType === 'CL') actor.clearances += 1;
    else if (event.possessionType === 'RE') actor.rebounds += 1;
    else if (event.possessionType === 'BC') actor.carries += 1;
    else if (event.possessionType === 'TC' || event.possessionType === 'IT') actor.touches += 1;
    else if (event.possessionType === 'FO') actor.fouls += 1;
  }
  return { count, gameCounts, possessionCounts, players };
}

function snapshotByClock(workload: PlayerWorkload, clockSeconds: number): PlayerWorkloadSnapshot | null {
  const clocks = workload.timeline.matchClockSeconds;
  if (!clocks.length || clockSeconds < (clocks[0] ?? 0)) return null;
  let low = 0;
  let high = clocks.length - 1;
  while (low <= high) { const middle = (low + high) >>> 1; if ((clocks[middle] ?? 0) <= clockSeconds) low = middle + 1; else high = middle - 1; }
  return workloadSnapshotAt(workload, workload.timeline.videoTimeMs[Math.max(0, high)] ?? 0);
}

function chartBuckets(workload: PlayerWorkload, snapshot: PlayerWorkloadSnapshot): Array<{ label: string; rate: number; live: boolean }> {
  const count = Math.floor(snapshot.matchClockSeconds / 300) + 1;
  const result: Array<{ label: string; rate: number; live: boolean }> = [];
  for (let bucket = 0; bucket < count; bucket += 1) {
    const start = snapshotByClock(workload, bucket * 300) ?? workloadSnapshotAt(workload, workload.timeline.videoTimeMs[0] ?? 0);
    const end = snapshotByClock(workload, Math.min(snapshot.matchClockSeconds, (bucket + 1) * 300));
    if (!start || !end) continue;
    const seconds = end.observedSeconds - start.observedSeconds;
    result.push({ label: `${bucket * 5}–${bucket * 5 + 5}`, rate: seconds > 0 ? (end.distanceM - start.distanceM) / seconds * 60 : 0, live: bucket === count - 1 });
  }
  return result;
}

function identityFor(workload: PlayerWorkload, roster: EventRosterPlayer[]): EventRosterPlayer | null {
  return roster.find((player) => player.team === workload.team && player.jerseyNum === workload.jerseyNum) ?? null;
}

export function AnalyticsPanel({ events, data, workloadConfig, currentTimeMs, frameIndex, clockLabel, homeName, awayName, observedOnly = false }: { events: EventDataset | null; data: TrackingData; workloadConfig: WorkloadConfig; currentTimeMs: number; frameIndex: number; clockLabel: string; homeName: string; awayName: string; observedOnly?: boolean }) {
  const [view, setView] = useState<View>('workload');
  const [teamFilter, setTeamFilter] = useState<'all' | 'home' | 'away'>('all');
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('player');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const selectedKey = useUiStore((state) => state.selectedPlayer) ?? '';
  const selectPlayer = useUiStore((state) => state.set);
  const tableRef = useRef<HTMLDivElement>(null);
  const roster = events?.roster ?? [];
  const eventStats = useMemo(() => liveEventStats(events, currentTimeMs), [events, currentTimeMs]);
  const livePlayers = useMemo(() => data.workloads.map((workload) => buildLiveWorkload(data, events, workload, currentTimeMs, frameIndex, workloadConfig)), [currentTimeMs, data, events, frameIndex, workloadConfig]);
  const visiblePlayers = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return livePlayers
      .filter((player) => teamFilter === 'all' || player.source.team === (teamFilter === 'home' ? 0 : 1))
      .filter((player) => {
        const identity = identityFor(player.source, roster);
        return !normalizedQuery || String(player.source.jerseyNum).includes(normalizedQuery)
          || (identity?.playerName ?? '').toLowerCase().includes(normalizedQuery);
      })
      .sort((left, right) => {
        const a = sortValue(left, sortKey);
        const b = sortValue(right, sortKey);
        const order = typeof a === 'string' && typeof b === 'string' ? a.localeCompare(b) : Number(a) - Number(b);
        return sortDirection === 'asc' ? order : -order;
      });
  }, [livePlayers, query, roster, sortDirection, sortKey, teamFilter]);

  useEffect(() => { if (!selectedKey && livePlayers[0]) selectPlayer({ selectedPlayer: livePlayers[0].source.key }); }, [livePlayers, selectPlayer, selectedKey]);
  useEffect(() => {
    if (!selectedKey) return;
    tableRef.current?.querySelector(`[data-player-key="${selectedKey}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [selectedKey]);

  const changeSort = (key: SortKey) => {
    if (sortKey === key) setSortDirection((direction) => direction === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDirection('asc'); }
  };
  const setSelectedKey = (key: string) => selectPlayer({ selectedPlayer: key });

  const selected = livePlayers.find((player) => player.source.key === selectedKey) ?? livePlayers[0] ?? null;
  const selectedIdentity = selected ? identityFor(selected.source, roster) : null;
  const selectedTechnical = selectedIdentity ? eventStats.players.get(selectedIdentity.playerId) ?? null : null;
  const buckets = selected ? chartBuckets(selected.source, selected.snapshot) : [];
  const chartMaximum = Math.max(1, ...buckets.map((bucket) => bucket.rate));
  const p = eventStats.possessionCounts;
  const scoreValidation = useMemo(() => events ? validateEventScore(events) : null, [events]);
  const possessionEvents = events?.events.filter((event) => event.eventTimeMs <= currentTimeMs && event.homeBall !== null) ?? [];
  const homePossession = possessionEvents.length
    ? possessionEvents.filter((event) => event.homeBall).length / possessionEvents.length * 100
    : null;

  return <section className="rounded-3xl border border-line bg-panel p-4 shadow-panel sm:p-6" aria-labelledby="analytics-title">
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div><div className="flex items-center gap-2 text-lime"><Radio aria-hidden="true" className="h-4 w-4" /><p className="text-xs font-bold uppercase tracking-[0.16em]">Live analysis · {clockLabel}</p></div><h2 id="analytics-title" className="mt-1 text-xl font-semibold text-white">Manager workload monitor</h2><p className="mt-1 text-sm text-muted">{observedOnly ? 'Played API frames only, through the current playback time. Not full-match totals. Workload checkpoints update about once per source second.' : 'Every value is limited to the current replay frame. Cumulative checkpoints update approximately once per second.'}</p></div>
      <div className="flex rounded-xl border border-line bg-black/20 p-1" role="tablist" aria-label="Analytics views">{(['match', 'player', 'workload'] as const).map((tab) => <button key={tab} type="button" role="tab" aria-selected={view === tab} disabled={!events && tab !== 'workload'} onClick={() => setView(tab)} className={`min-h-11 rounded-lg px-4 text-sm font-semibold capitalize transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${view === tab ? 'bg-lime text-[#152000]' : 'text-muted hover:text-white'}`}>{tab === 'workload' ? 'Live workload' : tab}</button>)}</div>
    </div>

    {view === 'match' && events ? <><div className="mt-5">
      {scoreValidation?.status === 'mismatch' ? <p className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-3 text-sm text-amber-200">Score hidden: raw G outcomes produce {scoreValidation.observedHome}–{scoreValidation.observedAway}, which does not match the known 1–2 result.</p> : null}
      {scoreValidation?.status === 'matched' ? <div className="grid gap-3 sm:grid-cols-2"><Metric label="Verified score" value={`${scoreValidation.observedHome}–${scoreValidation.observedAway}`} hint="Derived from validated SH/G events" icon={<ShieldCheck aria-hidden="true" className="h-4 w-4" />} />{homePossession !== null ? <Metric label="Home possession" value={`${homePossession.toFixed(0)}%`} hint="Share of events carrying home_ball" icon={<Activity aria-hidden="true" className="h-4 w-4" />} /> : null}</div> : null}
    </div><div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
      <Metric label="Events so far" value={eventStats.count.toLocaleString()} hint="At or before playhead" icon={<Activity aria-hidden="true" className="h-4 w-4" />} /><Metric label="Passes" value={String(p.PA ?? 0)} hint="Raw code PA" icon={<BarChart3 aria-hidden="true" className="h-4 w-4" />} /><Metric label="Shots" value={String(p.SH ?? 0)} hint="Raw code SH" icon={<Zap aria-hidden="true" className="h-4 w-4" />} /><Metric label="Challenges" value={String(p.CH ?? 0)} hint="Raw code CH" icon={<ShieldCheck aria-hidden="true" className="h-4 w-4" />} /><Metric label="Crosses" value={String(p.CR ?? 0)} hint="Raw code CR" icon={<TrendingDown aria-hidden="true" className="h-4 w-4 rotate-180" />} /><Metric label="Subs" value={String(eventStats.gameCounts.SUB ?? 0)} hint={`${p.FO ?? 0} coded fouls`} icon={<TimerReset aria-hidden="true" className="h-4 w-4" />} />
    </div></> : null}

    {view === 'player' && events ? <div className="mt-5"><label className="text-sm font-medium text-slate-200" htmlFor="technical-player">Player</label><select id="technical-player" value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)} className="ml-3 h-12 rounded-xl border border-line bg-[#0d1215] px-4 text-white">{livePlayers.map((player) => { const identity = identityFor(player.source, roster); return <option key={player.source.key} value={player.source.key}>{player.source.team === 0 ? homeName : awayName} · #{player.source.jerseyNum} · {identity?.playerName ?? 'Player'}</option>; })}</select><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Metric label="Involvements" value={String(selectedTechnical?.involvements ?? 0)} hint="Events so far" icon={<Activity aria-hidden="true" className="h-4 w-4" />} /><Metric label="Passes" value={`${selectedTechnical?.completedPasses ?? 0}/${selectedTechnical?.passes ?? 0}`} hint="Completed / attempted so far" icon={<BarChart3 aria-hidden="true" className="h-4 w-4" />} /><Metric label="Shots" value={String(selectedTechnical?.shots ?? 0)} hint={`${selectedTechnical?.goalOutcomes ?? 0} raw G outcomes`} icon={<Zap aria-hidden="true" className="h-4 w-4" />} /><Metric label="Actions" value={String((selectedTechnical?.challenges ?? 0) + (selectedTechnical?.crosses ?? 0) + (selectedTechnical?.touches ?? 0))} hint="Challenges + crosses + touches" icon={<ShieldCheck aria-hidden="true" className="h-4 w-4" />} /></div></div> : null}

    {view === 'workload' ? <div className="mt-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-wrap items-end gap-3"><div className="flex rounded-xl border border-line bg-black/20 p-1" aria-label="Filter workload by team">{(['all', 'home', 'away'] as const).map((filter) => <button key={filter} type="button" aria-pressed={teamFilter === filter} onClick={() => setTeamFilter(filter)} className={`min-h-11 rounded-lg px-4 text-sm font-semibold ${teamFilter === filter ? 'bg-white/10 text-white' : 'text-muted hover:text-white'}`}>{filter === 'all' ? 'All players' : filter === 'home' ? homeName : awayName}</button>)}</div><label className="text-xs text-muted" htmlFor="workload-search">Find player<input id="workload-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name or shirt" className="mt-1 block h-11 w-44 rounded-xl border border-line bg-[#0d1215] px-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-lime" /></label></div>
        <p className="text-xs text-muted">Output trend: recent 5 min vs previous valid 15 min · minimum quality {workloadConfig.minimumTrendQualityPercent}%</p>
      </div>
      <div ref={tableRef} className="mt-4 max-h-[70vh] overflow-auto rounded-2xl border border-line"><table className="w-full min-w-[1500px] border-collapse text-left text-sm"><caption className="sr-only">Live workload and output indicators for every tracked player at {clockLabel}</caption><thead className="sticky top-0 z-20 bg-[#101619] text-xs uppercase tracking-wider text-muted"><tr><SortHeader label="Player" column="player" active={sortKey} direction={sortDirection} onSort={changeSort} sticky /><SortHeader label="Live speed" column="speed" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Distance" column="distance" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Work rate" column="workRate" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="HSR distance" column="hsr" active={sortKey} direction={sortDirection} onSort={changeSort} title={`High-speed: ${workloadConfig.highSpeedThresholdMps.toFixed(1)}–${workloadConfig.sprintThresholdMps.toFixed(1)} m/s`} /><SortHeader label="Sprint dist." column="sprintDistance" active={sortKey} direction={sortDirection} onSort={changeSort} title={`Sprint: at least ${workloadConfig.sprintThresholdMps.toFixed(1)} m/s`} /><SortHeader label="Sprints" column="sprints" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Max / P95" column="maxSpeed" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Acc / Dec" column="efforts" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Observed" column="observed" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Quality" column="quality" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Work Δ" column="workTrend" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Intensity Δ" column="intensityTrend" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Sprint Δ" column="sprintTrend" active={sortKey} direction={sortDirection} onSort={changeSort} /><SortHeader label="Output trend" column="status" active={sortKey} direction={sortDirection} onSort={changeSort} /></tr></thead>
        <tbody>{visiblePlayers.map((player) => {
          const identity = identityFor(player.source, roster);
          const active = player.source.key === selected?.source.key;
          return <tr key={player.source.key} data-player-key={player.source.key} onClick={() => setSelectedKey(player.source.key)} className={`cursor-pointer border-t border-line transition-colors hover:bg-white/[0.04] ${active ? 'bg-lime/[0.06]' : ''}`}>
            <td className={`sticky left-0 z-10 px-3 py-3 shadow-[1px_0_0_#293136] ${active ? 'bg-[#18221a]' : 'bg-[#101619]'}`}><button type="button" onClick={() => setSelectedKey(player.source.key)} className="min-h-11 text-left"><span className="block font-semibold text-white">#{player.source.jerseyNum} {identity?.playerName ?? 'Tracked player'}</span><span className="block text-xs text-muted">{player.source.team === 0 ? homeName : awayName} · {identity?.position ?? 'Position n/a'}</span>{player.pitchStateLabel !== 'On pitch' ? <span className="mt-1 block text-xs text-sky-200">{player.pitchStateLabel}</span> : null}</button></td>
            <DataCell value={`${player.speedMps.toFixed(1)} m/s`} /><DataCell value={`${(player.snapshot.distanceM / 1000).toFixed(2)} km`} /><DataCell value={`${player.distanceRate.toFixed(0)} m/min`} /><DataCell value={`${player.snapshot.highSpeedDistanceM.toFixed(0)} m`} /><DataCell value={`${player.snapshot.sprintDistanceM.toFixed(0)} m`} /><DataCell value={String(player.snapshot.sprintCount)} /><DataCell value={`${player.snapshot.maxSpeedMps.toFixed(1)} / ${player.snapshot.speedP95Mps.toFixed(1)}`} /><DataCell value={`${player.snapshot.accelerationCount} / ${player.snapshot.decelerationCount}`} /><DataCell value={`${(player.snapshot.observedSeconds / 60).toFixed(1)} min`} />
            <td className="whitespace-nowrap px-3 py-3 text-right"><span title="Tracking quality in the recent and baseline comparison windows" className={`rounded-full border px-2 py-1 font-mono text-xs font-bold ${player.trendQuality !== null && player.trendQuality * 100 < workloadConfig.minimumTrendQualityPercent ? 'border-violet-300/35 bg-violet-300/10 text-violet-200' : 'border-line text-slate-200'}`}>{((player.trendQuality ?? player.quality) * 100).toFixed(0)}%</span></td>
            <TrendCell value={player.workRateTrend} /><TrendCell value={player.highSpeedTrend} /><TrendCell value={player.sprintTrend} /><td className="px-3 py-3"><span title={statusHint(player, workloadConfig.minimumTrendQualityPercent)} className={`inline-flex whitespace-nowrap rounded-full border px-2.5 py-1 text-xs font-bold ${statusClass(player.status)}`}>{player.status}</span></td>
          </tr>;
        })}</tbody>
      </table></div>
      {selected ? <div className="mt-5 grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]"><div className="rounded-2xl border border-line bg-black/20 p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="font-semibold text-white">#{selected.source.jerseyNum} {selectedIdentity?.playerName ?? 'Tracked player'} · five-minute work rate</h3><p className="mt-1 text-xs text-muted">Only intervals at or before {clockLabel} are shown; the striped bar is the live partial interval.</p></div><span className={`rounded-full border px-3 py-1 text-xs font-bold ${statusClass(selected.status)}`}>{selected.status}</span></div><div className="mt-5 flex h-48 items-end gap-1" role="img" aria-label={`Five-minute distance work-rate chart for jersey ${selected.source.jerseyNum}`}>{buckets.map((bucket) => <div key={bucket.label} title={`${bucket.label} min: ${bucket.rate.toFixed(1)} m/min${bucket.live ? ' (live)' : ''}`} className={`min-w-2 flex-1 rounded-t ${bucket.live ? 'live-workload-bar' : 'bg-lime/70'}`} style={{ height: `${Math.max(3, bucket.rate / chartMaximum * 100)}%` }} />)}</div><div className="mt-2 flex justify-between text-xs text-muted"><span>{observedOnly ? 'Observed history' : 'Kick-off'}</span><span>5-minute intervals</span><span>Now</span></div></div><div className="rounded-2xl border border-line bg-black/20 p-4"><h3 className="font-semibold text-white">Live output-decline indicators</h3><p className="mt-1 text-xs leading-5 text-muted">Changes compare recent output with this player’s own earlier baseline.</p><dl className="mt-4 space-y-3"><TrendRow label="Work rate" value={selected.workRateTrend} /><TrendRow label="High-intensity rate" value={selected.highSpeedTrend} /><TrendRow label="Sprint frequency" value={selected.sprintTrend} /></dl><div className="mt-4 border-t border-line pt-4 text-xs leading-5 text-muted">“Building baseline” means more valid minutes are needed. “Low data quality” suppresses unreliable trends. Output can change because of tactics, stoppages, position, or score state.</div></div></div> : <p className="mt-4 rounded-xl border border-dashed border-slate-600 p-4 text-sm text-muted">No tracked player has observations at this frame yet.</p>}
    </div> : null}
  </section>;
}

function DataCell({ value }: { value: string }) { return <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs tabular-nums text-slate-200">{value}</td>; }
function SortHeader({ label, column, active, direction, onSort, sticky = false, title }: { label: string; column: SortKey; active: SortKey; direction: 'asc' | 'desc'; onSort: (key: SortKey) => void; sticky?: boolean; title?: string }) {
  const selected = active === column;
  return <th aria-sort={selected ? (direction === 'asc' ? 'ascending' : 'descending') : 'none'} title={title} className={`whitespace-nowrap p-0 text-right ${sticky ? 'sticky left-0 z-30 bg-[#101619] text-left shadow-[1px_0_0_#293136]' : ''}`}><button type="button" onClick={() => onSort(column)} className={`min-h-11 w-full px-3 py-3 font-bold uppercase tracking-wider hover:bg-white/[0.05] hover:text-white ${sticky ? 'text-left' : 'text-right'}`}>{label}<span aria-hidden="true" className="ml-1 text-lime">{selected ? (direction === 'asc' ? '↑' : '↓') : ''}</span></button></th>;
}
function TrendCell({ value }: { value: number | null }) { const tone = value == null ? 'text-muted' : value < -10 ? 'text-amber-300' : 'text-sky-300'; return <td className={`whitespace-nowrap px-3 py-3 text-right font-mono text-xs font-bold tabular-nums ${tone}`}>{percentage(value)}</td>; }
function TrendRow({ label, value }: { label: string; value: number | null }) { return <div className="flex items-center justify-between gap-4"><dt className="text-sm text-slate-200">{label}</dt><dd className={`font-mono text-sm font-bold ${value == null ? 'text-muted' : value < -10 ? 'text-amber-300' : 'text-sky-300'}`}>{percentage(value)}</dd></div>; }

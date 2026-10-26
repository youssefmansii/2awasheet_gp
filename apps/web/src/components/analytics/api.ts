// Typed client for the Python analytics API (apps/api). Isolated: touches nothing else in the app.

export const API_BASE: string =
  (import.meta.env.VITE_ANALYTICS_API_URL as string | undefined) ?? 'http://localhost:8000';

export type Side = 'home' | 'away';

export interface TeamInfo {
  id: string | null;
  name: string | null;
}

export interface PeriodInfo {
  period: number;
  first_frame: number;
  last_frame: number;
  duration_s: number;
  attack_dir: { home: number; away: number };
}

export interface MatchInfo {
  match_id: string;
  fps: number;
  n_frames: number;
  teams: { home: TeamInfo; away: TeamInfo };
  periods: PeriodInfo[];
  bins: string[];
  pitch: { length: number; width: number; nx: number; ny: number };
  warmup: string;
  analytics_ready: Record<string, boolean>;
}

export interface FramePlayer {
  slot: number;
  side: Side;
  jersey: string;
  player_id: string | null;
  name: string;
  x: number;
  y: number;
  speed: number | null;
  obso?: number;
  obso_target?: [number, number];
}

export interface FramePayload {
  match_id: string;
  frame_num: number;
  idx: number;
  period: number;
  time_s: number;
  match_minute: number;
  attacking_side: Side;
  attack_dir: number;
  /** false before the first on-ball event of the match; then "attacking_side" is only a calculation default */
  possession_known?: boolean;
  /** false when the ball is not tracked (dead ball): DAS, OBSO and EPV cannot be computed */
  ball_tracked?: boolean;
  ball: { x: number; y: number } | null;
  players: FramePlayer[];
  grid: { nx: number; ny: number; length: number; width: number };
  pitch_control_home: number[][];
  das: number[][];
  /** C-OBSO grid (pitch control × transition × EPV/max) for the team in possession; row 0 = bottom of the pitch. */
  obso_grid?: number[][];
  /** where the DAS number comes from, e.g. "accessible-space 2.1.0" */
  das_source?: string;
  metrics: {
    das_value: number | null;
    as_value?: number | null;
    obso_total: number;
    epv_ball: number | null;
    epv_home: number | null;
    pc_home_share: number;
    defensive_line_x: number;
  };
  compute_ms?: number;
}

export interface LivePeriod {
  period: number;
  duration_s: number;
  /** [start, end] in seconds from the start of the period where the ball is tracked */
  segments: [number, number][];
  live_s: number;
}

export interface ActionRow {
  frame_num: number;
  period: number;
  match_minute: number;
  side: Side;
  player: string | null;
  jersey: string | null;
  action: string;
  success: boolean;
  x: number | null;
  y: number | null;
  end_x: number | null;
  end_y: number | null;
  epv_before: number;
  epv_after: number;
  delta_epv: number;
  vaep: number;
  obv: number;
}

export interface EpvPoint {
  frame_num: number;
  period: number;
  time_s: number;
  match_minute: number;
  possession: Side;
  epv_home: number;
}

export interface PlayerListItem {
  slot: number;
  player_id: string | null;
  name: string;
  jersey: string;
  side: Side;
  team: string;
  position: string | null;
  minutes_tracked: number;
}

export interface PlayersResponse {
  match_id: string;
  teams: { home: TeamInfo; away: TeamInfo };
  home: PlayerListItem[];
  away: PlayerListItem[];
}

export interface DecisionRow {
  frame_num: number;
  period: number;
  match_minute: number;
  chosen_name: string;
  best_name: string;
  n_options: number;
  chosen_exp_depv: number;
  max_exp_depv: number;
  regret: number | null;
  /** rank score: 1 = best option, 0 = worst option; null when fewer than 3 options existed */
  decision_score: number | null;
  /** (chosen - worst) / (best - worst) on expected dEPV */
  score_minmax: number | null;
  /** legacy: dEPV chosen / dEPV best */
  ratio_score: number | null;
  success: boolean;
  realised_depv: number;
}

export interface DecisionSummary {
  n_passes: number;
  n_scored?: number;
  mean_decision_score: number | null;
  mean_score_minmax?: number | null;
  mean_ratio_score?: number | null;
  mean_regret?: number | null;
  optimal_choice_rate: number | null;
  mean_chosen_depv: number | null;
  mean_max_depv: number | null;
  mean_realised_depv: number | null;
  pass_success_rate: number | null;
}

export interface FatigueBin {
  bin: number;
  label: string;
  minutes_on_pitch: number;
  area_m2: number | null;
  radius_m: number | null;
  n_samples: number;
  vaep_per_action: number | null;
  n_actions: number;
  decision_score: number | null;
  n_passes: number;
  distance_m: number;
  mean_speed: number | null;
  radius_change_pct: number | null;
  vaep_change: number | null;
}

export interface ChemistryPlayer {
  slot: number;
  name: string;
  jersey: string;
  player_id: string | null;
}

export interface Duo {
  passer: string;
  receiver: string;
  passer_slot: number;
  receiver_slot: number;
  passes: number;
  uplift: number;
  total_uplift: number;
}

export interface Chemistry {
  side?: Side;
  min_passes?: number;
  players: ChemistryPlayer[];
  matrix: (number | null)[][];
  counts: number[][];
  top_duos: Duo[];
  note?: string;
}

export interface DecoyRun {
  period: number;
  match_minute: number;
  frame_num: number;
  duration_s: number;
  mean_speed: number;
  delta_das: number;
}

export interface DecoyReport {
  n_decoy_runs: number;
  total_delta_das?: number;
  mean_delta_das?: number | null;
  positive_share?: number | null;
  top_runs?: DecoyRun[];
  note?: string;
}

export interface PressReport {
  threshold?: number;
  /** an action counts as 'under pressure' when an opponent is closer than this many metres */
  radius_m?: number;
  mean_nearest_opp_m?: number | null;
  n_actions: number;
  n_pressured?: number;
  pressured_share?: number | null;
  success_rate_under_pressure?: number | null;
  success_rate_free?: number | null;
  value_generated?: number;
  rating?: number | null;
  /** v1.7: smooth pressure score 0..1 (0 = nobody within 5 m, 1 = an opponent on top of him), averaged over his actions */
  mean_pressure?: number | null;
  pressure_levels?: { level: 'low' | 'medium' | 'high'; n: number; success_rate: number | null }[];
  radius_sensitivity?: { radius_m: number; n_pressured: number; share: number | null }[];
  /** 95 % interval of the success rate under pressure */
  success_rate_under_pressure_ci?: [number, number] | null;
  /** false when he had fewer than 15 pressured actions: the percentages are only a rough guide */
  reliable?: boolean;
}

export interface PlayerReport {
  player: {
    slot: number;
    player_id: string | null;
    name: string;
    jersey: string;
    side: Side;
    team: string;
    position: string | null;
    minutes_tracked: number;
  };
  decision_efficiency: { summary: DecisionSummary; passes: DecisionRow[] };
  fatigue_decay: { bins: FatigueBin[]; baseline_bin: string | null; note: string };
  chemistry: Chemistry;
  decoy_run_index: DecoyReport;
  press_resistance: PressReport;
}

export interface AnalyticsStatus {
  match_id: string;
  state: string;
  ready: Record<string, boolean>;
  error: string | null;
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal: signal ?? null });
  if (!res.ok) {
    let detail = '';
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail ?? '';
    } catch {
      /* body was not JSON */
    }
    throw new Error(`${res.status} ${res.statusText}${detail ? ` - ${detail}` : ''}`);
  }
  return (await res.json()) as T;
}

// ---- matches on the server, and sending files to it ----
export interface MatchDetail {
  match_id: string;
  tracking_bytes: number | null;
  has_events: boolean;
  loaded: boolean;
  analytics_state: string;
}

export interface MatchList {
  matches: string[];
  details: MatchDetail[];
  upload_enabled: boolean;
}

/** The match id written in a PFF tracking file (first line: gameRefId), or the digits in its file name. */
export async function readTrackingMatchId(file: File): Promise<string | null> {
  try {
    const firstLine = (await file.slice(0, 2_000_000).text()).split('\n', 1)[0] ?? '';
    const value = (JSON.parse(firstLine) as { gameRefId?: unknown }).gameRefId;
    if (value !== null && value !== undefined && /^\d+(\.0)?$/.test(String(value))) return String(Math.trunc(Number(value)));
  } catch {
    /* not parseable here: fall back to the file name */
  }
  return /(\d{3,})/.exec(file.name)?.[1] ?? null;
}

export interface UploadParts {
  tracking: File | null;
  events: File | null;
  metadata: File | null;
}

/** POSTs the given files to the analytics server (XHR so that we can show upload progress). */
export function uploadMatch(parts: UploadParts, matchId: string | null, onProgress: (fraction: number) => void, signal?: AbortSignal): Promise<{ match_id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/api/matches/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: { detail?: string; match_id?: string } = {};
      try {
        body = JSON.parse(xhr.responseText) as typeof body;
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body.match_id) resolve({ match_id: body.match_id });
      else reject(new Error(`${xhr.status} ${xhr.statusText}${body.detail ? ` - ${body.detail}` : ''}`));
    };
    xhr.onerror = () => reject(new TypeError('network error'));
    signal?.addEventListener('abort', () => {
      xhr.abort();
      reject(new DOMException('aborted', 'AbortError'));
    });
    const form = new FormData();
    if (parts.tracking) form.append('tracking', parts.tracking);
    if (parts.events) form.append('events', parts.events);
    if (parts.metadata) form.append('metadata', parts.metadata);
    if (matchId) form.append('match_id', matchId);
    xhr.send(form);
  });
}

export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError';
}

export function errorMessage(e: unknown): string {
  if (e instanceof TypeError) {
    return `Cannot reach the analytics API at ${API_BASE}. Start it with:  cd apps/api  then  uvicorn main:app --port 8000`;
  }
  return e instanceof Error ? e.message : String(e);
}

/** Adds ?match_id=... (or &match_id=...) to a path. */
export function withMatch(path: string, matchId: string): string {
  return `${path}${path.includes('?') ? '&' : '?'}match_id=${encodeURIComponent(matchId)}`;
}

// ---- small formatting helpers ----
export function fmt(n: number | null | undefined, digits = 3): string {
  return n === null || n === undefined || !Number.isFinite(n) ? '–' : n.toFixed(digits);
}

export function signed(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '–';
  return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}`;
}

export function pct(n: number | null | undefined, digits = 0): string {
  return n === null || n === undefined || !Number.isFinite(n) ? '–' : `${(n * 100).toFixed(digits)}%`;
}

export function mmss(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// ---- live player statistics (what had each player done by this moment?) ----
export interface LiveStats {
  slot: number;
  player_id: string | null;
  name: string;
  jersey: string;
  side: Side;
  team: string;
  position: string | null;
  minutes: number;
  on_pitch: boolean;
  x: number | null;
  y: number | null;
  speed_kmh: number | null;
  // passing
  passes: number;
  passes_ok: number;
  pass_acc: number | null;
  progressive_passes: number;
  final_third_passes: number;
  box_passes: number;
  key_passes: number;
  assists: number;
  xa: number;
  lines_broken: number;
  crosses: number;
  crosses_ok: number;
  cross_acc: number | null;
  receptions: number;
  receptions_pressured: number;
  carries: number;
  // shooting
  shots: number;
  goals: number;
  shots_on_target: number;
  shots_saved: number;
  shots_off: number;
  shots_blocked: number;
  shots_other: number;
  goals_ruled_out: number;
  xg: number;
  // defending and duels
  tackles: number;
  tackles_won: number;
  aerials: number;
  aerials_won: number;
  dribbles: number;
  dribbles_won: number;
  ground_duels: number;
  ground_duels_won: number;
  duels: number;
  duels_won: number;
  duel_win_pct: number | null;
  interceptions: number;
  blocks: number;
  clearances: number;
  defensive_actions: number;
  on_ball_actions: number;
  cautions: number;
  // goalkeeper
  saves: number;
  goals_conceded: number;
  shots_on_target_faced: number;
  save_pct: number | null;
  // physical
  distance_m: number;
  hsr_m: number;
  sprint_m: number;
  sprints: number;
  hsr_runs: number;
  top_speed_kmh: number;
  // model values
  value_actions: number;
  vaep_oa: number;
  vaep_raw: number;
  obv: number;
}

export interface LiveTeamTotals {
  name: string | null;
  passes: number;
  passes_ok: number;
  pass_acc: number | null;
  shots: number;
  shots_on_target: number;
  goals: number;
  xg: number;
  distance_m: number;
}

export interface LivePlayersResponse {
  match_id: string;
  frame_num: number;
  period: number;
  minute: number;
  players: LiveStats[];
  teams: { home: LiveTeamTotals; away: LiveTeamTotals };
  ball: { x: number; y: number } | null;
  has_events: boolean;
  compute_ms: number;
}

export interface LivePass {
  clock: string;
  minute: number;
  frame_num: number;
  x: number;
  y: number;
  ex: number | null;
  ey: number | null;
  ok: boolean;
  key: boolean;
  assist: boolean;
  cross: boolean;
  to: string | null;
}

export interface LiveShot {
  clock: string;
  minute: number;
  frame_num: number;
  x: number;
  y: number;
  ex: number | null;
  ey: number | null;
  xg: number;
  cat: string;
  label: string;
}

export interface LiveDefending {
  clock: string;
  minute: number;
  frame_num: number;
  x: number;
  y: number;
  type: string;
}

export interface LiveDuel {
  clock: string;
  minute: number;
  frame_num: number;
  x: number;
  y: number;
  type: string;
  won: boolean;
}

export interface RankEntry {
  label: string;
  value: number | null;
  avg: number | null;
  n: number;
  rank: number | null;
}

export interface SummaryLine {
  tone: 'good' | 'ok' | 'poor' | 'info';
  text: string;
}

export interface LiveDetail {
  match_id: string;
  frame_num: number;
  period: number;
  minute: number;
  stats: LiveStats;
  now: { x: number; y: number; speed_kmh: number | null } | null;
  heat: {
    nx: number;
    ny: number;
    cells: number[][];
    avg: [number, number] | null;
    thirds: [number, number, number] | null;
    width: number | null;
  };
  maps: { passes: LivePass[]; shots: LiveShot[]; defending: LiveDefending[]; duels: LiveDuel[] };
  team: Record<string, RankEntry>;
  /** who he is ranked against, e.g. 'midfielders of his team' */
  team_pool: { size: number; label: string };
  summary: SummaryLine[];
  has_events: boolean;
  events_note: string | null;
  compute_ms: number;
}

export interface TimelineRow {
  /** minutes of play so far, in the order they were played (the match clock restarts in the 2nd half, so it cannot be an axis) */
  k: number;
  period: number;
  match_minute: number;
  frame_num: number;
  on_pitch: boolean;
  distance_km: number;
  hsr_m: number;
  sprint_m: number;
  passes: number;
  passes_ok: number;
  shots: number;
  value: number;
  defensive: number;
}

export interface TimelineMoment {
  k: number;
  period: number;
  match_minute: number;
  clock: string;
  frame_num: number;
  tone: 'goal' | 'shot' | 'assist' | 'key' | 'card' | 'save' | 'def';
  label: string;
}

export interface LiveTimeline {
  slot?: number;
  minutes: TimelineRow[];
  moments: TimelineMoment[];
  period_starts?: number[];
  first_k?: number;
  last_k?: number;
}

/** 45+3' style clock from the period and the (period-aware) match minute */
export function clockLabel(period: number, minute: number): string {
  const end = period === 1 ? 45 : period === 2 ? 90 : period === 3 ? 105 : 120;
  if (minute > end) return `${end}+${Math.ceil(minute - end)}'`;
  return `${Math.max(1, Math.ceil(minute))}'`;
}

// ---- Top attacks (attacks of the match with a 1-100 danger score, and the replay data of one attack) ----
export interface AttackShot {
  event_idx: number;
  frame_num: number;
  /** seconds from the start of the attack */
  t: number;
  player: string;
  slot: number;
  xg: number;
  penalty: boolean;
  goal: boolean;
  /** the ball went in but the goal was ruled out (offside / VAR): its xG does not count */
  disallowed: boolean;
  outcome: string;
  distance_m: number;
  x: number;
  y: number;
  header: boolean;
  /** outfield opponents inside the triangle ball - post - post */
  defenders_in_way: number | null;
  keeper_from_goal_m: number | null;
  /** xG of 0.38 or more, or a penalty */
  big_chance: boolean;
}

export interface AttackRow {
  id: number;
  side: Side;
  team: string;
  period: number;
  minute: number;
  start_frame: number;
  end_frame: number;
  replay_start_frame: number;
  replay_end_frame: number;
  key_frame: number;
  duration_s: number;
  /** StatsBomb-style play pattern: Regular play, Counter-attack, Corner, Free kick, Penalty, Throw-in, Goal kick, Kick-off, Drop ball */
  pattern: string;
  set_piece: boolean;
  n_events: number;
  passes: number;
  passes_completed: number;
  players_involved: number;
  start_x: number;
  max_x: number;
  reached_final_third: boolean;
  entered_box: boolean;
  shots: AttackShot[];
  n_shots: number;
  /** chance that at least one counted shot went in */
  xg: number;
  goal: boolean;
  disallowed_goal: boolean;
  /** highest expected threat at a spot the team controlled */
  peak_xt: number;
  xt_gained: number;
  /** max(xg, peak_xt) */
  danger_p: number;
  /** 1-100 on a fixed logarithmic scale */
  danger: number;
  danger_label: string;
  danger_source: 'shot' | 'position';
  /** Opta / StatsBomb-style labels: Big chance, Fast break, High turnover, Direct attack, Build-up */
  tags: string[];
  /** share of the ball's travel that went towards goal (0..1) */
  directness: number;
  /** metres gained towards goal per second (Opta's direct speed) */
  direct_speed: number;
  progress_m: number;
  /** raw EPV of the best controlled spot (before the 0.852 scale) */
  peak_epv_raw: number;
  /** share of that spot the team controlled (pitch control), shown as context */
  peak_control: number;
  /** best controlled spot BEFORE the first shot (used for the non-circular check) */
  pre_shot_xt: number;
  das: number | null;
  obso: number | null;
  defenders_goal_side: number | null;
  story: string;
  rank_goals_first: number;
  rank_danger: number;
  rank_goals_first_team: number;
  rank_danger_team: number;
}

export interface AttackTeamSummary {
  team: string | null;
  attacks: number;
  shots: number;
  goals: number;
  dangerous: number;
  xg: number;
  mean_danger: number;
}

export interface AttacksResponse {
  match_id: string;
  version: string;
  compute_ms: number;
  teams: { home: string | null; away: string | null };
  scale: { p_min: number; p_max: number; bands: { from: number; label: string }[] };
  summary: { home: AttackTeamSummary; away: AttackTeamSummary };
  attacks: AttackRow[];
}

export interface AttackTrackFrame {
  f: number;
  /** seconds from the start of the attack (negative = lead-in) */
  t: number;
  b: [number, number] | null;
  /** [slot, x, y] */
  p: [number, number, number][];
}

export interface AttackTrackEvent {
  t: number;
  frame_num: number;
  type: string;
  side: Side | null;
  slot: number;
  player: string;
  x: number | null;
  y: number | null;
  end_x: number | null;
  end_y: number | null;
  completed: boolean;
  in_attack: boolean;
}

export interface AttackTrack {
  attack: AttackRow;
  fps: number;
  step: number;
  frames: AttackTrackFrame[];
  /** [t, xT of the ball] */
  xt: [number, number][];
  events: AttackTrackEvent[];
  players: { slot: number; side: Side; jersey: string; name: string }[];
  attack_dir: number;
  pitch: { length: number; width: number };
}


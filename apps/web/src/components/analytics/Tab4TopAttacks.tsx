import { Pause, Play, RotateCcw } from 'lucide-react';
import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { useUiStore } from '../../store';
import { drawPitch } from './AnalyticsPitch';
import {
  clockLabel,
  errorMessage,
  getJson,
  isAbort,
  withMatch,
  type AttackRow,
  type AttacksResponse,
  type AttackTrack,
  type AttackTrackFrame,
  type FramePayload,
  type MatchInfo,
  type Side
} from './api';
import { ErrorBox, Panel, Spinner, Stat } from './ui';

// ---------------------------------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------------------------------
type Scope = 'match' | Side;
type Order = 'goals' | 'danger' | 'chances';

const BANDS = [
  { from: 75, label: 'Very dangerous', color: '#f43f5e' },
  { from: 55, label: 'Dangerous', color: '#f59e0b' },
  { from: 35, label: 'Promising', color: '#c8f45a' },
  { from: 1, label: 'Low threat', color: '#64748b' }
] as const;

function band(score: number) {
  return BANDS.find((b) => score >= b.from) ?? BANDS[BANDS.length - 1]!;
}

const ACTION_NAME: Record<string, string> = {
  PA: 'Pass',
  CR: 'Cross',
  SH: 'Shot',
  BC: 'Carry',
  CH: 'Duel',
  CL: 'Clearance',
  RE: 'Rebound',
  IT: 'Receives',
  TC: 'Touch'
};

function rgba(hex: string, a: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m || !m[1]) return `rgba(255,255,255,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function sortAttacks(list: AttackRow[], order: Order): AttackRow[] {
  if (order === 'chances') {
    // "Best chances": only attacks with a counted shot, ranked by shot quality alone
    return list.filter((a) => a.xg > 0).sort((a, b) => b.xg - a.xg || b.danger_p - a.danger_p);
  }
  return [...list].sort((a, b) =>
    order === 'goals'
      ? Number(b.goal) - Number(a.goal) || b.danger_p - a.danger_p || b.xt_gained - a.xt_gained
      : b.danger_p - a.danger_p || b.xt_gained - a.xt_gained
  );
}

const TAG_COLOR: Record<string, string> = {
  'Big chance': '#fde047',
  'Fast break': '#38bdf8',
  'High turnover': '#a78bfa',
  'Direct attack': '#fb923c',
  'Build-up': '#34d399'
};
const TAG_HELP: Record<string, string> = {
  'Big chance': 'A shot with xG of 0.38 or more, or a penalty (0.38 is the typical value of an Opta big chance; the threshold is ours).',
  'Fast break': 'Ball won in the own half and the final third reached within 12 seconds.',
  'High turnover': 'Ball won from the opponent 40 m or less from their goal (Opta definition).',
  'Direct attack': 'Starts in the own half, at least half of the ball movement goes towards goal, ends in a shot or the box (Opta definition).',
  'Build-up': '10 or more passes ending in a shot or the box (Opta definition).'
};
function Tags({ tags }: { tags: string[] }) {
  return (
    <>
      {tags.map((t) => (
        <span key={t} title={TAG_HELP[t] ?? t}>
          <Chip color={TAG_COLOR[t] ?? '#93a0a8'}>{t}</Chip>
        </span>
      ))}
    </>
  );
}

/** circular gauge with the 1-100 danger score */
function DangerGauge({ score, size = 54, label = false }: { score: number; size?: number; label?: boolean }) {
  const b = band(score);
  const r = size / 2 - 5;
  const c = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center" title={`Danger ${score}/100 · ${b.label}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-label={`Danger ${score} out of 100`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" strokeWidth={5} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={b.color}
          strokeWidth={5}
          strokeLinecap="round"
          strokeDasharray={`${(c * score) / 100} ${c}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ filter: `drop-shadow(0 0 6px ${rgba(b.color, 0.55)})` }}
        />
        <text x="50%" y="50%" dominantBaseline="central" textAnchor="middle" fill="#fff" fontWeight={800} fontSize={size * 0.34} fontFamily="ui-monospace, monospace">
          {score}
        </text>
      </svg>
      {label ? <span className="mt-1 text-[10px] font-bold uppercase tracking-wider" style={{ color: b.color }}>{b.label}</span> : null}
    </div>
  );
}

function Chip({ children, color }: { children: ReactNode; color?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
      style={color ? { borderColor: rgba(color, 0.5), color, background: rgba(color, 0.08) } : { borderColor: '#2b3339', color: '#93a0a8' }}
    >
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// danger timeline: every attack of the match, home above the line, away below
// ---------------------------------------------------------------------------------------------------------------------
interface TimelinePoint {
  id: number;
  x: number;
  y: number;
  danger: number;
  goal: boolean;
  side: Side;
  clock: string;
  story: string;
}

const DangerTimeline = memo(function DangerTimeline({
  attacks,
  info,
  selected,
  onSelect,
  homeColor,
  awayColor
}: {
  attacks: AttackRow[];
  info: MatchInfo;
  selected: number | null;
  onSelect: (id: number) => void;
  homeColor: string;
  awayColor: string;
}) {
  const firstHalfMin = (info.periods[0]?.duration_s ?? 45 * 60) / 60;
  const toX = (a: AttackRow) => (a.period <= 1 ? a.minute : firstHalfMin + 1 + (a.minute - 45));
  const tick = (x: number) => (x <= firstHalfMin ? `${Math.round(x)}'` : `${Math.round(45 + x - firstHalfMin - 1)}'`);
  const pts = (side: Side): TimelinePoint[] =>
    attacks
      .filter((a) => a.side === side)
      .map((a) => ({ id: a.id, x: toX(a), y: side === 'home' ? a.danger : -a.danger, danger: a.danger, goal: a.goal, side, clock: clockLabel(a.period, a.minute), story: a.story }));
  const maxX = Math.max(firstHalfMin + 47, ...attacks.map(toX));
  const shape = (color: string) =>
    function Dot(props: unknown) {
      const { cx, cy, payload } = props as { cx?: number; cy?: number; payload?: TimelinePoint };
      if (cx === undefined || cy === undefined || !payload) return <g />;
      const r = 2.5 + payload.danger / 22;
      const sel = payload.id === selected;
      return (
        <g style={{ cursor: 'pointer' }}>
          {payload.goal ? <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="#fde047" strokeWidth={2} /> : null}
          {sel ? <circle cx={cx} cy={cy} r={r + 8} fill="none" stroke="#c8f45a" strokeWidth={2} /> : null}
          <circle cx={cx} cy={cy} r={r} fill={rgba(color, 0.35 + payload.danger / 170)} stroke={color} strokeWidth={1} />
        </g>
      );
    };
  const click = (pt: unknown) => {
    const id = (pt as { payload?: { id?: number } } | undefined)?.payload?.id;
    if (typeof id === 'number') onSelect(id);
  };
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" />
          <XAxis type="number" dataKey="x" domain={[0, Math.ceil(maxX)]} tickFormatter={tick} stroke="#93a0a8" fontSize={11} ticks={[0, 15, 30, 45, firstHalfMin + 1 + 15, firstHalfMin + 1 + 30, firstHalfMin + 1 + 45]} />
          <YAxis type="number" dataKey="y" domain={[-100, 100]} ticks={[-100, -50, 0, 50, 100]} tickFormatter={(v: number) => String(Math.abs(v))} stroke="#93a0a8" fontSize={11} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.25)" />
          <ReferenceLine x={firstHalfMin + 0.5} stroke="rgba(255,255,255,0.18)" strokeDasharray="4 4" label={{ value: 'HT', fill: '#93a0a8', fontSize: 10, position: 'insideTop' }} />
          <ReferenceLine y={55} stroke="rgba(245,158,11,0.25)" strokeDasharray="3 5" />
          <ReferenceLine y={-55} stroke="rgba(245,158,11,0.25)" strokeDasharray="3 5" />
          <Tooltip
            cursor={false}
            content={({ active, payload }) => {
              const p = active ? (payload?.[0]?.payload as TimelinePoint | undefined) : undefined;
              if (!p) return null;
              return (
                <div className="max-w-xs rounded-lg border border-line bg-ink/95 px-3 py-2 text-xs text-slate-200 shadow-panel">
                  <div className="font-bold text-white">
                    {p.clock} · danger {p.danger}/100 {p.goal ? '⚽' : ''}
                  </div>
                  <div className="mt-0.5 text-muted">{p.story}</div>
                  <div className="mt-1 text-[10px] text-lime">Click to watch</div>
                </div>
              );
            }}
          />
          <Scatter name="home" data={pts('home')} shape={shape(homeColor)} onClick={click} isAnimationActive={false} />
          <Scatter name="away" data={pts('away')} shape={shape(awayColor)} onClick={click} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
});

// ---------------------------------------------------------------------------------------------------------------------
// replay pitch: players, ball, ball trail, passes, shots (+ optional pitch control when paused)
// ---------------------------------------------------------------------------------------------------------------------
function frameAt(frames: AttackTrackFrame[], t: number): { a: AttackTrackFrame; b: AttackTrackFrame; k: number } | null {
  if (!frames.length) return null;
  const first = frames[0]!;
  const last = frames[frames.length - 1]!;
  if (t <= first.t) return { a: first, b: first, k: 0 };
  if (t >= last.t) return { a: last, b: last, k: 0 };
  const dt = frames.length > 1 ? (last.t - first.t) / (frames.length - 1) : 1;
  const i = Math.min(frames.length - 2, Math.max(0, Math.floor((t - first.t) / dt)));
  const a = frames[i]!;
  const b = frames[i + 1]!;
  const k = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 0;
  return { a, b, k };
}

function AttackPitch({
  track,
  t,
  homeColor,
  awayColor,
  homeText,
  awayText,
  showNumbers,
  pcGrid
}: {
  track: AttackTrack;
  t: number;
  homeColor: string;
  awayColor: string;
  homeText: string;
  awayText: string;
  showNumbers: boolean;
  pcGrid: number[][] | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(820);
  const L = track.pitch.length;
  const W = track.pitch.width;
  const h = Math.round((w * W) / L);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      if (e) setW(Math.max(320, Math.floor(e.contentRect.width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const sideOf = useMemo(() => new Map(track.players.map((p) => [p.slot, p])), [track]);
  const involved = useMemo(() => new Set(track.events.filter((e) => e.in_attack && e.side === track.attack.side && e.slot >= 0).map((e) => e.slot)), [track]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const sx = w / L;
    const sy = h / W;
    const px = (x: number) => (x + L / 2) * sx;
    const py = (y: number) => (W / 2 - y) * sy;
    const attSide = track.attack.side;
    const attColor = attSide === 'home' ? homeColor : awayColor;

    drawPitch(ctx, w, h, L, W);

    // optional pitch control (only when paused): smooth, like the Core metrics view
    if (pcGrid && pcGrid.length && pcGrid[0]?.length) {
      const ny = pcGrid.length;
      const nx = pcGrid[0].length;
      const off = document.createElement('canvas');
      off.width = nx;
      off.height = ny;
      const octx = off.getContext('2d');
      if (octx) {
        for (let iy = 0; iy < ny; iy++) {
          const row = pcGrid[iy];
          if (!row) continue;
          for (let ix = 0; ix < nx; ix++) {
            const v = row[ix];
            if (v === undefined) continue;
            const a = Math.min(0.55, Math.max(0, Math.abs(v - 0.5) - 0.01) * 1.3);
            if (a < 0.01) continue;
            octx.fillStyle = rgba(v >= 0.5 ? homeColor : awayColor, a);
            octx.fillRect(ix, ny - 1 - iy, 1, 1);
          }
        }
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(off, 0, 0, nx, ny, 0, 0, w, h);
        ctx.restore();
      }
    }

    // attack direction
    ctx.fillStyle = rgba(attColor, 0.9);
    ctx.font = '700 11px ui-sans-serif, system-ui';
    ctx.textAlign = track.attack_dir > 0 ? 'right' : 'left';
    ctx.fillText(`${track.attack.team} attack ${track.attack_dir > 0 ? '→' : '←'}`, track.attack_dir > 0 ? w - 10 : 10, h - 10);
    ctx.textAlign = 'left';

    // ball trail from the start of the attack to now
    const trail = track.frames.filter((f) => f.b && f.t >= -0.2 && f.t <= t);
    if (trail.length > 1) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (let i = 1; i < trail.length; i++) {
        const p0 = trail[i - 1]!;
        const p1 = trail[i]!;
        if (!p0.b || !p1.b) continue;
        if (Math.hypot(p1.b[0] - p0.b[0], p1.b[1] - p0.b[1]) > 8) continue; // a jump in the data, not a movement
        const age = (t - p1.t) / Math.max(1, t - trail[0]!.t);
        ctx.strokeStyle = rgba('#fde047', 0.25 + 0.65 * (1 - age));
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(px(p0.b[0]), py(p0.b[1]));
        ctx.lineTo(px(p1.b[0]), py(p1.b[1]));
        ctx.stroke();
      }
    }

    // passes made so far (numbered), and shots
    let n = 0;
    let nShot = 0;
    const labels: { x: number; y: number; txt: string }[] = []; // drawn last, on top of the players
    for (const e of track.events) {
      if (!e.in_attack || e.t > t || e.x === null || e.y === null) continue;
      if ((e.type === 'PA' || e.type === 'CR') && e.end_x !== null && e.end_y !== null && e.side === attSide) {
        n += 1;
        const x0 = px(e.x);
        const y0 = py(e.y);
        const x1 = px(e.end_x);
        const y1 = py(e.end_y);
        ctx.strokeStyle = e.completed ? 'rgba(255,255,255,0.75)' : 'rgba(248,113,113,0.85)';
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.setLineDash([]);
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = 'rgba(8,11,13,0.85)';
        ctx.beginPath();
        ctx.arc((x0 + x1) / 2, (y0 + y1) / 2, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '700 10px ui-sans-serif, system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(n), (x0 + x1) / 2, (y0 + y1) / 2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      }
      if (e.type === 'SH' && e.side === attSide) {
        const shot = track.attack.shots.find((s) => s.frame_num === e.frame_num);
        const gx = px((L / 2) * track.attack_dir);
        const gy = py(0);
        const x0 = px(e.x);
        const y0 = py(e.y);
        ctx.strokeStyle = shot?.goal ? '#fde047' : 'rgba(253,224,71,0.6)';
        ctx.lineWidth = shot?.goal ? 3 : 2;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(gx, gy);
        ctx.stroke();
        // star
        ctx.fillStyle = '#fde047';
        ctx.beginPath();
        for (let k = 0; k < 10; k++) {
          const r = k % 2 === 0 ? 9 : 4;
          const a = -Math.PI / 2 + (k * Math.PI) / 5;
          const xx = x0 + r * Math.cos(a);
          const yy = y0 + r * Math.sin(a);
          if (k === 0) ctx.moveTo(xx, yy);
          else ctx.lineTo(xx, yy);
        }
        ctx.closePath();
        ctx.fill();
        if (shot) {
          const txt = `${shot.goal ? 'GOAL · ' : shot.disallowed ? 'RULED OUT · ' : ''}xG ${shot.xg.toFixed(2)}`;
          const up = y0 - 26 - 22 * (nShot % 3); // several shots close together: stack the labels
          const ly = up < 4 ? y0 + 12 + 22 * (nShot % 3) : up; // near the top edge: stack them below the shot instead
          labels.push({ x: x0 + 10, y: ly, txt });
          nShot += 1;
        }
      }
    }

    // players and ball (interpolated between samples)
    const fr = frameAt(track.frames, t);
    if (fr) {
      const { a, b, k } = fr;
      const bpos = new Map(b.p.map((q) => [q[0], q] as const));
      for (const q of a.p) {
        const [slot, x0, y0] = q;
        const nb = bpos.get(slot);
        const x = nb ? x0 + (nb[1] - x0) * k : x0;
        const y = nb ? y0 + (nb[2] - y0) * k : y0;
        const info = sideOf.get(slot);
        const home = info?.side === 'home';
        const fill = home ? homeColor : awayColor;
        const text = home ? homeText : awayText;
        const cx = px(x);
        const cy = py(y);
        if (involved.has(slot)) {
          ctx.strokeStyle = '#c8f45a';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(cx, cy, 13, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.fillStyle = fill;
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, 9.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (showNumbers && info) {
          ctx.fillStyle = text;
          ctx.font = '800 10px ui-sans-serif, system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(info.jersey, cx, cy + 0.5);
          ctx.textAlign = 'left';
          ctx.textBaseline = 'alphabetic';
        }
      }
      const ba = a.b;
      const bb = b.b ?? a.b;
      if (ba && bb) {
        const x = ba[0] + (bb[0] - ba[0]) * k;
        const y = ba[1] + (bb[1] - ba[1]) * k;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px(x), py(y), 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // shot labels on top of everything, kept inside the pitch
    ctx.font = '800 12px ui-sans-serif, system-ui';
    for (const lb of labels) {
      const tw = ctx.measureText(lb.txt).width;
      const lx = Math.min(lb.x, w - tw - 14);
      ctx.fillStyle = 'rgba(8,11,13,0.9)';
      ctx.fillRect(lx, lb.y, tw + 10, 18);
      ctx.strokeStyle = 'rgba(253,224,71,0.6)';
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, lb.y, tw + 10, 18);
      ctx.fillStyle = '#fde047';
      ctx.fillText(lb.txt, lx + 5, lb.y + 13);
    }

    // phase label
    const phase = t < 0 ? 'Before the attack' : t > track.attack.duration_s ? 'After the attack' : 'Attack in progress';
    ctx.fillStyle = 'rgba(8,11,13,0.7)';
    ctx.fillRect(8, 8, 190, 22);
    ctx.fillStyle = t < 0 || t > track.attack.duration_s ? '#93a0a8' : '#c8f45a';
    ctx.font = '700 11px ui-sans-serif, system-ui';
    ctx.fillText(`${phase} · ${t >= 0 ? '+' : ''}${t.toFixed(1)} s`, 15, 23);
  }, [track, t, w, h, L, W, homeColor, awayColor, homeText, awayText, showNumbers, pcGrid, sideOf, involved]);

  return (
    <div ref={wrapRef} className="w-full overflow-hidden rounded-xl border border-line">
      <canvas ref={canvasRef} className="block" aria-label="Replay of the selected attack" />
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// the selected attack: replay, xT curve, why this score
// ---------------------------------------------------------------------------------------------------------------------
function AttackDetail({ matchId, attack, rank }: { matchId: string; attack: AttackRow; rank: number }) {
  const homeColor = useUiStore((s) => s.homeColor);
  const homeText = useUiStore((s) => s.homeTextColor);
  const awayColor = useUiStore((s) => s.awayColor);
  const awayText = useUiStore((s) => s.awayTextColor);
  const showNumbers = useUiStore((s) => s.showNumbers);
  const [track, setTrack] = useState<AttackTrack | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showPc, setShowPc] = useState(false);
  const [pcGrid, setPcGrid] = useState<number[][] | null>(null);
  const tRef = useRef(0);

  useEffect(() => {
    const ctrl = new AbortController();
    setTrack(null);
    setError(null);
    setPlaying(false);
    getJson<AttackTrack>(withMatch(`/api/attacks/${attack.id}/track`, matchId), ctrl.signal)
      .then((tr) => {
        setTrack(tr);
        const t0 = tr.frames[0]?.t ?? 0;
        tRef.current = t0;
        setT(t0);
        setPlaying(true);
      })
      .catch((e: unknown) => {
        if (!isAbort(e)) setError(errorMessage(e));
      });
    return () => ctrl.abort();
  }, [matchId, attack.id]);

  // playback
  useEffect(() => {
    if (!playing || !track) return;
    const tEnd = track.frames[track.frames.length - 1]?.t ?? 0;
    let raf = 0;
    let last = performance.now();
    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const nt = Math.min(tEnd, tRef.current + dt * speed);
      tRef.current = nt;
      setT(nt);
      if (nt >= tEnd) {
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, track, speed]);

  // pitch control of the current moment (only when paused)
  useEffect(() => {
    if (!showPc || playing || !track) {
      setPcGrid(null);
      return;
    }
    const fr = frameAt(track.frames, t);
    if (!fr) return;
    const ctrl = new AbortController();
    const id = window.setTimeout(() => {
      getJson<FramePayload>(withMatch(`/api/metrics/frame/${fr.a.f}`, matchId), ctrl.signal)
        .then((p) => setPcGrid(p.pitch_control_home))
        .catch(() => setPcGrid(null));
    }, 200);
    return () => {
      window.clearTimeout(id);
      ctrl.abort();
    };
  }, [showPc, playing, track, t, matchId]);

  const seek = (nt: number) => {
    tRef.current = nt;
    setT(nt);
  };
  const tStart = track?.frames[0]?.t ?? -3;
  const tEnd = track?.frames[track.frames.length - 1]?.t ?? attack.duration_s + 2;
  const b = band(attack.danger);
  const teamColor = attack.side === 'home' ? homeColor : awayColor;
  const xtData = useMemo(() => (track ? track.xt.map(([tt, v]) => ({ t: tt, xt: v })) : []), [track]);
  const xtMax = Math.max(0.05, ...xtData.map((d) => d.xt)) * 1.15;
  const tChart = Math.round(t * 10) / 10;
  const inAttack = useMemo(() => (track ? track.events.filter((e) => e.in_attack) : []), [track]);
  const xtAt = (tt: number) => {
    let best = 0;
    let bd = Infinity;
    for (const d of xtData) {
      const dd = Math.abs(d.t - tt);
      if (dd < bd) {
        bd = dd;
        best = d.xt;
      }
    }
    return best;
  };
  const shotPct = attack.xg * 100;
  const bestShot = attack.shots.filter((s) => !s.disallowed).sort((x, y) => y.xg - x.xg)[0];
  const shotHint = !attack.n_shots
    ? 'no shot'
    : !bestShot
      ? 'the only goal was ruled out (counts 0)'
      : bestShot.penalty
        ? 'penalty (0.79)'
        : `${attack.n_shots} shot${attack.n_shots > 1 ? 's' : ''}; best: ${bestShot.distance_m.toFixed(0)} m, ${bestShot.header ? 'header' : 'foot'}, ${bestShot.defenders_in_way ?? '–'} defender${bestShot.defenders_in_way === 1 ? '' : 's'} in the way, keeper ${bestShot.keeper_from_goal_m ?? '–'} m off his goal`;
  const xtPct = attack.peak_xt * 100;
  const pctMax = Math.max(shotPct, xtPct, 1);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-line bg-ink/60 p-3" style={{ boxShadow: `inset 4px 0 0 ${teamColor}` }}>
        <DangerGauge score={attack.danger} size={72} label />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-bold text-muted">#{rank}</span>
            <span className="text-base font-bold" style={{ color: teamColor }}>{attack.team}</span>
            <span className="font-mono text-sm text-white">{clockLabel(attack.period, attack.minute)}</span>
            <Chip>{attack.pattern}</Chip>
            {attack.goal ? <Chip color="#fde047">⚽ Goal</Chip> : null}
            {attack.disallowed_goal ? <Chip color="#f87171">Goal ruled out</Chip> : null}
            <Tags tags={attack.tags} />
          </div>
          <p className="mt-1 text-sm leading-snug text-slate-200">{attack.story}</p>
          <p className="mt-1 text-xs text-muted">
            <span className="font-mono font-bold text-white">{(attack.danger_p * 100).toFixed(1)}%</span> chance of a goal · set by the{' '}
            {attack.danger_source === 'shot' ? 'shot' : 'best spot reached'} · {attack.progress_m.toFixed(0)} m upfield at {attack.direct_speed.toFixed(1)} m/s
          </p>
        </div>
      </div>

      {error ? <ErrorBox message={error} /> : null}
      {!track && !error ? <Spinner label="Loading the attack…" /> : null}
      {track ? (
        <>
          <AttackPitch track={track} t={t} homeColor={homeColor} awayColor={awayColor} homeText={homeText} awayText={awayText} showNumbers={showNumbers} pcGrid={pcGrid} />
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                if (!playing && t >= tEnd - 0.05) seek(tStart);
                setPlaying((p) => !p);
              }}
              className="flex h-9 w-9 items-center justify-center rounded-lg bg-lime text-ink"
              aria-label={playing ? 'Pause' : 'Play'}
            >
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
            <button type="button" onClick={() => { seek(tStart); setPlaying(true); }} className="flex h-9 w-9 items-center justify-center rounded-lg border border-line text-slate-200 hover:bg-white/[0.05]" aria-label="Replay from the start">
              <RotateCcw className="h-4 w-4" />
            </button>
            <input
              type="range"
              min={tStart}
              max={tEnd}
              step={0.05}
              value={t}
              onChange={(e) => {
                setPlaying(false);
                seek(Number(e.target.value));
              }}
              className="h-2 flex-1 accent-lime"
              aria-label="Position in the attack"
            />
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="h-9 rounded-lg border border-line bg-panel px-2 text-sm text-slate-200" aria-label="Playback speed">
              <option value={0.25}>0.25×</option>
              <option value={0.5}>0.5×</option>
              <option value={1}>1× real time</option>
              <option value={2}>2×</option>
            </select>
            <button
              type="button"
              onClick={() => {
                const k = track.frames.find((f) => f.f >= attack.key_frame);
                if (k) {
                  setPlaying(false);
                  seek(k.t);
                  setShowPc(true);
                }
              }}
              className="h-9 rounded-lg border border-line px-3 text-xs font-semibold text-slate-200 hover:bg-white/[0.05]"
              title="Pause on the key moment (the best shot, or the most dangerous spot) with pitch control"
            >
              Key moment
            </button>
            <label className="flex items-center gap-1.5 text-xs text-muted" title="Shown when the replay is paused">
              <input type="checkbox" checked={showPc} onChange={(e) => setShowPc(e.target.checked)} className="accent-lime" />
              Pitch control (when paused)
            </label>
          </div>

          <div className="rounded-xl border border-line bg-ink/40 p-2">
            <div className="mb-1 flex items-center justify-between px-1 text-[11px] text-muted">
              <span className="font-semibold uppercase tracking-wider">Expected threat (xT) of the ball during the attack</span>
              <span>★ shot · ● pass · shaded = the attack</span>
            </div>
            <div className="h-32">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={xtData} margin={{ top: 6, right: 10, bottom: 0, left: -18 }} onClick={(s) => { const v = (s as unknown as { activeLabel?: number | string } | null)?.activeLabel; if (v !== undefined) { setPlaying(false); seek(Number(v)); } }}>
                  <defs>
                    <linearGradient id="xtFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={b.color} stopOpacity={0.55} />
                      <stop offset="100%" stopColor={b.color} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="t" type="number" domain={[tStart, tEnd]} tickFormatter={(v: number) => `${v.toFixed(0)}s`} stroke="#93a0a8" fontSize={10} />
                  <YAxis domain={[0, xtMax]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} stroke="#93a0a8" fontSize={10} />
                  <ReferenceArea x1={0} x2={attack.duration_s} fill="rgba(200,244,90,0.06)" />
                  <Area type="monotone" dataKey="xt" stroke={b.color} strokeWidth={2} fill="url(#xtFill)" isAnimationActive={false} />
                  {inAttack.filter((e) => (e.type === 'PA' || e.type === 'CR') && e.side === attack.side).map((e, i) => (
                    <ReferenceDot key={`p${i}`} x={e.t} y={xtAt(e.t)} r={3} fill="#fff" stroke="none" />
                  ))}
                  {attack.shots.map((s) => (
                    <ReferenceDot key={`s${s.event_idx}`} x={s.t} y={xtAt(s.t)} r={6} fill="#fde047" stroke="#000" />
                  ))}
                  <ReferenceLine x={tChart} stroke="#c8f45a" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <div className="rounded-xl border border-line bg-ink/40 p-3">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">Why {attack.danger}/100?</div>
              <div className="mt-2 space-y-2 text-xs">
                {[
                  { name: 'Shot quality (attack xG)', v: shotPct, hint: shotHint },
                  {
                    name: 'Best spot reached (peak xT)',
                    v: xtPct,
                    hint: `chance of a goal from the best spot they controlled (EPV ${attack.peak_epv_raw.toFixed(3)} × 0.852); they controlled ${Math.round(attack.peak_control * 100)}% of that spot`
                  }
                ].map((r) => (
                  <div key={r.name}>
                    <div className="flex justify-between text-slate-200">
                      <span>{r.name}</span>
                      <span className="font-mono font-bold">{r.v.toFixed(1)}%</span>
                    </div>
                    <div className="mt-1 h-2 rounded bg-white/5">
                      <div className="h-2 rounded" style={{ width: `${(100 * r.v) / pctMax}%`, background: r.v >= Math.max(shotPct, xtPct) ? b.color : '#475569' }} />
                    </div>
                    <div className="mt-0.5 text-[10px] text-muted">{r.hint}</div>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-snug text-muted">
                Danger = the larger of the two = <span className="font-bold text-white">{(attack.danger_p * 100).toFixed(1)}% chance of a goal</span>, shown on a fixed scale where 2% = 30, 5% = 48, 10% = 61,
                20% = 74, 40% = 87 and a penalty (79%) = 100.
              </p>
              <div className="relative mt-2 h-2 rounded" style={{ background: 'linear-gradient(90deg,#64748b 0%,#64748b 34%,#c8f45a 35%,#c8f45a 54%,#f59e0b 55%,#f59e0b 74%,#f43f5e 75%)' }}>
                <div className="absolute -top-1 h-4 w-1 rounded bg-white" style={{ left: `calc(${attack.danger}% - 2px)` }} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="xT gained" value={attack.xt_gained.toFixed(3)} hint="sum of forward steps" help="Expected threat added by completed passes and carries (only steps that increased the threat count)." />
              <Stat label="Passes" value={`${attack.passes_completed}/${attack.passes}`} hint={`${attack.players_involved} players · ${attack.duration_s.toFixed(1)} s`} />
              <Stat label="Direct speed" value={`${attack.direct_speed.toFixed(1)} m/s`} hint={`${attack.progress_m.toFixed(0)} m upfield · ${Math.round(attack.directness * 100)}% direct`} help="Opta's direct speed: metres gained towards goal per second. Directness = share of the ball's travel that went towards goal." />
              <Stat label="Defenders goal-side" value={attack.defenders_goal_side === null ? '–' : String(attack.defenders_goal_side)} hint="at the key moment" help="Opponents still between the ball and their goal at the key moment (the best shot, or the most dangerous spot). Fewer = more dangerous." />
              <Stat label="Dangerous space" value={attack.das === null ? '–' : `${attack.das.toFixed(1)} m²`} hint={attack.obso === null ? 'DAS at the key moment' : `OBSO ${attack.obso.toFixed(3)}`} help="DAS: dangerous space the attackers could reach with a pass at the key moment (tracking data)." />
            </div>
          </div>

          <div className="rounded-xl border border-line bg-ink/40 p-2">
            <div className="px-1 pb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">Actions in this attack (click to jump)</div>
            <div className="flex flex-wrap gap-1.5">
              {inAttack.map((e, i) => (
                <button
                  key={`${e.frame_num}-${i}`}
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    seek(e.t);
                  }}
                  className={`rounded-md border px-2 py-1 text-[11px] ${Math.abs(e.t - t) < 0.4 ? 'border-lime text-white' : 'border-line text-slate-300 hover:bg-white/[0.05]'}`}
                >
                  <span className="font-mono text-muted">{e.t.toFixed(1)}s</span> {ACTION_NAME[e.type] ?? e.type}
                  {e.player ? ` · ${e.player.split(' ').slice(-1)[0]}` : ''}
                  {e.type === 'SH' ? ' ★' : ''}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// the tab
// ---------------------------------------------------------------------------------------------------------------------
export function Tab4TopAttacks({ info }: { info: MatchInfo }) {
  const homeColor = useUiStore((s) => s.homeColor);
  const awayColor = useUiStore((s) => s.awayColor);
  const [data, setData] = useState<AttacksResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scope, setScope] = useState<Scope>('match');
  const [order, setOrder] = useState<Order>('goals');
  const [setPieces, setSetPieces] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);
  const detailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    getJson<AttacksResponse>(withMatch('/api/attacks', info.match_id), ctrl.signal)
      .then((d) => setData(d))
      .catch((e: unknown) => {
        if (!isAbort(e)) setError(errorMessage(e));
      });
    return () => ctrl.abort();
  }, [info.match_id]);

  const home = info.teams.home.name ?? 'Home';
  const away = info.teams.away.name ?? 'Away';
  const pool = useMemo(() => (data ? data.attacks.filter((a) => (scope === 'match' || a.side === scope) && (setPieces || !a.set_piece)) : []), [data, scope, setPieces]);
  const top = useMemo(() => sortAttacks(pool, order).slice(0, 10), [pool, order]);
  const sel = data?.attacks.find((a) => a.id === selected) ?? top[0] ?? null;
  const selRank = sel ? sortAttacks(pool, order).findIndex((a) => a.id === sel.id) + 1 : 0;

  const choose = (id: number) => {
    setSelected(id);
    detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  if (error) return <ErrorBox message={`Could not load the attacks: ${error}`} />;
  if (!data) return <Spinner label="Finding every attack of the match and rating it… (about 20 s the first time)" />;
  if (!data.attacks.length) return <ErrorBox message="No attacks found. This match needs an events file (passes, shots) on the server." />;

  const seg = (cur: string, opts: [string, string][], set: (v: string) => void) => (
    <div className="flex gap-1 rounded-lg border border-line bg-ink/60 p-0.5">
      {opts.map(([k, label]) => (
        <button key={k} type="button" onClick={() => set(k)} className={`min-h-8 rounded-md px-3 text-xs font-semibold transition ${cur === k ? 'bg-lime text-ink' : 'text-muted hover:text-white'}`}>
          {label}
        </button>
      ))}
    </div>
  );

  const teamCard = (side: Side) => {
    const s = data.summary[side];
    const color = side === 'home' ? homeColor : awayColor;
    const mine = data.attacks.filter((a) => a.side === side);
    const counts = BANDS.map((bd, i) => ({
      ...bd,
      n: mine.filter((a) => a.danger >= bd.from && (i === 0 || a.danger < BANDS[i - 1]!.from)).length
    }));
    const best = sortAttacks(mine, 'danger')[0];
    return (
      <div className="flex flex-col rounded-xl border border-line bg-ink/60 p-3" style={{ boxShadow: `inset 0 3px 0 ${color}` }}>
        <div className="text-sm font-bold" style={{ color }}>{s.team ?? side}</div>
        <div className="mt-2 grid grid-cols-4 gap-2 text-center">
          {[
            ['Attacks', String(s.attacks)],
            ['Dangerous', String(s.dangerous)],
            ['Shots', String(s.shots)],
            ['Goals', String(s.goals)]
          ].map(([k, v]) => (
            <div key={k}>
              <div className="font-mono text-xl font-bold text-white">{v}</div>
              <div className="text-[10px] uppercase tracking-wider text-muted">{k}</div>
            </div>
          ))}
        </div>
        <div className="mt-2 text-[11px] text-muted">
          Average danger <span className="font-mono font-bold text-white">{s.mean_danger.toFixed(0)}</span> · attack xG <span className="font-mono font-bold text-white">{s.xg.toFixed(2)}</span>
        </div>
        <div className="mt-3 text-[10px] font-semibold uppercase tracking-wider text-muted">Attacks by danger</div>
        <div className="mt-1 flex h-3 overflow-hidden rounded-full bg-white/5" aria-label="Attacks by danger band">
          {[...counts].reverse().map((c) =>
            c.n ? <div key={c.label} title={`${c.label}: ${c.n}`} style={{ width: `${(100 * c.n) / Math.max(1, mine.length)}%`, background: c.color }} /> : null
          )}
        </div>
        <div className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px]">
          {counts.map((c) => (
            <div key={c.label} className="flex items-center justify-between text-slate-300">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: c.color }} />
                {c.label}
              </span>
              <span className="font-mono font-bold text-white">{c.n}</span>
            </div>
          ))}
        </div>
        {best ? (
          <button
            type="button"
            onClick={() => choose(best.id)}
            className="mt-auto flex items-center gap-2 rounded-lg border border-line px-2 py-1.5 text-left text-[11px] text-slate-300 hover:bg-white/[0.05]"
            style={{ marginTop: 12 }}
          >
            <DangerGauge score={best.danger} size={34} />
            <span className="min-w-0">
              <span className="block font-semibold text-white">Most dangerous: {clockLabel(best.period, best.minute)} · {best.pattern}</span>
              <span className="block truncate">{best.story}</span>
            </span>
          </button>
        ) : null}
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="Top attacks"
        subtitle="Every attack of the match rated 1–100 by how close it came to a goal (shot xG and expected threat). Click any attack to watch it from start to finish."
        right={
          <div className="flex flex-wrap items-center gap-2">
            {seg(scope, [['match', 'Whole match'], ['home', home], ['away', away]], (v) => setScope(v as Scope))}
            {seg(order, [['goals', 'Goals first'], ['danger', 'Pure danger'], ['chances', 'Best chances']], (v) => setOrder(v as Order))}
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input type="checkbox" checked={setPieces} onChange={(e) => setSetPieces(e.target.checked)} className="accent-lime" />
              Set pieces
            </label>
          </div>
        }
      >
        <div className="grid gap-3 lg:grid-cols-[1fr_1fr_2fr]">
          {teamCard('home')}
          {teamCard('away')}
          <div className="rounded-xl border border-line bg-ink/60 p-2">
            <div className="flex items-center justify-between px-1 text-[11px] text-muted">
              <span className="font-semibold uppercase tracking-wider">Danger flow</span>
              <span>
                <span style={{ color: homeColor }}>●</span> {home} above · <span style={{ color: awayColor }}>●</span> {away} below · <span className="text-yellow-300">◯</span> goal
              </span>
            </div>
            <DangerTimeline attacks={pool} info={info} selected={sel?.id ?? null} onSelect={choose} homeColor={homeColor} awayColor={awayColor} />
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[minmax(340px,5fr)_7fr]">
        <Panel
          title={scope === 'match' ? 'Top 10 · whole match' : `Top 10 · ${scope === 'home' ? home : away}`}
          subtitle={
            order === 'goals'
              ? 'Goals first (by chance quality), then the most dangerous attacks'
              : order === 'danger'
                ? 'Ranked only by danger: what the attack was worth, whatever happened'
                : 'Only attacks with a shot, ranked by shot quality (attack xG) alone'
          }
          right={null}
        >
          <ol className="flex flex-col gap-2">
            {top.map((a, i) => {
              const color = a.side === 'home' ? homeColor : awayColor;
              const active = sel?.id === a.id;
              return (
                <li key={a.id}>
                  <button
                    type="button"
                    onClick={() => choose(a.id)}
                    className={`flex w-full items-center gap-3 rounded-xl border p-2.5 text-left transition ${active ? 'border-lime bg-lime/[0.06]' : 'border-line bg-ink/50 hover:bg-white/[0.04]'}`}
                    style={{ boxShadow: `inset 4px 0 0 ${color}` }}
                  >
                    <span className="w-6 text-center font-mono text-lg font-black text-muted">{i + 1}</span>
                    <DangerGauge score={a.danger} size={50} />
                    <span className="min-w-0 flex-1">
                      <span className="flex flex-wrap items-center gap-1.5">
                        <span className="text-sm font-bold" style={{ color }}>{a.team}</span>
                        <span className="font-mono text-xs text-white">{clockLabel(a.period, a.minute)}</span>
                        <Chip>{a.pattern}</Chip>
                        {a.goal ? <Chip color="#fde047">⚽ Goal</Chip> : null}
                        {a.disallowed_goal ? <Chip color="#f87171">Ruled out</Chip> : null}
                        <Tags tags={a.tags} />
                      </span>
                      <span className="mt-0.5 line-clamp-2 block text-xs leading-snug text-slate-300">{a.story}</span>
                      <span className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-muted">
                        <span><b className="font-mono text-white">{(a.danger_p * 100).toFixed(1)}%</b> chance</span>
                        <span>xG <b className="font-mono text-slate-200">{a.xg.toFixed(2)}</b></span>
                        <span>peak xT <b className="font-mono text-slate-200">{a.peak_xt.toFixed(2)}</b></span>
                        <span>{a.passes} {a.passes === 1 ? 'pass' : 'passes'}</span>
                        <span>{a.duration_s.toFixed(0)} s</span>
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
          <details className="mt-3 rounded-lg border border-line bg-ink/40 p-3 text-xs text-slate-300">
            <summary className="cursor-pointer font-semibold text-white">How is the danger score calculated?</summary>
            <ul className="mt-2 list-disc space-y-1 pl-4 leading-snug">
              <li><b>Attack</b> = a passage of play by one team (Opta's "sequence"): it starts when the team wins the ball or restarts play and ends when the opponent controls the ball, the ball goes dead or play stops. Blocks and rebounds keep it alive.</li>
              <li><b>Shot quality</b> = chance that at least one of its shots went in, from the expected-goals (xG) of each shot. The xG model was fitted on about 10,700 public StatsBomb shots (World Cup 2022 kept out for testing) and knows distance, angle, headers, direct free kicks, defenders between the ball and the goal, the goalkeeper's position and opponents close to the shooter, read from the tracking data. Penalties count 0.79; goals ruled out for offside count nothing.</li>
              <li><b>Best spot reached</b> = expected threat (xT): the chance of a goal from the most dangerous place where the team controlled the ball (Laurie Shaw's EPV grid, rescaled by 0.852 so it matches real goal rates at shot spots). The spot of a shot itself is valued by the shot's xG. This credits attacks that got close without shooting.</li>
              <li><b>Tags</b> follow published definitions: counter-attack (StatsBomb style: ball won outside the final third, at least 75% direct, 16.5 m+ progress), direct attack, build-up and high turnover (Opta), fast break, and big chance (xG 0.38+ or a penalty).</li>
              <li><b>Danger</b> = the larger of the two, turned into 1–100 on a fixed scale (2% = 30, 5% = 48, 10% = 61, 20% = 74, 40% = 87, penalty 79% = 100), so scores can be compared between matches.</li>
              <li>Bands: 75+ very dangerous · 55–74 dangerous · 35–54 promising · below 35 low threat.</li>
            </ul>
          </details>
        </Panel>

        <div ref={detailRef}>
          <Panel title="Watch the attack" subtitle="From 3 s before it starts to 2 s after it ends. Yellow = the ball's path, numbered arrows = passes, ★ = shots, green rings = players involved." right={null}>
            {sel ? <AttackDetail key={sel.id} matchId={info.match_id} attack={sel} rank={selRank} /> : <p className="text-sm text-muted">Pick an attack.</p>}
          </Panel>
        </div>
      </div>
    </div>
  );
}

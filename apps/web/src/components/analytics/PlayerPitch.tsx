import { useMemo, useState } from 'react';
import type { LiveDetail, LiveStats } from './api';

export type PitchMode = 'now' | 'heat' | 'passes' | 'shots' | 'defending';

interface Props {
  mode: PitchMode;
  length: number;
  width: number;
  /** all players at this moment (used by the "Now" view) */
  players: LiveStats[];
  ball: { x: number; y: number } | null;
  selectedSlot: number | null;
  onSelect: (slot: number) => void;
  /** the selected player's detail (heat map, passes, shots...) in "attacking left to right" coordinates */
  detail: LiveDetail | null;
  homeColor: string;
  awayColor: string;
  homeText: string;
  awayText: string;
  /** which attack direction to draw as an arrow in the maps */
  teamColor: string;
}

const LINE = 'rgba(255,255,255,0.55)';

function Lines({ L, W }: { L: number; W: number }) {
  const gx = L / 2;
  const box = (dir: 1 | -1, depth: number, half: number) => {
    const x = dir === 1 ? gx - depth : -gx;
    return <rect key={`${dir}-${depth}`} x={x + L / 2} y={W / 2 - half} width={depth} height={2 * half} fill="none" stroke={LINE} strokeWidth={0.25} />;
  };
  return (
    <g>
      <rect x={0} y={0} width={L} height={W} fill="#0f2a1b" stroke={LINE} strokeWidth={0.3} />
      {Array.from({ length: 10 }, (_, i) => (i % 2 === 0 ? <rect key={i} x={(i * L) / 10} y={0} width={L / 10} height={W} fill="rgba(255,255,255,0.025)" /> : null))}
      <line x1={L / 2} y1={0} x2={L / 2} y2={W} stroke={LINE} strokeWidth={0.25} />
      <circle cx={L / 2} cy={W / 2} r={9.15} fill="none" stroke={LINE} strokeWidth={0.25} />
      <circle cx={L / 2} cy={W / 2} r={0.4} fill={LINE} />
      {([1, -1] as const).map((d) => (
        <g key={d}>
          {box(d, 16.5, 20.16)}
          {box(d, 5.5, 9.16)}
          <circle cx={d === 1 ? L - 11 : 11} cy={W / 2} r={0.4} fill={LINE} />
          <line x1={d === 1 ? L : 0} y1={W / 2 - 3.66} x2={d === 1 ? L : 0} y2={W / 2 + 3.66} stroke="rgba(255,255,255,0.95)" strokeWidth={0.7} />
        </g>
      ))}
    </g>
  );
}

const SHOT_COLOR: Record<string, string> = {
  goal: '#4ade80',
  saved: '#facc15',
  off: '#f87171',
  blocked: '#94a3b8',
  disallowed: '#c084fc',
  other: '#64748b'
};

export function PlayerPitch(p: Props) {
  const { mode, length: L, width: W, detail } = p;
  const [hover, setHover] = useState<string | null>(null);
  const X = (x: number) => x + L / 2;
  const Y = (y: number) => W / 2 - y;

  const heat = useMemo(() => {
    if (mode !== 'heat' || !detail) return null;
    const h = detail.heat;
    let max = 0;
    for (const row of h.cells) for (const v of row) max = Math.max(max, v);
    return { h, max };
  }, [mode, detail]);

  return (
    <div className="relative">
      <svg viewBox={`-1 -1 ${L + 2} ${W + 2}`} className="block h-auto w-full rounded-xl" role="img" aria-label="Pitch">
        <defs>
          <marker id="arrow-ok" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M0,0 L6,3 L0,6 z" fill="#4ade80" />
          </marker>
          <marker id="arrow-bad" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M0,0 L6,3 L0,6 z" fill="#f87171" />
          </marker>
          <marker id="arrow-key" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M0,0 L6,3 L0,6 z" fill="#facc15" />
          </marker>
        </defs>
        <Lines L={L} W={W} />

        {mode !== 'now' ? (
          <g opacity={0.8}>
            <line x1={L / 2 - 14} y1={W + 0.2} x2={L / 2 + 14} y2={W + 0.2} stroke={p.teamColor} strokeWidth={0.5} />
            <path d={`M${L / 2 + 14},${W + 0.2} l-1.6,-0.9 v1.8 z`} fill={p.teamColor} />
          </g>
        ) : null}

        {heat ? (
          <g>
            {heat.h.cells.map((row, iy) =>
              row.map((v, ix) => {
                if (v <= 0 || heat.max <= 0) return null;
                const cw = L / heat.h.nx;
                const ch = W / heat.h.ny;
                const t = v / heat.max;
                return (
                  <rect
                    key={`${iy}-${ix}`}
                    x={ix * cw}
                    y={W - (iy + 1) * ch}
                    width={cw}
                    height={ch}
                    fill={`rgba(255,${Math.round(190 - 150 * t)},40,${(0.08 + 0.72 * Math.sqrt(t)).toFixed(3)})`}
                  />
                );
              })
            )}
            {heat.h.avg ? (
              <g>
                <circle cx={X(heat.h.avg[0])} cy={Y(heat.h.avg[1])} r={1.6} fill="#fff" stroke="#0b1210" strokeWidth={0.4} />
                <text x={X(heat.h.avg[0])} y={Y(heat.h.avg[1]) - 2.4} fontSize={2.6} fill="#fff" textAnchor="middle" fontWeight={700}>
                  average position
                </text>
              </g>
            ) : null}
          </g>
        ) : null}

        {mode === 'passes' && detail ? (
          <g>
            {detail.maps.passes.map((a, i) => {
              if (a.ex === null || a.ey === null) return null;
              const col = a.key ? '#facc15' : a.ok ? '#4ade80' : '#f87171';
              const mk = a.key ? 'arrow-key' : a.ok ? 'arrow-ok' : 'arrow-bad';
              return (
                <g key={i} onMouseEnter={() => setHover(`${a.clock}' ${a.cross ? 'cross' : 'pass'} ${a.assist ? '(ASSIST) ' : a.key ? '(key pass) ' : ''}${a.ok ? 'completed' : 'not completed'}${a.to ? ' → ' + a.to : ''}`)} onMouseLeave={() => setHover(null)}>
                  <line x1={X(a.x)} y1={Y(a.y)} x2={X(a.ex)} y2={Y(a.ey)} stroke={col} strokeWidth={a.key ? 0.55 : 0.32} strokeOpacity={a.ok ? 0.8 : 0.65} markerEnd={`url(#${mk})`} strokeDasharray={a.cross ? '1.2 0.8' : undefined} />
                  <circle cx={X(a.x)} cy={Y(a.y)} r={0.65} fill={col} />
                  <line x1={X(a.x)} y1={Y(a.y)} x2={X(a.ex)} y2={Y(a.ey)} stroke="transparent" strokeWidth={2.2} />
                </g>
              );
            })}
          </g>
        ) : null}

        {mode === 'shots' && detail ? (
          <g>
            {detail.maps.shots.map((s, i) => {
              const col = SHOT_COLOR[s.cat] ?? '#64748b';
              const r = 1.2 + Math.sqrt(Math.max(s.xg, 0)) * 3.2;
              return (
                <g key={i} onMouseEnter={() => setHover(`${s.clock}' ${s.label} · xG ${s.xg.toFixed(2)}`)} onMouseLeave={() => setHover(null)}>
                  {s.ex !== null && s.ey !== null ? <line x1={X(s.x)} y1={Y(s.y)} x2={X(Math.min(s.ex, L / 2))} y2={Y(s.ey)} stroke={col} strokeWidth={0.3} strokeDasharray="1 0.8" strokeOpacity={0.8} /> : null}
                  <circle cx={X(s.x)} cy={Y(s.y)} r={r} fill={col} fillOpacity={0.55} stroke={col} strokeWidth={0.4} />
                  {s.cat === 'goal' ? <text x={X(s.x)} y={Y(s.y) + 0.9} fontSize={2.6} textAnchor="middle" fill="#052e16" fontWeight={800}>G</text> : null}
                </g>
              );
            })}
          </g>
        ) : null}

        {mode === 'defending' && detail ? (
          <g>
            {detail.maps.defending.map((d, i) => (
              <g key={`d${i}`} onMouseEnter={() => setHover(`${d.clock}' ${d.type}`)} onMouseLeave={() => setHover(null)}>
                <rect x={X(d.x) - 1} y={Y(d.y) - 1} width={2} height={2} transform={`rotate(45 ${X(d.x)} ${Y(d.y)})`} fill="#38bdf8" fillOpacity={0.7} stroke="#bae6fd" strokeWidth={0.3} />
              </g>
            ))}
            {detail.maps.duels.map((d, i) => (
              <g key={`u${i}`} onMouseEnter={() => setHover(`${d.clock}' ${d.type} · ${d.won ? 'won' : 'lost'}`)} onMouseLeave={() => setHover(null)}>
                <circle cx={X(d.x)} cy={Y(d.y)} r={1.2} fill={d.won ? '#4ade80' : '#f87171'} fillOpacity={0.35} stroke={d.won ? '#4ade80' : '#f87171'} strokeWidth={0.4} />
              </g>
            ))}
          </g>
        ) : null}

        {mode === 'now' ? (
          <g>
            {p.players.map((pl) => {
              if (pl.x === null || pl.y === null) return null;
              const sel = pl.slot === p.selectedSlot;
              const col = pl.side === 'home' ? p.homeColor : p.awayColor;
              const txt = pl.side === 'home' ? p.homeText : p.awayText;
              return (
                <g key={pl.slot} onClick={() => p.onSelect(pl.slot)} style={{ cursor: 'pointer' }} onMouseEnter={() => setHover(`#${pl.jersey} ${pl.name}`)} onMouseLeave={() => setHover(null)}>
                  {sel ? <circle cx={X(pl.x)} cy={Y(pl.y)} r={3.2} fill="none" stroke="#c8f45a" strokeWidth={0.6} /> : null}
                  <circle cx={X(pl.x)} cy={Y(pl.y)} r={1.9} fill={col} stroke="#fff" strokeWidth={sel ? 0.5 : 0.25} />
                  <text x={X(pl.x)} y={Y(pl.y) + 0.85} fontSize={2.3} textAnchor="middle" fill={txt} fontWeight={700} style={{ pointerEvents: 'none' }}>
                    {pl.jersey}
                  </text>
                </g>
              );
            })}
            {p.ball ? <circle cx={X(p.ball.x)} cy={Y(p.ball.y)} r={0.95} fill="#fff" stroke="#111" strokeWidth={0.3} /> : null}
          </g>
        ) : null}

        {mode !== 'now' && detail?.now ? <circle cx={X(detail.now.x)} cy={Y(detail.now.y)} r={1.1} fill="#c8f45a" stroke="#0b1210" strokeWidth={0.35} /> : null}
      </svg>
      {hover ? <div className="pointer-events-none absolute left-2 top-2 max-w-[80%] rounded-md bg-black/70 px-2 py-1 text-xs text-white">{hover}</div> : null}
    </div>
  );
}

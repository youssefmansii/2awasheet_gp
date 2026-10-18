import { useEffect, useRef, useState } from 'react';
import type { ActionRow, FramePayload } from './api';

export interface PitchLayers {
  pc: boolean;
  das: boolean;
  obso: boolean;
  vaep: boolean;
}

interface Props {
  frame: FramePayload | null;
  layers: PitchLayers;
  actions: ActionRow[];
  windowFrames: number;
  homeColor: string;
  awayColor: string;
  homeText: string;
  awayText: string;
  showNumbers: boolean;
  /** > 0: glide players and ball from the previous frame to the new one over this many ms (used while playing) */
  animateMs?: number;
  length: number;
  width: number;
}

function rgba(hex: string, a: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m || !m[1]) return `rgba(255,255,255,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function drawPitch(ctx: CanvasRenderingContext2D, w: number, h: number, L: number, W: number) {
  const sx = w / L;
  const sy = h / W;
  const px = (x: number) => (x + L / 2) * sx;
  const py = (y: number) => (W / 2 - y) * sy;
  ctx.fillStyle = '#0f2a1b';
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.025)';
      ctx.fillRect((i * w) / 10, 0, w / 10, h);
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(1, 1, w - 2, h - 2);
  ctx.beginPath();
  ctx.moveTo(px(0), 0);
  ctx.lineTo(px(0), h);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px(0), py(0), 9.15 * sx, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.beginPath();
  ctx.arc(px(0), py(0), 2, 0, Math.PI * 2);
  ctx.fill();
  for (const dir of [-1, 1]) {
    const gx = (L / 2) * dir;
    const box = (depth: number, halfW: number) => {
      const x0 = px(gx);
      const x1 = px(gx - dir * depth);
      ctx.strokeRect(Math.min(x0, x1), py(halfW), Math.abs(x1 - x0), 2 * halfW * sy);
    };
    box(16.5, 20.16);
    box(5.5, 9.16);
    ctx.beginPath();
    ctx.arc(px(gx - dir * 11), py(0), 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px(gx), py(3.66));
    ctx.lineTo(px(gx), py(-3.66));
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1.5;
  }
}

type Pt = [number, number];
interface Drawn {
  target: number | null;
  players: Map<number, Pt>;
  ball: Pt | null;
}

export function AnalyticsPitch(props: Props) {
  const { frame, layers, actions, windowFrames, homeColor, awayColor, homeText, awayText, showNumbers, length, width } = props;
  const animateMs = props.animateMs ?? 0;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawnRef = useRef<Drawn>({ target: null, players: new Map(), ball: null });
  const [w, setW] = useState(900);
  const h = Math.round((w * width) / length);

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
    if (!frame) {
      drawPitch(ctx, w, h, length, width);
      return;
    }

    const sx = w / length;
    const sy = h / width;
    const px = (x: number) => (x + length / 2) * sx;
    const py = (y: number) => (width / 2 - y) * sy;

    // ---- where everybody is going to be (and where they were drawn last) ----
    const toPlayers = new Map<number, Pt>(frame.players.map((p) => [p.slot, [p.x, p.y] as Pt]));
    const toBall: Pt | null = frame.ball ? [frame.ball.x, frame.ball.y] : null;
    const prev = drawnRef.current;
    let glide = animateMs > 0 && prev.target !== null && prev.target !== frame.frame_num;
    if (glide) {
      let far = 0; // a big jump (seek, other half) is not a movement: do not slide across the pitch
      for (const [slot, to] of toPlayers) {
        const from = prev.players.get(slot);
        if (from) far = Math.max(far, Math.hypot(to[0] - from[0], to[1] - from[1]));
      }
      if (far > 12) glide = false;
    }
    const fromPlayers = prev.players;
    const fromBall = prev.ball;

    const render = (k: number) => {
      const lerp = (a: number, b: number) => a + (b - a) * k;
      const curPlayers = new Map<number, Pt>();
      for (const [slot, to] of toPlayers) {
        const from = glide && k < 1 ? fromPlayers.get(slot) : undefined;
        curPlayers.set(slot, from ? [lerp(from[0], to[0]), lerp(from[1], to[1])] : to);
      }
      const curBall: Pt | null = toBall && glide && k < 1 && fromBall ? [lerp(fromBall[0], toBall[0]), lerp(fromBall[1], toBall[1])] : toBall;
      drawnRef.current = { target: frame.frame_num, players: curPlayers, ball: curBall };
      const at = (slot: number, fx: number, fy: number): Pt => curPlayers.get(slot) ?? [fx, fy];

      drawPitch(ctx, w, h, length, width);

      // ---- grid layers: row 0 is the bottom of the pitch (y = -width/2) ----
      const paintGrid = (grid: number[][], colorFor: (v: number) => string | null) => {
        const ny = grid.length;
        const nx = grid[0]?.length ?? 0;
        if (!ny || !nx) return;
        const cw = w / nx;
        const ch = h / ny;
        for (let iy = 0; iy < ny; iy++) {
          const row = grid[iy];
          if (!row) continue;
          for (let ix = 0; ix < nx; ix++) {
            const v = row[ix];
            if (v === undefined) continue;
            const c = colorFor(v);
            if (!c) continue;
            ctx.fillStyle = c;
            ctx.fillRect(ix * cw, h - (iy + 1) * ch, cw + 0.5, ch + 0.5);
          }
        }
      };

      if (layers.pc) {
        paintGrid(frame.pitch_control_home, (v) => {
          const a = Math.min(0.62, Math.abs(v - 0.5) * 1.35);
          return a < 0.03 ? null : rgba(v >= 0.5 ? homeColor : awayColor, a);
        });
      }
      let labelY = 16;
      const label = (text: string, color: string) => {
        ctx.fillStyle = color;
        ctx.font = '600 11px ui-sans-serif, system-ui';
        ctx.fillText(text, 8, labelY);
        labelY += 14;
      };
      const noBall = frame.ball_tracked === false;
      if (layers.das) {
        let max = 1e-9; // values are danger-weighted m² per cell (they add up to the DAS number), so scale to this frame
        for (const row of frame.das) for (const v of row) if (v > max) max = v;
        paintGrid(frame.das, (v) => {
          const t = Math.pow(v / max, 0.7);
          return t < 0.12 ? null : rgba('#c8f45a', Math.min(0.85, t * 0.85));
        });
        label(noBall ? 'DAS: needs the ball (not tracked here)' : 'DAS', 'rgba(200,244,90,0.95)');
      }

      // ---- OBSO (C-OBSO): the team grid, and each off-ball attacker's value at his own position ----
      if (layers.obso) {
        const grid = frame.obso_grid;
        if (grid && grid.length && !noBall) {
          let gmax = 1e-9;
          for (const row of grid) for (const v of row) if (v > gmax) gmax = v;
          paintGrid(grid, (v) => {
            const t = Math.pow(v / gmax, 0.6);
            return t < 0.1 ? null : rgba('#facc15', Math.min(0.8, t * 0.8));
          });
        }
        label(noBall ? 'OBSO: needs the ball (not tracked here)' : 'OBSO', 'rgba(250,204,21,0.95)');
        const att = frame.players.filter((p) => p.obso !== undefined);
        const maxO = Math.max(1e-9, ...att.map((p) => p.obso ?? 0));
        const ranked = [...att].sort((a, b) => (b.obso ?? 0) - (a.obso ?? 0));
        for (const p of att) {
          const [qx, qy] = at(p.slot, p.x, p.y);
          const rel = (p.obso ?? 0) / maxO;
          ctx.strokeStyle = `rgba(250,204,21,${0.35 + 0.65 * rel})`;
          ctx.lineWidth = 1.5 + 2.5 * rel;
          ctx.beginPath();
          ctx.arc(px(qx), py(qy), 11 + 7 * rel, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.font = '600 11px ui-sans-serif, system-ui';
        for (const p of ranked.slice(0, 3)) {
          const [qx, qy] = at(p.slot, p.x, p.y);
          ctx.fillStyle = '#fde047';
          ctx.fillText((p.obso ?? 0).toFixed(3), px(qx) + 14, py(qy) - 14);
        }
      }

      // ---- VAEP markers (last few seconds) ----
      if (layers.vaep) {
        ctx.textAlign = 'center';
        for (const a of actions) {
          if (a.x === null || a.y === null) continue;
          const age = frame.frame_num - a.frame_num;
          if (age < 0 || age > windowFrames) continue;
          const fade = 1 - age / windowFrames;
          const r = 5 + Math.min(14, Math.abs(a.vaep) * 380);
          const col = a.vaep >= 0 ? '74,222,128' : '248,113,113';
          ctx.fillStyle = `rgba(${col},${0.25 + 0.35 * fade})`;
          ctx.strokeStyle = `rgba(${col},${0.5 + 0.5 * fade})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(px(a.x), py(a.y), r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          if (Math.abs(a.vaep) >= 0.01) {
            ctx.fillStyle = `rgba(255,255,255,${0.5 + 0.5 * fade})`;
            ctx.font = '700 10px ui-sans-serif, system-ui';
            ctx.fillText(`${a.vaep >= 0 ? '+' : ''}${a.vaep.toFixed(3)}`, px(a.x), py(a.y) - r - 4);
          }
        }
        ctx.textAlign = 'start';
      }

      // ---- players and ball ----
      const pr = Math.max(8, sx * 1.05);
      for (const p of frame.players) {
        const [qx, qy] = at(p.slot, p.x, p.y);
        const home = p.side === 'home';
        ctx.fillStyle = home ? homeColor : awayColor;
        ctx.strokeStyle = 'rgba(0,0,0,0.65)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px(qx), py(qy), pr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (showNumbers) {
          ctx.fillStyle = home ? homeText : awayText;
          ctx.font = `700 ${Math.round(pr * 1.05)}px ui-sans-serif, system-ui`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(p.jersey, px(qx), py(qy) + 0.5);
          ctx.textAlign = 'start';
          ctx.textBaseline = 'alphabetic';
        }
      }
      if (curBall) {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px(curBall[0]), py(curBall[1]), Math.max(4.5, pr * 0.5), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    };

    if (!glide) {
      render(1);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const k = Math.min(1, (now - t0) / animateMs);
      render(k);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    render(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frame, layers, actions, windowFrames, homeColor, awayColor, homeText, awayText, showNumbers, length, width, w, h, animateMs]);

  return (
    <div ref={wrapRef} className="w-full overflow-hidden rounded-xl border border-line bg-[#0f2a1b]">
      <canvas ref={canvasRef} role="img" aria-label="Pitch with player positions and metric overlays" className="block" />
    </div>
  );
}

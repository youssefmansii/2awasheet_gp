import { metersToCanvas, PLAYER_SLOTS, TOTAL_SLOTS, TrackingFlag, type TrackingData } from '@replay/shared';
import { useEffect, useRef, type MutableRefObject } from 'react';
import type { PositionMode } from '../store';

export interface PlaybackSnapshot {
  index: number;
  nextIndex: number;
  alpha: number;
}

export interface EventOverlay {
  playerKey: string | null;
  targetKey: string | null;
  type: string;
}

interface PitchCanvasProps {
  data: TrackingData;
  playback: MutableRefObject<PlaybackSnapshot>;
  pitchLength: number;
  pitchWidth: number;
  positionMode: PositionMode;
  showNumbers: boolean;
  showBall: boolean;
  interpolate: boolean;
  flipSides: boolean;
  homeColor: string;
  homeTextColor: string;
  awayColor: string;
  awayTextColor: string;
  trailLength: number;
  selectedPlayer: string | null;
  onSelectPlayer: (key: string | null) => void;
  eventOverlay: EventOverlay | null;
}

type SpritePair = { home: HTMLCanvasElement; away: HTMLCanvasElement };

function shirtSprite(fill: string): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d')!;
  context.scale(2, 2);
  context.fillStyle = fill;
  context.strokeStyle = '#11181c';
  context.lineWidth = 1.5;
  context.lineJoin = 'round';
  context.beginPath();
  context.moveTo(9, 8);
  context.lineTo(3, 11);
  context.lineTo(5, 18);
  context.lineTo(9, 16);
  context.lineTo(9, 27);
  context.lineTo(23, 27);
  context.lineTo(23, 16);
  context.lineTo(27, 18);
  context.lineTo(29, 11);
  context.lineTo(23, 8);
  context.quadraticCurveTo(20, 12, 16, 12);
  context.quadraticCurveTo(12, 12, 9, 8);
  context.closePath();
  context.fill();
  context.stroke();
  return canvas;
}

function drawPitch(context: CanvasRenderingContext2D, width: number, height: number): void {
  const pad = 16;
  const x = pad;
  const y = pad;
  const w = width - pad * 2;
  const h = height - pad * 2;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#315f3b';
  context.beginPath();
  context.roundRect(0, 0, width, height, 22);
  context.fill();
  for (let stripe = 0; stripe < 10; stripe += 1) {
    context.fillStyle = stripe % 2 === 0 ? 'rgba(255,255,255,.055)' : 'rgba(0,0,0,.04)';
    context.fillRect(x + (stripe * w) / 10, y, w / 10, h);
  }
  context.strokeStyle = 'rgba(240,248,241,.78)';
  context.fillStyle = 'rgba(240,248,241,.78)';
  context.lineWidth = Math.max(1.25, width / 650);
  context.strokeRect(x, y, w, h);
  context.beginPath();
  context.moveTo(x + w / 2, y);
  context.lineTo(x + w / 2, y + h);
  context.stroke();
  context.beginPath();
  context.arc(x + w / 2, y + h / 2, (9.15 / 68) * h, 0, Math.PI * 2);
  context.stroke();
  context.beginPath();
  context.arc(x + w / 2, y + h / 2, 2.2, 0, Math.PI * 2);
  context.fill();

  const penaltyDepth = (16.5 / 105) * w;
  const penaltyHeight = (40.32 / 68) * h;
  const goalDepth = (5.5 / 105) * w;
  const goalHeight = (18.32 / 68) * h;
  context.strokeRect(x, y + (h - penaltyHeight) / 2, penaltyDepth, penaltyHeight);
  context.strokeRect(x + w - penaltyDepth, y + (h - penaltyHeight) / 2, penaltyDepth, penaltyHeight);
  context.strokeRect(x, y + (h - goalHeight) / 2, goalDepth, goalHeight);
  context.strokeRect(x + w - goalDepth, y + (h - goalHeight) / 2, goalDepth, goalHeight);
  const spotOffset = (11 / 105) * w;
  for (const spotX of [x + spotOffset, x + w - spotOffset]) {
    context.beginPath();
    context.arc(spotX, y + h / 2, 2.2, 0, Math.PI * 2);
    context.fill();
  }
  const arcRadius = (9.15 / 68) * h;
  const arcAngle = Math.acos((16.5 - 11) / 9.15);
  context.beginPath();
  context.arc(x + spotOffset, y + h / 2, arcRadius, -arcAngle, arcAngle);
  context.stroke();
  context.beginPath();
  context.arc(x + w - spotOffset, y + h / 2, arcRadius, Math.PI - arcAngle, Math.PI + arcAngle);
  context.stroke();
  const corner = Math.min(10, w / 60);
  for (const [cx, cy, start] of [[x, y, 0], [x + w, y, Math.PI / 2], [x + w, y + h, Math.PI], [x, y + h, Math.PI * 1.5]] as const) {
    context.beginPath();
    context.arc(cx, cy, corner, start, start + Math.PI / 2);
    context.stroke();
  }
}

function nextMatchingSlot(data: TrackingData, mode: PositionMode, nextIndex: number, team: number, jersey: number): number {
  const jerseys = mode === 'smoothed' ? data.smoothedJerseys : data.rawJerseys;
  const base = nextIndex * TOTAL_SLOTS + team * PLAYER_SLOTS;
  for (let slot = 0; slot < PLAYER_SLOTS; slot += 1) {
    if (jerseys[base + slot] === jersey) return team * PLAYER_SLOTS + slot;
  }
  return -1;
}

export function PitchCanvas(props: PitchCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const staticPitchRef = useRef<HTMLCanvasElement | null>(null);
  const sizeRef = useRef({ width: 0, height: 0, dpr: 1 });
  const propsRef = useRef(props);
  const spritesRef = useRef<SpritePair>({ home: shirtSprite(props.homeColor), away: shirtSprite(props.awayColor) });
  propsRef.current = props;

  useEffect(() => {
    spritesRef.current = { home: shirtSprite(props.homeColor), away: shirtSprite(props.awayColor) };
  }, [props.homeColor, props.awayColor]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(320, Math.round(entry.contentRect.width));
      const height = Math.round(width * props.pitchWidth / props.pitchLength);
      canvas.style.height = `${height}px`;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      sizeRef.current = { width, height, dpr: ratio };
      const staticCanvas = document.createElement('canvas');
      staticCanvas.width = canvas.width;
      staticCanvas.height = canvas.height;
      const context = staticCanvas.getContext('2d')!;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      drawPitch(context, width, height);
      staticPitchRef.current = staticCanvas;
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [props.pitchLength, props.pitchWidth]);

  useEffect(() => {
    let animationFrame = 0;
    const render = () => {
      const canvas = canvasRef.current;
      const staticPitch = staticPitchRef.current;
      const current = propsRef.current;
      const { width, height, dpr } = sizeRef.current;
      if (canvas && staticPitch && width > 0) {
        const context = canvas.getContext('2d')!;
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(staticPitch, 0, 0);
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        const viewport = { x: 16, y: 16, width: width - 32, height: height - 32 };
        const snapshot = current.playback.current;
        const positions = current.positionMode === 'smoothed' ? current.data.smoothedPlayers : current.data.rawPlayers;
        const jerseys = current.positionMode === 'smoothed' ? current.data.smoothedJerseys : current.data.rawJerseys;
        const flags = current.positionMode === 'smoothed' ? current.data.smoothedFlags : current.data.rawFlags;
        const frameBase = snapshot.index * TOTAL_SLOTS;
        const positionBase = snapshot.index * TOTAL_SLOTS * 2;
        const playerPoints = new Map<string, { x: number; y: number }>();

        if (current.selectedPlayer && current.trailLength > 0) {
          const [teamText, jerseyText] = current.selectedPlayer.split(':');
          const team = Number(teamText);
          const jersey = Number(jerseyText);
          context.lineWidth = 2;
          for (let age = current.trailLength; age >= 1; age -= 1) {
            const trailFrame = Math.max(0, snapshot.index - age);
            const slot = nextMatchingSlot(current.data, current.positionMode, trailFrame, team, jersey);
            if (slot < 0) continue;
            const offset = trailFrame * TOTAL_SLOTS * 2 + slot * 2;
            const point = metersToCanvas(positions[offset]!, positions[offset + 1]!, current.pitchLength, current.pitchWidth, viewport, current.flipSides);
            context.globalAlpha = (1 - age / (current.trailLength + 1)) * 0.42;
            context.fillStyle = team === 0 ? current.homeColor : current.awayColor;
            context.beginPath();
            context.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
            context.fill();
          }
          context.globalAlpha = 1;
        }

        for (let slot = 0; slot < TOTAL_SLOTS; slot += 1) {
          const jersey = jerseys[frameBase + slot] ?? 0;
          if (!jersey) continue;
          const offset = positionBase + slot * 2;
          let x = positions[offset]!;
          let y = positions[offset + 1]!;
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          if (current.interpolate && snapshot.alpha > 0 && snapshot.nextIndex !== snapshot.index) {
            const team = slot >= PLAYER_SLOTS ? 1 : 0;
            const nextSlot = nextMatchingSlot(current.data, current.positionMode, snapshot.nextIndex, team, jersey);
            if (nextSlot >= 0) {
              const nextOffset = snapshot.nextIndex * TOTAL_SLOTS * 2 + nextSlot * 2;
              const nextX = positions[nextOffset]!;
              const nextY = positions[nextOffset + 1]!;
              if (Number.isFinite(nextX) && Number.isFinite(nextY)) {
                x += (nextX - x) * snapshot.alpha;
                y += (nextY - y) * snapshot.alpha;
              }
            }
          }
          const point = metersToCanvas(x, y, current.pitchLength, current.pitchWidth, viewport, current.flipSides);
          const team = slot >= PLAYER_SLOTS ? 1 : 0;
          const playerKey = `${team}:${jersey}`;
          playerPoints.set(playerKey, point);
          const selected = current.selectedPlayer === playerKey;
          context.globalAlpha = flags[frameBase + slot] === TrackingFlag.Faded ? 0.5 : 1;
          if (selected) {
            context.strokeStyle = '#ffffff';
            context.lineWidth = 2.5;
            context.beginPath();
            context.arc(point.x, point.y, 18, 0, Math.PI * 2);
            context.stroke();
          }
          const sprite = team === 0 ? spritesRef.current.home : spritesRef.current.away;
          context.drawImage(sprite, point.x - 16, point.y - 16, 32, 32);
          if (current.showNumbers) {
            context.fillStyle = team === 0 ? current.homeTextColor : current.awayTextColor;
            context.font = '700 10px ui-sans-serif, system-ui, sans-serif';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(String(jersey), point.x, point.y + 3);
          }
        }

        if (current.eventOverlay) {
          const actor = current.eventOverlay.playerKey ? playerPoints.get(current.eventOverlay.playerKey) : undefined;
          const target = current.eventOverlay.targetKey ? playerPoints.get(current.eventOverlay.targetKey) : undefined;
          context.globalAlpha = 1;
          if (actor && target && current.eventOverlay.type === 'PA') {
            context.strokeStyle = 'rgba(255,255,255,.9)';
            context.lineWidth = 2;
            context.setLineDash([6, 5]);
            context.beginPath();
            context.moveTo(actor.x, actor.y);
            context.lineTo(target.x, target.y);
            context.stroke();
            context.setLineDash([]);
            context.fillStyle = '#fff';
            context.beginPath();
            context.arc(target.x, target.y, 4, 0, Math.PI * 2);
            context.fill();
          }
          if (actor) {
            context.strokeStyle = current.eventOverlay.type === 'SH' ? '#fbbf24' : '#ffffff';
            context.lineWidth = 3;
            context.beginPath();
            context.arc(actor.x, actor.y, 20, 0, Math.PI * 2);
            context.stroke();
          }
        }

        if (current.showBall) {
          const balls = current.positionMode === 'smoothed' ? current.data.smoothedBall : current.data.rawBall;
          const visible = current.positionMode === 'smoothed' ? current.data.smoothedBallVisible : current.data.rawBallVisible;
          if (visible[snapshot.index]) {
            const ballOffset = snapshot.index * 3;
            let x = balls[ballOffset]!;
            let y = balls[ballOffset + 1]!;
            let z = balls[ballOffset + 2] || 0;
            if (current.interpolate && snapshot.nextIndex !== snapshot.index) {
              const nextOffset = snapshot.nextIndex * 3;
              const t = snapshot.alpha;
              x += (balls[nextOffset]! - x) * t;
              y += (balls[nextOffset + 1]! - y) * t;
              z += ((balls[nextOffset + 2] || 0) - z) * t;
            }
            const point = metersToCanvas(x, y, current.pitchLength, current.pitchWidth, viewport, current.flipSides);
            const radius = Math.min(9, 6 + Math.max(0, z) * 0.65);
            context.globalAlpha = 0.24;
            context.fillStyle = '#000';
            context.beginPath();
            context.ellipse(point.x + 2, point.y + radius, radius, radius * 0.42, 0, 0, Math.PI * 2);
            context.fill();
            context.globalAlpha = 1;
            context.fillStyle = '#fff';
            context.strokeStyle = '#15191c';
            context.lineWidth = 2;
            context.beginPath();
            context.arc(point.x, point.y, radius, 0, Math.PI * 2);
            context.fill();
            context.stroke();
            context.fillStyle = '#20262a';
            context.beginPath();
            context.moveTo(point.x, point.y - radius * 0.45);
            context.lineTo(point.x + radius * 0.45, point.y - radius * 0.1);
            context.lineTo(point.x + radius * 0.28, point.y + radius * 0.42);
            context.lineTo(point.x - radius * 0.28, point.y + radius * 0.42);
            context.lineTo(point.x - radius * 0.45, point.y - radius * 0.1);
            context.closePath();
            context.fill();
          }
        }
        context.globalAlpha = 1;
      }
      animationFrame = requestAnimationFrame(render);
    };
    animationFrame = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  const selectAt = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const current = propsRef.current;
    const rect = canvas.getBoundingClientRect();
    const viewport = { x: 16, y: 16, width: rect.width - 32, height: rect.height - 32 };
    const { index } = current.playback.current;
    const positions = current.positionMode === 'smoothed' ? current.data.smoothedPlayers : current.data.rawPlayers;
    const jerseys = current.positionMode === 'smoothed' ? current.data.smoothedJerseys : current.data.rawJerseys;
    let nearest: { key: string; distance: number } | null = null;
    for (let slot = 0; slot < TOTAL_SLOTS; slot += 1) {
      const jersey = jerseys[index * TOTAL_SLOTS + slot] ?? 0;
      if (!jersey) continue;
      const offset = index * TOTAL_SLOTS * 2 + slot * 2;
      const point = metersToCanvas(positions[offset]!, positions[offset + 1]!, current.pitchLength, current.pitchWidth, viewport, current.flipSides);
      const distance = Math.hypot(clientX - rect.left - point.x, clientY - rect.top - point.y);
      if (distance < 24 && (!nearest || distance < nearest.distance)) nearest = { key: `${slot >= PLAYER_SLOTS ? 1 : 0}:${jersey}`, distance };
    }
    current.onSelectPlayer(nearest?.key ?? null);
  };

  return (
    <canvas
      ref={canvasRef}
      className="block w-full cursor-crosshair rounded-[22px] bg-[#315f3b] shadow-2xl"
      role="img"
      aria-label="Football pitch showing tracked player and ball positions. Click a player to select a trail."
      onClick={(event) => selectAt(event.clientX, event.clientY)}
    />
  );
}

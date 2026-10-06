export interface PitchViewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function metersToCanvas(
  xMeters: number,
  yMeters: number,
  pitchLength: number,
  pitchWidth: number,
  viewport: PitchViewport,
  flipSides = false
): { x: number; y: number } {
  const direction = flipSides ? -1 : 1;
  return {
    x: viewport.x + ((direction * xMeters + pitchLength / 2) / pitchLength) * viewport.width,
    y: viewport.y + ((pitchWidth / 2 - direction * yMeters) / pitchWidth) * viewport.height
  };
}

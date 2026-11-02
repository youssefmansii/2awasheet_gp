export function findFrameByTime(times: Float64Array, targetMs: number): number {
  if (times.length === 0) return -1;
  if (targetMs <= (times[0] ?? 0)) return 0;
  const last = times.length - 1;
  if (targetMs >= (times[last] ?? 0)) return last;
  let low = 0;
  let high = last;
  while (low <= high) {
    const middle = (low + high) >>> 1;
    const value = times[middle] ?? 0;
    if (value <= targetMs) low = middle + 1;
    else high = middle - 1;
  }
  return Math.max(0, high);
}

export function formatMatchClock(seconds: number, period: number): string {
  const adjusted = period === 2 && seconds < 45 * 60 ? seconds + 45 * 60 : seconds;
  const safe = Math.max(0, Math.floor(adjusted));
  return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`;
}

export function periodLabel(period: number): string {
  if (period === 1) return '1st half';
  if (period === 2) return '2nd half';
  return period > 0 ? `Period ${period}` : 'Pre-match';
}

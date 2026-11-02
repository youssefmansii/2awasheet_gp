import type { MatchMetadata } from './types';

export function parseMetadata(text: string): MatchMetadata {
  const parsed: unknown = JSON.parse(text);
  const metadata = Array.isArray(parsed) ? parsed[0] : parsed;
  if (!metadata || typeof metadata !== 'object') throw new Error('Metadata must be an object or a one-item array.');
  return metadata as MatchMetadata;
}

export function playableRanges(metadata: MatchMetadata | null): Array<[number, number]> {
  if (!metadata) return [];
  const candidates = [
    [metadata.startPeriod1, metadata.endPeriod1],
    [metadata.startPeriod2, metadata.endPeriod2]
  ];
  return candidates.flatMap(([start, end]) =>
    typeof start === 'number' && typeof end === 'number' && end >= start
      ? [[start * 1000, end * 1000] as [number, number]]
      : []
  );
}

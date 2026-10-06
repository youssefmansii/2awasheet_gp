import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

const filePath = process.argv[2];
if (!filePath) {
  console.error('Usage: node scripts/scan-jsonl.mjs <tracking.jsonl>');
  process.exitCode = 1;
} else {
  const keyPaths = new Map();
  const gameEvents = new Map();
  const possessionEvents = new Map();
  let lines = 0;
  let blank = 0;
  let malformed = 0;

  function count(map, key) {
    map.set(key, (map.get(key) ?? 0) + 1);
  }

  function visit(value, path = '') {
    if (value === null || typeof value !== 'object') return;
    if (Array.isArray(value)) {
      for (const item of value) visit(item, `${path}[]`);
      return;
    }
    for (const [key, child] of Object.entries(value)) {
      const next = path ? `${path}.${key}` : key;
      count(keyPaths, next);
      visit(child, next);
    }
  }

  const reader = createInterface({
    input: createReadStream(filePath, { encoding: 'utf8' }),
    crlfDelay: Infinity
  });

  for await (const line of reader) {
    lines += 1;
    if (!line.trim()) {
      blank += 1;
      continue;
    }
    try {
      const frame = JSON.parse(line);
      visit(frame);
      if (frame.game_event?.game_event_type) count(gameEvents, String(frame.game_event.game_event_type));
      if (frame.possession_event?.possession_event_type) count(possessionEvents, String(frame.possession_event.possession_event_type));
    } catch {
      malformed += 1;
    }
  }

  const sorted = (map) => [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  console.log(JSON.stringify({
    file: filePath,
    lines,
    blank,
    malformed,
    keyPaths: Object.fromEntries(sorted(keyPaths)),
    gameEventTypes: Object.fromEntries(sorted(gameEvents)),
    possessionEventTypes: Object.fromEntries(sorted(possessionEvents))
  }, null, 2));
}

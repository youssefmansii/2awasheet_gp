import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { performance } from 'node:perf_hooks';
import { describe, expect, it } from 'vitest';
import { parseTrackingLine } from './parser';
import { WorkloadAccumulator } from './workload';

const RUN = process.env.RUN_PERFORMANCE === '1';
const FRAME_COUNT = 150_000;
const GENERATED_PATH = 'performance/generated-150k.jsonl';

describe.skipIf(!RUN)('streaming parser performance regression', () => {
  it('parses and accumulates 150k frames inside generous time and memory budgets', async () => {
    await mkdir('performance', { recursive: true });
    const output = createWriteStream(GENERATED_PATH, { encoding: 'utf8' });
    for (let frame = 0; frame < FRAME_COUNT; frame += 1) {
      const time = frame * 100;
      const x = Math.sin(frame / 30) * 10;
      const record = {
        frameNum: frame, videoTimeMs: time, period: frame < 75_000 ? 1 : 2,
        periodGameClockTime: frame < 75_000 ? time / 1000 : (time - 7_500_000) / 1000,
        homePlayers: [{ jerseyNum: '10', x, y: 1, confidence: 'HIGH', visibility: 'VISIBLE' }],
        homePlayersSmoothed: [{ jerseyNum: '10', x, y: 1, confidence: 'HIGH', visibility: 'VISIBLE' }],
        awayPlayers: [{ jerseyNum: '9', x: -x, y: -1, confidence: 'HIGH', visibility: 'VISIBLE' }],
        awayPlayersSmoothed: [{ jerseyNum: '9', x: -x, y: -1, confidence: 'HIGH', visibility: 'VISIBLE' }],
        balls: []
      };
      if (!output.write(`${JSON.stringify(record)}\n`)) await once(output, 'drain');
    }
    output.end();
    await once(output, 'finish');

    const workload = new WorkloadAccumulator();
    let frames = 0;
    let peakRss = process.memoryUsage().rss;
    const started = performance.now();
    for await (const line of createInterface({ input: createReadStream(GENERATED_PATH), crlfDelay: Infinity })) {
      const result = parseTrackingLine(line);
      if (result.kind === 'frame') {
        workload.add(result.frame);
        frames += 1;
      }
      if (frames % 5_000 === 0) peakRss = Math.max(peakRss, process.memoryUsage().rss);
    }
    const elapsedMs = performance.now() - started;
    const players = workload.finish();

    expect(frames).toBe(FRAME_COUNT);
    expect(players).toHaveLength(2);
    expect(elapsedMs).toBeLessThan(60_000);
    expect(peakRss).toBeLessThan(600 * 1024 * 1024);
  }, 90_000);
});

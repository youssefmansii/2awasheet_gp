import { expect, test } from '@playwright/test';

test('loads local files, plays, and renders live workload rows', async ({ page }) => {
  await page.goto('/');
  const tracking = Array.from({ length: 30 }, (_, frame) => JSON.stringify({
    frameNum: frame,
    videoTimeMs: frame * 100,
    period: 1,
    periodGameClockTime: frame / 10,
    gameRefId: 'smoke',
    homePlayers: [{ jerseyNum: '10', x: frame * 0.1, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }],
    homePlayersSmoothed: [{ jerseyNum: '10', x: frame * 0.1, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }],
    awayPlayers: [{ jerseyNum: '9', x: -frame * 0.1, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }],
    awayPlayersSmoothed: [{ jerseyNum: '9', x: -frame * 0.1, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }],
    balls: []
  })).join('\n');
  const inputs = page.locator('input[type="file"]');
  await inputs.nth(0).setInputFiles({ name: 'smoke.jsonl', mimeType: 'application/x-ndjson', buffer: Buffer.from(tracking) });
  await inputs.nth(1).setInputFiles({
    name: 'metadata.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ id: 'smoke', homeTeam: { shortName: 'HOME' }, awayTeam: { shortName: 'AWAY' }, stadium: { pitches: [{ length: 105, width: 68 }] } }))
  });
  const table = page.getByRole('table', { name: /Live workload and output indicators/ });
  await expect(table).toBeVisible();
  await expect(table.locator('tbody tr')).not.toHaveCount(0);

  const canvas = page.getByRole('img', { name: /Football pitch showing tracked player/ });
  await expect(canvas).toBeVisible();
  const painted = await canvas.evaluate((element) => {
    const target = element as HTMLCanvasElement;
    const context = target.getContext('2d');
    return Boolean(context && context.getImageData(target.width / 2, target.height / 2, 1, 1).data[3]);
  });
  expect(painted).toBe(true);

  await page.getByRole('button', { name: 'Play replay' }).click();
  await expect(page.getByRole('button', { name: 'Pause replay' })).toBeVisible();
});

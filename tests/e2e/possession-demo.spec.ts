import { expect, test } from '@playwright/test';

function trackingFixture() {
  return Buffer.from(Array.from({ length: 181 }, (_, i) => {
    const ball = { x: Math.min(10, Math.max(0, (i / 30 - 1) * 10)), y: 0, z: 0, visibility: 'VISIBLE' };
    const home = [{ jerseyNum: '10', x: 0, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' },
      { jerseyNum: '9', x: 10, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }];
    const away = [{ jerseyNum: '8', x: 25, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }];
    return JSON.stringify({ frameNum: i, period: 1, videoTimeMs: i / 30 * 1000,
      homePlayers: home, awayPlayers: away, balls: [ball], homePlayersSmoothed: home,
      awayPlayersSmoothed: away, ballsSmoothed: ball });
  }).join('\n'));
}

test('tracking-only demo requires explicit analysis and supports replay, filters and mobile', async ({ page }, testInfo) => {
  const errors: string[] = [];
  let uploads = 0;
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('request', (r) => { if (r.url().endsWith('/files')) uploads++; });
  await page.goto('/?view=possession');
  await expect(page.getByRole('heading', { name: 'Possession lab.' })).toBeVisible();
  await page.getByLabel('Choose tracking file', { exact: true }).setInputFiles({ name: 'demo.jsonl', mimeType: 'application/x-ndjson', buffer: trackingFixture() });
  await expect(page.getByRole('button', { name: 'Analyse locally', exact: true })).toBeEnabled();
  expect(uploads).toBe(0);
  await page.screenshot({ path: testInfo.outputPath('demo-file-selected.png'), fullPage: true });
  await page.getByRole('button', { name: 'Analyse locally', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Every movement. In context.' })).toBeVisible({ timeout: 45_000 });
  expect(uploads).toBe(1);
  await expect(page.getByRole('img', { name: 'Football tracking pitch' })).toBeVisible();
  const slider = page.getByRole('slider', { name: 'Seek tracking replay' });
  const before = Number(await slider.inputValue());
  await page.getByRole('button', { name: 'Play replay', exact: true }).click();
  await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(before + .15);
  await page.getByRole('button', { name: 'Pause replay', exact: true }).click();
  await page.getByLabel('Filter event type').selectOption('touch_candidate');
  await expect(page.locator('.pd-table tbody tr').first()).toBeVisible();
  await page.getByRole('textbox', { name: 'Search event players' }).fill('not-a-player');
  await expect(page.getByText('No candidates match these filters.')).toBeVisible();
  await page.getByRole('textbox', { name: 'Search event players' }).fill('home:10');
  await expect(page.locator('.pd-table tbody tr').first()).toBeVisible();
  await page.locator('.pd-time-button').first().click();
  await expect(page.locator('.pd-selected-row')).toHaveCount(1);
  await page.getByRole('button', { name: 'Player interaction', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Player interaction totals' })).toBeVisible();
  await page.getByRole('button', { name: 'Validation & method', exact: true }).click();
  await expect(page.getByText('Estimates are ready. Validation needs an events file.')).toBeVisible();
  for (const width of [320, 768, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.screenshot({ path: testInfo.outputPath('demo-validation.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('optional event file displays measured validation rather than tracking-only scores', async ({ page }) => {
  await page.goto('/?view=possession');
  await page.getByLabel('Choose tracking file', { exact: true }).setInputFiles({ name: 'validated-demo.jsonl', mimeType: 'application/x-ndjson', buffer: trackingFixture() });
  const reference = [{ eventTime: 1., gameEvents: { period: 1, playerId: 1, homeTeam: true },
    possessionEvents: { possessionEventType: 'PA', passerPlayerId: 1, receiverPlayerId: 2, passOutcomeType: 'C' },
    homePlayers: [{ playerId: 1, jerseyNum: 10 }, { playerId: 2, jerseyNum: 9 }] }];
  await page.getByLabel('Choose events file', { exact: true }).setInputFiles({ name: 'events.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(reference)) });
  await page.getByRole('button', { name: 'Analyse locally', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Every movement. In context.' })).toBeVisible({ timeout: 45_000 });
  await page.getByRole('button', { name: 'Validation & method', exact: true }).click();
  await expect(page.getByText('Measured against the supplied events', { exact: true })).toBeVisible();
  await expect(page.getByText('Estimates are ready. Validation needs an events file.')).toHaveCount(0);
});

test('cancel during reservation releases the job and allows another analysis', async ({ page }) => {
  await page.route('**/api/possession/jobs', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.continue();
  });
  await page.goto('/?view=possession');
  await page.getByLabel('Choose tracking file', { exact: true }).setInputFiles({ name: 'cancel.jsonl', mimeType: 'application/x-ndjson', buffer: trackingFixture() });
  await page.getByRole('button', { name: 'Analyse locally', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel analysis', exact: true }).click();
  await expect.poll(async () => {
    const response = await page.request.get('http://127.0.0.1:8010/api/possession/health');
    const data = await response.json() as { current_job: { state: string } | null };
    return data.current_job?.state;
  }).toBe('cancelled');
  await expect(page.getByRole('button', { name: 'Analyse locally', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Analyse locally', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Every movement. In context.' })).toBeVisible({ timeout: 45_000 });
});

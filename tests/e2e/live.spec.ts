import { expect, test } from '@playwright/test';
const id = 'a'.repeat(32);
const key = 'local-test-only-key-0123456789abcdef';
const base = 'https://tracking-cv-api.youssefmansi2.workers.dev';
const frame = (n: number) => ({ frameNum: n, videoTimeMs: n * 100, period: 1, periodGameClockTime: n / 10,
  homePlayers: [{ jerseyNum: '10', x: n * 0.5, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }], awayPlayers: [], balls: [] });
const snapshot = (start: number, count: number, total = start + count) => ({ sessionId: id, receivedFrames: total,
  retainedFrames: count, lastSequence: 0, receivedAt: new Date().toISOString(), warnings: [],
  frames: Array.from({ length: count }, (_, i) => frame(start + i)) });

test('live API receives overlapping snapshots once and stops without deleting producer data', async ({ page, context }, testInfo) => {
  const methods: string[] = []; let polls = 0; let advance = false;
  await context.route(base + '/**', async (route) => {
    const req = route.request(); methods.push(req.method());
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: {
      'Access-Control-Allow-Origin': 'http://127.0.0.1:5173', 'Access-Control-Allow-Headers': 'Authorization', 'Access-Control-Allow-Methods': 'GET'
    } });
    expect(req.method()).toBe('GET'); expect(req.url()).toBe(base + '/api/cv/sessions/' + id + '/frames?limit=300');
    expect(req.headers().authorization).toBe('Bearer ' + key); polls++;
    await route.fulfill({ status: 200, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' },
      body: JSON.stringify(advance ? snapshot(5, 10) : snapshot(0, 10)) });
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Local replay', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Live API', exact: true }).click();
  await page.getByLabel('Session ID', { exact: true }).fill(id);
  await page.getByLabel('API key (without Bearer)').fill(key);
  await page.getByLabel('GET interval').selectOption('250');
  await page.getByRole('button', { name: 'Connect live API' }).click();
  await expect(page.getByRole('status').first()).toContainText('10 frames observed');
  await expect(page.getByLabel('API key (without Bearer)')).toHaveValue('');
  const table = page.getByRole('table', { name: /Live workload and output indicators/ });
  await expect(table).toBeVisible();
  await expect(table.locator('tbody tr').filter({ hasText: '#10' })).toHaveCount(1);
  advance = true;
  await expect(page.getByRole('status').first()).toContainText('15 frames observed');
  const before = polls; await expect.poll(() => polls).toBeGreaterThan(before);
  await expect(page.getByRole('status').first()).toContainText('15 frames observed');
  await expect(page.getByRole('img', { name: /Football pitch/ })).toBeVisible();
  await page.getByRole('button', { name: 'Stop receiving' }).click();
  await expect(page.getByRole('status').first()).toContainText('Stopped');
  await expect(page.getByRole('status').first()).toContainText('not deleted');
  expect(methods.every((m) => ['GET', 'OPTIONS'].includes(m))).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('live-api-desktop.png'), fullPage: true });
  for (const width of [320, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 850 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.setViewportSize({ width: 375, height: 812 });
  await page.screenshot({ path: testInfo.outputPath('live-api-mobile.png'), fullPage: true });
  await page.getByRole('button', { name: 'Local replay', exact: true }).click();
  await expect(page.getByText('Turn tracking data into a match you can watch.')).toBeVisible();
});

test('live API rejects invalid inputs and reports auth errors without exposing the key', async ({ page, context }) => {
  await context.route(base + '/**', (route) => route.fulfill({ status: 401, contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' }, body: JSON.stringify({ error: { code: 'UNAUTHORIZED' } }) }));
  await page.goto('/'); await page.getByRole('button', { name: 'Live API', exact: true }).click();
  await page.getByRole('button', { name: 'Connect live API' }).click();
  await expect(page.getByRole('alert')).toContainText('valid 32-character');
  await page.getByLabel('Session ID', { exact: true }).fill(id);
  await page.getByLabel('API key (without Bearer)').fill(key);
  await page.getByRole('button', { name: 'Connect live API' }).click();
  await expect(page.getByRole('alert')).toContainText('API key rejected');
  await expect(page.getByRole('button', { name: 'Connect live API' })).toBeEnabled();
  expect(await page.locator('body').textContent()).not.toContain(key);
});

test('live API reports late join and absent clocks instead of complete match workload', async ({ page, context }) => {
  const data = snapshot(100, 5);
  const withoutClocks = data.frames.map((frame) => { const copy: Record<string, unknown> = { ...frame }; delete copy.periodGameClockTime; return copy; });
  await context.route(base + '/**', (route) => route.fulfill({ status: 200, contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' }, body: JSON.stringify({ ...data, frames: withoutClocks }) }));
  await page.goto('/'); await page.getByRole('button', { name: 'Live API', exact: true }).click();
  await page.getByLabel('Session ID', { exact: true }).fill(id);
  await page.getByLabel('API key (without Bearer)').fill(key);
  await page.getByRole('button', { name: 'Connect live API' }).click();
  await expect(page.getByText(/Partial history: earlier frames/)).toBeVisible();
  await expect(page.getByText(/Workload.trend display requires/)).toBeVisible();
  await expect(page.getByRole('table', { name: /Live workload/ })).toHaveCount(0);
  await page.getByRole('button', { name: 'Stop receiving' }).click();
});

test('plays a 300-frame batch over ten seconds, pauses, waits, and resumes with the next batch', async ({ page, context }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', (error) => errors.push(error.message));
  let nextBatch = false;
  await context.route(base + '/**', (route) => {
    const input = nextBatch ? snapshot(240, 90, 330) : snapshot(0, 300);
    input.frames = input.frames.map((f) => ({ ...f, videoTimeMs: f.frameNum * 1000 / 30,
      periodGameClockTime: f.frameNum / 30, homePlayers: [{ ...f.homePlayers[0]!, x: f.frameNum / 10 }] }));
    return route.fulfill({ status: 200, contentType: 'application/json',
      headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' }, body: JSON.stringify(input) });
  });
  await page.goto('/'); await page.getByRole('button', { name: 'Live API', exact: true }).click();
  await page.getByLabel('Session ID', { exact: true }).fill(id);
  await page.getByLabel('API key (without Bearer)').fill(key);
  await page.getByLabel('GET interval').selectOption('250');
  await page.getByRole('button', { name: 'Connect live API' }).click();
  const status = page.getByRole('status').first();
  await expect(status).toContainText('300 frames observed');
  const sourceFrame = page.getByTestId('live-playback-frame');
  await expect(sourceFrame).toBeVisible();
  const first = Number((await sourceFrame.textContent())!.split('#')[1]);
  expect(first).toBeLessThan(60); // Receipt must not jump directly to #299.
  const table = page.getByRole('table', { name: /Live workload and output indicators/ });
  await expect(table).toBeVisible();
  const distance = table.locator('tbody tr').first().locator('td').nth(2);
  expect(parseFloat((await distance.textContent())!)).toBeLessThan(0.03);
  await page.getByRole('button', { name: 'Pause playback', exact: true }).click();
  await expect(status).toContainText('Paused');
  const pausedFrame = await sourceFrame.textContent();
  await page.waitForTimeout(400);
  expect(await sourceFrame.textContent()).toBe(pausedFrame);
  await page.getByRole('button', { name: 'Resume playback', exact: true }).click();
  await expect(status).toContainText('300 frames played', { timeout: 15000 });
  await expect(status).toContainText('Waiting for next batch');
  await expect(sourceFrame).toContainText('#299');
  await page.waitForTimeout(300);
  await expect(sourceFrame).toContainText('#299');
  nextBatch = true;
  await expect(status).toContainText('330 frames played');
  await expect(sourceFrame).toContainText('#329');
  await expect(table).toBeVisible();
  await expect(distance).toContainText('0.03 km');
  expect(errors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('buffered-playback.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop receiving' }).click();
});

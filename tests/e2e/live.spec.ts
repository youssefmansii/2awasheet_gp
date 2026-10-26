import { expect, test } from '@playwright/test';
const id = 'a'.repeat(32);
const key = 'local-test-only-key-0123456789abcdef';
const base = 'https://tracking-cv-api.youssefmansi2.workers.dev';
const frame = (n: number) => ({ frameNum: n, videoTimeMs: n * 100, period: 1, periodGameClockTime: n / 10,
  homePlayers: [{ jerseyNum: '10', x: n * 0.5, y: 0, confidence: 'HIGH', visibility: 'VISIBLE' }], awayPlayers: [], balls: [] });
const frames = (start: number, count: number) => Array.from({ length: count }, (_, i) => frame(start + i));
type Frame = Record<string, unknown> & { frameNum: number; videoTimeMs: number };
/** Mock of the server's durable batch archive: GET /batches/next?after=N returns the next stored batch, or batch=null. */
function archive(batches: Frame[][], firstOffset = 0) {
  return (url: string) => {
    const after = Number(new URL(url).searchParams.get('after'));
    let offset = firstOffset;
    const stored = batches.map((list, sequence) => { const b = { sequence, frameOffset: offset, frames: list }; offset += list.length; return b; });
    const next = stored.find((b) => b.sequence > after) ?? null;
    return JSON.stringify({ sessionId: id, receivedFrames: offset, retainedFrames: 0, lastSequence: stored.length ? stored.length - 1 : null,
      receivedAt: new Date().toISOString(), warnings: [], storedBatches: stored.length, archivedBytes: 1,
      batch: next && { ...next, frameCount: next.frames.length, firstFrameNum: next.frames[0]!.frameNum, lastFrameNum: next.frames.at(-1)!.frameNum,
        firstVideoTimeMs: next.frames[0]!.videoTimeMs, lastVideoTimeMs: next.frames.at(-1)!.videoTimeMs, bytes: 1, receivedAt: new Date().toISOString() } });
  };
}

test('one-minute GET interval connects immediately and keeps buffered playback running', async ({ page, context }) => {
  let polls = 0; const serve = archive([frames(0, 10)]);
  await context.route(base + '/**', (route) => {
    polls++;
    return route.fulfill({ status: 200, contentType: 'application/json',
      headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' }, body: serve(route.request().url()) });
  });
  await page.goto('/'); await page.getByRole('button', { name: 'Live API', exact: true }).click();
  const interval = page.getByLabel('GET interval');
  await expect(interval.locator('option[value="60000"]')).toHaveCount(1);
  await interval.selectOption({ label: '1 min' });
  await page.getByLabel('Session ID', { exact: true }).fill(id);
  await page.getByLabel('API key (without Bearer)').fill(key);
  await page.getByRole('button', { name: 'Connect live API' }).click();
  await expect(page.getByRole('status').first()).toContainText('10 frames played');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.waitForTimeout(1100); // Must not silently fall back to the default one-second poll.
  expect(polls).toBe(2); // The stored batch, then one immediate check for a newer one; then the 1-minute wait.
  await page.getByRole('button', { name: 'Stop receiving' }).click();
});

test('live API consumes stored batches one by one and stops without deleting producer data', async ({ page, context }, testInfo) => {
  const methods: string[] = []; let polls = 0; const stored = [frames(0, 10)]; const serve = archive(stored);
  await context.route(base + '/**', async (route) => {
    const req = route.request(); methods.push(req.method());
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: {
      'Access-Control-Allow-Origin': 'http://127.0.0.1:5173', 'Access-Control-Allow-Headers': 'Authorization', 'Access-Control-Allow-Methods': 'GET'
    } });
    expect(req.method()).toBe('GET'); expect(req.url()).toMatch(new RegExp(`^${base}/api/cv/sessions/${id}/batches/next\\?after=-?\\d+$`));
    expect(req.headers().authorization).toBe('Bearer ' + key); polls++;
    await route.fulfill({ status: 200, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' },
      body: serve(req.url()) });
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
  stored.push(frames(10, 5));
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
  const withoutClocks = frames(100, 5).map((frame) => { const copy: Frame = { ...frame }; delete copy.periodGameClockTime; return copy; });
  // Batches before frame 100 predate the archive: frameOffset 100 reveals the missing history.
  const serve = archive([withoutClocks], 100);
  await context.route(base + '/**', (route) => route.fulfill({ status: 200, contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' }, body: serve(route.request().url()) }));
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
  const at30 = (list: ReturnType<typeof frames>) => list.map((f) => ({ ...f, videoTimeMs: f.frameNum * 1000 / 30,
    periodGameClockTime: f.frameNum / 30, homePlayers: [{ ...f.homePlayers[0]!, x: f.frameNum / 10 }] }));
  const stored = [at30(frames(0, 300))]; const serve = archive(stored);
  await context.route(base + '/**', (route) => route.fulfill({ status: 200, contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173' }, body: serve(route.request().url()) }));
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
  stored.push(at30(frames(300, 30)));
  await expect(status).toContainText('330 frames played');
  await expect(sourceFrame).toContainText('#329');
  await expect(table).toBeVisible();
  await expect(distance).toContainText('0.03 km');
  expect(errors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('buffered-playback.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop receiving' }).click();
});

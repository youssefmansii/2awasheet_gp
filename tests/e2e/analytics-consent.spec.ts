import { expect, test } from '@playwright/test';

test('analytics requires consent per file selection, including metadata-only updates', async ({ page }, testInfo) => {
  const uploads: string[] = [];
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.route('http://localhost:8000/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const headers = { 'Access-Control-Allow-Origin': 'http://127.0.0.1:5173', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 'Access-Control-Allow-Headers': 'content-type' };
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
    if (path === '/api/matches/upload') {
      uploads.push(route.request().postData() ?? '');
      return route.fulfill({ headers, json: { match_id: '3816' } });
    }
    if (path === '/api/matches') return route.fulfill({ headers, json: {
      matches: uploads.length ? ['3816'] : [],
      details: uploads.length ? [{ match_id: '3816', tracking_bytes: 1, has_events: false, loaded: false, analytics_state: 'idle' }] : [],
      upload_enabled: true
    } });
    if (path === '/api/match') return route.fulfill({ status: 503, headers, json: { detail: 'Fixture: no computation' } });
    return route.fulfill({ headers, json: {} });
  });
  await page.goto('/');
  const tracking = Buffer.from(JSON.stringify({ gameRefId: 3816, frameNum: 1, period: 1,
    videoTimeMs: 1000, periodGameClockTime: 0, homePlayers: [{ jerseyNum: 10, x: 0, y: 0 }], awayPlayers: [], balls: [] }));
  await page.getByRole('button', { name: 'Choose Tracking file', exact: true }).locator('input').setInputFiles({
    name: 'tracking.jsonl', mimeType: 'application/x-ndjson', buffer: tracking
  });
  await page.getByRole('button', { name: 'Advanced Analytics', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Send selected files', exact: true })).toBeVisible();
  expect(uploads).toHaveLength(0);
  for (const width of [320, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 850 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.screenshot({ path: testInfo.outputPath('analytics-consent.png'), fullPage: true });
  await page.getByRole('button', { name: 'Send selected files', exact: true }).click();
  await expect.poll(() => uploads.length).toBe(1);
  await page.getByRole('button', { name: 'Match Replay', exact: true }).click();
  await page.getByRole('button', { name: 'Choose Metadata', exact: true }).locator('input').setInputFiles({
    name: 'metadata.json', mimeType: 'application/json', buffer: Buffer.from('{"pitchLength":100,"pitchWidth":64}')
  });
  // Analytics remains mounted but hidden: this must NOT trigger another POST.
  await page.getByRole('button', { name: 'Advanced Analytics', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Send selected files', exact: true })).toBeVisible();
  expect(uploads).toHaveLength(1);
  await page.getByRole('button', { name: 'Send selected files', exact: true }).click();
  await expect.poll(() => uploads.length).toBe(2);
  expect(uploads[1]).toContain('name="metadata"');
  expect(uploads[1]).not.toContain('name="tracking"');
  await expect(page.getByRole('button', { name: 'Send selected files', exact: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});

import { expect, test, type Page } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';

async function loadRecording(page: Page) {
  await page.goto('/');
  // Use the owner's real tracking sample locally; public CI has a small contract fixture.
  const path = 'samples/3816-10s.jsonl';
  const buffer = existsSync(path) ? readFileSync(path) : Buffer.from(Array.from({ length: 1001 }, (_, n) => JSON.stringify({
    frameNum: n, videoTimeMs: 100000 + n * 100, gameRefId: 3816, period: 1, periodGameClockTime: n / 10,
    homePlayers: [{ jerseyNum: '10', x: n / 100, y: 0 }], awayPlayers: [], balls: []
  })).join('\n'));
  await page.getByRole('button', { name: 'Choose Tracking file', exact: true }).locator('input').setInputFiles({
    name: 'sample.jsonl', mimeType: 'application/x-ndjson', buffer
  });
  await expect(page.getByRole('group', { name: 'Local playback view' })).toBeVisible();
}

const fixtureEvents = (gameId = 3816, withHighlights = true) => JSON.stringify((withHighlights ? [
  { time: 112, game: 'SHOT', possession: 'SH', outcome: 'G' },
  { time: 116, game: 'YC', possession: null, outcome: null },
  { time: 130, game: 'OTB', possession: 'PA', outcome: 'C' },
  { time: 152, game: 'SUB', possession: null, outcome: null }
] : [{ time: 130, game: 'OTB', possession: 'PA', outcome: 'C' }]).map((e, i) => ({
  gameId, gameEventId: i, eventTime: e.time, startTime: e.time, endTime: e.time,
  gameEvents: { gameEventType: e.game, period: 1, playerName: 'Test event actor', homeTeam: true, teamName: 'HOME' },
  possessionEvents: { possessionEventType: e.possession, shotOutcomeType: e.outcome, formattedGameClock: '00:12' },
  homePlayers: [], awayPlayers: []
})));

async function loadEvents(page: Page, body: string) {
  await page.getByRole('button', { name: 'Choose Events', exact: true }).locator('input').setInputFiles({
    name: 'test-events.json', mimeType: 'application/json', buffer: Buffer.from(body)
  });
  await expect(page.getByRole('heading', { name: 'Event timeline', exact: true })).toBeVisible();
}

test('local replay offers Timeline and automatic Highlights with sequential playback and navigation', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', (error) => errors.push(error.message));
  await loadRecording(page);
  const views = page.getByRole('group', { name: 'Local playback view' });
  await views.getByRole('button', { name: 'Highlights', exact: true }).click();
  await expect(page.getByText('Load a matching Match events JSON file to generate automatic highlights.')).toBeVisible();
  await views.getByRole('button', { name: 'Timeline', exact: true }).click();
  await loadEvents(page, fixtureEvents());
  await views.getByRole('button', { name: 'Highlights', exact: true }).click();
  const choice = page.getByLabel('Choose highlight', { exact: true });
  await expect(choice.locator('option')).toHaveCount(2); // Shot + nearby card are merged; pass is excluded.
  await expect(choice.locator('option').first()).toContainText('Goal candidate');
  await expect(choice.locator('option').first()).toContainText('Card');
  await expect(page.getByLabel('Seek through match', { exact: true })).toHaveCount(0);
  const source = page.getByTestId('local-source-frame');
  const first = Number((await source.textContent())!.split('#')[1]);
  await page.getByRole('button', { name: 'Next highlight', exact: true }).click();
  await expect(choice).toHaveValue('1');
  const second = Number((await source.textContent())!.split('#')[1]); expect(second).toBeGreaterThan(first + 100);
  await page.getByRole('button', { name: 'Previous highlight', exact: true }).click();
  await expect(choice).toHaveValue('0');
  const seek = page.getByLabel('Seek through highlights', { exact: true });
  await seek.focus(); await seek.press('End');
  await expect(choice).toHaveValue('1');
  await seek.press('Home'); await expect(choice).toHaveValue('0');
  await page.getByLabel('Speed', { exact: true }).selectOption('8');
  await page.getByRole('button', { name: 'Play highlights', exact: true }).click();
  await expect(choice).toHaveValue('1'); // Must move between clips without another click.
  await expect(page.getByRole('button', { name: 'Play highlights', exact: true })).toBeVisible(); // Stops at last clip.
  await page.getByRole('button', { name: 'Play highlights', exact: true }).click();
  await expect(choice).toHaveValue('0'); // Replay the reel after it finishes.
  await page.getByRole('button', { name: 'Pause highlights', exact: true }).click();
  await views.locator('..').screenshot({ path: testInfo.outputPath('local-highlights-desktop.png') });
  for (const width of [320, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 850 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.setViewportSize({ width: 375, height: 812 });
  await views.locator('..').screenshot({ path: testInfo.outputPath('local-highlights-mobile.png') });
  await views.getByRole('button', { name: 'Timeline', exact: true }).click();
  await expect(page.getByLabel('Seek through match', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Play replay', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('highlights reject unrelated events, handle empty results and stay out of Live API', async ({ page }) => {
  await loadRecording(page);
  await loadEvents(page, fixtureEvents(9999));
  const views = page.getByRole('group', { name: 'Local playback view' });
  await views.getByRole('button', { name: 'Highlights', exact: true }).click();
  await expect(page.getByText(/events match ID does not match the tracking file/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Play highlights', exact: true })).toHaveCount(0);
  await loadEvents(page, fixtureEvents(3816, false));
  await views.getByRole('button', { name: 'Highlights', exact: true }).click();
  await expect(page.getByText('No goals, shots, cards or substitutions have available tracking in this recording.')).toBeVisible();
  await page.getByRole('button', { name: 'Live API', exact: true }).click();
  await expect(views).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Automatic highlights playback' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Connect live API', exact: true })).toBeVisible();
});

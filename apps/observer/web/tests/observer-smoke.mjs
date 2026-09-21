import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-mvp-v1';
const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8766/';
const browser = await chromium.launch({ headless: true });
const issues = [];
const result = { browser: 'isolated headless Chromium', base, desktop: {}, mobile: {}, issues };

async function openAt(width, height) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', error => issues.push(`pageerror: ${error}`));
  page.on('console', message => { if (message.type() === 'error') issues.push(`console: ${message.text()}`); });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByRole('combobox').nth(0).selectOption({ index: 1 });
  await page.getByRole('combobox').nth(1).selectOption({ index: 1 });
  await page.getByRole('button', { name: /타석 관전하기/ }).click();
  await page.locator('.candidate').first().waitFor();
  return page;
}

try {
  const desktop = await openAt(1440, 900);
  assert.equal(await desktop.locator('.candidate').count(), 3);
  assert.equal(await desktop.locator('.timeline-pitch').count(), 0);
  await desktop.screenshot({ path: path.join(output, 'browser-desktop-pitch.png'), fullPage: true });
  await desktop.getByRole('button', { name: /다음 실제 공 확인/ }).click();
  await desktop.locator('.timeline-pitch').first().waitFor();
  assert.equal(await desktop.locator('.timeline-pitch').count(), 1);
  await desktop.screenshot({ path: path.join(output, 'browser-desktop-after-first.png'), fullPage: true });
  await desktop.getByRole('button', { name: /다음 실제 공 확인/ }).click();
  await desktop.locator('.terminal-panel').waitFor();
  assert.equal(await desktop.locator('.timeline-pitch').count(), 2);
  assert.match(await desktop.locator('.selected-pitch').innerText(), /2구/);
  await desktop.locator('.timeline-pitch').first().click();
  assert.match(await desktop.locator('.chart-caption').innerText(), /0볼 0스트라이크/);
  assert.equal(await desktop.locator('.candidate').count(), 3);
  await desktop.locator('.zone-buttons button').first().click();
  await desktop.getByRole('button', { name: '목표 메모 저장' }).click();
  await desktop.locator('.saved-note').waitFor();
  await desktop.screenshot({ path: path.join(output, 'browser-desktop-terminal.png'), fullPage: true });
  const saved = await desktop.locator('.saved-note').innerText();
  await desktop.reload({ waitUntil: 'networkidle' });
  await desktop.locator('.terminal-panel').waitFor();
  assert.equal(await desktop.locator('.timeline-pitch').count(), 2);
  assert.equal(await desktop.locator('.saved-note').innerText(), saved);
  result.desktop = { first_pitch_hidden: true, revealed_pitches: 2, old_pitch_prestate_recommendation: true, terminal_last_pitch: 2, manual_saved: true, reload_restored: true };
  await desktop.close();

  const mobile = await openAt(390, 844);
  const dimensions = await mobile.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth, scoreboard: (() => { const x = document.querySelector('.scoreboard').getBoundingClientRect(); return { left: x.left, right: x.right }; })() }));
  assert.ok(dimensions.scrollWidth <= dimensions.viewport + 1, JSON.stringify(dimensions));
  assert.ok(dimensions.scoreboard.left >= -1 && dimensions.scoreboard.right <= dimensions.viewport + 1, JSON.stringify(dimensions));
  assert.equal(await mobile.locator('.candidate').count(), 3);
  await mobile.screenshot({ path: path.join(output, 'browser-mobile-390.png'), fullPage: true });
  result.mobile = { ...dimensions, candidates: 3, no_horizontal_overflow: true };
  await mobile.close();
  assert.deepEqual(issues, []);
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'browser-smoke.json'), JSON.stringify(result, null, 2) + '\n');
}

console.log(JSON.stringify(result));

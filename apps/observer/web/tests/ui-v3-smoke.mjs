import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const before = 'http://127.0.0.1:8766/';
const after = 'http://127.0.0.1:8767/';
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/D-UI-003/browser';
const sessionId = 'a245dac4-3db6-495a-916f-500f10b77247';
fs.mkdirSync(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const report = { before, after, sessionId, screenshots: {}, checks: {}, errors: [] };
async function makePage(base, width, height, savedId = sessionId) {
  const context = await browser.newContext({ viewport: { width, height } });
  await context.addInitScript(id => localStorage.setItem('pitcheezy.observer.session.v1', id), savedId);
  const page = await context.newPage();
  page.on('pageerror', error => report.errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
  const start = performance.now();
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('.scoreboard').waitFor();
  return { context, page, loadMs: Math.round(performance.now() - start) };
}
try {
  const old = await makePage(before, 1440, 900);
  await old.page.screenshot({ path: path.join(output, 'before-desktop.png'), fullPage: true });
  report.screenshots.beforeDesktop = path.join(output, 'before-desktop.png');
  await old.context.close();
  const oldMobile = await makePage(before, 390, 844);
  await oldMobile.page.screenshot({ path: path.join(output, 'before-mobile-390.png'), fullPage: true });
  report.screenshots.beforeMobile = path.join(output, 'before-mobile-390.png');
  await oldMobile.context.close();

  const desktop = await makePage(after, 1440, 900);
  report.checks.desktopLoadMs = desktop.loadMs;
  await desktop.page.getByRole('heading', { name: /던지기 전의 저장 추천/ }).waitFor();
  await desktop.page.getByRole('heading', { name: /같은 3구, 실제 기록과 비교/ }).waitFor();
  assert.equal(await desktop.page.locator('.recommendation-stage').count(), 1);
  assert.equal(await desktop.page.locator('.comparison-stage').count(), 1);
  assert.equal(await desktop.page.locator('.event-card').count(), 1);
  assert.equal(await desktop.page.locator('.observation-details').evaluate(el => el.open), false);
  await desktop.page.screenshot({ path: path.join(output, 'after-desktop.png'), fullPage: true });
  report.screenshots.afterDesktop = path.join(output, 'after-desktop.png');
  const comparisonBefore = await desktop.page.locator('.comparison-stage').innerText();
  await desktop.page.getByText('구역 그림과 투수 기록 자세히 보기').click();
  assert.equal(await desktop.page.locator('.observation-details').evaluate(el => el.open), true);
  await desktop.page.locator('.pitch-timeline button').first().click();
  assert.match(await desktop.page.locator('.comparison-stage').innerText(), /실제 1구/);
  await desktop.page.reload({ waitUntil: 'networkidle' });
  await desktop.page.locator('.scoreboard').waitFor();
  assert.equal(await desktop.page.locator('.comparison-stage').count(), 1);
  assert.equal(await desktop.page.evaluate(() => localStorage.getItem('pitcheezy.observer.session.v1')), sessionId);
  report.checks.reloadAndTimeline = true;
  report.checks.initialComparison = comparisonBefore.replace(/\s+/g, ' ').slice(0, 240);
  await desktop.context.close();

  const incomplete = await makePage(after, 1440, 900, 'b91cc3e7-53fd-496f-9c72-83b2b67774b1');
  await incomplete.page.getByRole('button', { name: '다음 공 추천' }).click();
  await incomplete.page.getByRole('heading', { name: '아직 던지지 않은 다음 공의 추천' }).waitFor();
  await incomplete.page.getByRole('button', { name: '방금 던진 공' }).click();
  await incomplete.page.getByRole('heading', { name: /던지기 전의 저장 추천/ }).waitFor();
  report.checks.nextAndPastTiming = true;
  await incomplete.context.close();

  const mobile = await makePage(after, 390, 844);
  report.checks.mobileLoadMs = mobile.loadMs;
  const sizes = await mobile.page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth }));
  report.checks.mobileWidth = sizes;
  assert.ok(sizes.document <= sizes.viewport + 1, JSON.stringify(sizes));
  await mobile.page.screenshot({ path: path.join(output, 'after-mobile-390.png'), fullPage: true });
  report.screenshots.afterMobile = path.join(output, 'after-mobile-390.png');
  await mobile.page.getByRole('link', { name: '교체 직전 기록', exact: true }).click();
  await mobile.page.getByRole('heading', { name: '교체 직전 기록', exact: true }).waitFor();
  report.checks.inningAccessible = true;
  await mobile.context.close();

  const firstPitch = await makePage(after, 390, 844, 'c80d73f2-48ee-47e0-a954-89c6e9bda836');
  const actionBounds = await firstPitch.page.getByRole('button', { name: /다음 실제 공 확인/ }).boundingBox();
  assert.ok(actionBounds && actionBounds.y + actionBounds.height <= 844, JSON.stringify(actionBounds));
  report.checks.firstScreenAction = { y: Math.round(actionBounds.y), bottom: Math.round(actionBounds.y + actionBounds.height), visibleInFirstViewport: true };
  await firstPitch.page.screenshot({ path: path.join(output, 'after-mobile-next-390.png'), fullPage: true });
  report.screenshots.afterMobileNext = path.join(output, 'after-mobile-next-390.png');
  await firstPitch.context.close();

  const broken = await browser.newContext({ viewport: { width: 390, height: 844 } });
  await broken.route('**/api/catalog', route => route.abort());
  const errorPage = await broken.newPage();
  await errorPage.goto(after, { waitUntil: 'networkidle' });
  await errorPage.getByRole('alert').waitFor();
  await errorPage.getByRole('button', { name: '다시 연결' }).click();
  await errorPage.getByRole('alert').waitFor();
  report.checks.errorAndRetryVisible = true;
  await broken.close();
  assert.deepEqual(report.errors, []);
  report.passed = true;
} catch (error) {
  report.passed = false;
  report.error = String(error);
  throw error;
} finally {
  fs.writeFileSync(path.join(output, 'ui-v3-report.json'), JSON.stringify(report, null, 2));
  await browser.close();
}

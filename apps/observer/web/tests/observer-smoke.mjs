import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

// K/HR UI integration cases; never performance evaluation data.
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ABCD-integration-v1';
const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8769/';
const browser = await chromium.launch({ headless: true });
const issues = [];
const result = { browser: 'isolated headless Chromium (in-app browser tool unavailable)', base, cases: [], issues };
let passed = false;

async function postView(page, action) {
  const response = page.waitForResponse(r => r.request().method() === 'POST' && r.url().includes('/api/sessions'));
  await action();
  const received = await response;
  assert.equal(received.status(), 200);
  return received.json();
}

async function runCase({ game, pa, count, event, width, height }) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', error => issues.push(`pageerror: ${error}`));
  page.on('console', message => { if (message.type() === 'error') issues.push(`console: ${message.text()}`); });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByRole('combobox', { name: '경기', exact: true }).selectOption(String(game));
  await page.getByRole('combobox', { name: '타석', exact: true }).selectOption(String(pa));
  let view = await postView(page, () => page.getByRole('button', { name: /타석 관전하기/ }).click());
  await page.locator('.lead-choice').waitFor();
  assert.equal(view.history.length, 0);
  assert.equal(view.last_pitch, null);
  assert.equal(view.event_analysis, null);
  assert.equal(await page.locator('.actual-strip').count(), 0);
  assert.equal(await page.locator('.timeline-pitch').count(), 0);
  assert.equal(await page.locator('.candidate-details').getAttribute('open'), null);
  assert.equal(await page.locator('.probability-details[open]').count(), 0);
  await page.screenshot({ path: path.join(output, `browser-${event}-${width}-before.png`), fullPage: true });
  const recommendations = [];
  for (let i = 0; i < count; i++) {
    recommendations.push(view.recommendation);
    view = await postView(page, () => page.getByRole('button', { name: /다음 실제 공 확인/ }).click());
    assert.equal(view.history.length, i + 1);
    assert.deepEqual(view.last_pitch.recommendation, recommendations[i]);
    await page.locator('.same-pitch-comparison').waitFor();
  }
  assert.equal(view.complete, true);
  await page.locator('.event-card').waitFor();
  const analysis = view.event_analysis;
  assert.equal(analysis.status, 'partial');
  assert.equal(analysis.evidence.reference_status, 'compatible');
  assert.equal(analysis.linkage.pitch_id, view.last_pitch.id);
  assert.equal(analysis.linkage.recommendation_id, recommendations.at(-1).id);
  assert.ok(Number.isFinite(analysis.values.total_pp));
  assert.equal(analysis.components.unallocated_residual_pp, analysis.values.total_pp);
  for (const name of ['strategy_contrast_pp', 'execution_contrast_pp', 'outcome_residual_pp']) {
    assert.equal(analysis.components[name].value_pp, null);
    assert.equal(analysis.components[name].abs_share, null);
  }
  assert.match(await page.locator('.event-total').innerText(), /마지막 공의 사전 기준/);
  await page.locator('.event-technical > summary').click();
  assert.equal(await page.locator('.event-components strong').allTextContents().then(x => x.filter(y => y === '계산 불가').length), 3);
  await page.locator('.timeline-pitch').first().click();
  await page.locator('.observation-details > summary').click();
  assert.match(await page.locator('.chart-caption').innerText(), /0볼 0스트라이크/);
  await page.locator('.zone-buttons button').first().click();
  view = await postView(page, () => page.getByRole('button', { name: '목표 메모 저장' }).click());
  await page.locator('.saved-note').waitFor();
  assert.deepEqual(view.event_analysis, analysis);
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.saved-note').waitFor();
  const restored = await page.request.get(`${base}api/sessions/${view.id}`).then(r => r.json());
  assert.deepEqual(restored.event_analysis, analysis);
  assert.deepEqual(restored.history.map(p => p.recommendation), recommendations);
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
  assert.ok(dimensions.scrollWidth <= dimensions.viewport + 1, JSON.stringify(dimensions));
  await page.screenshot({ path: path.join(output, `browser-${event}-${width}-after.png`), fullPage: true });
  result.cases.push({ game, pa, event, width, count, session_id: view.id, total_pp: analysis.values.total_pp,
    status: analysis.status, first_pitch_hidden: true, saved_same_pitch_recommendations_unchanged: true,
    manual_note_not_used_as_intent: true, reload_restored: true, no_horizontal_overflow: true, dimensions });
  await page.close();
}

try {
  await runCase({ game: 776703, pa: 3, count: 3, event: 'strikeout', width: 1440, height: 1000 });
  await runCase({ game: 776710, pa: 10, count: 2, event: 'home_run', width: 390, height: 844 });
  assert.deepEqual(issues, []);
  passed = true;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'browser-smoke.json'), JSON.stringify({ ...result, passed }, null, 2) + '\n');
}
console.log(JSON.stringify(result));

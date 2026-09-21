/** Real API integration plus test-only boundary fixtures for observational sample gates. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8766/';
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-context-ui-v1';
fs.mkdirSync(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [], report = { passed: false, checks: {}, errors };
page.on('pageerror', (error) => errors.push(String(error)));
page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
const cutoff = '타자 성향·구종 구성은 경기 전날까지의 기록으로 계산';
const boundary = {
  prior_pitch_count: 54, times_facing_batter: 2, condition_inference: false,
  reference_window_days: 90, hidden_current_pitch_type: 'HIDDEN_TEST_PITCH',
  speed_by_pitch_type: [
    { pitch_type: 'SL', pitch_label: '슬라이더', recent_pitch_count: 5, recent_measured_count: 2, recent_mean_mph: 85, prior90_measured_count: 50, prior90_mean_mph: 75.1, delta_mph: 9.9 },
    { pitch_type: 'CU', pitch_label: '커브', recent_pitch_count: 0, recent_measured_count: 0, recent_mean_mph: null, prior90_measured_count: 100, prior90_mean_mph: 80, delta_mph: null },
    { pitch_type: 'FF', pitch_label: '포심', recent_pitch_count: 5, recent_measured_count: 3, recent_mean_mph: 93.5, prior90_measured_count: 30, prior90_mean_mph: 95, delta_mph: -1.5 },
    { pitch_type: 'CH', pitch_label: '체인지업', recent_pitch_count: 5, recent_measured_count: 5, recent_mean_mph: 84, prior90_measured_count: 29, prior90_mean_mph: 75.2, delta_mph: 8.8 },
  ],
};
let mode = 'live';

try {
  await page.route('**/api/sessions**', async (route) => {
    const response = await route.fetch();
    if (!response.ok()) return route.fulfill({ response });
    const body = await response.json();
    if (mode === 'boundary') {
      body.context = boundary;
      body.context_notes = [cutoff, '테스트용 중복 구속 메모 888.8 mph'];
    } else if (mode === 'legacy') {
      delete body.context;
      body.context_notes = ['이 투구 전까지 던진 공 12개', cutoff];
    }
    await route.fulfill({ response, json: body });
  });
  await page.goto(base, { waitUntil: 'networkidle' });
  const created = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/sessions' && response.request().method() === 'POST');
  await page.getByRole('button', { name: /타석 관전하기/ }).click();
  const live = await (await created).json();
  assert.ok(live.context, 'The v2 service must publish structured pre-pitch context.');
  await page.locator('.observed-summary-card').waitFor();
  assert.ok((await page.locator('.observed-summary').innerText()).includes(String(live.context.prior_pitch_count)));
  report.checks.real_v2_context_rendered = true;

  mode = 'boundary';
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.observed-context').waitFor();
  const primary = await page.locator('.observed-context > .speed-table [data-pitch-type]').evaluateAll((rows) => rows.map((row) => row.getAttribute('data-pitch-type')));
  assert.deepEqual(primary, ['CH', 'FF', 'SL']);
  const fastball = page.locator('[data-pitch-type="FF"]');
  assert.equal(await fastball.locator('.speed-comparison').innerText(), '-1.5');
  assert.match(await fastball.innerText(), /최근 측정 N=3/);
  assert.match(await fastball.innerText(), /과거 측정 N=30/);
  for (const type of ['CH', 'SL']) assert.equal(await page.locator(`[data-pitch-type="${type}"] .speed-comparison`).innerText(), '표본 적음');
  const card = await page.locator('.observed-context').innerText();
  assert.ok(!card.includes('9.9') && !card.includes('8.8'));
  assert.equal(await page.locator('.other-speeds').evaluate((element) => element.open), false);
  await page.locator('.other-speeds > summary').click();
  assert.equal(await page.locator('[data-pitch-type="CU"] .speed-comparison').innerText(), '표본 적음');
  assert.match(await page.locator('.condition-note').innerText(), /피로·부상.*교체.*아닙니다/);
  assert.equal(await page.locator('.profile-cutoff-notes').innerText(), cutoff);
  const body = await page.locator('body').innerText();
  assert.ok(!body.includes('888.8') && !body.includes('HIDDEN_TEST_PITCH'));
  assert.ok(!body.includes('NaN'));
  report.checks.thresholds_recent3_past30_and_no_hidden_type = true;
  report.checks.top3_order_remainder_and_deduplicated_notes = true;
  await page.locator('.observed-context').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, 'context-desktop-boundary-fixture.png'), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
  assert.ok(dimensions.scrollWidth <= dimensions.viewport + 1, JSON.stringify(dimensions));
  await page.screenshot({ path: path.join(output, 'context-mobile-boundary-fixture.png'), fullPage: true });
  report.checks.mobile_layout = dimensions;

  mode = 'legacy';
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.context-notes').waitFor();
  assert.equal(await page.locator('.observed-context').count(), 0);
  assert.match(await page.locator('.context-notes').innerText(), /이 투구 전까지 던진 공 12개/);
  assert.ok((await page.locator('.context-notes').innerText()).includes(cutoff));
  report.checks.v1_notes_fallback_preserved = true;
  assert.deepEqual(errors, []);
  report.passed = true;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'context-card-smoke.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify(report));

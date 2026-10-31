import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8771';
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/tmp/pitcheezy-inning-ui';
fs.mkdirSync(output, { recursive: true });
const report = { passed: false, base, checks: {}, errors: [] };
const browser = await chromium.launch({ headless: true });
const resultRoute = '**/api/inning-decisions/*/resolve';
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    const context = await browser.newContext({ viewport: { width, height } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(String(error)));
    const requests = [];
    page.on('request', request => requests.push(new URL(request.url()).pathname));
    await page.addInitScript(() => localStorage.setItem('pitcheezy.observer.session.v1', 'untouched-session-marker'));
    await page.goto(`${base}/inning-decisions`);
    await page.getByRole('radio').first().waitFor();
    assert.equal(requests.filter(url => url.endsWith('/resolve')).length, 0);
    assert.equal(await page.getByRole('heading', { name: '조건부 이닝 전망' }).count(), 0);
    await page.getByRole('radio').first().check();
    assert.equal(requests.filter(url => url.endsWith('/resolve')).length, 0);
    await page.getByRole('button', { name: '이닝 전망 보기' }).click();
    await page.getByRole('heading', { name: '조건부 이닝 전망' }).waitFor();
    assert.match(await page.locator('.inning-result').innerText(), /52\.10%–53\.47%/);
    assert.match(await page.locator('.inning-result').innerText(), /미해결 확률 1\.36%/);
    assert.match(await page.locator('.inning-result').innerText(), /실제 교체 효과 미측정/);
    assert.match(await page.locator('.inning-situation').innerText(), /원정 2/);
    assert.equal(await page.evaluate(() => localStorage.getItem('pitcheezy.observer.session.v1')), 'untouched-session-marker');
    assert.ok(!requests.some(url => /\/api\/sessions/.test(url)));
    const size = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
    assert.ok(size.scroll <= size.width + 1, JSON.stringify(size));
    await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });
    report.checks[name] = { real_api: true, explicit_selection: true, session_untouched: true, ...size };

    await page.route(resultRoute, route => route.fulfill({ status: 409, json: { detail: '선택한 교체 시점이나 경기 상태가 저장된 기록과 다릅니다.' } }));
    await page.getByRole('button', { name: '이닝 전망 보기' }).click();
    await page.getByRole('alert').waitFor();
    assert.equal(await page.locator('.inning-result').count(), 0);
    await page.unroute(resultRoute);
    await page.getByRole('button', { name: '다시 시도', exact: true }).click();
    await page.getByRole('heading', { name: '조건부 이닝 전망' }).waitFor();
    report.checks[`${name}_conflict_retry`] = true;
    await context.close();
  }

  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  page.on('pageerror', error => report.errors.push(String(error)));
  await page.route('**/api/inning-decision-games', route => route.fulfill({ json: { schema_version: 'inning-decision-v1', mode: 'historical_decision_review', games: [] } }));
  await page.goto(`${base}/inning-decisions`);
  await page.getByText('아직 등록된 교체 직전 기록이 없어요.').waitFor();
  assert.equal(await page.locator('.inning-result').count(), 0);
  await page.screenshot({ path: path.join(output, 'empty-mobile.png'), fullPage: true });
  report.checks.empty = true;
  await page.unroute('**/api/inning-decision-games');
  await page.getByRole('button', { name: '목록 새로고침' }).click();
  await page.getByRole('radio').first().check();
  await page.route(resultRoute, async route => {
    const response = await route.fetch();
    const payload = await response.json();
    payload.context.initial_state.outs = 1;
    await route.fulfill({ json: payload });
  });
  await page.getByRole('button', { name: '이닝 전망 보기' }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.locator('.inning-result').count(), 0);
  report.checks.invalid_200_hidden = true;
  await page.unroute(resultRoute);

  await page.route(resultRoute, async route => {
    const response = await route.fetch();
    const payload = await response.json();
    payload.result.status = 'unavailable';
    payload.result.reason = 'missing_evaluation_result';
    payload.result.estimate.lower = payload.result.estimate.upper = null;
    payload.result.coverage = { resolved_mass: null, unresolved_mass: null, unresolved_reasons: {}, model_calls: null };
    await route.fulfill({ json: payload });
  });
  await page.getByRole('button', { name: '다시 시도', exact: true }).click();
  await page.getByRole('heading', { name: '조건부 이닝 전망' }).waitFor();
  assert.match(await page.locator('.inning-result').innerText(), /계산 결과 없음/);
  assert.doesNotMatch(await page.locator('.inning-result').innerText(), /52\.10|53\.47|0\.00%/);
  report.checks.synthetic_unavailable_no_numbers = true;
  await page.unroute(resultRoute);

  // Synthetic second game lets a selection overtake an in-flight real result.
  await page.route('**/api/inning-decision-games', async route => {
    const response = await route.fetch();
    const payload = await response.json();
    payload.games.push({ game_id: 777064, date: '2025-07-22', decision_count: 1 });
    await route.fulfill({ json: payload });
  });
  await page.reload();
  await page.getByRole('radio').first().check();
  let release;
  let completed;
  const gate = new Promise(resolve => { release = resolve; });
  const done = new Promise(resolve => { completed = resolve; });
  await page.route(resultRoute, async route => {
    const response = await route.fetch();
    await gate;
    try { await route.fulfill({ response }); } catch { /* Aborted by the selection change. */ }
    finally { completed(); }
  });
  await page.getByRole('button', { name: '이닝 전망 보기' }).click();
  await page.getByRole('button', { name: /전망 확인 중/ }).waitFor();
  await page.locator('#inning-game').selectOption('777064');
  await page.getByText('이 경기에는 등록된 교체 직전 기록이 없습니다.').waitFor();
  release(); await done;
  assert.equal(await page.locator('.inning-result').count(), 0);
  report.checks.late_result_after_selection_hidden = true;
  await page.unroute(resultRoute);
  await page.unroute('**/api/inning-decision-games');

  await page.route('**/api/inning-decision-games', route => route.abort());
  await page.reload();
  await page.getByRole('alert').waitFor();
  assert.match(await page.getByRole('alert').innerText(), /서버에 연결하지 못했어요/);
  await page.unroute('**/api/inning-decision-games');
  await page.getByRole('button', { name: '다시 시도', exact: true }).click();
  await page.getByRole('radio').first().waitFor();
  report.checks.offline_retry = true;

  await page.getByRole('link', { name: /관전 화면으로/ }).click();
  await page.getByRole('link', { name: '교체 직전 기록', exact: true }).waitFor();
  const homeSize = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
  assert.ok(homeSize.scroll <= homeSize.width + 1, JSON.stringify(homeSize));
  await page.getByRole('link', { name: '교체 직전 기록', exact: true }).click();
  await page.getByRole('heading', { name: '교체 직전 기록', exact: true }).waitFor();
  report.checks.home_navigation = homeSize;
  await context.close();
  assert.deepEqual(report.errors, []);
  report.passed = true;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify(report));

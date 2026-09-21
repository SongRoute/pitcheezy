/** Expanded-catalog UI test. Uses real public metadata and session APIs.
 * A test-only unknown field checks that search never indexes outcome payloads.
 * Does not depend on a particular batter, PA length, outcome, or recommendation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8766/';
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-browse-v1';
fs.mkdirSync(output, { recursive: true });
const sessionKey = 'pitcheezy.observer.session.v1';
const hiddenToken = 'test_hidden_outcome_not_searchable_937';
const browser = await chromium.launch({ headless: true });
const issues = [];
const report = { base, passed: false, checks: {}, issues };
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();
page.on('pageerror', (error) => issues.push(String(error)));
page.on('console', (message) => { if (message.type() === 'error') issues.push(message.text()); });
const pathname = (response) => new URL(response.url()).pathname;

async function sessionAction(button, predicate) {
  const response = page.waitForResponse(predicate);
  await button.click();
  const returned = await response;
  assert.equal(returned.status(), 200);
  const view = await returned.json();
  await page.waitForFunction(({ key, id }) => localStorage.getItem(key) === id, { key: sessionKey, id: view.id });
  await page.locator('.scoreboard').waitFor();
  await page.waitForFunction((count) => document.querySelectorAll('.timeline-pitch').length === count, view.history.length);
  return view;
}

async function openPicker() {
  if (!await page.locator('.session-disclosure').evaluate((element) => element.open)) {
    await page.locator('.session-disclosure > summary').click();
  }
}

try {
  const response = await page.request.get(new URL('/api/catalog', base).href);
  assert.equal(response.status(), 200);
  const catalog = await response.json();
  const game = catalog.games.find((item) => item.plate_appearances?.length >= 2);
  assert.ok(game, 'Expanded-catalog validation needs a game with at least two public PA metadata entries.');
  const pa = game.plate_appearances[0];
  report.game_id = game.id;
  report.public_catalog_games = catalog.games.length;
  report.public_game_pas = game.plate_appearances.length;

  await page.route('**/api/catalog', async (route) => {
    const upstream = await route.fetch();
    const body = await upstream.json();
    body.games[0].test_future_outcome = hiddenToken;
    body.games[0].plate_appearances[0].test_future_outcome = hiddenToken;
    await route.fulfill({ response: upstream, json: body });
  });
  await page.goto(base, { waitUntil: 'networkidle' });
  const search = page.getByLabel('팀·투수·타자 검색', { exact: true });
  await search.fill(hiddenToken);
  await page.locator('.catalog-empty').waitFor();
  assert.match(await page.locator('.catalog-count').innerText(), /^0경기/);
  assert.equal(await page.getByRole('button', { name: '타석 관전하기', exact: true }).count(), 0);
  await page.getByRole('button', { name: '전체 경기 보기', exact: true }).click();
  assert.equal(await search.inputValue(), '');
  report.checks.outcome_fields_excluded_from_search = true;

  await search.fill(`${game.home_team} ${pa.pitcher_label}`);
  await page.getByLabel('경기 날짜', { exact: true }).fill(game.date);
  await page.getByRole('combobox', { name: '경기', exact: true }).selectOption(String(game.id));
  await page.getByRole('combobox', { name: '타석', exact: true }).selectOption(String(pa.id));
  const gameOptions = await page.getByRole('combobox', { name: '경기', exact: true }).locator('option:not([disabled])').allTextContents();
  assert.ok(gameOptions.length > 0 && gameOptions.every((text) => text.includes(game.date)));
  report.checks.team_player_and_date_filter = true;

  const first = await sessionAction(page.getByRole('button', { name: /타석 관전하기/ }), (result) => pathname(result) === '/api/sessions' && result.request().method() === 'POST');
  assert.equal(first.game.id, game.id);
  assert.equal(first.plate_appearance.id, pa.id);
  assert.equal(first.history.length, 0);
  assert.equal(first.last_pitch, null);
  assert.equal(await page.locator('.actual-strip').count(), 0);
  if (first.recommendation?.status === 'ready') {
    assert.deepEqual(await page.locator('.recommendation-basis li').allTextContents(), first.recommendation.basis);
    assert.equal(await page.locator('.probability-details').evaluate((element) => element.open), false);
    await page.locator('.probability-details > summary').click();
    assert.match(await page.locator('.probability-details').innerText(), /모델/);
    assert.match(await page.locator('.probability-details').innerText(), /실제 경기.*검증/);
    report.checks.server_grounded_basis_and_explicit_model_values = true;
  } else {
    assert.match(await page.locator('.recommendations').innerText(), /추천을 제공할 수 없어요/);
    report.checks.honest_unavailable_recommendation = true;
  }

  const revealed = await sessionAction(page.getByRole('button', { name: /다음 실제 공 확인/ }), (result) => pathname(result) === `/api/sessions/${first.id}/advance` && result.request().method() === 'POST');
  assert.equal(revealed.history.length, 1);
  const before = revealed.last_pitch.pre_state;
  assert.match(await page.locator('.chart-caption').innerText(), new RegExp(`${before.balls}볼 ${before.strikes}스트라이크`));
  if (revealed.last_pitch.recommendation?.status === 'ready') {
    assert.deepEqual(await page.locator('.recommendation-basis li').allTextContents(), revealed.last_pitch.recommendation.basis);
  }
  report.checks.saved_pre_pitch_comparison_preserved = true;

  const following = await sessionAction(page.getByRole('button', { name: /다음 타석.*새 관전/ }), (result) => pathname(result) === '/api/sessions' && result.request().method() === 'POST');
  assert.equal(following.plate_appearance.id, game.plate_appearances[1].id);
  assert.equal(following.history.length, 0);
  assert.notEqual(following.id, first.id);
  const restored = await sessionAction(page.getByRole('button', { name: /이전 타석.*이어보기/ }), (result) => pathname(result) === `/api/sessions/${first.id}` && result.request().method() === 'GET');
  assert.equal(restored.id, first.id);
  assert.equal(restored.history.length, 1);
  report.checks.adjacent_pa_new_session_and_existing_resume = true;

  await openPicker();
  await sessionAction(page.getByRole('button', { name: '현재 관전 계속보기', exact: true }), (result) => pathname(result) === `/api/sessions/${first.id}` && result.request().method() === 'GET');
  await openPicker();
  const restarted = await sessionAction(page.getByRole('button', { name: /이 타석 처음부터 보기/ }), (result) => pathname(result) === '/api/sessions' && result.request().method() === 'POST');
  assert.notEqual(restarted.id, first.id);
  assert.equal(restarted.history.length, 0);
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.scoreboard').waitFor();
  assert.equal(await page.evaluate((key) => localStorage.getItem(key), sessionKey), restarted.id);
  assert.equal(await page.locator('.timeline-pitch').count(), 0);
  report.checks.restart_is_distinct_and_reload_resumes = true;

  await openPicker();
  await search.fill(hiddenToken);
  await page.locator('.catalog-empty').waitFor();
  assert.equal(await page.locator('.scoreboard').count(), 1);
  assert.equal(await page.evaluate((key) => localStorage.getItem(key), sessionKey), restarted.id);
  await page.getByRole('button', { name: '전체 경기 보기', exact: true }).click();
  report.checks.empty_search_preserves_current_view = true;
  await page.locator('.session-disclosure > summary').click();
  await page.screenshot({ path: path.join(output, 'browse-desktop.png'), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await openPicker();
  await search.fill(pa.batter_label);
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
  assert.ok(dimensions.scrollWidth <= dimensions.viewport + 1, JSON.stringify(dimensions));
  await page.screenshot({ path: path.join(output, 'browse-mobile-search.png'), fullPage: true });
  report.checks.batter_search_and_mobile_layout = dimensions;
  assert.deepEqual(issues, []);
  report.passed = true;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'browse-smoke.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify(report));

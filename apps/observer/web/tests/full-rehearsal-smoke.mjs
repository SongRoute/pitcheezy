/** Real-server browser rehearsal. The child smokes exercise the public UI and APIs. */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const scripts = [
  ['observer-replay', 'observer-smoke.mjs', 'browser-smoke.json'],
  ['inning-decisions', 'inning-ui-smoke.mjs', 'browser-report.json'],
];
const here = path.dirname(fileURLToPath(import.meta.url));
const base = new URL(process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8766/').href;
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-FULL-001/browser';
const report = { passed: false, base, checks: {}, suites: {}, error: null };
fs.mkdirSync(output, { recursive: true });

async function getJson(url) {
  const response = await fetch(new URL(url, base), { signal: AbortSignal.timeout(15_000) });
  assert.equal(response.status, 200, `${url}: HTTP ${response.status}`);
  return response.json();
}

async function checkNavigation() {
  const browser = await chromium.launch({ headless: true });
  const issues = [];
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', error => issues.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') issues.push(message.text()); });
  try {
    const catalog = await getJson('/api/catalog');
    const game = catalog.games.find(item => item.plate_appearances?.length >= 2);
    assert.ok(game, 'Adjacent PA navigation requires two real catalog entries.');
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.getByRole('combobox', { name: '경기', exact: true }).selectOption(String(game.id));
    await page.getByRole('combobox', { name: '타석', exact: true }).selectOption(String(game.plate_appearances[0].id));
    const startedResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/sessions' && response.request().method() === 'POST');
    await page.getByRole('button', { name: /타석 관전하기/ }).click();
    const first = await (await startedResponse).json();
    assert.equal(first.plate_appearance.id, game.plate_appearances[0].id);
    assert.equal(first.history.length, 0);
    assert.equal(first.recommendation.status, 'ready');

    const advanceResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/sessions/${first.id}/advance` && response.request().method() === 'POST');
    await page.getByRole('button', { name: /다음 실제 공 확인/ }).click();
    const advanced = await (await advanceResponse).json();
    assert.deepEqual(advanced.last_pitch.recommendation, first.recommendation);
    const nextResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/sessions' && response.request().method() === 'POST');
    await page.getByRole('button', { name: /다음 타석.*새 관전/ }).click();
    const next = await (await nextResponse).json();
    assert.equal(next.plate_appearance.id, game.plate_appearances[1].id);
    assert.notEqual(next.id, first.id);
    const previousResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/sessions/${first.id}` && response.request().method() === 'GET');
    await page.getByRole('button', { name: /이전 타석.*이어보기/ }).click();
    const restored = await (await previousResponse).json();
    assert.equal(restored.id, first.id);
    assert.deepEqual(restored.last_pitch.recommendation, first.recommendation);
    await page.getByRole('link', { name: '교체 직전 기록', exact: true }).click();
    await page.getByRole('heading', { name: '교체 직전 기록', exact: true }).waitFor();
    assert.equal(await page.evaluate(() => localStorage.getItem('pitcheezy.observer.session.v1')), first.id);
    await page.getByRole('link', { name: /관전 화면으로/ }).click();
    await page.locator('.scoreboard').waitFor();
    assert.equal(await page.evaluate(() => localStorage.getItem('pitcheezy.observer.session.v1')), first.id);
    assert.deepEqual(issues, []);
    const dimensions = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
    assert.ok(dimensions.scrollWidth <= dimensions.viewport + 1, JSON.stringify(dimensions));
    await page.screenshot({ path: path.join(output, 'observer-navigation.png'), fullPage: true });
    return { game_id: game.id, first_pa: first.plate_appearance.id, next_pa: next.plate_appearance.id,
      preserved_recommendation_id: first.recommendation.id, returned_session_id: first.id,
      inning_route_and_back: true, no_horizontal_overflow: true, dimensions };
  } finally {
    await browser.close();
  }
}

try {
  const health = await getJson('/api/health');
  assert.equal(health.status, 'ok');
  assert.equal(health.dataset_ready, true);
  assert.equal(health.model_ready, true);
  report.checks.health = health;

  const games = await getJson('/api/inning-decision-games');
  assert.ok(games.games.some(game => game.game_id === 777063 && game.decision_count > 0),
    'Real decision game 777063 must be available.');
  const decisions = await getJson('/api/inning-decisions?game_id=777063');
  assert.ok(decisions.decisions.length > 0, 'Game 777063 must contain a real decision.');
  report.checks.inning_decisions = {
    game_id: 777063,
    count: decisions.decisions.length,
    decision_ids: decisions.decisions.map(decision => decision.decision_id),
  };

  report.checks.navigation = await checkNavigation();

  for (const [name, script, filename] of scripts) {
    const suiteOutput = path.join(output, name);
    fs.mkdirSync(suiteOutput, { recursive: true });
    const child = spawnSync(process.execPath, [path.join(here, script)], {
      cwd: path.dirname(here),
      env: { ...process.env, OBSERVER_SMOKE_URL: name === 'inning-decisions' ? base.replace(/\/$/, '') : base,
        OBSERVER_SMOKE_OUTPUT: suiteOutput },
      encoding: 'utf8',
      timeout: 300_000,
      maxBuffer: 4 * 1024 * 1024,
    });
    fs.writeFileSync(path.join(suiteOutput, 'stdout.log'), child.stdout || '');
    fs.writeFileSync(path.join(suiteOutput, 'stderr.log'), child.stderr || '');
    const suiteReportPath = path.join(suiteOutput, filename);
    const suiteReport = fs.existsSync(suiteReportPath) ? JSON.parse(fs.readFileSync(suiteReportPath, 'utf8')) : null;
    report.suites[name] = {
      passed: child.status === 0 && suiteReport?.passed === true,
      exit_code: child.status,
      signal: child.signal,
      report: suiteReportPath,
      ...(suiteReport ? { checks: suiteReport.checks ?? suiteReport.cases, issues: suiteReport.issues ?? suiteReport.errors } : {}),
    };
    assert.equal(child.error, undefined, `${name}: ${child.error}`);
    assert.equal(child.status, 0, `${name}: ${child.stderr || child.stdout}`);
    assert.equal(suiteReport?.passed, true, `${name}: report did not pass`);
  }
  report.passed = true;
} catch (error) {
  report.error = String(error);
  process.exitCode = 1;
} finally {
  fs.writeFileSync(path.join(output, 'full-rehearsal-report.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify(report));

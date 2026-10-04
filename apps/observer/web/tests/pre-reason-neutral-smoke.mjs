// A "no recommendation" reason that was decided from the actual row (pitch-clock call, plate appearance over
// without a pitch, missing type, thrown type outside TRAIN) must not show before the reveal (Opus review F1 of
// 585358c, T48d). Pre-decision reasons (unknown pitcher) still show at once; after the reveal the real reason shows.
// Part 1 pins the UI text list to STATUS_TEXT in scripts/demo_precompute.py. Part 2: synthetic game on the built dist.
//   PZ_WEB_DIST=/tmp/dist PLAYWRIGHT_BROWSERS_PATH=... node tests/pre-reason-neutral-smoke.mjs
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, '..', '..', '..', '..');
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };

// Part 1: the four actual-dependent STATUS_TEXT entries appear verbatim in the UI list, and nothing else does.
const OUTCOME = ['UNSUPPORTED_NO_LOGGED_ACTION', 'NO_DECISION', 'UNSUPPORTED_MISSING_ACTION_LABEL', 'UNSUPPORTED_LOGGING_POSITIVITY'];
const precompute = readFileSync(path.join(repo, 'scripts', 'demo_precompute.py'), 'utf8');
const statusText = Object.fromEntries([...precompute.matchAll(/'([A-Z_]+)': '([^']+)'/g)].map((m) => [m[1], m[2]]));
const ui = readFileSync(path.join(here, '..', 'src', 'WatchAlong.tsx'), 'utf8');
const block = /const OUTCOME_REASONS = new Set\(\[([\s\S]*?)\]\);/.exec(ui);
const listed = block ? [...block[1].matchAll(/'([^']+)'/g)].map((m) => m[1]) : [];
const expected = OUTCOME.map((k) => statusText[k]);
if (expected.some((t) => !t)) fail(`STATUS_TEXT missing one of ${OUTCOME.join(', ')}`);
if (JSON.stringify([...listed].sort()) !== JSON.stringify([...expected].sort())) fail(`UI list ${JSON.stringify(listed)} != STATUS_TEXT ${JSON.stringify(expected)}`);
const NEUTRAL = /const PRE_NEUTRAL_REASON = '([^']+)'/.exec(ui)?.[1];
if (!NEUTRAL) fail('no PRE_NEUTRAL_REASON');

// Part 2: synthetic timeline, one decision per reason.
const REASONS = [statusText.UNSUPPORTED_UNKNOWN_PITCHER, ...expected];
const decisions = REASONS.map((reason, index) => ({ index, pa_id: `900100:${index + 1}`, at_bat_number: index + 1, pitch_number: 1,
  situation: { inning: 1, half: 'Top', outs: Math.min(index, 2), balls: 0, strikes: 0, bases: 0, home_score: 0, away_score: 0 },
  pitcher: { id: 1, name: '홈 투수', hand: 'R' }, batter: { id: 100 + index, name: `타자${index + 1}`, side: 'R' },
  pre: { status: 'unsupported', reason, recommendation: null } }));
const timeline = { schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전',
  game: { game_pk: 900100, date: '2025-09-01', game_type: 'R', away_team: 'Away Club', home_team: 'Home Club' },
  policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: 0.5, validated: false, note: '' },
  location: { basis: 'synthetic', evaluated: false, note: '', zone_bounds: { bottom: 1.6, top: 3.3899 } },
  we_note: '', coverage: { pitch_decisions: decisions.length, ready: 0, ready_share_of_pitches: 0 }, decisions };
const reveal = (i) => ({ index: i, actual: { pitch_type: null, pitch_label: '투구 없음', description: 'automatic_ball', result_label: 'automatic_ball',
  event: null, event_label: null, play_text: null, speed_mph: null, x: null, z: null, zone_label: '위치 정보 없음' },
  we: { home_before: 0.5, home_after: 0.5, home_delta: 0, batting_delta: 0 } });

const browser = await chromium.launch();
const results = [];
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    await context.addInitScript(() => { try { localStorage.setItem('pitcheezy.guide.v1', '1'); } catch { /* ignore */ } });
    await context.route('http://pz.test/**', async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/watch/900100') return route.fulfill({ json: timeline });
      const hit = url.pathname.match(/^\/api\/watch\/900100\/reveal\/(\d+)$/);
      if (hit) return route.fulfill({ json: reveal(Number(hit[1])) });
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
      const file = path.join(dist, url.pathname);
      const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    for (const d of decisions) {
      await page.goto(`http://pz.test/watch?game=900100#i=${d.index}`);
      await page.getByText('추천 없음').waitFor({ timeout: 8000 });
      const before = await page.locator('.wa-reason').first().innerText();
      const outcome = expected.includes(d.pre.reason);
      if (outcome && before !== NEUTRAL) fail(`${width} #${d.index}: before reveal shows "${before}"`);
      if (outcome && /자동|투구 없이|구종 정보|실제 던진/.test(before)) fail(`${width} #${d.index}: outcome words before reveal`);
      if (!outcome && before !== d.pre.reason) fail(`${width} #${d.index}: pre-decision reason hidden: "${before}"`);
      if (await page.locator('.wa-reveal').count()) fail(`${width} #${d.index}: reveal visible before click`);
      await page.getByRole('button', { name: '실제 투구 공개' }).click();
      await page.locator('.wa-reveal').waitFor({ timeout: 5000 });
      const after = await page.locator('.wa-reason').first().innerText();
      if (after !== d.pre.reason) fail(`${width} #${d.index}: after reveal shows "${after}"`);
      results.push({ width, index: d.index, outcome, before: before.slice(0, 40), after: after.slice(0, 40) });
    }
    // next pitch starts unrevealed again (neutral line back); a fresh load so the reveals cached above are gone
    await page.goto('about:blank');
    await page.goto(`http://pz.test/watch?game=900100#i=1`);
    await page.getByRole('button', { name: '실제 투구 공개' }).click();
    await page.locator('.wa-reveal').waitFor({ timeout: 5000 });
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => window.location.hash === '#i=2', null, { timeout: 3000 }).catch(() => fail(`${width}: → did not move`));
    if ((await page.locator('.wa-reason').first().innerText()) !== NEUTRAL) fail(`${width}: next pitch did not reset to the neutral line`);
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    if (scrollWidth > width) fail(`${width}: horizontal overflow ${scrollWidth}`);
    if (errors.length) fail(`${width}: page errors ${errors.join(' | ')}`);
    await context.close();
  }
} finally { await browser.close(); }
console.log(JSON.stringify({ checked: results.length, outcome_texts: expected.length, results }, null, 0));
if (!process.exitCode) console.log('ok');

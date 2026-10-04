// Watch-along navigation must not tell how the game ends before the viewer gets there (spoiler audit T23):
// the half-inning row is always 1초–9말 (a 9말 that was never played jumps to the last pitch), extra innings appear only
// once reached (remembered on this device), the plate-appearance list stops at the latest pitch reached and the progress
// shows no total. Two synthetic games (ends after 9초 / goes to 10말), canned responses on the built dist/, no server.
//   PZ_WEB_DIST=/tmp/dist PLAYWRIGHT_BROWSERS_PATH=... node tests/watch-spoiler-smoke.mjs
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };
const REGULAR = Array.from({ length: 18 }, (_, i) => `${Math.floor(i / 2) + 1}${i % 2 ? '말' : '초'}`);

function timeline(gamePk, lastInning, endsAfterTop) {
  const decisions = [];
  for (let inning = 1; inning <= lastInning; inning++) for (const half of ['Top', 'Bot']) {
    if (endsAfterTop && inning === lastInning && half === 'Bot') continue;
    for (let pa = 0; pa < 3; pa++) for (let pitch = 1; pitch <= 2; pitch++) {
      const index = decisions.length, ab = Math.floor(index / 2) + 1;
      decisions.push({ index, pa_id: `${gamePk}:${ab}`, at_bat_number: ab, pitch_number: pitch,
        situation: { inning, half, outs: pa, balls: 0, strikes: pitch - 1, bases: 0, home_score: inning > 4 ? 2 : 0, away_score: 0 },
        pitcher: { id: half === 'Top' ? 1 : 2, name: half === 'Top' ? '홈 투수' : '원정 투수', hand: 'R' },
        batter: { id: 100 + ab, name: `타자${ab}`, side: 'R' }, status: 'UNSUPPORTED_UNKNOWN_PITCHER',
        pre: { status: 'unsupported', reason: '합성 경기: 추천 없음', recommendation: null } });
    }
  }
  return { schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전 · 위치는 실제 투구 분포 근사',
    game: { game_pk: gamePk, date: '2025-09-01', game_type: 'R', away_team: 'Away Club', home_team: 'Home Club' },
    policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: 0.5, validated: false, note: '' },
    location: { basis: 'synthetic', evaluated: false, note: '', zone_bounds: { bottom: 1.6, top: 3.3899 } },
    we_note: '', coverage: { pitch_decisions: decisions.length, ready: 0, ready_share_of_pitches: 0 }, decisions };
}
const games = { 900001: timeline(900001, 9, true), 900002: timeline(900002, 10, false), 900003: timeline(900003, 10, false) };
const firstOf = (t, label) => t.decisions.find((d) => `${d.situation.inning}${d.situation.half === 'Top' ? '초' : '말'}` === label).index;

const browser = await chromium.launch();
try {
const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
await context.addInitScript(() => { try { localStorage.setItem('pitcheezy.guide.v1', '1'); } catch { /* ignore */ } });
await context.route('http://pz.test/**', async (route) => {
  const url = new URL(route.request().url());
  const hit = url.pathname.match(/^\/api\/watch\/(\d+)$/);
  if (hit && games[hit[1]]) return route.fulfill({ json: games[hit[1]] });
  if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
  const file = path.join(dist, url.pathname);
  const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
  return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
});
const page = await context.newPage();
const errors = [];
page.on('pageerror', (error) => errors.push(String(error)));
const row = () => page.locator('.wa-innings button').allInnerTexts();
const current = () => page.locator('.wa-innings button.current').innerText();
const open = async (game, i) => { await page.goto(`http://pz.test/watch?game=${game}#i=${i}`); await page.locator('.wa-board').waitFor({ timeout: 8000 }); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Game 1 ends after 9초 (home team ahead): the old row ended at 9초 and told the winner.
const g1 = games[900001], last1 = g1.decisions.length - 1;
await open(900001, 0);
if (!same(await row(), REGULAR)) fail(`game 1 row at the first pitch is not 1초–9말: ${(await row()).join(',')}`);
const progress = await page.locator('.wa-progress').innerText();
if (progress.includes('/') || progress.includes(String(g1.decisions.length))) fail(`progress shows the total: ${progress}`);
if (progress !== '경기 1번째 공') fail(`progress wording differs from the at-bat '구째' label: ${progress}`);
if ((await page.locator('select option').count()) !== 1) fail(`plate-appearance list shows ${await page.locator('select option').count()} options at the first pitch`);
await page.locator('.wa-innings button', { hasText: '9말' }).click({ timeout: 3000 }).catch(() => fail('no 9말 button'));
await page.waitForFunction((i) => window.location.hash === `#i=${i}`, last1, { timeout: 3000 }).catch(() => fail('9말 that was never played did not jump to the last pitch'));
if ((await current()) !== '9초') fail(`current half at the last pitch is ${await current()}`);
if ((await page.locator('select option').count()) !== g1.decisions.length / 2) fail('reached plate appearances not all listed at the end');
await page.locator('.wa-innings button', { hasText: '3초' }).click();   // going back keeps what was reached
if ((await page.locator('select option').count()) !== g1.decisions.length / 2) fail('going back shrank the reached list');

// Game 2 goes to 10말: extra innings appear only once reached, and stay after a reload.
const g2 = games[900002], last2 = g2.decisions.length - 1;
await open(900002, 0);
if (!same(await row(), REGULAR)) fail(`game 2 row at the first pitch shows extra innings: ${(await row()).join(',')}`);
if ((await page.locator('select option').count()) !== 1) fail('game 2 lists plate appearances reached in game 1');  // furthest is per game
await open(900002, firstOf(g2, '10초'));
if (!same(await row(), [...REGULAR, '10초'])) fail(`at 10초 the row is ${(await row()).join(',')}`);
await open(900002, last2);
if (!same(await row(), [...REGULAR, '10초', '10말'])) fail(`at the last pitch the row is ${(await row()).join(',')}`);
await open(900002, 0);
await page.reload(); await page.locator('.wa-board').waitFor();
if (!same(await row(), [...REGULAR, '10초', '10말'])) fail(`reached extra innings forgotten after reload: ${(await row()).join(',')}`);
if ((await current()) !== '1초') fail(`current half after reload is ${await current()}`);

// Game 3 opened straight at its last pitch by a shared link: going back and reloading keeps the extra innings reached.
const last3 = games[900003].decisions.length - 1;
await open(900003, last3);
await page.locator('.wa-innings button', { hasText: /^1초$/ }).click();
await page.waitForFunction(() => window.location.hash === '#i=0', null, { timeout: 3000 }).catch(() => fail('1초 did not jump to the first pitch'));
await page.reload(); await page.locator('.wa-board').waitFor();
if (!same(await row(), [...REGULAR, '10초', '10말'])) fail(`extra innings reached by a shared link forgotten after reload: ${(await row()).join(',')}`);

const small = await page.evaluate(() => [...document.querySelectorAll('.wa-innings button, .wa-nav select')]
  .map((el) => el.getBoundingClientRect()).filter((r) => r.width && (r.height < 44 || r.width < 44)).length);
if (small) fail(`${small} navigation tap targets under 44px`);
const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
if (scrollWidth > 390) fail(`horizontal scroll at phone width: ${scrollWidth}`);
if (errors.length) fail(`page errors: ${errors.join(' | ')}`);
} catch (error) { fail(String(error)); } finally { await browser.close(); }
if (!process.exitCode) console.log('watch spoiler smoke: ok');

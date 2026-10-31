// WS launch layout check (2026-10-17, workstream D): the four public-screen defects reported 10/12, measured on both bundles.
//  1. 320x568 in progress: the header's team names never run under the LIVE button, and the "다음 타자" row is never clipped.
//  2. A "정정됨" tag on a pitch card never covers that card's ball.
//  3. 320-wide 이닝별 기록: R·H·E·사사구 are visible without side-scrolling the table.
//  4. A game-level 확인 중 (display_state.state = unconfirmed): the status line never says 실시간으로 받는 중 at the same time.
// Synthetic rows built in memory from tests/fixtures/service-849843.json; served on an own loopback port (port 0), closed at the end.
// No backend, no network.   npm run build && npm run build:public && node tests/ws-launch-layout-check.mjs [screenshotDir] [tag]
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const shots = process.argv[2] ? path.resolve(process.argv[2]) : null; if (shots) mkdirSync(shots, { recursive: true });
const tag = process.argv[3] ?? 'run';
const DISTS = { api: path.join(here, '..', 'dist'), public: path.join(here, '..', 'dist-public') };
let passed = 0, failures = 0;
const ok = (cond, msg) => { if (cond) { passed++; console.log('ok  ', msg); } else { failures++; console.error('FAIL', msg); } };
const clone = (v) => JSON.parse(JSON.stringify(v));

// ---------- synthetic game: the longest Korean short names, two-digit scores, a WS series line ----------
const PK = 849843;
function body(state) {
  const b = clone(fx.live);
  b.source = { kind: 'live', session_id: null };
  Object.assign(b.game, { away_team: 'San Francisco Giants', home_team: 'St. Louis Cardinals', away_abbr: 'SF', home_abbr: 'STL',
    away_score: 10, home_score: 12, series_description: 'World Series', series_game_number: 7, venue: 'Busch Stadium' });
  const cur = b.pitches.find((p) => p.pa_key === b.current.pa_key);
  // the current plate appearance: two received pitches, both corrected, in the two upper corners of the window (where a tag sits)
  const second = { ...clone(cur), index: cur.index + 1, key: `${PK}:11:2`, pitch_number: 2, actual: { ...clone(cur.actual), x: -0.9, z: 3.75 } };
  cur.actual.x = 0.9; cur.actual.z = 3.75;
  for (const p of b.pitches) p.receipt = { status: 'received', corrected: false };
  cur.receipt = second.receipt = { status: 'received', corrected: true };
  b.pitches.push(second);
  b.cutoff = { index: second.index };
  const pa = b.pas.find((p) => p.pa_key === b.current.pa_key); pa.last_index = second.index; pa.pitch_count = 2;
  b.current.situation = { ...b.current.situation, balls: 1, strikes: 1, home_score: 12, away_score: 10 };
  b.current.context = { batter: { at_bats: 4, hits: 3, on_base: 4 }, pitcher: { pitches: 104, strikeouts: 11, earned_runs: 10 },
    next_batters: [{ id: 1, name: 'Christopher Morel-Hernandez', order: 4 }, { id: 2, name: 'Alexander Canario-Rodriguez', order: 5 }] };
  b.next = { ...b.next, pitch_key: `${PK}:11:3` };
  b.linescore = { innings: Array.from({ length: 9 }, (_, i) => i < 2 ? { away: i ? 10 : 0, home: i ? 2 : 10 } : { away: null, home: null }),
    teams: { away: { runs: 10, hits: 14, errors: 2, bb_hbp: 5 }, home: { runs: 12, hits: 15, errors: 1, bb_hbp: 7 } } };
  b.win_expectancy = { status: 'available', home_probability: 0.61, perspective: 'home' };
  const shown = state === 'live';
  const ds = { live: ['recommended', '다음 공 추천', null], confirming: ['unconfirmed', '확인 중', '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.'],
    final: ['final', '경기 종료', '경기가 끝났어요.'] }[state];
  b.display_state = { schema: 'pitcheezy-display-state-v1', state: ds[0], reason: 'synthetic', label: ds[1], message: ds[2], pitch_key: state === 'final' ? null : `${PK}:11:3`,
    recommendation_shown: shown, correction: null, feed: { ok: true, delayed: false, lag_s: null } };
  if (state === 'final') { Object.assign(b.game, { status: 'final', status_text: 'Final', live: false, inning: null, half: null }); b.current = null; b.next = null; }
  return b;
}

// ---------- own loopback server: dist/ answers /api/service/**, dist-public/ answers v1/** (stamped now) ----------
let dist = DISTS.api, game = null, seq = 200;
const stamp = (o) => { const at = new Date().toISOString(); return { ...o, publish: { contract: 'v1', sequence: seq, published_at: at } }; };
const TYPES = { '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.html': 'text/html' };
const server = createServer((req, res) => {
  const u = new URL(req.url, 'http://localhost');
  const json = (o) => { res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(o)); };
  const card = { ...game.game };
  if (u.pathname === '/api/service/dates') return json({ dates: [card.date_kst] });
  if (u.pathname === '/api/service/games') return json({ date: card.date_kst, games: [card] });
  if (u.pathname === `/api/service/games/${PK}`) return json(game);
  if (u.pathname.startsWith('/v1/')) {
    seq++;
    const f = u.pathname.slice(4);
    if (f === 'health.json') return json(stamp({ schema: 'pitcheezy-publish-health-v1', sequence: seq, published_at: new Date().toISOString(), publish_interval_s: 2, unavailable_games: [], unavailable_dates: [], feed_failing_games: [] }));
    if (f === 'manifest.json') return json(stamp({ schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', published_at: new Date().toISOString(), publish_interval_s: 2, fixed_delay_s: 0, live_dates: [card.date_kst] }));
    if (f === 'dates.json') return json(stamp({ schema: 'pitcheezy-service-dates-v2', source: { kind: 'live', session_id: null }, dates: [card.date_kst] }));
    if (f === `games/${card.date_kst}.json`) return json(stamp({ schema: 'pitcheezy-service-games-v2', source: { kind: 'live', session_id: null }, date: card.date_kst, games: [card], live_ok: true }));
    if (f === `game/${PK}.json`) return json(stamp({ schema: 'pitcheezy-service-game-v2', ...game }));
    res.writeHead(404); return res.end();
  }
  let file = path.join(dist, u.pathname);
  if (!existsSync(file) || !statSync(file).isFile()) file = path.join(dist, 'index.html');
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] ?? 'text/html' }); res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch();

const rect = (loc) => loc.evaluate((e) => { const r = e.getBoundingClientRect(); return { l: r.left, r: r.right, t: r.top, b: r.bottom, w: r.width, h: r.height }; });
const overlap = (a, b) => Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l)) * Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
const SIZES = [[320, 568], [375, 667], [390, 844], [1440, 900]];

try {
  for (const mode of ['public', 'api']) {
    dist = DISTS[mode];
    if (!existsSync(path.join(dist, 'index.html'))) { ok(false, `${mode}: ${dist} is missing (build first)`); continue; }
    for (const [width, height] of SIZES) for (const state of ['live', 'confirming', 'final']) {
      game = body(state);
      const ctx = await browser.newContext({ viewport: { width, height } }), page = await ctx.newPage(), errors = [];
      page.on('pageerror', (e) => errors.push(String(e)));
      await ctx.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
      await page.goto(origin + (mode === 'public' ? `/?game=${PK}` : `/watch/${PK}`));
      await page.locator('.sv-head-score').waitFor({ timeout: 10000 });
      await page.locator(`.sv-state[data-state="${game.display_state.state}"]`).waitFor({ timeout: 10000 }).catch(() => {});
      await page.waitForTimeout(300);
      const id = `${mode} ${width}x${height} ${state}`;
      ok(await page.evaluate(() => document.scrollingElement.scrollWidth <= innerWidth + 1), `${id}: no side scroll`);
      // 1. header: each team name and score stays clear of the LIVE / 종료 button
      const live = await rect(page.locator('.sv-head .sv-live'));
      const parts = page.locator('.sv-head-score > *');
      let hit = 0; for (let i = 0; i < await parts.count(); i++) hit += overlap(await rect(parts.nth(i)), live);
      const score = await rect(page.locator('.sv-head-score'));
      ok(hit === 0 && score.r <= live.l + 0.5, `${id}: team names clear of the status button (score right ${score.r.toFixed(1)} ≤ button left ${live.l.toFixed(1)})`);
      // 1. next batters: when the row is shown, no name is cut (no ellipsis-hidden or horizontally clipped text)
      const up = page.locator('.sv-upcoming');
      if (await up.isVisible()) {
        const cut = await up.evaluate((e) => [...e.querySelectorAll('span')].some((s) => s.scrollWidth > s.clientWidth + 1 || s.scrollHeight > s.clientHeight + 1));
        ok(!cut, `${id}: 다음 타자 names not clipped`);
      }
      // 2. a corrected pitch card: the tag never covers the ball
      if (state !== 'final') {
        const cards = page.locator('.sv-strip .sv-card').filter({ has: page.locator('.sv-card-tag') });
        const n = await cards.count();
        ok(n === 2, `${id}: two corrected cards carry the tag (${n})`);
        for (let i = 0; i < n; i++) {
          const card = cards.nth(i), t = await rect(card.locator('.sv-card-tag'));
          const ball = card.locator('.sv-zone.mini .sv-actual circle').first();
          const dot = await ball.isVisible() ? await rect(ball) : null;  // the 10/13 strip draws no mini zone: nothing to cover
          ok(dot === null || overlap(t, dot) === 0, `${id}: 정정됨 tag clear of the ball on card ${i + 1} (${dot ? `overlap ${overlap(t, dot).toFixed(1)} px²` : 'no mini zone drawn'})`);
          const cap = await card.locator('.sv-card-cap > span').evaluateAll((ss) => ss.map((s) => { const r = document.createRange(); r.selectNodeContents(s); const b = r.getBoundingClientRect(); return { l: b.left, r: b.right, t: b.top, b: b.bottom }; }));
          const capHit = cap.reduce((a, c) => a + overlap(t, c), 0);
          ok(capHit === 0, `${id}: 정정됨 tag clear of card ${i + 1}'s text (overlap ${capHit.toFixed(1)} px²)`);
        }
      }
      // 4. one story: game-level 확인 중 never sits next to 실시간으로 받는 중
      const line = (await page.locator('.sv-notice.line .sv-line-text').innerText().catch(() => '')).trim();
      const chip = (await page.locator('.sv-chip').innerText().catch(() => '')).trim();
      if (state === 'confirming') ok(/확인 중/.test(chip) && !/실시간으로 받는 중/.test(line) && /확인 중/.test(line), `${id}: chip "${chip}" and status line "${line}" agree`);
      if (state === 'live') ok(/실시간으로 받는 중/.test(line), `${id}: live status line "${line}"`);
      if (shots) await page.screenshot({ path: path.join(shots, `${tag}-${mode}-${width}x${height}-${state}.png`), fullPage: true });
      // 3. linescore sheet: R·H·E·사사구 visible without side-scrolling
      if (state === 'live' && mode === 'public') {
        const link = width >= 760 ? page.locator('.sv-panel-link').filter({ hasText: '이닝별 점수표' }) : page.locator('.sv-inning');
        await link.first().click();
        if (width < 760) await page.locator('.sv-sheet').getByRole('button', { name: '박스스코어', exact: true }).click();
        const box = page.locator('.sv-sheet .sv-box');
        if (await box.count()) {
          const scroller = await rect(page.locator('.sv-sheet .sv-box-scroll'));
          const heads = await page.locator('.sv-sheet .sv-box thead th').evaluateAll((ths) => ths.slice(-4).map((t) => { const r = t.getBoundingClientRect(); return { text: t.textContent, r: r.right }; }));
          const hidden = heads.filter((h) => h.r > scroller.r + 0.5).map((h) => h.text);
          ok(hidden.length === 0, `${id}: 이닝별 기록 R·H·E·사사구 visible without side scroll${hidden.length ? ` (hidden: ${hidden.join(',')})` : ''}`);
          if (shots) await page.screenshot({ path: path.join(shots, `${tag}-${mode}-${width}x${height}-boxscore.png`) });
        } else ok(false, `${id}: linescore sheet did not open`);
      }
      ok(errors.length === 0, `${id}: no page errors ${errors.join(' | ')}`);
      await ctx.close();
    }
  }
} finally { await browser.close(); await new Promise((r) => server.close(r)); }
console.log(`${passed} passed, ${failures} failed`);
if (failures) process.exit(1);

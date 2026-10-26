// WS launch layout check (2026-10-17, workstream D), moved to the 10/26 screen (docs/handoffs/UI-design-1026.md), both bundles:
//  1. In progress at 320–1440: score and the single status marker share header row 1 without overlap; player names are not cut.
//  2. A "정정됨" tag on a corrected pitch (at-bat list) never covers that pitch's mark or words.
//  3. 이닝별 기록 (opened from the header linescore): R·H·E·사사구 are visible without side-scrolling the table.
//  4. A game-level 확인 중: the marker says 확인 중 and no candidate appears in cells, squares or labels.
//  Also: no spoiler / limitation prose on the game screen; the guarded win probability (undecided sheet) still opens, from 기록.
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
let dist = DISTS.api, game = null, seq = 200, fixedDelay;
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
    if (f === 'manifest.json') return json(stamp({ schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', published_at: new Date().toISOString(), publish_interval_s: 2, ...(fixedDelay === undefined ? {} : { fixed_delay_s: fixedDelay }), live_dates: [card.date_kst] }));
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
      fixedDelay = state === 'live' ? { 320: -1, 375: 15, 390: 0 }[width] : 0;
      game = body(state);
      const ctx = await browser.newContext({ viewport: { width, height } }), page = await ctx.newPage(), errors = [];
      page.on('pageerror', (e) => errors.push(String(e)));
      await ctx.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
      await page.goto(origin + (mode === 'public' ? `/?game=${PK}` : `/watch/${PK}`));
      await page.locator('.sv-hd-score').waitFor({ timeout: 10000 });
      await page.locator(`.sv-state[data-state="${game.display_state.state}"]`).waitFor({ timeout: 10000 }).catch(() => {});
      await page.waitForTimeout(300);
      const id = `${mode} ${width}x${height} ${state}`;
      ok(await page.evaluate(() => document.scrollingElement.scrollWidth <= innerWidth + 1), `${id}: no side scroll`);
      if (width === 390 && state !== 'final') {
        const ab = await rect(page.locator('.sv-atbat-head'));
        const nav = await rect(page.locator('.sv-bottom-nav'));
        ok(ab.b <= nav.t, `${id}: at-bat line clears the fixed bottom bar (${ab.b.toFixed(1)} ≤ ${nav.t.toFixed(1)})`);
      }
      ok(await page.locator('.sv-limit-line').count() === 0, `${id}: main view omits the old limitation banner`);
      // UI 10/26 (decision line): no spoiler / limitation prose on the game screen in any state; the game list keeps its line (below)
      ok(await page.locator('.sv-game .sv-spoiler-line').count() === 0 && !/스포일러|TV 중계보다/.test(await page.locator('.sv-page').innerText()), `${id}: no spoiler line on the game screen`);
      // UI 10/26 ①⑥: row 1 is back · score · ONE status marker (none on a finished game); score and marker never overlap.
      const scoreBox = await rect(page.locator('.sv-hd-score > span'));
      const chip = page.locator('.sv-hd .sv-hd-status');
      if (state === 'final') ok(await chip.count() === 0, `${id}: a finished game has no status marker`);
      else {
        const c = await rect(chip);
        ok(overlap(scoreBox, c) === 0 && c.r <= width + 0.5 && Math.abs((c.t + c.b) / 2 - (scoreBox.t + scoreBox.b) / 2) < 12, `${id}: score and status marker share row 1 without overlap`, { scoreBox, c });
        const status = (await chip.innerText()).trim();
        ok(status === (state === 'live' ? '라이브' : '확인 중'), `${id}: status marker reflects the state (${status})`);
      }
      // row 2: the linescore (two team rows, R) sits in the header; the half in progress is highlighted, a finished game highlights nothing
      const ls = page.locator('.sv-hd-2 .sv-ls');
      const cur = await ls.locator('.cur').count(), teams = await ls.locator('.t').allInnerTexts();
      ok(await ls.isVisible() && teams.join('|') === 'SF|STL' && (state === 'final' ? cur === 0 : cur === 3), `${id}: header linescore with both teams (${teams.join('|')}, ${cur} highlighted cells)`);
      // UI 10/26 ⑤: the win probability has no place on the main screen; the undecided sheet itself (guards unchanged) opens from 기록
      ok(!/승리확률|61%|39%/.test(await page.locator('.sv-page').evaluate((e) => { const c = e.cloneNode(true); c.querySelectorAll('.sv-backdrop').forEach((x) => x.remove()); return c.innerText; })), `${id}: no win probability on the main screen`);
      if (state !== 'confirming') {
        await page.locator('.sv-bottom-nav button', { hasText: '기록' }).click();
        await page.locator('.sv-sheet').getByRole('button', { name: '승리확률' }).click();
        const bars = page.locator('.sv-sheet .sv-we-bar');
        if (state === 'live') ok(await bars.count() === 2 && (await bars.allInnerTexts()).some(t => /61%/.test(t)) && (await bars.allInnerTexts()).some(t => /39%/.test(t)), `${id}: guarded current probability still opens two-team 61/39 bars (from 기록)`);
        else ok(await bars.count() === 0 && /정보 없음/.test(await page.locator('.sv-sheet').innerText()), `${id}: finished game's snapshot is not reused as a current probability`);
        await page.keyboard.press('Escape'); await page.getByRole('dialog').waitFor({ state: 'detached' });
      }
      // the at-bat list opens in place; it carries the corrected tags and the past-pitch entry
      if (state !== 'final') {
        await page.locator('.sv-atbat-toggle').click();
        ok(await page.locator('.sv-atbat-toggle').getAttribute('aria-expanded') === 'true' && await page.locator('#sv-atbat-list').isVisible(), `${id}: 펼치기 opens the at-bat list in place`);
      }
      if (state === 'live' && width === 390) {
        // a past pitch's detail never changes the live marker (transport state only)
        await page.locator('.sv-pitch-row').first().click();
        await page.getByRole('dialog').waitFor();
        ok((await chip.innerText()).trim() === '라이브', `${id}: past detail leaves the live marker as 라이브`);
        await page.keyboard.press('Escape');
        await page.getByRole('dialog').waitFor({ state: 'detached' });
      }
      // 1. row 3: no player name is cut (the next-batter row of the old layout is gone)
      if (await page.locator('.sv-who').count()) {
        const cut = await page.locator('.sv-who b').evaluateAll((bs) => bs.some((b) => b.scrollWidth > b.clientWidth + 1));
        ok(!cut, `${id}: batter / pitcher names not clipped`);
      }
      // 2. a corrected pitch: its 정정됨 tag never covers the result mark or the words of that row
      if (state !== 'final') {
        const rows = page.locator('.sv-pitch-row').filter({ has: page.locator('.sv-card-tag') });
        const n = await rows.count();
        ok(n === 2, `${id}: two corrected pitches carry the tag (${n})`);
        for (let i = 0; i < n; i++) {
          const row = rows.nth(i), t = await rect(row.locator('.sv-card-tag'));
          const dot = await rect(row.locator('.sv-g1'));
          ok(overlap(t, dot) === 0, `${id}: 정정됨 tag clear of the mark on row ${i + 1}`);
          const words = await row.locator('.sv-pr-what, .sv-pr-res').evaluateAll((ss) => ss.map((s) => { const r = document.createRange(); r.selectNodeContents(s); const b = r.getBoundingClientRect(); return { l: b.left, r: b.right, t: b.top, b: b.bottom }; }));
          const hit = words.reduce((a, c) => a + overlap(t, c), 0);
          ok(hit === 0, `${id}: 정정됨 tag clear of row ${i + 1}'s text (overlap ${hit.toFixed(1)} px²)`);
        }
      }
      ok(await page.locator('.sv-notice.line,.sv-chip').count() === 0, `${id}: no duplicate receive-time/status row`);
      if (state === 'confirming') {
        const pending = page.locator('.sv-state[data-state="unconfirmed"]');
        const cells = await pending.locator('.sv-rec-cell').allInnerTexts();
        ok(await pending.isVisible() && cells.length === 3 && cells.every((c) => c.trim() === '—') && await page.locator('.sv-zrow .sv-rec').count() === 0
          && /확인/.test(await pending.locator('[role="status"]').textContent()) && !/순위/.test(await pending.locator('.sv-reclist').getAttribute('aria-label')),
          `${id}: unconfirmed recommendation stays explicit ('—' cells, no square, no candidate in labels; sentence in the live region)`);
      }
      if (shots) await page.screenshot({ path: path.join(shots, `${tag}-${mode}-${width}x${height}-${state}.png`), fullPage: true });
      // 3. linescore sheet (opened from the header linescore): R·H·E·사사구 visible without side-scrolling
      if (state === 'live' && mode === 'public') {
        await page.locator('.sv-hd-2 .sv-ls').click();
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
  // Missing or out-of-range source values must never become a fabricated probability bar (sheet reached from 기록).
  dist = DISTS.public;
  for (const [label, value] of [['missing', null], ['out-of-range', 1.1]]) {
    game = body('live'); fixedDelay = 0;
    game.win_expectancy = value === null ? null : { status: 'available', home_probability: value, perspective: 'home' };
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } }), page = await ctx.newPage();
    await page.goto(`${origin}/?game=${PK}`);
    await page.locator('.sv-bottom-nav button', { hasText: '기록' }).click();
    await page.locator('.sv-sheet').getByRole('button', { name: '승리확률' }).click();
    ok(await page.locator('.sv-sheet .sv-we-bar').count() === 0 && /정보 없음/.test(await page.locator('.sv-sheet').innerText()), `public win ${label}: guarded empty state`);
    await ctx.close();
  }
  // A long series title must leave a full-width row for team names on a phone list card.
  dist = DISTS.public; game = body('live');
  for (const width of [320, 390]) {
    fixedDelay = width === 320 ? 15 : 0;
    const ctx = await browser.newContext({ viewport: { width, height: 844 } }), page = await ctx.newPage();
    await page.goto(`${origin}/?date=${game.game.date_kst}`);
    const card = page.locator('.sv-list .sv-card.live');
    await card.waitFor();
    const teams = await card.locator('.sv-card-team').evaluateAll((els) => els.map((e) => ({ height: e.getBoundingClientRect().height, text: e.textContent.trim() })));
    const row = await card.locator('.sv-card-teams').evaluate((e) => getComputedStyle(e).gridRowStart);
    ok(teams.length === 2 && teams.every(t => t.height <= 32) && row === '2', `public list ${width}: team names use their own full-width row (${teams.map(t => `${t.text}:${t.height.toFixed(1)}`).join(', ')})`);
    const words = await page.locator('.sv-list .sv-spoiler-line').innerText();
    ok(width === 320 ? /공통 15초 지연 설정/.test(words) : /공통 지연 설정 없이/.test(words), `public list ${width}: spoiler copy follows manifest`);
    if (shots) await page.screenshot({ path: path.join(shots, `${tag}-public-list-${width}.png`) });
    await ctx.close();
  }
} finally { await browser.close(); await new Promise((r) => server.close(r)); }
console.log(`${passed} passed, ${failures} failed`);
if (failures) process.exit(1);

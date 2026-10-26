// UI 10/26 capture + check (docs/handoffs/UI-design-1026.md, docs/design/ui-1026/*.png). Runs the replay preview of the public screen
// (real finished game 849843, first 65 pitches; live id 990003; preview build only, never deployed) on an own loopback port and, for
// the nine states 0·5·11·13·21·30·32·48·77 at 390·360·320·402 phone widths plus tablet and PC, checks what the design fixes and
// saves a screenshot. At 390x844 it also opens the expanded at-bat, the status bubble and every approved sheet (half and full).
//   npx vite build --config vite.preview.config.ts --mode public
//   node tests/ui-1026-capture.mjs <outDir> [distDir]        (distDir defaults to dist-preview-classic)
// Screenshots of the replay are derived from real game data: keep them outside the repository unless told otherwise.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] ? path.resolve(process.argv[2]) : null;
if (!out) { console.error('usage: node tests/ui-1026-capture.mjs <outDir> [distDir]'); process.exit(2); }
const dist = path.resolve(process.argv[3] ?? path.join(here, '..', 'dist-preview-classic'));
if (!existsSync(path.join(dist, 'index.html'))) { console.error(`missing ${dist}/index.html (build the preview first)`); process.exit(2); }
mkdirSync(out, { recursive: true });
let passed = 0, failed = 0;
const log = [];
const ok = (cond, msg, info) => { if (cond) passed++; else failed++; const line = `${cond ? 'ok  ' : 'FAIL'} ${msg}${cond || info === undefined ? '' : ` ${JSON.stringify(info)}`}`; log.push(line); console.log(line); };

const TYPES = { '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.html': 'text/html', '.json': 'application/json' };
const server = createServer((req, res) => {
  const u = new URL(req.url, 'http://localhost');
  let file = path.join(dist, decodeURIComponent(u.pathname));
  if (!file.startsWith(dist) || !existsSync(file) || !statSync(file).isFile()) file = path.join(dist, 'index.html');
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] ?? 'application/octet-stream', 'Cache-Control': 'no-store' }); res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch();

// step → what the design fixes for it (UI-design-1026 ⑥, figures 01–11)
const STATES = {
  0: { name: 'pregame', chip: '경기 전', tone: 'pregame', row3: 'none', recs: 'none', inning: /\d\d:\d\d 시작/ },
  5: { name: 'checking', chip: '확인 중', tone: 'checking', row3: 'matchup', recs: 'dash', inning: /1회초 ▾/ },
  11: { name: 'late', chip: '자료 늦음', tone: 'late', row3: 'matchup', recs: 'named', inning: /1회초 ▾/ },
  13: { name: 'break', chip: '라이브', tone: 'live', row3: 'line', recs: 'dash', inning: /1회초 종료/ },
  21: { name: 'corrected', chip: '추천 바뀜', tone: 'fixed', row3: 'matchup', recs: 'named', inning: /1회말 ▾/ },
  30: { name: 'atbat', chip: '라이브', tone: 'live', row3: 'matchup', recs: 'named', inning: /2회초 ▾/ },
  32: { name: 'lost', chip: '연결 끊김', tone: 'lost', row3: 'matchup', recs: 'held', inning: /2회초 ▾/ },
  48: { name: 'unsupported', chip: '추천 없음', tone: 'none', row3: 'matchup', recs: 'dash', inning: /2회말 ▾/ },
  77: { name: 'final', chip: null, tone: null, row3: 'none', recs: 'any', inning: /경기 종료/ },
};
const PHONES = [[390, 844], [360, 780], [320, 568], [402, 874]];
const WIDE = [[820, 1180], [1440, 900]];

async function open(width, height, step, extra = '') {
  const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, isMobile: width < 700, hasTouch: width < 700, locale: 'ko-KR' });
  await ctx.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
  const page = await ctx.newPage(); const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.addInitScript(() => sessionStorage.clear());
  await page.goto(`${origin}/?game=990003&step=${step}&play=0${extra}`);
  await page.locator('.sv-hd-score').waitFor({ timeout: 15000 });
  // the replay panel is preview chrome, not the screen: hide it from captures
  await page.addStyleTag({ content: '#pz-replay-bar{display:none!important}' });
  await page.waitForTimeout(700);
  return { ctx, page, errors };
}
const rect = (loc) => loc.evaluate((e) => { const r = e.getBoundingClientRect(); return { l: r.left, r: r.right, t: r.top, b: r.bottom, w: r.width, h: r.height }; });
const overlap = (a, b) => Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l)) * Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
const shots = [];
async function shot(page, name, full = false) { const file = path.join(out, `${name}.png`); await page.screenshot({ path: file, fullPage: full }); shots.push(file); }

async function checkMain(page, step, width, height, id) {
  const want = STATES[step];
  ok(await page.evaluate(() => document.scrollingElement.scrollWidth <= innerWidth + 1), `${id}: no sideways scroll`);
  const chip = page.locator('.sv-hd-status');
  if (want.chip === null) ok(await chip.count() === 0, `${id}: no status marker on a finished game`);
  else {
    const label = (await chip.innerText()).trim(), tone = await chip.getAttribute('data-tone');
    ok(label === want.chip && tone === want.tone, `${id}: status marker "${want.chip}" (${want.tone})`, { label, tone });
    // row 1: back, score and marker never overlap; the marker stays inside the screen
    const back = await rect(page.locator('.sv-hd-1 .sv-back')), score = await rect(page.locator('.sv-hd-score > span')), c = await rect(chip);
    ok(overlap(score, c) === 0 && overlap(back, score) === 0 && c.r <= width + 0.5 && back.l >= -0.5, `${id}: row 1 back / score / marker apart`, { back, score, c });
  }
  const inning = (await page.locator('.sv-hd-2 .sv-inning').innerText()).trim();
  ok(want.inning.test(inning), `${id}: row 2 says "${inning}"`);
  if (want.row3 === 'none') ok(await page.locator('.sv-mu').count() === 0, `${id}: no row 3`);
  if (want.row3 === 'line') ok(/^공수 교대 · 다음 \d회[초말] 첫 타자/.test((await page.locator('.sv-mu.line').innerText()).trim()), `${id}: row 3 is the inning-break line`);
  if (want.row3 === 'matchup') {
    const bat = await rect(page.locator('.sv-who.bat')), pit = await rect(page.locator('.sv-who.pit')), cnt = await rect(page.locator('.sv-mu-count'));
    ok(bat.r <= cnt.l + 0.5 && cnt.r <= pit.l + 0.5, `${id}: batter left · count · pitcher right`);
    const roles = [await page.locator('.sv-who.bat small').innerText(), await page.locator('.sv-who.pit small').innerText()];
    ok(roles[0].startsWith('타자') && roles[1].startsWith('투수'), `${id}: roles ${roles.join(' | ')}`);
    const names = await page.locator('.sv-who > b').evaluateAll((bs) => bs.map((b) => ({ text: b.innerText.trim(), cut: b.scrollWidth > b.clientWidth + 1 })));
    ok(names.every((n) => !n.cut), `${id}: player names not cut`, names);
    if (width < 360) ok(names.every((n) => !n.text.includes(' ') || /Jr\.|Sr\./.test(n.text)), `${id}: family names only below 360px`, names);
  }
  // linescore: inside the header row, current column highlighted while a game is in progress
  if (await page.locator('.sv-ls').count()) { const ls = await rect(page.locator('.sv-ls')); ok(ls.r <= width + 0.5 && ls.l >= 0, `${id}: linescore inside the screen`, ls); }
  else ok(step === 0, `${id}: only the pregame header has no linescore`);
  ok(await page.locator('.sv-spoiler-line, .sv-game-context, .sv-previous, .sv-limit-line').count() === 0, `${id}: no spoiler/limitation prose, no removed panels`);
  // recommendation row
  if (want.recs !== 'none' && want.recs !== 'any') {
    const cells = await page.locator('.sv-rec-sec .sv-rec-cell').allInnerTexts();
    if (want.recs === 'named') ok(cells.length === 3 && cells.every((c) => /^\d\s*\S/.test(c.trim())), `${id}: three named candidates`, cells);
    if (want.recs === 'dash') ok(cells.length === 3 && cells.every((c) => c.trim() === '—'), `${id}: three '—' cells`, cells);
    if (want.recs === 'held') ok(/마지막으로 받은 추천/.test(await page.locator('.sv-rec-sec .sv-rec-head').innerText()) && Number(await page.locator('.sv-rec-sec .sv-reclist').evaluate((e) => getComputedStyle(e).opacity)) < 1, `${id}: last received candidates, dimmed`);
  }
  // bottom bar: fixed, full width, four entries; at 390x844 the at-bat line sits above it
  const nav = await rect(page.locator('.sv-bottom-nav'));
  ok(Math.abs(nav.b - height) < 1 && nav.l <= 0.5 && Math.abs(nav.r - width) < 1 && await page.locator('.sv-bottom-nav button').count() >= 4, `${id}: bottom bar fixed across the bottom`, nav);
  if (width === 390 && height === 844 && want.row3 === 'matchup') {   // the at-bat line exists while someone is at bat
    const ab = await rect(page.locator('.sv-atbat-head'));
    ok(ab.b <= nav.t, `${id}: at-bat line above the bottom bar (${ab.b.toFixed(1)} ≤ ${nav.t.toFixed(1)})`);
  }
}

const sheetBox = (page) => rect(page.locator('.sv-sheet'));
// one approved sheet: half (upper content visible, rest not), keyboard expand/shrink on the handle, a real pointer drag up (the click the
// browser sends after it must not undo it), a cancelled gesture that changes nothing, Tab kept inside, Escape closing with focus back
async function sheet(page, name, openIt, opener, halfCheck, fullCheck) {
  const H = page.viewportSize().height;
  await openIt(); const dialog = page.getByRole('dialog'); await dialog.waitFor(); await page.waitForTimeout(350);
  const half = await sheetBox(page);
  ok(await dialog.getAttribute('aria-modal') === 'true' && await page.locator('.sv-sheet.frame-c.half').count() === 1 && half.h >= H * 0.45 && half.h <= H * 0.62 && Math.abs(half.b - H) < 1.5,
    `${name}: opens half (${half.h.toFixed(0)}px of ${H}), modal`, half);
  const hc = await halfCheck(dialog, half); ok(hc.ok, `${name}: half shows ${hc.what}`, hc.info);
  await shot(page, `sheet-${name}-half-390x844`);
  const grip = page.locator('.sv-sheet-grip');
  // keyboard: the handle is a button (Enter opens to full, Enter again back to half)
  await grip.focus(); await page.keyboard.press('Enter'); await page.waitForTimeout(350);
  const kFull = await sheetBox(page);
  ok(await grip.getAttribute('aria-expanded') === 'true' && kFull.h > half.h + 100, `${name}: keyboard Enter on the handle opens it to full (${kFull.h.toFixed(0)}px)`);
  await page.keyboard.press('Enter'); await page.waitForTimeout(350);
  ok(await grip.getAttribute('aria-expanded') === 'false' && Math.abs((await sheetBox(page)).h - half.h) < 2, `${name}: Enter again goes back to half`);
  // a cancelled gesture (pointercancel) resets: no move, no open, no close
  await grip.dispatchEvent('pointerdown', { pointerId: 7, clientY: half.t + 10, isPrimary: true, bubbles: true });
  await grip.dispatchEvent('pointermove', { pointerId: 7, clientY: half.t - 150, isPrimary: true, bubbles: true });
  await grip.dispatchEvent('pointercancel', { pointerId: 7, isPrimary: true, bubbles: true }); await page.waitForTimeout(350);
  const afterCancel = await sheetBox(page);
  ok(await dialog.count() === 1 && await grip.getAttribute('aria-expanded') === 'false' && Math.abs(afterCancel.h - half.h) < 2 && Math.abs(afterCancel.b - H) < 1.5 && await page.locator('.sv-sheet.dragging').count() === 0,
    `${name}: a cancelled gesture changes nothing`, afterCancel);
  // real pointer drag up on the handle → full; the click that follows the drag does not toggle it back
  const g = await rect(grip);
  await page.mouse.move(g.l + g.w / 2, g.t + g.h / 2); await page.mouse.down(); await page.mouse.move(g.l + g.w / 2, g.t - 60, { steps: 4 }); await page.mouse.move(g.l + g.w / 2, g.t - 140, { steps: 4 }); await page.mouse.up();
  await page.waitForTimeout(400);
  const full = await sheetBox(page);
  ok(await grip.getAttribute('aria-expanded') === 'true' && full.h > half.h + 100 && full.t <= 60, `${name}: dragging the handle up opens it to full and stays there (top ${full.t.toFixed(0)})`, full);
  const fc = await fullCheck(dialog, full); ok(fc.ok, `${name}: full shows ${fc.what}`, fc.info);
  await shot(page, `sheet-${name}-full-390x844`);
  // focus stays inside; Escape closes and focus returns to the opener
  const inside = []; for (let i = 0; i < 12; i++) { await page.keyboard.press('Tab'); inside.push(await page.evaluate(() => !!document.activeElement?.closest('.sv-sheet'))); }
  ok(inside.every(Boolean), `${name}: Tab stays inside the sheet`, inside);
  await page.keyboard.press('Escape'); await dialog.waitFor({ state: 'detached' });
  ok(await opener().evaluate((e) => e === document.activeElement), `${name}: Escape closes, focus back on the opener`);
}

try {
  // ---- interactions at 390x844 on the design-01 state (step 30) ----
  {
    const { ctx, page, errors } = await open(390, 844, 30);
    await page.locator('.sv-atbat-toggle').click(); await page.waitForTimeout(250);
    const rows = await page.locator('.sv-atbat-list > li').evaluateAll((ls) => ls.map((l) => ({ cls: l.className, text: l.innerText.replace(/\s+/g, ' ').trim(), kind: l.querySelector('.sv-g1')?.className ?? null })));
    ok(await page.locator('.sv-atbat-toggle').getAttribute('aria-expanded') === 'true' && /^4 4구 · 다음 공$/.test(rows[0]?.text ?? '') && rows.slice(1).map((r) => parseInt(r.text, 10)).join() === '3,2,1'
      && /ball/.test(rows[1].kind) && /strike/.test(rows[2].kind) && /ball/.test(rows[3].kind) && /접기/.test(await page.locator('.sv-atbat-toggle').innerText()),
      'atbat 390x844: opens in place — 4구 · 다음 공, then 3 (ball) · 2 (foul) · 1 (ball), newest first (design 16)', rows);
    await shot(page, 'atbat-expanded-390x844', true);
    await page.locator('.sv-atbat-toggle').click();
    // ⑧ sheets
    const vis = (loc) => loc.evaluate((e) => { const r = e.getBoundingClientRect(); return r.height > 0 && r.bottom <= innerHeight + 0.5; });
    await sheet(page, 'rec', () => page.locator('.sv-rec-sec .sv-state-more').click(), () => page.locator('.sv-rec-sec .sv-state-more'),
      async (d) => ({ ok: await d.locator('.sv-rec-table tbody tr').count() === 3 && await vis(d.locator('.sv-rec-table tbody tr').nth(2)) && /^다음 공 추천 · 4구$/.test((await d.locator('h2').innerText()).trim()), what: 'title 다음 공 추천 · 4구 and the whole table' }),
      async (d) => ({ ok: await vis(d.getByRole('button', { name: '기준과 한계 보기' })) && /직전 투구/.test(await d.innerText()), what: 'the note, 직전 투구 and the links' }));
    await page.locator('.sv-atbat-toggle').click();
    await sheet(page, 'pitch', () => page.locator('.sv-pitch-row').filter({ has: page.locator('.sv-g1', { hasText: /^1$/ }) }).click(), () => page.locator('.sv-pitch-row').filter({ has: page.locator('.sv-g1', { hasText: /^1$/ }) }),
      async (d) => { const a = await d.locator('.sv-detail-zone .sv-actual').evaluate((g) => ({ out: g.classList.contains('out'), x: g.dataset.x, z: g.dataset.z, above: g.querySelector('circle').getBoundingClientRect().bottom <= g.closest('svg').querySelector('.sv-zone-box').getBoundingClientRect().top }));
        return { ok: /^2회 초 · 1구 · Seiya Suzuki$/.test((await d.locator('h2').innerText()).trim()) && await vis(d.locator('.sv-pitch-line')) && await vis(d.locator('.sv-detail-zone')) && !a.out && a.above, what: 'title, the result line and the small zone with pitch 1 at its true place above the grid (not an edge marker)', info: a }; },
      async (d) => ({ ok: await vis(d.locator('.sv-detail.rec .sv-rec-table')) && /원본 좌표/.test(await d.innerText()) && /이 공 자료에 없어요/.test(await d.innerText()), what: 'facts, the missing-measurements line and the before-pitch table' }));
    await page.locator('.sv-atbat-toggle').click();
    await sheet(page, 'player-batter', () => page.locator('.sv-who.bat').click(), () => page.locator('.sv-who.bat'),
      async (d) => ({ ok: await d.locator('[role=tab][aria-selected=true]').innerText() === '타자' && await vis(d.locator('.sv-facts dd').nth(2)), what: 'the 타자 tab with 등번호 · 시즌 · 오늘' }),
      async (d) => ({ ok: /경기 전 시즌 기록/.test(await d.innerText()), what: 'the note' }));
    await sheet(page, 'player-pitcher', () => page.locator('.sv-who.pit').click(), () => page.locator('.sv-who.pit'),
      async (d) => ({ ok: await d.locator('[role=tab][aria-selected=true]').innerText() === '투수' && await vis(d.locator('.sv-mix-table tbody tr').nth(1)), what: 'the 투수 tab with today\'s mix rows', info: null }),
      async (d) => ({ ok: await vis(d.locator('.sv-mix-table tbody tr').last()), what: 'every mix row' }));
    await sheet(page, 'mix', () => page.locator('.sv-bottom-nav button', { hasText: '구종' }).click(), () => page.locator('.sv-bottom-nav button', { hasText: '구종' }),
      async (d) => ({ ok: /^King 오늘 구종$/.test((await d.locator('h2').innerText()).trim()) && await vis(d.locator('.sv-mix-table tbody tr').last()) && await d.locator('.sv-mix-map').evaluate((e) => getComputedStyle(e).display === 'none'), what: 'the table only (map hidden, design 25)' }),
      async (d) => ({ ok: await vis(d.locator('.sv-mix-map .sv-zone-box')) && /싱=싱커/.test(await d.locator('.sv-mix-key').innerText()), what: 'the location map with its letter key (design 26)' }));
    // undecided sheets: the common frame only
    for (const [name, openIt] of [['record', () => page.locator('.sv-bottom-nav button', { hasText: '기록' }).click()], ['pa-select', () => page.locator('.sv-hd-2 .sv-inning').click()], ['linescore', () => page.locator('.sv-hd-2 .sv-ls').click()]]) {
      await openIt(); await page.getByRole('dialog').waitFor(); await page.waitForTimeout(350);
      ok(await page.locator('.sv-sheet.frame-c.half .sv-sheet-grip').count() === 1, `${name}: common frame (half, handle)`);
      await shot(page, `sheet-${name}-half-390x844`); await page.keyboard.press('Escape'); await page.getByRole('dialog').waitFor({ state: 'detached' });
    }
    ok(errors.length === 0, `interactions 390x844: no page errors ${errors.join(' | ')}`);
    await ctx.close();
  }
  // ---- the opened at-bat scrolled to the end: every row (and the toggle) reachable above the fixed bottom bar ----
  for (const [w, h] of [[320, 568], [390, 844]]) {
    const { ctx, page } = await open(w, h, 30);
    await page.locator('.sv-atbat-toggle').click(); await page.waitForTimeout(200);
    await page.evaluate(() => window.scrollTo(0, document.scrollingElement.scrollHeight)); await page.waitForTimeout(300);
    const r = await page.evaluate(() => { const nav = document.querySelector('.sv-bottom-nav').getBoundingClientRect().top;
      return { nav, rows: [...document.querySelectorAll('.sv-atbat-list > li, .sv-atbat-toggle')].map((e) => ({ t: e.textContent.trim().slice(0, 14), top: e.getBoundingClientRect().top, bottom: e.getBoundingClientRect().bottom })) }; });
    ok(r.rows.length >= 4 && r.rows.every((x) => x.bottom <= r.nav + 0.5), `atbat ${w}x${h} at maximum scroll: every row clears the bottom bar (${r.rows.length} rows, last ${r.rows.at(-1)?.bottom.toFixed(1)} ≤ ${r.nav.toFixed(1)})`, r);
    await shot(page, `atbat-expanded-maxscroll-${w}x${h}`);
    await ctx.close();
  }
  // ---- status bubble (design 08) on the late state ----
  {
    const { ctx, page } = await open(390, 844, 11);
    const chip = page.locator('.sv-hd-status');
    await chip.click(); const pop = page.locator('#sv-status-pop'); await pop.waitFor();
    const text = (await pop.innerText()).replace(/\s+/g, ' '), pb = await rect(pop), cb = await rect(chip);
    ok(/^자료 늦음 경기 자료가/.test(text) && /늦을 수 있어요/.test(text) && await chip.getAttribute('aria-expanded') === 'true' && pb.t >= cb.b && pb.r <= 390 && pb.l >= 0, 'bubble 390x844: title and the state sentence under the marker', { text, pb });
    await shot(page, 'popover-late-390x844');
    await page.keyboard.press('Escape'); await pop.waitFor({ state: 'detached' });
    ok(await chip.evaluate((e) => e === document.activeElement) && await chip.getAttribute('aria-expanded') === 'false', 'bubble: Escape closes, focus back on the marker');
    await chip.click(); await pop.waitFor(); await page.mouse.click(195, 600); await pop.waitFor({ state: 'detached' });
    ok(true, 'bubble: a tap outside closes it');
    await ctx.close();
  }
  for (const step of Object.keys(STATES).map(Number)) for (const [width, height] of [...PHONES, ...WIDE]) {
    const { ctx, page, errors } = await open(width, height, step);
    const id = `step ${step} (${STATES[step].name}) ${width}x${height}`;
    await checkMain(page, step, width, height, id);
    if (width >= 700) { const hd = await rect(page.locator('.sv-hd-1')); ok(hd.w <= 961 && Math.abs((hd.l + hd.r) / 2 - width / 2) < 1, `${id}: header content 960px wide, centred`, hd); }
    await shot(page, `step${String(step).padStart(2, '0')}-${STATES[step].name}-${width}x${height}`);
    ok(errors.length === 0, `${id}: no page errors ${errors.join(' | ')}`);
    await ctx.close();
  }
} finally {
  writeFileSync(path.join(out, 'checks.txt'), `${log.join('\n')}\n${passed} passed, ${failed} failed\n`);
  writeFileSync(path.join(out, 'captures.json'), JSON.stringify(shots.map((f) => path.basename(f)), null, 1));
  await browser.close(); await new Promise((r) => server.close(r));
}
console.log(`${passed} passed, ${failed} failed · ${shots.length} captures in ${out}`);
if (failed) process.exitCode = 1;

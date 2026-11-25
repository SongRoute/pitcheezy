// Route-mocked layout contract for the current service entry. Run after npm run build.
//   node tests/studio-layout-check.mjs [screenshot-directory]
// Moved to the 10/26 screen (docs/handoffs/UI-design-1026.md): header rows 1–3 (batter left, pitcher right), the zone with the at-bat's
// published pitches as numbered marks, a three-cell recommendation row and a folded at-bat line, above a fixed bottom bar. Phones may
// scroll vertically; nothing is hidden by width. On a live game a past pitch opens its own sheet and the main stage stays current.
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fixture = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const out = process.argv[2]; if (out) mkdirSync(out, { recursive: true });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let passed = 0, failed = 0;
const check = (yes, name, detail) => { if (yes) passed++; else { failed++; console.log('FAIL', name, JSON.stringify(detail)); } };
const browser = await chromium.launch();

for (const [width, height] of [[390, 844], [375, 667], [320, 568], [375, 560], [390, 650], [375, 701], [390, 736], [375, 780], [375, 781]]) for (const state of [701, 736, 780, 781].includes(height) ? ['final_stress'] : width === 390 && height === 844 ? ['live', 'live_real', 'final'] : width === 320 ? ['live', 'live_left', 'live_missing', 'final', 'final_long', 'final_stress'] : width === 390 && height === 650 ? ['live', 'live_bottom', 'final', 'final_long', 'final_stress'] : ['live', 'final', 'final_long', 'final_stress']) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, isMobile: true });
  const page = await context.newPage();
  const final = state.startsWith('final');
  const body = structuredClone(fixture[final ? 'final' : 'live']);
  if (!final && state !== 'live_missing') {
    body.current.batter.jersey = '23'; body.current.pitcher.jersey = '123';
    body.pregame = { players: { [body.current.batter.id]: { avg: '.288', ops: '.794' }, [body.current.pitcher.id]: { era: '2.87' } } };
  }
  // The fixture has long player names in earlier pitches; the live state also exercises pre-reveal.
  if (state === 'live') {
    body.current.batter.name = 'An Extraordinarily Long Batter Name';
    body.current.pitcher.name = 'An Extraordinarily Long Pitcher Name';
    body.next.rec.candidates[0].target = { x: 1.6, z: 4.2 };
  }
  if (state === 'live_left') {
    body.current.batter.side = 'L';
    body.next.rec.candidates[0].target = { x: -1.6, z: 1.0 };
  }
  if (state === 'live_missing') {
    body.current.batter = null; body.current.pitcher = null;
    body.pitches = [];
    body.next.rec.candidates[0].target = null;
  }
  if (state === 'live_bottom') body.current.situation.half = 'Bottom';
  if (state === 'final_long') {
    body.pitches[0].actual.result_label = '루킹 스트라이크 아웃 뒤 공수 교대 확인 중';
    body.pitches[0].actual.event = null;
  }
  if (state === 'final_stress') {
    const first = body.pitches[0];
    first.batter.name = 'Christian Encarnacion-Strand'; first.pitcher.name = 'Simeon Woods Richardson';
    first.actual.x = -1.6; first.actual.z = 1.0; first.actual.result_label = '루킹 스트라이크 아웃 뒤 공수 교대 확인 중'; first.actual.event = null;
    first.rec.candidates[0].target = { x: 1.6, z: 4.2 };
    first.context_before = { batter: { at_bats: 2, hits: 1, on_base: 2 }, pitcher: { pitches: 36, strikeouts: 2, earned_runs: 0 },
      next_batters: [{ id: 1, order: 1, name: 'Christian Encarnacion-Strand' }, { id: 2, order: 2, name: 'Simeon Woods Richardson' }] };
  }
  if (state === 'live_real') {
    const context = { batter: { at_bats: 2, hits: 1, on_base: 2 }, pitcher: { pitches: 36, strikeouts: 2, earned_runs: 0 },
      next_batters: [{ id: 1, name: 'Shohei Ohtani', order: 1 }, { id: 2, name: 'Mookie Betts', order: 2 }] };
    body.current.context = context;
    body.pitches.forEach(p => { p.context_before = context; });
  }
  await page.route('http://pz.test/**', route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/service/games/849843') return route.fulfill({ json: body });
    if (pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'fixture' } });
    const file = path.join(dist, pathname);
    const asset = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(asset), contentType: types[path.extname(asset)] ?? 'application/octet-stream' });
  });
  await page.goto('http://pz.test/watch/849843');
  await page.waitForSelector('svg.sv-zone:not(.mini) .sv-zone-box');
  await page.waitForTimeout(150);
  const geometry = await page.evaluate(() => {
    const rect = selector => document.querySelector(selector)?.getBoundingClientRect() ?? null;
    const zone = rect('svg.sv-zone:not(.mini) .sv-zone-box'), head = rect('.sv-hd'), mu = rect('.sv-mu');
    const recs = rect('.sv-rec-sec'), atbat = rect('.sv-atbat');
    const markTop = Math.min(...[...document.querySelectorAll('svg.sv-zone:not(.mini) .sv-rec, svg.sv-zone:not(.mini) .sv-actual')].map(e => e.getBoundingClientRect().top));
    return { head: head.bottom, mu: mu?.top ?? null, muBottom: mu?.bottom ?? null, markTop, zone: zone.top, zoneBottom: zone.bottom, recs: recs.top, atbat: atbat?.top ?? null,
      navTop: document.querySelector('.sv-bottom-nav')?.getBoundingClientRect().top ?? innerHeight, documentHeight: document.documentElement.scrollHeight,
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
      // the page must stay at the CONFIGURED width: a mobile page that overflows grows innerWidth with it, so compare with the setting
      widths: { inner: innerWidth, client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth, vv: visualViewport.width, scale: visualViewport.scale } };
  });
  const key = `${width}x${height} ${state}`;
  // Every piece of content is reachable: nothing off to the side, nothing clipped, no nested scroller, every element can be scrolled
  // above the fixed bottom bar (the 10/26 screen hides no section by width).
  const allVisible = await page.evaluate(() => {
    const selectors = ['.sv-hd', '.sv-hd-score', '.sv-ls', '.sv-mu', 'svg.sv-zone:not(.mini) .sv-zone-box', 'svg.sv-zone:not(.mini) .sv-rec', 'svg.sv-zone:not(.mini) .sv-actual', '.sv-zone-legend', '.sv-rec-sec', '.sv-reclist', '.sv-atbat-head', '.sv-who b', '.sv-who-today'];
    const navHeight = document.querySelector('.sv-bottom-nav')?.getBoundingClientRect().height ?? 0;
    const height = document.documentElement.scrollHeight;
    const shown = e => e.getClientRects().length > 0;
    const hidden = selectors.flatMap(s => [...document.querySelectorAll(s)].filter(e => !shown(e)).map(() => s));
    const bad = selectors.flatMap(s => [...document.querySelectorAll(s)].filter(shown).filter(e => { const r = e.getBoundingClientRect(); return r.top + scrollY < -1 || r.bottom + scrollY > height - navHeight + 1 || r.left < -1 || r.right > innerWidth + 1; }).map(e => ({ selector: s, text: e.textContent?.slice(0, 30), box: e.getBoundingClientRect().toJSON() })));
    const nested = [...document.querySelectorAll('.sv-page *')].filter(e => { const style = getComputedStyle(e); return e.closest('.sv-ls') === null && /auto|scroll/.test(style.overflowY) && e.scrollHeight > e.clientHeight + 1; }).map(e => e.className);
    const clippedText = [...document.querySelectorAll('.sv-who b,.sv-who-today,.sv-rec-cell')].filter(shown)
      .filter(e => e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1).map(e => ({ text: e.textContent, className: e.className }));
    // Key information readable on first paint without scrolling, above the fixed bottom bar.
    const navTop = document.querySelector('.sv-bottom-nav')?.getBoundingClientRect().top ?? innerHeight;
    const core = innerWidth === 390 && innerHeight >= 800
      ? ['.sv-hd', '.sv-mu', 'svg.sv-zone:not(.mini) .sv-zone-box', '.sv-reclist', '.sv-atbat-head']
      : ['.sv-hd', '.sv-mu'];
    const belowFold = core.filter(s => { const e = document.querySelector(s); /* a finished game has no row 3 */ if (!e) return !(s === '.sv-mu' && /경기 종료/.test(document.querySelector('.sv-hd-2')?.textContent ?? '')); const r = e.getBoundingClientRect(); return r.top < -1 || r.bottom > navTop + 1; })
      .map(s => ({ selector: s, bottom: document.querySelector(s)?.getBoundingClientRect().bottom, navTop }));
    return { bad, hidden, nested, clippedText, belowFold, scrollY, height, viewport: innerHeight, navHeight };
  });
  check(allVisible.bad.length === 0 && allVisible.hidden.length === 0 && allVisible.nested.length === 0 && allVisible.clippedText.length === 0,
    `${key} all content reachable above the bottom bar without sideways overflow, clipping or nested scroll`, allVisible);
  check(allVisible.scrollY === 0 && allVisible.belowFold.length === 0, `${key} key information visible without scrolling`, allVisible.belowFold);
  // Reading order: header, row 3 (live only), zone with its marks, recommendation row, at-bat line.
  check(geometry.head <= geometry.zone && (geometry.mu === null || (geometry.head <= geometry.mu + 0.5 && geometry.muBottom + 4 <= geometry.markTop)) && geometry.zoneBottom <= geometry.recs + 2 && (geometry.atbat === null || geometry.recs < geometry.atbat),
    `${key} visual reading order`, geometry);
  check(!geometry.horizontalOverflow, `${key} no horizontal overflow`, geometry);
  const W = geometry.widths;
  check(W.inner === width && W.client === width && W.scroll <= width && Math.abs(W.vv - width) < 0.5 && W.scale === 1, `${key} page stays at the configured ${width}px (inner ${W.inner}, client ${W.client}, scroll ${W.scroll}, visual ${W.vv} × ${W.scale})`, W);
  const cells = await page.locator('.sv-rec-sec .sv-rec-cell').allInnerTexts();
  check(cells.length === 3 && cells.every(c => /^\d\s*\S/.test(c.trim())), `${key} three recommendations`, cells);
  if (!final) {
    const roles = [await page.locator('.sv-who.bat small').innerText(), await page.locator('.sv-who.pit small').innerText()];
    check(roles[0].startsWith('타자') && roles[1].startsWith('투수'), `${key} batter is always left and pitcher right`, roles);
  } else check(await page.locator('.sv-mu').count() === 0 && await page.locator('.sv-hd .sv-hd-status').count() === 0 && /경기 종료/.test(await page.locator('.sv-hd-2').innerText()), `${key} finished game: final header, no row 3, no marker`);
  if (state === 'live_missing') check((await page.locator('.sv-who b').allInnerTexts()).every(s => s.trim() === '—') && await page.locator('svg.sv-zone:not(.mini) .sv-rec').count() === 2,
    `${key} missing players and target stay unknown without fabricated mark`, await page.locator('.sv-who b').allInnerTexts());
  if (!final) {
    // Before reveal the main zone draws only already-published pitches of the current at-bat (UI 10/26: every one of them), each at its
    // published coordinate; the pending pitch has no mark and nothing names its type, speed or result.
    const published = body.pitches.filter(p => p.pa_key === body.current?.pa_key);
    const marks = await page.locator('svg.sv-zone:not(.mini) .sv-actual').evaluateAll(es => es.map(e => ({
      dot: e.classList.contains('sv-pa-dot'), x: Number(e.dataset.x), z: Number(e.dataset.z), n: Number(/^(\d+)구/.exec(e.querySelector('title')?.textContent.trim() ?? '')?.[1]) })));
    const onPublished = m => m.dot && published.some(p => p.pitch_number === m.n && Math.abs(p.actual.x - m.x) < 1e-6 && Math.abs(p.actual.z - m.z) < 1e-6);
    check(marks.length <= published.length && marks.every(onPublished) && await page.locator('.sv-actual-line').count() === 0,
      `${key} actual pitch stays hidden before reveal`, { marks, published: published.map(p => p.key) });
    if (state === 'live' || state === 'live_left') check(await page.locator('svg.sv-zone:not(.mini) .sv-rec.out .sv-out-arrow').count() >= 1, `${key} outside recommendation keeps edge arrow`);
    check(!/km\/h|mph/.test(await page.locator('.sv-stage').innerText()), `${key} no speed on the main stage before reveal`);
  } else {
    check(await page.locator('.sv-actual-line .sv-actual-name').count() === 1, `${key} published actual of the selected pitch remains`);
  }
  if (out) await page.screenshot({ path: path.join(out, `${key.replace(/ /g, '-')}.png`), fullPage: true });
  if (state === 'live' && width === 390) {
    // long names: the full name falls back to the family name inside its cell; widening the screen restores it, narrowing falls back
    // again, and a later poll does not flip it (fit measured against the cell, not against the shown name)
    const fit = () => page.evaluate(() => [...document.querySelectorAll('.sv-who')].map(w => { const b = w.querySelector(':scope > b'), r = b.getBoundingClientRect(), c = w.getBoundingClientRect();
      return { tight: w.classList.contains('tight'), shown: getComputedStyle(b.querySelector('.sur')).display !== 'none' ? b.querySelector('.sur').textContent : b.querySelector('.full').textContent, inCell: r.left >= c.left - .5 && r.right <= c.right + .5, cut: b.scrollWidth > b.clientWidth + 1 }; }));
    const narrow = await fit();
    check(narrow.every(f => f.tight && f.shown === 'Name' && f.inCell && !f.cut), `${key} long names show the family name inside their cells`, narrow);
    await page.setViewportSize({ width: 1100, height }); await page.waitForTimeout(400);
    const wide = await fit();
    check(wide.every(f => !f.tight && /Extraordinarily/.test(f.shown) && f.inCell && !f.cut), `${key} widened: full names come back`, wide);
    await page.setViewportSize({ width, height }); await page.waitForTimeout(400);
    const back = await fit(); await page.waitForTimeout(2600); const later = await fit();
    const w2 = await page.evaluate(() => ({ client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth, inner: innerWidth }));
    check(JSON.stringify(back) === JSON.stringify(narrow) && JSON.stringify(later) === JSON.stringify(narrow) && w2.client === width && w2.scroll <= width && w2.inner === width, `${key} narrowed again and after a poll: same fallback, page at ${width}px`, { back, later, w2 });
    // the bottom bar takes an ordinary tap (nothing outside the configured viewport)
    await page.locator('.sv-bottom-nav button', { hasText: '구종' }).click({ timeout: 5000 });
    const dlg = page.getByRole('dialog'); await dlg.waitFor(); check(true, `${key} bottom bar answers an ordinary tap`); await page.keyboard.press('Escape'); await dlg.waitFor({ state: 'detached' });
  }
  if (state === 'live_real') {
    check((await page.locator('.sv-who.bat .sv-who-today').innerText()).includes('2타수') && (await page.locator('.sv-who.pit .sv-who-today').innerText()).includes('36구'), `${key} current player data preserved in row 3`);
    await page.locator('.sv-rec-sec .sv-reclist').click();
    check(await page.locator('.sv-sheet').getByText('아직 던지지 않았어요.').count() === 1 && await page.locator('.sv-sheet .sv-detail-zone').count() === 0,
      `${key} detail keeps unrevealed actual hidden`);
    await page.locator('.sv-sheet-head').getByRole('button', { name: '닫기' }).click();
    // A past pitch of a live game opens its own sheet; the main stage stays on the unrevealed current pitch and the marker is unchanged.
    const stageBefore = await page.locator('.sv-stage').innerText(), liveBefore = await page.locator('.sv-hd-status').evaluate(e => [e.dataset.tone, e.innerText]);
    const selected = body.pitches.at(-1);
    await page.locator('.sv-atbat-toggle').click();
    const row = page.locator('.sv-pitch-row').first();
    check((await row.locator('.sv-g1').innerText()).trim() === String(selected.pitch_number), `${key} first listed pitch is the latest published pitch`, await row.innerText());
    const stageOpen = await page.locator('.sv-stage').innerText();
    await row.click();
    const dialog = page.getByRole('dialog'); await dialog.waitFor();
    const sheet = (await dialog.innerText()).replace(/\s+/g, ' ');
    check(sheet.includes(`${selected.pitch_number}구`) && sheet.includes(selected.batter.name) && sheet.includes(selected.actual.pitch_label) && /km\/h/.test(sheet) &&
      await dialog.locator('.sv-actual:not(.sv-pa-dot)').count() === 1 && await page.locator('[role=dialog]').count() === 1,
      `${key} past-pitch selection reveals only the selected pitch in its detail sheet`, sheet);
    check(await page.locator('.sv-actual-line').count() === 0 && await page.locator('.sv-stage').innerText() === stageOpen && stageOpen.startsWith(stageBefore.split('\n')[0]),
      `${key} past-pitch selection keeps the main stage on the unrevealed current pitch`);
    const liveAfter = await page.locator('.sv-hd-status').evaluate(e => [e.dataset.tone, e.innerText]);
    check(JSON.stringify(liveAfter) === JSON.stringify(liveBefore), `${key} live selection does not change the marker`, { liveBefore, liveAfter });
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'detached' });
    check(await page.locator('svg.sv-zone:not(.mini) .sv-actual:not(.sv-pa-dot)').count() === 0, `${key} latest stays pre-reveal`);
  }
  if (state === 'live_bottom') {
    // In the bottom half the pitcher is the away (CHC) player: the pitcher cell (row 3, right) opens the 투수 tab; focus returns.
    const opener = page.locator('.sv-who.pit');
    await opener.click();
    const dialog = page.getByRole('dialog'); await dialog.waitFor();
    const sheet = (await dialog.innerText()).replace(/\s+/g, ' ');
    check((await page.locator('.sv-sheet-head').innerText()).includes(body.current.pitcher.name) && /컵스/.test(sheet) && sheet.includes('#123') && await dialog.locator('[role=tab][aria-selected=true]').innerText() === '투수',
      `${key} away pitcher sheet mapping`, sheet);
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'detached' });
    check(await opener.evaluate(e => e === document.activeElement), `${key} player focus restored`);
  }
  if (!final && state !== 'live_missing' && state !== 'live_bottom') {
    // 선수 tab: 타자 first, jersey from the as-of player data, and the 투수 tab carries today's pitch mix
    await page.locator('.sv-bottom-nav button', { hasText: '선수' }).click();
    const dialog = page.getByRole('dialog'); await dialog.waitFor();
    check(await dialog.locator('[role=tab][aria-selected=true]').innerText() === '타자' && (await dialog.innerText()).includes('#23'), `${key} 선수 opens the batter tab with jersey #23`);
    await dialog.getByRole('tab', { name: '투수' }).click();
    check((await dialog.innerText()).includes('#123') && await dialog.locator('.sv-mix-table, .sv-muted').count() >= 1, `${key} pitcher tab: jersey #123 and today's mix`);
    await page.keyboard.press('Escape'); await dialog.waitFor({ state: 'detached' });
  }
  console.log(key, JSON.stringify(geometry));
  await context.close();
}
await browser.close();
console.log(`${passed} ok, ${failed} FAIL`); process.exitCode = failed ? 1 : 0;

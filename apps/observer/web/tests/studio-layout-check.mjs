// Route-mocked layout contract for the current service entry. Run after npm run build.
//   node tests/studio-layout-check.mjs [screenshot-directory]
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
  await page.waitForTimeout(100);
  const geometry = await page.evaluate(() => {
    const rect = selector => document.querySelector(selector).getBoundingClientRect();
    const zone = rect('svg.sv-zone:not(.mini) .sv-zone-box'), duel = rect('.sv-duel'), row = rect('.sv-zrow');
    const summary = rect('.sv-pitch-summary'), strip = rect('.sv-strip-shell'), context = rect('.sv-context');
    const markTop = Math.min(...[...document.querySelectorAll('svg.sv-zone:not(.mini) .sv-rec, svg.sv-zone:not(.mini) .sv-actual')].map(e => e.getBoundingClientRect().top));
    return { duel: duel.top, duelBottom: duel.bottom, markTop, row: row.top, rowBottom: row.bottom, zone: zone.top, zoneBottom: zone.bottom, summary: summary.top,
      strip: strip.top, context: context.top, documentHeight: document.documentElement.scrollHeight,
      below: getComputedStyle(document.querySelector('.sv-stage')).getPropertyValue('--sv-below'),
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1 };
  });
  const key = `${width}x${height} ${state}`;
  const allVisible = await page.evaluate(() => {
    const selectors = ['.sv-head', '.sv-board', '.sv-duel', 'svg.sv-zone:not(.mini) .sv-zone-box', 'svg.sv-zone:not(.mini) .sv-rec', 'svg.sv-zone:not(.mini) .sv-actual', '.sv-pitch-summary', '.sv-rec-summary', '.sv-actual-summary', '.sv-strip-shell', '.sv-context', '.sv-players', '.sv-upcoming', '.sv-duel b', '.sv-player-name', '.sv-player-2', '.sv-player-3', '.sv-upcoming span'];
    const bad = selectors.flatMap(s => [...document.querySelectorAll(s)].filter(e => { const r = e.getBoundingClientRect(); return r.top < -1 || r.bottom > innerHeight + 1 || r.left < -1 || r.right > innerWidth + 1; }).map(e => ({ selector: s, text: e.textContent?.slice(0, 30), box: e.getBoundingClientRect().toJSON() })));
    const nested = [...document.querySelectorAll('.sv-page *')].filter(e => { const style = getComputedStyle(e); return /auto|scroll/.test(style.overflowY) && e.scrollHeight > e.clientHeight + 1; }).map(e => e.className);
    const clippedText = [...document.querySelectorAll('.sv-duel b,.sv-player-name b,.sv-player-2,.sv-player-3,.sv-upcoming span')]
      .filter(e => e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1).map(e => ({ text: e.textContent, className: e.className }));
    return { bad, nested, clippedText, height: document.documentElement.scrollHeight, viewport: innerHeight };
  });
  check(allVisible.height <= height + 1 && allVisible.bad.length === 0 && allVisible.nested.length === 0 && allVisible.clippedText.length === 0,
    `${key} all content in viewport without vertical scrolling or clipping`, allVisible);
  check(geometry.duel < geometry.zone && geometry.duelBottom + 4 <= geometry.markTop && geometry.zoneBottom <= geometry.summary + 2 && geometry.summary < geometry.strip && geometry.strip < geometry.context,
    `${key} visual reading order`, geometry);
  check(!geometry.horizontalOverflow, `${key} no horizontal overflow`, geometry);
  check(await page.locator('.sv-rec-summary .sv-rec-row').count() === 3, `${key} three recommendations`);
  check(await page.locator('.sv-context .sv-player').count() === 2 && await page.locator('.sv-context .sv-upcoming span').count() === 2,
    `${key} all player and upcoming context remains`);
  const duel = await page.locator('.sv-duel b').allInnerTexts();
  check(duel[0]?.startsWith('타자: ') && duel[1]?.startsWith('투수: '), `${key} batter is always left and pitcher right`, duel);
  if (!final && state !== 'live_missing') {
    const footer = await page.locator('.sv-player-name').allInnerTexts();
    const expected = state === 'live_bottom' ? ['투수: CHC #123', '타자: SD #23'] : ['타자: CHC #23', '투수: SD #123'];
    check(footer.every((s, i) => s.includes(expected[i])) && footer.every(s => !s.includes(body.current.batter.name) && !s.includes(body.current.pitcher.name)),
      `${key} footer maps away/home roles and jersey without duplicate names`, footer);
  }
  if (state === 'live_missing') check(duel.every(s => s.includes('선수 미확인')) && await page.locator('svg.sv-zone:not(.mini) .sv-rec').count() === 2,
    `${key} missing players and target stay unknown without fabricated mark`, duel);
  if (!final) {
    check(await page.locator('svg.sv-zone:not(.mini) .sv-actual').count() === 0 && await page.locator('.sv-actual-summary.waiting').count() === 1,
      `${key} actual pitch stays hidden before reveal`);
    if (state === 'live' || state === 'live_left') check(await page.locator('svg.sv-zone:not(.mini) .sv-rec.out .sv-out-arrow').count() >= 1, `${key} outside recommendation keeps edge arrow`);
    check(!(await page.locator('.sv-actual-summary').innerText()).includes('km/h'), `${key} actual speed stays hidden`);
  } else {
    check(await page.locator('.sv-actual-summary.waiting').count() === 0 && await page.locator('.sv-actual-summary .sv-result').count() === 1,
      `${key} published actual result remains`);
  }
  if (out) await page.screenshot({ path: path.join(out, `${key.replace(/ /g, '-')}.png`), fullPage: true });
  if (state === 'live_real') {
    check((await page.locator('.sv-player-3').allInnerTexts()).some(s => s.includes('2타수')) &&
      (await page.locator('.sv-player-3').allInnerTexts()).some(s => s.includes('36구')) &&
      (await page.locator('.sv-upcoming').innerText()).includes('Mookie Betts'), `${key} current player and upcoming data preserved`);
    await page.locator('.sv-rec-summary .sv-reclist').click();
    check(await page.locator('.sv-sheet').getByText('아직 던지지 않았어요.').count() === 1 && await page.locator('.sv-sheet .sv-detail-zone').count() === 0,
      `${key} detail keeps unrevealed actual hidden`);
    await page.locator('.sv-sheet-head button').click();
    await page.locator('.sv-strip .sv-card:not(.pending)').first().click();
    check(await page.locator('.sv-actual-summary.waiting').count() === 0 && await page.locator('.sv-actual-summary .sv-result').count() === 1,
      `${key} past-pitch navigation reveals only selected pitch`);
    check((await page.locator('.sv-live').innerText()).includes('최신으로'), `${key} pinned view offers latest navigation`);
    await page.locator('.sv-live').click();
    check(await page.locator('.sv-actual-summary.waiting').count() === 1, `${key} latest returns to pre-reveal`);
  }
  if (state === 'live_bottom') {
    await page.locator('.sv-player.away').click();
    check((await page.locator('.sv-sheet-head').innerText()).includes(body.current.pitcher.name), `${key} away pitcher sheet mapping`);
    await page.keyboard.press('Escape');
    check(await page.locator('.sv-player.away').evaluate(e => e === document.activeElement), `${key} player focus restored`);
  }
  console.log(key, JSON.stringify(geometry));
  await context.close();
}
await browser.close();
console.log(`${passed} ok, ${failed} FAIL`); process.exitCode = failed ? 1 : 0;

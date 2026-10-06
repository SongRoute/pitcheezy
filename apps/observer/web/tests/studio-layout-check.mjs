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

for (const [width, height] of [[390, 844], [375, 667], [320, 568]]) for (const state of width === 390 ? ['live', 'live_real', 'final'] : width === 320 ? ['live', 'live_left', 'final'] : ['live', 'final']) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, isMobile: true });
  const page = await context.newPage();
  const body = structuredClone(fixture[state === 'final' ? 'final' : 'live']);
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
    return { duel: duel.top, row: row.top, rowBottom: row.bottom, zone: zone.top, zoneBottom: zone.bottom, summary: summary.top,
      strip: strip.top, context: context.top, documentHeight: document.documentElement.scrollHeight,
      below: getComputedStyle(document.querySelector('.sv-stage')).getPropertyValue('--sv-below'),
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1 };
  });
  const key = `${width}x${height} ${state}`;
  check(geometry.duel < geometry.zone && geometry.zoneBottom <= geometry.summary + 2 && geometry.summary < geometry.strip && geometry.strip < geometry.context,
    `${key} visual reading order`, geometry);
  check(!geometry.horizontalOverflow, `${key} no horizontal overflow`, geometry);
  check(await page.locator('.sv-rec-summary .sv-rec-row').count() === 3, `${key} three recommendations`);
  check(await page.locator('.sv-context .sv-player').count() === 2 && await page.locator('.sv-context .sv-upcoming span').count() === 2,
    `${key} all player and upcoming context remains`);
  if (state !== 'final') {
    check(await page.locator('svg.sv-zone:not(.mini) .sv-actual').count() === 0 && await page.locator('.sv-actual-summary.waiting').count() === 1,
      `${key} actual pitch stays hidden before reveal`);
    if (state !== 'live_real') check(await page.locator('svg.sv-zone:not(.mini) .sv-rec.out .sv-out-arrow').count() >= 1, `${key} outside recommendation keeps edge arrow`);
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
  console.log(key, JSON.stringify(geometry));
  await context.close();
}
await browser.close();
console.log(`${passed} ok, ${failed} FAIL`); process.exitCode = failed ? 1 : 0;

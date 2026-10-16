// v3 public bundle (npm run build:v3 → dist-v3/): the v3 screens on published files only. Same fixtures as static-publish-smoke
// (tests/fixtures/publish/v1, real 849843 pitches; 990003 = the synthetic live id). No server, no network: canned answers.
// Covers: list → live card → game; only v1 files are asked (never /api, never a query); candidates only when the server shows them;
// an unconfirmed pitch is neither drawn nor paired; 갱신 지연 from server time; a missing file is said calmly; a lost connection is
// said; no sideways scroll from 320 to 1440 px; a phone sheet is modal (Esc closes, focus returns); the PC panel is NOT modal and the
// live view keeps updating beside it; a public bundle ignores variant query parameters; no page errors.
//   npm run build:v3 && node tests/v3-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST_V3 ?? path.join(here, '..', 'dist-v3');
const files = path.join(here, 'fixtures', 'publish', 'v1');
const shots = process.argv[2]; if (shots) mkdirSync(shots, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.woff2': 'font/woff2', '.woff': 'font/woff' };
let failures = 0, passed = 0;
const ok = (cond, message) => { if (cond) { passed++; console.log('ok  ', message); } else { failures++; console.error('FAIL', message); } };
const clone = (v) => JSON.parse(JSON.stringify(v));
const httpDate = (ms) => new Date(ms).toUTCString();
const iso = (ms) => new Date(ms).toISOString();
const FILE_AT = Date.parse('2026-09-30T05:00:00Z');
const LIVE = 990003;
const disk = (f) => JSON.parse(readFileSync(path.join(files, f), 'utf8'));
const liveBase = disk('game/990003.json');
const stamp = (body, seq, at) => ({ ...body, publish: { contract: 'v1', sequence: seq, published_at: iso(at), final: false } });
const health = (seq, at) => ({ schema: 'pitcheezy-publish-health-v1', sequence: seq, published_at: iso(at), publish_interval_s: 10, files_written: 6,
  unavailable_games: [], unavailable_dates: [], feed_failing_games: [], publish: { contract: 'v1', sequence: seq, published_at: iso(at) } });
const ALLOWED = /^\/pub\/(?:(?:index\.html)?(?:\?.*)?|favicon\.svg|assets\/[\w.-]+|v1\/(?:dates|manifest|health)\.json|v1\/games\/\d{4}-\d{2}-\d{2}\.json|v1\/game\/\d+\.json)$/;

const browser = await chromium.launch();
async function block(name, fn) { try { await fn(); } catch (e) { ok(false, `${name}: aborted ${String(e).split('\n')[0]}`); for (const c of browser.contexts()) await c.close().catch(() => {}); } }
/** answer(file, n) → { json, date, status, abort } or undefined = the fixture file from disk with Date = fixture time + 1 s */
async function open({ width = 390, height = 844, answer = () => undefined, query = '' } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, hasTouch: width < 720, isMobile: width < 720 });
  const page = await context.newPage();
  const asks = [], counts = new Map(), errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.route('http://pz.test/**', async (route) => {
    const url = new URL(route.request().url());
    asks.push(url.pathname + url.search);
    if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 500, json: { detail: 'static mode must not call the API' } });
    const rel = url.pathname.replace(/^\/pub(?=\/)/, '');
    if (rel.startsWith('/v1/')) {
      const file = rel.slice(4), n = (counts.get(file) ?? 0) + 1; counts.set(file, n);
      const a = await answer(file, n);
      if (a?.abort) return route.abort('connectionfailed').catch(() => {});
      const headers = { 'content-type': 'application/json', date: httpDate(a?.date ?? FILE_AT + 1000) };
      if (a?.status) return route.fulfill({ status: a.status, contentType: 'text/html', body: '<html>not found</html>' });
      if (a?.json) return route.fulfill({ headers, body: JSON.stringify(a.json) }).catch(() => {});
      const f = path.join(files, file);
      if (!existsSync(f)) return route.fulfill({ status: 404, contentType: 'text/html', body: '<html>not found</html>' });
      return route.fulfill({ headers, body: readFileSync(f) });
    }
    const f = path.join(dist, rel);
    const target = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  await page.goto(`http://pz.test/pub/${query}`);
  return { context, page, asks, errors, counts };
}
const shot = async (page, name) => { if (shots) await page.screenshot({ path: path.join(shots, `${name}.png`) }); };
const noSideScroll = (page) => page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const dataAsks = (asks) => asks.filter((a) => a.includes('/v1/'));

if (!existsSync(path.join(dist, 'index.html'))) { console.error(`no bundle at ${dist}: run npm run build:v3`); process.exit(2); }

await block('list', async () => {
  const { context, page, asks, errors } = await open();
  await page.locator('.v3-livecard').waitFor({ timeout: 9000 });
  ok(await page.locator('.v3-livecard').count() === 1, 'list: the live game is pinned as one large card');
  ok(await page.locator('.v3-gamerow.final').count() === 1, 'list: the finished game is listed below');
  ok((await page.locator('.v3-livecard').getAttribute('href'))?.endsWith(`?game=${LIVE}`), 'list: the live card links to ?game=<pk>');
  ok(await noSideScroll(page), 'list: no sideways scroll at 390');
  await shot(page, 'list');
  await page.locator('.v3-livecard').click();
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  ok(page.url().endsWith(`?game=${LIVE}`), 'list → game by the card');
  ok(asks.every((a) => ALLOWED.test(a)), 'list: only the page, its assets and v1 files are asked');
  ok(dataAsks(asks).every((a) => !a.includes('?')) && asks.every((a) => !a.includes('/api/')), 'list: no /api, no query on a data file');
  ok(errors.length === 0, `list: no page errors ${errors.join(' | ')}`);
  await context.close();
});

await block('live', async () => {
  const { context, page, errors } = await open({ query: `?game=${LIVE}&zone=A&list=side` });
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  ok(await page.locator('.v3-cand').count() === 3, 'live: three candidates when the server shows them');
  ok(await page.locator('.v3-zone.main .v3-z-rec').count() >= 1, 'live: recommendation cells are drawn');
  ok(await page.locator('.v3-zone.main').getAttribute('data-mark') === 'B', 'live: a public bundle keeps the built zone mark (ignores ?zone=A)');
  ok(await page.locator('.v3-pitches.side').count() === 0 && await page.locator('.v3-pitches.row').count() === 1, 'live: a public bundle keeps the built list layout (ignores ?list=side)');
  ok(await page.locator('.v3-zone.main').getAttribute('data-view') === 'catcher', 'live: catcher view by default');
  ok(/포수 쪽에서 본 스트라이크존/.test(await page.locator('.v3-zone.main').getAttribute('aria-label') ?? ''), 'live: the zone has a text alternative');
  ok(await page.locator('.v3-chip.conn.ok').count() === 1, 'live: connection chip says ok');
  await shot(page, 'live-390');
  ok(errors.length === 0, `live: no page errors ${errors.join(' | ')}`);
  await context.close();
});

await block('hidden', async () => {
  const body = clone(liveBase); body.display_state = { ...body.display_state, recommendation_shown: false, state: 'unconfirmed', label: '확인 중', message: '경기 상황을 다시 확인하고 있어요.' };
  const { context, page } = await open({ query: `?game=${LIVE}`, answer: (f) => f === `game/${LIVE}.json` ? { json: body } : undefined });
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  ok(await page.locator('.v3-cand').count() === 0, 'shown=false: no candidate chips');
  ok(await page.locator('.v3-z-rec').count() === 0, 'shown=false: no recommendation cell in the zone');
  ok((await page.locator('.v3-state').textContent())?.includes('확인 중'), 'shown=false: the state is said in words');
  await context.close();
});

await block('unconfirmed', async () => {
  const body = clone(liveBase);
  const last = body.pitches[body.pitches.length - 1];
  last.receipt = { status: 'unconfirmed', corrected: false };
  last.actual = { ...last.actual, pitch_type: null, pitch_label: null, speed_mph: null, x: null, z: null, description: null, event: null, result_label: '확인 중' };
  const { context, page } = await open({ query: `?game=${LIVE}`, answer: (f) => f === `game/${LIVE}.json` ? { json: body } : undefined });
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  ok(await page.locator('.v3-zone.main .v3-z-dot').count() === 0, 'unconfirmed: the pitch is not drawn in the zone');
  ok((await page.locator('.v3-pitch.pending').textContent())?.includes('확인 중'), 'unconfirmed: the list says 확인 중');
  await page.locator('.v3-pitch.pending').click();
  await page.locator('.v3-panel').waitFor({ timeout: 5000 });
  ok(await page.locator('.v3-panel .v3-table.compare').count() === 0, 'unconfirmed: its panel pairs nothing with a recommendation');
  await context.close();
});

await block('late', async () => {
  // files answer, but the newest publish is 2 minutes older than the host clock → 갱신 지연
  const late = FILE_AT + 120000;
  const { context, page } = await open({ query: `?game=${LIVE}`, answer: (f) => f === 'health.json' ? { json: health(120, FILE_AT), date: late } : f === `game/${LIVE}.json` ? { json: stamp(liveBase, 120, FILE_AT), date: late } : undefined });
  const seen = await page.locator('.v3-banner[data-kind="late"]').waitFor({ timeout: 12000 }).then(() => true, () => false);
  ok(seen, 'late: 갱신 지연 appears from server time');
  ok(seen && (await page.locator('.v3-banner').textContent())?.includes('갱신 지연'), 'late: in words');
  await shot(page, 'late');
  await context.close();
});

await block('missing', async () => {
  const { context, page } = await open({ query: '?game=123456', answer: () => ({ status: 404 }) });
  ok(await page.locator('.v3-unavailable').waitFor({ timeout: 9000 }).then(() => true, () => false), 'missing: a game with no file is said calmly');
  await context.close();
});

await block('lost', async () => {
  let down = false;
  const { context, page } = await open({ query: `?game=${LIVE}`, answer: () => down ? { abort: true } : undefined });
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  down = true;
  ok(await page.locator('.v3-banner[data-kind="lost"]').waitFor({ timeout: 45000 }).then(() => true, () => false), 'lost: 연결 끊김 is said once reads fail');
  ok(await page.locator('.v3-rec.held').count() === 1, 'lost: the last recommendation stays, marked as last received');
  await context.close();
});

for (const [w, h] of [[320, 640], [375, 667], [390, 844], [768, 1024], [1024, 768], [1280, 800], [1440, 900]]) {
  await block(`fit ${w}`, async () => {
    const { context, page } = await open({ width: w, height: h, query: `?game=${LIVE}` });
    await page.locator('.v3-rec').waitFor({ timeout: 9000 });
    ok(await noSideScroll(page), `fit: no sideways scroll at ${w}x${h}`);
    await shot(page, `game-${w}`);
    await context.close();
  });
}

await block('sheet', async () => {
  const { context, page } = await open({ query: `?game=${LIVE}` });
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  const opener = page.locator('.v3-rec-head .v3-link');
  await opener.click();
  const dlg = page.locator('.v3-panel[role="dialog"][aria-modal="true"]');
  ok(await dlg.waitFor({ timeout: 5000 }).then(() => true, () => false), 'sheet: phone detail is a modal dialog');
  await page.keyboard.press('Escape');
  ok(await dlg.count() === 0, 'sheet: Esc closes it');
  ok(await page.evaluate(() => document.activeElement?.classList.contains('v3-link')), 'sheet: focus returns to the opener');
  await context.close();
});

await block('dock', async () => {
  let seq = 120, score = null;
  const { context, page } = await open({ width: 1440, height: 900, query: `?game=${LIVE}`, answer: (f) => {
    if (f !== `game/${LIVE}.json`) return undefined;
    const body = clone(liveBase); if (score !== null) body.game = { ...body.game, home_score: score };
    return { json: stamp(body, seq, FILE_AT) };
  } });
  await page.locator('.v3-rec').waitFor({ timeout: 9000 });
  await page.locator('.v3-pitches .v3-pitch.latest').click();
  const panel = page.locator('.v3-panel.dock');
  ok(await panel.waitFor({ timeout: 5000 }).then(() => true, () => false), 'dock: PC detail opens docked');
  ok(await panel.getAttribute('role') === 'complementary' && await panel.getAttribute('aria-modal') === null, 'dock: not modal');
  ok(await page.locator('.v3-backdrop').count() === 0, 'dock: no backdrop over the live view');
  score = 7; seq = 121;
  ok(await page.locator('.v3-score strong', { hasText: '7' }).waitFor({ timeout: 9000 }).then(() => true, () => false), 'dock: the live view keeps updating beside the panel');
  await page.locator('.v3-cand').first().click();
  ok((await page.locator('.v3-panel.dock h2').textContent())?.includes('추천'), 'dock: the live view stays usable (a chip switches the panel)');
  await shot(page, 'dock-1440');
  // review fix: the game column is hidden (not removed) while docked, so focus returns to the button that opened the panel
  await page.locator('.v3-panel.dock .v3-panel-head button').click();
  const opener = page.locator('.v3-col-game .v3-link', { hasText: '경기 기록 전체' });
  await opener.focus(); await page.keyboard.press('Enter');
  ok(await page.locator('.v3-col-game').isHidden(), 'dock: the game column steps aside while docked');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(100);
  ok(await page.evaluate(() => document.activeElement?.textContent === '경기 기록 전체'), 'dock: focus returns to the opener after close');
  ok(await page.locator('.v3-sr[role=status]').count() === 1, 'a11y: one always-present live region');
  ok(await page.locator('.v3-cands [role=listitem]').count() === 0 && await page.locator('.v3-cands button').count() > 0, 'a11y: candidate chips stay buttons');
  await context.close();
});

await browser.close();
console.log(`${passed} passed, ${failures} failed`);
if (failures) { console.error(`${failures} check(s) failed`); process.exit(1); }

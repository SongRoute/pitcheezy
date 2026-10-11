// Public static-mode bundle (VITE_PUBLISH_BASE set at build time): the screens read published files only.
// Contract: docs/pm/service/public-publish-contract-draft.md (v0). No server, no network: canned answers on the built bundles.
// Files come from tests/fixtures/publish/v1/** (built by tests/fixtures/build-publish-fixture.mjs from the real-game UI fixture);
// scenario answers are made from the same fixture in memory. Covers: dates → games → game from files under a hosting prefix;
// every request logged and none to /api/, none with ?demo= or ?after=; operator screen not routed / not bundled; a lower publish
// sequence is discarded; "갱신 지연" appears and clears (publisher / hosting / device clock); 404, network failure and a hung
// request keep the last state; display.shown=false hides candidates and an unconfirmed pitch is never paired with a recommendation;
// both placements of publish_seq / published_at; one-screen fit at 375 px (and the other widths); the empty host slot;
// and that the API-mode bundle (dist/) still asks /api/service/** only.
//   npm run build && npm run build:public && node tests/static-publish-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const distPublic = process.env.PZ_WEB_DIST_PUBLIC ?? path.join(here, '..', 'dist-public');
const distApi = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const files = path.join(here, 'fixtures', 'publish', 'v1');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const shots = process.argv[2]; if (shots) mkdirSync(shots, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json' };
let failures = 0, passed = 0;
const ok = (cond, message) => { if (cond) { passed++; console.log('ok  ', message); } else { failures++; console.error('FAIL', message); } };
const clone = (v) => JSON.parse(JSON.stringify(v));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const httpDate = (ms) => new Date(ms).toUTCString();
const FILE_AT = Date.parse('2026-09-30T05:00:00Z');   // published_at of the fixture files

// ---------- payload builders (synthetic display states on real pitches) ----------
const LIVE = 990003;
function display(pitch_key, { shown = true, state = 'recommended', label = '다음 공 추천', message = null } = {}) {
  return { schema: 'pitcheezy-display-state-v1', state, reason: 'fixture', label, message, pitch_key, recommendation_shown: shown, correction: null,
    feed: { ok: true, delayed: false, lag_s: null } };
}
/** A live game file body: the fixture's real pitches marked as the live route's answer. */
function liveBody(base, pitch_key, opts = {}) {
  const body = clone(base);
  body.source = { kind: 'live', session_id: null };
  body.game = { ...body.game, game_pk: LIVE, status: 'live', status_text: 'In Progress', live: true };
  body.receipts = { version: 'fixture', corrected: 0, paired: 0, unconfirmed: 0 };
  body.next = { ...body.next, pitch_key };
  body.display_state = display(pitch_key, opts);
  return body;
}
const iso = (ms) => new Date(ms).toISOString();
/** primary placement (publisher on svc/ws-publisher): the body plus one top-level `publish` key */
const top = (body, seq, at, final = false) => ({ ...body, publish: { contract: 'v1', sequence: seq, published_at: iso(at), final } });
const legacy = (body, seq, at) => ({ ...body, publish_seq: seq, published_at: iso(at) });          // older guess, still accepted
const envelope = (body, seq, at) => ({ publish_seq: seq, published_at: iso(at), body });           // older guess, still accepted
const health = (seq, at, unavailable = []) => ({ schema: 'pitcheezy-publish-health-v1', sequence: seq, published_at: iso(at), publish_interval_s: 10, files_written: 6,
  unavailable_games: unavailable, unavailable_dates: [], feed_failing_games: [], publish: { contract: 'v1', sequence: seq, published_at: iso(at) } });

const browser = await chromium.launch();
/** One scenario: a wait that times out fails that scenario by name and the run goes on to the next one. */
async function block(fn) { try { await fn(); } catch (e) { ok(false, `scenario aborted: ${String(e).split('\n')[0]} @ ${(e.stack ?? '').split('\n').find((l) => l.includes('static-publish-smoke'))?.trim() ?? ''}`); for (const c of browser.contexts()) await c.close().catch(() => {}); } }
/** `answer(file, n)` → { json, status, date, abort, hang } for `v1/<file>` (n = how many times that file was asked), or undefined to
 *  serve the fixture file from disk with a hosting `Date` one second after the fixture's publish time. Every request is logged. */
async function open({ width = 390, height = 664, dist = distPublic, answer = () => undefined, api = null, hidden = false, skewMs = 0 } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  const asks = [], counts = new Map();
  // `hidden`: the tab starts hidden (document.hidden is steered by the test, see setHidden); `skewMs`: the device clock is wrong by this much
  await page.addInitScript(([startHidden, skew]) => {
    window.__hidden = startHidden;
    Object.defineProperty(Document.prototype, 'hidden', { configurable: true, get() { return window.__hidden; } });
    Object.defineProperty(Document.prototype, 'visibilityState', { configurable: true, get() { return window.__hidden ? 'hidden' : 'visible'; } });
    if (skew) { const real = Date.now.bind(Date); Date.now = () => real() + skew; }
  }, [hidden, skewMs]);
  await page.addInitScript(() => { window.__slot = []; window.addEventListener('pitcheezy:host-slot', (e) => window.__slot.push({ screen: e.detail.screen, gamePk: e.detail.gamePk, id: e.detail.element?.id })); });
  await page.route('http://pz.test/**', async (route) => {
    const url = new URL(route.request().url());
    asks.push(url.pathname + url.search);
    if (url.pathname.startsWith('/api/')) return api ? api(route, url) : route.fulfill({ status: 500, json: { detail: 'static mode must not call the API' } });
    const rel = url.pathname.replace(/^\/pub(?=\/)/, '');
    if (rel.startsWith('/v1/')) {
      const file = rel.slice(4), n = (counts.get(file) ?? 0) + 1; counts.set(file, n);
      const a = await answer(file, n);
      if (a?.hang) return;                                   // never answered: the client's own timeout has to end it
      if (a?.delay) await sleep(a.delay);
      if (a?.abort) return route.abort('connectionfailed').catch(() => {});
      if (a?.raw !== undefined) return route.fulfill({ status: 200, contentType: a.contentType ?? 'text/html', body: a.raw }).catch(() => {});
      const headers = { 'content-type': 'application/json', ...(a?.date === null ? {} : { date: httpDate(a?.date ?? FILE_AT + 1000) }), ...(a?.age ? { age: String(a.age) } : {}) };
      if (a) return a.status && a.status !== 200 ? route.fulfill({ status: a.status, contentType: 'text/html', body: '<html>not found</html>' }) : route.fulfill({ headers, body: JSON.stringify(a.json) }).catch(() => {});
      const disk = path.join(files, file);
      if (!existsSync(disk)) return route.fulfill({ status: 404, contentType: 'text/html', body: '<html>not found</html>' });
      return route.fulfill({ headers, body: readFileSync(disk) });
    }
    const file = path.join(dist, rel);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = []; page.on('pageerror', (e) => errors.push(String(e)));
  return { context, page, asks, errors, counts };
}
const setHidden = (page, v) => page.evaluate((h) => { window.__hidden = h; document.dispatchEvent(new Event('visibilitychange')); }, v);
/** Extra public bundles with other build-time config (same source, same `--mode public`), built into dist-public-<name>/. */
function variant(name, env) {
  const out = path.join(here, '..', `dist-public-${name}`);
  execFileSync('npx', ['vite', 'build', '--mode', 'public', '--outDir', out, '--logLevel', 'error'], { cwd: path.join(here, '..'), env: { ...process.env, ...env }, stdio: 'inherit' });
  return out;
}
/** What a request may be in static mode: the page, its assets, or one of the published files — nothing else, and never with a query. */
const ALLOWED = /^\/pub\/(?:(?:index\.html)?(?:\?.*)?|favicon\.svg|assets\/[\w.-]+|v1\/(?:dates|manifest|health)\.json|v1\/games\/\d{4}-\d{2}-\d{2}\.json|v1\/game\/\d+\.json)$/;
const fitsOneScreen = (page) => page.evaluate(() => document.scrollingElement.scrollHeight <= window.innerHeight + 1 && document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const noSideScroll = (page) => page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const dataAsks = (asks) => asks.filter((a) => a.includes('/v1/'));
const clean = (asks) => asks.every((a) => !a.includes('/api/'));
/** requests for data files never carry a query at all (no demo binding, no cursor) */
const bare = (asks) => dataAsks(asks).every((a) => !a.includes('?'));
const kind = (page) => page.locator('.sv-notice.line').getAttribute('data-kind');
const waitKind = (page, k, timeout = 9000) => page.locator(`.sv-notice.line[data-kind="${k}"]`).waitFor({ timeout }).then(() => true, () => false);
const waitPitch = (page, key, timeout = 9000) => page.locator(`.sv-state[data-pitch="${key}"]`).waitFor({ timeout }).then(() => true, () => false);

// ---------- 1. files on disk under a hosting prefix: dates → games → game ----------
await block(async () => {
  const { context, page, asks, errors } = await open({ width: 375, height: 553 });
  await page.goto('http://pz.test/pub/');
  await page.locator('.sv-list .sv-card').first().waitFor();
  const body = await page.locator('body').innerText();
  ok(/2026년 9월 30일 \(수\)/.test(body), 'files: list opens on the newest published date (dates.json + manifest live_dates) that is not in the future');
  ok(await page.getByRole('button', { name: /이전 날짜 2026년 9월 29일/ }).count() === 1, 'files: 2026-09-29 is reachable only through manifest live_dates');
  ok(await page.locator('.sv-list > ul > li').count() === 2, 'files: both games of v1/games/2026-09-30.json listed');
  const href = await page.locator('.sv-list .sv-card').first().getAttribute('href');
  ok(/^\.\/\?game=\d+$/.test(href), `files: game links use the query form on the page's own directory (${href})`);
  ok(await noSideScroll(page), 'files: list fits 375 px wide');
  ok(!/갱신 지연/.test(body), 'files: no late notice when health.json is fresh by the hosting clock');
  ok(await page.locator('#pitcheezy-host-slot[data-screen="list"]').count() === 1 && await page.locator('#pitcheezy-host-slot > *').count() === 0, 'slot: one empty host slot on the list');
  await page.locator('.sv-list .sv-card[href="./?game=849843"]').click();
  await page.locator('.sv-zone').first().waitFor();
  ok(new URL(page.url()).pathname === '/pub/' && new URL(page.url()).search === '?game=849843', 'files: the game opens under the same prefix');
  const game = await page.locator('body').innerText();
  ok(/컵스/.test(game) && /샌디에이고/.test(game) && /종료/.test(game), 'files: finished game 849843 drawn from v1/game/849843.json');
  ok(await page.locator('.sv-rec-row').count() > 0, 'files: the recorded pre-pitch recommendation of the first pitch is shown');
  ok(await fitsOneScreen(page), 'files: finished game fits one screen at 375x553');
  ok(await page.locator('.sv-notice').count() === 0, 'files: a settled final file (sequence 3, health at 120) shows no connection or late notice');
  const slot = await page.evaluate(() => window.__slot);
  ok(slot.length >= 1 && slot.at(-1).screen === 'game' && slot.at(-1).gamePk === 849843 && slot.at(-1).id === 'pitcheezy-host-slot', 'slot: pitcheezy:host-slot event names the screen and game');
  ok(await page.locator('.sv-back').getAttribute('href') === './', 'files: back link stays inside the prefix');
  await sleep(2500);
  const data = dataAsks(asks);
  ok(['/pub/v1/dates.json', '/pub/v1/manifest.json', '/pub/v1/games/2026-09-30.json', '/pub/v1/health.json', '/pub/v1/game/849843.json'].every((f) => data.includes(f)), `files: read ${[...new Set(data)].join(' ')}`);
  ok(data.filter((a) => a === '/pub/v1/game/849843.json').length <= 2, 'files: a finished game is not polled');
  ok(clean(asks) && bare(asks), 'files: no request to /api/, none with demo= or after=');
  ok(asks.every((a) => a.startsWith('/pub/')), 'files: nothing is asked outside the hosting prefix');
  if (shots) await page.screenshot({ path: path.join(shots, 'static-final-375.png') });
  ok(errors.length === 0, `files: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- 2. live game from disk at every width; demo binding and operator route ignored ----------
for (const [width, height] of [[375, 553], [320, 568], [360, 640], [390, 664], [390, 844]]) {
  const { context, page, asks, errors } = await open({ width, height });
  await page.goto(`http://pz.test/pub/?game=${LIVE}&demo=abc123`);
  await page.locator('.sv-rec-row').first().waitFor();
  const tag = `live ${width}x${height}`;
  ok(await page.locator('.sv-rec-row').count() === 3 && await page.locator('.sv-chip').innerText() === '다음 공 추천', `${tag}: three candidates under the server's 다음 공 추천`);
  ok(await kind(page) === 'ok', `${tag}: status line calm (fresh by the hosting clock)`);
  ok(await fitsOneScreen(page), `${tag}: one screen, no scroll`);
  await sleep(2600);
  ok(dataAsks(asks).filter((a) => a.endsWith(`/v1/game/${LIVE}.json`)).length >= 2, `${tag}: the live game file is polled`);
  ok(clean(asks) && bare(asks), `${tag}: no /api/, and ?demo= on the page URL is not forwarded`);
  ok(await page.locator('.sv-back').getAttribute('href') === './', `${tag}: no demo binding on links`);
  if (shots && width === 375) await page.screenshot({ path: path.join(shots, 'static-live-375.png') });
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  await context.close();
}
await block(async () => {
  const { context, page, asks } = await open({});
  await page.goto('http://pz.test/demo-operator');
  await page.locator('.sv-list .sv-card').first().waitFor();
  ok(!/운영자/.test(await page.locator('body').innerText()), 'operator: /demo-operator shows the ordinary list, never the operator screen');
  ok(clean(asks), 'operator: no /api/ request');
  const js = readdirSync(path.join(distPublic, 'assets')).filter((f) => f.endsWith('.js')).map((f) => readFileSync(path.join(distPublic, 'assets', f), 'utf8')).join('\n');
  ok(!/운영자 화면|demo-operator|operator\/sessions/.test(js), 'operator: the public bundle holds no operator screen code');
  await context.close();
});

// ---------- 3. publish sequence: a lower sequence is discarded ----------
await block(async () => {
  const now = FILE_AT;
  const steps = [top(liveBody(fx.live, '849843:11:2'), 10, now), top(liveBody(fx.live_more, '849843:12:2'), 11, now + 2000),
    top(liveBody(fx.live, '849843:11:2'), 9, now - 2000), top(liveBody(fx.live, '849843:11:2'), 9, now - 2000)];
  let step = 0, served = [];
  const { context, page, asks, errors } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(11, now + 2000), date: now + 3000 };
    if (file === `game/${LIVE}.json`) { const s = steps[Math.min(step, steps.length - 1)]; served.push(s.publish.sequence); return { json: s, date: now + 3000 }; }
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'seq: sequence 10 applied');
  const cards10 = await page.locator('.sv-strip .sv-card').count();
  step = 1;
  ok(await waitPitch(page, '849843:12:2'), 'seq: higher sequence 11 applied');
  step = 2; served = [];
  await sleep(5000);
  ok(served.filter((s) => s === 9).length >= 2, `seq: the older file (9) was served ${served.length} times`);
  ok(await page.locator('.sv-state').getAttribute('data-pitch') === '849843:12:2', 'seq: lower sequence 9 discarded, sequence 11 stays on screen');
  ok(await kind(page) === 'ok', 'seq: a discarded older copy is not shown as a lost connection');
  steps.push(top(liveBody(fx.live, '849843:13:1'), 12, now + 4000)); step = 4;
  ok(await waitPitch(page, '849843:13:1'), 'seq: the next higher sequence (12) is applied again');
  ok(cards10 > 0 && clean(asks) && bare(asks), 'seq: whole file every time, no cursor, no /api/');
  ok(errors.length === 0, `seq: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- 4. "갱신 지연": appears, keeps the last state, clears ----------
// (a) hosting clock: Date header − health.json's publish time
await block(async () => {
  const t0 = FILE_AT; let hostNow = t0 + 1000, seq = 20, at = t0;
  const { context, page, asks, errors } = await open({ width: 375, height: 553, answer: (file) => {
    if (file === 'health.json') return { json: health(seq, at), date: hostNow };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq, at), date: hostNow };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'late/host: fresh publish → calm line');
  hostNow = t0 + 29000; await sleep(2600);
  ok(await kind(page) === 'ok', 'late/host: 29 s behind (threshold 30 s, provisional) → still calm');
  hostNow = t0 + 95000;
  ok(await waitKind(page, 'publish-late'), 'late/host: 95 s without a publish → the line turns to 갱신 지연');
  ok(/갱신 지연/.test(await page.locator('.sv-notice.line .sv-line-text').innerText()), 'late/host: the line says 갱신 지연 in words');
  ok(await page.locator('.sv-notice.line').getAttribute('data-publish-clock') === 'host', 'late/host: judged on the hosting clock, not the device clock');
  ok(await page.locator('.sv-chip').innerText().then((t) => /갱신 지연/.test(t)), 'late/host: the slot chip says 갱신 지연');
  ok(/마지막으로 받은 내용/.test(await page.locator('.sv-state-live').innerText()) || /마지막으로 받은/.test(await page.locator('.sv-state').innerText()), 'late/host: says it is the last received state');
  ok(await page.locator('.sv-rec-row').count() === 3 && await page.locator('.sv-state.held').count() === 1, 'late/host: the last candidates stay, dimmed as held');
  ok(await page.locator('.sv-strip .sv-card').count() > 0 && /컵스|샌디에이고/.test(await page.locator('.sv-head-main').innerText()), 'late/host: score and pitch cards stay on screen');
  ok(await page.locator('.sv-live.on').count() === 0, 'late/host: no LIVE dot while late');
  ok(await fitsOneScreen(page), 'late/host: one screen at 375x553 while late');
  if (shots) await page.screenshot({ path: path.join(shots, 'static-late-375.png') });
  seq = 21; at = t0 + 94000;
  ok(await waitKind(page, 'ok'), 'late/host: a new publish clears the notice by polling alone');
  ok(await page.locator('.sv-chip').innerText() === '다음 공 추천' && await page.locator('.sv-state.held').count() === 0, 'late/host: chip and candidates back to normal');
  ok(clean(asks) && bare(asks), 'late/host: no /api/');
  ok(errors.length === 0, `late/host: no page errors ${errors.join(' | ')}`);
  await context.close();
});
// (b) publisher clock only: health.json moved on, this game file did not
await block(async () => {
  const t0 = FILE_AT; let healthAt = t0, healthSeq = 30, fileAt = t0, fileSeq = 30;
  const { context, page, errors } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(healthSeq, healthAt), date: healthAt + 1000 };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), fileSeq, fileAt), date: healthAt + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'late/publisher: calm at the start');
  healthAt = t0 + 80000; healthSeq = 60;
  ok(await waitKind(page, 'publish-late') && await page.locator('.sv-notice.line').getAttribute('data-publish-clock') === 'publisher', 'late/publisher: game file 80 s behind health.json → 갱신 지연 from two publisher times');
  fileAt = t0 + 80000; fileSeq = 60;
  ok(await waitKind(page, 'ok'), 'late/publisher: clears when the game file is published again');
  ok(errors.length === 0, `late/publisher: no page errors ${errors.join(' | ')}`);
  await context.close();
});
// (c) no readable Date header and no health.json: newest publish time seen + elapsed time (the device clock is never used)
await block(async () => {
  let behind = 2000, seq = 40;
  const { context, page, errors } = await open({ answer: (file) => {
    if (file === 'health.json') return { status: 404 };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq++, Date.now() - behind), date: null };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'late/elapsed: calm without health.json and without a Date header');
  behind = 120000;
  ok(await waitKind(page, 'publish-late') && await page.locator('.sv-notice.line').getAttribute('data-publish-clock') === 'elapsed', 'late/elapsed: judged from the newest publish time seen + elapsed time (data-publish-clock=elapsed), never the device clock');
  behind = 2000;
  ok(await waitKind(page, 'ok'), 'late/elapsed: clears');
  ok(errors.length === 0, `late/elapsed: no page errors ${errors.join(' | ')}`);
  await context.close();
});
// (d) list: late only while a listed game is in play
await block(async () => {
  const { context, page } = await open({ width: 375, height: 553, answer: (file) => file === 'health.json' ? { json: health(120, FILE_AT), date: FILE_AT + 300000 } : undefined });
  await page.goto('http://pz.test/pub/');
  await page.locator('.sv-publish-late').waitFor({ timeout: 5000 }).catch(() => {});
  ok(/갱신 지연/.test(await page.locator('.sv-list').innerText()), 'late/list: a list with a game in play says 갱신 지연 when health.json is old');
  ok(await page.locator('.sv-list > ul > li').count() === 2 && await noSideScroll(page), 'late/list: the games stay listed, 375 px fits');
  await page.goto('http://pz.test/pub/?date=2026-09-29');
  await page.getByText('이 날짜에는 경기가 없어요.').waitFor();
  await sleep(500);
  ok(!/갱신 지연/.test(await page.locator('.sv-list').innerText()), 'late/list: no late notice on a date with no game in play');
  await context.close();
});

// (e) today comes from manifest live_dates (dates.json lists archived dates only); a followed date with no file is an empty day
await block(async () => {
  const kst = (ms) => new Date(ms + 9 * 3600000).toISOString().slice(0, 10);
  const today = kst(Date.now()), tomorrow = kst(Date.now() + 86400000);
  const card = { ...liveBody(fx.live, '849843:11:2').game, date_kst: today, start: iso(Date.now() - 3600000) };
  const { context, page, asks, errors } = await open({ width: 375, height: 553, answer: (file) => {
    if (file === 'manifest.json') return { json: top({ schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', live_dates: [today, tomorrow] }, 5, Date.now()), date: Date.now() };
    if (file === 'health.json') return { json: health(5, Date.now()), date: Date.now() };
    if (file === `games/${today}.json`) return { json: top({ date: today, games: [card], live_ok: true }, 5, Date.now()) };
  } });
  await page.goto('http://pz.test/pub/');
  await page.locator('.sv-list .sv-card').first().waitFor();
  ok(new RegExp(`${Number(today.slice(5, 7))}월 ${Number(today.slice(8))}일`).test(await page.locator('.sv-datebar h1').innerText()) && dataAsks(asks).includes(`/pub/v1/games/${today}.json`), 'today: the list opens on today (KST) from manifest live_dates although dates.json does not list it');
  await page.getByRole('button', { name: /다음 날짜/ }).click();
  await page.getByText('이 날짜에는 경기가 없어요.').waitFor();
  ok(await page.locator('.sv-alert').count() === 0, 'today: a followed date with no games file is an empty day, not an error');
  ok(clean(asks) && errors.length === 0, `today: no /api/, no page errors ${errors.join(' | ')}`);
  await context.close();
});
// (f) a game the publisher could not build keeps its old file while health.json moves on (unavailable_games) → 갱신 지연; a pregame file too
await block(async () => {
  const t0 = FILE_AT; let healthAt = t0;
  const pre = liveBody(fx.live, null, { state: 'pregame', label: '경기 전', message: '경기 시작 전이에요.', shown: false });
  pre.game = { ...pre.game, status: 'scheduled', status_text: 'Scheduled', live: false, away_score: null, home_score: null, inning: null, half: null };
  pre.pitches = []; pre.pas = []; pre.cutoff = { index: -1, pitch_key: null }; pre.next = null; pre.current = { phase: 'pregame', situation: null, batter: null, pitcher: null, pa_key: null };
  const { context, page, errors } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(90, healthAt, healthAt > t0 ? [{ game_pk: LIVE, status: 503 }] : []), date: healthAt + 1000 };
    if (file === `game/${LIVE}.json`) return { json: top(pre, 2, t0), date: healthAt + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  await page.locator('.sv-chip').waitFor();
  ok(await page.locator('.sv-chip').innerText() === '경기 전' && await kind(page) === 'ok', 'pregame: calm 경기 전');
  healthAt = t0 + 120000;
  ok(await waitKind(page, 'publish-late', 25000) && /갱신 지연/.test(await page.locator('.sv-chip').innerText()), 'pregame: the file stops while health.json moves on → 갱신 지연 (slow pregame polling)');
  ok(errors.length === 0, `pregame: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- 5. 404 / network failure / hung request keep the last state ----------
await block(async () => {
  // the publisher keeps publishing all through this scenario (server time moves on with real time): only the reads fail
  const born = Date.now(); const t0 = FILE_AT; let mode = 'ok';
  const at = () => t0 + (Date.now() - born), seqNow = () => 50 + Math.floor((Date.now() - born) / 1000);
  const { context, page, asks, errors } = await open({ width: 375, height: 553, answer: (file) => {
    if (file === 'health.json') return mode === 'ok' ? { json: health(seqNow(), at()), date: at() + 1000 } : mode === 'hang' ? { hang: true } : { status: 404 };
    if (file !== `game/${LIVE}.json`) return undefined;
    if (mode === '404') return { status: 404 };
    if (mode === 'down') return { abort: true };
    if (mode === 'hang') return { hang: true };
    if (mode === 'html') return { raw: '<!doctype html><html><body>host fallback page</body></html>' };   // 200 + HTML, not JSON
    return { json: top(liveBody(fx.live, '849843:11:2'), seqNow(), at()), date: at() + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'lost: loaded');
  const cards = await page.locator('.sv-strip .sv-card').count();
  const head = await page.locator('.sv-head-main').innerText();
  const kept = async (tag) => {
    ok(/연결 끊김/.test(await page.locator('.sv-notice.line').innerText()) && /연결 끊김/.test(await page.locator('.sv-chip').innerText()), `${tag}: line and chip say 연결 끊김`);
    ok(await page.locator('.sv-rec-row').count() === 3 && await page.locator('.sv-state.held').count() === 1, `${tag}: last candidates kept, dimmed`);
    ok(await page.locator('.sv-strip .sv-card').count() === cards && await page.locator('.sv-head-main').innerText() === head, `${tag}: pitch cards and score kept (screen not blanked)`);
    ok(await fitsOneScreen(page), `${tag}: one screen at 375x553`);
  };
  for (const m of ['404', 'down', 'html']) {
    mode = m;
    ok(await waitKind(page, 'client'), `lost/${m}: the status line turns to the disconnected alert`);
    await kept(`lost/${m}`);
    mode = 'ok';
    ok(await waitKind(page, 'ok') && await page.locator('.sv-state.held').count() === 0, `lost/${m}: recovers by polling alone`);
  }
  mode = 'hang';
  ok(await waitKind(page, 'client', 22000), 'lost/hang: a request that never answers ends in the disconnected alert (client timeout)');
  await kept('lost/hang');
  mode = 'ok';
  ok(await waitKind(page, 'ok', 22000), 'lost/hang: recovers by polling alone');
  ok(clean(asks) && bare(asks), 'lost: no /api/ fallback is ever tried');
  ok(errors.length === 0, `lost: no page errors ${errors.join(' | ')}`);
  await context.close();
});
await block(async () => {  // first load: a missing game file is an answer (shown, with a way back); an unreachable host is retried by itself
  let mode = '404';
  const { context, page, asks } = await open({ answer: (file) => file === 'game/555.json' ? (mode === '404' ? { status: 404 } : mode === 'down' ? { abort: true } : { json: top(liveBody(fx.live, '849843:11:2'), 1, FILE_AT) }) : undefined });
  await page.goto('http://pz.test/pub/?game=555');
  await page.locator('.sv-unavailable').waitFor();
  ok(/지금 볼 수 없어요/.test(await page.locator('.sv-unavailable').innerText()) && await page.locator('.sv-alert').count() === 0 && await page.locator('.sv-unavailable a.sv-btn').getAttribute('href') === './', 'first load 404: a missing game file reads "지금 볼 수 없어요" (not an error box), with a list link inside the prefix');
  mode = 'down'; await page.getByRole('button', { name: '다시 확인' }).click();
  await page.locator('.sv-retrying').waitFor();
  mode = 'ok';
  ok(await waitPitch(page, '849843:11:2', 12000), 'first load: an unreachable host is retried by itself and the game appears');
  ok(clean(asks) && bare(asks), 'first load: no /api/ fallback');
  await context.close();
});

// ---------- 6. fail-closed display rules hold in static mode ----------
await block(async () => {
  // the server says nothing is shown, yet the file still carries ready candidates: they never reach the screen
  const hidden = liveBody(fx.live, '849843:11:2', { shown: false, state: 'unconfirmed', label: '확인 중', message: '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.' });
  ok(hidden.next.rec.status === 'ready' && hidden.next.rec.candidates.length === 3, 'closed: (fixture) next.rec still carries 3 ready candidates');
  // an unconfirmed pitch whose row still carries its recommendation and its actual pitch
  const idx = hidden.pitches.length - 1;
  hidden.pitches[idx].receipt = { status: 'unconfirmed', corrected: false };
  ok(hidden.pitches[idx].rec?.status === 'ready' && hidden.pitches[idx].actual?.pitch_type, 'closed: (fixture) the unconfirmed row still carries rec + actual');
  // a file with no display_state at all
  const none = liveBody(fx.live, '849843:11:2'); delete none.display_state;
  let which = 'hidden';
  const { context, page, asks, errors } = await open({ width: 375, height: 553, answer: (file) => {
    if (file === 'health.json') return { json: health(70, FILE_AT) };
    if (file === `game/${LIVE}.json`) return { json: top(which === 'hidden' ? hidden : none, which === 'hidden' ? 70 : 71, FILE_AT) };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'closed: loaded');
  ok(await page.locator('.sv-rec-row').count() === 0 && await page.locator('.sv-zone .sv-rec').count() === 0, 'closed: recommendation_shown=false → no candidate rows and no target marks');
  ok(/확인 중$/.test(await page.locator('.sv-chip').innerText()), 'closed: the slot says 확인 중');
  ok(await fitsOneScreen(page), 'closed: one screen at 375x553');
  const un = page.locator('.sv-strip .sv-card.unconfirmed');
  ok(await un.count() === 1 && /확인 중/.test(await un.innerText()), 'closed: the unconfirmed pitch card says 확인 중, no pitch type or speed');
  await un.click();
  await page.locator('.sv-actual-caption.pending').waitFor();
  ok(await page.locator('.sv-rec-row').count() === 0 && await page.locator('.sv-zone .sv-rec').count() === 0 && await page.locator('.sv-zone .sv-actual').count() === 0, 'closed: selecting it shows no recommendation and no pitch location');
  ok(/확인 중/.test(await page.locator('.sv-pitch-summary').innerText()), 'closed: its summary says 확인 중');
  ok(await fitsOneScreen(page), 'closed: one screen at 375x553 on the unconfirmed pitch');
  which = 'none';
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  await page.locator('.sv-state').waitFor();
  await sleep(300);
  ok(await page.locator('.sv-rec-row').count() === 0 && /확인 중$/.test(await page.locator('.sv-chip').innerText()), 'closed: a live file without display_state shows no recommendation (확인 중)');
  ok(clean(asks) && bare(asks), 'closed: no /api/');
  ok(errors.length === 0, `closed: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- 7. the two older placements of the publish extras are still read ----------
await block(async () => {
  let step = 0;
  const bodies = [envelope(liveBody(fx.live, '849843:11:2'), 80, FILE_AT), envelope(liveBody(fx.live_more, '849843:12:2'), 81, FILE_AT + 2000), envelope(liveBody(fx.live, '849843:11:2'), 79, FILE_AT)];
  const { context, page, asks, errors } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(81, FILE_AT + 2000), date: FILE_AT + 3000 };
    if (file === 'manifest.json') return { status: 404 };
    if (file === 'dates.json') return { json: legacy({ dates: ['2026-09-30'] }, 81, FILE_AT) };
    if (file === 'games/2026-09-30.json') return { json: legacy({ date: '2026-09-30', games: [bodies[0].body.game] }, 81, FILE_AT) };
    if (file === `game/${LIVE}.json`) return { json: bodies[step], date: FILE_AT + 3000 };
  } });
  await page.goto('http://pz.test/pub/');
  await page.locator(`.sv-list .sv-card[href="./?game=${LIVE}"]`).click();
  ok(await waitPitch(page, '849843:11:2'), 'older forms: top-level publish_seq (dates, games) and { publish_seq, published_at, body } (game) read the same');
  step = 1; ok(await waitPitch(page, '849843:12:2'), 'older forms: higher sequence applied');
  step = 2; await sleep(4500);
  ok(await page.locator('.sv-state').getAttribute('data-pitch') === '849843:12:2', 'older forms: lower sequence discarded');
  ok(clean(asks) && errors.length === 0, `older forms: no /api/, no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ====================================================================================================================
// Review rework (2026-10-11): each block names the claim it reproduces.
// ====================================================================================================================

// ---------- R1. a crafted query can never produce a request outside <base>/v1/ ----------
await block(async () => {
  const crafted = ['2026-09-30-/../../../../api/probe?after=1&x=', '../../api/x', '..', '2026-02-30', '2026-9-30', '2026-09-30/../../game/1', '%2e%2e%2fapi', '2026-09-30?demo=1'];
  const { context, page, asks, errors } = await open({});
  for (const d of crafted) {
    await page.goto(`http://pz.test/pub/?date=${encodeURIComponent(d)}`);
    await page.locator('.sv-datebar h1').waitFor();
    await sleep(900);
  }
  for (const g of ['849843/../../../api/probe', '1e3', '849843?after=1', '-1', '0x10']) { await page.goto(`http://pz.test/pub/?game=${encodeURIComponent(g)}`); await sleep(900); }
  const bad = asks.filter((a) => !ALLOWED.test(a) || (a.includes('/v1/') && a.includes('?')));
  ok(bad.length === 0, `inject: every request stays inside the prefix and the published layout (${bad.length ? bad.join(' ') : `${asks.length} requests checked`})`);
  ok(clean(asks), 'inject: no /api/ request from a crafted ?date= or ?game=');
  await page.goto(`http://pz.test/pub/?date=${encodeURIComponent(crafted[0])}`);
  await page.locator('.sv-list .sv-card').first().waitFor();
  ok(/9월 30일/.test(await page.locator('.sv-datebar h1').innerText()), 'inject: a date that is not a real calendar date is ignored (the list opens on its default date)');
  ok(errors.length === 0, `inject: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- R2. a body that stalls after the headers ends in the client timeout and polling goes on (real throwaway server) ----------
await block(async () => {
  let mode = 'ok', stalled = 0;
  const live = JSON.stringify(top(liveBody(fx.live, '849843:11:2'), 300, Date.now()));
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x'); const rel = url.pathname.replace(/^\/pub(?=\/)/, '');
    if (rel === `/v1/game/${LIVE}.json`) {
      res.writeHead(200, { 'content-type': 'application/json' });
      if (mode === 'stall') { stalled++; res.write(live.slice(0, 2000)); return; }   // headers + part of the body, then silence
      return res.end(JSON.stringify(top(liveBody(fx.live, '849843:11:2'), 300 + stalled, Date.now())));
    }
    if (rel === '/v1/health.json') { res.writeHead(200, { 'content-type': 'application/json' }); return res.end(JSON.stringify(health(300, Date.now()))); }
    const file = path.join(distPublic, rel); const target = existsSync(file) && statSync(file).isFile() ? file : path.join(distPublic, 'index.html');
    res.writeHead(200, { 'content-type': TYPES[path.extname(target)] ?? 'application/octet-stream' }); res.end(readFileSync(target));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));   // an OS-assigned free port on loopback only
  const origin = `http://127.0.0.1:${server.address().port}`;
  const context = await browser.newContext({ viewport: { width: 390, height: 664 } });
  const page = await context.newPage();
  await page.goto(`${origin}/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'stall: loaded from the throwaway server');
  mode = 'stall';
  ok(await waitKind(page, 'client', 25000), 'stall: a body that stops after the headers ends in the disconnected alert (timeout covers the body)');
  ok(await page.locator('.sv-rec-row').count() === 3, 'stall: the last state stays on screen');
  mode = 'ok';
  ok(await waitKind(page, 'ok', 25000), 'stall: later polls are not blocked; the screen recovers by itself');
  ok(stalled >= 1, `stall: (fixture) ${stalled} stalled answers were served`);
  await context.close(); server.closeAllConnections(); await new Promise((r) => server.close(r));
});

// ---------- R3. switching date while the first date's list is still on its way ----------
await block(async () => {
  let slowA = true;
  const { context, page, errors } = await open({ answer: async (file) => file === 'games/2026-09-30.json' && slowA ? { delay: 3000, json: JSON.parse(readFileSync(path.join(files, file), 'utf8')) } : undefined });
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  await page.getByRole('button', { name: /이전 날짜 2026년 9월 29일/ }).click();
  await page.getByText('이 날짜에는 경기가 없어요.').waitFor();
  await sleep(4500);
  ok(/9월 29일/.test(await page.locator('.sv-datebar h1').innerText()) && await page.locator('.sv-list .sv-card').count() === 0, 'date switch: the late list of the date left behind never overwrites the date now shown');
  ok(errors.length === 0, `date switch: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- R4. the list refreshes, recovers by itself and notices a publisher that stops after entry ----------
await block(async () => {
  let mode = 'down', hostNow = FILE_AT + 1000, liveScore = 1;
  const listBody = () => { const b = JSON.parse(readFileSync(path.join(files, 'games/2026-09-30.json'), 'utf8')); b.games[1] = { ...b.games[1], away_score: liveScore }; return b; };
  const { context, page, asks, errors, counts } = await open({ width: 375, height: 553, answer: (file) => {
    if (mode === 'down') return { abort: true };
    if (mode === 'html') return { raw: '<!doctype html><html><body>index</body></html>' };
    if (file === 'health.json') return { json: health(120, FILE_AT), date: hostNow };
    if (file === 'games/2026-09-30.json') return { json: listBody(), date: hostNow };
    return undefined;
  } });
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  await page.locator('.sv-alert').waitFor();
  ok(await page.locator('.sv-list .sv-card').count() === 0, 'list: first load failed → the error is shown');
  mode = 'ok';
  ok(await page.locator('.sv-list .sv-card').first().waitFor({ timeout: 9000 }).then(() => true, () => false), 'list: recovers by itself after a failed first load (no tap)');
  const before = counts.get('games/2026-09-30.json');
  liveScore = 7;
  ok(await page.locator(`.sv-card[href="./?game=${LIVE}"] .sv-card-team b`).first().filter({ hasText: '7' }).waitFor({ timeout: 14000 }).then(() => true, () => false), 'list: the list is read again at the configured interval (live score updates)');
  ok(counts.get('games/2026-09-30.json') > before, 'list: (requests) the games file was asked again');
  ok(!/갱신 지연/.test(await page.locator('.sv-list').innerText()), 'list: no late notice while publishing is fresh');
  hostNow = FILE_AT + 400000;
  ok(await page.locator('.sv-publish-late').waitFor({ timeout: 14000 }).then(() => true, () => false), 'list: publishing stops after entry → 갱신 지연 appears without a reload');
  mode = 'html';
  await sleep(11500);
  ok(await page.locator('.sv-list .sv-card').count() === 2 && await page.locator('.sv-alert').count() === 0, 'list: a real non-JSON answer (HTML with 200) keeps the list that was on screen');
  mode = 'ok'; await setHidden(page, true); await sleep(500);
  const hiddenAt = dataAsks(asks).length; await sleep(12000);
  ok(dataAsks(asks).length === hiddenAt, `list: no request while the tab is hidden (${dataAsks(asks).length - hiddenAt} made)`);
  await setHidden(page, false); await sleep(1200);
  ok(dataAsks(asks).length > hiddenAt, 'list: reading resumes at once when the tab is visible again');
  ok(await noSideScroll(page), 'list: 375 px fits');
  ok(clean(asks) && bare(asks) && errors.length === 0, `list: no /api/, no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- R5. a cached health.json / stale Date header cannot hide a stopped publisher ----------
const distQuick = variant('quick', { VITE_PUBLISH_STALE_S: '6' });   // same code, a short (test-only) late threshold so elapsed time can be watched
await block(async () => {
  // (a) every answer is the same cached copy: same sequence, same publish time, same (old) Date header, no Age
  const t0 = FILE_AT;
  let fresh = false, seq = 400;
  const { context, page, errors } = await open({ dist: distQuick, answer: (file) => {
    if (file === 'health.json') return fresh ? { json: health(++seq, Date.now()), date: Date.now() } : { json: health(400, t0), date: t0 + 1000 };
    if (file === `game/${LIVE}.json`) return fresh ? { json: top(liveBody(fx.live, '849843:11:2'), seq, Date.now()), date: Date.now() } : { json: top(liveBody(fx.live, '849843:11:2'), 400, t0), date: t0 + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'cached: calm at first (1 s behind by the hosting clock)');
  ok(await waitKind(page, 'publish-late', 16000), 'cached: the same cached answer repeating → elapsed time accumulates and 갱신 지연 appears');
  ok(['host', 'elapsed'].includes(await page.locator('.sv-notice.line').getAttribute('data-publish-clock')), 'cached: judged without the device clock');
  fresh = true;
  ok(await waitKind(page, 'ok'), 'cached: clears when a new publish arrives');
  ok(errors.length === 0, `cached: no page errors ${errors.join(' | ')}`);
  await context.close();
});
await block(async () => {
  // (b) health.json has no readable Date, the game file's answer does: the hosting time is still used, not the device clock
  const t0 = FILE_AT; let hostNow = t0 + 1000;
  const { context, page, errors } = await open({ skewMs: -3600000, answer: (file) => {
    if (file === 'health.json') return { json: health(410, t0), date: null };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), 410, t0), date: hostNow };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'mixed headers: calm');
  hostNow = t0 + 95000;
  ok(await waitKind(page, 'publish-late') && await page.locator('.sv-notice.line').getAttribute('data-publish-clock') === 'host', 'mixed headers: any readable server time wins over the device clock (which is an hour slow here)');
  ok(errors.length === 0, `mixed headers: no page errors ${errors.join(' | ')}`);
  await context.close();
});
await block(async () => {
  // (c) Age is honoured: a cached copy says how old it is
  const t0 = FILE_AT;
  const { context, page } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(420, t0), date: t0 + 1000, age: 90 };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), 420, t0), date: t0 + 1000, age: 90 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitKind(page, 'publish-late'), 'age: Date + Age = 91 s behind → 갱신 지연 at once');
  await context.close();
});
// ---------- R6. a wrong device clock ----------
for (const skewMs of [600000, -600000]) await block(async () => {
  const { context, page, errors } = await open({ dist: distQuick, skewMs, answer: (() => { let stop = false, seq = 500, at = Date.now(); const f = (file) => {
    if (!stop) { seq++; at = Date.now(); }
    if (file === 'health.json') return { json: health(seq, at), date: null };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq, at), date: null };
  }; f.stop = () => { stop = true; }; f.go = () => { stop = false; }; return f; })() });
  const tag = `skew ${skewMs / 60000} min`;
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), `${tag}: loaded (no Date header anywhere)`);
  await sleep(5000);
  ok(await kind(page) === 'ok', `${tag}: fresh publishes → no false 갱신 지연 from the device clock`);
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  await context.close();
});
await block(async () => {
  let stop = false, seq = 520, at = Date.now();
  const { context, page } = await open({ dist: distQuick, skewMs: -600000, answer: (file) => {
    if (!stop) { seq++; at = Date.now(); }
    if (file === 'health.json') return { json: health(seq, at), date: null };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq, at), date: null };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'skew stop: calm');
  stop = true;
  ok(await waitKind(page, 'publish-late', 16000) && await page.locator('.sv-notice.line').getAttribute('data-publish-clock') === 'elapsed', 'skew stop: publishing stops → 갱신 지연 from time elapsed since the newest publish was seen, device clock (10 min slow) not used');
  stop = false;
  ok(await waitKind(page, 'ok'), 'skew stop: clears');
  await context.close();
});

// ---------- R7. a game file that is missing at first is asked again by itself ----------
await block(async () => {
  let there = false;
  const { context, page, counts } = await open({ answer: (file) => file === 'game/556.json' ? (there ? { json: top(liveBody(fx.live, '849843:11:2'), 1, FILE_AT) } : { status: 404 }) : undefined });
  await page.goto('http://pz.test/pub/?game=556');
  await page.locator('.sv-unavailable').waitFor();
  there = true;
  ok(await waitPitch(page, '849843:11:2', 14000), 'missing file: once it is published the game appears without a tap');
  ok(counts.get('game/556.json') >= 2, 'missing file: (requests) asked again at the configured interval');
  await context.close();
});
// ---------- R8. nothing is asked while the tab is hidden, also before the first answer ----------
await block(async () => {
  let mode = 'down';
  const { context, page, counts } = await open({ hidden: true, answer: (file) => file === `game/${LIVE}.json` && mode === 'down' ? { abort: true } : undefined });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  await page.locator('.sv-alert').waitFor();
  const n = counts.get(`game/${LIVE}.json`);
  await sleep(9500);
  ok(counts.get(`game/${LIVE}.json`) === n, `hidden: no retry of a failed first load while hidden (${counts.get(`game/${LIVE}.json`) - n} made)`);
  await page.evaluate(() => window.dispatchEvent(new Event('online'))); await sleep(800);
  ok(counts.get(`game/${LIVE}.json`) === n, 'hidden: an online event does not start a request while hidden');
  mode = 'ok'; await setHidden(page, false);
  ok(await page.locator('.sv-rec-row').first().waitFor({ timeout: 6000 }).then(() => true, () => false), 'hidden: visible again → asked at once and the game appears');
  await context.close();
});
// ---------- R9. health.json requests never overlap; a late failure of an older one cannot undo a newer answer ----------
await block(async () => {
  const t0 = FILE_AT; let phase = 0, open9 = 0, maxOpen = 0;
  const { context, page, counts } = await open({ answer: async (file, n) => {
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), 600, t0), date: t0 + 1000 };
    if (file !== 'health.json') return undefined;
    open9++; maxOpen = Math.max(maxOpen, open9);
    try {
      if (phase === 1 && n === slowN) { await sleep(6000); return { abort: true }; }
      return { json: health(600 + (phase ? 60 : 0), phase ? t0 + 80000 : t0), date: (phase ? t0 + 80000 : t0) + 1000 };
    } finally { open9--; }
  } });
  let slowN = -1;
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'health order: calm');
  slowN = counts.get('health.json') + 1; phase = 1;
  const sawLate = await waitKind(page, 'publish-late', 16000);
  await sleep(7000);
  ok(maxOpen === 1, `health order: health.json requests never overlap (max ${maxOpen} at once)`);
  ok(sawLate && await kind(page) === 'publish-late', 'health order: the older request failing late does not wipe the newer health (갱신 지연 stays)');
  await context.close();
});
// ---------- R10. a pregame file is polled, but nothing is expected to be published before the scheduled start ----------
await block(async () => {
  const t0 = FILE_AT;
  const pre = (start) => { const b = liveBody(fx.live, null, { state: 'pregame', label: '경기 전', message: '경기 시작 전이에요.', shown: false });
    b.game = { ...b.game, status: 'scheduled', status_text: 'Scheduled', live: false, away_score: null, home_score: null, inning: null, half: null, start: iso(start) };
    b.pitches = []; b.pas = []; b.cutoff = { index: -1, pitch_key: null }; b.next = null; b.current = { phase: 'pregame', situation: null, batter: null, pitcher: null, pa_key: null }; return b; };
  let start = t0 + 6 * 3600000;
  const { context, page, counts } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(700, t0), date: t0 + 3600000 };          // the publisher has been idle for an hour
    if (file === `game/${LIVE}.json`) return { json: top(pre(start), 700, t0), date: t0 + 3600000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  await page.locator('.sv-chip').waitFor(); await sleep(11500);
  ok(counts.get(`game/${LIVE}.json`) >= 2 && await kind(page) === 'ok' && await page.locator('.sv-chip').innerText() === '경기 전', 'idle day: a pregame file hours before the start, publisher idle → polled, but no 갱신 지연');
  start = t0 + 1800000;   // the start time has passed by the hosting clock and still nothing new
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitKind(page, 'publish-late', 14000), 'idle day: past the scheduled start with no publish → 갱신 지연');
  await context.close();
});
// ---------- R11. time values come from config and stay consistent with a slow poll interval ----------
await block(async () => {
  const distSlow = variant('slow', { VITE_PUBLISH_POLL_MS: '8000', VITE_PUBLISH_STALE_S: '3', VITE_PUBLISH_LOST_MS: '3000', VITE_PUBLISH_TICK_MS: '500' });
  let seq = 800;
  const { context, page, counts } = await open({ dist: distSlow, answer: (file) => {
    if (file === 'health.json') return { json: health(seq, Date.now()), date: Date.now() };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), ++seq, Date.now()), date: Date.now() };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'slow poll: loaded');
  const seen = new Set();
  for (let i = 0; i < 44; i++) { seen.add(await kind(page)); await sleep(500); }
  ok(counts.get(`game/${LIVE}.json`) >= 3 && counts.get(`game/${LIVE}.json`) <= 5, `slow poll: polled every 8 s as configured (${counts.get(`game/${LIVE}.json`)} reads in 22 s)`);
  ok(seen.size === 1 && seen.has('ok'), `slow poll: thresholds shorter than the poll interval do not flash late/disconnected between polls (saw ${[...seen].join(',')})`);
  await context.close();
});
// ---------- R12. real non-JSON for a game file; game requests are single-flight ----------
await block(async () => {
  const t0 = FILE_AT; let mode = 'ok', n22 = 0;
  const { context, page, errors } = await open({ answer: async (file) => {
    if (file === 'health.json') return { json: health(900, t0), date: t0 + 1000 };
    if (file !== `game/${LIVE}.json`) return undefined;
    if (mode === 'html') return { raw: '<!doctype html><html><body>fallback page</body></html>' };
    if (mode === 'race') { n22++; return n22 === 1 ? { delay: 4000, json: top(liveBody(fx.live, '849843:11:2'), 901, t0), date: t0 + 1000 } : { json: top(liveBody(fx.live_more, '849843:12:2'), 902, t0), date: t0 + 1000 }; }
    return { json: top(liveBody(fx.live, '849843:11:2'), 900, t0), date: t0 + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'race: loaded');
  mode = 'html';
  ok(await waitKind(page, 'client'), 'non-JSON: an HTML page answered with 200 for the game file → disconnected alert');
  ok(await page.locator('.sv-rec-row').count() === 3, 'non-JSON: the last state stays');
  mode = 'race'; await sleep(2300);                       // a poll is now waiting for the slow sequence 901
  ok(await page.locator('.sv-notice.line .sv-link-btn').isDisabled(), 'race: one game request at a time — manual refresh is disabled while an answer is on its way, so two answers cannot cross');
  ok(await waitPitch(page, '849843:12:2', 12000), 'race: the slow answer (901) is applied when it lands, then the next one (902)');
  ok(errors.length === 0, `race: no page errors ${errors.join(' | ')}`);
  await context.close();
});
// ---------- R13. the public bundle contains no API / demo / cursor / operator code ----------
for (const dir of [distPublic, distQuick]) {
  const js = readdirSync(path.join(dir, 'assets')).filter((f) => f.endsWith('.js')).map((f) => readFileSync(path.join(dir, 'assets', f), 'utf8')).join('\n');
  const found = [['/api', /\/api\b/], ['after=', /after=/], ['demo forwarding', /demo=|get\(["'`]demo["'`]\)/], ['legacy feeds', /\/live\/games|\/reveal\/|service\/games/], ['operator', /demo-operator|DemoOperator|운영자 화면/]]
    .filter(([, re]) => re.test(js)).map(([name]) => name);
  ok(found.length === 0, `bundle ${path.basename(dir)}: no API, cursor, demo-forwarding, legacy-feed or operator code (${found.length ? 'found: ' + found.join(', ') : 'clean'})`);
}

// ---------- 8. API mode (dist/, env unset) is unchanged: /api/service/** only, no published file, no host slot ----------
if (existsSync(path.join(distApi, 'index.html'))) {
  const final = clone(fx.final);
  const api = (route, url) => {
    if (url.pathname === '/api/service/dates') return route.fulfill({ json: { dates: ['2026-09-30'] } });
    if (url.pathname === '/api/service/games') return route.fulfill({ json: { date: '2026-09-30', games: [final.game] } });
    if (url.pathname === '/api/service/games/849843') return route.fulfill({ json: final });
    return route.fulfill({ status: 404, json: { detail: '이 경기의 기록이 없습니다.' } });
  };
  const { context, page, asks, errors } = await open({ dist: distApi, api, width: 375, height: 553 });
  await page.goto('http://pz.test/watch?demo=abc');
  await page.locator('.sv-list .sv-card').first().waitFor();
  ok(await page.locator('.sv-list .sv-card').first().getAttribute('href') === '/watch/849843?demo=abc', 'api mode: links keep /watch/<pk> and the demo binding');
  await page.locator('.sv-list .sv-card').first().click();
  await page.locator('.sv-zone').first().waitFor();
  ok(asks.includes('/api/service/dates?demo=abc') && asks.includes('/api/service/games?date=2026-09-30&demo=abc') && asks.includes('/api/service/games/849843?demo=abc'), 'api mode: the three service routes are asked, with ?demo= forwarded');
  ok(asks.every((a) => !a.includes('/v1/')), 'api mode: no published file is asked');
  ok(await page.locator('#pitcheezy-host-slot').count() === 0, 'api mode: no host slot');
  ok(await fitsOneScreen(page), 'api mode: one screen at 375x553');
  ok(errors.length === 0, `api mode: no page errors ${errors.join(' | ')}`);
  await context.close();
} else ok(false, `api mode: ${distApi} is missing (run npm run build)`);

await browser.close();
console.log(`${passed} passed, ${failures} failed`);
if (failures) { console.error(`${failures} check(s) failed`); process.exit(1); }
console.log('static publish smoke passed');

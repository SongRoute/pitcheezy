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
  body.receipts = null;
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
  unavailable_games: unavailable, publish: { contract: 'v1', sequence: seq, published_at: iso(at) } });

const browser = await chromium.launch();
/** `answer(file, n)` → { json, status, date, abort, hang } for `v1/<file>` (n = how many times that file was asked), or undefined to
 *  serve the fixture file from disk with a hosting `Date` one second after the fixture's publish time. Every request is logged. */
async function open({ width = 390, height = 664, dist = distPublic, answer = () => undefined, api = null } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  const asks = [], counts = new Map();
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
      if (a?.abort) return route.abort('connectionfailed');
      const headers = { 'content-type': 'application/json', ...(a?.date === null ? {} : { date: httpDate(a?.date ?? FILE_AT + 1000) }) };
      if (a) return a.status && a.status !== 200 ? route.fulfill({ status: a.status, contentType: 'text/html', body: '<html>not found</html>' }) : route.fulfill({ headers, body: JSON.stringify(a.json) });
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
{
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
}

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
{
  const { context, page, asks } = await open({});
  await page.goto('http://pz.test/demo-operator');
  await page.locator('.sv-list .sv-card').first().waitFor();
  ok(!/운영자/.test(await page.locator('body').innerText()), 'operator: /demo-operator shows the ordinary list, never the operator screen');
  ok(clean(asks), 'operator: no /api/ request');
  const js = readdirSync(path.join(distPublic, 'assets')).filter((f) => f.endsWith('.js')).map((f) => readFileSync(path.join(distPublic, 'assets', f), 'utf8')).join('\n');
  ok(!/운영자 화면|demo-operator|operator\/sessions/.test(js), 'operator: the public bundle holds no operator screen code');
  await context.close();
}

// ---------- 3. publish sequence: a lower sequence is discarded ----------
{
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
}

// ---------- 4. "갱신 지연": appears, keeps the last state, clears ----------
// (a) hosting clock: Date header − health.json's publish time
{
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
}
// (b) publisher clock only: health.json moved on, this game file did not
{
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
}
// (c) no readable Date header and no health.json: the device clock against the game file's publish time (known weakness)
{
  let behind = 2000, seq = 40;
  const { context, page, errors } = await open({ answer: (file) => {
    if (file === 'health.json') return { status: 404 };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq++, Date.now() - behind), date: null };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'late/device: calm without health.json and without a Date header');
  behind = 120000;
  ok(await waitKind(page, 'publish-late') && await page.locator('.sv-notice.line').getAttribute('data-publish-clock') === 'client', 'late/device: falls back to the device clock and says so in data-publish-clock');
  behind = 2000;
  ok(await waitKind(page, 'ok'), 'late/device: clears');
  ok(errors.length === 0, `late/device: no page errors ${errors.join(' | ')}`);
  await context.close();
}
// (d) list: late only while a listed game is in play
{
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
}

// (e) today comes from manifest live_dates (dates.json lists archived dates only); a followed date with no file is an empty day
{
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
}
// (f) a game the publisher could not build keeps its old file while health.json moves on (unavailable_games) → 갱신 지연; a pregame file too
{
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
}

// ---------- 5. 404 / network failure / hung request keep the last state ----------
{
  const t0 = FILE_AT; let mode = 'ok', seq = 50;
  const { context, page, asks, errors } = await open({ width: 375, height: 553, answer: (file) => {
    if (file === 'health.json') return mode === 'ok' ? { json: health(seq, t0), date: t0 + 1000 } : mode === 'hang' ? { hang: true } : { status: 404 };
    if (file !== `game/${LIVE}.json`) return undefined;
    if (mode === '404') return { status: 404 };
    if (mode === 'down') return { abort: true };
    if (mode === 'hang') return { hang: true };
    if (mode === 'html') return { json: 'not json at all', date: t0 + 1000 };
    return { json: top(liveBody(fx.live, '849843:11:2'), seq, t0), date: t0 + 1000 };
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
}
{  // first load: a missing game file is an answer (shown, with a way back); an unreachable host is retried by itself
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
}

// ---------- 6. fail-closed display rules hold in static mode ----------
{
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
}

// ---------- 7. the two older placements of the publish extras are still read ----------
{
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

// Public static-mode bundle (VITE_PUBLISH_BASE set at build time): the screens read published files only.
// Contract: docs/pm/service/public-publish-contract-draft.md (v0). No server, no network: canned answers on the built bundles.
// Files come from tests/fixtures/publish/v1/** (built by tests/fixtures/build-publish-fixture.mjs from the real-game UI fixture);
// scenario answers are made from the same fixture in memory. Covers: dates → games → game from files under a hosting prefix;
// every request logged and none to /api/, none with ?demo= or ?after=; operator screen not routed / not bundled; a lower publish
// sequence is discarded; "갱신 지연" appears and clears (publisher / hosting / device clock); 404, network failure and a hung
// request keep the last state; display.shown=false hides candidates and an unconfirmed pitch is never paired with a recommendation;
// both placements of publish_seq / published_at; key info on the first screen at 375 px (and the other widths; vertical scroll allowed since 2026-10-13); the empty host slot;
// and that the API-mode bundle (dist/) still asks /api/service/** only.
//   npm run build && npm run build:public && node tests/static-publish-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import http from 'node:http';
import { build as bundle } from 'esbuild';
import { build as viteBuild } from 'vite';
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
/** the same body published under another game number (a file is only drawn when its body is the game that was asked for) */
const asGame = (pk, body) => ({ ...body, game: { ...body.game, game_pk: pk } });
/** primary placement (publisher on svc/ws-publisher): the body plus one top-level `publish` key */
const top = (body, seq, at, final = false) => ({ ...body, publish: { contract: 'v1', sequence: seq, published_at: iso(at), final } });
const legacy = (body, seq, at) => ({ ...body, publish_seq: seq, published_at: iso(at) });          // older guess, still accepted
const envelope = (body, seq, at) => ({ publish_seq: seq, published_at: iso(at), body });           // older guess, still accepted
const health = (seq, at, unavailable = []) => ({ schema: 'pitcheezy-publish-health-v1', sequence: seq, published_at: iso(at), publish_interval_s: 10, files_written: 6,
  unavailable_games: unavailable, unavailable_dates: [], feed_failing_games: [], publish: { contract: 'v1', sequence: seq, published_at: iso(at) } });

// Adapter/pure-rule checks also run when the sandbox cannot launch Chromium. No sockets or real fetches.
async function coreChecks() {
  const built = await bundle({ stdin: { contents: "export * from './api'; export * from './publish';", resolveDir: path.join(here, '../src/service') },
    bundle: true, write: false, format: 'esm', platform: 'browser', define: { __PZ_STATIC__: 'true', 'import.meta.env': JSON.stringify({ VITE_PUBLISH_BASE: '.' }) } });
  const api = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
  ok(api.PUBLISH_CONFIG.livePollMs === 10000, 'core: live/health fallback is 10 seconds');
  for (const bad of [undefined, null, '10', 0, -1, NaN, Infinity, 0.0001, 2147484]) ok(api.manifestPollMs(bad) === null, `core: invalid manifest period ${String(bad)} falls back`);
  for (const seconds of [0.5, 10, 75]) ok(api.manifestPollMs(seconds) === seconds * 1000, `core: manifest ${seconds}s overrides fallback`);
  ok(api.staleAfterMs(20000) === 60000, 'core: T is at least three periods');
  const check = { checked_at: iso(FILE_AT), sequence_checked: 10, file_sequence: 8, feed_ok: true };
  const h = { seq: 10, publishedAt: iso(FILE_AT), seenAt: performance.now(), final: false, gameChecks: { [LIVE]: check }, failingGames: [] };
  ok(api.gameCheck(h, LIVE).check.file_sequence === 8, 'core: valid game check accepted');
  for (const bad of [null, [], {}, { ...check, checked_at: null }, { ...check, checked_at: 'bad' }, { ...check, sequence_checked: null },
    { ...check, sequence_checked: -1 }, { ...check, file_sequence: '8' }, { ...check, file_sequence: 11 }, { ...check, feed_ok: null }]) {
    const got = api.gameCheck({ ...h, gameChecks: { [LIVE]: bad } }, LIVE);
    ok(got.present && got.check === null, `core: malformed check stays unverified (${JSON.stringify(bad)})`);
  }
  ok(api.gameCheck({ ...h, gameChecks: {} }, LIVE).present === false, 'core: absent game check uses legacy rules');
  ok(api.gameCheck({ ...h, gameChecks: { [LIVE]: { ...check, file_sequence: null } } }, LIVE).check.file_sequence === null, 'core: explicit withdrawal accepted');
  ok(api.gameCheckLag(h, check, { ms: FILE_AT + 30000, clock: 'host' }).lagMs === 30000, 'core: new freshness uses check/health time');
  ok(api.isPublishStale(api.gameCheckLag(h, check, { ms: FILE_AT + 30001, clock: 'host' }), 10000), 'core: new freshness is late above T');
  const subsecond = { ...h, publishedAt: iso(FILE_AT + 500) };
  ok(api.gameCheckLag(subsecond, { ...check, checked_at: iso(FILE_AT + 500) }, { ms: FILE_AT, clock: 'host' }).lagMs < 1000,
    'core: second-resolution HTTP Date cannot falsely mark a fresh subsecond check late');
  ok(api.gameCheckLag(h, null, { ms: FILE_AT, clock: 'host' }).lagMs === Infinity, 'core: missing check cannot be healthy');
  ok(api.gameCheckLag(h, { ...check, checked_at: iso(FILE_AT + 1) }, { ms: FILE_AT, clock: 'host' }).lagMs === Infinity, 'core: future check cannot be healthy');
  const oldFile = { ...h, publishedAt: iso(FILE_AT - 80000) };
  ok(api.publishLag(oldFile, h, { ms: FILE_AT, clock: 'host' }).lagMs === 80000, 'core: legacy file-age rule preserved');
  // UI-1026 removed data-publish-clock from the page; verify the same provenance contract directly, while browser scenarios verify the warnings.
  ok(api.publishLag(oldFile, h, { ms: FILE_AT, clock: 'host' }).clock === 'publisher', 'core: file behind health uses publisher clock');
  ok(api.publishLag(h, h, { ms: FILE_AT + 95000, clock: 'host' }).clock === 'host', 'core: hosting Date uses host clock');
  ok(api.publishLag(h, null, { ms: FILE_AT + 95000, clock: 'elapsed' }).clock === 'elapsed', 'core: no server Date uses elapsed clock');
  let reply, parses = 0; const requests = [];
  const realFetch = globalThis.fetch, realWindow = globalThis.window;
  globalThis.window = { setTimeout, clearTimeout };
  globalThis.fetch = async (url, init) => { requests.push({ url, init }); return { status: reply.status ?? 200, ok: !reply.status || reply.status === 200,
    headers: new Headers(reply.headers), body: null, json: async () => { parses++; return reply.body; } }; };
  try {
    reply = { body: top(liveBody(fx.live, '849843:11:2'), 8, FILE_AT), headers: { ETag: '"one"' } };
    const a = await api.fetchPublishedDetail(LIVE);
    ok(requests.at(-1).init.headers === undefined && parses === 1, 'core: first 200 parsed without conditional header');
    reply = { status: 304 }; const b = await api.fetchPublishedDetail(LIVE);
    ok(requests.at(-1).init.headers['If-None-Match'] === '"one"' && parses === 1 && b.detail === a.detail && b.stamp === a.stamp, 'core: 304 reuses validated body and stamp without parsing/normalizing');
    reply = { body: top(liveBody(fx.live_more, '849843:12:2'), 9, FILE_AT), headers: { ETag: '"two"' } };
    const c = await api.fetchPublishedDetail(LIVE);
    ok(c.stamp.seq === 9 && c.detail !== a.detail && parses === 2, 'core: 200 → 304 → 200 applies a changed body');
    reply = { body: top(liveBody(fx.live, '849843:11:2'), 7, FILE_AT), headers: { ETag: '"old"' } };
    ok((await api.fetchPublishedDetail(LIVE)).detail === c.detail, 'core: a backwards sequence is rejected');
    reply = { status: 304 }; await api.fetchPublishedDetail(LIVE);
    ok(requests.at(-1).init.headers['If-None-Match'] === '"two"', 'core: backwards response cannot replace validated ETag');
    reply = { body: top(liveBody(fx.live, '849843:13:1'), 10, FILE_AT) };
    await api.fetchPublishedDetail(LIVE); await api.fetchPublishedDetail(LIVE);
    ok(requests.at(-1).init.headers === undefined, 'core: no exposed ETag/Date/Age works as ordinary 200 reads');
    reply = { status: 304 };
    ok(await api.fetchPublishedDetail(559).then(() => false, e => e.status === 0), 'core: cacheless 304 is unreadable, never healthy');
    reply = { body: top({ game: null }, 11, FILE_AT), headers: { ETag: '"bad"' } };
    ok(await api.fetchPublishedDetail(LIVE).then(() => false, e => e.status === 0), 'core: invalid shape cannot populate conditional cache');
    reply = { body: health(20, FILE_AT), headers: { ETag: '"health"' } }; await api.fetchPublishHealth();
    ok(requests.at(-1).init.headers === undefined, 'core: ETags are isolated by URL');
    reply = { body: top({ live_dates: [], publish_interval_s: 17 }, 21, FILE_AT) };
    ok((await api.fetchPublishedManifest()).pollMs === 17000, 'core: manifest adapter exposes polling interval');
    const reads = requests.length;
    ok(await api.isGamePublishedSoon(559) === false && requests.length === reads, 'core: missing-game retries reuse the recent manifest without extra dates/manifest reads');
  } finally { globalThis.fetch = realFetch; if (realWindow === undefined) delete globalThis.window; else globalThis.window = realWindow; }
}
await coreChecks();
console.log(`Core checks: ${passed} passed, ${failures} failed (browser scenarios not yet run)`);
if (process.env.PZ_STATIC_CORE_ONLY === '1') process.exit(failures ? 1 : 0);

const browser = await chromium.launch();
// Keep long-standing regression scenarios at their original test cadence; protocol scenarios below use the real public defaults.
const distRegression = variant('regression', { VITE_PUBLISH_POLL_MS: '2000' });

/** One scenario: a wait that times out fails that scenario by name and the run goes on to the next one. */
const ONLY = process.env.PZ_ONLY ?? null;   // comma-separated scenario tags (also used to replay a scenario on an older build)
async function block(fn, tag = null) { if (ONLY && !ONLY.split(',').includes(tag)) return; try { await fn(); } catch (e) { ok(false, `scenario aborted: ${String(e).split('\n')[0]} @ ${(e.stack ?? '').split('\n').find((l) => l.includes('static-publish-smoke'))?.trim() ?? ''}`); for (const c of browser.contexts()) await c.close().catch(() => {}); } }
/** `answer(file, n)` → { json, status, date, abort, hang } for `v1/<file>` (n = how many times that file was asked), or undefined to
 *  serve the fixture file from disk with a hosting `Date` one second after the fixture's publish time. Every request is logged. */
async function open({ width = 390, height = 664, dist = distRegression, protocol = false, clock = false, answer = () => undefined, api = null, hidden = false, skewMs = 0 } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  await context.route('**/*', route => route.abort()); // no unhandled request may leave the test
  const page = await context.newPage();
  if (clock) { await page.clock.install({ time: FILE_AT }); await page.clock.pauseAt(FILE_AT + 1000); }
  const asks = [], counts = new Map();
  // `hidden`: the tab starts hidden (document.hidden is steered by the test, see setHidden); `skewMs`: the device clock is wrong by this much
  await page.addInitScript(([startHidden, skew]) => {
    window.__hidden = startHidden;
    Object.defineProperty(Document.prototype, 'hidden', { configurable: true, get() { return window.__hidden; } });
    Object.defineProperty(Document.prototype, 'visibilityState', { configurable: true, get() { return window.__hidden ? 'hidden' : 'visible'; } });
    // steerable clocks: `__skew` shifts the device wall clock; `__perfLag` holds the monotonic clock back (what a device sleep does)
    window.__skew = skew; window.__perfLag = 0;
    const realNow = Date.now.bind(Date), realPerf = performance.now.bind(performance);
    Date.now = () => realNow() + window.__skew; performance.now = () => realPerf() - window.__perfLag;
    // every status-line kind and chip text the screen ever shows (a flash between two reads is caught too)
    window.__kinds = []; window.__chips = [];
    const note = () => { const chip = document.querySelector('.sv-hd-status'), text = chip?.textContent ?? '';
      const k = text.includes('갱신 지연') ? 'publish-late' : chip?.dataset.state === 'disconnected' ? 'client' : chip?.dataset.state === 'unconfirmed' ? 'checking' : chip && ['recommended', 'pregame', 'live'].includes(chip.dataset.state) ? 'ok' : null;
      if (k && window.__kinds.at(-1) !== k) window.__kinds.push(k);
      const c = document.querySelector('.sv-hd-status')?.textContent; if (c && window.__chips.at(-1) !== c) window.__chips.push(c); };
    new MutationObserver(note).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
  }, [hidden, skewMs]);
  await page.addInitScript(() => { window.__slot = []; window.addEventListener('pitcheezy:host-slot', (e) => window.__slot.push({ screen: e.detail.screen, gamePk: e.detail.gamePk, id: e.detail.element?.id })); });
  await page.route('http://pz.test/**', async (route) => {
    const url = new URL(route.request().url());
    asks.push(url.pathname + url.search);
    if (url.pathname.startsWith('/api/')) return api ? api(route, url) : route.fulfill({ status: 500, json: { detail: 'static mode must not call the API' } });
    const rel = url.pathname.replace(/^\/pub(?=\/)/, '');
    if (rel.startsWith('/v1/')) {
      const file = rel.slice(4), n = (counts.get(file) ?? 0) + 1; counts.set(file, n);
      let a = await answer(file, n, route.request().headers());
      if (!protocol && file === 'manifest.json' && (!a || a.json)) {
        const json = clone(a?.json ?? JSON.parse(readFileSync(path.join(files, file), 'utf8'))); delete json.publish_interval_s;
        a = { ...a, json };
      }
      if (a?.hang) return;                                   // never answered: the client's own timeout has to end it
      if (a?.delay) await sleep(a.delay);
      if (a?.abort) return route.abort('connectionfailed').catch(() => {});
      if (a?.raw !== undefined) return route.fulfill({ status: 200, contentType: a.contentType ?? 'text/html', body: a.raw }).catch(() => {});
      const headers = { 'content-type': 'application/json', ...(a?.date === null ? {} : { date: httpDate(a?.date ?? FILE_AT + 1000) }), ...(a?.age ? { age: String(a.age) } : {}), ...(a?.etag ? { etag: a.etag } : {}) };
      if (a?.status === 304) return route.fulfill({ status: 304, headers, body: '' });
      if (a) return a.status && a.status !== 200 ? route.fulfill(a.json ? { status: a.status, headers, body: JSON.stringify(a.json) } : { status: a.status, contentType: 'text/html', body: '<html>not found</html>' }) : route.fulfill({ headers, body: JSON.stringify(a.json) }).catch(() => {});
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
  execFileSync(process.execPath, [path.join(here, '../node_modules/vite/bin/vite.js'), 'build', '--mode', 'public', '--outDir', out, '--logLevel', 'error'], { cwd: path.join(here, '..'), env: { ...process.env, ...env }, stdio: 'inherit' });
  return out;
}
/** What a request may be in static mode: the page, its assets, or one of the published files — nothing else, and never with a query. */
const ALLOWED = /^\/pub\/(?:(?:index\.html)?(?:\?.*)?|favicon\.svg|assets\/[\w.-]+|v1\/(?:dates|manifest|health)\.json|v1\/games\/\d{4}-\d{2}-\d{2}\.json|v1\/game\/\d+\.json)$/;
/** A throwaway HTTP server on an OS-assigned loopback port (never a fixed one), for what route interception cannot do: an answer whose
 *  headers arrive and whose body does not. `handle(rel, res)` returns true when it answered; otherwise the bundle is served. */
async function withServer(dist, handle, fn) {
  const server = http.createServer((req, res) => {
    const rel = new URL(req.url, 'http://x').pathname.replace(/^\/pub(?=\/)/, '');
    if (handle(rel, res)) return;
    if (rel.startsWith('/v1/')) { const disk = path.join(files, rel.slice(4)); if (!existsSync(disk)) { res.writeHead(404); return res.end('not found'); }
      res.writeHead(200, { 'content-type': 'application/json' }); return res.end(readFileSync(disk)); }
    const file = path.join(dist, rel); const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    res.writeHead(200, { 'content-type': TYPES[path.extname(target)] ?? 'application/octet-stream' }); res.end(readFileSync(target));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  try { await fn(`http://127.0.0.1:${server.address().port}`); } finally { server.closeAllConnections(); await new Promise((r) => server.close(r)); }
}
/** Game-screen fit. Until 2026-10-13 this asked for one screen with no scroll at all. The approved redesign (docs/decisions.md 2026-10-13,
 *  UI-CONNECTION-2026-10-13.md) lets the phone screen scroll vertically under a fixed bottom tab bar. What the old check protected
 *  still has to hold: no sideways scroll; at first paint (scrollY 0) the UI-1026 header (score and state) and matchup/count row
 *  lie fully above the tab bar and the zone (where the recommendation is drawn) starts above it; on a tall phone (>= 800 px)
 *  the whole core down to the folded at-bat toggle does; scrolled to the
 *  end, nothing is left under the tab bar. Returns '' when it fits, else what does not (printed with the check). */
const keyOnFirstScreen = (page) => page.evaluate(async () => {
  const bad = [], root = document.scrollingElement;
  if (root.scrollWidth > innerWidth + 1) bad.push(`sideways ${root.scrollWidth}>${innerWidth}`);
  window.scrollTo(0, 0); await new Promise((r) => requestAnimationFrame(() => r()));
  const nav = document.querySelector('.sv-bottom-nav'), nr = nav?.getBoundingClientRect();
  const fold = nr && nr.height && getComputedStyle(nav).position === 'fixed' ? nr.top : innerHeight;
  const box = (sel) => { const e = document.querySelector(sel); const r = e?.getBoundingClientRect(); return r && r.width && r.height ? r : null; };
  for (const sel of ['.sv-hd-1', '.sv-zone']) if (!box(sel)) bad.push(`${sel} missing`);  // UI-1026 header is .sv-hd-1
  for (const sel of ['.sv-hd-1', '.sv-mu']) { const r = box(sel); if (r && r.bottom > fold + 1) bad.push(`${sel} bottom ${Math.round(r.bottom)} > ${Math.round(fold)}`); }
  const zone = box('.sv-zone'); if (zone && zone.top >= fold) bad.push(`.sv-zone top ${Math.round(zone.top)} >= ${Math.round(fold)}`);
  const strip = box('.sv-atbat-head'); if (innerHeight >= 800 && strip && strip.bottom > fold + 1) bad.push(`.sv-atbat-head bottom ${Math.round(strip.bottom)} > ${Math.round(fold)}`);
  window.scrollTo(0, root.scrollHeight); await new Promise((r) => requestAnimationFrame(() => r()));
  const fold2 = nav && nav.getBoundingClientRect().height ? nav.getBoundingClientRect().top : innerHeight;
  let low = 0, who = '';
  for (const el of document.querySelectorAll('.sv-page *')) { if (el.closest('.sv-bottom-nav, .sv-backdrop, .sr-only')) continue; const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
    if (!r.width || !r.height || cs.visibility === 'hidden' || cs.position === 'fixed') continue; if (r.bottom > low) { low = r.bottom; who = String(el.className || el.tagName); } }
  if (low > fold2 + 1) bad.push(`at the end ${who} bottom ${Math.round(low)} stays under the tab bar (${Math.round(fold2)})`);
  window.scrollTo(0, 0);
  return bad.join('; ');
}).then((bad) => { if (bad) console.error('     fit:', bad); return bad === ''; });
const noSideScroll = (page) => page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const dataAsks = (asks) => asks.filter((a) => a.includes('/v1/'));
const clean = (asks) => asks.every((a) => !a.includes('/api/'));
/** requests for data files never carry a query at all (no demo binding, no cursor) */
const bare = (asks) => dataAsks(asks).every((a) => !a.includes('?'));
// Status is in the visible header; data-kind on the sr-only note denotes recommendation changes, not transport status.
function screenKind(wanted) {
  const chip = document.querySelector('.sv-hd-status'), text = chip?.textContent ?? '';
  const k = text.includes('갱신 지연') ? 'publish-late' : chip?.dataset.state === 'disconnected' ? 'client'
    : chip?.dataset.state === 'unconfirmed' ? 'checking' : chip && ['recommended', 'pregame', 'live'].includes(chip.dataset.state) ? 'ok' : null;
  return wanted === undefined ? k : k === wanted;
}
const kind = (page) => page.evaluate(screenKind);
const waitKind = (page, k, timeout = 9000) => page.waitForFunction(screenKind, k, { timeout }).then(() => true, () => false);
const openAtBat = async (page) => { const toggle = page.locator('.sv-atbat-toggle'); if (await toggle.getAttribute('aria-expanded') === 'false') await toggle.click(); };
const statusWords = async (page) => { const chip = page.locator('.sv-hd-status'); if (await chip.getAttribute('aria-expanded') !== 'true') await chip.click(); const words = await page.locator('.sv-pop').innerText(); await chip.click(); return words; };
const waitPitch = (page, key, timeout = 9000) => page.locator(`.sv-state[data-pitch="${key}"]`).waitFor({ timeout }).then(() => true, () => false);

// ---------- 1. files on disk under a hosting prefix: dates → games → game ----------
await block(async () => {
  const { context, page, asks, errors } = await open({ width: 375, height: 553 });
  await page.goto('http://pz.test/pub/');
  await page.locator('.sv-list .sv-card').first().waitFor();
  const body = await page.locator('body').innerText();
  ok(/2026년 9월 30일 \(수\)/.test(body), 'files: list opens on the newest published date (dates.json + manifest live_dates) that is not in the future');
  ok(await page.getByRole('button', { name: /이전 날짜 2026년 9월 29일/ }).count() === 1, 'files: 2026-09-29 is reachable only through manifest live_dates');
  ok(await page.locator('.sv-list > section > ul > li').count() === 2, 'files: both games of v1/games/2026-09-30.json listed');
  const href = await page.locator('.sv-list .sv-card').first().getAttribute('href');
  ok(/^\.\/\?game=\d+$/.test(href), `files: game links use the query form on the page's own directory (${href})`);
  ok(await noSideScroll(page), 'files: list fits 375 px wide');
  ok(!/갱신 지연/.test(body), 'files: no late notice when health.json is fresh by the hosting clock');
  ok(await page.locator('#pitcheezy-host-slot[data-screen="list"]').count() === 1 && await page.locator('#pitcheezy-host-slot > *').count() === 0, 'slot: one empty host slot on the list');
  await page.locator('.sv-list .sv-card[href="./?game=849843"]').click();
  await page.locator('.sv-zone').first().waitFor();
  ok(new URL(page.url()).pathname === '/pub/' && new URL(page.url()).search === '?game=849843', 'files: the game opens under the same prefix');
  const game = await page.locator('body').innerText();
  ok(/컵스|CHC/.test(game) && /샌디에이고|SD/.test(game) && /종료|최종/.test(game), 'files: finished game 849843 drawn from v1/game/849843.json');
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() > 0, 'files: the recorded pre-pitch recommendation of the first pitch is shown');
  ok(await keyOnFirstScreen(page), 'files: finished game keeps its key info on the first screen at 375x553');
  ok(await page.locator('.sv-hd-status').count() === 0, 'files: a settled final file (sequence 3, health at 120) shows no connection or late notice');
  const slot = await page.evaluate(() => window.__slot);
  ok(slot.length >= 1 && slot.at(-1).screen === 'game' && slot.at(-1).gamePk === 849843 && slot.at(-1).id === 'pitcheezy-host-slot', 'slot: pitcheezy:host-slot event names the screen and game');
  ok(await page.locator('.sv-back').getAttribute('href') === './', 'files: back link stays inside the prefix');
  await sleep(2500);
  const data = dataAsks(asks);
  ok(['/pub/v1/dates.json', '/pub/v1/manifest.json', '/pub/v1/games/2026-09-30.json', '/pub/v1/health.json', '/pub/v1/game/849843.json'].every((f) => data.includes(f)), `files: read ${[...new Set(data)].join(' ')}`);
  ok(data.filter((a) => a === '/pub/v1/game/849843.json').length <= 2, 'files: finished legacy game keeps a bounded poll cadence');
  ok(clean(asks) && bare(asks), 'files: no request to /api/, none with demo= or after=');
  ok(asks.every((a) => a.startsWith('/pub/')), 'files: nothing is asked outside the hosting prefix');
  if (shots) await page.screenshot({ path: path.join(shots, 'static-final-375.png') });
  ok(errors.length === 0, `files: no page errors ${errors.join(' | ')}`);
  await context.close();
});

// ---------- 2. live game from disk at every width; demo binding and operator route ignored ----------
for (const [width, height] of [[375, 553], [320, 568], [360, 640], [390, 664], [390, 844]]) await block(async () => {
  const { context, page, asks, errors } = await open({ width, height });
  await page.goto(`http://pz.test/pub/?game=${LIVE}&demo=abc123`);
  await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').first().waitFor();
  const tag = `live ${width}x${height}`;
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 3 && /추천/.test(await page.locator('.sv-rec-head').innerText())  /* UI-1026: 추천 · N구 */, `${tag}: three candidates under the server's 다음 공 추천 (rows ${await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count()}, head ${JSON.stringify(await page.locator('.sv-rec-head').innerText().catch(() => null))})`);
  ok(await kind(page) === 'ok', `${tag}: status line calm (fresh by the hosting clock)`);
  ok(await keyOnFirstScreen(page), `${tag}: key info on the first screen, no sideways scroll`);
  await sleep(2600);
  ok(dataAsks(asks).filter((a) => a.endsWith(`/v1/game/${LIVE}.json`)).length >= 2, `${tag}: the live game file is polled`);
  ok(clean(asks) && bare(asks), `${tag}: no /api/, and ?demo= on the page URL is not forwarded`);
  ok(await page.locator('.sv-back').getAttribute('href') === './', `${tag}: no demo binding on links`);
  if (shots && width === 375) await page.screenshot({ path: path.join(shots, 'static-live-375.png') });
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'live-widths');
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
  await openAtBat(page);
  const cards10 = await page.locator('.sv-pitch-row').count();
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
  ok(/갱신 지연/.test(await statusWords(page)), 'late/host: the line says 갱신 지연 in words');
  ok(await kind(page) === 'publish-late', 'late/host: host clock input triggers the warning (clock provenance checked in core)');
  ok(await page.locator('.sv-hd-status').innerText().then((t) => /갱신 지연/.test(t)), 'late/host: the header status marker says 갱신 지연');
  ok(/마지막으로 받은 내용/.test(await page.locator('.sv-state-live').textContent()) || /마지막으로 받은/.test(await page.locator('.sv-state').innerText()), 'late/host: says it is the last received state');
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 3 && await page.locator('.sv-state.held').count() === 1, 'late/host: the last candidates stay, dimmed as held');
  await openAtBat(page);
  ok(await page.locator('.sv-pitch-row').count() > 0 && /CHC|SD/.test(await page.locator('.sv-hd-score').innerText()), 'late/host: score and pitch cards stay on screen');
  ok(await page.locator('.sv-hd-status[data-tone="live"]').count() === 0, 'late/host: no LIVE dot while late');
  ok(await keyOnFirstScreen(page), 'late/host: key info on the first screen at 375x553 while late');
  if (shots) await page.screenshot({ path: path.join(shots, 'static-late-375.png') });
  seq = 21; at = t0 + 94000;
  ok(await waitKind(page, 'ok'), 'late/host: a new publish clears the notice by polling alone');
  ok(/추천/.test(await page.locator('.sv-rec-head').innerText())  /* UI-1026: 추천 · N구 */ && await page.locator('.sv-state.held').count() === 0, 'late/host: chip and candidates back to normal');
  ok(clean(asks) && bare(asks), 'late/host: no /api/');
  ok(errors.length === 0, `late/host: no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'late-host');
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
  ok(await waitKind(page, 'publish-late'), 'late/publisher: game file 80 s behind health.json → 갱신 지연 from two publisher times');
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
    // The screen also reads the manifest: this scenario must expose no hosting clock on any successful response.
    if (file === 'manifest.json') return { json: top({ live_dates: ['2026-09-30'] }, 1, Date.now()), date: null };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq++, Date.now() - behind), date: null };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'late/elapsed: calm without health.json and without a Date header');
  behind = 120000;
  ok(await waitKind(page, 'publish-late'), 'late/elapsed: judged from the newest publish time seen + elapsed time (clock provenance checked in core), never the device clock');
  behind = 2000;
  ok(await waitKind(page, 'ok'), 'late/elapsed: clears');
  ok(errors.length === 0, `late/elapsed: no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'late-elapsed');
// (d) list: late only while a listed game is in play
await block(async () => {
  const { context, page } = await open({ width: 375, height: 553, answer: (file) => file === 'health.json' ? { json: health(120, FILE_AT), date: FILE_AT + 300000 } : undefined });
  await page.goto('http://pz.test/pub/');
  await page.locator('.sv-publish-late').waitFor({ timeout: 5000 }).catch(() => {});
  ok(/갱신 지연/.test(await page.locator('.sv-list').innerText()), 'late/list: a list with a game in play says 갱신 지연 when health.json is old');
  ok(await page.locator('.sv-list > section > ul > li').count() === 2 && await noSideScroll(page), 'late/list: the games stay listed, 375 px fits');
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
// (f) a game that is NOT in play (pregame, rain delay, postponed, doubleheader wait) with a fresh health.json is never late, however old its own file
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
  await page.locator('.sv-hd-status').waitFor();
  ok(await page.locator('.sv-hd-status').innerText() === '경기 전' && await kind(page) === 'ok', 'pregame: calm 경기 전');
  healthAt = t0 + 4 * 3600000;   // hours past the scheduled start (02:00Z): a rain delay; the publisher is alive, this file just has nothing new
  await sleep(23000);
  ok(await kind(page) === 'ok' && await page.locator('.sv-hd-status').innerText() === '경기 전' && !(await page.evaluate(() => window.__kinds)).includes('publish-late'), 'pregame: start time long past, health.json fresh, own file old → still calm (a delay is not a late publish)');
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
  await openAtBat(page);
  const cards = await page.locator('.sv-pitch-row').count();
  const head = await page.locator('.sv-hd-score').innerText();
  const kept = async (tag) => {
    ok(/연결 끊김/.test(await statusWords(page)) && /연결 끊김/.test(await page.locator('.sv-hd-status').innerText()), `${tag}: line and chip say 연결 끊김`);
    ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 3 && await page.locator('.sv-state.held').count() === 1, `${tag}: last candidates kept, dimmed`);
    ok(await page.locator('.sv-pitch-row').count() === cards && await page.locator('.sv-hd-score').innerText() === head, `${tag}: pitch cards and score kept (screen not blanked)`);
    ok(await keyOnFirstScreen(page), `${tag}: key info on the first screen at 375x553`);
  };
  for (const m of ['404', 'down', 'html']) {
    mode = m;
    ok(await waitKind(page, 'client'), `lost/${m}: the status line turns to the disconnected alert`);
    await kept(`lost/${m}`);
    mode = 'ok';
    ok(await waitKind(page, 'ok') && await page.locator('.sv-state.held').count() === 0, `lost/${m}: recovers by polling alone`);
  }
  mode = 'hang';
  ok(await waitKind(page, 'client', 37000), 'lost/hang: a request that never answers ends in the disconnected alert (client timeout)');
  await kept('lost/hang');
  mode = 'ok';
  ok(await waitKind(page, 'ok', 22000), 'lost/hang: recovers by polling alone');
  ok(clean(asks) && bare(asks), 'lost: no /api/ fallback is ever tried');
  ok(errors.length === 0, `lost: no page errors ${errors.join(' | ')}`);
  await context.close();
});
await block(async () => {  // first load: a missing game file is an answer (shown, with a way back); an unreachable host is retried by itself
  let mode = '404';
  const { context, page, asks } = await open({ answer: (file) => file === 'game/555.json' ? (mode === '404' ? { status: 404 } : mode === 'down' ? { abort: true } : { json: top(asGame(555, liveBody(fx.live, '849843:11:2')), 1, FILE_AT) }) : undefined });
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
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && await page.locator('.sv-zone .sv-rec').count() === 0, 'closed: recommendation_shown=false → no candidate rows and no target marks');
  ok(/확인 중$/.test(await page.locator('.sv-hd-status').innerText()), 'closed: the slot says 확인 중');
  ok(await keyOnFirstScreen(page), 'closed: key info on the first screen at 375x553');
  await openAtBat(page);
  const un = page.locator('.sv-pitch-row.unconfirmed');
  ok(await un.count() === 1 && /확인 중/.test(await un.innerText()), 'closed: the unconfirmed pitch card says 확인 중, no pitch type or speed');
  // Since the 2026-10-13 redesign a past pitch of a live game opens in a detail sheet and the live stage behind it stays; the
  // fail-closed rule is checked on both: neither shows a recommendation, a target mark, a pitch location, a type or a speed.
  await un.click();
  const sheet = page.getByRole('dialog'); await sheet.waitFor();
  const sheetText = await sheet.innerText();
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && await page.locator('.sv-zone .sv-rec').count() === 0 && await page.locator('.sv-zone .sv-actual').count() === 0
    && await sheet.locator('.sv-table, .sv-detail-zone, .sv-rec, .sv-actual').count() === 0 && !/km\/h|mph|정책 선택 비중/.test(sheetText),
    'closed: selecting it shows no recommendation and no pitch location (detail sheet and the stage behind it)');
  ok(/이 공을 다시 확인하고 있어요/.test(sheetText), 'closed: its detail says it is being checked (확인 중)');
  await sheet.getByRole('button', { name: '닫기' }).click(); await sheet.waitFor({ state: 'detached' });
  ok(await keyOnFirstScreen(page), 'closed: key info on the first screen at 375x553 after looking at the unconfirmed pitch');
  which = 'none';
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  await page.locator('.sv-state').waitFor();
  await sleep(300);
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && /확인 중$/.test(await page.locator('.sv-hd-status').innerText()), 'closed: a live file without display_state shows no recommendation (확인 중)');
  ok(clean(asks) && bare(asks), 'closed: no /api/');
  ok(errors.length === 0, `closed: no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'fail-closed');

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
    const file = path.join(distRegression, rel); const target = existsSync(file) && statSync(file).isFile() ? file : path.join(distRegression, 'index.html');
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
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 3, 'stall: the last state stays on screen');
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
  ok(await page.locator('.sv-list .sv-card').first().waitFor({ timeout: 14000 }).then(() => true, () => false), 'list: recovers by itself after a failed first load (no tap)');
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
const distQuick = variant('quick', { VITE_PUBLISH_POLL_MS: '2000', VITE_PUBLISH_STALE_S: '6', VITE_PUBLISH_MISSING_RETRY_MS: '1000', VITE_PUBLISH_MISSING_MAX_MS: '4000', VITE_PUBLISH_PINNED_POLL_MS: '1500' });   // same code, a short (test-only) late threshold so elapsed time can be watched
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
  ok(await kind(page) === 'publish-late', 'cached: stale repeated server answers warn (clock provenance checked in core)');
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
  ok(await waitKind(page, 'publish-late'), 'mixed headers: any readable server time wins over the device clock (which is an hour slow here)');
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
  ok(await waitKind(page, 'publish-late', 16000), 'skew stop: publishing stops → 갱신 지연 from time elapsed since the newest publish was seen, device clock (10 min slow) not used');
  stop = false;
  ok(await waitKind(page, 'ok'), 'skew stop: clears');
  await context.close();
});

// ---------- R7. a game file that is missing at first is asked again by itself ----------
await block(async () => {
  let there = false;
  const listed = JSON.parse(readFileSync(path.join(files, 'games/2026-09-30.json'), 'utf8')); listed.games.push({ ...listed.games[1], game_pk: 556 });
  const { context, page, counts } = await open({ answer: (file) => file === 'game/556.json' ? (there ? { json: top(asGame(556, liveBody(fx.live, '849843:11:2')), 1, FILE_AT) } : { status: 404 })
    : file === 'games/2026-09-30.json' ? { json: listed } : undefined });
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
  const { context, page, counts, asks } = await open({ hidden: true, answer: (file) => file === `game/${LIVE}.json` && mode === 'down' ? { abort: true } : undefined });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  await page.locator('.sv-center [role="status"]').waitFor();
  await sleep(9500);
  ok(dataAsks(asks).length === 0, 'hidden: no initial data request before the tab is first shown');
  await page.evaluate(() => window.dispatchEvent(new Event('online'))); await sleep(800);
  ok(dataAsks(asks).length === 0, 'hidden: an online event does not start an initial request');
  await setHidden(page, false); await page.locator('.sv-alert').waitFor();
  await setHidden(page, true); const n = counts.get(`game/${LIVE}.json`);
  await sleep(9500);
  ok(counts.get(`game/${LIVE}.json`) === n, `hidden: no retry of a failed first load while hidden (${counts.get(`game/${LIVE}.json`) - n} made)`);
  await page.evaluate(() => window.dispatchEvent(new Event('online'))); await sleep(800);
  ok(counts.get(`game/${LIVE}.json`) === n, 'hidden: an online event does not retry while hidden');
  mode = 'ok'; await setHidden(page, false);
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').first().waitFor({ timeout: 6000 }).then(() => true, () => false), 'hidden: visible again → asked at once and the game appears');
  await context.close();
}, 'hidden-first-load');
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
  await page.locator('.sv-hd-status').waitFor(); await sleep(11500);
  ok(counts.get(`game/${LIVE}.json`) >= 2 && await kind(page) === 'ok' && await page.locator('.sv-hd-status').innerText() === '경기 전', 'idle day: a pregame file hours before the start, publisher idle → polled, but no 갱신 지연');
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
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 3, 'non-JSON: the last state stays');
  await page.locator('.sv-hd-status').click();
  mode = 'race'; await sleep(2300);                       // a poll is now waiting for the slow sequence 901
  ok(await page.locator('.sv-pop-retry').isDisabled(), 'race: one game request at a time — manual refresh is disabled while an answer is on its way, so two answers cannot cross');
  ok(await waitPitch(page, '849843:12:2', 12000), 'race: the slow answer (901) is applied when it lands, then the next one (902)');
  ok(errors.length === 0, `race: no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'game-race');
// ====================================================================================================================
// Second review round (2026-10-11)
// ====================================================================================================================

// ---------- S1. date A's HEADERS arrive, the viewer switches to B, then A's BODY completes ----------
await block(async () => {
  const bodyA = readFileSync(path.join(files, 'games/2026-09-30.json'), 'utf8');
  let held = null;
  await withServer(distPublic, (rel, res) => {
    if (rel !== '/v1/games/2026-09-30.json') return false;
    res.writeHead(200, { 'content-type': 'application/json' }); res.write(bodyA.slice(0, 40)); held = res; return true;   // headers + a little body, rest withheld
  }, async (origin) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 664 } }); const page = await context.newPage();
    await page.goto(`${origin}/pub/?date=2026-09-30`);
    for (let i = 0; i < 50 && !held; i++) await sleep(100);
    await sleep(400);                                             // the page has A's headers and is reading its body
    await page.getByRole('button', { name: /이전 날짜 2026년 9월 29일/ }).click();
    await page.getByText('이 날짜에는 경기가 없어요.').waitFor();
    held.end(bodyA.slice(40));                                    // A's body completes after the switch
    await sleep(2000);
    ok(held !== null && /9월 29일/.test(await page.locator('.sv-datebar h1').innerText()) && await page.locator('.sv-list .sv-card').count() === 0, 'date body: A (headers in, body late) never overwrites B after the switch');
    await context.close();
  });
}, 'date-body');

// ---------- S2. dates.json / manifest.json are read again: a new followed date appears without a reload ----------
await block(async () => {
  let more = false;
  const { context, page, counts } = await open({ answer: (file) => {
    if (file === 'manifest.json') return { json: top({ schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', live_dates: more ? ['2026-09-29', '2026-09-30', '2026-10-01'] : ['2026-09-29', '2026-09-30'] }, more ? 121 : 120, FILE_AT) };
  } });
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  await page.locator('.sv-list .sv-card').first().waitFor();
  ok(await page.getByRole('button', { name: '다음 날짜 없음' }).count() === 1, 'dates: no later date at first');
  more = true;
  ok(await page.getByRole('button', { name: /다음 날짜 2026년 10월 1일/ }).waitFor({ timeout: 65000 }).then(() => true, () => false), 'dates: a date the publisher starts to follow appears without a reload');
  ok(counts.get('manifest.json') >= 2 && counts.get('dates.json') >= 2 && /9월 30일/.test(await page.locator('.sv-datebar h1').innerText()), 'dates: (requests) both files were read again; the shown date did not change');
  await context.close();
}, 'dates');

// ---------- S3. hidden while the list answer is on its way: no follow-up health.json request ----------
await block(async () => {
  let slow = false;
  const { context, page, counts } = await open({ answer: async (file) => file === 'games/2026-09-30.json' && slow ? { delay: 1500, json: JSON.parse(readFileSync(path.join(files, file), 'utf8')) } : undefined });
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  await page.locator('.sv-list .sv-card').first().waitFor(); await sleep(600);
  slow = true; const n = counts.get('games/2026-09-30.json');
  for (let i = 0; i < 140 && counts.get('games/2026-09-30.json') === n; i++) await sleep(100);   // the next list read has started
  await setHidden(page, true);
  const h = counts.get('health.json'), d = counts.get('dates.json');
  await sleep(3500);
  ok(counts.get('games/2026-09-30.json') === n + 1 && counts.get('health.json') === h && counts.get('dates.json') === d, `list hidden: the tab went hidden while the list was on its way → no follow-up request (health +${counts.get('health.json') - h}, dates +${counts.get('dates.json') - d})`);
  await context.close();
}, 'list-hidden');

// ---------- S4. one published_at in the future does not poison server time ----------
for (const withDate of [false, true]) await block(async () => {
  let bad = false, seq = 1000;
  const tag = `future ${withDate ? 'with' : 'without'} Date header`;
  const { context, page } = await open({ dist: distQuick, answer: (file) => {
    const at = Date.now() + (bad ? 3600000 : 0); seq++;
    if (file === 'health.json') return { json: health(seq, at), date: withDate ? Date.now() : null };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq, at), date: withDate ? Date.now() : null };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', `${tag}: calm`);
  bad = true; await sleep(3000); bad = false;                    // a few answers stamped one hour ahead, then normal ones again
  await sleep(withDate ? 5000 : 20000);
  await page.evaluate(() => { window.__kinds = []; });
  await sleep(9000);
  const kinds = await page.evaluate(() => window.__kinds);
  ok(await kind(page) === 'ok' && !kinds.includes('publish-late'), `${tag}: after normal answers return there is no lasting false 갱신 지연 (saw ${kinds.join(',') || 'no change'})`);
  await context.close();
}, 'future');

// ---------- S5. coming back: read first, say 확인 중, never flash 갱신 지연 from what was on screen before ----------
await block(async () => {
  let seq = 1100, slow = 0;
  const { context, page } = await open({ dist: distQuick, answer: async (file) => { seq++;
    if (file === 'health.json') return { delay: slow, json: health(seq, Date.now()), date: Date.now() };
    if (file === `game/${LIVE}.json`) return { delay: slow, json: top(liveBody(fx.live, '849843:11:2'), seq, Date.now()), date: Date.now() };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'resume: calm');
  await setHidden(page, true); await sleep(11000);               // in the background for longer than the (test) late threshold; the publisher goes on
  await page.evaluate(() => { window.__kinds = []; window.__chips = []; });
  slow = 1500;                                                    // the first answers after coming back take a moment
  await setHidden(page, false); await sleep(600);
  ok(await kind(page) === 'checking' && /확인 중/.test(await statusWords(page)), 'resume: back in view → the line says 확인 중 while the first read is on its way');
  slow = 0; await sleep(4000);
  const kinds = await page.evaluate(() => window.__kinds), chips = await page.evaluate(() => window.__chips);
  ok(!kinds.includes('publish-late') && !chips.some((c) => /갱신 지연/.test(c)), `resume: 갱신 지연 never shown from the old data (line: ${kinds.join(',')}; chip: ${chips.join(',')})`);
  ok(await kind(page) === 'ok', 'resume: calm again after the fresh answer');
  await context.close();
}, 'resume');
await block(async () => {  // a device sleep: the wall clock jumps, the page's monotonic clock does not
  let seq = 1200, stop = false, at = Date.now();
  const { context, page } = await open({ dist: distQuick, answer: (file) => { if (!stop) { seq++; at = Date.now(); }
    if (file === 'health.json') return { json: health(seq, at), date: Date.now() };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq, at), date: Date.now() };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'sleep: calm');
  await page.evaluate(() => { window.__kinds = []; window.__perfLag += 600000; });   // ten minutes pass for the world, none for performance.now()
  await sleep(7000);
  const kinds = await page.evaluate(() => window.__kinds);
  ok(await kind(page) === 'ok' && !kinds.includes('publish-late'), `sleep: after a 10 min sleep gap, fresh answers re-anchor time; no false 갱신 지연 (saw ${kinds.join(',') || 'no change'})`);
  stop = true;
  ok(await waitKind(page, 'publish-late', 16000), 'sleep: and a publisher that then stops is still noticed (time re-anchored from the answers)');
  await context.close();
}, 'sleep');

// ---------- S6. a game in play with fresh health but its own file behind is late; health itself stale is late ----------
await block(async () => {
  const t0 = FILE_AT; let healthAt = t0;
  const { context, page } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(1300 + Math.round((healthAt - t0) / 1000), healthAt, [{ game_pk: LIVE, status: 503, code: 'feed' }]), date: healthAt + 1000 };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), 1300, t0), date: healthAt + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'in play: calm');
  healthAt = t0 + 80000;
  ok(await waitKind(page, 'publish-late'), 'in play: health.json fresh, this game file 80 s behind, status in progress → 갱신 지연');
  await context.close();
}, 'in-play');

// ---------- S7. a game file that never exists: bounded backoff, and no asking at all once no followed list names the game ----------
await block(async () => {
  const stamps = [];
  const listed = JSON.parse(readFileSync(path.join(files, 'games/2026-09-30.json'), 'utf8')); listed.games.push({ ...listed.games[1], game_pk: 557 });
  let inList = true;
  const { context, page, counts } = await open({ dist: distQuick, answer: (file) => {
    if (file === 'game/557.json') { stamps.push(Date.now()); return { status: 404 }; }
    if (file === 'games/2026-09-30.json') return inList ? { json: listed } : undefined;
  } });
  await page.goto('http://pz.test/pub/?game=557');
  await page.locator('.sv-unavailable').waitFor();
  await sleep(17000);
  const gaps = stamps.slice(1).map((t, i) => Math.round((t - stamps[i]) / 100) / 10);
  ok(gaps.length >= 3 && gaps.at(-1) >= 3.5 && gaps.at(-1) <= 6 && gaps[0] < gaps.at(-1), `never there: listed but missing → asked again with growing, bounded gaps (${gaps.join(' s, ')} s; test config 1 s → max 4 s)`);
  ok((counts.get('health.json') ?? 0) >= stamps.length, 'never there: health-first checks also precede missing-game attempts');
  ok(/지금 볼 수 없어요/.test(await page.locator('.sv-unavailable').innerText()), 'never there: the calm 지금 볼 수 없어요 state stays meanwhile');
  inList = false; await sleep(6000);
  const n = stamps.length; await sleep(9000);
  ok(stamps.length === n, `never there: once no followed list names the game, asking stops (${stamps.length - n} more requests in 9 s)`);
  ok(await page.getByRole('button', { name: '다시 확인' }).count() === 1, 'never there: the manual 다시 확인 stays');
  await context.close();
}, 'never-there');

// ---------- S8. an older health.json arriving after a newer one (a cache serving an old copy) ----------
await block(async () => {
  const t0 = FILE_AT; let old = false;
  const { context, page } = await open({ answer: (file) => {
    if (file === 'health.json') return old ? { json: health(1390, t0 - 600000), date: t0 + 1000 } : { json: health(1400, t0), date: t0 + 1000 };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), 1400, t0), date: t0 + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2') && await kind(page) === 'ok', 'health inversion: calm');
  old = true; await page.evaluate(() => { window.__kinds = []; }); await sleep(6500);
  const kinds = await page.evaluate(() => window.__kinds);
  ok(await kind(page) === 'ok' && kinds.every((k) => k === 'ok'), 'health inversion: a lower-sequence health.json after a higher one is discarded (requests are single-flight, so answers cannot cross on the wire)');
  await context.close();
}, 'health-inversion');

// ====================================================================================================================
// Final review round (2026-10-11): malformed files never blank the page, never leak raw errors, never show the wrong game
// ====================================================================================================================
const listFile = () => JSON.parse(readFileSync(path.join(files, 'games/2026-09-30.json'), 'utf8'));
// ---------- T1. a start time that is not a time; any render exception ends in a plain fallback, not a blank page ----------
await block(async () => {
  const list = listFile(); list.games[0] = { ...list.games[0], start: 'soon' }; list.games[1] = { ...list.games[1], start: { at: 1 } };
  const badStart = liveBody(fx.live, '849843:11:2'); badStart.game = { ...badStart.game, start: 'not a time' };
  const broken = liveBody(fx.live, '849843:11:2'); broken.game = { ...broken.game, away_team: 12345 };   // makes the screen's own code throw while drawing
  let which = 'badStart';
  const { context, page, errors } = await open({ width: 375, height: 553, answer: (file) => {
    if (file === 'games/2026-09-30.json') return { json: list };
    if (file === 'health.json') return { json: health(1500, FILE_AT) };
    if (file === `game/${LIVE}.json`) return { json: top(which === 'badStart' ? badStart : broken, which === 'badStart' ? 1500 : 1501, FILE_AT) };
  } });
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  ok(await page.locator('.sv-list .sv-card').first().waitFor({ timeout: 6000 }).then(() => true, () => false) && await page.locator('.sv-list .sv-card').count() === 2, 'bad start: the list still shows both games');
  ok((await page.locator('.sv-list').innerText()).split('시간 미정').length === 3, 'bad start: each unreadable start time reads 시간 미정');
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2', 6000) && /시간 미정/.test(await page.locator('.sv-hd-score').getAttribute('aria-label')), 'bad start: the game screen is drawn, start time 시간 미정');
  ok(errors.length === 0, `bad start: no page errors ${errors.join(' | ')}`);
  which = 'broken';
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await page.locator('.sv-crash').waitFor({ timeout: 6000 }).then(() => true, () => false), 'boundary: a render exception shows the fallback instead of a blank page');
  const text = await page.locator('body').innerText();
  ok(/화면을 표시하지 못했어요/.test(text) && await page.getByRole('button', { name: '새로 고침' }).count() === 1 && !/TypeError|undefined|is not a function|split/.test(text), 'boundary: plain Korean words and a reload control, no raw error text');
  await context.close();
}, 'bad-start');
// ---------- T2. null entries, non-numeric and duplicate game numbers in a list ----------
await block(async () => {
  const list = listFile();
  list.games = [null, list.games[0], 'x', { ...list.games[1], game_pk: 'abc' }, { ...list.games[1], game_pk: -4 }, { ...list.games[1], game_pk: 12.5 }, list.games[1], { ...list.games[1] }, 7];
  let shape = 'mixed';
  const { context, page, errors } = await open({ answer: (file) => file === 'games/2026-09-30.json' ? { json: shape === 'mixed' ? list : { ...list, games: 'nope' } } : undefined });
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  ok(await page.locator('.sv-list .sv-card').first().waitFor({ timeout: 6000 }).then(() => true, () => false), 'bad list: the list is drawn');
  const hrefs = await page.locator('.sv-list .sv-card').evaluateAll((els) => els.map((e) => e.getAttribute('href')));
  ok(hrefs.length === 2 && new Set(hrefs).size === 2 && hrefs.every((h) => /^\.\/\?game=[1-9]\d*$/.test(h)), `bad list: only cards with a positive integer game number, each once (${hrefs.join(' ')})`);
  ok(await page.locator('.sv-alert').count() === 0 && errors.length === 0, `bad list: no error box, no page errors ${errors.join(' | ')}`);
  shape = 'broken';
  await page.goto('http://pz.test/pub/?date=2026-09-30');
  await page.locator('.sv-alert').waitFor({ timeout: 6000 }).catch(() => {});
  const alert = await page.locator('.sv-alert').count() ? await page.locator('.sv-alert').innerText() : await page.locator('.sv-list').innerText();
  ok(!/TypeError|is not a function|undefined|Cannot read|\.map|null/.test(alert), `bad list: a list that is not a list never shows raw exception text ("${alert.replace(/\s+/g, ' ').slice(0, 60)}")`);
  await context.close();
}, 'bad-list');
// ---------- T3. a game file whose body is another game ----------
await block(async () => {
  const other = liveBody(fx.live, '849843:11:2');   // its game.game_pk is 990003
  let wrong = true;
  const { context, page } = await open({ answer: (file) => {
    if (file === 'game/558.json') return { json: top(wrong ? other : { ...other, game: { ...other.game, game_pk: 558 } }, wrong ? 1 : 2, FILE_AT) };
    if (file === `game/${LIVE}.json`) return { json: top(wrong && mid ? { ...liveBody(fx.live_more, '849843:12:2'), game: { ...other.game, game_pk: 558 } } : other, mid ? 1601 : 1600, FILE_AT), date: FILE_AT + 1000 };
    if (file === 'health.json') return { json: health(1600, FILE_AT), date: FILE_AT + 1000 };
  } });
  let mid = false;
  await page.goto('http://pz.test/pub/?game=558');
  ok(await page.locator('.sv-unavailable').waitFor({ timeout: 6000 }).then(() => true, () => false) && await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0, 'wrong game: a file for another game is not drawn on first load (지금 볼 수 없어요)');
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'wrong game: the right file loads');
  mid = true; await sleep(5000);
  ok(await page.locator('.sv-state').getAttribute('data-pitch') === '849843:11:2', 'wrong game: a later answer for another game is not applied over the game on screen');
  await context.close();
}, 'wrong-game');
// ---------- T4. a future published_at with a readable Date cannot hide a stopped publisher; a non-integer sequence is not applied ----------
await block(async () => {
  let seq = 1700, stop = false, frozen = null;
  const { context, page } = await open({ dist: distQuick, answer: (file) => {
    if (!stop) { seq++; frozen = Date.now() + 3600000; }           // every publish is stamped one hour ahead (publisher clock wrong)
    if (file === 'health.json') return { json: health(seq, frozen), date: Date.now() };
    if (file === `game/${LIVE}.json`) return { json: top(liveBody(fx.live, '849843:11:2'), seq, frozen), date: Date.now() };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'future+host: loaded');
  await sleep(4000);
  ok(await kind(page) === 'ok', 'future+host: publishes keep coming (stamped ahead) → calm');
  stop = true;
  ok(await waitKind(page, 'publish-late', 20000), 'future+host: the publisher stops → 갱신 지연 still appears (time since the last new publish was seen)');
  stop = false;
  ok(await waitKind(page, 'ok'), 'future+host: clears when publishes resume');
  await context.close();
}, 'future-host');
await block(async () => {
  let mode = 'good';
  const { context, page } = await open({ answer: (file) => {
    if (file === 'health.json') return { json: health(1800, FILE_AT), date: FILE_AT + 1000 };
    if (file === 'game/559.json') { const b = top({ ...liveBody(fx.live, '849843:11:2'), game: { ...liveBody(fx.live, '849843:11:2').game, game_pk: 559 } }, 1, FILE_AT); b.publish.sequence = '1'; return { json: b }; }
    if (file !== `game/${LIVE}.json`) return undefined;
    if (mode === 'good') return { json: top(liveBody(fx.live, '849843:11:2'), 1800, FILE_AT), date: FILE_AT + 1000 };
    const b = top(liveBody(fx.live_more, '849843:12:2'), 0, FILE_AT); b.publish.sequence = mode === 'string' ? '5' : 12.5; return { json: b, date: FILE_AT + 1000 };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'sequence type: loaded with an integer sequence');
  for (const m of ['string', 'fraction']) {
    mode = m; await sleep(4500);
    ok(await page.locator('.sv-state').getAttribute('data-pitch') === '849843:11:2', `sequence type: a file whose sequence is a ${m} is not applied (it cannot be ordered)`);
  }
  await page.goto('http://pz.test/pub/?game=559'); await sleep(2500);
  ok(await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && await page.locator('.sv-zone').count() === 0, 'sequence type: on first load such a file is not drawn either');
  await context.close();
}, 'sequence-type');
// ---------- T5. ?game= is decimal digits only ----------
await block(async () => {
  const { context, page, asks } = await open({});
  for (const g of ['1e3', '0x10', ' 849843 ', '849843.0', '+849843', '８４９８４３']) { await page.goto(`http://pz.test/pub/?game=${encodeURIComponent(g)}`); await page.locator('.sv-datebar h1').waitFor(); await sleep(500); }
  const games = dataAsks(asks).filter((a) => a.includes('/v1/game/'));
  ok(games.length === 0, `game number: anything but plain decimal digits opens the list and asks for no game file (${games.join(' ') || 'none asked'})`);
  await context.close();
}, 'game-digits');

// ====================================================================================================================
// After the real-file check (2026-10-12)
// ====================================================================================================================
// ---------- U1. a screen pinned on a past pitch keeps reading (slowly): a pitch turned back to 확인 중 is blanked there too ----------
await block(async () => {
  let mode = 'normal', seq = 2000;
  const body = () => { const b = liveBody(fx.live, '849843:11:2'); seq++;
    if (mode === 'unsure' || mode === 'final') { const p = b.pitches.at(-1); p.receipt = { status: 'unconfirmed', reason: 'incomplete_identity', corrected: false };
      p.actual = { ...Object.fromEntries(Object.keys(p.actual).map((k) => [k, null])), result_label: '확인 중', call_label: '확인 중' }; }
    if (mode === 'final') { b.game = { ...b.game, status: 'final', status_text: 'Final', live: false }; b.current = null; b.next = null; b.display_state = display(null, { state: 'final', label: '경기 종료', message: '경기가 끝났어요.', shown: false }); }
    return top(b, seq, Date.now(), mode === 'final'); };
  const { context, page, counts, errors } = await open({ dist: distQuick, answer: (file) => {
    if (file === 'health.json') return { json: health(seq, Date.now()), date: Date.now() };
    if (file === `game/${LIVE}.json`) return { json: body(), date: Date.now() };
  } });
  await page.goto(`http://pz.test/pub/?game=${LIVE}`);
  ok(await waitPitch(page, '849843:11:2'), 'pinned: loaded, following the latest');
  // Since the 2026-10-13 redesign (decisions 2026-10-13) a past pitch of a live game opens in a detail sheet while the live stage
  // behind it keeps following the latest (LIVE stays on), so the file keeps being read at the live cadence. What this scenario
  // protects is unchanged: the open past pitch is redrawn from every new file, blanked when the server turns it back to 확인 중,
  // and the view stays on that pitch (no jump to another one).
  await openAtBat(page);
  const cards = page.locator('.sv-pitch-row');
  await cards.first().click();                                       // the viewer opens the last thrown pitch
  const sheet = page.getByRole('dialog'); await sheet.waitFor();
  const label = fx.live.pitches.at(-1).actual.pitch_label, title = await sheet.locator('h2').innerText();
  ok(await sheet.locator('.sv-detail-zone .sv-actual').count() === 1 && (await sheet.innerText()).includes(label) && /km\/h|mph/.test(await sheet.innerText()),
    `pinned: the past pitch shows its type and location (${label}, in its detail sheet)`);
  ok(await page.locator('.sv-hd-status[data-tone="live"]').count() === 1 && await page.locator('.sv-pitch-summary .sv-reclist:not(.empty) .sv-rec-cell').count() === 3, 'pinned: the live stage behind the sheet keeps the latest recommendation and LIVE');
  const n0 = counts.get(`game/${LIVE}.json`); await sleep(4000);
  const pinnedReads = counts.get(`game/${LIVE}.json`) - n0;
  ok(pinnedReads >= 2 && pinnedReads <= 4, `pinned: the game file is still read while a past pitch is open (${pinnedReads} reads in 4 s; live poll 2 s)`);
  mode = 'unsure';
  ok(await sheet.getByText('이 공을 다시 확인하고 있어요', { exact: false }).waitFor({ timeout: 6000 }).then(() => true, () => false), 'pinned: the server turns that pitch back to 확인 중 → the open view says 확인 중');
  ok(await sheet.locator('.sv-detail-zone, .sv-actual, .sv-table, .sv-rec').count() === 0 && !(await sheet.innerText()).includes(label) && !/km\/h|mph/.test(await sheet.innerText())
    && await page.locator('.sv-pitch-row.unconfirmed').count() === 1 && await page.locator('.sv-zone-wrap .sv-actual').count() === 0, 'pinned: its location, type and paired recommendation are gone (fail closed, sheet and stage)');
  ok(await sheet.count() === 1 && await sheet.locator('h2').innerText() === title, `pinned: the view stays on that pitch (no jump to another one: ${title})`);
  mode = 'final'; await sleep(4000);
  const n1 = counts.get(`game/${LIVE}.json`); await sleep(5000);
  ok(counts.get(`game/${LIVE}.json`) > n1, `pinned: final legacy files keep checking for corrections (${counts.get(`game/${LIVE}.json`) - n1} more in 5 s)`);
  ok(errors.length === 0, `pinned: no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'pinned');
// ---------- U2. a plate appearance with no pitch (first/last index null): its own result, no jump ----------
await block(async () => {
  const b = asGame(560, clone(fx.final)); const model = b.pas[0];
  b.pas.splice(1, 0, { ...model, pa_key: '849843:999', at_bat_number: 999, first_index: null, last_index: null, pitch_count: 0, complete: true, result_text: '고의4구', event_label: '고의4구', batter: { ...model.batter, name: 'Zero Pitch' } });
  delete b.pas[1].batting_order;
  const { context, page, errors } = await open({ width: 375, height: 553, answer: (file) => file === 'game/560.json' ? { json: top(b, 5, FILE_AT, true) } : undefined });
  await page.goto('http://pz.test/pub/?game=560');
  await page.locator('.sv-zone').first().waitFor();
  await openAtBat(page);
  const before = { inning: await page.locator('.sv-inning').innerText(), cards: await page.locator('.sv-pitch-row').count(), duel: await page.locator('.sv-actual-line').innerText() };
  await page.locator('.sv-inning').click(); await page.locator('.sv-sheet select').selectOption(`${model.inning}:${model.half}`);
  await page.locator('.sv-pa-item', { hasText: 'Zero Pitch' }).click();
  const sheet = page.getByRole('dialog');
  ok(await sheet.getByText('투구 기록이 없어요', { exact: false }).waitFor({ timeout: 4000 }).then(() => true, () => false), 'no-pitch PA: a plain 투구 기록이 없어요 line');
  ok(/Zero Pitch/.test(await sheet.innerText()) && /고의4구/.test(await sheet.innerText()), "no-pitch PA: that plate appearance's own batter and result");
  ok(!/undefined|null|NaN/.test(await sheet.innerText()), 'no-pitch PA: no leaked value');
  await sheet.getByRole('button', { name: '닫기' }).click();
  ok(await page.locator('.sv-inning').innerText() === before.inning && await page.locator('.sv-pitch-row').count() === before.cards && await page.locator('.sv-actual-line').innerText() === before.duel, 'no-pitch PA: the screen behind does not jump to another plate appearance');
  ok(await keyOnFirstScreen(page) && errors.length === 0, `no-pitch PA: key info on the first screen at 375x553, no page errors ${errors.join(' | ')}`);
  await context.close();
}, 'no-pitch-pa');

// ---------- V. polling-reduction protocol, real public defaults and deterministic browser time ----------
// Clock advances execute the actual screen timers; all request counts exclude initial page/data loading.
async function advance(page, ms) {
  for (let left = ms; left > 0; left -= 1000) { await page.clock.runFor(Math.min(left, 1000)); await sleep(40); }
}
const manifestBody = (period, seq = 120) => top({ live_dates: ['2026-09-30'], publish_interval_s: period }, seq, FILE_AT);
async function protocolScreen({ checks = true, period = 10, final = false, pregame = false, dist = distPublic, list = false, etag = null } = {}) {
  const state = { checks, period, fileSequence: 1, bodySequence: 1, feedOk: true, failures: [], fileMode: 'ok', final, checkedAt: null,
    healthAt: null, healthMode: 'ok', pitchScore: null, key: '849843:11:2', malformed: undefined, healthSequence: 10000, manifestSequence: 120, etag, gameHeaders: [], pregame, start: FILE_AT + 3600000 };
  let h;
  h = await open({ dist, protocol: true, clock: true, answer: async (file, n, headers) => {
    const now = await h.page.evaluate(() => Date.now());
    if (file === 'manifest.json') return { json: manifestBody(state.period, state.manifestSequence), date: now };
    if (file === 'health.json') {
      if (state.healthMode === 'abort') return { abort: true };
      if (state.healthMode === 'hide') { state.healthMode = 'ok'; await setHidden(h.page, true); }
      const json = health(++state.healthSequence, state.healthAt ?? now);
      if (state.checks) json.game_checks = { [LIVE]: state.malformed !== undefined ? state.malformed : {
        checked_at: iso(state.checkedAt ?? now), sequence_checked: 100000, file_sequence: state.fileSequence, feed_ok: state.feedOk } };
      json.feed_failing_games = state.failures;
      return { json, date: now };
    }
    if (file === `game/${LIVE}.json`) {
      state.gameHeaders.push(headers['if-none-match']);
      if (state.fileMode === 'notice') return { status: 503, json: { error: 'service_notice', notice: { id: 'data_hold' } } };
      if (state.fileMode === '304') return { status: 304, etag: state.etag, date: now };
      if (state.fileMode === 'abort') return { abort: true };
      if (state.fileMode === '404') return { status: 404 };
      if (state.etag && headers['if-none-match'] === state.etag) return { status: 304, etag: state.etag, date: now };
      const body = state.final ? { ...asGame(LIVE, fx.final), game: { ...fx.final.game, game_pk: LIVE, away_score: state.bodySequence } } : liveBody(state.bodySequence > 1 ? fx.live_more : fx.live, state.key);
      if (state.final && state.pitchScore !== null) body.pitches = body.pitches.map((p, i) => i === 0
        ? { ...p, situation_before: { ...p.situation_before, away_score: state.pitchScore } } : p);
      if (state.pregame) { body.game = { ...body.game, status: 'scheduled', live: false, start: iso(state.start) }; body.next = null; body.current = null;
        body.display_state = display(null, { shown: false, state: 'pregame', label: '경기 전' }); }
      return { json: top(body, state.bodySequence, FILE_AT, state.final), date: now, etag: state.etag };
    }
  } });
  await h.page.goto(list ? 'http://pz.test/pub/?date=2026-09-30' : `http://pz.test/pub/?game=${LIVE}`);
  await h.page.locator(list ? '.sv-list .sv-card' : '.sv-zone').first().waitFor();
  await sleep(100);
  return { ...h, state };
}
const count = (h, file) => h.counts.get(file) ?? 0;

await block(async () => {
  // Build the actual requested baseline screen/adapter sources, without checking out or mutating this worktree.
  const baseline = path.join(here, '../dist-public-baseline');
  const sourceFiles = new Set(['GameScreen.tsx', 'WatchList.tsx', 'api.ts', 'publish.ts']);
  await viteBuild({ root: path.join(here, '..'), mode: 'public', logLevel: 'error', build: { outDir: baseline },
    define: { 'import.meta.env.VITE_PUBLISH_POLL_MS': JSON.stringify('2000'), 'import.meta.env.VITE_PUBLISH_DATES_POLL_MS': JSON.stringify('10000') },
    plugins: [{ name: 'baseline-screen', enforce: 'pre', load(id) {
      if (id.startsWith(path.join(here, '../src/service/')) && sourceFiles.has(path.basename(id)))
        return execFileSync('git', ['show', `ef4e4df:apps/observer/web/src/service/${path.basename(id)}`], { cwd: here, encoding: 'utf8' });
    } }] });
  for (const scenario of [
    { name: 'baseline ef4e4df', dist: baseline, checks: false, expected: [30, 30, 0] },
    { name: 'old-style files at P=10', checks: false, expected: [6, 6, 1] },
    { name: 'game_checks unchanged at P=10 (60s notice revalidation)', checks: true, expected: [6, 1, 1] },
  ]) {
    const h = await protocolScreen(scenario); const paths = ['health.json', `game/${LIVE}.json`, 'manifest.json'];
    const before = paths.map(f => count(h, f)); const allBefore = dataAsks(h.asks).length;
    await advance(h.page, 60000);
    const got = paths.map((f, i) => count(h, f) - before[i]); const total = dataAsks(h.asks).length - allBefore;
    ok(got.every((n, i) => n === scenario.expected[i]) && total === scenario.expected.reduce((a, b) => a + b),
      `requests/min ${scenario.name}: health/game/manifest=${got.join('/')} total=${total} (expected ${scenario.expected.join('/')})`);
    await h.context.close();
  }
  const changing = await protocolScreen(); const beforeChanging = dataAsks(changing.asks).length;
  for (let tick = 0; tick < 6; tick++) {
    if (tick % 2 === 0) { changing.state.fileSequence++; changing.state.bodySequence++; }
    await advance(changing.page, 10000);
  }
  const changedTotal = dataAsks(changing.asks).length - beforeChanging;
  ok(changedTotal === 10 && count(changing, `game/${LIVE}.json`) === 4, `requests/min three content changes: ${changedTotal} total, expected 7+3=10`);
  await changing.context.close();
  const h = await protocolScreen({ list: true }); const files = ['health.json', 'games/2026-09-30.json', 'dates.json', 'manifest.json'];
  const before = files.map(f => count(h, f)); await advance(h.page, 60000);
  const got = files.map((f, i) => count(h, f) - before[i]);
  ok(got.join('/') === '6/6/1/1', `requests/min list: health/list/dates/manifest=${got.join('/')} total=${got.reduce((a, b) => a + b)} (expected 14)`);
  await h.context.close();
}, 'request-counts');

await block(async () => {
  const h = await protocolScreen(); const { page, state } = h;
  await advance(page, 60000);
  ok(count(h, `game/${LIVE}.json`) === 2 && await kind(page) === 'ok', 'checks: unchanged old file stays fresh with one 60s notice revalidation');
  state.fileSequence = 2; state.bodySequence = 3; state.key = '849843:12:2';
  await advance(page, 10000);
  ok(await waitPitch(page, state.key) && count(h, `game/${LIVE}.json`) === 3, 'checks: advance fetches once and accepts body ahead of announced sequence');
  state.fileSequence = 3; await advance(page, 10000);
  ok(count(h, `game/${LIVE}.json`) === 3, 'checks: body already ahead means no redundant fetch');
  state.fileSequence = 4; state.bodySequence = 2; await advance(page, 10000);
  ok(await kind(page) === 'checking' && await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && await page.locator('.sv-zone-wrap .sv-actual').count() === 0,
    'checks: announced but backwards body → 확인 중, no old recommendation or actual marker');
  ok(await page.locator('.sv-state').getAttribute('data-pitch') === '849843:12:2', 'checks: backwards body cannot replace applied state');
  state.fileMode = 'abort'; const n = count(h, `game/${LIVE}.json`);
  await advance(page, 20000);
  ok(await kind(page) === 'checking' && count(h, `game/${LIVE}.json`) === n + 2, 'checks: failed target retried every P and stays neutral below T');
  await advance(page, 15000);
  ok(await kind(page) === 'publish-late' && /갱신 지연/.test(await page.locator('.sv-hd-status').innerText()) && await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0,
    'checks: pending longer than T → 갱신 지연, old recommendation remains hidden');
  state.fileMode = 'ok'; state.bodySequence = 4; state.key = '849843:13:1'; await advance(page, 10000);
  ok(await waitPitch(page, state.key) && await kind(page) === 'ok', 'checks: same target succeeds later and restores current content');
  state.checkedAt = FILE_AT; await advance(page, 10000);
  ok(await kind(page) === 'publish-late', 'checks: fresh health cannot hide a stale checked_at');
  state.checkedAt = null; state.healthAt = FILE_AT; await advance(page, 10000);
  ok(await kind(page) === 'publish-late', 'checks: fresh checked_at cannot hide stale health');
  state.healthAt = null; await advance(page, 10000);
  ok(await kind(page) === 'ok', 'checks: fresh health and check recover');
  const f = count(h, `game/${LIVE}.json`); state.feedOk = false; await advance(page, 10000);
  ok(await kind(page) === 'client' && /연결 끊김/.test(await page.locator('.sv-hd-status').innerText()), 'checks: feed_ok:false immediately uses disconnected copy');
  await advance(page, 10000);
  ok(await kind(page) === 'client' && count(h, `game/${LIVE}.json`) === f, 'checks: fresh health does not clear feed failure or fetch an unchanged file');
  state.feedOk = true; state.failures = [LIVE]; await advance(page, 10000);
  ok(await kind(page) === 'client', 'checks: feed_failing_games overrides feed_ok:true');
  state.failures = [{ game_pk: LIVE, status: 503 }]; await advance(page, 10000);
  ok(await kind(page) === 'client', 'checks: object entries in feed_failing_games also fail this game');
  state.failures = [LIVE + 1]; await advance(page, 10000);
  ok(await kind(page) === 'ok', 'checks: another game failure does not affect this game');
  const beforeWithdrawal = count(h, `game/${LIVE}.json`);
  state.fileSequence = null; await advance(page, 10000);
  ok(await page.locator('.sv-unavailable').count() === 1 && await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && count(h, `game/${LIVE}.json`) === beforeWithdrawal,
    'checks: withdrawal clears the game without fetching its old file');
  state.fileSequence = 5; state.bodySequence = 5; await advance(page, 10000);
  ok(await waitPitch(page, state.key), 'checks: withdrawn game returns automatically when republished');
  ok(h.errors.length === 0, `checks: no page errors ${h.errors.join(' | ')}`);
  await h.context.close();
}, 'game-checks');

await block(async () => {
  const h = await protocolScreen({ final: true }); const { page, state } = h;
  state.healthAt = FILE_AT; state.checkedAt = FILE_AT; await advance(page, 120000);
  ok(await page.locator('.sv-hd-status').count() === 0 && count(h, `game/${LIVE}.json`) === 3, 'final checks: no time warnings; two 60s notice revalidations in 120s');
  await openAtBat(page);
  const selected = page.locator('.sv-pitch-row[aria-pressed="true"]');
  const selectedLabel = await selected.getAttribute('aria-label');
  const score = page.locator('.sv-hd-score strong').first();
  ok(await score.innerText() === '1', 'final checks: the header shows the final score while the first pitch stays selected');
  state.fileSequence = 2; state.bodySequence = 2; await advance(page, 10000);
  ok(count(h, `game/${LIVE}.json`) === 4 && await score.innerText() === '2' && await selected.getAttribute('aria-label') === selectedLabel, 'final checks: the final total updates without changing the selected pitch');
  // UI-1026 keeps the final total in the header even while the selected pitch's historical context changes.
  state.fileSequence = 3; state.bodySequence = 3; state.pitchScore = 2; await advance(page, 10000);
  ok(count(h, `game/${LIVE}.json`) === 5 && await score.innerText() === '3' && await selected.getAttribute('aria-label') === selectedLabel,
    'final checks: higher file_sequence keeps final header and selected pitch independently');
  state.feedOk = false; await advance(page, 10000);
  ok(await kind(page) === 'client', 'final checks: explicit failure remains visible');
  state.fileSequence = null; await advance(page, 10000);
  ok(await page.locator('.sv-unavailable').count() === 1, 'final checks: withdrawal still clears the screen');
  await h.context.close();
}, 'final-checks');

await block(async () => {
  const h = await protocolScreen(); const { page, state } = h;
  for (const bad of [null, {}, [], { checked_at: 'bad', sequence_checked: 10, file_sequence: 1, feed_ok: true },
    { checked_at: iso(FILE_AT), sequence_checked: 10, file_sequence: 1, feed_ok: null }]) {
    state.malformed = bad; const n = count(h, `game/${LIVE}.json`); await advance(page, 10000);
    ok(count(h, `game/${LIVE}.json`) === n + 1 && await kind(page) !== 'ok' && await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0,
      `malformed check: full read, never healthy (${JSON.stringify(bad)})`);
  }
  state.checks = false; await advance(page, 10000);
  ok(await kind(page) === 'publish-late', 'fallback: absent game_checks restores legacy file-age staleness');
  await h.context.close();
}, 'malformed-checks');

await block(async () => {
  const h = await protocolScreen(); const { page } = h;
  await openAtBat(page);
  await page.locator('.sv-pitch-row').first().click();
  const g = count(h, `game/${LIVE}.json`), initial = count(h, 'health.json');
  await advance(page, 10000);
  ok(count(h, 'health.json') === initial + 1 && count(h, `game/${LIVE}.json`) === g, 'pinned checks: health uses P while the file stays unchanged');
  h.state.fileSequence = 2; h.state.fileMode = '404'; await advance(page, 10000);
  ok(await kind(page) === 'checking' && await page.locator('.sv-reclist:not(.empty) .sv-rec-cell').count() === 0 && await page.locator('.sv-zone-wrap .sv-actual').count() === 0,
    'pinned checks: a pending correction hides paired recommendation and actual pitch');
  await setHidden(page, true); const hidden = dataAsks(h.asks).length; await advance(page, 20000);
  ok(dataAsks(h.asks).length === hidden, 'pinned checks: hidden tab sends nothing');
  const before = count(h, 'health.json'); await setHidden(page, false); await sleep(200);
  ok(count(h, 'health.json') === before + 1, 'pinned checks: tab return checks health immediately, before P');
  const online = count(h, 'health.json'); await page.evaluate(() => window.dispatchEvent(new Event('online'))); await sleep(200);
  ok(count(h, 'health.json') === online + 1, 'pinned checks: online also checks immediately');
  await h.context.close();
}, 'check-wake');

await block(async () => {
  for (const period of [undefined, '10', 0, -1, 75]) {
    const h = await protocolScreen({ period }); h.state.period = period; h.state.manifestSequence++;
    // undefined uses protocolScreen's default for initial loading; apply the invalid manifest on wake.
    await h.page.evaluate(() => window.dispatchEvent(new Event('online'))); await sleep(200);
    const n = count(h, 'health.json'), m = count(h, 'manifest.json'); await advance(h.page, period === 75 ? 75000 : 10000);
    ok(count(h, 'health.json') === n + 1, `period ${String(period)}: ${period === 75 ? '75s manifest' : '10s fallback'} cadence`);
    ok(count(h, 'manifest.json') - m === (period === 75 ? 1 : 0), `period ${String(period)}: manifest reread uses max(60s,P)`);
    await h.context.close();
  }
}, 'manifest-period');

await block(async () => {
  const h = await protocolScreen({ pregame: true }); const { page, state } = h;
  state.healthAt = FILE_AT; state.checkedAt = FILE_AT; await advance(page, 60000);
  ok(await kind(page) === 'ok' && count(h, 'health.json') === 7 && count(h, `game/${LIVE}.json`) === 2,
    'pregame checks: polls at P plus 60s notice revalidation, no age warning before scheduled start');
  state.start = FILE_AT; state.fileSequence = 2; state.bodySequence = 2; await advance(page, 10000);
  ok(await kind(page) === 'publish-late', 'pregame checks: after scheduled start, stale health/check warns');
  state.healthAt = null; state.checkedAt = null; await advance(page, 10000);
  ok(await kind(page) === 'ok', 'pregame checks: fresh verification clears warning without rewriting the old file');
  state.pregame = false; state.fileSequence = 3; state.bodySequence = 3; await advance(page, 10000);
  ok(await page.locator('.sv-hd-status').innerText() === '라이브', 'pregame checks: transition to live is fetched');
  state.checkedAt = FILE_AT; await advance(page, 10000);
  ok(await kind(page) === 'publish-late', 'live checks: in-play time rule applies after transition');
  await h.context.close();
}, 'pregame-checks');

// 60s is the active-tab revalidation budget, not a maximum discovery latency: the next poll and network still apply.
await block(async () => {
  for (const final of [false, true]) {
    const h = await protocolScreen({ final, etag: '"notice-check"' }); const { page, state } = h;
    const gamePath = `game/${LIVE}.json`, label = final ? 'final' : 'live';
    await advance(page, 59000);
    ok(count(h, gamePath) === 1, `${label} notice: no unchanged-game read before 60s`);
    await advance(page, 1000);
    ok(count(h, gamePath) === 2 && state.gameHeaders[1] === state.etag, `${label} notice: 60s conditional read accepts 304`);
    await advance(page, 59000);
    ok(count(h, gamePath) === 2, `${label} notice: 304 restarts the 60s budget`);
    state.fileMode = 'notice'; await advance(page, 1000);
    ok(count(h, gamePath) === 3 && /자료를 확인하고 있어요/.test(await page.locator('.sv-alert').innerText()) && await page.locator('.sv-zone').count() === 0,
      `${label} notice: game-only notice discovered despite unchanged healthy sequence; old content hidden`);
    state.fileMode = '304'; await advance(page, 10000);
    ok(state.gameHeaders.at(-1) === undefined && await page.locator('.sv-zone').count() === 0,
      `${label} notice: held retry requires a body; even an unsolicited 304 cannot restore old content`);
    state.fileMode = 'ok'; state.etag = null; state.bodySequence = 0; await advance(page, 10000);
    ok(await page.locator('.sv-zone').count() === 0, `${label} notice: lower sequence cannot lift hold`);
    state.bodySequence = 1; await advance(page, 10000);
    ok(await page.locator('.sv-zone').count() > 0 && await page.locator('.sv-unavailable').count() === 0,
      `${label} notice: accepted current body restores content`);
    state.fileMode = 'abort'; await advance(page, 60000); const failedAt = count(h, gamePath);
    state.fileMode = 'ok'; await advance(page, 10000);
    ok(count(h, gamePath) === failedAt + 1 && await page.locator('.sv-zone').count() > 0,
      `${label} notice: failed revalidation retries at next poll, not after another minute`);
    state.bodySequence = 0; await advance(page, 60000); const oldAt = count(h, gamePath);
    state.bodySequence = 1; await advance(page, 10000);
    ok(count(h, gamePath) === oldAt + 1, `${label} notice: lower sequence does not reset verification budget`);
    await setHidden(page, true); const hiddenAt = dataAsks(h.asks).length;
    await advance(page, 120000);
    ok(dataAsks(h.asks).length === hiddenAt, `${label} notice: no requests while hidden, including past deadline`);
    await setHidden(page, false); await sleep(200);
    ok(count(h, gamePath) === oldAt + 2, `${label} notice: visible wake revalidates overdue game once`);
    state.healthMode = 'hide'; const beforeHiddenHealth = count(h, gamePath);
    await advance(page, 60000);
    ok(count(h, gamePath) === beforeHiddenHealth, `${label} notice: hiding during health skips the follow-up game request`);
    await setHidden(page, false); await sleep(200);
    ok(h.errors.length === 0, `${label} notice: no page errors`);
    await h.context.close();
  }
}, 'game-notice');

// HTTP behaviour has exact parse-count assertions in coreChecks; this exercises the rendered screen as well.
await block(async () => {
  let phase = 0; const conditional = [];
  const h = await open({ dist: distPublic, protocol: true, clock: true, answer: (file, n, headers) => {
    if (file === 'manifest.json') return { json: manifestBody(10) };
    if (file === `game/${LIVE}.json`) {
      conditional.push(headers['if-none-match']);
      if (phase === 1) return { status: 304, etag: '"a"' };
      return { json: top(liveBody(phase === 2 ? fx.live_more : fx.live, phase === 2 ? '849843:12:2' : '849843:11:2'), phase === 2 ? 2 : 1, FILE_AT), etag: phase === 2 ? '"b"' : '"a"', date: null };
    }
  } });
  await h.page.goto(`http://pz.test/pub/?game=${LIVE}`); await h.page.locator('.sv-state').waitFor();
  phase = 1; await advance(h.page, 10000);
  ok(conditional[0] === undefined && conditional[1] === '"a"' && await waitPitch(h.page, '849843:11:2'), 'ETag screen: 200 → 304 keeps validated game');
  phase = 2; await advance(h.page, 10000);
  ok(conditional[2] === '"a"' && await waitPitch(h.page, '849843:12:2'), 'ETag screen: subsequent 200 applies changed game');
  ok(h.errors.length === 0, 'ETag screen: no page errors'); await h.context.close();
}, 'etag-screen');

// ---------- integration review: withdrawal remains closed; changing P keeps the same-date list ----------
await block(async () => {
  const h = await protocolScreen(); const { page, state } = h;
  state.fileSequence = null; await advance(page, 10000);
  ok(await page.locator('.sv-unavailable').count() === 1, 'withdrawal recovery: explicit null clears the game');
  const reads = count(h, `game/${LIVE}.json`);
  for (const mode of ['abort', 'absent', 'invalid']) {
    state.healthMode = mode === 'abort' ? 'abort' : 'ok'; state.checks = mode !== 'absent';
    state.malformed = mode === 'invalid' ? {} : undefined;
    await advance(page, 10000);
    ok(count(h, `game/${LIVE}.json`) === reads && await page.locator('.sv-unavailable').count() === 1 && await page.locator('.sv-zone').count() === 0,
      `withdrawal recovery: ${mode} health never reopens a residual old game file`);
  }
  state.healthMode = 'ok'; state.checks = true; state.malformed = undefined; state.fileSequence = 2;
  await advance(page, 10000);
  ok(await page.locator('.sv-zone').count() === 0, 'withdrawal recovery: valid republication waits for its announced file sequence');
  state.bodySequence = 2; state.key = '849843:12:2'; await advance(page, 10000);
  ok(await waitPitch(page, state.key) && await page.locator('.sv-unavailable').count() === 0, 'withdrawal recovery: valid check plus matching file reopens the game');
  ok(h.errors.length === 0, 'withdrawal recovery: no page errors'); await h.context.close();
}, 'integration-review');

await block(async () => {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const h = await open({ dist: distPublic, protocol: true, clock: true, answer: async file => {
    if (file === 'manifest.json') { await gate; return { json: manifestBody(20) }; }
  } });
  await h.page.goto('http://pz.test/pub/?date=2026-09-30');
  await h.page.locator('.sv-list .sv-card').first().waitFor();
  const cards = await h.page.locator('.sv-list .sv-card').count();
  const reads = count(h, 'games/2026-09-30.json');
  await h.page.evaluate(() => {
    window.__listCleared = false;
    new MutationObserver(() => { if (!document.querySelector('.sv-list .sv-card')) window.__listCleared = true; })
      .observe(document.querySelector('.sv-list'), { subtree: true, childList: true });
  });
  const delivered = h.page.waitForResponse('**/v1/manifest.json'); release(); await delivered; await sleep(150);
  ok(await h.page.locator('.sv-list .sv-card').count() === cards && !(await h.page.evaluate(() => window.__listCleared)) && count(h, 'games/2026-09-30.json') === reads,
    'list period: late manifest changes P without blanking cards or duplicating the same-date request');
  await advance(h.page, 10000); const next = count(h, 'games/2026-09-30.json');
  await advance(h.page, 19000);
  ok(next === reads + 1 && count(h, 'games/2026-09-30.json') === next, 'list period: pending timer finishes and the next timer uses the new 20s period');
  await advance(h.page, 1000);
  ok(count(h, 'games/2026-09-30.json') === next + 1 && h.errors.length === 0, 'list period: exactly one next read, no page errors');
  await h.context.close();
}, 'integration-review');

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
  ok(await keyOnFirstScreen(page), 'api mode: key info on the first screen at 375x553');
  ok(errors.length === 0, `api mode: no page errors ${errors.join(' | ')}`);
  await context.close();
} else ok(false, `api mode: ${distApi} is missing (run npm run build)`);

await browser.close();
console.log(`${passed} passed, ${failures} failed`);
if (failures) { console.error(`${failures} check(s) failed`); process.exit(1); }
console.log('static publish smoke passed');

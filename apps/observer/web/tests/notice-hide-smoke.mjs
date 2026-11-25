// NOTICE-FIX-01: a service notice (503 `service_notice`, ids maintenance / data_hold) on a page that is ALREADY OPEN hides everything it
// received before (recommendation cards, zone recommendation marks, pitch cards, matchup, score, open sheets, list cards) and shows
// only the notice; the next good answer (200) draws the game / list again by polling alone (no reload).
// Synthetic own-loopback only: the public bundle reads tests/fixtures/publish (stamps moved to "now") from a fake host on an
// OS-assigned loopback port; the API bundle is answered by page.route on http://pz.test. No backend, no network, no 8766-8772.
//   npm run build && npm run build:public && node tests/notice-hide-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const publicDist = path.join(here, '..', 'dist-public'), apiDist = path.join(here, '..', 'dist');
const v1 = path.join(here, 'fixtures', 'publish', 'v1');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const shots = process.argv[2]; if (shots) mkdirSync(shots, { recursive: true });
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json' };
let failures = 0, passes = 0;
const ok = (cond, message) => { if (cond) { passes++; console.log('ok  ', message); } else { failures++; console.error('FAIL', message); } };
async function waited(message, wait) { try { await wait(); ok(true, message); return true; } catch (e) { ok(false, `${message} — ${String(e).split('\n')[0].slice(0, 160)}`); return false; } }

const NOTICE = {
  maintenance: { status: 503, body: { error: 'service_notice', notice: { id: 'maintenance', message: 'DO_NOT_DISPLAY_SYNTHETIC_SERVER_TEXT' } }, expect: '서비스 점검 중이에요.' },
  data_hold: { status: 503, body: { error: 'service_notice', notice: { id: 'data_hold', message: 'DO_NOT_DISPLAY_SYNTHETIC_SERVER_TEXT' } }, expect: '자료를 확인하고 있어요.' },
};
const host = { mode: 'ok' };   // what the fake host answers now: 'ok' | 'maintenance' | 'data_hold'

// ---------- public bundle: fake static host ----------
const started = Date.now();
const isoNow = () => new Date(Date.now() - 1000).toISOString().replace(/\.\d{3}Z$/, 'Z');
function freshen(body) {   // same idea as tests/ui-preview.mjs --clock now: the files look freshly published, content unchanged
  const at = isoNow(), add = Math.floor((Date.now() - started) / 10000);
  const stamp = (o) => { if (!o || typeof o !== 'object') return; if (typeof o.published_at === 'string') o.published_at = at; if (Number.isSafeInteger(o.sequence)) o.sequence += add; };
  stamp(body); stamp(body.publish);
  if (body.schema?.includes('health') && body.game_checks && typeof body.game_checks === 'object') for (const c of Object.values(body.game_checks)) {
    if (!c || typeof c !== 'object') continue;
    if (typeof c.checked_at === 'string') c.checked_at = at;
    for (const k of ['sequence_checked', 'file_sequence']) if (Number.isSafeInteger(c[k])) c[k] += add;
  }
  return body;
}
const publicAsks = [];
const server = createServer((req, res) => {
  res.setHeader('Date', new Date().toUTCString()); res.setHeader('Cache-Control', 'no-store');
  const u = new URL(req.url, 'http://x');
  const m = u.pathname.match(/\/v1(\/.*)$/);
  if (m) {
    publicAsks.push(u.pathname);
    if (host.mode !== 'ok') { const n = NOTICE[host.mode]; res.writeHead(n.status, { 'content-type': 'application/json' }); return res.end(JSON.stringify(n.body)); }
    const disk = path.resolve(v1, '.' + m[1]);
    if (!disk.startsWith(v1 + path.sep) || !existsSync(disk) || !statSync(disk).isFile()) { res.writeHead(404, { 'content-type': 'text/plain' }); return res.end('not found'); }
    res.writeHead(200, { 'content-type': 'application/json' }); return res.end(JSON.stringify(freshen(JSON.parse(readFileSync(disk, 'utf8')))));
  }
  const file = path.resolve(publicDist, '.' + u.pathname);
  const target = file.startsWith(publicDist) && existsSync(file) && statSync(file).isFile() ? file : path.join(publicDist, 'index.html');
  res.writeHead(200, { 'content-type': TYPES[path.extname(target)] ?? 'application/octet-stream' }); res.end(readFileSync(target));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
if ([8766, 8767, 8768, 8769, 8770, 8771, 8772].includes(server.address().port)) throw Error('reserved port');

// ---------- API bundle: canned answers on http://pz.test ----------
const apiBody = () => {
  const body = JSON.parse(JSON.stringify(fx.live));
  body.source = { kind: 'live', session_id: null }; body.receipts = null;
  body.display_state = { schema: 'pitcheezy-display-state-v1', state: 'recommended', reason: 'ledger_recorded', label: '다음 공 추천', message: null,
    pitch_key: '849843:200:2', recommendation_shown: true, correction: null,
    feed: { ok: true, delayed: false, lag_s: 18.2, lag_max_s: 24.0, lag_measured: true, lag_threshold_s: 45.0, thresholds_provisional: true,
      last_ok_at: '2026-10-11T01:02:03Z', failing_since: null, failing_for_s: null, received_at: '2026-10-11T01:02:03Z', provider_order: 'ordered' },
    ledger: { status: 'recorded', outcome: 'first', why: null, store_ok: true } };
  return body;
};

const browser = await chromium.launch();
async function open(mode, width, height) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  const errors = [], external = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (mode === 'public') { if (url.origin !== origin) { external.push(url.href); return route.abort('blockedbyclient'); } return route.continue(); }
    if (url.origin !== 'http://pz.test') { external.push(url.href); return route.abort('blockedbyclient'); }
    if (url.pathname.startsWith('/api/')) {
      if (host.mode !== 'ok') { const n = NOTICE[host.mode]; return route.fulfill({ status: n.status, json: n.body }); }
      if (url.pathname === '/api/service/games/849843') return route.fulfill({ json: apiBody() });
      return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
    }
    const file = path.join(apiDist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(apiDist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  return { context, page, errors, external };
}

// what must not stay on screen while a notice stands (all of it was received before the notice)
const LIVE_PARTS = ['.sv-zone .sv-rec', '.sv-rec-summary', '.sv-reclist', '.sv-rec-sec', '.sv-pitch-row', '.sv-atbat-toggle', '.sv-atbat-list', '.sv-pitch-summary', '.sv-zone', '.sv-card', '.sv-duel', '.sv-board', '.sv-head-score', '.sv-sheet', '.sv-recent-pas', '.sv-notice.line'];
const counts = (page) => page.evaluate((sels) => Object.fromEntries(sels.map((s) => [s, document.querySelectorAll(s).length])), LIVE_PARTS);
const lum = (c) => { const v = c.match(/[\d.]+/g).slice(0, 3).map(Number).map((x) => x / 255).map((x) => x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4); return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
const alertLook = (page) => page.evaluate(() => {
  const a = document.querySelector('.sv-alert'), p = a.querySelector('p'), r = a.getBoundingClientRect(), s = getComputedStyle(p);
  let bg = 'rgba(0, 0, 0, 0)', el = p; while (el && /rgba\(0, 0, 0, 0\)|transparent/.test(bg)) { bg = getComputedStyle(el).backgroundColor; el = el.parentElement; }
  return { text: a.innerText, color: s.color, bg, wordBreak: s.wordBreak, left: r.left, right: r.right, top: r.top, bottom: r.bottom, sideScroll: document.scrollingElement.scrollWidth > innerWidth + 1 };
});
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const WORDS = /맞았|틀렸|맞음|더 낫|이겼|적중|정확도|[✓✔]/;

async function noticeShown(page, tag, which, width, height, errors) {
  const n = NOTICE[which];
  if (!(await waited(`${tag} ${which}: notice shown on the open page without reload`, () => page.waitForFunction((t) => document.querySelector('.sv-alert')?.innerText.includes(t), n.expect, { timeout: 25000 })))) return;
  await page.waitForTimeout(150);
  const c = await counts(page);
  for (const s of LIVE_PARTS) ok(c[s] === 0, `${tag} ${which}: ${s} hidden (${c[s]})`);
  const look = await alertLook(page);
  ok(!look.text.includes('DO_NOT_DISPLAY'), `${tag} ${which}: host text never shown`);
  ok(!WORDS.test(await page.evaluate(() => document.body.innerText)), `${tag} ${which}: no "right/better" words or check marks`);
  ok(look.text.includes('자동으로 다시 연결하고 있어요'), `${tag} ${which}: says it keeps asking by itself`);
  ok(!look.sideScroll && look.left >= 0 && look.right <= width + 0.5 && look.bottom <= height + 0.5, `${tag} ${which}: notice fits ${width}x${height} (${Math.round(look.left)}..${Math.round(look.right)}, bottom ${Math.round(look.bottom)})`);
  ok(look.wordBreak === 'keep-all', `${tag} ${which}: Korean keep-all (${look.wordBreak})`);
  const cr = ratio(look.color, look.bg);
  ok(cr >= 4.5, `${tag} ${which}: text contrast ${cr.toFixed(2)}:1 ≥ 4.5 (${look.color} on ${look.bg})`);
  ok(errors.length === 0, `${tag} ${which}: no page errors ${errors.join(' | ')}`);
  if (shots) await page.screenshot({ path: path.join(shots, `${tag}-${which}.png`) });
}
async function gameShown(page, tag, step) {
  await waited(`${tag} ${step}: recommendation cards and zone recommendation marks shown`, async () => {
    await page.locator('.sv-zone .sv-rec').first().waitFor({ timeout: 25000 });
    await page.locator('.sv-reclist:not(.empty)').first().waitFor({ timeout: 25000 });  // UI-1026: the recommendation row (three cells)
  });
  const c = await counts(page);
  // UI-1026 folds the at-bat list behind a toggle; the pitch list counts as shown when its toggle or its rows are on screen
  ok(c['.sv-zone .sv-rec'] > 0 && c['.sv-reclist'] > 0 && c['.sv-pitch-row'] + c['.sv-atbat-toggle'] > 0, `${tag} ${step}: zone marks ${c['.sv-zone .sv-rec']}, rec rows ${c['.sv-reclist']}, pitch list ${c['.sv-pitch-row']}+${c['.sv-atbat-toggle']}`);
  ok(await page.locator('.sv-alert').count() === 0, `${tag} ${step}: no notice`);
  ok(!(await page.evaluate(() => document.body.innerText)).includes('이전:'), `${tag} ${step}: no change note naming a recommendation from before the notice`);
  ok(await page.evaluate(() => document.scrollingElement.scrollWidth <= innerWidth + 1), `${tag} ${step}: no side scroll`);
}

for (const mode of ['public', 'api']) for (const [width, height] of [[390, 844], [320, 568]]) {
  const tag = `${mode}-${width}x${height}`;
  host.mode = 'ok';
  const { context, page, errors, external } = await open(mode, width, height);
  await page.goto(mode === 'public' ? `${origin}/?game=990003` : 'http://pz.test/watch/849843');
  await gameShown(page, tag, 'before');
  await page.evaluate(() => { window.__sameDocument = true; });
  if (shots) await page.screenshot({ path: path.join(shots, `${tag}-0-before.png`) });
  // a sheet opened before the notice is closed by it (it carried received content)
  await page.locator('.sv-rec-sec .sv-state-more').first().click();  // UI-1026: 추천 자세히 보기 opens the recommendation sheet
  ok(await page.locator('.sv-sheet').count() === 1, `${tag}: detail sheet open before the notice`);
  host.mode = 'maintenance';
  await noticeShown(page, tag, 'maintenance', width, height, errors);
  host.mode = 'data_hold';
  await noticeShown(page, tag, 'data_hold', width, height, errors);
  host.mode = 'ok';
  await gameShown(page, tag, 'after');
  ok(await page.locator('.sv-sheet').count() === 0, `${tag} after: the sheet stays closed`);
  ok(await page.evaluate(() => window.__sameDocument === true), `${tag}: no reload at any step`);
  if (shots) await page.screenshot({ path: path.join(shots, `${tag}-3-after.png`) });
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  ok(external.length === 0, `${tag}: no request leaves loopback (${external.length})`);
  await context.close();
}

// ---------- public list: the open list hides its cards during a notice ----------
for (const [width, height] of [[390, 844], [320, 568]]) {
  const tag = `public-list-${width}x${height}`;
  host.mode = 'ok';
  const { context, page, errors, external } = await open('public', width, height);
  await page.goto(`${origin}/?date=2026-09-30`);
  await waited(`${tag}: list cards shown`, () => page.locator('.sv-list li').first().waitFor({ timeout: 20000 }));
  await page.evaluate(() => { window.__sameDocument = true; });
  host.mode = 'maintenance';
  if (await waited(`${tag}: notice replaces the open list`, () => page.waitForFunction(() => document.querySelector('.sv-alert')?.innerText.includes('서비스 점검 중이에요.'), null, { timeout: 25000 }))) {
    ok(await page.locator('.sv-list li').count() === 0, `${tag}: list cards hidden during the notice`);
    if (shots) await page.screenshot({ path: path.join(shots, `${tag}-maintenance.png`) });
  }
  host.mode = 'ok';
  await waited(`${tag}: list back after the notice clears`, () => page.locator('.sv-list li').first().waitFor({ timeout: 25000 }));
  ok(await page.locator('.sv-alert').count() === 0, `${tag}: no notice after it clears`);
  ok(await page.evaluate(() => window.__sameDocument === true), `${tag}: no reload`);
  ok(errors.length === 0 && external.length === 0, `${tag}: no page errors / external requests`);
  await context.close();
}
ok(publicAsks.length > 0 && publicAsks.every((p) => p.startsWith('/v1/')), `public bundle asked only /v1/ files (${publicAsks.length})`);

await browser.close();
await new Promise((r) => server.close(r));
console.log(`\n${passes} ok / ${failures} FAIL`);
process.exit(failures ? 1 : 0);

// /game state-request timeout: a request with no answer for 15 s (STATE_TIMEOUT_MS) is dropped and counts as a failed
// poll, so a hung request clears itself with no viewer action. Mid-game the last state stays on screen behind the stale
// band, which now has its own retry button. A slow answer within the limit (5 s, above the measured 4.6 s max) is not
// dropped. Canned responses on the built dist/, no server, no model. Takes about 1.5 minutes (real 15 s waits).
//   PZ_WEB_DIST=/tmp/dist PLAYWRIGHT_BROWSERS_PATH=... node tests/game-timeout-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const { states } = JSON.parse(readFileSync(path.join(here, 'fixtures', 'game-frame-states.json'), 'utf8'));
const shots = process.argv[2];
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };
const LIMIT = 15000;

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 360, height: 560 }, deviceScaleFactor: 2 });
let mode = 'hang';               // how the server treats the next state request: 'hang' | 'slow' | 'ok'
const asks = [];                 // when each state request arrived (ms)
const hung = [];                 // releases for requests held without an answer
await context.route('http://pz.test/**', async (route) => {
  const url = new URL(route.request().url());
  if (/^\/api\/live\/\d+\/state$/.test(url.pathname)) {
    asks.push(Date.now());
    if (mode === 'hang') { await new Promise((resolve) => hung.push(resolve)); return route.abort('timedout').catch(() => undefined); }
    if (mode === 'slow') await new Promise((resolve) => setTimeout(resolve, 5000));
    return route.fulfill({ json: states.pre_last_of_pa }).catch(() => undefined);
  }
  if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
  const file = path.join(dist, url.pathname);
  const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
  return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
});
const errors = [];
const dropped = [];              // when the page itself cancelled a state request
const page = await context.newPage();
page.on('pageerror', (error) => errors.push(String(error)));
page.on('requestfailed', (request) => {
  if (/\/api\/live\/\d+\/state/.test(request.url()) && /ERR_ABORTED/.test(request.failure()?.errorText ?? '')) dropped.push(Date.now());
});
const chip = (text, timeout) => page.locator('.gm-stage .gm-chip', { hasText: text }).waitFor({ timeout });
const live = () => page.locator('.gm-live').innerText();
const nextHang = async (what, timeout = 6000) => {   // until one more request is being held (bounded, so a broken build fails)
  const n = hung.length, end = Date.now() + timeout;
  while (hung.length === n) { if (Date.now() > end) { fail(`${what}: no new state request within ${timeout} ms`); return false; } await page.waitForTimeout(50); }
  return true;
};

// 1. The very first request never answers and the viewer does nothing. It is dropped at about 15 s (not before), the
//    page says the server did not answer, the next poll asks again by itself and the game appears.
let t = Date.now();
await page.goto('http://pz.test/game?game=849845');
await page.waitForTimeout(LIMIT - 2000);
if (dropped.length) fail(`a request was dropped after ${dropped[0] - t} ms, before the ${LIMIT} ms limit`);
if (asks.length !== 1) fail(`polls piled up behind a hung first request: ${asks.length} asks`);
await chip('연결 안 됨', 5000).catch(() => fail('a hung first request never turned into a failed poll'));
if (!dropped.length) fail('the hung first request was not cancelled');
else if (dropped[0] - t < LIMIT - 500) fail(`first request dropped after ${dropped[0] - t} ms`);
await page.getByText('15초 동안 답하지 않았어요', { exact: false }).waitFor({ timeout: 500 }).catch(() => fail('no timeout message'));
if (shots) await page.screenshot({ path: `${shots}/timeout-1-first.png` });
mode = 'ok';
await chip('다음 공', 4000).catch(() => fail('the poll after a timeout did not load the game'));
if (await live() !== 'LIVE') fail('header is not LIVE after the first state arrived');

// 2. Mid-game a poll hangs. Within the limit nothing changes; after it the last state stays behind the stale band
//    ("연결 끊김" in the header too) and the band's own retry button brings the page back at once.
mode = 'hang';
await nextHang('mid-game hang');
t = Date.now();
await page.locator('.gm-stale').waitFor({ timeout: LIMIT + 3000 }).catch(() => fail('a hung mid-game request never showed the stale band'));
if (Date.now() - t < LIMIT - 1000) fail(`stale band after ${Date.now() - t} ms, before the limit`);
if (await live() !== '연결 끊김') fail(`header says ${await live()} while the request timed out`);
const band = page.locator('.gm-stale');
if (!/에 받은 상황/.test(await band.innerText({ timeout: 1000 }).catch(() => ''))) fail('stale band does not say when the shown state was received');
const again = band.getByRole('button', { name: '다시 시도' });
const hasRetry = (await again.count()) > 0;
if (!hasRetry) fail('no retry button in the stale band');
const box = hasRetry ? await again.boundingBox() : null;
if (hasRetry && (!box || box.x + box.width > 360)) fail(`stale-band retry button off screen: ${JSON.stringify(box)}`);
const edge = await band.boundingBox();  // the band clips its own overflow: a button past its right edge is cut, not just off screen
if (hasRetry && box && (!edge || box.x + box.width > edge.x + edge.width)) fail(`stale-band retry button cut by the band: ${JSON.stringify({ box, edge })}`);
if (box && box.height < 44) fail(`stale-band retry button is ${box.height} px tall, under the 44 px tap size`);
if (shots) await page.screenshot({ path: `${shots}/timeout-2-stale.png` });
mode = 'ok';
t = Date.now();
if (hasRetry) await again.click({ timeout: 1000 }).catch(() => fail('stale-band retry could not be pressed'));
await band.waitFor({ state: 'detached', timeout: 1500 }).catch(() => fail('retry in the stale band did not bring the page back'));
if (!asks.some((a) => a >= t)) fail('retry in the stale band did not ask again');
if (await live() !== 'LIVE') fail('header is not LIVE after the stale-band retry');

// 3. Mid-game hang again, but this time the viewer does nothing: after the timeout the next poll recovers by itself.
mode = 'hang';
await nextHang('second mid-game hang');
await page.locator('.gm-stale').waitFor({ timeout: LIMIT + 3000 }).catch(() => fail('second hang: no stale band'));
mode = 'ok';
await page.locator('.gm-stale').waitFor({ state: 'detached', timeout: 4000 }).catch(() => fail('no automatic recovery after a timeout'));

// 4. A slow answer inside the limit (5 s) is waited for: not dropped, no failure, no stale band, polls do not pile up.
mode = 'slow';
const d = dropped.length;
t = Date.now();
await page.waitForTimeout(12000);
if (dropped.length !== d) fail(`a slow answer inside the limit was dropped (${dropped.length - d})`);
if (await page.locator('.gm-stale').count()) fail('a slow answer inside the limit showed the stale band');
const slow = asks.filter((a) => a >= t).length;
if (slow < 1 || slow > 3) fail(`expected 1–3 asks in 12 s with 5 s answers, got ${slow}`);
mode = 'ok';

hung.splice(0).forEach((release) => release());
if (errors.length) fail(`page errors: ${errors.join(' | ')}`);
await context.close();
await browser.close();
console.log(process.exitCode ? 'game timeout smoke: FAILED' : 'game timeout smoke: ok');

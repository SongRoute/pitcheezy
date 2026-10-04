// /game hung request: a state request that never answers (dropped packets, a stuck server) must not freeze the page.
// The timer's own poll still waits for it, but the retry button, the network coming back (online) and coming back to
// the page each drop the stuck request and ask again; a dropped request is not counted as a failure and leaves no
// second polling loop behind. Canned responses on the built dist/, no server, no model.
//   PZ_WEB_DIST=/tmp/dist PLAYWRIGHT_BROWSERS_PATH=... node tests/game-hang-smoke.mjs [screenshotDir]
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

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 360, height: 560 }, deviceScaleFactor: 2 });
let mode = 'hang';               // how the server treats the next state request: 'hang' | 'down' | 'ok'
const asks = [];                 // when each state request arrived (ms)
const hung = [];                 // releases for requests held without an answer
await context.route('http://pz.test/**', async (route) => {
  const url = new URL(route.request().url());
  if (/^\/api\/live\/\d+\/state$/.test(url.pathname)) {
    asks.push(Date.now());
    if (mode === 'hang') { await new Promise((resolve) => hung.push(resolve)); return route.abort('timedout').catch(() => undefined); }
    return mode === 'down' ? route.abort('internetdisconnected') : route.fulfill({ json: states.pre_last_of_pa });
  }
  if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
  const file = path.join(dist, url.pathname);
  const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
  return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
});
const errors = [];
const open = async () => {
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('http://pz.test/game?game=849845');
  return page;
};
const chip = (page, text, timeout = 9000) => page.locator('.gm-stage .gm-chip', { hasText: text }).waitFor({ timeout });
const since = (t) => asks.filter((a) => a >= t).length;
const waitHung = async (n, timeout = 5000) => {   // until n requests are being held
  const end = Date.now() + timeout;
  while (hung.length < n) { if (Date.now() > end) throw new Error(`no hung request after ${timeout} ms`); await new Promise((r) => setTimeout(r, 50)); }
};
const setHidden = (page, hidden) => page.evaluate((h) => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => h });
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (h ? 'hidden' : 'visible') });
  document.dispatchEvent(new Event('visibilitychange'));
}, hidden);

// 1. The very first request never answers. The timer does not pile up requests behind it; the network coming back asks
//    again at once and the game appears.
let page = await open();
await waitHung(1);
await page.waitForTimeout(2500);                 // more than one 2 s poll interval
if (asks.length !== 1) fail(`polls piled up behind a hung first request: ${asks.length} asks`);
await chip(page, '불러오는 중', 500);
mode = 'ok';
let t = Date.now();
await page.evaluate(() => window.dispatchEvent(new Event('online')));
await chip(page, '다음 공', 1500).catch(() => fail('online did not replace the hung first request'));
if (since(t) < 1) fail('online event did not ask again while a request was hung');
await page.close();

// 2. Failing first load, then the next automatic poll hangs: the retry button still asks at once. The dropped request is
//    not counted as a failure (실패 2번 → 3번 after one failed retry, not 4번), and a working retry loads the game.
mode = 'down';
page = await open();
await page.getByText('실패 2번', { exact: false }).waitFor({ timeout: 6000 });
mode = 'hang';
const held = hung.length;
await waitHung(held + 1, 3000);
mode = 'down';
const retry = page.getByRole('button', { name: '지금 다시 시도' });
t = Date.now();
await retry.click();
await page.getByText('실패 3번', { exact: false }).waitFor({ timeout: 1000 }).catch(() => fail('retry did nothing while a request was hung'));
if (await page.getByText('실패 4번', { exact: false }).count()) fail('the dropped hung request was counted as a failure');
if (since(t) < 1) fail('retry did not send a new request');
if (shots) await page.screenshot({ path: `${shots}/hang-1-retry.png` });
mode = 'ok';
await retry.click();
await chip(page, '다음 공', 1500).catch(() => fail('a working retry did not load the game'));

// 3. Mid-game, a poll hangs while the viewer is away. Coming back to the page asks again at once, and afterwards there is
//    still exactly one polling loop (about one ask per 2 s).
mode = 'hang';
await waitHung(hung.length + 1, 3000);
await setHidden(page, true);
mode = 'ok';
t = Date.now();
await setHidden(page, false);
await page.waitForTimeout(600);
if (since(t) < 1) fail('coming back to the page did not replace the hung request');
if (await page.locator('.gm-stage .gm-chip', { hasText: '연결 안 됨' }).count()) fail('dropping the hung request showed a connection failure');
await page.waitForTimeout(1000);
t = Date.now();
await page.waitForTimeout(6100);
const steady = since(t);
if (steady < 2 || steady > 4) fail(`after replacing a hung request expected about 3 asks in 6 s, got ${steady}`);
if (await page.locator('.gm-live').innerText() !== 'LIVE') fail('header is not LIVE after recovering');
if (shots) await page.screenshot({ path: `${shots}/hang-2-live.png` });

hung.splice(0).forEach((release) => release());
if (errors.length) fail(`page errors: ${errors.join(' | ')}`);
await context.close();
await browser.close();
console.log(process.exitCode ? 'game hang smoke: FAILED' : 'game hang smoke: ok');

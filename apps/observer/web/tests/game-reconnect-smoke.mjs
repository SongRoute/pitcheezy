// /game reconnect: a first load that cannot reach the server says so plainly and offers a retry button; coming back to
// the page (visibilitychange) or getting the network back (online) asks at once instead of waiting out the hidden-page
// poll; repeated wake-ups never start a second polling loop. Canned responses on the built dist/, no server, no model.
//   PZ_WEB_DIST=/tmp/dist PLAYWRIGHT_BROWSERS_PATH=... node tests/game-reconnect-smoke.mjs [screenshotDir]
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
const page = await context.newPage();
let down = true;                 // the server cannot be reached until this turns false
const asks = [];                 // when each state request arrived (ms)
await page.route('http://pz.test/**', async (route) => {
  const url = new URL(route.request().url());
  if (/^\/api\/live\/\d+\/state$/.test(url.pathname)) {
    asks.push(Date.now());
    if (asks.length === 1) await new Promise((resolve) => setTimeout(resolve, 1500));  // a slow first answer: still loading
    return down ? route.abort('internetdisconnected') : route.fulfill({ json: states.pre_last_of_pa });
  }
  if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
  const file = path.join(dist, url.pathname);
  const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
  return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
});
const errors = [];
page.on('pageerror', (error) => errors.push(String(error)));
const chip = (text, timeout = 9000) => page.locator('.gm-stage .gm-chip', { hasText: text }).waitFor({ timeout });
const since = (t) => asks.filter((a) => a >= t).length;
const setHidden = (hidden) => page.evaluate((h) => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => h });
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (h ? 'hidden' : 'visible') });
  document.dispatchEvent(new Event('visibilitychange'));
}, hidden);

// 1. First load cannot reach the server: plain words, automatic retries counted, a real tap target to retry now.
await page.goto('http://pz.test/legacy/game?game=849845');
await chip('불러오는 중', 1200);
if (await page.locator('.gm-live').innerText() !== '연결 중') fail('header does not say 연결 중 while the first answer is pending');
await chip('연결 안 됨');
await page.getByText('관전 서버에 연결하지 못했어요', { exact: false }).waitFor();
if (await page.locator('.gm-live').innerText() === 'LIVE') fail('header says LIVE before any answer arrived');
if (!(await page.getByRole('status').filter({ hasText: '자동으로 계속 다시 연결' }).innerText()).match(/^자동으로 계속 다시 연결하고 있어요$/)) fail('live region carries more than the steady sentence');
await page.getByText('실패 2번', { exact: false }).waitFor({ timeout: 6000 });   // the 2 s poll keeps trying by itself
const retry = page.getByRole('button', { name: '지금 다시 시도' });
const box = await retry.boundingBox();
if (!box || box.height < 44 || box.y + box.height > 560) fail(`retry button not a full tap target on screen: ${JSON.stringify(box)}`);
if (await page.evaluate(() => Math.max(document.documentElement.scrollHeight, document.body.scrollHeight) - window.innerHeight) > 0) fail('error stage scrolls');
if (shots) await page.screenshot({ path: `${shots}/reconnect-1-down.png` });

// 2. The retry button asks at once (well before the next 2 s poll) and the game appears.
down = false;
await page.waitForTimeout(150);
let t = Date.now();
const lastAuto = asks.at(-1);                    // the automatic poll that produced "실패 2번"; the next one is ~2 s later
if (t - lastAuto > 1000) fail(`retry check inconclusive: clicked ${t - lastAuto} ms after the last automatic ask`);
await retry.click();
await chip('다음 공', 1500);
const firstAsk = asks.find((a) => a >= t);
if (firstAsk === undefined || firstAsk - t > 700 || firstAsk - lastAuto >= 1500) fail(`retry button did not ask at once (${since(t)} asks, ${firstAsk - lastAuto} ms after the last automatic ask)`);
if (await page.getByRole('button', { name: '지금 다시 시도' }).count()) fail('retry button still shown after the game loaded');
if (await page.locator('.gm-live').innerText() !== 'LIVE') fail('header does not return to LIVE once the game loaded');

// 3. Hidden page polls slowly (every 10 s); coming back asks at once.
await setHidden(true);
await page.waitForTimeout(2500);                 // let a poll already scheduled at 2 s pass
t = Date.now();
await page.waitForTimeout(4000);
if (since(t) > 1) fail(`hidden page kept polling fast: ${since(t)} asks in 4 s`);
t = Date.now();
await setHidden(false);
await page.waitForTimeout(600);
if (since(t) < 1) fail('coming back to the page did not ask at once');

// 4. Network back (online event) asks at once, even while the next poll is scheduled far away.
await setHidden(true);
await page.waitForTimeout(2500);
await page.evaluate(() => Object.defineProperty(document, 'hidden', { configurable: true, get: () => false }));
t = Date.now();
await page.evaluate(() => window.dispatchEvent(new Event('online')));
await page.waitForTimeout(600);
if (since(t) < 1) fail('online event did not ask at once');

// 5. Many wake-ups do not leave extra polling loops behind: steady state is still one ask per 2 s.
await setHidden(false);
for (let i = 0; i < 6; i += 1) { await setHidden(false); await page.evaluate(() => window.dispatchEvent(new Event('online'))); }
await page.waitForTimeout(1000);
t = Date.now();
await page.waitForTimeout(6100);
const steady = since(t);
if (steady < 2 || steady > 4) fail(`after repeated wake-ups expected about 3 asks in 6 s, got ${steady}`);
if (shots) await page.screenshot({ path: `${shots}/reconnect-2-live.png` });

if (errors.length) fail(`page errors: ${errors.join(' | ')}`);
await context.close();
await browser.close();
console.log(process.exitCode ? 'game reconnect smoke: FAILED' : 'game reconnect smoke: ok');

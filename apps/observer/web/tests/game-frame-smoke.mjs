// One-screen game page (/game): nothing scrolls at phone sizes, the stage follows the game by itself, sheets open.
// Runs on the built dist/ with canned /api/live responses (tests/fixtures/game-frame-states.json): no server, no model.
//   npm run build && PLAYWRIGHT_BROWSERS_PATH=... node tests/game-frame-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, '..', 'dist');
const { states } = JSON.parse(readFileSync(path.join(here, 'fixtures', 'game-frame-states.json'), 'utf8'));
const shots = process.argv[2];
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
// iPhone 17 Safari's visible area (Playwright device descriptor); Galaxy S26 is 360 wide, its visible height is an assumption.
const SIZES = [['iphone17', 402, 681], ['galaxy-s26', 360, 620], ['short', 360, 560]];
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };

const browser = await chromium.launch();
for (const [device, width, height] of SIZES) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2 });
  const page = await context.newPage();
  let current = 'pre_last_of_pa';
  await page.route('http://pz.test/**', (route) => {
    const url = new URL(route.request().url());
    if (/^\/api\/live\/\d+\/state$/.test(url.pathname)) return route.fulfill({ json: states[current] });
    if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  const chip = (text, timeout = 9000) => page.locator('.gm-stage .gm-chip', { hasText: text }).waitFor({ timeout });
  const check = async (label) => {
    await page.waitForTimeout(450);  // let the stage's entry animation settle before measuring
    const found = await page.evaluate(() => {
      const stage = document.querySelector('.gm-stage'), bar = document.querySelector('.gm-bar').getBoundingClientRect();
      const small = [...document.querySelectorAll('.gm-page button, .gm-page a')].map((el) => [el, el.getBoundingClientRect()])
        .filter(([, r]) => r.width && (r.height < 44 || r.width < 44)).map(([el]) => el.className || el.textContent);
      return { page: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight) - window.innerHeight,
        wide: document.documentElement.scrollWidth - window.innerWidth, bar: bar.bottom - window.innerHeight,
        clipped: stage.scrollHeight - stage.clientHeight, small };
    });
    if (found.page > 0) fail(`${device} ${label}: page scrolls by ${found.page}px`);
    if (found.wide > 0) fail(`${device} ${label}: page is ${found.wide}px too wide`);
    if (found.bar > 0) fail(`${device} ${label}: buttons end ${found.bar}px below the screen`);
    if (found.clipped > 1) fail(`${device} ${label}: stage content is cut by ${found.clipped}px`);
    if (found.small.length) fail(`${device} ${label}: tap targets under 44px: ${found.small.join(', ')}`);
    if (shots) await page.screenshot({ path: `${shots}/game-${device}-${label}.png` });
  };

  await page.goto('http://pz.test/game?game=849845');
  await chip('다음 공');
  await check('1-pre');

  current = 'new_batter_after_strikeout';           // the strikeout takes the stage before the new batter's recommendation
  await chip('타석 끝');
  if (!/삼진/.test(await page.locator('.gm-hero').innerText())) fail(`${device}: strikeout not shown as the result`);
  if (!/%p/.test(await page.locator('.gm-we').innerText())) fail(`${device}: no win-probability change on the result`);
  await check('2-event');

  current = 'second_pitch';                         // a called strike: result first, then the next recommendation by itself
  await chip('방금 던진 공');
  await check('3-post');
  await chip('다음 공');
  if (!/가운데 낮게/.test(await page.locator('.gm-where').innerText())) fail(`${device}: location is not in batter-relative words`);

  current = 'third_pitch';
  await chip('방금 던진 공');
  await chip('다음 공');
  if (!/바깥쪽 낮게/.test(await page.locator('.gm-where').innerText())) fail(`${device}: low-right to a right-hander should read 바깥쪽 낮게`);
  if (await page.locator('.gm-batter circle').evaluate((c) => Number(c.getAttribute('cx')) > 185)) fail(`${device}: right-handed batter drawn on the wrong side`);
  await check('4-pre');
  await page.locator('.gm-name').click();    // pitch name → plain-words card
  await page.getByRole('dialog', { name: '슬라이더' }).getByText('옆으로 휘면서 떨어지는 공').waitFor();
  await check('5-glossary');
  await page.getByRole('button', { name: '닫기' }).click();
  await page.getByRole('button', { name: '자세히' }).click();
  if (await page.locator('.gm-table tbody tr').count() !== 3) fail(`${device}: candidates table should list three`);
  await check('6-detail');
  await page.getByRole('button', { name: '투수 오늘' }).click();
  if (!await page.locator('.gm-table tbody tr').count()) fail(`${device}: no pitch mix for the pitcher`);
  await page.getByRole('button', { name: '닫기' }).click();
  await page.getByRole('button', { name: '박스 스코어' }).click();
  await page.getByRole('button', { name: '닫기' }).click();
  await page.getByRole('button', { name: '지난 장면' }).click();
  await page.getByRole('button', { name: 'LIVE로 돌아가기' }).waitFor();
  const before = await page.locator('.gm-speed').innerText();
  await page.locator('.gm-speed').click();          // km/h ↔ mph
  if (before === await page.locator('.gm-speed').innerText() || !/km\/h/.test(before)) fail(`${device}: speed unit did not switch from km/h`);
  await page.locator('.gm-speed').click();
  await check('7-history');
  await page.getByRole('button', { name: '이전' }).click();
  await page.getByRole('button', { name: 'LIVE로 돌아가기' }).click();
  await chip('다음 공');

  current = 'between_innings';
  await chip('공수 교대');
  await check('8-wait');

  current = 'left_handed_batter';                   // after a break the old result does not replay; a lefty mirrors words and figure
  await chip('다음 공');
  if (!/바깥쪽/.test(await page.locator('.gm-where').innerText())) fail(`${device}: middle-left to a left-hander should read 바깥쪽`);
  if (await page.locator('.gm-batter circle').evaluate((c) => Number(c.getAttribute('cx')) < 185)) fail(`${device}: left-handed batter drawn on the wrong side`);
  await check('9-lefty');

  current = 'unsupported_pitcher';
  await page.locator('.gm-hero', { hasText: '추천 없음' }).waitFor({ timeout: 15000 });
  await check('10-abstain');

  if (errors.length) fail(`${device}: page errors: ${errors.join(' | ')}`);
  await context.close();
}
await browser.close();
console.log(process.exitCode ? 'game frame smoke: FAILED' : 'game frame smoke: ok');

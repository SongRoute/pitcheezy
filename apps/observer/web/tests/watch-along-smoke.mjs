// Watch-along smoke on a running observer (default http://127.0.0.1:8766) with a precomputed game.
//   PLAYWRIGHT_BROWSERS_PATH=... node tests/watch-along-smoke.mjs [baseUrl] [gamePk] [screenshotDir]
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://127.0.0.1:8766';
const game = process.argv[3] ?? '849843';
const shots = process.argv[4];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };
await page.goto(`${base}/watch?game=${game}#i=0`);
await page.getByText('검증 전 실험 버전 · 위치는 실제 투구 분포 근사').waitFor();
await page.locator('.wa-board').waitFor();
// first visit: one-screen guide, dismissed once and remembered (localStorage)
await page.getByRole('dialog', { name: '이렇게 보세요' }).waitFor();
if (shots) await page.screenshot({ path: `${shots}/watch-guide.png`, fullPage: true });
await page.getByRole('button', { name: '알겠어요' }).click();
await page.reload();
await page.locator('.wa-board').waitFor();
if (await page.locator('.wa-guide').count()) fail('guide shown again after dismissal');
if (await page.locator('.wa-reveal').count()) fail('actual pitch visible before reveal');
const before = await page.locator('.wa-card').innerText();
if (!/투구 전 추천/.test(before)) fail('no pre-pitch card');
if (shots) await page.screenshot({ path: `${shots}/watch-pre.png`, fullPage: true });
await page.getByRole('button', { name: '실제 투구 공개' }).click();
await page.locator('.wa-reveal').waitFor();
const after = await page.locator('.wa-reveal').innerText();
if (!/승리확률/.test(after)) fail('no WE change after reveal');
if (shots) await page.screenshot({ path: `${shots}/watch-revealed.png`, fullPage: true });
await page.getByRole('button', { name: '다음 공 →' }).click();
await page.waitForFunction(() => window.location.hash === '#i=1');
if (await page.locator('.wa-reveal').count()) fail('reveal leaked into the next pitch');
// jump to the first unsupported pitch through the plate-appearance selector
const options = await page.locator('select option').allInnerTexts();
const small = await page.evaluate(() => [...document.querySelectorAll('.wa-page button, .wa-page select, .wa-back, .wa-page summary')]
  .map((el) => el.getBoundingClientRect()).filter((r) => r.width && (r.height < 44 || r.width < 44)).length);
if (small) fail(`${small} tap targets under 44px`);
// jump to the second half-inning from the inning row
await page.locator('.wa-innings button').nth(1).click();
if (!/말/.test(await page.locator('.wa-innings button.current').innerText())) fail('inning jump did not move');
const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
if (scrollWidth > 390) fail(`horizontal scroll at phone width: ${scrollWidth}`);
// an unsupported pitch shows its reason before any reveal
const timeline = await (await fetch(`${base}/api/watch/${game}`)).json();
const unsupported = timeline.decisions.find((d) => d.pre.status === 'unsupported');
if (unsupported) {
  await page.goto(`${base}/watch?game=${game}#i=${unsupported.index}`);
  await page.getByText('추천 없음').waitFor();
  if (!(await page.getByText(unsupported.pre.reason).count())) fail('unsupported reason not shown');
  if (shots) await page.screenshot({ path: `${shots}/watch-unsupported.png`, fullPage: true });
}
console.log(JSON.stringify({ plate_appearances: options.length, pre: before.slice(0, 80), after: after.slice(0, 120) }));
await browser.close();

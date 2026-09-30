// Phone-width smoke of the watch-along + delayed-live service on a running server.
//   node tests/service-smoke.mjs [baseUrl] [liveGamePk] [screenshotDir]
// For a rehearsal live game: demo_precompute.py live-replay ... then serve.sh with PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR.
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://127.0.0.1:8766';
const liveGame = process.argv[3];
const shots = process.argv[4];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, locale: 'ko-KR' });
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };
const noSideScroll = async (where) => {
  const width = await page.evaluate(() => document.documentElement.scrollWidth);
  if (width > 390) fail(`horizontal scroll on ${where}: ${width}`);
};
const report = {};

const health = await (await fetch(`${base}/api/health`)).json();
report.health = health.status;
if (!health.demo?.watch_dir_ready) fail('watch directory not ready');

// root goes to the service home; the picker lists games with coverage and no final score
await page.goto(base);
await page.waitForURL(/\/watch$/);
await page.getByText('끝난 경기 · 중계 영상과 함께 보기').waitFor();
const games = page.locator('.wa-game[href^="/watch?game="]');
await games.first().waitFor();
report.games = await games.count();
if (!(await page.locator('.wa-coverage').count())) fail('no coverage bar on the picker');
if (/최종|final/i.test(await page.locator('.wa-games').last().innerText())) fail('picker spoils the final score');
await noSideScroll('picker');
if (shots) await page.screenshot({ path: `${shots}/1-picker.png`, fullPage: true });

// watch-along: recommendation first, reveal, sequence strip, WE card
await games.first().click();
await page.locator('.wa-board').waitFor();
await page.getByText('투구 전 추천').waitFor();
if (await page.locator('.wa-reveal').count()) fail('actual pitch visible before reveal');
await page.getByRole('button', { name: '실제 투구 공개' }).click();
await page.locator('.wa-we-card').waitFor();
await page.getByRole('button', { name: '다음 공 →' }).click();
await page.locator('.wa-strip li').nth(1).waitFor();
if (await page.locator('.wa-reveal').count()) fail('reveal leaked into the next pitch');
await noSideScroll('watch');
if (shots) await page.screenshot({ path: `${shots}/2-watch.png`, fullPage: true });

if (liveGame) {
  await page.goto(`${base}/live?game=${liveGame}`);
  await page.locator('.wa-board').waitFor({ timeout: 20000 });
  await page.getByText('검증 전 실험 버전 · 위치는 실제 투구 분포 근사').waitFor();
  await page.locator('.wa-type, .wa-none:not(.wa-pulse)').first().waitFor({ timeout: 60000 });
  report.live_recommendation = await page.locator('.wa-card').first().innerText();
  await page.locator('.wa-prev').waitFor({ timeout: 30000 });
  report.live_previous = (await page.locator('.wa-prev').innerText()).slice(0, 120);
  await noSideScroll('live');
  if (shots) await page.screenshot({ path: `${shots}/3-live.png`, fullPage: true });
}
console.log(JSON.stringify(report));
await browser.close();

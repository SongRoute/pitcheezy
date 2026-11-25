// Phone-width smoke of the watch-along + delayed-live service on a running server.
//   node tests/service-smoke.mjs [baseUrl] [liveGamePk|-] [screenshotDir|-] [replayGamePk]
// For a rehearsal live game: demo_precompute.py live-replay ... then serve.sh with PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR.
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://127.0.0.1:8766';
const liveGame = process.argv[3] && process.argv[3] !== '-' ? process.argv[3] : null;
const shots = process.argv[4] && process.argv[4] !== '-' ? process.argv[4] : null;
const replayGame = process.argv[5];
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
await page.getByRole('button', { name: '알겠어요' }).click();  // first-visit guide
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
if (replayGame) {  // a finished game replayed at its real pace through the live pipeline
  await page.goto(`${base}/watch`);
  await page.locator(`.wa-replay-link[href="/live?game=${replayGame}&replay=0"]`).click();
  await page.locator('.wa-board').waitFor({ timeout: 20000 });
  await page.waitForURL(/t0=/);  // the replay clock is pinned into the link
  await page.getByText('다시 보기 · 실제 속도').waitFor();
  const clock = () => page.locator('.wa-sync strong').innerText();
  const before = await clock();
  if (await page.locator('.wa-prev').count()) fail('a pitch is shown before the first pitch was thrown');
  await page.locator('.wa-type, .wa-none:not(.wa-pulse)').first().waitFor({ timeout: 60000 });
  report.replay_recommendation = (await page.locator('.wa-card').first().innerText()).slice(0, 80);
  await page.locator('.wa-prev').waitFor({ timeout: 60000 });  // the first pitch arrives about 20 s in
  report.replay_first_pitch_at = await clock();
  const t0 = () => Number(new URL(page.url()).searchParams.get('t0'));
  const pinned = t0();
  await page.getByRole('button', { name: '5초 늦게' }).click();
  if (Math.abs(t0() - pinned - 5) > 0.01) fail(`+5 did not move the replay clock: ${t0() - pinned}`);
  if ((await page.locator('.wa-innings a').count()) < 2) fail('no inning jump targets on the replay');
  await noSideScroll('replay');
  if (shots) await page.screenshot({ path: `${shots}/4-replay.png`, fullPage: true });
  report.replay_clock = [before, await clock()];
  // live sync control: the viewer's own delay is remembered on the device
  await page.goto(`${base}/live?game=${replayGame}`);
  await page.getByText('중계 맞추기').waitFor();  // open on a device that has not set its delay yet
  await page.getByRole('button', { name: '5초 늦게' }).click();
  await page.getByText('내 화면이 35초 늦음').waitFor();
  if (await page.evaluate(() => localStorage.getItem('pitcheezy.live.delay')) !== '35') fail('viewer delay not remembered');
  if (shots) await page.screenshot({ path: `${shots}/5-live-sync.png`, fullPage: true });
}
console.log(JSON.stringify(report));
await browser.close();

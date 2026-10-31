// WatchSummaryPanel on mocked /api responses (no backend, no data): stale/abort, step-back clearing, retry,
// through=-1, null/zero WE gaps, 360px overflow. Starts its own Vite dev server on a free port.
//   node tests/watch-summary-panel.mjs [screenshotDir]   (PLAYWRIGHT_BROWSERS_PATH if browsers are not in the default cache)
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = fileURLToPath(new URL('..', import.meta.url));
const shots = process.argv[2];
const GAME = 900002;
const HARNESS = `
import { createElement, useState } from 'react';
import { createRoot } from 'react-dom/client';
import WatchSummaryPanel from '/src/WatchSummaryPanel.tsx';
import '/src/styles.css';
import '/src/watch-along.css';
function Harness() {
  const [props, setProps] = useState(null);
  window.__setProps = setProps;
  return createElement('div', { className: 'wa-page', style: { padding: '12px' } }, props && createElement(WatchSummaryPanel, props));
}
createRoot(document.getElementById('root')).render(createElement(Harness));
window.__ready = true;`;
const harness = {
  name: 'ws-harness',
  resolveId: (id) => id === 'virtual:ws-harness' ? '\0virtual:ws-harness' : null,
  load: (id) => id === '\0virtual:ws-harness' ? HARNESS : null,
  configureServer(server) {
    server.middlewares.use('/__ws', async (req, res) => {
      const html = '<!doctype html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>'
        + '<body><div id="root"></div><script type="module" src="/@id/virtual:ws-harness"></script></body></html>';
      res.setHeader('Content-Type', 'text/html');
      res.end(await server.transformIndexHtml(req.originalUrl, html));
    });
  },
};

const cell = (runs, status = 'complete') => ({ runs, status });
const play = (index, inning, half, label, before, after, runs = 0) => ({
  index, at_bat_number: index, pitch_number: 1, inning, half, pitcher: { id: 1, name: 'P One' }, batter: { id: 2, name: 'B Two' },
  event_label: label, result_label: label, play_text: `${label} play`,
  runs_scored: half === 'Top' ? { away: runs, home: 0 } : { away: 0, home: runs }, score_after: { away: 0, home: 0 },
  we: { kind: 'observed_model_change', home_before: before, home_after: after, home_delta: after - before,
        batting_delta: half === 'Top' ? before - after : after - before } });
const point = (index, inning, half, before, after) => ({ index, inning, half, at_bat_number: index, pitch_number: 1, home_before: before, home_after: after });
const summary = (through, extra = {}) => ({
  schema: 'pitcheezy-watch-summary-v1', game_pk: GAME, through, revealed_pitches: through + 1, complete: false,
  teams: { away: 'Los Angeles Dodgers', home: 'Philadelphia Phillies' },
  score: { away: 0, home: 0, source: through < 0 ? 'pre_pitch_situation' : 'next_pre_pitch_situation', after_index: through < 0 ? null : through },
  innings: [], we_curve: [], events: [], largest_we_changes: [],
  notes: { we: '승리확률 변화는 고정된 모델 추정치의 관측 차이이며, 선수 책임이나 인과 기여도가 아닙니다.', we_model: '표시용', attribution: null },
  issues: [], ...extra });
// through=3: values before 0..3 = .5, null, .4, 0 and after(3) = .45 → segments [0], [2,3,4]
const EARLY = summary(3, {
  innings: [{ inning: 1, away: cell(0), home: cell(null, 'in_progress') }],
  we_curve: [point(0, 1, 'Top', .5, null), point(1, 1, 'Top', null, .4), point(2, 1, 'Bot', .4, 0), point(3, 1, 'Bot', 0, .45)],
  largest_we_changes: [play(3, 1, 'Bot', '볼넷', 0, .45), play(2, 1, 'Bot', '삼진', .4, 0)] });
// through=7: a later, 12-inning game state with a unique play label that must never show at through=3
const LATE = summary(7, {
  complete: true, score: { away: 3, home: 4, source: 'final_record', after_index: 7 },
  innings: Array.from({ length: 12 }, (_, i) => ({ inning: i + 1, away: cell(i === 2 ? 3 : 0), home: i === 11 ? cell(1) : cell(i === 4 ? 3 : 0) })),
  we_curve: Array.from({ length: 8 }, (_, i) => point(i, i + 5, 'Top', .3 + i * .05, .35 + i * .05)),
  largest_we_changes: [play(7, 12, 'Bot', '끝내기 홈런', .6, 1, 1)] });

const server = await createServer({ root, logLevel: 'error', plugins: [harness], server: { port: 0, host: '127.0.0.1', strictPort: false } });
await server.listen();
const base = server.resolvedUrls.local[0].replace(/\/$/, '');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 360, height: 780 } });
let failed = 0;
const fail = (message) => { console.error('FAIL', message); failed++; };
const requests = [];
const delays = new Map();
let errorOnce = new Set();
await page.route('**/api/watch/**', async (route) => {
  const url = new URL(route.request().url());
  requests.push(url.pathname + url.search);
  const through = Number(url.searchParams.get('through'));
  await new Promise((r) => setTimeout(r, delays.get(through) ?? 0));
  if (errorOnce.delete(through)) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: '관전 자료 형식을 확인할 수 없습니다.' }) });
  const body = through === -1 ? summary(-1) : through === 3 ? EARLY : through === 7 ? LATE : through === 9 ? { ...EARLY, through: 8 } : summary(through);
  return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
});
page.on('pageerror', (e) => fail(`page error: ${e.message}`));
const panel = page.locator('.ws-panel');
const set = (through, gamePk = GAME) => page.evaluate(([g, t]) => window.__setProps({ gamePk: g, through: t }), [gamePk, through]);
const text = () => panel.innerText();
const overflow = () => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

try {
  await page.goto(`${base}/__ws`);
  await page.waitForFunction(() => window.__ready && window.__setProps);

  // through=-1: nothing revealed, starting score, no chart, explicit through in the request
  await set(-1);
  await page.getByText('아직 공개된 공이 없어요. 공을 공개하면').waitFor();
  if (!/경기 시작 전/.test(await text())) fail('through=-1 caption');
  if (!requests.at(-1).includes('through=-1')) fail(`request without through=-1: ${requests.at(-1)}`);

  // gaps: null breaks the line, 0 is a value; pp labels and not-causal note
  await set(3);
  await page.getByText('볼넷', { exact: true }).waitFor();
  const lines = await panel.locator('svg polyline.we').evaluateAll((els) => els.map((e) => e.getAttribute('points').trim().split(/\s+/).length));
  const dots = await panel.locator('svg circle.we-dot').count();
  if (JSON.stringify(lines) !== '[3]' || dots !== 1) fail(`segments ${JSON.stringify(lines)} dots ${dots} (want [3] and 1)`);
  const t3 = await text();
  if (!/\+45\.0%p/.test(t3) || !/−40\.0%p/.test(t3)) fail('pp deltas (0 → 45%, 40% → 0%) not shown');
  if (!/0% → 45%/.test(t3)) fail('zero WE shown as missing');
  if (!/퍼센트포인트/.test(t3) || !/인과 기여도가 아닙니다/.test(t3)) fail('pp / not-causal note missing');
  if (!/자료 없는 1개 지점은 선을 끊음/.test(t3)) fail('gap count caption');
  if (!(await panel.locator('svg[role=img]').getAttribute('aria-label')).includes('필라델피아')) fail('sparkline aria label');
  if (!/\?/.test(await panel.locator('tbody tr').nth(1).innerText())) fail('unknown in-progress runs not "?"');
  if (shots) await page.screenshot({ path: `${shots}/ws-early.png`, fullPage: true });

  // later state with extra innings: page never overflows, the table scrolls inside
  await set(7);
  await page.getByText('끝내기 홈런', { exact: true }).waitFor();
  if ((await overflow()) > 0) fail(`page overflows horizontally at 360px by ${await overflow()}px`);
  const scroll = await panel.locator('.ws-table').evaluate((e) => e.scrollWidth > e.clientWidth);
  if (!scroll) fail('12-inning table should scroll inside its box');
  const [tableBox, totalBox] = [await panel.locator('.ws-table').boundingBox(), await panel.locator('tbody td.total').first().boundingBox()];
  if (totalBox.x + totalBox.width > tableBox.x + tableBox.width + 1) fail('R column is cut off instead of staying visible');
  if (!/최종 점수/.test(await text())) fail('final caption');
  if (shots) await page.screenshot({ path: `${shots}/ws-late.png`, fullPage: true });

  // step back with a slow response: the later summary disappears immediately, never re-appears
  delays.set(3, 800);
  await set(3);
  const immediately = await text();
  if (/끝내기 홈런|최종 점수/.test(immediately)) fail('later summary still visible right after stepping back');
  if (!/불러오는 중/.test(immediately)) fail('no loading state while stepping back');
  await page.getByText('볼넷', { exact: true }).waitFor();

  // stale ordering: slow 7 then fast 3 → 3 stays
  delays.set(7, 700); delays.set(3, 0);
  await set(7); await set(3);
  await page.getByText('볼넷', { exact: true }).waitFor();
  await page.waitForTimeout(1000);
  if (/끝내기 홈런/.test(await text())) fail('stale through=7 response replaced through=3');
  delays.clear();

  // error → retry
  errorOnce = new Set([5]);
  await set(5);
  await page.getByRole('alert').waitFor();
  const retry = panel.getByRole('button', { name: '다시 시도' });
  const box = await retry.boundingBox();
  if (!box || box.height < 44 || box.width < 44) fail('retry tap target under 44px');
  await retry.click();
  await page.getByText('6구 공개 후', { exact: true }).waitFor();

  // a response for another point is refused, invalid props make no request
  await set(9);
  await page.getByText('경기 요약 자료 형식을 확인할 수 없어요.').waitFor();
  const before = requests.length;
  await set(-2);
  await page.getByText('요약할 투구 위치가 올바르지 않아요.').waitFor();
  if (requests.length !== before) fail('invalid through still requested');
  if (requests.some((r) => !/[?&]through=-?\d+/.test(r))) fail('a request without through');
} catch (error) {
  fail(error.stack ?? String(error));
} finally {
  await browser.close();
  await server.close();
}
console.log(failed ? `${failed} failure(s)` : `PASS watch-summary-panel (${requests.length} mocked requests)`);
process.exitCode = failed ? 1 : 0;

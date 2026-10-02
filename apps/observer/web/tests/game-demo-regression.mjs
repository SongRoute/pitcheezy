// /game demo regressions: stale feed and recovery, catcher setup after the pitch only, replay scene jump and pause,
// /watch → /game link, the 2026 copy. Runs on the built dist/ with SYNTHETIC /api responses: the state bodies reuse
// tests/fixtures/game-frame-states.json, the setup, outline, game list and replay clock are made up here. No server, no model.
//   npm run build && PLAYWRIGHT_BROWSERS_PATH=... node tests/game-demo-regression.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, '..', 'dist');
const { states } = JSON.parse(readFileSync(path.join(here, 'fixtures', 'game-frame-states.json'), 'utf8'));
const shots = process.argv[2];
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const GAME = 849845;
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };
const clone = (v) => JSON.parse(JSON.stringify(v));

// Synthetic setup lines: 11:1 has a mitt at x −0.05 ft (middle), 11:2 is unavailable, 10:4 is off to the right.
const SETUP = { schema: 'pitcheezy-setup-estimates-v1', badge: '합성 시험 자료', game_pk: GAME, display: { horizontal: true, vertical: false },
  pitches: [{ at_bat_number: 10, pitch_number: 4, status: 'estimated', plate_x_feet: 0.9, x_band: 'off_right', unavailable_reason: null },
    { at_bat_number: 11, pitch_number: 1, status: 'estimated', plate_x_feet: -0.05, x_band: 'middle', unavailable_reason: null },
    { at_bat_number: 11, pitch_number: 2, status: 'unavailable', plate_x_feet: null, x_band: null, unavailable_reason: 'occluded' }] };
const OUTLINE = { badge: 'synthetic', pitches: 120, halves: [{ label: '1초', first: 0 }, { label: '1말', first: 14 }, { label: '2초', first: 31 }, { label: '2말', first: 47 }] };

const browser = await chromium.launch();
async function open(width, height) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2 });
  const page = await context.newPage();
  const server = { current: 'second_pitch', mode: 'ok', requests: [], setupCalls: 0, outlineCalls: 0, t0: null, index: 0 };
  await page.route('http://pz.test/**', (route) => {
    const url = new URL(route.request().url());
    if (/^\/api\/live\/\d+\/state$/.test(url.pathname)) {
      server.requests.push({ at: Date.now(), query: Object.fromEntries(url.searchParams) });
      if (server.mode === 'down') return route.abort('connectionrefused');
      const body = clone(states[server.current]);
      if (server.mode === 'feed_down') body.sync = { ...body.sync, feed_ok: false };
      const start = url.searchParams.get('replay');
      if (start !== null) {  // the replay clock this synthetic server pins: a fresh t0 when the link has none
        const t0 = url.searchParams.has('t0') ? Number(url.searchParams.get('t0')) : Date.now() / 1000 - 1;
        body.replay = { start: Number(start), t0, speed: 1, lag_s: 0, elapsed_s: 600 + server.index * 20, pitches: OUTLINE.pitches, index: Math.max(Number(start), server.index) };
      }
      return route.fulfill({ json: body });
    }
    if (url.pathname === `/api/watch/${GAME}/setup`) { server.setupCalls += 1; return route.fulfill({ json: SETUP }); }
    if (url.pathname === `/api/replay/${GAME}/outline`) { server.outlineCalls += 1; return route.fulfill({ json: OUTLINE }); }
    if (url.pathname === '/api/watch/games') return route.fulfill({ json: { games: [{ game_pk: GAME, date: '2026-09-29', game_type: 'F', away_team: 'Philadelphia Phillies', home_team: 'Atlanta Braves', ready_share: 0.8 }] } });
    if (url.pathname === '/api/live/games') return route.fulfill({ json: { games: [] } });
    if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this test' } });
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  return { context, page, server, errors };
}
const layout = async (page, device, label) => {
  await page.waitForTimeout(400);
  // /watch: only the replay links are checked; the brand link there predates this test
  const found = await page.evaluate((targets) => ({ page: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight) - window.innerHeight,
    wide: document.documentElement.scrollWidth - window.innerWidth,
    small: [...document.querySelectorAll(targets)].map((el) => [el, el.getBoundingClientRect()])
      .filter(([, r]) => r.width && (r.height < 44 || r.width < 44)).map(([el]) => el.className || el.textContent) }), label === 'watch' ? '.wa-replay-link' : '.wa-page button, .wa-page a');
  if (label !== 'watch' && found.page > 0) fail(`${device} ${label}: page scrolls by ${found.page}px`);
  if (found.wide > 0) fail(`${device} ${label}: page is ${found.wide}px too wide`);
  if (found.small.length) fail(`${device} ${label}: tap targets under 44px: ${found.small.join(', ')}`);
  if (shots) await page.screenshot({ path: `${shots}/demo-${device}-${label}.png` });
};
const stageChip = (page, text, timeout = 9000) => page.locator('.gm-stage .gm-chip', { hasText: text }).waitFor({ timeout });

for (const [device, width, height] of [['iphone17', 402, 681], ['galaxy-s26', 360, 620]]) {
  // 1. Stale feed: the last good state stays but is not called current; recovery clears it.
  {
    const { context, page, server, errors } = await open(width, height);
    server.current = 'third_pitch';
    await page.goto(`http://pz.test/game?game=${GAME}`);
    await stageChip(page, '다음 공');
    if (await page.locator('.gm-stale').count()) fail(`${device}: stale banner on a healthy feed`);
    server.mode = 'down';
    await page.locator('.gm-stale', { hasText: '연결 끊김' }).waitFor({ timeout: 9000 });
    await stageChip(page, '마지막 상황');
    if (await page.locator('.gm-stage .gm-chip', { hasText: '다음 공' }).count()) fail(`${device}: stale state still labelled 다음 공`);
    if (!/연결 끊김/.test(await page.locator('.gm-live').innerText())) fail(`${device}: header still shows the feed as live`);
    if (!/받은 상황/.test(await page.locator('.gm-stale').innerText())) fail(`${device}: stale banner has no last-received time`);
    if (!await page.locator('.gm-hero').count()) fail(`${device}: last good state was dropped`);
    await layout(page, device, 'stale-down');
    server.mode = 'ok';
    await stageChip(page, '다음 공');
    if (await page.locator('.gm-stale').count()) fail(`${device}: stale banner did not clear on recovery`);
    if ((await page.locator('.gm-live').innerText()).trim() !== 'LIVE') fail(`${device}: header did not return to LIVE`);
    if (await page.locator('.gm-page.stale').count()) fail(`${device}: page still dimmed after recovery`);
    server.mode = 'feed_down';                         // the server answers, but its MLB feed is failing
    await page.locator('.gm-stale', { hasText: '경기 기록 수신 끊김' }).waitFor({ timeout: 9000 });
    if (!/기록 멈춤/.test(await page.locator('.gm-live').innerText())) fail(`${device}: feed_ok false not shown in the header`);
    await layout(page, device, 'stale-feed');
    server.mode = 'ok';
    await page.locator('.gm-stale').waitFor({ state: 'detached', timeout: 9000 });
    await stageChip(page, '다음 공');
    // 2026 copy and the experimental badge
    await page.getByRole('button', { name: '자세히' }).click();
    await page.getByRole('button', { name: '안내' }).click();
    const note = await page.locator('.gm-sheet').innerText();
    if (!note.includes('2026 정규시즌 1회 평가: 개선 근거를 확인하지 못했습니다')) fail(`${device}: 2026 evaluation copy missing`);
    if (/성능 평가는 아직 없어요|안전/.test(note)) fail(`${device}: obsolete or harmlessness copy present`);
    if (/포수 미트/.test(note)) fail(`${device}: setup note on a live game`);
    if (!await page.locator('.gm-badge', { hasText: '실험판' }).count()) fail(`${device}: experimental badge missing`);
    if (server.setupCalls) fail(`${device}: live game asked for catcher setup`);
    if (errors.length) fail(`${device} stale: page errors: ${errors.join(' | ')}`);
    await context.close();
  }

  // 2. Catcher setup: only on the pitch it belongs to, only after it is thrown; nothing for pitches without a line.
  {
    const { context, page, server, errors } = await open(width, height);
    server.current = 'new_batter_after_strikeout';  // next is 11:1; the zone shows the recommendation only
    await page.goto(`http://pz.test/game?game=${GAME}&replay=0&t0=${Date.now() / 1000 - 5}`);
    await stageChip(page, '다음 공');
    await page.waitForTimeout(500);
    if (await page.locator('.gm-setup').count()) fail(`${device}: setup drawn before the pitch`);
    server.current = 'second_pitch';                // 11:1 has just been thrown
    await stageChip(page, '방금 던진 공');
    if (await page.locator('.gm-setup').count() !== 1) fail(`${device}: setup missing on 11:1 after the pitch`);
    const line = await page.locator('.gm-setup line').evaluate((l) => ({ x1: Number(l.getAttribute('x1')), x2: Number(l.getAttribute('x2')) }));
    if (line.x1 !== line.x2 || Math.abs(line.x1 - 185) > 6) fail(`${device}: setup at x −0.05 ft should be a vertical line near the zone centre, got ${JSON.stringify(line)}`);
    if (await page.locator('.gm-setup circle').count()) fail(`${device}: setup drawn as a point (implies a height)`);
    await layout(page, device, 'setup-after');
    await stageChip(page, '다음 공');                 // hold over: the next recommendation has no mitt
    if (await page.locator('.gm-setup').count()) fail(`${device}: setup leaked into the next recommendation`);
    server.current = 'third_pitch';                  // 11:2 is unavailable: nothing drawn, nothing said
    await stageChip(page, '방금 던진 공');
    if (await page.locator('.gm-setup').count()) fail(`${device}: setup drawn for an unavailable pitch`);
    if (await page.getByText(/자료 없음|미트 위치 없음/).count()) fail(`${device}: missing-setup text shown`);
    await stageChip(page, '다음 공');
    await page.getByRole('button', { name: '지난 장면' }).click();   // looking back at 11:1 shows its mitt again
    await page.getByRole('button', { name: '이전' }).click();
    await page.locator('.gm-stage .gm-chip', { hasText: '1구' }).waitFor();
    if (await page.locator('.gm-setup').count() !== 1) fail(`${device}: setup missing when looking back at 11:1`);
    await page.getByRole('button', { name: 'LIVE로 돌아가기' }).click();
    await page.getByRole('button', { name: '자세히' }).click();
    await page.getByRole('button', { name: '안내' }).click();
    if (!/포수 미트 위치는 영상에서 읽은 추정이며 검토 전입니다/.test(await page.locator('.gm-sheet').innerText())) fail(`${device}: setup caveat missing in detail`);
    if (/의도|제구 오차|투수가 노린/.test(await page.locator('.gm-sheet').innerText())) fail(`${device}: setup described as intent`);
    if (server.setupCalls !== 1) fail(`${device}: setup fetched ${server.setupCalls} times`);
    if (errors.length) fail(`${device} setup: page errors: ${errors.join(' | ')}`);
    await context.close();
  }

  // 3. Replay controls: scene jump through the outline, restart, and a pause that moves t0 by its length.
  {
    const { context, page, server, errors } = await open(width, height);
    server.current = 'third_pitch';
    await page.goto(`http://pz.test/game?game=${GAME}&replay=0`);
    await page.waitForFunction(() => new URLSearchParams(window.location.search).has('t0'));
    await stageChip(page, '다음 공');
    await page.getByRole('button', { name: '다시 보기 조작' }).click();
    await page.getByRole('group', { name: '장면으로 이동' }).waitFor();
    if (await page.locator('.gm-scenes button').count() !== OUTLINE.halves.length + 1) fail(`${device}: scene list does not match the outline`);
    await layout(page, device, 'replay-sheet');
    server.index = 33; const before = server.requests.length;
    await page.locator('.gm-scenes button', { hasText: '2초' }).click();
    await page.waitForFunction(() => new URLSearchParams(window.location.search).get('replay') === '31' && new URLSearchParams(window.location.search).has('t0'));
    const jumped = server.requests.slice(before);
    if (!jumped.length || jumped[0].query.replay !== '31' || 't0' in jumped[0].query) fail(`${device}: scene jump did not ask for a fresh replay=31 clock: ${JSON.stringify(jumped[0])}`);
    if (jumped.some((r) => r.query.replay !== '31')) fail(`${device}: old scene still polled after the jump`);
    await stageChip(page, '다음 공');
    await page.getByRole('button', { name: '다시 보기 조작' }).click();
    if ((await page.locator('.gm-scenes button.current').innerText()).trim() !== '2초') fail(`${device}: current scene not marked`);
    await page.getByRole('button', { name: '닫기' }).click();

    const t0 = Number(new URLSearchParams(new URL(page.url()).search).get('t0'));
    await page.getByRole('button', { name: '일시정지' }).click();
    const pausedAt = server.requests.length;
    await page.locator('.gm-paused').waitFor();
    if (!/일시정지/.test(await page.locator('.gm-live').innerText())) fail(`${device}: header does not say paused`);
    await layout(page, device, 'paused');
    await page.waitForTimeout(4500);
    if (server.requests.length - pausedAt > 1) fail(`${device}: polled ${server.requests.length - pausedAt} times while paused`);
    await page.getByRole('button', { name: '이어서 보기' }).click();
    await page.locator('.gm-paused').waitFor({ state: 'detached' });
    await page.waitForFunction((old) => Number(new URLSearchParams(window.location.search).get('t0')) > old, t0);
    const resumed = Number(new URLSearchParams(new URL(page.url()).search).get('t0'));
    if (resumed - t0 < 4.4 || resumed - t0 > 8) fail(`${device}: resume moved t0 by ${(resumed - t0).toFixed(2)} s, expected the ~4.5 s pause`);
    await page.waitForTimeout(2500);
    const last = server.requests[server.requests.length - 1].query;
    if (Number(last.t0) !== resumed || last.replay !== '31') fail(`${device}: polling after resume does not use the moved t0`);

    server.index = 0;
    await page.getByRole('button', { name: '다시 보기 조작' }).click();
    await page.getByRole('button', { name: '처음부터' }).click();
    await page.waitForFunction(() => new URLSearchParams(window.location.search).get('replay') === '0' && new URLSearchParams(window.location.search).has('t0'));
    await stageChip(page, '다음 공');
    if (server.outlineCalls !== 1 || server.setupCalls !== 1) fail(`${device}: outline/setup refetched on jump (${server.outlineCalls}/${server.setupCalls})`);
    if (errors.length) fail(`${device} replay: page errors: ${errors.join(' | ')}`);
    await context.close();
  }

  // 4. /watch list links each finished game to the /game replay; the old links stay.
  {
    const { context, page, errors } = await open(width, height);
    await page.goto('http://pz.test/watch');
    const link = page.locator(`a[href="/game?game=${GAME}&replay=0"]`);
    await link.waitFor();
    if (!await page.locator(`a[href="/watch?game=${GAME}"]`).count()) fail(`${device}: step-through watch link lost`);
    if (!await page.locator(`a[href="/live?game=${GAME}&replay=0"]`).count()) fail(`${device}: real-pace /live link lost`);
    await layout(page, device, 'watch');
    await link.click();
    await page.waitForURL(/\/game\?game=849845&replay=0/);
    await page.locator('.gm-live', { hasText: '다시 보기' }).waitFor();
    if (errors.length) fail(`${device} watch: page errors: ${errors.join(' | ')}`);
    await context.close();
  }
}
await browser.close();
console.log(process.exitCode ? 'game demo regression: FAILED' : 'game demo regression: ok');

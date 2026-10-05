// Service UI (redesign 2026-10-05, frontend-001): /watch KST list → /watch/{game} one game screen.
// Canned CONTRACT rev1 answers (tests/fixtures/service-849843.json = real 849843 first 70 pitches) on the built dist/.
// Checks list rules, one-screen fit at 320/360/390/430, equal candidates, no reveal button, archived rec + actual together,
// missing/unsupported/<3 candidates honest, live follow / pin / 최신으로 / auto off / manual refresh / stale recovery,
// incremental ?after= merge and seek-back refetch, demo param forwarding without a banner, 423 embargo, sheet focus.
//   npm run build && node tests/service-ui-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const shots = process.argv[2]; if (shots) mkdirSync(shots, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let failures = 0;
const ok = (cond, message) => { if (cond) console.log('ok  ', message); else { failures++; console.error('FAIL', message); } };
const clone = (v) => JSON.parse(JSON.stringify(v));

// Synthetic edits (marked): pitch 2 has no stored rec, pitch 3 unsupported, pitch 4 only two candidates.
const final = clone(fx.final);
final.pitches[2].rec = { provenance: 'absent', status: 'missing', reason: null, recorded_at: null, candidates: [], policy: null };
final.pitches[3].rec = { provenance: 'asof_replay', status: 'unsupported', reason: '자동 볼·스트라이크(투구 없음)라 추천 대상이 아닙니다.', recorded_at: null, candidates: [], policy: null };
final.pitches[4].rec.candidates = final.pitches[4].rec.candidates.slice(0, 2);
const card = (g) => g.game;
const list = { schema: 'pitcheezy-service-games-v2', source: { kind: 'archive', session_id: null }, date: '2026-09-30', games: [
  card(final),
  // synthetic: a doubleheader pair, a live game and an embargoed game on the same KST date
  { ...card(final), game_pk: 990001, away_team: 'New York Mets', home_team: 'Atlanta Braves', start: '2026-09-29T16:10:00Z', date_kst: '2026-09-30', away_score: 3, home_score: 2, double_header: true, game_number: 1 },
  { ...card(final), game_pk: 990002, away_team: 'New York Mets', home_team: 'Atlanta Braves', start: '2026-09-29T20:40:00Z', date_kst: '2026-09-30', away_score: 1, home_score: 4, double_header: true, game_number: 2 },
  { ...card(final), game_pk: 990003, away_team: 'Boston Red Sox', home_team: 'New York Yankees', start: '2026-09-30T10:05:00Z', date_kst: '2026-09-30', status: 'live', status_text: 'In Progress', live: true, away_score: 1, home_score: 1, inning: 5, half: 'Bot' },
  { ...card(final), game_pk: 990004, away_team: 'Houston Astros', home_team: 'Seattle Mariners', start: '2026-09-30T08:00:00Z', date_kst: '2026-09-30', status: 'other', status_text: '', live: false, away_score: null, home_score: null, embargoed: true },
] };

const browser = await chromium.launch();
async function open({ width, height, handler }) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  const asks = [];
  await page.route('http://pz.test/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith('/api/')) { asks.push(url); return handler(route, url, asks); }
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = []; page.on('pageerror', (e) => errors.push(String(e)));
  return { context, page, asks, errors };
}
const archive = (route, url) => {
  if (url.pathname === '/api/service/dates') return route.fulfill({ json: { dates: ['2026-09-30', '2026-09-29'] } });
  if (url.pathname === '/api/service/games') return route.fulfill({ json: url.searchParams.get('date') === '2026-09-30' ? list : { ...list, date: url.searchParams.get('date'), games: [] } });
  if (url.pathname === '/api/service/games/849843') return route.fulfill({ json: final });
  return route.fulfill({ status: 404, json: { detail: '이 경기의 기록이 없습니다.' } });
};
const fitsOneScreen = (page) => page.evaluate(() => document.scrollingElement.scrollHeight <= window.innerHeight + 1 && document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const box = async (page, sel) => page.locator(sel).first().boundingBox();

// ---------- 1. list ----------
{
  const { context, page, errors } = await open({ width: 390, height: 664, handler: archive });
  await page.goto('http://pz.test/watch');
  await page.locator('.sv-card').first().waitFor();
  const body = await page.locator('body').innerText();
  ok(/2026년 9월 30일 \(수\)/.test(body), 'list: KST date heading 2026-09-30 (수) from dates endpoint');
  ok(!/검증 전 실험 버전|위치는 실제 투구 분포 근사|오늘의 포스트시즌|보는 법|지금 경기|끝난 경기|다시 보기|영상/.test(body), 'list: no badge, no section titles, no modes, no video words');
  ok(await page.locator('.sv-list > ul > li').count() === 5, 'list: one flat list with all 5 games');
  ok(/1차전/.test(body) && /2차전/.test(body), 'list: doubleheader 1차전/2차전 shown');
  const first = await page.locator('.sv-card').first().innerText();
  ok(/01:10/.test(first), 'list: sorted by start; 16:10Z shows as KST 01:10');
  const cubs = page.locator('.sv-card[href="/watch/849843"]');
  ok(/11:00/.test(await cubs.innerText()) && /0/.test(await cubs.innerText()) && /8/.test(await cubs.innerText()) && /경기 종료/.test(await cubs.innerText()), 'list: 849843 at 11:00 KST, final score 0:8, 경기 종료');
  ok(/LIVE/.test(await page.locator('.sv-card[href="/watch/990003"]').innerText()) && /5회 말/.test(await page.locator('.sv-card[href="/watch/990003"]').innerText()), 'list: live card shows LIVE + 5회 말');
  ok(!/\d/.test((await page.locator('.sv-card[href="/watch/990004"] .sv-card-teams').innerText())), 'list: embargoed card shows no score');
  if (shots) await page.screenshot({ path: path.join(shots, 'list-390.png'), fullPage: true });
  await cubs.click(); await page.waitForURL('**/watch/849843');
  ok(true, 'list: one tap goes straight to the game screen (no mode step)');
  ok(errors.length === 0, `list: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 2. finished game at four widths ----------
for (const [width, height] of [[320, 568], [360, 640], [375, 553], [390, 664], [430, 740]]) {
  const { context, page, errors } = await open({ width, height, handler: archive });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-zone').waitFor();
  const tag = `game ${width}x${height}`;
  const body = await page.locator('body').innerText();
  ok(/9월 30일 \(수\) 11:00/.test(body) && /2026년 9월 30일/.test(await page.locator('.sv-title').getAttribute('aria-label')) && /컵스/.test(body) && /샌디에이고/.test(body), `${tag}: KST date + start time and both teams visible`);
  ok(await fitsOneScreen(page), `${tag}: no page scroll (one screen)`);
  const zone = await box(page, '.sv-zone-wrap');
  for (const sel of ['.sv-board', '.sv-recbox', '.sv-nav', '.sv-mode']) { const b = await box(page, sel); ok(zone.width * zone.height > b.width * b.height, `${tag}: zone larger than ${sel}`); }
  const cands = page.locator('.sv-cand');
  const n = await cands.count();
  const widths = []; const fonts = [];
  for (let i = 0; i < n; i++) { widths.push(Math.round((await cands.nth(i).boundingBox()).width)); fonts.push(await cands.nth(i).locator('.sv-cand-type').evaluate((e) => getComputedStyle(e).fontSize)); }
  ok(n === 3 && new Set(widths).size === 1 && new Set(fonts).size === 1, `${tag}: 3 candidates, equal width ${widths} and font ${fonts}`);
  ok(await cands.nth(0).getAttribute('aria-checked') === 'true', `${tag}: rank 1 selected by default`);
  ok(/정책 선택 비중/.test(await cands.nth(0).getAttribute('aria-label')) && /정책 선택 비중/.test(body), `${tag}: share labelled 정책 선택 비중`);
  const minFont = await page.evaluate(() => Math.min(...[...document.querySelectorAll('.sv-page *')].filter((e) => e.childNodes.length && [...e.childNodes].some((c) => c.nodeType === 3 && c.textContent.trim()) && e.closest('svg') === null).map((e) => parseFloat(getComputedStyle(e).fontSize))));
  ok(minFont >= 13, `${tag}: smallest HTML text ${minFont}px ≥ 13px`);
  if (shots) await page.screenshot({ path: path.join(shots, `game-final-${width}x${height}.png`) });
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 3. interactions on the finished game ----------
{
  const { context, page, errors } = await open({ width: 390, height: 664, handler: archive });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-zone').waitFor();
  ok(await page.getByRole('button', { name: /공개/ }).count() === 0, 'final: no reveal button');
  ok(await page.locator('.sv-zone .sv-rec').count() === 1 && await page.locator('.sv-zone .sv-actual').count() === 1, 'final: archived rec square and actual circle shown together');
  ok(/이 공 던지기 전 추천/.test(await page.locator('.sv-rec-head').innerText()) && /실제 투구/.test(await page.locator('.sv-actual-line').innerText()), 'final: titles say before-pitch rec and actual');
  await page.locator('.sv-cand').nth(1).click();
  ok(await page.locator('.sv-zone .sv-rec').getAttribute('data-rank') === '2', 'final: picking rank 2 moves the zone mark to rank 2');
  // pitch buttons of PA 1
  const pa1 = final.pas[0];
  ok(await page.locator('.sv-pitch').count() === pa1.pitch_count, `final: ${pa1.pitch_count} pitch buttons for PA 1`);
  await page.locator('.sv-pitch').nth(2).click();
  ok(/이 공은 추천 기록이 없어요/.test(await page.locator('.sv-recbox').innerText()) && await page.locator('.sv-zone .sv-rec').count() === 0, 'final: missing rec says so, no invented mark');
  await page.getByRole('button', { name: '다음 타석' }).click();  // PA 2 starts at index 3
  ok(await page.locator('.sv-pitch.on').innerText() === '1', 'final: 다음 타석 selects its first pitch');
  ok(/자동 볼·스트라이크/.test(await page.locator('.sv-recbox').innerText()), 'final: unsupported shows the source reason');
  await page.locator('.sv-pitch').nth(1).click();
  ok(await page.locator('.sv-cand').count() === 2, 'final: only the 2 real candidates shown (no padding)');
  // PA sheet
  await page.locator('.sv-pa-open').click();
  const items = page.locator('.sv-pa-item');
  ok(await items.count() === final.pas.length, `PA sheet: all ${final.pas.length} plate appearances listed`);
  ok(/1회 초 · Pete Crow-Armstrong @ Michael King/.test(await items.nth(0).innerText()) && /파울팁 삼진/.test(await items.nth(0).innerText()), 'PA sheet: "1회 초 · 타자 @ 투수" + detailed result');
  ok(/타구 아웃/.test(await page.locator('.sv-pa-item', { hasText: 'Pedro Ramírez' }).innerText()), 'PA sheet: null result_text falls back to source event label');
  const scrollable = await page.locator('.sv-sheet-body').evaluate((e) => e.scrollHeight > e.clientHeight && getComputedStyle(e).overflowY === 'auto');
  ok(scrollable, 'PA sheet: list scrolls inside the sheet');
  if (shots) await page.screenshot({ path: path.join(shots, 'pa-sheet-390.png') });
  await page.locator('.sv-pa-item', { hasText: 'Fernando Tatis' }).click();
  ok(/중견수 방면 홈런/.test(await page.locator('.sv-actual-line').innerText()), 'PA pick: home-run PA shows its grounded result');
  await page.keyboard.press('Escape');
  await page.locator('.sv-btn', { hasText: '자세히' }).click();
  const dialog = page.getByRole('dialog');
  ok(await dialog.count() === 1 && /정책 선택 비중은 성공·적중 확률이 아니에요/.test(await dialog.innerText()) && /경기 뒤, 그 공을 던지기 전 정보만으로 계산한 추천/.test(await dialog.innerText()), 'detail: share meaning + as-of provenance on demand');
  if (shots) await page.screenshot({ path: path.join(shots, 'detail-390.png') });
  await page.keyboard.press('Escape');
  ok(await dialog.count() === 0 && await page.evaluate(() => document.activeElement?.textContent) === '자세히', 'detail: Esc closes and focus returns to 자세히');
  const unnamed = await page.evaluate(() => [...document.querySelectorAll('button,a')].filter((e) => !(e.getAttribute('aria-label') || e.textContent).trim()).length);
  ok(unnamed === 0, 'a11y: every control has a name');
  ok(errors.length === 0, `final interactions: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 4. live / demo binding ----------
{
  let phase = 'live'; let down = false; let seekBack = false;
  const handler = (route, url, asks) => {
    if (down) return route.abort('internetdisconnected');
    if (url.searchParams.get('demo') !== 'fixture') return route.fulfill({ status: 404, json: { detail: 'demo param missing' } });
    if (url.pathname !== '/api/service/games/849843') return archive(route, url);
    const full = clone(seekBack ? fx.live : phase === 'live' ? fx.live : fx.live_more);
    const after = url.searchParams.get('after');
    if (after !== null) full.pitches = full.pitches.filter((p) => p.index > Number(after));
    return route.fulfill({ json: full });
  };
  const { context, page, asks, errors } = await open({ width: 390, height: 664, handler });
  await page.goto('http://pz.test/watch/849843?demo=fixture');
  await page.locator('.sv-zone').waitFor();
  const cut = fx.live_cut;
  const body = await page.locator('body').innerText();
  ok(!/시연|demo|DEMO|데모/.test(body), 'live(demo): no demo banner or wording on the ordinary screen');
  ok(/LIVE/.test(await page.locator('.sv-top').innerText()), 'live: LIVE chip from game status');
  ok(/다음 공 추천 · \d+구/.test(await page.locator('.sv-rec-head').innerText()) && /방금 던진 공/.test(await page.locator('.sv-actual-line').innerText()), 'live latest: "방금 던진 공" vs "다음 공 추천 · n구" separated');
  ok(await page.locator('.sv-zone .sv-actual').count() === 0, 'live latest: last actual is not drawn as this rec\'s outcome (only grey 직전)');
  ok(/따라가는 중/.test(await page.locator('.sv-mode').innerText()), 'live: follow state shown');
  ok(await fitsOneScreen(page), 'live 390x664: one screen');
  if (shots) await page.screenshot({ path: path.join(shots, 'live-latest-390.png') });
  // pin a past pitch, then new pitches arrive
  await page.locator('.sv-pa-open').click();
  await page.locator('.sv-pa-item').nth(1).click();
  const pinned = await page.locator('.sv-mode').innerText();
  ok(/지난 기록 보는 중/.test(pinned), 'live: picking a past PA → 지난 기록 보는 중');
  ok(/LIVE/.test(await page.locator('.sv-top').innerText()), 'live: LIVE chip stays (viewer state is separate)');
  if (shots) await page.screenshot({ path: path.join(shots, 'live-past-390.png') });
  phase = 'more';
  const before = asks.length;
  await page.waitForTimeout(4500);
  ok(asks.length > before, 'live: still polling while viewing the past');
  ok(asks.slice(before).some((u) => u.searchParams.get('after') === String(cut)), `live: incremental poll uses ?after=${cut}`);
  ok(await page.locator('.sv-mode').innerText() === pinned, 'live: new pitches do not move the pinned selection');
  await page.locator('.sv-btn', { hasText: '최신으로' }).click();
  await page.waitForTimeout(500);
  ok(/따라가는 중/.test(await page.locator('.sv-mode').innerText()), '최신으로: back to following');
  await page.locator('.sv-pa-open').click();
  const lastLive = fx.live_more.pas[fx.live_more.pas.length - 1];
  ok(/타석 진행 중/.test(await page.locator('.sv-pa-item').last().innerText()) && await page.locator('.sv-pa-item').count() === fx.live_more.pas.length, `live: PA list has only occurred PAs (${fx.live_more.pas.length}), last = 타석 진행 중 (${lastLive.pa_key})`);
  await page.keyboard.press('Escape');
  // auto off → no polling; manual refresh → one request
  await page.getByRole('button', { name: '자동 갱신 끄기' }).click();
  const offAt = asks.length;
  await page.waitForTimeout(4500);
  ok(asks.length === offAt, 'auto OFF: no requests for 4.5 s');
  ok(/LIVE/.test(await page.locator('.sv-top').innerText()) && /자동 갱신 꺼짐/.test(await page.locator('.sv-mode').innerText()), 'auto OFF: LIVE chip unchanged, mode says 자동 갱신 꺼짐');
  await page.getByRole('button', { name: '지금 새로고침' }).click();
  await page.waitForTimeout(600);
  ok(asks.length === offAt + 1, 'manual refresh: exactly one request');
  // seek back (cutoff moves backwards) → full refetch, later pitches hidden again
  seekBack = true;
  await page.getByRole('button', { name: '지금 새로고침' }).click();
  await page.waitForTimeout(800);
  const refetchedFull = asks.slice(-2).some((u) => u.searchParams.get('after') === null);
  await page.locator('.sv-pa-open').click();
  ok(refetchedFull && await page.locator('.sv-pa-item').count() === fx.live.pas.length, 'seek back: full refetch, PA list shrinks back');
  await page.keyboard.press('Escape');
  // connection lost → stale/offline state with a way back
  await page.getByRole('button', { name: '자동 갱신 켜기' }).click();
  down = true;
  await page.waitForTimeout(3000);
  ok(/연결 끊김/.test(await page.locator('.sv-mode').innerText()) && await page.getByRole('button', { name: '지금 새로고침' }).count() === 1, 'offline: 연결 끊김 + 새로고침 shown');
  if (shots) await page.screenshot({ path: path.join(shots, 'live-offline-390.png') });
  down = false;
  await page.waitForTimeout(2600);
  ok(!/연결 끊김/.test(await page.locator('.sv-mode').innerText()), 'offline: recovers automatically when the server answers again');
  ok(asks.every((u) => u.searchParams.get('demo') === 'fixture'), 'demo: every API call carries ?demo=');
  ok(errors.length === 0, `live: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 4b. live next rec absent (e.g. LIVE_ARMB off) + new demo session ----------
{
  let session = 'fixture';
  const handler = (route, url) => {
    if (url.pathname !== '/api/service/games/849843') return archive(route, url);
    const full = clone(fx.live);
    full.next = { pitch_key: null, rec: { provenance: 'absent', status: 'missing', reason: '이 실행에서는 실시간 추천을 껐습니다.', recorded_at: null, candidates: [], policy: null } };
    full.source.session_id = session;
    if (session !== 'fixture') { full.pitches = full.pitches.slice(0, 5); full.pas = full.pas.slice(0, 2); full.cutoff.index = 99999; }
    const after = url.searchParams.get('after');
    if (after !== null) full.pitches = full.pitches.filter((p) => p.index > Number(after));
    return route.fulfill({ json: full });
  };
  const { context, page, asks } = await open({ width: 390, height: 664, handler });
  await page.goto('http://pz.test/watch/849843?demo=fixture');
  await page.locator('.sv-zone').waitFor();
  const text = await page.locator('.sv-recbox').innerText();
  ok(/다음 공 추천이 없어요\. 이 실행에서는 실시간 추천을 껐습니다\./.test(text) && await page.locator('.sv-cand').count() === 0 && await page.locator('.sv-zone .sv-rec').count() === 0, 'live: absent next rec stated with source reason, nothing invented');
  session = 'other';
  await page.waitForTimeout(2600);
  await page.locator('.sv-pa-open').click();
  ok(asks.slice(-1)[0].searchParams.get('after') === null && await page.locator('.sv-pa-item').count() === 2 && await page.locator('.sv-pitch').count() <= 5, 'live: session id change → full refetch, old cached pitches dropped');
  await context.close();
}

// ---------- 5. embargo + short height ----------
{
  const { context, page } = await open({ width: 360, height: 640, handler: (route) => route.fulfill({ status: 423, json: { detail: '시연 중인 경기입니다.' } }) });
  await page.goto('http://pz.test/watch/849843');
  await page.getByRole('alert').waitFor();
  ok(/지금 기록을 볼 수 없어요/.test(await page.getByRole('alert').innerText()), '423: neutral message, nothing revealed');
  await context.close();
}
{
  const { context, page } = await open({ width: 320, height: 460, handler: archive });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-zone').waitFor();
  const clipped = await page.evaluate(() => [...document.querySelectorAll('.sv-page section, .sv-page nav')].some((e) => getComputedStyle(e).overflow === 'hidden' && e.scrollHeight > e.clientHeight + 1));
  ok(!clipped, '320x460: no section clips its content (page may scroll instead)');
  const reach = await page.evaluate(() => { const n = document.querySelector('.sv-nav'); n.scrollIntoView(); const r = n.getBoundingClientRect(); return r.bottom <= window.innerHeight + 1; });
  ok(reach, '320x460: bottom controls reachable by scrolling');
  if (shots) await page.screenshot({ path: path.join(shots, 'game-final-320x460.png'), fullPage: true });
  await page.evaluate(() => { document.body.style.zoom = '1.3'; });
  ok(await page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth * 1.3 + 2), '320 + 130% zoom: no horizontal overflow');
  await context.close();
}

await browser.close();
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

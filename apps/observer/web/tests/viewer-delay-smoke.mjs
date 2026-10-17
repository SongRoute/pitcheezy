// Viewer delay ("중계 맞추기") on the fan screen. Canned answers on the built dist/ (no server, no model, no network), in the
// style of display-state-smoke.mjs. The feature is OFF unless the build is not the public one AND the payload says
// `features.viewer_delay.enabled` (owner decision 2026-10-11).
// Covers: off = old behaviour (no delay parameter, no storage, no sync UI, nothing folded); on = the delay is learned from the
// first answer (which is never drawn) and every later request carries it; first-open prompt once; folded until synced with the
// recommendation still showing; sync by tap (mocked endpoint: wait → one more tap → a number); ± adjust; persistence across a
// reload; storage blocked; a changed delay drops everything downloaded (no newer pitch stays, no false "추천이 갱신됐어요");
// buffering view; list without a live score; key info on the first screen at the five sizes (vertical scroll allowed since 2026-10-13); the PUBLIC build (VITE_PUBLISH_BASE) shows no delay
// UI and sends no delay even when a payload says enabled.
//   npm run build && node tests/viewer-delay-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const DIST = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const shots = process.argv[2]; if (shots) mkdirSync(shots, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let failures = 0;
const ok = (cond, message) => { if (cond) console.log('ok  ', message); else { failures++; console.error('FAIL', message); } };
const clone = (v) => JSON.parse(JSON.stringify(v));
const GAME = '/api/service/games/849843';
const FEATURES = (enabled, extra = {}) => ({ provisional: true, viewer_delay: { enabled, default_s: 30, sync_prompt: enabled, fold_actual_until_synced: enabled, ...extra },
  profile_reference: { enabled: false }, step5_expectation: { enabled: false } });
const STATE = (state, reason, label, message, shown, pitch_key) => ({ schema: 'pitcheezy-display-state-v1', state, reason, detail: null, label, message, pitch_key, recommendation_shown: shown, correction: null,
  feed: { ok: true, delayed: false, lag_s: 18.2, lag_max_s: 24, lag_measured: true, lag_threshold_s: 45, thresholds_provisional: true, last_ok_at: null, failing_since: null, failing_for_s: null, received_at: null, provider_order: 'ordered' },
  ledger: { status: 'recorded', outcome: 'first', why: null, store_ok: true } });
/** The live answer a viewer on `delay` gets: the fixture's earlier state (`live`) when 45 s or more behind, else the later one. */
function liveBody({ enabled = true, delay = 0, features = {}, base } = {}) {
  const body = clone(base ?? (delay >= 45 ? fx.live : fx.live_more));
  body.source = { kind: 'live', session_id: null }; body.receipts = null;
  body.display_state = STATE('recommended', 'ledger_recorded', '다음 공 추천', null, true, `849843:${delay >= 45 ? 7 : 9}:2`);
  body.features = FEATURES(enabled, features);
  body.viewer = { enabled, default_s: 30, provisional: true, delay_s: enabled ? delay : 0, max_delay_s: 120, buffering: false, wait_s: 0 };
  return body;
}
function bufferingBody(delay) {
  const body = liveBody({ delay });
  Object.assign(body, { pitches: [], pas: [], next: null, linescore: null, cutoff: { index: -1, pitch_key: null },
    current: { phase: null, situation: null, batter: null, pitcher: null, pa_key: null, context: null },
    display_state: STATE('waiting', 'buffering', '대기 중', '내 중계 시점까지 기다리는 중이에요 (약 23초).', false, null) });
  body.game = { ...body.game, away_score: null, home_score: null, inning: null, half: null, score_hidden: true, status: 'live', live: true };
  body.viewer = { ...body.viewer, buffering: true, wait_s: 23 };
  return body;
}

const browser = await chromium.launch();
async function open({ width = 390, height = 844, handler, dist = DIST, blockStorage = false, storage = null }) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  if (blockStorage) await context.addInitScript(() => { Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('blocked', 'SecurityError'); } }); });
  if (storage) await context.addInitScript((value) => { try { if (!sessionStorage.getItem('seeded')) { localStorage.setItem('pitcheezy.viewer', value); sessionStorage.setItem('seeded', '1'); } } catch { /* blocked */ } }, JSON.stringify(storage));
  const page = await context.newPage();
  const asks = [];
  await page.route('http://pz.test/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith('/api/') || url.pathname.includes('/v1/')) { asks.push(url); return handler(route, url, asks); }
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = []; page.on('pageerror', (e) => errors.push(String(e)));
  return { context, page, asks, errors };
}
/** A server that honours `?delay=` like the backend with the switch on (or ignores it when `enabled` is false). */
function server({ enabled = true, features = {}, tap = null, buffering = false } = {}) {
  const s = { enabled, features, tap, buffering, taps: [] };
  s.handler = async (route, url) => {
    const delay = s.enabled ? Math.min(120, Number(url.searchParams.get('delay') ?? 0)) : 0;
    if (url.pathname === `${GAME}/tap`) { s.taps.push(url); return route.fulfill(s.tap ? { json: s.tap(url, s.taps.length) } : { status: 404, json: { detail: 'off' } }); }
    if (url.pathname === GAME) {
      const body = s.buffering && delay > 0 ? bufferingBody(delay) : liveBody({ enabled: s.enabled, delay, features: s.features });
      const after = url.searchParams.get('after');
      if (after !== null) body.pitches = body.pitches.filter((p) => p.index > Number(after));
      return route.fulfill({ json: body });
    }
    if (url.pathname === '/api/service/dates') return route.fulfill({ json: { dates: ['2026-10-02'] } });
    if (url.pathname === '/api/service/games') {
      const card = { ...clone(fx.live.game), status: 'live', live: true, away_score: 2, home_score: 5, inning: 7, half: 'Top', date_kst: '2026-10-02' };
      const hidden = s.enabled && delay > 0;
      return route.fulfill({ json: { games: [hidden ? { ...card, away_score: null, home_score: null, inning: null, half: null, score_hidden: true } : card], live_ok: true, features: FEATURES(s.enabled) } });
    }
    return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
  };
  return s;
}
const gameAsks = (asks) => asks.filter((u) => u.pathname === GAME);
const delays = (asks) => gameAsks(asks).map((u) => u.searchParams.get('delay'));
const stored = (page) => page.evaluate(() => { try { return localStorage.getItem('pitcheezy.viewer'); } catch { return 'BLOCKED'; } });
/* same rule as static-publish-smoke.mjs (kept in step by hand) */
/** Game-screen fit. Until 2026-10-13 this asked for one screen with no scroll at all. The approved redesign (docs/decisions.md 2026-10-13,
 *  UI-CONNECTION-2026-10-13.md) lets the phone screen scroll vertically under a fixed bottom tab bar. What the old check protected
 *  still has to hold: no sideways scroll; at first paint (scrollY 0) the header (score, LIVE / state), the status line (late /
 *  disconnected alerts), the count board and the matchup lie fully above the tab bar and the zone (where the recommendation is drawn)
 *  starts above it; on a tall phone (>= 800 px) the whole core down to the pitch cards does (as ws-responsive-smoke); scrolled to the
 *  end, nothing is left under the tab bar. Returns '' when it fits, else what does not (printed with the check). */
const keyOnFirstScreen = (page) => page.evaluate(async () => {
  const bad = [], root = document.scrollingElement;
  if (root.scrollWidth > innerWidth + 1) bad.push(`sideways ${root.scrollWidth}>${innerWidth}`);
  window.scrollTo(0, 0); await new Promise((r) => requestAnimationFrame(() => r()));
  const nav = document.querySelector('.sv-bottom-nav'), nr = nav?.getBoundingClientRect();
  const fold = nr && nr.height && getComputedStyle(nav).position === 'fixed' ? nr.top : innerHeight;
  const box = (sel) => { const e = document.querySelector(sel); const r = e?.getBoundingClientRect(); return r && r.width && r.height ? r : null; };
  for (const sel of ['.sv-top', '.sv-zone']) if (!box(sel)) bad.push(`${sel} missing`);
  for (const sel of ['.sv-top', '.sv-notice.line', '.sv-board', '.sv-duel']) { const r = box(sel); if (r && r.bottom > fold + 1) bad.push(`${sel} bottom ${Math.round(r.bottom)} > ${Math.round(fold)}`); }
  const zone = box('.sv-zone'); if (zone && zone.top >= fold) bad.push(`.sv-zone top ${Math.round(zone.top)} >= ${Math.round(fold)}`);
  const strip = box('.sv-strip-shell'); if (innerHeight >= 800 && strip && strip.bottom > fold + 1) bad.push(`.sv-strip-shell bottom ${Math.round(strip.bottom)} > ${Math.round(fold)}`);
  window.scrollTo(0, root.scrollHeight); await new Promise((r) => requestAnimationFrame(() => r()));
  const fold2 = nav && nav.getBoundingClientRect().height ? nav.getBoundingClientRect().top : innerHeight;
  let low = 0, who = '';
  for (const el of document.querySelectorAll('.sv-page *')) { if (el.closest('.sv-bottom-nav, .sv-backdrop, .sr-only')) continue; const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
    if (!r.width || !r.height || cs.visibility === 'hidden' || cs.position === 'fixed') continue; if (r.bottom > low) { low = r.bottom; who = String(el.className || el.tagName); } }
  if (low > fold2 + 1) bad.push(`at the end ${who} bottom ${Math.round(low)} stays under the tab bar (${Math.round(fold2)})`);
  window.scrollTo(0, 0);
  return bad.join('; ');
}).then((bad) => { if (bad) console.error('     fit:', bad); return bad === ''; });
const overflow = (page) => page.evaluate(() => [document.scrollingElement.scrollHeight - window.innerHeight, document.scrollingElement.scrollWidth - window.innerWidth]);
const box = (page, sel) => page.evaluate((s) => { const r = document.querySelector(s)?.getBoundingClientRect(); return r ? [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 2) / 2).join(',') : 'none'; }, sel);
const lineSpill = (page) => page.evaluate(() => { const l = document.querySelector('.sv-notice.line'); const r = l.getBoundingClientRect();
  return [...l.querySelectorAll('.sv-line-text,.sv-line-time,button')].filter((e) => { const b = e.getBoundingClientRect(); return b.width && (b.right > r.right + 0.5 || b.bottom > r.bottom + 0.5); }).map((e) => e.className); });
const cards = (page) => page.locator('.sv-strip .sv-card:not(.pending):not(.sv-card-idle)');
const SYNC_WORDS = /중계 맞추기|맞추기|늦춰 보는 중|맞춘 뒤 표시/;

// ---------- 1. switched off (and an older server): the screen as before the feature ----------
for (const [name, make] of [['switch off', () => server({ enabled: false })], ['older server (no viewer, no features)', () => { const s = server({ enabled: false }); const h = s.handler;
  s.handler = (route, url) => url.pathname === GAME ? route.fulfill({ json: (() => { const b = liveBody({ enabled: false }); delete b.viewer; delete b.features; return b; })() }) : h(route, url); return s; }]]) {
  const s = make();
  const { context, page, asks, errors } = await open({ handler: s.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  await page.waitForTimeout(4500);
  ok(gameAsks(asks).length >= 2 && delays(asks).every((d) => d === null), `${name}: no request carries a delay (${gameAsks(asks).length} asks)`);
  ok(await page.locator('.sv-sync-open, .sv-fold, .sv-card.folded, .sv-sync').count() === 0 && !SYNC_WORDS.test(await page.locator('.sv-page').innerText()), `${name}: no sync entry, no prompt, nothing folded`);
  ok(/실시간으로 받는 중/.test(await page.locator('.sv-notice.line').innerText()), `${name}: the status line reads as before`);
  ok(await stored(page) === null, `${name}: nothing is stored on the device`);
  ok(asks.every((u) => !u.pathname.endsWith('/tap')) && errors.length === 0, `${name}: no tap request, no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 2. switched on: learned from the first answer, prompt once, folded until synced ----------
{
  // one tap is never enough here (the server asks for another); two taps are answered after one more "wait"
  let pairs = 0;
  const s = server({ tap: (url) => url.searchParams.get('ago').split(',').length < 2
    ? { status: 'wait', delay_s: null, hint: 'tap_again', message: '좋아요. 다음 공에서 한 번 더 눌러 주세요.', retry_s: 0.3, max_delay_s: 120 }
    : pairs++ < 1 ? { status: 'wait', delay_s: null, hint: null, message: '맞추는 중이에요. 잠시만 기다려 주세요.', retry_s: 0.3, max_delay_s: 120 }
    : { status: 'ok', delay_s: 33, hint: null, message: null, retry_s: null, max_delay_s: 120 } });
  const { context, page, asks, errors } = await open({ handler: s.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-sync').waitFor();
  ok(delays(asks)[0] === null && delays(asks)[1] === '30', `on: the first answer only tells the switch; the game is asked again at the default delay (${delays(asks).slice(0, 3).join(', ')})`);
  ok(/미트에 들어오는 순간 누르기/.test(await page.locator('.sv-sync-tap').innerText()) && /중계 맞추기/.test(await page.locator('.sv-sheet-head h2').innerText()), 'on: first open of a live game asks once to sync (big button)');
  ok(/아직 맞추지 않았어요. 지금은 30초 늦춰서 보여 드려요/.test(await page.locator('.sv-sync-status').innerText()), 'on: the sheet says it is not synced yet and shows the default');
  ok(!/delay|buffer|sync|viewer/i.test(await page.locator('.sv-sheet').innerText()), 'on: plain words only in the sheet');
  if (shots) await page.screenshot({ path: path.join(shots, '01-first-open-sync-prompt-390x844.png') });
  await page.locator('.sv-sync-foot [data-act="later"]').click();
  await page.locator('.sv-sync').waitFor({ state: 'detached' });
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  // Since the 2026-10-13 redesign the stage's idle "실제 투구" line (.sv-state-idle, which holds the .sv-fold button) is not drawn on the
  // game screen (display:none, decisions 2026-10-13): the folded state stays in the DOM, and what the viewer sees is checked instead —
  // no thrown pitch described on the stage, folded cards, and the status line's 중계 맞추기 entry (below).
  ok(await page.locator('.sv-fold').count() === 1 && (await page.locator('.sv-fold').textContent()).trim() === '중계 맞추기 후 표시'
    && !/km\/h|mph/.test(await page.locator('.sv-stage').innerText()) && await page.locator('.sv-zone-wrap .sv-actual').count() === 0,
    'unsynced: the actual pitch / result is folded ("중계 맞추기 후 표시"); nothing of it is drawn on the stage');
  ok(await page.locator('.sv-pitch-summary .sv-rec-row').count() >= 1, 'unsynced: the recommendation still shows');
  const n = await cards(page).count();
  ok(n >= 1 && await page.locator('.sv-strip .sv-card.folded').count() === n && /맞춘 뒤 표시/.test(await cards(page).first().innerText()), `unsynced: every thrown-pitch card is folded (${n})`);
  ok(!/km\/h|mph/.test(await page.locator('.sv-strip').innerText()), 'unsynced: no pitch speed or type on the cards');
  ok(/30초 늦춰 보는 중/.test(await page.locator('.sv-line-text').innerText()) && (await page.locator('.sv-sync-open').innerText()).trim() === '중계 맞추기', 'unsynced: the status line says 30초 늦춰 보는 중 and offers 중계 맞추기');
  ok(await page.locator('.sv-head-score strong').first().innerText() !== '–' && await page.locator('.sv-bso').count() === 1, 'unsynced: score and count stay');
  ok(await keyOnFirstScreen(page) && (await lineSpill(page)).length === 0, 'unsynced: key info on the first screen, nothing spills from the status line');
  if (shots) await page.screenshot({ path: path.join(shots, '02-unsynced-folded-row-with-recommendation-390x844.png') });
  await page.locator('.sv-inning').click();
  ok((await page.locator('.sv-pa-item:not(.open) .sv-pa-result').allInnerTexts()).every((t) => /^중계 맞추기 후 표시 · /.test(t)), 'unsynced: finished plate appearances do not say how they ended');
  await page.keyboard.press('Escape');
  ok(JSON.parse(await stored(page)).asked === true && JSON.parse(await stored(page)).synced === false, 'later: remembered as asked, still unsynced');
  await page.reload();
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  await page.waitForTimeout(800);
  ok(await page.locator('.sv-sync').count() === 0 && await page.locator('.sv-fold').count() === 1, 'reload: not asked a second time; still folded');
  // ---- sync by tap
  // the folded pitch leads to syncing (the folded row is not drawn since 2026-10-13; a folded card does the same job)
  await cards(page).first().click();
  await page.locator('.sv-sync').waitFor();
  ok(/미트에 들어오는 순간 누르기/.test(await page.locator('.sv-sync').innerText()) && !/km\/h|mph/.test(await page.locator('.sv-sync').innerText()), 'a folded pitch card opens the sync sheet (no pitch shown)');
  await page.locator('.sv-sync-tap').click();
  await page.waitForFunction(() => /한 번 더/.test(document.querySelector('.sv-sync-status')?.textContent ?? ''));
  ok(true, 'tap 1: "다음 공에서 한 번 더 눌러 주세요"');
  await page.waitForTimeout(3200);   // a double press within 3 s would be one tap for the server; a real second pitch comes later
  await page.locator('.sv-sync-tap').click();
  await page.waitForFunction(() => /내 화면이 약 33초 늦어요/.test(document.querySelector('.sv-sync-status')?.textContent ?? ''), null, { timeout: 8000 });
  const last = s.taps[s.taps.length - 1];
  ok(last.searchParams.get('ago').split(',').length === 2 && last.searchParams.get('current') === '30' && last.searchParams.get('sure') === '0', `tap 2: both taps are sent as ages (${last.search})`);
  ok(Number(last.searchParams.get('ago').split(',')[0]) > 3, 'tap: the age of a tap grows while the client keeps asking (no clock comparison)');
  await page.waitForFunction(() => document.querySelectorAll('.sv-strip .sv-card').length > 0);
  ok(delays(asks).at(-1) === '33', 'synced: requests carry the measured delay');
  ok(await page.locator('.sv-sync').count() === 1, 'synced: the sheet stays open across the reload of the game');
  if (shots) await page.screenshot({ path: path.join(shots, '04-adjust-sheet-after-sync-390x844.png') });
  // ---- ± adjust
  await page.locator('.sv-sync-adjust button', { hasText: '+1초' }).click();
  await page.waitForFunction(() => /34/.test(document.querySelector('.sv-sync-now')?.textContent ?? ''));
  await page.waitForTimeout(400);
  ok(delays(asks).at(-1) === '34', '+1: asked again at 34 s');
  await page.locator('.sv-sync-adjust button', { hasText: '-5초' }).click();
  await page.waitForTimeout(500);
  ok(delays(asks).at(-1) === '29' && /내 화면이 약 29초 늦어요/.test(await page.locator('.sv-sync-status').innerText()), '-5: asked again at 29 s');
  await page.locator('.sv-sync-foot [data-act="later"]').click();
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  ok(await page.locator('.sv-fold, .sv-card.folded').count() === 0 && /km\/h/.test(await page.locator('.sv-strip').innerText()), 'synced: nothing is folded any more');
  ok(/29초 늦춰 보는 중/.test(await page.locator('.sv-line-text').innerText()) && (await page.locator('.sv-sync-open').innerText()).trim() === '맞추기', 'synced: status line "29초 늦춰 보는 중 · 맞추기"');
  if (shots) await page.screenshot({ path: path.join(shots, '03-synced-live-390x844.png') });
  // ---- persistence
  const before = gameAsks(asks).length;
  await page.reload();
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  const again = delays(asks).slice(before);
  ok(again[0] === null && again[1] === '29' && await page.locator('.sv-sync').count() === 0 && await page.locator('.sv-fold').count() === 0, `reload: the device keeps 29 s, stays synced, no prompt (${again.slice(0, 3).join(', ')})`);
  // ---- a longer delay goes BACK: nothing newer stays, and no false change note
  const many = await cards(page).count(); const head = await page.locator('.sv-pitch-summary').getAttribute('data-pitch');
  await page.locator('.sv-sync-open').click();
  for (let i = 0; i < 4; i++) await page.locator('.sv-sync-adjust button', { hasText: '+5초' }).click();
  await page.waitForTimeout(600);
  await page.locator('.sv-sync-foot [data-act="later"]').click();
  await page.locator('.sv-pitch-summary[data-pitch="849843:7:2"]').waitFor();
  const lastAsk = gameAsks(asks).at(-1);
  ok(delays(asks).includes('49') && head === '849843:9:2', 'delay 29 → 49: the game is asked again at 49 s');
  ok(gameAsks(asks).filter((u) => u.searchParams.get('delay') === '49')[0].searchParams.get('after') === null, 'delay change: the first request at the new delay is a FULL one (no cursor kept)');
  const shownKeys = await page.evaluate(() => [...document.querySelectorAll('.sv-strip .sv-card')].length);
  ok((await page.locator('.sv-state-note').innerText()).trim() === '' && !/갱신됐어요|표시하지 않아요/.test(await page.locator('.sv-pitch-summary').innerText()), 'delay change: no false "추천이 갱신됐어요"');
  await page.locator('.sv-inning').click();
  const pas = await page.locator('.sv-sheet select option').count();
  await page.keyboard.press('Escape');
  ok(pas >= 1 && shownKeys >= 1 && lastAsk !== undefined, `delay change: the screen now holds the earlier state only (${fx.live.pitches.length} pitches served, was ${fx.live_more.pitches.length}; cards before ${many})`);
  ok(await page.evaluate((n) => !document.body.innerText.includes(n), '\u0000') && errors.length === 0, `on: no page errors ${errors.join(' | ')}`);
  // ---- "중계 없이 보기"
  await page.locator('.sv-sync-open').click();
  await page.locator('.sv-sync-foot [data-act="none"]').click();
  await page.waitForTimeout(500);
  ok(delays(asks).at(-1) === '0' && JSON.parse(await stored(page)).synced === true && JSON.parse(await stored(page)).delay_s === 0, '중계 없이 보기: delay 0, counts as synced');
  await page.locator('.sv-sync-foot [data-act="later"]').click();
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  ok(/실시간으로 받는 중/.test(await page.locator('.sv-line-text').innerText()) && await page.locator('.sv-fold').count() === 0, '중계 없이 보기: the line reads 실시간으로 받는 중, nothing folded');
  await context.close();
}

// ---------- 2b. review round 1: another tab, a narrow sheet, a server that clamps without saying its limit ----------
{
  const s = server();
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  await context.addInitScript((value) => { if (!sessionStorage.getItem('seeded')) { if (!localStorage.getItem('pitcheezy.viewer')) localStorage.setItem('pitcheezy.viewer', value); sessionStorage.setItem('seeded', '1'); } }, JSON.stringify({ delay_s: 30, synced: true, asked: true }));
  const tabs = [];
  for (const name of ['A', 'B']) {
    const page = await context.newPage(); const asks = [];
    await page.route('http://pz.test/**', async (route) => { const url = new URL(route.request().url());
      if (url.pathname.startsWith('/api/')) { asks.push(url); return s.handler(route, url); }
      const file = path.join(DIST, url.pathname); const target = existsSync(file) && statSync(file).isFile() ? file : path.join(DIST, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' }); });
    await page.goto('http://pz.test/watch/849843');
    await page.locator('.sv-pitch-summary[data-pitch="849843:9:2"]').waitFor();
    tabs.push({ page, asks });
  }
  const [a, b] = tabs;
  await a.page.locator('.sv-sync-open').click();
  for (let i = 0; i < 4; i++) await a.page.locator('.sv-sync-adjust button', { hasText: '+5초' }).click();   // 30 → 50: further back
  await b.page.locator('.sv-pitch-summary[data-pitch="849843:7:2"]').waitFor({ timeout: 6000 }).then(() => ok(true, 'second tab: follows the longer delay set in the first tab (the later moment is dropped)'), () => ok(false, 'second tab: follows the longer delay set in the first tab'));
  await b.page.waitForTimeout(2500);
  ok(delays(b.asks).at(-1) === '50' && /50초 늦춰 보는 중/.test(await b.page.locator('.sv-line-text').innerText()), `second tab: asks at 50 s from then on (${delays(b.asks).slice(-3).join(', ')})`);
  ok(gameAsks(b.asks).filter((u) => u.searchParams.get('delay') === '50')[0].searchParams.get('after') === null, 'second tab: the first request at the new delay is a full one');
  // F1: quick successive changes in the first tab: the second tab ends at the LAST one, not at one in between
  for (let i = 0; i < 6; i++) await a.page.locator('.sv-sync-adjust button', { hasText: '-5초' }).click({ delay: 20 });   // 50 → 20, six writes
  await b.page.waitForFunction(() => /20초 늦춰 보는 중/.test(document.querySelector('.sv-line-text')?.textContent ?? ''), null, { timeout: 8000 }).then(() => ok(true, 'second tab: after six quick changes it shows the last delay (20 s)'), () => ok(false, 'second tab: after six quick changes it shows the last delay (20 s)'));
  await b.page.waitForTimeout(4500);
  ok(delays(b.asks).slice(-2).every((d) => d === '20') && JSON.parse(await stored(b.page)).delay_s === 20, `second tab: keeps asking at 20 s and did not write an older value back (${delays(b.asks).slice(-3).join(', ')})`);
  // F2: a list page in a third tab follows too
  const list = await context.newPage(); const listAsks = [];
  await list.route('http://pz.test/**', async (route) => { const url = new URL(route.request().url());
    if (url.pathname.startsWith('/api/')) { listAsks.push(url); return s.handler(route, url); }
    const file = path.join(DIST, url.pathname); const target = existsSync(file) && statSync(file).isFile() ? file : path.join(DIST, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' }); });
  await list.goto('http://pz.test/watch?date=2026-10-02');
  await list.locator('.sv-list .sv-card').first().waitFor();
  const listDelays = () => listAsks.filter((u) => u.pathname === '/api/service/games').map((u) => u.searchParams.get('delay'));
  ok(listDelays().at(-1) === '20', `list tab: asks at the device's delay (${listDelays().join(', ')})`);
  await a.page.locator('.sv-sync-adjust button', { hasText: '+5초' }).click();
  await list.waitForTimeout(1500);
  ok(listDelays().at(-1) === '25' && (await list.locator('.sv-card-team b').allInnerTexts()).every((t) => t.trim() === ''), `list tab: follows a delay changed in another tab (${listDelays().slice(-2).join(', ')})`);
  await context.close();
}
{
  const s = server();
  const { context, page, errors } = await open({ width: 320, height: 568, handler: s.handler, storage: { delay_s: 30, synced: true, asked: true } });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  await page.locator('.sv-sync-open').click();
  await page.locator('.sv-sync-adjust').waitFor();
  const clipped = await page.evaluate(() => { const now = document.querySelector('.sv-sync-now'); const r = now.getBoundingClientRect();
    const hit = [...document.querySelectorAll('.sv-sync-adjust button')].some((b) => { const q = b.getBoundingClientRect(); return q.left < r.right - 0.5 && q.right > r.left + 0.5; });
    return hit || now.scrollWidth > now.clientWidth + 1 || document.querySelector('.sv-sheet-body').scrollWidth > document.querySelector('.sv-sheet-body').clientWidth + 1; });
  ok(!clipped && /30/.test(await page.locator('.sv-sync-now').innerText()), '320 px: the seconds between the ± buttons are not clipped or covered');
  if (shots) await page.screenshot({ path: path.join(shots, '06-adjust-sheet-320x568.png') });
  ok(errors.length === 0, `320 px sheet: no page errors ${errors.join(' | ')}`);
  await context.close();
}
{
  const s = server(); const base = s.handler;   // serves at most 20 s and does not say so (no max_delay_s)
  s.handler = (route, url) => { if (url.pathname !== GAME) return base(route, url);
    const asked = Number(url.searchParams.get('delay') ?? 0); const body = liveBody({ delay: Math.min(20, asked) }); body.viewer.max_delay_s = null;
    const after = url.searchParams.get('after'); if (after !== null) body.pitches = body.pitches.filter((p) => p.index > Number(after));
    return route.fulfill({ json: body }); };
  const { context, page, asks, errors } = await open({ handler: s.handler, storage: { delay_s: 50, synced: true, asked: true } });
  await page.goto('http://pz.test/watch/849843');
  const loaded = await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor({ timeout: 8000 }).then(() => true, () => false);
  ok(loaded && await page.locator('.sv-alert').count() === 0, 'unstated clamp: the game is shown, not an error screen');
  await page.waitForTimeout(2500);
  ok(loaded && /20초 늦춰 보는 중/.test(await page.locator('.sv-line-text').innerText()) && delays(asks).at(-1) === '20', `unstated clamp: the screen takes the delay the server serves (${delays(asks).slice(-3).join(', ')})`);
  ok(errors.length === 0, `unstated clamp: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 3. sync_prompt off: no automatic question, the entry stays ----------
{
  const s = server({ features: { sync_prompt: false } });
  const { context, page, errors } = await open({ handler: s.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  await page.waitForTimeout(1200);
  ok(await page.locator('.sv-sync').count() === 0 && await page.locator('.sv-sync-open').count() === 1 && await page.locator('.sv-fold').count() === 1, 'sync_prompt off: no automatic prompt; manual entry and fold remain');
  ok(errors.length === 0, `sync_prompt off: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 4. storage blocked ----------
{
  const s = server({ tap: () => ({ status: 'ok', delay_s: 41, hint: null, message: null, retry_s: null, max_delay_s: 120 }) });
  const { context, page, asks, errors } = await open({ handler: s.handler, blockStorage: true });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-sync').waitFor();
  ok(await stored(page) === 'BLOCKED' && delays(asks).includes('30'), 'storage blocked: the screen loads at the default delay and asks to sync');
  await page.locator('.sv-sync-tap').click();
  await page.waitForFunction(() => /내 화면이 약 41초 늦어요/.test(document.querySelector('.sv-sync-status')?.textContent ?? ''));
  await page.locator('.sv-sync-foot [data-act="later"]').click();
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  await page.waitForTimeout(2500);
  ok(delays(asks).at(-1) === '41' && await page.locator('.sv-fold').count() === 0 && /41초 늦춰 보는 중/.test(await page.locator('.sv-line-text').innerText()), 'storage blocked: sync works for this page (41 s, unfolded)');
  ok(errors.length === 0, `storage blocked: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 5. buffering ----------
{
  const s = server({ buffering: true });
  const { context, page, errors } = await open({ handler: s.handler, storage: { delay_s: 30, synced: true, asked: true } });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary.sv-state[data-state="waiting"]').waitFor();
  const text = await page.locator('.sv-page').innerText();
  ok(/내 중계 시점까지 기다리는 중이에요 \(약 23초\)/.test(text) && /내 중계 시점이 되면 보여 드려요/.test(text), 'buffering: one calm sentence from the server');
  ok((await page.locator('.sv-head-score strong').allInnerTexts()).every((t) => t.trim() === '–') && await cards(page).count() === 0 && await page.locator('.sv-rec-row').count() === 0, 'buffering: no score, no pitch, no recommendation');
  ok(/LIVE/.test(await page.locator('.sv-live').innerText()) && await keyOnFirstScreen(page), 'buffering: still a live screen, key info on the first screen');
  if (shots) await page.screenshot({ path: path.join(shots, '05-buffering-390x844.png') });
  s.buffering = false;
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor({ timeout: 6000 });
  ok(true, 'buffering → the game appears by polling alone');
  ok(errors.length === 0, `buffering: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 6. the list ----------
{
  const s = server();
  const { context, page, asks, errors } = await open({ handler: s.handler, storage: { delay_s: 30, synced: true, asked: true } });
  await page.goto('http://pz.test/watch?date=2026-10-02');
  await page.locator('.sv-list .sv-card').first().waitFor();
  const list = asks.filter((u) => u.pathname === '/api/service/games').map((u) => u.searchParams.get('delay'));
  ok(list[0] === null && list[1] === '30', `list: asked again with the device's delay once the switch is known (${list.join(', ')})`);
  ok((await page.locator('.sv-card-team b').allInnerTexts()).every((t) => t.trim() === '') && /LIVE/.test(await page.locator('.sv-status').innerText()), 'list: a live card carries no score for a delayed viewer');
  ok(errors.length === 0, `list: no page errors ${errors.join(' | ')}`);
  await context.close();
  const off = server({ enabled: false });
  const second = await open({ handler: off.handler });
  await second.page.goto('http://pz.test/watch?date=2026-10-02');
  await second.page.locator('.sv-list .sv-card').first().waitFor();
  ok(second.asks.filter((u) => u.pathname === '/api/service/games').every((u) => u.searchParams.get('delay') === null) && (await second.page.locator('.sv-card-team b').allInnerTexts()).join('') === '25', 'list, switch off: no delay, score shown as before');
  await second.context.close();
}

// ---------- 7. layout: the entry and the folded row move nothing ----------
for (const [width, height] of [[390, 844], [390, 664], [375, 553], [360, 640], [320, 568]]) {
  const measure = async (srv, storage) => {
    const { context, page } = await open({ width, height, handler: srv.handler, storage });
    await page.goto('http://pz.test/watch/849843');
    await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
    await page.waitForTimeout(300);
    const out = { over: await overflow(page), slot: await box(page, '.sv-pitch-summary'), line: await box(page, '.sv-notice.line'), spill: await lineSpill(page) };
    await context.close();
    return out;
  };
  const off = await measure(server({ enabled: false }), null);
  const synced = await measure(server(), { delay_s: 30, synced: true, asked: true });
  const folded = await measure(server(), { delay_s: null, synced: false, asked: true });
  for (const [name, m] of [['synced', synced], ['unsynced (folded)', folded]]) {
    ok(m.over[0] <= Math.max(1, off.over[0]) && m.over[1] <= 1, `${width}x${height} ${name}: no more vertical scroll than with the feature off (${m.over[0]} vs ${off.over[0]}), none sideways`);
    ok(m.slot === off.slot && m.line === off.line, `${width}x${height} ${name}: status slot and status line where they are with the feature off (${m.slot})`);
    ok(m.spill.length === 0, `${width}x${height} ${name}: the status line holds its words and the sync entry ${m.spill.join(',')}`);
  }
}

// ---------- 8. the PUBLIC build never shows a delay control, whatever the payload says ----------
{
  const out = mkdtempSync(path.join(os.tmpdir(), 'pz-public-'));
  try {
    execFileSync('npx', ['vite', 'build', '--outDir', out, '--emptyOutDir', '--logLevel', 'error'], { cwd: path.join(here, '..'), env: { ...process.env, VITE_PUBLISH_BASE: '.' }, stdio: 'pipe' });
    const bundle = readdirSync(path.join(out, 'assets')).filter((f) => f.endsWith('.js')).map((f) => readFileSync(path.join(out, 'assets', f), 'utf8')).join('\n');
    for (const words of ['미트에 들어오는 순간 누르기', '중계 없이 보기', 'pitcheezy.viewer', '/tap?ago=']) ok(!bundle.includes(words), `public build: the bundle does not contain "${words}"`);
    console.log(`note public build: fold label "중계 맞추기 후 표시" in bundle: ${bundle.includes('중계 맞추기 후 표시')} (unreachable at run time: viewer is always null there)`);
    const published = async (route, url) => {
      const stamp = { contract: 'v1', sequence: 1, published_at: new Date().toISOString() };
      const body = url.pathname.endsWith('/v1/manifest.json') ? { publish_interval_s: 10, live_dates: [], publish: stamp }
        : url.pathname.endsWith('/v1/health.json') ? { publish: stamp }
        : url.pathname.endsWith('/v1/game/849843.json') ? { ...liveBody({ enabled: true }), publish: { ...stamp, final: false } } : null;
      return route.fulfill({ status: body ? 200 : 404, contentType: 'application/json', body: JSON.stringify(body ?? {}) });
    };
    const { context, page, asks, errors } = await open({ handler: published, dist: out, storage: { delay_s: 45, synced: false, asked: false } });
    await page.goto('http://pz.test/watch/849843');
    await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
    await page.waitForTimeout(4500);
    ok(asks.some((u) => u.pathname.endsWith('/v1/game/849843.json')) && asks.every((u) => !u.pathname.startsWith('/api/') && !u.searchParams.has('delay')), 'public build: payload says enabled, yet only published files are read and none carries a delay');
    ok(await page.locator('.sv-sync-open, .sv-fold, .sv-card.folded, .sv-sync').count() === 0 && !SYNC_WORDS.test(await page.locator('.sv-page').innerText()), 'public build: no sync entry, no prompt, nothing folded');
    ok(asks.every((u) => !u.pathname.endsWith('/tap')) && errors.length === 0, `public build: no tap request, no page errors ${errors.join(' | ')}`);
    await context.close();
  } finally { rmSync(out, { recursive: true, force: true }); }
}

await browser.close();
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

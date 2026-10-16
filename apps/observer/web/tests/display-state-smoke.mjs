// Live display state (contract 2026-10-11, `display_state` schema pitcheezy-display-state-v1) on the fan game screen.
// Canned answers on the built dist/ (no server, no model, no network): a live payload is made from the real-game fixture
// (tests/fixtures/service-849843.json `live` / `live_more`) by marking it `source.kind = 'live'` and attaching a display state.
// Covers: every state and sub-reason with the server's copy, one state at a time, no reason codes on screen; the D1 notes
// (server `replaced` / `withdrawn`, and the client guard when the server sends none); the reserved status line (feed late with
// the measured lag, feed down, client fetch failing / timing out) and recovery by polling alone (no reload); first-load retry;
// unconfirmed pitch card and corrected tag; receipts.version change → full refetch; pregame / final; no layout jump between
// states at 390x844, 390x664, 375x553, 360x640 and 320x568 (390x844 core above fold; smaller phones scroll); absent / unknown
// display_state → the screen as before the contract. RC 2026-10-11: (12) a screen opened before play starts itself for every
// pre-play shape; (13) between innings nothing of the finished half is shown as current; (14) a past pitch shows the recommendation
// recorded before it next to the actual pitch, a calm line when there is none, nothing for an unconfirmed pitch; 확인 중 copy has
// none of 자료/시각/순서/격리/스냅샷/원장.
//   npm run build && node tests/display-state-smoke.mjs [screenshotDir]
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
const GAME = '/api/service/games/849843';

// ---------- payload builders (synthetic display states on real pitches) ----------
// 확인 중 family (server copy rewritten 2026-10-11 for a beginner fan): one short calm sentence, none of the words in TECH.
const UNSURE = '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.';
const UNSURE_PITCH = '방금 던진 공을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.';
const TECH = /자료|시각|순서|격리|스냅샷|원장/;
const SCORING = /적중|정확도|맞았|틀렸|맞음|틀림|일치|성공률|[✓✔✗✘○×]/;   // a past recommendation is never presented as a hit / miss
const COPY = {  // (label, message) exactly as the backend authors them (display_state.COPY)
  'recommended/ledger_recorded': ['다음 공 추천', null],
  'recommended/ledger_conflict_first_kept': ['다음 공 추천', null],
  'recommended/shown_unrecorded_no_ledger': ['다음 공 추천', null],
  'unconfirmed/stale_provider_stamp': ['확인 중', UNSURE],
  'unconfirmed/unorderable_provider_stamp': ['확인 중', UNSURE],
  'unconfirmed/same_stamp_conflict_provider_stamp': ['확인 중', UNSURE],
  'unconfirmed/provider_order_unknown': ['확인 중', UNSURE],
  'unconfirmed/prior_pitch_unconfirmed': ['확인 중', UNSURE_PITCH],
  'unconfirmed/receipts_unreadable': ['확인 중', UNSURE],
  'unsupported/unknown_pitcher': ['추천 없음', '이 투수는 2025년까지 쌓인 기록이 없어 추천하지 않아요.'],
  'unsupported/no_supported_pitch_types': ['추천 없음', '이 투수가 이 타석 방향에 던진 구종 기록이 부족해 추천하지 않아요.'],
  'unsupported/at_bat_rejected': ['추천 없음', '이 타석은 앞선 공 기록을 이어 볼 수 없어 추천을 쉬어요.'],
  'unsupported/policy_refused': ['추천 없음', '이 상황은 추천 대상이 아니에요.'],
  'unsupported/game_type_out_of_scope': ['추천 없음', '이 경기는 추천 대상이 아니에요.'],
  'feed_delayed/feed_lag_above_threshold': ['자료 늦음', '경기 자료가 52초 넘게 늦게 도착했어요. 추천이 실제 경기보다 늦을 수 있어요.'],
  'disconnected/poll_failed_last_snapshot_only': ['연결 끊김', '경기 자료 연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.'],
  'ledger_unavailable/write_failed': ['추천 보류', '추천 기록을 저장하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'],
  'ledger_unavailable/display_history_unreadable': ['추천 보류', '앞서 보여 드린 추천 기록을 확인하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'],
  'ledger_unavailable/store_open_failed': [null, null],   // server copy not fixed here: the screen falls back to its own words for the state
  'waiting/between_innings': ['공수 교대', '공수 교대 중이에요.'],
  'waiting/resolving': ['결과 확인 중', '방금 공의 결과를 기다리고 있어요.'],
  'waiting/computing': ['계산 중', '추천을 계산하고 있어요.'],
  'waiting/no_pending_pitch': ['대기 중', '다음 공을 기다리고 있어요.'],
  'waiting/buffering': ['대기 중', '내 중계 시점까지 기다리고 있어요.'],
  'unavailable/recommender_off': ['추천 꺼짐', '이 실행에서는 실시간 추천을 제공하지 않아요.'],
  'unavailable/recommender_unavailable': ['추천 준비 안 됨', '추천 모델을 준비하지 못했어요.'],
  'unavailable/compute_failed': ['추천 없음', '이 상황의 추천을 계산하지 못했어요.'],
  'pregame/scheduled': ['경기 전', '경기 시작 전이에요.'],
  'pregame/not_live': ['진행 중 아님', '지금은 경기가 진행 중이 아니에요.'],
  'final/game_over': ['경기 종료', '경기가 끝났어요.'],
};
const SHOWN = new Set(['recommended', 'feed_delayed']);
const PREV = { pitch_type: 'SL', pitch_label: '슬라이더', zone_id: 'low_left', zone_label: '낮은 왼쪽' };  // catcher's view → shown as '낮은 왼쪽'
const FIX = { replaced: '상황이나 기록이 바뀌어 추천이 달라졌어요.', withdrawn: '조금 전에 보여 드린 추천은 지금 표시하지 않아요.' };
const FALLBACK = { label: '추천 보류', message: '추천 기록을 저장하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.' };  // api.ts STATE_COPY
let serial = 100;
const NO_CONTEXT = { basis: 'event_prefix_current_feed', batter: { at_bats: null, hits: null, on_base: null }, pitcher: { pitches: null, strikeouts: null, earned_runs: null }, next_batters: [] };
const LINESCORE = { basis: 'revealed_events', innings: [{ inning: 1, away: 0, home: 2 }, { inning: 2, away: 0, home: null }, ...[3, 4, 5, 6, 7, 8, 9].map((inning) => ({ inning, away: null, home: null }))],
  teams: { away: { runs: 0, hits: 1, errors: 0, bb_hbp: 0 }, home: { runs: 2, hits: 3, errors: 0, bb_hbp: 1 } }, current: { inning: 2, half: 'Top' } };
function display(id, { shown = SHOWN.has(id.split('/')[0]), correction = null, feed = {}, pitch_key } = {}) {
  const [state, reason] = id.split('/');
  const [label, message] = COPY[id];
  return { schema: 'pitcheezy-display-state-v1', state, reason, detail: `diag:${reason}/ledger provider stamp`, label, message,
    pitch_key: pitch_key === undefined ? `849843:${serial++}:2` : pitch_key, recommendation_shown: shown,
    correction: correction && { kind: correction, message: FIX[correction], previous: PREV, previous_decided: '2026-10-11T01:02:03Z' },
    feed: { ok: state !== 'disconnected', delayed: state === 'feed_delayed', lag_s: state === 'feed_delayed' ? 52.9 : 18.2, lag_max_s: state === 'feed_delayed' ? 71.0 : 24.0, lag_measured: true, lag_threshold_s: 45.0,
      thresholds_provisional: true, last_ok_at: '2026-10-11T01:02:03Z', failing_since: null, failing_for_s: null, received_at: '2026-10-11T01:02:03Z', provider_order: 'ordered', ...feed },
    ledger: { status: state === 'ledger_unavailable' ? 'withheld' : 'recorded', outcome: 'first', why: state === 'ledger_unavailable' ? reason : null, store_ok: state !== 'ledger_unavailable' } };
}
/** A live answer: the fixture's 31 real pitches, `source.kind = 'live'`, and the given display state. As the server does, a state
 *  that shows nothing carries an absent `next.rec`, and an inning break / pending result carries no `next` at all. */
function live(id, opts = {}) {
  const body = clone(opts.base ?? fx.live);
  body.source = { kind: 'live', session_id: null };
  body.receipts = null;
  body.display_state = display(id, opts);
  const [state, reason] = id.split('/');
  if (opts.targets) body.next.rec.candidates.forEach((c, i) => { c.target = { x: [-0.55, 0.55, 0][i], z: opts.targets[i] }; });
  if (reason === 'resolving') { body.current.phase = reason; body.next = null; }
  else if (reason === 'between_innings') {   // the real shape (replay of game 849831): no situation, nobody at the plate, no score on the card
    body.current = { phase: 'between_innings', situation: null, batter: null, pitcher: null, pa_key: null, context: clone(NO_CONTEXT) };
    body.game = { ...body.game, away_score: null, home_score: null, inning: null, half: null };
    body.linescore = clone(opts.linescore ?? LINESCORE); body.next = null;
  }
  else if (!body.display_state.recommendation_shown) body.next = { pitch_key: body.display_state.pitch_key, rec: { provenance: 'absent', status: 'missing', reason: body.display_state.message, recorded_at: null, candidates: [], policy: null } };
  if (state === 'pregame') {
    body.game = { ...body.game, status: 'scheduled', status_text: 'Scheduled', live: false, away_score: null, home_score: null, inning: null, half: null };
    body.pitches = []; body.pas = []; body.cutoff = { index: -1, pitch_key: null }; body.next = null;
    body.current = { phase: 'pregame', situation: null, batter: null, pitcher: null, pa_key: null };
  }
  if (state === 'final') { body.game = { ...body.game, status: 'final', status_text: 'Final', live: false }; body.current = null; body.next = null; }
  if (opts.names) return JSON.parse(JSON.stringify(body).replaceAll('Nico Hoerner', opts.names[0]).replaceAll('Michael King', opts.names[1]));
  return body;
}
/** Server rule for `?after=N`: newer pitches, plus older ones whose receipt is unconfirmed or corrected. */
function answer(body, url) {
  const after = url.searchParams.get('after');
  if (after === null) return body;
  const out = clone(body);
  out.pitches = out.pitches.filter((p) => p.index > Number(after) || (p.receipt && (p.receipt.status !== 'received' || p.receipt.corrected)));
  return out;
}

const browser = await chromium.launch();
async function open({ width, height, handler }) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  const asks = [];
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== 'http://pz.test') return route.abort('blockedbyclient');
    if (url.pathname.startsWith('/api/')) { asks.push(url); return handler(route, url, asks); }
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = []; page.on('pageerror', (e) => errors.push(String(e)));
  return { context, page, asks, errors };
}
/** One mutable answer per page: `server.body` is what the next poll gets. */
function serve() {
  const server = { body: null, down: false, hang: false, hung: [], delay: 0 };
  server.handler = async (route, url) => {
    if (url.pathname !== GAME) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
    if (server.hang) { server.hung.push(route); return undefined; }   // never answered: the client must time out by itself
    if (server.down) return route.abort('internetdisconnected');
    const body = answer(server.body, url);
    if (server.delay) await new Promise((r) => setTimeout(r, server.delay));
    return route.fulfill({ json: body }).catch(() => {});
  };
  return server;
}
const slot = (page) => page.locator('.sv-pitch-summary');
/** A wait that times out is a FAIL line, not a crash of the whole run. */
async function waited(message, wait) { try { await wait(); ok(true, message); return true; } catch (e) { ok(false, `${message} — ${String(e).split('\n')[0].slice(0, 160)}`); return false; } }
/** Words of the status line: the live-region text plus the (aria-hidden) time stamp, without the retry control. */
const lineWords = (page) => page.evaluate(() => { const l = document.querySelector('.sv-notice.line'); const t = l.querySelector('.sv-line-text').cloneNode(true); t.querySelectorAll('.sr-only').forEach((e) => e.remove()); return [t.textContent, l.querySelector('.sv-line-time')?.textContent ?? ''].join(' ').replace(/\s+/g, ' ').trim(); });
/** Everything a fan can read or hear: visible text plus aria-label and title attributes. */
const spoken = (page) => page.evaluate(() => [document.querySelector('.sv-page').innerText, ...[...document.querySelectorAll('.sv-page [aria-label], .sv-page [title]')].map((e) => `${e.getAttribute('aria-label') ?? ''} ${e.getAttribute('title') ?? ''}`)].join(' \n '));
const boxes = (page) => page.evaluate(() => Object.fromEntries(['.sv-notice.line', '.sv-board', '.sv-duel', '.sv-zrow', '.sv-zrow .sv-zone-wrap', ].map((sel) => {
  const r = document.querySelector(sel)?.getBoundingClientRect(); return [sel, r ? [r.left + scrollX, r.top + scrollY, r.width, r.height].map((v) => Math.round(v * 2) / 2).join(',') : 'none']; })));
const sameBoxes = (a, b) => Object.keys(a).filter((k) => a[k] !== b[k]).map((k) => `${k}: ${a[k]} → ${b[k]}`);
/** Nothing inside the fixed-height slot or the status line spills out of it, and the page has no horizontal scroll. */
const spill = (page) => page.evaluate(() => {
  const out = [];
  for (const sel of ['.sv-pitch-summary.sv-state', '.sv-notice.line']) {
    const box = document.querySelector(sel); if (!box) continue;
    const r = box.getBoundingClientRect();
    for (const el of box.querySelectorAll('*')) { if (el.classList.contains('sr-only') || el.closest('.sr-only')) continue; const e = el.getBoundingClientRect(); if (!e.width || !e.height) continue;
      if (e.bottom > r.bottom + 0.5 || e.right > r.right + 0.5 || e.left < r.left - 0.5) out.push(`${sel} > ${el.className || el.tagName}`); }
  }
  if (document.scrollingElement.scrollWidth > window.innerWidth + 1) out.push('page scrolls sideways');
  return out;
});
const fitsWidth = (page) => page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const fitsViewport = (page) => page.evaluate(() => {
  const root = document.scrollingElement;
  const core = document.querySelector('.sv-strip-shell')?.getBoundingClientRect();
  return root.scrollWidth <= innerWidth + 1 && (innerWidth !== 390 || innerHeight < 800 || !core || core.bottom + scrollY <= innerHeight + 1);
});
const JARGON = /ledger|provider|stamp|diag:|receipt|unconfirmed|recommended|poll_failed|write_failed|[a-z]+_[a-z]+/;
/** Wait until the screen shows the answer carrying this pitch_key (set by the poll, no reload). */
async function shown(page, body, timeout = 6000) {
  const d = { ...body.display_state, label: body.display_state.label ?? FALLBACK.label, message: body.display_state.message ?? (body.display_state.state === 'ledger_unavailable' ? FALLBACK.message : null) };
  await page.locator(`.sv-pitch-summary.sv-state[data-state="${d.state}"]${d.pitch_key ? `[data-pitch="${d.pitch_key}"]` : ''}`).first().waitFor({ timeout });
  await page.waitForFunction(([label, message]) => { const s = document.querySelector('.sv-pitch-summary.sv-state'); return !!s && s.querySelector('.sv-chip')?.textContent.includes(label) && (message === null || s.textContent.includes(message)); }, [d.label, d.message], { timeout });
  await page.waitForTimeout(120);
}

// A bounded rerun can select completed test sections; the default always runs every assertion.
const SECTION_START = Number(process.env.PZ_TEST_START ?? 1), SECTION_END = Number(process.env.PZ_TEST_END ?? 14);
if (!Number.isInteger(SECTION_START) || !Number.isInteger(SECTION_END) || SECTION_START < 1 || SECTION_END > 14 || SECTION_START > SECTION_END) throw Error('invalid test section range');
const VIEWPORTS = [[390, 844], [390, 664], [375, 553], [360, 640], [320, 568]];
const VIEWPORT = process.env.PZ_TEST_VIEWPORT;
if (VIEWPORT && !VIEWPORTS.some(v => v.join('x') === VIEWPORT)) throw Error('invalid section 1 viewport');
console.log(`display state sections ${SECTION_START}–${SECTION_END} (default 1–14); section 1 viewport ${VIEWPORT ?? 'all'}`);
// ---------- 1. every state, one at a time, same layout (390x844; fit also at 390x664) ----------
if (SECTION_START <= 1 && SECTION_END >= 1) {
const LONG_NAMES = [['Vladimir Guerrero Jr.', 'Yoshinobu Yamamoto'], ['Pete Crow-Armstrong', 'Simeon Woods Richardson'], ['Jackson Chourio', 'Christopher Sánchez'], ['Lourdes Gurriel Jr.', 'Emmanuel Clase']];
const SWEEP = [
  ['recommended/ledger_recorded'], ['recommended/ledger_conflict_first_kept'], ['recommended/shown_unrecorded_no_ledger'],
  ['feed_delayed/feed_lag_above_threshold'],
  ['unconfirmed/stale_provider_stamp'], ['unconfirmed/unorderable_provider_stamp'], ['unconfirmed/same_stamp_conflict_provider_stamp'],
  ['unconfirmed/provider_order_unknown'], ['unconfirmed/prior_pitch_unconfirmed'], ['unconfirmed/receipts_unreadable'],
  ['unsupported/unknown_pitcher'], ['unsupported/no_supported_pitch_types'], ['unsupported/at_bat_rejected'], ['unsupported/policy_refused'], ['unsupported/game_type_out_of_scope'],
  ['ledger_unavailable/write_failed'], ['ledger_unavailable/display_history_unreadable'], ['ledger_unavailable/store_open_failed'],
  ['disconnected/poll_failed_last_snapshot_only'], ['disconnected/poll_failed_last_snapshot_only', { shown: true }, 'held'],
  ['waiting/between_innings'], ['waiting/resolving'], ['waiting/computing'], ['waiting/no_pending_pitch'], ['waiting/buffering'],
  ['unavailable/recommender_off'], ['unavailable/recommender_unavailable'], ['unavailable/compute_failed'],
  // low and below-zone targets (reviewer probe P4): the slot must not move whatever is painted in the zone
  ['recommended/ledger_recorded', { targets: [1.75, 1.75, 1.75] }, 'low-in-zone z=1.75'], ['unconfirmed/prior_pitch_unconfirmed', {}, 'after-low'],
  ['recommended/ledger_recorded', { targets: [1.2, 2.5, 3.0] }, 'below-zone z=1.2'], ['waiting/computing', {}, 'after-below'],
  ['recommended/ledger_recorded', { targets: [0.3, 0.3, 2.5] }, 'far-below z=0.3'],
  ['recommended/ledger_recorded', { correction: 'replaced' }, 'replaced'],
  ['unconfirmed/prior_pitch_unconfirmed', { correction: 'withdrawn' }, 'withdrawn'],
  ['feed_delayed/feed_lag_above_threshold', { correction: 'replaced' }, 'replaced'],
  ['ledger_unavailable/write_failed', { correction: 'withdrawn' }, 'withdrawn'],
  ['ledger_unavailable/display_history_unreadable', { correction: 'withdrawn' }, 'withdrawn'],
  ['disconnected/poll_failed_last_snapshot_only', { correction: 'withdrawn' }, 'withdrawn'],
  // real long names (review round 1, M1): the matchup allows two lines while its anchors stay stable
  ...LONG_NAMES.flatMap((names) => [['recommended/ledger_recorded', { names }, `names ${names.join(' vs ')}`], ['unconfirmed/prior_pitch_unconfirmed', { names }, `names ${names.join(' vs ')}`]]),
  ['ledger_unavailable/display_history_unreadable', { correction: 'withdrawn', names: LONG_NAMES[0] }, `withdrawn+names ${LONG_NAMES[0].join(' vs ')}`],
];
for (const [width, height] of VIEWPORTS.filter(v => !VIEWPORT || v.join('x') === VIEWPORT)) {
  const server = serve();
  server.body = live(SWEEP[0][0]);
  const { context, page, errors } = await open({ width, height, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  await page.evaluate(() => { window.__sameDocument = true; });
  const tag = `${width}x${height}`;
  let first = null;
  for (const [id, opts = {}, variant = ''] of SWEEP) {
    const name = `${id}${variant ? `+${variant}` : ''}`;
    server.body = live(id, opts);
    const d = { ...server.body.display_state };
    if (id === 'ledger_unavailable/store_open_failed') Object.assign(d, FALLBACK);
    await shown(page, server.body);
    const state = slot(page);
    const text = await state.innerText();
    const expectShown = d.recommendation_shown;
    if (width === 390 && height === 844) {
      ok(await page.locator('.sv-pitch-summary').count() === 1 && await state.getAttribute('data-state') === d.state, `${name}: exactly one state slot, data-state=${d.state}`);
      ok((await state.locator('.sv-chip').innerText()).includes(d.label), `${name}: chip says "${d.label}" in words`);
      const sentence = state.locator('.sv-state-msg');
      ok(await state.locator('[role="status"]').count() === 1 && await state.locator('.sv-state-live[role="status"]').count() === 1 && await state.locator('.sv-state-live').evaluate((e) => getComputedStyle(e).display !== 'none'), `${name}: one always-present polite live region for sentence + note`);
      ok(!/Ⅱ/.test(await state.locator('.sv-chip').innerText()), `${name}: chip mark is not a numeral-like glyph`);
      ok(await state.locator('.sv-state-more').count() === 1, `${name}: the slot offers 자세히 in this state`);
      if (d.message === null) ok((await sentence.innerText()).trim() === '', `${name}: no sentence for a plain recommendation`);
      else if (expectShown && d.correction) ok(await sentence.evaluate((e) => e.classList.contains('sr-only')) && (await sentence.textContent()) === d.message, `${name}: sentence kept for screen readers beside the change note`);
      else ok((await sentence.innerText()).trim() === d.message && await sentence.getAttribute('role') === null, `${name}: one calm sentence, server copy: "${d.message}"`);
      const rows = await state.locator('.sv-rec-row').count(), squares = await page.locator('.sv-zrow .sv-zone .sv-rec').count();
      ok(expectShown ? rows === 3 && squares >= 1 : rows === 0 && squares === 0, `${name}: ${expectShown ? 'candidates listed (type) and drawn (target squares)' : 'no candidate and no target square'} (rows ${rows}, squares ${squares})`);
      if (expectShown) ok(/싱커/.test(await state.locator('.sv-rec-row').first().innerText()) && await page.locator('.sv-zrow .sv-zone .sv-rec').first().isVisible(), `${name}: pitch type first, target location on the zone`);
      const visible = await spoken(page);
      ok(!JARGON.test(visible) && !/추천 대기 중/.test(visible), `${name}: no reason code, diagnostic or stale wording in text, aria-label or title`);
      if (variant === 'held') ok(await state.evaluate((e) => e.classList.contains('held')) && /마지막으로 받은 추천/.test(await state.locator('.sv-state-head').innerText()) && Number(await state.locator('.sv-rec-summary').evaluate((e) => getComputedStyle(e).opacity)) < 1 && await page.locator('.sv-live.on').count() === 0, `${name}: kept candidates are dimmed and labelled 마지막으로 받은 추천; LIVE dot off`);
      const noteText = (await state.locator('.sv-state-note').innerText()).trim();
      if (d.correction) ok(noteText === `${FIX[d.correction.kind]} (이전: 슬라이더 · 낮은 왼쪽)` && await state.locator('.sv-state-note').isVisible() && await state.locator('.sv-state-live[role="status"] .sv-state-note').count() === 1, `${name}: visible change note with the previous pick, inside the live region: "${noteText}"`);
      else ok(noteText === '', `${name}: no change note`);
      const line = page.locator('.sv-notice.line');
      const lineText = await lineWords(page);
      ok(await line.getAttribute('role') === null && await line.locator('.sv-line-text[role="status"]').count() === 1 && await line.locator('.sv-line-time').getAttribute('aria-hidden') === 'true' && await line.locator('[role="status"] button').count() === 0, `${name}: only the status words are a live region (time stamp hidden from it, retry control outside)`);
      if (d.state === 'feed_delayed') ok(/자료 늦음 · 52초 넘게 늦음 · \d\d:\d\d 수신/.test(lineText) && !/약/.test(lineText) && await line.getAttribute('data-kind') === 'late', `${name}: status line "${lineText}" (lower bound 52.9 → floor, never '약')`);
      else if (d.state === 'disconnected') ok(/연결 끊김 · \d\d:\d\d 수신/.test(lineText) && await line.getAttribute('data-kind') === 'feed' && await line.getByRole('button').count() === 1, `${name}: status line "${lineText}" with a retry control`);
      // game-level 확인 중: the line tells the chip's story (WS launch D, 2026-10-17), never 실시간으로 받는 중 beside it
      else if (d.state === 'unconfirmed') ok(/^확인 중… · \d\d:\d\d 수신/.test(lineText) && !/실시간으로 받는 중/.test(lineText) && await line.getAttribute('data-kind') === 'ok', `${name}: status line agrees with the chip "${lineText}"`);
      else ok(/실시간으로 받는 중 · \d\d:\d\d 수신/.test(lineText) && await line.getAttribute('data-kind') === 'ok', `${name}: status line quiet "${lineText}"`);
      if (shots) await page.screenshot({ path: path.join(shots, `state-${name.replace(/[/+]/g, '-')}-390x844.png`) });
    }
    const now = await boxes(page);
    if (first === null) first = now;
    const moved = sameBoxes(first, now);
    ok(moved.length === 0, `${tag} ${name}: stable live header, matchup and zone anchors (summary height may grow)${moved.length ? ` — ${moved.join(' | ')}` : ''}`);
    const out = await spill(page);
    ok(out.length === 0, `${tag} ${name}: nothing spills out of the reserved slot, no sideways scroll${out.length ? ` — ${out.join(' | ')}` : ''}`);
    const over = await page.evaluate(() => document.scrollingElement.scrollHeight - window.innerHeight);
    ok(await fitsViewport(page), `${tag} ${name}: no sideways scroll; 390x844 core above fold, shorter screens may scroll (${over}px)`);
    if (opts.names) {
      await page.waitForFunction((n) => document.querySelector('.sv-duel')?.getAttribute('aria-label')?.includes(n), opts.names[0], { timeout: 6000 }).catch(() => {});
      const duel = await page.evaluate(() => { const d = document.querySelector('.sv-duel'); const lines = [...d.querySelectorAll('b')].map((b) => Math.round(b.getBoundingClientRect().height / parseFloat(getComputedStyle(b).lineHeight)));
        return { lines, label: d.getAttribute('aria-label'), titles: [...d.querySelectorAll('span[title]')].map((e) => e.getAttribute('title')) }; });
      ok(duel.lines.every((n) => n >= 1 && n <= 2) && duel.label === `타자 ${opts.names[0]} 대 투수 ${opts.names[1]}` && duel.titles.join('|') === opts.names.join('|'), `${tag} ${name}: matchup within two readable lines; full names in aria-label and title (${JSON.stringify(duel.lines)})`);
      if (shots && (width === 375 || width === 320) && opts.names === LONG_NAMES[0] && id.startsWith('recommended')) await page.screenshot({ path: path.join(shots, `long-names-${tag}.png`) });
    }
    if (d.state === 'unconfirmed') {   // (d) every word a fan reads or hears in the 확인 중 family is plain
      const heard = await spoken(page);
      ok(!TECH.test(heard), `${tag} ${name}: no 자료/시각/순서/격리/스냅샷/원장 anywhere on screen${TECH.test(heard) ? ` — "${heard.match(TECH)[0]}"` : ''}`);
      if (width === 390 && height === 844) {
        ok(/^[^.]+\. [^.]+\.$/.test(d.message) && d.message.length <= 40, `${name}: short calm copy (${d.message.length} characters)`);
        await page.locator('.sv-state-more').click();
        const sheetText = (await page.getByRole('dialog').locator('.sv-sheet-head, .sv-detail:nth-of-type(1), .sv-detail:nth-of-type(2)').allInnerTexts()).join(' ');
        ok(sheetText.includes(d.message) && !TECH.test(sheetText), `${name}: 자세히 repeats the sentence with no technical word`);
        await page.keyboard.press('Escape');
        if (shots && id.endsWith('same_stamp_conflict_provider_stamp')) await page.screenshot({ path: path.join(shots, 'unconfirmed-copy-390x844.png') });
        if (shots && id.endsWith('prior_pitch_unconfirmed') && !variant) await page.screenshot({ path: path.join(shots, 'unconfirmed-copy-prior-pitch-390x844.png') });
      }
    }
    if (shots && width === 375 && ['recommended/ledger_recorded', 'ledger_unavailable/display_history_unreadable+withdrawn', 'waiting/between_innings', 'unconfirmed/prior_pitch_unconfirmed'].includes(name)) await page.screenshot({ path: path.join(shots, `fit-375x553-${name.replace(/[/+]/g, '-')}.png`) });
  }
  ok(await fitsViewport(page), `${tag}: live core fits 390x844; shorter screens remain scrollable`);
  ok(await page.evaluate(() => window.__sameDocument === true), `${tag}: all ${SWEEP.length} states arrived by polling, never by reload`);
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 2. D1 client guard: the list changes for the same pending pitch and the server sends no note ----------
if (SECTION_START <= 2 && SECTION_END >= 2) {
{
  const server = serve();
  const key = '849843:11:2';
  server.body = live('recommended/ledger_recorded', { pitch_key: key });
  const { context, page, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const note = page.locator('.sv-state-note');
  ok((await note.innerText()).trim() === '', 'guard: first list for this pitch, no note');
  ok(/2구/.test(await page.locator('.sv-state-head').innerText()) && await page.locator('.sv-card.pending').count() === 1, 'guard: pending pitch number and card come from the pitch slot (849843:11:2 → 2구)');
  const before = await boxes(page);
  const swapped = live('recommended/ledger_recorded', { pitch_key: key });
  const c = swapped.next.rec.candidates; [c[0].rank, c[1].rank] = [2, 1];   // same pending pitch, another order
  server.body = swapped;
  await page.waitForFunction(() => /스위퍼/.test(document.querySelector('.sv-state .sv-rec-row')?.textContent ?? ''), null, { timeout: 6000 });
  ok((await note.innerText()).trim() === '추천이 갱신됐어요 (이전: 싱커 · 가운데 왼쪽)' && await note.getAttribute('data-kind') === 'client', `guard: silent swap is announced: "${(await note.innerText()).trim()}"`);
  ok(sameBoxes(before, await boxes(page)).length === 0, 'guard: the note appears without moving the layout');
  if (shots) await page.screenshot({ path: path.join(shots, 'guard-updated-390x844.png') });
  await page.waitForTimeout(4500);
  ok((await note.innerText()).trim() === '추천이 갱신됐어요 (이전: 싱커 · 가운데 왼쪽)', 'guard: the note stays over later polls (no auto-dismiss)');
  server.body = live('unconfirmed/prior_pitch_unconfirmed', { pitch_key: key });
  await shown(page, server.body);
  ok((await note.innerText()).trim() === '조금 전에 보여 드린 추천은 지금 표시하지 않아요 (이전: 스위퍼 · 가운데 오른쪽)', `guard: a list that disappears is announced: "${(await note.innerText()).trim()}"`);
  server.body = swapped;
  await shown(page, server.body);
  ok((await note.innerText()).trim() === '', 'guard: the same list coming back is not called a change');
  server.body = live('recommended/ledger_recorded', { pitch_key: key, correction: 'replaced' });
  await page.waitForFunction(() => /싱커/.test(document.querySelector('.sv-state .sv-rec-row')?.textContent ?? ''), null, { timeout: 6000 });
  ok((await note.innerText()).trim() === `${FIX.replaced} (이전: 슬라이더 · 낮은 왼쪽)` && await note.getAttribute('data-kind') === 'replaced', 'guard: a server note wins over the client note');
  server.body = live('recommended/ledger_recorded', { pitch_key: '849843:11:3' });
  await page.waitForFunction(() => /3구/.test(document.querySelector('.sv-state-head')?.textContent ?? ''), null, { timeout: 6000 });
  ok((await note.innerText()).trim() === '', 'guard: a new pending pitch starts without a note');
  // reviewer probe P1: A for key K → an answer without a pitch_key → B for the same K, no server note
  const K2 = '849843:11:3';
  server.body = live('recommended/ledger_recorded', { pitch_key: K2 });
  await shown(page, server.body);
  server.body = live('waiting/no_pending_pitch', { pitch_key: null });
  await shown(page, server.body);
  ok((await note.innerText()).trim() === '조금 전에 보여 드린 추천은 지금 표시하지 않아요 (이전: 싱커 · 가운데 왼쪽)', `guard: a key-less answer is the same slot with nothing drawn: "${(await note.innerText()).trim()}"`);
  const swapped2 = live('recommended/ledger_recorded', { pitch_key: K2 }); { const c2 = swapped2.next.rec.candidates; [c2[0].rank, c2[1].rank] = [2, 1]; }
  server.body = swapped2;
  await waited('guard: list B arrives after the key-less gap', () => page.waitForFunction(() => /스위퍼/.test(document.querySelector('.sv-state .sv-rec-row')?.textContent ?? ''), null, { timeout: 6000 }));
  ok((await note.innerText()).trim() === '추천이 갱신됐어요 (이전: 싱커 · 가운데 왼쪽)', `guard: A → (no key) → B is announced: "${(await note.innerText()).trim()}"`);
  // a key-less answer for ANOTHER batter must not inherit "이전" from the previous one
  const other = live('waiting/no_pending_pitch', { pitch_key: null });
  other.current = { ...other.current, pa_key: '849843:12', batter: { id: 699393, name: 'Pedro Ramírez', side: 'L' } };
  server.body = other;
  await waited('guard: next batter shown', () => page.waitForFunction(() => /Pedro/.test(document.querySelector('.sv-duel')?.textContent ?? ''), null, { timeout: 6000 }));
  await page.waitForTimeout(150);
  ok((await note.innerText()).trim() === '', 'guard: a new plate appearance without a key starts clean (no "이전" from the previous batter)');
  ok(errors.length === 0, `guard: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 3. connection: feed down → back, client fetch failing → back, timeout → back; all without reload ----------
if (SECTION_START <= 3 && SECTION_END >= 3) {
{
  const server = serve();
  server.body = live('recommended/ledger_recorded');
  const { context, page, asks, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  await page.evaluate(() => { window.__sameDocument = true; });
  const line = page.locator('.sv-notice.line');
  const base = await boxes(page);
  server.body = live('disconnected/poll_failed_last_snapshot_only', { feed: { failing_since: '2026-10-11T01:02:03Z', failing_for_s: 12 } });
  await shown(page, server.body);
  ok(await line.getAttribute('data-kind') === 'feed' && /연결 끊김/.test(await lineWords(page)) && /마지막으로 받은 내용/.test(await slot(page).innerText()), 'feed down: status line 연결 끊김 + calm sentence, last snapshot stays');
  server.body = live('recommended/ledger_recorded');
  await shown(page, server.body);
  ok(await line.getAttribute('data-kind') === 'ok' && await slot(page).locator('.sv-rec-row').count() === 3, 'feed back: next poll restores the recommendation and the quiet status line');
  // the client's own requests fail
  server.down = true;
  const failingFrom = asks.length;
  await waited('client fetch failing: status line turns to 연결 끊김', () => page.locator('.sv-notice.line[data-kind="client"]').waitFor({ timeout: 6000 }));
  ok(/연결 끊김 · \d\d:\d\d 수신/.test(await lineWords(page)) && await slot(page).locator('.sv-rec-row').count() === 3, 'client fetch failing: 연결 끊김 with the last received time; the last candidates stay');
  ok((await slot(page).locator('.sv-chip').innerText()).includes('연결 끊김') && !/다음 공 추천/.test(await slot(page).locator('.sv-chip').innerText()) && await slot(page).getAttribute('data-state') === 'disconnected'
    && /마지막으로 받은 내용을 보여 드려요\./.test(await slot(page).locator('.sv-state-msg').innerText()), `client fetch failing: the slot no longer looks current (chip "${(await slot(page).locator('.sv-chip').innerText()).replace(/\s+/g, ' ').trim()}", sentence "${(await slot(page).locator('.sv-state-msg').innerText()).trim()}")`);
  ok(await page.locator('.sv-live.on').count() === 0 && await page.locator('.sv-live i:visible').count() === 0, 'client fetch failing: LIVE dot is off');
  ok(await slot(page).evaluate((e) => e.classList.contains('held')) && /마지막으로 받은 추천/.test(await slot(page).locator('.sv-state-head').innerText()) && Number(await slot(page).locator('.sv-rec-summary').evaluate((e) => getComputedStyle(e).opacity)) < 1
    && Number(await page.locator('.sv-zrow .sv-zone .sv-rec').first().evaluate((e) => getComputedStyle(e).opacity)) < 1 && /마지막으로 받은 추천 후보/.test(await slot(page).locator('.sv-reclist').getAttribute('aria-label')), 'client fetch failing: last candidates dimmed (list and zone) and labelled 마지막으로 받은 추천');
  ok(sameBoxes(base, await boxes(page)).length === 0, 'client fetch failing: the status line changes in place (no layout jump)');
  if (shots) await page.screenshot({ path: path.join(shots, 'client-fetch-failing-390x844.png') });
  await page.waitForTimeout(4500);
  ok(asks.length - failingFrom >= 2, `client fetch failing: polling keeps going (${asks.length - failingFrom} more asks in ~5 s)`);
  server.down = false;
  await waited('client fetch back: recovers on the next poll by itself', () => page.locator('.sv-notice.line[data-kind="ok"]').waitFor({ timeout: 6000 }));
  ok((await slot(page).locator('.sv-chip').innerText()).includes('다음 공 추천') && await page.locator('.sv-live.on').count() === 1 && !(await slot(page).evaluate((e) => e.classList.contains('held'))), 'client fetch back: chip, LIVE dot and full-strength candidates return');
  // a request that never answers: the client gives up after its timeout, says so, and recovers
  server.hang = true;
  await waited('timeout: a hung request turns into 연결 끊김 by itself', () => page.locator('.sv-notice.line[data-kind="client"]').waitFor({ timeout: 22000 }));
  server.hang = false;
  await waited('timeout: the next poll after the timeout loads again', () => page.locator('.sv-notice.line[data-kind="ok"]').waitFor({ timeout: 8000 }));
  ok(await page.evaluate(() => window.__sameDocument === true), 'timeout: recovered by polling, the page was never reloaded');
  // old heuristic must not fire when a display state is present: a long break has a large stale_s but an OK feed
  const quiet = live('waiting/between_innings'); quiet.feed = { ...quiet.feed, stale_s: 240 };
  server.body = quiet;
  await shown(page, server.body);
  ok(await line.getAttribute('data-kind') === 'ok', 'long break: feed.stale_s 240 with display.feed.ok does NOT show a connection warning');
  for (const r of server.hung) await r.abort().catch(() => {});
  ok(errors.length === 0, `connection: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 4. first load: retried by itself only when the server cannot be reached or fails (0 / 5xx) ----------
if (SECTION_START <= 4 && SECTION_END >= 4) {
{
  const server = serve();
  server.body = live('recommended/ledger_recorded');
  server.down = true;
  const { context, page, asks, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.getByRole('alert').waitFor();
  ok(await page.getByRole('button', { name: '다시 시도' }).count() === 1 && /자동으로 다시 연결하고 있어요/.test(await page.getByRole('alert').innerText()), 'first load failed (network): manual 다시 시도 + "자동으로 다시 연결하고 있어요"');
  const n = asks.length;
  await page.waitForTimeout(4600);
  ok(asks.length > n, `first load failed (network): retried by itself (${asks.length - n} more asks)`);
  server.down = false;
  await waited('first load: the game appears by itself once the server answers (no tap, no reload)', () => shown(page, server.body, 9000));
  ok(await page.getByRole('alert').count() === 0, 'first load: the error is gone');
  ok(errors.length === 0, `first load: no page errors ${errors.join(' | ')}`);
  await context.close();
}
for (const [status, retried] of [[423, false], [400, false], [404, false], [503, true]]) {  // reviewer probe P6
  let n = 0;
  const { context, page } = await open({ width: 390, height: 844, handler: (route) => { n++; return route.fulfill({ status, json: { detail: '시험용 응답' } }); } });
  await page.goto('http://pz.test/watch/849843');
  await page.getByRole('alert').waitFor();
  await page.waitForTimeout(9000);
  const alert = (await page.getByRole('alert').innerText()).replace(/\s+/g, ' ');
  ok(retried ? n >= 2 && /자동으로 다시 연결하고 있어요/.test(alert) : n === 1 && !/자동으로/.test(alert), `first load ${status}: ${retried ? 'retried by itself' : 'an answer, not an outage: not retried, no retry wording'} (${n} request${n === 1 ? '' : 's'} in ~9 s; "${alert}")`);
  if (status === 423) ok(/지금 기록을 볼 수 없어요/.test(alert), 'first load 423: the embargo message as before');
  await context.close();
}


}
// ---------- 5. receipts: unconfirmed pitch card, corrected tag, version change → full refetch ----------
if (SECTION_START <= 5 && SECTION_END >= 5) {
{
  // 33 real pitches; plate appearance 11 is open with three pitches thrown (synthetic: its ending is removed)
  const base = clone(fx.live_more);
  base.pitches = base.pitches.filter((p) => p.index <= 32);
  base.pas = base.pas.filter((p) => p.pa_key !== '849843:12').map((p) => p.pa_key === '849843:11' ? { ...p, complete: false, event: null, event_label: null, result_text: null, play_text: null, pitch_count: 3, last_index: 32 } : p);
  base.current = clone(fx.live.current);
  base.cutoff = { index: 32, pitch_key: '849843:11:3' };
  base.next = clone(fx.live.next);
  Object.assign(base.pitches[32].actual, { event: null, event_label: null, play_text: null });
  const withReceipts = (version, unconfirmedIndex) => {
    const body = live(unconfirmedIndex === null ? 'recommended/ledger_recorded' : 'unconfirmed/prior_pitch_unconfirmed', { base, pitch_key: '849843:11:4' });
    for (const p of body.pitches) p.receipt = { status: 'received', reason: null, corrected: p.index === 31 };
    if (unconfirmedIndex !== null) {
      const p = body.pitches[unconfirmedIndex];
      p.receipt = { status: 'unconfirmed', reason: 'stale_provider_stamp', corrected: false };
      p.actual = { ...Object.fromEntries(Object.keys(p.actual).map((k) => [k, null])), result_label: '확인 중', call_label: '확인 중' };  // as the server blanks it
    }
    body.receipts = { version, unconfirmed: unconfirmedIndex === null ? 0 : 1, corrected: 1 };
    if (unconfirmedIndex !== null) body.pas = body.pas.map((p) => p.pa_key === '849843:11' ? { ...p, result_text: '확인 중', event: null, event_label: null } : p);  // as the server sends it
    return body;
  };
  const server = serve();
  server.body = withReceipts('aaaaaaaaaaaa', 32);
  const { context, page, asks, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const cards = page.locator('.sv-strip .sv-card:not(.pending)');
  ok(await cards.count() === 3 && await page.locator('.sv-card.pending').count() === 1, 'receipts: three thrown pitches + the pending card');
  const un = cards.nth(2);
  ok(await un.evaluate((e) => e.classList.contains('unconfirmed')) && /3\. 확인 중/.test(await un.innerText()) && await un.getAttribute('aria-label') === '3구 확인 중', `receipts: unconfirmed card says 확인 중 ("${(await un.innerText()).replace(/\s+/g, ' ')}")`);
  ok(await un.locator('.sv-actual').count() === 0 && !/km\/h|mph|\d{2,3}/.test((await un.innerText()).replace(/^3\./, '')), 'receipts: unconfirmed card has no dot, no speed, no pitch type');
  const all = await page.locator('.sv-page').innerText();
  ok(!/구종 미상|0\s*km\/h|null|undefined|NaN/.test(all), 'receipts: no "구종 미상 0km/h"-style filler anywhere');
  const fixedCard = cards.nth(1);
  ok(/정정됨/.test(await fixedCard.innerText()) && /기록이 정정된 공/.test(await fixedCard.getAttribute('aria-label')) && await fixedCard.locator('.sv-actual').count() === 1, 'receipts: corrected pitch keeps its record and carries 정정됨 (label "기록이 정정된 공")');
  ok(await cards.nth(0).locator('.sv-card-tag').count() === 0 && !/정정/.test(await cards.nth(0).getAttribute('aria-label')), 'receipts: an ordinary pitch has no tag');
  if (shots) await page.screenshot({ path: path.join(shots, 'receipts-latest-390x844.png') });
  await page.waitForTimeout(2600);
  ok(asks.slice(1).every((u) => u.searchParams.get('after') === '32') && asks.length >= 2, `receipts: steady polls are incremental (?after=32), ${asks.length - 1} so far`);
  // look at the unconfirmed pitch, then at the corrected one
  const liveBefore = await page.locator('.sv-stage').innerText();
  await un.click();
  const summary = page.getByRole('dialog').locator('.sv-detail').nth(1);
  { const dialog = page.getByRole('dialog'); await dialog.waitFor(); const t = await summary.innerText();
    ok(/이 공을 다시 확인하고 있어요/.test(t) && !/판정|km\/h|mph/.test(t) && await dialog.locator('.sv-detail-zone, .sv-table').count() === 0,
      'receipts: unconfirmed detail has no actual position, pitch result or recommendation table');
    ok(await page.locator('.sv-stage').innerText() === liveBefore, 'receipts: past detail leaves current live state unchanged');
    ok(!TECH.test(t), 'receipts: unconfirmed detail has no technical diagnosis');
    if (shots) await page.screenshot({ path: path.join(shots, 'receipts-unconfirmed-selected-390x844.png') });
    await page.keyboard.press('Escape'); }
  await page.locator('.sv-inning').click();    // PA sheet
  { const item = page.locator('.sv-pa-item', { hasText: 'Nico Hoerner' }).last(); const t = (await item.innerText()).replace(/\s+/g, ' ');
    ok(/확인 중 · 3구/.test(t) && !/삼진|아웃|안타|볼넷|타석 진행 중/.test(t), `receipts: PA sheet shows the plate appearance as 확인 중, never an outcome ("${t}")`);
    if (shots) await page.screenshot({ path: path.join(shots, 'receipts-pa-sheet-390x844.png') });
    await page.keyboard.press('Escape'); }
  await fixedCard.click();
  ok(/이 공의 기록은 처음 받은 뒤에 정정됐어요/.test(await summary.innerText()) && /km\/h/.test(await summary.innerText()), 'receipts: selected corrected pitch shows its record with the 정정됨 tag');
  if (shots) await page.screenshot({ path: path.join(shots, 'receipts-corrected-selected-390x844.png') });
  // back to live; the unconfirmed pitch is admitted → receipts.version changes → the next answer is taken in full
  await page.keyboard.press('Escape');
  await shown(page, server.body);
  await page.waitForTimeout(2300);
  const mark = asks.length;
  server.body = withReceipts('bbbbbbbbbbbb', null);
  await shown(page, server.body);
  const later = asks.slice(mark).map((u) => u.searchParams.get('after'));
  ok(later.includes(null), `receipts.version change: full history taken again (requests after the change: ${later.map((a) => a === null ? 'full' : `after=${a}`).join(', ')})`);
  ok(await page.locator('.sv-card.unconfirmed').count() === 0 && /싱커|포심|스위퍼|체인지업|커브|커터|슬라이더/.test(await cards.nth(2).innerText()), `receipts.version change: the pitch is shown as received again ("${(await cards.nth(2).innerText()).replace(/\s+/g, ' ')}")`);
  const settled = asks.length;
  await page.waitForTimeout(4500);
  ok(asks.slice(settled).length >= 1 && asks.slice(settled).every((u) => u.searchParams.get('after') === '32'), 'receipts: same version afterwards → incremental polls again (no refetch loop)');
  // a blanked row without a receipt key is still never drawn as a pitch
  const bare = withReceipts('bbbbbbbbbbbb', null); bare.receipts = null;
  for (const p of bare.pitches) delete p.receipt;
  bare.pitches[32].actual = { ...Object.fromEntries(Object.keys(bare.pitches[32].actual).map((k) => [k, null])), result_label: '확인 중', call_label: '확인 중' };
  server.body = bare;
  await waited('receipts: a row the server blanked to 확인 중 (no receipt key) is still an unconfirmed card', () => page.locator('.sv-card.unconfirmed').waitFor({ timeout: 8000 }));
  ok(!/구종 미상/.test(await page.locator('.sv-page').innerText()), 'receipts: … and never reads 구종 미상');
  ok(errors.length === 0, `receipts: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 6. pregame and final ----------
if (SECTION_START <= 6 && SECTION_END >= 6) {
{
  const server = serve();
  server.body = live('pregame/scheduled', { pitch_key: null });
  const { context, page, asks, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const text = await page.locator('.sv-page').innerText();
  ok(/경기 전/.test(await slot(page).locator('.sv-chip').innerText()) && /경기 시작 전이에요\./.test(await slot(page).innerText()), 'pregame: chip 경기 전 + "경기 시작 전이에요."');
  ok(!/추천 대기 중|다음 공 대기 중|추천 기록 없음/.test(text) && await page.locator('.sv-zrow .sv-zone .sv-rec').count() === 0, 'pregame: no stale waiting wording, no mark');
  ok(await fitsViewport(page) && (await spill(page)).length === 0, 'pregame: fits, nothing spills');
  if (shots) await page.screenshot({ path: path.join(shots, 'state-pregame-scheduled-390x844.png') });
  const n = asks.length;
  server.body = live('recommended/ledger_recorded');
  await shown(page, server.body, 14000);
  ok(asks.length > n && /LIVE/.test(await page.locator('.sv-live').innerText()), 'pregame: the screen starts by itself when the game goes live (slow poll, no reload)');
  server.body = live('final/game_over', { pitch_key: null });
  await waited('final: the status line says the game is over', () => page.locator('.sv-notice.line', { hasText: '경기가 끝났어요' }).waitFor({ timeout: 6000 }));
  const end = await page.locator('.sv-page').innerText();
  ok(/경기가 끝났어요/.test(end) && !/추천 대기 중|다음 공 대기 중/.test(end) && /종료/.test(await page.locator('.sv-live').innerText()), 'final: "경기가 끝났어요" shown calmly, no stale waiting wording, header says 종료');
  ok(await page.locator('.sv-notice.line').getAttribute('data-kind') === 'ok', 'final: not a warning');
  if (shots) await page.screenshot({ path: path.join(shots, 'state-final-game_over-390x844.png') });
  const stop = asks.length;
  await page.waitForTimeout(4500);
  ok(asks.length === stop, 'final: polling stops');
  ok(errors.length === 0, `pregame/final: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 7. no display_state by design (demo, archive) → the screen as before the contract ----------
if (SECTION_START <= 7 && SECTION_END >= 7) {
for (const [name, edit] of [
  ['absent (demo fixture)', (b) => b],
  ['display_state on a demo payload is ignored', (b) => { b.display_state = display('unconfirmed/prior_pitch_unconfirmed'); return b; }],
]) {
  const server = serve();
  server.body = edit(clone(fx.live));
  const { context, page, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary').waitFor();
  await page.waitForTimeout(200);
  const text = await page.locator('.sv-pitch-summary').innerText();
  ok(await page.locator('.sv-state').count() === 0 && await page.locator('.sv-notice').count() === 0 && await page.locator('.sv-chip').count() === 0, `${name}: no state slot, no status line`);
  ok(/다음 공 추천 · 2구/.test(text) && /실제 투구/.test(text) && /다음 공 대기 중/.test(text) && await page.locator('.sv-pitch-summary .sv-rec-row').count() === 3 && await page.locator('.sv-actual-summary').count() === 1, `${name}: summary as before (다음 공 추천 · 2구 / 실제 투구 · 다음 공 대기 중, 3 candidates)`);
  ok(await page.evaluate(() => document.querySelector('.sv-rec-summary').getBoundingClientRect().bottom <= document.querySelector('.sv-actual-summary').getBoundingClientRect().top + 1), `${name}: recommendation precedes actual-pitch status without overlap`);
  if (name.startsWith('absent')) {
    const old = edit(clone(fx.live)); old.feed = { ...old.feed, stale_s: 40 };
    server.body = old;
    await waited(`${name}: old notice appears`, () => page.locator('.sv-notice.warn').waitFor({ timeout: 6000 }));
    ok(await page.locator('.sv-notice.warn').getAttribute('role') === 'alert' && /갱신 늦어짐/.test(await page.locator('.sv-notice.warn').innerText()), `${name}: old feed.stale_s > 30 notice kept without a display state`);
  }
  ok(errors.length === 0, `${name}: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 7b. LIVE route fails closed (review round 1, M2 + minor 4): no usable display_state → no recommendation, calm 확인 중 ----------
{
  const K = '849843:11:2';
  const full = () => live('recommended/ledger_recorded', { pitch_key: K });
  const CASES = [
    ['display_state missing, next.rec ready', () => { const b = full(); delete b.display_state; return b; }, 'unconfirmed', '확인 중', UNSURE],
    ['display_state null, next.rec ready', () => { const b = full(); b.display_state = null; return b; }, 'unconfirmed', '확인 중', UNSURE],
    ['unknown state name with recommendation_shown true', () => { const b = full(); b.display_state.state = 'quarantined'; b.display_state.label = '격리'; b.display_state.message = '격리된 자료'; return b; }, 'unconfirmed', '확인 중', UNSURE],
    ['unknown state name with recommendation_shown false', () => { const b = full(); b.display_state.state = 'something_new'; b.display_state.recommendation_shown = false; return b; }, 'unconfirmed', '확인 중', UNSURE],
    ['known state unconfirmed but recommendation_shown true + candidates', () => { const b = full(); Object.assign(b.display_state, { state: 'unconfirmed', reason: 'provider_order_unknown', label: '확인 중', message: UNSURE, recommendation_shown: true }); return b; }, 'unconfirmed', '확인 중', UNSURE],
    ['known state waiting but recommendation_shown true + candidates', () => { const b = full(); Object.assign(b.display_state, { state: 'waiting', reason: 'computing', label: '계산 중', message: '추천을 계산하고 있어요.', recommendation_shown: true }); return b; }, 'waiting', '계산 중', '추천을 계산하고 있어요.'],
    ['recommended but recommendation_shown missing', () => { const b = full(); delete b.display_state.recommendation_shown; return b; }, 'recommended', '다음 공 추천', '추천을 준비하고 있어요.'],
  ];
  for (const first of [false, true]) for (const [name, build, state, chip, message] of CASES) {
    const tag = `fail closed [${name}${first ? ', as the first answer' : ''}]`;
    const server = serve();
    server.body = first ? build() : full();
    const { context, page, errors } = await open({ width: 390, height: 844, handler: server.handler });
    await page.goto('http://pz.test/watch/849843');
    await page.locator('.sv-pitch-summary.sv-state').waitFor();
    if (!first) { await shown(page, server.body); ok(await slot(page).locator('.sv-rec-row').count() === 3, `${tag}: list A on screen first`); server.body = build(); }
    await waited(`${tag}: slot turns to ${state}`, () => page.waitForFunction(([st, c]) => { const e = document.querySelector('.sv-pitch-summary.sv-state'); return e?.getAttribute('data-state') === st && e.querySelector('.sv-chip')?.textContent.includes(c) && !e.querySelector('.sv-rec-row'); }, [state, chip], { timeout: 6000 }));
    await page.waitForTimeout(150);
    ok(await page.locator('.sv-rec-row').count() === 0 && await page.locator('.sv-zrow .sv-zone .sv-rec').count() === 0 && !/추천 \d순위/.test(await page.locator('.sv-zrow .sv-zone').getAttribute('aria-label')), `${tag}: no candidate row, no target square`);
    ok((await slot(page).locator('.sv-state-msg').innerText()).trim() === message && await page.locator('.sv-notice.line').count() === 1, `${tag}: state box with chip "${chip}" and "${message}", status line present`);
    const heard = await spoken(page);
    ok(!/싱커|스위퍼/.test(heard.replace(/\(이전: [^)]*\)/g, '')) && !/격리|quarantined|something_new/.test(heard), `${tag}: no candidate named anywhere outside the change note, no unknown label`);
    const note = (await slot(page).locator('.sv-state-note').innerText()).trim();
    if (!first) ok(note === '조금 전에 보여 드린 추천은 지금 표시하지 않아요 (이전: 싱커 · 가운데 왼쪽)', `${tag}: the D1 guard says the list went away: "${note}"`);
    else ok(note === '', `${tag}: nothing was shown before, no note`);
    if (shots && !first && name.startsWith('display_state missing')) await page.screenshot({ path: path.join(shots, 'fail-closed-display-state-missing-390x844.png') });
    if (shots && !first && name.startsWith('unknown state name with recommendation_shown true')) await page.screenshot({ path: path.join(shots, 'fail-closed-unknown-state-390x844.png') });
    if (!first) { server.body = full(); await shown(page, server.body); await slot(page).locator('.sv-rec-row').first().waitFor({ timeout: 6000 }).catch(() => {}); await page.waitForTimeout(120); ok(await slot(page).locator('.sv-rec-row').count() === 3 && (await slot(page).locator('.sv-state-note').innerText()).trim() === '', `${tag}: the same list coming back with a proper state is shown again, not called a change`); }
    ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
    await context.close();
  }
  // review round 2, N1: a card that is neither live nor final but carries pitches and a pending pitch is not called 경기 전
  for (const [name, game] of [['card postponed (Suspended, live false)', { status: 'postponed', status_text: 'Suspended: Rain', live: false }], ['card other (status missing)', { status: undefined, status_text: undefined, live: undefined }], ['card scheduled, live false', { status: 'scheduled', status_text: 'Scheduled', live: false }]]) {
    const tag = `fail closed [no display_state, ${name}, pitches + next present]`;
    const server = serve(); const b = live('recommended/ledger_recorded', { pitch_key: '849843:11:2' }); delete b.display_state; b.game = { ...b.game, ...game }; server.body = b;
    const { context, page, errors } = await open({ width: 390, height: 844, handler: server.handler });
    await page.goto('http://pz.test/watch/849843');
    await page.locator('.sv-pitch-summary.sv-state').waitFor();
    await page.waitForTimeout(2600);   // also after an incremental (?after=) answer, which lists no old pitch
    const t = await spoken(page);
    ok(await slot(page).getAttribute('data-state') === 'unconfirmed' && (await slot(page).locator('.sv-chip').innerText()).includes('확인 중') && (await slot(page).locator('.sv-state-msg').innerText()).trim() === UNSURE && !/경기 전|경기 시작 전/.test(t),
      `${tag}: chip 확인 중 with the default sentence, never 경기 전 (header "${(await page.locator('.sv-live').innerText()).trim()}")`);
    ok(await page.locator('.sv-rec-row').count() === 0 && await page.locator('.sv-zrow .sv-zone .sv-rec').count() === 0 && errors.length === 0, `${tag}: no candidate, no page errors ${errors.join(' | ')}`);
    await context.close();
  }
  // without a display state the fallback still tells the card's own story: final and pregame are not called 확인 중
  for (const [name, build, state, chip] of [['final card', () => { const b = live('final/game_over', { pitch_key: null }); delete b.display_state; return b; }, null, null],
    ['scheduled card', () => { const b = live('pregame/scheduled', { pitch_key: null }); delete b.display_state; return b; }, 'pregame', '경기 전']]) {
    const server = serve(); server.body = build();
    const { context, page } = await open({ width: 390, height: 844, handler: server.handler });
    await page.goto('http://pz.test/watch/849843');
    await page.locator('.sv-notice.line').waitFor();
    const t = await spoken(page);
    ok(!/확인 중|다시 확인하고/.test(t) && (state === null ? /경기가 끝났어요/.test(await lineWords(page)) : (await slot(page).locator('.sv-chip').innerText()).includes(chip)), `fail closed [${name}, no display_state]: reads ${state ?? 'final'}, not 확인 중`);
    await context.close();
  }
}

// ---------- 7c. tab hidden longer than the stale limit, then visible with a slow answer (review round 1, minor 1) ----------
{
  const server = serve();
  server.body = live('recommended/ledger_recorded', { pitch_key: '849843:11:2' });
  const { context, page, asks, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, get: () => window.__hidden === true }); window.__hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
  const n = asks.length;
  await page.waitForTimeout(37000);
  ok(asks.length - n <= 1, `hidden tab: polling pauses (${asks.length - n} request(s) in 37 s)`);
  ok(await slot(page).getAttribute('data-state') === 'recommended' && await page.locator('.sv-notice.line').getAttribute('data-kind') === 'ok', 'hidden tab: the paused time is not called a lost connection while hidden');
  server.delay = 1500;
  await page.evaluate(() => { window.__hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
  const seen = new Set(); const t0 = Date.now();
  while (Date.now() - t0 < 1300) { seen.add(await page.evaluate(() => `${document.querySelector('.sv-pitch-summary').getAttribute('data-state')}/${document.querySelector('.sv-notice.line').getAttribute('data-kind')}/${document.querySelector('.sv-pitch-summary').classList.contains('held')}`)); await page.waitForTimeout(60); }
  ok(seen.size === 1 && seen.has('recommended/ok/false'), `hidden tab → visible, answer still on its way: no false 연결 끊김 / 갱신 늦어짐 / dimmed list (${[...seen].join(', ')})`);
  server.delay = 0;
  await page.waitForTimeout(3000);
  ok(asks.length - n >= 2 && await page.locator('.sv-notice.line').getAttribute('data-kind') === 'ok', 'hidden tab → visible: polling resumed at once');
  // a real silence while visible still turns into ONE story: chip and status line both say 연결 끊김
  server.hang = true;
  await waited('visible and silent for 30 s+: the screen says so', () => page.locator('.sv-notice.line.warn').waitFor({ timeout: 45000 }));
  ok(/연결 끊김/.test(await lineWords(page)) && !/갱신 늦어짐/.test(await lineWords(page)) && (await slot(page).locator('.sv-chip').innerText()).includes('연결 끊김'), `visible and silent: chip and status line tell the same story ("${await lineWords(page)}")`);
  server.hang = false; for (const r of server.hung) await r.abort().catch(() => {});
  ok(errors.length === 0, `hidden tab: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 8. server says nothing is shown but next.rec still carries READY candidates (api.ts blanking; reviewer probe P3) ----------
if (SECTION_START <= 8 && SECTION_END >= 8) {
{
  const server = serve();
  const full = live('recommended/ledger_recorded', { pitch_key: '849843:11:2' });
  const hidden = live('unconfirmed/prior_pitch_unconfirmed', { pitch_key: '849843:11:2' }); hidden.next = clone(full.next);
  server.body = hidden;
  const { context, page, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const h2 = await page.locator('.sv-stage h2.sr-only').textContent();
  ok(await page.locator('.sv-rec-row').count() === 0 && await page.locator('.sv-zrow .sv-zone .sv-rec').count() === 0 && !/추천 \d순위/.test(await page.locator('.sv-zrow .sv-zone').getAttribute('aria-label')), 'recommendation_shown=false with ready candidates in next.rec: 0 rows, 0 squares, zone label names no target');
  ok(/확인 중/.test(h2) && !/기록한 추천|계산한 추천|정책 선택 비중/.test(h2), `recommendation_shown=false: the hidden heading does not read as if a recommendation existed ("${h2}")`);
  // no-recommendation state still opens the detail sheet (자세히 → 기준과 한계 보기, speed unit)
  await page.locator('.sv-state-more').click();
  const dialog = page.getByRole('dialog');
  const t = await dialog.innerText();
  ok(t.includes(UNSURE_PITCH) && !/싱커|스위퍼|정책 선택 비중은/.test(t) && await dialog.getByRole('button', { name: '기준과 한계 보기' }).count() === 1 && await dialog.getByRole('button', { name: /속도 단위/ }).count() === 1, '자세히 in a no-recommendation state: state sentence, no candidate table, 기준과 한계 보기 + speed unit reachable');
  await page.keyboard.press('Escape');
  ok(await dialog.count() === 0 && await page.evaluate(() => document.activeElement?.classList.contains('sv-state-more')), '자세히: Esc closes and focus returns to the opener');
  const b = await page.locator('.sv-state-more').evaluate((e) => { const r = e.getBoundingClientRect(), a = getComputedStyle(e, '::after'); return { h: r.height + 22, w: r.width + 16, inset: a.inset ?? `${a.top} ${a.left}` }; });
  ok(b.h >= 40, `자세히: touch area extended around the small label (${Math.round(b.w)}x${Math.round(b.h)} css px)`);
  ok(errors.length === 0, `blanking: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 9. a long server sentence never spills over the cards; the whole text is in 자세히 (reviewer probe P9) ----------
if (SECTION_START <= 9 && SECTION_END >= 9) {
for (const [width, height] of [[390, 664], [320, 568]]) {
  const LONG = '이 투수는 이번 시즌 중간에 팀을 옮겨 예전 기록과 지금 기록을 그대로 이어 볼 수 없어서, 이 타석에서는 추천을 쉬고 경기 상황만 계속 보여 드려요. 다음 타석부터 다시 살펴볼게요.';
  const KC = { pitch_type: 'KC', pitch_label: '너클 커브', zone_id: 'x', zone_label: '가운데 높이 오른쪽' };
  const server = serve();
  const b = live('unsupported/policy_refused');
  b.display_state.message = LONG;
  b.display_state.correction = { kind: 'withdrawn', message: FIX.withdrawn, previous: KC };
  server.body = b;
  const { context, page } = await open({ width, height, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await waited(`${width}x${height} long sentence: state shown`, () => page.locator(`.sv-state[data-pitch="${b.display_state.pitch_key}"]`).waitFor({ timeout: 6000 }));
  await page.waitForTimeout(200);
  const base = await boxes(page);
  let out = await spill(page);
  ok(out.length === 0, `${width}x${height}: ${LONG.length}-character sentence + withdrawn note stay inside the slot${out.length ? ` — ${out.join(' | ')}` : ''}`);
  ok(await fitsViewport(page), `${width}x${height}: core above fold where required, otherwise scrollable`);
  ok(await page.locator('.sv-state-note').isVisible() && /너클 커브/.test(await page.locator('.sv-state-note').innerText()), `${width}x${height}: the change note stays visible under the cut sentence`);
  if (shots) await page.screenshot({ path: path.join(shots, `long-sentence-withdrawn-${width}x${height}.png`) });
  await page.locator('.sv-state-more').click();
  const sheet = await page.getByRole('dialog').innerText();
  ok(sheet.includes(LONG) && sheet.includes(FIX.withdrawn), `${width}x${height}: the whole sentence and the note are readable in 자세히`);
  await page.keyboard.press('Escape');
  const r = live('recommended/ledger_recorded', { correction: 'replaced' }); r.display_state.correction.previous = KC;
  server.body = r;
  await shown(page, server.body);
  out = await spill(page);
  ok(out.length === 0, `${width}x${height}: note with a longer previous pick beside the list stays inside${out.length ? ` — ${out.join(' | ')}` : ''}`);
  ok(sameBoxes(base, await boxes(page)).length === 0, `${width}x${height}: no layout jump between the two`);
  await context.close();
}


}
// ---------- 10. short phones ----------
if (SECTION_START <= 10 && SECTION_END >= 10) {
for (const [width, height, fit] of [[360, 640, true], [375, 553, true]]) {
  const server = serve();
  server.body = live('ledger_unavailable/display_history_unreadable', { correction: 'withdrawn' });
  const { context, page } = await open({ width, height, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const out = await spill(page);
  ok(out.length === 0, `${width}x${height}: longest stock sentence + change note stay inside the slot${out.length ? ` — ${out.join(' | ')}` : ''}`);
  const over = await page.evaluate(() => document.scrollingElement.scrollHeight - window.innerHeight);
  // 375x553: the zone row keeps its minimum height (the page's own rule: scroll rather than shrink the zone), so the page scrolls a little
  ok(await fitsViewport(page), `${width}x${height}: vertical scroll allowed (${over}px), no sideways overflow`);
  if (shots) await page.screenshot({ path: path.join(shots, `state-ledger-withdrawn-${width}x${height}.png`) });
  await context.close();
}


}
// ---------- 11. archive: a row with no data and no receipt keeps the old '구종 미상' (never 확인 중) ----------
if (SECTION_START <= 11 && SECTION_END >= 11) {
{
  const final = clone(fx.final);
  final.pitches[1].actual = { ...Object.fromEntries(Object.keys(final.pitches[1].actual).map((k) => [k, null])), result_label: '', call_label: null };
  const { context, page, errors } = await open({ width: 390, height: 844, handler: (route, url) => url.pathname === GAME ? route.fulfill({ json: final }) : route.fulfill({ status: 404, json: { detail: 'x' } }) });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-strip .sv-card').first().waitFor();
  const card = page.locator('.sv-strip .sv-card').nth(1);
  ok(/구종 미상/.test(await card.innerText()) && !/확인 중/.test(await card.innerText()) && await page.locator('.sv-card.unconfirmed').count() === 0 && await page.locator('.sv-state, .sv-notice').count() === 0, `archive: empty row without a receipt reads as before ("${(await card.innerText()).replace(/\s+/g, ' ')}")`);
  ok(errors.length === 0, `archive: no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 12. (a) a screen opened before play starts itself: every pre-play shape the live route can answer, then live, no reload ----------
if (SECTION_START <= 12 && SECTION_END >= 12) {
{
  const pre = (id, edit = (b) => b) => { const b = live(id, { pitch_key: null }); return edit(b); };
  const scheduled = (b) => { b.game = { ...b.game, status: 'scheduled', status_text: 'Pre-Game', live: false, away_score: null, home_score: null, inning: null, half: null };
    b.pitches = []; b.pas = []; b.cutoff = { index: -1, pitch_key: null }; b.next = null; b.current = { phase: 'pregame', situation: null, batter: null, pitcher: null, pa_key: null, context: clone(NO_CONTEXT) }; return b; };
  const SHAPES = [
    ['pregame/scheduled', () => pre('pregame/scheduled'), /예정/],
    ['pregame/not_live', () => scheduled(pre('pregame/not_live')), /예정/],
    ['pregame/not_live, card says Suspended', () => { const b = scheduled(pre('pregame/not_live')); b.game.status = 'other'; b.game.status_text = 'Suspended: Rain'; return b; }, /대기/],
    ['no display_state (older backend), scheduled', () => { const b = scheduled(pre('pregame/scheduled')); delete b.display_state; return b; }, /예정/],
    ['unknown display state name, scheduled', () => { const b = scheduled(pre('pregame/scheduled')); b.display_state.state = 'something_new'; return b; }, /예정/],
    ['a state other than pregame while the card is not live (game type out of scope)', () => scheduled(pre('unsupported/game_type_out_of_scope')), /예정/],
    // Warmup and "live, no play yet": the card is already live, nobody has thrown; the first pitch slot then arrives by the ordinary poll
    ['Warmup (card live, no pitch)', () => { const b = live('unavailable/recommender_off', { pitch_key: '849843:1:1' }); b.game.status_text = 'Warmup'; b.pitches = []; b.pas = []; b.cutoff = { index: -1, pitch_key: null }; b.current.pa_key = '849843:1'; return b; }, /LIVE/],
    ['live with no play yet (nothing pending)', () => { const b = live('waiting/no_pending_pitch', { pitch_key: null }); b.pitches = []; b.pas = []; b.cutoff = { index: -1, pitch_key: null }; b.current = null; b.next = null; return b; }, /LIVE/],
    // a suspended game that already has pitches opens on its first pitch; once live it must follow the latest by itself
    ['not live with earlier pitches (resumed game)', () => { const b = pre('pregame/not_live'); b.game = { ...b.game, status: 'other', status_text: 'Suspended', live: false }; b.next = null; b.current = null; return b; }, /대기/],
    // 'Delayed Start' before the first pitch is not live (review round 1, minor 6): with the server's flag, and from the words alone
    ['Delayed Start, live:false', () => { const b = scheduled(pre('pregame/scheduled')); b.game.status_text = 'Delayed Start'; return b; }, /예정/],
    ['Delayed Start, words only (no live flag, no status kind)', () => { const b = scheduled(pre('pregame/scheduled')); delete b.game.live; b.game.status = 'Delayed Start'; b.game.status_text = 'Delayed Start'; return b; }, /예정/],
    ['503 before the first snapshot, then pregame', () => pre('pregame/scheduled'), /예정/, 503],
  ];
  await Promise.all(SHAPES.map(async ([name, build, label, firstStatus], i) => {
    const server = serve();
    server.body = build();
    let failFirst = firstStatus ? 2 : 0;
    const handler = (route, url) => url.pathname === GAME && failFirst-- > 0 ? route.fulfill({ status: firstStatus, json: { detail: 'MLB 경기 정보를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.' } }) : server.handler(route, url);
    const { context, page, asks, errors } = await open({ width: 390, height: 844, handler });
    await page.goto('http://pz.test/watch/849843');
    if (firstStatus) { await page.getByRole('alert').waitFor(); ok(/자동으로 다시 연결하고 있어요/.test(await page.getByRole('alert').innerText()), `pregame → live [${name}]: the ${firstStatus} is retried by itself`); }
    if (!await waited(`pregame → live [${name}]: the screen opens`, () => page.locator('.sv-live').waitFor({ timeout: 15000 }))) { await context.close(); return; }
    await page.waitForTimeout(300);
    await page.evaluate(() => { window.__sameDocument = true; });
    const before = (await page.locator('.sv-live').innerText()).trim();
    if (/Delayed Start/.test(name)) { const m0 = asks.length; await page.waitForTimeout(4500); ok(asks.length - m0 <= 1 && await page.locator('.sv-live.on').count() === 0, `pregame → live [${name}]: not LIVE and not on the 2 s poll (${asks.length - m0} request(s) in 4.5 s)`); }
    ok(label.test(before) && await page.locator('.sv-rec-row').count() === 0, `pregame → live [${name}]: opened before play (header "${before}", no recommendation)`);
    if (shots && i === 0) await page.screenshot({ path: path.join(shots, 'pregame-to-live-1-opened-pregame-390x844.png') });
    const n = asks.length;
    server.body = live('recommended/ledger_recorded');
    const went = await waited(`pregame → live [${name}]: live state on screen without reload`, () => shown(page, server.body, 16000));
    if (went) {
      const head = (await page.locator('.sv-live').innerText()).trim();
      ok(/LIVE/.test(head) && await page.locator('.sv-live.on').count() === 1 && await page.evaluate(() => window.__sameDocument === true) && asks.length > n,
        `pregame → live [${name}]: header LIVE with the dot, same document, ${asks.length - n} poll(s) after the server went live`);
      ok(await slot(page).locator('.sv-rec-row').count() === 3 && /Nico Hoerner/.test(await page.locator('.sv-duel').innerText()) && await page.locator('.sv-bso').count() === 1 && await page.locator('.sv-card.pending').count() === 1,
        `pregame → live [${name}]: following the latest pitch (candidates, matchup, count and the pending card)`);
      const m = asks.length; await page.waitForTimeout(4500);
      ok(asks.length - m >= 2, `pregame → live [${name}]: ordinary 2 s polling from then on (${asks.length - m} asks in 4.5 s)`);
      if (shots && i === 0) await page.screenshot({ path: path.join(shots, 'pregame-to-live-2-live-without-reload-390x844.png') });
    }
    ok(errors.length === 0, `pregame → live [${name}]: no page errors ${errors.join(' | ')}`);
    await context.close();
  }));
  // mid-game interruptions stay live when the server says live
  for (const words of ['Delayed: Rain', 'Suspended: Rain', 'Manager challenge: Tag play', 'Umpire Review', 'Delayed']) {
    const s2 = serve(); s2.body = live('recommended/ledger_recorded'); s2.body.game.status_text = words;
    const h = await open({ width: 390, height: 844, handler: s2.handler });
    await h.page.goto('http://pz.test/watch/849843');
    const up = await waited(`mid-game "${words}" with live:true: screen opens`, () => shown(h.page, s2.body));
    if (up) { const m0 = h.asks.length; await h.page.waitForTimeout(4500); ok(/LIVE/.test(await h.page.locator('.sv-live').innerText()) && h.asks.length - m0 >= 2, `mid-game "${words}" with live:true: stays LIVE on the 2 s poll (${h.asks.length - m0} asks in 4.5 s)`); }
    await h.context.close();
  }
  // a finished game from the live route is not polled (nothing more will come)
  const server = serve();
  server.body = live('final/game_over', { pitch_key: null });
  const { context, page, asks } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-live').waitFor();
  const n = asks.length; await page.waitForTimeout(11500);
  ok(asks.length === n, `final from the live route: not polled (${asks.length - n} asks in 11.5 s)`);
  await context.close();
}


}
// ---------- 13. (b) between innings: nothing of the finished half is presented as current; the score stays; nothing moves ----------
if (SECTION_START <= 13 && SECTION_END >= 13) {
for (const [width, height] of [[390, 844], [375, 553], [320, 568]]) {
  const tag = `between innings ${width}x${height}`;
  const server = serve();
  const play = () => live('unavailable/recommender_off', { pitch_key: '849843:11:2' });
  server.body = play();
  const { context, page, errors } = await open({ width, height, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  await page.evaluate(() => { window.__sameDocument = true; });
  const inPlay = await boxes(page);
  ok(/Nico Hoerner/.test(await page.locator('.sv-duel').innerText()) && await page.locator('.sv-bso').count() === 1, `${tag}: in play the matchup and the count are shown`);
  // the third out came on a play that also scored a run: the card has no score, the linescore has 1 : 2
  const line = clone(LINESCORE); line.teams.away.runs = 1; line.innings[1].away = 1;
  server.body = live('waiting/between_innings', { pitch_key: null, linescore: line });
  await shown(page, server.body);
  const heard = await spoken(page);
  ok(!/Nico Hoerner|Michael King|Hoerner|King/.test(heard), `${tag}: the previous batter and pitcher are not named anywhere (text, aria-label, title)`);
  ok(await page.locator('.sv-bso').count() === 0 && await page.locator('.sv-bases').count() === 0 && !/아웃|스트라이크 \d|볼 \d|주자/.test(heard), `${tag}: no count, no outs, no runners of the finished half`);
  ok(!/추천 대기 중|다음 공 대기 중|선수 미확인|타자 미확인|타자:|투수:/.test(heard), `${tag}: no stale waiting wording, no placeholder matchup`);
  ok((await slot(page).locator('.sv-chip').innerText()).includes('공수 교대') && /공수 교대 중이에요\./.test(await slot(page).innerText()) && (await page.locator('.sv-board-idle').innerText()).trim() === '공수 교대', `${tag}: chip 공수 교대 + "공수 교대 중이에요." and the board says 공수 교대`);
  ok(!/다음 공|\d구/.test(await slot(page).innerText()), `${tag}: the slot names no "next pitch" number during the break ("${(await slot(page).innerText()).replace(/\s+/g, ' ')}")`);
  const inning = page.locator('.sv-inning');
  ok(/2회초 종료/.test(await inning.innerText()) && /2회 초 종료/.test(await inning.getAttribute('aria-label')), `${tag}: the half that just ended is named as ended ("${(await inning.innerText()).replace(/\s+/g, ' ')}")`);
  ok(/컵스 1 대 2 샌디에이고/.test(await page.locator('.sv-head-main').getAttribute('aria-label')) && /1\s*vs\s*2/.test(await page.locator('.sv-head-score').innerText()), `${tag}: score kept from the as-of linescore, including the run on the last play (${(await page.locator('.sv-head-score').innerText()).replace(/\s+/g, ' ')})`);
  ok(await page.locator('.sv-strip .sv-card:not(.sv-card-idle)').count() === 0 && await page.locator('.sv-zrow .sv-zone .sv-rec, .sv-zrow .sv-zone .sv-actual').count() === 0 && await page.locator('.sv-batband').count() === 0, `${tag}: no pitch card of the last batter, empty zone, no batter figure`);
  ok(await page.locator('.sv-player.idle').count() === 2 && await page.locator('.sv-player:not([disabled])').count() === 0 && /^CHC––$/.test((await page.locator('.sv-player.away').innerText()).replace(/\s+/g, '')), `${tag}: player cells show the two teams only ("${(await page.locator('.sv-player.away').innerText()).replace(/\s+/g, ' ')}")`);
  ok(/LIVE/.test(await page.locator('.sv-live').innerText()) && await page.locator('.sv-notice.line').getAttribute('data-kind') === 'ok', `${tag}: still LIVE, quiet status line`);
  const moved = sameBoxes(inPlay, await boxes(page));
  ok(moved.length === 0, `${tag}: entering the break moves nothing (status line, board, zone, state slot, cards, players)${moved.length ? ` — ${moved.join(' | ')}` : ''}`);
  ok(await fitsViewport(page) && (await spill(page)).length === 0, `${tag}: no sideways overflow, nothing spills`);
  if (shots) await page.screenshot({ path: path.join(shots, `between-innings-${width}x${height}.png`) });
  // the PA sheet still reaches the finished half
  await inning.click();
  ok(await page.getByRole('dialog').locator('.sv-pa-item').count() >= 1, `${tag}: 타석 선택 still lists the finished half's plate appearances`);
  await page.keyboard.press('Escape');
  // older backend: phase only, no display state
  const bare = live('waiting/between_innings', { pitch_key: null, linescore: line }); delete bare.display_state;
  server.body = bare;
  await page.waitForTimeout(2600);
  { const t = await spoken(page); ok(await page.locator('.sv-board-idle').count() === 1 && (await slot(page).locator('.sv-chip').innerText()).includes('공수 교대') && !/Hoerner|King|추천 대기 중|확인 중/.test(t) && /공수 교대 중이에요/.test(t) && sameBoxes(inPlay, await boxes(page)).length === 0,
    `${tag}: (no display_state) the break is still drawn from current.phase, with the same slot and no previous matchup`); }
  // the next half starts
  server.body = play();
  await shown(page, server.body);
  ok(/Nico Hoerner/.test(await page.locator('.sv-duel').innerText()) && await page.locator('.sv-bso').count() === 1 && await page.locator('.sv-board-idle').count() === 0 && sameBoxes(inPlay, await boxes(page)).length === 0, `${tag}: play resumes by polling, same layout`);
  ok(await page.evaluate(() => window.__sameDocument === true) && errors.length === 0, `${tag}: no reload, no page errors ${errors.join(' | ')}`);
  await context.close();
}


}
// ---------- 14. (c) a past pitch shows the recommendation recorded before it next to what was thrown; calm line when there is none ----------
if (SECTION_START <= 14 && SECTION_END >= 14) {
{
  const base = clone(fx.live_more);
  base.pitches = base.pitches.filter((p) => p.index <= 32);
  base.pas = base.pas.filter((p) => p.pa_key !== '849843:12').map((p) => p.pa_key === '849843:11' ? { ...p, complete: false, event: null, event_label: null, result_text: null, play_text: null, pitch_count: 3, last_index: 32 } : p);
  base.current = clone(fx.live.current);
  base.cutoff = { index: 32, pitch_key: '849843:11:3' };
  base.next = clone(fx.live.next);
  Object.assign(base.pitches[32].actual, { event: null, event_label: null, play_text: null });
  const RECORDED = { provenance: 'captured_live', status: 'ready', reason: null, recorded_at: '2026-09-30T02:31:07Z', policy: { name: 'ARM-B', identity: 'arm-b-test' }, ledger: { status: 'recorded', outcome: 'first' },
    pairing: { status: 'paired', rule: null },
    candidates: [{ rank: 1, pitch_type: 'SL', pitch_label: '슬라이더', zone_id: 'low_left', zone_label: '낮은 왼쪽', target: { x: -0.55, z: 1.9 }, detail: { probability: 0.52 } },
      { rank: 2, pitch_type: 'FF', pitch_label: '포심', zone_id: 'high_right', zone_label: '높은 오른쪽', target: { x: 0.55, z: 3.09 }, detail: { probability: 0.31 } },
      { rank: 3, pitch_type: 'KC', pitch_label: '너클커브', zone_id: 'middle_middle', zone_label: '가운데 높이 중앙', target: { x: 0, z: 2.49 }, detail: { probability: 0.17 } }] };
  const ABSENT = (reason) => ({ provenance: 'absent', status: 'missing', reason, recorded_at: null, policy: null, candidates: [], pairing: { status: 'none', rule: 'no_recorded_recommendation' } });
  const body = (state = 'recommended/ledger_recorded') => {
    const b = live(state, { base, pitch_key: state.startsWith('final') ? null : '849843:11:4' });
    for (const p of b.pitches) { p.receipt = { status: 'received', reason: null, corrected: false }; p.rec = ABSENT('이 투구의 저장된 추천이 없습니다.'); }   // the pre-contract stock sentence
    b.pitches[30].rec = clone(RECORDED);                                   // PA 11, 1구: recorded
    b.pitches[31].rec = ABSENT('이 공은 기록된 추천이 없어요.');            // 2구: the server's calm sentence
    b.pitches[32].rec = clone(RECORDED);                                   // 3구: unconfirmed, although the row carries a record
    b.pitches[32].receipt = { status: 'unconfirmed', reason: 'stale_provider_stamp', corrected: false };
    b.pitches[32].actual = { ...Object.fromEntries(Object.keys(b.pitches[32].actual).map((k) => [k, null])), result_label: '확인 중', call_label: '확인 중' };
    b.pitches[29].rec = ABSENT(null);                                      // no sentence at all
    b.receipts = { version: 'cccccccccccc', unconfirmed: 1, corrected: 0 };
    return b;
  };
  for (const [width, height] of [[390, 844], [390, 664], [375, 553], [360, 640], [320, 568], [1440, 900]]) {
    const tag = `past pitch ${width}x${height}`, server = serve();
    server.body = body();
    const { context, page, errors } = await open({ width, height, handler: server.handler });
    await page.goto('http://pz.test/watch/849843');
    await shown(page, server.body);
    const cards = page.locator('.sv-strip .sv-card:not(.pending)');
    const mainBefore = await page.locator('.sv-stage').innerText();
    await cards.nth(0).click();
    const dialog = page.getByRole('dialog'), rec = dialog.locator('.sv-detail').nth(0), actual = dialog.locator('.sv-detail').nth(1);
    await dialog.waitFor();
    ok(await page.locator('.sv-stage').innerText() === mainBefore && await page.locator('.sv-live.on').count() === 1,
      `${tag}: history selection changes only detail, current recommendation and LIVE remain`);
    ok(await dialog.getAttribute('aria-modal') === (width >= 1200 ? null : 'true'), `${tag}: desktop nonmodal / phone modal detail`);
    const rt = (await rec.innerText()).replace(/\s+/g, ' '), at = (await actual.innerText()).replace(/\s+/g, ' ');
    ok(/이 공을 던지기 전 기록/.test(rt) && /경기 중 투구 전에 기록한 추천 · 11:31 기록/.test(rt) && await rec.locator('tbody tr').count() === 3,
      `${tag}: recorded recommendation provenance, time and all three candidates`);
    ok(/슬라이더/.test(await rec.locator('tbody tr').first().innerText()) && /낮은 왼쪽/.test(await rec.locator('tbody tr').first().innerText()),
      `${tag}: type and target words retain catcher view`);
    ok(at.includes(server.body.pitches[30].actual.pitch_label) && /km\/h/.test(at) && await actual.locator('.sv-actual').count() === 1 && await actual.locator('.sv-rec').count() === 3,
      `${tag}: detail shows actual pitch and all recorded target marks`);
    ok(/성공·적중 확률이 아니에요/.test(rt) && !/ARM-B|arm-b/i.test(await dialog.innerText()), `${tag}: selection share is not success probability; no policy code`);
    ok(!SCORING.test((await dialog.innerText()).replace('정책 선택 비중은 성공·적중 확률이 아니에요.', '')), `${tag}: no hit/miss scoring or accuracy claim in historical detail`);
    server.body = body('unconfirmed/prior_pitch_unconfirmed');
    await shown(page, server.body);
    ok(await dialog.count() === 1 && /슬라이더/.test(await rec.innerText()), `${tag}: new live state updates behind a retained past detail`);
    await page.keyboard.press('Escape');
    await cards.nth(1).click(); await dialog.waitFor();
    ok((await rec.innerText()).includes('이 공은 기록된 추천이 없어요.') && await rec.locator('table').count() === 0 && await actual.locator('.sv-rec').count() === 0 && await actual.locator('.sv-actual').count() === 1,
      `${tag}: no recorded recommendation -> calm absence, no invented targets, actual remains`);
    await page.keyboard.press('Escape');
    await cards.nth(2).click(); await dialog.waitFor();
    ok(await dialog.locator('.sv-table, .sv-detail-zone').count() === 0 && /이 공을 다시 확인하고 있어요/.test(await actual.innerText()) && !/km\/h|mph|슬라이더|너클커브/.test(await actual.innerText()),
      `${tag}: unconfirmed historical pitch has no recommendation, coordinates, speed or result`);
    await page.keyboard.press('Escape');
    await page.locator('.sv-inning').click();
    const item = (await page.locator('.sv-pa-item', { hasText: 'Nico Hoerner' }).last().innerText()).replace(/\s+/g, ' ');
    ok(/확인 중 · 3구/.test(item) && !/타석 진행 중/.test(item), `${tag}: PA containing unconfirmed pitch has no inferred outcome`);
    await page.locator('.sv-pa-item').first().click(); await dialog.waitFor();
    ok((await rec.innerText()).includes('이 공은 기록된 추천이 없어요.'), `${tag}: older stock missing-record sentence becomes calm absence`);
    ok(errors.length === 0 && await fitsWidth(page), `${tag}: no page errors or sideways scroll`);
    if (shots) await page.screenshot({ path: path.join(shots, `past-detail-${width}x${height}.png`) });
    await context.close();
  }
  const server = serve(), end = body('final/game_over'); end.pitches[0].rec = clone(RECORDED); server.body = end;
  const { context, page, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary .sv-rec-row').first().waitFor();
  ok(/종료/.test(await page.locator('.sv-live').innerText()) && /투구 전에 기록된 추천/.test(await page.locator('.sv-rec-summary').innerText()), 'final screen: first pitch still has its archived recommendation');
  await page.locator('.sv-strip .sv-card').nth(1).click();
  const dialog = page.getByRole('dialog'); await dialog.waitFor();
  ok(/이 공은 기록된 추천이 없어요/.test(await dialog.locator('.sv-detail').first().innerText()) && await dialog.locator('.sv-table').count() === 0,
    'final detail: missing recorded recommendation is not reconstructed');
  ok(errors.length === 0, `final screen: no page errors ${errors.join(' | ')}`);
  await context.close();
}

}

await browser.close();
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

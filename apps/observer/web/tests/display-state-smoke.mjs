// Live display state (contract 2026-10-11, `display_state` schema pitcheezy-display-state-v1) on the fan game screen.
// Canned answers on the built dist/ (no server, no model, no network): a live payload is made from the real-game fixture
// (tests/fixtures/service-849843.json `live` / `live_more`) by marking it `source.kind = 'live'` and attaching a display state.
// Covers: every state and sub-reason with the server's copy, one state at a time, no reason codes on screen; the D1 notes
// (server `replaced` / `withdrawn`, and the client guard when the server sends none); the reserved status line (feed late with
// the measured lag, feed down, client fetch failing / timing out) and recovery by polling alone (no reload); first-load retry;
// unconfirmed pitch card and corrected tag; receipts.version change → full refetch; pregame / final; no layout jump between
// states at 390x844 and 320x568; absent / unknown display_state → the screen as before the contract.
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
const COPY = {  // (label, message) exactly as the backend authors them (display_state.COPY)
  'recommended/ledger_recorded': ['다음 공 추천', null],
  'recommended/ledger_conflict_first_kept': ['다음 공 추천', null],
  'recommended/shown_unrecorded_no_ledger': ['다음 공 추천', null],
  'unconfirmed/stale_provider_stamp': ['확인 중', '방금 받은 자료가 앞선 자료보다 이른 시각이라, 확인될 때까지 추천을 잠시 멈췄어요.'],
  'unconfirmed/unorderable_provider_stamp': ['확인 중', '받은 자료의 순서를 확인할 수 없어, 확인될 때까지 추천을 잠시 멈췄어요.'],
  'unconfirmed/same_stamp_conflict_provider_stamp': ['확인 중', '같은 시각에 내용이 다른 자료가 와서, 확인될 때까지 추천을 잠시 멈췄어요.'],
  'unconfirmed/provider_order_unknown': ['확인 중', '받은 자료의 순서를 확인하는 중이라 추천을 잠시 멈췄어요.'],
  'unconfirmed/prior_pitch_unconfirmed': ['확인 중', '앞선 공의 기록을 확인하는 중이라 추천을 잠시 멈췄어요.'],
  'unconfirmed/receipts_unreadable': ['확인 중', '받은 기록을 확인할 수 없어 추천을 잠시 멈췄어요.'],
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
  'final/game_over': ['경기 종료', '경기가 끝났어요.'],
};
const SHOWN = new Set(['recommended', 'feed_delayed']);
const PREV = { pitch_type: 'SL', pitch_label: '슬라이더', zone_id: 'low_left', zone_label: '낮은 왼쪽' };  // catcher's view → shown as '낮은 오른쪽'
const FIX = { replaced: '상황이나 기록이 바뀌어 추천이 달라졌어요.', withdrawn: '조금 전에 보여 드린 추천은 지금 표시하지 않아요.' };
const FALLBACK = { label: '추천 보류', message: '추천 기록을 저장하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.' };  // api.ts STATE_COPY
let serial = 100;
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
  if (reason === 'between_innings' || reason === 'resolving') { body.current.phase = reason; body.next = null; }
  else if (!body.display_state.recommendation_shown) body.next = { pitch_key: body.display_state.pitch_key, rec: { provenance: 'absent', status: 'missing', reason: body.display_state.message, recorded_at: null, candidates: [], policy: null } };
  if (state === 'pregame') {
    body.game = { ...body.game, status: 'scheduled', status_text: 'Scheduled', live: false, away_score: null, home_score: null, inning: null, half: null };
    body.pitches = []; body.pas = []; body.cutoff = { index: -1, pitch_key: null }; body.next = null;
    body.current = { phase: 'pregame', situation: null, batter: null, pitcher: null, pa_key: null };
  }
  if (state === 'final') { body.game = { ...body.game, status: 'final', status_text: 'Final', live: false }; body.current = null; body.next = null; }
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
/** One mutable answer per page: `server.body` is what the next poll gets. */
function serve() {
  const server = { body: null, down: false, hang: false, hung: [] };
  server.handler = (route, url) => {
    if (url.pathname !== GAME) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
    if (server.hang) { server.hung.push(route); return undefined; }   // never answered: the client must time out by itself
    if (server.down) return route.abort('internetdisconnected');
    return route.fulfill({ json: answer(server.body, url) });
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
const boxes = (page) => page.evaluate(() => Object.fromEntries(['.sv-notice.line', '.sv-board', '.sv-duel', '.sv-zrow', '.sv-zrow .sv-zone-wrap', '.sv-pitch-summary', '.sv-strip-shell', '.sv-context'].map((sel) => {
  const r = document.querySelector(sel)?.getBoundingClientRect(); return [sel, r ? [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 2) / 2).join(',') : 'none']; })));
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
const fitsOneScreen = (page) => page.evaluate(() => document.scrollingElement.scrollHeight <= window.innerHeight + 1 && document.scrollingElement.scrollWidth <= window.innerWidth + 1);
const JARGON = /ledger|provider|stamp|diag:|receipt|unconfirmed|recommended|poll_failed|write_failed|[a-z]+_[a-z]+/;
/** Wait until the screen shows the answer carrying this pitch_key (set by the poll, no reload). */
async function shown(page, body, timeout = 6000) {
  const d = { ...body.display_state, label: body.display_state.label ?? FALLBACK.label, message: body.display_state.message ?? (body.display_state.state === 'ledger_unavailable' ? FALLBACK.message : null) };
  await page.locator(`.sv-pitch-summary.sv-state[data-state="${d.state}"]${d.pitch_key ? `[data-pitch="${d.pitch_key}"]` : ''}`).first().waitFor({ timeout });
  await page.waitForFunction(([label, message]) => { const s = document.querySelector('.sv-pitch-summary.sv-state'); return !!s && s.querySelector('.sv-chip')?.textContent.includes(label) && (message === null || s.textContent.includes(message)); }, [d.label, d.message], { timeout });
  await page.waitForTimeout(120);
}

// ---------- 1. every state, one at a time, same layout (390x844; fit also at 390x664) ----------
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
];
for (const [width, height] of [[390, 844], [390, 664], [320, 568]]) {
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
      if (d.correction) ok(noteText === `${FIX[d.correction.kind]} (이전: 슬라이더 · 낮은 오른쪽)` && await state.locator('.sv-state-note').isVisible() && await state.locator('.sv-state-live[role="status"] .sv-state-note').count() === 1, `${name}: visible change note with the previous pick, inside the live region: "${noteText}"`);
      else ok(noteText === '', `${name}: no change note`);
      const line = page.locator('.sv-notice.line');
      const lineText = await lineWords(page);
      ok(await line.getAttribute('role') === null && await line.locator('.sv-line-text[role="status"]').count() === 1 && await line.locator('.sv-line-time').getAttribute('aria-hidden') === 'true' && await line.locator('[role="status"] button').count() === 0, `${name}: only the status words are a live region (time stamp hidden from it, retry control outside)`);
      if (d.state === 'feed_delayed') ok(/자료 늦음 · 52초 넘게 늦음 · \d\d:\d\d 수신/.test(lineText) && !/약/.test(lineText) && await line.getAttribute('data-kind') === 'late', `${name}: status line "${lineText}" (lower bound 52.9 → floor, never '약')`);
      else if (d.state === 'disconnected') ok(/연결 끊김 · \d\d:\d\d 수신/.test(lineText) && await line.getAttribute('data-kind') === 'feed' && await line.getByRole('button').count() === 1, `${name}: status line "${lineText}" with a retry control`);
      else ok(/실시간으로 받는 중 · \d\d:\d\d 수신/.test(lineText) && await line.getAttribute('data-kind') === 'ok', `${name}: status line quiet "${lineText}"`);
      if (shots) await page.screenshot({ path: path.join(shots, `state-${name.replace(/[/+]/g, '-')}-390x844.png`) });
    }
    const now = await boxes(page);
    if (first === null) first = now;
    const moved = sameBoxes(first, now);
    ok(moved.length === 0, `${tag} ${name}: no layout jump (status line, board, zone, state slot, cards, players)${moved.length ? ` — ${moved.join(' | ')}` : ''}`);
    const out = await spill(page);
    ok(out.length === 0, `${tag} ${name}: nothing spills out of the reserved slot, no sideways scroll${out.length ? ` — ${out.join(' | ')}` : ''}`);
  }
  ok(await fitsOneScreen(page), `${tag}: live screen with the state slot and status line fits one screen`);
  ok(await page.evaluate(() => window.__sameDocument === true), `${tag}: all ${SWEEP.length} states arrived by polling, never by reload`);
  ok(errors.length === 0, `${tag}: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 2. D1 client guard: the list changes for the same pending pitch and the server sends no note ----------
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
  ok((await note.innerText()).trim() === '추천이 갱신됐어요 (이전: 싱커 · 가운데 오른쪽)' && await note.getAttribute('data-kind') === 'client', `guard: silent swap is announced: "${(await note.innerText()).trim()}"`);
  ok(sameBoxes(before, await boxes(page)).length === 0, 'guard: the note appears without moving the layout');
  if (shots) await page.screenshot({ path: path.join(shots, 'guard-updated-390x844.png') });
  await page.waitForTimeout(4500);
  ok((await note.innerText()).trim() === '추천이 갱신됐어요 (이전: 싱커 · 가운데 오른쪽)', 'guard: the note stays over later polls (no auto-dismiss)');
  server.body = live('unconfirmed/prior_pitch_unconfirmed', { pitch_key: key });
  await shown(page, server.body);
  ok((await note.innerText()).trim() === '조금 전에 보여 드린 추천은 지금 표시하지 않아요 (이전: 스위퍼 · 가운데 왼쪽)', `guard: a list that disappears is announced: "${(await note.innerText()).trim()}"`);
  server.body = swapped;
  await shown(page, server.body);
  ok((await note.innerText()).trim() === '', 'guard: the same list coming back is not called a change');
  server.body = live('recommended/ledger_recorded', { pitch_key: key, correction: 'replaced' });
  await page.waitForFunction(() => /싱커/.test(document.querySelector('.sv-state .sv-rec-row')?.textContent ?? ''), null, { timeout: 6000 });
  ok((await note.innerText()).trim() === `${FIX.replaced} (이전: 슬라이더 · 낮은 오른쪽)` && await note.getAttribute('data-kind') === 'replaced', 'guard: a server note wins over the client note');
  server.body = live('recommended/ledger_recorded', { pitch_key: '849843:11:3' });
  await page.waitForFunction(() => /3구/.test(document.querySelector('.sv-state-head')?.textContent ?? ''), null, { timeout: 6000 });
  ok((await note.innerText()).trim() === '', 'guard: a new pending pitch starts without a note');
  // reviewer probe P1: A for key K → an answer without a pitch_key → B for the same K, no server note
  const K2 = '849843:11:3';
  server.body = live('recommended/ledger_recorded', { pitch_key: K2 });
  await shown(page, server.body);
  server.body = live('waiting/no_pending_pitch', { pitch_key: null });
  await shown(page, server.body);
  ok((await note.innerText()).trim() === '조금 전에 보여 드린 추천은 지금 표시하지 않아요 (이전: 싱커 · 가운데 오른쪽)', `guard: a key-less answer is the same slot with nothing drawn: "${(await note.innerText()).trim()}"`);
  const swapped2 = live('recommended/ledger_recorded', { pitch_key: K2 }); { const c2 = swapped2.next.rec.candidates; [c2[0].rank, c2[1].rank] = [2, 1]; }
  server.body = swapped2;
  await waited('guard: list B arrives after the key-less gap', () => page.waitForFunction(() => /스위퍼/.test(document.querySelector('.sv-state .sv-rec-row')?.textContent ?? ''), null, { timeout: 6000 }));
  ok((await note.innerText()).trim() === '추천이 갱신됐어요 (이전: 싱커 · 가운데 오른쪽)', `guard: A → (no key) → B is announced: "${(await note.innerText()).trim()}"`);
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

// ---------- 3. connection: feed down → back, client fetch failing → back, timeout → back; all without reload ----------
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
  ok(await page.locator('.sv-live.on').count() === 0 && await page.locator('.sv-live i').count() === 0, 'client fetch failing: LIVE dot is off');
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

// ---------- 4. first load: retried by itself only when the server cannot be reached or fails (0 / 5xx) ----------
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

// ---------- 5. receipts: unconfirmed pitch card, corrected tag, version change → full refetch ----------
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
  ok(await un.evaluate((e) => e.classList.contains('unconfirmed')) && /3\. 확인 중/.test(await un.innerText()) && await un.getAttribute('aria-label') === '3구 기록 확인 중', `receipts: unconfirmed card says 확인 중 ("${(await un.innerText()).replace(/\s+/g, ' ')}")`);
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
  await un.click();
  const summary = page.locator('.sv-actual-summary');
  ok(/확인 중/.test(await summary.innerText()) && /이 공의 기록을 확인하고 있어요/.test(await summary.innerText()) && !/km\/h|mph/.test(await summary.innerText()) && await page.locator('.sv-zrow .sv-zone .sv-actual').count() === 0, 'receipts: selected unconfirmed pitch shows 확인 중, nothing presented as an observed pitch');
  if (shots) await page.screenshot({ path: path.join(shots, 'receipts-unconfirmed-selected-390x844.png') });
  await page.locator('.sv-reclist').click();   // detail sheet of the unconfirmed pitch
  { const dialog = page.getByRole('dialog'); const t = await dialog.locator('.sv-detail').nth(1).innerText();
    ok(/실제 투구/.test(t) && /이 공의 기록을 확인하고 있어요/.test(t) && !/원본 좌표|판정|구종|km\/h|mph/.test(t.replace('구종과 위치를 보여 드려요', '')) && await dialog.locator('.sv-detail-zone').count() === 0, 'receipts: detail sheet of an unconfirmed pitch says 확인 중이에요, no type / speed / location / call rows');
    await page.keyboard.press('Escape'); }
  await page.locator('.sv-inning').click();    // PA sheet
  { const item = page.locator('.sv-pa-item', { hasText: 'Nico Hoerner' }).last(); const t = (await item.innerText()).replace(/\s+/g, ' ');
    ok(/확인 중 · 3구/.test(t) && !/삼진|아웃|안타|볼넷|타석 진행 중/.test(t), `receipts: PA sheet shows the plate appearance as 확인 중, never an outcome ("${t}")`);
    if (shots) await page.screenshot({ path: path.join(shots, 'receipts-pa-sheet-390x844.png') });
    await page.keyboard.press('Escape'); }
  await fixedCard.click();
  ok(await summary.locator('.sv-tag').innerText() === '정정됨' && await summary.locator('.sv-tag').getAttribute('aria-label') === '기록이 정정된 공' && /km\/h/.test(await summary.innerText()), 'receipts: selected corrected pitch shows its record with the 정정됨 tag');
  if (shots) await page.screenshot({ path: path.join(shots, 'receipts-corrected-selected-390x844.png') });
  // back to live; the unconfirmed pitch is admitted → receipts.version changes → the next answer is taken in full
  await page.locator('.sv-live').click();
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

// ---------- 6. pregame and final ----------
{
  const server = serve();
  server.body = live('pregame/scheduled', { pitch_key: null });
  const { context, page, asks, errors } = await open({ width: 390, height: 844, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const text = await page.locator('.sv-page').innerText();
  ok(/경기 전/.test(await slot(page).locator('.sv-chip').innerText()) && /경기 시작 전이에요\./.test(await slot(page).innerText()), 'pregame: chip 경기 전 + "경기 시작 전이에요."');
  ok(!/추천 대기 중|다음 공 대기 중|추천 기록 없음/.test(text) && await page.locator('.sv-zrow .sv-zone .sv-rec').count() === 0, 'pregame: no stale waiting wording, no mark');
  ok(await fitsOneScreen(page) && (await spill(page)).length === 0, 'pregame: fits, nothing spills');
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

// ---------- 7. absent / unknown display_state → the screen as before the contract ----------
for (const [name, edit] of [
  ['absent (demo fixture)', (b) => b],
  ['absent (live source, older backend)', (b) => { b.source = { kind: 'live', session_id: null }; return b; }],
  ['unknown state name', (b) => { b.source = { kind: 'live', session_id: null }; b.display_state = { ...display('recommended/ledger_recorded'), state: 'something_new' }; return b; }],
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
  ok(await page.evaluate(() => getComputedStyle(document.querySelector('.sv-pitch-summary')).display) === 'grid', `${name}: two-column summary layout unchanged`);
  if (name.startsWith('absent (live')) {
    const old = edit(clone(fx.live)); old.feed = { ...old.feed, stale_s: 40 };
    server.body = old;
    await waited(`${name}: old notice appears`, () => page.locator('.sv-notice.warn').waitFor({ timeout: 6000 }));
    ok(await page.locator('.sv-notice.warn').getAttribute('role') === 'alert' && /갱신 늦어짐/.test(await page.locator('.sv-notice.warn').innerText()), `${name}: old feed.stale_s > 30 notice kept without a display state`);
  }
  ok(errors.length === 0, `${name}: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 8. server says nothing is shown but next.rec still carries READY candidates (api.ts blanking; reviewer probe P3) ----------
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
  ok(/앞선 공의 기록을 확인하는 중이라 추천을 잠시 멈췄어요\./.test(t) && !/싱커|스위퍼|정책 선택 비중은/.test(t) && await dialog.getByRole('button', { name: '기준과 한계 보기' }).count() === 1 && await dialog.getByRole('button', { name: /속도 단위/ }).count() === 1, '자세히 in a no-recommendation state: state sentence, no candidate table, 기준과 한계 보기 + speed unit reachable');
  await page.keyboard.press('Escape');
  ok(await dialog.count() === 0 && await page.evaluate(() => document.activeElement?.classList.contains('sv-state-more')), '자세히: Esc closes and focus returns to the opener');
  const b = await page.locator('.sv-state-more').evaluate((e) => { const r = e.getBoundingClientRect(), a = getComputedStyle(e, '::after'); return { h: r.height + 22, w: r.width + 16, inset: a.inset ?? `${a.top} ${a.left}` }; });
  ok(b.h >= 40, `자세히: touch area extended around the small label (${Math.round(b.w)}x${Math.round(b.h)} css px)`);
  ok(errors.length === 0, `blanking: no page errors ${errors.join(' | ')}`);
  await context.close();
}

// ---------- 9. a long server sentence never spills over the cards; the whole text is in 자세히 (reviewer probe P9) ----------
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
  ok(await fitsOneScreen(page), `${width}x${height}: still one screen`);
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

// ---------- 10. short phones ----------
for (const [width, height, fit] of [[360, 640, true], [375, 553, false]]) {
  const server = serve();
  server.body = live('ledger_unavailable/display_history_unreadable', { correction: 'withdrawn' });
  const { context, page } = await open({ width, height, handler: server.handler });
  await page.goto('http://pz.test/watch/849843');
  await shown(page, server.body);
  const out = await spill(page);
  ok(out.length === 0, `${width}x${height}: longest stock sentence + change note stay inside the slot${out.length ? ` — ${out.join(' | ')}` : ''}`);
  const over = await page.evaluate(() => document.scrollingElement.scrollHeight - window.innerHeight);
  // 375x553: the zone row keeps its minimum height (the page's own rule: scroll rather than shrink the zone), so the page scrolls a little
  ok(fit ? over <= 1 : over <= 16 && await page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth + 1), fit ? `${width}x${height}: one screen` : `${width}x${height}: page scrolls ${over}px vertically (≤ 16), nothing clipped, no sideways scroll`);
  if (shots) await page.screenshot({ path: path.join(shots, `state-ledger-withdrawn-${width}x${height}.png`) });
  await context.close();
}

// ---------- 11. archive: a row with no data and no receipt keeps the old '구종 미상' (never 확인 중) ----------
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

await browser.close();
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

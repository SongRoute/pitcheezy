// A2 reference card and B2 "던지기 전 예상" on the fan screen (backend service_extras.py; payload-driven, default off).
// Canned answers on the built dist/ (no server, no model, no network), in the style of display-state-smoke.mjs.
// A2: beside the UNCHANGED "추천 없음" state a labelled reference line (chip "참고 · 기록 통계" + past frequencies) that opens the
//     full card (title, "구종 NN%", sample line, the note that it is not a recommendation); no rank badge, nothing on the strike
//     zone, no recommendation list; absent without the field, ignored beside any other state; five sizes.
// B2: one quiet line only in the detail sheet of the past key-moment pitch that carries it; nowhere in the main view, on no
//     other pitch; no 추천 / 예측 구종 wording; hidden while the realized pitch is folded (viewer delay on, device not synced).
//   npm run build && node tests/extras-smoke.mjs [screenshotDir]
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
const NOTE = '올해 9월 27일까지 이 투수가 같은 상황에서 실제로 던진 공 160개의 비율입니다. 추천이나 다음 공 예측이 아닙니다.';
// the card profile_reference gave for Grant Taylor, 0-0 against a right-handed batter (game 849831, in-process check 2026-10-11)
const CARD = { kind: 'profile_reference', reason_code: 'unsupported/profile_reference', title: '이 투수가 올해 이 카운트·타자 방향에서 자주 던진 공', chip: '참고 · 기록 통계',
  items: [['SI', '싱커', 29], ['FF', '포심', 28], ['SL', '슬라이더', 24], ['CU', '커브', 19]].map(([pitch_type, pitch_label, share_pct]) => ({ pitch_type, pitch_label, share_pct })),
  other: null, sample: { pitches: 160, batter_side: 'R', balls: 0, strikes: 0 }, as_of: '2026-09-27', stale: false, note: NOTE,
  };   // the public subset the route sends (no provenance)
const EXPECT = { kind: 'step5_expectation', lead: '던지기 전 예상', pitch_type: 'SL', pitch_label: '슬라이더', items: [{ leaf: 'narrow_whiff', label: '헛스윙', pct: 12, pct_text: '12%' }, { leaf: 'ground_ball_in_play', label: '땅볼', pct: 9, pct_text: '9%' }],
  text: '던지기 전 예상: 헛스윙 12% · 땅볼 9%', note: '2024 모델 · 참고', explain: '이 공 하나가 어떻게 끝날지 본 비율이에요. 헛스윙은 휘둘러서 못 맞힌 경우(파울팁 제외), 땅볼은 이 공을 쳐서 땅볼 타구가 된 경우예요. 스윙했을 때만 따진 비율이 아니에요.',
  basis: 'per_pitch', recorded: 'live_before_pitch' };   // the public subset the route sends
const UNSUPPORTED = { unknown_pitcher: '이 투수는 2025년까지 쌓인 기록이 없어 추천하지 않아요.', no_supported_pitch_types: '이 투수가 이 타석 방향에 던진 구종 기록이 부족해 추천하지 않아요.' };
function state(name, reason, label, message, shown) {
  return { schema: 'pitcheezy-display-state-v1', state: name, reason, detail: null, label, message, pitch_key: '849843:12:1', recommendation_shown: shown, correction: null,
    feed: { ok: true, delayed: false, lag_s: 18.2, lag_max_s: 24, lag_measured: true, lag_threshold_s: 45, thresholds_provisional: true, last_ok_at: null, failing_since: null, failing_for_s: null, received_at: null, provider_order: 'ordered' },
    ledger: { status: 'recorded', outcome: 'first', why: null, store_ok: true } };
}
const LONG = { ...CARD, items: [['SL', '슬라이더', 40], ['CH', '체인지업', 26], ['FF', '포심', 23]].map(([pitch_type, pitch_label, share_pct]) => ({ pitch_type, pitch_label, share_pct })) };
function liveBody({ reason = 'unknown_pitcher', reference = CARD, recommended = false, expectation, viewer = null } = {}) {
  const body = clone(fx.live_more);
  body.source = { kind: 'live', session_id: null }; body.receipts = null;
  if (recommended) body.display_state = state('recommended', 'ledger_recorded', '다음 공 추천', null, true);
  else { body.display_state = state('unsupported', reason, '추천 없음', UNSUPPORTED[reason], false);
    body.next = { pitch_key: '849843:12:1', rec: { provenance: 'absent', status: 'unsupported', reason: UNSUPPORTED[reason], recorded_at: null, policy: null, candidates: [] } }; }
  if (reference) body.next.reference = clone(reference);
  if (expectation !== undefined) {   // the previous plate appearance ends in a swinging strikeout on its third pitch (self-consistent, with or without the line)
    const p = body.pitches.find((x) => x.index === 32);
    p.actual = { ...p.actual, pitch_type: 'SL', pitch_label: '슬라이더', description: 'Swinging Strike', call_label: '헛스윙', result_label: '삼진', event: 'strikeout', event_label: '삼진',
      play_text: 'Nico Hoerner strikes out swinging.', x: 0.62, z: 1.45, zone_label: '낮은 오른쪽', speed_mph: 86.4 };
    const pa = body.pas.find((x) => x.pa_key === p.pa_key); Object.assign(pa, { event: 'strikeout', event_label: '삼진', result_text: '헛스윙 삼진', play_text: p.actual.play_text });
    if (expectation) p.expectation = clone(expectation);
  }
  if (viewer) { body.viewer = { enabled: true, default_s: 30, provisional: true, delay_s: 30, max_delay_s: 120, buffering: false, wait_s: 0 };
    body.features = { viewer_delay: { enabled: true, default_s: 30, sync_prompt: false, fold_actual_until_synced: true } }; }
  return body;
}
const browser = await chromium.launch();
async function open({ width = 390, height = 844, body }) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  await page.route('http://pz.test/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === GAME) { const out = clone(typeof body === 'function' ? body() : body); const after = url.searchParams.get('after'); if (after !== null) out.pitches = out.pitches.filter((p) => p.index > Number(after)); return route.fulfill({ json: out }); }
    if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
  });
  const errors = []; page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('http://pz.test/watch/849843');
  await page.locator('.sv-pitch-summary').first().waitFor();
  return { context, page, errors };
}
const box = (page, sel) => page.evaluate((s) => { const r = document.querySelector(s)?.getBoundingClientRect(); return r ? [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 2) / 2).join(',') : 'none'; }, sel);
const overflow = (page) => page.evaluate(() => [document.scrollingElement.scrollHeight - window.innerHeight, document.scrollingElement.scrollWidth - window.innerWidth]);
const spill = (page) => page.evaluate(() => { const b = document.querySelector('.sv-pitch-summary.sv-state'); const r = b.getBoundingClientRect(); const out = [];
  for (const el of b.querySelectorAll('*')) { if (el.classList.contains('sr-only') || el.closest('.sr-only')) continue; const e = el.getBoundingClientRect(); if (!e.width || !e.height) continue;
    if (e.bottom > r.bottom + 0.5 || e.right > r.right + 0.5 || e.left < r.left - 0.5) out.push(el.className || el.tagName); } return out; });
const REC_WORDS = /추천 후보|1순위|2순위|정책 선택 비중/;

// ---------- A2 ----------
for (const reason of ['unknown_pitcher', 'no_supported_pitch_types']) {
  const { context, page, errors } = await open({ body: liveBody({ reason }) });
  await page.locator('.sv-ref-teaser').waitFor();
  const slot = page.locator('.sv-pitch-summary.sv-state');
  ok(await slot.getAttribute('data-state') === 'unsupported' && (await slot.locator('.sv-chip').innerText()).trim().endsWith('추천 없음'), `A2 ${reason}: the state is still "추천 없음"`);
  ok((await slot.locator('.sv-state-msg').innerText()).trim() === UNSUPPORTED[reason], `A2 ${reason}: its sentence is unchanged`);
  const teaser = (await page.locator('.sv-ref-teaser').innerText()).replace(/\s+/g, ' ');
  ok(/^참고 · 기록 통계 자주 던진 공 싱커 29% · 포심 28% … ›$/.test(teaser), `A2 ${reason}: chip, "자주 던진 공", the first two and … ("${teaser}")`);
  ok(await page.evaluate(() => ['.sv-ref-chip', '.sv-ref-line'].every((s) => { const e = document.querySelector(s); return e.scrollWidth <= e.clientWidth + 1; })), `A2 ${reason}: nothing in the line is cut off`);
  ok(await page.locator('.sv-zone:not(.mini) .sv-rec').count() === 0 && await slot.locator('.sv-rk, .sv-rec-row, .sv-reclist').count() === 0, `A2 ${reason}: nothing on the strike zone, no rank badge, no recommendation list`);
  ok(!REC_WORDS.test(await slot.innerText()) && /추천이 아니에요/.test(await page.locator('.sv-ref-teaser').getAttribute('aria-label')), `A2 ${reason}: spoken as a reference, not a recommendation`);
  const [teal, ink] = await page.evaluate(() => [getComputedStyle(document.documentElement).getPropertyValue('--sv-teal').trim(), getComputedStyle(document.querySelector('.sv-ref-line')).color]);
  ok(await page.evaluate((t) => { const probe = document.createElement('i'); probe.style.color = t; document.body.append(probe); const c = getComputedStyle(probe).color; probe.remove(); return c; }, teal) !== ink, `A2 ${reason}: not in the recommendation colour`);
  if (shots && reason === 'unknown_pitcher') await page.screenshot({ path: path.join(shots, 'A2-reference-line-390x844.png') });
  await page.locator('.sv-ref-teaser').click();
  await page.locator('.sv-ref-card').waitFor();
  const card = (await page.locator('.sv-sheet').innerText()).replace(/\s+/g, ' ');
  ok(/참고 · 기록 통계/.test(card) && card.includes(CARD.title), `A2 ${reason}: the card has the chip and the title`);
  ok(['싱커 29%', '포심 28%', '슬라이더 24%', '커브 19%'].every((t) => card.includes(t)), `A2 ${reason}: items read "구종 NN%"`);
  ok(card.includes('2026년 9월 27일까지 같은 상황 160구') && card.includes(NOTE), `A2 ${reason}: sample line and the note that it is not a recommendation or a prediction`);
  ok(card.includes('추천 없음') && !REC_WORDS.test(card) && !/abcdef|SERVICE_PROFILE|sha|scope/.test(card) && await page.locator('.sv-sheet .sv-zone, .sv-sheet .sv-rk').count() === 0, `A2 ${reason}: the sheet repeats the state, shows no rank, zone or provenance`);
  if (shots && reason === 'unknown_pitcher') await page.screenshot({ path: path.join(shots, 'A2-reference-card-sheet-390x844.png') });
  await page.keyboard.press('Escape');
  ok(errors.length === 0, `A2 ${reason}: no page errors ${errors.join(' | ')}`);
  await context.close();
}
const shaped = (change) => { const b = liveBody(); change(b); return b; };
for (const [name, body] of [['field absent (switch off)', liveBody({ reference: null })], ['a recommendation is shown', liveBody({ recommended: true })],
  ['unsupported for an at-bat reason', shaped((b) => { b.display_state.reason = 'at_bat_rejected'; })], ['not a pending pitch (resolving)', shaped((b) => { b.current.phase = 'resolving'; })],
  ['a card without its chip', shaped((b) => { delete b.next.reference.chip; })],
  ['another kind of card', liveBody({ reference: { ...CARD, kind: 'something_else' } })], ['a card without items', liveBody({ reference: { ...CARD, items: [] } })]]) {
  const { context, page, errors } = await open({ body });
  await page.waitForTimeout(600);
  ok(await page.locator('.sv-ref-teaser, .sv-ref-card').count() === 0 && !/참고 · 기록 통계/.test(await page.locator('.sv-page').innerText()), `A2 ${name}: no reference on screen`);
  ok(errors.length === 0, `A2 ${name}: no page errors ${errors.join(' | ')}`);
  await context.close();
}
{ // R2/R3: the chip is the screen's own words; the date is the server's, in full, whatever the year
  const { context, page } = await open({ body: liveBody({ reference: { ...CARD, chip: '오늘의 추천', as_of: '2025-09-27' } }) });
  await page.locator('.sv-ref-teaser').waitFor();
  ok((await page.locator('.sv-ref-chip').innerText()).trim() === '참고 · 기록 통계' && !/오늘의 추천/.test(await page.locator('.sv-page').innerText()), 'A2: the chip always reads "참고 · 기록 통계" (a payload cannot relabel it)');
  await page.locator('.sv-ref-teaser').click();
  await page.locator('.sv-ref-card').waitFor();
  const text = await page.locator('.sv-ref-sample').innerText();
  ok(text.trim() === '2025년 9월 27일까지 같은 상황 160구' && !/올해/.test(text), `A2: the sample line states the date as sent, no "올해" assumed ("${text.trim()}")`);
  await context.close();
}
{ // stale (server: profile cutoff more than 2 days before the game date): dated up front, "기록 기준 MM/DD"; a fresh card is not
  for (const [width, height] of [[390, 844], [320, 568]]) {
    const { context, page, errors } = await open({ width, height, body: liveBody({ reference: { ...CARD, stale: true } }) });
    await page.locator('.sv-ref-teaser').waitFor();
    ok((await page.locator('.sv-ref-asof').innerText()).trim() === '기록 기준 09/27', `A2 stale ${width}x${height}: the line says "기록 기준 09/27"`);
    ok(/기록 기준 09\/27/.test(await page.locator('.sv-ref-teaser').getAttribute('aria-label')), `A2 stale ${width}x${height}: and so does its label`);
    ok((await spill(page)).length === 0 && (await overflow(page))[1] <= 0, `A2 stale ${width}x${height}: nothing spills out of the state slot or the page`);
    await page.locator('.sv-ref-teaser').click();
    await page.locator('.sv-ref-card').waitFor();
    ok((await page.locator('.sv-ref-stale').innerText()).includes('기록 기준 09/27'), `A2 stale ${width}x${height}: the sheet dates it too`);
    ok(errors.length === 0, `A2 stale ${width}x${height}: no page errors ${errors.join(' | ')}`);
    await context.close();
  }
  for (const stale of [false, undefined]) {
    const { context, page } = await open({ body: liveBody({ reference: { ...CARD, stale } }) });
    await page.locator('.sv-ref-teaser').waitFor();
    ok(await page.locator('.sv-ref-asof, .sv-ref-stale').count() === 0, `A2 stale=${stale}: no "기록 기준" line`);
    await context.close();
  }
}
{ // R4: the sheet opens only by a tap; when the card goes the sheet goes, and it does not come back by itself
  let body = liveBody();
  const { context, page, errors } = await open({ body: () => body });
  await page.locator('.sv-ref-teaser').click();
  await page.locator('.sv-ref-card').waitFor();
  body = shaped((b) => { b.display_state = state('waiting', 'resolving', '결과 확인 중', '방금 공의 결과를 기다리고 있어요.', false); b.current.phase = 'resolving'; b.next = null; });
  await page.locator('.sv-ref-card').waitFor({ state: 'detached', timeout: 6000 });
  ok(await page.locator('.sv-sheet').count() === 0, 'A2: the sheet closes when the state is no longer unsupported');
  body = liveBody();
  await page.locator('.sv-ref-teaser').waitFor({ timeout: 6000 });
  await page.waitForTimeout(2500);
  ok(await page.locator('.sv-sheet').count() === 0, 'A2: back to unsupported: the line returns, the sheet does not reopen by itself');
  ok(errors.length === 0, `A2 sheet lifecycle: no page errors ${errors.join(' | ')}`);
  await context.close();
}
{ // viewer delay on and the device not synced: the card still shows (it reveals no thrown pitch)
  const { context, page } = await open({ body: liveBody({ viewer: true }) });
  await page.locator('.sv-ref-teaser').waitFor();
  ok(await page.locator('.sv-sync-open').count() === 1 && await page.locator('.sv-ref-teaser').count() === 1, 'A2 with viewer delay on and unsynced: the reference line still shows');
  await context.close();
}
for (const [width, height] of [[390, 844], [390, 664], [375, 553], [360, 640], [320, 568]]) {   // long pitch names
  const { context, page } = await open({ width, height, body: liveBody({ reference: LONG }) });
  await page.locator('.sv-ref-teaser').waitFor();
  const text = (await page.locator('.sv-ref-teaser').innerText()).replace(/\s+/g, ' ');
  ok(await page.evaluate(() => ['.sv-ref-chip', '.sv-ref-line'].every((s) => { const e = document.querySelector(s); return e.scrollWidth <= e.clientWidth + 1; })) && /^참고 · 기록 통계 자주 던진 공 슬라이더 40%/.test(text) && /… ›$/.test(text),
    `A2 ${width}x${height} long names: not cut off ("${text}")`);
  await context.close();
}
for (const [width, height] of [[390, 844], [390, 664], [375, 553], [360, 640], [320, 568]]) {
  for (const reason of ['unknown_pitcher', 'no_supported_pitch_types']) {
    const plain = await open({ width, height, body: liveBody({ reason, reference: null }) });
    await plain.page.waitForTimeout(300);
    const base = { slot: await box(plain.page, '.sv-pitch-summary'), line: await box(plain.page, '.sv-notice.line'), over: await overflow(plain.page) };
    await plain.context.close();
    const { context, page } = await open({ width, height, body: liveBody({ reason }) });
    await page.locator('.sv-ref-teaser').waitFor();
    const over = await overflow(page), out = await spill(page);
    ok(await box(page, '.sv-pitch-summary') === base.slot && await box(page, '.sv-notice.line') === base.line, `A2 ${width}x${height} ${reason}: status slot and line do not move (${base.slot})`);
    ok(over[0] <= Math.max(1, base.over[0]) && over[1] <= 1, `A2 ${width}x${height} ${reason}: no more scroll than without the card (${over[0]} vs ${base.over[0]})`);
    ok(out.length === 0, `A2 ${width}x${height} ${reason}: the line stays inside the slot ${out.join(',')}`);
    const visible = await page.evaluate(() => { const m = document.querySelector('.sv-state-msg').getBoundingClientRect(), t = document.querySelector('.sv-ref-teaser').getBoundingClientRect(); return t.height >= 24 && t.top >= m.bottom - 0.5; });
    ok(visible, `A2 ${width}x${height} ${reason}: sentence and reference line do not overlap`);
    const whole = () => page.evaluate(() => ['.sv-ref-chip', '.sv-ref-line'].every((s) => { const e = document.querySelector(s); return e.scrollWidth <= e.clientWidth + 1; }) && /자주 던진 공/.test(document.querySelector('.sv-ref-line').innerText));
    ok(await whole(), `A2 ${width}x${height} ${reason}: chip whole, "자주 던진 공" and the shown items not cut off ("${(await page.locator('.sv-ref-teaser').innerText()).replace(/\s+/g, ' ')}")`);
    if (shots && reason === 'unknown_pitcher' && width === 375) await page.screenshot({ path: path.join(shots, 'A2-reference-line-375x553.png') });
    await context.close();
  }
}

// ---------- B2 ----------
async function toStrikeout(page) {   // the previous plate appearance (pitches 30-32), then its last pitch
  await page.locator('.sv-inning').click();
  await page.locator('.sv-pa-item:not(.open)').last().click();
  await page.locator('.sv-strip .sv-card').nth(2).click();
  await page.waitForFunction(() => document.querySelector('.sv-strip .sv-card[aria-pressed="true"]')?.textContent?.trim().startsWith('3'));
}
for (const [name, expectation] of [['switch on', EXPECT], ['switch off', null]]) {
  const { context, page, errors } = await open({ body: liveBody({ recommended: true, reference: null, expectation }) });
  ok(!/던지기 전 예상/.test(await page.locator('.sv-page').innerText()), `B2 ${name}: nothing in the live main view`);
  await toStrikeout(page);
  ok(!/던지기 전 예상|2024 모델/.test(await page.locator('.sv-page').innerText()), `B2 ${name}: nothing in the main view of the key-moment pitch`);
  await page.locator('.sv-rec-summary button').first().click();
  await page.locator('.sv-sheet .sv-detail').first().waitFor();
  const sheet = (await page.locator('.sv-sheet').innerText()).replace(/\s+/g, ' ');
  if (expectation) {
    ok(sheet.includes('던지기 전 예상: 헛스윙 12% · 땅볼 9%') && sheet.includes('2024 모델 · 참고') && sheet.includes('스윙했을 때만 따진 비율이 아니에요'), 'B2: the detail sheet of the key-moment pitch has the line, the note and the one-line explanation');
    const line = (await page.locator('.sv-expect').innerText()).replace(/\s+/g, ' ');
    ok(!/추천|예측 구종|구종 예측|최적|던져야/.test(line) && await page.locator('.sv-expect .sv-rk, .sv-expect .sv-zone').count() === 0, `B2: no 추천 / 예측 구종 wording, no rank or zone ("${line.slice(0, 60)}…")`);
    ok(await page.evaluate(() => { const e = document.querySelector('.sv-expect'); const sections = [...document.querySelectorAll('.sv-sheet .sv-detail')]; return sections.indexOf(e.closest('.sv-detail')) === 1; }), 'B2: it sits in the realized-pitch part, not beside the recommendation');
    await page.locator('.sv-expect').evaluate((e) => e.scrollIntoView({ block: 'end' }));
    if (shots) await page.screenshot({ path: path.join(shots, 'B2-detail-sheet-with-expectation-390x844.png') });
  } else {
    ok(!/던지기 전 예상|2024 모델/.test(sheet) && await page.locator('.sv-expect').count() === 0, 'B2 switch off: the same detail sheet has no such line');
    await page.locator('.sv-sheet .sv-detail').nth(1).locator('dl').evaluate((e) => e.scrollIntoView({ block: 'end' }));
    if (shots) await page.screenshot({ path: path.join(shots, 'B2-detail-sheet-switch-off-390x844.png') });
  }
  await page.keyboard.press('Escape');
  if (expectation) {
    await page.locator('.sv-strip .sv-card').nth(1).click();
    await page.locator('.sv-rec-summary button').first().click();
    await page.locator('.sv-sheet .sv-detail').first().waitFor();
    ok(await page.locator('.sv-expect').count() === 0, 'B2: another pitch of the same plate appearance has no line');
    await page.keyboard.press('Escape');
  }
  ok(errors.length === 0, `B2 ${name}: no page errors ${errors.join(' | ')}`);
  await context.close();
}
{ // viewer delay on, device not synced: the realized pitch is folded and the line with it
  const { context, page, errors } = await open({ body: liveBody({ recommended: true, reference: null, expectation: EXPECT, viewer: true }) });
  await page.locator('.sv-inning').click();
  await page.locator('.sv-pa-item:not(.open)').last().click();
  await page.locator('.sv-rec-summary button').first().waitFor();
  await page.locator('.sv-rec-summary button').first().click();
  await page.locator('.sv-sheet .sv-detail').first().waitFor();
  ok(await page.locator('.sv-expect').count() === 0 && /중계 맞추기 후 표시/.test(await page.locator('.sv-sheet').innerText()), 'B2 folded (unsynced viewer delay): the line is not shown');
  ok(errors.length === 0, `B2 folded: no page errors ${errors.join(' | ')}`);
  await context.close();
}

await browser.close();
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

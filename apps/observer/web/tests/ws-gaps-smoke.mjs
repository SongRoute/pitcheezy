// DEMO-WS-GAPS-1028 smoke: the exhibition replay screens on the World Series fixtures (fixtures/publish-ws, three final games).
//   npm run build:public && node tests/ws-gaps-smoke.mjs
// Starts `tests/ui-preview.mjs --ws` on a loopback port of its own, drives the public bundle with Playwright and checks, against
// values read from the fixture files (never typed in):
//   #9  SKIP UI-1026 미포팅 기능: the former game header carried the date list's series words ("World Series · 4차전") although the game file's own are null;
//   #7  no win-probability entry when the file carries no `win_expectancy`;
//   #11 tracking numbers with one decimal (extension) / whole numbers (spin); the record and pitch-mix sheets follow the selected pitch;
//   #10 the 정정됨 tag only for a settled correction (reason null); held pitches are counted in one line of the record sheet;
//   #2  the full UI-1026 pitch-mix sheet preserves each location/label (coincident marks may merge, pitcher's view).
// Exit 0 = all checks passed; the server it started is stopped either way.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), web = path.join(here, '..');
const { chromium } = createRequire(path.join(web, 'package.json'))('playwright');
const fixture = (date, pk) => JSON.parse(readFileSync(path.join(here, 'fixtures', 'publish-ws', date, 'v1', 'game', `${pk}.json`), 'utf8'));
const G = { 849805: fixture('2026-10-24', 849805), 849804: fixture('2026-10-27', 849804), 849803: fixture('2026-10-28', 849803) };
const list = (date) => JSON.parse(readFileSync(path.join(here, 'fixtures', 'publish-ws', date, 'v1', 'games', `${date}.json`), 'utf8'));

// ---- the preview server (port 0: the OS picks one; the script prints the address) ----
const server = spawn(process.execPath, [path.join(here, 'ui-preview.mjs'), '--ws', '--port', '0'], { cwd: web, stdio: ['ignore', 'pipe', 'pipe'] });
const base = await new Promise((resolve, reject) => {
  let buf = ''; const t = setTimeout(() => reject(new Error('preview did not start')), 15000);
  server.stdout.on('data', (d) => { buf += d; const m = /open\s+(http:\/\/127\.0\.0\.1:\d+\/pub\/)/.exec(buf); if (m) { clearTimeout(t); resolve(m[1]); } });
  server.stderr.on('data', (d) => { buf += d; });
  server.on('exit', (c) => reject(new Error(`preview exited ${c}: ${buf}`)));
});
let failed = 0, browser = null;
const check = async (name, fn) => { try { await fn(); console.log(`ok   ${name}`); } catch (e) { failed++; console.log(`FAIL ${name}: ${e.message}`); } };
try {
  browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true, locale: 'ko-KR', timezoneId: 'Asia/Seoul' });
  const page = await ctx.newPage();
  const errors = []; page.on('pageerror', (e) => errors.push(String(e))); page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const squash = (s) => (s ?? '').replace(/\s+/g, ' ').trim();
  const text = async (sel) => squash(await page.locator(sel).first().innerText().catch(() => ''));
  const sheet = page.locator('.sv-sheet');
  const closeSheet = async () => { if (await sheet.count()) { await sheet.locator('.sv-sheet-head button', { hasText: '닫기' }).click(); await sheet.waitFor({ state: 'detached' }); } };
  const unfold = async () => { const toggle = page.locator('.sv-atbat-toggle'); if (await toggle.getAttribute('aria-expanded') === 'false') await toggle.click(); };
  const openGame = async (pk) => { await page.goto(`${base}?game=${pk}`); await page.locator('.sv-pitch-summary').waitFor(); await page.waitForTimeout(500); await unfold(); };
  const nav = (name) => page.locator('nav.sv-bottom-nav button', { hasText: name }).click().then(() => sheet.waitFor());
  const gotoPitch = async (pk, index) => {
    const j = G[pk], p = j.pitches.find(x => x.index === index), half = `${p.inning}:${p.half}`;
    const nth = j.pas.filter(x => `${x.inning}:${x.half}` === half).findIndex(x => x.pa_key === p.pa_key);
    await closeSheet(); await page.locator('button.sv-inning').click(); await sheet.waitFor();
    await sheet.locator('select').selectOption(half); await page.waitForTimeout(100);
    await sheet.locator('.sv-pa-item').nth(nth).click(); await page.waitForTimeout(250);
    const first = j.pas.find(x => x.pa_key === p.pa_key).first_index;
    for (const q of j.pitches.filter(x => x.index > first && x.index <= index)) await sheet.getByRole('button', { name: '다음 공', exact: true }).click();
    await sheet.locator('.sv-sheet-head h2').filter({ hasText: `${p.pitch_number}구` }).waitFor();
    await closeSheet(); await unfold(); return p;
  };
  const dd = async (label) => { const dt = (await sheet.locator('.sv-dl dt').allInnerTexts()).map(squash), dl = (await sheet.locator('.sv-dl dd').allInnerTexts()).map(squash); return dl[dt.indexOf(label)]; };
  const fixed1 = (v) => (Math.round(v * 10) / 10).toFixed(1);

  // ---- G4 849803: series words, no win entry, number formats ----
  const g4 = G[849803], row4 = list('2026-10-28').games.find(g => g.game_pk === 849803);
  await openGame(849803);
  await check('#9 header series words from the date list (game file has null)', () => {
    assert.equal(g4.game.series_description, null); assert.equal(row4.series_description, 'World Series'); assert.equal(row4.series_game_number, 4);
  });
  console.log('SKIP #9 header series words / aria-label: 미포팅 기능 — UI-1026 .sv-hd-1 has no series metadata (G4).');
  await check('UI-1026 score header is accessible', async () => assert.ok(await page.locator('.sv-hd-1 h1').getAttribute('aria-label')));
  await check('#7 no win-probability entry without win_expectancy', async () => { assert.equal('win_expectancy' in g4, false); await nav('기록'); assert.equal(await sheet.getByRole('button', { name: '승리확률', exact: true }).count(), 0); await closeSheet(); });
  await page.locator('.sv-pitch-row').last().click(); await sheet.waitFor();
  const t0 = g4.pitches[0].actual.tracking;
  await check('#11 extension one decimal, spin whole', async () => { assert.equal(await dd('익스텐션'), `${fixed1(t0.extension_feet)} ft`); assert.equal(await dd('회전수'), `${Math.round(t0.spin_rate_rpm)} rpm`); assert.doesNotMatch(await dd('익스텐션'), /\d\.\d{2,}/); });
  await closeSheet();
  // the first settled correction of G4 keeps its tag
  const settled4 = g4.pitches.find(p => p.receipt?.corrected && !p.receipt.reason);
  await gotoPitch(849803, settled4.index);
  await check('#10 settled correction: tag on the caption and the card', async () => { assert.equal(await text('.sv-actual-line .sv-tag'), '정정됨'); assert.ok(await page.locator('.sv-atbat-list .sv-card-tag').count() >= 1); });
  // record sheets as of pitch 100
  const mid = await gotoPitch(849803, 100);
  const expectPas = g4.pas.filter(p => p.first_index <= 100), open = g4.pas.find(p => p.first_index <= 100 && 100 < p.last_index);
  await nav('기록');
  await check('#11 record log cut at the selected pitch', async () => {
    assert.equal(await sheet.locator('.sv-recent-pas li').count(), expectPas.length);
    const first = squash(await sheet.locator('.sv-recent-pas li').first().innerText());
    if (open) { const thrown = g4.pitches.filter(p => p.pa_key === open.pa_key && p.index <= 100).length; assert.match(first, new RegExp(`타석 진행 중 · ${thrown}구$`)); }
    assert.match(await text('.sv-basis'), new RegExp(`고른 공\\(${mid.inning}회 ${mid.half === 'Top' ? '초' : '말'} ${mid.pitch_number}구\\)까지의 타석`));
  });
  const corr4 = g4.pitches.filter(p => p.index <= 100 && p.receipt?.corrected).length, held4 = g4.pitches.filter(p => p.index <= 100 && p.receipt?.reason === 'held_last_accepted').length;
  const tagged4 = g4.pitches.filter(p => p.index <= 100 && p.receipt?.corrected && !p.receipt.reason).length;
  await check('#10 corrections counted in one line (as of the pitch)', async () => { const line = await text('.sv-corrections'); if (corr4 || held4) { assert.match(line, new RegExp(`^고른 공까지 정정된 기록 ${corr4}구\\(투구 카드에 ‘정정됨’ 표시 ${tagged4}구`)); if (held4) assert.match(line, new RegExp(`보류한 공 ${held4}구`)); } else assert.equal(line, ''); });
  await sheet.locator('.sv-row button', { hasText: '박스' }).first().click(); await page.waitForTimeout(150);
  await check('#11 box basis line when a past pitch is selected', async () => assert.match(await text('.sv-basis'), /경기 종료 시점 기준/));
  await closeSheet();
  // the first pitcher's last pitch: the mix sheet shows only his pitches up to there, with the scatter
  const jaxLast = g4.pitches.filter(p => p.pitcher.id === g4.pitches[0].pitcher.id).at(-1);
  const his = g4.pitches.filter(p => p.index <= jaxLast.index && p.pitcher.id === jaxLast.pitcher.id && p.actual?.pitch_type);
  await gotoPitch(849803, jaxLast.index); await nav('구종');
  await check('#2/#11 pitch mix as of the selected pitch: table and full location map', async () => {
    assert.match(await text('.sv-sheet-body > p'), new RegExp(`확인된 ${his.length}구 기준 · 고른 공\\(`));
    const table = await sheet.locator('.sv-mix-table tbody tr').evaluateAll(rows => rows.map(r => [...r.cells].map(c => c.textContent)));
    assert.equal(table.reduce((n, row) => n + Number(row[1]), 0), his.length);
    assert.equal(table.length, new Set(his.map(p => p.actual.pitch_type)).size);
    await sheet.getByRole('button', { name: '시트 크게 보기', exact: true }).click();
    const located = his.filter(p => p.actual.x !== null && p.actual.z !== null);
    await sheet.locator('.sv-mix-map').waitFor();
    const drawn = await sheet.locator('.sv-mix-map .sv-pa-dot').evaluateAll(els => els.map(e => ({
      x: Number(e.dataset.x), z: Number(e.dataset.z), labels: e.querySelector('text').textContent.split('·'), out: e.classList.contains('out'),
    })));
    // UI-1026 merges coincident marks; expanding labels preserves pitch multiplicity.
    assert.deepEqual(drawn.flatMap(d => d.labels.map(() => `${d.x},${d.z}`)).sort(), located.map(p => `${p.actual.x},${p.actual.z}`).sort());
    const key = await text('.sv-mix-key');
    for (const row of table) assert.ok(key.includes(`=${row[0]}`));
    const abbrev = new Map(key.split(' · ').map(item => item.split('=')).map(([letter, label]) => [label, letter]));
    assert.deepEqual(drawn.flatMap(d => d.labels.map(label => `${label}@${d.x},${d.z}`)).sort(),
      located.map(p => `${abbrev.get(p.actual.pitch_label)}@${p.actual.x},${p.actual.z}`).sort());
    const bounds = g4.zone_bounds, xMax = .83 + .42, zLo = bounds.bottom - .45, zHi = bounds.top + .45;
    for (const d of drawn) assert.equal(d.out, Math.abs(d.x) > xMax || d.z < zLo || d.z > zHi);
  });
  await closeSheet();

  // ---- G3 849804: held pitches carry no card tag; one aggregate line ----
  const g3 = G[849804];
  await openGame(849804);
  await check('#10 G3: no tag on held pitches', async () => { assert.equal(g3.pitches[0].receipt.reason, 'held_last_accepted'); assert.equal(g3.pitches[0].receipt.corrected, true);
    assert.equal(await page.locator('.sv-actual-line .sv-tag').count(), 0); assert.equal(await page.locator('.sv-atbat-list .sv-card-tag').count(), 0); });
  await page.locator('.sv-pitch-row').last().click(); await sheet.waitFor();
  await check('#10 G3 detail says the record is held', async () => assert.ok((await sheet.locator('.sv-detail > p.sv-muted').allInnerTexts()).some(t => /마지막으로 받아들인 기록/.test(t))));
  await closeSheet();
  await gotoPitch(849804, g3.pitches.at(-1).index); await nav('기록');
  const corr3 = g3.pitches.filter(p => p.receipt?.corrected).length, held3 = g3.pitches.filter(p => p.receipt?.reason === 'held_last_accepted').length;
  await check('#10 G3 aggregate line: every corrected pitch, none tagged, the held count, the overlap said', async () => { const line = await text('.sv-corrections');
    assert.match(line, new RegExp(`^고른 공까지 정정된 기록 ${corr3}구\\(투구 카드에 ‘정정됨’ 표시 0구, 보류 중이라 표시하지 않은 ${corr3}구\\)`)); assert.match(line, new RegExp(`보류한 공 ${held3}구`)); assert.match(line, /겹칠 수 있어요/); });
  await closeSheet();

  // ---- G1 849805: series words, Misiorowski's scatter ----
  const g1 = G[849805];
  await openGame(849805);
  console.log('SKIP #9 G1 header series words: 미포팅 기능 — UI-1026 header has no series line.');
  const misoLast = g1.pitches.filter(p => p.pitcher.id === g1.pitches[0].pitcher.id).at(-1);
  const miso = g1.pitches.filter(p => p.index <= misoLast.index && p.pitcher.id === misoLast.pitcher.id && p.actual?.pitch_type);
  await gotoPitch(849805, misoLast.index); await nav('구종');
  await sheet.getByRole('button', { name: '시트 크게 보기', exact: true }).click();
  await check('#2 Misiorowski: every located pitch represented', async () => assert.equal(await sheet.locator('.sv-mix-map .sv-pa-dot text').evaluateAll(els => els.reduce((n, e) => n + e.textContent.split('·').length, 0)), miso.filter(p => p.actual.x !== null && p.actual.z !== null).length));
  await closeSheet();
  await check('no page errors', () => assert.deepEqual(errors, []));
} finally {
  await browser?.close();
  server.kill('SIGTERM');
}
console.log(failed ? `ws-gaps-smoke: ${failed} check(s) FAILED` : 'ws-gaps-smoke: all checks passed');
process.exit(failed ? 1 : 0);

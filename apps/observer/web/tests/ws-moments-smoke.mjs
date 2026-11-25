// Smoke test of the replayed mid-game moments of World Series game 4 (fixtures/publish-ws-moments/849803, README there) on the PUBLIC
// bundle: for every moment SELECTION.json lists, `node tests/ui-preview.mjs --ws-moment <label>` is started and the screen must show the
// pending pitch's state as the moment body says it — the stage headed '다음 공 추천' with its N구 and exactly the recorded candidates (the
// visible UI-1026 heading carries N구 and the sr-only heading reads 추천), or 추천 없음 with the A2 teaser where the body carries next.reference — with the
// win-probability sheet carrying no number (현재 승리확률 정보 없음), no page error and no 갱신 지연 notice right
// after opening (the now clock). Read-only. Needs dist-public (npm run build:public) and Playwright.
//   node tests/ws-moments-smoke.mjs
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, 'fixtures', 'publish-ws-moments', '849803');
const selection = JSON.parse(readFileSync(path.join(root, 'SELECTION.json'), 'utf8'));
const labels = Object.entries(selection.chosen).filter(([, v]) => v).map(([k]) => k);
const squash = (s) => (s ?? '').replace(/\s+/g, ' ').trim();
let failed = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}${detail ? ` — ${detail}` : ''}`); if (!ok) failed++; };

async function serve(label) {
  const child = spawn(process.execPath, [path.join(here, 'ui-preview.mjs'), '--ws-moment', label, '--port', '0'], { stdio: ['ignore', 'pipe', 'pipe'] });
  const url = await new Promise((resolve, reject) => {
    let text = '';
    const look = (chunk) => { text += chunk; const m = text.match(/http:\/\/127\.0\.0\.1:(\d+)\//); if (m) resolve(`http://127.0.0.1:${m[1]}/pub/`); };
    child.stdout.on('data', look); child.stderr.on('data', look);
    child.on('exit', (code) => reject(new Error(`preview exited ${code}: ${text}`)));
    setTimeout(() => reject(new Error(`no port line: ${text}`)), 15000);
  });
  return { child, url };
}

const browser = await chromium.launch();
try {
  for (const label of labels) {
    const body = JSON.parse(readFileSync(path.join(root, label, 'v1', 'game', '849803.json'), 'utf8'));
    const next = body.next ?? {}, disp = body.display_state ?? {};
    const expectedN = Number(String(next.pitch_key ?? '').split(':')[2]);
    const candidates = next.rec?.candidates ?? [];
    const { child, url } = await serve(label);
    try {
      const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, locale: 'ko-KR', timezoneId: 'Asia/Seoul' });
      const page = await ctx.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(String(e)));
      await page.goto(`${url}?game=849803`);
      await page.locator('.sv-pitch-summary').waitFor();
      await page.waitForTimeout(900);
      const stage = page.locator('.sv-stage');
      const stageLabel = await stage.getAttribute('aria-label');
      const srHeading = squash(await stage.locator('h2.sr-only').first().innerText().catch(() => ''));
      const recHeading = squash(await page.locator('.sv-rec-head h2').innerText());
      const pitchKey = await page.locator('.sv-state').getAttribute('data-pitch');
      const rows = (await page.locator('.sv-stage .sv-reclist:not(.empty) .sv-rec-cell').allInnerTexts()).map(squash);
      const stateMsg = squash(await page.locator('.sv-state-msg').first().textContent().catch(() => ''));
      const live = squash(await page.locator('.sv-hd-status').first().innerText().catch(() => ''));
      const teaser = await page.locator('.sv-ref-teaser').count();
      const notice = live;
      check(body.source?.kind === 'live' && typeof next.pitch_key === 'string', `${label}: a live body with a pending pitch`, next.pitch_key);
      check(live.length > 0, `${label}: the live indicator is drawn`, live);
      check(stageLabel === '다음 공 추천' && recHeading.endsWith(`${expectedN}구`) && pitchKey === next.pitch_key, `${label}: the stage is headed 다음 공 추천 with ${expectedN}구`, `${stageLabel} / ${recHeading} / ${pitchKey}`);
      if (disp.state === 'recommended') {
        check(srHeading === '추천', `${label}: UI-1026 screen-reader heading 추천 (N구 checked above)`, srHeading);
        check(rows.length === candidates.length && candidates.length > 0, `${label}: one row per recorded candidate (${candidates.length})`, `${rows.length} rows`);
        const names = await page.locator('.sv-reclist:not(.empty) .sv-rec-type').allTextContents();
        check(JSON.stringify(names) === JSON.stringify(candidates.map(c => c.pitch_label)), `${label}: exact recorded candidate order`, names.join(' | '));
        check(teaser === 0, `${label}: no A2 teaser beside a recommendation`);
      } else {
        check(rows.length === 0, `${label}: no candidate rows without a recommendation`, `${rows.length} rows`);
        check(stateMsg.length > 0, `${label}: the state sentence is shown`, stateMsg);
        if (next.reference) {
          check(teaser === 1, `${label}: the A2 teaser is drawn for next.reference`);
          await page.locator('.sv-ref-teaser').click();
          const sheet = page.locator('.sv-sheet');
          await sheet.waitFor();
          const chip = squash(await sheet.locator('.sv-ref-chip').first().innerText().catch(() => ''));
          const items = await sheet.locator('.sv-ref-card li').count();
          const expectedItems = next.reference.items.length + ((next.reference.other?.share_pct ?? 0) > 0 ? 1 : 0);   // + the 그 밖의 공 row
          check(chip === '참고 · 기록 통계' && items === expectedItems, `${label}: reference sheet chip and ${expectedItems} rows`, `${chip} / ${items}`);
          await sheet.locator('.sv-sheet-head button', { hasText: '닫기' }).click();
        }
      }
      // UI-1026: 기록 → 승리확률; unavailable remains text only.
      await page.locator('nav.sv-bottom-nav button', { hasText: '기록' }).click();
      await page.locator('.sv-sheet').getByRole('button', { name: '승리확률', exact: true }).click();
      const winText = squash(await page.locator('.sv-sheet-body').innerText());
      check(body.win_expectancy?.status !== 'available' && winText.includes('현재 승리확률 정보 없음') && !/\d+%/.test(winText)
        && await page.locator('.sv-we-bars').count() === 0, `${label}: unavailable win probability has no number`, winText);
      await page.locator('.sv-sheet').getByRole('button', { name: '닫기', exact: true }).click();
      check(!notice.includes('갱신 지연'), `${label}: no 갱신 지연 right after opening (now clock)`, notice);
      check(errors.length === 0, `${label}: no page errors`, errors.join('; '));
      await ctx.close();
    } finally {
      child.kill();
    }
  }
} finally {
  await browser.close();
}
console.log(failed ? `ws-moments-smoke: ${failed} check(s) failed` : 'ws-moments-smoke: all checks passed');
process.exit(failed ? 1 : 0);

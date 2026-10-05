// layout-final-017 copy of review-integration-002/live_pinned_stale_fit.mjs (targets unchanged). LIVE past-pitch pinned + stale/offline states at 320x568 and 375x553, default and 125% text, on the route-mocked
// fixture of tests/responsive-fit-smoke.mjs (no server, output only to argv[2]). Headless Chromium, not Safari.
// Responsive fit (responsive-fix-013): the game screen must fit one screen with NO page scroll at CSS widths >= 320 and
// heights >= 553, at default text and at 125% text, with the zone as the largest block and every control on screen.
// Same metrics and the same 125% text injection as review-integration-002/integrated_qa.mjs (html{font-size:125%} +
// -webkit-text-size-adjust:125%, which Chromium mobile emulation applies to px text). Headless Chromium only — NOT Safari,
// NOT a physical iPhone. 320x460 and 288x512 (page-zoom proxy) are reported as INFO (page-scroll fallback), never as pass.
// Canned CONTRACT answers from tests/fixtures/service-849843.json on the built dist/; no server, no files outside argv[2].
//   npm run build && node tests/responsive-fit-smoke.mjs [outDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const out = process.argv[2]; if (out) mkdirSync(out, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const STRICT = [[320, 568], [360, 640], [375, 553], [390, 600], [375, 667], [390, 844]];
const INFO = [[320, 460], [288, 512]];
let failures = 0; const rows = [];
const ok = (cond, msg, info) => { if (!cond) failures++; console.log(cond ? 'ok  ' : 'FAIL', msg, cond ? '' : JSON.stringify(info)); };

const browser = await chromium.launch();
async function measure(w, h, big, which, state) {
  let failApi = false;
  const c = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ko-KR' });
  const p = await c.newPage();
  await p.route('http://pz.test/**', (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === '/api/service/games/849843') {
      if (failApi) return route.fulfill({ status: 503, json: { detail: '기록 형식을 확인할 수 없습니다.' } });
      const body = JSON.parse(JSON.stringify(which === 'live' ? fx.live : fx.final));
      const after = u.searchParams.get('after');
      if (after !== null) body.pitches = body.pitches.filter((x) => x.index > Number(after));
      return route.fulfill({ json: body });
    }
    if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this test' } });
    const f = path.join(dist, u.pathname); const t = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(t), contentType: TYPES[path.extname(t)] ?? 'application/octet-stream' });
  });
  await p.goto(`http://pz.test/watch/849843${which === 'live' ? '?demo=fixture' : ''}`);
  await p.waitForSelector('.sv-board');
  if (state === 'pinned' || state === 'stale') { await p.locator('.sv-pitches .sv-pitch:not(.next)').first().click(); await p.waitForTimeout(300); }
  if (state === 'stale') { failApi = true; await p.waitForTimeout(4500); }
  if (big) { await p.addStyleTag({ content: 'html{font-size:125% !important;-webkit-text-size-adjust:125%}' }); await p.waitForTimeout(200); }
  const L = await p.evaluate(() => {
    const modeBtns = [...document.querySelectorAll('.sv-mode button')].map((b) => { const r = b.getBoundingClientRect(); return { t: b.textContent.trim(), aria: b.getAttribute('aria-label'), in: r.width > 0 && r.bottom <= innerHeight + 1 && r.right <= innerWidth + 1 }; });
    const modeText = document.querySelector('.sv-mode')?.innerText ?? '';
    const modeOverlap = (() => { const bs = [...document.querySelectorAll('.sv-mode .sv-mode-text, .sv-mode button')].map((e) => e.getBoundingClientRect()); for (let i = 0; i < bs.length; i++) for (let j = i + 1; j < bs.length; j++) { const a = bs[i], b = bs[j]; if (a.bottom > b.top + 1 && b.bottom > a.top + 1 && a.right > b.left + 1 && b.right > a.left + 1) return true; } return false; })();
    const vis = (sel) => { const e = document.querySelector(sel); if (!e) return null; const r = e.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, h: r.height, w: r.width }; };
    const se = document.scrollingElement;
    const cands = [...document.querySelectorAll('.sv-cand')].map((e) => { const r = e.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height), getComputedStyle(e.querySelector('.sv-cand-type') ?? e).fontSize]; });
    const clipped = [...document.querySelectorAll('.sv-page *')].filter((e) => { const s = getComputedStyle(e); return (s.overflowY === 'hidden' || s.overflow === 'hidden') && e.scrollHeight > e.clientHeight + 2 && e.clientHeight > 0 && !e.closest('.sv-sheet') && !e.classList.contains('sr-only'); })  /* .sr-only = intentional screen-reader text */.map((e) => e.className).slice(0, 5);
    const offscreen = [...document.querySelectorAll('.sv-page button, .sv-page a')].filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && (r.bottom > innerHeight + 1 || r.right > innerWidth + 1); }).map((e) => e.textContent.trim().slice(0, 20)).slice(0, 8);
    const hidden = ['.sv-score', '.sv-count', '.sv-bases', '.sv-matchup', '.sv-title', '.sv-pa-open', '.sv-pitches', '.sv-rec-head', '.sv-actual-line'].filter((s) => { const e = document.querySelector(s); return !e || e.getBoundingClientRect().height === 0; });
    const paResult = (() => { const e = document.querySelector('.sv-pa-open small'); return !!e && e.getBoundingClientRect().height > 0; })();
    const overlap = (() => { const els = [...document.querySelectorAll('.sv-top, .sv-board, .sv-mode, .sv-rec-head, .sv-zone-wrap, .sv-cand, .sv-norec, .sv-actual-line, .sv-pa-row, .sv-pitches')]; for (let i = 0; i < els.length; i++) for (let j = i + 1; j < els.length; j++) { const a = els[i].getBoundingClientRect(), b = els[j].getBoundingClientRect(); if (els[i].contains(els[j]) || els[j].contains(els[i])) continue; if (a.bottom > b.top + 1 && b.bottom > a.top + 1 && a.right > b.left + 1 && b.right > a.left + 1) return [els[i].className, els[j].className]; } return null; })();
    return { modeBtns, modeText, modeOverlap, scrollH: se.scrollHeight, clientH: se.clientHeight, innerH: innerHeight, scrollW: se.scrollWidth, innerW: innerWidth,
      zone: vis('.sv-zone-wrap svg') ?? vis('.sv-zone-wrap'), board: vis('.sv-board'), recbox: vis('.sv-recbox'), nav: vis('.sv-nav'), cands, clipped, offscreen, hidden, paResult, overlap,
      escaped: (() => { const box = document.querySelector('.sv-recbox')?.getBoundingClientRect(); return box ? [...document.querySelectorAll('.sv-cand')].filter((c) => { const r = c.getBoundingClientRect(); return r.bottom > box.bottom + 1 || r.top < box.top - 1; }).length : 0; })(),
      charWrap: [...document.querySelectorAll('.sv-cand-type, .sv-title')].filter((e) => { const r = document.createRange(); r.selectNodeContents(e); const lines = new Set([...r.getClientRects()].map((x) => Math.round(x.top))); return e.textContent.trim().length <= 5 && lines.size > 1; }).map((e) => e.textContent),
      titleCut: (() => { const t = document.querySelector('.sv-title'); return !!t && t.scrollWidth > t.clientWidth + 1; })(),
      shareLabel: /정책 선택 비중/.test(document.body.innerText), rootFont: getComputedStyle(document.documentElement).fontSize, candFont: getComputedStyle(document.querySelector('.sv-cand-type')).fontSize };
  });
  if (out) await p.screenshot({ path: path.join(out, `${which}-${state}-${w}x${h}${big ? '-font125' : ''}.png`) });
  await c.close();
  return L;
}


for (const state of ['pinned', 'stale']) for (const [w, h] of [[320, 568], [375, 553]]) for (const big of [false, true]) {
  const L = await measure(w, h, big, 'live', state);
  const tag = `live-${state} ${w}x${h}${big ? '-font125' : ''}`;
  const zoneArea = L.zone ? Math.round(L.zone.w * L.zone.h) : 0;
  const others = [L.board, L.recbox, L.nav].filter(Boolean).map((r) => Math.round(r.w * r.h));
  rows.push({ tag, scroll: L.scrollH - L.innerH, modeText: L.modeText, modeBtns: L.modeBtns, zoneArea, others, offscreen: L.offscreen });
  ok(L.scrollH <= L.clientH + 1 && L.scrollH <= L.innerH + 1, `${tag}: no vertical page scroll`, [L.scrollH, L.clientH]);
  ok(!L.offscreen.length, `${tag}: every control inside the first viewport`, L.offscreen);
  ok(!L.clipped.length, `${tag}: no overflow-hidden clipping`, L.clipped);
  ok(!L.overlap && !L.modeOverlap, `${tag}: no overlapping blocks or mode-bar items`, [L.overlap, L.modeOverlap]);
  ok(L.scrollW <= L.innerW + 1, `${tag}: no horizontal overflow`, [L.scrollW, L.innerW]);
  ok(zoneArea >= Math.max(...others), `${tag}: zone is the largest block`, [zoneArea, others]);
  const has = (re) => L.modeBtns.some((b) => re.test(b.t + ' ' + (b.aria ?? '')) && b.in);
  if (state === 'pinned') ok(/지난 기록/.test(L.modeText) && has(/최신으로/) && has(/자동/) && has(/새로고침/), `${tag}: pinned text + 최신으로 + 자동 + 새로고침 visible`, L.modeBtns);
  else ok(/연결 끊김|늦어지고/.test(L.modeText) && has(/새로고침/), `${tag}: stale/offline text + 새로고침 retry visible`, [L.modeText, L.modeBtns]);
  ok(!/LIVE/.test(L.modeText), `${tag}: viewer bar does not reuse the LIVE game label`, L.modeText);
}
await browser.close();
if (out) writeFileSync(path.join(out, 'metrics.json'), JSON.stringify(rows, null, 1));
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

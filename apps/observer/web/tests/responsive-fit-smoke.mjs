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
async function measure(w, h, big, which) {
  const c = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ko-KR' });
  const p = await c.newPage();
  await p.route('http://pz.test/**', (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === '/api/service/games/849843') {
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
  if (big) { await p.addStyleTag({ content: 'html{font-size:125% !important;-webkit-text-size-adjust:125%}' }); await p.waitForTimeout(200); }
  const L = await p.evaluate(() => {
    const vis = (sel) => { const e = document.querySelector(sel); if (!e) return null; const r = e.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, h: r.height, w: r.width }; };
    const se = document.scrollingElement;
    const cands = [...document.querySelectorAll('.sv-cand')].map((e) => { const r = e.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height), getComputedStyle(e.querySelector('.sv-cand-type') ?? e).fontSize]; });
    const clipped = [...document.querySelectorAll('.sv-page *')].filter((e) => { const s = getComputedStyle(e); return (s.overflowY === 'hidden' || s.overflow === 'hidden') && e.scrollHeight > e.clientHeight + 2 && e.clientHeight > 0 && !e.closest('.sv-sheet'); }).map((e) => e.className).slice(0, 5);
    const offscreen = [...document.querySelectorAll('.sv-page button, .sv-page a')].filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && (r.bottom > innerHeight + 1 || r.right > innerWidth + 1); }).map((e) => e.textContent.trim().slice(0, 20)).slice(0, 8);
    const hidden = ['.sv-score', '.sv-count', '.sv-bases', '.sv-matchup', '.sv-title', '.sv-pa-open', '.sv-pitches', '.sv-rec-head', '.sv-actual-line'].filter((s) => { const e = document.querySelector(s); return !e || e.getBoundingClientRect().height === 0; });
    const paResult = (() => { const e = document.querySelector('.sv-pa-open small'); return !!e && e.getBoundingClientRect().height > 0; })();
    const overlap = (() => { const els = [...document.querySelectorAll('.sv-top, .sv-board, .sv-mode, .sv-rec-head, .sv-zone-wrap, .sv-cand, .sv-norec, .sv-actual-line, .sv-pa-row, .sv-pitches')]; for (let i = 0; i < els.length; i++) for (let j = i + 1; j < els.length; j++) { const a = els[i].getBoundingClientRect(), b = els[j].getBoundingClientRect(); if (els[i].contains(els[j]) || els[j].contains(els[i])) continue; if (a.bottom > b.top + 1 && b.bottom > a.top + 1 && a.right > b.left + 1 && b.right > a.left + 1) return [els[i].className, els[j].className]; } return null; })();
    return { scrollH: se.scrollHeight, clientH: se.clientHeight, innerH: innerHeight, scrollW: se.scrollWidth, innerW: innerWidth,
      zone: vis('.sv-zone-wrap svg') ?? vis('.sv-zone-wrap'), board: vis('.sv-board'), recbox: vis('.sv-recbox'), nav: vis('.sv-nav'), cands, clipped, offscreen, hidden, paResult, overlap,
      escaped: (() => { const box = document.querySelector('.sv-recbox')?.getBoundingClientRect(); return box ? [...document.querySelectorAll('.sv-cand')].filter((c) => { const r = c.getBoundingClientRect(); return r.bottom > box.bottom + 1 || r.top < box.top - 1; }).length : 0; })(),
      charWrap: [...document.querySelectorAll('.sv-cand-type, .sv-title')].filter((e) => { const r = document.createRange(); r.selectNodeContents(e); const lines = new Set([...r.getClientRects()].map((x) => Math.round(x.top))); return e.textContent.trim().length <= 5 && lines.size > 1; }).map((e) => e.textContent),
      titleCut: (() => { const t = document.querySelector('.sv-title'); return !!t && t.scrollWidth > t.clientWidth + 1; })(),
      shareLabel: /정책 선택 비중/.test(document.body.innerText), rootFont: getComputedStyle(document.documentElement).fontSize, candFont: getComputedStyle(document.querySelector('.sv-cand-type')).fontSize };
  });
  if (out) await p.screenshot({ path: path.join(out, `${which}-${w}x${h}${big ? '-font125' : ''}.png`) });
  await c.close();
  return L;
}

for (const which of ['final', 'live']) {
  for (const [w, h] of [...STRICT, ...INFO]) for (const big of [false, true]) {
    const L = await measure(w, h, big, which);
    const tag = `${which} ${w}x${h}${big ? '-font125' : ''}`;
    const zoneArea = L.zone ? Math.round(L.zone.w * L.zone.h) : 0;
    const others = [L.board, L.recbox, L.nav].filter(Boolean).map((r) => Math.round(r.w * r.h));
    rows.push({ tag, scroll: L.scrollH - L.innerH, zoneArea, others, offscreen: L.offscreen, candFont: L.candFont, paResult: L.paResult });
    const strict = w >= 320 && h >= 553;
    if (strict) {
      ok(L.scrollH <= L.clientH + 1 && L.scrollH <= L.innerH + 1, `${tag}: scrollHeight ${L.scrollH} <= clientHeight ${L.clientH}`, [L.scrollH, L.clientH]);
      ok(!L.offscreen.length, `${tag}: every control inside the first viewport`, L.offscreen);
      ok(zoneArea >= Math.max(...others), `${tag}: zone ${zoneArea} is the largest block`, others);
    } else console.log('INFO', `${tag}: page scroll ${L.scrollH - L.innerH}px, offscreen before scroll ${JSON.stringify(L.offscreen)}, zone ${zoneArea} vs ${JSON.stringify(others)}`);
    ok(!L.clipped.length, `${tag}: no overflow-hidden clipping`, L.clipped);
    ok(!L.hidden.length, `${tag}: score/count/bases/players/date/PA/pitch nav/rec head/actual all shown`, L.hidden);
    ok(L.paResult, `${tag}: PA result line shown in the PA button`, L.paResult);
    ok(!L.overlap, `${tag}: no overlapping blocks`, L.overlap);
    ok(L.cands.length >= 1 && L.cands.every((x) => Math.abs(x[0] - L.cands[0][0]) <= 1 && Math.abs(x[1] - L.cands[0][1]) <= 1 && x[2] === L.cands[0][2]), `${tag}: candidates equal size/font`, L.cands);
    ok(!L.escaped, `${tag}: candidates stay inside their column`, L.escaped);
    ok(!L.charWrap.length, `${tag}: short pitch names not broken across lines`, L.charWrap);
    ok(!L.titleCut, `${tag}: date title not cut`, L.titleCut);
    ok(L.shareLabel, `${tag}: 정책 선택 비중 label visible`, null);
    ok(L.scrollW <= L.innerW + 1, `${tag}: no horizontal overflow`, [L.scrollW, L.innerW]);
  }
}
await browser.close();
if (out) writeFileSync(path.join(out, 'metrics.json'), JSON.stringify(rows, null, 1));
console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exitCode = failures ? 1 : 0;

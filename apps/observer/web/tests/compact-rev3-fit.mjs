// ui-rev3-030: compact game screen (header scores + LIVE, B/S/O dots, fixed away/home player cells, centred zone with all
// candidates at once, square pitch-card strip). Route-mocked on tests/fixtures/service-849843.json + in-memory variants
// (L/R batter, overlapping targets, >10 pitches). No server; screenshots only to argv[2]. Headless Chromium, not Safari.
// Brought up to the current screen (2026-10-11; the checks had been stale since f407bcb and the script crashed since 5488c93):
// recommendations are listed under the zone and the batter silhouette marks the side; '다음 공 추천' is the list's label and a
// pinned live view says '최신으로 ↗' on the header button (no separate notice); the pitch cards are equal frameless tiles with a
// two-line caption (number + type, speed + unit); the smallest text is the 11px unit; a demo session never shows the red LIVE dot.
//   npx vite build --outDir <tmp> && PZ_WEB_DIST=<tmp> node tests/compact-rev3-fit.mjs [outDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const out = process.argv[2]; if (out) mkdirSync(out, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let failures = 0;
const ok = (cond, msg, info) => { if (!cond) failures++; console.log(cond ? 'ok  ' : 'FAIL', msg, cond ? '' : JSON.stringify(info)); };

function variant(which, side, many) {
  const b = JSON.parse(JSON.stringify(which === 'live' ? fx.live : fx.final));
  const setSide = (x) => { if (x?.batter) x.batter.side = side; };
  b.pitches.forEach(setSide); setSide(b.current);
  // live follows the current PA (last pitches); a finished game opens on the first pitch (first PA)
  const key = which === 'live' ? b.current?.pa_key ?? b.pitches[b.pitches.length - 1].pa_key : b.pitches[0].pa_key;
  if (many) (which === 'live' ? b.pitches.slice(-12) : b.pitches.slice(0, 12)).forEach((p, i) => { p.pa_key = key; p.pitch_number = i + 1; });
  // overlap: rank 3 shares rank 1's target (combined '1·3'), and the actual pitch lands on it
  for (const r of [b.next, ...b.pitches.map((p) => p.rec)]) if (r?.status === 'ready' && r.candidates.length >= 3 && r.candidates[0].target) {
    r.candidates[2].target = { ...r.candidates[0].target }; }
  for (const p of b.pitches) if (p.rec?.candidates?.[0]?.target && p.actual) { p.actual.x = p.rec.candidates[0].target.x; p.actual.z = p.rec.candidates[0].target.z; }
  return b;
}

const browser = await chromium.launch();
async function run(w, h, big, which, state, side, many) {
  let failApi = false; const body = variant(which, side, many);
  const c = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ko-KR' });
  const p = await c.newPage(); let polls = 0;
  await p.route('http://pz.test/**', (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === '/api/service/games/849843') {
      polls++;
      if (failApi) return route.fulfill({ status: 503, json: { detail: '기록 형식을 확인할 수 없습니다.' } });
      const r = JSON.parse(JSON.stringify(body)); const after = u.searchParams.get('after');
      if (after !== null) r.pitches = r.pitches.filter((x) => x.index > Number(after));
      return route.fulfill({ json: r });
    }
    if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this test' } });
    const f = path.join(dist, u.pathname); const t = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(t), contentType: TYPES[path.extname(t)] ?? 'application/octet-stream' });
  });
  await p.goto(`http://pz.test/watch/849843${which === 'live' ? '?demo=fixture' : ''}`);
  await p.waitForSelector('.sv-board'); await p.waitForTimeout(250);
  if (state === 'pinned' || state === 'stale') { await p.locator('.sv-card:not(.pending)').first().click(); await p.waitForTimeout(300); }
  if (state === 'stale') { failApi = true; await p.locator('.sv-link-btn, .sv-live').first().click().catch(() => {}); await p.waitForTimeout(800); }
  if (big) { await p.addStyleTag({ content: 'html{font-size:125% !important;-webkit-text-size-adjust:125%}' }); await p.waitForTimeout(250); }
  const pollsBefore = polls; if (state === 'pinned') await p.waitForTimeout(2600);
  const L = await p.evaluate(() => {
    const se = document.scrollingElement; const r = (s) => document.querySelector(s)?.getBoundingClientRect();
    const page = r('.sv-page'), box = r('.sv-zone-wrap .sv-zone-box'), strip = document.querySelector('.sv-strip');
    const cards = [...document.querySelectorAll('.sv-card')].map((e) => e.getBoundingClientRect());
    const vis = document.querySelector('.sv-page').cloneNode(true); vis.querySelectorAll('.sr-only').forEach((e) => e.remove()); document.body.append(vis); const text = vis.innerText; vis.remove();
    const offscreen = [...document.querySelectorAll('.sv-page button, .sv-page a')].filter((e) => { const b = e.getBoundingClientRect(); return b.width > 0 && (b.bottom > innerHeight + 1 || b.right > innerWidth + 1) && !e.closest('.sv-strip'); }).map((e) => e.className);
    const minFont = Math.min(...[...document.querySelectorAll('.sv-page *')].filter((e) => e.childNodes.length && [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()) && e.getBoundingClientRect().height > 0 && !e.closest('svg')).map((e) => parseFloat(getComputedStyle(e).fontSize)));
    const dots = r('.sv-card .sv-actual circle');
    return { scrollH: se.scrollHeight, innerH: innerHeight, scrollW: se.scrollWidth, innerW: innerWidth,
      zoneDx: box && page ? (box.left + box.width / 2) - (page.left + page.width / 2) : null, zoneH: r('.sv-zone-wrap .sv-zone')?.height ?? 0,
      recs: [...document.querySelectorAll('.sv-zone-wrap .sv-zone .sv-rec')].map((g) => g.getAttribute('data-rank')),
      actual: !!document.querySelector('.sv-zone-wrap .sv-zone .sv-actual'), prev: !!document.querySelector('.sv-prev'),
      cardsEqual: cards.every((c) => Math.abs(c.width - cards[0].width) < 1.5 && Math.abs(c.height - cards[0].height) < 1.5), cardsRow: new Set(cards.map((c) => Math.round(c.top))).size <= 1, n: cards.length,
      stripScroll: strip ? strip.scrollWidth > strip.clientWidth : false, dotR: dots ? dots.width / 2 : 0,
      caption: [...(document.querySelector('.sv-card:not(.pending) .sv-card-cap')?.children ?? [])].map((e) => e.textContent.trim()), badge: !!document.querySelector('.sv-card .sv-card-num'),
      banned: ['선택 비중', '방금 던진 공', '자동 ON', '새로고침'].filter((t) => text.includes(t)), toLatest: text.includes('최신으로'),
      live: document.querySelector('.sv-live')?.className ?? '', liveText: document.querySelector('.sv-live')?.textContent.trim() ?? '', liveDisabled: document.querySelector('.sv-live')?.disabled ?? null, notice: document.querySelector('.sv-notice')?.className ?? '',
      players: [...document.querySelectorAll('.sv-player')].map((e) => e.className + '|' + (e.querySelector('.sv-player-name')?.textContent ?? '')),
      dotsN: document.querySelectorAll('.sv-bso i').length, bat: document.querySelector('.sv-zrow .sv-batband.at-left') ? 'left' : document.querySelector('.sv-zrow .sv-batband.at-right') ? 'right' : 'none', listUnder: (() => { const z = r('.sv-zrow .sv-zone-box'), l = r('.sv-reclist'); return !!z && !!l && l.top >= z.bottom; })(),
      offscreen, minFont, overlapActualLabel: (() => { const t = [...document.querySelectorAll('.sv-zone-wrap .sv-rec text')].map((e) => e.getBoundingClientRect()); const a = document.querySelector('.sv-zone-wrap .sv-actual text')?.getBoundingClientRect(); return !!a && t.some((b) => !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom)); })() };
  });
  const tag = `${w}x${h}${big ? '@125' : ''} ${which}/${state} ${side}${many ? ' 12p' : ''}`;
  // with 125% text the page may scroll (zone row keeps a 15em minimum instead of shrinking; ui-pitcher-view-1006) — reported, not asserted
  const fits = !big;
  (fits ? ok : (c2, m, i) => console.log(c2 ? 'ok  ' : 'INFO', m, c2 ? '' : JSON.stringify(i)))(L.scrollH <= L.innerH + 1 && L.offscreen.length === 0, `${tag} one screen, controls on screen`, { scrollH: L.scrollH, innerH: L.innerH, off: L.offscreen });
  ok(L.scrollW <= L.innerW + 1, `${tag} no horizontal page scroll`, L);
  ok(L.zoneDx !== null && Math.abs(L.zoneDx) <= 2, `${tag} zone centred (dx ${L.zoneDx?.toFixed(2)})`, L.zoneDx);
  ok(L.zoneH >= 110, `${tag} zone height ${Math.round(L.zoneH)}`, L.zoneH);
  // pitcher's view (ui-pitcher-view-1006): a left-handed batter stands on the screen LEFT (and vice versa); the list is under the zone
  ok(L.bat === (side === 'L' ? 'left' : 'right') && L.listUnder, `${tag} batter silhouette on the ${side} side, list under the zone (${L.bat})`, [L.bat, L.listUnder]);
  ok(L.banned.length === 0, `${tag} removed visible texts`, L.banned);
  ok(L.toLatest === (state === 'pinned' && which === 'live'), `${tag} '최신으로' only on a pinned live view`, [L.toLatest, L.liveText]);
  ok(L.minFont >= 11, `${tag} min text ${L.minFont}px`, L.minFont);
  ok(L.cardsEqual && L.cardsRow, `${tag} ${L.n} equal cards in one row`, L);
  ok(!L.badge && L.caption.length === 2 && /^\d+\. \S+/.test(L.caption[0]) && !/km|mph/.test(L.caption[0]) && /^(\d+(\.\d)?|–) ?(km\/h|mph)$/.test(L.caption[1]), `${tag} card caption '${L.caption.join(' / ')}'`, L.caption);
  ok(L.dotR >= 5, `${tag} mini dot radius ${L.dotR?.toFixed(1)}px`, L.dotR);
  ok(L.dotsN === 7, `${tag} B3 S2 O2 dots`, L.dotsN);
  ok(L.players.length === 2 && L.players[0].includes('away') && L.players[1].includes('home'), `${tag} away left / home right`, L.players);
  ok(!L.prev, `${tag} no previous-pitch marker`, L);
  if (many) ok(L.stripScroll, `${tag} >10 pitches scroll horizontally in the strip`, L);
  if (state === 'latest' && which === 'live') ok(!L.actual && L.liveText === '시연 중' && !L.live.includes('on'), `${tag} latest: no actual marker, demo label without the LIVE dot`, L);
  if (state === 'pinned') {
    ok(L.recs.length >= 2 && L.recs.some((x) => x.includes('·')) && L.actual, `${tag} all ranks at once + combined label + actual (${L.recs.join(' ')})`, L.recs);
    ok(!L.overlapActualLabel, `${tag} actual label does not cover rank labels`, L);
    ok(which === 'live' ? /^최신으로/.test(L.liveText) && L.liveDisabled === false : L.liveText === '종료' && L.liveDisabled === true, `${tag} pinned: header button '${L.liveText}'`, [L.liveText, L.liveDisabled]);
    if (which === 'live') ok(polls === pollsBefore, `${tag} no fetch while pinned (auto off)`, { polls, pollsBefore });
  }
  if (state === 'stale') ok(L.notice.includes('warn') && L.liveDisabled === false, `${tag} network failure: warn notice, header button still usable`, L);
  if (which === 'final') ok(!L.live.includes('on'), `${tag} finished: LIVE gray`, L.live);
  if (out) await p.screenshot({ path: path.join(out, `${tag.replace(/[ /@]/g, '_')}.png`) });
  await c.close();
}
for (const [w, h] of [[375, 667], [375, 553], [390, 844], [320, 568]]) for (const big of [false, true]) for (const side of ['R', 'L']) {
  await run(w, h, big, 'live', 'latest', side, false);
  await run(w, h, big, 'live', 'pinned', side, side === 'L');
}
for (const [w, h] of [[375, 553], [320, 568]]) { await run(w, h, false, 'live', 'stale', 'R', false); await run(w, h, false, 'final', 'pinned', 'R', true); }
await browser.close();
console.log(failures ? `${failures} FAIL` : 'all ok');
process.exit(failures ? 1 : 0);

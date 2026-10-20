// ui-rev3-030: compact game screen (header scores + LIVE, B/S/O dots, fixed away/home player cells, centred zone with all
// candidates at once, square pitch-card strip). Route-mocked on tests/fixtures/service-849843.json + in-memory variants
// (L/R batter, overlapping targets, >10 pitches). No server; screenshots only to argv[2]. Headless Chromium, not Safari.
// Brought up to the current screen (2026-10-11; the checks had been stale since f407bcb and the script crashed since 5488c93):
// recommendations are listed under the zone and the batter silhouette marks the side; '다음 공 추천' is the list's label and a
// pinned live view says '최신으로 ↗' on the header button (no separate notice); the pitch cards are equal frameless tiles with a
// two-line caption (number + type, speed + unit); the smallest text is the 11px unit; a demo session never shows the red LIVE dot.
// Migrated to the approved 2026-10-13 redesign (0fe66af, UI-CONNECTION-2026-10-13.md; 2026-10-17): phones scroll vertically under a
// fixed bottom tab bar, so 'one screen' became 'score/count/zone above the fold, 390x844 strip above the tab bar (precedent:
// display-state-smoke fitsViewport, ws-responsive-smoke), every control reachable (not stranded under the tab bar, no sideways
// overflow)'; pitcher's view puts a right-handed batter on the RIGHT; the card caption's second line may end in ' · <result>'; the
// strip draws no mini zone (the current PA's last 3 pitches are numbered dots on the main zone instead); on a LIVE game a tapped past
// card opens the detail sheet only (main view, '시연 중' header and polling continue; the past pitch's ranks/actual are checked in the
// sheet), on a finished game it also moves the main view. The sheet's '정책 선택 비중' table header is an approved explanation, so the
// removed-text check reads the main screen only.
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
const numberCases = new Set();
const ok = (cond, msg, info) => { if (!cond) failures++; console.log(cond ? 'ok  ' : 'FAIL', msg, cond ? '' : JSON.stringify(info)); };

function variant(which, side, many, numberCase) {
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
  // Separate number fixtures preserve the original many-pitch / recommendation-overlap coverage above.
  if (numberCase) {
    b.pitches.slice(-3).forEach((p, i) => { p.pa_key = key; p.pitch_number = i + 1; p.actual.x = i === 2 ? 1.6 : .3; p.actual.z = 2.4; });
    b.next.rec.candidates[0].target = { x: .3, z: numberCase === 'dense' ? 2.4 : 2.55 };
    if (numberCase === 'dense') { b.next.rec.candidates[1].target = { x: .55, z: 2.7 }; b.next.rec.candidates[2].target = { x: .05, z: 2.1 }; }
  }
  return b;
}

const browser = await chromium.launch();
async function run(w, h, big, which, state, side, many, numberCase = null) {
  let failApi = false; const body = variant(which, side, many, numberCase);
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
  // stale: the card opened the (modal, on phones) detail sheet; close it so the header button is reachable, then fail the next fetch
  if (state === 'stale') { await p.keyboard.press('Escape'); await p.waitForTimeout(200); failApi = true;
    await p.locator('.sv-live').click({ timeout: 3000 }).catch(() => {}); await p.waitForTimeout(800); }
  if (big) { await p.addStyleTag({ content: 'html{font-size:125% !important;-webkit-text-size-adjust:125%}' }); await p.waitForTimeout(250); }
  const pollsBefore = polls; if (state === 'pinned') await p.waitForTimeout(2600);
  const L = await p.evaluate(() => {
    const se = document.scrollingElement; const r = (s) => document.querySelector(s)?.getBoundingClientRect();
    const page = r('.sv-page'), box = r('.sv-zone-wrap .sv-zone-box'), strip = document.querySelector('.sv-strip');
    const cards = [...document.querySelectorAll('.sv-card')].map((e) => e.getBoundingClientRect());
    // main screen text only: the detail sheet (.sv-backdrop) is an on-demand explanation and may name '정책 선택 비중'
    const vis = document.querySelector('.sv-page').cloneNode(true); vis.querySelectorAll('.sr-only, .sv-backdrop').forEach((e) => e.remove()); document.body.append(vis); const text = vis.innerText; vis.remove();
    // the fold: the fixed bottom tab bar's top on phones (it covers the viewport's bottom), else the viewport bottom
    const nav = document.querySelector('.sv-bottom-nav'); const navFixed = !!nav && getComputedStyle(nav).position === 'fixed' && nav.getBoundingClientRect().height > 0;
    const fold = navFixed ? nav.getBoundingClientRect().top : innerHeight; const navH = innerHeight - fold;
    const absBottom = (s) => { const b = r(s); return b && b.height > 0 ? b.bottom + scrollY : null; };
    const above = { score: absBottom('.sv-head-score'), bso: absBottom('.sv-bso'), zone: absBottom('.sv-zone-wrap .sv-zone-box'), strip: absBottom('.sv-strip'), fold };
    // every page control can be brought into view: not wider than the viewport, and (at maximum scroll) not left under the fixed tab bar
    const offscreen = [...document.querySelectorAll('.sv-page button, .sv-page a')].filter((e) => { const b = e.getBoundingClientRect();
      if (!(b.width > 0) || e.closest('.sv-strip, .sv-backdrop, .sv-bottom-nav')) return false;
      return b.right > innerWidth + 1 || b.left < -1 || b.bottom + scrollY > se.scrollHeight - navH + 1; }).map((e) => e.className + '|' + e.textContent.trim().slice(0, 12));
    // controls inside an open sheet: the sheet sits in the viewport, its body scrolls when its content is taller, nothing overflows sideways
    const sh = document.querySelector('.sv-sheet'), sb = sh?.querySelector('.sv-sheet-body');
    const sheetOff = !sh ? [] : (() => { const s = sh.getBoundingClientRect(); const bad = [];
      if (s.top < -1 || s.bottom > innerHeight + 1 || s.right > innerWidth + 1 || s.left < -1) bad.push('sheet-outside-viewport');
      if (sb && sb.scrollHeight > sb.clientHeight + 1 && !/auto|scroll/.test(getComputedStyle(sb).overflowY)) bad.push('sheet-body-not-scrollable');
      for (const e of sh.querySelectorAll('button, a')) { const b = e.getBoundingClientRect(); if (b.width > 0 && (b.right > s.right + 1 || b.left < s.left - 1)) bad.push(e.className + '|' + e.textContent.trim().slice(0, 12)); }
      return bad; })();
    const paDots = [...document.querySelectorAll('.sv-zone-wrap .sv-pa-dot')];
    const capNums = [...document.querySelectorAll('.sv-strip .sv-card:not(.pending) .sv-card-cap > span:first-child')].map((e) => parseInt(e.textContent, 10)).filter(Number.isFinite);
    // a caption line wider than its card is cut off (the second line now carries the result, e.g. '155 km/h · 타구 아웃')
    const capClip = [...document.querySelectorAll('.sv-strip .sv-card:not(.pending)')].flatMap((card) => { const cb = card.getBoundingClientRect();
      // the glyphs themselves (a Range), against the card's inner box
      const cs = getComputedStyle(card), inL = cb.left + parseFloat(cs.borderLeftWidth), inR = cb.right - parseFloat(cs.borderRightWidth);
      // a line may be shortened with a visible ellipsis inside its own card (full words in the card's aria-label and detail sheet,
      // 2026-10-17); what must never happen is text painted past the card edge (under the next card) or cut with no mark
      return [...card.querySelectorAll('.sv-card-cap > span')].map((s) => { const rg = document.createRange(); rg.selectNodeContents(s); const g = rg.getBoundingClientRect();
        const box = s.getBoundingClientRect(), st = getComputedStyle(s); const clipped = st.overflow === 'hidden' && st.textOverflow === 'ellipsis';
        return [s, clipped ? box : g, clipped ? null : g]; })
        .filter(([, b]) => b.width > 0 && (b.left < inL - 0.5 || b.right > inR + 0.5))
        .map(([s, b]) => `${s.textContent.trim()} (text ${Math.round(b.width)}px, card inner ${Math.round(inR - inL)}px)`); });
    const capAria = [...document.querySelectorAll('.sv-strip .sv-card:not(.pending):not(.unconfirmed):not(.folded)')].filter((card) => { const last = card.querySelector('.sv-card-cap > span:last-child');
      const result = last?.textContent.split(' · ').slice(1).join(' · ').trim(); return result && !(card.getAttribute('aria-label') ?? '').includes(result); }).map((c) => c.getAttribute('aria-label'));
    const sel = document.querySelector('.sv-strip .sv-card[aria-pressed="true"]:not(.pending) .sv-card-cap > span:first-child')?.textContent.trim() ?? '';
    const minFont = Math.min(...[...document.querySelectorAll('.sv-page *')].filter((e) => e.childNodes.length && [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()) && e.getBoundingClientRect().height > 0 && !e.closest('svg')).map((e) => parseFloat(getComputedStyle(e).fontSize)));
    const dots = r('.sv-card .sv-actual circle');
    const marks = [...document.querySelectorAll('.sv-zone .sv-actual,.sv-zone .sv-rec,.sv-zone .sv-mitt')].map(g => {
      const e = g.querySelector('circle,rect,line');
      const x = e.tagName === 'circle' ? Number(e.getAttribute('cx')) : e.tagName === 'rect' ? Number(e.getAttribute('x')) + Number(e.getAttribute('width')) / 2 : Number(e.getAttribute('x1'));
      return { kind: g.getAttribute('class'), rawX: Number(g.dataset.x), x };
    });
    const lum = color => color.match(/[\d.]+/g).slice(0, 3).map(Number).map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4).reduce((a, v, i) => a + v * [.2126, .7152, .0722][i], 0);
    const numberContrast = paDots.map(g => {
      const text = g.querySelector('text'), merged = text.textContent.includes('·');
      const css = getComputedStyle(text);
      // Clear floating numbers use the darker gray background as a bound; dense fallback numbers use their white halo.
      const bg = merged && css.stroke !== 'none' ? css.stroke : getComputedStyle(merged ? g.closest('svg').querySelector('.sv-zone-outside') : g.querySelector('circle')).fill;
      const fg = css.fill, a = lum(fg), b = lum(bg);
      return { kind: merged ? 'merged' : g.classList.contains('out') ? 'outside' : 'inside', fg, bg, ratio: (Math.max(a, b) + .05) / (Math.min(a, b) + .05) };
    });
    const mergedNumbers = paDots.flatMap(g => {
      const text = g.querySelector('text'); if (!text.textContent.includes('·')) return [];
      const svg = g.closest('svg'), b = text.getBoundingClientRect(), v = svg.getBoundingClientRect(), css = getComputedStyle(text);
      const hit = e => { const a = e.getBoundingClientRect(); return b.left < a.right && a.left < b.right && b.top < a.bottom && a.top < b.bottom; };
      return [{ halo: css.stroke, haloScreenPx: parseFloat(css.strokeWidth) * text.getScreenCTM().a, paintOrder: css.paintOrder,
        visible: b.left >= v.left && b.right <= v.right && b.top >= v.top && b.bottom <= v.bottom,
        markOverlap: [...svg.querySelectorAll('.sv-rec rect')].some(hit),
        contentOverlap: [...svg.querySelectorAll('.sv-rec text,.sv-mitt text,.sv-pa-dot circle,.sv-pa-dot text')].filter(e => e !== text).some(hit) }];
    });
    const paCoordinates = paDots.filter(g => !g.classList.contains('out')).map(g => ({ rawX: Number(g.dataset.x), rawZ: Number(g.dataset.z), x: Number(g.querySelector('circle').getAttribute('cx')), y: Number(g.querySelector('circle').getAttribute('cy')) }));
    return { scrollH: se.scrollHeight, innerH: innerHeight, scrollW: se.scrollWidth, innerW: innerWidth,
      marks, numberContrast, mergedNumbers, paCoordinates,
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
      offscreen, minFont, inDrawingLabel: !!document.querySelector('.sv-zone-wrap .sv-actual:not(.sv-pa-dot) text'), named: (document.querySelector('.sv-actual-caption b')?.textContent ?? '').trim(),
      above, sheetOff, capClip, capAria, mainSingleActual: !!document.querySelector('.sv-zone-wrap .sv-zone .sv-actual:not(.sv-pa-dot)'),
      miniHidden: [...document.querySelectorAll('.sv-strip .sv-card .sv-zone')].every((e) => getComputedStyle(e).display === 'none'),
      paNums: paDots.flatMap((g) => (g.querySelector('text')?.textContent ?? '').split('·').map(Number)).sort((a, b) => a - b), paWant: capNums.slice(-3).sort((a, b) => a - b),
      paDotR: Math.min(...paDots.map((g) => (g.querySelector('circle')?.getBoundingClientRect().width ?? 0) / 2)),
      sel, sheetTitle: sh?.querySelector('.sv-sheet-head h2')?.textContent.trim() ?? null, sheetModal: sh?.getAttribute('aria-modal') ?? null,
      sheetRecs: [...(sh?.querySelectorAll('.sv-detail-zone .sv-rec') ?? [])].map((g) => g.getAttribute('data-rank')), sheetActual: !!sh?.querySelector('.sv-detail-zone .sv-actual'),
      sheetInLabel: !!sh?.querySelector('.sv-detail-zone .sv-actual text'), sheetNamed: (sh?.querySelector('.sv-dl:not(.sv-pair) dd')?.textContent ?? '').trim() };
  });
  const tag = `${w}x${h}${big ? '@125' : ''} ${which}/${state} ${side}${many ? ' 12p' : ''}${numberCase ? ` numbers-${numberCase}` : ''}`;
  // with 125% text the layout is reported, not asserted (zone row keeps a 15em minimum instead of shrinking; ui-pitcher-view-1006)
  const fits = !big; const soft = fits ? ok : (c2, m, i) => console.log(c2 ? 'ok  ' : 'INFO', m, c2 ? '' : JSON.stringify(i));
  // was 'one screen, controls on screen' (scrollH <= innerH and no control below the fold). The approved phone layout scrolls
  // vertically under a fixed tab bar, so: the key state (score, count, zone) is above the tab bar without scrolling, the pitch
  // strip too at 390x844 (display-state-smoke / ws-responsive-smoke precedent; shorter screens may scroll to it), and every control
  // can be scrolled into view (none stranded under the tab bar or off to the side; sheet controls reachable inside the sheet).
  const A = L.above, keyAbove = A.score !== null && A.score <= A.fold + 1 && A.bso !== null && A.bso <= A.fold + 1 && A.zone !== null && A.zone <= A.fold + 1;
  soft(keyAbove && (w !== 390 || h !== 844 || (A.strip !== null && A.strip <= A.fold + 1)), `${tag} score, count and zone above the fold${w === 390 && h === 844 ? ', strip above the tab bar' : ''}`, A);
  soft(L.offscreen.length === 0 && L.sheetOff.length === 0, `${tag} every control reachable (page scroll / sheet scroll, none under the tab bar)`, { scrollH: L.scrollH, innerH: L.innerH, off: L.offscreen, sheet: L.sheetOff });
  ok(L.scrollW <= L.innerW + 1, `${tag} no horizontal page scroll`, L);
  ok(L.zoneDx !== null && Math.abs(L.zoneDx) <= 2, `${tag} zone centred (dx ${L.zoneDx?.toFixed(2)})`, L.zoneDx);
  ok(L.zoneH >= 110, `${tag} zone height ${Math.round(L.zoneH)}`, L.zoneH);
  // Pitcher's view: a right-handed batter stands on the screen RIGHT (and vice versa); the list is under the zone
  ok(L.bat === (side === 'R' ? 'right' : 'left') && L.listUnder, `${tag} pitcher's view: ${side} batter silhouette on the ${side === 'R' ? 'right' : 'left'}, list under the zone (${L.bat})`, [L.bat, L.listUnder]);
  ok(L.marks.length > 0 && L.marks.every(m => Math.sign(m.x) === -Math.sign(m.rawX)), `${tag} main/detail and hidden mini SVG marks mirror source x`, L.marks);
  if (which === 'live') {
    L.numberContrast.forEach(c => numberCases.add(c.kind));
    ok(L.numberContrast.length > 0 && L.numberContrast.every(c => c.ratio >= 4.5), `${tag} pitch numbers have at least 4.5:1 contrast`, L.numberContrast);
  }
  if (numberCase) {
    ok(L.paCoordinates.length > 0 && L.paCoordinates.every(p => p.x === -p.rawX * 100 && p.y === -p.rawZ * 100), `${tag} number layout never moves the measured dots`, L.paCoordinates);
    ok(L.mergedNumbers.length === 1 && L.mergedNumbers.every(n => n.visible && !n.contentOverlap), `${tag} merged numbers stay in the SVG and clear other text and dots`, L.mergedNumbers);
    ok(L.mergedNumbers.every(n => numberCase === 'dense' ? n.halo === 'rgb(255, 255, 255)' && n.haloScreenPx >= 2.99 && n.haloScreenPx <= 3.01 && n.paintOrder === 'stroke' : n.halo === 'none' && !n.markOverlap),
      `${tag} ${numberCase === 'dense' ? 'dense fallback uses a 3 screen-pixel white halo' : 'merged numbers avoid recommendation squares'}`, L.mergedNumbers);
  }
  ok(L.banned.length === 0, `${tag} removed visible texts (main screen)`, L.banned);
  // a tapped past card never leaves the latest view on a live game any more, so '최신으로' must not appear in any of these states
  ok(!L.toLatest, `${tag} no '최신으로' while the main view follows the latest / on a finished game`, [L.toLatest, L.liveText]);
  ok(L.minFont >= 11, `${tag} min text ${L.minFont}px`, L.minFont);
  ok(L.cardsEqual && L.cardsRow, `${tag} ${L.n} equal cards in one row`, L);
  ok(!L.badge && L.caption.length === 2 && /^\d+\. \S+/.test(L.caption[0]) && !/km|mph/.test(L.caption[0]) && /^(\d+(\.\d)?|–) ?(km\/h|mph)( · \S.*)?$/.test(L.caption[1]), `${tag} card caption '${L.caption.join(' / ')}'`, L.caption);
  ok(L.capClip.length === 0, `${tag} every card caption line stays inside its card (a long line ends in an ellipsis there, never under the next card)`, L.capClip);
  ok(L.capAria.length === 0, `${tag} a card's full result stays in its aria-label`, L.capAria);
  // was 'mini dot radius >= 5px' on the card's mini zone. The strip no longer draws a mini zone; where the pitch went is shown by the
  // main zone's numbered dots (the current PA's last 3 pitches), which must stay readable (r >= 5px) and carry the right numbers.
  ok(L.miniHidden, `${tag} strip cards draw no mini zone`, L.miniHidden);
  if (which === 'live') {
    ok(L.paNums.length > 0 && JSON.stringify(L.paNums) === JSON.stringify(L.paWant) && L.paDotR >= 5, `${tag} main zone numbers the PA's last pitches (${L.paNums.join(',')}), dot radius ${L.paDotR?.toFixed(1)}px`, { got: L.paNums, want: L.paWant, r: L.paDotR });
  }
  ok(L.dotsN === 7, `${tag} B3 S2 O2 dots`, L.dotsN);
  ok(L.players.length === 2 && L.players[0].includes('away') && L.players[1].includes('home'), `${tag} away left / home right`, L.players);
  ok(!L.prev, `${tag} no previous-pitch marker`, L);
  if (many) ok(L.stripScroll, `${tag} >10 pitches scroll horizontally in the strip`, L);
  // latest: no single 'actual' marker for a thrown pitch (only the PA's numbered dots), demo label without the LIVE dot
  if (state === 'latest' && which === 'live') ok(!L.mainSingleActual && !L.inDrawingLabel && L.liveText === '시연 중' && !L.live.includes('on'), `${tag} latest: no single actual marker, demo label without the LIVE dot`, { single: L.mainSingleActual, liveText: L.liveText, live: L.live });
  if (state === 'pinned') {
    // the tapped pitch opens its detail sheet (all games): all ranks at once with the combined label and the actual, no label on the
    // actual inside the drawing, the pitch named in words in the sheet
    ok(L.sheetTitle !== null && L.sel !== '' && L.sheetTitle.includes(`${parseInt(L.sel, 10)}구`), `${tag} tapped card '${L.sel}' opens its detail sheet ('${L.sheetTitle}')`, [L.sel, L.sheetTitle]);
    ok(L.sheetRecs.length >= 2 && L.sheetRecs.some((x) => x.includes('·')) && L.sheetActual, `${tag} sheet: all ranks at once + combined label + actual (${L.sheetRecs.join(' ')})`, [L.sheetRecs, L.sheetActual]);
    ok(!L.sheetInLabel && L.sheetNamed.length > 0 && L.sheetNamed.startsWith(L.sel.replace(/^\d+\.\s*/, '')), `${tag} sheet: no label on the actual inside the drawing, pitch named in words ('${L.sheetNamed}')`, [L.sheetInLabel, L.sheetNamed, L.sel]);
    if (which === 'live') {
      // live: the main stage stays on the latest (no single actual, the PA's numbered dots stay), the header keeps '시연 중', polling continues
      ok(!L.mainSingleActual && !L.inDrawingLabel, `${tag} live: main stage stays on the latest pitch while the sheet is open`, { single: L.mainSingleActual, label: L.inDrawingLabel });
      ok(L.liveText === '시연 중' && L.liveDisabled === false && !L.live.includes('on'), `${tag} live: header button stays '${L.liveText}' (enabled, demo has no LIVE dot)`, [L.liveText, L.liveDisabled, L.live]);
      ok(polls > pollsBefore, `${tag} live: polling continues while the detail sheet is open`, { polls, pollsBefore });
    } else {
      // finished game: the selection also moves the main view (all ranks + combined label + actual, pitch named under the zone)
      ok(L.recs.length >= 2 && L.recs.some((x) => x.includes('·')) && L.actual, `${tag} all ranks at once + combined label + actual (${L.recs.join(' ')})`, L.recs);
      ok(!L.inDrawingLabel && L.named.length > 0, `${tag} no label on the actual pitch inside the drawing, pitch named under the zone ('${L.named}')`, [L.inDrawingLabel, L.named]);
      ok(L.liveText === '종료' && L.liveDisabled === true, `${tag} pinned: header button '${L.liveText}'`, [L.liveText, L.liveDisabled]);
    }
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
for (const numberCase of ['clear', 'dense']) await run(390, 844, false, 'live', 'latest', 'R', false, numberCase);
ok(['inside', 'outside', 'merged'].every(kind => numberCases.has(kind)), 'number contrast covers inside, hollow outside and merged labels', [...numberCases]);
await browser.close();
console.log(failures ? `${failures} FAIL` : 'all ok');
process.exit(failures ? 1 : 0);

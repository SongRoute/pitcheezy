// Display check (rev3-label-polish-040; cases from final-review-039, author rerun, not independent): for each case the actual
// marker, its arrow, every rank square+number, the mitt label and any in-drawing label must lie fully inside the drawing (main),
// and the texts must not overlap one another. Since f407bcb the main drawing carries NO label for the actual pitch (its name is
// in the summary under the zone), so the check is now: no label inside the drawing, the pitch named in the summary.
// UI 10/26: the strip and its mini cards are gone; the same never-cut rule now applies to the pitch sheet's small zone.
//   PZ_WEB_DIST=<dist> node tests/zone-label-clip-check.mjs [outDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8'));
const out = process.argv[2]; if (out) mkdirSync(out, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let fails = 0, oks = 0; const ok = (c, m, i) => { if (c) oks++; else { fails++; console.log('FAIL', m, JSON.stringify(i)); } };
const BOUNDS = { normal: { bottom: 1.6, top: 3.39 }, tall: { bottom: 1.2, top: 4.05 }, short: { bottom: 1.75, top: 3.1 } };
const CASES = [['tall', [0, .9], null], ['tall', [0, .4], null], ['normal', [-1.5, 4.6], -1.0], ['normal', [0, 3.8], null], ['short', [0, 3.8], null],
  ['normal', [1.6, 2.5], 1.4], ['normal', [.3, 2.4], .3], ['tall', [-1.5, 4.6], 1.2], ['normal', [1.25, 2.5], null], ['normal', [.86, 1.58], -1.0]];
const b = await chromium.launch();
for (const [vw, vh] of [[375, 553], [320, 568], [375, 667], [390, 844]]) for (const [bn, [x, z], mitt] of CASES) {
  const c = await b.newContext({ viewport: { width: vw, height: vh }, deviceScaleFactor: 2, isMobile: true }); const p = await c.newPage();
  const body = JSON.parse(JSON.stringify(fx.final)); body.zone_bounds = BOUNDS[bn];
  const pt = body.pitches[0]; pt.actual.x = x; pt.actual.z = z; pt.actual.catcher_setup = mitt === null ? null : { status: 'estimated', x_band: 'x', plate_x_feet: mitt };
  pt.rec.candidates.forEach((cd, i) => { cd.target = i === 0 ? { x, z } : i === 1 ? { x: -x, z } : { x: .2, z: 2.6 }; });
  await p.route('http://pz.test/**', (route) => { const u = new URL(route.request().url());
    if (u.pathname === '/api/service/games/849843') return route.fulfill({ json: body });
    if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'x' } });
    const f = path.join(dist, u.pathname); const t = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(t), contentType: TYPES[path.extname(t)] ?? 'application/octet-stream' }); });
  await p.goto('http://pz.test/watch/849843'); await p.waitForSelector('svg.sv-zone:not(.mini) .sv-zone-box'); await p.waitForTimeout(200);
  const m = await p.evaluate(() => {
    const R = (e) => { const r = e.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; };
    const inBox = (a, o, tol = .5) => a.x >= o.x - tol && a.y >= o.y - tol && a.x + a.w <= o.x + o.w + tol && a.y + a.h <= o.y + o.h + tol;
    const hit = (a, b) => a.x < b.x + b.w - .5 && b.x < a.x + a.w - .5 && a.y < b.y + b.h - .5 && b.y < a.y + a.h - .5;
    const main = document.querySelector('svg.sv-zone:not(.mini)'); const sv = R(main);
    const parts = [...main.querySelectorAll('.sv-actual circle, .sv-actual path, .sv-actual text, .sv-rec rect, .sv-rec text, .sv-rec path, .sv-mitt text, .sv-zone-caption')]
      .map(e => ({ k: e.closest('g')?.getAttribute('class') + ' ' + e.tagName + (e.classList.contains('sv-zone-caption') ? ' caption' : ''), r: R(e) }));
    const clipped = parts.filter(q => !inBox(q.r, sv)).map(q => q.k);
    const texts = parts.filter(q => q.k.includes('text') || q.k.includes('caption'));
    const overlaps = []; for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) if (hit(texts[i].r, texts[j].r)) overlaps.push([texts[i].k, texts[j].k]);
    // the G1 number inside the mark is part of the mark; what must not be in the drawing is a NAME label for the pitch
    const labelShown = !!main.querySelector('.sv-actual .sv-actual-label'); const caption = !!main.querySelector('.sv-zone-caption');
    return { clipped, overlaps, labelShown, caption, named: (document.querySelector('.sv-actual-line .sv-actual-name')?.textContent ?? '').trim(), out: main.querySelector('.sv-actual')?.classList.contains('out') };
  });
  // UI 10/26: the pitch strip (and its mini zones) is gone; the pitch's own sheet draws it in a small zone instead. There the mark, its
  // ring, any edge arrow and every square must be painted inside the drawing (the window widens to the pitch, never clips it).
  await p.locator('.sv-rec-sec .sv-state-more').click(); await p.locator('.sv-detail-zone svg.sv-zone').waitFor(); await p.waitForTimeout(450);
  const d = await p.evaluate(() => { const svg = document.querySelector('.sv-detail-zone svg.sv-zone'), s = svg.getBoundingClientRect();
    const parts = [...svg.querySelectorAll('.sv-actual circle, .sv-actual path, .sv-actual text, .sv-rec rect, .sv-rec text')].map(e => { const r = e.getBoundingClientRect(); return { k: e.closest('g').getAttribute('class') + ' ' + e.tagName, l: r.left, t: r.top, r: r.right, b: r.bottom }; });
    return { n: parts.length, cut: parts.filter(q => q.l < s.left - .5 || q.t < s.top - .5 || q.r > s.right + .5 || q.b > s.bottom + .5).map(q => q.k) }; });
  m.miniN = d.n; m.miniClipped = d.cut.length; m.sheetCut = d.cut;
  const tag = `${vw}x${vh} ${bn} (${x},${z}) mitt ${mitt}`;
  ok(!m.clipped.length, `${tag}: main marks/labels inside drawing`, m.clipped);
  ok(!m.overlaps.length, `${tag}: no text overlaps`, m.overlaps);
  ok(!m.labelShown && m.named.length > 0, `${tag}: no label inside the drawing, pitch named in the summary`, m);
  ok(m.miniN >= 1 && !m.miniClipped, `${tag}: pitch sheet zone: mark, ring, arrow and squares all inside the drawing`, m);
  if (out) await p.screenshot({ path: path.join(out, `${bn}_${vw}x${vh}_${x}_${z}.png`) });
  await c.close();
}
await b.close(); console.log(`${oks} ok, ${fails} FAIL`); process.exitCode = fails ? 1 : 0;

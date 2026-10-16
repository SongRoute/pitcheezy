// Zone geometry regression (integration rev3, after geometry-review-037; catcher's view since ws-ui-connect-1013). Route-mocked fixture, headless Chromium.
// For every probe point: if it lies inside the visible window it must be drawn at its TRUE coordinate (recovered from the
// drawn zone box within 0.03 ft); if it lies beyond the window it must be an `.out` edge marker drawn OUTSIDE the zone box.
// A pitch outside the strike zone must never render inside the zone box. Checked for the main zone and the opened full detail zone,
// normal bounds 1.6–3.39 and tall bounds 1.2–4.05, at 375x667 and 320x568. Also: all 3 targets + actual visible when overlapping.
//   PZ_WEB_DIST=<built dist> node tests/zone-geometry-check.mjs [outDir]
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
const PROBES = [[1.0, 2.5], [1.25, 2.5], [-1.05, 2.5], [0, 1.3], [0, 1.45], [0, .9], [0, 3.8], [.5, 2.0], [-.6, 3.0], [1.6, 1.0], [0.9, 3.5]];
const BOUNDS = { normal: { bottom: 1.6, top: 3.39 }, tall: { bottom: 1.2, top: 4.05 } };
const b = await chromium.launch();
async function shot(w, h, bounds, [x, z], targets) {
  const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true }); const p = await c.newPage();
  const body = JSON.parse(JSON.stringify(fx.final)); body.zone_bounds = bounds;
  body.pitches[0].actual.x = x; body.pitches[0].actual.z = z;
  if (targets) body.pitches[0].rec.candidates.forEach((cd, i) => { cd.target = targets[i]; });
  await p.route('**/*', (route) => { const u = new URL(route.request().url());
    if (u.origin !== 'http://pz.test') return route.abort('blockedbyclient');
    if (u.pathname === '/api/service/games/849843') return route.fulfill({ json: body });
    if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'x' } });
    const f = path.join(dist, u.pathname); const t = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(t), contentType: TYPES[path.extname(t)] ?? 'application/octet-stream' }); });
  await p.goto('http://pz.test/watch/849843'); await p.waitForSelector('svg.sv-zone:not(.mini) .sv-zone-box'); await p.waitForTimeout(150);
  const measure = async () => p.evaluate((B) => { const one = (svg) => { const box = svg.querySelector('.sv-zone-box').getBoundingClientRect(); const a = svg.querySelector('.sv-actual');
      if (!a) return null; const cr = a.querySelector('circle').getBoundingClientRect(); const cx = cr.left + cr.width / 2, cy = cr.top + cr.height / 2;
      // Catcher's view (2026-10-13): screen right = positive plate_x; z unchanged
      const fx = (cx - (box.left + box.width / 2)) / (box.width / 1.66), fz = B.bottom + (box.bottom - cy) / (box.height / (B.top - B.bottom));
      const inBox = cx >= box.left && cx <= box.right && cy >= box.top && cy <= box.bottom;
      return { fx, fz, inBox, out: a.classList.contains('out'), dataX: Number(a.dataset.x), dataZ: Number(a.dataset.z), screenSide: Math.sign(cx - (box.left + box.width / 2)), boxW: box.width, svgW: svg.getBoundingClientRect().width,
        recs: [...svg.querySelectorAll('.sv-rec')].map(g => { const r = g.querySelector('rect').getBoundingClientRect(); const t = g.querySelector('text').getBoundingClientRect();
          const covered = document.elementFromPoint(t.left + t.width / 2, t.top + t.height / 2); return { label: g.dataset.rank, visible: !!covered && g.contains(covered) }; }) }; };
    // "Visible" is probed with elementFromPoint, which is a HIT test: it skips anything with pointer-events:none. Since afe893f the
    // zone row and its drawing are pointer-events:none on purpose (they must not swallow taps on the summary under their lower edge),
    // so the probe saw through every label and reported them hidden although the pixels were unchanged. Record that intent, then
    // make the drawing hit-testable for this measurement only, so the probe again answers "is something painted over the label".
    const tapThrough = getComputedStyle(document.querySelector('svg.sv-zone:not(.mini)')).pointerEvents === 'none' && getComputedStyle(document.querySelector('.sv-zrow')).pointerEvents === 'none';
    const probe = document.createElement('style'); probe.textContent = '.sv-zrow,.sv-zrow .sv-zone{pointer-events:auto!important}'; document.head.append(probe);
    const main = document.querySelector('svg.sv-zone:not(.mini)'); const mini = document.querySelector('svg.sv-zone.mini');
    return { tapThrough, main: one(main), mini: null, centre: main.querySelector('.sv-zone-box').getBoundingClientRect().left + main.querySelector('.sv-zone-box').getBoundingClientRect().width / 2 - innerWidth / 2 }; }, bounds);
  const m = await measure();
  await p.locator('.sv-strip .sv-card').first().click();
  await p.locator('.sv-detail-zone .sv-zone').waitFor();
  const detail = await p.evaluate(B => { const svg=document.querySelector('.sv-detail-zone .sv-zone'), box=svg.querySelector('.sv-zone-box').getBoundingClientRect(), a=svg.querySelector('.sv-actual'), cr=a.querySelector('circle').getBoundingClientRect(), cx=cr.left+cr.width/2, cy=cr.top+cr.height/2; return {fx:(cx-(box.left+box.width/2))/(box.width/1.66),fz:B.bottom+(box.bottom-cy)/(box.height/(B.top-B.bottom)),out:a.classList.contains('out'),dataX:Number(a.dataset.x),dataZ:Number(a.dataset.z),screenSide:Math.sign(cx-(box.left+box.width/2)),inBox:cx>=box.left&&cx<=box.right&&cy>=box.top&&cy<=box.bottom,boxW:box.width}; },bounds);
  m.detail=detail;
  await p.getByRole('button',{name:'닫기',exact:true}).click();
  return { p, c, m };
}
const rows = [];
for (const [w, h] of [[375, 667], [320, 568]]) for (const [bn, B] of Object.entries(BOUNDS)) for (const pt of PROBES) {
  const { c, m } = await shot(w, h, B, pt);
  const strikeX = Math.abs(pt[0]) <= .83, strikeZ = pt[1] >= B.bottom && pt[1] <= B.top;
  for (const [view, mg] of [['main', { x: .12, z: .24 }], ['detail', { x: .42, z: .45 }]]) {
    const r = m[view]; if (!r) { ok(false, `${view} missing`, pt); continue; }
    const inWin = Math.abs(pt[0]) <= .83 + mg.x && pt[1] >= B.bottom - mg.z && pt[1] <= B.top + mg.z;
    const tag = `${w}x${h} ${bn} ${view} (${pt})`;
    if (inWin) ok(!r.out && Math.abs(r.fx - pt[0]) < .03 && Math.abs(r.fz - pt[1]) < .03, `${tag}: true coordinate`, r);
    else ok(r.out, `${tag}: beyond window -> edge marker`, r);
    ok(r.dataX === pt[0] && r.dataZ === pt[1], `${tag}: stored coordinate unchanged in data-x/z`, r);
    if (Math.abs(pt[0]) > .05) ok(r.screenSide === Math.sign(pt[0]), `${tag}: catcher view (plate_x ${pt[0]} drawn on the ${pt[0] > 0 ? 'right' : 'left'})`, r);
    if (!(strikeX && strikeZ)) ok(!r.inBox, `${tag}: ball not drawn inside zone`, r);
    rows.push({ tag, ...r, recs: undefined });
  }
  ok(Math.abs(m.centre) <= 1, `${w}x${h} ${bn} (${pt}) main zone centred`, m.centre);
  await c.close();
}
// overlap: actual exactly on rank 1, rank 3 on rank 1 too, rank 2 elsewhere
for (const [w, h] of [[375, 667], [320, 568]]) {
  const { p, c, m } = await shot(w, h, BOUNDS.normal, [.3, 2.4], [{ x: .3, z: 2.4 }, { x: -.4, z: 2.0 }, { x: .3, z: 2.4 }]);
  ok(m.main.recs.length === 2 && m.main.recs.every(r => r.visible) && m.main.recs.some(r => r.label === '1·3'), `${w}x${h} overlap: merged '1·3' + '2' labels visible over actual`, m.main.recs);
  ok(!m.main.out && Math.abs(m.main.fx - .3) < .03, `${w}x${h} overlap: actual still at its true spot`, m.main);
  ok(m.tapThrough, `${w}x${h} overlap: zone row and drawing still let taps through (afe893f)`, m.tapThrough);
  if (out) await p.screenshot({ path: path.join(out, `overlap-${w}x${h}.png`) });
  await c.close();
}
const sizes = {}; for (const r of rows) { const k = r.tag.split(' ').slice(0, 3).join(' '); sizes[k] = Math.round(r.boxW); }
console.log('zone box widths (px):', JSON.stringify(sizes));
await b.close(); console.log(`${oks} ok, ${fails} FAIL`); process.exitCode = fails ? 1 : 0;

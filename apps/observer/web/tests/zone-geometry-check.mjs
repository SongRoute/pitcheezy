// Zone geometry regression (integration rev3, after geometry-review-037; pitcher's view (SONG-03)). Route-mocked fixture, headless Chromium.
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
// same split as model.ts resultKind (UI 10/26 ②): ball family / in play / everything else is strike·foul; no description → unknown
const kindOf = (d) => !d ? 'unknown' : ['ball', 'blocked_ball', 'pitchout', 'hit_by_pitch', 'intent_ball', 'automatic_ball'].includes(d) ? 'ball' : d.startsWith('hit_into_play') ? 'inplay' : 'strike';
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
      // Recover raw catcher coordinates from the mirrored pitcher view; z unchanged
      const fx = -(cx - (box.left + box.width / 2)) / (box.width / 1.66), fz = B.bottom + (box.bottom - cy) / (box.height / (B.top - B.bottom));
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
  const detail = await p.evaluate(B => { const svg=document.querySelector('.sv-detail-zone .sv-zone'), box=svg.querySelector('.sv-zone-box').getBoundingClientRect(), a=svg.querySelector('.sv-actual'), cr=a.querySelector('circle').getBoundingClientRect(), cx=cr.left+cr.width/2, cy=cr.top+cr.height/2; return {fx:-(cx-(box.left+box.width/2))/(box.width/1.66),fz:B.bottom+(box.bottom-cy)/(box.height/(B.top-B.bottom)),out:a.classList.contains('out'),dataX:Number(a.dataset.x),dataZ:Number(a.dataset.z),screenSide:Math.sign(cx-(box.left+box.width/2)),inBox:cx>=box.left&&cx<=box.right&&cy>=box.top&&cy<=box.bottom,boxW:box.width}; },bounds);
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
    if (Math.abs(pt[0]) > .05) ok(r.screenSide === -Math.sign(pt[0]), `${tag}: pitcher view (plate_x ${pt[0]} drawn on the ${pt[0] > 0 ? 'left' : 'right'})`, r);
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
  // UI 10/26 ②: a shared target is ONE square in the highest rank's look; rank 2 light fill; every border solid (no dashed rank 3)
  const look = await p.evaluate(() => [...document.querySelectorAll('svg.sv-zone:not(.mini) .sv-rec')].map(g => { const r = getComputedStyle(g.querySelector('rect'));
    return { rank: g.dataset.rank, cls: g.getAttribute('class'), fill: r.fill, dash: r.strokeDasharray, text: getComputedStyle(g.querySelector('text')).fill }; }));
  const merged = look.find(l => l.rank === '1·3'), second = look.find(l => l.rank === '2');
  ok(merged && /\br1\b/.test(merged.cls) && merged.fill === 'rgb(6, 119, 111)' && merged.text === 'rgb(255, 255, 255)' && second && second.fill === 'rgb(220, 239, 233)' && look.every(l => l.dash === 'none'),
    `${w}x${h} overlap: '1·3' drawn as rank 1 (filled teal, white number), rank 2 light, no dashed border`, look);
  // the selected (finished-game) pitch is a G1 mark with its number and the outer ring, its kind from the pitch description
  const g1 = await p.evaluate(() => { const a = document.querySelector('svg.sv-zone:not(.mini) .sv-actual'); return { kind: a.dataset.kind, ring: !!a.querySelector('.sv-ring'), num: a.querySelector('.sv-g1-num, .sv-g1-out')?.textContent ?? null }; });
  ok(g1.kind === kindOf(fx.final.pitches[0].actual.description) && g1.ring && g1.num === String(fx.final.pitches[0].pitch_number), `${w}x${h} selected pitch: G1 '${g1.kind}' with number ${g1.num} and ring`, g1);
  if (out) await p.screenshot({ path: path.join(out, `overlap-${w}x${h}.png`) });
  await c.close();
}
// UI 10/26 ②: on a live game the at-bat's thrown pitches are G1 marks (kind from `description`), the newest one ringed; numbers stay
// on their marks and the marks stay at their true coordinates (checked above for the selected pitch, here per thrown pitch).
for (const [w, h] of [[375, 667], [320, 568]]) {
  const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true }); const p = await c.newPage();
  const body = JSON.parse(JSON.stringify(fx.live_more));
  // the last five pitches become the current at-bat; the three drawn ones cover every kind across the two runs
  const pa = body.pitches.slice(-5);
  const descs = w === 375 ? ['ball', 'called_strike', 'hit_into_play', 'foul', 'ball'] : ['foul', 'ball', 'blocked_ball', null, 'swinging_strike'];
  pa.forEach((x, i) => { x.pa_key = body.current.pa_key; x.pitch_number = i + 1; x.actual.description = descs[i]; x.actual.x ??= 0.2 * i - 0.4; x.actual.z ??= 2.4; });
  await p.route('**/*', (route) => { const u = new URL(route.request().url());
    if (u.origin !== 'http://pz.test') return route.abort('blockedbyclient');
    if (u.pathname === '/api/service/games/849843') return route.fulfill({ json: body });
    if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'x' } });
    const f = path.join(dist, u.pathname); const t = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(t), contentType: TYPES[path.extname(t)] ?? 'application/octet-stream' }); });
  await p.goto('http://pz.test/watch/849843?demo=fixture'); await p.waitForSelector('svg.sv-zone:not(.mini) .sv-zone-box'); await p.waitForTimeout(200);
  const dots = await p.evaluate(() => [...document.querySelectorAll('svg.sv-zone:not(.mini) .sv-pa-dot')].map(g => ({ kind: g.dataset.kind, latest: g.classList.contains('latest'), ring: !!g.querySelector('.sv-ring'),
    n: g.querySelector('text').textContent, out: g.classList.contains('out'), x: Number(g.dataset.x), z: Number(g.dataset.z), fill: getComputedStyle(g.querySelector('circle')).fill, numFill: getComputedStyle(g.querySelector('text')).fill })));
  const shown = pa.filter(x => x.actual?.x != null && x.actual?.z != null).slice(-3);
  ok(dots.length === shown.length && dots.every(d => { const src = shown.find(x => String(x.pitch_number) === d.n); return src && d.kind === kindOf(src.actual.description) && d.x === src.actual.x && d.z === src.actual.z; }),
    `${w}x${h} live: thrown marks carry the kind of their own pitch (${dots.map(d => `${d.n}:${d.kind}`).join(' ')})`, { dots, shown: shown.map(x => [x.pitch_number, x.actual.description]) });
  const fills = { ball: 'rgb(255, 255, 255)', strike: 'rgb(92, 107, 115)', inplay: 'rgb(25, 46, 56)', unknown: 'rgb(255, 255, 255)' };
  // a number inside its mark is white on the filled kinds; beside an edge marker it is the dark label colour
  ok(dots.every(d => d.fill === fills[d.kind] && d.numFill === (d.out || d.kind === 'ball' || d.kind === 'unknown' ? 'rgb(74, 90, 98)' : 'rgb(255, 255, 255)')), `${w}x${h} live: G1 colours (ball hollow, strike grey, in play dark; readable numbers)`, dots);
  const newest = String(Math.max(...shown.map(x => x.pitch_number)));
  ok(dots.filter(d => d.latest && d.ring).length === 1 && dots.find(d => d.latest)?.n === newest, `${w}x${h} live: only the newest pitch (${newest}) has the outer ring`, dots);
  await c.close();
}
const sizes = {}; for (const r of rows) { const k = r.tag.split(' ').slice(0, 3).join(' '); sizes[k] = Math.round(r.boxW); }
console.log('zone box widths (px):', JSON.stringify(sizes));
await b.close(); console.log(`${oks} ok, ${fails} FAIL`); process.exitCode = fails ? 1 : 0;

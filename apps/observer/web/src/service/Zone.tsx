import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Bounds, Candidate, ResultKind } from './model';

/** Pitcher's view. Drawing mirrors the source plate_x; measured marks never move to avoid overlap.
 * Off-window marks use the existing explicit edge-arrow contract. Mitt remains x-only.
 * Thrown pitches use the colourless G1 marks (UI 10/26 ②): ball = hollow, strike/foul = grey, in play = dark; the latest one has an
 * outer ring. `label` replaces the number (pitch-mix map: the type's first letter). */
export type Mark = { x: number; z: number };
export type Thrown = Mark & { number: number; kind?: ResultKind; latest?: boolean; label?: string };
export const PLATE_HALF = .83;
/** Full detail / mini margins in feet. The live main window uses x=.12, z=.24. */
export const MARGIN = { main: { x: .42, z: .45 }, mini: { x: .34, z: .36 } };
const U = 100;  // drawing units per foot
/** -1 = pitcher view; this changes drawing only, never source coordinates. */
export const VIEW_X = -1;
const ux = (x: number) => VIEW_X * x * U;     // feet (catcher-view plate_x) → screen units
const uz = (z: number, aspect = 1) => -z * U * aspect;             // z up

/** How far a sheet's full window may widen to keep its selected pitch at its true place (feet; beyond this it is an edge marker). */
export const DETAIL_REACH = { x: 2.2, zLo: -0.5, zHi: 5.5 };
export function zoneWindow(bounds: Bounds, mini: boolean, _compact = false, fullWindow = false, include: Mark | null = null) {
  const m = mini ? MARGIN.mini : fullWindow ? MARGIN.main : { x: .12, z: .24 };
  const aspect = 1;
  let xMax = PLATE_HALF + m.x, zLo = bounds.bottom - m.z, zHi = bounds.top + m.z;
  // a sheet's window (design 19/20) widens to contain its selected pitch, with room for the mark and its ring; the main zone never does
  if (fullWindow && include) { const room = .3;
    if (Math.abs(include.x) <= DETAIL_REACH.x) xMax = Math.max(xMax, Math.abs(include.x) + room);
    if (include.z >= DETAIL_REACH.zLo && include.z <= DETAIL_REACH.zHi) { zLo = Math.min(zLo, include.z - room); zHi = Math.max(zHi, include.z + room); } }
  // drawn area = window + a pad so an edge marker (and its arrow) is fully visible; the pad changes no coordinate or out-of-window rule
  const pad = mini ? .2 : fullWindow ? .16 : .08;
  return { aspect, margin: m, xMin: -xMax, xMax, zLo, zHi, vb: { x: -(xMax + pad) * U, y: uz(zHi + pad, aspect), w: 2 * (xMax + pad) * U, h: (zHi - zLo + 2 * pad) * U * aspect } };
}
/** Where a mark is drawn and whether it is beyond the window (then drawn as an edge marker, never a normal dot). */
export function place(p: Mark, w: ReturnType<typeof zoneWindow>) {
  const out = p.x < w.xMin || p.x > w.xMax || p.z < w.zLo || p.z > w.zHi;
  const x = Math.max(w.xMin, Math.min(w.xMax, p.x)), z = Math.max(w.zLo, Math.min(w.zHi, p.z));
  // dx = SCREEN direction of the clipped-off part (feet direction × VIEW_X), used for the edge inset and the outward arrow
  return { out, x: ux(x), y: uz(z, w.aspect), dx: VIEW_X * Math.sign(p.x - x), dz: Math.sign(p.z - z) };
}
export const KIND_WORD: Record<ResultKind, string> = { ball: '볼', strike: '스트라이크·파울', inplay: '타격', unknown: '결과 미확인' };
const ft = (p: Mark) => `가로 ${p.x.toFixed(2)}ft (포수 기준), 높이 ${p.z.toFixed(2)}ft`;

/** Candidates sharing a target are drawn once with a combined label ('1·3'). */
export function groupTargets(cands: Candidate[]): { x: number; z: number; ranks: number[] }[] {
  const out: { x: number; z: number; ranks: number[] }[] = [];
  for (const c of [...cands].sort((a, b) => a.rank - b.rank)) {
    if (!c.target) continue;
    const g = out.find(o => Math.abs(o.x - c.target!.x) < .05 && Math.abs(o.z - c.target!.z) < .05);
    if (g) g.ranks.push(c.rank); else out.push({ x: c.target.x, z: c.target.z, ranks: [c.rank] });
  }
  return out;
}

export default function Zone({ bounds, candidates, actual, actualLabel, setupX, mini = false, keepOut = null, fullWindow = false, actuals = [], actualKind = 'unknown', actualNumber = null }:
  { bounds: Bounds; candidates: Candidate[]; actual: Mark | null; actualLabel: string; setupX: number | null; mini?: boolean; fullWindow?: boolean; actuals?: Thrown[];
    /** the single selected pitch: its G1 kind and number (drawn with the latest ring); no number → a plain mark as before */
    actualKind?: ResultKind; actualNumber?: number | null;
    /** CSS selector of a sibling block (the recommendation band) painted over this svg element's top-left/right corner. The drawing
     *  reserves horizontal room, and the recommendation block aligns with the grid top. */
    keepOut?: string | null }) {
  const [compact, setCompact] = useState(() => window.innerWidth < 360);
  useEffect(() => { const resize = () => setCompact(window.innerWidth < 360); window.addEventListener('resize', resize); return () => window.removeEventListener('resize', resize); }, []);
  const w = zoneWindow(bounds, mini, compact && !fullWindow, fullWindow, actual);
  const left = -PLATE_HALF * U, right = PLATE_HALF * U, top = uz(bounds.top, w.aspect), bottom = uz(bounds.bottom, w.aspect);  // symmetric, view-independent
  const groups = groupTargets(candidates).map(g => ({ ...g, at: place(g, w) }));
  const act = actual ? place(actual, w) : null;
  // pitches at the same spot AND of the same result share one mark (numbers '1·3'). Different results at one spot are 'stacked': the
  // mark is split into equal wedges around the SAME centre (one per result), and each number goes beside it in a small badge of its own
  // result, so no number is drawn in another result's look and no mark moves. The ring follows the newest pitch.
  const actualGroups: (Thrown & { numbers: (number | string)[]; stacked: boolean; slot: number; slots: number })[] = [];
  for (const p of actuals) { const n = p.label ?? p.number;
    const g = actualGroups.find(g => g.x === p.x && g.z === p.z && g.kind === p.kind);
    if (g) { g.numbers.push(n); g.latest = g.latest || p.latest; } else actualGroups.push({ ...p, numbers: [n], stacked: false, slot: 0, slots: 1 }); }
  for (const g of actualGroups) { const here = actualGroups.filter(o => o.x === g.x && o.z === g.z); g.stacked = here.length > 1; g.slot = here.indexOf(g); g.slots = here.length; }
  const parts = [...groups.map(g => `추천 ${g.ranks.join('·')}순위 위치 네모${g.at.out ? `(화면 밖 ${ft(g)})` : ''}`),
    actual ? `실제 투구${actualLabel ? ` ${actualLabel}` : ''} 동그라미${act!.out ? `(화면 밖, ${ft(actual)})` : ''}` : null].filter(Boolean);
  // Labels keep a readable on-screen size whatever the zone's drawn scale (SVG text otherwise scales with the drawing).
  const ref = useRef<SVGSVGElement>(null);
  const [scale, setScale] = useState(1);
  // fit = the svg element's whole visible area in drawing units (the drawing window w.vb sits inside it, centred horizontally) plus the
  // keep-out block in the same units. Labels may use the area around the drawing; marks never move.
  type Fit = { x: number; y: number; w: number; h: number; ko: { x: number; y: number; w: number; h: number } | null };
  const [fit, setFit] = useState<Fit | null>(null);
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return;
    const block = keepOut ? el.closest('.sv-zrow')?.querySelector<HTMLElement>(keepOut) ?? null : null;
    const measure = () => {
      const r = el.getBoundingClientRect(); if (!r.width || !r.height) return;
      const PAD = mini ? 0 : 8;  // px kept free above and below the drawing (edge arrows and labels near the window edge)
      const W = r.width, H = r.height, k1 = Math.min(W / w.vb.w, (H - 2 * PAD) / w.vb.h);
      let k = k1;
      const row = el.closest<HTMLElement>('.sv-zrow');
      const b = block?.getBoundingClientRect();
      // Reserve space beside the grid for recommendation rows, aligned to its top edge.
      if (b && row) {
        const rr = row.getBoundingClientRect();
        const sideRoom = rr.width / 2 - b.width - 4;
        // A stable visible window reserves room for the outside-zone dot/arrow too. It never depends on the selected pitch.
        k = Math.max(.01, Math.min(k1, (sideRoom - 8) / (w.xMax * U)));
      }
      const dh = w.vb.h * k, topPx = (H - dh) / 2;
      const bandTop = topPx + (uz(bounds.top, w.aspect) - w.vb.y) * k;
      if (row && !mini) { const off = r.top - row.getBoundingClientRect().top;
        row.style.setProperty('--sv-band-top', `${off + bandTop}px`);
        row.style.setProperty('--sv-zone-mid', `${off + topPx + dh / 2}px`); }
      const vw = W / k, vh = H / k, x0 = w.vb.x - (vw - w.vb.w) / 2, y0 = w.vb.y - topPx / k;
      setScale(k);
      setFit({ x: x0, y: y0, w: vw, h: vh, ko: b ? { x: x0 + (b.left - r.left) / k, y: y0 + bandTop / k, w: b.width / k, h: b.height / k } : null });
    };
    measure(); const ro = new ResizeObserver(measure); ro.observe(el); if (block) ro.observe(block); return () => ro.disconnect();
  }, [w.vb.w, w.vb.h, w.vb.x, w.vb.y, keepOut, mini, bounds.top, JSON.stringify(candidates.map(c => c.target)), actual?.x, actual?.z, setupX, w.aspect]);
  const room = fit ?? { x: w.vb.x, y: w.vb.y, w: w.vb.w, h: w.vb.h, ko: null };
  // free space under the lowest PAINTED item (marks, labels, zone) inside the zone row, published as --sv-below so the result line
  // can sit midway between the zone and the cards (display only; read after every render, no React state)
  useLayoutEffect(() => {
    const el = ref.current; const row = el?.closest<HTMLElement>('.sv-zrow'); if (!el || !row || mini) return;
    const lowest = Math.max(...[...el.querySelectorAll('rect, circle, path, line, text')].map(e => e.getBoundingClientRect().bottom));
    (row.parentElement ?? row).style.setProperty('--sv-below', `${Math.max(0, Math.round(row.getBoundingClientRect().bottom - lowest - 8))}px`);  // on the stage: the result line is the row's sibling
    // --sv-below-min: free space under the DRAWING WINDOW (no mark, arrow or label is ever painted below it), so a block that must
    // not move when marks appear or disappear (the live state slot) is placed from a value that does not depend on what is painted.
    const ctm = el.getScreenCTM();
    const windowBottom = ctm ? ctm.f + ctm.d * (w.vb.y + w.vb.h) : row.getBoundingClientRect().bottom;
    (row.parentElement ?? row).style.setProperty('--sv-below-min', `${Math.max(0, Math.floor(row.getBoundingClientRect().bottom - windowBottom))}px`);
  });
  const u = (n: number) => n / scale;  // on-screen px → drawing units
  const px = (n: number) => `${u(n).toFixed(2)}px`;
  // the small full-window zone of a sheet uses smaller marks (design 19/20) so a combined square stays within about one cell
  const half = mini ? 0 : fullWindow ? u(11) : u(15), rA = mini ? u(6.5) : fullWindow ? u(9) : u(10.5), rOut = mini ? u(4.5) : rA, ring = fullWindow ? u(3) : u(4);
  const labels = { '--zl-s': px(13), '--zl-m': px(14), '--zl-l': px(15) } as CSSProperties;
  const arrowLen = mini ? u(3) : u(7);
  // An edge marker is not a coordinate (the pitch is beyond the window); it is drawn just inside the window edge so that it and its
  // arrow are fully visible. The inset never reaches the zone: it is capped below the outside margin.
  const inset = (p: ReturnType<typeof place>, size: number) => {
    if (!p.out) return p;
    const sx_ = Math.min(size + arrowLen, w.margin.x * U - u(3)), sz_ = Math.min(size + arrowLen, w.margin.z * U - u(3));
    return { ...p, x: p.x - p.dx * sx_, y: p.y + p.dz * sz_ };
  };
  // the outward pointer: a small triangle whose tip is arrowLen beyond the mark's edge (same reach as the former line)
  const arrow = (p: { x: number; y: number; dx: number; dz: number }, r: number) => {
    const ax = p.dx, ay = -p.dz, n = Math.hypot(ax, ay) || 1, ux_ = ax / n, uy = ay / n, L = r + arrowLen, w = arrowLen * .75;
    const bx = p.x + ux_ * (r + .5), by = p.y + uy * (r + .5);
    return <path className="sv-out-arrow" d={`M${p.x + ux_ * L} ${p.y + uy * L} L${bx - uy * w} ${by + ux_ * w} L${bx + uy * w} ${by - ux_ * w}Z`} />;
  };
  const mitt = typeof setupX === 'number' ? place({ x: setupX, z: (bounds.top + bounds.bottom) / 2 }, w) : null;
  // ---- display-only label layout (marks keep their positions) ----
  type Box = { x: number; y: number; w: number; h: number };
  const hit = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  // labels (actual, mitt) may use the whole visible svg area around the drawing (still on screen, never under the recommendation band)
  const visible = (a: Box) => a.x >= room.x && a.y >= room.y && a.x + a.w <= room.x + room.w && a.y + a.h <= room.y + room.h && !(room.ko && hit(a, room.ko));
  const recs0 = groups.map(g => { const hw0 = Math.max(half, u(5 + 4.5 * (g.ranks.join('·').length))); const d = inset(g.at, Math.max(hw0, half));
    // a square near the drawing edge is drawn smaller (its centre = the target stays exact), never moved
    const room = (v: number, lo: number, hi: number) => Math.min(v - lo, hi - v);
    const hw = Math.max(u(9), Math.min(hw0, room(d.x, w.vb.x, w.vb.x + w.vb.w) - u(1))), hh = Math.max(u(9), Math.min(half, room(d.y, w.vb.y, w.vb.y + w.vb.h) - u(1)));
    return { ...g, hw, hh, d }; });
  // rank numbers of squares that overlap an earlier square are written just outside their own square (label only)
  const numBoxes: Box[] = [];
  const recs = recs0.map((g, i) => { const box = { x: g.d.x - g.hw, y: g.d.y - g.hh, w: 2 * g.hw, h: 2 * g.hh };
    const clash = recs0.slice(0, i).some(o => box.x < o.d.x + o.hw && o.d.x - o.hw < box.x + box.w && box.y < o.d.y + o.hh && o.d.y - o.hh < box.y + box.h);
    const tw = u(6 + 8 * g.ranks.join('·').length);
    const spots: [number, number][] = clash ? [[0, g.hh + u(15)], [0, -g.hh - u(4)], [g.hw + tw / 2 + u(2), u(5.5)], [-g.hw - tw / 2 - u(2), u(5.5)]] : [[0, u(5.5)]];
    const boxAt = ([dx, dy]: [number, number]) => ({ x: g.d.x + dx - tw / 2, y: g.d.y + dy - u(14), w: tw, h: u(16) });
    const fits = (b: Box) => b.x >= w.vb.x && b.y >= w.vb.y && b.x + b.w <= w.vb.x + w.vb.w && b.y + b.h <= w.vb.y + w.vb.h;
    const pick = spots.find(sp => { const b = boxAt(sp); return fits(b) && !numBoxes.some(n => n.x < b.x + b.w && b.x < n.x + n.w && n.y < b.y + b.h && b.y < n.y + n.h); }) ?? spots[0];
    numBoxes.push(boxAt(pick));
    return { ...g, tx: g.d.x + pick[0], ty: g.d.y + pick[1] }; });
  // the number box is as wide as its text ('1·2·3' is wider than '1')
  const recBoxes: Box[] = recs.flatMap(g => { const tw = u(6 + 8 * g.ranks.join('·').length);
    return [{ x: g.d.x - g.hw, y: g.d.y - g.hh, w: 2 * g.hw, h: 2 * g.hh }, { x: g.tx - tw / 2, y: g.ty - u(15), w: tw, h: u(18) }]; });
  const actD = act ? inset(act, act.out ? rOut : rA) : null;
  const rD = act?.out ? rOut : rA;
  const rDot = rD + (actualNumber !== null && !mini ? ring : 0);  // the selected pitch's ring is part of its painted mark
  const dotBox: Box | null = actD ? { x: actD.x - rDot - u(2), y: actD.y - rDot - u(2), w: 2 * rDot + u(4), h: 2 * rDot + u(4) } : null;
  const mittText = mitt ? (mitt.out ? (mitt.dx > 0 ? '미트→' : '←미트') : '미트') : '';
  const mittW = u(mitt?.out ? 34 : 26), lh = u(16);
  const mittLabel = (() => {
    if (!mitt) return { x: 0, y: 0, box: null as Box | null };
    const x = Math.max(w.vb.x + mittW / 2, Math.min(w.vb.x + w.vb.w - mittW / 2, mitt.out ? mitt.x - mitt.dx * u(14) : mitt.x));
    for (const y of [top - u(6), bottom + u(16), top - u(24), bottom + u(34)]) { const box = { x: x - mittW / 2, y: y - lh + u(3), w: mittW, h: lh };
      if (visible(box) && !recBoxes.some(b => hit(b, box)) && !(dotBox && hit(dotBox, box))) return { x, y, box }; }
    return { x, y: top - u(6), box: { x: x - mittW / 2, y: top - u(6) - lh + u(3), w: mittW, h: lh } };
  })();
  // the actual's label is its pitch type + speed (e.g. '커브 133'); width from the text at the 15px label size (Hangul ≈1em,
  // Latin/digits ≈0.62em, space ≈0.3em) so the placement below keeps it inside the drawing and off the recommendation marks
  const textPx = [...actualLabel].reduce((a, ch) => a + (/[\uac00-\ud7a3]/.test(ch) ? 15 : ch === ' ' ? 4.5 : 9.3), 0);
  const labW = u(Math.max(30, textPx + 6));
  const actLabel = (() => {
    if (!actD || mini || !actualLabel) return null;
    const o = Math.max(rD, half) + u(4), so = Math.max(rD, ...recs.map(g => g.hw), u(9)) + u(4), cands = [[0, o + lh - u(2)], [0, -o - u(3)], [so + labW / 2, u(5)], [-so - labW / 2, u(5)], [so + labW / 2, -o], [-so - labW / 2, -o], [so + labW / 2, o + lh], [-so - labW / 2, o + lh], [0, 2 * o + lh], [0, -2 * o], [0, 3 * o + lh], [0, -3 * o]];
    const avoid = [...recBoxes, ...(mittLabel.box ? [mittLabel.box] : []), ...(dotBox ? [dotBox] : [])];
    let first: { x: number; y: number } | null = null;
    // a wide label near the drawing edge slides sideways to stay fully visible (only the text moves; the dot stays at its coordinate)
    const slide = (x: number) => Math.max(room.x + labW / 2 + u(1), Math.min(room.x + room.w - labW / 2 - u(1), x));
    for (const [dx, dy] of cands) { const x = slide(actD.x + dx), y = actD.y + dy, box = { x: x - labW / 2, y: y - lh - u(1), w: labW, h: lh + u(5) };  // glyph box + 2px
      if (!visible(box)) continue; first = first ?? { x, y }; if (!avoid.some(b => hit(b, box))) return { x, y, box }; }
    return first ? { ...first, box: { x: first.x - labW / 2, y: first.y - lh - u(1), w: labW, h: lh + u(5) } } : null;
  })();
  // the selected pitch's number sits in its mark unless the mark is under a recommendation square (drawn above it); then it goes just
  // outside, on the first free side (display only: the mark stays at its coordinate)
  const actNum = (() => {
    if (!actD || actualNumber === null || mini) return null;
    if (!(dotBox && recBoxes.some(bx => hit(bx, dotBox)))) return { x: actD.x, y: actD.y + u(4), out: false };
    const tw = u(6 + 7 * String(actualNumber).length), R = rD + ring;
    const spots = [[0, -R - u(6)], [0, R + u(14)], [R + tw / 2 + u(4), u(4)], [-R - tw / 2 - u(4), u(4)]].map(([dx, dy]) => ({ x: actD.x + dx - tw / 2, y: actD.y + dy - u(11), w: tw, h: u(14) }));
    const avoid = [...recBoxes, ...(mittLabel.box ? [mittLabel.box] : [])];
    const bx = spots.find(q => visible(q) && !avoid.some(o => hit(o, q))) ?? spots.find(visible) ?? spots[0];
    return { x: bx.x + tw / 2, y: bx.y + u(11), out: true };
  })();
  const paMarks = actualGroups.map(p => { const at = place(p, w); return { p, at, d: inset(at, rA) }; });
  const paDotBoxes = paMarks.map(({ p, d }) => { const r = rA + (p.latest ? ring : 0); return { x: d.x - r - u(2), y: d.y - r - u(2), w: 2 * r + u(4), h: 2 * r + u(4) }; });
  const paNumberBoxes: Box[] = [];
  const paLabels = paMarks.map(({ p, d }) => {
    if (p.numbers.length === 1 && !p.stacked) return { x: d.x, y: d.y + u(4), halo: false };
    // a stacked group's number sits in a badge (a small circle of its result), so its box is at least the badge
    const tw = Math.max(u(6 + 7 * p.numbers.join('·').length), p.stacked ? u(19) : 0), side = rA + (p.latest ? ring : 0) + tw / 2 + u(4);
    // positions scale with the drawn dot (and its ring): just above, just below, either side, then one row further out
    const R = rA + (p.latest ? ring : 0), spots = [[0, -R - u(6)], [0, R + u(14)], [side, u(4)], [-side, u(4)], [0, -R - u(22)], [0, R + u(32)]]
      .map(([dx, dy]) => ({ x: d.x + dx - tw / 2, y: d.y + dy - u(p.stacked ? 14 : 11), w: tw, h: u(p.stacked ? 19 : 14) }));
    const textAvoid = [...numBoxes, ...paDotBoxes, ...paNumberBoxes, ...(mittLabel.box ? [mittLabel.box] : []), ...(actLabel ? [actLabel.box] : [])];
    const clear = spots.find(b => visible(b) && ![...recBoxes, ...textAvoid].some(a => hit(a, b)));
    // ponytail: six label positions; dense views fall back to a screen-sized halo. Expand positions only if real cases need it.
    const box = clear ?? spots.find(b => visible(b) && !textAvoid.some(a => hit(a, b))) ?? spots.find(visible) ?? spots[0];
    paNumberBoxes.push(box);
    return { x: box.x + tw / 2, y: box.y + u(p.stacked ? 14 : 11), halo: !clear && !p.stacked };
  });
  // the selected pitch is painted ABOVE the squares (its fill and ring stay visible); a rank number its mark would cover is repainted
  // above it (same place, same words) so both stay readable
  const lifted = actD && dotBox ? recs.filter(g => { const tw = u(6 + 8 * g.ranks.join('·').length); return hit(dotBox, { x: g.tx - tw / 2, y: g.ty - u(15), w: tw, h: u(18) }); }) : [];
  const wedge = (cx: number, cy: number, r: number, i: number, n: number) => { const a0 = -Math.PI / 2 + 2 * Math.PI * i / n, a1 = a0 + 2 * Math.PI / n;
    return `M${cx} ${cy}L${cx + r * Math.cos(a0)} ${cy + r * Math.sin(a0)}A${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${cx + r * Math.cos(a1)} ${cy + r * Math.sin(a1)}Z`; };
  return <svg ref={ref} style={labels} className={`sv-zone${mini ? ' mini' : ''}`} viewBox={`${room.x} ${room.y} ${room.w} ${room.h}`} preserveAspectRatio="xMidYMid meet" role="img"
    aria-label={mini ? undefined : `투수 시점 스트라이크존${parts.length ? ': ' + parts.join(', ') : ''}${actuals.length ? `, 이번 타석 ${actuals.length}개 투구` : ''}`} aria-hidden={mini || undefined}>
    <rect x={-w.xMax * U} y={uz(w.zHi, w.aspect)} width={2 * w.xMax * U} height={(w.zHi - w.zLo) * U * w.aspect} className="sv-zone-outside" />
    <rect x={left} y={top} width={right - left} height={bottom - top} className="sv-zone-box" />
    {[1, 2].map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="sv-zone-grid" />)}
    {[1, 2].map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="sv-zone-grid" />)}
    {mitt && <g className={`sv-mitt${mitt.out ? ' out' : ''}`} data-x={setupX}>
      <line x1={mitt.x} x2={mitt.x} y1={top - u(4)} y2={bottom + u(4)} />
      <text x={mittLabel.x} y={mittLabel.y} textAnchor="middle">{mittText}</text></g>}
    {recs.map(g => <g key={g.ranks.join()} className={`sv-rec r${g.ranks[0]}${g.at.out ? ' out' : ''}`} data-rank={g.ranks.join('·')} data-x={g.x} data-z={g.z} data-out={g.at.out || undefined}>
      <rect x={g.d.x - g.hw} y={g.d.y - g.hh} width={2 * g.hw} height={2 * g.hh} rx={u(5)} />{g.at.out && arrow(g.d, Math.min(g.hw, g.hh))}
      <text x={g.tx} y={g.ty} textAnchor="middle" visibility={lifted.includes(g) ? 'hidden' : undefined}>{g.ranks.join('·')}</text></g>)}
    {/* the selected pitch over the squares, then any rank number it would cover */}
    {act && <g className={`sv-actual${act.out ? ' out' : ''}${actualNumber !== null ? ` g1 ${actualKind} latest` : ''}`} data-out={act.out || undefined} data-x={actual!.x} data-z={actual!.z} data-kind={actualNumber !== null ? actualKind : undefined}>
      <circle cx={actD!.x} cy={actD!.y} r={act.out ? rOut : rA} />{act.out && arrow(actD!, rOut)}
      {actualNumber !== null && !mini && <><circle className="sv-ring" cx={actD!.x} cy={actD!.y} r={rD + ring} />
        {actNum && <text className={actNum.out ? 'sv-g1-out' : 'sv-g1-num'} x={actNum.x} y={actNum.y} textAnchor="middle" style={{ fontSize: px(actNum.out ? 11 : 12), stroke: actNum.out ? 'white' : 'none', strokeWidth: actNum.out ? px(3) : undefined }}>{actualNumber}</text>}</>}
      {!mini && actLabel && <text className="sv-actual-label" x={actLabel.x} y={actLabel.y} textAnchor="middle">{actualLabel}</text>}</g>}
    {lifted.map(g => <text key={g.ranks.join()} className={`sv-rec-lift r${g.ranks[0]}`} data-rank={g.ranks.join('·')} x={g.tx} y={g.ty} textAnchor="middle" aria-hidden="true">{g.ranks.join('·')}</text>)}
    {paMarks.map(({ p, at, d }, i) => { const label = paLabels[i]; const inside = p.numbers.length === 1 && !p.stacked && !at.out;
      return <g key={`${p.x}:${p.z}:${p.kind ?? '-'}`} className={`sv-actual sv-pa-dot g1 ${p.label ? 'label' : p.kind ?? 'unknown'}${p.latest ? ' latest' : ''}${at.out ? ' out' : ''}`} data-x={p.x} data-z={p.z} data-out={at.out || undefined} data-kind={p.kind ?? 'unknown'}>
        {p.stacked ? <><circle className="sv-anchor" cx={d.x} cy={d.y} r={rA} /><path className="sv-wedge" d={wedge(d.x, d.y, rA, p.slot, p.slots)} /></> : <circle cx={d.x} cy={d.y} r={rA} />}
        {at.out && arrow(d, rA)}{p.latest && <circle className="sv-ring" cx={d.x} cy={d.y} r={rA + ring} />}
        {p.stacked && <circle className="sv-badge" cx={label.x} cy={label.y - u(4)} r={u(8.5)} />}
        <text className={inside ? 'sv-g1-num' : p.stacked ? 'sv-g1-num badge' : 'sv-g1-out'} x={label.x} y={label.y} textAnchor="middle" style={{ fontSize: px(inside || p.stacked ? 12 : 11), stroke: inside || !label.halo ? 'none' : 'white', strokeWidth: label.halo && !inside ? px(3) : undefined }}>{p.numbers.join('·')}</text>
        <title>{p.numbers.join('·')}{typeof p.numbers[0] === 'number' ? '구' : ''}{p.kind ? ` · ${KIND_WORD[p.kind]}` : ''} · {ft(p)}{at.out ? ' · 화면 밖' : ''}</title></g>; })}
  </svg>;
}

/** The same G1 mark outside the drawing (at-bat list, pitch sheet): a small circle with the pitch number. */
export function G1Dot({ kind, n, ring = false }: { kind: ResultKind; n: number | string; ring?: boolean }) {
  return <i className={`sv-g1 ${kind}${ring ? ' ring' : ''}`} aria-hidden="true">{n}</i>;
}
/** The rank square as on the zone: 1 filled, 2 light, 3 white with a solid border (UI 10/26 ③). */
export const RankSq = ({ rank }: { rank: number }) => <i className={`sv-rk r${Math.min(rank, 3)}`} aria-hidden="true">{rank}</i>;

/** One-line key under the zone (UI 10/26 ②): the recommendation square and the three G1 result marks, then the view. */
export function ZoneLegend({ side, actualOnly = null }: { side: 'L' | 'R' | null; actualOnly?: string | null }) {
  return <div className="sv-zone-legend"><p aria-hidden="true"><span><i className="k rec" />추천</span>{actualOnly ? <span><i className={`k ${actualOnly}`} />실제 ({actualOnly === 'ball' ? '볼' : actualOnly === 'inplay' ? '타격' : actualOnly === 'strike' ? '스트라이크·파울' : '결과 미확인'})</span>
    : <><span><i className="k ball" />볼</span><span><i className="k strike" />스트라이크·파울</span><span><i className="k inplay" />타격</span></>}{actualOnly && <span>투수 시점</span>}</p>
    {!actualOnly && <p>투수 시점 · {side === 'L' ? '좌타자' : side === 'R' ? '우타자' : '타석 방향 미확인'}</p>}</div>;
}

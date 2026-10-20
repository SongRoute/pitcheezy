import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Bounds, Candidate } from './model';

/** Pitcher's view. Drawing mirrors the source plate_x; measured marks never move to avoid overlap.
 * Off-window marks use the existing explicit edge-arrow contract. Mitt remains x-only. */
export type Mark = { x: number; z: number };
export const PLATE_HALF = .83;
/** Full detail / mini margins in feet. The live main window uses x=.12, z=.24. */
export const MARGIN = { main: { x: .42, z: .45 }, mini: { x: .34, z: .36 } };
const U = 100;  // drawing units per foot
/** -1 = pitcher view; this changes drawing only, never source coordinates. */
export const VIEW_X = -1;
const ux = (x: number) => VIEW_X * x * U;     // feet (catcher-view plate_x) → screen units
const uz = (z: number, aspect = 1) => -z * U * aspect;             // z up

export function zoneWindow(bounds: Bounds, mini: boolean, _compact = false, fullWindow = false) {
  const m = mini ? MARGIN.mini : fullWindow ? MARGIN.main : { x: .12, z: .24 };
  const aspect = 1;
  const xMax = PLATE_HALF + m.x, zLo = bounds.bottom - m.z, zHi = bounds.top + m.z;
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

export default function Zone({ bounds, candidates, actual, actualLabel, setupX, mini = false, keepOut = null, fullWindow = false, actuals = [] }:
  { bounds: Bounds; candidates: Candidate[]; actual: Mark | null; actualLabel: string; setupX: number | null; mini?: boolean; fullWindow?: boolean; actuals?: (Mark & { number: number })[];
    /** CSS selector of a sibling block (the recommendation band) painted over this svg element's top-left/right corner. The drawing
     *  reserves horizontal room, and the recommendation block aligns with the grid top. */
    keepOut?: string | null }) {
  const [compact, setCompact] = useState(() => window.innerWidth < 360);
  useEffect(() => { const resize = () => setCompact(window.innerWidth < 360); window.addEventListener('resize', resize); return () => window.removeEventListener('resize', resize); }, []);
  const w = zoneWindow(bounds, mini, compact && !fullWindow, fullWindow);
  const left = -PLATE_HALF * U, right = PLATE_HALF * U, top = uz(bounds.top, w.aspect), bottom = uz(bounds.bottom, w.aspect);  // symmetric, view-independent
  const groups = groupTargets(candidates).map(g => ({ ...g, at: place(g, w) }));
  const act = actual ? place(actual, w) : null;
  const actualGroups: (Mark & { numbers: number[] })[] = [];
  for (const p of actuals) { const g = actualGroups.find(g => g.x === p.x && g.z === p.z); if (g) g.numbers.push(p.number); else actualGroups.push({ ...p, numbers: [p.number] }); }
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
  const half = mini ? 0 : u(13), rA = mini ? u(6.5) : u(8), rOut = mini ? u(4.5) : rA;
  const labels = { '--zl-s': px(13), '--zl-m': px(14), '--zl-l': px(15) } as CSSProperties;
  const arrowLen = mini ? u(3) : u(7);
  // An edge marker is not a coordinate (the pitch is beyond the window); it is drawn just inside the window edge so that it and its
  // arrow are fully visible. The inset never reaches the zone: it is capped below the outside margin.
  const inset = (p: ReturnType<typeof place>, size: number) => {
    if (!p.out) return p;
    const sx_ = Math.min(size + arrowLen, w.margin.x * U - u(3)), sz_ = Math.min(size + arrowLen, w.margin.z * U - u(3));
    return { ...p, x: p.x - p.dx * sx_, y: p.y + p.dz * sz_ };
  };
  const arrow = (p: { x: number; y: number; dx: number; dz: number }, r: number) => {
    const ax = p.dx, ay = -p.dz, n = Math.hypot(ax, ay) || 1, L = r + arrowLen;
    return <path className="sv-out-arrow" d={`M${p.x + ax / n * r} ${p.y + ay / n * r} L${p.x + ax / n * L} ${p.y + ay / n * L}`} />;
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
  const dotBox: Box | null = actD ? { x: actD.x - rD - u(2), y: actD.y - rD - u(2), w: 2 * rD + u(4), h: 2 * rD + u(4) } : null;
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
  return <svg ref={ref} style={labels} className={`sv-zone${mini ? ' mini' : ''}`} viewBox={`${room.x} ${room.y} ${room.w} ${room.h}`} preserveAspectRatio="xMidYMid meet" role="img"
    aria-label={mini ? undefined : `투수 시점 스트라이크존${parts.length ? ': ' + parts.join(', ') : ''}${actuals.length ? `, 이번 타석 ${actuals.length}개 투구` : ''}`} aria-hidden={mini || undefined}>
    <rect x={-w.xMax * U} y={uz(w.zHi, w.aspect)} width={2 * w.xMax * U} height={(w.zHi - w.zLo) * U * w.aspect} className="sv-zone-outside" />
    <rect x={left} y={top} width={right - left} height={bottom - top} className="sv-zone-box" />
    {[1, 2].map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="sv-zone-grid" />)}
    {[1, 2].map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="sv-zone-grid" />)}
    {mitt && <g className={`sv-mitt${mitt.out ? ' out' : ''}`} data-x={setupX}>
      <line x1={mitt.x} x2={mitt.x} y1={top - u(4)} y2={bottom + u(4)} />
      <text x={mittLabel.x} y={mittLabel.y} textAnchor="middle">{mittText}</text></g>}
    {/* actual is drawn first so recommendation numbers stay on top of it; its label is placed apart from the dot */}
    {act && <g className={`sv-actual${act.out ? ' out' : ''}`} data-out={act.out || undefined} data-x={actual!.x} data-z={actual!.z}>
      <circle cx={actD!.x} cy={actD!.y} r={act.out ? rOut : rA} />{act.out && arrow(actD!, rOut)}
      {!mini && actLabel && <text className="sv-actual-label" x={actLabel.x} y={actLabel.y} textAnchor="middle">{actualLabel}</text>}</g>}
    {recs.map(g => <g key={g.ranks.join()} className={`sv-rec${g.at.out ? ' out' : ''}`} data-rank={g.ranks.join('·')} data-x={g.x} data-z={g.z} data-out={g.at.out || undefined}>
      <rect x={g.d.x - g.hw} y={g.d.y - g.hh} width={2 * g.hw} height={2 * g.hh} rx="3" />{g.at.out && arrow(g.d, Math.min(g.hw, g.hh))}
      <text x={g.tx} y={g.ty} textAnchor="middle">{g.ranks.join('·')}</text></g>)}
    {actualGroups.map(p => { const at = place(p, w), d = inset(at, rA); return <g key={p.numbers.join()} className={`sv-actual sv-pa-dot${at.out ? ' out' : ''}`} data-x={p.x} data-z={p.z} data-out={at.out || undefined}><circle cx={d.x} cy={d.y} r={rA} />{at.out && arrow(d, rA)}<text x={d.x} y={d.y + u(p.numbers.length > 1 ? -12 : 3.5)} textAnchor="middle" style={{fontSize:px(10),fill:p.numbers.length > 1 || at.out ? 'var(--sv-actual)' : 'white',stroke:'none'}}>{p.numbers.join('·')}</text><title>{p.numbers.join('·')}구 · {ft(p)}{at.out ? ' · 화면 밖' : ''}</title></g>; })}
  </svg>;
}

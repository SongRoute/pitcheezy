import { useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Bounds, Candidate } from './model';

/** Catcher's-view strike zone, the largest element of the game screen.
 *  Recommendation = hollow dashed SQUARE with its rank number; actual pitch = filled CIRCLE labelled '실제'.
 *  Shapes and words, not only colour, tell them apart. Mitt = dashed vertical line (x only, never height).
 *
 *  Geometry (integration rev3, after geometry-review-037): ONE linear transform feet → drawing units for every mark.
 *  The visible window is built from this batter's zone (half-width 0.83 ft, bounds.bottom..bounds.top) plus a fixed outside
 *  margin, so the whole zone always fits. A mark inside the window is drawn at its true position. A mark beyond the window
 *  is NOT moved into view as a normal mark: it becomes a hollow edge marker with an outward arrow, placed on the window
 *  edge (which is always outside the zone), and its accessible text gives the true coordinates. Labels can move; marks don't. */
export type Mark = { x: number; z: number };
export const PLATE_HALF = .83;
/** outside margins in feet around the zone (main view / mini cards) */
export const MARGIN = { main: { x: .42, z: .45 }, mini: { x: .34, z: .36 } };
const U = 100;  // drawing units per foot
const ux = (x: number) => x * U;              // x right (catcher's view)
const uz = (z: number) => -z * U;             // z up

export function zoneWindow(bounds: Bounds, mini: boolean) {
  const m = mini ? MARGIN.mini : MARGIN.main;
  const xMax = PLATE_HALF + m.x, zLo = bounds.bottom - m.z, zHi = bounds.top + m.z;
  // drawn area = window + a pad so an edge marker (and its arrow) is fully visible; the pad changes no coordinate or out-of-window rule
  const pad = mini ? .2 : .16;
  return { margin: m, xMin: -xMax, xMax, zLo, zHi, vb: { x: ux(-xMax - pad), y: uz(zHi + pad), w: ux(2 * (xMax + pad)), h: (zHi - zLo + 2 * pad) * U } };
}
/** Where a mark is drawn and whether it is beyond the window (then drawn as an edge marker, never a normal dot). */
export function place(p: Mark, w: ReturnType<typeof zoneWindow>) {
  const out = p.x < w.xMin || p.x > w.xMax || p.z < w.zLo || p.z > w.zHi;
  const x = Math.max(w.xMin, Math.min(w.xMax, p.x)), z = Math.max(w.zLo, Math.min(w.zHi, p.z));
  return { out, x: ux(x), y: uz(z), dx: Math.sign(p.x - x), dz: Math.sign(p.z - z) };
}
const ft = (p: Mark) => `가로 ${p.x.toFixed(2)}ft, 높이 ${p.z.toFixed(2)}ft`;

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

export default function Zone({ bounds, candidates, actual, actualLabel, setupX, mini = false }:
  { bounds: Bounds; candidates: Candidate[]; actual: Mark | null; actualLabel: string; setupX: number | null; mini?: boolean }) {
  const w = zoneWindow(bounds, mini);
  const left = ux(-PLATE_HALF), right = ux(PLATE_HALF), top = uz(bounds.top), bottom = uz(bounds.bottom);
  const groups = groupTargets(candidates).map(g => ({ ...g, at: place(g, w) }));
  const act = actual ? place(actual, w) : null;
  const parts = [...groups.map(g => `추천 ${g.ranks.join('·')}순위 위치 네모${g.at.out ? `(화면 밖 ${ft(g)})` : ''}`),
    actual ? `${actualLabel} 투구 동그라미${act!.out ? `(화면 밖, ${ft(actual)})` : ''}` : null].filter(Boolean);
  // Labels keep a readable on-screen size whatever the zone's drawn scale (SVG text otherwise scales with the drawing).
  const ref = useRef<SVGSVGElement>(null);
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return;
    const measure = () => { const r = el.getBoundingClientRect(); if (r.width && r.height) setScale(Math.min(r.width / w.vb.w, r.height / w.vb.h)); };
    measure(); const ro = new ResizeObserver(measure); ro.observe(el); return () => ro.disconnect();
  }, [w.vb.w, w.vb.h]);
  const u = (n: number) => n / scale;  // on-screen px → drawing units
  const px = (n: number) => `${u(n).toFixed(2)}px`;
  const half = mini ? 0 : u(13), rA = mini ? u(5.5) : u(8), rOut = mini ? u(4) : rA;
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
  const inside = (a: Box) => a.x >= w.vb.x && a.y >= w.vb.y && a.x + a.w <= w.vb.x + w.vb.w && a.y + a.h <= w.vb.y + w.vb.h;
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
  const recBoxes: Box[] = recs.flatMap(g => [{ x: g.d.x - g.hw, y: g.d.y - g.hh, w: 2 * g.hw, h: 2 * g.hh }, { x: g.tx - u(9), y: g.ty - u(14), w: u(18), h: u(16) }]);
  const actD = act ? inset(act, act.out ? rOut : rA) : null;
  const rD = act?.out ? rOut : rA;
  const dotBox: Box | null = actD ? { x: actD.x - rD - u(2), y: actD.y - rD - u(2), w: 2 * rD + u(4), h: 2 * rD + u(4) } : null;
  const mittText = mitt ? (mitt.out ? (mitt.dx > 0 ? '미트→' : '←미트') : '미트') : '';
  const mittW = u(mitt?.out ? 34 : 26), lh = u(16);
  const mittLabel = (() => {
    if (!mitt) return { x: 0, y: 0, box: null as Box | null };
    const x = Math.max(w.vb.x + mittW / 2, Math.min(w.vb.x + w.vb.w - mittW / 2, mitt.out ? mitt.x - mitt.dx * u(14) : mitt.x));
    for (const y of [top - u(6), bottom + u(16)]) { const box = { x: x - mittW / 2, y: y - lh + u(3), w: mittW, h: lh };
      if (inside(box) && !recBoxes.some(b => hit(b, box)) && !(dotBox && hit(dotBox, box))) return { x, y, box }; }
    return { x, y: top - u(6), box: { x: x - mittW / 2, y: top - u(6) - lh + u(3), w: mittW, h: lh } };
  })();
  const labW = u(30);
  const actLabel = (() => {
    if (!actD || mini) return null;
    const o = Math.max(rD, half) + u(4), so = Math.max(rD, ...recs.map(g => g.hw), u(9)) + u(4), cands = [[0, o + lh - u(2)], [0, -o - u(3)], [so + labW / 2, u(5)], [-so - labW / 2, u(5)], [so + labW / 2, -o], [-so - labW / 2, -o], [so + labW / 2, o + lh], [-so - labW / 2, o + lh], [0, 2 * o + lh], [0, -2 * o]];
    const avoid = [...recBoxes, ...(mittLabel.box ? [mittLabel.box] : [])];
    let first: { x: number; y: number } | null = null;
    for (const [dx, dy] of cands) { const x = actD.x + dx, y = actD.y + dy, box = { x: x - labW / 2, y: y - lh + u(3), w: labW, h: lh };
      if (!inside(box)) continue; first = first ?? { x, y }; if (!avoid.some(b => hit(b, box))) return { x, y, box }; }
    return first ? { ...first, box: { x: first.x - labW / 2, y: first.y - lh + u(3), w: labW, h: lh } } : null;
  })();
  const capBox: Box = { x: w.vb.x + u(3), y: w.vb.y + u(1), w: u(58), h: lh };
  const showCaption = ![...recBoxes, ...(dotBox ? [dotBox] : []), ...(mittLabel.box ? [mittLabel.box] : []), ...(actLabel ? [actLabel.box] : [])].some(b => hit(b, capBox));
  return <svg ref={ref} style={labels} className={`sv-zone${mini ? ' mini' : ''}`} viewBox={`${w.vb.x} ${w.vb.y} ${w.vb.w} ${w.vb.h}`} preserveAspectRatio="xMidYMid meet" role="img"
    aria-label={mini ? undefined : `포수 쪽에서 본 스트라이크존${parts.length ? ': ' + parts.join(', ') : ''}`} aria-hidden={mini || undefined}>
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
    {!mini && showCaption && <text x={capBox.x} y={capBox.y + capBox.h - u(2)} className="sv-zone-caption">포수 시점</text>}
  </svg>;
}

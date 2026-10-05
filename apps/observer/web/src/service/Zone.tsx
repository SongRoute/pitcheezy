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
  const pad = mini ? .1 : .16;
  return { xMin: -xMax, xMax, zLo, zHi, vb: { x: ux(-xMax - pad), y: uz(zHi + pad), w: ux(2 * (xMax + pad)), h: (zHi - zLo + 2 * pad) * U } };
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
  const half = mini ? 0 : u(13), rA = mini ? u(5.5) : u(8);
  const labels = { '--zl-s': px(13), '--zl-m': px(14), '--zl-l': px(15) } as CSSProperties;
  const arrow = (p: { x: number; y: number; dx: number; dz: number }, r: number) => {
    const ax = p.dx, ay = -p.dz, n = Math.hypot(ax, ay) || 1, L = r + u(7);
    return <path className="sv-out-arrow" d={`M${p.x + ax / n * r} ${p.y + ay / n * r} L${p.x + ax / n * L} ${p.y + ay / n * L}`} />;
  };
  const mitt = typeof setupX === 'number' ? place({ x: setupX, z: (bounds.top + bounds.bottom) / 2 }, w) : null;
  return <svg ref={ref} style={labels} className={`sv-zone${mini ? ' mini' : ''}`} viewBox={`${w.vb.x} ${w.vb.y} ${w.vb.w} ${w.vb.h}`} preserveAspectRatio="xMidYMid meet" role="img"
    aria-label={mini ? undefined : `포수 쪽에서 본 스트라이크존${parts.length ? ': ' + parts.join(', ') : ''}`} aria-hidden={mini || undefined}>
    <rect x={left} y={top} width={right - left} height={bottom - top} className="sv-zone-box" />
    {[1, 2].map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="sv-zone-grid" />)}
    {[1, 2].map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="sv-zone-grid" />)}
    {mitt && <g className={`sv-mitt${mitt.out ? ' out' : ''}`} data-x={setupX}>
      <line x1={mitt.x} x2={mitt.x} y1={top - u(4)} y2={bottom + u(4)} />
      <text x={mitt.out ? mitt.x - mitt.dx * u(14) : mitt.x} y={top - u(6)} textAnchor="middle">{mitt.out ? (mitt.dx > 0 ? '미트→' : '←미트') : '미트'}</text></g>}
    {/* actual is drawn first so recommendation numbers stay on top of it; its own label sits below the dot */}
    {act && <g className={`sv-actual${act.out ? ' out' : ''}`} data-out={act.out || undefined} data-x={actual!.x} data-z={actual!.z}>
      <circle cx={act.x} cy={act.y} r={rA} />{act.out && arrow(act, rA)}
      {!mini && <text x={act.x - act.dx * u(10)} y={act.y + Math.max(rA, groups.some(g => Math.abs(g.at.x - act.x) < 2 * half && Math.abs(g.at.y - act.y) < 2 * half) ? half : 0) + u(17)} textAnchor="middle">{actualLabel}</text>}</g>}
    {groups.map(g => { const hw = Math.max(half, u(5 + 4.5 * (g.ranks.join('·').length))); return <g key={g.ranks.join()} className={`sv-rec${g.at.out ? ' out' : ''}`} data-rank={g.ranks.join('·')} data-x={g.x} data-z={g.z} data-out={g.at.out || undefined}>
      <rect x={g.at.x - hw} y={g.at.y - half} width={2 * hw} height={2 * half} rx="3" />{g.at.out && arrow(g.at, half)}
      <text x={g.at.x} y={g.at.y + u(5.5)} textAnchor="middle">{g.ranks.join('·')}</text></g>; })}
    {!mini && <text x={w.vb.x + u(3)} y={w.vb.y + u(13)} className="sv-zone-caption">포수 시점</text>}
  </svg>;
}

import { useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Bounds, Candidate } from './model';

/** Catcher's-view strike zone, the largest element of the game screen.
 *  Recommendation = hollow dashed SQUARE with its rank number; actual pitch = filled CIRCLE labelled '실제';
 *  the pitch just thrown (when the stage shows the NEXT recommendation) = small grey circle labelled '직전'.
 *  Shapes and words, not only colour, tell them apart. Mitt = dashed vertical line (x only, never height). */
export type Mark = { x: number; z: number };
const W = 200, H = 220;
const sx = (v: number) => W / 2 + v / 1.4 * 94;            // ±1.4 ft across
const sz = (v: number) => 204 - (v - .9) / 3.4 * 190;       // 0.9–4.3 ft up
const clampX = (v: number) => Math.max(-1.3, Math.min(1.3, v));
const clampZ = (v: number) => Math.max(1.0, Math.min(4.2, v));

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
  const left = sx(-.83), right = sx(.83), top = sz(bounds.top), bottom = sz(bounds.bottom);
  const groups = groupTargets(candidates);
  const parts = [...groups.map(g => `추천 ${g.ranks.join('·')}순위 위치 네모`), actual ? `${actualLabel} 투구 동그라미` : null].filter(Boolean);
  // Labels keep a readable on-screen size whatever the zone's drawn scale (SVG text otherwise scales with the drawing).
  const ref = useRef<SVGSVGElement>(null);
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return;
    const measure = () => { const r = el.getBoundingClientRect(); if (r.width && r.height) setScale(Math.min(r.width / W, r.height / H)); };
    measure(); const ro = new ResizeObserver(measure); ro.observe(el); return () => ro.disconnect();
  }, []);
  const u = (n: number) => n / scale;  // on-screen px → drawing units
  const px = (n: number) => `${u(n).toFixed(2)}px`;
  // mini cards: a big dot that stays visible at card size; the full zone: ~13px squares
  const half = Math.max(12, u(13)), rA = mini ? Math.max(18, u(7)) : Math.max(8, u(8));
  const labels = { '--zl-s': px(13), '--zl-m': px(14), '--zl-l': px(15) } as CSSProperties;
  const ax = actual ? sx(clampX(actual.x)) : 0, az = actual ? sz(clampZ(actual.z)) : 0;
  return <svg ref={ref} style={labels} className={`sv-zone${mini ? ' mini' : ''}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img"
    aria-label={mini ? undefined : `포수 쪽에서 본 스트라이크존${parts.length ? ': ' + parts.join(', ') : ''}`} aria-hidden={mini || undefined}>
    <rect x={left} y={top} width={right - left} height={bottom - top} className="sv-zone-box" />
    {[1, 2].map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="sv-zone-grid" />)}
    {[1, 2].map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="sv-zone-grid" />)}
    <path d={`M${sx(-.71)} 210 h${sx(.71) - sx(-.71)} l-8 6 h${-(sx(.71) - sx(-.71) - 16)} z`} className="sv-plate" />
    {typeof setupX === 'number' && Math.abs(setupX) <= 1.3 && <g className="sv-mitt" data-x={setupX}>
      <line x1={sx(setupX)} x2={sx(setupX)} y1={top - 4} y2={bottom + 4} /><text x={sx(setupX)} y={top - 4 - u(4)} textAnchor="middle">미트</text></g>}
    {/* actual is drawn first so recommendation numbers stay on top of it; its own label goes below the dot */}
    {actual && <g className="sv-actual"><circle cx={ax} cy={az} r={rA} />
      {!mini && <text x={ax} y={az + Math.max(rA, groups.some(g => Math.abs(sx(g.x) - ax) < 2 * half && Math.abs(sz(g.z) - az) < 2 * half) ? half : 0) + u(17)} textAnchor="middle">{actualLabel}</text>}</g>}
    {groups.map(g => { const w = Math.max(half, u(5 + 4.5 * (g.ranks.join('·').length))); return <g key={g.ranks.join()} className="sv-rec" data-rank={g.ranks.join('·')}>
      <rect x={sx(g.x) - w} y={sz(g.z) - half} width={2 * w} height={2 * half} rx="3" />
      <text x={sx(g.x)} y={sz(g.z) + u(5.5)} textAnchor="middle">{g.ranks.join('·')}</text></g>; })}
    {!mini && <text x="4" y={2 + u(14)} className="sv-zone-caption">포수 시점</text>}
  </svg>;
}

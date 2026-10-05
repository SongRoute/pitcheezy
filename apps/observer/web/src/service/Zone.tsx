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

export default function Zone({ bounds, candidate, actual, actualLabel, previous, setupX }:
  { bounds: Bounds; candidate: Candidate | null; actual: Mark | null; actualLabel: string; previous: Mark | null; setupX: number | null }) {
  const left = sx(-.83), right = sx(.83), top = sz(bounds.top), bottom = sz(bounds.bottom);
  const t = candidate?.target ?? null;
  const parts = [t ? `추천 ${candidate!.rank}순위 위치 네모` : null, actual ? `${actualLabel} 투구 동그라미` : null, previous ? '직전 투구 회색 동그라미' : null].filter(Boolean);
  return <svg className="sv-zone" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img"
    aria-label={`포수 쪽에서 본 스트라이크존${parts.length ? ': ' + parts.join(', ') : ''}`}>
    <rect x={left} y={top} width={right - left} height={bottom - top} className="sv-zone-box" />
    {[1, 2].map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="sv-zone-grid" />)}
    {[1, 2].map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="sv-zone-grid" />)}
    <path d={`M${sx(-.71)} 210 h${sx(.71) - sx(-.71)} l-8 6 h${-(sx(.71) - sx(-.71) - 16)} z`} className="sv-plate" />
    {typeof setupX === 'number' && Math.abs(setupX) <= 1.3 && <g className="sv-mitt" data-x={setupX}>
      <line x1={sx(setupX)} x2={sx(setupX)} y1={top - 4} y2={bottom + 4} /><text x={sx(setupX)} y={top - 8} textAnchor="middle">미트</text></g>}
    {previous && <g className="sv-prev"><circle cx={sx(clampX(previous.x))} cy={sz(clampZ(previous.z))} r="6" />
      <text x={sx(clampX(previous.x))} y={sz(clampZ(previous.z)) + 17} textAnchor="middle">직전</text></g>}
    {t && <g className="sv-rec" data-rank={candidate!.rank}>
      <rect x={sx(t.x) - 12} y={sz(t.z) - 12} width="24" height="24" rx="3" />
      <text x={sx(t.x)} y={sz(t.z) + 5} textAnchor="middle">{candidate!.rank}</text></g>}
    {actual && <g className="sv-actual"><circle cx={sx(clampX(actual.x))} cy={sz(clampZ(actual.z))} r="8" />
      <text x={sx(clampX(actual.x))} y={sz(clampZ(actual.z)) - 12} textAnchor="middle">{actualLabel}</text></g>}
    <text x="4" y="12" className="sv-zone-caption">포수 시점</text>
  </svg>;
}

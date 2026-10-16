import type { Bounds, Candidate } from '../service/model';
import type { View, ZoneMark } from './options';
import { candidateCell, frame } from './zones';
import type { Cell } from './zones';

/** The v3 strike zone. Recommendations are CELLS (squares) of the 3×3 zone with their rank and pitch code; actual pitches are DOTS at
 *  their true position (never snapped or moved). The window around the zone is fixed; a pitch beyond it is drawn as a hollow edge
 *  marker with an outward tick, and its text alternative says so. Shape and words, not colour alone, tell the two apart.
 *  Marks (spec "투구 점의 투명도·크기·번호 표시"): A all numbered · B only the latest numbered · C small dots, numbers in a side column. */
export type ZonePitch = { n: number; x: number | null; z: number | null; latest: boolean };
type Props = { bounds: Bounds; view: View; mark: ZoneMark; batter: 'L' | 'R' | null; recs: Candidate[]; pitches: ZonePitch[];
  held?: boolean; label: string; size?: 'main' | 'detail' | 'mini'; showBatter?: boolean };

const sameCell = (a: Cell, b: Cell) => a.row === b.row && a.col === b.col;

export default function Zone({ bounds, view, mark, batter, recs, pitches, held = false, label, size = 'main', showBatter = true }: Props) {
  const mini = size === 'mini';
  const f = frame(bounds, view, mini ? { x: 0.42, z: 0.4 } : undefined);
  const gutter = mark === 'C' && !mini ? 46 : 0;
  // candidates grouped by cell: two ranks on one cell share one square ('1·3')
  const groups: { cell: Cell; ranks: number[]; codes: string[] }[] = [];
  for (const c of [...recs].sort((a, b) => a.rank - b.rank)) {
    const cell = candidateCell(c, batter, bounds); if (!cell) continue;
    const g = groups.find(o => sameCell(o.cell, cell));
    if (g) { g.ranks.push(c.rank); g.codes.push(c.pitch_type); } else groups.push({ cell, ranks: [c.rank], codes: [c.pitch_type] });
  }
  const vb = f.vb;
  const pad = 14;   // room for an edge marker's outward tick
  const clamp = (x: number, z: number) => {
    const cx = Math.max(-f.xMax, Math.min(f.xMax, x)), cz = Math.max(f.zLo, Math.min(f.zHi, z));
    return { out: cx !== x || cz !== z, X: f.X(cx), Y: f.Z(cz), dx: Math.sign(f.X(x) - f.X(cx)), dy: Math.sign(f.Z(z) - f.Z(cz)) };
  };
  const dots = pitches.filter(p => p.x !== null && p.z !== null).map(p => ({ ...p, at: clamp(p.x!, p.z!) }));
  // C: label column right of the window, one row per pitch in height order, at least 22 units apart
  const lab = mark === 'C' && !mini ? (() => {
    const sorted = [...dots].sort((a, b) => a.at.Y - b.at.Y);
    let prev = -Infinity; const lo = vb.y + 10, hi = vb.y + vb.h - 10;
    const out = sorted.map((d, i) => { let y = Math.max(d.at.Y, prev + 22, lo); y = Math.min(y, hi - (sorted.length - 1 - i) * 22); prev = y; return { d, y }; });
    return out;
  })() : [];
  const rDot = (latest: boolean) => mini ? (latest ? 7 : 5) : mark === 'A' ? (latest ? 13 : 10) : mark === 'B' ? (latest ? 13 : 6) : (latest ? 7.5 : 5.5);
  const numbered = (latest: boolean) => !mini && (mark === 'A' || (mark === 'B' && latest));
  const batterX = batter ? f.X(batter === 'R' ? -f.xMax : f.xMax) : 0;
  return <svg className={`v3-zone ${size}${held ? ' held' : ''}`} viewBox={`${vb.x - pad} ${vb.y - pad} ${vb.w + gutter + 2 * pad} ${vb.h + 2 * pad}`}
    preserveAspectRatio="xMidYMid meet" role="img" aria-label={label} data-mark={mark} data-view={view}>
    <rect className="v3-z-window" x={vb.x} y={vb.y} width={vb.w} height={vb.h} rx={mini ? 4 : 8} />
    <rect className="v3-z-zone" x={f.zone.x} y={f.zone.y} width={f.zone.w} height={f.zone.h} />
    {[1, 2].map(i => <line key={`v${i}`} className="v3-z-grid" x1={f.zone.x + f.zone.w * i / 3} x2={f.zone.x + f.zone.w * i / 3} y1={f.zone.y} y2={f.zone.y + f.zone.h} />)}
    {[1, 2].map(i => <line key={`h${i}`} className="v3-z-grid" y1={f.zone.y + f.zone.h * i / 3} y2={f.zone.y + f.zone.h * i / 3} x1={f.zone.x} x2={f.zone.x + f.zone.w} />)}
    {groups.map(g => { const r = f.cellRect(g.cell); const rank = g.ranks[0]; const inset = mini ? 2 : 4;
      return <g key={`${g.cell.row}${g.cell.col}`} className={`v3-z-rec r${rank} m${mark}`} data-rank={g.ranks.join('·')}>
        <rect x={r.x + inset} y={r.y + inset} width={r.w - 2 * inset} height={r.h - 2 * inset} rx={mini ? 2 : 5} />
        {!mini && <g className="v3-z-tag"><rect x={r.x + inset + 3} y={r.y + inset + 3} width={g.ranks.length > 1 ? 44 : 34} height={17} rx={8.5} />
          <text x={r.x + inset + 3 + (g.ranks.length > 1 ? 22 : 17)} y={r.y + inset + 15.5} textAnchor="middle">{g.ranks.length > 1 ? g.ranks.join('·') : `${g.ranks[0]} ${g.codes[0]}`}</text></g>}
      </g>; })}
    <g className="v3-z-dots">{dots.map(d => { const r = rDot(d.latest); const hollow = d.at.out || (mark === 'B' && !d.latest && !mini);
      return <g key={d.n} className={`v3-z-dot${d.latest ? ' latest' : ' past'}${d.at.out ? ' out' : ''}${hollow ? ' hollow' : ''}`} data-n={d.n}>
        {d.latest && !mini && <circle className="v3-z-ring" cx={d.at.X} cy={d.at.Y} r={r + 3} />}
        <circle cx={d.at.X} cy={d.at.Y} r={r} />
        {d.at.out && <line className="v3-z-tick" x1={d.at.X} y1={d.at.Y} x2={d.at.X + d.at.dx * (r + 7)} y2={d.at.Y + d.at.dy * (r + 7)} />}
        {numbered(d.latest) && <text x={d.at.X} y={d.at.Y + (d.latest ? 4.6 : 3.8)} textAnchor="middle">{d.n}</text>}
      </g>; })}</g>
    {lab.map(({ d, y }) => <g key={`l${d.n}`} className={`v3-z-lab${d.latest ? ' latest' : ''}`}>
      <polyline points={`${d.at.X},${d.at.Y} ${vb.x + vb.w - 4},${y} ${vb.x + vb.w + 8},${y}`} />
      <rect x={vb.x + vb.w + 8} y={y - 9} width={30} height={18} rx={9} /><text x={vb.x + vb.w + 23} y={y + 4.5} textAnchor="middle">{d.n}</text></g>)}
    {showBatter && batter && !mini && <text className="v3-z-batter" x={batterX + (batter === 'R' ? 1 : -1) * f.sx * 8} y={f.zone.y - 8} textAnchor={(batter === 'R') === (f.sx > 0) ? 'start' : 'end'}>{batter === 'R' ? '우타자' : '좌타자'}</text>}
  </svg>;
}

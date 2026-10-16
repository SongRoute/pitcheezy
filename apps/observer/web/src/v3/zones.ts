/** Zone cells and words for the v3 screens.
 *  Data are always catcher-view feet (Statcast plate_x: + = catcher's right) and catcher-view zone ids
 *  ('{low|middle|high}_{left|middle|right}', domain.py). A batter-relative id ('low_away', 'high_in') is also read: it needs the
 *  batter's side and is left unplaced when the side is unknown. The view option only mirrors the drawing. */
import type { Bounds, Candidate } from '../service/model';
import type { View } from './options';

export const PLATE_HALF = 0.83;
export type Cell = { row: 0 | 1 | 2; col: 0 | 1 | 2 };   // row 0 = low, col 0 = catcher's left
const ROWS: Record<string, 0 | 1 | 2> = { low: 0, middle: 1, mid: 1, high: 2 };

/** catcher's-left column for a batter-relative word: a right-handed batter stands on the catcher's left (3B side), so 'in' = left */
function sideCol(word: string, batter: 'L' | 'R' | null): 0 | 1 | 2 | null {
  if (word === 'left') return 0; if (word === 'middle' || word === 'center') return 1; if (word === 'right') return 2;
  if (!batter) return null;
  if (word === 'in' || word === 'inside') return batter === 'R' ? 0 : 2;
  if (word === 'away' || word === 'outside') return batter === 'R' ? 2 : 0;
  return null;
}
export function cellOf(zoneId: string | null | undefined, batter: 'L' | 'R' | null): Cell | null {
  const m = /^(low|middle|mid|high)_([a-z]+)$/.exec(zoneId ?? '');
  if (!m) return null;
  const col = sideCol(m[2], batter);
  return col === null ? null : { row: ROWS[m[1]], col };
}
/** the cell a point falls in, or null outside the zone */
export function cellAt(x: number, z: number, b: Bounds): Cell | null {
  if (Math.abs(x) > PLATE_HALF || z < b.bottom || z > b.top) return null;
  const col = Math.min(2, Math.max(0, Math.floor((x + PLATE_HALF) / (2 * PLATE_HALF) * 3))) as 0 | 1 | 2;
  const row = Math.min(2, Math.max(0, Math.floor((z - b.bottom) / (b.top - b.bottom) * 3))) as 0 | 1 | 2;
  return { row, col };
}
/** batter-relative words for a column; '포수 기준' words when the batter's side is unknown */
function colWord(col: 0 | 1 | 2, batter: 'L' | 'R' | null) {
  if (col === 1) return '가운데';
  if (!batter) return col === 0 ? '포수 왼쪽' : '포수 오른쪽';
  const inside = (batter === 'R') === (col === 0);
  return inside ? '몸쪽' : '바깥쪽';
}
const ROW_WORD = ['낮게', '가운데', '높게'];
export function cellWords(c: Cell, batter: 'L' | 'R' | null) {
  if (c.row === 1 && c.col === 1) return '한가운데';
  if (c.col === 1) return `가운데 ${ROW_WORD[c.row]}`;
  return c.row === 1 ? colWord(c.col, batter) : `${colWord(c.col, batter)} ${ROW_WORD[c.row]}`;
}
/** words for where a pitch crossed: a cell, or which way it missed the zone */
export function pointWords(x: number | null, z: number | null, b: Bounds, batter: 'L' | 'R' | null): string {
  if (x === null || z === null) return '위치 정보 없음';
  const c = cellAt(x, z, b);
  if (c) return cellWords(c, batter);
  const high = z > b.top, low = z < b.bottom, wide = Math.abs(x) > PLATE_HALF;
  const col = x < -PLATE_HALF ? 0 : x > PLATE_HALF ? 2 : 1;
  const parts = [wide ? colWord(col as 0 | 2, batter) : '', high ? '높게' : low ? '낮게' : ''].filter(Boolean);
  return `${parts.join(' ')} · 존 밖`;
}
/** a candidate's place: its cell when the id is readable, else the cell its target point falls in */
export function candidateCell(c: Candidate, batter: 'L' | 'R' | null, b: Bounds): Cell | null {
  return cellOf(c.zone_id, batter) ?? (c.target ? cellAt(c.target.x, c.target.z, b) : null);
}
export function candidateWords(c: Candidate, batter: 'L' | 'R' | null, b: Bounds): string {
  const cell = candidateCell(c, batter, b);
  return cell ? cellWords(cell, batter) : c.zone_label || '위치 없음';
}

/** Drawing geometry: feet → drawing units. The window = zone + a fixed outside margin (never moves with the data). */
export const U = 100;
export function frame(b: Bounds, view: View, margin = { x: 0.6, z: 0.45 }) {
  const sx = view === 'pitcher' ? -1 : 1;   // pitcher's view mirrors x only
  const xMax = PLATE_HALF + margin.x, zLo = b.bottom - margin.z, zHi = b.top + margin.z;
  const X = (x: number) => sx * x * U, Z = (z: number) => -z * U;
  const cellRect = (c: Cell) => {
    const w = 2 * PLATE_HALF / 3, h = (b.top - b.bottom) / 3;
    const x0 = -PLATE_HALF + c.col * w, x1 = x0 + w, z0 = b.bottom + c.row * h, z1 = z0 + h;
    const a = X(x0), bb = X(x1);
    return { x: Math.min(a, bb), y: Z(z1), w: Math.abs(bb - a), h: (z1 - z0) * U };
  };
  return { sx, xMax, zLo, zHi, X, Z, cellRect,
    vb: { x: -xMax * U, y: Z(zHi), w: 2 * xMax * U, h: (zHi - zLo) * U },
    zone: { x: -PLATE_HALF * U, y: Z(b.top), w: 2 * PLATE_HALF * U, h: (b.top - b.bottom) * U } };
}

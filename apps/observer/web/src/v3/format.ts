import { useSyncExternalStore } from 'react';
import type { Actual, PitchRow } from '../service/model';
import { detailResult } from '../service/model';

/** Speed unit: the same per-viewer key as the classic screen, so a viewer's choice carries over. */
const UNIT_KEY = 'pitcheezy.unit';
let unit: 'kmh' | 'mph' = (() => { try { return localStorage.getItem(UNIT_KEY) === 'mph' ? 'mph' : 'kmh'; } catch { return 'kmh'; } })();
const subs = new Set<() => void>();
export const useUnit = () => useSyncExternalStore(l => { subs.add(l); return () => subs.delete(l); }, () => unit, () => unit);
export function toggleUnit() { unit = unit === 'kmh' ? 'mph' : 'kmh'; try { localStorage.setItem(UNIT_KEY, unit); } catch { /* ignore */ } subs.forEach(l => l()); }
export const speedNum = (mph: number | null | undefined, u: 'kmh' | 'mph') => typeof mph === 'number' ? (u === 'kmh' ? String(Math.round(mph * 1.609344)) : mph.toFixed(1)) : '—';
export const unitText = (u: 'kmh' | 'mph') => u === 'kmh' ? 'km/h' : 'mph';

export const handText = (h: string | null | undefined, role: 'p' | 'b') => h === 'R' ? (role === 'p' ? '우투' : '우타') : h === 'L' ? (role === 'p' ? '좌투' : '좌타') : h === 'S' ? '양타' : '';
export const dash = (v: number | null | undefined) => typeof v === 'number' ? String(v) : '—';
export const stat3 = (v: number | null | undefined) => typeof v === 'number' ? v.toFixed(3).replace(/^0/, '') : '—';
export const era = (v: number | null | undefined) => typeof v === 'number' ? v.toFixed(2) : '—';
export const shareText = (p: number | null | undefined) => typeof p === 'number' ? `${Math.round(p * 100)}%` : '—';

/** The call of one pitch in a few words: 볼 / 스트라이크 / 헛스윙 / 파울 / 인플레이 result. Unconfirmed → 확인 중. */
export function callText(p: PitchRow): string {
  if (p.receipt?.status === 'unconfirmed') return '확인 중';
  const a = p.actual; if (!a) return '기록 없음';
  if (a.event) return detailResult(a);
  return a.result_label || '기록 없음';
}
/** colour family of a call, for the small tag beside it (words always carry the meaning) */
export function callKind(a: Actual | null | undefined): 'ball' | 'strike' | 'foul' | 'play' | 'none' {
  const d = a?.description ?? '';
  if (!a) return 'none';
  if (a.event) return 'play';
  if (/ball/.test(d) || a.result_label === '볼') return 'ball';
  if (/foul/.test(d) || /파울/.test(a.result_label)) return 'foul';
  if (/strike|swinging|called/.test(d) || /스트라이크|헛스윙/.test(a.result_label)) return 'strike';
  return 'none';
}
export const ordinal = (n: number) => `${n}구`;

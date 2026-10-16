/** v3 screen options: the variants under comparison (2026-10 UI spec, "시안을 보고 결정할 것").
 *  A public bundle takes them from build-time env (`.env.v3`: VITE_V3_*) and never from the page URL.
 *  A review build (`__PZ_REVIEW__`) also reads `?zone=&list=&stats=&density=&view=` and the in-page switcher, and remembers the
 *  viewer's last choice in localStorage (a per-viewer convenience only). */
import { useSyncExternalStore } from 'react';

export type ZoneMark = 'A' | 'B' | 'C';          // A 모두 번호 / B 직전만 번호 / C 번호는 오른쪽 라벨
export type ListLayout = 'row' | 'stack' | 'side'; // 이번 타석 목록: 가로 한 줄 / 여러 줄 / 존 옆
export type StatsLayout = 'table' | 'card';       // 선수 기록: 표 / 나란한 카드
export type Density = 'full' | 'lite';            // 메인 정보량: 명세 기본 / 필수만
export type View = 'catcher' | 'pitcher';         // 존을 보는 쪽
export type Options = { zone: ZoneMark; list: ListLayout; stats: StatsLayout; density: Density; view: View };

const CHOICES: { [K in keyof Options]: readonly Options[K][] } = {
  zone: ['A', 'B', 'C'], list: ['row', 'stack', 'side'], stats: ['table', 'card'], density: ['full', 'lite'], view: ['catcher', 'pitcher'],
};
export const OPTION_LABELS: { [K in keyof Options]: { title: string; values: Record<Options[K], string> } } = {
  zone: { title: '존 표현', values: { A: 'A 모두 번호', B: 'B 직전만 번호', C: 'C 오른쪽 라벨' } },
  list: { title: '투구 목록', values: { row: '가로 한 줄', stack: '여러 줄', side: '존 옆' } },
  stats: { title: '선수 기록', values: { table: '표', card: '카드' } },
  density: { title: '정보량', values: { full: '기본', lite: '라이트' } },
  view: { title: '존 시점', values: { catcher: '포수 쪽', pitcher: '투수 쪽' } },
};
const env = import.meta.env;
const pick = <K extends keyof Options>(k: K, v: unknown, fallback: Options[K]): Options[K] =>
  (CHOICES[k] as readonly unknown[]).includes(v) ? v as Options[K] : fallback;
/** Build defaults. The fallbacks are the values written in `.env.v3`. */
export const DEFAULTS: Options = {
  zone: pick('zone', env.VITE_V3_ZONE, 'B'),
  list: pick('list', env.VITE_V3_LIST, 'row'),
  stats: pick('stats', env.VITE_V3_STATS, 'table'),
  density: pick('density', env.VITE_V3_DENSITY, 'full'),
  view: pick('view', env.VITE_V3_VIEW, 'catcher'),
};
const KEY = 'pitcheezy.v3.options';
function initial(): Options {
  if (!__PZ_REVIEW__) return DEFAULTS;
  let saved: Partial<Options> = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) ?? '{}') ?? {}; } catch { saved = {}; }
  const q = new URLSearchParams(window.location.search);
  const out = { ...DEFAULTS };
  for (const k of Object.keys(CHOICES) as (keyof Options)[]) {
    const v = q.get(k) ?? saved[k];
    (out as Record<string, unknown>)[k] = pick(k, v, DEFAULTS[k]);
  }
  return out;
}
let current: Options = initial();
const listeners = new Set<() => void>();
export function setOption<K extends keyof Options>(k: K, v: Options[K]) {
  if (!__PZ_REVIEW__) return;
  current = { ...current, [k]: v };
  try { localStorage.setItem(KEY, JSON.stringify(current)); } catch { /* per-viewer convenience only */ }
  listeners.forEach(l => l());
}
export const optionChoices = <K extends keyof Options>(k: K) => CHOICES[k];
export function useOptions(): Options {
  return useSyncExternalStore(l => { listeners.add(l); return () => listeners.delete(l); }, () => current, () => current);
}

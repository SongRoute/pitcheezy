import { publicGameHref, publicListHref } from '../service/publish';

/** Links as in the classic screens: public bundle = query form next to index.html; API mode = /watch routes with the demo binding. */
const demo = () => __PZ_STATIC__ ? null : new URLSearchParams(window.location.search).get('demo');
/** Preview builds are also opened inside hosts that pass no query string (only a plain #anchor): they route by `#game-<pk>` / `#list`. */
export const listHref = (date?: string | null) => {
  if (__PZ_REPLAY__) return '#list';
  if (__PZ_STATIC__) return publicListHref(date);
  const q = new URLSearchParams(); if (date) q.set('date', date); const d = demo(); if (d) q.set('demo', d);
  const s = q.toString(); return `/watch${s ? `?${s}` : ''}`;
};
export const gameHref = (pk: number) => {
  if (__PZ_REPLAY__) return `#game-${pk}`;
  if (__PZ_STATIC__) return publicGameHref(pk);
  const d = demo(); return `/watch/${pk}${d ? `?demo=${encodeURIComponent(d)}` : ''}`;
};

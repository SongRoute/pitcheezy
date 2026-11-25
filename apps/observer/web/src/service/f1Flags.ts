/** F1 screen switches in one place (PM 11/22: "항목별로 켜고 끌 수 있게"). Build-time default from VITE_PZ_*; a demo URL can override
 *  with ?pzf=name:on,name:off (only these names; anything else is ignored). Defaults follow the ★ picks and owner's 11/22 answers. */
const env = (k: string): string => (((import.meta.env ?? {}) as Record<string, string | undefined>)[k] ?? '').trim();
const DEFAULTS = {
  /** D4-① / D5-① 주요 장면 분석: one line under the result card for the top-10 plate appearances, its sheet, and the top-10 list */
  scenes: env('VITE_PZ_SCENES') !== 'off',
  /** D6 (a)②: when an outside recommendation square is redrawn over the newest pitch, the newest pitch is drawn over it again */
  latestOver: env('VITE_PZ_LATEST_OVER') !== 'off',
  /** D7 8-3 ②: "던지기 전 추천 1순위 … · 실제 …" under the result card (not picked → off) */
  recVsActual: env('VITE_PZ_REC_VS_ACTUAL') === 'on',
  /** owner 11/22 plan 1: bottom tabs 추천 · 경기 · 선수 · 분석 (off = the earlier 추천 · 선수 · 구종 · 기록 sheets) */
  tabs: env('VITE_PZ_TABS') !== 'off',
  /** owner 11/22 (7): 라이브 멈춤 + 지난 공을 누르면 예측 탭에 그때 예측 (off = 지난 공은 시트, 라이브 표시는 설명만) */
  pause: env('VITE_PZ_PAUSE') !== 'off',
  /** booth monitor (≥1600 px): two columns (left game·players·zone, right 추천·이번 타석) instead of one zoomed column.
   *  On by default since 11/22 (consultant, delegated; owner picks from captures). VITE_PZ_BOOTH_COLS=off or ?pzf=boothCols:off = one column */
  boothCols: env('VITE_PZ_BOOTH_COLS') !== 'off',
  /** 주요 장면 시트의 "그 밖" 줄 (타석 시작 상태 차이) */
  sceneOther: env('VITE_PZ_SCENE_OTHER') !== 'off',
  /** failure drill B-5 (11/22): when the publisher has written nothing for QUIET_S seconds (before the 30 s 갱신 지연), the 라이브 chip
   *  becomes a quiet "잠시 기다리는 중" so a frozen last pitch is not presented as live. Final games are never affected. */
  quietWait: env('VITE_PZ_QUIET_WAIT') !== 'off',
};
export type F1Flag = keyof typeof DEFAULTS;
function fromUrl(): Partial<Record<F1Flag, boolean>> {
  try {
    const raw = new URLSearchParams(window.location.search).get('pzf'); if (!raw) return {};
    const o: Partial<Record<F1Flag, boolean>> = {};
    for (const part of raw.split(',')) { const [k, v] = part.split(':'); if (k in DEFAULTS && (v === 'on' || v === 'off')) o[k as F1Flag] = v === 'on'; }
    return o;
  } catch { return {}; }
}
export const F1: Record<F1Flag, boolean> = { ...DEFAULTS, ...fromUrl() };
/** seconds without any new publish (health or game file) before the quiet wait chip; build env VITE_PZ_QUIET_S */
export const QUIET_S = (() => { const n = Number(env('VITE_PZ_QUIET_S')); return Number.isFinite(n) && n >= 5 ? n : 15; })();

/** Booth monitor layouts (owner 11/22: "세가지 모두 준비"). Landscape ≥1600×900:
 *   l1 = two columns, one of them the strike zone top to bottom; l2 = three columns, the zone in the middle;
 *   l3 = the phone column, larger (the rotated 1080×1920 monitor always uses it); one = the earlier one zoomed column.
 *  wp = the win-probability card on the booth screen (owner decides where; ★ default places below).
 *  Build default VITE_PZ_BOOTH_LAYOUT (l1) / VITE_PZ_BOOTH_WP (on since owner 11/22); demo URL ?pzb=l2,wp or ?pzb=l1,wp:off.
 *  owner 11/22: prepare both L1 (landscape monitor) and L3 (rotated monitor) — the monitor shape is not known in advance; big text off.
 *  Columns can be rearranged from the URL for owner's review: ?pzc=zone|game.mu.rec.tl+wp — '|' between columns, '.' top to bottom,
 *  '+' side by side in one row (cards: game mu zone rec tl wp). */
export type BoothLayout = 'l1' | 'l2' | 'l3' | 'one';
export type BoothCard = 'game' | 'mu' | 'zone' | 'rec' | 'tl' | 'wp';
const LAYOUTS: BoothLayout[] = ['l1', 'l2', 'l3', 'one'];
const CARDS: BoothCard[] = ['game', 'mu', 'zone', 'rec', 'tl', 'wp'];
/** ★ first drawings (owner edits them in the screen session) */
export const BOOTH_COLS: Record<'l1' | 'l2', string[][]> = {
  l1: [['zone'], ['game', 'mu', 'rec', 'tl+wp']],
  l2: [['game', 'mu', 'rec'], ['zone'], ['tl', 'wp']],  // review 11/22: no empty left column when the win-probability card is off
};
function boothFrom(): { layout: BoothLayout; wp: boolean; big: boolean; cols: string[][] | null } {
  let layout: BoothLayout = (LAYOUTS as string[]).includes(env('VITE_PZ_BOOTH_LAYOUT')) ? env('VITE_PZ_BOOTH_LAYOUT') as BoothLayout : 'l1';
  let wp = env('VITE_PZ_BOOTH_WP') !== 'off';  // owner 11/22: win probability on in L1 and L3 (VITE_PZ_BOOTH_WP=off or ?pzb=wp:off removes it)
  /** big: key words read from 2 m (score, recommended pitch names, their %) drawn at about 45 monitor px (PM 11/22) */
  let big = env('VITE_PZ_BOOTH_BIG') === 'on';
  let cols: string[][] | null = null;
  if (!F1.boothCols) layout = 'one';
  try {
    const q = new URLSearchParams(window.location.search);
    for (const part of (q.get('pzb') ?? '').split(',')) {
      if ((LAYOUTS as string[]).includes(part)) layout = part as BoothLayout;
      else if (part === 'wp' || part === 'wp:on') wp = true; else if (part === 'wp:off') wp = false;
      else if (part === 'big' || part === 'big:on') big = true; else if (part === 'big:off') big = false;
    }
    const c = q.get('pzc');
    if (c) {
      const parsed = c.split('|').map(col => col.split('.').map(it => it.split('+').filter(k => (CARDS as string[]).includes(k)).join('+')).filter(Boolean)).filter(col => col.length);
      const flat = parsed.flat().flatMap(it => it.split('+'));
      if (parsed.length >= 2 && parsed.length <= 3 && flat.includes('zone') && new Set(flat).size === flat.length) cols = parsed;
    }
  } catch { /* ignore */ }
  return { layout, wp, big, cols };
}
export const BOOTH = boothFrom();
/** columns for the active landscape layout: each column a list of rows, each row one card or cards side by side ('tl+wp').
 *  Any card left out is added to the last column without the zone, so nothing disappears; wp is dropped when the card is off. */
export function boothCols(layout: BoothLayout, wp: boolean): BoothCard[][][] | null {
  if (layout !== 'l1' && layout !== 'l2') return null;
  const base = (BOOTH.cols ?? BOOTH_COLS[layout]).map(c => c.map(it => it.split('+') as BoothCard[]));
  const have = new Set(base.flat(2));
  const spare = [...base].reverse().find(c => !c.flat().includes('zone')) ?? base[base.length - 1];
  for (const k of CARDS) if (!have.has(k)) spare.push([k]);
  return base.map(c => c.map(r => r.filter(k => k !== 'wp' || wp)).filter(r => r.length));
}

/** Professor demo (PM 11/22): ?pzdemo=late reads a frozen late-game snapshot (pz/demo-late/v1/…, all ten key scenes already played)
 *  instead of the live files, and the header says "데모 보기" so it is never mistaken for the booth replay. Only names listed here. */
export const DEMO: string | null = (() => { try { const v = new URLSearchParams(window.location.search).get('pzdemo'); return v === 'late' ? v : null; } catch { return null; } })();

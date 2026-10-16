/** World Series strip on the list (spec: "월드시리즈만 서비스 — 1차전부터 나열, 시리즈 스코어, 진행 중·다음 경기 고정").
 *  Off unless the build names the two teams AND the series' first date:
 *    VITE_V3_SERIES_TEAMS="LAD,NYY"   (abbreviations as in the published game cards)
 *    VITE_V3_SERIES_FROM=2026-10-24   (KST date of game 1; earlier meetings of the same teams are not part of the series)
 *  Everything is computed from published files only. The strip is all-or-nothing: it is drawn only from a read in which EVERY
 *  published date from FROM on was read successfully, and a later failed read keeps the last complete result (never a partial
 *  score, never an invented 0–0). Like the classic list, nothing is read while the tab is hidden. */
import { useEffect, useState } from 'react';
import { fetchPublishedDates, fetchPublishedGames } from '../service/api';
import type { GameCard } from '../service/model';
import { isCalendarDate } from '../service/publish';
import { kstTime, short } from '../service/model';
import { gameHref } from './links';

const env = import.meta.env;
const TEAMS = String(env.VITE_V3_SERIES_TEAMS ?? '').split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
const FROM = String(env.VITE_V3_SERIES_FROM ?? '').trim();
export const SERIES_ON = __PZ_STATIC__ && TEAMS.length === 2 && isCalendarDate(FROM);
const TITLE = String(env.VITE_V3_SERIES_TITLE ?? '월드시리즈');
const MAX_GAMES = 7;
const READ_MS = 60000;   // a series changes once per game: read slowly (each read asks every published date's list)
const abbr = (g: GameCard, side: 'away' | 'home') => (side === 'away' ? g.away_abbr : g.home_abbr)?.toUpperCase() ?? '';
const isSeries = (g: GameCard) => { const t = [abbr(g, 'away'), abbr(g, 'home')]; return TEAMS.every(x => t.includes(x)); };
const md = (ymd: string) => { const [, m, d] = ymd.split('-').map(Number); return `${m}/${d}`; };

/** One complete read, or null when any date failed (or the result cannot be a series). */
async function readSeries(signal: AbortSignal): Promise<GameCard[] | null> {
  const { dates } = await fetchPublishedDates(signal);
  const out: GameCard[] = [];
  for (const d of [...new Set(dates)].filter(x => x >= FROM).sort()) {   // one date at a time: never a burst of requests
    if (signal.aborted || document.hidden) return null;
    out.push(...(await fetchPublishedGames(d, signal)).filter(isSeries));   // a failed date throws: the whole read is dropped
  }
  const seen = new Set<number>();
  // a postponed card is dropped before de-duplication, so the same game_pk on its new date is the one kept
  const list = out.filter(g => g.status !== 'postponed').filter(g => !seen.has(g.game_pk) && seen.add(g.game_pk) !== undefined)
    .sort((a, b) => (a.start ?? '').localeCompare(b.start ?? '') || a.game_pk - b.game_pk);
  return list.length > MAX_GAMES ? null : list;   // more than seven meetings: not one series → draw nothing
}

export default function Series() {
  const [games, setGames] = useState<GameCard[] | null>(null);
  useEffect(() => {
    if (!SERIES_ON) return;
    let alive = true, timer = 0, busy: AbortController | null = null;
    const run = async () => {
      window.clearTimeout(timer);
      if (!alive || busy || document.hidden) return;
      const ctrl = busy = new AbortController();
      try { const list = await readSeries(ctrl.signal); if (alive && list) setGames(list); } catch { /* keep the last complete result */ }
      if (busy === ctrl) busy = null;
      if (alive) timer = window.setTimeout(() => void run(), READ_MS);
    };
    const wake = () => { if (!document.hidden) void run(); };
    void run();
    document.addEventListener('visibilitychange', wake);
    return () => { alive = false; busy?.abort(); window.clearTimeout(timer); document.removeEventListener('visibilitychange', wake); };
  }, []);
  if (!SERIES_ON || !games || games.length === 0) return null;
  const wins = (t: string) => games.filter(g => g.status === 'final' && g.away_score !== null && g.home_score !== null
    && ((abbr(g, 'away') === t && g.away_score > g.home_score) || (abbr(g, 'home') === t && g.home_score > g.away_score))).length;
  const [a, b] = TEAMS;
  const slots = Array.from({ length: MAX_GAMES }, (_, i) => games[i] ?? null);
  return <section className="v3-series" aria-label={`${TITLE} ${a} ${wins(a)} 대 ${wins(b)} ${b}`}>
    <div className="v3-series-head"><h2>{TITLE}</h2><p aria-hidden="true"><span>{a}</span><b>{wins(a)}</b><i>–</i><b>{wins(b)}</b><span>{b}</span></p></div>
    <ol className="v3-series-pips" aria-hidden="true">{slots.map((g, i) => {
      const winner = g && g.status === 'final' && g.away_score !== null && g.home_score !== null ? (g.away_score > g.home_score ? abbr(g, 'away') : abbr(g, 'home')) : null;
      return <li key={i} className={g ? g.status : 'none'}><small>{i + 1}</small><b>{winner ?? (g?.status === 'live' ? 'LIVE' : '')}</b></li>; })}</ol>
    <ol className="v3-series-games">{games.map((g, i) => <li key={g.game_pk}><a href={gameHref(g.game_pk)} className={g.status}>
      <b>{i + 1}차전</b><span>{md(g.kst_date)} {g.status === 'scheduled' ? kstTime(g.start) : ''}</span>
      <span className="v3-series-score">{g.away_abbr || short(g.away_team)} {g.away_score ?? ''}{g.away_score !== null ? ' : ' : ' vs '}{g.home_score ?? ''} {g.home_abbr || short(g.home_team)}</span>
      <em>{g.status === 'live' ? '진행 중' : g.status === 'final' ? '종료' : g.status === 'scheduled' ? '예정' : g.status_text}</em></a></li>)}</ol>
    <p className="v3-muted small">{md(FROM)}부터 발행된 결과로 센 시리즈 기록이에요.{games.length < 4 ? ' 남은 경기는 일정이 올라오면 보여 드려요.' : ''}</p>
  </section>;
}

/** World Series strip on the list (spec: "월드시리즈만 서비스 — 1차전부터 나열, 시리즈 스코어, 진행 중·다음 경기 고정").
 *  Off unless the build names the two teams: VITE_V3_SERIES_TEAMS="LAD,NYY" (abbreviations as in the published game cards).
 *  Everything is computed from published files only: every published date (dates.json + manifest live_dates) is read, the games of
 *  those two teams are numbered by start time, and the series score counts finished games by their final score. Read again slowly. */
import { useEffect, useState } from 'react';
import { fetchPublishedDates, fetchPublishedGames } from '../service/api';
import type { GameCard } from '../service/model';
import { kstTime, short } from '../service/model';
import { gameHref } from './links';

const env = import.meta.env;
const TEAMS = String(env.VITE_V3_SERIES_TEAMS ?? '').split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
export const SERIES_ON = __PZ_STATIC__ && TEAMS.length === 2;
const TITLE = String(env.VITE_V3_SERIES_TITLE ?? '월드시리즈');
const READ_MS = 60000;   // a series changes once per game: read slowly (each read asks every published date's list)
const abbr = (g: GameCard, side: 'away' | 'home') => (side === 'away' ? g.away_abbr : g.home_abbr)?.toUpperCase() ?? '';
const isSeries = (g: GameCard) => { const t = [abbr(g, 'away'), abbr(g, 'home')]; return TEAMS.every(x => t.includes(x)); };
const md = (ymd: string) => { const [, m, d] = ymd.split('-').map(Number); return `${m}/${d}`; };

export default function Series() {
  const [games, setGames] = useState<GameCard[] | null>(null);
  useEffect(() => {
    if (!SERIES_ON) return;
    let alive = true, timer = 0;
    const read = async () => {
      const ctrl = new AbortController();
      try {
        const { dates } = await fetchPublishedDates(ctrl.signal);
        const out: GameCard[] = [];
        for (const d of [...dates].sort()) {   // one date at a time: never a burst of requests
          try { out.push(...(await fetchPublishedGames(d, ctrl.signal)).filter(isSeries)); } catch { /* a missing date leaves a gap, not an error */ }
        }
        const seen = new Set<number>();
        if (alive) setGames(out.filter(g => !seen.has(g.game_pk) && seen.add(g.game_pk) !== undefined).sort((a, b) => (a.start ?? '').localeCompare(b.start ?? '') || a.game_pk - b.game_pk));
      } catch { if (alive) setGames(g => g ?? []); }
      if (alive) timer = window.setTimeout(read, READ_MS);
    };
    void read();
    return () => { alive = false; window.clearTimeout(timer); };
  }, []);
  if (!SERIES_ON || !games) return null;
  const wins = (t: string) => games.filter(g => g.status === 'final' && g.away_score !== null && g.home_score !== null
    && ((abbr(g, 'away') === t && g.away_score > g.home_score) || (abbr(g, 'home') === t && g.home_score > g.away_score))).length;
  const [a, b] = TEAMS;
  const slots = Array.from({ length: 7 }, (_, i) => games[i] ?? null);
  return <section className="v3-series" aria-label={`${TITLE} ${a} ${wins(a)} 대 ${wins(b)} ${b}`}>
    <div className="v3-series-head"><h2>{TITLE}</h2><p aria-hidden="true"><span>{a}</span><b>{wins(a)}</b><i>–</i><b>{wins(b)}</b><span>{b}</span></p></div>
    <ol className="v3-series-pips" aria-hidden="true">{slots.map((g, i) => {
      const winner = g && g.status === 'final' && g.away_score !== null && g.home_score !== null ? (g.away_score > g.home_score ? abbr(g, 'away') : abbr(g, 'home')) : null;
      return <li key={i} className={g ? g.status : 'none'}><small>{i + 1}</small><b>{winner ?? (g?.status === 'live' ? 'LIVE' : '')}</b></li>; })}</ol>
    <ol className="v3-series-games">{games.map((g, i) => <li key={g.game_pk}><a href={gameHref(g.game_pk)} className={g.status}>
      <b>{i + 1}차전</b><span>{md(g.kst_date)} {g.status === 'scheduled' ? kstTime(g.start) : ''}</span>
      <span className="v3-series-score">{g.away_abbr || short(g.away_team)} {g.away_score ?? ''}{g.away_score !== null ? ' : ' : ' vs '}{g.home_score ?? ''} {g.home_abbr || short(g.home_team)}</span>
      <em>{g.status === 'live' ? '진행 중' : g.status === 'final' ? '종료' : g.status === 'scheduled' ? '예정' : g.status_text}</em></a></li>)}</ol>
    <p className="v3-muted small">발행된 결과로 센 시리즈 기록이에요.{games.length < 4 ? ' 남은 경기는 일정이 올라오면 보여 드려요.' : ''}</p>
  </section>;
}

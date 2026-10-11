import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import { fetchGames, getJson } from './api';
import type { GameList } from './api';
import type { GameCard } from './model';
import { dateHeading, inningText, kstTime, shiftDate, short, todayKst } from './model';
import { teamColor } from './teams';
import { InfoSheet } from './Sheets';
import { VIEWER_KEY, reloadViewer } from './viewer';
import './service.css';

/** Ordinary list: one KST date as the heading, that date's games in start order. No sections, no modes, no badge. */
export const gameHref = (pk: number) => {
  const demo = new URLSearchParams(window.location.search).get('demo');
  return `/watch/${pk}${demo ? `?demo=${encodeURIComponent(demo)}` : ''}`;
};
const listHref = (date: string) => {
  const q = new URLSearchParams(window.location.search); q.set('date', date);
  return `/watch?${q.toString()}`;
};

export function StatusChip({ g }: { g: GameCard }) {
  if (g.status === 'live') return <span className="sv-status live"><i aria-hidden="true" />LIVE{g.inning && g.half ? <b>{inningText({ inning: g.inning, half: g.half })}</b> : null}</span>;
  if (g.status === 'final') return <span className="sv-status final">경기 종료</span>;
  if (g.status === 'scheduled') return <span className="sv-status">경기 전</span>;
  if (g.status === 'postponed') return <span className="sv-status">연기·중단</span>;
  return <span className="sv-status">{g.embargoed ? '진행 정보 없음' : g.status_text || '상태 확인 중'}</span>;
}

function Card({ g }: { g: GameCard }) {
  const scored = g.away_score !== null && g.home_score !== null;
  const name = `${short(g.away_team)} 대 ${short(g.home_team)}`;
  return <a className={`sv-card ${g.status}`} href={gameHref(g.game_pk)}
    aria-label={`${name}${g.double_header ? ` ${g.game_number}차전` : ''}, ${kstTime(g.start)} 시작, ${scored ? `${g.away_score} 대 ${g.home_score}, ` : ''}${g.status === 'live' ? '경기 중' : g.status === 'final' ? '경기 종료' : g.status === 'scheduled' ? '경기 전' : '상태 확인 중'}`}>
    <span className="sv-card-time">{kstTime(g.start)}{g.double_header && <em>{g.game_number}차전</em>}</span>
    <span className="sv-card-teams">
      <span className="sv-card-team" style={{ '--team': teamColor(g.away_team) } as CSSProperties}><i aria-hidden="true" />{short(g.away_team)}<b>{scored ? g.away_score : ''}</b></span>
      <span className="sv-card-team" style={{ '--team': teamColor(g.home_team) } as CSSProperties}><i aria-hidden="true" />{short(g.home_team)}<b>{scored ? g.home_score : ''}</b></span>
    </span>
    <StatusChip g={g} />
  </a>;
}

export default function WatchList() {
  const params = new URLSearchParams(window.location.search);
  const [date, setDate] = useState<string | null>(params.get('date'));
  const [dates, setDates] = useState<string[] | null>(null);
  const [games, setGames] = useState<GameList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState(false);
  const [viewerChange, setViewerChange] = useState(0);  // another tab changed the viewer delay: ask the list again at it
  useEffect(() => {
    if (import.meta.env.VITE_PUBLISH_BASE) return;
    const onStorage = (e: StorageEvent) => { if (e.key === VIEWER_KEY) { reloadViewer(); setViewerChange(n => n + 1); } };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  useEffect(() => {  // available KST dates (contract §1); without them, today's KST date
    const ctrl = new AbortController();
    getJson<{ dates: string[] }>('/service/dates', ctrl.signal).then(b => { setDates(b.dates); setDate(d => d ?? b.dates[0] ?? todayKst()); })
      .catch(() => { if (!ctrl.signal.aborted) { setDates([]); setDate(d => d ?? todayKst()); } });
    return () => ctrl.abort();
  }, []);
  useEffect(() => {
    if (!date) return;
    const ctrl = new AbortController();
    setGames(null); setError(null);
    fetchGames(date, ctrl.signal).then(setGames).catch(e => { if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : '경기 목록을 불러오지 못했어요.'); });
    return () => ctrl.abort();
  }, [date, viewerChange]);
  useEffect(() => { document.title = date ? `${dateHeading(date)} 경기 · 피치이지` : '피치이지'; }, [date]);

  // Previous/next: adjacent dates that have games when the server lists them, otherwise one calendar day.
  const sorted = dates?.length ? [...dates].sort() : null;
  const step = (dir: -1 | 1) => {
    if (!date) return null;
    if (!sorted) return shiftDate(date, dir);
    return (dir < 0 ? [...sorted].reverse().find(d => d < date) : sorted.find(d => d > date)) ?? null;
  };
  const prev = step(-1), next = step(1);
  const go = (d: string | null) => { if (!d) return; window.history.replaceState(null, '', listHref(d)); setDate(d); };

  return <div className="sv-page sv-list-page">
    <header className="sv-top">
      <a className="sv-brand" href={listHref(date ?? todayKst())}>Pitcheezy<span>.</span></a>
      <button className="sv-icon-btn" onClick={() => setInfo(true)} aria-label="서비스 정보와 한계 보기">정보</button>
    </header>
    <nav className="sv-datebar" aria-label="날짜 이동">
      <button className="sv-icon-btn" onClick={() => go(prev)} disabled={!prev} aria-label={prev ? `이전 날짜 ${dateHeading(prev)}` : '이전 날짜 없음'}>‹</button>
      <h1>{date ? dateHeading(date) : '날짜 확인 중'}<small>한국 시간</small></h1>
      <button className="sv-icon-btn" onClick={() => go(next)} disabled={!next} aria-label={next ? `다음 날짜 ${dateHeading(next)}` : '다음 날짜 없음'}>›</button>
    </nav>
    <main className="sv-list">
      {error ? <div className="sv-alert" role="alert"><p>{error}</p><button className="sv-btn" onClick={() => { const d = date; setDate(null); setTimeout(() => setDate(d), 0); }}>다시 시도</button></div>
        : games === null ? <p className="sv-muted" role="status">경기 목록을 불러오는 중…</p>
        : games.length === 0 ? <p className="sv-muted">이 날짜에는 경기가 없어요.</p>
        : <ul>{games.map(g => <li key={g.game_pk}><Card g={g} /></li>)}</ul>}
      {games && games.liveOk === false && <p className="sv-muted" role="status">실시간 경기 일정을 지금 불러오지 못해, 저장된 경기만 보여요.</p>}
    </main>
    {info && <InfoSheet onClose={() => setInfo(false)} notes={[]} />}
  </div>;
}

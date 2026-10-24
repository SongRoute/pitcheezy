/** v3 game list. Reading rules are the classic list's (src/service/WatchList.tsx: dates + manifest live dates, one reader per shown
 *  date, health.json lateness only while a game is in play, nothing read while the tab is hidden); copied because that module brings
 *  the classic styles. The look: a game in play is pinned on top as a large card; the rest are grouped as upcoming / finished. */
import { useEffect, useState } from 'react';
import { FetchError, fetchDates, fetchGames, fetchPublishHealth, fetchPublishedDates, fetchPublishedGames } from '../service/api';
import type { GameList } from '../service/api';
import type { GameCard } from '../service/model';
import { dateHeading, inningText, kstTime, shiftDate, short, todayKst } from '../service/model';
import { PUBLISH_CONFIG, clockWindowMs, isCalendarDate, isPublishStale, publishLag, serverNow } from '../service/publish';
import type { PublishStamp } from '../service/publish';
import { gameHref, listHref } from './links';
import Panel from './Panel';
import { InfoDetail } from './details';
import Series from './Series';

function startReader(read: (signal: AbortSignal) => Promise<boolean>, okMs: number, failMs: number, onWake?: () => void) {
  let alive = true, busy: AbortController | null = null, timer = 0;
  const run = async () => {
    window.clearTimeout(timer);
    if (!alive || busy || document.hidden) return;
    const ctrl = busy = new AbortController();
    let good = false;
    try { good = await read(ctrl.signal); } catch { good = false; }
    if (busy === ctrl) busy = null;
    if (alive) timer = window.setTimeout(() => void run(), good ? okMs : failMs);
  };
  const wake = () => { if (!document.hidden) { onWake?.(); void run(); } };
  void run();
  document.addEventListener('visibilitychange', wake); window.addEventListener('online', wake);
  return () => { alive = false; busy?.abort(); window.clearTimeout(timer); document.removeEventListener('visibilitychange', wake); window.removeEventListener('online', wake); };
}
const team = (abbr: string | null | undefined, name: string) => abbr || short(name);
const statusWords = (g: GameCard) => g.status === 'live' ? (g.inning && g.half ? inningText({ inning: g.inning, half: g.half }) : '경기 중')
  : g.status === 'final' ? '경기 종료' : g.status === 'scheduled' ? `${kstTime(g.start)} 시작` : g.status === 'postponed' ? '연기·중단' : g.embargoed ? '진행 정보 없음' : g.status_text || '상태 확인 중';

export function LiveCard({ g }: { g: GameCard }) {
  return <a className="v3-livecard" href={gameHref(g.game_pk)} aria-label={`${short(g.away_team)} ${g.away_score ?? ''} 대 ${g.home_score ?? ''} ${short(g.home_team)}, ${statusWords(g)}, 경기 보기`}>
    <span className="v3-chip conn ok"><i aria-hidden="true" />LIVE</span>
    <span className="v3-livecard-score" aria-hidden="true"><span>{team(g.away_abbr, g.away_team)}</span><b>{g.away_score ?? '–'}</b><i>:</i><b>{g.home_score ?? '–'}</b><span>{team(g.home_abbr, g.home_team)}</span></span>
    <span className="v3-livecard-meta" aria-hidden="true">{statusWords(g)}{g.venue ? ` · ${g.venue}` : ''}</span>
    <span className="v3-btn solid" aria-hidden="true">지금 보기</span>
  </a>;
}
export function Row({ g }: { g: GameCard }) {
  const scored = g.away_score !== null && g.home_score !== null;
  return <li><a className={`v3-gamerow ${g.status}`} href={gameHref(g.game_pk)} aria-label={`${short(g.away_team)} 대 ${short(g.home_team)}${g.double_header ? ` ${g.game_number}차전` : ''}, ${scored ? `${g.away_score} 대 ${g.home_score}, ` : ''}${statusWords(g)}`}>
    <span className="v3-gamerow-teams" aria-hidden="true"><span>{team(g.away_abbr, g.away_team)}<b>{scored ? g.away_score : ''}</b></span><span>{team(g.home_abbr, g.home_team)}<b>{scored ? g.home_score : ''}</b></span></span>
    <span className="v3-gamerow-meta" aria-hidden="true">{g.double_header && <em>{g.game_number}차전</em>}{statusWords(g)}</span>
    <span className="v3-chev" aria-hidden="true">›</span>
  </a></li>;
}

export default function List() {
  const params = new URLSearchParams(window.location.search);
  const [date, setDate] = useState<string | null>(() => { const d = params.get('date'); return __PZ_STATIC__ && !isCalendarDate(d) ? null : d; });
  const [dates, setDates] = useState<string[] | null>(null);
  const [games, setGames] = useState<GameList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (__PZ_STATIC__) return;
    const ctrl = new AbortController();
    fetchDates(ctrl.signal).then(list => { setDates(list); setDate(d => d ?? list[0] ?? todayKst()); })
      .catch(() => { if (!ctrl.signal.aborted) { setDates([]); setDate(d => d ?? todayKst()); } });
    return () => ctrl.abort();
  }, []);
  useEffect(() => {
    if (__PZ_STATIC__ || !date) return;
    const ctrl = new AbortController();
    setGames(null); setError(null);
    fetchGames(date, ctrl.signal).then(list => { if (!ctrl.signal.aborted) setGames(list); }).catch(e => { if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : '경기 목록을 불러오지 못했어요.'); });
    return () => ctrl.abort();
  }, [date, retry]);
  const [health, setHealth] = useState<PublishStamp | null>(null);
  const [healthFresh, setHealthFresh] = useState(false);
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!__PZ_STATIC__) return;
    return startReader(async signal => {
      try {
        const p = await fetchPublishedDates(signal);
        if (signal.aborted) return false;
        const today = todayKst(); setDates(p.dates);
        setDate(d => d ?? (p.dates.includes(today) ? today : p.dates.find(x => x <= today) ?? p.dates[p.dates.length - 1] ?? today));
        return true;
      } catch { if (!signal.aborted) { setDates(d => d ?? []); setDate(d => d ?? todayKst()); } return false; }
    }, PUBLISH_CONFIG.datesPollMs, PUBLISH_CONFIG.firstLoadRetryMs);
  }, []);
  useEffect(() => {
    if (!__PZ_STATIC__ || !date) return;
    let shown = false;
    setGames(null); setError(null);
    const stop = startReader(async signal => {
      let good = true;
      try {
        const list = await fetchPublishedGames(date, signal);
        if (signal.aborted) return false;
        shown = true; setGames(list); setError(null);
      } catch (e) {
        if (signal.aborted) return false;
        good = false;
        if (!shown) setError(e instanceof FetchError ? e.message : '경기 목록을 불러오지 못했어요.');
      }
      if (signal.aborted || document.hidden) return good;
      try { const h = await fetchPublishHealth(signal); if (!signal.aborted) { setHealth(h); setHealthFresh(true); } } catch { /* the last health stays */ }
      return good;
    }, PUBLISH_CONFIG.listPollMs, PUBLISH_CONFIG.firstLoadRetryMs, () => setHealthFresh(false));
    const tick = window.setInterval(() => setTick(t => t + 1), PUBLISH_CONFIG.tickMs);
    return () => { stop(); window.clearInterval(tick); };
  }, [date, retry]);
  const late = __PZ_STATIC__ && healthFresh && isPublishStale(publishLag(null, health, serverNow(clockWindowMs(PUBLISH_CONFIG.listPollMs))), PUBLISH_CONFIG.listPollMs);
  useEffect(() => { document.title = date ? `${dateHeading(date)} 경기 · 피치이지` : '피치이지'; }, [date]);
  const sorted = dates?.length ? [...dates].sort() : null;
  const step = (dir: -1 | 1) => {
    if (!date) return null;
    if (!sorted) return shiftDate(date, dir);
    return (dir < 0 ? [...sorted].reverse().find(d => d < date) : sorted.find(d => d > date)) ?? null;
  };
  const prev = step(-1), next = step(1);
  const go = (d: string | null) => { if (!d) return; window.history.replaceState(null, '', listHref(d)); setDate(d); };
  const live = games?.filter(g => g.status === 'live') ?? [];
  const upcoming = games?.filter(g => g.status === 'scheduled') ?? [];
  const rest = games?.filter(g => g.status !== 'live' && g.status !== 'scheduled') ?? [];
  return <div className="v3 v3-page v3-list">
    <header className="v3-board list"><span className="v3-brand">Pitcheezy</span><button className="v3-btn ghost dark" onClick={() => setInfo(true)} aria-label="서비스 정보와 한계 보기">정보</button></header>
    <nav className="v3-datebar" aria-label="날짜 이동">
      <button className="v3-btn ghost" onClick={() => go(prev)} disabled={!prev} aria-label={prev ? `이전 날짜 ${dateHeading(prev)}` : '이전 날짜 없음'}>‹</button>
      <h1>{date ? dateHeading(date) : '날짜 확인 중'}<small>한국 시간</small></h1>
      <button className="v3-btn ghost" onClick={() => go(next)} disabled={!next} aria-label={next ? `다음 날짜 ${dateHeading(next)}` : '다음 날짜 없음'}>›</button>
    </nav>
    <main className="v3-list-main">
      <Series />
      {error ? <div className="v3-alert" role="alert"><p>{error}</p><button className="v3-btn" onClick={() => setRetry(r => r + 1)}>다시 시도</button></div>
        : games === null ? <p className="v3-muted" role="status">경기 목록을 불러오는 중…</p>
        : games.length === 0 ? <p className="v3-empty-line">이 날짜에는 경기가 없어요.</p>
        : <>
          {live.map(g => <LiveCard key={g.game_pk} g={g} />)}
          {upcoming.length > 0 && <section><h2>예정</h2><ul>{upcoming.map(g => <Row key={g.game_pk} g={g} />)}</ul></section>}
          {rest.length > 0 && <section><h2>{rest.every(g => g.status === 'final') ? '종료' : '그 밖의 경기'}</h2><ul>{rest.map(g => <Row key={g.game_pk} g={g} />)}</ul></section>}
        </>}
      {late && live.length > 0 && <p className="v3-banner late" role="status"><b>갱신 지연</b><span> · 새 내용이 늦어지고 있어요. 마지막으로 올라온 내용을 보여 드려요.</span></p>}
      {games && games.liveOk === false && <p className="v3-muted small" role="status">실시간 경기 일정을 지금 불러오지 못해, 저장된 경기만 보여요.</p>}
    </main>
    {info && <Panel mode={window.innerWidth >= 720 ? 'overlay' : 'sheet'} title="이 서비스의 기준과 한계" onClose={() => setInfo(false)}><InfoDetail notes={[]} /></Panel>}
  </div>;
}

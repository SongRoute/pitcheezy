import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { FetchError, fetchDates, fetchGames, fetchPublishHealth, fetchPublishedDates, fetchPublishedGames } from './api';
import type { GameList } from './api';
import type { GameCard } from './model';
import { dateHeading, inningText, kstTime, shiftDate, short, todayKst } from './model';
import { teamColor } from './teams';
import { InfoSheet, SPOILER_LINE, publishDelayLine } from './Sheets';
import { VIEWER_KEY, reloadViewer } from './viewer';
import { PUBLISH_CONFIG, clockWindowMs, isCalendarDate, isPublishStale, publicGameHref, publicListHref, publishLag, serverNow } from './publish';
import type { PublishStamp } from './publish';
import './service.css';

/** Ordinary list: one KST date as the heading, that date's games in start order. No sections, no modes, no badge. */
export const gameHref = (pk: number) => {
  if (__PZ_STATIC__) return publicGameHref(pk);
  const demo = new URLSearchParams(window.location.search).get('demo');
  return `/watch/${pk}${demo ? `?demo=${encodeURIComponent(demo)}` : ''}`;
};
const listHref = (date: string) => {
  if (__PZ_STATIC__) return publicListHref(date);
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
    <span className="sv-card-time">{kstTime(g.start)}{g.series_description && <em>{g.series_description}{g.series_game_number ? ` · ${g.series_game_number}차전` : ''}</em>}{g.double_header && <em>{g.game_number}차전</em>}</span>
    <span className="sv-card-teams">
      <span className="sv-card-team" style={{ '--team': teamColor(g.away_team) } as CSSProperties}><i aria-hidden="true" />{short(g.away_team)}<b>{scored ? g.away_score : ''}</b></span>
      <span className="sv-card-team" style={{ '--team': teamColor(g.home_team) } as CSSProperties}><i aria-hidden="true" />{short(g.home_team)}<b>{scored ? g.home_score : ''}</b></span>
    </span>
    <StatusChip g={g} />{(g.probable_pitchers?.away || g.probable_pitchers?.home) && <small className="sv-card-starters">예상 선발 · {g.probable_pitchers?.away?.name ?? '미정'} / {g.probable_pitchers?.home?.name ?? '미정'}</small>}
  </a>;
}

/** Public bundle: one repeating reader. One request chain at a time; nothing is asked while the tab is hidden (it starts again when the
 *  tab is visible or the network is back); `read` answers true (next read after `okMs`) or false (after `failMs`). Returns the stop
 *  function, which also aborts the signal `read` was given. */
function startReader(read: (signal: AbortSignal) => Promise<boolean>, okMs: () => number, failMs: () => number, onWake?: () => void) {
  let alive = true, busy: AbortController | null = null, timer = 0;
  const run = async () => {
    window.clearTimeout(timer);
    if (!alive || busy || document.hidden) return;
    const ctrl = busy = new AbortController();
    let good = false;
    try { good = await read(ctrl.signal); } catch { good = false; }
    if (busy === ctrl) busy = null;
    if (alive) timer = window.setTimeout(() => void run(), good ? okMs() : failMs());
  };
  const wake = () => { if (!document.hidden) { onWake?.(); void run(); } };
  void run();
  document.addEventListener('visibilitychange', wake); window.addEventListener('online', wake);
  return () => { alive = false; busy?.abort(); window.clearTimeout(timer);
    document.removeEventListener('visibilitychange', wake); window.removeEventListener('online', wake); };
}

export default function WatchList() {
  const params = new URLSearchParams(window.location.search);
  // public bundle: `?date=` is used only when it is a real calendar date (it ends up in a file name)
  const [date, setDate] = useState<string | null>(() => { const d = params.get('date'); return __PZ_STATIC__ && !isCalendarDate(d) ? null : d; });
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

  // ---- API mode: read once (as before) ----
  useEffect(() => {  // available KST dates (contract §1); without them, today's KST date
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
  }, [date, viewerChange]);

  // ---- public bundle: published files, read again at the configured intervals ----
  const [manifestMs, setManifestMs] = useState<number | null>(null);
  const [fixedDelayS, setFixedDelayS] = useState<number | null>(null);
  const listPollMs = manifestMs ?? PUBLISH_CONFIG.listPollMs;
  const listPeriod = useRef(listPollMs);
  listPeriod.current = listPollMs;  // the next timer uses the new period without clearing this date's cards
  const [health, setHealth] = useState<PublishStamp | null>(null);
  const [healthFresh, setHealthFresh] = useState(false);   // health.json was read since the tab last came back into view
  const [, setTick] = useState(0);
  useEffect(() => {  // dates.json + manifest live_dates, read again so a date the publisher starts to follow appears without a reload
    if (!__PZ_STATIC__) return;
    let period = PUBLISH_CONFIG.listPollMs;
    return startReader(async signal => {
      try {
        const p = await fetchPublishedDates(signal);
        if (signal.aborted) return false;
        const today = todayKst(); setDates(p.dates); setManifestMs(p.pollMs); setFixedDelayS(p.fixedDelayS); period = p.pollMs ?? PUBLISH_CONFIG.listPollMs;
        // first answer only: today when the publisher follows it, else the newest published date that is not in the future
        setDate(d => d ?? (p.dates.includes(today) ? today : p.dates.find(x => x <= today) ?? p.dates[p.dates.length - 1] ?? today));
        return true;
      } catch { if (!signal.aborted) { setDates(d => d ?? []); setDate(d => d ?? todayKst()); } return false; }
    }, () => Math.max(60000, period), () => Math.max(60000, period));
  }, []);
  useEffect(() => {
    if (!__PZ_STATIC__ || !date) return;
    // One reader per shown date; its abort signal is this date's generation: an answer for a date the viewer has left is never applied.
    let shown = false;
    setGames(null); setError(null);
    const stop = startReader(async signal => {
      let good = true;
      try { const h = await fetchPublishHealth(signal); if (!signal.aborted) { setHealth(h); setHealthFresh(true); } } catch { /* the last health stays */ }
      if (signal.aborted || document.hidden) return false;
      try {
        const list = await fetchPublishedGames(date, signal);
        if (signal.aborted) return false;
        shown = true; setGames(list); setError(null);
      } catch (e) {
        if (signal.aborted) return false;
        good = false;
        if (!shown || (e instanceof FetchError && e.notice)) setError(e instanceof FetchError ? e.message : '경기 목록을 불러오지 못했어요.');   // a failed re-read never blanks a list on screen; a service notice does (restored by the next good read)
      }
      return good;
    }, () => listPeriod.current, () => listPeriod.current, () => setHealthFresh(false));
    const tick = window.setInterval(() => setTick(t => t + 1), PUBLISH_CONFIG.tickMs);  // lets the late notice appear between reads
    return () => { stop(); window.clearInterval(tick); };
  }, [date]);
  // judged only on a health.json read since the tab came back: what was on screen before a background period is never called late
  const late = __PZ_STATIC__ && healthFresh && isPublishStale(publishLag(null, health, serverNow(clockWindowMs(listPollMs))), listPollMs);
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
      {!error && games?.some(g => g.status === 'live') && <p className="sv-spoiler-line" role="note"><b>스포일러 주의</b> {__PZ_STATIC__ && publishDelayLine(fixedDelayS)}{SPOILER_LINE}</p>}
      {error ? <div className="sv-alert" role="alert"><p>{error}</p><button className="sv-btn" onClick={() => { const d = date; setDate(null); setTimeout(() => setDate(d), 0); }}>다시 시도</button></div>
        : games === null ? <p className="sv-muted" role="status">경기 목록을 불러오는 중…</p>
        : games.length === 0 ? <p className="sv-muted">이 날짜에는 경기가 없어요.</p>
        : <>{(['live', 'scheduled', 'other'] as const).map(section => { const rows = games.filter(g => section === 'other' ? g.status !== 'live' && g.status !== 'scheduled' : g.status === section); return rows.length ? <section key={section}><h2>{section === 'live' ? '지금' : section === 'scheduled' ? '다음 · 예정 경기' : '경기 기록'}</h2><ul>{rows.map(g => <li key={g.game_pk}><Card g={g} /></li>)}</ul></section> : null; })}</>}
      {/* only while a game on this list is in play: between games the publisher may be idle on purpose */}
      {late && games?.some(g => g.status === 'live') && <p className="sv-muted sv-publish-late" role="status">갱신 지연 · 새 내용이 늦어지고 있어요. 마지막으로 올라온 내용을 보여 드려요.</p>}
      {games && games.liveOk === false && <p className="sv-muted" role="status">실시간 경기 일정을 지금 불러오지 못해, 저장된 경기만 보여요.</p>}
    </main>
    {info && <InfoSheet onClose={() => setInfo(false)} notes={[]} />}
  </div>;
}

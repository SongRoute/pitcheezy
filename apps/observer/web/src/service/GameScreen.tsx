import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { FetchError, fetchDetail, mergeDetail } from './api';
import type { Actual, Candidate, Detail, GameCard, Linescore, Person, PitchRow, PlayerInfo, Rec, Situation } from './model';
import { dateHeading, inningText, kstTime, policyShare, short } from './model';
import { teamColor } from './teams';
import Zone from './Zone';
import { DetailSheet, InfoSheet, PaSheet, Sheet, provenanceText } from './Sheets';
import './service.css';

/** One game screen for live, demo and finished games (UI redesign 2026-10-05).
 *  Two independent states, never mixed: the GAME's status (LIVE dot, from the server) and the VIEWER's state
 *  (following the latest vs. looking at a past pitch; auto refresh on/off). */
export const POLL_MS = 2000;
export const STALE_MS = 30000;
type Selection = { kind: 'latest' } | { kind: 'pitch'; index: number };

const UNIT_KEY = 'pitcheezy.unit';
const readUnit = () => { try { return localStorage.getItem(UNIT_KEY) === 'mph' ? 'mph' : 'kmh'; } catch { return 'kmh'; } };
const speedText = (mph: number | null, unit: 'kmh' | 'mph') => mph === null ? '' : unit === 'kmh' ? `${Math.round(mph * 1.609344)}km/h` : `${mph.toFixed(1)}mph`;
const clockText = (ms: number) => new Date(ms).toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul', hour12: false });
const hhmm = (ms: number) => new Date(ms).toLocaleTimeString('en-GB', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit' });
export const gameIdFromPath = () => {
  const m = /^\/(?:watch|game|live)\/(\d+)\/?$/.exec(window.location.pathname);
  const q = new URLSearchParams(window.location.search).get('game');
  const id = Number(m?.[1] ?? q);
  return Number.isInteger(id) && id > 0 ? id : null;
};

function Bases({ mask }: { mask: number }) {
  const on = (b: number) => (mask & (1 << (b - 1))) !== 0;
  const runners = [1, 2, 3].filter(on);
  return <svg className="sv-bases" viewBox="0 0 44 32" role="img" aria-label={`주자 ${runners.length ? runners.join('·') + '루' : '없음'}`}>
    {[[22, 6, 2], [36, 18, 1], [8, 18, 3]].map(([x, y, b]) => <rect key={b} x={x - 5.5} y={y - 5.5} width="11" height="11" transform={`rotate(45 ${x} ${y})`} className={on(b) ? 'on' : ''} />)}
  </svg>;
}
const handText = (h: string | null | undefined, role: 'p' | 'b') => h === 'R' ? (role === 'p' ? '우투' : '우타') : h === 'L' ? (role === 'p' ? '좌투' : '좌타') : h === 'S' ? '양타' : '';
const stat3 = (v: number | null | undefined) => typeof v === 'number' ? v.toFixed(3).replace(/^0/, '') : '–';
const era = (v: number | null | undefined) => typeof v === 'number' ? v.toFixed(2) : '–';

/** B 3 green / S 2 yellow / O 2 red dots; the count is spoken, not printed as a number. */
function Count({ s }: { s: Situation }) {
  const dots = (n: number, on: number, cls: string) => Array.from({ length: n }, (_, i) => <i key={i} className={`${cls}${i < on ? ' on' : ''}`} />);
  return <span className="sv-bso" role="img" aria-label={`볼 ${s.balls}, 스트라이크 ${s.strikes}, ${s.outs}아웃`}>
    <span aria-hidden="true"><b>B</b>{dots(3, s.balls, 'b')}</span><span aria-hidden="true"><b>S</b>{dots(2, s.strikes, 's')}</span><span aria-hidden="true"><b>O</b>{dots(2, s.outs, 'o')}</span></span>;
}

/** Away always on the LEFT, home on the RIGHT; only the pitcher/batter roles swap with the half-inning. */
function PlayerCell({ team, abbr, role, person, hand, info, onOpen, side, today }: { team: string; abbr?: string | null; role: 'p' | 'b'; person: Person | null; hand: string | null | undefined;
  info: PlayerInfo | undefined; onOpen: () => void; side: 'away' | 'home'; today: { text: string; spoken: string } | null }) {
  const name = person?.name ?? '—';
  const stats = role === 'p' ? `ERA ${era(info?.era)}` : `AVG ${stat3(info?.avg)} OPS ${stat3(info?.ops)}`;
  const handed = handText(hand, role);
  const spoken = `${short(team)} ${role === 'p' ? '투수' : '타자'} ${name}, ${handed}, ${stats}, ${today?.spoken ?? '오늘 기록 미확인'}. 선수 상세 열기`;
  return <button className={`sv-player ${side}`} style={{ '--team': teamColor(team) } as CSSProperties} onClick={onOpen} aria-haspopup="dialog" aria-label={spoken} title={name}>
    <span className="sv-player-name"><b>{role === 'b' ? '타자' : '투수'}: {abbr || short(team)} #{info?.jersey ?? '–'}</b> <small>{handed}</small></span>
    <span className="sv-player-2">{stats}</span>
    <span className="sv-player-3">{today?.text ?? (role === 'b' ? '–타수 · –안타 · –출루' : '–구 · –삼진 · –자책')}</span>
  </button>;
}

/** Pitcher's view: a right-handed batter stands on the SCREEN RIGHT, a left-handed batter on the left (broadcast camera).
 *  The figure holds the bat on the side away from the plate. */
function Silhouette({ side }: { side: 'L' | 'R' }) {
  const atLeft = side === 'L';
  return <svg className={`sv-batter ${side}`} viewBox="0 0 40 100" role="img" aria-label={side === 'R' ? '우타자, 화면 오른쪽 타석' : '좌타자, 화면 왼쪽 타석'}>
    <circle cx="20" cy="12" r="8" /><path d="M10 26h20l3 34h-7l-2 36h-6l-1-30-1 30h-6l-2-36H7z" />
    <line x1={atLeft ? 10 : 30} y1="30" x2={atLeft ? 2 : 38} y2="4" />
  </svg>;
}

/** Ranked list on the side opposite the batter; % = policy selection share (spoken, never a success probability). */
function RecList({ rec, emptyText, onOpen }: { rec: Rec | null; emptyText: string; onOpen: () => void }) {
  if (!rec || rec.status !== 'ready' || !rec.candidates.length) return <button className="sv-reclist empty" onClick={onOpen} aria-haspopup="dialog">
    <span className="sv-norec">{rec?.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황' : rec ? `추천 기록 없음${rec.reason ? ` · ${rec.reason}` : ''}` : emptyText}</span></button>;
  return <button className="sv-reclist" onClick={onOpen} aria-haspopup="dialog"
    aria-label={`추천 후보 ${rec.candidates.map(c => `${c.rank}순위 ${c.pitch_label} ${c.zone_label || '위치 없음'}, 정책 선택 비중 ${policyShare(c.probability)}`).join('; ')}. 퍼센트는 성공 확률이 아니에요. 자세히 보기`}>
    {rec.candidates.map((c: Candidate) => <span key={c.rank} className="sv-rec-row" aria-hidden="true">
      <i className="sv-rk">{c.rank}</i><span className="sv-rec-pct">{policyShare(c.probability)}</span><span className="sv-rec-type">{c.pitch_label}</span></span>)}
  </button>;
}

function BoxSheet({ ls, game, at, onClose }: { ls: Linescore | null | undefined; game: GameCard; at: Situation | null; onClose: () => void }) {
  const n = Math.max(9, ls?.innings ?? 0, at?.inning ?? 0);
  const cell = (v: number | null | undefined) => typeof v === 'number' ? v : '–';
  return <Sheet title="이닝별 기록" onClose={onClose}>
    {ls ? <div className="sv-box-scroll"><table className="sv-box">
      <thead><tr><th scope="col">팀</th>{Array.from({ length: n }, (_, i) => <th key={i} scope="col" className={at?.inning === i + 1 ? 'cur' : ''}>{i + 1}</th>)}
        <th scope="col">R</th><th scope="col">H</th><th scope="col">E</th><th scope="col" title="볼넷+몸에 맞는 공">사사구</th></tr></thead>
      <tbody>{(['away', 'home'] as const).map(k => { const t = ls[k]; const team = k === 'away' ? game.away_team : game.home_team; return <tr key={k}>
        <th scope="row">{short(team)}</th>{Array.from({ length: n }, (_, i) => <td key={i} className={at?.inning === i + 1 && (at.half === 'Top') === (k === 'away') ? 'cur' : at?.inning === i + 1 ? 'curcol' : ''}>{cell(t.innings[i])}</td>)}
        <td><b>{cell(t.r)}</b></td><td>{cell(t.h)}</td><td>{cell(t.e)}</td><td>{cell(t.bb_hbp)}</td></tr>; })}</tbody></table></div>
      : <p className="sv-muted">이닝별 기록은 아직 준비 중이에요. 지금 점수는 {short(game.away_team)} {at?.away_score ?? game.away_score ?? '–'} : {at?.home_score ?? game.home_score ?? '–'} {short(game.home_team)}예요.</p>}
    <p className="sv-muted">공개된 플레이까지만 셉니다. 아직 오지 않은 이닝은 –로 표시해요.{at ? ` 강조한 칸은 지금 보는 ${inningText(at)}이에요.` : ''}</p>
  </Sheet>;
}

export default function GameScreen({ gamePk }: { gamePk: number }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<{ text: string; status: number } | null>(null);
  const [auto, setAuto] = useState(true);
  const [sel, setSel] = useState<Selection | null>(null);
  const [lastOk, setLastOk] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [sheet, setSheet] = useState<'pa' | 'detail' | 'info' | 'box' | 'player-away' | 'player-home' | null>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const sheetOpener = useRef<HTMLElement | null>(null);
  const openSheet = (kind: typeof sheet) => { sheetOpener.current = document.activeElement as HTMLElement; setSheet(kind); };
  const closeSheet = () => { setSheet(null); const opener = sheetOpener.current; requestAnimationFrame(() => opener?.isConnected && opener.focus()); };
  const [edges, setEdges] = useState({ left: false, right: false });
  const [busy, setBusy] = useState(false);
  const [unit, setUnit] = useState<'kmh' | 'mph'>(readUnit);
  const detailRef = useRef<Detail | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const load = useCallback(async (full = false, force = false) => {
    // a background poll never cancels a slower answer still on its way (only a manual refresh does)
    if (inflight.current && !inflight.current.signal.aborted && !force) return;
    inflight.current?.abort();
    const ctrl = new AbortController(); inflight.current = ctrl;
    setBusy(true);
    try {
      const old = detailRef.current;
      const last = old?.pitches.length ? old.pitches[old.pitches.length - 1].index : undefined;
      let next = await fetchDetail(gamePk, ctrl.signal, full || !old || last === undefined ? undefined : last);
      if (old && !full && last !== undefined) {
        const merged = mergeDetail(old, next);
        next = merged ?? await fetchDetail(gamePk, ctrl.signal);  // cutoff moved back (demo seek/reset): take the full history again
      }
      if (ctrl.signal.aborted) return;
      detailRef.current = next; setDetail(next); setError(null); setLastOk(Date.now());
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof FetchError ? { text: e.message, status: e.status } : { text: '자료를 불러오지 못했어요.', status: 0 });
    } finally { if (inflight.current === ctrl) { inflight.current = null; setBusy(false); } }
  }, [gamePk]);

  useEffect(() => { void load(true, true); return () => inflight.current?.abort(); }, [load]);
  const live = detail ? detail.session_status !== 'ended' && (detail.game.status === 'live' || (detail.next !== null && detail.game.status !== 'final')) : false;
  // Polling follows the viewer's switch only; the game's LIVE status decides whether there is anything to poll.
  useEffect(() => {
    if (!auto || !live) return;
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, POLL_MS);
    return () => window.clearInterval(id);
  }, [auto, live, load]);
  useEffect(() => { const id = window.setInterval(() => setNow(Date.now()), 5000); return () => window.clearInterval(id); }, []);
  useEffect(() => {  // coming back to the tab: refresh at once when following
    const onVis = () => { if (!document.hidden && auto && live) void load(); };
    document.addEventListener('visibilitychange', onVis); return () => document.removeEventListener('visibilitychange', onVis);
  }, [auto, live, load]);

  // default selection once data arrives: live → follow latest; finished → first pitch
  useEffect(() => { if (detail && sel === null) setSel(live ? { kind: 'latest' } : detail.pitches.length ? { kind: 'pitch', index: detail.pitches[0].index } : { kind: 'latest' }); }, [detail, sel, live]);
  // a pinned pitch that disappeared (demo seek back) falls back to latest
  useEffect(() => { if (detail && sel?.kind === 'pitch' && !detail.pitches.some(p => p.index === sel.index)) setSel({ kind: 'latest' }); }, [detail, sel]);
  // following the latest keeps the newest card in view; a pinned card never moves the strip
  useEffect(() => { const el = stripRef.current; if (!el) return; if (sel?.kind !== 'pitch') el.scrollLeft = el.scrollWidth; else { const selected = el.querySelector<HTMLElement>('[aria-pressed="true"]'); if (selected) el.scrollLeft = selected.offsetLeft - el.offsetLeft - (el.clientWidth - selected.clientWidth) / 2; } }, [sel, detail?.pitches.length]);
  useEffect(() => { if (detail) document.title = `${short(detail.game.away_team)} 대 ${short(detail.game.home_team)} · 피치이지`; }, [detail?.game.away_team, detail?.game.home_team]);

  useEffect(() => {
    const el = stripRef.current; if (!el) return;
    const measure = () => setEdges({ left: el.scrollLeft > 2, right: el.scrollWidth - el.clientWidth - el.scrollLeft > 2 });
    const ro = new ResizeObserver(measure); ro.observe(el);
    for (const child of el.children) ro.observe(child);
    el.addEventListener('scroll', measure, { passive: true }); measure();
    return () => { ro.disconnect(); el.removeEventListener('scroll', measure); };
  }, [detail, sel]);

  const byIndex = useMemo(() => new Map(detail?.pitches.map(p => [p.index, p]) ?? []), [detail?.pitches]);
  if (!detail) {
    return <div className="sv-page"><TopBar /><main className="sv-center">{error
      ? <div className="sv-alert" role="alert"><p>{error.status === 423 ? '이 경기는 지금 기록을 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.' : error.text}</p>
        <div className="sv-row"><button className="sv-btn" onClick={() => void load(true)}>다시 시도</button><a className="sv-btn" href="/watch">경기 목록</a></div></div>
      : <p className="sv-muted" role="status">경기 기록을 불러오는 중…</p>}</main></div>;
  }

  const { game } = detail;
  const pitches = detail.pitches;
  const lastPitch = pitches[pitches.length - 1] ?? null;
  const pitch: PitchRow | null = sel?.kind === 'pitch' ? byIndex.get(sel.index) ?? null : null;
  const latest = pitch === null;  // a pin whose pitch is gone (demo seek back) shows latest until the effect resets it
  // What the stage shows
  const situation: Situation | null = latest ? detail.current?.situation ?? lastPitch?.situation ?? null : pitch!.situation;
  const rec: Rec | null = latest ? (live ? detail.next : lastPitch?.rec ?? null) : pitch!.rec;
  const actual: Actual | null = latest ? lastPitch?.actual ?? null : pitch!.actual;
  const nextPhase = latest && live;
  const shownPitch = latest ? lastPitch : pitch;
  const paKey = latest ? detail.current?.pa_key ?? lastPitch?.pa_key ?? null : pitch!.pa_key;
  const paPitches = pitches.filter(p => p.pa_key === paKey);
  const ending = shownPitch?.actual?.event ? detail.pas.find(p => p.pa_key === shownPitch.pa_key && p.complete)?.result ?? null : null;
  const nextNumber = nextPhase ? (detail.current?.pa_key && lastPitch?.pa_key === detail.current.pa_key ? lastPitch.pitch_number + 1 : 1) : null;
  const batterP = latest ? detail.current?.batter ?? lastPitch?.batter ?? null : pitch!.batter;
  const pitcherP = latest ? detail.current?.pitcher ?? lastPitch?.pitcher ?? null : pitch!.pitcher;
  const side = batterP?.side === 'R' || batterP?.side === 'L' ? batterP.side : null;  // switch hitters: only the side actually taken; never guessed

  const stale = (live && auto && lastOk !== null && now - lastOk > STALE_MS) || (live && (detail.feed.ok === false || (detail.feed.stale_s ?? 0) > 30));
  const scored = situation ?? null;
  const awayScore = latest && game.away_score !== null ? game.away_score : scored?.away_score ?? null;
  const homeScore = latest && game.home_score !== null ? game.home_score : scored?.home_score ?? null;

  // browsing a past pitch pins the view and switches auto refresh off (no fetch while pinned); LIVE brings both back
  const pick = (index: number) => { setSel({ kind: 'pitch', index }); setAuto(false); };
  const goLatest = () => { setSel({ kind: 'latest' }); setAuto(true); void load(false, true); };
  const recTitle = nextPhase ? `다음 공 추천${nextNumber ? ` · ${nextNumber}구` : ''}` : '이 공 던지기 전 추천';
  const actualTitle = nextPhase ? `직전 투구${lastPitch ? ` · ${inningText(lastPitch.situation)} ${lastPitch.pitch_number}구` : ''}` : `실제 투구${pitch ? ` · ${pitch.pitch_number}구` : ''}`;
  const top = situation?.half === 'Top';
  const info = (p: Person | null) => p ? detail.players?.[String(p.id)] : undefined;
  const openPa = () => openSheet('pa');
  const cands = rec?.status === 'ready' ? rec.candidates : [];
  const venue = game.venue ? `${game.venue} · ` : '';
  const when = `${dateHeading(game.kst_date)} ${kstTime(game.start)}${game.double_header ? ` · ${game.game_number}차전` : ''}`;
  const emptyRec = nextPhase ? (detail.current?.phase === 'between_innings' ? '공수 교대 중' : detail.current?.phase === 'resolving' ? '결과 확인 중' : '추천 대기 중') : '이 시점에는 추천이 없어요';
  const actualWait = detail.current?.phase === 'between_innings' ? '공수 교대 중' : detail.current?.phase === 'resolving' ? '결과 확인 중' : '다음 공 대기 중';
  const list = <RecList rec={rec} emptyText={emptyRec} onOpen={() => openSheet('detail')} />;
  // New as-of player context is optional; absent fields stay unknown rather than guessed from an incomplete pitch list.
  const context = latest ? detail.current_context : pitch?.context;
  const fmt = (v: number | null | undefined) => typeof v === 'number' ? v : '–';
  const bt = context?.batter;
  const pt = context?.pitcher;
  const batterToday = { text: `${fmt(bt?.at_bats)}타수 · ${fmt(bt?.hits)}안타 · ${fmt(bt?.on_base)}출루`, spoken: `이 시점까지 ${fmt(bt?.at_bats)}타수 ${fmt(bt?.hits)}안타 ${fmt(bt?.on_base)}출루` };
  const pitcherToday = { text: `${fmt(pt?.pitches)}구 · ${fmt(pt?.strikeouts)}삼진 · ${fmt(pt?.earned_runs)}자책`, spoken: `이 시점까지 ${fmt(pt?.pitches)}구 ${fmt(pt?.strikeouts)}삼진 ${fmt(pt?.earned_runs)}자책` };
  const upcoming = context?.next_batters ?? [];
  const isDemo = detail.session !== null;
  const statusLabel = isDemo && detail.session_status === 'ended' ? '시연 종료' : isDemo && detail.session_status === 'ready' && latest ? '시연 대기' : live ? (!latest || !auto ? '최신으로 ↗' : isDemo ? (detail.session_status === 'paused' ? '일시정지' : '시연 중') : 'LIVE') : game.status === 'final' ? '종료' : game.status === 'scheduled' ? '예정' : '대기';
  const playerSheetSide = sheet === 'player-away' ? 'away' : 'home';
  const playerRole = (playerSheetSide === 'away') === top ? 'b' : 'p';
  const sheetPerson = playerRole === 'b' ? batterP : pitcherP;
  const sheetInfo = info(sheetPerson);

  return <div className="sv-page sv-game">
    <header className="sv-top sv-head">
      <a className="sv-back" href={backHref()} aria-label="경기 목록으로">‹</a>
      <h1 className="sv-head-main" aria-label={`${short(game.away_team)} ${awayScore ?? '–'} 대 ${homeScore ?? '–'} ${short(game.home_team)}. ${venue}${dateHeading(game.kst_date)} ${kstTime(game.start)} 시작`}>
        <span className="sv-head-score" aria-hidden="true"><span className="sv-tn">{short(game.away_team)}</span><strong style={{ color: scoreColor(game.away_team) }}>{awayScore ?? '–'}</strong><i>vs</i><strong style={{ color: scoreColor(game.home_team) }}>{homeScore ?? '–'}</strong><span className="sv-tn">{short(game.home_team)}</span></span>
        <small aria-hidden="true">{game.venue && <span>{game.venue}</span>}<span>{when} KST</span></small>
      </h1>
      <button className={`sv-live ${live && latest && auto && !isDemo ? 'on' : ''}`} onClick={live ? goLatest : undefined} disabled={!live}
        aria-label={`${isDemo ? '지난 경기 시연. ' : ''}${statusLabel}${live ? ', 최신 공으로 이동하고 자동 갱신 켜기' : ''}`}>
        {live && latest && auto && !isDemo && <i aria-hidden="true" />}{statusLabel}</button>
    </header>
    {/* Connection problems remain explicit alerts; history state is in the header. */}
    {(stale || error) && <p className="sv-notice warn" role="alert" title={error?.text}>
      {error ? '⚠ 연결 끊김' : '⏱ 갱신 늦어짐'}{lastOk ? ` · ${hhmm(lastOk)} 수신` : ''}<span className="sr-only">{error ? ` ${error.text}` : ''}{lastOk ? ` (${clockText(lastOk)})` : ''}</span>
        <button className="sv-link-btn" onClick={() => void load(true, true)} disabled={busy}>{busy ? '…' : '다시'}</button>
    </p>}

    <section className="sv-board" aria-label="경기 상황">
      <div className="sv-board-row">
        {situation ? <button className="sv-inning" onClick={openPa} aria-haspopup="dialog" aria-label={`${inningText(situation)}, 타석 선택`}>{situation.inning}회{situation.half === 'Top' ? '초' : '말'} ▾<small>타석 선택</small></button> : <span />}
        {situation && <Count s={situation} />}
        {situation && <Bases mask={situation.bases} />}
      </div>
    </section>

    <section className="sv-stage" aria-label={nextPhase ? '다음 공 추천' : '이 공의 추천과 실제 투구'}>
      <h2 className="sr-only">{recTitle} ({rec ? provenanceText(rec) : '추천 없음'}). 퍼센트는 정책 선택 비중이며 성공 확률이 아니에요.</h2>
      <div className="sv-duel" aria-label="타자와 투수 대결">
        <span><b>타자: {batterP?.name ?? '선수 미확인'}</b></span>
        <i aria-hidden="true">vs</i>
        <span><b>투수: {pitcherP?.name ?? '선수 미확인'}</b></span>
      </div>
      {/* Pitcher's view, with the batter on the correct screen side. The zone owns the centre;
          recommendations now live below it, leaving its outside-coordinate margin unobstructed. */}
      <div className={`sv-zrow ${side === 'L' ? 'bat-l' : 'bat-r'}`}>
        {side && <div className={`sv-batband ${side === 'L' ? 'at-left' : 'at-right'}`}><Silhouette side={side} /></div>}
        <div className="sv-zone-wrap">
          <Zone bounds={detail.zone_bounds} candidates={cands}
            actual={!nextPhase && actual && actual.x !== null && actual.z !== null ? { x: actual.x, z: actual.z } : null} actualLabel=""
            setupX={!nextPhase ? actual?.setup_x ?? null : null} />
        </div>
      </div>
      <div className="sv-pitch-summary">
        <div className="sv-rec-summary"><span className="sv-summary-label">{recTitle}</span>{list}</div>
        <div className={`sv-actual-summary${nextPhase ? ' waiting' : ''}`}><span className="sv-summary-label">{nextPhase ? '실제 투구' : actualTitle}</span>{nextPhase ? <p className="sv-muted" role="status">{actualWait}</p> : actual ? <>
        <p className="sv-actual-caption"><i className="sv-key-actual" /><b>{actual.pitch_label}</b> {actual.speed_mph !== null && <><b>{unit === 'kmh' ? Math.round(actual.speed_mph * 1.609344) : actual.speed_mph.toFixed(1)}</b><small>{unit === 'kmh' ? 'km/h' : 'mph'}</small></>}</p>
        <p className="sv-result">{ending ?? actual.result_label}</p>
      </> : <p className="sv-muted">실제 투구 기록 없음</p>}</div>
      </div>

      <div className={`sv-strip-shell${edges.left ? ' fade-left' : ''}${edges.right ? ' fade-right' : ''}`}>
      <div className="sv-strip" ref={stripRef} role="group" aria-label="이 타석의 투구, 좌우로 스크롤">
        {paPitches.map(p => { const on = !latest && pitch?.index === p.index; const a = p.actual;
          const sp = a?.speed_mph != null ? (unit === 'kmh' ? Math.round(a.speed_mph * 1.609344) : a.speed_mph.toFixed(1)) : '';
          return <button key={p.index} className={`sv-card ${on ? 'on' : ''}`} aria-pressed={on} onClick={() => pick(p.index)}
            aria-label={`${p.pitch_number}구 ${a?.pitch_label ?? '구종 미상'}${a?.speed_mph != null ? ` ${speedText(a.speed_mph, unit)}` : ''} ${a?.result_label ?? ''}`}>
            <Zone mini bounds={detail.zone_bounds} candidates={[]} actualLabel="" setupX={null} actual={a && a.x !== null && a.z !== null ? { x: a.x, z: a.z } : null} />
            <span className="sv-card-cap" aria-hidden="true"><span>{p.pitch_number}. {a?.pitch_label || '구종 미상'}</span><span>{sp || '–'} <small>{unit === 'kmh' ? 'km/h' : 'mph'}</small></span></span></button>; })}
        {nextPhase && nextNumber && paKey === detail.current?.pa_key && <button className="sv-card pending on" aria-pressed="true" onClick={goLatest} aria-label={`다음 공 ${nextNumber}구, 아직 안 던짐`}>
          <span className="sv-card-q" aria-hidden="true">?</span><span className="sv-card-cap" aria-hidden="true">{nextNumber}. 다음</span></button>}
      </div>
      </div>
    </section>

    <section className="sv-context" aria-label="선수와 다음 타자">
      <div className="sv-players">
        <PlayerCell side="away" team={game.away_team} abbr={game.away_abbr} role={top ? 'b' : 'p'} person={top ? batterP : pitcherP} hand={top ? batterP?.side : pitcherP?.hand} info={info(top ? batterP : pitcherP)} onOpen={() => openSheet('player-away')} today={top ? batterToday : pitcherToday} />
        <PlayerCell side="home" team={game.home_team} abbr={game.home_abbr} role={top ? 'p' : 'b'} person={top ? pitcherP : batterP} hand={top ? pitcherP?.hand : batterP?.side} info={info(top ? pitcherP : batterP)} onOpen={() => openSheet('player-home')} today={top ? pitcherToday : batterToday} />
      </div>
      <div className="sv-upcoming" aria-label="다음 두 타자">{[0, 1].map(i => <span key={i} title={upcoming[i]?.name ?? undefined}>{upcoming[i] ? `${upcoming[i].order}번 · ${upcoming[i].name ?? '타자 미확인'}` : '–번 · 타자 미확인'}</span>)}</div>
    </section>

    {sheet === 'pa' && <PaSheet pas={detail.pas} selected={paKey} onClose={closeSheet} onBox={() => setSheet('box')}
      onPick={p => { setSheet(null); if ((live && p.pa_key === detail.current?.pa_key && !p.complete) || typeof p.first !== 'number' || !byIndex.has(p.first)) goLatest(); else pick(p.first); }} />}
    {/* a selected past pitch shows the linescore as of that pitch (never the latest one, which would reveal later play) */}
    {sheet === 'box' && <BoxSheet ls={latest ? detail.linescore_asof : pitch!.line ?? null} game={game} at={situation} onClose={closeSheet} />}
    {sheet === 'detail' && <DetailSheet bounds={detail.zone_bounds} onClose={closeSheet} rec={rec} recTitle={rec ? `${recTitle} (${provenanceText(rec)})` : recTitle}
      actual={nextPhase ? null : actual} actualTitle={actualTitle} paResult={ending} notes={detail.notes} speed={mph => speedText(mph, unit)}
      extra={<div className="sv-row"><button className="sv-btn" onClick={() => setSheet('info')}>기준과 한계 보기</button>
        <button className="sv-btn" onClick={() => { const u = unit === 'kmh' ? 'mph' : 'kmh'; setUnit(u); try { localStorage.setItem(UNIT_KEY, u); } catch { /* ignore */ } }}
          aria-label={`속도 단위 바꾸기, 지금 ${unit === 'kmh' ? 'km/h' : 'mph'}`}>속도 단위: {unit === 'kmh' ? 'km/h' : 'mph'}</button></div>} />}
    {(sheet === 'player-away' || sheet === 'player-home') && <Sheet title={sheetPerson?.name ?? '선수 정보'} onClose={closeSheet}>
      <dl className="sv-dl"><dt>팀</dt><dd>{short(playerSheetSide === 'away' ? game.away_team : game.home_team)} · {playerRole === 'b' ? '타자' : '투수'}</dd>
        <dt>등번호</dt><dd>#{sheetInfo?.jersey ?? '–'}</dd><dt>시즌 기록</dt><dd>{playerRole === 'b' ? `AVG ${stat3(sheetInfo?.avg)} · OPS ${stat3(sheetInfo?.ops)}` : `ERA ${era(sheetInfo?.era)}`}</dd>
        <dt>오늘 기록</dt><dd>{playerRole === 'b' ? batterToday.text : pitcherToday.text}</dd></dl>
      <p className="sv-muted">{sheetInfo?.stats_as_of ? `${sheetInfo.stats_as_of} 경기 전 시즌 기록` : '경기 전 시즌 기록 미확인'}. 오늘 기록은 현재 공식 피드의 해당 시점까지 사건을 재구성한 값으로 이후 기록 정정이 반영될 수 있습니다. 출루는 안타+볼넷+몸에 맞는 공입니다. 확인되지 않은 값은 –로 표시합니다.</p>
    </Sheet>}
    {sheet === 'info' && <InfoSheet onClose={closeSheet} notes={detail.notes} />}
  </div>;
}

const backHref = () => { const demo = new URLSearchParams(window.location.search).get('demo'); return demo ? `/watch?demo=${encodeURIComponent(demo)}` : '/watch'; };
function TopBar({ right, title, label }: { right?: ReactNode; title?: string; label?: string }) {
  const demo = new URLSearchParams(window.location.search).get('demo');
  return <header className="sv-top">
    <a className="sv-back" href={demo ? `/watch?demo=${encodeURIComponent(demo)}` : '/watch'} aria-label="경기 목록으로">‹</a>
    {title ? <h1 className="sv-title" aria-label={label}>{title}</h1> : <a className="sv-brand small" href="/watch">Pitcheezy<span>.</span></a>}
    <span className="sv-top-right">{right}</span>
  </header>;
}

/** Darken only low-contrast team colours for score text; team identity hue remains. */
function scoreColor(team: string) {
  const hex = teamColor(team); const rgb = [1,3,5].map(i => parseInt(hex.slice(i,i+2),16));
  const lum = (v: number[]) => v.map(x => x / 255).map(x => x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4).reduce((a,x,i) => a+x*[.2126,.7152,.0722][i],0);
  while ((.928 + .05) / (lum(rgb) + .05) < 4.5) for (let i=0;i<3;i++) rgb[i] = Math.floor(rgb[i] * .94);
  return '#' + rgb.map(x=>x.toString(16).padStart(2,'0')).join('');
}

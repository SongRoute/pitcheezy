import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { FetchError, fetchDetail, fetchPublishHealth, fetchPublishedDetail, fetchPublishedManifest, isGamePublishedSoon, mergeDetail } from './api';
import { PUBLISH_CONFIG, gameCheck, gameCheckLag, staleAfterMs, clockWindowMs, isPublishStale, lostAfterMs, publicListHref, publishLag, serverNow } from './publish';
import type { PublishStamp, PublishHealth } from './publish';
import type { Actual, Candidate, Detail, DisplayStateName, GameCard, Linescore, Pa, Person, PitchRow, PreviousPick, Rec, Situation } from './model';
import { teamColor } from './teams';
import { heldReceipt, inningText, kstTime, pitchName, plainZone, pitchKind, resultKind, settledCorrection, short, surname } from './model';
import Zone, { G1Dot, RankSq } from './Zone';
import { ABSENT_PAST, DetailSheet, InfoSheet, PaSheet, PitchSheet, Sheet, SheetFrame, absentLine } from './Sheets';
import SyncSheet, { FOLD_TEXT } from './SyncSheet';
import { ExpectationLine, ReferenceSheet, ReferenceTeaser } from './Reference';
import type { ViewerSetting } from './viewer';
import { VIEWER_KEY, learnViewer, readViewer, reloadViewer, requestDelay, wantedDelay, writeViewer } from './viewer';
import { MixTable, RecordSheet, PitchMixSheet } from './LivePanels';
import { BOOTH, DEMO, F1, QUIET_S, boothCols } from './f1Flags';
import type { BoothCard } from './f1Flags';
const BOOTH_MQ = '(min-width: 1600px) and (min-height: 900px)';
import { AnalysisSheet, SceneLine, SceneList, SceneSheet, useScenes } from './Scenes';
import { BatterSplits, BatterToday, HeadToHead, HeadToHeadToday, PitchLog, PitchMap, PitcherMix, SpeedBands, WinProbCard, candidatesFor, livePolicy, useTabData } from './Tabs';
import './service.css';

/** One game screen for live, demo and finished games (UI redesign 2026-10-05).
 *  Two independent states, never mixed: the GAME's status (LIVE dot, from the server) and the VIEWER's state
 *  (following the latest vs. looking at a past pitch; auto refresh on/off). */
export const POLL_MS = 2000;
export const STALE_MS = 30000;
/** a scheduled live game is asked again slowly so the screen starts by itself; a first load that failed is retried by itself */
export const PREGAME_POLL_MS = 10000;
export const FIRST_LOAD_RETRY_MS = 4000;
/** API mode keeps these cadences; static mode uses the manifest period or build-time fallbacks. */
const CADENCE = __PZ_STATIC__ ? { live: PUBLISH_CONFIG.livePollMs, pregame: PUBLISH_CONFIG.pregamePollMs, retry: PUBLISH_CONFIG.firstLoadRetryMs }
  : { live: POLL_MS, pregame: PREGAME_POLL_MS, retry: FIRST_LOAD_RETRY_MS };
/** The client's own connection is failing: what is on screen is the last answer, not the current state. */
const CLIENT_LOST = { state: 'disconnected' as DisplayStateName, label: '연결 끊김', message: '연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.' };
/** Static mode: the files are reachable but the last publish is older than the (provisional) threshold. Same held look as a lost connection. */
const QUIET = { label: '잠시 기다리는 중', message: '새 내용을 기다리고 있어요. 마지막으로 받은 내용을 보여 드려요.' };
const PUBLISH_LATE = { state: 'disconnected' as DisplayStateName, label: '갱신 지연', message: '새 내용이 늦어지고 있어요. 마지막으로 받은 내용을 보여 드려요.' };
/** Static mode: back in view (or back online / awake) and the first read is still on its way: neutral, never "late" from old data. */
const PUBLISH_CHECKING = { state: 'unconfirmed' as DisplayStateName, label: '확인 중', message: '새 내용을 확인하고 있어요.' };
const unconfirmed = (p: PitchRow | null | undefined) => p?.receipt?.status === 'unconfirmed';
/** Live view only (nextPhase): the newest pitch's reaction kind for the optional 3D batter skin (pz-skin/pz-batter.js), read from
 *  data attributes on .sv-zrow — nothing is drawn. take = no swing, miss = swing and miss, contact = foul / batted out, hit = base hit. */
const HITS = new Set(['single', 'double', 'triple', 'home_run']);
const pzKind = (a: Actual | null | undefined): string => {
  const code = a?.description?.trim().toLowerCase().replaceAll(' ', '_') ?? '', k = resultKind(code);
  return k === 'ball' ? 'take' : k === 'strike' ? (/swinging|missed_bunt/.test(code) ? 'miss' : /foul/.test(code) ? 'contact' : 'take')
    : k === 'inplay' ? (HITS.has(a?.event ?? '') ? 'hit' : 'contact') : '';
};
const pickText = (pitch: string, zone: string | null | undefined) => zone ? `${pitch} · ${zone}` : pitch;
const previousText = (p: PreviousPick | null) => p ? pickText(pitchName(p.pitch_label, p.pitch_type), p.zone_label ? plainZone(p.zone_label) : null) : '';
/** D1 client guard: what was last drawn for the pending pitch slot, so a list that changes without a server note is still announced. */
type Drawn = { key: string | null; lastIdx: number; who: string; seen: Detail; sig: string; top: string; prev: { sig: string; top: string } | null };
type Selection = { kind: 'latest' } | { kind: 'pitch'; key: string };

const speedText = (mph: number | null, unit: 'kmh' | 'mph') => mph === null ? '' : unit === 'kmh' ? `${Math.round(mph * 1.609344)}km/h` : `${mph.toFixed(1)}mph`;
const shortGameDate = (ymd: string) => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(ymd)) return '날짜 미정';
  const [y, m, d] = ymd.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d));
  if (date.getUTCFullYear() !== y || date.getUTCMonth() !== m - 1 || date.getUTCDate() !== d) return '날짜 미정';
  const day = new Intl.DateTimeFormat('en-US', { weekday: 'short', timeZone: 'UTC' }).format(date);
  return `${ymd.slice(2)} (${day})`;
};
export const gameIdFromPath = () => {
  const m = /^\/(?:watch|game|live)\/(\d+)\/?$/.exec(window.location.pathname);
  const q = new URLSearchParams(window.location.search).get('game');
  // public bundle: `?game=` counts only as plain decimal digits (no 1e3, 0x10, signs or spaces)
  const id = Number(m?.[1] ?? (__PZ_STATIC__ && !/^[0-9]+$/.test(q ?? '') ? NaN : q));
  return Number.isInteger(id) && id > 0 ? id : null;
};

function Bases({ mask }: { mask: number }) {
  const on = (b: number) => (mask & (1 << (b - 1))) !== 0;
  return <svg className="sv-runners" viewBox="0 0 44 32" aria-hidden="true">
    {[[22, 9, 2], [34, 21, 1], [10, 21, 3]].map(([x, y, b]) => <rect key={b} x={x - 6} y={y - 6} width="12" height="12" transform={`rotate(45 ${x} ${y})`} className={on(b) ? 'on' : ''} />)}
  </svg>;
}
export const handText = (h: string | null | undefined, role: 'p' | 'b') => h === 'R' ? (role === 'p' ? '우투' : '우타') : h === 'L' ? (role === 'p' ? '좌투' : '좌타') : h === 'S' ? '양타' : '';
const stat3 = (v: number | null | undefined) => typeof v === 'number' ? v.toFixed(3).replace(/^0/, '') : '—';
const era = (v: number | null | undefined) => typeof v === 'number' ? v.toFixed(2) : '—';
/** '14구' when known, '자책 —' when not: a missing number keeps its word and an em dash, never a 0. */
const part = (v: number | null | undefined, unit: string) => typeof v === 'number' ? `${v}${unit}` : `${unit} —`;

/** F1 game card (owner 11/21): runners diamond beside B·S·O dot rows (balls 3, strikes 2, outs 2), next to the box score. */
function Bso({ s }: { s: Situation }) {
  const runners = [1, 2, 3].filter(b => (s.bases & (1 << (b - 1))) !== 0);
  const dots = (n: number, on: number, k: string) => Array.from({ length: n }, (_, i) => <i key={i} className={i < on ? `on ${k}` : ''} />);
  return <div className="f1-bso" role="img" aria-label={`볼 ${s.balls}, 스트라이크 ${s.strikes}, ${s.outs}아웃, 주자 ${runners.length ? runners.join('·') + '루' : '없음'}`}>
    <span aria-hidden="true"><Bases mask={s.bases} /></span>
    <span className="f1-bso-rows" aria-hidden="true"><span><b>B</b>{dots(3, s.balls, 'b')}</span><span><b>S</b>{dots(2, s.strikes, 's')}</span><span><b>O</b>{dots(2, s.outs, 'o')}</span></span></div>;
}

/** Header row 3 side: the batter is always on the LEFT and the pitcher on the RIGHT, whichever team is at bat. Below 360px only the
 *  family name is drawn (the full name stays in the label and title). */
function Who({ role, person, hand, team, today, onOpen }: { role: 'b' | 'p'; person: Person | null; hand: string | null | undefined; team: string; today: ReactNode; onOpen: () => void }) {
  const name = person?.name ?? null;
  const handed = handText(hand, role);
  const roleWord = role === 'b' ? '타자' : '투수';
  // a full name that does not fit its cell falls back to the family name (as below 360px), never an ellipsis
  const nameRef = useRef<HTMLElement>(null);
  const [tight, setTight] = useState(false);
  useLayoutEffect(() => { const b = nameRef.current; if (!b) return;
    const fit = () => { const full = b.querySelector<HTMLElement>('.full'); if (full) setTight(full.getBoundingClientRect().width > b.clientWidth + 0.5); };  // an inline span has no scroll width: use its box
    fit(); const ro = new ResizeObserver(fit); ro.observe(b); return () => ro.disconnect(); }, [name]);
  return <button className={`sv-who ${role === 'b' ? 'bat' : 'pit'}${tight ? ' tight' : ''}`} onClick={onOpen} aria-haspopup="dialog" title={name ?? undefined}
    aria-label={`${short(team)} ${roleWord} ${name ?? '정보 없음'}${handed ? `, ${handed}` : ''}. 선수 상세 열기`}>
    <small>{roleWord}{handed ? ` · ${handed}` : ''}</small>
    {name ? <b ref={nameRef}><span className="full">{name}</span><span className="sur">{surname(name)}</span></b> : <b>—</b>}
    <span className="sv-who-today">{today}</span></button>;
}

/** Header row 2: 1–9 (more in extra innings) and R. Missing cells stay blank; the column of the half in progress is highlighted and its
 *  not-yet-played cell reads '–'. Taps open the inning-by-inning sheet. */
function LineScore({ ls, away, home, awayR, homeR, current, onOpen }: { ls: Linescore | null | undefined; away: string; home: string; awayR: number | null; homeR: number | null;
  current: number | null; onOpen: () => void }) {
  const n = Math.max(9, ls?.innings ?? 0, current ?? 0);
  const cell = (k: 'away' | 'home', i: number) => { const v = ls?.[k].innings[i]; return typeof v === 'number' ? v : ''; };
  const r = (k: 'away' | 'home') => ls?.[k].r ?? (k === 'away' ? awayR : homeR) ?? '–';
  return <button className="sv-ls" onClick={onOpen} aria-haspopup="dialog" aria-label={`이닝별 점수. ${away} ${r('away')}, ${home} ${r('home')}. 이닝별 기록 보기`}>
    <span className="sv-ls-grid" style={{ '--n': n } as CSSProperties} aria-hidden="true">
      <span />{Array.from({ length: n }, (_, i) => <span key={i} className={current === i + 1 ? 'h cur' : 'h'}>{i + 1}</span>)}<span className="h r">R</span>
      {(['away', 'home'] as const).map(k => [<span key={`${k}t`} className="t">{k === 'away' ? away : home}</span>,
        ...Array.from({ length: n }, (_, i) => <span key={`${k}${i}`} className={current === i + 1 ? 'cur v' : 'v'}>{cell(k, i)}</span>), <span key={`${k}r`} className="r v">{r(k)}</span>])}
    </span></button>;
}

type Tone = 'live' | 'late' | 'lost' | 'checking' | 'none' | 'fixed' | 'pregame';
const NavIcon = ({ d }: { d: string }) => <svg viewBox="0 0 24 24" aria-hidden="true"><path d={d} /></svg>;
const ICONS = {
  rec: 'M4 4h16v16H4zM4 9.3h16M4 14.7h16M9.3 4v16M14.7 4v16',
  player: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4.5 20.5c.8-3.9 3.8-6 7.5-6s6.7 2.1 7.5 6',
  mix: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9 3.6c2.2 2.4 2.2 14.4 0 16.8M15 3.6c-2.2 2.4-2.2 14.4 0 16.8',
  record: 'M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01',
};

/** Pitcher's view: a right-handed batter stands on the SCREEN RIGHT, a left-handed batter on the left (broadcast camera).
 *  The figure holds the bat on the side away from the plate. */
function Silhouette({ side }: { side: 'L' | 'R' }) {
  const atLeft = side === 'L';
  return <svg className={`sv-batter ${side}`} viewBox="0 0 40 100" role="img" aria-label={side === 'R' ? '우타자, 화면 오른쪽 타석' : '좌타자, 화면 왼쪽 타석'}>
    <circle cx="20" cy="12" r="8" /><path d="M10 26h20l3 34h-7l-2 36h-6l-1-30-1 30h-6l-2-36H7z" />
    <line x1={atLeft ? 10 : 30} y1="30" x2={atLeft ? 2 : 38} y2="4" />
  </svg>;
}

/** Main screen: one ruled row of three cells, rank square + pitch name (UI 10/26 ③; location and numbers stay in the detail sheet, D49).
 *  Without candidates the cells read '—' and the reason is kept for screen readers. Any cell opens the recommendation sheet. */
function RecList({ rec, emptyText, onOpen, lead = '', absentText = ABSENT_PAST }: { rec: Rec | null; emptyText: string; onOpen: () => void; lead?: string; absentText?: string }) {
  const cands = rec?.status === 'ready' ? [...rec.candidates].sort((a, b) => a.rank - b.rank).slice(0, 3) : [];
  if (!cands.length) { const why = rec?.status === 'unsupported' ? rec.reason ?? '예측을 지원하지 않는 상황' : rec ? absentLine(rec, absentText) : emptyText;
    // F1 review (11/22): one plain sentence, not three '—' cells that read as a load failure
    return <button className="sv-reclist empty f1-rec-none" onClick={onOpen} aria-haspopup="dialog" aria-label={`${why}. 자세히 보기`}>
      <span aria-hidden="true">{why}</span></button>; }
  return <button className="sv-reclist" onClick={onOpen} aria-haspopup="dialog"
    aria-label={`${lead}예측 후보 ${cands.map(c => `${c.rank}순위 ${c.pitch_label}`).join('; ')}. 자세히 보기`}>
    {[0, 1, 2].map(i => { const c: Candidate | undefined = cands[i];
      return <span key={i} className="sv-rec-cell sv-rec-row" aria-hidden="true">{c ? <><RankSq rank={c.rank} /><b className="sv-rec-type">{c.pitch_label}</b>{BOOTH.big && c.probability !== null && <em className="f1-big-pct">{Math.round(c.probability * 100)}%</em>}</> : '—'}</span>; })}
  </button>;
}

/** Result hold card (proposal 2026-11-19 ①): "방금 공 · N구" over "{구종} {구속} · {위치} · {판정}" (+ "· {타석 결과}"), in the box of the
 *  recommendation row (same height), so the one-screen layout does not move. Only recorded values; a missing one is left out. */
/** Result hold card (proposal 2026-11-19 ①, review H): "방금 공 · N구 · {타석 결과}" over "{구종} {구속} · {위치} · {판정}", in the box of
 *  the recommendation row (same height), so the one-screen layout does not move. The PA-ending pitch's own call is left out when the
 *  PA result already says it ("삼진" / "헛스윙 삼진", "볼" / "볼넷", any ball in play): the result is said once. Only recorded values. */
function ResultCard({ pitch, actual, ending, speed, unit, onOpen }: { pitch: PitchRow; actual: Actual; ending: string | null; speed: (mph: number | null) => string; unit: 'kmh' | 'mph'; onOpen: () => void }) {
  // F1 (owner 11/21): the PA result in large type when known, then "{batter} · {N구} · {pitch}" in one line
  const zone = actual.zone_label || null, rawCall = actual.result_label || null;
  const call = rawCall && ending && (ending.includes(rawCall) || rawCall.includes(ending) || resultKind(actual.description) === 'inplay') ? null : rawCall;
  const typeSpeed = [actual.pitch_label, actual.speed_mph !== null ? speed(actual.speed_mph) : ''].filter(Boolean).join(' ');
  const body = [typeSpeed, zone, call].filter(Boolean).join(' · ');
  const spoken = actual.speed_mph === null ? '' : unit === 'kmh' ? ` 시속 ${Math.round(actual.speed_mph * 1.609344)}킬로미터` : ` 시속 ${actual.speed_mph.toFixed(1)}마일`;
  const label = `방금 던진 공, ${pitch.pitch_number}구, ${actual.pitch_label}${spoken}${zone ? `, ${zone}` : ''}${call ? `, ${call}` : ''}${ending ? `, 타석 결과 ${ending}` : ''}. 자세히 보기`;
  return <button className={`sv-result-card${ending ? ' has-ending' : ''}`} onClick={onOpen} aria-haspopup="dialog" aria-label={label}>
    <span className="sv-rc-title" aria-hidden="true">{ending ? <b>{ending}</b> : <>방금 공 · {pitch.pitch_number}구</>}</span>
    <span className="sv-rc-body" aria-hidden="true"><G1Dot kind={pitchKind(actual)} n={pitch.pitch_number} /><b>{[pitch.batter?.name, ending ? `${pitch.pitch_number}구` : null, body].filter(Boolean).join(' · ')}</b></span>
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
  const [error, setError] = useState<{ text: string; status: number; notice?: boolean } | null>(null);
  const [auto, setAuto] = useState(true);
  const [sel, setSel] = useState<Selection | null>(null);
  const [lastOk, setLastOk] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  // the tab was hidden: polling paused on purpose, so that time never counts as a late or lost connection (re-based on return)
  const [hidden, setHidden] = useState(() => document.hidden);
  const [resumed, setResumed] = useState(0);
  const [sheet, setSheet] = useState<'pa' | 'pa-empty' | 'detail' | 'info' | 'box' | 'player-b' | 'player-p' | 'player-h' | 'sync' | 'reference' | 'history' | 'mix' | 'win' | 'model' | 'scene' | 'scenes' | 'game' | 'analysis' | null>(null);
  const [view, setView] = useState<'rec' | 'game' | 'player'>('rec');
  // owner 11/22 (7): 라이브 button — tap (or pick a past pitch) freezes the screen on this snapshot; tap again → latest. Polling continues.
  const [frozen, setFrozen] = useState<Detail | null>(null);
  const [pRole, setPRole] = useState<'p' | 'b' | 'h'>('p');
  const viewScroll = useRef<Record<string, number>>({});
  // owner 11/23 (A): looking at something pauses live — the 경기·선수 tabs, any sheet, 자세히 — so nothing moves under the viewer; the way
  // back is ▶ 라이브로 or the 예측 tab. Data keeps arriving behind the frozen snapshot (newSince counts it).
  const holdLive = () => { if (F1.tabs && F1.pause && live && detailRef.current) { const d = detailRef.current; setFrozen(f => f ?? d); } };
  const goView = (v: 'rec' | 'game' | 'player') => { if (v !== 'rec') holdLive(); viewScroll.current[view] = window.scrollY; setSheet(null); setView(v);
    requestAnimationFrame(() => window.scrollTo({ top: viewScroll.current[v] ?? 0 })); };
  const [recordTab, setRecordTab] = useState('문자 로그');
  const [recOpen, setRecOpen] = useState(false);
  const [sceneRank, setSceneRank] = useState<number | null>(null);
  // the header status button's explanation bubble (UI 10/26 ⑥): the state's sentence lives here, not on the main screen
  const [pop, setPop] = useState(false);
  // service notice hold (ui/notice-hide-1021, review P2-1): set by a notice answer, cleared only when a good answer is applied, so a
  // network error or 404 between the notice and the next good answer never redraws the pre-notice game
  const [hold, setHold] = useState(false);
  const holdRef = useRef(false);  // mirrors `hold` for the load closure
  // viewer delay (중계 맞추기): this device's setting, what the server last said about its limits, and the one-time question
  const [setting, setSetting] = useState<ViewerSetting>(readViewer);
  const viewerMeta = useRef<{ max: number | null } | null>(null);
  const prompted = useRef(false);
  const [emptyPa, setEmptyPa] = useState<Pa | null>(null);  // a plate appearance with no pitch, shown by itself in a sheet
  const sheetOpener = useRef<HTMLElement | null>(null);
  const openSheet = (kind: typeof sheet) => { if (kind !== 'sync') holdLive(); sheetOpener.current = document.activeElement as HTMLElement; setSheet(kind); };
  const closeSheet = () => { setSheet(null); const opener = sheetOpener.current; requestAnimationFrame(() => opener?.isConnected && opener.focus()); };
  const [busy, setBusy] = useState(false);
  const unit = 'kmh' as 'kmh' | 'mph';  // owner 11/23: speed-unit toggle removed, km/h only
  const detailRef = useRef<Detail | null>(null);
  // static mode only: what the last game file and health.json said about publish time
  const [manifestMs, setManifestMs] = useState<number | null>(null);
  useEffect(() => {
    if (!__PZ_STATIC__) return;
    let timer = 0, stopped = false, active = false, period = CADENCE.live;
    const ctrl = new AbortController();
    const read = async () => {
      window.clearTimeout(timer);
      if (stopped || active || document.hidden) return;
      active = true;
      try { const m = await fetchPublishedManifest(ctrl.signal); if (!stopped) { setManifestMs(m.pollMs); period = m.pollMs ?? CADENCE.live; } } catch { /* keep the last manifest */ }
      active = false;
      if (!stopped) timer = window.setTimeout(() => void read(), Math.max(60000, period));
    };
    const wake = () => { if (!document.hidden) void read(); };
    void read(); document.addEventListener('visibilitychange', wake); window.addEventListener('online', wake);
    return () => { stopped = true; ctrl.abort(); window.clearTimeout(timer); document.removeEventListener('visibilitychange', wake); window.removeEventListener('online', wake); };
  }, []);
  const [fileStamp, setFileStamp] = useState<PublishStamp | null>(null);
  const [health, setHealth] = useState<PublishHealth | null>(null);
  const [pendingSince, setPendingSince] = useState<number | null>(null);
  const pending = useRef<number | null>(null);
  const pendingTarget = useRef<number | null>(null);
  const [withdrawn, setWithdrawn] = useState(false);
  const withdrawal = useRef(false);  // only a valid, explicit non-null game check can reopen a withdrawn game
  const appliedSequence = useRef<number | null>(null);
  // Product request budget, not measured notice latency: while visible, revalidate an unchanged game after 60s
  // at the next regular poll. Failures/older bodies do not reset it; 304 does. Existing single-flight/visibility gates apply.
  const gameVerifiedAt = useRef<number | null>(null);
  const [checking, setChecking] = useState(false);
  const scenes = useScenes(gamePk, F1.scenes);
  const tabData = useTabData(gamePk);
  // booth monitor: landscape ≥1600×900 places the cards in columns (l1/l2); narrower or portrait screens keep the phone order
  // a timeline that scrolls inside its card (booth, phone first screen): keep the newest pitch in view and fade the top only when cut
  useEffect(() => { const l = document.getElementById('sv-atbat-list'); if (!l) return; const over = l.scrollHeight > l.clientHeight + 1; l.classList.toggle('f1-tl-over', over); if (over) l.scrollTop = l.classList.contains('f1-newest-first') ? 0 : l.scrollHeight; });
  const [boothWide, setBoothWide] = useState(() => typeof matchMedia === 'function' && matchMedia(BOOTH_MQ).matches);
  useEffect(() => { if (typeof matchMedia !== 'function') return; const m = matchMedia(BOOTH_MQ); const on = () => setBoothWide(m.matches); on(); m.addEventListener('change', on); return () => m.removeEventListener('change', on); }, []);
  const inflight = useRef<AbortController | null>(null);
  const drawn = useRef<Drawn | null>(null);
  const picked = useRef(false);  // the viewer chose a past pitch (as opposed to the default first pitch of a game that is not live)

  const load = useCallback(async (full = false, force = false) => {
    if (__PZ_STATIC__ && document.hidden) return;
    // a background poll never cancels a slower answer still on its way (only a manual refresh does)
    if (inflight.current && !inflight.current.signal.aborted && !force) return;
    inflight.current?.abort();
    const ctrl = new AbortController(); inflight.current = ctrl;
    setBusy(true);
    try {
      if (__PZ_STATIC__) {  // one health-first chain; malformed/old health falls back to a whole-file read
        let h: PublishHealth | null = null;
        try { h = await fetchPublishHealth(ctrl.signal); if (!ctrl.signal.aborted) setHealth(h); } catch (e) { if (e instanceof FetchError && e.notice) throw e; /* keep the last health */ }
        if (ctrl.signal.aborted || document.hidden) return;
        const { present, check } = gameCheck(h, gamePk);
        if (withdrawal.current && !check) { setError({ text: '이 경기는 지금 볼 수 없어요.', status: 404 }); return; }
        if (present) { setLastOk(Date.now()); if (!holdRef.current) setError(null); }
        if (check && check.file_sequence !== null && (appliedSequence.current === null || check.file_sequence > appliedSequence.current)) {
          pendingTarget.current = Math.max(pendingTarget.current ?? 0, check.file_sequence);
          pending.current ??= performance.now(); setPendingSince(pending.current);
        }
        if (check?.file_sequence === null) {
          pendingTarget.current = null; pending.current = null; setPendingSince(null); withdrawal.current = true; setWithdrawn(true); detailRef.current = null; setDetail(null); setFileStamp(null);
          setError({ text: '이 경기는 지금 볼 수 없어요.', status: 404 });
          return;
        }
        if (!holdRef.current && pendingTarget.current === null && check && detailRef.current && appliedSequence.current !== null && check.file_sequence! <= appliedSequence.current && gameVerifiedAt.current !== null && performance.now() - gameVerifiedAt.current < 60000) {
          setError(null); setLastOk(Date.now()); return;
        }
        const got = await fetchPublishedDetail(gamePk, ctrl.signal, holdRef.current);
        if (ctrl.signal.aborted) return;
        if (pendingTarget.current !== null && got.stamp.seq! < pendingTarget.current) return;  // propagation delay: retry this target next tick
        pendingTarget.current = null; pending.current = null; setPendingSince(null); appliedSequence.current = got.stamp.seq; gameVerifiedAt.current = got.verifiedAt;
        withdrawal.current = false; setWithdrawn(false); detailRef.current = got.detail; setDetail(got.detail); setFileStamp(got.stamp); setError(null); setLastOk(Date.now()); holdRef.current = false; setHold(false);
        return;
      }
      const old = detailRef.current;
      const last = old?.pitches.length ? old.pitches[old.pitches.length - 1].index : undefined;
      let next = await fetchDetail(gamePk, ctrl.signal, full || !old || last === undefined ? undefined : last, requestDelay());
      if (old && !full && last !== undefined) {
        const merged = mergeDetail(old, next);
        next = merged ?? await fetchDetail(gamePk, ctrl.signal, undefined, requestDelay());  // cutoff moved back (demo seek/reset): take the full history again
      }
      // Viewer delay: the answer states the server's switch and the delay it was served at. Switched off (or an older server) → no
      // delay parameter from now on. Served at another delay than this device's (the switch just came on, another default, a
      // clamp) → that answer is a different moment of the game and is never drawn: ask again in full at this device's delay.
      for (let round = 0; ; round++) {
        if (learnViewer({ features: { viewer_delay: { enabled: next.viewer_on === true, default_s: next.viewer?.default_s } } })) setSetting(readViewer());
        if (!next.viewer) break;
        const want = next.viewer.max_delay_s !== null ? Math.min(wantedDelay(), next.viewer.max_delay_s) : wantedDelay();
        if (Math.abs(next.viewer.delay_s - want) < 0.05) break;
        // the server keeps serving another delay than asked (a limit it did not state): take the one it serves rather than fail
        if (ctrl.signal.aborted) return;  // a newer request owns the setting now: an overtaken one never writes it
        if (round >= 2) { setSetting(writeViewer({ delay_s: next.viewer.delay_s })); break; }
        next = await fetchDetail(gamePk, ctrl.signal, undefined, requestDelay());
      }
      if (ctrl.signal.aborted) return;
      detailRef.current = next; setDetail(next); setError(null); setLastOk(Date.now()); holdRef.current = false; setHold(false);
    } catch (e) {
      if (ctrl.signal.aborted) return;
      const notice = e instanceof FetchError && e.notice;
      if (notice) { holdRef.current = true; setHold(true); }
      if (__PZ_STATIC__ && pending.current !== null && !notice) return;  // announced file is still propagating; keep neutral and retry
      setError(e instanceof FetchError ? { text: e.message, status: e.status, notice } : { text: '자료를 불러오지 못했어요.', status: 0 });
    } finally { if (inflight.current === ctrl) { inflight.current = null; setBusy(false); if (__PZ_STATIC__) setChecking(false); } }
  }, [gamePk]);

  useEffect(() => { void load(true, true); return () => inflight.current?.abort(); }, [load]);
  /** Another delay is another moment of the game (a longer one is further BACK): nothing already downloaded stays on screen, the
   *  cursor starts over, and the screen's own D1 memory is cleared so the jump is not announced as a changed recommendation. */
  const changeDelay = useCallback((seconds: number) => {
    const max = viewerMeta.current?.max ?? null;
    setSetting(writeViewer({ delay_s: Math.max(0, max !== null ? Math.min(seconds, max) : seconds), synced: true, asked: true }));
    inflight.current?.abort(); detailRef.current = null; drawn.current = null;
    setDetail(null); setError(null);
    void load(true, true);
  }, [load]);
  // another tab of this browser changed the delay: this one must not keep showing a later moment of the game
  useEffect(() => {
    if (import.meta.env.VITE_PUBLISH_BASE) return;  // public build: no viewer delay, nothing to follow
    const onStorage = (e: StorageEvent) => {
      if (e.key !== VIEWER_KEY || detailRef.current?.viewer == null) return;
      const before = wantedDelay(), next = reloadViewer();
      setSetting(next);
      if (wantedDelay(next) === before) return;
      inflight.current?.abort(); detailRef.current = null; drawn.current = null;
      setDetail(null); setError(null);
      void load(true, true);
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [load]);
  const viewer = detail && detail.session === null ? detail.viewer ?? null : null;
  useEffect(() => { if (viewer) viewerMeta.current = { max: viewer.max_delay_s }; }, [viewer]);
  // the reference sheet belongs to the card on screen: when the card goes (state changed), the sheet goes; it opens only by a tap
  useEffect(() => { if (sheet === 'reference' && !(detail?.reference && detail.display?.state === 'unsupported')) setSheet(null); }, [detail, sheet]);
  // first open of a live game on a device that never synced: ask once (the entry in the status line stays for later)
  useEffect(() => {
    if (!detail || !viewer || !viewer.prompt || prompted.current || setting.synced || setting.asked || detail.game.status !== 'live' || sheet !== null) return;
    prompted.current = true; setSheet('sync');
  }, [detail, viewer, setting, sheet]);
  const live = detail ? detail.session_status !== 'ended' && (detail.game.status === 'live' || (detail.next !== null && detail.game.status !== 'final')) : false;
  // 30x check 11/22: a frozen screen whose game stops being live (end of a replay loop) returns to the latest view by itself — the
  // ▶ 라이브로 button only exists while the game is live, so the viewer would otherwise be left on a frozen screen without a way back
  useEffect(() => { if (frozen && !live) { setFrozen(null); setSel({ kind: 'latest' }); } }, [frozen, live]);
  // Polling follows the viewer's switch only; the game's LIVE status decides whether there is anything to poll.
  // Failed or slow answers never stop the loop: the next tick asks again, so the screen recovers without a reload.
  // A game the LIVE route answers for but that is not in play (scheduled, warm-up not started, delayed, suspended, or any state the
  // server names while the card is not live) is asked again slowly, with or without a display state, so the screen starts by itself.
  const waitingToStart = detail !== null && !live && detail.session === null && detail.game.status !== 'final'
    && (detail.display?.state === 'pregame' || detail.live_source === true);
  const pollMs = live ? (__PZ_STATIC__ ? manifestMs ?? CADENCE.live : CADENCE.live) : waitingToStart ? (__PZ_STATIC__ ? manifestMs ?? CADENCE.pregame : CADENCE.pregame) : __PZ_STATIC__ && (detail !== null || withdrawn || pendingSince !== null) ? manifestMs ?? CADENCE.live : 0;
  useEffect(() => {
    if (!auto || !pollMs) return;
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, pollMs);
    return () => window.clearInterval(id);
  }, [auto, pollMs, load]);
  // static mode: a screen pinned on a past pitch (auto off) still checks, including final-game corrections. If the server turns the
  // pinned pitch back to 확인 중, the same fail-closed rules blank it here too; the view stays pinned (no jump to the latest).
  const pinnedMs = __PZ_STATIC__ && !auto && pollMs > 0 ? manifestMs ?? PUBLISH_CONFIG.pinnedPollMs : 0;
  useEffect(() => {
    if (!pinnedMs) return;
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, pinnedMs);
    return () => window.clearInterval(id);
  }, [pinnedMs, load]);
  // first load failed (nothing to show yet): keep trying by itself, but only when the server could not be reached or failed (network
  // error / timeout = 0, 5xx). A refusal (423 embargo, 400, 404) is an answer, not an outage: it is shown as before and not retried.
  // A service notice (maintenance / data_hold) on a page that already shows a game: nothing received before it stays on screen
  // (recommendations, zone marks, pitches, sheets). Polling goes on; the next good answer draws the game again, and the change note
  // never names a recommendation from before the notice.
  const noticeHold = hold || error?.notice === true;
  useEffect(() => { if (noticeHold) { drawn.current = null; setFrozen(null); setPop(false); setSheet(s => s === 'sync' ? s : null); } }, [noticeHold]);
  // (a notice on an open page is retried the same way, so it clears by itself even where the game is no longer polled)
  const firstLoadRetry = (detail === null || noticeHold) && error !== null && (error.status === 0 || error.status >= 500);
  // static mode: a game with no file yet (404) is asked again slowly, so it appears by itself once it is published; and nothing is
  // asked while the tab is hidden (the retry starts again when it is visible)
  const missing = __PZ_STATIC__ && !withdrawn && detail === null && error !== null && error.status === 404;
  const holdHidden = __PZ_STATIC__ && hidden;
  useEffect(() => {
    if (!firstLoadRetry || holdHidden) return;
    const id = window.setTimeout(() => void load(true), CADENCE.retry);
    return () => window.clearTimeout(id);
  }, [firstLoadRetry, holdHidden, error, load]);
  // A missing file is not asked for ever: the gap doubles up to a maximum, and before each new try the lists of the dates the
  // publisher is following are looked at; when none of them names this game (a demo runs on it, it is long over, the number is
  // wrong) asking stops. The calm "지금 볼 수 없어요" stays, with the manual button.
  const missingTries = useRef(0);
  const [gaveUp, setGaveUp] = useState(false);
  useEffect(() => {
    if (!missing || holdHidden || gaveUp) return;
    const ctrl = new AbortController();
    const id = window.setTimeout(async () => {
      const soon = await isGamePublishedSoon(gamePk, ctrl.signal);
      if (ctrl.signal.aborted) return;
      if (soon === false) { setGaveUp(true); return; }
      missingTries.current += 1;
      void load(true);
    }, Math.min(PUBLISH_CONFIG.missingRetryMs * 2 ** missingTries.current, PUBLISH_CONFIG.missingMaxMs));
    return () => { window.clearTimeout(id); ctrl.abort(); };
  }, [missing, holdHidden, gaveUp, error, load, gamePk]);
  const askAgain = () => { missingTries.current = 0; setGaveUp(false); void load(true, true); };
  useEffect(() => { const id = window.setInterval(() => setNow(Date.now()), __PZ_STATIC__ ? PUBLISH_CONFIG.tickMs : 5000); return () => window.clearInterval(id); }, []);
  useEffect(() => {  // coming back to the tab: refresh at once when following
    // static mode: coming back (tab visible, network back, device awake) reads FIRST; until that read ends the screen says 확인 중
    // instead of judging lateness from what was on screen before
    const recheck = () => { if (__PZ_STATIC__ && pollMs && detailRef.current !== null) setChecking(true); };
    const onVis = () => { setHidden(document.hidden); if (!document.hidden) { setResumed(Date.now()); setNow(Date.now()); } if (!document.hidden && (auto || __PZ_STATIC__) && pollMs) { recheck(); void load(); }
      if (__PZ_STATIC__ && !document.hidden && detailRef.current === null) void load(true); };
    const onOnline = () => { if (__PZ_STATIC__ && document.hidden) return; if ((auto || __PZ_STATIC__) && (pollMs || detailRef.current === null)) { recheck(); void load(detailRef.current === null); } };
    document.addEventListener('visibilitychange', onVis); window.addEventListener('online', onOnline);
    // a device sleep is seen as the wall clock running ahead of the page's monotonic clock (only the difference of the two is used)
    let wall = Date.now(), mono = performance.now();
    const awake = !__PZ_STATIC__ ? 0 : window.setInterval(() => {
      const w = Date.now(), m = performance.now(), gap = (w - wall) - (m - mono); wall = w; mono = m;
      if (gap > 2 * PUBLISH_CONFIG.tickMs && !document.hidden && (auto || __PZ_STATIC__) && pollMs) { setResumed(Date.now()); setNow(Date.now()); recheck(); void load(); }
    }, PUBLISH_CONFIG.tickMs);
    return () => { document.removeEventListener('visibilitychange', onVis); window.removeEventListener('online', onOnline); window.clearInterval(awake); };
  }, [auto, pollMs, load]);

  // default selection once data arrives: live → follow latest; finished → first pitch
  useEffect(() => { if (detail && sel === null) setSel(live ? { kind: 'latest' } : detail.pitches.length ? { kind: 'pitch', key: detail.pitches[0].key } : { kind: 'latest' }); }, [detail, sel, live]);
  // a screen opened before play (or in a suspension) that the viewer never touched follows the latest pitch once the game is live
  const wasLive = useRef<boolean | null>(null);
  useEffect(() => {
    if (!detail) return;
    if (live && wasLive.current === false && !picked.current && sel?.kind === 'pitch') setSel({ kind: 'latest' });
    wasLive.current = live;
  }, [detail, live, sel]);
  // a pinned pitch that disappeared (demo seek back) falls back to latest
  useEffect(() => { if (detail && sel?.kind === 'pitch' && !detail.pitches.some(p => p.key === sel.key)) { setSel({ kind: 'latest' }); setSheet(s => s === 'detail' ? null : s); } }, [detail, sel]);
  const byIndex = useMemo(() => new Map(detail?.pitches.map(p => [p.index, p]) ?? []), [detail?.pitches]);
  // Live only (demo 2026-11-17; 15 s by owner 2026-11-19): when the plate appearance of the newest pitch ends (the file names the next
  // batter, or the half-inning break), keep that plate appearance on the stage (its batter, pitchers and thrown pitches; no next
  // recommendation) for RESULT_HOLD_MS while the live view and its swing attributes stay on, so a viewer who saw the result on the
  // broadcast can still check the pitch here. 15 s: World Series G5 feed, PA complete → next PA's first pitch within a half-inning
  // n=52, median 32.3 s, p10 25.9 s, min 24.9 s, so the hold ends ~10 s or more before the next first pitch. It ends early when a
  // pitch of the new plate appearance arrives; at a half-inning change it ends into the inning-break screen.
  // Not on the first file seen (nothing new happened on screen), never twice for the same pitch, not after a correction that removed
  // pitches, and it also starts when the PA switch arrives one file after its last pitch (the PA stayed on the stage until then). A reset (no file: another delay, a withdrawal), another game or leaving live clears it.
  const RESULT_HOLD_MS = 15000;
  const lastKeyNow = detail?.pitches.length ? detail.pitches[detail.pitches.length - 1].key : null;
  const curPaNow = detail?.current?.pa_key ?? null;
  const heldSeen = useRef<{ game: number; last: string; pa: string | null } | undefined>(undefined);
  const heldDone = useRef<string | null>(null);  // newest pitch whose hold already started (it never starts twice)
  const heldTimer = useRef<number | undefined>(undefined);
  const heldDetail = useRef(detail); heldDetail.current = detail;
  const [heldKey, setHeldKey] = useState<string | null>(null);
  useEffect(() => {
    const d = heldDetail.current;
    const release = () => { window.clearTimeout(heldTimer.current); heldTimer.current = undefined; setHeldKey(null); };
    if (!live || !d || lastKeyNow === null) { release(); heldSeen.current = undefined; heldDone.current = null; return; }
    const prev = heldSeen.current;
    heldSeen.current = { game: gamePk, last: lastKeyNow, pa: curPaNow };
    if (prev === undefined || prev.game !== gamePk) { release(); heldDone.current = null; return; }
    const lp = d.pitches[d.pitches.length - 1];
    if (curPaNow === lp.pa_key) { release(); return; }   // the stage is on the newest pitch's PA anyway (new pitch of the new PA)
    // only a FINISHED plate appearance is held: the newest pitch ended it (its event) or the file lists that PA as complete. A stage that
    // moves away from an unfinished PA (a feed or state change without a result) shows the new state at once.
    if (!lp.actual?.event && !d.pas.some(p => p.pa_key === lp.pa_key && p.complete)) { release(); return; }
    // the stage would move away from the newest pitch's PA: hold it once, if that PA was on the stage until now
    // (a new newest pitch: only when it came after the previous newest; a correction that removed pitches holds nothing)
    const prevIdx = prev.last === lastKeyNow ? -1 : d.pitches.findIndex(p => p.key === prev.last);
    const moved = prev.last !== lastKeyNow ? prevIdx >= 0 && prevIdx < d.pitches.length - 1 : prev.pa === lp.pa_key;
    if (!moved || heldDone.current === lastKeyNow) return;
    heldDone.current = lastKeyNow;
    window.clearTimeout(heldTimer.current);
    setHeldKey(lastKeyNow);
    const key = lastKeyNow;
    heldTimer.current = window.setTimeout(() => { heldTimer.current = undefined; setHeldKey(k => (k === key ? null : k)); }, RESULT_HOLD_MS);
  }, [live, lastKeyNow, curPaNow, gamePk]);
  useEffect(() => () => window.clearTimeout(heldTimer.current), []);
  // The sync sheet sits beside the page (not inside it) so it stays open, with its own state, while a changed delay reloads the game.
  const closeSync = () => { if (!readViewer().asked) setSetting(writeViewer({ asked: true })); closeSheet(); };
  const syncSheet = !import.meta.env.VITE_PUBLISH_BASE && sheet === 'sync' ? <div className="sv-sync-host"><SyncSheet gamePk={gamePk} delay={viewer?.delay_s ?? wantedDelay(setting)} max={viewerMeta.current?.max ?? null}
    synced={setting.synced} onDelay={changeDelay} onClose={closeSync} /></div> : null;
  if (!detail || noticeHold) {
    return <><div className="sv-page"><TopBar /><main className="sv-center">{!noticeHold && __PZ_STATIC__ && pendingSince !== null
      ? <p className="sv-muted" role="status">{performance.now() - pendingSince > staleAfterMs(pollMs) ? '갱신 지연 · 새 내용이 늦어지고 있어요.' : '확인 중 · 새 내용을 확인하고 있어요.'}</p>
      : error
      // public bundle: no file for a game (not published yet, or withheld while a demo runs on it) = not available now, said calmly
      ? __PZ_STATIC__ && error.status === 404 ? <div className="sv-unavailable"><p className="sv-muted" role="status">이 경기는 지금 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.</p>
        <div className="sv-row"><button className="sv-btn" onClick={askAgain}>다시 확인</button><a className="sv-btn" href={publicListHref()}>경기 목록</a></div>
        {!gaveUp && <p className="sv-muted sv-retrying">올라오면 자동으로 보여 드려요.</p>}</div>
      : <div className="sv-alert" role="alert"><p>{error.status === 423 ? '이 경기는 지금 기록을 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.' : error.text}</p>
        <div className="sv-row"><button className="sv-btn" onClick={() => void load(true, true)}>다시 시도</button><a className="sv-btn" href={__PZ_STATIC__ ? publicListHref() : '/watch'}>경기 목록</a></div>
        {firstLoadRetry && <p className="sv-muted sv-retrying">자동으로 다시 연결하고 있어요.</p>}</div>
      : <p className="sv-muted" role="status">경기 기록을 불러오는 중…</p>}</main></div>{syncSheet}</>;
  }

  const liveD = detail; const shownD = frozen ?? detail;
  const { game } = shownD;
  const verification = __PZ_STATIC__ ? gameCheck(health, gamePk) : { present: false, check: null };
  const awaitingFile = __PZ_STATIC__ && pendingSince !== null;
  // owner 11/23 (깜빡임): a newer file announced by health while the screen already shows a verified one is fetched in the background —
  // the verified screen stays (no 확인 중, nothing hidden, no scroll jump) until the new file lands or the wait turns into 갱신 지연
  const quietFetch = awaitingFile && appliedSequence.current !== null && performance.now() - pendingSince! <= staleAfterMs(pollMs);
  const maskPublished = __PZ_STATIC__ && ((awaitingFile && !quietFetch) || (verification.present && (!verification.check || checking)));

  const pitches = shownD.pitches;
  const lastPitch = pitches[pitches.length - 1] ?? null;
  const selectedPitch = sel?.kind === 'pitch' ? pitches.find(p => p.key === sel.key) ?? null : null;
  // Live state never follows the separate shownD selection. Polling and corrections continue.
  const heldPitch = live && heldKey !== null ? pitches.find(p => p.key === heldKey) ?? null : null;
  // a past pitch without its own ready record falls back to the bundled as-of record (same policy only, candidatesFor)
  const pastRec = (p: PitchRow): Rec | null => { if (p.rec?.status === 'ready') return p.rec; const c = candidatesFor(p, tabData.recs, livePolicy(shownD));
    return c.length ? { status: 'ready', reason: null, candidates: c, provenance: 'asof_replay', policy: p.rec?.policy ?? null } : p.rec; };
  const pitch: PitchRow | null = live && !frozen ? null : selectedPitch;
  const latest = pitch === null;  // a pin whose pitch is gone (demo seek back) shows latest until the effect resets it
  // Live display state (absent for archive / demo / older backends: then everything below behaves as before the contract).
  const disp = shownD.display ?? null;
  // Between innings the live route sends no situation, batter or pitcher. Nothing of the finished half is then presented as current:
  // no count, outs, runners or matchup, and no pitch cards of the last batter; only the score and which half just ended stay.
  const breakNow = latest && live && heldPitch === null && shownD.live_source === true && (shownD.phase === 'between_innings' || disp?.reason === 'between_innings');
  const ended = breakNow && lastPitch ? { inning: lastPitch.situation.inning, half: lastPitch.situation.half } : null;
  // What the stage shows
  const situation: Situation | null = latest ? (heldPitch ? heldPitch.situation : breakNow ? null : shownD.current?.situation ?? lastPitch?.situation ?? null) : pitch!.situation;
  const rec: Rec | null = maskPublished ? null : latest ? (live ? (heldPitch ? null : shownD.next) : lastPitch?.rec ?? null) : pastRec(pitch!);
  const actual: Actual | null = maskPublished ? null : latest ? lastPitch?.actual ?? null : pitch!.actual;
  const nextPhase = latest && live;
  const shownPitch = latest ? lastPitch : pitch;
  const paKey = latest ? (heldPitch ? heldPitch.pa_key : shownD.current?.pa_key ?? lastPitch?.pa_key ?? null) : pitch!.pa_key;
  const paPitches = breakNow || maskPublished ? [] : pitches.filter(p => p.pa_key === paKey);
  const endingShown = !maskPublished && shownPitch?.actual?.event ? shownD.pas.find(p => p.pa_key === shownPitch.pa_key && p.complete)?.result ?? null : null;
  const stateSlot = disp !== null && latest && heldPitch === null && (nextPhase || disp.state === 'pregame');
  const slotNumber = Number(/^\d+:\d+:(\d+)$/.exec(disp?.pitch_key ?? '')?.[1] ?? NaN);
  const nextNumber = nextPhase && !breakNow && heldPitch === null ? (stateSlot && Number.isInteger(slotNumber) ? slotNumber
    : shownD.current?.pa_key && lastPitch?.pa_key === shownD.current.pa_key ? lastPitch.pitch_number + 1 : 1) : null;
  const shownRec = stateSlot ? disp!.shown && rec?.status === 'ready' && rec.candidates.length > 0 : rec?.status === 'ready';
  const batterP = latest ? (heldPitch ? heldPitch.batter : breakNow ? null : shownD.current?.batter ?? lastPitch?.batter ?? null) : pitch!.batter;
  const pitcherP = latest ? (heldPitch ? heldPitch.pitcher : breakNow ? null : shownD.current?.pitcher ?? lastPitch?.pitcher ?? null) : pitch!.pitcher;
  // live route with nobody at the plate (before the first pitch, between innings): no matchup is named and no role is guessed
  const idle = latest && shownD.live_source === true && batterP === null && pitcherP === null;
  const side = batterP?.side === 'R' || batterP?.side === 'L' ? batterP.side : null;  // switch hitters: only the side actually taken; never guessed

  // Connection: the client's own polling (clientStale / error) and the server's upstream feed. With a display state the server part
  // comes from display.feed; the old `stale_s > 30` guess (which also fires in long breaks) is used only without one.
  const clientStale = live && auto && !hidden && lastOk !== null && now - Math.max(lastOk, resumed) > (__PZ_STATIC__ ? lostAfterMs(pollMs) : STALE_MS);
  const rawCheck = health?.gameChecks?.[String(gamePk)];
  const checkFeedFailed = rawCheck !== null && typeof rawCheck === 'object' && 'feed_ok' in rawCheck && rawCheck.feed_ok === false;
  const healthFeedFailed = verification.present && (checkFeedFailed || health?.failingGames.some(g =>
    g === gamePk || g === String(gamePk) || (g !== null && typeof g === 'object' && 'game_pk' in g && String(g.game_pk) === String(gamePk))));
  const feedDown = __PZ_STATIC__ && verification.present ? healthFeedFailed : disp ? !disp.feed.ok : live && (shownD.feed.ok === false || (shownD.feed.stale_s ?? 0) > 30);
  const stale = clientStale || feedDown;
  // static mode: the files answer, but nothing new has been published for longer than the threshold. "Is polled" is not "publishing is
  // expected": a game in play is expected to move; a game that has not started is expected to be published only once its scheduled
  // start has passed (before that the publisher may be idle on purpose, e.g. a day without a live game).
  //  - a game whose own status says it is in play: late when health.json itself is stale, or health.json is fresh and this file is behind it;
  //  - any other game that is polled (pregame, delayed start, postponed, doubleheader wait): only health.json is judged, and only once
  //    the scheduled start has passed (before that an idle publisher is normal). A fresh health.json is never late for such a game.
  const srv = __PZ_STATIC__ ? serverNow(clockWindowMs(pollMs)) : null;
  const startMs = game.start ? Date.parse(game.start) : NaN;
  const timeWarnings = !(fileStamp?.final && game.status === 'final') && (live || (waitingToStart && srv !== null && Number.isFinite(startMs) && srv.ms >= startMs));
  // DEMO: a frozen snapshot is old on purpose — no 갱신 지연 / 잠시 기다리는 중 from its age
  const lag = !__PZ_STATIC__ || DEMO ? null : verification.present ? (timeWarnings ? gameCheckLag(health, verification.check, srv) : null) : live ? publishLag(fileStamp, health, srv)
    : waitingToStart && srv !== null && Number.isFinite(startMs) && srv.ms >= startMs ? publishLag(null, health, srv) : null;
  const pendingLate = awaitingFile && !(fileStamp?.final && game.status === 'final') && performance.now() - pendingSince! > staleAfterMs(pollMs);
  const lateNow = __PZ_STATIC__ && (pendingLate || isPublishStale(lag, pollMs));
  const publishStale = lateNow && !checking;
  // B-5 (11/22): nothing new from the publisher for QUIET_S s (shorter than 갱신 지연): 라이브 → a quiet 잠시 기다리는 중 (never for a final game)
  const quietWait = F1.quietWait && __PZ_STATIC__ && !publishStale && game.status !== 'final' && lag !== null && lag.lagMs > QUIET_S * 1000;
  const publishChecking = !publishStale && ((awaitingFile && !quietFetch) || (verification.present && (!verification.check || checking)) || (lateNow && checking));
  // lag_s is a proven lower bound (never an estimate): floor it and say '넘게', never '약'
  // Viewer delay on (live route, feature on): how far behind this screen is shown, and whether thrown-pitch details are folded.
  // Folded until this device is synced: every realized pitch shownD of a LIVE game (type, speed, location, call/result of each
  // pitch card, the actual-pitch row of a selected pitch, the shownD sheet's actual part and the plate-appearance results).
  // Score, count, outs, runners, inning, the linescore and the players' game totals stay (PM decision).
  const behind = viewer ? Math.round(viewer.delay_s) : 0;
  const fold = viewer !== null && viewer.fold && !setting.synced && game.status === 'live';
  const openSync = () => openSheet('sync');
  const lagText = disp?.feed.delayed ? (disp.feed.lag_s !== null ? `${Math.floor(disp.feed.lag_s)}초 넘게 늦음` : '평소보다 늦음') : null;
  // D1 "no silent flip": server note first; otherwise compare with what this screen last drew for the same pending pitch.
  let clientNote: string | null = null;
  if (stateSlot) {
    // Slot identity: the server's pitch_key. An answer without one (e.g. a short 'waiting') is the SAME slot with nothing drawn, unless a
    // pitch was thrown since or the plate appearance / batter changed; only a different non-null key starts a new slot.
    const key = disp!.pitch_key, lastIdx = lastPitch?.index ?? -1;
    const who = shownD.current?.pa_key ?? (shownD.current?.batter ? String(shownD.current.batter.id) : '');
    const sig = shownRec ? rec!.candidates.map(c => `${c.pitch_type}@${c.zone_id ?? ''}`).join('|') : '';
    const top = shownRec ? pickText(rec!.candidates[0].pitch_label, rec!.candidates[0].zone_label) : '';
    let d = drawn.current;
    const other = !d || d.lastIdx !== lastIdx || (key !== null && d.key !== null && d.key !== key) || (key === null && who !== '' && d.who !== '' && d.who !== who);
    if (other) d = drawn.current = { key, lastIdx, who, seen: shownD, sig, top, prev: null };
    else if (d!.seen !== shownD) {  // once per received answer (safe under repeated renders)
      d!.seen = shownD; if (key !== null) d!.key = key; if (who !== '') d!.who = who;
      if (d!.sig !== sig) {
        const left = d!.sig !== '' ? { sig: d!.sig, top: d!.top } : d!.prev;
        d!.prev = left && left.sig !== sig ? left : null;
        d!.sig = sig; d!.top = top;
      }
    }
    if (d!.prev) clientNote = `${d!.sig ? '예측이 갱신됐어요' : '조금 전에 보여 드린 예측은 지금 표시하지 않아요'} (이전: ${d!.prev.top})`;
  }
  const ending = fold ? null : endingShown;
  // every published pitch of the at-bat on the main zone, G1 marks from the result. The ring belongs to the at-bat's NEWEST pitch only:
  // when that pitch has no coordinate or is being confirmed it is not drawn, and no older mark inherits its ring.
  const newestKey = paPitches[paPitches.length - 1]?.key ?? null;
  const thrown = nextPhase && !fold ? paPitches.filter(p => !unconfirmed(p) && p.actual?.x != null && p.actual?.z != null)
    .map(p => ({ x: p.actual!.x!, z: p.actual!.z!, number: p.pitch_number, kind: pitchKind(p.actual), latest: p.key === newestKey })) : [];
  // A2: a reference card rides only on the pending pitch's unsupported state; it is never drawn as a recommendation
  const reference = !maskPublished && stateSlot && latest && nextPhase && !shownRec && disp!.state === 'unsupported' ? shownD.reference ?? null : null;
  const fix = stateSlot ? disp!.correction : null;
  const fixPrev = previousText(fix?.previous ?? null);
  const note = maskPublished ? null : fix ? `${fix.message}${fixPrev ? ` (이전: ${fixPrev})` : ''}` : clientNote;
  // The client's own requests failing (or silent for too long): the slot says so instead of looking current.
  const clientLost = disp !== null && (error !== null || clientStale || (__PZ_STATIC__ && healthFeedFailed === true));
  const eff = !stateSlot ? null : clientLost ? CLIENT_LOST : publishStale ? PUBLISH_LATE : publishChecking ? PUBLISH_CHECKING
    : { state: disp!.state, label: disp!.label, message: disp!.message ?? (shownRec || disp!.state !== 'recommended' ? null : '예측을 준비하고 있어요.') };
  // candidates kept from the last answer while disconnected: still visible, dimmed, labelled as last received
  const held = stateSlot && shownRec && eff!.state === 'disconnected';
  const scored = situation ?? null;
  // between innings the card carries no score: the as-of linescore (runs up to the last published play) keeps it on screen
  const breakScore = (k: 'away' | 'home') => breakNow ? shownD.linescore_asof?.[k].r ?? lastPitch?.situation[`${k}_score`] ?? null : null;
  // a finished game's header always carries its final result (figure 11), whichever past pitch is selected below
  const atEnd = latest || game.status === 'final';
  const awayScore = atEnd && game.away_score !== null ? game.away_score : scored?.away_score ?? breakScore('away');
  const homeScore = atEnd && game.home_score !== null ? game.home_score : scored?.home_score ?? breakScore('home');

  // Historical shownD is keyed independently; selecting it never pauses live polling.
  const pick = (index: number) => { picked.current = true; const p = byIndex.get(index); if (!p) return;
    // owner 11/22 (5·7): a picked pitch is shown on the 예측 page as it was predicted then; a live game freezes until 라이브
    if (F1.tabs && F1.pause) { if (live) setFrozen(f => f ?? liveD); setSel({ kind: 'pitch', key: p.key }); if (view !== 'rec') goView('rec'); else window.scrollTo({ top: 0 }); return; }
    setSel({ kind: 'pitch', key: p.key }); openSheet('detail'); };
  const toLive = () => { setFrozen(null); setRecOpen(false); picked.current = false; setSel({ kind: 'latest' }); };
  const newSince = frozen ? Math.max(0, liveD.pitches.length - frozen.pitches.length) : 0;
  // ③ the recommendation entries (cells, 자세히, bottom 예측) open the CURRENT recommendation on a live game: a past pitch opened
  // from the at-bat list before must not stay selected behind them. A finished game keeps its browsed pitch (that is its current view).
  // owner 11/23: 자세히 shows the pitch on screen — a chosen past pitch stays chosen; only with no pitch chosen does it open the latest
  const openRec = () => { if (live && latest) { picked.current = false; setSel({ kind: 'latest' }); } openSheet('detail'); };
  // 11/22 tabs: on the live next pitch with a ready recommendation, 자세히 opens inside the card; elsewhere the shownD sheet as before
  const openRecInline = () => { if (F1.tabs && nextPhase && latest && rec?.status === 'ready') { if (!recOpen) holdLive(); setRecOpen(o => !o); } else openRec(); };
  const goLatest = () => { setFrozen(null); picked.current = false; setSel({ kind: 'latest' }); setAuto(true); void load(false, true); };
  // A past pitch of a live game: the recommendation the ledger recorded BEFORE that pitch, next to what was thrown. Never a score.
  const pastUnconfirmed = !nextPhase && unconfirmed(shownPitch);
  const recTitle = nextPhase ? `다음 공 예측${nextNumber ? ` · ${nextNumber}구` : ''}` : '지난 공 예측';
  const absentText = maskPublished ? '새 내용을 확인하고 있어요.' : pastUnconfirmed ? '확인 중이에요.' : nextPhase ? '예측 기록이 없어요.' : ABSENT_PAST;
  const actualTitle = nextPhase ? `직전 투구${lastPitch ? ` · ${inningText(lastPitch.situation)} ${lastPitch.pitch_number}구` : ''}` : `실제 투구${pitch ? ` · ${pitch.pitch_number}구` : ''}`;
  const top = situation?.half === 'Top';
  const info = (p: Person | null) => p ? shownD.players?.[String(p.id)] : undefined;
  const openPa = () => openSheet('pa');
  // fail closed: no candidate reaches the row, the drawing or any label unless this state may show it (an unconfirmed past pitch shows none)
  const mainRec = stateSlot ? (shownRec ? rec : null) : pastUnconfirmed ? null : rec;
  const cands = shownRec && mainRec ? mainRec.candidates : [];
  const venue = game.venue ?? '경기장 정보 없음';
  const when = `${shortGameDate(game.kst_date)} ${kstTime(game.start)}${game.double_header ? ` · ${game.game_number}차전` : ''}`;
  const phase = shownD.current?.phase ?? shownD.phase ?? null;
  const emptyRec = maskPublished ? '확인 중' : nextPhase ? (phase === 'between_innings' ? '공수 교대 중' : phase === 'resolving' ? '결과 확인 중' : '예측 대기 중') : '이 시점에는 예측이 없어요';
  // Result hold (proposal 2026-11-19 ①): the finished PA's last pitch in one line where the three '—' cells were (same height); it opens
  // that pitch's sheet. Not while masked, folded (viewer delay, not synced) or while the pitch is still being confirmed: then the '—' cells stay.
  const heldActual = heldPitch && !maskPublished && !fold && !unconfirmed(heldPitch) ? heldPitch.actual : null;
  const heldEnding = heldActual ? shownD.pas.find(p => p.pa_key === heldPitch!.pa_key && p.complete)?.result ?? null : null;
  const list = heldPitch && heldActual ? <ResultCard pitch={heldPitch} actual={heldActual} ending={heldEnding} speed={mph => speedText(mph, unit)} unit={unit} onOpen={() => pick(heldPitch.index)} />
    : <RecList rec={mainRec} emptyText={stateSlot ? eff!.message ?? eff!.label : emptyRec} onOpen={openRec} lead={held ? '마지막으로 받은 ' : ''} absentText={absentText} />;
  // 주요 장면 (D5-①): the finished PA's line under the result card, only for a top-10 PA that the published game has already finished
  const heldPa = heldPitch && heldEnding ? shownD.pas.find(p => p.pa_key === heldPitch.pa_key && p.complete) ?? null : null;
  const heldScene = scenes && heldPa ? scenes.scenes.find(s => s.at_bat === heldPa.at_bat_number) ?? null : null;
  const doneAbs = new Set(shownD.pas.filter(p => p.complete).map(p => p.at_bat_number));
  const openScene = (rank: number) => { setSceneRank(rank); openSheet('scene'); };
  const shownScene = scenes && sceneRank !== null ? scenes.scenes.find(s => s.rank === sceneRank && doneAbs.has(s.at_bat)) ?? null : null;
  // D7 8-3 ② (off unless picked): the top-1 recommendation before the pitch and the thrown pitch, in words that read as neither right nor wrong
  const heldTop1 = F1.recVsActual && heldPitch && heldActual && heldPitch.rec?.status === 'ready' ? heldPitch.rec.candidates[0] ?? null : null;
  // fan review 11/22 #1: say whose recommendation and of what ("예측 · 2구" alone did not)
  const recHead = stateSlot ? (held ? `마지막으로 받은 예측${nextNumber ? ` · ${nextNumber}구` : ''}` : nextNumber ? `${nextNumber}구 예측` : '다음 공 예측') : nextPhase ? (nextNumber ? `${nextNumber}구 예측` : '다음 공 예측')  // owner 11/22: "N구 예측"
    : `${recTitle}${shownPitch ? ` · ${shownPitch.pitch_number}구` : ''}`;
  // a past pitch (finished game): the thrown pitch named in one line under the row; folded / being confirmed say so and nothing more
  const actualLine = !shownPitch || !actual ? <span className="sv-muted">실제 투구 기록 없음</span>
    : fold ? <><span className="sv-al-label">실제 투구 · {shownPitch.pitch_number}구</span><button className="sv-fold" onClick={openSync} aria-haspopup="dialog">{FOLD_TEXT}</button></>
    : pastUnconfirmed ? <><span className="sv-al-label">실제 투구 · {shownPitch.pitch_number}구</span><span className="sv-muted">확인 중 · 이 공을 다시 확인하고 있어요.</span></>
    : <><span className="sv-al-label">실제 투구 · {shownPitch.pitch_number}구 · {shownPitch.batter.name ?? '타자 미확인'}</span>
      <G1Dot kind={pitchKind(actual)} n={shownPitch.pitch_number} /><b className="sv-actual-name">{actual.pitch_label}</b>
      {actual.speed_mph !== null && <small>{speedText(actual.speed_mph, unit)}</small>}<span>· {ending ?? actual.result_label}</span>
      {settledCorrection(shownPitch.receipt) && <span className="sv-tag" role="note" aria-label="기록이 정정된 공">정정됨</span>}</>;
  // live and not masked: the newest pitch, or an empty key when no pitch is thrown yet; masked (확인 중) or not live: no attributes at all
  const pzLast = !nextPhase || maskPublished ? {} : !lastPitch ? { 'data-pz-last-key': '' } : { 'data-pz-last-key': lastPitch.key, 'data-pz-last-kind': unconfirmed(lastPitch) ? '' : pzKind(lastPitch.actual),
    'data-pz-last-side': lastPitch.batter?.side === 'R' || lastPitch.batter?.side === 'L' ? lastPitch.batter.side : '' };
  const top1 = !nextPhase && !pastUnconfirmed && rec?.status === 'ready' ? rec.candidates[0] ?? null : null;
  const pairText = top1 && actual && !fold ? { rec: pickText(top1.pitch_label, top1.zone_label), actual: pickText(actual.pitch_label, actual.zone_label) } : null;
  // New as-of player context is optional; absent fields stay unknown rather than guessed from an incomplete pitch list.
  const context = latest ? shownD.current_context : pitch?.context;
  // Fable eye 11/22 (필수 1): while a result is held, the names are the held pitch's players but current_context already belongs to the
  // next pitch — show a today line only when it is the same player (else no line, never another player's numbers)
  const sameP = (a: { id?: number | null } | null | undefined, b: { id?: number | null } | null | undefined) => !heldPitch || (a?.id != null && a.id === b?.id);
  const boxB = heldPitch?.batter?.id != null ? [...(context?.boxscore?.away?.batters ?? []), ...(context?.boxscore?.home?.batters ?? [])].find(x => x.id === heldPitch.batter!.id) : undefined;
  const bt = sameP(heldPitch?.batter, shownD.current?.batter) ? context?.batter : boxB ? { at_bats: boxB.at_bats, hits: boxB.hits, on_base: boxB.on_base } : undefined;
  const pt = sameP(heldPitch?.pitcher, shownD.current?.pitcher) ? context?.pitcher : undefined;
  const isDemo = shownD.session !== null;
  // Header status button (UI 10/26 ⑥): ONE marker tells the whole state; its sentence opens in a bubble. Order: connection trouble,
  // then a late feed, then a changed recommendation, then the pending pitch's own state. A healthy finished game has no marker.
  const recvTime = lastOk !== null ? ` · ${kstTime(new Date(lastOk).toISOString())} 수신` : '';
  const chip: { label: string; tone: Tone; state: string; lines: string[] } | null = (() => {
    if (DEMO) return { label: '데모 보기', tone: 'checking', state: 'demo', lines: ['교수 데모용 고정 화면이에요. 경기 후반 한 시점에 멈춰 있고, 부스 재생과는 따로예요.'] };
    if (stateSlot) {
      const e = eff!, lines = [e.message, note].filter((x): x is string => !!x);
      if (e.state === 'disconnected') return { label: e.label, tone: 'lost', state: e.state, lines: [...lines, `${e.label}${recvTime}`] };
      if (e.state === 'feed_delayed') return { label: e.label, tone: 'late', state: e.state, lines: lagText ? [...lines, lagText] : lines };
      if (note) return { label: '예측 바뀜', tone: 'fixed', state: e.state, lines };
      if (e.state === 'unsupported' || e.state === 'unavailable') return { label: e.label, tone: 'none', state: e.state, lines };
      if (e.state === 'pregame') return { label: e.label, tone: 'pregame', state: e.state, lines };
      // design 09: an inning break of a game in play still reads 라이브 (row 3 says 공수 교대)
      if (quietWait && (e.state === 'recommended' || e.state === 'waiting')) return { label: QUIET.label, tone: 'checking', state: e.state, lines: [QUIET.message] };
      if (e.state === 'recommended' || (e.state === 'waiting' && breakNow)) return { label: '라이브', tone: 'live', state: e.state, lines: lines.length ? lines : [`실시간으로 받는 중${recvTime}`] };
      // unconfirmed, waiting, ledger_unavailable: the neutral 확인 중 look with the server's own words
      return { label: e.label, tone: 'checking', state: e.state, lines };
    }
    // no display state (archive, demo, older server): the transport story as before the contract
    if (error || feedDown || clientStale || disp?.state === 'disconnected') return { label: '연결 끊김', tone: 'lost', state: 'disconnected', lines: [error?.text ?? CLIENT_LOST.message] };
    if (game.status === 'final') return null;
    if (maskPublished || publishChecking) return { label: '확인 중', tone: 'checking', state: 'unconfirmed', lines: [PUBLISH_CHECKING.message] };
    if (isDemo && shownD.session_status === 'ended') return { label: '시연 종료', tone: 'checking', state: 'demo', lines: ['지난 경기 시연이 끝났어요.'] };
    if (isDemo && shownD.session_status === 'ready') return { label: '시연 대기', tone: 'checking', state: 'demo', lines: ['지난 경기 시연이 곧 시작돼요.'] };
    if (isDemo && shownD.session_status === 'paused') return { label: '일시정지', tone: 'checking', state: 'demo', lines: ['지난 경기 시연이 멈춰 있어요.'] };
    if (publishStale || (live && (!auto || behind > 0 || !!lagText || disp?.state === 'feed_delayed')))
      return { label: publishStale ? PUBLISH_LATE.label : '딜레이', tone: publishStale ? 'lost' : 'late', state: publishStale ? PUBLISH_LATE.state : 'delayed', lines: [publishStale ? PUBLISH_LATE.message : lagText ?? (behind > 0 ? `${behind}초 늦춰 보여 드리고 있어요.` : '새 내용이 늦어지고 있어요.')] };
    if (live && quietWait && !isDemo) return { label: QUIET.label, tone: 'checking', state: 'waiting', lines: [QUIET.message] };
    if (live) return isDemo ? { label: '시연 중', tone: 'checking', state: 'demo', lines: ['지난 경기를 다시 보여 드리고 있어요.'] } : { label: '라이브', tone: 'live', state: 'live', lines: [`실시간으로 받는 중${recvTime}`] };
    return game.status === 'scheduled' ? { label: '경기 전', tone: 'pregame', state: 'pregame', lines: ['경기 시작 전이에요.'] } : { label: '대기', tone: 'checking', state: 'waiting', lines: ['지금은 경기가 진행 중이 아니에요.'] };
  })();
  const canRetry = !!(error || stale || publishStale);
  const playerRole = F1.tabs ? (pRole === 'p' ? 'p' : 'b') : sheet === 'player-p' ? 'p' : 'b';
  const playerH = F1.tabs ? pRole === 'h' : sheet === 'player-h';
  const sheetPerson = playerRole === 'b' ? batterP : pitcherP;
  const sheetInfo = info(sheetPerson);
  // the guarded current probability (undecided sheet, content unchanged): reached from the 기록 sheet until its place is designed
  const canShowWe = latest && !maskPublished && !error && !stale && !publishChecking && !lateNow && phase === 'pre_pitch' && disp?.state === 'recommended' && disp.shown && shownD.win_expectancy?.status === 'available';
  // sheets that list the game's record follow the pitch on screen: a past pitch shows the record up to that pitch, the latest shows all
  const asOfIndex = latest ? null : (shownPitch?.index ?? null);
  // public bundle: a published replay of a finished game that carries no win-expectancy slot shows no 승리확률 entry at all
  const weKnown = !__PZ_STATIC__ || live || shownD.win_expectancy != null;
  const homeProbability = canShowWe && typeof shownD.win_expectancy?.home_probability === 'number' && Number.isFinite(shownD.win_expectancy.home_probability)
    && shownD.win_expectancy.home_probability >= 0 && shownD.win_expectancy.home_probability <= 1 ? shownD.win_expectancy.home_probability : null;
  const shownLine = latest || game.status === 'final' ? shownD.linescore_asof : pitch?.line;
  const abbr = (k: 'away' | 'home') => game[`${k}_abbr`] || short(game[`${k}_team`]);
  // a finished game's own header (figure 11) only while the screen is at the end; a selected past pitch keeps its as-of header
  const isFinal = game.status === 'final';
  const pregameHead = !isFinal && !situation && !ended && (game.status === 'scheduled' || disp?.state === 'pregame') && pitches.length === 0;
  const curInning = isFinal ? null : (situation ?? ended)?.inning ?? null;
  // the half after the one that just ended, and its first batter only when the server names it (never guessed)
  const nextHalf = ended ? (ended.half === 'Top' ? `${ended.inning}회말` : `${ended.inning + 1}회초`) : '';
  const batterLine = bt ? <>{part(bt.at_bats, '타수')} {part(bt.hits, '안타')}</> : null;
  const pitcherLine = pt ? <>{typeof pt.pitches === 'number' ? <><b>{pt.pitches}</b>구</> : '—'}</> : null;

  // cards as values so the booth layouts (f1Flags BOOTH) can place them in columns; the phone keeps the order game · players · stage
  const cGame = <section className="f1-card f1-game" aria-label="경기 상황">
      {pregameHead ? <p className="f1-pregame">{kstTime(game.start)} 시작</p>
        : <LineScore ls={shownLine} away={abbr('away')} home={abbr('home')} awayR={awayScore} homeR={homeScore} current={curInning} onOpen={() => openSheet('box')} />}
      {!pregameHead && situation && !(isFinal && latest) && <Bso s={situation} />}
    </section>;
  const cMu = breakNow ? <section className="sv-mu line f1-card" aria-label="경기 상황"><p>공수 교대 · 다음 {nextHalf} 첫 타자 <b>—</b></p>
        <button className="f1-pa-open f1-break-pa" onClick={openPa} aria-haspopup="dialog" aria-label={`${ended ? `${inningText(ended)} 종료. ` : ''}지난 타석 선택`}>지난 타석 ▾</button></section>
      : !pregameHead && !(isFinal && latest) && (situation || idle) ? <section className="sv-mu f1-card f1-players" aria-label="타자와 투수">{idle && !situation ? <p className="sv-mu-idle">타자·투수 정보 없음</p> : <>
        <Who role="b" person={batterP} hand={batterP?.side} team={top ? game.away_team : game.home_team} today={batterLine} onOpen={() => F1.tabs ? (setPRole('b'), goView('player')) : openSheet('player-b')} />
        <Who role="p" person={pitcherP} hand={pitcherP?.hand} team={top ? game.home_team : game.away_team} today={pitcherLine} onOpen={() => F1.tabs ? (setPRole('p'), goView('player')) : openSheet('player-p')} />
      </>}</section> : null;
  const stageH2 = <h2 className="sr-only">{stateSlot && !shownRec ? `다음 공: ${eff!.label}` : held ? '마지막으로 받은 예측' : '예측'}</h2>;
  const cZone = <div className="f1-card f1-zone">
      <div className={`sv-zrow ${side === 'L' ? 'bat-l' : 'bat-r'}${held ? ' held' : ''}`} {...pzLast}>
        {side && <div className={`sv-batband ${side === 'R' ? 'at-right' : 'at-left'}`}><Silhouette side={side} /></div>}
        <div className="sv-zone-wrap">
          <Zone bounds={shownD.zone_bounds} candidates={cands} actuals={thrown}
            actual={!nextPhase && !fold && !pastUnconfirmed && actual && actual.x !== null && actual.z !== null ? { x: actual.x, z: actual.z } : null} actualLabel=""
            actualKind={pitchKind(actual)} actualNumber={!nextPhase && shownPitch ? shownPitch.pitch_number : null}
            setupX={!nextPhase && !fold ? actual?.setup_x ?? null : null} />
        </div>
      </div>
      </div>;
  const cRec = <section className={`f1-card sv-rec-sec sv-pitch-summary${heldPitch ? ' f1-hold' : ''}${stateSlot ? ' sv-state' : ''}${held ? ' held' : ''}`} data-state={stateSlot ? eff!.state : undefined} data-pitch={stateSlot ? disp!.pitch_key ?? undefined : undefined} aria-label="예측">
        {!heldPitch && <div className="sv-rec-head"><h2>{recHead}</h2>
          <button className="sv-state-more" onClick={openRecInline} aria-expanded={F1.tabs && nextPhase && latest && rec?.status === 'ready' ? recOpen : undefined} aria-label="예측 자세히 보기">자세히{F1.tabs && nextPhase && latest && rec?.status === 'ready' ? (recOpen ? ' ▴' : ' ▾') : ''}</button></div>}
        {!(F1.tabs && recOpen && nextPhase && latest && !heldPitch && rec?.status === 'ready') && list}
        {heldTop1 && <p className="f1-rec-vs">던지기 전 예측 1순위 <b>{heldTop1.pitch_label}</b> · 실제 <b>{heldActual!.pitch_label}</b></p>}
        {heldScene && <SceneLine scene={heldScene} onOpen={() => openScene(heldScene.rank)} />}
        {F1.tabs && recOpen && nextPhase && latest && !heldPitch && rec?.status === 'ready' && <div className="f1-rec-more"><p className="f1-tc-note f1-rm-pct">%는 모델이 이 공을 고른 비율</p><ol className="f1-rm-list">{[...rec.candidates].sort((x, y) => x.rank - y.rank).slice(0, 3).map(c => <li key={c.rank}>
            <span className="f1-rm-rank"><RankSq rank={c.rank} /></span><b>{c.pitch_label}</b>
            {c.probability !== null && <span className="f1-rm-share"><i><i style={{ width: `${Math.max(0, Math.min(1, c.probability)) * 100}%` }} /></i>{Math.round(c.probability * 100)}%</span>}</li>)}</ol>

          <div className="sv-row f1-rm-links"><button className="sv-link-text" onClick={() => openSheet('info')}>기준과 한계 · 모델 소개</button></div></div>}
        {stateSlot && <div className="sv-state-live sr-only" role="status"><p className="sv-state-msg">{eff!.message}</p><p className="sv-state-note" data-kind={note ? (fix ? fix.kind : 'client') : undefined}>{note}</p></div>}
        {reference && eff!.state === 'unsupported' && !note && <ReferenceTeaser card={reference} onOpen={() => openSheet('reference')} />}
        {!nextPhase && <p className="sv-actual-line">{actualLine}</p>}
      </section>;
  const cTl = paKey && !breakNow && !pregameHead && <section className="sv-atbat f1-card f1-timeline" aria-label="이번 타석">
        <div className="sv-atbat-head"><h2><button className="f1-pa-open" onClick={openPa} aria-haspopup="dialog"
          aria-label={`이번 타석 ${paPitches.length}구 던짐${situation ? `, ${inningText(situation)}` : ''}. 타석 선택`}>이번 타석 · {paPitches.length}구 ▾</button></h2></div>
        <ol className="sv-atbat-list f1-newest-first" id="sv-atbat-list">
          {nextPhase && nextNumber && !frozen && paKey === shownD.current?.pa_key && <li className="next"><span className="f1-tl-dot" aria-hidden="true" /><span>{nextNumber}구 · 다음 공 기다리는 중</span></li>}
          {[...paPitches].reverse().map(p => { const on = selectedPitch?.key === p.key; const a = p.actual; const un = unconfirmed(p); const fixed = !un && settledCorrection(p.receipt);
            if (fold) return <li key={p.key}><button className="sv-pitch-row folded" aria-pressed={on} onClick={openSync} aria-haspopup="dialog" aria-label={`${p.pitch_number}구, ${FOLD_TEXT}`}>
              <G1Dot kind="unknown" n={p.pitch_number} /><span className="sv-pr-what">{FOLD_TEXT}</span></button></li>;
            const paEnd = !un && a?.event ? shownD.pas.find(q => q.pa_key === p.pa_key && q.complete)?.result ?? null : null; const hl = heldPitch?.key === p.key;
            return <li key={p.key} className={hl ? 'f1-hl' : undefined}><button className={`sv-pitch-row${un ? ' unconfirmed' : ''}`} aria-pressed={on} onClick={() => pick(p.index)} aria-haspopup="dialog"
              aria-label={un ? `${p.pitch_number}구 확인 중` : `${p.pitch_number}구 ${a?.pitch_label || '구종 미상'}${a?.speed_mph != null ? ` ${speedText(a.speed_mph, unit)}` : ''} ${a?.result_label ?? ''}${fixed ? ', 기록이 정정된 공' : ''}`}>
              <G1Dot kind={un ? 'unknown' : pitchKind(a)} n={p.pitch_number} />
              {un ? <span className="sv-pr-what" aria-hidden="true">확인 중</span> : <span className="sv-pr-what" aria-hidden="true"><b>{a?.pitch_label || '구종 미상'}</b>{a?.speed_mph != null && <small>{speedText(a.speed_mph, unit)}</small>}</span>}
              {fixed && <i className="sv-card-tag" aria-hidden="true">정정됨</i>}
              <span className="sv-pr-res" aria-hidden="true">{un ? '–' : paEnd || a?.result_label || '기록 없음'}</span></button></li>; })}
        </ol>
        {heldScene && <SceneLine scene={heldScene} onOpen={() => openScene(heldScene.rank)} />}
      </section>;
  // the rotated monitor (l3) gives the card the full width: a wider drawing, same height
  const bwpWide = !boothWide && typeof matchMedia === 'function' && matchMedia('(orientation: portrait) and (min-width: 1000px) and (min-height: 1600px)').matches;
  const cWp = BOOTH.wp && tabData.we && !maskPublished && !fold ? <div className="f1-bwp"><WinProbCard detail={shownD} we={tabData.we} scenes={scenes} homeShort={short(game.home_team)} awayShort={short(game.away_team)} onAnalysis={() => openSheet('analysis')} vw={bwpWide ? 560 : undefined} /></div> : null;
  const cards: Record<BoothCard, ReactNode> = { game: cGame, mu: cMu, zone: <>{stageH2}{cZone}</>, rec: heldPitch && heldActual ? null : cRec, tl: cTl, wp: cWp };
  const bcols = boothWide ? boothCols(BOOTH.layout, BOOTH.wp) : null;
  // owner 11/22 (B): the bottom tabs switch the page in place — header and tab bar stay, no 닫기; sheets only for drill-downs
  // owner 11/23: the line score lives in the shared game card (cGame) at the top of every tab, so 경기 starts with the win-probability card
  const gameInner = <>
      {tabData.we && !maskPublished && !fold && <WinProbCard detail={shownD} we={tabData.we} scenes={scenes} homeShort={short(game.home_team)} awayShort={short(game.away_team)} onAnalysis={() => openSheet('scenes')} onScene={rank => { setSceneRank(rank); openSheet('scene'); }} abbr={{ away: abbr('away'), home: abbr('home') }} />}
      {!maskPublished && !fold && <PitchLog detail={shownD} speed={mph => speedText(mph, unit)} onPick={pick} pack={tabData.recs} />}
      {!maskPublished && <div className="f1-tab-rows">{(['박스', '라인업'] as const).map(k => <button key={k} onClick={() => { setRecordTab(k); openSheet('history'); }}><b>{k}</b><small>{k === '박스' ? '타자 타수·안타·출루 / 투수 투구·삼진·자책' : '양 팀 1~9번'}</small><span aria-hidden="true">›</span></button>)}</div>}
  </>;
  const playerInner = <>
      <div className="sv-tabs" role="tablist" aria-label="선수">{(F1.tabs ? ['p', 'b', 'h'] as const : ['b', 'p'] as const).map(r => <button key={r} role="tab" aria-selected={(playerH ? 'h' : playerRole) === r} onClick={() => F1.tabs ? setPRole(r) : setSheet(r === 'b' ? 'player-b' : r === 'p' ? 'player-p' : 'player-h')}>{r === 'b' ? '타자' : r === 'p' ? '투수' : '맞대결'}</button>)}</div>
      {playerH ? <div role="tabpanel"><HeadToHead pitcherId={pitcherP?.id ?? null} batterId={batterP?.id ?? null} pitcherName={pitcherP?.name ?? null} batterName={batterP?.name ?? null} mu={tabData.mu} /><HeadToHeadToday detail={shownD} pitcherId={pitcherP?.id ?? null} batterId={batterP?.id ?? null} speed={mph => speedText(mph, unit)} /></div> :
      <div role="tabpanel">
      <p className="sv-player-title"><b>{sheetPerson?.name ?? '—'}</b> <span>{short((playerRole === 'b') === top ? game.away_team : game.home_team)}{handText(playerRole === 'b' ? batterP?.side : pitcherP?.hand, playerRole) ? ` · ${handText(playerRole === 'b' ? batterP?.side : pitcherP?.hand, playerRole)}` : ''}</span></p>
      <dl className="sv-dl sv-facts"><dt>등번호</dt><dd>{sheetInfo?.jersey ? `#${sheetInfo.jersey}` : '—'}</dd>
        <dt>{/* publisher pregame stats: MLB StatsAPI byDateRange gameType=R, as_of = 5차전 전 (11/23 확인) */ sheetInfo?.stats_as_of?.slice(0, 4) ? `${sheetInfo.stats_as_of.slice(0, 4)} 정규시즌` : '시즌'}</dt><dd>{playerRole === 'b' ? `AVG ${stat3(sheetInfo?.avg)} · OPS ${stat3(sheetInfo?.ops)}` : `ERA ${era(sheetInfo?.era)}`}</dd>
        <dt>{latest ? '이 경기' : '이 경기 · 이 공 전'}</dt><dd>{playerRole === 'b' ? [part(bt?.at_bats, '타수'), part(bt?.hits, '안타'), part(bt?.on_base, '출루')].join(' · ') : [part(pt?.pitches, '구'), part(pt?.strikeouts, '삼진'), part(pt?.earned_runs, '자책')].join(' · ')}</dd></dl>
      {/* owner 11/22: 오늘 구종별 위치 → 구속 → 비율 */}
      {F1.tabs && playerRole === 'p' && !maskPublished && <><PitchMap detail={shownD} pitcherId={pitcherP?.id ?? null} upto={asOfIndex} /><SpeedBands detail={shownD} pitcherId={pitcherP?.id ?? null} upto={asOfIndex} unit={unit} /></>}
      {playerRole === 'p' && !maskPublished && (F1.tabs ? <PitcherMix detail={shownD} pitcherId={pitcherP?.id ?? null} side={batterP?.side ?? null} mu={tabData.mu} upto={asOfIndex} />
        : <MixTable detail={shownD} pitcherId={pitcherP?.id ?? null} fold={fold} heading asOfIndex={asOfIndex} />)}
      {F1.tabs && playerRole === 'b' && <><BatterToday detail={shownD} batterId={batterP?.id ?? null} batterName={batterP?.name ?? null} />
        <BatterSplits batterId={batterP?.id ?? null} pitcherId={pitcherP?.id ?? null} pitcherHand={pitcherP?.hand ?? null} pitcherName={pitcherP?.name ?? null}
          season={sheetInfo ? { avg: sheetInfo.avg ?? null, ops: sheetInfo.ops ?? null } : null} mu={tabData.mu} opponent={short(top ? game.home_team : game.away_team)} /></>}
      {!F1.tabs && <p className="sv-muted">{sheetInfo?.stats_as_of ? '시즌 기록은 이 경기 전까지예요' : '시즌 기록 기준일 미확인'}. 오늘 기록은 지금까지 나온 공식 기록으로 계산했어요. 나중에 기록이 고쳐지면 달라질 수 있어요. {playerRole === 'b' ? '출루는 안타+볼넷+몸에 맞는 공이에요. ' : ''}확인되지 않은 값은 —로 표시해요.</p>}
      </div>}
  </>;
  const tabOn = F1.tabs ? (view === 'game' ? 1 : view === 'player' ? 2 : 0) : 0;
  return <SheetFrame.Provider value="c"><div className={`sv-page sv-game f1 f1-bl-${BOOTH.layout}${BOOTH.wp ? ' f1-bwp-on' : ''}${BOOTH.big ? ' f1-big' : ''}${F1.tabs && recOpen && nextPhase && latest ? ' f1-recopen' : ''}${F1.tabs && view !== 'rec' ? ' f1-inview' : ''}${disp ? ' has-display' : ''}`}>
    <header className="sv-hd">
      <div className="sv-hd-1">
        <a className="sv-back" href={backHref()} aria-label="경기 목록으로"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" /></svg></a>
        <h1 className="sv-hd-score" aria-label={pregameHead ? `${short(game.away_team)} 대 ${short(game.home_team)}, ${kstTime(game.start)} 시작 (한국 시간)` : `${short(game.away_team)} ${awayScore ?? '–'} 대 ${homeScore ?? '–'} ${short(game.home_team)}. ${venue}, ${when} 한국 시간`}>
          {pregameHead ? <span aria-hidden="true"><span className="t">{abbr('away')}</span><i>vs</i><span className="t">{abbr('home')}</span></span>
            : <span aria-hidden="true"><span className="t">{abbr('away')}</span><strong>{awayScore ?? '–'}</strong><i>:</i><strong>{homeScore ?? '–'}</strong><span className="t">{abbr('home')}</span></span>}
        </h1>
        {F1.tabs && F1.pause && live && (frozen || pitch) ? <button className="sv-hd-status f1-paused" data-tone="paused" onClick={toLive} aria-label={`멈춘 화면${newSince ? `, 새 공 ${newSince}개` : ''}. 라이브로 돌아가기`}>
          <i aria-hidden="true" /><span className="f1-pz-go">▶ 라이브로</span>{newSince ? <span className="f1-pz-new">새 공 {newSince}</span> : null}</button>
        : chip && <button className="sv-hd-status" data-tone={chip.tone} data-state={chip.state} aria-expanded={pop} aria-controls="sv-status-pop" onClick={() => F1.tabs && F1.pause && live && chip.tone === 'live' ? (setPop(false), setFrozen(liveD)) : setPop(o => !o)}
          aria-label={`${isDemo ? '지난 경기 시연. ' : ''}상태: ${chip.label}. 설명 보기`}><i aria-hidden="true" /><span aria-live="polite">{chip.label}</span></button>}
        {!chip && isFinal && <span className="sv-hd-status f1-final" data-tone="final">경기 종료</span>}
      </div>
    </header>
    {chip && pop && <StatusPop chip={chip} onClose={() => setPop(false)} onRetry={canRetry ? () => void load(true, true) : null} busy={busy} />}
    {F1.tabs && view !== 'rec' ? <main className={`sv-live-layout f1-view f1-view-${view}`} aria-label={view === 'game' ? '경기' : '선수'}>{cGame}<div className="sv-sheet frame-c full f1-inplace"><div className="sv-sheet-body">{view === 'game' ? gameInner : playerInner}</div></div></main> :
    <main className={`sv-live-layout${bcols ? ' f1-bcols' : ''}`}>
    {/* F1 (owner 11/21): cards top to bottom — game (box score · runners · count, no inning words), players, zone, recommendation, at-bat timeline */}
    {bcols ? bcols.map((col, i) => <div key={i} className={`f1-bcol${col.flat().includes('zone') ? ' f1-bcol-zone' : ''}`}>{col.map(r => r.length > 1
        ? <div key={r.join('+')} className={`f1-brow${r.includes('tl') ? ' has-tl' : ''}`}>{r.map(k => <Fragment key={k}>{cards[k]}</Fragment>)}</div>
        : <Fragment key={r[0]}>{cards[r[0]]}</Fragment>)}</div>)
      : <>{cGame}{cMu}
    <section className="sv-stage" aria-label={nextPhase ? '다음 공 예측' : '이 공의 예측과 실제 투구'}>
      {stageH2}
      {cZone}
      {!(heldPitch && heldActual) && cRec}
      {cTl}
    </section>
    {cWp}</>}
    </main>}
    {F1.tabs ? <nav className="sv-bottom-nav" aria-label="화면 메뉴">
      <button className={tabOn === 0 ? 'f1-on' : undefined} aria-current={tabOn === 0 ? 'page' : undefined} onClick={() => { goView('rec'); if (frozen) toLive(); else if (!latest) goLatest(); }}><NavIcon d={ICONS.rec} />예측</button>
      <button className={tabOn === 1 ? 'f1-on' : undefined} aria-current={tabOn === 1 ? 'page' : undefined} onClick={() => goView('game')}><NavIcon d={ICONS.record} />경기</button>
      <button className={tabOn === 2 ? 'f1-on' : undefined} aria-current={tabOn === 2 ? 'page' : undefined} onClick={() => goView('player')}><NavIcon d={ICONS.player} />선수</button>
      {viewer && <button className="sv-sync-open" onClick={openSync} aria-haspopup="dialog" aria-label={`중계 맞추기. 지금 ${behind > 0 ? `${behind}초 늦춰 보는 중` : '늦추지 않고 보는 중'}${setting.synced ? '' : ', 아직 맞추지 않았어요'}`}>{setting.synced ? '맞추기' : '중계 맞추기'}</button>}</nav> : <nav className="sv-bottom-nav" aria-label="라이브 상세">
      <button className="f1-on" onClick={openRec} aria-haspopup="dialog"><NavIcon d={ICONS.rec} />예측</button>
      <button onClick={() => openSheet('player-b')} aria-haspopup="dialog"><NavIcon d={ICONS.player} />선수</button>
      <button onClick={() => openSheet('mix')} aria-haspopup="dialog"><NavIcon d={ICONS.mix} />구종</button>
      <button onClick={() => openSheet('history')} aria-haspopup="dialog"><NavIcon d={ICONS.record} />기록</button>
      {viewer && <button className="sv-sync-open" onClick={openSync} aria-haspopup="dialog" aria-label={`중계 맞추기. 지금 ${behind > 0 ? `${behind}초 늦춰 보는 중` : '늦추지 않고 보는 중'}${setting.synced ? '' : ', 아직 맞추지 않았어요'}`}>{setting.synced ? '맞추기' : '중계 맞추기'}</button>}</nav>}

    {!maskPublished && sheet === 'pa' && <PaSheet pas={shownD.pas} selected={paKey} onClose={closeSheet} onBox={() => setSheet('box')} foldText={fold ? FOLD_TEXT : null}
      onPick={p => {
        // a finished plate appearance with no pitch at all (intentional walk and the like): its own result, no jump elsewhere
        if (p.complete && !pitches.some(x => x.pa_key === p.pa_key) && (typeof p.first !== 'number' || !byIndex.has(p.first))) { setEmptyPa(p); setSheet('pa-empty'); return; }
        setSheet(null); if ((live && p.pa_key === shownD.current?.pa_key && !p.complete) || typeof p.first !== 'number' || !byIndex.has(p.first)) goLatest(); else pick(p.first); }} />}
    {!maskPublished && sheet === 'pa-empty' && emptyPa && <Sheet title={`${inningText(emptyPa)} · ${emptyPa.batter}`} onClose={closeSheet}>
      <p><b>{fold ? FOLD_TEXT : emptyPa.result}</b></p><p className="sv-muted">이 타석은 투구 기록이 없어요. 공을 던지지 않고 끝난 타석이에요.</p>
    </Sheet>}
    {/* a selected past pitch shows the linescore as of that pitch (never the latest one, which would reveal later play) */}
    {sheet === 'box' && <BoxSheet ls={latest ? shownD.linescore_asof : pitch!.line ?? null} game={game} at={situation} onClose={closeSheet} />}
    {!maskPublished && sheet === 'detail' && !selectedPitch && <DetailSheet bounds={shownD.zone_bounds} onClose={closeSheet} rec={stateSlot && !shownRec ? null : rec} recTitle={stateSlot && !shownRec ? `다음 공: ${eff!.label}` : held ? `마지막으로 받은 예측${nextNumber ? ` · ${nextNumber}구` : ''}` : recTitle}
      pair={pairText} absentText={absentText}
      actual={nextPhase ? null : actual} actualTitle={actualTitle} paResult={ending} actualFolded={fold && !nextPhase && actual !== null ? FOLD_TEXT : null}
      actualExtra={!nextPhase && !fold && shownPitch?.expectation && shownPitch.actual?.event ? <ExpectationLine e={shownPitch.expectation} /> : null} notes={shownD.notes} speed={mph => speedText(mph, unit)} unit={unit}
      recNote={stateSlot ? [eff!.message, note].filter(Boolean).join(' ') || null : pastUnconfirmed ? '이 공을 확인하는 동안에는 예측을 함께 보여 드리지 않아요.' : null} actualPending={!nextPhase && unconfirmed(shownPitch)} actualCorrected={!nextPhase && shownPitch?.receipt?.corrected === true} actualHeld={!nextPhase && heldReceipt(shownPitch?.receipt)}
      extra={<div className="sv-row"><button className="sv-link-text sv-model-link" onClick={() => setSheet('info')} aria-haspopup="dialog">기준과 한계 · 모델 소개</button></div>} />}
    {!maskPublished && sheet === 'detail' && selectedPitch && (() => { const at = pitches.indexOf(selectedPitch);
      return <PitchSheet pitch={selectedPitch} bounds={shownD.zone_bounds} onClose={closeSheet} speed={mph => speedText(mph, unit)} unit={unit}
        onPrev={at > 0 ? () => pick(pitches[at - 1].index) : null} onNext={at >= 0 && at < pitches.length - 1 ? () => pick(pitches[at + 1].index) : null}
        paResult={!fold && selectedPitch.actual?.event ? shownD.pas.find(p => p.pa_key === selectedPitch.pa_key && p.complete)?.result ?? null : null}
        folded={fold ? FOLD_TEXT : null} pending={unconfirmed(selectedPitch)} corrected={selectedPitch.receipt?.corrected === true} held={heldReceipt(selectedPitch.receipt)}
        actualExtra={!fold && selectedPitch.expectation && selectedPitch.actual?.event ? <ExpectationLine e={selectedPitch.expectation} /> : null} recIn={pastRec(selectedPitch)} />; })()}
    {!maskPublished && sheet === 'history' && <RecordSheet key={recordTab} initial={recordTab} detail={shownD} fold={fold} onClose={closeSheet} onPick={pick} asOfIndex={asOfIndex}
      headExtra={weKnown ? <button className="sv-sheet-link" onClick={() => setSheet('win')} aria-haspopup="dialog">승리확률</button> : undefined} />}
    {sheet === 'win' && <Sheet title="현재 상황의 승리확률 · 모델 추정" onClose={closeSheet}>{homeProbability === null
      ? <p className="sv-muted" role="status">현재 승리확률 정보 없음</p>
      : <div className="sv-we-bars" role="img" aria-label={`현재 원정팀 ${short(game.away_team)} 승리확률 ${Math.round((1 - homeProbability) * 100)}퍼센트, 홈팀 ${short(game.home_team)} 승리확률 ${Math.round(homeProbability * 100)}퍼센트`}>
        {([['away', game.away_team, 1 - homeProbability], ['home', game.home_team, homeProbability]] as const).map(([sideName, team, value]) => <div className="sv-we-bar" key={sideName} aria-hidden="true"><span>{short(team)}</span><div><i style={{ width: `${value * 100}%`, background: teamColor(team) }} /></div><b>{Math.round(value * 100)}%</b></div>)}</div>}
      <p className="sv-muted">타석 시작 상황으로 학습한 모델 추정이며 볼·스트라이크 카운트는 반영하지 않아요.</p></Sheet>}
    {!maskPublished && sheet === 'mix' && <PitchMixSheet detail={shownD} pitcherId={pitcherP?.id ?? null} pitcherName={pitcherP?.name ?? null} fold={fold} onClose={closeSheet} asOfIndex={asOfIndex} />}
    {!F1.tabs && (sheet === 'player-b' || sheet === 'player-p' || sheet === 'player-h') && <Sheet title={F1.tabs ? '선수' : sheetPerson?.name ?? '선수 정보'} onClose={closeSheet} startFull={F1.tabs} half={!F1.tabs && playerRole === 'p' ? 'max(56dvh, min(420px, calc(100dvh - 56px)))' : null}>
{playerInner}
    </Sheet>}
    {F1.tabs && sheet === 'analysis' && <AnalysisSheet data={scenes} done={doneAbs} onClose={closeSheet} onPick={sc => { setSceneRank(sc.rank); setSheet('scene'); }} onInfo={() => setSheet('info')} />}
    {sheet === 'scene' && scenes && shownScene && <SceneSheet data={scenes} scene={shownScene} other={F1.sceneOther} onClose={closeSheet} onList={() => setSheet('scenes')}
      card={F1.tabs ? (() => { // SC-b: the deciding pitch — the recommendation before it and the pitch thrown (published record, else the same-policy pack)
        const key = tabData.recs?.scenePitch.get(shownScene.at_bat) ?? [...pitches].reverse().find(x => x.at_bat_number === shownScene.at_bat)?.key;
        const sp = pitches.find(x => x.key === key); const a = sp?.actual; if (!sp || !a || a.x === null || a.z === null) return null;
        const cands = candidatesFor(sp, tabData.recs, livePolicy(shownD)); if (!cands.length) return null;
        return <div className="f1-sc-card"><p className="f1-sc-card-t">타석 마지막 공 · {sp.pitch_number}구 <small>투수 시점</small></p><div className="f1-sc-zones">
          <figure><figcaption>던지기 전 예측</figcaption><Zone mini bounds={shownD.zone_bounds} candidates={cands} actual={null} actualLabel="" setupX={null} /><p>{cands.map(c => `${c.rank} ${c.pitch_label}`).join(' · ')}</p></figure>
          <figure><figcaption>실제 던진 공</figcaption><Zone mini bounds={shownD.zone_bounds} candidates={[]} actual={{ x: a.x, z: a.z }} actualLabel="" setupX={null} actualKind={pitchKind(a)} /><p>{a.pitch_label}{a.speed_mph !== null ? ` ${speedText(a.speed_mph, unit)}` : ''}</p></figure></div></div>; })() : null} />}
    {sheet === 'scenes' && scenes && <SceneList data={scenes} done={doneAbs} onClose={closeSheet} onPick={sc => { setSceneRank(sc.rank); setSheet('scene'); }} />}
    {sheet === 'info' && <InfoSheet onClose={closeSheet} notes={shownD.notes} />}
    {sheet === 'reference' && reference && <ReferenceSheet card={reference} state={eff ? `${eff.label}${eff.message ? ` · ${eff.message}` : ''}` : null} onClose={closeSheet} />}
  </div>{syncSheet}</SheetFrame.Provider>;
}

/** The status bubble (UI 10/26 ⑥, design 08): the marker's word as a title and the state's own sentence(s). Escape, a tap outside or
 *  the marker itself close it; focus returns to the marker. The same sentences are also in the screen's polite live region. */
function StatusPop({ chip, onClose, onRetry, busy }: { chip: { label: string; tone: Tone; lines: string[] }; onClose: () => void; onRetry: (() => void) | null; busy: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose); closeRef.current = onClose;
  useEffect(() => {
    const opener = document.querySelector<HTMLElement>('.sv-hd-status');
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); closeRef.current(); } };
    const onDown = (e: PointerEvent) => { const t = e.target as Node; if (!box.current?.contains(t) && !opener?.contains(t)) closeRef.current(); };
    window.addEventListener('keydown', onKey); window.addEventListener('pointerdown', onDown);
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('pointerdown', onDown); if (opener?.isConnected) opener.focus(); };
  }, []);
  return <><div className="sv-pop-dim" aria-hidden="true" />
    <div className="sv-pop" id="sv-status-pop" role="dialog" aria-labelledby="sv-pop-title" tabIndex={-1} ref={box} data-tone={chip.tone}>
      <h2 id="sv-pop-title"><i aria-hidden="true" />{chip.label}</h2>
      {chip.lines.map(l => <p key={l}>{l}</p>)}
      {onRetry && <button className="sv-pop-retry" onClick={onRetry} disabled={busy}>다시 확인</button>}
    </div></>;
}

const backHref = () => { if (__PZ_STATIC__) return publicListHref(); const demo = new URLSearchParams(window.location.search).get('demo'); return demo ? `/watch?demo=${encodeURIComponent(demo)}` : '/watch'; };
function TopBar({ right, title, label }: { right?: ReactNode; title?: string; label?: string }) {
  return <header className="sv-top">
    <a className="sv-back" href={backHref()} aria-label="경기 목록으로">‹</a>
    {title ? <h1 className="sv-title" aria-label={label}>{title}</h1> : <a className="sv-brand small" href={__PZ_STATIC__ ? publicListHref() : '/watch'}>Pitcheezy<span>.</span></a>}
    <span className="sv-top-right">{right}</span>
  </header>;
}

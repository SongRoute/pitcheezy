import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { FetchError, fetchDetail, fetchPublishHealth, fetchPublishedDetail, fetchPublishedManifest, isGamePublishedSoon, mergeDetail } from './api';
import { PUBLISH_CONFIG, gameCheck, gameCheckLag, staleAfterMs, clockWindowMs, isPublishStale, lostAfterMs, publicListHref, publishLag, serverNow } from './publish';
import type { PublishStamp, PublishHealth } from './publish';
import type { Actual, Candidate, Detail, DisplayStateName, GameCard, Linescore, Pa, Person, PitchRow, PlayerInfo, PreviousPick, Rec, Situation } from './model';
import { dateHeading, inningText, kstTime, pitchName, plainZone, policyShare, short } from './model';
import { teamColor } from './teams';
import Zone from './Zone';
import { ABSENT_PAST, DetailSheet, InfoSheet, PaSheet, Sheet, absentLine, provenanceText } from './Sheets';
import './service.css';

/** One game screen for live, demo and finished games (UI redesign 2026-10-05).
 *  Two independent states, never mixed: the GAME's status (LIVE dot, from the server) and the VIEWER's state
 *  (following the latest vs. looking at a past pitch; auto refresh on/off). */
export const POLL_MS = 2000;
export const STALE_MS = 30000;
/** a scheduled live game is asked again slowly so the screen starts by itself; a first load that failed is retried by itself */
export const PREGAME_POLL_MS = 10000;
export const FIRST_LOAD_RETRY_MS = 4000;
/** Static (public) mode takes the same three cadences from build-time config; the defaults are the values above. */
const CADENCE = __PZ_STATIC__ ? { live: PUBLISH_CONFIG.livePollMs, pregame: PUBLISH_CONFIG.pregamePollMs, retry: PUBLISH_CONFIG.firstLoadRetryMs }
  : { live: POLL_MS, pregame: PREGAME_POLL_MS, retry: FIRST_LOAD_RETRY_MS };
/** Display state → chip mark. The state is always written as words (server label); the mark and the colour only support it. */
/** U+FE0E asks for the text (not emoji) form of the two symbols. */
const LATE = '⏱\uFE0E', WARN = '⚠\uFE0E';
const STATE_MARK: Record<DisplayStateName, string> = { recommended: '', unconfirmed: '…', unsupported: '–', feed_delayed: LATE, disconnected: WARN,
  ledger_unavailable: '', waiting: '…', unavailable: '–', pregame: '', final: '' };
/** The client's own connection is failing: what is on screen is the last answer, not the current state. */
const CLIENT_LOST = { state: 'disconnected' as DisplayStateName, label: '연결 끊김', message: '연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.' };
/** Static mode: the files are reachable but the last publish is older than the (provisional) threshold. Same held look as a lost connection. */
const PUBLISH_LATE = { state: 'disconnected' as DisplayStateName, label: '갱신 지연', message: '새 내용이 늦어지고 있어요. 마지막으로 받은 내용을 보여 드려요.' };
/** Static mode: back in view (or back online / awake) and the first read is still on its way: neutral, never "late" from old data. */
const PUBLISH_CHECKING = { state: 'unconfirmed' as DisplayStateName, label: '확인 중', message: '새 내용을 확인하고 있어요.' };
const isDemoSession = (d: Detail) => d.session !== null;
const unconfirmed = (p: PitchRow | null | undefined) => p?.receipt?.status === 'unconfirmed';
const pickText = (pitch: string, zone: string | null | undefined) => zone ? `${pitch} · ${zone}` : pitch;
const previousText = (p: PreviousPick | null) => p ? pickText(pitchName(p.pitch_label, p.pitch_type), p.zone_label ? plainZone(p.zone_label) : null) : '';
/** D1 client guard: what was last drawn for the pending pitch slot, so a list that changes without a server note is still announced. */
type Drawn = { key: string | null; lastIdx: number; who: string; seen: Detail; sig: string; top: string; prev: { sig: string; top: string } | null };
type Selection = { kind: 'latest' } | { kind: 'pitch'; index: number };

const UNIT_KEY = 'pitcheezy.unit';
const readUnit = () => { try { return localStorage.getItem(UNIT_KEY) === 'mph' ? 'mph' : 'kmh'; } catch { return 'kmh'; } };
const speedText = (mph: number | null, unit: 'kmh' | 'mph') => mph === null ? '' : unit === 'kmh' ? `${Math.round(mph * 1.609344)}km/h` : `${mph.toFixed(1)}mph`;
const clockText = (ms: number) => new Date(ms).toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul', hour12: false });
const hhmm = (ms: number) => new Date(ms).toLocaleTimeString('en-GB', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit' });
export const gameIdFromPath = () => {
  const m = /^\/(?:watch|game|live)\/(\d+)\/?$/.exec(window.location.pathname);
  const q = new URLSearchParams(window.location.search).get('game');
  // public bundle: `?game=` counts only as plain decimal digits (no 1e3, 0x10, signs or spaces)
  const id = Number(m?.[1] ?? (__PZ_STATIC__ && !/^[0-9]+$/.test(q ?? '') ? NaN : q));
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
function PlayerCell({ team, abbr, role, person, hand, info, onOpen, side, today, idle = false }: { team: string; abbr?: string | null; role: 'p' | 'b'; person: Person | null; hand: string | null | undefined;
  info: PlayerInfo | undefined; onOpen: () => void; side: 'away' | 'home'; today: { text: string; spoken: string } | null; idle?: boolean }) {
  // nobody at the plate (before the first pitch, between innings): the team only, no role, no numbers of the finished half
  if (idle && !person) return <button className={`sv-player ${side} idle`} style={{ '--team': teamColor(team) } as CSSProperties} disabled aria-label={`${short(team)}, 지금은 타석이 없어요`}>
    <span className="sv-player-name"><b>{abbr || short(team)}</b></span><span className="sv-player-2">–</span><span className="sv-player-3">–</span></button>;
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
function RecList({ rec, emptyText, onOpen, lead = '', pair = false, absentText = ABSENT_PAST }: { rec: Rec | null; emptyText: string; onOpen: () => void; lead?: string;
  /** past pitch: each row also names the target zone in words, so it reads next to the actual pitch's type and location */
  pair?: boolean; absentText?: string }) {
  if (!rec || rec.status !== 'ready' || !rec.candidates.length) return <button className="sv-reclist empty" onClick={onOpen} aria-haspopup="dialog">
    <span className="sv-norec">{rec?.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황' : rec ? absentLine(rec, absentText) : emptyText}</span></button>;
  return <button className={`sv-reclist${pair ? ' pair' : ''}`} onClick={onOpen} aria-haspopup="dialog"
    aria-label={`${lead}추천 후보 ${rec.candidates.map(c => `${c.rank}순위 ${c.pitch_label} ${c.zone_label || '위치 없음'}, 정책 선택 비중 ${policyShare(c.probability)}`).join('; ')}. 퍼센트는 성공 확률이 아니에요. 자세히 보기`}>
    {rec.candidates.map((c: Candidate) => <span key={c.rank} className="sv-rec-row" aria-hidden="true">
      <i className="sv-rk">{c.rank}</i><span className="sv-rec-pct">{policyShare(c.probability)}</span><span className="sv-rec-type">{c.pitch_label}</span>{pair && c.zone_label && <span className="sv-rec-zone">{c.zone_label}</span>}</span>)}
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
  // the tab was hidden: polling paused on purpose, so that time never counts as a late or lost connection (re-based on return)
  const [hidden, setHidden] = useState(() => document.hidden);
  const [resumed, setResumed] = useState(0);
  const [sheet, setSheet] = useState<'pa' | 'pa-empty' | 'detail' | 'info' | 'box' | 'player-away' | 'player-home' | null>(null);
  const [emptyPa, setEmptyPa] = useState<Pa | null>(null);  // a plate appearance with no pitch, shown by itself in a sheet
  const stripRef = useRef<HTMLDivElement>(null);
  const sheetOpener = useRef<HTMLElement | null>(null);
  const openSheet = (kind: typeof sheet) => { sheetOpener.current = document.activeElement as HTMLElement; setSheet(kind); };
  const closeSheet = () => { setSheet(null); const opener = sheetOpener.current; requestAnimationFrame(() => opener?.isConnected && opener.focus()); };
  const [edges, setEdges] = useState({ left: false, right: false });
  const [busy, setBusy] = useState(false);
  const [unit, setUnit] = useState<'kmh' | 'mph'>(readUnit);
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
  const appliedSequence = useRef<number | null>(null);
  const [checking, setChecking] = useState(false);
  const inflight = useRef<AbortController | null>(null);
  const drawn = useRef<Drawn | null>(null);
  const picked = useRef(false);  // the viewer chose a past pitch (as opposed to the default first pitch of a game that is not live)

  const load = useCallback(async (full = false, force = false) => {
    // a background poll never cancels a slower answer still on its way (only a manual refresh does)
    if (inflight.current && !inflight.current.signal.aborted && !force) return;
    inflight.current?.abort();
    const ctrl = new AbortController(); inflight.current = ctrl;
    setBusy(true);
    try {
      if (__PZ_STATIC__) {  // one health-first chain; malformed/old health falls back to a whole-file read
        let h: PublishHealth | null = null;
        try { h = await fetchPublishHealth(ctrl.signal); if (!ctrl.signal.aborted) setHealth(h); } catch { /* keep the last health */ }
        if (ctrl.signal.aborted) return;
        const { present, check } = gameCheck(h, gamePk);
        if (present) { setLastOk(Date.now()); setError(null); }
        if (check && check.file_sequence !== null && (appliedSequence.current === null || check.file_sequence > appliedSequence.current)) {
          pendingTarget.current = Math.max(pendingTarget.current ?? 0, check.file_sequence);
          pending.current ??= performance.now(); setPendingSince(pending.current);
        }
        if (check?.file_sequence === null) {
          pendingTarget.current = null; pending.current = null; setPendingSince(null); setWithdrawn(true); detailRef.current = null; setDetail(null); setFileStamp(null);
          setError({ text: '이 경기는 지금 볼 수 없어요.', status: 404 });
          return;
        }
        if (pendingTarget.current === null && check && detailRef.current && appliedSequence.current !== null && check.file_sequence! <= appliedSequence.current) {
          setError(null); setLastOk(Date.now()); return;
        }
        const got = await fetchPublishedDetail(gamePk, ctrl.signal);
        if (ctrl.signal.aborted) return;
        if (pendingTarget.current !== null && got.stamp.seq! < pendingTarget.current) return;  // propagation delay: retry this target next tick
        pendingTarget.current = null; pending.current = null; setPendingSince(null); appliedSequence.current = got.stamp.seq;
        setWithdrawn(false); detailRef.current = got.detail; setDetail(got.detail); setFileStamp(got.stamp); setError(null); setLastOk(Date.now());
        return;
      }
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
      if (__PZ_STATIC__ && pending.current !== null) return;  // announced file is still propagating; keep neutral and retry
      setError(e instanceof FetchError ? { text: e.message, status: e.status } : { text: '자료를 불러오지 못했어요.', status: 0 });
    } finally { if (inflight.current === ctrl) { inflight.current = null; setBusy(false); if (__PZ_STATIC__) setChecking(false); } }
  }, [gamePk]);

  useEffect(() => { void load(true, true); return () => { inflight.current?.abort(); }; }, [load]);
  const live = detail ? detail.session_status !== 'ended' && (detail.game.status === 'live' || (detail.next !== null && detail.game.status !== 'final')) : false;
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
  // static mode: a screen pinned on a past pitch (auto off) still reads, slowly, while the game is not final. If the server turns the
  // pinned pitch back to 확인 중, the same fail-closed rules blank it here too; the view stays pinned (no jump to the latest).
  const pinnedMs = __PZ_STATIC__ && !auto && pollMs > 0 ? manifestMs ?? PUBLISH_CONFIG.pinnedPollMs : 0;
  useEffect(() => {
    if (!pinnedMs) return;
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, pinnedMs);
    return () => window.clearInterval(id);
  }, [pinnedMs, load]);
  // first load failed (nothing to show yet): keep trying by itself, but only when the server could not be reached or failed (network
  // error / timeout = 0, 5xx). A refusal (423 embargo, 400, 404) is an answer, not an outage: it is shown as before and not retried.
  const firstLoadRetry = detail === null && error !== null && (error.status === 0 || error.status >= 500);
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
  useEffect(() => { if (detail && sel === null) setSel(live ? { kind: 'latest' } : detail.pitches.length ? { kind: 'pitch', index: detail.pitches[0].index } : { kind: 'latest' }); }, [detail, sel, live]);
  // a screen opened before play (or in a suspension) that the viewer never touched follows the latest pitch once the game is live
  const wasLive = useRef<boolean | null>(null);
  useEffect(() => {
    if (!detail) return;
    if (live && wasLive.current === false && !picked.current && sel?.kind === 'pitch') setSel({ kind: 'latest' });
    wasLive.current = live;
  }, [detail, live, sel]);
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
      // public bundle: no file for a game (not published yet, or withheld while a demo runs on it) = not available now, said calmly
      ? __PZ_STATIC__ && error.status === 404 ? <div className="sv-unavailable"><p className="sv-muted" role="status">이 경기는 지금 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.</p>
        <div className="sv-row"><button className="sv-btn" onClick={askAgain}>다시 확인</button><a className="sv-btn" href={publicListHref()}>경기 목록</a></div>
        {!gaveUp && <p className="sv-muted sv-retrying">올라오면 자동으로 보여 드려요.</p>}</div>
      : <div className="sv-alert" role="alert"><p>{error.status === 423 ? '이 경기는 지금 기록을 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.' : error.text}</p>
        <div className="sv-row"><button className="sv-btn" onClick={() => void load(true, true)}>다시 시도</button><a className="sv-btn" href={__PZ_STATIC__ ? publicListHref() : '/watch'}>경기 목록</a></div>
        {firstLoadRetry && <p className="sv-muted sv-retrying">자동으로 다시 연결하고 있어요.</p>}</div>
      : <p className="sv-muted" role="status">경기 기록을 불러오는 중…</p>}</main></div>;
  }

  const { game } = detail;
  const verification = __PZ_STATIC__ ? gameCheck(health, gamePk) : { present: false, check: null };
  const awaitingFile = __PZ_STATIC__ && pendingSince !== null;
  const maskPublished = __PZ_STATIC__ && (awaitingFile || (verification.present && (!verification.check || checking)));

  const pitches = detail.pitches;
  const lastPitch = pitches[pitches.length - 1] ?? null;
  const pitch: PitchRow | null = sel?.kind === 'pitch' ? byIndex.get(sel.index) ?? null : null;
  const latest = pitch === null;  // a pin whose pitch is gone (demo seek back) shows latest until the effect resets it
  // Live display state (absent for archive / demo / older backends: then everything below behaves as before the contract).
  const disp = detail.display ?? null;
  // Between innings the live route sends no situation, batter or pitcher. Nothing of the finished half is then presented as current:
  // no count, outs, runners or matchup, and no pitch cards of the last batter; only the score and which half just ended stay.
  const breakNow = latest && live && detail.live_source === true && (detail.phase === 'between_innings' || disp?.reason === 'between_innings');
  const ended = breakNow && lastPitch ? { inning: lastPitch.situation.inning, half: lastPitch.situation.half } : null;
  // What the stage shows
  const situation: Situation | null = latest ? (breakNow ? null : detail.current?.situation ?? lastPitch?.situation ?? null) : pitch!.situation;
  const rec: Rec | null = maskPublished ? null : latest ? (live ? detail.next : lastPitch?.rec ?? null) : pitch!.rec;
  const actual: Actual | null = maskPublished ? null : latest ? lastPitch?.actual ?? null : pitch!.actual;
  const nextPhase = latest && live;
  const shownPitch = latest ? lastPitch : pitch;
  const paKey = latest ? detail.current?.pa_key ?? lastPitch?.pa_key ?? null : pitch!.pa_key;
  const paPitches = breakNow || maskPublished ? [] : pitches.filter(p => p.pa_key === paKey);
  const ending = !maskPublished && shownPitch?.actual?.event ? detail.pas.find(p => p.pa_key === shownPitch.pa_key && p.complete)?.result ?? null : null;
  const stateSlot = disp !== null && latest && (nextPhase || disp.state === 'pregame');
  const slotNumber = Number(/^\d+:\d+:(\d+)$/.exec(disp?.pitch_key ?? '')?.[1] ?? NaN);
  const nextNumber = nextPhase && !breakNow ? (stateSlot && Number.isInteger(slotNumber) ? slotNumber
    : detail.current?.pa_key && lastPitch?.pa_key === detail.current.pa_key ? lastPitch.pitch_number + 1 : 1) : null;
  const shownRec = stateSlot ? disp!.shown && rec?.status === 'ready' && rec.candidates.length > 0 : rec?.status === 'ready';
  const batterP = latest ? (breakNow ? null : detail.current?.batter ?? lastPitch?.batter ?? null) : pitch!.batter;
  const pitcherP = latest ? (breakNow ? null : detail.current?.pitcher ?? lastPitch?.pitcher ?? null) : pitch!.pitcher;
  // live route with nobody at the plate (before the first pitch, between innings): no matchup is named and no role is guessed
  const idle = latest && detail.live_source === true && batterP === null && pitcherP === null;
  const side = batterP?.side === 'R' || batterP?.side === 'L' ? batterP.side : null;  // switch hitters: only the side actually taken; never guessed

  // Connection: the client's own polling (clientStale / error) and the server's upstream feed. With a display state the server part
  // comes from display.feed; the old `stale_s > 30` guess (which also fires in long breaks) is used only without one.
  const clientStale = live && auto && !hidden && lastOk !== null && now - Math.max(lastOk, resumed) > (__PZ_STATIC__ ? lostAfterMs(pollMs) : STALE_MS);
  const rawCheck = health?.gameChecks?.[String(gamePk)];
  const checkFeedFailed = rawCheck !== null && typeof rawCheck === 'object' && 'feed_ok' in rawCheck && rawCheck.feed_ok === false;
  const healthFeedFailed = verification.present && (checkFeedFailed || health?.failingGames.some(g =>
    g === gamePk || g === String(gamePk) || (g !== null && typeof g === 'object' && 'game_pk' in g && String(g.game_pk) === String(gamePk))));
  const feedDown = __PZ_STATIC__ && verification.present ? healthFeedFailed : disp ? !disp.feed.ok : live && (detail.feed.ok === false || (detail.feed.stale_s ?? 0) > 30);
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
  const lag = !__PZ_STATIC__ ? null : verification.present ? (timeWarnings ? gameCheckLag(health, verification.check, srv) : null) : live ? publishLag(fileStamp, health, srv)
    : waitingToStart && srv !== null && Number.isFinite(startMs) && srv.ms >= startMs ? publishLag(null, health, srv) : null;
  const pendingLate = awaitingFile && !(fileStamp?.final && game.status === 'final') && performance.now() - pendingSince! > staleAfterMs(pollMs);
  const lateNow = __PZ_STATIC__ && (pendingLate || isPublishStale(lag, pollMs));
  const publishStale = lateNow && !checking;
  const publishChecking = !publishStale && (awaitingFile || (verification.present && !verification.check) || (lateNow && checking));
  // lag_s is a proven lower bound (never an estimate): floor it and say '넘게', never '약'
  const lagText = disp?.feed.delayed ? (disp.feed.lag_s !== null ? `${Math.floor(disp.feed.lag_s)}초 넘게 늦음` : '평소보다 늦음') : null;
  const line = error || feedDown ? { kind: 'warn', text: `${WARN} 연결 끊김` } : clientStale ? { kind: 'warn', text: disp ? `${WARN} 연결 끊김` : `${LATE} 갱신 늦어짐` }  /* with a state slot the chip says 연결 끊김: one story */
    : publishStale ? { kind: 'late', text: `${LATE} 갱신 지연` } : publishChecking ? { kind: 'ok', text: '확인 중…' }
    : lagText ? { kind: 'late', text: `${LATE} 자료 늦음 · ${lagText}` }
    : { kind: 'ok', text: live ? (auto ? '실시간으로 받는 중' : '자동 갱신 멈춤') : disp?.message?.replace(/\.$/, '') ?? disp?.label ?? '' };
  // D1 "no silent flip": server note first; otherwise compare with what this screen last drew for the same pending pitch.
  let clientNote: string | null = null;
  if (stateSlot) {
    // Slot identity: the server's pitch_key. An answer without one (e.g. a short 'waiting') is the SAME slot with nothing drawn, unless a
    // pitch was thrown since or the plate appearance / batter changed; only a different non-null key starts a new slot.
    const key = disp!.pitch_key, lastIdx = lastPitch?.index ?? -1;
    const who = detail.current?.pa_key ?? (detail.current?.batter ? String(detail.current.batter.id) : '');
    const sig = shownRec ? rec!.candidates.map(c => `${c.pitch_type}@${c.zone_id ?? ''}`).join('|') : '';
    const top = shownRec ? pickText(rec!.candidates[0].pitch_label, rec!.candidates[0].zone_label) : '';
    let d = drawn.current;
    const other = !d || d.lastIdx !== lastIdx || (key !== null && d.key !== null && d.key !== key) || (key === null && who !== '' && d.who !== '' && d.who !== who);
    if (other) d = drawn.current = { key, lastIdx, who, seen: detail, sig, top, prev: null };
    else if (d!.seen !== detail) {  // once per received answer (safe under repeated renders)
      d!.seen = detail; if (key !== null) d!.key = key; if (who !== '') d!.who = who;
      if (d!.sig !== sig) {
        const left = d!.sig !== '' ? { sig: d!.sig, top: d!.top } : d!.prev;
        d!.prev = left && left.sig !== sig ? left : null;
        d!.sig = sig; d!.top = top;
      }
    }
    if (d!.prev) clientNote = `${d!.sig ? '추천이 갱신됐어요' : '조금 전에 보여 드린 추천은 지금 표시하지 않아요'} (이전: ${d!.prev.top})`;
  }
  const fix = stateSlot ? disp!.correction : null;
  const fixPrev = previousText(fix?.previous ?? null);
  const note = maskPublished ? null : fix ? `${fix.message}${fixPrev ? ` (이전: ${fixPrev})` : ''}` : clientNote;
  // The client's own requests failing (or silent for too long): the slot says so instead of looking current.
  const clientLost = disp !== null && (error !== null || clientStale || (__PZ_STATIC__ && healthFeedFailed === true));
  const eff = !stateSlot ? null : clientLost ? CLIENT_LOST : publishStale ? PUBLISH_LATE : publishChecking ? PUBLISH_CHECKING
    : { state: disp!.state, label: disp!.label, message: disp!.message ?? (shownRec || disp!.state !== 'recommended' ? null : '추천을 준비하고 있어요.') };
  // candidates kept from the last answer while disconnected: still visible, dimmed, labelled as last received
  const held = stateSlot && shownRec && eff!.state === 'disconnected';
  const liveDot = live && latest && auto && !isDemoSession(detail) && !(disp !== null && (clientLost || feedDown)) && !lateNow && !maskPublished;
  const scored = situation ?? null;
  // between innings the card carries no score: the as-of linescore (runs up to the last published play) keeps it on screen
  const breakScore = (k: 'away' | 'home') => breakNow ? detail.linescore_asof?.[k].r ?? lastPitch?.situation[`${k}_score`] ?? null : null;
  const awayScore = latest && game.away_score !== null ? game.away_score : scored?.away_score ?? breakScore('away');
  const homeScore = latest && game.home_score !== null ? game.home_score : scored?.home_score ?? breakScore('home');

  // browsing a past pitch pins the view and switches auto refresh off (no fetch while pinned); LIVE brings both back
  const pick = (index: number) => { picked.current = true; setSel({ kind: 'pitch', index }); setAuto(false); };
  const goLatest = () => { picked.current = false; setSel({ kind: 'latest' }); setAuto(true); void load(false, true); };
  // A past pitch of a live game: the recommendation the ledger recorded BEFORE that pitch, next to what was thrown. Never a score.
  const pastUnconfirmed = !nextPhase && unconfirmed(shownPitch);
  const liveRecord = !nextPhase && (detail.live_source === true || rec?.provenance === 'captured_live');
  const recTitle = nextPhase ? `다음 공 추천${nextNumber ? ` · ${nextNumber}구` : ''}` : liveRecord ? '투구 전에 기록된 추천' : '이 공 던지기 전 추천';
  const recHeading = liveRecord ? (pastUnconfirmed ? `${recTitle}: 확인 중` : rec?.status === 'ready' ? recTitle : `${recTitle} 없음`) : `${recTitle} (${rec ? provenanceText(rec) : '추천 없음'})`;
  const absentText = maskPublished ? '새 내용을 확인하고 있어요.' : pastUnconfirmed ? '확인 중이에요.' : nextPhase ? '추천 기록이 없어요.' : ABSENT_PAST;
  const actualTitle = nextPhase ? `직전 투구${lastPitch ? ` · ${inningText(lastPitch.situation)} ${lastPitch.pitch_number}구` : ''}` : `실제 투구${pitch ? ` · ${pitch.pitch_number}구` : ''}`;
  const top = situation?.half === 'Top';
  const info = (p: Person | null) => p ? detail.players?.[String(p.id)] : undefined;
  const openPa = () => openSheet('pa');
  const cands = shownRec ? rec!.candidates : [];
  const venue = game.venue ? `${game.venue} · ` : '';
  const when = `${dateHeading(game.kst_date)} ${kstTime(game.start)}${game.double_header ? ` · ${game.game_number}차전` : ''}`;
  const phase = detail.current?.phase ?? detail.phase ?? null;
  const emptyRec = maskPublished ? '확인 중' : nextPhase ? (phase === 'between_innings' ? '공수 교대 중' : phase === 'resolving' ? '결과 확인 중' : '추천 대기 중') : '이 시점에는 추천이 없어요';
  const actualWait = maskPublished ? '확인 중' : phase === 'between_innings' ? '공수 교대 중' : phase === 'resolving' ? '결과 확인 중' : '다음 공 대기 중';
  const list = <RecList rec={rec} emptyText={emptyRec} onOpen={() => openSheet('detail')} lead={held ? '마지막으로 받은 ' : liveRecord ? '투구 전에 기록된 ' : ''} pair={!nextPhase} absentText={absentText} />;
  const top1 = !nextPhase && !pastUnconfirmed && rec?.status === 'ready' ? rec.candidates[0] ?? null : null;
  const pairText = top1 && actual ? { rec: pickText(top1.pitch_label, top1.zone_label), actual: pickText(actual.pitch_label, actual.zone_label) } : null;
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

  return <div className={`sv-page sv-game${disp ? ' has-display' : ''}`}>
    <header className="sv-top sv-head">
      <a className="sv-back" href={backHref()} aria-label="경기 목록으로">‹</a>
      <h1 className="sv-head-main" aria-label={`${short(game.away_team)} ${awayScore ?? '–'} 대 ${homeScore ?? '–'} ${short(game.home_team)}. ${venue}${dateHeading(game.kst_date)} ${kstTime(game.start)} 시작`}>
        <span className="sv-head-score" aria-hidden="true"><span className="sv-tn">{short(game.away_team)}</span><strong style={{ color: scoreColor(game.away_team) }}>{awayScore ?? '–'}</strong><i>vs</i><strong style={{ color: scoreColor(game.home_team) }}>{homeScore ?? '–'}</strong><span className="sv-tn">{short(game.home_team)}</span></span>
        <small aria-hidden="true">{game.venue && <span>{game.venue}</span>}<span>{when} KST</span></small>
      </h1>
      <button className={`sv-live ${liveDot ? 'on' : ''}`} onClick={live ? goLatest : undefined} disabled={!live}
        aria-label={`${isDemo ? '지난 경기 시연. ' : ''}${statusLabel}${live ? ', 최신 공으로 이동하고 자동 갱신 켜기' : ''}`}>
        {liveDot && <i aria-hidden="true" />}{statusLabel}</button>
    </header>
    {/* Connection problems remain explicit alerts; history state is in the header. */}
    {/* The status line is a plain box; only its words are a live region. The time stamp and the retry control sit outside it so a
        screen reader is not re-triggered every minute or every poll. */}
    {disp || (__PZ_STATIC__ && (healthFeedFailed || publishStale || publishChecking)) ? <div className={`sv-notice line ${line.kind}`} data-kind={error ? 'client' : feedDown ? 'feed' : clientStale ? 'slow' : publishStale ? 'publish-late' : publishChecking ? 'checking' : lagText ? 'late' : 'ok'} data-publish-clock={publishStale ? lag?.clock ?? 'elapsed' : undefined}>
      <span className="sv-line-text" role="status">{line.kind === 'ok' && live && auto && <i className="sv-line-dot" aria-hidden="true" />}{line.text}<span className="sr-only">{error ? ` ${error.text} 자동으로 다시 연결하고 있어요.` : feedDown ? ' 경기 자료 연결이 잠시 끊겼어요.' : publishStale && !clientStale ? ` ${PUBLISH_LATE.message}` : ''}</span></span>
      {lastOk && <span className="sv-line-time" aria-hidden="true">· {hhmm(lastOk)} 수신</span>}
      {line.kind !== 'ok' && <button className="sv-link-btn" onClick={() => void load(true, true)} disabled={busy} aria-label="지금 다시 불러오기">{busy ? '…' : '다시'}</button>}
    </div> : (stale || error) && <p className="sv-notice warn" role="alert" title={error?.text}>
      {error ? '⚠ 연결 끊김' : '⏱ 갱신 늦어짐'}{lastOk ? ` · ${hhmm(lastOk)} 수신` : ''}<span className="sr-only">{error ? ` ${error.text}` : ''}{lastOk ? ` (${clockText(lastOk)})` : ''}</span>
        <button className="sv-link-btn" onClick={() => void load(true, true)} disabled={busy}>{busy ? '…' : '다시'}</button>
    </p>}

    <section className="sv-board" aria-label="경기 상황">
      <div className="sv-board-row">
        {situation ? <button className="sv-inning" onClick={openPa} aria-haspopup="dialog" aria-label={`${inningText(situation)}, 타석 선택`}>{situation.inning}회{situation.half === 'Top' ? '초' : '말'} ▾<small>타석 선택</small></button>
          : ended ? <button className="sv-inning" onClick={openPa} aria-haspopup="dialog" aria-label={`${inningText(ended)} 종료, 지난 타석 선택`}>{ended.inning}회{ended.half === 'Top' ? '초' : '말'} 종료 ▾<small>타석 선택</small></button> : <span />}
        {breakNow && <span className="sv-board-idle">공수 교대</span>}
        {situation && <Count s={situation} />}
        {situation && <Bases mask={situation.bases} />}
      </div>
    </section>

    <section className="sv-stage" aria-label={nextPhase ? '다음 공 추천' : '이 공의 추천과 실제 투구'}>
      <h2 className="sr-only">{stateSlot && !shownRec ? `다음 공: ${eff!.label}` : held ? `마지막으로 받은 추천 (${rec ? provenanceText(rec) : '추천 없음'}). 퍼센트는 정책 선택 비중이며 성공 확률이 아니에요.` : `${recHeading}.${shownRec ? ' 퍼센트는 정책 선택 비중이며 성공 확률이 아니에요.' : ''}`}</h2>
      {idle ? <div className="sv-duel idle"><span>{breakNow ? '다음 타자를 기다리고 있어요' : game.status === 'final' ? '' : '경기가 시작되면 타자와 투수를 보여 드려요'}</span></div>
      : <div className="sv-duel" role="group" aria-label={`타자 ${batterP?.name ?? '선수 미확인'} 대 투수 ${pitcherP?.name ?? '선수 미확인'}`}>
        <span title={batterP?.name ?? undefined}><b>타자: {batterP?.name ?? '선수 미확인'}</b></span>
        <i aria-hidden="true">vs</i>
        <span title={pitcherP?.name ?? undefined}><b>투수: {pitcherP?.name ?? '선수 미확인'}</b></span>
      </div>}
      {/* Pitcher's view, with the batter on the correct screen side. The zone owns the centre;
          recommendations now live below it, leaving its outside-coordinate margin unobstructed. */}
      <div className={`sv-zrow ${side === 'L' ? 'bat-l' : 'bat-r'}${held ? ' held' : ''}`}>
        {side && <div className={`sv-batband ${side === 'L' ? 'at-left' : 'at-right'}`}><Silhouette side={side} /></div>}
        <div className="sv-zone-wrap">
          <Zone bounds={detail.zone_bounds} candidates={cands}
            actual={!nextPhase && actual && actual.x !== null && actual.z !== null ? { x: actual.x, z: actual.z } : null} actualLabel=""
            setupX={!nextPhase ? actual?.setup_x ?? null : null} />
        </div>
      </div>
      {stateSlot ? <div className={`sv-pitch-summary sv-state${shownRec ? ' has-rec' : ''}${held ? ' held' : ''}`} data-state={eff!.state} data-pitch={disp!.pitch_key ?? undefined}>
        {/* ONE state at a time, in a slot of constant height: chip (words + mark), then the candidates and/or one calm sentence. */}
        <div className="sv-state-head">
          <span className="sv-chip" data-state={eff!.state}>{STATE_MARK[eff!.state] && <i aria-hidden="true">{STATE_MARK[eff!.state]}</i>}{eff!.label}</span>
          {nextPhase && <span className="sv-summary-label">{held ? `마지막으로 받은 추천${nextNumber ? ` · ${nextNumber}구` : ''}` : eff!.state === 'recommended' ? (nextNumber ? `${nextNumber}구` : '') : shownRec ? recTitle : breakNow ? '' : `다음 공${nextNumber ? ` · ${nextNumber}구` : ''}`}</span>}
          <button className="sv-state-more" onClick={() => openSheet('detail')} aria-haspopup="dialog" aria-label="추천 상태 자세히 보기">자세히</button>
        </div>
        <div className="sv-state-body">
          {shownRec && <div className="sv-rec-summary">{list}</div>}
          <div className="sv-state-text">
            {/* one always-present polite live region for the sentence and the change note (so a note that appears is announced).
                Beside a candidate list the note has priority; the sentence then stays for screen readers (chip + status line carry it). */}
            <div className="sv-state-live" role="status">
              <p className={`sv-state-msg${shownRec && note ? ' sr-only' : ''}${note ? ' with-note' : ''}`}>{eff!.message}</p>
              <p className="sv-state-note" data-kind={note ? (fix ? fix.kind : 'client') : undefined}>{note}</p>
            </div>
            {!eff!.message && !note && nextPhase && <p className="sv-state-idle"><span className="sv-summary-label">실제 투구</span><span className="sv-muted">{actualWait}</span></p>}
          </div>
        </div>
      </div> : <div className="sv-pitch-summary">
        <div className="sv-rec-summary"><span className="sv-summary-label">{recTitle}</span>{list}</div>
        <div className={`sv-actual-summary${nextPhase ? ' waiting' : ''}`}><span className="sv-summary-label">{nextPhase ? '실제 투구' : actualTitle}</span>{nextPhase ? <p className="sv-muted" role="status">{actualWait}</p> : actual && unconfirmed(shownPitch) ? <>
        <p className="sv-actual-caption pending">확인 중</p>
        <p className="sv-muted">이 공을 다시 확인하고 있어요.</p>
      </> : actual ? <>
        <p className="sv-actual-caption"><i className="sv-key-actual" /><b>{actual.pitch_label}</b>{actual.zone_label && <span className="sv-actual-zone"> · {actual.zone_label}</span>}
          {shownPitch?.receipt?.corrected && <span className="sv-tag" role="note" aria-label="기록이 정정된 공">정정됨</span>}</p>
        <p className="sv-result">{actual.speed_mph !== null && <small className="sv-actual-speed"><b>{unit === 'kmh' ? Math.round(actual.speed_mph * 1.609344) : actual.speed_mph.toFixed(1)}</b>{unit === 'kmh' ? 'km/h' : 'mph'}</small>}{ending ?? actual.result_label}</p>
      </> : <p className="sv-muted">실제 투구 기록 없음</p>}</div>
      </div>}

      <div className={`sv-strip-shell${edges.left ? ' fade-left' : ''}${edges.right ? ' fade-right' : ''}`}>
      <div className="sv-strip" ref={stripRef} role="group" aria-label="이 타석의 투구, 좌우로 스크롤">
        {paPitches.map(p => { const on = !latest && pitch?.index === p.index; const a = p.actual; const un = unconfirmed(p); const fixed = !un && p.receipt?.corrected === true;
          const sp = a?.speed_mph != null ? (unit === 'kmh' ? Math.round(a.speed_mph * 1.609344) : a.speed_mph.toFixed(1)) : '';
          return <button key={p.index} className={`sv-card ${on ? 'on' : ''}${un ? ' unconfirmed' : ''}`} aria-pressed={on} onClick={() => pick(p.index)}
            aria-label={un ? `${p.pitch_number}구 확인 중` : `${p.pitch_number}구 ${a?.pitch_label || '구종 미상'}${a?.speed_mph != null ? ` ${speedText(a.speed_mph, unit)}` : ''} ${a?.result_label ?? ''}${fixed ? ', 기록이 정정된 공' : ''}`}>
            <Zone mini bounds={detail.zone_bounds} candidates={[]} actualLabel="" setupX={null} actual={!un && a && a.x !== null && a.z !== null ? { x: a.x, z: a.z } : null} />
            {fixed && <i className="sv-card-tag" aria-hidden="true">정정됨</i>}
            {un ? <span className="sv-card-cap" aria-hidden="true"><span>{p.pitch_number}. 확인 중</span><span>–</span></span>
              : <span className="sv-card-cap" aria-hidden="true"><span>{p.pitch_number}. {a?.pitch_label || '구종 미상'}</span><span>{sp || '–'} <small>{unit === 'kmh' ? 'km/h' : 'mph'}</small></span></span>}</button>; })}
        {nextPhase && nextNumber && paKey === detail.current?.pa_key && <button className="sv-card pending on" aria-pressed="true" onClick={goLatest} aria-label={`다음 공 ${nextNumber}구, 아직 안 던짐`}>
          <span className="sv-card-q" aria-hidden="true">?</span><span className="sv-card-cap" aria-hidden="true">{nextNumber}. 다음</span></button>}
        {/* no card to show (before the first pitch, between innings): the strip keeps its height so nothing below it moves */}
        {idle && paPitches.length === 0 && <span className="sv-card sv-card-idle" aria-hidden="true" />}
      </div>
      </div>
    </section>

    <section className="sv-context" aria-label="선수와 다음 타자">
      <div className="sv-players">
        <PlayerCell side="away" team={game.away_team} abbr={game.away_abbr} role={top ? 'b' : 'p'} person={top ? batterP : pitcherP} hand={top ? batterP?.side : pitcherP?.hand} info={info(top ? batterP : pitcherP)} onOpen={() => openSheet('player-away')} today={top ? batterToday : pitcherToday} idle={idle} />
        <PlayerCell side="home" team={game.home_team} abbr={game.home_abbr} role={top ? 'p' : 'b'} person={top ? pitcherP : batterP} hand={top ? pitcherP?.hand : batterP?.side} info={info(top ? pitcherP : batterP)} onOpen={() => openSheet('player-home')} today={top ? pitcherToday : batterToday} idle={idle} />
      </div>
      <div className="sv-upcoming" aria-label="다음 두 타자">{[0, 1].map(i => <span key={i} title={upcoming[i]?.name ?? undefined}>{upcoming[i] ? `${upcoming[i].order}번 · ${upcoming[i].name ?? '타자 미확인'}` : idle ? '–' : '–번 · 타자 미확인'}</span>)}</div>
    </section>

    {!maskPublished && sheet === 'pa' && <PaSheet pas={detail.pas} selected={paKey} onClose={closeSheet} onBox={() => setSheet('box')}
      onPick={p => {
        // a finished plate appearance with no pitch at all (intentional walk and the like): its own result, no jump elsewhere
        if (p.complete && !pitches.some(x => x.pa_key === p.pa_key) && (typeof p.first !== 'number' || !byIndex.has(p.first))) { setEmptyPa(p); setSheet('pa-empty'); return; }
        setSheet(null); if ((live && p.pa_key === detail.current?.pa_key && !p.complete) || typeof p.first !== 'number' || !byIndex.has(p.first)) goLatest(); else pick(p.first); }} />}
    {sheet === 'pa-empty' && emptyPa && <Sheet title={`${inningText(emptyPa)} · ${emptyPa.batter}`} onClose={closeSheet}>
      <p><b>{emptyPa.result}</b></p><p className="sv-muted">이 타석은 투구 기록이 없어요. 공을 던지지 않고 끝난 타석이에요.</p>
    </Sheet>}
    {/* a selected past pitch shows the linescore as of that pitch (never the latest one, which would reveal later play) */}
    {sheet === 'box' && <BoxSheet ls={latest ? detail.linescore_asof : pitch!.line ?? null} game={game} at={situation} onClose={closeSheet} />}
    {!maskPublished && sheet === 'detail' && <DetailSheet bounds={detail.zone_bounds} onClose={closeSheet} rec={stateSlot && !shownRec ? null : rec} recTitle={stateSlot && !shownRec ? `다음 공: ${eff!.label}` : liveRecord ? recTitle : rec ? `${held ? '마지막으로 받은 추천' : recTitle} (${provenanceText(rec)})` : recTitle}
      pair={pairText} absentText={absentText}
      actual={nextPhase ? null : actual} actualTitle={actualTitle} paResult={ending} notes={detail.notes} speed={mph => speedText(mph, unit)}
      recNote={stateSlot ? [eff!.message, note].filter(Boolean).join(' ') || null : pastUnconfirmed ? '이 공을 확인하는 동안에는 추천을 함께 보여 드리지 않아요.' : null} actualPending={!nextPhase && unconfirmed(shownPitch)} actualCorrected={!nextPhase && shownPitch?.receipt?.corrected === true}
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

const backHref = () => { if (__PZ_STATIC__) return publicListHref(); const demo = new URLSearchParams(window.location.search).get('demo'); return demo ? `/watch?demo=${encodeURIComponent(demo)}` : '/watch'; };
function TopBar({ right, title, label }: { right?: ReactNode; title?: string; label?: string }) {
  return <header className="sv-top">
    <a className="sv-back" href={backHref()} aria-label="경기 목록으로">‹</a>
    {title ? <h1 className="sv-title" aria-label={label}>{title}</h1> : <a className="sv-brand small" href={__PZ_STATIC__ ? publicListHref() : '/watch'}>Pitcheezy<span>.</span></a>}
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

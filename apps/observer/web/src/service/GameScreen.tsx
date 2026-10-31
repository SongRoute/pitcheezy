import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { FetchError, fetchDetail, fetchPublishHealth, fetchPublishedDetail, fetchPublishedManifest, isGamePublishedSoon, mergeDetail } from './api';
import { PUBLISH_CONFIG, gameCheck, gameCheckLag, staleAfterMs, clockWindowMs, isPublishStale, lostAfterMs, publicListHref, publishLag, serverNow } from './publish';
import type { PublishStamp, PublishHealth } from './publish';
import type { Actual, Candidate, Detail, DisplayStateName, GameCard, Linescore, Pa, Person, PitchRow, PreviousPick, Rec, Situation } from './model';
import { teamColor } from './teams';
import { heldReceipt, inningText, kstTime, pitchName, plainZone, resultKind, settledCorrection, short, surname } from './model';
import Zone, { G1Dot, RankSq, ZoneLegend } from './Zone';
import { ABSENT_PAST, DetailSheet, InfoSheet, PaSheet, PitchSheet, Sheet, SheetFrame, absentLine } from './Sheets';
import SyncSheet, { FOLD_TEXT } from './SyncSheet';
import { ExpectationLine, ReferenceSheet, ReferenceTeaser } from './Reference';
import type { ViewerSetting } from './viewer';
import { VIEWER_KEY, learnViewer, readViewer, reloadViewer, requestDelay, wantedDelay, writeViewer } from './viewer';
import { MixTable, RecordSheet, PitchMixSheet } from './LivePanels';
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
const PUBLISH_LATE = { state: 'disconnected' as DisplayStateName, label: '갱신 지연', message: '새 내용이 늦어지고 있어요. 마지막으로 받은 내용을 보여 드려요.' };
/** Static mode: back in view (or back online / awake) and the first read is still on its way: neutral, never "late" from old data. */
const PUBLISH_CHECKING = { state: 'unconfirmed' as DisplayStateName, label: '확인 중', message: '새 내용을 확인하고 있어요.' };
const unconfirmed = (p: PitchRow | null | undefined) => p?.receipt?.status === 'unconfirmed';
const pickText = (pitch: string, zone: string | null | undefined) => zone ? `${pitch} · ${zone}` : pitch;
const previousText = (p: PreviousPick | null) => p ? pickText(pitchName(p.pitch_label, p.pitch_type), p.zone_label ? plainZone(p.zone_label) : null) : '';
/** D1 client guard: what was last drawn for the pending pitch slot, so a list that changes without a server note is still announced. */
type Drawn = { key: string | null; lastIdx: number; who: string; seen: Detail; sig: string; top: string; prev: { sig: string; top: string } | null };
type Selection = { kind: 'latest' } | { kind: 'pitch'; key: string };

const UNIT_KEY = 'pitcheezy.unit';
const readUnit = () => { try { return localStorage.getItem(UNIT_KEY) === 'mph' ? 'mph' : 'kmh'; } catch { return 'kmh'; } };
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

/** Header row 3 centre: the count in large type, runners and outs below it (UI 10/26 ①). */
function CountBox({ s }: { s: Situation }) {
  const runners = [1, 2, 3].filter(b => (s.bases & (1 << (b - 1))) !== 0);
  return <div className="sv-mu-count" role="img" aria-label={`볼 ${s.balls}, 스트라이크 ${s.strikes}, ${s.outs}아웃, 주자 ${runners.length ? runners.join('·') + '루' : '없음'}`}>
    <b aria-hidden="true">{s.balls}<i>-</i>{s.strikes}</b>
    <span aria-hidden="true"><Bases mask={s.bases} /><span className="sv-outs">{[0, 1, 2].map(i => <i key={i} className={i < s.outs ? 'on' : ''} />)}</span></span></div>;
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
  if (!cands.length) { const why = rec?.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황' : rec ? absentLine(rec, absentText) : emptyText;
    return <button className="sv-reclist empty" onClick={onOpen} aria-haspopup="dialog" aria-label={`${why}. 자세히 보기`}>
      {[0, 1, 2].map(i => <span key={i} className="sv-rec-cell" aria-hidden="true">—</span>)}</button>; }
  return <button className="sv-reclist" onClick={onOpen} aria-haspopup="dialog"
    aria-label={`${lead}추천 후보 ${cands.map(c => `${c.rank}순위 ${c.pitch_label}`).join('; ')}. 자세히 보기`}>
    {[0, 1, 2].map(i => { const c: Candidate | undefined = cands[i];
      return <span key={i} className="sv-rec-cell sv-rec-row" aria-hidden="true">{c ? <><RankSq rank={c.rank} /><b className="sv-rec-type">{c.pitch_label}</b></> : '—'}</span>; })}
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
  const [sheet, setSheet] = useState<'pa' | 'pa-empty' | 'detail' | 'info' | 'box' | 'player-b' | 'player-p' | 'sync' | 'reference' | 'history' | 'mix' | 'win' | null>(null);
  // the header status button's explanation bubble (UI 10/26 ⑥): the state's sentence lives here, not on the main screen
  const [pop, setPop] = useState(false);
  const [abOpen, setAbOpen] = useState(false);  // ④ the at-bat list, closed by default
  // viewer delay (중계 맞추기): this device's setting, what the server last said about its limits, and the one-time question
  const [setting, setSetting] = useState<ViewerSetting>(readViewer);
  const viewerMeta = useRef<{ max: number | null } | null>(null);
  const prompted = useRef(false);
  const [emptyPa, setEmptyPa] = useState<Pa | null>(null);  // a plate appearance with no pitch, shown by itself in a sheet
  const sheetOpener = useRef<HTMLElement | null>(null);
  const openSheet = (kind: typeof sheet) => { sheetOpener.current = document.activeElement as HTMLElement; setSheet(kind); };
  const closeSheet = () => { setSheet(null); const opener = sheetOpener.current; requestAnimationFrame(() => opener?.isConnected && opener.focus()); };
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
  const withdrawal = useRef(false);  // only a valid, explicit non-null game check can reopen a withdrawn game
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
        try { h = await fetchPublishHealth(ctrl.signal); if (!ctrl.signal.aborted) setHealth(h); } catch (e) { if (e instanceof FetchError && e.notice) throw e; /* keep the last health */ }
        if (ctrl.signal.aborted) return;
        const { present, check } = gameCheck(h, gamePk);
        if (withdrawal.current && !check) { setError({ text: '이 경기는 지금 볼 수 없어요.', status: 404 }); return; }
        if (present) { setLastOk(Date.now()); setError(null); }
        if (check && check.file_sequence !== null && (appliedSequence.current === null || check.file_sequence > appliedSequence.current)) {
          pendingTarget.current = Math.max(pendingTarget.current ?? 0, check.file_sequence);
          pending.current ??= performance.now(); setPendingSince(pending.current);
        }
        if (check?.file_sequence === null) {
          pendingTarget.current = null; pending.current = null; setPendingSince(null); withdrawal.current = true; setWithdrawn(true); detailRef.current = null; setDetail(null); setFileStamp(null);
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
        withdrawal.current = false; setWithdrawn(false); detailRef.current = got.detail; setDetail(got.detail); setFileStamp(got.stamp); setError(null); setLastOk(Date.now());
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
      detailRef.current = next; setDetail(next); setError(null); setLastOk(Date.now());
    } catch (e) {
      if (ctrl.signal.aborted) return;
      const notice = e instanceof FetchError && e.notice;
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
  const noticeHold = error?.notice === true;
  useEffect(() => { if (noticeHold) { drawn.current = null; setSheet(s => s === 'sync' ? s : null); } }, [noticeHold]);
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

  const { game } = detail;
  const verification = __PZ_STATIC__ ? gameCheck(health, gamePk) : { present: false, check: null };
  const awaitingFile = __PZ_STATIC__ && pendingSince !== null;
  const maskPublished = __PZ_STATIC__ && (awaitingFile || (verification.present && (!verification.check || checking)));

  const pitches = detail.pitches;
  const lastPitch = pitches[pitches.length - 1] ?? null;
  const selectedPitch = sel?.kind === 'pitch' ? pitches.find(p => p.key === sel.key) ?? null : null;
  // Live state never follows the separate detail selection. Polling and corrections continue.
  const pitch: PitchRow | null = live ? null : selectedPitch;
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
  const endingShown = !maskPublished && shownPitch?.actual?.event ? detail.pas.find(p => p.pa_key === shownPitch.pa_key && p.complete)?.result ?? null : null;
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
  const publishChecking = !publishStale && (awaitingFile || (verification.present && (!verification.check || checking)) || (lateNow && checking));
  // lag_s is a proven lower bound (never an estimate): floor it and say '넘게', never '약'
  // Viewer delay on (live route, feature on): how far behind this screen is shown, and whether thrown-pitch details are folded.
  // Folded until this device is synced: every realized pitch detail of a LIVE game (type, speed, location, call/result of each
  // pitch card, the actual-pitch row of a selected pitch, the detail sheet's actual part and the plate-appearance results).
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
  const ending = fold ? null : endingShown;
  // every published pitch of the at-bat on the main zone, G1 marks from the result. The ring belongs to the at-bat's NEWEST pitch only:
  // when that pitch has no coordinate or is being confirmed it is not drawn, and no older mark inherits its ring.
  const newestKey = paPitches[paPitches.length - 1]?.key ?? null;
  const thrown = nextPhase && !fold ? paPitches.filter(p => !unconfirmed(p) && p.actual?.x != null && p.actual?.z != null)
    .map(p => ({ x: p.actual!.x!, z: p.actual!.z!, number: p.pitch_number, kind: resultKind(p.actual!.description), latest: p.key === newestKey })) : [];
  // A2: a reference card rides only on the pending pitch's unsupported state; it is never drawn as a recommendation
  const reference = !maskPublished && stateSlot && latest && nextPhase && !shownRec && disp!.state === 'unsupported' ? detail.reference ?? null : null;
  const fix = stateSlot ? disp!.correction : null;
  const fixPrev = previousText(fix?.previous ?? null);
  const note = maskPublished ? null : fix ? `${fix.message}${fixPrev ? ` (이전: ${fixPrev})` : ''}` : clientNote;
  // The client's own requests failing (or silent for too long): the slot says so instead of looking current.
  const clientLost = disp !== null && (error !== null || clientStale || (__PZ_STATIC__ && healthFeedFailed === true));
  const eff = !stateSlot ? null : clientLost ? CLIENT_LOST : publishStale ? PUBLISH_LATE : publishChecking ? PUBLISH_CHECKING
    : { state: disp!.state, label: disp!.label, message: disp!.message ?? (shownRec || disp!.state !== 'recommended' ? null : '추천을 준비하고 있어요.') };
  // candidates kept from the last answer while disconnected: still visible, dimmed, labelled as last received
  const held = stateSlot && shownRec && eff!.state === 'disconnected';
  const scored = situation ?? null;
  // between innings the card carries no score: the as-of linescore (runs up to the last published play) keeps it on screen
  const breakScore = (k: 'away' | 'home') => breakNow ? detail.linescore_asof?.[k].r ?? lastPitch?.situation[`${k}_score`] ?? null : null;
  // a finished game's header always carries its final result (figure 11), whichever past pitch is selected below
  const atEnd = latest || game.status === 'final';
  const awayScore = atEnd && game.away_score !== null ? game.away_score : scored?.away_score ?? breakScore('away');
  const homeScore = atEnd && game.home_score !== null ? game.home_score : scored?.home_score ?? breakScore('home');

  // Historical detail is keyed independently; selecting it never pauses live polling.
  const pick = (index: number) => { picked.current = true; const p = byIndex.get(index); if (p) { setSel({ kind: 'pitch', key: p.key }); openSheet('detail'); } };
  // ③ the recommendation entries (cells, 자세히, bottom 추천) open the CURRENT recommendation on a live game: a past pitch opened
  // from the at-bat list before must not stay selected behind them. A finished game keeps its browsed pitch (that is its current view).
  const openRec = () => { if (live) { picked.current = false; setSel({ kind: 'latest' }); } openSheet('detail'); };
  const goLatest = () => { picked.current = false; setSel({ kind: 'latest' }); setAuto(true); void load(false, true); };
  // A past pitch of a live game: the recommendation the ledger recorded BEFORE that pitch, next to what was thrown. Never a score.
  const pastUnconfirmed = !nextPhase && unconfirmed(shownPitch);
  const liveRecord = !nextPhase && (detail.live_source === true || rec?.provenance === 'captured_live');
  const recTitle = nextPhase ? `다음 공 추천${nextNumber ? ` · ${nextNumber}구` : ''}` : liveRecord ? '투구 전에 기록된 추천' : '이 공 던지기 전 추천';
  const absentText = maskPublished ? '새 내용을 확인하고 있어요.' : pastUnconfirmed ? '확인 중이에요.' : nextPhase ? '추천 기록이 없어요.' : ABSENT_PAST;
  const actualTitle = nextPhase ? `직전 투구${lastPitch ? ` · ${inningText(lastPitch.situation)} ${lastPitch.pitch_number}구` : ''}` : `실제 투구${pitch ? ` · ${pitch.pitch_number}구` : ''}`;
  const top = situation?.half === 'Top';
  const info = (p: Person | null) => p ? detail.players?.[String(p.id)] : undefined;
  const openPa = () => openSheet('pa');
  // fail closed: no candidate reaches the row, the drawing or any label unless this state may show it (an unconfirmed past pitch shows none)
  const mainRec = stateSlot ? (shownRec ? rec : null) : pastUnconfirmed ? null : rec;
  const cands = shownRec && mainRec ? mainRec.candidates : [];
  const venue = game.venue ?? '경기장 정보 없음';
  const when = `${shortGameDate(game.kst_date)} ${kstTime(game.start)}${game.double_header ? ` · ${game.game_number}차전` : ''}`;
  const phase = detail.current?.phase ?? detail.phase ?? null;
  const emptyRec = maskPublished ? '확인 중' : nextPhase ? (phase === 'between_innings' ? '공수 교대 중' : phase === 'resolving' ? '결과 확인 중' : '추천 대기 중') : '이 시점에는 추천이 없어요';
  const list = <RecList rec={mainRec} emptyText={stateSlot ? eff!.message ?? eff!.label : emptyRec} onOpen={openRec} lead={held ? '마지막으로 받은 ' : ''} absentText={absentText} />;
  const recHead = stateSlot ? `${held ? '마지막으로 받은 추천' : '추천'}${nextNumber ? ` · ${nextNumber}구` : ''}` : nextPhase ? `추천${nextNumber ? ` · ${nextNumber}구` : ''}`
    : `${recTitle}${shownPitch ? ` · ${shownPitch.pitch_number}구` : ''}`;
  // a past pitch (finished game): the thrown pitch named in one line under the row; folded / being confirmed say so and nothing more
  const actualLine = !shownPitch || !actual ? <span className="sv-muted">실제 투구 기록 없음</span>
    : fold ? <><span className="sv-al-label">실제 투구 · {shownPitch.pitch_number}구</span><button className="sv-fold" onClick={openSync} aria-haspopup="dialog">{FOLD_TEXT}</button></>
    : pastUnconfirmed ? <><span className="sv-al-label">실제 투구 · {shownPitch.pitch_number}구</span><span className="sv-muted">확인 중 · 이 공을 다시 확인하고 있어요.</span></>
    : <><span className="sv-al-label">실제 투구 · {shownPitch.pitch_number}구 · {shownPitch.batter.name ?? '타자 미확인'}</span>
      <G1Dot kind={resultKind(actual.description)} n={shownPitch.pitch_number} /><b className="sv-actual-name">{actual.pitch_label}</b>
      {actual.speed_mph !== null && <small>{speedText(actual.speed_mph, unit)}</small>}<span>· {ending ?? actual.result_label}</span>
      {settledCorrection(shownPitch.receipt) && <span className="sv-tag" role="note" aria-label="기록이 정정된 공">정정됨</span>}</>;
  const top1 = !nextPhase && !pastUnconfirmed && rec?.status === 'ready' ? rec.candidates[0] ?? null : null;
  const pairText = top1 && actual && !fold ? { rec: pickText(top1.pitch_label, top1.zone_label), actual: pickText(actual.pitch_label, actual.zone_label) } : null;
  // New as-of player context is optional; absent fields stay unknown rather than guessed from an incomplete pitch list.
  const context = latest ? detail.current_context : pitch?.context;
  const bt = context?.batter;
  const pt = context?.pitcher;
  const isDemo = detail.session !== null;
  // Header status button (UI 10/26 ⑥): ONE marker tells the whole state; its sentence opens in a bubble. Order: connection trouble,
  // then a late feed, then a changed recommendation, then the pending pitch's own state. A finished game has no marker.
  const recvTime = lastOk !== null ? ` · ${kstTime(new Date(lastOk).toISOString())} 수신` : '';
  const chip: { label: string; tone: Tone; state: string; lines: string[] } | null = (() => {
    if (stateSlot) {
      const e = eff!, lines = [e.message, note].filter((x): x is string => !!x);
      if (e.state === 'disconnected') return { label: e.label, tone: 'lost', state: e.state, lines: [...lines, `${e.label}${recvTime}`] };
      if (e.state === 'feed_delayed') return { label: e.label, tone: 'late', state: e.state, lines: lagText ? [...lines, lagText] : lines };
      if (note) return { label: '추천 바뀜', tone: 'fixed', state: e.state, lines };
      if (e.state === 'unsupported' || e.state === 'unavailable') return { label: e.label, tone: 'none', state: e.state, lines };
      if (e.state === 'pregame') return { label: e.label, tone: 'pregame', state: e.state, lines };
      // design 09: an inning break of a game in play still reads 라이브 (row 3 says 공수 교대)
      if (e.state === 'recommended' || (e.state === 'waiting' && breakNow)) return { label: '라이브', tone: 'live', state: e.state, lines: lines.length ? lines : [`실시간으로 받는 중${recvTime}`] };
      // unconfirmed, waiting, ledger_unavailable: the neutral 확인 중 look with the server's own words
      return { label: e.label, tone: 'checking', state: e.state, lines };
    }
    if (game.status === 'final') return null;
    // no display state (archive, demo, older server): the transport story as before the contract
    if (error || feedDown || clientStale || disp?.state === 'disconnected') return { label: '연결 끊김', tone: 'lost', state: 'disconnected', lines: [error?.text ?? CLIENT_LOST.message] };
    if (maskPublished || publishChecking) return { label: '확인 중', tone: 'checking', state: 'unconfirmed', lines: [PUBLISH_CHECKING.message] };
    if (isDemo && detail.session_status === 'ended') return { label: '시연 종료', tone: 'checking', state: 'demo', lines: ['지난 경기 시연이 끝났어요.'] };
    if (isDemo && detail.session_status === 'ready') return { label: '시연 대기', tone: 'checking', state: 'demo', lines: ['지난 경기 시연이 곧 시작돼요.'] };
    if (isDemo && detail.session_status === 'paused') return { label: '일시정지', tone: 'checking', state: 'demo', lines: ['지난 경기 시연이 멈춰 있어요.'] };
    if (live && (!auto || behind > 0 || publishStale || !!lagText || disp?.state === 'feed_delayed'))
      return { label: '딜레이', tone: 'late', state: 'delayed', lines: [publishStale ? PUBLISH_LATE.message : lagText ?? (behind > 0 ? `${behind}초 늦춰 보여 드리고 있어요.` : '새 내용이 늦어지고 있어요.')] };
    if (live) return isDemo ? { label: '시연 중', tone: 'checking', state: 'demo', lines: ['지난 경기를 다시 보여 드리고 있어요.'] } : { label: '라이브', tone: 'live', state: 'live', lines: [`실시간으로 받는 중${recvTime}`] };
    return game.status === 'scheduled' ? { label: '경기 전', tone: 'pregame', state: 'pregame', lines: ['경기 시작 전이에요.'] } : { label: '대기', tone: 'checking', state: 'waiting', lines: ['지금은 경기가 진행 중이 아니에요.'] };
  })();
  const canRetry = !!(error || stale || publishStale);
  const playerRole = sheet === 'player-p' ? 'p' : 'b';
  const sheetPerson = playerRole === 'b' ? batterP : pitcherP;
  const sheetInfo = info(sheetPerson);
  // the guarded current probability (undecided sheet, content unchanged): reached from the 기록 sheet until its place is designed
  const canShowWe = latest && !maskPublished && !error && !stale && !publishChecking && !lateNow && phase === 'pre_pitch' && disp?.state === 'recommended' && disp.shown && detail.win_expectancy?.status === 'available';
  // sheets that list the game's record follow the pitch on screen: a past pitch shows the record up to that pitch, the latest shows all
  const asOfIndex = latest ? null : (shownPitch?.index ?? null);
  // public bundle: a published replay of a finished game that carries no win-expectancy slot shows no 승리확률 entry at all
  const weKnown = !__PZ_STATIC__ || live || detail.win_expectancy != null;
  const homeProbability = canShowWe && typeof detail.win_expectancy?.home_probability === 'number' && Number.isFinite(detail.win_expectancy.home_probability)
    && detail.win_expectancy.home_probability >= 0 && detail.win_expectancy.home_probability <= 1 ? detail.win_expectancy.home_probability : null;
  const shownLine = latest || game.status === 'final' ? detail.linescore_asof : pitch?.line;
  const abbr = (k: 'away' | 'home') => game[`${k}_abbr`] || short(game[`${k}_team`]);
  // a finished game's own header (figure 11) only while the screen is at the end; a selected past pitch keeps its as-of header
  const isFinal = game.status === 'final';
  const pregameHead = !isFinal && !situation && !ended && (game.status === 'scheduled' || disp?.state === 'pregame') && pitches.length === 0;
  const curInning = isFinal ? null : (situation ?? ended)?.inning ?? null;
  // the half after the one that just ended, and its first batter only when the server names it (never guessed)
  const nextHalf = ended ? (ended.half === 'Top' ? `${ended.inning}회말` : `${ended.inning + 1}회초`) : '';
  const batterLine = <>오늘 {part(bt?.at_bats, '타수')} {part(bt?.hits, '안타')}</>;
  const pitcherLine = <>오늘 {typeof pt?.pitches === 'number' ? <><b>{pt.pitches}</b>구</> : '—'}</>;

  return <SheetFrame.Provider value="c"><div className={`sv-page sv-game${disp ? ' has-display' : ''}`}>
    <header className="sv-hd">
      <div className="sv-hd-1">
        <a className="sv-back" href={backHref()} aria-label="경기 목록으로"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" /></svg></a>
        <h1 className="sv-hd-score" aria-label={pregameHead ? `${short(game.away_team)} 대 ${short(game.home_team)}, ${kstTime(game.start)} 시작 (한국 시간)` : `${short(game.away_team)} ${awayScore ?? '–'} 대 ${homeScore ?? '–'} ${short(game.home_team)}. ${venue}, ${when} 한국 시간`}>
          {pregameHead ? <span aria-hidden="true"><span className="t">{abbr('away')}</span><i>vs</i><span className="t">{abbr('home')}</span></span>
            : <span aria-hidden="true"><span className="t">{abbr('away')}</span><strong>{awayScore ?? '–'}</strong><i>:</i><strong>{homeScore ?? '–'}</strong><span className="t">{abbr('home')}</span></span>}
        </h1>
        {chip && <button className="sv-hd-status" data-tone={chip.tone} data-state={chip.state} aria-expanded={pop} aria-controls="sv-status-pop" onClick={() => setPop(o => !o)}
          aria-label={`${isDemo ? '지난 경기 시연. ' : ''}상태: ${chip.label}. 설명 보기`}><i aria-hidden="true" /><span aria-live="polite">{chip.label}</span></button>}
      </div>
      <div className="sv-hd-2">
        {pregameHead ? <span className="sv-inning static">{kstTime(game.start)} 시작</span>
          : isFinal ? <button className="sv-inning" onClick={openPa} aria-haspopup="dialog" aria-label="경기 종료, 타석 선택">경기 종료</button>
          : situation ? <button className="sv-inning" onClick={openPa} aria-haspopup="dialog" aria-label={`${inningText(situation)}, 타석 선택`}>{situation.inning}회{situation.half === 'Top' ? '초' : '말'} ▾</button>
          : ended ? <button className="sv-inning" onClick={openPa} aria-haspopup="dialog" aria-label={`${inningText(ended)} 종료, 지난 타석 선택`}>{ended.inning}회{ended.half === 'Top' ? '초' : '말'} 종료</button> : <span className="sv-inning static">—</span>}
        {!pregameHead && <LineScore ls={shownLine} away={abbr('away')} home={abbr('home')} awayR={awayScore} homeR={homeScore} current={curInning} onOpen={() => openSheet('box')} />}
      </div>
    </header>
    {chip && pop && <StatusPop chip={chip} onClose={() => setPop(false)} onRetry={canRetry ? () => void load(true, true) : null} busy={busy} />}
    {breakNow ? <section className="sv-mu line" aria-label="경기 상황"><p>공수 교대 · 다음 {nextHalf} 첫 타자 <b>—</b></p></section>
      : !pregameHead && !isFinal && (situation || idle) ? <section className="sv-mu" aria-label="경기 상황">{idle && !situation ? <p className="sv-mu-idle">타자·투수 정보 없음</p> : <>
        <Who role="b" person={batterP} hand={batterP?.side} team={top ? game.away_team : game.home_team} today={batterLine} onOpen={() => openSheet('player-b')} />
        {situation ? <CountBox s={situation} /> : <div className="sv-mu-count" />}
        <Who role="p" person={pitcherP} hand={pitcherP?.hand} team={top ? game.home_team : game.away_team} today={pitcherLine} onOpen={() => openSheet('player-p')} />
      </>}</section> : null}

    <main className="sv-live-layout">
    <section className="sv-stage" aria-label={nextPhase ? '다음 공 추천' : '이 공의 추천과 실제 투구'}>
      <h2 className="sr-only">{stateSlot && !shownRec ? `다음 공: ${eff!.label}` : held ? '마지막으로 받은 추천' : '추천'}</h2>
      {/* Pitcher's view, with the batter on the correct screen side; the zone owns the centre. */}
      <div className={`sv-zrow ${side === 'L' ? 'bat-l' : 'bat-r'}${held ? ' held' : ''}`}>
        {side && <div className={`sv-batband ${side === 'R' ? 'at-right' : 'at-left'}`}><Silhouette side={side} /></div>}
        <div className="sv-zone-wrap">
          <Zone bounds={detail.zone_bounds} candidates={cands} actuals={thrown}
            actual={!nextPhase && !fold && !pastUnconfirmed && actual && actual.x !== null && actual.z !== null ? { x: actual.x, z: actual.z } : null} actualLabel=""
            actualKind={resultKind(actual?.description)} actualNumber={!nextPhase && shownPitch ? shownPitch.pitch_number : null}
            setupX={!nextPhase && !fold ? actual?.setup_x ?? null : null} />
        </div>
      </div>
      <ZoneLegend side={side} />
      {/* ③ one ruled row of three; the state's sentence and change note live in the header bubble and in this polite region */}
      <section className={`sv-rec-sec sv-pitch-summary${stateSlot ? ' sv-state' : ''}${held ? ' held' : ''}`} data-state={stateSlot ? eff!.state : undefined} data-pitch={stateSlot ? disp!.pitch_key ?? undefined : undefined} aria-label="추천">
        <div className="sv-rec-head"><h2>{recHead}</h2>
          <button className="sv-state-more" onClick={openRec} aria-haspopup="dialog" aria-label="추천 자세히 보기">자세히</button></div>
        {list}
        {stateSlot && <div className="sv-state-live sr-only" role="status"><p className="sv-state-msg">{eff!.message}</p><p className="sv-state-note" data-kind={note ? (fix ? fix.kind : 'client') : undefined}>{note}</p></div>}
        {reference && eff!.state === 'unsupported' && !note && <ReferenceTeaser card={reference} onOpen={() => openSheet('reference')} />}
        {!nextPhase && <p className="sv-actual-line">{actualLine}</p>}
      </section>
      {/* ④ the at-bat in one line; opened in place as a list, newest first, with the pending pitch (or the result) on top */}
      {paKey && !breakNow && !pregameHead && <section className="sv-atbat" aria-label="이번 타석">
        <div className="sv-atbat-head"><h2>이번 타석 · {paPitches.length}구 던짐</h2>
          <button className="sv-atbat-toggle" aria-expanded={abOpen} aria-controls="sv-atbat-list" onClick={() => setAbOpen(o => !o)}>{abOpen ? '접기 ▴' : '펼치기 ▾'}</button></div>
        {abOpen && <ol className="sv-atbat-list" id="sv-atbat-list">
          {nextPhase && nextNumber && paKey === detail.current?.pa_key && <li className="next"><span className="sv-rk next" aria-hidden="true">{nextNumber}</span><b>{nextNumber}구 · 다음 공</b></li>}
          {!nextPhase && ending && <li className="result"><b>{ending}</b></li>}
          {[...paPitches].reverse().map(p => { const on = selectedPitch?.key === p.key; const a = p.actual; const un = unconfirmed(p); const fixed = !un && settledCorrection(p.receipt);
            if (fold) return <li key={p.key}><button className="sv-pitch-row folded" aria-pressed={on} onClick={openSync} aria-haspopup="dialog" aria-label={`${p.pitch_number}구, ${FOLD_TEXT}`}>
              <G1Dot kind="unknown" n={p.pitch_number} /><span className="sv-pr-what">{FOLD_TEXT}</span></button></li>;
            return <li key={p.key}><button className={`sv-pitch-row${un ? ' unconfirmed' : ''}`} aria-pressed={on} onClick={() => pick(p.index)} aria-haspopup="dialog"
              aria-label={un ? `${p.pitch_number}구 확인 중` : `${p.pitch_number}구 ${a?.pitch_label || '구종 미상'}${a?.speed_mph != null ? ` ${speedText(a.speed_mph, unit)}` : ''} ${a?.result_label ?? ''}${fixed ? ', 기록이 정정된 공' : ''}`}>
              <G1Dot kind={un ? 'unknown' : resultKind(a?.description)} n={p.pitch_number} />
              {un ? <span className="sv-pr-what" aria-hidden="true">확인 중</span> : <span className="sv-pr-what" aria-hidden="true"><b>{a?.pitch_label || '구종 미상'}</b>{a?.speed_mph != null && <small>{speedText(a.speed_mph, unit)}</small>}</span>}
              {fixed && <i className="sv-card-tag" aria-hidden="true">정정됨</i>}
              <span className="sv-pr-res" aria-hidden="true">{un ? '–' : a?.result_label || '기록 없음'}</span></button></li>; })}
        </ol>}
      </section>}
    </section>
    </main>
    <nav className="sv-bottom-nav" aria-label="라이브 상세">
      <button onClick={openRec} aria-haspopup="dialog"><NavIcon d={ICONS.rec} />추천</button>
      <button onClick={() => openSheet('player-b')} aria-haspopup="dialog"><NavIcon d={ICONS.player} />선수</button>
      <button onClick={() => openSheet('mix')} aria-haspopup="dialog"><NavIcon d={ICONS.mix} />구종</button>
      <button onClick={() => openSheet('history')} aria-haspopup="dialog"><NavIcon d={ICONS.record} />기록</button>
      {viewer && <button className="sv-sync-open" onClick={openSync} aria-haspopup="dialog" aria-label={`중계 맞추기. 지금 ${behind > 0 ? `${behind}초 늦춰 보는 중` : '늦추지 않고 보는 중'}${setting.synced ? '' : ', 아직 맞추지 않았어요'}`}>{setting.synced ? '맞추기' : '중계 맞추기'}</button>}</nav>

    {!maskPublished && sheet === 'pa' && <PaSheet pas={detail.pas} selected={paKey} onClose={closeSheet} onBox={() => setSheet('box')} foldText={fold ? FOLD_TEXT : null}
      onPick={p => {
        // a finished plate appearance with no pitch at all (intentional walk and the like): its own result, no jump elsewhere
        if (p.complete && !pitches.some(x => x.pa_key === p.pa_key) && (typeof p.first !== 'number' || !byIndex.has(p.first))) { setEmptyPa(p); setSheet('pa-empty'); return; }
        setSheet(null); if ((live && p.pa_key === detail.current?.pa_key && !p.complete) || typeof p.first !== 'number' || !byIndex.has(p.first)) goLatest(); else pick(p.first); }} />}
    {!maskPublished && sheet === 'pa-empty' && emptyPa && <Sheet title={`${inningText(emptyPa)} · ${emptyPa.batter}`} onClose={closeSheet}>
      <p><b>{fold ? FOLD_TEXT : emptyPa.result}</b></p><p className="sv-muted">이 타석은 투구 기록이 없어요. 공을 던지지 않고 끝난 타석이에요.</p>
    </Sheet>}
    {/* a selected past pitch shows the linescore as of that pitch (never the latest one, which would reveal later play) */}
    {sheet === 'box' && <BoxSheet ls={latest ? detail.linescore_asof : pitch!.line ?? null} game={game} at={situation} onClose={closeSheet} />}
    {!maskPublished && sheet === 'detail' && !selectedPitch && <DetailSheet bounds={detail.zone_bounds} onClose={closeSheet} rec={stateSlot && !shownRec ? null : rec} recTitle={stateSlot && !shownRec ? `다음 공: ${eff!.label}` : held ? `마지막으로 받은 추천${nextNumber ? ` · ${nextNumber}구` : ''}` : recTitle}
      pair={pairText} absentText={absentText}
      actual={nextPhase ? null : actual} actualTitle={actualTitle} paResult={ending} actualFolded={fold && !nextPhase && actual !== null ? FOLD_TEXT : null}
      actualExtra={!nextPhase && !fold && shownPitch?.expectation && shownPitch.actual?.event ? <ExpectationLine e={shownPitch.expectation} /> : null} notes={detail.notes} speed={mph => speedText(mph, unit)}
      recNote={stateSlot ? [eff!.message, note].filter(Boolean).join(' ') || null : pastUnconfirmed ? '이 공을 확인하는 동안에는 추천을 함께 보여 드리지 않아요.' : null} actualPending={!nextPhase && unconfirmed(shownPitch)} actualCorrected={!nextPhase && shownPitch?.receipt?.corrected === true} actualHeld={!nextPhase && heldReceipt(shownPitch?.receipt)}
      extra={<div className="sv-row"><button className="sv-link-text" onClick={() => setSheet('info')}>기준과 한계 보기</button>
        <button className="sv-link-text" onClick={() => { const u = unit === 'kmh' ? 'mph' : 'kmh'; setUnit(u); try { localStorage.setItem(UNIT_KEY, u); } catch { /* ignore */ } }}
          aria-label={`속도 단위 바꾸기, 지금 ${unit === 'kmh' ? 'km/h' : 'mph'}`}>속도 단위: {unit === 'kmh' ? 'km/h' : 'mph'}</button></div>} />}
    {!maskPublished && sheet === 'detail' && selectedPitch && (() => { const at = pitches.indexOf(selectedPitch);
      return <PitchSheet pitch={selectedPitch} bounds={detail.zone_bounds} onClose={closeSheet} speed={mph => speedText(mph, unit)}
        onPrev={at > 0 ? () => pick(pitches[at - 1].index) : null} onNext={at >= 0 && at < pitches.length - 1 ? () => pick(pitches[at + 1].index) : null}
        paResult={!fold && selectedPitch.actual?.event ? detail.pas.find(p => p.pa_key === selectedPitch.pa_key && p.complete)?.result ?? null : null}
        folded={fold ? FOLD_TEXT : null} pending={unconfirmed(selectedPitch)} corrected={selectedPitch.receipt?.corrected === true} held={heldReceipt(selectedPitch.receipt)}
        actualExtra={!fold && selectedPitch.expectation && selectedPitch.actual?.event ? <ExpectationLine e={selectedPitch.expectation} /> : null} />; })()}
    {!maskPublished && sheet === 'history' && <RecordSheet detail={detail} fold={fold} onClose={closeSheet} onPick={pick} asOfIndex={asOfIndex}
      headExtra={weKnown ? <button className="sv-sheet-link" onClick={() => setSheet('win')} aria-haspopup="dialog">승리확률</button> : undefined} />}
    {sheet === 'win' && <Sheet title="현재 상황의 승리확률 · 모델 추정" onClose={closeSheet}>{homeProbability === null
      ? <p className="sv-muted" role="status">현재 승리확률 정보 없음</p>
      : <div className="sv-we-bars" role="img" aria-label={`현재 원정팀 ${short(game.away_team)} 승리확률 ${Math.round((1 - homeProbability) * 100)}퍼센트, 홈팀 ${short(game.home_team)} 승리확률 ${Math.round(homeProbability * 100)}퍼센트`}>
        {([['away', game.away_team, 1 - homeProbability], ['home', game.home_team, homeProbability]] as const).map(([sideName, team, value]) => <div className="sv-we-bar" key={sideName} aria-hidden="true"><span>{short(team)}</span><div><i style={{ width: `${value * 100}%`, background: teamColor(team) }} /></div><b>{Math.round(value * 100)}%</b></div>)}</div>}
      <p className="sv-muted">타석 시작 상황으로 학습한 모델 추정이며 볼·스트라이크 카운트는 반영하지 않아요.</p></Sheet>}
    {!maskPublished && sheet === 'mix' && <PitchMixSheet detail={detail} pitcherId={pitcherP?.id ?? null} pitcherName={pitcherP?.name ?? null} fold={fold} onClose={closeSheet} asOfIndex={asOfIndex} />}
    {(sheet === 'player-b' || sheet === 'player-p') && <Sheet title={sheetPerson?.name ?? '선수 정보'} onClose={closeSheet} half={playerRole === 'p' ? 'max(56dvh, min(420px, calc(100dvh - 56px)))' : null}>
      <div className="sv-tabs" role="tablist" aria-label="선수">{(['b', 'p'] as const).map(r => <button key={r} role="tab" aria-selected={playerRole === r} onClick={() => setSheet(r === 'b' ? 'player-b' : 'player-p')}>{r === 'b' ? '타자' : '투수'}</button>)}</div>
      <div role="tabpanel">
      <p className="sv-player-title"><b>{sheetPerson?.name ?? '—'}</b> <span>{short((playerRole === 'b') === top ? game.away_team : game.home_team)}{handText(playerRole === 'b' ? batterP?.side : pitcherP?.hand, playerRole) ? ` · ${handText(playerRole === 'b' ? batterP?.side : pitcherP?.hand, playerRole)}` : ''}</span></p>
      <dl className="sv-dl sv-facts"><dt>등번호</dt><dd>{sheetInfo?.jersey ? `#${sheetInfo.jersey}` : '—'}</dd>
        <dt>시즌</dt><dd>{playerRole === 'b' ? `AVG ${stat3(sheetInfo?.avg)} · OPS ${stat3(sheetInfo?.ops)}` : `ERA ${era(sheetInfo?.era)}`}</dd>
        <dt>오늘</dt><dd>{playerRole === 'b' ? [part(bt?.at_bats, '타수'), part(bt?.hits, '안타'), part(bt?.on_base, '출루')].join(' · ') : [part(pt?.pitches, '구'), part(pt?.strikeouts, '삼진'), part(pt?.earned_runs, '자책')].join(' · ')}</dd></dl>
      {playerRole === 'p' && !maskPublished && <MixTable detail={detail} pitcherId={pitcherP?.id ?? null} fold={fold} heading />}
      <p className="sv-muted">{sheetInfo?.stats_as_of ? `${sheetInfo.stats_as_of} 경기 전 시즌 기록` : '경기 전 시즌 기록 미확인'}. 오늘 기록은 현재 공식 피드의 해당 시점까지 사건을 재구성한 값으로 이후 기록 정정이 반영될 수 있습니다. 출루는 안타+볼넷+몸에 맞는 공입니다. 확인되지 않은 값은 —로 표시합니다.</p>
      </div>
    </Sheet>}
    {sheet === 'info' && <InfoSheet onClose={closeSheet} notes={detail.notes} />}
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

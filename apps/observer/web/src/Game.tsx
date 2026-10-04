import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { ZONE_BOUNDS, get, short } from './WatchAlong';
import type { Actual, Bounds, Candidate, Situation } from './WatchAlong';
import './game.css';

/** One-screen game page (D135–D140, docs/handoffs/D-UI-004-frame.md): the zone stays put and the stage above it
 *  follows the game by itself — next pitch, pitch just thrown, plate-appearance result, break, no recommendation.
 *  Same `/api/live/{game}/state` feed as Live.tsx; nothing here changes the model or what it is asked. */

type Pitched = Actual & { description?: string | null };
type Pre = { status: 'ready' | 'unsupported'; reason: string | null; recommendation: { candidates: Candidate[] } | null };
type PaPitch = { key: string; pitch_number: number; status: string; pre: Pre; actual: Pitched | null };
type LiveRec = { status: 'ready' | 'unsupported' | 'computing' | 'unavailable'; reason: string | null; key?: string;
  recommendation?: { candidates: Candidate[] } | null; pa_pitches?: PaPitch[];
  previous?: { key: string; actual: Pitched; inning: number; half: 'Top' | 'Bot'; home_we_before: number | null } | null;
  home_we_now?: number | null; policy_identity?: string };
type Replay = { start: number; t0: number; speed: number; elapsed_s: number; pitches: number; index: number };
type Mix = { pitch_type: string; pitch_label: string; count: number; share: number; avg_mph: number | null };
type LiveState = { status: string; reason: string | null; delay_s: number; replay?: Replay | null;
  sync?: { max_delay_s: number; event_age_s?: number; feed_lag_s?: number; feed_ok?: boolean };
  game?: { game_pk: number; date: string; status: string; away_team: string; home_team: string };
  situation?: Situation | null; pitcher?: { id: number; name: string | null; hand: string } | null;
  batter?: { id: number; name: string | null; side: string } | null;
  pitcher_today?: { pitches: number; mix: Mix[] } | null;
  last_pitch?: { key: string; pitch_type: string | null; pitch_label: string; speed_mph: number | null; result_label: string | null; play_text: string | null } | null;
  recommendation: LiveRec | null };
type Cell = { col: number; row: number };  // catcher's view; row 0 is the top of the zone
type Kind = 'strike' | 'ball' | 'play';
type Seen = { key: string; pa: string; n: number; side: string; top: Candidate | null; actual: Pitched };
type Unit = 'kmh' | 'mph';
type SheetName = 'detail' | 'box' | 'sync' | 'more' | { glossary: string };
type Outline = { pitches: number; halves: { label: string; first: number }[] };
// Catcher setup read from the broadcast video after the game (D151, UI-setup-display-work-order): horizontal only.
type SetupView = { badge?: string; pitches: { at_bat_number: number; pitch_number: number; status: string; plate_x_feet: number | null }[] };
type Setups = { badge: string | null; x: Map<string, number> };

const POLL_MS = 2000;
// A state request with no answer by then is dropped and counts as a failed poll. Measured /state answers (recorder polls
// through the Tailscale address, 4 games 10/2–10/4, 13,106 answers, 0 timeouts): median about 0.9 s, max 4.6 s. 15 s is
// about 3× that max; phone-network time on top of it is not measured (docs/reviews/PM-game-timeout-2026-10-04.md).
const STATE_TIMEOUT_MS = 15000;
const HOLD_MS = { pitch: 6000, event: 10000 };  // how long a result stays before the next recommendation takes the stage (F13: proposal)
const DELAY_KEY = 'pitcheezy.live.delay', UNIT_KEY = 'pitcheezy.unit';
const WAIT: Record<string, [string, string]> = { resolving: ['결과 확인 중', '결과를 기다려요'], buffering: ['준비 중', '내 중계에 맞추는 중'], between_innings: ['공수 교대', '공수 교대'],
  not_live: ['경기 없음', '지금은 경기 중이 아니에요'], unsupported_game_type: ['지원 안 함', '지원하지 않는 경기' ] };
// Plain words for each pitch type (F23): fixed sentences, not model output.
const GLOSSARY: Record<string, string> = { FF: '가장 빠른 직구', SI: '빠르게 가라앉는 직구', FC: '끝에서 살짝 꺾이는 빠른 공', SL: '옆으로 휘면서 떨어지는 공',
  ST: '옆으로 크게 휘는 공', SV: '슬라이더와 커브 사이로 휘는 공', CU: '크게 떨어지는 느린 공', KC: '빠르게 뚝 떨어지는 커브',
  CH: '직구처럼 보이는 느린 공', FS: '직구처럼 오다 뚝 떨어지는 공' };

const stored = (key: string) => { try { return localStorage.getItem(key); } catch { return null; } };
const store = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch { /* storage may be unavailable */ } };
const clock = (seconds: number) => { const s = Math.max(0, Math.round(seconds)); return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };
const pct = (v: number | null | undefined) => v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`;
const paOf = (key: string) => key.split(':').slice(0, 2).join(':');
const timeText = (ms: number) => new Date(ms).toLocaleTimeString('ko-KR', { hour12: false });
const SETUP_NOTE = '포수 미트 위치는 영상에서 읽은 추정이며 검토 전입니다.';
const speedText = (mph: number | null, unit: Unit) => mph === null ? '' : unit === 'kmh' ? `${Math.round(mph * 1.609344)} km/h` : `${mph.toFixed(1)} mph`;

const cellOf = (zoneId: string | null): Cell | null => {
  const [height, side] = (zoneId ?? '').split('_');
  const row = ['high', 'middle', 'low'].indexOf(height), col = ['left', 'middle', 'right'].indexOf(side);
  return row < 0 || col < 0 ? null : { col, row };
};
function cellAt(x: number, z: number, b: Bounds): Cell & { out: boolean } {
  const third = (b.top - b.bottom) / 3;
  return { col: x < -.83 / 3 ? 0 : x > .83 / 3 ? 2 : 1, row: z > b.top - third ? 0 : z < b.bottom + third ? 2 : 1, out: Math.abs(x) > .83 || z < b.bottom || z > b.top };
}
/** Batter-relative words (D138): to a right-handed batter the catcher's left is inside. */
function words(cell: Cell, side: string) {
  if (cell.col === 1 && cell.row === 1) return '한가운데';
  const across = cell.col === 1 ? '가운데' : (cell.col === 0) === (side !== 'L') ? '몸쪽' : '바깥쪽';
  const height = ['높게', '', '낮게'][cell.row];
  return height ? `${across} ${height}` : across;
}
function kind(a: Pitched): Kind {
  const code = a.description ?? '', label = a.event_label ?? a.result_label ?? '';
  if (/ball|hit_by_pitch|pitchout/.test(code) || /^볼|몸에 맞/.test(label)) return 'ball';
  if (code === 'hit_into_play' || /인플레이|안타|루타|홈런|아웃|병살|희생/.test(label)) return 'play';
  return 'strike';
}
const matches = (top: Candidate | null, a: Pitched, b: Bounds) => {
  if (!top || !a.pitch_type) return null;
  const want = cellOf(top.zone_id), got = a.x !== null && a.z !== null ? cellAt(a.x, a.z, b) : null;
  return { type: top.pitch_type === a.pitch_type, spot: want && got ? !got.out && want.col === got.col && want.row === got.row : null };
};

function Lights({ letter, on, total, tone }: { letter: string; on: number; total: number; tone: string }) {
  return <span className={`gm-lights ${tone}`} role="img" aria-label={`${letter} ${on}`}><b>{letter}</b>{Array.from({ length: total }, (_, i) => <i key={i} className={i < on ? 'on' : ''} />)}</span>;
}
function Diamond({ mask }: { mask: number }) {
  const on = (base: number) => (mask & (1 << (base - 1))) !== 0;
  return <svg className="gm-bases" viewBox="0 0 52 36" role="img" aria-label={`주자 ${[1, 2, 3].filter(on).join(', ') || '없음'}`}>
    {[[26, 8, 2], [42, 22, 1], [10, 22, 3]].map(([x, y, base]) => <rect key={base} x={x - 6} y={y - 6} width="12" height="12" transform={`rotate(45 ${x} ${y})`} className={on(base) ? 'on' : ''} />)}</svg>;
}
function Board({ s, away, home, pitcher, pitcherNote, batter, batterNote, idle }: { s: Situation; away: string; home: string; pitcher: string | null; pitcherNote: string; batter: string | null; batterNote: string; idle: boolean }) {
  return <section className="gm-board" aria-live="polite">
    <div className="gm-score"><span>{short(away)}</span><strong>{s.away_score}</strong><i>:</i><strong>{s.home_score}</strong><span>{short(home)}</span></div>
    <div className={`gm-count ${idle ? 'idle' : ''}`}><strong>{s.inning}회 {s.half === 'Top' ? '초' : '말'}</strong>
      <Lights letter="B" on={idle ? 0 : s.balls} total={3} tone="ball" /><Lights letter="S" on={idle ? 0 : s.strikes} total={2} tone="strike" /><Lights letter="O" on={idle ? 0 : s.outs} total={2} tone="out" /><Diamond mask={idle ? 0 : s.bases} /></div>
    <div className="gm-matchup">{pitcher && batter && <><span><em>투</em><b>{pitcher}</b><small>{pitcherNote}</small></span><span><em>타</em><b>{batter}</b><small>{batterNote}</small></span></>}</div>
  </section>;
}

type Dot = { key: string; n: number; x: number; z: number; kind: Kind };
const VW = 370, VH = 285, BW = 124, TOP = 68;
/** Catcher's view. The batter stands where he really does, so inside and outside read off the picture (D138). */
function Zone({ side, target, dots, big, empty, mini, setupX }: { side: string; target: Cell | null; dots: Dot[]; big?: string | null; empty?: boolean; mini?: boolean; setupX?: number | null }) {
  const b = ZONE_BOUNDS, bh = BW * (b.top - b.bottom) / 1.66, left = (VW - BW) / 2, cw = BW / 3, ch = bh / 3;
  const x = (v: number) => VW / 2 + Math.max(-1.3, Math.min(1.3, v)) / .83 * (BW / 2);
  const y = (v: number) => TOP + (b.top - Math.max(.7, Math.min(4.2, v))) / (b.top - b.bottom) * bh;
  const flip = side === 'L' ? -1 : 1, cx = VW / 2 - flip * BW;  // batter: catcher's left for a right-hander
  const centre = target && { x: left + cw * (target.col + .5), y: TOP + ch * (target.row + .5) };
  const thrown = big ? dots.find(d => d.key === big) : null;
  return <svg className={`gm-zone ${mini ? 'mini' : ''}`} viewBox={mini ? `${left - 30} ${TOP - 14} ${BW + 60} ${bh + 90}` : `0 0 ${VW} ${VH}`} role="img" aria-label="포수 쪽에서 본 스트라이크존">
    {!mini && <g className="gm-batter"><circle cx={cx} cy={TOP - bh * .4} r={BW * .11} /><rect x={cx - BW * .15} y={TOP - bh * .24} width={BW * .3} height={bh * .72} rx={BW * .1} />
      <rect x={cx - BW * .13} y={TOP + bh * .44} width={BW * .26} height={VH} rx={BW * .06} />
      <line x1={cx + flip * BW * .16} y1={TOP - bh * .1} x2={cx + flip * BW * .4} y2={TOP - bh * .6} /><text x={cx} y={TOP + bh * .16} textAnchor="middle">타자</text></g>}
    <rect x={left} y={TOP} width={BW} height={bh} rx="3" className={`gm-zone-box ${empty ? 'empty' : ''}`} />
    {[1, 2].map(i => <g key={i} className="gm-zone-grid"><line x1={left + cw * i} x2={left + cw * i} y1={TOP} y2={TOP + bh} /><line y1={TOP + ch * i} y2={TOP + ch * i} x1={left} x2={left + BW} /></g>)}
    {target && centre && <g className={`gm-target ${thrown ? 'faint' : ''}`}><rect x={left + cw * target.col} y={TOP + ch * target.row} width={cw} height={ch} />
      {!thrown && <><circle cx={centre.x} cy={centre.y} r={cw * .26} className="ring" /><circle cx={centre.x} cy={centre.y} r={cw * .08} /></>}</g>}
    {setupX !== undefined && setupX !== null && <g className="gm-setup" role="img" aria-label="포수 미트 가로 위치">
      {/* a vertical line and two edge ticks: where across the plate the mitt was, with no claim about height */}
      <line x1={x(setupX)} x2={x(setupX)} y1={TOP - 4} y2={TOP + bh + 4} />
      <path d={`M${x(setupX) - 7} ${TOP - 13}h14l-7 9z`} /><path d={`M${x(setupX) - 7} ${TOP + bh + 13}h14l-7 -9z`} />
      {!mini && <text x={x(setupX)} y={TOP - 17} textAnchor="middle">미트</text>}</g>}
    {thrown && centre && <line className="gm-miss" x1={centre.x} y1={centre.y} x2={x(thrown.x)} y2={y(thrown.z)} />}
    {dots.map(d => { const r = (d.key === big ? 11.5 : 8.5) * (mini ? 1.3 : 1);
      return <g key={d.key} className={`gm-dot ${d.kind} ${big && d.key !== big ? 'old' : ''}`}><circle cx={x(d.x)} cy={y(d.z)} r={r} /><text x={x(d.x)} y={y(d.z) + r * .38} textAnchor="middle" fontSize={r * 1.1}>{d.n}</text></g>; })}
  </svg>;
}

function Where({ cell, side }: { cell: Cell; side: string }) {
  const centre = cell.col === 1 && cell.row === 1;
  return <p className="gm-where"><svg viewBox="0 0 24 24" aria-hidden="true" style={{ transform: `rotate(${centre ? 0 : Math.atan2(cell.row - 1, cell.col - 1) * 180 / Math.PI}deg)` }}>
    {centre ? <circle cx="12" cy="12" r="5" fill="currentColor" stroke="none" /> : <><path d="M4 12h15" /><path d="M13 6l6 6-6 6" /></>}</svg>{words(cell, side)}</p>;
}
function Marks({ match }: { match: { type: boolean; spot: boolean | null } | null }) {
  if (!match) return null;
  const mark = (label: string, ok: boolean) => <span className={`gm-mark ${ok ? 'ok' : ''}`}>{label}<svg viewBox="0 0 24 24" aria-label={ok ? '같음' : '다름'}>{ok ? <path d="M4 12l5 5L20 6" /> : <><path d="M6 6l12 12" /><path d="M18 6L6 18" /></>}</svg></span>;
  return <>{mark('구종', match.type)}{match.spot !== null && mark('위치', match.spot)}</>;
}
function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => { const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); }; window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey); }, [onClose]);
  return <div className="gm-scrim" onClick={e => { if (e.target === e.currentTarget) onClose(); }}><div className="gm-sheet" role="dialog" aria-modal="true" aria-label={title}>
    <div className="gm-sheet-head"><h2>{title}</h2><button className="gm-icon" onClick={onClose} aria-label="닫기"><svg viewBox="0 0 24 24"><path d="M6 6l12 12" /><path d="M18 6L6 18" /></svg></button></div>
    <div className="gm-sheet-body">{children}</div></div></div>;
}

export default function Game() {
  const params = new URLSearchParams(window.location.search);
  const gamePk = Number(params.get('game'));
  const [replayStart, setReplayStart] = useState<number | null>(() => params.get('replay') === null ? null : Number(params.get('replay')));
  const speed = Number(params.get('speed')) || 1, lag = params.get('lag');
  const [t0, setT0] = useState<number | null>(params.get('t0') ? Number(params.get('t0')) : null);
  const [delay, setDelay] = useState<number | null>(() => { const v = stored(DELAY_KEY); return replayStart !== null ? 0 : v === null ? null : Number(v); });
  const [state, setState] = useState<LiveState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [failures, setFailures] = useState(0);                  // polls failed in a row; any answer resets it
  const [unit, setUnit] = useState<Unit>(() => stored(UNIT_KEY) === 'mph' ? 'mph' : 'kmh');
  const [sheet, setSheet] = useState<SheetName | null>(null);
  const [tab, setTab] = useState(0);
  const [tapNote, setTapNote] = useState<string | null>(null);
  const [hold, setHold] = useState<string | null>(null);        // key of the result holding the stage
  const [past, setPast] = useState<number | null>(null);        // index into the seen pitches while looking back
  const [paused, setPaused] = useState<number | null>(null);    // replay: wall seconds when the viewer paused it
  const [lastGood, setLastGood] = useState<number | null>(null); // when the feed was last known to be current
  const [outline, setOutline] = useState<Outline | null>(null);
  const [setups, setSetups] = useState<Setups | null>(null);
  const seen = useRef(new Map<string, Seen>());                 // every pitch this screen has watched, in order
  const shown = useRef(new Map<string, Candidate | null>());    // recommendation shown before each pitch
  const sides = useRef(new Map<string, string>());              // batter side per plate appearance
  const lastPrev = useRef<string | null>(null), wasReady = useRef(false);
  const board = useRef<{ s: Situation; away: string; home: string } | null>(null);
  const pollNow = useRef<() => void>(() => undefined);           // ask afresh at once instead of waiting for the next poll

  useEffect(() => {
    if (!gamePk || paused !== null) return;  // a paused replay asks nothing; resuming moves t0 by the pause
    let stop = false, timer = 0, ask = 0, out: AbortController | null = null;  // `ask` numbers requests; only the newest counts
    const query = new URLSearchParams();
    if (delay !== null) query.set('delay', String(delay));
    if (replayStart !== null) { query.set('replay', String(replayStart)); query.set('speed', String(speed)); if (lag !== null) query.set('lag', lag); if (t0 !== null) query.set('t0', String(t0)); }
    const tick = async (fresh = false) => {
      if (stop) return;
      // One request at a time. The timer's own poll waits for an answer already on its way; the viewer's retry, coming
      // back to the page or the network returning (fresh) drop that request instead, since it may never answer.
      if (out) { if (!fresh) return; out.abort(); }
      const mine = ++ask, request = new AbortController(); out = request; window.clearTimeout(timer);
      let late = false;  // dropped by the watchdog below, not by a newer ask: a failed poll like any other
      const watchdog = window.setTimeout(() => { late = true; request.abort(); }, STATE_TIMEOUT_MS);
      try {
        const next = await get<LiveState>(`/live/${gamePk}/state?${query}`, request.signal);
        if (stop || mine !== ask) return;
        if (next.replay && t0 === null) {  // pin the replay clock into the link, as Live.tsx does
          const url = new URLSearchParams(window.location.search); url.set('t0', String(next.replay.t0));
          window.history.replaceState(null, '', `?${url}`); setT0(next.replay.t0); return;
        }
        const rec = next.recommendation, side = next.batter?.side ?? 'R';
        if (rec?.key) { sides.current.set(paOf(rec.key), side); if (rec.status === 'ready' || rec.status === 'unsupported') shown.current.set(rec.key, rec.recommendation?.candidates[0] ?? null); }
        const remember = (key: string, n: number, top: Candidate | null, actual: Pitched) => { if (!seen.current.has(key)) seen.current.set(key, { key, pa: paOf(key), n, side: sides.current.get(paOf(key)) ?? side, top, actual }); };
        rec?.pa_pitches?.forEach(p => { if (!shown.current.has(p.key)) shown.current.set(p.key, p.pre.recommendation?.candidates[0] ?? null); if (p.actual) remember(p.key, p.pitch_number, shown.current.get(p.key) ?? null, p.actual); });
        if (rec?.previous) remember(rec.previous.key, Number(rec.previous.key.split(':')[2]) || 1, shown.current.get(rec.previous.key) ?? null, rec.previous.actual);
        const prev = rec?.previous?.key ?? null;
        if (wasReady.current && prev && prev !== lastPrev.current) setHold(prev);  // a pitch just landed: it takes the stage first
        wasReady.current = next.status === 'ready' || next.status === 'resolving';  // a result still being posted is the same at-bat, not a break
        if (next.status === 'ready') lastPrev.current = prev;
        if (next.situation && next.game) board.current = { s: next.situation, away: next.game.away_team, home: next.game.home_team };
        setState(next); setError(null); setFailures(0);
        if (next.sync?.feed_ok !== false) setLastGood(Date.now());
      } catch (e) {
        if (!stop && mine === ask) { setError(late ? `관전 서버가 ${STATE_TIMEOUT_MS / 1000}초 동안 답하지 않았어요.` : (e as Error).message); setFailures(n => n + 1); }
      } finally { window.clearTimeout(watchdog); if (mine === ask) out = null; }
      if (!stop && mine === ask) timer = window.setTimeout(() => void tick(), document.hidden ? POLL_MS * 5 : POLL_MS);
    };
    // Back from the broadcast app, or the network is back: ask now rather than wait out the slower hidden-page poll.
    const wake = () => { if (!document.hidden) void tick(true); };
    document.addEventListener('visibilitychange', wake); window.addEventListener('online', wake);
    pollNow.current = () => void tick(true);
    void tick();
    return () => { stop = true; window.clearTimeout(timer); out?.abort(); pollNow.current = () => undefined;
      document.removeEventListener('visibilitychange', wake); window.removeEventListener('online', wake); };
  }, [gamePk, delay, replayStart, speed, lag, t0, paused]);
  const isReplay = replayStart !== null;
  useEffect(() => {  // once per game: scene list of the replay, and the catcher setup (after-the-game data, never live)
    if (!gamePk || !isReplay) return;
    let stop = false;
    get<Outline>(`/replay/${gamePk}/outline`).then(o => { if (!stop) setOutline(o); }).catch(() => { if (!stop) setOutline({ pitches: 0, halves: [] }); });
    get<SetupView>(`/watch/${gamePk}/setup`).then(v => {
      if (stop) return;
      const x = new Map<string, number>();
      v.pitches.forEach(p => { if (p.status !== 'unavailable' && typeof p.plate_x_feet === 'number' && Number.isFinite(p.plate_x_feet)) x.set(`${gamePk}:${p.at_bat_number}:${p.pitch_number}`, p.plate_x_feet); });
      setSetups({ badge: v.badge ?? null, x });
    }).catch(() => undefined);  // most games have none (404): nothing is drawn and nothing is said
    return () => { stop = true; };
  }, [gamePk, isReplay]);
  useEffect(() => {
    if (!hold) return;
    const timer = window.setTimeout(() => setHold(null), seen.current.get(hold)?.actual.event_label ? HOLD_MS.event : HOLD_MS.pitch);
    return () => window.clearTimeout(timer);
  }, [hold]);

  const pinT0 = useCallback((next: number) => {
    const url = new URLSearchParams(window.location.search); url.set('t0', String(next));
    window.history.replaceState(null, '', `?${url}`); setT0(next);
  }, []);
  const shift = useCallback((seconds: number) => {
    if (replayStart !== null) {
      if (t0 !== null) pinT0(Math.min(Date.now() / 1000, t0 + seconds / speed));
    } else {
      const next = Math.max(0, Math.min(state?.sync?.max_delay_s ?? 120, (delay ?? state?.delay_s ?? 30) + seconds));
      store(DELAY_KEY, String(next)); setDelay(next);
    }
  }, [replayStart, t0, speed, delay, state, pinT0]);
  /** The replay clock is wall time since t0, so a pause is honest only by moving t0 forward by its length on resume. */
  const togglePause = useCallback(() => {
    if (paused === null) { if (t0 !== null) setPaused(Date.now() / 1000); return; }
    if (t0 !== null) pinT0(Math.min(Date.now() / 1000, t0 + (Date.now() / 1000 - paused)));
    setPaused(null);
  }, [paused, t0, pinT0]);
  /** Scene jump: a new replay link from the first pitch of a half-inning (the server starts 15 s before it). */
  const jump = useCallback((first: number) => {
    const url = new URLSearchParams(window.location.search); url.set('replay', String(first)); url.delete('t0');
    window.history.replaceState(null, '', `?${url}`);
    seen.current.clear(); shown.current.clear(); sides.current.clear(); lastPrev.current = null; wasReady.current = false; board.current = null;
    setState(null); setError(null); setFailures(0); setHold(null); setPast(null); setPaused(null); setLastGood(null); setSheet(null); setT0(null); setReplayStart(first);
  }, []);
  const tap = useCallback(async () => {
    const current = delay ?? state?.delay_s ?? 30;
    try {
      const r = await get<{ status: string; delay_s: number; reason?: string }>(`/live/${gamePk}/tap?delay=${current}`);
      if (r.status !== 'ok') { setTapNote(r.reason ?? '다시 눌러 주세요.'); return; }
      store(DELAY_KEY, String(r.delay_s)); setDelay(r.delay_s); setTapNote(`맞췄어요: 내 화면이 ${r.delay_s.toFixed(1)}초 늦어요.`);
    } catch (e) { setTapNote((e as Error).message); }
  }, [gamePk, delay, state]);
  const flipUnit = useCallback(() => setUnit(u => { const next = u === 'kmh' ? 'mph' : 'kmh'; store(UNIT_KEY, next); return next; }), []);
  const close = useCallback(() => setSheet(null), []);

  const rec = state?.recommendation ?? null, b = ZONE_BOUNDS;
  // Last good state stays on screen but is never shown as current (polling failed, or the server lost the MLB feed).
  const stale = state !== null && paused === null && (error !== null || state.sync?.feed_ok === false);
  const replay = state?.replay ?? null;
  const currentHalf = outline && replay ? outline.halves.reduce((found, h, i) => h.first <= replay.index ? i : found, 0) : -1;
  const list = [...seen.current.values()];
  const dotsOf = (pa: string): Dot[] => list.filter(p => p.pa === pa && p.actual.x !== null && p.actual.z !== null).map(p => ({ key: p.key, n: p.n, x: p.actual.x!, z: p.actual.z!, kind: kind(p.actual) }));
  const speedButton = (mph: number | null) => mph === null ? null : <button className="gm-speed" onClick={flipUnit} aria-label="속도 단위 바꾸기">{speedText(mph, unit)}</button>;
  const side = state?.batter?.side ?? 'R';
  const top = rec?.status === 'ready' ? rec.recommendation?.candidates[0] ?? null : null;
  const held = hold ? seen.current.get(hold) ?? null : null;
  const looking = past !== null ? list[Math.min(past, list.length - 1)] ?? null : null;
  const currentPa = rec?.key ? paOf(rec.key) : null;
  const stripPa = looking?.pa ?? held?.pa ?? currentPa;
  const strip = stripPa ? list.filter(p => p.pa === stripPa) : [];
  const nextNumber = !looking && !held && rec?.key && state?.status === 'ready' ? Number(rec.key.split(':')[2]) || strip.length + 1 : null;

  /** A pitch that has been thrown: the result first, then how it compared with what was recommended before it. */
  const thrown = (p: Seen, label: ReactNode, event: boolean) => {
    const match = matches(p.top, p.actual, b), want = p.top ? cellOf(p.top.zone_id) : null;
    const setupX = setups?.x.get(p.key) ?? null;  // exact game:at_bat:pitch, only once the pitch has been thrown
    const k = kind(p.actual), we = event && rec?.previous?.key === p.key && rec.previous.home_we_before !== null && rec.home_we_now !== null && rec.home_we_now !== undefined && state?.game
      ? { home: rec.previous.half === 'Top', before: rec.previous.home_we_before!, now: rec.home_we_now } : null;
    const share = (v: number) => Math.round((we!.home ? v : 1 - v) * 100);  // the fielding team's side (D51)
    const delta = we ? (we.home ? 1 : -1) * (we.now - we.before) * 100 : 0;
    return <>
      <div className="gm-toprow"><span className={`gm-chip ${past !== null ? 'plain' : 'actual'}`}>{label}</span>
        {past !== null && p.top && want && <span className="gm-tag">추천은 {p.top.pitch_label} · {words(want, p.side)}</span>}</div>
      {event ? <div className="gm-event"><div><h1 className="gm-hero big">{p.actual.event_label}</h1><p className="gm-sub">{p.actual.pitch_label} {speedButton(p.actual.speed_mph)}</p><div className="gm-row"><Marks match={match} /></div></div>
          <Zone side={p.side} target={want} dots={dotsOf(p.pa)} big={p.key} setupX={setupX} mini /></div>
        : <><h1 className="gm-hero">{p.actual.pitch_label} {speedButton(p.actual.speed_mph)}</h1>
          <div className="gm-row"><span className={`gm-result ${k}`}>{p.actual.event_label ?? p.actual.result_label}</span><Marks match={match} /></div>
          <Zone side={p.side} target={want} dots={dotsOf(p.pa)} big={p.key} setupX={setupX} /></>}
      {event && we && <div className="gm-we"><div><span>{short(we.home ? state!.game!.home_team : state!.game!.away_team)} 승리확률 {share(we.now)}%</span>
          <strong>{delta >= 0 ? '+' : '−'}{Math.abs(delta).toFixed(1)}%p</strong></div>
        <div className="gm-we-bar"><i style={{ width: `${share(we.now)}%` }} /><b style={{ left: `${share(we.before)}%` }} /></div></div>}
      {event && <div className="gm-why"><p><b>무엇 때문에</b><span>미측정</span></p>{['선택', '실행', '타자'].map(name => <div key={name}><span>{name}</span><i /></div>)}</div>}
    </>;
  };

  let stage: ReactNode;
  if (looking) stage = thrown(looking, `${looking.n}구`, false);
  else if (past !== null) stage = <><div className="gm-toprow"><span className="gm-chip plain">지난 공</span></div><h1 className="gm-hero mid">아직 지난 공이 없어요</h1></>;
  else if (!state) stage = <><div className="gm-toprow"><span className="gm-chip plain">{error ? '연결 안 됨' : '불러오는 중'}</span></div>
    {error && <><p className="gm-sub">{error}</p><p className="gm-note"><span role="status">자동으로 계속 다시 연결하고 있어요</span>{failures > 1 ? ` · 실패 ${failures}번` : ''}</p>
      <button className="gm-wide" onClick={() => pollNow.current()}>지금 다시 시도</button></>}</>;
  else if (state.status !== 'ready') {
    const [chip, text] = WAIT[state.status] ?? ['잠시만요', '잠시 기다려 주세요'], p = state.last_pitch;
    stage = <><div className="gm-toprow"><span className="gm-chip plain">{chip}</span></div><h1 className="gm-hero mid">{isReplay && state.status === 'not_live' ? '경기 끝' : text}</h1>
      {state.status !== 'between_innings' && state.status !== 'resolving' && state.reason && <p className="gm-sub">{state.reason}</p>}
      {p && <div className="gm-last"><span>방금 던진 공</span><b>{p.pitch_label} {speedButton(p.speed_mph)}</b>{p.result_label && <em>{p.result_label}</em>}</div>}</>;
  } else if (held) stage = thrown(held, stale ? '마지막 상황' : held.actual.event_label ? '타석 끝' : '방금 던진 공', Boolean(held.actual.event_label));
  else if (top) {
    const cell = cellOf(top.zone_id), all = rec!.recommendation!.candidates;
    const usual = top.detail.reference_probability >= Math.max(...all.map(c => c.detail.reference_probability));
    stage = <><div className="gm-toprow"><span className={`gm-chip ${stale ? 'plain' : ''}`}>{stale ? '마지막 상황' : '다음 공'}</span><span className="gm-tag">{usual ? `평소에도 1순위 · ${pct(top.detail.reference_probability)}` : `평소 ${pct(top.detail.reference_probability)} · 모델 ${pct(top.detail.probability)}`}</span></div>
      <h1 className="gm-hero"><button className="gm-name" onClick={() => setSheet({ glossary: top.pitch_type })} aria-label={`${top.pitch_label} 쉬운 풀이`}>{top.pitch_label}</button></h1>
      {cell && <Where cell={cell} side={side} />}
      <Zone side={side} target={cell} dots={currentPa ? dotsOf(currentPa) : []} /></>;
  } else {
    const computing = rec?.status === 'computing';
    stage = <><div className="gm-toprow"><span className="gm-chip plain">{stale ? '마지막 상황' : '다음 공'}</span></div><h1 className="gm-hero mid">{computing ? '추천 계산 중' : '추천 없음'}</h1>
      <p className="gm-sub">{computing ? rec?.reason : rec?.status === 'unsupported' ? rec.reason ?? '기록이 부족한 투수' : rec?.reason ?? '이 상황은 추천하지 않아요'}</p>
      {/* no recommendation is about this pitcher, not the feed: keep the last received pitch on stage so the game stays followable */}
      {rec?.status === 'unsupported' && rec.previous && <div className="gm-last"><span>{stale ? '마지막으로 받은 공' : '방금 던진 공 · 경기 기록은 계속 받는 중'}</span>
        <b>{rec.previous.actual.pitch_label} {speedButton(rec.previous.actual.speed_mph)}</b><em>{rec.previous.actual.event_label ?? rec.previous.actual.result_label}</em></div>}
      <Zone side={side} target={null} dots={currentPa ? dotsOf(currentPa) : []} empty /></>;
  }

  if (!gamePk) return <div className="wa-page gm-page"><header className="gm-top"><a className="gm-icon" href="/watch" aria-label="경기 목록으로">←</a></header><p className="gm-sub gm-pad">경기 번호가 없어요.</p></div>;
  const shownBoard = board.current, idle = state?.status !== 'ready' && state?.status !== 'resolving';
  const mix = state?.pitcher_today?.mix ?? [], glossary = sheet && typeof sheet === 'object' ? sheet.glossary : null;
  const usedDelay = Math.round(delay ?? state?.delay_s ?? 30);
  const candidates = rec?.status === 'ready' ? rec.recommendation?.candidates ?? [] : [];
  const replayClock = replay ? ` ${clock(replay.elapsed_s)}` : '';
  const offline = stale || (state === null && error !== null);
  const waiting = state === null && !offline;  // never say LIVE before the first answer has arrived
  const liveLabel = stale ? (error !== null ? '연결 끊김' : '기록 멈춤') : offline ? '연결 안 됨' : waiting ? '연결 중' : paused !== null ? `일시정지${replayClock}` : isReplay ? `다시 보기${replayClock}` : 'LIVE';
  return <div className={`wa-page gm-page ${stale ? 'stale' : ''}`}>
    <header className="gm-top"><a className="gm-icon" href="/watch" aria-label="경기 목록으로"><svg viewBox="0 0 24 24"><path d="M19 12H5" /><path d="M11 6l-6 6 6 6" /></svg></a>
      {past !== null ? <span className="gm-live off">지난 공</span>
        : <button className={`gm-live ${offline ? 'stale' : isReplay || waiting ? 'off' : ''}`} onClick={() => setSheet('sync')} aria-label={isReplay ? '다시 보기 조작' : '중계와 맞추기'}>{liveLabel}</button>}
      {isReplay && past === null && <button className="gm-icon" onClick={togglePause} disabled={t0 === null} aria-label={paused === null ? '일시정지' : '이어서 보기'}>
        <svg viewBox="0 0 24 24" className="fill">{paused === null ? <><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></> : <path d="M7 5l12 7-12 7z" />}</svg></button>}
      <span className="gm-grow" /><span className="gm-badge">실험판</span>
      <button className="gm-icon" onClick={() => setSheet('more')} aria-label="더 보기"><svg viewBox="0 0 24 24" className="fill"><circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" /></svg></button></header>
    {shownBoard ? <Board s={shownBoard.s} away={shownBoard.away} home={shownBoard.home} idle={idle}
      pitcher={state?.pitcher?.name ?? null} pitcherNote={state?.pitcher_today ? `${state.pitcher_today.pitches}구` : ''}
      batter={state?.batter?.name ?? null} batterNote={state?.batter ? (state.batter.side === 'L' ? '좌타' : '우타') : ''} /> : <section className="gm-board" />}
    <main className={`gm-stage ${stale ? 'stale' : ''}`}>
      {stale && <p className="gm-stale" role="status" title={error ?? undefined}><b>{error !== null ? '연결 끊김' : '경기 기록 수신 끊김'}</b>
        <span>{lastGood !== null ? `${timeText(lastGood)}에 받은 상황 · ` : ''}다시 연결 중</span>
        {error !== null && <button onClick={() => pollNow.current()}>다시 시도</button>}</p>}
      {paused !== null && <p className="gm-paused" role="status"><b>일시정지</b><span>▶을 누르면 멈춘 곳부터 이어져요</span></p>}
      {stage}</main>
    <nav className={`gm-dots ${strip.length > 4 ? 'tight' : ''}`} aria-label="이 타석 투구 순서">
      {strip.map(p => <button key={p.key} className={looking?.key === p.key || held?.key === p.key ? 'current' : ''} onClick={() => setPast(list.findIndex(q => q.key === p.key))} aria-label={`${p.n}구 보기`}>
        <i className={kind(p.actual)}>{p.n}</i><span>{p.actual.pitch_label}</span></button>)}
      {nextNumber !== null && <span className="next"><i>{nextNumber}</i></span>}</nav>
    {past !== null ? <div className="gm-bar"><button className="narrow" disabled={!looking || past <= 0} onClick={() => setPast(Math.max(0, Math.min(past, list.length - 1) - 1))}>이전</button>
        <button className="primary" onClick={() => setPast(null)}>LIVE로 돌아가기</button>
        <button className="narrow" disabled={!looking || past >= list.length - 1} onClick={() => setPast(Math.min(list.length - 1, past + 1))}>다음</button></div>
      : <div className="gm-bar"><button onClick={() => { setTab(0); setSheet('detail'); }}>자세히</button><button onClick={() => setSheet('box')}>박스 스코어</button>
        <button onClick={() => setPast(Math.max(0, list.length - 1))}>지난 장면</button></div>}

    {sheet === 'detail' && <Sheet title="자세히" onClose={close}>
      <div className="gm-tabs">{['후보', '투수 오늘', '안내'].map((name, i) => <button key={name} className={tab === i ? 'on' : ''} onClick={() => setTab(i)}>{name}</button>)}</div>
      {tab === 0 && (candidates.length ? <><table className="gm-table"><thead><tr><th colSpan={2}>구종 · 위치</th><th>모델 선택</th><th>평소</th></tr></thead>
          <tbody>{candidates.map(c => { const cell = cellOf(c.zone_id); return <tr key={c.rank}><td><i className={c.rank === 1 ? 'on' : ''}>{c.rank}</i></td>
            <td><b>{c.pitch_label}</b>{cell && <span> · {words(cell, side)}</span>}</td><td><b>{pct(c.detail.probability)}</b></td><td>{pct(c.detail.reference_probability)}</td></tr>; })}</tbody></table>
          <p className="gm-note">두 숫자가 비슷하면 “평소대로”에 가까운 추천이에요.</p></> : <p className="gm-note">지금은 추천이 없어요.</p>)}
      {tab === 1 && (mix.length ? <table className="gm-table"><thead><tr><th colSpan={2}>{state?.pitcher?.name} · {state?.pitcher_today?.pitches}구</th><th>비율</th><th>평균</th></tr></thead>
          <tbody>{mix.map(m => <tr key={m.pitch_type}><td /><td><b>{m.pitch_label}</b><span> · {m.count}구</span></td><td><b>{pct(m.share)}</b></td><td>{speedText(m.avg_mph, unit)}</td></tr>)}</tbody></table>
        : <p className="gm-note">이 투수는 아직 던진 공이 없어요.</p>)}
      {tab === 2 && <><p className="gm-note">검증 전 실험 버전이에요. 추천은 2025년까지 기록으로 학습해 고정한 모델이에요. 2026 정규시즌 1회 평가: 개선 근거를 확인하지 못했습니다.{rec?.policy_identity ? ` 식별자 ${rec.policy_identity.slice(0, 8)}` : ''}</p>
        <p className="gm-note">위치는 이 투수의 2025년까지 실제 투구 분포로 어림한 값이며 평가되지 않았어요. 존 그림은 포수 쪽에서 본 모습이에요.</p>
        <p className="gm-note">“구종 ✓ / 위치 ✕”는 실제 공이 추천과 같았는지만 보여 줘요. 추천이 좋았는지와는 다른 이야기예요.</p>
        {setups && setups.x.size > 0 && <p className="gm-note">{SETUP_NOTE} 공이 들어온 뒤 존 그림의 세로선과 위·아래 삼각형(미트)이 포수가 미트를 댄 가로 위치예요. 높이는 표시하지 않아요.{setups.badge ? ` (${setups.badge})` : ''}</p>}</>}</Sheet>}
    {glossary && <Sheet title={mix.find(m => m.pitch_type === glossary)?.pitch_label ?? top?.pitch_label ?? glossary} onClose={close}>
      <p className="gm-plain">{GLOSSARY[glossary] ?? '이 구종의 풀이는 준비 중이에요.'}</p>
      {mix.filter(m => m.pitch_type === glossary).map(m => <div key={m.pitch_type} className="gm-card"><span>{state?.pitcher?.name} 오늘</span><b>{m.count}구 · 평균 {speedText(m.avg_mph, unit)}</b></div>)}
      <ul className="gm-list">{mix.filter(m => m.pitch_type !== glossary).map(m => <li key={m.pitch_type}><b>{m.pitch_label}</b><span>{GLOSSARY[m.pitch_type] ?? ''}{m.avg_mph !== null ? ` · 오늘 평균 ${speedText(m.avg_mph, unit)}` : ''}</span></li>)}</ul></Sheet>}
    {sheet === 'box' && <Sheet title="박스 스코어" onClose={close}>
      {shownBoard && <div className="gm-card"><span>{shownBoard.s.inning}회 {shownBoard.s.half === 'Top' ? '초' : '말'}</span><b>{short(shownBoard.away)} {shownBoard.s.away_score} : {shownBoard.s.home_score} {short(shownBoard.home)}</b></div>}
      {state?.pitcher_today && <div className="gm-card"><span>지금 투수</span><b>{state.pitcher?.name} · {state.pitcher_today.pitches}구</b></div>}
      <p className="gm-note">이닝별 점수와 타자 기록은 다음 단계에서 연결돼요.</p></Sheet>}
    {sheet === 'sync' && <Sheet title={isReplay ? '영상과 맞추기' : '중계와 맞추기'} onClose={close}>
      {stale && <p className="gm-note gm-error" role="status">{error ?? '서버가 경기 기록을 받지 못하고 있어요.'} 마지막으로 받은 상황을 보여 주고 있어요.</p>}
      <div className="gm-card"><span>{isReplay ? (paused !== null ? '경기 시각 · 일시정지' : '경기 시각') : '내 화면이'}</span><b aria-live="polite">{isReplay ? (replay ? clock(replay.elapsed_s) : '—') : `${usedDelay}초 늦음`}</b></div>
      {isReplay && <button className="gm-wide" onClick={togglePause} disabled={t0 === null}>{paused === null ? '일시정지' : '이어서 보기'}</button>}
      <div className="gm-shift">{[-5, -1, 1, 5].map(d => <button key={d} onClick={() => shift(d)} aria-label={`${Math.abs(d)}초 ${d < 0 ? '빠르게' : '늦게'}`}>{d > 0 ? `+${d}` : d}</button>)}</div>
      <p className="gm-note">공 결과가 내 화면보다 먼저 뜨면 +, 늦게 뜨면 − 를 눌러요.{!isReplay && ' 이 기기에 기억해 둬요.'}</p>
      {!isReplay && state?.status === 'ready' && <button className="gm-wide" onClick={() => void tap()}>공이 미트에 들어오는 순간 누르기</button>}
      {tapNote && <p className="gm-note" aria-live="polite">{tapNote}</p>}
      {isReplay && <><h3 className="gm-subhead">장면으로 이동</h3>
        <p className="gm-note">회를 누르면 그 회 첫 공 15초 전부터 다시 시작해요.</p>
        <div className="gm-scenes" role="group" aria-label="장면으로 이동">
          <button onClick={() => jump(0)} className="wide">처음부터</button>
          {outline?.halves.map((h, i) => <button key={h.first} className={i === currentHalf ? 'current' : ''} aria-current={i === currentHalf ? 'true' : undefined} onClick={() => jump(h.first)}>{h.label}</button>)}</div>
        {outline && !outline.halves.length && <p className="gm-note">회 목록을 불러오지 못했어요. “처음부터”는 쓸 수 있어요.</p>}</>}</Sheet>}
    {sheet === 'more' && <Sheet title="더 보기" onClose={close}>
      <button className="gm-wide" onClick={flipUnit}>속도 단위: {unit === 'kmh' ? 'km/h' : 'mph'} (눌러서 바꾸기)</button>
      <a className="gm-wide" href="/watch">경기 목록</a>
      <a className="gm-wide" href={`/live?${params}`}>예전 화면으로 보기</a></Sheet>}
  </div>;
}

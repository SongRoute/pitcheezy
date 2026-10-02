import { useCallback, useEffect, useRef, useState } from 'react';
import { BADGE, Board, Header, PitchStrip, Recommendation, WeCard, ZONE_BOUNDS, get, plainZone, short } from './WatchAlong';
import type { Actual, Candidate, Situation, StripItem } from './WatchAlong';
import './watch-along.css';

type Pre = { status: 'ready' | 'unsupported'; reason: string | null; recommendation: { candidates: Candidate[] } | null };
type PaPitch = { key: string; pitch_number: number; status: string; pre: Pre; actual: Actual | null };
type LiveRec = { status: 'ready' | 'unsupported' | 'computing' | 'unavailable'; reason: string | null; key?: string;
  recommendation?: { candidates: Candidate[] } | null; pa_pitches?: PaPitch[];
  previous?: { key: string; actual: Actual; inning: number; half: 'Top' | 'Bot'; home_we_before: number | null } | null;
  home_we_now?: number | null; policy_identity?: string; tally?: { pitches: number; recommended: number; same_type: number } };
type Replay = { start: number; t0: number; speed: number; elapsed_s: number; pitches: number; index: number };
type LiveState = { status: string; reason: string | null; badge?: string; delay_s: number; replay?: Replay | null;
  sync?: { max_delay_s: number; event_age_s?: number; feed_lag_s?: number };
  game?: { game_pk: number; date: string; status: string; away_team: string; home_team: string };
  situation?: (Situation & { runners: Record<string, boolean> }) | null; pitcher?: { id: number; name: string | null; hand: string } | null;
  batter?: { id: number; name: string | null; side: string } | null;
  pitcher_today?: { pitches: number; mix: { pitch_type: string; pitch_label: string; count: number; share: number; avg_mph: number | null }[] } | null;
  last_pitch?: { key: string; pitch_type: string | null; pitch_label: string; speed_mph: number | null; result_label: string | null; play_text: string | null } | null;
  pitch_sequence?: { pitch_number: number; pitch_label: string | null; description: string | null }[];
  recommendation: LiveRec | null };
type Outline = { pitches: number; halves: { label: string; first: number }[] };

const POLL_MS = 2000;
const DELAY_KEY = 'pitcheezy.live.delay';
const WAIT_TEXT: Record<string, string> = { resolving: '방금 공의 결과를 기다려요.', buffering: '내 중계 시점에 맞추는 중이에요.', between_innings: '이닝 교대 중이에요. 다음 이닝 첫 공을 기다려요.',
  not_live: '지금은 경기 중이 아니에요.', unsupported_game_type: '이 경기 종류는 지원하지 않아요.' };
const clock = (seconds: number) => { const s = Math.max(0, Math.round(seconds)); return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };
const savedDelay = () => { try { const v = localStorage.getItem(DELAY_KEY); return v === null ? null : Number(v); } catch { return null; } };

/** Lines the screen up with the viewer's broadcast: + shows pitches later, − sooner. One line once it is set. */
function SyncBar({ label, value, hint, onShift, onTap, tapNote, open }: { label: string; value: string; hint: string; onShift: (seconds: number) => void; onTap?: () => void; tapNote?: string | null; open: boolean }) {
  return <details className="wa-sync" open={open || undefined}>
    <summary><span className="wa-sync-label">{label}</span><strong aria-live="polite">{value}</strong><em>조절</em></summary>
    <div className="wa-sync-buttons">{[-5, -1, 1, 5].map(d => <button key={d} onClick={() => onShift(d)} aria-label={`${Math.abs(d)}초 ${d < 0 ? '빠르게' : '늦게'}`}>{d > 0 ? `+${d}` : d}</button>)}</div>
    <p>{hint}</p>
    {onTap && <><button className="wa-sync-tap" onClick={onTap}>내 화면에서 공이 미트에 들어오는 순간 누르기</button>{tapNote && <p aria-live="polite">{tapNote}</p>}</>}</details>;
}

export default function Live() {
  const params = new URLSearchParams(window.location.search);
  const gamePk = Number(params.get('game'));
  const replayStart = params.get('replay') === null ? null : Number(params.get('replay'));
  const speed = Number(params.get('speed')) || 1;
  const [t0, setT0] = useState<number | null>(params.get('t0') ? Number(params.get('t0')) : null);
  const [delay, setDelay] = useState<number | null>(replayStart === null ? savedDelay() : 0);
  const [state, setState] = useState<LiveState | null>(null);
  const [outline, setOutline] = useState<Outline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updated, setUpdated] = useState<Date | null>(null);
  const [tapNote, setTapNote] = useState<string | null>(null);
  const [syncOpen] = useState(() => replayStart === null ? savedDelay() === null : params.get('t0') === null);  // open until first set
  const shown = useRef(new Map<string, Candidate | null>());  // recommendation shown before each pitch (this session)

  useEffect(() => {
    if (!gamePk) return;
    let stop = false, timer = 0;
    const query = new URLSearchParams();
    if (delay !== null) query.set('delay', String(delay));
    if (replayStart !== null) { query.set('replay', String(replayStart)); query.set('speed', String(speed)); if (t0 !== null) query.set('t0', String(t0)); }
    const tick = async () => {
      try {
        const next = await get<LiveState>(`/live/${gamePk}/state?${query}`);
        if (stop) return;
        if (next.replay && t0 === null) {  // pin the replay clock into the link: reloads and a second device stay on it
          const url = new URLSearchParams(window.location.search); url.set('t0', String(next.replay.t0));
          window.history.replaceState(null, '', `?${url}`); setT0(next.replay.t0); return;
        }
        const rec = next.recommendation;
        if (rec?.key && (rec.status === 'ready' || rec.status === 'unsupported')) shown.current.set(rec.key, rec.recommendation?.candidates[0] ?? null);
        rec?.pa_pitches?.forEach(p => { if (!shown.current.has(p.key)) shown.current.set(p.key, p.pre.recommendation?.candidates[0] ?? null); });
        setState(next); setError(null); setUpdated(new Date());
      } catch (e) { if (!stop) setError((e as Error).message); }
      if (!stop) timer = window.setTimeout(tick, document.hidden ? POLL_MS * 5 : POLL_MS);
    };
    void tick();
    return () => { stop = true; window.clearTimeout(timer); };
  }, [gamePk, delay, replayStart, speed, t0]);
  useEffect(() => {
    if (gamePk && replayStart !== null) get<Outline>(`/replay/${gamePk}/outline`).then(setOutline).catch(() => undefined);
  }, [gamePk, replayStart]);

  const shift = useCallback((seconds: number) => {
    if (replayStart !== null) {  // a replay moves its own clock, so the video may be ahead of or behind this screen
      if (t0 === null) return;
      const next = Math.min(Date.now() / 1000, t0 + seconds / speed);
      const url = new URLSearchParams(window.location.search); url.set('t0', String(next));
      window.history.replaceState(null, '', `?${url}`); setT0(next);
    } else {
      const next = Math.max(0, Math.min(state?.sync?.max_delay_s ?? 120, (delay ?? state?.delay_s ?? 30) + seconds));
      try { localStorage.setItem(DELAY_KEY, String(next)); } catch { /* storage may be unavailable */ }
      setDelay(next);
    }
  }, [replayStart, t0, speed, delay, state]);

  const tap = useCallback(async () => {  // fine sync: the server measures this viewer's delay from the pitch they just saw
    const current = delay ?? state?.delay_s ?? 30;
    try {
      const r = await get<{ status: string; delay_s: number; reason?: string }>(`/live/${gamePk}/tap?delay=${current}`);
      if (r.status !== 'ok') { setTapNote(r.reason ?? '다시 눌러 주세요.'); return; }
      try { localStorage.setItem(DELAY_KEY, String(r.delay_s)); } catch { /* storage may be unavailable */ }
      setDelay(r.delay_s); setTapNote(`맞췄어요: 내 화면이 ${r.delay_s.toFixed(1)}초 늦어요.`);
    } catch (e) { setTapNote((e as Error).message); }
  }, [gamePk, delay, state]);

  if (!gamePk) return <div className="wa-page"><Header badge={BADGE} back /><main className="wa-main"><p className="wa-error">경기 번호가 없어요.</p><a className="wa-button" href="/watch">경기 목록으로</a></main></div>;
  const rec = state?.recommendation ?? null;
  const s = state?.situation ?? null, game = state?.game;
  const replay = state?.replay ?? null, isReplay = replayStart !== null;
  const usedDelay = Math.round(delay ?? state?.delay_s ?? 30);
  const previous = rec?.previous ?? null;
  const prevTop = previous ? shown.current.get(previous.key) ?? null : null;
  const strip: StripItem[] = [...(rec?.pa_pitches ?? []).map(p => ({ key: p.key, label: p.actual?.pitch_label ?? '—', result: p.actual?.result_label ?? '',
    match: p.actual?.pitch_type && p.pre.recommendation ? p.pre.recommendation.candidates[0].pitch_type === p.actual.pitch_type : null })),
    ...(rec?.key ? [{ key: rec.key, label: '?', result: '다음 공', match: null, current: true }] : [])];
  const prevBattingHome = previous?.half === 'Bot';
  const weBefore = previous?.home_we_before ?? null, weNow = rec?.home_we_now ?? null;
  const currentHalf = outline && replay ? outline.halves.reduce((found, h, i) => h.first <= replay.index ? i : found, 0) : -1;
  const late = !isReplay && state?.sync?.feed_lag_s !== undefined && state.sync.event_age_s !== undefined && state.sync.event_age_s < 90 && state.sync.feed_lag_s > usedDelay + 2;
  return <div className="wa-page"><Header badge={state?.badge ?? BADGE} back />
    {s && game && state?.pitcher && state.batter ? <Board s={s} away={game.away_team} home={game.home_team} pitcher={state.pitcher.name ?? String(state.pitcher.id)}
      batter={`${state.batter.name ?? state.batter.id} (${state.batter.side === 'L' ? '좌' : '우'}타)`} extra={replay ? `경기 ${clock(replay.elapsed_s)}` : `${usedDelay}초 늦게`} /> : null}
    <main className="wa-main">
      {game && <p className="wa-live-title"><span className="wa-live-dot">{isReplay ? '다시 보기 · 실제 속도' : '지연 중계'}</span> {short(game.away_team)} @ {short(game.home_team)}{isReplay && speed !== 1 ? ` · ${speed}배속` : ''}</p>}
      {isReplay && outline && <nav className="wa-innings" aria-label="회로 이동">{outline.halves.map((h, i) =>
        <a key={h.first} className={i === currentHalf ? 'current' : ''} aria-current={i === currentHalf ? 'true' : undefined} href={`/live?game=${gamePk}&replay=${h.first}${speed !== 1 ? `&speed=${speed}` : ''}`}>{h.label}</a>)}</nav>}
      {isReplay ? <SyncBar open={syncOpen} label="영상 맞추기" value={replay ? `경기 ${clock(replay.elapsed_s)}` : '—'} onShift={shift}
          hint="영상보다 이 화면이 먼저 가면 +, 늦으면 − 를 눌러 맞춰요. 회 버튼을 누르면 그 회 첫 공 15초 전부터 시작해요." />
        : <SyncBar open={syncOpen} label="중계 맞추기" value={`내 화면이 ${usedDelay}초 늦음`} onShift={shift} onTap={state?.status === 'ready' ? () => void tap() : undefined} tapNote={tapNote}
          hint="공 결과가 내 중계 화면보다 먼저 뜨면 +, 늦게 뜨면 − 를 눌러요. 이 기기에 기억해 둬요." />}
      {late && <p className="wa-sync-late">기록이 내 중계보다 늦게 도착하고 있어요(약 {Math.round(state!.sync!.feed_lag_s!)}초). + 를 눌러 여유를 두면 추천을 공보다 먼저 볼 수 있어요.</p>}
      {error && <p className="wa-error">{error} {state ? '마지막으로 받은 상황을 보여 주고 있어요. 계속 다시 시도해요.' : ''}</p>}
      {!state && !error && <p className="wa-loading">경기 상황을 불러오는 중…</p>}
      {state && state.status !== 'ready' && <article className="wa-card"><h2 className={`wa-none ${state.status === 'buffering' ? 'wa-pulse' : ''}`}>{isReplay && state.status === 'not_live' ? '경기가 끝났어요.' : WAIT_TEXT[state.status] ?? '잠시 기다려 주세요.'}</h2>
        <p className="wa-reason">{state.reason}</p>{state.status === 'not_live' && !isReplay && <p className="wa-reason">끝난 경기는 정리되면 <a href="/watch">경기 목록</a>의 “중계 영상과 함께 보기”에 올라와요.</p>}</article>}
      {state && state.status !== 'ready' && state.last_pitch && (() => { const p = state.last_pitch, top = shown.current.get(p.key) ?? null;
        return <article className="wa-card wa-prev"><p className="wa-eyebrow">방금 던진 공</p>
          <h3>{p.pitch_label}{p.speed_mph !== null && <span> · {p.speed_mph.toFixed(1)} mph</span>}</h3>
          {p.result_label && <p className="wa-result">{p.result_label}</p>}
          {top && p.pitch_type && <p className={`wa-match ${top.pitch_type === p.pitch_type ? 'same' : ''}`}>{top.pitch_type === p.pitch_type ? '직전 추천 1순위 구종과 같았어요' : `직전 추천 1순위는 ${top.pitch_label}(${plainZone(top.zone_label)})`}</p>}
          {p.play_text && <p className="wa-play">{p.play_text}</p>}</article>; })()}
      {state?.status === 'ready' && <>
        <PitchStrip items={strip} />
        {rec?.tally && rec.tally.recommended > 0 && <p className="wa-tally">지금까지 추천한 공 <b>{rec.tally.recommended}구</b> 중 실제로 같은 구종 <b>{rec.tally.same_type}구</b>
          <span> · 투수의 선택과 얼마나 겹쳤는지일 뿐, 추천의 성적은 아니에요</span></p>}
        <article className="wa-card">
          <p className="wa-eyebrow">다음 공 · 투구 전 추천</p>
          {rec?.status === 'ready' && rec.recommendation ? <Recommendation candidates={rec.recommendation.candidates} bounds={ZONE_BOUNDS} actual={null} note="위치는 이 투수의 2025년까지 실제 투구 분포 기반 근사이며 평가되지 않았습니다." />
            : rec?.status === 'computing' ? <><h2 className="wa-none wa-pulse">추천 계산 중…</h2><p className="wa-reason">{rec.reason}</p></>
            : <><h2 className="wa-none">추천 없음</h2><p className="wa-reason">{rec?.reason ?? '이 상황은 추천하지 않아요.'}</p></>}
        </article>
        {state.pitcher_today && state.pitcher_today.pitches > 0 && <article className="wa-card wa-today">
          <p className="wa-eyebrow">이 투수 오늘 · {state.pitcher_today.pitches}구 (경기 기록)</p>
          <ul>{state.pitcher_today.mix.map(m => <li key={m.pitch_type}><b>{m.pitch_label}</b><span className="bar"><i style={{ width: `${Math.round(m.share * 100)}%` }} /></span>
            <em>{Math.round(m.share * 100)}%</em><small>{m.avg_mph !== null ? `${m.avg_mph.toFixed(1)} mph` : ''}</small></li>)}</ul>
        </article>}
        {previous && <article className="wa-card wa-prev">
          <p className="wa-eyebrow">방금 던진 공</p>
          <h3>{previous.actual.pitch_label}{previous.actual.speed_mph !== null && <span> · {previous.actual.speed_mph.toFixed(1)} mph</span>}</h3>
          <p className="wa-result">{previous.actual.event_label ?? previous.actual.result_label} · {plainZone(previous.actual.zone_label)}</p>
          {prevTop && previous.actual.pitch_type && <p className={`wa-match ${prevTop.pitch_type === previous.actual.pitch_type ? 'same' : ''}`}>{prevTop.pitch_type === previous.actual.pitch_type ? '직전 추천 1순위 구종과 같았어요' : `직전 추천 1순위는 ${prevTop.pitch_label}(${plainZone(prevTop.zone_label)})`}</p>}
          {previous.actual.play_text && <p className="wa-play">{previous.actual.play_text}</p>}
          {game && <WeCard team={prevBattingHome ? game.home_team : game.away_team}
            before={weBefore === null ? null : prevBattingHome ? weBefore : 1 - weBefore} after={weNow === null ? null : prevBattingHome ? weNow : 1 - weNow} />}
        </article>}
      </>}
      <footer className="wa-foot"><p>검증 전 실험 버전: 추천은 2025년까지 기록으로 학습해 고정한 모델(연구 이름 ARM-B)이에요. 2026 성능 평가는 아직 없어요.{rec?.policy_identity ? ` 식별자 ${rec.policy_identity.slice(0, 8)}` : ''}</p>
        <p>{isReplay ? '끝난 경기의 기록을 실제 경기와 같은 간격으로 다시 흘려보내요.' : '공식 기록이 도착하는 대로, 내 중계 시점에 맞춰 보여 줘요.'} 2초마다 자동으로 새로 고쳐요.{updated && ` 마지막 갱신 ${updated.toLocaleTimeString('ko-KR', { hour12: false })}`}</p></footer>
    </main></div>;
}

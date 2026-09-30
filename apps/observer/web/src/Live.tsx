import { useEffect, useRef, useState } from 'react';
import { BADGE, Board, Header, PitchStrip, Recommendation, WeCard, ZONE_BOUNDS, get, plainZone, short } from './WatchAlong';
import type { Actual, Candidate, Situation, StripItem } from './WatchAlong';
import './watch-along.css';

type Pre = { status: 'ready' | 'unsupported'; reason: string | null; recommendation: { candidates: Candidate[] } | null };
type PaPitch = { key: string; pitch_number: number; status: string; pre: Pre; actual: Actual | null };
type LiveRec = { status: 'ready' | 'unsupported' | 'computing' | 'unavailable'; reason: string | null; key?: string;
  recommendation?: { candidates: Candidate[] } | null; pa_pitches?: PaPitch[];
  previous?: { key: string; actual: Actual; inning: number; half: 'Top' | 'Bot'; home_we_before: number | null } | null;
  home_we_now?: number | null; policy_identity?: string };
type LiveState = { status: string; reason: string | null; badge?: string; delay_s: number;
  game?: { game_pk: number; date: string; status: string; away_team: string; home_team: string };
  situation?: (Situation & { runners: Record<string, boolean> }) | null; pitcher?: { id: number; name: string | null; hand: string } | null;
  batter?: { id: number; name: string | null; side: string } | null;
  pitch_sequence?: { pitch_number: number; pitch_label: string | null; description: string | null }[];
  recommendation: LiveRec | null };

const POLL_MS = 5000;
const WAIT_TEXT: Record<string, string> = { buffering: '지연 중계를 준비하는 중이에요.', between_innings: '이닝 교대 중이에요. 다음 이닝 첫 공을 기다려요.',
  not_live: '지금은 경기 중이 아니에요.', unsupported_game_type: '이 경기 종류는 지원하지 않아요.' };

export default function Live() {
  const gamePk = Number(new URLSearchParams(window.location.search).get('game'));
  const [state, setState] = useState<LiveState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updated, setUpdated] = useState<Date | null>(null);
  const shown = useRef(new Map<string, Candidate | null>());  // recommendation shown before each pitch (this session)

  useEffect(() => {
    if (!gamePk) return;
    let stop = false, timer = 0;
    const tick = async () => {
      try {
        const next = await get<LiveState>(`/live/${gamePk}/state`);
        if (stop) return;
        const rec = next.recommendation;
        if (rec?.key && (rec.status === 'ready' || rec.status === 'unsupported')) shown.current.set(rec.key, rec.recommendation?.candidates[0] ?? null);
        rec?.pa_pitches?.forEach(p => { if (!shown.current.has(p.key)) shown.current.set(p.key, p.pre.recommendation?.candidates[0] ?? null); });
        setState(next); setError(null); setUpdated(new Date());
      } catch (e) { if (!stop) setError((e as Error).message); }
      if (!stop) timer = window.setTimeout(tick, document.hidden ? POLL_MS * 3 : POLL_MS);
    };
    void tick();
    return () => { stop = true; window.clearTimeout(timer); };
  }, [gamePk]);

  if (!gamePk) return <div className="wa-page"><Header badge={BADGE} back /><main className="wa-main"><p className="wa-error">경기 번호가 없어요.</p><a className="wa-button" href="/watch">경기 목록으로</a></main></div>;
  const rec = state?.recommendation ?? null;
  const s = state?.situation ?? null, game = state?.game;
  const previous = rec?.previous ?? null;
  const prevTop = previous ? shown.current.get(previous.key) ?? null : null;
  const strip: StripItem[] = [...(rec?.pa_pitches ?? []).map(p => ({ key: p.key, label: p.actual?.pitch_label ?? '—', result: p.actual?.result_label ?? '',
    match: p.actual?.pitch_type && p.pre.recommendation ? p.pre.recommendation.candidates[0].pitch_type === p.actual.pitch_type : null })),
    ...(rec?.key ? [{ key: rec.key, label: '?', result: '다음 공', match: null, current: true }] : [])];
  const prevBattingHome = previous?.half === 'Bot';
  const weBefore = previous?.home_we_before ?? null, weNow = rec?.home_we_now ?? null;
  return <div className="wa-page"><Header badge={state?.badge ?? BADGE} back />
    {s && game && state?.pitcher && state.batter ? <Board s={s} away={game.away_team} home={game.home_team} pitcher={state.pitcher.name ?? String(state.pitcher.id)}
      batter={`${state.batter.name ?? state.batter.id} (${state.batter.side === 'L' ? '좌' : '우'}타)`} extra={`약 ${Math.round(state.delay_s)}초 지연`} /> : null}
    <main className="wa-main">
      {game && <p className="wa-live-title"><span className="wa-live-dot">지연 중계</span> {short(game.away_team)} @ {short(game.home_team)}</p>}
      {error && <p className="wa-error">{error} {state ? '마지막으로 받은 상황을 보여 주고 있어요. 5초마다 다시 시도해요.' : ''}</p>}
      {!state && !error && <p className="wa-loading">경기 상황을 불러오는 중…</p>}
      {state && state.status !== 'ready' && <article className="wa-card"><h2 className={`wa-none ${state.status === 'buffering' ? 'wa-pulse' : ''}`}>{WAIT_TEXT[state.status] ?? '잠시 기다려 주세요.'}</h2>
        <p className="wa-reason">{state.reason}</p>{state.status === 'not_live' && <p className="wa-reason">끝난 경기는 정리되면 <a href="/watch">경기 목록</a>의 “중계 영상과 함께 보기”에 올라와요.</p>}</article>}
      {state?.status === 'ready' && <>
        <PitchStrip items={strip} />
        <article className="wa-card">
          <p className="wa-eyebrow">다음 공 · 투구 전 추천</p>
          {rec?.status === 'ready' && rec.recommendation ? <Recommendation candidates={rec.recommendation.candidates} bounds={ZONE_BOUNDS} actual={null} note="위치는 이 투수의 2025년까지 실제 투구 분포 기반 근사이며 평가되지 않았습니다." />
            : rec?.status === 'computing' ? <><h2 className="wa-none wa-pulse">추천 계산 중…</h2><p className="wa-reason">{rec.reason}</p></>
            : <><h2 className="wa-none">추천 없음</h2><p className="wa-reason">{rec?.reason ?? '이 상황은 추천하지 않아요.'}</p></>}
        </article>
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
        <p>실제 중계보다 늦게 보여 줘요. 5초마다 자동으로 새로 고쳐요.{updated && ` 마지막 갱신 ${updated.toLocaleTimeString('ko-KR', { hour12: false })}`}</p></footer>
    </main></div>;
}

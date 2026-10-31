import { useCallback, useEffect, useRef, useState } from 'react';
import { buildInningPresentation, type InningPresentation } from './inningResult';
import { listDecisionGames, listDecisions, resolveDecision, type DecisionGame, type DecisionSummary } from './inningDecisionApi';
import './inning-decisions.css';

const message = (error: unknown) => error instanceof Error ? error.message : '기록을 불러오지 못했어요. 다시 시도해 주세요.';
const half = (value: 'Top' | 'Bot') => value === 'Top' ? '초' : '말';
const bases = (mask: number) => [1, 2, 3].filter(base => (mask & (1 << (base - 1))) !== 0);
const dateLabel = (date: string) => /^\d{4}-\d{2}-\d{2}$/.test(date) ? date.replace(/-/g, '.') : date;

function Situation({ decision }: { decision: DecisionSummary }) {
  const state = decision.context.initial_state;
  const occupied = bases(state.bases);
  return <div className="inning-situation" aria-label={`${state.inning}회 ${half(state.topbot)}, 원정 ${state.away_score} 대 홈 ${state.home_score}, ${state.outs}아웃, ${state.balls}볼 ${state.strikes}스트라이크, 주자 ${occupied.length ? occupied.join(', ') + '루' : '없음'}`}>
    <div className="inning-situation-top"><span>{state.inning}회 {half(state.topbot)}</span><strong>원정 {state.away_score}<i>:</i>{state.home_score} 홈</strong></div>
    <div className="inning-situation-bottom"><span>{state.outs}아웃</span><span>주자 {occupied.length ? occupied.join(' · ') + '루' : '없음'}</span><span>{state.balls}볼 · {state.strikes}스트라이크</span></div>
  </div>;
}

export default function InningDecisions() {
  const [games, setGames] = useState<DecisionGame[]>([]);
  const [gameId, setGameId] = useState<number | null>(null);
  const [decisions, setDecisions] = useState<DecisionSummary[]>([]);
  const [decisionId, setDecisionId] = useState('');
  const [presentation, setPresentation] = useState<InningPresentation | null>(null);
  const [phase, setPhase] = useState<'games' | 'decisions' | 'resolve' | null>(null);
  const [gamesError, setGamesError] = useState<string | null>(null);
  const [decisionsError, setDecisionsError] = useState<string | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);

  const begin = useCallback(() => {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    generation.current += 1;
    setPresentation(null);
    setResultError(null);
    return { signal: next.signal, id: generation.current };
  }, []);

  const loadDecisions = useCallback(async (id: number) => {
    const request = begin();
    setGameId(id);
    setDecisions([]);
    setDecisionId('');
    setDecisionsError(null);
    setPhase('decisions');
    try {
      const next = await listDecisions(id, request.signal);
      if (generation.current !== request.id) return;
      setDecisions(next);
    } catch (error) {
      if (generation.current !== request.id || request.signal.aborted) return;
      setDecisionsError(message(error));
    } finally {
      if (generation.current === request.id) setPhase(null);
    }
  }, [begin]);

  const loadGames = useCallback(async () => {
    const request = begin();
    setGames([]);
    setGameId(null);
    setDecisions([]);
    setDecisionId('');
    setGamesError(null);
    setDecisionsError(null);
    setPhase('games');
    try {
      const next = await listDecisionGames(request.signal);
      if (generation.current !== request.id) return;
      setGames(next);
      if (next.length) await loadDecisions(next[0].game_id);
      else setPhase(null);
    } catch (error) {
      if (generation.current !== request.id || request.signal.aborted) return;
      setGamesError(message(error));
      setPhase(null);
    }
  }, [begin, loadDecisions]);

  useEffect(() => {
    void loadGames();
    return () => { controller.current?.abort(); generation.current += 1; };
  }, [loadGames]);

  const selected = decisions.find(item => item.decision_id === decisionId) ?? null;
  const game = games.find(item => item.game_id === gameId);

  async function showResult() {
    if (!selected) return;
    const request = begin();
    setPhase('resolve');
    try {
      const result = await resolveDecision(selected, request.signal);
      if (generation.current !== request.id) return;
      setPresentation(buildInningPresentation(result));
    } catch (error) {
      if (generation.current !== request.id || request.signal.aborted) return;
      setResultError(message(error));
    } finally {
      if (generation.current === request.id) setPhase(null);
    }
  }

  function selectDecision(id: string) {
    controller.current?.abort();
    generation.current += 1;
    setPhase(null);
    setDecisionId(id);
    setPresentation(null);
    setResultError(null);
  }

  return <>
    <header className="inning-header"><a className="inning-brand" href="/" aria-label="Pitcheezy 홈으로"><span className="inning-brand-mark">P<span>·</span></span>pitcheezy</a><div className="inning-header-right"><span className="inning-mode"><i />과거 기록 검토</span><a href="/" className="inning-home">관전 화면으로 <span aria-hidden="true">↗</span></a></div></header>
    <main className="inning-page">
      <div className="inning-intro"><span className="eyebrow">HISTORICAL DECISION REVIEW</span><h1>교체 직전 기록</h1><p>저장된 교체 직전 상황을 골라, 그 시점의 조건부 이닝 전망을 살펴보세요.</p></div>
      <section className="inning-picker" aria-labelledby="inning-picker-title">
        <div className="inning-picker-head"><div><span className="eyebrow">01 · 기록 선택</span><h2 id="inning-picker-title">어느 교체 직전인가요?</h2></div><span className="inning-picker-note">서버에 등록된 과거 기록</span></div>
        {phase === 'games' ? <p className="inning-status" role="status"><span className="spinner" />등록된 경기를 확인하는 중…</p> : gamesError ? <div className="inning-alert" role="alert"><p>{gamesError}</p><button type="button" className="button secondary" onClick={() => void loadGames()}>다시 시도</button></div> : games.length === 0 ? <div className="inning-empty"><strong>아직 등록된 교체 직전 기록이 없어요.</strong><p>기록이 등록되면 여기에서 경기와 결정 시점을 선택할 수 있습니다.</p><button type="button" className="button secondary" onClick={() => void loadGames()}>목록 새로고침</button></div> : <>
          <label className="inning-field" htmlFor="inning-game">등록된 경기<span>경기 날짜와 기록 수로 구분합니다.</span><select id="inning-game" value={gameId ?? ''} onChange={event => void loadDecisions(Number(event.target.value))}><option value="" disabled>경기를 선택하세요</option>{games.map(item => <option key={item.game_id} value={item.game_id}>{dateLabel(item.date)} · 경기 #{item.game_id} · 교체 직전 {item.decision_count}건</option>)}</select></label>
          {phase === 'decisions' ? <p className="inning-status" role="status"><span className="spinner" />교체 직전 기록을 확인하는 중…</p> : decisionsError ? <div className="inning-alert" role="alert"><p>{decisionsError}</p><button type="button" className="button secondary" onClick={() => gameId !== null && void loadDecisions(gameId)}>다시 시도</button></div> : decisions.length === 0 ? <div className="inning-empty"><strong>이 경기에는 등록된 교체 직전 기록이 없습니다.</strong><p>다른 경기를 선택하거나 목록을 새로고침해 주세요.</p></div> : <fieldset className="inning-decisions"><legend>교체 직전 시점</legend><p>상황을 선택한 뒤 이닝 전망을 요청할 수 있습니다.</p><div className="inning-decision-list">{decisions.map((item, index) => { const state = item.context.initial_state; return <label key={item.decision_id} className={`inning-decision-option ${decisionId === item.decision_id ? 'selected' : ''}`}><input type="radio" name="inning-decision" value={item.decision_id} checked={decisionId === item.decision_id} onChange={() => selectDecision(item.decision_id)} /><span className="inning-option-number">{String(index + 1).padStart(2, '0')}</span><span><strong>{state.inning}회 {half(state.topbot)} · {state.outs}아웃</strong><small>{dateLabel(state.date)} · 투수 교체 직전 · 투수 #{item.context.linkage.keep_pitcher_id}</small></span><span className="inning-option-arrow" aria-hidden="true">↗</span></label>; })}</div></fieldset>}
        </>}
      </section>
      {selected && <section className="inning-review" aria-labelledby="inning-review-title"><div className="inning-review-head"><div><span className="eyebrow">02 · 선택한 시점</span><h2 id="inning-review-title">당시 경기 상황</h2></div><span className="inning-review-date">{dateLabel(game?.date ?? selected.context.initial_state.date)}</span></div><Situation decision={selected} /><div className="inning-action"><p>이 기록은 교체 직전 문맥에 저장된 과거 평가입니다. 현재 재생 화면의 타석과 연결되지 않습니다.</p><button type="button" className="button primary" disabled={phase === 'resolve'} onClick={() => void showResult()}>{phase === 'resolve' ? <><span className="spinner light" />전망 확인 중…</> : '이닝 전망 보기'}<span aria-hidden="true">→</span></button></div>{resultError && <div className="inning-alert" role="alert"><p>{resultError}</p><button type="button" className="button secondary" onClick={() => void showResult()}>다시 시도</button></div>}</section>}
      {presentation && selected && <section className="inning-result" aria-labelledby="inning-result-title"><div className="inning-result-head"><div><span className="eyebrow">03 · 저장된 평가 결과</span><h2 id="inning-result-title">{presentation.title}</h2></div><span className="inning-result-tag">과거 기록</span></div><p className="inning-result-scope">{presentation.scope}</p><div className="inning-result-main"><span>{presentation.valueLabel}</span>{presentation.rangeLabel ? <strong>{presentation.rangeLabel}</strong> : <strong className="inning-no-result">계산 결과 없음</strong>}<p>{presentation.unresolvedLabel ?? presentation.reasonLabel}</p></div><div className="inning-result-facts"><div><span>고정 조건</span><strong>{presentation.assumptions[0]}</strong></div><div><span>미해결 범위</span><strong>{presentation.unresolvedLabel ?? '계산 결과 없음'}</strong></div><div><span>기본 프로필</span><strong>{presentation.profileNote}</strong></div><div><span>교체 비교</span><strong>{presentation.replacementLabel}</strong></div></div><details className="inning-assumptions"><summary>모형의 조건과 해석 범위 보기 <span aria-hidden="true">＋</span></summary><ul>{presentation.assumptions.map(item => <li key={item}>{item}</li>)}</ul><p>이 값은 현재 반이닝 또는 경기 종료까지 조건을 고정한 경기 승리 확률의 범위입니다. 신뢰구간이나 실제 교체 효과를 뜻하지 않습니다.</p></details></section>}
    </main>
  </>;
}

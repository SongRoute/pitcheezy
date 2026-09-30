import { useCallback, useEffect, useMemo, useState } from 'react';
import './watch-along.css';

type Bounds = { bottom: number; top: number };
type Target = { x: number; z: number };
type Candidate = { rank: number; pitch_type: string; pitch_label: string; zone_id: string | null; zone_label: string; target: Target | null;
  detail: { probability: number; reference_probability: number; kernel_mass: number | null; kernel_ess: number | null } };
type Situation = { inning: number; half: 'Top' | 'Bot'; outs: number; balls: number; strikes: number; bases: number; home_score: number; away_score: number };
type Person = { id: number; name: string | null };
type Decision = { index: number; pa_id: string; at_bat_number: number; pitch_number: number; situation: Situation;
  pitcher: Person & { hand: string }; batter: Person & { side: string }; status: string;
  pre: { status: 'ready' | 'unsupported'; reason: string | null; recommendation: { candidates: Candidate[] } | null } };
type Reveal = { index: number;
  actual: { pitch_type: string | null; pitch_label: string; result_label: string; event_label: string | null; play_text: string | null;
    speed_mph: number | null; x: number | null; z: number | null; zone_label: string };
  we: { home_before: number | null; home_after: number | null; home_delta: number | null } };
type Timeline = { badge: string; game: { game_pk: number; date: string; game_type: string; away_team: string; home_team: string };
  policy: { name: string; identity_sha256: string; tau: number; note: string }; location: { note: string; zone_bounds: Bounds };
  we_note: string; coverage: { pitch_decisions: number; ready: number; ready_share_of_pitches: number | null }; decisions: Decision[] };
type GameItem = Timeline['game'];

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`); } catch { throw new Error('관전 서버에 연결하지 못했어요.'); }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : '자료를 불러오지 못했어요.');
  return body as T;
}
const halfLabel = (s: Situation) => `${s.inning}회 ${s.half === 'Top' ? '초' : '말'}`;
const percent = (v: number | null) => v === null ? '—' : `${Math.round(v * 100)}%`;
const short = (team: string) => team.split(' ').slice(-1)[0];
const readHash = () => { const m = /i=(\d+)/.exec(window.location.hash); return m ? Number(m[1]) : null; };

function Bases({ mask }: { mask: number }) {
  const on = (base: number) => (mask & (1 << (base - 1))) !== 0;
  const runners = [1, 2, 3].filter(on);
  return <svg className="wa-bases" viewBox="0 0 60 44" role="img" aria-label={`주자 ${runners.length ? runners.join(', ') + '루' : '없음'}`}>
    {[[30, 6, 2], [48, 22, 1], [12, 22, 3]].map(([x, y, base]) => <rect key={base} x={x - 7} y={y - 7} width="14" height="14" transform={`rotate(45 ${x} ${y})`} className={on(base) ? 'on' : ''} />)}
  </svg>;
}

function Zone({ bounds, candidates, actual }: { bounds: Bounds; candidates: Candidate[]; actual: Reveal['actual'] | null }) {
  const x = (v: number) => 90 + v / 1.6 * 80, y = (v: number) => 200 - (v - .6) / 3.8 * 190;
  const left = x(-.83), right = x(.83), top = y(bounds.top), bottom = y(bounds.bottom);
  const cells = [0, 1, 2];
  const seen = new Set<string>();
  return <svg className="wa-zone" viewBox="0 0 180 210" role="img" aria-label="포수 시점 스트라이크존: 추천 대략 위치와 실제 투구">
    <rect x={left} y={top} width={right - left} height={bottom - top} className="wa-zone-box" />
    {cells.map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="wa-zone-grid" />)}
    {cells.map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="wa-zone-grid" />)}
    {candidates.filter(c => c.target).map(c => {
      const key = `${c.target!.x}:${c.target!.z}`, offset = seen.has(key) ? 9 : 0; seen.add(key);
      return <g key={c.rank} className={`wa-target rank-${c.rank}`}><circle cx={x(c.target!.x) + offset} cy={y(c.target!.z)} r={c.rank === 1 ? 11 : 8} /><text x={x(c.target!.x) + offset} y={y(c.target!.z) + 4} textAnchor="middle">{c.rank}</text></g>;
    })}
    {actual && actual.x !== null && actual.z !== null && <circle className="wa-actual" cx={x(Math.max(-1.1, Math.min(1.1, actual.x)))} cy={y(Math.max(.7, Math.min(4.3, actual.z)))} r="7" />}
    <text x="90" y="207" textAnchor="middle" className="wa-zone-caption">포수 시점</text>
  </svg>;
}

export default function WatchAlong() {
  const params = new URLSearchParams(window.location.search);
  const [gamePk, setGamePk] = useState<number | null>(params.get('game') ? Number(params.get('game')) : null);
  const [games, setGames] = useState<GameItem[] | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [index, setIndex] = useState(0);
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (gamePk !== null) return;
    get<{ games: GameItem[] }>('/watch/games').then(r => setGames(r.games)).catch(e => setError(e.message));
  }, [gamePk]);
  useEffect(() => {
    if (gamePk === null) return;
    setTimeline(null); setError(null);
    get<Timeline>(`/watch/${gamePk}`).then(t => {
      setTimeline(t);
      let start = readHash();
      if (start === null) { try { start = Number(localStorage.getItem(`pitcheezy.watch.${gamePk}`) ?? 0); } catch { start = 0; } }
      setIndex(Math.min(Math.max(0, start || 0), t.decisions.length - 1));
    }).catch(e => setError(e.message));
  }, [gamePk]);
  useEffect(() => {
    if (!timeline) return;
    window.history.replaceState(null, '', `?game=${timeline.game.game_pk}#i=${index}`);
    try { localStorage.setItem(`pitcheezy.watch.${timeline.game.game_pk}`, String(index)); } catch { /* storage may be unavailable */ }
  }, [index, timeline]);

  const decision = timeline?.decisions[index] ?? null;
  const plateAppearances = useMemo(() => {
    const out: { first: number; label: string }[] = [];
    timeline?.decisions.forEach(d => { if (!out.length || timeline.decisions[out[out.length - 1].first].pa_id !== d.pa_id) out.push({ first: d.index, label: `${halfLabel(d.situation)} · ${d.batter.name ?? d.batter.id}` }); });
    return out;
  }, [timeline]);
  const currentPa = plateAppearances.reduce((found, pa, i) => pa.first <= index ? i : found, 0);

  const doReveal = useCallback(async () => {
    if (!timeline || busy) return;
    setBusy(true);
    try { setReveal(await get<Reveal>(`/watch/${timeline.game.game_pk}/reveal/${index}`)); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }, [timeline, index, busy]);
  const go = useCallback((next: number) => { setReveal(null); setIndex(next); }, []);
  const step = useCallback((delta: number) => { if (timeline) go(Math.min(Math.max(0, index + delta), timeline.decisions.length - 1)); }, [timeline, index, go]);
  useEffect(() => {  // a typed or shared #i=N link jumps to that pitch
    const onHash = () => { const target = readHash(); if (timeline && target !== null && target < timeline.decisions.length) go(target); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [timeline, go]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.tagName === 'SELECT') return;
      if (event.key === 'ArrowRight' || event.key === ' ') { event.preventDefault(); if (reveal) step(1); else void doReveal(); }
      if (event.key === 'ArrowLeft') step(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reveal, step, doReveal]);

  if (gamePk === null) return <div className="wa-page"><Header badge="검증 전 실험 버전 · 위치는 실제 투구 분포 근사" />
    <main className="wa-main"><h1 className="wa-title">중계 영상과 함께 보기</h1>
      {error && <p className="wa-error">{error}</p>}
      {games === null ? <p className="wa-muted">경기 목록을 불러오는 중…</p> : games.length === 0 ? <p className="wa-muted">사전 계산된 경기가 없습니다. <code>scripts/demo_precompute.py precompute --game-pk N</code>을 먼저 실행해 주세요.</p> :
        <ul className="wa-games">{games.map(g => <li key={g.game_pk}><button onClick={() => setGamePk(g.game_pk)}><strong>{g.away_team} @ {g.home_team}</strong><span>{g.date} · gamePk {g.game_pk}</span></button></li>)}</ul>}
    </main></div>;
  if (!timeline || !decision) return <div className="wa-page"><Header badge="검증 전 실험 버전 · 위치는 실제 투구 분포 근사" /><main className="wa-main">{error ? <p className="wa-error">{error}</p> : <p className="wa-muted">경기 자료를 불러오는 중…</p>}</main></div>;

  const s = decision.situation, rec = decision.pre.recommendation, top = rec?.candidates[0];
  const game = timeline.game;
  return <div className="wa-page"><Header badge={timeline.badge} />
    <section className="wa-board" aria-live="polite">
      <div className="wa-score"><span>{short(game.away_team)}</span><strong>{s.away_score}</strong><i>:</i><strong>{s.home_score}</strong><span>{short(game.home_team)}</span></div>
      <div className="wa-inning"><strong>{halfLabel(s)}</strong><span>{s.outs}아웃</span><Bases mask={s.bases} /><span className="wa-count">B{s.balls} S{s.strikes}</span></div>
      <div className="wa-matchup"><span>투수 <b>{decision.pitcher.name ?? decision.pitcher.id}</b></span><span>타자 <b>{decision.batter.name ?? decision.batter.id}</b> ({decision.batter.side === 'L' ? '좌' : '우'}타)</span><span className="wa-pitchno">{decision.pitch_number}구째</span></div>
    </section>
    <main className="wa-main">
      <nav className="wa-nav" aria-label="투구 이동">
        <button onClick={() => step(-1)} disabled={index === 0} aria-label="이전 공">←</button>
        <select value={currentPa} onChange={e => go(plateAppearances[Number(e.target.value)].first)} aria-label="타석으로 이동">
          {plateAppearances.map((pa, i) => <option key={pa.first} value={i}>{pa.label}</option>)}
        </select>
        <span className="wa-progress">{index + 1}/{timeline.decisions.length}</span>
        <button onClick={() => step(1)} disabled={index === timeline.decisions.length - 1} aria-label="다음 공">→</button>
      </nav>
      {error && <p className="wa-error">{error}</p>}
      <article className="wa-card">
        {decision.pre.status === 'ready' && top ? <>
          <p className="wa-eyebrow">투구 전 추천</p>
          <div className="wa-rec">
            <div><h2 className="wa-type">{top.pitch_label}</h2><p className="wa-zone-label">대략 위치 · {top.zone_label}</p>
              <ol className="wa-others">{rec!.candidates.slice(1).map(c => <li key={c.rank}><b>{c.rank}</b> {c.pitch_label} <span>· {c.zone_label}</span></li>)}</ol></div>
            <Zone bounds={timeline.location.zone_bounds} candidates={rec!.candidates} actual={reveal?.actual ?? null} />
          </div>
          <details className="wa-numbers"><summary>숫자 보기</summary>
            <table><thead><tr><th>구종</th><th>추천 확률</th><th>기준(평소) 확률</th></tr></thead>
              <tbody>{rec!.candidates.map(c => <tr key={c.rank}><td>{c.pitch_label}</td><td>{percent(c.detail.probability)}</td><td>{percent(c.detail.reference_probability)}</td></tr>)}</tbody></table>
            <p>{timeline.location.note}</p></details>
        </> : <><p className="wa-eyebrow">투구 전 추천</p><h2 className="wa-none">추천 없음</h2><p className="wa-reason">{decision.pre.reason}</p></>}
        {!reveal ? <button className="wa-primary" onClick={() => void doReveal()} disabled={busy}>실제 투구 공개</button> : <Revealed reveal={reveal} top={top ?? null} decision={decision} game={game} onNext={() => step(1)} last={index === timeline.decisions.length - 1} />}
      </article>
      <footer className="wa-foot"><p>{timeline.policy.note}</p><p>추천: {timeline.policy.name} · τ {timeline.policy.tau} · 식별자 {timeline.policy.identity_sha256.slice(0, 8)}</p>
        <p>이 경기에서 추천을 낸 공 {timeline.coverage.ready}/{timeline.coverage.pitch_decisions} · {timeline.we_note}</p><p className="wa-muted">키보드: → 또는 스페이스 = 공개/다음, ← = 이전</p></footer>
    </main></div>;
}

function Header({ badge }: { badge: string }) {
  return <header className="wa-header"><a className="wa-brand" href="/">Pitcheezy<span>.</span></a><span className="wa-badge">{badge}</span></header>;
}

function Revealed({ reveal, top, decision, game, onNext, last }: { reveal: Reveal; top: Candidate | null; decision: Decision; game: GameItem; onNext: () => void; last: boolean }) {
  const a = reveal.actual, we = reveal.we;
  const battingHome = decision.situation.half === 'Bot';
  const team = battingHome ? game.home_team : game.away_team;
  const before = we.home_before === null ? null : battingHome ? we.home_before : 1 - we.home_before;
  const after = we.home_after === null ? null : battingHome ? we.home_after : 1 - we.home_after;
  const delta = before !== null && after !== null ? after - before : null;
  return <section className="wa-reveal">
    <p className="wa-eyebrow">실제 투구</p>
    <h3>{a.pitch_label}{a.speed_mph !== null && <span> · {a.speed_mph.toFixed(1)} mph</span>}</h3>
    <p className="wa-result">{a.event_label ?? a.result_label} · {a.zone_label}</p>
    {top && a.pitch_type && <p className={`wa-match ${top.pitch_type === a.pitch_type ? 'same' : ''}`}>{top.pitch_type === a.pitch_type ? '추천 1순위 구종과 같았어요' : `추천 1순위(${top.pitch_label})와 다른 구종`}</p>}
    {a.play_text && <p className="wa-play">{a.play_text}</p>}
    <p className="wa-we">{short(team)} 공격 승리확률 {percent(before)} → {percent(after)}{delta !== null && <b className={delta > 0 ? 'up' : delta < 0 ? 'down' : ''}> ({delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)}%p)</b>}</p>
    {!last && <button className="wa-primary" onClick={onNext}>다음 공 →</button>}
  </section>;
}

import { useCallback, useEffect, useMemo, useState } from 'react';
import './watch-along.css';

export const BADGE = '검증 전 실험 버전 · 위치는 실제 투구 분포 근사';
export type Bounds = { bottom: number; top: number };
type Target = { x: number; z: number };
export type Candidate = { rank: number; pitch_type: string; pitch_label: string; zone_id: string | null; zone_label: string; target: Target | null;
  detail: { probability: number; reference_probability: number; kernel_mass: number | null; kernel_ess: number | null } };
export type Situation = { inning: number; half: 'Top' | 'Bot'; outs: number; balls: number; strikes: number; bases: number; home_score: number; away_score: number };
type Person = { id: number; name: string | null };
type Decision = { index: number; pa_id: string; at_bat_number: number; pitch_number: number; situation: Situation;
  pitcher: Person & { hand: string }; batter: Person & { side: string }; status: string;
  pre: { status: 'ready' | 'unsupported'; reason: string | null; recommendation: { candidates: Candidate[] } | null } };
export type Actual = { pitch_type: string | null; pitch_label: string; result_label: string; event_label: string | null; play_text: string | null;
  speed_mph: number | null; x: number | null; z: number | null; zone_label: string };
type Reveal = { index: number; actual: Actual; we: { home_before: number | null; home_after: number | null; home_delta: number | null } };
type Timeline = { badge: string; game: { game_pk: number; date: string; game_type: string; away_team: string; home_team: string };
  policy: { name: string; identity_sha256: string; tau: number; note: string }; location: { note: string; zone_bounds: Bounds };
  we_note: string; coverage: { pitch_decisions: number; ready: number; ready_share_of_pitches: number | null }; decisions: Decision[] };
type GameItem = Timeline['game'] & { pitches?: number; ready?: number; ready_share?: number | null };
type LiveGame = { game_pk: number; game_type: string; start: string | null; status: string; live: boolean; away_team: string; home_team: string };

export const ZONE_BOUNDS: Bounds = { bottom: 1.6, top: 3.3899 };
const TEAMS: Record<string, string> = {
  'Arizona Diamondbacks': '애리조나', 'Athletics': '애슬레틱스', 'Atlanta Braves': '애틀랜타', 'Baltimore Orioles': '볼티모어',
  'Boston Red Sox': '보스턴', 'Chicago Cubs': '컵스', 'Chicago White Sox': '화이트삭스', 'Cincinnati Reds': '신시내티',
  'Cleveland Guardians': '클리블랜드', 'Colorado Rockies': '콜로라도', 'Detroit Tigers': '디트로이트', 'Houston Astros': '휴스턴',
  'Kansas City Royals': '캔자스시티', 'Los Angeles Angels': '에인절스', 'Los Angeles Dodgers': '다저스', 'Miami Marlins': '마이애미',
  'Milwaukee Brewers': '밀워키', 'Minnesota Twins': '미네소타', 'New York Mets': '메츠', 'New York Yankees': '양키스',
  'Philadelphia Phillies': '필라델피아', 'Pittsburgh Pirates': '피츠버그', 'San Diego Padres': '샌디에이고',
  'San Francisco Giants': '샌프란시스코', 'Seattle Mariners': '시애틀', 'St. Louis Cardinals': '세인트루이스',
  'Tampa Bay Rays': '탬파베이', 'Texas Rangers': '텍사스', 'Toronto Blue Jays': '토론토', 'Washington Nationals': '워싱턴' };
const SERIES: Record<string, string> = { F: '와일드카드', D: '디비전시리즈', L: '챔피언십시리즈', W: '월드시리즈', R: '정규시즌' };

export async function get<T>(path: string): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`, { cache: 'no-store' }); } catch { throw new Error('관전 서버에 연결하지 못했어요. 네트워크(Tailscale) 연결을 확인해 주세요.'); }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : '자료를 불러오지 못했어요.');
  return body as T;
}
export const halfLabel = (s: { inning: number; half: 'Top' | 'Bot' }) => `${s.inning}회 ${s.half === 'Top' ? '초' : '말'}`;
export const percent = (v: number | null | undefined) => v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`;
export const short = (team: string) => TEAMS[team] ?? team.split(' ').slice(-1)[0];
/** Plainer zone words: '가운데 높이 중앙' → '한가운데'. Left/right stay catcher's view (the zone caption says so). */
export const plainZone = (label: string) => label.replace('가운데 높이 중앙', '한가운데').replace('가운데 높이 ', '가운데 ');
const GUIDE_KEY = 'pitcheezy.guide.v1';
const guideSeen = () => { try { return localStorage.getItem(GUIDE_KEY) === '1'; } catch { return false; } };
const readHash = () => { const m = /i=(\d+)/.exec(window.location.hash); return m ? Number(m[1]) : null; };
const nyDate = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York' }).format(new Date());
const kstTime = (iso: string | null) => iso ? new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(iso)) : '';

export function Bases({ mask }: { mask: number }) {
  const on = (base: number) => (mask & (1 << (base - 1))) !== 0;
  const runners = [1, 2, 3].filter(on);
  return <svg className="wa-bases" viewBox="0 0 60 44" role="img" aria-label={`주자 ${runners.length ? runners.join(', ') + '루' : '없음'}`}>
    {[[30, 6, 2], [48, 22, 1], [12, 22, 3]].map(([x, y, base]) => <rect key={base} x={x - 7} y={y - 7} width="14" height="14" transform={`rotate(45 ${x} ${y})`} className={on(base) ? 'on' : ''} />)}
  </svg>;
}

function Outs({ outs }: { outs: number }) {
  return <span className="wa-outs" aria-label={`${outs}아웃`}>{[0, 1, 2].map(i => <i key={i} className={i < outs ? 'on' : ''} />)}<em>{outs}아웃</em></span>;
}

export function Zone({ bounds, candidates, actual }: { bounds: Bounds; candidates: Candidate[]; actual: { x: number | null; z: number | null } | null }) {
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
    <text x="90" y="207" textAnchor="middle" className="wa-zone-caption">포수 쪽에서 본 존 · 숫자=추천 · 주황=실제</text>
  </svg>;
}

export function Header({ badge, back, onHelp }: { badge: string; back?: boolean; onHelp?: () => void }) {
  return <header className="wa-header">
    <div className="wa-header-row">{back ? <a className="wa-back" href="/watch" aria-label="경기 목록으로">←</a> : null}<a className="wa-brand" href="/watch">Pitcheezy<span>.</span></a>
      {onHelp && <button className="wa-help" onClick={onHelp}>보는 법</button>}</div>
    <p className="wa-badge">{badge}</p></header>;
}

/** First-visit, one-screen guide; dismissal is remembered on this browser only. */
export function Guide({ onClose }: { onClose: () => void }) {
  const close = () => { try { localStorage.setItem(GUIDE_KEY, '1'); } catch { /* private mode: show again next time */ } onClose(); };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); e.stopImmediatePropagation(); close(); } };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  });
  return <div className="wa-guide" role="dialog" aria-modal="true" aria-labelledby="wa-guide-title"><div className="wa-guide-box">
    <h2 id="wa-guide-title">이렇게 보세요</h2>
    <ol>
      <li><b>영상과 맞추기</b> 중계 영상을 틀고, 위쪽 <em>회</em> 버튼이나 타석 목록으로 영상과 같은 장면을 찾아요.</li>
      <li><b>던지기 전에 보기</b> 투수가 공을 던지기 전에 추천 구종과 대략 위치를 봐요.</li>
      <li><b>넘겨서 비교하기</b> 공이 들어오면 아래 버튼을 눌러 실제 투구를 공개하고, 한 번 더 누르면 다음 공으로 가요.</li>
    </ol>
    <p className="wa-guide-note">추천은 2025년까지 기록으로 만든 실험 버전이라 아직 성능 검증 전이에요.</p>
    <button className="wa-primary" onClick={close} autoFocus>알겠어요</button>
  </div></div>;
}

export function Board({ s, away, home, pitcher, batter, extra }: { s: Situation; away: string; home: string; pitcher: string; batter: string; extra?: string }) {
  return <section className="wa-board" aria-live="polite">
    <div className="wa-score"><span>{short(away)}</span><strong>{s.away_score}</strong><i>:</i><strong>{s.home_score}</strong><span>{short(home)}</span></div>
    <div className="wa-inning"><strong>{halfLabel(s)}</strong><Outs outs={s.outs} /><Bases mask={s.bases} /><span className="wa-count" aria-label={`볼 ${s.balls}, 스트라이크 ${s.strikes}`}>{s.balls}-{s.strikes}</span>{extra && <span className="wa-pitchno">{extra}</span>}</div>
    <div className="wa-matchup"><span>투수 <b>{pitcher}</b></span><span>타자 <b>{batter}</b></span></div>
  </section>;
}

/** Type first, then the approximate zone; numbers folded (D49). */
export function Recommendation({ candidates, bounds, actual, note }: { candidates: Candidate[]; bounds: Bounds; actual: { x: number | null; z: number | null } | null; note: string }) {
  const top = candidates[0];
  return <>
    <div className="wa-rec">
      <div className="wa-rec-text"><h2 className="wa-type">{top.pitch_label}</h2><p className="wa-zone-label">{plainZone(top.zone_label)} <span>쪽으로</span></p>
        {candidates.length > 1 && <><p className="wa-kicker wa-others-title">다른 후보</p>
          <ol className="wa-others">{candidates.slice(1).map(c => <li key={c.rank}><b>{c.rank}</b><span>{c.pitch_label} <em>· {plainZone(c.zone_label)}</em></span></li>)}</ol></>}</div>
      <Zone bounds={bounds} candidates={candidates} actual={actual} />
    </div>
    <details className="wa-numbers"><summary>숫자로 보기</summary>
      <table><thead><tr><th>구종</th><th>모델이 고를 확률</th><th>이 투수 평소</th></tr></thead>
        <tbody>{candidates.map(c => <tr key={c.rank}><td>{c.pitch_label}</td><td>{percent(c.detail.probability)}</td><td>{percent(c.detail.reference_probability)}</td></tr>)}</tbody></table>
      <p><b>모델이 고를 확률</b>은 추천 모델이 이 상황에서 그 구종을 고르는 비율, <b>이 투수 평소</b>는 비슷한 상황에서 이 투수가 2025년까지 던진 방식을 흉내 낸 기준 모델의 비율이에요. 두 숫자가 비슷하면 “평소대로”에 가까운 추천이에요.</p>
      <p>이 공이 볼·스트라이크·인플레이가 될 확률은 이 자료에 없어요(미측정).</p>
      <p>{note}</p></details>
  </>;
}

/** Win-expectancy change for the team at bat; event contribution beyond WE is not computed yet (D43). */
export function WeCard({ team, before, after }: { team: string; before: number | null; after: number | null }) {
  const delta = before !== null && after !== null ? after - before : null;
  return <div className="wa-we-card">
    <p className="wa-eyebrow">승리확률 변화 · {short(team)} 공격</p>
    <p className="wa-we-line"><span>{percent(before)}</span><i>→</i><strong>{percent(after)}</strong>
      {delta !== null && <b className={delta > 0.0005 ? 'up' : delta < -0.0005 ? 'down' : ''}>{delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)}%p</b>}</p>
    <p className="wa-we-note">{delta !== null && Math.abs(delta) < 0.0005 ? '볼·스트라이크만으로는 움직이지 않아요. 타석이 끝나거나 주자·아웃이 바뀔 때 변해요.' : '이 공 뒤 공격 팀이 이길 확률이 이만큼 바뀌었어요.'}</p>
  </div>;
}

export type StripItem = { key: string; label: string; result: string; match: boolean | null; current?: boolean };
export function PitchStrip({ items }: { items: StripItem[] }) {
  if (!items.length) return null;
  return <ol className="wa-strip" aria-label="이 타석 투구 순서">{items.map((p, i) =>
    <li key={p.key} className={p.current ? 'current' : ''}><span className="n">{i + 1}</span><b>{p.label}</b><em>{p.result}</em>{p.match !== null && <i className={p.match ? 'hit' : ''} title={p.match ? '추천 1순위와 같은 구종' : '추천 1순위와 다른 구종'}>{p.match ? '✓' : '·'}</i>}</li>)}
  </ol>;
}

function Picker() {
  const [games, setGames] = useState<GameItem[] | null>(null);
  const [live, setLive] = useState<LiveGame[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  useEffect(() => {
    get<{ games: GameItem[] }>('/watch/games').then(r => setGames(r.games)).catch(e => setError(e.message));
    get<{ games: LiveGame[] }>(`/live/games?date=${nyDate()}`).then(r => setLive(r.games.filter(g => g.game_type !== 'R' || g.live))).catch(e => setLiveError(e.message));
  }, []);
  return <div className="wa-page"><Header badge={BADGE} />
    <main className="wa-main">
      <h1 className="wa-title">오늘의 포스트시즌</h1>
      <section className="wa-section">
        <h2>지금 경기 · 지연 중계</h2>
        <p className="wa-muted">실제 중계보다 약 30초 늦게, 다음 공의 추천 구종과 대략 위치를 먼저 보여 줘요.</p>
        {liveError ? <p className="wa-error">{liveError}</p> : live === null ? <p className="wa-loading">오늘 경기를 불러오는 중…</p> : live.length === 0 ? <p className="wa-empty">오늘(미국 동부 기준) 예정된 포스트시즌 경기가 없어요.</p> :
          <ul className="wa-games">{live.map(g => <li key={g.game_pk}><a className={`wa-game ${g.live ? 'is-live' : ''}`} href={`/live?game=${g.game_pk}`}>
            <span className="wa-game-teams"><strong>{short(g.away_team)} @ {short(g.home_team)}</strong><span>{SERIES[g.game_type] ?? g.game_type}</span></span>
            <span className="wa-game-meta">{g.live ? <em className="wa-live-dot">경기 중</em> : <em>{g.status === 'Final' || g.status === 'Game Over' ? '종료' : `${kstTime(g.start)} 시작(한국)`}</em>}</span></a></li>)}</ul>}
      </section>
      <section className="wa-section">
        <h2>끝난 경기 · 중계 영상과 함께 보기</h2>
        <p className="wa-muted">한 공씩 넘기며 추천 → 실제 투구를 비교해요. 목록에 최종 점수는 없어요.</p>
        {error && <p className="wa-error">{error}</p>}
        {games === null ? (!error && <p className="wa-loading">경기 목록을 불러오는 중…</p>) : games.length === 0 ? <p className="wa-empty">아직 준비된 경기가 없어요. 맥미니에서 <code>scripts/demo_precompute.py sync</code>를 실행하면 끝난 경기가 추가돼요. (저장장치 T7이 연결돼 있어야 해요)</p> :
          <ul className="wa-games">{games.map(g => <li key={g.game_pk}><a className="wa-game" href={`/watch?game=${g.game_pk}`}>
            <span className="wa-game-teams"><strong>{short(g.away_team)} @ {short(g.home_team)}</strong><span>{g.date} · {SERIES[g.game_type] ?? g.game_type}</span></span>
            {g.ready_share !== undefined && g.ready_share !== null && <span className="wa-coverage" title="추천을 낼 수 있었던 공의 비율">
              <span className="bar"><i style={{ width: `${Math.round(g.ready_share * 100)}%` }} /></span>추천 {Math.round(g.ready_share * 100)}%</span>}</a></li>)}</ul>}
        <p className="wa-footnote">추천 비율이 낮은 경기는 2025년 4월 이후 데뷔했거나 기록이 적은 투수가 많이 던진 경기예요. 그런 투수의 공은 추천 없이 이유만 보여 줘요.</p>
      </section>
    </main></div>;
}

export default function WatchAlong() {
  const params = new URLSearchParams(window.location.search);
  const gamePk = params.get('game') ? Number(params.get('game')) : null;
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [index, setIndex] = useState(0);
  const [reveals, setReveals] = useState<Record<number, Reveal>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [guide, setGuide] = useState(() => !guideSeen());

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
  const reveal = reveals[index] ?? null;
  const plateAppearances = useMemo(() => {
    const out: { first: number; label: string }[] = [];
    timeline?.decisions.forEach(d => { if (!out.length || timeline.decisions[out[out.length - 1].first].pa_id !== d.pa_id) out.push({ first: d.index, label: `${halfLabel(d.situation)} · ${d.batter.name ?? d.batter.id}` }); });
    return out;
  }, [timeline]);
  const halves = useMemo(() => {  // jump targets for syncing with the broadcast video: first pitch of each half-inning
    const out: { first: number; label: string }[] = [];
    timeline?.decisions.forEach(d => { const label = `${d.situation.inning}${d.situation.half === 'Top' ? '초' : '말'}`; if (!out.length || out[out.length - 1].label !== label) out.push({ first: d.index, label }); });
    return out;
  }, [timeline]);
  const currentHalf = halves.reduce((found, h, i) => h.first <= index ? i : found, 0);
  const currentPa = plateAppearances.reduce((found, pa, i) => pa.first <= index ? i : found, 0);
  const paStart = plateAppearances[currentPa]?.first ?? 0;

  const fetchReveal = useCallback(async (i: number) => {
    if (!timeline) return null;
    const r = await get<Reveal>(`/watch/${timeline.game.game_pk}/reveal/${i}`);
    setReveals(prev => ({ ...prev, [i]: r }));
    return r;
  }, [timeline]);
  // Earlier pitches of this plate appearance are already in the past: fill the sequence strip.
  useEffect(() => {
    if (!timeline) return;
    for (let i = paStart; i < index; i++) if (!reveals[i]) void fetchReveal(i).catch(() => undefined);
  }, [timeline, paStart, index, reveals, fetchReveal]);

  const doReveal = useCallback(async () => {
    if (!timeline || busy) return;
    setBusy(true);
    try { await fetchReveal(index); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }, [timeline, index, busy, fetchReveal]);
  const go = useCallback((next: number) => { setError(null); setIndex(next); window.scrollTo({ top: 0 }); }, []);
  const step = useCallback((delta: number) => { if (timeline) go(Math.min(Math.max(0, index + delta), timeline.decisions.length - 1)); }, [timeline, index, go]);
  useEffect(() => {  // a typed or shared #i=N link jumps to that pitch
    const onHash = () => { const target = readHash(); if (timeline && target !== null && target < timeline.decisions.length) go(target); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [timeline, go]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (guide || (event.target as HTMLElement)?.tagName === 'SELECT' || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === 'ArrowRight' || event.key === ' ') { event.preventDefault(); if (reveal) step(1); else void doReveal(); }
      if (event.key === 'ArrowLeft') step(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reveal, step, doReveal, guide]);
  useEffect(() => { document.querySelector('.wa-innings .current')?.scrollIntoView({ block: 'nearest', inline: 'center' }); }, [currentHalf, timeline]);
  useEffect(() => { if (reveal) document.querySelector('.wa-reveal')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }, [reveal]);

  if (gamePk === null) return <Picker />;
  if (!timeline || !decision) return <div className="wa-page"><Header badge={BADGE} back /><main className="wa-main">{error ? <><p className="wa-error">{error}</p><a className="wa-button" href="/watch">경기 목록으로</a></> : <p className="wa-loading">경기 자료를 불러오는 중…</p>}</main></div>;

  const s = decision.situation, rec = decision.pre.recommendation, top = rec?.candidates[0];
  const game = timeline.game;
  const strip: StripItem[] = timeline.decisions.slice(paStart, index + 1).map(d => {
    const r = reveals[d.index]; const t = d.pre.recommendation?.candidates[0];
    return { key: String(d.index), label: r ? r.actual.pitch_label : d.index === index ? '?' : '…', result: r ? (r.actual.event_label ?? r.actual.result_label) : d.index === index ? '다음 공' : '',
      match: r && t && r.actual.pitch_type ? t.pitch_type === r.actual.pitch_type : null, current: d.index === index };
  });
  const last = index === timeline.decisions.length - 1;
  return <div className="wa-page wa-has-bar"><Header badge={timeline.badge} back onHelp={() => setGuide(true)} />
    <Board s={s} away={game.away_team} home={game.home_team} pitcher={decision.pitcher.name ?? String(decision.pitcher.id)}
      batter={`${decision.batter.name ?? decision.batter.id} (${decision.batter.side === 'L' ? '좌' : '우'}타)`} extra={`${decision.pitch_number}구째`} />
    <main className="wa-main">
      <nav className="wa-innings" aria-label="회로 이동">{halves.map((h, i) =>
        <button key={h.first} className={i === currentHalf ? 'current' : ''} aria-current={i === currentHalf ? 'true' : undefined} onClick={() => go(h.first)}>{h.label}</button>)}</nav>
      <div className="wa-nav">
        <select value={currentPa} onChange={e => go(plateAppearances[Number(e.target.value)].first)} aria-label="타석으로 이동">
          {plateAppearances.map((pa, i) => <option key={pa.first} value={i}>{pa.label}</option>)}
        </select>
        <span className="wa-progress">{index + 1}/{timeline.decisions.length}구</span>
      </div>
      <PitchStrip items={strip} />
      {error && <p className="wa-error">{error}</p>}
      <article className="wa-card">
        <p className="wa-eyebrow">투구 전 추천</p>
        {decision.pre.status === 'ready' && top ? <Recommendation candidates={rec!.candidates} bounds={timeline.location.zone_bounds} actual={reveal?.actual ?? null} note={timeline.location.note} />
          : <><h2 className="wa-none">추천 없음</h2><p className="wa-reason">{decision.pre.reason}</p><p className="wa-muted wa-small">실제 투구는 아래 버튼으로 그대로 볼 수 있어요.</p></>}
        {reveal && <Revealed reveal={reveal} top={top ?? null} decision={decision} game={game} />}
      </article>
      <footer className="wa-foot">
        <p className="wa-keys">키보드: → 또는 스페이스 = 공개/다음 · ← = 이전</p>
        <details><summary>이 추천은 어떻게 만들었나요?</summary>
          <p>{timeline.policy.note}</p>
          <p>2025년까지 기록으로 학습해 고정한 추천 모델(연구 이름 ARM-B)이 구종을 고르고, 위치는 그 투수가 실제로 던진 공의 분포로 어림해요.</p>
          <p>이 경기에서 추천을 낸 공 {timeline.coverage.ready}/{timeline.coverage.pitch_decisions} · {timeline.we_note}</p>
          <p className="wa-tech">{timeline.policy.name} · τ {timeline.policy.tau} · 식별자 {timeline.policy.identity_sha256.slice(0, 8)}</p></details></footer>
    </main>
    <div className="wa-actionbar" role="toolbar" aria-label="투구 넘기기">
      <button className="wa-step" onClick={() => step(-1)} disabled={index === 0} aria-label="이전 공">←</button>
      {!reveal ? <button className="wa-primary" onClick={() => void doReveal()} disabled={busy}>{busy ? '불러오는 중…' : '실제 투구 공개'}</button>
        : last ? <a className="wa-primary" href="/watch">마지막 공이에요 · 목록으로</a>
        : <button className="wa-primary" onClick={() => step(1)}>다음 공 →</button>}
      <button className="wa-step" onClick={() => step(1)} disabled={last} aria-label="공개하지 않고 다음 공">→</button>
    </div>
    {guide && <Guide onClose={() => setGuide(false)} />}
  </div>;
}

function Revealed({ reveal, top, decision, game }: { reveal: Reveal; top: Candidate | null; decision: Decision; game: GameItem }) {
  const a = reveal.actual, we = reveal.we;
  const battingHome = decision.situation.half === 'Bot';
  const before = we.home_before === null ? null : battingHome ? we.home_before : 1 - we.home_before;
  const after = we.home_after === null ? null : battingHome ? we.home_after : 1 - we.home_after;
  return <section className="wa-reveal">
    <p className="wa-eyebrow">실제 투구</p>
    <h3>{a.pitch_label}{a.speed_mph !== null && <span> · {a.speed_mph.toFixed(1)} mph</span>}</h3>
    <p className="wa-result">{a.event_label ?? a.result_label} · {plainZone(a.zone_label)}</p>
    {top && a.pitch_type && <p className={`wa-match ${top.pitch_type === a.pitch_type ? 'same' : ''}`}>{top.pitch_type === a.pitch_type ? '추천 1순위 구종과 같았어요' : `추천 1순위(${top.pitch_label})와 다른 구종`}</p>}
    {a.play_text && <p className="wa-play">{a.play_text}</p>}
    <WeCard team={battingHome ? game.home_team : game.away_team} before={before} after={after} />
  </section>;
}

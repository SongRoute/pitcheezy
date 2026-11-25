import { useEffect, useId, useState } from 'react';
import { short } from './WatchAlong';
import './watch-summary.css';

/** `pitcheezy-watch-summary-v1` (apps/observer/CONTRACT.md): revealed-prefix line score, home WE curve, notable plays. */
type Half = 'Top' | 'Bot';
type Cell = { runs: number | null; status: 'complete' | 'in_progress' | 'not_reached' | 'not_played' | 'unknown' };
type Teams = { away: number | null; home: number | null };
type CurvePoint = { index: number; inning: number; half: Half; at_bat_number: number | null; pitch_number: number | null;
  home_before: number | null; home_after: number | null };
type Play = { index: number; at_bat_number: number | null; pitch_number: number | null; inning: number; half: Half;
  pitcher: { id: number | null; name: string | null }; batter: { id: number | null; name: string | null };
  event_label: string | null; result_label: string | null; play_text: string | null; runs_scored: Teams; score_after: Teams;
  we: { kind: 'observed_model_change'; home_before: number | null; home_after: number | null; home_delta: number | null; batting_delta: number | null } };
export type WatchSummary = { schema: 'pitcheezy-watch-summary-v1'; game_pk: number; through: number; revealed_pitches: number; complete: boolean;
  teams: { away: string | null; home: string | null };
  score: Teams & { source: 'next_pre_pitch_situation' | 'final_record' | 'pre_pitch_situation' | 'unavailable'; after_index: number | null };
  innings: { inning: number; away: Cell; home: Cell }[]; we_curve: CurvePoint[]; events: Play[]; largest_we_changes: Play[];
  notes: { we: string; we_model: string | null; attribution: null }; issues: string[] };
type Props = { gamePk: number; through: number };
type Result = { key: string; data?: WatchSummary; error?: string };

const TOP = 5;
const REGULATION = 9;
const W = 320, H = 116, LEFT = 30, RIGHT = 8, TOP_Y = 8, BOTTOM_Y = 92;
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const pct = (v: number | null) => finite(v) ? `${Math.round(v * 100)}%` : '—';
/** Signed percentage points of the displayed model's home WE, one decimal (0 stays 0.0). */
const pp = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v * 100).toFixed(1)}%p`;
const half = (p: { inning: number; half: Half }) => `${p.inning}회${p.half === 'Top' ? '초' : '말'}`;
const team = (name: string | null, fallback: string) => name ? short(name) : fallback;

async function fetchSummary(gamePk: number, through: number, signal: AbortSignal): Promise<WatchSummary> {
  let response: Response;
  try { response = await fetch(`/api/watch/${gamePk}/summary?through=${through}&top=${TOP}`, { cache: 'no-store', signal }); }
  catch (e) { if (signal.aborted) throw e; throw new Error('관전 서버에 연결하지 못했어요. 네트워크(Tailscale) 연결을 확인해 주세요.'); }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : '경기 요약을 불러오지 못했어요.');
  // A response for another game or another revealed point is never shown (it could carry later pitches).
  if (body?.schema !== 'pitcheezy-watch-summary-v1' || body.game_pk !== gamePk || body.through !== through
      || !Array.isArray(body.innings) || !Array.isArray(body.we_curve) || !Array.isArray(body.largest_we_changes) || !body.score) {
    throw new Error('경기 요약 자료 형식을 확인할 수 없어요.');
  }
  return body as WatchSummary;
}

/** Each pitch's pre-pitch home WE, then the last pitch's after value; null where the dataset has none. */
function series(curve: CurvePoint[]) {
  const values = curve.map(p => finite(p.home_before) ? p.home_before : null);
  if (curve.length) { const last = curve[curve.length - 1].home_after; values.push(finite(last) ? last : null); }
  return values;
}

/** Runs of consecutive known values; a missing value always breaks the line (never interpolated). */
function segments(values: (number | null)[]) {
  const out: { x: number; v: number }[][] = [];
  let run: { x: number; v: number }[] = [];
  values.forEach((v, x) => { if (v === null) { if (run.length) out.push(run); run = []; } else run.push({ x, v }); });
  if (run.length) out.push(run);
  return out;
}

function LineScore({ data }: { data: WatchSummary }) {
  const columns = Math.max(REGULATION, data.innings.length);
  const rows = [['away', team(data.teams.away, '원정')], ['home', team(data.teams.home, '홈')]] as const;
  let marks = false;
  const cell = (c: Cell | undefined) => {
    if (!c || c.status === 'not_reached') return { text: '', label: '진행 전', cls: 'later' };
    if (c.status === 'not_played') { marks = true; return { text: 'X', label: '경기 종료로 진행하지 않음', cls: 'x' }; }
    if (c.status === 'unknown' || c.runs === null) { marks = true; return { text: '?', label: '득점 기록 없음', cls: 'unknown' }; }
    return { text: String(c.runs), label: `${c.runs}점${c.status === 'in_progress' ? ' (진행 중)' : ''}`, cls: c.status === 'in_progress' ? 'now' : '' };
  };
  const body = rows.map(([side, name]) => <tr key={side}>
    <th scope="row">{name}</th>
    {Array.from({ length: columns }, (_, i) => {
      const c = cell(data.innings[i]?.[side]);
      return <td key={i} className={c.cls} aria-label={`${i + 1}회 ${c.label}`}>{c.text}</td>;
    })}
    <td className="total" aria-label={`합계 ${data.score[side] ?? '기록 없음'}`}>{data.score[side] ?? '—'}</td>
  </tr>);
  const caption = data.score.source === 'final_record' ? '최종 점수'
    : data.score.source === 'unavailable' ? '점수 기록 없음'
    : data.through < 0 ? '경기 시작 전' : `${data.revealed_pitches}구 공개 후`;
  return <>
    <div className="ws-table" role="region" aria-label="이닝별 점수 (가로로 밀어 보기)" tabIndex={0}>
      <table>
        <caption className="ws-sr">이닝별 점수, {caption}</caption>
        <thead><tr><th scope="col"><span className="ws-sr">팀</span></th>
          {Array.from({ length: columns }, (_, i) => <th key={i} scope="col">{i + 1}</th>)}<th scope="col" className="total">R</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    <p className="ws-muted">{caption}{marks ? ' · X 진행하지 않음 · ? 기록 없음' : ''}</p>
  </>;
}

function Sparkline({ data }: { data: WatchSummary }) {
  const values = series(data.we_curve);
  if (!values.length) return <p className="ws-empty">아직 공개된 공이 없어요. 공을 공개하면 승리확률 흐름이 그려져요.</p>;
  const n = Math.max(values.length - 1, 1);
  const x = (i: number) => LEFT + (i / n) * (W - LEFT - RIGHT), y = (v: number) => TOP_Y + (1 - v) * (BOTTOM_Y - TOP_Y);
  const parts = segments(values);
  const missing = values.filter(v => v === null).length;
  const known = values.filter(finite);
  const first = known.length ? known[0] : null, current = values[values.length - 1];
  const home = team(data.teams.home, '홈팀');
  const marks: { i: number; inning: number }[] = [];
  data.we_curve.forEach((p, i) => {
    const prev = data.we_curve[i - 1];
    if (i > 0 && prev.inning !== p.inning) marks.push({ i, inning: p.inning });
  });
  let lastLabel = -Infinity;
  const label = `${home} 승리확률 흐름, 공개된 ${data.we_curve.length}구. 처음 ${pct(first)}, 지금 ${pct(current)}`
    + (missing ? `. ${missing}개 지점은 자료가 없어 선을 끊었습니다` : '');
  return <figure className="ws-spark">
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {[1, .5, 0].map(v => <g key={v}><line x1={LEFT} x2={W - RIGHT} y1={y(v)} y2={y(v)} className={v === .5 ? 'mid' : 'edge'} />
        <text x={LEFT - 4} y={y(v) + 3} textAnchor="end">{v * 100}%</text></g>)}
      {marks.map(m => {
        const show = x(m.i) - lastLabel >= 14; if (show) lastLabel = x(m.i);
        return <g key={m.i}><line x1={x(m.i)} x2={x(m.i)} y1={TOP_Y} y2={BOTTOM_Y} className="inning" />
          {show && <text x={x(m.i)} y={H - 10} textAnchor="middle">{m.inning}</text>}</g>;
      })}
      <text x={LEFT} y={H - 10} textAnchor="middle">{data.we_curve[0].inning}</text>
      {parts.map(run => run.length > 1
        ? <polyline key={run[0].x} points={run.map(p => `${x(p.x)},${y(p.v)}`).join(' ')} className="we" />
        : <circle key={run[0].x} cx={x(run[0].x)} cy={y(run[0].v)} r="2" className="we-dot" />)}
      {finite(current) && <circle cx={x(values.length - 1)} cy={y(current)} r="4" className="we-now" />}
    </svg>
    <figcaption className="ws-muted">
      {home} 기준 · 처음 {pct(first)} → 지금 <b>{pct(current)}</b>{missing ? ` · 자료 없는 ${missing}개 지점은 선을 끊음` : ''} · 아래 숫자는 이닝
    </figcaption>
  </figure>;
}

function Moments({ data }: { data: WatchSummary }) {
  const home = team(data.teams.home, '홈팀');
  const plays = data.largest_we_changes.filter(p => finite(p.we.home_delta));
  if (!plays.length) return <p className="ws-empty">{data.through < 0 ? '아직 공개된 공이 없어요.' : '승리확률 변화가 계산된 공이 아직 없어요.'}</p>;
  return <ol className="ws-moments">
    {plays.map(p => {
      const delta = p.we.home_delta as number;
      const runs = (p.half === 'Top' ? p.runs_scored.away : p.runs_scored.home) ?? 0;
      return <li key={p.index}>
        <div className="ws-moment-head">
          <span className="ws-chip">{half(p)}</span>
          <strong>{p.event_label ?? p.result_label ?? '기록 없음'}</strong>
          {runs > 0 && <span className="ws-runs">{runs}점</span>}
          <span className={`ws-delta ${delta > 0 ? 'up' : delta < 0 ? 'down' : ''}`}>{home} {pp(delta)}</span>
        </div>
        <p className="ws-moment-who">{p.batter.name ?? '타자 기록 없음'} vs {p.pitcher.name ?? '투수 기록 없음'} · {pct(p.we.home_before)} → {pct(p.we.home_after)}</p>
        {p.play_text && <p className="ws-moment-text">{p.play_text}</p>}
      </li>;
    })}
  </ol>;
}

export function WatchSummaryPanel({ gamePk, through }: Props) {
  const id = useId();
  const key = `${gamePk}:${through}`;
  const valid = Number.isInteger(gamePk) && gamePk > 0 && Number.isInteger(through) && through >= -1;
  const [result, setResult] = useState<Result | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!valid) return;
    const controller = new AbortController();
    fetchSummary(gamePk, through, controller.signal).then(
      data => { if (!controller.signal.aborted) setResult({ key, data }); },
      (e: Error) => { if (!controller.signal.aborted) setResult({ key, error: e.message }); });
    return () => controller.abort();
  }, [gamePk, through, key, valid, attempt]);
  // Only a result for exactly these props is rendered: stepping back never shows a later summary, even for one frame.
  const current = result?.key === key ? result : null;
  const data = current?.data;
  return <section className="ws-panel" aria-labelledby={`${id}-title`} aria-busy={valid && !current}>
    <header className="ws-head">
      <h2 id={`${id}-title`}>경기 흐름</h2>
      <span className="ws-muted">{!data ? '' : data.complete ? '경기 끝까지 공개' : data.through < 0 ? '공개 전' : `${data.revealed_pitches}구째까지`}</span>
    </header>
    {!valid ? <p className="ws-empty">요약할 투구 위치가 올바르지 않아요.</p>
      : current?.error ? <div className="ws-error" role="alert"><p>{current.error}</p>
        <button type="button" onClick={() => { setResult(null); setAttempt(a => a + 1); }}>다시 시도</button></div>
      : !data ? <p className="ws-loading" role="status">경기 요약을 불러오는 중…</p>
      : <>
        <LineScore data={data} />
        {data.issues.length > 0 && <p className="ws-warn">기록 불일치가 있어 일부 점수는 확인되지 않았어요.</p>}
        <h3>승리확률 흐름</h3>
        <Sparkline data={data} />
        <h3>승리확률이 크게 움직인 공</h3>
        <Moments data={data} />
        <p className="ws-note">%p는 모델이 추정한 승리확률의 공 전후 차이(퍼센트포인트)입니다. {data.notes.we}</p>
      </>}
  </section>;
}

export default WatchSummaryPanel;

import type { ObservedPitchContext, ObservedSpeed } from './types';

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const count = (value: unknown) => finite(value) && Number.isInteger(value) && value >= 0 ? value : null;
const displayCount = (value: unknown) => count(value) ?? '—';

function SpeedRow({ speed }: { speed: ObservedSpeed }) {
  const recent = count(speed.recent_measured_count), past = count(speed.prior90_measured_count);
  const enough = recent !== null && past !== null && recent >= 3 && past >= 30;
  const comparable = enough && finite(speed.recent_mean_mph) && finite(speed.prior90_mean_mph) && finite(speed.delta_mph);
  const lowSample = recent !== null && past !== null && !enough;
  return <div className="speed-row" role="row" data-pitch-type={speed.pitch_type}>
    <div role="cell"><strong>{speed.pitch_label || speed.pitch_type}</strong><small>최근 측정 N={displayCount(speed.recent_measured_count)} / {displayCount(speed.recent_pitch_count)}구</small></div>
    <div role="cell"><span>{finite(speed.recent_mean_mph) ? speed.recent_mean_mph.toFixed(1) : '—'}</span><small>과거 측정 N={displayCount(speed.prior90_measured_count)}</small></div>
    <div role="cell" className="speed-comparison">{comparable ? <span>{speed.delta_mph! > 0 ? '+' : ''}{speed.delta_mph!.toFixed(1)}</span> : <span className="sample-status">{lowSample ? '표본 적음' : '비교 정보 없음'}</span>}</div>
  </div>;
}

export function PitcherContextSummary({ context, notes }: { context?: ObservedPitchContext | null; notes: string[] }) {
  if (!context) return notes?.length ? <div className="context-notes">{notes.map((note, index) => <p key={index}>{note}</p>)}</div> : null;
  const cutoffNotes = (notes || []).filter((note) => /타자|성향|프로필|구종 구성/.test(note) && /전날|전일|이전 날짜|기준일|cutoff|컷오프/.test(note));
  return <div className="observed-summary-card"><div className="observed-summary"><div><span>이 경기에서 이전 투구</span><strong>{displayCount(context.prior_pitch_count)}<small>구</small></strong></div><div><span>이번 경기 동일 타자</span><strong>{displayCount(context.times_facing_batter)}<small>번째 대결</small></strong></div></div>{cutoffNotes.length > 0 && <div className="profile-cutoff-notes">{cutoffNotes.map((note, index) => <p key={index}>{note}</p>)}</div>}</div>;
}

export default function ObservedContext({ context }: { context?: ObservedPitchContext | null }) {
  if (!context) return null;
  const speeds = Array.isArray(context.speed_by_pitch_type) ? [...context.speed_by_pitch_type].filter((speed) => speed && typeof speed.pitch_type === 'string').sort((a, b) => (count(b.recent_measured_count) ?? -1) - (count(a.recent_measured_count) ?? -1) || a.pitch_type.localeCompare(b.pitch_type)) : [];
  const head = speeds.slice(0, 3), rest = speeds.slice(3);
  const referenceDays = count(context.reference_window_days) || 90;
  return <section className="observed-context observed-speed-card" aria-label="현재 투구 전 관측 정보">
    <div className="speed-heading"><h3>구종별 관측 구속</h3><span>mph</span></div>
    <p className="speed-window">현재 투구 전, 같은 구종 최대 5구의 측정 평균입니다.</p>
    {head.length ? <div className="speed-table" role="table" aria-label="구종별 관측 구속"><div className="speed-row speed-table-head" role="row"><span role="columnheader">구종 · 표본</span><span role="columnheader">최근 평균</span><span role="columnheader">이전 {referenceDays}일 대비</span></div>{head.map((speed) => <SpeedRow key={speed.pitch_type} speed={speed} />)}</div> : <p className="speed-empty">비교할 구종별 구속 기록이 없습니다.</p>}
    {rest.length > 0 && <details className="other-speeds"><summary>나머지 {rest.length}개 구종 보기<span aria-hidden="true">＋</span></summary><div className="speed-table" role="table" aria-label="추가 구종의 관측 구속"><div className="speed-row speed-table-head" role="row"><span role="columnheader">구종 · 표본</span><span role="columnheader">최근 평균</span><span role="columnheader">이전 {referenceDays}일 대비</span></div>{rest.map((speed) => <SpeedRow key={speed.pitch_type} speed={speed} />)}</div></details>}
    <p className="speed-sample-note">차이는 최근 측정 3구 이상·경기 전 {referenceDays}일 측정 30구 이상일 때만 표시합니다.</p>
    <p className="condition-note">관측 기록이며 피로·부상·컨디션이나 투수 교체를 판단한 값이 아닙니다.</p>
  </section>;
}

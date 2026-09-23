import { useCallback, useEffect, useRef, useState } from 'react';
import type { Analysis, Bounds, Catalog, EventAnalysis, GameState, Pitch, Recommendation, View, Zone, Zones } from './types';
import CatalogPicker from './CatalogPicker';
import ChoiceExplanation from './ChoiceExplanation';
import ObservedContext, { PitcherContextSummary } from './ObservedContext';
import { bookmarkKey, readBookmarks, rememberSession } from './catalog';

const SESSION_KEY = 'pitcheezy.observer.session.v1';
class ApiError extends Error { constructor(message: string, public status: number) { super(message); } }
async function api<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); }
  catch { throw new ApiError('관전 서버에 연결하지 못했어요. 서버 상태를 확인하고 다시 시도해 주세요.', 0); }
  let result: unknown;
  try { result = await response.json(); } catch { throw new ApiError('서버 응답을 읽지 못했어요. 잠시 후 다시 시도해 주세요.', response.status); }
  if (!response.ok) { const detail = (result as { detail?: unknown })?.detail; throw new ApiError(typeof detail === 'string' ? detail : '요청을 처리하지 못했어요. 다시 시도해 주세요.', response.status); }
  return result as T;
}
const errorText = (error: unknown) => error instanceof Error ? error.message : '요청을 처리하지 못했어요.';
const halfLabel = (state: Pick<GameState, 'inning' | 'half'>) => `${state.inning}회 ${state.half === 'Top' ? '초' : '말'}`;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const signed = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
function actualZoneLabel(pitch: Pitch) {
  const { x, z, zone_bounds: bounds } = pitch;
  if (!finite(x) || !finite(z) || !bounds || bounds.top <= bounds.bottom) return '위치 정보 없음';
  if (Math.abs(x) > .83 || z < bounds.bottom || z > bounds.top) return '존 바깥';
  const column = Math.min(2, Math.max(0, Math.floor((x + .83) / 1.66 * 3)));
  const row = Math.min(2, Math.max(0, Math.floor((z - bounds.bottom) / (bounds.top - bounds.bottom) * 3)));
  return `${['낮은', '가운데 높이', '높은'][row]} ${['왼쪽', '중앙', '오른쪽'][column]}`;
}
function savedSession() { try { return localStorage.getItem(SESSION_KEY); } catch { return null; } }
function saveSession(id: string | null) { try { if (id) localStorage.setItem(SESSION_KEY, id); else localStorage.removeItem(SESSION_KEY); } catch { /* Private browser storage may be unavailable. */ } }

function BrandMark() { return <svg viewBox="0 0 40 40" aria-hidden="true"><path d="M12 30V10h9c10 0 10 13 0 13h-4v7M17 15v3h4c3 0 3-3 0-3z" fill="currentColor" /><circle cx="30" cy="29" r="3" fill="#e57743" /></svg>; }
function Bases({ bases }: { bases: number }) { return <div className="bases" role="img" aria-label={`주자 ${[1, 2, 3].filter((base) => bases & (1 << (base - 1))).join(', ') || '없음'}${bases ? '루' : ''}`}>{[2, 3, 1].map((base) => <span key={base} className={`base base-${base} ${bases & (1 << (base - 1)) ? 'occupied' : ''}`} />)}</div>; }
function CountDots({ label, value, maximum, kind }: { label: string; value: number; maximum: number; kind: string }) { return <div className="count-row"><span>{label}</span><div className="dots" role="img" aria-label={`${kind === 'ball' ? '볼' : kind === 'strike' ? '스트라이크' : '아웃'} ${value}개`}>{Array.from({ length: maximum }, (_, i) => <i key={i} className={`${kind} ${i < value ? 'on' : ''}`} />)}</div><b>{value}</b></div>; }

function StrikeZone({ recommendation, pitch, bounds }: { recommendation: Recommendation | null; pitch: Pitch | null; bounds: Bounds | null }) {
  if (!bounds || !finite(bounds.bottom) || !finite(bounds.top) || bounds.top <= bounds.bottom) return <div className="zone-empty">이 공의 스트라이크존 정보를 불러오지 못했어요.</div>;
  const actual = pitch && finite(pitch.x) && finite(pitch.z) ? { x: pitch.x, z: pitch.z } : null;
  const candidates = (recommendation?.status === 'ready' ? recommendation.candidates : []).slice(0, 3).map((candidate, i) => ({ ...candidate, rank: i + 1 })).filter((c) => finite(c.target.x) && finite(c.target.z));
  const groupedTargets = new Map<string, { x: number; z: number; ranks: number[] }>();
  for (const candidate of candidates) {
    const key = `${candidate.target.x}:${candidate.target.z}`;
    const existing = groupedTargets.get(key);
    if (existing) existing.ranks.push(candidate.rank);
    else groupedTargets.set(key, { ...candidate.target, ranks: [candidate.rank] });
  }
  const points = [...candidates.map((c) => c.target), ...(actual ? [actual] : [])];
  const extent = Math.max(1.65, ...points.map((point) => Math.abs(point.x) + .3));
  const minZ = Math.min(.5, bounds.bottom - .65, ...points.map((point) => point.z - .3));
  const maxZ = Math.max(4.4, bounds.top + .7, ...points.map((point) => point.z + .3));
  const x = (value: number) => 48 + (value + extent) / (2 * extent) * 424;
  const y = (value: number) => 432 - (value - minZ) / (maxZ - minZ) * 380;
  const left = x(-.83), right = x(.83), top = y(bounds.top), bottom = y(bounds.bottom);
  const width = right - left, height = bottom - top;
  return <svg className="zone-svg" viewBox="0 0 520 490" role="img" aria-label={`포수 시점 스트라이크존. ${candidates.length}개 추천 목표${actual ? '와 실제 투구 위치' : ''}.`}>
    <defs><pattern id="zone-dots" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".85" fill="#d7dbd2" /></pattern></defs>
    <rect x="24" y="20" width="472" height="430" rx="18" fill="url(#zone-dots)" />
    <text x="260" y="34" className="svg-caption" textAnchor="middle">포수 시점 · 타자 손에 따라 반전하지 않음</text>
    <rect x={left} y={top} width={width} height={height} rx="3" fill="#fdfcf7" fillOpacity=".85" stroke="#23443e" strokeWidth="2" />
    {[1, 2].map((i) => <g key={i}><line x1={left + width * i / 3} x2={left + width * i / 3} y1={top} y2={bottom} stroke="#cbd1c8" strokeDasharray="4 5" /><line x1={left} x2={right} y1={top + height * i / 3} y2={top + height * i / 3} stroke="#cbd1c8" strokeDasharray="4 5" /></g>)}
    {[...groupedTargets].map(([key, target]) => <g key={key}><circle cx={x(target.x)} cy={y(target.z)} r={target.ranks.includes(1) ? 27 : 22} fill="#17695b" fillOpacity={target.ranks.includes(1) ? '.13' : '.05'} stroke="#17695b" strokeWidth={target.ranks.includes(1) ? 2 : 1} strokeDasharray={target.ranks.includes(1) ? undefined : '3 3'} /><text x={x(target.x)} y={y(target.z) + 5} className="target-number" textAnchor="middle" style={target.ranks.length > 1 ? { fontSize: 12 } : undefined}>{target.ranks.join('·')}</text></g>)}
    {actual && <g><circle cx={x(actual.x)} cy={y(actual.z)} r="11" fill="#e57743" stroke="#fffdf6" strokeWidth="4" /><circle cx={x(actual.x)} cy={y(actual.z)} r="16" fill="none" stroke="#e57743" strokeOpacity=".35" /></g>}
    <path d="M236 456H284V466L260 482L236 466Z" fill="#fdfcf7" stroke="#bac4b9" strokeWidth="1.5" />
    <text x="60" y="474" className="svg-caption">왼쪽</text><text x="460" y="474" className="svg-caption" textAnchor="end">오른쪽</text>
  </svg>;
}

function RecommendationCards({ recommendation, comparing }: { recommendation: Recommendation | null; comparing: boolean }) {
  const ready = recommendation?.status === 'ready' && recommendation.candidates.length > 0;
  const candidates = recommendation?.candidates.slice(0, 3) || [];
  return <section className="recommendations" aria-labelledby="recommendation-title">
    <div className="section-heading"><div><span className="eyebrow">선택의 순간</span><h2 id="recommendation-title">{comparing ? '이 공을 던지기 전 추천' : '다음 공의 첫 번째 제안'}</h2></div><span className="small-tag">모델 제안</span></div>
    {!ready ? <div className="empty-state compact"><span className="empty-symbol">—</span><h3>추천을 제공할 수 없어요</h3><p>{recommendation?.reason || '이 공에 저장된 추천이 없습니다.'}</p><p>추천이 없어도 실제 기록은 계속 확인할 수 있어요.</p></div> : <>
      <div className="lead-choice"><span>추천 구종과 목표</span><strong>{candidates[0].pitch_label || candidates[0].pitch_type}<small>{candidates[0].pitch_type}</small></strong><b>{candidates[0].zone_label}</b><p>포수 시점의 목표 구역입니다.</p></div>
      <details className="candidate-details"><summary>다른 후보와 추천 근거 보기<span aria-hidden="true">＋</span></summary>
        <div className="candidate-list">{candidates.slice(1).map((candidate, i) => <article className="candidate" key={`${candidate.pitch_type}-${candidate.zone_id}-${i}`}><span className="rank">0{i + 2}</span><div className="candidate-main"><div className="candidate-title"><h3>{candidate.pitch_label || candidate.pitch_type}</h3><span>{candidate.pitch_type}</span></div><p>{candidate.zone_label}</p></div></article>)}</div>
        <div className="recommendation-basis"><h3>이 선택을 살펴볼 근거</h3>{recommendation?.basis?.length ? <ul>{recommendation.basis.map((item, i) => <li key={i}>{item}</li>)}</ul> : <p>이 추천에 함께 제공된 상세 근거는 없습니다.</p>}</div>
      </details>
      <p className="recommendation-note">구역은 근사 모델의 제안입니다. 실제 투구 의도나 승률 향상을 확인한 결과는 아니에요.</p>
      {recommendation && <ChoiceExplanation recommendation={recommendation} />}
      <details className="probability-details"><summary>모델 내부 비교값 보기<span aria-hidden="true">＋</span></summary><p>같은 상황에서 구종·구역을 비교하기 위한 모델의 수비팀 승률 추정치입니다. 실제 경기에서 이 선택의 효과가 검증되었다는 뜻은 아닙니다.</p><div className="probability-table" role="table" aria-label="추천별 모델 내부 비교값"><div role="row" className="probability-row probability-head"><span role="columnheader">추천 순위</span><span role="columnheader">모델 수비 승률</span><span role="columnheader">기준 대비</span></div>{candidates.map((candidate, i) => <div role="row" className="probability-row" key={i}><span role="cell">{i + 1} · {candidate.pitch_label || candidate.pitch_type}</span><span role="cell">{finite(candidate.value) ? `${(candidate.value * 100).toFixed(1)}%` : '정보 없음'}</span><span role="cell">{finite(candidate.delta_pp) ? `${signed(candidate.delta_pp)}%p` : '정보 없음'}</span></div>)}</div>{finite(recommendation?.baseline_value) && <p className="probability-baseline">모델의 기준 선택 승률 · {(recommendation!.baseline_value * 100).toFixed(1)}%</p>}</details>
    </>}
  </section>;
}

function TerminalAnalysis({ view, zones, busy, onManual }: { view: View; zones: Zone[]; busy: boolean; onManual: (zone: string) => void }) {
  const analysis: Analysis | null = view.analysis;
  const [selected, setSelected] = useState(analysis?.manual_zone_id || '');
  useEffect(() => { setSelected(analysis?.manual_zone_id || ''); }, [analysis?.manual_zone_id, view.id]);
  const sorted = [...zones].sort((a, b) => b.row - a.row || a.column - b.column);
  return <section className="terminal-panel" aria-labelledby="terminal-title"><div className="section-heading"><div><span className="eyebrow">타석을 돌아보다</span><h2 id="terminal-title">{view.summary?.headline || '타석이 끝났습니다'}</h2></div><span className="result-badge">{view.summary?.result_label || '타석 종료'}</span></div>
    {view.summary && <p className="terminal-intro">{view.summary.pitch_count}개의 공을 확인했어요. 마지막 {view.summary.selected_pitch_number}구를 함께 돌아봅니다.</p>}
    <div className="analysis-layout"><div><div className="no-video"><span aria-hidden="true">▧</span><div><h3>영상 기반 의도 분석은 제공되지 않아요</h3><p>{analysis?.message || '포수의 미트 위치나 투구 의도는 기록만으로 알 수 없습니다.'}</p></div></div>
      {analysis && <p className="selected-pitch">분석 대상 · {analysis.selected_pitch_number}구 / {analysis.selection_reason}</p>}
      {analysis && ['queued', 'running'].includes(analysis.status) && <p className="pending-line" role="status"><span className="spinner" /> 마지막 공의 분석 상태를 확인하고 있어요.</p>}
      {analysis?.status === 'failed' && <p className="inline-warning">분석을 마무리하지 못했어요. 실제 투구 기록은 아래 타임라인에서 확인할 수 있어요.</p>}
      {analysis?.comparisons && <div className="comparison"><div><span>당시 추천 구역</span><b>{analysis.comparisons.recommended_zone_label || '정보 없음'}</b></div><div><span>직접 표시한 목표</span><b>{analysis.comparisons.intended_zone_label || '정보 없음'}</b></div><div><span>실제 도착 구역</span><b>{analysis.comparisons.actual_zone_label || '정보 없음'}</b></div><p>{analysis.comparisons.interpretation}</p></div>}
      {analysis?.narrative?.map((line, i) => <p className="analysis-narrative" key={i}>{line}</p>)}
      {view.summary?.notes?.map((line, i) => <p className="muted" key={i}>{line}</p>)}
    </div><div className="manual-zone"><h3>생각한 목표를 직접 표시해 보세요</h3><p>타석의 마지막 공에 대한 내 메모입니다.<br />영상에서 확인한 사실이나 다음 공 추천이 아니에요.</p><div className="zone-buttons" role="group" aria-label="마지막 공의 의도한 목표 구역, 포수 시점">{sorted.map((zone) => <button type="button" key={zone.id} aria-pressed={selected === zone.id} className={selected === zone.id ? 'selected' : ''} disabled={busy} onClick={() => setSelected(zone.id)}>{zone.label}</button>)}</div>{!zones.length && <p className="inline-warning">목표 구역 정보를 불러오지 못했어요. 화면 상단에서 다시 연결해 주세요.</p>}<button className="button secondary full-width" disabled={!selected || busy || !zones.length} onClick={() => onManual(selected)}>{busy ? '저장하는 중…' : analysis?.manual_zone_id ? '목표 메모 수정' : '목표 메모 저장'}</button>{analysis?.source === 'manual' && <p className="saved-note">직접 입력한 목표가 저장되어 있어요.</p>}</div></div>
  </section>;
}

function EventCard({ event }: { event: EventAnalysis | null | undefined }) {
  if (!event) return null;
  const developmentOnly = event.evidence?.development_only;
  const statusLabel = { complete: '모델 계산 완료', partial: '일부만 계산 가능', unavailable: '계산 불가', failed: '계산 실패' }[event.status];
  const reasonLabel: Record<string, string> = { missing_intent: '투구 전 의도 근거 없음', missing_compatible_total: '같은 모델에서 비교 가능한 전후 값 없음', missing_compatible_values: '비교에 필요한 값 부족', calculation_error: '모델 계산 오류' };
  const sourceLabel: Record<string, string> = { historical_replay_record: '과거 경기 기록', observed_post_pa_frozen_we: '고정 승률 모델', synthetic_contract_example: '개발용 합성 사례' };
  const displayNumber = (value: number | null | undefined) => finite(value) ? `${signed(value)} %p` : '계산 불가';
  const components = [
    ['작전 선택 대비', event.components?.strategy_contrast_pp],
    ['의도와 실행 대비', event.components?.execution_contrast_pp],
    ['결과 잔차', event.components?.outcome_residual_pp],
  ] as const;
  return <section className="event-card" aria-labelledby="event-title"><div className="section-heading"><div><span className="eyebrow">중요 사건 · 수비팀 승률 모델</span><h2 id="event-title">이번 타석의 변화</h2></div><span className={`event-status ${event.status}`}>{statusLabel}</span></div>
    {developmentOnly ? <p className="event-empty">개발용 합성 사례입니다. 실제 경기의 기여 수치로 표시하지 않습니다.</p> : <>
      <p className="event-total">{finite(event.values?.total_pp) ? <><strong>{displayNumber(event.values.total_pp)}</strong><span>이 타석 전 기준과 종료 후 수비팀 승률 추정치의 차이</span></> : <span>이 사건의 승률 변화를 계산할 수 없습니다.</span>}</p>
      {event.status === 'partial' && <p className="event-callout">투구 의도 정보가 없어 선택·실행의 몫은 분리할 수 없습니다. 남은 차이를 선수에게 배분하지 않습니다.</p>}
      {event.status === 'failed' && <p className="event-callout">분석 계산이 실패했습니다. 기록과 저장된 사전 추천은 계속 볼 수 있습니다.</p>}
      {(event.status === 'complete' || event.status === 'partial') && <div className="event-components">{components.map(([label, component]) => <div key={label}><span>{label}</span><strong>{displayNumber(component?.value_pp)}</strong>{finite(component?.abs_share) && event.shares?.stable && <small>계산된 절댓값 중 {(component.abs_share * 100).toFixed(0)}%</small>}</div>)}</div>}
      {finite(event.components?.unallocated_residual_pp) && <p className="event-residual">배분하지 않은 차이 · {displayNumber(event.components.unallocated_residual_pp)}</p>}
      <p className="event-evidence">근거 · 실제 결과 {sourceLabel[event.evidence?.actual_source] || '출처 정보 없음'} / 투구 의도 {event.evidence?.intent_source ? '영상 입력 근거 있음' : '확인되지 않음'}</p>
      {event.reason && <p className="event-reason">계산 범위 · {reasonLabel[event.reason] || '자세한 계산 사유는 모델 안내에서 확인할 수 있습니다.'}</p>}
      <p className="event-caveat">결과 잔차와 비중은 모델의 기술적 분해이며 선수의 인과적 책임이나 실제 추천 효과가 아닙니다. 교체 판단은 별도 이닝 종료 분석이 필요합니다.</p>
    </>}
  </section>;
}

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [zones, setZones] = useState<Zone[]>([]);
  const [view, setView] = useState<View | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [bookmarks, setBookmarks] = useState(readBookmarks);
  const [chartMode, setChartMode] = useState<'next' | 'actual'>('next');
  const [selectedPitchId, setSelectedPitchId] = useState<string | number | null>(null);
  const [runtime, setRuntime] = useState<unknown>(null);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [reload, setReload] = useState(0);
  const sessionRef = useRef<string | null>(null);
  const acceptView = useCallback((next: View) => { sessionRef.current = next.id; setView((previous) => previous?.id === next.id && previous.revision > next.revision ? previous : next); saveSession(next.id); setBookmarks(rememberSession(next.game.id, next.plate_appearance.id, next.id)); }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const saved = savedSession();
    void Promise.allSettled([api<Catalog>('/catalog'), api<Zones>('/zones'), saved ? api<View>(`/sessions/${encodeURIComponent(saved)}`) : Promise.resolve(null)]).then(([catalogResult, zoneResult, sessionResult]) => {
      if (cancelled) return;
      const errors: string[] = [];
      if (catalogResult.status === 'fulfilled') { setCatalog(catalogResult.value); } else errors.push(errorText(catalogResult.reason));
      if (zoneResult.status === 'fulfilled') setZones(zoneResult.value.zones); else errors.push(errorText(zoneResult.reason));
      if (sessionResult.status === 'fulfilled' && sessionResult.value) { const resumed = sessionResult.value; acceptView(resumed); setChartMode(resumed.last_pitch ? 'actual' : 'next'); }
      else if (sessionResult.status === 'rejected') { if (sessionResult.reason instanceof ApiError && sessionResult.reason.status === 404) { saveSession(null); errors.push('이전 관전 기록을 찾지 못했어요. 타석을 선택해 다시 시작해 주세요.'); } else errors.push(errorText(sessionResult.reason)); }
      setError(errors.length ? [...new Set(errors)].join(' ') : null); setLoading(false);
    });
    return () => { cancelled = true; };
  }, [acceptView, reload]);

  useEffect(() => {
    if (!view || !view.analysis || !['queued', 'running'].includes(view.analysis.status)) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const id = view.id;
    const poll = async () => {
      if (busyRef.current) { timer = setTimeout(poll, 2000); return; }
      try { const next = await api<View>(`/sessions/${encodeURIComponent(id)}`); if (!cancelled && sessionRef.current === id) acceptView(next); }
      catch (problem) { if (!cancelled) setError(errorText(problem)); }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    timer = setTimeout(poll, 2000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [view?.id, view?.analysis?.status, acceptView]);

  async function mutate(kind: 'start' | 'advance' | 'manual', zone?: string, selection?: { game_id: number; pa_id: number }) {
    if (busyRef.current || (kind !== 'start' && !view) || (kind === 'start' && !selection)) return;
    busyRef.current = true; setBusy(true); setError(null);
    try {
      const path = kind === 'start' ? '/sessions' : `/sessions/${encodeURIComponent(view!.id)}/${kind === 'advance' ? 'advance' : 'manual-intent'}`;
      const body = kind === 'start' ? selection : kind === 'advance' ? { revision: view!.revision } : { revision: view!.revision, zone_id: zone };
      const next = await api<View>(path, body); acceptView(next);
      if (kind !== 'manual') { setChartMode(kind === 'start' && !next.last_pitch ? 'next' : 'actual'); setSelectedPitchId(null); }
    } catch (problem) {
      setError(errorText(problem));
      if (problem instanceof ApiError && problem.status === 409 && view) {
        try { acceptView(await api<View>(`/sessions/${encodeURIComponent(view.id)}`)); setError('다른 요청에서 기록이 갱신되어 최신 상태를 불러왔어요. 확인 후 다시 눌러 주세요.'); } catch (refreshError) { setError(errorText(refreshError)); }
      }
    } finally { busyRef.current = false; setBusy(false); }
  }

  async function resume(id: string): Promise<boolean> {
    if (busyRef.current) return false;
    busyRef.current = true; setBusy(true); setError(null);
    try {
      const restored = await api<View>(`/sessions/${encodeURIComponent(id)}`);
      acceptView(restored); setSelectedPitchId(null); setChartMode(restored.last_pitch ? 'actual' : 'next');
      return true;
    } catch (problem) { setError(errorText(problem)); return false; }
    finally { busyRef.current = false; setBusy(false); }
  }

  function openAdjacent(paId: number) {
    if (!view) return;
    const previousSession = bookmarks[bookmarkKey(view.game.id, paId)];
    if (previousSession) void resume(previousSession);
    else void mutate('start', undefined, { game_id: view.game.id, pa_id: paId });
  }

  async function retry() {
    if (!view || !catalog || !zones.length) { setReload((n) => n + 1); return; }
    if (busyRef.current) return;
    busyRef.current = true; setBusy(true);
    try { acceptView(await api<View>(`/sessions/${encodeURIComponent(view.id)}`)); setError(null); } catch (problem) { setError(errorText(problem)); } finally { busyRef.current = false; setBusy(false); }
  }

  const currentGame = catalog?.games.find((game) => game.id === view?.game.id);
  const gamePAs = currentGame?.plate_appearances || [];
  const currentPAIndex = gamePAs.findIndex((pa) => pa.id === view?.plate_appearance.id);
  const previousPA = currentPAIndex > 0 ? gamePAs[currentPAIndex - 1] : null;
  const nextPA = currentPAIndex >= 0 ? gamePAs[currentPAIndex + 1] : null;
  const selectedPitch = view?.history.find((pitch) => pitch.id === selectedPitchId) || view?.last_pitch || null;
  const comparing = !!view && (chartMode === 'actual' || view.complete) && !!selectedPitch;
  const recommendation = comparing ? selectedPitch!.recommendation : view?.recommendation || null;
  const bounds = comparing ? selectedPitch!.zone_bounds : recommendation?.zone_bounds || null;
  const isLastSelected = selectedPitch?.id === view?.last_pitch?.id;
  const savedTop = comparing && recommendation?.status === 'ready' ? recommendation.candidates[0] : null;

  return <><header className="site-header"><a className="brand" href="/" aria-label="피치이지 처음 화면"><span className="brand-icon"><BrandMark /></span><span>pitcheezy<span className="brand-dot">.</span></span></a><div className="header-right"><a className="header-lab-link" href="/video-lab" title="실험용 영상 검토실 · 경기 결과가 포함될 수 있습니다">영상 검토실<small>실험용</small></a><span className="replay-tag"><i />기록으로 보는 야구</span><span className="header-caption">한 공 더 깊게</span></div></header>
    <main className={view ? "observing" : "welcome-mode"}><section className="intro"><div><span className="eyebrow">한 공씩 읽는 야구</span><h1>다음 한 공을,<br className="mobile-break" /> 함께 생각하다.</h1><p>기록을 한 공씩 열어 보며, 그 순간의 선택을 비교해 보세요.</p></div><div className="intro-number" aria-hidden="true">01<span>한 공의 가능성</span></div></section>
      <CatalogPicker catalog={catalog} view={view} loading={loading} busy={busy} bookmarks={bookmarks} onStart={(gameId, paId) => void mutate('start', undefined, { game_id: gameId, pa_id: paId })} onResume={resume} />
      {error && <div className="error-banner" role="alert"><div><strong>잠시 확인해 주세요</strong><p>{error}</p></div><button className="button text-button" disabled={busy || loading} onClick={() => void retry()}>다시 연결</button></div>}
      {loading && !view ? <div className="initial-loading" role="status"><span className="spinner" /><h2>관전 노트를 준비하고 있어요</h2><p>공개된 경기와 저장한 타석을 확인합니다.</p></div> : !view ? <section className="welcome empty-state"><div className="welcome-diamond" aria-hidden="true"><span /></div><span className="eyebrow">아직 열지 않은 한 공</span><h2>{catalog?.games.length ? '타석을 선택하면, 이야기가 시작됩니다.' : '관전할 경기 기록이 아직 없어요.'}</h2><p>{catalog?.games.length ? '추천을 먼저 살펴보고 실제 공을 한 개씩 확인하세요. 결과는 직접 열어 보기 전까지 표시되지 않습니다.' : '경기 목록이 준비되면 위에서 타석을 선택할 수 있습니다.'}</p><div className="welcome-steps"><span>01 · 상황 읽기</span><span>02 · 추천 살펴보기</span><span>03 · 실제 공과 비교</span></div></section> : <>
        <section className="scoreboard" aria-label="현재 경기 상황"><div className="game-meta"><span className="eyebrow">과거 경기 기록</span><span>{view.game.date}</span><span className="inning-label">{halfLabel(view.state)}{view.complete ? ' · 타석 종료' : ''}</span></div><div className="team-score"><div><span>원정</span><b>{view.game.away_team}</b></div><strong>{view.state.away_score}</strong><i>:</i><strong>{view.state.home_score}</strong><div><span>홈</span><b>{view.game.home_team}</b></div></div><div className="situation"><Bases bases={view.state.bases} /><div className="counts"><CountDots label="B" value={view.state.balls} maximum={3} kind="ball" /><CountDots label="S" value={view.state.strikes} maximum={2} kind="strike" /><CountDots label="O" value={view.state.outs} maximum={2} kind="out" /></div></div></section>
        <nav className="pa-navigation" aria-label="같은 경기의 타석 이동"><button className="pa-nav-button" disabled={busy || !previousPA} onClick={() => { if (previousPA) openAdjacent(previousPA.id); }}><span aria-hidden="true">←</span><span>이전 타석<small>{previousPA && bookmarks[bookmarkKey(view.game.id, previousPA.id)] ? '이어보기' : '새 관전'}</small></span></button><p>{currentPAIndex >= 0 ? `${currentPAIndex + 1} / ${gamePAs.length} 타석` : '현재 타석'}<span>{view.plate_appearance.batter_label}</span></p><button className="pa-nav-button" disabled={busy || !nextPA} onClick={() => { if (nextPA) openAdjacent(nextPA.id); }}><span>다음 타석<small>{nextPA && bookmarks[bookmarkKey(view.game.id, nextPA.id)] ? '이어보기' : '새 관전'}</small></span><span aria-hidden="true">→</span></button></nav>
        <section className={`advance-panel ${view.complete ? 'completed' : 'sticky-controls'}`}><div><span className="eyebrow">{view.complete ? '이번 타석의 기록' : '이제, 실제 공을 볼까요?'}</span><h2>{view.complete ? '타석의 모든 공을 확인했어요.' : view.history.length ? `${view.history.length}개의 공을 확인했어요.` : '아직 공개된 공이 없어요.'}</h2><p>{view.complete ? '아래 타임라인에서 각 공을 다시 선택할 수 있습니다.' : '한 번 누를 때마다 실제 투구 한 개가 공개됩니다.'}</p></div><button className="button primary advance-button" disabled={busy || view.complete} onClick={() => void mutate('advance')}>{busy ? <><span className="spinner light" />공을 확인하는 중…</> : view.complete ? '타석 종료' : <>다음 실제 공 확인<span aria-hidden="true">→</span></>}</button></section>
        <div className="workspace"><section className="zone-panel" aria-labelledby="zone-title"><div className="section-heading"><div><span className="eyebrow">공이 향하는 곳</span><h2 id="zone-title">{comparing ? `${selectedPitch!.pitch_number}구, 추천과 실제` : '다음 공의 추천 구역'}</h2></div><span className="zone-view-tag">포수 시점</span></div><div className="chart-tabs" role="group" aria-label="투구 비교 시점"><button aria-pressed={!comparing} disabled={view.complete} className={!comparing ? 'active' : ''} onClick={() => { setChartMode('next'); setSelectedPitchId(null); }}>다음 공 추천</button><button aria-pressed={comparing} disabled={!view.last_pitch} className={comparing ? 'active' : ''} onClick={() => { setChartMode('actual'); setSelectedPitchId(null); }}>{comparing && !isLastSelected ? '선택한 공 비교' : '방금 던진 공'}</button></div>
          <StrikeZone recommendation={recommendation} pitch={comparing ? selectedPitch : null} bounds={bounds} />
          <div className="zone-legend"><span><i className="legend-target">1</i>추천 목표 · 순위</span><span><i className="legend-actual" />실제 투구</span></div>
          {comparing ? <div className="actual-strip"><div><span className="eyebrow">실제 {selectedPitch!.pitch_number}구</span><strong>{selectedPitch!.pitch_label || selectedPitch!.pitch_type || '구종 정보 없음'} <small>{finite(selectedPitch!.speed_mph) ? `${selectedPitch!.speed_mph!.toFixed(1)} mph` : '구속 정보 없음'}</small></strong></div><span className="actual-result">{selectedPitch!.result_label}</span></div> : <div className="next-strip"><span className="small-dot" /> 아직 공개하지 않은 공입니다. 추천을 살펴본 뒤 확인해 보세요.</div>}
          {comparing && <div className="same-pitch-comparison" aria-label="이 공의 사전 추천과 실제 기록 비교"><div><span>이 공의 사전 추천</span><strong>{savedTop ? `${savedTop.pitch_label || savedTop.pitch_type} · ${savedTop.zone_label}` : '추천 정보 없음'}</strong></div><span aria-hidden="true">→</span><div><span>실제 기록</span><strong>{selectedPitch!.pitch_label || selectedPitch!.pitch_type || '구종 정보 없음'} · {actualZoneLabel(selectedPitch!)}</strong></div></div>}
          {comparing && (!finite(selectedPitch!.x) || !finite(selectedPitch!.z)) && <p className="inline-warning">실제 투구 위치 기록이 없어 위치 점은 표시하지 않습니다.</p>}
          <p className="chart-caption">{comparing ? `추천은 이 공을 던지기 전 ${selectedPitch!.pre_state.balls}볼 ${selectedPitch!.pre_state.strikes}스트라이크 상황에 저장된 내용입니다.` : '목표 구역과 실제 투구 위치는 다를 수 있어요.'}</p>
        </section><aside className="right-column"><section className="matchup" aria-label="투수와 타자"><span className="eyebrow">마운드와 타석</span><div className="matchup-player"><div className="player-symbol" aria-hidden="true">P</div><div><span>투수</span><h2>{view.plate_appearance.pitcher_label}</h2></div></div><div className="batter-row"><span>상대 타자</span><strong>{view.plate_appearance.batter_label}</strong>{view.plate_appearance.batter_stand && <span className="hand-badge">{view.plate_appearance.batter_stand === 'L' ? '좌타' : view.plate_appearance.batter_stand === 'R' ? '우타' : view.plate_appearance.batter_stand}</span>}</div><PitcherContextSummary context={view.context} notes={view.context_notes || []} /></section><RecommendationCards recommendation={recommendation} comparing={comparing} /><ObservedContext context={view.context} /></aside></div>

        <section className="timeline-panel" aria-labelledby="timeline-title"><div className="section-heading"><div><span className="eyebrow">되짚어 보는 투구</span><h2 id="timeline-title">이 타석의 타임라인</h2></div><span className="small-tag">공개된 {view.history.length}개</span></div>{!view.history.length ? <p className="timeline-empty">실제 공을 확인하면 이곳에 기록이 쌓입니다.</p> : <div className="pitch-timeline">{view.history.map((pitch) => <button className={`timeline-pitch ${comparing && selectedPitch?.id === pitch.id ? 'selected' : ''}`} key={pitch.id} aria-pressed={comparing && selectedPitch?.id === pitch.id} onClick={() => { setSelectedPitchId(pitch.id); setChartMode('actual'); }}><span className="pitch-index">{pitch.pitch_number}<small>구</small></span><span className="pitch-name">{pitch.pitch_label || pitch.pitch_type || '구종 미상'}</span><span className="pitch-count">투구 전 {pitch.pre_state.balls}–{pitch.pre_state.strikes}</span><span className="pitch-result">{pitch.result_label}</span></button>)}</div>}</section>
        {view.complete && <><EventCard event={view.event_analysis} /><TerminalAnalysis view={view} zones={zones} busy={busy} onManual={(zone) => void mutate('manual', zone)} /></>}
        {!!view.notices?.length && <div className="notices">{view.notices.map((notice, i) => <p key={i}>{notice}</p>)}</div>}
      </>}
      <details className="model-details"><summary>모델과 데이터 안내<span aria-hidden="true">＋</span></summary><div className="details-content"><p>과거 경기 기록을 한 공씩 재생하는 관전 도구입니다. 실시간 중계가 아니며, 공개하지 않은 실제 투구 결과는 화면에 표시하지 않습니다.</p><p>표시된 승률은 모델 추정치입니다. 추천을 따랐을 때의 실제 성과나 투구 의도를 입증하지 않습니다.</p>{catalog?.model_version && <p>모델 버전 · {catalog.model_version}</p>}{catalog?.limitations?.map((item, i) => <p key={i}>{item}</p>)}<button className="button text-button" disabled={runtimeLoading} onClick={async () => { setRuntimeLoading(true); try { setRuntime(await api('/runtime')); } catch (problem) { setError(errorText(problem)); } finally { setRuntimeLoading(false); } }}>{runtimeLoading ? '확인 중…' : '로컬 실행 정보 확인'}</button>{runtime !== null && <pre>{JSON.stringify(runtime, null, 2)}</pre>}</div></details>
    </main><footer><a className="footer-brand" href="/">pitcheezy.</a><p>야구를 보는 또 하나의 시선.</p><span>과거 기록 재생 · 로컬 관전 노트</span></footer><div className="sr-only" aria-live="polite">{busy ? '요청을 처리하고 있습니다.' : view ? `공개된 투구 ${view.history.length}개. ${view.complete ? '타석이 종료되었습니다.' : '다음 공을 확인할 수 있습니다.'}` : ''}</div>
  </>;
}

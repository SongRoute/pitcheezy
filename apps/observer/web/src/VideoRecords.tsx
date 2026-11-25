import { useState } from 'react';
import type { VideoAnnotation, VideoClip } from './videoAnnotation';

export interface TracePoint {
  time: number; accepted: boolean; confidence: number | null; x: number | null; y: number | null;
  width: number; height: number; reason: string | null; frame_difference: number | null;
}
export interface TrackingResult {
  status: string; abstain_reason: string | null; processed_frames: number; accepted_frames: number;
  matched_frames: number; trace: TracePoint[]; annotation_id?: string; run_id?: string;
  coordinate_frame: string; pixel_target: { x: number; y: number } | null;
  limitations?: string[];
}
export interface SavedVideoAnnotation { id: string; annotation: VideoAnnotation; result: TrackingResult | null }
export async function videoApi<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api/video-lab${path}`, { method, ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) }); }
  catch { throw new Error('로컬 영상 서버에 연결하지 못했습니다. 저장된 라벨은 서버에 유지됩니다.'); }
  let value: unknown;
  try { value = await response.json(); } catch { throw new Error('영상 서버 응답을 읽지 못했습니다. 다시 시도해 주세요.'); }
  if (!response.ok) { const detail = (value as { detail?: unknown })?.detail; throw new Error(typeof detail === 'string' ? detail : '요청을 처리하지 못했습니다. 저장된 라벨은 유지됩니다.'); }
  return value as T;
}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
export const sourceLabel = (value: string) => value === 'user_manual' ? '사용자 수동 표시' : value === 'assistant_visual_estimate' ? '어시스턴트 육안 추정' : value;
const dateLabel = (value: string) => Number.isNaN(Date.parse(value)) ? value : new Date(value).toLocaleString('ko-KR', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

export function SavedVideoLibrary({ records, clips, activeId, busy, loading, error, onLoad, onRefresh }: {
  records: SavedVideoAnnotation[]; clips: VideoClip[]; activeId: string | null; busy: boolean; loading: boolean;
  error: string | null; onLoad: (id: string) => void; onRefresh: () => void;
}) {
  const [selection, setSelection] = useState('');
  const selected = records.some((record) => record.id === selection) ? selection : activeId || records[0]?.id || '';
  return <section className="saved-video-library" aria-labelledby="saved-video-title"><div><span className="eyebrow">로컬 라벨 보관함</span><h2 id="saved-video-title">저장한 표시를 다시 살펴보세요</h2><p>수정해서 저장하면 새 버전이 추가되며 이전 저장본은 남습니다.</p></div><div className="saved-library-controls"><label htmlFor="saved-video-label">저장된 라벨<select id="saved-video-label" aria-label="저장된 라벨" value={selected} disabled={busy || loading || !records.length} onChange={(event) => setSelection(event.target.value)}><option value="" disabled>{loading ? '저장본 확인 중…' : '저장한 라벨이 없습니다'}</option>{records.map((record) => <option key={record.id} value={record.id}>{clips.find((clip) => clip.id === record.annotation.clip_id)?.title || record.annotation.clip_id} · v{record.annotation.annotation_version} · {sourceLabel(record.annotation.label_source)} · {dateLabel(record.annotation.annotated_at)} · {record.id.slice(0, 8)}</option>)}</select></label><button className="button secondary" disabled={busy || loading || !selected} onClick={() => onLoad(selected)}>선택한 저장본 불러오기</button><button className="button text-button" disabled={busy || loading} onClick={onRefresh}>목록 새로고침</button></div>{error && <p className="saved-library-error" role="alert">{error}</p>}</section>;
}

const reasons: Record<string, string> = {
  insufficient_seed_texture: '시작 영역의 무늬가 부족해 매칭을 시작하지 못했습니다.',
  shot_cut_or_high_frame_difference: '화면 전환 또는 큰 영상 변화로 추적을 멈췄습니다.',
  low_match_confidence_or_occlusion: '템플릿 유사도가 낮거나 가림이 의심되어 추적을 멈췄습니다.',
  insufficient_tracking_frames: '추적을 이어갈 프레임이 충분하지 않습니다.',
  frame_dimensions_or_format_changed: '영상 크기 또는 형식이 달라 추적을 멈췄습니다.',
  frame_limit: '허용한 최대 프레임 수에 도달해 추적을 멈췄습니다.',
};
const reasonLabel = (reason: string | null) => reason ? reasons[reason] || reason : '서버가 상세 이유를 제공하지 않았습니다.';

export function VideoTrackingPanel({ saved, dirty, busy, error, onTrack }: {
  saved: SavedVideoAnnotation | null; dirty: boolean; busy: string | null; error: string | null; onTrack: () => void;
}) {
  const result = saved?.result;
  const trace = result?.trace || [];
  const matched = trace.filter((point) => point.accepted && point.reason !== 'supplied_seed' && finite(point.x) && finite(point.y));
  const last = matched[matched.length - 1];
  return <section className="video-tracking-panel" aria-labelledby="tracking-title"><div className="tracking-heading"><div><span className="eyebrow">05 · 저장본에만 실행</span><h2 id="tracking-title">이미지 안의 추적 관측</h2><p>수동 시작 영역과 비슷한 무늬를 찾는 실험입니다. 미트 검출이나 투구 의도를 확인하는 절차가 아닙니다.</p></div><button className="button primary" disabled={!saved || dirty || !!busy} onClick={onTrack}>{busy === 'tracking' ? <><span className="spinner light" />저장본 추적 중…</> : result ? '저장한 추적 결과 확인' : '저장한 라벨로 추적 실행'}</button></div>
    {!saved ? <p className="tracking-guidance">라벨을 먼저 로컬 저장한 뒤, 추적 버튼을 눌러 주세요.</p> : <div className="saved-provenance"><span>저장본 v{saved.annotation.annotation_version} · {saved.id.slice(0, 12)}</span><span>{sourceLabel(saved.annotation.label_source)} · {saved.annotation.review_status === 'unreviewed' ? '검토 전 (unreviewed)' : '저장된 검토 상태: ' + saved.annotation.review_status}</span></div>}
    {dirty && saved && <p className="tracking-dirty" role="status">입력이 저장본과 다릅니다. 새 버전으로 저장한 뒤 추적할 수 있어요. 아래 결과는 수정 전 저장본의 결과입니다.</p>}
    {error && <div className="tracking-error" role="alert"><strong>추적 요청을 마치지 못했습니다.</strong><p>{error}</p><p>저장한 라벨은 유지됩니다. 상태를 확인한 뒤 추적 버튼으로 다시 요청할 수 있습니다.</p></div>}
    {busy === 'tracking' && <p className="tracking-pending" role="status">요청한 구간만 확인 중입니다. 결과가 돌아오기 전에는 추적 성공으로 표시하지 않습니다.</p>}
    {result && <div className={`tracking-result ${result.status === 'abstained' ? 'abstained' : ''}`}><div className="tracking-status"><strong>{result.status === 'abstained' ? '추적 보류' : result.status === 'tracked' ? '지정 구간의 이미지 매칭 완료' : '추적 상태: ' + result.status}</strong><span>{result.processed_frames}프레임 처리 · 시작점을 제외한 매칭 {result.matched_frames}개</span></div>{result.status === 'abstained' && <p className="abstain-reason">{reasonLabel(result.abstain_reason)}</p>}
      <div className="last-match"><h3>보류 전 또는 구간 내 마지막 매칭</h3>{last ? <p>{last.time.toFixed(3)}초 · 이미지 중심 x {last.x!.toFixed(1)}, y {last.y!.toFixed(1)} px</p> : <p>수동 시작점 이후 확인된 매칭이 없습니다.</p>}<small>이 위치를 실제 포수 목표나 제구 결과로 해석하지 않습니다. 수동 시작점은 자동 매칭 결과에 포함하지 않습니다.</small></div>
      <details className="tracking-trace"><summary>프레임별 기록 {trace.length}개 보기<span aria-hidden="true">＋</span></summary><div className="trace-table-wrap"><table><caption>이미지 매칭 기록 · 유사도는 검출 정확도나 성공 확률이 아닙니다.</caption><thead><tr><th scope="col">시각</th><th scope="col">처리</th><th scope="col">템플릿 유사도</th><th scope="col">이미지 중심 (px)</th></tr></thead><tbody>{trace.map((point, index) => <tr key={index}><td>{point.time.toFixed(3)}초</td><td>{point.reason === 'supplied_seed' ? '수동 시작점' : point.accepted ? '매칭 수용' : '보류'}</td><td>{point.reason === 'supplied_seed' ? '—' : finite(point.confidence) ? point.confidence.toFixed(3) : '정보 없음'}</td><td>{finite(point.x) && finite(point.y) ? `${point.x.toFixed(1)}, ${point.y.toFixed(1)}` : '—'}</td></tr>)}</tbody></table></div></details>
      <p className="tracking-scope">실제 길이·깊이·포수 시점 좌표가 아닙니다. 관전 추천이나 선수의 제구·피로 평가에 연결하지 않습니다.</p>
    </div>}
  </section>;
}

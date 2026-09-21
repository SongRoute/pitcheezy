export interface VideoClip {
  id: string; title: string; source_url: string; clip_sha256: string;
  duration_seconds: number | null; width: number | null; height: number | null; fps: number | null;
  video_url: string; usable_for_tracking: boolean; reason: string | null;
}
export interface Point { x: number; y: number }
export interface ROI { x: number; y: number; w: number; h: number }
export type LabelSource = 'user_manual' | 'assistant_visual_estimate';
export interface VideoDraft {
  seed_time: number | null; end_time: number | null; release_time: number | null;
  roi: ROI | null; calibration_corners: Point[] | null; label_source: LabelSource | '';
}
export interface VideoAnnotation extends Omit<VideoDraft, 'label_source' | 'seed_time' | 'end_time' | 'release_time' | 'roi'> {
  schema_version: 1; annotation_version: number; clip_id: string; clip_sha256: string;
  source_url: string; pitch_id: string | null; seed_time: number; end_time: number; release_time: number;
  roi: ROI; image_dimensions: { width: number; height: number }; label_source: LabelSource;
  review_status: 'unreviewed' | 'user_reviewed'; annotated_at: string;
}
export const emptyVideoDraft = (): VideoDraft => ({ seed_time: null, end_time: null, release_time: null, roi: null, calibration_corners: null, label_source: '' });
export const rounded = (value: number) => Math.round(value * 1e6) / 1e6;

export function validCorners(corners: Point[]): boolean {
  if (corners.length !== 4) return false;
  const turns = corners.every((a, index) => {
    const b = corners[(index + 1) % 4], c = corners[(index + 2) % 4];
    return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) > 1e-6 && Math.hypot(b.x - a.x, b.y - a.y) >= 2;
  });
  const area = Math.abs(corners.reduce((sum, a, index) => { const b = corners[(index + 1) % 4]; return sum + a.x * b.y - a.y * b.x; }, 0)) / 2;
  return turns && area >= 16 && corners[0].y + corners[1].y < corners[2].y + corners[3].y && corners[0].x + corners[3].x < corners[1].x + corners[2].x;
}

export function annotationIssues(clip: VideoClip, draft: VideoDraft, dimensions: { width: number; height: number }, duration: number): string[] {
  const issues: string[] = [];
  if (!clip.usable_for_tracking) issues.push(clip.reason || '이 영상은 추적용 라벨을 만들 수 없습니다.');
  if (!/^[a-f0-9]{64}$/.test(clip.clip_sha256 || '')) issues.push('영상 식별값을 확인할 수 없습니다.');
  if (!(dimensions.width > 0 && dimensions.height > 0 && duration > 0)) issues.push('영상의 원본 크기와 길이를 먼저 불러와 주세요.');
  if (!(clip.fps && Number.isFinite(clip.fps) && clip.fps > 0)) issues.push('FPS 정보가 없어 프레임 간격을 정할 수 없습니다.');
  const times = [draft.seed_time, draft.end_time, draft.release_time];
  if (times.some((time) => time === null || !Number.isFinite(time))) issues.push('시작·종료·릴리스 시각을 모두 표시해 주세요.');
  else {
    const [seed, end, release] = times as number[];
    if (!(0 <= seed && seed < end && end < release && release < duration)) issues.push('시작 < 종료 < 릴리스 순서이며, 모두 영상 길이 안에 있어야 합니다.');
    if (end - seed > 3 + 1e-6) issues.push('추적 구간은 최대 3초입니다. 종료 시각을 앞당겨 주세요.');
    if (clip.fps && Math.floor((end - seed) * clip.fps + 1e-6) + 1 > 600) issues.push('추적 구간은 최대 600프레임입니다.');
  }
  const roi = draft.roi;
  if (!roi || !Object.values(roi).every(Number.isFinite) || roi.x < 0 || roi.y < 0 || roi.w < 4 || roi.h < 4 || roi.x + roi.w > dimensions.width || roi.y + roi.h > dimensions.height) issues.push('원본 영상 안에 최소 4 × 4픽셀인 추적 영역을 표시해 주세요.');
  if (draft.calibration_corners && (!validCorners(draft.calibration_corners) || draft.calibration_corners.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.y < 0 || p.x > dimensions.width || p.y > dimensions.height))) issues.push('보정점은 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 순서의 볼록한 사각형이어야 합니다.');
  if (!['user_manual', 'assistant_visual_estimate'].includes(draft.label_source)) issues.push('누가 영상을 보고 표시했는지 선택해 주세요.');
  return issues;
}

export function annotationPayload(clip: VideoClip, draft: VideoDraft, image_dimensions: { width: number; height: number }, duration: number): VideoAnnotation {
  const issues = annotationIssues(clip, draft, image_dimensions, duration);
  if (issues.length) throw new Error(issues[0]);
  return {
    schema_version: 1, annotation_version: 1, clip_id: clip.id, clip_sha256: clip.clip_sha256,
    source_url: clip.source_url, pitch_id: null,
    seed_time: draft.seed_time!, end_time: draft.end_time!, release_time: draft.release_time!,
    roi: draft.roi!, image_dimensions, calibration_corners: draft.calibration_corners,
    label_source: draft.label_source as LabelSource, review_status: 'unreviewed', annotated_at: new Date().toISOString(),
  };
}

export function draftFromAnnotation(annotation: VideoAnnotation): VideoDraft {
  return { seed_time: annotation.seed_time, end_time: annotation.end_time, release_time: annotation.release_time,
    roi: annotation.roi, calibration_corners: annotation.calibration_corners, label_source: annotation.label_source };
}

import { useEffect, useRef, useState } from 'react';
import type { PointerEvent } from 'react';
import { annotationIssues, annotationPayload, draftFromAnnotation, emptyVideoDraft, rounded, validCorners } from './videoAnnotation';
import type { LabelSource, Point, ROI, VideoAnnotation, VideoClip, VideoDraft } from './videoAnnotation';
import { SavedVideoLibrary, VideoTrackingPanel, videoApi } from './VideoRecords';
import type { SavedVideoAnnotation } from './VideoRecords';
import './video-lab.css';

type Mode = 'inspect' | 'roi' | 'calibration';
const cornerLabels = ['왼쪽 위', '오른쪽 위', '오른쪽 아래', '왼쪽 아래'];
const formatTime = (value: number | null) => value === null ? '미지정' : `${value.toFixed(3)}초`;
const draftKey = (draft: VideoDraft) => JSON.stringify([draft.seed_time, draft.end_time, draft.release_time,
  draft.roi ? [draft.roi.x, draft.roi.y, draft.roi.w, draft.roi.h] : null,
  draft.calibration_corners?.map((point) => [point.x, point.y]) || null, draft.label_source]);
function localVideoUrl(value: string) {
  try { const url = new URL(value, window.location.origin); return url.origin === window.location.origin && url.pathname.startsWith('/api/video-lab/clips/') ? url.href : null; }
  catch { return null; }
}
function sourceUrl(value: string) { try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; } }

export default function VideoLab() {
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [limitations, setLimitations] = useState<string[]>([]);
  const [clipId, setClipId] = useState('');
  const [loading, setLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [drafts, setDrafts] = useState<Record<string, VideoDraft>>({});
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [seekInput, setSeekInput] = useState('0');
  const [seeking, setSeeking] = useState(false);
  const [mode, setMode] = useState<Mode>('inspect');
  const [anchor, setAnchor] = useState<Point | null>(null);
  const [corners, setCorners] = useState<Point[]>([]);
  const [note, setNote] = useState('');
  const [downloaded, setDownloaded] = useState(false);
  const [records, setRecords] = useState<SavedVideoAnnotation[]>([]);
  const [bases, setBases] = useState<Record<string, SavedVideoAnnotation>>({});
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [trackingError, setTrackingError] = useState<string | null>(null);
  const [labBusy, setLabBusy] = useState<string | null>(null);
  const requestBusy = useRef(false);
  const payloadCache = useRef<{ key: string; payload: VideoAnnotation } | null>(null);
  const pendingSeek = useRef<number | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const clip = clips.find((item) => item.id === clipId) || null;
  const draft = drafts[clipId] || emptyVideoDraft();
  const fps = clip?.fps && Number.isFinite(clip.fps) && clip.fps > 0 ? clip.fps : null;
  const usable = !!clip?.usable_for_tracking && dimensions.width > 0 && !!fps && !mediaError && !labBusy;
  const videoUrl = clip ? localVideoUrl(clip.video_url) : null;
  const issues = clip ? annotationIssues(clip, draft, dimensions, duration) : [];
  const saved = bases[clipId] || null;
  const dirty = !!saved && draftKey(draft) !== draftKey(draftFromAnnotation(saved.annotation));

  useEffect(() => { void refreshRecords(); }, []);
  useEffect(() => {
    if (dimensions.width && pendingSeek.current !== null) {
      const time = pendingSeek.current;
      pendingSeek.current = null;
      seekTo(time);
    }
  }, [dimensions.width]);

  async function refreshRecords() {
    setRecordsLoading(true); setRecordsError(null);
    try {
      const response = await videoApi<{ annotations: SavedVideoAnnotation[] }>('/annotations');
      setRecords((previous) => {
        const merged = new Map(previous.map((record) => [record.id, record]));
        for (const record of response.annotations) merged.set(record.id, record.result || !merged.get(record.id)?.result ? record : merged.get(record.id)!);
        return [...merged.values()].sort((a, b) => b.annotation.annotated_at.localeCompare(a.annotation.annotated_at));
      });
    } catch (error) { setRecordsError(error instanceof Error ? error.message : '저장본을 불러오지 못했습니다.'); }
    finally { setRecordsLoading(false); }
  }

  function remember(record: SavedVideoAnnotation) {
    setRecords((previous) => [record, ...previous.filter((item) => item.id !== record.id)]);
    setBases((previous) => ({ ...previous, [record.annotation.clip_id]: record }));
  }

  function currentPayload() {
    if (!clip) throw new Error('영상을 먼저 선택해 주세요.');
    if (saved && !dirty) return saved.annotation;
    const key = JSON.stringify([clip.id, draftKey(draft), dimensions.width, dimensions.height, saved?.id]);
    if (payloadCache.current?.key === key) return payloadCache.current.payload;
    const payload = annotationPayload(clip, draft, dimensions, duration);
    payload.annotation_version = saved ? saved.annotation.annotation_version + 1 : 1;
    payloadCache.current = { key, payload };
    return payload;
  }

  async function loadRecord(id: string) {
    if (requestBusy.current) return;
    requestBusy.current = true; setLabBusy('loading'); setRecordsError(null);
    try {
      const record = await videoApi<SavedVideoAnnotation>(`/annotations/${encodeURIComponent(id)}`);
      const annotation = record.annotation;
      const registered = clips.find((item) => item.id === annotation.clip_id);
      if (!registered || registered.clip_sha256 !== annotation.clip_sha256) throw new Error('저장본과 일치하는 등록 영상을 찾을 수 없습니다. 기존 입력은 유지됩니다.');
      if (registered.width !== annotation.image_dimensions.width || registered.height !== annotation.image_dimensions.height) throw new Error('저장본의 원본 크기가 등록 영상과 다릅니다.');
      setDrafts((previous) => ({ ...previous, [annotation.clip_id]: draftFromAnnotation(annotation) }));
      remember(record); setMode('inspect'); setAnchor(null); setCorners([]); setDownloaded(false); setSaveError(null); setTrackingError(null);
      if (annotation.clip_id === clipId && video.current && dimensions.width) seekTo(annotation.seed_time);
      else pendingSeek.current = annotation.seed_time;
      setClipId(annotation.clip_id); setNote('저장본을 불러왔습니다. 출처와 저장된 검토 상태를 그대로 유지합니다.');
    } catch (error) { setRecordsError(error instanceof Error ? error.message : '저장본을 불러오지 못했습니다.'); }
    finally { requestBusy.current = false; setLabBusy(null); }
  }

  async function saveRecord() {
    if (requestBusy.current || issues.length || !usable || mode !== 'inspect' || seeking) return;
    requestBusy.current = true; setLabBusy('saving'); setSaveError(null);
    try {
      const record = await videoApi<SavedVideoAnnotation>('/annotations', 'POST', currentPayload());
      remember(record); setTrackingError(null); setNote('로컬 서버에 라벨을 저장했습니다. 추적은 별도 버튼을 눌러야 실행됩니다.');
    } catch (error) { setSaveError(error instanceof Error ? error.message : '라벨을 저장하지 못했습니다. 현재 입력은 유지됩니다.'); }
    finally { requestBusy.current = false; setLabBusy(null); }
  }

  async function trackRecord() {
    if (requestBusy.current || !saved || dirty) return;
    requestBusy.current = true; setLabBusy('tracking'); setTrackingError(null);
    try {
      const record = await videoApi<SavedVideoAnnotation>(`/annotations/${encodeURIComponent(saved.id)}/track`, 'POST');
      remember(record); setNote('저장본의 추적 응답을 받았습니다. 아래에서 보류 여부와 프레임 기록을 확인해 주세요.');
    } catch (error) { setTrackingError(error instanceof Error ? error.message : '추적을 요청하지 못했습니다. 라벨은 저장되어 있습니다.'); }
    finally { requestBusy.current = false; setLabBusy(null); }
  }

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setCatalogError(null);
    void (async () => {
      try {
        const response = await fetch('/api/video-lab/catalog', { signal: controller.signal });
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '영상 목록을 불러오지 못했습니다.');
        if (!Array.isArray(body.clips)) throw new Error('영상 목록의 형식을 확인할 수 없습니다.');
        if (!controller.signal.aborted) { setClips(body.clips); setLimitations(body.limitations || []); setClipId((previous) => body.clips.some((item: VideoClip) => item.id === previous) ? previous : body.clips[0]?.id || ''); }
      } catch (error) { if (!controller.signal.aborted) setCatalogError(error instanceof Error ? error.message : '영상 서버에 연결하지 못했습니다.'); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  }, [reload]);

  useEffect(() => {
    setDimensions({ width: 0, height: 0 }); setDuration(0); setCurrentTime(0); setSeekInput('0');
    setMediaError(null); setMode('inspect'); setAnchor(null); setCorners([]); setNote(''); setDownloaded(false); setSeeking(false); setSaveError(null); setTrackingError(null);
  }, [clipId]);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const ctx = element.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    // These are seed-frame labels, not tracked boxes on later video frames.
    if (mode === 'inspect' && draft.seed_time !== null && fps && Math.abs(currentTime - draft.seed_time) > 1 / fps) return;
    ctx.lineWidth = Math.max(2, dimensions.width / 220);
    ctx.font = `bold ${Math.max(16, dimensions.width / 38)}px sans-serif`;
    if (draft.roi) {
      ctx.strokeStyle = '#59e7b6'; ctx.fillStyle = '#59e7b61a';
      ctx.fillRect(draft.roi.x, draft.roi.y, draft.roi.w, draft.roi.h);
      ctx.strokeRect(draft.roi.x, draft.roi.y, draft.roi.w, draft.roi.h);
    }
    const points = mode === 'calibration' ? corners : draft.calibration_corners || [];
    if (points.length) {
      ctx.beginPath(); ctx.strokeStyle = '#ffbc67';
      points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
      if (points.length === 4) ctx.closePath();
      ctx.stroke();
    }
    [...points, ...(anchor ? [anchor] : [])].forEach((point, index) => {
      const radius = Math.max(5, dimensions.width / 100);
      ctx.beginPath(); ctx.fillStyle = anchor && index === points.length ? '#59e7b6' : '#ffbc67';
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#17342b'; ctx.lineWidth = Math.max(2, dimensions.width / 450);
      const label = String(index + 1), x = point.x + radius * 1.5, y = point.y - radius;
      ctx.strokeText(label, x, y); ctx.fillText(label, x, y);
    });
  }, [dimensions, draft.roi, draft.calibration_corners, draft.seed_time, anchor, corners, mode, currentTime, fps]);

  function update(patch: Partial<VideoDraft>) {
    setDrafts((previous) => ({ ...previous, [clipId]: { ...(previous[clipId] || emptyVideoDraft()), ...patch } }));
    setDownloaded(false); setSaveError(null); setTrackingError(null);
  }
  function seekTo(time: number, keepMode = false) {
    const element = video.current;
    if (!element || !Number.isFinite(time) || !duration) return;
    element.pause();
    const maximum = fps ? Math.max(0, duration - 1 / fps) : duration;
    const target = Math.min(maximum, Math.max(0, time));
    if (Math.abs(element.currentTime - target) > .000001) { setSeeking(true); element.currentTime = target; }
    setCurrentTime(target); setSeekInput(target.toFixed(3));
    if (!keepMode) { setMode('inspect'); setAnchor(null); setCorners([]); }
  }
  function mark(field: 'seed_time' | 'end_time' | 'release_time') {
    if (!video.current || !fps || seeking) return;
    video.current.pause(); setMode('inspect'); setAnchor(null); setCorners([]);
    const time = rounded(Math.min(Math.max(0, duration - 1 / fps), Math.max(0, Math.round(video.current.currentTime * fps) / fps)));
    if (field === 'seed_time') { update({ seed_time: time, roi: null, calibration_corners: null }); setNote('시작 프레임을 표시했습니다. 이 프레임에서 추적 영역을 새로 지정해 주세요.'); }
    else { update({ [field]: time }); setNote(`${field === 'end_time' ? '추적 종료' : '릴리스'} 시각을 표시했습니다.`); }
  }
  function begin(next: 'roi' | 'calibration') {
    if (draft.seed_time === null) { setNote('먼저 추적 시작 시각을 표시해 주세요.'); return; }
    seekTo(draft.seed_time, true); setMode(next); setAnchor(null); setCorners([]);
    setNote(next === 'roi' ? '시작 프레임에서 추적할 영역의 두 모서리를 차례로 눌러 주세요.' : '시작 프레임에서 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래를 눌러 주세요.');
  }
  function pointAt(event: PointerEvent<HTMLCanvasElement>) {
    if (mode === 'inspect' || seeking || !usable) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const point = { x: rounded(Math.min(dimensions.width, Math.max(0, (event.clientX - bounds.left) / bounds.width * dimensions.width))), y: rounded(Math.min(dimensions.height, Math.max(0, (event.clientY - bounds.top) / bounds.height * dimensions.height))) };
    if (mode === 'roi') {
      if (!anchor) { setAnchor(point); return; }
      const roi = { x: Math.min(anchor.x, point.x), y: Math.min(anchor.y, point.y), w: rounded(Math.abs(point.x - anchor.x)), h: rounded(Math.abs(point.y - anchor.y)) };
      if (roi.w < 4 || roi.h < 4) { setNote('원본 영상에서 최소 4 × 4픽셀인 영역을 만들어 주세요.'); return; }
      update({ roi }); setAnchor(null); setMode('inspect'); setNote('시작 프레임의 추적 영역을 표시했습니다.');
    } else {
      const next = [...corners, point]; setCorners(next);
      if (next.length === 4) {
        if (!validCorners(next)) { setNote('순서가 맞지 않거나 사각형이 겹칩니다. 왼쪽 위부터 다시 표시해 주세요.'); setCorners([]); return; }
        update({ calibration_corners: next }); setMode('inspect'); setCorners([]); setNote('이미지 안의 구역 보정점을 저장했습니다. 실제 길이 단위나 포수 시점은 아닙니다.');
      }
    }
  }
  function download() {
    if (!clip || issues.length || mode !== 'inspect' || seeking) return;
    try {
      const payload = currentPayload();
      const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' }));
      const link = document.createElement('a'); link.href = url;
      link.download = `video-label-${clip.id.replace(/[^a-zA-Z0-9_-]/g, '_')}-v${payload.annotation_version}.json`;
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 1500);
      setDownloaded(true); setNote('JSON 파일을 브라우저에서 내려받았습니다. 다운로드만으로 로컬 서버에 저장되거나 추적이 실행되지는 않습니다.');
    } catch (error) { setNote(error instanceof Error ? error.message : 'JSON을 만들지 못했습니다.'); }
  }

  return <><header className="site-header video-lab-header"><a className="brand" href="/"><img src="/favicon.svg" width="34" height="34" alt="" />pitcheezy.</a><a className="lab-back-link" href="/">← 관전으로 돌아가기</a></header>
    <main className="video-lab"><section className="lab-intro"><span className="eyebrow">실험용 · 수동 영상 라벨링</span><h1>영상 검토실</h1><p>한 프레임을 멈추고, 추적할 영역과 시각을 직접 표시합니다.</p><div className="lab-disclosure">영상에는 경기 결과가 포함될 수 있습니다. 관전 화면의 아직 공개하지 않은 공과는 분리된 실험 공간입니다.</div></section>
      {loading ? <div className="initial-loading" role="status"><span className="spinner" /><p>로컬 영상 목록을 확인하고 있어요.</p></div> : catalogError ? <div className="error-banner" role="alert"><p>{catalogError}</p><button className="button secondary" onClick={() => setReload((value) => value + 1)}>영상 목록 다시 불러오기</button></div> : !clips.length ? <div className="empty-state welcome"><h2>검토할 영상이 아직 없습니다.</h2><p>등록된 로컬 영상이 준비되면 여기에서 선택할 수 있습니다.</p><button className="button secondary" onClick={() => setReload((value) => value + 1)}>영상 목록 다시 확인</button></div> : clip && <>
        <section className="lab-clip-picker"><label htmlFor="video-clip">검토할 영상<select aria-label="검토할 영상" id="video-clip" value={clipId} onChange={(event) => setClipId(event.target.value)}>{clips.map((item) => <option key={item.id} value={item.id}>{item.title}{item.usable_for_tracking ? '' : ' · 추적 불가'}</option>)}</select></label><div><span className={`lab-status ${clip.usable_for_tracking ? '' : 'unavailable'}`}>{clip.usable_for_tracking ? '수동 라벨 작성 가능' : '추적용 라벨 작성 불가'}</span>{sourceUrl(clip.source_url) && <a href={sourceUrl(clip.source_url)!} target="_blank" rel="noreferrer">영상 출처 ↗</a>}</div></section>
        {!clip.usable_for_tracking && <div className="lab-unavailable" role="status"><strong>이 영상은 추적 실험에 사용할 수 없어요.</strong><p>{clip.reason || '추적할 실제 투구 장면을 확인할 수 없습니다.'}</p></div>}
        <div className="lab-workspace"><section className="lab-viewer" aria-label="영상과 프레임 검토"><div className="lab-player-top"><h2>{clip.title}</h2><span>{dimensions.width ? `${dimensions.width} × ${dimensions.height}px` : '원본 크기 확인 중'}{fps ? ` · ${fps} FPS` : ''}</span></div>
          {mediaError || !videoUrl ? <div className="lab-media-error" role="alert"><h3>영상을 재생할 수 없습니다.</h3><p>{mediaError || '등록된 로컬 영상 주소가 아닙니다.'}</p><button className="button secondary" onClick={() => { setMediaError(null); video.current?.load(); }}>영상 다시 열기</button></div> : <div className={`lab-video-stage ${mode !== 'inspect' ? 'drawing' : ''}`}><video key={clip.id} ref={video} src={videoUrl} controls={mode === 'inspect'} preload="metadata" playsInline onLoadedMetadata={(event) => { const element = event.currentTarget; setDimensions({ width: element.videoWidth, height: element.videoHeight }); setDuration(Number.isFinite(element.duration) ? element.duration : 0); }} onTimeUpdate={(event) => { setCurrentTime(event.currentTarget.currentTime); setSeekInput(event.currentTarget.currentTime.toFixed(3)); }} onSeeking={() => setSeeking(true)} onSeeked={(event) => { setSeeking(false); setCurrentTime(event.currentTarget.currentTime); setSeekInput(event.currentTarget.currentTime.toFixed(3)); }} onError={() => setMediaError('파일을 읽지 못했거나 브라우저가 이 영상 형식을 지원하지 않습니다.')} aria-label="원본 영상" /><canvas ref={canvas} width={Math.max(1, dimensions.width)} height={Math.max(1, dimensions.height)} className={mode === 'inspect' ? 'passive' : ''} onPointerDown={pointAt} aria-label={mode === 'roi' ? '추적 영역의 두 모서리 선택' : '이미지 구역의 보정점 선택'} /></div>}
          <div className="frame-controls"><div className="frame-time"><span>현재 시각</span><strong>{formatTime(currentTime)}</strong></div><div className="frame-buttons"><button className="button secondary" disabled={!dimensions.width || !fps || seeking} onClick={() => seekTo((video.current?.currentTime || 0) - 1 / fps!)}>← 1프레임</button><button className="button secondary" disabled={!dimensions.width || !fps || seeking} onClick={() => seekTo((video.current?.currentTime || 0) + 1 / fps!)}>1프레임 →</button></div></div><form className="seek-form" onSubmit={(event) => { event.preventDefault(); seekTo(Number(seekInput)); }}><label htmlFor="video-seek">시각 이동 (초)<input id="video-seek" type="number" step="any" min="0" max={duration || undefined} value={seekInput} onChange={(event) => setSeekInput(event.target.value)} /></label><button className="button secondary" disabled={!dimensions.width || seeking || !seekInput || !Number.isFinite(Number(seekInput))}>이 시각으로 이동</button></form>
          <p className="lab-small-note">프레임 이동은 영상의 FPS 기준 1/{fps || '?'}초 간격입니다. 자동 추적이나 의도 판정은 실행하지 않습니다.</p>
          {mode !== 'inspect' && <div className="drawing-instruction" role="status"><p>{mode === 'roi' ? anchor ? '반대쪽 모서리를 눌러 주세요.' : '첫 번째 모서리를 눌러 주세요.' : `${corners.length + 1}/4 · ${cornerLabels[corners.length]} 모서리를 눌러 주세요.`}{seeking && ' 시작 프레임으로 이동 중…'}</p><button className="button text-button" onClick={() => { setMode('inspect'); setAnchor(null); setCorners([]); setNote('이번 점 선택을 취소했습니다. 기존 라벨은 유지됩니다.'); }}>점 선택 취소</button></div>}
        </section><aside className="lab-editor"><section className="lab-card"><div className="section-heading"><div><span className="eyebrow">01 · 시간 표시</span><h2>추적할 구간을 정하세요</h2></div></div><p className="lab-help">원하는 프레임에서 각각 표시하세요. 시작 &lt; 종료 &lt; 릴리스, 추적 구간은 최대 3초·600프레임입니다.</p><div className="time-markers">{(['seed_time', 'end_time', 'release_time'] as const).map((field, index) => <div key={field}><button className="button secondary" disabled={!usable || seeking} onClick={() => mark(field)}>{['추적 시작 표시', '추적 종료 표시', '릴리스 표시'][index]}</button><output>{formatTime(draft[field])}</output></div>)}</div></section>
          <section className="lab-card"><span className="eyebrow">02 · 시작 프레임의 영역</span><h2>추적할 물체를 감싸세요</h2><p className="lab-help">두 모서리를 눌러 사각형을 만듭니다. 저장 좌표는 화면 크기가 아닌 원본 영상의 픽셀입니다.</p><div className="lab-button-row"><button className="button secondary" disabled={!usable || seeking || draft.seed_time === null} onClick={() => begin('roi')}>{draft.roi ? '추적 영역 다시 표시' : '추적 영역 표시'}</button><button className="button text-button" disabled={!draft.roi} onClick={() => { update({ roi: null }); setMode('inspect'); setAnchor(null); }}>영역 지우기</button></div>{draft.roi && <p className="roi-readout">x {draft.roi.x.toFixed(1)} · y {draft.roi.y.toFixed(1)} · 폭 {draft.roi.w.toFixed(1)} · 높이 {draft.roi.h.toFixed(1)} px</p>}<details className="lab-coordinate-details"><summary>영역 좌표를 숫자로 입력</summary><div className="roi-inputs">{(['x', 'y', 'w', 'h'] as const).map((field, index) => <label key={field}>{['왼쪽 x', '위쪽 y', '폭', '높이'][index]}<input type="number" min="0" step="any" disabled={!usable || draft.seed_time === null} value={draft.roi?.[field] ?? ''} onChange={(event) => { const next: ROI = { x: 0, y: 0, w: 0, h: 0, ...draft.roi, [field]: Number(event.target.value) }; update({ roi: next }); }} /></label>)}</div></details></section>
          <section className="lab-card"><span className="eyebrow">03 · 선택 사항</span><h2>이미지 구역의 네 모서리</h2><p className="lab-help">왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래. 이미지 안의 상대 구역을 정할 때만 사용하세요.</p><div className="lab-button-row"><button className="button secondary" disabled={!usable || seeking || draft.seed_time === null} onClick={() => begin('calibration')}>{draft.calibration_corners ? '보정점 다시 표시' : '네 모서리 표시'}</button><button className="button text-button" disabled={!draft.calibration_corners} onClick={() => { update({ calibration_corners: null }); setMode('inspect'); setCorners([]); }}>보정점 지우기</button></div><p className="coordinate-caution">좌표계: annotated_image_zone · 포수 시점이 보장되지 않으며, 피트(ft) 등 실제 길이로 변환하지 않습니다.</p>{draft.calibration_corners && <ol className="corner-readout">{draft.calibration_corners.map((point, index) => <li key={index}>{cornerLabels[index]} · {point.x.toFixed(1)}, {point.y.toFixed(1)} px</li>)}</ol>}</section>
        </aside></div>
        <section className="lab-export"><div><span className="eyebrow">04 · 검토 전 수동 라벨</span><h2>표시한 내용을 JSON으로 저장</h2><p>이 표시는 추정값이며 검토 상태는 항상 <b>unreviewed</b>입니다. 투구 ID는 연결하지 않고, 클립 ID로 로컬 영상과 연결합니다.</p><label htmlFor="label-source">표시한 사람 / 라벨 출처<select aria-label="표시한 사람 / 라벨 출처" id="label-source" value={draft.label_source} disabled={!usable} onChange={(event) => update({ label_source: event.target.value as LabelSource | '' })}><option value="">라벨 출처를 선택하세요</option><option value="user_manual">사용자가 직접 보고 표시</option><option value="assistant_visual_estimate">어시스턴트의 육안 추정</option></select></label></div><div className="lab-export-actions"><button className="button primary" disabled={!!issues.length || !usable || mode !== 'inspect' || seeking} onClick={download}>라벨 JSON 내려받기 ↓</button><button className="button text-button" onClick={() => { update(emptyVideoDraft()); setMode('inspect'); setAnchor(null); setCorners([]); setNote('현재 클립의 입력을 모두 초기화했습니다.'); }}>이 클립 입력 초기화</button>{downloaded && <span className="lab-download-status">JSON을 내려받았습니다.</span>}</div>
          {!!issues.length && <ul className="lab-issues" aria-label="내보내기 전 필요한 입력">{issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul>}<p className="lab-local-only">JSON 다운로드는 브라우저에서 파일만 만듭니다. 아래 저장 버튼은 라벨을 로컬 서버에 보관하며, 추적 버튼은 저장본에만 별도로 실행됩니다.</p>
        </section>
        <div className="lab-button-row"><button className="button primary" disabled={!!issues.length || !usable || mode !== 'inspect' || seeking} onClick={saveRecord}>{labBusy === 'saving' ? '저장 중…' : '라벨을 로컬 서버에 저장'}</button>{saveError && <p role="alert">{saveError}</p>}</div>
        <SavedVideoLibrary records={records} clips={clips} activeId={saved?.id || null} busy={!!labBusy} loading={recordsLoading} error={recordsError} onLoad={loadRecord} onRefresh={refreshRecords} />
        <VideoTrackingPanel saved={saved} dirty={dirty} busy={labBusy} error={trackingError} onTrack={trackRecord} />
        <div className="lab-notice" role="status" aria-live="polite">{note}</div>
      </>}
      <details className="model-details"><summary>영상 실험의 범위와 제한<span aria-hidden="true">＋</span></summary><div className="details-content"><p>포수의 미트 위치와 선수의 실제 의도는 같은 정보가 아닙니다. 명시적으로 요청한 저장 라벨에만 이미지 추적을 수행하며 투구 성과를 선수에게 귀속하지 않습니다.</p>{limitations.map((item, index) => <p key={index}>{item}</p>)}</div></details>
    </main><footer><a className="footer-brand" href="/">pitcheezy.</a><p>영상 검토실 · 실험용 수동 라벨</p><a className="lab-back-link" href="/">관전 화면으로 돌아가기</a></footer>
  </>;
}

import { useEffect, useRef, useState } from 'react';
import { BADGE, get, short } from './WatchAlong';
import type { Situation } from './WatchAlong';
import source from './showcase-scenes.json';
import './showcase.css';

type Decision = { index: number; at_bat_number: number; pitch_number: number; situation: Situation;
  batter: { name: string | null }; pre: { status: string } };
type Timeline = { game: { game_pk: number; away_team: string; home_team: string }; decisions: Decision[] };

/** Display-only, pinned video annotations. No interpolation from feed UTC or scoring of model quality. */
export default function Showcase() {
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const initial = Number(new URLSearchParams(window.location.search).get('scene'));
  const [selected, setSelected] = useState(Number.isInteger(initial) && initial >= 0 && initial < source.scenes.length ? initial : 0);
  const [loaded, setLoaded] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [copyNote, setCopyNote] = useState('');
  const [take, setTake] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const stopAt = useRef<number | null>(null);
  const scene = source.scenes[selected];
  const decision = timeline?.decisions.find(d => d.at_bat_number === scene.atBat && d.pitch_number === scene.pitchNumber);
  const watchPath = decision ? `/watch?game=${source.gamePk}#i=${decision.index}` : null;

  useEffect(() => {
    const controller = new AbortController();
    void get<Timeline>(`/watch/${source.gamePk}`).then(data => {
      if (controller.signal.aborted) return;
      if (data.game.game_pk !== source.gamePk) throw new Error('영상과 경기 자료가 일치하지 않습니다.');
      setTimeline(data);
    }).catch(e => { if (!controller.signal.aborted) setError((e as Error).message); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('scene', String(selected));
    window.history.replaceState(null, '', `?${params}`);
    setCopyNote('');
    stopAt.current = null;
    if (video.current) {
      video.current.pause();
      if (video.current.readyState >= 1) video.current.currentTime = Math.max(0, scene.releaseSeconds - 3);
    }
  }, [selected, scene.releaseSeconds]);

  function startClip() {
    const player = video.current;
    if (!player || player.readyState < 1) return;
    player.currentTime = Math.max(0, scene.releaseSeconds - 3);
    stopAt.current = scene.releaseSeconds + 4;
    setTake(value => value + 1); // Reset the recommendation panel; previously revealed results must not carry over.
    void player.play().catch(() => { stopAt.current = null; setVideoError(true); });
  }

  async function copyLink() {
    if (!watchPath) return;
    try {
      await navigator.clipboard.writeText(new URL(watchPath, window.location.origin).href);
      setCopyNote('링크를 복사했어요. 아이폰에서 열어 주세요.');
    } catch { setCopyNote('아래 관전 링크를 길게 눌러 복사해 주세요.'); }
  }

  const label = (d: Decision | undefined, n: number) => d
    ? `${d.situation.inning}회 ${d.situation.half === 'Top' ? '초' : '말'} · ${d.batter.name ?? '타자'} · ${n}구째`
    : `영상 장면 · ${n}구째`;
  return <div className="sc-page">
    <header className="sc-head"><a href="/watch">← 경기 목록</a><span>{BADGE}</span></header>
    <main>
      <h1>영상으로 보는 투구 선택</h1>
      <p className="sc-sub">{timeline ? `${short(timeline.game.away_team)} · ${short(timeline.game.home_team)}` : '경기 자료 확인 중'}
        {' — '}공을 던지기 전 추천을 보고, 영상이 끝나면 실제 투구를 공개해 보세요.</p>
      <div className="sc-layout">
        <section className="sc-player" aria-label="경기 영상">
          <div className="sc-video">
            <video ref={video} controls playsInline preload="metadata" src={loaded ? source.mediaUrl : undefined}
              onLoadedMetadata={() => { setVideoReady(true); if (video.current) video.current.currentTime = Math.max(0, scene.releaseSeconds - 3); }}
              onError={() => setVideoError(true)}
              onTimeUpdate={() => {
                if (video.current && stopAt.current !== null && video.current.currentTime >= stopAt.current) {
                  video.current.pause(); stopAt.current = null;
                }
              }} />
            {!loaded && <button className="sc-load" onClick={() => { setLoaded(true); setVideoError(false); }}>공식 경기 영상 불러오기</button>}
          </div>
          {loaded && !videoReady && !videoError && <p role="status">영상 정보를 불러오는 중…</p>}
          {videoError && <p role="alert">영상 재생을 확인하지 못했어요. <a href={source.mediaUrl} target="_blank" rel="noreferrer">공식 영상 따로 열기</a></p>}
          <label className="sc-label">비교할 장면
            <select value={selected} onChange={e => setSelected(Number(e.target.value))}>
              {source.scenes.map((item, i) => {
                const d = timeline?.decisions.find(row => row.at_bat_number === item.atBat && row.pitch_number === item.pitchNumber);
                return <option key={item.pitchId} value={i}>{i + 1}. {label(d, item.pitchNumber)}</option>;
              })}
            </select>
          </label>
          <div className="sc-controls">
            <button disabled={selected === 0} onClick={() => setSelected(i => i - 1)}>이전 장면</button>
            <button className="sc-primary" disabled={!videoReady || !decision || videoError} onClick={startClip}>이 공 다시 보기</button>
            <button disabled={selected === source.scenes.length - 1} onClick={() => setSelected(i => i + 1)}>다음 장면</button>
          </div>
          <p className="sc-hint">선택한 공 전후 약 7초를 재생합니다. 영상 탐색 막대를 직접 움직였다면 장면을 다시 선택해 맞춰 주세요.</p>
          <details><summary>영상과 비교 범위</summary>
            <p>MLB 공식 압축 영상에 나온 39구만 비교합니다. 전체 경기 262구 중 일부이며 대부분 타석의 마지막 공입니다.</p>
            <p>영상 시점은 AI가 판독한 기존 표식으로 맞춥니다. 사람 검토 전이며, 실시간 영상 자동 인식이 아닙니다. 맞지 않는 장면은 시연에 사용하지 않습니다.</p>
            <p>2026 정규시즌 1회 평가에서 추천 정책의 개선 근거를 확인하지 못했습니다. 추천과 같은 공을 던졌다는 사실은 추천이 더 좋다는 증거가 아닙니다.</p>
          </details>
          {watchPath && <div className="sc-phone"><button onClick={() => void copyLink()}>아이폰용 관전 링크 복사</button>
            <a href={watchPath} target="_blank" rel="noreferrer">이 공 관전 화면 열기</a>
            <p>두 기기의 재생은 자동으로 연결되지 않습니다. 장면을 바꿀 때 아이폰에서도 해당 링크를 열어 주세요.</p>
            <p role="status">{copyNote}</p></div>}
        </section>
        <section className="sc-compare" aria-label="추천과 실제 투구 비교">
          <h2>{label(decision, scene.pitchNumber)}</h2>
          {error ? <p role="alert">{error}</p> : watchPath
            ? <iframe key={`${selected}:${take}`} src={watchPath} title="선택한 공의 추천과 실제 투구" />
            : <p>{timeline ? '이 장면에 대응하는 투구 자료가 없습니다. 다른 장면을 선택해 주세요.' : '추천 화면을 준비하는 중…'}</p>}
        </section>
      </div>
    </main>
  </div>;
}

import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import type { Actual, Candidate, Pa, Rec } from './model';
import { halfText, kstTime, policyShare } from './model';
import Zone from './Zone';
import type { Bounds } from './model';

/** Bottom sheet: the only place with inner scrolling. Esc / backdrop / 닫기 close it and focus returns to the opener. */
export function Sheet({ title, onClose, children, labelledBy = 'sv-sheet-title' }: { title: string; onClose: () => void; children: ReactNode; labelledBy?: string }) {
  const box = useRef<HTMLDivElement>(null);
  const [wide, setWide] = useState(() => window.matchMedia('(min-width: 1200px)').matches);
  useEffect(() => { const mq = window.matchMedia('(min-width: 1200px)'); const change = () => setWide(mq.matches); mq.addEventListener('change', change); return () => mq.removeEventListener('change', change); }, []);
  const closeRef = useRef(onClose); closeRef.current = onClose;
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); closeRef.current(); }
      if (e.key === 'Tab' && !wide) {
        const nodes = [...(box.current?.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],select,input,[tabindex="0"]') ?? [])];
        const first = nodes[0], last = nodes[nodes.length - 1];
        if (!first) { e.preventDefault(); return; }
        if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && (document.activeElement === last || document.activeElement === box.current)) { e.preventDefault(); first.focus(); }
      }
    };
    window.addEventListener('keydown', onKey);
    if (!wide) document.body.classList.add('sv-locked');
    return () => { window.removeEventListener('keydown', onKey); document.body.classList.remove('sv-locked'); opener?.focus?.(); };
  }, [wide]);
  return <div className={`sv-backdrop${wide ? ' nonmodal' : ''}`}  onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
    <div className="sv-sheet" role="dialog" aria-modal={wide ? undefined : true} aria-labelledby={labelledBy} tabIndex={-1} ref={box}>
      <div className="sv-sheet-head"><h2 id={labelledBy}>{title}</h2><button className="sv-btn" onClick={onClose}>닫기</button></div>
      <div className="sv-sheet-body">{children}</div>
    </div>
  </div>;
}

/** Fixed explanations: kept, but only on demand (never on the default screen). */
const LIMITS = [
  '추천은 검증 전 실험 버전이에요. 2026 정규시즌 1회 평가에서는 추천이 실제 선택보다 낫다는 근거를 확인하지 못했어요.',
  '‘정책 선택 비중’은 추천 정책이 이 상황에서 그 구종을 고르는 비중이에요. 성공·적중 확률이 아니에요.',
  '추천 위치는 그 투수가 실제로 던진 공의 분포(2025년까지)로 잡은 대략 위치이며, 위치 자체는 평가되지 않았어요.',
  '지난 경기의 추천은 경기 뒤 그 공을 던지기 전 정보만으로 계산한 기록이에요. 경기 중 실시간으로 보여 준 추천이 아니에요.',
  '포수 미트 표시는 중계 영상에서 읽은 가로 위치 추정이며, 높이는 다루지 않아요. 실제 투구 뒤에만 보여요.',
  '날짜와 시간은 모두 한국 시간이에요. 경기 날짜는 원래 예정 시작 시각 기준이에요.',
];
export function InfoSheet({ onClose, notes }: { onClose: () => void; notes: string[] }) {
  return <Sheet title="이 화면의 기준과 한계" onClose={onClose}>
    <ul className="sv-notes">{[...LIMITS, ...notes.filter(n => !LIMITS.includes(n))].map(n => <li key={n}>{n}</li>)}</ul>
  </Sheet>;
}

export function PaSheet({ pas, selected, onPick, onClose, onBox, foldText = null }: { pas: Pa[]; selected: string | null; onPick: (pa: Pa) => void; onClose: () => void; onBox: () => void;
  /** viewer delay, device not synced yet: finished plate appearances do not say how they ended */
  foldText?: string | null }) {
  const current = pas.find(p => p.pa_key === selected) ?? pas[0];
  const key = (p: Pa) => `${p.inning}:${p.half}`;
  const [half, setHalf] = useState(current ? key(current) : '');
  const halves = [...new Map(pas.map(p => [key(p), p])).entries()];
  const visible = pas.filter(p => key(p) === half);
  return <Sheet title="타석 선택" onClose={onClose}>
    <div className="sv-pa-controls"><label>이닝 <select aria-label="이닝과 초말 선택" value={half} onChange={e => setHalf(e.target.value)}>
      {halves.map(([k,p]) => <option key={k} value={k}>{p.inning}회 {halfText(p.half)}</option>)}
    </select></label><button className="sv-btn" onClick={onBox}>박스스코어</button></div>
    <ol className="sv-pa-list">{visible.map(pa => <li key={pa.pa_key}>
      <button className={`sv-pa-item ${pa.complete ? '' : 'open'}`} aria-current={pa.pa_key === selected ? 'true' : undefined} onClick={() => onPick(pa)}>
        <span className="sv-pa-who">{pa.batting_order ? `${pa.batting_order}번 · ` : ''}<b>{pa.batter}</b>{pa.pa_key === selected && <small> · 보는 중</small>}</span>
        <span className="sv-pa-result">{foldText && pa.complete ? foldText : pa.result} · {pa.pitch_count ?? (Number.isFinite(pa.first) && Number.isFinite(pa.last) ? Math.max(0, pa.last - pa.first + 1) : '–')}구</span>
      </button></li>)}</ol>
    {!visible.length && <p className="sv-muted">공개된 타석이 아직 없어요.</p>}
  </Sheet>;
}

const PROVENANCE: Record<Rec['provenance'], string> = {
  captured_live: '경기 중 투구 전에 기록한 추천',
  asof_replay: '경기 뒤, 그 공을 던지기 전 정보만으로 계산한 추천',
  absent: '기록된 추천 없음',
};
/** The one calm line for a pitch without a recorded recommendation: the server's sentence when it sends one (the pre-2026-10-11 stock
 *  sentence counts as none), else ours. */
export const ABSENT_PAST = '이 공은 기록된 추천이 없어요.';
const LEGACY_ABSENT = '이 투구의 저장된 추천이 없습니다.';
export const absentLine = (rec: Rec, fallback = ABSENT_PAST) => rec.reason && rec.reason.trim() !== LEGACY_ABSENT ? rec.reason : fallback;
export const provenanceText = (rec: Rec) => PROVENANCE[rec.provenance];

/** Tracking numbers as the sheet prints them: one decimal for lengths and speeds measured in tenths, whole numbers for spin, angle and
 *  distance (the feed sends 6.64219001212026 ft; the fan reads 6.6 ft). A missing value is '—', never 0. */
export const fixed1 = (v: number | null | undefined) => v == null ? null : (Math.round(v * 10) / 10).toFixed(1);
export const whole = (v: number | null | undefined) => v == null ? null : String(Math.round(v));

export function DetailSheet({ onClose, rec, recTitle, actual, actualTitle, paResult, notes, speed, extra, bounds, recNote = null, actualPending = false, actualCorrected = false, actualHeld = false, pair = null, absentText = ABSENT_PAST, actualFolded = null, actualExtra = null }:
  { onClose: () => void; rec: Rec | null; recTitle: string; actual: Actual | null; actualTitle: string; paResult: string | null; notes: string[]; speed: (mph: number | null) => string; extra?: ReactNode; bounds?: Bounds;
    /** live display state: the server's sentence (and any change note) for the pending pitch */
    recNote?: string | null; actualPending?: boolean; actualCorrected?: boolean;
    /** the publisher refused the feed's newest body of this pitch and shows the last accepted one (`held_last_accepted`) */
    actualHeld?: boolean;
    /** viewer delay, device not synced yet: the thrown pitch is not described */
    actualFolded?: string | null;
    /** an optional quiet line that belongs to the thrown pitch (never shown while the pitch is folded or being confirmed) */
    actualExtra?: ReactNode;
    /** past pitch with a recorded recommendation: its first pick and the pitch that was thrown, side by side in words (never a score) */
    pair?: { rec: string; actual: string } | null; absentText?: string }) {
  return <Sheet title={actual ? `투구 상세 · ${actualTitle}` : "추천 구종 · 대략 위치"} onClose={onClose}>
    <section className="sv-detail">
      <h3>{recTitle}</h3>
      {recNote && <p>{recNote}</p>}
      {recNote && (rec === null || rec.status !== 'ready') ? null : rec === null ? <p className="sv-muted">이 시점에는 추천이 없어요.</p>
        : rec.status !== 'ready' ? <p>{rec.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황이에요.' : absentLine(rec, absentText)}</p>
        : <>
          {pair && <dl className="sv-dl sv-pair"><dt>추천 1순위</dt><dd>{pair.rec}</dd><dt>실제 투구</dt><dd>{pair.actual}</dd></dl>}
          {pair && <p className="sv-muted">투구 전에 기록된 추천과 실제로 던진 공을 나란히 보여 드려요. 맞고 틀림을 매긴 것이 아니에요.</p>}
          <p className="sv-muted">{provenanceText(rec)}{rec.recorded_at ? ` · ${kstTime(rec.recorded_at)} 기록` : ''}</p>
          <table className="sv-table"><thead><tr><th scope="col">순위</th><th scope="col">구종</th><th scope="col">대략 위치</th><th scope="col">정책 선택 비중</th></tr></thead>
            <tbody>{rec.candidates.map((c: Candidate) => <tr key={c.rank}><td>{c.rank}</td><td>{c.pitch_label}</td><td>{c.zone_label || '위치 없음'}</td><td>{policyShare(c.probability)}</td></tr>)}</tbody></table>
          <p className="sv-muted">정책 선택 비중은 성공·적중 확률이 아니에요. 위치는 과거 투구 분포의 대략 위치이며, 칸별 가치 지도가 아닙니다.</p>
        </>}
    </section>
    <section className="sv-detail">
      <h3>{actualTitle}</h3>
      {actualFolded ? <p className="sv-muted">{actualFolded}. 화면 위쪽의 ‘중계 맞추기’에서 내 중계에 맞추면 구종·위치·결과를 보여 드려요.</p>
      : actualPending ? <p className="sv-muted">이 공을 다시 확인하고 있어요. 확인되면 구종과 위치를 보여 드려요.</p> : <>
      {actualHeld ? <p className="sv-muted">공식 피드가 이 공을 다시 보냈지만 받아들이지 않아, 마지막으로 받아들인 기록을 보여 드려요.{actualCorrected ? ' 받아들인 기록도 처음 받은 뒤에 바뀐 적이 있어요.' : ''}</p>
        : actualCorrected && <p className="sv-muted">이 공의 기록은 처음 받은 뒤에 정정됐어요.</p>}
      {actual && bounds && <div className="sv-detail-zone"><Zone fullWindow bounds={bounds} candidates={rec?.candidates ?? []} actual={actual.x !== null && actual.z !== null ? {x:actual.x,z:actual.z} : null} actualLabel="" setupX={actual.setup_x} /></div>}
      {actual === null ? <p className="sv-muted">아직 던지지 않았어요.</p> : <dl className="sv-dl">
        <dt>구종</dt><dd>{actual.pitch_label}{actual.speed_mph !== null ? ` · ${speed(actual.speed_mph)}` : ''}</dd>
        <dt>원본 좌표</dt><dd>{actual.x !== null && actual.z !== null ? `가로 ${actual.x.toFixed(2)} ft · 높이 ${actual.z.toFixed(2)} ft (포수 기준)` : '위치 정보 없음'}</dd><dt>판정</dt><dd>{actual.result_label || '기록 없음'}</dd>
        <dt>위치</dt><dd>{actual.zone_label || '위치 정보 없음'}</dd>
        <dt>회전수</dt><dd>{whole(actual.tracking?.spin_rate_rpm) === null ? '—' : `${whole(actual.tracking!.spin_rate_rpm)} rpm`}</dd>
        <dt>익스텐션</dt><dd>{fixed1(actual.tracking?.extension_feet) === null ? '—' : `${fixed1(actual.tracking!.extension_feet)} ft`}</dd>
        <dt>타구 속도</dt><dd>{fixed1(actual.tracking?.exit_velocity_mph) === null ? '—' : `${fixed1(actual.tracking!.exit_velocity_mph)} mph`}</dd>
        <dt>발사각 / 비거리</dt><dd>{whole(actual.tracking?.launch_angle_deg) === null ? '—' : `${whole(actual.tracking!.launch_angle_deg)}°`} / {whole(actual.tracking?.distance_feet) === null ? '—' : `${whole(actual.tracking!.distance_feet)} ft`}</dd>
        {paResult && <><dt>타석 결과</dt><dd>{paResult}</dd></>}
        {actual.play_text && <><dt>원본 기록</dt><dd lang="en">{actual.play_text}</dd></>}
      </dl>}{actual !== null && actualExtra}</>}
    </section>
    {notes.length > 0 && <section className="sv-detail"><h3>자료 설명</h3><ul className="sv-notes">{notes.map(n => <li key={n}>{n}</li>)}</ul></section>}
    {extra && <section className="sv-detail">{extra}</section>}
  </Sheet>;
}

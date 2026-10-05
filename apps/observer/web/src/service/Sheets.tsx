import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import type { Actual, Candidate, Pa, Rec } from './model';
import { halfText, policyShare } from './model';

/** Bottom sheet: the only place with inner scrolling. Esc / backdrop / 닫기 close it and focus returns to the opener. */
export function Sheet({ title, onClose, children, labelledBy = 'sv-sheet-title' }: { title: string; onClose: () => void; children: ReactNode; labelledBy?: string }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', onKey);
    document.body.classList.add('sv-locked');
    return () => { window.removeEventListener('keydown', onKey); document.body.classList.remove('sv-locked'); opener?.focus?.(); };
  }, [onClose]);
  return <div className="sv-backdrop" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
    <div className="sv-sheet" role="dialog" aria-modal="true" aria-labelledby={labelledBy} tabIndex={-1} ref={box}>
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

export function PaSheet({ pas, selected, onPick, onClose }: { pas: Pa[]; selected: string | null; onPick: (pa: Pa) => void; onClose: () => void }) {
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => { list.current?.querySelector<HTMLElement>('[aria-current="true"]')?.scrollIntoView({ block: 'center' }); }, []);
  let lastHalf = '';
  return <Sheet title={`전체 타석 (${pas.length})`} onClose={onClose}>
    <ol className="sv-pa-list" ref={list}>{pas.map(pa => {
      const half = `${pa.inning}회 ${halfText(pa.half)}`, heading = half !== lastHalf; lastHalf = half;
      return <li key={pa.pa_key}>
        {heading && <h3 className="sv-pa-half">{half}</h3>}
        <button className={`sv-pa-item ${pa.complete ? '' : 'open'}`} aria-current={pa.pa_key === selected ? 'true' : undefined} onClick={() => onPick(pa)}>
          <span className="sv-pa-who">{half} · <b>{pa.batter}</b> @ {pa.pitcher}</span>
          <span className="sv-pa-result">{pa.result}</span>
        </button>
      </li>;
    })}</ol>
  </Sheet>;
}

const PROVENANCE: Record<Rec['provenance'], string> = {
  captured_live: '경기 중 투구 전에 기록한 추천',
  asof_replay: '경기 뒤, 그 공을 던지기 전 정보만으로 계산한 추천',
  absent: '저장된 추천 기록 없음',
};
export const provenanceText = (rec: Rec) => PROVENANCE[rec.provenance];

export function DetailSheet({ onClose, rec, recTitle, actual, actualTitle, paResult, notes, speed, extra }:
  { onClose: () => void; rec: Rec | null; recTitle: string; actual: Actual | null; actualTitle: string; paResult: string | null; notes: string[]; speed: (mph: number | null) => string; extra?: ReactNode }) {
  return <Sheet title="자세히" onClose={onClose}>
    <section className="sv-detail">
      <h3>{recTitle}</h3>
      {rec === null ? <p className="sv-muted">이 시점에는 추천이 없어요.</p>
        : rec.status !== 'ready' ? <p>{rec.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황이에요.' : '이 공은 추천 기록이 없어요.'}</p>
        : <>
          <p className="sv-muted">{provenanceText(rec)}{rec.policy ? ` · ${rec.policy}` : ''}</p>
          <table className="sv-table"><thead><tr><th scope="col">순위</th><th scope="col">구종</th><th scope="col">대략 위치</th><th scope="col">정책 선택 비중</th></tr></thead>
            <tbody>{rec.candidates.map((c: Candidate) => <tr key={c.rank}><td>{c.rank}</td><td>{c.pitch_label}</td><td>{c.zone_label || '위치 없음'}</td><td>{policyShare(c.probability)}</td></tr>)}</tbody></table>
          <p className="sv-muted">정책 선택 비중은 성공·적중 확률이 아니에요.</p>
        </>}
    </section>
    <section className="sv-detail">
      <h3>{actualTitle}</h3>
      {actual === null ? <p className="sv-muted">아직 던지지 않았어요.</p> : <dl className="sv-dl">
        <dt>구종</dt><dd>{actual.pitch_label}{actual.speed_mph !== null ? ` · ${speed(actual.speed_mph)}` : ''}</dd>
        <dt>판정</dt><dd>{actual.result_label || '기록 없음'}</dd>
        <dt>위치</dt><dd>{actual.zone_label || '위치 정보 없음'}</dd>
        {paResult && <><dt>타석 결과</dt><dd>{paResult}</dd></>}
        {actual.play_text && <><dt>원본 기록</dt><dd lang="en">{actual.play_text}</dd></>}
      </dl>}
    </section>
    {notes.length > 0 && <section className="sv-detail"><h3>자료 설명</h3><ul className="sv-notes">{notes.map(n => <li key={n}>{n}</li>)}</ul></section>}
    {extra && <section className="sv-detail">{extra}</section>}
  </Sheet>;
}

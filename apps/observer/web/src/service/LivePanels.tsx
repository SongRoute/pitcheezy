import { useState } from 'react';
import type { Detail } from './model';
import { inningText, short } from './model';
import { Sheet } from './Sheets';

export function RecordSheet({ detail, fold, onClose, onPick }: { detail: Detail; fold: boolean; onClose: () => void; onPick: (index: number) => void }) {
  const [tab, setTab] = useState('문자 로그');
  const context = detail.current_context;
  return <Sheet title="경기 기록" onClose={onClose}>
    <div className="sv-row">{['문자 로그', '라인업', '박스', '분석'].map(t => <button className="sv-btn" key={t} aria-pressed={tab === t} onClick={() => setTab(t)}>{t}</button>)}</div>
    {tab === '문자 로그' && <ol className="sv-recent-pas">{[...detail.pas].reverse().map(p => <li key={p.pa_key}><button onClick={() => onPick(p.first)} disabled={!detail.pitches.some(x => x.index === p.first)}><b>{inningText(p)} · {p.batter}</b><span>{fold ? '중계를 맞춘 뒤 표시' : p.result} · {p.pitch_count ?? detail.pitches.filter(x => x.pa_key === p.pa_key).length}구</span></button></li>)}</ol>}
    {tab === '라인업' && (['away', 'home'] as const).map(side => <section key={side}><h3>{short(detail.game[`${side}_team`])}</h3>{context?.lineups?.[side]?.length ? <ol>{context.lineups[side].map(p => <li key={`${p.order}:${p.id}`}>{p.order}번 · {p.name ?? '선수 미확인'}</li>)}</ol> : <p className="sv-muted">라인업 정보 없음</p>}</section>)}
    {tab === '박스' && (['away', 'home'] as const).map(side => <section key={side}><h3>{short(detail.game[`${side}_team`])}</h3>{context?.boxscore?.[side] ? <>
      <table className="sv-table"><thead><tr><th>타자</th><th>타수</th><th>안타</th><th>출루</th></tr></thead><tbody>{context.boxscore[side].batters.map(p => <tr key={p.id}><td>{p.name}</td><td>{p.at_bats ?? '—'}</td><td>{p.hits ?? '—'}</td><td>{p.on_base ?? '—'}</td></tr>)}</tbody></table>
      <table className="sv-table"><thead><tr><th>투수</th><th>투구</th><th>삼진</th><th>자책</th></tr></thead><tbody>{context.boxscore[side].pitchers.map(p => <tr key={p.id}><td>{p.name}</td><td>{p.pitches ?? '—'}</td><td>{p.strikeouts ?? '—'}</td><td>{p.earned_runs ?? '—'}</td></tr>)}</tbody></table></> : <p className="sv-muted">박스 기록 정보 없음</p>}</section>)}
    {tab === '분석' && <div className="sv-unmeasured"><h3>분석 <span className="sv-tag">준비 중</span></h3><p>성분별 책임 분해·기대실점·타구 예상 결과는 아직 연결되지 않았어요.</p></div>}
    <p className="sv-muted">공개된 시점까지의 기록이며 이후 정정될 수 있어요. 출루는 안타·볼넷·몸에 맞는 공을 합한 값이에요.</p>
  </Sheet>;
}
export function PitchMixSheet({ detail, pitcherId, fold, onClose }: { detail: Detail; pitcherId: number | null; fold: boolean; onClose: () => void }) {
  const rows = fold ? [] : detail.pitches.filter(p => p.pitcher.id === pitcherId && p.receipt?.status !== 'unconfirmed' && p.actual?.pitch_type);
  const groups = [...new Set(rows.map(p => p.actual!.pitch_type))].map(type => { const pitches = rows.filter(p => p.actual!.pitch_type === type); const speeds = pitches.flatMap(p => p.actual!.speed_mph === null ? [] : [p.actual!.speed_mph]); return { type, label: pitches[0].actual!.pitch_label, n: pitches.length, speed: speeds.length ? speeds.reduce((a,b) => a+b, 0) / speeds.length * 1.609344 : null }; });
  return <Sheet title="구종 · 수신된 투구 기록" onClose={onClose}>
    {rows.length ? <><p>선택 투수의 수신된 기록 중 구종이 확인된 {rows.length}구 기준</p><table className="sv-table"><thead><tr><th>구종</th><th>투구 수</th><th>관측 비율</th><th>평균 구속</th></tr></thead><tbody>{groups.map(g => <tr key={g.type}><td>{g.label}</td><td>{g.n}</td><td>{Math.round(g.n / rows.length * 100)}%</td><td>{g.speed === null ? '—' : `${Math.round(g.speed)} km/h`}</td></tr>)}</tbody></table><p className="sv-muted">경기 전체 투구가 모두 수신됐다는 뜻은 아니에요. 구속이 없는 공은 평균에서 제외해요.</p></> : <p className="sv-muted">{fold ? '중계를 맞춘 뒤 표시해요.' : '확인된 구종 기록이 없어요.'}</p>}
    <p className="sv-muted">시즌별 사용률·헛스윙률·피xwOBA는 아직 연결되지 않았어요.</p>
  </Sheet>;
}

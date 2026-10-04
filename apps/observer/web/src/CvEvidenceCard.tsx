import evidence from './cv-m3-evidence.json';

// Pinned display copy of the PM M3 receipt (see source fields in the JSON). Reading agreement between the AI and one person
// under the same transform: not physical accuracy, not verified intent and not target height. No video or recommendation links.
const px = (v: number) => `${v.toFixed(1)}px`;
const inch = (feet: number) => `${(feet * 12).toFixed(1)}인치`;
const SAME = '같은 변환 기준의 판독 차이 · 물리 정확도 아님';
const ROLE: Record<string, string> = { representative: '대표(중앙값에 가장 가까움)', smallest: '출력 좌표 차이가 가장 작음',
  largest: '출력 좌표 차이가 가장 큼', fox_representative: 'FOX 대표 · 카메라 미확인' };

export default function CvEvidenceCard() {
  const p = evidence.pooled;
  return <details className="cv-card">
    <summary>
      <b>CV 검증 결과 · 포수 셋업 추정</b>
      <span>평가 {evidence.games.length}경기 · {p.frames}장</span>
      <span>AI 출력 {p.aiOutput} / 기권 {p.aiAbstained}</span>
      <span>사람과 비교한 {p.compared}구의 미트 점 차이 중앙값 {px(p.pixels.median)}</span>
      <i className="cv-more">경기별 결과와 한계 보기</i>
    </summary>
    <p>포수가 준비 자세에서 둔 미트를 추정했습니다. 두 경기 {p.frames}장 중 {p.aiOutput}장에서 좌표를 냈고, 그 {p.compared}장을 사람 판독과 비교한 점 차이 중앙값은 {px(p.pixels.median)}입니다. 아직 실제 의도나 목표 높이를 검증한 것은 아닙니다.</p>
    <h3>경기별</h3>
    <ul className="cv-games">{evidence.games.map(g => <li key={g.gamePk}>
      <b>{g.broadcast} 중계 · 경기 {g.gamePk}</b>
      <span>AI 출력 {g.aiOutput}/{g.frames}장 · 기권 {g.aiAbstained}장 (사람 표시 {g.personMarked}장)</span>
      <span>미트 점 차이 중앙값 {px(g.pixels.median)} · 90분위 {px(g.pixels.p90)} ({g.compared}구)</span>
      <span>출력 좌표 차이 중앙값 약 {inch(g.outputFeet.median)} · 90분위 {inch(g.outputFeet.p90)} <em>{SAME}</em></span>
      <span>{g.cameraCheck.status === 'measured' && g.cameraCheck.rmsFeet !== null
        ? `카메라 보정: 포구 공 ${'caughtBalls' in g.cameraCheck ? `${g.cameraCheck.caughtBalls}구 ` : ''}대조 RMS ${g.cameraCheck.rmsFeet.toFixed(2)}ft (고정 보정 파일 값)`
        : '카메라 물리 확인: 미측정 (카메라 미확인)'}</span></li>)}
    </ul>
    <p>합산: 출력 좌표 차이 중앙값 약 {inch(p.outputFeet.median)} · 90분위 {inch(p.outputFeet.p90)} <em>{SAME}</em></p>
    <p>AI가 기권한 {p.aiAbstained}장 중 {p.aiAbstainedPersonMarked}장은 사람이 미트를 표시했습니다. 기권한 장면은 차이 계산에서 빠지므로, 위 차이는 AI가 좌표를 낸 {p.compared}구에 대한 값입니다.</p>
    <h3>예시 투구</h3>
    <ul className="cv-examples">{evidence.examples.map(e => <li key={e.pitchId}>
      <b>{e.pitchId}</b> <span>{ROLE[e.role] ?? e.role}</span>
      <span>미트 점 차이 {px(e.pixels)} · 출력 좌표 차이 약 {inch(e.outputFeet)}</span></li>)}
    </ul>
    <p>예시는 시연 영상에 없는 경기라 영상 장면이나 추천 화면으로 이동하지 않습니다. 픽셀 차이와 출력 좌표 차이는 순서가 다를 수 있습니다.</p>
    <h3>한계</h3>
    <ul className="cv-limits">
      <li>투수·포수의 실제 의도와 목표 높이는 검증하지 않았습니다. 높이는 미검증입니다.</li>
      <li>사람 판독자는 1명이고, 두 압축 영상은 경기 전체 {p.gamePitchesTotal}구 중 {p.frames}구로 타석 마지막 공에 치우쳐 있습니다.</li>
      <li>이 측정만으로 전체 영상의 탐지율이나 제공 범위를 말할 수 없습니다.</li>
      <li>추천·9구역·예측 모델에는 사용하지 않습니다.</li>
    </ul>
    <p className="cv-src">출처: 팀 커밋 {evidence.source.teamCommit.slice(0, 7)} · PM 재계산 {evidence.source.receiptSha256.slice(0, 8)} · 계약 검사 {evidence.source.validatorRowsPassed}/{p.frames}</p>
  </details>;
}

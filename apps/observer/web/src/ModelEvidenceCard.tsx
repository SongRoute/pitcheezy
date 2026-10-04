import evidence from './model-evidence.json';

// Pinned display copy of committed results (source hashes in the JSON). Prediction accuracy and recommendation-policy effect
// are separate claims: a better predictor is not evidence that the recommendation helps. The research mix is not in service.
const f4 = (v: number) => v.toFixed(4);
const n = (v: number) => v.toLocaleString('ko-KR');
const lower = (v: number) => Math.abs(v).toFixed(4);
const signed = (v: number) => `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(4)}`;

export default function ModelEvidenceCard() {
  const p = evidence.prediction, m = evidence.researchMix, o = evidence.policy;
  const [lo, hi] = p.deltaNllCi95, [mlo, mhi] = m.deltaNllVsG0Ci95, [olo, ohi] = o.deltaBounds;
  return <details className="model-card">
    <summary>
      <b>예측 모델·추천 검증 결과</b>
      <span>예측: 2025 평가 {n(p.pitches)}구에서 상황 빈도 기준보다 로그손실 {lower(p.deltaNll)} 낮음</span>
      <span>추천: 2026 정규시즌 1회 평가 {n(o.games)}경기 · 개선 근거 없음</span>
      <i className="cv-more">예측과 추천을 나눠 보기</i>
    </summary>
    <p>예측이 더 정확한 것과 추천이 경기 결과를 좋게 만드는 것은 서로 다른 질문입니다. 예측은 좋아졌지만, 추천의 효과는 확인하지 못했습니다.</p>
    <h3>1. 예측 정확도 (서비스 모델)</h3>
    <ul className="model-list">
      <li><b>무엇을 예측하나</b><span>다음 공 한 개의 결과 {p.outcomeClasses}종(볼·스트라이크·파울·아웃·안타 종류 등)이 나올 확률</span></li>
      <li><b>평가 자료</b><span>{p.devWindow[0]} ~ {p.devWindow[1]} 전체 MLB {n(p.pitches)}구 · {n(p.games)}경기. 학습·보정 기간(~6월) 뒤의 자료입니다.</span></li>
      <li><b>로그손실 (낮을수록 좋음)</b><span>서비스 모델 {f4(p.g0.nll)} · 상황 빈도 기준 {f4(p.frequency.nll)}</span>
        <span>차이 −{lower(p.deltaNll)} (95% 구간 −{lower(lo)} ~ −{lower(hi)}, 경기 단위 재표본 {n(p.bootstrapDraws)}회, 시드 {p.negativeSeeds}/{p.seeds}개 모두 개선)</span></li>
      <li><b>확률 눈금(보정)</b><span>보정 오차 ECE는 빈도 기준 {f4(p.frequency.ece10)}이 서비스 모델 {f4(p.g0.ece10)}보다 작습니다. 모든 면에서 나은 것은 아닙니다.</span></li>
    </ul>
    <p>연구용 혼합 모델은 같은 평가에서 서비스 모델보다 로그손실이 {lower(m.deltaNllVsG0)} 더 낮았습니다(95% 구간 −{lower(mlo)} ~ −{lower(mhi)}, 시드 {m.negativeSeeds}/{m.seeds}). 서비스와 추천에는 아직 쓰지 않습니다.</p>
    <h3>2. 추천 정책 효과</h3>
    <ul className="model-list">
      <li><b>평가 자료</b><span>2026 정규시즌 {n(o.games)}경기 · {n(o.plateAppearances)}타석, 사전에 등록한 방법으로 1회만 평가 (학습·선택에는 쓰지 않음)</span></li>
      <li><b>결과</b><span>타석당 수비 승리확률 차이 범위 {signed(olo)} ~ {signed(ohi)}</span>
        <span>개선 문턱 +{o.improvementThreshold.toFixed(3)}에 미치지 못해 <strong>개선 근거 없음</strong>. 해가 없다는 증명도 아닙니다.</span></li>
      <li><b>추천하지 않는 경우</b><span>2025년까지 학습 기록이 없는 투수로 시작한 {n(o.excludedUnknownPitcher)}타석은 평가 전에 제외했습니다. 화면에서도 이런 투수에게는 추천을 만들지 않습니다.</span></li>
    </ul>
    <h3>한계</h3>
    <ul className="cv-limits">
      <li>2025 평가 자료는 여러 번 본 개발용 자료입니다. 새 자료로 다시 확인한 결과가 아닙니다.</li>
      <li>추천과 같은 공을 던졌다는 사실이나 공개 뒤 승리확률 변화는 추천의 효과를 보여 주지 않습니다.</li>
      <li>정책 평가 결과는 사전에 정한 가정 아래의 추정이며, 그 가정이 맞을 때만 인과 효과로 읽을 수 있습니다.</li>
      <li>10/4 탐색 실험들의 수치는 독립 검토 전이라 이 카드에 넣지 않았습니다.</li>
    </ul>
    <p className="cv-src">출처: 레포 {evidence.source.repoCommit.slice(0, 7)} · {Object.entries(evidence.source.files).map(([file, sha]) => `${file.replace(/^(results|docs)\//, '').replace(/\.(json|md)$/, '')} ${sha.slice(0, 8)}`).join(' · ')} · 결정 {evidence.source.decisions.join('·')}</p>
  </details>;
}

import type { Recommendation } from './types';

export default function ChoiceExplanation({ recommendation }: { recommendation: Recommendation }) {
  const explanation = recommendation.explanation;
  if (!explanation) return null;
  const comparison = explanation.comparison;
  const contributions = comparison?.contributions.filter(item => Number.isFinite(item.value_contribution_pp)) || [];
  const positive = [...contributions].sort((a, b) => b.value_contribution_pp - a.value_contribution_pp)[0];
  const negative = [...contributions].sort((a, b) => a.value_contribution_pp - b.value_contribution_pp)[0];
  const direction = (value: number) => value >= 0 ? '높게' : '낮게';
  return <details className="probability-details choice-explanation">
    <summary>왜 이 후보를 먼저 제안했나요?<span aria-hidden="true">＋</span></summary>
    <p>{explanation.notice}</p>
    {comparison && Number.isFinite(comparison.margin_pp) && <>
      <p>첫 후보와 두 번째 후보의 모델 평가 차이는 {comparison.margin_pp.toFixed(4)}%p입니다. {comparison.definition}</p>
      {positive && positive.value_contribution_pp > 0 && <p>첫 후보에 유리하게 작용한 항목: {positive.label} 확률을 {Math.abs(positive.probability_difference_pp).toFixed(2)}%p {direction(positive.probability_difference_pp)} 추정했습니다.</p>}
      {negative && negative.value_contribution_pp < 0 && <p>첫 후보에 불리하게 작용한 항목: {negative.label} 확률을 {Math.abs(negative.probability_difference_pp).toFixed(2)}%p {direction(negative.probability_difference_pp)} 추정했습니다.</p>}
    </>}
    {explanation.outcome_probabilities.slice(0, recommendation.candidates.length).map((outcomes, index) => <details key={index}>
      <summary>{index + 1}위 · {recommendation.candidates[index].pitch_label} · {recommendation.candidates[index].zone_label} 결과 분포</summary>
      <dl>{outcomes.map(item => <div key={item.outcome}><dt>{item.label}</dt><dd>{Number.isFinite(item.probability) ? `${(item.probability * 100).toFixed(1)}%` : '정보 없음'}</dd></div>)}</dl>
    </details>)}
  </details>;
}

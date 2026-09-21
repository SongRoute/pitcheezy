"""Auditably derive fan explanations from the same probabilities used by the planner."""
import math

OUTCOMES = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple', 'home_run', 'hbp', 'double_play')
LABELS = ('볼', '스트라이크', '파울', '타구 아웃', '단타', '2루타', '3루타', '홈런', '몸에 맞는 공', '병살')


def explain_choices(probabilities, candidate_values, values, terminal, balls, strikes):
    """Explain top choices, holding optimal continuation fixed after this pitch.

Centering continuation utilities on V makes contributions interpretable as
benefits/costs relative to the current state. Since probability differences sum
to zero, contributions still sum exactly to Q(best)-Q(runner-up).
"""
    labels = list(LABELS)
    labels[0] = '볼넷' if balls == 3 else '볼'
    labels[1] = '삼진' if strikes == 2 else '스트라이크'
    labels[2] = '파울 · 카운트 유지' if strikes == 2 else '파울'
    continuation = [terminal['walk'] if balls == 3 else float(values[balls+1, strikes, 0]),
                    terminal['strikeout'] if strikes == 2 else float(values[balls, strikes+1, 0]),
                    float(values[balls, min(2, strikes+1), 0]),
                    *[terminal[name] for name in OUTCOMES[3:]]]
    center = float(values[balls, strikes, 0])
    rows = [[float(p) for p in row] for row in probabilities]
    if not rows or any(len(row) != 10 or any(not math.isfinite(p) or p < 0 for p in row)
                       or abs(sum(row)-1) > 1e-7 for row in rows):
        raise ValueError('Explanation requires complete normalized outcome probabilities')
    marginals = [[{'outcome': name, 'label': labels[index], 'probability': row[index]}
                  for index, name in enumerate(OUTCOMES)] for row in rows]
    comparison = None
    if len(rows) > 1:
        terms = [{'outcome': name, 'label': labels[i],
                  'probability_difference_pp': 100*(rows[0][i]-rows[1][i]),
                  'value_contribution_pp': 100*(rows[0][i]-rows[1][i])*(continuation[i]-center)}
                 for i, name in enumerate(OUTCOMES)]
        margin = 100*(float(candidate_values[0])-float(candidate_values[1]))
        residual = sum(term['value_contribution_pp'] for term in terms)-margin
        if abs(residual) > 1e-7:
            raise ValueError('Explanation does not reconstruct the planner ranking margin')
        comparison = {'compared_candidate_index': 1, 'margin_pp': margin,
                      'contributions': terms, 'reconstruction_residual_pp': residual,
                      'definition': '같은 이후 투구 정책을 따른다는 모델 안에서 첫 공의 결과 확률 차이를 분해했습니다.'}
    return {'outcome_probabilities': marginals, 'comparison': comparison,
            'source': 'same_planner_probabilities', 'causal_effect': False,
            'notice': '모델이 추정한 한 공의 결과 분포입니다. 실제 빈도나 검증된 효과가 아니며 헛스윙과 루킹은 구분하지 않습니다.'}

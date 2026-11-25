/** "예측 모델 소개" sheet (proposal §3 ④). Only measured claims (11/22: replaced by the FINAL v4 lines below; the old notes kept as history):
 *  - service model vs. situation-frequency baseline ΔNLL −0.0137, 5/5 seeds, 2025-07~09 MLB 311,721 pitches (results/EXP-P11-001.json)
 *  - 73% (0.108 of 0.148) of the gap to the re-implemented B2 study model is post-release physics (D34) */
export const MODEL_LINES: string[] = [
  // PM 11/22 20:2x (owner "전부 추천대로"): booth model FINAL v4; wording and numbers = MUST-SAY.v4draft (구종 예측 모델 as subject)
  '다음 공의 구종을 예측하는 모델이에요. 나무 모델 하나와 순환 신경망 두 종류를 섞었고, 던지기 전 정보만 써요.',
  '학습에 안 쓴 올해 포스트시즌 31경기에서 구종 예측 모델의 1순위가 실제 구종과 같은 것은 약 2번 중 1번, 3순위 안은 약 10번 중 9번이었어요. 같은 31경기를 여러 번 읽었어요.',
  '문헌 방식 중 가장 좋았던 순환 신경망을 다시 만들어 비교했고, 포스트시즌 31경기에서 지금 모델의 예측 오차가 더 낮았어요. 지금 모델은 일부가 최근 3년 자료도 봤어요. 31경기는 여러 번 본 자료예요.',
  '2026 정규시즌으로 학습했고, 일부는 2023~2025 자료도 봤어요.',
];
/** TODO_AFTER_SCORING: the pitch-type predictor's one line, filled only after its scoring is done (measured numbers only).
 *  Empty = the sheet renders nothing for it. */
export const MODEL_PITCH_TYPE_LINE_TODO_AFTER_SCORING = '';

# T3 실행·평가 사전 규약

작성 2026-09-24. 실제 T3 추론·결과를 열기 전의 규약이다. 후보 ID는 G 결과의 사전 선택 규칙으로만 채우고 결과 해시와 함께 별도 실행 config에 고정한다. [T2/T3 입력 규약](ML-T2-T3-PROTOCOL.md)의 정확한 `matrix_stress.protocol()`을 복사한다.

- Cpanel, 같은 구·라벨, 후보 1개와 그 등록 대조군, seeds 0/1/2, 400 delivery draws를 유지한다. G1→G0, G2→G0, G3→G2, G4→G2이다. 성능 개선 후보가 없으면 별도 진단 후보 선택 근거를 등록하며 채택으로 표시하지 않는다.
- clean 포함 10시나리오 × 2모델 × 3seeds = 60회 추론이다. 새로운 fit은 없다. 60개 산출물이 모두 검증된 뒤 한 번에 평가한다.
- clean도 실제 재추론한다. 보관 raw/calibrated 예측 및 최종 seed 평균+혼합 확률과 최대 절대 오차 1e-6 이하의 일치를 확인한다. 16구 단위 스트리밍으로 생길 수 있는 수치 차이만 허용한다.
- May temperature, June ensemble blend와 각 seed의 June blend를 그대로 재사용한다. stress에서 재보정하거나 시나리오·강도를 고르지 않는다. 빈도 기준선의 입력은 이 stress에서 변하지 않는다.
- 과거 토큰의 노출·변경·누락 수와 비율은 query-history occurrence 단위로 계산한다. 400회 delivery 반복은 분모에 넣지 않으며 고유 과거 구 수도 함께 기록한다. 선수 context 누락은 별도 표시한다.

## 비교와 판정

1. 상대 강건성: 9개 비정상 시나리오 각각에서 후보−대조군의 NLL/Brier. 전체와 기존 12개 R 그룹을 평가한다.
2. 입력 안정성: 각 모델에서 9개 시나리오−그 모델의 clean NLL/Brier를 전체 표본에서 평가한다. 상대 강건성과 별도 상태로 보고한다.
3. 총 단측 상한 family는 `9 × ((전체+12그룹)+2모델) × 2지표 = 270`이다. 같은 경기 안 구를 함께 재표집하는 pitch-weighted percentile bootstrap, seed20260924, 200,000회, 각 상한 alpha=.05/270을 사용한다. 미관측 그룹도 family 슬롯을 유지한다.
4. 각 비교의 동시 상한이 ΔNLL≤.010, ΔBrier≤.002이면 해당 guardrail 통과다. 최소 30경기·500구 미달은 미측정이다. 하나라도 측정된 실패가 있으면 failed, 실패는 없지만 미측정이 있으면 unconfirmed, 모두 통과해야 passed다. Cpanel의 zero-TRAIN 그룹 부재를 전체 MLB 강건성으로 채우지 않는다.
5. 확률 예측의 절대 NLL/Brier·클래스별 값, seed 방향, 실제 변경 비율, fallback, 비용도 기록한다. 새로운 N 성공이나 독립 확인 성공으로 세지 않는다.

bootstrap은 학습·선택·보정 불확실성을 제외한 고정 예측 조건부 결과다. 극단 분위수에는 Monte Carlo 오차도 남는다(명목 꼬리 표본 약37개). stress는 가상의 입력 오류이며 현장 센서 오차의 경험적 분포가 아니다. unknown_pitcher는 encoder/routing 누락이고 delivery에는 기존 ID를 유지한다.

## 자원

단일 heavy lock, 개별 추론 7,200초 상한, family 8시간 예산이다. G 실제 추론 시간으로 60회 비용을 먼저 추정하고 첫 두 모델의 clean 추론으로 확인한다. 첫 두 clean 결과는 수치 동등성만 확인하며 후보 성능을 열지 않는다. 초과 예상 시 결과를 보기 전에 자원 계획을 수정한다. 표본·seed·draws를 몰래 줄이지 않는다.

구현: `run_ml_stress.py`, `score_ml_stress.py`, `matrix_stress_metrics.py`. 현재 코드·synthetic 검사 준비와 실제 실행 완료를 구분한다.

# COOP-022 — Fable 5.1 검열 재설계 검토 (COOP-021)

2026-09-30. 검토자 **Fable 5.1**(작성자 Opus 5.5와 다른 모델), 대상 [COOP-021](COOP-021-censoring-redesign-2026-09-30.md). 검토 원문은 총괄 세션을 거쳐 전달받았고, 아래는 그 요지를 정리한 것이다. 판정은 검토자의 것이며, 채택은 총괄 세션이 D116 자율 범위 안에서 결정했다.

## 판정: APPROVE-with-changes (blocker 0)

총괄 결정: **G-R 상한 = 0.005.** 상한 옆에는 통과 가능 효과 근사식을 함께 출력한다. 최악 경계로 남은 비율이 s이면 폭은 약 2s다.

## 필수 변경과 반영 위치

1. **런타임 v3**(`policy_runtime.py`, opt-in `positivity_record=True`; 기본값은 v2 pin을 바이트 단위로 유지)
   - 양성 판정을 후보 계산 뒤로 옮긴다.
   - 거절 행에 π·q̂와 `rho_candidate = rho_reference = 0.0`을 명시적으로 기록한다(0/0을 만들지 않음).
   - `mask[a]`가 참이거나 π_c[a]·π_r[a] 중 하나라도 0이 아니면 `IntegrityError`를 낸다.
   - 상태 이름, sticky `MID_PA`, `validity_violation`은 그대로 둔다.
   - 반영: `CONTRACT_V3`, `_evaluate`, `submit`, `build_runtime`/`build_reference_pair_runtime`.
2. **추정기**(`policy_estimator.py`, `censoring='l1r'`; 기본값 `'worst_case'`)
   - `_step`은 `logging[a] == 0`을 LOGGING_POSITIVITY이면서 두 π[a]가 0이고 기록된 ρ가 0인 경우에만 받는다(나눗셈 없음, `_positivity_step`).
   - 그런 PA는 점값이다(기울기 0, 보상 불필요).
   - L2 포함은 `OUTSIDE_POLICY_SUPPORT`와 일관되게 한다(COOP-019 F4). ρ만으로 미지값 없이 정해지는 PA, 즉 완료, 양성 ρ=0, 노드 전 ρ=0이 L2에 들어간다.
   - ESS 보고에 E0 전체 가중치를 넣는다(COOP-019 F3, `ess_all_e0`).
   - 기본값 경로는 봉인 S6-dr-a2를 재현한다. 검열 PA 463개의 경계를 봉인 원장에서 다시 계산한 최대 차는 0.0이다.
3. **등록**
   - 옵션 키: `S6_V4.estimator`, `mlb2026_ope.estimator`. 둘 다 `estimator_block`으로 검증한다.
   - G-R은 E0 기준으로 정의하고, 최악 경계 잔여로 세는 사유를 `WORST_CASE_RESIDUAL`에 명시한다.
   - S-C의 δ는 ≤2025 평가 행의 |v_c − v_r| q99 0.0080·q999 0.0278이다. 단일 노드 기준이라 여러 단계 차이를 과소 경계한다.
   - 민감도:
     - S-v1: D-5 최악 경계
     - S-B: 민감도 전용. 순서는 (game_date, game_pk, at_bat_number)이고, 부호 규칙을 미리 선언한다. S-B 라벨이 주 라벨과 다르면 보고서 첫 줄에 적되 판정은 주 추정치만 한다.
     - S-NP: `NO_LOGGED_ACTION`만 최악 경계로 두고 나머지는 L1-R
     - S-C
   - regime 문구는 계약 R6와 사전등록 §5a에 넣는다: "π를 첫 H_k-확정 사건까지, 그 뒤 로그 행동; V(π) 자체에 대한 판정 없음".
   - AUTOMATIC 확인: `preq.AUTOMATIC = {automatic_ball, automatic_strike}`이고, `logged_label`은 무투구 description을 구종보다 우선한다. S0 census(D97)에서 구종이 붙은 무투구 행은 0이다.
   - 가정 `P(no-pitch | H_k, type) = P(no-pitch | H_k)`를 명시한다.
4. **toy 검사**(`test_policy_validation.py`)
   - 양성 ρ=0 정확성: 참 b로 지원 행동의 ρ를 계산하면 기대 추정이 참값과 같다(1e-12).
   - (1−f) 편향식: π̂_b가 질량 f인 새 구종을 놓치면 정책별 오차가 정확히 −f·(V − v̂)다.
   - 도출한 크기: 401/44,230 × G0 TV q99 0.0058 ≈ **5e-5 WE/PA**(≤2025 S6 비율의 대략적 상한, 측정값 아님). 사전등록 §5a에 적는다.
5. **≤2025 S6 v3 재실행은 필수다.** 총괄이 큐로 실행하고, 그 뒤 M3 대리값을 실측값으로 바꾼다.

## 남은 확인

- D111은 b_V를 0.00272로 적었지만, 봉인 S5 `v2.json`에서 다시 계산하면 **0.0027347**(seed 0: 0.00098334 + 1.96·0.00089356)이다. 따라서 `IMPROVEMENT_SUPPORTED` 문턱은 0.003735다. 등록값은 계산값과 1e-12 안에서 같아야 한다(`run_ope_2026`이 강제).

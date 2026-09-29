# COOP-018 결정 선택지와 사용자 결정 (D93)

2026-09-29. Opus 5.5 단독(사용자 지시). 결정마다 세 렌즈(인과·통계 타당성, 구현·자료 위험, 계약·노출 정합성)가 독립 분석하고, 종합 1회, 반박 담당 1회를 거쳤다(11개 모두 반박 불성립). 같은 모델의 검토이며 독립 검토가 아니다. 사용자 결정: **D-1~D-10과 누락 결정 M-1~M-13은 추천안, D-11은 D안.** 실행 승인은 아니다. 기계 기록은 `configs/ML-POLICY-MATERIALIZATION-v1.json`의 `decisions`·`decisions_extra`.

## D-1 TRAIN BC 적합 모집단

- **사용자 결정:** A. BC-P 단일 artifact (추천). BC-E는 P8 재현 게이트와 기술 비교 전용
- 추천: A. BC-P 단일 artifact (추천). BC-E는 P8 재현 게이트와 기술 비교 전용 · 확신도 high · 반박 담당 판정: 반박 불성립

**A. BC-P 단일 artifact (추천). BC-E는 P8 재현 게이트와 기술 비교 전용** — TRAIN(2023-05-15..2025-04-30) 정규시즌 행 가운데 로그 행동이 있는 투구 전부로 CategoricalBC(prior_strength=20, minimum_action_count=1)를 하나 적합한다. 조건은 pitch_type 비결측, description ∉ {automatic_ball, automatic_strike}, 합법 카운트다. 결과 라벨·supported_pa·plate 좌표 조건은 쓰지 않는다. 이 BC 하나를 π̂_b(전체 어휘), MaskedReference(π_ref), P3 후보의 바탕, 지원 표 M에 모두 쓴다. D91/D92 런타임 구조를 그대로 쓰는 셈이다(policy_runtime.py:158-160에서 logging_law와 reference가 같은 bc_sha256). BC-E는 p4_train_keys(1,252,824구, ordered rows 0a2e81ae…)로 따로 내보내 두 용도로만 쓴다. (1) S1 fail-closed 게이트: P8 재현, 포함, 셀 단조. (2) TRAIN 셀 차이 기술표. 추정·선택·민감도에는 쓰지 않는다.

  - 장점: 처치 후 선택이 없다. 필터는 '투구가 있었고 라벨이 있는가'와 투구 전 카운트뿐이다. BC-E의 supported_pa(data.py add_transitions_and_support)는 PA 종료 사건, PA 중 도루·폭투·포일, 번트 description, 교체, 연장 이닝(inning 1–9만 허용), 경기 마지막 PA 같은 결정 뒤 정보로 PA 전체를 뺀다 / D-2(R1: 모든 정규시즌 PA 분모)의 TRAIN판이다. 연장, 주자가 움직인 PA, 번트 PA에서도 π̂_b가 적합 범위 밖 외삽이 아니다. D89 §5가 q̂^cand:=Q̂^ref 제어변량 아래 후보 항의 불편성을 π̂_b 경로 하나에 맡기므로, 이 성질이 가장 직접 걸린다 / 필터가 WIP 요청 생성기의 R3(c) 조건과 같다. policy_requests.logged_label(commit a704983, 미검토)이 automatic 또는 결측을 NO_PITCH로 보낸다. 그래서 적합 모집단과 질의 모집단의 원칙이 하나로 맞는다
  - 단점: 기준·후보 정책의 정의가 P8 BC와 달라진다. 그래서 P8 수치(τ=.003, P0–P3 값, coverage)와 직접 비교할 수 없다. 다만 G2→G0 전환으로 τ는 어차피 다시 골라야 한다(D-9) / BC-P 행 수, 제외 사유 구성, 어휘가 미측정이다. 추가 행의 산술 상한은 1,386,362−1,252,824 = 133,538행이지만, 이 안에는 BC-P도 제외하는 결측·automatic·불법 카운트 행이 섞여 있다 / 지원이 넓어진다. minimum_action_count=1이므로 비eligible 행에서 한 번만 나온 코드(UN·PO·FA·EP, 오분류)도 투수 지원에 들어간다. 그 투수에 eligible 투구가 0이라도 JointDelivery의 리그·유형 tier pool을 통해 M에 들어갈 수 있다(수 미측정)
  - 추정량·분모 변경: 예, 구현 비용: medium

**B. BC-E 단일 artifact (P8 호환 유지)** — pin된 eligible D100 TRAIN 키 p4_train_keys(1,252,824구, 18구종, keys sha 372d0aaf…)를 select_keys로 재구성한다. 그 행으로 적합한 BC 하나를 π̂_b, π_ref, 후보 바탕, M에 모두 쓴다. run_ml_policy.py prepare의 fit_bc(store, parts['train'])와 같은 모집단이다. BC-P는 보고용으로만 둔다.

  - 장점: 행 키가 이미 pin되어 있다. select_keys는 개수나 순서 해시가 다르면 ValueError를 낸다. 새 선택 로직이 없고 S0를 기다리지 않아도 된다 / P8의 bc_train_pitches와 18구종을 그대로 재현한다. G0, frequency, pool과 같은 적합 모집단이라 모형 세계 안에서 일관된다 / 단일 BC라서 A와 같은 비 상쇄 구조를 유지한다
  - 단점: π̂_b가 π_b(a|H, S=1)이 된다. 여기서 S=supported_pa는 결정 뒤 사건이다. 폭투·포일(원바운드 변화구), 도루, 번트, 실책 종료처럼 구종과 연관될 수 있는 사건 때문에 특정 구종이 체계적으로 과소 추정될 수 있다(크기·방향 미측정). 후보 항이 π̂_b 경로에만 기대므로 이 편향이 Δ로 1차 전달된다 / D-2(R1)와 모순된다. 적합 범위가 아닌 연장, 주자 이동 PA, 번트 PA에 외삽하게 된다. 적합과 정합하는 분모(eligible)는 정책마다 달라지는 사후 집합이라 D87 §2가 금지한다 / 비eligible PA에서만 관측된 투수·구종은 DEV·2026에서 UNKNOWN_PITCHER나 LOGGING_POSITIVITY로 거절된다. 거절 구성이 결정 뒤 정보에 의존하게 된다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**C. 역할 분리 (π̂_b = BC-P, π_ref·M·P3 바탕 = BC-E)** — 기준·후보 정책 쌍의 정의(추정 대상)는 P8과 호환되는 eligible 기반으로 둔다. 로그 nuisance π̂_b만 BC-P로 바꿔 처치 후 선택을 없앤다. 해시가 다른 두 BC artifact를 각각 pin한다.

  - 장점: P8의 기준 정책 정의를 유지하면서 nuisance의 사후 선택을 제거한다 / BC-E ⊆ BC-P이고 minimum_action_count=1이므로 π_ref(a)>0 ⇒ π̂_b(a)>0이 구조적으로 성립한다(S1 확인 대상) / D89 §5의 '기준과 다른 이름·해시의 별도 identity' 문구를 글자 그대로 구현한다
  - 단점: 비 상쇄가 깨진다. ρ^ref = π_ref^E(a)/π̂_b^P(a)가 M이 로그 질량을 모두 덮어도 1이 아니다. D89 §5의 ρ^ref≡1 조건과 합성 검사 (j)의 전제가 무너지고 분산이 커질 수 있다(미측정) / PolicyRuntime.__init__, build_runtime, support_table_payload(bc_sha256·vocabulary_sha256 결속), policy_identity의 bc 절, bind의 inputs.bc 검사, ledger header, 테스트를 모두 바꿔야 한다. ML-POLICY-RUNTIME-v1 계약 개정, 새 합성 검사, decisions 기록도 필요하다 / 기준 정책이 사후 선택된 모집단의 습관이 되어 '투수가 평소 하는 것'이라는 해석이 약해진다
  - 추정량·분모 변경: 예, 구현 비용: high

**D. BC-R: 요청 생성기 거울 모집단** — 요청 생성기의 결정 전 거절 규칙을 TRAIN에 logging-only로 적용한다. 대상 규칙은 R2 순서, R3(c) NO_LOGGED_ACTION sticky, R6 INCOMPLETE_START, R8 INCONSISTENT_HISTORY다. 그 뒤 런타임이 실제로 π̂_b를 질의할 결정만으로 BC를 적합한다. UNKNOWN_PITCHER·EMPTY_SUPPORT·LOGGING_POSITIVITY는 BC 자신에 의존하므로(순환) 적용하지 않는다.

  - 장점: 적합 분포와 질의 분포가 연도 이동을 빼면 정확히 같다. '<UNKNOWN>'/NO_PITCH 직전 구종 셀도 사라진다 / 결정 전 정보만 쓰므로 처치 후 선택이 없다 / 요청 생성기를 DEV보다 먼저 TRAIN 1.39M행에 돌리게 되어 통합 시험 역할을 한다
  - 단점: 요청 생성기(commit a704983, WIP·미검토)가 검토·등록되어야 S1을 돌릴 수 있다 / BC sha가 D-3·D-4·D-5·R8 결정과 생성기 코드 버전에 묶인다. 규칙 하나가 바뀌면 BC, 지원 표, τ, 정책 식별자를 모두 다시 만들어야 하고, 계약 §2의 '생성기는 식별자 밖' 경계도 바꿔야 한다 / 거절 뒤 행도 BC 키(투수, 카운트, 타자면, 직전 구종) 아래에서는 유효한 로그 표본이다. 빼도 편향은 줄지 않고 표본만 준다(차이 미측정)
  - 추정량·분모 변경: 예, 구현 비용: high

**근거(요약).** 세 렌즈(인과·통계, 구현·자료, 계약·노출)가 독립적으로 모두 A를 골랐다. 근거는 네 가지다.

**추천이 바뀔 조건.** - D-2에서 R1(모든 정규시즌 PA)을 버리고 eligible에 가까운 분모를 채택하는 경우. 이는 D87 §2와 충돌하지만, 그렇게 되면 B의 적합·분모 정합 논리가 살아난다. 처치 후 선택 문제는 그대로 남는다. - S0/S1에서 BC-E ⊄ BC-P, BC-P 어휘 ≠ token 어휘 V(18), 또는 BC-E가 P8의 1,252,824구·18구종을 재현하지 못하는 경우. 이때는 BC-E로 갈아타지 않는다. FAILED_INTEGRITY로 멈추고 원인을 찾은 뒤 재등록한다. - S1에서 M_P ≠ M_E 항목(eligible 투구 0개인 투수×구종이 리그·유형 tier pool로 M에 들어온 경우)이나 UN·PO·FA·EP가 지원에 상당수 들어가고, 이것이 정당화되지 않는 경우. 이때는 먼저 코드 제외 목록이나 투수 수준 최소 횟수를 별도 결정으로 등록하는 것을 검토한다. C는 그다음 후보다. - D-10(M 밖 행동의 V2 delivery 정의)이 로깅 어휘와 개입 어휘의 분리를 요구하는 경우. 그러면 C의 역할 분리를 다시 검토한다. - 보고상 기준 정책이 P8과 정확히 같아야 한다는 외부 요구가 생기는 경우. 이때는 C를 검토한다. - 독립 검토(Sol·Astra)가 BC-P 필터에 결정 뒤 정보가 섞였다고 지적하는 경우. 예를 들어 automatic 판정의 성격이나 description 사용 방식이다. - 주 선택을 바꾸려면 DEV(V4·V5)나 2026 추정 결과를 보기 전에 새 결정으로 등록해야 한다. S1의 BC-P/BC-E TRAIN 차이표를 보고 바꾸는 것은 사전 고정 위반이다.

## D-2 평가 분모: 전 PA 계층 + 투구 전 시작 모집단

- **사용자 결정:** 2. 계층형: R1 전 PA 보고 분모 + 투구 전 정의 시작 모집단 S를 주 제한 추정량, supported/eligible은 사후 기술 층
- 추천: 2. 계층형: R1 전 PA 보고 분모 + 투구 전 정의 시작 모집단 S를 주 제한 추정량, supported/eligible은 사후 기술 층 · 확신도 medium · 반박 담당 판정: 반박 불성립

**1. R1 단독(현재 제안): 모든 정규시즌 PA 분모, 평가 가능 PA 조건부 점추정 + 전 PA [0,1] 경계** — 평가 창의 R 경기 PA `(game_pk, at_bat_number)`를 모두 요청 분모로 둔다. supported_pa와 eligible은 사후 층으로만 쓴다. 점추정은 policy_estimator.estimate의 `conditional`로, COMPLETE PA의 짝 Δ 평균이다. 나머지 PA는 정책마다 [0,1] 경계, 즉 PA당 Δ ±1로 `population_delta_bounds`에 넣는다. 투구 전 정보로 정한 제한 모집단 층은 따로 두지 않는다.

  - 장점: 결과 사건(unsupported_terminal_event), 경기 완결(incomplete_game), PA 중 상태 변화(mid_pa_state_change: 폭투·포일·도루)처럼 결정 뒤 정보로 PA를 거르지 않는다. D87 config `drop_PAs_by_later_observed_support_or_completion=false`에 맞는다. / supported_pa가 구조적으로 뺐던 PA가 들어온다. 경기 마지막 PA(missing_terminal_next_state, TRAIN 감사 4,673타석)와 연장 이닝(data.py 274행 `inning.between(1, 9)`)이 여기에 해당한다. ≤2025 리허설이 2026 주 모집단 `full_regular_season`과 같은 모양이 된다. / WIP 커밋 a704983의 pa_requests, pa_status, estimate, run_dr가 이미 이 구조를 전제한다. 층을 추가하는 작업이 없다.
  - 단점: D87 분모 사슬의 `eligible_PA_starts` 층을 '전부'로 접는다. 그래서 config `coverage_and_selection_bias.report`의 `restricted_predecision_population_estimand_named_separately`가 비어 있다. / `conditional`은 로깅 궤적에서 일어난 사건에 조건을 건다. 첫 투구 뒤 NO_LOGGED_ACTION, PA 중 모르는 투수로의 교체(sticky MID_PA), INCONSISTENT_HISTORY, no_terminal_event가 그렇다. 투구 전으로 정의되는 어떤 모집단의 V(cand)−V(ref)도 아니며 기술 통계일 뿐이다. / 경계 폭은 2×(평가 불가 PA 비율)이다(estimate 150행 식). 이 비율에 모르는 시작 투수, 빈 지원처럼 정당한 투구 전 제한까지 섞여 경계가 정보 없는 폭이 될 가능성이 크다. R1 기준 비율은 미측정이다. 참고로 기존 supported_pa 제외는 TRAIN 31,243/355,625타석(약 8.8%)이지만 다른 집합이다.
  - 추정량·분모 변경: 예, 구현 비용: medium

**2. 계층형: R1 전 PA 보고 분모 + 투구 전 정의 시작 모집단 S를 주 제한 추정량, supported/eligible은 사후 기술 층** — D87 config `denominator_chain`의 5층을 그대로 채운다. (1) all_R_games → collected_games → 모든 R 경기 PA. unsubmittable 포함 전 PA [0,1] 경계를 둔다(옵션 1의 분모). (2) eligible_PA_starts = S. S는 PA 첫 기록 행의 투구 전 필드와 동결 TRAIN 산출물만의 순수 함수다. 조건은 game_type R, 수집 경기, 첫 행 0-0, pitcher·stand 비결측, `CategoricalBC.fallback=False`, 첫 행 맥락에서 reference 마스크가 비어 있지 않음이다. logged_action, description, events, 이후 행, supported_pa, complete_game, next_*는 쓰지 않는다. 연장 이닝은 기본으로 포함하고 층으로 보고한다. (3) supported_requests → evaluable_PAs. 주 보고는 'S 제한 모집단' 추정량이다. S 안의 평가 가능 PA 조건부 점추정(좁은 이름)과 S 안 [0,1] 경계로 이루어진다. S를 정한 뒤의 거절(첫 행 NO_LOGGED_ACTION 포함), 교체, 종료 없음은 S에서 빼지 않고 S 안의 경계로 보낸다. supported_pa·eligible·P8 패널 층은 같은 원장에서 '선택 편향 가능' 라벨을 단 기술 표로만 낸다. ρ 규칙(D89 §2)은 바꾸지 않는다.

  - 장점: D87 config의 `denominator_chain`, `start_population: fixed_predecision_PA_starts_defined_from_predecision_fields`, `restricted_predecision_population_estimand_named_separately`, `full_population_causal_value: null_or_0_1_worst_case_bounds`, `drop_PAs_by_later_observed_support_or_completion=false`를 동시에 만족한다. D87·D89를 번복하지 않고 구체화만 하므로 decisions.md 새 한 줄로 충분하다. / S는 결정 전 정보로만 정해지는 처치 전 부분 모집단이다. 정당한 투구 전 제한(모르는 시작 투수, 빈 지원)은 S 밖으로 가고, 첫 투구 뒤 평가 불가만 S 안 경계 폭에 들어간다. 그래서 옵션 1보다 정보가 있는 구간이 나올 수 있다. 크기는 미측정이다. / 전 PA 경계(옵션 1)와 사후 supported 층(옵션 4의 연속성)이 같은 원장에서 함께 나오므로 정보를 잃지 않는다.
  - 단점: S는 TRAIN 이력 투수 쪽으로 기운 제한 추정량이다(D87 §6의 베테랑 쪽 치우침). MLB 전체 주장으로 쓸 수 없고 전 PA 경계를 반드시 병기해야 한다. 2026에서는 TRAIN이 2025-04-30에 끝나므로 S 밖 비율이 DEV보다 클 수 있다(미측정). / S 안 조건부 점추정에도 첫 투구 뒤 사건에 의한 선택이 남는다(PA 중 교체, NO_LOGGED_ACTION sticky, 종료 없음). 좁은 이름과 S 안 경계로만 다룰 수 있고, 편향이 없다고 주장할 수 없다. / runtime._evaluate의 거절 순서는 NO_LOGGED_ACTION → INCOMPLETE_START → INCONSISTENT_HISTORY → UNKNOWN_PITCHER → EMPTY_SUPPORT다. 원장 status로 S를 정하면 순서에 따라 모집단이 달라지므로 독립 함수로 따로 계산하고 테스트해야 한다.
  - 추정량·분모 변경: 예, 구현 비용: medium

**3. 옵션 2 + 통과 단계: 결정 없음·PA 중 미지원 상태는 ρ=1 인계, π̂_b(a)=0이고 a∉M이면 ρ=0** — 분모와 S는 옵션 2와 같다. 추가로 S를 정한 뒤 도달하는 두 종류의 단계를 거절하지 않고 '기록 투수 인계' 단계로 둔다. 하나는 투수 선택이 없는 행(automatic_ball·strike)이고, 다른 하나는 결정 전에 알 수 있는 미지원 상태(PA 중 모르는 새 투수, 빈 마스크)다. 두 단계에서 π_e'(·|H)=π_b(·|H)로 정의하므로 ρ=1이 정확하고 DR 항은 V_t=r_t+V_{t+1}이다. 로그 행동이 마스크 밖이고 π̂_b(a)=0이면 거절 대신 ρ=0(OUTSIDE_POLICY_SUPPORT와 같은 처리)으로 둔다. 경계는 자료 결함(종료 없음, 이력 불일치)에만 남긴다. 추정량 이름은 '후보 + 기록 투수 인계 정책'이다.

  - 장점: S 안 점추정에서 첫 투구 뒤 조건화가 대부분 사라진다. 투구 전으로 정의된 모집단의 반사실 대비에 가장 가깝다. / D-3(무행동 행), D-4(교체), LOGGING_POSITIVITY를 '결정이 없거나 정책이 정의되지 않는 단계는 통과'라는 한 원리로 정리한다. 추천이 없을 때 배터리가 스스로 정하는 서비스 현실과도 맞는다. / 경계로 가는 PA가 자료 결함으로 줄어 경계가 가장 좁아질 수 있다. 크기는 미측정이다.
  - 단점: D87 config `unsupported_request_policy.rule: 'realizable frozen fallback law, registered only if identifiable; never observed action or ratio=1'`과 D89 §2의 'ρ=1로 두지 않음'을 정면으로 뒤집는다. D-2 범위를 넘는 계약 번복이어서 별도 decisions.md 기록과 D87·D89 개정이 필요하다. / ρ=0 전환은 a∉M일 때만 타당하다. a∈M인데 π̂_b(a)=0이면 진짜 positivity 위반이어서 거절을 유지해야 한다. 현재 코드(policy_runtime 246행)는 두 경우를 구분하지 않고 mask[a] 확인 전에 거절한다. / 인계 단계도 순차 교환 가능성(감독 교체와 새 투수의 선택이 H_t만의 함수)에 기대며 입증되지 않았다(D89 §5 (ii)). 인계 뒤 이력에 NO_PITCH 토큰이 있을 때 BC의 직전 구종 조회가 어떻게 동작하는지 검증되지 않았다. 합성 toy(D89 §8)도 통과 단계를 열거하지 않는다.
  - 추정량·분모 변경: 예, 구현 비용: high

**4. P8·G0 호환: supported_pa ∧ D100 eligible PA만 분모** — pin된 eligible 키에 해당하는 PA만 요청으로 제출한다. TRAIN 1,252,824구, DEV 311,721구·1,161경기다. 점추정과 경계를 모두 이 모집단 기준으로 내고, 전 PA는 개수로만 보고한다. 추정량 이름은 '사후 완전·지원 PA 조건부'다.

  - 장점: 구현이 가장 싸다. 키와 순서가 rows_sha256으로 pin되어 있고, automatic 행·결측 구종·연장·경기 마지막 PA가 미리 빠진다. 그래서 FAILED_INTEGRITY, 미제출, pa_outcome의 경기 끝 경로를 거의 타지 않는다. / G0·BC-E·delivery 학습 모집단과 같아 q̂ 외삽 위험이 가장 작다.
  - 단점: 결정 뒤 정보로 PA를 고른다. 대상은 종료 사건 종류, 카운트 경로 일관성, post 점수 불일치, 경기 완결, 구종에 영향받을 수 있는 PA 중 폭투·포일이다. π_e 아래에서는 '나중에 지원될 확률'이 달라지므로 방향과 크기를 모르는 편향이 생긴다. / D87 시작 모집단 문구, config `drop_PAs_by_later_observed_support_or_completion=false`, D89 §2 분모 규칙을 뒤집는다. 주 추정량으로 쓰려면 D87 번복 기록이 필요하다. 2026 요청 경로(R3(c), R6)도 검증하지 못한다. / DEV 341,941행 중 30,220행이 빠진다(산술). 제외분에는 WE 레버리지가 가장 큰 경기 마지막 PA와 연장 이닝이 구조적으로 포함된다.
  - 추정량·분모 변경: 아니오, 구현 비용: low

**근거(요약).** 세 렌즈는 두 가지에 모두 동의한다. 첫째, supported_pa와 eligible로 요청을 고르면 안 된다(옵션 4 기각). 둘째, P8·G0와의 수치 비교는 어떤 분모로도 성립하지 않는다(P8은 모형 내부, G0은 투구 단위 NLL). 남은 질문은 전 PA만 볼지(옵션 1), 그 안에 투구 전 제한 모집단을 둘지(옵션 2), 거기서 더 나아가 ρ 규칙까지 바꿀지(옵션 3)다.

**추천이 바뀔 조건.** - D87 config `unsupported_request_policy`와 D89 §2 개정(인계 단계 ρ=1, a∉M이고 π̂_b(a)=0이면 ρ=0)이 별도 결정으로 승인되면 옵션 3으로 옮긴다. 이때 승인과 등록은 S4·S6 결과 열람 전이어야 한다. - S 판정 함수를 원장 status와 독립으로 구현하고 불변성 테스트를 통과시키지 못하면 옵션 1로 물러선다. 예를 들어 첫 행 맥락에서 reference 마스크를 투구 전 정보만으로 재현할 수 없는 경우다. 이때 `conditional`은 기술 통계로 강등하고, 머리 결론은 전 PA 경계로 두며, D87의 제한 추정량 요구는 '미충족'으로 명시한다. - D-1에서 BC-E(eligible 적합)가 주 nuisance로 정해지면 분모와 nuisance 모집단이 어긋난다. 이 경우 옵션 2를 유지하되 BC-P를 주로 바꾸라고 되돌려 제안한다. 옵션 4로 가지는 않는다. - 사용자가 D52 예측 트랙을 위해 P8·G0 표와의 연속성을 주 요구로 둔다면, 옵션 2의 supported 기술 층 비중을 키운다. 이 경우에도 옵션 4를 주 추정량으로 쓰는 것은 D87 번복 기록 없이는 불가하다. - S4 V5 수치(S 안 경계 폭, S 밖 비율)는 추천을 바꾸는 근거로 쓰면 안 된다. 쓰면 결과 의존 모집단 선택이 된다. 다만 S0 전에 등록한 문턱(예: 'S 안 경계 폭이 X를 넘으면 V4를 결론 없음으로 보고')은 미리 정해 둘 수 있다. X의 값은 정해지지 않았다.

## D-3 투구 없는 행 처리: 원인별로 나눠 거절

- **사용자 결정:** A′. 원인 분리 거절 + sticky + 자료 가드: 자동 판정은 UNSUPPORTED_NO_LOGGED_ACTION, 실제 투구의 라벨 결측은 UNSUPPORTED_MISSING_ACTION_LABEL
- 추천: A′. 원인 분리 거절 + sticky + 자료 가드: 자동 판정은 UNSUPPORTED_NO_LOGGED_ACTION, 실제 투구의 라벨 결측은 UNSUPPORTED_MISSING_ACTION_LABEL · 확신도 medium · 반박 담당 판정: 반박 불성립

**A. 현재 제안 유지: 단일 거절 상태 UNSUPPORTED_NO_LOGGED_ACTION + sticky(a704983 그대로)** — description이 automatic_ball/strike인 행과 pitch_type이 결측이거나 빈 문자열인 행에 모두 '<NO_PITCH>'를 붙인다(policy_requests.logged_label, 31-35행). 이 라벨은 그 행의 로그 행동이면서 이후 이력의 행동도 된다. 런타임은 해당 요청을 UNSUPPORTED_NO_LOGGED_ACTION으로 거절하고(policy_runtime._evaluate 210-211행), 같은 PA의 이후 요청은 UNSUPPORTED_MID_PA로 둔다. PA는 R1 분모에 남아 [0,1] 최악 경계로 들어가며, 평가 가능 PA 조건부 점추정(D-5)에서는 빠진다.

  - 장점: R1, D87 §2·§6(결측·미지원은 분모에 남긴다), D89 §2(관측 행동 대입이나 ρ=1 금지)와 모두 맞는다. 전체 모집단 Δ 경계는 결측 원인과 관계없이 유효하다. / 새 식별 가정이 없다. 거절 뒤에는 BC·G0를 부르지 않으므로 G0가 학습 때 본 적 없는 이력(구종 0, outcome 'unknown', 결측 물리)이 들어가지 않는다. 후보 MC Q 비용도 늘지 않는다. / 이미 WIP 커밋에 구현돼 있고 합성 검사도 있다(tests/test_policy_validation.py 117-133·161-165행). 추가 비용이 거의 없다.
  - 단점: 두 원인이 한 상태로 섞인다. 테스트 165행이 logged_label(NaN,'ball')==NO_PITCH를 고정해 두었기 때문에, 실제로 던졌지만 구종 분류만 빠진 공이 '투수 선택 없음'(policy_runtime.py 32행 주석)으로 보고된다. 그래서 커버리지 현상(피치클록 판정)과 자료 품질 결함(추적 실패)을 D87 §2의 사유 보고에서 구분할 수 없다. / PolicyRuntime.summary()는 decision_index>0에서 나온 거절을 PA 수준에서 MID_PA로 접는다(316행). S4 V5의 '무행동 PA 비율'이 런타임 요약에 나오지 않고, 첫 원인은 policy_estimator.pa_status의 refused[0]에만 남는다. / census()는 missing_pitch_type_rows와 automatic_call_rows를 따로만 세고 교차표를 만들지 않는다. 그래서 영향받는 PA 수와 첫 무행동 행의 위치를 알 수 없다.
  - 추정량·분모 변경: 아니오, 구현 비용: low

**A′. 원인 분리 거절 + sticky + 자료 가드: 자동 판정은 UNSUPPORTED_NO_LOGGED_ACTION, 실제 투구의 라벨 결측은 UNSUPPORTED_MISSING_ACTION_LABEL** — 거절, 분모 유지, sticky, [0,1] 경계는 A와 똑같이 두고 네 가지를 더한다. (1) 원인을 나눈다. description이 등록된 무투구 판정 목록(현재 automatic_ball/strike)에 있으면 UNSUPPORTED_NO_LOGGED_ACTION, 이력 sentinel '<NO_PITCH>'. pitch_type이 결측이고 description이 무투구 목록 밖이면 새 UNSUPPORTED_MISSING_ACTION_LABEL, sentinel '<MISSING_LABEL>'. 비결측 어휘 밖 코드는 지금처럼 FAILED_INTEGRITY(R3(b))다. 둘 다 sticky이고 원장 직전 행동 검사는 각 sentinel을 그대로 대조한다. (2) PA 수준 첫 거절 원인을 런타임 summary와 추정기 양쪽에 남긴다. (3) S0 census를 label-blind 구조 수 범위 안에서 넓힌다. 교차표, 영향 PA 수, 첫 무행동 위치(첫 행 / 중간 / PA 마지막 행), 실제 결정이 0개인 PA 수, description 허용 목록 밖 행 수를 센다. (4) 무투구 목록과 판정 우선순위는 config에 등록하고 stage manifest에 pin한다. 규칙은 S0 구조 수를 본 뒤까지만 조정할 수 있고, S4·S6 결과 값을 본 뒤에는 바꾸지 않는다고 지금 고정한다.

  - 장점: 추정량, R1 분모, 평가 가능 PA 층은 A와 같다. 거절되는 PA 집합이 동일하고 사유 표시만 달라진다. 그래서 A의 계약 정합성(D87 §2·§6, D89 §2, R1)과 G0 외삽 회피를 그대로 가진다. / D87 §2의 사유 보고 요구를 채운다. D89 §2가 로그 행동 상태를 원인별로 나눈 원칙과도 같은 방식이다. 2026 M3 QA에서도 피치클록 판정 이동과 추적 결측을 같은 틀로 나눠 볼 수 있다. / census의 위치 분리로 A의 처치 후 선택이 어디서 생기는지(PA 중간 자동 판정)를 수로 드러낸다. 이 수가 나중에 통과 처리(B)를 새 식별자로 등록할지 판단하는 근거가 된다.
  - 단점: 런타임 상태가 하나 늘어나는 계약 변경이다. ML-POLICY-RUNTIME-v1 상태 목록, D89 §2 표, decisions.md 한 줄, 테스트를 함께 바꿔야 한다. 요청 fingerprint와 원장 스키마 버전도 바뀐다. / A와 같이 보수적이다. PA를 끝내는 자동 판정(볼넷·삼진 확정)이나 PA 중간 자동 판정 하나만 있어도 PA 전체가 경계로 간다. 경계 폭은 발생 비율이 미측정이라 모른다. / 조건부 점추정의 궤적 선택은 이름과 보고로만 다룬다. 없애지는 못한다.
  - 추정량·분모 변경: 아니오, 구현 비용: low

**B. 자동 판정 통과(비결정 환경 전이), 실제 투구의 라벨 결측만 거절 + sticky** — automatic_ball/strike 행은 결정이 아닌 환경 전이로 본다. 원장에는 비거절·비평가 상태로 한 행을 기록하고, 두 정책 모두 행동이 없으므로 DR 재귀에서 V_t = r_t + V_{t+1}로 통과시킨다. PA의 이후 결정은 계속 평가한다. 자동 판정이 PA를 끝내면 보상은 그 행에 붙는다. pitch_type 결측 실제 투구는 A′처럼 거절하고 sticky로 둔다. 정확히 두 자동 코드만 통과시키는 fail-closed 규칙을 둔다. 변형으로, PA를 끝내는 자동 판정만 통과시키는 방식도 있다.

  - 장점: 자동 판정 PA가 평가 가능 층에 남는다. 그래서 [0,1] 경계가 좁아지고, 조건부 점추정의 모집단이 D87 'PA 시작' 모집단에 가까워진다. / 투구 없는 전이를 환경 동역학으로 보는 순차 DR의 표준 처리다. R7(알려진 투수 교체)과 원리가 같다. 통과 가정이 성립하면 ρ≡1이라 분산도 늘지 않는다. / 요청 분모와 결정 분모를 나눠 보고할 수 있다.
  - 단점: 새 식별 가정이 필요하고 검정할 수 없다. H가 주어졌을 때 자동 판정 발생이 의도한 구종·정책과 독립이어야 한다. 사인 교환이나 PitchCom 지연으로 이 가정이 위반되면 편향의 부호와 크기는 미측정이다. / D89 §1(결정 시점)과 §5 재귀, D91 원장 불변식(decision_index==len(history), EVALUATED 의미), 추정기 pa_dr/_step을 개정해야 한다. D87의 '평가 시작 = 0-0 첫 투구' 문구도 고쳐야 한다. / BC 키가 조용히 어긋난다. BC-P 적합 경로(matrix_policy.state_from_row의 bc_only, 46행)는 결측 직전 행을 '<UNKNOWN>'으로, 비결측 automatic 행은 실제 코드로 두는데, 요청 경로는 '<NO_PITCH>'를 쓴다. 그러면 CategoricalBC 셀 조회가 오류 없이 투수 prior로 떨어져 π̂_b가 적합한 법칙과 달라진다. 이는 후보 DR의 유일한 식별 경로를 약하게 만든다.
  - 추정량·분모 변경: 예, 구현 비용: high

**D. 해당 PA를 요청 전에 제외하거나 실행을 중단(P8·G0 eligible 방식)** — 자동 판정이나 pitch_type 결측 행이 하나라도 있는 PA를 supported_pa처럼 제출 전에 빼서 분모와 경계에서 모두 없앤다. 다른 변형은 a704983 이전처럼 FAILED_INTEGRITY로 실행 전체를 멈추는 것이다.

  - 장점: 구현이 가장 단순하고 새 상태가 없다. / 이 원인 축에 한해 P8·G0 eligible 모집단(DEV eligible 311,721구)과 가장 비슷하다.
  - 단점: 처치 후 선택이다. PA 중간이나 끝에서 관측된 사건으로 PA를 빼므로 D87 §2·§6, R1, D-2 제안과 정면으로 충돌한다. 전체 모집단 경계가 더 이상 전체 모집단의 경계가 아니다. / 중단 변형은 실데이터에 무행동 행이 한 건만 있어도 실행이 불가능하다(빈도 미측정). / 2026에서는 제외율 이동(피치클록·ABS 체제 변화)이 분모에서 보이지 않게 되어 결과와 섞인다.
  - 추정량·분모 변경: 예, 구현 비용: low

**근거(요약).** 세 렌즈 모두 D(사전 제외·중단)는 R1과 D87 §2·§6을 깨는 처치 후 선택이라 기각했다. 또 세 렌즈 모두 현재 코드(a704983)가 서로 다른 두 원인을 한 상태로 섞는 것이 결함이라고 봤다. 자동 판정은 투수 선택이 없는 경우이고, pitch_type만 결측인 실제 투구는 선택은 있었지만 라벨이 없는 경우다. 테스트 165행의 logged_label(NaN,'ball')==NO_PITCH가 이 혼동을 굳혀 두었다. 그러니 원인은 어느 안을 고르든 나눠야 하고, 남은 쟁점은 자동 판정을 거절할지(A′) 통과시킬지(B)다.

**추천이 바뀔 조건.** - S0 census에서 PA 중간이나 PA 종료 위치의 자동 판정 PA가 경계를 쓸모없게 만들 만큼 많다면 B를 새 정책 식별자로 등록할지 따로 결정한다. 그 경우 G0·BC 토큰 정렬, τ 재선정, 통과 가정 합성 민감도가 함께 필요하다. 이 문턱은 S0 전에 Song이 정해야 하고 지금 값은 null이다. S4·S6 결과 값을 본 뒤에는 정하지 않는다. - G0 전처리와 BC-P 적합 규칙을 바꾸지 않고도 통과 뒤 이력을 학습 분포 안에 둘 방법이 코드로 확인되면 B의 비용이 크게 줄어 재검토한다. 예를 들어 이미 동결된 입력 규칙이 자동 판정 행을 건너뛰는 이력과 일치하는 경우다. - S0에서 'pitch_type 결측이면서 무투구 목록 밖 description'인 행이 TRAIN과 DEV 모두 0이면 A′의 새 상태는 비어 있게 된다. 그래도 2026 가드로 유지할 가치는 있지만, 계약 변경을 줄이려고 A로 되돌리는 것도 합리적이다. - a704983의 적대적 코드 검토에서 원장 체인·sticky·summary 처리가 틀린 것으로 밝혀지면, D-3 결정 전에 그 수정이 먼저다. - 레포 문서로 확인되지 않은 Statcast 기록 방식이 있다. 무투구 고의4구가 어떻게 기록되는지, 2026 ABS 번복 기록 방식(BLK-06)이 어떤지다. 이것이 새 description 코드나 결정 0 거절을 대량으로 만든다고 확인되면 무투구 목록과 규칙을 재등록해야 한다. 2026 코드 조사는 M3 QA에서만 한다.

## D-4 PA 중 투수·타자 교체 처리

- **사용자 결정:** 2. 계속 평가(주) + 사전 등록 교체 기록·census·기술 층 + '교체 뒤 자연 경과' 2차 추정량
- 추천: 2. 계속 평가(주) + 사전 등록 교체 기록·census·기술 층 + '교체 뒤 자연 경과' 2차 추정량 · 확신도 medium · 반박 담당 판정: 반박 불성립

**1. 현 제안 그대로: 알려진 새 투수·타자는 계속 평가, 미지 새 투수는 거절 후 sticky** — R7과 run_policy_validation.py:46의 등록값 'continue_known_pitcher'를 그대로 쓴다. policy_requests.pa_requests가 이미 행마다 실제 pitcher·stand로 PAState를 만들고 같은 PA의 이전 투구 전부를 history로 넘기므로, 알려진 새 투수의 결정은 새 H_t로 π_cand·π_ref·π̂_b를 계산한다. 새 투수가 BC에 없으면 UNSUPPORTED_UNKNOWN_PITCHER가 되고, 그 뒤 결정은 UNSUPPORTED_MID_PA(sticky, policy_runtime.py 207·221행)가 된다. 교체 기록, 층 보고, 추가 가정 명시는 없다.

  - 장점: D89 §1(H_t에 pitcher_id·batter_stand 포함)과 §5 DR 식을 바꾸지 않는다. 주 추정량과 R1·D-2 분모가 그대로다 / PA 중 등판한 구원 투수에게도 추천하는 서비스 상황과 정의가 같다 / 알려진 교체를 결정 뒤 사건으로 걸러내지 않는다. mid_pa_state_change(도루·폭투)를 계속 평가하는 R1 원칙과도 일관된다
  - 단점: D87 config explicit_handling의 'substitution_multiple_pitchers'를 채우지 못하고, D89 config의 mid_pa_pitcher_change_rule=null도 사실상 미결로 남는다. 원장에는 타자 id가 없어(context_key=game:ab:pitch) 같은 손 대타를 복원할 수 없다. runtime.summary()는 중간 거절을 MID_PA로 합친다(316행) / 교체 직후 결정에서 nuisance 두 개가 함께 약해진다. G0·delivery pool은 pitcher_or_batter_changes를 뺀 eligible 행으로 적합됐다(data.py:281). π̂_b는 previous 키가 이전 투수의 구종이라 셀이 희소하면 prior_strength=20 축소로 투수 주변 빈도 쪽으로 기운다(rollout_policy.py CategoricalBC). 후보 쪽 불편성이 π̂_b 경로 하나에 기대는 구조(D89 §4(iv))에서 이 구간이 가장 약하다. 크기는 미측정이다 / 미지 새 투수가 PA 중에 등판하면 PA가 UNSUPPORTED가 된다. 이는 첫 투구 뒤 사건으로 조건부 점추정 모집단을 거르는 처치 후 선택이다([0,1] 경계는 유효). 2026에는 TRAIN 뒤 데뷔한 투수가 많아 이 비율이 커질 수 있다(미측정)
  - 추정량·분모 변경: 아니오, 구현 비용: low

**2. 계속 평가(주) + 사전 등록 교체 기록·census·기술 층 + '교체 뒤 자연 경과' 2차 추정량** — 주 추정량·분모·런타임 동작은 1과 같다. 결과 열람(S4 V5·S6 V4) 전에 다음을 함께 고정한다. (a) 요청 생성기가 PA별 교체 manifest를 쓴다. 투구 전 정보만으로 정의하며 내용은 PA 시작 투수 id, 투수가 바뀐 첫 결정 index t*, 타자 id 변경, stand 변경, 새 투수의 TRAIN BC 포함 여부다. 요청 fingerprint에는 넣지 않고 행 SHA로 stage manifest에 pin한다. (b) S0 census를 확장한다(label-blind). (c) PA 층 보고는 교체 없음 / 알려진 투수 교체 / 미지 투수 교체 / 대타(같은 손·다른 손)이며, '시작 뒤 사건으로 나눈 기술 층, 추정량 아님'이라는 라벨을 붙인다. (d) 같은 원장으로 재추론 없이 계산하는 2차 추정량을 등록한다. 목표 정책은 'PA 시작 투수의 결정에는 π, 투수가 바뀐 첫 결정부터는 실제 로깅 법칙'이다. 따라서 t* 이후 ρ≡1과 V_{t*}=관측 PA 보상은 정의상 정확하다. 교체 PA를 빼는 식의 제외형 민감도는 쓰지 않는다. (e) 식별 가정에 '교체 전이는 관측 이력의 함수인 환경 법칙이며, 구종 선택에 영향을 준 비관측 요인(부상 등)과 무관'을 추가한다. (f) 2차 결과는 주 결론·후보·채택을 바꾸지 않는다고 등록한다.

  - 장점: 주 추정량·분모는 D87/D89·R1·D91 그대로다. 서비스 정합성도 유지된다. 계약 변경은 '규칙 확정 + 보고 층 + 2차 추정량 추가'로 한정된다 / 2차 추정량이 렌즈 1의 핵심 우려 두 가지를 정면으로 드러낸다. 하나는 미지 새 투수 교체로 생기는 처치 후 거절(2차에서는 거절이 아님)이고, 다른 하나는 교체 뒤 교차 투수 셀의 π̂_b·q̂ 약점(2차에서는 쓰이지 않음)이다. 주와 2차의 차이가 교체 처리의 민감도가 된다 / 원장 행에 결정별 pitcher가 이미 있어(policy_runtime.py 76·295행) 2차 추정은 추정기 분기만 추가하면 된다. 추가 추론 비용이 없다
  - 단점: 생성기 manifest, census 필드, 추정기 2차 분기·층 함수, 계약 문서 세 곳, 테스트가 필요하다(중간 비용) / 주 추정량의 교체 뒤 nuisance 약점은 고치지 않고 드러내기만 한다 / 추정량이 두 개라 사후에 해석을 고르려는 유혹이 생긴다. '주 고정, 2차는 채택에 쓰지 않음'을 등록 문구로 강제하고 Holm family에서 뺄지 정해야 한다
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**3. 투수 교체 뒤 자연 경과 동적 정책을 주 추정량으로(1의 계속 평가는 2차)** — 렌즈 1의 선호안이다. 목표 정책을 'PA 시작 투수의 결정에서는 π(cand/ref), 투수가 바뀐 첫 결정부터 PA 끝까지는 실제 로깅 법칙'으로 재정의해 주 추정량으로 삼는다. 알려진 새 투수와 미지 새 투수에 똑같이 적용하고, 타자 교체는 π로 계속 평가한다. cand와 ref가 같은 V_{t*}(관측 보상)를 공유하므로 paired Δ는 교체 전 결정에서만 생긴다. 런타임에는 새 평가 상태(예: NATURAL_COURSE_AFTER_SUBSTITUTION)를 추가하거나, 추정기가 manifest의 t*로 재귀를 자른다.

  - 장점: 교체 뒤 구간에서는 π̂_b(교차 투수 셀), q̂(교차 투수 이력), TRAIN→평가 로깅 법칙 이동, positivity가 모두 필요 없다. 식별 가정이 원래 투수의 정상 셀에만 걸린다 / 미지 투수의 PA 중 등판이 거절 사유가 아니게 된다. 처치 후 거절이 사라지고, 2026에 미지 투수가 늘어도 이 경로로 경계가 넓어지지 않는다 / 교체 뒤 ρ 곱과 q̂ 잡음이 없어져 분산이 줄어든다
  - 단점: 주 추정량이 바뀐다. D89 §1·§2 표, D87, D92 목록(D-4가 추정량 변경 목록에 없음), docs/decisions.md를 모두 고쳐야 한다 / 교체 PA에서 추정 대상이 '시작 투수에게만 개입한 효과'가 된다. 구원 투수에게도 추천하는 서비스와 어긋난다 / 교체 전 결정의 Q̂^ref는 여전히 '같은 투수가 끝까지'를 가정하므로 q̂ 오차가 남는다
  - 추정량·분모 변경: 예, 구현 비용: medium

**4. PA 중 투수·타자 교체 전부 sticky 거절** — supported_pa의 pitcher_or_batter_changes와 방향을 맞춘다. PA 첫 요청과 투수·타자가 다른 첫 요청을 새 상태(예: UNSUPPORTED_SUBSTITUTION)로 거절하고, 이후 결정은 MID_PA로 둔다. PA는 R1 분모에 남고 R6 [0,1] 최악 경계로만 들어간다. 같은 손 대타를 감지하려면 생성기가 타자 id 변경 플래그를 요청·fingerprint에 넣어야 한다.

  - 장점: q̂(G0·pool·시뮬레이터)와 BC-E가 적합된 모집단(교체 없는 PA)과 평가 대상이 일치해 모델 밖 외삽이 없다 / 교체 PA 비중이 경계 폭 증가분으로 그대로 드러난다. 가장 보수적인 fail-closed다
  - 단점: 교체는 첫 투구 뒤 사건이고 앞선 행동·결과에 의존할 수 있다. 조건부 점추정이 정책별로 다른 부분 모집단을 조건으로 삼게 된다. D87 §2가 경고한 사후 '완전·지원 PA' 선택과 같은 구조라, 좁은 추정량 이름이 필요하다 / 부상처럼 비관측 원인으로 일어난 교체라면 U에 대한 선택이 더해진다 / 도루·폭투(mid_pa_state_change)는 계속 평가하면서 교체만 거절하게 되어, 시뮬레이터가 똑같이 모형화하지 않는 두 현상을 다르게 다룬다
  - 추정량·분모 변경: 예, 구현 비용: medium

**근거(요약).** 주 규칙은 '계속 평가'가 맞습니다. 세 렌즈 모두 교체를 결정 뒤 사건으로 걸러내는 방식(선택지 4, 분모 제외)을 D87 §2의 사후 선택 경고, R1, D91 원장 분모 원칙과 충돌한다고 봤습니다. D89 §1의 H_t에는 pitcher_id·batter_stand가 들어 있고 §5 DR 식에는 같은 투수 가정이 없습니다. 그래서 알려진 새 투수는 새 H_t일 뿐이고, 서비스(구원 투수에게도 추천)와도 정의가 같습니다. 현재 코드(policy_requests.pa_requests, policy_runtime._evaluate)도 이미 이렇게 동작합니다.

**추천이 바뀔 조건.** - S0 census에서 Statcast가 PA 중 교체 때 행별 투수·타자 id를 따로 기록하지 않고 최종 matchup id로 덮어쓰는 것으로 확인되면, 교체 자체를 감지할 수 없습니다. 1·2·3 모두 설계대로 구현할 수 없으므로, 먼저 fail-closed 감지 규칙(예: 던지는 손 불일치, PA 첫·끝 행 비교)을 정해야 합니다. - S0 census에서 '미지 새 투수 교체' PA가 평가 가능 PA 대비 무시하기 어려운 비율로 나오면, 선택지 3(자연 경과 주)으로 옮기는 편이 낫습니다. 이 문턱은 Song이 S0 전에 정해야 하며, 현재 값은 없습니다. 2026에서는 TRAIN 뒤 데뷔 투수 때문에 이 비율이 더 클 수 있습니다(미측정). - 이력 의존 교체 합성 toy에서 주(계속 평가) DR이 참값에서 뚜렷이 벗어나고 자연 경과 DR은 맞으면, 3을 주로 올립니다. - D-1에서 BC-E가 주로 채택되면, 교체 뒤 π̂_b가 적합 자료 밖 외삽이 됩니다. 이때는 계속 평가의 근거가 무너지므로 3 또는 4 쪽으로 옮깁니다. - 연구 책임자가 서비스 대상을 'PA 시작 투수에게만 추천'으로 정의하면, 3이 서비스와 추정량을 동시에 맞춥니다. - 교체 PA가 극히 드물면(문턱 사전 등록) 2차 추정량 없이 층 보고만 남긴 경량판으로 줄일 수 있습니다.

## D-5 불완전 PA 처리와 경계 방식

- **사용자 결정:** C. E0 + IS 가중 부분 궤적 구간 (검열 노드 가치를 [0,1] 미지값으로)
- 추천: C. E0 + IS 가중 부분 궤적 구간 (검열 노드 가치를 [0,1] 미지값으로) · 확신도 medium · 반박 담당 판정: 반박 불성립

**A′. 현재 제안 유지 + 분모·FAILED 교정** — R6 제안을 그대로 쓰고, WIP(a704983) 코드의 결함만 고친다. 점추정은 COMPLETE PA만의 per-decision DR Δ 평균이며 이름은 '평가 가능 PA 조건부'다. 나머지 PA는 PA당 Δ∈[−1,1]로 세어 무가중 경계를 만든다. 교정할 부분은 세 가지다. (1) 경계 분모를 제출 PA가 아니라 pas_in_split(+unsubmittable)로 바꾼다. 지금은 run_policy_validation.run_dr 271–276행이 제출 PA만 estimate에 넘긴다. (2) FAILED가 하나라도 있으면 estimate를 거부한다. 지금은 policy_estimator.estimate 146행이 FAILED를 others로 흡수한다. (3) 경계 끝점에 경기 단위 bootstrap을 붙인다.

  - 장점: 구현이 거의 끝나 있다(pa_status·pa_dr·game_bootstrap·ESS). 교정은 수십 줄이다. / D87 §6의 '전체 모집단 값은 null 또는 [0,1] 최악 경계', '결과로 PA를 빼지 않음' 문구와 1:1로 대응한다. / V4(cand=ref이면 Δ̂=0) 대수·무결성 검사에 바로 쓸 수 있다.
  - 단점: 점추정의 조건 집합 COMPLETE가 처치 뒤 사건으로 정해진다. LOGGING_POSITIVITY, NO_LOGGED_ACTION, PA 중 미지 투수, 이력 불일치, 종료 없음이 그 예다. 그래서 어떤 모집단의 정책 가치도 아니다. D89 §5가 π̂_b 재정규화를 금지한 근거('버리면 처치 후 선택')가 PA 단위에서도 그대로 성립한다. / 무가중 ±1 경계는 검열이 정책과 무관할 때만 유효하다. 필요한 질량은 P_π(검열)인데 이 방식은 P_b(검열)을 넣는다. 렌즈 1의 1결정 반례에서 경계는 [0,.5]이고 참값 .9는 그 밖에 있다. π̂_b가 정확해도 q̂가 틀리면 이렇게 무효가 된다. / 투구 전 제외(신규 투수·빈 지원·0-0 아님)와 시작 후 탈락이 한 UNSUPPORTED 칸에 섞인다. D87 §6이 요구한 '투구 전 필드로 정한 제한 모집단 결과'가 없다.
  - 추정량·분모 변경: 아니오, 구현 비용: low

**B. 투구 전 제한 모집단 E0 + 시작 후 불완전은 무가중 [0,1] 경계 (P8식 2층)** — PA를 UNSUBMITTABLE, EXCLUDED_PRE_START, POST_START_UNSUPPORTED(사유·k), INCOMPLETE_NO_TERMINAL(사유), COMPLETE, FAILED로 나누고 모두 분모에 남긴다. E0(=S_pre)는 첫 기록 행 0-0, TRAIN BC 투수, 0-0 맥락의 로깅 지원과 개입 마스크 M이 모두 비어 있지 않음으로 정한다. 로그 행동·description·events·이후 행은 보지 않는다. 보고 층은 넷이다. L0 = R1 전체 PA 경계, L1(주) = E0 경계(시작 후 불완전은 PA당 ±1), L2 = E0∩COMPLETE 조건부 평균(좁은 이름, 판정 없음), L3 = supported_pa 호환 기술 층. 판정은 P8처럼 L1 worst-case game-bootstrap CI 하한>0이 강한 관문이고, L2만 통과하면 예비다.

  - 장점: D87 §2(투구 전 필드로만 정한 시작 집합)와 §6(투구 전 필드로 정한 제한 모집단)을 그대로 따른다. 좁은 이름 조항은 L2로 지킨다. / P8 계약(ML-OFFLINE-RL-FEASIBILITY-v1)의 imputed/worst-case 2층 관문과 같아 새 판정 원칙이 필요 없다. / 수정 범위가 작다(pa_status 분류, 사전 판정 함수, run_dr 분모, 끝점 bootstrap). 신규 투수로 인한 제외가 E0 경계를 넓히지 않는다.
  - 단점: E0 안의 PA 중간 거절·종료 미관측에는 A와 같은 무가중 경계 문제가 남는다. 후보가 검열 상태로 더 자주 가면 과소 포함한다. / 주 결과가 경계로 옮겨 가므로 시작 후 불완전 비율(미측정)이 크면 정보가 없다. / PolicyRuntime._evaluate는 NO_PITCH 검사를 INCOMPLETE_START·투수·지원 검사보다 먼저 한다(210–230행). 그래서 E0 판정에는 원장과 별도인 사전 판정 경로가 필요하다.
  - 추정량·분모 변경: 예, 구현 비용: low

**C. E0 + IS 가중 부분 궤적 구간 (검열 노드 가치를 [0,1] 미지값으로)** — B의 분류·층·관문 구조를 그대로 쓴다. 다만 E0 안에서 결정 k에 검열된 PA를 빼지도, ±1로 세지도 않는다. 대신 원장에 이미 있는 k 이전 행(result·q·ρ)으로 D89 §5 DR 재귀를 돌리고 V_k=c로 둔다. 그러면 V_0(c)=base+(Π_{t<k}ρ_t)·c이고 기울기가 0 이상이므로, c=0과 c=1이 정책별 하한·상한이 된다. 경우는 셋이다. (a) 요청 거절 또는 PA 미종료: 후보·기준의 미지값이 서로 독립이다. Δ 경계는 [δ(U_c=0,U_r=1), δ(U_c=1,U_r=0)]이고 PA 폭은 Π_{t<k}ρ^c+Π_{t<k}ρ^r다. (b) 종료는 관측됐지만 종료 상태 값만 결측(final_result_undetermined, incomplete_game, invalid_next_state): 두 정책이 로그 보상 r 하나를 공유하므로 폭은 |Πρ^c−Πρ^r|다. (c) k=0 거절(E0 안의 LOGGING_POSITIVITY·NO_LOGGED_ACTION): 폭 2다. 주 결과는 L1 = E0 평균 Δ의 구간이다. 두 끝점과 L2는 같은 경기 재표집 인덱스로 bootstrap한다. 추가 보고는 정책별 추정 검열 확률 mean(Πρ^π), 관측 검열 비율, 검열 가중 ESS다. 선언 대체값(검열 노드 경기 상태의 동결 WE, P8 cap 꼬리 규약) 점추정은 '꼬리 가정 의존' 기술 층으로만 둔다.

  - 장점: 기대값 의미에서 유효한 경계다. 가정은 π̂_b가 정확하다는 것 하나이고, D89 후보 경로가 이미 기대는 가정이라 새 가정이 아니다. q̂가 틀려도 유효하다. 렌즈 1의 반례에서 구간은 [0,1]이 되어 참값 .9를 포함한다. / 후보가 미지원 상태로 끌고 가는 정도를 끝점 차이(=P_π(검열) 추정치)로 직접 보고한다. D87 §6이 기록하라고 요구한 항목이다. / 원장에 기록된 ρ·q만 다시 쓴다. 런타임 호출, 새 nuisance, 2026 적합이 없고 실행 비용은 A와 같다.
  - 단점: 끝점에 ρ 곱이 들어가 꼬리가 무겁다. 유한 표본 폭과 분산은 미측정이며 B보다 불안정할 수 있어 ESS·집중도 진단이 필수다. / 경계 폭 자체가 π̂_b(TRAIN→평가 창 이동) 정확성에 기댄다. 검열 노드에서 LOGGING_POSITIVITY가 났다는 것은 그 노드에서 π̂_b가 틀렸다는 뜻이다. / per-PA DR 값은 [0,1] 밖일 수 있으므로 식별된 sharp 경계가 아니라 추정 경계다. 쉬운 말 설명이 필요하다.
  - 추정량·분모 변경: 예, 구현 비용: medium

**D. 위임(abstention)·혼합 정책 점추정 (기각 기준안)** — 미지원 노드부터 두 정책이 모두 실제 투수 행동에 맡긴다고 본다. 위임 뒤에는 ρ=1로 두고 관측 continuation을 쓰며, non-E0 PA는 Δ=0으로 둔다. 그러면 L0 전체에 점추정이 하나 나온다. 쓰려면 Song이 decisions.md로 D87 §6·D89 §2를 뒤집어야 하고, 그 경우에도 이름 붙인 보조 estimand로만 사전 등록한다.

  - 장점: 구간 폭이 거의 없고 ρ 곱이 짧아져 분산이 작다. / '지원될 때만 추천한다'는 서비스 배치 동작과 맞는다.
  - 단점: D87 §6과 D89 §2가 명시적으로 금지한 방식이다('관측 행동을 넣거나 비를 1로 두지 않음'). / 정책이 평가 연도 투수 행동에 의존하는 혼합 체제가 된다. D89 결정성·policy identity 계약과 충돌한다. / Δ가 지원 비율만큼 0 쪽으로 희석된다. 그래서 신규 투수가 많은 2026과 DEV를 비교할 수 없다.
  - 추정량·분모 변경: 예, 구현 비용: medium

**근거(요약).** 세 렌즈가 모두 동의한 점이 네 가지다. (1) 현재 제안의 점추정은 조건 집합 COMPLETE가 처치 뒤 사건으로 정해진다. 그래서 어떤 모집단의 정책 가치도 아니며, D89 §5의 처치 후 선택 논리가 PA 단위에서도 성립한다. (2) 평가 모집단은 D87 §2·§6대로 투구 전 필드만으로 정해야 한다. (3) WIP 코드에는 두 결함이 있다. run_dr은 경계 분모에서 unsubmittable PA를 빼고, estimate는 FAILED를 경계로 흡수한다. (4) 경계 끝점에는 경기 bootstrap이 필요하다(P8 worst-case game-bootstrap 선례). 이 넷에 대해서는 B와 C가 같다.

**추천이 바뀔 조건.** 다음 경우에는 추천이 바뀔 수 있다.  - S0 census에서 검열이 거의 없을 때: census는 결과 값을 보지 않고 상태 개수만 센다. D-3·D-4·D-6을 적용한 뒤에도 E0 안의 시작 후 검열(k≥1 거절, 종료 미관측)이 거의 없다면 B와 C의 차이가 작다. 그때는 단순한 B로 충분하다. 이 판단 규칙 자체를 census 전에 적어 두어야 한다. - 검열 가중이 불안정할 때: V1 toy나 V2 준합성에 정책 의존 hazard를 넣어 봤을 때 끝점 분산이나 ESS가 무너지면, C 끝점이 판정에 쓸 수 없을 만큼 불안정하다는 뜻이다. 그때는 B로 물러나되 '정책 독립 검열 가정' 라벨을 명시하거나, 주 보고를 L0/L1 기술 층으로 낮춘다. - Song이 decisions.md로 서비스 estimand를 '지원될 때만 추천'으로 정할 때: 그러면 D를 이름 붙인 보조 estimand로 추가 등록한다. 이 경우에도 C는 주 결과로 유지한다. - D-3에서 automatic 행을 환경 전이(ρ=1)로 두고, D-6에서 truncated_pa·주루사 블록을 다음 행 상태 WE로 종료 처리할 때: 검열 대부분이 사라진다. C를 유지하되 구현 우선순위는 낮아진다. - Astra 독립 검토에서 부분 궤적 경계 유도에 오류가 나올 때(예: 공유 r 경우의 부호, 거절 노드의 π̂_b 불일치 처리): 수정하거나 B로 되돌린다. - 비E0 비율이나 검열 비율이 매우 커서 어느 방식으로도 구간에 정보가 없을 때: 선택지 간 차이는 의미를 잃는다. 그래도 결과를 본 뒤 주 보고 항목을 대체값 점추정으로 바꾸지 않는다.

## D-6 PA 종료 상태·보상 계산

- **사용자 결정:** C. B + 구조적 종료 (fail-closed 검사는 무조건 먼저 구현하고, truncated 포함 규칙은 S0 TRAIN census 뒤·S6 DEV 보상 열람 전에 고정)
- 추천: C. B + 구조적 종료 (fail-closed 검사는 무조건 먼저 구현하고, truncated 포함 규칙은 S0 TRAIN census 뒤·S6 DEV 보상 열람 전에 고정) · 확신도 medium · 반박 담당 판정: 반박 불성립

**A. 현행 R5 + WIP pa_outcome 그대로** — 종료는 PA 마지막 행의 events가 비결측이고 truncated_pa가 아닐 때다. 보상은 같은 경기 다음 행의 투구 전 상태(next_*)에서 계산한 초기 수비 팀의 동결 C0 WE다. 다음 행이 없으면 PA 자신의 post_* 점수로 최종 승패 {0,1}을 쓴다. 점수 불일치는 flag만 달고, 그 개수만 run_policy_validation.py:277 reward_flags로 보고한다. 구현은 a704983의 policy_requests.pa_outcome(미검토)이다.

  - 장점: 보상이 관측 상태다. PA 중 도루·폭투·견제와 실제 진루 결과가 모두 반영된다 / next_*는 다음 PA 첫 행이므로 WinExpectancy.labeled_states의 적합 영역(PA 첫 행) 안에 있다. EmpiricalAdvancement도 TRAIN next_*로 적합했다 / D87 §3·D89 §5의 W(S_PA_end) 문구, C1 observed_post_pa_frozen_we 관습과 같다
  - 단점: 코드에서 확인한 fail-open 경로가 네 가지다. (1) complete_game 열이 없으면 검사를 건너뛰고 NaN이면 bool(NaN)=True로 통과한다(L107). (2) 최종 승패 분기가 이 PA가 경기 마지막 PA인지 확인하지 않는다. (3) 다음 행은 at_bat_number <= last만 거절해 +1 인접성과 0-0 첫 행을 보지 않는다. (4) 정렬이 game_date 우선(data.py:326)이라 한 game_pk가 여러 날짜에 걸치면 경기가 두 블록으로 갈라질 수 있다. 네 경로 모두 [0,1] 범위 검사를 통과하므로 결과만 봐서는 알아챌 수 없다 / truncated_pa를 종료에서 빼는 것은 로그 궤적의 결과로 고르는 것이다. 그래서 '평가 가능 PA 조건부' 점추정은 어떤 고정 모집단의 인과 대비도 아니다. 이런 PA가 [0,1] 경계로 가면서 구간도 넓어진다(비율 미측정) / truncated_pa는 INCOMPLETE인데 events가 주자 사건인 마지막 행은 COMPLETE가 된다. 타자 결과 없는 종료가 두 갈래로 갈린다(존재·빈도 미측정)
  - 추정량·분모 변경: 아니오, 구현 비용: low

**B. 보상 정의 유지 + fail-closed 구조 검사 + 불일치 민감도 (truncated_pa는 현행대로 INCOMPLETE)** — 보상 정의와 R5 종료 문구는 A와 같다. 다음 검사를 추가한다. 전체 경기 frame에서 game_pk가 연속된 한 블록인지 확인하고, complete_game·post 점수 열이 없거나 NaN이면 FAILED_INTEGRITY로 멈춘다. 경기 마지막 PA는 경기 최대 at_bat_number로 판정한다. 다음 행 인접성은 at_bat +1, pitch_number 첫 행, 0-0, 합법 반이닝 전환, 같은 반이닝 안에서 아웃·점수 비감소로 확인한다. 위반하면 새 R6 사유(INCOMPLETE_UNOBSERVED_END)로 [0,1] 경계에 넣는다. flag는 PA 행까지 전달하고 flag PA를 경계로 옮긴 민감도를 사전 고정한다. 실데이터에서 pa_outcome과 data.py next_*의 parity를 검사한다. 구조적 종료(truncated 포함)는 S0 뒤 필요하면 부차 분석으로 따로 등록한다.

  - 장점: A의 fail-open 경로가 조용한 오답 대신 명시된 사유나 중단으로 바뀐다 / D87/D89/C0의 W(S_PA_end) 정의와 R5 종료 문구를 바꾸지 않아 계약 개정이 가장 적다 / 결과로 PA를 빼지 않는다. 구조 결함 PA는 분모 안에서 경계로만 간다
  - 단점: truncated_pa에서 생기는 처치 후 선택과 경계 폭 문제는 그대로 남는다 / gap PA가 COMPLETE에서 빠지므로 조건부 점추정의 부분집합이 A와 조금 달라진다(비율 미측정) / 사유 코드, 민감도, parity, census 항목이 늘어 구현·테스트 비용이 A보다 크다
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**C. B + 구조적 종료: 끝 상태가 관측되면 truncated_pa·비타자 종료도 COMPLETE(종료 사건 층)** — B의 fail-closed 검사를 모두 넣는다. 여기에 더해 PA 종료를 events 라벨이 아니라 구조로 정한다. 같은 경기에서 인접한 다음 PA 첫 행(at_bat +1, 0-0, 합법 전환)이 관측되거나 경기가 공식 종료되면 PA가 끝난 것으로 본다. truncated_pa, 주자 사건 종료, field_error·catcher_interf 같은 비표준 종료도 끝 상태가 관측되면 COMPLETE로 두고 end_kind 층(batter_event / non_batter_end / game_final)으로 보고한다. 경기를 끝낸 PA 도중 사건은 최종 승패로 계산한다. 보상 함수는 A·B와 같다. OPE 전용 종료 규칙은 이름에 버전을 붙여(예: structural-end-v1) config에 등록한다. data.py의 is_pa_terminal은 건드리지 않는다. 규칙과 층 목록은 S0 TRAIN census 뒤, S6 DEV 보상을 열기 전에 decisions.md 한 줄과 R5·R6 개정으로 고정한다. 기존 R5 규칙(truncated→INCOMPLETE)은 사전 등록 민감도로 남긴다.

  - 장점: 처치 후 선택을 없앤다. 남는 INCOMPLETE는 행 누락·미종료 경기 같은 자료 결함뿐이어서, 구종 선택과 무관하다는 가정이 A·B보다 방어하기 쉽다 / R1 분모(모든 정규시즌 PA)는 그대로이고 COMPLETE 집합만 커진다. 경계 폭이 줄어든다(얼마나 줄지는 미측정) / D87 §2의 '투구 없는 종료·이닝 전환 명시 처리'와 '사후 완전 PA 선택 금지'를 코드 수준에서 지킨다. truncated_pa와 주자 사건 종료의 불일치도 한 규칙으로 정리된다
  - 단점: R5 종료 문구와 평가 가능 부분집합의 이름이 바뀐다. decisions.md 한 줄, 계약·config R5·R6 개정, §5 S0 범위 문구('events 유무만') 수정이 필요하다 / q̂(game.terminal_values)는 잘림과 PA 중 주자 이동을 모형화하지 않는다. 그래서 non_batter_end 층의 DR 불편성은 π̂_b 경로에만 기대고, 잔차 분산이 커진다(크기 미측정) / truncated_pa의 의미 근거는 src/pitcheezy/baselines/b2.py:26 주석('경기·이닝 중단')뿐이다. 다음 행 상태가 그 사건을 제대로 반영하는지 저장소에서 확인되지 않았다
  - 추정량·분모 변경: 예, 구현 비용: medium

**D. 모형 사건 기대 보상: r = terminal_values(마지막 결정 상태)[관측 사건 10-class]** — 다음 행 대신 관측 종료 사건을 TERMINALS class로 묶는다. 보상은 동결 EmpiricalAdvancement+WE(game_values)로 계산한 기대 WE다. q̂의 terminal 정의와 정확히 같다. 이 안을 주 보상으로 쓰는 경우다(참고: 세 렌즈 모두 supported_pa식 결과 결함 제외안은 R1·D87 위반으로 기각했다).

  - 장점: 보상과 q̂의 terminal 사상이 같아 DR 잔차에서 진루 잡음과 PA 사이 사건이 빠진다. 분산이 줄 가능성이 있으나 미측정이다 / 다음 행 인접성, 경기 분할, 점수 정정 문제에 영향을 받지 않는다
  - 단점: 추정량이 관측 W(S_PA_end)에서 모형화한 사건 가치로 바뀐다. D87/D89 문구, C1 observed 관습과 맞지 않는다 / 같은 사건 class 안에서 구종이 진루 질에 주는 효과가 지워진다. PA 중 도루·견제·실책 진루도 사라진다 / 사건 어휘가 모듈마다 어긋난다(strikeout_double_play는 EVENT_OUTS에만, other_out은 _EVENT_GROUP에만 있다. field_error·catcher_interf는 매핑이 없다). 매핑이 새 자유도가 되고, 미지원으로 두면 다시 결과 기반 선택이 된다
  - 추정량·분모 변경: 예, 구현 비용: medium

**근거(요약).** 세 렌즈가 공통으로 합의한 것은 네 가지다. (1) 보상 위치는 관측된 다음 PA 첫 행의 투구 전 상태로 계산한 동결 C0 WE이고, 경기 끝은 최종 승패다. Statcast 원열(data.py RAW_COLUMNS)에는 사후 주자·아웃이 없어 이것이 유일한 관측 종료 상태다. WE와 EmpiricalAdvancement도 모두 PA 첫 행과 next_*로 적합했으므로 정의역이 맞는다. (2) 점수 불일치는 주 추정에서 빼지 않고 flag로 둔다. (3) supported_pa식 제외는 결정 뒤 선택이므로 기각한다. 모형 사건 보상(D)은 주 추정량이 아니라 진단 열로만 둔다. (4) WIP pa_outcome에는 조용히 틀린 값이 나오는 경로가 있다. policy_requests.py에서 직접 확인했다. complete_game 열이 없으면 검사를 건너뛰고 NaN이면 통과한다. 최종 승패 분기는 경기 마지막 PA인지 확인하지 않는다. 다음 행은 `at_bat_number <= last`만 거절한다. 그래서 A를 그대로 쓰는 것은 어느 렌즈도 지지하지 않는다.

**추천이 바뀔 조건.** B로 바꿀 조건은 다섯 가지다. (1) S0 TRAIN census에서 truncated_pa 다음 행이 인접한 합법 PA 시작이 아닌 경우가 흔하면, 끝 상태를 관측할 수 없으므로 구조적 종료의 이점이 사라진다. (2) truncated_pa·비타자 종료가 매우 드물면 계약 개정 비용이 이득보다 크다. (3) 연구 책임자가 D87/R5 문구 안정성과 P8/G0 근처 비교를 조건부 점추정의 해석보다 우선하면 B가 맞다. (4) V2 변형(모형 밖 종료 사건 주입)에서 non_batter_end 층의 DR 편향이나 분산이 크게 나오면, 그 층을 주 추정에서 분리하는 쪽을 다시 검토한다. (5) 점추정을 보고하지 않고 경계만 쓰기로 하면(D-5), truncated 처리는 폭 문제일 뿐이므로 B로 충분하다. at_bat_number 공백이 Statcast에서 규칙적으로 생기면(투구 없는 PA 등, 미측정) 인접성 규칙을 'gap 허용 + flag'로 완화해야 한다. D를 주 보상으로 올리려면 D87/D89 개정이 필요한데, 현재 근거로는 그럴 이유가 없다.

## D-7 ≤2025 타자 style prior 고정 시점

- **사용자 결정:** C. 창 시작 스냅샷 + 등록 보강 (pin된 artifact, 경계 가드, 미지 타자 규칙, 2026 날짜 동시 고정, 무라벨 진단)
- 추천: C. 창 시작 스냅샷 + 등록 보강 (pin된 artifact, 경계 가드, 미지 타자 규칙, 2026 날짜 동시 고정, 무라벨 진단) · 확신도 high · 반박 담당 판정: 반박 불성립

**A. 현재 제안 그대로: 창 시작 스냅샷을 WIP 경로로 즉석 계산, S2만 rolling** — a704983(미검토)의 `policy_requests.style_snapshot(frame, as_of)`는 `game_date < as_of` 행만으로 C0 12열(rate 6 + reliability 6)을 계산하고, `apply_style_snapshot`이 as_of 이후 DEV 행을 덮어쓴다. `run_policy_validation.py` 409-411행에서 D-7='window_start_snapshot'이면 dr-evaluate에만 적용한다. 바꾸는 것은 config에 `le2025_validation_plan.profile_as_of.dev = 2025-07-01`을 넣는 것뿐이다. π̂_b(`CategoricalBC._key` = 투수·볼·스트라이크·타자면·직전 구종), 지원 M(pitcher·p_throws·stand), masked π_ref, V5 분모는 style을 쓰지 않으므로 그대로다. 바뀌는 것은 후보 P3와 q̂(G0 맥락 context[:,11:28])뿐이다. 현재 제안 대비 추정 대상은 같다.

  - 장점: 구현이 거의 끝났다. `test_style_snapshot_freezes_window`는 창 안 events를 모두 home_run으로 바꿔도 스냅샷이 그대로임을 합성 자료로 확인한다 / D87 §4·D89 §1의 '2026 이전 동결 프로필'과 같은 모양의 고정 정책이다. 창 안 결과로 후보 입력이 갱신되지 않아 경기 cluster bootstrap의 독립 가정이 깨끗하다 / 스냅샷이 쓰는 자료(<2025-07-01)는 rolling DEV 값이 이미 쓰던 자료의 부분집합이므로 새 노출 범주가 생기지 않는다
  - 단점: 현재 config에는 `profile_as_of` 키가 없다(확인: le2025_validation_plan 키 목록에 없음). 빠뜨리면 무거운 frame 로드 뒤 line 410에서 KeyError가 난다(fail-closed이지만 늦다) / as_of 검증이 없다. 문서는 '2025-06-30까지'라고 쓰는데 코드는 `<` 비교다. 06-30으로 적으면 6월 30일 자료가 조용히 빠지고, 07-02 이후로 적으면 창 안 결과가 섞이면서 as_of 이전 DEV 행은 rolling으로 남는다. 테스트도 07-03을 쓴다 / 스냅샷이 저장·해시되지 않는다. D92 정책 식별자는 맥락 행을 요청 fingerprint에만 넣으므로 as-of 규칙이 manifest·식별자에 드러나지 않는다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**B. rolling 유지 (D-7='rolling_prior')** — dr-evaluate·S5도 `regular_frame`(`add_batter_style_history` → `assign_fold`)의 경기일 엄격 이전 rolling 값을 그대로 쓴다. 러너의 DECISION_VALUES가 이미 이 값을 허용한다. 분모·π̂_b·M·π_ref는 그대로다. 후보 정책은 '창 안 로그 결과에 매일 적응하는 정책'으로 바뀐다(현재 제안 대비 추정 대상 변경).

  - 장점: 추가 코드가 없다. S2 probe(atol 1e-6)의 입력이 dr 입력과 같아서 실자료 재현 검증이 그대로 이어진다 / G0 학습·보정, 봉인 G0 DEV 평가(311,721구), T3 stress와 입력 분포가 같다. q̂ 품질에 측정 근거가 있고 DR 분산이 가장 작을 것으로 예상된다(미측정) / PA 단위 추정 대상에서 이전 날짜 결과는 처치 전 공변량이다. 스냅샷과 rolling이 섞이는 경계 오류가 원천적으로 생기지 않는다
  - 단점: 2026에서 금지한 동작(평가 창 결과로 프로필 갱신 = 통계 구성요소 추정, D87 §4·D89 §1)을 리허설한다. ≤2025 결과를 2026 절차 검증으로 옮길 수 없다 / 후보 입력이 평가 표본의 이전 경기 결과에 의존한다. 경기 부트스트랩은 특징값을 고정한 채 재표집하므로 교차 경기 의존을 무시한다(영향 미측정) / 실제로 배치하면 rolling 입력은 후보가 만든 결과로 갱신된다. OPE는 로그 결과를 넣으므로 배치 가능한 정책의 가치라는 해석이 약해진다
  - 추정량·분모 변경: 예, 구현 비용: low

**C. 창 시작 스냅샷 + 등록 보강 (pin된 artifact, 경계 가드, 미지 타자 규칙, 2026 날짜 동시 고정, 무라벨 진단)** — 주 규칙은 A와 같다: `as_of_exclusive = 2025-07-01`, `game_date < as_of`. 다음을 등록 전에 고정한다. ① S1(materialize-bc)에서 as_of 이전 행만으로 타자 스냅샷 표와 league 기본행(as_of 시점 league 축소율, reliability 0)을 만든다. 저장하고 SHA·as_of·원본 행 수·ordered key hash·커밋을 manifest에 기록한다. 창 안 명단은 읽지 않는다. ② dr-evaluate·S5는 등록 SHA로 로드한다. 스냅샷에 없는 타자는 league 행에 매핑하고 그 수를 보고한다. 이 값은 `add_batter_style_history`가 첫 등장 타자에게 주는 값(prior 분모 0 → league fallback, weight 0)과 같아 A와 수치가 같다. ③ 가드: as_of == 등록 창 시작, 평가 블록 최소 날짜 >= as_of(혼합 금지), 첫 DEV 날짜 행에서 스냅샷 == rolling(float32 비트 동일). 어기면 rollout 전에 FAILED_INTEGRITY로 멈춘다. ④ 2026 `common.profile_as_of`를 날짜 규칙('2026 평가 창 첫날 배타, ≤2025 정규시즌 전부')으로 함께 고정한다. ⑤ V4 보상 열람 전에 라벨 없는 진단을 사전 선언한다. rolling으로 가치를 다시 추정하는 arm은 이번 family에서 하지 않는다. 현재 제안 대비 추정 대상·분모는 같다.

  - 장점: A의 장점(2026 모사, 고정 정책, 새 노출 없음, π̂_b·M·π_ref·V5 분모 불변)을 모두 유지하면서 저장소에서 확인된 구멍 다섯 개를 닫는다: config 키 누락, off-by-one, 혼합, 미pin, 창 명단 미리 보기 / 2026 실행과 같은 자료 경로(사전 동결 artifact → SHA 로드 → 미지 타자 규칙)를 ≤2025에서 먼저 돌려 본다. 2026에서 'batter missing from the style snapshot'으로 전체가 멈출 위험을 미리 제거한다 / 첫 DEV 날짜 비트 동일 검사가 rolling S2 실자료 probe와 스냅샷 경로를 잇는다. `style_snapshot`에 넘긴 frame이 rolling 계산 때와 다른 행 풀이면(필터 적용 등) 즉시 드러난다
  - 단점: 새 코드가 필요하다: 저장·로드 함수, 가드 3개, 미지 타자 매핑, 합성 테스트 3-4개. S1 산출물과 config 필드(profile_as_of, 스냅샷 pin)가 늘어난다 / 2026 계약 config의 제안 값을 같이 고치므로 decisions.md 기록이 두 줄 필요하다(D-7, 2026 profile_as_of) / DEV staleness(창 안 0-3개월, offseason 없음)는 2026(offseason + 시즌 중 최대 약 6개월)보다 약하다. 리허설이 2026의 q̂ 악화와 ESS 하락을 과소평가할 수 있다
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**D. 더 이른 고정 시점 (TRAIN 종료 as_of 2025-05-01, 또는 2025 개막일 전)** — 타자 style prior를 투수 cluster·BC·pool과 같은 TRAIN 경계(<2025-05-01)나 2025 정규시즌 개막 전으로 고정해 May·June·DEV 전부에 쓴다. 서비스 트랙 선례는 `configs/EXP-CHOICE-DIAG-002.json`의 'frozen batter profiles as_of 2025-04-30'이지만 다른 모델 계열이다. 주 규칙으로 쓰거나 C의 두 번째 진단 arm으로 쓸 수 있다. 분모는 그대로이고 후보 정책 입력이 달라진다.

  - 장점: 모든 동결 구성요소가 한 경계를 공유한다(투수 TRAIN 고정, 타자 창 시작이라는 비대칭 해소) / DEV 안 staleness가 2-5개월(개막 전 as-of면 offseason까지 포함)로, 2026 후반기 staleness에 A보다 가깝다. 보수적(나쁜) 쪽 리허설이다 / earlystop·May·June 결과를 타자 입력으로 쓰지 않아 노출 설명이 가장 단순하다. 함수는 같고 as_of만 바꾸면 된다
  - 단점: 2026 규칙이 '≤2025 전 시즌 사용'이면 2026에서 한 시즌을 버리는 규칙이 되어 대응물이 아니다 / 2025 신인·call-up 대부분이 reliability 0이 되어 membership이 균등 쪽으로 줄어든다. 이 효과가 staleness 효과와 섞인다 / 첫 DEV 날짜에서 rolling과 달라져 '첫날 비트 동일' 연결 확인이 사라지고, G0 DEV·P8와의 입력 차이가 모든 DEV 행에서 A보다 크다
  - 추정량·분모 변경: 예, 구현 비용: low

**근거(요약).** 세 렌즈 모두 '창 시작 스냅샷을 주 규칙으로, S2만 rolling'이라는 방향에 동의했다. 차이는 보강을 얼마나 하느냐뿐이다.

**추천이 바뀔 조건.** - 2026 계약에서 `profile_as_of`를 TRAIN 종료(<2025-05-01)나 2025 개막 전으로 정하면 D가 정확한 대응물이 된다. 그러면 D-7도 그 날짜로 옮겨야 한다. - D87 §4를 뒤집어 2026 결과로 프로필을 갱신하는 것을 허용하면 B가 맞는 리허설이 된다. 현재 절대 규칙상 가능성은 낮다. - S1 산출물 추가나 2026 config 동시 수정을 이번 등록에서 허용하지 않으면 최소안으로 물러난다. 최소안은 A에 세 가지를 더한 것이다: config 키(`as_of_exclusive` 2025-07-01), 혼합 가드, 첫날 비트 동일 가드. - S2 봉인 probe 64행이 모두 첫 DEV 날짜에 있음이 확인되면(날짜 분포는 미측정) 추가 스냅샷 probe를 생략해도 된다. 반대로 64행이 여러 날짜에 흩어져 있으면 스냅샷 경로의 실자료 확인을 첫날 행에 한정해 따로 둔다. - 무라벨 진단에서 G0 예측 이동이 크게 나와도 이번 family의 주 규칙은 바꾸지 않는다(사전 고정). 다만 스냅샷 입력 G0 NLL 진단(라벨 사용)을 별도 stage로 등록할 근거가 된다.

## D-8 투구 손 규칙: 단일 손 사전 고정과 투구 전 거절

- **사용자 결정:** 3. 단일 손 규칙 사전 고정 + 투구 전 거절 상태(분모 유지, sticky)
- 추천: 3. 단일 손 규칙 사전 고정 + 투구 전 거절 상태(분모 유지, sticky) · 확신도 high · 반박 담당 판정: 반박 불성립

**1. 현행 유지: census 뒤 결정, (pitcher, batter_side) 키, 마스크가 다르면 생성 거부('refuse')** — S0 census(policy_requests.census의 train_pitchers_with_two_hands, pitcher_hand_pairs_unseen_in_train)를 본 뒤 D-8을 정한다. S1은 policy_identity.support_rows(two_hand_rule='refuse')로, TRAIN에서 손마다 마스크가 다른 투수가 한 명이라도 있으면 생성 전체를 거부한다. 런타임은 BoundComponents.check_support의 마스크 동일성 비교(policy_identity.py:247)에만 맡긴다. 선택한다면 계약 §3(56행)과 §6의 문구 '평가 맥락 손이 TRAIN과 다르면 FAILED_INTEGRITY'를 코드에 맞게 고쳐야 한다.

  - 장점: 추가 구현이 거의 없다. DECISION_VALUES['D-8']에 'refuse'가 있고, 합성 테스트(tests/test_policy_validation.py:323 부근)도 있다 / 지원 표 schema, 상태 집합, 요청 fingerprint를 바꾸지 않는다 / census는 투구 전 필드만 읽으므로 라벨 노출은 없다
  - 단점: 제안 문구와 코드가 다르다(코드 읽기로 확인, 실행 확인은 없음). check_support는 p_throws를 직접 비교하지 않는다. PolicyInputs.pool(matrix_policy.py:124-141)은 JointDelivery.TIERS의 league tier (pitch_type, p_throws, stand)까지 내려간다. 그래서 TRAIN에서 R만 던진 투수가 평가에서 L로 나와도 마스크가 같게 다시 계산되면 조용히 통과한다. 이때 후보 Q̂는 반대 손 league delivery로 계산되고, π̂_b는 R 이력 그대로여서 ρ가 잘못 지정된다 / 로깅 전용 런타임(S4 V5)은 support_check=None이다. DecisionRequest fingerprint와 원장 행에도 p_throws가 없어, 손 문제가 분모 보고에 전혀 나타나지 않는다 / 마스크가 달라서 걸리면 FAILED_INTEGRITY가 run 전체를 멈춘다(policy_runtime.py:301 halted=True). 투구 전에 알 수 있는 자료 성질 하나 때문에 2026 단발 실행이 없어질 수 있다. 이 처리는 D92의 FAILED_INTEGRITY 용도(pin된 산출물끼리의 불일치)와도 어긋난다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**2. 교집합('intersect') + 런타임 부분집합 검사** — support_rows(two_hand_rule='intersect')로 두 손 투수의 마스크를 TRAIN 손들의 교집합으로 등록한다. 등록하려면 check_support를 해당 투수에 한해 '표 마스크 ⊆ 재계산 마스크' 검사로 완화해야 한다.

  - 장점: S1 생성이 충돌 때문에 멈추지 않는다 / 행동을 추가하지 않아, 관측되지 않은 행동으로 외삽하지 않는다 / 생성 코드와 결정 값이 이미 있다(run_policy_validation.py:48)
  - 단점: 지금 코드 그대로면 모순이다. check_support는 np.array_equal이라, 더 넓은 마스크를 가진 손으로 나온 요청마다 FAILED_INTEGRITY로 멈춘다. 검사를 부분집합으로 완화하면 D92가 막으려던 '표와 pool 불일치'의 한 방향을 해당 투수에서 검출하지 못한다 / 두 손 투수의 M이 줄어 기준·후보 정책 정의가 바뀐다. π̂_b(M|H)<1이 되어 ρ^ref≡1(D89 §5)이 깨지고, OUTSIDE_POLICY_SUPPORT와 ρ 분산이 커진다 / CategoricalBC._key(rollout_policy.py:84)에 손이 없어 π̂_b가 두 손을 섞는 문제는 그대로 남는다
  - 추정량·분모 변경: 예, 구현 비용: medium

**3. 단일 손 규칙 사전 고정 + 투구 전 거절 상태(분모 유지, sticky) — 추천** — S0 census 전에 규칙을 등록한다. matrix_panel.py:81-82의 hand_rule을 재사용한다. TRAIN 관측이 모두 같은 L/R이고 결측이 없으면 그 손, 아니면 AMBIGUOUS다. 이 투수별 손 등록부를 S1 산출물로 pin하고 정책 식별자에 넣는다. 지원 표 키 (pitcher, batter_side)와 BC 키는 바꾸지 않는다. template은 단일 손 투수에서만 만든다. 런타임은 로깅 전용과 후보 두 모드 모두에서 UNKNOWN_PITCHER 판정 직후, 로깅 법칙 계산 전에 요청의 p_throws를 등록부와 비교한다. 다음 경우 새 거절 상태(가칭 UNSUPPORTED_PITCHER_HAND)를 매긴다. (a) 등록부가 AMBIGUOUS (b) 요청 손이 결측이거나 L/R 밖 (c) 요청 손이 등록부와 다름. 이 거절은 분모에 남고, 기존 refused 메커니즘으로 sticky다. 같은 PA 안에서 손이 바뀌면 한쪽이 반드시 (c)에 걸린다. FAILED_INTEGRITY는 '맥락 행 p_throws ≠ 요청 p_throws' 같은 입력끼리의 불일치에만 쓴다. census는 거절 비용을 보고하는 데만 쓴다.

  - 장점: 단일 손 투수에서는 손이 투수로 완전히 정해진다. 그래서 손을 뺀 π̂_b·지원 표와 손을 넣은 pool·G0·PolicyInputs.support 캐시(matrix_policy.py:145)가 같은 조건부 법칙이 된다. BC 키, P8 재현(BC-E의 1,252,824구·18구종), ρ^ref≡1을 건드리지 않는다 / 거절 기준(투수의 TRAIN 손 이력, 투구 전 p_throws)이 모두 결정 전 정보다. D91/D92의 분류(R3(c)·R7·R8 = UNSUPPORTED, 분모 유지, sticky)와 맞는다. R1 분모와 D-5 [0,1] 경계 구조도 그대로다 / 반대 손 league pool로 조용히 통과하는 경로(옵션 1의 핵심 결함)와 한 행 때문에 run이 멈추는 경로를 둘 다 없앤다
  - 단점: 런타임 상태 집합(D91 ML-POLICY-RUNTIME-v1), DecisionRequest fingerprint, 지원 표(또는 별도 등록부) schema가 바뀐다. 계약 버전 상향, decisions.md 한 줄, 계약·config·테스트 수정이 필요하다. 아직 실데이터 원장이 없으므로 호환성 손실은 합성 원장에 한정된다 / 진짜 양손 투수의 PA는 전부 빠진다. 코딩 오류 몇 행 때문에 출장이 많은 투수가 AMBIGUOUS가 될 수도 있다. 소수 손 행을 다시 라벨링하는 것은 동결된 G0·pool 입력을 바꾸므로 허용하지 않고, 손실은 보고로만 다룬다. 해당 PA 비율은 미측정이다 / '평가 가능 PA 조건부' 점추정의 부분모집단 구성이 바뀌고, 그만큼 전체 구간이 넓어진다(미측정)
  - 추정량·분모 변경: 예, 구현 비용: medium

**4. 손을 넣은 키 (pitcher, p_throws, batter_side) + (선택) 손 조건부 BC** — 지원 표, MaskedReference, 런타임 조회 키를 PolicyInputs.support 캐시 서명과 JointDelivery.TIERS, P8 runner의 template 키 (pitcher, p_throws, stand)에 맞춘다. PAState 또는 DecisionRequest에 손을 넣고, TRAIN에 없던 (pitcher, hand) 쌍은 거절한다. 완전히 일관되려면 CategoricalBC._key에도 손을 넣는다.

  - 장점: pool·G0·지원·(선택 시) 로깅 법칙이 모두 같은 손 조건을 쓰는, 구조적으로 가장 일관된 형태다 / 진짜 양손 투수도 손별로 평가할 수 있어 coverage 손실이 없다 / refuse/intersect 결정 자체가 필요 없어진다
  - 단점: BC 키에 손을 넣으면 로깅 법칙이 바뀐다. 그러면 BC-E의 P8 재현(1,252,824구·18구종)이 깨지고, 셀이 잘게 나뉘어 π̂_b 분산과 ρ 분포가 바뀐다(미측정). BC를 그대로 두면 두 손 투수의 π̂_b 오지정이 남는다 / PAState는 rollout·planning_seed·JointSimulator 공용 타입이다. 손을 추가하면 결정성 해시(D89 §2), 지원 표 schema, load_support_table, check_support, 원장 fingerprint가 연쇄적으로 바뀌어 회귀 위험이 크다 / 영향 크기(양손 투수 수)가 미측정인 상태에서 가장 큰 비용을 먼저 치르는 순서다
  - 추정량·분모 변경: 예, 구현 비용: high

**근거(요약).** 세 렌즈(인과·통계, 구현·자료 위험, 계약·노출 규칙)가 독립적으로 같은 안을 골랐습니다. 핵심 사실은 코드에서 직접 확인했습니다.

**추천이 바뀔 조건.** (a) 다른 이유로 BC를 새 버전으로 다시 적합하게 되면(BC 키 변경이 이미 예정된 경우) P8 재현 비교를 잃는 비용이 사라집니다. 그때는 옵션 4(손 포함 키 + 손 조건부 BC)가 더 낫습니다.  (b) 런타임 상태 집합을 2026 등록 전까지 더 늘릴 수 없다는 제약이 생기면 차선책을 씁니다. 옵션 1의 문구를 코드에 맞게 고치고, p_throws 결측·L/R 밖 값을 pa_requests의 구조 결함(missing_identity 계열, 분모 유지)으로 처리합니다. 다만 반대 손 조용한 통과는 남습니다.  (c) Statcast p_throws가 투구별 실제 손이 아니라 선수 프로필 값으로 확인되면 진짜 양손 투수는 자료에 나타나지 않습니다. 이 경우 규칙은 결측·코드 이상만 막는 안전장치로 축소되지만, 여전히 해가 없으므로 추천은 유지합니다.  (d) S0 census(보고용)에서 AMBIGUOUS로 인한 거절 PA 비율이 크게 나와도 등록된 규칙은 바꾸지 않습니다. 바꾸려면 새 attempt 등록과 이전 시도 공개가 필요하고, 다음 계약 버전에서 옵션 4를 검토하는 근거로만 씁니다.  (e) G0·P8 비교 가능성 때문에 두 손 투수를 반드시 포함해야 한다는 연구 요구가 생기면 옵션 4로 갑니다.

## D-9 G0 P3 τ·탐색 설정 선정 방식

- **사용자 결정:** 3. 2단계 분리: S3로 탐색 기계 설정 고정 → June R1 결과 비열람 Q 원장에서 overlap·잡음 규칙으로 τ 선택
- 추천: 3. 2단계 분리: S3로 탐색 기계 설정 고정 → June R1 결과 비열람 Q 원장에서 overlap·잡음 규칙으로 τ 선택 · 확신도 medium · 반박 담당 판정: 반박 불성립

**1. P8 설정을 그대로 이전(재튜닝 없음)** — τ=.003, search samples 2, pitch_cap 8, planning_seed 701을 'transferred_from_P8_G2_not_reselected' 라벨로 configs/ML-POLICY-MATERIALIZATION-v1.json의 search 필드에 옮긴다. budget만 S3 profile로 정한다. June·DEV를 새로 쓰지 않는다. D89 §7.3의 '예측기 변경 시 τ 재선택'을 뒤집는 것이므로 decisions.md에 번복 한 줄이 필요하다.

  - 장점: 새 노출이 없고 비용이 가장 낮다. WIP run_policy_validation.py 401행이 요구하는 search 5필드를 바로 채울 수 있다 / 자료를 보고 고르는 단계가 없어서 숨은 튜닝이 끼어들 여지가 가장 적다 / 2026 이전에 고정되므로 2026 Δ̂에 선택 편향이 없다
  - 단점: D89 §7.3과 config의 search.p8_reference_not_transferable 표기를 정면으로 뒤집는다 / τ는 절대 WE 단위다(kl_policy ∝ π_ref·exp(Q/τ)). G0 5member Q의 폭·MC 잡음은 G2/G3와 다르고 그 차이는 미측정이라, 같은 .003이 같은 이탈 강도를 뜻하지 않는다 / 옮겨 올 근거 자체가 약하다. P8 June 선정(ML-matrix-execution 보고서, 지원 120타석/77경기)에서 τ별 평균 WE 폭은 .472977~.474604로 .001627이었다. 반면 P0 대비 paired MC SE는 .002410~.002540이었다. 모든 P3 τ가 P0보다 낮았고, DEV P3−P0은 −.000335였다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**2. June 2025 재사용, P8 절차 이식(G0 모델 세계 가치 argmax)** — S3 May profile로 samples·cap·seed·budget·June 시작 수를 정하고, June 값을 열기 전에 실행 config를 커밋한다. 그다음 June 시작 상태에서 G0로 P0/P2/P3(τ grid .001/.003/.01/.03) rollout을 돌린다. P8 규칙('벌점 없는 평균 수비 WE 최대, 동률이면 큰 τ', run_ml_policy.py 297–317행 verify_tuning)으로 τ를 고르고 tuning_freeze로 봉인한다. June 라벨은 exposed_policy_tuning을 유지하고 추가 사용을 기록한다.

  - 장점: D89 §7.3의 문구(재선택, June에 라벨 붙여 사용)와 P8 선례(profile → config 동결 → June 선정 → tuning_freeze)를 그대로 따른다 / June 결과 라벨(WE·events)을 읽지 않는다. P8에서도 selection_uses_outcomes=False였다 / DEV를 S4–S6 검증용으로 온전히 남긴다
  - 단점: 자기 참조 문제가 있다. P8은 G3(simulators['candidate'])로 계획하고 G2 control 세계로 평가한 교차 모델 구조였다(run_ml_policy.py 441–449행, 478행 'Primary cross-model internal'). G0는 세계가 하나라 계획과 평가가 같은 오차를 공유하므로 탐욕적인(작은) τ로 기운다. 게다가 G0 June 가중치(0.7467…)는 June에서 적합됐다 / 가치 기준은 2026 DR에서 실제로 문제가 되는 ρ^cand 분산·ESS를 보지 않는다. 선택된 τ가 D87 overlap 게이트(PA ESS≥100, game ESS≥30)에 미달하면 UNCONFIRMED_WEAK_OVERLAP이 된다 / P8 러너는 소급 supported_pa로 거른 패널 시작(1,338타석 요청 중 128개 선택, 120개 지원)에서 튜닝했다. 이는 R1(모든 PA 0-0 시작)과 다른, 처치 후 선택된 모집단이다
  - 추정량·분모 변경: 아니오, 구현 비용: high

**3. 2단계 분리: S3로 탐색 기계 설정 고정 → June R1 결과 비열람 Q 원장에서 overlap·잡음 규칙으로 τ 선택** — D-9a: samples·pitch_cap·budget은 S3 May profile(품질 값 비열람)에서 잰 조건부 행/초만으로 정한다. 사전 등록 후보 목록 중 stage 상한 안에 드는 가장 큰 값을 고른다. planning_seed는 사전 상수로 둔다. D-9b(새 stage S3b): June 2025에서 R1 규칙대로 PA 요청을 만든다. 경기 단위 사전 해시 부분 표본을 쓰고 supported_pa 필터는 쓰지 않는다. 이 요청을 후보 모드 런타임에 한 번 제출한다. 원장에는 q_reference·q_mc_se·mask·π_ref·π̂_b·로그 구종만 남기고, 보상·WE·events는 읽지 않는다. Q̂^ref는 τ와 무관하므로(policy_runtime.py 391–396행: q_values를 한 번 계산하고 kl_policy만 적용) 고정된 계산 함수로 grid 모든 τ의 π_cand를 다시 계산한다. 이어서 D87 식의 PA/game ESS, ESS/n, 평균 KL(π_cand‖π_ref), 잡음 비 q_mc_se/τ를 구한다. 사전 등록한 overlap 문턱과 잡음 문턱을 모두 만족하는 τ 중 가장 작은 값(가장 적극적인 값)을 고른다. 만족하는 τ가 없으면 G0 P3를 '평가 불가'로 기록하고, grid를 넓히거나 P2로 자동 전환하지 않는다. G0 모델 세계 가치는 선택에 쓰지 않는다.

  - 장점: 선택 기준이 2026 DR의 실제 실패 모드(ρ^cand 꼬리, D87 overlap 게이트)를 직접 겨냥한다. D87 표 63행이 '≤2025/합성 작업으로 근거를 만든 뒤 고정'하라고 한 게이트의 근거 자료도 함께 만든다 / 결과를 보지 않고 설계를 정한다. 보상을 읽지 않으므로 winner's curse가 없고, G0 June 가중치의 in-sample 낙관도 τ로 새지 않는다. 한 모델로 자기 평가하는 순환도 피한다 / June은 BC·pool·G0 member 적합(TRAIN) 밖이다. 그래서 TRAIN 안에서 계산할 때보다 π̂_b overlap을 낙관하지 않는다
  - 단점: 가치 개선을 보장하지 않는다. 제약을 만족하는 가장 작은 τ라도 Δ≈0일 수 있고, 이는 S5 V2에서야 알 수 있다 / overlap 문턱(ESS/n 하한 또는 게이트 여유), 잡음 비 문턱, 부분 표본 크기에 실측 근거가 없다. 이 값들은 '설계 규약값(측정 아님)'으로 라벨을 붙여 June 원장을 열기 전에 등록해야 한다 / June과 2026 사이의 π_b 이동(새 구종, 레퍼토리 변화) 때문에 June ESS가 2026 ESS를 과대 추정할 수 있다(미측정)
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**4. τ 여러 개를 사전 등록하고 선택 없이 family로 보고** — τ 2~3개를 각각 별도 candidate identity로 등록한다. DEV·2026에서 모두 추정해 Holm으로 보고하고, 하나를 골라 승격하지 않는다. June은 쓰지 않는다.

  - 장점: June 재노출도 선택 단계도 없다 / τ 민감도가 사전 등록된 형태로 드러난다
  - 단점: 추정 대상이 Δ 하나에서 family로 바뀐다. D87의 'family당 주 대비 1개'와 충돌하고 검정력이 분산된다 / 2026 결과를 본 뒤 '가장 좋은 τ'를 인용하면 그 자체가 2026 튜닝이 되어 절대 규칙 위반이다 / build_runtime과 identity는 후보 하나만 받는다. 다중 후보 런타임을 새로 구현하거나, τ마다 q̂^cand continuation MC를 따로 돌려야 한다. 이 경우 2026 비용이 k배로 늘어난다(미측정)
  - 추정량·분모 변경: 예, 구현 비용: high

**근거(요약).** 질문의 핵심은 'June을 다시 쓰면 2026이 편향되느냐'가 아닙니다. τ는 2026 이전에 고정되므로 어느 ≤2025 창을 쓰든 2026 Δ̂는 고정된 π_τ에 대해 선택 편향이 없습니다. 실제로 갈리는 점은 두 가지입니다. τ를 무엇을 기준으로 고르는가, 그리고 고르는 모집단이 2026 평가 모집단과 같은 규칙을 따르는가입니다.

**추천이 바뀔 조건.** - S3 profile에서 G0 5member의 결정당 비용이 너무 커서, 등록 상한 안의 June 부분 표본이 D87 게이트 규모의 ESS를 계산할 수 없을 만큼 작아진다고 가정해 보자. 그러면 June 대신 May(exposed_calibration, 2025-05-16..05-31)에 같은 규칙을 적용하는 변형으로 가거나, P3 후보 자체를 보류한다. 1·2안으로는 가지 않는다. - 감당 가능한 최대 samples에서도 q_mc_se가 grid 모든 τ에 비해 커서 잡음 문턱을 아무 τ도 통과하지 못하면, G0 P3는 '평가 불가'로 기록한다. 다른 후보(예: P2)를 쓸지는 새 결정 항목으로 다룬다. - Song이 June 추가 사용을 원치 않으면 May 변형을 쓰거나, 1안(decisions.md 번복 한 줄 포함)을 '근거 없는 이전'이라고 명시하고 택한다. - 연구 책임자가 OPE 가능성보다 모델 세계 개선 신호를 우선한다면 혼합안이 가능하다. 3안의 overlap·잡음 제약을 통과한 τ 중에서 G0 모델 세계 가치 최대를 고르는 방식이다. 다만 자기 평가 낙관이 남고, 비용은 통과 τ 수만큼 늘어난다. - D-2에서 분모가 R1이 아닌 쪽으로 정해지면 June 요청 모집단도 그 규칙을 따른다. - D87 overlap 게이트 정의(PA/game ESS 식·문턱)가 바뀌면 3안의 문턱도 같이 바꾼다. 단 June 원장을 열기 전에만 바꿀 수 있다. - Astra 독립 검토가 τ 선정에 π̂_b를 쓰는 것을 식별 가정 (iv)의 순환으로 판정하면, π_cand/π_ref만 쓰는 KL·ratio 상한 규칙(π̂_b 비사용)으로 좁힌다.

## D-10 V2 세계의 M 밖 로깅 행동: ρ=0 흡수 종료

- **사용자 결정:** B. ρ=0 흡수 종료를 등록 세계로 쓰고, 합성 불변성 테스트로 고정
- 추천: B. ρ=0 흡수 종료를 등록 세계로 쓰고, 합성 불변성 테스트로 고정 · 확신도 high · 반박 담당 판정: 반박 불성립

**A. 현재 제안: tier pool + league fallback으로 PA 계속 진행 (WIP world_pool rule='fallback')** — M 밖 로깅 행동이 나오면 먼저 PolicyInputs.pool의 tier pool을 쓰고, 없으면 JointDelivery.fallback(구종·손·타자면을 섞은 TRAIN league 400 draw)으로 delivery를 채운 뒤 G0로 전이를 만들어 PA를 이어 간다. token 어휘(type_map 18개) 밖 행동은 WorldRefused가 나고, 현재 run_world(policy_semisynthetic.py 79–82행)는 그 로그만 `continue`로 버린 뒤 남은 로그로 delta_dr_mean을 낸다. 이 안을 쓰려면 적어도 '거절 1건 이상 = run 실패'로 고쳐야 한다. 고치고 나면 BC-P 어휘에 token 밖 코드가 있을 때 V2가 NOT_EXECUTABLE이 된다.

  - 장점: 이미 구현돼 있고 합성 테스트(test_v2_known_logging_law_matches_world_truth, CH를 fallback으로 전달)를 통과했다 / 로깅 법칙을 M으로 재정규화하지 않는다. token 어휘 안의 행동에 대해서는 D89 V2 전제 문구('모든 행동의 delivery·전이')를 문자 그대로 만족한다 / M 밖 행동 뒤에도 결정이 런타임에 계속 제출된다. 그래서 이력에 M 밖 구종이 든 상태의 런타임 경로가 V2에서 실행된다
  - 단점: WIP 코드 그대로 두면 처치 후 선택이 생긴다. token 밖 코드를 뽑은 로그만 버리면 남은 궤적은 '그 코드를 뽑지 않은 사건'에 조건부가 되고, 전체 어휘 π̂_b를 쓰는 DR과 어긋나 V2 비교 자체가 편향된다(D89 §5가 금지한 경로). 테스트 test_world_refuse_rule_and_v3_transport는 refused>0인지만 보고 결과를 무효로 만들지 않는다 / world_pool의 넓은 `except ValueError`는 context identity 불일치 같은 무결성 오류도 조용히 fallback delivery로 바꾼다 / fallback 물리는 구종과 무관하다. one-hot은 X인데 물리는 섞인 분포인 OOD 입력을 G0에 넣게 된다. ρ=0 뒤라 값에는 영향이 없지만 PA 길이·절단·비용 같은 V2 진단은 오염된다
  - 추정량·분모 변경: 예, 구현 비용: low

**B. ρ=0 흡수 종료를 등록 세계로 쓰고, 합성 불변성 테스트로 고정 (추천)** — 생성 법칙(V2는 전체 어휘 π̂_b, V3는 등록된 대체 법칙)은 재정규화하지 않는다. 로깅 행동 a_t가 M 밖이면 그 결정 요청까지는 런타임에 제출해 OUTSIDE_POLICY_SUPPORT, ρ^cand=ρ^ref=0, q̂(H_t)를 기록한다. 그 뒤 PA는 선언된 흡수 전이로 끝낸다. 보상은 run_world가 cap 절단에 이미 쓰는 components.we.cutoff(H_t)로 두고, outcome에 absorbed_off_mask 플래그를 단다. M 안 행동은 정책과 같은 components.inputs.pool만 쓴다. world_pool, league fallback, WorldRefused 폐기 경로는 없앤다. 근거는 코드에 있다. pa_dr(policy_estimator.py 80행)은 ρ=0이면 value = p[mask]@q[mask] + 0이라 이후 행·보상을 버리고 가중치도 0이 된다. _step(47행)과 runtime._row(184행)가 두 정책의 M 밖 질량을 0으로 강제한다. truth rollout의 reference는 improvement.bc = MaskedReference(policy_runtime.py 386–387행)라 M 밖 전이를 방문하지 않는다. 따라서 같은 prefix에서 PA별 DR 값, 가중치, ESS, 세계 참값이 fallback 세계와 정확히 같다.

  - 장점: 추정량과 PA 분모가 바뀌지 않는다. 로그 폐기도 재정규화도 없고, 흡수된 PA는 COMPLETE로 분모와 ESS에 남는다 / token 밖 코드를 포함한 모든 π̂_b>0 행동에 전이를 정의한다. 그래서 D-1(BC-P/BC-E) 선택과 독립이다 / OOD fallback 물리, ρ=0 뒤의 Q MC 비용, pool_tiers 진단 오염, 'fallback 미사용' 규칙과의 이름 충돌이 모두 사라진다
  - 단점: D89 V2 전제 문구('모든 행동의 delivery·전이 정의')를 '전이 = 흡수 종료, delivery 불필요'로 해석하는 설계 결정이다. decisions.md 한 줄과 보고서 §4 58행, 계약 §5 S5 행·§6 D-10 행, config decisions.D-10의 동시 수정이 필요하다 / M 밖 행동 뒤에 이어지는 실데이터형 결정 행(이력의 M 밖·token 밖 구종 → token slot 0, 원장 이력 검사)은 V2에서 실행되지 않는다. 이 경로는 S4(V5)와 S6(V4)에서 처음 실행된다 / 정확성은 두 조건에 달려 있다. 추정기가 clip·자기정규화 없는 D89 plain DR이어야 하고, 후보와 기준의 지원이 둘 다 M이어야 한다. 정책 형태(ARM-A 등)나 추정기(WDR, clip, SNIPS)가 바뀌면 다시 등록해야 한다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**C. B에 더해, 결과 열람 전에 등록한 소수 DEV 시작에서 fallback 계속 진행과의 짝 불변성 probe 실행** — 등록 추정과 ESS는 B로 낸다. 결과를 보기 전에 규모·시작·seed·허용 오차를 등록한 소수 DEV PA 시작에서, 같은 PA별 RNG substream으로 두 PA를 만든다. 하나는 token 안 M 밖 행동을 tier+fallback으로 계속 진행하고(token 밖은 흡수), 다른 하나는 흡수 PA다. 합격 조건은 두 가지다. PA별 pa_dr와 가중치가 비트 단위로 같아야 하고, 계속 진행한 행이 FAILED_*나 이후 거절 없이 원장에 들어가야 한다. 어긋나면 V2 전체가 FAILED_INTEGRITY다. probe의 fallback 값은 추정에 쓰지 않는다.

  - 장점: B의 통계적 장점을 모두 유지한다 / 실제 G0·pool·런타임 연결에서, 이력에 M 밖 구종이 든 상태가 실패하지 않는지 2026 전에 한 번 확인한다 / fallback 코드를 probe 전용으로 격리해 결과 경로로 새지 않게 한다
  - 단점: world_pool·fallback 코드와 그 identity를 남겨야 해서 코드와 pin 표면이 늘어난다. 넓은 except도 'pool 없음'만 잡게 좁혀야 한다 / DEV probe 비용이 추가된다(미측정). 짝 로직과 등록 필드도 추가된다 / token 밖 코드의 계속 경로는 probe에서도 검사하지 못한다
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**D. tier 전용 세계 + 시작 단위 사전 제외 (D89 문구를 가장 보수적으로 읽기)** — fallback도 흡수도 쓰지 않는다. 등록된 DEV PA 시작마다 생성 전에 (투수, p_throws, stand)에서 π̂_b>0인 모든 행동이 token 어휘 안에 있고 12카운트 모두에 action-specific pool을 갖는지 S1 표로 검사한다. 하나라도 아니면 그 시작 전체를 V2_WORLD_UNDEFINED로 분모에 남겨 층으로 보고한다. V2는 정의된 시작에서만 돌며, 로그 단위 폐기는 금지한다.

  - 장점: D89 V2 전제를 문구 그대로 만족하고 세계 가정을 새로 만들지 않는다 / 결정 전 공변량(투수·손·타자면·동결 pool)만으로 거르므로 처치 후 선택이 없다 / 정책 식별자와 같은 pool만 쓰고 OOD 물리가 없다. 구현도 S1 표 검사 하나다
  - 단점: tier 전용 세계에서 'delivery 가능'은 사실상 M과 같다. 그래서 살아남는 시작은 π̂_b(M|H)≈1, 즉 ρ^ref≈1이 되어 V2가 D89의 핵심인 ρ^ref≠1 경로를 거의 검사하지 못한다 / 빠지는 시작은 M 밖 질량이 있는 투수 쪽이다. 따라서 V2 ESS 분포가 DEV나 2026보다 낙관적이 되고 BLK-08 근거로 약하다 / V2 추정 대상이 '세계 정의 시작 조건부 Δ'로 바뀐다. D-2(R1: 모든 정규시즌 PA)와 어긋나는 분모 변경이다
  - 추정량·분모 변경: 예, 구현 비용: low

**근거(요약).** 추천은 B이고, 세 렌즈 모두 같은 결론이다(렌즈1 C, 렌즈2 B/C, 렌즈3 B).

**추천이 바뀔 조건.** - 추정기가 WDR, clip, SNIPS, 자기정규화 쪽으로 바뀌거나, 후보 지원이 기준 지원(M)과 달라지면(ARM-A, 다른 마스크, argmax) 흡수 세계는 더 이상 동치가 아니다. 이때는 A(fail-closed로 고친 것)나 별도 전이 정의로 돌아가 다시 등록해야 한다. - Song이 D89 V2 전제를 문자 그대로 유지하기로 하면 A가 된다. 이 경우에도 거절 1건 이상은 run 실패로 처리한다. BC-P에 token 밖 코드가 있으면 V2는 NOT_EXECUTABLE이거나 D-1을 BC-E로 옮겨야 한다. - S4(V5) 실측에서 ρ=0 뒤 계속 행의 거절(NO_PITCH, 교체, 이력 불일치 → sticky UNSUPPORTED)이 흔하다고 나오면, 그 경로를 V2 단계에서 미리 점검할 가치가 커진다. 그러면 C(DEV probe 추가)로 올린다. - S0/S1에서 π̂_b(M|H)≈1이 거의 모든 시작에 해당하면 D-10은 사실상 무의미해진다. 이때도 B가 가장 싸지만, V2가 ρ^ref≠1을 거의 검사하지 못한다는 한계를 결과에 명시해야 한다. - 합성 불변성 테스트가 비트 단위 일치에 실패하면 원인(RNG 공유, planning_seed 비결정성, 수치 누설)을 해결하기 전까지 B를 등록하지 않는다.

## D-11 stage 비용 상한의 집행·계정 방식과 S4 profile gate

- **사용자 결정:** D. 측정 먼저: label-blind stage는 상한 없이(또는 hang 방지용 큰 값) 완결성만 강제 (추천안 C 대신 사용자가 D 선택)
- 추천: C. B + S4 앞 cost-only profile(S4a)과 사전 고정 공식 gate + 완결성 봉인 + 범위 축소 금지 · 확신도 medium · 반박 담당 판정: 반박 불성립

**A. 현재 제안 그대로: 고정 벽시계 상한 5개, family = 합계** — config `le2025_validation_plan.stages.*.cap_seconds`에 S0 900, S1 1,800, S2 1,200, S3 1,800, S4 3,600초를 등록하고 `family_cap_seconds_proposed` 9,300초를 family 상한으로 쓴다. 러너는 a704983 WIP 그대로 둔다. 이때 `Deadline`은 S3 profile(419행), S4 v5-denominators(398행), dr-evaluate(426행)에만 걸리고, `stage()` 타이머는 `load_inputs`와 heavy lock 획득이 끝난 뒤에야 시작한다. 모든 값은 미측정 제안값이다.

  - 장점: config와 계약 §5에 이미 들어 있어 추가 작업이 거의 없다 / 상한을 넘으면 조용히 자르지 않고 BudgetExceeded로 끝난다. stage()가 failure-*.json을 남기므로, 상한을 지키는 동안에는 추정량과 분모가 바뀌지 않는다(§2: budget은 identity 밖) / S0–S2와 S4는 label-blind라서 상한 값을 정하는 일이 결과 열람과 무관하다
  - 단점: S0–S2 상한은 선언만 있고 집행되지 않는다. census, materialize-bc, bind-probe 분기에는 Deadline이 없다. fit_bc나 pickle 로드처럼 한 번에 끝나는 호출은 in-process 검사로 끊을 수도 없다 / main()이 load_inputs(처리 캐시 frame, MatrixHistoryStore, aux pickle)를 dispatch 안의 `with heavy_lock(root), stage(...)`보다 먼저 호출한다. 그래서 로드 시간은 어떤 상한에도 잡히지 않는다. 로드 중에 실패하거나 LOCK_NB 잠금 충돌이 나면 failure 기록이 남지 않아 보이지 않는 attempt가 생긴다. 24GB 맥미니에서 heavy 로드 두 개가 동시에 돌 수도 있다 / 측정 경계가 P8(로드 포함 TimedBudget)이나 D86(Popen→wait 합)과 달라서 참고 실측과 같은 척도로 비교할 수 없다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**B. 숫자는 유지하고 측정 경계·집행·family 계정을 보강** — 다섯 숫자는 제안값으로 그대로 둔다. 대신 무엇을 어떻게 재는지를 등록한다. (1) 공식 비용은 프로세스 시작부터 manifest 봉인까지 걸린 벽시계 시간이다. load_inputs, bind, verify_components를 포함하며 D86의 Popen→wait와 같은 경계다. (2) heavy lock을 load_inputs보다 먼저 잡고, 로드도 stage() 안에서 해서 로드 실패도 failure-*.json으로 남긴다. (3) S0–S4 전 stage에서 상한을 집행한다. 외부 감독기가 subprocess timeout을 걸고, 반복문 stage는 요청 단위로 Deadline을 검사한다. (4) family 9,300초는 실패 attempt를 포함한 누적 계정으로 바꾼다. 끝난 attempt는 실제 wall을, 끝나지 않았거나 강제 종료된 attempt는 cap 전액을 계정한다. (5) manifest에 load·bind·loop 초, CPU 초, peak RSS, 요청 수, 원장 바이트를 기록한다. (6) config 키를 통일하고, §5 문구를 '벽시계 초과 = stage failure'로 고친다.

  - 장점: 지금 적힌 상한이 실제로 집행된다. 기록된 비용을 P8이나 D86 참고값과 같은 경계로 비교할 수 있다 / 잠금을 로드보다 먼저 잡으므로 단일 heavy 큐 가정이 코드에서도 참이 된다 / 실패와 미해결 attempt까지 계정하므로 family 상한이 실제로 구속력을 갖는다. D86식 '실패·재시도 0회' 보고도 가능해진다
  - 단점: 다섯 숫자의 근거는 여전히 없다. 특히 S4는 profile 없이 확정되어 D89 보고서 §4의 'profile 먼저'와 계속 어긋난다 / 로드를 포함해서 재므로 같은 숫자라도 실제로는 더 빡빡한 상한이 된다. 로드 시간도 미측정이다 / 외부 감독기, 잠금 순서 재배치, RSS 기록, schema 검사를 새로 구현하고 합성 테스트를 붙여야 한다
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**C. B + S4 앞 cost-only profile(S4a)과 사전 고정 공식 gate + 완결성 봉인 + 범위 축소 금지** — B의 (1)–(6)을 모두 적용하고 다음을 더한다. (7) S4a v5-profile stage를 새로 둔다. DEV 경기를 canonical_hash(planning_seed, game_pk) 순으로 정렬해 앞의 k개만 제출한다. 산출물은 비용 필드로 제한한다: 로드 포함 wall·CPU, 요청 수, 요청당 초, fsync 횟수와 행당 초, 원장 바이트, peak RSS. 상태 분포와 π̂_b(M|H) 요약은 만들지 않는다. 원장은 해시만 기록하고 봉인한다. 라벨은 `exposed_development`, '비용 전용, 분모 아님'으로 단다. S4a 자체의 상한도 사전에 등록한다. (8) S4 상한은 지금 공식으로 고정한다: projected = 로드·bind 고정비 + 요청당 초 × S0 census의 DEV rows(요청 수 상한), cap_S4 = min(3,600, 배수 × projected). 배수를 곱하기 전의 projected만으로도 3,600을 넘으면 S4를 시작하지 않는다. 그때는 decisions.md에 한 행을 쓰고 재등록한다. k와 배수는 제안값으로만 등록하고 근거는 '미측정'으로 표시한다. (9) S3은 그 자체가 비용 profile이다. S3의 상한은 D-9 등록 때 RowBudget(조건부 행, 결정적 카운터)을 주 예산으로 함께 정하고, 벽시계는 P8 TimedBudget처럼 consume마다 검사한다. (10) S4 결과는 두 조건이 맞을 때만 봉인한다. 제출 PA와 unsubmittable PA를 더한 값이 S0 census의 DEV `pas`와 같아야 하고, 원장 decision 행 수가 제출 요청 수와 같아야 한다. 실패한 stage의 부분 원장은 경기 순서의 prefix이므로 인용을 금지한다고 failure 기록에 표시한다. (11) 상한 실패 뒤에는 범위(경기·기간·PA)를 그대로 두고, 상한이나 성능만 고친 새 attempt만 허용한다. subset 축소는 금지한다. stage당 attempt 상한도 등록한다(제안값).

  - 장점: D89 보고서 §4의 'profile 먼저'와 계약 §5 S4행의 '한 경기 subset profile'을 실제 stage로 만든다. D59, D66, D76, D86에서 쓴 'profile → 외삽 → gate → 실행' 관례와도 같은 모양이다 / 공식과 배수를 데이터를 보기 전에 고정하므로, profile을 본 뒤 숫자를 고르는 분기가 없다 / 비용 위험이 가장 큰 per-row fsync 원장을 전체 DEV를 돌리기 전에 잰다. 그래서 치명적인 attempt 손실이 공개 이력에 쌓이는 일이 줄어든다
  - 단점: stage, subcommand, 등록 단계가 하나씩 늘어난다. 경기 선택 규칙, 비용 전용 출력, census 대조 봉인에 테스트가 필요하다 / k경기로 DEV 1,161경기를 외삽하므로 오차가 클 수 있다(미측정). 경기 길이 차이, 캐시 워밍업, 원장이 커질수록 Ledger.rows와 by_request 메모리가 늘어나는 효과를 배수가 흡수해야 한다 / S4a 원장에도 상태가 기록되므로 '보지 않는다'는 것은 봉인과 산출 필드 제한으로만 지켜진다
  - 추정량·분모 변경: 아니오, 구현 비용: medium

**D. 측정 먼저: label-blind stage는 상한 없이(또는 hang 방지용 큰 값) 완결성만 강제** — S0, S1, S2, S4를 느슨한 안전 timeout만 걸고 한 번씩 돌려 비용을 기록한다. 완결성(census 대조)과 fail-closed만 강제한다. 벽시계 상한은 실측을 본 뒤에 정하고, 결정적 상한은 후보 rollout(S3, 이후 V2/V4)의 RowBudget에만 둔다.

  - 장점: 미측정 숫자를 지어낼 필요가 없고 실측만 남는다 / 상한 실패와 재등록이 거의 없어져 범위를 줄이고 싶은 유혹이 원천적으로 줄어든다 / S0–S4는 추정값을 만들지 않으므로 타당성 손실이 없다
  - 단점: D75와 D80에서 이어진 '실행 전 비용 상한 등록' 관행, 그리고 D89 §4의 '예산 등록'과 충돌한다 / 상한 없이 긴 작업이 단일 heavy lock을 잡으면 맥미니 큐 전체가 막힌다. 다른 작업은 LOCK_NB 때문에 곧바로 실패한다 / 실행 뒤에 정한 상한은 사전 고정이라 부를 수 없다
  - 추정량·분모 변경: 아니오, 구현 비용: low

**근거(요약).** D-11은 추정량이 아니라 실패했을 때의 동작을 정하는 결정입니다. 계약 §2에 따라 budget은 identity에 들어가지 않고, 상한을 넘으면 잘라서 끝내지 않고 실패로 처리합니다. 그래서 어느 선택지를 골라도 R1/R2 분모와 추정량 자체는 바뀌지 않습니다. 중요한 것은 세 가지입니다. 상한이 실제로 집행되는지, 무엇을 재는지, 그리고 실패 뒤에 무엇이 허용되는지입니다.

**추천이 바뀔 조건.** - B로 내려가는 경우: Song이 V5를 기술 통계일 뿐이라 보고 S4a stage를 추가하는 비용이 크다고 판단하거나, 다른 범위에서라도 행당 fsync와 요청 생성 비용이 이미 측정되어 3,600초가 넉넉하다는 근거가 문서로 생기면, S4a 없이 B(집행·계정 보강과 범위 축소 금지)로 충분합니다. - fsync 완화를 별도 결정으로 올리는 경우: S4a에서 fsync가 비용을 지배하고, 공식 값이 배수를 곱하기 전에도 3,600을 넘으면 렌즈 2의 fsync 완화(파일 핸들 유지, PA 경계에서 fsync)를 별도 D행으로 다룹니다. 이 변경은 D91의 record-then-return 계약과 2026 경로를 바꾸므로 D-11 안에서 처리하지 않습니다. - 외부 감독기를 미루는 경우: 감독기와 subprocess timeout 구현 비용이 너무 크면, main() 시작 시각을 기준으로 in-process에서 재는 방식으로 대체할 수 있습니다. 단, 한 번에 끝나는 호출(fit_bc, pickle 로드)은 끊지 못한다는 한계를 기록해야 합니다. - S3 상한을 지금 정하는 경우: D-9에서 τ 없이 P2만 쓰는 profile을 허용하도록 러너를 고치기로 하면, S3 상한도 지금 벽시계 값으로 확정할 수 있습니다.

## 목록에 빠졌던 결정 M-1~M-13 (모두 추천안 채택)

**M-1. S5·S6 표본 설계: 평가할 경기/PA 시작 집합과 크기, S5·S6 비용 상한**

- 필요 이유: D-11은 S0–S4 상한만 다루고, config의 S5_V2_V3·S6_V4 cap_seconds는 null이다. 그런데 `run_policy_validation.py`의 `dr-evaluate`는 DEV split의 모든 PA(`preq.pa_blocks(... split=='dev')`, 341,941행·1,161경기)를 제출하고 결정마다 후보 MC Q̂를 계산한다. 러너는 없는 키 `caps['V4_dr_evaluate']`를 읽는다(config 이름은 `S6_V4`). 상한을 넘으면 BudgetExceeded가 stage 전체를 FAILED_RUNTIME으로 만들고 부분 결과는 남지 않는다. 참고로 P8은 DEV 128 시작, 시작 시점만, 3 member로 71,408 조건부 행·85초였다. V4처럼 모든 결정에 5 member를 쓰는 비용은 미측정이다. 결과를 본 뒤 표본을 줄이면 금지된 사후 변경이 된다. S4의 '한 경기 subset profile'도 어느 경기를 쓸지 정해져 있지 않다.
- 선택지: (a) DEV 전체 PA. 상한은 S3 속도로 외삽 | (b) 등록된 salt의 game_pk 해시로 고른 완전 경기 집합(경기 안 모든 PA 유지, 월 층화). 경기 수는 S3 전에 등록한 공식(측정 속도×상한÷경기당 결정 수×안전계수)으로 정함 | (c) P8 `select_pa_requests`처럼 역할 층화 해시로 고른 PA 시작 표본(교체 없음)
- 채택: (b)를 권합니다. R1/D-2 분모(선택 경기 안의 모든 PA)와 경기 단위 부트스트랩이 함께 유지됩니다. salt·층·크기 공식·안전계수는 S3 전에 등록하고, S3는 라벨 없이 잰 비용만 공식에 넣습니다. V2 시작 집합도 같은 방식(DEV PA 시작, 해시·층화)으로 정합니다. S4용 한 경기 profile도 같은 해시 순서의 첫 경기로 고정합니다.

**M-2. 등록의 단계화·수정 절차와 stage 사이 pin 전달**

- 필요 이유: 러너는 `config['decisions']`(D-1~D-8만 검사), `registered_inputs.bc/support`(S1 산출물 SHA), `expected_identity_sha256`(S2에서 처음 생김), `profile_as_of.dev`, `bootstrap.draws/seed`, `le2025_validation_plan.local_config.sha256`, `V4_dr_evaluate`를 요구한다. 지금 config에는 이 키가 하나도 없다. S1·S2 결과를 pin하려면 family 도중 config가 바뀌어야 하고, 그러면 stage identity의 `config_sha256`도 바뀐다. S0에서 어휘 밖 코드가 나오면 R3(b)는 '재등록'을 요구한다. 어떤 필드를 어느 stage 뒤에, 누구 승인으로 채울 수 있는지 정해지지 않았다. results/·experiments.md 기록(시드·데이터 버전·커밋) 단위도 정해지지 않았다.
- 선택지: (a) 기본 등록 1개(모든 문턱·표본·seed 확정) + stage별 append-only addendum. addendum에는 봉인된 manifest의 SHA pin만 넣고 부모 SHA로 사슬을 만듦 | (b) stage마다 config 전체를 재등록 | (c) 자리표시자를 두고 러너가 봉인 manifest에서 자동으로 채움
- 채택: (a)를 권합니다. addendum에는 SHA pin만 허용하고 문턱·표본·후보 필드는 금지합니다. addendum마다 Song이 승인하고 decisions에 한 줄 남깁니다. stage마다 요약을 results/ML-POLICY-VAL-v1-<stage>.json으로 커밋합니다(시드·processed_sha256·커밋 포함).

**M-3. 독립 검토 요건의 처리(거버넌스)**

- 필요 이유: 계약 §6은 'Astra 독립 과학 검토 1회와 Song 결정 전에는 등록·실행하지 않는다'고 한다. 그런데 config의 `independent_review`는 `pending_sol_astra_unavailable_in_this_environment`이다. 커밋 a704983(요청 생성기·DR·V2/V3·러너)은 'WIP, unreviewed'이고 적대적 코드 검토도 수행되지 않았다. 어떤 검토로 어느 stage를 열 수 있는지 정하지 않으면 등록 자체가 성립하지 않는다.
- 선택지: (a) Codex 협업 환경에서 Sol 재현·Astra 검토가 끝날 때까지 전체 대기 | (b) 라벨을 보지 않는 S0–S2만 코드 검토 뒤 먼저 열고, S3 이후는 독립 검토 뒤에 엶 | (c) Song이 승인한 대체 검토(다른 모델)를 '독립 검토 아님'으로 표시하고 진행
- 채택: (b)를 권합니다. S0/S1/S2는 결과 라벨을 읽지 않습니다(S2는 봉인 예측의 primary/level만 읽음). 결과 라벨을 여는 S6와 합격 판정이 걸린 S5는 독립 검토 뒤에만 엽니다.

**M-4. 불확실성 명세: B·seed·재표집 단위·비평가 PA 포함 여부·무효 replicate·최소 표본**

- 필요 이유: D87 §5는 'seed·무효 replicate 처리는 등록 시 고정(현재 null)'이라고 적어 두었다. 러너가 읽는 `plan['bootstrap']`는 config에 없다. `policy_estimator.game_bootstrap`는 COMPLETE PA만, 그것이 있는 경기만 재표집한다. 그래서 평가 가능 비율의 변동이 CI에 들어가지 않는다. 또 2경기만 있어도 CI를 낸다. P8의 `POLICY_INFERENCE`는 30경기·50 시작 미만이면 추론 칸을 null로 뒀다. population bound에는 CI가 없다. Q̂ MC 오차의 전파 방식도 정해지지 않았다.
- 선택지: 재표집 단위: 경기 / 투수 군집 / 두 방향 군집 | 대상: COMPLETE PA만 / 제출한 모든 PA(조건부 평균과 경계를 replicate마다 다시 계산) | B·seed: P8 선례(10,000, 20260924) 재사용 / 새 seed | 무효 replicate(평가 가능 PA 0개): 제외하고 개수 보고 / 실행 실패 | 최소 표본: P8 기준(30경기·50 PA) / 새 값
- 채택: 경기 단위, B=10,000, 새로 등록한 seed를 권합니다. 선택 경기의 모든 PA를 재표집하고 replicate마다 조건부 평균과 [0,1] 경계를 다시 계산합니다. 무효 replicate는 제외하고 개수를 보고하되, 등록한 비율을 넘으면 CI를 null로 둡니다. P8 최소 표본을 재사용합니다. 투수 군집 부트스트랩은 민감도로 두고, MC SE는 따로 보고합니다.

**M-5. ESS 계산·사용 규칙과 V2 결과를 2026 BLK-08 문턱으로 옮기는 사전 규칙**

- 필요 이유: 보고서 §4 V2의 목적은 'ESS 분포를 측정해 BLK-08 게이트 근거 마련'이다. D87의 PA ESS≥100·game ESS≥30은 '≤2025/합성 작업 뒤 고정'할 휴리스틱이다. `effective_sample_size`는 후보·기준 각각의 PA 누적 비로 계산된다. 어느 정책의 ESS를 게이트로 쓸지, ≤2025 V4에서 `UNCONFIRMED_WEAK_OVERLAP` 라벨을 붙일지는 정해지지 않았다. V2 결과를 보고 옮기는 규칙을 정하면 문턱이 사후 조정된다.
- 선택지: 게이트 대상: 후보 ESS / min(후보, 기준) / 둘 다 보고만 | ≤2025 V4에서: 라벨만 / 중단 | BLK-08 도출: D87 값 유지하고 V2는 보고만 / V2에서 편향·CI 포함률이 허용 범위에 드는 최소 ESS 구간을 채택하는 규칙
- 채택: PA와 game 모두 min(후보, 기준) ESS로 게이트합니다. ≤2025에서는 라벨만 붙이고 중단하지 않습니다. V2→BLK-08 도출 규칙(허용 편향·포함률 기준과 ESS 구간 폭)은 S5 전에 등록하고, 수치는 V2 전까지 null로 둡니다.

**M-6. V2 합격 기준과 V2/V3 세계 설계 파라미터(S5 실행 경로 포함)**

- 필요 이유: `policy_semisynthetic.run_world`는 starts, logs_per_start, cap, truth_rollouts, seed, draws가 필요하다. 합격 기준은 등록돼 있지 않다(합성 테스트는 시작별 4·SE, 커밋 메시지는 약 1.5 SE). V3의 대안 로깅 법칙도 정해지지 않았다(D89는 빈도 법칙·섭동만 언급). cap에 걸려 잘린 생성 PA는 cutoff 보상으로 COMPLETE 처리된다. 러너 COMMANDS에 V2/V3 subcommand가 없어 S5를 등록된 경로로 실행할 수 없다.
- 선택지: 합격: 시작별 |DR−truth| ≤ k·√(SE_DR²+SE_MC²)와 다중 보정 / 합친 Δ gap의 CI가 0을 포함하고 |gap| ≤ 등록 허용치(WE 단위) / 기술 보고만 | V3 법칙: D89 빈도 법칙만 / 빈도 법칙 + tempered BC π_b^α(α 격자) / 리그 혼합 | 잘린 생성 PA: COMPLETE(cutoff) / 별도 상태로 경계에 포함
- 채택: 주 합격 기준은 '합친 Δ gap의 CI가 0을 포함하고 |gap| ≤ 등록 허용치'로 두고, 시작별 k·SE는 진단으로 씁니다. V3는 빈도 법칙 + tempered BC 한 계열(α 격자 등록)로 두고 편향 곡선만 보고합니다(합격 판정 없음). 잘린 PA의 개수를 따로 보고합니다. V2 subcommand와 그 등록 필드를 S5 전에 추가합니다.

**M-7. DR의 q̂ 출처: 계획 Q̂ 재사용 vs 독립 평가 MC, 후보 q̂^cand의 제어변량 여부**

- 필요 이유: D89 §5는 비용이 막힐 때만 q̂^cand := Q̂^ref를 '선언된 제어변량'으로 허용한다. 이 선택은 D-목록에 없다. `build_runtime.candidate`는 π_cand를 만든 같은 MC Q̂를 DR의 q̂로 다시 쓴다(잡음으로 높게 나온 행동을 고르는 winner's curse가 DM 항에 섞임). P8은 planning seed 701과 evaluation seed 1701을 분리했다. 선택에 따라 추정량 identity와 S3 profile로 잴 경로가 달라진다. cap 꼬리의 최악 경계를 q̂에 반영할지도 정해지지 않았다.
- 선택지: (a) 현행: 계획 Q̂ 재사용 + 후보 제어변량 | (b) 독립 evaluation seed로 Q̂ 재추정(비용 약 2배) | (c) 후보 continuation MC Q̂^cand(비용 J·R·Ce·(1+A·S·Cs))
- 채택: S3 전에 결정합니다. 주 추정량은 (b)로 하고, (b)가 등록 공식상 상한을 넘을 때만 (a)로 내립니다. 이 전환 규칙도 비용만 보고 판단하도록 미리 적습니다. (c)는 ≤2025 주 분석에서 뺍니다.

**M-8. 구조 결함 PA(순서 오류·불법 카운트·투수/타자면 결측)의 분모·경계 처리**

- 필요 이유: `pa_requests`는 'ordering'/'illegal_count'/'missing_identity'인 PA를 제출하지 않는다. `run_dr`는 제출된 PA만 `estimate`에 넘기므로 이 PA들은 `population_delta_bounds`의 n에서 빠지고 개수로만 보고된다. R1/D-2(모든 PA 분모)·D-5([0,1] 경계)와 어긋난다. D-5의 범주(종료 없음·잘림·첫 행 0-0 아님·거절)에도 없다.
- 선택지: 별도 상태 `UNSUBMITTABLE_<reason>`로 분모에 남기고 [−1,1] Δ 경계에 포함 | 제외하고 개수만 보고 | FAILED_INTEGRITY로 중단
- 채택: 분모에 남기고 경계에 포함하기를 권합니다. S0 census에서 이유별 수를 먼저 세고, 코드를 고칠 때도 분모 의미는 바꾸지 않습니다.

**M-9. R8 이력 불일치(카운트 경로 단절)와 'unknown' 결과의 처리**

- 필요 이유: R8의 제안 상태 `UNSUPPORTED_INCONSISTENT_HISTORY`는 a704983에서 거절+sticky로 이미 구현됐다. 그런데 D-1~D-11 어디에도 이 결정이 없다. 거절로 둘지 FAILED_INTEGRITY로 둘지에 따라 분모가 달라지고, DEV 전체 실행이 결함 하나로 멈출 수도 있다. 이력 결과가 'unknown'이면 `next_count`가 검사를 건너뛴다. S0 `census`는 카운트 경로 단절 수를 세지 않는다.
- 선택지: 거절+sticky(현행) | FAILED_INTEGRITY 중단 | 무시하고 계속 평가
- 채택: 거절+sticky를 권합니다. 먼저 S0 census에 카운트 경로 단절 수와 'unknown' 결과 수를 추가해 라벨 없이 셉니다.

**M-10. stage별 합격·진행 규칙(go/no-go)과 M1 완료 판정**

- 필요 이유: 합격 기준이 있는 stage는 S2(atol)뿐이다. S0 결과 중 무엇이 계획을 멈추는지 정해지지 않았다. 어휘 밖 코드는 R3(b)로 멈추지만, D-8='refuse'에서 두 손 투수가 1명이라도 있으면 S1이 실패하고, INCOMPLETE 비율이 높을 때 어떻게 할지도 없다. S3 비용이 어느 수준이면 표본을 줄이거나 멈출지, V2가 실패하면 S6를 막을지도 없다. D89 V4의 cand=ref 짝 Δ=0 확인은 러너에 구현돼 있지 않고, 'V̂(ref) 대 관측 평균 PA WE' 기술 비교는 어느 PA에서 계산할지 정의되지 않았다. 보고서 §5의 'M1 완료 판정'에는 기준이 없다.
- 선택지: S0 전에 stage별 게이트 표 등록 / 사후 판단 | V2 합격을 S6(결과 라벨 열람) 선행 조건으로 / 독립 실행
- 채택: S0 전에 게이트 표를 등록합니다. 여기에는 stage별 중단 조건, 다음 stage를 여는 조건, 재등록 사유를 적습니다. V2 합격을 S6의 선행 조건으로 둡니다. S6에는 cand=ref 짝 실행(Δ 정확히 0)과 같은 평가 가능 PA에서의 V̂(ref)·관측 평균 WE 기술 비교를 넣습니다. M1 완료는 모든 게이트 통과와 독립 검토를 함께 요구합니다.

**M-11. seed 체계·정책 반복과 수치 결정성**

- 필요 이유: CLAUDE.md는 탐색 {0,1,2}, 채택 {0..4}의 시드를 요구한다. planning seed는 후보 정책 identity의 일부다(MC 잡음이 π_cand를 바꿈). 그런데 bootstrap·V2 생성·V2 truth·선택 salt seed는 등록돼 있지 않다. 코드에서 확인한 문제가 두 가지 있다. `run_world`는 생성에 `default_rng(seed)`, truth에 `seed+i`를 쓰므로 시작 0의 truth와 생성 로그가 같은 난수열을 공유한다. S3 profile 시작은 `canonical_hash([planning_seed, pa_id])`로 골라져 정책 seed가 바뀌면 표본도 바뀐다. G0 member seed 5개는 정책 반복이 아니다(D87). MPS 연산의 비트 단위 결정성과 재실행 허용오차도 정해지지 않았다.
- 선택지: planning seed 1개(주) / {0,1,2} 반복 | seed 파생: 역할별 SeedSequence spawn key / 임의 정수
- 채택: 역할별 seed 표(planning, evaluation q̂, bootstrap, V2 생성, V2 truth, 선택 salt)를 SeedSequence spawn으로 서로 겹치지 않게 등록합니다. V2는 planning seed {0,1,2}로 반복합니다. V4는 주 seed 1개로 하고, 나머지는 비용이 허락할 때 민감도로 돌립니다. 재실행 허용오차는 S2 기준(1e-6)을 준용합니다.

**M-12. 주 명세와 민감도 목록, 다중성(≤2025)**

- 필요 이유: 갈림길이 많다: BC-P/BC-E(D-1), π̂_b 빈도 법칙(D89 §5), clip {10,20,50}·IS/SNIS/DM-only(D87 §5 보조), D-7 스냅샷/rolling, D-8 refuse/intersect, planning seed, V3 법칙. 추정기 코드는 DR만 계산한다. 주 명세와 민감도 목록을 결과 열람 전에 정하지 않으면 DEV를 본 뒤 고르는 선택이 된다. D87은 'ARM별 family, family당 주 대비 1개, 부분 분석은 Holm'을 제안했다.
- 선택지: (a) 주 명세 1개 + 고정 민감도 목록(기술 보고, Holm 없음) | (b) 민감도 전체를 Holm family로 | (c) 민감도 없음
- 채택: (a)를 권합니다. 주 명세는 BC-P, 스냅샷, clip 없는 DR, 주 seed입니다. ≤2025 결과에는 확인적 주장이 없다고 명시하고, p값을 쓴다면 한 family 안에서 Holm을 적용합니다. 보조 추정기는 진단 전용으로 이름을 붙입니다.

**M-13. 부분군·역할·경기 단위 정의**

- 필요 이유: D87 진단은 역할·월·투수 볼륨별 ESS를 요구한다. SP/RP 정의는 두 가지가 있다: D87은 경기별 팀 첫 투수를 SP로, P8 `select_pa_requests`는 TRAIN panel의 `train_role`을 쓴다. 볼륨 구간, 오프너·야수 등판 플래그도 정해지지 않았다. 부트스트랩 군집은 game_pk이다. `assign_fold`가 행 날짜로 split을 나누므로 중단 뒤 재개된 경기가 6/7월 경계를 넘을 수 있다.
- 선택지: 역할: 경기 첫 투수(D87, 투구 전 계산 가능) / TRAIN 역할(P8) | 볼륨: TRAIN 투구 수 분위 구간 등록 / 없음 | 경계를 넘는 경기: PA별 행 날짜로 소속, 플래그
- 채택: 역할은 D87 정의를 쓰고, 볼륨은 TRAIN 분위로 등록한 구간을 씁니다. 월은 game_date 기준입니다. 경기 단위는 game_pk로 하고 경계를 넘는 경기에는 플래그를 붙입니다. 부분군은 기술 보고만 합니다.

## 함께 정해야 했던 결정 쌍

- D-1 ↔ D-10: BC-P는 plate 좌표·supported_pa 조건을 빼므로, eligible D100 행으로만 적합한 23,728 pool에 없는 행동(예: PO·EP·UN·SC 같은 코드)에 로깅 질량이 더 생길 수 있다(미측정). 그만큼 V2에서 세계 delivery가 필요한 M 밖 로그가 늘어난다. D-1은 BC 지원도 바꿔 M·ρ^ref·π̂_b(M|H) 분포까지 바꾼다.
- D-1 ↔ D-3: BC-P는 automatic_ball/strike·구종 결측 행을 적합에서 빼고, D-3은 그 행을 거절(NO_PITCH)한다. 둘을 함께 채택해야 π̂_b가 '투구가 있을 때의 구종 법칙'으로 일관된다.
- D-8 ↔ D92 요청별 지원 검사: 'intersect'로 만든 표 마스크는 `BoundComponents.check_support`가 비교하는 손별 `inputs.support(state)`와 다를 수 있다. 그러면 첫 해당 요청에서 FAILED_INTEGRITY로 실행 전체가 멈춘다. 'refuse'는 충돌 투수가 하나라도 있으면 S1 생성을 거부한다. 따라서 D-8은 S0 census 결과와 검사 규칙 수정을 함께 정해야 한다.
- D-2 ↔ D-5 ↔ M-8: '모든 PA 분모'와 [0,1] 경계는 구조 결함으로 제출되지 않은 PA까지 포함해야 성립한다. 현재 `run_dr`는 그런 PA를 경계 n에서 뺀다.
- D-5 ↔ D-6: 보상을 정의할 수 없는 사유(invalid_next_state, incomplete_game, final_result_undetermined 등)가 모두 INCOMPLETE_NO_TERMINAL 경계로 들어간다. 보상 규칙은 불완전 PA 규칙과 함께 정해야 한다.
- D-4 ↔ D-5 ↔ M-9: 알려진 투수로 교체된 뒤 계속 평가할지(D-4), 거절·sticky 상태의 범위(D-3·R8)가 평가 가능 PA 집합을 함께 정한다. D-4 제목에 타자 교체도 명시해야 한다.
- D-7 ↔ S2 probe ↔ M-6 V2 시작: 스냅샷 as-of는 DEV에서 '2025-07-01'(엄격히 이전)로 적어야 한다(config의 `profile_as_of.dev`가 없음). S2는 rolling 예외이고, V2 시작 맥락도 같은 규칙을 따라야 한다.
- D-9 ↔ D-11 ↔ M-1 ↔ M-7: τ·samples·pitch_cap·q̂ 출처가 결정당 비용을 정하고, 상한이 V4·V2에서 가능한 표본 크기를 정한다. June 2025에서 τ를 다시 고르기로 하면 S0–S6에 없는 튜닝 stage(표본·선정 기준·상한)를 추가해야 하고, 그 stage는 S5·S6보다 먼저 와야 한다.
- D-9 ↔ M-11: planning seed는 식별자 `search`의 일부다. seed 반복 여부는 곧 정책 identity를 몇 개 둘지의 결정이다.
- D-10 ↔ M-6: 'refuse'면 로그가 거부된 시작이 빠져 선택이 생기고, 'fallback'이면 세계 정의가 바뀐다. 합격 기준은 채택한 규칙에 맞춰 정해야 한다.
- M-5 ↔ M-6 ↔ D87 BLK-08: ESS→문턱 도출 규칙과 V2 합격 기준(허용 편향·포함률)을 같은 등록에서 함께 고정해야 한다.
- M-4 ↔ D-5 ↔ M-13: 재표집 단위(경기)와 경계 CI 여부는 불완전 PA 처리와 경기 단위 정의에 의존한다.
- M-2 ↔ D-1·D-8 ↔ R3(b): S1 산출물(BC·지원 표)과 S2 식별자 SHA를 addendum으로 pin하는 절차가 있어야 S2 이후가 실행된다. S0에서 어휘 밖 코드가 나오면 재등록 경로도 같은 절차로 처리한다.
- M-10 ↔ M-3: S5·S6를 여는 조건은 독립 검토 완료 여부와 함께 정해야 한다.
- M-12 ↔ D-1·D-7·D-8: BC-E, rolling prior, intersect를 민감도로 쓸지 여부는 각 결정의 '비교안'을 정하는 일과 같다.

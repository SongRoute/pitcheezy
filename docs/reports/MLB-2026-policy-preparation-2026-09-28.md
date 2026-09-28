# MLB 2026 정책 정의 준비 — 2026-09-28

COOP-015(D88). 작성 Opus5.5, 지원 Sol, 독립 과학 검토 Astra 1회 지적 반영; 최종 통합 확인 Root. 기준 `e5bfa7f`. 계약 [MLB-2026-POLICY-PREPARATION-v1](../contracts/MLB-2026-POLICY-PREPARATION-v1.md), 설정 [config](../../configs/MLB-2026-POLICY-PREPARATION-v1.json).

**통합 확인(D89):** Root가 검토 기준 충족과 통합 후 신규22검사 통과를 확인했다. Graphify 로컬 AST 갱신도 완료했으며 문서/PDF 의미 분석은 수행하지 않았다. 작성·수정·검토 이력은 [검토 기록](../reviews/COOP-015-policy-preparation-2026-09-28.md)을 따른다. 아래 작업 범위의 미수행 목록은 작성자 단계 기록이며 인수인계·PR 반영은 Root가 맡는다.

이 문서의 `COORD`는 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260928-2026-policy-preparation`이다.

## 1. 요약

- **M1 부분 준비다.** 후보·기준 정책이 지켜야 할 호출 계약(확률·마스크·어휘 해시·실패/거절 처리·결정성)과 ARM-B/ARM-A 식을 제안했고, 하나의 열거된 toy PA에서 식과 일부 실패 처리가 맞는 것을 확인했다(일반 증명 아님). 계약 전체 중 **구현·검사된 부분은 합성 스크립트의 일부 함수뿐**이고, 모르는 투수 거절·분모 원장·PA 중 투수 교체 처리는 제안만 있다. **정책은 동결되지 않았다**: 실제 BC artifact, Q 시뮬레이터(예측기·delivery pool·WE), τ, 지원 표, π̂_b의 identity와 해시가 모두 `null`이다.
- ARM-A는 식을 적을 수 있지만 기준의 목표 위치 법칙 κ와 로그 법칙이 없어 `BLOCKED_NOT_IDENTIFIED` 그대로다. 야구로 말하면, 포수 사인을 기록한 적이 없으니 "평소 사인대로 던진 투수"라는 기준 투수를 만들 수 없다.
- ARM-B는 기존 코드(`CategoricalBC` + P3 `kl_policy`)로 바로 호출 가능한 형태가 있다. 다만 `enabled=false` 그대로다. 2026 대비를 인과로 읽으려면 일관성, 순차 교환 가능성, positivity, 그리고 π̂_b가 2026 참 로그 법칙이거나 정책별 q̂^π가 2026의 참 Q라는 조건 중 하나가 **모두** 필요하다(계약 §5). 어느 것도 입증되지 않았다.
- **이번 수정의 핵심(Root 설계 지적 수용):** 로그 nuisance π̂_b는 정책 마스크 M으로 제한하지 않은 TRAIN BC(전체 TRAIN 어휘)다. 정책은 delivery 가능한 지원으로 재정규화할 수 있지만, 그렇다고 투수의 로그 법칙이 조건부 법칙으로 바뀌지 않는다. 그래서 ρ^ref는 일반적으로 1이 아니며, π̂_b(M|H)=1일 때만 1이다. 야구로 말하면, 우리가 추천할 수 있는 구종을 4개로 줄여도 투수가 실제로 5번째 구종을 던진 기록은 사라지지 않는다.
- 레거시 `ips.pitch_ratios`는 π_b가 없으면 ρ=1, 0이면 1e-12로 바닥을 깐다. D87의 fail-closed와 충돌하므로 새 평가에 재사용하지 않는다. 반면 `dr.traj_dr_terms_rows`의 plain DR은 유효 입력에서 D87 식과 경로별로 같은 값을 낸다.

## 2. 세 층 구분: 제안 식 / 구현 가능한 모의 / 실제 identity

| 구성요소 | 제안 식(계약) | 지금 돌아가는 모의·기존 코드 | 실제 artifact identity |
|---|---|---|---|
| ARM-B 기준 | 공통 지원 제한 TRAIN BC | `CategoricalBC`, `SupportedBC`, `fit_bc`(TRAIN 2023-05-15..2025-04-30 가드) | `null`. EXP-P8-001은 run 안에서 fit했고 독립 BC artifact·해시는 미확인 |
| ARM-B 후보 | P3 `∝ π_ref·exp(Q̂^ref/τ)` on M | `kl_policy`, `RolloutImprovement.policy("P3")` | `null`. τ=.003은 EXP-P8-001(G2 control 세계, June 2025 선택, D65) 값이며 예측기가 바뀌면 새 identity |
| Q̂^ref 예측기 | 동결 조건부 10-class 예측 | `FrozenGEnsemble`(3 member 고정) | `null`. 동결 G0는 5 member → 어댑터 필요. G0는 정책 아님 |
| WE continuation | `defense-we-pa-v1` | `game.terminal_values`, `FrozenWE` | `game_values.pkl` SHA `aa6c4e48…`(`model-v1.json`)는 레거시 참조, 미선정·bytes 미재검증 |
| π̂_b | M으로 제한하지 않은 전체 어휘 TRAIN BC(기준과 별도 identity) + 빈도 법칙 민감도 | `CategoricalBC.probabilities`(`SupportedBC` 아님) | `null` |
| ARM-A 기준/로그 | BC·κ, π_b(g,z) | Observer 커널 기준(`observer-repertoire-kernel-v1`) = 도달 위치 커널 | 존재하지 않음 |
| 2026 코드 대응 | 정렬 어휘 + 해시, 미지 코드는 integrity 실패 | 합성 `action_index` | `null` |
| 모르는 투수 거절·분모 원장 | 거절, 분모 유지 | **미구현.** 테스트는 `CategoricalBC.fallback` 표시만 확인 | 해당 없음 |

Sol 재사용 목록(`COORD/sol-inventory.md`, 스냅샷 `301ac07`)과 교차 확인했고 결론이 같다.

## 3. 실제 실행한 합성 검사

인터프리터 `/Users/song/Projects/pitcheezy/.venv/bin/python`(3.12.14), 작업 디렉터리 worktree, `PYTHONPATH=src:experiments/pitchmdp`. 자료·모델·캐시 입력 없음. 최신 원본 출력은 `COORD/opus-validation-002.json`, 첫 검증 기록 `COORD/opus-validation.json`은 그대로 보존한다(20/53 기준, 이번 수정 전).

1. `scripts/check_2026_policy_contract.py` → exit 0, 9개 gate 모두 true. toy 경로 276개, 경로 확률 합 1(부동소수 오차 1e-16).
   - 참값 V(cand)=0.549958241, V(ref)=0.5482305296, Δ=+0.0017277114(+0.17277114%p, 공격 관점 −). toy 수치이며 야구 결과가 아니다.
   - 참 π_b + 틀린 q̂: 오차 cand 0.0, ref 4.4e-16. 틀린 π̂_b + 참 q̂: cand 2.2e-16, ref 0.0. 둘 다 틀림: cand +0.16792, ref +0.06259(편향 검출).
   - clip 1.5: 참 q̂면 오차 ≤1.1e-16, 틀린 q̂면 cand −0.0088015(ref는 ratio가 1.5를 넘지 않아 clip 비활성, 4.4e-16).
   - **마스크 제한 기준(신규):** 전체 지원 로그 법칙을 CH가 빠진 정책 마스크로 제한한 기준 V=0.5484506941. ρ^ref 범위 0~1.4286(1이 아님). 전체 법칙 π̂_b: 오차 2.2e-16(틀린 q̂)/1.1e-16(q̂=0). 같은 법칙을 마스크로 재정규화한 π̂_b: CH 로그 경로 제외(유지 질량 0.61211) 시 조건부 오차 −0.0028564(틀린 q̂)/−0.0028781(q̂=0), 제외 경로를 0으로 셀 때 −0.2144870/−0.2145003.
2. `pytest -q tests/test_2026_policy_contract.py` → 22 passed in 0.85s. 짝 기댓값 = 참 대비, 같은 법칙의 독립 callable 짝 Δ=0(어긋난 짝은 검출), 레거시 plain DR 경로별 일치(atol 1e-12), 레거시 비율 도우미의 비 fail-closed 동작 고정, `policy_row` 실패 7종, DR 입력 거부 6종, 로그 행동 세 상태 구분, 마스크 제한 기준의 ρ^ref≠1·재정규화 편향·마스크가 전체 질량을 덮을 때만 ρ^ref≡1, WE 부호·단위·범위, `CategoricalBC`/`kl_policy` 행 통과·모르는 투수 **fallback 표시**·비 TRAIN 거부.
3. 기존 재사용 검사 포함 묶음: `pytest -q experiments/pitchmdp/tests/test_rollout_policy.py experiments/pitchmdp/tests/test_game_planner.py experiments/pitchmdp/tests/test_matrix_policy.py tests/test_ope_dr.py tests/test_2026_policy_contract.py` → 55 passed, 8 subtests passed in 1.75s. 기존 33+8은 Sol 별도 실행에서도 PASS(`COORD/sol-existing-synthetic-checks.md`). 이전 기록의 1.47초/1.53초/1.60초는 서로 다른 실행이며 최신 값만 위에 적는다.

**이 toy에서 확인된 것:** 열거된 한 toy PA에서 두 방향의 DR 일치, 재정규화 편향, 일부 실패/거절이 조용히 고쳐지지 않음, 기존 plain DR과의 수치 일치. 일반 증명이 아니다. **확인되지 않은 것:** 2026 구종 배정의 교환 가능성, 연도 간 이동, 실제 overlap·ESS, 실제 정책 identity, 모르는 투수 거절·분모 원장 구현.

## 4. ≤2025 후속 검증 계획 (실행 승인 아님, 자료 미열람)

이 계획이 제안하는 구간은 이미 노출된 기존 집합이다: TRAIN 2023-05-15..2025-04-30(적합), June 2025(EXP-P8-001 τ 선택, D86 보정), 2025년 7–9월 DEV(G0 평가, P8 DEV). 이 목록 밖의 ≤2025 모집단은 보유·사용 여부 모두 모름이며, 넓게 "노출됐다"거나 "미사용"이라고 주장하지 않는다. 인증된 미노출 모집단 수는 0이다. 아래는 모두 `exposed_development` 라벨의 **방법 검증**이며 확인 증거가 아니다.

| ID | 무엇 | 자료·노출 | 예산 |
|---|---|---|---|
| V1 | toy 정확 oracle(완료) | 없음 | 실측 <1초 |
| V2 | **모델 세계 준합성**: 동결 시뮬레이터 안에서 알려진 전체 어휘 π_b(=M으로 제한하지 않은 BC)로 로그를 생성하고 ARM-B DR Δ를 같은 세계의 rollout Δ와 비교. ESS 분포를 측정해 BLK-08 게이트 근거 마련 | DEV PA 시작 맥락만(결과 라벨 불필요) | `null`, profile 먼저 |
| V3 | V2에서 로그 법칙을 π̂_b와 다르게(빈도 법칙·섭동) 두고 편향 크기 측정 → 이동 가정 민감도 척도 | 같음 | `null` |
| V4 | 실제 2025 DEV 검사: cand=ref 짝 Δ̂=0은 대수·무결성만 확인. V̂(ref) 대 관측 로그 정책 평균 PA WE의 차이는 실제 기준–로그 정책 차이와 nuisance·이동·모형 오차가 섞인 기술적 차이이며, 이동·보정 오차를 단독 식별하지 못함 | DEV 결과 라벨, 이미 노출 | `null` |
| V5 | DEV 지원·거절 분모(모르는 투수, 빈 지원, 미지 코드, 마스크 밖 로그 행동 비율, π̂_b(M\|H) 분포) 기술 통계 | DEV 투구 전 필드·로그 구종 | `null` |

**V2 등록·실행 전제:** 시뮬레이터는 정책 마스크 M 밖을 포함하여 전체 어휘 로그 법칙이 양의 확률을 주는 **모든 행동**의 delivery·전이를 정의해야 한다. 하나라도 없으면 V2는 위 정의대로 실행할 수 없으며, 로그 법칙을 M으로 조용히 재정규화해 해결하지 않는다. 모의 ESS는 실제 2026 overlap·이동·교란 통제나 BLK-08 문턱을 입증하지 않는다.

**V4 해석 제한:** 기준 정책과 실제 로그 정책이 같다는 사실을 독립적으로 확정하지 않는 한 위 차이를 순수 TRAIN→DEV 이동·보정 오차로 해석하지 않는다. cand=ref 짝 0은 추정량의 대수·무결성 확인일 뿐 타당성 증거가 아니다.

비용 참고값: EXP-P8-001 P0–P3 큐 293.31초(128+128 시작, D65). 후보 π_cand를 매 결정에서 MC로 계산하므로 결정 수 N에 대해 약 `N·A·S·Cs` 조건부 행이 든다. N·A·S·Cs 실제 값은 미측정. 새 작업의 예산 근거로 쓰지 않는다.

## 5. 다음 단계 (의존 순서)

1. Astra 독립 과학 검토 1회(`COORD/astra-focused-review.json`, 검토 커밋 `287c884b`)의 문구 수정 기준을 Sol이 기계적으로 확인한 뒤 Root가 최종 통합 확인. Astra가 수정 후 bytes를 다시 검토했다고 주장하지 않는다.
2. Song 결정: π̂_b를 M과 독립인 전체 어휘 법칙으로 두는 이번 제안의 채택, Q̂ 예측기(G0 5 member vs P8의 G2/G3), π̂_b 주·민감도, τ 튜닝 구간(노출된 June 2025 재사용 여부), PA 중 투수 교체 규칙. ARM-B 활성화는 별개 결정.
3. additive 구현 + 합성 검사: G0 5 member 어댑터, BC 직렬화·어휘/지원 해시, 2026 어댑터(모르는 투수 거절과 분모 원장, strict ratio, 로그 행동 세 상태). 레거시 함수는 수정하지 않는다.
4. V2–V5 설정 등록(예산·seed·출력 경로), profile 먼저, 단일 heavy lock.
5. V2/V3 실행 → ESS 게이트·민감도 척도를 근거로 D87 BLK-08 제안값 갱신.
6. 정책·nuisance identity 동결과 독립 검토 → 그때 M1 완료 여부 판정 → D87 M2.

BLK-05(정확한 과거 사용 경기 ID)와 이후 2026 수집 범위는 이 작업과 무관하게 모름으로 남는다.

## 6. 하지 않은 것·제약

2026 또는 ≤2025 원자료·parquet 헤더·npz/npy/pickle·모델 가중치 열람, payload 해시, 실데이터 기반 fit·추론·OPE·수집, 기존 코드·설정·계약 변경, decisions/handoff 수정, push/merge 없음. 새 파일 5개(계약·설정·보고서·합성 스크립트·테스트)만 추가했다. 새 스크립트는 인자·파일 입력이 없고 2026 평가를 호출하지 않는다. 작성 worktree에는 graph가 없어, 통합 후 Root가 `graphify update .`를 완료했다.

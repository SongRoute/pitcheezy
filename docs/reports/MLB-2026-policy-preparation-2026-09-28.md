# MLB 2026 정책 정의 준비 — 2026-09-28

COOP-015(D88). 작성 Opus5.5, 지원 Sol, 최종 검토 Astra(예정). 기준 `e5bfa7f`. 계약 [MLB-2026-POLICY-PREPARATION-v1](../contracts/MLB-2026-POLICY-PREPARATION-v1.md), 설정 [config](../../configs/MLB-2026-POLICY-PREPARATION-v1.json).

## 1. 요약

- **M1 부분 준비다.** 후보·기준 정책이 지켜야 할 호출 계약(확률·마스크·어휘 해시·실패/거절 처리·결정성)과 ARM-B/ARM-A 식을 제안했고, 합성 검사로 식과 실패 처리를 확인했다. **정책은 동결되지 않았다**: 실제 BC artifact, Q 시뮬레이터(예측기·delivery pool·WE), τ, 지원 표, π̂_b의 identity와 해시가 모두 `null`이다.
- ARM-A는 식을 적을 수 있지만 기준의 목표 위치 법칙 κ와 로그 법칙이 없어 `BLOCKED_NOT_IDENTIFIED` 그대로다. 야구로 말하면, 포수 사인을 기록한 적이 없으니 "평소 사인대로 던진 투수"라는 기준 투수를 만들 수 없다.
- ARM-B는 기존 코드(`CategoricalBC` + P3 `kl_policy`)로 바로 호출 가능한 형태가 있다. 다만 `enabled=false` 그대로이고, 2026 대비의 타당성은 "≤2025로 만든 π̂_b가 2026 투수들의 실제 구종 선택 법칙과 같다"는 이동 가정에 전부 걸려 있다. 합성 검사는 이 가정을 검증하지 못한다.
- 레거시 `ips.pitch_ratios`는 π_b가 없으면 ρ=1, 0이면 1e-12로 바닥을 깐다. D87의 fail-closed와 충돌하므로 새 평가에 재사용하지 않는다. 반면 `dr.traj_dr_terms_rows`의 plain DR은 유효 입력에서 D87 식과 경로별로 같은 값을 낸다.

## 2. 세 층 구분: 제안 식 / 구현 가능한 모의 / 실제 identity

| 구성요소 | 제안 식(계약) | 지금 돌아가는 모의·기존 코드 | 실제 artifact identity |
|---|---|---|---|
| ARM-B 기준 | 공통 지원 제한 TRAIN BC | `CategoricalBC`, `SupportedBC`, `fit_bc`(TRAIN 2023-05-15..2025-04-30 가드) | `null`. EXP-P8-001은 run 안에서 fit했고 독립 BC artifact·해시는 미확인 |
| ARM-B 후보 | P3 `∝ BC·exp(Q̂^BC/τ)` | `kl_policy`, `RolloutImprovement.policy("P3")` | `null`. τ=.003은 EXP-P8-001(G2 control 세계, June 2025 선택, D65) 값이며 예측기가 바뀌면 새 identity |
| Q̂^BC 예측기 | 동결 조건부 10-class 예측 | `FrozenGEnsemble`(3 member 고정) | `null`. 동결 G0는 5 member → 어댑터 필요. G0는 정책 아님 |
| WE continuation | `defense-we-pa-v1` | `game.terminal_values`, `FrozenWE` | `game_values.pkl` SHA `aa6c4e48…`(`model-v1.json`)는 레거시 참조, 미선정·bytes 미재검증 |
| π̂_b | TRAIN BC(별도 nuisance 이름) + 빈도 법칙 민감도 | 같은 코드 | `null` |
| ARM-A 기준/로그 | BC·κ, π_b(g,z) | Observer 커널 기준(`observer-repertoire-kernel-v1`) = 도달 위치 커널 | 존재하지 않음 |
| 2026 코드 대응 | 정렬 어휘 + 해시, 미지 코드는 integrity 실패 | 합성 `action_index` | `null` |

Sol 재사용 목록(`COORD/sol-inventory.md`, 스냅샷 `301ac07`)과 교차 확인했고 결론이 같다.

## 3. 실제 실행한 합성 검사

인터프리터 `/Users/song/Projects/pitcheezy/.venv/bin/python`, 작업 디렉터리 worktree, `PYTHONPATH=src:experiments/pitchmdp`. 자료·모델·캐시 입력 없음. 원본 출력은 `COORD/opus-validation.json`.

1. `scripts/check_2026_policy_contract.py` → exit 0, 6개 gate 모두 true. toy 경로 276개, 경로 확률 합 1(부동소수 오차 1e-16).
   - 참값 V(cand)=0.549958241, V(ref)=0.5482305296, Δ=+0.0017277114(+0.17277114%p, 공격 관점 −). toy 수치이며 야구 결과가 아니다.
   - 참 π_b + 틀린 q̂: 오차 cand 0.0, ref 4.4e-16. 틀린 π̂_b + 참 q̂: cand 2.2e-16, ref 0.0. 둘 다 틀림: cand +0.16792, ref +0.06259(편향 검출).
   - clip 1.5: 참 q̂면 오차 ≤1.1e-16, 틀린 q̂면 cand −0.0088015(ref는 ratio가 1.5를 넘지 않아 clip 비활성, 4.4e-16).
2. `pytest -q tests/test_2026_policy_contract.py` → 14 passed. 짝 기댓값 = 참 대비, 레거시 plain DR 경로별 일치(atol 1e-12), 레거시 비율 도우미의 비 fail-closed 동작 고정, 정책 행 실패 7종, WE 부호·단위·범위, `CategoricalBC`/`kl_policy` 행 통과·모르는 투수 fallback 표시·비 TRAIN 거부.
3. 기존 재사용 검사 포함 묶음: `pytest -q experiments/pitchmdp/tests/test_rollout_policy.py experiments/pitchmdp/tests/test_game_planner.py experiments/pitchmdp/tests/test_matrix_policy.py tests/test_ope_dr.py tests/test_2026_policy_contract.py` → 47 passed, 8 subtests passed, 1.53초. 실행 전 네 기존 파일에서 자료 경로·parquet/np.load/pickle 패턴을 grep해 없음을 확인했다.

**증명된 것:** 식 구현이 정확 기댓값에서 이중 강건성을 만족한다. 실패/거절이 조용히 고쳐지지 않는다. 기존 plain DR이 같은 식이다. **증명되지 않은 것:** 2026 구종 배정의 교환 가능성, 연도 간 이동, 실제 overlap·ESS, 실제 정책 identity.

## 4. ≤2025 후속 검증 계획 (실행 승인 아님, 자료 미열람)

모든 ≤2025 구간은 이미 노출됐다: TRAIN(적합), June 2025(EXP-P8-001 τ 선택, D86 보정), 2025년 7–9월 DEV(G0 평가, P8 DEV). 인증된 미노출 ≤2025 모집단은 0개이고, 2023-05-15 이전 자료의 보유·가용성은 모름. 따라서 아래는 모두 `exposed_development` 라벨의 **방법 검증**이며 확인 증거가 아니다.

| ID | 무엇 | 자료·노출 | 예산 |
|---|---|---|---|
| V1 | toy 정확 oracle(완료) | 없음 | 실측 <1초 |
| V2 | **모델 세계 준합성**: 동결 시뮬레이터 안에서 알려진 π_b(=BC)로 로그를 생성하고 ARM-B DR Δ를 같은 세계의 rollout Δ와 비교. ESS 분포를 측정해 BLK-08 게이트 근거 마련 | DEV PA 시작 맥락만(결과 라벨 불필요) | `null`, profile 먼저 |
| V3 | V2에서 로그 법칙을 π̂_b와 다르게(빈도 법칙·섭동) 두고 편향 크기 측정 → 이동 가정 민감도 척도 | 같음 | `null` |
| V4 | 실제 2025 DEV placebo: cand=ref면 Δ̂가 정확히 0인지(무결성), V̂(ref) 대 관측 평균 PA WE 차이(TRAIN→DEV 이동 진단, 식별 아님) | DEV 결과 라벨, 이미 노출 | `null` |
| V5 | DEV 지원·거절 분모(모르는 투수, 빈 지원, 미지 코드) 기술 통계 | DEV 투구 전 필드 | `null` |

비용 참고값: EXP-P8-001 P0–P3 큐 293.31초(128+128 시작, D65). 후보 π_cand를 매 결정에서 MC로 계산하므로 결정 수 N에 대해 약 `N·A·S·Cs` 조건부 행이 든다. N·A·S·Cs 실제 값은 미측정. 새 작업의 예산 근거로 쓰지 않는다.

## 5. 다음 단계 (의존 순서)

1. Root 통합 검토, Astra 최종 과학 검토(`COORD/opus-review-brief.md`).
2. Song 결정: Q̂ 예측기(G0 5 member vs P8의 G2/G3), π̂_b 주·민감도, τ 튜닝 구간(노출된 June 2025 재사용 여부), PA 중 투수 교체 규칙. ARM-B 활성화는 별개 결정.
3. additive 구현 + 합성 검사: G0 5 member 어댑터, BC 직렬화·어휘/지원 해시, 2026 어댑터(모르는 투수 거절, strict ratio). 레거시 함수는 수정하지 않는다.
4. V2–V5 설정 등록(예산·seed·출력 경로), profile 먼저, 단일 heavy lock.
5. V2/V3 실행 → ESS 게이트·민감도 척도를 근거로 D87 BLK-08 제안값 갱신.
6. 정책·nuisance identity 동결과 독립 검토 → 그때 M1 완료 여부 판정 → D87 M2.

BLK-05(정확한 과거 사용 경기 ID)와 이후 2026 수집 범위는 이 작업과 무관하게 모름으로 남는다.

## 6. 하지 않은 것·제약

2026 또는 ≤2025 원자료·parquet 헤더·npz/npy/pickle·모델 가중치 열람, payload 해시, fit·추론·OPE·수집, 기존 코드·설정·계약 변경, decisions/handoff 수정, push/merge 없음. 새 파일 5개(계약·설정·보고서·합성 스크립트·테스트)만 추가했다. 새 스크립트는 인자·파일 입력이 없고 2026 평가를 호출하지 않는다. 코드 변경이 있으므로 Root에서 `graphify update .`가 필요하다(worktree에는 graph 없음).

# MLB 2026 정책 정의 준비 v1 — M1 준비, 미동결·미등록

2026-09-28, COOP-015(D88), 기준 `e5bfa7f`. 기계 필드는 [configs/MLB-2026-POLICY-PREPARATION-v1.json](../../configs/MLB-2026-POLICY-PREPARATION-v1.json), 근거·검사 결과·다음 단계는 [보고서](../reports/MLB-2026-policy-preparation-2026-09-28.md). 상위 초안은 [MLB-2026-OPE-PREREG-DRAFT-v1](MLB-2026-OPE-PREREG-DRAFT-v1.md)(D87)이며 그 상태·ARM 선택·blocker를 바꾸지 않는다.

> **상태: `PROPOSAL_UNREGISTERED`, `policy_frozen=false`, `execution.enabled=false`.** 이 문서는 호출 가능한 정책이 **지켜야 할 계약과 식을 제안**한다. 식을 적었다고 정책이 동결된 것이 아니다. 실제 정책 identity(fit된 BC, Q/시뮬레이터, WE, 지원 표, τ, 해시)는 아직 선택·등록되지 않았고 `null`이다. 합성 검사 통과는 식·실패 처리의 산술을 확인할 뿐 2026 자료에서의 식별·이동·overlap을 입증하지 않는다. ARM-A는 `BLOCKED_NOT_IDENTIFIED`, ARM-B는 `enabled=false`/`auto_opt_in=false` 그대로다. G0는 10-class 예측기이지 정책이 아니다. RE24 결과를 WE로 부르지 않는다.

## 1. 공통 정의

**의사결정 시점 t와 이력 `H_t`.** PA i의 t번째 투구 직전. `H_t` = 투구 전 경기 상태(inning, topbot, outs, bases, home/away score, balls, strikes, pitcher_id, batter_stand) + 같은 PA에서 **엄격히 이전** 투구들의 (구종, 물리 벡터, 결과, 투구 전 카운트) + 동결 프로필. 현재 공의 구종·위치·물리·결과, 이후 공, 최종 PA 길이는 입력 슬롯이 없다(C0, `rollout_policy.PAState`). 기존 `CategoricalBC`는 `H_t` 중 (pitcher, balls, strikes, batter_side, 직전 구종)만 쓴다. 더 많이 쓰는 정책은 그 충분성을 따로 적는다.

**프로필 as-of.** BC·지원·레퍼토리·delivery pool은 TRAIN `2023-05-15..2025-04-30`(기존 `matrix_policy.fit_bc` 가드)에서 동결한다. 타자 프로필은 C0 규칙(경기 날짜보다 엄격히 이전)이지만 **2026 평가에서는 2026 이전 스냅샷에 고정**한다(D87 §4: 2026으로 굴려 갱신하는 것은 통계 구성요소 추정). 2026 관측은 추론 맥락(카운트, 같은 PA 이전 투구)으로만 들어간다.

**평가 시작·horizon·가치.** D87 제안 그대로: 각 PA의 첫 투구(0-0) 하나, 가중치 1. 보상은 PA 종료 단계의 `W(S_PA_end)` 하나, C0 `defense-we-pa-v1`: **초기 상태의 수비 팀**(Top→홈, Bot→원정)의 동결 WE 확률 [0,1], 이닝이 바뀌어도 같은 팀(`game.terminal_values`). 효과 `Δ = V(π_cand) − V(π_ref)`, `delta_pp = 100·Δ`, 양수 = 후보가 초기 수비 팀에 유리. 공격 관점은 `−Δ`로 따로 표시한다. 모델 내부 rollout의 cap 꼬리는 초기 상태 동결 WE + [0,1] 최악 경계(P8 규약). 이것은 관측 최종 승리·경기 전체 개입·이닝 교체 가치가 아니다.

**행동 어휘와 대응.** 정렬된 고유 문자열 tuple `V`와 `vocabulary_sha256 = sha256("\n".join(V))`를 정책·nuisance와 함께 등록한다. 호출 결과는 `(probs[|V|], mask[|V|] bool)`. 로그 행동 라벨은 `V.index(label)`로만 대응한다.

## 2. 호출 계약 (모든 후보·기준·π̂_b 공통)

| 경우 | 처리 | 분모 |
|---|---|---|
| 어휘 순서/해시 불일치, shape·dtype 불일치, NaN·음수, 합≠1(atol 1e-9), 마스크 밖 양의 질량, 어휘에 없는 로그 구종 코드, WE 값이 [0,1] 밖 | `FAILED_INTEGRITY`, 즉시 중단. 재정규화·마스킹·대체 없음 | 실행 전체 실패로 보존 |
| 빈 지원(`mask` 전부 False), 2026 이전 TRAIN 이력 없는 투수(`CategoricalBC.fallback=True`), 로그 행동의 π̂_b = 0 | `UNSUPPORTED` 거절(정당). 관측 행동을 넣거나 ρ=1로 두지 않음 | 요청 분모에 남김. 전체 모집단 값은 `null` 또는 [0,1] 최악 경계 |
| 후보·기준의 로그 행동 질량 0 | 정당한 ρ=0 | 포함 |
| PA 중 투수 교체로 새 투수가 미지원 | 그 PA를 `UNSUPPORTED_MID_PA`로 표시, 이후 결정을 채우지 않음 | 남김. 처리 규칙 최종안 `null`(등록 전 확정) |

기존 league fallback(`CategoricalBC.support`가 모르는 투수에게 리그 지원을 줌)은 모델 내부 연구용이다. 2026 평가 어댑터는 `fallback(state)`가 참이면 **거절**한다. 식별된 fallback 법칙이 등록되기 전까지 신규 투수를 리그 BC로 채우지 않는다.

**결정성.** 같은 동결 identity와 같은 `H_t`는 항상 같은 확률 벡터를 낸다. MC Q를 쓰는 후보는 `planning_seed(state, actions, base_seed)`(전체 맥락·이력·지원의 해시)로 탐색 난수를 정하고, 평가 난수는 쓰지 않는다. 결정적(argmax) 후보의 동점은 정렬 어휘의 첫 인덱스(`np.argmax`)다. 이 결정성이 있어야 로그 행동의 π_cand(a_t|H_t)가 ratio에 쓸 수 있는 하나의 값이 된다.

## 3. ARM-B 식 (구종-only, 준비만, 비활성)

- **기준 π_ref^B(g|H_t)** = 동결 TRAIN `CategoricalBC(prior_strength=20, minimum_action_count=1)`를 공통 지원 `M(H_t)`(BC 양의 지원 ∩ TRAIN token 어휘 ∩ 12카운트 모두에 action별 TRAIN delivery pool, `matrix_policy.PolicyInputs.support`)에 제한한 `SupportedBC.probabilities`. 이 제한·재정규화는 **동결 전 TRAIN에서 한 번** 정하는 정책 정의이며, 평가 시 불일치를 고치는 수단이 아니다.
- **후보 π_cand^B(g|H_t) = kl_policy(Q̂^BC(H_t,·), π_ref^B(·|H_t), M(H_t), τ)** ∝ π_ref^B·exp(Q̂^BC/τ) on M (기존 P3). Q̂^BC는 동결 시뮬레이터(delivery pool에서 물리 추출 → 동결 예측기 결과 → 동결 WE terminal)로 BC continuation의 MC 추정, 매 결정마다 반복. τ는 절대 WE 단위. 후보 지원 = 기준 지원이므로 π̂_b 지원이 M을 덮으면 positivity가 수치상 성립한다(인과 식별은 아님).
- delivery: 모델 항은 동결 TRAIN pool로 주변화, 가중 항의 실제 도달 위치는 구종 선택 뒤 결과(D87).
- 선택 이유: P1/P2(argmax)는 선택 행동 외 ρ=0이라 ESS가 나쁘고, P3는 BC와 같은 지원에서 확률을 옮겨 ratio가 유한하다. **이 선택은 제안이며 τ·예측기·pool identity는 미선정**이다.

## 4. ARM-A 식 (구종×목표, 식만, 식별 보류)

- 행동 = Observer `CandidateAction(index, pitch_type, zone_id, target)`, 대응 표 `index↔(pitch_type, zone_id)` + 해시. 지원 = 기존 `supported_actions`(모든 카운트 kernel ESS≥20, mass≥.01).
- 후보 π_cand^A(g,z|H) ∝ π_ref^A(g,z|H)·exp(Q_A(H,(g,z))/τ_A) on 지원, Q_A = `Recommender.evaluate_pre_pitch`의 PA-WE Q. 기준 π_ref^A(g,z|H) = BC(g|H)·κ(z|g,H), κ는 TRAIN **의도 목표** 법칙.
- **κ와 로그 법칙 π_b(g,z|H)는 존재하지 않는다.** 기존 `observer-repertoire-kernel-v1` 기준선은 실제 도달 위치 커널이며 TRAIN categorical BC가 아니고 의도도 아니다. 그래서 ARM-A 기준 identity·ratio·추정량은 `null`, 상태 `BLOCKED_NOT_IDENTIFIED`(BLK-04). 공식이 쓰여 있어도 대체 실행하지 않는다.

## 5. 추정량 (D87 §5 식, ARM-B에만 적용 가능)

π ∈ {cand, ref}마다 `V^π_T=0`, `V^π_t = v̂^π(H_t) + ρ^π_t (r_t + V^π_{t+1} − q̂^π(H_t,a_t))`, `v̂^π(H_t)=Σ_g π(g|H_t) q̂^π(H_t,g)`, `ρ^π_t = π(a_t|H_t)/π̂_b(a_t|H_t)`, clip·자기정규화 없음, 보상은 PA 종료만. 같은 PA에서 `V^cand_0 − V^ref_0`를 짝지어 평균.

- **q̂^π:** 기준에는 Q̂^BC. 후보용 Q̂^cand(후보 continuation MC)는 비용이 P2/P3 상한 `J·R·Ce·(1+A·S·Cs)`행이다. 비용이 막히면 q̂^cand := Q̂^BC를 **선언된 제어변량**으로 쓸 수 있다. 이때 불편성은 전적으로 π̂_b가 맞다는 가정에 기댄다(아래 합성 검사 (b)).
- **π̂_b(로그 nuisance):** 제안 = 기준과 같은 TRAIN BC를 **별도 nuisance identity**로 등록. 두 역할이 같은 객체라도 이름·해시를 따로 둔다. 이 경우 ρ^ref ≡ 1이라 V̂(ref)는 관측 PA WE + 평균 0 제어변량 잡음에 가깝고, 대비 Δ의 타당성은 **π̂_b가 2026 실제 로그 법칙이라는 연도 간 이동 가정에 전적으로 의존**한다. 대안 π̂_b(투수 빈도)로 민감도를 사전 지정한다. 수치 floor 없음.
- 2026 적합 없음. 교차 적합 예외는 `deferred_not_enabled`(D87).

## 6. 재사용·호환 요약

| 경로 | 쓸 수 있는 것 | 차이·금지 |
|---|---|---|
| `rollout_policy.CategoricalBC/kl_policy/RolloutImprovement/planning_seed` | ARM-B 기준·후보의 호출 가능한 형태 | 모르는 투수에 league fallback → 어댑터가 거절해야 함. identity 없음 |
| `matrix_policy.fit_bc/PolicyInputs.support/SupportedBC/FrozenWE` | TRAIN 날짜 가드, 공통 지원, WE terminal·cutoff | `FrozenGEnsemble`은 3 member 고정(동결 G0는 5) → 예측기 어댑터 필요 |
| `game.terminal_values`, `WinExpectancy.predict_defense` | 초기 수비 팀 WE | WE 아티팩트 `game_values.pkl`(`model-v1.json`)은 레거시 참조, 이번 평가용 미선정·bytes 미검증 |
| `ope.dr.traj_dr_terms_rows` plain | 유효 입력에서 §5 식과 수치 일치(합성 검사) | WDR 자기정규화는 주 추정량 아님. 입력 검증 없음 |
| `ope.ips.pitch_ratios/restrict_support`, `ope.behavior.crossfit_logged` | 없음(레거시 RE24) | NaN π_b→ρ=1, π_b=0→1e-12 floor, 조용한 재정규화, 2026 교차 적합 → **재사용 금지**(fail-closed 위반, 합성 검사로 고정) |
| Observer `evaluate_pre_pitch`/`supported_actions` | ARM-A 대응 표·지원·Q_A | `probabilities`는 결과 확률이지 행동 확률 아님. 기준은 커널, BC 아님 |

## 7. 동결 전에 만들어야 할 것 (미래 등록 대상, 지금 해시 없음)

1. ≤2025 TRAIN BC 직렬화 artifact + 어휘 해시 + 지원 표 해시(P8은 run 안에서 fit, 독립 artifact identity 미확인).
2. Q̂^BC 시뮬레이터 identity: 예측기(G0 5 member 어댑터 또는 P8의 G2/G3 중 무엇인지 결정), delivery pool 400개/action, WE/advancement bytes 검증, sample·cap·seed.
3. τ(ARM-B): 기존 τ=.003은 EXP-P8-001 G2 control 세계의 June 2025 선택값이다. 예측기를 바꾸면 새 정책 identity이며 τ 재선택이 필요하다. 튜닝 구간은 이미 노출된 June 2025뿐이므로 라벨을 붙여 쓸지 결정 필요.
4. π̂_b nuisance 등록(주·민감도), 2026 코드 → 어휘 대응 표(미지 코드는 integrity 실패).
5. PA 중 투수 교체·불완전 PA·ABS 처리 규칙(D87 BLK-06).
6. 위 전부의 Astra 독립 검토와 사용자 결정. 그 전에는 `policy_frozen=false`.

## 8. 합성 검사 (실제 실행됨, SYNTHETIC-ONLY)

`scripts/check_2026_policy_contract.py`(인자·파일 입력 없음, 메모리 toy PA)와 `tests/test_2026_policy_contract.py`. 역사 의존 toy PA(3구종, 4결과, 2볼 볼넷/2스트라이크 삼진, 경로 276개)에서 모든 경로를 확률과 함께 열거해 **기댓값을 정확히** 계산한다: (a) 참 π_b + 틀린 q̂ → 오차 ≤1e-12, (b) 틀린 π̂_b + 참 q̂ → ≤1e-12, (c) 둘 다 틀림 → 편향 확인, (d) clip 1.5는 q̂가 틀릴 때만 편향, (e) 같은 정책 짝 Δ=0, 짝 기댓값 = 참 대비, (f) 레거시 plain DR과 경로별 일치, (g) 실패 처리 표 §2, (h) WE 부호·단위·범위, (i) `CategoricalBC`/`kl_policy` 행이 계약을 통과하고 모르는 투수는 fallback 표시. 명령·출력은 보고서 §3. **식별 증거가 아니다.**

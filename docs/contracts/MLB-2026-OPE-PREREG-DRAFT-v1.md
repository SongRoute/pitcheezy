# MLB 2026 OPE 사전 등록 초안 v1 — DRAFT, 미등록

2026-09-28 작성. 기준 커밋 `522339e`. 기계 판독 필드는 [configs/MLB-2026-OPE-PREREG-DRAFT-v1.json](../../configs/MLB-2026-OPE-PREREG-DRAFT-v1.json), 사용 이력 감사·준비 상태·release blocker는 [준비도 보고서](../reports/MLB-2026-evaluation-readiness-2026-09-28.md)에 있다. 세 파일의 같은 이름 필드는 같은 뜻이다.

> **상태: 초안(DRAFT_UNREGISTERED), `preregistration_complete=false`, `execution.enabled=false`.** 등록 계약이 아니며 2026 자료의 수집·열람·학습·추론·OPE·채점을 허가하지 않는다. 기존 계약에 없는 수치 문턱은 모두 **제안값**이고 측정된 사실이 아니다. 아직 없는 정책·모델·자료·자원 identity는 `null`이며 해시를 만들어 넣지 않았다. 오늘 문서를 해시해도 과거 노출이 지워지거나 원래 의미의 사전 등록이 되지 않는다.

> **2026-09-30 개정(D100·D107 반영, 여전히 DRAFT).** 평가 arm을 **ARM-B(구종만)**로 정했고(D100 ⑨, CV 의도 없음) ARM-A는 보류로 남는다. ≤2025 리허설이 동결한 후보·기준·로깅 법칙·추정기를 §3에 pin하고, **≤2025 S6 결과를 읽기 전에** 성공 기준(§5a), 노출 공개(§9), 2026 실행 전 게이트(§10)를 적는다. 최종 등록과 실행은 §10 게이트 뒤 별도로 한다.

> **2026-09-30 개정 2(Fable 5.1 좁은 검토 COOP-020 조건 반영, 여전히 DRAFT).** 판정 라벨을 통계적 개선(`IMPROVEMENT_SUPPORTED_STATISTICAL`, D108 규칙 그대로)과 편향 한계를 넘는 개선(`IMPROVEMENT_SUPPORTED`)으로 나눴고(§5a), 보상 WE identity를 pin했고(§3), 1회 실행의 hang guard·중단 규칙(§5a), 검정력 공개 자리(§5a), 부분군 다중성 정리(§5), H_t 필드 목록(§5), 스냅샷 출처(§9), ESS 게이트가 τ 0.1에서 구속력이 없다는 사실(§5a)을 적었다. D108 문턱(MEI 0.001, ESS PA 100·경기 30, 무효 0.05)은 바꾸지 않았다.

## 1. 전제

- 2026 정규시즌은 **OPE 전용**이다(CLAUDE.md). 학습·튜닝·모델 선택에 쓰지 않는다. 2026을 G0 예측 NLL 홀드아웃으로 쓰지 않는다.
- 2026의 과거 수집 범위(03-25~09-09)는 **이미 노출됐다**. 모든 2026 행이 노출됐다고 주장하지 않지만 미사용 입증도 없다. P0/P1에서 2026 로그로 π_b를 교차 적합했고(D18), 지지 구종을 2026 표본으로 정했고(D19), 평가 정책 π_e ∝ π_b·exp(Q/τ)의 π_b가 그 폴드별 2026 적합이었으며(D20, `src/pitcheezy/ope/behavior.py::crossfit_logged`), 그 OPE로 모델을 비교·채택했다(D22·D27·D30·D31·D39). 과거 사용에는 nuisance 적합과 **평가 정책·지지 집합의 2026 적응**이 모두 들어 있다. 당시 시행한 절차로 기록과 한계를 보존하되 현재의 2026 학습 허가로 해석하지 않는다.
- 노출 장부는 네 범주로 적는다: (a) 완료 실행 증거가 있는 노출, (b) 설정상 접근 가능, (c) 모름, (d) 제한된 감사 범위 안에서 미사용이 입증됨. **모름은 미사용이 아니다.** 날짜 여집합이나 파일 목록 부재로 미사용을 입증하지 않는다. 현재 (d) 인증 수는 0이다.
- **제한된 메타데이터 사용 이력 감사는 완료됐고 Astra가 독립 검토(PASS)했다**: [감사 보고서](../reports/MLB-2026-use-history-audit-2026-09-28.md), [감사 JSON](../../results/MLB-2026-use-history-audit-v1.json)(Sol 작성, Root 통합), 근거 bytes SHA256 `4db014942c5a14ace5cb5da483ac1126b6139faa9987f8bdb50a6cb103176c40`. 그 범위를 넘는 정확한 경기 ID·전체 시즌 가용성·미사용 인증은 모름으로 남는다(`BLK-05`). 노출 구간을 명시한 탐색적 분석에는 미사용 인증이 필요 조건이 아니다.
- G0는 10-class 결과 예측 모델이며 호출 가능한 행동 법칙·정책이 아니다. June 보정(D86)의 세 대비(B1−B0, B2−B0, B2−B1)는 모두 N3 미확정이다. B1/B2는 선정되지 않았고, B0는 후보가 아니라 동결 G0 기준선으로 **유지**된다. 예측기 시드 5개는 정책 확인 5번이 아니다.

## 2. 평가 대상(universe)

| 필드 | 초안 값 |
|---|---|
| 리그·경기 유형 | MLB, Statcast `game_type == "R"`만. 포스트시즌·시범경기·올스타 제외 |
| 기간 | 제안 달력 경계 2026-03-25~2026-09-27(미국 날짜). 출처: [MLB 2026 정규시즌 일정 발표](https://www.mlb.com/press-release/press-release-mlb-announces-2026-regular-season-schedule), Root가 2026-09-28 웹으로 공개 확인(`public-context-supplement.json`). 이 초안 작성자의 독립 열람은 아니다. 공식 달력은 실제 보유 수집 범위(09-09까지)나 아직 없는 검증 스냅샷과 다르다. 포스트시즌 시작 2026-09-29(Root 보고) |
| 투수 | 선발·불펜 전체. 역할은 경기별 팀 첫 투수=SP, 나머지=RP(제안). 오프너·야수 등판은 제외하지 않고 플래그 |
| 시작 모집단 | 투구 전 필드로만 정의한 고정 PA 시작 집합. **평가 시작점은 각 PA의 첫 투구(0-0) 하나, PA마다 가중치 1**(제안). C0의 현재 카운트→PA 종료 정의만으로는 시작 상태 표집이 정해지지 않기 때문이다. 불완전 PA·교체·여러 투수·투구 없는 종료·비표준 카운트·이닝 전환을 명시 처리. 사후 "완전·지원 PA"만 고르면 선택 편향이 생기므로 그 경우 좁은 추정량 이름을 따로 붙인다 |
| 노출 구간 | `W_exposed` 2026-03-25~2026-09-09(과거 수집본 범위), `W_post` 2026-09-10~정규시즌 종료. `W_post`는 노출 범주 **(c) 모름**이며 감사로 (d)가 입증될 때만 사전 지정 부분 분석으로 따로 보고한다 |
| 주 모집단 | 정규시즌 전체, 라벨 `contains_previously_exposed_window`. 공식 달력 범위와 실제 보유 수집 범위(09-09까지)를 구분한다. 어떤 결과도 독립 확인이라 부르지 않는다. 노출 자료의 새 분석은 수정된 탐색적 평가다 |
| 분모 | 전체 R 경기 → 수집 경기 → 적격 PA 시작 → 지원 요청 → 평가 가능 PA. 각 단계 수·제외 사유 보고. 미지원·결측 요청은 분모에 남는다 |

## 3. 추정량

### ARM-A — 제품 주 추정량 (기본 주 추정량, **식별 전까지 보류**)

- 가치: [C0](C0-v1.md) `defense-we-pa-v1`. 초기 수비 팀의 동결 `W(S_PA_end)` [0,1], 현재 카운트부터 **이번 타석 종료까지**, 이후 동결 WE continuation. 이닝이 바뀌어도 같은 수비 팀. 모델 가치 기반 PA continuation이지 관측 최종 승리 개선·경기 전체 개입·이닝 교체 가치가 아니다.
- 비교: 후보 π_cand(동결된 구종×목표 위치 정책) 대 기준 π_ref(**동결 TRAIN BC**). 관측 2026 행동은 BC와 같지 않으며 별도 기술 비교다. 알 수 없는 로깅 행동을 실행 가능한 기준 정책 대신 쓰지 않는다.
- 효과: `Δ = V_WE(π_cand) − V_WE(π_ref)`, 무차원 확률 차, 표시 `delta_pp = 100·Δ`(%p). **양수 = 후보가 초기 수비 팀에 유리.**
- 상태 `BLOCKED_NOT_IDENTIFIED`: 관측 plate 위치는 실제 도달이지 의도 목표가 아니다. 관측 위치로 목표 propensity를 만들어 인과 WE라 주장하지 않는다. CV 의도 추정이나 도달 밀도 모형만으로도 교환 가능성·실제 logging propensity가 성립하지 않는다(`BLK-04`).

### ARM-B — 구종-only 제한 추정량 (**평가 arm, D100**)

- 같은 WE 단위·방향·PA horizon. 후보와 동결 TRAIN BC 기준이 **구종 확률만** 다르다. 모델 항의 delivery는 2026 이전 동결 입력으로 주변화하고, 가중 항에서 실제 도달 위치는 구종 선택 뒤의 결과로 둔다. [ML_EXPERIMENT_MATRIX](../ML_EXPERIMENT_MATRIX.md) §8 P1~P3의 축소 가정과 같다.
- 목표 위치 효과나 서비스 추천의 인과 WE가 아니다. 구종 배정의 교환 가능성(H_t 조건부)과 ≤2025→2026 이동은 **가정으로 명시하고 입증하지 않는다**(`BLK-04` 미해소). 결과는 항상 "구종-only, 명시 가정 조건부, 노출 구간 포함 탐색적 평가"로 표시한다.
- 사용자 결정(D100 ⑨)으로 **주 추정량 = ARM-B**다. 새 2026 열람 전에 정했고 결과를 본 뒤 바꾸지 않는다. ARM-A의 자동 대체가 아니며 ARM-A는 `BLOCKED_NOT_IDENTIFIED`로 남는다. 위치는 observer `experimental_location_proxy` 표시용 근사일 뿐 OPE 대상이 아니다.
- **동결 구성요소(≤2025 리허설 [ML-POLICY-MATERIALIZATION-v1](ML-POLICY-MATERIALIZATION-v1.md), D104–D107):**
  - 후보 π_cand: TRAIN 레퍼토리(지원 표 M) 위 P3 `kl_policy`, π ∝ π_ref·exp(Q_ref/τ), **τ = 0.1**(S3b-tau-a2), samples 6·pitch_cap 12(S3-profile-a2). 최종 정책 식별자 `c13cc98994ae9eb08a8f939a583ebe712312594ac25c05314a9ca790bd46f73c`(addendum 5 `tau_freeze.json` sha `1aedb2f5…`).
  - 기준 π_ref: SupportedBC(BC-P를 M으로 제한·재정규화, 동결 전 1회). BC-P sha `b4f274eb…`, 지원 표 sha `152cdab4…`(addendum 2).
  - 로깅 법칙 π̂_b: **전체 어휘 BC-P, 재정규화하지 않는다**(D-1). 2026 적합 없음.
  - DR q̂: 평가 seed의 동결 simulator MC(M-7, S3b에서 기계적으로 정해 식별자에 포함). 후보는 q̂ = Q^ref라 **단일 강건**, 기준은 이중 강건(COOP-019 F5).
  - 보상 WE(`FrozenWE(inputs, game_values)`, `game.terminal_values`·`WinExpectancy.predict_defense`): C0 계약 [model-v1.json](model-v1.json) sha256 `d0f1d452331d3f7ad32ca310442bad3ef61bab1b16747829f4190c0f0653ac13`, bundle manifest `43ece920cb60c6c24ea9f1e720a7000b3b83038a3e24ac27d77ef17a2e8f0f1f`, source lineage(`source_hashes.json`) `6d6f777bacde98d5c60b2d61cd594fda172fd63f99c315a002b5e7291afb099a`, `game_values.pkl` `aa6c4e486cc6b027758d966ab680566a4a4ce2afa4a918dd26f8fb65c8957677`(ML-POLICY-MATERIALIZATION-v1 `identity_registration`·`model-v1.json`; 계약→manifest→lineage→bytes 순 확인, ≤2025 리허설과 같은 WE). 이 WE는 원래 빌드에서 **2025 DEV Brier gate를 통과**했다(DEV 노출, P8 기록). 2026에는 적합하지 않는다.
  - 추정기: DR v2, 검열 노드 [0,1] 경계(D-5, L0/L1/L2·E0), 경기 부트스트랩 10,000회, ESS 게이트 PA 100·경기 30(두 정책 중 최소).
  - 타자 프로필: `profile_as_of` = 2026-03-25(배타), 원천 날짜 가드 `max(game_date) ≤ 2025-12-31`와 2025-12-31 rolling 동일성 검사(COOP-019 F2, `fa82afb`).
  - R7 타석 중 투수 교체: **보류**(D100 ⑦). 현 제안 규칙(알려진 새 투수 계속, 미지 새 투수 `UNSUPPORTED_UNKNOWN_PITCHER`→`UNSUPPORTED_MID_PA`)을 임시로 쓰고 비율·민감도를 보고한다.

### LEGACY-RE24 — 감사 부록 (분리 보존만)

기존 P0/P1 RE24 결정 단위 1스텝 SNIPS/DR(`results/EXP-P0-*.json`, `results/EXP-P1-*.json`의 OPE 행)은 별도 부록으로만 인용한다. WE로 단위 변환·라벨링하지 않으며 ARM-A/B가 막혔을 때의 대체 결과도 아니다. D28의 "freeze 뒤 재실행"은 새 2026 nuisance 적합을 포함하므로 `deferred_not_enabled`다.

## 4. Nuisance·상태 입력

- **기본: 어떤 구성요소도 2026으로 적합하지 않는다.** propensity, 결과 회귀, 보정·temperature, 의도/delivery, 지지 집합에서 파생한 평가 정책 모두. 2026 폴드 교차 적합도 2026 학습이다. `nuisance_2026_crossfit_exception`은 `deferred_not_enabled`이며 켜려면 허용 필드·모델·폴드·평가 정책 독립성·불확실성 처리·노출 장부 수정이 담긴 사용자 결정과 decisions.md 기록이 필요하다.
- ~~WE continuation: [model-v1.json](model-v1.json)의 `game_values.pkl`은 **레거시 참조**일 뿐 이번 평가용으로 선택되거나 bytes를 새로 검증한 identity가 아니다. 선택·payload 검증은 수행하지 않았다(`null`).~~ **대체됨(2026-09-30, COOP-020 M1):** ARM-B 보상 WE는 §3 ARM-B에 전체 SHA로 pin한 `FrozenWE` identity(≤2025 리허설과 동일)다. 최종 bytes 재확인은 M3b에서 한다.
- 실제 기록된 투구 전 배정 확률이 있으면 우선한다. 없으면 ≤2025 동결 π_b는 해시가 있을 뿐 2026 참 법칙이 아니다. 연도 간 이동 가정과 비측정 교란 민감도 분석이 필요하고, 수치 floor가 positivity를 만들지 않는다. 평가 정책을 2026과 독립으로 정해도 nuisance가 비편향이 되지 않으며 DR은 숨은 교란·잘못된 개입 라벨·clip 편향을 고치지 않는다.
- **수치 overlap은 인과 식별이 아니다.** 인과 해석에는 일관성, 순차적 교환 가능성, 양성(support), 로깅 배정·연도 간 이동 근거가 모두 필요하다. 이 가정이 근거로 뒷받침되지 않으면 overlap이 좋아도 `BLOCKED_NOT_IDENTIFIED`다. 동결 π_b의 2026 로그 행동 NLL·보정 진단은 교환 가능성을 입증하지 않으며, 미래에 쓰려면 별도 등록·승인이 필요하고 예측기 선택에 쓰지 않는다. 식별 가정은 근거가 있지만 overlap 게이트에 실패하면 `UNCONFIRMED_WEAK_OVERLAP`. 효과를 0이나 성공으로 채우지 않는다.
- **동결 모델 ≠ 동결 상태.** 별도 승인된 미래 평가에서 고정 정책은 등록된 입력 스키마로 2026의 투구 전 경기·카운트 필드와 같은 PA의 엄격히 이전 투구 이력을 소비할 수 있다. 이것은 추론 맥락이다. 반면 2026 선수 프로필·레퍼토리를 굴려 갱신하는 것은 통계 구성요소 추정이므로 기본적으로 2026 이전 스냅샷에 동결한다. 현재 공의 물리·결과 값은 추천에 들어가지 않는다.
- 지지는 TRAIN·합법 제약에서만 정한다. 2026은 overlap을 진단할 수 있으나 정책·지지 재정규화·사례 교체·문턱 재조정에 쓰지 않는다. 지원·action mapping 불일치(지원 밖 질량, 해시 불일치, 구종 코드 누락)는 **fail-closed**: 중단·`FAILED_INTEGRITY`, 대체·축소 없음. 평가 정책 질량 0은 정당한 비 0이고, 로깅 확률 0/모름은 별개 문제로 기록한다.

## 5. 추론·진단 (제안값)

| 필드 | 제안 | 근거·상태 |
|---|---|---|
| 주 추정기 | ARM-A: `null`(보류). ARM-B: clip 없는 per-decision sequential DR, 고정 nuisance, 같은 PA에서 후보–기준 짝. π ∈ {후보, 기준}마다 따로: PA i의 결정 t=0..T−1, `V^π_T=0`, `V^π_t = v̂^π(H_t) + ρ^π_t·(r_t + V^π_{t+1} − q̂^π(H_t,a_t))`, `v̂^π(H_t)=Σ_a π(a|H_t)·q̂^π(H_t,a)`, `ρ^π_t = π(a_t|H_t)/π_b(a_t|H_t)`(구종 행동). `H_t`는 등록된 요청 필드 목록이다(`matrix_policy.SAFE_COLUMNS` + `policy_requests` 요청 상태): 현재 카운트(`balls`, `strikes`), 같은 PA의 엄격히 이전 투구 전부(구종 라벨, 정규화 물리, 10-class 결과, 투구 전 카운트), 이닝·초말(`inning`, `inning_topbot`)·아웃(`outs_when_up`)·주자(`bases`)·점수(`home_score`, `away_score`), 타자 `stand`와 as-of 타자 프로필(`batter_style_*_prior`·`_reliability`, 6개 스타일 × 2), 투수 식별자·손(`pitcher`, `p_throws`, TRAIN 단일 손 등록부)과 TRAIN 레퍼토리(지원 표 M). 이 필드가 구종 배정에 충분하다는 것(순차적 교환 가능성)은 **식별 가정**이며 입증하지 않는다. 보상은 PA 종료 단계의 `W(S_PA_end)`만. 두 정책을 각각 재귀한 뒤 같은 PA에서 `V^cand_0 − V^ref_0`를 짝지어 평균, 자기정규화 없음 | 225행동 궤적 IS는 ESS≈0(D19). ARM-B는 구종 행동으로 축소. 식별·overlap 조건부. 구종-only로 자동 전환하지 않는다 |
| 보조 | IS/SNIS(분모는 재표집마다 재계산), DM-only, 투구 단계 비에 clip {10, 20, 50}을 곱하기 전 적용하는 민감도 | 진단 전용. 유리한 추정기 선택 금지. DM-only는 OPE 근거 아님(D31 순환) |
| 불확실성 | 경기 단위 paired 재표집(PA 유지), B=10,000, 양측 percentile 95% CI, 재표집 안에서 자기정규화 분모 재계산 | 제안. seed·무효 replicate 처리는 등록 시 고정(현재 `null`). CI는 동결 정책·nuisance·WE 조건부 |
| 다중성 | ARM-B 한 family, **주 대비 1개, 확증적 2차 대비 0개**. 부분군(`W_post`·SP/RP 포함)은 모두 기술 전용이며 다중성 주장·판정이 없다(§5a) | 제안 |
| overlap 게이트 | PA ESS = (Σ_i w_i)²/Σ_i w_i², w_i는 PA 종료 누적 비. game ESS는 경기별 w 합으로 같은 식. PA ESS ≥ 100, game ESS ≥ 30 | 휴리스틱 제안. ≤2025/합성 작업으로 근거를 만든 뒤 고정 |
| 최소 효과 | **0.001 WE/PA**(제안, §5a) | 판단값. RE24 해상도(D31)·ML ΔNLL −.003·R78·kernel ESS20·clip20을 OPE 정리로 상속하지 않는다 |
| 진단 | 투구별 비 분위·모멘트, 누적 PA/경기 집중도, 0/무효 분모, 지원 밖 질량, clip 비율, 결측·중도 절단, 역할·월·투수 볼륨별 ESS | — |

판정 상태: `IMPROVEMENT_SUPPORTED`·`IMPROVEMENT_SUPPORTED_STATISTICAL`(§5a; 식별 가정의 근거 ∧ CI 하한 > 0 ∧ 등록 최소 효과 ∧ overlap·무결성 통과), `HARM_SUPPORTED`(CI 상한 < 0, 같은 게이트), `NO_EFFECT_DETECTED`, `UNCONFIRMED_WEAK_OVERLAP`, `NOT_IDENTIFIED_CAUSAL_NULL`, `BLOCKED_NOT_IDENTIFIED`, `FAILED_INTEGRITY`. 계산 불가 값은 `null`과 사유. 최소 효과가 `null`인 동안은 우위 판정이 없는 기술적 연구이며 모델 선택·서비스 승격은 따라오지 않는다. ARM-B의 구체 판정 규칙은 §5a가 이 표보다 우선한다.

## 5a. ARM-B 성공 기준 (≤2025 S6 결과 열람 전 고정)

> **개정 3(2026-09-30, COOP-021/022, ≤2025 S6 열람 뒤 — 공개).** ≤2025 S6(D115)에서 v1 L1 폭이 검열 질량의 약 2배(0.091)로 고정되는 것을 본 뒤 검열 처리를 바꿨다. 그래서 이 개정은 S6 열람 뒤의 변경이며, 새 등록 ID(`mlb2026_ope`, 러너 `ope-2026`)로 한다. MEI 0.001, ESS 게이트, 무효 0.05, 판정 순서는 그대로 둔다.
> - **주 추정치 = L1-R**
>   - 로깅 양성 노드: ρ = 0(두 정책 모두 그 행동에 질량 0이라 정확함). 런타임 v3가 기록한 π·q̂로 V_k = v_π(H_k)를 계산한다.
>   - H_k로 정해지는 타석 중 거절(`NO_LOGGED_ACTION`, `UNKNOWN_PITCHER`, `PITCHER_HAND`, `EMPTY_SUPPORT`, `INCONSISTENT_HISTORY`)은 **regime으로 처리한다: π를 첫 H_k-확정 사건까지 따르고, 그 뒤는 로그 행동. V(π) 자체에 대한 판정은 없다.** 무투구(`automatic_ball/strike`만; 구종이 붙은 무투구 행 0, D97)에는 `P(no-pitch | H_k, type) = P(no-pitch | H_k)`를 가정한다.
>   - 나머지 검열(NO_TERMINAL, `MISSING_ACTION_LABEL`, 구조 결함)은 D-5 최악 경계로 둔다.
> - **G-R 게이트:** E0에서 최악 경계로 남은 비율 s ≤ **0.005**. 넘으면 `NOT_DECIDABLE_CENSORING`이다. 결과 옆에 통과 가능 효과 근사식 "Δ > MEI (+ b_V) + s + 1.96·SE, 최악 경계 폭 ≈ 2s"를 출력한다.
> - **b_V = 0.002734729542731599**(봉인 S5 `v2.json`에서 계산한 값 그대로 등록; D111의 0.00272는 반올림 오기). 따라서 `IMPROVEMENT_SUPPORTED` 문턱은 CI 하한 > MEI + b_V = **0.0037347**이다.
> - **민감도(판정 없음):**
>   - S-v1: D-5 최악 경계
>   - S-NP: 무투구만 최악 경계
>   - S-B: 이 PA 이전(더 이른 날짜, 또는 같은 경기의 앞 타석)에 TRAIN 지원 밖 구종을 던진 투수의 PA를 E0에서 뺀다. **부호 규칙:** S-B 라벨이 주 라벨과 다르면 보고서 첫 줄에 적는다.
>   - S-C: 노드에서 |V_c − V_r| ≤ δ. δ는 ≤2025 q99 0.0080·q999 0.0278이며, 단일 노드 기준이라 여러 단계 차이를 과소 경계한다.
> - **π̂_b 오지정 크기:** 새 구종 질량 f를 놓치면 다른 비율이 (1 − f)배로 줄고, 한 단계 오차는 −f·(V − v̂)다(toy로 정확히 확인). ≤2025 비율로 보면 401/44,230 × G0 TV q99 0.0058 ≈ 5e-5 WE/PA다(대략적 상한, 측정값 아님).
> - **hang guard:** ≤2025 S6(v3 재실행) 실측 초 × (2026 경기 수 / 150) × 등록 배수. 경기 수는 불러온 스냅샷에서 결과를 보지 않고 세며, 그 직후 guard를 건다(M3에서 미리 세는 대신 같은 스냅샷에서 기계적으로 셈).
> - **시도 규칙:** 한 등록에서 봉인은 1회다. 실패 시도가 `max_attempts`(2)에 이르면 `FAILED_INFRA`다. 필요한 자료가 없는 거절은 시도 디렉터리를 만들기 전에 낸다.
> - **2026 스냅샷 처리:** `d20260930-h2026f`(sha pin)만 읽는다. 정규시즌(R)과 2026-03-25~09-27만 받고, 여러 날짜에 걸친 경기(서스펜디드)는 일정만 보고 빼며 목록을 공개한다. 타자 프로필은 개막일 스냅샷(F2 가드와 end-of-history 검사)을 쓴다.
> - **어휘 매핑 규칙(Fable 5.1 F-M1):** TRAIN 어휘 밖 2026 구종 코드는 실행을 막지 않고 **구종 누락(MISSING)으로 바꾼다.** 그 결정은 `UNSUPPORTED_MISSING_ACTION_LABEL` 거절(타석 안에서 sticky)이 되어 최악 경계 잔여로 G-R 비율 s에 들어간다. 이력 토큰은 원래도 미지 구종(0)이라 바뀌지 않는다. 바뀐 행 수·비율과 코드별 개수를 결과(`vocabulary_mapping`)에 공개한다. 바뀐 행이 구종 표시가 있는 행 중 등록 상한 `vocab_unmapped_share_max`를 넘으면 `REFUSED_VOCABULARY`로 실행하지 않으며, 이 거절은 시도 디렉터리를 만들기 전에 내므로 시도로 세지 않는다(`FAILED_INFRA` 아님).

2026-09-30, ≤2025 S6(`S6_V4`) 결과를 읽기 전에 적는다. S6·V2 결과를 본 뒤 아래 값을 바꾸면 그 사실과 이유를 공개하고 새 등록 ID로 한다.

- **주 추정치:** L1(투구 전 시작 모집단 E0) `Δ = V(π_cand) − V(π_ref)`, 초기 수비 팀 WE, PA당, 같은 PA 짝. 검열 노드 경계 때문에 Δ는 구간 `[Δ_lo, Δ_hi]`다.
- **최소 효과(MEI) = 0.001 WE/PA(0.1%p) — 판단값, 측정 근거 없음.** 근거: (1) 한 팀은 경기마다 수십 PA를 수비하므로 PA당 0.1%p가 더해진다면 시즌 단위로 무시할 수 없는 규모다(PA 간 가산성은 가정, 승수 환산은 미측정). (2) 이보다 작은 값은 동결 π_b·q̂의 모형 오차와 구분하기 어렵다고 본다. **주의:** ≤2025 V2 편향 허용(tolerance 0.01, `delta_gap_se ≤ 0.005`)은 MEI보다 10배 크므로 V2 통과가 MEI 해상도의 비편향을 보증하지 않는다. τ = 0.1 후보는 평균 KL 0.0074로 BC에 매우 가까워 MEI 미만 결과가 나올 가능성이 크다.
- **판정 규칙(한 번만 적용):**
  - `IMPROVEMENT_SUPPORTED_STATISTICAL` ⇔ ① `Δ_lo`의 95% 경기 부트스트랩 CI 하한 > 0.001 ② 두 정책 모두 ESS 게이트(PA ≥ 100, 경기 ≥ 30) 통과 ③ 부트스트랩 무효 비율 ≤ 0.05 ④ 무결성 실패 0(`FAILED_INTEGRITY` 없음, 짝 identity 실행 cand=ref 차 정확히 0). (D108 규칙 그대로, 이름만 바뀜.)
  - `IMPROVEMENT_SUPPORTED` ⇔ ②–④ 통과 **그리고** `Δ_lo`의 95% CI 하한 > MEI + b_V. 편향 한계 `b_V := |delta_gap_V2| + 1.96·delta_gap_se_V2`는 봉인된 ≤2025 S5 `v2.json`의 V2(`pi_b_hat`) 실행에서 읽는다. S5는 V2를 planning seed {0,1,2}(`seeds.planning_v2`, 주 seed `planning_main` = 0 포함)마다 돌리므로 **seed별 b_V의 최댓값**을 쓴다. `IMPROVEMENT_SUPPORTED`는 `IMPROVEMENT_SUPPORTED_STATISTICAL`을 함축한다.
  - **V3 오지정 보고(판정 조건 아님, 등록된 필수 보고):** 같은 `v2.json`의 V3 tempered α = 0.5(`tempered_alpha_0.5`) 실행의 `|delta_gap_V3|`(seed별 최댓값)를 2026 결과 옆에 싣는다. `|gap_V3| > MEI`이면 결과에 `pi_b_misspecification_sensitivity_exceeds_MEI` 표시를 붙인다.
  - 이유: V2 로그는 π̂_b 자신이 생성하므로 V2는 π̂_b가 틀린 경우(실제 로깅 법칙 ≠ π̂_b)를 전혀 시험하지 못한다. 그런데 후보 쪽 DR은 q̂ = Q^ref라 **단일 강건**이어서(COOP-019 F5) 편향이 π̂_b 정확도에 그대로 달려 있다. 그래서 V2가 측정한 추정기 편향 한계(b_V)만큼 문턱을 올려야 "MEI를 넘는 개선"이라 말할 수 있고, 로깅 법칙을 π̂_b와 다르게 둔 V3의 차이를 오지정 민감도로 함께 공개한다. 대안(MEI 자체를 0.01 이상으로 올려 V2 허용치와 맞추기)은 채택하지 않았다: D108 MEI를 바꾸지 않고 통계적 결과(`_STATISTICAL`)와 편향 한계를 넘는 결과를 분리해 둘 다 보고하는 쪽을 택했다.
  - `HARM_SUPPORTED` ⇔ `Δ_hi`의 95% CI 상한 < 0 이고 ②–④ 통과.
  - ②·③ 실패 → `UNCONFIRMED_WEAK_OVERLAP`, ④ 실패 → `FAILED_INTEGRITY`. 그 밖 → **"개선 근거 없음"**(`NO_EVIDENCE_OF_IMPROVEMENT`; 0 < CI 하한 ≤ MEI도 여기다).
  - ESS 게이트는 τ 0.1에서 **사실상 구속력이 없다**: ≤2025 June S3b에서 후보 PA ESS 9,779(기준의 93.4%), 경기 ESS 147로 문턱(100·30)을 크게 넘었다(D107). 따라서 `UNCONFIRMED_WEAK_OVERLAP`은 이번 등록에서 살아 있는 안전장치로 기대하지 않는다(2026에서 실패하면 규칙대로 적용한다).
  - 어느 경우에도 인과 식별은 가정이며(`identification = assumed_not_evidenced`), 결과는 서비스 승격을 자동으로 허가하지 않는다.
- **1회 규칙:** 동결 스냅샷 `d20260930-h2026f`에서 **정확히 한 번** 실행한다. 재실행은 인프라 실패(hang guard·OOM·디스크 등, 보상/WE/결과 열람 전, 실패 로그·부분 원장 보존)일 때만 같은 등록·같은 범위로 한다. 결과 열람 뒤 재실행·문턱·τ·추정기·표본 변경은 없다.
  - **hang guard(M3b에서 고정):** ≤2025 S6 실측 비용(초) × (n_2026_games / 150) × 2. n_2026_games는 스냅샷의 적격 R 경기 수(M3에서 결과 비열람으로 셈).
  - **중단 시:** 부분 원장은 해시로 봉인하고 **열지 않는다**(보상·WE·Δ 요약 포함 어떤 값도 읽지 않음).
  - **시도 수:** 최대 2회. 두 번째도 인프라 실패면 `FAILED_INFRA`로 끝내며 세 번째 시도는 없다(새 등록 ID로만 가능, §7).
  - **체크포인트 재개 없음:** 비트 단위 결정론이 입증되고 그 재개 절차가 M3b에 등록된 경우가 아니면 처음부터 다시 실행한다.
- **함께 보고(판정에 쓰지 않음):**
  - L0(전 PA 경계), L2(E0 ∩ 완료, 처치 후 선택 조건부 평균 — 판정 없음), 검열 비율(사유별).
  - R7: 타석 중 투수 교체 비율·미지 새 투수 비율과 2차 추정량(첫 교체 뒤 자연 경과) 민감도, 표시 "보류".
  - 부분군(기술, 다중성 보정 없음): 선발/불펜(D87 역할 규칙), `bc_p_only_pitcher` 층, 월, TRAIN 볼륨 3분위, 연장, `W_exposed`/`W_post`.
  - COOP-019 F3: 게이트 옆에 E0 전체(완료 ∪ 검열) 가중치의 최소 ESS. F5: 후보는 q̂ = Q^ref라 단일 강건(π̂_b가 틀리면 후보 쪽 편향 가능), 기준은 이중 강건이라는 문장.
  - 보조 추정기 IS/SNIS/DM-only, clip 민감도 — 진단 전용.
- **검정력 공개(자리만, 문턱 불변):** ≤2025 S6 봉인 뒤 S6 Δ 점추정·SE(`SE_S6`)와 예상 2026 SE = `SE_S6·√(150 / n_2026_games)`를 여기에 기록한다. 공개용이며 MEI·게이트·판정 규칙을 바꾸지 않는다. 현재 값: `null`(S6 미봉인).

## 6. 커버리지·ABS

- 2025 이전 이력이 없는 2026 신규 투수·새 구종은 지원 밖이다. 빼면 베테랑 쪽으로 치우친다. 관측 행동을 넣거나 비를 1로 두는 방식으로 전체 효과를 만들지 않는다. 미지원 요청에서 쓸 fallback은 호출 가능한 동결 법칙이고 식별될 때만 등록할 수 있다(`null`). 그 전까지 전체 모집단 인과 가치는 `null` 또는 [0,1] 최악 경계, 요청·커버 분모와 투구 전 필드로 정한 제한 모집단 결과만 보고한다. 후보가 새 미지원 상태로 이끌 수 있다는 점도 기록한다. 이후 관측된 지원·완료로 PA를 빼지 않으며, 불완전 PA·중도 절단 처리 규칙은 등록 전 확정(`null`).
- 2026 ABS 챌린지 도입은 경계 판정 분포를 바꿀 수 있는 **잠재적 체제 변화**다([design.md](../design.md) 동일성 경계). 규칙이 생겼다는 사실이 이 자료의 변화 크기를 측정하지 않으며 크기·스키마 영향은 측정되지 않았다. [MLB 공식 ABS 안내](https://www.mlb.com/news/abs-challenge-system-mlb-2026)에 특수 이벤트 예외가 있으므로 2026 모든 경기에 ABS가 쓰였다고 가정하지 않는다. 이벤트별 처리는 M2 스키마 계획에서 정한다. nuisance는 인간 심판 존에서 학습됐다. 관측 위치의 경계 띠 투구 비율은 **기술 진단**으로만 보고한다. 실현된 위치로 투구를 빼는 것은 처치 후 선택이며 PA 궤적을 깨므로 OPE 민감도로 쓰지 않는다. 경계 폭·대응은 `null`. Statcast가 번복 판정을 어떻게 기록하는지, 챌린지 잔여 수 같은 상태 변수가 있는지는 미확인(`null`, `BLK-06`).

## 7. 실패·재시도

- 결과 라벨을 연 뒤 재시도·문턱 조정·추정기/후보 교체·표본 축소·optional stopping 없음. 등록된 추론 칸은 null을 포함해 모두 남긴다. 실패는 보존한다.
- 새 시도는 새 등록 ID로 하고 이전 시도를 공개한다. 해당 구간은 이후 `reused_after_unblinding`.
- 라벨 열람 전 무결성 실패는 수정·재등록할 수 있고 이력을 남긴다.

## 8. 단계

`M0` 메타데이터 사용 이력 감사·이 초안(현재) → `M1` 정책·추정량·출처·자료 가용성 해소(≤2025/합성으로 정책 어댑터·정규화·WE 부호/단위·fail-closed·알려진 정책 OPE 복원 검증, 식별 검토 포함) → `M2` **새 자료 접근 전** 과학·수집/QA 계획 등록(추정량·정책·nuisance pin, 달력·필드 허용 목록·ABS 이벤트 처리·접근 기록·품질 비노출 QA·중단 규칙·문턱, Astra 독립 검토) → `M3` 별도 승인된 **격리 스냅샷과 품질 비노출 QA**. 실제 자료 identity(스냅샷 버전·manifest 해시)는 이 단계에서 처음 생긴다 → `M3b` **최종 실행 등록 게이트**: 실제 스냅샷/manifest 해시, source C, plan D, 환경, 예산을 pin하고 독립 검토를 받는다 → `M4` 제한 profile·실행(M3b 통과 뒤 별도 사용자 실행 승인·Root release, 단일 heavy lock) → `M5` 감사·보고. 어느 단계도 이전 단계 통과만으로 자동 승인·release되지 않으며, 이 초안은 새 권한을 만들지 않는다. **이번 산출물은 M0까지다.** blocker와 담당은 [준비도 보고서](../reports/MLB-2026-evaluation-readiness-2026-09-28.md) §4.

2026-09-30 개정: M1(정책·추정량 정의, ≤2025 리허설)은 S5·S6만 남았다. 이 개정은 M2의 과학 계획 부분 초안이며 등록이 아니다.

## 9. 노출 공개 (보고서에 그대로 싣는다)

- 평가 스냅샷 `d20260930-h2026f`: 2026-03-25~09-27, 716,792행, sha256 `c0c1eda4b515fc14dab79defe577b449e2695d40dd036f5c35ae52e37093ed9b`, 동결(`frozen=true`, [data/versions.md](../../data/versions.md)). 2026 적합·튜닝·선택에 쓰지 않는다.
  - **수집 출처:** 수집 커밋이 `6bd76fd-dirty`로 기록돼 있다. `-dirty`는 `git status --porcelain`에 변경이 있으면 붙는다(`statcast_fetch.git_commit`, `data/versions.md`만 제외). 이 저장소는 로컬에서 대상이 바뀐 `runs` symlink가 늘 변경으로 잡히고, 수집 중에는 `experiments/pitchmdp/pitchmdp/policy_artifacts.py` 편집도 진행 중이었을 수 있다(수집 시작 시점의 정확한 목록은 기록되지 않음). 수집 코드 `src/pitcheezy/data/statcast_fetch.py`·`scripts/fetch_data.py`는 바뀌지 않았다: `git diff 6bd76fd 70968eb -- src/pitcheezy/data scripts/fetch_data.py`가 비어 있다(2026-09-30 확인).
  - **경로:** 상대 경로는 이전 수집본과 같은 `holdout_2026/statcast_2026.parquet`지만 버전 디렉터리가 다르다(`d20260911-s2325/holdout_2026/` 대 `d20260930-h2026f/holdout_2026/`). 이전 파일(sha `374e8b95…`, 647,896행, `frozen=false`)은 덮어쓰지 않았고 versions.md의 기존 행도 수정하지 않았다.
  - **격리:** ≤2025 단계 실행기의 `guard_dates`가 2025-12-31 이후 날짜 행을 거부하므로 이 스냅샷은 ≤2025 리허설(S0–S6)에 들어갈 수 없다.
- 이전 수집본 `d20260911-s2325`(03-25~09-09)는 과거 **P0 12건·P1 8건 OPE**에 쓰였다. 선택 표본 68,425PA(ip100, 1,993경기), 2026 π_b 교차 적합(D18)·지원 집합(D19)·평가 정책 적응(D20)·모델 비교·채택(D22·D27·D30·D31·D39) 포함. 정확한 사용 경기 ID는 모름.
- 따라서 `W_exposed`(03-25~09-09)는 노출 구간이고, `W_post`(09-10~09-27)는 이전 수집본 밖이지만 미사용 인증이 없어 **(c) 모름**이다. 주 결과는 전체 시즌이며 `contains_previously_exposed_window` 라벨을 붙이고 독립 확인이라 부르지 않는다.
- 이번 후보·기준·π̂_b·q̂·τ·프로필은 모두 ≤2025 자료로만 정했다(τ는 June 2025 겹침·잡음 규칙, 결과값 비열람; D106의 격자 확장은 June 겹침 표를 본 뒤였다). 과거 2026 노출이 이번 후보 설계에 들어간 경로는 알려진 것이 없지만, 설계자가 과거 2026 OPE 결과를 알고 있었다는 사실은 남는다.

## 10. 2026 실행 전에 모두 성립해야 하는 게이트

1. ≤2025 S5 V2 채택: 실행기 accept **그리고** `delta_gap_se ≤ 0.005`(COOP-019 F1). 아니면 V2 미확정으로 S6·2026 모두 진행하지 않는다.
2. S6 열람 전 봉인 V2 원장의 IS-only(q̂ := 0) 확인(COOP-019 조건 2).
3. ≤2025 S6 완료·봉인(짝 identity 실행 통과, F3·F5 보고 포함).
4. 이 초안(특히 §3 ARM-B pin, §5a, §9)에 대한 **Fable 5.1 좁은 검토**(D100 ③).
5. M3b 최종 실행 등록: 스냅샷·manifest sha, source commit, config/addendum sha, 부트스트랩 seed(`seeds.base` 파생 규칙), 2026 비용 예산(미측정)·hang guard, `registered=true`로 전환.
6. **사용자 서명**(최종 등록과 1회 실행 승인). 어느 게이트도 앞 게이트 통과만으로 자동으로 열리지 않는다.

> **개정(2026-10-01, D122/D123):** 후보 식별자 pin을 `c13cc98994ae9eb08a8f939a583ebe712312594ac25c05314a9ca790bd46f73c`로 바꿨다. 런타임 v3 소스 변경으로 식별자(소스 closure 포함)를 S2→S6에서 다시 인증했고, τ 표·V2 gap은 이전과 같다. ≤2025 L1-R 리허설: Δ [−0.00065, +0.00054], NO_EVIDENCE_OF_IMPROVEMENT. 2026 실행은 저메모리 원장(D123 차단 요인) 해결 뒤다.

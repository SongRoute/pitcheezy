# 정책 런타임 구현 — G0 5member 어댑터·TRAIN BC 아티팩트·미지원 분모 원장 (COOP-016)

2026-09-28, 기준 `f83caea`, D88 역할(Opus 메인·Sol 서브·Astra 최소 검토). 상위 계약은 [MLB-2026-POLICY-PREPARATION-v1](../contracts/MLB-2026-POLICY-PREPARATION-v1.md) §2·§5·§7(D89)이며 그 상태·식·ARM 선택을 바꾸지 않는다.

> **구현 ≠ fit·동결·평가.** 실제 BC/지원 표/G0/WE/τ identity는 여전히 `null`, `policy_frozen=false`, ARM-A 보류, ARM-B 비활성(자동 활성 없음), G0는 예측기다. 이번 작업은 실데이터·배열 헤더·모델 payload·2026을 열거나 해시하지 않았고 실데이터 fit/추론/OPE·수집은 0이다. 검사는 전부 temp 디렉터리의 합성 BC·가짜 체크포인트·상수 가짜 member로 했다. **실제 아티팩트 검증은 주장하지 않는다.**

**통합 검토(D91):** Opus 작성 커밋 `1ecdbb4`를 Sol이 독립 구현 검토하고 Astra가 핵심 과학 검토 1회로 **구현 범위 PASS**를 판정했다. Root 통합 후 신규15검사도 통과했다. [검토 기록](../reviews/COOP-016-policy-runtime-implementation-2026-09-28.md)을 따른다. 완전한 실제 정책 식별자·자료 출처 인증·실행 승인의 통과가 아니다.

## 1. 무엇을 만들었나 (신규 코드/검사 3파일, 기존 실행 코드 수정 0)

| 파일 | 내용 |
|---|---|
| `experiments/pitchmdp/pitchmdp/policy_artifacts.py` | TRAIN BC JSON 직렬화·로드, 개입 지원 표, 5member G0 manifest 검증·factory |
| `experiments/pitchmdp/pitchmdp/policy_runtime.py` | 요청 평가(로깅 법칙·마스크 기준·후보), 해시 체인 JSONL 원장, `build_runtime` 진입점 |
| `experiments/pitchmdp/tests/test_policy_runtime.py` | 합성 15검사(아티팩트·G0·원장·end-to-end) |

진입 경로 하나: `build_runtime(bc_path, bc_sha, support_path, support_sha, ledger_path, *, g0=…, pool, terminal, cutoff, budget, we_identity, tau, samples, pitch_cap, seed)` → `PolicyRuntime.submit(DecisionRequest)` → `summary()`. g0 없이 부르면 로깅 법칙·기준만 평가한다. g0를 주면 후보 P3(`RolloutImprovement.policy('P3')` over `JointSimulator(pool, FrozenG0Ensemble, terminal)`)를 만들며 **τ·MC·WE 설정은 기본값이 없다**(미등록).

## 2. G0 5member 어댑터

- 기존 3member `FrozenGEnsemble`은 그대로 두고 하위 클래스 `FrozenG0Ensemble`을 추가했다. `__call__`(= `calibrated_conditional`)을 그대로 재사용한다: member별 May 온도로 softmax → 5개 평균 → **공통 June 앙상블 가중치**로 frequency와 혼합. 이는 `matrix_g0_whole_metrics.frozen_five_predictions`의 primary 식(`w·mean(member) + (1−w)·frequency`)과 같다. member별 June 가중치는 단일 member 진단용이라 앙상블에 쓰지 않는다(검사로 고정).
- `validate_g0_manifest`: protocol `g0_research_frozen_v1`, research_only/미승격, member 키 정확히 `0..4` 순서, source arm g/g/g/c1/c1, 필드 집합, sha 형식, 체크포인트 pin = member `model_sha256`, 5개 서로 다름, 온도 ∈[.5,2.5](기존 `frozen_weights` 범위), 가중치 ∈[0,1].
- `load_g0_ensemble(bundle, inputs, paths, load_member, load_frequency, frequency_role)`: 경로는 **호출자가 역할별로 명시**(기본·하드코딩 없음). 각 파일을 bundle pin과 해시 비교한 뒤에만 읽는다. `p11_frozen_calibration`에서 앙상블 가중치를 읽고 May 온도·seed 가중치가 manifest와 정확히 같은지, `refit_on_whole_mlb=false`인지 검사한다. `p4_preparation`에서 baseline 온도와 token 어휘를 읽어 `PolicyInputs.types`와 대조한다.
- member는 실제 G0 래퍼 `matrix_sharing.SharingPredictor`(cell `G0-global`)여야 하고, 그 안의 네트워크 `seed`가 manifest 슬롯과 같고 10클래스여야 한다. 래퍼가 맥락의 마지막 7개 routing/cluster 열을 떼고 전역 네트워크를 부르는지 합성 검사로 확인했다(20→13열).
- 실행 시 검사: member 출력 `[N,10]` 유한, frequency 원시 행은 frozen clip/log/온도 변환 **전에** 유한·[0,1]·합 1 검사(음수·NaN을 clip으로 고치지 않음), 출력 shape. member 누락·순서 뒤바뀜·중복·체크포인트/보정 파일 변조·어휘 불일치는 모두 `IntegrityError`.

**메타데이터 공백(추정으로 채우지 않음).** (1) G0 bundle은 앙상블 가중치·baseline 온도를 인라인하지 않는다. Sol 조사가 pin 파일 텍스트에서 인용한 값(0.7467754577062221, 1.030298001849738)은 이번에 내가 열어 확인하지 않았고, factory는 값을 코드에 넣지 않고 pin 파일에서 읽는다. (2) 새 상태용 **frequency 예측기 객체**가 bundle의 어느 pin 파일인지 명시돼 있지 않다(P8은 `inputs.pkl`의 baseline을 썼다). 그래서 `frequency_role`을 필수 인자로 두고 해당 파일 해시만 강제한다. (3) 실제 member 로더는 주입식이다. 실제 결합은 `run_ml_g0_whole.load_member`의 검사(체크포인트 sha·kind/seed/width/클래스/network/파라미터 수/device)를 유지해야 하며, scripts 모듈이라 패키지에서 import하지 않았다. (4) 실제 G0용 `PolicyInputs`(맥락 encoder·400 delivery pool·safe rows) 구성은 호출자 몫이다. 이 넷은 실제 materialization 등록 때 닫아야 한다.

## 3. TRAIN BC 아티팩트

- 형식: canonical JSON(`sort_keys`, 구분자 고정, NaN 금지) envelope `{schema: pitcheezy.train_bc.v1, content_sha256, payload}`. payload = 클래스·`prior_strength`·`minimum_action_count`·정렬 어휘+`vocabulary_sha256`(D89 식)·league/pitcher/cell 개수표(타입 있는 레코드, 문자열화한 tuple 아님)·provenance·미지 투수 규칙.
- identity: `bc_sha256 = canonical_hash(payload)`, 파일 `file_sha256`. 레코드 입력 순서가 달라도 같은 바이트(검사).
- 저장: 같은 디렉터리 temp → fsync → `os.link`(대상이 있으면 실패) → temp 삭제. 기존 동결 파일을 덮어쓰지 않고 부분 파일을 남기지 않는다.
- 로드: **sha pin 필수**. 파일 해시 → JSON(잘림 거부) → schema/version → canonical 바이트 일치 → content 해시 → 구조 검사(양의 int 개수만, bool/float 거부, 어휘 밖 action, 중복 pitcher/cell, 합법 카운트, league=Σpitcher=Σcell, 어휘=league action, 정렬). content 해시까지 다시 맞춘 변조도 구조 검사가 잡는다(검사).
- provenance: TRAIN 시작/끝이 ISO 날짜이고 `fit_bc` 창 2023-05-15..2025-04-30 안, `source_ids`(sha), `config_sha256`, 40자리 `code_commit`, `data_version`, `train_rows`. `dates` 필드가 `observed_by_export_train_bc`(향후 `export_train_bc`가 기존 `fit_bc` 가드를 통과한 행의 min/max를 기록) 또는 `declared_unverified`(호출자 선언)를 구분한다. **로드는 원 TRAIN 행을 재검증하지 못하며, 관측 날짜도 모집단 완전성·선택을 입증하지 않는다.**
- 로깅 법칙 = 전체 어휘 `CategoricalBC.probabilities`(재정규화·floor 없음). 기준 = `MaskedReference`(= 기존 `SupportedBC` 산술, 단 live pool 대신 pin된 지원 표). 합성 입력에서 `SupportedBC`와 1e-15 이내 일치(검사). 지원 표(`pitcheezy.intervention_support.v1`)는 BC sha·어휘 sha에 묶이고 `(pitcher, batter_side)` 키. 같은 키에 다른 마스크가 오면(투수 손 차이 등) 생성 거부. 마스크는 어휘 길이의 bool 벡터만 받고(NaN·실수 강제변환 없음), 알려진 투수의 TRAIN BC 지원 안이어야 한다. `train_rows`는 개수표 합과 같아야 한다.
- `PolicyRuntime`은 받은 BC·지원 표를 pin으로 다시 해시해 확인한 뒤 **자기 사본**(지원 마스크 read-only)을 쓴다. 호출자가 나중에 원래 객체를 바꿔도 같은 런타임 sha 아래 법칙이 바뀌지 않는다(검사).

## 4. 미지원 요청 분모 원장

| 상황 | 요청 상태 | PA 상태 |
|---|---|---|
| 평가됨, 로그 행동이 마스크 안 | `SUPPORTED`, ρ 기록 | 모두 평가되면 `SUPPORTED` |
| 로그 행동이 어휘 안·마스크 밖, π̂_b>0 | `OUTSIDE_POLICY_SUPPORT`, ρ=0(정당) | 평가된 것으로 셈 |
| TRAIN 이력 없는 투수(league fallback) | `UNSUPPORTED_UNKNOWN_PITCHER` | 첫 결정이면 그 상태, 이후면 `UNSUPPORTED_MID_PA` |
| BC 지원 또는 BC∩개입 지원이 빔 | `UNSUPPORTED_EMPTY_SUPPORT` | 〃 |
| 로그 행동의 추정 π̂_b=0 | `UNSUPPORTED_LOGGING_POSITIVITY` | 〃 |
| 같은 PA에서 앞서 거절된 뒤의 결정 | `UNSUPPORTED_MID_PA`(평가 안 함, 결과 `null`) | 〃 |
| 어휘 밖 라벨, 잘못된 확률/마스크 밖 질량, 다른 런타임 pin, 이력 불일치 | `FAILED_INTEGRITY` 기록 후 예외, 실행 중단 | `FAILED_INTEGRITY` |
| 그 밖의 예외(예: `BudgetExceeded`) | `FAILED_RUNTIME` 기록 후 원래 예외 | `FAILED_RUNTIME` |

- **기록 후 반환/예외.** 모든 요청 ID는 결정 행 하나를 남긴다. 거절·실패도 분모에 남고 관측 행동으로 대체하거나 ρ=1로 두지 않는다. 결과(outcome)로 행을 거르지 않는다. `summary()`는 원장만으로 요청·PA 분모를 세고 `population_value`는 항상 `None`이다.
- **이력의 국소 일관성.** `decision_index`는 그 PA의 다음 번호여야 하고 `len(history)`와 같아야 하며, 이력 마지막 구종이 직전 요청의 로그 행동과 같아야 한다.
- **중복·재시도.** 같은 ID·같은 내용(fingerprint)은 저장된 행을 그대로 돌려주고 새 행을 쓰지 않는다(재시작 후에도). 같은 ID·다른 내용은 `conflict` 감사 행을 남기고(분모 제외) 실행을 중단한다. ID가 문자열이 아니거나 비어 있는 요청은 분모 키가 없으므로 `malformed` 감사 행(분모 제외)을 남기고 중단한다. 중단 상태는 재개 시 원장에서 복원된다.
- **영속성.** JSONL, 행마다 `seq`·이전 행 해시·자기 해시. 재개 시 전체 재생: 잘린 마지막 행, 변조·재정렬·비canonical 행, 다른 pin의 header는 거부한다. 행 경계에서 뒤쪽 행을 통째로 지우는 롤백은 체인만으로 안 보이므로 `summary()`가 `ledger_head_sha256`·행 수를 내고, 호출자가 저장해 둔 head와 비교해야 잡힌다(검사). 단일 writer 가정(`ponytail:` 주석).
- 원장 행은 투구 전 맥락(context_key, 투수, 타자 손, 카운트, 이전 구종 목록), 로그 라벨, pin(런타임 sha = BC·지원 표·후보·예측기 identity의 해시), 확률 벡터·ρ만 남긴다. 물리 벡터·결과값은 복사하지 않는다.
- **D89 대비 추가/미정.** `FAILED_RUNTIME`은 D89 표에 없던 치명 상태로, 감사 행을 잃지 않기 위해 추가했다. `UNSUPPORTED_MID_PA` 이후 PA를 추정량에서 어떻게 다룰지(D89 "최종안 null")와 불완전 PA·PA 종료 보상 기록은 이번 원장이 정하지 않는다. 원장은 분모만 보존한다. 2026 원 구종 코드 → 어휘 대응 표도 미구현이며, 지금은 어휘에 정확히 있는 라벨만 받는다.

## 5. 검사 (실제 실행)

`PYTHONPATH=src:experiments/pitchmdp /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest …`

- 신규 `test_policy_runtime.py` 최종 **15 passed**. 실패 이력 보존: 첫 실행 2 failed(가짜 체크포인트를 제자리에서 덮어쓴 fixture 누수, E2E 가짜 member가 공통 난수에서 Q를 못 바꿈) → 테스트 fixture만 고쳐 12 passed → 치명 상태·head pin·빈 BC 지원 보강 13 passed → Sol 조기 검토 반영 후 3 failed(새 `train_rows`·지원 부분집합 검사가 낡은 fixture를 정당하게 거부) → fixture 수정 15 passed.
- 변이 8종을 임시로 넣어 모두 실패 확인 후 원복: 로깅 법칙을 마스크로 재정규화, MID_PA sticky 제거, 로깅 positivity 검사 제거, 앙상블 가중치 교체, 5→3 member 절단, cell/pitcher 합 검사 제거, member seed 슬롯 검사 제거, conflict 후 미중단. 마지막 변이는 처음에 살아남아(재개 후만 확인) 실행 중 인스턴스 검사를 추가했다.
- 기존 회귀 포함: 신규 + `tests/test_2026_policy_contract.py` + `test_matrix_policy.py` + `test_rollout_policy.py` = **54 passed**(신규 15 포함, 중복 합산 없음). `scripts/check_2026_policy_contract.py` `all_pass: true`.
- Sol 조기 검토(`sol-review-artifacts.md` S1–S3, `sol-review-runtime.md` R1–R3)를 이번 시도 안에서 모두 반영했다: S1 `train_rows` 대조, S2 마스크 dtype·BC 지원 부분집합, S3 `SharingPredictor`·seed 슬롯, R1 conflict 중단·복원, R2 malformed/콜백 예외 감사 행, R3 사본 소유. R 질문(미지원 투수 교체 결정의 요청 라벨)은 D89 표대로 요청=실제 사유, PA=`UNSUPPORTED_MID_PA`로 두었다.
- 실행하지 않은 것: 실데이터/실제 모델 테스트, `test_b_recommendation.py`, `freeze_g0_research.py`, 전체 테스트 묶음.

## 6. 남은 인과 경계

런타임은 결정별 확률·비율·분모만 만든다. 후보의 시뮬레이터 구성요소 중 G0와 `we_identity` 문자열만 identity에 들어가며, delivery pool·terminal·cutoff 함수의 identity는 호출자 책임이고 등록되지 않았다. 식별 가정 (i)–(v)(D89 §5)는 하나도 입증되지 않았다. 로그 라벨 일관성, 2026 교환 가능성, 참 positivity, 연도 간 로깅 법칙 이동, WE continuation 가치는 이 코드와 무관하게 열려 있다. ρ^ref≠1은 마스크가 로깅 질량을 다 덮지 않을 때 정상이다(검사: ρ^ref = 1/π̂_b(M|H)). 다음은 실제 TRAIN BC/지원 표/G0 입력의 materialization 등록(위 공백 4개 포함), MID_PA·불완전 PA 규칙 확정, ≤2025 검증의 자료핀·비용 등록이다.


## 7. 식별자·분모·시간 정보의 정확한 범위

- `runtime_sha`는 현재 기록한 구성요소를 묶은 실행 식별자이며 **완전한 동결 정책 식별자가 아니다**. 임의의 다른 pool/terminal/cutoff 또는 전처리/member·frequency 로더를 연결해도 같은 P3 행동 법칙임을 보증하지 않는다. 실제 정책 동결 전에는 이 구성요소들의 descriptor·소스/아티팩트 pin을 추가하고 실제로 연결된 구현·아티팩트와 일치하는지 검증해야 한다. 여기서는 실제 정책을 동결했다고 주장하지 않는다.
- 원장의 분모는 **이 런타임에 제출된** 유효 키 요청과 PA다. 재시도는 중복 집계하지 않으며 키 없는 malformed·충돌 이벤트는 별도 감사 기록이다. 제출되지 않은 경기/투구까지 포함한 전체 MLB 커버리지나 모집단 완전성을 입증하지 않는다.
- 요청 fingerprint는 이전 구종뿐 아니라 이전 카운트·물리·결과를 포함하므로 같은 ID의 내용 변경을 검출한다. 그러나 순서·길이·직전 구종 확인은 전달된 이력의 **국소 일관성**만 검사한다. 실제로 그 시점 전에 관측 가능한 정보였는지, 올바른 원자료에서 왔는지는 인증하지 못한다. 실제 자료 연결 시 별도 출처·시간 누수 검증이 필요하다.

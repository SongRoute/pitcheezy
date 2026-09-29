# 정책 구성요소 실제 연결·식별자·≤2025 검증 준비 v1 — 결정됨(D93), 미등록

2026-09-29, COOP-017(식별자, D92) → COOP-018(검증 코드·결정, D93), 기준 `713c7ea`. 기계 필드는 [configs/ML-POLICY-MATERIALIZATION-v1.json](../../configs/ML-POLICY-MATERIALIZATION-v1.json)(D93부터 **이 config가 권위**), 근거는 [D92 보고서](../reports/ML-policy-materialization-prep-2026-09-29.md), [D93 보고서](../reports/ML-policy-validation-code-2026-09-29.md), [결정 선택지](../reviews/COOP-018-decision-options-2026-09-29.md), [코드 검토](../reviews/COOP-018-code-review-2026-09-29.md). 상위 계약 [MLB-2026-POLICY-PREPARATION-v1](MLB-2026-POLICY-PREPARATION-v1.md)(D89)의 DR 식·ARM 선택은 바꾸지 않는다. D93은 그 config의 `mid_pa_pitcher_change_rule`·`profile_as_of`를 proposal 상태로 채웠다(다음 Astra 검토 항목).

> **상태: `PROPOSAL_UNREGISTERED`, `policy_frozen=false`, `execution.enabled=false`.** 사용자가 2026-09-29 D-1~D-10과 누락 결정 M-1~M-13을 추천안대로, D-11을 D안(측정 먼저)으로 정했다(D93). 실행기·요청 생성기·추정기·τ 선정·V2/V3 코드는 구현되어 합성 검사를 통과했다. 그러나 등록, 코드 검토 게이트(S0–S2), 독립 검토 게이트(S3 이후), source commit, 아래 `null` 필드가 남아 있어 **아무 stage도 실행할 수 없다**. 실제 TRAIN BC 생성·G0 가중치 로드·자료 로드·추론·OPE·2026 열람은 0이다. 검토는 같은 모델(Opus 5.5)의 적대적 검토이며 독립 검토가 아니다.

## 1. 구성요소 출처 (조사 결과)

"pin"은 G0 동결 번들 `configs/G0-RESEARCH-FROZEN-v1.json`의 파일 SHA256이다. 값 인용은 모두 해당 pin과 SHA가 일치하는 JSON 메타데이터에서 이번에 읽었다(payload 제외).

| 구성요소 | 생성 코드 | 필요한 입력 | 기존 pin·메타데이터 | 누락·미확인 | 검증 방법 |
|---|---|---|---|---|---|
| TRAIN BC (로깅 법칙 π̂_b, 기준의 바탕) | `matrix_policy.fit_bc` → `policy_artifacts.export_train_bc`(D91) | 정규시즌 frame(`split=='train'`), BC용 직전 구종 이력 | 독립 artifact 없음. P8은 run 안에서 적합: 1,252,824구, 18구종(`EXP-P8-001/preparation.json`) | 선택 규칙(§3 D-1), 실제 artifact·해시 | `load_train_bc`(sha·canonical·구조), export가 관측 날짜 기록 |
| 개입 지원 표 M | `PolicyInputs.support`(BC ∩ token 어휘 ∩ 12카운트 전부의 구종별 TRAIN pool) → `save_support_table` | TRAIN BC, 연결된 pool, 단일 손 투수 template(D-8) | 없음 | 실제 표 | `load_support_table` + **요청마다 맥락 행 일치와 pool 재계산 일치 확인(신규)** |
| G0 member 5개 | `run_ml_g0_whole.load_member`(`MatrixModel.load`+`SharingPredictor` G0-global), `load_g0_ensemble`(D91) | 체크포인트 5, `p11_preparation` member 기록, `p11_registered_config` | member `model_sha256` 5개; `p11_preparation` `aa831981…`; config `b6565f16…`: flatten_mlp, width 128, context 52, 141,834 parameters, device `mps` | 실제 로드 미수행, MPS 필요 | 기록↔번들, 로드된 네트워크↔기록(network·parameter·device·temperature), content digest, §5 S2 probe |
| May temperature·June 앙상블 가중치 | C1 보고서 → `matrix_g0_whole.frozen_weights` | `p11_frozen_calibration` | `e3b96ef4…`: weight 0.7467754577062221, `refit_on_whole_mlb=false`, temperature 5개 = 번들 | — | D91 factory가 파일 해시 뒤 읽고 번들과 대조 |
| frequency 객체 | `run_ml_matrix._fit_aux`: `ContextFrequencyBaseline(HierarchicalFrequencyBaseline())`, eligible D100 TRAIN 1,252,824구 적합, `aux.pkl['baseline']` | `p4_auxiliary`, `p4_preparation.baseline_temperature` | aux `87ad95e6…` = `p4_preparation.artifact_hashes['aux.pkl']`; T = 1.030298001849738(May 2,603구); 보관 예측 `bbf4f02a…` | **D91 공백 해소:** 역할 = `p4_auxiliary`의 `baseline` 키(P8의 `inputs.pkl['baseline']`도 같은 EXP-P4-001 aux에서 옴). bytes 미검증 | 등록 class·보관 report 일치, S2에서 `mlb_dev_raw` 재현 |
| 전처리(`PolicyInputs`) | `PolicyInputs(rows, SharingContext(aux['context'], clusters), aux['delivery'], 어휘, aux sha, bc)`, `safe_rows`, `state_from_row` | aux의 `context`·`normalizer`, `p4_preparation`의 clusters·tokens | 어휘 18개, history 5, token 38채널; clusters representation sha 기록; normalizer는 TRAIN 전체 1,386,362행 적합 | 요청 생성기(자료→H_t)는 미구현(§4) | report·representation·token 배치 대조, **합성 probe로 평가 경로와 일치(신규)** |
| delivery pool | `JointDelivery.fit`(eligible D100 TRAIN, draws 400, seed 42) | `aux.pkl['delivery']` | pool 23,728개, tier 4단계 | league fallback pool은 정책이 쓰지 않음(행동 미지원) | class·report·draws·tiers·pool 수·배열 모양, 읽기 전용, wiring 확인 |
| WE terminal/cutoff | `FrozenWE(inputs, game_values)`: `game.terminal_values`, `WinExpectancy.predict_defense` | C0 계약, bundle manifest, source lineage, `game_values.pkl` | `model-v1.json` `d0f1d452…`, manifest `43ece920…`, lineage `6d6f777b…`, game_values `aa6c4e48…`; 현재 WE 소스 3파일 = lineage(이번 확인) | game_values bytes 미재검증; WE는 2025 DEV Brier gate를 원래 빌드에서 통과(노출, P8 기록) | 계약→manifest→lineage→bytes→class 순서, wiring 확인 |
| 탐색 설정 τ·MC·cap·seed·budget | `RolloutImprovement`, `RowBudget` | 등록 필요 | 참고: P8 τ grid .001–.03, 선택 τ=.003(June 2025, G2 세계), search 2/cap 8, 평가 4/cap 8, seed 701/1701, stage당 460,000행·1,500초 | **G0용 전부 `null`**(예측기 변경 → τ 재선정 필요, D89 §7.3) | identity의 `search`에 포함, 없으면 후보 생성 거부 |
| 코드·환경 | — | — | 현재 소스가 G0 실행 소스 30파일과 바이트 동일(이번 확인). G0 환경 python 3.12.14, torch 2.14.0, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1 | 실제 결합 시 PYTHONPATH/실행 파일 고정 | 연결된 객체에서 도달하는 저장소 class 파일 SHA + 런타임 경로 + 버전 |

## 2. 완전한 정책 식별자와 실제 연결 검증 (구현·합성 검사 완료)

구현: `experiments/pitchmdp/pitchmdp/policy_identity.py`(신규), `policy_runtime.py`(수정). 계약 이름 `ML-POLICY-IDENTITY-v1`.

**진입 경로.** `bind_components(bundle_path, bundle_sha256, paths, bc_artifact=…, context_rows=…, member_loader=…, classes=…, we_contract_sha256=…)` → `BoundComponents` → `build_runtime(…, components=…, budget, tau, samples, pitch_cap, seed, expected_identity_sha256=None)`. G0 번들 JSON도 파일 pin으로만 읽는다. 후보 모드에서 pool·예측기·terminal·cutoff는 **이 묶음에서만** 가져온다. D91의 느슨한 `pool=`/`terminal=`/`cutoff=`/`we_identity=` 인자는 제거했다(호출자는 테스트뿐이었다). 직접 만든 후보 `PolicyRuntime`은 `verify_components()`에서 거부된다.

**식별자 구성** (`components.identity` + `search` + 지원 표 SHA → `sha256`):
`g0_bundle_file_sha256`, `bc`(BC·어휘 sha), `predictor`(D91 G0 identity), `members`(loader 함수 qualname·파일·SHA, 기록/config SHA, seed별 network SHA·parameter 수·device), `frequency`(역할·키·class·report SHA·temperature·변환식), `preprocessing`(PolicyInputs class, SAFE_COLUMNS, encoder/base class, context report·clusters·tokens·어휘·normalizer report SHA, aux·preparation SHA), `delivery_pool`(class·400·tiers·pool 수·report SHA·source hash·fallback 금지 규칙), `we`(계약·manifest·lineage·game_values SHA, class, terminal/cutoff 정의), `sources`(연결된 객체의 class·함수 모듈과 런타임 경로 모듈, 그리고 그 모듈들이 모듈 수준에서 import하는 저장소 모듈의 closure SHA), `environment`(python·numpy·scipy·pandas·torch 버전, OS·arch), `search`(P3, τ, samples, pitch_cap, seed). budget은 확률이 아니라 실패만 바꾸므로 넣지 않는다. 맥락 행 값은 요청 입력이므로 식별자가 아니라 **요청 fingerprint**에 들어간다(아래 3).

**검증 층.**
1. 파일 pin: 번들 JSON부터 모든 파일을 읽거나 unpickle하기 **전에** SHA 대조(pickle은 코드 실행이므로 pin이 신뢰 경계). loader가 체크포인트를 다시 여는 틈은 로드 뒤 재해시로 닫는다. factory가 해시 뒤 다시 읽는 보정 JSON은 여기서 검증한 bytes의 값과 대조한다.
2. 선언↔객체: 등록 class와 실제 class, 그 class가 **저장소 코드**일 것, 객체 report와 G0 preparation에 보관된 report의 정확한 일치, delivery draws·tiers·pool 수·배열 모양, member 기록과 번들, loader가 돌려준 네트워크와 기록(network·parameter·device·temperature).
3. 산출물·요청 연결: clusters와 representation SHA, P11과 P4 clusters, `inputs.bc`와 pin된 BC, WE 계약→manifest→lineage→bytes. 요청마다 (a) 맥락 행의 투수·타자면이 요청과 같은지(다르면 `PolicyInputs.support`가 빈 지원으로 삼키고 캐시를 오염시킴), (b) 지원 표 마스크 = 연결된 pool로 다시 계산한 지원인지 확인한다. 둘 다 빈 마스크 거절보다 먼저다. (c) 맥락 행 SHA를 요청 fingerprint와 원장 행 `context_sha256`에 넣어, 같은 식별자로 다른 행 값을 다시 연결한 재시도는 충돌로 중단된다.
4. 코드·환경: 위 `sources`와 `environment`.
5. 연결 뒤: `runtime.verify_components()`는 식별자 해시, 소스 재해시, 객체 연결(`g0.inputs`·frequency 객체·`inputs.delivery`·encoder가 검증한 그 객체인지), in-memory digest(aux, WE, clusters, class 수준 TIERS, 어휘·type_map, 맥락 행과 인코딩, BC, member 가중치, 보정값, WE 표), 시뮬레이터의 pool/예측기/terminal/cutoff wiring을 확인한다. **실행기는 stage 결과 봉인 전 반드시 호출**한다. pool 배열은 읽기 전용이다.
6. 행동 probe: `integrated_predictions`(행동의 정확한 400 draw pool 평균)가 평가 경로의 G0 primary와 같아야 한다. 합성에서 **`bind_components`로 연결한 정책 경로**가 실제 평가 코드(`JointDelivery`·`MatrixHistoryStore`·`SharingContext`·`predict_streamed`·`temperature_predictions`)와 12행에서 최대 차 4.4e-16으로 일치했다. 평가가 league fallback pool을 쓴 3행은 정책 경로가 거절했다. pool seed·member temperature 순서·frequency temperature·clusters를 바꾸면 6.2e-3~2.4e-2 차이로 검출됐다. 실제 probe는 §5 S2.

**상태 변경 기록(D92).** 후보 모드에서 지원 표에 키가 없거나 빈 마스크인데 연결된 pool은 지원하는 요청은 D89 §2의 `UNSUPPORTED_EMPTY_SUPPORT` 대신 `FAILED_INTEGRITY`(중단)다. 두 pin된 산출물이 서로 맞지 않는다는 뜻이기 때문이다. 맥락 행과 요청의 투수·타자면이 다른 경우도 `FAILED_INTEGRITY`다. 로깅 전용 런타임의 상태는 바뀌지 않았다.

**상태 변경 기록(D93, 런타임 계약 `ML-POLICY-RUNTIME-v2`).** 원장 pin에 상태 목록과 손 등록부 SHA가 들어가 v1 원장과 섞이지 않는다. (1) 무행동을 원인별로 나눈다: 등록 무투구 description(현재 `automatic_ball/strike`)은 `UNSUPPORTED_NO_LOGGED_ACTION`(이력 `<NO_PITCH>`), 구종 라벨만 없는 실제 투구는 새 `UNSUPPORTED_MISSING_ACTION_LABEL`(이력 `<MISSING_LABEL>`). 둘 다 sticky. (2) 요청에 투구 손(`pitcher_hand`)이 들어가고, TRAIN 단일 손 등록부(모든 TRAIN 행이 같은 L/R이고 결측 없음, 아니면 AMBIGUOUS)와 다르거나 결측·AMBIGUOUS면 새 `UNSUPPORTED_PITCHER_HAND`(sticky, 두 모드 모두). (3) 후보 모드에서 맥락 행이 없거나 투수·타자면·손이 요청과 다르면 **어떤 자료 거절보다 먼저** `FAILED_INTEGRITY`. (4) `LOGGING_POSITIVITY`는 후보 탐색 전에 판정(거절된 요청이 예산을 쓰지 않음). (5) 직전 실제 투구의 결과가 10-class로 매핑되지 않으면 카운트 경로를 확인할 수 없으므로 `UNSUPPORTED_INCONSISTENT_HISTORY`(이전: 통과). (6) stage 중단은 `aborted` 원장 행을 남겨 HALTED로 읽힌다. (7) 후보 기록에 DR용 Q(등록되면 독립 평가 seed, M-7), 계획 Q, 계획 Q의 짝 차이 표준오차가 들어간다.

**한계.** 식별자는 과학적 타당성·인과 식별을 입증하지 않는다. digest는 같은 프로세스 비교이며, 파생 캐시(pool·지원·frequency 캐시)는 digest하지 않는다(`ponytail:` 주석). 4,096개를 넘는 컨테이너는 앞 64개만 표본으로 보고 class를 찾는다(자료 컨테이너 가정). Python 안에서 `BoundComponents`를 직접 만들어 식별자를 꾸미는 것은 막지 못한다. 등록된 실행기만 `bind_components`를 쓰는 절차가 경계다. 실제 객체가 추론 중 내부 상태를 바꾸면 digest가 거짓 실패를 낼 수 있으며 이는 S2에서 처음 확인된다(미측정). 요청 생성기(원자료→H_t) 코드는 정책 식별자가 아니라 실행 등록의 stage manifest에 pin한다.

## 3. TRAIN BC·손 등록부·지원 표 (결정 D-1·D-8, 미실행)

**자료.** 처리 캐시 `processed_sha256 9f65d1db…`(2,145,111행) → `run_ml_benchmark.regular_frame`(game_type R, `add_batter_style_history`, `assign_fold(…, 2025)`) → 실행기에서 `(game_pk, at_bat_number, pitch_number)`로 정렬.

**BC-P (결정: 등록 π̂_b이자 기준 바탕).** `policy_requests.bc_population_mask`: `split=='train'` & 로그 라벨이 sentinel이 아님(등록 무투구 description이면 구종이 있어도 제외, 구종 결측·빈 문자열 제외) & 볼 0–3·스트라이크 0–2. 결과 라벨·`supported_pa`·plate 좌표를 쓰지 않는다(결정 뒤 정보로 적합 모집단을 고르지 않는다). 식별 가정: H가 주어졌을 때 라벨 결측은 구종과 무관(MAR). 행 수는 미측정(상한 1,386,362).

**BC-E (재현 게이트·기술 비교 전용).** pin된 eligible D100 TRAIN 키. 추정·선택·민감도에 쓰지 않는다. S1 fail-closed 게이트(결과 보기 전 config에 고정): BC-E 행 = 1,252,824, BC-E 행동 목록 = token 어휘, BC-E 키 ⊆ BC-P 키, 모든 (셀, 행동)에서 BC-P 수 ≥ BC-E 수, BC-P 행 수 ∈ [BC-E, TRAIN], BC-P 어휘 = token 어휘. 하나라도 어긋나면 stage 실패. BC-P/BC-E 차이표를 본 뒤 주 선택을 바꾸지 않는다.

**공통.** `CategoricalBC(prior_strength=20, minimum_action_count=1)`(config `bc_parameters`를 실행기가 읽음). BC 적합 이력은 `<UNKNOWN>`/`<START>`를 쓰고 요청은 `<NO_PITCH>`/`<MISSING_LABEL>`을 쓴다. 이 불일치는 두 sentinel이 sticky 거절이라 그 뒤 BC 질의가 없을 때만 무해하다. provenance `source_ids`에 모집단 규칙 해시와 정렬 키 해시를 넣는다.

**손 등록부(D-8).** TRAIN 정규시즌 전체 행에서 BC 투수마다 단일 손 또는 AMBIGUOUS(`pitcheezy.pitcher_hands.v1`, 런타임 pin). 비율 문턱은 두지 않는다.

**지원 표.** 단일 손 투수 × 타자면 {L, R}에 등록 손 template로 `PolicyInputs.support`를 저장한다. AMBIGUOUS 투수는 행이 없고 지원 표 조회 전에 거절된다. 런타임은 지원 표 투수 ⊆ 단일 손 투수를 확인한다.

**S1 보고.** π̂_b(M|H) TRAIN 분포, S1 게이트 결과, BC-P/BC-E 셀 차이 수, 손 등록부 수, 지원 안의 드문 코드(UN·PO·FA·EP) 수, TRAIN 투구 수 삼분위 경계(부분군용).

## 4. PA·시간 출처 규칙 (결정됨; 모두 구현·합성 검사)

| ID | 규칙 | 결정 |
|---|---|---|
| R1 분모 | L0 = 평가 창의 모든 정규시즌 PA(구조 결함으로 제출 못 한 PA 포함). L1 = 투구 전 시작 모집단 E0(첫 행 0-0, 알려진 투수, 등록 단일 손 일치, BC 지원·개입 마스크가 비지 않음; 첫 행의 투구 전 필드와 동결 TRAIN 산출물만 사용, 로그 행동·description·events·이후 행 불사용). `supported_pa`·eligibility는 사후 기술 층 | D-2 |
| R2 결정 시점 | PA의 모든 행을 `pitch_number` 순서로 요청(`request_id=game:ab:pitch`). 행 k의 구조 결함(순서 오류, pitch_number 공백, 첫 행이 pitch 1이 아님, 불법 카운트, 투수·타자면 결측)은 0..k-1을 제출하고 k에서 검열. **종료 사건이 있는 무투구 마지막 행**(등록 무투구 판정, 또는 구종·description이 모두 없는 주자 사건 행)은 결정이 아니라 상태 전이라 요청을 만들지 않는다. 단, pitch_number가 그 행까지 1..n으로 이어져야 한다(바로 앞 실제 투구가 빠졌으면 그 지점에서 `missing_row` 검열). 그런 행 하나뿐인 PA는 pitch 1일 때만 `NO_DECISION`(두 정책이 같으므로 Δ=0, L0에만) | M-8, D-6 비판 2 |
| R3 구종 코드 | (a) 코드 ∈ V 그대로. (b) 비결측 코드 ∉ V → FAILED_INTEGRITY, S0 게이트. (c) 등록 무투구 description(구종보다 우선) → `NO_LOGGED_ACTION`; 구종 결측·빈 문자열 → `MISSING_ACTION_LABEL`. 무투구 목록은 S0 구조 수로만 조정, S4·S6 값을 본 뒤에는 바꾸지 않음 | D-3 |
| R4 투구 전 정보 | SAFE_COLUMNS + 같은 PA의 엄격히 이전 **모든** 행(구종·정규화 물리·10-class 결과·카운트). 5행 제한 없음 | — |
| R4b 프로필 as-of | 투수 TRAIN 고정. 타자 style C0는 창 시작 스냅샷(`game_date < as_of_exclusive`; DEV 2025-07-01, May 2025-05-16, June 2025-06-01), 창 명단을 읽지 않고, 처음 보는 타자는 league 행(신뢰도 0). pin된 artifact, as-of 날짜 행에서 rolling 값과 비트 동일 검사, 평가 행이 as-of 이전이면 거부. S2 probe만 rolling | D-7 |
| R4c 출처 검사 | 요청 불변성(결정 뒤 필드·이후 행 변경), 카운트 경로, 스냅샷 명단 불변성 테스트 | — |
| R5 PA 종료·보상 | `structural-end-v1`: 마지막 행 `events`가 있고(코드 무관, `truncated_pa` 포함) 인접한 다음 PA 첫 행(같은 경기, at_bat+1, pitch 1, 0-0, 합법 전환: 같은 반이닝이면 아웃·점수 비감소, 바뀌면 아웃 0·주자 없음 또는 연장 2루 주자)이 관측되거나, 경기의 마지막 PA이고 **game_final_v1**(경기 마지막 행에 events가 있고, 사후 점수가 있고 동률이 아니며, 5회 이상 — MLB 정식 경기)일 때 종료. data.py `complete_game` 휴리스틱은 쓰지 않고 S0에서 불일치 수만 센다. r = 그 투구 전 상태의 초기 수비 팀 동결 C0 WE, 또는 최종 승패. 사후 점수 열이 없거나 선택 경기의 마지막 행 사후 점수가 결측이면 FAILED_INTEGRITY(조용한 검열 없음). 사후 점수 민감도에서 경기 중간 행 값이 없으면 공유 미지값 검열. `end_kind` 층(batter_event/non_batter_end/game_final), 점수 불일치 flag. 민감도: `r5-events-v1`(truncated→검열), flag→경계, 사후 점수 사용 | D-6 |
| R6 불완전 PA | 상태: COMPLETE, CENSORED(REFUSED / NO_TERMINAL / TERMINAL_VALUE_MISSING, 노드 k), EXCLUDED_PRE_START, UNSUBMITTABLE, NO_DECISION, FAILED(추정 거부). 검열 노드 값 c ∈ [0,1]을 D89 재귀 안에 두어 V_0 = base + (Π_{t<k} ρ_t)·c. NO_TERMINAL은 정책별 독립 미지값, TERMINAL_VALUE_MISSING은 두 정책이 공유. 유효 조건: 검열이 H_k로 정해지거나 공유 종료값이고, 후보 경로는 π̂_b가 정확. LOGGING_POSITIVITY 노드는 위반으로 따로 셈. 층: L0(전 PA 경계), **L1(E0 경계, 주)**, L2(E0∩완결, 처치 후 선택 조건부 평균, 판정 없음) | D-5 |
| R7 PA 중 교체 | 알려진 새 투수·타자 교체: 계속 평가(주). 미지 새 투수: `UNKNOWN_PITCHER` → `MID_PA`. 2차(기술): 첫 투수 교체 t* 뒤를 실제 진행으로 두는 추정량(t*는 원장 pitcher, manifest와 대조), 전체와 주 완결 집합 두 번 보고. S0 전에 정할 두 문턱(2차를 주로 올릴 미지 교체 비율, 경량판 문턱)은 `null` | D-4 |
| R8 이력 불일치 | 카운트 경로 단절 또는 직전 실제 투구 결과 매핑 불가 → `INCONSISTENT_HISTORY`(sticky). 원장 순번·길이·직전 구종 불일치는 FAILED_INTEGRITY | M-9 |

## 5. ≤2025 검증 자료·비용·실행 계획 (결정됨, 미등록·미실행)

**노출 이력.** 인증된 미노출 모집단은 0개다. 모두 **개발 검증**이며 독립 확인이 아니다.

| 집합 | 기간 | 과거 사용 | 라벨 |
|---|---|---|---|
| TRAIN | 2023-05-15..2025-04-30 | G0·빈도·pool·normalizer·BC·WE 적합 | `exposed_fit` |
| earlystop | 2025-05-01..05-15 | G0 조기 종료 | `exposed_model_selection` |
| May | 2025-05-16..05-31 | member·frequency temperature, P8 profile, **S3 비용 profile** | `exposed_calibration` |
| June | 2025-06 | G0 June 가중치, P8 τ, D81/D86, **S3b τ 설계(결과 비사용)** | `exposed_policy_tuning` |
| DEV | 2025-07-01..09-30 | G0·P8·D86 평가, **S4·S5·S6** | `exposed_development` |
| 2023-03-30..05-14 | `unused` 분할 | 미확인 | `unknown` |

**단계.** 단일 큐, 공유 `.heavy.lock`, 자료 로드는 잠금과 새 stage 디렉터리 **안에서**, 출력은 등록 루트 바로 아래 새 디렉터리, 실패 보존, 범위 축소 없는 새 attempt로만 재시도. 앞 stage 산출물은 부모 SHA 사슬의 append-only addendum(SHA pin만)으로만 다음 stage에 전달하며(M-2), 등록 입력은 **그 입력을 만드는 명령의 봉인된 stage manifest.json에 같은 SHA로 있어야** 한다(실패한 stage 산출물이나 다른 stage 파일은 등록 불가). S2의 `identity.json`(부품 SHA)은 `bind_identity`로 등록되고, S3 이후는 SHA가 다른 부품을 거부한다. 등록·결정·게이트·필수 필드·선행 조건(해당 분할의 S0 코드 게이트, S2 합격, S3 선택, 선택된 τ, S4 봉인, V2 합격)은 모두 **자료 로드 전**에 확인한다. S0–S2는 코드 검토 게이트, S3 이후는 독립 검토 게이트 뒤에만 열린다(M-3). M-1 표본은 분할 경계를 넘는 경기(예: 6월 중단·7월 재개)를 제외하고, 블록은 해당 분할의 행만 쓴다.

| stage | 명령 | 내용 | 등록 필요(`null`) |
|---|---|---|---|
| S0 | `census` | 코드·무행동 원인 교차표·PA 구조·카운트 경로·교체·손·BC-P 제외 수. 결과 인접 항목(종료 코드·truncated·점수 불일치·경기 끝)은 TRAIN만 | — |
| S1 | `materialize-bc` | §3 BC-P·BC-E·게이트·손 등록부·지원 표·보고 | — |
| S1b | `style-snapshot` | DEV·May·June 스냅샷 artifact | — |
| S2 | `bind-probe` | 실제 bind와 봉인 예측 재현(64행), **불합격이면 stage 실패** | — |
| S3 | `profile` | May 4개 PA 시작, 비용만. τ는 grid 첫 값을 "선택 아님" 표시. **등록 후보마다 실제 조건부 행 수를 측정**(행 수가 pitch_cap에 비례한다고 가정하지 않음)하며, 계획 탐색 행과 평가 seed 탐색 행을 따로 기록한다. 계획 행으로 선택 예산(= S3b 행 예산)에 드는 최대 설정을 고르되 잡음 규칙 최소 samples(등록값 3) 미만은 고르지 않는다(D-9a) | 후보, RowBudget, 선택 예산 |
| S3b | `tau-select` | June 경기 해시 표본을 후보 런타임에 한 번 제출, 보상·WE·events 종류 비열람. 원장 계획 Q로 τ별 π_cand 재계산, 두 정책 PA/경기 ESS·ESS 비·짝 차이 잡음비(samples ≥ 3 필요). 구조 결함으로 검열된 PA는 가중하지 않음. 기준 불합격 → ARM-B 평가 불가, 조건을 만족하는 가장 작은 τ, 없으면 P3 평가 불가. 경기 게이트 문턱 = 등록 ESS 게이트. 선택 τ로 다시 만든 런타임이 첫 2개 E0 PA의 계획 Q를 재현하는지 확인(D-9 note 5). **M-7 규칙을 기계적으로 적용**: S3에서 잰 평가 포함 행/결정 × S6 계획 결정 수가 S6 행 예산을 넘으면 계획 Q 재사용, 아니면 평가 seed. 그 결과와 게이트(문턱·grid·ESS 게이트)를 기록하고 최종 식별자 SHA에 넣는다(D-9b). S5/S6는 이 기록을 따르며 현재 게이트가 다르면 거부 | 경기 수, 문턱 5개, RowBudget |
| S4 | `v5-denominators` | DEV 전 PA 로깅 전용 분모. 봉인: 처리 PA = S0 census, 원장 결정 행 = 제출 요청 | — |
| S5 | `v2-world` | `absorb_off_mask` 세계(M 밖 로그 행동은 ρ=0 흡수 종료), M-1 경기 표본의 모든 E0 PA 시작, planning seed {0,1,2} × {V2, V3 빈도, V3 tempered α}, S3b 기록의 q̂ 규칙, 실행마다 RowBudget 하나를 탐색·세계가 공유(참값 rollout은 계획 Q만 사용). 실행별 후보 식별자를 기록하고 S6는 주 seed V2 실행의 식별자 = 최종 식별자를 요구. 합격: 합친 Δ 차이 95% 구간이 0 포함 & |차이| ≤ 허용치(로그·참값 rollout ≥ 2). 선택 사항: 등록한 `declared_hazard`와 반복 수로 D-5 경계·끝점 CI가 참값을 포함하는 비율을 기록(합격과 무관) | 경기 수, 로그 수, 참값 rollout, cap, 실행당 RowBudget, α grid, 허용치 |
| S6 | `dr-evaluate` | M-1 경기 해시 표본(월 비례, 경기 안 모든 PA), 스냅샷, 선택 τ, DR q̂ = 독립 평가 seed(M-7), L0/L1/L2, 경기 부트스트랩 B=10,000(무효 비율 등록, P8 최소 30경기·50 PA), ESS min(후보, 기준) 라벨(D87 100/30; ESS 정의 불가도 실패), 같은 원장 민감도 3개, 부분군(L0·L1 경계, E0 비율, ESS, 종료 잔차; BC-P 전용 투수 층 포함), 보상 열람 전 D-7 무라벨 진단(등록 부분표본의 G0 10-class TV·KL, membership TV, 미지 타자 PA 수), **독립 cand=ref 후보 callable의 짝 실행**(같은 요청, 완결 집합 동일·Δ 정확히 0이 아니면 S6 실패), V̂(ref) 옆 관측 평균 종료값. 기대 식별자 = S3b 기록의 최종 식별자. 봉인: 처리 PA = S0 census의 선택 경기 PA 수. 선행: V2 합격, S4 봉인 | 경기 수, RowBudget, 무효 비율 |

**비용 상한(D-11 D, 측정 먼저).** 벽시계 상한을 미리 정하지 않는다. S0–S4에는 안전 timeout(hang guard) 7,200초만 둔다(근거: CLAUDE.md D10 로컬 2시간 기준, 비용 예산 아님). 후보 rollout(S3, S3b, S5, S6)에는 결정적 RowBudget을 등록한다. S0–S4 실측 뒤에 벽시계 상한을 등록한다. 실패 뒤 범위를 줄이지 않는다.

**seed(M-11).** base 20260929에서 역할별 `SeedSequence([base, 역할 번호])`(평가·부트스트랩·V2·선택 salt·profile salt). planning 주 seed 0, V2 planning seed {0,1,2}.

**다중성·민감도(M-12).** 주 명세 = BC-P, 창 시작 스냅샷, 순수 DR, planning seed 0. 같은 원장 민감도는 기술 보고. ≤2025는 확인적 주장이 없다.

**식별자·환경·명령.** source commit·등록 config SHA 미정. 환경은 G0 번들 환경. `run_policy_validation.py --config … [--addendum …] --local-config experiments/pitchmdp/configs/local.json(sha eb364ca7…) --output <루트>/<새 attempt> <명령>`. 실행기는 등록·결정 값·게이트·필드·선행 조건을 자료 로드 전에 확인하고, **코드 경로(experiments/src/scripts)가 등록 source commit 이후 바뀌지 않았고 미커밋 변경이 없을 것**(등록 커밋은 그 위에 쌓일 수 있음; 로컬에서 대상이 바뀐 `runs` symlink는 코드 경로 밖)과 **--config·모든 addendum이 HEAD에 바이트 그대로 커밋**되어 있을 것, member loader SHA를 강제한다. hang guard는 원장 append를 끊지 않는다(append를 critical section으로 감싸 인터럽트를 완료 뒤로 연기, 어느 스레드가 신호를 받아도 성립). hang guard 예외는 결정 행으로 기록되지 않고 abort 행만 남긴다.

## 6. 결정 기록 (D93)

| ID | 결정 |
|---|---|
| D-1 | BC-P 단일 artifact, BC-E는 S1 재현 게이트·기술 비교 전용 |
| D-2 | L0 전 PA 경계 + 투구 전 시작 모집단 E0를 주 추정 대상, supported/eligible은 사후 기술 층 |
| D-3 | 원인 분리 거절(무투구 / 라벨 결측) + sticky + S0 구조 수 기반 목록 조정 규칙 |
| D-4 | 계속 평가(주) + 교체 기록·층 + 교체 뒤 자연 경과 2차 추정량(기술) |
| D-5 | E0 + 검열 노드 값을 [0,1] 미지값으로 둔 재귀 안 경계 |
| D-6 | 구조적 종료(structural-end-v1) + fail-closed 검사 + 민감도 |
| D-7 | 창 시작 스냅샷 + pin·경계 가드·미지 타자 league 행·2026 규칙 proposal |
| D-8 | TRAIN 단일 손 등록부 + 투구 전 거절(`PITCHER_HAND`), intersect 삭제 |
| D-9 | 2단계: S3 비용으로 탐색 설정, S3b June 결과 비열람 overlap·잡음 규칙으로 τ |
| D-10 | M 밖 로깅 행동은 ρ=0 흡수 종료(`absorb_off_mask`), 불변성 테스트 |
| D-11 | **D안(사용자 선택): 측정 먼저**, hang guard + RowBudget, 실측 뒤 상한 등록 |
| M-1~M-13 | 추천안 채택(config `decisions_extra`) |

남은 등록 조건: 수정 코드의 코드 검토 PASS(S0–S2), 독립 검토 PASS(S3 이후), source commit, 위 `null` 필드, 사용자 실행 승인. 2026 자료는 열람하지 않는다.

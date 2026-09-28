# 정책 구성요소 실제 연결·식별자·≤2025 검증 준비 v1 — 제안, 미등록

2026-09-29, COOP-017(Claude 직접 재개, D92), 기준 `713c7ea`. 기계 필드는 [configs/ML-POLICY-MATERIALIZATION-v1.json](../../configs/ML-POLICY-MATERIALIZATION-v1.json), 근거·검사 기록은 [보고서](../reports/ML-policy-materialization-prep-2026-09-29.md). 상위 계약 [MLB-2026-POLICY-PREPARATION-v1](MLB-2026-POLICY-PREPARATION-v1.md)(D89)과 D91 런타임의 식·상태·ARM 선택을 바꾸지 않는다.

> **상태: `PROPOSAL_UNREGISTERED`, `policy_frozen=false`, `execution.enabled=false`.** 이 문서의 §2 식별자·연결 검증만 코드로 구현되고 합성 검사를 통과했다. §3–§5는 실행 전 등록안이며 실제 TRAIN BC 생성·G0 가중치 로드·데이터 로드·추론·OPE를 승인하지 않는다. 실제 값이 없는 필드는 `null`/미측정으로 둔다. 이 세션에서 읽은 것은 코드와 JSON 메타데이터뿐이며 pickle/npz/parquet/모델 payload는 열거나 해시하지 않았다. Sol·Astra 호출은 이 Claude 환경에서 불가능했으며 독립 검토는 미수행이다([검토 패킷](../reviews/COOP-017-policy-materialization-review-packet-2026-09-29.md)).

## 1. 구성요소 출처 (조사 결과)

"pin"은 G0 동결 번들 `configs/G0-RESEARCH-FROZEN-v1.json`의 파일 SHA256이다. 값 인용은 모두 해당 pin과 SHA가 일치하는 JSON 메타데이터에서 이번에 읽었다(payload 제외).

| 구성요소 | 생성 코드 | 필요한 입력 | 기존 pin·메타데이터 | 누락·미확인 | 검증 방법 |
|---|---|---|---|---|---|
| TRAIN BC (로깅 법칙 π̂_b, 기준의 바탕) | `matrix_policy.fit_bc` → `policy_artifacts.export_train_bc`(D91) | 정규시즌 frame(`split=='train'`), BC용 직전 구종 이력 | 독립 artifact 없음. P8은 run 안에서 적합: 1,252,824구, 18구종(`EXP-P8-001/preparation.json`) | 선택 규칙(§3 D-1), 실제 artifact·해시 | `load_train_bc`(sha·canonical·구조), export가 관측 날짜 기록 |
| 개입 지원 표 M | `PolicyInputs.support`(BC ∩ token 어휘 ∩ 12카운트 전부의 구종별 TRAIN pool) → `save_support_table` | TRAIN BC, 연결된 pool, 투수·손·타자면 template | 없음 | 실제 표, 양손 투수 처리(§3 D-8) | `load_support_table` + **요청마다 맥락 행 일치와 pool 재계산 일치 확인(신규)** |
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

**한계.** 식별자는 과학적 타당성·인과 식별을 입증하지 않는다. digest는 같은 프로세스 비교이며, 파생 캐시(pool·지원·frequency 캐시)는 digest하지 않는다(`ponytail:` 주석). 4,096개를 넘는 컨테이너는 앞 64개만 표본으로 보고 class를 찾는다(자료 컨테이너 가정). Python 안에서 `BoundComponents`를 직접 만들어 식별자를 꾸미는 것은 막지 못한다. 등록된 실행기만 `bind_components`를 쓰는 절차가 경계다. 실제 객체가 추론 중 내부 상태를 바꾸면 digest가 거짓 실패를 낼 수 있으며 이는 S2에서 처음 확인된다(미측정). 요청 생성기(원자료→H_t) 코드는 정책 식별자가 아니라 실행 등록의 stage manifest에 pin한다.

## 3. TRAIN BC·지원 표 생성 계획 (미실행)

**자료.** 처리 캐시 `processed_sha256 9f65d1db…`(2,145,111행; statcast 2023/2024/2025 parquet SHA는 G0 번들 `archived_references`) → `run_ml_benchmark.regular_frame`(game_type R, `add_batter_style_history`, `assign_fold(…, 2025)`). G0 계열과 같은 loader·identity 검사를 쓴다.

**선택 조건(제안, D-1 결정 필요).**
- **BC-P(주 제안):** `split=='train'`(2023-05-15..2025-04-30) 정규시즌 행 중 `pitch_type` 비결측, `description ∉ {automatic_ball, automatic_strike}`, 합법 카운트(볼 0–3, 스트라이크 0–2). **결과 라벨·`supported_pa`·plate 좌표 조건을 쓰지 않는다.** 이유: eligibility는 PA 종료 사건·완결성 같은 결정 뒤 정보를 쓰므로 π̂_b가 "PA가 나중에 지원될 조건부 법칙"이 된다(처치 후 선택). 모집단 상한은 TRAIN 전체 1,386,362행(normalizer 보고), 실제 행 수는 미측정.
- **BC-E(P8 호환 비교):** pin된 eligible D100 TRAIN 키 `p4_train_keys`(1,252,824구, `rows_sha256 0a2e81ae…`). P8의 `bc_train_pitches`·18구종과 일치해야 한다(재현 확인).
- 공통: `CategoricalBC(prior_strength=20, minimum_action_count=1)`, 직전 구종은 같은 PA의 엄격히 이전 행(`state_from_row(bc_only=True)`, 결측은 `<UNKNOWN>`, 없으면 `<START>`). 어휘 = 적합된 league 행동(정렬, `vocabulary_sha256`). token 어휘(18개)에 없는 구종은 로깅 법칙에는 남고 개입 지원에서는 빠진다.

**provenance.** `export_train_bc(store, train, path, source_ids={statcast 3개, processed_cache}, config_sha256=<등록 config SHA>, code_commit=<실행 커밋 40자>, data_version=…)`. 날짜는 `observed_by_export_train_bc`.

**지원 표.** 연결된 `PolicyInputs`(§2의 bind)로, BC의 **모든 투수 × 타자면 {L, R}**에 대해 TRAIN에서 관측한 투구 손의 template 맥락을 만들고 `inputs.support`를 저장한다. 빈 마스크도 행으로 남긴다(빠진 키는 런타임에서 pool 불일치로 FAILED_INTEGRITY가 된다). TRAIN에서 두 손으로 던진 투수는 마스크가 다르면 생성이 거부된다(수 미측정, D-8).

**함께 보고할 TRAIN 표(기술 통계).** π̂_b(M|H) 분포(셀·TRAIN 결정 가중 분위, =1 비율) — D89 §5의 ρ^ref≡1 조건 확인용. BC-P와 BC-E의 셀별 차이 요약.

**출력(새 경로, 배타적 쓰기).** `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/ML-POLICY-VAL-v1/S1-materialize/`의 `bc_primary.json`, `bc_p8_compatible.json`, `support_primary.json`, `pib_mass_on_mask.json`, `manifest.json`. 기존 run 디렉터리는 읽기 전용.

**재현 명령(실행기 미구현).** `PYTHONPATH="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/deps:experiments/pitchmdp" python3.12 experiments/pitchmdp/scripts/run_policy_validation.py --config configs/ML-POLICY-MATERIALIZATION-v1.json --local-config <local.json> --output <S1 경로> materialize-bc`. 실제 해시·행 수는 실행 전이므로 채우지 않는다.

## 4. PA·시간 출처 규칙 (제안; ○ 구현됨, △ 부분, ✕ 미구현)

| ID | 규칙 | 상태 |
|---|---|---|
| R1 분모 | 평가 창의 정규시즌 PA 전부 `(game_pk, at_bat_number)`. `supported_pa`·결과 eligibility로 요청을 고르지 않는다(결정 뒤 정보). 사후 층으로만 보고. **P8·G0의 eligible 분모와 다르므로 추정량/분모 변경 — D92 기록, D-2 결정 필요** | △ 원장은 제출 요청 분모만 셈(D91) |
| R2 결정 시점 | PA의 모든 행을 `pitch_number` 순서로 요청: `request_id=game:ab:pitch`, `pa_id=game:ab`, `decision_index`=PA 내 순번 | ✕ 요청 생성기 |
| R3 구종 코드 | (a) 코드 ∈ V → 그대로(병합·재매핑 없음). (b) 비결측 코드 ∉ V → FAILED_INTEGRITY(중단). 그래서 **실행 전 코드 조사(S0)가 필수**, 발견 시 실행하지 않고 재등록. (c) `pitch_type` 결측 또는 `automatic_ball/strike`(투구 없는 피치클록 판정) → 투수 선택이 없다 → 제안 상태 `UNSUPPORTED_NO_LOGGED_ACTION`(거절, 분모 유지, 이후 같은 PA는 `UNSUPPORTED_MID_PA`) | (a)(b) ○, (c) ✕ — 현재 런타임에 넣으면 FAILED_INTEGRITY로 중단되므로 실데이터 전 구현 필수(D-3) |
| R4 투구 전 정보 | 현재 행에서는 SAFE_COLUMNS(키·날짜·투수·타자·손·이닝·아웃·주자·점수·카운트·타자 style prior)만. 같은 PA의 엄격히 이전 행의 구종·정규화 물리·10-class 결과는 이력으로 허용. 현재 공의 구종·물리·description·events와 이후 행은 상태에 없음(로그 행동은 별도 라벨) | ○ `safe_rows`/`state_from_row`/`PolicyInputs`; 요청 생성기 검사 ✕ |
| R4b 프로필 as-of | 투수 cluster·profile은 TRAIN 고정. 타자 style prior는 C0(경기 날짜보다 엄격히 이전). **≤2025 리허설 제안: 평가 창 시작 전(DEV는 2025-06-30까지) 스냅샷으로 고정**해 2026 규칙과 맞춘다. S2 연결 probe만 봉인 예측 재현을 위해 기존 rolling 값을 쓴다(D-7) | ✕ 스냅샷 함수 |
| R4c 출처 검사 | 요청 생성기에서: 이력 행이 같은 PA의 이전 `pitch_number`인지(`state_from_row` 기존 검사), 맥락이 whitelist뿐인지, 카운트가 직전 결과와 이어지는지, 결정 뒤 필드(현재 물리·description·events·이후 행)를 바꿔도 요청·확률이 같아야 함(불변성 검사) | ✕ |
| R5 PA 종료·보상 | 종료 = PA 마지막 행의 `events` 비결측(단 `truncated_pa` 제외). 보상 r = 초기 수비 팀의 동결 C0 WE를 **관측된 다음 행의 투구 전 상태**(`next_*`)에서 계산. 경기 마지막 PA는 최종 승패 {0,1}. 사후 점수와 다음 행 점수가 다르면(`post_pitch_score_disagrees_next_pitch`) 플래그로 보고 | ✕ 추정기(D-6) |
| R6 불완전 PA | PA 상태: COMPLETE, INCOMPLETE_NO_TERMINAL(종료 없음·잘림·자료 끝), INCOMPLETE_START(첫 행이 0-0 아님·누락), UNSUPPORTED(요청 하나라도 거절, sticky). 점추정은 "평가 가능 PA 조건부"로만 이름 붙여 보고하고, 나머지 PA는 정책마다 [0,1] 최악 경계로 전체 구간에 넣는다(D87). 결과로 PA를 빼지 않는다 | △ 요청 sticky ○, PA 상태·경계 ✕(D-5) |
| R7 PA 중 교체 | 새 투수가 TRAIN 이력 없음 → 그 요청 `UNSUPPORTED_UNKNOWN_PITCHER`, 이후 `UNSUPPORTED_MID_PA`(구현). 새 투수가 알려진 경우 → **제안: 계속 평가**(교체는 환경 동역학, 시뮬레이터는 교체를 모형화하지 않아 q̂ 오차로 남고 DR의 π̂_b 경로에 기댄다). 타자 교체도 같은 원칙 | 미지원 경로 ○, 알려진 교체 규칙 D-4 |
| R8 이력 불일치 | 직전 결과와 카운트가 이어지지 않는 요청(행 누락 등)은 결정 전에 알 수 있는 자료 결함 → 제안 상태 `UNSUPPORTED_INCONSISTENT_HISTORY`(거절·sticky). 전달된 이력과 원장의 불일치(순번·길이·직전 구종)는 지금처럼 FAILED_INTEGRITY | 원장 국소 검사 ○, 카운트 경로 ✕ |

D91의 한계는 그대로다: 원장 분모는 **제출된** 요청/PA이고, 이력 검사는 국소 일관성이다. 위 R2·R4c가 구현되어도 전체 경기 모집단 완전성은 스냅샷 manifest의 경기·PA 수(R1 계층: 전체 R 경기 → 수집 경기 → PA → 요청 상태 → 평가 가능 PA)로 따로 보고한다.

## 5. ≤2025 검증 자료·비용·실행 계획 (미등록, 미실행)

**노출 이력.** 인증된 미노출 모집단은 0개다. 아래는 모두 **개발 검증**이며 독립 확인이 아니다.

| 집합 | 기간 | 과거 사용 | 라벨 |
|---|---|---|---|
| TRAIN | 2023-05-15..2025-04-30 | G0·빈도·pool·normalizer·BC·WE 적합 | `exposed_fit` |
| earlystop | 2025-05-01..05-15 | G0 조기 종료(16,000구·195경기 표본) | `exposed_model_selection` |
| May | 2025-05-16..05-31 | member·frequency temperature, P8 profile | `exposed_calibration` |
| June | 2025-06 | G0 June 가중치, P8 τ 선택, D81/D86 보정 family | `exposed_policy_tuning` |
| DEV | 2025-07-01..09-30 | G0 전체 MLB 평가(341,941행 중 eligible 311,721구·1,161경기), P8 DEV, D86 | `exposed_development` |
| 2023-03-30..05-14 | `unused` 분할 | 과거 사용 여부 메타데이터로 미확인 | `unknown`(미사용 주장 안 함) |
| ≤2022, 포스트시즌 | 처리 캐시 밖 | — | 가용성 `unknown` |

**단계(단일 큐, 공유 `.heavy.lock`, stage마다 새 출력, 실패 보존, 새 attempt 없이 재시도 금지).** 비용은 모두 **미측정**이며 상한은 제안값이다.

| stage | 내용 | 읽는 자료 | 제안 상한 | 실행 가능성 |
|---|---|---|---|---|
| S0 census | TRAIN·DEV의 구종 코드·결측·`automatic_*` 수, PA 구조(첫 행 0-0, 종료 유무, PA 중 투수/타자 교체), TRAIN 투수 손 일관성 | 투구 전 필드, `pitch_type`, `description`의 automatic 여부, `events` 유무만 | 900초 | 실행기 필요 |
| S1 materialize | §3 BC-P·BC-E·지원 표·π̂_b(M|H) TRAIN 분포 | TRAIN 행 | 1,800초 | 실행기 필요 |
| S2 bind-probe | 실제 bind(aux·체크포인트 5·game_values), 완전 식별자 기록, 연결 probe: `mlb_dev` 고정 순서에서 5개 member 모두 `dev_delivery_level ≥ 0`인 첫 64행의 G0 primary(atol 1e-6, EXP-P11-001 probe 기준)와 frequency raw(atol 1e-12) 재현, `verify_components` | aux·체크포인트·game_values·해당 행과 PA 이력·봉인 예측의 `keys`/`primary`/`mlb_dev_raw`/level 배열(라벨 배열은 읽지 않음) | 1,200초 | 실행기 필요 |
| S3 profile | 고정 소수 PA 시작(May, P8 profile 규모 4개)에서 G0 5member P2/P3 결정 비용(조건부 행/초) 측정. 품질 값 비열람 | May 맥락 | 1,800초 | 실행기 필요, τ 없으면 P2만 |
| S4 V5 | DEV 전체 PA의 로깅 전용 런타임 상태 분모(미지 투수·빈 지원·코드·무행동·마스크 밖 로그 비율·π̂_b(M|H)) | DEV 투구 전 필드와 로그 구종 | 3,600초 | 실행기·R2·R3(c) 필요. 원장 행마다 fsync라 비용 위험(미측정) → 한 경기 subset으로 먼저 profile |
| S5 V2/V3 | 모형 세계 준합성·이동 민감도 | DEV PA 시작 맥락 | `null` | 차단: M 밖 로깅 행동의 delivery/전이 정의(D-10), 로그 생성기·DR 추정기 미구현 |
| S6 V4 | 실제 DEV 검사 | DEV 결과 | `null` | 차단: R5·R6 추정기 미구현 |

family 상한 제안 9,300초(S0–S4). 참고 실측(다른 범위, 예산 근거 아님): P8 prepare 186.6초(BC 적합 포함), P8 3member stage 83–112초/71–106천 조건부 행, G0 전체 MLB 공식 평가 1,821.2초, D86 473.5초.

**식별자·환경·명령.** source C = 실행기 포함 커밋(미정), 등록 config D = 이 config의 등록본 SHA(미정), 환경 = G0 번들 환경(`/opt/homebrew/…/python3.12`, PYTHONPATH `…/ML-MATRIX-20260924/deps:experiments/pitchmdp`, 버전 §1). 출력 루트 `…/ML-MATRIX-20260924/ML-POLICY-VAL-v1/`(존재하면 거부). 명령 형식은 §3과 같고 subcommand만 `census|materialize-bc|bind-probe|profile|v5-denominators`.

**중단·실패.** stage 시간/행 예산 초과는 `BudgetExceeded` → 원장 `FAILED_RUNTIME`·stage 실패 기록 보존. FAILED_INTEGRITY는 즉시 중단. 결과를 본 뒤 문턱·표본·후보 변경 없음. 재시도는 새 attempt 등록으로만 하고 이전 시도를 공개한다.

## 6. 결정이 필요한 항목

| ID | 결정 | 제안 |
|---|---|---|
| D-1 | TRAIN BC 주 선택 | BC-P 주, BC-E 비교 |
| D-2 | 분모 | 모든 정규시즌 PA(R1) |
| D-3 | 무행동 행 | 새 거절 상태 + sticky |
| D-4 | 알려진 투수의 PA 중 교체 | 계속 평가 |
| D-5 | 불완전 PA | 평가 가능 PA 조건부 점추정 + [0,1] 경계 |
| D-6 | PA 종료 상태 | 관측 다음 행 상태 + 불일치 플래그 |
| D-7 | ≤2025 타자 프로필 | 평가 창 시작 스냅샷 |
| D-8 | 양손 투수 지원 표 | 조사 후 결정(수 미측정) |
| D-9 | G0 τ·탐색 설정 | June 2025 재사용 여부 포함 별도 등록 |
| D-10 | V2 세계에서 M 밖 로깅 행동 | 선언된 세계 정의로 JointDelivery tier+fallback 사용 제안 |
| D-11 | stage 상한 | 위 제안값 |

Astra 독립 과학 검토 1회와 Song 결정 전에는 등록·실행하지 않는다.

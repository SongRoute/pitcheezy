# 정책 구성요소 실제 연결·완전한 식별자·≤2025 검증 준비 (COOP-017)

2026-09-29, 기준 `713c7ea`(검증 완료 `ec514f5`의 후손), 브랜치 `codex/ml-matrix-execution`. 계약 [ML-POLICY-MATERIALIZATION-v1](../contracts/ML-POLICY-MATERIALIZATION-v1.md), 설정 [configs/ML-POLICY-MATERIALIZATION-v1.json](../../configs/ML-POLICY-MATERIALIZATION-v1.json), 검토 패킷 [COOP-017](../reviews/COOP-017-policy-materialization-review-packet-2026-09-29.md), 기계 기록 [results/ML-policy-materialization-prep-v1.json](../../results/ML-policy-materialization-prep-v1.json).

> **구현·합성 검사·등록안 작성까지다.** 실제 TRAIN BC 생성, G0 가중치·aux·WE payload 로드, 데이터 로드·추론·OPE, 2026 접근은 0이다. 이 세션은 코드와 JSON 메타데이터만 읽었다. 과학 규약·ARM 상태·G0 유지·`policy_frozen=false`는 그대로다. Sol·Astra는 이 환경에서 호출할 수 없어 **독립 검토는 미수행**이다(패킷만 준비).

## 1. 한 줄 요약

후보 정책이 쓰는 모든 부품(G0 5개 모델과 로더, 빈도 모델, 입력 전처리, 400개 투구 궤적 pool, 승리확률 WE)을 고정 파일에서 **직접** 조립하고 대조하는 코드를 만들었다. 부품 하나만 바뀌어도 정책 ID가 바뀌거나 실행이 거부된다. 야구로 비유하면, 선발 명단에 이름만 적는 대신 경기 전에 실제 선수의 등번호·체격·장비를 하나씩 확인하고 경기 중 선수가 몰래 바뀌면 경기를 멈추는 장치다. 실제 자료 실행 계획(TRAIN BC 생성, PA·시간 규칙, ≤2025 검증 단계·비용 상한)도 등록 전 문서로 정리했다.

## 2. 조사 결과 (계약 §1 표 요약)

- **frequency 역할(D91 공백) 해소:** `run_ml_matrix._fit_aux`가 `aux.pkl['baseline']`에 `ContextFrequencyBaseline`을 저장한다. G0 예측의 frequency 항과 P8의 `inputs.pkl['baseline']`은 모두 EXP-P4-001의 같은 aux(`p4_auxiliary` `87ad95e6…`, preparation의 `artifact_hashes['aux.pkl']`와 일치)에서 왔다. temperature 1.030298001849738은 `p4_preparation`에 있다. 코드와 메타데이터 근거이며 pickle bytes는 열지 않았다.
- **보정값 확인:** June 앙상블 가중치 0.7467754577062221과 May temperature 5개는 해시가 번들 pin과 같은 `EXP-P11-001/frozen_calibration.json`에서 이번에 직접 읽었다. D91 보고서가 "열어 확인하지 않았다"고 적은 두 값(0.7467754577062221, 1.030298001849738)이 이것으로 확인됐다.
- **member 로더:** `run_ml_g0_whole.load_member`. 기록상 flatten_mlp, width 128, context 52, parameter 141,834, device `mps`.
- **전처리·pool:** `SharingContext(aux['context'], clusters)`, 어휘 18개, history 5, token 38채널. pool은 eligible D100 TRAIN 1,252,824구로 seed 42에서 적합했고 23,728개다. normalizer만 TRAIN 전체 1,386,362행으로 적합됐다.
- **WE:** C0 `model-v1.json`(`d0f1d452…`)→manifest→lineage→`game_values.pkl`(`aa6c4e48…`). 현재 WE 소스 3파일이 lineage와 같다. WE는 원래 빌드에서 2025 DEV Brier gate를 거쳤다(노출).
- **코드:** 현재 소스가 G0 실행 소스 30파일과 바이트 동일하다.
- **남은 공백:** G0용 τ·MC·cap·seed·budget은 전부 미등록이다. 요청 생성기(원자료→H_t), 추정기, 실행기가 없다. 실제 payload에서 등록 class·report 일치는 아직 확인하지 않았다.

## 3. 구현

| 파일 | 변경 |
|---|---|
| `experiments/pitchmdp/pitchmdp/policy_identity.py` (신규) | `bind_components`(pin된 번들 파일)/`BoundComponents`, pin 뒤에만 unpickle, 선언↔객체·저장소 class·산출물 연결 검사, 로드 뒤 체크포인트 재해시, `load_pinned_we`(`run_ml_policy.verify_we`의 명시 경로판), 소스 import closure·환경·content digest·객체 연결 확인, 맥락 행 SHA, `integrated_predictions`/`compare_probe` |
| `experiments/pitchmdp/pitchmdp/policy_runtime.py` | `build_runtime(components=…, expected_identity_sha256=…)`. 느슨한 `g0/pool/terminal/cutoff/we_identity` 인자 삭제. 요청마다 맥락 행·지원 표 = pool 확인(`support_check`, 빈 마스크 거절보다 먼저). 맥락 행 SHA를 요청 fingerprint·원장 행에 기록. `verify_components()` 추가(직접 만든 후보는 거부) |
| `experiments/pitchmdp/tests/test_policy_identity.py` (신규) | 합성 pin 파일·가짜 부품·가짜 로더와 실제 평가 경로 probe |
| `experiments/pitchmdp/tests/test_policy_runtime.py` | E2E를 bound components 경로로 이전(기대값·단언은 유지) |

D91 의미는 보존했다: 로깅 법칙은 전체 어휘 BC이고 재정규화하지 않는다. G0는 member별 보정 → 5확률 평균 → 공통 앙상블 혼합 순서를 지킨다. runtime SHA는 이제 후보 모드에서 완전한 식별자를 포함한다. 로깅 전용 런타임은 후보 식별자가 없다. 원장 분모는 여전히 **제출된** 요청/PA이고, 이력 검사는 국소 일관성이다. **상태 변경:** 후보 모드에서 지원 표 키 누락·빈 마스크인데 pool이 지원하는 요청과 맥락 행의 투수·타자면 불일치 요청은 `UNSUPPORTED_EMPTY_SUPPORT`가 아니라 `FAILED_INTEGRITY`다(계약 §2, D92).

## 4. 검사

모든 검사는 합성이다(임시 디렉터리의 가짜 pin 파일·가짜 부품·가짜 로더, 작은 합성 frame). 명령은 `PYTHONPATH=src:experiments/pitchmdp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider …`.

- **신규·수정 집중 검사:** `test_policy_identity.py`(신규 8) + `test_policy_runtime.py`(15, E2E 이전) = **23 passed, 23 subtests**.
- **관련 묶음:** 위 두 파일 + `tests/test_2026_policy_contract.py` + `test_matrix_policy.py` + `test_rollout_policy.py` = **62 passed, 23 subtests**(신규 포함, 중복 합산 없음). `scripts/check_2026_policy_contract.py` exit 0, `all_pass: true`. 영향 확인용 `test_g0_whole_mlb.py`·`test_g0_whole_metrics.py` 24 passed.
- **연결 probe(합성):** `bind_components`로 연결한 정책 경로가 실제 평가 코드와 12행에서 최대 차 4.4e-16. fallback pool을 쓴 3행은 정책 경로가 거절했다. pool seed 43 6.2e-3, clusters 7.0e-3, frequency T 8.8e-3, member temperature 순서 2.4e-2로 교체를 검출했다.
- **변이 검사 28종 모두 검출:** 매번 한 곳을 임시로 고치고 두 집중 파일을 실행한 뒤 바이트 원복(`git status`로 확인). 목록은 기계 기록에 있다. import closure 변이는 처음에 빈 loop 본문으로 문법 오류가 나서 검출로 세지 않았고, `pass`로 고쳐 실제 실패를 확인했다.
- **실패·수정 이력:** 첫 구현은 한 번에 22 passed였다. 이어 (1) 소스 pin이 모듈 수준 import를 놓치는 공백을 스스로 찾아 import closure를 추가했다. (2) 같은 모델(Claude Opus) 하위 에이전트의 자기 점검이 major 3·minor 6을 찾았고, 실행 데모로 확인된 항목을 포함해 모두 고쳤다(아래). 수정 뒤 첫 실행은 식별자 키 집합 단언 1건이 예상대로 실패했고, 단언을 고쳐 23 passed.
- **자기 점검 지적과 수정(독립 검토 아님):**
  - `verify`가 실제 추론 경로의 객체(frequency 객체, `inputs.delivery`, `g0.inputs`, 맥락 인코딩, class 수준 TIERS)를 보지 않음 → 객체 연결 검사와 digest 확장.
  - 맥락 행 값이 식별자·fingerprint에 없음 → 요청 fingerprint와 원장 `context_sha256`에 포함.
  - 투수·타자면 불일치 요청이 빈 지원 거절로 바뀌고 공유 캐시를 오염 → 지원 계산 전에 FAILED_INTEGRITY.
  - 직접 만든 후보 런타임의 공허한 verify → 거부.
  - 좁은 예외 변환 → 결합 중 실패는 모두 FAILED_INTEGRITY.
  - 번들 dict 무결성 → 파일 pin.
  - 체크포인트 재오픈 틈 → 로드 뒤 재해시.
  - 저장소 밖 class → 거부.
  - factory의 보정 JSON 재읽기 → 검증 bytes와 대조.
  - 식별자 dict 변경 → 해시 재확인.
  - fallback 검사가 어휘 밖 행동이던 문제 → 실제 fallback tier 행으로 교체.
  - probe가 bind를 거치지 않음 → bind 경로로 교체.
  - 후보 모드의 빈 지원 → 무결성 실패로 바뀐 상태 변경을 계약·D92에 기록.
- **Graphify:** AST 갱신 6,552 nodes·18,433 edges·268 communities. 의미 분석·LLM 라벨 갱신은 하지 않았다.
- **하지 않은 검사:** 실제 payload·모델·데이터 검사, 전체 테스트 묶음, Sol 재현·Astra 검토.

## 5. 하지 않은 것·한계

- pickle/npz/parquet/pt payload 열람·해시, 데이터 로드, 실제 bind, BC 생성, 추론·OPE, 2026 접근은 모두 0이다. 기존 봉인 파일·config·attempt는 수정하지 않았다(새 파일과 두 코드 파일, 테스트만 변경).
- 식별자와 연결 검사는 "같은 부품이 연결됐다"를 보장할 뿐 인과 식별·정책 우위를 입증하지 않는다. content digest는 같은 프로세스 비교이며, 실제 객체의 내부 캐시가 거짓 실패를 만들지는 S2에서 처음 확인된다.
- §4 규칙 중 R2·R3(c)·R4b·R4c·R5·R6·R8은 미구현이다. 특히 투구 없는 행(R3(c))이 현재 런타임에 들어오면 FAILED_INTEGRITY로 전체가 멈추므로 실데이터 전에 구현해야 한다.
- 비용은 모두 미측정이며 상한은 제안값이다.

## 6. 남은 조건과 바로 다음 작업

1. Sol 재현과 Astra 핵심 검토 1회(Codex 협업 환경 필요), 그리고 Song의 D-1~D-11 결정.
2. 실행기 `run_policy_validation.py`와 요청 생성기를 구현하고 합성 검사한다(census·materialize-bc·bind-probe·profile·v5-denominators; R2·R3(c)·R4b·R4c·R8). 실데이터는 쓰지 않는다.
3. 검토·승인 뒤 `ML-POLICY-VAL-v1`을 source C·config D·환경으로 등록하고, 단일 큐에서 S0→S4를 실행한다. S2가 실제 부품의 첫 연결 확인이다.
4. 그 뒤 추정기(R5·R6), V2 세계(D-10), G0용 τ 등록(D-9) 순서다.

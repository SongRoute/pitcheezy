VERDICT: PASS_WITH_NOTES

# CODE-REVIEW-A2B-SONNET-002-CLOSURE — EXP-OPE-RES-02(트랙 A2b) 구현 코드, G-CODE 2차 마감 확인

- **검토자 모델:** Claude Sonnet(검토자) (claude-sonnet-5-5). 작성자(Claude Opus 5.5)와 다른 모델. 읽기 전용(코드 읽기·합성 테스트·스크래치 사본 변이). 1차는 `CODE-REVIEW-A2B-SONNET-001.md`(FAIL: MAJOR 4·MINOR 7·NOTE 11).
- **시각:** 2026-10-28 21:31 KST. **대상:** `/Users/song/Projects/pitcheezy-worktrees/track-a-a2b-1028`, HEAD = `53df4e2293d9dfcbe270ddbb1a1aa4dd6732352a`(`git rev-parse` 확인; 시작·끝 `git status --short --ignored` 빈 출력, `__pycache__` 0). 1차 대상 `821d79cb` 위 커밋 11개(aa9bfb0f … 53df4e22).
- **설계 파일:** 이번에는 md/JSON을 열지 않았다. v2.3은 1차에서 읽은 내용, v2.4는 PM 패치 목록 `scratchpad/v24-changes.md`(읽기 전용)를 기준으로 판정했다. v2.4와 코드의 최종 일치는 v2.4 완성 뒤 별도 확인이 필요하다.
- **기존 파일 수정 0:** `git diff --diff-filter=M --name-only b63369d0..HEAD` 출력 없음. `821d79cb..HEAD`는 a2b 파일 15개의 M뿐(모듈 5·실행기 1·테스트 9; 추가·삭제·이름변경 없음, `ope_decision_a2b.py`·`NEW_FILES` 목록 불변). `conftest.py`·`pytest.ini` 추가 없음.
- **테스트(직접):** `python -B -m pytest tests/test_ope_a2b_*.py -q -p no:cacheprovider` → **103 passed, 114 subtests passed, 18.27 s**, 최대 RSS 474,152,960 B(0.442 GiB; 최대 메모리 사용량 343,704,560 B = 0.320 GiB) — PM 수치와 일치.
- **개수:** BLOCKER 0 · MAJOR 0 · **MINOR 1** · NOTE 6. 1차 MAJOR 4개는 모두 CLOSED. 코드 기준 OPEN 없음.

## 1. 1차 항목별 판정
| 1차 항목 | 판정 | 근거 |
|---|---|---|
| MAJOR-1 λ=100 고정 행 공개 | **CLOSED** | `ope_registry_a2b.py:75-76,387`(`PUBLIC_ROWS` 24·`I_ROW`), `ope_sealed_a2b.py:305-306,439,447`(`held`/`target`, `sealed['lambda_fixed_row_public']`), `run_ope_a2b.py:960-975`(UB 공개). 직접 재현 §2-1 |
| MAJOR-2 AppleDouble | **CLOSED** | `ledger_a2b.py:433` `is_appledouble` 건너뜀. 합성 0x80 사이드카 통과 재현, `rglob`류 열거는 이 한 곳뿐(grep) |
| MAJOR-3 종결 FAILED_INTEGRITY | **CLOSED** | `run_ope_a2b.py:395-465,508`. 직접 재현 §2-3 |
| MAJOR-4 구동기 동등성 | **CLOSED**(구현·검증) | `ledger_a2b.py:306-411`, `run_ope_a2b.py:617-664`. §2-4 (i)~(vi) |
| MINOR-1 TRAIN 재읽기 | **CLOSED** | `kappa_prime_from_s0`(`run_ope_a2b.py:569`) = 봉인 `kappa_calibration.TAB`; A2b 코드에 TRAIN 읽기·`kappa_prime_inputs` 없음(grep). `run_ope_a2.py:1355-1357`이 S5X·TAB에 같은 `hit`·`groups`·`delta`를 넘기고 `names`만 다름을 다시 확인 |
| MINOR-2 `select` | **CLOSED** | `run_ope_a2b.py:1037-`: `env.nice()` + `with env.heavy_lock(local)` 안에서 적재; 테스트 있음, 변이(E6) 검출 |
| MINOR-3 장식 pin | **CLOSED** | `ledger_a2b.py:193-219`(`runtime_setting`·`dev_style_snapshot` 실제 값 대조), `run_ope_a2b.py:702`(pyproject, A2b-S3:R3), `ope_registry_a2b.py:629-641`(`COMPARED_AT_RUN`·`RECORD_ONLY_PINS`). NOTE-C |
| MINOR-4 G-INT-R3 빈 비교 | **CLOSED** | `ope_denominator_a2b.py:264` 6키 필수, 부재 키는 FAIL; 변이 E4 검출 |
| MINOR-5 테스트 구멍 | **CLOSED**(잔여 → MINOR-A) | 1차 생존 7개 전부 검출(§5); 전 단계 체인 테스트 실행(§2-9) |
| MINOR-6 B2 유효성 | **CLOSED** | `ope_denominator_a2b.py:179` 무효 = 0질량·비유한 ρ·누락/형식 오류 벡터; support 차이는 서술 개수; 변이 E5 검출 |
| MINOR-7 SM 범위 | 수용(v2.4 문구) | 코드 변경 없음, 패치 목록에 있음 |
| NOTE-1·3·9·10 | 수용 | 변화 없음(패치 목록: 1·3 반영) |
| NOTE-2 A2 시드 라벨 | **CLOSED** | `ope_synthetic_a2b.py:68-73`(실제 κ·ζ 라벨), 테스트 |
| NOTE-4 gint_a2/r3 형식 | **CLOSED** | `validate_key_record`(`ope_registry_a2b.py:515`), 러너가 두 기록에 호출(`:715,:774`) |
| NOTE-5 int/float 혼합 | 변화 없음(수용) | `integrity_equal` 불변 |
| NOTE-6 프레임 재적재 | 변화 없음 | 아래 Q3 |
| NOTE-7 등록본 기대값 | 변화 있음 | 아래 Q3·NOTE-C |
| NOTE-8 실패 범위 | **CLOSED** | `failure_scope: all` 기록·검증(`ope_registry_a2b.py:462,486-487`) |
| NOTE-11 `__main__` 위치 | **CLOSED** | 7개 테스트 파일 모두 마지막 클래스 뒤 |

## 2. 코드 확인
1. **MAJOR-1 (직접 재현, `probe2_lambda.py`):** 선택 λ·강제 100·강제 30 세 경우 모두 공개 파일(`sa.seal` 바이트)에 `R1_bxR2_1xB0_lambda100`·`lambda100` 문자열이 없고 공개 행은 24개다. 강제 30에서는 고정 행 값(float 22개) 중 공개 파일에 나타나는 것이 0개; 100이면 값이 선택 행 자신의 것과 같아 새 정보가 없고 동일성 플래그는 봉인에만 있다. 게이트의 `PublicView.get` 51회 중 고정 행 접근 0회. 고정 행이 든 공개 파일은 `sa.seal`·`validate_public`·`require_descriptive`가 모두 거절한다(테스트와 변이 A1~A5 검출). 판정은 `dc.condition_ii`·`s5_pass`·`decide`가 후보 행만 읽는다(B2 행은 서술이라 게이트 대상이 아니다).
2. **MAJOR-2:** 위 표. 매직 없는 `._notes.json`은 일반 파일로 계속 검사한다(테스트).
3. **MAJOR-3 (직접 재현, `probe2_decide.py`):** A2b-L 소진 → 종결 FAILED_INTEGRITY, `blocked` = [G, S3], 게이트 sha `None`; S0 소진 → `blocked` = [G, L, S3]; G 소진 → 종결; R3만 소진 → 종결 아님(게이트 없음 거절). `plan_decide`(:433)는 SM·S0·L·S3:R3·S3·G를 먼저 읽고, 2단계 소진도 게이트 기록과 함께 종결(`phase_2_upstream` 폐포). 라벨 우선순위: 종결(1) → `dc.decide`의 NOT_EVALUABLE(2)…(`decide` 안 1 대 2 순서도 테스트로 고정). 거절(게이트 없음)을 3번 해도 A2b-S4 attempt-start 0건 — 시도 소모 없음(`probe2_attempts.py`; `PLANS`가 `roa.stage` 진입 전에 돈다).
4. **MAJOR-4:** (i) 두 경로 모두 S6 행은 `bw.read_ledger` 허용 필드뿐(직접 출력: 키·`result` 키 모두 A3 허용 목록), 제출은 `submit()` → `outcome=None`(스파이: 경로 A 1회·경로 B 2회 모두 `None`, `outcome_function`에 AssertionError를 심어도 불호출), 기록에 보상·중심 키 없음. (ii) `path`·`runtime_sha256_reproduced`·`style_source`·`reason`·`games`·개수·`max_abs_delta`가 `driver_equivalence.json`과 `ledger_record.json['driver_equivalence']`(`run_ope_a2b.py:636,661`)에 남고 체인 테스트가 확인한다(:847-850). 단 이 두 파일에는 타입 검사기가 없다 — 통과하는 검사는 `check_no_reward_outputs`의 금지 키 훑기뿐(NOTE-A). (iii) 결정 집합 동일(`missing`·`extra`·`not_comparable` 모두 0 요구), 실수 절대 1e-12(`integrity_equal`)·그 외 정확(`ledger_a2b.py:306-321`); 테스트: 5e-13 통과·2e-12 실패·정수 필드·행 누락/추가 실패. (iv) `require_equivalence`(:406)·경로 B 쌍 sha(:401)·경로 B 신원(`run_ope_a2b.py:640-642`)은 모두 IntegrityError이고 러너 연결을 직접 재현했다(`probe2_wires.py`: 실패 기록 주입 → "driver equivalence failed … (deterministic integrity)", 다른 신원 → "path-B runtimes are not the new ledger runtime", pyproject 불일치 → R3 단계 실패). (v) `world.s6_games`는 pa_id에서 만들고 `lg.selection`이 `select_games` 재계산과의 일치를 먼저 요구한다(:626-627, 동등성은 :632 뒤) → 동등성은 검증된 목록만 쓴다; `replay_games`는 원장 순서 첫·끝 경기(같은 경기면 거절). (vi) `read_ledger(runtime_sha256=pins['s6_runtime_sha256_header'])`가 헤더 sha = pin(및 모든 행)을 강제하므로 `driver_equivalence`의 `header_sha`는 곧 등록 pin이다. 경로 A는 합성 S6(등록 `dispatch` 체인이 만든 runtime-v3 `dr-evaluate`)에서 sha를 재현해 통과; 실제 S6-dr-a5 헤더에서의 재현 여부는 **미확인**(열지 않음) — 안 되면 경로 B로 가고 그 사실이 공개 기록에 남는다(PM 수용).
5. **MINOR-1:** `kappa_prime_from_s0`는 키 집합·그룹·`kappa_p == s0['kappa_p']['TAB']`를 검사하고 TRAIN을 읽지 않는다. 변이 E7(`TAB_GROUPS` 동일성 단언 삭제)만 생존 — 상수 단언이라 무시.
6. **MINOR-2/3/4/6:** 위 표. `runtime_setting`은 타입까지 같아야 하고(`type(pinned) is type(used)`), `evaluation_role`은 문자 그대로 `'evaluation'`(동결 `dr_q.source`가 평가 시드일 때).
7. **역할 9:** 값 3206983116을 numpy로 재계산해 일치(1차 계산과 동일). 리허설은 이 시드로 pigeonhole을 봉인에만 기록(공개 파일에 키 없음, 테스트). `config_check`는 역할 집합 일치를 요구하므로 9역할 초안은 거절된다 — 합성 초안에서 `pigeonhole`을 뺀 변이를 테스트가 거절함으로 확인(실제 JSON은 열지 않음). 결함 아님.
8. **NOTE-2/4/8/11:** 위 표.
9. **전 단계 체인(`FullChainTests`, runner 테스트 :786-):** `rr.main`으로 smoke → s0 → ledger(경로 A) → s3-r3 → s3 → gate → s2a-gint×3 → s2a-gint-record → s2a×10 → s2a-pool → s1 → decide → unblind를 실제 본체로 실행한다. 패치된 것은 설정 기대값(합성 후보 신원)·크기(`N_GAMES`·TAB 크기·cap·반복 수)·게이트 조정(`dc.condition_ii`·`s5_pass`)·`require_s5x_consistent`뿐이고 `body_*`는 모킹되지 않았다. 체인 끝에서 봉인 순서(`stage_seal` 24개)·`stage_failure` 0건·UB의 `lambda_fixed_row_public`·공개 24행을 단언한다. 합성 환경 훅(S5 DEV 선택 세계 대신 고정물 세계, B2 stub, A2 `Environment`의 봉인·결과 읽기)은 대체되므로 실제 훅은 이 테스트가 덮지 않는다(A2 `FullChainTests`와 같은 한계, NOTE-E).

## 3. 새 발견
- **MINOR-A — 러너의 fail-closed 연결선 3개가 테스트로 보호되지 않음 (변이 생존).** `run_ope_a2b.py:637`(`lg.require_equivalence`), `:640-642`(경로 B 신원 = 새 원장 런타임), `:702`(pyproject pin)과 `:715,:774`(`validate_key_record`)를 지워도 103개가 모두 통과한다(D1·D2·E2·E3). 코드는 맞다(위 직접 재현). 시나리오: 후속 편집이 이 줄을 지우면 동등성이 실패해도 A2b-L이 봉인된다. 수정: 합성 체인 환경에서 (a) `driver_equivalence`가 실패 기록을 돌려줄 때 `ledger`가 IntegrityError로 봉인되지 않음, (b) 경로 B 기록의 `runtime_sha256`이 틀리면 거절, (c) `pyproject_sha256()`이 다르면 `s3-r3`가 실패하는 테스트 3개 — `probe2_wires.py`가 그대로 뼈대다. G-REG를 막지 않는다.
- **NOTE-A — A2b-L 기록에 형식 검사기가 없다.** `driver_equivalence.json`·`ledger_record.json`은 금지 키 훑기(`check_no_reward_outputs`)만 통과한다. 키는 모두 보상과 무관하고 불일치 키 이름은 최대 20개(`pa_id#decision_index`·필드 경로)뿐이다. v2.4에 이 두 기록의 필드 목록을 적을 것.
- **NOTE-B — A2b-L의 RSS 여유 미측정.** 새 원장 앞에 런타임 조립이 3~4번 더 순차로 돈다(경로 A 1, 경로 B 2; `del`+`gc.collect()` 있음). 설계 기록은 원장 6.77 GiB, 상한 8 GiB. 실제 증가분은 미측정 — smoke(G-SMOKE)가 못 재는 단계이므로 첫 실행의 RSS 기록(`peak_rss_bytes`)을 보고 판단.
- **NOTE-C — pyproject pin은 실행 checkout의 파일과 비교된다.** `Environment.pyproject_sha256()`은 러너가 속한 checkout의 `pyproject.toml`이다. A3 기록 sha `5eb30162…`는 이 worktree의 파일과 같지만 메인 checkout(브랜치 `codex/ml-matrix-execution`)의 현재 파일(sha256 `cb614006…`)과는 다르다. 등록 실행이 어느 checkout에서 도는지에 따라 R3 단계가 결정적으로 실패(B2 행만 무효)할 수 있다. G-REG에서 pin을 실행 checkout 기준으로 확정.
- **NOTE-D — pin 분류의 빈칸.** `ml_policy_d127_chain`·`s2_prime_runs`는 등록 시점 동일성 검사만 있고 `COMPARED_AT_RUN`/`RECORD_ONLY_PINS` 어느 쪽에도 안 적혀 있다(문서 한 줄).
- **NOTE-E — 체인 테스트의 경계.** 위 §2-9 대체 훅 외에 `assertLess(…, 120)` 벽시계 단언이 있다(측정 약 6초, 부하 시 흔들릴 수 있음).
- **NOTE-F — 경로 A의 실 S6 재현은 미확인.** 헤더의 `pins.provenance.style_report` 존재와 sha 재현은 합성 S6로만 확인했다. 또 경로 A(와 1차부터의 G-INT-S2EQ)의 절대 1e-12 정확 비교는 G0 추론의 실행 간 결정성에 기댄다 — 장치는 pin된 신원에 들어 있고(`policy_identity.py:398`) 실행 간 변동은 미측정이다. 거짓 실패는 fail-closed(결정적 무결성 실패)로 끝나며 잘못된 숫자를 내지 않는다.

## 4. PM 질문
- **Q1 (`processed_frame_aligned` 기록용):** 수용. A2b-L이 읽는 DEV 프레임·store는 `Environment.policy_inputs`(A2 것을 A2b가 재정의하지 않음, `run_ope_a2.py:1881-1883`)에서 `verify_processed_pin`(:1013-1024)이 **`processed_frame_identity` pin**(경로 = 등록 `processed/pitches.parquet`, sha256 = 상속 `data_quality.json`의 `processed_sha256` = 파일 바이트)과 대조한 뒤에야 `rpv.load_inputs`로 적재한다. 적재기 자체도 `regular_frame`이 처리 캐시를 검증하고 `sequence_data_identity`·`matrix_source_provenance`를 고정된 G0 parent 준비 파일과 맞춘다(`run_ml_benchmark.py:110-117`; 그 파일은 등록 사슬의 G0 번들 pin 안). 정렬본(`processed_frame_aligned`)은 TRAIN 행 읽기 전용이었고 A2b는 읽지 않는다(grep: 레지스트리 형태 검사와 보호 경로뿐). `processed_frame_identity.sha256 == aligned.source_sha256`은 등록 시점에 계속 강제된다(`check_pins`). 따라서 MAJOR 아님.
- **Q2 (경로 B):** 있다. 두 런타임 sha 동일 요구 `ledger_a2b.py:401`, 그 sha = 새 원장 런타임 sha 요구 `run_ope_a2b.py:640-642`(새 원장 제출 전, 조립 직후). 둘 다 직접 재현했으나 테스트가 지우는 변이를 못 잡는다(MINOR-A). 쌍 sha 검사(D4)는 `runtime_sha256`이 비교 필드에 있어 `compare_allowed_fields`와 중복이다.
- **Q3:** NOTE-6은 그대로다(`s2a`·`s2a-gint` 명령마다 `s2_world` → `s5x_inputs`로 프레임·store 적재; 비용 게이트 벽시계는 적재 뒤; 시간 미측정, 패치 목록에 한계로 기록). NOTE-7은 일부 바뀌었다: 역할 9 추가, `runtime_setting`·`dev_style_snapshot`은 이제 A2b-L 시작에서 등록 사슬 값과 **같아야** 하고(τ·samples·pitch_cap 타입 포함, 스타일 경로는 resolve 후 동일), `b2_inference_environment.pyproject_sha256`은 실행 checkout 파일과 같아야 하며(NOTE-C), `processed_frame_aligned`·`smoke_train_subset`은 기록용으로 내려갔다. protocol 문자열·추가 pin 9개·`identity_manifest_path`·`a2_results_file.path`(저장소 상대)·`a3_weight_settings` 자료형 미검사는 그대로다.

## 5. 변이 재확인 (스크래치 사본, 30개: 24 검출 · 6 생존)
- **1차 생존 7개 전부 검출:** `lg.submit` PA 끝 가드, 라벨 1/2 우선순위, S-2′ 예산 분할 가드, γ_a 불변식, `run_record.valid`의 불변식 항, B2 0비 불변, B2 2026 행 가드.
- **MAJOR-1 가드 5개 검출:** 고정 행을 공개로 되돌림, `I_ROW`에 25행 허용, 봉인 항목 미생성, UB에서 미공개, 공개 `counts` 25. **MAJOR-2 1개:** 사이드카 건너뜀 삭제. **MAJOR-3 4개:** S4 상류 요구 복원, 공통 단계 종결 삭제, R3 소진을 종결로 만듦, 2단계 소진 종결 삭제.
- **MAJOR-4·MINOR 연결선:** 검출 — 경로 A 강제, 비교의 누락/추가 무시, 재생 경기 하나로 축소, 런타임 pin 비교 삭제, G-INT-R3 6키 필수 삭제, support 차이를 무효로 복원, `select`의 lock 삭제. **생존 6개:** D1 `require_equivalence` 호출 삭제, D2 경로 B 신원 검사 삭제, D4 쌍 sha 검사 삭제(중복), E2 pyproject 검사 삭제, E3 `validate_key_record(gint_a2)` 삭제, E7 `TAB_GROUPS` 단언 삭제 → MINOR-A(D4·E7은 무시 가능).
- 작성자 보고(19/19 검출)와 모순 없음: 내가 돌린 1차 생존 7개 + 수정 가드 전부 검출.

## 6. v2.4 패치 목록에서 빠진 코드-설계 차이
1. **공통 단계 범위:** 목록은 "SM·S0·L·S3"이나 코드는 **G도 포함**하고, **2단계 소진**도 게이트 기록과 함께 종결 처리한다(`run_ope_a2b.py:395,433-`). 종결 기록 필드: `stage_status`·`failed_integrity`·`blocked`·`not_run`·`phase_2`·`gate_record_sha256`.
2. **경로 B:** 목록의 "새 DEV 런타임에 같은 요청 두 번"은 코드에서 **같은 신원의 런타임 2개**(각자 RowBudget·원장; 한 런타임은 같은 `request_id`를 캐시해 불가)이고, 두 sha가 같고 새 원장 런타임 sha와 같아야 한다.
3. **공개 기록 필드:** `ledger_record.json['driver_equivalence']`에 `style_source`·`runtime_sha256_reproduced`·`s6_runtime_sha256`·`reason`·`tolerance`·불일치 키 이름(최대 20개)·`registered_pins_compared`가 더 있다(NOTE-A).
4. **기록용 pin:** 코드의 `RECORD_ONLY_PINS`는 목록의 4개 외에 `smoke_train_subset`·`processed_frame_aligned`(sha 필드)를 포함하고, `ml_policy_d127_chain`·`s2_prime_runs`는 등록 시점 동일성만(NOTE-D). DEV 프레임 신원의 실제 검증 pin은 `processed_frame_identity`(Q1).
5. **B2 무효 조건:** 목록의 "설계 그대로"에 누락/형식 오류 벡터도 무효라는 점을 덧붙일 것.
6. **UB 공개:** `unblinded.json`의 `lambda_fixed_row_public`에 `equals_R1_bxR2_1xB0`(λ=100이 선택됐는지) 플래그가 들어간다(판정 입력 아님, UB 때 공개).
7. **pyproject:** A2b-S3:R3가 실행 checkout의 `pyproject.toml` sha를 pin과 비교(NOTE-C). 

## 7. G-REG 전 필수와 Codex 범위
- **G-REG 전 필수:** (1) v2.4 완성 뒤 코드–설계 일치 확인(별도 요청; §6 7항목 반영 여부 포함). (2) 마감 커밋에서 `run_ope_a2b.py pins`로 `new_files_sha256` 16개 재계산(이번 11개 커밋으로 모두 바뀜). (3) 1차 §5의 등록본 설정(protocol `exp_ope_res_02_v1`, 추가 pin 9개+`identity_manifest_path`, 허용 목록 단일값화, 결과 파일 커밋·재pin, 역할 9를 JSON `seeds.roles`에) + 새로 생긴 것: `runtime_setting`·`dev_style_snapshot.{path,sha256}`는 등록 사슬 값과 정확히 같게, `pyproject_sha256`은 실행 checkout 기준(NOTE-C). (4) 권고(비차단): MINOR-A 테스트 3개.
- **Codex 재확인(11/02 18:57 KST 이후) 범위:** 1차 §6 목록 중 이번에 바뀐 부분으로 좁힌다 — `ledger_a2b.driver_equivalence`·`require_runtime_pins`·`check_no_reward_outputs`, 러너 `plan_decide`/`body_ledger`/`body_unblind`/`select_command`, `ope_sealed_a2b`의 `held`/`lambda_fixed_row_public`, 역할 9, `validate_key_record`, B2 유효성·G-INT-R3, 그리고 체인 테스트의 대체 훅 경계(실제 A2 `Environment` 읽기 경로).

## 8. 운영 기록
- **worktree 무변경:** HEAD `53df4e22…` 확인, 시작·끝 `git status --short --ignored` 빈 출력, `__pycache__` 0건. 모든 실행은 `python -B`·`PYTHONDONTWRITEBYTECODE=1`·`-p no:cacheprovider`. 변이는 `git archive` 사본(작은 로컬 git 하나 초기화; 삭제함)에서만 했다.
- **금지 구역:** `$PITCHEEZY_DATA_DIR`·`/Volumes/T7 Shield`·`runs/`·S6/DEV/TRAIN 파일과 설계 md/JSON 미열람. 메인 checkout에서는 `git diff -- pyproject.toml`과 파일 sha256만 읽었다.
- **`df -g /`(끝):** 228 GB 중 사용 11, 가용 31, 28 %(1차와 같음).
- **스크래치(`.../scratchpad/gcode-a2b/`, 약 128 KB, 12 MB 사본 `mut/` 삭제):** 이번 라운드 신규 — `make_muts3.py`·`muts3.json`·`muts3_out.txt`(변이 30), `probe2_lambda.py`(§2-1)·`probe2_decide.py`·`probe2_attempts.py`(§2-3)·`probe2_reward_free.py`(§2-4 i)·`probe2_wires.py`(§2-4 iv, MINOR-A); 1차 파일은 그대로. 재현은 `git archive 53df4e22 experiments/pitchmdp configs pyproject.toml scripts src`로 `mut/`을 다시 만들고 `git init` 후 `mut.py muts3.json`.

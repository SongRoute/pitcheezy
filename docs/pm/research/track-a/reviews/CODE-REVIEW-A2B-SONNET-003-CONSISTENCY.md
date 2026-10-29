VERDICT: PASS_WITH_NOTES

# CODE-REVIEW-A2B-SONNET-003-CONSISTENCY — EXP-OPE-RES-02(트랙 A2b) 설계 v2.4(패치 2) ↔ 마감 커밋 3dce62d9 최종 일치 확인

- **검토자 모델:** Claude Sonnet(검토자) (claude-sonnet-5-5). 작성자(Claude Opus 5.5)와 다른 모델. 읽기 전용(코드·설계 읽기, 합성 테스트, 스크래치 사본 변이). 앞선 검토: `CODE-REVIEW-A2B-SONNET-001.md`(FAIL), `…-002-CLOSURE.md`(PASS_WITH_NOTES).
- **시각:** 2026-10-28 22:37 KST. **대상:** `/Users/song/Projects/pitcheezy-worktrees/track-a-a2b-1028`, HEAD = `3dce62d941304b33ffd145fa656d1c7796393ccb`(`git rev-parse` 확인; 시작·끝 `git status --short --ignored` 빈 출력, `__pycache__` 0). `53df4e22` 위 커밋 2개: `88718bc1`(A2b-L 기록 검사기·pin 세 분류), `3dce62d9`(연결선 테스트 8개, 체인 벽시계 단언 삭제); 6개 파일 +541/−49.
- **설계 입력(열기 전에 `shasum -a 256`, 기대값과 일치):** md `9c94c6ca1cf0e4763093962e3984451997df3a424e189360632a94d893d75a01`(580줄), 초안 JSON `69855e6405cfe996bfdfbad3991673ec4bc47c0989a7137687a14b4bec0d60ea`(3,472줄).
- **기존 파일 수정 0:** `git diff --diff-filter=M --name-only b63369d0..HEAD` 출력 없음, `--name-status`는 `A` 16건뿐. `run_ope_a2b.py pins`의 `new_files_sha256` 16개 = `git show 3dce62d9:<경로> | shasum -a 256` 16개(전부 일치).
- **테스트(직접):** `python -B -m pytest tests/test_ope_a2b_*.py -q -p no:cacheprovider` → **115 passed, 161 subtests passed, 26.71 s**(스크래치 사본 25.02 s·25.00 s). `/usr/bin/time -l`: maximum resident set size 487,129,088 B(0.454 GiB), peak memory footprint 354,518,072 B(0.330 GiB). PM 수치(115·161·25.0 s·"RSS 0.33")와 일치하나 0.33은 peak memory footprint다(NOTE-3).
- **개수:** BLOCKER 0 · MAJOR 0 · **MINOR 1**(JSON 가드 키) · NOTE 4. 코드 결함 없음. 2차 잔여 항목 전부 CLOSED. MINOR-1은 JSON 한 줄 수정(G-REG 전)이고 G-CODE 종결을 막지 않는다.

## 1. 2차 잔여 항목 판정
| 2차 항목 | 판정 | 근거 |
|---|---|---|
| MINOR-A 러너 fail-closed 연결선 | **CLOSED** | `FailClosedWiringTests` 8개(`test_ope_a2b_runner.py:946-`). 기준선 115 통과 확인 뒤 연결선 7개를 하나씩 지움 → 7/7 검출, 각각 정확히 테스트 1개만 실패(1:1, §5). 8번째 `test_the_real_path_b_seals_the_ledger_stage`는 양성 대조(신원 검사 반전 변이가 이 테스트도 실패시킴) |
| NOTE-A A2b-L 기록 형식 | **CLOSED** | `validate_equivalence_record`(`ope_registry_a2b.py:576`, 19키)·`validate_ledger_record`(:614, 12키)를 쓰기 전에 호출(`run_ope_a2b.py:636,664`). 키 집합이 md §9.2·JSON `record_schema`와 글자 단위로 같음(§2 #4). 세부 제약의 테스트 보호는 일부만(NOTE-1) |
| NOTE-D pin 분류 | **CLOSED** | `COMPARED_AT_RUN` 27·`REGISTRATION_TIME_PINS` 3·`RECORD_ONLY_PINS` 11 = `PIN_KEYS` 41(`:765-816`). 합집합·겹침 0은 import 시 assert 2개(:815-816)·`PinClassTests`·`pins` 출력이 보장; 겹침·누락 변이는 import에서 AssertionError(2/2). 코드·JSON `pin_checks_v2_4`·md §9.2 세 목록이 집합으로 일치(스크립트 대조) |
| NOTE-E 체인 벽시계 단언 | **CLOSED** | 테스트 코드에 `perf_counter`·`assertLess(time…)` 0건; "약 6초"는 독스트링에만 |
| NOTE-B·C·F(설계 쪽) | **CLOSED** | md §9.2(RSS 증가분 미측정)·§9.6(실행 checkout; 워크트리 pyproject `5eb30162…` 대 메인 `cb614006…`를 `shasum`으로 재확인)·§13(G0 결정성·경로 A 실 S6 미확인) |

## 2. 설계 v2.4 ↔ 코드 3dce62d9 일치표
| # | 설계 | 코드 | 판정 |
|---|---|---|---|
| 1 | 2차 §6의 빠진 7건(공통 단계에 G·2단계 소진·종결 필드 / 경로 B 런타임 2개 / 공개 기록 필드 / 기록용 pin / B2 무효 조건 / UB 플래그 / pyproject) | md §8.2·§9.2·§4.1·§3·§9.5·§9.6에 모두 있고 아래 항목과 일치 | 일치 |
| 2 | 1차 (a)·(c)1~11: 로그 단계 포함 A2 순서(§5.2), `rpv.prerequisites` 원본 호출(§9.1), 역할 9(§9.4), `A2b-S3:R3`(§7·§9.3), G-INT-S2EQ 명령·`paths`·`failure_scope`(§9.3), κ_p′=봉인 A2-S0 TAB(§5.2), R5 규칙(§4.2), κ_p′ 비용 게이트(§9.6), SM 범위(§9.3), `select` 예외(§9.6) | `S2P_PATH` registry:215·`ope_synthetic_a2b.py:334-341`; `run_ope_a2b.py:624`; `ROLES` :118; `R3_STAGE` :74; `kappa_prime_from_s0` :569; `s2p_cost_gate` registry:363(2,169.5×2); `body_smoke` :532; `select_command` :1039 | 일치 |
| 3 | §3·§4.3·§9.5: 25행 계산, 공개 24행, λ=100 고정 행은 `sealed.lambda_fixed_row_public`(`equals_R1_bxR2_1xB0`)에만 쓰고 A2b-UB에서 `unblinded.json`으로 공개 | `PUBLIC_ROWS` 24·`SEALED_UNTIL_UNBLIND_ROWS` 1(registry:75-76), `ope_sealed_a2b.py:439,447`, `body_unblind` :962 | 일치 |
| 4 | §9.2 기록 형식: `driver_equivalence.json` 19키, `ledger_record.json` 12키, 하위(sealed_counts·dates·identity·m10·pins·summary) 제약 | 상수 :538-558·검사기 :576-670. 이름 집합이 md·JSON과 일치(순서만 다름); 제약 문구(경로-사유-재현-게임 수, pass=개수, 20개 이름, 1e-12 등) 일치 | 일치(하위 제약 일부는 설계가 덜 적음, NOTE-2) |
| 5 | §9.2 경로 A/B: 첫·끝 S6 경기, 헤더 sha 재현 때만 비교, 실수 절대 1e-12·그 밖 정확, 집합 동일; 경로 B 런타임 2개·상호 sha 동일·새 원장 sha와 동일; 쓰기 전 검사 | `ledger_a2b.py:345-403`(:401 쌍 sha), `run_ope_a2b.py:636-643`; 실패 기록을 쓴 뒤 pass 요구(테스트 확인) | 일치 |
| 6 | §8.2 종결: SM·S0·L·S3·G 소진=게이트 기록 없이 종결, 2단계 소진=게이트 기록과 함께, R3 소진=B2 행만, 필드 6개(`gate_record_sha256` null) | `COMMON_STAGES` :395, `plan_decide` :433-463, `body_decide` :938-942, R3 :482-487 | 일치 |
| 7 | §9.3·§7: G-INT-EVAL 12행, G-INT-S2EQ 정확 형태, 2단계 작업 목록·TAB ±20% | `ope_decision_a2b.gate_record`(work_list), `tab_reuse` registry:247, 테스트 존재 | 일치 |
| 8 | §9.6·§7 값: hang guard 10개(7,200·14,400·3,600·600·1,800…), 433.9×N′=4,339/23,864.5/2,169.5, 비용 게이트 2배, RSS 8·여유 4 GiB, 창 13:30–06:30 | `HANG_GUARDS` registry:340, `S5X_RUN_SECONDS` :349, `s2p_cost_gate` :363, `RSS_CAP_GIB, FREE_MEMORY_GIB` | 코드 일치. **JSON `hang_guards_seconds`에 코드에 없는 키 `A2b-S3:R3`** → MINOR-1 |
| 9 | §9.4 역할 0~9 값 | numpy 2.5.3 재계산 = 코드 = md = JSON 10/10, 세계 시드 76개 | 일치 |
| 10 | §9.2 pin 3분류와 사용처 | 27개 중 사용처 확인(A2 상속 `output_root` run_ope_a2.py:591·`processed_frame_identity` :1017·`policy_registration_chain` :1747), 기록용 11개는 비교 코드 없음, 등록 시점 3개는 `check_pins`(registry:923-979) | 일치 |
| 11 | §9.1 테스트: 8개 연결선 테스트, 체인 벽시계 단언 삭제, 115·161 | §1 | 일치 |
| 12 | §9.6 운영: AppleDouble 출력 열거는 한 곳 | `rglob` 1곳(`ledger_a2b.py:433`, `is_appledouble` 건너뜀) | 일치 |
| 13 | §13 한계: ∞ 선택 노출 수용, G0 결정성·경로 A 실 S6·RSS 미측정 | 코드 변화 없음(`test_lambda_inf_identity`), 문서 한계 | 일치 |
| 14 | JSON `config_check` 시험(메모리에서): 기대 30개 중 실패 `$.protocol` 1개, 역할 10/10·라벨·단계 id 통과, 가드 불일치, pin 키 32 대 41 | 프로토콜·pin 9개는 G-REG, 가드 키는 MINOR-1. 세 가지를 메모리에서 고치면 `DRAFT_UNREGISTERED`로 통과 | 아래 §3 |

## 3. 불일치 목록
| ID | 설계 위치 | 코드 위치 | 심각도 | 고칠 쪽 |
|---|---|---|---|---|
| MINOR-1 | JSON `cost_budget.hang_guards_seconds["A2b-S3:R3"]`(J:2626, 값 3600), md §9.6 "하위 단계 자체 가드" | `HANG_GUARDS`/`GUARD_OF` registry:340-343에 키 없음(러너 :104가 `A2b-S3`의 3,600초로 매핑), `config_check` :891이 사전 동일성을 요구 → 등록본이 "hang guards deviate from the registry"로 거절됨(`chk_config3.py`) | MINOR | **설계(JSON, PM):** 키 삭제 — 값은 `stages[A2b-S3].sub_stages[0].hang_guard_seconds`에 이미 있다. 코드에 키를 더하는 안은 레지스트리 수정이라 `new_files_sha256` 재pin·재검토가 붙어 비추. PM의 `g_reg_todo_v2_4`에 없음 |
| NOTE-1 | md §9.2 검사기 제약 | registry:576-670. 검사기 완화 변이 59개 중 22개 생존(1개 동치): `check` 값, 게임>0, 경로 B `runtime_sha256` hex, `max_abs_delta`≥0, `sealed_counts`, 날짜 형식, 최상위 sha hex, `candidate_sha256`·`m10.check`·`m10.runtime_sha256`, 핀 키 집합, `planning_seed`, 스타일 키·sha, `summary` 필수 키·정수·`run_status`·head hex 등 | NOTE | 테스트(선택): 음성 사례 추가. 같은 라운드의 연결선 변이 7개는 생존 0 |
| NOTE-2 | md §9.2·JSON `record_schema`가 `identity.checks` 5개 이름, `identity.runtime_sha256`=최상위 sha, `m10.runtime_sha256` hex64를 적지 않음; JSON `stages[A2b-L].outputs`에 `selection.json`·`facts.reward_free.json` 없음; `implementation.new_files[ope_registry_a2b.py]`에 새 검사기·pin 분류 없음 | registry:551,639,646; `run_ope_a2b.py:658-659`(R3 단계가 읽음 :722-723) | NOTE | 설계(코드가 정본이라 비차단) |
| NOTE-3 | md §9.1·§10, JSON `code_closing_commit.tests` "최대 RSS 0.33 GiB" | 측정: maximum resident set size 0.454 GiB, peak memory footprint 0.330 GiB | NOTE | 설계 문구("최대 메모리 사용량 0.33 GiB") |
| NOTE-4 | md §7 산수 "76회 × 433.9초 = 32,977.6"(합계 38,561.5초) | 76×433.9=32,976.4(합계 38,560.3초); 10.71시간·판정에 영향 없음 | NOTE | 설계(1.2초) |

## 4. 설계가 코드보다 더/덜 약속하는 곳
- **더(설계가 코드보다 앞서는 것):** (1) JSON 가드 표의 `A2b-S3:R3` 키 — 코드는 별도 항목이 아니라 `A2b-S3` 값을 쓴다(MINOR-1). (2) 설계 문구 "RSS 0.33"은 측정 항목 이름이 다르다(NOTE-3). 그 밖에 설계가 약속하고 코드에 없는 동작은 찾지 못했다(§2 #1~13을 항목별로 대조).
- **덜(코드가 설계보다 엄격하거나 더 한다):** 검사기가 `identity.runtime_sha256`=최상위 sha, `m10.runtime_sha256` hex64, `summary` 정수·상태 개수 형식, 정확한 `identity.checks` 이름 5개를 요구한다(md·JSON에 없음, NOTE-2). 실패한 동등성 기록도 쓴 뒤에 pass를 요구한다(md는 "불일치는 결정적 무결성 실패"만). A2b-L이 `selection.json`·`facts.reward_free.json`(보상 없음)을 쓴다.

## 5. 변이 결과 (스크래치 사본, 기준선 115 통과 확인 뒤, 매번 원복·원복 뒤 115 통과 재확인)
- **연결선 제거 7개: 7/7 검출, 정확히 1개 테스트만 실패.** `require_equivalence`→`…ledger_stage_never_seals`, 경로 B 신원→`…path_b_must_have_run…`, pyproject pin→`…pyproject_mismatch…`, `validate_key_record` gint_r3·gint_a2→각 테스트, 새 검사기 2곳→각 테스트. 추가 변이 3개: 경로 B 신원 반전(대응 테스트 + 양성 대조 2개 실패)과 pyproject 반전(3개 실패)은 검출; 신원 검사를 경로 A에도 거는 변이(W8)는 생존 — 런타임 sha가 경기 선택과 무관해(합성 체인에서 S6 경기 200·203과 새 경기 201·202의 sha가 같은 `431d38aa…`, `probe3_sha.py`) 동치 변이로 본다.
- **검사기 완화 59개: 37 검출 · 22 생존**(NOTE-1; E19는 동치). pin 분류 assert 2개: 겹침·누락 모두 import 오류로 검출(2/2). 앞선 수치(1차 34/43, 2차 24/30)는 그대로. 작성자 보고의 31/31(기준선 확인 후)은 검토자가 재현하지 않았다.

## 6. G-REG 전용 목록 확인(결함 아님)
- 추가 pin 9개(`output_root`·`a2_digest_manifest_path`·`a2_results_file`·`a3_freeze_primary_manifest_path`·`a3_s6_weights_record`·`a3_weight_settings`·`policy_registration_chain`·`processed_frame_identity`·`train_raw_files_sha256`; JSON은 32개)와 `processed_frame_aligned.identity_manifest_path`(JSON 4키), protocol `exp_ope_res_02_v1`(JSON `…registration_draft_v2_4`), 허용 목록 단일값화(현재 3·6개), `new_files_sha256_scope` 문장에 16개 명시(현재 "items 1-7 and the new tests"), `new_files_sha256` 16개(3dce62d9 기준 값은 `pins` 출력), 결과 파일 `results/EXP-OPE-RES-01.json`(미추적 `??`, sha256 `069bcd6b…` = 초안 pin) 커밋 뒤 재pin, pyproject pin = 실행 checkout 기준, `runtime_setting`·`dev_style_snapshot.{path,sha256}`를 등록 사슬 값과 타입까지 같게, S6 헤더 `style_report` dry check, 등록 커밋에 연결선 테스트 8개 포함. **여기에 MINOR-1(JSON 가드 키)을 더할 것.**

## 7. Codex 재확인 범위 제안 (2026-11-02 18:57 KST 뒤, `3dce62d9` 기준; 2차 §7을 갱신)
- 정독 우선순위: (1) 맹검 경계 — `PUBLIC_SCHEMA`·`validate_public`·`PUBLIC_ROWS`, `ope_sealed_a2b.evaluate`의 public/sealed 분리·`lambda_fixed_row_public`, `validate_integrity_record`·`validate_key_record`, UB 공개; (2) `ope_synthetic_a2b.run_s2_prime` ↔ A2 `run_s5x` 줄 단위(`scheme_seeds`·`pool_cell`·로그 단계 순서)·G-INT-S2EQ; (3) 러너 상태기계 — `plan_*`·`attempt_stage`·`UPSTREAM`·`plan_decide` 종결 두 경로·R3 범위; (4) `ledger_a2b`의 `candidate/bind/snapshot_for` ↔ `run_policy_validation.dispatch` 닫힌 함수 3개, `driver_equivalence`·`require_runtime_pins`와 이번에 정본이 된 두 검사기; (5) B2 — `b2_validity`·`b2_decision_stream`·`gint_r3`·`load_frozen_b2`; (6) 채워진 등록본 대 `CONFIG_EXPECTATIONS`/`check_pins`/`config_check`(가드 키 포함); (7) 합성 체인이 대체한 실제 훅 — A2 `Environment`의 `a2_sealed`·`a2_results`·`s6_record`·`rehearsal_inputs`·`raw_join`·`policy_inputs`(`verify_processed_pin`), `b2_dev_table`.
- 다시 찾지 않아도 되는 것: 1차 MAJOR 4개, 2차 MINOR-A·NOTE-A·D·E(위 변이 기록). 직접 재현은 `chk_config3.py`(§3)·`probe3_sha.py`·`mut3.py muts4.json`을 권한다.

## 8. 운영 기록
- **worktree 무변경:** HEAD `3dce62d9` 시작·끝 `git status --short --ignored` 빈 출력, `__pycache__` 0. 모든 실행은 `python -B`·`PYTHONDONTWRITEBYTECODE=1`·`-p no:cacheprovider`. 변이는 `git archive` 사본(작은 로컬 git, 삭제함)에서만 했다. 메인 checkout에서는 설계 파일 읽기, `shasum`, `git ls-files`·`git status`(읽기)만 했다.
- **금지 구역:** `$PITCHEEZY_DATA_DIR`·`/Volumes/T7 Shield`·`runs/`·S6/DEV/TRAIN 파일·2026 파일 미열람. 설계 md/JSON은 sha256 확인 뒤 열었다.
- **`df -g /`(끝):** 228 GB 중 사용 11, 가용 30, 28 %(2차 때 가용 31; 이번 라운드 스크래치 사본은 최대 12 MB였고 삭제함).
- **스크래치(`.../scratchpad/gcode-a2b/`, 약 236 KB: 소형 스크립트·출력만, 12 MB 사본 `mut/` 삭제):** 이번 라운드 신규 `direct_tests3.txt`·`mut3.py`·`muts4~8.json`·`muts4~7_out.txt`·`make_muts5.py`·`chk_pins3.py`·`chk_keys3.py`·`chk_config3.py`·`chk_config4.py`·`chk_seeds3.py`·`chk_pin_usage.py`·`probe3_sha.py`. 재현: `git archive 3dce62d9 experiments/pitchmdp configs pyproject.toml scripts src`로 `mut/`을 만들고 `git init` 후 `mut3.py muts4.json`.

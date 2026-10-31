# COOP-018 코드 검토 기록 (a704983 → 0135379)

2026-09-29. Opus 5.5 단독. **같은 모델의 적대적 검토이며 독립 검토(Sol·Astra)가 아니다.** 합성 자료만 사용했고 실데이터·payload·2026 접근은 0이다.

## 1차 검토: a704983 (WIP 코드)

워크플로 `coop018-code-review`: 여섯 렌즈(추정기, 요청 생성기, 런타임, 실행기, 준합성, 식별자·테스트)가 결함을 찾고, 결함마다 세 검증자(재현 / 명세 대조 / 반박)가 판정했다. 에이전트 127개. **확인 36건**(검증자 2/3 이상), 기각 4건, 누락 점검 12항목.

### 확인된 결함과 조치

| # | 파일:행 | 결함(요약) | 확인 표 | 조치(0135379) |
|---|---|---|---|---|
| C0 | `policy_estimator.py:150` | Worst-case bounds and the conditional mean are biased when a PA becomes non-evaluable after decision 0 | 3/3 | estimator v2: censored node value inside the recursion (affine base/slope), L0/L1/L2 layers (D-5 C); regression test with the reviewers' counterexample and TOY action-dependent censoring |
| C1 | `run_policy_validation.py:273` | PAs that pa_requests cannot submit are left out of the population-bound denominator | 3/3 | run_dr/submit_pas keep every PA; UNSUBMITTABLE status in L0 (M-8) |
| C2 | `policy_semisynthetic.py:80` | V2 'refuse' rule drops individual refused logs and compares selection-biased DR means with the truth | 3/3 | D-10 absorb_off_mask: no log is ever dropped; refuse rule removed |
| C3 | `policy_estimator.py:90` | estimate() still aggregates a halted ledger, treating FAILED_INTEGRITY and FAILED_RUNTIME PAs as ordinary worst-case slots | 2/3 | estimate() raises on any FAILED_* decision (halted ledger never aggregated) |
| C4 | `policy_requests.py:57` | pa_requests drops the whole PA, earlier decisions included, because of a later row's count or identity (R4c/R1) | 3/3 | pa_requests submits rows before the defect and returns its index (prefix + censor at k) |
| C5 | `run_policy_validation.py:273` | Unsubmittable PAs are left out of the DR population denominator and the [0,1] worst-case bounds | 3/3 | same as the unsubmittable fix (L0 denominator) |
| C6 | `policy_requests.py:141` | S0 census checks codes against the G0 token vocabulary, not the BC vocabulary that the runtime's R3(b) integrity check enforces | 2/3 | census adds codes_outside_train_labels (BC vocabulary proxy at S0; the BC does not exist yet) |
| C7 | `policy_requests.py:58` | A missing mid-PA row (pitch_number gap) is not detected; the request gets a silently truncated 'full' history | 2/3 | pitch_number gap / first row not pitch 1 -> 'missing_row' at that index |
| C8 | `policy_runtime.py:210` | New data refusals run before the D92 context/support integrity check, so a malformed candidate-mode request is recorded as a legitimate refusal and the run does not halt | 2/3 | check_context() runs before any data refusal in candidate mode (FAILED_INTEGRITY) |
| C9 | `policy_runtime.py:47` | An 'unknown' previous outcome disables the count check entirely, so impossible transitions are accepted as SUPPORTED | 3/3 | uncheckable previous outcome of a real pitch -> UNSUPPORTED_INCONSISTENT_HISTORY |
| C10 | `policy_runtime.py:233` | The candidate's MC Q search runs before the LOGGING_POSITIVITY refusal, so legitimate refusals spend the row budget and can be recorded as FAILED_RUNTIME | 3/3 | LOGGING_POSITIVITY decided before the candidate search (test: candidate not called) |
| C11 | `run_policy_validation.py:444` | Full data/pickle load runs before the heavy lock, output validation and stage record | 3/3 | dispatch(load=...) loads inside heavy_lock + stage; census never builds the store |
| C12 | `run_policy_validation.py:273` | DR population bounds silently drop unsubmittable PAs | 3/3 | same as the unsubmittable fix |
| C13 | `run_policy_validation.py:390` | Failed S2 connection probe is sealed as a successful stage; downstream stages never check it | 3/3 | bind-probe raises after dumping probe.json when pass is false; later stages require a pinned passing probe record |
| C14 | `run_policy_validation.py:364` | Registered stage caps are not enforced for S0-S2, and only partially for S3/S4/V4 | 3/3 | D-11 D: hang guard (SIGALRM) on every stage + RowBudget for rollouts; manifest records measured cost |
| C15 | `run_policy_validation.py:70` | Source-commit, clean-tree and member-loader pins are recorded but never enforced | 3/3 | enforce_source(): HEAD == registered source_commit, clean tree, member loader SHA (CLI path) |
| C16 | `run_policy_validation.py:239` | Stage wall-cap expiry leaves the ledger looking like a clean OK run (no FAILED_RUNTIME) | 3/3 | submit_pas aborts the ledger (aborted row -> HALTED) on BudgetExceeded / hang guard |
| C17 | `run_policy_validation.py:345` | --output is not confined to the registered ML-POLICY-VAL-v1 root; a stage can be created inside sealed run/stage directories | 3/3 | check_output(): stage directory directly under the registered output_root |
| C18 | `run_policy_validation.py:369` | Registered BC is not bound to decision D-1 (artifact provenance does not record the selection rule) | 2/3 | BC provenance source_ids carry population_rule_sha256 and ordered_row_keys_sha256; materialize.json records the rule |
| C19 | `run_policy_validation.py:349` | Recorded config SHA is taken after the long load, not from the parsed bytes; other pins are hash-then-reopen | 3/3 | config SHA from the bytes read once (load_registration); pinned_bytes() for npz/parquet/json inputs |
| C20 | `run_policy_validation.py:408` | D-7 snapshot date is not validated against the DEV window start; profile ignores D-7 | 2/3 | snapshot_for(): as-of == registration, blocks not before as-of, bit-equality with rolling on the as-of date; profile uses the May snapshot |
| C21 | `run_policy_validation.py:393` | dispatch falls through to dr-evaluate for any unrecognised command | 2/3 | registration(command) rejects unknown commands; explicit branches |
| C22 | `policy_semisynthetic.py:80` | World-refused logs are silently dropped, so per-start DR is compared with unconditioned truth (post-treatment selection) | 3/3 | D-10 absorb_off_mask + excluded starts reported (no silent drop) |
| C23 | `test_policy_validation.py:479` | V2 test is vacuous: it passes with estimators that ignore the policies, renormalise pi_b onto the mask, or swap candidate and reference | 3/3 | discriminating V2 world (TypedNetwork): |truth delta| > 4 s.e. of the gap; absorbing-value invariance test |
| C24 | `policy_semisynthetic.py:92` | Truth is computed for every start, so a start the runtime legitimately refuses crashes the whole V2/V3 run | 3/3 | truth computed only for starts in E0 with logs; excluded starts reported |
| C25 | `policy_semisynthetic.py:96` | Truth RNG for start 0 is the same stream as log generation (seed + 0 == seed) | 2/3 | per-PA substreams SeedSequence([seed,0,i,j]); truth SeedSequence([seed,1,i]) |
| C26 | `policy_semisynthetic.py:52` | generate_pa does not validate the generating law, so an unnormalised V3 law silently changes the logging distribution | 3/3 | _law() validates the generating law (probability vector, inside pi_b_hat > 0) |
| C27 | `policy_identity.py:503` | Support table drops the throwing hand, so a DEV (pitcher, hand) unseen in TRAIN halts V4 but passes V5 as SUPPORTED | 3/3 | D-8 hand registry + PITCHER_HAND refusal in both modes; hand in requests |
| C28 | `run_policy_validation.py:73` | Request generator, reward and DR estimator code is pinned by neither the policy identity nor the stage manifest; a dirty tree is accepted | 3/3 | stage identity records runner_sources hashes; clean tree enforced |
| C29 | `test_policy_identity.py:99` | Fake WE ignores defender_is_home, so a wrong defender in defense_we or pa_outcome goes undetected | 2/3 | fake WE depends on defender_is_home; reward test checks the initial defender |
| C30 | `test_policy_validation.py:349` | Two-hand support logic (refuse, conflict detection, per-hand templates) is never exercised by any test | 2/3 | tests: hand registry refusals, AMBIGUOUS templates, registry validation |
| C31 | `run_policy_validation.py:417` | The identity verified in S2 is never tied to the later stages, and registered pins are not enforced | 3/3 | expected_identity_sha256 from the addendum chain; tau-select records final_identity_sha256 |
| C32 | `policy_identity.py:501` | Hand conflicts are computed over all TRAIN pitchers before the BC-pitcher filter | 3/3 | hand registry built for BC pitchers only |
| C33 | `run_policy_validation.py:273` | Unsubmittable PAs are left out of population_delta_bounds | 3/3 | same as the unsubmittable fix |
| C34 | `run_policy_validation.py:285` | The runner hashes a file and then re-opens it (TOCTOU), unlike the same-bytes pin rule in §2 | 3/3 | pinned_bytes(): hash and parse the same bytes |
| C35 | `run_policy_validation.py:444` | The heavy input load runs outside the heavy lock and before any stage record exists | 3/3 | same as the load-order fix |

### 기각된 주장

| 파일:행 | 결함 주장 | 확인 표 |
|---|---|---|
| `policy_estimator.py:148` | Per-policy population bounds required by R6 and the module docstring are not reported | 0/3 |
| `policy_requests.py:90` | pa_outcome assumes a game's rows are contiguous; a game split in frame order gets its partial score as the 'final result' | 1/3 |
| `policy_runtime.py:214` | Count-path check trusts the caller's history[-1] count instead of the ledger's previous row, so it can hide a count jump (SUPPORTED) or turn a history/ledger mismatch into a refusal | 1/3 |
| `policy_semisynthetic.py:113` | V2/V3 report records neither seed, cap, truth size nor the generating law, so V2 and V3 results are indistinguishable and not reproducible | 1/3 |

### 누락 점검(completeness) — 모두 D93 코드에서 다룸

- **Config fields the runner reads vs the committed config (cross-file schema)** — The runner reads keys the config does not have, or names them differently: `decisions` (config only has the prose `open_decisions`, config:367), `le2025_validation_plan.local_config.sha256` (runner:441), `stages.S2_bind_probe.rows` (config has `probe_rows`, a prose string, config:297; runner:385/390), `stages.V4_dr_evaluate.cap_seconds` (config names the stage `S6_V4`, config:320; runner:426), `pr
- **BC fit parameters and other plan fields are not taken from the config** — `train_bc_plan.bc_parameters` (prior_strength 20.0, minimum_action_count 1; config:199) is never read. run_materialize calls `export_train_bc(store, rows, path, **provenance)` and falls back to that function's hard-coded defaults. If the registered values change, the BC is still fitted with 20/1 and nothing reports it. This breaks the CLAUDE.md rule that parameters come only from configs/. Other p
- **S1 materialize outputs vs contract §3 and train_bc_plan.reports** — The contract requires a 'BC_P vs BC_E cell differences' report (config train_bc_plan.reports[1], contract §3). The runner never computes it: materialize.json has only per-BC identities, the support table and the mass report. The P8 reproduction check for BC_E ('must match P8 bc_train_pitches and 18 types') is not enforced, because the BC_E vocabulary size is never compared with 18. Output names al
- **S3 profile stage vs contract S3 / D-9 ordering** — Contract §5 S3 says to measure the G0 5-member P2/P3 decision cost, and to profile P2 only when τ is absent. S3 is what informs the search/budget registration (D-9). The runner instead refuses `profile` unless tau, samples, pitch_cap, planning_seed and budget are all registered (runner:401-402), and it only ever profiles the P3 candidate. So S3 cannot run before D-9 is decided, P2 cost is never me
- **S5 V2/V3 wiring and the D-10 world rule** — `run_world` can only be called from tests. COMMANDS (runner:42) has no V2/V3 stage, so there is no heavy lock, fresh directory, manifest or cap for it. D-10 is missing from DECISION_VALUES (runner:44-48), so the world rule ('fallback'/'refuse') is a free call argument rather than a registered decision. No code selects the registered V2 PA starts ('DEV PA start contexts'). `dr-evaluate` (S6/V4) can
- **S0 census read scope vs its label-blind claim** — Contract S0 says it reads only pre-decision fields, pitch_type, the automatic flag and whether events is present. For `census`, load_inputs still builds MatrixHistoryStore over the whole frame (runner:313). That computes `outcome_labels` from description/events for every split including DEV, normalises current-pitch physics, and unpickles aux.pkl. The stage then writes `label_blind: True` (runner:
- **R1 denominator completeness: PAs with no rows at all** — pa_blocks builds the PA universe only from rows that exist, and nothing checks at_bat_number continuity within a game. Reproduced in scratchpad/review/gap_r1.py: after deleting every row of DEV PA 200:2 from the test fixture, pa_blocks and census report 13 DEV PAs instead of 14, with no flag. The missing PA is in no denominator and no worst-case bound. No stage reports the R1 layering required by 
- **PA-end reward continuity (R5/D-6)** — pa_outcome only rejects `nxt.at_bat_number <= last.at_bat_number`, and its only consistency flag is a score mismatch. It never checks that the next row is at_bat_number+1 or that outs/bases are consistent. Reproduced: with PA 200:2 missing, pa_outcome for 200:1 returns end='next_row_state', reward = WE at 200:3's pre-pitch state, and flags=[]. The reward therefore silently spans two PAs and biases
- **Provenance chain from S1 outputs to the inputs registered for S2–V4** — The S1 BC records provenance config_sha256 as the S1-time config (runner:371). Adding `registered_inputs.bc/support` then necessarily changes the config bytes. No later stage checks the loaded BC or support provenance (config_sha256, code_commit, data_version) or the S1 manifest against the running config. `registered()` accepts any path whose file SHA matches the value written into the config.
- **R6 PA status vocabulary in the V4 output** — R6 registers four PA statuses: COMPLETE, INCOMPLETE_NO_TERMINAL, INCOMPLETE_START and UNSUPPORTED. pa_status has no INCOMPLETE_START, so a PA whose first row is not 0-0 is counted as UNSUPPORTED; the distinction survives only as a reason string. dr.json also carries two PA tallies that disagree. For example, a PA refused at decision 2 for NO_LOGGED_ACTION is 'UNSUPPORTED' in the estimator's `statu
- **Runner dispatch test coverage** — The `materialize-bc` and `bind-probe` dispatch branches, load_inputs, sealed_probe_arrays, pinned_npz/pinned_parquet, and main's local-config pin are never executed. The test config uses keys the committed config lacks, which is why the schema mismatch in the first gap (e.g. KeyError 'rows' in bind-probe) is not caught.
- **Overlap and ratio diagnostics (D87 §5)** — No defect against ML-POLICY-MATERIALIZATION-v1, since the D87 thresholds are proposals. Not covered: per-decision ratio quantiles and moments, zero/invalid denominator counts, ESS by role/month/pitcher volume, and the proposed PA ESS ≥ 100 / game ESS ≥ 30 gate. The V2 and V4 outputs report only overall PA/game ESS over COMPLETE PAs.

## 2차 검토: 0135379 (결정 반영 + 1차 수정)

워크플로 `coop018-rereview`: 일곱 렌즈(추정기, 요청 생성기, 런타임, 실행기, 준합성·τ, **결정 준수**, **1차 수정 확인**), 결함마다 검증자 3명, 에이전트 215개. **확인 61건**(렌즈 사이 중복 많음), 기각 8건, 누락 점검 13항목. 1차 36건 중 되살아난 결함은 없었고, 새 코드와 결정 세부(각 결정의 비판 담당 요구 변경 포함)에서 결함이 나왔다.

### 확인된 결함 묶음과 조치

| 묶음 | 확인 번호 | 조치(b416701) |
|---|---|---|
| M-10 짝 항등 검사가 항상 참 | C0, C52 | 독립 MaskedReference 후보 callable을 런타임에 통과시키는 실제 짝 실행으로 교체(완결 집합 동일·Δ 정확히 0이 아니면 S6 실패) |
| 타석을 닫는 무투구 행이 요청이 되어 완결 타석이 검열됨 | C4 | 종료 사건이 있는 무투구 마지막 행은 요청을 만들지 않음, 무결정 타석은 NO_DECISION(Δ=0, L0만) |
| 경기 종료를 data.py complete_game 휴리스틱으로 판정 | C6, C42 | game_final_v1(마지막 행 events, 사후 점수 동률 아님, 5회 이상)로 교체, S0에서 불일치 수 집계 |
| 분할 경계를 넘는 경기가 표본에 섞임 | C5, C21 | 경계 경기 제외, 블록은 해당 분할 행만 |
| S6가 식별자 pin 없이 실행 | C11, C18 | tau_freeze.final_identity_sha256을 S6 기대 식별자로 강제 |
| 등록 필드 누락이 전체 실행 뒤에야 드러남 | C1, C19, C39, C20, C44, C30, C45, C58, C38, C14 | 필수 필드를 config 전체 경로로 확장(부트스트랩 무효 비율, S3 후보·예산, S5 허용치·최소 로그 수, hang guard, D-4 문턱), dr_q_source 값 검사, S3/S3b 예산 일치, S3b 경기 게이트 = ESS 게이트 |
| 선행 조건이 자료 로드 뒤 확인, 등록 입력이 봉인 산출물과 무관 | C25 | 자료 로드 전 선행 조건 검사, 등록 입력은 봉인된 stage manifest에 같은 SHA로 있어야 함, S4 기록·V2 합격 요구 |
| hang guard가 원장 append 중간에 끊음 | C10, C24, C53, C59 | append 동안 SIGALRM 차단(pthread_sigmask), 테스트로 재현·확인 |
| source commit 강제가 등록 절차와 모순 | C29 | 코드 경로(experiments/src/scripts)가 source commit 이후 불변이고 미커밋 변경이 없을 것, 등록 커밋은 허용 |
| ESS 게이트가 정의 불가 ESS를 건너뜀 | C2, C36 | ESS가 None이면 게이트 실패, 완결 PA가 없어도 약한 overlap 라벨 |
| τ 표가 구조 결함 PA를 완결로 가중 | C32, C41 | 구조 결함 PA 제외, 잡음 NaN이면 거부(samples ≥ 3) |
| 탐색 설정 비용이 samples×pitch_cap 비례 가정 | C31 | S3가 등록 후보마다 실제 행 수를 측정해 선택 |
| V2가 S6와 다른 q̂(계획 Q 재사용)로 검증 | C15, C22, C37 | V2·S3 비용에 S6와 같은 평가 seed q̂ 사용 |
| V2 시작이 M-1 표본 규칙을 따르지 않음 | C9, C27, C43 | M-1 경기 표본의 모든 E0 PA 시작 |
| S5 행 예산이 stage 상한이 아님 | C26 | row_budget_per_run 하나를 후보 탐색·세계 시뮬레이터가 공유 |
| S2 probe 검사가 다른 PA 위치까지 봄, 무투구 목록 미전달 | C17, C23, C60 | 블록 안 probe 위치만 비교, 등록 무투구 목록 전달 |
| dr.json에 스냅샷 SHA·as-of·미지 타자 수 없음 | C16, C48 | provenance 기록 |
| 손 값 빈 문자열·pd.NA 처리 불일치 | C8, C13 | normalize_hand 공용화 |
| 지원 표가 단일 손 투수의 부분집합만 확인 | C12, C51 | 같음을 요구 |
| S0/S1 보고 항목 누락 | C7, C28, C46, C47, C55, C56 | census 확장(분할 경계 경기, 경기별 PA 수, 타자면 결측, TRAIN 결과 인접 항목, 결측 라벨 카운트, AMBIGUOUS 규칙 공용), S1 게이트(TRAIN 행 수·어휘·타자면), S1 보고(BC-P 전용 투수, league tier 전용 행동, 교체 직후 셀), 모든 분할 코드 게이트 |
| strata·parquet 정보 누락 | C3 | E0 비율·ESS, parquet에 stratum 열 |
| 필수 테스트 누락 | C33, C35, C49, C57 | τ 재계산 = 런타임, V2 기각·V3 가드·D-10 이어짐 동치, 무행동 뒤 질의 없음, 맥락 손 불일치 중단 |
| 선언된 검열 hazard 주입 없음 | C50 | V2 세계에 선택적 hazard, 경계가 참값 포함 확인 |
| 등록값 무시 | C54 | S0 결과 인접 분할·무투구 목록·dr_q_source 반영 |
| D-9 잡음 규칙 samples ≤ 2 | C34, C40 | τ 선택은 samples ≥ 3 요구, NaN 잡음 거부 |

묶음에 없는 확인 번호: 없음.

### 기각된 주장

| 파일:행 | 주장 | 표 |
|---|---|---|
| `policy_estimator.py:162` | Action-dependent MISSING_LABEL censoring breaks the L1 bound but is not counted as a validity violation | 1/3 |
| `policy_requests.py:287` | Snapshot-vs-rolling guard passes vacuously when no rows fall on as_of | 1/3 |
| `policy_requests.py:331` | S0 census reads evaluation-split events despite the registered read scope | 1/3 |
| `run_policy_validation.py:423` | Only BudgetExceeded/HangGuardExceeded leave an 'aborted' row; other stage-aborting errors leave a partial ledger that reads as run_status OK | 1/3 |
| `run_policy_validation.py:209` | stage() hang guard does not save or restore an enclosing timer; a nested stage disables the outer guard | 1/3 |
| `run_policy_validation.py:828` | The V2 gate runs the DR with the planning-reuse q-hat, but S6 uses the evaluation-seed q-hat (M-7), so M-10's 'V2 accept before S6' certifies a different estimator configuration and identity | 1/3 |
| `run_policy_validation.py:161` | Source-commit enforcement (fix for :70, also relied on by :73 and :417) can never pass for a registered in-repo config | 1/3 |
| `run_policy_validation.py:672` | D-7 as-of guard (fix for :408) passes vacuously when no split row is dated as_of; as_of is never tied to the split's first date | 1/3 |

### 누락 점검 — b416701에서 다룸 (D-7의 G0 예측 이동 진단과 D-10 테스트의 BC 전용 코드 경우는 미구현으로 남김)

- **S2 request generator vs registered R3 no-pitch list (requests -> probe)** — R3's adjustment_rule allows the no-pitch list to change after S0 (for example adding intent_ball), and S2 runs after S0. Once the list changes, S2 labels those rows with their raw type or MISSING instead of NO_PITCH. Its history actions and its structural problem/index then differ from the requests 
- **M-10 S6 count seal (runner -> ledger)** — The S6 seal can never fail, so the M-10 go/no-go item 'S4/S6 count seals' is vacuous for S6. It is not compared with any independent count, such as the S0 census or a pinned per-game PA count. The S4 seal does compare with the census.
- **Runner outputs: JSON type fidelity** — Verified: v5.json start_population.outside_reasons ({'a': '2'}, line 451) and probe.json probe_dates (line 392) store the counts as strings instead of integers. A downstream reader that sums the S4 E0-exclusion counts gets string concatenation or a TypeError. Nothing fails at write time.
- **S5 config values vs run_world arithmetic** — Verified with the synthetic fixture: logs_per_start=1 gives dr_se=NaN, then delta_gap_se=NaN and accept=False. dump(allow_nan=False) then raises 'Out of range float values are not JSON compliant' after the whole S5 run (every planning seed and law), so v2.json is never written. truth_rollouts=1 fail
- **D-9 critic change 5: tau thresholds must be the same registered gate as the S6/2026 ESS gate** — A registration with S3b game_ess_min=10 and ess_gate.game=30 passes registration(). S3b can then select a tau whose June game ESS is 12, and S6 labels the same policy UNCONFIRMED_WEAK_OVERLAP under a different gate. Tau is chosen against a gate that was never registered as the evaluation gate.
- **D-7 required label-blind diagnostics** — None of the required D-7 diagnostics is computed in any stage (S1b, S3-S6). The recorded decision is implemented without its registered diagnostic set, so the size of the snapshot-vs-rolling shift cannot be reported before S6.
- **D-9 note 5: post-selection determinism check** — A search that is not deterministic, or a component that changes between S3b and S6, is caught only by the identity hash. The registered q-reproducibility check never runs. I ran the synthetic chain: the identity hashes do match, but q reproduction is never checked.
- **D-1 S1 gates vs registered constants** — If the processed frame's TRAIN population differs from the registered 1,386,362 rows, S1 still passes. The upper-bound gate is tautological. If the config vocabulary list and the pinned prep vocabulary disagree, S1 passes and the census R3 gate uses a different vocabulary from the one S1 checked.
- **D-6/R5 missing post scores: primary vs registered sensitivity** — Verified: with a NaN post_home_score on a non-final PA, the primary gives reward 0.63 with a flag, and score_source='post_pitch' raises 'ValueError: cannot convert float NaN to integer'. In S6 this happens in variant_facts after the full candidate run and primary estimate, so dr.json is never writte
- **D-11 D: safety timeout required on label-blind stages** — A registration with a null hang_guard_seconds for S0-S4 runs those stages with no timeout while holding the single heavy lock, which contradicts the recorded D-11 D rule. Nothing refuses it before data load.
- **Test adequacy: S2 bind-probe path** — The reported S2 probe-prefix guard defect and the no_pitch default (above) are both in code that has no synthetic coverage. The M-10 'S2 pass' prerequisite is satisfied in tests by a stub, not by the probe.
- **Test adequacy: tau_freeze -> expected identity -> S6 pin** — No defect found. I ran a synthetic chain with an addendum pinning the tau_freeze identity, and S6 reproduced it exactly (072de6b4...). But a change that puts context-dependent state into components.identity would make S6 unrunnable, and no test would catch it.
- **Test adequacy: registered sensitivities and bootstrap/ESS branches** — The post_pitch NaN crash above and any wrong kind or slope in flags_to_bounds would pass the suite. M-4's 'CI null above the registered share' has no test.

## 3차 검증: b416701

워크플로 `coop018-round3-verify`: 2차 확인 61건·누락 13항목의 수정 확인(4개 묶음) + 새 코드 세 렌즈, 결함마다 검증자 3명, 에이전트 121개. **확인 36건, 기각 2건.** 대부분은 2차 수정이 불완전하거나 새 결함을 만든 경우다. 조치는 다음 커밋에 있다.

| 묶음(확인 36건) | 조치 |
|---|---|
| 종료용 무투구 행을 뺄 때 투구 번호 연속성 검사가 빠짐(마지막 실제 투구 유실이 완결로 처리), 앞 투구가 없는 단독 종료 행이 NO_DECISION | pitch_number 연속성을 종료 행까지 포함해 검사, 끊기면 그 지점에서 missing_row 검열, 단독 종료 행은 pitch 1일 때만 NO_DECISION |
| 경기 마지막 행 사후 점수 결측이 조용히 검열됨(등록 규칙은 FAILED_INTEGRITY) | game_table이 선택 경기만 읽고, 마지막 행 사후 점수 결측이면 FAILED_INTEGRITY |
| D-9a가 samples < 3을 골라 S3b가 영구 거부, 최소값이 코드 상수 | config의 잡음 규칙 최소 samples를 선택과 S3b 모두에 적용 |
| hang guard 원장 보호가 스레드별 signal mask라 torch 작업 스레드가 있으면 무력 | append를 critical section으로 감싸 인터럽트를 append 완료 뒤로 연기(어느 스레드가 신호를 받아도 성립), 실제 보조 스레드 + itimer 테스트 |
| 탐색 도중 hang guard가 FAILED_RUNTIME 결정 행이나 malformed 행으로 기록됨 | HangGuardExceeded를 BaseException으로: 진행 중 요청은 기록되지 않고 abort 행만 |
| config가 어떤 커밋에도 묶이지 않아 미커밋 config 수정이 통과 | --config와 모든 addendum이 HEAD blob과 바이트 동일해야 함 |
| V2 합격·S2 probe가 S6 식별자·부품과 무관 | S2 identity.json을 bind_identity로 등록해 S3 이후 부품 SHA 강제, V2 기록에 실행별 후보 식별자, S6는 주 seed V2 실행의 식별자 = 최종 식별자 요구, 등록 입력은 그 입력을 만드는 명령의 봉인 manifest에서만 |
| planning_reuse 전환이 M-7 비용 규칙 없이 허용 | M-7을 기계적 규칙으로: S3가 후보마다 평가 seed 탐색 비용까지 측정, S3b에서 (평가 포함 행/결정 × S6 계획 결정 수 > S6 행 예산)이면 planning 재사용, 결과를 동결 식별자에 기록하고 S5/S6가 따름 |
| τ가 다른 게이트·설정으로 동결돼도 S5/S6가 받아들임 | tau_freeze에 게이트(문턱·grid·ESS 게이트) 기록, S5/S6에서 현재 등록값과 같아야 함 |
| 등록 필드 형식(draws, 최소 표본, 민감도 이름, ESS 문턱)이 S6 뒤에야 검사 | 등록 단계에서 검사 |
| 층별 경계가 L0뿐, 검열 ESS·2차 층 ESS 없음 | 층별 L0·L1 경계, 검열 가중치 ESS, 2차 층 ESS |
| S0 census 등록 항목 누락 | AMBIGUOUS 투수 TRAIN PA 수, 손 결측·미관측 PA 수, P8 패널 손 × 등록부 교차표, data.py next_* parity, 결측 라벨 투수별 수, S1 BC-P−BC-E 구종 분포 |
| D-6 end_kind별 잔차 진단 없음 | 완결 PA의 종료 잔차(r − q̂) 층별 평균 |
| D-7 진단 불완전(G0 TV/KL, membership TV, 미지 타자 PA 수) | S6 보상 열람 전 등록 부분표본에서 G0 10-class TV·KL과 membership TV, 미지 타자 PA 수 |
| BC-P 전용 투수 규칙이 등록값이 아니고 층이 없음 | config에 규칙 등록, S6 층 bc_p_only_pitcher |
| V2 참값 rollout이 버려질 평가 seed 탐색을 매 상태 수행 | 참값은 계획 Q만 쓰는 planning_policy |
| D-5 hazard 검사가 틀린 처리도 통과, 끝점 CI 포함률 없음, S5에서 실행 불가 | 검열 PA 폭 = Πρ_c + Πρ_r을 실제 원장 값으로 정확히 확인, S5 선택적 declared_hazard + 반복으로 경계·끝점 CI 포함률 기록 |
| 테스트 공백 | τ 재계산을 여러 τ에서, 민감도가 실제로 바뀌는 PA, 지원 표 같음, 투수별·행 유실 경우 |

### 기각된 주장 (S3 비용을 평가 seed 포함으로 재는 것은 보수적이라 결함 아님)

| 파일:행 | 주장 | 표 |
|---|---|---|
| `run_policy_validation.py:939` | The D-9a selection is checked against the S3b row budget using evaluator-inclusive cost, but S3b runs no evaluator | 0/3 |
| `run_policy_validation.py:936` | S3 profile measures cost with the evaluation-seed search, but D-9a checks it against the S3b budget, and S3b does not run that search | 0/3 |

### 남은 한계(미구현으로 기록)

- D-10 필수 테스트 중 BC 전용(token 밖) 코드가 이어지는 경우: 합성 fixture의 BC 어휘가 token 어휘와 같아 만들지 않았다.
- D-3 description 허용 목록 fail-closed 게이트: 런타임이 매핑 불가 결과 뒤 요청을 `INCONSISTENT_HISTORY`로 거절하므로 멈추지 않고 분모에 남긴다.
- membership TV는 결합된 context 인코더에 archetypes가 있을 때만 계산된다(합성 인코더에는 없음, 실제 SequenceContext에는 있음).
- 모든 검토는 같은 모델의 자기 검토이며 독립 검토가 아니다. 4차 검토는 하지 않았다(사용자가 ultracode를 끈 뒤 수정·커밋·푸시만 요청).

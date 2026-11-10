# NOV14H 정리 기록 — 2026-11-10

작업 ID `NOV14H-CLOSE-20261110` (큐 #37 CLAIM 6756161345). 작성: Claude 문서 작업자. 원 실험: `NOV14H-20261102` (Codex, 2026-11-02 22:21 ~ 11-03 KST).

## 맨 위 요약

- **돌린 것:** 다음 공을 맞히는 작은 신경망(T 계열)의 설정을 바꿔 A5000 GPU 5장으로 **170 fit을 띄웠다** (001~008 165개 + FINAL 5개). GPU 학습 실패는 0. H1은 CPU 비교 실험으로 2번 시도했고 둘 다 STOP했다.
- **가장 중요한 결과:** 등록 규칙으로 표적마다 하나씩 뽑았다. 결과 7갈래 표적은 004 `lr030_drop020_e30`(3seed ΔNLL −0.0061284), 투구 전 구종 9갈래 표적은 003 `state_t32`(빈도표 대비 3seed ΔNLL −0.0538731). 둘 다 이미 여러 번 본 2026 개발 자료에서 고른 것이라 **운영 승격 근거가 아니다**.
- **안 끝난 것:** FINAL outcome7 3건은 파일이 T7에 와 있지만 배차기 검증에서 걸려(KeyError 'variant') "수용" 기록이 없다. 그 STOP 때문에 25 fit(009 전체 24 + FINAL prepitch9-s2)이 시작되지 않았다. 007 반복확인 채점과 008 채점(STOP)은 비어 있다. 010은 등록만 됐다. H1 시도 3은 검토 대기에서 멈췄다.

> 이 문서는 새 채점·추론·학습·해시 계산 없이, 맥미니에 남은 장부·보고 JSON·등록 문서·T7 폴더 메타데이터만 읽고 썼다. 숫자는 모두 기존 JSON 값을 옮긴 것이다(소수점 아래 7자리 반올림, 원값은 해당 JSON 키). 없는 값은 "미측정"이다.

경로 약어:
- `OPS/` = `/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/operations/`
- `REP/` = `OPS/reports/`
- `T7/` = `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/GPU14H-20261102/`

## 쉬운 말 풀이

- **fit:** 모델을 한 번 학습시키는 것. 투수 한 명에게 불펜 세션 한 번을 시키는 것과 비슷하다.
- **seed:** 학습을 시작하는 무작위 번호. 같은 설정이어도 seed가 다르면 결과가 조금 달라진다. 같은 투수도 날마다 공이 조금씩 다른 것과 같다. 여기서는 seed0으로 먼저 고르고(screen) seed1·2를 더해 다시 본다(stability).
- **OOF(out-of-fold):** 7월을 빼고 학습한 모델로 7월을 맞히는 식의 채점. 8월·9월도 같은 방식이다. 안 본 경기로 치르는 모의고사라고 보면 된다. 7월 108,804구, 8월 122,643구, 9월 105,829구, 합계 337,276구다.
- **pooled:** 세 달을 합쳐 한 번에 계산한 값.
- **3seed 확률평균:** seed 0·1·2 모델이 낸 확률을 평균해 채점한 값. 선발 세 명의 평균자책점을 합쳐 보는 것과 비슷하다.
- **NLL(음의 로그우도):** 실제로 나온 결과에 모델이 낮은 확률을 줬을수록 커지는 벌점. 타자가 직구를 쳤는데 "직구 5%"라고 했다면 벌점이 크다. **작을수록 좋다.**
- **Brier:** 예측 확률과 실제(나왔으면 1, 아니면 0)의 차이를 제곱해 더한 값. 강수확률 예보를 채점하는 방식과 같다. **작을수록 좋다.**
- **ECE10:** 확률을 10칸으로 나눠 "70%라고 한 공들이 실제로 70% 나왔나"를 본 어긋남. 3할 타자라고 소개한 선수가 실제로 3할을 치는지 보는 것과 같다. **작을수록 좋다.** 선택 기준이 아니라 참고 지표다.
- **Δ(델타):** **후보 − 기준.** 세 지표 모두 작을수록 좋으므로 **음수면 후보가 낫고, 양수면 나쁘다.**
- **표적(target) 두 가지:** `outcome7`은 현재 구종을 안다는 전제에서 그 공의 결과를 7갈래로 맞히는 원 T 모델 계열이다(001·002·004·006·007). `prepitch9`는 공을 던지기 전에 구종을 9그룹으로 맞힌다(003·005·008·009·010). 7지선다와 9지선다는 NLL 크기가 원래 다르므로 **두 표적의 NLL은 직접 비교하지 않는다.**
- **Q_pkg / E9_pkg:** 9구종 실험의 채점 대상. 기존 패키지가 원래 결과 7갈래에 맞는 투구만 골라 둔 것이라 전체 투구가 아니다. 그중 9그룹에 매핑되는 공(337,196구)만 채점한다.
- **동결 빈도표:** 학습 구간에서 "이 투수가 이 카운트에서 무엇을 던졌나"를 센 표. 9구종 실험의 가장 단순한 비교 기준이다.
- **2026 EXPOSED DEVELOPMENT:** 이미 여러 번 들여다본 2026 개발 자료. 여기서 고른 승자는 한 번 더 독립된 자료로 확인하기 전까지 "가능성 있는 후보"일 뿐이다.

## 장부 다시 세기 (`OPS/dispatch-ledger.jsonl`, 1,789줄)

- `launch` 170 / `recovered` 167 / `pull_failed` 3 → PM 값과 **일치**.
- `launch` 이벤트의 `launcher_returncode`는 170건 모두 0.
- `pull_failed` 3건은 `H14-FINAL-outcome7-s0/1/2`, `error` = `KeyError('variant')` → **일치**.
- 마지막 줄 `multi_phase_terminal`(2026-11-02T17:49:02Z = 11/3 02:49 KST)의 `remaining` 25개, `stopped: true`, `unknown_intents: []` → **일치**.
- 첫 launch 2026-11-02T13:55:32Z(`H14-lr030-july-s0`), 마지막 launch 17:42:34Z(`H14-FINAL-prepitch9-s1`), 마지막 recovered 17:49:01Z.
- GPU 번호별 launch 수: idx0 31, idx1 36, idx2 34, idx3 36, idx4 33.
- 회수된 167건 `returned_bytes` 합: 1,683,742,071 B.

## 실험별 정리

### EXP-GPU14H-001 — 학습률·규제·크기 10조건 (outcome7)

- 가설: 원 T의 학습률·weight decay·dropout·depth·width 하나를 바꾸면 나아지는가.
- 바꾼 설정: `lr030`, `lr060`, `wd001`, `wd010`, `drop000`, `drop020`, `depth1`, `depth3`, `width32`, `width128`.
- fit: 계획 30 + 최대 12 = 42 / 실행 42 / 회수 42 (`REP/screen30-family.json` `launches`/`recovered` 30/30, `REP/stability12-family.json` 12/12, `failed` 0).
- 채점: 완료 (`REP/score-screen-run.json`, `REP/score-stability-run.json`, 둘 다 `status` COMPLETED).
- seed0 1차 선별, 원 T seed0 대비 ΔNLL / ΔBrier (`candidate_deltas.<variant>`):
  - drop020 −0.0112497 / −0.0022427
  - lr030 −0.0108237 / −0.0019750
  - width32 −0.0067499 / −0.0011203
  - depth1 −0.0062040 / −0.0011746
  - width128 −0.0037988 / −0.0006725
  - lr060 −0.0027402 / −0.0004402
  - wd010 −0.0002302 / +0.0000618 (Brier 악화)
  - wd001 +0.0000949 / +0.0000009
  - depth3 +0.0012699 / +0.0001413
  - drop000 +0.0529251 / +0.0079752
- 선택: `selected_variants` = drop020, lr030 (등록 규칙: 조건 충족 후보 중 최대 2개).
- 3seed 확률평균, 원 T 3seed 대비 (`candidate_deltas`):
  - drop020 ΔNLL −0.0042187 / ΔBrier −0.0008643 / ΔECE +0.0009945
  - lr030 ΔNLL −0.0040990 / ΔBrier −0.0008506 / ΔECE +0.0010118
- 판정: 둘 다 ΔNLL<0·ΔBrier≤0을 충족해 **FINAL 후보 자격**이 있다(`REP/final-select-run.json` `selection.eligible.outcome7`). ECE는 둘 다 악화. `promotion` false.

### EXP-GPU14H-002 — 세 가설 (outcome7)

- 가설: 위치 embedding 제거 / 7leaf joint NLL 추가 / 60epoch 고정 중 원 T보다 나은 것이 있는가.
- fit: 계획 9 + 최대 6 = 15 / 실행 15 / 회수 15 (`REP/batch2-screen9-family.json` 9/9, `REP/batch2-stability6-family.json` 6/6).
- 채점: 완료 (`REP/batch2-score-screen-run.json`, `REP/batch2-score-stability-run.json`).
- seed0, 원 T seed0 대비 ΔNLL / ΔBrier / ΔECE (`pooled[].delta`):
  - no_position −0.0048932 / −0.0006745 / −0.0010602
  - joint_leaf_nll −0.0003285 / +0.0002994 / +0.0024858 (Brier 악화 → 자격 없음)
  - epochs60 +0.0356569 / +0.0041050 / +0.0084150
- 선택: no_position.
- 3seed, 원 T 3seed 대비 (`pooled.delta`): ΔNLL −0.0015610 / ΔBrier −0.0001932 / ΔECE −0.0003322. 8월만 ΔBrier +0.0000043 (`monthly.august.delta.brier`).
- 판정: 자격 있음(FINAL pool 포함).

### EXP-GPU14H-003 — 투구 전 구종 9그룹 (prepitch9, 새 표적)

- 가설: 투구 전 상태 11개 + 이전 32구 이력으로 구종을 맞히면 동결 빈도표보다 나은가.
- 바꾼 설정: `state_mlp`, `state_t32`(Transformer), `state_c32`(합성곱).
- 기준: 원 T가 아니라 **동결 빈도표**(새 표적이라 원 T 비교값이 없다).
- fit: 계획 9 + 최대 12 = 21 / 실행 21 / 회수 21 (`REP/batch3-screen9-family.json` 9/9, `REP/batch3-stability12-family.json` 12/12). 빈도표 count fit 3.
- 채점: 완료 (`REP/batch3-score-screen-run.json`, `REP/batch3-score-stability-run.json`). 채점 행 337,196.
- seed0, 빈도표 대비 ΔNLL / ΔBrier / ΔECE (`pooled[].delta`):
  - state_mlp +0.3836209 / +0.0929253 / −0.0197468 (빈도표보다 나쁨)
  - state_t32 −0.0415403 / −0.0204057 / +0.0030056
  - state_c32 −0.0182316 / −0.0128515 / +0.0062936
- 선택: state_t32, state_c32.
- 3seed, 빈도표 대비 (`pooled[].pooled.delta`):
  - state_t32 −0.0538731 / −0.0249452 / −0.0080738 (NLL 1.3742001 대 1.4280732)
  - state_c32 −0.0320236 / −0.0173917 / −0.0031158
- 판정: 둘 다 자격 있음. state_t32가 prepitch9 FINAL 승자.

### EXP-GPU14H-004 — 학습률·dropout·학습 길이 조합 (outcome7)

- 가설: 001에서 좋았던 낮은 학습률과 높은 dropout을 합치거나 60epoch로 늘리면 더 나은가.
- 바꾼 설정: `lr030_drop020_e30`, `lr030_e60`, `drop020_e60`, `lr030_drop020_e60`.
- fit: 계획 12 + 최대 12 = 24 / 실행 24 / 회수 24 (`REP/batch4-screen12-family.json` 12/12, `REP/batch4-stability12-family.json` 12/12).
- 채점: 완료 (`REP/batch4-score-screen-run.json` `factorial_status` COMPLETE_8_CELLS, `REP/batch4-score-stability-run.json`).
- seed0, 원 T seed0 대비 ΔNLL / ΔBrier / ΔECE:
  - lr030_drop020_e30 −0.0159706 / −0.0032418 / −0.0017947
  - lr030_drop020_e60 −0.0067392 / −0.0016769 / +0.0009733
  - drop020_e60 +0.0050632 / −0.0000040 / +0.0055350
  - lr030_e60 +0.0077790 / +0.0008463 / +0.0024706
- 선택: lr030_drop020_e30, lr030_drop020_e60.
- 3seed, 원 T 3seed 대비:
  - lr030_drop020_e30 −0.0061284 / −0.0012898 / +0.0026162 (NLL 1.5733819)
  - lr030_drop020_e60 −0.0017925 / −0.0004794 / +0.0038179
- 판정: 둘 다 자격 있음. lr030_drop020_e30이 outcome7 FINAL 승자. 두 후보 모두 ECE 악화.

### EXP-GPU14H-005 — 투수 ID 추가 (prepitch9)

- 가설: 003 모델에 투수별 16차원 보정(ID offset)을 더하면 나아지는가.
- 바꾼 설정: `pitcher_mlp`, `pitcher_t32`, `pitcher_c32`.
- 기준: 같은 구조·fold·seed의 003 **ID 없는** 모델(paired). 보조로 동결 빈도표.
- fit: 계획 9 + 최대 6 + 대조 최대 6 = 21 / 실행 21 / 회수 21 (`REP/batch5-screen9-family.json` 9/9, `REP/batch5-stability12-family.json` 12/12; 대조는 ID 없는 MLP seed1·2).
- 채점: 완료 (`REP/batch5-score-screen-run.json`, `REP/batch5-score-stability-run.json`).
- seed0, ID 없는 같은 구조 대비 ΔNLL / ΔBrier / ΔECE:
  - pitcher_mlp −0.4318976 / −0.1070133 / +0.0396689
  - pitcher_t32 −0.0682053 / −0.0127167 / +0.0249367 (raw NLL 1.3183276, 셋 중 최저)
  - pitcher_c32 −0.0723918 / −0.0134160 / +0.0216138
- 선택: pitcher_mlp. `selection_rule` = "minimum paired pooled delta NLL, not minimum raw candidate NLL" (개선폭이 가장 큰 것을 고름. 원래 MLP가 빈도표보다 나빴기 때문에 개선폭이 크게 나온다).
- 3seed (`candidates[0]`):
  - ID 없는 MLP 대비 `pooled.delta`: −0.4359849 / −0.1073736 / +0.0393688
  - 빈도표 대비 `vs_frozen_frequency.delta`: −0.0536620 / −0.0147734 / +0.0180801
  - 처음 보는 투수(15,401구) `known_cold_start.cold_start_pitcher` ΔNLL +0.2527423 (악화), 아는 투수(321,795구) −0.4689472
- 판정: 자격 있음(FINAL pool 포함). FINAL에서는 003 state_t32(3seed NLL 1.3742001)가 pitcher_mlp(1.3744112)보다 낮아 승자가 아니다. ECE는 모든 ID 후보에서 악화.

### EXP-GPU14H-006 — 위치 embedding 제거 × 정규화 (outcome7)

- 가설: 001·004의 정규화 부모에서 위치 embedding을 빼면 더 나아지는가(사후 가설).
- 바꾼 설정: `nopos_lr030`, `nopos_drop020`, `nopos_lr030_drop020`.
- fit: 계획 상한 21(9 + 6 + 부모 대조 최대 6) / 실행 15 / 회수 15 (`REP/batch6-screen9-family.json` 9/9, `REP/batch6-stability6-family.json` 6/6, `new_control_fits` 0).
- 채점: 완료 (`REP/batch6-score-screen-run.json`, `REP/batch6-score-stability-run.json`).
- seed0, 위치 있는 부모 대비 ΔNLL / ΔBrier (`pooled.delta`), 원 T 대비 (`vs_original_T.delta`):
  - nopos_lr030 −0.0013895 / −0.0002457 · 원 T 대비 −0.0122132 / −0.0022207
  - nopos_drop020 −0.0024181 / −0.0004176 · 원 T 대비 −0.0136678 / −0.0026603
  - nopos_lr030_drop020 −0.0004257 / −0.0000407 · 원 T 대비 −0.0163963 / −0.0032825
- 선택: nopos_lr030_drop020 (등록 규칙: 후보 raw NLL 최소).
- 3seed:
  - 004 부모 대비 ΔNLL **+0.0001745** / ΔBrier **+0.0001056** / ΔECE −0.0006050
  - 원 T 대비 −0.0059539 / −0.0011842 / +0.0020112
- 판정: seed 하나씩은 부모보다 낫지만 3seed 평균에서는 NLL·Brier가 부모보다 나쁘다(`limitations`). **위치 제거의 추가 이득은 확인되지 않음.** 006은 FINAL pool 대상이 아니다.

### EXP-GPU14H-007 — 배치 크기·학습률 (outcome7)

- 가설: 004 부모(배치 2048, lr .0003)에서 배치를 키우고 학습률을 같이 올리면 품질은 어떤가.
- 바꾼 설정: `b4096_lr030`, `b4096_lr060`, `b8192_lr120`.
- fit: 계획 9 + 6 = 15 / 실행 15 / 회수 15 (`REP/batch7-screen9-family.json` 9/9, `REP/batch7-stability6-family.json` 6/6).
- 채점: **screen만 완료** (`REP/batch7-score-screen-run.json`). 반복확인 채점 보고가 없고 `T7/EXP-GPU14H-007/attempt-001/evaluation/`에도 `score-screen`만 있다.
- seed0, 004 부모 대비 ΔNLL / ΔBrier / ΔECE, 원 T 대비 ΔNLL:
  - b4096_lr030 −0.0011597 / −0.0002569 / +0.0002081 · 원 T 대비 −0.0171303
  - b4096_lr060 +0.0014872 / +0.0003834 / +0.0009784 · 원 T 대비 −0.0144835
  - b8192_lr120 +0.0023886 / +0.0006206 / +0.0040319 · 원 T 대비 −0.0135820
- 선택: b4096_lr030. seed1·2 6 fit은 회수됐지만 채점 전이다.
- 판정: **미판정**(3seed 비교 미측정).

### EXP-GPU14H-008 — 타자 ID (prepitch9)

- 가설: T32에 타자 ID 보정을 더하면(단독 또는 투수 ID와 함께) 나아지는가. 005에서 뽑히지 않은 pitcher T32를 seed1·2로 따로 반복확인한다.
- 바꾼 설정: `batter_only`, `pitcher_batter`, 대조 `control-pitcher_t32` seed1·2.
- fit: 계획 상한 18(6 + 대조 6 + 최대 6) / 실행 12 / 회수 12 (`REP/batch8-screen6-family.json` 6/6, `REP/batch8-control6-family.json` 6/6).
- 채점: **STOP** (`REP/batch8-score-screen-run.json`): `status` STOP, `exit_code` 87, `failure` = "KeyError: 'batter' at experiments/gpu14h_batter/run.py:215 accessing p['seq'][role]", `receipt_created` false, `score_control_started` false, `automatic_retries` 0.
- 수치: 미측정. 선택이 없어 반복확인 6 fit도 시작되지 않았다.
- 판정: **미판정**.

### EXP-GPU14H-009 — 투수 ID T32의 학습률 × dropout (prepitch9)

- 가설: 001에서 본 정규화 효과가 9구종 표적에도 있는가(사후 가설, 두 표적 수치는 비교하지 않음).
- 바꾼 설정: 학습률 .001/.0006/.0003 × dropout .1/.2/.3 중 기준(.001, .1)을 뺀 8조건.
- fit: 계획 24 + 최대 12 = 36 / **실행 0** / 회수 0. 장부상 대기열 추가는 됐다(`registry_extended`, `dispatch-registry.live-plus-batch9.json`, 17:41:04Z). `T7/EXP-GPU14H-009/`에는 빈 `returned`·`gpu-cli-tmp` 폴더만 있다.
- 채점: 없음. 판정: **미실행**.
- 근거: `/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch9/REGISTRATION.md`(등록 커밋 09994d5d), `OPS/grants/dispatch.batch9-screen.json`.

### EXP-GPU14H-010 — 투수 ID × 현재 상태 상호작용 (prepitch9)

- 가설: 005 pitcher T32에 "ID × 상태" 잔차를 더하는 것이 같은 크기의 "상태" 잔차보다 나은가.
- 바꾼 설정: `state_residual`, `pitcher_state_residual`(각 추가 파라미터 320).
- fit: 계획 18(2조건 × 3달 × seed0·1·2, 점수로 고르지 않음) / **실행 0**.
- 상태: 과학 등록만 완료(`/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch10/REGISTRATION.md` "최종 등록", 커밋 eb9282df). 운영 grant가 `OPS/grants/`에 없고 배차 대기열에도 추가되지 않았다.
- 판정: **미실행**.

### EXP-GPU14H-FINAL-001 — 연구용 최종 가중치

- 목적: 001~005에서 표적별 승자 하나를 고르고, 학습 구간 전체(FIT 720,505)로 seed 3개씩 다시 학습해 **가중치만 보존**한다. 새 채점은 없다.
- 선택(`REP/final-select-run.json` `selection.winners`, `pool_closed` true):
  - outcome7: 004 `lr030_drop020_e30`, `candidate_nll` 1.5733819
  - prepitch9: 003 `state_t32`, `candidate_nll` 1.3742001
  - `selection_basis` = "raw3seed pooled exposedOOF; targets separate; no new metrics or predictions"
  - 첫 실행은 출력 폴더가 없어 준비 단계에서 실패했고(`preparation_failure.reason` "registered output_root absent", 자료 읽기 0) 같은 등록으로 다시 돌려 COMPLETED.
- fit: 계획 6 / 실행 5 / 장부 recovered 2 / pull_failed 3 / 미시작 1.
  - `H14-FINAL-prepitch9-s0`·`s1`: recovered (511,509 B / 511,507 B).
  - `H14-FINAL-outcome7-s0`·`s1`·`s2`: pull_failed `KeyError('variant')` (아래 "끝나지 않은 것" 참고).
  - `H14-FINAL-prepitch9-s2`: 시작 안 됨.
- 판정: prepitch9 가중치 3개 중 2개 보존, outcome7은 파일은 있으나 수용 미기록. 이 가중치들은 과거 OOF 정답까지 학습에 넣었으므로 OOF 점수로 검증할 수 없다(receipt `performance_validated` false).

### EXP-PREPITCH-H1-001 — 동결 B2 대 상황 빈도 (CPU, 2025 자료)

- 가설: 같은 2025 E9 투구에서 기존 A3 B2 모델이 004 방식 상황 빈도표보다 손실이 작은가. GPU 실험이 아니며 2026 자료가 아니라 **2025 자료**(TRAIN 2023-05-15~2025-04-30, EVAL 2025-07~09)를 쓴다.
- 시도 1 (`/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/operations-h1/report.json`): `status` STOP_IMPLEMENTATION_DEPENDENCY, `exit_code` 87, `failure` "ModuleNotFoundError: No module named run_sequence_context_frequency at pinned auxiliary pickle load", `time_external_wall_seconds` 11.92. 빈도표는 만들어졌고(`count_table_generated` true) B2 추론 0, 비교 없음.
- 시도 2 (`operations-h1/report.attempt-002.json`): `status` STOP, worker `returncode` −6, `failure` "OpenMP Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.", `external_time_real_seconds` 14.61. `count_fits_cumulative_including_failed_attempts` 2, `archive_inference_calls` 0, `comparisons` 0.
- 시도 3 (`operations-h1/execution-grant.attempt-003.json`): `execute` false, `verdict` PENDING_REVIEW. `preflight.attempt-003.json` `status` PENDING_REVIEW. `T7/EXP-PREPITCH-H1-001/`에 attempt-003 폴더가 없다 → **실행 안 됨**.
- 판정: 수치 미측정.

## 끝나지 않은 것

### FINAL outcome7 3건 (장부 "pull_failed")

- 원인(배차기 코드 `OPS/dispatch.py` `recover()` 226~227행): 회수 뒤 `receipt_expectations[task]`의 모든 키가 receipt와 같은지 확인한다. 등록된 기대값(`OPS/grants/dispatch.final-fit.json`)에는 최상위 `variant` 키가 있는데, outcome7(T) 학습기의 receipt에는 `origin_variant`만 있고 최상위 `variant`가 없다. 그래서 `KeyError('variant')`가 났다. prepitch9 receipt에는 `variant` 키가 있어 통과했다.
- **실제 파일은 맥에 있다.** `gpu pull`은 세 건 모두 `returncode` 0, "회수 완료"(`OPS/events/H14-FINAL-outcome7-s*/pull.json`)였다. `T7/EXP-GPU14H-FINAL-001/attempt-001/returned/gpu14h/H14-FINAL-outcome7-s{0,1,2}/attempt-001/`에 `model-state.pt`(651,731 B), `receipt.json`, `SUPERVISOR.json`, `encoder.json`, `model-recipe.json`, `worker.log`가 있다.
- 배차기가 남긴 `terminal.json`의 기록값을 서로 대조만 해 봤다(새 해시 계산 없음). 세 건 모두 `variant` 외 기대 키는 전부 일치하고, receipt `outputs_sha256`와 배차기가 회수 직후 기록한 `files.*.sha256`도 일치한다. supervisor는 COMPLETED / returncode 0 / child_reaped true다. 다만 배차기가 이 검사를 끝까지 돌린 것은 아니므로 **"수용(recovered)" 기록은 없다.**
- 서버가 닫혀도 회수에는 문제가 없다(이미 맥에 있음). 남은 일은 기대값 오류를 기록하고, 이 파일들을 수용할지 독립 검토하는 것이다.
- 영향: 첫 `pull_failed`(17:46:49Z)에서 배차기가 새 배차 STOP을 걸어 아래 25 fit이 시작되지 않았다. 이미 돌던 잡은 끝까지 지켜보고 회수했다.

### 미실행 25 fit (`multi_phase_terminal.remaining`)

- `H14-FINAL-prepitch9-s2` 1개.
- `H14-B9-<변형>-<july|august|september>-s0` 24개 = 009 screen 전체. 변형 8개: lr100_drop020, lr100_drop030, lr060_drop010, lr060_drop020, lr060_drop030, lr030_drop010, lr030_drop020, lr030_drop030.

### 채점이 비거나 STOP한 것

- 007 반복확인 채점: 실행 기록 없음(보고·이벤트·T7 평가 폴더 모두 없음). 원인 기록 없음.
- 008 score_screen: STOP, `KeyError: 'batter'`(위). 같은 보고의 `notes`는 "payload를 0번 읽었다고 주장하지 말 것"이라고 적어 두었다. score_control은 시작 안 됨.
- H1 시도 1·2: STOP(위), 시도 3: 검토 대기.

### 010

- 과학 등록 완료, 운영 grant 없음, 배차 대기열에 없음. 실행 0.

### 종료 기록

- 큐 #37에 NOV14H의 DONE/RELEASE와 최종 요약이 없다. #35의 마지막 NOV14H 진행 요약은 2026-11-02T16:42:11Z(11/3 01:42 KST, 댓글 6582702314)다. 그 안의 001·003 수치는 보고 JSON과 같다.

## 자원·보존 상태

- T7 산출물 합계 2.1G (`du -sh T7/`). 실험별: 001 469M, 002 169M, 003 270M, 004 268M, 005 266M, 006 169M, 007 168M, 008 157M, 009 896K(빈 폴더), FINAL-001 16M, H1 202M(시도 1·2 각 약 100MB, 빈도표 파일).
- 회수 단위 폴더 수(`returned/gpu14h/` 아래): 001 42, 002 15, 003 21, 004 24, 005 21, 006 15, 007 15, 008 12, 009 0, FINAL 5. fit 폴더 하나에는 `model-state.pt`, `oof-predictions.npz`, `receipt.json`, `SUPERVISOR.json`, `worker.log`, `tmp/`가 있다(001 예시).
- 맥 쪽 운영 기록: `OPS/`(장부, 279개 이벤트 폴더·파일, grant, source snapshot), 실험별 등록 `author*/`, H1 `operations-h1/`, 검토 `review/`·`review-h1/`.
- 서버에만 있던 것(기록에서 알 수 있는 범위):
  - 입력 패키지: 기존 T OOF 3개·FINAL 학습 패키지가 서버의 이전 잡 폴더 안에 있었다(grant의 `package_root`). 맥에 같은 사본이 있는지는 이 문서에서 확인하지 않았다(미확인).
  - 서버 공유 폴더의 원격 grant 사본, task claim, source/input metadata.
  - 서버 잡 폴더의 실행기 로그·코드 스냅샷. 회수는 잡별 결과 하위폴더만 했다.
- 운영 효율(`REP/efficiency-through-first165.json`, FINAL 5건 제외):
  - `completed_fits` 165, `GPU_fit_failed` 0.
  - fit당 worker 시간 중앙값(`groups_by_config_sha256.*.worker_wall_seconds.median`, 학습·예측·저장 포함): 001 278.4초, 002 315.6초, 003 265.1초, 004 439.5초, 005 89.4초, 006 275.0초, 007 199.8초, 008 270.8초. config SHA와 실험 번호는 fit 수·시간 범위로 짝지었다(001=0c89…, 002=da0e…, 003=90e0…, 004=94ed…, 005=d798…, 006=2171…, 007=0a24…, 008=8368…; 008만 보고에 config SHA가 직접 있음).
  - intent→launch 중앙값 3.458~3.519초(실험별 `intent_to_launch_seconds.median`, GPU CLI 전송·시작 포함).
  - 빈 대기열 구간: 007→008 464.8초(`empty_registered_queue_gap_007_to_008.seconds`), live 전환 491.6초(`live_transition_gap.seconds`).
  - 17:46:01Z 순간 관측(`latest_instant_observation`): 실행 중 5, 대기 25, 보존 바이트 1,842,862,869 B(공유 예산 6 GiB = 6,442,450,944 B 대비), 서버 여유 디스크 32,321,785,856 B.
  - FINAL fit receipt `elapsed_compute_seconds`: outcome7 s0/s1/s2 403.7 / 407.6 / 422.3초, prepitch9 s0 377.2초.
- GPU 사용 구간은 장부 기준 13:55:32Z~17:49:01Z다. 승인 창은 11/3 03:21:56Z(12:21:56 KST)까지였고, 그 뒤 GPU 배차 기록은 없다.

## 한계

- 001~008·FINAL은 모두 **2026 EXPOSED DEVELOPMENT**에서 나왔다. 같은 7·8·9월 OOF를 여러 실험이 반복해 보고 seed0으로 후보를 골랐다. 그래서 선택 편향이 있고, 독립 확증이나 운영 승격 근거가 아니다(각 보고 `independent_test` false, `promotion` false).
- 004·005·006·007·009·010은 앞 결과를 보고 만든 사후 가설이다.
- 9구종 표적은 Q_pkg(결과 7갈래에 맞는 투구만 고른 패키지) 안에서만 채점했다. 전체 투구를 대표하지 않으며 실시간 도착 여부는 UNKNOWN이다.
- outcome7 후보 대부분과 모든 ID 후보에서 ECE가 악화됐다. NLL 개선이 보정 개선을 뜻하지 않는다.
- 005의 ID 모델은 처음 보는 투수에서 ID 없는 모델보다 나쁘다(ΔNLL +0.2527423).
- FINAL 가중치는 과거 OOF 정답을 학습에 넣었으므로 OOF 점수로 검증할 수 없다.
- H1은 2025 개발 자료(이미 노출)다. 실행돼도 2026 결과와 직접 순위를 매기지 않는다.
- worker 시간은 순수 GPU 계산 시간이 아니다. 장치·동시 부하가 통제되지 않았다(`measurement_limits`).

## 맥미니로 이어서 할 만한 후보 (판단 재료만)

결정은 PM·Song이 한다. 스텝 수는 기존 receipt의 `optimization_steps`로 셌다. 이 값은 ⌈FIT 행 ÷ 배치 2048⌉ × 30epoch와 일치한다(7월 FIT_E9 375,407 → 5,520, 8월 484,179 → 7,110, 9월 606,805 → 8,910 — 005 pitcher T32 receipt; FINAL 720,505 → 10,560 — FINAL receipt). MPS 결과는 기존 규칙대로 CUDA 결과의 근거가 아니며, 같은 recipe라도 숫자가 같다고 가정하지 않는다. 입력 패키지의 맥 사본이 있는지는 미확인이다.

1. **FINAL outcome7 3건 수용 검토** — 0 fit. 파일이 이미 T7에 있다. 기대값 키 오류를 기록하고 독립 검토하면 outcome7 연구 가중치 3seed가 완성된다. 다시 학습하면 3 fit × 10,560 = 31,680 스텝이지만 필요하지 않을 수 있다.
2. **FINAL prepitch9-s2** — 1 fit, 10,560 스텝. prepitch9 연구 가중치 3seed가 완성된다.
3. **007 반복확인 채점** — 0 fit(CPU 채점만). 회수된 6 fit 예측으로 b4096_lr030의 3seed 판정을 낼 수 있다. 배치 4096은 처리시간 해석과도 얽혀 과학적 의미는 작다.
4. **008 채점 수정 후 재채점** — 0 fit + 코드 수정 1건. 그 뒤 선택되면 반복확인 최대 6 fit(seed1·2 × 3달 = 2 × 21,540 = 43,080 스텝). 회수된 pitcher T32 대조 6 fit은 010의 부모로도 쓰인다.
5. **009 screen** — 24 fit, 8 × (5,520 + 7,110 + 8,910) = 172,320 스텝. 이후 최대 12 fit(2변형 × 2seed × 21,540 = 86,160 스텝). 9구종 표적 정규화 효과를 보는 사후 가설이다.
6. **010** — 18 fit, 2조건 × 3seed × 21,540 = 129,240 스텝. 운영 grant를 새로 만들어야 하고, 부모로 005 seed0과 008 대조 seed1·2가 필요하다(둘 다 회수됨, 008 대조는 채점 전).
7. **H1 시도 3** — GPU 0, CPU에서 count 1회 + B2 추론 1회 + 비교 1회. 검토 대기에서 멈춘 상태다.

참고: 001~006 outcome7 후보의 원 T 3seed 대비 ΔNLL은 −0.0015610 ~ −0.0061284 범위로 작다(006은 원 T 대비 −0.0059539, 부모 대비 +0.0001745). 9구종 쪽 차이는 그보다 크지만 표적이 달라 크기를 직접 비교할 수 없다.

## #37 종료 댓글 초안 (PM 게시용)

```
DONE / RELEASE — NOV14H-20261102 (정리: NOV14H-CLOSE-20261110) / actor: Claude PM
A5000 대여 종료로 서버 접속이 닫혀, 맥미니 기록만으로 정리했다. 정리 문서: docs/pm/research/gpu14h-closeout-2026-11-10.md (branch research/gpu14h-closeout-1110).
장부: launch 170 / recovered 167 / pull_failed 3, GPU fit 실패 0. 11/3 02:49 KST 배차 STOP, 대기 25 미실행.
선정(등록 규칙, 2026 EXPOSED DEVELOPMENT): outcome7 004 lr030_drop020_e30 (3seed ΔNLL −0.0061284 vs 원T), prepitch9 003 state_t32 (3seed ΔNLL −0.0538731 vs 빈도표). 두 표적 NLL은 비교하지 않음. 승격·독립확증 아님.
FINAL: prepitch9 s0/s1 보존, s2 미실행. outcome7 s0~s2는 파일이 T7에 있으나 등록 기대값 키(variant) 불일치로 수용 미기록 → 별도 검토 필요.
미판정: 007 반복확인 채점 없음, 008 채점 STOP(KeyError 'batter'), 009 24 fit 미실행, 010 등록만, H1 시도 1·2 STOP·시도 3 검토 대기.
이 작업 ID의 파일·자원 점유를 반환한다. Codex 브랜치·worktree·T7 산출물은 수정하지 않았다. 후속 실행은 별도 CLAIM·등록·검토로만.
```

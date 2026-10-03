# P30 (DRAFT-CUDA-WORKLOAD-DROPOUT-20261004) 구현 검토 — Opus 5.5, 2026-10-04

- 대상: `d35f1c5` vs `c7bbc21` (7 files, +1468/−5). 읽기 전용 검토, 구현 수정 없음.
- **같은 모델 검토**(작성·검토 모두 Claude 계열)이다. 다른 모델의 독립 검토 관문을 대신하지 않는다.
- 범위: 실데이터·GPU·SSH 없음, 전체 테스트 스위트 안 돌림. Mac mini CPU, `OMP_NUM_THREADS=1`, 관련 테스트만 실행. 채택·판정 없음.

## 요약
- 막는 수준(Blocker/High) 결함 **없음**. 10팔 × 새 시드 [3..7] = 50 fit, 12 대비, 옛 경로 보존, 동결 소스 무변경 모두 확인했다.
- 남은 위험은 운영 쪽이다. (M1) 가족 예산 36000초를 50 fit으로 나누면 fit당 평균 720초 이하여야 하는데 P27/P29 fit 시간이 레포에 없어 **미측정**. (M2) MODEL_SOURCES가 늘어나 옛 가족을 새 코드로 재개·검증하면 identity가 어긋난다. (L1–L4) 사소한 가드 빈틈.
- 실제 CUDA 스모크는 **여전히 필요**하다(아래 §6).

## 1. 설계 일치 — 확인됨
- 팔 순서(설정 = `ARM_ORDER`): control, periodic, periodic_d10, periodic_d20, periodic_counts, _d10, _d20, periodic_block, _d10, _d20 → 10팔. `seeds [3,4,5,6,7]`, `fits 50`, 대비 12개, `status DRAFT`, `execution.enabled false`.
- `source_configs` sha256 4개 모두 현재 파일과 일치. `export`·`device`·`training`·`calibration`은 P27(DATA-COMBO-RUN)·P29(DROPOUT-RUN) 설정과 동일(dict 비교 True).
- 드롭아웃 0 팔: `control`·`periodic`·`periodic_counts`·`periodic_block`이 P27 설정 항목과 dict 동일, `control`·`periodic`은 P29 항목과도 동일.
  - 참고: DRAFT-CUDA-DIRECT(P24 원본)의 `periodic`은 `priority 2`이고 여기서는 1이다. 네트워크는 같지만, 문서의 "P24/P29 spec 그대로"는 P27/P29 항목 기준이라고 써야 정확하다(Info).
- 연속 인덱스: none = 11..51(32 제외) 40개, counts = +52, block = +52..55(`rest_missing` 55 포함). combo 레지스트리의 `periodic_counts`/`periodic_block` 맵과 같다. 설정에서 각 팔의 꼬리를 확인함(…51 / …52 / …55).
- 대비 12개: 같은 작업량에서 d−0 6개, 같은 드롭아웃에서 counts/block−none 6개. `check_contrasts`가 각 대비가 정확히 한 요인만 바꾸는지, 기준 쪽이 0 또는 none인지, `held` 값, 연속 맵을 다시 검사한다.

## 2. 옛 경로(P24/P27/P29) 의미 — 보존됨
- `config_check`: 작업량-드롭아웃 팔이 없으면 예전처럼 `seeds == [0,1,2]`만 허용한다. 새로 생긴 제한은 `repeat_tag` 키 거부 하나인데, 현재 configs/에서 이 키를 가진 파일은 P30 DRAFT뿐이다(grep). 그래서 기존 설정에 영향이 없다.
- 팔 분기: `combo` 분기가 먼저 실행되고, 새 팔 분기는 그 뒤에 `continue`로 끝나서 기존 분기 순서를 바꾸지 않는다. `build_model`/`load_model`은 `workload_dropout is True`일 때만 새 경로로 간다. 옛 팔에는 이 키가 없다.
- `score_family`는 `workload_dropout`이 없으면 예전처럼 3-tuple을 돌려준다. 결과 키도 조건부로만 추가된다.
- 동결 소스 무변경: `direct_dropout_model.py`, `direct_dropout_registry.py`, `direct_periodic_model.py`, `direct_combo_model.py`, `direct_combo_registry.py`, `matrix_models.py`, `matrix_metrics.py`, `cuda_direct_queue.py` 모두 `git diff --stat c7bbc21 d35f1c5`에서 변경 0이다. 새 모델 파일은 `PeriodicHeadDropoutModel`을 수정 없이 import한다.

## 3. 모델·체크포인트·가드 — 확인됨
- 현재 투구 물리·결과 0: `arm_inputs`의 가드(모든 팔 공통)와 상속된 `DirectPeriodicModel`의 0-physics 가드가 둘 다 적용된다. 작업량 열은 52개 문맥 열 뒤, 공유 열을 떼기 전에 붙는다(P27과 같은 경로).
- 드롭아웃 0 동등성: 드롭아웃 0 팔은 옛 항목·옛 코드 경로를 그대로 쓰므로 구조적으로 같다. 테스트도 none/counts/block 각각에서 비트 단위 일치를 주장한다(통과 확인).
- 로더: `PeriodicHeadDropoutModel.load`가 format과 kind를 확인하고, `load_workload_dropout`이 head_dropout, 연속 맵, `n_context == 52 + len(workload)`를 확인한다. P29의 `load_dropout`은 `n_context == 52`를 요구하므로 P30 작업량 아카이브는 P29 로더에서 거부된다.
- 신원: `expected_identity`에 arm spec, seed, `config_digest`(seeds·repeat_tag 포함), source_sha256이 들어간다.
- 채점: REGISTERED·enabled일 때만 채점한다. 대비 검증은 예측을 읽기 전에 한다. 작업량-드롭아웃 가족인데 대비 키가 없으면 거부한다(스크립트로 재현: `check_contrasts`가 None을 돌려주면 `score`가 raise). `verify_family`는 arms × `config['seeds']` 전체에 state.json이 있어야 통과하므로 50개가 모두 필요하다. 결과에는 `seed_block`과 "새 데이터 아님" 문구가 들어가고, decision과 ranking은 null이다.
- 큐: `Queue`는 GPU를 `min(2, max_parallel)`개까지만 받고, GPU마다 flock을 잡고, 허용 목록 UUID만 쓴다. 설정은 P29와 같은 UUID 2개, `max_parallel 2`다. 큐 코드는 변경 없음. 순서는 `(priority, seed, arm index)`이고 전부 priority 1이다.

## 4. 발견 사항

### M1 (Medium, 운영) — 가족 예산 36000초 / 50 fit의 충분성 미측정
- 근거: `cuda_direct_queue.Queue.spent()`가 모든 status 단계의 seconds를 **합산**한다(벽시계 시간이 아님). `spent() < budget`일 때만 새 task를 띄운다(L319). 50 fit이면 재시도를 포함해 fit당 평균 **720초 이하**여야 다 끝난다. P29는 18 fit에 같은 36000초를 썼다.
- P27/P29의 fit당 시간은 레포에도 `~/gpu-pull`에도 없어 **미측정**이다. 드롭아웃 .2와 56폭 문맥은 처음 쓰는 조합이라 epoch 수(조기 종료)도 달라질 수 있다.
- 영향: 예산이 모자라면 일부 fit이 빠진다 → `verify_family`가 "Incomplete family"로 **전체 채점을 거부**한다(fail-closed라 결과가 오염되지는 않지만 GPU 시간이 날아간다). `no_new_fit_after 19:00`도 같은 위험이 있다.
- 권고: 등록 전에 P27/P29 큐 결과의 `family_seconds / fits`를 읽고 예산과 시각을 다시 정한다. 또는 CUDA 스모크 시간을 근거로 정한다.

### M2 (Medium, 재현성) — 옛 가족을 새 코드로 재개·검증하면 source identity 불일치
- 근거: `own_hashes()` = SELF + MODEL_SOURCES. 이번 커밋이 MODEL_SOURCES에 파일 2개를 더해 dict 키가 바뀌었다. `run_task` L562와 큐의 `is_complete`/`verify_task`는 현재 `own_hashes()`와 state의 `source_sha256`을 비교한다.
- 영향: P27/P29가 아직 진행·재시도 중이거나, 이 커밋 위에서 옛 가족을 다시 띄우거나 완료 여부를 확인하면 identity가 어긋나 완료 task가 미완료로 보이거나 거부된다. 채점은 `--source-manifest`로 피할 수 있다(문서에 적혀 있음). 하지만 **runner·queue 쪽 재개는 옛 커밋 스냅샷에서만 해야 한다**는 점이 문서에 없다.
- 권고: 진행 중인 옛 가족은 그 가족의 스냅샷(`gpu run` 스냅샷 커밋)에서만 재개한다고 인수인계에 한 줄 적는다. 코드 변경은 필요 없다.

### L1 (Low) — 실수형 시드가 가드를 통과함 (기존부터 있던 문제)
- 재현: `seeds = [3.0,4,5,6,7]` → `rc.config_check` **ACCEPTED**(리스트 `==`는 3.0과 3을 같게 본다).
- 이후 흐름: 큐 명령이 `--seed 3.0`이 되고, `task_dir` 경로가 `"3.0"`이 되어 CLI int 경로 `"3"`과 어긋난다. 결국 실행 실패나 채점 시 "Missing"으로 끝나므로 fail-closed다. 옛 `[0,1,2]` 비교에도 같은 성질이 있었다.
- 권고(선택): `check_seeds`에서 `all(type(s) is int ...)`.

### L2 (Low) — `repeat_tag: null` 명시가 허용됨
- 재현: `x['repeat_tag'] = None` → ACCEPTED(`SEED_BLOCKS[None]`). 결과의 `seed_block.repeat_tag`도 null이라 해롭지는 않다. 다만 "첫 블록에는 repeat_tag가 없다"는 문서와 달리 키가 있어도 통과하고, config_digest만 달라진다. Info에 가깝다.

### L3 (Low) — 두 번째 반복 블록의 `experiment_id` 분리를 강제하지 않음
- 재현: `seeds [8..12]` + `repeat_tag "repeat2_seeds_8_12"` → ACCEPTED. `experiment_id`와 출력 경로가 첫 블록과 달라야 한다는 검사는 없다. 시드 디렉터리가 겹치지 않고 `verify_family`가 설정의 시드만 보므로 섞여 채점되지는 않는다. 그래도 같은 ID의 results/ 파일을 덮어쓸 수 있으니 등록할 때 다른 ID를 쓰는지 사람이 확인해야 한다.

### L4 (Low) — 설정 검사는 GPU 1개 허용 목록도 통과
- 재현: `allowed_gpu_uuids`를 1개로 줄이면 ACCEPTED(조건이 `len > 2`만 막는다). 기존 동작과 같다. 큐는 `--gpu` 개수로 제한하므로 "최대 2개"는 지켜진다. "정확히 2개"가 필요하면 별도 검사가 필요하다(Info).

### 알려진 사항(총괄이 고치기로 함)
- `score_cuda_direct.korean_table` L272 `'시드 3개 평균'` 하드코딩. 이 가족은 5시드다. 범위상 다시 다루지 않았다.
- 같은 테이블 계열의 다른 하드코딩은 grep으로 더 찾지 못했다.

## 5. 테스트 실측 (CPU, threads 1)
```
cd experiments/pitchmdp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_cuda_workload_dropout.py -q   # 42 passed in 2.90s
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_cuda_dropout.py tests/test_cuda_direct.py \
  tests/test_cuda_frontier_run.py tests/test_cuda_data_combo.py tests/test_cuda_workload_dropout.py -q    # 176 passed in 22.69s
```
- 작성자 문서(`docs/reviews/CUDA-workload-dropout-2026-10-04.md`)의 수치는 42 / 176이다. 지시문의 "170"과는 다르지만 **실측 176이 문서와 일치**한다.
- 추가 가드 탐침(직접 스크립트, torch 불필요): 실제 설정 통과, 대비 통과, d10 팔 `workload_dropout false` 거부, `head_dropout 0.1000001` 거부, 대비 키 삭제 시 레지스트리 None → scorer 거부 경로, 결과는 위 L1–L4.

## 6. 실제 스모크는 여전히 필요
- CPU 합성 테스트만 있다. CUDA에서의 드롭아웃 난수 흐름, `n_context 56`의 PeriodicHeadDropoutNetwork CUDA 실행, fit당 시간(M1 근거), 피크 메모리는 모두 **미측정**이다.
- `require_executable`은 DRAFT에서 스모크만 허용한다. 스모크 시드도 `config['seeds']` 안(3..7)이어야 한다(0/2/8 거부는 테스트로 확인). P29 선례처럼 같은 코드로 50-task CUDA 스모크를 끝까지 돌리고 산출물 검증을 통과한 뒤에 정식 실행을 권고한다.

## 7. 결론
- 구현은 의도한 10팔 × 시드 3..7 = 50 fit 가족과 맞고, 옛 가족 의미와 동결 소스를 바꾸지 않는다. 시드·설정·해시 쪽에서 결과를 조용히 오염시킬 결함은 찾지 못했다(L1–L3은 모두 fail-closed이거나 해가 없음).
- 등록 전 할 일: M1 예산 근거 확보, M2 재개 규칙 한 줄, 시드 수 캡션 수정(총괄), CUDA 스모크.

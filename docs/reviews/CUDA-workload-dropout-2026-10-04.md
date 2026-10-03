# CUDA periodic 작업량 × 헤드 드롭아웃, 새 모델 시드 (DRAFT, 2026-10-04)

상태: DRAFT, `execution.enabled false`. 실데이터·GPU·SSH 실행 없음. EXP 등록 안 함 — 총괄이 검토·GPU 스모크 후 ID(제안 P30)·날짜·마감·GPU(3/4, 공유 락)를 정해 등록한다.

## 왜 (결과를 본 뒤 설계한 후속)
- P27(총괄 판독): periodic_counts −0.00321 / −0.00130, periodic_block −0.00337 / −0.00142 (보정 직접 헤드 NLL, control 대비, main / mlb_dev).
- P29: periodic에 헤드 드롭아웃만 더한 증분 효과는 불확실.
- 모두 이미 열람한 반복 DEV. 질문: 작업량과 헤드 드롭아웃이 **합쳐지는가**, 그리고 부호가 **새 모델 난수 시드**에서도 유지되는가.
- 새 시드는 새 데이터가 아니고 독립 확증 검정도 아니다(같은 export, 같은 DEV 행). 승격·판정·순위·우승팔·교호작용 판정·2026 없음.

## 설계 (configs/DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json)
flatten control + periodic 3×3 (작업량 none/counts/block × 헤드 드롭아웃 0/.1/.2) = 10팔 × 새 시드 [3,4,5,6,7] = 50 fit.

| 작업량 \ 드롭아웃 | 0 (옛 항목 그대로) | .1 | .2 |
|---|---|---|---|
| none | `periodic` | `periodic_d10` | `periodic_d20` |
| counts | `periodic_counts` (P27 combo) | `periodic_counts_d10` | `periodic_counts_d20` |
| block | `periodic_block` (P27 combo) | `periodic_block_d10` | `periodic_block_d20` |

- 드롭아웃 0 팔 4개(`control`, `periodic`, `periodic_counts`, `periodic_block`)는 P27/P29 설정 항목과 **바이트 단위 동일**(같은 플래그·같은 코드 경로·옛 동작).
- 새 6팔: P29 `direct_dropout_model.PeriodicHeadDropoutModel`을 **수정 없이** 재사용. 동결 periodic 네트워크 + 헤드 GELU 두 곳 뒤 `nn.Dropout(p)`. PLR 수치 임베딩·임베딩되는 연속 입력·원입력은 드롭하지 않음. 파라미터 수는 같은 작업량의 드롭아웃 0 팔과 동일.
- 연속 인덱스는 P27과 동일: none 40개(11..51, 32 제외), counts = +52, block = +52..55 (결측 지시자 `rest_missing` 55 포함).
- `periodic_d10`은 P29 `periodic_dropout`과 같은 네트워크·드롭아웃(플래그·이름만 다름).
- 예산: 30 epoch, patience 5, lr 5e-4, batch 1024, AdamW wd .01, 균등 손실, 외부 온도 보정(같은 범위). 큐: task 3600초, 가족 36000초, 최대 2회 시도, GPU 2개(P29와 같은 UUID — GPU 3/4 대응은 총괄 확인).

## 전체 스키마
최상위 키: `protocol`(cuda_direct_v1), `experiment_id`, `status`(DRAFT), `decision`, `purpose`, `source_configs`(base / frontier / data_combo / dropout_run 경로+sha256), `export`(P29와 동일, manifest pin 동일), `training`, `calibration`, `arm_notes`, `control_reuse`(none), `seeds`, `seed_rule`, `fits`(50), `arms`(10), `comparison`, `device`, `queue`, `data_rules`, `execution`. 두 번째 반복에서만 `repeat_tag`.

새 팔 항목 (`direct_workload_dropout_registry.ARMS`와 정확히 같아야 함):
```
{"name", "priority": 1, "workload_dropout": true, "workload": <레벨의 열 목록>,
 "family": "workload_dropout", "model": "periodic_head_dropout_mlp", "base_model": "periodic_flatten_mlp",
 "width": 128, "head_dropout": .1|.2, "workload_level": "none"|"counts"|"block", "context_continuous": [...]}
```
`comparison.workload_dropout_contrasts` (12개, 각 대비는 한 요인만 바꿈, 후보 − 기준):
- `isolates: head_dropout`, `held: {workload}`: 같은 작업량에서 d10−d0, d20−d0 (3×2 = 6)
- `isolates: workload`, `held: {head_dropout}`: 같은 드롭아웃에서 counts−none, block−none (3×2 = 6)
- flatten control은 각 팔의 vs-control 블록에만 쓰인다(구조가 다르므로 위 대비에 안 들어감). 교호작용은 위 대비의 차로 읽을 수 있으나 별도 판정 대비는 두지 않았다.

## 가드 (기존 + 추가)
- 시드: `workload_dropout` 팔이 있는 설정만 새 시드를 쓸 수 있다. 정확히 [3..7]이고 `repeat_tag` 없음, 또는 정확히 [8..12]이고 `repeat_tag: "repeat2_seeds_8_12"`. 순서·길이·다른 값·0–2·임의 태그는 거부. 그 밖의 모든 cuda_direct_v1 설정은 이전처럼 [0,1,2]만, `repeat_tag`가 있으면 거부(전면 완화 없음).
- 가족: 10팔 이름·순서 고정, 드롭아웃 0 팔은 옛 항목과 정확히 동일, 새 팔은 레지스트리와 정확히 동일(추가 키 = 미검사 파라미터 거부), priority 1, workload는 레벨 열, `fits` 50. P29 `dropout_head` 팔과 섞이면 거부. 두 번째 반복도 같은 고정 팔만.
- 기존 runner 가드 그대로: protocol, TRAINING 고정, 보정 범위, CUDA 전용·GPU 2개·CPU 폴백 없음, 2026 행 금지(last_year 2025, split별 연도), export manifest/배열 해시, 현재 투구 물리·결과 0 가드, DRAFT는 스모크만, 결과 디렉터리 덮어쓰기 금지·실패 보존, 등록 안 된 시드 거부.
- 신원: `expected_identity`(config sha, arm spec, seed, smoke, device, source_sha256)와 state.json 일치. 체크포인트 로더는 kind·head_dropout·연속 맵·context 폭(52 + 작업량)이 다르면 거부.
- 채점: 설정이 REGISTERED·enabled일 때만, 대비 목록을 예측을 읽기 **전에** 검증, 이 가족은 대비 키가 없으면 거부, 50 fit 전부 있어야 함(하나라도 없으면 아무것도 안 읽음), 같은 paired whole-game 부트스트랩 10,000(같은 시드 + 5시드 앙상블), `decision`/`ranking` null, 결과에 `seed_block`과 "새 데이터 아님" 문구.
- 원천: MODEL_SOURCES에 `direct_workload_dropout_model.py`, `direct_workload_dropout_registry.py` 추가(→ runner own_hashes 변경; 옛 가족 재채점은 기존처럼 `--source-manifest`로 그때 소스 맵 고정). SCORER_SOURCES에 레지스트리 추가.

## 한계
- 반복 DEV + 사후 설계: 신뢰구간은 고정 예측 조건부(재학습·보정·선택 불확실성 미포함). 시드를 늘려도 DEV 재사용 문제는 그대로.
- 드롭아웃 .2는 처음 쓰는 값이다(P25/P29는 .1만).
- CPU 합성 테스트만. CUDA에서 드롭아웃 난수 흐름·50 fit 소요 시간은 미측정(총괄 GPU 스모크 필요). 실제 성능 수치 미측정.

## 테스트 (Mac mini CPU, 합성, 스레드 1)
```
cd experiments/pitchmdp
OMP_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_workload_dropout.py -q   # 42 passed
OMP_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_dropout.py tests/test_cuda_direct.py tests/test_cuda_frontier_run.py tests/test_cuda_data_combo.py tests/test_cuda_workload_dropout.py -q   # 176 passed
```
전체 스위트는 돌리지 않음. 확인: 작업량 none/counts/block 각각 드롭아웃 0 경로가 옛 팔(combo/base periodic)과 학습 이력·파라미터·logits 비트 단위 일치, 드롭아웃 팔은 헤드 두 곳에만 Dropout(p)·파라미터 수 동일, 저장·불러오기 일치와 다른 팔 로더 거부, CPU 스모크 task(context 폭 56, 시드 0/2/8 거부, DRAFT 실 fit 거부), 설정·시드 이탈 20종 거부, 반복 태그 허용 범위, 다른 가족 시드 0–2 유지, 대비 부호(후보−기준 = vs-control 차)·한 요인만 변경, 미완료·시드 불일치·config 불일치·대비 누락은 예측 읽기 전 거부. 채점 테스트는 속도를 위해 부트스트랩 200회로 monkeypatch(코드 기본값 10,000 그대로).

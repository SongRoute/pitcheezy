# E4 workload 입력 러너 (DRAFT, 2026-10-03, Opus 5.5 작업자)

범위: Root가 승인한 고정 입력 두 개를 직접(direct) 투구 전 head에 붙이는 러너와 DRAFT config만 만들었다. **실제 데이터 fit·추론·채점, 실제 특성 감사 재실행, 2026 열람, newsrv는 하지 않았다.** 성능은 미측정이다. EXP-P24 ID는 PM이 검토·smoke 뒤 붙인다.

## 산출물
- `experiments/pitchmdp/scripts/run_ml_direct_workload.py` — 새 러너 (protocol `ml_direct_workload_v1`)
- `configs/DRAFT-DIRECT-WORKLOAD-20261003.json` — DRAFT, `execution.enabled=false`, 변형 2개 × 시드 0,1,2
- `experiments/pitchmdp/tests/test_run_ml_direct_workload.py` — 합성 CPU 테스트 12개

## 고정 정의 (결과 보기 전에 고정, 특성 HPO 없음)
| 변형 | 추가 맥락 열 | 추가 폭 | 추가 파라미터 |
|---|---|---|---|
| counts | `prior_records_per100` = 이 경기에서 이 투수의 앞선 Statcast 기록 수 / 100 (상한 없음) | 1 | 256 |
| block | counts + `(rebuilt_thruorder-1)/3`, `log1p(clip(rest_days,0,180))/log181` (결측→0), `rest_missing` | 4 | 1,024 |

- 추가 파라미터: flatten MLP 첫 층이 `Linear(n_flat, 256)`이라 맥락 스칼라 1개당 256개. 전체 개수는 fit마다 `fit.json.parameter_count`에 기록한다(실데이터 폭은 미측정).
- 기록 수에는 자동 볼·스트라이크 같은 비투구 기록이 섞일 수 있다. 피로라고 부르지 않는다.
- 같은 날 앞 경기(더블헤더)는 정의상 휴식일 계산에서 빠진다(`workload_features.rebuild`, d2e2484, 수정 없이 재사용).

## 데이터 경로
- 원본은 `configs/local.json`의 `raw_root` + `raw_allowlist`(정확히 `statcast_2023/2024/2025.parquet`)에서 **`game_date, game_pk, at_bat_number, pitch_number, pitcher` 다섯 열만** 읽는다. 결과·물리·Savant workload·미래 열은 열지 않는다. 파일마다 그 시즌 밖 행이 있으면 거부한다.
- 적격성·코호트 필터 **전 전체 원본**으로 계산한 뒤, 동결된 부모 행에 정확한 키로 붙인다(`one_to_one`). 원본에 없는 키, 중복 키, 날짜·투수 불일치, 2023-2025 밖 부모 행은 거부한다. 행 집합·분할(train/May/June/DEV)은 바꾸지 않는다.
- 추가 열은 sharing 맥락의 끝 7열(군집 플래그 5, 군집 id, 라우팅 id) **앞**에 들어간다. 그래서 `training_arrays`가 자른 global head 입력에는 들어가고, 라우팅은 그대로다. 래퍼(`WorkloadContext`)는 조회할 때마다 키를 다시 확인한다. 추가 열이 0개면 기존 맥락 출력을 그대로 돌려준다.

## 바꾸지 않은 것
EXP-P15-001과 같다: 현재 투구 물리 0, 균일 손실(최근 가중치 없음, P20), PLR 없음, flatten MLP 128 / 30 epoch / patience 5 / lr 5e-4 / batch 1024, May 조기 종료·온도, June blend, 시드 0-2, 장치 auto(맥미니 MPS). 대조군은 동결된 EXP-P15-001과 그 G0 1/2 혼합이고, 채점은 `run_ml_direct_recency.describe`를 재사용해 서술 비교만 한다. 6개(2×3) 모두 끝나기 전에는 채점을 거부한다.

## 안전장치
- identity: direct-head 소스 해시, 러너·`workload_features.py`·`run_ml_direct_recency.py` 해시, 특성 배치(열 이름·삽입 위치·원본 열·상한), 원본 3개 파일 sha256, 장치 옵션과 실제 장치.
- preparation에 특성 sha256을 저장하고, 재개할 때 다시 계산한 값과 다르면 거부한다.
- fit·predict는 상태 파일과 해시가 맞으면 재사용하고, 부분 산출물이 있으면 덮어쓰지 않고 거부한다. calibration.json은 예측이 성공한 뒤에만 쓴다.
- `--smoke`는 행 수와 2 epoch만 쓰고, 출력 이름이 `-smoke`로 끝나는 곳에만 쓴다. `--mlb` 지원.

## 테스트 (맥미니 CPU, 합성 데이터)
```
cd experiments/pitchmdp
DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib \
  /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_run_ml_direct_workload.py -q
# 12 passed in 1.56s
... -m pytest tests/test_run_ml_direct_workload.py tests/test_workload_features.py tests/test_run_ml_direct_head.py \
  tests/test_run_ml_direct_recency.py tests/test_run_ml_direct_periodic.py tests/test_run_ml_direct_residual.py \
  tests/test_run_ml_direct_head_confirm.py -q
# 63 passed in 3.11s
```
확인한 것: 손 계산 값(앞선 기록 수, 9타자 뒤 회차 증가, 180일 상한, 같은 날 앞 경기 제외, 결측 표시), 필터된 부모 행이 아니라 전체 원본으로 센 값, 모든 접두 구간에서 값 불변, 뒤 행 추가해도 앞 값 불변, 섞인 순서 정렬, 모르는 키·중복 키·불일치 거부, 다섯 열만 읽기·2026 allowlist와 시즌 밖 행 거부, 삽입 위치, 0열이면 기존 경로와 같은 배열, fit 입력의 현재 물리 0·예산·sample_weight 없음, 추가 파라미터 1,024개, 저장·로드 동일, 검증된 재사용, 부분 산출물 보존, 특성 바뀐 재개 거부, auto가 CPU로 풀리면 대조군과 비교 불가 표시.

## 다음 명령 (PM 검토 뒤)
```
cd experiments/pitchmdp && DYLD_LIBRARY_PATH=... python scripts/run_ml_direct_workload.py \
  --config ../../configs/DRAFT-DIRECT-WORKLOAD-20261003.json --local-config configs/local.json \
  --output "<runs>/DRAFT-DIRECT-WORKLOAD-20261003-smoke" run --variant block --seeds 0 --mlb --smoke
```
실행 시간은 미측정이다. 프로세스마다 원본 키 열 읽기와 rebuild가 추가된다.

## 검토·수정 (2026-10-04, smoke 전, Opus 5.5 — 작성자와 같은 모델이라 독립 모델 검토 아님)

범위: 러너·테스트·DRAFT config·이 문서만. 공유 동결 모델·헬퍼는 건드리지 않았다. 실제 데이터·GB 원본 재열람·실행·등록 없음.

**주요 결함 (수정함): 원본 바이트가 동결 원본에 고정되지 않았다.**
- 부모 경로(`run_ml_sharing.load_data` → `run_ml_benchmark.regular_frame` → `matrix_data.load_verified_processed_cache`)는 processed·cache 바이트만 매니페스트와 대조하고, 원본은 `"raw_hash_verification": "historical_only"`로 표시만 한다(원본 파일이 있는지만 본다). 그런데 이 러너는 원본 파일 자체를 다시 읽는다.
- 수정 전에는 `raw_file_sha256`을 identity에 그때 계산해 기록만 했다. 첫 실행에서는 비교할 기준이 없어서, 같은 이름의 다른 원본(재수집본 등)이어도 통과하고 그 해시가 기준이 됐다.
- 수정: `FROZEN_RAW_SHA256`(러너 상수)와 config `feature_rules.raw_sha256`을 두고 `config_check`가 둘이 같은지 본다. `read_raw`는 파일 바이트를 한 번 읽어 그 바이트를 해시·대조한 뒤 같은 바이트를 파싱한다(해시와 읽기 사이 교체 불가). `main`은 무거운 부모 로드 전에 identity의 원본 해시를 고정값과 대조한다.
- 고정값 출처 대조(파일 재열람 없이 레포 메타데이터만): `configs/G0-RESEARCH-FROZEN-v1.json`, `configs/EXP-P19-001.yaml`, `results/EXP-P15-001.json` 세 곳 모두 2023 `46c354be…2984730`, 2024 `3cf05edd…a7e9a274`, 2025 `4c785756…dc0249`로 같다. 테스트가 G0 동결 파일과 P15 결과 파일의 statcast 출처 전부를 상수와 대조한다.

**같은 장치 대조군 (보강함)**
- `comparable_device_to_control`은 준비 단계에서 `resolved_device == 'mps'` 문자열 비교였다. 채점에서는 이제 대조군 EXP-P15-001 시드별 `fit.json.report.device`(그 `fit_state.json` 해시 확인 뒤)와 후보 시드별 실제 fit 장치를 짝으로 기록(`fit_device_control_vs_candidate`)하고, 하나라도 다르면 비교 불가로 표시·경고한다. 같은 장치 대조군 arm은 이 러너에 없다. 채점을 거부하지는 않는다(서술 전용, GPU 이전은 Root가 따로 조정).

**확인만 하고 바꾸지 않은 것**
- 접두 인과성: `workload_features.rebuild`는 (날짜, game_pk, 타석, 투구) 정렬 뒤 같은 경기·투수 `cumcount`(현재 제외), 타석 시작 누적 −1(현재 타석 제외), 서로 다른 날짜끼리 `diff`(같은 날 앞 경기 제외)라서 앞선 키만 쓴다. 기존 접두·추가 행 불변 테스트가 있다.
- 현재 결과·물리 없음: 원본에서 5열만 읽고, 입력은 `direct.direct_arrays`가 현재 물리 0으로 만든다(fit 입력 테스트로 확인).
- 부분 예측 거부: 상태 파일 없이 예측 산출물이 하나라도 있으면 덮어쓰지 않고 거부, calibration.json은 예측 성공 뒤 기록, mlb 예측은 패널 상태가 맞아야만 실행.
- 채점은 2변형×3시드 `prediction_state.json`이 모두 있어야 시작하고 `members`가 해시를 확인한다. mlb는 6개 모두 있을 때만 서술한다(일부만 있으면 빠짐).

**테스트 (맥미니 CPU, 합성 데이터)**
```
cd experiments/pitchmdp && DYLD_LIBRARY_PATH=.../torch/lib .../.venv/bin/python -m pytest tests/test_run_ml_direct_workload.py -q
# 14 passed in 1.56s  (새 테스트 2개: 동결 원본 바이트 고정, 실제 fit 장치 대조)
... tests/test_run_ml_direct_workload.py tests/test_workload_features.py tests/test_run_ml_direct_head.py \
  tests/test_run_ml_direct_recency.py tests/test_run_ml_direct_periodic.py tests/test_run_ml_direct_residual.py \
  tests/test_run_ml_direct_head_confirm.py -q
# 65 passed in 3.05s
```
smoke에서 실제 원본 해시가 고정값과 다르면 `Raw Statcast files differ from the frozen 2023-2025 bytes`로 부모 로드 전에 멈춘다. 그 경우 원본 교체 여부를 먼저 확인해야 한다.

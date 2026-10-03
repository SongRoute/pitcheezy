# CUDA 구조 × 드롭아웃 후속 가족 (DRAFT, 2026-10-04)

상태: DRAFT, `execution.enabled false`. 실데이터·GPU·SSH 실행 없음. 총괄이 검토·GPU 스모크 후 ID(제안 P29)·날짜·GPU를 정해 등록한다.

## 왜 (결과를 본 뒤 설계한 탐색)
- P25: flatten 128에 dropout .1 → 보정 직접 헤드 NLL이 대조군보다 dev·mlb_dev 모두 낮음(앙상블 Δ −0.00115 / −0.00087, 시드 3/3). weight decay 단독은 중립.
- P24: periodic은 dev 개선, residual은 mlb_dev 쪽이나 범위 간 불확실.
- 둘 다 이미 열람한 반복 DEV다. 이 가족은 그 결과를 읽은 **뒤** 고른 사후 탐색이며 승격·판정·순위·2026 없음.

## 설계 (configs/DRAFT-CUDA-DROPOUT-RUN-20261004.json)
3×2 요인: 구조(flatten, residual, periodic) × 헤드 드롭아웃(0, .1), 6팔 × 시드 0–2 = 18 fit.
- `control` = MatrixModel flatten_mlp 128 (콤보 가족 control과 동일 항목)
- `dropout` = frontier 레지스트리 `dropout` 그대로 (Dropout .1 after both hidden GELU, AdamW wd .01)
- `residual` = 동결 DirectResidualModel, `periodic` = 동결 DirectPeriodicModel(40개 연속 인덱스) — 같은 가족 새 무드롭아웃 팔
- `residual_dropout` = residual + 출력 헤드 GELU 뒤 Dropout(.1)
- `periodic_dropout` = periodic + 헤드 GELU 두 곳 뒤 Dropout(.1). PLR 수치 임베딩·원입력은 드롭하지 않음
- 모든 팔: 30 epoch, patience 5, lr 5e-4, batch 1024, AdamW wd .01, 균등 손실, 외부 온도 보정, workload 없음, 현재 투구 물리 0 가드 유지.
- 고정 대비 `regularization_contrasts`: dropout−control, residual_dropout−residual, periodic_dropout−periodic (같은 구조, 같은 시드·앙상블, 경기 단위 부트스트랩 10,000, 같은 확률 체인). 각 팔 vs control 블록도 그대로 유지.

## 한계 (꼭 읽을 것)
- **residual은 0 vs .1이 아니다.** 동결 residual 블록에 이미 Dropout(.1)이 있다(등록된 구조). 따라서 residual 대비는 "블록 드롭아웃 위에 헤드 드롭아웃 .1 추가"다. 세 구조의 드롭아웃 '양'은 같지 않다(flatten 2곳, residual 헤드 1곳 추가, periodic 2곳).
- 원천 게이트: MODEL_SOURCES에 `direct_dropout_model.py`, `direct_dropout_registry.py`가 추가되어 현재 runner의 own_hashes가 바뀐다. P24/P25/P27 과제는 이 가족과 섞을 수 없고(control_reuse none), 옛 가족 재채점은 기존처럼 `--source-manifest`로 그때 소스 맵을 고정해야 한다. scorer의 SCORER_SOURCES에 세 레지스트리를 추가했다.
- 데이터 게이트: 기존 고정 export 그대로(manifest pin 동일). 2026 행 없음. DEV/mlb_dev는 예측만, 선택에 쓰지 않음.
- 반복 DEV + 사후 선택: 신뢰구간은 고정 예측 조건부(재학습·보정·선택 불확실성 미포함).
- CPU 합성 테스트만 했다. CUDA에서 드롭아웃 난수 흐름은 미검증(총괄 GPU 스모크 필요). 실제 성능 수치는 미측정.

## 구현
- 신규 `pitchmdp/direct_dropout_registry.py`(torch 없음): 팔·참조·대비 고정, `check_family`/`check_contrasts`.
- 신규 `pitchmdp/direct_dropout_model.py`: 동결 네트워크를 상속해 같은 모듈을 Dropout과 함께 다시 감쌈(초기화 순서·파라미터 수 동일), 학습은 상속한 DirectResidualModel.fit, 자체 아카이브 형식, load는 종류·동결 구조·head_dropout·연속 맵이 다르면 거부. 동결 파일(matrix_models / direct_residual_model / direct_periodic_model) 수정 없음, 몽키패치 없음.
- runner: `"dropout_head": true` 팔 검사·build/load 분기, 해당 팔이 있으면 6팔 가족 전체를 고정 검사.
- scorer: `comparison.regularization_contrasts`를 예측 읽기 전에 검증해 같은 paired_block으로 채점, 한국어 표 절 추가. 키가 없는 기존 가족은 이전과 동일.

## 테스트 (Mac mini CPU, 합성)
```
cd experiments/pitchmdp
/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_dropout.py -q        # 32 passed
/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_direct.py tests/test_cuda_frontier_run.py tests/test_cuda_data_combo.py tests/test_cuda_dropout.py -q   # 134 passed
```
전체 스위트는 돌리지 않음. 확인 내용: dropout 0 경로가 동결 기본과 학습·logits 비트 단위 일치(CPU 소규모 fit), 파라미터 수 동일, train 모드 드롭아웃 활성/eval 모드 꺼짐, 저장·불러오기 후 head_dropout·logits 보존과 다른 팔/기본 로더 거부, 설정 이탈 거부, 대비 부호(후보−기준) 검증, 현재 물리 0 가드, 미완료 가족·잘못된 대비는 예측 읽기 전 거부.

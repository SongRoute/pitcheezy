# 직접 head 최근성 가중(E3) 구현 보고 (2026-10-03)

작성: Claude Opus 5.5 (Codex PM 위임, 단일 작성자, 하위 에이전트 없음). 기준 `8b41992`. 검토 없음 — 등록 전 root PM 검토 필요.
지위: **DRAFT.** 실데이터 fit·예측·채점 0회. 실험 ID 미배정. 봉인 소스·모델·config·결과는 수정하지 않았다.

## 바뀐 파일

- `experiments/pitchmdp/scripts/run_ml_direct_recency.py` (새 runner, 추가만)
- `experiments/pitchmdp/tests/test_run_ml_direct_recency.py` (합성 테스트 11개)
- `configs/DRAFT-DIRECT-RECENCY-20261003.json` (`status: DRAFT`, `execution.enabled: false`)
- 이 문서

## 무엇이 바뀌는가 (한 가지)

TRAIN 손실 가중치 `w = exp2(-age/half_life)`, age = 2025-04-30에서 거슬러 센 일수, 평균 1로 정규화. 입력은 `game_date`뿐이며 결과·예측·DEV는 쓰지 않는다. half-life 격자 {180, 365, 730}일은 미리 고정한 family 변형이고 각각 같은 고정 대조와만 비교한다.

그대로인 것: EXP-P15-001 입력(현재 공 물리 0, `direct_arrays`·`NoPhysicsDelivery` 재사용), `flatten_mlp` width 128·epochs 30·patience 5·batch 1024·lr 5e-4, **가중하지 않은** May 1–15 early stop, May 16–31 Cpanel temperature, June Cpanel blend, Cpanel DEV·전체 MLB DEV(1 draw). 가중치는 기존 `MatrixModel.fit(sample_weight=...)`로 넘긴다. 이 함수는 가중치를 평균 1로 다시 나누고 `(losses * w).mean()`을 쓰며, `sample_weight=None`은 모두 1과 같다. 그래서 가중치가 모두 같으면 P15 경로와 같다.

대조: 봉인 EXP-P15-001 직접 head(균등, preparation sha256 `2e3f553f…`, seed 0 fit.json 기준 `device: mps`)와 그 head와 G0-global의 고정 1/2 혼합(G15 mix 구성), seed 0–2. 대조 쪽 새 fit은 0이다.

## 안전장치

- `config_check`: protocol·격자·anchor·가중 규칙·학습 설정·대조 ID·혼합 1/2·해시 pin을 확인한다. DRAFT config는 `--smoke`만 허용하고, 본 실행은 `status REGISTERED` + `enabled true`일 때만 된다.
- `verify_control`: P15 preparation 해시 pin을 맞추고 artifact 해시를 검사한다. **현재 `run_ml_direct_head.py` 계열 소스 해시가 P15 봉인 identity와 같아야** 실행한다(12:0x KST 확인: 같음).
- 부모 G0 preparation 해시·소스 검사(`verify_parent`)와 2026 행 거부(`guard_dates`, LAST_YEAR 2025)는 기존 `load`를 그대로 쓴다. anchor 뒤 날짜·결측 날짜·잘못된 half-life는 가중치 단계에서 거부한다.
- identity = 직접 head identity(config·local·소스·환경) + 이 runner sha256 + variant·half-life·anchor. 실행 중에는 단계마다 소스가 바뀌었는지 확인한다. preparation은 소스 사본과 `registered_config.json` 스냅샷을 남긴다.
- 출력은 `<family>/<variant>/` 아래 preparation과 `<seed>/`로 나뉜다. 완료 state는 해시를 검증한 뒤 재사용하고, 부분 산출물이 있으면 거부한다(`Incomplete fit/preparation`). 같은 디렉터리에 다른 variant를 쓰면 `identity changed`로 거부한다. 기존 heavy lock을 쓰며, 출력은 `ML-MATRIX-20260924` 아래 새 디렉터리여야 한다.
- `score`: 3 variant × 3 seed가 **모두 끝나야** 연다. 서술 비교만 `analysis/descriptive.json`에 쓴다. N 판정·우승 선택·repo `results/` 쓰기·정책 주장은 없다.
- 비용 기록: fit.json(학습 전체·배열 준비 초 분리), prediction_runtime.json(Cpanel), mlb_prediction_runtime.json(전체 MLB)을 따로 남긴다. 시간 상한은 smoke의 행·epoch 제한(train 8,192·earlystop 2,048·2 epoch)뿐이다. 벽시계 timeout은 넣지 않았다.

## 테스트

`pytest tests/test_run_ml_direct_recency.py tests/test_run_ml_direct_head.py -q` → **20 passed**(새 11 + 기존 9). 검사 내용:
- 시간 순서·평균 1·정확히 1 half-life에서 2배, ESS 정의와 짧은 half-life에서 ESS 감소
- 같은 날짜면 정확히 균등, half-life가 매우 크면 균등에 수렴
- 미래(2025-05-01, 2026)·결측·파싱 불가 날짜와 0·음수·inf·bool half-life 거부
- DRAFT config의 본 실행 거부, 격자·anchor·비중·width·해시 변경 거부, 학습 설정이 P15와 같음
- identity에 runner 해시·variant 기록
- 가짜 모델로 fit에 날짜 가중치가 전달되는지, early stop 무가중, 현재 공 물리 0, variant별 state 분리·검증 재사용·identity 불일치·부분 산출물 거부
- 서술 비교: 같은 예측이면 Δ 0, 판정 키 없음

물리 비누수 실데이터형 검사는 기존 `test_run_ml_direct_head.py`가 맡는다.

## 실행 전 측정 (결과 없음, 날짜만)

TRAIN 키(`EXP-P4-001/train_keys.parquet`, 1,252,824행)와 원천 2023–2025 parquet의 `game_pk`·`game_date` 두 열만 읽었다(라벨·결과 열 미열람, 0.8초). 결측 날짜 0, 범위 2023-05-15 ~ 2025-04-30. 행 수: 2023년 488,708, 2024년 644,486, 2025년 119,630.

| variant | ESS 비 | 가중 몫 2023/2024/2025 | 최소–최대 w |
|---|---|---|---|
| hl180 | 0.577 | 0.115 / 0.570 / 0.315 | 0.223–3.513 |
| hl365 | 0.856 | 0.231 / 0.583 / 0.186 | 0.517–2.013 |
| hl730 | 0.961 | 0.306 / 0.558 / 0.136 | 0.733–1.447 |

균등일 때 몫은 0.390 / 0.514 / 0.095다(행 수 비율로 계산).

## 실행 명령 (root 배정 뒤)

공통 접두어(P15와 같은 환경):
```
cd experiments/pitchmdp && DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib \
  /Users/song/Projects/pitcheezy/.venv/bin/python scripts/run_ml_direct_recency.py \
  --config ../../configs/DRAFT-DIRECT-RECENCY-20261003.json --local-config configs/local.json
```
- smoke(DRAFT로 가능, 실데이터 적재 — root 허가 뒤): `... --output "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/DRAFT-DIRECT-RECENCY-20261003-smoke" run --variant hl365 --seeds 0 --mlb --smoke`
- 본 실행: config를 배정 ID로 복사·`experiment_id`/`status REGISTERED`/`execution.enabled true`로 커밋한 뒤 `... --output ".../<ID>" run --variant {hl180|hl365|hl730} --seeds 0 1 2 --mlb`(variant 3번)
- 채점: `... --output ".../<ID>" score`

비용 추정(측정 아님): P15 seed 0 실측(fit 70.3초, Cpanel 예측 7.2초, 전체 MLB 1 draw 123.4초) + 자료 적재로 variant·seed당 약 4–6분, 9 fit + `--mlb` 약 40–55분, 맥미니 MPS.

## 막힌 것·남은 일

- 실행 허가와 ID 배정(root). 등록 전에 다른 모델 검토가 필요하다.
- config의 경로가 맥미니 절대경로라 newsrv에서는 별도 local config와 경로가 필요하다. CUDA 실행은 대조(MPS)와 장치가 섞이므로 이 family는 맥미니 실행을 권한다.
- 등록 판정 규칙(N 문턱·Holm k=3·탐색 후 선택 표시)은 아직 없다. 지금 `score`는 서술만 한다.
- 관련 4파일(`test_run_ml_direct_recency`·`direct_head`·`direct_head_confirm`·`delivery_pool`) 28 passed. 전체 `pytest tests -q`는 돌리지 않았다.

## Codex PM review and registration

Reviewed source and reproduced 28 focused tests. Actual DRAFT smoke hl365/seed0 completed in about10s (including supervisor), 2 epochs on 8192 TRAIN rows, Cpanel and MLB predictions256rows each; maxRSS5.63GB. Frozen original source hashes verified; no scores read. Registered EXP-P20-001 before full fits: all3half-lives × seeds0,1,2; 600s per stage,7200s family wall budget and existing demo-sync quiet windows. Full comparison is descriptive only, official scientific verdicts unchanged.

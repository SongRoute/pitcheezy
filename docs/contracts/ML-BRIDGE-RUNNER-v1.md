# F1 batter-representation bridge runner, version 1

상태: **synthetic 검사만, 실제 실행 없음.** 실제 준비·profile·fit·predict·점수는
한 번도 실행하지 않았다. 부모 해시(`ROOT_TO_ASSIGN`)를 총괄이 채우고 등록한 뒤에만
실행한다. 비교·판정 규약의 원문은 [ML-BATTER-BRIDGE-v1.md](ML-BATTER-BRIDGE-v1.md)이며
이 문서는 그 구현(실행기·scorer)의 CLI·산출물·검증 규칙만 정한다. 규약과 충돌하면
규약이 우선한다.

## 파일

- `experiments/pitchmdp/pitchmdp/matrix_bridge.py`: 마스킹(`context[:, 11:28]=0`),
  `BatterMaskedModel`, `masked_g0_predictor`, `fit_masked`, `network_signature`,
  D100 TRAIN 타자 표본 수·q25.
- `experiments/pitchmdp/scripts/run_ml_bridge.py`: `prepare`/`profile`/`fit`/`predict`/`status`.
- `experiments/pitchmdp/scripts/score_ml_bridge.py`: 전체 masked 3개 완료 후 점수.
- `configs/EXP-P9-001.yaml`: 템플릿. 고정값은 코드 상수와 정확히 같아야 한다.
- 테스트: `experiments/pitchmdp/tests/test_ml_bridge_runner.py`, `test_ml_bridge_scoring.py`
  (도움 모듈 `bridge_synthetic.py`).

frozen G 소스(`matrix_sharing.py`, `run_ml_sharing.py`, `score_ml_sharing.py` 등)는
수정하지 않는다. G의 소스 해시는 부모 preparation identity 검증으로 강제한다.

## 설정

`config_check`는 스키마를 정확히 요구한다(추가 키는 `registration`만 허용).
`protocol=ml_bridge_v1`, `parent_experiment_id=EXP-P4-001`, `parent_cell=G0-global`,
`scope=Cpanel`, seeds `[0,1,2]`, 400draw, `flatten_mlp` width128, budget
30epoch/patience5/batch1024/lr.0005, mask `{start:11, stop:28}`, context 52 + routing 7,
표본 수(TRAIN1,252,824·early16,000·May2,603·June4,821·DEV12,334/328경기),
profile(65,536구×2epoch·early2,048·May64구), 한도(600/7,200/14,400초),
bootstrap(10,000회·seed20260924), 판정 문턱, R(12집단×2지표×1비교=24, 100,000회,
30경기·500구, NLL+.010/Brier+.002), 타자 q25(linear, 경계 low 포함).
`parent_preparation_sha256`·`parent_analysis_sha256`는 64자리 hex여야 하며
`ROOT_TO_ASSIGN`이면 거부한다. `device`는 `auto`로 고정한다(G0 기본값). fit·predict 전에
현재 기본 장치가 보존 G0 `fit.json`의 `report.device`와 다르면 거부한다.
`registration.output`이 있으면 `--output`과 같아야 한다.
`run_ml_bridge.py`의 source 해시 목록에는 `score_ml_bridge.py`와
`matrix_group_metrics.py`도 들어 있어 판정 규칙이 prepare 시점에 동결된다.

## CLI

```
DYLD_LIBRARY_PATH=<torch lib> python scripts/run_ml_bridge.py \
  --config configs/EXP-P9-001.yaml --local-config <local.json> --output <runs/ML-MATRIX-20260924/EXP-P9-001> \
  prepare | profile | fit --seed S | predict --seed S | status
python scripts/score_ml_bridge.py --config ... --local-config ... --output ...
```

- `status`를 제외한 모든 명령(scorer 포함)은 ML matrix heavy lock을 잡는다(동시 무거운 작업 1개).
- `ledger.jsonl`: 명령 진입 시(검증·로드 포함) `start`, 종료 시 `end`(completed/failed, 초)를
  남긴다. `end` 없는 `start`(SIGKILL·OOM)는 다음 `start`까지의 벽시계 시간, 단계 한도
  (profile 600, 그 외 7,200)로 상한을 둔 값으로 보수적으로 청구한다. fit·predict는
  ledger(현재 명령 제외)+현재 경과+projection이 14,400초를 넘으면 시작하지 않는다.
- **시간 제한은 호출자가 강제한다**: profile `timeout 600`, fit `timeout 7200`,
  predict `timeout (7200 − 기록된 fit 초)`(`status`가 출력). 코드는 사후 검사와
  profile 외삽 gate만 한다. SIGTERM은 `SystemExit`로 바꿔 ledger에 실패를 남긴다.
- 완료 단위 resume은 identity·해시 검사 후에만 한다. 불완전 산출물은 실패 검토를
  요구하며 덮어쓰지 않는다.

## 단계별 규칙

**prepare** — 부모 G(`EXP-P4-001`)의 `registered_config.json`으로 `run_ml_sharing.verify`
(identity·소스·artifact·external 해시)를 다시 통과시키고, preparation·panel analysis
(results/manifest/predictions) 해시를 config와 대조한다. G0-global seeds0/1/2의 fit state·
prediction state·dependency(자기 `global/model.pt`만)·calibration을 확인한다. 동일
TRAIN/early/May/June/DEV 키 파일, `aux.pkl`, `baseline_predictions.npz`,
`dev_metadata.parquet`, `parent_preparation.json`을 복사하고 해시 일치를 확인한다.
정규화·타자 통계·투수 표현·frequency·delivery는 **재적합하지 않는다**. 보존 G0 member로
`summarize_cell`을 다시 계산해 raw/calibrated/primary/seed_primary 배열과 June 가중치
(앙상블·seed별), 주요 지표가 frozen analysis와 **정확히** 같은지 확인한다(다르면 거부).
실제 자료를 한 번 읽어 key 재구성, context 배치(기본 특성 이름 순서, 기본 28=11+타자 17,
투수 표현 23+표본 수 1, routing 7로 총 59), DEV metadata
타자 정렬을 확인하고 D100 TRAIN 타자별 표본 수와 q25를 `batter_train_volume.json`에
고정한다. masked DEV 점수는 계산하지 않는다.

**profile** — TRAIN/early/May 표본만 선택해 로드한다(June·DEV 행은 선택하지 않음). TRAIN 앞
65,536구×2epoch(patience 2), early 2,048구로 masked 모델을 학습하고 May 앞 64구로 delivery
temperature 보정·400draw 적분을 잰다. 30epoch·전체 CAL/DEV·명령별 시작/검증 overhead와
로드를 선형 외삽해 member(fit+predict) ≤7,200초,
family(준비+profile+3member) ≤14,400초 gate를 기록한다. 600초 초과 시 실패.

**fit --seed S** — profile gate, 장치, ledger 잔여 예산을 확인하고, 학습 전에 masked 입력
모양(n_context 52, n_token, length)이 보존 G0 network config와 같은지 확인한다. `SharingContext` 59채널에서
routing 7개를 제거한 52채널의 `[11:28]`을 학습·early stopping 모두 0으로 두고
`MatrixModel('flatten_mlp', seed=S, width=128)`을 학습한다(optimizer/weight decay는 frozen
소스 그대로). 학습 후 네트워크 config·파라미터 수·state_dict 모양이 보존 G0 seed S 모델과
같지 않으면 거부한다. `best_training.pt`는 학습 종료 시 저장한 최선 early-stop 상태다
(에폭별 체크포인트가 아님).

**predict --seed S** — 먼저 보존 G0 seed S 모델과 기록된 temperature로 June 앞 64구를
공유 `SharingPredictor` 경로로 재계산해 보존 예측과 비교한다(최대 절대 차 ≤1e-6, 아니면
거부). 이어 masked 모델을 **같은 `SharingPredictor('G0-global')` wrapper**(내부 모델만
마스킹)로 감싸 May temperature를 새로 적합하고 June·DEV 적분 예측을 저장한다. full 예측은
만들지 않는다.

**score** — masked 3개 모두 완료 전에는 열지 않는다. 보존 G0 재구성을 다시 검증한 뒤
masked의 June seed/앙상블 혼합을 같은 `summarize_cell`로 적합한다.
- 주 family: NLL `full−masked` 1개. ΔNLL≤−.003, 경기 paired 95%CI 상한<0, 단측
  null-centered p≤.05, ΔBrier CI 상한≤+.001, 3seed 중 2개 이상 음수(seed끼리 짝).
  10,000회·seed20260924·whole-game·투구 가중. 다중성 조정 없음(단일 가설).
  반대 방향 점 추정은 표기만 하고 자동 승격하지 않는다.
- R: 12집단×2지표=24 slot을 모두 유지. Bonferroni .05/24, 100,000회, 30경기·500구
  gate, `full−masked` 동시 상한 ≤ NLL .010/Brier .002. 표본이 없는 `volume_zero`는
  `structural_missing=true`로 표기하며 통과로 세지 않는다(미확인).
- 타자별·D100 TRAIN 표본 수 zero/low(≤q25)/high 결과는 기술 통계다.
- 비용: 보존 G0의 논리적 fit/predict 초와 이번 family의 실제 ledger 지출(실패 포함)을 분리.
- 산출물 `analysis/bridge/{results.json, predictions.npz, manifest.json}`. manifest는
  SHA256과 모든 입력 해시를 담고, 세 파일을 읽기 전용으로 바꾼다. 기존 폴더가 있으면 거부.

## 산출물 배치

```
<output>/preparation.json, registered_config.json, sharing_preparation.json,
         parent_preparation.json, aux.pkl, baseline_predictions.npz, dev_metadata.parquet,
         {train,earlystop,temperature,blend,dev}_keys.parquet, batter_train_volume.json,
         source/…, ledger.jsonl (가변, 해시 대상 아님)
<output>/profile/{profile.json,state.json}
<output>/members/masked/seed{S}/{model.pt,best_training.pt,fit.json,fit_state.json,
         calibration.json,predictions.npz,prediction_runtime.json,prediction_state.json}
<output>/analysis/bridge/{results.json,predictions.npz,manifest.json}
```

## 범위 밖

MLB 전체(`mlb_dev`) 예측, 추가 5seed 확인, F4 전이, 정책 가치·직접 목표 위치 주장.

## 총괄 결정(2026-09-27, 실행 전)

- full arm 재구성 검증의 June 64행 replay 허용오차 1e-6은 T3 stress 규약의 동등성 기준과 같으므로 유지한다. 정확한 비트 일치는 요구하지 않는다.
- `prepare`의 시간 상한은 규약에 없으므로 중단 시 member 상한 7,200초로 보수 계상한다.
- fit+predict 합산 7,200초 제한은 predict 상한을 `7,200 − 기록된 fit 초`로 강제하는 현재 구현을 따른다.
- `volume_zero`는 이 panel에서 구조적 결측(D61)이므로 R은 `passed`가 아닌 `unconfirmed(structural)`까지만 나온다.
- 실패한 fit은 부분 산출물을 보존하고 같은 ID에서 재시도하지 않는다. 필요하면 새 attempt ID를 등록한다(EXP-P2-001-v2 선례).

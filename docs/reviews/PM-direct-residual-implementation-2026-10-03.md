# 직접 head ResNet형 backbone(E1) 구현 보고 (2026-10-03)

작성: Claude Opus 5.5 (Codex PM 위임 D165, 단일 작성자, 하위 에이전트 없음). 기준 `f19c69e`. 검토 없음 — 등록 전 root PM 검토 필요.
지위: **DRAFT.** 실데이터 fit·예측·채점 0회. 실험 ID 미배정. 봉인 소스(`matrix_models.py`, `run_ml_direct_head.py` 계열, `run_ml_direct_recency.py`)·config·결과는 수정하지 않았다.

## 바뀐 파일 (모두 새 파일)

- `experiments/pitchmdp/pitchmdp/direct_residual_model.py` — 네트워크·학습 클래스·전용 저장 형식
- `experiments/pitchmdp/scripts/run_ml_direct_residual.py` — runner
- `experiments/pitchmdp/tests/test_run_ml_direct_residual.py` — CPU 합성 테스트 10개(전체 약 2초)
- `configs/DRAFT-DIRECT-RESIDUAL-20261003.json` — `status: DRAFT`, `execution.enabled: false`
- 이 문서

## 무엇이 바뀌는가 (한 가지: backbone 구조 계열)

문헌: Gorishniy et al. 2021 (https://arxiv.org/abs/2106.11959, 공식 코드 https://github.com/yandex-research/rtdl-revisiting-models). **충실한 재현이 아니라 적응이다.** 제3자 코드는 받거나 쓰지 않았다.

정확한 네트워크(config 고정, HPO 0회):

```
x = concat(flatten(mask된 tokens), mask, context)      # flatten_mlp와 같은 입력, n_flat 286
z = Linear(286, 128)
2회: z = z + Linear(256,128)(Dropout(.1)(GELU(Linear(128,256)(LayerNorm(z)))))
logits = Linear(128, 10)(GELU(LayerNorm(z)))
```

- 논문 블록과 다른 점: ReLU 대신 GELU(대조 flatten_mlp와 같게 두어 활성함수가 두 번째 변경이 되지 않게), hidden dropout .1 하나, residual-branch dropout 없음, feature 전처리 추가 없음.
- 깊이·폭 모양·정규화·dropout이 함께 바뀐다. 이것을 "구조 계열 하나"로 공개한다.
- 파라미터 수: 후보 **170,634** vs 대조 **141,834**(EXP-P15-001 seed 0 fit.json, 입력 n_context 52·n_token 38·length 6), 1.20배. 용량 효과와 구조 효과가 섞인다 → 같은 파라미터 수의 넓은 MLP 대조는 후속.

그대로인 것: EXP-P15-001 입력(현재 공 물리 0, `direct_arrays`·`NoPhysicsDelivery` 재사용), 10 결과, AdamW(lr 5e-4, wd .01)·epochs 30·patience 5·batch 1024, 가중하지 않은 May 1–15 early stop, May 16–31 Cpanel temperature, June Cpanel blend, Cpanel DEV·전체 MLB DEV(1 draw). 대조: 봉인 EXP-P15-001 직접 head와 그 head + G0-global 고정 1/2 혼합(G15 mix), seed 0–2. 대조 쪽 새 fit 0.

## 학습 코드 재사용 방식과 이유

`MatrixModel.fit`이 `MatrixNetwork`를 함수 안에서 직접 만들기 때문에, 봉인 파일 수정이나 전역 monkeypatch 없이 네트워크만 바꿔 끼울 방법이 없다. 그래서 `DirectResidualModel(MatrixModel)`이 `logits`·`predict`·`_base_report`는 상속하고, `fit`은 torch 분기를 줄 단위로 옮긴 사본에서 네트워크 생성만 `self.network(...)`로 바꿨다(시드 순서, AdamW, 섞기, 가중 CE, clip 5, early stop min delta 1e-5, best state 복원 동일).

동등성 증거: 테스트 `test_mirrored_fit_reproduces_baseline_matrix_model_fit`이 같은 사본 loop에 flatten_mlp 네트워크를 넣었을 때 `MatrixModel.fit`과 state_dict·history·best epoch·update 수·logits가 **비트 단위로 같음**을 CPU에서 확인한다. 봉인 `fit`이 바뀌면 이 테스트가 깨진다.

저장 형식은 `direct_residual_model_v1`(전용 `load`, `torch.load(weights_only=True)`). 봉인 `MatrixModel.load`는 이 파일을 거부하고, 전용 `load`는 `matrix_model_v1` 파일과 architecture와 맞지 않는 network_config를 거부한다.

## 안전장치

- `config_check`: protocol·seed·대조 ID·혼합 1/2·대조 학습 설정(P15 그대로)·후보 구조(상수 `ARCHITECTURE`와 일치)·예산·HPO 0·등록 파라미터 수(입력 계약으로 다시 계산)·해시 pin. DRAFT는 `--smoke`만, 본 실행은 `REGISTERED` + `enabled true`.
- `fit`: 실제 입력 모양이 등록 입력 계약(52/38/6)과 다르거나 학습된 파라미터 수가 등록값과 다르면 거부.
- identity = 직접 head identity(config·local·P15 소스·환경) + 이 runner·모델 파일·재사용한 recency runner의 sha256 + 구조 + `device_option`. 단계마다 소스 변경 검사, 바뀌면 거부(테스트 있음). preparation은 소스 사본·config 스냅샷을 남긴다.
- 대조 확인은 `run_ml_direct_recency.verify_control` 재사용(P15 preparation 해시 pin + 현재 직접 head 소스 해시 = P15 봉인 identity).
- 2026 행 거부·부모 G0 검사는 기존 `load`(LAST_YEAR 2025) 그대로.
- 완료 state는 해시 검증 뒤 재사용, 부분 산출물·identity 변경은 거부. heavy lock 그대로.
- `predict`: `run_ml_direct_head.predict`의 사본이며 loader만 `DirectResidualModel.load`로 다르다(봉인 함수가 `MatrixModel.load`를 고정 사용).
- `score`: 3 seed 모두 끝나야 연다. `run_ml_direct_recency.describe/describe_mlb` 재사용, 서술 비교만 `analysis/descriptive.json`. N 판정·`results/` 쓰기·정책 주장 없음.

## 장치

`--device auto`(기본) = MPS 있으면 MPS, 없으면 CPU — 대조의 규칙과 같다. `mps/cpu/cuda`를 명시할 수 있으나 identity에 들어가고(재개 시 바뀌면 거부) `comparable_device_to_control`이 cpu/cuda에서 false가 된다. **CPU/CUDA 점수는 MPS 대조와 자동 비교 대상이 아니다.** newsrv·GPU 사용 없음. MPS 경로는 이번에 실행하지 않았다(미검증).

## 비용

**미측정.** smoke로 측정한다. 참고(대조 EXP-P15-001 seed 0): fit 70.3초, Cpanel 예측 7.2초, 전체 MLB 1-draw 예측 123.4초.

## 테스트 (CPU, 실데이터 없음)

`pytest tests/test_run_ml_direct_residual.py` 10 passed(1.8초). 직접 head·recency·confirm·matrix 관련 기존 테스트 포함 137 passed. 다룬 것: 봉인 fit과 비트 동등, 고정 구조·파라미터 수, residual 경로·mask·dropout(eval off), 체크포인트 왕복·스키마·교차 loader 거부·변조 거부, config 거부 규칙, identity·소스 변경 거부·장치, runner fit/predict 배선·재개·입력 계약·부분 산출물 거부, 현재 공 물리 0, prepare 거부, 서술 비교.

## 실행 명령 (root PM이 등록 후; 지금은 실행하지 않음)

EXP-P20-001과 같은 heavy lock을 쓰므로 동시에 돌지 않는다. 앞 실행이 끝난 뒤에만.

```
cd experiments/pitchmdp && DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib \
  /Users/song/Projects/pitcheezy/.venv/bin/python scripts/run_ml_direct_residual.py \
  --config ../../configs/DRAFT-DIRECT-RESIDUAL-20261003.json --local-config configs/local.json \
  --output "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/DRAFT-DIRECT-RESIDUAL-20261003-smoke" \
  run --seeds 0 --mlb --smoke
```

본 실행(등록 ID `<ID>`로 config 복사·`status REGISTERED`·`enabled true` 후): 같은 prefix에 `--config ../../configs/<ID>.yaml --output <runs>/<ID> run --seeds 0 1 2 --mlb`, 끝나면 `--output <runs>/<ID> score`. `--device`를 쓰면 run과 score에 같은 값을 준다.

## 남은 일·한계

- 실데이터 smoke 미실행 → 실제 load·MPS 학습·예측 경로는 미검증, 비용 미측정.
- 등록 config(ID·`REGISTERED`)와 판정 규칙(G-dev 등)은 root PM 몫. 현재 score는 서술뿐.
- 같은 파라미터 수 넓은 MLP 대조 없음(용량 vs 구조 미분리).
- `fit`·`predict`는 봉인 함수의 사본이다. 동등성은 fit만 비트 테스트로, predict는 배선 테스트로 확인했다.

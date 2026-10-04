# 가중치 평균(EMA) 실험 설계 — 실행 전 고정 (2026-10-04)

상태: 설계와 코드만 있다. 실행·smoke·등록은 하지 않았다. 검토 상태는 `self_tested; same_model_review_pending; Codex_review_pending`. 판정·순위·채택 필드는 만들지 않는다. D131/D155는 그대로다. 2026 자료는 쓰지 않는다.

## 질문
학습 중 가중치의 지수 이동 평균(EMA)으로 early stop을 판단하고, 그 평균 가중치를 저장·예측에 쓰면, 같은 학습을 원래 가중치로 끝낸 경우보다 보정된 직접 head의 NLL이 낮아지는가.
- 쉬운 말: 매 걸음 위치를 그대로 쓰지 않고, 최근 걸음들의 평균 위치를 쓰면 흔들림이 줄어 예측이 더 안정적인가.

## 왜 이것인가 (이미 해 본 것 확인)
- 이미 한 것:
  - 크기: P25 width145·width203
  - 입력 빼기: P25 no_batter·no_batter_cluster·no_history
  - 정규화: P25 dropout·weight decay, P29 구조×dropout
  - 구조: P24·P27 periodic·residual·workload, P30 periodic×workload×head dropout, P31 TabM, P32 임베딩+잔차
  - 저장된 확률 섞기: P27·P31·P32 saved ensemble
- 가중치 평균은 P0~P32 어디에도 없다. `matrix_offline_rl.py`의 polyak은 RL target network용이다.
- 남은 목록의 다른 후보(투수 ID 임베딩, 좌우 대칭 증강, P13, F4)는 새 export가 필요하다. 녹화 중에는 맥에서 무거운 작업을 할 수 없다.
- 1차 문헌(초록 확인, 2026-10-04):
  - Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018, arXiv:1803.05407 (SWA, 균등 평균; 우리는 EMA라 같은 방법이 아니다)
  - Morales-Brotons, Vogels, Hendrikx, "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits", TMLR 2024, arXiv:2411.18704. 일반화와 보정(calibration)이 좋아졌다고 보고한다.
- 우리 구현은 응용이다. 논문 설정을 재현했다고 주장하지 않는다. 논문은 이미지 분류가 중심이고 우리는 10-class 표 자료다.

## 기준과 팔 (2×2, 한 번에 하나씩 바꿈)
- 기준 `control`: MatrixModel flatten_mlp 128. 같은 장치에서 새로 학습한다.
- `ema`: control과 같은 학습 + EMA decay .999
- `dropout`: frontier `dropout` 팔 그대로(p .1, wd .01)
- `dropout_ema`: dropout + EMA decay .999
- 고정 대비(후보 − 기준):
  - `ema_vs_control`
  - `dropout_ema_vs_dropout`
  - `dropout_vs_control`(재현 확인용)
- EMA 규칙:
  - 매 optimizer step 뒤 `ema ← d·ema + (1−d)·w`, `d_t = min(.999, (1+t)/(10+t))`
  - early stop 점수, best 상태, 저장 모델은 모두 EMA 가중치다.
  - 원래 가중치의 early stop NLL도 epoch마다 기록한다(진단용).
- decay를 .999로 정한 근거: P32 control seed 16 `fit.json`이 11 epoch에 13,464 update, epoch당 약 1,224 update다. .999는 warm-up이 끝난 뒤(step 8,990 ≈ 7.3 epoch) 약 0.8 epoch를 평균한다. 그 전에는 평균 지연이 epoch 1에서 0.1, epoch 6에서 0.6 epoch다. decay 탐색은 하지 않는다.
- EMA는 난수를 쓰지 않는다. 그래서 같은 seed에서 원래 가중치의 학습 경로는 기준 팔과 같다(테스트로 확인). 차이는 평가·멈춤·저장하는 가중치뿐이다. 그래서 멈추는 epoch와 update 수가 기준 팔과 달라질 수 있다(EMA는 보통 늦게 멈춘다). seed별 Δbest_epoch·Δoptimizer_updates를 같이 적는다.
- 파라미터 수는 기준과 같다(141,834).

## 고정 실행 조건
- seed: 19, 20, 21, 22, 23. 새 모델 난수일 뿐 새 평가 자료가 아니다.
- fit: 4 팔 × 5 seed = 20
- 예산: 30 epoch, patience 5, lr 5e-4, batch 1024, AdamW wd .01, uniform loss. 보정은 외부(JointDelivery)
- 자료: 기존 immutable export `49097109…f578`. 새 export 없음
- 장치: newsrv GPU3/GPU4(UUID 고정), 동시 2 fit, 공통 lock
- 예상 비용(추정, 미측정): control 평균 82.5 s(P32). EMA는 epoch마다 평가를 한 번 더 하므로 조금 더 든다. 합계 약 2,000 worker-s, 2장으로 약 20분. EMA 팔이 더 오래 돌 수 있어 최악은 30 epoch × 약 7.5 s × 1.1 ≈ 250 s/fit, 20 fit 약 5,000 worker-s. family 예산 36,000 worker-s, task 3,600 s
- 멈춤 조건:
  - task당 최대 2회 시도
  - 한 task라도 끝내 실패하면 family 미완성으로 기록하고 비교하지 않는다.
  - 중간 결과를 보고 팔·seed를 바꾸지 않는다.

## 보고 (기술값만)
- 대상: dev(Cpanel 12,334구), mlb_dev(311,721구). 둘 다 P24~P32에서 여러 번 본 DEV다. 독립 확인이 아니다.
- 지표:
  - 보정 NLL, Brier, top-label ECE
  - 같은 seed끼리의 Δ, seed 방향 수
  - 경기 단위 부트스트랩 95% 구간
- 팔별 실행 기록: best epoch, 실행 epoch, 학습 시간, 최대 GPU 메모리, delivery temperature
- early stop 자료에서 원래 가중치와 EMA 가중치의 NLL 차이는 진단으로만 적는다. 평가 결과가 아니다.
  - `raw_rule_best_epoch`: 같은 멈춤 규칙을 원래 가중치 점수에 다시 적용한 best epoch(EMA 실행이 먼저 멈추면 잘린다). 그 epoch의 EMA early stop NLL과 함께 진단으로만 적는다.
  - EMA 팔의 `history[*].earlystop_conditional_nll`은 EMA 점수다. 팔끼리 비교할 때는 `raw_earlystop_conditional_nll`과 기준 팔의 `earlystop_conditional_nll`을 쓴다.
- 예측 NLL이 좋아져도 추천 정책이 좋아졌다는 뜻이 아니다.

## 다음 단계 (이 문서 뒤)
1. 다른 Opus 작업자의 검토(같은 모델 검토, 대상 `f7282bb`) → 처리 기록
2. runner·scorer에 EMA 팔 연결과 DRAFT config `configs/DRAFT-CUDA-EMA-20261004.json`(실행 꺼짐), 연결 테스트 `tests/test_cuda_ema.py` — 다음 커밋에서 추가함. 이 연결 부분은 1번 검토 범위 밖이므로 따로 검토가 필요하다.
3. 20 task 전부 CUDA smoke
4. 모두 통과하면 등록 config(새 ID)로 전체 실행

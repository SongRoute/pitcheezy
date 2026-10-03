# 예측 연구 전선 검토 — 다음 학습 실험 후보 (2026-10-03)

작성: Claude Opus 5.5 (Codex PM 위임, 단일 작성자, 같은 모델 검토 없음). 기준 커밋 `0abda69` (`codex/research-roadmap-1006`).
범위: 문서·메타데이터만. 새 학습·추론·채점 0, 2026 payload 열람 0, 결과 수치 새 측정 0. 아래 수치는 모두 이미 커밋된 기록에서 옮겼고 출처를 붙였다.
지위: **제안·사전설계 초안**이다. 등록(config 커밋)이 아니며 실행 허가를 뜻하지 않는다. 기존 판정(G15 mix 연구 기준 D145, 서비스·정책 G0, D131, D155 닫힘)과 2026 사용 금지·봉인 파일은 바꾸지 않는다.

## 0. 요약

- ≤2025에는 **검증된 미노출 확인 모집단이 0개**다(기존 감사 `results/ML-unopened-data-audit-2026-09-27.json`; 이후 June 전체도 P12에서 사용). 그래서 아래 실험은 전부 **노출된 2025 DEV 위의 개발 비교**이고 독립 확인이 아니다.
- 지금까지 시험한 축(구조 7종, 공유 단위, 자료량, 타자 표현, 레퍼토리, delivery pool, 직접 head, 보정, seed 수)과 겹치지 않는 축으로 **5개 후보**를 고른다. 모두 G15 mix의 *직접 head 구성요소 하나*만 바꾸고 G0 구성요소·혼합 비중 1/2·보정 절차는 그대로 둔다.
- 다른 Opus의 혼합 비중·temperature 격자와 겹치지 않게 **혼합 비중·temperature·보정 단계는 바꾸지 않는다**.
- 1단계는 맥미니 CPU/MPS로 가능하다(직접 head fit 실측 규모 기반 추정 약 10–20분/실험). **현재 코드에 CUDA 경로가 없다**(`device='auto'`가 mps 또는 cpu만 고름) — newsrv로 옮기면 그대로는 CPU로 돈다. 상세는 [GPU 사전점검](../handoffs/PM-newsrv-preflight-2026-10-03.md).

## 1. ≤2025 시간 자료 노출 감사 (payload 미열람)

근거: `experiments/pitchmdp/pitchmdp/data.py:69` `add_splits`, `configs/ML-JUNE-ELIGIBILITY-v1.json`, `configs/EXP-P15-001.yaml`·`EXP-P16-001.yaml`의 `splits`/`data_rules`, `results/ML-unopened-data-audit-2026-09-27.json`(2026-09-27, head `cf20fb4`), decisions D81·D86·D131–D155. 원천 parquet은 **스키마 이름만** 읽었다(§1.3).

### 1.1 분할 규칙 (코드)

`add_splits`: `game_date < 2023-05-15` → history, `≤ 2025-04-30` → train, `≤ 2025-06-30` → calibration, 그 밖 → dev. 원자료 허용 목록은 `statcast_2023/2024/2025.parquet`뿐(`experiments/pitchmdp/configs/local.json` `raw_allowlist`). 같은 raw 폴더의 `holdout_2026/`은 허용 목록 밖이다.

### 1.2 모집단별 노출 상태

| 모집단 | 날짜 | 역할(문서 근거) | 상태 |
|---|---|---|---|
| history | 2023 시즌 시작 ~ 2023-05-14 | 전날까지 타자 이력 맥락 | 행 단위 사용 미열거 → 미노출 아님 |
| TRAIN | 2023-05-15 ~ 2025-04-30 (적격 1,252,824행, D100) | fit·정규화·빈도·delivery 표본 | 사용됨 |
| 2024 May/June/DEV | 2024-05-01 ~ 2024-09-30 | 과거 temporal 실험의 earlystop·temperature·blend·DEV, 이후 2025 TRAIN | 사용됨 |
| May 2025 | 2025-05-01 ~ 05-15 / 05-16 ~ 05-31 | early stopping / Cpanel temperature | 사용됨(나머지 일부 미확인) |
| June 2025 Cpanel | 2025-06-01 ~ 06-30 (4,821구) | 빈도 blend 비중, I1 fit | 사용됨 |
| June 2025 전체 | 2025-06-01 ~ 06-30 (적격 104,970구/394경기, D81) | P12 B1/B2 blend fit(D86), P17/P18 정책 결정 150경기 | **사용됨** (2026-09-27 감사 당시 "미확인"이었으나 이후 사용) |
| DEV 2025 | 2025-07-01 ~ 2025-09-30 (적격 311,721구/1,161경기; Cpanel 12,334구/328경기) | P7-003·P11·P13·P14·P15·P16 채점, 다수 진단 | 사용됨(반복 노출) |
| DEV 제외분 | 같은 기간 요청 341,941 − 적격 311,721 = 30,220구 | 적격성 처리 | 목표 사용 미확인, 독립 예비 아님 |
| 2025-10 이후·포스트시즌 | — | 승인 cache는 정규시즌만 | 가용성·노출 미확인 |
| 2023 이전 시즌 | — | cache 없음, 검색·인증 안 됨 | 가용성 미확인 |

**결론: 아직 안 본 ≤2025 확인 모집단은 0개로 인정한다.** "파일 이름이 다르다"는 이유로 미노출로 보지 않는다(감사 `definitions.unknown`). 이전 탐색 채점: Cpanel DEV는 P13·P14·P15(3 seed)·P16(5 seed)에서 같은 연구 계열이 반복 채점했고, 전체 MLB DEV는 P7-003 이후 P11·P13·P14·P15·P16에서 채점했다. 따라서 새 후보의 DEV 결과는 **개발 비교**이며 여러 후보 중 고르면 **탐색 후 선택**이 된다.

독립 확인 선택지(실행 아님, Song 결정 사항): (a) 2026 — 학습·튜닝·선택 금지라 예측 모델 선택 확인으로 쓸 수 없음. (b) 2022 이전 Statcast를 새로 받아 시간 역방향 확인 — 자료 범위 변경이고, 2023 규칙 변경(피치 클록·시프트 제한)으로 분포가 달라 확인력이 약하다. (c) 확인 없이 "노출 DEV 개발 결과"로만 보고. **추천: (c)로 진행하고 (b)는 후보가 5 seed 단계까지 살아남을 때 다시 묻는다.**

### 1.3 스키마만 읽은 사실

`statcast_2025.parquet` 스키마(pyarrow `read_schema`, 값 미열람)에 `n_thruorder_pitcher`, `n_priorpa_thisgame_player_at_bat`, `pitcher_days_since_prev_game`, `batter_days_since_prev_game`, `arm_angle`, `age_pit`, `age_bat`, `fielder_2`, `umpire`, `bat_speed`, `swing_length` 열이 있다. `experiments/pitchmdp/pitchmdp/*.py`에서 `n_thruorder_pitcher`·`pitcher_days_since_prev_game`·`arm_angle`·`bat_speed` grep 결과 0건(나머지 열은 검색하지 않음). 2023·2024 파일의 열 유무와 결측률은 **미확인**이다. `bat_speed`·`swing_length`는 스윙 뒤 측정값이라 입력으로 쓰면 누수다.

## 2. 지금까지 시험한 축 (P2–P18)

| 축 | 실험 | 결과(출처 docs/experiments.md·decisions) |
|---|---|---|
| 결과 모델 자료량 D25→D100 | P2-001, P6-001 | D100 N 통과, auxiliary 자료량도 N 통과 |
| 구조 7종(MLP·Melville·LightGBM·LSTM·GRU·Transformer·선형), G0 적분 틀 안 | P3-001 | N 통과 0, MLP 유지 |
| 공유 단위(개인·군집·부분 공유) | P4-001 | N/G 통과 0 |
| 이전 타석 이력 32/128구 | P4-002 | 비용 초과로 미실행 |
| 구조×자료량 상호작용 | P5-001 | 미확정 |
| 새 선수·입력 stress·전체 MLB 이전 | P7-001~003 | 진단, 승격 없음 |
| 정책(BC·계획·KL·IQL·CQL) | P8-001/002 | 모델 내부 결과, 이번 범위 밖 |
| 타자 표현 F1 마스킹 | P9-001/002 | full 유지 근거(개발) |
| seed 수 1/3/5 | P10-001 | 기술 통계 |
| 클래스 편향·volume 보정 | P11-002, P12-001/002 | 최소 폭 미달 |
| 투수 레퍼토리 특징 | P13-001 | 미확정 |
| delivery pool tier | P14-001 | 혼합 후 미확정 |
| 직접 pre-pitch head, 고정 1/2 혼합 | P15-001/002, P16-001 | 직접 head 단독 미확정, **혼합 N 통과(5 seed) → G15 mix 연구 기준** |
| 정책 예측기 교체·탐색 잡음 | P17-001, P18-001 | WITHIN_SEARCH_NOISE, 정책 재검증 닫힘(D155) |

매트릭스에서 **한 번도 실행하지 않은 행**: MX-D3/D4(최근성), D5, G5–G7, F2, F5, **F6(workload)**, F7, F9, L1–L4, C1/C2. 이번 후보는 이 미실행 행과, 직접 head 위의 새 backbone·입력 표현에서 고른다. 다음 축은 **겹치므로 제외**한다: 구조 7종 재실행(G0 적분 틀 안), 혼합 비중·temperature·June blend 격자(다른 Opus 담당), seed 수, 보정 방식, delivery pool, 레퍼토리 특징, 타자 표현 마스킹.

## 3. 문헌 (관련성은 추론이며 효과는 우리 자료에서 미측정)

URL은 이 세션에서 가져오지 않았다(WebFetch 미사용). 제목·arXiv 번호는 기억에 근거하므로 등록 전 확인이 필요하다.

- Guo et al. 2017, *On Calibration of Modern Neural Networks*, https://arxiv.org/abs/1706.04599 — scalar temperature. **이미 파이프라인에 있음**(May Cpanel per-seed temperature). 새 후보로 쓰지 않는다.
- Lakshminarayanan et al. 2017, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, https://arxiv.org/abs/1612.01474 — seed 앙상블. **이미 있음**(3/5 seed 확률 평균, P10). G15 mix 자체가 이질 앙상블이다. 새 구성원 추가는 혼합 비중 문제와 얽혀 이번에는 제외.
- Gorishniy et al. 2021, *Revisiting Deep Learning Models for Tabular Data*, https://arxiv.org/abs/2106.11959, 코드 https://github.com/yandex-research/rtdl-revisiting-models — ResNet-MLP·FT-Transformer. 직접 head는 평탄화한 표 입력의 MLP(`flatten_mlp`, width 128)라 backbone 교체가 한 가지 변경으로 성립한다. P3의 Transformer는 H5 토큰 위 sequence 모델이었고 G0 적분 틀 안이었으므로 FT-Transformer(특징 토큰)와 같은 실험이 아니다. 다만 "Transformer 계열이 MLP를 못 이겼다"는 P3 근거가 있어 우선순위를 낮춘다.
- Gorishniy et al. 2022, *On Embeddings for Numerical Features in Tabular Deep Learning*, https://arxiv.org/abs/2203.05556 — 수치 입력의 periodic(PLR)·piecewise-linear 임베딩. backbone은 그대로 두고 입력 표현만 바꾸는 변경이다.
- Gorishniy et al. 2024, *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, https://arxiv.org/abs/2410.24210 — 한 모델 안 다중 head 앙상블. seed 앙상블과 겹치는 부분이 있어 GPU 단계 선택 후보로만 둔다.

## 4. 공통 설계 (5개 후보 모두)

- **기준(baseline):** G15 mix = 직접 head(EXP-P15-001/P16-001 seed별 봉인 체크포인트) + G0-global 같은 seed member 확률 1/2, 그 뒤 May temperature·June 빈도 blend는 기존 scorer 그대로 (`configs/G15-MIX-RESEARCH-FROZEN-v1.json`). 탐색 단계 seed {0,1,2}는 봉인 예측을 재사용하므로 기준 쪽 새 fit 0.
- **후보:** 직접 head 하나만 바꾼 새 fit(seed {0,1,2}) + 같은 G0 member, 비중 1/2 고정, 같은 보정 절차. 혼합 비중·temperature 격자는 건드리지 않는다.
- **분할:** EXP-P4-001 키 그대로 — TRAIN fit, May 1–15 early stopping, May 16–31 Cpanel temperature, June Cpanel blend, DEV(Cpanel 주, 전체 MLB 보조), frozen `mlb_dev`. `data_rules` 문구도 P15/P16과 같게: "TRAIN rows only for fitting; May/June Cpanel only for temperature and blend; DEV never used for fitting, calibration or selection." 2026 행 0.
- **의존 소스:** `experiments/pitchmdp/scripts/run_ml_direct_head.py`(P15 fit/predict), `run_ml_direct_head_confirm.py`(P16, 봉인 소스 해시 검사), `score_ml_matrix.py`, parent run `ML-MATRIX-20260924/EXP-P4-001`(preparation sha256 `d544bf38…`, T7 785MB). 각 후보는 **새 runner 또는 새 옵션 + 새 config**로 만들고 봉인 runner를 수정하지 않는다(P16의 소스 해시 거부 규칙 유지).
- **하이퍼파라미터:** 결과 열람 전 config에 고정, `hpo_trials 0`. 필요한 경우 탐색은 May 1–15 early-stop 손실로만 하고 DEV를 보지 않는다.
- **비용(추정, 측정 아님):** 직접 head G0 규모 fit 83–101초/seed(EXP-P4-001 fit.json, P15 config 인용), P15 smoke+3seed 약 10분(D139: 04:54–05:04), P16 2 fit 약 7분(D143: 12:03–12:10), 최대 RSS 7.6–8.6GB. 따라서 MLP 규모 후보는 맥미니에서 실험당 약 10–20분, ResNet은 1–2배, FT-Transformer는 **미측정**(특징 수 제곱 attention, smoke로 먼저 측정).

### 4.1 판정 게이트 (실행 전 고정)

- **실험별 개발 판정 (G-dev):** 주 비교 = Cpanel DEV(June blend 후) 후보 mix − G15 mix, 경기 부트스트랩 짝 ΔNLL. 통과 = ΔNLL ≤ −0.003 AND 95% CI 상한 < 0 AND ΔBrier CI 상한 ≤ +0.001 AND seed 2/3 이상 음수 (기존 N 규칙 그대로). 보조: 전체 MLB DEV 짝 ΔNLL(보정 그대로), ECE10. 미달은 "미확정"이지 "효과 없음"이 아니다.
- **탐색 후 선택 (G-sel, G-dev와 분리):** 1단계에서 k개 후보를 돌리면 Holm(k)로 p를 보정하고, 통과 후보가 여럿이면 사전 순서(E1 > E2 > E3 > E4 > E5)로 하나만 올린다. 선택된 후보는 "노출 DEV에서 k개 중 탐색 후 선택"으로 표시한다.
- **확인 단계 (G-conf):** 선택 후보만 seed 3·4 새 fit을 더해 5 seed 대 G15 mix 5 seed(봉인)로 같은 규칙을 다시 적용. 같은 DEV라 독립 확인이 아니다(P16과 같은 지위). 연구 기준 교체 여부는 Song 결정(D158 §3).
- **정책·서비스:** 이 단계에서 정책 재검증을 열지 않는다(D155 닫힘 유지).

## 5. 후보 실험 5개 (한 가지 변경씩)

ID는 PM이 배정한다(가칭 E1–E5). 모두 seed {0,1,2}, 분할·게이트는 §4.

**E1. 직접 head backbone: MLP → ResNet-MLP** (Gorishniy 2021)
- 변경: `flatten_mlp`(width 128)를 같은 width의 residual block 2개(BatchNorm 또는 LayerNorm, dropout 고정)로. 입력·출력·optimizer·epochs 30·patience 5·batch 1024·lr 5e-4 그대로.
- 근거: 같은 입력에서 MLP보다 깊은 표 모델의 이득 여부. P3는 G0 적분 틀 안에서 비교했으므로 직접 head에서는 미시험.
- 위험: 파라미터 수가 늘어 용량 효과와 구조 효과가 섞임 → config에 파라미터 수를 기록하고 같은 파라미터 수의 넓은 MLP 대조는 통과 시 후속으로.
- 비용: 맥미니 약 15–30분(추정).

**E2. 직접 head 수치 입력 임베딩: 원값 → periodic(PLR) 임베딩** (Gorishniy 2022)
- 변경: 연속 입력 채널(과거 공 물리·맥락 수치)에 PLR 임베딩만 추가, backbone은 MLP 그대로.
- 근거: 위치·구속 같은 수치에서 비선형 경계를 MLP가 학습하기 어렵다는 가설. 입력 표현 축으로 미시험.
- 위험: 임베딩 주파수 σ가 하이퍼파라미터 → 고정값(논문 기본 범위 안 하나)을 등록, DEV 미사용.
- 비용: 약 15–30분(추정).

**E3. 학습 손실 최근성 가중 (MX-D4, 미실행 행)**
- 변경: TRAIN 손실에 날짜 기준 지수 가중(half-life 365일 하나 고정, 2025-04-30 기준). 표본 목록·모델·보정 그대로.
- 근거: 2023 규칙 변경 이후 리그 분포 이동. 자료 축이라 구조 후보들과 독립.
- 위험: 가중 ESS 감소 → 가중 ESS를 결과 열람 전 산출·보고. half-life를 여러 개 돌리면 탐색이므로 하나만.
- 비용: MLP와 같음, 약 10–20분(추정).

**E4. workload 입력 블록 추가 (MX-F6, 미실행 행)**
- 변경: `n_thruorder_pitcher`, `pitcher_days_since_prev_game`, 경기 내 누적 투구 수(기존 키로 계산) 3채널만 맥락에 추가.
- 선행조건: 2023·2024 열 존재·결측률을 **TRAIN 행에서만** 확인(현재 미확인), 투구 전 시점 값인지 문서 확인. 불충족 시 E4는 실행하지 않는다.
- 근거: 피로·타순 회차는 결과 분포를 바꿀 수 있는 투구 전 정보이며 지금 입력에 없다. "실제 피로 측정"이라고 부르지 않는다.
- 비용: 자료 준비 별도(미측정) + fit 약 10–20분(추정). parent preparation을 바꾸므로 새 prepare 해시를 등록.

**E5. 직접 head backbone: MLP → FT-Transformer** (Gorishniy 2021) — GPU 단계 후보
- 변경: E1과 같은 자리에서 backbone만 FT-Transformer(블록 3, d 192 등 논문 기본값 하나 고정).
- 근거: 특징 간 상호작용을 attention으로. P3 Transformer(H5 sequence)와는 다른 실험이나 같은 계열 실패 근거가 있으므로 E1이 미확정 이상일 때만 실행하는 조건부 후보.
- 비용: **미측정.** 맥미니 smoke 1 epoch로 측정 후 2시간 초과 예상이면 newsrv 단계로.

제외·후속: TabM(앙상블 축 겹침), 2022 역방향 확인(자료 범위 결정 필요), F7(포수·심판 — `umpire` 열 값 품질 미확인), 이전 타석 이력 F4(P4-002 비용 초과 판정 존중).

## 6. 단계 계획

1. **S0 준비(맥미니, 가벼움):** 후보별 runner 옵션 + config 초안 + 합성 테스트(누수 가드: 현재 공 물리 0 유지, E4 입력이 투구 전 값인지). E4 열 결측률은 TRAIN만 읽는 별도 메타 점검으로.
2. **S1 smoke(맥미니):** seed 0, `--smoke`로 실행 경로·시간·RSS 측정. 이 값으로 S2 장소를 정한다(로컬 2시간 기준, CLAUDE.md D10).
3. **S2 탐색(맥미니 또는 newsrv):** E1–E4 seed {0,1,2}, 무거운 실행은 한 번에 하나(PM 규칙), 경기 기록 시간대 회피(D159, 10/4 01:40 KST까지 종료).
4. **S3 채점:** 4개 결과를 같은 scorer로 한 번에 채점, Holm(k=실행 수), G-sel 적용.
5. **S4 확인:** 선택 1개만 seed 3·4 새 fit → 5 seed G-conf. E5는 E1 결과에 따라 S2에 추가 또는 생략.

장치 주의: G15 mix의 봉인 직접 head는 맥미니(MPS 또는 CPU)에서 학습됐다. newsrv에서 CUDA로 후보만 학습하면 **장치 차이가 변경점에 섞인다.** GPU로 옮길 경우 같은 장치에서 기준 직접 head도 다시 학습한 대조 arm을 함께 두거나, 탐색은 모두 맥미니에서 한다. 추천: **S2까지 맥미니, newsrv는 E5와 후속 확장만**.

## 7. 한계

- 노출 감사는 문서·config·스키마 근거이며 파일 접근 전수 감사가 아니다. 2026-09-27 감사 이후 사용 이력은 decisions에서 옮겼다.
- 비용은 과거 직접 head 실행 기록에서 추정했고 새 backbone 비용은 측정하지 않았다.
- 문헌의 효과 크기는 우리 자료에 옮겨 오지 않았다. 후보 선택은 미시험 축이라는 이유이지 개선 예측이 아니다.
- 이 문서는 같은 모델 검토도 받지 않았다. 등록 전 다른 모델 검토를 권한다(PM-charter §4).

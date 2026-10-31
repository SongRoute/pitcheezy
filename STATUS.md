# Pitcheezy 프로토타입 현황 (STATUS)

- 작성일: 2026-10-07
- 읽는 사람: 이 레포를 처음 보는 통합 상대 팀원
- 작성 방식: 코드를 읽고, 테스트를 실행하고, 이미 떠 있는 로컬 서버에 GET 요청을 보내 확인했다. 코드에서 확인하지 못한 내용은 **(추정)**으로 표시했다. 실행하지 못한 항목은 **(미실행)**으로 표시했다.
- 수치 원칙: 측정 파일이 없는 수치는 적지 않았다. 없는 값은 "미측정"으로 썼다.

## 0. 먼저 알아야 할 것: 이 레포에는 "버전"이 여럿 있다

브랜치가 365개, 등록된 worktree가 296개다. 서비스 코드는 브랜치마다 다르다. 이 문서는 아래 두 커밋을 기준으로 쓴다.

| 이름 | 커밋 / 브랜치 | 내용 | 현재 실행 |
|---|---|---|---|
| **R (연구·구버전 앱)** | `f7e8f47` / `codex/ml-matrix-execution` (현재 체크아웃) | 데이터·연구 코드 전체 + 구버전 관전 앱(기록 재생) | `127.0.0.1:8766` (9/23 기동) |
| **S (최신 서비스)** | `5488c93` / `codex/demo-pa-anchors-1006` | R의 연구 코드 + 실시간 피드·새 관전 화면 | Tailscale `100.108.252.111:8771` (10/6 기동) |

- 중간 버전 `demo/ws-2026`(`09e12e9`)은 S의 조상이다. Tailscale 8766에서 별도로 돌고 있다.
- S는 R보다 `apps`·`src`·`scripts` 기준으로 118개 파일, +19,291줄 많다.
- 아직 어느 버전도 `main`에 합쳐지지 않았다.
  - GitHub 공개 `main`은 `70aa549`(9/23)이다.
  - 로컬 커밋 다수는 push되지 않았다. (근거: `docs/pm/operations-inventory.md`)
- 각 절에는 어느 버전 이야기인지 [R] 또는 [S]로 표시했다.

---

## 1. 서비스 정의

MLB 경기를 보는 **초보·중급 야구팬용 보조 화면**이다. 투구 전에는 모델 기준 "다음 공 구종·목표 위치 후보"를 보여 준다. 투구 후에는 실제 투구와 비교해 보여 준다. 경기 영상이 주 화면이고 이 앱은 옆에 띄워 두는 용도다(`docs/product/experience.md`).

모델의 추천이 실제로 승률을 높인다는 근거는 아직 없다(7·8절). 화면에도 "검증 전 실험", "정책 선택 비중이며 성공 확률이 아님"이라고 표시한다.

### [S] 실제로 서빙되는 화면 (8771, 번들 JS를 직접 확인)

- **경기 목록 `/watch`** (`service/WatchList.tsx`)
  - 답하는 질문: "그날 어떤 경기가 있고, 지금 어떤 상태인가(LIVE·종료·예정·연기)?"
- **경기 화면 `/watch/{gamePk}`** (`service/GameScreen.tsx`)
  - **다음 공 추천 1~3순위**(구종 + 비중 % + 목표 구역). 답하는 질문: "모델 정책이라면 다음 공으로 어떤 구종을 어디에 던지나?"
  - **실제 투구 비교**(구종·구속·도착 위치·판정·플레이 문구). 답하는 질문: "추천과 실제 투구가 어떻게 달랐나?"
  - **존 그림**(포수 시점): 추천 목표점, 실제 도착점, 포수 셋업 가로 위치(CV 추정)를 함께 보여 준다. 답하는 질문: "어디를 노렸고(추정) 어디로 갔나?"
  - **상황 머리**: 점수·이닝·B-S-O·주자. **타자/투수 카드**: 등번호, 경기 전 ERA·AVG·OPS.
  - **타석 투구 띠·타석 선택 시트·박스스코어·다음 두 타자**. 답하는 질문: "이 타석과 경기가 지금까지 어떻게 흘러왔나?"
  - **추천 출처 배지**. 답하는 질문: "이 추천은 언제 계산된 것인가?"
    - `captured_live`: 경기 중 투구 전에 계산했다.
    - `asof_replay`: 경기 뒤에 그 공 직전 정보만으로 계산했다.
  - **미지원 표시**. 답하는 질문: "왜 추천이 없나?" 예: "2025년까지 학습 기록이 없는 투수라 추천하지 않습니다."
  - **'기준과 한계' 시트**. 답하는 질문: "이 숫자를 어디까지 믿어도 되나?"
- **시연 운영자 `/demo-operator`** (`demo-operator/DemoOperator.tsx`)
  - 저장된 경기를 공 단위 시계로 재생, 일시정지, 이동, 배속 조절한다. 토큰이 필요하다.
  - 답하는 질문: "시연에서 어디까지 공개할까?"
- `/showcase`도 HTTP 200을 반환하지만, 실제로는 경기 목록이 뜬다. Showcase·WatchAlong·Live·Game 등 옛 화면은 `src/`에만 있고 번들에서 빠져 있다(`main.tsx:18-21` 주석과 번들 문자열 검사로 확인).

### [R] 구버전 관전 앱 (8766 loopback, 기록 재생 전용)

- **타석 관전 `/`**
  - 2025-08~09의 12경기, 249타석 중 하나를 골라 한 공씩 공개한다.
  - 공마다 사전 추천(구종×9구역, 수비팀 승률 %, 기준 대비 %p)과 실제 투구를 비교한다.
  - 답하는 질문: "그때 모델은 무엇을 추천했고, 실제로는 무엇이 왔나?"
- **사건 분석 카드**: 타석 종료 시 수비 승률 변화(total_pp)를 전략·실행·잔차로 나눈다. 의도 데이터가 없으면 전부 "미배분 잔차"로 표시한다.
  - 답하는 질문: "이 타석 결과로 승률이 얼마나 움직였나?"
- **수동 의도 메모**: 사용자가 마지막 공의 의도 구역을 직접 입력한다.
- **교체 직전 기록 `/inning-decisions`**: 투수를 유지할 때 반이닝 승리확률의 하한~상한을 보여 준다. 교체 효과는 구조적으로 "미측정"이다. 데이터는 1경기(777063) 1건뿐이다.
  - 답하는 질문: "이 투수를 그대로 두면 이번 이닝은 어떻게 될까?"
- **영상 검토실 `/video-lab`**: 로컬 클립에 ROI·보정점을 라벨링하고 추적한다. 현재 서버에는 영상이 없어 빈 화면이다.

---

## 2. 전체 구조 (데이터 → 모델 → 서빙 → 화면)

```
① Statcast 수집 ──► ② 전처리·특징 ──► ③ 학습·동결 ──► ④ 추론(사전/실시간) ──► ⑤ API ──► ⑥ React 화면
 (Baseball Savant)   (parquet)         (T7 SSD 번들)     (T7 JSON 캐시)         (FastAPI)
                                                             ▲
                                     MLB StatsAPI 라이브 피드 ┘ [S만]
```

1. **수집**: `scripts/fetch_data.py` → `src/pitcheezy/data/statcast_fetch.py`
   - pybaseball로 2023~2025 정규시즌을 받는다. 2026은 홀드아웃으로 따로 받는다.
   - 결과: `data/raw/<버전>/statcast_<시즌>.parquet`. 이 경로는 Google Drive로 가는 심볼릭 링크다.
2. **전처리·특징**: 연구 트리가 두 갈래다.
   - (구) `src/pitcheezy/data/*`, `src/pitcheezy/interfaces/*`: P0/P1 1세대. 9/20 이후 변경이 없고, 서비스에서 import하지 않는다.
   - (현) `experiments/pitchmdp/pitchmdp/*.py`: 현재 연구 정본. `sequence_data.py`, `archetypes.py`, `matrix_*`, `policy_*` 등 파이썬 약 56,700줄.
3. **학습·동결**: `experiments/pitchmdp/scripts/*`, `configs/*.json|yaml`
   - 맥미니 MPS/CPU와 학교 A5000 서버(`gpu run`)에서 학습한다.
   - 동결 산출물은 외장 SSD `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/...`에만 있다.
4. **추론**
   - [R] `apps/observer/backend/observer_app/recommender.py`: 요청 시 계산하고 T7에 JSON으로 캐시한다. 모델 코드는 `apps/observer/runtime_src/`(연구 코드의 해시 검증 사본)다.
   - [S] 종료 경기: `scripts/demo_precompute.py`가 경기 전체를 미리 계산해 T7 `DEMO-WS-2026/watch/{pk}.json`에 쓴다.
   - [S] 진행 중 경기: `observer_app/live_feed.py`가 StatsAPI를 폴링하고, `live_armb.py`가 백그라운드 스레드에서 공마다 계산한다.
5. **API**: `apps/observer/backend/observer_app/main.py` (FastAPI)
   - [S] 새 API는 `service_api.py`의 `/api/service/*`다. 옛 `/api/live`, `/api/watch`, `/api/sessions`도 남아 있다.
6. **화면**: `apps/observer/web/src` (React + Vite)
   - [S] 화면은 `service/*`, 응답 정규화는 `service/api.ts`가 맡는다.
   - FastAPI가 빌드 결과 `web/dist`를 정적으로 서빙한다.

---

## 3. 계층별 상태

### 3.1 데이터 수집과 전처리 — **동작 확인됨 (수집 재실행은 미실행)**

- **실행 명령**
  - `python scripts/fetch_data.py --seasons 2023 2024 2025 --holdout 2026 --tag s2325` (미실행. 네트워크로 수 시간 걸리는 작업이라 이번에 돌리지 않았다)
  - 테스트: `PYTHONPATH=src .venv/bin/python -m pytest tests -q` → **365 passed, 0 skip (27.6초)**. 원본 데이터 없이 196행 fixture로 돈다.
- **실제 데이터** (`data/versions.md`, 로컬 존재 확인)
  - 2023: 720,684행 / 2024: 711,899행 / 2025: 712,528행 (119열)
  - 2026: 716,792행 (9/27 동결, 홀드아웃)
- **잘 된 점**
  - 월 단위로 이어받기를 지원한다.
  - dtype을 고정하고, sha256과 행 수를 버전 장부에 남긴다.
  - **2026 데이터 학습 금지를 코드로 강제한다.** `statcast_fetch.HOLDOUT_SEASONS`와 `experiment.py:36,45-53`이 2026 시즌을 넣으면 `ValueError`를 낸다.
- **전처리 산출물** (구 트랙)
  - 선발 풀: 100이닝 이상, 219명
  - 타자 군집: K=6, 좌우 3개씩
  - RE24 표, 투구 표
  - 현재 연구 트리(`experiments/pitchmdp`)는 같은 parquet을 자체 `RAW_COLUMNS`로 다시 읽는다. 전처리가 사실상 두 벌이다.
- **약점**
  - **확인된 버그**: `.gitignore` 4~6행의 `!data/re24/  # 주석`은 git이 같은 줄 주석을 지원하지 않아 동작하지 않는다(`git check-ignore`로 확인). 그래서 `data/re24`, `data/pitchers`, `data/batters`는 **한 번도 커밋된 적이 없다**. 그런데 `experiment.py:73,79,88`은 이 경로를 `REPO/data/...`로 고정해 둬서, 새로 클론하면 구 트랙 실험이 돌지 않는다.
  - 원본 데이터가 Google Drive 심볼릭 링크와 T7 SSD에 흩어져 있다.
  - `scripts/` 일부와 `configs/*.json`에 `/Volumes/T7 Shield`, `/Users/song` 절대경로가 다수 박혀 있다.
  - 중복 코드: `load_pitches` 3벌, 타석 경계 로직 3벌.

### 3.2 모델 (종류별)

> 핵심 구분: 이 프로젝트의 "예측 모델"은 **투수가 무엇을 던질지가 아니라, 주어진 구종을 던졌을 때 그 공의 결과(볼·스트라이크·파울·아웃·안타 등)의 확률**을 예측한다. "추천"은 이 결과 확률을 승률 모델에 넣고 계획(DP)해서 만든다. **예측 성능이 좋아진 것과 추천이 좋아진 것은 별개이고, 추천 쪽은 개선 근거가 없다.**

#### (M1) 투구 결과 예측기 — 서비스용 MVP `minimal-pitch-service-v1` [R 서빙] — **동작 확인됨(테스트·서버 health), 6명 한정**
- **예측 대상**: 다음 공의 결과 10클래스 확률(ball, strike, foul, out, single, double, triple, home_run, hbp, double_play).
- **방법**
  - 평탄 MLP(width 128), seed 42~46 5개 앙상블.
  - 입력: 상황 맥락 23차원(카운트·아웃·이닝·점수차·주자·좌우·타자 성향 6개와 신뢰도·군집 소속 5개)과 현재 공의 물리량.
  - 물리량은 TRAIN 실제 투구 400개 표본으로 적분한다. 투수/타자 ID는 입력하지 않는다.
  - 계층 빈도 모델과 고정 혼합한다(신경망 가중치 0.625).
  - 코드: `apps/observer/runtime_src/minimal_pitch_service.py:184-219`
- **데이터 분할**: TRAIN 2023-05-15~2025-04-30 / 2025-05 조기 종료·온도 보정 / 2025-06 혼합 비율 / DEV 2025-07 이후.
- **지원 투수**: 6명(Webb, Wheeler, Gilbert, Burnes, Nola, Berríos).
- **성능**: DEV 2025, 5,048구 / 60경기. 출처는 `experiments/pitchmdp/MINIMAL_SERVICE_RESULTS.md`이고, 원 결과 `evaluation.json`은 T7에만 있으며 레포에 커밋되지 않았다.
  - logloss: 1.4927 (상황 빈도 기준 1.5057). 차이의 경기 단위 bootstrap 95% CI [−0.0183, −0.0083]
  - Brier: 0.7220 (기준 0.7253)
  - accuracy 0.3875, ECE10 0.0096

#### (M2) 투구 결과 예측기 — 연구 기준 G0 / G15 mix [연구 전용, S의 ARM-B가 G0을 사용] — **동작 확인됨(결과 커밋)**
- **G0**
  - 평탄 MLP 입력: 같은 타석 이전 5구의 물리량·결과, 시점 이전 타자·경기 맥락, TRAIN에서 만든 투수 표현.
  - 5 seed, 6월 빈도 혼합. 설정: `configs/EXP-P4-001.yaml`, `configs/G0-RESEARCH-FROZEN-v1.json`
- **성능** (`results/EXP-P11-001.json`)
  - 측정: 2025 전체 MLB DEV, 311,721구 / 1,161경기
  - NLL: 1.4904 vs 빈도 1.5041, Δ −0.0137 [−0.0144, −0.0129]. 5 seed 중 5개 개선
  - Brier Δ −0.0029
  - ECE10: 0.0087로 빈도(0.0017)보다 **나쁘다**.
- **G15 mix** (`results/EXP-P16-001.json`)
  - 방법: 현재 공 물리량 없이 직접 예측하는 head와 G0를 반반 섞는다.
  - G0 대비 ΔNLL −0.0041 [−0.0044, −0.0038]
  - accuracy 0.376, macro-F1 0.107. 드문 클래스는 사실상 맞히지 못한다.
  - `research_only=true`이고 서비스에 쓰지 않는다.
- **공통 한계**
  - 모든 비교가 **2025 DEV를 여러 번 들여다본 개발 비교**다. 미열람 확증 세트가 없다.
  - 개선 폭은 NLL 약 0.9%로 작다.

#### (M3) 위치(9구역) 근사 — `observer-zone-v1` [R] / `observed-train-delivery-gaussian-v1` [S] — **일부 동작 (위치 자체를 예측하지 않음)**
- **방법**
  - 위치는 모델이 예측하지 않는다.
  - TRAIN 실제 투구 400개를 각 구역 중심 주변에서 Gaussian(σ=0.45ft)으로 가중해 "그 구역을 노렸을 때"의 결과 분포를 근사한다(`recommender.py:149-239`).
  - 지원 조건: 유효 표본 20개 이상, 커널 질량 0.01 이상.
- **한계**
  - 제구 오차나 의도에 대한 인과 모형이 아니다.
  - 존 밖 후보는 만들지 않는다.
  - [R]은 구종×9구역 조합 전체를 계획기에 넣고 승률이 가장 높은 조합을 고른다.
  - [S]의 정책 이름 자체가 "pitch type only"다. 정책은 구종만 고르고, 위치는 그 구종의 동결 TRAIN 투구 풀에서 현재 카운트 기준 **커널 질량이 가장 큰 구역**, 즉 "이 구종이 평소 가장 많이 가는 곳"을 붙인다(`5488c93:scripts/demo_precompute.py:353-374`). 최적 목표가 아니다. 지원 구역이 없으면 "위치 근사 불가"로 표시한다.
- **성능**: 위치 추천의 정확도·효과는 미측정이다.

#### (M4) 승리확률(WE) 모델 — **동작 확인됨**
- **방법**: 실제 경기 승패에 단조 제약 로지스틱 회귀를 적합한다. 입력은 점수차·이닝·초말·아웃·주자이고, 주자 진루는 경험 전이 확률로 처리한다(`experiments/pitchmdp/pitchmdp/game.py:124,216`).
- **성능**: 2025 DEV, 타석 시작 상태 87,559개 / 1,161경기. 출처는 T7 `evaluation.json`이다.
  - Brier 0.1540 (상수 기준 0.2495), logloss 0.4588, ECE10 0.0142
  - 같은 경기 라벨이 반복되므로 독립 표본은 아니다.
- 구 트랙은 RE24(득점 기대) 표를 쓴다(`src/pitcheezy/data/re24.py`). 예: 0아웃 무주자 0.509.

#### (M5) 타자 성향·군집 — **동작 확인됨, 효과 약함**
- **현 트랙**: 성향 6개에 가중 k-means 5중심, soft membership을 계산해 예측기 입력으로만 쓴다(`archetypes.py:81`).
  - 군집 추가 이득은 확인되지 않았다.
  - 타자 맥락 전체를 가리면 ΔNLL이 0.0052 나빠지므로(`EXP-P9-002`), 타자 맥락 자체는 유효하다.
- **구 트랙**: K=6 군집(`src/pitcheezy/data/batters.py`). P0-006 대비 효과가 없었다.

#### (M6) 추천 정책 / 계획기 — **동작하지만 개선 근거 없음**
- **[R] 계획기**
  - `planner.solve_pa`: 카운트별로 타석 종료까지 수비 WE를 최대화하는 정확한 DP다.
  - 기준 정책은 TRAIN 구종 빈도 × 구역 질량이다.
  - 내부 파일럿 +0.316%p는 같은 모델로 평가한 **순환 평가**다.
  - 분리 평가기로 150타석을 다시 재면 +0.004%p [−0.0008, +0.0087]로 **0을 포함한다**(`experiments/pitchmdp/SERVICE_FOLLOWUP_RESULTS.md`).
  - 후보 간 value 차이는 실제 응답에서 0.01%p 수준이다.
- **[S] ARM-B P3** (identity `a6dffaea…`, "pitch type only", ≤2025 동결)
  - G0 위에서 구종 정책을 낸다. 기준 정책은 지원 범위로 제한한 행동 모방(SupportedBC)이다.
  - 정책식: π ∝ π_ref·exp(Q/τ), τ=0.1. Q는 G0와 WE로 타석 끝까지 표본 계획한 수비 WE다(`experiments/pitchmdp/pitchmdp/policy_tau.py`, `docs/contracts/ML-POLICY-MATERIALIZATION-v1.md`).
  - τ는 6월 자료로 정했다(D107). 평균 KL이 0.0074라 **기준 정책(BC)과 거의 같은 분포**를 낸다.
  - 동결 사슬: `configs/ML-POLICY-MATERIALIZATION-v1.json`과 `addendum-1..6-d107`. G0 동결 검증 결과는 `self_contained=false`이고, 가중치는 T7에만 있다.
  - ≤2025 봉인 평가: 150경기 / 11,350타석. Δ 경계 [−0.00065, +0.00054] WE/PA로 0을 포함한다(`results/ML-POLICY-VAL-v1-S6-v3.json`).
  - **2026 OPE**: 2,429경기 / 184,288PA(그중 시작 상태가 지원되는 E0 149,525). L1-R Δ 경계 [−0.00120, +0.00092]. **판정 `NO_EVIDENCE_OF_IMPROVEMENT`**. 해로움도 확인되지 않았다(`results/MLB-2026-OPE-v1.json`, `docs/decisions.md` D131).
  - **식별자 불일치(확인 필요)**: 2026 OPE가 평가한 정책 식별자는 `c13cc989…`(D125 재인증)인데, 서빙 코드가 검사하는 식별자는 `a6dffaea…`(D107)다. S6-v3 파일의 식별자는 `6eb85ba7…`다. D122는 "코드 closure 변경에 따른 재인증이고 계산 의미는 같다"고 기록했지만, 둘이 같은 추천을 내는지는 이번에 직접 확인하지 않았다(추정).
  - 계획 seed만 바꿔도 1순위 구종이 11.3%의 결정에서 바뀐다(D155).
  - offline RL(IQL/CQL) 시도는 판정 불가였다.
- **구 트랙**: RE24 가치 반복(`src/pitcheezy/policy/vi.py`)과 OPE(IPS/SNIPS/DR, `src/pitcheezy/ope/*`). 2026 OPE 결과는 노출 이력이 있다(D87).

#### (M7) 빈도 기준선 — **동작 확인됨**
- 계층 빈도 모델(카운트 → 구종 → 투수 순 수축)이다. 비교 기준이자 M1 혼합의 한쪽이다.
- logloss: DEV 1.5057, 전체 MLB 1.5041.

#### (M8) CV 의도(포수 셋업) — **미완성 (외부 팀원 모듈, 이 레포에는 결과 파일만)**
- `src/pitcheezy/cv`, `decomp`는 빈 패키지다.
- [S]는 T7의 셋업 추정치(가로 위치만, 검토 전)를 `setup_estimates.py`로 서빙한다.
- M3 측정 사본(`web/src/cv-m3-evidence.json`): 픽셀 오차 중앙값 2.53, 피트 오차 중앙값 0.092. "물리 정확도·의도 검증 아님"이라고 명시돼 있다.

#### 모델 테스트
- `experiments/pitchmdp/tests`: **757 passed, 1 skipped, 90 subtests passed (89초)**
- 학습 재실행: **미실행**

#### 모델 코드 트리 세 벌
- `src/pitcheezy`: 구 트랙, 서비스 미사용
- `experiments/pitchmdp`: 현 정본
- `apps/observer/runtime_src`: 해시 검증 추출본, 현재 원본과 해시 일치
- `runtime_src`에는 `matrix_*`, `policy_*`가 없어서 [R] 앱은 G0/ARM-B를 실행할 수 없다. [S]는 연구 `.venv`와 T7을 직접 불러 ARM-B를 실행한다.

### 3.3 서빙 — [R] **동작 확인됨(기록 재생)** / [S] **일부 동작(종료 경기 확인, 실시간 경로는 이번 프로세스에서 0회 계산)**

- **공통**: FastAPI + uvicorn. 인증은 없다([S] 운영자 API만 토큰 사용). 모든 예외를 503 `{detail}`로 바꾼다.
- **[R] 실행**
  - `sh apps/observer/run.sh` (기본 127.0.0.1:8766)
  - 변형: `run_standalone.sh`(8768), `run_improved.sh`(8767)
  - `web/dist`가 없으면 시작을 거부한다.
- **[R] 방식**
  - `dataset.json`(약 5MB, 경기→타석→투구, sha256 검증)은 사전 생성이다.
  - **추천은 첫 요청 때 CPU에서 계산하고 T7에 캐시**한다(`RUN/recommendation_cache/<sha>.json`). 12개 카운트를 한 번에 계산한다.
  - 전역 RLock으로 직렬 처리해서 동시 사용자에 약하다. 캐시가 없을 때의 지연은 미측정이다.
- **[R] 저장**
  - SQLite WAL: sessions, recommendations(UPDATE 금지 트리거), jobs, manual_annotations, event_results, inning_decisions
  - 낙관적 동시성(revision 불일치 시 409)
  - `GET /api/sessions/{id}`가 DB 쓰기를 한다(부작용 있는 GET).
- **[R] 워커**: 0.5초 폴링 하위 프로세스지만, 영상 어댑터가 없어 항상 `unavailable:no_media`를 낸다. 실질 기능이 없다.
- **[S] 실행**: `apps/observer/serve.sh`
  - `PITCHEEZY_OBSERVER_HOST=tailscale`이면 Tailscale IP에 바인딩한다.
  - src가 dist보다 새로우면 `npm ci && npm run build`를 자동 실행한다.
  - launchd 예시 `deploy/*.plist.example`이 있지만 설치하지 않았다.
- **[S] 방식**
  - 종료 경기: 사전 계산 JSON(`asof_replay`)
  - 진행 중 경기: StatsAPI GUMBO를 폴링(기본 10초, 조건부 5초)하고, 백그라운드 스레드가 공마다 ARM-B를 계산한다(문서상 경기당 바인딩 5~8초, 상태당 약 0.15초. 이번에 재측정하지 않음).
  - API는 최대 1초 기다리고, 그 안에 결과가 없으면 `status:"missing"`을 반환한다.
  - 새 서비스 경로는 **지연 0인 최신 스냅샷**을 쓴다(`main.py:420`). 방송 지연 맞추기는 옛 `/api/live`에만 있다.
  - 8771 health는 `live_policy.computed:0`이다. 현재 프로세스(10/6 09:24 기동) 이후 실시간 추천이 한 번도 계산되지 않았다.
- **[S] 실경기 운영 기록** (`docs/pm/service/state.md`, 저장 로그 분석)
  - 849829에서 첫 추천 가능: 89/260
  - 30초 중계 지연 가정 시 5초 여유 확보율: 93.6%(목표 95% 미달)
  - 피드 지연: StatsAPI에 공이 나타나기까지 약 11초(849844 1건, `live_config.json`)
  - 이번 감사에서 T7 watch 19경기를 다시 계산한 추천 가능 비율은 경기별 0.24~1.0이다.
- **테스트**
  - [R] `PYTHONPATH=apps/observer/backend .venv-observer/bin/python -m pytest apps/observer/backend/tests -q` → **203 passed**. 단, T7 마운트가 필요하다. 4개 파일이 T7 파일을 skip 처리 없이 읽는다.
  - [R] 테스트는 `FakeRecommender`를 쓰므로 **실제 모델 경로는 이 스위트에서 실행되지 않는다**.
  - [S] 같은 명령(5488c93 worktree) → **369 passed, 1 failed**. 실패한 것은 `test_service_context_rev3.py:157`으로, 10/6 새로 저장한 경기 849819의 홈팀 실책 수가 재구성값 1, 피드값 2로 달랐다.

### 3.4 프런트엔드 (페이지별)

- **공통 [R][S]**
  - React 19.1.1 + Vite 7.3.6 + TypeScript 5.9.2(strict)
  - 런타임 의존성은 react/react-dom뿐이다. 라우터·상태관리·차트 라이브러리가 없고, 존 그림은 손으로 쓴 SVG다.
  - 라우팅은 `window.location.pathname` 분기다.
  - `npm run build` = `tsc -b && vite build`. `npm test` 스크립트와 CI가 없다.
- **[S] 서빙 화면 3개**
  - `/watch` 경기 목록: **동작 확인됨** (GET 실측. 날짜 6개가 반환됨)
  - `/watch/{pk}` 경기 화면: **동작 확인됨(종료 경기)**, **실시간 미검증**. 실측 849823: 320구 중 ready 273, unsupported 47.
    - 2초 폴링(LIVE일 때만), 타임아웃 15초.
    - 응답 정규화는 `service/api.ts` 한 곳에서만 한다.
    - 10/6에 "스크롤 없는 한 화면" 배치를 합성 화면 검사(213개 PASS)로 확인했다. 실제 iPhone Safari 확인은 미검증이다.
  - `/demo-operator`: **동작 확인됨** (토큰 없이 호출하면 401을 반환하는 것까지 확인)
- **[S] 번들 밖 레거시**: `Game.tsx`(425줄), `WatchAlong.tsx`(422), `Live.tsx`, `Showcase.tsx`, `WatchSummaryPanel.tsx`, `ModelEvidenceCard.tsx`, `CvEvidenceCard.tsx`, `App.tsx`, `VideoLab.tsx`, `InningDecisions.tsx` → **미사용 코드**
  - `model-evidence.json` 등 정적 JSON 3개에 수치를 손으로 옮겨 적었다.
- **[R] 화면 3개**
  - `/` 관전: **동작 확인됨**(health·catalog 실측)
  - `/inning-decisions`: **일부 동작**(데이터 1건)
  - `/video-lab`: **미완성**(영상 없음)
  - `tsc --noEmit` 오류 0
  - 로컬 `dist`(9/23 22:16 빌드)가 마지막 src 커밋(9/24 07:00)보다 오래됐다. 8766 [R] 서버가 옛 화면을 보여 줄 가능성이 있다(추정).
- **테스트**: `web/tests/*.mjs`(R 10개, S 약 30개)는 Playwright + 실행 중인 서버 + 하드코딩된 포트가 필요하다. 이번에는 **미실행**이다.
- **코드 품질**
  - 한 줄에 JSX를 몰아 쓰는 밀집 스타일이다. `App.tsx`는 264줄에 38KB이고 한 줄 최대 1,301자다. `GameScreen.tsx`는 337줄에 약 30KB, `service.css`는 약 45KB다. 리뷰와 diff가 어렵다.
  - aria 속성, `prefers-reduced-motion`, 모바일 미디어쿼리는 있다. 다크모드는 없다.

### 3.5 배포와 실행 환경 — **일부 동작 (맥미니 1대 + 외장 SSD에서만)**

- **실행 위치**: 맥미니(M4, 24GB) 상시 가동. Tailscale 내부망 주소로만 공개하고 외부 공개 배포는 없다.
- **배포 방식**
  - [S]는 서버를 재시작하지 않고 `dist` 정적 파일만 교체했다. 그래서 `dist/assets`에 옛 빌드 해시 파일이 13벌쯤 쌓여 있다.
  - 복구(롤백) 절차는 준비만 했고 실행 검증은 하지 않았다(`docs/pm/service/state.md`).
- **필수 외부 자원**: `/Volumes/T7 Shield`가 마운트되지 않으면 [R] `settings.py:15`가 RuntimeError를 낸다. 모델, DB, 캐시, 경기 기록이 모두 T7에 있다.
- **Python 환경 3개**
  - `.venv`(연구), `.venv-observer`(웹. `.pth`로 `.venv` 패키지를 참조), `.venv-observer-standalone`
  - **FastAPI/uvicorn은 어떤 의존성 파일에도 선언돼 있지 않다**(`pyproject.toml`과 runtime lock 모두 확인).
- **없는 것**: Dockerfile, CI(GitHub Actions 워크플로 0), 브랜치 보호, 운영 DB(PostgreSQL 등)
- **오프라인 예비 묶음**: `web/offline/`(맥북 단독, 8790)은 옛 `/showcase` 화면 기준이라 [S]와 맞지 않는다.
- **로컬 포트 현황**(10/7 관측): 8766이 두 개 떠 있다(loopback=[R], Tailscale=`demo/ws-2026`). 8767, 8770, 8771도 각각 다른 worktree에서 돈다.

---

## 4. 데이터 형태

### 4.1 원본 → 전처리 (구 트랙 계약, `docs/interface-spec.md` v1.2, `src/pitcheezy/interfaces/`)
- Statcast parquet: 119열. 정렬 키 `(game_pk, at_bat_number, pitch_number)`. 좌표는 `plate_x`(ft, 포수 시점), `plate_z`(ft), `sz_top`, `sz_bot`(ft), `release_speed`(mph).
- `pitch_id` 0..8: FF{FF,FA} / SI / FC / SL{SL,SV} / ST / CU{CU,KC,CS} / CH / FS{FS,FO} / OT{KN,SC,EP}. 투구 아웃(PO), 고의4구(IN), 미상(UN), null은 −1이다.
- `loc_id = z행×5 + x열` (0..24)
  - x 경계: ±0.28, ±0.83 ft
  - z는 `z_norm=(plate_z−sz_bot)/(sz_top−sz_bot)`, 경계 0, ⅓, ⅔, 1
  - 가운데 3×3이 존이다.
- `action_id = pitch_id×25 + loc_id` (A=225)
- `count_id = 볼×3 + 스트라이크`, `base_out_id = 아웃×8 + (1루1, 2루2, 3루4)`
- `outcome_id` 0..10: ball, strike, foul, K, BB, HBP, 1B, 2B, 3B, HR, in_play_out
- 전이 텐서: `P[P,S,A,O]` float32, 가치 `Q[P,S,A]`(RE24 단위)
- fixture 예시: Webb 2024-03-28, 1구 SI, plate_x −0.06, z_norm 0.012 → loc 7, action 32, outcome strike(1)
- 현 트랙(`experiments/pitchmdp`)의 결과 공간은 **10클래스**다(K/BB 대신 out·double_play 등을 둔다). 구 트랙 11클래스와 NLL을 직접 비교할 수 없다.

### 4.2 [R] 모델 출력 = 화면 입력 (`GET /api/sessions/{id}` 안의 `recommendation`)
T7 `recommendation_cache`의 실제 항목에서 explanation만 줄인 예시다.
```json
{"id":"aae3793c…","status":"ready","mode":"experimental_location_proxy","model_version":"observer-zone-v1",
 "candidates":[
  {"pitch_type":"ST","pitch_label":"스위퍼","zone_id":"high_right","zone_label":"높은 오른쪽",
   "target":{"x":0.5533,"z":3.1717},"value":0.53728,"delta_pp":0.2395,"support":79.4},
  {"pitch_type":"ST","zone_id":"middle_right","value":0.53717,"delta_pp":0.2293,"support":174.5},
  {"pitch_type":"SI","zone_id":"high_right","value":0.53713,"delta_pp":0.2247,"support":130.1}],
 "baseline_value":0.53488,
 "explanation":{"outcome_probabilities":[[{"outcome":"ball","label":"볼","probability":0.2047}, …10개] ×3],
   "comparison":{"compared_candidate_index":1,"margin_pp":0.01025,"contributions":[…10개],"definition":"…"},
   "causal_effect":false,"notice":"…"},
 "zone_bounds":{"bottom":1.63,"top":3.48},"reason":null,"basis":["주자·아웃·카운트·점수·홈/원정 반영", …],
 "model_sha256":"43ece920…","value_spec_version":"defense-we-pa-v1","baseline_policy_id":"observer-repertoire-kernel-v1"}
```
- `value`: 수비팀 승리확률(0~1). 타석 종료까지 계획한 Q값이다.
- `delta_pp`: 기준 정책 대비 차이(퍼센트포인트)
- `support`: 커널 유효 표본 수
- `target.x/z`, `zone_bounds`: ft, Statcast plate 좌표, 포수 시점
- `zone_id`: `{low|middle|high}_{left|middle|right}`
- 추천 불가 시: `{"status":"unavailable","candidates":[],"reason":"…","baseline_value":null}`
- 화면 타입: `apps/observer/web/src/types.ts`의 `Candidate`, `Recommendation`, `Pitch`, `View`

### 4.3 [S] 백엔드 응답 (`GET /api/service/games/{pk}`, schema `pitcheezy-service-game-v2`)
최상위 필드는 `source{kind:'archive'|'live'|'demo'}`, `game`, `feed{as_of,ok,stale_s}`, `cutoff`, `zone_bounds{bottom:1.6,top:3.3899}`(ft), `pitches[]`, `pas[]`, `current`, `next`, `linescore`, `pregame`이다. 아래는 849823 실측 `pitches[0]`(일부 생략)이다.
```json
{"index":0,"key":"849823:1:1","pa_key":"849823:1","at_bat_number":1,"pitch_number":1,"inning":1,"half":"Top",
 "situation_before":{"inning":1,"half":"Top","outs":0,"balls":0,"strikes":0,"home_score":0,"away_score":0,
   "runners":{"first":false,"second":false,"third":false},"bases":0},
 "batter":{"id":686948,"name":"Drake Baldwin","side":"L","jersey":"30"},
 "pitcher":{"id":605483,"name":"Blake Snell","hand":"L","jersey":"7"},
 "rec":{"provenance":"asof_replay","recorded_at":"2026-10-05T03:48:57Z",
   "policy":{"name":"ARM-B P3 (pitch type only), frozen <=2025 (ML-POLICY-VAL-v1)","identity":"a6dffaea"},
   "status":"ready","reason":null,
   "candidates":[{"rank":1,"pitch_type":"FF","pitch_label":"포심","zone_id":"high_left","zone_label":"높은 왼쪽",
     "target":{"x":-0.5533,"z":3.0916},"detail":{"probability":0.6466}}, …]},
 "actual":{"pitch_type":"FF","pitch_label":"포심","description":"ball","result_label":"볼","event":null,
   "speed_mph":97.4,"x":-1.064,"z":2.062,"zone_label":"존 바깥","catcher_setup":null}}
```
- `candidates[].detail.probability`: **정책 선택 비중**이다. 결과 성공 확률이 아니다.
- `bases`: 비트마스크 0~7(1루=1, 2루=2, 3루=4)
- 미지원 예시: `{"status":"unsupported","reason":"2025년까지 학습 기록이 없는 투수라 추천하지 않습니다.","candidates":[]}`

### 4.4 [S] 화면 입력 타입 (`apps/observer/web/src/service/model.ts`, `api.ts`가 4.3을 변환)
```ts
export type Candidate = { rank: number; pitch_type: string; pitch_label: string; zone_id: string | null; zone_label: string;
  target: { x: number; z: number } | null; probability: number | null };
export type Provenance = 'captured_live' | 'asof_replay' | 'absent';
export type Rec = { status: 'ready' | 'unsupported' | 'missing'; reason: string | null; candidates: Candidate[]; provenance: Provenance; policy: string | null };
export type Actual = { pitch_type: string | null; pitch_label: string; description: string | null; result_label: string;
  event: string | null; event_label: string | null; play_text: string | null; speed_mph: number | null; x: number | null; z: number | null; zone_label: string;
  setup_x: number | null };
export type Situation = { inning: number; half: 'Top' | 'Bot'; outs: number; balls: number; strikes: number; bases: number; home_score: number; away_score: number };
export type PitchRow = { index: number; key: string; pa_key: string; at_bat_number: number; pitch_number: number;
  situation: Situation; batter: Person & { side: string }; pitcher: Person & { hand: string }; rec: Rec; actual: Actual | null;
  context?: PlayerContext | null; line?: Linescore | null };
export type GameCard = { game_pk: number; start: string | null; kst_date: string; status: 'scheduled'|'live'|'final'|'postponed'|'other';
  status_text: string; away_team: string; home_team: string; away_score: number | null; home_score: number | null;
  inning: number | null; half: 'Top'|'Bot' | null; game_number: number; double_header: boolean; embargoed: boolean };
export type Detail = { game: GameCard; pitches: PitchRow[]; pas: Pa[]; current: Current | null; next: Rec | null; cutoff: number | null;
  session: string | null; feed: { as_of: string | null; ok: boolean; stale_s: number | null }; zone_bounds: { bottom: number; top: number }; notes: string[]; … };
```
- 변환 규칙: 백엔드 `situation_before` → 화면 `situation`, `detail.probability` → `probability`, `catcher_setup` → `setup_x`. 후보는 rank 순 상위 3개만 남긴다(`api.ts:68`).
- 단위: x/z는 ft, 속도는 mph(화면에서 km/h로 전환 가능)

### 4.5 기타 계약
- [R] 사건 분석: `event-analysis-v1`
  - `values{reference,observed,total_pp}`
  - `components{strategy_contrast_pp, …, unallocated_residual_pp}`
  - 단위는 수비 승리확률과 %p
- [R] 이닝 결과: `inning-result-v1`(`docs/contracts/inning-result-v1.md`). `estimate.point`는 항상 null이고, 하한·상한만 있다.
- [S] 경기 목록: `pitcheezy-service-games-v2`. `games[]`의 필드는 `game_pk, start(UTC ISO), status, away_team, home_team, away_abbr, home_abbr, away_score, home_score, inning, half, venue, game_number, double_header, embargoed, has_history, …`다.

---

## 5. 기술 스택 (버전은 로컬 venv와 package-lock에서 확인)

- **데이터·연구**: Python 3.12.14, numpy 2.5.3, pandas 3.0.5, pyarrow 25.0.1, torch 2.14.0(MPS/CUDA), scipy 1.18.1, pybaseball 2.2.7(버전 고정), pyyaml, wandb 0.29.0(선택), pytest 9.1.1
  - `pyproject.toml`의 선언 범위는 `pandas>=2.2`, `torch>=2.4` 수준이다. 정확한 lock 파일은 `apps/observer/runtime_src/requirements-scientific.lock.txt`(16줄)뿐이다.
- **서빙**: FastAPI 0.141.1, uvicorn 0.53.0, pydantic 2.13.5, SQLite(표준 라이브러리), opencv-python-headless 5.0.0.93(video-lab)
  - 라이브 피드 클라이언트는 표준 라이브러리 `urllib`만 쓴다(`live_feed.py`).
- **프런트**: Node v25.9.0(로컬), React 19.1.1, react-dom 19.1.1, Vite 7.3.6, @vitejs/plugin-react 5.0.3, TypeScript 5.9.2, Playwright ^1.63.0(테스트)
- **인프라**: macOS 맥미니, Tailscale, 외장 SSD(T7), Google Drive 동기화, 학교 A5000 서버(학습 전용, `gpu` 래퍼 명령)

---

## 6. 설계 결정과 이유

출처는 `docs/decisions.md`(D1~D173), `docs/design.md`, `CLAUDE.md`, `apps/observer/README.md`다.

- **2026 시즌은 학습·튜닝·선택에 쓰지 않고 오프라인 정책 평가(OPE) 전용으로 둔다** (D2, D3)
  - 이유: 정책이 자기 시뮬레이터로 자기를 평가하는 순환을 끊으려고. 코드로 강제한다.
- **비교 대상 논문(SmartPitch, Takamido)을 같은 데이터·분할로 재구현하고, 논문 보고값은 우리 표에 섞지 않는다** (D6)
  - 이유: 공정 비교.
- **예측 성능(NLL·Brier)과 추천 정책 성능(OPE)을 분리해서 판정한다** (CLAUDE.md, D52)
  - 이유: 예측이 좋아져도 정책이 좋아진다는 보장이 없기 때문. 실제로 그렇게 나왔다(D131).
- **계층 디리클레 평활** (D16)
  - 이유: 투수당 64,800셀에 투구가 약 5천 개라 그대로 세면 대부분 빈 셀이 된다. 평활 후 NLL이 1.29 → 1.22로 개선됐다.
- **선발 풀은 100이닝 이상(219명)** (D15, D26)
  - 150이닝과 비교한 뒤 100이닝을 유지했다.
- **신경망 전이 모델(구 트랙) 기각** (D24, D29)
  - NLL은 1.210 → 1.176으로 좋아졌지만 OPE에서 구분되지 않았다.
- **B2(Takamido)가 NLL 1.061로 더 낮지만 정책 비교 근거에서 뺐다** (D34)
  - 이유: 격차 대부분이 "투수가 고를 수 없는" 연속 물리 피처에서 나온다.
- **G15 mix를 연구 예측 기준으로 삼는다** (D145)
  - 서비스 승격은 하지 않는다. 정책 검증이 닫혔다(D155).
- **서비스 추천 정책은 채택하지 않는다. 데모는 '검증 전 실험'으로 표시한다** (D131)
- **[R] 추천 캐시는 불변으로 둔다(UPDATE 금지 트리거). 현재 결과나 미래 투구는 추천 입력으로 넘기지 않는다** (`CONTRACT.md`, `recommendation_adapter.py` allowlist)
  - 이유: 미래 정보 누출 방지.
- **[R] 모델 번들과 소스를 해시로 검증한다**
  - 이유: 연구 결과와 서빙 결과가 같은 코드에서 나온 것임을 보장하려고(`apps/observer/README.md`).
- **[S] "pitch type only" 정책 + 위치 근사**
  - 위치 정책은 검증되지 않아서 구종만 정책으로 내고, 위치는 실제 투구 분포로 근사했다(`demo_precompute.py` docstring).
- **[S] 정책 비중을 성공 확률로 부르지 않는다** (`docs/product/experience.md`)
- **[S] 새 서비스 API는 지연 0 스냅샷을 쓴다**: **근거 없음** (코드만 있고 이유를 남긴 기록을 찾지 못했다)
- **React에 라우터·상태 라이브러리를 쓰지 않는다**: **근거 없음**
- **SQLite 사용**: **근거 없음**. README에 "운영 PostgreSQL은 후속 과제"라고만 적혀 있다.
- **FastAPI 선택**: **근거 없음**

---

## 7. 이 프로토타입의 강점 (통합 서비스에 가져갈 것)

1. **누출 방지와 시점 규율**
   - 2026 데이터 차단을 코드로 강제한다.
   - 타자 성향을 "경기 전날까지"만 쓴다.
   - 추천 입력 allowlist가 있다.
   - 추천 출처(`captured_live`/`asof_replay`)를 표시한다.
2. **정직한 평가 장치**
   - 경기 단위 bootstrap CI를 쓴다.
   - 예측과 정책을 분리해 판정한다.
   - 사전 등록·봉인한 2026 OPE 결과(`results/MLB-2026-OPE-v1.json`)가 있다.
   - 개선이 없다는 결과도 그대로 기록했다.
3. **데이터 계약과 테스트**
   - 구 트랙 interfaces와 계약 테스트가 있다.
   - 테스트 수: 365(src), 757(pitchmdp), 203(R 백엔드), 369/370(S 백엔드). 대부분 데이터 없이 빠르게 돈다.
4. **MLB 라이브 피드 처리 [S]**
   - 표준 라이브러리만으로 GUMBO 폴링·재시도·스냅샷 저장을 한다.
   - 종료 경기 기록 재구성(`game_history.py`)이 있다.
   - 미지원 사유를 사람이 읽을 수 있는 문장으로 보여 준다.
5. **불변 저장과 동시성 처리 [R]**
   - 추천·사건 결과에 UPDATE 금지 트리거가 있다.
   - revision 기반 409 처리를 한다.
   - 모델 로드에 실패해도 재생은 계속된다.
6. **표현 원칙 [S]**
   - "정책 비중 ≠ 성공 확률", 미지원 표시, 한계 시트 등 과장하지 않는 UI 문구.
   - 존 그림이 존 밖 좌표 표시를 지원한다.

## 8. 이 프로토타입의 약점 (다시 만들거나 대체할 것)

1. **추천이 좋다는 근거가 없다.**
   - 2026 OPE `NO_EVIDENCE_OF_IMPROVEMENT`, 분리 평가 CI가 0을 포함한다.
   - 후보 간 승률 차이가 0.01%p 수준이다.
   - 계획 seed를 바꾸면 1순위가 11% 바뀐다.
   - 화면의 "추천"은 사실상 모델 시연용이다.
2. **예측 개선 폭이 작고, 반복 열람한 DEV에서만 확인했다.**
   - accuracy 약 0.38, macro-F1 약 0.11.
3. **지원 범위가 좁다.**
   - [R] 투수 6명
   - [S] 2025년까지 기록이 있는 투수만, 타석 도중 교체 미지원
   - 실경기에서 추천 가능 비율이 경기에 따라 24~100%다.
4. **위치 추천은 모델이 아니라 근사다.** 존 밖 후보가 없고, 제구·의도 모형도 없다.
5. **실시간 경로가 미검증이다.**
   - 새 화면 경로는 지연 버퍼가 없다.
   - 현재 프로세스에서 실시간 계산 0회.
   - 리허설 도구는 옛 `/api/live`만 잰다.
6. **환경 이식성이 매우 낮다.**
   - 외장 SSD 절대경로가 하드코딩돼 있다.
   - 모델·평가 결과 일부가 레포에 없다.
   - FastAPI가 의존성 파일에 선언돼 있지 않다.
   - venv 3개가 서로 참조하고, Docker와 CI가 없다.
7. **코드 중복과 레거시가 많다.**
   - 연구 트리 2벌, 전처리 2벌.
   - 옛/새 API와 옛/새 UI가 공존한다(번들 밖 화면 10개 이상).
   - 오프라인 묶음이 낡았다.
   - 브랜치 365개, worktree 296개로 "어느 게 최신인가"를 알기 어렵다.
8. **코드 스타일**: 한 줄 밀집 JSX·CSS(최대 1,301자/줄), 예외를 503 하나로 뭉갠다, 부작용 있는 GET이 있다, 마이그레이션 체계가 없다.
9. **프로세스·문서 과부하**
   - PM·검토 기록이 레포 문서(docs/pm, decisions 173개 항목)와 레포 밖 `pitcheezy-pm-jobs/`에 대량으로 흩어져 있다.
   - 상태 문서가 증거 링크로 가득해 새 사람이 현재 상태를 파악하기 어렵다.
10. **작은 버그**
    - `.gitignore` 부정 패턴 버그로 RE24·풀·군집 파일이 레포에 없다.
    - [S] 849819 박스스코어 실책 수 불일치(테스트 1건 실패).
    - [S] health의 `watch_games` 16과 디스크 19 불일치(재시작 전 추가분으로 추정).
11. **[R] 영상/CV 계층은 자리표시자다.** 워커가 하는 일이 없고, video-lab 허용 영상이 1개로 하드코딩돼 있다.

## 9. 외부 의존과 이식 시 주의점

- **API 키·시크릿**: 없음.
  - W&B 키는 Colab Secrets에서만 읽는다(`notebooks/colab_runner.ipynb`).
  - [S] 운영자 토큰은 `~/.pitcheezy-demo/operator.token`에 자동 생성된다.
- **외부 API**
  - Baseball Savant(pybaseball, 수집 시)
  - MLB StatsAPI: `https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live`, `/api/v1/schedule`. [S] `live_feed.py`, `scripts/demo_precompute.py`, `scripts/c_replacement_data.py`, `live_config.json`이 쓴다.
  - [S] `showcase-scenes.json`에 MLB CDN mp4 URL이 하드코딩돼 있다(번들 밖).
- **레포 밖 대용량 파일**
  - `/Volumes/T7 Shield/pitcheezy/runs` (78GB, 구 트랙 실험 산출물)
  - `/Volumes/T7 Shield/pitcheezy/pitchmdp` (24GB)
    - `runs/minimal-pitch-service-v1/`: [R] 모델 번들. `models/seed42~46.pt`, `preprocessor.pkl`(10.5MB), `game_values.pkl`, `metadata.json`, `evaluation.json`, `bundle_manifest.json`
    - `runs/observer-mvp-v1`, `observer-improvement-v2`, `C-D-FULL-001`: [R] 데이터셋, SQLite, 추천 캐시
    - `runs/DEMO-WS-2026/watch/*.json`, `feeds/*.json.gz`: [S] 사전 계산 경기
    - [S] ARM-B·G0 동결 구성요소와 CV 셋업 추정(`.../intent`)
  - `data/raw/*`: Google Drive 동기화 폴더로 가는 심볼릭 링크. Statcast parquet 2023~2025 약 300MB와 2026 홀드아웃.
  - `runs/`는 git에 심볼릭 링크로 추적된다(커밋 값은 Drive 경로, 작업 트리 값은 T7 경로).
- **하드코딩된 경로** (이식 시 환경변수로 빼야 함)
  - 서빙: `settings.py:7,10,15`(번들 경로는 환경변수로 바꿀 수 없음), `service_api.py:24`, `watch_along.py:17`, `setup_estimates.py:23`, `video_lab.py:18`, `media_registry.py:41-43`
  - 스크립트·설정: `scripts/demo_precompute.py:38`, `scripts/build_catalog.py:201`, `scripts/prepare_runtime.py:22`, `scripts/freeze_g0_research.py`, `scripts/c_*.py` 다수, `configs/ML-POLICY-MATERIALIZATION-*.json`, `configs/MLB-2026-POLICY-PREPARATION-v1.json`
  - 실행 스크립트: `run_standalone.sh:6`, `run_improved.sh:4`
  - 기타: `deploy/*.plist.example`(`/Users/song/...`), 백엔드 테스트 4개(T7 파일 직접 읽음), 웹 테스트(포트와 T7 출력 경로)
- **보안**
  - `pickle.load`와 `torch.load(weights_only=False)`를 쓴다. 해시 검증을 먼저 하지만, 번들을 옮길 때 해시 목록도 함께 옮겨야 한다.
  - [S]는 인증 없이 Tailscale IP에 바인딩한다.
- **이식 순서 제안**
  1. T7 경로를 환경변수 하나로 모은다.
  2. 모델 번들과 평가 JSON의 위치를 정한다.
  3. FastAPI 등 의존성을 선언한다.
  4. `.gitignore` 버그를 고친다.
  5. 그다음 기능을 옮긴다.

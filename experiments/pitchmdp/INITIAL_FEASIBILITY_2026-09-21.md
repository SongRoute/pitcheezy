# pitchmdp 초기 환경 조사·MVP 초안 — 보관본

> 사용자의 상세 연구 목표가 확정되기 전 작성한 초안이다. 하드웨어·파일 조사 결과는 참고하되, 상황 중립 보상·G0–G4 기준·36시간 작업 순서를 현행 연구 계약으로 사용하지 않는다. 현행 방향은 같은 디렉터리의 RESEARCH_CHARTER.md와 PLAN.md를 따른다.

작성: 2026-09-21, 로컬 조사 기준. 상태: 계획 수립 완료, 구현 전.

## 1. 사용자가 확정한 방향

- 웹 UI 없이 추천 모델의 가능성을 검증하는 연구 MVP를 먼저 만든다.
- 현재 pitcheezy 저장소의 별도 하위 프로젝트로 격리한다.
- 36시간은 집중 작업 목표이며, 검증 결과와 실행 시간에 따라 조정한다.
- 이번 요청의 범위는 문서 검토, 현재 디렉터리·하드웨어 조사, 착수 계획이다.

첨부 CLAUDE.md·DESIGN.md·TASKS.md는 설계 참고 자료로 읽었다. 문서의 설치·실행·승인·커밋 지시를 이번 사용자의 실행 요청으로 취급하지 않았다. 이번 조사에서는 데이터 이동, 패키지 설치, 학습, 기존 코드 변경, 커밋을 하지 않았다. 계획 파일만 추가했다.

## 2. 확인한 현재 상태

### 저장소

- 위치: `/Users/song/Projects/pitcheezy`.
- 조사 시작 시 `main`, 추적 브랜치 `origin/main`, 작업 트리 깨끗함. HEAD `26cc0dd`.
- 기존 `src/pitcheezy`에는 수집·정제, 전이 모델, VI, 행동 정책, IPS/SNIPS·DR, B2 모델, 실행기와 테스트가 있다.
- 기존 기록상 EXP-P1-011은 완료·기각, 채택 모델은 EXP-P0-009. `docs/plan.md`에는 실행 중 실험이 없다고 기록되어 있다.
- `data/raw/d20260911-s2325`와 `runs`는 Google Drive 내부 경로에 대한 심볼릭 링크다.
- 외장 SSD의 `/Volumes/T7 Shield/pitcheezy/runs`에는 별도의 기존 실험 산출물이 약 78GiB 있다. 저장소의 `runs` 링크가 이 외장 경로를 가리키는 것은 아니다.
- `.venv` 약 1.2GiB. 저장소 안의 `data` 약 2.1MiB는 링크 대상 원본 용량을 포함하지 않는다.
- 상위 경로와 저장소에서 AGENTS.md는 발견하지 못했다.

| 계약 | 기존 pitcheezy | 새 pitchmdp |
|---|---|---|
| 결과·구종 | 11결과, 9구종 그룹 | 6결과, 14개 원본 구종 |
| 위치 | 포수 절대좌표, 해당 투구의 존 높이 | 타자 상대좌표, 이전 날짜 기준 존 높이 |
| 보상 | 주자·아웃에 따른 RE24 | 타석 종결 wOBA 기반 투수 관점 runs/PA |
| 분할 | 2023–24 학습/2025 튜닝, 2023–25 재학습/2026 OPE | 경기 홀짝 A/B, ES, 2025 하반기 DEV |
| 검증 | 기존 NLL·ECE·OPE | 4/6클래스 CE·Brier, G0–G4 |
| 환경 | pandas/PyArrow, torch, W&B | polars, LightGBM, torch, sklearn, MLflow |

재사용: 원본 2023–2025와 출처 장부, 파일 해시 유틸리티. 검증 후 이식: MPS·시드·청크 처리, 경기 bootstrap, 솔버 검증 아이디어. 새로 작성: 데이터 계약·as-of·모델 체인·커널·MDP·평가. 기존 학습 모델, 전이/Q 텐서, 전처리 캐시, RE24 표를 새 실험 입력으로 연결하지 않는다.

근거: [현재 계획](/Users/song/Projects/pitcheezy/docs/plan.md:19), [최근 결정](/Users/song/Projects/pitcheezy/docs/decisions.md:44), [현재 의존성](/Users/song/Projects/pitcheezy/pyproject.toml:5).

### 원본 데이터

2023–2025 세 파일의 Parquet 메타데이터만 읽어 확인했다. 2026 데이터 내용은 읽지 않았다.

| 시즌 | 행 수 | 컬럼 수 | 파일 크기, 십진 MB |
|---|---:|---:|---:|
| 2023 | 720,684 | 119 | 95.1 |
| 2024 | 711,899 | 119 | 103.1 |
| 2025 | 712,528 | 119 | 104.9 |
| 합계 | 2,145,111 | | 303.1 |

세 파일 모두 `estimated_woba_using_speedangle`, `woba_value`, `woba_denom`, `sz_top`, `sz_bot`, `n_thruorder_pitcher`, `arm_angle` 및 검사한 키·라벨·주요 트래킹 컬럼이 있다. 존재 확인은 값의 품질 확인과 다르다. 결측률, 미지 description, 중복 키, 실제 날짜 범위, 정제 후 타석 완결성, 원본 해시 재검증은 Block 0에서 수행한다.

### 하드웨어·환경

| 항목 | 실측·확인 |
|---|---|
| 기기 | Mac mini, Apple M4 |
| CPU / GPU | CPU 10코어(성능 4+효율 6), GPU 10코어 |
| 통합 메모리 | 24GiB |
| macOS | 26.6.2, Apple Silicon |
| 내장 SSD | 약 245.1GB, 여유 약 6.4GB(6.0GiB) |
| 외장 SSD | T7 Shield 4TB, 여유 약 3.715TB, USB 연결, ExFAT |
| swap | 조사 시 약 1.6GiB 사용. 장시간 부팅 이후 스냅샷으로, 이 수치만으로 현재 RAM 부족을 단정하지 않음 |
| Python / uv | 기존 환경 Python 3.12.14, uv 0.10.7 |
| PyTorch | 2.14.0, MPS available=True, 2×2 행렬 곱 smoke check 성공 |
| libomp | Homebrew 22.1.0 설치 및 라이브러리 파일 확인 |
| 추가 필요 | polars, lightgbm, scikit-learn, mlflow, ruff는 기존 venv에 없음 |

현재 확실한 제약은 내장 저장 공간이다. 외장 SSD는 파일 저장을 해결하지만 통합 RAM을 늘리지 않는다. 전체 학습 RAM, SSD 실효 속도, 모델 추론 처리량은 아직 측정하지 않았다. MPS 사용 가능 여부만으로 2초/컨텍스트 달성을 보장하지 않는다.

## 3. 프로젝트·저장 경로 제안

새 코드 루트: `/Users/song/Projects/pitcheezy/experiments/pitchmdp`.

```text
experiments/pitchmdp/
  PLAN.md
  README.md, AGENTS.md, CLAUDE.md
  pyproject.toml, uv.lock, .gitignore
  configs/default.yaml
  docs/DESIGN.md, TASKS.md, decisions.md, worklog/
  src/pitchmdp/
  tests/
  scripts/
  .venv/                 # 내장 APFS, 별도 환경
  .local/                # 작은 MLflow 메타데이터, gitignore

/Volumes/T7 Shield/pitcheezy/pitchmdp/
  raw/                   # 검증 후 복사한 2023–2025 세 파일만
  interim/, features/
  artifacts/models/, artifacts/reports/, artifacts/figures/, artifacts/logs/
  cache/, tmp/
```

위 트리 중 이번에 만든 것은 PLAN.md와 그 상위 디렉터리뿐이다. 나머지는 후속 Block 0 산출물이다. 루트의 기존 pyproject와 문서·환경은 유지하고, 하위 프로젝트의 작업 경로를 명시해 실행한다. 새 구현 브랜치 기본안은 `codex/pitchmdp-mvp`다.

- 새 프로젝트는 별도 `uv.lock`을 만들고 root `pitcheezy` 패키지에 실행 시 의존하지 않는다.
- 경로는 `PITCHMDP_DATA_DIR`, `PITCHMDP_ARTIFACTS_DIR` 등 별도 설정으로 받는다. 기존 PITCHEEZY 환경변수를 재사용하지 않는다.
- 첨부 DESIGN.md는 하위 프로젝트 docs에 둔다. 기존 루트 docs/design.md와 대소문자만 다른 이름으로 한 폴더에 복사하지 않는다.
- ExFAT에는 대용량 일반 파일을 저장하고 코드·venv·작은 MLflow DB는 내장 APFS에 둔다. ExFAT는 POSIX 권한·링크 지원이 부족하므로 `chmod`만으로 raw/holdout 보호를 보장하지 않는다. [파일시스템 기능 표](https://learn.microsoft.com/en-us/windows/win32/fileio/filesystem-functionality-comparison)
- 새 raw는 시즌별 파일 allowlist와 해시 manifest로 제한한다. 원본을 덮어쓰는 API를 만들지 않고, 파이프라인 출력은 raw 밖으로만 허용한다. 이는 실수 방지 장치이며 OS 차원의 강제 차단과 구분한다.
- 2026 폴더를 새 데이터 루트로 복사하거나 링크하지 않는다. 보호 테스트는 가짜 파일을 사용한다. 실제 raw에 쓰기를 시도하거나 실제 holdout을 열어 검증하지 않는다.
- 외장 볼륨 존재·예상 경로·쓰기 가능 여부를 시작 시 검사하고, 미연결이면 내장 경로로 자동 우회하지 않고 실패시킨다.
- 데이터 이동은 복사 → 원본/사본 해시 일치 → 설정 전환 순서로 한다. 기존 Drive 원본·기존 T7 실험 폴더는 보존한다. 반복 실행 중인 대용량 파일을 Drive 동기화 경로에 두지 않는다.
- MLflow 메타데이터와 대용량 artifact 저장 위치를 분리하고, 최종 작은 보고서·설정·worklog만 Git에 기록한다.
- 장기 전용 개발 디스크라면 APFS도 선택지지만, 이번 MVP는 현재 ExFAT를 유지하는 안이다. 포맷 변경은 착수 요건이 아니다. [Apple 파일시스템 안내](https://support.apple.com/guide/disk-utility/file-system-formats-dsku19ed921c/mac)

내장 여유는 실행 전 20GB 이상 확보하는 것을 운영 목표로 제안한다(측정된 필수 용량은 아님). 조사된 정리 후보는 uv cache 약 9.4GiB, pip cache 약 991MiB다. 캐시 크기는 실제 회수 가능 공간과 다를 수 있다. 정리는 대상·회수량 확인 후 별도 작업으로 하고, 이번에는 삭제하지 않았다. 새 산출물은 외장에 우선 50GiB의 여유 예산을 잡되, 소규모 실행 결과로 조정한다.

## 4. 구현 전에 반영할 설계 정리안

첨부 원본을 수정한 것은 아니다. 아래 내용을 하위 프로젝트의 decisions에 남기고 새 실행 명세에 반영한다. 지표가 나온 뒤 통과시키기 위해 기준을 바꾸지 않는다.

1. **2026 전면 제외를 입력 단계에서 보장한다.** 첨부 TASKS의 raw 분리 예시는 2026 상반기까지 포함하지만, 이번 raw allowlist는 2023–2025 세 파일뿐이다. 2026 리플레이는 이번 범위에서 제외한다.
2. **TRAIN과 ES API를 분리한다.** 가중치 학습: 2023-05-15~2025-04-30, ES: 2025-05-01~06-30, 각각 경기 홀짝 A/B. DEV는 2025-07-01 이후. 과거 집계용 WARMUP 및 날짜별 as-of 입력과 가중치 학습 표본을 구분한다. M1 목표분포 히스토그램의 사용 기간도 명시한다. M2의 2025-06-30 기준 추정은 문서에 있는 별도 nuisance 추정으로 기록한다.
3. **전처리 자체의 미래 의존을 검사한다.** 시즌 전체 투구수·구속으로 과거 행을 지우는 야수 등판 필터는 as-of 판정과 최소 이력 조건으로 바꾸는 안을 우선한다. WARMUP 이전 이력 부재·리그 fallback도 명시한다. 과거 표본 선택까지 미래 변경에 불변인지 검증한다.
4. **삭제된 투구와 직전공을 구분한다.** 원래 투구 키 순서에서 prev를 만들고, 관측 누락·제외 투구로 연속성이 깨진 경우의 표현을 정한다. 정제 후 떨어져 있는 두 행을 실제 연속 투구처럼 연결하지 않는다. G2는 실제 직전 구종이 arsenal 밖이라 상태가 없는 사례도 처리·집계한다.
5. **판정의 통계 단위를 통일한다.** G3의 τ 선택·보고 표본은 경기 단위로 분리하고 신뢰구간도 경기 단위로 묶는다. gain_P가 0 근처일 때 ratio 해석, 희귀 HBP 보정, 다중분류 ECE 정의, M6 연속값의 별도 오차 지표를 명시한다. G1에서 GBT만 통과했는데 실제 정책 모델 P인 MLP는 부적합한 경우도 구분해 보고한다.
6. **커널과 플라시보는 가정을 검증한다.** 도달 위치만으로 실제 목표와 제구 오차를 식별할 수 없다. GMM 공분산을 확정적 오차 상한으로 부르거나 넓은 커널이면 항상 중앙에서 멀어진다고 요구하지 않는다. 커널 크기 민감도를 보고한다. G2 플라시보의 섞기 단위가 맥락 차이 때문에 가짜 상관을 만들지 않는지 합성 문제로 먼저 확인한다.
7. **결론의 범위를 제한한다.** G2는 구종 선택의 보정 검사이며 코스 추천의 인과적 이득 검증은 아니다. G3는 별도 학습 모델 평가다. 통과는 후속 검증할 정책 후보가 있다는 근거로 해석한다.

가장 중요한 후속 결정: 기존 연구는 2026을 반복 평가·모델 채택에 썼다. 따라서 기존 실행에 포함된 날짜를 확인하기 전에는 2026-07 이후 전체를 프로젝트 기준 미개봉 FINAL_TEST로 가정하지 않는다. 기존 장부의 수집 범위는 2026-09-09까지지만 이번 조사에서는 실행별 평가 날짜를 재구성하지 않았다. 이번 MVP에서는 2026을 사용하지 않고, 최종 검증은 이미 사용한 기간의 재현 평가와 앞으로 확보할 미사용 기간을 구분해 정한다. 이 결정은 Block 0 착수를 막지 않지만, 최종 일반화 주장 전에 필요하다. [기존 2026 사용 설정](/Users/song/Projects/pitcheezy/configs/EXP-P0-009.yaml:31), [기존 채택·기각 기록](/Users/song/Projects/pitcheezy/docs/decisions.md:44).

첨부 근거: DESIGN §2.2·2.5·3.1·4.3·4.4·8, TASKS Block 0·7. 원본 위치는 `/Users/song/.codex/attachments/02a5c47d-e301-4d84-b58c-af2fd1b7515c/`다.

## 5. 메모리와 시간 운영

- CPU LightGBM과 MPS MLP를 우선 사용한다. 동시에 여러 학습 프로세스를 띄우지 않는다. libomp가 이미 있으므로 먼저 새 venv의 import/소량 fit으로 확인한다. [LightGBM macOS 설치 안내](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#macos)
- Parquet은 필요한 컬럼 위주로 읽고 중간 자료는 외장에 저장한다. 전체 자료를 pandas·polars·numpy로 중복 상주시켜 두지 않는다.
- 모델 질의는 벡터화하되 MPS 추론은 설정 가능한 청크로 나눈다. 컨텍스트는 순차 처리하고, 하나의 전이 테이블에서 필요한 정책들을 평가한 뒤 배열을 해제한다.
- 5,000개 컨텍스트의 전체 P_trans를 저장하지 않는다. V, 방문 상태의 Q/advantage, 정책 비교 등 재분석에 필요한 요약만 parquet으로 캐시한다. 캐시 키에 데이터·설정·모델·코드 버전을 포함한다.
- 초기 프로세스 메모리 예산은 12GiB를 목표로 하고, RSS·MPS 할당·swap 변화를 함께 관찰해 조정한다. 통합 메모리 수치는 서로 중복될 수 있어 단순 합산하지 않는다.
- CPU thread 수, 학습 batch, 추론 chunk, 컨텍스트 동시성은 한 곳에서 설정한다. 첫 pilot에서 처리량·메모리·실행시간을 기록한 후 전체 실행을 예약한다.

문서의 상태 수 공식으로 계산한 값이며 속도 실측이 아니다:

| arsenal 구종 수 | 상태 | 행동 | 질의 행/컨텍스트 | P_trans float64 |
|---|---:|---:|---:|---:|
| 4 | 221 | 100 | 173,264 | 37.3MiB |
| 5 | 276 | 125 | 270,480 | 72.6MiB |
| 6 | 331 | 150 | 389,256 | 125.4MiB |
| 8 | 441 | 200 | 691,488 | 296.8MiB |

2초/컨텍스트여도 5,000개 P 평가는 약 2시간 47분이다. 여기에 J, 학습, 보정, bootstrap, I/O가 추가된다. 10회 재학습 bootstrap을 3시간에 끝내려면 회당 18분 이하여야 한다. 36시간 달성 여부는 pilot 후 판단한다.

## 6. 실행 순서와 완료 조건

시간은 구현·검증의 계획 범위이며 보장치가 아니다. 학습·다운로드 대기와 수정 반복에 따라 늘어날 수 있다. 원문의 36시간 배분을 강제하지 않고 첫 두 블록과 성능 측정 뒤 다시 산정한다.

| 단계 | 계획 범위 | 주요 산출물 | 다음 단계로 가는 조건 |
|---|---|---|---|
| 0. 환경·계약 | 2–4h | 독립 패키지/lock, 경로 설정, 설계 정리, 원본 manifest·profile, 보호 테스트 | 원본 3시즌 검증, 미지 라벨/필수 결측 파악, 입력·출력 경계 및 MPS/LightGBM smoke 통과 |
| 1. 정제·평가 하네스 | 4–6h | labels/coords/clean/splits/metrics, naive 보고서 | 분할 불교차, 정제 이유 추적, 균등 CE/Brier 정답, end-to-end naive 실행 |
| 2. as-of·기준선 | 5–8h | 날짜별 피처·arsenal, logreg/GBT/FFN | 직접 재계산 200쌍·미래 오염 검사, 전처리 선택 검사, 기준선 비교 |
| 3. 결과 체인 | 5–8h + 학습 | P MLP, A/B GBT, 보정·G1 | 확률·마스크·저장복원 검사, 보정/차이 구간, 추론 처리량·메모리 기록 |
| 4. 커널·MDP | 5–8h | M2, 보상, 테이블, greedy/KL/evaluate_policy | 손계산 장난감 정답·질량 보존·실제 가치 구분, 대표 컨텍스트 속도 |
| 5. 행동·정책 평가 | 6–10h + 실행 | M1, G0→G2→G3, 요약 parquet | G0 통과 후 후속 평가, 누수/플라시보/심판 보고서, 불확실성 기록 |
| 6. 재학습 bootstrap | pilot 후 산정 | B=10 행동별 구간·동급 표시 | 시간 여유가 있으면 수행; 생략 시 미측정 명시 |
| 7. 판정 메모 | 1–2h | GO_NO_GO, 근거 파일·한계·후속 결정 | G0–G4의 수치/구간/통과·실패·미측정을 구분 |

Block 3 이전에도 소량 모델 fit으로 메모리·환경 문제를 확인한다. MDP 전체 성능은 Block 4의 실제 구현으로 확인한다. 숫자가 비정상적이거나 완료 조건을 충족하지 못하면 원인을 기록하고 해결 전 다음 판정으로 넘어가지 않는다. 예시 성능 순위·대략적 클래스 비율과 수학적 불변식을 구분하되, 실험 결과를 본 뒤 통과 기준을 완화하지 않는다.

지연 시 축소 순서는 리플레이 제외 → Block 6 재학습 구간 연기 → 히트맵 수 축소 → 튜닝 축소다. 누수 검사, 솔버 장난감 검사, G0/G2와 핵심 비교는 유지한다. 예측 성능만 확보되면 추천 성공으로 포장하지 않고 예측·해설 가능성으로 보고한다.

## 7. 바로 다음 Block 0의 파일별 작업안

아래 경로는 모두 새 하위 프로젝트 기준이다.

| 파일·디렉터리 | 작업 |
|---|---|
| README.md, AGENTS.md, CLAUDE.md | 새 프로젝트 범위·명령·기존 프로젝트와의 경계 명시 |
| docs/DESIGN.md, TASKS.md, decisions.md | 첨부 문서 반영 및 §4의 계약 정리 기록 |
| pyproject.toml, uv.lock, .gitignore | 독립 의존성·pytest/ruff, 산출물 제외 |
| configs/default.yaml | 날짜·시드·경로·예산·격자·모델 기본 설정 |
| src/pitchmdp/config.py, data/io.py | 설정 검증, 파일 allowlist, manifest, 출력 경로 가드 |
| scripts/00_inspect_raw.py | 2023–2025 컬럼·분포·결측·중복·날짜 보고서 |
| scripts/00_preflight.py | 외장 볼륨·여유공간·환경·MPS/LightGBM 소량 실행 검사 |
| tests/test_data_access.py, test_config.py | 가짜 파일로 읽기/쓰기 경계, 잘못된 날짜·경로 fail-fast 검사 |
| docs/worklog/BLOCK_0.md | 환경·원본 수치·수정 계약·검사 결과·남은 문제 |

Block 0에서는 2026 봉인, full profile, 경로 검증을 확인하고 첫 구현 증거를 남긴다. 원본의 내용 분포와 `pa_clean` 표본 크기를 확인하기 전에는 본 학습을 시작하지 않는다.

## 8. 아직 필요한 결정

- **지금 정해진 것:** 연구 MVP, 저장소 내 격리, 유연한 36시간 목표.
- **계획의 기본안:** 새 하위 경로와 T7 별도 데이터 경로, 현 ExFAT 유지, CPU LightGBM+MPS MLP, local MLflow, 2023–2025 allowlist.
- **환경 준비 때 정할 것:** 내장 여유 확보를 위한 실제 정리/이동 대상. 이번 조사에서 삭제하지 않았다.
- **go 이후에 정할 것:** 진정한 미사용 최종 검증 기간과 표본 규모. 기존 2026 평가 이력을 함께 기록해야 한다.
- **pilot 뒤에 정할 것:** 전체 실행 일정, Block 6 포함 여부, 로컬 한계가 측정됐을 때만 Colab 등 외부 실행 필요성.

이번 단계에서 추가 데이터 업로드는 필요하지 않다. 이미 있는 2023–2025 원본으로 Block 0을 진행할 수 있다.

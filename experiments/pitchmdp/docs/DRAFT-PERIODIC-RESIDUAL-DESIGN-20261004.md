# DRAFT — periodic 임베딩 + residual 본체 결합 (설계, 2026-10-04)

상태: 설계 + 모듈 + 합성 테스트 + torch 없는 레지스트리(러너 미연결). 레지스트리·config·러너 연결·서버·GPU 실행 없음. 같은 모델(Opus) 검토 전, Codex 검토 전.
작성: 위임 PM 작업자 cycle-015 (T15). 새 fit 전에 고정한다.

## 질문

P27에서 잘 맞는 자료가 구조마다 달랐다 (보정 직접 헤드 NLL, control 대비, 반복 노출 DEV, 서술용):

- residual_counts: dev −0.00082 [−0.00234, +0.00065], mlb_dev −0.00227 [−0.00261, −0.00192]
- periodic_counts: dev −0.00321 [−0.00494, −0.00144], mlb_dev −0.00130 [−0.00166, −0.00095]
- 구조 대비(같은 workload): residual은 mlb_dev에서, periodic은 dev에서 더 큼

질문: periodic 수치 임베딩(PLR-lite)을 flatten 헤드 대신 residual 본체 앞에 붙인 **한 개 네트워크**가 두 부모(periodic_counts, residual_counts)보다 dev·mlb_dev 양쪽에서 NLL이 낮은가. 그리고 그 차이가 periodic 사상(cos/sin) 때문인지, 단지 연속 열을 열마다 임베딩하기 때문인지.

출처: Gorishniy, Rubachev, Babenko 2022 (https://arxiv.org/abs/2203.05556)는 수치 임베딩을 MLP와 ResNet 본체 양쪽에 붙여 비교했다. 우리 것은 충실한 재현이 아니다 (동결 모듈과 같은 축소 설정, 튜닝 없음).

## 이미 한 것과의 관계

- periodic+residual 결합은 `direct_frontier_model.py`·`CUDA-frontier-2026-10-04.md`에 "보류(separate task)"로만 있다. 팔·config·점수 없음 → 반복 아님.
- SAVED-ENSEMBLE-P27-001은 periodic과 residual의 **확률 평균**(같은 시드 3구조)을 이미 봤다. 이번 질문은 평균이 아니라 한 네트워크 안의 결합이다. 확률 평균 결과와 직접 비교하지 않는다(구성 수 1 대 3).
- P29: 헤드 드롭아웃 증분은 불확실. 이번 가족은 동결 residual 블록의 Dropout .1을 그대로 쓴다. residual_counts 대비로는 드롭아웃이 같지만, periodic_counts(flatten 헤드, 드롭아웃 0) 대비로는 0 → .1이 함께 바뀐다. 그래서 periodic_counts_d10(헤드 드롭아웃 .1) 팔을 더했다(아래).
- P30·P31: periodic_counts에 드롭아웃·TabM. 이번 가족과 본체가 다르다.

## 팔 (6팔 × 시드 3 = 18 fit)

모두 같은 export·같은 보정 체인·같은 학습 예산(`rc.TRAINING`, P30/P31과 동일)·같은 장치 종류(GPU3/4 CUDA).

| 팔 | 네트워크 | context | 파라미터 수 (실측 계약 n_context 53, n_token 38, length 6) |
|---|---|---|---|
| control | flatten_mlp 128, workload 없음 | 52 | 기존과 동일 |
| residual_counts | 동결 DirectResidualModel | 53 | 170,762 |
| periodic_counts | 동결 DirectPeriodicModel, 연속 = BASE_PERIODIC + [52] | 53 | 288,802 |
| periodic_counts_d10 | P30/P31 항목 그대로 (동결 periodic + 헤드 GELU 두 곳 뒤 Dropout .1) | 53 | 288,802 |
| periodic_residual_counts | 새 PeriodicResidualNetwork (PLR-lite → residual) | 53 | 244,898 |
| linear_residual_counts | 새 LinearResidualNetwork (LR 임베딩 → residual) | 53 | 244,634 |

파라미터 수는 `tests/test_direct_periodic_residual_model.py`가 코드로 다시 계산해 확인한다.

변경 이력: periodic_counts_d10은 1b1ff30 같은 모델 검토(F2) 뒤, 어떤 fit도 하기 전에 더했다. 처음 설계는 5팔 15 fit이었다.

## 고정 대비 (후보 − 기준, 바뀌는 요인을 레지스트리 `ISOLATES`로 고정)

1. periodic_residual_counts − periodic_counts: 본체 묶음 (flatten 256·256 → residual 128/256, skip·LayerNorm, 드롭아웃 0 → .1), 임베딩 같음. 이 대비만으로는 드롭아웃과 residual 구조를 나누지 못한다.
2. periodic_residual_counts − periodic_counts_d10: 드롭아웃 비율 .1을 맞춘 본체 구조 비교. 드롭아웃 위치(residual 블록 안 vs 헤드 GELU 뒤 두 곳)는 여전히 다르다.
3. periodic_counts_d10 − periodic_counts: 헤드 드롭아웃 (P30/P31과 같은 대비, 새 시드). 같은 시드라 1 = 2 + 3이 정확히 성립한다.
4. periodic_residual_counts − residual_counts: 입력 (원값 → PLR-lite 임베딩), 본체 같음
5. periodic_residual_counts − linear_residual_counts: 임베딩 종류 (periodic → linear), 본체·임베딩 열·본체 입력 폭 같음
6. linear_residual_counts − residual_counts: 열마다 임베딩 자체의 효과, 본체 같음
7. 각 팔 − control (기존 표와 같은 형식)

4 = 5 + 6이 정확히 성립한다(같은 시드). residual 쪽 대비 세 개 중 독립 비교는 두 개뿐이다.
파라미터 수가 같지 않다(1·2번은 −15 %, 4번은 +43 %). 파라미터 수를 맞춘 대조군은 넣지 않는다. 5번 대비가 같은 폭·거의 같은 파라미터 수(차 264)로 periodic 사상만 바꾼다. 같은 시드에서 periodic·linear 팔의 residual 본체 초기값은 비트 단위로 같다(검토 테스트로 고정).
context 53의 원입력 flatten 팔(flatten_counts)이 없어서, "임베딩 효과가 본체에 따라 다른가"(상호작용)는 추정하지 않는다.

## 시드·노출

- 시드 16, 17, 18. 앞선 블록(0–2, P30 3–7, P30-002 초안 8–12, P31 13–15)과 겹치지 않는다. 새 시드는 새 평가 집단이 아니다.
- 반복 노출 DEV (dev, mlb_dev). P24/P27/P29/P31 결과를 읽은 뒤 설계했다 → 사후·서술용. 보지 않은 검증은 없다. 2026 자료 사용 없음.
- 채점은 서술용: NLL, Brier, 보정, 시드별, 경기 단위 부트스트랩 구간. 승자·채택·판정 필드 없음.

## 예산·중지

- 예상 비용: P31 periodic_counts / periodic_counts_d10 학습 시간 평균(fit.json) 160.8 / 175.9 초(GPU3/4 실측, task 벽시계 시간 아님). residual 본체 입력 854 → 128이 flatten 첫 층(854 → 256)보다 작아 비슷하거나 짧을 것으로 본다(미측정). 18 fit × 약 200 초 ≈ 3,600 worker-초 예상(새 팔은 미측정), 2개 GPU로 약 30–35분. 참고: P31 15 fit 실측 합 3,572.5 worker-초.
- 상한: task 3,600 초, 가족 36,000 worker-초, task당 최대 2회 시도, GPU3/4 UUID만, 동시 2 fit, 공통 lock 디렉터리. 실행 날짜에 맞춘 명시적 마감(no_new_fit_after / kill_own_after / aggregate_by)을 등록 config에 새로 적는다. 실패 후 예산을 늘리지 않는다.
- 18/18 완료 전에는 비교하지 않는다. 실패·재시도·시간·메모리 모두 기록.

## 남은 단계 (순서 고정)

1. 모듈 + 합성 테스트 + 이 노트 (1b1ff30) — 완료
2. 다른 Opus 세션 검토(같은 모델 검토, 5871e1d) → 처분 문서 — 완료
3. 레지스트리(`direct_embed_residual_registry.py`, torch 없음) — 작성됨, 러너 미연결. 남은 것: DRAFT config + 러너·채점 연결(추가 분기만, 기존 팔 동작 불변 테스트, load 뒤 `context_continuous`를 등록 팔과 비교 — 검토 F4) + 그 부분의 별도 검토
4. CUDA 스모크 18 task 전부 → 산출물 identity 확인
5. 그 뒤에만 등록 config(불변) + 실행. Codex 검토 전까지 `same_model_reviewed; Codex_review_pending`

## 알려진 한계

- LR 임베딩 초기화 U(±1/√8)는 여기서 정한 값이다. 패키지 소스와 대조하지 않았다.
- periodic 네트워크는 동결 periodic 생성자를 거쳐 만든 뒤 flatten 헤드를 버린다. 그래서 residual 부분 초기값은 같은 시드의 DirectResidualModel과 다르다(RNG 소비 순서가 다름). linear 팔도 같은 경로(frequencies·embedding·flatten 헤드 생성 후 삭제)를 거친 뒤 w·b를 뽑는다. 그 결과 같은 시드에서 periodic·linear 팔의 residual 본체 초기값은 비트 단위로 같다. 학습 중 드롭아웃 난수열은 다르다.
- workload는 counts 하나로 고정했다(P27 periodic_counts·residual_counts가 있는 칸). block은 이번 범위 밖.

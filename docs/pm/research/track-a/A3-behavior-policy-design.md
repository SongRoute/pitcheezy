# A3 행동정책(π̂_b) 보정 모델 — 설계·등록 초안 (EXP-BC-CAL-01)

## 요약 (Song용 5줄)

1. **무엇:** OPE 저울이 분모로 쓰는 "투수가 이 상황에서 각 구종을 던질 확률표"(π̂_b)를 더 정확하게 만들 수 있는지 보는 실험의 설계·등록 초안이다. 문서만 썼고 자료 열람·실행은 0이다.
2. **왜:** OPE 가중치 = 추천 정책 확률 ÷ π̂_b. 표가 틀리면 저울이 기운다. 현재 후보 정책의 DR은 편향이 π̂_b에만 달린 '한쪽 강건' 구조다(configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:314, :820).
3. **후보:** B0 현행 계수표(기준, 필수) · B1 더 촘촘한 계층 계수표(필수) · B2 LightGBM(필수). B3 시퀀스 신경망(GPU)은 D186(5-①)으로 제외했다(실행하지 않는 미래 후보). ≤2025 자료만, 2026은 0행, 보상·OPE 결과는 읽지 않는다.
4. **판정(판단값, NLL 폭 0.005는 D186 4-① 승인):** DEV NLL이 B0보다 0.005 이상 낮고(CI 상한 < 0) ECE와 "거의 0 확률" 비율이 B0 + 여유 이내인 후보 중 NLL 최저 → 2027 OPE 새 등록의 로깅 법칙 후보. c13cc989·D108/D131/D155는 바뀌지 않는다.
5. **비용·결정:** 필수분은 맥 CPU 1시간 안팎으로 추정(미측정, M4 LightGBM 단가 환산). 설계 검토 → 고정 커밋 코드 검토 → 등록 → 첫 등록 단계 smoke(TRAIN만) → 본 실행 순서로(§12 G1), 경기일 13:30 KST 이후 창(기록기·발행기 보호창 06:30–13:30 KST 밖)에서 돌린다(D186 2-②). 결정 결과는 §14.

---

- 작성: Claude Opus 5.5(연구 설계 작업자, 하위 에이전트), 2026-10-27 KST
- 상태: **설계 초안 · 미등록(DRAFT_UNREGISTERED) · execute false.** 작성 중 실자료 계산·GPU·ssh·기존 파일 수정 없음. Song 결정 D186(docs/decisions.md:325, 2026-10-27)으로 범위(B3 제외·가중치 진단은 S6 150경기 재사용·교차 적합 arm은 B0-XF·B1-XF만)·문턱·실행 창이 고정됐다(§14). 순서(§12 G1): 설계 검토 → 고정 커밋 코드 검토(코드 읽기만) → 등록(status REGISTERED, execute true, 등록 필드 채움) → 첫 등록 단계 smoke(TRAIN만, 지표 없음) → 본 실행. 검토는 작성자와 다른 모델.
- 개정(2026-10-27): 고정 커밋 코드 검토(docs/pm/research/track-a/reviews/CODE-REVIEW-A3-ASTRA-001.md) 반영 — S6 원장 판독을 '파싱 때 허용 목록 투영(`object_pairs_hook`), 금지 값은 보관·사용 안 함, 원시 바이트 해시는 별도'로 정정(§7.4, T10), 14,400초 상한이 시도 전체(smoke~decide, 명령 간 누적)에 걸림을 명시(§11).
- 개정(2026-10-27): 구현 검토(docs/pm/research/track-a/reviews/CODE-REVIEW-A3-ASTRA-002.md #14) 반영 — parquet 행 그룹을 min/max 날짜 통계로 고르고, 경계에 걸친 그룹은 디코드 뒤 범위 밖 행을 특징 구성 전에 버리며 그 수를 기록한다는 완화를 공개(§6), smoke의 'TRAIN 행만' 뜻을 명시(§12 G1).
- 개정(2026-10-27): 구현 검토 후속(docs/pm/research/track-a/reviews/CODE-REVIEW-A3-ASTRA-002.md, PM 결정) — 등록 전 입력 준비 단계(분할 정렬 재분할, 레포 밖 출력, sha256·식별 매니페스트 pin, 바이트 동일성 증명) 추가(§6, §12 G1 ②′, T15); 판독기는 분할 정렬이 아닌 입력을 거부하고 디코드 뒤 버리는 행 수는 기대 0.
- 개정(2026-10-27): PM read-only header check 2026-10-27 16:4x 반영 — 머리말 `runtime_sha256`은 런타임 자신의 식별(`af59057b…`, 등록 때 머리말에서 pin)이고 정책 식별자 `c13cc989…`는 머리말 `pins.candidate.sha256`에서 대조한다(§7.4, T9).
- 실험 ID: **EXP-BC-CAL-01**(D186에서 확정; docs/experiments.md·docs/decisions.md·pm-jobs 등록에 같은 ID 없음 확인)
- 짝 파일: [등록 초안 JSON](A3-behavior-policy-registration-draft.json)
- 근거: D184 트랙 A 설계·등록 초안 병행 (docs/decisions.md:321), D186 남은 결정 확정 (docs/decisions.md:325), 제안서 A3 행 (docs/pm/research/experiment-proposals-2026-10-26.md:26)

## 1. 쉬운 설명 — 무엇을, 왜

- **π̂_b(행동정책 추정)** = "이 투수는 2-1 카운트, 좌타자, 직전 공 슬라이더 뒤에 직구 55%, 슬라이더 25%, 체인지업 20%" 같은 확률표다. 스카우팅 리포트의 구종 비율표라고 보면 된다.
- **OPE 가중치(ρ)** = 실제로 던진 공 하나를 "추천 정책이 이 공을 고를 확률 ÷ 투수가 평소 이 공을 고를 확률(π̂_b)"로 다시 무게 매기는 것. 이 무게로 "추천대로 던졌다면"의 결과를 추정한다.
- **표가 틀리면:** 투수가 실제로는 자주 던지는 공을 표가 '드문 공'으로 적어 두면 그 공이 과하게 무거워지고, 반대면 가벼워진다. 저울이 한쪽으로 기운다.
- 지금 표(B0)는 투수·카운트·타자 좌우·직전 구종만 본다. 아웃·주자·점수·이닝·타순 회차·투구 수는 안 본다. MLB 구종 선택에는 음의 자기상관(같은 공을 연달아 덜 던짐)이 있다(Kovash & Levitt, NBER w15347, 초록 확인). 이력으로 어느 정도 예측된다는 뜻이다.
- **정확도가 아니라 '확률이 맞는지'가 중요하다.** 다음 공 1순위 맞히기는 이미 한계에 가깝다(Ahn 외 2026, arXiv 2602.07030 초록: 다음 공 약 64%). OPE는 "30%라던 공이 정말 30%로 나오는지"(보정, NLL)를 요구한다.
- Hanna·Niekum·Stone(ICML 2019, arXiv 1806.01347, 초록 확인): 행동정책을 자료로 추정해 쓰면 IS 평가 오차가 줄어드는 경우가 많다. 표를 잘 맞추는 것 자체가 저울을 고치는 수단이 될 수 있다.

## 2. 지금 상태 — 확인한 사실

### 2.1 B0 = 등록된 π̂_b

- 코드: `CategoricalBC` (experiments/pitchmdp/pitchmdp/rollout_policy.py:68–124)
  - 키 = (투수, 볼, 스트라이크, 타자 좌우, 직전 구종 또는 `<START>`) (:84–86)
  - 셀 빈도 + 20 × (그 투수의 TRAIN 구종 빈도, 지지 안에서 정규화)로 수축 (:110–122). prior 20·최소 1구 (configs/ML-POLICY-MATERIALIZATION-v1.json:311–312)
  - 지지 = 투수의 TRAIN 레퍼토리(1구 이상) (:103–108). TRAIN에 없는 투수는 리그 빈도로 대체하고 표시 (:104, :123–124)
  - TRAIN 전용: `fit_bc`는 2023-05-15..2025-04-30 행만 받는다 (experiments/pitchmdp/pitchmdp/matrix_policy.py:53–55). 직전 공은 구종만 본다 (`state_from_row(bc_only=True)`, :38–39)
- 등록 산출물 BC_P: 파일 sha256 `b4f274eb…` (configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:283). TRAIN 1,381,056행, 단일 손 등록 1,138명·모호 0 (docs/results-all.md:1473, D104 docs/decisions.md:154). 셀 81,132는 T7 `S1-materialize-a3/materialize.json`(TRAIN 전용 보고)에만 있다 — 레포 기록 요약 없음, 검토 범위에서 미검증
- 행동 공간 ARM-B = 18개 코드 `CH CS CU EP FA FC FF FO FS KC KN PO SC SI SL ST SV UN` (configs/ML-POLICY-MATERIALIZATION-v1.json:250)
- B0의 두 역할
  - **로깅 법칙(분모)** = BC_P 전체 어휘, 재정규화 없음 (D-1; prereg:307–309; experiments/pitchmdp/pitchmdp/policy_estimator.py:5–6)
  - **기준 정책 SupportedBC** = BC_P를 개입 지지 M에서 한 번 재정규화 (prereg:282; `MaskedReference`, experiments/pitchmdp/pitchmdp/policy_runtime.py:547–563). 후보 τ 0.1 P3 = M 위에서 π_ref·exp(Q_ref/τ) (prereg:297–303, 식별자 c13cc989)
- 소비처
  - ρ_t = π(a_t|H_t) / π̂_b(a_t|H_t), 클리핑·자기정규화 없음 (policy_estimator.py:3–8)
  - 후보 경로는 π̂_b가 맞아야 하고, 로깅 양성 위반 노드는 따로 보고 (:16–18). π̂_b 오지정은 별도 편향 경로 (:24)
  - 런타임은 로깅 법칙·지지·미지 투수 거부·양성 판정을 모두 같은 BC 객체에서 얻는다 (policy_runtime.py:391–412). 실제 구종의 로깅 확률이 0이면 `UNSUPPORTED_LOGGING_POSITIVITY` (:410)
  - τ 선택(S3b) 문턱: PA ESS 비율 ≥ 0.01, 경기 ESS ≥ 30, 후보/기준 ESS 비 ≥ 0.5, 잡음 q90 ≤ 0.5, 안전계수 2 (configs/ML-POLICY-MATERIALIZATION-v1.json:587–593; 규칙 experiments/pitchmdp/pitchmdp/policy_tau.py:75–90). 2026 판정 ESS 문턱 PA 100·경기 30 (prereg:519–521)

### 2.2 교차 적합은 지금 어떻게 하나

- **현행(pitchmdp 줄기): 교차 적합 없음.** π̂_b를 TRAIN(≤2025-04-30)으로만 맞추고 DEV·2026에 그대로 쓴다(시간 분리). 2026 로그로 맞추지 않는다, `fit_on_2026: false` (prereg:309). 2026 성가신 모수(nuisance) 교차 적합 예외는 꺼져 있다 (prereg:384–388).
- **옛 RE24 줄기:** 2026 로그에서 `game_pk % 5` 경기 폴드 교차 적합 (src/pitcheezy/ope/behavior.py:74–90, :103–131; configs/EXP-P0-002.yaml:30; D18 docs/decisions.md:23). 같은 자료로 맞추면 로그 행동 확률이 부풀어 가중치 평균이 무너졌다. 무너진 값은 D18·config 주석이 0.05(이론 1), 코드 주석(behavior.py:8)이 E[ρ]≈0.24로 **서로 다르게 적혀 있다.**
- 2026 로그 위 동결 π̂_b의 NLL·보정 진단은 "제안, 별도 등록·승인 필요" 상태다 (prereg:389–395). A3는 2026을 쓰지 않는다.

### 2.3 이미 측정된 숫자 (모두 인용, 이번에 새로 잰 것 없음)

- **B0의 전체 MLB 성능은 미측정.** 가장 가까운 측정은 P8의 BC(prior 20, TRAIN 1,252,824행 = BC_E와 같은 행 집합이라 등록 BC_P와 다름)를 Cpanel 투수에만 쓴 값이다:
  - DEV 13,246구(지원 라벨 13,187 / 미지원 59) 조건부 NLL **1.370029** vs 투수 빈도 1.441127
  - June 5,212구 1.353548 vs 1.456130 (docs/reports/ML-matrix-execution-2026-09-24.md:319–326)
  - 불펜 1.2505, 선발 1.4255 (:367–368). NLL은 지원 라벨 부분집합 기준 (experiments/pitchmdp/scripts/run_ml_policy.py:393–394)
- **B0에서는 기준 정책 가중치가 거의 1이다.** TRAIN 결정 가중 π̂_b의 M 위 질량 q01 0.99817·q05 0.99981·중앙 1.0, 질량 = 1인 결정 84.5% (T7 materialize.json `logging_mass_on_mask`에서 인용 — 레포 기록 요약 없음, 미검증). ρ_ref = 1/BC_P(M)이므로 거의 1 — 기준 정책이 분모와 같은 표에서 나왔기 때문이다. 표가 틀려도 이 구조에서는 기준 쪽 가중치가 그 오류를 드러내지 못한다.
- **S6 리허설**(DEV 150경기, 11,350PA; 식별자 c13cc989 재현본 S6-dr-a5)
  - 로깅 양성 해소 401PA, 자연 경과 55, 최악 잔여 6, 공유 끝 1 (results/ML-POLICY-VAL-v1-S6-v3.json:42 부근; D123 docs/decisions.md:192)
  - 전체 E0 ESS 후보 PA 9,120.22·경기 144.28, 기준 PA 9,725.69·경기 144.16 (results/ML-POLICY-VAL-v1-S6-v3.json:92–99)
  - 미지 투수로 시작 전 제외 1,219PA (results/ML-POLICY-VAL-v1-S6.json:12). S6-dr-a5가 층·ESS를 그대로 재현 (D125 docs/decisions.md:196)
- June S3b 후보 PA ESS 9,779 = 기준의 93.4%, 경기 147 (prereg:831). 후보–기준 평균 KL 0.0074 (prereg:491).
- **오지정 민감도(S5 합성 세계 V3):** 로깅 법칙을 빈도표로 바꾼 V3의 gap +0.00113 / +0.00027 / +0.00012 WE, tempered α 0.5는 +0.00030 / −0.00087 / −0.00054 (D111 docs/decisions.md:168). 세 값은 planning seed 0/1/2의 V3 gap이다. D111은 V3의 표준오차를 적지 않았다(적힌 se 0.00089/0.00080/0.00106은 V2 값). 그래서 이 문서는 V3 값의 유의성을 주장하지 않는다. 기록상 빈도표 V3의 seed 0 점 값 +0.00113이 MEI(0.001 WE/PA)를 넘고, 최대 |gap_V3(α 0.5)| 0.00087은 MEI 미만이라 D111이 오지정 경고를 내지 않았다.
- **외부 평가 R-3(옛 RE24 줄기):** π_b 자신의 1스텝 SNIPS +0.0203 vs 실측 평균 0.01399 (docs/reviews/external-evaluation-2026-10-19.md:117; 값 docs/results-p0.md:12, results/EXP-P1-010.json:155). 그런데 같은 장부는 "π_b 궤적 = 타석 평균 +0.0140(실제 평균과 일치), 1스텝은 결정 가중이라 값이 다름"이라고 적었다 (docs/experiments.md:73). → 이 차이가 π̂_b 오지정 때문인지 **가중 방식(투구 가중 vs 타석 가중) 차이 때문인지 분리되지 않았다.** A3는 같은 가중끼리 비교하는 보상 없는 검사로 다시 묻는다(§7.3).

## 3. 질문

**주 질문:** 더 잘 보정된 π̂_b(홀드아웃 NLL 낮음, 보정 같거나 좋음, 양성 문제 적음)가 OPE 가중치를 의미 있게 바꾸는가? 2027 프로토콜에 들어가기 전에 무엇으로 검증하는가?

- **Q1 표의 품질(판정 대상):** B1·B2가 DEV에서 B0보다 NLL이 낮고, ECE·"거의 0 확률" 문제가 나빠지지 않는가?
- **Q2 가중치가 얼마나 움직이나(서술):** 등록된 짝(후보 c13cc989 vs SupportedBC)의 S6 결정에서 분모만 바꾸면 ESS·가중치 분포·가중치 평균이 얼마나 달라지나?
- **Q3 자기 일관성(서술):** B0가 DEV에서 스스로 일관적인가(구종군 총량, 가중치 평균 = 1)? 새 후보가 그 어긋남을 줄이나?
- **묻지 않는 것:** OPE 추정값·Δ·판정이 바뀌는지(보상을 읽어야 함 → 범위 밖, §9 후속), 인과 식별, 위치(구종×구역) 성향.

**2027 전 검증 경로:** ① 합성 테스트 → ② 등록 B0를 원장에서 정확히 재현(무결성) → ③ DEV 1회 채점(Q1 판정) → ④ S6 가중치 진단(Q2·Q3) → ⑤ (선택, 별도 등록) 합성 세계에서 "참 로깅 법칙이 채택 후보일 때 B0를 쓰면 DR이 얼마나 치우치나"를 S5 V3 방식으로 측정(실보상 안 읽음) → ⑥ 2027 등록에서 채택 법칙을 1순위, B0를 민감도로 함께 보고.

## 4. 후보 (고정; 결과를 보고 늘리거나 바꾸지 않는다)

공통: 모든 후보는 **B0와 같은 지지**(투수의 TRAIN 레퍼토리 R_j)를 쓴다. R_j 밖 확률은 0, 안에서 합 1. 이유는 §7.2.

- **B0 — 필수, 기준.** 등록 BC_P 산출물을 sha 확인 뒤 그대로 쓴다(다시 맞추지 않음). OOF·교차 적합용 벡터화 재구현 `VectorBC`는 B0와 최대 차 ≤ 1e-12로 일치해야 한다.
- **B1 — 필수. 계층 백오프 계수표(CategoricalBC 확장).**
  - 사슬(순서 고정): ℓ0 리그 → ℓ1 투수 → ℓ2 ×타자 좌우 → ℓ3 ×카운트(12) → ℓ4 ×직전 구종(19 = 18코드 + `<START>`; = B0 키) → ℓ5 ×베이스-아웃(아웃 0/1/2 × 주자 {없음, 1루만, 득점권} = 9) → ℓ6 ×타순 회차(1, 2, 3+) → ℓ7 ×타자 군집(TRAIN 적합 5개 소프트 멤버십 중 최대 + 미상 = 6)
  - 수축 형태: p_ℓ(a | k_ℓ) = (n_ℓ(a, k_ℓ) + α_ℓ · p_{ℓ−1}(a | k_{ℓ−1})) / (n_ℓ(k_ℓ) + α_ℓ). 모든 층에서 R_j 밖 0, 안에서 재정규화
  - B0는 특수형이다: α1 = 0(리그 수축 없음), α2 = α3 = ∞(층 건너뜀), α4 = 20, α5..7 = ∞
  - α 격자 {0(ℓ1만), 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ∞} — **판단값:** 이 프로젝트 계수 모형의 α 선별 범위 5–500(docs/results-p0.md:27, 전이 모형)을 양끝으로 넓히고, ∞를 넣어 B0를 탐색 공간 안에 둔다
  - 맞추기: 시작점 = B0 설정. 조기 종료 분할 NLL로 사슬 순서대로 좌표 하강 2바퀴, 동률이면 큰 α(**판단값:** 2바퀴는 계산이 싸고 순서 의존을 줄이는 최소, 큰 α는 더 매끄러운 쪽). 시작점이 B0라 조기 종료 NLL은 B0 이하가 보장된다(DEV는 보장 없음)
  - 층 순서(**판단값**): 카운트·베이스-아웃이 구종 선택에 가장 크게, 타자 군집이 가장 작게 작용한다고 보고 깊은 쪽에 둔다. 표본이 없는 깊은 셀은 자동으로 부모로 돌아간다
- **B2 — 필수. LightGBM 18-class(투구 전 특징 63개, §5).**
  - 투수 정체: 범주형 ID는 쓰지 않고 **TRAIN 전용 OOF 표적 인코딩** = B0식 확률 18개(TRAIN 행: 다른 4개 TRAIN 경기 폴드로 맞춘 `VectorBC`; 그 밖 분할: 등록 BC_P) + log1p(투수 TRAIN 투구 수). **판단:** 1,138명 규모의 범주는 과적합·미지 투수 처리가 어렵고, B0 확률이 투수×카운트×좌우×직전 정보를 이미 수축된 형태로 담는다
  - 하이퍼파라미터(고정, HPO 없음, M4 선례 그대로; /Users/song/Projects/pitcheezy-pm-jobs/20261008-until-0830/multiclass-tree-run-launch-001/attempt-001/result.json의 params): multiclass, num_class 18, learning_rate 0.05, num_leaves 15, max_bin 63, min_data_in_leaf 100, lambda_l2 1.0, feature_fraction 1.0, bagging_fraction 1.0, deterministic, force_col_wise, num_threads 2, seed 20261043. 범주형 특징 = count_id, prev_type_1..5, prev_outcome_1..5
  - 라운드: 300라운드를 모두 학습하고 조기 종료 분할 NLL이 처음 최소가 되는 라운드를 쓴다. **판단값:** M4는 200라운드 중 130·192에서 최소였다(…/multiclass-tree-run-launch-001/REPORT.md:14–17) — 상한에 가까웠으므로 1.5배 여유
  - 예측 뒤 R_j 밖 0·재정규화 → §4.1 보정. bagging·feature 표본을 끈 결정론적 설정이라 seed는 결과를 바꾸지 않는다(단일 적합, seed 민감도는 재지 않음)
- **B3(소형 시퀀스 신경망, GPU) — 제외(D186 5-①).** arm이 아니고 이 등록에서 실행·seed·비용·GPU가 없다. 다시 넣으려면 새 등록 ID가 필요하다(지금의 CUDA 수출본은 구종 표적 누수가 있어 새 수출본도 필요).
- **서술용 팔(채택 후보 아님):** B0-T(B0 + 온도 1개, "보정만으로 얼마나" 분해) · B1·B2 보정 전 확률 · B0-XF·B1-XF(필수 서술; 교차 적합 arm은 이 둘만 — D186 5-①, B2-XF는 돌리지 않음)(§8) · 레퍼토리 밖 질량 분석(모형 없음, §7.3)

### 4.1 공통 보정 절차 (B1·B2, 서술용 B0-T)

- 온도 분할(2025-05-16..31)의 S_in 행으로 (a) 온도 T(경계 [0.5, 2.5], run_cuda_direct.py `BOUNDS` :180 선례) (b) 벡터 스케일링 p ∝ exp(a_c·log p_c + b_c), a_c > 0, L2 벌점 1e-3·Σ((a_c−1)² + b_c²)(**판단값:** 희귀 구종 계수 안정)를 맞춘다. B0-T는 온도만.
- 선택: 온도 분할 안 5겹 경기 그룹 CV 평균 NLL. 벡터가 온도보다 0.0005 이상 낮을 때만 벡터(**판단값:** 36개 매개변수를 거의 0인 이득에 쓰지 않는다; WS 줄기 결과 모형에서는 벡터+Platt가 후보가 못 됐다, docs/experiments.md:263). 동률이면 온도. 선택된 방식을 온도 분할 전체로 다시 맞춘다. 벡터 최적화가 실패하면(수렴 실패·비유한 값) 온도로 대체하고 기록한다. 온도 최적화도 실패하면 그 후보는 `CALIBRATION_FAILED`로 기록하고 채택 자격이 없다(보정 전 확률은 서술로만 남는다). B0-T의 온도 실패는 B0-T 서술 칸을 비운다.
- 구현은 experiments/ws_seq/calibrate_logits.py의 `vector_objective`·`fit_vector`·`cv_folds`(:238, :303, :324) 수식을 새 모듈로 옮긴다(원 파일은 수정하지 않음).
- 비교 기준: CalArena(Berta·Holzmüller·Bach·Jordan, arXiv 2605.30188, 초록 확인)는 보정 방법을 ECE 추정치보다 고유 점수 규칙(proper scoring rule) 개선으로 비교하자고 제안한다 → 주 지표 NLL, ECE는 가드. 구조화 행렬 스케일링(arXiv 2511.03685)은 초록에서 "다중 클래스 1위"를 확인하지 못했고, 온도 분할 약 2주치로 18×18 행렬을 맞추기엔 매개변수가 많아 이번 후보에서 뺀다(판단).

## 5. 특징 — B2 63개 정확한 목록

모든 값은 그 투구를 던지기 전에 알 수 있는 것만 쓴다. 쓰는 것은 세 가지뿐이다: 같은 PA·같은 경기의 **이전 행**, 동결 TRAIN 산출물, 그리고 등록된 A3 타자 성향 이력(§5 E) — TRAIN 행은 필터 안의 strictly-prior-date 이력, earlystop은 창 시작 2025-05-01 스냅샷, temperature는 2025-05-16 스냅샷, DEV는 2025-07-01 스냅샷이며, 모두 TRAIN·earlystop·temperature 행(2023-05-15..2025-05-31, June·DEV·2023-03-30..05-14·2026 제외)만으로 만든다.

- **A. 투수 인코딩(TRAIN 전용, TRAIN 행은 OOF)**
  - 1–18 `b0_p_{CH…UN}`: B0식 확률 18개(§4 B2)
  - 19 `pitcher_train_log_volume` = log1p(투수 TRAIN 투구 수) (라벨과 무관하므로 전체 TRAIN 수)
- **B. 카운트·PA 안 이력**(`MatrixHistoryStore`의 이전 행, 현재 행 제외)
  - 20 `balls` · 21 `strikes` · 22 `count_id`(12 범주) · 23 `pa_pitch_index`(PA 안 이전 투구 수)
  - 24–28 `prev_type_1..5`(18코드 + UNKNOWN + NONE)
  - 29–33 `prev_outcome_1..5`(10결과 + unknown + NONE; 이전 공은 대부분 ball/strike/foul)
- **C. 경기 상황**(G0 맥락 기본 채널, experiments/pitchmdp/pitchmdp/matrix_bridge.py:37–38)
  - 34 `outs` · 35–37 `first_base`·`second_base`·`third_base` · 38 `inning`(10 이상은 10) · 39 `defensive_score_lead` · 40 `defender_is_home`
- **D. 손**
  - 41 `batter_left` · 42 `pitcher_left` · 43 `same_hand`(플래툰)
- **E. 타자 성향**(G0 맥락 [11:28] 17채널, matrix_bridge.py:20–23). **입력 행 필터(고정):** 2023-05-15 ≤ game_date이고 June(2025-06-01..30)·DEV·2026이 아닌 행 — 즉 TRAIN·earlystop·temperature 행만. 결과에서 나온 이력 열(`events`·`description`·`is_pa_terminal`·`launch_angle`, experiments/pitchmdp/pitchmdp/policy_requests.py:284)도 이 필터 안의 이전 날짜 행에서만 읽는다. TRAIN 행은 필터 안의 strictly-prior-date 이력(`archetypes.add_batter_style_history`), 다른 분할은 창 시작 스냅샷(earlystop 2025-05-01, temperature 2025-05-16, DEV 2025-07-01)을 필터한 프레임으로 `style_snapshot`(policy_requests.py:277–298)에서 만든다. 이 A3 스냅샷은 런타임 S1b/D-7 산출물(configs/ML-POLICY-MATERIALIZATION-v1.json:506–514 — 각 as-of 이전의 모든 행을 읽어 June·2023-03-30..05-14가 들어감)과 다른 A3 전용 산출물이며, sha256를 freeze-primary 매니페스트(§10)에 넣는다
  - 44–49 성향 비율 6 · 50–55 신뢰도 6 · 56–60 소프트 멤버십 5
- **F. 작업량**(같은 경기의 이전 기록만; `workload_features.rebuild`, 브랜치 research/direct-head-confirm-1026의 experiments/pitchmdp/pitchmdp/workload_features.py:33–45 이식)
  - 61 `prior_pitches_game`(이 투수가 이 경기에서 앞서 남긴 Statcast 기록 수) · 62 `times_through_order`(재구성 타순 회차, :41) · 63 `is_game_starter`(그 경기 수비 팀의 첫 투수, D87 역할)

**B1 키 추가분:** 아웃·주자 3군(C), 타순 회차 1/2/3+(F62), 타자 군집 = E 소프트 멤버십의 최대(신뢰도 0이면 미상).

**금지(누수):**
- 현재 투구의 구종·물리(`release_speed`, `release_spin_rate`, `spin_axis`, `pfx_x/z`, `plate_x/z`, `effective_speed`), `sz_top/bot`, 현재 `description`·`events`·`type`, `post_*` 점수, `launch_*` (prereg:408 `current_pitch_physics_or_outcome_as_input: false`와 같은 원칙)
- 이후 행, PA 전체를 보고 정하는 `supported_pa`(matrix_features.py:27), `*_days_until_next_game`(workload_features.py:16), Savant `n_thruorder_pitcher`·`pitcher_days_since_prev_game` 원값(재구성 값만 씀)
- `game_date`·월·시즌, 원 투수·타자 ID(TRAIN 인코딩으로만)
- 전체 TRAIN으로 만든 투수 구종 비율(G0 맥락 투수 프로필 23채널)을 TRAIN 행에 그대로 쓰는 것(표적 누수)
- **이전 공의 물리·위치도 이번엔 뺀다(판단):** 2026부터 `plate_x/z` 정의가 바뀌어(제안서 A4 행, experiment-proposals-2026-10-26.md:27) 2027 재적합 때 시즌 간 정의가 섞인다. A4 정합 감사 뒤 따로 본다

## 6. 분할·모집단·자료 역할

- 분할(`assign_fold`, experiments/pitchmdp/scripts/run_temporal_blend.py:35–48)
  - TRAIN 2023-05-15..2025-04-30 — 적합(B1 계수, B2 학습, OOF 인코딩)
  - earlystop 2025-05-01..15 — B1 α, B2 라운드 선택
  - temperature 2025-05-16..31 — 보정
  - blend 2025-06 — **안 씀**(타자 성향 이력 입력에도 넣지 않음, §5 E)
  - DEV 2025-07-01..09-30 — 1회 채점(노출된 개발 자료)
  - 2026 — 0행. 2023-03-30..05-14(노출 상태 unknown)도 안 씀 — 이력 입력 포함 (configs/ML-POLICY-MATERIALIZATION-v1.json:433–438)
- 모집단 P(OPE에 쓰이는 결정): 정규시즌, 18코드 라벨(자동 볼·스트라이크·라벨 없음 제외; `bc_population_mask`, experiments/pitchmdp/pitchmdp/policy_requests.py:63–68과 같은 규칙), 합법 카운트, TRAIN 단일 손 등록 투수이고 그 행의 투수 손이 등록 손과 같음(D-8, policy_runtime.py:327–338 `_hand`: 모호·결측·불일치는 P 밖), 타자 L/R
  - **S_in** = P ∩ {실제 구종 ∈ R_j} — 주 지표 계산 대상
  - **S_off** = P ∩ {실제 구종 ∉ R_j} — 구조적 양성 위반(새 구종). 모든 후보 공통, 보고만
  - **P⁺(서술용 부모 집단)** = P에서 'TRAIN 단일 손 등록 투수·손 일치' 조건만 뺀 집합. **TRAIN0(미지 투수)** = P⁺ ∩ {투수 TRAIN 행 없음} — volume_zero 서술 층은 P⁺에서만 낸다(P 안에서는 비어 있다). TRAIN0에는 R_j가 없으므로 모든 모형이 B0의 대체 지지(리그 TRAIN 지지, rollout_policy.py:103–105)를 쓰고, 판정에 쓰지 않는다. OPE 런타임은 이들을 거부한다 (policy_runtime.py:391–392). 등록됐지만 손이 모호·결측·불일치인 행은 P⁺ \ P 개수로만 보고
- 행 수는 **미측정**. census 단계는 구종 라벨을 읽지 않고 투구 전 필드(정규시즌·합법 카운트·투수 TRAIN 등록·손 일치·타자 L/R)로 셀 수 있는 것만 센다. 라벨이 필요한 수(18코드 라벨로 정하는 P, S_in, S_off)는 라벨을 읽어도 되는 단계에서 낸다 — TRAIN·earlystop·temperature는 fit·calibrate, DEV는 score-dev(§10 순서). 참고: 결과 모형의 DEV 전체 MLB는 311,721구 (docs/reviews/external-evaluation-2026-10-19.md:103)
- 자료 버전: processed sha256 `9f65d1db…`(configs/ML-POLICY-MATERIALIZATION-v1.json `train_bc_plan.data`), raw statcast 2023–2025 고정 해시(run_ml_direct_workload.py `FROZEN_RAW_SHA256`, 같은 브랜치), G0 번들 `c839dbf7…`
- 2026 방어: 로더가 2025-09-30 이후 날짜를 거부한다. 타자 성향 코드도 2026 행을 거부한다 (experiments/pitchmdp/pitchmdp/archetypes.py:36)
- **입력 준비(등록 전 1회, PM 결정·구현 검토 CODE-REVIEW-A3-ASTRA-002 후속):** 고정된 processed parquet(`9f65d1db…`)의 행 그룹은 분할 경계와 맞지 않을 수 있어, 등록 전에 결정론적 재분할을 한 번 한다 — pyarrow로 행·순서·스키마를 그대로 두고 행 그룹 경계만 정확히 2023-05-15(TRAIN 시작)·2025-05-01·2025-05-16·2025-06-01·2025-07-01·2025-09-30 끝에 맞춘다(원본 행이 날짜 순이 아니면 준비가 멈춘다). 결과는 레포 밖에 쓰고 그 sha256와 식별 매니페스트 sha256를 A3 processed 입력으로 pin한다. 원본과의 행 단위 동일성은 모든 열의 바이트 해시로 증명한다 — 이 해시는 `pitch_type`을 포함한 모든 열의 바이트를 흘려 읽는 **바이트 동일성 검사이지 라벨 열람이 아니며**, 어떤 라벨 값도 계산에 들어가지 않는다. PM이 코드 검토 PASS 뒤, 06:30–13:30 KST 창 밖에서 한 번 실행하고 그 기록을 등록 블록 부록에 남긴다(§12 G1).
- **행 그룹 선택(구현 검토 CODE-REVIEW-A3-ASTRA-002 #14):** 판독기는 준비된 입력의 parquet 행 그룹을 footer의 min/max 날짜 통계로 고르고, 범위 밖 그룹은 디코드하지 않는다. 분할 경계에 걸친 행 그룹이 footer 통계에 보이면(분할 정렬이 아닌 입력) 판독기가 그 입력을 거부한다. 그래서 등록 입력에서는 디코드했다가 버리는 행이 없어야 하며, 그 수(기대 0)를 단계 기록과 읽기 로그에 남긴다

## 7. 평가 — 결과 보기 전에 고정

**DEV 지표(이 문서의 정의):** §7.1~§7.4가 DEV·S6 행에서 내는 모든 값 — 후보·서술 arm의 NLL·Brier·ECE·P1·정확도·층별 값, SC-1~SC-3, 새 π̂_b 분모로 계산하는 가중치 진단과 E5. 예외: §7.4의 등록된 고정 B0 무결성 검사(S6 식별 대조·R1b·R2)는 DEV 지표가 아니며 `s6-integrity` 단계에서 먼저 돈다.

### 7.1 확률 품질 (DEV S_in, 투구 가중)

- **NLL**(자연로그), **Brier**(18-class 합), **top-label ECE10**(10칸 동일 폭; experiments/pitchmdp/pitchmdp/matrix_metrics.py:33–51 규약). 그 함수는 10-class 고정이라(:11, :25) 같은 정의의 18-class 새 함수를 쓴다. classwise ECE·top-1 정확도는 서술(판정 아님).
- B0 대비 짝 차이: 경기 단위 부트스트랩 10,000회, seed 20260924, 단측 p는 null 중심화(matrix_metrics.py:77–121 관례). ECE 차이 CI는 경기별 bin 충분통계(개수·확신 합·정답 합)로 매 복제마다 정확히 다시 계산.
- 층(서술): 투수 TRAIN volume low/middle/high(P 위)와 zero(P⁺의 TRAIN0, §6)(투수 TRAIN 투구 수의 q25·q75, experiments/pitchmdp/pitchmdp/matrix_panel.py:91–94, :149), 카운트 12, 역할 SP/RP(D87), 월, 2스트라이크, 주자 유무, 플래툰.

### 7.2 양성(positivity)

- **구조적:** S_off 비율(전체·volume별). 모든 후보가 같은 지지를 쓰므로 후보 간 같다 — 보고만.
- **모형별 "거의 0"(P1):** S_in에서 π̂_b(a_t | H_t) < 1e-3인 비율. **판단값 1e-3:** 0.1% 미만이면 그 공 하나가 6.9 nat 이상의 손실이고, ρ가 추천 확률의 1,000배 이상이 될 수 있다. P2(< 1e-2)는 서술.
- 구조적 위반 수를 판정에 쓰지 않는 이유: 확률 바닥값이나 전체 어휘 softmax는 0을 없애 위반이 준 것처럼 보이게 하지만 식별을 만들지 않는다(prereg:401 `probability_floor_creates_positivity: false`). 지지를 고정하면 런타임의 L1-R 회계(`rho_zero` 해소, `validity_violation` 수)도 후보 간 비교할 수 있다.

### 7.3 자기 일관성 (보상 없음)

- **SC-1 구종군 총량:** P 전체(S_off 포함)에서 "관측 비율 − 예측 비율". 4군 {직구 FF·SI·FC·FA / 브레이킹 SL·ST·CU·KC·SV·CS / 오프스피드 CH·FS·FO·SC / 기타 KN·EP·PO·UN}과 코드별, 전체·volume·카운트. "π̂_b가 말하는 로깅 정책 자신의 값(행동만으로 정한 가짜 보상) vs 경험 평균"이다. R-3의 질문을 보상 없이, 같은 투구 가중끼리 묻는다.
- **SC-2 가중치 평균 = 1:** S6 E0에서 등록 짝(후보, 기준)의 PA 끝 가중치 W_T(§7.4) 평균과 결정별 ρ 평균. 맞는 π̂_b라면 기댓값은 1이다(옛 줄기 D18이 쓴 진단과 같음). 경기 부트스트랩 CI.
- **SC-3 작은 확률 관측/예측:** 모든 (결정, 구종) 쌍을 예측 확률 구간 [0, 1e-3), [1e-3, 1e-2), [1e-2, 0.05), [0.05, 0.2), [0.2, 0.5), [0.5, 1]로 묶어 관측/예측 비를 낸다. **판단값 구간:** ρ를 크게 만드는 작은 확률 쪽을 촘촘히 본다.
- 해석 도움(방향만, 크기 미측정): 지지를 TRAIN 레퍼토리로 고정하면 새 구종(S_off) 몫만큼 M 안 확률이 과대 → ρ가 체계적으로 작아져 SC-2 평균이 1 아래로 기울 것으로 예상한다.
- 보상을 쓰는 원래 형태(π̂_b IS로 본 기준 정책 가치 vs 관측 평균)는 **하지 않는다**(보상 열람 금지 게이트).

### 7.4 OPE 가중치 진단 (S6 리허설; E5만 판정, 나머지 서술)

- **입력:** `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/ML-POLICY-VAL-v1/S6-dr-a5/ledger-dev-0.jsonl`을 **읽기 전용**으로 쓴다(파일 크기는 레포 기록이 없어 미측정).
  - **식별은 등록에 pin한 해시로 한다.** 원장 파일과 같은 폴더 `manifest.json`의 sha256를 등록 때 pin하고(등록자가 봉인 폴더에서 해시만 계산해 매니페스트 자체 기록과 대조), 읽기 전후에 둘 다 다시 대조한다. 두 식별자를 구분한다(PM 머리말 읽기 확인 2026-10-27 16:4x): (a) **정책 식별자** — 원장 머리말 `pins.candidate.sha256`이 기대값 `c13cc98994ae9eb08a8f939a583ebe712312594ac25c05314a9ca790bd46f73c`(D125 최종 식별자; configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:303 `final_identity_sha256`, results/ML-POLICY-VAL-v1-S6-v3.json:111)와 같아야 한다. (b) **런타임 식별자** — 머리말 `runtime_sha256`은 PolicyRuntime 자신의 식별(자기 pins의 canonical hash, policy_runtime.py:290–291; 행마다 :489)이며 정책 식별자가 아니다. 그 전체 값(PM 확인 앞자리 `af59057b…`)을 등록 때 머리말에서 pin하고, 모든 결정 행의 `runtime_sha256`이 pin한 머리말 값과 같아야 한다. 하나라도 다르면 FAILED_INTEGRITY. 봉인 dr.json의 pin(addendum-8-d127, `5e56119d…`)은 참고로만 적고 그 파일은 열지도 해시하지도 않는다.
  - 허용 필드만 읽는다: 머리말의 `runtime_sha256`·`pins`(식별 메타데이터만), 결정 행의 `kind, request_id, runtime_sha256, pa_id, decision_index, status, pitcher, pitcher_hand, batter_side, balls, strikes, history_actions, logged_action, context_key, result.{mask, logging, reference, candidate, logged_index, rho_reference, rho_candidate}` (원장 행 필드 정의: policy_runtime.py:487–491)
  - **읽는 방식(코드 검토 반영, 구현 4a187af1):** 원장 각 줄을 키 단위 스캐너로 읽어 파싱 때 허용 목록으로 투영한다 — 허용 목록 밖 키의 값은 디코드하지 않고 건너뛴다(`object_pairs_hook`보다 엄격). `q_*`·`mc_se`·보상·값 같은 금지 필드의 값은 객체로 만들어지거나 보관·사용되지 않는다. 원장·매니페스트 sha256는 원시 바이트를 따로 해시하는 승인된 별도 작업이다. 같은 폴더의 `pa_values.parquet`·`dr.json`은 **열지 않는다**(판독기가 경로를 거부). 투영된 레코드에 허용 목록 밖 키가 하나라도 있으면 FAILED_INTEGRITY
- **E0 재구성:** 각 PA 첫 행의 투구 전 필드와 동결 TRAIN 산출물(BC_P, 지지표, 손 등록)만으로, `PolicyRuntime.start_population` 규칙 그대로 (policy_runtime.py:340–360).
- **무결성 R1b:** 벡터화 B0가 원장의 `logging` 벡터를 최대 차 ≤ 1e-12로 재현한다.
- **무결성 R2:** 원장으로 다시 계산한 B0 가중치의 전체 E0 ESS가 S6-v3 `ess_all_e0`(후보 PA 9,120.218896471748·경기 144.2828223791589, 기준 PA 9,725.694335061527·경기 144.16008032451504)와 상대 1e-9 안에서 같다. R1b·R2 중 하나라도 어긋나면 FAILED_INTEGRITY, DEV 채점 없음(식별 대조·R1b·R2는 xf-fit·score-dev 전 `s6-integrity` 단계에서 돈다, §10).
- **새 π̂_b 가중치:** SUPPORTED 결정에서 ρ_t = π(a_t) / π̂_b^new(a_t). OUTSIDE_POLICY_SUPPORT·LOGGING_POSITIVITY 행은 π(a_t) = 0이라 어떤 분모든 ρ_t = 0. **잔차 가중치** W_t = Π_{s≤t} ρ_s(현재 행동까지의 곱 — policy_estimator.py:142–157의 `affine` 재귀를 펼친 것). **PA 끝 가중치** W_end = `affine`의 slope(`weight_<정책>`): 완료 타석은 마지막 결정까지의 곱, ρ = 0이 나오면 그 지점에서 0으로 멈춘 접두 곱. 노드 k에서 양성 외 사유로 거절·검열된 타석은 노드 k 앞까지의 곱 Π_{s<k} ρ_s를 쓰고(policy_estimator.py:259), 양성 위반(POSITIVITY) 거절은 노드 k를 ρ_k = 0으로 포함해 0으로 닫는다(:256–258) — ESS·PA 가중치 분위·SC-2는 W_T를 쓴다(L1-R과 같은 규칙, :138–157, :255–270).
- **보고:** 두 정책의 PA·경기 ESS(전체 E0), 후보/기준 ESS 비(S3b 문턱 0.5 참고), 결정별 ρ 분위(q50·q90·q99·최대)와 ρ > 10 비율, PA 끝 가중치 W_T 분위와 W_T > 10 비율(**판단값 10:** 한 공·한 타석이 열 개처럼 세지는 지점), SC-2, 결정 평균 KL(π̂_b^new ‖ B0), **가중치 이동 D** = 타석 평균 Σ_t |W_t^new − W_t^B0|(W_t는 위 잔차 가중치; 정책별·짝 합).
- **D의 뜻:** DR 타석 값은 V_0 = v_0 + Σ_t W_t (y_t − q_t(a_t))로 펼쳐진다(W_t는 현재 행동까지의 곱; y_t = 다음 결정의 v 또는 마지막 단계의 끝 값; v·q·y가 모두 [0, 1]이라 |y_t − q_t| ≤ 1). v_0와 q는 목표 정책·동결 Q로 정해져 분모를 바꿔도 같으므로, 분모 교체로 생기는 타석 값 변화는 Σ_t |ΔW_t| 이하다(보수적 상한; 실제 잔차 크기는 보상을 안 읽으므로 미측정). 같은 상한이 L1-R 경계의 두 끝점 각각에도 적용된다.
  - D(짝 합) < MEI 0.001 → "이 S6 경기들의 **고정된 경험 평균 대비**(Δ 점 값과 각 경계 끝점의 평균)가 분모 교체로 0.001 WE/타석 미만 바뀐다"까지만 말한다(위 유계 잔차 가정 아래). 부트스트랩 CI 기반 판정이나 ESS 게이트가 그대로라는 보장은 아니다(둘은 가중치 분포 전체에 달림, configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:800–828)
  - D ≥ MEI → "바뀔 수 있다"까지만 말한다. 실제 편향 크기는 (π̂_b 오차) × (Q 오차)의 곱이라 보상 없이는 모른다
- **E5 사용성(판정):** 새 π̂_b에서 두 정책 ESS ≥ PA 100·경기 30(prereg:519–521). 못 넘으면 채택 불가.
- **S6 경기 재사용 공개:** EXP-STAB-01은 완료된 ≤2025 OPE 표본 재사용을 피하려고 S6 150경기를 뺐다(/Users/song/Projects/pitcheezy-pm-jobs/20261020-stab01/REGISTRATION.json `data.why_allowed`). A3는 등록 짝의 가중치를 정확히 재현해야 해서 S6를 쓰되, 확률 필드만 읽고 보상과 결합하지 않으며 S6 결과는 그대로 둔다(Song 결정 D186 5-①: 가중치 진단 표본 = S6 150경기 재사용, §14-2).

## 8. 교차 적합 규칙

- **원칙:** π̂_b를 그것이 평가하는 로그(같은 창)로 맞추면 반드시 경기 폴드 교차 적합을 하고 out-of-fold 확률만 쓴다.
- **폴드:** K = 5(**판단값:** D18 선례, 각 적합이 창의 80%를 씀). fold(g) = int(sha256("EXP-BC-CAL-01/xf:" + str(game_pk)).hexdigest(), 16) mod 5. `game_pk % K` 대신 소금 해시를 쓴다(경기 번호 순서와 무관; M-1 표본 선택과 같은 방식). 한 경기의 모든 투구는 같은 폴드. 폴드별 재조정 없음(α·라운드·보정은 TRAIN-only 적합값 고정).
- **TRAIN 안 OOF 표적 인코딩**(B2): 같은 함수, 소금 "EXP-BC-CAL-01/te".
- **A3 주 평가는 교차 적합이 필요 없다.** TRAIN으로 맞추고 DEV에서 평가해 겹침이 없다. 2027에 "작년까지로 맞춰 올해에 쓰는" 시간 이동과 같은 꼴이다.
- **서술 팔 XF(D186 5-①: B0-XF·B1-XF만, B2-XF 없음):** B0-XF·B1-XF = TRAIN ∪ (DEV 다른 4폴드)로 맞춰 DEV 폴드 k를 예측한다(지지도 그 자료로 다시 정의). 적합은 primary 동결 뒤의 별도 단계 `xf-fit`(§10)에서 한다 — 레시피(폴드 소금·α·지지 규칙)는 등록 때 고정, 결정론적, α는 TRAIN-only 적합값 그대로, 폴드별 조정 없음. 그 산출물의 freeze-xf 매니페스트를 pin하기 전에는 DEV 지표(§7 정의)를 계산하지 않는다. **비교 규칙:** XF와 B0의 짝 NLL은 **원래 B0의 S_in**(주 비교와 같은 행) 위에서만 낸다. XF가 지지를 넓혀 확률을 주는 S_off 행('구조된' 새 구종)은 따로 — 구조 비율과 그 행의 XF NLL만 보고한다(B0는 그 행에서 확률 0이라 짝 비교가 없다). 이 결과는 2027 등록이 교차 적합 예외(prereg:384–388)를 요청할지의 근거 자료다. DEV 라벨(구종만)을 적합에 쓴 사실을 노출 장부에 남긴다. 미래 경기 사용(as-of 위반)은 OPE 성가신 모수 교차 적합에서 허용되는 형태이며 서비스에는 쓰지 않는다.

## 9. 판정 규칙 (문턱은 모두 판단값; NLL 폭 0.005는 D186 4-① 승인)

- 후보 집합 C = {B1, B2}(B3는 D186 5-①로 제외)
- **채택 자격(모두 만족)**
  - **E1 NLL 우위:** ΔNLL = NLL_c − NLL_B0 ≤ −0.005, CI95 상한 < 0, 단측 p의 Holm 보정(실행한 후보 수 = B1·B2의 2) ≤ 0.05. **판단값 0.005(Song이 D186 4-①로 초안 값 그대로 승인, DEV 지표 계산 전 고정):** 측정된 BC−빈도 격차(Cpanel DEV −0.071, docs/reports/ML-matrix-execution-2026-09-24.md:324–326)의 약 7%. 새 성가신 모수를 등록할 만한 최소 개선으로 본다
  - **E2 ECE 비열등:** ΔECE(top-label ECE10) CI95 상한 ≤ +0.002. **판단값:** WS-CAL-001에서 80,584행 ECE 차이의 CI 반폭이 0.0016–0.0029였다(C1 0.0021994, C2 0.0028627, C3 0.00160005 — docs/experiments.md:263의 구간 끝점에서 산술). DEV는 더 크므로 이만큼 나빠지는 것까지만 "같다"로 본다
  - **E3 양성 비열등:** ΔP1 CI95 상한 ≤ +0.0005. **판단값:** "거의 0" 결정이 2,000개당 1개보다 많이 늘면 가중치 꼬리가 길어질 수 있어 허용하지 않는다
  - **E4 무결성:** R1(BC_P 파일 sha `b4f274eb…`), R1b, R2, 누수·합성 테스트 전부 통과, 2026 0행
  - **E5 사용성:** §7.4
  - 보정 실패 후보(`CALIBRATION_FAILED`, §4.1)는 E1~E5와 상관없이 자격이 없다
- **선택:** 자격 후보 중 DEV NLL 점추정 최저. 두 후보가 모두 자격이 있고 DEV NLL 점추정 차이가 0.001 미만(< 0.001, 엄격 부등호; **판단값:** 실익 없는 복잡도 회피)이면 단순한 B1
- **라벨:** `ADOPT_FOR_2027_REGISTRATION:<후보>` / `KEEP_B0`(자격 후보 없음) / `FAILED_INTEGRITY` / `NOT_EVALUABLE`
- 그 밖(층별, SC-1~3, S6 가중치 이동, XF, B0-T, 보정 전 확률, 정확도)은 모두 서술이다.
- **채택이 뜻하는 것**
  1. 2027 전향 OPE용 **새 등록**에서 이 레시피(코드 커밋·config·보정 규칙)를 로깅 법칙 1순위로 제안한다. 2027 as-of 재적합 규칙과 교차 적합 여부는 그 등록에서 정하고 독립 검토를 받는다.
  2. B0는 그 등록의 로깅 법칙 민감도로 함께 보고한다.
  - **A2 R3에 넘기는 산출물:** 채택 후보의 primary 동결 산출물만(XF 산출물은 아님). A2의 현재 자격 규칙(A2 설계 §3 R3, JSON `arms.R3_behavior_policy.eligibility`)은 A3 라벨 `ADOPT_FOR_2027_REGISTRATION:<후보>`, sha256 pin·`BehaviorModel` 규약·A3 독립 검토 PASS, 그리고 적합·레시피 안 선택(B1 α·B2 라운드)·보정·특징의 동결 입력이 모두 2025-05-31 이전 행(TRAIN·earlystop·temperature)뿐이고 June·DEV/S6·2026 행이 없음을 산출물 출처 매니페스트로 확인하는 것이다. A3 primary 산출물은 이 조건대로 만들어지고(§5 E 입력 필터, §6 분할), freeze-primary 매니페스트가 각 산출물의 입력 행 날짜 범위를 담아 그 확인을 가능하게 한다(§10). A2에서 R3는 진단 전용이다.
  3. 런타임이 로깅 모형을 BC와 따로 받으려면 새 코드가 필요하다. 소스 import closure가 정책 식별자에 들어가므로(D122, docs/decisions.md:190) 새 식별자로 재인증한다. 기준 정책을 SupportedBC(B0)로 둘지 새 표 기반으로 바꿀지도 그 등록의 설계 문제다(바꾸면 후보 정책 자체가 바뀐다).
  4. (선택, 별도 등록) 합성 세계 점검: S5 V3 틀(configs/ML-POLICY-MATERIALIZATION-v1.json:608–628, D111)에 "로깅 법칙 = 채택 후보"를 넣어 B0를 썼을 때의 DR gap을 MEI와 비교한다. 실보상을 읽지 않는다. 비용 선례 S5-v2-a1 2,900초(D111), A3용 비용은 미측정.
  - **바뀌지 않는 것:** c13cc989, D108/D109 문턱, D131 판정(NO_EVIDENCE_OF_IMPROVEMENT), D155 닫힘, 봉인 S5/S6 결과, BC_P 산출물, 서비스·데모·G0·G15. 이 등록 아래에서 2026 OPE나 ≤2025 S6를 새 π̂_b로 다시 계산하지 않는다.

## 10. 구현 계획 (후속; 별도 worktree, 새 파일만 추가, 기존 모듈 수정 없음)

- 브랜치 제안: `research/a3-behavior-model`(worktree 분리; #37 CLAIM은 PM)
- **experiments/pitchmdp/pitchmdp/behavior_model.py**
  - `BehaviorModel`(Protocol): `actions`(18코드, config의 token_vocabulary에서 읽음), `support(state) -> bool[18]`, `probabilities(state) -> float[18]`(지지 밖 0, 합 1), `fallback(state) -> bool`, `batch(table, rows) -> float[n,18]`(같은 법칙), `identity() -> dict`(매개변수·산출물·코드 sha256). `CategoricalBC`는 support·probabilities·fallback을 이미 같은 이름으로 가지므로 얇은 감싸개로 맞춘다
  - `RegisteredBC`: `policy_artifacts.load_train_bc(path, expected_sha256)`(experiments/pitchmdp/pitchmdp/policy_artifacts.py:290)로 BC_P를 sha 확인 뒤 감싼다(B0)
  - `VectorBC`: B0 레시피의 groupby 벡터화(OOF·XF용), `CategoricalBC`와 일치 테스트
  - `BackoffBC`(B1): 층 사슬·α·`tune(early_rows, grid)`
  - `LGBMBehavior`(B2): 고정 params, 지지 마스크·재정규화
  - `Calibrated`: 어떤 모형이든 온도·벡터 보정을 씌움
  - 투구 전 맥락 표는 `PAState.context_key`(game:ab:pitch)로 찾는다. PAState는 카운트·이력만 들고, 경기 상황·타자·작업량은 표에서 가져온다(rollout_policy.py:41–56의 "context_key = 투구 전 동결 맥락" 개념)
- **behavior_features.py:** `population_masks`(P, S_in, S_off, 분할), `history_features`(`MatrixHistoryStore`의 이전 행만, 현재 토큰 버림), `context_features`(동결 G0 맥락 인코더로 28채널; 타자 성향 입력은 §5 E의 A3 스냅샷이며 S1b 산출물이 아님), `style_inputs`(§5 E 입력 행 필터), `workload_features`(rebuild 이식 + `is_game_starter`), `oof_b0`, `game_fold(game_pk, salt, k)`, `assert_pre_pitch(columns)`(허용 목록 밖이면 예외)
- **behavior_calibration.py:** `fit_temperature`, `fit_vector`, `cv_select`
- **behavior_metrics.py:** 18-class `nll_brier`, `top_label_ece10`, `ece_bootstrap`(경기별 bin 충분통계), `paired_game_bootstrap`, `near_zero_share`, `group_calibration`(SC-1), `ratio_bins`(SC-3)
- **behavior_weights.py:** `read_ledger(path, manifest)`(키 단위 스캐너로 파싱 때 허용 목록 투영 — 허용 밖 값은 디코드 없이 건너뜀, 원시 바이트 해시는 별도, 금지 파일 경로 거부), `e0_from_ledger`, `pa_weights(rows, logging_override)`, `reproduce_b0_ess(expected)`, `weight_shift`, `mean_weight_test`. ESS는 `policy_estimator.effective_sample_size`(:293–300) 재사용
- **scripts/run_behavior_calibration.py:** 단계(고정 순서) `smoke(등록 뒤 첫 단계, §12 G1) → census(라벨 없음) → fit → calibrate → freeze-primary → s6-integrity → xf-fit → freeze-xf → score-dev → s6-weights → decide`. 앞 단계가 통과해야 다음 단계가 열린다
  - `--config configs/EXP-BC-CAL-01.yaml`(값은 config에서만, CLAUDE.md 규칙)
  - 공통 heavy lock(configs/ML-POLICY-MATERIALIZATION-v1.json:681 경로) 비차단 획득, 단계별 hang guard, 2026 날짜 거부
  - **동결 규칙(M-2 부록 관례, 등록 블록에 추가 전용):** ① `freeze-primary` = B1·B2 적합, 보정기, B0-T 온도, OOF 인코더, A3 타자 성향 스냅샷(§5 E)의 sha256와 각 산출물의 입력 행 날짜 범위를 담은 매니페스트. 이것이 pin되기 전에는 DEV 행의 구종 라벨을 하나도 읽지 않는다. XF 레시피(코드·config·폴드 소금)는 등록 때 이미 고정이다. ② `s6-integrity`(S6 식별 pin 대조 → E0 재구성 → R1b → R2, §7.4)는 S6 경기의 로그 행동을 처음 읽는 단계이며, 등록된 고정 B0 무결성 비교(S6 식별·R1b·R2)만 하고 DEV 지표(§7 정의)는 내지 않는다. 실패하면 FAILED_INTEGRITY로 멈추고 뒤 단계는 없다. ③ `xf-fit`은 DEV 구종 라벨로 B0-XF·B1-XF만 맞추는 별도 단계다(결정론적, α는 TRAIN-only 적합값, 폴드별 조정 없음, 라벨 사용은 노출 장부에 기록). ④ `freeze-xf` 매니페스트가 pin되기 전에는 DEV 지표(§7 정의: 후보·서술 arm의 채점 지표와 새 분모 가중치 진단)를 계산하지 않는다 — 예외는 ②의 등록된 무결성 검사뿐이다. ⑤ `score-dev`가 DEV 라벨 집계(P·S_in·S_off·P⁺·TRAIN0)와 모든 DEV 지표를 낸다
  - 출력은 새 폴더(`…/ML-MATRIX-20260924/EXP-BC-CAL-01/<attempt>`). S5·S6 폴더에는 쓰지 않는다
- **테스트(합성 자료만)**
  - T1 `VectorBC` = `CategoricalBC`(무작위 합성, 최대 차 ≤ 1e-12), B1의 B0 설정 = B0
  - T2 누수 교란: 행 i의 자기 구종·물리·결과·description과 이후 행을 바꿔도 행 i의 특징·확률이 같다
  - T3 현재 투구 차단: 현재 pitch_type·물리를 바꿔도 같다
  - T4 OOF 인코딩: 자기 라벨을 바꿔도 자기 인코딩이 같다
  - T5 폴드: 결정론적, 경기 단위, 소금이 다르면 다른 배정
  - T6 지지: 밖 0, 합 1, B0와 같은 지지
  - T7 보정: 합성 logit에서 알려진 T 회복, 벡터 보정이 마스크 보존
  - T8 지표: 18-class 함수에 10-class 자료를 넣으면 matrix_metrics와 같은 값, 부트스트랩 재현
  - T9 가중치·식별: 합성 원장에서 ESS = `effective_sample_size`, 양성·OUTSIDE 행 ρ = 0, 재현 게이트 동작; 결정 행의 `runtime_sha256`이 pin한 머리말 값과 다르거나 머리말 `pins.candidate.sha256`이 기대 정책 식별자와 다르면 식별 게이트가 FAILED_INTEGRITY
  - T10 허용 목록: 키 단위 스캐너 투영으로 만든 레코드에 `q_*`·`mc_se`·보상 키가 없음(금지 키가 섞인 합성 원장 줄로 확인), 원시 바이트 sha256가 투영과 따로 계산됨, 판독기가 `pa_values.parquet`·`dr.json` 경로를 거부
  - T11 2026 방어: 2026 날짜 한 행에도 예외
  - T12 SC-2 합성: 모형이 만든 로그에서 평균 ≈ 1, 일부러 틀린 모형에서 어긋남 검출
  - T13 단계 순서: freeze-primary 전 DEV 라벨 읽기, freeze-xf 전 DEV 지표(§7 정의) 계산, s6-integrity 실패 뒤 xf-fit·score-dev 진입, 등록 전 smoke를 실행기가 모두 거부(s6-integrity의 S6 식별·R1b·R2는 허용)
  - T14 성향 입력 필터: June·DEV·2023-03-30..05-14·2026 행 하나가 섞여도 §5 E 스냅샷 입력에서 빠지거나 예외
  - T15 입력 준비: 합성 parquet 재분할이 행·순서·스키마와 모든 열 바이트 해시를 보존하고 행 그룹 경계가 등록 날짜에 정확히 놓임; 날짜 순이 아닌 원본은 준비가 거부; 분할 경계에 걸친 행 그룹이 있는 입력은 판독기가 footer 통계로 거부; 정렬된 입력에서 score-dev 전까지 DEV 행 그룹을 디코드하지 않음
- 마무리(CLAUDE.md 흐름): configs/EXP-BC-CAL-01.yaml → results/EXP-BC-CAL-01.json → docs/experiments.md 한 행, 채택이면 docs/decisions.md 한 줄

## 11. 비용·자원·보호 (선례만 인용; 이 실험 자체 비용은 미측정)

- 자료 준비 선례: P8 prepare 186.6초 (configs/ML-POLICY-MATERIALIZATION-v1.json:663). B0 채점 선례: P8 p0 평가 4.02초(Cpanel) (docs/reports/ML-matrix-execution-2026-09-24.md:372)
- B0·B1: 계수 연산이라 분 단위로 예상(미측정). B1 좌표 하강은 조기 종료 분할 평가 최대 약 7층 × 12값 × 2바퀴
- B2 선례
  - M4(≤2025 자료): 10-class, 200라운드, 2스레드 — 556,467행 104.5초, 623,184행 161.6초, 최대 RSS 1.45 / 1.71 GB (/Users/song/Projects/pitcheezy-pm-jobs/20261008-until-0830/multiclass-tree-run-launch-001/REPORT.md:16–19 — child seconds 104.462·161.557, RSS 1,450,721,280·1,709,735,936 B, 200 round; 학습 행 556,467·623,184는 attempt-001/result.json:395, :891)
  - WS 줄기(2026 자료): EXP-WS-RECENCY-002 CPU 재적합 — 부모 10-class 179.11초, 이진 head 8.39 / 4.82초, 전체 398.18초, 표본 최대 RSS 5.38 GB (results/EXP-WS-RECENCY-002.json:22)
  - A3 B2(18-class, TRAIN 약 138만 행 — 1,381,056행 docs/results-all.md:1473, 300라운드)는 **미측정**이다. M4 단가(행·라운드·클래스당, 위 두 실측; 열 수 199 대 63의 차이는 반영하지 않음)로 환산하면 1회 약 12–16분(산술 약 700–967초), RSS 수 GB로 추정한다(투영, 측정 아님)
- 필수 단계 합계: 맥 CPU 1시간 안팎으로 추정(미측정). B2-XF는 D186으로 돌리지 않는다. 상한(**판단값**, 등록 JSON `cost_budget`과 같음): 시도(attempt) 전체 벽시계 14,400초(1시간 추정의 4배) — smoke부터 decide까지 모든 단계에 적용되고, 여러 명령으로 나눠 돌려도 쓴 시간을 attempt 폴더에 누적 기록해 이어서 센다. 넘으면 중단하고 산출물 보존, B2 적합 1회 소프트 상한 3,600초(12–16분 투영의 3.75–5배; 넘으면 경고를 기록하고 계속하며, 중단은 단계 hang guard 7,200초와 전체 상한 14,400초가 한다)
- **실행 장소·창(D186 2-②):** 맥미니 CPU(자료가 T7·Drive에 있음), 경기일 13:30 KST 이후 창(기록기·발행기 보호창 06:30–13:30 KST 밖), num_threads 2. WS 종료를 기다리지 않고, 트랙 A 세 실험은 공통 heavy lock 아래 순차로 돌린다. 서버 CPU는 ≤2025 processed·raw 자료 이동과 `.gpurc` 링크 추가가 필요해 기본안이 아니다. GPU는 쓰지 않는다(B3 제외)
- **보호:** 공통 `.heavy.lock` 비차단 획득. 시작 전 memory_pressure 여유 ≥ 40%(**판단값:** STAB-01은 63%에서 시작). 프로세스 RSS 10 GiB(10,737,418,240 B) 넘으면 중단(**판단값:** S6 최대 6.77 GiB 선례보다 여유를 두고 24 GB 맥에서 기록기·서비스 몫 확보). 기록기·수집기에 신호 보내지 않음, 보호 패턴으로 pgrep/ps 하지 않음, 포트 8766–8772 손대지 않음, push 없음, 맥 예약 작업과 겹치는지 실행 직전 확인

## 12. 게이트·노출·중단 규칙

- **게이트**
  - G1 순서(D186 + PM 확정 2026-10-27, 트랙 A 공통): ① 설계 검토(이 문서·JSON) → ② 고정 커밋의 코드·config 검토(smoke 코드 포함, 코드 읽기만, 실자료 실행 없음) → ②′ 입력 준비(§6: PM이 코드 검토 PASS 뒤 06:30–13:30 KST 밖에서 1회, 분할 정렬 재분할·바이트 동일성 증명) → ③ 등록(status REGISTERED, execute true, null 필드 채움, 준비 입력의 sha256·식별 매니페스트 sha256 pin, 준비 실행 기록을 등록 부록으로) → ④ smoke = 첫 등록 단계(TRAIN 행만, 비용·무결성만, 지표 계산·열람 없음; 기록은 등록 블록에 추가 전용 부록으로). 'TRAIN 행만'은 TRAIN 행만 어떤 계산에든 들어간다는 뜻이다 — 분할 정렬된 준비 입력에서는 smoke가 TRAIN 행 그룹만 디코드하며, 경계에 걸친 그룹이 있으면 판독기가 입력을 거부한다(§6 입력 준비·행 그룹 선택) → ⑤ 본 실행(§10 단계 순서). 등록 전에는 smoke를 포함해 어떤 실자료 단계도 하지 않는다. 검토 통과 기준은 아래 '독립 검토 요구'
  - G2 자료 역할·as-of: ≤2025만, DEV = 노출된 개발 자료, June 안 씀(타자 성향 이력 입력 포함, §5 E), 2026 0행, S6 원장은 허용 필드만 읽기 전용
  - G3 봉인·OPE 경계: 보상·가치·OPE 결과를 읽지 않는다(π̂_b는 자기 우도와 진단으로만 평가). 봉인 원장·S5/S6 산출물은 그대로 둔다(sha 전후 확인). c13cc989·D108/D131/D155 불변
  - G4 분할·지표·문턱은 이 문서로 결과 전 고정
  - G5 seed·자료 버전·커밋을 결과에 기록
- **노출 장부:** TRAIN exposed_fit · earlystop exposed_model_selection · temperature exposed_calibration(TRAIN·earlystop·temperature는 타자 성향 이력 입력으로도 쓰임, §5 E) · June·2023-03-30..05-14 미사용(이력 입력 포함) · DEV exposed_development(s6-integrity가 S6 경기 로그 행동을 읽고, xf-fit이 DEV 구종 라벨로 XF만 적합하고, score-dev가 1회 채점) · S6 원장은 허용 필드(확률·식별 메타데이터)만 읽기 · 2026 없음
- **중단 규칙**
  - S6 식별 pin(원장·매니페스트 sha256, 머리말 `pins.candidate.sha256`, 행별 `runtime_sha256`) 불일치 또는 B0 재현(R1·R1b·R2) 실패 → FAILED_INTEGRITY, xf-fit·DEV 채점 안 함
  - 단계 순서 위반 시도(등록 전 smoke, freeze-primary 전 DEV 라벨, freeze-xf 전 DEV 지표 — §7 정의, s6-integrity 무결성 검사 제외) → 실행기가 거부하고 중단·기록
  - 누수·합성 테스트 실패 → 실자료 단계 시작 안 함
  - 2026 행이나 2025-09-30 이후 날짜를 읽음 → 즉시 중단
  - 금지 파일(보상·값) 접근 시도 → 판독기 거부, 중단·기록
  - 메모리·hang guard(단계당 7,200초, config 선례 :447 = CLAUDE.md D10 로컬 2시간 상한)·경기 창 → 중단, 부분 산출물은 인용 불가
  - B2가 300라운드 끝까지 개선 중 → `CAP_HIT` 기록, 이 등록 안에서 늘리지 않음
  - B2 적합이 소프트 상한 3,600초를 넘음 → 경고만 기록하고 계속(중단은 hang guard·전체 상한)
  - 벡터 보정 실패 → 등록된 대체(온도만)로 가고 기록. 온도 보정도 실패 → 그 후보 `CALIBRATION_FAILED`(자격 없음), 기록
  - DEV 지표가 한 번 쓰이면 재채점·후보 추가·문턱 변경 없음. 바꾸려면 새 등록 ID와 공개
  - 버그 수정 재시도는 DEV 지표가 쓰이기 전까지 기록 후 허용(경량 게이트)
- **독립 검토 요구:** 작성자(Opus)와 다른 에이전트. 권장(D182): Codex Astra 교차 검토(누수·허용 목록·결정론 코드) + Fable 판단 검토(판정 규칙·문턱). 대상 = G1의 ①(이 문서·등록 JSON)과 ②(고정 커밋의 코드·config, smoke 코드 포함). 설계 검토는 코드 검토를 대신하지 않는다. 등록을 여는 판정은 두 검토 모두 `PASS`이거나, `PASS_WITH_NOTES`이고 모든 NOTE가 닫혔거나 PM이 수용으로 기록한 경우뿐이다. `PASS_WITH_CONDITIONS`는 쓰지 않는다

## 13. 주장하지 않는 것

- 어떤 정책의 OPE 값·Δ도 내지 않는다. D131 판정이 바뀐다고 말하지 않는다.
- 인과 식별 없음. 수치상 겹침(ESS)은 식별이 아니다 (prereg:559 `numeric_overlap_is_identification: false`).
- 독립 확인 아님(DEV는 노출된 개발 자료).
- 서비스·정책 채택 없음. c13cc989·G0·G15·서비스 변경 없음.
- 정확도는 목표가 아니다(top-1은 서술).
- 의도 모형 아님: π̂_b는 Statcast가 분류한 **실제 투구의 구종 라벨** 분포다(포수 사인·의도와 다를 수 있음; 가정은 명시만 하고 입증하지 않음).
- 위치(구종×구역) 성향 없음: 관측 위치는 의도 목표가 아니라 실제 도달이라 ARM-A는 식별 불가로 막혀 있다 (prereg:255).
- 2026이나 2027로의 이전 가능성 주장 없음.

## 14. Song 결정 (D186, 2026-10-27)

D186(docs/decisions.md:325)으로 정해진 것과 남은 것. 초안 때의 선택지·추천은 이력으로 괄호 안에 남긴다.

1. **B3(GPU 시퀀스 신경망) — 결정됨(D186 5-①): 제외.** arm이 아니며 실행하지 않는 미래 후보로만 기록한다(§4). (초안 선택지: ① 이번엔 안 함(추천, 채택) ② 함께 실행 ③ B0~B2 결과 뒤 별도 등록으로 판단.)
2. **가중치 진단 표본 — 결정됨(D186 5-①): S6 150경기 원장 재사용**(§7.4, 확률 필드만 읽기 전용). (② STAB-01의 S6 밖 150경기, ③ 둘 다는 채택하지 않음.)
3. **교차 적합 팔 — 결정됨(D186 5-①): B0-XF·B1-XF만**(§8). B2-XF는 돌리지 않는다. (② B2-XF까지, ③ 없음은 채택하지 않음.)
4. **실행 시점 — 결정됨(D186 2-②): 경기일 13:30 KST 이후 창**(맥 CPU, 기록기·발행기 보호창 06:30–13:30 KST 밖, §11). WS 종료를 기다리지 않는다. 순서는 §12 G1(설계 검토 → 코드 검토 → 등록 → smoke → 본 실행), 트랙 A는 순차. (초안 선택지: ① 월드시리즈 끝난 뒤 첫 비경기일(당시 추천) ② 경기일 13:30 KST 이후 창(채택) ③ 서버 CPU.)
5. **행동 공간 — D186에서 따로 정하지 않음.** 이 설계는 ①(구종 18개만, ARM-B)을 쓰고 등록 JSON `action_space`도 같다. 제안서 A3 행의 "225-way" 문구는 따르지 않는다(위치는 의도 미식별로 막혀 있음). ②(구종 × 거친 구역 서술 추가)로 넓히려면 새 결정이 필요하다.

## 15. 확인 못 한 것·주의

- **"WS 줄기 M4 LightGBM 6-head 337초"는 확인하지 못했다.** 336.9초는 EXP-WS-STEP5-CAL-001의 전체 실행(학습 0, 보정 측정; results/EXP-WS-STEP5-CAL-001.json:3347)이다. 학습 비용 선례는 §11의 M4·RECENCY-002 값을 쓴다.
- **Ahn 외 2026의 0.637 vs 0.633(2018 기준):** 초록에는 "다음 공 약 64%"만 있고 2018 기준 비교 수치는 없다(본문 미확인). 제안서는 "+0.004"라고 적었다(experiment-proposals-2026-10-26.md:39).
- **CalArena:** arXiv 2511.03685는 "Structured Matrix Scaling for Multi-Class Calibration"(같은 저자들)이고 CalArena는 arXiv 2605.30188이다. "구조화 행렬 스케일링 1위"는 두 초록 모두에서 확인하지 못했다(제안서 :35의 "승률 > 0.7"은 본문 출처로 보이며 미확인).
- 옛 줄기 가중치 평균 붕괴값이 기록마다 다르다(0.05 vs 0.24, §2.2).
- `workload_features.py`·`run_cuda_direct.py`는 현재 체크아웃(브랜치 codex/ml-matrix-execution)에 없고 브랜치 research/direct-head-confirm-1026(worktree /Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026)에 있다. 구현 때 이식한다.
- S6 원장의 결정 수와 "원장만으로 재구성한 E0 = 러너의 E0"는 실행 때 R2가 확인한다(지금은 미확인).
- STAB-01 원장에 실제 구종(`logged_index`)이 남아 있는지 미확인.
- A3의 모든 모집단 행 수, B0의 전체 MLB NLL·ECE, 모든 비용은 미측정이다.
- S6 원장 파일 크기, materialize.json의 셀 수 81,132·`logging_mass_on_mask` 분위는 레포 기록 요약이 없어 검토 범위에서 미검증이다. S6-dr-a5 원장 머리말 `runtime_sha256`(런타임 자신의 식별, PM 확인 앞자리 `af59057b…`)의 전체 값은 등록 때 머리말에서 pin한다. 정책 식별자 `c13cc989…`는 머리말 `pins.candidate.sha256`에서 대조한다(§7.4).

## 16. 주요 출처

- 코드: experiments/pitchmdp/pitchmdp/{rollout_policy.py, matrix_policy.py, policy_estimator.py, policy_runtime.py, policy_tau.py, policy_requests.py, policy_artifacts.py, matrix_features.py, matrix_sharing.py, matrix_bridge.py, matrix_metrics.py, matrix_panel.py, matrix_models.py, archetypes.py, sequence_data.py}, experiments/pitchmdp/scripts/{run_temporal_blend.py, run_ml_policy.py}, src/pitcheezy/ope/behavior.py, experiments/ws_seq/calibrate_logits.py; 브랜치 research/direct-head-confirm-1026의 experiments/pitchmdp/pitchmdp/workload_features.py, experiments/pitchmdp/scripts/{run_cuda_direct.py, run_ml_direct_workload.py}
- 설정: configs/ML-POLICY-MATERIALIZATION-v1.json, configs/MLB-2026-OPE-PREREG-DRAFT-v1.json, configs/EXP-P0-002.yaml
- 결과: results/ML-POLICY-VAL-v1-S6.json, results/ML-POLICY-VAL-v1-S6-v3.json, results/EXP-P1-010.json, results/EXP-WS-RECENCY-002.json, results/EXP-WS-SEQ-001.json, results/EXP-WS-STEP5-CAL-001.json; T7 S1-materialize-a3/materialize.json; pm-jobs 20261008-until-0830/multiclass-tree-run-launch-001; pm-jobs 20261020-stab01
- 이번 검토 반영에 쓴 위치: experiments/pitchmdp/pitchmdp/policy_requests.py:277–298(성향 스냅샷, 이력 열 :284), policy_estimator.py:142–157(affine 재귀), policy_runtime.py:291·:489(runtime_sha256)·:327–338(손 확인), configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:303·:800–828, configs/ML-POLICY-MATERIALIZATION-v1.json:506–514, configs/ML-POLICY-MATERIALIZATION-v1.addendum-8-d127.json, docs/results-all.md:1473, pm-jobs multiclass-tree-run-launch-001 REPORT.md:16–19·attempt-001/result.json:395, :891
- 문서: docs/decisions.md(D18:23, D104:154, D108:162, D109:164, D111:168, D115:176, D122:190, D123:192, D125:196, D131:208, D155:258, D180:313, D184:321, D186:325), docs/experiments.md:73·:263, docs/results-p0.md:12·:27, docs/reports/ML-matrix-execution-2026-09-24.md:319–372, docs/reviews/external-evaluation-2026-10-19.md:103·:117, docs/pm/research/experiment-proposals-2026-10-26.md
- 문헌(초록 확인): Hanna·Niekum·Stone arXiv 1806.01347; Kovash & Levitt NBER w15347; Ahn 외 arXiv 2602.07030; Berta 외 arXiv 2511.03685, arXiv 2605.30188(CalArena)

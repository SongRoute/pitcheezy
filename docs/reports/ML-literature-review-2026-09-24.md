# ML0·ML1 실행 전 문헌 검토

확인일: 2026-09-24. 저장소 기준: `61ca0f9`(시작 시 clean). 요청 범위는 **기존 참고 논문과 추가 비교 후보의 정리**다. 이 문서는 문헌 검토와 실험 제안이며, ML0 사전 고정 사양이나 실행 결과가 아니다. 새 학습·평가, 2026 경기 데이터·최종셋 열람은 하지 않았다.

읽은 시작점: [세션 인수인계](../SESSION_HANDOFF.md), [ML 실행 인수인계](../handoffs/ML-experiments-next-session.md), [기존 기준선](../baselines.md), [기존 문헌 목록](../../experiments/pitchmdp/LITERATURE.md), [기존 연구 차이 검토](../../experiments/pitchmdp/docs/RESEARCH_GAP_REVIEW.md).

후속 요청에 따른 [전체 실험 설계](../ML_EXPERIMENT_DESIGN.md)는 이 15편을 실제 대조 실험에 연결하고, offline RL·정책 평가·강건성 방법론 7편을 추가한다. 검증 모집단은 MLB 정규시즌 선발·불펜 전체이며 현재는 설계 단계다.

## 1. 프로젝트에서 비교할 문제

주 과제는 **투구 전 정보와 후보 행동을 조건으로 다음 투구의 결과 확률을 예측**하는 것이다. 구종 자체의 예측, 실제 물리를 관측한 뒤의 결과 예측, 타석 종결 결과 예측은 별도 과제다. 논문 점수를 그대로 옮겨 하나의 순위표를 만들 수 없다.

현재 서비스 계열의 결과는 `ball / strike / foul / out / single / double / triple / home_run / hbp / double_play`의 10개다. [라벨 코드](../../experiments/pitchmdp/pitchmdp/model.py)의 `outcome_labels`와 적격성 조건을 확인했다. 마지막 볼·스트라이크도 여기서는 ball/strike이며 타석 종료는 전이에서 처리한다. 예전 `src/pitcheezy` 11-class 연구와도 구분한다.

현재 공의 실제 구속·회전·도달 위치는 투구 전에는 알 수 없다. 물리 조건부 모델을 비교할 때는 TRAIN에서 얻은 같은 delivery 표본으로 적분한다. 이 예측 적분과 **의도한 목표 위치에 대한 실행 분포의 검증**은 별개다.

| 기존 프로젝트 근거 | 문헌 선택에 주는 의미 |
|---|---|
| 2025 DEV NLL: 빈도 1.505737, MLP+빈도 1.491226, Transformer+빈도 1.490642. Transformer−MLP는 −0.000584, 경기 bootstrap 95% 구간 [−0.002555, +0.001297] | 강한 MLP를 기준으로 유지. Transformer가 이미 우수하다고 가정하지 않는다. |
| 연속 타자 성향의 유용성은 확인했지만, 군집과 같은 타석 과거 5구의 추가 이득은 일관되게 확인하지 못함 | 단순 ID·군집·0/5구 반복보다 다른 시간 범위 또는 다른 학습 가설이 필요하다. |
| 기존 NN 학습 276,821행, 전체 TRAIN 1,252,824행 | ML1의 ‘100%’가 무엇인지 먼저 정의해야 한다. |

근거: [시간 분할·혼합 결과](../../experiments/pitchmdp/TEMPORAL_BLEND_RESULTS.md), [타자 표현·이력 결과](../../experiments/pitchmdp/REPRESENTATION_HISTORY_RESULTS.md), [0/5구 추가 검증](../../experiments/pitchmdp/HISTORY_BATTER_VALIDATION_RESULTS.md). 모두 이미 노출된 DEV 결과이며 새 일반화 검증이 아니다.

## 2. 기존에 참고하던 핵심 논문 5편

### E1. Otremba Jr. — SmartPitch (MIT MEng, 2022)

**핵심:** 신경망이 strike/ball/foul/in-play 확률을 추정하고, 12개 카운트 상태 MDP가 이를 이용해 투구 전략을 계산한다. 개인 타자 성향과 존 높이를 이미 사용한다. 주자·아웃에 따른 가치 확장은 하지 않고 보상을 무주자·무사 기준으로 둔다.

- 원문 §7.1, pp.70–76: 2019 자료 70/30 분할, 77차원 입력, 최선 MLP hidden 128×2. 빈도·로지스틱·RF·GBDT와 비교한다.
- Table 7.1의 MLP cross-entropy **0.861**, Brier **0.482**는 4-class 원문 참고값이다. 현재 공 물리와 실제 도달 좌표를 입력으로 쓴다.
- **적용:** ML0의 확률 지표·단순 기준선, ML2의 MLP/GBDT 비교. 기존 B1 축소 MDP는 충실 재현이 아니다. 타자 시즌 집계·평균 존의 계산 시점을 감사한다.

확인: 공식 PDF 본문 직접 확인. [MIT 서지](https://dspace.mit.edu/handle/1721.1/145144), [PDF](https://dspace.mit.edu/server/api/core/bitstreams/64b16c4f-a7e6-49a6-a7e5-806f9234f2a9/content).

### E2. Melville et al. — A Game Theoretical Approach to Optimal Pitch Sequencing (Sloan, 2023)

**핵심:** OptimusPitch가 타자 embedding과 타석 이력을 이용해 9개 투구 결과를 예측한다. 목표→도달 오차 및 타자가 공을 얼마나 관측했는지에 따라 달라지는 게임 균형을 함께 다룬다.

- §4, p.11의 구조는 일반 LSTM/GRU가 아니다. 결과 예측은 `MLP(x_t, batter_embedding, h_t)`, 상태 갱신은 `h_(t+1) = h_t + MLP(x_t, embedding(y_t))`다. 현재 결과 `y_t`는 다음 공부터 사용한다.
- 2021–22 타석 단위 무작위 75/25 분할. §4.1, pp.12–13: CE **1.11**, memoryless **1.12**. 후반 공 번호별 비교와 9-class 보정 곡선을 제시한다.
- **적용:** ML2의 자체 recurrent 기준선, ML0의 공 번호별 진단. 현재 공 물리 입력·무작위 분할을 바꾸면 적응 실험으로 표기한다.

확인: Sloan 공식 PDF 본문 직접 확인. [발표](https://www.sloansportsconference.com/research-papers/a-game-theoretical-approach-to-optimal-pitch-sequencing), [PDF](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc22080794_A%20Game%20Theoretical%20Approach%20to%20Optimal%20Pitch%20Sequencing.pdf).

### E3. Takamido & Nakamoto — Counterfactual Optimization of Baseball Pitch Sequences and Estimation of Its Impact on Season-Level Statistics (arXiv, 2026)

**핵심:** 현재 공을 포함한 마지막 6구의 물리 시퀀스와 상황·타자 성향을 Transformer에 넣어, 2스트라이크 종결구의 in-play/swing-out을 예측한다. 결정구·셋업구를 교체한 모델 출력 변화로 시즌 지표 변화를 추정한다.

- 2018–24 학습(2020 제외), 2025 평가. §2.5는 TRAIN 내부 validation F1로 임계를 고른다고 명시한다. 우리 적응은 임계를 별도 CAL에서 정한다.
- §3.1 보고값: accuracy .756, precision .755, recall .879, F1 .812, AUC .811, 임계 .43.
- **감사 발견:** §2.2는 in-play=1이지만 Table 2에서 위 P/R/F1은 swing-out을 양성으로 계산해야 재현된다. 아래 §5에 산술을 남겼다.
- **적용:** 시퀀스 기준선·이진 보조 과제. §2.6의 후보 물리 평균은 **2025 시즌 자료**를 쓰므로 전향적 적응은 TRAIN/시점 이전 추정으로 바꿔야 한다. 모델 내부 반사실·회귀 환산은 실제 정책 효과가 아니다.

확인: 원문 §§2.2–2.6, 3.1. [서지](https://arxiv.org/abs/2606.17345), [PDF](https://arxiv.org/pdf/2606.17345), [저자 코드](https://github.com/takamido/Pitch_sequence_analysis). 코드 라이선스·정확한 판본은 실행 전 재확인 대상이다.

### E4. Douglas et al. — Computing an Optimal Pitching Strategy in a Baseball At-Bat (arXiv, 2021)

**핵심:** 투수·타자의 구종×위치 통계 텐서를 CNN으로 결합한다. 목표→도달 분포, 스윙 조건부 결과, 타자의 바깥공 반응을 나누고 OBP를 최소화하는 확률 게임을 푼다.

- 스윙 조건부 결과는 헛스윙 strike/foul/hit/out 4개이며 전체 투구 4-class가 아니다. 2015–18 자료에서 선수 비중복 분할을 기술한다.
- 목표 라벨 대신 3-0에서는 일정 지점의 스트라이크를 노린다는 가정으로 제구 Gaussian을 추정한다. 실제 의도 라벨 검증과 다르다.
- **적용:** 계층적 결과 head, 선수 표현, delivery 분리의 근거. 관측 OBP .329와 모델 균형 .242의 차이를 실경기 개선으로 해석하지 않는다.

확인: HTML §§3–5. 기존 인수인계의 평가 절 ‘§6.1’은 확인한 v1 HTML 기준 **§5, Outcome Predictions**로 정정한다. [원문](https://arxiv.org/html/2110.04321v1).

### E5. Mott et al. — The Impacts of Increasingly Complex Matchup Models on Baseball Win Probability (arXiv, 2025)

**핵심:** 계층적 Bayesian log5 계열에 투수→타자→최근 기록→주루를 차례로 추가한다. 9개 **타석 결과**와 주자·아웃 전이를 분리해 감독 의사결정의 승리 확률에 연결한다.

- §2.3은 최근 최대 2,000타석을 500타석씩 나누어 더 최근 표본을 강조한다. §2.5는 2015~2024년 6월 TRAIN, 7~10월 validation과 outcome/transition log loss를 사용한다.
- **적용:** ML1의 최근 가중 가설, 이후 적은 표본 선수의 shrinkage 및 WE 연결 참고. 투구별 10-class 직접 기준선은 아니다.
- 시뮬레이터가 가장 상세한 BR 모델을 참값으로 삼는다. 약 1승/162경기는 해당 환경의 추정이며 실제 정책 효과를 입증하지 않는다.

확인: HTML §§2–3. [원문](https://arxiv.org/html/2511.17733v1).

## 3. 추가로 찾은 야구 논문 6편

### N1. Kneita — Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization (Sloan, 2025)

**가장 가까운 추가 예측 비교 후보.** 타자별 최근 400구(현재 공 포함)를 여러 타석·경기에 걸쳐 모은다. 12층 Transformer로 투구 결과 10-class·타구 위치 9-class와 연속 보조 목표를 함께 학습한다.

- §§3.1–3.4: 2015–22 학습/검증, 2023–24 테스트로 설명하며 주요 결과표는 2023이다. 현재 공 결과는 마스킹하지만 구속·회전·릴리스 등은 유지한다.
- §4.1의 ‘top-4 precision’은 정답이 상위 4개에 포함되는 비율로, 우리의 NLL과 다르다. 10-class에도 walk/strikeout이 있고 우리 foul/double_play와 대응이 달라 매핑이 필요하다.
- **가져올 가설:** 타석을 넘는 최근 이력이 기존 연속 타자 성향에 추가 정보를 주는가. ML3에서 짧은 이력/동일 과거 범위 집계와 비교한다. 긴 이력·새 backbone·다중과제를 동시에 추가하지 않는다.
- 공개 코드·라이선스는 이번에 확인하지 못했다. 원문 재현 가능성 감사 후 우선순위를 확정한다.

확인: 공식 PDF §§3–4, Table 1. [발표](https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization), [PDF](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc220808eb_Transformer-Based%20Baseball%20Modeling%20for%20Pitch%20Outcome%20Prediction%20and%20Strategy%20Optimization.pdf).

### N2. Alcorn — (batter\|pitcher)2vec: Statistic-Free Talent Modeling With Neural Player Embeddings (Sloan, 2018)

**핵심:** 타자·투수 ID의 학습 embedding을 연결해 타석 결과 분포를 예측한다. 2013–15 Retrosheet로 학습하고, 학습에서 본 선수들의 2016년 미관측 대결 조합을 평가한다. 원문 출력은 49개 타석 결과다.

**적용:** 선수 ID 표현과 연속 성향의 비교 근거, ML5에서 ‘새 조합’과 ‘새 선수’를 구분하는 평가 설계. 현재 프로젝트는 이미 ID 대 연속 성향을 비교했으므로 같은 ID 실험을 새 개선으로 반복할 이유는 약하다. 새 선수 일반화를 입증한 논문으로 인용하지 않는다.

확인: 저자 저장소의 논문 LaTeX와 README. MIT 코드 라이선스 표시를 확인했으나 코드를 가져오지는 않았다. [저자 논문·코드](https://github.com/airalcorn2/batter-pitcher-2vec), [논문 소스](https://github.com/airalcorn2/batter-pitcher-2vec/blob/master/paper/batter_pitcher_2vec.tex).

### N3. Heaton & Mitra — Learning to Describe Player Form in the MLB (2021 공개본; MLSA 논문집 2022)

**핵심:** 최근 경기에서 선수가 보인 플레이를 Transformer와 대조학습으로 72차원 form에 담는다. 타자는 20타석에서 겹치는 15타석 두 구간, 투수는 100타석에서 90타석 두 구간을 만들어 가까운 시점의 표현을 학습한다.

**적용:** ML3의 ‘현재 폼’ 후보. 동일 시간 범위의 단순 이동평균/가중 성향보다 나은지 먼저 확인한다. 원문은 표현·군집 분석 중심이며, 우리 투구 NLL 개선을 입증한 결과가 아니다. 전체 시즌으로 만든 form이나 미래가 포함된 window를 과거 예측에 사용하지 않는다.

확인: 공개본 §3·결론. [PDF](https://arxiv.org/pdf/2109.05280), [논문집](https://link.springer.com/book/10.1007/978-3-031-02044-5).

### N4. Nakahara, Takeda & Fujii — Pitching strategy evaluation via stratified analysis using propensity score (JQAS, 2023)

**핵심:** 실제 포수 요구 위치가 있는 2014–19 NPB 자료에서, 포심의 몸쪽/바깥쪽 **목표 선택** 효과를 propensity score·IPW·층화 분석으로 평가한다. 실제 도달 위치와 목표를 구분하는 것이 핵심이다.

**적용:** CV 의도 인수 후 추천 정책 평가의 우선 참고문헌. 현재 ML0·ML1의 예측 모델 기준선은 아니다. 관측 교란 조정을 하더라도 미측정 교란이 없다는 가정은 남으며, NPB 포심 결과를 MLB 모든 구종으로 옮길 수 없다.

확인: 공개본 §§2–5와 출판 서지. [원문](https://arxiv.org/html/2208.03492v1), [JQAS](https://doi.org/10.1515/jqas-2021-0060).

### N5. Ahn et al. — Neural Sabermetrics with World Model: Play-by-play Predictive Modeling with Large Language Model (arXiv, 2026)

**핵심:** 경기 이벤트를 텍스트 순서로 바꾸고 Llama-3.2 3B를 계속 사전학습해 구종과 스윙 결정을 예측한다. 원문은 약 700만 투구·30억 토큰, TPU-v4-64 학습을 기술한다.

**적용:** 장기적으로 이벤트 표현·여러 과제 공유의 참고. ML0·ML1 우선 실행 대상으로 삼을 근거는 약하다. 평가 과제와 계산 규모가 다르다.

**원문 감사:** Table 1은 fastball/non-fastball accuracy .637인데 §6 본문은 next-pitch 약 84%라고 쓴다. 별도로 타석 중 최소 1구를 맞힌 비율 83.8%가 제시된다. 이 수치들을 같은 정확도로 인용하지 않는다. 현재 공 물리의 토큰 순서·예측 prefix도 확인해야 한다.

확인: HTML §§3–6, Tables 1–2. [원문](https://arxiv.org/html/2602.07030v1). 공개본의 보고 일관성 문제를 기록한 것이며, 저자 의도나 수치 오류의 원인을 확정하지 않는다.

### N6. Park et al. — Structure of Pitch-Pattern Motifs in Major League Baseball (arXiv v2, 2026)

**핵심:** 2008–25 투구 패턴의 다양성·빈도 구조를 분석한다. 배합에 비무작위적 구조가 있지만 그 구조와 ERA·승수의 관계는 제한적이라고 보고한다.

**적용:** ‘배합 패턴이 존재한다’와 ‘배합 피처가 예측을 개선한다’를 구분하는 참고. 우리 0/5구 결과를 설명하는 확정적 증거는 아니며 새로운 backbone 기준선도 아니다.

확인: v2 초록·본문 논의. [원문](https://arxiv.org/html/2601.11904v2).

## 4. ML 실험 방법에 직접 쓸 논문 4편

| 논문 | 핵심 아이디어 | 프로젝트 적용 및 한계 |
|---|---|---|
| **Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022)**, *Why do tree-based models still outperform deep learning on typical tabular data?* | 여러 tabular 데이터에서 NN과 tree를 탐색 비용까지 비교하며, 무관한 입력·불규칙 함수 등에 대한 차이를 분석 | ML2에 GBDT를 포함하고 입력·탐색 예산을 맞춘다. 중간 규모 tabular의 관찰을 우리 순차·물리 적분 문제의 승자 예측으로 사용하지 않는다. [논문](https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) |
| **McElfresh et al. (NeurIPS 2023)**, *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* | 176 데이터셋·19 알고리즘에서 많은 경우 모델 계열 차이보다 데이터 특성·가벼운 튜닝의 중요성을 확인 | NN/GBDT 양쪽에 같은 탐색 기회를 주고 총 시간을 공개한다. 특정 계열의 보편적 우월성은 주장하지 않는다. [논문](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract.html) |
| **Viering & Loog (TPAMI 2023; 공개본 2021/22)**, *The Shape of Learning Curves: A Review* | 학습량에 따른 일반화 곡선과 추정 문제를 정리. 데이터 증가가 항상 개선을 보장하지 않음 | ML1의 25/50/100% 비교를 정당화하되, 세 점만으로 보편적 scaling law나 필요한 최종 데이터량을 단정하지 않는다. 중첩 경기 표본은 우리의 실험 설계 제안이다. [저자 공개본](https://arxiv.org/abs/2103.10948), [대학 공개 PDF](https://pure.tudelft.nl/ws/portalfiles/portal/153554064/The_Shape_of_Learning_Curves_A_Review.pdf) |
| **Guo et al. (ICML 2017)**, *On Calibration of Modern Neural Networks* | 분류의 정확도와 확률 보정은 다르며, 한 개 온도 파라미터를 이용한 사후 보정이 유효한 경우를 제시 | ML0부터 raw/온도 보정/빈도 혼합 결과를 함께 보고 CAL에서만 보정한다. 물리 적분·앙상블·온도 적용 순서도 고정한다. 이 논문이 우리 delivery 적분 방식까지 검증한 것은 아니다. [논문](https://proceedings.mlr.press/v70/guo17a.html) |

방법론 4편은 공식 초록·서지와 공개 원문의 해당 설명을 확인한 수준이며 저자 코드를 실행·감사하지 않았다. 이 자료들은 실험 설계의 근거이며 야구 성능 수치를 제공하는 직접 기준선은 아니다.

## 5. ML0에서 먼저 고정할 재현 차이

| 항목 | 확인 내용 | 사양에 반영할 것 |
|---|---|---|
| 원문 출력 공간 | SmartPitch 4, Melville 9, Takamido 2. Kneita는 10이어도 클래스 의미가 다름 | 원래 과제와 공통 10-class 표를 별도로 둔다. 병합·제외 분모를 기록한다. |
| 입력 시점 | 여러 논문이 현재 공 실제 물리를 사용 | 원문 사후 조건부 예측과 투구 전 적분/직접 예측을 분리한다. |
| Melville 구조 | 상태에 투구·결과의 변환을 **더하는** 자체 RNN | LSTM을 구현하고 충실 재현이라고 부르지 않는다. 당시 공의 결과를 당시 예측에 넣지 않는다. |
| Takamido positive class | 아래 산술상 보고 P/R/F1은 swing-out 기준 | 두 클래스 지표·라벨 순서·혼동행렬을 함께 저장. AUC는 예측 score 방향까지 맞춘다. |
| Takamido threshold | validation F1 최대로 선택한다고 §2.5 명시 | 원문 사실을 남기고, 우리 별도 CAL 적용을 이탈 목록에 기록한다. |
| Takamido 후보 물리 | 2025 시즌 평균으로 2025 반사실 후보를 구성 | 투구 전 적응에서는 TRAIN 또는 시점 이전의 동일 delivery 사용. |
| 기존 seed | 재현 실행기는 `{42,43,44,45,46}`, 새 계획은 탐색 `{0,1,2}` | ML0 기존 산출물 재현 seed와 새 실험 seed를 구분한다. |
| 학습량 기준 | 기존 NN 표본은 전체 TRAIN의 임의 25만 행에 코호트 선발 투구를 추가한 뒤 중복 제거 | NN 276,821행을 전체 TRAIN 100%라고 표기하지 않는다. ML1 경기 표본을 새로 만들면 100% 대조군도 같은 방법으로 재실행한다. |

Takamido Table 2를 읽어 계산한 값(새 데이터 평가가 아님):

| 실제 / 예측 | in-play | swing-out |
|---|---:|---:|
| in-play | 3,774 | 2,827 |
| swing-out | 1,194 | 8,692 |

- swing-out 양성: precision `8692/(8692+2827)=.75458`, recall `8692/(8692+1194)=.87922`, F1 `.81215`.
- in-play 양성: precision `.75966`, recall `.57173`, F1 `.65243`.
- 원문이 기술한 모델 label=1과 보고 지표의 양성 클래스가 다르다는 **산술 확인**이다. 저자 코드의 실제 평가 구현은 추가 감사 대상이다. [PDF p.18, Table 2](https://arxiv.org/pdf/2606.17345).

샘플링·seed 근거: [기존 시간 분할 실행기](../../experiments/pitchmdp/scripts/run_temporal_blend.py)의 `SEEDS`, `samples_for`.

## 6. 문헌을 반영한 실행 우선순위 제안

ML0·ML1은 기존 인수인계의 순서를 유지한다. 새 논문이 발견됐다는 이유로 첫 데이터 실험에 모든 구조를 추가하지 않는다. 아래 선택은 **권고**이며, 최종 수치·ID·config는 `ML-BENCHMARK-v1` 작성 때 확정한다.

| 순서 | 실행할 비교 | 문헌 연결 |
|---|---|---|
| ML0 | 기존 빈도·MLP·Transformer의 저장 예측/체크포인트 재현. 동일 행·라벨·물리 표본·온도/혼합 순서·seed 확인 | SmartPitch 확률 평가, Guo 보정, Melville 공 번호 진단 |
| ML1 첫 묶음 | 같은 기존 MLP를 고정하고 중첩 **경기** 표본 25/50/100%. 같은 DEV에서 학습 곡선 | Viering & Loog. backbone보다 데이터량 효과를 먼저 식별 |
| ML1 별도 묶음 | 같은 수의 학습 표본에서 최근 기간/누적 기간 비교 또는 같은 표본의 균등/최근 가중 비교 | Mott의 recency. 양과 시점 구성을 동시에 바꾸지 않음 |
| ML2 | 선형·MLP·Transformer·GBDT·Melville 자체 recurrent의 공통 정보 비교 | SmartPitch, Melville, Grinsztajn, McElfresh |
| ML3 | 오차 근거가 있을 때 타석 간 이력 또는 최근 form 중 한 축 | Kneita, Heaton & Mitra. 긴 이력/자기지도/다중과제는 각각 별도 가설 |
| CV 이후 | 목표 위치·실제 도달·정책 효과 평가 | Douglas, Nakahara |

ML1 사양에서 특히 필요한 선택:

1. **학습량의 분모:** 코호트는 평가 대상을 고정하는 규칙이며 학습 데이터를 그 6명으로만 제한한다는 뜻이 아니다. 기존 학습 모집단과 코호트 투구의 가중 정도를 고정한다.
2. **기간 구성:** 각 기간에 걸친 경기 순서를 결과와 무관한 seed/hash로 미리 정해 25%⊂50%⊂100%를 만든다. 같은 경기와 타석은 통째로 선택하고 실제 경기/투구 수·기간별 비중을 보고한다. 단순히 앞 25% 기간을 자르면 데이터량과 최근성이 섞인다.
3. **보조 데이터의 범위:** 정규화·범주·타자 과거 성향·빈도표·delivery도 각 subset으로 다시 추정할지, 전체 TRAIN으로 고정할지 명시한다. 후자는 ‘NN 지도학습 표본량 효과’, 전자는 ‘학습 파이프라인 전체 증거량 효과’다. 첫 비교는 보조 데이터 추정을 전체 TRAIN으로 고정한 NN 표본량 효과로 좁히는 것을 제안하되 전체 TRAIN을 계속 사용하는 항목을 공개한다.
4. **학습 비용:** 같은 epoch에서는 큰 표본이 더 많은 update를 받는다. 동일 조기 종료 규칙·최대 예산과 실제 epoch/update/시간을 함께 기록해 데이터량 효과의 해석 범위를 명시한다.
5. **선정:** 고정 NLL·Brier, 클래스/투수별 지표와 지원률, 짝지은 경기 불확실성으로 판단한다. CAL은 동일하게 두되 각 학습량 모델을 같은 절차로 보정하고 보정 전·후를 모두 보고한다. 2026·최종셋은 사용하지 않는다.

현재 읽기 우선순위는 **SmartPitch·Melville → Kneita → 학습 곡선·보정 방법론 → Mott의 recency → 나머지 정보 확장·정책 평가 문헌**이다. 기존 0/5구 비교의 우열 미확정으로 장기 이력까지 부정할 수 없지만, 긴 이력이 유리하다는 결론도 아직 없다.

## 7. 확인 범위와 보존 기록

- 기존 5편, 추가 야구 6편, 방법론 4편을 역할별로 정리했다. 제목 검색과 기존 논문의 참고문헌을 따라 찾은 목적 지향 조사이며 전수 systematic review·최초성 증명은 아니다.
- SmartPitch·Melville·Kneita는 공식 PDF를 임시 경로에서 추출했다. 다른 출처는 arXiv 본문/저자 소스/공식 출판 페이지를 확인했다. 논문 PDF·저자 코드를 저장소에 복제하지 않았다.
- 임시 작업 경로: `/tmp/pitcheezy-literature-20260924/`. PDF 추출 도구도 여기에만 설치했으며 프로젝트 가상환경은 변경하지 않았다. 임시 파일은 영속 아티팩트가 아니며 출처 링크로 다시 받을 수 있다.
- SmartPitch PDF SHA256: `53ffe9640836d2eec1eb0ae8a2c50b3350315acf0df15c39dd3c40fb5318c715`.
- Melville PDF SHA256: `b75bc3c81a823df7ab6745a99e7a5f00a96ea12c1135d98b449ab52b83e3ecb9`.
- Kneita PDF SHA256: `55cc6a9b00b792f668356a1b32ae49a5c81c431f4bd7c39bde1a7689e2ab52bb`.
- 기존 OPE 참고문헌 Jiang & Li / Voloshin et al. / CQL은 [기존 목록](../../experiments/pitchmdp/LITERATURE.md)에 보존했다. 이번 초점인 예측 모델·데이터 실험의 새 기준선에는 넣지 않았다.
- 미완료: 저자 코드 충실 재현, 개별 config·HPO·예산 사전 고정, 데이터/노출 manifest 감사, ML0·ML1 실제 실행. 연구의 예측 개선과 서비스 추천 정책 채택은 계속 구분한다.

2026년 충북대학교 소프트웨어학부 졸업작품연구과제

| AI 투구 추천 야구 관전 보조 시스템(Pitcheezy) |
| --- |
| 1차 스프린트 - 개발 명세서 |

2026년 9월 20일

문서번호 : 2026-Pitcheezy-Doc-006-1

소    속 : 충북대학교 소프트웨어학과

팀    명 : Pitcheezy

팀    원 : 한강민, 송동선, 조현준

교    수 : 아지즈 교수님

목   차

1. 프로젝트 개요 	1

2. 태스크 목록 	2

3. 기술 명세 	2

3.1 클래스 명세 	4

3.2 핵심 알고리즘 명세 	4

3.3 DB 스키마 명세  	4 

3.4 API 명세  	4 

3. 사용자 인터페이스  	4

4. 기타 구현 참고 사항 	9
---

1. 프로젝트 개요 

1.1 시스템명

AI 투구 추천 야구 관전 보조 시스템(Pitcheezy)

1.2 시스템 개요  

시스템은 MLB Statcast 투구 기록과 MLB 공식 경기 피드를 바탕으로, 투구 전 상황(볼카운트·주자·아웃·투수·타자)에서 수비 팀의 기대 승리확률을 가장 높이는 구종과 목표 구역 상위 3개를 추천한다.

사용자는 중계를 보면서 보조 화면(‘두 번째 전광판’)으로 추천을 확인하고, 실제 투구(구종·구속·도착 위치·판정)와 추천의 차이를 비교한다.

시스템은 추천 근거(선택 비중, 지원 여부, 추천 출처)를 함께 표시하고, 지원하지 않는 투수·상황은 숨기지 않고 ‘미지원’으로 알린다.

예측 성능(결과 확률 모델의 NLL·보정)과 추천 정책 효과(오프라인 정책 평가)는 분리해 검증하며, 2026 시즌 자료는 사전 등록한 정책 평가에만 쓴다.

2. 태스크 목록

1차 스프린트 백로그의 태스크 목록과 완료율은 [표 1]과 같다.

| US_ID | 스토리(요약) | TK_ID | 테스크 명 | 담당자 | 종료일  | 완료% |
| --- | --- | --- | --- | --- | --- | --- |
| US-29 | Statcast 수집 | TK-01 | Statcast 월 단위 수집기(statcast_fetch) 구현, dtype 고정 | 송동선 | 09/12 | 100 |
|  |  | TK-02 | 수집 장부·sha256·재시도 기록 | 송동선 | 09/12 | 100 |
| US-31 | 데이터 버전·홀드아웃 고정 | TK-03 | 데이터 버전 장부(data/versions.md) 작성, d20260911-s2325 | 송동선 | 09/12 | 100 |
| US-32 | 데이터 버전·홀드아웃 고정 | TK-04 | 2026 홀드아웃 분리와 학습 수집 거부 게이트 | 송동선 | 09/14 | 100 |
| US-33 | 결과 확률 모델 학습·보정 | TK-05 | 인터페이스 계약 코드화(states·grid·outcomes·tensor·value)와 계약 테스트 | 송동선 | 09/17 | 100 |
|  |  | TK-06 | RE24·dRE24 계산(re24.compute)과 자산 저장 | 송동선 | 09/16 | 100 |
|  |  | TK-07 | 선발 투수 풀·타자 k-means 군집 | 송동선 | 09/16 | 100 |
|  |  | TK-08 | 계층 디리클레 전이 모델 count.fit·smoothing | 송동선 | 09/17 | 100 |
|  |  | TK-09 | 공유 MLP 전이 모델 neural.SharedNet | 송동선 | 09/19 | 100 |
| US-34 | 결과 확률 모델 학습·보정 | TK-10 | 홀드아웃 NLL·ECE 평가(holdout_metrics)와 α 선택 | 송동선 | 09/17 | 100 |
| US-35 | 추천 정책·정책 평가 | TK-11 | 가치 반복·완화 정책(policy.vi) | 송동선 | 09/17 | 100 |
| US-36 | 추천 정책·정책 평가 | TK-12 | 행동 정책 추정·SNIPS·경기 부트스트랩(ope) | 송동선 | 09/17 | 100 |
|  |  | TK-13 | 실험 러너 experiment.py·configs·results JSON | 송동선 | 09/17 | 100 |
|  |  | TK-14 | EXP-P0-001~010 실행과 P0 채택(D27) | 송동선 | 09/18 | 100 |
|  |  | TK-15 | B1(SmartPitch)·B2(Takamido) 축소 재구현 | 송동선 | 09/20 | 100 |
|  |  | TK-16 | EXP-P1 맥락 v1·신경망 판정(D29~D39) | 송동선 | 09/20 | 100 |
| US-37 | 동결·서비스 연결 | TK-17 | 사전 등록 규칙·결정 기록(docs/decisions.md, experiments.md) | 송동선 | 09/15 | 100 |
| US-38 | 동결·서비스 연결 | TK-18 | 산출물 sha256 기록·검증(_io.write_sha256/verify_sha256) | 송동선 | 09/17 | 100 |

 

[표 1] 태스크 목록

3. 기술 명세

3.1 클래스 명세   

1차 스프린트에서 개발·변경한 클래스(모듈)의 구조는 [그림 1]과 같다. 파이썬 모듈 수준 함수 묶음은 클래스에 준해 명세한다.

[그림 1] 클래스 다이어그램

[1] 클래스 : C-001 RE24Table (interfaces/re24.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-001 | 멤버<br>변수 | RE24: float64[24] | 주자·아웃 24상태의 기대 득점 | Public |  |
|  |  | dRE24: float64[8,24] | 종결 결과 8종별 기대 득점 변화 | Public |  |
|  |  | meta: dict | re24_version·data_version·season_window·compute_commit·n_plate_appearances | Public |  |
|  | 멤버<br>함수 | save(dir) | RE24.npy·dRE24.npy·meta.json·sha256.txt 저장 | Public |  |
|  |  | load(dir, check_hash=True) | sha256 검증 후 로드 | Public |  |
|  |  | validate(t) -> list[str] | 형상·dtype·NaN·음수·필수 키 검사 | Public |  |

[2] 클래스 : C-002 TransitionTensor (interfaces/tensor.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-002 | 멤버<br>변수 | P: float32[P,S,A,O] | 투수·상태·행동별 결과 확률(행합 1±1e-5) | Public |  |
|  |  | valid: bool[P,S,A] | 지지(레퍼토리 100구 이상) 마스크 | Public |  |
|  |  | n_obs: int32[P,S,A] | 학습 관측 수 | Public |  |
|  |  | pitchers·states·outcomes: DataFrame | 축 설명표 | Public |  |
|  |  | meta: dict (16키) | seed·train_commit·holdout_nll·holdout_ece 등 | Public |  |
|  | 멤버<br>함수 | n_pitchers, K, C, context_kind | 축 크기 속성 | Public |  |
|  |  | save(dir) / load(dir, check_hash, mmap) | npy·parquet·sha256 저장·로드 | Public |  |
|  |  | expected_shape(n_pitchers, K, C) | 기대 형상 계산 | Public |  |

[3] 클래스 : C-003 ValueBundle (interfaces/value.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-003 | 멤버<br>변수 | Q: float32[P,S,A] | 행동 가치(단위 RE24) | Public |  |
|  |  | V: float32[P,S] | 상태 가치 = max Q | Public |  |
|  |  | policy: float32[P,S,A] | 완화 정책 | Public |  |
|  |  | meta: dict (11키) | transition_sha256·gamma·relax·seed 등 | Public |  |
|  | 멤버<br>함수 | save(dir) / load(dir) | 저장·로드 | Public |  |
|  |  | lookup(Q, pitcher, state, pitch, x, z, mode) | 위치 → 격자 스냅 조회 | Public |  |

[4] 클래스 : C-004 LeagueLayers (transition/smoothing.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-004 | 멤버<br>변수 | cs, group, mask_s, mask_c | 카운트·구종 군 인덱스와 규칙 마스크 | Public |  |
|  |  | L0: [S,A,O], Lcg | 리그 상위 층 분포 | Public |  |
|  | 멤버<br>함수 | __init__(n_sa, count_of_state, group_of_action, …, alpha, rule_mask) | 리그 층 계산 | Public |  |
|  |  | pitcher_layer(n2, lg, alpha_pitcher) | 투수 층 L2 = (n₂ + α_p·m)/(N + α_p) | Public |  |
|  |  | smooth_hierarchical(…) | 계층 평활 결합 | Public |  |

[5] 클래스 : C-005 SharedNet (transition/neural.py, nn.Module)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-005 | 멤버<br>변수 | emb_count·emb_base_out·emb_cluster·emb_pitch·emb_loc | 축별 임베딩 | Public |  |
|  |  | emb_pitcher(16)·emb_pitcher_pitch | 투수·투수×구종 임베딩 | Public |  |
|  |  | mlp: 256×2 | 공유 MLP | Public |  |
|  | 멤버<br>함수 | forward(p, c, b, k, g, l, x) | 규칙 마스크 위치를 −inf로 막은 결과 로짓 | Public |  |
|  |  | embed_l2() | 임베딩 L2 정규화 항 | Public |  |
|  |  | train() · predict_tensor() · fit() | AdamW 학습(조기 종료 patience 3)·텐서 생성 | Public |  |

[6] 클래스 : C-006 BehaviorFactors (ope/behavior.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-006 | 멤버<br>변수 | t_psg: [P,S,G] | 구종 군 선택 확률 | Public |  |
|  |  | l_pcgl: [P,NC,G,L] | 구종·카운트별 위치 확률 | Public |  |
|  |  | cid_of_state, floor | 카운트 인덱스·하한 | Public |  |
|  | 멤버<br>함수 | block(lo, hi) -> π_b | 투수 블록의 행동 정책 | Public |  |
|  |  | behavior_factors() · crossfit_logged() | 계층 평활·game_pk%5 교차 적합 | Public |  |
|  |  | tilt(pb, Q, support, tau) | 평가 정책 π_e ∝ π_b·exp(Q/τ) | Public |  |

[7] 클래스 : C-007 Experiment (experiment.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-007 | 멤버<br>변수 | cfg, seed, K, C, context_kind, arch | 설정 파일 값 | Public |  |
|  |  | commit, re24, pool, pitcher_index, clusters, timing | 실행 맥락과 자산 | Public |  |
|  | 멤버<br>함수 | pitches(season) · sid(df) | 투구 표 로드·상태 id | Public |  |
|  |  | stage_transition() · stage_value() · stage_ope() | 단계 실행과 저장 | Public |  |
|  |  | run() · load_config() · aggregate() | 전체 실행, 2026 학습 거부, 시드 집계 | Public |  |

[8] 클래스 : C-008 statcast_fetch (data/statcast_fetch.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-008 | 멤버<br>변수 | SeasonResult / LedgerRow / RunResult | 수집 결과·장부 행·실행 결과 dataclass | Public |  |
|  |  | FetchError · SchemaError · DtypeError | 예외 | Public |  |
|  | 멤버<br>함수 | fetch_season(season, …) | 월 단위 pybaseball.statcast 수집, dtype 고정 | Public |  |
|  |  | run_fetch(cfg) | 버전 id·sha256·행 수 장부 기록 | Public |  |
|  |  | verify(ledger) | 장부 검증, 2026 학습 수집 거부 | Public |  |

3.2 핵심 알고리즘 명세 

3.1절의 멤버 함수 중 처리 로직이 중요한 것의 행위를 명세한다.

[1] 메소드 : fit( ) @ transition.count — 계층 디리클레 전이 모델 학습

1. 학습 투구(2023~24)의 (투수 p, 상태 s, 행동 a, 결과 o) 관측 수 n[p,s,a,o]를 투수별 bincount로 센다.

2. 리그 층 : base(카운트) → Lca(카운트·행동) → L0(상태·행동) 순으로 (n + α·상위층)/(N + α)로 평활한다. 구종 군 층 Lcg도 만든다.

3. 투수 층 : L1(투수·카운트·구종) ← Lcg, 기대 분포 m ∝ L0·L1/Lcg, L2 = (n₂ + α_p·m)/(N + α_p).

4. 규칙 마스크(3볼에 볼 불가, 2스트라이크에 스트라이크 불가) 셀은 0으로 두고 재정규화한다. valid = 그 구종의 학습 투구 ≥ 100.

5. α ∈ {20, 50, 100}마다 2025 홀드아웃 NLL·ECE를 재고 NLL이 가장 낮은 α로 2023~25 전체를 다시 적합해 TransitionTensor로 저장(sha256)한다.

[2] 메소드 : value_iteration( ) @ policy.vi — 타석 MDP 가치 반복

1. 보상표 R[s,o] = −dRE24[o, base_out(s)] (종결 결과), 비종결 결과는 0. 다음 상태표 next(s,o)는 카운트 규칙으로 만든다.

2. 투수 16명 블록마다 반복한다 : Q = Σ_o P[p,s,a,o]·(R[s,o] + V[next(s,o)]),  V = valid 행동 중 max Q.

3. |ΔV| < 1e-9이거나 200회에 이르면 멈춘다.

4. policy = softmax(Q/τ) | top-k | greedy (valid 위에서만) 로 완화하고 ValueBundle로 저장한다.

[3] 메소드 : estimate( ) @ ope.ips — 오프라인 정책 평가(SNIPS)

1. 행동 정책 π_b = π_type(구종|상태,투수) × π_loc(위치|구종,카운트,투수)를 계층 평활·game_pk%5 교차 적합으로 추정한다.

2. 평가 정책 π_e = tilt : π_b·exp(Q/τ), 지지 = valid ∩ 평가 시즌 20구 이상 구종.

3. 투구별 비율 ρ_t = min(π_e/π_b, 20), 보상 r = −ΔRE24(타석, 한 투수가 끝까지 던진 타석만).

4. 1스텝 가중치 w = Σρ, 궤적 가중치 w = Πρ,  SNIPS = Σw·r / Σw.  ESS와 경기 클러스터 부트스트랩(1000회) 2.5/97.5% 구간을 보고한다.

5. 보조로 1스텝 DR(제어변량 q̂_b)과 궤적 DR/WDR을 계산해 results/{ID}.json에 기록한다.

3.3 DB 스키마 명세  

1차 스프린트는 DB를 만들지 않았다. 파일 기반 저장 스키마(parquet·npy·JSON)를 테이블에 준해 명세한다.

[1] 테이블(파일) : 투구 표 캐시 runs/_cache/pitches_{data}_{pool}_{season}.parquet (20열)

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| season | int16 |  |
| game_pk, at_bat_number, pitch_number, inning | int64 / int32 / int16 / int16 | 투구 식별 |
| inning_topbot, stand | category |  |
| pitcher, pitcher_idx, batter | int64 / int32 / int64 | 선수 id·인덱스 |
| count_id, base_out_id, cluster_id | int16 | 상태 축 |
| pitch_id, loc_id, action_id, outcome_id | int16 | 행동·결과 축 |
| terminal | bool | 타석 종결 여부 |
| bat_score, post_bat_score | int16 | 득점 보상 계산용 |

[2] 테이블(파일) : data/versions.md 데이터 버전 장부 (11열)

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| version_id (예 d20260911-s2325) | str | 수집일·시즌 범위 |
| season, file, date_range | int / path / str |  |
| rows, sha256 | int / hex | 행 수·해시 |
| pybaseball_version, pyarrow_version | str | 2.2.7 / ≥14 |
| fetch_commit, frozen, fetch_cutoff | hex / bool / date | 수집 커밋·동결 여부 |

[3] 테이블(파일) : results/{ID}.json 실험 결과

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| id, params, seed | str / dict / int | 실험 ID·설정 |
| transition.holdout_metrics | dict (nll, ece, ece_hr, accuracy_top1) | 예측 성능 |
| ope.policies.{point_snips, ci_low_min, ci_high_max, ess_frac} | dict | 정책 평가 |
| data_version, commit | str / hex | 재현 정보 |

3.4 API 명세 

외부 시스템·다른 서브시스템과의 인터페이스만 명세한다.

 

| 인터페이스명  | 클래스명  | 인터페이스 대상  | 설명  |
| --- | --- | --- | --- |
| pybaseball.statcast(start, end) | C-008 statcast_fetch | Baseball Savant (Statcast CSV) | 월 단위 수집, 재시도 3회, timeout (30,120), game_type R만 남김 |
| wandb.log(metrics) | C-007 Experiment | Weights & Biases | 실험 지표·설정 기록 |

[표 2] API 명세

4. 사용자 인터페이스(GUI)  

1차 스프린트는 연구 파이프라인(명령줄 실행)만 개발해 사용자 인터페이스가 없다 — 해당 없음.

 

5. 기타 참고 사항  

오픈소스 : pybaseball 2.2.7 (MIT) : Statcast 수집, pandas·pyarrow·NumPy (BSD) : 자료 처리, PyTorch 2.14 (BSD) : 전이 모델·시퀀스 모델 학습, scipy (BSD) : 추천 어댑터 수치 계산, pytest 9.1 (MIT) : 계약 테스트, Weights & Biases : 실험 기록.

재사용 코드 : 선행 연구 B1(SmartPitch)·B2(Takamido)는 논문 설명을 바탕으로 직접 축소 재구현했고 원저자 코드는 가져오지 않았다(docs/baselines.md).

다음 스프린트 고려 : 결과 확률 모델을 서비스에 연결하려면 투구 전 입력 허용 목록·불변 추천 저장·공개 단계 제어가 필요하다. 승리확률(WE) 기반 pitchmdp 계열과 RE24 계열의 단위 차이를 문서에 밝힌다.

<끝>
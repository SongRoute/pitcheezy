2026년 충북대학교 소프트웨어학부 졸업작품연구과제

| AI 투구 추천 야구 관전 보조 시스템(Pitcheezy) |
| --- |
| 2차 스프린트 - 개발 명세서 |

2026년 10월 3일

문서번호 : 2026-Pitcheezy-Doc-006-2

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

2차 스프린트 백로그의 태스크 목록과 완료율은 [표 1]과 같다.

| US_ID | 스토리(요약) | TK_ID | 테스크 명 | 담당자 | 종료일  | 완료% |
| --- | --- | --- | --- | --- | --- | --- |
| US-03 | 관전 시작 | TK-19 | FastAPI 관전 서버(세션 생성·진행·공개) 구현 | 송동선 | 09/23 | 100 |
|  |  | TK-20 | SQLite Store(불변 추천·UPDATE 금지 트리거·작업 큐) | 송동선 | 09/23 | 100 |
| US-04 | 관전 시작 | TK-21 | React 재생 화면(카탈로그·존·추천 카드·세션 복원) | 송동선 | 09/24 | 100 |
| US-05 | 추천 후보 표시 | TK-22 | Recommender·투구 전 허용 목록 입력·캐시 키 | 송동선 | 09/25 | 100 |
|  |  | TK-23 | pitchmdp 런타임(solve_pa·WE·진루 분포) 연결 | 송동선 | 09/25 | 100 |
| US-06 | 추천 후보 표시 | TK-24 | 지원 범위(레퍼토리 20구·ESS≥20) 판정과 미지원 표시 | 송동선 | 09/26 | 100 |
| US-10 | 실제 투구 공개 | TK-25 | 실제 투구 공개(reveal)와 추천 비교 표시 | 송동선 | 09/24 | 100 |
| US-11 | 존 그림 | TK-26 | 포수 시점 존 그림(StrikeZone)·목표 구역 입력 | 송동선 | 09/24 | 100 |
| US-13 | 상황 머리글·선수 카드 | TK-27 | 상황 머리글·선수 카드·쉬운 말 표기(D138) | 송동선 | 10/02 | 100 |
| US-19 | 사건 기여도 카드 | TK-28 | 사건 분해 event_analysis 뼈대(%p 3성분+잔여) | 송동선 | 09/26 | 100 |
| US-21 | 실제 속도 재생 | TK-29 | 저장 경기 실제 속도 재생·849843 검증(D134) | 송동선 | 10/02 | 100 |
| US-23 | 피드 폴링·장부 기록 | TK-30 | 실시간 피드 폴링 초기 구현·849844 첫 실경기 수신(D144) | 송동선 | 10/02 | 100 |
| US-30 | Statcast 수집 | TK-31 | 피드 이벤트와 Statcast 행의 투구 식별자 연결 | 송동선 | 10/02 | 100 |
| US-33 | 결과 확률 모델 학습·보정 | TK-32 | ML 실험 매트릭스(G0·G15 mix, EXP-P11/P16)와 예측 기준 결정(D145) | 송동선 | 10/03 | 100 |
| US-34 | 결과 확률 모델 학습·보정 | TK-33 | 경기 bootstrap 10,000회 NLL·Brier·ECE 비교 기록 | 송동선 | 10/03 | 100 |
| US-36 | 추천 정책·정책 평가 | TK-34 | 2026 정책 평가 사전 등록·1회 실행(MLB-2026-OPE-v1, D131) | 송동선 | 10/01 | 100 |
| US-37 | 동결·서비스 연결 | TK-35 | G0 동결(sha256 목록)·변조 거부 테스트 | 송동선 | 10/03 | 100 |
| US-42 | CV 추정 결과 검토·연결 | TK-36 | CV 의도 계약 검사(747139·849843)·setup API 연결(D150·D151) | 조현준·송동선 | 10/02 | 100 |

 

[표 1] 태스크 목록

3. 기술 명세

3.1 클래스 명세   

2차 스프린트에서 개발·변경한 클래스(모듈)의 구조는 [그림 1]과 같다. 파이썬 모듈 수준 함수 묶음은 클래스에 준해 명세한다.

[그림 1] 클래스 다이어그램

[1] 클래스 : C-009 ObserverService (observer_app/service.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-009 | 멤버<br>변수 | dataset: DemoDataset | 재생 자료 | Public |  |
|  |  | recommender: Recommender | 투구 전 추천 | Public |  |
|  |  | store: Store | 세션·추천 저장소 | Public |  |
|  |  | inning_decisions: InningDecisionRepository | 교체 직전 기록 | Public |  |
|  | 멤버<br>함수 | catalog() | 경기·타석 목록 | Public |  |
|  |  | create(game_id, pa_id) / get(sid) | 세션 생성·조회 | Public |  |
|  |  | advance(sid, revision) | 다음 공 공개, revision 불일치 시 409 | Public |  |
|  |  | manual_intent(sid, revision, zone_id) | 타석 종료 후 목표 구역 입력 | Public |  |
|  |  | _recommend() · _view() · _ensure_event_result() | 불변 추천·화면 본문·사건 결과 | Public |  |

[2] 클래스 : C-010 Store (observer_app/store.py, SQLite WAL)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-010 | 멤버<br>변수 | sessions, recommendations, jobs | 세션·투구별 추천(PK session_id+pitch_id)·작업 큐 | Public |  |
|  |  | event_results, inning_decisions | 사건 결과·교체 기록(UPDATE·DELETE 금지) | Public |  |
|  | 멤버<br>함수 | transaction(write) | 쓰기 트랜잭션 | Public |  |
|  |  | save_recommendation(sid, pitch_id, payload) | 최초 1회만 저장 | Public |  |
|  |  | save_event_result(…) | 연결 sha 검사 후 저장 | Public |  |
|  |  | enqueue() · claim(lease_seconds, max_attempts) · finish() | 작업 큐 임대·완료 | Public |  |

[3] 클래스 : C-011 Recommender · PrePitchInput · CandidateAction (recommender.py, recommendation_adapter.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-011 | 멤버<br>변수 | engine, identity, model_sha256 | 동결 모델 번들 | Public |  |
|  |  | cache, runtime_kind, last_latency_ms | 캐시·실행 종류·지연 | Public |  |
|  |  | PrePitchInput(frozen) | 투구 전 허용 목록 입력 | Public |  |
|  |  | CandidateAction(pitch_type, zone_id, value, delta_pp, support) | 후보 | Public |  |
|  | 멤버<br>함수 | recommend(pitch, pa) | 투구 전 추천 상위 3 | Public |  |
|  |  | evaluate_pre_pitch(request) | 후보·근거 계산 | Public |  |
|  |  | _compute(request, bounds, repertoire, key) | 캐시 키 = sha(모델·요청·존·레퍼토리) | Public |  |
|  |  | ObservedDeliveryKernel.weights(xz, targets, σ=0.45) | 가우시안 커널 가중·ESS·질량 | Public |  |

[4] 클래스 : C-012 PlanResult · solve_pa (runtime_src/pitchmdp/planner.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-012 | 멤버<br>변수 | values, q_values, baseline_values | 카운트·상태별 가치 | Public |  |
|  |  | myopic_q_values, policy, diagnostics | 근시 가치·정책·진단 | Public |  |
|  | 멤버<br>함수 | solve_pa(probabilities[4,3,nprev,na,10], terminal_values, action_next_prev, baseline_policy) | 타석을 카운트 역방향으로 풀고 2스트라이크 파울 고리는 선형해 | Public |  |
|  |  | topk(b, s, prev, k) | 상위 k 후보 | Public |  |

[5] 클래스 : C-013 GameState · WinExpectancy · EmpiricalAdvancement (runtime_src/pitchmdp/game.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-013 | 멤버<br>변수 | GameState: inning, half, outs, bases, score_diff | 경기 상태 | Public |  |
|  |  | WinExpectancy: 규제 로지스틱 계수 | 수비팀 승리확률(볼카운트 미반영) | Public |  |
|  |  | EmpiricalAdvancement: 진루 분포표 | 결과별 주자 진루 | Public |  |
|  | 멤버<br>함수 | we.predict(state) -> p(home win) | 승리확률 | Public |  |
|  |  | terminal_values(state, we, adv) | 종결 결과별 수비팀 승리확률 | Public |  |

[6] 클래스 : C-014 event_analysis · intent (observer_app/event_analysis.py, intent.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-014 | 멤버<br>변수 | reference, plan, execution, observed | 수비팀 승리확률 4값 | Public |  |
|  |  | status ∈ full · partial | 의도 근거 유무 | Public |  |
|  | 멤버<br>함수 | analyze_event(…) | total=100·(observed−reference) %p, strategy·execution·outcome 분해, 합 검증 | Public |  |
|  |  | failed_event(…) | 분해 불가 시 기록 | Public |  |
|  |  | validate_intent_estimate(payload) | CV 추정 결과 계약 검사(픽셀→존→ft→zone9) | Public |  |

[7] 클래스 : C-015 DemoDataset (observer_app/dataset.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-015 | 멤버<br>변수 | games[{plate_appearances[{pitches[{state, request, actual}]}]}] | 재생 자료 JSON | Public |  |
|  |  | dataset_manifest (sha256) | 무결성 | Public |  |
|  | 멤버<br>함수 | catalog() · get(game_id, pa_id) | 목록·타석 로드 | Public |  |
|  |  | reveal(pitch, rec, bounds) | 실제 투구 공개 본문 | Public |  |

3.2 핵심 알고리즘 명세 

3.1절의 멤버 함수 중 처리 로직이 중요한 것의 행위를 명세한다.

[1] 메소드 : _compute( ) @ C-011 Recommender — 투구 전 추천 계산

1. 요청은 투구 전 정보(허용 목록)만 받고, 캐시 키 = sha(모델 identity, 요청, 존 경계, 레퍼토리)로 재계산을 피한다.

2. 구종 후보 = 최근 90일 20구 이상 던진 구종. 12 카운트 × 구종마다 학습 실투구 표본을 뽑아 시퀀스 신경망 앙상블의 softmax(로짓/T) 결과 확률을 얻는다.

3. 9구역 목표마다 가우시안 커널(σ = 0.45 ft)로 가중합하고 빈도 기준선과 혼합한다. 지원 = 모든 카운트에서 ESS ≥ 20이고 질량 ≥ 0.01.

4. terminal = 종결 사건별 수비팀 승리확률(WE × 진루 분포). solve_pa로 카운트를 거꾸로 풀어 카운트별 상위 3후보 {구종, 구역, value, Δ%p, ESS}와 근거 분해를 돌려준다.

[2] 메소드 : analyze_event( ) @ C-014 — 사건 승리확률 변화 분해

1. total = 100·(observed − reference) [%p].  strategy = plan − reference,  execution = execution − plan,  outcome = observed − execution.

2. 세 성분이 모두 있으면 합이 total과 같아야 하고 다르면 오류로 기록한다. 투구 전 의도 근거가 없으면 status = partial로 두고 남는 차이를 배분하지 않는다.

3. abs_share = |성분| / Σ|성분| 로 요인 비중을 표시한다(화면 표시는 검증 게이트 뒤로 미룸).

[3] 메소드 : advance( ) @ C-009 ObserverService — 공개 단계 제어

1. 세션의 revision을 낙관적 잠금으로 확인한다(불일치 시 409). 2. 현재 커서의 투구에 대해 추천이 없으면 투구 전 정보로 계산해 Store에 최초 1회 저장한다.

3. 실제 투구를 공개(reveal)하고 커서를 다음 공으로 옮긴다. 4. 타석이 끝나면 사건 결과를 작업 큐에 넣고 화면 본문(View)을 돌려준다.

3.3 DB 스키마 명세  

시스템은 관계형 DB 서버를 쓰지 않는다. 임베디드 SQLite 파일과 parquet·npy·JSON 파일의 스키마를 테이블에 준해 명세한다.

[1] 테이블 : observer.sqlite3 — sessions / recommendations / jobs / event_results

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| sessions(id PK, game_id, pa_id, dataset_identity, cursor, revision, created) | TEXT / INTEGER | 세션 |
| recommendations(session_id, pitch_id, payload JSON) PK(session_id, pitch_id) | TEXT / JSON | UPDATE 금지 트리거 |
| jobs(id, pitch_id, version, status, attempts, lease_until, lease_token, revision, message, cv_status) UNIQUE(pitch_id, version) | INTEGER / TEXT | 작업 큐 |
| event_results(session_id, pitch_id, revision, payload) | TEXT / JSON | UPDATE 금지 |
| inning_decisions(decision_id PK, game_id, revision, context_json, result_json, result_sha256, created_at) | TEXT / JSON | UPDATE·DELETE 금지 |

[2] 테이블 : intent_labels.sqlite3 — estimates / reviews (불변)

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| estimates(id PK, payload JSON) | TEXT / JSON | CV 추정 결과 |
| reviews(id PK, estimate_id FK, reviewer_id, payload) | TEXT / JSON | 검토 기록 |

[3] 테이블(파일) : 동결 설정 configs/G0-RESEARCH-FROZEN-v1.json · results/MLB-2026-OPE-v1.json

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| G0: files[{path, sha256}], seeds[0..4], train_commit 5e82beb | JSON | 변조 거부 테스트 |
| OPE: estimator DR V2+L1-R, n_games 2429, n_pa 184288, delta_ci [−0.001205, +0.000921], verdict | JSON | 사전 등록 1회, 봉인 |

3.4 API 명세 

외부 시스템·다른 서브시스템과의 인터페이스만 명세한다.

 

| 인터페이스명  | 클래스명  | 인터페이스 대상  | 설명  |
| --- | --- | --- | --- |
| GET /api/catalog · POST /api/sessions · GET /api/sessions/{id} | C-009 ObserverService | React 재생 화면 | 경기·타석 목록, 세션 생성·조회(View 본문) |
| POST /api/sessions/{id}/advance · /manual-intent | C-009 ObserverService | React 재생 화면 | {revision} 낙관적 잠금, 409 충돌 |
| GET /api/zones · GET /api/health | FastAPI main.py | React 재생 화면 | 9구역 정의·상태 |
| GET https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live | live_feed(초기) | MLB Stats API | 10초 폴링, User-Agent 고정 |
| GET /api/watch/{pk}/setup | setup_estimates | 팀원 CV 의도 모듈 산출물 | sha256·전 줄 검증 후 가로 위치만 제공(D151) |

[표 2] API 명세

4. 사용자 인터페이스(GUI)  

2차 스프린트의 재생 화면은 경기·타석 선택(카탈로그) → 포수 시점 존 그림과 추천 카드 → 실제 투구 공개 → 타석 종료 분석·사건 카드 순서다(apps/observer/web/src, React).

화면 틀은 D135~D146으로 확정했다 : 홈 H1(목록형), 경기 페이지 A안(존 무대, 상황 5가지 자동 전환, 아래 버튼 셋), 지난 경기 분석 P1~P3. 10/2 기준 /game 1·2단계까지 구현했고 실행 화면 캡처는 3·4차 명세서(공개 화면)에 수록한다.

 

5. 기타 참고 사항  

오픈소스 : FastAPI 0.141·uvicorn 0.53·pydantic 2.13 (MIT) : 관전 서버·발행기, React 19.1·Vite 7.3·TypeScript 5.9 (MIT/Apache) : 공개 화면, Playwright 1.63 (Apache) : 화면 검사, SQLite(공개 도메인) : append-only 원장.

재사용 : 1차 스프린트의 pitcheezy 인터페이스·전이 모델 코드를 그대로 쓰고, 서비스 추천은 experiments/pitchmdp 동결 번들(minimal-pitch-service-v1)을 해시 검증 후 로드한다.

다음 스프린트 고려 : 실시간 수신을 불변 원장과 표시 게이트로 감싸고(확인 중·정정), 발행 파일 계약을 정해 공개 호스트와 분리한다. 승리확률 모델은 볼카운트를 반영하지 않는다는 한계를 표기한다.

<끝>
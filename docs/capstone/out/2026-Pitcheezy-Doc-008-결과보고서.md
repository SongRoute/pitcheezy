프로젝트 결과보고서

  팀 번호( 04-05 )  

| 프로젝트명<br>(주제) | (국문) 피치이지(Pitcheezy): AI 투구 추천 기반 야구 관전 보조 시스템 |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | (영문) Pitcheezy: An AI Pitch-Recommendation Companion for Baseball Watching |  |  |  |  |  |
| 팀장 | 학부(과) |  | 소프트웨어학과 | 학번  | [학번] |  |
|  | 성  명 |  | 한강민 | 역할  | 팀장 · 문서 · 백엔드 |  |
| 개발기간 | 2025년 10월 1일 ～ 2026년 11월 30일 [확인] |  |  |  |  |  |
| GitHub저장소 | https://github.com/SongRoute/pitcheezy (정본) · Pitcheezy/transition-models (CV) |  |  |  |  |  |
| 참여학생 | 학부(과) | 학년/학번  |  | 성명 |  | 역할  |
|  | 소프트웨어학과 | [학년/학번] |  | 한강민 |  | 팀장·문서·백엔드 |
|  | 소프트웨어학과 | [학년/학번] |  | 송동선 |  | 모델·발행기·웹·운영 |
|  | 소프트웨어학과 | [학년/학번] |  | 조현준 |  | 영상 CV·공개 호스트 |
| 지도교수 | 아지즈 교수 |  |  |  |  |  |
| 프로젝트(주제) 수행 계획에 대한 요약 |  |  |  |  |  |  |
| 시스템 <br>설명 | MLB 투구 전 상황에서 구종·목표 구역 추천 상위 3개를 보여 주고, 투구 후 실제 공과 비교해 주는 웹 기반 관전 보조 화면(‘두 번째 전광판’)이다. 2023~2025 Statcast 자료로 학습·동결한 결과 확률 모델과 추천 정책(ARM-B)을 쓰고, 예측 성능과 추천 정책 효과를 분리해 검증했다. 실시간 경기는 맥미니 발행기가 MLB 공식 피드를 10초 주기로 읽어 공통 지연 15초를 적용한 파일로 발행한다. |  |  |  |  |  |
| 시스템<br>운영 환경  | 발행기·시험 서버 : Mac mini(Apple M4, 24GB, macOS 26) 상시 가동, 외장 SSD(T7 Shield)에 자료·모델 보관<br>학습·평가 : 학교 공용 GPU 서버(NVIDIA RTX A5000 ×5, Ubuntu 24.04)와 맥미니 MPS<br>공개(설계) : 팀원 공개 호스트가 발행 파일(v1/*.json)과 정적 React 번들을 제공하는 구조. 월드시리즈 기간에는 업로드를 끄고 맥미니 로컬 발행만 운영(공개 이용자 0명, 10/26 공동 시험 FAIL_OR_STOPPED)<br>외부 시스템 : MLB Stats API(실시간 피드), Baseball Savant(Statcast), 팀원 CV 의도 모듈 |  |  |  |  |  |
| 개발 적용<br>기술 | Python 3.12 · PyTorch(결과 확률 모델: 계층 디리클레 평활 + 공유 MLP) · 타석 MDP 가치 반복과 수비팀 승리확률(WE) 기반 solve_pa<br>오프라인 정책 평가(행동 정책 교차 적합, SNIPS·이중 강건 DR, 경기 단위 부트스트랩) · 사전 등록·동결(sha256)<br>FastAPI·SQLite(append-only 원장) · MLB Stats API 폴링·정정 추적 · 원자적 파일 발행(ETag CAS 업로드) · launchd 운영 감독·알림<br>React 19·TypeScript·Vite 공개 화면(모바일 우선, 포수 시점 존) · Playwright 화면 검사 · pybaseball 수집 · 포수 미트 셋업 CV 계약 |  |  |  |  |  |

| 개발목표결과물<br>(시스템 구성도,<br>실행 화면) |  |
| --- | --- |
| 프로젝트 <br>산출물<br>목록 | 문서 : 시스템 정의서(Doc-001), 프로젝트 관리 계획서(Doc-002), 소프트웨어 기능 아키텍처(Doc-003), 프로덕트 백로그(Doc-004), 1~4차 스프린트 백로그·명세서(Doc-005-n, Doc-006-n), 시험결과서(Doc-007), 결과보고서(Doc-008)<br>코드 : GitHub SongRoute/pitcheezy(연구 파이프라인·관전 서버·공개 발행기·웹, 출시 브랜치 ws/launch-2026), Pitcheezy/transition-models(CV 의도 모듈)<br>모델·자료 : 동결 예측기 G0(configs/G0-RESEARCH-FROZEN-v1.json), 연구 기준 G15 mix, 추천 정책 ARM-B(ML-POLICY-MATERIALIZATION-v1), Statcast 2023~2026 버전 장부(data/versions.md), results/ 결과 파일(추적 146개, HEAD 11eeb6e2)<br>설계·기록 : docs/design.md, docs/decisions.md(D1~D193), docs/experiments.md, 발행 파일 계약 v1, 2인·AI 협업 규약 |
| 시스템<br>활용 효과  | ① 263구 : 월드시리즈 1차전(849805) 263구를 사람 손 없이 받아 발행 — 업로드를 끈 비공개 무인 운영, 공개 이용자 0명(D181). 2~4차전 341·294·367구, 재시작 0(pm-jobs/ws-ops/runs).<br>② −0.0137 : G0의 NLL이 상황 빈도 기준선보다 0.0137 낮다(95% CI [−0.0144, −0.0129], 2025-07~09 개발 평가 311,721구, 5/5 시드, EXP-P11-001). 보정(ECE)은 기준선이 더 좋다(0.0017 vs 0.0087).<br>③ 180구 : 1차전 지원 투수 공 185구 중 180구에 투구 전 추천 기록(미지원 선발 78구 제외). 공당 계산 0.13초는 10/2 리허설(849844) 값, 1차전 실측은 투구→첫 수신 p50 21.0초. 2~4차전 추천 0(메모리 문턱, 운영 결손)·274·301구.<br>④ 정책 평가 : 2026 사전 등록 OPE(2,429경기·184,288타석)의 타석당 수비 승리확률 차이 경계 [−0.12, +0.09]%p(점추정 없음) — 0 포함, 개선 근거 없음(D131). +0.09%p는 윗끝이며 추천 정책은 미채택.<br>팬은 추천·근거를 실제 공과 비교하며 본다(정답 공 하나를 제시하지 않음). 범위 밖 : 사건 기여도 분해(D178)와 CV 의도 표시(라이브 미연결)는 전시 뒤 재개한다. |
|   Key Words (5개) : MLB Statcast, 투구 추천, 결과 확률 예측, 오프라인 정책 평가, 실시간 관전 보조 화면 |  |

작성자    성명 : 송동선                        확인자  성명 : [확인자]

※ 전체 분량은 2쪽으로 작성합니다. 

부록: 최종 개발된 소프트웨어 세부 내용 작성

1. 최종 시스템 아키텍처 (System Architecture)

1.1 최종 시스템 아키텍처 개요

시스템은 [그림 1]과 같이 맥미니(발행기·연구 파이프라인), 공개 호스트(팀원 관리), 팬 브라우저, 외부 시스템(MLB Stats API, Baseball Savant, CV 의도 모듈), 학교 GPU 서버로 구성한다.

맥미니의 운영 감독기(ws_operate.py)는 경기 첫 투구 30분 전에 발행기를 시작하고, LiveFeed가 MLB 피드를 10초 주기로 읽어 SQLite 원장(append-only)에 기록한다. ARM-B 추천 스레드는 동결된 ≤2025 정책으로 후보 상위 3개를 계산해 원장에 기록한 뒤에만 공개한다.

Publisher는 공통 지연 15초를 적용한 발행 파일(v1/manifest·health·dates·games·game)을 임시 파일 → fsync → 교체로 원자적으로 쓰고, Upload worker가 Bearer 토큰·ETag 조건부 PUT으로 공개 호스트에 올린다(업로드는 구현·공동 시험까지 했고 월드시리즈 기간에는 꺼 두었다). 팬 화면은 발행 파일만 읽고 맥미니로 가는 요청이 없다.

연구 파이프라인은 Baseball Savant에서 수집한 Statcast parquet을 T7 SSD에 두고 전이·승리확률 모델을 학습·동결한다. 큰 실험은 학교 A5000 서버에서 gpu run으로 실행하고 결과를 맥미니로 회수한다.

운영자는 Tailscale로 맥미니 관전 서버(:8766)에 접속해 시연·운영자 화면을 쓰고, ntfy·iMessage로 장애 알림을 받는다.

[그림 1] 시스템 구성도

2. 소프트웨어 아키텍처 (Software Architecture) 

소프트웨어는 연구 서브시스템(자료 수집·전처리, 모델 학습, 정책, 정책 평가, 실험 파이프라인)과 서비스 서브시스템(재생·서비스 본문, 실시간 수신·추천, 발행기, 운영, 공개 웹, CV 의도 계약)으로 나눈다.

두 서브시스템은 동결 모델 번들(sha256)과 발행 파일 계약(v1)으로만 연결한다. 서비스는 연구 코드의 pitchmdp 런타임만 가져다 쓰고, 공개 화면은 발행 파일만 읽는다.

2.1 아키텍처 구조 

컴포넌트·패키지 간 의존 관계는 [그림 2]의 패키지 다이어그램과 같다. 점선 화살표는 ‘사용한다’ 방향이다.

[그림 2] 패키지 다이어그램

2.2 컴포넌드/패키지 설명 

2.1절의 컴포넌트·패키지를 [표 1]과 같이 설명한다.

 

| 번호 | 컴포넌트/패키지 이름 | 설명  | 비고  |
| --- | --- | --- | --- |
| 1 | pitcheezy.data | Statcast 시즌별 수집·dtype 고정·버전 장부(statcast_fetch), 투구→상태·행동·결과 id 전처리(prepare), RE24·dRE24 계산(re24), 선발 풀·타자 군집(pitchers, batters) | pybaseball 2.2.7, pyarrow |
| 2 | pitcheezy.interfaces | 상태(카운트 12×주자·아웃 24×타자 군집 K×맥락 C), 행동(구종 9×위치 25), 결과 11종의 축 정의와 전이 텐서·가치 저장 형식(sha256)·검증 계약 | 계약 테스트 59건이 강제 |
| 3 | pitcheezy.transition | 전이 확률 모델 P(결과 \| 상태, 행동, 투수): 계층 디리클레 평활(count·smoothing)과 공유 MLP(neural.SharedNet) | torch, MPS/CUDA |
| 4 | pitcheezy.policy | 타석 MDP 가치 반복(vi.value_iteration)과 완화 정책(softmax·top-k·greedy) | 단위 RE24 |
| 5 | pitcheezy.ope | 행동 정책 추정(교차 적합)·tilt·SNIPS·이중 강건(DR)·경기 단위 부트스트랩·블록 평가 | 1스텝·궤적 |
| 6 | pitcheezy.experiment + configs/ | 준비 → 텐서 → VI → OPE → results/{ID}.json 파이프라인, W&B 기록, 2026 학습 시즌 거부, 시드·커밋 기록 | scripts/run_experiment.py |
| 7 | pitchmdp 런타임 | 수비팀 승리확률(WinExpectancy)·진루 분포(EmpiricalAdvancement)·타석 계획 solve_pa·추천 정책 런타임(policy_runtime, ARM-B) — 서비스가 쓰는 모델 계열 | experiments/pitchmdp, apps/observer/runtime_src |
| 8 | observer_app 재생·서비스 본문 | 세션·불변 사전 추천·공개 단계 제어(ObserverService, Store), 추천 어댑터(Recommender), 사건 분해(event_analysis), 서비스 JSON 본문(ServiceBodies), FastAPI 라우트 | SQLite WAL, UPDATE 금지 트리거 |
| 9 | observer_app 실시간 수신·추천 | MLB 피드 폴링(LiveFeed), 불변 스냅샷·투구 원장(LiveSnapshotStore), 표시 게이트(display_state, live_ledger.admit), 실시간 추천 스레드(LiveArmB, LivePolicy) | ws/launch-2026 |
| 10 | observer_app 발행기 | 발행 파일 생성·원자적 쓰기(Publisher, LocalDirSink), 조건부 업로드(HttpSink, MirrorSink, UploadWorker), 발행 계약 스키마·감사(publish_contract v1, publish_audit) | ws/launch-2026 |
| 11 | observer ops 운영 | 경기일 자동 시작·감시·재시작(ws_operate.Supervisor), 알림(ws_alert), 일정 동기화(ws_schedule_sync), 운영 설정(fixed_delay_s=15) | launchd 예시 |
| 12 | web/src/service 공개 화면 | 경기 목록(WatchList), 경기 화면(GameScreen: 상황판·존·추천·투구 띠·시트), 발행 파일 판독(publish.ts)·API 클라이언트(api.ts), 시연 운영자(DemoOperator) | React 19, Vite 7 |
| 13 | CV 의도 계약 | 영상 모듈 산출물 검사(validate_intent_estimate)·보관(intent_store)·서버 제공(setup_estimates, sha256 검증). CV 모델 자체는 팀원 저장소 | GET /api/watch/{pk}/setup |

[표 1] 컴포넌트·패키지 설명

3. 모듈 명세 (Module Specification) 

2.2절의 각 패키지에 포함된 주요 클래스·모듈을 명세한다.

3.1 패키지/컴포넌트 #1 : pitcheezy.interfaces (계약)

| 클래스명 | 역할 및 책임 | 주요기능 및 메서드 | 비고 |
| --- | --- | --- | --- |
| RE24Table | 24개 주자·아웃 상태의 기대 득점 RE24와 종결 결과별 변화 dRE24를 메타(버전·커밋)와 함께 저장·검증 | terminal_names, save(dir), load(dir, check_hash), validate(t) | npy + meta.json + sha256 |
| TransitionTensor | 전이 확률 텐서 P[P,S,A,O]·유효 마스크·관측 수와 투수·상태·결과 표, 메타 16키 | n_pitchers, K, C, context_kind, save, load(mmap), expected_shape | 행합 1±1e-5 검증 |
| ValueBundle | 가치 반복 결과 Q[P,S,A]·V[P,S]·정책과 메타 11키 | save, load, lookup(Q, pitcher, state, pitch, x, z, mode) | 단위 RE24 |
| states / grid / outcomes 모듈 | state_id = ((count×24+base_out)×K+cluster)×C+ctx, action_id = pitch×25+loc, 결과 11종·규칙 마스크·다음 카운트 | decode_state, decode_action, outcome_id, next_count, rule_mask | K ≤ 8, C ∈ {1,4,7} |
| pitch_types / validate 모듈 | Statcast 구종 코드를 9군으로 병합, 텐서·가치 위반 목록 검사(투수 16명 단위) | STATCAST_TO_PITCH_ID, validate_transition, validate_value | PO·IN·UN 제외 |

3.2 패키지/컴포넌트 #2 : pitcheezy.transition · policy · ope · experiment (학습·정책·평가)

| 클래스명 | 역할 및 책임 | 주요기능 및 메서드 | 비고 |
| --- | --- | --- | --- |
| LeagueLayers (smoothing) | 리그 수준 계층(카운트 → 카운트·행동 → 상태·행동)의 디리클레 평활 층을 만들고 투수 층과 결합 | __init__(n_sa, …, alpha, rule_mask), pitcher_layer, smooth_hierarchical | count.fit가 사용 |
| count 모듈 | 2023~24 학습·2025 홀드아웃으로 α를 고르고 최종 텐서를 적합, 홀드아웃 NLL·ECE 계산 | fit(df, …)→TransitionTensor, holdout_metrics, repertoire_counts | valid = 구종 학습 투구 ≥ 100 |
| SharedNet (neural, nn.Module) | 카운트·주자·군집·구종·위치·맥락·투수 임베딩 + MLP(256×2)로 결과 로짓을 내고 규칙 마스크를 −inf로 막음 | forward(p,c,b,k,g,l,x), embed_l2, train, predict_tensor, fit | AdamW, 조기 종료 patience 3 |
| policy.vi 모듈 | 보상표 R[s,o] = −dRE24, 다음 상태표, 가치 반복(\|ΔV\|<1e-9, 최대 200회), 완화 정책 | reward_table, next_state_table, value_iteration→(Q,V), relax(method), policy_evaluation | 투수 16명 블록 |
| BehaviorFactors (ope.behavior) | 행동 정책 π_b = π_type(구종\|상태,투수)×π_loc(위치\|구종,카운트,투수), game_pk%5 교차 적합, tilt로 평가 정책 생성 | block(lo, hi), behavior_factors, crossfit_logged, tilt(pb, Q, support, tau) | 지지 = valid ∩ 20구 이상 |
| ips / dr / blocks 모듈 | 비율 클리핑(20)·1스텝/궤적 SNIPS·DR·경기 클러스터 부트스트랩(1000회), 행 단위 저장 | estimate(w,r), bootstrap, estimate_dr1, estimate_traj_dr, compute_rows, save_rows | ESS 보고 |
| Experiment (experiment.py) | 설정 파일 하나로 준비→전이→가치→OPE 단계를 실행하고 results JSON·W&B에 기록, 2026 학습 거부 | pitches(season), stage_transition, stage_value, stage_ope, run, load_config, aggregate | configs/EXP-*.yaml |
| B2Net (baselines.b2) | Takamido류 재구현: 직전 6구 물리 피처 Transformer(d 256, 4 heads, 2층) + 현재 투구·맥락 → 11 로짓 | forward(win, pad, cur, pid, lid, ctx, count) | 축소 재구현 |

3.3 패키지/컴포넌트 #3 : pitcheezy.data (수집·전처리)

| 클래스명 | 역할 및 책임 | 주요기능 및 메서드 | 비고 |
| --- | --- | --- | --- |
| SeasonResult / LedgerRow / RunResult (statcast_fetch) | 월 단위 pybaseball 수집, dtype 고정, sha256·행 수 장부 기록, 재시도·게이트 | fetch_season, run_fetch, verify; FetchError·SchemaError·DtypeError | 2026 학습 수집 거부 |
| RE24Result (re24) | 타석 전후 상태로 RE24·dRE24를 계산(마지막 하프이닝 제외) | compute(df, exclude_last_half_inning) | data/re24/{ver} |
| prepare 모듈 | 투구 표를 20열 캐시(parquet)로 만들고 상태·맥락 id와 타석 보상을 계산 | prepare_season, state_ids, context_ids, pa_rewards, load_or_prepare | runs/_cache |
| pitchers / batters 모듈 | 선발 투수 풀(아웃·이닝 필터), 타자 k-means 군집(특징 10개) | make_pitchers, cluster_batters | data/pitchers, data/batters |

3.4 패키지/컴포넌트 #4 : observer_app 재생·과학 런타임 (apps/observer)

| 클래스명 | 역할 및 책임 | 주요기능 및 메서드 | 비고 |
| --- | --- | --- | --- |
| ObserverService (service.py) | 관전 세션 생성·진행·수동 의도 입력, 투구 전 불변 추천과 공개 단계 제어, 낙관적 잠금(revision) | catalog, create(game_id, pa_id), get(sid), advance(sid, revision), manual_intent | 409 충돌 |
| Store (store.py) | SQLite WAL 저장소: sessions·recommendations·jobs·event_results·inning_decisions, UPDATE 금지 트리거 | transaction, save_recommendation, save_event_result, enqueue, claim, finish | append-only |
| Recommender / PrePitchInput / CandidateAction | 투구 전 허용 목록 입력으로 후보(구종·구역·value·Δ%p·ESS) 계산, 캐시 키 = sha(모델·요청·존·레퍼토리) | recommend(pitch, pa), evaluate_pre_pitch, _compute | runtime_kind, latency 기록 |
| PlanResult / solve_pa (planner.py) | 카운트별 종결 가치로 타석 MDP를 역방향으로 풀고 2스트라이크 파울 고리를 선형해로 처리 | solve_pa(probabilities, terminal_values, …), topk(b, s, prev, k) | values·q_values·policy |
| GameState / WinExpectancy / EmpiricalAdvancement (game.py) | 경기 상태, 규제 로지스틱 승리확률, 경험 진루 분포로 종결 가치 계산 | terminal_values(state, we, adv) | WE는 볼카운트 미반영 |
| event_analysis 모듈 | 사건의 승리확률 변화(%p)를 기준·계획·실행·결과 잔여로 분해, 의도 근거 없으면 partial | analyze_event, failed_event, value_point | G4 뼈대 |
| DemoDataset / InningDecisionRepository / intent | 해시 검사한 재생 자료 로드·공개, 교체 직전 기록 저장, CV 의도 추정 계약 검사 | catalog, reveal; import_result, resolve; validate_intent_estimate | intent_labels.sqlite3 |

3.5 패키지/컴포넌트 #5 : observer_app 실시간·발행·운영 (ws/launch-2026)

| 클래스명 | 역할 및 책임 | 주요기능 및 메서드 | 비고 |
| --- | --- | --- | --- |
| LiveFeed / ReplayTransport | MLB schedule·feed/live 폴링(10초, 조건부 5초), 녹화 재생 전송 | snapshot(pk, delay_s)→(entry, wait_s), schedule(date) | User-Agent 고정 |
| LiveSnapshotStore / PersistentLiveFeed | 스냅샷 sha·사건 시각으로 new/duplicate/correction/quarantine 판정, 투구 원장·추천 기록, 복원·백업 | ingest(pk, feed, fetched, source), append_recommendation, restore, backup | UPDATE·DELETE 금지 |
| display_state / live_ledger | final·pregame·unsupported·unconfirmed(확인 중)·disconnected·waiting 게이트 판정, 원장 기록된 추천만 공개 | feed_facts, pitch_receipts, gate, decide; admit | 쉬운 말 문구 |
| LiveArmB / LivePolicy | 게이트 통과 시 ARM-B 상위 3구종과 구역 근사(ESS≥20, 질량≥0.01) 계산, 캐시 256·대기열 4 | lookup(feed, first, wait_s); __call__(feed) | 가용 메모리 ≥ 8GiB |
| ServiceBodies / GameArchive | 라우트와 발행기가 함께 쓰는 dates·games·game JSON 본문, 끝난 경기 보관 | dates(), games(date, delay), game(pk, after, delay) | 공통 지연 적용 |
| Publisher / LocalDirSink / HttpSink / MirrorSink | 10초 주기 발행(순번·시각), 임시 파일→fsync→교체, 조건부 PUT/DELETE(ETag, Bearer, sha256 헤더) | cycle()→health, run(once), guard_schedule; head/write/delete | 412 차단, 429 대기 |
| UploadWorker | 업로드 의도 큐를 스레드·자식 프로세스로 처리, 작업 마감 15초 | submit(path), request(method, path), close() | upload.enabled 기본 꺼짐 |
| Supervisor (ops/ws_operate.py) | 첫 투구 30분 전 시작, 비정상 종료 재시작(30/60/120초, ≤3회), 상태 파일·알림 | loop, prestart_with_retries, start_segment, stop_segment, observe, write_status | caffeinate |

3.6 패키지/컴포넌트 #6 : web/src/service 공개 화면 (React)

| 클래스명 | 역할 및 책임 | 주요기능 및 메서드 | 비고 |
| --- | --- | --- | --- |
| WatchList | KST 날짜별 지금·예정·기록 경기 카드 목록, 날짜 이동 | fetch games/<date>, 카드 선택 | /watch?date= |
| GameScreen | 상황판(점수·B/S/O·주자), 존 무대, 추천 후보 3, 실제 투구·이번 타석 띠, 선수·승률·점수표, 하단 시트 버튼 | 폴링 갱신, 상태 줄(확인 중·갱신 지연·연결 끊김) | /watch/{pk} |
| Zone / RecList / Sheets | 포수 시점 존 그림(추천 ▣·실제 점·미트 선), 순위·구종·선택 비중·구역 목록, 상세·기록·구종·선수·승률·정보 시트 | SVG 렌더, 시트 열기 | 쉬운 말 표기 |
| api.ts / publish.ts | API 모드(/api/service/*)와 공개 모드(v1/*.json, health 우선, ETag 304, 순번 역행 폐기, Date/Age 지연 판정) | fetchGame, fetchGames, applied Map | __PZ_STATIC__ |
| DemoOperator | 토큰 인증 뒤 시연 재생·일시정지·배속 제어 | X-Pitcheezy-Operator 헤더 | API 빌드만 |
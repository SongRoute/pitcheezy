2026년 충북대학교 소프트웨어학부 졸업작품연구과제

| AI 투구 추천 야구 관전 보조 시스템(Pitcheezy) |
| --- |
| 4차 스프린트 - 개발 명세서 |

2026년 10월 22일

문서번호 : 2026-Pitcheezy-Doc-006-4

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

4차 스프린트 백로그의 태스크 목록과 완료율은 [표 1]과 같다.

| US_ID | 스토리(요약) | TK_ID | 테스크 명 | 담당자 | 종료일  | 완료% |
| --- | --- | --- | --- | --- | --- | --- |
| US-20 | 사건 기여도 카드 | TK-57 | G5 재정의(D177)·G4 사건 기여도 첫 판 설계(D178, 의도 없이 가능한 분해) | 송동선 | 10/19 | 100 |
|  |  | TK-58 | RS-02 전향 재확인 계획(D179, 2026-11-06 열람) | 송동선 | 10/20 | 100 |
| US-26 | 파일 발행·공통 지연 | TK-59 | 공통 지연 15초 결정·ws_ops_config 반영(시청자별 지연 꺼짐) | 송동선 | 10/19 | 100 |
|  |  | TK-60 | 리허설 E(849817, 추천 끔)·E2(849815, 추천 켬) F1~F8 | 송동선 | 10/20 | 100 |
| US-27 | 운영 감독·알림 | TK-61 | 운영 감독기 ws_operate(Supervisor)·launchd 예시·상태 파일 | 송동선 | 10/17 | 100 |
|  |  | TK-62 | 일정 동기화 ws_schedule_sync·경기일 자동 시작 | 송동선 | 10/17 | 100 |
| US-28 | 운영 감독·알림 | TK-63 | 알림 ws_alert(ntfy·iMessage·macOS) 15초 루프 | 송동선 | 10/17 | 100 |
| US-38 | 동결·서비스 연결 | TK-64 | 골든 재생성·최종 후보 영수증·독립 검토(PASS_WITH_NOTES) | 송동선 | 10/19 | 100 |
|  |  | TK-65 | 출시 블로커 BLK-01~06 처리(비공개 백업, 옛 서버 종료, 공개 문구 초안) | 송동선 | 10/19 | 100 |
|  |  | TK-66 | MLB 허가 게이트 제외 결정·출시 연결 사슬 일정(D180) | 송동선 | 10/20 | 100 |
|  |  | TK-67 | 전시 포스터 구성안·제목 후보(D178) | 한강민·송동선 | 10/19 | 80 |
|  |  | TK-68 | 산출물 D01~D08 작성(저장소 기록 기준 생성) | 전원 | 10/20 | 90 |
|  |  | TK-69 | UI 마무리(10/21 18:00)·팀원 전달·공동 시험·판단(10/22) | 전원 | 10/22 | 0 |

 

[표 1] 태스크 목록

3. 기술 명세

3.1 클래스 명세   

4차 스프린트에서 개발·변경한 클래스(모듈)의 구조는 [그림 1]과 같다. 파이썬 모듈 수준 함수 묶음은 클래스에 준해 명세한다.

[그림 1] 클래스 다이어그램

[1] 클래스 : C-024 Supervisor (ops/ws_operate.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-024 | 멤버<br>변수 | config: ws_ops_config.json (fixed_delay_s 15, armb_min_free_gib 8) | 운영 설정 | Public |  |
|  |  | status.json, runs/<KST날짜>/{RUN.json, events.jsonl, publisher.log} | 상태·기록 | Public |  |
|  |  | restart backoff 30/60/120초, 최대 3회 | 재시작 규칙 | Public |  |
|  | 멤버<br>함수 | loop() | 경기일 감시 루프 | Public |  |
|  |  | prestart_with_retries() | 첫 투구 30분 전 발행기 시작 | Public |  |
|  |  | start_segment() · stop_segment() | 경기 구간 시작·종료(전 경기 종료 900초 뒤 정지) | Public |  |
|  |  | observe() · write_status() | 프로세스·메모리·피드 관찰, 상태 기록 | Public |  |

[2] 클래스 : C-025 ws_alert · ws_schedule_sync (ops/)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-025 | 멤버<br>변수 | loop 15초, channels: ntfy, iMessage, macOS | 알림 경로 | Public |  |
|  |  | schedule: MLB /api/v1/schedule → 경기일·첫 투구 시각 | 일정 | Public |  |
|  | 멤버<br>함수 | check(status) -> alerts | 장애·지연·메모리 판정 | Public |  |
|  |  | notify(level, message) | 알림 발송 | Public |  |
|  |  | sync(date) | 일정 동기화 | Public |  |

[3] 클래스 : C-026 event_analysis (G4 첫 판, D178)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-026 | 멤버<br>변수 | reference, plan, execution, observed | 수비팀 승리확률 | Public |  |
|  |  | status ∈ full · partial | 의도 근거 유무 | Public |  |
|  | 멤버<br>함수 | analyze_event(…) | total=100·(observed−reference)%p, 3성분+잔여, 의도 없으면 실행·타자·교체 ‘미측정’ | Public |  |
|  |  | shadow_record(event) | WS 뒤 0단계 내부 그림자 기록(계획) | Public |  |

[4] 클래스 : C-027 설정·영수증 (ws_ops_config.json · publish_config.json · final_candidate_*)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-027 | 멤버<br>변수 | fixed_delay_s=15, viewer_delay=off, armb_min_free_gib=8 | 운영 값 | Public |  |
|  |  | reference_card=on, upload.enabled=false | 발행 값 | Public |  |
|  |  | client_pins, known_failures(4종) | 최종 후보 고정 | Public |  |
|  | 멤버<br>함수 | load() · validate() | 설정 검증 | Public |  |
|  |  | receipt(commit) -> pins | 실행본 고정 | Public |  |

3.2 핵심 알고리즘 명세 

3.1절의 멤버 함수 중 처리 로직이 중요한 것의 행위를 명세한다.

[1] 메소드 : loop( ) @ C-024 Supervisor — 경기일 운영 감독

1. 일정(ws_schedule_sync)에서 오늘 경기와 첫 투구 시각을 읽는다. 2. 첫 투구 30분 전에 발행기를 시작하고(prestart_with_retries) caffeinate로 잠자기를 막는다.

3. observe()로 프로세스 생존·가용 메모리·피드 상태를 보고, 비정상 종료면 30/60/120초 간격으로 최대 3회 재시작한다. 가용 메모리가 8GiB 미만이면 추천(ARM-B)을 끈다.

4. 전 경기 종료 900초 뒤 구간을 멈추고 status.json·events.jsonl에 기록한다. 배포는 경기와 경기 사이에만 한다.

[2] 메소드 : check( ) @ C-025 ws_alert — 장애 판정·알림

1. 15초마다 status.json과 health.json을 읽는다. 2. 발행 순번 정체, 피드 실패, 재시작 횟수 초과, 메모리 부족을 수준별로 판정한다.

3. ntfy → iMessage → macOS 알림 순으로 보내고 같은 알림은 억제 창 안에서 반복하지 않는다.

[3] 메소드 : analyze_event( ) @ C-026 — 사건 기여도 첫 판(의도 없이 가능한 분해)

1. 타석 승리확률 변화 total을 계산한다. 2. 끝 공의 구종 선택 대비(기준 정책 vs 실제 선택)만 분해하고, 실행·타자·교체 요인은 ‘미측정’으로 둔다.

3. 10/22 출시 범위에서는 빼고, 월드시리즈 뒤 0단계 내부 그림자 기록부터 시작한다(D178).

3.3 DB 스키마 명세  

시스템은 관계형 DB 서버를 쓰지 않는다. 임베디드 SQLite 파일과 parquet·npy·JSON 파일의 스키마를 테이블에 준해 명세한다.

[1] 테이블(파일) : ws-ops/status.json · runs/<KST날짜>/RUN.json · events.jsonl

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| status: phase, publisher_pid, last_sequence, free_gib, armb_enabled, restarts, updated_at | JSON | 감독기 상태 |
| RUN.json: date, games[], started_at, segment{start, stop}, config_hash | JSON | 경기일 실행 |
| events.jsonl: ts, kind(start/stop/restart/alert), detail | JSON Lines | append-only |

[2] 테이블(파일) : 추천 영수증(receipts) · 최종 후보 고정(final_candidate_receipt)

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| receipts.version, pitches[{key, rec_recorded_at, shown, correction}] | JSON | game 파일 안 |
| client_pins{web_sha, publisher_sha, config_sha}, known_failures[4] | JSON | 실행본 v5→v6 |

3.4 API 명세 

외부 시스템·다른 서브시스템과의 인터페이스만 명세한다.

 

| 인터페이스명  | 클래스명  | 인터페이스 대상  | 설명  |
| --- | --- | --- | --- |
| POST https://ntfy.sh/<topic> · osascript iMessage | C-025 ws_alert | 운영자 폰 | 장애 알림 |
| GET statsapi.mlb.com/api/v1/schedule?sportId=1&date= | C-025 ws_schedule_sync | MLB Stats API | 경기일 일정 |
| PUT <host>/_publish/v1/* (업로드 기본 꺼짐) | C-021 HttpSink | 팀원 공개 호스트 | 10/22 공동 시험에서 실연결 |

[표 2] API 명세

4. 사용자 인터페이스(GUI)  

4차에서는 표시 상태별 문구와 복구 동작을 다듬었다(고정 자료 기준 390×844 캡처).

[그림 2] 확인 중 : 방금 공을 다시 확인하는 동안 추천을 보류

[그림 3] 자료 늦음 : 측정된 피드 지연을 알리고 추천은 유지

[그림 4] 연결 끊김 : 마지막 수신 내용을 유지, 폴링만으로 복구

[그림 5] 추천 보류 : 원장 기록 실패 시 추천만 숨기고 경기 상황은 유지

 

5. 기타 참고 사항  

오픈소스 : FastAPI 0.141·uvicorn 0.53·pydantic 2.13 (MIT) : 관전 서버·발행기, React 19.1·Vite 7.3·TypeScript 5.9 (MIT/Apache) : 공개 화면, Playwright 1.63 (Apache) : 화면 검사, pytest 9.1 (MIT) : 계약 테스트. 산출물 생성에 python-hwpx(MIT)·lxml을 썼다.

재사용 : 3차의 발행기·공개 화면을 그대로 쓰고 운영 설정만 바꿨다(fixed_delay_s 0 → 15).

다음 단계 고려 : 월드시리즈 1차전 뒤 공통 지연 재검토, RS-01·RS-02 결과 열람(2026-11-06), G4 그림자 기록, 전시(11월 말)용 재생 화면, 팬 사용성 평가(현재 0회).

<끝>
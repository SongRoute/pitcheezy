2026년 충북대학교 소프트웨어학부 졸업작품연구과제

| AI 투구 추천 야구 관전 보조 시스템(Pitcheezy) |
| --- |
| 3차 스프린트 - 개발 명세서 |

2026년 10월 12일

문서번호 : 2026-Pitcheezy-Doc-006-3

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

3차 스프린트 백로그의 태스크 목록과 완료율은 [표 1]과 같다.

| US_ID | 스토리(요약) | TK_ID | 테스크 명 | 담당자 | 종료일  | 완료% |
| --- | --- | --- | --- | --- | --- | --- |
| US-01 | 경기 목록 | TK-37 | 공개 화면 경기 목록 WatchList(날짜별 지금·예정·기록) | 송동선 | 10/12 | 100 |
| US-02 | 경기 목록 | TK-38 | 경기 화면 GameScreen(상황판·존·추천·투구 띠·시트) | 송동선 | 10/12 | 100 |
| US-07 | 지원 여부·출처 표시 | TK-39 | 미지원 사유 문구·참고 카드(과거 구종 빈도) | 송동선 | 10/11 | 100 |
| US-08 | 지원 여부·출처 표시 | TK-40 | 추천 출처 배지(captured_live / asof_replay)·receipts | 송동선 | 10/10 | 100 |
| US-09 | 실제 투구 공개 | TK-41 | LiveFeed 폴링·정정 추적과 LiveSnapshotStore 불변 원장 | 송동선 | 10/08 | 100 |
|  |  | TK-42 | display_state 게이트·live_ledger.admit·쉬운 말 문구 | 송동선 | 10/11 | 100 |
| US-12 | 존 그림 | TK-43 | 지난 공 존에 포수 미트 세로선 표시(setup 자료) | 송동선 | 10/12 | 100 |
| US-14 | 상황 머리글·선수 카드 | TK-44 | 선수 칸·구종 시트·km/h 표기 | 송동선 | 10/12 | 100 |
| US-15 | 박스스코어·투구 띠 | TK-45 | 점수표(linescore)·박스스코어 as-of 계산 | 송동선 | 10/12 | 100 |
| US-16 | 박스스코어·투구 띠 | TK-46 | 다음 타자 표시(next_batters) | 송동선 | 10/12 | 100 |
| US-17 | 승리확률 표시 | TK-47 | 홈팀 승리확률 줄(win_expectancy, 조건부) | 송동선 | 10/12 | 100 |
| US-18 | 지난 장면 목록 | TK-48 | 타석 선택 시트·지난 공 상세 | 송동선 | 10/12 | 100 |
| US-22 | 시연 운영자 제어 | TK-49 | 지도교수 시연(10/6, 813031 재생)·시연 운영자 콘솔 | 전원 | 10/06 | 100 |
| US-24 | 피드 폴링·장부 기록 | TK-50 | LiveArmB 실시간 추천 스레드·LivePolicy·원장 기록 후 공개 | 송동선 | 10/10 | 100 |
| US-25 | 파일 발행·공통 지연 | TK-51 | Publisher·LocalDirSink·발행 계약 v1 스키마·예제 15개 | 송동선 | 10/11 | 100 |
|  |  | TK-52 | HttpSink·MirrorSink·UploadWorker(ETag CAS) | 송동선 | 10/12 | 100 |
|  |  | TK-53 | 849831 오프라인 재생 위반 0·849809 첫 실폴링 시험 운영 | 송동선 | 10/12 | 100 |
| US-39 | 사람 라벨 표시 | TK-54 | M3 사람 라벨 84건·CV 영수증(PM-M3-receipt) | 조현준 | 10/04 | 100 |
| US-40 | 사람 라벨 표시 | TK-55 | 스트라이크존 모서리 라벨·좌표 변환 검증 | 조현준 | 10/04 | 100 |
| US-41 | CV 추정 결과 검토·연결 | TK-56 | CV 추정 결과 검토 절차·setup 파일 sha256 검증 | 조현준·송동선 | 10/12 | 70 |

 

[표 1] 태스크 목록

3. 기술 명세

3.1 클래스 명세   

3차 스프린트에서 개발·변경한 클래스(모듈)의 구조는 [그림 1]과 같다. 파이썬 모듈 수준 함수 묶음은 클래스에 준해 명세한다.

[그림 1] 클래스 다이어그램

[1] 클래스 : C-016 LiveFeed · ReplayTransport (observer_app/live_feed.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-016 | 멤버<br>변수 | interval_s = 10 (25초 미만 지연 시청자 있으면 5) | 폴링 간격 | Public |  |
|  |  | schedule cache, user_agent | 일정 캐시·식별 | Public |  |
|  |  | ReplayTransport · RecordedPaceReplay | 녹화 재생 전송(상속) | Public |  |
|  | 멤버<br>함수 | snapshot(pk, delay_s) -> (entry, wait_s) | 지연 d를 적용한 최신 본문 | Public |  |
|  |  | schedule(date) | /api/v1/schedule 조회(재시도 4회, 백오프 1·2·4초) | Public |  |

[2] 클래스 : C-017 LiveSnapshotStore · PersistentLiveFeed (observer_app/live_store.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-017 | 멤버<br>변수 | snapshots, pitch_events, context_decisions, served_recommendations | SQLite 테이블(UPDATE·DELETE 금지) | Public |  |
|  |  | WAL, synchronous=FULL | 내구성 설정 | Public |  |
|  | 멤버<br>함수 | ingest(pk, feed, fetched, source) -> outcome | sha·사건 시각으로 new / duplicate / correction / quarantine 판정 | Public |  |
|  |  | append_recommendation(…) | 추천 원장 기록(최초 불변, 정정은 revision+1) | Public |  |
|  |  | restore() · backup() | 복구·백업 | Public |  |

[3] 클래스 : C-018 display_state · live_ledger (observer_app/display_state.py, live_ledger.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-018 | 멤버<br>변수 | state ∈ final·pregame·unsupported·unconfirmed·disconnected·waiting·recommended·feed_delayed·ledger_unavailable·unavailable | 표시 상태 | Public |  |
|  |  | COPY[(state, reason)] -> (label, message) | 쉬운 말 문구표 | Public |  |
|  | 멤버<br>함수 | feed_facts(feed) · pitch_receipts(store) | 피드·영수증 사실 | Public |  |
|  |  | gate(facts) · decide(…) | 상태·사유 판정 | Public |  |
|  |  | admit(rec, ledger) -> shown \| withheld | 원장 기록된 추천만 공개 | Public |  |

[4] 클래스 : C-019 LiveArmB · LivePolicy (observer_app/live_armb.py, scripts/demo_precompute.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-019 | 멤버<br>변수 | cache(256), queue(4), thread | 캐시·대기열·스레드 | Public |  |
|  |  | policy: 동결 ≤2025 ARM-B, τ 0.1 | 정책 번들 | Public |  |
|  | 멤버<br>함수 | lookup(feed, first, wait_s) | 게이트 통과 시 후보 계산 | Public |  |
|  |  | LivePolicy.__call__(feed) -> 후보 3 {구종, 구역, 선택 비중} | candidate law + location_proxy(σ 0.45, ESS ≥ 20, 질량 ≥ 0.01) | Public |  |

[5] 클래스 : C-020 ServiceBodies · GameArchive (observer_app/service_api.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-020 | 멤버<br>변수 | delay_s, zone_bounds, archive_dir | 설정 | Public |  |
|  |  | GameArchive | 끝난 경기 보관소 | Public |  |
|  | 멤버<br>함수 | dates() · games(date, delay) | 날짜·경기 카드 본문 | Public |  |
|  |  | game(pk, after, delay) | game 본문: current·cutoff·pitches·pas·linescore·display_state·receipts·win_expectancy | Public |  |

[6] 클래스 : C-021 Publisher · LocalDirSink · HttpSink · MirrorSink (observer_app/publisher.py)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-021 | 멤버<br>변수 | sequence, sinks, interval_s=10, fixed_delay_s | 발행 상태·설정 | Public |  |
|  |  | state file <out>.state.json | 순번 저장 | Public |  |
|  | 멤버<br>함수 | cycle() -> health | dates·games·game 빌드 → 임시 파일 → fsync → os.replace → health | Public |  |
|  |  | run(once) · guard_schedule() | 주기 실행·일정 보호 | Public |  |
|  |  | HttpSink.head/write/delete(path) | Bearer·If-Match/If-None-Match·X-Content-SHA256, 412 차단·429 대기 | Public |  |
|  |  | MirrorSink.write() | 로컬에 쓰고 업로드 의도를 큐에 넣음 | Public |  |

[7] 클래스 : C-022 UploadWorker · publish_schema · publish_audit (observer_app/upload_worker.py 외)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-022 | 멤버<br>변수 | queue, child process, deadline 15s | 업로드 작업 | Public |  |
|  |  | schemas: manifest·health·dates·games·game·setup | 계약 v1 | Public |  |
|  | 멤버<br>함수 | submit(path) · request(method, path) · close() | 업로드 제출·요청·종료 | Public |  |
|  |  | validate(file) · audit(dir) | 스키마 검증·감사 | Public |  |

[8] 클래스 : C-023 공개 화면 WatchList · GameScreen · Zone · RecList · Sheets · publish.ts (web/src/service)

| C_ID | 분류  | 멤버 변수/함수  | 설명  | 타입   | 참고  |
| --- | --- | --- | --- | --- | --- |
| C-023 | 멤버<br>변수 | __PZ_STATIC__, POLL 10000ms, STALE_S 30, LOST 30000ms | 빌드 상수·환경 | Public |  |
|  |  | localStorage: pitcheezy.unit, pitcheezy.viewer | 단위·시청자 설정 | Public |  |
|  | 멤버<br>함수 | WatchList(date) · GameScreen(pk) | 목록·경기 화면 | Public |  |
|  |  | Zone(rec, actual, setup) · RecList(candidates) · Sheets(kind) | 존 그림·후보 목록·시트 | Public |  |
|  |  | publish.ts: fetchHealth → fetchGame(ETag, 순번 역행 폐기, Date/Age 지연 판정) | 공개 모드 판독 | Public |  |

3.2 핵심 알고리즘 명세 

3.1절의 멤버 함수 중 처리 로직이 중요한 것의 행위를 명세한다.

[1] 메소드 : ingest( ) @ C-017 LiveSnapshotStore — 실시간 수신·정정 판정

1. 폴링 간격이 지났으면 feed/live를 1회 GET하고 수신 완료 시각을 기록한다.

2. 본문 sha가 같으면 duplicate, 사건 시각이 최신보다 늦으면 new, 같은 슬롯에 내용이 다르면 correction, 이상하면 quarantine으로 분류한다.

3. 투구별(playId + 타석·투구 번호) 최초 수신은 불변으로 두고 정정은 revision+1로 덧붙인다(UPDATE·DELETE 금지).

4. snapshot(pk, d)는 사건 시각 ≤ now − d 인 최신 본문을 돌려준다(공통 지연 d).

[2] 메소드 : gate( )·admit( ) @ C-018 — 표시 게이트와 실시간 추천 공개

1. final / pregame / unsupported / unconfirmed(‘확인 중’) / disconnected / waiting 을 순서대로 판정한다.

2. 게이트를 통과했을 때만 LivePolicy(feed)로 ARM-B 후보 상위 3구종을 얻고, 구종별 구역은 location_proxy로 붙인다(모델이 고른 목표가 아니라 근사).

3. admit : 원장 기록이 성공해야 공개(shown), 아니면 withheld. 미승인 투구의 실제 값은 null로 발행한다.

4. 문구는 COPY 표(쉬운 말, 기술 용어·채점 문구 금지)에서만 가져온다.

[3] 메소드 : cycle( ) @ C-021 Publisher — 발행 주기와 업로드

1. sequence += 1 → state 저장. 2. dates·games·game 파일을 만들되 진행 중 경기는 digest가 같으면 건너뛰고 종료 경기는 600초마다 재확인한다.

3. 쓰기 : 임시 파일 → fsync → os.replace. 모든 파일 끝에 publish{contract, sequence, published_at, final}를 넣고 health를 마지막에 쓴다.

4. MirrorSink : HEAD → If-Match / If-None-Match 조건부 PUT·DELETE. 412면 차단, 429면 대기 후 재시도. 업로드는 upload.allowed와 --upload가 모두 있어야 켜진다.

3.3 DB 스키마 명세  

시스템은 관계형 DB 서버를 쓰지 않는다. 임베디드 SQLite 파일과 parquet·npy·JSON 파일의 스키마를 테이블에 준해 명세한다.

[1] 테이블 : live store(store.sqlite3) — snapshots / pitch_events / context_decisions / served_recommendations

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| snapshots(seq, game_pk, event_wall, version, sha, first_fetched, fetched, source, body gzip) | INTEGER / TEXT / BLOB | UPDATE 금지 |
| pitch_events(provider_event_id, at_bat_number, pitch_number, outcome ∈ new/correction/quarantine, revision, first_received, …) | TEXT / INTEGER | UPDATE·DELETE 금지 |
| context_decisions, context_reconciliations | TEXT / JSON | 맥락 판정 기록 |
| served_recommendations(pk, pitch_key, rank, pitch_type, zone_id, recorded_at, …) | TEXT / INTEGER | 추천 원장(append-only) |

[2] 테이블(파일) : 발행 파일 계약 v1 — manifest / health / dates / games / game / setup

| Attributes | Types | Stored procedure (if any) |
| --- | --- | --- |
| manifest: commit, contract, files, fixed_delay_s, live_dates, publish_interval_s, publish{} | JSON | 입구 |
| health: sequence, files_written, unavailable_games[], feed_failing_games, game_checks{} | JSON | 상태 |
| dates: dates[]  /  games/<날짜>: date, games[카드], live_ok | JSON | 목록 |
| game/<pk>: current, cutoff, feed, game, linescore, next{rec{candidates[{rank, pitch_type, zone_id, target, detail.probability}], policy, recorded_at}}, pas[], pitches[{key, rec, actual, receipt}], display_state, win_expectancy | JSON | 경기 본문 |
| setup/<pk>: pitches[{plate_x_feet, x_band, review_status, is_intent_proxy}] | JSON | 포수 미트 가로 위치 |

3.4 API 명세 

외부 시스템·다른 서브시스템과의 인터페이스만 명세한다.

 

| 인터페이스명  | 클래스명  | 인터페이스 대상  | 설명  |
| --- | --- | --- | --- |
| GET /api/service/dates · games?date&delay · games/{pk}?after&delay · games/{pk}/tap | C-020 ServiceBodies | 공개 화면(API 모드)·발행기 | Cache-Control: no-store, 시연 중 423 |
| /api/service/operator/* (헤더 X-Pitcheezy-Operator) | FastAPI main.py | 시연 운영자 콘솔 | 재생·일시정지·배속 |
| GET statsapi.mlb.com/api/v1/schedule?sportId=1&date= · /api/v1.1/game/{pk}/feed/live | C-016 LiveFeed | MLB Stats API | timeout 10초, 재시도 4회 |
| HEAD/PUT/DELETE <host>/_publish/v1/<path> (Bearer, If-Match, X-Content-SHA256) | C-021 HttpSink | 공개 호스트(팀원) | 본문 ≤ 5MiB, ETag 32hex |
| GET <BASE>/v1/*.json (If-None-Match) | C-023 publish.ts | 공개 호스트 | 팬 화면 읽기 전용 |

[표 2] API 명세

4. 사용자 인터페이스(GUI)  

공개 화면(ws/launch-2026, 고정 자료 849843 기준 390×844 캡처)은 사용자의 관전 순서대로 다음과 같다.

[그림 2] 투구 전 : 상황판·존 무대·‘다음 공 추천’ 후보 3(구종·선택 비중)·이번 타석 띠

[그림 3] 미지원 투수 : ‘추천 없음’과 사유 문구

[그림 4] 공수 교대 : 끝난 이닝 정보를 현재로 보이지 않음

 

5. 기타 참고 사항  

오픈소스 : FastAPI 0.141·uvicorn 0.53·pydantic 2.13 (MIT) : 관전 서버·발행기, React 19.1·Vite 7.3·TypeScript 5.9 (MIT/Apache) : 공개 화면, Playwright 1.63 (Apache) : 화면 검사, SQLite(공개 도메인) : append-only 원장. 알림은 ntfy(공개 서비스)와 macOS iMessage를 쓴다.

재사용 : 2차의 관전 서버·pitchmdp 런타임을 출시 브랜치에서 그대로 쓰고, 공개 화면은 발행 파일만 읽도록 publish.ts를 새로 썼다.

다음 스프린트 고려 : 경기일 자동 시작·감시·알림(운영 감독기), 공통 지연 값 확정, 실호스트 업로드·공동 시험, 가용 메모리 8GiB 문턱 관리, 실기기(iPhone Safari) 확인.

<끝>
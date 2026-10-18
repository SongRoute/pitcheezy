# CX-03 — 이용 측정 준비 상태 감사

2026-10-18 KST · 작성 `coordination_docs` · [CLAIM](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248464086).
**정적 감사 완료·작성자와 다른 native `coordination_review` 검토 PASS·root 대표 수용.** 동일 모델의 별도 에이전트 정적 검토이며 재실행은 0회다. 선택 참여 세션·활성 시간·선택형 의견의 저장/집계 부품은 이미 있다. 실제 뷰어 연결·수집 활성화·운영 추출은 미확인이다. 고유 사람 수, 과거 최대 동시접속, 실제 투구 전 화면 도달률을 현재 부품의 측정값으로 제시할 수 없다.
공개 고정 소스만 읽었고 코드 실행·테스트·빌드·실호스트 요청·운영/사용자 로그·private/자격·실자료 원문 열람은 하지 않았다. 소스 안의 스키마와 기록 코드를 근거로 삼으며 graphify는 탐색 보조다. 수집 승인·숫자 목표·법률 판단은 범위 밖이다.

## 근거 pin과 읽은 경로

`O:`는 우리 `dccde711748d40565a78c50356cf8cb5c4f43d3e`, `T:`는 [팀원 공개 소스](https://github.com/Pitcheezy/transition-models/tree/724888caaa65414db030260ffdcc0cae2aff2135)다. 아래 약호 뒤 숫자는 소스 줄 번호다.

| 약호 | 고정 파일 | 읽은 바이트 SHA-256 |
|---|---|---|
| slot | O: `apps/observer/web/src/service/HostSlot.tsx` | `e2a29a17ac7cec43bfad18c0d8a83fe72c41e038d0c2139cf01a616f54d4960b` |
| clock | O: `apps/observer/web/src/service/publish.ts` | `f7fa3e9516dfc705a1b3f9fdae3470c92c363e5f7a5d02989930f1a4fff2d947` |
| pair | O: `apps/observer/backend/observer_app/rec_pairing.py` | `22f7ae013673dcdcd7705b5e7b7bc6d1942c55b277aec33ae6d8f6f453b75b61` |
| store | O: `apps/observer/backend/observer_app/live_store.py` | `b019f00da0c1a065c9af91b79e9cc1d19568589e43ebe6b72b9e4b1f1f3c3d2d` |
| tel | T: `ops/public-host/telemetry.mjs` | `c90d0f23dbf80eb965d343bef8fb75129cf3ce7ee9e64fbe6f683228ad0f646a` |
| client | T: `ops/public-host/client/usage-client.mjs` | `ae6b401f838b2404b61d2b5c9682dd57760b03e03195e403000458c75c8d9f48` |
| widget | T: `ops/public-host/client/widget.mjs` | `0c62b29c4a0e75f060af2fd7b191a82c282bc52860cfa851b553869da1eb5138` |
| sql | T: `ops/public-host/migrations/0001_telemetry.sql` | `479ded40208b893fe979b40f36db5ee0b55cb6bfa8b83c0e10f0af02fefc28b4` |
| export | T: `ops/public-host/export-usage.mjs` | `2d23893eec16eeba0e3592a6112878cae4e8909a6dee163b75a757bda0a90de7` |
| report | T: `ops/public-host/report.mjs` | `4570f08fbd49318f7d8a755906b6b19de644853fb359f99a7a7fe938ec95695a` |

추가 줄 근거는 위 commit에 고정한다: O의 `web/src/main.tsx`, `web/vite.config.ts`, `web/src/service/api.ts`, `backend/observer_app/{publisher,live_feed}.py`, `publish_contract/v1/schema/game.schema.json`은 모두 `apps/observer/` 아래다. T의 `gateway.mjs`, `access.mjs`, `worker.mjs`, `wrangler.example.jsonc`, `TELEMETRY.md`, `REPORT.md`는 `ops/public-host/` 아래다. T의 `src/operations/release_bundle.py`, `docs/WS_STAGE5_READINESS_20261018.md`도 읽었다. 아래 짧은 파일명은 이 경로를 가리킨다.

## 실제 기록 경로와 상태

1. 우리 공개 빌드는 `VITE_PUBLISH_BASE`로 `__PZ_STATIC__`가 정해지며 `main.tsx:31–36`에서 빈 slot을 넣는다. `slot:15–20`의 이벤트에는 element/screen/gamePk만 있다. 방문·추천 노출·표시 시각을 보내거나 저장하지 않는다.
2. 팀원 `widget:4–27,56–71`는 선택 체크박스와 의견 버튼을 만들고30초 flush·visibilitychange·pagehide를 연결한다. `client:71–105,122–143`가 두 POST로 전송한다. **slot→mountUsageWidget→setGame/destroy 연결과 실제 공개 자산 삽입은 아직 증거가 없다**(TELEMETRY.md:91–118,126–138). release bundle에 client 파일이 포함된 사실만으로 실행 연결을 증명하지 않는다(release_bundle.py:18–31).
3. `gateway.mjs:292–326`은 notice→viewer gate를 거친 뒤 telemetry를 호출한다. `tel:14–34,144–160`는 정확한 필드만 받아 private D1에 저장한다. `/session`과 `/feedback` POST만 있고 공개 목록/집계 GET은 없다.
4. `tel:174–189`는 UTC일/경기별 세션 수·활성 시간 합/평균·의견 분포를 집계한다. `export:86–112`는 **운영자가 이미 취득한 로컬 SQLite 사본**을 읽기 전용으로 집계하며 원격 취득기가 아니다. `report:127–181,225–305`는 보고 초안을 만들고 자동 게시하지 않는다.
5. 예시 설정은 telemetry/feedback OFF(wrangler.example.jsonc:8–19). Stage5 후보도 gateON·읽기/업로드/수집/피드백OFF이며 deployable=false(WS_STAGE5_READINESS_20261018.md:28). **현 운영 계정을 조회하지 않았으므로 실제 스위치 상태를 새로 확인했다고 쓰지 않는다.**

## 지표별 준비도

`READY`는 고정 소스에 부품이 있다는 뜻이며 수집/실운영 검증 완료가 아니다.

| 지표 | 상태 | 현재 가능한 것 / 빠진 것 |
|---|---|---|
| 선택 참여 탭 세션 수 | READY 부품 / 연결 PARTIAL | 새 UUID 행 COUNT. 탭·새로고침·경기·UTC날짜별로 나뉜다. 실제 연결·ON·동의·보존 범위 확인 필요(tel:105–110,180–182; client:38–45,115–120) |
| 고유 방문 **사람 수** | MISSING | 영구 개인 식별자가 없다. 접속 쿠키도 사람 ID가 아니며 D1 session_id와 별개다. 세션 수를 고유 인원으로 바꾸지 않는다(access.mjs:2–6; sql:1–20) |
| 경기별 과거 최대 동시접속 | MISSING | latest `updated_ms`와 누적 active_ms만 덮어쓴다. pulse/visible 구간 이력·정상 종료 이벤트가 없어 과거 동시 활성 인원 곡선을 재구성할 수 없다(sql:1–10; tel:121–126) |
| 누적 화면 활성 시간 합·평균 | READY 부품 / 해석 PARTIAL | hidden·긴 sleep 구간 제외한 클라이언트 보고값. 실제 주의/시청 시간 아님. 현재 집계는 합/평균만(tel:180–182; client:43–52) |
| 세션 길이 중앙값 | PARTIAL | 보존된 각 행의 active_ms로 별도 사전고정 집계는 가능하나 현재 summary/export/report에는 중앙값 없음. 체류 wall duration이나 종료완료 세션의 길이와 같지 않음 |
| 짧은 의견 분포 | READY 부품 / 연결 PARTIAL | helpful/unclear/problem, reason 목록 지원. 기본 widget은 세 버튼만 보내 reason=none. 응답 ID 단위 중복 제거이며 사람별1회/세션별 응답률은 계산 불가(widget:30–36; client:122–141; tel:29–32) |
| 팬의 이해·추천 신뢰 | PARTIAL / 신뢰 MISSING | “도움이 돼요/설명이 어려워요/불편”은 반응 자료다. 이해 향상·정확도·추천 신뢰의 별도 측정 도구가 아니다. 질문/해석을 먼저 결정 |
| 실제 투구 전 추천 **발행률** | PARTIAL | 결정원장·provider startTime·발행시각 부품은 있으나 물리투구 기준 join/분모/시계·첫발행 이력 확보가 별도 필요. 현재 pairing은 첫 수신 전 기록이지 실제 투구 전 발행률 아님 |
| 실제 투구 전 **화면 표시 도달률** | MISSING | DOM에 표시된 추천의 decision_id·pitch key·표시시각·visible 상태 ack가 없다. 파일 수신/발행/health 성공을 표시 증거로 쓰지 않음 |

## 식별·시간·중복·결측의 실제 동작

- 세션 입력은 `{v,consent,session_id,game_pk,sequence,active_ms}`, 의견은 `{v,consent,feedback_id,game_pk,rating,reason}`뿐이다. D1에는 서버의 created/updated/expires 시각을 더한다. 클라이언트 event 시각·visibility 구간·end 이유·추천 키는 없다(tel:14–34,96–126; sql:1–20).
- UUID는 메모리만 사용한다. 첫 sequence0/active0 저장 성공 뒤부터 시간을 센다. 새로고침/경기/클라이언트 UTC날짜 전환은 새 세션이며 다른 탭도 별개다. created_ms는 첫 서버 저장 시각, updated_ms는 마지막 수락 갱신이지 방문 종료가 아니다(client:22–52,96–120).
- 활성 시간은 performance.now 차이로 계산하며 이전 표본에서 visible인 구간만 더한다. 60초 초과 간격은 제외하고8시간에 제한한다. 이는 현재 구현 상수이며 새 성과목표가 아니다. 서버는 누적시간이 서버 경과+1초를 넘으면 거부한다(client:43–52; tel:121–140).
- session 동일/과거 sequence 재전송은 합산하지 않는다. 과거 sequence와 과거 active_ms의 재전송은 저장값을 유지하며 accepted이고, 새 sequence에서 active_ms가 감소하면409다. 동일 sequence의 값 충돌·경기/서버 UTC날짜 변화도409다. client의 UTC날짜는 Date.now, 서버 날짜는 Date.now이므로 자정 부근 기기 시계 차이는 충돌/누락 원인이 될 수 있다(tel:97,121–140; client:23,31,45).
- 같은 pending payload 재시도, 연속3회 실패 시 pause, certain gate401이면 동의를 해제한다. 종료/전환 직전 시간·송신 실패·sleep·사전동의 전 구간은 누락된다. 의견은 같은 ID/선택 재시도이며 새 ID를 자동 생성해 불확실한 응답을 중복 저장하지 않는다(client:75–105,118–143). 전체 방문 대비 opt-in 비율은 비참여 분모를 수집하지 않아 미측정이다.

## 발행·표시와 시계의 경계

- `publisher.py:981–1001`의 published_at은 파일 작성 주기의 로컬 시각이며 이 시각을 넣은 뒤 upload를 큐에 넣는다. 원격 저장 성공이나 팬 표시 시각이 아니다. `publisher.py:1145–1161`의 health/game_checks는 발행기 점검 상태다. 팬 사용량/동시접속으로 세지 않는다.
- `store:67–83,454–459`에는 provider_event_time=startTime, 최초/현재수신, receipt_clock이 있다. 결정원장의 decided/decision_clock/store_seq는 `store:114–125,660–702`에 있다. 원장은 팬 화면 체류나 표시 ack가 아니며 target은 UNKNOWN이다. `pair:3–29`는 첫수신보다 이전 결정과 입력동치를 확인하고 화면표시를 주장하지 않는다.
- 실제 투구 기준을 정한다면 provider startTime 의미·보정/정정·누락과 observer clock 오차를 사전고정해야 한다. feed의 endTime은 정정될 수 있고 event_wall은 결과도착 목적이라 그대로 투구발생 시각으로 쓰지 않는다(live_feed.py:168–176,194–207). 이번에는 실제 기록을 읽어 시계 오차나 분모를 측정하지 않았다.
- `clock:84–118`은 HTTP Date/Age 또는 published_at을 performance.now에 연결하는 지연표시용 추정이다. `api.ts:75–97,103–115`의 seenAt은 파일 적용 시점의 메모리값이지 DOM paint/추천 표시 증거가 아니며 저장·전송하지 않는다. NTP 동기화/오차 한계가 검증된 측정 시계로 간주하지 않는다.

## 안내·보관·권한

- 통계 체크와 의견 선택은 독립 동의이며 기본 OFF다. retainDays 누락/오류면 둘 다 꺼진다. DNT/GPC를 client와 server가 거부하고 서버는 정확한 HTTPS Origin을 확인한다(client:11–34; widget:11–27; tel:40–53,150–156). 이는 동작 감사이며 법률 적합성 판정이 아니다.
- gate가 켜지면 same-origin 쿠키가 전송되지만 D1에 저장하지 않는다. 접근쿠키 expiry/현재 digest 허용목록과 통계 UUID를 혼동하지 않는다(access.mjs:85–104; client:61–65). TELEMETRY.md:112–116은 WebKit의 omit에도 쿠키가 전송된 기존 한계를 남기며 실제 Safari/인프라 로그는 별도 확인이다.
- 보관일은1~90 범위의 운영 설정이다. expires_ms는 최초 저장 때 고정하고 pulse로 연장하지 않는다. worker scheduled가 만료행을 지우고 수집 OFF여도 정리한다(tel:98,163–172; worker.mjs:5–7). 집계는 즉시 만료행 제외, 실제 cron·백업/Time Travel·집계 내보내기 보관은 별도 확인(TELEMETRY.md:67–78).
- `export`는 snapshot 포장에 coverage=unverified/aggregation=utc_day를 유지한다. `report:148–162`는 보존행의 날짜전체 부분합으로 쓰고 정밀 경기창 집계라고 하지 않는다. OFF/행 없음/부분/0세션을 구별하며 관측 실패를 이용자0명으로 바꾸지 않는다. 운영 D1 추출은 계정 담당의 별도 권한/범위가 필요(REPORT.md:13–18,33–55,107–112).

## 후속 PLAN — 5개, 구현·활성화 수락 아님

| 제안 | 기존 부품 재사용 / 최소 완료 조건 | 담당·사람 결정 |
|---|---|---|
| MEAS-01 정의 초안 | 세션 수/활성시간/의견의 분모·UTC날짜·결측·상태를 고정. “고유 사람” 대신 관측 세션 라벨. 중앙값은 보존 active_ms의 중앙값으로 별도 요구 여부 결정 | Codex 문서 별도 CLAIM, Song 지표·숫자목표 확정 |
| host-slot 연결 준비 | 기존 widget 사용, 늦은 script 로드/중복 mount/게임전환/해제 처리와 실제 자산 경로·서버/클라이언트 flags 일치 증거. 기존 뷰어 코드 병렬편집 금지 | 팀원 호스트 담당·Claude 뷰어 소유자 인계; 아직 미수락 |
| 집계/보관 운영 인계 | 승인된 D1→snapshot→export→report 재사용. 기간·취득자·삭제/cron·집계 보관 및 OFF/누락 표시 확정 | Song·팀원 계정 담당, 실제 추출/활성화는 별도 승인 |
| 발행률 정의/근거 확보 | 결정→첫 파일발행→원격반영→provider투구의 서로 다른 시각·키·적격/보류 분모를 고정. 기존 연구 RS-01과 중복되지 않게 조율 | Claude 기존 원장 소유자·MEAS 담당, 자료 실행 게이트 유지 |
| 동시접속/실제표시 필요성 결정 | 현 schema로 불가능한 지표는 미측정 유지 또는 최소 별도 이벤트/집계 설계. 표시 ID/visible·clock오차·누락/중복 계약 먼저; 기존 session API에 미정 필드 추가 금지 | Song 우선순위, 양쪽 계약 소유자 승인 후 별도 CLAIM |

다음 작업은 MEAS-01 문서 초안으로 연결할 수 있다. 현재 측정 부품의 존재를 운영 수집 성공이나 팬 이해·추천 정책 효과의 검증으로 바꾸지 않는다.

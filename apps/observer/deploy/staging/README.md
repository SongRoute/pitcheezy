# 라이브 저장소(선택) 스테이징 런북

`observer_app/live_store.py`는 라이브 지연 버퍼(뷰어마다 `delay_s` 전 상태를 보여 주는 스냅샷 목록)를 SQLite(WAL)에
같이 써 두는 **선택 기능**이다. `PITCHEEZY_OBSERVER_LIVE_STORE=<db 경로>`를 줄 때만 켜지고, 없으면 기존과 똑같이 메모리만 쓴다.

## 왜 필요한가 (확인한 빈틈)
- 기존 경로: MLB 피드 → `LiveFeed.buffers`(프로세스 메모리) → `/api/live/{gamePk}/state`. 원본은 `live_snapshots/*.json.gz`에 기록되지만
  재시작 때 아무것도 다시 읽지 않는다.
- 그래서 서버가 경기 중 재시작되면 (코드상) (1) 지연 D초 뷰어는 첫 스냅샷의 사건이 D초 묵을 때까지 `buffering`(D는 최대 120),
  (2) 재시작 순간 MLB 피드가 안 되면 503(스테이징 6단계에서 실측 `nostore state HTTP 503`).
- 메모리 경로는 늦게 도착한 **예전** 피드(같거나 이른 사건 시각, 더 이른 `metaData.timeStamp`)로 마지막 스냅샷 내용을 덮어써
  화면이 되감길 수 있다(`tests/test_live_store.py`의 `rewinding` 확인). 저장소 경로는 이를 `stale`로 버리고 `sync.stale_dropped`로 센다.

## 저장 규칙 (요약, 상세는 모듈 docstring)
- 슬롯 = 스냅샷 안 가장 최근 사건의 벽시계 시각(`event_wall`, 원천 시각). 도착 시각(`fetched`)과 첫 도착(`first_fetched`)은 따로 저장.
- 같은 슬롯의 내용 변경은 `version`+1인 새 행(append-only, UPDATE 금지 트리거). 같은 내용은 `duplicate`로 쓰지 않음.
- 쓰기는 게임 단위로 `BEGIN IMMEDIATE` 한 트랜잭션. 메모리 버퍼는 커밋 뒤에만 바뀐다. 저장소가 실패하면 메모리로 계속 서비스하고
  `sync.store_ok=false`·`/api/health` `live_store.error`로 알린다.
- 재생 커서 `GET /api/live/{gamePk}/log?after=<seq>&delay=<s>`: 커밋된 행만, 뷰어 지연보다 새로운 사건은 돌려주지 않는다.
- 원본(MLB 피드·녹화 파일)이 기준이다. DB는 지우고 `python -m observer_app.live_store ingest`로 다시 만들 수 있다.

## 스테이징 실행 (이 맥, 127.0.0.1만)
```sh
cd <이 브랜치 작업트리>
export PZ_STAGING_DIR=/작업전용/디렉터리          # pid·log·db·identity.jsonl·backups 가 여기에만 생김
export PITCHEEZY_STAGING_PORT=8791                 # 8766/8767/8768/8770/8771 은 스크립트가 거부
export PZ_STAGING_REPLAY_DIR=/녹화/조각/디렉터리   # 선택. 없으면 실제 MLB 피드를 조회(기록은 이 작업트리의 live_snapshots/)
sh apps/observer/deploy/staging/live_store_staging.sh start     # 저장소 켬
sh apps/observer/deploy/staging/live_store_staging.sh state 849844 0
sh apps/observer/deploy/staging/live_store_staging.sh health
sh apps/observer/deploy/staging/live_store_staging.sh status    # 프로세스 + DB quick_check·게임별 행 수
sh apps/observer/deploy/staging/live_store_staging.sh backup    # 실행 중에도 안전한 온라인 백업(SQLite backup API)
sh apps/observer/deploy/staging/live_store_staging.sh stop
sh apps/observer/deploy/staging/live_store_staging.sh restore $PZ_STAGING_DIR/backups/live-YYYYmmdd-HHMMSS.sqlite3  # 멈춘 상태에서만, 기존 db는 .replaced-* 로 남김
sh apps/observer/deploy/staging/live_store_staging.sh nostore   # 롤백: 같은 빌드를 저장소 없이(기존 기본 동작)
```
- 시작할 때마다 `identity.jsonl`에 커밋·dirty 여부·브랜치·파이썬·포트·DB·재생 경로를 한 줄 남긴다.
- ARM-B 추천(`PITCHEEZY_OBSERVER_LIVE_ARMB=0`)과 예전 화면(`LEGACY=0`)은 꺼서 상태 경로만 본다. 웹 빌드는 하지 않는다(API만).
- `com.pitcheezy.live-store-staging.plist.example`은 예시일 뿐, 설치·로드하지 않았다.

## 운영 사실 (2026-10-05 01:00 KST 스테이징 실측, 커밋은 리뷰 패킷 참고)
- 녹화 849844 스냅샷 8개(복사본, sha256 기록) 재생: 오프라인 ingest 4개 `real 0.11s`(파이썬 시작 포함, 작은 표본, 용량 주장 아님).
- 서버 재생 중 0~2번 파일이 다시 오자 `stale` 3건, 3번은 `duplicate` — 화면은 되감기지 않음.
- `kill -9` 뒤 피드 끊긴 채 재시작: 관측 상태(상황·투수·타자·투구 순서·오늘 투구) 동일, `sync={restored:true, feed_ok:false, store_ok:true}`.
- 백업→멈춤→복원→재시작: 같은 상태 제공. 저장소 끈 롤백 실행은 같은 조건에서 503(기존 빈틈).

## 프로덕션 전환 제안 (미실행, 리뷰·승인 필요)
1. 리뷰 후 8766 실행 환경에 `PITCHEEZY_OBSERVER_LIVE_STORE=<T7 아닌 로컬 SSD 경로>`만 추가(코드 기본값은 그대로).
2. 경기 전 `backup`, 경기 중 재시작 리허설 1회(`kill` 대신 `stop`/`start`).
3. 되돌리기: 환경 변수를 지우고 재시작(=기존 메모리 경로). DB 파일은 그대로 둬도 무해.
4. 보존: 경기당 약 400 스냅샷 × gzip 수십 KB. 정리 정책은 아직 없음(미측정 → 경기 후 `status`로 크기 확인).

## 한계
- 정확히 한 번(exactly-once)이나 운영 준비 완료를 주장하지 않는다. 검증은 작은 고정 표본·단일 프로세스·단일 맥.
- `stale` 판정은 `metaData.timeStamp`가 있을 때만 한다. 없으면 메모리 경로와 같은 "마지막 슬롯 교정"으로 처리.
- 재시작 직후 `fetched_at`은 복원하지 않아 바로 한 번 조회한다(의도). 시계가 크게 바뀌면 지연 계산도 바뀐다.
- 여러 uvicorn 워커가 같은 DB를 쓰는 구성은 시험하지 않았다(현재 서비스는 단일 프로세스).

## 2026-10-06 격리 통합 후보의 복구 의미
- 저장 실패 뒤 다음 poll은 현재 지연 버퍼를 먼저 재저장한다. 첫 도착 시각·슬롯 시각은 유지한다. 메모리보다 오래된 응답으로 되감지 않는다. 최대 지연 창 밖으로 이미 가지치기된 장애 중 기록은 복원하지 못하며 이 DB는 완전한 감사 원장이 아니다. 장애 중 재시작하면 아직 저장되지 않은 메모리는 소실된다.
- gzip/JSON 복원 오류는 메모리로 폴백하고 재시작 전까지 degraded를 유지한다. 손상 DB는 자동 수정하지 않는다. 격리 백업 점검 후 새 DB 또는 검증된 백업으로 재시작해야 한다.
- health는 configured 및 오류 종류만 공개한다. 자세한 오류/경로는 서버 로그에 남는다.
- stop은 PID 시작 시각과 전체 명령이 launch 기록과 같을 때만 종료한다. identity가 없는 과거 인스턴스는 자동 종료하지 않는다.
- 단일 worker/단일 DB writer만 지원하며 실행 중 CLI ingest 금지. 보존 정책·전체 경기 크기·실제 동시 기기는 미검증. 현재 통합은 별도 작업트리 후보이며 8766/8771 미배포.

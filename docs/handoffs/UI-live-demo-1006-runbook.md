# UI 개편·라이브 시연 운영 문서 (2026-10-06 16:00 KST 보고용)

작성: review-integration-002 → integration-resume-005 → 006 (같은 Claude Opus 5.5 세션). 최종 갱신 2026-10-05 22:08 KST (호스트 시계 확인). 마지막으로 읽은 PM 노트: PM-REVIEW-NOTES.md의 22:02 이후 Root 노트("Root read qa-preview8771-final …").
단계 로그: `/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign/LIVE-INTEGRATION.md` (append-only). 외부 요약: 같은 폴더 `DELIVERY.md`.

## 0. 현재 상태 (한눈에)
- 미리보기: `http://100.108.252.111:8771/watch` (Tailscale 기기 전용). PID **21786** (21:57:55 시작), cwd `pitcheezy-worktrees/codex-ui-live-preview-1005`.
- 서빙 소스 커밋: **`641c7c3`** (브랜치 `codex/ui-live-preview-1005`, base `282dc6f`). 22:07:30에 바깥 빌드(`/tmp/pz-build-006`)를 검토·테스트한 뒤 서빙 dist로 옮겼다(프로세스 재시작 없음).
- 서빙 자산: `index-BXsRGI24.js` sha256 `d609982fc8cf23f8…`, `index-DFQ6H-5X.js`(운영자 화면, 지연 로드) `e0bdb6e2…`, `index-BhmuV1aO.css` `2e9a740cd5cdb8ca…`, `index.html` `f8fc7056f0e62211…`.
- 데이터: overlay-014 읽기 전용 사본(기존 15경기 바이트 동일 + 813031) = 16경기. 시연 저장소 `/Users/song/Projects/pitcheezy-demo-sessions/preview8771`. 활성 시연 세션 없음.
- 단계별: 구현 ✅ / 통합 검토 ✅ / 미리보기 반영 ✅ / **실기기: 일부만** (아래 §1-물리).

## 1. 단계별 상태
구현·검토·미리보기
- §2 목록(KST 원래 시작시각 날짜, 날짜 아래 평면 목록, 점수·상태, 더블헤더 `N차전`, 모드 선택·상단 실험 문구 없음, 한계는 '정보' 시트): 구현·검토·반영 완료. 849843 = **2026-09-30(수) 11:00 KST**(MLB officialDate 9/29는 표시에 안 씀).
- §3·§4 공통 경기 화면·존 중심 추천(라이브·종료·시연 같은 화면, 존이 가장 큼, 1·2·3 같은 크기, 실제=주황 원+'실제'/추천=점선 번호 네모, '정책 선택 비중' 원본 값, 미지원은 이유만): 완료.
  - 한 화면(헤드리스 Chromium, 종료 경기·라이브 최신): 320×568·360×640·375×553·390×600·375×667·390×844, 기본·125% 글자 모두 통과(8771 QA 157/159).
  - 라이브 지난 기록(고정)·연결 끊김 상태(22:04~22:07 집중 점검, 320×568·375×553 × 기본·125%): **60 ok / 4 FAIL**. 375×553에서 125% 글자일 때 고정·끊김 두 상태 모두 22px 스크롤(투구 번호 '1'·'자세히'가 첫 화면 밖, 잘림 없음). 원인: 버튼 3개(최신으로·자동·새로고침)가 둘째 줄로 내려가고 날짜 제목이 두 줄. 이번 수정(`641c7c3`, 문구 축약)으로 320×568 전 상태 통과, 375×553 125%는 46px→22px로 줄었지만 **미해결**.
  - 허용된 한계(Root 수용): 320×460·288×512에서 존이 타석 영역보다 약간 작음.
- §5 라이브 동작(자동 갱신 기본·끄기·수동 새로고침·과거 고정·'최신으로'·LIVE는 경기 상태만·끊김 표시와 재시도): 픽스처·시연 세션으로 검증. **진행 중인 실제 MLB 경기로는 미검증**(확인 시각에 진행 경기 없음). 이 실행은 `LIVE_ARMB=0`이라 실제 라이브의 '다음 공 추천'은 "제공하지 않음"으로 표시된다.
- §6 타석·투구 탐색(이닝 버튼 없음, 전체 타석 시트, '타석 진행 중', 투구 번호, 원본 필드 기반 결과 예 '중견수 방면 홈런'): 완료.
- §7 시점별 추천(최신=방금 던진 공/다음 공 추천 분리, 과거=당시 추천+실제, 공개 버튼 없음, 없음은 없다고): 완료. 저장 추천은 `asof_replay` = "경기 뒤, 그 공을 던지기 전 정보만으로 계산한 추천"으로 표기.
- §8·§10 시연 시계(서버 공통 시계, 운영자 토큰, 세션 격리, 재접속·두 클라이언트 같은 상태, 시작 전 상태 복원, 미래 기록 비노출, 배너 없음): 완료. 투구 사이 사건(도루·견제·폭투 득점·교체)은 다음 투구 공개 때까지 보류, 최종 상태는 마지막 사건 뒤. 두 브라우저 컨텍스트 수렴 ≤1.8초(허용 4.5초).

§1-물리 (실기기)
- ✅ **Song 아이폰 Safari, 기본 글자 (21:50~21:51 응답)**: `/watch/849843`에서 "스크롤 없이 보이고 조작도 돼요"(투구 번호·자세히가 스크롤 없이 보이고 타석 선택·상세 동작). 확인한 것은 013 이전 새 UI 배치다(10e901e 빌드, 그 뒤 d478795 운영자 수정 빌드 중 어느 것을 받았는지는 알 수 없음). 옛 282dc6f 화면이 아니다.
- ✅ **Song 맥북, MLB 공식 YouTube `45Rqrp4aZM4` (21:49 응답)**: "로그인 없이 재생돼요" = 813031 전체 경기 영상의 무료 재생 확인.
- ✗ 미확인: 현재(013 이후) 배치의 실기기 표시, 125% 글자 실기기, 맥북+아이폰 두 기기 시연 시계, 영상과 결과 시각 대응, 5분 실리허설.

## 2. 실행 주소
- 목록 `http://100.108.252.111:8771/watch` (날짜 ‹ ›)
- 849843 Cubs@Padres `/watch/849843` — 2026-09-30(수) 11:00 KST, 최종 0–8, CV 미트 x 있음.
- 813031 Brewers@Dodgers (2025 NLCS G4) `/watch/813031` — **2025-10-18(토) 09:38 KST**, 최종 1–5, 추천 262/262 `asof_replay`(review-011이 26개 대상 불변성 확인, 기능 재현용), CV 없음.
- 운영자 `/demo-operator` (토큰 필요). 관람 주소는 세션 생성 시 `/watch/<경기>?demo=<세션ID>`로 나온다. 일반 화면엔 시연 표시가 없다.
- 시연 세션이 ready/running/paused인 동안 그 경기의 일반 주소는 잠긴다(목록 점수 비움, 상세 "지금 볼 수 없어요"). 미래 노출 방지 설계.
- overlay는 고정된 사본이다. T7에 새로 쌓이는 보관 경기를 따라가지 않는다(임시 미리보기용). 실제 라이브 경기는 MLB 실시간 피드에서 그대로 온다(보관 경기로 착각하지 않는다).

## 3. 시연 운영 절차
대응표 현황 (둘 다 **진행 속도표**, 영상 대조 아님)
- `g849843-ab4-8-pace`: 1회 말 AB4~8, 11구, 298초. `game_timestamps_unsynced`, verified=false.
- `g813031-ab1-3-pace`: 1회 초 AB1~3, 16구, 316.6초. `game_timestamps_unsynced`, verified=false.
- 둘 다 경기 피드 시각으로 만든 것이다. 영상 재생 가능 ≠ 동기화. 시작 시점 하나를 맞춘다고 이후 투구가 영상과 맞는 것이 아니다(편집·광고·리플레이). 투구 공개 시점은 **영상에서 결과가 보이는 순간** 기준이어야 하고 릴리스 순간이 아니다.

지금 할 수 있는 것: 시계만 쓰는 기능 재현 (영상 없이)
1. 맥미니에서 토큰 출력(화면에만, 문서·메시지에 붙이지 않음):
   `PITCHEEZY_DEMO_DIR=/Users/song/Projects/pitcheezy-demo-sessions/preview8771 /Users/song/Projects/pitcheezy/.venv-observer/bin/python -c "import sys; sys.path.insert(0,'/Users/song/Projects/pitcheezy-worktrees/codex-ui-live-preview-1005/apps/observer/backend'); from observer_app.demo_clock import DemoClock; print(DemoClock().token())"`
2. 맥북 `/demo-operator` → 토큰 입력 → 대응표 선택 → 세션 만들기(ready: 시작 직전 상태가 보임) → 관람 주소를 아이폰에서 연다.
3. '시작'(1배). 일시정지·재개·특정 투구로 이동·처음부터 다시·속도는 운영자 화면에서. 화면에 마지막 공개 투구, 진행 시각, 다음 공개까지 초가 보인다.
4. 발표 전 "과거 경기 기록을 실제 진행처럼 재현"한다고 고지하고, 영상과 맞췄다고 말하지 않는다.
5. 끝나면 '시연 종료(전체 기록 다시 열기)'. 구간 끝(마지막 대응 투구)에서는 시계가 멈출 뿐 경기 최종 결과는 공개되지 않는다.

영상과 함께 하려면 (조건: 검증된 결과 시각 대응표가 생긴 뒤에만)
- 다음 단계(유료 계정 불필요): Song이 맥북에서 `45Rqrp4aZM4`를 재생하며 813031 1회 초 AB1~3의 16구 각각에 대해 "결과가 화면에 보이는 영상 시각(초)"을 적는다. 틀은 `pm-jobs/20261005-ui-live-redesign/media-sequential-012/mapping-813031-AB1-3-PENDING.csv`(현재 0/16 PENDING). 기준점 3개 이상(첫 공, 각 삼진 마지막 공)으로 오차를 확인하고, 편집·리플레이가 끼면 비고에 적는다.
- 그 CSV로 `python apps/observer/scripts/build_demo_mapping.py --game 813031 --from-csv <csv> --video <파일 또는 영상 ID 기록> --verified-by <이름> --id g813031-ab1-3-video`를 만들고 검토한 뒤에만 `kind=video_verified`로 쓴다.
- 그 뒤에야: "3, 2, 1"에 영상 재생과 '시작'을 같이 누르고, 정지는 영상→시계, 재개는 시계→영상. 서버는 영상 플레이어를 제어하지 않는다.
- 그 전까지는 위 '시계만' 절차를 쓴다. 속도표를 영상에 대고 한 번 맞춰 동기화된 것처럼 진행하지 않는다.

대체 절차
- 아이폰 접속 실패: 맥북 브라우저에서 같은 관람 주소(서버 시계라 같은 상태).
- 영상 플레이어 실패: 시계 일시정지 → '시계만' 절차로 마지막 공개 투구부터.
- 재접속: 새로고침·셀룰러 전환 뒤 같은 투구로 돌아온다(두 브라우저 컨텍스트로 확인, 두 실기기 미확인).

## 4. 되돌리기 (8771만)
현재: PID 21786, cwd `codex-ui-live-preview-1005`, env = 교체 전 env + `PITCHEEZY_DEMO_DIR=/Users/song/Projects/pitcheezy-demo-sessions/preview8771` + `PITCHEEZY_WATCH_DIR`/`PITCHEEZY_FEED_DIR=/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign/preview-data-overlay-014/overlay/{watch,feeds}`. 로그 `review-integration-002/preview8771-integrated.log`.
교체 전: PID 80222, cwd `codex-showcase-1006` @282dc6f, 자산 `index-C2cOnYq0.js`, `index.html` sha256 `47df5af1…`, WATCH/FEED/DEMO 변수 없음(코드 기본값 = T7).
```
kill $(lsof -tnP -iTCP:8771 -sTCP:LISTEN)        # 현재 PID 21786
cd /Users/song/Projects/pitcheezy-worktrees/codex-showcase-1006
env -u PITCHEEZY_WATCH_DIR -u PITCHEEZY_FEED_DIR -u PITCHEEZY_DEMO_DIR \
    PITCHEEZY_OBSERVER_PORT=8771 PITCHEEZY_OBSERVER_HOST=tailscale PITCHEEZY_OBSERVER_LEGACY=0 PITCHEEZY_OBSERVER_LIVE_ARMB=0 \
    PITCHEEZY_OBSERVER_PYTHON=/Users/song/Projects/pitcheezy/.venv-observer/bin/python \
    PITCHEEZY_DATA_DIR=/Users/song/Projects/pitcheezy/data PITCHEEZY_RUNS_DIR=/Users/song/Projects/pitcheezy/runs \
    nohup sh apps/observer/serve.sh >> /Users/song/Projects/pitcheezy-pm-jobs/20261004-autonomy/preview8771.log 2>&1 &
curl -s http://100.108.252.111:8771/api/health
```
8766·8770·기록기는 교체·재시작하지 않았다.

## 5. 소스 (최종 검증된 것)
`codex/ui-live-preview-1005` = 282dc6f 위에:
- `372b129` backend-001 `e33d59f` 서비스 API v2·서버 공통 시연 시계·운영자 API/화면·우회 경로 잠금
- `e9b6260` state-fix-003 `9e6e0ae` 피드 사건 순서로 투구 전 상황(15경기 4,319구 불일치 74→0)
- `a0ed276` 통합: 투구 사이 사건 보류, 종료 상태는 마지막 사건 뒤
- `61ab735` frontend-001 `9cd4d71` KST 목록·공통 경기 화면
- `453d25b` 통합: 일반 번들에서 옛 영상·쇼케이스 제외(MLB 영상 주소·장면 투구 목록 없음), 존 라벨 크기
- `d3e0fbe` 통합: temporal-review-008 F1~F3
- `10e901e` 통합: 후보 2줄·존 최소 높이
- `d478795` 통합: 운영자 화면 HTTP 동작(`crypto.randomUUID` 제거)
- `8c56c7f` responsive-fix-013 `7571f1e`: 짧은/좁은 화면 존·후보 나란히, 125% 글자 한 화면
- `641c7c3` 통합: 라이브 고정·끊김 상태 문구 축약(한 줄 상태 표시)
- 문서 커밋은 그 뒤(이 파일).

과정상 실수(최종 소스와 분리해 기록): 21:40 교체 뒤 `d478795`, `8c56c7f` 두 번은 서빙 dist에 바로 빌드해 검증 전에 반영됐다(곧바로 8771에서 재검증해 통과). 22:07의 `641c7c3`은 바깥 빌드로 테스트한 뒤 옮겼다.

## 6. 검증 근거 (`pm-jobs/20261005-ui-live-redesign/review-integration-002/`)
- 백엔드 pytest 287 passed. temporal-review-008 테스트 5/5(이 트리).
- 탐침 `demo_probe.py`(8771 실서버): 849843·813031 각 90/90 (`probe-preview8771-e-*.txt`), 최종 번들 849843 90/90 (`probe-preview8771-g.txt`).
- 엄격 브라우저 QA `integrated_qa.mjs`(8771 실서버, 헤드리스 Chromium): 157/159 (`qa-preview8771-final.log`; 실패 2 = 허용된 320×460·288×512).
- 라이브 고정·끊김 집중 점검 `live_pinned_stale_fit.mjs`(픽스처): 수정 전 6 FAIL (`live-pinned-stale-8c56c7f.log`), 수정 후 60 ok/4 FAIL (`live-pinned-stale-fix2.log`, 스크린샷 같은 이름 폴더). 하네스 정정 2건(새 문구, `.sr-only` 제외)은 `live_pinned_stale_fit.v1.mjs`로 원본 보존.
- 픽스처 테스트(바깥 빌드): `responsive-fit-smoke` 통과, `service-ui-smoke` 110 ok.
- 이전 QA 기록(`qa-dryrun`, `qa-integrated-1~6`, `qa-preview8771*`)은 보존. `qa-integrated-3~5`는 Root가 거부한 '확대 대용 자동 통과' 규칙이라 근거로 쓰지 않는다.
- 옛(legacy) 웹 테스트는 실행하지 않았다.

## 7. 남은 문제
1. 라이브 고정·끊김 상태, 375×553, 125% 글자: 22px 스크롤(미해결).
2. 실기기: 013 이후 배치, 125% 글자, 맥북+아이폰 두 기기 시계 — 미확인.
3. 영상: 849843 전체 영상 막힘(로그인 서비스 없음, 13:51 압축본은 전체 아님). 813031 무료 공식 영상은 Song 재생 확인, 결과 시각 대응 0/16, 5분 무편집 연속성 미확인(편집판으로 쉬는 시간 약 32분 컷), 두 기기 리허설 미실시. 헤드리스 브라우저 재생은 43.9초에서 오류로 중단, 그 경로는 닫았다.
4. 진행 중인 실제 MLB 경기 미검증, 실라이브 다음 공 추천 꺼짐(LIVE_ARMB=0).
5. 허용된 한계: 320×460·288×512 존 크기.
6. 운영자 토큰은 Tailscale HTTP로 오간다. 운영자 화면은 Song 기기에서만.
7. 시계 상태는 단일 uvicorn 프로세스 기준(파일 원자 교체).

## 8. 사고 공개
- frontend-001이 21:03~21:05 옛 웹 테스트를 일괄 실행해 `127.0.0.1:8766`(PID 94628, 9/23 C-D-FULL-001 검토 서버)에 세션 4개를 만들었고(T7 `observer.sqlite3`에 영구 기록), T7 `C-D-FULL-001/browser/` 산출물 6개를 덮어썼다. 원본 일부는 복구 수단이 없다. 사용자용 데모 `100.108.252.111:8766`(PID 3277)은 닿지 않았다(incident-audit-004). 복구·삭제는 하지 않았다(Root 결정).
- 따라서 이 작업 전체에 대해 "8766 무접촉"이라고 말할 수 없다. 통합 담당은 8766·8770에 요청·재시작을 하지 않았고 PID가 그대로다(8766: 3277·94628, 8770: 149).

## 9. 작업자 시각 (호스트 시계·파일 기준)
media-001 20:47~20:52 · backend-001 20:47~21:00 · state-fix-003 21:04~21:11 · frontend-001 20:47~21:24(보고) · incident-audit-004 21:12~21:16 · media-rework-007 21:22~21:25 · temporal-review-008 ~21:28 · overlay-014 21:48~21:53 · responsive-fix-013 21:43~21:54(커밋 7571f1e) · 통합 세션 20:53 시작 → 21:16경 보조 대기 명령만 Root가 중단 → 21:17 재개(005) → 21:40:03 첫 교체 → 21:57:55 재시작 → 22:02 반환 → 22:03 재개(006) → 22:07:30 자산 갱신 → 22:08 문서.

## 10. 다음 실기기 확인 (Root 경유, 이미 확인된 기본 확인은 반복하지 않음)
1. 현재 배치(013 이후) 기본 글자: 아이폰 Safari `/watch/849843`에서 존 왼쪽·후보 오른쪽 배치가 스크롤 없이 보이는지(짧은 화면일 때).
2. 라이브처럼 보기: 운영자가 세션 시작 → 아이폰 관람 주소에서 투구 번호로 지난 공 고정, '최신으로' 복귀, 자동 끄기/새로고침.
3. 맥북 브라우저와 아이폰에서 같은 관람 주소: 같은 투구, 새로고침 뒤 같은 상태, 정지·재개.
4. (선택) 아이폰 큰 글자 설정에서 같은 화면.
5. 813031 영상 결과 시각 16개 기록(§3), 그 뒤 5분 영상 리허설.

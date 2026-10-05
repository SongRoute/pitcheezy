# UI 개편·라이브 시연 운영 문서 (2026-10-06 16:00 KST 보고용)

작성: review-integration-002 → integration-resume-005/006 → 019 (같은 Claude Opus 5.5 세션). 최종 갱신 2026-10-05 22:23 KST (호스트 시계). 마지막으로 읽은 PM 노트: PM-REVIEW-NOTES.md의 "Root read mapping-review-018/REVIEW.md and accepts5ec5fa9 …" 노트.
단계 로그: `/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign/LIVE-INTEGRATION.md` (append-only). 외부 요약: 같은 폴더 `DELIVERY.md`.

## 0. 현재 상태 (한눈에)
- 미리보기: `http://100.108.252.111:8771/watch` (Tailscale 기기 전용). PID **25166** (22:22:03 시작), cwd `pitcheezy-worktrees/codex-ui-live-preview-1005`.
- 서빙 소스 커밋: **`15a4357`** (브랜치 `codex/ui-live-preview-1005`, base `282dc6f`). 바깥 빌드 `/tmp/pz-build-019`를 집중 점검·픽스처 테스트한 뒤 서빙 dist로 옮기고, 백엔드가 바뀌었으므로(5ec5fa9) 같은 env로 재시작했다.
- 서빙 자산 sha256: `index-BZFtlKyU.js` `f162c563ffec93cd…`, `index-ZR9im65s.js`(운영자, 지연 로드) `035f558e97ca6862…`, `index-B2ja2VoT.css` `e7934071bfc6aec1…`, `index.html` `33ff1422b4daffcb…`.
- 데이터: overlay-014 읽기 전용 사본 16경기(기존 15 바이트 동일 + 813031). 시연 저장소 `/Users/song/Projects/pitcheezy-demo-sessions/preview8771`(대응표 2개, 둘 다 진행 속도표 verified=false). 활성 시연 세션 없음.
- 단계별: 구현 ✅ / 통합 검토 ✅ / 미리보기 반영 ✅ / **실기기: 일부만**(§1-물리).

## 1. 단계별 상태
구현·검토·미리보기
- §2 목록(KST 원래 시작시각 날짜, 날짜 아래 평면 목록, 점수·상태, 더블헤더 `N차전`, 모드 선택·상단 실험 문구 없음, 한계는 '정보' 시트): 구현·검토·반영 완료. 849843 = **2026-09-30(수) 11:00 KST**(MLB officialDate 9/29는 표시에 안 씀).
- §3·§4 공통 경기 화면·존 중심 추천(라이브·종료·시연 같은 화면, 존이 가장 큼, 1·2·3 같은 크기, 실제=주황 원+'실제'/추천=점선 번호 네모, '정책 선택 비중' 원본 값, 미지원은 이유만): 완료.
  - 한 화면(헤드리스 Chromium, 종료 경기·라이브 최신): 320×568·360×640·375×553·390×600·375×667·390×844, 기본·125% 글자 모두 통과(8771 엄격 QA 157/159는 017 이전 빌드 기준, 최종 빌드는 responsive-fit 392 ok).
  - 라이브 지난 기록(고정)·연결 끊김 상태(320×568·375×553 × 기본·125%): 최종 빌드 **64/64 통과**(22:21, `live-pinned-stale-019.log`). 375×553·125%의 22px 넘침은 layout-final-017(`a0c341c` → `15a4357`, 620px 이하 높이에서 줄 간격·여백만 조정, 글자·컨트롤·내용 변화 없음)로 해결. 여유가 약 3px뿐이라 Safari 글꼴 차이는 실기기로 확인해야 한다.
  - 허용된 한계(Root 수용): 320×460·288×512에서는 페이지가 스크롤되고 존이 타석 영역보다 약간 작다(통과로 세지 않음).
- §5 라이브 동작(자동 갱신 기본·끄기·수동 새로고침·과거 고정·'최신으로'·LIVE는 경기 상태만·끊김 표시와 재시도): 픽스처·시연 세션으로 검증. **진행 중인 실제 MLB 경기로는 미검증**(확인 시각에 진행 경기 없음). 이 실행은 `LIVE_ARMB=0`이라 실제 라이브의 '다음 공 추천'은 "제공하지 않음"으로 표시된다.
- §6 타석·투구 탐색(이닝 버튼 없음, 전체 타석 시트, '타석 진행 중', 투구 번호, 원본 필드 기반 결과 예 '중견수 방면 홈런'): 완료.
- §7 시점별 추천(최신=방금 던진 공/다음 공 추천 분리, 과거=당시 추천+실제, 공개 버튼 없음, 없음은 없다고): 완료. 저장 추천은 `asof_replay` = "경기 뒤, 그 공을 던지기 전 정보만으로 계산한 추천"으로 표기.
- §8·§10 시연 시계(서버 공통 시계, 운영자 토큰, 세션 격리, 재접속·두 클라이언트 같은 상태, 시작 전 상태 복원, 미래 기록 비노출, 배너 없음): 완료. 투구 사이 사건(도루·견제·폭투 득점·교체)은 다음 투구 공개 때까지 보류, 최종 상태는 마지막 사건 뒤. 두 브라우저 컨텍스트 수렴 ≤1.8초(허용 4.5초).

§1-물리 (실기기, 일부)
- ✅ Song 아이폰 Safari, 기본 글자 (21:50~21:51 응답): `/watch/849843`에서 "스크롤 없이 보이고 조작도 돼요". 013 이전 새 UI 배치(10e901e 또는 d478795 빌드 — 어느 쪽을 받았는지는 모름). 옛 282dc6f 화면이 아니다.
- ✅ Song 맥북, MLB 공식 YouTube `45Rqrp4aZM4` (21:49 응답): "로그인 없이 재생돼요" = 813031 전체 경기 영상 무료 재생 확인. 대응표가 아니다.
- ✗ 미확인: 현재 배치(013·017 이후) 실기기, 125% 글자 실기기, 맥북+아이폰 두 기기 시연 시계, 영상 결과 시각 16개, 5분 영상 리허설.

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

영상과 함께 하려면 (조건: 16구 모두 관찰한 검증 대응표를 가져온 뒤에만)
- 재생 확인(21:49 Song 맥북)은 대응표가 아니다. 영상 재생 가능 ≠ 동기화.
- 다음 할 일(계정·구매 불필요, 사람이 직접):
  0. 현재 813031 결과 시각 **3/16 관찰, 13 대기** (observed-anchors-020, Song 원문 "7:28, 8:58, 11:13" + "각 타석의 마지막 결과 시각이에요", 기록 `observed-anchors-020/OBSERVATIONS.md`):
     - idx5 `813031:1:6` 투랭 볼넷 7:28 = 448초
     - idx9 `813031:2:4` 추리오 삼진 8:58 = 538초
     - idx15 `813031:3:6` 옐리치 삼진 11:13 = 673초
     나머지 13구(타석 중간 공)는 비어 있다. 보간하지 않는다. 16행이 모두 채워지기 전에는 가져오기·`video_verified`·구간 연속성 주장 없음. Root가 13개를 Song에게 비동기로 요청함(응답 대기).
  1. **3/16이 채워진 `pm-jobs/20261005-ui-live-redesign/observed-anchors-020/mapping-813031-AB1-3-PARTIAL-3of16-OBSERVED.csv`(sha256 `b8ebe23a…`)의 사본**에 나머지 13행을 채운다. 빈 틀 `remote-mapping-016/mapping-813031-AB1-3-REVIEW-TEMPLATE.csv`(sha256 `ed60612a…`)는 참고용으로만 둔다. (예전 `media-sequential-012/mapping-813031-AB1-3-PENDING.csv`는 하이픈 키라 가져오기에서 거부된다. 쓰지 않는다.)
  2. 맥북에서 `https://www.youtube.com/watch?v=45Rqrp4aZM4`를 보며 1회 초 AB1~3의 16구 각각에 대해 `video_offset_result_visible_s` = **그 공의 결과가 화면에 처음 보이는 영상 시각(초)**을 적는다(포수 포구+볼/스트라이크 판정, 헛스윙, 파울, 인플레이; 타석 마지막 공은 삼진·아웃 확정). 릴리스 순간이 아니다. 행마다 `status=OBSERVED_RESULT_VISIBLE`, `verified_by`, `evidence`(예: "포구+볼 판정, 스코어버그 1-0")를 채운다.
  3. 같은 창 안에 편집 컷·리플레이 삽입이 있는지 보고 비고에 적는다. AB1~3이 끊김 없이 이어져야 한다. 영상(2:08:35)이 경기(161분)보다 짧아 편집판인 것은 알지만, 무엇이 빠졌는지는 이 구간을 직접 확인하기 전까지 모른다.
  4. 16행이 모두 채워졌을 때만 가져온다(일부 행만이면 거부되거나 짧은 구간이 되며, AB1~3 완료라고 부르지 않는다):
```
cd /Users/song/Projects/pitcheezy-worktrees/codex-ui-live-preview-1005 && /Users/song/Projects/pitcheezy/.venv-observer/bin/python apps/observer/scripts/build_demo_mapping.py \
  --game 813031 --id g813031-ab1-3-yt-observed \
  --from-csv <채운 사본 경로> \
  --video-url 'https://www.youtube.com/watch?v=45Rqrp4aZM4' --observed-channel '<화면에 보인 채널명>' \
  --observed-duration-s <플레이어에 보인 길이(초)> --observed-on <본 날짜 YYYY-MM-DD> --verified-by '<이름>' \
  [--evidence-file <메모·스크린샷 묶음>] \
  --feed-dir /Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign/preview-data-overlay-014/overlay/feeds \
  --demo-dir /Users/song/Projects/pitcheezy-demo-sessions/preview8771
```
     결과 대응표는 `kind=video_verified`, `video.source=remote_observed`, `video.sha256=null`(스트림만 봤으므로 영상 바이트 해시는 없다), `verification.csv_sha256`=채운 CSV의 해시. URL을 해시해 영상 해시처럼 쓰지 않는다. 대응표는 바꾸지 않고 새 `--id`로 다시 만든다. 운영자 화면 대응표 목록에는 `verified`로만 보이므로 원격 관찰본인지는 이 문서와 대응표 JSON의 `video.source`로 확인한다.
- 맞춰 시작하는 법(검증 대응표가 있을 때):
  1. 운영자 화면에서 그 대응표로 세션을 만든다. 시계의 처음 진행 시각 = 선택한 시작 행의 `reveal_at_s` − `lead_s`(기본 5초). 운영자 화면 '영상 기준 시각'에 그 값이 보인다.
  2. YouTube 플레이어를 **그와 같은 시각**으로 옮기고 일시정지한다(0초·첫 투구 릴리스·임의의 시작점이 아님).
  3. "3, 2, 1"에 영상 재생과 '시작'을 손으로 같이 누른다. 정지는 영상→시계, 재개는 시계→영상, 둘 다 손으로. 서버는 YouTube를 제어하지 않는다.
  4. 각 공의 결과는 영상에 결과가 보이는 시각에 화면에 공개되고, 그때 다음 공 추천으로 넘어간다.
- 그 전까지는 위 '시계만' 절차를 쓴다. 속도표(`*-pace`)를 영상에 한 번 맞춰 동기화된 것처럼 진행하지 않는다. 브라우저 자동 재생 경로는 닫았다(다시 시도하지 않음).

대체 절차
- 아이폰 접속 실패: 맥북 브라우저에서 같은 관람 주소(서버 시계라 같은 상태).
- 영상 플레이어 실패: 시계 일시정지 → '시계만' 절차로 마지막 공개 투구부터.
- 재접속: 새로고침·셀룰러 전환 뒤 같은 투구로 돌아온다(두 브라우저 컨텍스트로 확인, 두 실기기 미확인).

## 4. 되돌리기 (8771만)
현재: PID 25166, cwd `codex-ui-live-preview-1005` @15a4357, env = 교체 전 env + `PITCHEEZY_DEMO_DIR=/Users/song/Projects/pitcheezy-demo-sessions/preview8771` + `PITCHEEZY_WATCH_DIR`/`PITCHEEZY_FEED_DIR=/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign/preview-data-overlay-014/overlay/{watch,feeds}`. 로그 `review-integration-002/preview8771-integrated.log`.
교체 전: PID 80222, cwd `codex-showcase-1006` @282dc6f, 자산 `index-C2cOnYq0.js`, `index.html` sha256 `47df5af1…`, WATCH/FEED/DEMO 변수 없음(코드 기본값 = T7).
```
kill $(lsof -tnP -iTCP:8771 -sTCP:LISTEN)        # 현재 PID 25166
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
- `e9b6260` state-fix-003 `9e6e0ae` 피드 사건 순서로 투구 전 상황
- `a0ed276` 통합: 투구 사이 사건 보류, 종료 상태는 마지막 사건 뒤
- `61ab735` frontend-001 `9cd4d71` KST 목록·공통 경기 화면
- `453d25b` 통합: 일반 번들에서 옛 영상·쇼케이스 제외, 존 라벨 크기
- `d3e0fbe` 통합: temporal-review-008 F1~F3
- `10e901e` 통합: 후보 2줄·존 최소 높이
- `d478795` 통합: 운영자 화면 HTTP 동작
- `8c56c7f` responsive-fix-013 `7571f1e`: 짧은/좁은 화면 존·후보 나란히
- `641c7c3` 통합: 라이브 고정·끊김 상태 문구 축약
- `1293bd1` remote-mapping-016 `5ec5fa9` (review-018 ACCEPT): `--video-url` 원격 관찰 대응표 가져오기(`video.source=remote_observed`, `sha256=null`, 16행 모두 관찰·검토 필수), 검토용 CSV 틀 생성
- `15a4357` layout-final-017 `a0c341c`: 375×553·125% 라이브 고정·끊김 한 화면
- 문서 커밋은 그 뒤.

과정상 실수(최종 소스와 분리): 21:40 교체 뒤 `d478795`, `8c56c7f`는 서빙 dist에 바로 빌드돼 검증 전에 반영됐다(즉시 8771 재검증 통과). 그 뒤(`641c7c3` 22:07, `15a4357` 22:22)는 바깥 빌드로 테스트한 뒤 옮겼다.

## 6. 검증 근거 (`pm-jobs/20261005-ui-live-redesign/review-integration-002/`)
- 백엔드 pytest(`apps/observer/backend`, 1293bd1 통합 후): **320 passed**. review-018: 작성자 33 + 독립 탐침 36 + 시연 회귀 56 통과, 실제 PENDING CSV는 가져오기 거부(대응표 미생성).
- 집중 점검 `live_pinned_stale_fit.mjs`(원본 그대로, 픽스처, 헤드리스 Chromium): 8c56c7f 6 FAIL → 641c7c3 60/4 → **최종 바깥 빌드 64/64** (`live-pinned-stale-019.log`, 스크린샷 `live-pinned-stale-019/`). 017 저장소의 사본과 판정 내용 동일(머리 주석만 다름).
- 픽스처 테스트(최종 바깥 빌드): `responsive-fit-smoke` 392 ok·INFO 8(허용된 320×460·288×512), `service-ui-smoke` 110 ok.
- 8771 재시작 뒤: health·`/watch`·`/watch/849843`·`/watch/813031`·`/demo-operator`·2025-10-18 목록·849843 CV setup 200, 탐침 813031 **90/90** (`probe-preview8771-019.txt`; 탐침 세션은 스스로 종료), 활성 세션 없음.
- 이전(22:07 이전) 기록: 엄격 브라우저 QA 157/159 (`qa-preview8771-final.log`, 실패 2 = 허용 한계), 탐침 849843·813031 각 90/90. 레이아웃이 바뀐 017 뒤 전체 엄격 QA는 다시 돌리지 않았다(범위 밖, 집중·픽스처 점검으로 대체).
- 옛(legacy) 웹 테스트는 실행하지 않았다.

## 7. 남은 문제 (사용자만 할 수 있는 것 포함)
1. 실기기: 현재 배치(013·017 이후) 기본·큰 글자, 맥북+아이폰 두 기기 시연 시계 — 미확인. 375×553·125%는 여유 3px라 Safari 확인 필요.
2. 813031 영상 결과 시각: 3/16 관찰(각 타석 마지막 공 448·538·673초), 13 대기 — Song 응답 대기(§3). 16개 전에는 가져오기·영상 동기화·5분 리허설 불가. 구간 연속성 미확인.
3. 849843 전체 영상은 여전히 막힘(로그인된 다시보기 서비스 없음, 13:51 압축본은 전체 아님).
4. 진행 중인 실제 MLB 경기 미검증, 실라이브 다음 공 추천 꺼짐(LIVE_ARMB=0).
5. 허용된 한계: 320×460·288×512.
6. 운영자 화면 대응표 목록은 원격 관찰본도 `verified`로만 보인다(출처는 대응표 JSON `video.source`와 이 문서로 확인).
7. 운영자 토큰은 Tailscale HTTP로 오간다. 운영자 화면은 Song 기기에서만. 시계 상태는 단일 uvicorn 프로세스 기준.

## 8. 사고 공개
- frontend-001이 21:03~21:05 옛 웹 테스트를 일괄 실행해 `127.0.0.1:8766`(PID 94628, 9/23 C-D-FULL-001 검토 서버)에 세션 4개를 만들었고(T7 `observer.sqlite3`에 영구 기록), T7 `C-D-FULL-001/browser/` 산출물 6개를 덮어썼다. 원본 일부는 복구 수단이 없다. 사용자용 데모 `100.108.252.111:8766`(PID 3277)은 닿지 않았다(incident-audit-004). 복구·삭제는 하지 않았다(Root 결정).
- 따라서 이 작업 전체에 대해 "8766 무접촉"이라고 말할 수 없다. 통합 담당은 8766·8770에 요청·재시작을 하지 않았고 PID가 그대로다(8766: 3277·94628, 8770: 149).

## 9. 작업자 시각 (호스트 시계·파일 기준)
media-001 20:47~20:52 · backend-001 20:47~21:00 · state-fix-003 21:04~21:11 · frontend-001 20:47~21:24(보고) · incident-audit-004 21:12~21:16 · media-rework-007 21:22~21:25 · temporal-review-008 ~21:28 · responsive-fix-013 21:43~21:54 · overlay-014 21:48~21:53 · remote-mapping-016 ~22:11(보고) · layout-final-017 22:13~22:20(커밋) · mapping-review-018 22:16(보고) · 통합 세션 20:53 시작 → 21:16경 보조 대기 명령만 Root가 중단 → 21:17 재개 → 21:40:03 첫 교체 → 21:57:55 재시작 → 22:02 반환 → 22:03 재개 → 22:07:30 자산 갱신 → 22:10 반환 → 22:17 재개(019) → 22:22:03 재시작 → 22:23 문서.

## 10. 다음 할 일 (사람이 직접, 계정·구매 불필요, 이미 확인된 기본 확인은 반복하지 않음)
1. 맥북에서 813031 나머지 13개 결과 시각을 3/16 부분 CSV(§3) 사본에 채우고, AB1~3 구간에 편집 컷·리플레이가 없는지 적는다 → §3 명령으로 가져오기.
2. 그 대응표로 세션을 만들어, 플레이어를 운영자 화면 '영상 기준 시각'에 맞춘 뒤 맥북(영상+운영자)과 아이폰(관람 주소)으로 두 기기 리허설: 자동 진행, 새로고침 뒤 같은 투구, 정지·재개, 지난 공 고정·'최신으로'.
3. 같은 기회에 아이폰 Safari로 현재 배치(013·017 이후)를 기본 글자와 큰 글자로 한 번씩 본다.
4. 대응표가 없으면 '시계만' 절차로 기능 리허설만 한다.

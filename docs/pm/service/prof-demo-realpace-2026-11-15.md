# 11/17 데모 새 구성 — 실제 투구 타이밍 재생 (조사·시험 기록)

상위: [데모 대본](prof-demo-script-2026-11-17.md) · [맥북 한 장](prof-demo-macbook-onepager-2026-11-16.md) · 작성 2026-11-15 16시대(Claude, 차선 3). 실행 기록: `~/Projects/pitcheezy-pm-jobs/20261115-demo/runRT-*`, 분석 스크립트 `~/Projects/pitcheezy-pm-jobs/20261115-realpace/`.

**새 구성(2026-11-15 오후 Song):**
- 맥북은 MLB 풀경기 다시보기 영상을 계속 튼다.
- 아이폰은 우리 화면을 **실제 투구 타이밍**으로 보여 준다. 실제 서비스처럼 기록이 경기 시계대로 도착한다고 가정한다.
- 아이폰은 Tailscale 없이 pitcheezy.com(팀원 호스트의 앱)으로 본다.
- 모니터 2 클립과 장면별 수동 맞춤은 주 방식에서 뺀다. 이전 장면 A·B(고정 간격 재생)는 예비로 남긴다.

## 1. 실제 속도 재생 — 있는가, 무엇을 바꿨나

- **10월 D134 "실제 속도 재생"은 코드에 있었다.** `scripts/demo_precompute.py`의 `TimedReplay`가 끝난 경기를 투구의 실제 `startTime` 간격대로 내준다. 하지만 이것은 예전 관전 서버의 `/api/live/{game}/state?replay=…`에서만 썼다(시청자 링크마다 시계가 따로 있음).
- **공개 발행기 경로(`run_scene.sh` → `observer_app.publisher`)는 이 기능을 쓰지 않았다.** 쓰던 것은 둘이다. `ReplayTransport`는 한 번 읽을 때마다 다음 장을 낸다. `--interval 10`과 poll 20을 같이 써서 한 공에 약 10초가 걸렸다. `RecordedPaceReplay`는 수집 영수증 시각을 쓰므로, 수집 폴더가 없는 849843에는 쓸 수 없다.
- **이번 변경(작게):** env `PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE=game`이면 발행기가 `TimedReplay`를 쓴다(`live_feed.GameClockReplay`가 일정은 재생 폴더에서, 경기 본문은 `TimedReplay`에서 낸다). 손잡이는 env 6개다: `…_GAME`, `…_START`(시작 공 번호, 0부터), `…_T0`(시계 기준 epoch), `…_LEAD_S`(T0에서 시작 공 투구까지 초), `…_LAG_S`(투구 뒤 기록이 늦게 보이는 초), `…_SPEED`. 바뀐 코드는 `main.py` 한 갈래와 `live_feed.py`의 작은 클래스 하나다. 시험 파일은 `tests/test_game_clock_replay.py`다(관련 시험 124개와 `tests/test_demo_precompute.py` 11개 통과).
- **실행:** `~/Projects/pitcheezy-pm-jobs/20261115-demo/run_realpace.sh <port> <시작 공> <lead 초> [lag 초=0] [화면 폴더=B 묶음] [발행 간격=10] [poll=2]`. 끄는 법은 기존 `stop_scene.sh <회차 폴더>`와 같다.
- **임의 공에서 시작:** 된다. 시작 공 번호를 주면 그 공의 투구 순간이 실행 뒤 lead초가 된다(2절 표의 번호). 시작 직후 첫 발행은 약 6초다(아래 측정).
- **건너뛰기·되감기:** 돌고 있는 재생의 시계는 옮길 수 없다. 끄고 새 시작 공으로 다시 띄운다(새 회차 폴더·새 store). 다시 띄우면 sequence가 1부터 다시 시작한다. 그래서 **열려 있던 화면은 새로고침해야 한다**: 화면은 sequence가 내려간 파일을 버린다(`web/src/service/api.ts:103-113`). 11/15 07시대 장애 전환 시험에서 본 "재시작 뒤 69.6초"도 같은 원인으로 보인다.
- **경기 바꾸기:** 쉽다. `GAME=<gamePk> run_realpace.sh …`로 띄운다. 필요한 것은 `DEMO-WS-2026/feeds/<gamePk>_final.json.gz` 하나다(지금 40경기 있음). 일정 파일과 발행 날짜(KST)는 첫 실행 때 그 피드에서 만든다. 추천은 발행기가 피드로 그 자리에서 계산하므로 경기별 사전 준비가 따로 없다. 다만 투수에 따라 "추천 없음"이 나온다. 장면 대본(투수·카드)은 경기마다 새로 써야 한다.

## 2. 849843 실제 시간표 (최종 피드 `startTime` 기준)

CHC @ SD, 2026-09-29(현지). 예정 시각은 02:00Z, 첫 공은 02:11:13Z이다.
- 투구 262개.
- 첫 공에서 마지막 공까지 9,624초(2:40:24).
- 공 간격: 전체 중앙값 22.1초, 90% 62.8초, 최대 250.9초(7초→7말).
- 반이닝 안 공 간격: 중앙값 21.7초, 90% 41.0초, 최대 178.5초(5말 Assad→Rolison 교체).
- 공수 교대(앞 반이닝 마지막 공 → 다음 첫 공): 중앙값 164초, 154~251초.
- 반이닝 안 투수 교체 3번: 137→138 Assad→Rolison 178초, 174→175 Rolison→Civale 166초, 212→213 Civale→Webb 162초.

**영상 대조 기준점: 반이닝 첫 공 17개.** 9회 말이 없어서 17개다. "경과"는 1회 초 첫 공(0번)에서 잰 시간이다. 영상에서 1회 초 첫 공의 위치를 V라 하면, 영상이 원래 경기 시계를 그대로 두었다는 가정에서 각 첫 공은 V+경과에 있어야 한다. 영상이 광고를 잘라 냈으면 어긋나고, 이 점은 **미확인**이다.

| 반이닝 | 첫 공 번호 | 경과 | 투수 → 첫 타자 |
|---|---|---|---|
| 1초 | 0 | 0:00:00 | King → Crow-Armstrong |
| 1말 | 11 | 0:06:02 | Boyd → Tatis Jr. |
| 2초 | 22 | 0:13:50 | King → Suzuki |
| 2말 | 35 | 0:21:39 | Boyd → Hays |
| 3초 | 45 | 0:27:42 | King → Kelly |
| 3말 | 60 | 0:36:01 | Boyd → Campusano |
| 4초 | 79 | 0:47:31 | King → Busch |
| 4말 | 92 | 0:54:41 | Boyd → Merrill |
| 5초 | 110 | 1:05:09 | King → Ramírez |
| 5말 | 120 | 1:11:21 | Assad → Campusano |
| 6초 | 146 | 1:28:32 | King → Crow-Armstrong |
| 6말 | 167 | 1:38:23 | Rolison → Bogaerts |
| 7초 | 178 | 1:46:59 | King → Happ |
| 7말 | 192 | 1:57:51 | Civale → Tatis Jr. |
| 8초 | 214 | 2:14:56 | Matsui → Swanson |
| 8말 | 225 | 2:21:54 | Webb → Hays |
| 9초 | 240 | 2:30:00 | Musgrove → Busch |

전체 공 목록(번호·반이닝·startTime·경과·투수·타자)은 `~/Projects/pitcheezy-pm-jobs/20261115-realpace/timeline-849843.tsv`에 있다. 투수 이름은 최종 피드 타석의 마지막 투수다.

## 3. 영상과 맞추는 절차 (초안, 시험 전)

**시계의 뜻.** `run_realpace.sh`를 실행한 순간이 T0다. 시작 공 S는 재생 시계에서 T0 + lead에 던져진다. 그 뒤 공들은 원래 간격대로 이어진다. 발행기는 공 S가 던져지고 약 1초 뒤(인플레이 공은 플레이가 끝난 시각) 그 결과와 다음 공 추천을 낸다. lag를 주면 그만큼 더 늦게 낸다.

**맞추는 법(초안):**
1. 영상을 공 S의 투구 장면 **10초 앞**에 두고 멈춘다. 예: 1회 초 첫 공이면 V−10초.
2. 맥미니에서 `run_realpace.sh 8876 S 20 …`을 실행한다(lead 20초). 시작 직후 추천 계산 시간을 둔 것이다. 첫 발행은 실행 뒤 약 6초다(16:05 측정).
3. 실행 뒤 **10 + B초**에 영상 재생을 누른다. B는 "영상이 기록보다 늦는 초"다.
   - B = 10이면 영상 투구와 재생 시계 투구가 같은 순간이 된다. 맥북에서 `sleep` 뒤 `say`로 신호를 낼 수 있다(아래 예).
4. 이후 영상과 화면은 원래 간격대로 같이 간다. 영상을 멈추거나 건너뛰면 어긋난다. 다시 맞추려면 1~3을 그 자리의 반이닝 첫 공으로 되풀이한다(2절 표). 화면 탭은 새로고침한다.

**손잡이(공통 지연 값):**
- **데모 손잡이는 B(재생을 누르는 때)와 lag다.** 화면을 영상보다 앞서게 하려면 B를 키운다(영상을 늦게 튼다). 뒤서게 하려면 lag를 키운다. 화면이 영상보다 몇 초 앞서는지는 "B − lag − 1 − (발행·화면 갱신 지연)"이다(계산, 실측 아님).
- 실제 서비스의 공통 지연은 발행기의 `fixed_delay_s`(`PITCHEEZY_PUBLISH_DELAY_S`, 월드시리즈 15초)다. 발행기가 파일을 만들기 전에 적용한다(`publisher.py` → `create_app(service_delay_s=…)` → `feed.snapshot(game, delay)`). 화면은 이 값을 안내 문구에만 쓴다.
- 데모에서도 이 값을 쓸 수는 있다(시험 안 함). 하지만 실제 속도 재생에서는 lag와 같은 효과이고, 둘을 같이 쓰면 헷갈리므로 **lag 하나만 쓰기를 권한다.**

**실제 서비스 재현(기록이 영상보다 먼저 옴):**
- 10월 실측: 피드는 투구 뒤 중앙값 19초에 도착했고 중계는 30초 이상 늦었다(D134·D144).
- 데모로 옮기면 `lag 19`, B는 30 + 10이다(위 3번 기준: 실행 뒤 40초에 영상 재생). 화면이 영상보다 약 10초 먼저 다음 공 추천을 보인다(계산).
- 발행 간격 10초를 그대로 두면 그 위에 0~10초가 더 흔들린다. 그러면 화면이 영상보다 늦는 공도 생긴다(1회차 측정: 파일이 0.3~10.1초 늦음). **간격 2초를 권한다**(5절).

맥북 신호 예(맥미니 명령이 바로 돌아온다는 가정, 시험 안 함):
```sh
ssh <사용자>@<맥미니 이름> '~/Projects/pitcheezy-pm-jobs/20261115-demo/run_realpace.sh 8876 0 20 19 "" 2'; sleep 38; say 재생
```
`sleep` 값은 "10 + B − ssh 왕복"이다. B = 40이면 약 38초다(ssh 왕복 2초는 가정).

## 4. 실제 속도 5~10분 시험 (2026-11-15, 127.0.0.1, B 묶음, 브라우저 없음)

브라우저 잠금은 차선 1(`front-3d-1115`)이 잡고 있었다. 그래서 **발행 파일만 쟀다.** 화면(브라우저) 지연은 미측정이다. 비교 기준은 TimedReplay가 공 n−1을 "본" 시각이다(투구 + 1초, 인플레이는 플레이 끝). 그 시각에 발행 파일의 공 수가 n이 되어야 한다.

**1회차 `runRT-155411`:** 시작 공 0(1회 초 첫 공), lead 40, lag 0, 발행 간격 10, poll 2, 10분 40초.
- 첫 발행: +5.7초. 첫 추천(1구 전): +10.9초.
- 공 17개(1회 초 전체와 1회 말 6구)가 모두 순서대로 나왔다. 건너뛴 공은 0개다.
- 늦음은 공마다 +0.3~+10.1초, 중앙값 +4.1초다.
- **누적 오차는 없었다.** 처음 공과 마지막 공의 늦음 차이는 −8.6초인데, 569초 동안 쌓인 것이 아니라 발행 간격 10초 안에서 위상이 흔들린 것이다.
- 공수 교대(11→12번 공, 경기 시계 116초)도 원래 간격대로 지나갔다.
- 발행기 RSS 최대 1.85 GB. 로그 오류 0.

**2회차 `runRT-160503`:** 시작 공 120(5회 말 첫 공, 경기 중간에서 시작), lead 20, lag 0, 발행 간격 2, poll 1, 5분 30초.
- 첫 발행: +7.6초. 첫 추천: +8.1초. 시작 공 120 투구는 +20초다.
- 공 12개(120~131번 결과)가 모두 순서대로 나왔다. 건너뛴 공은 0개다.
- 늦음은 공마다 +0.3~+2.4초, 중앙값 +1.3초다. 누적 오차는 없었다(처음과 마지막 차이 −1.5초, 경기 시계 315초 동안).
- 발행기 RSS 최대 1.86 GB. 로그 오류 0.

**경기 바꾸기 확인 `runRT-161056`:** `GAME=849803`(4차전), 시작 공 0, 간격 2, 70초. 일정 파일(`schedule_2026-10-27.json`)과 발행 날짜(KST 2026-10-28)가 자동으로 만들어졌다. 70초 안에 공 3개와 다음 공 추천(`ready`)이 발행됐다. 로그 오류 0.

**정리:** 실제 속도로 돌려도 발행 시각이 원래 투구 간격에서 벗어나 쌓이지 않는다. 공마다의 흔들림은 발행 간격 이하다(간격 10초에서 최대 10.1초, 2초에서 최대 2.4초). 데모에는 **간격 2초를 권한다.**

## 5. 화면 갱신 간격 — 발행 간격을 따른다

공개 화면은 진행 중 경기를 **manifest의 `publish_interval_s`마다** 읽는다(`web/src/service/GameScreen.tsx:332`, `api.ts:163`). 없을 때만 10초(`livePollMs`)를 쓴다. 그래서 11/15 07시대의 "화면이 0~10초 늦다"는 발행 간격 10초 때문이었다.
- 발행 간격을 2초로 하면 화면도 2초마다 읽는다(코드 기준). 이것은 맥미니 로컬에서 브라우저로 재지 않았고 **미측정**이다.
- 호스트로 보낼 때는 폰마다 2초에 한 번씩 요청이 간다. 호스트 요청 한도는 **모름**(6절).

## 6. 아이폰 경로 — pitcheezy.com(팀원 호스트)으로 (조사만, 실행·업로드 없음)

**월드시리즈 때 실제로는 올라가지 않았다.**
- 발행기의 업로드는 꺼져 있었고, 공개 시청자는 0명이었다(D181, `docs/reviews/external-evaluation-2026-10-24.md`).
- 실제 호스트에 대한 시험은 10/26 합성 공동시험 하나다. HEAD·PUT·GET·DELETE는 됐지만 마지막 HEAD가 `upload_unknown`으로 끝나 판정은 `FAIL_OR_STOPPED`였다(`~/Projects/pitcheezy-pm-jobs/20261026-joint-test/PUBLIC-SUMMARY.md`).
- 지금 호스트의 849843(c47)은 업로드로 간 것이 아니다. Song이 ZIP을 전달했고 팀원이 직접 올렸다.

**설계된 경로(코드로 확인):**
- 맥미니가 **밀어 올린다**(호스트가 가져가지 않음).
- 발행기 안의 백그라운드 업로드 일꾼이 파일마다 이렇게 보낸다: 인증된 HEAD로 원격 ETag를 받고, `If-Match`(없으면 `If-None-Match: *`)와 `X-Content-SHA256`, Bearer 토큰을 붙여 PUT하고, 200 영수증을 검사한다(`publisher.py` HttpSink, `upload_worker.py`).
- 주기는 발행 간격이고, 진행 중 경기는 바뀔 때만 다시 쓴다.
- 설정: `publish_config.json`의 `upload.enabled`(기본 false), `upload.base_url`(HTTPS, `/_publish` 경로 포함), 토큰은 env `PITCHEEZY_PUBLISH_TOKEN` 또는 저장소 밖 0600 `token_file`.
- 운영 실행기는 `ws_operate start --upload`이고, 주소·토큰은 `PITCHEEZY_WS_OPS_UPLOAD_BASE_URL`·`PITCHEEZY_WS_OPS_TOKEN_FILE`로 준다(값은 저장소에 없다).

**849843 재생을 이 경로로 보내려면:**
- 우리 쪽:
  - 업로드 켜기, `base_url`과 토큰.
  - `--date 2026-09-30`(이미 스크립트에 있음).
  - 회차마다 새 out·state.
  - 연결 점검: 시작할 때 manifest에 HEAD를 한 번 보낸다.
- 팀원 쪽:
  - 업로드 받기 켜기(꺼져 있으면 503).
  - 발행 토큰 발급. 10/26 토큰은 `v1/game/<key>.json` 한 파일 범위였다. 발행기는 manifest·health·dates·games/<날짜>·game·setup을 모두 올리므로 그 범위로는 403 영구 거절이 난다. 그래서 범위 확대(또는 범위 없음)와 재생 시간 전체를 덮는 만료가 필요하다.

**호스트의 "끝난 경기" 849843과 겹침:**
- 412 충돌은 거의 나지 않는다. 발행기가 먼저 HEAD로 원격 ETag를 받아 그것으로 덮어쓰기 때문이다(코드로 확인).
- 대신 **모든 사람이 보는 849843 최종 화면을 진행 중 화면으로 덮어쓴다.** 경로가 `v1/game/849843.json` 하나뿐이고 manifest·dates·games 파일도 바뀐다. 끝난 뒤 c47로 되돌리는 절차가 필요하다.
- "끝난 경기는 다시 쓰지 않는다"는 같은 발행기 state 안에서만 지켜진다.
- 다른 경기 번호나 접두사로 올리는 방법은 **지금 없다.** 업로드 허용 목록이 `v1/` 고정이고, 화면도 `VITE_PUBLISH_BASE/v1/…` 고정이다.
  - 대안(추정): 팀원이 별도 공개 위치를 만들고 우리가 `VITE_PUBLISH_BASE`를 바꿔 다시 빌드한다.
- **sequence 문제(확인):** 새 회차는 sequence를 1부터 센다. 호스트의 c47 파일보다 낮으면, c47을 본 적 있는 열린 폰은 새로고침 전까지 재생 파일을 버린다. 되감기(다시 띄우기)마다 같은 일이 생긴다.

**접근 코드 켠 상태에서 폰이 따라오는가:**
- 확인된 것:
  - 11/12에 코드 없이 열면 앱과 v1 자료 모두 401(`access_required`)이었다. 즉 자료 읽기에도 문이 걸려 있다.
  - 화면은 401을 받으면 "새로고침해 다시 접속" 문구를 띄운다.
  - 폴링은 오류가 나도 멈추지 않는다. 탭이 숨겨지면 건너뛴다.
- 추정: 접근은 쿠키 세션이다. 같은 출처라 폴링에도 쿠키가 실린다.
- 모름: 쿠키 수명·만료, 실제 폰(Safari·셀룰러)에서 장시간 따라오는지.

**팀원에게 요청할 항목:**
- [ ] 대상 호스트 확정(전시 호스트와 10/26 시험 호스트 중 어디인지). c47 파일이 업로드 저장소 객체인지 정적 자산인지.
- [ ] 업로드 받기를 켜는 시간대와 끄는 방법.
- [ ] 발행 토큰: 6개 경로(manifest, health, dates, games/2026-09-30, game/849843, setup/849843) 범위 또는 범위 없음, 재생 시간을 덮는 만료, 끝난 뒤 회수. 토큰은 공개 이슈에 쓰지 않는다.
- [ ] 진행 중 본문으로 c47 최종 파일을 덮어써도 되는지, 끝난 뒤 c47로 되돌리는 방법(ZIP 다시 올리기 또는 우리가 원본 바이트를 PUT).
- [ ] 또는 별도 공개 위치를 줄 수 있는지(그 경우 우리가 화면을 다시 빌드).
- [ ] 호스트가 `Cache-Control: no-store`를 그대로 쓰는지, 가장자리 캐시와 ETag/304 동작.
- [ ] 접근 코드 쿠키의 수명·만료 조건, v1 자료 읽기도 같은 쿠키로 통과하는지, 3시간 동안 2~10초 폴링으로 만료되지 않는지.
- [ ] 요청 한도(429). 폰 몇 대 × 2초 폴링을 감당하는지.
- [ ] 같은 토큰으로 쓰는 사람이 우리 하나뿐인지(단일 작성자).
- [ ] 실제 폰(아이폰 Safari)에서 접근 코드를 켠 채 재생을 따라오는지 함께 확인.

출처: 이번 조사(읽기 전용, 하위 작업자 보고를 일부 코드로 대조함). sequence 버림(`api.ts:103-113`)과 manifest 간격 폴링(`GameScreen.tsx:332`)은 직접 확인했다. 나머지 파일·줄 번호는 `publisher.py`, `upload_worker.py`, `publish_config.json`, `ops/ws_ops_config.json`, `docs/pm/service/ws-operations-runbook.md`, `docs/pm/transfers/COLLAB-001-reply6-draft-2026-11-12.md`(메인 체크아웃, 미추적)에 있다.

## 7. 못 한 것·미측정

- 화면(브라우저·아이폰)에서 본 실제 속도 재생 지연: 브라우저 잠금이 다른 차선에 있어서 미측정.
- 영상과 실제로 나란히 맞추기, 영상의 경기 시계 보존 여부(광고 편집): 영상이 없어서 미확인.
- 호스트 업로드, 접근 코드 쿠키, 폰 장시간 폴링: Song 결정 범위라 실행 안 함. 미측정.
- `fixed_delay_s`를 실제 속도 재생과 함께 쓰는 것: 시험 안 함.
- ssh로 띄울 때 명령이 바로 돌아오는지: 시험 안 함.

## 부록. `run_realpace.sh` 본문 (`~/Projects/pitcheezy-pm-jobs/20261115-demo/`, 11/15 16시대)

분석 스크립트: `20261115-realpace/timeline.py`(최종 피드 → 투구 시간표), `drift.py`(회차 폴더의 `watch_rt.tsv`와 시간표 비교). 발행 파일 감시는 `20261115-demo/watch_rt.py`.

```sh
#!/bin/zsh
# usage: run_realpace.sh <port> <start-pitch> <lead-s> [lag-s=0] [screen-dir] [publish-interval=10] [poll-s=2]
# 849843 at its REAL pitch times (PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE=game, demo only). Pitch <start> is released
# <lead-s> after the moment this script runs (T0, recorded in t0.txt); every later pitch follows at the game's own spacing.
# lag-s: each pitch becomes visible to the publisher that long after its release (feed lag rehearsal).
# Another game: GAME=<gamePk> run_realpace.sh ... (needs DEMO-WS-2026/feeds/<gamePk>_final.json.gz; the schedule file
# and the KST publish date are made from that feed on first use, in sched-<gamePk>/).
set -eu
PORT=$1; START=$2; LEAD=$3; LAG=${4:-0}
D=$HOME/Projects/pitcheezy-pm-jobs/20261115-demo
WT=$HOME/Projects/pitcheezy-worktrees/prof-demo-1117
J=$HOME/Projects/pitcheezy-pm-jobs/exhibition-849843-1031
if [[ -e $HOME/Projects/pitcheezy-pm-jobs/20261115-night/TRAINING ]]; then echo "TRAINING 표시 있음 — 띄우지 않음"; exit 2; fi
GAME=${GAME:-849843}; SCHED=$D/sched-$GAME
FINAL="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/feeds/${GAME}_final.json.gz"
[[ -f $FINAL ]] || { echo "최종 피드 없음: $FINAL"; exit 4; }
if [[ ! -f $SCHED/kst_date.txt ]]; then mkdir -p $SCHED; python3 -I - "$FINAL" "$SCHED" <<'PY'
import gzip, json, sys
from datetime import datetime, timedelta
f = json.load(gzip.open(sys.argv[1])); g = f['gameData']; d = g['datetime']
s = {'dates': [{'games': [{'gamePk': g['game']['pk'], 'gameType': g['game']['type'], 'gameDate': d.get('dateTime'),
     'status': {'abstractGameState': 'Live', 'detailedState': 'In Progress (rehearsal)'},
     'teams': {k: {'team': {'name': g['teams'][k]['name']}} for k in ('away', 'home')}}]}]}
json.dump(s, open(f"{sys.argv[2]}/schedule_{d['officialDate']}.json", 'w'))
kst = (datetime.fromisoformat(d['dateTime'].replace('Z', '+00:00')) + timedelta(hours=9)).date().isoformat()
open(f'{sys.argv[2]}/kst_date.txt', 'w').write(kst + '\n')
PY
fi
PDATE=$(cat $SCHED/kst_date.txt)
SCREEN=${5:-$D/bundles-0738/B-color-only/site}; INTERVAL=${6:-10}; POLL=${7:-2}
[[ -f $SCREEN/index.html ]] || { echo "화면 폴더에 index.html 없음: $SCREEN"; exit 3; }
RUN=$D/runRT-$(date +%H%M%S); mkdir -p $RUN/out
rsync -a --exclude '/v1' $SCREEN/ $RUN/out/
echo "$SCREEN" > $RUN/screen.txt
T0=$(python3 -c 'import time;print(round(time.time(),3))')
print "T0 $T0 game $GAME start $START lead $LEAD lag $LAG interval $INTERVAL poll $POLL" > $RUN/t0.txt; echo $T0 >> $RUN/t0.txt
cd $WT
env -i HOME=$HOME USER=$USER LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8 TMPDIR=$TMPDIR PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=apps/observer/backend PYTHONUNBUFFERED=1 \
  PITCHEEZY_OBSERVER_LIVE_STORE=$RUN/store.sqlite3 PITCHEEZY_OBSERVER_LIVE_ARMB=1 PITCHEEZY_OBSERVER_LIVE_DELAY_S=0 \
  PITCHEEZY_OBSERVER_REC_LEDGER=1 PITCHEEZY_OBSERVER_LEGACY=0 PITCHEEZY_OBSERVER_VIEWER_DELAY=0 PITCHEEZY_OBSERVER_STEP5_EXPECTATION=0 \
  PITCHEEZY_FEED_DIR=$D/feeds-empty \
  PITCHEEZY_WATCH_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/watch" \
  PITCHEEZY_INTENT_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/intent" \
  PITCHEEZY_DEMO_DIR=$D/empty/demo PITCHEEZY_RUNS_DIR=$HOME/Projects/pitcheezy/runs PITCHEEZY_DATA_DIR=$HOME/Projects/pitcheezy/data \
  PITCHEEZY_PLAYER_SNAPSHOT_DIR=$HOME/Projects/pitcheezy-pm-jobs/20261020-ws-pregame-cache-001/cache \
  PITCHEEZY_OBSERVER_PROFILE_SOURCES=$J/profiles/LATEST.json \
  PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR=$SCHED PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE=game \
  PITCHEEZY_OBSERVER_LIVE_REPLAY_GAME=$GAME PITCHEEZY_OBSERVER_LIVE_REPLAY_START=$START PITCHEEZY_OBSERVER_LIVE_REPLAY_T0=$T0 \
  PITCHEEZY_OBSERVER_LIVE_REPLAY_LEAD_S=$LEAD PITCHEEZY_OBSERVER_LIVE_REPLAY_LAG_S=$LAG PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S=$POLL \
  $HOME/Projects/pitcheezy/.venv-observer/bin/python -B -m observer_app.publisher --out $RUN/out --interval $INTERVAL --date $PDATE \
  --config apps/observer/publish_config.json > $RUN/publisher.log 2>&1 &
echo $! > $RUN/publisher.pid
python3 -m http.server $PORT --bind 127.0.0.1 --directory $RUN/out > $RUN/http.log 2>&1 &
echo $! > $RUN/http.pid
echo $RUN
```

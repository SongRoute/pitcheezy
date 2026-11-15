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

## 4.2 화면에서 잰 실제 속도 재생 — 묶음 D + 마네킹 (2026-11-15 17:51~18:05, `runRT-175147`)

조건: 맥미니 127.0.0.1:8876, 849843 시작 공 0, lead 90, lag 0, **발행 간격 2초**, poll 2. 화면은 묶음 D(`D-color-3d-fit-indexed.zip` sha256 `e874207b…`, 대조함)를 서빙 사본(`20261115-demo/serveD/site`)으로 풀고, 마네킹 모델 두 파일(`batter-indexed.glb` `4b744d1c…`, `.meta.json` `e55fc807…`, 대조함)을 **서빙 사본의 `pz/batter3d/assets/`에만** 넣었다. 저장소·묶음에는 넣지 않았다. 헤드리스 Chromium(Playwright, 모바일 모드, 배율 3), 네 크기를 동시에 **13분(780초)** 열어 두었다. 범위는 1회초 첫 공 ~ 2회초 첫 타자(타자 9명, 공수 교대 2번). 스크립트 `20261115-demo/measureD.mjs`, `halfD.mjs`.

**파일 → 화면 반영 지연** (발행 파일이 바뀐 시각(0.25초 감시) → 화면 글자가 바뀐 시각, 파일 변화 23번 중 맞춘 수):

| 화면 | 맞춤 | 중앙값 | 90% | 최대 |
|---|---|---|---|---|
| 393×659 | 22/23 | 0.72초 | 1.39초 | 1.91초 |
| 402×680 | 21/23 | 0.73초 | 1.30초 | 1.43초 |
| 393×852 | 22/23 | 0.84초 | 1.30초 | 1.43초 |
| 402×874 | 22/23 | 0.87초 | 1.56초 | 1.65초 |

- 화면 감시 주기는 약 0.25초+네 화면 검사 시간이다. 그보다 짧은 차이는 못 가른다. 맞추지 못한 1~2번은 글자가 바뀌지 않는 파일 변화로 본다(추정).
- 따라서 화면 쪽 지연은 **발행 간격(2초) 안**이다. 공 하나의 영상 대비 총 지연 = 4절 발행 지연(0.3~2.4초) + 이 값. 둘을 같은 공에서 더해 재지는 않았다.
- 명령 → 첫 추천: 이번에는 화면을 명령 31초 뒤에 열어서 **미측정**(발행 파일 기준 첫 추천은 명령 8.0초 뒤).

**10분 넘게 둔 동안** (30초마다 표본, 화면별 25~26개):
- `PZFit.status().runs`: 6 → 97~99로 늘었다. 맞춤 값(`applied`)은 화면마다 내내 같았다(393×659 209, 나머지 216; 680·659는 "여백 줄임 + 맞춤", 852·874는 "그대로").
- JS 힙: 시작 19.1~21.4 MB, 끝 12.5~13.0 MB. 늘어나는 흐름 없음.
- 세로·가로 넘침: 네 화면 모두 표본 전부 0px. 3D 상자 아래끝은 늘 보이는 영역 안.
- 콘솔: 페이지 오류 0, 콘솔 오류 0, 실패 요청 0. 경고는 화면마다 1번(WebGL "GPU stall due to ReadPixels", 헤드리스 GPU 성능 경고).
- 3D 연결(`attached`)은 내내 true.

**타자 교체·공수 교대 때 3D와 배치:** 화면 글자의 좌타/우타, 존 줄의 `bat-l/bat-r`, 3D 쪽(`PZSongSkin.status().side`), 3D 상자의 `side-r` 네 값이 **바뀐 기록 전부에서 일치**했다(어긋남 0줄, 네 화면). 타자 9명(좌·우·좌·우·우·우·우·좌·우), 1회초→1회말→2회초. 감시 간격(~0.25초)보다 짧은 순간 어긋남은 못 잡는다.

**맥북 반쪽 창(예비안), 2회초 Suzuki 첫 공 장면:**
- 960×1080: 넘침 0/0, 맞춤은 "넓은 화면·가로 화면"으로 적용 안 함. 3D 타자는 존에서 멀리 오른쪽 끝에 붙고, 3D 상자가 존 줄 오른쪽 끝을 22px 넘는다(창 안). 아래 약 40%가 빈 공간. 깨지지는 않음.
- 720×900: 넘침 0/0, 맞춤 "그대로"(216). 존과 3D 타자가 한 화면에 들어오고 하단 메뉴까지 다 보인다.
- 두 창 모두 오류 0. 예비안 창은 **720×900 쪽이 보기 좋다**(눈으로 본 판단).

**대표 캡처**(`~/Projects/pitcheezy-pm-jobs/20261115-demo/runRT-175147/capD/`, 전체 26장):
- `402x680-02-batter1.png` — 좌타→우타 교체 직후(Bregman), 3D 우타 쪽
- `393x659-03-half-change.png` — 1회말 첫 타자(Tatis), 공수 교대 직후
- `half-960x1080.png`, `half-720x900.png` — 예비안 반쪽 창

**ssh localhost:** 다시 시도했다(11/15 17:52). 이 맥의 키가 자기 `authorized_keys`에 없어 거부된다. 키 등록은 보안 설정 변경이라 하지 않았다. **미측정**(파이프 대리 시험의 추정은 7절 그대로).

## 4.3 발행 간격 5초에서 화면 — c47 화면과 묶음 D (2026-11-15 20:32~20:41)

조건: 849843 시작 공 0, lead 30, lag 0, **간격 5초**, poll 2. 두 회차를 동시에 띄웠다. c47은 11/1 전달 묶음 `bundle-849843-main-c47fe6cb-20261101`의 `site/`(zip sha256 `3c3b8006…8da51`), 8876, `runRT-203241`. D는 4.2절과 같은 서빙 사본, 8875, `runRT-203242`. 헤드리스 Chromium 한 개에 393×659 페이지 두 개를 8분(480초) 열었다. 스크립트 `measure5.mjs`. 처음 띄운 `runRT-203221-collided`는 두 회차가 같은 폴더를 써서 버렸다. 그래서 스크립트 폴더 이름에 포트를 붙였다.

| 간격 | 화면 | 파일→화면 중앙값 | 90% | 최대 | 오류 | 4xx·실패 요청 | "갱신 지연" |
|---|---|---|---|---|---|---|---|
| 2초 | D (4.2절, 4개 크기) | 0.72~0.87초 | 1.30~1.56초 | 1.43~1.91초 | 0 | 0 | 기록 안 함 |
| 5초 | c47 | 4.05초 | 4.33초 | 4.67초 | 0 | 0 (응답 140개) | 0번 |
| 5초 | D | 2.76초 | 3.12초 | 3.17초 | 0 | 0 (응답 152개) | 0번 |
| 10초 | D | 미측정 | | | | | |

- 파일 변화 17번 중 14번을 맞췄다. 나머지는 페이지를 닫은 뒤(480초 이후)의 변화다.
- c47 화면도 진행 중 재생을 읽고 갱신한다(1회말 Machado까지, 캡처 `~/Projects/pitcheezy-pm-jobs/20261115-demo/runRT-203241/end5-c47-393x659.png`). 3D 대신 실루엣이고, 393×659에서는 "이번 타석" 줄이 하단 메뉴에 가린다(c47의 알려진 높이 문제).
- 10초 간격의 화면 지연은 이번에 재지 않았다. 11/15 07시대 B 묶음에서 "0~10초"로 본 기록만 있다(5절).
- 출처 로그: `~/Projects/pitcheezy-pm-jobs/20261115-demo/runRT-203241/dom5-c47.tsv`·`sum5-c47.json`·`watch_rt.tsv`, `~/Projects/pitcheezy-pm-jobs/20261115-demo/runRT-203242/dom5-D.tsv`·`sum5-D.json`·`watch_rt.tsv`.

## 4.4 업로드 요청 수 — 계산과 로컬 가짜 수신기 실측 (2026-11-15 20:45~20:52)

**계산(코드):** 진행 중 경기 하나, `--date` 하나일 때 한 주기마다 `games/<날짜>`·`dates`·`manifest`·`health`는 항상 쓴다(`publisher.py:1134, 1136-1140, 1198, 1259`). 각 파일에 순번·시각이 들어가 매번 바이트가 다르다(`publisher.py:1019`). `game/849843`은 내용이 바뀔 때만 쓴다(`publisher.py:1436-1439`). `setup/849843`은 끝난 경기만 쓴다(`publisher.py:1450, 1479`). 파일마다 HEAD 1번 + PUT 1번이다. ETag를 기억하지 않아 매번 HEAD를 보낸다(`publisher.py:618, 652-657`). 그래서 한 주기에 8~10회다.

**실측:** `fake_receiver.py`(테스트의 CAS 모델 `Endpoint` 재사용, 127.0.0.1:8870~8872) + 간격별 회차 3개를 동시에 6분 돌렸다. 처음 1분을 뺀 5.5분 평균이다.

| 간격 | 계산(분당) | 실측(분당) | 1분 칸별 | 호스트 한도 60 |
|---|---|---|---|---|
| 10초 | 48~60 | 49.4 (경기 파일 빠짐, 아래) | 48~49 | 한도 안(경기 파일 포함 시 약 55~56, 계산) |
| 5초 | 96~120 | 103.6 | 96~110 | **넘음(약 1.7배)** |
| 2초 | 240~300 | 246.5 | 236~252 | **넘음(약 4배)** |

- 경로별(5초): manifest·health·dates·games 각 HEAD 12.1 + PUT 12.1회/분, game/849843 HEAD 3.2 + PUT 3.2회/분(1회초~1회말 구간).
- 한도를 넘으면 호스트가 429를 줄 것으로 본다(호스트 동작은 미측정). 우리 발행기는 429를 `retry`로 두고 Retry-After(없으면 60초) 동안 그 경로를 보내지 않는다(`publisher.py:402-405, 582, 661-662`). 그 동안 폰의 그 파일은 멈춘다.
- 출처: `~/Projects/pitcheezy-pm-jobs/20261115-demo/fakerx-8870.tsv`(5초), `fakerx-8871.tsv`(2초), `fakerx-8872.tsv`(10초), 회차 `runRT-204516-8873/-8874/-8875`, 설정 `pubconf-fake-887*.json`(가짜 토큰 0600 파일, 값은 합성).

**같이 찾은 것 두 가지:**
1. `run_realpace.sh`가 화면 파일을 `out/`에 미리 복사하던 탓에, 업로드를 켜도 **한 번도 올라가지 않았다**(`upload_recovery_blocked`, `publisher.py:929-930`. 상태 파일 없이 `out/`에 파일이 있으면 막는다). 첫 시도 회차는 `*-recoveryblocked`로 남겼다. 고침: 화면은 `site/`에 두고 `site/v1 → out/v1` 링크로 보여 준다. `out/`은 빈 채로 시작한다.
2. 10초 회차에서 시작 직후 `game/849843.json`을 PUT(200)한 뒤 작업자 쪽 예외(`upload_failed`, `upload_worker.py:93`)가 나서 그 경로가 `inflight`로 남았다. 이후 6분 동안 경기 파일을 다시 보내지 않았다(`_eligible`이 `inflight`를 제외, `publisher.py:579-580`). → 원인을 찾아 고쳤다(4.5절).

**`blocked`가 생기면:** 그 경로만 자동 재전송을 멈추고(`publisher.py:580`), 다른 파일의 로컬 발행·업로드는 계속된다(`publisher.py:1141-1142, 1484-1497`). health의 `upload.pending`과 `code: upload_failed`로 드러난다(`publisher.py:1195-1197`). 풀어 주는 명령은 없다(`ws-operations-runbook.md:139-147`).

**`--check-remote`:** 인증 HEAD `v1/manifest.json` 1번, 쓰기 없음(`publisher.py:1560-1566`). 가짜 수신기로 실행해 HEAD 1번(404 → 통과)을 확인했다(`~/Projects/pitcheezy-pm-jobs/20261115-demo/fakerx-check.tsv`).

## 4.5 경기 파일 업로드 멈춤 — 원인과 수정 (2026-11-15 20:56~21:17)

**원인 두 겹:**
1. **응답을 놓치는 경합** — `upload_worker.py:146-153`(수정 전 152-153). 부모는 자식(HTTP 한 번)의 응답을 0.05초씩 `poll`하고, 없으면 `is_alive()`를 본다. 자식이 그 사이에 응답을 쓰고 끝나면 파이프에 응답이 있는데도 `break` → `upload_unknown`(blocked)이 된다. 시작 직후 본 스레드가 바빠(GIL) 이 틈이 넓어질 때 잘 난다. 확인: 자식 종료 코드를 찍어 보니 실패한 자식은 모두 **종료 코드 0**(정상 종료, 응답 보냄)이었다(`trace_publisher.py`의 `_reap` 기록).
2. **그 오류를 못 잡음** — `python -m observer_app.publisher`(우리 스크립트와 `ws_operate.py:70` 모두)로 띄우면 publisher.py는 `__main__`으로 돈다. 그런데 `upload_worker.py:22, 57, 127`의 `from .publisher import UploadError`는 **두 번째 사본**을 불러온다. 그래서 작업자가 던진 `UploadError`가 `publisher.py:619`·`659`의 `except UploadError`에 걸리지 않는다. 이미 `inflight`로 저장된 시도(`publisher.py:648`)가 그대로 남고, `_eligible`이 `inflight`를 영영 제외한다(`publisher.py:579-580`). 429·412 같은 정상 오류도 같은 길로 새서 `retry`가 되지 않았다.

**수정(최소):**
- `upload_worker.py`: 자식이 죽은 것을 본 뒤 `receive.poll(0)`으로 한 번 더 보고 응답이 있으면 읽는다(2줄).
- `publisher.py` 끝: `__main__`으로 돌 때 `sys.modules['observer_app.publisher']`를 자기 자신으로 둔다(`setdefault`, 2줄 + 주석).
- 테스트 2개: `tests/test_upload_worker_race.py`(늦게 읽힌 응답) — 수정 전 실패·후 통과. `tests/test_publisher_main_module.py`(`-m` 실행에서 작업자가 같은 `UploadError`를 봄) — 수정 전 실패·후 통과.

**재현 (가짜 수신기, 회차당 170초, 3개 동시, `repro_batch.sh`):**

| | 주입 없음 | 첫 경기 PUT에 429 주입 | 계 |
|---|---|---|---|
| 수정 전 (`pre-A`, `pre-B`) | 10초 1·5초 1 → **멈춤 2/2** | 10초 2·5초 2 → **멈춤 4/4** | **6/6** |
| 수정 후 (`post-A`, `post-B`) | 10초 1·5초 1 → 멈춤 0/2 (경기 PUT 16·19번) | 10초 2·5초 2 → 멈춤 0/4 (429 뒤 70초에 재전송, 경기 PUT 11~13번) | **0/6** |

- 수정 전은 주입 없이도 매번 멈췄다(시작 직후 경합). 앞서 4.4절 5초·2초 회차가 멈추지 않은 것은 그 회차에서 경합이 경기 파일에 걸리지 않았기 때문으로 본다(추정).
- 수정 후 남은 `upload_failed` 1번은 주입한 429다(`retry` → 60초 뒤 재전송, 의도대로).
- 디버그 래퍼 `trace_publisher.py`(작업 폴더, 저장소 밖)는 `-m`과 같은 방식으로 돌리며 traceback과 자식 종료 코드만 더 찍는다. 처음 판은 `__main__` 보호가 없어 자식이 발행기를 하나 더 띄웠다 → 그 3개는 내가 멈췄고 래퍼를 고쳤다(`repro/b1`은 버림).
- 로그: `~/Projects/pitcheezy-pm-jobs/20261115-demo/repro/{pre,post}-{A,B}.txt`(회차 목록), `repro/*-rx-887*.tsv`(수신 기록), 각 회차 `publisher.log`·`out.state.json`. 요약 스크립트 `repro_summary.py`.

**테스트:** publisher 테스트 전체(`tests/test_publisher*.py`, `test_publish_reference.py`, 새 2개) **344 passed**. 백엔드 전체 1176 passed / 3 failed / 1 xfailed. 3개 중 2개는 전부터 있던 10/19 화면 핀 불일치(`test_public_viewer_e2e`, `test_public_viewer_gateway_e2e`)이다. 1개(`test_publisher_background.py::test_blocked_prefix_cannot_starve_later_eligible_path`, `141 == 140`)는 시간에 따라 흔들린다. 단독 10회 실행에서 수정 전 2/10, 수정 후 2/10 실패로 같아서, 이번 수정과 무관한 기존 흔들림으로 본다.

**token_file 방식 끝까지 (가짜 수신기, 임시 가짜 토큰 0600 파일, 디버그 래퍼 없이 `-m`):** check-remote 통과(HEAD 1번) → 발행 100초(간격 5초) 요청 191번(PUT 95·HEAD 96, 오류 0, 경기 PUT 11번) → `stop_scene.sh`로 멈춤. 멈춘 뒤 `upload_attempts` 비어 있음, 대기 0, 프로세스·포트 0. **통과.** 로그 `~/Projects/pitcheezy-pm-jobs/20261115-demo/repro/e2e-token/`, 회차 `runRT-211521-8873`. 실제 토큰 파일과 실제 호스트는 쓰지 않았다.

## 4.6 PUT 실패는 영구 `blocked` — 확인과 푸는 법 (2026-11-15 21:35~21:38, 독립 검토 P1 후속)

코드: PUT/DELETE의 412·5xx·예상 밖 2xx·영수증 이상은 `blocked`(`publisher.py:400-401, 408, 425, 430`), 소켓 타임아웃(`timeout_s` 5초)·연결 끊김은 `upload_unknown` → `blocked`(`publisher.py:429-430`), 작업자 마감 15초도 `blocked`(`upload_worker.py:154`). `blocked`는 다시 보내지 않는다(`publisher.py:580`), 재시작해도 남은 `inflight`는 `blocked`로 바뀐다(`publisher.py:490-491`). HEAD 실패는 상태를 바꾸지 않고 다음 주기에 다시 보낸다(`publisher.py:619-621`).

가짜 수신기 재현(각 1회, 간격 5초, `~/Projects/pitcheezy-pm-jobs/20261115-demo/repro/p1/`):

| 장면 | 주입 | 결과 |
|---|---|---|
| S1 업로드 꺼짐 흉내 | HEAD 503 40번 + PUT 503 40번 | HEAD 503 동안은 PUT을 안 보내 막히지 않음. HEAD가 통과하기 시작한 뒤 PUT 503 → **5개 경로 전부 `blocked`**(health 포함) |
| S2 | 경기 파일 PUT 503 1번 | 경기 파일 `blocked`, 나머지 정상 |
| S3 | 경기 파일 PUT 8초 지연(서버는 저장함) | 클라이언트 5초 타임아웃 → 경기 파일 `blocked`(서버에는 이미 저장됨) |

푸는 법 확인: 같은 수신기에 **새 회차(새 `out/`·상태 파일)** 로 다시 띄우면 S2·S3 모두 경기 파일이 바로 다시 올라갔다(HEAD로 현재 ETag를 받아 `If-Match` PUT, 각 16회 정상). S1은 수신기가 아직 PUT 503을 주는 상태라 새 회차도 다시 `blocked` — 호스트가 꺼진 채면 재시작해도 소용없다.

**503·타임아웃을 429처럼 재시도로 바꾸는 수정 — 의견: 내일 전에는 바꾸지 않는다.** 타임아웃·5xx 뒤에는 서버가 이미 저장했는지 모른다(S3가 그 예). 같은 `expected` ETag로 다시 PUT하면 저장된 경우 412(→ 어차피 `blocked`), `If-None-Match: *`였다면 역시 412다. 안전한 재시도는 "HEAD로 현재 SHA를 보고, 같으면 성공 처리·다르면 새 ETag로 PUT"하는 대조(reconciliation)가 필요하다. 이것이 설정 주석의 "block until reviewed reconciliation"이고, 새 코드와 검토가 필요하다. 내일은 절차(새 회차 + 1분 안 확인, 체크리스트 맨 위)로 막는다. 대조 구현은 데모 뒤 후속으로 둔다.

**함께 고친 것:**
- `publisher.py` 끝의 별칭: `setdefault`가 다른 사본을 조용히 남기는 경우(먼저 import됨) **시작을 거부**하게 바꿨다(커밋 `3e40baa3`, 테스트 `test_refuses_to_start_when_already_imported`). 실제 `-m` 실행은 그대로 시작한다(`--help` 확인).
- 흔들리던 `test_blocked_prefix_cannot_starve_later_eligible_path`: 실패 메시지 `assert 141 == 140`(`len(sink.pending)`)을 부하 중 20회 반복에서 1번 잡았다. 원인은 테스트가 "서버에 저장됨"만 기다리고 바로 `pending`을 세는 것 — `pending`은 작업자가 자식의 응답을 읽은 뒤에야 비워진다(`publisher.py:666-670`). 48eefe9f와 무관(수정 전 2/10, 수정 후 2/10, 같은 메시지). 테스트가 `pending`에서 빠질 때까지 기다리게 고쳤다(커밋 `c2b12fc7`): 부하 중 30회 0 실패.
- publisher 테스트 전체 **345 passed**(새 테스트 포함).
- `test_public_viewer_e2e`·`test_public_viewer_gateway_e2e` 실패는 환경 문제가 아니라 의도된 핀 거부다: 이 가지의 `apps/observer/web/src/preview/…` 파일이 10/19 핀(`dccde711748d`)과 달라 "source pin mismatch"로 멈춘다(브라우저 시작 전).

## 5. 화면 갱신 간격 — 발행 간격을 따른다

공개 화면은 진행 중 경기를 **manifest의 `publish_interval_s`마다** 읽는다(`web/src/service/GameScreen.tsx:332`, `api.ts:163`). 없을 때만 10초(`livePollMs`)를 쓴다. 그래서 11/15 07시대의 "화면이 0~10초 늦다"는 발행 간격 10초 때문이었다.
- 발행 간격을 2초로 하면 화면도 2초마다 읽는다(코드 기준). 11/15 17시대 브라우저로 재 보니 파일 → 화면 반영 중앙값 0.72~0.87초, 최대 1.91초였다(4.2절, 묶음 D).
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

- 브라우저에서 본 지연은 4.2절에서 쟀다(헤드리스 Chromium). **실제 아이폰 Safari에서는 미측정.**
- 영상과 실제로 나란히 맞추기, 영상의 경기 시계 보존 여부(광고 편집): 영상이 없어서 미확인.
- 호스트 업로드, 접근 코드 쿠키, 폰 장시간 폴링: Song 결정 범위라 실행 안 함. 미측정.
- `fixed_delay_s`를 실제 속도 재생과 함께 쓰는 것: 시험 안 함.
- ssh로 띄울 때 명령이 바로 돌아오는지: 실제 ssh는 못 했다(자기 키 미등록). 출력 파이프는 0.05초에 닫혔고 발행기는 계속 돌았다(11/15 16:27). 그래서 바로 돌아온다고 추정한다.
- 화면 반영 지연과 맥북 반쪽 창 배치: 16시대에는 건너뛰었고, 17:51~18:05에 묶음 D로 쟀다(4.2절). 명령 → 화면 첫 추천은 여전히 미측정.
- ssh 바로 돌아오기: 17:52 재시도도 키 미등록으로 거부. 미측정(추정만).
- 10초 간격의 화면 지연(묶음 D), 실제 호스트의 429 동작: 미측정(4.3·4.4절). 경기 파일 멈춤은 4.5절에서 원인을 찾고 고쳤다. 실제 호스트에서의 재발 여부는 미측정.

## 부록. `run_realpace.sh` 본문 (`~/Projects/pitcheezy-pm-jobs/20261115-demo/`, 11/15 22시대, `site/` 분리·폴더 이름에 포트·`PZ_TRACE`·`SPEED`)

분석 스크립트: `20261115-realpace/timeline.py`(최종 피드 → 투구 시간표), `drift.py`(회차 폴더의 `watch_rt.tsv`와 시간표 비교). 발행 파일 감시는 `20261115-demo/watch_rt.py`.

```sh
#!/bin/zsh
# usage: run_realpace.sh <port> <start-pitch> <lead-s> [lag-s=0] [screen-dir] [publish-interval=10] [poll-s=2]
# 849843 at its REAL pitch times (PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE=game, demo only). Pitch <start> is released
# <lead-s> after the moment this script runs (T0, recorded in t0.txt); every later pitch follows at the game's own spacing.
# lag-s: each pitch becomes visible to the publisher that long after its release (feed lag rehearsal).
# Publish config: PUBCONF=<file outside the repo> run_realpace.sh ... (default apps/observer/publish_config.json, upload OFF).
#   Upload stays off unless that file says upload.enabled true; its token goes in a 0600 token_file outside the repo.
# PZ_TRACE=1: run through trace_publisher.py (debug only: upload_failed lines carry their traceback).
# SPEED=<x> (default 1): replay x times faster (counting runs only, not for the demo).
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
RUN=$D/runRT-$(date +%H%M%S)-$PORT; [[ -e $RUN ]] && { echo "회차 폴더가 이미 있음: $RUN"; exit 5; }; mkdir -p $RUN/out
# Screen files go to site/, NOT out/: a non-empty out/ with no state makes the publisher set upload_recovery_blocked
# (publisher.py:929-930) and then nothing is ever uploaded. site/v1 -> out/v1 so the local page reads the same files.
mkdir -p $RUN/site; rsync -a --exclude '/v1' $SCREEN/ $RUN/site/; ln -s ../out/v1 $RUN/site/v1
echo "$SCREEN" > $RUN/screen.txt; echo "${PUBCONF:-apps/observer/publish_config.json}" > $RUN/config.txt
T0=$(python3 -c 'import time;print(round(time.time(),3))')
print "T0 $T0 game $GAME start $START lead $LEAD lag $LAG interval $INTERVAL poll $POLL speed ${SPEED:-1}" > $RUN/t0.txt; echo $T0 >> $RUN/t0.txt
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
  PITCHEEZY_OBSERVER_LIVE_REPLAY_SPEED=${SPEED:-1} PITCHEEZY_OBSERVER_LIVE_REPLAY_LEAD_S=$LEAD PITCHEEZY_OBSERVER_LIVE_REPLAY_LAG_S=$LAG PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S=$POLL \
  $HOME/Projects/pitcheezy/.venv-observer/bin/python -B ${=${PZ_TRACE:+$D/trace_publisher.py}:--m observer_app.publisher} --out $RUN/out --interval $INTERVAL --date $PDATE \
  --config ${PUBCONF:-apps/observer/publish_config.json} > $RUN/publisher.log 2>&1 &
echo $! > $RUN/publisher.pid
python3 -m http.server $PORT --bind 127.0.0.1 --directory $RUN/site > $RUN/http.log 2>&1 &
echo $! > $RUN/http.pid
echo $RUN
```

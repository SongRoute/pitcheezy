# 10/6 16:00 시연 운영 문서 (현재본)

갱신 2026-10-06 07:12 KST (호스트 시계): 투수 시점 화면(ui-pitcher-view, 검토 PASS·Root 수용 `ab2393b`)을 프로세스 재시작 없이 정적 자산만 교체해 반영 — 1·5·7·8절. 이전: 2026-10-06 05:35 KST (호스트 시계): 차분한 화면(ui-calm-polish, 독립 검토 ACCEPT·Root 수용 `2b32e30`) 반영 — 1·5·7·8절. 이전 갱신 00:42 KST(통합 담당 Claude Opus 5.5). 이전 내용은 [보관본](UI-live-demo-1006-runbook-archive-20261006.md)에 그대로 있다(사고 기록 포함).
진행 로그: `pm-jobs/20261006-four-hour/demo-delivery/LIVE.md`.

## 1. 지금 상태
- 미리보기 `http://100.108.252.111:8771/watch` (Tailscale 기기 전용). PID **74837** (2026-10-06 05:33:44 시작), cwd `pitcheezy-worktrees/codex-ui-live-preview-1005`, 브랜치 `codex/demo-pa-anchors-1006`, 서빙 소스 **`ab2393b`** (화면만 바뀜; 백엔드 코드는 2b32e30과 같아 05:33 시작 프로세스 그대로. a99f63b에서 fast-forward, push 안 함). 이전 PID 39361(소스 00d460c, 화면은 a08cbeb와 같음).
- 서빙 자산 sha256 (ab2393b): `index-CgS2UPt-.js` 619e3d39…, `index-DtJIeU32.js`(운영자 화면) 9ef035e8…, `index-DQLUaASQ.css` d98cd901…, `index.html` 46d38d29…. 열려 있던 화면용으로 이전 자산(`index-Dqklzlty.js` 40aac20b…, `index-Bb-VAAQr.js`, `index-BRy8UNf-.css`)도 dist/assets에 남겨 둠. 서빙 자산 sha256: `index-Dqklzlty.js` 40aac20b…, `index-Bb-VAAQr.js`(운영자 화면) b08a73f6…, `index-BRy8UNf-.css` ea17c39a…, `index.html` a4308027… (작성자·검토자 빌드와 바이트 동일). 이전(00d460c): `index-BP8kCSE6.js` 249daf8c…, `index-0OqeeY6u.js` ec4d9183…, `index-CJLA7S2E.css` db53763d…, `index.html` 9b883845….
- env: `PITCHEEZY_OBSERVER_PORT=8771 HOST=tailscale LEGACY=0 LIVE_ARMB=0`, `PITCHEEZY_DEMO_DIR=/Users/song/Projects/pitcheezy-demo-sessions/preview8771`, `PITCHEEZY_WATCH_DIR`/`PITCHEEZY_FEED_DIR` = `pm-jobs/20261005-ui-live-redesign/preview-data-overlay-014/overlay/{watch,feeds}`(16경기 읽기 전용 사본), `PITCHEEZY_PLAYER_SNAPSHOT_DIR` = `…/pregame-data-032/cache`(경기 전 기록 2경기).
- 화면: rev3 구조 그대로(가운데 존, 추천 3개, 투구 카드, 선수 칸, 이닝표) + 차분한 색·굵기(2b32e30). 존 좌표·창 밖 표시 규칙 그대로. 격자 390px 144px(이전 128), 이름·날짜 줄바꿈(잘림 없음), 날짜에 연도, 경기 목록 카드 깨짐 수정.
- **투수 시점(ab2393b)**: 실제 공·추천 네모·미트·창 밖 표시·미니 카드 가로만 거울(화면 x = −plate_x, 저장 좌표·높이·응답 그대로). 우타자 화면 오른쪽. 추천은 타자 반대쪽 한 줄씩(순위·비중·구종) 존 바로 위, 실제 표식 글자 '구종 구속', 결과는 존과 카드 사이 크게, '지난 기록' 줄은 맨 아래, 카드 캡션은 구종 코드(CS=슬로 커브), 투수 투구 수·타자 몇 번째 타석(공개된 기록만). '포수 시점' 문구 없음.
- 응답 추가 필드 `pitches[].linescore_before`(그 공 직전 시점 이닝표). 지난 공을 고르면 이닝별 기록이 그 시점까지만 나온다(이전에는 마지막 시점 기록이 보였음). 설명: `apps/observer/CONTRACT.md` '선택 시점 이닝표'.

## 2. 주소
- 목록 `/watch` · 849843 `/watch/849843`(2026-09-30(수) 11:00 KST, CHC 0–8 SD, CV 미트 x) · 813031 `/watch/813031`(2025-10-18(토) 09:38 KST, MIL 1–5 LAD) · 운영자 `/demo-operator`.
- 운영자 토큰(맥미니에서 화면으로만 확인, 메시지·문서에 붙이지 않음):
  `PITCHEEZY_DEMO_DIR=/Users/song/Projects/pitcheezy-demo-sessions/preview8771 /Users/song/Projects/pitcheezy/.venv-observer/bin/python -c "import sys; sys.path.insert(0,'/Users/song/Projects/pitcheezy-worktrees/codex-ui-live-preview-1005/apps/observer/backend'); from observer_app.demo_clock import DemoClock; print(DemoClock().token())"`
- 시연 세션이 진행 중인 동안 그 경기의 일반 주소는 잠긴다(미래 기록 노출 방지). 끝나면 '시연 종료'.

## 3. 시연 방식 (위에서부터 우선)
### A. 영상 + 타석 끝 공개 (기본) — 대응표 `g813031-ab1-3-pa-end`
- 근거: Song이 YouTube `45Rqrp4aZM4`에서 직접 확인한 **타석 끝 결과 시각 3개**: AB1 투랑 볼넷 7:28(448초), AB2 추리오 헛스윙 삼진 8:58(538초), AB3 옐리치 루킹 삼진 11:13(673초). 나머지 13구 시각은 모름(보간 안 함).
- 동작: 각 타석이 끝나는 관찰 시각에 그 타석 투구가 **한꺼번에** 공개되고 다음 타자 대기 화면으로 넘어간다. 투구별 동기화는 주장하지 않는다(`verified=false`, 종류 `pa_end_observed`).
- 순서
  1. 맥북: 운영자 화면 → 토큰 → 대응표 `g813031-ab1-3-pa-end` → 세션 만들기. 상태 ready, '영상 기준 시각' **6:58**(=448−30초), 아이폰에는 1회 초 시작 전(0:0, 투구 없음).
  2. 아이폰 Safari: 운영자 화면의 관람 주소 `/watch/813031?demo=<세션ID>`.
  3. 맥북 YouTube를 **6:58**로 옮겨 일시정지(0초·임의 지점이 아님).
  4. "3, 2, 1"에 영상 재생과 '시작'을 손으로 같이. 정지는 영상→시계, 재개는 시계→영상. 서버는 YouTube를 제어하지 않는다.
  5. 7:28·8:58·11:13에 아이폰 화면이 타석 단위로 바뀐다. 어긋나면 일시정지 후 '특정 투구로 이동'(5·9·15만 가능)으로 맞춘다.
  6. 끝나면 '시연 종료(전체 기록 다시 열기)'.
- 길이: 6:58→11:13 = 4분 15초(1배). 6:58은 AB1 도중일 수 있다(피드상 AB1 약 1분 26초, 영상 위치는 모름).
- 선택: AB1을 더 앞에서 보여 주려면 대응표 `g813031-ab1-3-pa-end-lead90`(같은 관찰 3개, 시작 **5:58**, 길이 5분 15초). 화면은 7:28까지 1회 초 시작 전 상태로 기다린다. 5:58이 AB1 시작이라는 뜻은 아니다.
- 실측(헤드리스, 1배속, 스테이징): 서버 공개 30.42/120.06/255.37초(영상 7:28/8:58/11:13), 화면 갱신은 공개 뒤 1.0–1.9초(2초 갱신 주기). 4배속(8771): 공개 7.58/30.24/63.94초(기대 7.5/30/63.75), 시작 전 화면 투구 0·점수 0:0, 시청자 둘 일치, 탐침 90/90. 스테이징에서 시연 중 서버 재시작: 중단 0.75초, 같은 컷오프로 이어짐.

### B. 영상 없이 시계만 (영상이 안 될 때)
- 대응표 `g813031-ab1-3-pace` 또는 `g849843-ab4-8-pace`(경기 피드 시각 진행 속도표, 영상 대조 아님). 세션 만들기 → 시작. "과거 경기 기록을 실제 진행처럼 재현"한다고 고지, 영상과 맞췄다고 말하지 않는다.
### C. 네트워크가 안 될 때
- 정적 캡처 `pm-jobs/20261006-four-hour/demo-delivery/offline-fallback/index.html`(외부 요청 없음, sha256 21d11cae…): A의 네 시점(시작 전·AB1/2/3 끝 직후) 화면, 캡처 2026-10-06 00:40 KST, 같은 빌드. 실시간이 아니라고 말한다.
- CV 비교 자료(오프라인): 첨부 `~/.codex/attachments/2fe05916-…/index.html`(sha256 32159ccf…). MLB 화면 포함 → 현장 로컬에서만 연다.

## 4. 5분 대본 (숫자는 실측·기존 합의 문구 그대로)
- 0:00–0:20 고지: "지난 경기를 실제 진행처럼 다시 보여 드립니다. 영상은 맥북에서 따로 틀고, 휴대폰은 서버 시계로 자동 갱신됩니다."
- 0:20 시작(A의 4단계). 화면 설명: 가운데 스트라이크존, 추천 1·2·3(점선 네모, 퍼센트는 정책이 고른 비중이지 성공 확률이 아님), 아래 투구 카드, 위 선수 칸(경기 전 정규시즌 기록, 오타니는 투수 ERA 2.87).
- 7:28(영상) AB1 끝: 화면이 바로 다음 타자(추리오) 대기로 넘어간다(약 1–2초 뒤). 선수 칸을 눌러 타석 목록에서 투랑 타석을 고르면 그 타석 6구 카드가 나오고, 카드를 누르면 그 공 던지기 전 추천과 실제가 함께 보인다. LIVE를 눌러 최신으로 복귀.
- 8:58 AB2 끝(1아웃), 11:13 AB3 끝(2아웃, 다음 타자 콘트레라스 대기 — 이닝은 아직 안 끝남). 사이 시간에 이닝표·타석 목록 시트를 보여 준다.
- 끝 30–40초 CV(849843 화면 또는 요약 PNG): "평가 2경기(849845, 823407) 86장 중 AI 출력 58장·기권 28장, 둘 다 표시한 58구 점 차이 중앙값 2.5px, 같은 변환 기준 약 1.1인치. 물리 정확도·투수 의도 검증 아님." 시연 경기 849843은 평가 경기가 아니고, 39구 중 27구만 실제 투구 공개 **뒤** 가로(x) 위치만 표시(12구 기권은 표시 없음).
- 하지 않을 말: "실시간 추천", "투구별 영상 동기화", "예측 성능 향상", "물리 정확도 1인치", "의도 검증".
- 추천 출처: 경기 뒤 그 공을 던지기 전 정보만으로 계산한 기록(asof_replay). 정책·과학적 판정은 바뀌지 않았다.

## 5. 시연 전 점검 (15:30)
- 한 번에(권장, 현재 화면 해시판 v4): `sh /Users/song/Projects/pitcheezy-pm-jobs/20261006-ui-pitcher-view/integrate-001/preflight-pitcher-ab2393b.sh` — v3와 같은 점검에 ab2393b 자산 해시를 고정. 마지막 줄 `v4 checks 0 failure(s)`와 종료 코드 0이면 통과(2026-10-06 07:12 실행 0/0).
- (보관) v3 — 2b32e30 자산 고정이라 지금은 자산 줄 실패가 정상: 한 번에(권장, 현재 화면 해시판 v3): `sh /Users/song/Projects/pitcheezy-pm-jobs/20261006-ui-polish/integrate-001/preflight-calm-2b32e30.sh` — 원본 점검 + 2b32e30 번들·CSS·index 고정 해시 + 예비·CV 해시 + 두 경기 `linescore_before` 확인. 마지막 줄 `v3 checks 0 failure(s)`와 종료 코드 0이면 통과(2026-10-06 05:35 실행 0/0).
- (보관) v2 `next-research/preflight-verified.sh`는 이전 00d460c 번들 해시를 고정하므로 지금은 자산 줄에서 **실패하는 것이 정상**이다. 증거로 남겨 둔다. 원래 설명: — 아래 원본 점검을 그대로 돌린 뒤 서빙 번들·운영자 JS·CSS·index, 예비 화면, CV 오프라인 index·manifest의 sha256을 고정값과 **비교**한다(01:17 KST 실행: 원본 0 failure, 비교 8/8 OK). 마지막 줄 `hash comparisons 0 failure(s)`와 종료 코드 0이면 통과.
- 원본(출력만): `sh /Users/song/Projects/pitcheezy-pm-jobs/20261006-four-hour/demo-delivery/preflight.sh` (읽기만 함; 8771·자산·세션·대응표 4개·overlay 해시·경기 전 기록·예비 파일 확인, 실패 0이면 통과). 00:52 실행 결과 0 failure.
1. `curl -s http://100.108.252.111:8771/api/health` → status ok.
2. 운영자 화면에서 활성 세션 없음 확인(있으면 '시연 종료').
3. 맥북 YouTube 6:58 일시정지, 아이폰 Tailscale 켜짐.
4. 예비 C 파일 두 개를 맥북에서 연다(오프라인 확인).

## 6. 사람만 할 수 있는 확인 3가지
1. 아이폰 Safari(기본·큰 글자)에서 현재 화면이 스크롤 없이 보이고 존이 가운데인지.
2. A 방식 리허설 1회: 맥북 영상+운영자, 아이폰 관람 주소, 7:28/8:58/11:13에 화면이 바뀌는지, 새로고침해도 같은지.
3. 나머지 13구 결과 시각(선택): 채우면 투구별 대응표(`build_demo_mapping.py --from-csv … --video-url …`, 보관본 §3)로 바꿀 수 있다. 없어도 A로 시연 가능.

## 7. 재시작·되돌리기 (8771만)
- 맥미니 재부팅·프로세스 종료 시: `sh /Users/song/Projects/pitcheezy-pm-jobs/20261006-four-hour/demo-delivery/start-preview-8771.sh`(현재와 같은 env, 8771이 이미 쓰이면 거부). 시연 세션·대응표는 디스크에 있어 그대로 이어진다(스테이징 재시작 실측 0.75초, 같은 컷오프). 05:33 재시작 때 활성 세션 0(16개 모두 ended).
- **현재 되돌리기 지점 = a99f63b(제품 2b32e30)**, 재시작 불필요(백엔드 같음): `rsync -a pm-jobs/20261006-ui-pitcher-view/integrate-001/rollback/dist-a99f63b/ <preview 트리>/apps/observer/web/dist/`(index.html 포함; 새 자산은 남아도 무해) 후 `git -C <preview 트리> checkout --detach a99f63b`(또는 브랜치를 `reset --hard a99f63b`). 해시는 `pm-jobs/20261006-ui-pitcher-view/integrate-001/rollback/dist-a99f63b.sha256`, 확인은 v3 점검(0 실패여야 함).
- 그 이전 되돌리기 지점 00d460c(백엔드도 되돌려야 해 재시작 필요)(타석 끝 관찰 시연 포함, 3ca9efc 아님): 활성 세션 확인 → `git -C <preview 트리> checkout --detach 00d460c`(또는 `git reset --hard 00d460c`는 브랜치를 되돌림) → `rsync -a --delete pm-jobs/20261006-ui-polish/integrate-001/rollback/dist-00d460c/ <preview 트리>/apps/observer/web/dist/` → 8771 PID가 preview 트리 cwd인지 확인 후 그 PID만 `kill` → `start-preview-8771.sh`. 자산 해시는 `pm-jobs/20261006-ui-polish/integrate-001/rollback/dist-00d460c.sha256`, env 11개는 `pm-jobs/20261006-ui-polish/integrate-001/rollback/env-39361.txt`. 00d460c 백엔드는 `linescore_before`를 안 보내고, 화면도 이전 것이라 서로 맞다.
- 더 이전 3ca9efc(타석 끝 대응표 기능 전, 지금의 되돌리기 지점 아님): `git -C /Users/song/Projects/pitcheezy worktree add <새 디렉터리> 3ca9efc` → 그 트리 `apps/observer/web/dist`에 `pm-jobs/20261006-four-hour/demo-delivery/rollback-dist-3ca9efc/` 복사 → 현재 8771 리스너 종료 → 그 트리에서 1절의 env 그대로 `sh apps/observer/serve.sh`. (그 백엔드는 `pa_end_observed` 대응표를 목록에서 건너뛴다.)
  - **되돌리기 전에 반드시 A 방식(`pa_end_observed`) 세션을 운영자 화면에서 '시연 종료'한다.** 3ca9efc 백엔드는 디스크에 남은 그 세션을 그대로 읽지만 `pre_index`를 모르므로 시작 전 컷오프를 `start_index−1`=4로 계산해, AB1 투랑 타석 6구 중 5구를 7:28 전에 보여 준다(독립 검토 재현: 새 코드 −1, 옛 코드 4). 되돌린 뒤에는 B(시계만) 대응표로 새 세션을 만든다.
- 더 이전: c84905a(`rev3-label-polish-040/rollback-dist-c84905a/`), 15a4357, 최후 282dc6f(보관본 §4).

## 8. 남은 것
- 실기기(아이폰 Safari·두 기기) 미확인. 헤드리스 두 컨텍스트는 대신하지 않는다. 새 화면도 아이폰 미확인.
- 375×553 + 글자 125%처럼 아주 짧은 화면에서는 이름·날짜를 자르지 않는 대신 존 격자가 91~118px로 이전(119px)보다 작다(Root 수용). 키 큰 화면은 존 위아래 여백이 남는다(격자 크기가 폭에 묶임).
- 투구별 영상 동기화 없음(13/16 모름). 849843 전체 영상 막힘.
- 실제 진행 중 MLB 경기 미검증, 실라이브 다음 공 추천 꺼짐(LIVE_ARMB=0), 경기 전 기록은 2경기만.
- SQLite 라이브 저장소(c6010ef+e5b943b, 검토 ACCEPT WITH FIX)는 실시간 지연 버퍼용이라 이 시연(보관 경기 재현)에 쓰이지 않는다. 시연 시계 상태는 이미 세션 파일로 재시작을 견딘다(0.75초). 16:00 전에는 넣지 않는다.
- 사고 기록(127.0.0.1:8766 세션 4개·T7 파일 6개)은 보관본에 그대로. 8766·8770은 이 작업에서 건드리지 않았다(PID 3277·94628, 149).

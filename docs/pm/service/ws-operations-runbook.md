# 월드시리즈 운영 런북 (초안, 2026-10-17)

월드시리즈(미국 날짜 10/23–31, 시작 시각 미정) 공개 서비스를 맥미니에서 돌릴 때 보는 한 장짜리 안내다. 기준 코드는 `ws/launch-2026` `a9d52c4`이고, 값은 `apps/observer/publish_config.json`·`live_config.json`·코드에서 읽었다. 코드·설정에서 확인하지 못한 것은 **미확인**, 아직 실제로 재 보지 않은 숫자는 **잠정**으로 적는다.

용어 한 줄 풀이:
- **발행기(publisher):** 경기 자료로 공개 화면이 읽을 JSON 파일을 미리 만들어 두는 프로그램. 팬 브라우저는 맥에 직접 닿지 않고 이 파일만 읽는다.
- **업로드 일꾼(upload worker):** 발행기 안에서 도는 보조 부분. 만든 파일을 팀원 호스트로 하나씩 올린다. 따로 띄우는 프로그램이 아니다.
- **health.json:** 발행기가 매 주기 쓰는 "지금 살아 있음" 파일. 화면은 이 파일로 자료가 신선한지 판단한다.

## 1. 켜고 끄기

### 발행기 켜기
- 명령(코드 기준): `PYTHONPATH=apps/observer/backend .venv-observer/bin/python -m observer_app.publisher --out <출력 폴더>`
  - `--once`를 빼면 `publish_interval_s`(기본 10초, **잠정**)마다 계속 돈다.
  - `--date`를 안 주면 오늘(KST) 앞뒤 1일 경기를 매 주기 다시 만든다(`live_days_back/ahead` 1, **잠정**).
  - 출력 폴더는 저장소 밖이어야 한다. `--out` 또는 `PITCHEEZY_PUBLISH_OUT_DIR`.
- 반드시 필요한 것: 실시간 저장소 `PITCHEEZY_OBSERVER_LIVE_STORE`와 추천 장부. 둘 중 하나라도 없으면 시작을 거절한다. 도는 중 장부를 끄면(`PITCHEEZY_OBSERVER_REC_LEDGER=0`) 그 주기부터 아무것도 쓰지 않는다.
- 추천(ARM-B)을 끄고 돌리려면 `PITCHEEZY_OBSERVER_LIVE_ARMB=0`. 켜 두면 시작할 때 고정 정책을 백그라운드로 데운다.
- 공통 지연: `fixed_delay_s`(기본 0) 또는 `--delay`/`PITCHEEZY_PUBLISH_DELAY_S`. `live_config.json` `max_delay_s` 120을 넘으면 시작을 거절한다. **0초/15초는 Song 결정 대기.**
- 한 출력 폴더에는 발행기 하나만. 폴더 옆 `.<폴더이름>.lock`을 잡으며, 이미 다른 발행기가 잡고 있으면 `AlreadyRunning`으로 시작하지 않는다.
- 실경기용 정확한 환경 변수 묶음(피드·프로필·저장소 경로)과 실행 위치(tmux 창, launchd 여부): **미확인.** 저장소에 발행기용 launchd 예시 파일은 없다(`apps/observer/deploy/`에는 observer·live-store-staging·demo-sync 예시만 있다). 리허설 E에서 정해 이 절에 적는다.
- **운영 실행기(2026-10-18 H, 검증 전):** `apps/observer/ops/ws_operate.py` `plan` / `start --date <KST 날짜>` / `status` / `stop`. 리허설 E의 환경 묶음(최소 자식 환경, 날짜별 새 저장소, 빈 피드 폴더)에 참고 카드 경로를 더해 발행기를 띄우고, 예상 밖 종료는 30/60/120초 뒤 최대 3번 다시 띄우며, 경기 중 다른 커밋으로는 다시 띄우지 않는다. 사용법은 `apps/observer/README.md` "월드시리즈 운영 실행기", 경고 감시기가 읽는 상태 파일은 [ws-ops-status-contract.md](ws-ops-status-contract.md). launchd 예시는 설치하지 않았다. 설치하면 리허설이 매일 07:00 start와 겹치지 않게 한다(겹치면 30분 재시도 뒤 긴급 `start_refused`). 경기 없는 날의 start는 아무것도 쓰지 않는다.

### 업로드 켜기
- `publish_config.json` `upload.enabled: true`, `upload.base_url`(전용 HTTPS `/_publish` 경로를 정확히 한 번 포함), 토큰은 `PITCHEEZY_PUBLISH_TOKEN`(또는 `token_file`). 기본은 꺼짐이다.
- 켜기 전 확인: `python -m observer_app.publisher --check-remote` — 인증 HEAD 한 번만 하고 쓰기는 안 한다.
- 네트워크 없이 경로만 보고 싶으면 `--dry-run-upload`(로컬 파일은 실제로 만든다).
- 실제 호스트 주소·토큰 발급/폐기 절차: **미확인**(팀원 쪽 F 작업, 아직 연결 안 함). 주소와 토큰은 공개 이슈·문서에 적지 않는다.

### 참고 카드(A2 `next.reference`) — 2026-10-18 Song 결정으로 공개 발행에서 켬
- 뜻: 추천 모델이 모르는 투수(2026 신인 등)의 다음 공 자리에 "이 투수가 올해 이 카운트·타자 방향에서 자주 던진 공"을 비율로 보여 주는 참고 카드. 추천·예측이 아니고 순위·목표 위치가 없다. 모델을 돌리지 않고 매일 갱신되는 2026 투구 기록 프로필에서 횟수만 센다.
- 설정: `apps/observer/publish_config.json` `profile_reference.enabled: true`(커밋됨). 경로는 저장소에 적지 않는다(`profile_sources: null`). 발행기를 켤 때 환경 변수로 준다:
  - `PITCHEEZY_OBSERVER_PROFILE_SOURCES=~/Projects/pitcheezy-pm-jobs/profile-refresh-daily/profiles/LATEST.json` (매일 21:00 KST launchd `com.pitcheezy.profile-refresh`가 새 프로필을 만든 뒤 옮기는 포인터. 코드는 `~`를 풀지 않으므로 셸이 펼친 절대 경로가 들어가야 한다. 따옴표로 감싸지 않는다.)
  - 끄기: `PITCHEEZY_OBSERVER_PROFILE_REFERENCE=0`(설정 파일을 고치지 않고 발행기만 다시 켬). 끄면 카드 필드가 사라질 뿐 나머지 본문은 같다.
- 읽는 규칙(코드 `observer_app/profile_reference.py`): 포인터를 최대 30초마다 다시 읽고, 프로필 마감일(cutoff)이 경기 officialDate보다 **엄격히 앞선** 것 중 가장 새것을 쓴다. 정상이면 마감일 = 경기 전날(1일 차).
  - 칸 최소 투구 수 20(카운트×타자 방향, **잠정**, 유지), 넓은(모든 카운트) 대체 카드 **끔**(유지).
  - 신선도: 마감일이 경기일보다 **2일 넘게** 앞서면(일일 갱신이 두 번 이상 연속 실패) 카드는 그대로 나가되 `stale: true`가 붙고 화면 줄에 "기록 기준 MM/DD"가 보인다. 2일은 **잠정**(PM 선택).
  - 포인터가 없거나 깨졌거나 쓸 수 있는 프로필이 없으면 카드만 없다. 발행·추천에는 영향이 없다.
- 켤 때 확인: 발행기 시작 로그 한 줄 `reference card (A2): {...}` — `enabled: True`, `usable`에 `True`와 `cutoff`(어제 날짜)가 보여야 한다. `WARNING reference card (A2) is on but no profile source ...`가 보이면 경로를 고친다.
- 매일 확인(경기 전날 밤): **21:20 KST까지** `~/Projects/pitcheezy-pm-jobs/profile-refresh-daily/logs/RESULTS.log` 마지막 줄이 `ok target=<KST 오늘 날짜> cutoff=<어제 날짜>`인지 본다(예: 10/17 21:00 KST 실행 → `2026-10-17T12:01:28Z ok target=2026-10-17 cutoff=2026-10-16`). target은 그 미국 날짜 경기에 쓰일 프로필이라는 뜻이다. `failed`이면 포인터는 이전 것을 그대로 두므로 카드는 하루 묵은 기록으로 나간다(1일 실패는 아직 stale 아님). 이틀 연속 실패면 카드에 "기록 기준"이 붙는다. 디스크 여유가 12 GiB 하한 근처면 그날 갱신이 건너뛰어지므로 경기 주간에는 여유를 먼저 확보한다(삭제는 Song 확인).
- 감사(PublishAudit): `reference_card_integrity` 불변식이 카드에 순위·목표·위치·확률 키가 없는지, 비율 합 ≤ 100, `as_of` < 경기일, 투수 단위 "추천 없음" 상태의 다음 공에만 붙었는지 본다.

### 끄기
- 발행기를 띄운 터미널에서 Ctrl-C. 코드는 KeyboardInterrupt를 받아 `close()`로 업로드 자식 프로세스를 정리한 뒤 출력 lock을 놓는다(`shutdown_grace_s` 5초, **잠정**).
- 다른 방식(kill·launchctl 등)으로 멈추면 `close()`가 돌지 않는다. 그때 전송 중이던 업로드는 재시작 때 `blocked`로 남는다(4.4 참고). 그러니 Ctrl-C로 끈다. **기록기·운영 서버 등 다른 프로세스에는 절대 신호를 보내지 않는다.**
- 업로드만 끄고 로컬 발행은 계속하려면: `upload.enabled: false`로 바꾸고 발행기를 다시 켠다(밀린 기록은 상태 파일에 남는다).

## 2. 무엇을 보나

모두 `<출력 폴더>/v1/health.json`과 발행기 로그에서 본다.

| 항목 | 뜻 | 정상 |
|---|---|---|
| `published_at`, `sequence` | 마지막 주기 시각과 순번 | 매 주기(10초) 앞으로 간다 |
| `game_checks[경기].checked_at` | 그 경기를 마지막으로 확인한 시각 | 진행 중 경기는 매 주기 갱신 |
| `game_checks[경기].feed_ok` | 피드·본문 확인 성공 | true |
| `game_checks[경기].sequence_checked` vs `file_sequence` | 확인한 주기 순번 vs 경기 파일이 마지막으로 바뀐 순번 | `file_sequence`가 뒤처지는 것은 정상이다(C 이후 내용이 같으면 파일을 다시 쓰지 않음). 확인 자체(`checked_at`)가 멈추면 이상 |
| `feed_failing_games` | 피드가 실패 중인 경기 | 비어 있음 |
| `unavailable_games` | 경기별 실패 코드(`build_failed`, `write_failed`, `not_public`, `refused`, `setup_failed`, `upload_failed`, `withdraw_failed`, `archive_unread`) | 비어 있음 |
| 경기 파일 `display_state.feed.delayed`, `lag_s` | 피드가 늦게 오는지 | delayed false |
| `upload.pending`, `upload.code` | 아직 못 올린 파일 수 | 0, null |
| 상태 파일(기본 `<출력 폴더>.state.json`)의 `upload_attempts` | 경로별 전송 시도·`phase=blocked` 사유 | blocked 없음 |
| 맥 메모리 | 기록기·수집이 먼저다 | 아래 임계 참고 |

참고 실측(849831 오프라인 재생, 실경기 아님): ARM-B 켬 peak RSS 약 2.0 GB, 끔 약 0.12 GB. 주기 처리 시간 p50 약 0.03초(ARM-B 끔). 정책 계산 중앙값 약 0.12초, 최대 6.9초.

## 3. 경고 기준

| 무엇 | 기준 | 근거 |
|---|---|---|
| health 정체 | `published_at`이 30초(3주기) 넘게 안 바뀜 | 화면의 "갱신 지연" 기준 `max(VITE_PUBLISH_STALE_S=30, 3×주기)`와 같게 둠. **잠정**(리허설에서 실제 주기·흔들림을 재서 정함) |
| 경기 확인 정체 | 진행 중 경기 `checked_at`이 30초 넘게 멈춤 | 위와 같음. **잠정** |
| 피드 끊김 | `feed_ok:false` 또는 `feed_failing_games`에 경기 | 화면이 즉시 "연결 끊김"을 띄움 |
| 피드 지연 | `delayed:true` (lag > 45초) | `display_state.py` `feed_lag_delayed_s` 45. 10/2 실측 피드 지연 중앙값 약 19초. 45는 **잠정** |
| 업로드 밀림 | `upload.pending` > 0이 3주기 넘게 계속 | **잠정**(실제 호스트로 재 본 적 없음) |
| 업로드 막힘 | `upload_attempts`에 `phase=blocked` 하나라도 | 자동으로 풀리지 않으므로 바로 사람 확인 |
| 맥 메모리 | 메모리 압박 "노랑" 이상, 또는 스왑 증가 | **잠정**. 24 GB 맥에서 수치 기준은 재지 않았다 |
| 프로필 일일 갱신 | 21:20 KST에 `RESULTS.log` 마지막 줄이 그날 `ok`가 아님 | 하루 실패는 카드가 하루 묵음, 이틀 연속이면 "기록 기준 MM/DD" 표시(1절 참고 카드) |

## 4. 사고별 대응

### 4.1 피드 끊김
- 증상: `feed_ok:false`, `feed_failing_games`. 화면은 추천·실제 투구를 가리고 "연결 끊김"을 띄운다.
- 할 일: 발행기는 그대로 둔다(다음 주기에 다시 묻는다). 로그에서 MLB 응답 오류를 확인하고, 다른 경로(브라우저 등)로 MLB 피드 자체가 살아 있는지 본다.
- 5분 넘게 계속되면 팀원 쪽 안내 문구 전환(4.6)을 고려하고 이슈에 기록한다. 5분은 **잠정**.

### 4.2 투구 ID가 빠짐(공급자가 이미 준 공의 ID나 내용을 바꿈)
- B 규칙으로 이미 받아들인 공은 마지막 수락 내용으로 계속 보인다(`held_last_accepted`). 한 번도 받아들이지 않은 내용은 나가지 않는다.
- 현재 타석 공이 보류·확인 중이면 추천만 거둔다. 이전 타석만 문제면 추천은 계속 나간다.
- 할 일: 보통은 없음. 같은 경기에서 확인 중 상태가 몇 분씩 이어지면 로그와 저장소의 격리(quarantine) 줄 수를 기록해 두고 경기 뒤 본다. 경기 중 코드 수정·배포는 하지 않는다(5절).

### 4.3 같은 시각 표시 충돌(same-stamp conflict)
- 뜻: MLB가 같은 시각 표시로 내용이 다른 본문을 보냄. 추천 정책 입력이 바뀐 경우에만 경기 단위로 잠깐 멈춘다(849831 재생에서 1177주기 중 21주기).
- 할 일: 몇 주기 안에 풀리면 정상. 한 경기에서 연달아 계속되면 시각과 경기를 기록하고 경기 뒤 본다. 기준 "연달아 몇 주기"는 **잠정, 미정**.

### 4.4 업로드가 멈춤(pending 증가, 412, unknown)
- 코드는 timeout·비정상 200·예상 밖 2xx·5xx·412(다른 쪽이 먼저 바꿈)·재시작 때 남은 전송을 `blocked`로 남기고 **자동으로 다시 보내지 않는다.** 로컬 발행은 계속되고 health에 부분 실패가 보인다.
- 429는 Retry-After만큼 기다렸다가 다시 시도한다(정상).
- 하지 말 것: `upload_pending`·`upload_attempts`·목적지 기록을 지우기, 최신 ETag로 오래된 본문을 보내기. 자동으로 풀어 주는 reset 명령은 없다.
- 할 일:
  1. 업로드를 끈다(`upload.enabled: false` 후 Ctrl-C로 끄고 다시 켬). 로컬 발행은 유지.
  2. 상태 파일, 최신 로컬 SHA·revision, 고정 주소의 인증 HEAD 결과, 이전 발행기가 끝났다는 증거를 모은다.
  3. 별도 대사(맞춰 보기) 검토를 거친 뒤 다시 켠다. 그동안 원격에 이미 올라간 파일은 즉시 거둘 수 없으므로 필요하면 4.6.
- 412가 났다면 "다른 발행기가 같은 원격 경로에 쓰고 있는지"부터 확인한다(원격 경로당 발행기 하나 원칙).

### 4.5 맥 메모리 압박
- 원칙: **기록기·수집 프로세스가 먼저다.** 그쪽에는 신호를 보내지 않는다.
- 할 일(덜 아픈 순서):
  1. 같은 맥에서 도는 시연·재생·실험(무거운 작업, `.heavy.lock` 쓰는 일)을 멈추거나 시작하지 않는다(내가 띄운 것만).
  2. 그래도 부족하면 발행기를 ARM-B 끔(`PITCHEEZY_OBSERVER_LIVE_ARMB=0`)으로 다시 켠다. 이 경우 추천 없이 경기 상황만 나간다. 경기 중 재시작은 추천 결정 ID 문제(5절)와 겹치므로 Song에게 먼저 알린다.
  3. 발행기를 끈다. 화면은 health 정체로 "갱신 지연"을 띄운다. 팀원 쪽 안내 문구 전환(4.6)을 함께 한다.

### 4.6 즉시 내림 스위치(팀원 쪽)
- 협업 규약 초안: 문제가 보이면 두 사람 누구나 공개 화면을 안내 문구로 바꿀 수 있다. 스위치는 팀원 쪽 호스트에 있다. 내린 뒤 이슈에 사유를 적는다.
- 스위치 위치·방법·누가 누를 수 있는지: **미확인**(팀원 답 대기). 발행기를 끄는 것만으로는 이미 올라간 파일이 사라지지 않는다는 점에 주의한다.

## 5. 배포 규칙
- **코드 교체는 경기와 경기 사이에만 한다.** 특히 B(결정 키 변경)가 들어간 빌드를 경기 중에 올리면 모든 추천 결정 ID가 바뀌어, 이전 결정의 저장된 추천을 찾지 못하고 같은 공에 새 "최초" 줄이 생긴다.
- 10/21 이후에는 고장 수정만 허용한다(협업 규칙).
- 공개 화면은 health 우선 빌드여야 한다. 옛 화면 빌드는 C(안 바뀐 파일 건너뛰기) 때문에 거짓 "갱신 지연"을 보일 수 있다.
- 교체 전에 맥에서 백엔드 테스트와 웹 검사를 돌린다. 백엔드의 알려진 시간 의존 불안정 테스트: `tests/test_publisher_background.py::test_retry_cursor_passes_large_head_failure_prefix`(전체 실행 부하에서 가끔 pending 141 vs 140으로 실패, 단독 재실행은 통과; 2026-10-18 J 검토). 이것 하나만 실패하면 단독 재실행으로 확인한다. 웹 검사의 알려진 실패 7건(390×844 투구 띠가 탭 막대 밑으로 들어감)은 Song 화면 점검 전까지 허용한다.

## 6. 연락·보고
- 연락 채널: **미정(10/20까지 정함).** 팀원과 정한 뒤 여기에 적는다(공개 문서이므로 주소·번호는 적지 않고 "어디"만 적는다).
- 기록: 사고와 조치는 상황판 이슈(SongRoute/pitcheezy#35)에 댓글로 남긴다. 비밀 값·접속 주소는 적지 않는다.
- 단계(자리표시자):
  1. 발견한 쪽 → 연락 채널로 즉시 알림. 담당: _미정_
  2. 화면에 잘못된 내용이 보이면 → 즉시 내림(4.6). 담당: _팀원(미확인)_
  3. 코드·설정 판단이 필요하면 → Song. 응답 기대 시간: _미정_

## 7. 이상 알림 감시기 (2026-10-18, 초안)

`apps/observer/ops/ws_alert.py`(표준 라이브러리만)가 발행이 잘못되면 Song 폰으로 알린다. **읽기만 한다:** 출력 폴더의 `v1/health.json`·`v1/games/<KST 날짜>.json`·`v1/game/<경기>.json`, 상태 파일(`<출력 폴더>.state.json`), 운영 상태 파일(있으면), 프로필 갱신 `RESULTS.log`, `vm_stat`, 디스크 여유. 프로세스 목록을 보거나 신호를 보내지 않고, 발행기 폴더에 쓰지 않는다. 자기 기록은 `~/.pitcheezy/ws-alert/`(`alert_state.json`, 추가 전용 `alerts.jsonl`)에만 남긴다. 기준값은 `apps/observer/ops/ws_alert_config.json`에 있고 **모두 잠정**이다(3절에서 옮기거나 감시기용으로 고른 값, 실경기로 잰 적 없음).

### 7.1 실행
- 확인만: `python3 apps/observer/ops/ws_alert.py --once --dry-run --out <출력 폴더>` — 무엇이 걸렸고 무엇을 보낼지 출력만 하고, 보내지도 기록하지도 않는다.
- 한 번: `--once`. 계속: `--loop`(`loop_interval_s` 15초). **`--once`·`--loop`는 폰 알림 환경 변수(7.4)가 설정돼 있으면 폰으로도 보낸다.** 손으로 시험할 때는 `--dry-run`이나 `--notifiers fake --alert-dir <임시 폴더>`를 쓴다(`--dry-run`은 기록을 남기지 않으므로 유지 시간·`file_stuck`처럼 여러 번 봐야 하는 규칙은 `fake`로 반복 실행해 본다).
- 경로 우선순위: 출력 폴더는 `--out` > 운영 상태 파일의 `out_dir`(그날 실행 폴더) > `PITCHEEZY_PUBLISH_OUT_DIR`. 상태 파일은 운영 상태의 `state_path` > `PITCHEEZY_PUBLISH_STATE` > `<출력 폴더>.state.json`. **상태 파일 이름은 발행기가 실제 쓰는 것과 같아야 한다**(리허설 폴더는 `publisher.state.json`이므로 운영 상태 파일이 없으면 `PITCHEEZY_PUBLISH_STATE`로 꼭 준다). 운영 상태 파일 기본 위치 `~/Projects/pitcheezy-pm-jobs/ws-ops/status.json`(`PITCHEEZY_WS_OPS_ROOT`, `PITCHEEZY_WS_OPS_STATUS`, `--ops-status`로 바꿈, 없으면 읽지 않음). 기록 폴더 `--alert-dir`/`PITCHEEZY_ALERT_DIR`.
- 알림 시험: `--test-notify`는 맥 알림 + `alerts.jsonl`만(로컬). `--test-notify --push`가 아래 폰 알림까지 보낸다.
- 띄우는 곳: 리허설에서 tmux `runs` 창에 `--loop`로 띄운다(launchd 등록은 아직 안 함, 정하면 여기 적는다). 감시기를 끄는 것은 그 창의 Ctrl-C.

### 7.2 규칙
"경기 시간대"는 시작 30분 전부터 경기가 final이 될 때까지(최대 시작+6시간). 경기 목록은 발행기의 `games/<어제·오늘·내일>.json`, 운영 상태 파일(그날 것만), 설정 `schedule`(발행기가 아직 목록을 못 만든 경우의 수동 보조)에서 모은다. 연기(`postponed`, 서스펜디드 포함)·final·other 경기는 시간대를 열지 않는다. 시간대가 열리면 "감시 시작: 경기 <번호>", 닫히면 "감시 끝" 알림을 한 번씩 보낸다 — **경기 날 "감시 시작"이 안 오면 감시기가 꺼진 것이다.** **경기 시간대가 아니면 경기 관련 규칙은 평가하지 않는다(조용함).**

| 키 | 언제 | 등급 | 경기 시간대만 |
|---|---|---|---|
| `health_stalled` | `published_at`이 30초 넘게 안 바뀜 | 경고 | 예 |
| `publisher_down` | health 없음/읽기 실패, 120초 넘게 멈춤, 또는 운영 상태가 `publisher.running:false`·`state` backoff/failed/refused/stopped | 긴급 | 예 |
| `game_check_stalled:<경기>` | 진행 중 경기 `checked_at` 30초 넘게 멈춤, 또는 `game_checks`에 경기 없음 60초 | 경고 | 예 |
| `feed_down:<경기>` | `feed_ok:false` 또는 `feed_failing_games`가 60초 넘게 계속 | 긴급 | 예 |
| `feed_delayed:<경기>` | 경기 파일 `display_state.feed.delayed`(발행기 판단: lag > max(45초, 시청 지연))가 20초 넘게 | 경고 | 예 |
| `file_stuck:<경기>` | 경기가 실제 진행 중(`status_text`가 `In Progress`)이고 확인·피드는 정상인데 `file_sequence`가 300초 넘게 그대로, 또는 운영 상태의 `upstream_changed_at`이 그보다 새롭고 60초가 지남. Warmup·Delayed·Suspended·비디오 판독 중에는 보지 않고, 그때마다 시계를 0으로 돌린다. `status_text`가 없을 때 운영 상태 `games[].status: live`만으로는 보지 않는다(실행기의 live는 예정 시각이 지났다는 시계 판단이지 경기 진행 확인이 아님) | 경고 | 예 |
| `game_unavailable:<경기>` | `unavailable_games`에 그 경기 코드가 30초 넘게(`upload_failed`는 제외 — 업로드 규칙이 본다) | 경고 | 예 |
| `upload_pending` | `upload.pending`이 5건 넘고 4주기 연속 줄지 않음. 백그라운드 업로드에서는 매 주기 2~3건 대기와 `upload.code='upload_failed'`가 **정상**이라 `upload.code`는 쓰지 않는다 | 경고 | 예 |
| `upload_stalled` | 상태 파일의 대기 중 전송(blocked·rejected 제외) 가운데 가장 오래된 것(revision)이 60초 넘게 그대로 | 경고 | 예 |
| `upload_rejected` | `upload_attempts`에 `rejected`(영구 4xx) 하나라도 | 긴급 | 아니오 |
| `upload_conflict` | 상태 파일 `upload_attempts`에 `blocked` 사유 `upload_conflict`(HTTP 412) | 긴급 | 아니오 |
| `upload_unknown` | `blocked` 사유 `upload_unknown`·`interrupted_unknown`·`legacy_unknown` | 긴급 | 아니오 |
| `upload_blocked` | 그 밖의 `blocked` 사유, `upload_recovery_blocked` | 긴급 | 아니오 |
| `state_unreadable` | 상태 파일이 있는데 30초 넘게 못 읽음 | 경고 | 아니오 |
| `memory_low` | `vm_stat` free+inactive+speculative < 2 GiB가 60초 넘게(메모리 압박 색과 같은 값은 아님) | 경고 | 예 |
| `disk_low` | 홈·출력 폴더 상위의 여유 < 12.5 GiB | 경고 | 아니오 |
| `profile_refresh:<날짜>` | 21:20 KST 이후 `RESULTS.log` 마지막 줄이 `ok target=<오늘> cutoff=<어제>`가 아님 | 경고 | 아니오 |
| `ops_state:<상태>` | 운영 감독 `state`가 backoff·refused·failed(긴급), stopped+`hard_end`(경고) | 긴급/경고 | 아니오 |
| `ops_problem:<키>` | 운영 상태 `problems`(severity `urgent`→긴급), 문자열 `alerts` 중 감시기가 직접 보지 않는 것(`protected_process_changed`는 긴급). 감시기가 같은 근거로 보는 것(`memory_low`, `health_stale`, `feed_failing`, `upload_blocked`)은 되풀이하지 않는다. 파일이 깨졌으면 `ops_problem:unreadable` | 파일이 준 등급 | 아니오 |
| `ops_status_stale` | 운영 상태 heartbeat(`updated_at`, 없으면 `written_epoch`)가 running/backoff에서 90초, 그 밖(waiting 등)에서 300초 넘게 지남. stopped/refused/failed와 다른 날짜(`kst_date`) 파일은 보지 않음 | 경고 | 아니오 |
| `watch_config` | 출력 폴더가 지정되지 않음 | 경고 | 아니오 |

알림 방식:
- **한 번만:** 조건이 시작되면(유지 시간을 넘긴 뒤) 한 번 알린다. 계속되면 긴급 15분·경고 30분마다 "[계속]"(경기 시간대 밖이면 6시간마다).
- **복구:** 알렸던 조건이 풀리면 "[복구]" 한 번.
- **쿨다운:** 풀린 뒤 5분 안에 같은 키가 다시 걸리면 바로 보내지 않고(`suppressed`로 기록만), 5분이 지나도 계속이면 보낸다.
- **조용히 닫기:** 경기가 끝나 더 평가하지 않는 키는 복구 알림 없이 `closed`로 기록만 한다.
- 발행기가 `--dry-run-upload`이면(`health.upload.dry_run: true`) 업로드 규칙을 모두 건너뛴다(그 모드는 pending이 줄지 않는다).
- 한 번 돌 때 생긴 알림은 **한 메시지로 묶는다**(제목 예: `Pitcheezy 긴급 2건 09:30 KST`). 폰 메시지에는 경로·주소·토큰을 넣지 않는다(경기 번호와 수치만).

### 7.3 운영 상태 파일 연결
운영 실행기(`ws_operate.py`)가 쓰는 `pitcheezy-ws-ops-status-v1`(계약: `docs/pm/service/ws-ops-status-contract.md`, H 작업)을 읽는다. 모르는 키는 무시하고, 형식이 틀린 항목은 건너뛴다. 감시기가 쓰는 필드:
- `kst_date`: 오늘이나 어제가 아니면 파일 전체를 무시한다.
- `updated_at`(v1.1) 또는 `written_epoch`/`written_at`: heartbeat. `state`: 위 `ops_state`·`publisher_down`·`ops_status_stale`.
- `publisher.running`(v1.1), `state_reason`, `guard`, `restarts.last_exit.why`: 알림 문구.
- `games[]`: `game_pk`, `start`(v1.1) 또는 `start_utc`, `status`(postponed 제외), `window_start`/`window_end`, `final_seen_at`(있으면 final), `upstream_changed_at`(있으면 `file_stuck`을 빨리 판단).
- `problems[]`(v1.1: `key`, `severity` urgent/warn, `message`, `active`)와 `alerts[]`(문자열).
- `out_dir`, `state_path`(v1.1): 그날 발행기 출력·상태 파일. 있으면 이것을 따른다(`--out`이 우선).
- 아직 없는 것(계약 쪽 요청): 경기별 `upstream_changed_at`(수집기가 본 마지막 경기 변화 시각)이 채워져야 `file_stuck`이 300초보다 빨리 잡는다.

### 7.4 폰 알림 고르기와 설정(기본 꺼짐)
맥 알림 + `alerts.jsonl`은 항상 켜져 있다(맥 앞에 없으면 못 본다). 폰 알림은 환경 변수로만 켠다. 저장소에는 비밀 값이 없다.

| 방식 | 장점 | 단점 |
|---|---|---|
| **ntfy**(구현, 추천) | 폰 앱 하나, 계정 불필요, 긴급은 우선순위 5로 소리 남. 맥은 HTTPS POST만 함 | 공용 서버 `ntfy.sh`에서는 **주제 이름이 곧 비밀번호**(추측되면 남이 읽음) → 길고 무작위로. 내용이 외부 서버를 지난다(경기 번호·수치만 보냄) |
| **iMessage**(구현) | 외부 서비스 없음, 폰 기본 앱 | 맥미니 Messages가 로그인돼 있어야 하고 첫 실행 때 자동화 허용 창이 뜬다(GUI 세션 필요). **맥미니와 폰이 같은 Apple ID면 "나에게 보낸 메시지"라 알림이 안 울릴 수 있다** → 다른 Apple ID에서 보내는 것이 확실 |
| 메일(`mail`, 미구현) | 설정 파일 없음 | 맥의 로컬 메일 발송이 설정돼 있지 않으면 안 가거나 스팸함. 실시간 알림에 부적합 |
| 텔레그램·Pushover(미구현) | 안정적 푸시 | 봇 토큰·계정이 필요(환경 변수로는 가능). 지금 단계에서는 ntfy로 충분 |

Song이 할 일(ntfy):
1. 폰에 ntfy 앱 설치 → 주제 구독. 주제 이름은 길고 무작위로(예: 맥에서 `python3 -c "import secrets;print('pz-'+secrets.token_urlsafe(18))"`). 이 이름을 이슈·문서에 적지 않는다.
2. 감시기를 띄우는 셸에서 `export PITCHEEZY_ALERT_PUSH=ntfy PITCHEEZY_ALERT_NTFY_TOPIC=<주제>`(직접 운영하는 ntfy 서버면 `PITCHEEZY_ALERT_NTFY_SERVER`, 접근 토큰은 파일로 `PITCHEEZY_ALERT_NTFY_TOKEN_FILE`).
3. `python3 apps/observer/ops/ws_alert.py --test-notify --push`로 폰에 "알림 시험"이 오는지 본다.

Song이 할 일(iMessage, 쓰려면):
1. 맥미니 Messages에 iMessage 로그인(가능하면 폰과 다른 Apple ID).
2. `export PITCHEEZY_ALERT_PUSH=imessage PITCHEEZY_ALERT_IMESSAGE_TO=<폰 번호 또는 Apple ID 메일>`(둘 다 쓰려면 `ntfy,imessage`).
3. 맥 앞에서 `--test-notify --push`를 한 번 돌려 자동화 허용 창에 "허용" → 폰 수신 확인.

맥 알림을 끄려면 `PITCHEEZY_ALERT_MACOS=0`. 알림 경로가 실패하면 감시기는 멈추지 않고 `alerts.jsonl`에 `notify_error`로 남긴다.

### 7.5 아직 안 정한 것
- 기준값 전부(리허설 E에서 실제 주기·흔들림·피드 지연을 재서 고친다). 특히 `file_stuck` 300초는 이닝 교대·투수 교체로 거짓 경고가 날 수 있다.
- 감시기 자체가 죽은 것은 "감시 시작" 알림이 없는 것으로만 안다(그 밖에는 `alert_state.json`의 `last_run`). launchd KeepAlive나 운영 감독 아래에서 띄울지 함께 정한다.
- ~~운영 상태 v1.1 필드가 실제로 들어오면 가짜 감독 시험으로 한 번 맞춰 본다.~~ 2026-10-18 통합 시험 `apps/observer/ops/tests/test_ws_ops_alert_integration.py`(실제 감독 + 가짜 발행기 + 감시기 `--once`)로 맞췄다. 실경기·리허설에서 같이 돌려 본 적은 아직 없다.
- 발행기가 죽으면 `publisher_down`·`health_stalled`·`game_check_stalled`가 함께 뜬다(한 메시지로 묶임).

### 7.6 운영 실행기와 함께 띄우기 (2026-10-18 통합, 실경기 미검증)
실행기(`ws_operate.py`)가 발행기를 띄우고 `<ops_root>/status.json`을 쓰며, 감시기는 그 파일 하나로 그날 출력 폴더(`out_dir`)·상태 파일(`state_path`)·경기 시간대(`games[].window_start/window_end`)를 찾는다. 두 프로그램은 서로 프로세스를 보거나 신호를 보내지 않는다.

1. 실행기(경기 날, 첫 투구 30분 전보다 일찍): `apps/observer/README.md` "월드시리즈 운영 실행기"대로 `python3 apps/observer/ops/ws_operate.py start --date <KST 날짜>`(먼저 `--dry-run`). 분리된 감독이 뜨고 대기(`waiting`) 중에도 60초마다 `status.json`을 쓴다.
2. 감시기(tmux `runs` 창 등 별도 셸, 실행기보다 먼저 띄워도 됨): `python3 apps/observer/ops/ws_alert.py --loop`. **`--out`을 주지 않는다**(주면 운영 상태의 `out_dir`보다 우선해 그날 폴더를 따라가지 않는다). 처음에는 `--once --dry-run`으로 무엇이 보일지 확인.
3. 환경 변수:
   - `PITCHEEZY_WS_OPS_ROOT`: 기본(`~/Projects/pitcheezy-pm-jobs/ws-ops`)을 바꿨다면 **두 셸에 같은 값**. 감시기는 `<root>/status.json`을 읽는다(`PITCHEEZY_WS_OPS_STATUS`·`--ops-status`가 더 우선).
   - 실행기 쪽만: `PITCHEEZY_OBSERVER_PROFILE_SOURCES`(참고 카드), 업로드를 켤 때 `PITCHEEZY_PUBLISH_TOKEN` 또는 `PITCHEEZY_WS_OPS_TOKEN_FILE`, `PITCHEEZY_WS_OPS_UPLOAD_BASE_URL`.
   - 감시기 쪽만: 폰 알림 `PITCHEEZY_ALERT_PUSH`와 그 방식의 값(7.4), 기록 폴더 `PITCHEEZY_ALERT_DIR`(기본 `~/.pitcheezy/ws-alert`), 프로필 로그 `PITCHEEZY_PROFILE_REFRESH_LOG`. `PITCHEEZY_PUBLISH_OUT_DIR`·`PITCHEEZY_PUBLISH_STATE`는 실행기와 함께 쓸 때 필요 없다(운영 상태가 그날 것이면 그쪽을 따른다).
4. 무엇이 오나(통합 시험에서 확인한 순서): 정상 운영은 "감시 시작: 경기 <번호>" 한 번만. 발행기가 죽어 재시작 대기(`backoff`)면 긴급 `ops_state:backoff`+`publisher_down`(+경고 `ops_problem:publisher_restarting`), 재시작 한도를 넘으면 긴급 `ops_state:failed`+`ops_problem:restart_limit`, 시작 점검 재시도 중이면 경고 `ops_problem:prestart_retrying`(시간대 안이면 긴급 `publisher_down`도), 시작 거부면 긴급 `ops_state:refused`+`ops_problem:prestart_refused`, 업로드 막힘은 감시기 자신의 긴급 `upload_blocked` 한 줄(실행기의 같은 문제는 되풀이하지 않음). 모든 경기가 final이고 감독이 `final_settled`로 끝나면 조용히 "감시 끝" 한 번.
5. 겹치는 판단: 실행기 `problems` 가운데 `memory_low`·`health_stale`·`feed_failing`·`upload_blocked`는 감시기가 같은 근거(health·상태 파일·vm_stat)로 직접 보므로 넘기지 않는다(`ws_alert.OPS_ALERTS_COVERED`, 통합 시험이 실행기 `PROBLEM_SEVERITY`와 맞는지 확인). 나머지 키(긴급: `prestart_refused`, `restart_limit`, `deploy_guard`, `resource_guard`, `another_publisher`, `supervisor_error` / 경고: `prestart_retrying`, `publisher_restarting`, `protected_process_changed`, `schedule_fallback`, `identity_unreadable`)는 `ops_problem:<키>`로 실행기가 준 등급 그대로 보낸다. 끝난 상태(refused/failed)에서는 `ops_state`와 해당 `ops_problem`이 같은 메시지에 함께 묶여 온다.
6. 오래됨: 실행기 heartbeat는 운영 중 10초·대기 60초, 감시기의 "감독 멈춤"(`ops_status_stale`) 기준은 running/backoff 90초·그 밖(waiting·prestart·stopping) 300초. prestart·stopping도 실행기는 10초마다 쓰지만 감시기는 짧은 상태라 300초로 넉넉히 본다(잠정, 리허설에서 다시 본다).
7. 끄기: 실행기는 `ws_operate.py stop --date <날짜>`, 감시기는 그 창의 Ctrl-C. 끄는 순서는 상관없다(감시기가 먼저 꺼지면 알림만 없다).

모르는 것(아직 정하지 않음): 감시기를 launchd KeepAlive로 띄울지·실행기 감독 아래 둘지, 폰 알림 방식(ntfy 주제 미설정), 모든 기준값(잠정), 실경기·리허설에서 둘을 같이 돌린 기록 없음, 운영 상태의 `upstream_changed_at`은 실행기가 채우지 않음(`file_stuck`은 300초 기준만).

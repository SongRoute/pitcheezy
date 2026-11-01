# 2026-11-24 전시 런북

상태: 실행 절차 초안. 이 문서 작성으로 배포·설정 변경·실자료 재생을 실행하지 않았다. 기준 코드 main `d04afab1`, 고정 빌드 `fba6b2ff`. 근거는 [10/31 빌드 기록](exhibition-build-2026-10-31.md) §2·§6과 Song이 전달한 10/31 인계 §2·§5다. 기능 동결 11/20, 리허설 11/21~23, 전시 11/24. 11/11 저녁 공동 연결시험은 **제안 일정이며 확정 전**이다.

## 1. 화면·역할·정직 표기

| 자리 | 보여 줄 것 | 담당·확인 |
|---|---|---|
| 모니터 1 | 849843 CHC@SD, 발행기 재생 → v1 → UI-1026 공개 빌드. 추천과 미지원 구간 참고 카드 | Song: 서비스·모델·발행·결정 |
| 모니터 2 | 같은 경기 풀 중계 우선. 확보되지 않으면 MLB 공식 클립 체인 | 현준: 영상·클립·호스트, 사용 가능한 소스 확정 전 |
| 관람객 폰 | QR → pitcheezy.com의 팀원 Cloudflare 호스트에 올린 고정 빌드 | 현준: URL·경로·호스팅 / Song: 전달 빌드 / 강민: QR·발표물 |
| 발표물 | UI-1026 캡처·근거가 붙은 수치·설명 | 강민. 포스터 구조는 유지하고 동결 뒤 한 번에 정정 |

이것은 **완료 경기의 재생 시연**이다. LIVE 표시나 재생 시계는 실제 경기 생중계·실수신 시각을 보증하지 않는다. 전시 G0는 동결하며 연구 후보와 교체하지 않는다. 참고 카드는 추천이 아니다. 모델 성능·승리 개선을 이 시연의 성공 횟수로 주장하지 않는다. 폰의 고정 최종 경기 파일은 모니터 1의 움직이는 시점과 자동 동기화되지 않는다.

## 2. 고정 자산과 실행 전 확인

- 경기 849843: 현지 2026-09-29, KST 9/30. 빌드 기록상 262구, ready 246구, 미지원 16구. 이는 **최종 archive 파일의 집계**다.
- 맥미니 작업 폴더: `~/Projects/pitcheezy-pm-jobs/exhibition-849843-1031/` (아래 `task_assets`). 원본 `out4/v1`, `profiles/LATEST.json`, ZIP, 기존 stamped 재생 폴더를 덮어쓰지 않는다.
- 원본 ZIP: `bundle-849843-fba6b2ff.zip`, 867,682 bytes, SHA-256 `af10293d5134219b7f750d00786366b93b0134c895519b939fd99a356d77a4bc`. 새 코드로 만든 묶음에는 새 커밋·파일 목록·SHA를 기록한다. 이 SHA를 재사용하지 않는다.
- T7 읽기 경로: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/{feeds,watch,intent}`. 연결 여부와 `849843_final.json.gz` 존재를 확인한다. 누락 시 수집·fit을 시작하지 않고 고정 묶음으로 전환한다.
- 참고 카드 포인터: `profiles/LATEST.json`, 대상 9/29·컷오프 9/28, profile SHA `f64abb8511fa071ebcd3561e7634a76ead8fbaf8e3db4cda2f5f78178016b7f3`. 최신 날짜 프로필로 바꾸지 않는다.
- 실행할 worktree의 커밋·변경 여부, 사용 Python/Node, 시작 투구 순번, 출력 폴더, store 경로, 재생 poll/발행 interval을 회차 기록에 적는다. 다른 작업자의 기록기·GPU·운영 포트·프로세스는 건드리지 않는다.

## 3. 모니터 1 시작

### 3.1 회차 준비와 재생 파일 생성

아래는 맥미니에서 운영자가 실행할 명령이다. `cd`의 경로는 검토 완료한 전시 worktree로 채운다. 이 작업의 타임스탬프 수정이 포함된 커밋인지 먼저 확인한다. 기존 `d04afab1`만 쓰면 3.2의 구버전 경로가 필요하다. 설치·전역 설정 변경 없이 기존 환경을 사용한다.

```sh
cd /검토완료한/전시-worktree
task_repo="$PWD"
task_python="$HOME/Projects/pitcheezy/.venv-observer/bin/python"
task_assets="$HOME/Projects/pitcheezy-pm-jobs/exhibition-849843-1031"
task_run=$(mktemp -d "$task_assets/rehearsal-XXXXXX")
task_replay="$task_run/replay"
mkdir -p "$task_run/empty/feed" "$task_run/empty/demo" "$task_run/site"
git rev-parse HEAD
git status --short
(cd apps/observer/web && npm run build:public)
cp -R apps/observer/web/dist-public/. "$task_run/site/"
PYTHONPATH=src:.:apps/observer/backend "$task_python" -B scripts/demo_precompute.py live-replay \
  --game-pk 849843 --start 0 --count 262 --output "$task_replay"
```

기본 시작은 `--start 0`(첫 투구 전), 순번은 **0부터**다. 장면 점프는 새 폴더·새 store로 다시 생성한다. 기존 출력 폴더에 짧은 구간을 덮어쓰면 남은 뒷부분 파일까지 재생될 수 있다. Rolison 참고 카드는 빌드 기록의 5회말 138~145구 구간을 안내 표지로 쓰되, 실제 선수·이닝·타석을 확인해 점프 지점을 리허설 기록에 적는다. `--count`는 양수이며 범위 끝은 262구 이내로 정한다.

### 3.2 타임스탬프 호환성

수정 코드로 만든 파일은 별도 stamp 스크립트 없이 `metaData.timeStamp`가 단조 증가하는 `YYYYMMDD_HHMMSS`다. 공식 경기일+절대 순번으로 만든 **재생용 순서 표시**이며 실수신 시각이 아니다. 재생을 처음으로 되돌리면 반드시 새 store를 사용한다. 과거 회차 store의 더 큰 도장을 유지한 채 되감으면 stale 판정이 정상이다.

구버전 `d04afab1`을 재현할 때만 빌드 §6의 우회 절차를 쓴다. 원본 재생 파일은 보존한다.

```sh
"$task_python" "$task_assets/stamp_snapshots.py" "$task_replay" "$task_run/replay-stamped" 25
task_replay="$task_run/replay-stamped"
```

이 명령의 `25`는 도장 간격이고 화면 속도가 아니다. 수정 코드와 구버전 stamp 우회를 한 회차에서 혼용하지 않는다.

### 3.3 발행기 실행

빌드 §2의 archive 환경을 §6의 재생 환경으로 바꾼 명령이다. `FEED_DIR`는 빈 폴더, `LIVE_REPLAY_DIR`는 위 재생 입력이며 `--once`를 붙이지 않는다. 출력은 이 회차의 `site/v1`, store도 이 회차 전용이다. 업로드 옵션·토큰은 넣지 않는다.

```sh
env -i HOME="$HOME" USER="$USER" LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8 \
  TMPDIR="${TMPDIR:-/tmp}" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=apps/observer/backend PYTHONUNBUFFERED=1 \
  PITCHEEZY_OBSERVER_LIVE_STORE="$task_run/store.sqlite3" \
  PITCHEEZY_OBSERVER_LIVE_ARMB=1 PITCHEEZY_OBSERVER_LIVE_DELAY_S=0 \
  PITCHEEZY_OBSERVER_REC_LEDGER=1 PITCHEEZY_OBSERVER_LEGACY=0 \
  PITCHEEZY_OBSERVER_VIEWER_DELAY=0 PITCHEEZY_OBSERVER_STEP5_EXPECTATION=0 \
  PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR="$task_replay" PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S=20 \
  PITCHEEZY_FEED_DIR="$task_run/empty/feed" \
  PITCHEEZY_WATCH_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/watch" \
  PITCHEEZY_INTENT_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/intent" \
  PITCHEEZY_DEMO_DIR="$task_run/empty/demo" \
  PITCHEEZY_RUNS_DIR="$HOME/Projects/pitcheezy/runs" PITCHEEZY_DATA_DIR="$HOME/Projects/pitcheezy/data" \
  PITCHEEZY_PLAYER_SNAPSHOT_DIR="$HOME/Projects/pitcheezy-pm-jobs/20261020-ws-pregame-cache-001/cache" \
  PITCHEEZY_OBSERVER_PROFILE_SOURCES="$task_assets/profiles/LATEST.json" \
  "$task_python" -B -m observer_app.publisher --out "$task_run/site" \
  --date 2026-09-30 --interval 10 --config apps/observer/publish_config.json \
  > "$task_run/publisher.log" 2>&1
```

이 터미널은 실행 중인 상태로 둔다.

별도 터미널에서 `task_run`에 위에서 생성한 절대 경로를 설정하고 빈 로컬 포트를 고른다. 아래 8876은 예시이며 사용 중이면 다른 빈 번호로 바꾼다. 다른 프로세스를 중지하지 않는다.

```sh
lsof -nP -iTCP:8876 -sTCP:LISTEN
/Users/song/Projects/pitcheezy/.venv-observer/bin/python -m http.server 8876 \
  --bind 127.0.0.1 --directory "$task_run/site"
```

모니터 1에서 `http://127.0.0.1:8876/?game=849843`을 연다.

### 3.4 화면 확인·속도·종료

- `site/v1/game/849843.json`의 `source.kind=live`, `next.pitch_key` 진행, `display_state.state/reason`을 확인한다. 지원 구간은 `recommended/ledger_recorded`, 미지원 구간은 추천 없이 `next.reference`를 확인한다. `unorderable_provider_stamp`이면 정상 출발로 간주하지 않는다.
- `LIVE_REPLAY_POLL_S=20`, 발행 `--interval 10`은 빌드 §6의 재현값이다. 그 기록의 한 구 약 10초는 과거 관찰이며 새 회차 속도 보장이 아니다. 실제 화면 전환 시각과 투구 순번을 기록해 속도·영상 오프셋을 맞춘다. `/live?...&speed=`는 다른 경로이며 참고 카드가 없으므로 본 전시 경로 대체로 쓰지 않는다.
- live-replay는 각 투구 **전** 스냅샷 목록이다. ReplayTransport는 마지막 파일을 반복하며 자동으로 경기 종료 화면이나 처음으로 돌아가지 않는다. 대본 종료 시 중지하거나 archive 최종 경기 화면으로 전환한다.
- 중지는 자신이 시작한 발행기·정적 서버 터미널에서만 Ctrl-C. 운영 기록기나 다른 터미널에 신호를 보내지 않는다. 다음 회차는 새 폴더·새 store로 시작한다.

## 4. 모니터 2와 폰 QR

2026-10-31 Song 확인: 확보한 경기 영상/VOD·공식 클립 URL과 팀원 호스트의 최종 관전 페이지 경로는 모두 **미확정**이다. 아래 예시 URL로 QR을 확정하거나 영상·폰 연결 검증을 완료 처리하지 않는다.

1. 영상 경기·팀·날짜를 849843과 대조하고 확보한 소스, 재생 위치, 중단·재시작 방법을 기록한다. 풀 중계 가능 여부는 미확정이다. 안 되면 현준이 준비한 공식 클립 체인을 사용하고 연속 경기와 같은 것으로 설명하지 않는다.
2. 모니터 1을 첫 추천 장면에서 확인하고 영상도 같은 투수·타자·이닝·카운트에 맞춘다. 추천이 해당 투구보다 먼저 보이는지 관찰한다. 자동 동기화는 없으며 **오프셋 초 값은 미측정**이다. 두 운영자가 리허설에서 측정해 회차표에 적는다.
3. 영상이 광고·버퍼링·클립 경계로 어긋나면 임의의 offset을 추측하지 않는다. 양쪽을 대본 시작점으로 맞추며, 재생기 되감기는 새 회차로 한다. 클립 체인에서는 장면 단위 설명으로 전환한다.
4. QR URL은 팀원 호스트의 최종 배치 경로를 확인한 후 만든다. 루트에 직접 공개 빌드가 있으면 `https://pitcheezy.com/?game=849843`, 홈 아래라면 **검증한 실제 경로**를 쓴다. 현재 경로·배포·QR 실기기 스캔은 미측정이다.
5. 폰의 정적 최종 파일에는 `next`가 없어 참고 카드가 나오지 않는다. 모니터 1에서 해당 장면을 설명한다. 폰에서 과거 투구 선택·기록·구종 시트·닫기·뒤로가기, Wi-Fi/셀룰러 접속과 iPhone Safari를 확인한다.

## 5. 장애 대응

| 증상 | 확인·복구 | 계속 못 하면 |
|---|---|---|
| `unorderable_provider_stamp` | 코드 커밋·재생 파일 도장 형식 확인. 수정 코드 또는 3.2 구버전 경로 중 하나로 새 입력·새 store 준비 | 원본 고정 archive 빌드로 전환. 순서 게이트를 끄지 않음 |
| stale/같은 도장 충돌·되감기 실패 | 입력 폴더와 store 회차가 맞는지 확인. 같은 store에 재시작하지 않음 | 새 회차 준비 전까지 archive 화면 |
| 갱신 지연/연결 끊김/확인 중 | 자기 발행기 로그·출력 갱신·health·game 파일 확인. UI 상태와 파일의 사유를 함께 기록 | 고정 archive로 전환하고 재생 중단을 설명 |
| Rolison 구간 추천 없음 | 미지원 16구 중 해당 구간인지, 참고 카드 컷오프 9/28과 출처 확인 | 지원 구간 추천을 끌어와 채우지 않고 미지원 그대로 설명 |
| 참고 카드 없음 | 최종 archive 파일인지 먼저 확인. 재생 경로의 프로필 포인터·다음 타석 확인 | 준비된 장면으로 다시 시작하거나 미표시 사실 설명 |
| T7·모델 환경·프로필 누락 | 경로만 확인. 현장 학습·다운로드·운영 교체 없음 | 검증된 ZIP의 로컬 정적 화면과 준비된 영상 |
| 호스트 장애·긴급 안내 | 폰에 안내가 뜨면 추천·존·목록이 숨는지 확인. 안내 뒤 일반 오류에서도 과거 추천이 돌아오면 시연 보류 | 모니터 로컬 화면으로 설명하고 폰 체험 중단 |
| 영상 확보 실패·동기화 이탈 | 공식 클립 대본으로 전환, 장면 단위 설명 | 영상 없이 관전 화면 조작 설명 |
| 로컬 포트 사용 중 | 빈 포트를 골라 URL 수정 | 기존 점유 프로세스 종료 금지 |

원본 ZIP fallback은 **새 폴더**에 풀고 `bundle-849843-fba6b2ff/site`를 위 로컬 서버로 제공한다. 이것은 최종 보관 경기 탐색이며 움직이는 live replay라고 설명하지 않는다. 호스트 배포·DNS·긴급 안내 설정 변경은 담당자의 결정 절차를 따른다.

## 6. 리허설 체크리스트·회차 기록

- [ ] 11/11 연결시험 일정 확정, 영상 소스/클립 대본·QR 최종 경로 담당자 확인.
- [ ] 11/20: 커밋·번들 SHA·프로필 SHA·도구 버전 동결. 모델은 G0, 새 연구 결과는 분리.
- [ ] API/public 빌드와 관련 스모크 결과를 실제 실행 로그와 함께 기록. DOM 포팅 이후 잔여 실패를 PASS로 적지 않음. 시리즈 헤더 검사는 미포팅 기능 skip임을 명시.
- [ ] 긴급 안내 ON→추천·존·타석 숨김, 안내→일반 오류에서 계속 숨김, 정상 응답 뒤 자동 복구. health는 정상이고 game만 안내인 호스트 조건도 확인(초기 안내 탐지 보장 미확인).
- [ ] 모니터 1 첫 추천·타석 변화·Rolison 참고 카드·마지막 스냅샷 정지·새 store 재시작 확인.
- [ ] 모니터 2 동일 장면·선행 추천·측정한 오프셋·클립 fallback 확인.
- [ ] iPhone Safari/실제 QR, 좁은 화면·가로 넘침·스크롤·상태 팝업·시트 닫기 확인.
- [ ] network/호스트 장애와 로컬 fallback 전환을 실제로 연습. 복구 소요시간은 측정 후 기입.
- [ ] 11/21~23: 두 모니터+폰으로 대본 전체 리허설. 11/24 개장 전 동일 회차 점검.

회차 기록 항목: 날짜·운영자 / 코드 커밋·dirty 여부 / 번들 SHA / 실행 경로·store·입력 / 시작 순번·count / poll·interval / 영상 출처·시작 장면·오프셋 / 폰 URL·기기·브라우저 / 관찰 시각·투구 전환 / 오류와 fallback 시간 / PASS·FAIL·미측정.

빌드 §6의 **300초 관찰 recommended 69회·참고 카드 27회**는 기존 회차의 상태 관찰 횟수다. 262개의 서로 다른 투구 수나 새 리허설 결과로 합산하지 않는다. 이 런북의 명령 실행, 변경 코드 전체 262구 재생, 영상 오프셋, 실제 호스트 배포, 실기기 검수·복구시간은 작성 시점 **미측정**이다.

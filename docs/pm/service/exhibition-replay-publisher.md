# 전시 발행기 실행

상위: [전시 런북](exhibition-runbook-2026-11-24.md) · 근거: [10/31 빌드](exhibition-build-2026-10-31.md)

선행: [입력 준비](exhibition-replay-setup.md)의 셸 변수·새 입력·새 store.

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


다음: [화면·영상·폰 확인](exhibition-displays.md).

#!/bin/sh
# Pitcheezy 관전 서비스 한 번에 시작: 웹 화면 빌드(바뀐 경우만) + 백엔드(API·화면·지연 라이브 추천).
#   sh apps/observer/serve.sh                                   # 이 맥에서만 (127.0.0.1:8766)
#   PITCHEEZY_OBSERVER_HOST=tailscale sh apps/observer/serve.sh # 같은 tailnet 기기(폰)에서만
# 설정(환경 변수): PITCHEEZY_OBSERVER_HOST (127.0.0.1 | tailscale | 100.x.y.z), PITCHEEZY_OBSERVER_PORT (8766),
#   PITCHEEZY_OBSERVER_PYTHON, PITCHEEZY_WATCH_DIR, PITCHEEZY_OBSERVER_LIVE_DELAY_S,
#   PITCHEEZY_OBSERVER_LIVE_ARMB (0 = 라이브 추천 끄기), PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR (녹화 재생),
#   PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S (녹화 재생 때만: 본문 하나를 내주는 간격(초). 리허설 가속용, 0 = 요청마다 한 개),
#   PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE (recorded = 녹화 재생 때만: 본문을 캡처 수신 간격 그대로, 사건 시각을 오늘 시계로 옮겨
#   내준다. 시청 지연 리허설용. capture_to_replay.py 폴더의 MANIFEST.json 필요; _MAX_GAP_S = 긴 공백을 이 초로 줄임),
#   PITCHEEZY_OBSERVER_LEGACY (1 = 예전 과거 기록 관전 화면도 켜기; T7에 DB를 씀).
#   PITCHEEZY_OBSERVER_LIVE_STORE (SQLite 경로 = 수신 스냅샷·투구 원장·추천 원장을 append-only로 저장; 켜면 추천은
#   원장에 기록된 뒤에만 표시), PITCHEEZY_OBSERVER_REC_LEDGER (0 = 추천 원장만 끄기).
# 기능 스위치(잠정값. 웹을 다시 빌드하지 않고 끌 수 있다. 기본값은 live_config.json "features", 설명은 observer_app/features.py):
#   모두 기본 꺼짐(공개 서비스는 지연 0 본문 하나를 모두에게 배포한다. 시청자별 지연은 이 서버에 직접 붙는 폰 시연·내부용).
#   PITCHEEZY_OBSERVER_VIEWER_DELAY (1 = 팬 화면 시청 지연·중계 맞추기 켜기; 꺼져 있으면 ?delay= 는 무시되고 지연 0 그대로),
#   PITCHEEZY_OBSERVER_VIEWER_DELAY_DEFAULT_S (맞추기 전 기본 지연 초, 기본 30),
#   PITCHEEZY_OBSERVER_VIEWER_SYNC_PROMPT (0 = 첫 화면에서 맞추기를 묻지 않음; 수동 진입은 남음),
#   PITCHEEZY_OBSERVER_PROFILE_REFERENCE (1 = 켜기), PITCHEEZY_OBSERVER_PROFILE_SOURCES (경로 여러 개는 ':'로 구분),
#   PITCHEEZY_OBSERVER_STEP5_EXPECTATION (0/1), PITCHEEZY_OBSERVER_STEP5_LEDGER (경로), PITCHEEZY_OBSERVER_STEP5_ALLOW_REPLAYED (0/1).
#   값이 잘못되면 설정 파일 값을 그대로 쓰고 경고를 한 번 남긴다(서비스는 멈추지 않는다).
# 공개 주소(0.0.0.0, 공인 IP, Funnel)로는 열지 않는다.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
APP="$ROOT/apps/observer"
PY="${PITCHEEZY_OBSERVER_PYTHON:-$ROOT/.venv-observer/bin/python}"
HOST="${PITCHEEZY_OBSERVER_HOST:-127.0.0.1}"
PORT="${PITCHEEZY_OBSERVER_PORT:-8766}"
TAILSCALE="${PITCHEEZY_TAILSCALE:-/usr/local/bin/tailscale}"

if [ ! -x "$PY" ]; then
  echo "Python 환경이 없습니다: $PY (apps/observer/README.md 참고)" >&2; exit 1
fi
if [ "$HOST" = tailscale ]; then
  HOST=$("$TAILSCALE" ip -4 2>/dev/null | head -1) || true
  if [ -z "$HOST" ]; then echo 'Tailscale IP를 찾지 못했습니다. Tailscale 로그인 상태를 확인하세요.' >&2; exit 1; fi
fi
case "$HOST" in
  127.0.0.1|localhost|::1|100.*) ;;
  *) echo "비공개 주소(127.0.0.1 또는 Tailscale 100.x)만 허용합니다: $HOST" >&2; exit 1 ;;
esac

WEB="$APP/web"
if [ ! -f "$WEB/dist/index.html" ] || [ -n "$(find "$WEB/src" "$WEB/index.html" "$WEB/package.json" -newer "$WEB/dist/index.html" 2>/dev/null | head -1)" ]; then
  echo '웹 화면을 빌드합니다…' >&2
  (cd "$WEB" && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && npm run build) >&2
fi

export PYTHONPATH="$APP/backend${PYTHONPATH:+:$PYTHONPATH}"
# The legacy replay app (/) writes its database on T7; this service runs the watch-along/live product only.
export PITCHEEZY_OBSERVER_LEGACY="${PITCHEEZY_OBSERVER_LEGACY:-0}"
export PYTHONUNBUFFERED=1
echo "Pitcheezy 관전 서비스: http://$HOST:$PORT/watch (상태: /api/health)" >&2
exec "$PY" -m uvicorn observer_app.main:app --host "$HOST" --port "$PORT" --log-config "$APP/logging.json"

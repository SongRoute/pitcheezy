#!/bin/sh
# Pitcheezy 관전 서비스 한 번에 시작: 웹 화면 빌드(바뀐 경우만) + 백엔드(API·화면·지연 라이브 추천).
#   sh apps/observer/serve.sh                                   # 이 맥에서만 (127.0.0.1:8766)
#   PITCHEEZY_OBSERVER_HOST=tailscale sh apps/observer/serve.sh # 같은 tailnet 기기(폰)에서만
# 설정(환경 변수): PITCHEEZY_OBSERVER_HOST (127.0.0.1 | tailscale | 100.x.y.z), PITCHEEZY_OBSERVER_PORT (8766),
#   PITCHEEZY_OBSERVER_PYTHON, PITCHEEZY_WATCH_DIR, PITCHEEZY_OBSERVER_LIVE_DELAY_S,
#   PITCHEEZY_OBSERVER_LIVE_ARMB (0 = 라이브 추천 끄기), PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR (녹화 재생),
#   PITCHEEZY_OBSERVER_LEGACY (1 = 예전 과거 기록 관전 화면도 켜기; T7에 DB를 씀).
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

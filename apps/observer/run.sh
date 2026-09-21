#!/bin/sh
set -eu
OBSERVER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OBSERVER_PYTHON="${PITCHEEZY_OBSERVER_PYTHON:-$OBSERVER_ROOT/.venv-observer/bin/python}"
if [ ! -x "$OBSERVER_PYTHON" ]; then
  echo '별도 환경 .venv-observer가 필요합니다. apps/observer/README.md를 확인하세요.' >&2
  exit 1
fi
if [ ! -f "$OBSERVER_ROOT/apps/observer/web/dist/index.html" ]; then
  echo '웹 빌드가 필요합니다: apps/observer/web 에서 npm ci && npm run build' >&2
  exit 1
fi
export PYTHONPATH="$OBSERVER_ROOT/apps/observer/backend${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$OBSERVER_PYTHON" -m uvicorn observer_app.main:app --host 127.0.0.1 --port "${PITCHEEZY_OBSERVER_PORT:-8766}"

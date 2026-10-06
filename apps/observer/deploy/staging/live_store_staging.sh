#!/bin/sh
# Isolated staging of the opt-in live store (observer_app/live_store.py) on this Mac, 127.0.0.1 only.
# Everything it writes lives under $PZ_STAGING_DIR (pid, log, db, identity, backups); it never touches the
# production services (8766/8767/8768/8770/8771), the recorder or its live_snapshots.
#
#   export PZ_STAGING_DIR=/some/task/owned/dir          # required
#   export PZ_STAGING_REPLAY_DIR=/dir/with/{gamePk}_*.json.gz   # optional: recorded feeds (else the real MLB feed)
#   sh apps/observer/deploy/staging/live_store_staging.sh start|stop|status|health|state GAME|backup|restore FILE|nostore
#
# PITCHEEZY_STAGING_PORT (8791). `nostore` starts the same build without the store (the rollback/default path).
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
APP="$ROOT/apps/observer"
PY="${PITCHEEZY_OBSERVER_PYTHON:-$ROOT/.venv-observer/bin/python}"
[ -x "$PY" ] || PY="$(git -C "$ROOT" worktree list --porcelain | sed -n '1s/^worktree //p')/.venv-observer/bin/python"
PORT="${PITCHEEZY_STAGING_PORT:-8791}"
DIR="${PZ_STAGING_DIR:?set PZ_STAGING_DIR to a task-owned directory}"
DB="$DIR/live.sqlite3"
PID="$DIR/server.pid"
LOG="$DIR/server.log"
case "$PORT" in 8766|8767|8768|8770|8771) echo "port $PORT belongs to an existing service" >&2; exit 2 ;; esac
mkdir -p "$DIR/backups"
export PYTHONPATH="$APP/backend" PYTHONUNBUFFERED=1

running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
owned() {
  [ -f "$DIR/server.identity" ] || return 1
  ps -p "$(cat "$PID")" -o lstart= > "$DIR/server.current" || return 1
  cmp -s "$DIR/server.identity" "$DIR/server.current" || return 1
  case "$(ps -p "$(cat "$PID")" -o command=)" in
    *" -m uvicorn observer_app.main:app --host 127.0.0.1 --port $PORT --log-config $APP/logging.json") return 0 ;;
    *) return 1 ;;
  esac
}

stop_owned() {
  if running && ! owned; then echo "refusing to stop unowned pid" >&2; return 1; fi
  if running; then
    kill "$(cat "$PID")"
    i=0
    while running; do
      i=$((i+1)); [ "$i" -lt 50 ] || { echo "stop timed out; ownership files retained" >&2; return 1; }
      sleep 0.2
    done
  fi
  rm -f "$PID" "$DIR/server.identity" "$DIR/server.current"
}

launch() {  # $1 = store|nostore
  if running; then echo "already running: pid $(cat "$PID")" >&2; exit 1; fi
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT is in use" >&2; exit 1; fi
  rm -f "$PID" "$DIR/server.identity" "$DIR/server.current"
  STORE_ENV=""
  [ "$1" = store ] && STORE_ENV="$DB"
  DELAY="${PITCHEEZY_OBSERVER_LIVE_DELAY_S:-${PZ_STAGING_REPLAY_DIR:+0}}"
  {
    printf '{"started":"%s","mode":"%s","commit":"%s","dirty":%s,"branch":"%s","python":"%s","port":%s,"db":"%s","replay_dir":"%s"}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$(git -C "$ROOT" rev-parse HEAD)" \
      "$([ -z "$(git -C "$ROOT" status --porcelain -- apps/observer/backend)" ] && echo false || echo true)" \
      "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" "$PY" "$PORT" "$STORE_ENV" "${PZ_STAGING_REPLAY_DIR:-}"
  } >> "$DIR/identity.jsonl"
  # Watch-along/legacy model stays off and ARM-B is not loaded: this staging is about the live state path only.
  env PITCHEEZY_OBSERVER_LEGACY=0 PITCHEEZY_OBSERVER_LIVE_ARMB=0 \
      PITCHEEZY_OBSERVER_LIVE_STORE="$STORE_ENV" \
      ${PZ_STAGING_REPLAY_DIR:+PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR="$PZ_STAGING_REPLAY_DIR"} \
      ${DELAY:+PITCHEEZY_OBSERVER_LIVE_DELAY_S="$DELAY"} \
      nohup "$PY" -m uvicorn observer_app.main:app --host 127.0.0.1 --port "$PORT" --log-config "$APP/logging.json" \
      >> "$LOG" 2>&1 &
  echo $! > "$PID"
  # Capture birth identity before readiness; exec preserves PID and start time.
  if ! ps -p "$(cat "$PID")" -o lstart= > "$DIR/server.identity"; then
    rm -f "$PID" "$DIR/server.identity"
    echo "process exited before ownership capture; see $LOG" >&2; return 1
  fi
  i=0
  until curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; do
    i=$((i+1)); [ $i -lt 50 ] || { echo "no health after 10 s; see $LOG" >&2; stop_owned; return 1; }; sleep 0.2
  done
  if ! owned; then echo "ownership mismatch after readiness" >&2; return 1; fi
  echo "started pid $(cat "$PID") on 127.0.0.1:$PORT ($1)"
}

case "${1:-}" in
  start) launch store ;;
  nostore) launch nostore ;;
  stop)
    stop_owned ;;
  status) running && echo "running pid $(cat "$PID")" || echo stopped; "$PY" -m observer_app.live_store status --db "$DB" ;;
  health) curl -fsS "http://127.0.0.1:$PORT/api/health" ;;
  state) curl -fsS "http://127.0.0.1:$PORT/api/live/${2:?gamePk}/state?delay=${3:-0}" ;;
  backup) "$PY" -m observer_app.live_store backup --db "$DB" --to "$DIR/backups/live-$(date +%Y%m%d-%H%M%S).sqlite3" ;;
  restore)  # stopped service only: keeps the replaced db next to it
    if running; then echo 'stop the service first' >&2; exit 1; fi
    SRC="${2:?backup file}"
    STAMP=$(date +%Y%m%d-%H%M%S)
    for f in "$DB" "$DB-wal" "$DB-shm"; do [ ! -f "$f" ] || mv "$f" "$f.replaced-$STAMP"; done
    cp "$SRC" "$DB" && echo "restored $SRC -> $DB" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac

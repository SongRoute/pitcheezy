#!/bin/sh
set -eu
OBSERVER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export PITCHEEZY_OBSERVER_RUN="${PITCHEEZY_OBSERVER_RUN:-/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-improvement-v2}"
export PITCHEEZY_OBSERVER_PORT="${PITCHEEZY_OBSERVER_PORT:-8767}"
exec sh "$OBSERVER_ROOT/apps/observer/run.sh"

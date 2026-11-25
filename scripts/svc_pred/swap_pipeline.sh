#!/bin/zsh
# EXP-SVC-PRED-001 swap loop: final-mix.json -> pins, gate, instance, 849799 predict, watch build, compare, E_PS npz, fan_numbers,
# booth lines (printed, never applied). Usage: scripts/svc_pred/swap_pipeline.sh <final-mix.json> [--name N] [--ref NAME=WATCH]
set -eu
here=${0:A:h}
exec python -B "$here/swap_pipeline.py" "$@"

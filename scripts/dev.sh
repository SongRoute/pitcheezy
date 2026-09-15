#!/usr/bin/env bash
# 맥미니 작업 세션 시작 (D11). tmux 세션 "pz"에 창 두 개:
#   claude  — Claude Code (레포 루트)
#   runs    — 실험 실행·로그 확인용 셸
# 이미 있으면 붙기만 한다 (없는 창만 추가). SSH가 끊겨도 세션은 남는다.
# 명령어 모음: docs/daily.md
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SESSION=pz

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n claude -c "$REPO"
  tmux send-keys -t "$SESSION:claude" 'claude' Enter
fi

if ! tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx runs; then
  tmux new-window -t "$SESSION" -n runs -c "$REPO"
  tmux send-keys -t "$SESSION:runs" 'mkdir -p runs/_logs; echo "실험 예: nohup python scripts/run_experiment.py --config configs/EXP-P0-001.yaml --seed 0 > runs/_logs/EXP-P0-001_s0.log 2>&1 &"' Enter
fi

tmux select-window -t "$SESSION:1"
if [ -n "${TMUX:-}" ]; then
  exec tmux switch-client -t "$SESSION"
else
  exec tmux attach -t "$SESSION"
fi

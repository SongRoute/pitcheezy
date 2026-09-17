#!/usr/bin/env bash
# P0 세 실험 × 시드 3. 순서 중요: 003 은 001 의 s0 텐서를 참조.
cd ~/Projects/pitcheezy
for id in EXP-P0-001 EXP-P0-002 EXP-P0-003 EXP-P0-004; do
  for s in 0 1 2; do
    .venv/bin/python scripts/run_experiment.py --config configs/$id.yaml --seed $s > runs/_logs/${id}_s$s.log 2>&1 || echo "FAIL $id s$s"
  done
done
echo DONE

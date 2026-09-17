#!/usr/bin/env bash
# EXP-P0-007~010 순차 + 위약 검사. K=6(009) 는 메모리 13GB 라 단독.
cd ~/Projects/pitcheezy
for id in EXP-P0-007 EXP-P0-008 EXP-P0-010 EXP-P0-009; do
  for s in 0 1 2; do
    .venv/bin/python scripts/run_experiment.py --config configs/$id.yaml --seed $s > runs/_logs/${id}_s$s.log 2>&1 || echo "FAIL $id s$s"
  done
  .venv/bin/python scripts/ope_placebo.py --config configs/$id.yaml --tau 0.005 > runs/_logs/${id}_placebo.log 2>&1 || echo "FAIL placebo $id"
  echo "done $id"
done
echo DONE

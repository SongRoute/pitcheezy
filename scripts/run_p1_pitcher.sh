#!/usr/bin/env bash
# P1 첫 축(D24) 투수별 구조 ⓑ: EXP-P1-001~003 × 시드 0,1,2 + 위약 검사(τ=0.005) + 짝지은 비교. K=1 이라 시드당 ~1분.
cd ~/Projects/pitcheezy
IDS="${IDS:-EXP-P1-001 EXP-P1-002 EXP-P1-003}"
for id in $IDS; do
  for s in 0 1 2; do
    .venv/bin/python scripts/run_experiment.py --config configs/$id.yaml --seed $s > runs/_logs/${id}_s$s.log 2>&1 || echo "FAIL $id s$s"
  done
  .venv/bin/python scripts/ope_placebo.py --config configs/$id.yaml --tau 0.005 > runs/_logs/${id}_placebo.log 2>&1 || echo "FAIL placebo $id"
  echo "done $id"
done
{
  for pair in ${PAIRS:-"EXP-P1-001 EXP-P0-006" "EXP-P1-001 EXP-P1-002" "EXP-P1-003 EXP-P1-001" "EXP-P1-003 EXP-P0-006" "EXP-P1-001 EXP-P0-002" "EXP-P1-003 EXP-P0-002"}; do
    for ms in 0 1 2; do .venv/bin/python scripts/ope_compare.py $pair --model-seed $ms || echo "FAIL compare $pair ms$ms"; done
  done
} > runs/_logs/EXP-P1-pitcher_compare.log 2>&1
echo DONE

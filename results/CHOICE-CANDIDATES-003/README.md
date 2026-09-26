# CHOICE-CANDIDATES-003 — two frozen current-action screens

A preregistered exactly two rules in `configs/EXP-CHOICE-CANDIDATES-003.json`
and `docs/contracts/choice-candidates-v3.md` as `23b9de4` after reviewing
the first diagnosis but before computing candidate outputs. The runner reads
the same 151 PA/563 pitch Q checkpoints and verifies their hashes against the
committed diagnosis summary. No new inference or training was used. The
repository `summary.json` includes the full source/result hashes and compact
aggregates; SSD `results.json` also contains each pitch's actions and all
five leave-one-member-out contrasts.

| Fixed candidate | Changed action/type | Ensemble current-action mean Δ (%p) | Mean worst-member regret Δ (%p) | Held-out member mean Δ (%p) | Decision |
|---|---:|---:|---:|---:|---|
| Near-value repertoire | 221 / 221 | −0.009174 | +0.015004 | −0.008522 | Reject |
| Minimax member regret | 64 / 30 | −0.000771 | −0.001003 | −0.000429 | Reject |

The game-cluster exploratory intervals for ensemble contrast are
−0.010722 to −0.007122 %p for repertoire and −0.001204 to −0.000408 %p
for minimax. Actual-type top match rose from frozen 147/563 to 211/563 with
repertoire and fell to 142/563 with minimax; this is descriptive only.
Minimax reduces worst-member regret slightly, but its five-fold action
unanimity is 0.730 versus the four-member mean-Q reference's 0.801.
Repertoire's unanimity is 0.840 versus 0.801, at the cost of larger
worst-member regret. Held-out member scores are model sensitivity checks,
not observed policy outcomes.

The frozen baseline is the argmax of the ensemble's own Q. Therefore no
changed action can improve that same ensemble's **current-action** Q by
construction. These fixed screens measure a robustness/concentration tradeoff;
their negative ensemble contrasts trigger the preregistered rejection rule.
Each judge retains its own optimized future PA continuation. This does not
estimate full rollout value, identified OPE, or causal defensive WE. No
service adoption follows.

Reproduce from pristine immutable artifacts with:

```sh
/Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python scripts/b_choice_candidates.py --config configs/EXP-CHOICE-CANDIDATES-003.json
```

The runner refuses an existing artifact directory. The first output was
preserved as `CHOICE-CANDIDATES-003-preaudit` while a committed diagnosis
hash pin, a synthetic held-out leakage test, and baseline robustness controls
were added. The final output at `CHOICE-CANDIDATES-003` has the same two
rejection decisions. The final run log is `CHOICE-CANDIDATES-003-final.log`.

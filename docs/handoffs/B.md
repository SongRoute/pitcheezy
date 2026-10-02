# B — frozen recommendation choice diagnosis

2026-09-24. A froze and committed `EXP-CHOICE-DIAG-002` as `885bb768`
before the new DEV run. B used the previously selected six July 2025 DEV games
and the five frozen checkpoints; no new model fit, 2026/final data, CV
implementation, service change, or live user DB access occurred.

Reproducible code: `scripts/b_choice_diagnosis.py`, pure metrics in
`scripts/b_choice_metrics.py`, and `tests/test_b_choice_metrics.py` (5 passed).
`results/CHOICE-DIAG-002/summary.json` and `README.md` give the exact counts,
hashes, interpretation, and reproduction command. The full 2.5 MB diagnostic
result and 151 PA checkpoints live on T7 under `CHOICE-DIAG-002`.

Confirmed: the observed/recommended type mixtures diverge on the eligible
cohort, with ST recommended top nearly twice its observed count; many model
Q gaps are small and member rank disagreement is frequent. These facts do not
show a benefit from copying actual mix or changing the policy. The code's
WE-maximizing objective and missing direct mix constraint are known mechanisms,
but their contributions to this specific discrepancy remain unmeasured.
Prediction quality, internal WE, and identifiable policy value remain separate;
OPE is null. A will review this diagnosis and only then run at most two frozen
offline candidates using the cached Q values.

## Two frozen follow-up screens

A subsequently froze exactly two current-action rules as `23b9de4`.
`scripts/b_choice_candidates.py` evaluated them from the 151 immutable Q
checkpoints; `tests/test_b_choice_candidates.py` added four focused tests,
including one that would catch leakage from a held-out model into its own
selection. [Candidate result](../../results/CHOICE-CANDIDATES-003/README.md)
records the paired changes, result hash and rejection. Both decrease the
frozen ensemble's current-action Q and are rejected for service adoption by
the predetermined rule. The baseline already maximizes this same Q, so
improvement on that measure was mechanically impossible for a changed action;
the screens measured robustness and concentration tradeoffs. Minimax reduces
worst-member regret but its held-out/member stability check does not establish
a benefit. Actual match is only a descriptive diagnostic.

Model used for this B work: `gpt-6-sol` with medium reasoning. Token/cost usage
was not exposed. No heavy process remains. The existing user review server on
port 8766 was untouched.

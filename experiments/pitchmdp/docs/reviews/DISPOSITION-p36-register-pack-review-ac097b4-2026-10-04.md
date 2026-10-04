# Disposition — same-model review of the P36 registration pack (ac097b4), T109

PM worker (Opus 5.5), 2026-10-04 23:48–23:50 KST. Review `OPUS-review-p36-register-pack-ac097b4-2026-10-04.md`
(same_model_review; Codex_review_pending; child helper 26125, exit 0, denials [], opus only). Labels after this
disposition: pack `same_model_reviewed; Codex_review_pending`. NOT registered, nothing archived or launched.

| # | review | disposition |
|---|---|---|
| BLOCKER-1 | gate keys did not exist in the probe | ACCEPTED (`bfe48bc` → `ff0a93b`). Same finding as the PM's 23:37 cross-branch check; the probe review (07ab288 on `codex/robust-clip-probe-1004`) independently names the same three prereg §4 (ii) gates. |
| MAJOR-2 | any REGISTERED EXP-P35-001.json accepted (dry run passes) | ACCEPTED (`0dceba7` → `6d8b644`): requires `registration.source_commit == 8864389` and probe summary `311f0d4f…`, both literals of the frozen `make_p35_config.py`. |
| MINOR-3 | summary not bound to the P36 probe; family not recomputed; bool passes as seconds | ACCEPTED (`96e1d13` → `bef16b0`). PM verified: `PROTOCOL = 'robust_clip_timing_probe_v1'` at probe `:62`; module file sha `a3d6cbce…52fa` = `MODULE_SHA`. |
| MINOR-4 | actual launch time not checked | ACCEPTED (`4467e46` → `5f2b740`): template refuses a start off `queue.date` or with < 5 h before `no_new_fit_after`. |
| README | item order, P35 pin line | ACCEPTED (`b6fbfe0` → `ba14cd0`); review doc `37c071d` → `d80a45e`. |
| NOTE-5..13 | notes; 11 argv kept (NOTE-7); stale DRAFT `decision` text (NOTE-13, frozen, replaced at registration) | Accepted as notes. |

PM checks (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10`, from `experiments/pitchmdp`):
- `tests/test_cuda_robust_clip_registered_pack.py tests/test_cuda_robust_clip_wiring.py tests/test_cuda_robust_clip_wiring_review.py`
  → 238 passed (6.04 s) — same as the reviewer.
- PM mutant: generator `GATE_KEYS` back to the guessed name → 2 failed, 64 errors; restored, `git status` clean.

Use only after: (a) EXP-P35-001 terminal 25/25 + scorer complete + descriptive report; (b) the real
`configs/EXP-P35-001.json` merged into this lineage and `test_cuda_ple_registered.py` 5/5 on it; (c) P36 smoke 20/20 via
`check_p36_smoke.py` (f988f0a); (d) the reviewed probe (07ab288) complete with every gate true. Sequencing with the
probe branch (probe review m4): merge pack + probe + `1085755` in the commit that records the probe result.

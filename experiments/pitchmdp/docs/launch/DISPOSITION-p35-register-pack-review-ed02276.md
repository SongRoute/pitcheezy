# Disposition — same-model review ed02276 of the P35 registration pack 34d36ab (2026-10-04 20:15 KST, v4 cycle-026)

Reviewer: distinct Opus 5.5 session (helper 97746, exit 0, permission denials []). Label `same_model_reviewed;
Codex_review_pending`. Still NOT registered, NOT launched: Song's go (DECISION-REQUEST-p35-go-ko.md) is unanswered.
No P35 score exists; nothing here reads results.

| Finding | Disposition |
|---|---|
| MINOR-1 no fixed deadline rule | ACCEPTED. Rule fixed now, before any go, from budget/task timeout only: L+5 h / +1 h / +30 min, one KST date. Enforced by `make_p35_config.py` (new `<launch HH:MM>` argv) and written into `queue.date_note`. |
| MINOR-2 outer timeout unspecified | ACCEPTED (README step 3): budget ≥ (aggregate_by − L) + 900 + 600 s, ≤ 40,000 s. |
| MINOR-3 procps pgrep false block | ACCEPTED as operational rule (README step 3); template unchanged (fail-closed only). |
| MINOR-4 go not referenced | ACCEPTED: required `<go reference>` argv → `registration.go` (inside the `registration` REGISTRATION_PATH); registered-config test must assert non-empty. |
| MINOR-5 argv not validated | ACCEPTED: exactly 9 args, `strptime('%Y-%m-%d')`, date ≥ today KST, strict HH:MM, numeric ordering via the rule. |
| MINOR-6 overwrite | ACCEPTED: `open(out, 'x')`. |
| NOTE-1 asserts under -O | README: plain `python` only. |
| NOTE-2 11 top-level keys | README step 1: registered-config test asserts the extra queue/protocol keys (P34 pattern). |
| NOTE-3 terminal done == N | Kept (fail closed). |
| NOTE-4 review wording | Kept (accurate). |
| NOTE-5 "without new code" | README wording fixed; extractor `config_diff` step added. |
| NOTE-6 scorer 900 s for P35 | Kept; duration 미측정. |

Checks after the fix (Mac, OMP 1, `perl alarm 120`, `.venv/bin/python`, summary sha 311f0d4f…):
- valid `2026-10-05 09:00 14:00 15:00 15:30 dry-run "dry go ref"` → exit 0, sha fe561a88…f1f0 (dry, not committed);
  extractor `config_diff` vs DRAFT: 10 paths, 0 outside `REGISTRATION_PATHS`, top-level key order equal; queue
  `Deadlines(queue)` constructs.
- refused (exit 1, each with its own message): same output path again (FileExistsError), no_new 13:59, kill 14:59,
  agg 15:29, date `garbage`, `9:00`, blank go, extra argv, past date 2026-10-03, old 7-arg form.
- `run-p35-pipeline.TEMPLATE.py` unchanged (still fails closed on the configsha placeholder).

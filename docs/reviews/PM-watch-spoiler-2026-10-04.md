# /watch navigation without spoilers — proposal branch (T23b, 2026-10-04)

Status: self-tested; same_model_review_not_run; Codex_review_pending. Not integrated, not deployed. Decision for root/Song
(option 2 of the audit `codex/continuous-pm-1004` `reports/SPOILER-replay-navigation-audit-ko.md`, commit 9245e12).

Why: the always-visible half-inning row of /watch is built from every pitch of the game. On the 12 watch datasets it ended
at "9초" in 8 games (home team won, 8/8) and showed "10말" in 1; the plate-appearance select listed every PA (61–87) and the
progress showed the total (243–319). Audit script/result: `pitcheezy-pm-jobs/20261004-continuation/spoiler-audit/`.

Change (`apps/observer/web/src/WatchAlong.tsx` only, screen only; API, embed mode and /game untouched):
- row is always 1초–9말; a half that was never played (9말 after a home lead) jumps to the last pitch;
- extra-inning halves appear only once the viewer has reached them (furthest pitch kept per game in localStorage
  `pitcheezy.watch.<game>.furthest`, so a reload does not hide them again);
- plate-appearance select lists only PAs up to the furthest pitch reached; progress `n구째` without the total.

Not changed: `/api/watch/{game}` still sends every pitch's situation (incl. the score before the last pitch = final in 12/12);
anyone reading the network payload still sees it (audit option 3, a contract change). /game "영상과 맞추기" sheet halves unchanged.

Checks (Mac, light; node_modules = temporary symlink to the main checkout's, removed):
- `tsc --noEmit -p tsconfig.json` exit 0; `vite build --outDir /tmp/pz-watch-spoiler-1004-dist` ok.
- NEW `tests/watch-spoiler-smoke.mjs` (two synthetic games, canned responses, no server): `PZ_WEB_DIST=/tmp/pz-watch-spoiler-1004-dist node tests/watch-spoiler-smoke.mjs` → ok.
- negative control on the 840eeee build (`/tmp/pz-reconnect-1004-old-dist`): 3 FAIL (row ends 9초, `1/102구`, 51 PA options) then a timeout on the missing 9말 button.
- mutants: extra innings always shown → 2 FAIL; furthest not restored after reload → 1 FAIL.
- regression `showcase-smoke.mjs` (temp copy with the dist path from env, deleted) → ok (embed still has no navigation).
- `watch-along-smoke.mjs` needs a running observer; not run (production 8766 untouched).

Limits: real iPhone Safari 미측정; whether viewers actually read the row as a spoiler is 미측정; a viewer who opens a shared
`#i=N` link reaches N at once (their own choice); clicking the never-played 9말 shows the game end.

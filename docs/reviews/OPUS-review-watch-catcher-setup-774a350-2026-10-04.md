# Opus review — catcher mitt (horizontal setup) in WatchAlong / Showcase iframe (`cc31d6e..774a350`)

`same_model_review; Codex_review_pending` — Claude Opus 5.5, distinct from the author. 2026-10-04.

Target: `git diff cc31d6e 774a350` — `apps/observer/web/src/WatchAlong.tsx`, `watch-along.css`, `tests/watch-setup-smoke.mjs` (new).
Read: the author's report, `docs/handoffs/CV-demo-team-reply-2026-10-04.md` (서비스 구현 작업 지시), `watch-catcher-setup-request.md`,
`Game.tsx` (existing setup mark), backend `setup_estimates.py`, `watch_along.py` and the routes in `main.py` (read only, no server).

## Disposition

**No blocking finding. Ready for the 8771 preview integration.** The real 1440/390 check against the real setup response is
still owed after integration (see "Not verified"). Four non-blocking findings follow (N1–N4). One test-only commit of mine
strengthens the key-matching check (C1).

Hard rules, checked:
- (a) No spoiler. The value is computed during render. The gate is `reveals[index]` **and** `reveal.index === decision.index`
  **and** `setups.game === game_pk`, with the exact key `${game_pk}:${at_bat_number}:${pitch_number}` (`WatchAlong.tsx:344`).
  The mark and text exist only inside `Revealed`, or in the Zone given `setupX`, which is `undefined` until that reveal.
  Nothing is stored per pitch, so a late answer, a failed reveal, ←/→, a scene change (the iframe is re-keyed per scene in
  `Showcase.tsx:129`) or a game change cannot carry a value over. Mutation M2 (gate removed) fails the smoke.
- (b) Absence. 404/503/abort/network go to `.catch(() => undefined)`. An answer for another game is dropped
  (`WatchAlong.tsx:250`; mutation M3 fails the smoke). A pitch with no row → `undefined` → nothing. A row that is `unavailable`
  or has a non-finite x → `null` → one plain line, no mark. `x = 0` passes `typeof === 'number'` and is drawn at the centre
  (smoke A, x1=90). Off-plate x is not clamped (mutation M4 fails the smoke). Beyond ±1.7 ft it is described in words.
- (c) Wording. The legend reads "점선 = 포수 미트 위치 추정 (가로만)". The collapsed caveat says: AI reading, before human
  review, no height, not the pitcher's intent or target, not used by recommendations, only some pitches, **this game's
  accuracy unmeasured, M3 is two other games**, plus the server badge. No proxy-agreement number (2.5 px / 1.1 in) appears,
  so nothing is presented as physical accuracy.
- (d) `CvEvidenceCard` and Showcase are not in the diff. cv-evidence, showcase, spoiler and pre-reason smokes pass.
  Space/←/→ and embed=1 are exercised in smoke B/C/M. No horizontal overflow at 390.
- (e) `Live.tsx`, `Game.tsx` and the backend are not in the diff. Live uses `Recommendation` without `setupX`, and its
  candidates are never empty, so the Zone caption (`{candidates.length ? '숫자=추천 · ' : ''}`) is unchanged there.

## Real data shape (read only, nothing copied)

- Backend `SetupEstimates.view` returns `game_pk: int`, `badge: str`, `pitches[]` with `at_bat_number: int`,
  `pitch_number: int`, `status: 'estimated'|'unavailable'`, `plate_x_feet: float|None`. These are the fields the code reads.
- Timeline `/api/watch/{game}` passes `at_bat_number` and `pitch_number` through from the dataset (`PRE_FIELDS`).
  In `DEMO-WS-2026/watch/849843.json` both are `int`: 262 decisions, no duplicate key.
- `DEMO-WS-2026/intent/849843.jsonl`: 39 rows, 27 estimated with x, 12 unavailable with `plate_feet = None`.
  **All 39 keys match a timeline decision.** x ranges from −1.029 to 0.696 ft (none is exactly 0, none beyond 1.7 ft).
  The 27 rows are at pre.status ready 25 and unsupported 2, so the "no recommendation + zone in the reveal" path will appear
  twice. The 39 Showcase scenes are exactly these 39 pitches: 27 will draw a line and 12 will say "읽지 못해".
- No separator-less key collision exists in 849843 (checked), so the code is right for the real game. I closed the test gap
  anyway (C1).

## Findings

### N1 (non-blocking) — the mitt and the ball use different picture bounds in the same Zone
`WatchAlong.tsx:84` draws the mitt up to ±1.7 ft. `WatchAlong.tsx:104` (pre-existing) pins the ball dot at ±1.1 ft.
Concrete case: mitt −1.65, ball −1.6. The picture shows the ball about 0.55 ft to the right of the mitt; the real gap is
0.05 ft. Screenshot: `/tmp/pz-opusrev-774a350-shots/edge--1.65-390.png`. In real 849843 the mitt never goes beyond 1.03 ft.
One pitch (52:5) has the ball at −1.35, drawn at −1.1, against a mitt at −0.57. The drawn gap is 0.53 ft against a real
0.78 ft, on the same side. That is minor, but the gap is now what viewers will compare. Suggested change (one bound for
both marks; the existing "존 그림 밖" text then covers the rest):
```diff
-export const SETUP_X_DRAWN = 1.7;  // feet either side of centre the zone picture can show; beyond it the mitt is described, not drawn
+export const SETUP_X_DRAWN = 1.1;  // same bound as the ball dot below, so the two marks never disagree; beyond it the mitt is described
```
I did not apply this. It is a display decision (−1.03 is still drawn either way), and the smoke's case G (2.5 ft) still holds.
The label "미트" at ±1.65 ft reaches −0.3/180.3 in a 0..180 viewBox, a 0.3 px clip. N1 would remove that too.

### N2 (non-blocking) — the visible legend does not say "AI · 검토 전"
`WatchAlong.tsx:398–399`: only the collapsed `<details>` says it is an unreviewed AI reading. On a phone most viewers will
not open it. The work order asks only for "간단한 범례 + 펼쳐 보는 한계 설명", which this meets. If PM wants it visible, a
suggestion:
```diff
-        : <><i aria-hidden="true" />점선 = 포수 미트 위치 추정 (가로만)</>}</p></>}
+        : <><i aria-hidden="true" />점선 = 포수 미트 위치 추정 (AI·검토 전, 가로만)</>}</p></>}
```

### N3 (non-blocking) — `reveal.index === decision.index` cannot be tested as written
`reveals` is keyed by the requested index, so this clause matters only if the server answers with another index.
Mutation M1 (clause dropped) passes the smoke. Keep it as a harmless guard, or add a smoke case where the reveal answer
carries a different `index`. Not required.

### N4 (non-blocking, a11y) — the no-recommendation Zone keeps the label "추천 대략 위치와 실제 투구"
`WatchAlong.tsx:91`. In the reveal-section Zone (`candidates=[]`) the screen-reader label still mentions recommendations.
Also, the `<g role="img" aria-label=…>` mark sits inside an `<svg role="img">`, whose children screen readers do not announce.
The text legend covers it. Suggestion:
`aria-label={candidates.length ? '포수 시점 스트라이크존: 추천 대략 위치와 실제 투구' : '포수 시점 스트라이크존: 실제 투구'}`.

### Observations (no change)
- When `#i=N` is opened, earlier pitches of the same plate appearance are revealed in the background (existing behaviour).
  Stepping back to them shows their mitt. They are past pitches, already shown in the sequence strip, so this is not a spoiler.
- The legend sits in the reveal section and the mark in the recommendation Zone above it. At 390 px after a reveal, both are
  on one screen (`viewport-after-reveal-390.png`).
- Dark scheme: `--deep` is defined in both schemes (`watch-along.css:2,7`).

## My commit

- **C1** `14789fa test(observer-web): setup smoke decoys catch separator-less and swapped pitch keys`.
  My mutation M5 built the key without the `:` separator on both sides, and it **passed** the author's smoke: the decoys
  around 4:1 cannot collide under concatenation. Index 4 is now 11:1, with decoy lines 1:11, 11:2, 111:1 and 11:11.
  Results: real code → `ok`; the M5 build → `FAIL 1440 F no line: a mark was drawn`.

## Commands run (Mac mini, `nice -n 10`, one at a time, `perl -e 'alarm 600; exec @ARGV'`, no MPS, no server, no port)

Working directory `apps/observer/web`. `node_modules` was a symlink to main's `apps/observer/web/node_modules`; nothing was
installed. The symlink was removed at the end with `rm apps/observer/web/node_modules`.

| command | exit |
|---|---|
| `npx tsc -b` | 0 |
| `npx vite build --outDir /tmp/pz-opusrev-774a350-dist --emptyOutDir` (55 modules, JS 379.60 kB, CSS 102.90 kB) | 0 |
| `PZ_WEB_DIST=/tmp/pz-opusrev-774a350-dist node tests/watch-setup-smoke.mjs /tmp/pz-opusrev-774a350-shots` → `ok`, 22 recorded cases | 0 |
| `PZ_WEB_DIST=… node tests/pre-reason-neutral-smoke.mjs` → `ok` | 0 |
| `PZ_WEB_DIST=… node tests/watch-spoiler-smoke.mjs` → `watch spoiler smoke: ok` | 0 |
| `DIST=… node tests/cv-evidence-smoke.mjs` → `ok (desktop 1440 + phone 390/360, synthetic timeline)` | 0 |
| `DIST=… node tests/model-evidence-smoke.mjs` → `ok (desktop 1440 + phone 390/360, synthetic timeline)` | 0 |
| `PZ_WEB_DIST=… node tests/game-frame-smoke.mjs` → `game frame smoke: ok` | 0 |
| `showcase-smoke.mjs` (hard-codes `../dist`): temporary symlink `dist -> /tmp/pz-opusrev-774a350-dist`, run, then `rm dist` → `showcase smoke: ok (synthetic WAV media + timeline/reveal fixtures, desktop + phone)` | 0 |
| after C1: `PZ_WEB_DIST=/tmp/pz-opusrev-774a350-dist node tests/watch-setup-smoke.mjs` → `ok` | 0 |

Mutations. Each was built to its own `/tmp/pz-opusrev-774a350-mut{n}-dist` and run with the smoke. Each was restored with
`git checkout -- src/WatchAlong.tsx`; `git diff --stat -- src` was empty after each.

| # | mutation | smoke | first failure |
|---|---|---|---|
| M1 | drop `reveal.index === decision.index` | **passes** (exit 0) | — see N3 (redundant guard) |
| M2 | drop the reveal gate (`setupX = setups?.game === … ? … : undefined`) | fails (1) | `1440 A setup arrived, not revealed: setup shown` |
| M3 | accept another game's answer (drop `v.game_pk !== gamePk`) | fails (1) | `1440 K wrong: setup shown` |
| M4 | clamp x onto the plate (±0.83) | fails (1) | `1440 B x=-1.0: mark x1 48.5 != -1ft (40)` |
| M5 | key without `:` between at_bat and pitch | passes on 774a350's smoke, **fails after C1** | `1440 F no line: a mark was drawn` |

Screenshots (local, not committed), `/tmp/pz-opusrev-774a350-shots/`:
- `setup-{ready,no-rec}-{1440,390}.png`: from the smoke.
- `edge-{1.65,-1.65,1.1,-1.03}-390.png` and `viewport-after-reveal-390.png`: from `/tmp/pz-opusrev-774a350-edge/edge.mjs`,
  synthetic, intercepted at host `pz.test`, no port.

At both 390 and 1440 the page is no wider than the viewport (`scrollWidth` 390/1440 for all four edge values). At 390 the
reveal adds three short lines: the legend, a collapsed "이 표시의 한계" and, for no-recommendation pitches, a 148 px zone.
Nothing overlaps and it does not look cluttered. With the caveat open it adds about 4 short paragraphs, which is acceptable
because it is collapsed by default.

## Not verified (for the integrator)
- The real `/watch?game=849843` and `/showcase` at 1440/390 against the real 8771 setup response, after integration
  (I ran no server and used no port).
- The offline fallback bundle containing the setup response.
- Left/right direction and units checked against the video or a person (rehearsal 10/5).

— Claude Opus 5.5

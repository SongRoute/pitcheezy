# 최종 후보 공개 화면 E2E 영수증 (CL-10 준비)

상태: **준비 실행 기록(2026-10-19)**. 최종 실행은 10/21 UI 수정 뒤 새 후보로 다시 한다. 작성 ClaudeCode(Opus 하위 작업자), 독립 검토 대기.
CLAIM: SongRoute/pitcheezy#37 comment 6257222210 + CLAIM-DELTA(같은 이슈, smoke `public-viewer-e2e-smoke.mjs` 쓰기 범위 추가, 코디네이터 승인). 근거: Codex PLAN 6248485785(CX-02 정적 대조), 포인터 6248803007(CX-05 실행기).
브랜치 `ws/cl10-1019`, base `dccde711748d40565a78c50356cf8cb5c4f43d3e`.

## 무엇을 바꿨나

문제(CX-02): `test_public_viewer_e2e.py`가 과거 BUILD 경로(`20261016-parallel-execution-001/.../dist-public`)를, gateway E2E가 과거 gateway fixture(`87168d7`, 파일 4개)를 코드에 고정했다. 그래서 PASS가 최신 화면을 증명하지 못했다.

- 후보 고정값을 **명시 입력**으로 바꿨다. `OBSERVER_E2E_CANDIDATE=<JSON 경로>`가 있으면 그 파일을, 없으면 `test_public_viewer_e2e.DEFAULT_CANDIDATE`를 쓴다. 형식 `pitcheezy-public-e2e-candidate-v1`: `source_commit`, `public_build{dir, members{이름:{bytes,sha256}}}`, `gateway{commit, runtime_dir, files{이름:sha256}, adapter{name,sha256}}`, `smoke{path[, sha256]}`. members 모양은 CX-05 manifest의 `public_build_members`와 같다.
- 브라우저·gateway를 띄우기 **전에** 모두 검사하고, 하나라도 다르면 `REFUSED candidate pin: …`으로 실패한다(대체 실행 없음).
  - source: 이 checkout의 web 빌드 입력(`index.html, public, src, package.json, package-lock.json, tsconfig.json, vite.config.ts, .env.public`, 작업트리 포함)이 `source_commit`과 같아야 한다.
  - build: 디렉터리 파일 목록이 members와 정확히 같고, 각 파일이 symlink가 아니며 길이·SHA가 같아야 한다.
  - gateway: runtime 파일별 SHA와 loopback 어댑터 SHA.
  - smoke: 경로(선택적으로 SHA)를 확인한다. 실행 뒤 smoke 영수증의 `buildHashes`가 후보 members와 같은지도 확인한다.
- `apps/observer/web/tests/public-viewer-e2e-smoke.mjs`의 과거 asset 표(구 17–23행)를 **없앴다.** 표는 env `PUBLIC_E2E_CANDIDATE`(후보 JSON 경로)의 `public_build.members`에서만 온다. pytest는 검사를 통과한 후보를 `tmp_path/candidate.json`으로 써서 넘긴다. 기본 표가 없으므로 env가 없거나 경로·SHA·길이가 하나라도 다르면 서버를 띄우기 전에 실패한다 — 다른 build에서 조용히 통과할 수 없다. 이 smoke의 호출자는 이 두 테스트뿐이다(grep 확인).
- 합성 거부 대조 테스트 2개 추가:
  - `test_candidate_pins_refuse_mismatch`: 정상 1, build 변조·추가 파일·symlink·gateway 변조·smoke SHA 불일치·없는 commit·web 소스가 다른 commit(`56b4347`)·schema 불일치 거부.
  - `test_smoke_refuses_without_or_with_wrong_candidate`: smoke를 env 없이 / 틀린 SHA 후보로 실행하면 영수증도 남기지 않고 실패.

## 현재 기본값 (2026-10-19 출시 후보)

| 항목 | 값 |
|---|---|
| source | `dccde711748d40565a78c50356cf8cb5c4f43d3e` (`ws/launch-2026` 기준 오늘 후보) |
| public build | worktree `apps/observer/web/dist-public`, `npm run build:public`(node_modules는 정본 레포 것을 symlink, npm install 없음) |
| `index.html` | 659 B `9c354a586b990873942cd40c3d4168f2a88d50225de9b7a65c7f799b1728513d` |
| `favicon.svg` | 249 B `bbfd72e5d077bd545b2cf568e5267c6637598d5253746f5b395e0d5f3fb144d9` |
| `assets/index-BLGE1MgF.js` | 287672 B `50ead108da5d8ada2a003c1ae2538d15ae43cf6b471676e5f4b64bef9575fcb6` |
| `assets/index-CUX1xjau.css` | 103215 B `90bb1d702395964f4eae9993407c85424f772bba729e412253fb9fd2d1e7cd18` |
| gateway | PR42 `48ec406589982547e298ff2ee4bc2d64ed0216c0`, runtime `pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/compat-v4.3/runtime` |
| gateway 파일 | worker `ba6bb83b…`, gateway `3fd0217b…`, notice `d94b5c22…`, access `67c3865b…`, telemetry `c90d0f23…` (전체 SHA는 테스트 파일) |
| 어댑터 | `local-https-gateway.mjs` `470c3cf7d760b83011af6f3177ba0421b8d978a420c87a0cdf93c17eff546864` |
| smoke | `apps/observer/web/tests/public-viewer-e2e-smoke.mjs` (변경 후 sha `9fdba64aaaf8e8d813ea2bc48e3463ae3eed8dfd96c41ef373ef79b84d9103ee`, 변경 전 `47abfdbf…`) |

dccde71에서 새로 빌드한 4파일은 partner pack v4.3.1(`6282c63`, manifest `7c8ef7e8…`)의 `public-build` 4파일과 **바이트까지 같다**. gateway 파일 5개 SHA는 pack v4.3 PR42 호환 시험(`compat-v4.3/test_pr42_scope_v43.py`)의 PINS와 같다.

## 실행 방법

Playwright는 worktree에 `node_modules`가 없어서 실패했던 것이다. 아래처럼 정본 레포의 설치본을 symlink하면 된다(Chromium 1243은 `~/Library/Caches/ms-playwright`에 이미 있음).

```sh
cd <worktree>/apps/observer/web
ln -s /Users/song/Projects/pitcheezy/apps/observer/web/node_modules node_modules   # 커밋하지 않음
npm run build:public
cd <worktree>
PYTHONPATH=apps/observer/backend /Users/song/Projects/pitcheezy/.venv-observer/bin/python -m pytest \
  apps/observer/backend/tests/test_public_viewer_e2e.py apps/observer/backend/tests/test_public_viewer_gateway_e2e.py -q -p no:cacheprovider
# 다른 후보: OBSERVER_E2E_CANDIDATE=/path/candidate.json 을 앞에 붙인다
```

발행기는 테스트 프로세스 안에서 함수로만 돈다(`observer_app.publisher` argv 실행·리허설 프로세스 조회·신호 없음). 모든 서버는 테스트가 소유한 loopback 임시 포트다.

## 결과 (2026-10-19 09:4x–10:0x KST, 맥미니)

| # | 실행 | 결과 |
|---|---|---|
| 1 | 변경 전 원본 두 테스트(과거 BUILD·과거 gateway) | 2 passed (3.82s). 과거 고정값으로는 여전히 통과 — 최신 화면 증거 아님 |
| 2 | 변경 후 기본값(오늘 후보) | **1 passed, 2 failed — 의도된 거부.** `REFUSED candidate pin: smoke public-viewer-e2e-smoke.mjs pins a different public build (missing index.html, index-BLGE1MgF.js, index-CUX1xjau.css)` |
| 3 | 진단: 기본값 + smoke만 scratch 사본(asset 표 4줄만 후보 값으로 교체, sha `689228a7…`), 후보 JSON sha `73d4d923…` | **3 passed (3.81s)** |
| 5 | **smoke 표 입력화 뒤 기본값(오늘 후보), 09:50 KST** | **4 passed (3.65s)** — 거부 대조 2 + 직접 TLS E2E + PR42 gateway E2E |
| 4 | Codex CX-05 실행기(커밋 `1d2e14d8`에서 `git show`로 꺼낸 사본, sha `ec872299…`) + pack v4.3.1 manifest + 이 worktree build | exit 0 `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`, 3시나리오, 32요청, browser/server 정리 true |

실행 3 세부: 직접 TLS 경로와 PR42 gateway 경로 모두 `PASS_SYNTHETIC_LOCAL_HTTPS`, 시나리오 390/1440 종료 경기(각 13요청)+503 안내, 총 32요청, smoke `buildHashes` = 후보 4 SHA. gateway 집계 PUT 7·HEAD 14·GET 15(총 36), GET 쿠키 모두 있음, gateway 종료 코드 0. 영수증 SHA: 직접 `36994c60…`, gateway `64bcf310…`, 집계 `02d3eb3d…`(scratchpad, 비커밋).

실행 5 세부: 두 E2E 모두 `PASS_SYNTHETIC_LOCAL_HTTPS`(direct_static / pinned_local_gateway_via_front), 각 32요청, 390·1440 종료 경기 각 13요청 + 503 안내, buildHashes = 후보 4 SHA. gateway 집계 PUT 7·HEAD 14·GET 15(총 36), GET 쿠키 모두 있음, 종료 코드 0. 실행 3과 같은 수치다.

해석: 오늘 후보 build와 PR42 gateway는 **asset 표만 맞추면** 기존 smoke의 화면 선택자·흐름(목록→종료 경기→503 안내)을 그대로 통과한다. 실행 2의 실패는 고정값 거부가 제대로 작동한 결과다.

## 남은 일·잔여 실패

1. ~~smoke asset 표 고정~~ → CLAIM-DELTA 승인 뒤 해결(실행 5). 실행 2는 해결 전 기록으로 남긴다. `apps/observer/web/src/**`와 Codex 파일은 건드리지 않았다.
2. CX-05 실행기 재사용: 호출로 재사용 가능함을 확인했다(실행 4). 다만 이 브랜치 base에 그 파일이 없고(Codex 브랜치, 미통합), 합성 JSON만 쓰며 발행기 출력·gateway를 거치지 않으므로 pytest에 엮지 않았다. COORD-04 통합 뒤 보조 검사로 같은 후보 manifest에 함께 돌리는 것을 권한다. 사본 수정은 하지 않았다.
3. gateway는 legacy 모드(`PUBLISH_KEY_SCOPE`/`PUBLISH_EXPIRES_AT` 미설정)로 돈다. 범위 제한 모드는 PR42 호환 시험이 따로 다루며 이 E2E 범위가 아니다.
4. 검증 안 한 것: 실제 TLS·배포 호스트·Cloudflare, 390×844 레이아웃, 실기기, 철회·복구, 10/21 UI 수정본. 높이 900 smoke 통과를 이것들로 넓히지 않는다.
5. 최종 실행(10/21 뒤): 새 후보에서 `npm run build:public` → 새 4파일 SHA·source commit으로 후보 JSON(또는 기본값) 갱신 → 위 명령 실행 → 이 표에 행 추가. smoke를 직접 돌릴 때는 `PUBLIC_E2E_CANDIDATE=<후보 JSON>`이 필수다. 10/20 08:30–09:30 KST(리허설 사전 시작 창)에는 돌리지 않는다.

검토자: (대기) — 작성자와 다른 에이전트가 테스트 diff, 기본값 SHA 대조(pack v4.3.1 manifest·PR42 PINS), 실행 2·3 재현을 확인한다.

## CL-10b 보강 (2026-10-19, 커밋 00db69d8, 검토 Q-cl10/REVIEW-002 PASS_WITH_NOTES)
- 출처 검사 추가: 빌드 입력 범위에 추적 안 된 파일·무시된 파일(`git status --porcelain --untracked-files=all --ignored=matching`)이 있으면 거부한다. `.env`, `.env.local`, `.env.public.local`을 입력에 넣었고, 셸에 `VITE_*` 변수가 있으면 거부한다.
- `OBSERVER_E2E_CANDIDATE`로 넘긴 후보는 `smoke.sha256`이 필수다. 영수증에 후보 출처(JSON 경로·sha 또는 DEFAULT_CANDIDATE)와 smoke가 읽은 후보 sha(`candidateSha256`)를 남긴다.
- smoke 새 sha `77e013dd0ee531bd591bac1f96bf7b6921e5fcd652ad7ceb8630c02893d9733f` (위 표의 9fdba64a…를 대체). 최종 후보 JSON에는 이 값을 넣는다.
- 실행: 기본 후보 5 passed, smoke sha를 넣은 env 후보 5 passed (루프백만).
- 남은 메모: R4(커밋 안 된 vite 설정·상위 postcss·NODE_ENV·node_modules는 미검사, 빌드 4파일 바이트 고정이 대신 막음), R5(`src/.DS_Store`도 거부됨), R3(gateway 모듈 5개+어댑터 유지).

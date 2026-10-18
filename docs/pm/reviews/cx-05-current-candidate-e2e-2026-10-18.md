# CX-05 — 명시 public 후보 pin의 합성 브라우저 검사

상태: 실행 전 독립 경계 PASS/root 수용 → 첫 합성 실행 PASS → 독립 결과 검토 PASS/root 최종 수용. 작성자 `coordination_docs`, 검토자 `coordination_review`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248706231 . base `26dee81579409496621a72af6c2ea7263cd08de9`.

## 대상과 재사용 판단

- **v4.3.1 준비 후보** source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`, manifest SHA `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`. 최신 배포·10/21 UI·최종 출시 GO가 아니다.
- `I-partner-pack/REVIEW-v4.3-001.json`은 public 자산4개와 독립 public rebuild 동치·민감 scan0을 기록한다. `REVIEW-v4.3.1-001.json`은 문서만 변경됐고 public 자산이 동일하다고 기록한다. dccde71까지의 handoff 문서 delta와 소스 동치 근거는 [CX-02](cx-02-release-evidence-2026-10-18.md); 실제 배포 동치와 다르다.
- 기존 `apps/observer/web/tests/public-viewer-e2e-smoke.mjs:18–24`는 과거4asset hash가 고정이고 backend `test_public_viewer_e2e.py:11`도 과거 BUILD 경로다. 원본은 수정하지 않았다.
- 기존 `verify_offline.py:20–30`은 pack 전체 rglob·22파일을 읽고 runner subprocess를 실행한다. `static-publish-smoke.mjs:24–25`는 실제 경기 UI fixture를 읽고 build를 한다. 이번 허용범위와 달라 실행하지 않고 기존 E2E의 server/Playwright 검사 형태만 재사용했다.
- 새 파일은 `tests/coordination/public_candidate_smoke.mjs` 1개다. Python 보조파일·새 dependency·npm install·제품 build는 없다.

## 경계

manifest SHA/source pin 확인 후 public_build_members의 index/favicon/해시명JS/CSS **정확4개만** 읽는다. 각 파일 크기/SHA와 manifest 중복 항목 일치를 검사한다. 절대경로의 모든 조상과 파일 symlink를 거부하며 허용 경로 형식·root 경계를 확인한다. 패키지 전체 탐색·private fixture·payload·모델·실서비스/자격 접근은 없다.

모든 pin 검사는 브라우저 import/서버 시작보다 먼저 끝난다. 새 output 디렉터리는 기존 경로를 재사용하거나 덮어쓰지 않는다. 출력 내부의 작은 합성 pin fixture로 정상1·manifest 변조/asset 불일치/symlink/누락/탈출 항목 거부를 검사하고 제거한다.

source `service/api.ts:128–166,386–408`의 어댑터와 `publish_contract/v1/schema/{game,health,games,dates,manifest}.schema.json`의 형식으로 새 final game과 JSON5개를 메모리에서 만든다. 빈 pitches/players와 합성 팀명·식별자만 사용하며 publisher 원본 출력·전체 JSON Schema 검증을 대신한다고 주장하지 않는다. fixture는 output에도 보존한다.

소유한 `127.0.0.1` 임시포트 HTTP 및 Playwright driver만 사용한다. 보호 포트8766–8772는 금지하며 TLS는 검증하지 않는다. Chromium1개에서3시나리오를 순차 수행한다. context route는 exact origin/GET/allowlist 외 요청을 거부하고 SW·페이지 WebSocket을 막는다. route에서61번째 요청을 continue 전에 차단하며 시도/실제서버수를 구분한다. 서버도 Host/GET/path·60상한을 검사한다. Chromium DNS는 loopback 외 해석을 막고 background networking을 끈다. import/launch/scenario 전체80초 race, finally에서 본인 BrowserServer close5초→실패 시 본인 handle kill2초·HTTP close1초로 정리를 제한하고 전체90초 초과/정리 실패는 FAIL이다. 실서비스 프로세스 조회·신호는 없다.

launch15초와 정리 여유를 확보하기 위해 실행60초 이후에는 새 browser launch를 거부한다. 드물게 launch promise가 경계를 넘어 pending이면 영수증의 launch 시작/완료 상태와 `ownedBrowserClosed=false`로 실패를 남기고, 뒤늦게 받은 본인 handle도 닫는다. handle이 없다는 이유로 정리 성공을 추정하지 않는다.

## 실행 계획

| 검사 | 기대 | 결과 |
|---|---|---|
| pin 정상/manifest·asset·symlink·누락·escape 거부 | 모두 확인, 서버 시작 전 | PASS(정상1+거부5) |
| 390×900 목록→합성 종료 경기 | 해당 JSON bytes 소비, pageerror0, 외부요청0 | PASS,13 GET |
| 1440×900 같은 경로 | 동일 | PASS,13 GET |
| 390×900 game/health503 | 오류 안내, 추천 행0 | PASS,6 GET |

390×844의 알려진 레이아웃 문제 해결 여부는 검사하지 않는다. 높이900의 smoke 성공을 폰 실기기·접근문·TLS·회수·배포·정책/추천 성능 검증으로 확대하지 않는다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/MANIFEST.json \
  7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/public-build \
  /Users/song/Projects/pitcheezy/apps/observer/web/node_modules/playwright/index.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX05/run-001
```

## 실제 결과와 인계

2026-10-18 위 명령1회: exit0, `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`,3시나리오 통과. routedAttempts=32/totalCalls=32, blocked=[]/pageErrors=[], elapsedMs=1275, ownedBrowserClosed=true/ownedServerClosed=true. 실제 운영 요청은0이다. 생성 PNG3개를 직접 확인했고 종료 표시·빈 합성 기록·503 안내가 기대와 일치했다. 기능 smoke이며 정밀 배치/접근성 전체 판정은 아니다.

| 증거 | SHA-256 또는 경로 |
|---|---|
| 실행기(검토·실행 후 불변) | `16de914d4f24a8e1132180bef4962862a7cabe07e850938d2b2e9af38837161a` |
| receipt.json | `a00e59854279bc1a801c2eaa25b390c8475a3fa05dece78ddda9bea719ae4a32` |
| synthetic-files.json | `278dd95a0738d18e7b0b040eeeaf02755267798b6f3add1c18c14b0ab51eab46` |
| 산출물 root | `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX05/run-001` |
| 화면 | 위 root의 `final-390.png`, `final-1440.png`, `503-390.png` |

receipt는 읽은4asset의 SHA와 생성5JSON의 SHA·요청별 실제 제공 byte hash를 기록한다. 기존 제품/원본 fixture 수정 없이 실행했다. 작성자와 다른 동일 모델 에이전트 `coordination_review`가 실행기/receipt SHA·결과표·PNG3개를 독립 대조해 PASS했고 root가 수용했다. 검토자의 추가 브라우저 실행은0회다.

COORD-04가 선별 커밋하며, CL-10 기존 소유자에게 재사용 가능한 명시 pin 검사를 인계한다. 고정 준비 후보의 합성 HTTP 증거이며 운영 배포/최종 후보 전체 검사·390×844 수리·TLS·기기시험은 계속 미완이다.

graphify: `tests/coordination`에서 `graphify update .` 1회 exit0(AST only,77 nodes/97 edges/10 communities). source_file은 기존 `test_onekey_acceptance.py`와 신규 `public_candidate_smoke.mjs` 두 파일뿐이다. 생성 `tests/coordination/graphify-out/`은 `.gitignore:35`로 제외되며 비커밋, 전체 저장소 bootstrap·의미 추출은 하지 않았다.

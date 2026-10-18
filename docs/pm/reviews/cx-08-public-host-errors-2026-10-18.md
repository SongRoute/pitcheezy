# CX-08 — 공개 준비 후보의 host 오류 안내와 수동 재시도

상태: 실행 전 독립 검토 PASS/root 수용 → 합성 실행1회 PASS → 독립 결과 검토 PASS/root 최종 수용. 작성자 `coordination_docs`, 검토자 `coordination_review`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248949825 . base `2bac32489c2e2237d016154b2d3f6b027515f36f`.

## 범위와 재사용

[CX-05](cx-05-current-candidate-e2e-2026-10-18.md)의 단일 실행기에 선택 인자 `--host-errors`만 추가했다. 기존 6개 필수 인자와 기본 3개 사례(390/1440 목록→종료, 초기503)는 그대로다. 새 의존성·build·제품 코드 변경은 없다. 선택 suite는 기존 context/route/요청/시간/정리 코드를 재사용한다.

대상은 v4.3.1 **준비 후보** source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`, manifest SHA `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`다. 정확4public자산 pin·문서 delta 동치 근거와 한계는 CX-05와 같다. 실제 운영/최신 출시 UI라고 주장하지 않는다.

안내 기대값은 동일 후보 계열의 `apps/observer/web/src/service/api.ts:16–26,84–89` 고정 한국어와 `GameScreen.tsx:392–394`의 오류 표시/다시 시도 버튼을 읽어 정했다. `ws-launch-a-1017`의 해당 파일을 읽었으며 제품 원본을 수정하지 않았다. 실제 검증 대상은 manifest로 고정한 JS/CSS다.

| 선택 사례 (각390×900) | 기대 안내/동작 | 결과 |
|---|---|---|
| 401 `access_required` | 접근 확인이 필요해요. 페이지를 새로고침해 다시 접속해 주세요. | PASS,6 GET |
| 503 `maintenance` | 서비스 점검 중이에요. 잠시 후 다시 확인해 주세요. | PASS,6 GET |
| 503 `data_hold` | 자료를 확인하고 있어요. 확인이 끝나면 다시 보여 드려요. | PASS,6 GET |
| 503 임의 string id | 서비스 안내로 자료 제공이 잠시 중단됐어요. 잠시 후 다시 확인해 주세요. | PASS,6 GET |
| 503→수동 다시 시도→200 | 점검 안내 후 종료 화면, alert 없음, 합성 game SHA 소비 | PASS,8 GET |

각 오류는 game/health에 순수 합성 JSON만 반환한다. top-level message/detail 및 notice message/detail에는 동일 sentinel을 넣고 화면 전체에서 비노출을 검사한다. 초기 오류에서 추천 행0을 확인한다. 정상 final fixture의 pitches/next가 비어 있으므로 **이전 추천 철회, live 상태 전환, 추천 품질 검증이 아니다**. 회복은 버튼 클릭 및 이후200 game bytes 일치로 확인한다.

## 실행 경계와 명령

4자산/manifest/source pin 검사가 모든 네트워크 시작보다 먼저 수행된다. symlink/탈출/누락/변조 거부 controls, 새 output 배타 생성, exact origin GET allowlist, SW/WebSocket 차단,60요청 상한,80초 실행/90초 전체 경계와 소유 BrowserServer/HTTP 정리를 보존했다. loopback HTTP만 사용하며 TLS·실호스트·접속 자격·실자료/private/model/운영 로그는 읽거나 실행하지 않는다. root 수용 후 선택 suite 1회만 수행한다. 기본3개는 이번에 재실행하지 않으며 CX-05의 이전 결과와 구분한다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/MANIFEST.json \
  7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/public-build \
  /Users/song/Projects/pitcheezy/apps/observer/web/node_modules/playwright/index.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX08/run-001 \
  --host-errors
```

## 결과와 인계

실행기 SHA `8aebd3d6bab427b30a4d1b0d8ce514d35b35894fcc76defdc09332d13ab3e977`. 위 명령1회 exit0, `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`;5사례6+6+6+6+8=32 GET, routedAttempts=32, elapsedMs=1223, blocked=[]/pageErrors=[], ownedBrowserClosed=true/ownedServerClosed=true. 기본 suite 재실행0회. 같은 모델의 별도 에이전트 `coordination_review`가 실행기/receipt SHA와 PNG5개(access/data_hold/unknown/retry 전후)를 독립 대조해 PASS했고 root가 최종 수용했다. 검토자의 추가 브라우저 실행은0회다. 영수증은 사례별 기대 문구·오류 body SHA·제공 요청 SHA·추천 행0·sentinel 비노출·회복 여부/소비 fixture SHA·PNG 경로를 기록한다.

높이900의 합성 smoke이며390×844 문제 해결·실기기·운영 배포 GO·실호스트 인증/회수 동작을 입증하지 않는다. 완료 후 COORD-04가 정확2파일만 선별 커밋한다.


산출물 root: `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX08/run-001`.
- receipt.json SHA `09fea6d8f1d08688b6d2bdbe05239aa39fea044457cf8719f51a985ce6c89053`.
- synthetic-files.json SHA `278dd95a0738d18e7b0b040eeeaf02755267798b6f3add1c18c14b0ab51eab46`.
- PNG6개: `access_required-390.png`, `maintenance-390.png`, `data_hold-390.png`, `unknown_notice-390.png`, `retry_final-390.png`, `retry_final-recovered-390.png`. 작성자가 모두 직접 확인했다. 오류4종·점검→종료 화면이 기대와 일치한다.
- 회복 시 health/game 각200; 소비 game SHA `13692f0be6ea6bb97590f688ac3ce9b2fd9037cb887bcb239f77358823afeff9`로 생성 fixture와 같다. 오류 사례별 body SHA와 제공 요청은 receipt에 보존했다.

graphify: `tests/coordination`에서 `graphify update .` 1회 exit0, AST109 nodes/164 edges/8 communities. 기존4source만 갱신했고 ignored `graphify-out/`은 비커밋이다. 전체 저장소/의미 추출은 실행하지 않았다.

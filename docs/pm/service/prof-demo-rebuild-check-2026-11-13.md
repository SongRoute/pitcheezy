# 지금 화면 재빌드 대조 — 2026-11-13

상위: [전달 기록](exhibition-delivery-2026-11-01.md) · 작업 기록: #37 `PROF-DEMO-PREP-20261113`

목적: 11/14에 팀원의 색·3D 타자를 우리 빌드에 넣기 전에, 손대지 않은 main에서 지금 전시 화면이 그대로 다시 만들어지는지 확인한다. 통합 뒤 달라진 것이 통합 때문인지 가릴 기준이다.

## 1. 재빌드

- 일시: 2026-11-13 17:5x KST, 맥미니. 작성·실행: Claude PM(Fable 5.2).
- worktree: `pitcheezy-worktrees/prof-demo-1117`, 브랜치 `demo/prof-demo-1117`, 기준 main `a7a13c9a`. 소스 변경 없음.
- `git diff c47fe6cb main -- apps/observer/web`: 차이 없음(전달 후보를 만든 커밋과 화면 소스가 같다).
- 도구: node v25.9.0, npm 11.12.1(전달 묶음 `TOOLS.json`과 같음). `package-lock.json` SHA-256 `e0f2008accd1cbe90365871bed5529e1ffcf981f67d47148622568ea143937b9`(같음).
- 명령: `cd apps/observer/web && npm ci --no-audit --no-fund && npm run build:public`. 셸 VITE 변수 없음, `.env.public` 그대로.
- 11/1과 다른 점: 11/1은 기존 `node_modules`를 다시 썼고, 이번에는 새 worktree라 `npm ci`로 71개 패키지를 새로 설치했다(85 MB).

## 2. 대조 결과

대조 대상: `pitcheezy-pm-jobs/codex-exhibition-delivery-20261101/bundle-849843-main-c47fe6cb-20261101.zip`. 이 파일의 SHA-256을 다시 계산해 `3c3b8006b892872bfff3423e2bd6ba49ea3de17ee92fbecef07e0b076bc8da51`(182,517바이트)임을 확인했다.

| 파일 | 바이트 | SHA-256(raw) | 묶음 `MANIFEST.tsv`와 |
|---|---|---|---|
| `index.html` | 659 | `85616c4b2f49c5f9f0252bc9a687606fbbe6ce2e54253424bc956803db5a4289` | 같음 |
| `favicon.svg` | 249 | `bbfd72e5d077bd545b2cf568e5267c6637598d5253746f5b395e0d5f3fb144d9` | 같음 |
| `assets/index-ks_hu7Ji.css` | 118,449 | `292dd392a798d3ee52bd752857edeba50810f19eda4bbf234d4399b106e3bfc0` | 같음 |
| `assets/index-X1sSTTQ4.js` | 300,041 | `e46d431a7495bf58efd0084b20af87642fea8e6544cf3ac76ef278223621c30c` | 같음 |

화면 파일 4개가 모두 바이트까지 같다. 공개 JSON 6개(`site/v1/…`)는 다시 만들지 않았다. 11/1에도 이전 검증본을 원본 바이트 그대로 썼다.

정정: 이전 메모에서 화면 파일을 "index.html과 assets 2개"로 적었는데, `favicon.svg`를 더해 4개다.

## 3. 범위 밖

- 호스트에 올라간 파일과의 대조는 하지 않았다(접근 코드를 받은 뒤 한다).
- 발행기 재생, 폰 실기기, 영상은 다루지 않았다.

## 4. 기존 화면 검사 기준선

소스를 고치지 않은 같은 worktree에서 기존 검사 5개를 돌렸다. 2026-11-13 17:52~18:08 KST. 실행: 하위 작업자(`sonnet`으로 요청, 응답 모델은 따로 검증하지 않음). 최종 집계 줄은 Claude PM이 로그에서 직접 다시 읽었다. 명령은 모두 `apps/observer/web`에서 인자 없이 실행했다.

| 검사 | 명령 | 종료 | 집계 | 걸린 시간 | 10/31 기록과 |
|---|---|---|---|---|---|
| notice-hide-smoke | `node tests/notice-hide-smoke.mjs` | 0 | 267 ok / 0 FAIL | 93초 | 같음(267) |
| ws-launch-layout-check | `node tests/ws-launch-layout-check.mjs` | 0 | 360 passed, 0 failed | 92초 | 같음(360) |
| zone-geometry-check | `node tests/zone-geometry-check.mjs` | 0 | 397 ok, 0 FAIL | 20초 | 같음(397) |
| zone-label-clip-check | `node tests/zone-label-clip-check.mjs` | 0 | 160 ok, 0 FAIL | 31초 | 같음(160) |
| static-publish-smoke | `node tests/static-publish-smoke.mjs` | 0 | 416 passed, 0 failed | 673초 | **다름**(10/31은 1~3절 64) |

static-publish-smoke가 다른 이유: 10/31 기록의 64는 옛 커밋(`32e79c2f`)에서 1~3절만 센 값이고 4절 이후는 그때 UI-1026으로 옮기지 않은 상태였다. 그 뒤 `790f1fbe`(검사를 UI-1026으로 포팅)와 `5e44e69e`(검토 지적 반영, 검사 추가)가 main에 들어갔다. 지금 main의 이 검사는 핵심 검사 47건과 브라우저 장면을 합쳐 416건이다. main 기준의 이전 집계 기록은 찾지 못했다. **11/14 통합 뒤 비교할 기준은 이 416건이다.**

실행 중 생긴 것:

- 검사 4개가 API 모드 빌드(`dist/`)를 요구해 `npm run build`를 1회 돌렸다(종료 0). `dist/`는 git이 무시한다(424K).
- static-publish-smoke가 스스로 임시 빌드 폴더 4개(`dist-public-baseline`·`-quick`·`-regression`·`-slow`, 합 약 1.7 MB)를 만들었다. git이 무시하는 폴더이고 그대로 두었다.
- 검사 뒤 남은 서버 없음. 추적 파일 변경 없음. 로그는 저장소 밖 임시 폴더(84K)에 있으며 보존 대상이 아니다.

돌리지 않은 것: `ws-gaps-smoke`, `ws-moments-smoke`, `ui-review-followups-check`, `ui-preview`, `ws-host-errors-smoke`. 11/14 통합 전에 필요한 것을 골라 같은 방식으로 기준선을 더 잡는다.

한계: 이 검사들은 헤드리스 브라우저의 자동 검사다. 실제 폰, 호스트, 접근 코드를 켠 상태, 3D 타자는 다루지 않는다.

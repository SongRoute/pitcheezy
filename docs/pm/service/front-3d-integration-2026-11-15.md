# 팀원 색 테마·3D 타자 통합 기록 — FRONT-INTEG-20261115 (2026-11-15)

상위: 인계 문서 `docs/pm/integration/handoffs/2026-11-15-session-survey-and-todo.md` 5·7·8·9절(본 체크아웃에만 있음, 미커밋) · [UI 확정 기준](../../handoffs/UI-design-1026.md) · 통합 전 기준선 `pitcheezy-worktrees/prof-demo-1117/docs/pm/service/prof-demo-rebuild-check-2026-11-13.md` · 작업 큐 #37 CLAIM `FRONT-INTEG-20261115`(PM 작성)

- 작성·실행: Claude Code 하위 세션(요청 모델 Opus 5.5). **독립 검토 전**(PM Fable이 한다). push·병합·배포·전달 없음.
- 작업 트리 `~/Projects/pitcheezy-worktrees/front-3d-1115`, 가지 `demo/front-3d-1115`, 기준 main `a7a13c9a`.
- 작업 폴더(저장소 밖) `~/Projects/pitcheezy-pm-jobs/20261115-front/`: `teammate-src/`(받은 원본), `logs/`, `bundles/`, `bin/`(검사·묶음 스크립트).
- 방식: Song 결정 "가"(팀원 방식 그대로 + 수정). 우리 화면 소스(`src/service`, `src/v3`, 경기 목록)는 **한 줄도 바꾸지 않았다**. 빌드 결과물의 앱 JS·CSS는 c47과 바이트까지 같다.

## 1. 받은 파일과 해시 대조

출처: `Pitcheezy/transition-models` 가지 `claude/exhibit-records-1112` 커밋 `10988ce65f5e3ecd33363b3f960fccbdb5abd9fc`, `gh api`(raw)로 받음.
README 표(팀원 `relay/tools/songskin/README.md`)와 다시 계산한 SHA-256을 대조했다. **불일치 0.** 로그: `logs/received-sha256.tsv`.

| 파일 | 크기 | SHA-256 | README 표 |
|---|---:|---|---|
| `songskin/index.html` | 926 | `66c51841…6fdc1bde`(전체: `66c518417f29409a348cbb06a65266916766f4889dc4512df4a9b704ac6d1bde`) | 같음 |
| `songskin/pz-theme.css` | 5,483 | `94b79728cfdf2a2df1cedb28319e0e8ca813c69b216a907d225a7939f6fdc463` | 같음 |
| `songskin/pz-batter.js` | 12,006 | `728aa0a43d4e41fbb0f4d8aa9c92d9ad80893dea09f19006d3ffb0d9889b91cb` | 같음 |
| `batter3d/batter3d.js` | 36,440 | `66a3664c954e9ba33e449b7868300642fa1853e9c29b1a88284e102f3e8b75d0` | 같음 |
| `vendor/three.module.js` | 658,900 | `104f56145899722094556490d787aae0617cc2fc3382f31df02d724ea1aee341` | 같음 |
| `vendor/three.core.js` | 1,474,799 | `613e370f76638fdef83926642bf256dc9615684922de1ce1f63a8583e5d78578` | 같음 |
| `vendor/addons/loaders/GLTFLoader.js` | 119,861 | `532ac726e120039acbcd6cf54328c57acbf7b438176afaa6951257126d11bff1` | 같음 |
| `vendor/addons/utils/BufferGeometryUtils.js` | 39,153 | `7bda2a8ba836b2bcf67cb938d4b9e1bb5d1b1bc616817103ed858c0b58296ce7` | 같음 |
| `vendor/addons/utils/SkeletonUtils.js` | 11,535 | `b1632a703206c3d830de9fcbe515696770d04b71a15ee6b50afa6d2c3298c86f` | 같음 |
| `songskin/theme-only/index.html` | 712 | `b50993b505bfa78d5759c7c65706beca678f73ebfd084395fe3acaeb5d50048b` | 같음(theme-only README) |
| `songskin/theme-only/pz/pz-theme.css` | 4,640 | `4badfb14d9661c46573dc63c701240cc4c8a92edb6a55f797ddf0e6274208fd3` | 같음(theme-only README) |
| `batter3d/assets/batter-indexed.meta.json` | 4,694 | `e55fc80717d23e64f318436e5e303ca30b0d2355f561ce2e3debed00f64ae020` | 같음(모델 표). 저장소·묶음에 넣지 않음 |
| `vendor/LICENSE` | 1,081 | `8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc` | 표에 없음(npm 원본과 같음) |

- **three.js 공식본 대조:** `npm pack three@0.187.0` tarball의 SHA-512가 레지스트리 `dist.integrity`(`sha512-Qiu7R+Cm…WY8etA==`)와 같음을 확인하고, 그 안의 `build/three.module.js`, `build/three.core.js`, `examples/jsm/loaders/GLTFLoader.js`, `examples/jsm/utils/BufferGeometryUtils.js`, `examples/jsm/utils/SkeletonUtils.js`, `LICENSE` 6개가 팀원 `vendor/` 파일과 **바이트까지 같다**(`logs/three-npm-compare.tsv`).
- `__MODEL__` 치환: 원본 `pz-batter.js`에 빌드 치환을 적용하면 `batter-ctl` 12,007 B `795c7abd…`, `batter-indexed` 12,011 B `6f0fe56a…`로 README 값과 같다(치환 방식이 팀원과 같음을 확인. 실제 묶음의 파일은 수정 ① 뒤라 해시가 다르다 — 5절).
- 받지 않은 것: 3D 모델(`.glb`)과 사람형 설명 파일(`batter-ctl.meta.json`)은 팀원 저장소에 없다(비공개 전달 대기).

## 2. 코드 읽은 결과 (실행 전, 받은 파일은 신뢰하지 않는 자료로 취급)

`pz-batter.js`(153줄), `pz-theme.css`(69줄), `theme-only/pz/pz-theme.css`, `batter3d.js`(412줄), 두 `index.html`을 끝까지 읽었다. three.js 본체·로더 5개는 npm 공식본과 바이트가 같아 따로 읽지 않았다. 문자열 검색(`fetch`·`XMLHttpRequest`·`WebSocket`·`sendBeacon`·`eval`·`new Function`·`import(`·스크립트 삽입·`innerHTML`·`localStorage`·`sessionStorage`·`indexedDB`·`cookie`·`postMessage`·`http(s)://`·`navigator.`·`Worker`) 결과: `logs/code-scan-grep.txt`.

- **외부 통신:** 없음. 네트워크 요청은 같은 출처의 상대 주소 두 개뿐: `batter3d.js` 69행 `fetch(<assets>/<모델>.meta.json, {cache: "no-store"})`, 70행 `GLTFLoader.loadAsync(<assets>/<모델>.glb)`. 주소는 `new URL("./batter3d/assets/", import.meta.url)`(모듈 옆).
- **eval·동적 코드·스크립트 삽입:** 없음. `innerHTML` 없음. DOM 변경은 `div.pz-bat3d` 하나 넣기, `.sv-zrow`에 `pz-has3d` 클래스, 실루엣 `svg`에 인라인 `visibility`뿐(팀원 README 설명과 같다).
- **저장소 접근:** 없음(localStorage·쿠키·IndexedDB 안 씀).
- **전역:** `window.PZSongSkin`(동결 객체, 상태 조회·`kindFromActual`만).
- **기타:** `batter3d.js`의 유니폼 색 바꾸기(`setUniform`, 캔버스 2D 픽셀 처리)는 `pz-batter.js`가 부르지 않는다. 사람형 모델의 WebP 그림은 three.js `GLTFLoader`가 `blob:` 주소로 읽는다(공식본 동작).
- **재시도:** 모델·초기화 실패는 페이지당 3번까지(`S.fails >= 3`이면 멈춤). WebGL 문맥 상실도 실패로 센다.
- **CSS:** 크기·간격 속성은 3D 상자 규칙(`.pz-bat3d`)뿐이고 나머지는 색이다. `@import`·`url()` 없음.
- 판단: 실행해도 되는 코드. 위험 요소 없음. 단 "다른 모델 검토"는 아직 아니다(팀원 소스도, 이번 수정도).

## 3. 넣은 방식 (빌드에서 재현)

- 팀원 파일을 `apps/observer/web/pz-skin/`에 원본 바이트로 먼저 커밋(`2802f5b8`)하고, 수정은 따로 커밋했다.
- `pz-skin/plugin.mjs`(Vite 빌드 플러그인, `vite.config.ts`에 등록)가 `npm run build:public` 때 빌드 결과물 옆에 `pz/`를 만들고 index.html에 태그 세 가지를 넣는다: ① 첫 모듈 스크립트 앞 import map, ② 앱 CSS 뒤 `pz/pz-theme.css` 링크, ③ 모듈 스크립트(수정 ③ 때문에 `pz-batter.js` 대신 작은 `pz/pz-boot.js`). `theme-color`는 `#091811`. 손으로 dist를 고치지 않는다.
- three.js는 `devDependencies`의 `three@0.187.0`(정확한 버전, package-lock 무결성 고정)에서 복사하고, 복사할 때마다 팀원 표의 SHA-256과 대조한다(다르면 빌드 실패). `vendor/` 2.3 MB를 저장소에 넣지 않았다.
- 변수: `PZ_SKIN=3d`(기본)|`theme`|`off`, `PZ_BATTER_MODEL=batter-ctl|batter-indexed`(비우면 `__MODEL__` 그대로 → 404 → 실루엣), `PZ_BATTER_ASSETS_DIR`(저장소 밖 모델 폴더, 주면 `pz/batter3d/assets/`로 복사). 모델 이름은 `^[a-z0-9][a-z0-9-]{0,63}$`만 받는다.
- 재현 확인:
  - `PZ_SKIN=off` 빌드 = c47 화면 파일 4개와 바이트까지 같다(index.html `85616c4b…`, JS `e46d431a…`, CSS `292dd392…`, favicon `bbfd72e5…`).
  - `PZ_SKIN=theme` 빌드의 index.html = 팀원 `theme-only/index.html`과 바이트까지 같다(`b50993b5…`, 712 B).
  - `PZ_SKIN=3d` 빌드의 앱 JS·CSS도 c47과 같은 이름·바이트다. index.html(925 B)은 팀원 926 B 파일과 스크립트 이름(`pz-boot.js`)·빈 줄 하나가 다르다.
- `npm run build`(API 모드 `dist/`)에도 같은 플러그인이 붙는다(기존 검사가 통합된 화면을 보게 하려고). `vite` 개발 서버에는 붙지 않는다.

## 4. 고친 것 4건

| # | 무엇 | 파일 | 원본 SHA-256 → 바뀐 뒤 | 커밋 |
|---|---|---|---|---|
| ① | 탭 복귀·재연결 때 같은 공 동작 재생 | `pz-batter.js` | `728aa0a4…` (12,006 B) → `a62fdbb93110ca5f292784eb62f838b3c3e90abeb4f5b780feb12685b14282dc` (13,004 B) | `517f2b24` |
| ② | 타격 주황 대비 1.70:1 | `pz-theme.css`, `theme-only/pz-theme.css` | 아래 ④와 합친 최종값 | `5a93ef93` |
| ④ | 359 px 이하 가로 밀림·320 존 표식 가림 | `pz-theme.css` | `94b79728…` (5,483 B) → `6d9d3559ffde391973e475f0e955b67e2ea4cd986f3648e059ba7b8b77eaa8b2` (6,833 B) | `368d1b22` |
| ② | (색만 판) | `theme-only/pz-theme.css` | `4badfb14…` (4,640 B) → `156e42114ec339e0d8b32ce76f74567ef9286611e733ad62cb30a168219ce570` (5,096 B) | `5a93ef93` |
| ③ | 목록 화면에서 3D 코드 2.35 MB 받음 | 새 `pz-boot.js` + 빌드 | (새 파일) `54c385200e3a0703a6d2cce2ae6cc2a346ffe4515a084932041761ee6b37437c` (1,284 B) | `a155de3a` |

`batter3d.js`는 바꾸지 않았다(`66a3664c…` 그대로). 전체 diff: `git diff 2802f5b8 368d1b22 -- apps/observer/web/pz-skin/`.

- **① `tick()`:** 전에는 '실제 투구' 줄의 열쇠가 바뀌기만 하면 재생했다. 이제 (가) 결과 종류(`kindFromActual`)가 있는 줄이고 (나) 마지막으로 재생한(또는 붙을 때 본) 공(`playedKey`)과 다를 때만 재생한다. 결과 없는 줄('기록 없음'·'확인 중')은 `playedKey`를 바꾸지 않는다 → A → 빈 줄 → A는 재생 안 함, A → B → A(관람자가 다시 고름)는 재생. 붙을 때 줄이 비어 있었으면 처음 나타난 결과 줄은 기준으로만 삼는다.
- **② 색:** `--sv-g1-play`를 `#ffb782` → `#c4500f`(같은 주황 계열, 진하게). 타격 번호를 짙은 글자 → 흰 글자로(짙은 글자는 새 바탕에서 3.21~3.57:1이라 4.5 미달). 계산(WCAG 상대 휘도, `bin/contrast.mjs`):

  | 대상 | 팀원 `#ffb782` | 바꾼 `#c4500f` | 기준 |
  |---|---:|---:|---:|
  | 표식 vs 흰 바탕 `#ffffff` | 1.70 | **4.67** | 3 |
  | 표식 vs 선택 줄 `#eef8dc` | 1.55 | **4.24** | 3 |
  | 표식 vs 화면 바탕 `#eef3e8` | 1.51 | **4.14** | 3 |
  | 번호 글자 vs 표식 | 9.79(짙은 글자) | **4.67**(흰 글자) | 4.5 |
  | 표식 vs 스트라이크 `#485d4f`(구분용, 기준 없음) | 4.18 | 1.52 | — |

  같은 토큰을 쓰는 "방금 던진 공" 고리(`.sv-ring`, `.sv-g1.ring`)와 범례 점도 함께 진해졌다(브라우저 측정은 6절 새 검사). 스트라이크(짙은 녹회색)와는 밝기 차이가 줄었지만 색상(주황 대 녹회색)으로 구분된다 — 실제 폰에서 볼 항목에 넣었다.
- **④ 자리:** 359 px 이하에서 화면 좌우 여백이 18 px(360 이상은 24 px)라 바깥으로 22 px 내민 140×210 상자가 4 px 넘쳤다. 그 폭에서만 16 px로 줄였다(상자 크기 그대로). 그리고 존(`.sv-zone-wrap`, 원래 `position: relative`)에 `z-index: 2`를 주어 3D 상자(`z-index: 1`) 위에 그린다 → 타자가 존 쪽으로 들어와도 존 밖 표식·추천 네모를 가리지 않는다(존 상자는 채움이 없어 타자는 선 뒤로 비친다). `:has()`를 모르는 브라우저를 위해 `pz-has3d` 클래스 규칙을 따로 둔다. 측정(`logs/measure-bat3d-{before,after}.log`, 우타자 상자 오른쪽 끝): 320 → 324 → **318**, 344 → 348 → **342**, 359 → 363 → **357**, 360 이상은 그대로(358 등). 좌타자 왼쪽 끝 −4 → **2**.
- **③ 불러오기:** index.html의 정적 `pz-batter.js` 태그 대신 `pz-boot.js`(1.3 KB)가 `.sv-game .sv-zrow`가 처음 나타날 때 한 번 `import("./pz-batter.js")` 한다. 실패하면 다시 시도하지 않는다(실루엣 유지). import map 태그는 그대로 문서에 둔다(내려받기 없음).

## 5. 검사 집계

모두 맥미니, 헤드리스 Chromium(Playwright), 공용 브라우저 잠금(`20261115-night/browser.lock`) 아래 한 번에 하나. 로그 `logs/baseline/`, `logs/after/`, `logs/pz-skin-check-*.log`.

| 검사 | 통합 전(이 트리, main `a7a13c9a`) | 통합 뒤 | 11/13 기준선 |
|---|---|---|---|
| zone-geometry-check | 397 ok, 0 FAIL | 397 ok, 0 FAIL | 397 |
| zone-label-clip-check | 160 ok, 0 FAIL | 160 ok, 0 FAIL | 160 |
| notice-hide-smoke | 267 ok / 0 FAIL | 267 ok / 0 FAIL | 267 |
| ws-launch-layout-check | 360 passed, 0 failed | 360 passed, 0 failed | 360 |
| static-publish-smoke | 돌리지 않음(11/13 기록 416 사용, 시간 절약) | 1회차 415 passed, **1 failed** → 허용 목록에 `pz/` 고정 이름 추가(`92913f20`) → **416 passed, 0 failed** | 416 |
| pz-skin-check(새) · dist-public | — | 158 ok, 0 FAIL | — |
| pz-skin-check · 묶음 A 사람형 / A 마네킹 / B 색만 | — | 158 / 158 / 57 ok, 0 FAIL | — |

- static-publish-smoke 1회차의 실패 1건: `inject: every request stays inside the prefix and the published layout`. 조작한 `?date=`·`?game=` 주소로 열어도 모든 요청이 발행 자리 안에 있어야 하는 검사인데, 목록 화면이 새로 받는 `pz/pz-theme.css`·`pz/pz-boot.js`를 몰랐다. 기준은 그대로 두고 허용 정규식에 이 묶음의 고정 파일 이름만 더했다(질의 문자열 금지 유지). 확인: `/pub/pz/../api/x`, `/pub/pz/evil.js`, `/pub/pz/pz-theme.css?x=1`, `/api/x`는 여전히 거부. 2회차 같은 줄 `(117 requests checked)` 통과.
- 걸린 시간(초): zone-geometry 22, zone-label-clip 32, notice-hide 92, ws-launch 93, static-publish 683/675, pz-skin-check 49~50(B 16).
- 기존 검사 4종(notice-hide·ws-launch·zone 둘)은 고칠 것 없이 같은 수로 통과했다. 이 검사들은 3D가 404로 실루엣이 남는 상태(모델 없음)의 통합 화면을 본 것이다.

## 6. 새 검사 `tests/pz-skin-check.mjs`

가짜 정적 호스트(127.0.0.1:8884, 8880~8889 범위)와 `tests/fixtures/publish/v1`(849843 종료 경기, 990003 합성 라이브 경기). 다른 출처 요청은 모두 막는다. 3D 빌드(`PZ_SKIN=3d`)와 색만 빌드(`PZ_SKIN=theme`)를 index.html로 알아보고 각각 맞는 항목만 본다. 손대지 않은 화면과 비교하려고 검사 안에서 `PZ_SKIN=off` 빌드(`dist-public-plain/`, git 무시)를 만든다.

- 모델 404(지금 상태): 실루엣 보임, 3D 상자·`pz-has3d` 남지 않음, 오류 0, 모델 요청 ≤ 3이고 `online` 재확인 3번 뒤에도 늘지 않음, 가로 넘침 0(390·320).
- 가짜 모델(검사가 만든 삼각형 하나짜리 glb, 동작 `idle`·`contact_hit1`·`miss`·`take`·`hit`): 실제 three.js + SwiftShader WebGL로 붙음, 실루엣 숨김, 첫 화면 동작 0, ① `online` 재확인 4번 + 다른 탭 앞에 두었다 돌아오기 뒤 재생 수 그대로, 다른 공 고르면 반응, 목록으로 가면 상자 해제.
- ① 합성 페이지(같은 `pz-batter.js`): A→기록 없음→A, A→확인 중→A 재생 0 / A→B 1 / B→빈 줄→B 그대로 / B→A 2 / 좌우 바뀜 따라감.
- ③ 목록 화면 요청에 `pz-batter.js`·`batter3d/`·three.js 없음, 목록→경기에서 `pz-batter.js` 1번·three 본체 받음, 외부 요청 0.
- ④ 320·344·359·360·390·402, 좌우 둘 다: 상자가 화면 안, 크기 140×210 그대로, 존 z-index > 상자, 상자를 빨갛게 칠했을 때 겹치는 곳의 존 테두리 픽셀이 빨강이 아님.
- ② 토큰 대비(3개 바탕), 페이지가 계산한 목록 표식·번호·고리·범례·존 표식 대비.
- 8절 보이기·숨기기(402×681, 360×620 × 종료·라이브 경기): "이번 타석" 접힘, 하단 메뉴 4개(추천·선수·구종·기록), 스위치 0, 메인의 ‹ › 0, 안내·V4 낱말 없음, 가로 넘침 0, **보이는 글자가 손대지 않은 화면과 같음**, 세로 스크롤이 손대지 않은 화면보다 늘지 않음.
- 감속 모션: 붙되 대기 반복 없음.
- 만들면서 고친 검사 자체의 잘못(1회차 122 ok / 12 FAIL, `logs/pz-skin-check-try1.log`): 테마 주석에 남은 `#ffb782`까지 센 것, 가짜 모델에 없는 `take` 동작을 쓰는 볼 공으로 재생을 센 것, 존 테두리 픽셀을 선 바깥 0.5 px에서 잰 것(스크린숏으로 존이 상자 위에 그려짐을 확인), 세로 스크롤을 절대 기준으로만 본 것(7절). 화면 코드는 이 때문에 바꾸지 않았다.

## 커밋 (가지 `demo/front-3d-1115`, 로컬만)

| 커밋 | 내용 |
|---|---|
| `2802f5b8` | 팀원 원본 4개를 바이트 그대로 |
| `517f2b24` | 수정 ① |
| `5a93ef93` | 수정 ② |
| `368d1b22` | 수정 ④ |
| `a155de3a` | 빌드 플러그인 + 수정 ③ + `three@0.187.0` |
| `54aa71c5`, `c87b8c87` | 새 검사 `pz-skin-check` |
| `92913f20` | static-publish-smoke 허용 목록 |
| (이 문서) | 기록 |

## 7. 보이기·숨기기 점검 결과 (인계 8절)

- 통과: "이번 타석" 기본 접힘, 하단 메뉴 4개, 화면 안 안내 문장·스위치·이전/다음 버튼 없음, 가로 넘침 0, 보이는 글자 c47과 같음.
- **통과 못 함(통합 전부터): 첫 화면 세로 스크롤 0 px.** 손대지 않은 c47 화면(`PZ_SKIN=off`)에서도 같은 값이다 — 통합이 늘린 것은 0 px.

  | 경기 | 402×681 | 360×620 |
  |---|---|---|
  | 849843(종료) | 22 px, "이번 타석" 줄 아래끝 619 > 하단 바 위끝 617 | 83 px, 619 > 556 |
  | 990003(라이브, 합성) | 46 px, 643 > 617 | 107 px, 643 > 556 |

  360×620에서는 추천 줄도 하단 바에 반쯤 가린다(스크린숏 `logs/shots-*/showhide-*-990003-360x620.png`). 기존 검사(ws-launch-layout-check)는 390×844에서만 "이번 타석 줄이 하단 바에 안 가림"을 본다. 고치지 않았다(이번 범위 4건 밖, 우리 화면 소스 변경) → 9절 "Song이 정할 것".

## 8. 묶음

`~/Projects/pitcheezy-pm-jobs/20261115-front/bundles/` — 각 폴더에 `site/`(빌드 + `v1/`), `MANIFEST.tsv`(path, bytes, sha256), `PROVENANCE.txt`, 3D 판은 `MODEL-FILES.txt`(모델 놓는 자리·기대 해시). `site/v1/` 6개는 11/1 c47 전달 묶음의 `site/v1`과 바이트까지 같다. ZIP은 `zip -X`, 파일 시각 고정.

| 묶음 | 내용 | 파일 수(site) | ZIP 바이트 · SHA-256 |
|---|---|---:|---|
| `A-color-3d-batter-ctl` | 색 + 3D, `MODEL = "batter-ctl"`(사람형) | 20 | 662,619 · `b9f0dedc69e70b91550f333bb6e5b73e48f5c248efa1ab31e864e5179699b51e` |
| `A-color-3d-batter-indexed` | 색 + 3D, `MODEL = "batter-indexed"`(마네킹) | 20 | 662,787 · `53e17f236ea87eb24c63169f85a9fa40c022889dea8789da358eea88b5fe27bd` |
| `B-color-only` | 색만(물러설 자리, ②만 해당) | 11 | 178,117 · `511dc34c4dd32278cccab648079897bcfbd28efc3d75c2e73425b3790e4ed41b` |

주요 파일(전체는 각 `MANIFEST.tsv`):

| 파일 | A 사람형 | A 마네킹 | B |
|---|---|---|---|
| `index.html` | 925 B `f97e2dec…547976c` | 같음 | 712 B `b50993b5…50048b`(= 팀원 theme-only) |
| `pz/pz-theme.css` | 6,833 B `6d9d3559…7eaa8b2` | 같음 | 5,096 B `156e4211…19ce570` |
| `pz/pz-batter.js` | 13,005 B `dc3478b2…ce62e1826` | 13,009 B `0407ea98…a4cd2835` | 없음 |
| `pz/pz-boot.js` | 1,284 B `54c38520…6b37437c` | 같음 | 없음 |
| `assets/index-X1sSTTQ4.js`, `assets/index-ks_hu7Ji.css`, `favicon.svg` | c47과 같음 | 같음 | 같음 |

- A 두 판은 `pz/pz-batter.js` 한 파일만 다르다(MANIFEST 비교). `site/pz/batter3d/assets/`는 비어 있다 — 모델·설명 파일은 받은 뒤 `MODEL-FILES.txt`대로 넣는다(팀원 표의 크기·해시를 적어 둠).
- 만든 방법: `bin/make-bundles.sh`(작업 트리가 깨끗할 때만 돈다). 최종 묶음의 출처 커밋은 `d5776199`(각 `PROVENANCE.txt`). 같은 스크립트로 `54aa71c5`에서 만든 첫 묶음과 `site/` 파일이 바이트까지 같다(세 MANIFEST 일치). ZIP 해시는 `PROVENANCE.txt`(커밋·만든 시각)가 들어 있어 만들 때마다 바뀐다 — 비교는 MANIFEST로 한다. pz-skin-check는 첫 묶음(같은 `site/`)에서 돌렸다.

폰 확인 목록: `bundles/PHONE-CHECKLIST.txt`(아이폰 사파리·갤럭시 16항목).

## 9. 못 한 것 · 가정 · Song이 정할 것

### 모델을 받으면 할 것
1. `PZ_BATTER_MODEL=<이름> PZ_BATTER_ASSETS_DIR=<저장소 밖 폴더> npm run build:public`로 묶음을 다시 만들고 모델·설명 파일 해시를 팀원 표와 대조.
2. `pz-skin-check`를 실제 모델로(가짜 glb 대신) — 붙음, 반응 종류(take·hit 포함), 좌우, 4개 폭의 그림 겹침을 눈으로.
3. 사람형: 설명 파일 없을 때 실루엣으로 남는지, WebP 그림, 대기 중 움직임(5절 7번), 30fps 대기 반복의 발열.
4. 실제 모델로 ① 탭 복귀(실제 폰 앱 전환·화면 잠금 포함), ④ 320에서 타자 그림이 존 선 뒤로 비치는 모습.
5. 팀원 알려진 문제 5·6(확인 중인데 움직임, 재생 중 끝까지 감)·7(WebGL 없어도 모델 먼저 요청)은 이번 범위 밖이라 그대로다.

### 못 한 것
- 실제 폰 확인 없음(목록만 만듦). 실제 모델로 3D를 띄워 보지 못함.
- 느린 망에서 경기 화면이 뜨는 시간은 재지 않았다(③은 "목록에서 받지 않음"만 증명).
- 다른 모델의 코드 검토 없음(PM이 한다).

### 가정
- 기존 검사 4종은 원래대로 OS가 고른 루프백 포트(port 0)를 쓴다. 새 검사와 내 측정만 8880~8889를 쓴다(기존 검사를 바꾸면 기준선이 흔들려서).
- API 모드 빌드(`npm run build`)에도 색·3D를 붙였다(기존 검사가 통합 화면을 보게). 로컬 운영 화면도 같이 바뀐다.
- `__MODEL__`이 비어 있는 기본 빌드는 팀원과 같이 `__MODEL__.glb`를 요청한다(404 → 실루엣).
- 붙을 때 '실제 투구' 줄이 비어 있으면 처음 나타난 결과는 재생하지 않는다(첫 화면에서 움직이지 않게).
- 타격 주황은 `#c4500f`로 정했다(같은 계열 중 흰 번호 4.5:1을 넘는 가장 밝은 쪽 근처).

### Song이 정할 것
1. 첫 화면 세로 스크롤(7절): c47부터 402×681·360×620에서 스크롤이 생기고 "이번 타석" 줄이 하단 바에 가린다. 데모 전에 우리 화면 배치를 고칠지(존 높이 등), 그대로 둘지.
2. 모델 선택(사람형/마네킹)과 사람형 대기 움직임(인계 5절 7번).
3. 타격 주황 `#c4500f`가 팀원 색 의도에 맞는지(눈으로).

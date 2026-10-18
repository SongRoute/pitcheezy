# CX35 — 경기 이동과 응답 identity의 합성 검사

- 작성: Codex implementation_1019, 2026-10-19 KST.
- 상태: 독립 실행 경계 및 결과·PNG 검토 PASS, Root 수용. schema 7body 및 신규 브라우저 1case PASS이며 별도 검토자의 재실행은 없다.
- 선점: 공동 작업큐 #37 단계1 CLAIM 6255499266, 단계2 CLAIM 6255518525, 실제 자동 재요청 범위 정정 CLAIM 6255534380.

## 실제 경로와 검사 목적

고정 제품 소스 `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 `main.tsx`는 문서 로드 시 game ID를 읽어 WatchList 또는 GameScreen을 만든다. WatchList 카드의 일반 anchor와 GameScreen의 목록 anchor는 `./?game=<id>`와 `./`로 문서를 이동한다. 같은 React 컴포넌트의 game prop을 교체하는 흐름으로 설명하지 않는다.

고정 `apps/observer/web/src/service/api.ts`의 `fetchPublishedDetail`(126–133행, 비교는 130행)은 요청 game ID와 정규화된 응답의 `detail.game.game_pk`를 대조하고 다르면 FetchError 404를 던진다. 기존 public-viewer-e2e, service-ui의 관련 인라인 검사와 현재 하네스 기본 검사는 목록에서 한 경기로 진입한다. 이 작업은 두 경기 사이를 이동하며 잘못 배달된 경기 본문을 거절하는 실제 UI 연결을 추가로 검사한다. 과거 service-ui의 외부 실제 경기 fixture는 읽거나 실행하지 않았다.

두 경기 모두 정상 `game_checks`, health/file/detail sequence 1을 유지했다. 첫 로드에서는 적용된 sequence가 없으므로 GameScreen 203–206행이 pending을 먼저 설정하고, identity 거절 후 247행에서 pending을 유지한다. 따라서 실제 기대는 오류 안내나 수동 버튼이 아니라 **‘확인 중’과 선수·추천 미표시**이다. 자동 갱신으로 올바른 응답을 받은 뒤 회복하는 현재 계약을 검증했으며 새 제품 결함으로 판정하지 않았다.

## 최소 변경과 입력

`tests/coordination/public_candidate_smoke.mjs`의 `--game-identity` 선택 분기 하나와 이 보고서만 추가·수정했다. 기존 정상 live fixture 생성기를 재사용하여 완전히 합성한 두 경기 990001/990002를 만들었다. A는 Synthetic Alpha 선수와 FF, B는 Synthetic Bravo 선수와 SL이다. B URL의 최초 응답만 schema-valid A 본문이며, 정상 B 본문도 같은 sequence이다. health의 두 game_checks와 60초 publish interval은 유지했다.

새 서버 지연이나 body 보류는 없다. 기존 allowlist의 B body를 교체하고 이 suite에서만 요청의 `performance.now()` 시각을 기록한다. 기존 함수와 검사 경로는 optional 추가를 역변환했을 때 이전 파일 전체 바이트와 동일하다. 문법 검사·diff-check와 tests/coordination 범위 graphify AST 갱신을 통과했다.

기존 CX14의 고정 5개 JSON Schema와 네트워크·외부 참조 차단 검증기를 재사용했다. 초기 응답 6개와 정상 B 복구 본문 1개를 대조해 **7body, invalid 0, schema error 0, network attempt 0, formats []**, exit 0을 확인했다. 브라우저 실행이 생성한 초기·복구 본문은 schema 검사 입력과 정확히 일치한다. schema-valid라는 결과는 잘못된 URL/body identity가 올바르다는 뜻이 아니다.

## 실제 1회 실행 결과

전용 CX35 출력의 `argv.json`에 고정 manifest/source/build/Playwright 인자와 새 `browser-001 --game-identity`를 기록했다. 390×844, 한 context/page에서 실제 anchor를 클릭해 **목록 → A → 목록 → B**, 문서 navigation은 총 4회였다.

| 단계 | 실제 관측 |
|---|---|
| A 진입 | Alpha 타자·투수, FF 추천 1행. Bravo 선수 없음 |
| B URL에 A 본문 200 | B URL 유지, ‘확인 중’, 선수 0·추천 0, 오류 alert 0. 실제 응답 SHA는 A와 동일 |
| 정상 B body 준비 후 자동 재요청 | 새 navigation 없이 B 선수와 SL 추천 1행만 표시. Alpha 선수 없음. 두 B 요청 모두 HTTP 200 |

실제 stdout:

```json
{"status":"PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP","scenarios":1,"calls":29,"ownedBrowserClosed":true,"ownedServerClosed":true}
```

exit 0, stderr 없음, 전체 **61109ms**, GET **29회**, blocked 0, page error 0이다. B 요청의 실제 monotonic 시각은 `1036.369333ms`와 `61057.691666ms`로 간격은 **60021.322333ms**였다. 정상 B 준비는 `1071.160875ms`였으며 복구 요청은 그 이후에 발생했다. 하네스는 실제 자동 60초 interval을 기다렸고 wake 이벤트·수동 재시도·reload·timer override를 사용하지 않았다.

HTTP Date 헤더와 fixture timestamp만 2030으로 고정했다. 브라우저 `Date.now`/`new Date`, `setInterval`, `performance.now`는 실제 동작 그대로다. 응답 대기는 70초, 요청 간격 55–70초는 검사 허용오차이며 제품 성능 기준이나 SLA가 아니다. 기존 실행 80초·전체 90초·60GET 상한을 유지했고 자체 브라우저와 서버를 닫았다. 하네스·manifest·4개 asset의 전후 핀이 모두 같다.

## 고정 증거

- public v4.3.1 manifest: `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`.
- 실행 하네스: `ec8722995868ce8b5bec4fd9f17c306efba60f3753fc87587b2e371257ce39ef`.
- `candidate-input.json`: `5b6ba23e556b5b068c89c3c9c7669a076ce3aa4684454335f59246d11e5f529c`.
- `schema-check-001.json`: `2db1c6187532b93177b9ae63e719f3a5e9e2c518624f6fb595ba8a01d2c4d097`.
- `browser-001/receipt.json`: `132b39e0e162441c1d7a647bbc6bc12bd118f0c1e2818228f885c869eb35615c`.
- `browser-001/live-phases.json`: `0ae32b900370c745936461692d084534f1dcf289bdf9324752013581f3434dab`.
- 오배달 A 본문: `04f43a89ac45bbd9889f64baae488030c15c770d96b63a842e8830ae4a960a00`.
- 정상 B 본문: `cc4e509d05457872a7a1132f0d692591877e4cfdd14ba17ef57b1a122f5e04f3`.

| 직접 열람한 PNG | SHA-256 |
|---|---|
| `game-identity-first-390.png` | `4cb2706a49bfaea0960727c93514a0e311d97d43fa2ca471ce3bdd4147d3fee8` |
| `game-identity-pending-390.png` | `315652f3b13f15d6eb45ee63c9b56001088156a44c14b281a71b5fd8e18431d6` |
| `game-identity-recovered-390.png` | `6b92a06ae203f7adcaecf0bcd7d7ae58759f23488305359756432df1edc21513` |

새 브라우저 검사는 1case이며 단계·schema 대조·기존 pin control을 별도 브라우저 case로 합산하지 않는다. 기존 suite는 재실행하지 않았다. Chromium·소유 loopback HTTP·합성 입력의 결과로서 실제 기기, 실호스트/TLS, 실제 경기 예측 성능 또는 운영 GO를 뜻하지 않는다. 제품 소스·원본 wrapper·실자료·설치·배포 변경은 없다.

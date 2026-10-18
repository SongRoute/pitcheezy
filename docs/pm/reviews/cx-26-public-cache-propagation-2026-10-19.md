# CX-26 — 조건부 캐시 응답과 전파 중 추천 숨김

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 독립 실행 경계·schema·최소 대기 수정·최종 결과·PNG 검토 PASS 및 Root 수용. 검토자의 추가 재실행 없음. Base `2b52902c45c906e0e63871b08a92822508889120`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254609048 .

## 범위와 검사

고정 공개 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 API는 ETag/304 캐시를 재사용하고 GameScreen은 health가 가리킨 새 file sequence보다 이전 game을 받으면 추천을 숨긴다. 이 경계를 `--cache-propagation` 1case/4phase로 검사했다. 제품 source·bundle·설정은 수정하지 않았다.

| 동일 page의 순서 | 실제 응답 / 조건부 요청 | DOM 기대 |
|---|---|---|
| initial_ff | health200 H1 / game200 G1, If-None-Match 없음 | 포심 추천1 / LIVE on |
| pending_file | health200 H2(file2), 요청 H1 / game304 G1, 요청 G1 | unconfirmed / 추천0 / LIVE off |
| recovered_sl | health304 H2, 요청 H2 / game200 G2, 요청 G1 | 슬라이더 추천1 / LIVE on |
| cached_sl | health304 H2, 요청 H2 / game GET0 | 슬라이더 추천1 유지 / LIVE on |

ETag는 각각 `"cx26-h1"`, `"cx26-h2"`, `"cx26-g1"`, `"cx26-g2"`다. navigation1/reload0을 유지하고 후속 phase는 수동 LIVE 버튼으로 시작한다. 서버 기록과 Playwright 실제 status를 각각 대조한다. 모든304는0bytes/빈 SHA이며 기존 Cache-Control no-store를 유지했다. 조건부 header 불일치는 서버 callback에서 throw하지 않고403/기존 budget rejection 경로로 소유 자원 정리를 보장한다. 기존 suite의200 기대는 그대로다.

기존4개 fixture 함수·partial case는 바이트 동일하며 pin/GET allowlist/외부·WS·SW 차단/소유 loopback/60GET·90초 상한과 cleanup을 보존했다. 기존 suite는 이번 작업에서 재실행하지 않았다.

생성20body(4phase×5schema, 중복 포함)를 고정5schema로 한 번 검사했다. `schema-001`: exit0,20대조/오류0/network attempt0, format 선언 없음. 이것은 날짜 의미·시계 정확도 검증이 아니며 실제304 빈 응답은 schema 검사 대상이 아니다. 생성 입력과 브라우저가 기록한 live-phases SHA는 모두 `4dc58482631f6dbb0f1177c2986041f0217cff0787fd4e808bd8a7772cb90de1`이다.

## 두 실행과 최소 수정

출력은 20261019-codex-continuous 전용 `CX26/` 아래에 보존했다. 기존 v4.3.1 manifest/public-build와 설치된 Playwright를 사용했다. Manifest SHA는 `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  PINNED_MANIFEST 7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f PINNED_PUBLIC_BUILD \
  INSTALLED_PLAYWRIGHT_INDEX NEW_CX26_BROWSER_OUTPUT --cache-propagation
```

`browser-001`은 하네스 SHA `50881e534c3fcc4fda4d4c634360b2d14774a5dd4612d34ba3d891e313502b34`에서 exit1/INCOMPLETE, 완료 case0,10GET 시도,60693ms였다. H2 200/G1 304까지 정상 제공한 뒤 약60초에 manifest 자동 polling이 발생했고 다음 health의 H2 조건부 header가 아직 phase2의 H1 기대와 달라 guard가 중단했다. 소유 browser/server cleanup은 모두 true, pin은 불변이다. 실패 stdout/stderr/receipt/PNG를 보존했다.

304의 `response.finished()` 대기가 진행을 막았다는 설명은 이 요청 순서·경과 시간과 최소 수정 후 결과에 부합하는 **추론**이다. 별도 Playwright 원인 재현을 수행하지 않았으며 제품 캐시 실패로 판정하지 않는다. 새 cache phase loop만 actual status를 즉시 검사하고 expected304에서는 body-finished 대기를 생략했다.200은10초 bounded finished를 기다린다. 서버304 empty bytes/SHA·header·DOM 기대는 모두 유지했다. Root와 독립 검토자가 이 최소 수정을 수용한 뒤 새 출력에서1회 재실행했다. fixture가 불변이므로 schema는 반복하지 않았다.

최종 하네스 SHA `aae0f3a4eab9e00a221b37c4893cd35bb7a3d91e6e00785744fd539a11235277`의 `browser-002`: **exit0/PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP**,1case/4phase,11GET(6/2/2/1),752ms. 실제 브라우저 상태도200/200 →200/304 →304/200 →304와 일치했다. page error0/금지 요청0, 소유 browser/server cleanup true. 하네스·manifest·4asset 전후 SHA 동일. 실패와 재시도를 두 개의 고유 검사로 합산하지 않는다.

3종 PNG를 작성자가 직접 열어 포심 추천, 전파 중 추천 숨김, 슬라이더 복구를 확인했다. 마지막 두 phase PNG는 동일하다. 복구 화면의 이전 포심 표시는 추천 변경 안내이며 현재 추천 행은 슬라이더다.

| 증거 | SHA-256 |
|---|---|
| schema-check-001.json | `4f245304c2575defcd68f93577b7b9274169a67378657566c97355eb3f26b889` |
| browser-001/receipt.json (실패 보존) | `204c1cecdb47713a51c4fd2d87e0a515b92c48f6918284dcb2b6c73d476b854a` |
| browser-002/receipt.json | `3f870a39cdec4d0581984a19406f6fce6acbb4aaebff05e78795866f44907b98` |
| initial_ff PNG | `287277cbf5c84a808048ef1d965b010bfaf0906b673c1301b2d67541f5a665be` |
| pending_file PNG | `38a9a7d0e503bffb6c64c5d17653ac636973501b663d525cb20cf83da7876045` |
| recovered_sl / cached_sl PNG | `a9147f40088e0bf32615b0982bf7acef3b549fcbf318406766d1e6e9c4c1f48b` |

`node --check`와 `git diff --check` 통과. 최종 코드에서 `tests/coordination` scoped AST-only graph update1회(141nodes/243edges/10communities), ignored 출력만 갱신했다.

## 한계와 인계

고정 공개 빌드·390×900·소유 HTTP 합성 입력의 cache 전파 경계 검사다. 실제 호스트/TLS/CDN·실자료·운영/배포·모델 성능을 검증하거나 승인하지 않는다. dependency 설치와 제품 파일 변경도 없다. 독립 결과·PNG 검토 후 COORD-05 index 소유자가 정확한 하네스와 보고서2파일을 커밋한다. 작성자는 stage/commit하지 않았다.

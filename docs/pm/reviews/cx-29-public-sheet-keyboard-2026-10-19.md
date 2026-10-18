# CX-29 — 추천 상세 sheet의 키보드 진입과 복귀

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 독립 실행 경계·최종 결과·PNG 검토 PASS 및 Root 수용. 2case 통과, 검토자의 추가 재실행 없음. Base `e4a435f3648db6849749110ef05b46c3b9828977`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254805710 .

고정 공개 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 Sheets.tsx는 모바일 dialog에서 Tab 경계를 순환시키고 body를 잠근다. Desktop(min-width1200)은 nonmodal로 열고 이 가두기를 사용하지 않는다. Escape로 닫을 때 Sheets와 GameScreen의 opener 복귀 경로가 작동한다. 이 계약을 실제 키 입력으로 검사하는 optional `--keyboard-sheet`만 추가했다.

기존 liveFixtures의 첫 정상 phase body를 그대로 사용한다. 390×844와1440×900에서 각각 새 context를 열며, 직접 `.focus()`나 클릭을 주입하지 않고 최대40회 Tab으로 추천 후보 버튼에 도달한 뒤 Enter로 연다. Dialog 이름 `추천 구종 · 대략 위치`는 role locator로 검증한다. focus trace에는 역할/표시 이름(aria-label 또는 textContent), dialog 내부 여부, focusable 여부와 원래 opener ElementHandle 동치가 남는다. 이 trace는 실패 시에도 receipt에 보존한다. Dialog의 trace 이름은 전체 접근성 이름 계산기가 아니라 텍스트 요약이며, 이름 계약은 별도 role locator가 검사한다.

모바일에서는 초기 dialog focus에서 Shift+Tab→마지막 속도 단위 버튼, Tab→닫기, 역방향 경계, 정방향 중간 버튼/마지막/순환까지7키 동작의 대상과 내부 containment를 확인한다. Desktop에서는 Tab→닫기→Shift+Tab으로 dialog 밖의 focusable 요소에 도달하는지 확인한다(body는 탈출 성공으로 세지 않음). 둘 다 Escape 이후 dialog 제거, 동일 opener 복귀, body lock 해제를 검사한다.

## 실행 결과

하네스 SHA `9c85730fc6d7e72ca1bdaaaf9c4d18a579cad923f48b01219c0b8d9985c2a05d`. 기존 v4.3.1 public-build/설치 Playwright와 manifest SHA `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`를 사용했다. 독립 source/diff/실행 경계 검토 뒤 새 전용 `CX29/browser-001`에서1회 실행했다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  PINNED_MANIFEST 7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f PINNED_PUBLIC_BUILD \
  INSTALLED_PLAYWRIGHT_INDEX NEW_CX29_BROWSER_OUTPUT --keyboard-sheet
```

실제 exit0/`PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`,2case/12GET(각6)/1090ms. 두 viewport 모두 **Tab5회로 원래 추천 버튼에 도달**했다. 모바일7키 순환은 모두 dialog 내부이며 aria-modal=true/body lock=true였다. Desktop은 aria-modal 생략/lock없음, dialog 밖 `기록` button으로 실제 focus가 이동한 뒤 Escape로 닫혔다. 두 case 모두 원래 opener focus 복귀/navigation1/추천1행 유지. page error0/금지 요청0, 소유 browser/server 종료 true, 하네스·manifest·4asset 핀 전후 동일이다.

open/closed PNG4장을 작성자가 직접 열어 모바일 닫기 focus/추천 버튼 복귀와 desktop 외부 기록 버튼 focus/추천 버튼 복귀를 확인했다. PNG4장을4case로 합산하지 않는다.

| 산출물 | SHA-256 |
|---|---|
| receipt.json | `986b14c4a026e43fd21f71078481bcacd05cf8f8c26f9786076f09230cb4edb2` |
| live-phases.json | `c83265c6ac42379c88b64756b462869d59f72b86e5f76094fd11b6d210a05927` |
| keyboard-open-390.png | `53273dc18173972107f34bf864ed8d7fe4c9e4b15bc2e37846f1f3a1fabc9023` |
| keyboard-closed-390.png | `60a29d23bbe04d32105a6dff1ef1555bcb7e3e1965d927916d4c7930d68f3f00` |
| keyboard-open-1440.png | `962ce9600627956c2813e432a8b86d78a99abfd9e0f7a4ad65143f83f6c3bda7` |
| keyboard-closed-1440.png | `1b728ea9860d45e969fbf99c5471ecb29ad888dcb74c1860f8c4ce0e988d42e5` |

기존5개 fixture 함수·partial/live/host/default 시나리오 본문은 바이트 동일하다. 새 선택 이름을 제외한 정상 body는 기존 검증 입력과 동일해서 schema를 반복하지 않았다. 기존7suite를 다시 실행하지 않았으며 CX27의17case 결과는 그때 고정 하네스에 대한 증거로 구분한다. optional scenario 인자의 기본값과 기존 height900을 유지했다. `node --check`/`git diff --check` 통과, 최종 코드의 scoped AST-only graph update1회(142nodes/247edges/9communities), ignored 출력만 갱신했다.

## 한계와 인계

이 결과는 고정 Chromium의 두 viewport 키보드 동작이다. 실기기·다른 브라우저·스크린리더 전체 접근성·실호스트/TLS·운영 GO를 검증하지 않는다. 기존 pin/60GET·90초/소유 loopback/외부·WS·SW 차단/cleanup은 그대로다. 제품 UI·원본 fixture·실자료·자격·운영 자원 변경이나 dependency 설치는 없다. 독립 결과 검토 후 COORD-05 index 소유자가 하네스와 보고서2파일만 커밋한다. 작성자는 stage/commit하지 않았다.

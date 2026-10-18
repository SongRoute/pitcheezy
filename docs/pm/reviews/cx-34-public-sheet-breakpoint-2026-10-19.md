# CX34 — 열린 추천 창의 breakpoint 전환 합성 검사

- 작성: Codex implementation_1019, 2026-10-19 KST.
- 상태: 독립 실행 경계 및 결과·PNG 검토 PASS, Root 수용. 신규 1case 실행 PASS이며 별도 검토자의 재실행은 없다.
- 선점: 공동 작업큐 #37 단계1 CLAIM 6255410617, 단계2 범위 갱신·CLAIM 6255426518. 제품 소스·배포물은 수정하지 않았다.

## 목적과 변경

[CX29](cx-29-public-sheet-keyboard-2026-10-19.md)의 두 크기 검사는 각각 새 browser context를 사용했다. 고정 소스의 `apps/observer/web/tests/ws-responsive-smoke.mjs`, `responsive-fit-smoke.mjs`도 같은 방식이라 열린 추천 창을 유지한 크기 전환은 검사하지 않았다. 이 작업은 `tests/coordination/public_candidate_smoke.mjs`에 `--resize-sheet` 선택 분기 하나를 추가한다. 기존 합성 추천 입력과 키보드 접근 검사를 재사용하며 기존 fixture 함수·검사 경로·네트워크 가드·정리 코드를 보존했다. 추가 분기와 선택자 변경을 역변환하면 이전 파일 전체 바이트와 일치한다.

고정 소스 `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 `Sheets.tsx`는 `(min-width: 1200px)` 변화에 따라 modal 상태와 키보드 핸들러·body 잠금을 갱신한다. `service.css`의 `body.sv-locked{overflow:hidden}` 및 `GameScreen.tsx`의 원래 opener 복귀 흐름을 기준으로 기대를 정했다. 같은 page와 dialog/opener 노드를 유지하며 390×844 → 1440×900 → 390×844로 전환한다. 폰 회전이나 실제 기기 검사는 아니다.

## 실행과 관측

전용 CX34 출력의 `argv.json`에 기록한 고정 6인자와 새 `browser-001`, `--resize-sheet`로 정확 1회 실행했다. 실제 stdout은 아래와 같고 stderr는 비어 있으며 exit는 0이다.

```json
{"status":"PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP","scenarios":1,"calls":6,"ownedBrowserClosed":true,"ownedServerClosed":true}
```

receipt의 실제 소요 시간은 1009ms, GET 6회, blocked 0, page error 0이다. 기존 60GET·90초 전체 예산을 유지했고 소유 브라우저와 서버가 모두 닫혔다. 실행 전후 하네스·manifest·4개 asset의 SHA가 동일하다.

| 단계 | aria-modal | body 잠금 / 실제 overflowY | 키보드 관측 |
|---|---|---|---|
| 390×844 최초 열기 | true | true / hidden | 실제 Tab 5회로 원래 opener 도달 후 Enter. 모바일 경계 7개 키 동작이 내부에 유지됨 |
| 1440×900 | 속성 없음 | false / visible | resize 효과가 같은 dialog에 포커스를 다시 둠. Tab 후 Shift+Tab으로 dialog 밖의 focusable `기록` 버튼에 이동 |
| 390×844 복귀 | true | true / hidden | 같은 dialog에 포커스 재설치. Shift+Tab으로 마지막 단위 버튼, Tab으로 첫 닫기 버튼 순환 |
| Escape 닫기 | dialog 제거 | false / visible | 최초와 동일한 opener 노드로 포커스 복귀 |

세 크기 단계 모두 동일 dialog/opener의 연결 상태, dialog 1개, navigation 1회, 단계별 추가 요청 0회를 확인했다. 합성 추천 행은 유지된다. 포커스 trace의 이름은 기존 하네스의 aria-label 또는 textContent 관측값이며 스크린리더 이름 계산 전체 검증은 아니다.

## 증거 핀

- manifest: `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93` (public v4.3.1).
- 실행 하네스: `ad827fa29edd10df4c024694af20a47f02bf499a574741bb7e964f5dbdbe33c9`.
- `browser-001/receipt.json`: `47f76e6546de907e21b1ac30f91d9dbe73155e8b9710fa1e81a7ab6ef462331c`.
- `browser-001/live-phases.json`: `12fcd213a3e5b21d4fd74678c937716df16aaf7de1ce2cb8cdad18ba8af1872c`.

| 직접 열람한 PNG | SHA-256 |
|---|---|
| `resize-0-390.png` | `d848e21d68ecef6f1ded2f7c382d796343b0d8ad24d501119f0138d045b7bed8` |
| `resize-1-1440.png` | `246d1c96e860f0217b96df2dcf014119bb048920ed2be9053889d6ac317441e7` |
| `keyboard-open-390.png` | `4fdcc959fa0bfe2fdeecf7682edd8793f0eeb97194f872d17bc7ef0afb544e8c` |
| `keyboard-closed-390.png` | `689494579c88f54046f2638ac82f8e4aa261c1c9a887424c012fce0b00209b5b` |

네 PNG에서 모바일 하단 창, 데스크톱 오른쪽 패널, 모바일 복귀 및 닫힌 후 원래 추천 버튼의 포커스 표시를 확인했다. 모바일 복귀 후 스크롤 위치는 처음과 달라졌으며 픽셀 동일성이나 스크롤 위치 보존을 검증한 것은 아니다. full-page PNG는 viewport 높이보다 길 수 있다. 시각 확인과 별도로 receipt의 DOM·실제 키보드 관측으로 동작을 판정한다.

## 범위와 한계

이번 신규 검사는 한 case의 세 크기 단계이며 기존 CX29 두 case나 CX27 일곱 suite를 다시 실행한 결과가 아니다. 새 fixture 생성 규칙이나 schema는 추가하지 않았다. Chromium의 합성 입력·소유 loopback HTTP 검증이며 실제 기기·스크린리더·외부 실호스트·TLS·실경기 성능·운영 GO를 뜻하지 않는다. 실제 자료, 제품 코드, 원본 wrapper, 설치 및 배포 변경은 없다.

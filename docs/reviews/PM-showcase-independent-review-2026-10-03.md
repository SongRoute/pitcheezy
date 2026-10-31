# 시연 통합 독립 검토

작성: Claude Fable 5.1 · 2026-10-03 KST. 변경 전 읽기 전용 검토.

**요약**
- 변경 파일 6개(+smoke·pytest)를 읽기 전용으로 검토. 미래 결과 유출·WE 부호·일시정지 시각 보정은 결함 없음.
- 실제 결함성 발견 1건(낮음): /showcase 바깥 제목·폰 링크가 iframe 안 /watch 이동을 따라가지 않아 장면 정체성이 어긋남. 나머지는 취향·미검증.
- 다음: 아래 1번만 고칠지 결정. 실영상 앵커 정확도와 iOS Safari iframe은 코드로 검증 못 함.

## 실제 결함 (우선순위순)

**1. 장면 라벨과 iframe 내용 불일치 (낮음, 재현 가능)**
- `apps/observer/web/src/Showcase.tsx:116`의 h2와 `:110-113`의 복사 링크·"이 공 관전 화면 열기"는 `scene`에 고정되는데, iframe 안 /watch는 자유롭게 다른 공으로 이동한다. `WatchAlong.tsx:317` "다음 공 →", `:318` →, `:264` 키보드(→·스페이스), `:289` 회 버튼, `:291` 타석 select.
- 재현: 장면 1 선택 → iframe에서 "실제 투구 공개" → "다음 공 →". 바깥 제목은 "1회 초 · 타자 · 3구째"인데 iframe은 4구째(또는 다음 타석). 복사 링크도 옛 공을 가리킨다. 다음 공은 대개 압축 영상에 없는 공이라 영상-추천 대응이 끊긴다.
- 유출은 아님(사용자가 명시적으로 공개). 정체성 표시 문제. 제안: 임베드 모드(`?embed=1`)에서 이동 UI를 숨기거나, iframe의 hashchange를 바깥에서 읽어 라벨을 따라가게.

**2. 클립 종료 정밀도·네이티브 컨트롤 (낮음, 취향에 가까움)**
- `Showcase.tsx:82-86` 정지는 `timeupdate`(약 4Hz)에 의존해 release+4를 최대 약 0.25초 넘긴다. `controls`가 켜져 있어 자동 정지 뒤 재생 버튼을 누르면 다음 공들로 계속 흘러간다. `:104` 안내문은 탐색 막대만 언급. 시연 운영상 알고만 있으면 됨.

## 결함 아님 · 확인한 것

- **미래 유출**: `watch_summary.py:82-88, 96, 130`이 읽는 미래 자료는 `decisions[through+1].situation`(점수·회)뿐이며, 이는 `/api/watch/{pk}` 타임라인이 이미 공개하는 값. `test_watch_summary.py:138-153`이 그 이상 읽지 않음을 고정. 장면 변경·재시청 시 `Showcase.tsx:118` key(`selected:take`)로 iframe이 재마운트되어 공개 상태가 남지 않음. `WatchSummaryPanel.tsx:40, :173`은 응답의 `through`·`game_pk`를 검증하고 현재 키와 다른 결과는 한 프레임도 안 그림.
- **WE 의미**: `watch_summary.py:52-57` home_delta=after−before, batting_delta는 Top이면 부호 반전. 패널은 홈팀 기준으로 일관 표기(`WatchSummaryPanel.tsx:149`), 득점 표시는 공격팀 쪽을 고름(`:143`). `WatchAlong.tsx:326-328` 공격팀 환산도 맞음.
- **일시정지 시각**: `Game.tsx:237-241` 재개 시 t0를 멈춘 길이만큼 앞당기고, `:174`에서 정지 중 폴링 중단. 정지 중 ±조정(`:228-235`)도 재개 때 반영됨. 정지 중 `hold` 타이머(`:218-222`)는 계속 돌아 결과 카드가 사라지고 마지막 받은 투구 전 추천으로 돌아가는데, 이미 받은 자료라 유출 아님.
- **포수 셋업**: `Game.tsx:280` 던진 공에만, 정확한 game:ab:pitch 키로만 표시. 투구 전 화면에는 안 나옴.
- **오래된 API 자료**: `get()`·`fetchSummary` 모두 `cache:'no-store'`, `watch_along.py:36-38` mtime으로 캐시 무효화. `Showcase.tsx:32` game_pk 일치 검사. 장면↔투구는 at_bat_number/pitch_number로 매핑해 index 재생성에 영향 없음(`showcase-smoke.mjs:43`).
- **장면 앵커 겹침**: JSON 기준 모든 클립 창(release−3..+4)이 다음 장면 release를 넘지 않음. 가장 가까운 쌍 33:4→34:2 간격 5.6초.
- **설계 메모(결함 아님)**: `through`는 클라이언트가 주장하는 값이라 서버 세션 없이 `through=261`로 전체를 받을 수 있음. 시연용 설계로 보임.

## 모바일

- 800px 이하 한 열, iframe 700px 고정(`showcase.css:16`). /watch 하단 바는 `position:fixed`(`watch-along.css:61`)라 iframe 하단에 붙음. 탭 타깃 44~48px 충족. 스모크는 Chromium 390px만 확인(`showcase-smoke.mjs:27`).

## 취향·가설 (결함으로 보지 않음)

- `showcase-scenes.json:3-6`은 타이밍 JSON 해시만 고정하고 영상 자체는 고정하지 않음. 같은 URL로 재인코딩되면 앵커가 조용히 어긋남. `duration` 기대값을 JSON에 두고 `onLoadedMetadata`에서 비교하면 잡힘.
- `WatchSummaryPanel.tsx:29` `pp()`는 +0.04%p를 "+0.0%p"로 표기. 표시만의 문제.

## 미검증 경로

- 실제 MLB 영상 재생·앵커 정확도(스모크가 영상 요청을 차단, `reviewStatus: assistant_annotations_unreviewed`).
- iOS Safari에서 iframe 높이 강제와 `WatchAlong.tsx:271` scrollIntoView가 부모 페이지를 스크롤하는 동작.
- `WatchAlong.tsx` 미커밋 변경의 diff(셸 금지라 작업트리 파일 그대로 검토).
- :8766/:8770 무변경 여부는 코드 읽기만으로 확정 못 함. 새 경로(`/showcase`, `/api/watch/{pk}/summary`)는 추가형이라 기존 경로 수정은 안 보임.
- `rec.previous.key`·`pa_pitches[].key`가 `game:ab:n` 형식이라는 가정(기존 `paOf`·`split(':')[2]` 사용과 일치).

이 검토는 과학적 결과 승인이 아니다.

## PM 후속 수정

iframe 전용 embed=1 모드에서 회·타석·이전/다음 공 이동을 제거하고 공통 이동 함수에서도 차단했다. 바깥 장면 선택만 대응 투구를 변경한다. 일반 /watch 이동은 유지한다. 중복 시작 안내는 iframe에서 생략한다. 회귀 검사에 iframe의 이동 UI 부재 및 키보드 이동 차단을 추가했다. 실제 영상 앵커·아이폰 Safari 확인은 여전히 별도 필요하다.

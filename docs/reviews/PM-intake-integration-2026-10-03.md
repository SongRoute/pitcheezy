# PM intake: integration

Status: initial read-only agent report; NOT a gate approval or verified implementation.
Models: claude-opus-5-5
Source: /tmp/pitcheezy-pm-bootstrap-40jNoB/integration.json

**요약**
- 한 일: PM 운영규칙, 화면 인수인계 문서, 셋업 작업 지시, LIVE-service-plan, PM-state, decisions D160~D163을 읽었습니다. **ui-frame·live-sync 두 워크트리는 읽기 권한이 승인되지 않아 소스 코드를 직접 확인하지 못했습니다.** 아래 코드 관련 내용은 문서에 적힌 내용을 근거로 합니다.
- 핵심: 추천 → 실제 공 → 비교 화면은 `/game`에 이미 있습니다(경기 849844를 6배속으로 90초 재생해 Chromium에서 확인). 영상 정렬은 아무것도 없고, 통합 브랜치와 실기기 확인도 아직 없습니다.
- 다음: 새 통합 워크트리에서 live-sync → ui-frame → intent-setup 순으로 합친 뒤, 영상 기준점 정렬 기능부터 만듭니다.

---

## 1. 이미 있는 것 (경기 선택 → 추천 → 실제 공 → 비교)

**경기 선택**
- 예전 `/live`에는 경기 목록과 "실제 속도로 다시 보기"(`/live?game=N&replay=0`)가 있습니다. 회 버튼, ±1·±5초 맞추기, `&speed=`, `&lag=` 옵션도 있습니다(`docs/handoffs/LIVE-service-plan.md:34`).
- `/game`용 홈(H1)은 없습니다. 지금은 주소의 `?game=`로만 경기를 고릅니다(`docs/handoffs/UI-frame-PM-handoff.md:32`, `:167`).

**추천**
- `GET /api/live/{경기}/state`가 동결된 ≤2025 ARM-B로 계산합니다. 첫 준비에 5~6초, 이후 공마다 중앙값 0.13~0.14초입니다(`LIVE-service-plan.md:22`).
- `/game`의 "다음 공" 단계는 구종, 타자 기준 위치 말, 꼬리표, 존의 목표 칸을 보여 줍니다(`UI-frame-PM-handoff.md:56`).

**실제 공·비교**
- "방금 던진 공" 단계를 6초 보여 줍니다. 실제 구종·구속·결과, "구종 ✓ / 위치 ✕", 존에 목표와 실제가 함께 나옵니다. 타석 끝 단계(10초)와 결과 대기 단계도 있습니다(`UI-frame-PM-handoff.md:57-61`).
- "✓"는 추천과 같았다는 뜻일 뿐, 추천이 좋았다는 뜻이 아닙니다(`:81`).

**포수 미트 가로 위치**
- `GET /api/watch/{경기}/setup`은 `demo/intent-setup` 브랜치에만 있습니다. 자료는 경기 849843 하나이고, 가로 위치가 있는 공은 27구입니다. 화면에는 아직 그리지 않았습니다(`docs/handoffs/UI-setup-display-work-order.md:14-21`).

**참고**
- 메인 체크아웃의 옛 백엔드에 `/api/video-lab/clips/{clip_id}`가 있습니다(`apps/observer/backend/observer_app/main.py:183`). live-sync에도 남아 있는지는 확인하지 못했습니다.

## 2. 10/6 영상+폰 시연의 우선 공백 3개

**1. 영상과 pitch_id 기준점 정렬 (가장 큼)**
- 영상 파일이 없습니다(D163, `docs/decisions.md:275`).
- 재생 시계는 피드의 `startTime` 기준입니다. 영상 시각과 맞추는 기능은 `/game`의 ±1·±5초 버튼과 "미트에 들어오는 순간 누르기"뿐입니다.
- 필요한 것: 시연 장면 직전 pitch_id로 바로 이동하는 기능, 그리고 한 번 눌러 기준점을 잡는 흐름.
- `/game`의 "지난 장면"은 페이지를 연 뒤에 본 공만 다시 볼 수 있습니다(`UI-frame-PM-handoff.md:137`). 따라서 장면 앞으로 이동하는 기능이 없으면 시연 동선이 깨집니다.
- `/game`에 `/live` 같은 회 이동 기능이 있는지는 `Game.tsx`를 못 읽어 미확인입니다.

**2. 통합 브랜치와 분리된 시연 인스턴스**
- ui-frame은 live-sync를 `4452331`까지만 합쳤습니다. live-sync는 현재 `09e12e9`이고, intent-setup은 합쳐지지 않았습니다(`PM-state.md:30-32`, `:51`).
- `:8770` 미리보기는 데이터를 운영 서버 `:8766`에서 받으므로 시연용으로 쓰면 안 됩니다.
- 화면 문구 Q4 "2026 성능 평가는 아직 없어요"(`Game.tsx:310`)도 고쳐야 합니다(`PM-state.md:133`).

**3. 폰 접속과 실기기 확인**
- 접속은 Tailscale 비공개 주소만 가능합니다(CP6).
- 교수 연구실 네트워크에서 아이폰 접속, 실기기, 사파리, 실제 속도에서의 전환 느낌은 모두 확인되지 않았습니다(`UI-frame-PM-handoff.md:109-113`, `PM-state.md:91`).
- 녹화나 정적 자료로 된 대체 동선도 아직 없습니다.

## 3. 안전한 브랜치·API 통합 순서

0. **건드리지 않을 것**: `pitcheezy-demo`, `:8766`, `pz` 창들. `ui-frame` 워크트리에서도 빌드하지 않습니다. `:8770`이 그 빌드 결과물을 서빙하고 있어서 덮어쓰게 됩니다(PM-charter §6).
1. **새 작업트리·브랜치를 만듭니다** (예: `demo/showcase-1006`). 기준은 `demo/live-sync` `09e12e9`입니다.
2. **`demo/ui-frame` `7bbc9ed`를 합칩니다.** 화면 쪽 변경은 `Game.tsx`, `game.css`, `main.tsx`, tests, `viewer.mjs`입니다. 합친 뒤 백엔드 테스트와 화면 스모크 검사를 돌립니다.
3. **`demo/intent-setup` `5a102ea`를 합칩니다.** live-sync `221436a` 기준이고 서버 파일 4개만 바뀌었습니다. 합친 뒤 `test_setup_estimates.py`를 돌립니다.
4. **API 계약을 고정합니다.** 대상은 `/api/live/{경기}/state`(쿼리 `replay·speed·lag`)와 `/api/watch/{경기}/setup`이고, 기준 문서는 `apps/observer/CONTRACT.md`입니다. 그 뒤에 UI 작업을 합니다: 미트 표시, 장면 이동·기준점, Q4 문구.
5. **node_modules를 분리합니다.** 지금처럼 live-sync에 심볼릭 링크하지 않고, 새 작업트리에서 lockfile 그대로 `npm ci` 합니다.
6. **시연 인스턴스는 별도 포트**(예: 8771)에서 자체 백엔드로 띄웁니다. 메모리는 경기당 약 1.4GB입니다. 시점은 10/4 기록·대조와 자원이 겹치지 않는 시간으로 합니다(D159).

**시연 경기 후보**: 849843
- 8배속 전체 재생에서 오류 0, 추천 가능 246/262구입니다.
- 셋업 자료가 있는 유일한 경기입니다.
- 영상 출처가 확인돼야 확정할 수 있습니다.

## 4. 관련 기존 테스트·검사

- `apps/observer/web/tests/game-frame-smoke.mjs`와 `tests/fixtures/game-frame-states.json` (ui-frame)
- `apps/observer/backend/tests/test_setup_estimates.py` (intent-setup)
- `demo_precompute.py live-check`, `timecode-check`, `scripts/live_rehearsal.py record|report|leads|simulate` (live-sync)
- 산출물(`demo/live-sync`):
  - `results/checks/LIVE-replay-849843-x8.json`
  - `LIVE-check-849844.json`
  - `LIVE-recorded-replay-849844.json`
  - `LIVE-rehearsal-849844.json`
- 메인 체크아웃의 옛 테스트: `apps/observer/backend/tests/test_inning_decision_games.py`
- 워크트리의 전체 테스트 목록은 읽지 못해 미확인입니다.

## 5. 구현 완료와 미검증 런타임 구분

**구현됐고, 이전 세션이 검증했다고 기록한 것** (제가 다시 돌리지는 않았습니다)
- 실시간 경로로 849843을 8배속 재생: 조회 2,248회 중 오류 0.
- 849844 실경기: 서버 오류 0. 1순위 구종이 경기 종료 뒤 사전 계산과 같은 비율은 처음 본 화면 기준 97.8%.
- `/game` 1·2단계: 화면 크기 3종에서 스크롤 0px, 잘린 내용 0. Chromium과 모의 응답으로 확인.
- 849844를 6배속으로 90초 재생했을 때 상황 전환이 이어지는 것을 확인.

**미검증이거나 아직 구현 안 된 것**
- 세 브랜치의 통합과 통합 후 테스트
- 미트 위치 화면 표시
- 영상 정렬과 장면 이동
- 실제 속도에서의 6초·10초 전환
- iPhone·Galaxy 실기기, 사파리·삼성 인터넷
- 교수 연구실에서의 Tailscale 접속
- 분리된 시연 인스턴스
- 대체 녹화 자료
- 10/4 디비전시리즈 4경기 결과: 실시간 관전을 시연에 넣을지가 이 결과에 달려 있습니다.

이번 확인은 읽기만 했고 명령 실행, 파일 수정, 작업 위임은 하지 않았습니다. 워크트리 소스를 읽을 수 있게 되면 `Game.tsx`의 장면 이동 기능 유무와 `CONTRACT.md`를 먼저 확인하겠습니다.

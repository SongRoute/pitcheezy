# /game 재연결 검토 처리 — fe741ea (2026-10-04)

상태: `same_model_reviewed; Codex_review_pending`. 검토자와 작성자 모두 Opus 5.5다(다른 세션). 쇼케이스 통합·배포는 하지 않았다. 통합은 root가 한다.

- 검토: [OPUS-review-game-reconnect-fe741ea-2026-10-04.md](OPUS-review-game-reconnect-fe741ea-2026-10-04.md), 원 커밋 `489cd12`(검토 브랜치). 결론은 통합 비차단.
- 처리 커밋: 이 문서와 같은 커밋(`fe741ea` 위)

## 지적별 처리
- F1 중간(응답 없는 요청이 루프·버튼·문구를 멈춤): **부분 처리, 남김**
  - 처리한 것: 문구의 숫자 약속을 뺐다. "2초마다 자동으로 다시 연결하고 있어요" → "자동으로 계속 다시 연결하고 있어요"
  - 남긴 것: 요청 제한 시간(AbortController). `/state` 응답 시간 분포가 미측정이라 값을 정할 수 없다. 실측한 뒤 설정값으로 넣는다. 840eeee에도 있던 동작이다.
  - 요청 중 버튼 비활성화는 넣지 않았다. 매 조회마다 화면이 다시 그려지고, 제한 시간이 없으면 F1을 해결하지 못하기 때문이다.
- F2 낮음(첫 응답 전 로딩 중에 `LIVE`): **수정**
  - 첫 응답 전·오류 없음 상태의 머리말을 `연결 중`(회색 점, `off`)으로 바꿨다.
  - 스모크는 첫 응답을 1.5초 늦추고 `불러오는 중` 동안 머리말이 `연결 중`인지 확인한다.
- F3 낮음(낭독기가 2초마다 다시 읽음): **수정**
  - `role="status"`에는 바뀌지 않는 문장만 남겼다. `실패 N번`은 live region 밖 일반 텍스트다.
  - 스모크가 status 영역 텍스트가 고정 문장 하나인지 확인한다.
- F4 낮음(재시도 검사가 타이밍 위상에 기댐): **수정**
  - 클릭 직전 마지막 자동 조회 시각을 잡는다. 클릭 뒤 첫 조회가 그 시각 + 1500ms 전에 와야 한다. 자동 주기는 약 2초다.
  - 클릭이 마지막 자동 조회 뒤 1초를 넘기면 "판단 불가"로 실패시킨다.
- F5 정보(iOS 복귀 직후 오래된 요청): 기록만 남긴다. F1 제한 시간과 함께 다룬다. 실기기 미측정.

## 검사 (맥, 고정 응답, 실제 기기 아님)
- `tsc --noEmit -p tsconfig.json`: exit 0
- `vite build --outDir /tmp/pz-reconnect-1004-r2-dist --emptyOutDir`: ok. 레포 `dist/`는 건드리지 않았다.
- `game-reconnect-smoke.mjs`: 새 빌드 `game reconnect smoke: ok`(21.6 s)
- 음성 대조
  - fe741ea 빌드: `FAIL header does not say 연결 중 while the first answer is pending`
  - 버튼 no-op 변형 빌드(`/tmp`, 소스는 원래대로 되돌림): 2/2 실패(`Timeout 1500ms … 다음 공`)
- 회귀
  - `game-frame-smoke.mjs`: 3개 크기 ok
  - `game-demo-regression.mjs`: ok. dist 경로 한 줄만 env로 바꾼 임시 사본으로 돌렸고 삭제했다.
- `node_modules`는 메인 체크아웃을 가리키는 임시 symlink를 썼고 삭제했다.

## 남은 것
- F1 제한 시간: `/state` p99 실측 → 설정값 → 무응답 스모크 단계
- 실제 iPhone Safari 복귀 확인(미측정)
- root 쇼케이스 통합

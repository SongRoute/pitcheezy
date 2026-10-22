---
name: ui-impl
description: Figma 프레임 + docs/ui/{화면}.md 체크리스트로 apps/observer/web 화면을 구현하고 PR을 연다. 사용 /ui-impl {화면 슬러그}
---

# /ui-impl {화면}

입력: `docs/ui/{화면}.md` (①이 작성). 없으면 멈추고 사용자에게 알린다.

## 절차
1. `docs/ui/{화면}.md`를 읽는다. 상단 메타(Figma URL, 라우트, mock fixture)와 CHK 항목 전부.
2. Figma MCP: URL에서 fileKey·nodeId 추출 → `create_session` → `get_design_context`(필수, `/figma-design-to-code` 스킬 먼저) + `get_screenshot`.
   출력은 참고용 레퍼런스다. `apps/observer/web/src`의 기존 컴포넌트·스타일을 재사용한다.
3. mock fixture가 체크리스트에 적혀 있는데 파일이 없으면 `src/mocks/fixtures/`에 만든다. `VITE_API_MOCK=1`에서 `/api/*`가 이 fixture로 응답해야 한다.
4. 구현. CHK 항목을 하나씩 보며 각 항목이 "확인:" 절차로 통과하는지 직접 점검한다 (`npm run dev` 후 Playwright 스모크 — `apps/observer/web/tests/` 패턴).
   390px / 1280px 둘 다.
5. `npm run build` 통과 확인.
6. 커밋 `feat(web): {화면} …`, PR 제목 `[UI] {화면}`. 본문은 `.github/PULL_REQUEST_TEMPLATE.md`의 UI 핸드오프 칸을 채운다.
   프리뷰 URL은 Vercel 봇 코멘트가 달리면 `gh pr edit`으로 채운다 (1~3분 대기 후 `gh pr view --comments`).
7. 보고: PR URL + "Astra에 보낼 것: {PR URL}" 한 줄.

## 하지 않는 것
- 체크리스트에 없는 기능 추가. 필요하면 PR 본문 "요약"에 제안만
- Figma 코드를 그대로 붙여넣기 (디자인 토큰·컴포넌트는 기존 것으로 치환)
- 백엔드 수정. `/api` 계약이 바뀌어야 하면 멈추고 알린다

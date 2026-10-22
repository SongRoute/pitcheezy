# UI 작업 흐름 — Figma → Claude Code → Astra 점검

2026-10-22 수립. 대상 앱: `apps/observer/web` (Vite + React). 화면 하나 = 체크리스트 하나 = PR 하나.

## 흐름

```
① 설계    Claude(채팅 프로젝트): 텍스트 요청 → Figma 프레임 + docs/ui/{화면}.md 체크리스트 → 커밋
② 구현    Claude Code: /ui-impl {화면} → Figma MCP로 프레임 읽기 + 체크리스트 읽기 → 구현 → PR
③ 점검    Astra(ChatGPT): PR URL 하나 받음 → 프리뷰 URL에서 체크리스트 항목별 PASS/FAIL → PR 코멘트
②′ 수정   Claude Code: /ui-fix {PR번호} → FAIL 항목 파싱 → 수정 → 코멘트로 재검 요청 → ③ 반복
```

Astra는 Claude 세션과 컨텍스트를 공유하지 않는다. 세 단계의 핸드오프는 전부 **브라우저로 닿는 것**만 쓴다:
Figma 노드 URL · `docs/ui/{화면}.md` · Vercel 프리뷰 URL · PR 코멘트. 노션·로컬 파일은 끼우지 않는다.

## 핸드오프 규칙

| 단계 | 넘기는 것 | 어디에 |
|---|---|---|
| ① → ② | Figma 노드 URL, 체크리스트 경로 | `docs/ui/{화면}.md` 상단 메타 + 커밋 |
| ② → ③ | 프리뷰 URL, 체크리스트 경로, Figma URL | PR 본문 (`.github/PULL_REQUEST_TEMPLATE.md`) |
| ③ → ②′ | `CHK-NN PASS/FAIL — 근거` 목록 | PR 코멘트 1개 |
| ②′ → ③ | "재검 요청: CHK-03, CHK-07" | PR 코멘트 |

## 체크리스트 작성 규칙 (①)

- 템플릿 `docs/ui/CHECKLIST_TEMPLATE.md`. ID는 `CHK-NN`, 화면 안에서만 유일
- 항목마다 **확인 방법**이 있어야 한다. 없으면 Astra가 추측으로 판정하거나 건너뛴다
- 눈으로만 판단되는 항목("예쁘다", "자연스럽다")은 넣지 않는다. 수치·텍스트·URL·요소 수·창 너비로 환원
- 모바일 390px / 데스크톱 1280px 두 폭을 기본으로. 설계 결정(2026-10-13)대로 넓은 화면에서 더 많은 정보 노출이면 그 차이도 항목으로
- 데이터 바인딩 항목은 mock 데이터 기준으로 쓴다 (프리뷰에는 백엔드가 없다)

## 프리뷰 배포 (Vercel)

- 프로젝트 Root Directory `apps/observer/web`, Build `npm run build`, Output `dist`
- PR마다 자동 프리뷰. URL은 Vercel 봇이 PR 코멘트로 단다 → PR 본문 프리뷰 칸에 복사
- Vercel Authentication은 **끈다** (Astra가 로그인 없이 열어야 함). 프리뷰 URL 자체를 외부에 올리지 않는다
- SPA 라우팅: `apps/observer/web/vercel.json`

### 블로커 — mock API 모드
`src/App.tsx` 등이 `/api/*`를 직접 호출한다. 프리뷰에는 백엔드가 없으므로 **`VITE_API_MOCK=1`일 때 fixture로 응답하는 모드**가 먼저 필요하다.
권장: MSW(browser) + `src/mocks/fixtures/*.json`. Vercel 환경변수 `VITE_API_MOCK=1`(Preview만). 이게 없으면 ③은 "화면이 비어 있음"만 보고 끝난다.

## Astra 쪽 (ChatGPT Work)

- 모델 GPT-6 Astra 선택. `docs/ui/ASTRA_PROMPT.md`를 프롬프트 템플릿으로 저장해 두고 PR URL만 바꿔 보낸다
- Astra의 권한: 읽기 + 클릭 + 리사이즈 + PR 코멘트 작성. 머지·승인·삭제·로그인 금지 (프롬프트에 명시)
- PR 코멘트를 쓰려면 Astra가 GitHub에 로그인한 브라우저여야 한다 → ChatGPT 데스크톱 내장 브라우저/확장에서 Song 계정으로 1회 로그인. 클라우드 브라우저는 매번 로그인 → 비추천

## Claude Code 쪽

- Figma MCP: `.mcp.json`에 등록(프로젝트 공유). 첫 사용 시 `claude mcp` 로 OAuth 1회
- 스킬: `.claude/skills/ui-impl`, `.claude/skills/ui-fix`
- 규약 §3 그대로: 세션 브랜치, PR 제목 접두 `[UI]`, 커밋 scope `web`

## 1회 셋업 체크 (Song)

- [ ] Vercel: GitHub 앱 연결 → Import `SongRoute/pitcheezy` → Root `apps/observer/web` → Authentication Off → Preview 환경변수 `VITE_API_MOCK=1`
- [ ] mock API 모드 구현 (위 블로커). `/ui-impl` 첫 실행 전에
- [ ] Claude Code에서 `claude mcp list`로 figma 보이는지 + OAuth
- [ ] ChatGPT Work: Astra 선택, `ASTRA_PROMPT.md` 저장, 내장 브라우저에서 GitHub 로그인
- [ ] 첫 화면 체크리스트 1개 작성 → 루프 1회 돌려보고 템플릿 손보기

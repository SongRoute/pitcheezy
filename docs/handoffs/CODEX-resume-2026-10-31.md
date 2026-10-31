# Pitcheezy 인계 — 2026-10-31 밤 (Claude Code 세션 → ChatGPT/Codex)

이 문서가 11/24 전시까지의 **현재 상태·방향·할 일의 정본**이다. 레포가 유일한 출처이고 노션은 아카이브다. 맨 아래 §6에 ChatGPT/Codex에 붙여 넣는 프롬프트가 있다.

## 1. 오늘(10/31) 끝난 것
- **결정·회신**: 팀원(조현준) 요청에 회신 3(전시 구성·사이트 정리·일정)과 회신 4(고정 빌드 전달)를 `SongRoute/pitcheezy#38`에 게시, 합격 기준 한 줄을 `Pitcheezy/transition-models#3`에 게시.
- **정본 복원**: 맥미니에만 있던 커밋 113개·결정 D193~D197·캡스톤 문서·미푸시 브랜치 18개를 origin에 올림. 운영(ws/launch-2026)·연구(codex/ml-matrix-execution)·PR #41을 합친 통합 main(PR #42 머지).
- **서비스 마무리**: `svc/ui-1026-main` → PR #43 머지(main `d04afab1`). UI-1026 + 긴급 안내 숨김 수정 + ARM-B off 경보 + DEMO-WS-GAPS 전시 재생 포팅 + Opus 독립 검토(BLOCK) 지적 P1 2건·P2 3건 수정.
- **전시 데이터**: 849843(WC 1차전 CHC@SD) v1 발행 파일(262구, 추천 246·미지원 16, setup 포함)을 보관 피드로 오프라인 발행. 고정 번들 `bundle-849843-fba6b2ff.zip`(867,682 bytes, SHA-256 `af10293d5134219b7f750d00786366b93b0134c895519b939fd99a356d77a4bc`) — 맥미니 `~/Projects/pitcheezy-pm-jobs/exhibition-849843-1031/`. 기록: `docs/pm/service/exhibition-build-2026-10-31.md`.
- **참고 카드 프로필**: 대상 2026-09-29(컷오프 9/28) 프로필 생성(`<job>/profiles/LATEST.json`). 발행기 재생 모드에서 미지원 투수 Rolison 타석에 참고 카드가 실리는 것까지 확인(스냅샷 도장 부여 스크립트 `<job>/stamp_snapshots.py` 필요).
- **정리**: PR #33·#34·#39·#40·#41 닫음, 월드시리즈용 launchd 3개 해제(plist는 `~/Library/LaunchAgents/disabled-2026-10-31/`).
- **못 한 것**: Codex CLI(0.166.0, gpt-6-astra)는 맥미니에서 "timed out negotiating with the code-mode host"로 명령 실행 불가 → 독립 검토는 Claude Opus로 대체. Codex **앱**에서는 아직 시도 안 함.

## 2. 방향 (확정)
- 마감: **11/24(화) 전시회**. 프로젝트는 **11/20 기능 동결**. 팀원 일정 11/21~23 리허설.
- 전시 화면: Song의 관전 본체(UI-1026, main). 팀원 사이트는 홈 1페이지·호스트·클립 자산·발표 원고만 남기고 watch/receiver/games/guide/report 삭제(회신 3·#3 합격 기준으로 통보함).
- 전시 구동: 모니터 2대. 모니터 1 = 발행기 재생 모드 → v1 파일 → 공개 빌드(한 구 약 10초, 참고 카드 포함). 모니터 2 = 849843 영상(풀 중계 1순위, MLB 공식 클립 체인 대비). 관람객 폰 = QR → pitcheezy.com(팀원 Cloudflare 호스트)의 고정 빌드.
- 호스트: 팀원 Cloudflare Worker/R2에 Song 빌드 + v1 파일. pitcheezy.com 도메인을 거기로. 공동 연결시험 11/11(수) 저녁 제안.
- 역할: 송동선 = 서비스·모델·발행·결정 / 조현준 = 호스팅·클립·CV(M3 고정) / 한강민 = 문서·포스터·발표.
- 포스터: 큰 구조 유지, 수치·세부는 **모든 작업이 끝난 뒤 한 번에** 정정. 포스터 수치(263구·−0.0137·180구/0.13초·+0.09%p)는 과장이 아니라 포스터용/문서용 분리이며, 정정 시 분모·조건을 붙인다(근거: `docs/pm/service/exhibition-build-2026-10-31.md`, 사이트 guide.html, D08).
- 모델: 전시용 모델은 G0 동결. 성능 개선은 **별도 작업으로 제로베이스** 고민·실험. 규칙(2026 데이터 금지, e2e OPE 채택 기준, −0.003 문턱)은 재검토 대상이되 바꾸면 decisions.md에 한 줄.
- AI 도구: ChatGPT Pro는 **11/6까지** → 그때까지 Codex에 검토·포팅·설계 작업을 몰아준다. 11/6 이후 Claude 단독(작성 Opus, 검토 Fable/Sonnet).

## 3. 할 일 목록

### A. 지금~11/2 (사람)
- [ ] 현준에게 ZIP·빌드 기록 비공개 전달(회신 4에 SHA 적어 둠). 공동시험 11/11 저녁 확정.
- [ ] 결정 기록 추가: D198 전시 구성(849843, 모니터 2대), D199 팀원 사이트 범위·호스트, D200 서비스 마무리 머지(PR #43)·검토 처리 — `docs/decisions.md`에 한 줄씩.
- [ ] 영상 소스 확정: MLB.TV/SPOTV VOD 가능 여부. 안 되면 클립 체인.

### B. ~11/6 (ChatGPT/Codex 우선)
1. **독립 검토 2건**(Codex 앱): (a) PR #43 diff(`git diff 61be7a86...d04afab1 -- apps/observer`) 재검토 — 프롬프트 `docs/reviews/REVIEW-PROMPT-svc-ui-1026-main-2026-10-31.md`, Opus 결과 `docs/reviews/OPUS-REVIEW-svc-ui-1026-main-32e79c2f.md`와 대조. (b) 전시 빌드 기록의 사실 대조(수치·커밋·SHA).
2. **옛 DOM 기준 스모크 포팅**: `static-publish-smoke.mjs` 4절 이후, `ws-gaps-smoke.mjs`, `ws-moments-smoke.mjs`를 UI-1026 DOM(.sv-hd-1/.sv-hd-status, .sv-reclist/.sv-rec-cell, .sv-pitch-row, 접힌 .sv-atbat-toggle, sr-only .sv-state-note[data-kind])으로.
3. **P3 정리**(Opus 검토): Zone.tsx 미사용 export, api.ts 승리확률 주석, asOfWords에 타자 이름, 기록 시트 정정 집계 문구, pasAsOf last=null 처리.
4. **demo_precompute live-replay 도장 수정**: 스냅샷 `metaData.timeStamp`를 단조 증가로 써서 `stamp_snapshots.py` 없이도 발행기 재생이 되게. 테스트 1개.
5. **전시 런북** `docs/pm/service/exhibition-runbook-2026-11-24.md`: 모니터 1(발행기 재생 명령·env·시작 구·속도), 모니터 2(영상 오프셋 맞추기), 폰 QR, 장애 시 절차, 리허설 체크리스트. 근거는 빌드 기록 §2·§6.
6. **모델 개선 설계(B9, 제로베이스)**: `docs/pm/research/model-improvement-2026-11.md` — 규칙 재검토(바꿀 것/지킬 것), 후보 목록과 비용(A5000 5장 기준), 3주 일정. 출발 자료: `docs/ML_EXPERIMENT_MATRIX.md`, `docs/handoffs/ML-experiments-next-session.md`, 10/27 트랙 A 결과(D184~D192), P48 B1(G15 대비 ΔNLL −0.0037), 2021–22 독립 확인셋 수집 안.
7. **예측기 승격 단계 C 설계**: G15 mix 또는 P48 B1을 서비스 예측기로 올릴 때의 ≤2025 S2→S6 재인증 등록서 초안(실행은 Song 승인 후).

### C. 11/7~11/19
- [ ] 공동 연결시험(11/11) → 팀원 호스트에 빌드 배포, pitcheezy.com 연결 확인.
- [ ] 발표·포스터 자료 팩(강민): 수치·분모·조건 표(빌드 기록 §1·§3, guide.html 문구, D08), 화면 캡처는 UI-1026으로 재촬영.
- [ ] 팀원 사이트 정리 확인(#3 합격 기준 1~5), CV 발표 범위 M3 고정 확인.
- [ ] 모델 실험 실행(A5000) → 결과는 `results/`·`docs/experiments.md`.
- [ ] 전시 리허설 1회(모니터 2대, 실기기 iPhone 검수).
- [ ] 11/6 이후 체제: `docs/AI_COLLABORATION.md`에 Claude 단독 절 추가, `docs/handoffs/CLAUDE-resume-prompt.md` 갱신.

### D. 11/20~11/24
- [ ] 11/20 동결 → 11/21~23 리허설 → 11/24 전시. 포스터 최종 정정은 동결 직후 한 번에.

## 4. 어디에 무엇이 있나
| 것 | 위치 |
|---|---|
| 정본 코드 | `origin/main` d04afab1 (맥미니 워크트리 `~/Projects/pitcheezy-wt/integrate-main-1031`, 브랜치 svc/ui-1026-main=머지됨) |
| 전시 작업 폴더 | 맥미니 `~/Projects/pitcheezy-pm-jobs/exhibition-849843-1031/` (out4/v1 발행 파일, bundle ZIP, profiles/, replay-849843-stamped/, stamp_snapshots.py, review/) |
| 보관 피드·프리컴퓨트 | T7 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/{feeds,watch,intent}` |
| 팀원 레포 | `Pitcheezy/transition-models` (보드 #3, Song 요청 #18) / Song 보드 `SongRoute/pitcheezy#35`, 작업큐 #37, 팀원 요청 #38 |
| 발행기 환경 | 빌드 기록 §2 (env -i … `.venv-observer/bin/python -m observer_app.publisher`) |
| 테스트 | 루트 `PYTHONPATH=src:.:apps/observer/backend .venv/bin/pytest tests -q` / observer 백엔드 `.venv-observer` / ops `.venv/bin/pytest apps/observer/ops/tests` / 웹 `node tests/<x>.mjs`(`/opt/homebrew/bin/node`, Playwright 캐시 있음) |

## 5. 주의(이번에 배운 것)
- 워크트리에서 파이썬 테스트는 `PYTHONPATH=src`를 앞세울 것(venv 편집 설치가 메인 체크아웃을 가리킴).
- 병합에 `merge=union`을 코드·CSS에 쓰지 말 것(service.css 사고). 문서(decisions.md 등)에만.
- 팀원 레포 문서의 "사용자"는 조현준 본인. Song 승인과 혼동 금지.
- 2026 데이터는 학습·튜닝·모델 선택에 쓰지 않는다(규칙 변경은 decisions.md 한 줄 뒤에).
- 외부 게시(이슈·메일)는 사람 승인 뒤에만.

## 6. ChatGPT/Codex에 붙여 넣을 프롬프트

```
너는 Pitcheezy(MLB 투구 추천 관전 서비스, 3인 졸업 캡스톤, 전시 2026-11-24)의 Codex 작업자다. 레포 SongRoute/pitcheezy의 main(d04afab1)에서 시작한다. 먼저 CLAUDE.md, AGENTS.md, docs/handoffs/CODEX-resume-2026-10-31.md(이 문서)를 읽고, 그 §2 방향과 §5 주의를 지킨다.

작업 원칙
- 한 작업 = 한 브랜치 = 한 PR. 커밋은 Conventional Commits(scope=모듈). PR 전에 관련 테스트를 돌리고 결과를 PR 본문에 적는다(루트 pytest는 PYTHONPATH=src:.:apps/observer/backend, 웹은 node tests/<x>.mjs).
- 숫자를 지어내지 않는다. 측정 못 한 것은 "미측정"이라고 쓴다. 2026 데이터로 학습·튜닝·모델 선택을 하지 않는다.
- GitHub 이슈·외부 발신·main 직접 push·설정 변경은 하지 않는다. 결정이 필요하면 선택지와 추천안을 적고 멈춘다.
- 기존 구현·실험 코드·검증 게이트를 "정리" 명목으로 지우지 않는다.

오늘 할 일(순서대로, 각각 별도 PR)
1. 독립 검토: git diff 61be7a86...d04afab1 -- apps/observer 를 docs/reviews/REVIEW-PROMPT-svc-ui-1026-main-2026-10-31.md의 기준으로 적대적으로 검토하고, docs/reviews/OPUS-REVIEW-svc-ui-1026-main-32e79c2f.md의 지적이 fba6b2ff에서 실제로 닫혔는지 코드로 확인한다. 결과를 docs/reviews/CODEX-REVIEW-main-d04afab1-2026-11.md 로 쓴다(P1/P2/P3, 파일:줄, 재현, 최소 수정, 판정).
2. 스모크 포팅: apps/observer/web/tests/{static-publish-smoke.mjs(4절 이후), ws-gaps-smoke.mjs, ws-moments-smoke.mjs}를 UI-1026 DOM에 맞춘다(.sv-hd-1/.sv-hd-status, .sv-reclist/.sv-rec-cell, .sv-pitch-row, .sv-atbat-toggle, sr-only .sv-state-note[data-kind]). notice-hide-smoke.mjs와 ui-1026-capture.mjs가 참고 선례다. 헤더 시리즈 문구 검사는 "미포팅 기능"으로 skip 처리하고 사유를 적는다.
3. demo_precompute.py live-replay가 스냅샷 metaData.timeStamp를 단조 증가(YYYYMMDD_HHMMSS)로 쓰게 고치고 테스트를 하나 추가한다. 발행기 재생 모드에서 display_state.reason이 unorderable_provider_stamp가 되지 않아야 한다(재현 방법은 docs/pm/service/exhibition-build-2026-10-31.md §6).
4. 전시 런북 docs/pm/service/exhibition-runbook-2026-11-24.md 작성(빌드 기록 §2·§6을 근거로, 모니터 1·2·폰 QR·장애 절차·리허설 체크리스트).
5. 모델 개선 설계 문서 docs/pm/research/model-improvement-2026-11.md: 규칙 재검토 표(바꿀 것/지킬 것/이유), 후보 목록(기대 효과·비용·A5000 5장 기준 일정), 3주 계획. 자료: docs/ML_EXPERIMENT_MATRIX.md, docs/handoffs/ML-experiments-next-session.md, docs/decisions.md D184~D197, results/EXP-P48-001.json. 실험 실행은 하지 않는다.

각 작업 끝에 "한 일 / 검사 결과 / 미측정·남은 조건 / 결정 필요" 네 줄로 보고한다.
```

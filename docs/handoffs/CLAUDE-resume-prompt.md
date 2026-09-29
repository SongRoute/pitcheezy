# Claude Code에 붙여 넣을 재개 프롬프트

아래 코드 블록 전체를 복사한다. 상세 상태와 실행 경계는 연결된 인수인계에 있으므로 이전 대화 전체를 붙일 필요는 없다. 이전 판은 git 이력에 보존된다. S0를 바로 돌리려면 `실행 승인: 없음` 줄을 `실행 승인: S0 census만`으로 바꿔 붙여 넣는다.

```text
/Users/song/Projects/pitcheezy에서 작업을 이어가줘. 완료한 구현·검토를 반복하지 말고 현재 상태를 이어받아줘.

먼저 AGENTS.md, CLAUDE.md, docs/AI_COLLABORATION.md, docs/handoffs/CLAUDE-resume-2026-09-29.md를 읽고 인수인계의 읽기 순서를 따라줘. git status/log, ListAgents, docs/decisions.md 꼬리로 이후 변경과 실행 중 작업을 확인하고, 기존 변경과 봉인 산출물은 보존해줘.

현재 상태(2026-09-29, D95 커밋 215348a 이후):
- 브랜치 codex/ml-matrix-execution, PR #34. reset·merge 금지. 로컬 `runs` symlink 변경은 건드리지 마.
- D93: Opus 5.5 단독. ≤2025 정책 검증 코드(런타임 v2, 요청 생성기, DR 추정기 v2, τ 규칙, V2/V3, 9단계 실행기) 완료. 같은 모델 적대적 검토 3회 반영(독립 검토 아님), 검토한 코드 5186810.
- D94: 코드 검토 게이트(S0–S2) PASS. 독립 검토 게이트(S3 이후)는 닫혀 있다.
- D95: S0 등록값 반영(D-4 문턱 1.0·0.0, source_commit 5186810). 등록·실행 스위치 4개(registered, status, execution.enabled, execution.real_data_enabled)는 꺼져 있다.
- 실제 자료 로드·BC 생성·bind·추론·OPE·2026 접근은 0.

실행 승인: 없음

할 일(인수인계 §3.2 순서):
1. 실행 승인이 없으면 S0 실행 승인 여부를 번호 선택지로 먼저 물어줘.
2. S0가 승인되면 스위치 4개를 켜고 커밋·푸시한 뒤, heavy lock과 ListAgents를 확인하고 §3.2 명령 그대로(.venv/bin/python, 새 출력 디렉터리 S0-census-a1) 세션 밖(tmux pz:runs 창)에서 실행해줘. 끝나면 census 결과와 실측 비용을 요약하고, D-3 무투구 목록 조정 여부와 다음 stage(S1·S1b) 승인을 물어줘.
3. stage마다 따로 승인받아줘. 결정에 영향받지 않는 구현·검사는 확인 질문 없이 진행해줘.

지킬 것:
- 로깅 BC 확률을 지원 마스크로 재정규화하지 마. G0는 member별 보정→5확률 평균→공통 앙상블 혼합 순서를 보존해줘. 결정 뒤 정보로 적합·평가 모집단을 고르지 마.
- Opus 5.5 단독. Fable 사용 금지, Claude→Codex 재호출 금지, 같은 모델 검토를 독립 검토라고 부르지 마.
- 승인된 stage 밖의 실제 payload 로드·BC 생성·실행은 하지 마. 2026 자료 추가 열람·수집·fit·추론·OPE, 정책/서비스 승격, 봉인 파일 덮어쓰기, PR merge 금지.

끝낼 때 바뀐 파일·검사 근거·남은 조건·다음 작업을 보고하고 docs/SESSION_HANDOFF.md와 docs/handoffs/ML-experiments-next-session.md를 갱신해줘.
```

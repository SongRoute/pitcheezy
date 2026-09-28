# Claude Code에 붙여 넣을 재개 프롬프트

아래 코드 블록 전체를 복사한다. 상세 상태와 실행 경계는 연결된 인수인계에 있으므로 이전 대화 전체를 붙일 필요는 없다. 이전(D91) 프롬프트는 git 이력에 보존된다.

```text
/Users/song/Projects/pitcheezy에서 이전 Codex–Claude 공동 작업을 이어서 진행해줘. 새로 시작하거나 완료한 실험·구현을 반복하지 말고 현재 상태를 이어받아줘.

먼저 AGENTS.md, CLAUDE.md, docs/AI_COLLABORATION.md와 docs/handoffs/CLAUDE-resume-2026-09-29.md를 읽어줘. 인수인계의 읽기 순서대로 D92 보고서·계약·config·검토 패킷을 확인해줘. git status/branch/log, ListAgents, decisions 꼬리와 SSD 작업대장(coordination/20260929-policy-materialization/board.json)을 확인해 이후 변경·실행 중 작업이 있는지 파악하고, 기존 변경/봉인 산출물을 보존해줘.

현재 상태:
- 브랜치 codex/ml-matrix-execution, 기존 PR #34. reset하거나 merge하지 마.
- D92(COOP-017): 완전한 정책 식별자(ML-POLICY-IDENTITY-v1)와 실제 연결 검증(bind_components/BoundComponents, 요청별 지원 표=pool 확인, verify_components, 정책 경로=평가 경로 합성 probe) 구현·합성 검사 완료. TRAIN BC(BC-P 주 제안·BC-E 비교)·PA/시간 규칙 R1–R8·≤2025 검증 S0–S6 계획은 제안·미등록 계약/config로 준비. Sol/Astra 독립 검토는 미수행(패킷만).
- 실제 BC 생성·payload 로드·bind·추론·OPE·2026 접근은 0. G0 연구 기준선 유지, ARM-A 보류, ARM-B 비활성, policy_frozen=false.

이번에 이어 할 일은 인수인계 §3 순서다: (1) 가능한 협업 환경이면 Sol 재현과 Astra 핵심 검토 1회, 불가능하면 사용한 척하지 말고 미완료로 둔다. (2) Song 결정 D-1~D-11이 없으면 결정이 필요한 항목을 짧게 묻는다. (3) 결정에 영향받지 않는 범위에서 실행기 run_policy_validation.py와 요청 생성기(R2·R3(c)·R4b·R4c·R8)를 실데이터 없이 구현·합성 검사한다.

원장 분모는 제출된 요청/PA이고 history 검사는 국소 일관성이며 식별자는 같은 부품의 연결만 보장한다는 점을 유지해줘. 로깅 BC 확률을 지원 마스크로 재정규화하지 말고, G0는 member별 보정→5확률 평균→공통 앙상블 혼합 순서를 보존해줘.

Opus 메인·Sol 서브·Astra 필수 최소 체제를 유지하고 Fable은 사용하지 마. Claude가 Codex를 재호출하는 순환 위임은 하지 마. 이 프롬프트로 실제 payload 로드·BC 생성·실험 실행을 승인했다고 해석하지 마. 2026 자료 추가 열람·수집·fit·추론·OPE, 정책/서비스 승격, 봉인 파일 덮어쓰기·PR merge는 하지 마. 이미 허용된 구현 작업은 반복 확인 질문 없이 진행해줘.

시작할 때 현재 상태와 첫 작업을 짧게 알리고 실제 작업을 진행해줘. 완료 시 바뀐 파일·검사 근거·남은 조건·바로 다음 작업을 보고하고 docs/SESSION_HANDOFF.md와 docs/handoffs/ML-experiments-next-session.md를 갱신해줘.
```

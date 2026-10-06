# 연구 PM 상태·실행 큐

초기값: 2026-10-06 09:46–09:48 감사에서 이관. 아래는 실시간 프로세스 상태가 아니다. 책임 연구 PM(Astra), 실제 기록은 지정 작업자. `/root/research_pm`이 이 전환 턴에 관리 역할 인수를 회신했다. 기존 작업자 `/root/status_research`(Astra)에 P47 준비를 followup으로 실제 배정했으며, 경로 생성·코드 작업·fit 시작 증거는 아직 없다. 새 Sol 호출은 `agent thread limit reached`로 실패했고 Sol 실행/쿼터 소진으로 기록하지 않는다.

최근90fit 완료·실패0(전체 누계 아님), 연구 G15 mix·서비스 G0, D131/D155 유지. 선수ID는 미지 선수 악화 가능, dropout 후속 후보. P47은 아직 초안·미등록·미실행. 당시 GPU0%·MPSfit 없음.

| ID | 상태 | 다음 실행 | 자원/선행조건 | 완료 기준 |
|---|---|---|---|---|
| RES-001 P47 | 준비 작업 배정 | 기존88860a4 기반 4팔×seed80–83 최소 구현·CPU 합성 검사·설계 명료화 | 현재 prep에는 fit·DEV평가·등록 금지 | 작성 후 별도 Astra 검토 → PM 수용 → 별도 등록/실행 배정 |
| RES-002 후속 큐 | 제안 | P47 결과에 따라 가설·우선순위 결정 | 반복 DEV 노출 명시 | 비교 가능 config·비용·검토 |
| RES-003 독립 평가/논문화 | 준비 | 미노출 자료 후보와 확증 절차 정리 | 2026 학습/선택 금지 | 등록 가능한 평가안, 미완료는 미완료로 표시 |

근거: [연구 인수인계](/Users/song/Projects/pitcheezy-pm-jobs/20261006-four-hour/RESEARCH-HANDOFF.md), [P47 초안](/Users/song/Projects/pitcheezy-pm-jobs/20261006-four-hour/NEXT-RESEARCH-DRAFT.md), [기존 실험 원장](../../experiments.md). 큐 준비와 실제 학습 실행은 구분한다.

# 처분 — 임베딩·잔차 계열 연결 검토 (040c8b4)

- 검토: `docs/reviews/OPUS-review-embed-residual-040c8b4-2026-10-04.md`, 커밋 `dce6519`(branch `codex/embed-residual-review-1004`). 같은 모델(Opus 5.5) 별도 세션 검토이고 Codex 검토는 아직이다. 상태 표기: `same_model_reviewed; Codex_review_pending`.
- 검토 결론: CUDA 사전 점검(18개 과제)과 이후 등록 모두 막는 문제 없음. 코드 결함은 찾지 못했다.
- 처분 작성: 위임 PM 작업자 cycle-016 (T16), 2026-10-04 09:4x KST. fit·서버·GPU 사용은 없다.

| 번호 | 내용 | 처분 |
|---|---|---|
| F1 | 기존 팔(control, residual, periodic, d10, TabM) 저장본을 새 팔로 읽는 경우를 실행기 단계에서 시험하지 않음 | 수용. 검토자 테스트 cherry-pick `3ab7f8d` |
| F2 | `bundle = structure + head_dropout` 항등식과 Brier 항등식 미검사 | 수용. 같은 cherry-pick |
| F3 | 옛 계열 회귀 목록(DROPOUT-RUN, FRONTIER-RUN)과 다른 계열에 붙인 embed 대비 키 거부 미검사 | 수용. 같은 cherry-pick |
| F4 | `FACTORS` 라벨이 팔 정의에서 유도되는지 미검사 | 수용. 같은 cherry-pick |
| F5 | 설계 노트 상태 줄이 040c8b4 이후와 맞지 않음 | 수용. 사전 점검 전에 고침. 노트 sha가 `5ce7490a…`에서 `7f9b00fb0a7f077b04dfbb9189638069693265b7c2bfce3e1bde7b4d86d62797`로 바뀌어 DRAFT config에 다시 고정함 |
| F6 | 표와 메모가 잔차 쪽 항등식만 설명함. 여섯 대비 중 실제 독립은 넷 | 수용. 채점기 문구와 seed_block 메모를 고침. 표 문구 테스트는 '독립은 넷'으로 바꿈 |
| F7 | `MODEL_SOURCES`가 늘어 옛 과제는 자기 스냅샷이나 `--source-manifest`로만 채점 가능 | 정보. 운영 규칙으로 기록 |
| F8 | 로더가 seed·n_token·length를 비교하지 않음 | 정보. TabM과 같음. task identity가 seed를 묶음 |
| F9 | 큐가 날짜를 글자 그대로 씀. 10/4 19:00 KST 뒤에는 큐로 사전 점검을 띄울 수 없음 | 정보. 늦어지면 DRAFT 날짜를 고친 뒤 사전 점검. 등록 때 digest는 어차피 바뀜 |
| F10 | "설계 전에 읽은 결과" 목록이 파일마다 조금 다름 | 수용. 레지스트리 docstring, 설계 노트, config purpose를 "P24/P27/P29/P30/P31 결과가 나온 뒤"로 통일함. 이 작업자가 직접 읽은 표는 P27과 P31 |

## 확인
- 명령: `cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_embed_residual.py tests/test_direct_embed_residual_registry.py tests/test_direct_periodic_residual_model.py tests/test_cuda_tabm_mini.py tests/test_cuda_workload_dropout.py -q -p no:cacheprovider`
- 결과: 178 passed (4.22 s).
- 편차: 작성자는 040c8b4 커밋 전에 zsh glob 오류로 파일 목록이 비어 `tests/` 전체를 실행했다. 1242 passed, 1 skipped, 113 s, 스레드 1. 맥 보호 시간대의 "전체 테스트 금지" 규칙을 어긴 것이므로 기록한다. 이후 실행은 모두 파일 목록을 명시했다.

## 다음
1. CUDA 사전 점검(18개 과제, GPU3/4, 공통 lock 디렉터리)을 이 처분 커밋에서 새 서버 아카이브로 진행한다. 조건은 10/4 19:00 KST 전이고, 다른 계열 큐가 돌고 있지 않아야 한다.
2. 그다음 등록한다. 마감은 명시적으로 적는다.

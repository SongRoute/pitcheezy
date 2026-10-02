# 재개 인수인계 — 2026-10-03 (세션 "구종 추천 모델 및 CV 모듈 현황")

이 세션(10/2 07:00 ~ 10/3 00:40 KST)이 한 일과 다음 세션이 이어받을 지점을 한 파일에 적었다. 결정 한 줄 기록은 [decisions.md](../decisions.md) D134(CV)·D137·D139·D141·D143·D145·D147·D150·D151·D152·D154·D155에 있다. 실험 수치는 [results-all.md](../results-all.md) 4장(EXP-P15·P16)과 5.7·5.8절(EXP-P17·P18)에 있다.

## 1. 지금 상태 한눈에

| 축 | 상태 | 근거 |
|---|---|---|
| 예측 모델 | **연구 기준 = G15 mix**(직접 head와 G0-global의 같은 seed 확률 1/2 혼합, 5 seed 평균). 전체 MLB NLL 1.486347, G0 1.490408 | D143 확인(EXP-P16-001), D145 교체. 목록 `configs/G15-MIX-RESEARCH-FROZEN-v1.json`, 검증 `.venv/bin/python scripts/freeze_g15_mix_research.py verify` |
| 추천 정책 | **G0 그대로**(식별자 `c13cc989…`, τ 0.1 P3). 2026 평가 "개선 근거 없음"(D131). 혼합 예측기로의 재검증은 **닫음** | D152·D155: 예측기 교체로 생긴 추천 차이(평균 TV 0.0323)가 계획 seed만 바꾼 차이(0.0497)보다 작음 |
| 서비스·데모 | G0 그대로, "검증 전 실험 버전" 표시. 실시간 서비스는 다른 세션(D134·D144·D148·D149·D153)이 운영 | 이 세션은 건드리지 않음 |
| CV 의도(팀원) | 형식 계약 맞음(우리 검사 747139 297/297, 849843 39/39). 시연 경기 849843 서버 연결 완료, 화면 표시는 작업 지시로 남김. M3(다른 경기 정확도)는 팀원이 사람 라벨 50구를 모으는 중(0구) | D150·D151, `demo/intent-setup` `5a102ea` |

닫힌 줄기 두 개: ① Statcast만으로 예측 정확도 올리기(D133 조건에서 P15가 통과했으므로 혼합으로 마무리) ② 혼합 예측기로 정책 재검증(D155). 둘 다 CV 의도 자료(M3)와 ARM-A 설계가 올 때 다시 연다.

## 2. 어디에 무엇이 있나

브랜치·worktree (모두 깨끗함, 원격에 올리지 않음):

| 브랜치 | worktree | 내용 |
|---|---|---|
| `codex/ml-matrix-execution` (주 체크아웃) | `/Users/song/Projects/pitcheezy` | 문서·결과·결정. 원격보다 앞서 있음(push 안 함) |
| `claude/exp-p16-direct-mix-confirm` `7dcd069` | `../pitcheezy-worktrees/exp-p16-direct-mix-confirm` | EXP-P15·P16 실행기·설정·결과(직접 head, 5 seed 확인) |
| `claude/exp-p17-policy-predictor-diagnostic` `a971a1d` | `../pitcheezy-worktrees/exp-p17-policy-predictor-diagnostic` | EXP-P17·P18 진단 실행기·설정·결과(P16 브랜치 위) |
| `demo/intent-setup` `5a102ea` | `../pitcheezy-worktrees/intent-setup` | `GET /api/watch/{game_pk}/setup`(포수 셋업 가로 위치), `demo/live-sync` `221436a` 기준 |

SSD(`/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/`):
- `ML-MATRIX-20260924/EXP-P15-001`, `EXP-P16-001`, `EXP-P17-001`, `EXP-P18-001`(각 `-smoke` 포함, `EXP-P17-001-smoke.failed1-fingerprint-rule`은 보존한 실패 smoke)
- `ML-MATRIX-20260924/coordination/EXP-P1{5,6,7,8}-001-chain.sh`·`.chain.status`·`.console.log`(실행 여부는 **여기**로 확인한다. `results/`에 파일이 없다고 미실행이 아니다)
- `DEMO-WS-2026/intent/849843.jsonl` + `849843.source.json`(팀원 커밋 `9f62b77`의 파일, sha256 `12c11e1d…`)

레포 결과 파일: `results/EXP-P15-001.json`, `EXP-P16-001.json`, `EXP-P17-001.json`, `EXP-P18-001.json`, `G15-MIX-RESEARCH-FROZEN-v1.json`.

팀원 레포 `Pitcheezy/transition-models` `feature/intent-v0`: 최신 `66224ad`(M3 진행 중). 우리가 올린 것은 `docs/INTENT_V0_WORK_ORDER.md`(`4d47fc9`)와 `docs/INTENT_V0_CONTRACT_FIXES.md`(`8301804`)뿐. 읽기 전용 사본은 임시 디렉터리에 있었고 다음 세션에서는 다시 받으면 된다.

## 3. 다음 세션이 할 일 (추천 순)

1. **화면에 포수 미트 위치 붙이기** — [작업 지시](UI-setup-display-work-order.md). `demo/intent-setup`을 화면 브랜치(`demo/ui-frame`)에 합치고 "방금 던진 공" 화면 존 그림에 가로 위치만 표시. 화면 담당 세션(PM 체계, D146 이후 PM 부트스트랩 문서 참조)이 맡는 것이 기본이고, 없으면 이 줄기를 이어받는다.
2. **팀원 M3 확인** — 사람 라벨 50구·평가 경기 2개의 `intent_accuracy_report_v0.json`이 오면 우리 검사로 재확인하고, 셋업 높이가 목표 대리값으로 쓸 만한지(849843에서는 실제 공보다 중앙값 1.35 ft 낮았음)를 ARM-A 설계의 전제로 정리한다.
3. **계획기 탐색 횟수(선택, 새 실험)** — seed만 바꿔도 1순위 구종이 11.3% 바뀐다(D155 부수 관찰). 탐색 횟수(samples 6)를 늘리면 안정되는지는 미측정. 하려면 설계안 → 승인 → 등록 순서.
4. 하지 않을 것: 2026 재평가, G0 서비스 교체, 봉인 파일 수정, 팀원 레포 수정.

## 4. 재개 방법

- 세션 시작: `ListAgents`, `git status`, `git log --oneline -15`, `tail docs/decisions.md`로 다른 세션의 변경부터 본다. 결정 번호가 겹칠 수 있으니(10/2에 D134가 두 번 쓰임) 쓰기 직전에 꼬리를 다시 확인한다. 다른 세션이 `docs/decisions.md`를 미커밋 상태로 두고 있으면 내 줄만 index에 올려 커밋한다(`git show HEAD:docs/decisions.md` + 내 줄 → `git hash-object -w` → `git update-index --cacheinfo`).
- 실험 실행 환경(정책·진단 실행기): `DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib PYTHONPATH="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/deps:experiments/pitchmdp" .venv/bin/python …`. 드라이버는 tmux `pz` 창에서 돌리고 상태 파일로 확인한다. 드라이버의 `pgrep -f` 보호가 내 셸 명령줄의 문자열에 걸릴 수 있으니(launch1-false-guard 선례) 실행 명령에 실행기 이름을 넣지 않는다.
- 서비스 테스트: `cd apps/observer/backend && /Users/song/Projects/pitcheezy/.venv-observer/bin/python -m pytest tests -q`(238개). 실험 테스트: 각 worktree의 `experiments/pitchmdp`에서 `pytest tests/test_run_ml_direct_head*.py tests/test_diagnose_policy_*.py -q`.
- 팀원 산출물 재검사: `apps/observer/backend/observer_app/intent.py`의 `validate_intent_estimate`에 JSONL 각 줄을 넣는다(D150의 방법).

## 5. 이 세션에서 배운 것

- "미실행"이라고 보고하기 전에 SSD의 `coordination/*.chain.status`와 실행 디렉터리를 본다(P15는 이미 끝나 있었는데 `results/`만 보고 틀리게 보고했다).
- 팀원 작업 지시서의 스키마 요약이 빠뜨린 항목(`plate_feet` 키, `blocked_by` 값)이 불일치의 원인이었다. 계약을 요약해 보낼 때는 검사 함수가 요구하는 키·값을 전부 적는다.
- 추천 정책의 계획 Q는 6회 탐색 평균이라 잡음이 크다. 예측기 개선 효과를 정책에서 보려면 먼저 탐색 잡음을 재고 넘겨야 한다.

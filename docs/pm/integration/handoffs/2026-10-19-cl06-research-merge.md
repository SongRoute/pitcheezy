# CL-06 준비 — 연구 병합 뒤 실시간 ARM-B 경로 고정 (2026-10-19)

- 작업: CL-06 준비(SongRoute/pitcheezy#37 댓글 6257222210에서 잡음). 작성자 Opus 5.5 하위 에이전트.
- 워크트리 `/Users/song/Projects/pitcheezy-worktrees/ws-cl06-1019`, 브랜치 `ws/cl06-research-merge-1019`,
  기반 `cc0d174`(연구 `93a80ef`를 공개 앱 `5c9c31a`에 병합). **배포 동결 중에는 출시 브랜치에 병합하지 않는다.** 푸시 안 함.

## 문제

연구 D117~D127이 `configs/ML-POLICY-MATERIALIZATION-v1.json`(sha `9f32615e…` → `869b2611…`)과
`experiments/pitchmdp/pitchmdp/policy_runtime.py`(v2 → v3)를 제자리에서 바꿨다. 실시간 경로
(`live_armb.py` → `scripts/demo_precompute.py` `LivePolicy`)는 D110 시점 설정 + `addendum-{1..6}-d107` 사슬과
S2 인증 구성요소 identity, tau 고정 identity `a6dffaea…`를 요구하므로 시작 시 거부했다.

## 왜 "새 경로에 파일 복사"만으로는 안 되는가

`policy_identity.project_sources`는 소스 닫힘(39개 파일)을 **자기 레포 루트(`policy_identity.py`의 `parents[3]`)
기준 상대 경로 → sha256**으로 identity에 넣는다. `policy_runtime.py`는 `RUNTIME_MODULES`로 항상 포함된다.
그래서 `policy_runtime_live.py` 같은 다른 이름은 경로 키가 달라져 identity가 바뀐다. 연구 쪽 identity
(`c13cc989…`)도 같은 경로의 v3 바이트를 고정하므로 한 경로에 두 버전을 둘 수 없다.

## 한 일

1. `apps/observer/live_runtime_d110/` — 레포 배치를 그대로 흉내 낸 고정 사본(레포 루트 역할).
   `experiments/pitchmdp/pitchmdp/` 전체(53개)와 `experiments/pitchmdp/scripts/` 중 필요한 18개를 `5c9c31a` 바이트 그대로.
   상대 경로·바이트가 S2 인증 때와 같으므로 구성요소 identity와 `a6dffaea…`가 그대로 검증된다.
   `MANIFEST.json`에 파일 71개의 sha256과 출처 커밋을 적었다.
2. `configs/live/ML-POLICY-MATERIALIZATION-v1.d110.json` — `5c9c31a`의 설정 바이트 사본(sha `9f32615e…`).
   d107 addendum 1~6은 원래 자리를 그대로 쓴다(바뀌지 않았고 parent sha 사슬로 검증됨).
3. `scripts/demo_precompute.py` — `REGISTRATION`을 위 사본으로, `_paths()`가 고정 사본을 `sys.path` 맨 앞에 둔다.
   새 `verify_live_runtime()`: 매니페스트 바이트·설정 sha를 확인하고, 이미 다른 곳(experiments/pitchmdp)에서
   `pitchmdp`나 사본 스크립트 모듈이 import된 프로세스는 "start an isolated process"로 거부한다.
   precompute/sync와 LivePolicy 모두 이 경로를 쓴다.
4. `tests/test_demo_precompute.py` — T7 실시간 테스트를 새 프로세스에서 돌린다(다른 테스트 모듈이 수집 때
   연구용 `pitchmdp`를 import하므로). 고정 사본 확인·격리 거부 테스트 1개 추가(T7 불필요).
5. 연구 쪽 파일(experiments/, 연구 설정·addendum)은 건드리지 않았다.

## 검증

- `tests/test_demo_precompute.py`: 10 passed(기존 9개 + 새 1개); 연구용 `pitchmdp`를 수집 때 import하는
  `tests/test_2026_policy_contract.py`와 같이 돌려도 33 passed(격리 subprocess 확인). T7 실시간 테스트는 849843 50개 상태가 watch-along
  사전계산과 비트 동일, bind 1회. 최대 RSS 약 1.8GB(단독 실행 측정).
- 연구 정책 테스트(`experiments/pitchmdp/tests/test_policy_*.py`, `test_matrix_policy.py`, `test_rollout_policy.py`):
  `93a80ef` 134 passed / 이 브랜치 134 passed.
- 관전 백엔드 합성 테스트(`.venv-observer`, `apps/observer/backend/tests`): `5c9c31a` 22 failed / 987 passed,
  이 브랜치 21 failed / 988 passed, 1 xfailed 동일. 실패 목록은 같고, 기준에만 있던
  `test_publisher_background::test_blocked_prefix_cannot_starve_later_eligible_path`는 기준에서 단독 재실행 시 통과(시간 의존 불안정).
  남은 21개는 기준에도 있는 실패(viewer_off_golden 14, rev3 4, profile_reference 1, public_viewer e2e 2).
- 주의: 자기 자식 프로세스에 신호를 보내는 7개 테스트를 빼려 했으나 `--deselect` 경로가 rootdir 기준이 아니어서
  빠지지 않았다(양쪽 모두 실행됨). 모두 tmp 디렉터리에서 테스트가 직접 띄운 자식만 대상으로 하며 기록기·수집 프로세스와 무관.

## 위험

- **두 런타임 분기:** 실시간은 v2(D110), 연구는 v3. 같은 이름 `pitchmdp`라 한 프로세스에 둘을 못 올린다.
  `PITCHEEZY_OBSERVER_LEGACY=1`(예전 추천기가 먼저 연구용 `pitchmdp`를 import)로 띄우면 LiveArmB가 거부한다.
  `serve.sh` 기본값은 0이라 지금 운영 경로는 영향 없음.
- **identity 표류:** 사본 파일 하나라도 바뀌면 매니페스트에서 거부(닫힘 밖 파일 포함). 파이썬·numpy 등 환경이 바뀌면
  identity의 `environment`가 달라져 거부된다(이전과 동일).
- 사본은 연구 수정(버그 수정 포함)을 받지 않는다. 실시간 정책을 v3로 옮기려면 새 S2/tau 재인증이 필요하다.

## 출시 뒤 병합에 필요한 것

1. 동결 해제 뒤 이 브랜치를 출시 브랜치 최신 위에 다시 병합(충돌 지점: `scripts/demo_precompute.py`, 테스트 파일).
2. 독립 검토(작성자와 다른 에이전트): 사본 바이트 = `5c9c31a`, 격리 검사, 테스트 subprocess화.
3. 맥미니 운영 환경(`.venv-observer`/`serve.sh`)에서 실시간 경로 시작 확인 — 기록기·수집 프로세스와 겹치지 않는 시간에.
4. 장기: 실시간 정책을 v3로 재인증할지, 사본을 유지할지 결정(`결정 필요`).

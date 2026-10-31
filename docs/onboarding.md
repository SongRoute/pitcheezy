# 신규 팀원 온보딩

소유: 통합 문서 담당 · 갱신: 진입 명령/의존성/권한 경로 변경 시 · 마지막 확인: 2026-10-06(기존 Mac CPU smoke).

[START HERE](../README.md) → [운영 안내](pm/overview.md) · [프로젝트 지도](pm/project-map.md). 아래 명령은 프로젝트 루트 기준이다.

## checkout과 권한부터 구별

2026-10-06 감사에서 공개 GitHub `SongRoute/pitcheezy`의 기본 main은 `70aa5499`이고, 현장 공통 문서 checkout은 `f7e8f475`/`codex/ml-matrix-execution`이었다. 공개 clone에 최신 로컬 PM 문서·서비스 후보·비공개 모델/자료가 모두 있다는 뜻이 아니다. 새 팀원은 먼저 담당자에게 **이 작업용 저장소·정확한 branch/commit·공개/비공개 문서 범위**를 받는다. 현재 로컬 checkout의 smoke 외에, 공개 main API SHA와 같은 로컬 object `70aa5499`의 `src`·두 테스트·pyproject를 임시 git archive로 추출한 최소 CPU 검사도 독립적으로 **10 passed**를 확인했다. 실제 원격 clone 전체 설치나 현재 문서·앱·모델/자료 재현을 검증한 것은 아니다.

[현장 인벤토리](pm/operations-inventory.md)는 호스트별 운영 기록이다. 경로/계정 정보를 공개 복사하지 않으며, 이번 문서 정리가 GitHub push 승인은 아니다. 공개 정본 수렴은 열린 PR·CI·보호 규칙·재현 검사를 별도 검토한 후 수행한다.

## 30~60분 온보딩

1. **0~10분:** 위 운영 안내·현재 상태·공통 계약을 읽는다. 현재 결정과 날짜가 붙은 과거 관측을 구별하고, 자신의 담당 PM과 파일 작성권을 확인한다.
2. **10~20분:** `git status --short`, `git branch --show-current`, `git rev-parse HEAD`로 작업 위치를 기록한다. 기존 dirty를 정리하지 않는다. 제품 작업은 담당자가 배정한 별도 worktree에서 한다.
3. **20~35분:** 아래 데이터 없는 CPU 계약 smoke를 실행하고 명령·출력·commit을 기록한다. 데이터 수집/학습/서버 실행은 하지 않는다.
4. **35~50분:** [구종·격자 계약 테스트](../tests/interfaces/test_grid.py)와 [해당 구현](../src/pitcheezy/interfaces/grid.py)을 읽고 입력·출력·좌우/단위를 설명해 본다. 이 격자 계약 통과가 서비스 G0의 실제 지원 범위나 성능을 증명하지는 않는다.
5. **50~60분:** 담당 PM에게 한 개의 작은 작업안(변경 파일·완료 조건·검사·필요 자료·막힘)을 제출한다. 독립 검토와 통합 담당이 정해지면 구현한다.

### 데이터·모델 없이 실행하는 첫 검사

Python 3.11 이상이 필요하다. 승인된 개발환경에 numpy/pandas/pytest가 있으면 프로젝트 루트에서 실행한다:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/interfaces/test_pitch_types.py tests/interfaces/test_grid.py -q -p no:cacheprovider
```

이 Mac에서는 `python` 대신 `.venv/bin/python`으로 위 명령을 실행해 **10 passed**를 확인했다(2026-10-06 정리 작업). 실제 데이터·모델·API·GPU를 사용하지 않는 구종/격자 계약 검사다. 새 컴퓨터에 환경이 없다면 자신의 독립 환경을 만든 뒤 최소 의존성을 설치한다(네트워크 다운로드 발생):

```sh
pz_onboarding_env="$(mktemp -d "${TMPDIR:-/tmp}/pitcheezy-onboarding.XXXXXX")"
python3 -m venv "$pz_onboarding_env"
"$pz_onboarding_env/bin/python" -m pip install numpy 'pandas>=2.2' pytest
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$pz_onboarding_env/bin/python" -m pytest tests/interfaces/test_pitch_types.py tests/interfaces/test_grid.py -q -p no:cacheprovider
```

이번 정리에서 별도 신규 venv를 만들고 numpy/pandas/pytest만 설치한 뒤 같은 명령으로 **10 passed**를 확인했다. 기존 site-packages를 공유하지 않았다. 이 결과는 현재 로컬 checkout의 최소 계약 경로에 한정되며 공개 main의 전체 앱 설치 성공을 뜻하지 않는다. 공개 main 최소 계약 재현 근거는 [독립 결과](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/docs/public-main-probe-001/RESULT.json)에 별도로 보존했다. 전체 개발 의존성은 [pyproject.toml](../pyproject.toml)이며 위 최소 환경은 전체 연구/웹 실행 환경이 아니다. 오류가 나면 Python 버전·명령·오류를 전달하고 운영 가상환경을 고치지 않는다.

모델/데이터가 없으면 여기까지 진행할 수 있다. 실제 앱 기동은 [Observer 안내](../apps/observer/README.md)와 담당 서비스 PM에게 모델/자료 manifest·접근 권한·빈 포트·새 출력 위치를 받은 뒤 진행한다. 기존 8766/8771/8775 실행 스크립트를 첫 실습으로 실행하지 않는다. 자료 경로가 보인다는 것과 열람/학습/재배포 권한은 다르다. 2026 자료의 연구 학습·튜닝·모델 선택 금지 및 sealed 결과 보존은 [공통 계약](pm/contracts.md)을 따른다.

# 온보딩 환경 재현

소유: 통합 문서 담당 · 갱신: 첫 검사·최소 의존성 변경 시 · 갱신일: 2026-11-01.

[온보딩으로 돌아가기](onboarding.md). 아래는 실제 자료·모델·API·GPU 없이 구종/격자 계약을 검사하는 최소 경로다. 전체 연구·웹 설치나 전시 서비스 검증을 대신하지 않는다. 명령은 작업 worktree 루트에서 실행한다.

## 기존 개발환경에서 검사

Python 3.11 이상과 numpy/pandas/pytest가 있는 승인된 환경을 사용한다. `python`은 해당 환경의 실행 파일로 바꿔도 된다. 먼저 `python --version`과 `git rev-parse HEAD`를 기록한다.

```sh
PYTHONPATH=src:.:apps/observer/backend PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/interfaces/test_pitch_types.py tests/interfaces/test_grid.py -q -p no:cacheprovider
```

`PYTHONPATH`는 venv의 편집 설치가 다른 checkout을 가리켜도 현재 worktree의 코드를 우선 읽게 한다. 기존 `.venv`·운영 환경의 의존성을 수정하지 않는다. 실패하면 Python 버전·명령·오류 원문을 담당자에게 전달한다.

## 새 컴퓨터의 최소 환경

개발환경 생성·네트워크 다운로드가 허용된 작업에서만 다음을 사용한다. 각 줄은 같은 셸에서 실행한다. 기존 환경의 site-packages를 공유하지 않는 임시 venv이며, 전체 개발 의존성은 [pyproject.toml](../pyproject.toml)에서 확인한다.

```sh
pz_onboarding_env="$(mktemp -d "${TMPDIR:-/tmp}/pitcheezy-onboarding.XXXXXX")"
python3 -m venv "$pz_onboarding_env"
"$pz_onboarding_env/bin/python" -m pip install numpy 'pandas>=2.2' pytest
PYTHONPATH=src:.:apps/observer/backend PYTHONDONTWRITEBYTECODE=1 "$pz_onboarding_env/bin/python" -m pytest tests/interfaces/test_pitch_types.py tests/interfaces/test_grid.py -q -p no:cacheprovider
```

위 설치 명령은 버전을 모두 고정한 환경 잠금 파일이 아니다. 새 환경의 성공 여부·설치 버전은 실행할 때 기록한다. 이번 문서 정비에서 새 환경 설치와 원격 clone 전체 재현은 **미측정**이다.

## 과거 검증은 별도 증거

2026-10-06의 공개 main·현장 checkout·임시 archive·신규 venv 검사 결과는 [당시 내용을 보존한 원문](https://github.com/SongRoute/pitcheezy/blob/6e2038fd0010657e6f6f2d1e49820490f6195da4/docs/onboarding.md)에 있다. 원문의 호스트 절대 경로는 그 호스트에서만 확인할 수 있다. 과거 통과 수를 현재 설치 결과로 옮기거나 앱·모델 재현 성공으로 확대하지 않는다. 현재 작업의 검사 결과는 해당 PR에 실행 커밋과 함께 남긴다.

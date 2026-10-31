# 문서 검사와 링크 그래프

상위: [문서 규약](documentation.md) · 구현: [check_docs.py](../../scripts/check_docs.py) · 검증: [합성 검사](../../tests/test_check_docs.py)

별도 의존성 없이 저장소 루트에서 실행한다. 일반 문서 3,000자 상한·상대 링크의 대상 존재·정확한 CLAUDE import를 검사한다. 최소 길이를 강제하지 않는다.

```sh
python3 scripts/check_docs.py AGENTS.md docs/pm/overview.md
python3 scripts/check_docs.py --base origin/main
PYTHONPATH=src:.:apps/observer/backend .venv/bin/pytest tests/test_check_docs.py -q
```

`--base`는 해당 Git 기준과 현재 작업트리 사이 변경된 추적 Markdown만 검사한다. 미추적 새 문서는 경로를 직접 전달하거나 stage한 뒤 검사한다. 다른 브랜치의 기존 이력 전체를 갑자기 실패시키지 않는다.

보존 예외가 필요하면 `docs/maintenance/length-exceptions.json`에 정확한 파일 경로와 사유를 기록한다. 예외는 길이만 면제하고 링크 검사는 유지한다. 고정 등록·증거·과거 원문·생성 파일만 보존 사유를 갖는다. 현행 안내의 미전환을 영구 면제하지 않는다.

## 그래프 생성

```sh
python3 scripts/check_docs.py --graph graphify-out/doc-links-2026-11-01.json
```

Git이 추적하는 Markdown의 링크를 읽어 노드(경로·글자 수)와 연결(출발·도착·표시 이름·대상 존재)을 생성한다. 기존 출력은 덮지 않는다. 링크를 바꾸면 새 출력으로 재생성하며 정본은 Markdown이다. 역방향 참조는 `to`로 모으면 되므로 수동으로 이중 관리하지 않는다. 공개 배포용 파일이 아니라 로컬 탐색물이다.

## 검사 범위

일반 inline 링크와 full/collapsed reference 링크를 처리하고 fenced/inline code를 제외한다. 괄호가 있는 대상은 `<…>` 또는 escape 표기를 사용한다. HTML·축약 reference·제목 anchor·외부 HTTP 상태는 검사하지 않는다. 호스트 절대경로·저장소 밖 경로는 탐색 그래프에서 제외한다. 그러므로 통과가 모든 링크의 완전성을 보증하지 않는다. 과학적 정확성이나 자료 접근 권한도 판정하지 않는다.

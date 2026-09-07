"""계약 테스트 자리표시자.

docs/interface-spec.md가 동기화되면 RE24·텐서·Q 스키마 계약 테스트로 교체한다.
PR 전 `pytest tests/interfaces -q` 통과 필수.
"""

import pitcheezy


def test_package_importable() -> None:
    assert pitcheezy.__version__

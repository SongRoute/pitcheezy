"""데이터 수집 (규약 §5).

사용:
    python scripts/fetch_data.py --version-id d20260908-s2326
    python scripts/fetch_data.py --version-id d20260908-s2326 --verify

규칙:
    - 수집은 이 스크립트로만. pybaseball 쿼리 파라미터를 코드에 고정하고 parquet sha256을
      검증해 누가 받아도 같은 파일이 나오게 한다.
    - 출력: $PITCHEEZY_DATA_DIR/  (기본 ./data).
      기록: data/versions.md (버전 ID, 시즌, 파라미터, 행 수, sha256, 수집 스크립트 커밋).
    - 데이터 버전 ID: d{YYYYMMDD}-{tag}.
    - 2026 검증셋은 정규시즌 종료 직후 1회 갱신 후 고정(frozen). 포스트시즌 제외 (ADR-2).

현재 상태: 뼈대. 인자 파싱만 있고 본 로직은 미구현.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--version-id", required=True, help="데이터 버전 ID. 형식 d{YYYYMMDD}-{tag}"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="수집하지 않고 기존 parquet의 sha256을 data/versions.md와 대조만 한다",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"데이터 수집 미구현 (version_id={args.version_id}, verify={args.verify})."
    )


if __name__ == "__main__":
    main()

"""데이터 수집 CLI (규약 §5). 로직은 src/pitcheezy/data/statcast_fetch.py 에 있고 여기는 인자 처리만.

사용:
    python scripts/fetch_data.py --version-id d20260908-s2326 --seasons 2023 2024 2025 2026   # 수집
    python scripts/fetch_data.py --version-id d20260908-s2326 --verify                         # 해시 대조
    python scripts/fetch_data.py --version-id d20260908-s2326 --sample --sample-season 2025    # 픽스처

규칙:
    - 수집은 이 스크립트로만. 창 길이(14일)·시즌 범위(3/15~11/15)·game_type=R·정렬 키는 코드 상수로 고정.
      실제 사용한 날짜 하한/상한(--start-date/--end-date)은 manifest.json 과 data/versions.md 에 기록된다.
    - 출력: {out_dir}/{version}/statcast_{season}.parquet + manifest.json. out_dir 기본값은
      $PITCHEEZY_DATA_DIR (colab_runner.ipynb 가 Drive 경로로 설정), 없으면 <repo>/data.
    - 창마다 _parts/ 에 중간 저장하므로 세션이 끊기면 같은 명령을 다시 실행해 이어받는다.
    - 완료 시 data/versions.md 에 한 줄 append 하고 같은 줄을 출력한다. Colab 클론에서는 커밋할 수 없으니
      그 줄을 로컬 장부에 붙여 커밋한다.
    - 2026 검증셋은 정규시즌 종료 직후 새 버전 ID 로 다시 받아 고정(frozen). 포스트시즌 제외 (ADR-2).
    - --sample: 시즌 parquet 에서 투수 2명 × ~100구(타석 단위) → tests/fixtures/sample.parquet.
      로컬 테스트는 이 경로만. 풀 수집은 Colab.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from pitcheezy.data import statcast_fetch as sf
except ModuleNotFoundError as exc:  # pragma: no cover
    sys.exit(f"{exc}. 먼저 `pip install -e .` (레포 루트) 를 실행할 것")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        epilog=__doc__.split("\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version-id", required=True, help="데이터 버전 ID. 형식 d{YYYYMMDD}-{tag}")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="기본 $PITCHEEZY_DATA_DIR, 없으면 <repo>/data"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify", action="store_true", help="수집하지 않고 manifest 의 행 수·sha256 과 장부 행을 대조만"
    )
    mode.add_argument(
        "--sample", action="store_true", help="시즌 parquet 에서 픽스처 샘플 생성 (네트워크 없음)"
    )
    parser.add_argument("--seasons", type=int, nargs="+", metavar="YYYY", help="수집할 시즌 (수집 모드 필수)")
    parser.add_argument(
        "--start-date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="전역 하한. 테스트·부분 수집용"
    )
    parser.add_argument(
        "--end-date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="전역 상한. 기본 어제 (당일 데이터 미완)"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "data" / "versions.md",
        help="데이터 버전 장부. 기본 <repo>/data/versions.md",
    )
    parser.add_argument("--sample-season", type=int, metavar="YYYY", help="--sample 대상 시즌. 기본 manifest 의 최신 시즌")
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "sample.parquet",
        help="--sample 출력 경로. tests/fixtures 안이면 README 표도 갱신",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not sf.VERSION_RE.match(args.version_id):
        parser.error(f"--version-id 형식 오류: {args.version_id!r} (d{{YYYYMMDD}}-{{tag}})")
    out_dir = args.out_dir or Path(os.environ.get("PITCHEEZY_DATA_DIR") or REPO_ROOT / "data")
    version_dir = out_dir / args.version_id

    try:
        if args.verify:
            problems = sf.verify(version_dir, args.ledger)
            if problems:
                for p in problems:
                    print(f"FAIL {p}", file=sys.stderr)
                return 1
            print(f"OK {args.version_id}: manifest 와 파일·장부가 일치")
            return 0
        if args.sample:
            sf.run_sample(
                version_dir,
                out_path=args.sample_out,
                fixtures_dir=REPO_ROOT / "tests" / "fixtures",
                season=args.sample_season,
            )
            return 0
        if not args.seasons:
            parser.error("수집 모드에는 --seasons 가 필요 (예: --seasons 2023 2024 2025 2026)")
        sf.run_fetch(
            version=args.version_id,
            seasons=args.seasons,
            out_dir=out_dir,
            ledger=args.ledger,
            repo_root=REPO_ROOT,
            floor=args.start_date,
            ceiling=args.end_date,
        )
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

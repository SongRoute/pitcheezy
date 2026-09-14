"""데이터 수집 CLI (규약 §5). 로직은 src/pitcheezy/data/statcast_fetch.py, 여기는 인자 처리와 출력만.

사용:
    python scripts/fetch_data.py --seasons 2023 2024 2025 --holdout 2026 --tag s2325 --out data/raw
    python scripts/fetch_data.py --seasons 2023 2024 2025 --holdout 2026 --tag s2325 --dry-run   # 시즌마다 첫 하루만
    python scripts/fetch_data.py --verify d20260910-s2325 --out data/raw                          # sha256·행 수 대조

규칙:
    - 버전 ID = d{오늘 YYYYMMDD}-{tag}. 시즌 날짜·청크(월)·정렬 키·dtype 은 코드 상수 (statcast_fetch.py).
    - 출력: {out}/{version}/statcast_{season}.parquet, 홀드아웃은 {out}/{version}/holdout_{season}/ 아래.
      Drive 경로는 --out 으로만 준다 (기본 <repo>/data/raw, gitignore).
    - 청크마다 _chunks/ 에 중간 저장하므로 세션이 끊기면 같은 명령을 다시 실행해 이어받는다.
      날이 바뀌었으면 --date 로 원래 버전 날짜를 지정해야 같은 버전으로 이어진다.
    - 완료 시 data/versions.md 에 시즌(파일)마다 한 행 append. 같은 버전 ID 가 이미 있으면 시작 전에 거부.
      Colab 클론에서는 커밋할 수 없으니 출력된 행을 로컬 장부에 붙여 커밋한다.
    - 홀드아웃 전용 시즌(HOLDOUT_SEASONS = {2026}, OPE·분해 전용)은 시즌 종료 여부와 무관하게 --seasons 로 거부한다.
      --holdout 으로만 받는다 (frozen=false). --freeze 는 수집 상한(실행일−2일)이 정규시즌 종료일 이상일 때만.
    - --dry-run 은 {out}/_dryrun/{version}/ 에 쓰고 장부도 그 안의 versions.md 에 쓴다. 로컬은 이것만.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from pitcheezy.data import statcast_fetch as sf
except ModuleNotFoundError as exc:  # pragma: no cover
    sys.exit(f"{exc}. 먼저 레포 루트에서 `pip install -e .` 를 실행할 것")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        epilog=__doc__.split("\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[], metavar="YYYY",
        help=(
            f"학습 시즌. 가능: {sorted(set(sf.SEASON_DATES) - sf.HOLDOUT_SEASONS)} "
            f"(홀드아웃 전용 {sorted(sf.HOLDOUT_SEASONS)} 은 항상 거부)"
        ),
    )
    parser.add_argument(
        "--holdout", type=int, nargs="+", default=[], metavar="YYYY",
        help="검증 시즌. holdout_{YYYY}/ 아래 별도 저장, frozen=false",
    )
    parser.add_argument("--tag", metavar="TAG", help="버전 태그 → 버전 ID d{오늘}-{tag} (예 s2325)")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "raw", metavar="DIR",
        help="출력 루트. Drive 경로는 여기로만 (기본 <repo>/data/raw)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="시즌마다 첫 하루만 받아 파이프라인 검증. {out}/_dryrun/ 에 쓰고 실제 장부는 건드리지 않음",
    )
    parser.add_argument(
        "--date", metavar="YYYYMMDD",
        help="버전 ID 의 날짜 (기본 오늘). 날이 바뀐 뒤 끊긴 수집을 같은 버전으로 이어받을 때만",
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="홀드아웃을 frozen=true 로 기록. 수집 상한(실행일−2일)이 정규시즌 종료일 이상일 때만 허용",
    )
    parser.add_argument(
        "--versions-file", type=Path, default=REPO_ROOT / "data" / "versions.md", metavar="PATH",
        help="데이터 버전 장부 (기본 <repo>/data/versions.md)",
    )
    parser.add_argument(
        "--verify", metavar="VERSION",
        help="수집하지 않고 장부의 sha256·행 수와 {out}/{VERSION}/ 의 파일을 대조",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verify:
        if args.seasons or args.holdout or args.tag or args.dry_run or args.freeze:
            parser.error("--verify 는 --out/--versions-file 외의 옵션과 같이 쓸 수 없음")
        problems = sf.verify(args.verify, args.out, args.versions_file)
        if problems:
            for p in problems:
                print(f"FAIL {p}", file=sys.stderr)
            return 1
        print(f"OK {args.verify}: 장부와 파일이 일치")
        return 0

    if not args.tag:
        parser.error("--tag 필요 (예: --tag s2325)")
    if not args.seasons and not args.holdout:
        parser.error("--seasons 또는 --holdout 필요 (예: --seasons 2023 2024 2025 --holdout 2026)")
    if args.date:
        try:
            today = datetime.strptime(args.date, "%Y%m%d").date()
        except ValueError:
            parser.error(f"--date 형식 오류: {args.date!r} (YYYYMMDD)")
    else:
        today = date.today()

    try:
        version = sf.version_id(args.tag, today)
        run = sf.run_fetch(
            version=version,
            seasons=args.seasons,
            holdouts=args.holdout,
            out_dir=args.out,
            ledger=args.versions_file,
            repo_root=REPO_ROOT,
            today=today,
            dry_run=args.dry_run,
            freeze=args.freeze,
        )
    except sf.FetchError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"{'DRY-RUN ' if args.dry_run else ''}완료 {run.version} → {run.version_dir}")
    for r in run.results:
        print(
            f"  {r.season}{' holdout' if r.holdout else ''}: {r.start}..{r.end}  "
            f"{r.rows:,}행 × {r.columns}컬럼  청크 {r.chunks_fetched}/{r.chunks_total} 수집"
            f"{'  (최종본 재사용)' if r.reused else f'  {r.fetch_seconds:.1f}s'}"
            f"{f'  중복키 {r.duplicate_keys:,}' if r.duplicate_keys else ''}  frozen={str(r.frozen).lower()}"
        )
    print(f"  장부: {run.ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

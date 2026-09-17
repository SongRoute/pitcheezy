"""tests/fixtures 소형 parquet 생성. 버전 있는 데이터셋에서 투수별로 시즌 첫 경기(game_date 최소) 한 판을 통째로 잘라낸다.

    python scripts/make_fixture.py --version d20260911-s2325 --season 2024 --pitchers 657277 666142

출력 tests/fixtures/statcast_{season}_{version}_p{N}.parquet (컬럼 전부, 정렬 키 유지) + README 표 행을 stdout 에 찍는다.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pitcheezy.data.statcast_fetch import SORT_KEYS, sha256_file

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--pitchers", type=int, nargs="+", required=True, help="MLBAM id")
    ap.add_argument("--data-dir", default=os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data"))
    ap.add_argument("--out", default=REPO / "tests" / "fixtures")
    a = ap.parse_args()

    src = Path(a.data_dir) / "raw" / a.version / f"statcast_{a.season}.parquet"
    t = pq.read_table(src)
    parts = []
    rows = []
    for pid in a.pitchers:
        tp = t.filter(pc.equal(t["pitcher"], pid))
        first_date = pc.min(tp["game_date"]).as_py()
        td = tp.filter(pc.equal(tp["game_date"], first_date))
        first_game = pc.min(td["game_pk"]).as_py()
        tg = tp.filter(pc.equal(tp["game_pk"], first_game))
        tg = tg.sort_by([(k, "ascending") for k in SORT_KEYS])
        parts.append(tg)
        name = tg["player_name"][0].as_py()
        date = str(tg["game_date"][0].as_py())[:10]
        rows.append((pid, name, first_game, date, tg.num_rows))
    out = pa.concat_tables(parts).sort_by([(k, "ascending") for k in SORT_KEYS])
    dst = Path(a.out) / f"statcast_{a.season}_{a.version}_p{len(a.pitchers)}.parquet"
    pq.write_table(out, dst, compression="snappy")
    print(f"wrote {dst} rows={out.num_rows} cols={out.num_columns} sha256={sha256_file(dst)}")
    for pid, name, gpk, date, n in rows:
        print(f"| {dst.name} | {a.version} | {name} ({pid}) | game_pk {gpk} ({date}) | {n} |")


if __name__ == "__main__":
    main()

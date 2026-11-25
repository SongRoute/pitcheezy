"""Rebuild the answer labels (actual pitch type of each scored pitch) from Statcast data you download yourself.

The labels are MLB-derived, so this repo does not ship them (see NOTICE). This script reads a Statcast pitch-level
file (CSV or parquet, e.g. saved with pybaseball.statcast(start_dt, end_dt) for the 2026 postseason) and writes

    key,actual          key = game_pk:at_bat_number:pitch_number, actual = Statcast pitch_type

With --check data/eps/rows.csv it also reports how many scored rows get the same 9-group label as the file you have.

    python scripts/labels_from_statcast.py --statcast statcast_2026_post.csv --out data/eps/labels-statcast.csv \
        --check data/eps/rows.csv

Not verified in this repo: the original labels were read from the MLB Stats API final game feeds
(allPlays[].playEvents[] with isPitch, details.type.code; at_bat_number = atBatIndex + 1, pitch_number = running
number of pitch and automatic-call events in the plate appearance). Statcast uses the same key columns, but whether
every pitch_number and pitch_type agrees with the feed for these 31 games has not been measured.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from pitcheezy.interfaces.pitch_types import STATCAST_TO_PITCH_ID  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--statcast', required=True, type=pathlib.Path, nargs='+')
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--check', type=pathlib.Path, help='rows.csv to compare against (key, actual columns)')
    a = ap.parse_args()
    cols = ['game_pk', 'at_bat_number', 'pitch_number', 'pitch_type']
    df = pd.concat([pd.read_parquet(p, columns=cols) if p.suffix == '.parquet' else pd.read_csv(p, usecols=cols)
                    for p in a.statcast], ignore_index=True)
    df = df.dropna(subset=['game_pk', 'at_bat_number', 'pitch_number'])
    df['key'] = (df['game_pk'].astype(int).astype(str) + ':' + df['at_bat_number'].astype(int).astype(str) + ':'
                 + df['pitch_number'].astype(int).astype(str))
    df = df.drop_duplicates('key')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df[['key', 'pitch_type']].rename(columns={'pitch_type': 'actual'}).to_csv(a.out, index=False)
    print(f'wrote {a.out}: {len(df)} pitches from {df["game_pk"].nunique()} games')
    if a.check:
        lab = dict(zip(df['key'], df['pitch_type']))
        with open(a.check, newline='') as f:
            rows = list(csv.DictReader(f))
        found = [r for r in rows if r['key'] in lab]
        same = sum(STATCAST_TO_PITCH_ID.get(str(lab[r['key']])) == STATCAST_TO_PITCH_ID.get(r['actual']) for r in found)
        print(f'check: {len(rows)} scored rows, {len(found)} keys found in Statcast, {same} with the same 9-group label')


if __name__ == '__main__':
    main()

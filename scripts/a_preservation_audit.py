"""Read-only, transaction-consistent preservation checks for the live replay DB.

User cursors/annotations/jobs may legitimately change while the server is live.
Immutable recommendations/event revisions and original session identities must
remain present. No model inference, source data loading or writes to the DB.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def snapshot(database):
    uri = Path(database).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        tables = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        rows = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            data = db.execute('SELECT * FROM ' + quoted)
            names = [column[0] for column in data.description]
            rows[table] = [dict(zip(names, row)) for row in data.fetchall()]
        integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
    immutable = {}
    for table in ('recommendations', 'recommendation_meta', 'event_results'):
        immutable[table] = sorted(digest(row) for row in rows.get(table, []))
    session_keys = ('id', 'game_id', 'pa_id', 'dataset_identity', 'created')
    sessions = {row['id']: digest({k: row[k] for k in session_keys})
                for row in rows.get('sessions', [])}
    return {'checked_at': datetime.now(timezone.utc).isoformat(),
            'database': str(Path(database).resolve()), 'integrity': integrity,
            'counts': {table: len(items) for table, items in rows.items()},
            'immutable_row_hashes': immutable, 'session_identity_hashes': sessions,
            'all_table_content_hashes': {table: digest(sorted(digest(row) for row in items))
                                         for table, items in rows.items()}}


def compare(before, after):
    immutable_missing = {table: sorted(set(hashes) - set(after['immutable_row_hashes'].get(table, [])))
                         for table, hashes in before['immutable_row_hashes'].items()}
    changed_sessions = [key for key, value in before['session_identity_hashes'].items()
                        if after['session_identity_hashes'].get(key) != value]
    return {'passed': after['integrity'] == 'ok' and not any(immutable_missing.values()) and not changed_sessions,
            'missing_immutable_rows': {table: len(rows) for table, rows in immutable_missing.items()},
            'missing_or_changed_session_identities': len(changed_sessions),
            'counts_before': before['counts'], 'counts_after': after['counts'],
            'all_table_content_unchanged': before['all_table_content_hashes'] == after['all_table_content_hashes'],
            'mutable_data_note': 'Cursors, jobs and manual annotations may legitimately change through user actions; before snapshot is preserved, no writes were made by this audit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline')
    args = parser.parse_args()
    current = snapshot(args.database)
    result = {'snapshot': current}
    if args.baseline:
        before = json.loads(Path(args.baseline).read_text())['snapshot']
        if before['database'] != current['database']:
            raise ValueError('Baseline refers to a different database')
        result['comparison'] = compare(before, current)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    summary = {'output': str(output), 'integrity': current['integrity'], 'counts': current['counts']}
    if 'comparison' in result:
        summary['comparison'] = result['comparison']
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if current['integrity'] != 'ok' or result.get('comparison', {}).get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

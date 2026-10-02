"""Extract post-event evidence and observed substitutions, without causal labels."""
from pathlib import Path
import hashlib
import json
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ['game_pk', 'at_bat_number', 'pitch_number']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if pd.isna(value):
        return None
    if hasattr(value, 'item'):
        return value.item()
    return value


def extract(frame, event_names):
    ordered = frame.sort_values(['game_date', *KEY])
    if ordered.duplicated(KEY).any():
        raise ValueError('duplicate pitch keys')
    if not pd.to_datetime(ordered.game_date).dt.year.eq(2025).all():
        raise ValueError('only fixed 2025 development games allowed')
    packets = []
    for _, row in ordered.loc[ordered.events.fillna('').isin(event_names)].iterrows():
        pa = ordered.loc[ordered.game_pk.eq(row.game_pk) & ordered.at_bat_number.eq(row.at_bat_number)]
        earlier = ordered.loc[ordered.game_pk.eq(row.game_pk) & ordered.pitcher.eq(row.pitcher) &
                              ((ordered.at_bat_number < row.at_bat_number) |
                               (ordered.at_bat_number.eq(row.at_bat_number) & (ordered.pitch_number < row.pitch_number)))]
        next_names = ['next_inning', 'next_half', 'next_outs', 'next_bases', 'next_home_score', 'next_away_score']
        next_state = {name.removeprefix('next_'): plain(row[name]) for name in next_names}
        packets.append({
            'pitch_id': ':'.join(str(int(row[k])) for k in KEY),
            'date': str(pd.Timestamp(row.game_date).date()), 'pitcher_id': int(row.pitcher),
            'batter_id': int(row.batter), 'event': row.events,
            'initial_defender': 'home' if row.inning_topbot == 'Top' else 'away',
            'pre_pitch_state': {name: plain(row[name]) for name in
                                ['inning', 'inning_topbot', 'outs_when_up', 'bases', 'home_score', 'away_score', 'balls', 'strikes']},
            'recorded_next_state': next_state,
            'next_state_complete': all(value is not None for value in next_state.values()),
            'pa_pitch_ids': [':'.join(str(int(item[k])) for k in KEY) for _, item in pa.iterrows()],
            'prior_same_game_pitch_count': len(earlier),
            'pa_supported_in_source': bool(pa.supported_pa.fillna(False).all()),
            'intent': None, 'available_replacement_candidates': None,
            'limitations': ['post-event evidence, not a pre-pitch feature packet',
                           'no accepted intent label; no pitcher/catcher/batter causal credit',
                           'observed appearances do not establish bullpen availability'],
        })
    substitutions = []
    for (game, team), rows in ordered.groupby(['game_pk', 'fielding_team'], sort=False):
        previous = None
        for _, row in rows.iterrows():
            current = int(row.pitcher)
            if previous is not None and current != previous:
                substitutions.append({'game_pk': int(game), 'fielding_team': str(team),
                    'first_observed_pitch_id': ':'.join(str(int(row[k])) for k in KEY),
                    'outgoing_pitcher': previous, 'incoming_pitcher': current,
                    'decision_timestamp': None, 'available_alternatives': None,
                    'source': 'consecutive recorded pitcher identities; actual decision time unavailable'})
            previous = current
    return packets, substitutions


def main():
    config_path = ROOT / 'configs/EXP-A-EVENTS-001.json'
    config = json.loads(config_path.read_text())
    manifest_path = ROOT / config['selection_manifest']
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['dataset_paths']['dev'])
    if sha(source) != manifest['sha256']['dev_full_games.parquet']:
        raise ValueError('selected games changed')
    packets, substitutions = extract(pd.read_parquet(source), config['events'])
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    payload = {'schema_version': 1, 'events': packets, 'observed_substitutions': substitutions}
    path = output / 'event_packets.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    summary = {'experiment_id': config['experiment_id'], 'config_sha256': sha(config_path),
        'code_sha256': sha(__file__), 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_manifest_sha256': sha(manifest_path), 'selected_games_sha256': sha(source),
        'output_path': str(path), 'output_sha256': sha(path), 'event_count': len(packets),
        'event_counts': pd.Series([x['event'] for x in packets]).value_counts().to_dict(),
        'observed_substitution_count': len(substitutions),
        'missing_next_state_count': sum(not x['next_state_complete'] for x in packets),
        'available_roster_status': 'unmeasured', 'real_intent_status': 'unavailable',
        'use': 'C event linkage and replacement input boundaries, not model training or causal validation'}
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ROOT / 'results/EXP-A-EVENTS-001.json'
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

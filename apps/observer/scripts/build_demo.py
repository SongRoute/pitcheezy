"""Build a small replay catalogue from hash-verified 2023–2025 records.

No outcome-based case selection. Profiles, repertoire and zone bounds exclude
the entire replay date. This writes only the new Observer run on the SSD.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'apps/observer/backend'), str(REPO/'experiments/pitchmdp')]
import pandas as pd
from observer_app.settings import BUNDLE, CONFIG, RUN, require_storage
from pitchmdp.sequence_data import prepare_frame
from pitchmdp.archetypes import add_batter_style_history, STYLE_COLUMNS, RELIABILITY_COLUMNS


def finite(value):
    return None if pd.isna(value) else float(value)


def state(row):
    return {'inning': int(row.inning), 'half': str(row.inning_topbot),
            'outs': int(row.outs_when_up), 'bases': int(row.bases),
            'home_score': int(row.home_score), 'away_score': int(row.away_score),
            'balls': int(row.balls), 'strikes': int(row.strikes)}


def main():
    require_storage()
    destination = RUN/'dataset.json'
    if destination.exists():
        from observer_app.dataset import DemoDataset
        existing = DemoDataset(destination)
        print(json.dumps({'status': 'verified_existing', 'games': len(existing.games), 'sha256': existing.identity}))
        return
    print('Loading and verifying approved data', flush=True)
    frame = prepare_frame(REPO/'experiments/pitchmdp/configs/local.json')
    identity = frame.attrs['sequence_data_identity']
    frame = add_batter_style_history(frame)
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    frame['observer_prior_pitch_count'] = frame.groupby(['game_pk', 'pitcher'], sort=False).cumcount()
    metadata = json.loads((BUNDLE/'metadata.json').read_text())
    pitchers = metadata['pitchers']
    candidates = frame.loc[dates.ge(CONFIG['demo_start_date']) & frame.pitcher.isin([int(p) for p in pitchers])]
    games, selected_pitchers = [], set()
    for game_id, raw_game in candidates.groupby('game_pk', sort=False):
        first = raw_game.iloc[0]
        pitcher = int(first.pitcher)
        if pitcher in selected_pitchers:
            continue
        date = pd.Timestamp(first.game_date).normalize()
        prior = frame.loc[dates.lt(date)]
        recent = prior.loc[(pd.to_datetime(prior.game_date) >= date-pd.Timedelta(days=CONFIG['repertoire_lookback_days'])) & prior.pitcher.eq(pitcher)]
        counts = recent.pitch_type.value_counts().to_dict()
        allowed = {name: int(counts.get(name, 0)) for name in pitchers[str(pitcher)]['pitch_types']
                   if counts.get(name, 0) >= CONFIG['repertoire_minimum_pitches']}
        if not allowed:
            continue
        league = prior[['sz_bot', 'sz_top']].median()
        pas = []
        for pa_id, rows in raw_game.groupby('at_bat_number', sort=False):
            rows = rows.sort_values('pitch_number')
            start, end = rows.iloc[0], rows.iloc[-1]
            if not rows.supported_pa.all() or not end.is_pa_terminal:
                continue
            if start.pitch_number != 1 or start.balls != 0 or start.strikes != 0 or rows.pitcher.nunique() != 1:
                continue
            if not rows.balls.between(0, 3).all() or not rows.strikes.between(0, 2).all():
                continue
            if pd.isna(end.next_inning):
                continue
            historical = prior.loc[prior.batter.eq(start.batter), ['sz_bot', 'sz_top']].median().fillna(league)
            bounds = {'bottom': float(historical.sz_bot), 'top': float(historical.sz_top)}
            if bounds['top'] <= bounds['bottom']:
                raise ValueError('Invalid historical zone bounds')
            pitches = []
            for _, row in rows.iterrows():
                before = state(row)
                request = {key: value for key, value in before.items() if key != 'half'}
                request.update(topbot=before['half'], pitcher_id=int(row.pitcher), batter_stand=str(row.stand),
                               date=date.date().isoformat(), top_k=3,
                               batter_profile={'rates': [float(row[c]) for c in STYLE_COLUMNS],
                                               'reliabilities': [float(row[c]) for c in RELIABILITY_COLUMNS],
                                               'as_of': (date-pd.Timedelta(days=1)).date().isoformat()})
                pitches.append({'id': f'{int(game_id)}:{int(pa_id)}:{int(row.pitch_number)}',
                    'pitch_number': int(row.pitch_number), 'state': before, 'request': request,
                    'context_notes': [f'이 투구 전까지 던진 공 {int(row.observer_prior_pitch_count)}개',
                                      '타자 성향·구종 구성은 경기 전날까지의 기록으로 계산'],
                    'actual': {'pitch_type': None if pd.isna(row.pitch_type) else str(row.pitch_type),
                        'x': finite(row.plate_x), 'z': finite(row.plate_z), 'speed_mph': finite(row.release_speed),
                        'description': '' if pd.isna(row.description) else str(row.description),
                        'event': None if pd.isna(row.events) else str(row.events)}})
            terminal = {'inning': int(end.next_inning), 'half': str(end.next_half),
                        'outs': int(end.next_outs), 'bases': int(end.next_bases),
                        'home_score': int(end.next_home_score), 'away_score': int(end.next_away_score),
                        'balls': 0, 'strikes': 0}
            pas.append({'id': int(pa_id), 'batter_label': f'타자 #{int(start.batter)}',
                        'pitcher_label': pitchers[str(pitcher)].get('name') or str(start.player_name),
                        'batter_stand': str(start.stand), 'inning': int(start.inning), 'half': str(start.inning_topbot),
                        'zone_bounds': bounds, 'repertoire_counts': allowed, 'pitches': pitches,
                        'terminal_state': terminal})
            if len(pas) >= CONFIG['demo_pas_per_game']:
                break
        if not pas:
            continue
        games.append({'id': int(game_id), 'date': date.date().isoformat(),
                      'home_team': str(first.home_team), 'away_team': str(first.away_team),
                      'title': f'{first.away_team} @ {first.home_team}', 'plate_appearances': pas})
        selected_pitchers.add(pitcher)
        if len(games) >= CONFIG['demo_games']:
            break
    if len(games) < CONFIG['demo_games']:
        raise RuntimeError('Insufficient supported replay games')
    data = {'schema_version': 1, 'games': games}
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode()
    manifest = {'sha256': hashlib.sha256(payload).hexdigest(), 'source_identity': identity,
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'config': CONFIG,
                'selection': 'first chronological supported PAs; distinct available cohort pitchers; no outcome selection',
                'games': len(games), 'plate_appearances': sum(len(g['plate_appearances']) for g in games),
                'pitches': sum(len(pa['pitches']) for g in games for pa in g['plate_appearances']),
                'limitations': ['Retrospective supported-PA filter excludes unsupported transitions.',
                                'Player names unavailable for batters; MLB IDs used only as display/join keys.']}
    destination.write_bytes(payload)
    (RUN/'dataset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ('sha256', 'games', 'plate_appearances', 'pitches')}), flush=True)


if __name__ == '__main__':
    main()

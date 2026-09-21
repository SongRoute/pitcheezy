"""Build v2 replay catalogue from verified 2023–2025 full-pool, past-only profiles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'apps/observer/backend'), str(REPO/'experiments/pitchmdp')]

import pandas as pd

from observer_app.context import ObservedContext, context_notes
from observer_app.player_metadata import load_player_names, player_label
from observer_app.settings import ARTIFACT_ROOT, BUNDLE, CONFIG
from pitchmdp.archetypes import add_batter_style_history, STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.sequence_data import prepare_frame

OUTPUT_ROOT = ARTIFACT_ROOT/'runs/observer-improvement-v2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    import math
    return None if pd.isna(value) or not math.isfinite(float(value)) else float(value)


def state(row):
    return {'inning': int(row.inning), 'half': str(row.inning_topbot), 'outs': int(row.outs_when_up),
            'bases': int(row.bases), 'home_score': int(row.home_score), 'away_score': int(row.away_score),
            'balls': int(row.balls), 'strikes': int(row.strikes)}


def pa_exclusions(rows):
    """Structural support checks only; never rank/select by terminal result or score."""
    start, end = rows.iloc[0], rows.iloc[-1]
    reasons = []
    if not rows.supported_pa.fillna(False).all():
        reasons.append('unsupported_transition')
    if not bool(end.is_pa_terminal):
        reasons.append('missing_terminal_pitch')
    if start.pitch_number != 1 or start.balls != 0 or start.strikes != 0:
        reasons.append('incomplete_pa_start')
    if rows.pitcher.nunique() != 1:
        reasons.append('pitcher_change_within_pa')
    if not rows.balls.between(0, 3).all() or not rows.strikes.between(0, 2).all():
        reasons.append('invalid_count')
    terminal_columns = ['next_inning', 'next_half', 'next_outs', 'next_bases', 'next_home_score', 'next_away_score']
    if end[terminal_columns].isna().any():
        reasons.append('missing_terminal_state')
    return reasons


def build_catalog(frame, pitcher_metadata, names=None, *, start_date='2025-08-16', end_date='2025-09-30',
                  max_games_per_pitcher=3, minimum_repertoire=20, lookback_days=90):
    """Pure builder entry point for full-frame synthetic leakage and coverage tests."""
    if max_games_per_pitcher < 1 or minimum_repertoire < 1 or lookback_days < 1:
        raise ValueError('Positive game/support/history limits required')
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    begin, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if begin > end or dates.isna().any() or not dates.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError('Chronological window and approved 2023–2025 source dates required')
    frame = frame.copy()
    frame['game_date'] = dates
    frame.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], inplace=True)
    if frame.duplicated(['game_pk', 'at_bat_number', 'pitch_number']).any():
        raise ValueError('Duplicate pitch keys')
    # Deliberately before ANY support/cohort/date filter; includes same-date exclusion.
    add_batter_style_history(frame)
    names = names or {}
    cohort = sorted(int(p) for p in pitcher_metadata)
    cohort_frame = frame.loc[frame.pitcher.isin(cohort)]
    contexts = ObservedContext(cohort_frame, lookback_days=lookback_days)
    in_window = frame.game_date.between(begin, end)
    candidate_games = set(frame.loc[in_window & frame.pitcher.isin(cohort), 'game_pk'].astype(int))
    games = {int(gid): game for gid, game in frame.loc[frame.game_pk.isin(candidate_games)].groupby('game_pk', sort=False)}
    # Cache strictly prior-date aggregates; never use current-game speeds for profiles/repertoire.
    prior_cache, pitcher_cache = {}, {}
    def priors(date, pitcher):
        if date not in prior_cache:
            previous = frame.loc[frame.game_date.lt(date)]
            prior_cache[date] = (previous.groupby('batter')[['sz_bot','sz_top']].median(),
                                 previous[['sz_bot','sz_top']].median())
        cache_key = (date, pitcher)
        if cache_key not in pitcher_cache:
            recent = cohort_frame.loc[cohort_frame.pitcher.eq(pitcher) & cohort_frame.game_date.lt(date) &
                                      cohort_frame.game_date.ge(date-pd.Timedelta(days=lookback_days))]
            counts = recent.pitch_type.value_counts()
            pitcher_cache[cache_key] = {kind: int(counts.get(kind, 0))
                for kind in pitcher_metadata[str(pitcher)]['pitch_types'] if counts.get(kind, 0) >= minimum_repertoire}
        return (*prior_cache[date], pitcher_cache[cache_key])

    selected = defaultdict(list)
    coverage_pitchers, excluded = [], []
    for pitcher in cohort:
        selected_game_count = 0
        stats = {'pitcher_id': pitcher, 'pitcher_label': player_label(pitcher, '투수', names),
                 'window_games': 0, 'window_pa_total': 0, 'window_supported_pa_total': 0,
                 'selected_games': 0, 'selected_pa_total': 0, 'selected_supported_pa_total': 0}
        for game_id, game in games.items():
            if not game.pitcher.eq(pitcher).any():
                continue
            stats['window_games'] += 1
            date = pd.Timestamp(game.iloc[0].game_date)
            batter_bounds, league_bounds, repertoire = priors(date, pitcher)
            candidates = []
            for pa_id, rows in game.groupby('at_bat_number', sort=False):
                if not rows.pitcher.eq(pitcher).any():
                    continue
                reasons = pa_exclusions(rows)
                if not repertoire:
                    reasons.append('no_supported_prior_repertoire')
                start = rows.iloc[0]
                bounds = (batter_bounds.loc[start.batter] if start.batter in batter_bounds.index else league_bounds).fillna(league_bounds)
                if bounds.isna().any() or not bounds.sz_top > bounds.sz_bot:
                    reasons.append('missing_prior_zone_bounds')
                candidates.append((int(pa_id), rows, reasons, bounds))
            stats['window_pa_total'] += len(candidates)
            supported = [case for case in candidates if not case[2]]
            stats['window_supported_pa_total'] += len(supported)
            if not supported or selected_game_count >= max_games_per_pitcher:
                continue
            selected_game_count += 1
            stats['selected_games'] += 1
            stats['selected_pa_total'] += len(candidates)
            stats['selected_supported_pa_total'] += len(supported)
            for pa_id, _, reasons, _ in candidates:
                if reasons:
                    excluded.append({'game_id': game_id, 'pa_id': pa_id, 'pitcher_id': pitcher, 'reasons': reasons})
            for pa_id, rows, _, bounds in supported:
                selected[game_id].append((pitcher, pa_id, rows, {'bottom': float(bounds.sz_bot), 'top': float(bounds.sz_top)}, repertoire))
        coverage_pitchers.append(stats)

    public_games = []
    for game_id, cases in selected.items():
        game = games[game_id]
        first = game.iloc[0]
        date = pd.Timestamp(first.game_date)
        pas = []
        for pitcher, pa_id, rows, bounds, repertoire in sorted(cases, key=lambda case: case[1]):
            start, end_row = rows.iloc[0], rows.iloc[-1]
            pitches = []
            for _, row in rows.iterrows():
                before = state(row)
                request = {key: value for key, value in before.items() if key != 'half'}
                request.update(topbot=before['half'], pitcher_id=int(row.pitcher), batter_stand=str(row.stand),
                    date=date.date().isoformat(), top_k=3,
                    batter_profile={'rates': [float(row[c]) for c in STYLE_COLUMNS],
                                    'reliabilities': [float(row[c]) for c in RELIABILITY_COLUMNS],
                                    'as_of': (date-pd.Timedelta(days=1)).date().isoformat()})
                observation = contexts.before(row, repertoire)
                pitches.append({'id': f'{game_id}:{pa_id}:{int(row.pitch_number)}', 'pitch_number': int(row.pitch_number),
                    'state': before, 'request': request, 'context': observation, 'context_notes': context_notes(observation),
                    'actual': {'pitch_type': None if pd.isna(row.pitch_type) else str(row.pitch_type),
                        'x': finite(row.plate_x), 'z': finite(row.plate_z), 'speed_mph': finite(row.release_speed),
                        'description': '' if pd.isna(row.description) else str(row.description),
                        'event': None if pd.isna(row.events) else str(row.events)}})
            terminal = {'inning': int(end_row.next_inning), 'half': str(end_row.next_half),
                'outs': int(end_row.next_outs), 'bases': int(end_row.next_bases),
                'home_score': int(end_row.next_home_score), 'away_score': int(end_row.next_away_score), 'balls': 0, 'strikes': 0}
            pas.append({'id': pa_id, 'batter_id': int(start.batter), 'pitcher_id': pitcher,
                'batter_label': player_label(start.batter, '타자', names), 'pitcher_label': player_label(pitcher, '투수', names),
                'batter_stand': str(start.stand), 'inning': int(start.inning), 'half': str(start.inning_topbot),
                'zone_bounds': bounds, 'repertoire_counts': repertoire, 'pitches': pitches, 'terminal_state': terminal})
        unique = {pa['id'] for pa in pas}
        if len(unique) != len(pas):
            raise ValueError('One supported PA cannot belong to multiple pitcher selections')
        public_games.append({'id': game_id, 'date': date.date().isoformat(), 'home_team': str(first.home_team),
            'away_team': str(first.away_team), 'title': f'{first.away_team} @ {first.home_team}', 'plate_appearances': pas})
    public_games.sort(key=lambda game: (game['date'], game['id']))
    included_count = sum(len(g['plate_appearances']) for g in public_games)
    # Mixed-pitcher excluded PAs can be attached to two cohort pitchers: headline counts are unique PAs.
    excluded_unique = {(item['game_id'], item['pa_id']) for item in excluded}
    coverage = {'scope': 'selected pitcher-game pairs; only structurally supported PAs enter replay',
        'per_pitcher': coverage_pitchers, 'included_pa_total': included_count,
        'excluded_pa_total': len(excluded_unique), 'selected_pa_total': included_count+len(excluded_unique),
        'support_rate': included_count/(included_count+len(excluded_unique)) if included_count+len(excluded_unique) else None,
        'exclusion_reason_counts': dict(Counter(reason for item in excluded for reason in item['reasons'])),
        'excluded_pas': excluded,
        'gap_notice': '지원하지 않는 전이·불완전한 타석은 재생 목록에서 제외됩니다. 목록은 경기 전체와 다를 수 있습니다.'}
    return {'schema_version': 1, 'games': public_games, 'coverage': coverage}, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--start-date', default='2025-08-16')
    parser.add_argument('--end-date', default='2025-09-30')
    parser.add_argument('--max-games-per-pitcher', type=int, default=3)
    parser.add_argument('--player-metadata', type=Path)
    parser.add_argument('--participants-only', action='store_true', help='Save selected IDs/PA keys for local name lookup; do not write a dataset')
    args = parser.parse_args()
    output = args.output.resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise SystemExit('Use mounted SSD observer-improvement-v2 output only; v1/frozen artifacts are immutable')
    if (output/'dataset.json').exists() or (output/'dataset_manifest.json').exists():
        raise SystemExit('Existing catalogue is immutable; choose a new subdirectory')
    metadata_path = BUNDLE/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if len(metadata['pitchers']) != 6:
        raise ValueError('Expected the frozen six-pitcher cohort')
    names = load_player_names(args.player_metadata)
    sources = [Path(__file__), REPO/'apps/observer/backend/observer_app/context.py',
               REPO/'apps/observer/backend/observer_app/player_metadata.py',
               REPO/'apps/observer/backend/observer_app/domain.py',
               REPO/'experiments/pitchmdp/pitchmdp/sequence_data.py',
               REPO/'experiments/pitchmdp/pitchmdp/archetypes.py']
    references = [metadata_path, REPO/'apps/observer/config.json', REPO/'experiments/pitchmdp/configs/local.json']
    if args.player_metadata:
        references.append(args.player_metadata.resolve())
    source_hashes = {str(path.relative_to(REPO)): sha256(path) for path in sources}
    reference_hashes = {str(path): sha256(path) for path in references}
    frame = prepare_frame(REPO/'experiments/pitchmdp/configs/local.json')
    identity = frame.attrs['sequence_data_identity']
    data, coverage = build_catalog(frame, metadata['pitchers'], names, start_date=args.start_date, end_date=args.end_date,
        max_games_per_pitcher=args.max_games_per_pitcher, minimum_repertoire=CONFIG['repertoire_minimum_pitches'],
        lookback_days=CONFIG['repertoire_lookback_days'])
    if not data['games']:
        raise RuntimeError('No supported replay games in requested window')
    selected_pas = [[game['id'], pa['id'], pa['pitcher_id'], pa['batter_id']]
                    for game in data['games'] for pa in game['plate_appearances']]
    participants = {'player_ids': sorted({player for row in selected_pas for player in row[2:]}),
                    'selected_pas': selected_pas, 'start_date': args.start_date, 'end_date': args.end_date,
                    'max_games_per_pitcher': args.max_games_per_pitcher}
    preview_path = output/'participants.json'
    if preview_path.exists() and json.loads(preview_path.read_text()) != participants:
        raise RuntimeError('Selected participants differ from prepared lookup; choose a new output directory')
    if args.participants_only:
        output.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(json.dumps(participants, ensure_ascii=False, indent=2))
        print(json.dumps({'participants_path': str(preview_path), 'players': len(participants['player_ids']),
                          'games': len(data['games']), 'plate_appearances': len(selected_pas)}), flush=True)
        return
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode()
    manifest = {'sha256': hashlib.sha256(payload).hexdigest(), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_identity': identity, 'source_hashes': source_hashes, 'reference_hashes': reference_hashes,
        'config': dict(CONFIG, start_date=args.start_date, end_date=args.end_date,
                       max_games_per_pitcher=args.max_games_per_pitcher, cohort_pitcher_ids=sorted(map(int, metadata['pitchers']))),
        'selection': 'First chronological supported games per frozen cohort pitcher; all supported PAs; no outcome ranking',
        'games': len(data['games']), 'plate_appearances': sum(len(g['plate_appearances']) for g in data['games']),
        'pitches': sum(len(pa['pitches']) for g in data['games'] for pa in g['plate_appearances']), 'coverage': coverage,
        'player_names': {'source': str(args.player_metadata) if args.player_metadata else None, 'loaded': len(names), 'fallback': 'MLB ID'},
        'limitations': ['Retrospective support filtering is disclosed; replay omits unsupported PAs.',
            'Velocity summaries describe observations only, not measured fatigue, injury, or causal condition.',
            'Current-game observations are display context only; model profiles/repertoire exclude the whole replay date.']}
    if source_hashes != {str(path.relative_to(REPO)): sha256(path) for path in sources} or reference_hashes != {str(path): sha256(path) for path in references}:
        raise RuntimeError('Source or metadata changed while building catalogue')
    output.mkdir(parents=True, exist_ok=True)
    (output/'dataset.json').write_bytes(payload)
    (output/'dataset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({key: manifest[key] for key in ['sha256','games','plate_appearances','pitches']}), flush=True)


if __name__ == '__main__':
    main()

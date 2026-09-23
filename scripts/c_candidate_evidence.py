"""Audit fixed C replacement candidates from immutable, retrospective sources.

This is an evidence inventory, not a model-value run. It never promotes a
historical roster listing or a later observed substitution to availability at
the manager's pre-substitution decision.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001')
BUNDLE = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1')
PINNED_SHA256 = {
    'roster': 'a327f1fecd7c9b97e4e2575328497520e373f9bcc325b28b341ae17303360371',
    'lineup': '70ec1a13869b7a73f6a2dc6eb97fc5b813c2a03e0300f3b67205018ad6cd4605',
    'anchor': 'e2a64575286160c6a30591a2a59b213b467393323a8884b0a802fb184611ac3c',
    'bundle': '43ece920cb60c6c24ea9f1e720a7000b3b83038a3e24ac27d77ef17a2e8f0f1f',
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(packet_path: Path, lineup_path: Path, anchor_path: Path,
          metadata_path: Path, bundle_manifest_path: Path) -> dict:
    packet_sha, lineup_sha, anchor_sha = (digest(p) for p in (packet_path, lineup_path, anchor_path))
    if (packet_sha, lineup_sha, anchor_sha, digest(bundle_manifest_path)) != tuple(PINNED_SHA256.values()):
        raise ValueError('pinned C evidence source hash mismatch')
    packet = json.loads(packet_path.read_text())
    lineup = json.loads(lineup_path.read_text())
    anchor = json.loads(anchor_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if packet_sha != lineup['original_packet_sha256'] or packet_sha != anchor['original_packet_sha256']:
        raise ValueError('roster source hash mismatch')
    if packet['actual_manager_availability'] is not None or anchor['eligible_replacements'] is not None:
        raise ValueError('unexpected manager availability assertion')
    if len(packet['decisions']) != 6 or {d['game_pk'] for d in packet['decisions']} != {777063, 777094, 777126, 777143, 777217, 777227}:
        raise ValueError('fixed six-game set changed')
    lineups = {d['game_pk']: d for d in lineup['lineups']}
    supported = set(metadata['pitchers']) & set(metadata['repertoire_counts'])
    cases = []
    for decision in packet['decisions']:
        game = decision['game_pk']
        if decision['actual_eligible_replacement_candidates'] is not None:
            raise ValueError('unexpected actual eligible set')
        if decision['prior_window_boxscore_failures'] or not decision['replacement_screen']:
            raise ValueError('incomplete source screen')
        predecision = lineups[game]['lineup_as_of'] is not None
        anchor_ready = game == anchor['game_pk'] and anchor['lineup_as_of'] is not None and anchor['cutoff_index_verified'] and anchor['cutoff_timestamp_verified']
        if predecision and any(slot['stand_for_substitute'] is not None for slot in lineups[game]['lineup_as_of']['ordered_slots']):
            raise ValueError('unexpected verified substitute matchup stance')
        keep_id = decision['keep_pitcher_id']
        screened = []
        for row in decision['replacement_screen']:
            pid = row['pitcher_id']
            if pid == keep_id or pid in decision['same_game_prior_pitcher_ids']:
                raise ValueError('screen contains current or previously used pitcher')
            if row['manager_available_at_decision'] is not None or row['roster_listing'] != 'active_on_queried_historical_date_retrospective':
                raise ValueError('candidate availability promoted from roster')
            apps = row['prior_7d_appearances']
            if apps is None or row['prior_7d_pitches'] != sum(a['pitches'] for a in apps):
                raise ValueError('incomplete candidate workload')
            if any(a['official_date'] >= decision['game_date'] for a in apps):
                raise ValueError('future workload crossed decision date')
            days = sorted({a['official_date'] for a in apps})
            consecutive_prior_days = 0
            cursor = date.fromisoformat(decision['game_date']) - timedelta(days=1)
            while cursor.isoformat() in days:
                consecutive_prior_days += 1
                cursor -= timedelta(days=1)
            if apps:
                latest = max(a['official_date'] for a in apps)
                rest = (date.fromisoformat(decision['game_date']) - date.fromisoformat(latest)).days - 1
                if row['full_rest_calendar_days_since_last_appearance'] != rest:
                    raise ValueError('rest-day mismatch')
            elif row['full_rest_calendar_days_since_last_appearance'] is not None:
                raise ValueError('unknown longer rest was imputed')
            screened.append({
                'pitcher_id': pid, 'name': row['name'],
                'availability_class': 'conditional_roster_screen',
                'manager_available_at_decision': None,
                'observed_deployment_after_decision': pid == decision['observed_incoming_pitcher_id_audit_only'],
                'model_supported': str(pid) in supported,
                'prior_7d_appearances': apps, 'prior_7d_pitches': row['prior_7d_pitches'],
                'full_rest_calendar_days_since_last_appearance': row['full_rest_calendar_days_since_last_appearance'],
                'consecutive_days_through_previous_date': consecutive_prior_days,
                'same_game_prior_appearance': False,
                'substitute_stance_verified': False,
                'inning_evaluation_ready': False,
                'unready_reasons': ['decision_time_availability_unknown',
                                    'stances_against_substitute_unknown'] +
                                   ([] if str(pid) in supported else ['frozen_pitcher_unsupported']),
            })
        if sum(r['observed_deployment_after_decision'] for r in screened) != 1:
            raise ValueError('observed incoming pitcher not in retrospective screen')
        keep = {'pitcher_id': keep_id, 'availability_class': 'evidenced_current_pitcher',
                'on_mound_before_change_evidenced': True, 'continued_use_feasible': None,
                'model_supported': str(keep_id) in supported,
                'same_game_prior_pitches': decision['same_game_prior_pitch_counts'][str(keep_id)],
                'prior_7d_appearances': decision['keep_prior_7d_appearances']}
        cases.append({
            'game_pk': game, 'official_game_date': decision['game_date'],
            'decision_first_observed_pitch_id': decision['decision_first_observed_pitch_id'],
            'decision_time_kind': 'logged_pitcher_change_action_pre_event' if anchor_ready else 'first_observed_pitch_proxy',
            'decision_time_utc': anchor['anchor_action_start_time_utc'] if anchor_ready else None,
            'roster_source_url': decision['roster_source_url'],
            'lineup_source_url': lineups[game]['source_url'],
            'source_time_provenance': {
                'roster': 'historical_date_query_retrieved_after_game_not_publication_at_decision',
                'workload': 'prior_official_dates_only_prior_game_completion_times_not_archived',
                'lineup': 'retrospective_play_by_play_predecision_actions_only',
            },
            'fielding_team': decision['fielding_team'], 'initial_defender': decision['fielding_side'],
            'keep': keep, 'lineup_order_verified_before_proxy': predecision,
            'pre_change_anchor_lineup_verified': bool(anchor_ready),
            'stance_vs_keep_verified': bool(anchor_ready),
            'stance_vs_substitutes_verified': False,
            'initial_state': anchor['state_as_of_anchor'] if anchor_ready else decision['state_at_first_observed_pitch'],
            'candidates': screened, 'unknown_availability_outside_screen': True,
            'actual_eligible_replacements': None,
            'model_comparison': {'status': 'unavailable', 'horizon': 'inning_end',
                                 'initial_defender': decision['fielding_side'], 'value_pp': None,
                                 'reason': 'no_jointly_supported_keep_and_substitute_with_verified_decision_availability_and_matchup_lineup'},
        })
    counts = {
        'games': len(cases), 'screened_replacement_entries': sum(len(c['candidates']) for c in cases),
        'evidenced_current_pitchers': len(cases), 'verified_decision_available_replacements': 0,
        'conditional_roster_screen_entries': sum(len(c['candidates']) for c in cases),
        'supported_keep_games': sum(c['keep']['model_supported'] for c in cases),
        'supported_replacement_entries': sum(r['model_supported'] for c in cases for r in c['candidates']),
        'jointly_supported_keep_and_replacement_games': sum(c['keep']['model_supported'] and any(r['model_supported'] for r in c['candidates']) for c in cases),
        'valid_inning_comparisons': 0,
    }
    if counts['jointly_supported_keep_and_replacement_games'] != 0:
        raise ValueError('joint support changed; revisit fixed comparison plan')
    return {
        'schema_version': 1, 'audit_id': 'C-EVIDENCE-003',
        'scope': 'six fixed 2025 DEV observed pitching changes; retrospective descriptive evidence',
        'source_sha256': {'roster_packet': packet_sha, 'lineup_supplement_v2': lineup_sha,
                          'decision_anchor_777063': anchor_sha, 'model_metadata': digest(metadata_path),
                          'bundle_manifest': digest(bundle_manifest_path)},
        'comparison_spec': {'horizon': 'inning_end', 'unit': 'initial_defender_win_probability_pp',
                            'required_common_inputs': ['same_game', 'same_pre_substitution_state', 'same_initial_defender',
                                                       'same_announced_lineup_and_matchup_stances', 'same_frozen_model',
                                                       'same_fixed_policy', 'same_inning_horizon', 'same_evaluator_caps'],
                            'pa_values_may_be_combined': False, 'actual_replacement_value_pp': None},
        'counts': counts, 'cases': cases,
        'interpretation': 'Observed deployment is known only retrospectively. Roster listings and prior workload make conditional screens, not a verified decision-time eligible set. Model support is a separate property; no fixed game supports both keep and a substitute.',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(SSD / 'replacement_packets.json', SSD / 'lineup_supplement_v2.json',
                   SSD / 'decision_anchor_777063.json', BUNDLE / 'metadata.json',
                   BUNDLE / 'bundle_manifest.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

"""Checks the real frozen decision boundary and the future-date guard."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c_replacement_data', ROOT / 'scripts/c_replacement_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_selection_uses_predecision_rows_and_six_fixed_games():
    cfg = json.loads(module.CONFIG.read_text())
    frame, events, games, *_ = module.load_inputs(cfg)
    selected = module.select_decisions(frame, events, games)
    frozen = json.loads((ROOT / 'results/EXP-C-ROSTER-001/selection.json').read_text())
    assert selected == frozen['selection']
    assert len(selected) == 6
    for decision in selected:
        assert decision['status'] == 'selected'
        assert decision['state_at_first_observed_pitch']['outs'] == 0
        assert decision['state_at_first_observed_pitch']['inning'] <= 8
        assert decision['lineup_as_of'] is None
        assert decision['decision_timestamp'] is None
        assert decision['keep_pitcher_id'] in decision['same_game_prior_pitcher_ids']
        assert decision['observed_incoming_pitcher_id_audit_only'] not in decision['same_game_prior_pitcher_ids']
        game, at_bat, _ = map(int, decision['decision_first_observed_pitch_id'].split(':'))
        state = decision['state_at_first_observed_pitch']
        same_half = frame.loc[frame.game_pk.eq(game) & frame.inning.eq(state['inning']) &
                              frame.inning_topbot.eq(state['half'])]
        assert at_bat == int(same_half.at_bat_number.min())


def test_prior_schedule_rejects_selected_date_or_later():
    class FakeFetcher:
        def get(self, _):
            return {'dates': [{'games': [{'officialDate': '2025-07-05', 'season': '2025',
                                          'gamePk': 1, 'teams': {'home': {'team': {'id': 133}},
                                                                  'away': {'team': {'id': 137}}}}]}]}

    with pytest.raises(ValueError, match='future/prior-season'):
        module.prior_games(FakeFetcher(), 133, module.date(2025, 7, 5), 7)


def test_prior_boxscore_workload_only_uses_team_pitchers():
    class FakeFetcher:
        def get(self, _):
            return {'teams': {
                'home': {'team': {'id': 133}, 'pitchers': [10],
                         'players': {'ID10': {'stats': {'pitching': {'numberOfPitches': 17, 'outs': 3}}}}},
                'away': {'team': {'id': 137}, 'pitchers': [20],
                         'players': {'ID20': {'stats': {'pitching': {'numberOfPitches': 8, 'outs': 3}}}}},
            }}

    by_pitcher, failures = module.workload(FakeFetcher(), 133, [('2025-07-04', 1)])
    assert failures == []
    assert list(by_pitcher) == [10]
    assert by_pitcher[10][0]['pitches'] == 17


def test_prior_schedule_rejects_nonfinal_historical_game():
    class FakeFetcher:
        def get(self, _):
            return {'dates': [{'games': [{'officialDate': '2025-07-04', 'season': '2025',
                                          'gamePk': 1, 'status': {'abstractGameState': 'Preview'},
                                          'teams': {'home': {'team': {'id': 133}},
                                                    'away': {'team': {'id': 137}}}}]}]}

    with pytest.raises(ValueError, match='not reported Final'):
        module.prior_games(FakeFetcher(), 133, module.date(2025, 7, 5), 7)


def test_real_packet_preserves_unknown_eligibility_and_source_integrity():
    cfg = json.loads(module.CONFIG.read_text())
    folder = Path(cfg['output_dir'])
    report = json.loads((ROOT / 'results/EXP-C-ROSTER-001/manifest.json').read_text())
    path = folder / 'replacement_packets.json'
    assert module.sha(path) == report['packet_sha256']
    packet = json.loads(path.read_text())
    assert len(packet['decisions']) == 6
    for raw in packet['raw_response_records']:
        assert not raw.get('error')
        assert module.sha(Path(raw['path'])) == raw['sha256']
    for d in packet['decisions']:
        assert d['actual_eligible_replacement_candidates'] is None
        assert d['lineup_as_of'] is None
        assert d['keep_pitcher_id'] not in {x['pitcher_id'] for x in d['replacement_screen']}
        assert not set(d['same_game_prior_pitcher_ids']) & {x['pitcher_id'] for x in d['replacement_screen']}
        assert all(module.date.fromisoformat(x['official_date']) < module.date.fromisoformat(d['game_date'])
                   for x in d['replacement_screen'] for x in (x['prior_7d_appearances'] or []))


def test_lineup_supplement_uses_frozen_decisions_and_abstains_on_ambiguous_orders():
    cfg = json.loads(module.CONFIG.read_text())
    folder = Path(cfg['output_dir'])
    manifest = json.loads((ROOT / 'results/EXP-C-ROSTER-001/lineup_supplement_manifest.json').read_text())
    path = folder / 'lineup_supplement.json'
    assert module.sha(path) == manifest['output_sha256']
    supplement = json.loads(path.read_text())
    assert supplement['original_packet_sha256'] == module.sha(folder / 'replacement_packets.json')
    assert len(supplement['lineups']) == 6
    assert {x['game_pk'] for x in supplement['lineups'] if x['lineup_as_of']} == {777094, 777143}
    for entry in supplement['lineups']:
        if entry['lineup_as_of']:
            lineup = entry['lineup_as_of']
            assert len(lineup['ordered_slots']) == 9
            assert lineup['ordered_slots'][lineup['next_slot_one_based'] - 1]['batter_id'] == lineup['current_batter_id']
        else:
            assert entry['missing_reason']


def test_lineup_v2_handles_non_pa_and_current_prepitch_substitution():
    folder = Path(json.loads(module.CONFIG.read_text())['output_dir'])
    manifest = json.loads((ROOT / 'results/EXP-C-ROSTER-001/lineup_supplement_v2_manifest.json').read_text())
    path = folder / 'lineup_supplement_v2.json'
    assert module.sha(path) == manifest['output_sha256']
    packet = json.loads(path.read_text())
    by_game = {x['game_pk']: x for x in packet['lineups']}
    assert {g for g, x in by_game.items() if x['lineup_as_of']} == {777094, 777126, 777143, 777217, 777227}
    assert by_game[777063]['missing_reason'] == 'current_pa_pre_first_pitch_nonpitching_substitution'
    assert by_game[777063]['current_pa_pre_first_pitch_action_types'] == ['offensive_substitution']
    assert by_game[777217]['excluded_non_pa_plays'] == [{'at_bat_index': 28, 'event_type': 'caught_stealing_2b'}]
    assert by_game[777126]['applied_predecision_substitutions'][0]['slot'] == 4
    assert any(len(s['observed_stands']) == 2 for s in by_game[777227]['lineup_as_of']['ordered_slots'])
    assert all(s['stand_for_substitute'] is None for x in by_game.values() if x['lineup_as_of']
               for s in x['lineup_as_of']['ordered_slots'])

    def cached_feed(game):
        url = f'https://statsapi.mlb.com/api/v1/game/{game}/playByPlay'
        key = hashlib.sha256(url.encode()).hexdigest()[:20]
        return json.loads((folder / 'raw_lineup' / f'{key}.json').read_text())['allPlays']

    decision = next(x for x in json.loads((folder / 'replacement_packets.json').read_text())['decisions']
                    if x['game_pk'] == 777094)
    plays = cached_feed(777094)
    assert module.reconstruct_predecision_lineup_v2(plays, decision)['lineup_as_of'] is not None
    changed = copy.deepcopy(plays)
    current = changed[int(decision['decision_first_observed_pitch_id'].split(':')[1]) - 1]
    current['playEvents'].insert(0, {'isSubstitution': True,
                                      'details': {'eventType': 'offensive_substitution'}})
    assert module.reconstruct_predecision_lineup_v2(changed, decision)['missing_reason'] == \
           'current_pa_pre_first_pitch_nonpitching_substitution'


def test_777063_anchor_precedes_pitching_change_and_excludes_later_pinch_hitter():
    folder = Path(json.loads(module.CONFIG.read_text())['output_dir'])
    report = json.loads((ROOT / 'results/EXP-C-ROSTER-001/decision_anchor_777063_manifest.json').read_text())
    path = folder / 'decision_anchor_777063.json'
    assert module.sha(path) == report['output_sha256']
    anchor = json.loads(path.read_text())
    assert anchor['lineup_missing_reasons'] == []
    assert anchor['anchor_action_index'] == 0
    assert anchor['keep_pitcher_id'] == 554430
    assert anchor['state_as_of_anchor'] == {'inning': 7, 'half': 'Top', 'outs': 0, 'bases': 0,
                                           'home_score': 2, 'away_score': 2, 'balls': 0, 'strikes': 0}
    lineup = anchor['lineup_as_of']
    assert lineup['next_slot_one_based'] == 7
    assert lineup['next_batter_id'] == 691785  # Marcelo Mayer before the pitcher decision
    assert lineup['next_batter_stand_vs_keep'] == 'L'
    assert len(lineup['ordered_slots']) == 9
    assert all(slot['stand_evidence'] and slot['stand_vs_keep'] in ('L', 'R')
               for slot in lineup['ordered_slots'])
    assert all(e['at_bat_index'] < 48 and e['pitcher_id'] == 554430
               for slot in lineup['ordered_slots'] for e in slot['stand_evidence'])
    assert anchor['source_play_index_max'] == 47
    assert anchor['source_play_end_time_max'] < anchor['anchor_action_start_time_utc']
    assert anchor['excluded_non_pa_plays'] == [{'at_bat_index': 25, 'event_type': 'caught_stealing_2b'}]
    assert anchor['excluded_after_anchor_audit_events'] == [{
        'event_index': 1, 'event_type': 'offensive_substitution',
        'start_time': '2025-07-22T00:22:07.083Z', 'player_id': 663853,
        'replaced_player_id': 691785, 'excluded_from_anchor_features': True,
    }]
    assert anchor['eligible_replacements'] is None and anchor['policy_value'] is None


def test_anchor_rejects_prior_play_with_future_timestamp():
    folder = Path(json.loads(module.CONFIG.read_text())['output_dir'])
    url = 'https://statsapi.mlb.com/api/v1/game/777063/playByPlay'
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    plays = json.loads((folder / 'raw_lineup' / f'{key}.json').read_text())['allPlays']
    decision = next(x for x in json.loads((folder / 'replacement_packets.json').read_text())['decisions']
                    if x['game_pk'] == 777063)
    shifted = copy.deepcopy(plays)
    shifted[47]['about']['endTime'] = '2025-07-22T00:21:00.000Z'
    with pytest.raises(ValueError, match='future cutoff'):
        module.decision_anchor_777063(shifted, decision)

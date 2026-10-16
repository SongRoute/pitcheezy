"""Generated prefix dependency check; no archived capture or model read."""
import pytest

from scripts.fullseason_native_identity_guard_audit import (
    _fresh_supervisor_output, relevant_unknown_attribution, remember_native,
)


def test_relevant_unknowns_respect_32_known_floor_and_same_pa_predecessor():
    latest = {}
    for pitch in range(1, 35):
        latest[(99, 1, pitch)] = {'status': 'native_candidate',
            'native_token': {'pitcher': 10}, 'guard': 'identity_hypothesis_present'}
    latest[(99, 1, 1)] = {'status': 'native_unknown', 'native_token': None,
                          'guard': 'reverse_action_chain_broken'}
    latest[(99, 1, 34)] = {'status': 'native_unknown', 'native_token': None,
                           'guard': 'end_matchup_anchor_invalid'}
    candidate = {'query': {'game_pk': 99, 'at_bat_number': 1,
                           'pitch_number': 35, 'pitcher': 10}}
    decision = {'context_counts': {'prior_known_same_pitcher_total': 32,
                                   'selected_known_tokens': 32,
                                   'prior_unknown_relevant': 1}}
    causes, keys = relevant_unknown_attribution(latest, candidate, decision)
    assert causes == {'end_matchup_anchor_invalid': 1}
    assert keys == {(99, 1, 34)}
    decision['context_counts']['prior_unknown_relevant'] = 2
    with pytest.raises(ValueError, match='divergence'):
        relevant_unknown_attribution(latest, candidate, decision)


def test_semantic_dedup_preserves_first_proof_then_correction_replaces_it():
    key = (99, 1, 1)
    latest = {}
    unknown = {'status': 'native_unknown', 'native_token': None,
               'reason': 'pitch_time_matchup_unverified'}
    remember_native(latest, key, unknown, 'reverse_action_chain_broken', 'new')
    remember_native(latest, key, unknown, 'end_matchup_anchor_invalid', 'unchanged_observation')
    assert latest[key]['guard'] == 'reverse_action_chain_broken'
    corrected = {'status': 'native_candidate', 'native_token': {'pitcher': 10},
                 'reason': 'source_native_six_fields_unpaired'}
    remember_native(latest, key, corrected, 'identity_hypothesis_present', 'correction')
    assert latest[key]['status'] == 'native_candidate'
    assert latest[key]['guard'] == 'source_native_six_fields_unpaired'


def test_supervisor_owned_worker_log_is_only_allowed_initial_output(tmp_path):
    with pytest.raises(ValueError, match='fresh supervised output'):
        _fresh_supervisor_output(tmp_path)
    (tmp_path / 'worker.log').write_text('')
    _fresh_supervisor_output(tmp_path)
    (tmp_path / 'old_receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh supervised output'):
        _fresh_supervisor_output(tmp_path)
    (tmp_path / 'old_receipt.json').unlink()
    (tmp_path / 'worker.log').unlink()
    (tmp_path / 'worker.log').symlink_to('missing')
    with pytest.raises(ValueError, match='regular supervisor worker log'):
        _fresh_supervisor_output(tmp_path)
    (tmp_path / 'worker.log').unlink()
    (tmp_path / 'worker.log').mkdir()
    with pytest.raises(ValueError, match='regular supervisor worker log'):
        _fresh_supervisor_output(tmp_path)

import numpy as np

from scripts.b_choice_candidates import choose_best, choose_minimax, choose_near_value, compare_pitch


def test_near_value_repertoire_respects_gap_and_tie_order():
    actions = [{'pitch_type': 'ST', 'zone_id': 'low'},
               {'pitch_type': 'SI', 'zone_id': 'low'},
               {'pitch_type': 'SI', 'zone_id': 'high'}]
    q = np.array([.5000, .4996, .4990])
    assert choose_best(q) == 0
    assert choose_near_value(q, actions, {'ST': 20, 'SI': 50}, .05) == 1
    assert choose_near_value(q, actions, {'ST': 20, 'SI': 50}, .01) == 0


def test_minimax_uses_within_member_regret_and_ensemble_tie():
    members = np.array([[.8, .7, .75], [.6, .8, .75]])
    # Both members have .05 worst regret for action 2, smaller than others.
    assert choose_minimax(members, np.array([.7, .7, .7])) == 2
    shifted = members + np.array([[10], [-10]])
    assert choose_minimax(shifted, np.array([.7, .7, .7])) == 2


def test_comparison_rejects_member_action_support_change():
    base = {'variant': 'ensemble', 'actions': [{'pitch_type': 'FF', 'zone_id': 'low'}],
            'pitches': [{'pitch_key': [1, 1, 1], 'q_values': [.5],
                         'q_by_action': {'FF|low': .5}, 'pitcher': 1, 'balls': 0,
                         'strikes': 0, 'actual_type': 'FF'}]}
    item = {'evaluations': [base] + [base | {'variant': f'member_seed_{s}',
            'actions': [{'pitch_type': 'SI', 'zone_id': 'low'}] if s == 43 else base['actions']}
            for s in range(42, 47)], 'repertoire_counts': {'FF': 30}}
    try:
        compare_pitch(item, 0, .05)
    except ValueError as exc:
        assert 'Action support/order differs' in str(exc)
    else:
        raise AssertionError('Changed support was accepted')


def test_leave_one_out_selection_excludes_held_member_then_scores_on_it():
    actions = [{'pitch_type': 'FF', 'zone_id': 'low'},
               {'pitch_type': 'SI', 'zone_id': 'low'}]
    def evaluation(name, q):
        return {'variant': name, 'actions': actions,
                'pitches': [{'pitch_key': [1, 1, 1], 'q_values': q,
                             'q_by_action': {'FF|low': q[0], 'SI|low': q[1]},
                             'pitcher': 1, 'balls': 0, 'strikes': 0,
                             'actual_type': 'FF'}]}
    item = {'repertoire_counts': {'FF': 10, 'SI': 100},
            'evaluations': [evaluation('ensemble', [.6, .6]),
                            evaluation('member_seed_42', [.9, .1])] +
                           [evaluation(f'member_seed_{seed}', [.6, .5998])
                            for seed in range(43, 47)]}
    near, _ = compare_pitch(item, 0, .05)
    held = near['heldout'][0]
    assert held['held_seed'] == 42
    assert held['chosen_index'] == 1  # four other members favor repertoire SI within .05 pp
    assert held['reference_index'] == 0
    assert held['heldout_delta_pp'] == -80  # scored on excluded seed42 Q

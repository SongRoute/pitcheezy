import numpy as np

from scripts.a_common_comparison import game_bootstrap, policy_matrix, NAMES, JUDGES


def test_game_bootstrap_and_policy_accounting():
    values = np.array([.2, .4, .6])
    got = game_bootstrap(values, np.array([1, 1, 2]), 20260923, 100)
    assert got['mean'] == np.mean(values)
    assert got['games'] == 2
    rows = []
    for game in (1, 2):
        rows.append({'game_pk': game, 'values': {judge: {'repertoire': .4,
            **{name: (.4 if name == 'count_hand' else .41) for name in NAMES}} for judge in JUDGES}})
    matrix = policy_matrix(rows, JUDGES, 20260923, 100)
    for judge in JUDGES:
        assert matrix[judge]['policies']['count_hand']['delta_vs_repertoire_pp']['mean'] == 0
        np.testing.assert_allclose(matrix[judge]['policies']['known_type']['delta_vs_repertoire_pp']['mean'], 1)

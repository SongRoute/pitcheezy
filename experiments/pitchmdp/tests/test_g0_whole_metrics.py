import numpy as np
import pandas as pd
import pytest

from pitchmdp.data import KEY
from pitchmdp.matrix_g0_whole_metrics import (frozen_five_predictions, aligned_population,
    replay_panel, evaluate_candidates, evaluate_whole)


def fixture():
    n = 600
    keys = np.column_stack([np.repeat(np.arange(60), 10), np.tile(np.arange(10), 60), np.ones(n, int)])
    y = np.arange(n) % 10
    b = {'dev': np.full((n, 10), .1), 'dev_keys': keys, 'dev_y': y,
         'dev_game_pk': keys[:, 0], 'dev_pitcher': np.arange(n) % 3}
    metadata = pd.DataFrame(keys, columns=KEY)
    metadata['in_cpanel'] = np.arange(n) % 3 == 0
    metadata['game_role'] = 'starter'
    metadata['throwing_hand'] = 'R'
    metadata['train_volume'] = 'high'
    metadata['two_strikes'] = False
    metadata['runners_on'] = False
    metadata['seen_pitcher'] = True
    metadata['seen_batter'] = True
    metadata['month'] = '2025-07'
    members = [{**b, 'dev_raw': b['dev'].copy()} for _ in range(5)]
    report = {'selection': {'model_weight': .7}, 'seeds': [{'blend_selection': {'model_weight': .2+i*.1}} for i in range(5)]}
    panel = {k: v[metadata.in_cpanel.to_numpy()] for k, v in b.items()}
    return b, metadata, members, report, panel


def test_exact_complement_keeps_shared_games_and_rejects_wrong_panel():
    b, metadata, members, report, panel = fixture()
    population, indices, counts = aligned_population(b, metadata, panel)
    assert counts['overlap_pitches'] == 200
    assert counts['complement_pitches'] == 400
    assert counts['shared_games_panel_complement'] == 60
    assert not (population['cpanel'] & population['non_cpanel']).any()
    bad = dict(panel, dev_y=(panel['dev_y']+1) % 10)
    with pytest.raises(ValueError, match='Panel identity'):
        aligned_population(b, metadata, bad)
    wrong_metadata = metadata.copy()
    wrong_metadata.loc[0, 'in_cpanel'] = False
    with pytest.raises(ValueError, match='intersection'):
        aligned_population(b, wrong_metadata, panel)


def test_five_seed_weights_and_replay():
    b, metadata, members, report, panel = fixture()
    members[0] = {**members[0], 'dev': np.roll(np.eye(10)[b['dev_y']] * .5 + .05, 1, axis=1)}
    predicted = frozen_five_predictions(members, b, report)
    assert np.allclose(predicted['primary'], .7*np.mean([m['dev'] for m in members], axis=0)+.3*b['dev'])
    assert np.allclose(predicted['seed_primary'][0], .2*members[0]['dev']+.8*b['dev'])
    with pytest.raises(ValueError, match='Exactly five'):
        frozen_five_predictions(members[:3], b, report)
    _, indices, _ = aligned_population(b, metadata, panel)
    p = {k: v[:, indices] if k == 'seed_primary' else v[indices] for k, v in predicted.items()}
    assert replay_panel(predicted, p, indices)['passed']
    p['primary'] = p['primary']+.001
    with pytest.raises(ValueError, match='replay failed'):
        replay_panel(predicted, p, indices)


def test_candidate_inventory_and_absent_bound_slots():
    b, metadata, members, report, panel = fixture()
    base = frozen_five_predictions(members, b, report)
    results = evaluate_candidates(b['dev_y'], b['dev_game_pk'], metadata, base,
        {'I1': base}, {'I1': True, 'I2': False}, draws=100, r_draws=100)
    assert results['multiplicity']['R_family_size'] == 52
    assert results['candidates']['I2']['adjusted_p'] == 1
    assert len(results['candidates']['I2']['R']['groups']) == 13
    assert results['candidates']['I1']['R']['status'] == 'unconfirmed'
    assert results['candidates']['I1']['R']['per_bound_alpha'] == .05/52
    with pytest.raises(ValueError, match='inventory'):
        evaluate_candidates(b['dev_y'], b['dev_game_pk'], metadata, base,
            {'I1': base}, {'I1': False, 'I2': False}, draws=100, r_draws=100)


def test_baseline_estimands_absolute_ci_and_structural_missingness():
    b, metadata, members, report, panel = fixture()
    predicted = frozen_five_predictions(members, b, report)
    result = evaluate_whole(b, metadata, predicted, panel, draws=100, r_draws=100)
    assert result['estimands']['whole']['n'] == 600
    assert result['estimands']['non_cpanel']['n'] == 400
    assert result['R']['family_size'] == 24
    assert result['R']['groups']['volume_zero']['passed'] is None
    assert np.allclose(result['estimands']['whole']['g0_absolute_ci']['nll_ci95'], np.log(10))
    assert result['independent_confirmation'] is None
    assert result['held_out_confirmation'] is False

"""Synthetic tests for the DRAFT weight x temperature predictor search (no real data)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import search_predictor_blend_temperature as search  # noqa: E402
from pitchmdp.matrix_metrics import prediction_metrics  # noqa: E402

CONFIG = REPO / 'configs' / 'DRAFT-PREDSEARCH-WT77-20261003.json'


@pytest.fixture(scope='module')
def arrays():
    return search.synthetic(n_blend=300, n_dev=400, n_mlb=1500, games=25)


@pytest.fixture(scope='module')
def reference(arrays):
    return search.jsonable(search.reference_from(*arrays))


@pytest.fixture(scope='module')
def result(arrays, reference):
    return search.search(*arrays, reference, 1e-9, 257)


def test_temper_identity_normalization_and_direction():
    p = np.random.default_rng(0).dirichlet(np.ones(10), 50)
    assert search.temper(p, 1) is p
    entropy = lambda q: -(q * np.log(q)).sum(1)
    for t in search.TEMPERATURES:
        assert np.allclose(search.temper(p, t).sum(1), 1)
    assert np.all(entropy(search.temper(p, .8)) < entropy(p)) and np.all(entropy(search.temper(p, 1.2)) > entropy(p))
    logits = np.log(p) + 3.0  # log-probabilities / T then softmax == logits / T then softmax
    soft = np.exp(logits / 1.1) / np.exp(logits / 1.1).sum(1, keepdims=True)
    assert np.allclose(search.temper(p, 1.1), soft)


def test_config_pins_grid_and_refuses_unregistered_full_run():
    config = json.loads(CONFIG.read_text())
    search.config_check(config, full=False)
    assert len(search.candidates()) == 77
    with pytest.raises(ValueError, match='REGISTERED'):
        search.config_check(config, full=True)
    for path, value in ((('grid', 'temperature'), [.9, 1, 1.1]), (('seeds',), [0, 1, 2])):
        bad = copy.deepcopy(config)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            search.config_check(bad, full=False)
    registered = {**copy.deepcopy(config), 'status': 'REGISTERED', 'experiment_id': 'EXP-P19-001'}
    registered['execution']['enabled'] = True
    search.config_check(registered, full=True)


def test_gate_reproduces_independent_p16_scorer_and_scores_all(result):
    gate = result['reproduction_gate']
    assert gate['passed'] and len(gate['checks']) == 21
    assert result['candidate_count'] == 77 and len(result['order_by_mean_nll_rank']) == 77
    mix = result['candidates']['w0.5_T1.00']
    assert mix['is_baseline'] == 'G15-direct-g0-mix'
    assert mix['cpanel']['delta_nll_vs_G15'] == 0 and mix['mlb']['delta_nll_vs_G15'] == 0
    g0 = result['candidates']['w0.0_T1.00']
    assert g0['mlb']['seed_delta_nll_vs_G0'] == [0.0] * 5


def test_gate_failure_withholds_candidates(arrays, reference):
    bad = copy.deepcopy(reference)
    bad['mlb_dev_blended']['comparison']['G15-direct-g0-mix']['candidate']['log_loss'] += 1e-6
    out = search.search(*arrays, bad, 1e-9, 257)
    assert out['reproduction_gate']['passed'] is False and 'candidates' not in out


def test_stream_is_chunk_invariant_and_matches_dense_metrics(arrays):
    direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb = arrays
    combo = (.3, .9)
    cp = search.cpanel(direct, g0, baseline, *combo, periods_cp)
    weights = {combo: (cp['june_weight'], cp['seed_june_weights'])}
    small = search.mlb_stream(direct_mlb, g0_mlb, frequency['mlb_dev'], [combo], weights, periods_mlb, 97)[combo]
    whole = search.mlb_stream(direct_mlb, g0_mlb, frequency['mlb_dev'], [combo], weights, periods_mlb, 10 ** 9)[combo]
    assert np.allclose(small['mean'], whole['mean'], atol=1e-12, rtol=0)
    assert np.allclose(small['seed_mean'], whole['seed_mean'], atol=1e-12, rtol=0)
    members = [search.temper(.3 * d['mlb_dev'] + .7 * g['dev'], .9) for d, g in zip(direct_mlb, g0_mlb)]
    dense = cp['june_weight'] * np.mean(members, 0) + (1 - cp['june_weight']) * frequency['mlb_dev']
    metrics = prediction_metrics(g0_mlb[0]['dev_y'], dense)
    assert abs(small['mean'][0] - metrics['log_loss']) < 1e-12
    assert abs(small['ece10'] - metrics['top_label_ece10']) < 1e-12
    assert sum(v['n'] for v in small['periods'].values()) == len(dense)


def test_missing_seed_and_row_realignment_refused(arrays, reference):
    direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb = arrays
    with pytest.raises(ValueError, match='five seeds'):
        search.search(direct[:4], g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb, reference)
    shuffled = [dict(m) for m in direct_mlb]
    order = np.random.default_rng(1).permutation(len(shuffled[2]['mlb_dev_keys']))
    shuffled[2] = {k: v[order] for k, v in shuffled[2].items()}
    with pytest.raises(ValueError, match='Unpaired'):
        search.search(direct, g0, baseline, shuffled, g0_mlb, frequency, periods_cp, periods_mlb, reference)
    with pytest.raises(ValueError, match='Period'):
        search.search(direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp[:-1], periods_mlb, reference)


def test_rank_pareto_is_descriptive_and_consistent(result):
    rows = result['candidates']
    for name in result['pareto_four_objective']:
        a = rows[name]
        va = [a['cpanel']['delta_nll_vs_G15'], a['mlb']['delta_nll_vs_G15'],
              a['cpanel']['delta_brier_vs_G15'], a['mlb']['delta_brier_vs_G15']]
        for other in rows.values():
            vb = [other['cpanel']['delta_nll_vs_G15'], other['mlb']['delta_nll_vs_G15'],
                  other['cpanel']['delta_brier_vs_G15'], other['mlb']['delta_brier_vs_G15']]
            assert not (all(b <= x for b, x in zip(vb, va)) and any(b < x for b, x in zip(vb, va)))
    assert all('p_value' not in json.dumps(r) for r in rows.values())


def test_period_mapping_refuses_2026_and_unknown_games(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    from pitchmdp.data import hash_file
    path = tmp_path / 'statcast_2025.parquet'
    pq.write_table(pa.table({'game_pk': [1, 1, 2], 'game_date': ['2025-07-01', '2025-07-01', '2025-08-02']}), path)
    local = {'raw_root': str(tmp_path), 'raw_allowlist': ['statcast_2025.parquet']}
    config = {'periods': {'source_files': [{'file': 'statcast_2025.parquet', 'sha256': hash_file(path)}]}}
    (out,) = search.game_periods(local, config, np.array([2, 1, 1]))
    assert out.tolist() == ['2025-08', '2025-07', '2025-07']
    with pytest.raises(ValueError, match='no pinned game_date'):
        search.game_periods(local, config, np.array([3]))
    config['periods']['source_files'][0]['file'] = 'statcast_2026.parquet'
    with pytest.raises(ValueError, match='<=2025'):
        search.game_periods(local, config, np.array([1]))


def test_archive_loader_accepts_json_string_paths_and_verifies_hash(tmp_path):
    p = tmp_path / 'frequency.npz'
    expected = np.arange(5)
    np.savez(p, dev_y=expected)
    spec = {'path': str(p), 'sha256': search.hash_file(p)}
    assert np.array_equal(search.load_npz(spec, ['dev_y'])['dev_y'], expected)
    spec['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='archive changed'):
        search.load_npz(spec, ['dev_y'])

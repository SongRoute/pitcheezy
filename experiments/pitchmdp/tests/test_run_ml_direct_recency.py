"""Synthetic checks for scripts/run_ml_direct_recency.py (no real data, no real fit)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(PROJECT / 'tests')]
import run_ml_direct_recency as runner  # noqa: E402
import run_ml_direct_head as direct  # noqa: E402
from pitchmdp.data import hash_file  # noqa: E402
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402
from test_run_ml_direct_head import Context, archives, arm, frame  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'DRAFT-DIRECT-RECENCY-20261003.json').read_text())


def test_weights_follow_chronology_and_normalise():
    dates = ['2023-05-15', '2024-04-30', '2024-10-01', '2025-04-30', '2025-04-30']
    w = runner.recency_weights(dates, half_life_days=365)
    assert w.mean() == pytest.approx(1.) and np.all(np.diff(w) >= 0)
    assert w[3] / w[1] == pytest.approx(2.)                    # exactly one half-life (365 days) apart
    assert w[3] == w[4]
    report = runner.weight_report(dates, w)
    assert report['ess'] == pytest.approx(w.sum() ** 2 / np.square(w).sum()) and 0 < report['ess_ratio'] <= 1
    assert sum(report['share_by_year'].values()) == pytest.approx(1.) and report['rows_by_year'] == {'2023': 1, '2024': 2, '2025': 2}
    shorter = runner.recency_weights(dates, half_life_days=180)
    assert runner.weight_report(dates, shorter)['ess'] < report['ess']   # shorter half-life concentrates weight


def test_uniform_limit_matches_control_semantics():
    same_day = runner.recency_weights(['2024-06-01'] * 4, half_life_days=180)
    assert np.array_equal(same_day, np.ones(4))                # equal dates -> exactly the uniform control weights
    long = runner.recency_weights(['2023-05-15', '2025-04-30'], half_life_days=1e9)
    assert np.allclose(long, 1., atol=1e-6)


@pytest.mark.parametrize('dates, half_life, message', [
    (['2025-05-01'], 365, 'after the recency anchor'),
    (['2026-04-01'], 365, 'after the recency anchor'),
    ([None, '2024-01-01'], 365, 'Missing'),
    (['not a date'], 365, 'Missing'),
    (['2024-01-01'], 0, 'positive'),
    (['2024-01-01'], -5, 'positive'),
    (['2024-01-01'], float('inf'), 'positive'),
    (['2024-01-01'], True, 'positive'),
])
def test_rejects_future_missing_dates_and_bad_half_life(dates, half_life, message):
    with pytest.raises(ValueError, match=message):
        runner.recency_weights(dates, half_life_days=half_life)


def test_draft_config_matches_runner_and_refuses_real_runs():
    runner.config_check(CONFIG, smoke=True)
    with pytest.raises(ValueError, match='registered'):
        runner.config_check(CONFIG)
    registered = copy.deepcopy(CONFIG)
    registered['execution']['enabled'] = True
    with pytest.raises(ValueError, match='registered'):
        runner.config_check(registered)                       # still DRAFT
    registered['status'] = 'REGISTERED'
    runner.config_check(registered)
    for path, value, message in ((('variants',), [{'name': 'hl365', 'half_life_days': 365}], 'grid'),
                                 (('anchor',), '2025-06-30', 'anchor'),
                                 (('mix', 'direct_weight'), .7, 'Controls'),
                                 (('unchanged_from_p15', 'training', 'width'), 256, 'Architecture'),
                                 (('control', 'preparation_sha256'), '', 'pinned')):
        bad = copy.deepcopy(CONFIG)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            runner.config_check(bad, smoke=True)
    assert CONFIG['unchanged_from_p15']['training'] == json.loads(
        (PROJECT.parents[1] / 'configs' / 'EXP-P15-001.yaml').read_text())['unchanged_from_g0']['training']


def test_identity_records_own_source_and_variant(tmp_path):
    local = tmp_path / 'local.json'
    local.write_text('{}')
    ident = runner.identity(CONFIG, local, 'hl180')
    assert ident['source_hashes'] == direct.source_hashes()      # what direct.predict checks
    assert ident['recency_source_sha256'] == hash_file(PROJECT / runner.SELF)
    assert (ident['variant'], ident['half_life_days'], ident['anchor']) == ('hl180', 180, '2025-04-30')
    assert ident != runner.identity(CONFIG, local, 'hl730')


class FakeModel:
    calls = []

    def __init__(self, kind, seed, width):
        self.report = {'kind': kind, 'seed': seed, 'width': width, 'best_epoch': 1}

    def fit(self, train, labels, early, early_labels, **kwargs):
        assert not train[0][:, -1, :8].any() and not early[0][:, -1, :8].any()   # current physics stays zero
        FakeModel.calls.append(kwargs)
        return self

    def save(self, path):
        Path(path).write_bytes(b'model')


def synthetic_data():
    f = frame()
    f['game_date'] = pd.to_datetime(['2023-06-01'] * 3 + ['2025-04-01'] * 3 + ['2025-05-02'] * 2)
    store = MatrixHistoryStore.from_frame(f)
    parts = {'train': f.iloc[:6], 'earlystop': f.iloc[6:]}
    return None, {'samples': {}}, store, Context(), parts, None


def prep_for(variant, tmp_path):
    local = tmp_path / 'local.json'
    local.write_text('{}')
    return {'identity': runner.identity(CONFIG, local, variant)}


def test_fit_passes_date_weights_and_isolates_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'MatrixModel', FakeModel)
    FakeModel.calls.clear()
    data = synthetic_data()
    out = {v: tmp_path / 'family' / v for v in ('hl180', 'hl730')}
    for variant, dest in out.items():
        runner.fit(CONFIG, data, dest, prep_for(variant, tmp_path), 0, smoke=False)
    w180, w730 = (np.asarray(call['sample_weight']) for call in FakeModel.calls)
    expected = runner.recency_weights(data[4]['train'].game_date, half_life_days=180)
    assert np.array_equal(w180, expected) and w180[0] < w180[-1] and w730[0] > w180[0]
    assert FakeModel.calls[0]['epochs'] == 30 and FakeModel.calls[0]['learning_rate'] == .0005
    for variant, dest in out.items():                              # separate state per variant
        report = json.loads((dest / '0' / 'fit.json').read_text())
        assert report['earlystop_weighting'] == 'uniform' and report['train_weights']['rows'] == 6
        assert (dest / '0' / 'fit_state.json').is_file()
    runner.fit(CONFIG, data, out['hl180'], prep_for('hl180', tmp_path), 0, smoke=False)   # verified reuse
    assert len(FakeModel.calls) == 2
    with pytest.raises(ValueError, match='identity changed'):
        runner.fit(CONFIG, data, out['hl180'], prep_for('hl730', tmp_path), 0, smoke=False)
    partial = tmp_path / 'family' / 'hl365'
    (partial / '1').mkdir(parents=True)
    (partial / '1' / 'model.pt').write_bytes(b'half')
    with pytest.raises(ValueError, match='Incomplete fit'):
        runner.fit(CONFIG, data, partial, prep_for('hl365', tmp_path), 1, smoke=False)


def test_prepare_refuses_unsafe_existing_output(tmp_path):
    dest = tmp_path / 'hl365'
    dest.mkdir()
    (dest / 'stray.txt').write_text('x')
    with pytest.raises(ValueError, match='Incomplete preparation'):
        runner.prepare(CONFIG, synthetic_data(), dest, prep_for('hl365', tmp_path)['identity'])


def test_descriptive_comparison_against_fixed_controls():
    baseline = archives(np.random.default_rng(0))
    g0, uniform = arm(baseline, (1, 2, 3)), arm(baseline, (4, 5, 6))
    same = runner.describe(uniform, uniform, g0, baseline, .5)
    assert all(same[a]['after_june_blend']['paired']['nll']['delta'] == 0 for a in ('direct', 'mix'))
    assert 'N' not in same['direct']['after_june_blend']                   # no decision attached
    better = [{**m, 'dev': (m['dev'] + np.eye(10)[m['dev_y']]) / 2, 'blend': (m['blend'] + np.eye(10)[m['blend_y']]) / 2}
              for m in uniform]
    result = runner.describe(better, uniform, g0, baseline, .5)
    assert result['direct']['after_june_blend']['paired']['nll']['delta'] < 0
    assert result['mix']['after_june_blend']['paired']['nll']['delta'] < 0

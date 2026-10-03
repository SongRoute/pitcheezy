"""Synthetic CPU checks for scripts/run_ml_direct_workload.py (no real data, no real fit)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(PROJECT / 'tests')]
import run_ml_direct_workload as runner  # noqa: E402
import run_ml_direct_head as direct  # noqa: E402
from pitchmdp.data import hash_file  # noqa: E402
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402
from pitchmdp.matrix_models import MatrixModel, MatrixNetwork  # noqa: E402
from test_run_ml_direct_head import frame  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'DRAFT-DIRECT-WORKLOAD-20261003.json').read_text())
RAW = runner.RAW_COLUMNS


def raw_rows(rows):
    return pd.DataFrame(rows, columns=RAW).assign(game_date=lambda f: pd.to_datetime(f.game_date))


def small_raw():
    # pitcher 10: 2023-09-01 game 5 (2 records); 2024-04-01 game 1 PA1 (3 records), PA2 by pitcher 20,
    # PA3..PA11 by pitcher 10 (one record each, 9 PAs -> second time through after 9); 2024-04-01 game 6
    # (same-day second game, pitcher 10). pitcher 30 appears once (no earlier date).
    rows = [('2023-09-01', 5, 1, 1, 10), ('2023-09-01', 5, 1, 2, 10),
            ('2024-04-01', 1, 1, 1, 10), ('2024-04-01', 1, 1, 2, 10), ('2024-04-01', 1, 1, 3, 10),
            ('2024-04-01', 1, 2, 1, 20)]
    rows += [('2024-04-01', 1, ab, 1, 10) for ab in range(3, 12)]
    rows += [('2024-04-01', 6, 1, 1, 10), ('2024-04-02', 7, 1, 1, 30)]
    return raw_rows(rows)


def test_feature_definitions_by_hand():
    raw = small_raw()
    parent = raw.sample(frac=1., random_state=3).reset_index(drop=True)    # mixed order
    values = runner.workload_columns(raw, parent)
    got = {tuple(k): v for k, v in zip(parent[runner.KEY].to_numpy(), values)}
    log = lambda d: np.log1p(d) / np.log(181)  # noqa: E731
    assert np.allclose(got[(5, 1, 1)], [0, 0, 0, 1]) and np.allclose(got[(5, 1, 2)], [.01, 0, 0, 1])
    assert np.allclose(got[(1, 1, 1)], [0, 0, log(180), 0]) and log(180) == 1.   # 213 days capped at 180
    assert np.allclose(got[(1, 1, 3)], [.02, 0, 1., 0])
    assert np.allclose(got[(1, 2, 1)], [0, 0, 0, 1])                    # pitcher 20: rest missing -> 0 plus flag
    assert np.allclose(got[(1, 3, 1)][:2], [.03, 0]) and np.allclose(got[(1, 10, 1)][:2], [.10, 0])
    assert np.allclose(got[(1, 11, 1)][:2], [.11, 1 / 3])               # tenth PA faced -> second time through
    assert np.allclose(got[(6, 1, 1)], [0, 0, log(180), 0])            # same-day earlier game omitted by definition
    assert np.allclose(got[(7, 1, 1)], [0, 0, 0, 1])
    assert values.shape == (len(parent), 4) and np.isfinite(values).all()


def test_counts_come_from_all_raw_records_not_the_filtered_parent():
    raw = small_raw()
    parent = raw[raw.pitch_number.eq(3) | raw.at_bat_number.eq(3)].reset_index(drop=True)   # eligibility-like subset
    values = runner.workload_columns(raw, parent)
    assert np.allclose(values[:, 0], [.02, .03])     # earlier raw records count even when not parent rows


def test_no_current_or_future_effect_and_whole_2025_prefix():
    raw = small_raw()
    full = runner.workload_columns(raw, raw)
    for cut in range(1, len(raw)):
        prefix = raw.iloc[:cut]
        assert np.array_equal(runner.workload_columns(prefix, prefix), full[:cut])
    later = pd.concat([raw, raw_rows([('2025-06-01', 99, 1, 1, 10), ('2024-04-01', 1, 12, 1, 10)])], ignore_index=True)
    assert np.array_equal(runner.workload_columns(later, raw), full)    # added later rows change nothing earlier


def test_alignment_rejects_unknown_duplicate_and_mismatched_rows():
    raw = small_raw()
    with pytest.raises(ValueError, match='exact raw'):
        runner.workload_columns(raw, raw_rows([('2024-04-01', 1, 1, 9, 10)]))
    with pytest.raises(ValueError, match='unique'):
        runner.workload_columns(raw, pd.concat([raw.iloc[:2], raw.iloc[:1]]))
    with pytest.raises(ValueError, match='unique'):
        runner.workload_columns(pd.concat([raw, raw.iloc[:1]], ignore_index=True), raw.iloc[:2])
    moved = raw.iloc[:2].copy()
    moved['pitcher'] = 99
    with pytest.raises(ValueError, match='disagree'):
        runner.workload_columns(raw, moved)
    with pytest.raises(ValueError, match='only keys'):
        runner.workload_columns(raw.assign(n_thruorder_pitcher=1), raw)


def write_raw(root, extra_columns=True, year_2024=None):
    root.mkdir(parents=True, exist_ok=True)
    raw = small_raw() if year_2024 is None else year_2024
    for year in runner.YEARS:
        part = raw[raw.game_date.dt.year.eq(year)].copy()
        if extra_columns:   # outcome / Savant workload / future columns exist in the file but are never read
            part = part.assign(description='ball', n_thruorder_pitcher=1, pitcher_days_until_next_game=4)
        part.to_parquet(root / f'statcast_{year}.parquet', index=False)


def pins(tmp_path):
    return {name: hash_file(tmp_path / 'raw' / name) for name in runner.ALLOWLIST}


def local_file(tmp_path, raw=None):
    write_raw(tmp_path / 'raw', year_2024=raw)
    local = tmp_path / 'local.json'
    local.write_text(json.dumps({'raw_root': str(tmp_path / 'raw'), 'raw_allowlist': runner.ALLOWLIST}))
    return local


def test_raw_reader_opens_only_keys_and_the_2023_2025_allowlist(tmp_path):
    local = json.loads(local_file(tmp_path).read_text())
    raw = runner.read_raw(local, pins(tmp_path))
    assert list(raw.columns) == RAW and len(raw) == len(small_raw())
    with pytest.raises(ValueError, match='allowlist'):
        runner.read_raw({**local, 'raw_allowlist': [*runner.ALLOWLIST, 'statcast_2026.parquet']}, pins(tmp_path))
    bad = small_raw()
    bad.loc[len(bad)] = [pd.Timestamp('2026-04-01'), 50, 1, 1, 10]
    pd.concat([bad[bad.game_date.dt.year.eq(2025)], bad.tail(1)]).to_parquet(tmp_path / 'raw' / 'statcast_2025.parquet')
    with pytest.raises(ValueError, match='outside season 2025'):
        runner.read_raw(local, pins(tmp_path))


def test_raw_files_must_equal_the_frozen_bytes(tmp_path):
    frozen = json.loads((PROJECT.parents[1] / 'configs' / 'G0-RESEARCH-FROZEN-v1.json').read_text())
    control = json.loads((PROJECT.parents[1] / 'results' / 'EXP-P15-001.json').read_text())

    def sources(node):   # every {'file': statcast_*, 'sha256': ...} record anywhere in the frozen JSON
        if isinstance(node, dict):
            if str(node.get('file', '')).startswith('statcast_') and 'sha256' in node:
                yield node['file'], node['sha256']
            for value in node.values():
                yield from sources(value)
        elif isinstance(node, list):
            for value in node:
                yield from sources(value)
    for document in (frozen, control):
        found = set(sources(document))
        assert found and found == set(runner.FROZEN_RAW_SHA256.items())
    assert CONFIG['feature_rules']['raw_sha256'] == runner.FROZEN_RAW_SHA256
    local = json.loads(local_file(tmp_path).read_text())
    good = pins(tmp_path)
    with pytest.raises(ValueError, match='frozen 2023-2025 bytes'):
        runner.read_raw(local)                                   # synthetic files are not the frozen bytes
    with pytest.raises(ValueError, match='frozen 2023-2025 bytes'):
        runner.check_raw_pins(good)
    small_raw().head(2).to_parquet(tmp_path / 'raw' / 'statcast_2023.parquet', index=False)   # same name, other bytes
    with pytest.raises(ValueError, match='frozen 2023-2025 bytes'):
        runner.read_raw(local, good)
    with pytest.raises(ValueError, match='allowlist'):
        runner.read_raw(local, {k: good[k] for k in runner.ALLOWLIST[:2]})


class Context:
    """Sharing-style context: two row-dependent columns plus seven routing columns."""
    def transform(self, part):
        n = len(part)
        return np.column_stack((part.pitch_number, part.at_bat_number, np.zeros((n, 6)), part.pitcher)).astype(np.float32)


def synthetic(tmp_path, variant='block', device='cpu'):
    f = pd.concat([frame()] * 4, ignore_index=True)
    f['game_pk'] = f.game_pk + 10 * np.repeat(np.arange(4), 8)
    f['game_date'] = f.game_date + pd.to_timedelta(np.repeat(np.arange(4), 8) * 7, unit='D')
    f['description'] = np.tile(['ball', 'called_strike', 'foul', 'swinging_strike'], 8)
    raw = pd.concat([f[RAW], raw_rows([('2023-09-01', 500, 1, 1, 10), ('2024-04-01', 1, 9, 1, 77)])], ignore_index=True)
    local = local_file(tmp_path, raw)
    store = MatrixHistoryStore.from_frame(f)
    parts = {'train': f.iloc[:20], 'earlystop': f.iloc[20:26], 'temperature': f.iloc[20:26],
             'blend': f.iloc[26:29], 'dev': f.iloc[29:]}
    base = (None, {'samples': {}, 'clusters': {'cluster_counts': {}}}, store, Context(), parts, None)
    data, values = runner.workload_data(json.loads(local.read_text()), base, variant, pins(tmp_path))
    prep = {'identity': runner.identity(CONFIG, local, variant, device), 'added_parameters': runner.added_parameters(
        len(runner.VARIANTS[variant]))}
    return base, data, values, prep, local


def test_context_wrapper_inserts_before_routing_and_zero_extra_is_the_control_path(tmp_path):
    base, data, values, _, _ = synthetic(tmp_path)
    store, wrapped = data[2], data[3]
    rows = np.array([30, 3, 17, 0, 9])                                  # mixed order
    plain = Context().transform(store.frame.iloc[rows])
    out = wrapped.transform(store.frame.iloc[rows])
    assert out.shape[1] == plain.shape[1] + 4 and out.dtype == plain.dtype
    assert np.isclose(values[6, 2], np.log1p(1) / np.log(181)) and values[0, 2] == 1. and not values[:8, 3].any()
    assert np.array_equal(out[:, :2], plain[:, :2]) and np.array_equal(out[:, -7:], plain[:, -7:])
    assert np.array_equal(out[:, 2:6], values[rows].astype(np.float32))
    stripped = runner.training_arrays(direct.direct_arrays(store, wrapped, rows))[2]
    assert stripped.shape[1] == 2 + 4 and np.array_equal(stripped[:, 2:], values[rows].astype(np.float32))
    zero = runner.WorkloadContext(Context(), store.frame, np.zeros((len(store.frame), 0)))
    for a, b in zip(direct.direct_arrays(store, zero, rows), direct.direct_arrays(store, Context(), rows)):
        assert np.array_equal(a, b)
    shuffled = store.frame.iloc[rows].copy()
    shuffled.index = shuffled.index[::-1]
    with pytest.raises(ValueError, match='misaligned'):
        wrapped.transform(shuffled)
    counts = runner.workload_data(json.loads(synthetic(tmp_path / 'c')[4].read_text()), base, 'counts', pins(tmp_path / 'c'))[0][3]
    assert np.array_equal(counts.transform(store.frame.iloc[rows])[:, 2], values[rows, 0].astype(np.float32))


def test_frozen_rows_outside_2023_2025_are_refused(tmp_path):
    base, _, _, _, local = synthetic(tmp_path)
    late = base[2].frame.copy()
    late.loc[0, 'game_date'] = pd.Timestamp('2026-04-01')
    store = type('S', (), {'frame': late})()
    with pytest.raises(ValueError, match='2023-2025'):
        runner.workload_data(json.loads(local.read_text()), (None, None, store, Context(), {}, None), 'counts')


def test_config_matches_runner_and_refuses_changes():
    runner.config_check(CONFIG, smoke=True)
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(CONFIG)
    for path, value, message in ((('variants', 1, 'columns'), ['prior_records_per100'], 'pre-fixed'),
                                 (('variants', 0, 'added_parameters'), 128, 'parameter counts'),
                                 (('feature_rules', 'rest_cap_days'), 365, 'Feature'),
                                 (('feature_rules', 'raw_columns'), [*RAW, 'n_thruorder_pitcher'], 'Feature'),
                                 (('feature_rules', 'raw_sha256', 'statcast_2024.parquet'), '0' * 64, 'Feature'),
                                 (('unchanged_from_p15', 'loss_weighting'), 'recency', 'uniform'),
                                 (('unchanged_from_p15', 'training', 'width'), 256, 'Architecture')):
        changed = copy.deepcopy(CONFIG)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            runner.config_check(changed, smoke=True)
    assert [v['added_parameters'] for v in CONFIG['variants']] == [256, 1024]


def test_identity_tracks_sources_layout_raw_files_and_device(tmp_path):
    *_, local = synthetic(tmp_path)
    ident = runner.identity(CONFIG, local, 'block')
    assert ident['source_hashes'] == direct.source_hashes() and 'pitchmdp/workload_features.py' in ident['workload_source_hashes']
    assert ident['feature_layout']['columns'] == list(runner.COLUMNS) and ident != runner.identity(CONFIG, local, 'counts')
    assert ident != runner.identity(CONFIG, local, 'block', 'cpu')
    raw_file = tmp_path / 'raw' / 'statcast_2024.parquet'
    assert ident['raw_file_sha256']['statcast_2024.parquet'] == hash_file(raw_file)
    pd.read_parquet(raw_file).head(3).to_parquet(raw_file, index=False)
    assert runner.identity(CONFIG, local, 'block') != ident
    stale = copy.deepcopy(ident)
    stale['workload_source_hashes'][runner.SELF] = '0' * 64
    with pytest.raises(ValueError, match='Sources changed'):
        runner.sources_unchanged({'identity': stale})


def test_fit_predict_save_load_verified_resume_and_partial_preservation(tmp_path, monkeypatch):
    _, data, values, prep, _ = synthetic(tmp_path)
    out = tmp_path / 'family'
    seen = []

    class Spy(MatrixModel):
        def fit(self, train, labels, early, early_labels, **kwargs):
            seen.append((train[0][:, -1, :8].copy(), early[0][:, -1, :8].copy(), train[2].shape[1], kwargs))
            return super().fit(train, labels, early, early_labels, **kwargs)

    monkeypatch.setattr(runner, 'MatrixModel', Spy)
    runner.fit(CONFIG, data, out, prep, 0, smoke=False)
    physics, early_physics, width, kwargs = seen[0]
    assert not physics.any() and not early_physics.any() and width == 2 + 4
    assert kwargs == {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}   # no sample_weight
    fitted = json.loads((out / '0' / 'fit.json').read_text())
    assert fitted['device'] == 'cpu' and fitted['context_width'] == 6 and fitted['added_parameters'] == 1024
    length, n_token = runner.training_arrays(direct.direct_arrays(data[2], Context(), [0]))[0].shape[1:]
    control = MatrixNetwork('flatten_mlp', 2, n_token, length, 128, 10)      # same inputs without workload
    assert fitted['parameter_count'] - sum(p.numel() for p in control.parameters()) == 1024
    stamp = (out / '0' / 'model.pt').stat().st_mtime_ns
    runner.fit(CONFIG, data, out, prep, 0, smoke=False)                          # verified reuse, no refit
    assert (out / '0' / 'model.pt').stat().st_mtime_ns == stamp and len(seen) == 1
    with pytest.raises(ValueError, match='identity changed'):
        runner.fit(CONFIG, data, out, {**prep, 'added_parameters': 0}, 0, smoke=False)
    runner.predict(data, out, prep, 0, smoke=False)
    with np.load(out / '0' / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(1), 1.) and np.all(p['dev_delivery_level'] == direct.NO_PHYSICS_LEVEL)
        assert np.array_equal(p['dev_keys'], data[4]['dev'][runner.KEY].to_numpy())
        dev = p['dev'].copy()
    runner.predict(data, out, prep, 0, smoke=False)                              # verified reuse
    loaded = MatrixModel.load(out / '0' / 'model.pt', 'cpu')
    ta = runner.training_arrays(direct.direct_arrays(data[2], data[3], data[4]['dev'].index.to_numpy()))
    assert np.array_equal(loaded.logits(ta), MatrixModel.load(out / '0' / 'model.pt', 'cpu').logits(ta))
    assert dev.shape == (len(data[4]['dev']), 10)
    (out / '0' / 'prediction_state.json').unlink()                               # interrupted after the archive
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out / '0').iterdir()}
    with pytest.raises(ValueError, match='Incomplete prediction'):
        runner.predict(data, out, prep, 0, smoke=False)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out / '0').iterdir()}
    (tmp_path / 'partial' / '1').mkdir(parents=True)
    (tmp_path / 'partial' / '1' / 'model.pt').write_bytes(b'half')
    with pytest.raises(ValueError, match='Incomplete fit'):
        runner.fit(CONFIG, data, tmp_path / 'partial', prep, 1, smoke=False)


def test_prepare_resume_verifies_features_and_refuses_partial(tmp_path):
    _, data, values, prep, _ = synthetic(tmp_path)
    made = runner.prepare(CONFIG, data, values, tmp_path / 'prep', prep['identity'])
    assert made['added_context_width'] == 4 and made['added_parameters'] == 1024 and made['dev_scores_read'] is False
    assert made['train_feature_report']['rest_missing']['rate'] == pytest.approx(values[:20, 3].mean())
    assert runner.prepare(CONFIG, data, values, tmp_path / 'prep', prep['identity']) == made
    shifted = values.copy()
    shifted[5, 0] += .01
    with pytest.raises(ValueError, match='differ from the prepared'):
        runner.prepare(CONFIG, data, shifted, tmp_path / 'prep', prep['identity'])
    stray = tmp_path / 'stray'
    stray.mkdir()
    (stray / 'x.txt').write_text('x')
    with pytest.raises(ValueError, match='Incomplete preparation'):
        runner.prepare(CONFIG, data, values, stray, prep['identity'])


def test_auto_cpu_preparation_is_not_comparable_to_mps_control(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    _, data, values, prep, local = synthetic(tmp_path, device='auto')
    assert prep['identity']['resolved_device'] == 'cpu'
    assert runner.prepare(CONFIG, data, values, tmp_path / 'cpu', prep['identity'])['comparable_device_to_control'] is False
    explicit = runner.identity(CONFIG, local, 'block', 'mps')
    assert runner.prepare(CONFIG, data, values, tmp_path / 'mps', explicit)['comparable_device_to_control'] is True


def test_score_compares_actual_fit_devices_with_the_control(tmp_path):
    control = tmp_path / 'control'
    for seed, device in ((0, 'mps'), (1, 'mps')):
        dest = control / str(seed)
        dest.mkdir(parents=True)
        (dest / 'fit.json').write_text(json.dumps({'report': {'device': device}}))
        runner.dump(dest / 'fit_state.json', {'artifact_hashes': runner.artifact_hashes(dest, ['fit.json'])})
    pairs = runner.device_pairs(control, {'0': {'device': 'mps'}, '1': {'device': 'cuda'}})
    assert pairs == {'0': ('mps', 'mps'), '1': ('mps', 'cuda')}
    (control / '0' / 'fit.json').write_text(json.dumps({'report': {'device': 'cpu'}}))
    with pytest.raises(ValueError):
        runner.device_pairs(control, {'0': {'device': 'mps'}})

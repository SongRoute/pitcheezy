"""Posthoc accounting for frozen B probabilities; only April game holdouts are refit."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'experiments/pitchmdp'),
                str(ROOT/'experiments/pitchmdp/scripts')]

import numpy as np
import pandas as pd
from pitchmdp.data import KEY
from pitchmdp.model import CountBaseline, OUTCOMES, outcome_labels
from a_small_eval import cohort_pa_rows, legal_baseline
from b_pa_history_eval import FamilyCountBaseline, add_prior, sha

CONFIG = ROOT/'configs/EXP-B-DIAG-001.json'
BCONFIG = ROOT/'configs/EXP-B-PAHISTORY-001.json'
BREPORT = ROOT/'results/EXP-B-PAHISTORY-001/results.json'
MANIFEST = ROOT/'results/EXP-A-S0S1-001/selection_manifest.json'
REPORT_DIR = ROOT/'results/EXP-B-DIAG-001'
BASE_KEYS = CountBaseline.keys
EXT_KEYS = [*BASE_KEYS, 'previous_family']


def loss(y, p):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def contribution(frame, group_columns):
    """Means are conditional; totals add to the global pitch-weighted loss delta."""
    out = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        out.append({'group': dict(zip(group_columns, map(str, key))), 'pitches': int(len(group)),
                    'baseline_nll': float(group.loss0.mean()), 'candidate_nll': float(group.loss1.mean()),
                    'mean_delta': float(group.delta.mean()), 'delta_sum': float(group.delta.sum()),
                    'global_contribution': float(group.delta.sum()/len(frame))})
    assert np.isclose(sum(row['delta_sum'] for row in out), frame.delta.sum(), atol=1e-10)
    return out


def support(train, dev):
    base_n = train.groupby(BASE_KEYS, dropna=False).size().to_dict()
    ext_n = train.groupby(EXT_KEYS, dropna=False).size().to_dict()
    def tagged(frame):
        b = np.array([base_n.get(key, 0) for key in frame[BASE_KEYS].itertuples(index=False, name=None)])
        e = np.array([ext_n.get(key, 0) for key in frame[EXT_KEYS].itertuples(index=False, name=None)])
        return {'pitches': len(frame), 'base_cells': len(base_n), 'extended_cells': len(ext_n),
                'base_unseen_pitches': int((b == 0).sum()), 'extended_unseen_pitches': int((e == 0).sum()),
                'extended_unseen_with_seen_base': int(((e == 0) & (b > 0)).sum()),
                'extended_support_pitches_le_2': int((e <= 2).sum()),
                'extended_support_pitches_le_5': int((e <= 5).sum()),
                'extended_support_pitches_le_10': int((e <= 10).sum()),
                'extended_support_quantiles': np.quantile(e, [0, .25, .5, .75, 1]).tolist(),
                'base_support_quantiles': np.quantile(b, [0, .25, .5, .75, 1]).tolist()}
    return {'train': tagged(train), 'dev': tagged(dev),
            'train_cell_sizes': {name: {'cells': int(sum(v == n for v in ext_n.values())),
                                        'pitches': int(sum(v for v in ext_n.values() if v == n))}
                                 for name, n in [('one', 1), ('two', 2)]},
            'train_extended_cell_size_quantiles': np.quantile(list(ext_n.values()), [0, .25, .5, .75, 1]).tolist()}


def run():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text())
    bcfg = json.loads(BCONFIG.read_text())
    bresult = json.loads(BREPORT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    dest = Path(cfg['artifact_dir'])
    if dest.exists() or REPORT_DIR.exists():
        raise ValueError('Immutable diagnostic output already exists')
    assert bresult['sha256']['config'] == sha(BCONFIG)
    assert bresult['sha256']['parent_manifest'] == sha(MANIFEST)
    assert cfg['parent_experiment_id'] == bresult['experiment_id']
    paths = {split: Path(manifest['dataset_paths'][split]) for split in ('train', 'dev')}
    for split, path in paths.items():
        assert sha(path) == manifest['sha256'][split+'_full_games.parquet']
    ppath = Path(bcfg['artifact_dir'])/'predictions.npz'
    assert sha(ppath) == bresult['sha256']['predictions']
    a_path = paths['dev'].parent/'predictions.npz'
    assert sha(a_path) == bresult['sha256']['parent_predictions']
    train, _, _ = cohort_pa_rows(pd.read_parquet(paths['train']), bcfg['train_cohort_pitchers'])
    bundle = Path(json.loads((ROOT/'configs/EXP-A-S0S1-001.json').read_text())['bundle_dir'])
    metadata = json.loads((bundle/'metadata.json').read_text())
    dev, _, _ = cohort_pa_rows(pd.read_parquet(paths['dev']), bcfg['train_cohort_pitchers'], metadata['pitchers'])
    train, dev = add_prior(train, bcfg), add_prior(dev, bcfg)
    with np.load(ppath) as b, np.load(a_path) as a:
        keys = dev[KEY].to_numpy(dtype=int)
        np.testing.assert_array_equal(keys, b['pitch_keys'])
        np.testing.assert_array_equal(b['pitch_keys'], a['pitch_keys'])
        np.testing.assert_array_equal(outcome_labels(dev), b['y'])
        np.testing.assert_array_equal(b['y'], a['y'])
        np.testing.assert_array_equal(b['count_only'], a['april_count'])
        np.testing.assert_array_equal(dev.previous_family.to_numpy(dtype='U16'), b['previous_family'])
        y, p0, p1 = b['y'].copy(), b['count_only'].copy(), b['count_plus_prior_family'].copy()
    for p in (p0, p1):
        assert p.shape == (len(dev), len(OUTCOMES)) and np.isfinite(p).all()
        assert (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-10)
    row = dev[['game_pk', 'pitcher', 'balls', 'strikes', 'pitch_number', 'previous_family']].copy()
    row['outcome'] = [OUTCOMES[i] for i in y]
    row['position'] = np.where(dev.pitch_number.eq(1), 'first', 'later')
    row['loss0'], row['loss1'] = loss(y, p0), loss(y, p1)
    row['delta'] = row.loss1-row.loss0
    known = bresult['tiers']['S1']
    np.testing.assert_allclose([row.loss0.mean(), row.loss1.mean(), row.delta.mean()],
        [known['count_only']['log_loss'], known['count_plus_prior_family']['log_loss'],
         known['plus_minus_only']['mean_log_loss_delta']], rtol=0, atol=1e-12)
    first = dev.pitch_number.eq(1)
    identity = {'train_first_pitches': int(train.pitch_number.eq(1).sum()),
                'dev_first_pitches': int(first.sum()),
                'train_none_iff_first': bool(train.previous_family.eq('NONE').equals(train.pitch_number.eq(1))),
                'dev_none_iff_first': bool(dev.previous_family.eq('NONE').equals(first)),
                'train_first_count_zero_zero': int((train.loc[train.pitch_number.eq(1), ['balls', 'strikes']] == 0).all(axis=1).sum()),
                'dev_first_count_zero_zero': int((dev.loc[first, ['balls', 'strikes']] == 0).all(axis=1).sum()),
                'train_zero_zero_nonfirst': int(((train.balls == 0) & (train.strikes == 0) & ~train.pitch_number.eq(1)).sum()),
                'dev_zero_zero_nonfirst': int(((dev.balls == 0) & (dev.strikes == 0) & ~first).sum())}
    assert identity['train_none_iff_first'] and identity['dev_none_iff_first']
    # Frozen estimator only. Every April held-out game is scored once; no choice among models.
    held = []
    for game in sorted(train.game_pk.unique()):
        fit, test = train.loc[train.game_pk.ne(game)], train.loc[train.game_pk.eq(game)]
        base = CountBaseline().fit(fit)
        ext = FamilyCountBaseline(bcfg['context_cell_count_baseline_pseudocount']).fit(fit)
        yy = outcome_labels(test)
        ll0 = loss(yy, legal_baseline(base.predict(test), test))
        ll1 = loss(yy, legal_baseline(ext.predict(test), test))
        held.append({'game_pk': int(game), 'pitches': int(len(test)), 'base_loss_sum': float(ll0.sum()),
                     'candidate_loss_sum': float(ll1.sum()), 'mean_delta': float((ll1-ll0).mean())})
    pooled_n = sum(item['pitches'] for item in held)
    train_holdout = {'games': held, 'pitches': pooled_n,
                     'baseline_nll': sum(item['base_loss_sum'] for item in held)/pooled_n,
                     'candidate_nll': sum(item['candidate_loss_sum'] for item in held)/pooled_n,
                     'mean_delta': sum(item['candidate_loss_sum']-item['base_loss_sum'] for item in held)/pooled_n}
    result = {'experiment_id': cfg['experiment_id'], 'phase': 'posthoc_diagnostic',
              'known_dev_delta_before_plan': known['plus_minus_only']['mean_log_loss_delta'],
              'dev_pitches': len(dev), 'dev_baseline_nll': float(row.loss0.mean()),
              'dev_candidate_nll': float(row.loss1.mean()), 'dev_mean_delta': float(row.delta.mean()),
              'support': support(train, dev), 'first_pitch_identity': identity,
              'dev_decomposition': {name: contribution(row, cols) for name, cols in {
                  'family': ['previous_family'], 'count': ['balls', 'strikes'],
                  'outcome': ['outcome'], 'game': ['game_pk'],
                  'first_vs_later': ['position']}.items()},
              'train_leave_one_game_out': train_holdout,
              'sha256': {'config': sha(CONFIG), 'script': sha(__file__), 'b_config': sha(BCONFIG),
                         'b_report': sha(BREPORT), 'a_selection': sha(MANIFEST),
                         'train_parquet': sha(paths['train']), 'dev_parquet': sha(paths['dev']),
                         'a_predictions': sha(a_path), 'b_predictions': sha(ppath)},
              'elapsed_seconds': time.perf_counter()-started,
              'interpretation_limit': 'Descriptive posthoc decomposition and April held-game evidence only; no causal attribution or DEV model selection.'}
    dest.mkdir(parents=True)
    REPORT_DIR.mkdir(parents=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    (dest/'results.json').write_text(payload)
    (REPORT_DIR/'results.json').write_text(payload)
    print(json.dumps({'result': str(REPORT_DIR/'results.json'), 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    run()

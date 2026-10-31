"""Secondary post-hoc DP legality conditioning of immutable saved predictions.

No fit, calibration, model inference, or DEV selection. DP is impossible with
two outs or no baserunners; remove its mass and condition on the other nine
classes. This mechanical constraint guarantees non-increasing log loss when
the excluded event is absent; it is not evidence of new learned information.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from audit_sequence_robustness import independent_bootstrap

KEY = ['game_pk', 'at_bat_number', 'pitch_number']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def condition_on_legality(p, impossible):
    p = np.asarray(p, dtype=float)
    impossible = np.asarray(impossible, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 10 or impossible.shape != (len(p),):
        raise ValueError('Expected ten-class probabilities and one legality flag per row')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid probability mass')
    constrained = p.copy()
    constrained[impossible, 9] = 0.
    remaining = constrained[impossible].sum(1)
    if (remaining <= 0).any():
        raise ValueError('Cannot condition rows assigning all probability to an impossible event')
    constrained[impossible] /= remaining[:, None]
    return constrained


def scores(y, p):
    return {'log_loss': -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)),
            'brier_multiclass': ((p-np.eye(10)[y])**2).sum(1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--ablations', type=Path)
    parser.add_argument('--robustness', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.run.resolve()
    root = base.parent.parent
    if not Path('/Volumes/T7 Shield').is_mount() or not base.is_relative_to(Path('/Volumes/T7 Shield').resolve()):
        raise SystemExit('Use the existing mounted SSD run')
    with np.load(base/'heldout_predictions.npz', allow_pickle=False) as saved:
        y, keys, games = saved['y'].copy(), saved['pitch_keys'].copy(), saved['game_pk'].copy()
    quality_path = root/'reports/data_quality.json'
    quality = read(quality_path)
    processed = root/'processed/pitches.parquet'
    assert sha(processed) == quality['processed_sha256']
    frame = pd.read_parquet(processed, columns=KEY+['game_date', 'outs_when_up', 'bases'])
    assert not frame.duplicated(KEY).any()
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all()
    states = frame.set_index(KEY).reindex(pd.MultiIndex.from_arrays(keys.T, names=KEY))
    assert not states.isna().any().any(), 'Prediction pitch keys must map to complete pre-pitch state'
    outs, bases = states.outs_when_up.to_numpy(int), states.bases.to_numpy(int)
    assert np.isin(outs, [0, 1, 2]).all() and np.isin(bases, np.arange(8)).all()
    impossible = (outs == 2) | (bases == 0)
    contradictions = impossible & (y == 9)
    if contradictions.any():
        raise ValueError('Eligible DP label contradicts proposed constraint; investigate these keys before projection: '+str(keys[contradictions].tolist()))
    output = (args.output or base/datetime.now(timezone.utc).strftime('legality-diagnostic-%Y%m%dT%H%M%SZ')).resolve()
    if not output.is_relative_to(root) or output == base or base.is_relative_to(output):
        raise ValueError('Use a new diagnostic directory inside the existing SSD artifact root')
    output.mkdir(parents=True, exist_ok=False)
    references = {str(quality_path): sha(quality_path), str(base/'heldout_predictions.npz'): sha(base/'heldout_predictions.npz')}
    sources = {}
    for path in [Path(__file__).resolve(), Path(__file__).with_name('audit_sequence_robustness.py').resolve()]:
        sources[path.name] = sha(path)
        shutil.copyfile(path, output/path.name)
    entries = []
    for variant, key in [('current_only', 'current_only'), ('flatten_mlp', 'flatten_mlp'),
                         ('full_transformer', 'transformer'), ('count_baseline', 'count_hand')]:
        entries.append(('base/'+variant, 42, base/'heldout_predictions.npz', key))
    if args.ablations:
        ablations = args.ablations.resolve()
        for variant in read(ablations/'config.json')['variants']:
            if variant in read(ablations/'ablation_results.json')['variants']:
                entries.append(('ablations/'+variant, 42, ablations/'heldout_predictions.npz', variant))
        references[str(ablations/'config.json')] = sha(ablations/'config.json')
    if args.robustness:
        robustness = args.robustness.resolve()
        config = read(robustness/'config.json')
        references[str(robustness/'config.json')] = sha(robustness/'config.json')
        for variant in config['variants']:
            for seed in config['model_seeds']:
                folder = robustness/'models'/f'seed{seed}'/variant
                if not (folder/'result.json').exists():
                    continue
                result = read(folder/'result.json')
                assert result['seed'] == seed and result['variant'] == variant
                assert sha(folder/'predictions.npz') == result['artifact_hashes']['predictions.npz']
                references[str(folder/'result.json')] = sha(folder/'result.json')
                entries.append((f'robustness/{variant}/seed{seed}', seed, folder/'predictions.npz', 'delivery_integrated_calibrated'))
    for _, _, path, _ in entries:
        references[str(path)] = sha(path)
    config = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'base_run': str(base),
              'model_entries': [{'name': name, 'seed': seed, 'source': str(path), 'array': key} for name, seed, path, key in entries],
              'constraint': 'if outs_when_up==2 OR bases==0: P(double_play)=0; divide all other nine probabilities by their total mass',
              'source_hashes': sources, 'reference_hashes': references, 'processed_sha256': quality['processed_sha256'],
              'scope': 'Secondary post-hoc common-rule diagnostic on inspected DEV; no fit, calibration, model inference, or model selection.',
              'interpretation': 'Valid impossible-event conditioning mechanically improves log loss; not new learned predictive skill. Brier change is empirical.',
              'probability_regime': 'Original saved calibrated delivery-integrated forecasting probabilities only'}
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    report = {'status': 'complete', 'n': len(y), 'games': len(np.unique(games)),
              'constrained_rows': int(impossible.sum()), 'outs_two_rows': int((outs == 2).sum()),
              'empty_bases_rows': int((bases == 0).sum()), 'dp_label_count': int((y == 9).sum()),
              'impossible_dp_labels': 0, 'models': {}, 'comparisons': {}, 'scope': config['scope'], 'interpretation': config['interpretation']}
    if args.robustness:
        expected_config = read(args.robustness.resolve()/'config.json')
        coverage = {variant: [seed for seed in expected_config['model_seeds']
                              if any(name == f'robustness/{variant}/seed{seed}' for name, _, _, _ in entries)]
                    for variant in expected_config['variants']}
        complete = all(seeds == expected_config['model_seeds'] for seeds in coverage.values())
        report['robustness_coverage'] = {'expected_seeds': expected_config['model_seeds'],
                                        'completed_seeds_by_variant': coverage, 'all_models_complete': complete}
        report['status'] = 'complete_robustness_snapshot' if complete else 'partial_robustness_snapshot'
    measured, archives = {}, {}
    for name, seed, path, key in entries:
        with np.load(path, allow_pickle=False) as saved:
            for identifier, expected in [('y', y), ('pitch_keys', keys), ('game_pk', games)]:
                np.testing.assert_array_equal(saved[identifier], expected)
            raw = saved[key].astype(float)
        constrained = condition_on_legality(raw, impossible)
        raw_scores, constrained_scores = scores(y, raw), scores(y, constrained)
        assert (constrained_scores['log_loss'] <= raw_scores['log_loss']+1e-7).all()
        np.testing.assert_array_equal(constrained[~impossible], raw[~impossible])
        measured[name] = {'raw': raw_scores, 'constrained': constrained_scores}
        report['models'][name] = {'seed': seed,
            'raw': {metric: float(values.mean()) for metric, values in raw_scores.items()},
            'constrained': {metric: float(values.mean()) for metric, values in constrained_scores.items()},
            'removed_dp_mass_mean_all_rows': float(np.where(impossible, raw[:, 9], 0).mean()),
            'removed_dp_mass_mean_constrained_rows': float(raw[impossible, 9].mean()),
            'constrained_minus_raw': independent_bootstrap(
                {metric: value[None] for metric, value in constrained_scores.items()},
                {metric: value[None] for metric, value in raw_scores.items()}, games, [seed])}
        safe_name = name.replace('/', '__')
        archives[safe_name+'__raw'], archives[safe_name+'__constrained'] = raw, constrained

    def compare(name, pairs, seeds):
        result = {'matched_seeds': seeds, 'models': pairs}
        if name.startswith('robustness_'):
            result['complete_five_seed_comparison'] = seeds == [42, 43, 44, 45, 46]
        for regime in ['raw', 'constrained']:
            left = {metric: np.stack([measured[a][regime][metric] for a, _ in pairs]) for metric in ['log_loss', 'brier_multiclass']}
            right = {metric: np.stack([measured[b][regime][metric] for _, b in pairs]) for metric in left}
            result[regime] = independent_bootstrap(left, right, games, seeds)
        report['comparisons'][name] = result

    compare('base_full_minus_flatten', [('base/full_transformer', 'base/flatten_mlp')], [42])
    compare('base_full_minus_count_baseline', [('base/full_transformer', 'base/count_baseline')], [42])
    for variant in ['no_game_context', 'no_batter_style']:
        if 'ablations/'+variant in measured:
            compare('seed42_'+variant+'_minus_full', [('ablations/'+variant, 'base/full_transformer')], [42])
    if args.robustness:
        for variant, reference in [('no_game_context', 'full_transformer'), ('no_batter_style', 'full_transformer'),
                                   ('full_transformer', 'flatten_mlp'), ('full_transformer', 'capacity_mlp'),
                                   ('no_clusters', 'full_transformer'), ('full_transformer', 'count_baseline')]:
            pairs, selected = [], []
            for seed in [42, 43, 44, 45, 46]:
                left = f'robustness/{variant}/seed{seed}'
                right = 'base/count_baseline' if reference == 'count_baseline' else f'robustness/{reference}/seed{seed}'
                if left in measured and right in measured:
                    pairs.append((left, right)); selected.append(seed)
            if pairs:
                compare('robustness_'+variant+'_minus_'+reference, pairs, selected)
    np.savez_compressed(output/'legality_predictions.npz', y=y, pitch_keys=keys, game_pk=games,
                        outs_when_up=outs, bases=bases, impossible_dp=impossible, **archives)
    report['keys_sha256'] = hashlib.sha256(keys.astype(np.int64).tobytes()).hexdigest()
    report['predictions_sha256'] = sha(output/'legality_predictions.npz')
    assert {path: sha(Path(path)) for path in references} == references
    assert {name: sha(Path(__file__).with_name(name)) for name in sources} == sources
    (output/'legality_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'models': len(entries),
                      'constrained_rows': int(impossible.sum()), 'impossible_dp_labels': 0}, indent=2))


if __name__ == '__main__':
    main()

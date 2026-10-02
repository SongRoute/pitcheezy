"""Frozen B-PAHISTORY-002 TRAIN selection and separately gated July DEV replay."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'experiments/pitchmdp'
sys.path[:0] = [str(ROOT / 'scripts'), str(PROJECT / 'scripts'), str(PROJECT)]

import numpy as np
import pandas as pd
from pitchmdp.data import KEY
from pitchmdp.model import CountBaseline, OUTCOMES, outcome_labels
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, TYPE_KEYS
from a_small_eval import cohort_pa_rows, legal_baseline, paired_bootstrap, score_metrics

CONFIG = ROOT / 'configs/EXP-B-PAHISTORY-002.json'
REPORT = ROOT / 'results/EXP-B-PAHISTORY-002'
FAMILY_CONFIG = ROOT / 'configs/EXP-B-PAHISTORY-001.json'
CONTRACT = ROOT / 'docs/contracts/B-next-experiment-v2.md'
HELPERS = (ROOT / 'scripts/a_small_eval.py', ROOT / 'scripts/b_pa_history_eval.py',
           PROJECT / 'scripts/run_sequence_frequency_baselines.py',
           PROJECT / 'scripts/diagnose_sequence_legality.py', PROJECT / 'pitchmdp/model.py')
CONTEXT_KEYS = [*TYPE_KEYS, 'previous_family']
PA_KEY = ['game_pk', 'at_bat_number']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def git_bytes(relative):
    return subprocess.check_output(['git', 'show', 'HEAD:' + str(relative)], cwd=ROOT)


def require_committed(path):
    relative = Path(path).relative_to(ROOT)
    if git_bytes(relative) != Path(path).read_bytes():
        raise ValueError(f'Uncommitted or changed prerequisite: {relative}')


def family(name, family_cfg):
    for label, names in family_cfg['families'].items():
        if name in names:
            return label
    return family_cfg['unknown_family']


def add_prior(frame, family_cfg):
    ordered = frame.sort_values(KEY, kind='stable').copy().reset_index(drop=True)
    if ordered.duplicated(KEY).any() or ordered[KEY].isna().any().any():
        raise ValueError('Duplicate or missing pitch key')
    previous = ordered.groupby(PA_KEY, sort=False).pitch_type.shift(1)
    ordered['previous_family'] = previous.map(lambda name: family(name, family_cfg) if pd.notna(name) else 'NONE')
    if not ordered.loc[ordered.pitch_number.eq(1), 'previous_family'].eq('NONE').all():
        raise ValueError('Prior pitch crossed PA boundary')
    for _, pa in ordered.groupby(PA_KEY, sort=False):
        if pa.pitch_number.to_numpy(dtype=int).tolist() != list(range(1, len(pa) + 1)):
            raise ValueError('Incomplete or gapped PA pitch sequence')
    return ordered


class PAHistoryModel:
    def __init__(self, alpha, pseudocount=50):
        self.alpha = float(alpha)
        self.pseudocount = int(pseudocount)

    def fit(self, train):
        if train.empty or not train.split.eq('train').all():
            raise ValueError('Only nonempty TRAIN rows may fit the model')
        self.parent = HierarchicalFrequencyBaseline(False).fit(train)
        labels = outcome_labels(train)
        if (labels < 0).any():
            raise ValueError('Invalid TRAIN label')
        work = train[CONTEXT_KEYS].copy()
        work['y'] = labels
        self.context = {}
        for key, group in work.loc[work.previous_family.ne('NONE')].groupby(CONTEXT_KEYS, sort=True):
            parent = self.parent.predict(pd.DataFrame([dict(zip(TYPE_KEYS, key[:-1]))]))[0]
            counts = np.bincount(group.y, minlength=len(OUTCOMES))
            self.context[key] = (counts + self.pseudocount * parent) / (len(group) + self.pseudocount)
        return self

    def predict_with_origin(self, frame):
        parent, parent_origin = self.parent.predict_with_origin(frame)
        result = parent.copy()
        context_seen = np.zeros(len(frame), dtype=bool)
        if self.alpha:
            for i, key in enumerate(frame[CONTEXT_KEYS].itertuples(index=False, name=None)):
                if key[-1] != 'NONE' and key in self.context:
                    result[i] = (1 - self.alpha) * parent[i] + self.alpha * self.context[key]
                    context_seen[i] = True
        return result, parent, parent_origin, context_seen

    def predict(self, frame):
        return self.predict_with_origin(frame)[0]


def checkpoint(model):
    def records(table):
        return [{'key': list(k), 'probability': np.asarray(v, float).tolist()} for k, v in sorted(table.items())]
    return {'format': 'PAHistoryModel-v1', 'alpha': model.alpha, 'pseudocount': model.pseudocount,
            'global_probability': model.parent.parent.global_p.tolist(),
            'count_table': records(model.parent.parent.table),
            'type_table': [
                {'key': list(k), 'probability': row.to_numpy(dtype=float).tolist()}
                for k, row in model.parent.type_table.iterrows()],
            'context_table': records(model.context)}


def restore(data):
    if data['format'] != 'PAHistoryModel-v1':
        raise ValueError('Unknown checkpoint format')
    model = PAHistoryModel(data['alpha'], data['pseudocount'])
    parent = HierarchicalFrequencyBaseline(False)
    parent.parent = CountBaseline()
    parent.parent.global_p = np.asarray(data['global_probability'], float)
    parent.parent.table = {tuple(row['key']): np.asarray(row['probability'], float) for row in data['count_table']}
    index = pd.MultiIndex.from_tuples([tuple(row['key']) for row in data['type_table']], names=TYPE_KEYS)
    parent.type_table = pd.DataFrame([row['probability'] for row in data['type_table']], index=index, columns=range(10))
    model.parent = parent
    model.context = {tuple(row['key']): np.asarray(row['probability'], float) for row in data['context_table']}
    return model


def validate_folds(train, cfg):
    manifest = {int(g['game_pk']): pd.Timestamp(g['game_date']) for group in cfg['selection']['train'].values() for g in group}
    if set(train.game_pk.unique()) != set(manifest):
        raise ValueError('TRAIN games differ from frozen selection')
    if train.groupby(PA_KEY).game_pk.nunique().gt(1).any():
        raise ValueError('PA spans games')
    for spec in cfg['training_folds']:
        fit, val = set(spec['fit_game_ids']), set(spec['validation_game_ids'])
        if not fit or not val or fit & val or not fit | val <= set(manifest):
            raise ValueError('Invalid forward fold game IDs')
        if max(manifest[g] for g in fit) >= min(manifest[g] for g in val):
            raise ValueError('Fold is not forward in time')
        for ids in (fit, val):
            if train.loc[train.game_pk.isin(ids)].empty:
                raise ValueError('Empty forward fold block')
            actual = pd.to_datetime(train.loc[train.game_pk.isin(ids), 'game_date'])
            if not all(actual.eq(train.loc[train.game_pk.isin(ids), 'game_pk'].map(manifest))):
                raise ValueError('TRAIN date differs from manifest')


def choose_alpha(blocks, grid):
    """Blocks contain equal-row legal probabilities for each candidate alpha."""
    scores = {}
    for alpha in grid:
        losses = []
        for block in blocks:
            y, p = block['y'], block['predictions'][alpha]
            losses.append(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
        scores[alpha] = {'blocks': [float(x.mean()) for x in losses],
                         'pooled': float(np.concatenate(losses).mean())}
    minimum = min(scores[a]['pooled'] for a in grid)
    best = min(a for a in grid if scores[a]['pooled'] <= minimum + 1e-10)
    base = scores[0.0]
    adopt = best != 0 and scores[best]['pooled'] < base['pooled'] and all(
        x <= y for x, y in zip(scores[best]['blocks'], base['blocks']))
    return (best if adopt else 0.0), best, scores


def common(cfg):
    if cfg['experiment_id'] != 'EXP-B-PAHISTORY-002' or cfg['alpha_grid'] != [0.0, .25, .5, .75, 1.0] or cfg['context_parent_pseudocount'] != 50:
        raise ValueError('Frozen experiment configuration changed')
    manifest_path = ROOT / cfg['parent_selection_manifest']
    manifest = json.loads(manifest_path.read_text())
    if manifest['experiment_id'] != 'EXP-A-S0S1-001' or cfg['pitchers'] != [657277, 554430]:
        raise ValueError('Parent selection mismatch')
    family_cfg = json.loads(FAMILY_CONFIG.read_text())
    return manifest, family_cfg


def train_phase():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text())
    for path in (CONFIG, Path(__file__), CONTRACT, FAMILY_CONFIG, ROOT / cfg['parent_selection_manifest'],
                 ROOT / cfg['parent_common_comparison'], *HELPERS):
        require_committed(path)
    manifest, family_cfg = common(cfg)
    artifact = Path(cfg['artifact_dir'])
    if artifact.exists() or (REPORT / 'train_selection.json').exists() or (REPORT / 'checkpoint.json').exists():
        raise ValueError('Immutable TRAIN output already exists')
    train_path = Path(manifest['dataset_paths']['train'])
    if digest(train_path) != manifest['sha256']['train_full_games.parquet']:
        raise ValueError('TRAIN data hash mismatch')
    full = pd.read_parquet(train_path)
    train, reasons, cohort_pas = cohort_pa_rows(full, cfg['pitchers'])
    train = add_prior(train, family_cfg)
    if train.empty or set(train.game_pk) != {g['game_pk'] for group in manifest['selection']['train'].values() for g in group}:
        raise ValueError('TRAIN support unexpectedly changed')
    if (len(train), int(train.groupby(PA_KEY).ngroups)) != (578, 145):
        raise ValueError('TRAIN support differs from frozen common comparison')
    validate_folds(train, {**cfg, 'selection': manifest['selection']})
    blocks, archive, fold_support = [], {}, []
    for i, spec in enumerate(cfg['training_folds'], 1):
        fit = train.loc[train.game_pk.isin(spec['fit_game_ids'])].copy()
        val = train.loc[train.game_pk.isin(spec['validation_game_ids'])].copy()
        model = PAHistoryModel(1, cfg['context_parent_pseudocount']).fit(fit)
        child, parent, _, seen = model.predict_with_origin(val)
        y = outcome_labels(val)
        if (y < 0).any():
            raise ValueError('Invalid validation labels')
        predictions = {}
        for a in cfg['alpha_grid']:
            candidate = parent.copy()
            if a:
                candidate[seen] = (1-a)*parent[seen] + a*child[seen]
            predictions[a] = legal_baseline(candidate, val)
            np.testing.assert_array_equal(predictions[a][~seen], predictions[0.0][~seen] if 0.0 in predictions else legal_baseline(parent, val)[~seen])
        blocks.append({'y': y, 'predictions': predictions})
        archive[f'fold{i}_keys'] = val[KEY].to_numpy(dtype=int)
        archive[f'fold{i}_y'] = y
        archive[f'fold{i}_game_pk'] = val.game_pk.to_numpy(dtype=int)
        archive[f'fold{i}_parent'] = predictions[0.0]
        archive[f'fold{i}_context_seen'] = seen
        fold_support.append({'fit_pitches': len(fit), 'validation_pitches': len(val),
                             'fit_pas': int(fit.groupby(PA_KEY).ngroups),
                             'validation_pas': int(val.groupby(PA_KEY).ngroups),
                             'none': int(val.previous_family.eq('NONE').sum()),
                             'context_seen': int(seen.sum()),
                             'unseen_context': int((~seen & val.previous_family.ne('NONE').to_numpy()).sum())})
        for j, a in enumerate(cfg['alpha_grid']):
            archive[f'fold{i}_alpha{j}'] = predictions[a]
    selected, minimum, scores = choose_alpha(blocks, cfg['alpha_grid'])
    final = PAHistoryModel(selected, cfg['context_parent_pseudocount']).fit(train)
    restored = restore(checkpoint(final))
    np.testing.assert_array_equal(final.predict(train), restored.predict(train))
    artifact.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact / 'train_predictions.npz', **archive)
    write_json(REPORT / 'checkpoint.json', checkpoint(final))
    write_json(artifact / 'checkpoint.json', checkpoint(final))
    if digest(REPORT / 'checkpoint.json') != digest(artifact / 'checkpoint.json'):
        raise ValueError('Checkpoint copies differ')
    disk_restored = restore(json.loads((REPORT / 'checkpoint.json').read_text()))
    np.testing.assert_array_equal(final.predict(train), disk_restored.predict(train))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = {'experiment_id': cfg['experiment_id'], 'phase': 'train_selected', 'selected_alpha': selected,
              'minimum_nll_alpha': minimum, 'gate': 'passed' if selected else 'rejected',
              'forward_nll': {str(a): scores[a] for a in cfg['alpha_grid']},
              'folds': cfg['training_folds'], 'fold_support': fold_support,
              'train_support': {'pitches': len(train), 'pas': train.groupby(PA_KEY).ngroups,
                                'cohort_pas': cohort_pas, 'excluded_pa_reasons': reasons,
                                'context_cells': len(final.context)},
              'artifact_dir': str(artifact), 'spec_git_commit': head, 'seed': cfg['seed'],
              'elapsed_seconds': time.perf_counter() - started,
              'maxrss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'sha256': {'config': digest(CONFIG), 'script': digest(__file__), 'contract': digest(CONTRACT),
                         'family_config': digest(FAMILY_CONFIG), 'parent_manifest': digest(ROOT / cfg['parent_selection_manifest']),
                         'common_config': digest(ROOT / cfg['parent_common_comparison']),
                         'train_parquet': digest(train_path), 'checkpoint': digest(REPORT / 'checkpoint.json'),
                         'train_predictions': digest(artifact / 'train_predictions.npz'),
                         'helpers': {str(path.relative_to(ROOT)): digest(path) for path in HELPERS}}}
    write_json(REPORT / 'train_selection.json', result)
    write_json(artifact / 'train_selection.json', result)
    print(json.dumps({'selected_alpha': selected, 'gate': result['gate'], 'report': str(REPORT / 'train_selection.json')}))


def dev_gate(cfg):
    selection_path, checkpoint_path = REPORT / 'train_selection.json', REPORT / 'checkpoint.json'
    for path in (CONFIG, Path(__file__), CONTRACT, FAMILY_CONFIG, ROOT / cfg['parent_selection_manifest'],
                 ROOT / cfg['parent_common_comparison'], *HELPERS, selection_path, checkpoint_path):
        require_committed(path)
    selection = json.loads(selection_path.read_text())
    if selection['experiment_id'] != cfg['experiment_id'] or selection['phase'] != 'train_selected':
        raise ValueError('Invalid committed TRAIN selection')
    if selection['selected_alpha'] == 0 or selection['gate'] != 'passed':
        raise ValueError('TRAIN rejected history: no DEV evaluation permitted')
    for key, path in [('config', CONFIG), ('script', Path(__file__)), ('contract', CONTRACT),
                      ('family_config', FAMILY_CONFIG), ('parent_manifest', ROOT / cfg['parent_selection_manifest']),
                      ('common_config', ROOT / cfg['parent_common_comparison']),
                      ('train_parquet', Path(json.loads((ROOT / cfg['parent_selection_manifest']).read_text())['dataset_paths']['train'])),
                      ('checkpoint', checkpoint_path),
                      ('train_predictions', Path(cfg['artifact_dir']) / 'train_predictions.npz')]:
        if digest(path) != selection['sha256'][key]:
            raise ValueError(f'Committed TRAIN identity changed: {key}')
    for path in HELPERS:
        if digest(path) != selection['sha256']['helpers'][str(path.relative_to(ROOT))]:
            raise ValueError(f'Committed TRAIN helper changed: {path}')
    model = restore(json.loads(checkpoint_path.read_text()))
    if model.alpha != selection['selected_alpha']:
        raise ValueError('Checkpoint alpha differs from committed selection')
    return selection, model


def dev_phase():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text())
    selection, model = dev_gate(cfg)  # No DEV data or predictions are opened above this line.
    manifest, family_cfg = common(cfg)
    artifact = Path(cfg['artifact_dir'])
    if any(path.exists() for path in (artifact / 'dev_results.json', artifact / 'dev_predictions.npz', REPORT / 'dev_results.json')):
        raise ValueError('Immutable DEV output already exists')
    dev_path = Path(manifest['dataset_paths']['dev'])
    if digest(dev_path) != manifest['sha256']['dev_full_games.parquet']:
        raise ValueError('DEV data hash mismatch')
    bundle = Path(json.loads((ROOT / cfg['parent_common_comparison']).read_text())['bundle_dir'])
    if digest(bundle / 'metadata.json') != manifest['sha256']['bundle_metadata']:
        raise ValueError('Frozen repertoire metadata changed')
    metadata = json.loads((bundle / 'metadata.json').read_text())
    dev_full = pd.read_parquet(dev_path)
    dev, reasons, cohort_pas = cohort_pa_rows(dev_full, cfg['pitchers'], metadata['pitchers'])
    dev = add_prior(dev, family_cfg)
    y = outcome_labels(dev)
    if (y < 0).any():
        raise ValueError('Invalid DEV labels')
    if (len(dev), int(dev.groupby(PA_KEY).ngroups)) != (563, 151):
        raise ValueError('DEV support differs from frozen common comparison')
    parent_raw = model.parent.predict(dev)
    child_raw, _, parent_origin, context_seen = model.predict_with_origin(dev)
    parent, child = legal_baseline(parent_raw, dev), legal_baseline(child_raw, dev)
    np.testing.assert_array_equal(parent[~context_seen], child[~context_seen])
    common_result = json.loads((ROOT / 'results/EXP-A-COMMON-001/results.json').read_text())
    common_predictions = Path(json.loads((ROOT / cfg['parent_common_comparison']).read_text())['artifact_dir']) / 'predictions.npz'
    if digest(common_predictions) != common_result['sha256']['predictions']:
        raise ValueError('Common comparison predictions changed')
    with np.load(common_predictions, allow_pickle=False) as saved:
        np.testing.assert_array_equal(dev[KEY].to_numpy(dtype=int), saved['pitch_keys'])
        np.testing.assert_array_equal(y, saved['y'])
        np.testing.assert_array_equal(parent, saved['known_type'])
    games = dev.game_pk.to_numpy(dtype=int)
    slices = {'all': np.ones(len(dev), dtype=bool), 'two_strikes': dev.strikes.to_numpy() == 2}
    result = {'experiment_id': cfg['experiment_id'], 'phase': 'evaluated_exploratory_dev',
              'selected_alpha': model.alpha, 'train_selection_sha256': digest(REPORT / 'train_selection.json'),
              'scores': {}, 'support': {'pitches': len(dev), 'pas': dev.groupby(PA_KEY).ngroups,
                                        'cohort_pas': cohort_pas, 'excluded_pa_reasons': reasons},
              'policy_value': None, 'adoption': 'none; exploratory known-type prediction screening only',
              'exposure': 'Already-exposed July 2025 DEV; not confirmatory',
              'artifact_dir': str(artifact),
              'train_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    for name, mask in slices.items():
        result['scores'][name] = {'pitches': int(mask.sum()), 'games': int(np.unique(games[mask]).size),
                                  'parent': score_metrics(y[mask], parent[mask]),
                                  'with_history': score_metrics(y[mask], child[mask]),
                                  'paired_nll': paired_bootstrap(y[mask], child[mask], parent[mask], games[mask],
                                                                  cfg['seed'], cfg['bootstrap_replicates']),
                                  'context_seen': int(context_seen[mask].sum()),
                                  'none': int((dev.previous_family.to_numpy()[mask] == 'NONE').sum()),
                                  'unseen_context': int((~context_seen[mask] & (dev.previous_family.to_numpy()[mask] != 'NONE')).sum()),
                                  'fallback': int((~context_seen[mask]).sum()),
                                  'parent_origin': {str(k): int(v) for k, v in sorted(Counter(parent_origin[mask]).items())}}
    np.savez_compressed(artifact / 'dev_predictions.npz', pitch_keys=dev[KEY].to_numpy(dtype=int), game_pk=games,
                        y=y, parent=parent, with_history=child, previous_family=dev.previous_family.to_numpy(dtype='U16'),
                        context_seen=context_seen, parent_origin=parent_origin)
    result['sha256'] = {'config': digest(CONFIG), 'script': digest(__file__), 'checkpoint': digest(REPORT / 'checkpoint.json'),
                        'dev_parquet': digest(dev_path), 'predictions': digest(artifact / 'dev_predictions.npz')}
    result['elapsed_seconds'] = time.perf_counter() - started
    result['maxrss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(artifact / 'dev_results.json', result)
    write_json(REPORT / 'dev_results.json', result)
    print(json.dumps({'phase': result['phase'], 'scores': {k: v['paired_nll'] for k, v in result['scores'].items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'dev'))
    args = parser.parse_args()
    if args.phase == 'train':
        train_phase()
    else:
        dev_phase()


if __name__ == '__main__':
    main()

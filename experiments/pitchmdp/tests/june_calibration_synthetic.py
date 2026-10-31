"""Synthetic miniature of every ML-JUNE-CALIBRATION-v1 parent; no real data is read.

The builder creates a verified processed/physics cache, a pickled auxiliary
bundle, five tiny G0-style checkpoints and the P4/P10/P11/D81 archives with the
same file layout and hash chain as the registered parents.  Archived Cpanel and
DEV member predictions are computed from the FULL-season store, so a runner that
truncates the store at June 30 must reproduce them through the frozen path.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / 'scripts') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'scripts'))

from pitchmdp import matrix_june_calibration as mj  # noqa: E402
from pitchmdp.archetypes import add_batter_style_history  # noqa: E402
from pitchmdp.data import KEY, RAW_ALLOWLIST  # noqa: E402
from pitchmdp.matrix_benchmark import predict_streamed  # noqa: E402
from pitchmdp.matrix_data import load_verified_processed_cache  # noqa: E402
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402
from pitchmdp.matrix_models import MatrixModel  # noqa: E402
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor, training_arrays  # noqa: E402
from pitchmdp.model import eligible, outcome_labels  # noqa: E402
from pitchmdp.sequence_data import CACHE_VERSION, SIDECAR_COLUMNS, PhysicalNormalizer  # noqa: E402
from pitchmdp.sequence_delivery import JointDelivery  # noqa: E402
from run_sequence_calibration import fit_blend  # noqa: E402
from run_sequence_frequency_baselines import temperature_predictions  # noqa: E402
from run_temporal_blend import assign_fold  # noqa: E402

REPO = PROJECT.parents[1]
SCIENTIFIC = REPO / 'configs' / 'ML-JUNE-CALIBRATION-v1.json'
DRAWS, PROFILE_ROWS, PROBE_ROWS, WIDTH, MAY_ROWS = 6, 256, 16, 8, 7
TEMPERATURE = 1.03
GROUP_PITCHERS = {'low': (11, 21), 'middle': (12, 22), 'high': (13, 23), 'zero': (14, 24)}
TRAIN_PLAYERS = {11: 100, 21: 150, 12: 900, 22: 1000, 13: 3000, 23: 2000}
PANEL_IDS = (11, 12, 13)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, default=str))
    return sha(path)


def savez(path, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return sha(path)


class StubContext:
    """Frozen-context stand-in: row-local current state plus a prior-date style feature."""

    def transform(self, frame):
        return np.column_stack([frame.balls.to_numpy(float) / 3, frame.strikes.to_numpy(float) / 2,
                                frame['batter_style_contact_prior'].to_numpy(float),
                                frame.stand.eq('L').to_numpy(float)]).astype(np.float32)

    def report(self):
        return {'stub': True}


class StubFrequency:
    """Frozen TRAIN frequency stand-in; a pure row-local lookup."""

    def predict(self, frame):
        base = np.ones((len(frame), 10))
        base[:, 0] += frame.balls.to_numpy(float)
        base[:, 1] += frame.strikes.to_numpy(float)
        base[:, 3] += frame.pitch_type.eq('SL').to_numpy(float)
        base[:, 2] += 0.5 * frame.outs_when_up.to_numpy(float)
        return base / base.sum(axis=1, keepdims=True)


def schedule():
    games = [('2024-03-10', 'S')]
    games += [(f'2024-06-{d:02d}', 'R') for d in range(1, 13)]
    games += [('2025-05-20', 'R'), ('2025-05-21', 'R')]
    games += [(f'2025-06-{d:02d}', 'R') for d in range(1, 31)] + [(f'2025-06-{d:02d}', 'R') for d in range(1, 11)]
    games += [(f'2025-07-{d:02d}', 'R') for d in range(1, 32)] + [(f'2025-08-{d:02d}', 'R') for d in range(1, 10)]
    return sorted(games)


def make_pitches(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for number, (date, game_type) in enumerate(schedule()):
        game_pk = 700000 + number
        pitchers = [GROUP_PITCHERS[g][number % 2] for g in ('low', 'middle', 'high', 'zero')]
        at_bat = 0
        for pitcher in pitchers:
            for _ in range(5):
                at_bat += 1
                batter = int(rng.integers(900, 930))
                stand = 'L' if rng.random() < .4 else 'R'
                for pitch in (1, 2, 3):
                    terminal = pitch == 3
                    description = 'hit_into_play' if terminal else str(rng.choice(['ball', 'called_strike', 'foul', 'swinging_strike']))
                    if rng.random() < .02:
                        description = 'mystery'  # unmapped outcome: ineligible, kept as history
                    rows.append({
                        'game_date': pd.Timestamp(date), 'game_pk': game_pk, 'at_bat_number': at_bat, 'pitch_number': pitch,
                        'game_type': game_type, 'pitch_type': str(rng.choice(['FF', 'SL', 'CH'])), 'pitcher': pitcher,
                        'batter': batter, 'stand': stand, 'p_throws': 'L' if pitcher in (11, 12) else 'R',
                        'balls': int(rng.integers(0, pitch)), 'strikes': int(rng.integers(0, min(pitch, 3))),
                        'outs_when_up': int(rng.integers(0, 3)), 'inning': int(rng.integers(1, 10)),
                        'inning_topbot': 'Top' if at_bat % 2 else 'Bot', 'home_score': int(rng.integers(0, 6)),
                        'away_score': int(rng.integers(0, 6)), 'bases': int(rng.integers(0, 8)),
                        'plate_x': float(rng.normal(0, .8)), 'plate_z': float(rng.normal(2.5, .6)),
                        'release_spin_rate': float(rng.normal(2300, 150)), 'pfx_x': float(rng.normal(0, .5)),
                        'pfx_z': float(rng.normal(1, .4)), 'effective_speed': float(rng.normal(92, 3)),
                        'spin_axis': float(rng.uniform(0, 360)),
                        'events': str(rng.choice(['single', 'field_out', 'double', 'grounded_into_double_play'])) if terminal else None,
                        'description': description, 'is_pa_terminal': terminal,
                        'launch_angle': float(rng.normal(10, 20)) if terminal else np.nan,
                        'starter_pitcher': pitchers[0], 'supported_pa': True, 'split': 'legacy'})
    frame = pd.DataFrame(rows)
    return frame.sort_values(['game_date', *KEY], ignore_index=True)


def full_season_frame(raw):
    """The original full-season G0 path (regular_frame): used only to build archived references."""
    regular = raw.loc[raw.game_type.eq('R')].copy().reset_index(drop=True)
    return assign_fold(add_batter_style_history(regular), 2025)


def build(root):
    root = Path(root)
    runs = root / 'runs' / 'ML-MATRIX-20260924'
    p4, p10, p11, p112, june_dir = (runs / name for name in ('EXP-P4-001', 'EXP-P10-001', 'EXP-P11-001', 'EXP-P11-002',
                                                             'ML-JUNE-ELIGIBILITY-v1'))
    for folder in (p4, p10, p11, p112, june_dir, root / 'processed', root / 'cache', root / 'reports'):
        folder.mkdir(parents=True, exist_ok=True)
    pitches = make_pitches()
    processed = root / 'processed' / 'pitches.parquet'
    pitches.drop(columns=list(SIDECAR_COLUMNS)).to_parquet(processed, index=False)
    cache = root / 'cache' / 'sequence_physics.parquet'
    pitches[KEY + list(SIDECAR_COLUMNS)].sample(frac=1, random_state=1).to_parquet(cache, index=False)
    sources = [{'file': name, 'bytes': 1, 'sha256': 'a' * 64} for name in RAW_ALLOWLIST]
    dump(root / 'reports' / 'data_quality.json', {'sources': sources, 'processed_sha256': sha(processed), 'rows': len(pitches)})
    dump(root / 'cache' / 'sequence_physics.json', {'identity': {
        'cache_version': CACHE_VERSION, 'sources': sources, 'processed_sha256': sha(processed), 'rows': len(pitches),
        'key': KEY, 'columns': list(SIDECAR_COLUMNS)}, 'sha256': sha(cache)})
    local = root / 'local.json'
    dump(local, {'artifact_root': str(root), 'raw_root': str(root / 'raw'), 'raw_allowlist': RAW_ALLOWLIST})
    raw = load_verified_processed_cache(json.loads(local.read_text()))
    frame = full_season_frame(raw)
    ok = np.asarray(eligible(frame), dtype=bool)
    split = frame.split.to_numpy()
    train_all = frame.loc[split == 'train']
    normalizer = PhysicalNormalizer().fit(train_all)
    vocabulary = sorted(train_all.pitch_type.dropna().unique())
    store = MatrixHistoryStore.from_frame(frame, normalizer=normalizer, history_length=5, type_vocabulary=vocabulary)
    tokens = store.report()
    train = frame.loc[(split == 'train') & ok]
    clusters = {'version': 'pitcher_clusters_v1', 'columns': ['c0', 'c1'],
                'pitcher_profiles': {str(p): [float(p % 3), float(p % 5)] for p in TRAIN_PLAYERS},
                'pitcher_counts': {str(p): n for p, n in TRAIN_PLAYERS.items()},
                'pitcher_cluster': {str(p): p % 4 for p in TRAIN_PLAYERS},
                'cluster_counts': {str(c): 100 for c in range(4)}}
    delivery = JointDelivery().fit(train, normalizer, draws=DRAWS, seed=0)
    aux = {'context': StubContext(), 'delivery': delivery, 'baseline': StubFrequency(), 'normalizer': normalizer}
    with (p4 / 'aux.pkl').open('wb') as stream:
        pickle.dump(aux, stream)
    context = SharingContext(aux['context'], clusters)

    june_mask = (split == 'blend') & ok
    dev_mask = (split == 'dev') & ok
    june, dev = frame.loc[june_mask], frame.loc[dev_mask]
    panel_mask_j, panel_mask_d = june.pitcher.isin(PANEL_IDS).to_numpy(), dev.pitcher.isin(PANEL_IDS).to_numpy()
    cpanel, cpanel_dev = june.loc[panel_mask_j], dev.loc[panel_mask_d]
    train_rows = train.index.to_numpy()
    arrays = training_arrays((*store.gather(train_rows), context.transform(frame.iloc[train_rows])))
    y_train = outcome_labels(train)

    members, weights_records = {}, []
    cpanel_members, dev_members = [], []
    for seed in mj.SEEDS:
        run = p4 if seed < 3 else p10
        model = MatrixModel('flatten_mlp', seed=seed, width=WIDTH).fit(arrays, y_train, arrays, y_train, epochs=1,
                                                                      patience=1, batch_size=256, learning_rate=.01)
        fit_dir = run / 'fits' / f'seed{seed}' / 'global'
        fit_dir.mkdir(parents=True, exist_ok=True)
        model.save(fit_dir / 'model.pt')
        dump(fit_dir / 'fit.json', {'report': model.report})
        dump(fit_dir / 'state.json', {'seed': seed})
        device = MatrixModel.load(fit_dir / 'model.pt').device
        temperature = 1.0 + 0.01 * seed
        predictor = SharingPredictor('G0-global', model, clusters, individual_tau=1000, cluster_tau=10000)
        predictor.delivery_temperature = temperature
        folder = run / 'members' / 'G0-global' / f'seed{seed}'
        cal, raw_p, levels = predict_streamed(predictor, delivery, store, context, cpanel.index.to_numpy())
        archive = {'blend': cal, 'blend_raw': raw_p, 'blend_delivery_level': levels,
                   'blend_keys': np.asfortranarray(cpanel[KEY].to_numpy(np.int64)), 'blend_y': outcome_labels(cpanel),
                   'blend_game_pk': cpanel.game_pk.to_numpy(np.int64), 'blend_pitcher': cpanel.pitcher.to_numpy(np.int64)}
        savez(folder / 'predictions.npz', **archive)
        cpanel_members.append(archive)
        dump(folder / 'prediction_state.json', {'seed': seed})
        dump(folder / 'calibration.json', {'cell': 'G0-global', 'delivery_temperature': temperature,
                                           'delivery_calibration_rows': MAY_ROWS})
        dcal, draw, dlev = predict_streamed(predictor, delivery, store, context, dev.index.to_numpy())
        dev_archive = {'dev': dcal, 'dev_raw': draw, 'dev_delivery_level': dlev,
                       'dev_keys': np.asfortranarray(dev[KEY].to_numpy(np.int64)), 'dev_y': outcome_labels(dev),
                       'dev_game_pk': dev.game_pk.to_numpy(np.int64), 'dev_pitcher': dev.pitcher.to_numpy(np.int64)}
        savez(p11 / 'members' / 'G0-global' / f'seed{seed}' / 'predictions.npz', **dev_archive)
        dev_members.append(dev_archive)
        members[str(seed)] = {
            'seed': seed, 'source_run': str(run), 'directory': str(folder),
            'state_path': str(folder / 'prediction_state.json'), 'prediction_state_sha256': sha(folder / 'prediction_state.json'),
            'predictions_sha256': sha(folder / 'predictions.npz'), 'calibration_sha256': sha(folder / 'calibration.json'),
            'model_path': str(fit_dir / 'model.pt'), 'model_sha256': sha(fit_dir / 'model.pt'), 'fit_dir': str(fit_dir),
            'fit_state_sha256': sha(fit_dir / 'state.json'), 'fit_sha256': sha(fit_dir / 'fit.json'),
            'delivery_temperature': temperature, 'device': device, 'network': model.report['network'],
            'parameter_count': model.report['parameter_count']}

    def frequency_part(part):
        raw_f = StubFrequency().predict(part)
        return raw_f, temperature_predictions(raw_f, TEMPERATURE)
    baseline = {}
    for name, part in (('blend', cpanel), ('dev', cpanel_dev), ('mlb_dev', dev)):
        raw_f, cal_f = frequency_part(part)
        baseline.update({name + '_raw': raw_f, name: cal_f, name + '_keys': np.asfortranarray(part[KEY].to_numpy(np.int64)),
                         name + '_y': outcome_labels(part), name + '_game_pk': part.game_pk.to_numpy(np.int64),
                         name + '_pitcher': part.pitcher.to_numpy(np.int64)})
    savez(p10 / 'parent_baseline_predictions.npz', **baseline)

    cy = baseline['blend_y']
    ensemble = np.mean([m['blend'] for m in cpanel_members], axis=0)
    selection = fit_blend(cy, ensemble, baseline['blend'], 'log_loss')
    seed_weights = [fit_blend(cy, m['blend'], baseline['blend'], 'log_loss')['model_weight'] for m in cpanel_members]
    weights = [selection['model_weight'], *seed_weights]
    results = {'reports': {'G0-global': {'selection': selection,
                                         'seeds': [{'blend_selection': {'model_weight': w}} for w in seed_weights]}}}
    dump(p10 / 'analysis' / 'results.json', results)
    dump(p10 / 'analysis' / 'manifest.json', {'results_sha256': sha(p10 / 'analysis' / 'results.json')})
    dump(p10 / 'preparation.json', {'synthetic': 'P10'})

    lookup = {p: ('low' if n <= 170 else 'middle' if n <= 1514 else 'high') for p, n in TRAIN_PLAYERS.items()}
    seed_cal = np.stack([m['blend'] for m in cpanel_members])
    savez(p112 / 'june_inputs.npz', keys=cpanel[KEY].to_numpy(np.int64), y=cy, game_pk=baseline['blend_game_pk'],
          pitcher=baseline['blend_pitcher'], frequency=baseline['blend'],
          groups=np.asarray([lookup.get(int(p), 'zero') for p in baseline['blend_pitcher']]),
          calibrated=ensemble, seed_calibrated=seed_cal,
          primary=weights[0] * ensemble + (1 - weights[0]) * baseline['blend'],
          seed_primary=np.stack([w * m + (1 - w) * baseline['blend'] for w, m in zip(weights[1:], seed_cal)]),
          weights=np.asarray(weights))

    dev_cal = np.mean([m['dev'] for m in dev_members], axis=0)
    savez(p11 / 'analysis' / 'predictions.npz', keys=dev[KEY].to_numpy(np.int64), y=outcome_labels(dev),
          game_pk=dev.game_pk.to_numpy(np.int64), pitcher=dev.pitcher.to_numpy(np.int64), calibrated=dev_cal,
          raw=np.mean([m['dev_raw'] for m in dev_members], axis=0),
          primary=weights[0] * dev_cal + (1 - weights[0]) * baseline['mlb_dev'],
          seed_primary=np.stack([w * m['dev'] + (1 - w) * baseline['mlb_dev'] for w, m in zip(weights[1:], dev_members)]))
    dump(p11 / 'analysis' / 'manifest.json', {'predictions_sha256': sha(p11 / 'analysis' / 'predictions.npz')})

    def metadata(part, research_split):
        return pd.DataFrame({
            **{k: part[k].to_numpy(np.int64) for k in KEY}, 'game_date': part.game_date.to_numpy(),
            'research_split': research_split, 'pitcher': part.pitcher.to_numpy(np.int64),
            'batter': part.batter.to_numpy(np.int64), 'in_cpanel': part.pitcher.isin(PANEL_IDS).to_numpy(),
            'train_volume': [lookup.get(int(p), 'zero') for p in part.pitcher],
            'game_role': np.where(part.pitcher.eq(part.starter_pitcher), 'starter', 'relief'),
            'throwing_hand': part.p_throws.to_numpy(), 'month': part.game_date.dt.to_period('M').astype(str).to_numpy()})
    june_meta = metadata(june, 'blend')
    june_meta['frozen_cpanel_eligible'] = june_meta.in_cpanel.to_numpy()
    june_meta.to_parquet(june_dir / 'eligible_metadata.parquet', index=False)
    june[KEY].to_parquet(june_dir / 'eligible_keys.parquet', index=False)
    dump(june_dir / 'result.json', {'counts': {'eligible': {'pitches': len(june), 'games': int(june.game_pk.nunique()),
                                                            'pitchers': int(june.pitcher.nunique())}}})
    june_meta.loc[june_meta.in_cpanel, [*KEY, 'pitcher', 'batter', 'in_cpanel', 'train_volume', 'game_role',
                                         'throwing_hand', 'month']].to_parquet(p4 / 'blend_metadata.parquet', index=False)
    dev_meta = metadata(dev, 'dev').drop(columns=['game_date', 'research_split'])
    dev_meta['seen_pitcher'] = dev.pitcher.isin(list(TRAIN_PLAYERS)).to_numpy()
    dev_meta['seen_batter'] = dev.batter.lt(915).to_numpy()
    dev_meta['train_pitches'] = [TRAIN_PLAYERS.get(int(p), 0) for p in dev.pitcher]
    dev_meta['train_role'] = 'unseen'
    dev_meta['two_strikes'] = pd.array(dev.strikes.eq(2).to_numpy(), dtype='boolean')
    dev_meta['runners_on'] = pd.array(dev.bases.gt(0).to_numpy(), dtype='boolean')
    dev_meta.to_parquet(p11 / 'dev_metadata.parquet', index=False)

    panel = {'volume_thresholds': {'q25': 170, 'q75': 1514}, 'pitcher_ids': list(PANEL_IDS),
             'train_players': [{'pitcher': p, 'train_pitches': n, 'train_volume': lookup[p]} for p, n in TRAIN_PLAYERS.items()]}
    dump(p4 / 'panel.json', panel)
    dump(p4 / 'parent_preparation.json', {'dataset_identity': raw.attrs['sequence_data_identity'],
                                          'source_provenance': raw.attrs['matrix_source_provenance']})
    dump(p4 / 'preparation.json', {'features': {'tokens': tokens}, 'clusters': clusters,
                                   'baseline_temperature': {'temperature': TEMPERATURE},
                                   'artifact_hashes': {name: sha(p4 / name) for name in
                                                       ('parent_preparation.json', 'blend_metadata.parquet', 'aux.pkl', 'panel.json')}})
    dump(p11 / 'preparation.json', {'features': {'tokens': tokens}, 'clusters': clusters, 'members': members,
                                    'frozen_calibration': {'june_ensemble_model_weight': weights[0],
                                                           'june_seed_model_weights': {str(s): weights[s + 1] for s in mj.SEEDS},
                                                           'may_delivery_temperatures': {str(s): 1.0 + 0.01 * s for s in mj.SEEDS}},
                                    'artifact_hashes': {'dev_metadata.parquet': sha(p11 / 'dev_metadata.parquet')}})
    for name in ('g0_bundle.json', 'june_config.json', 'june_manifest.json', 'p11_config.json', 'p11_calibration.json'):
        dump(root / 'misc' / name, {'synthetic': name})
    june[KEY].iloc[:10].to_parquet(june_dir / 'complement_eligible_keys.parquet', index=False)
    contract = root / 'contract.md'
    contract.write_text('synthetic contract\n')

    parents = {
        'g0_bundle': root / 'misc' / 'g0_bundle.json', 'june_config': root / 'misc' / 'june_config.json',
        'june_manifest': root / 'misc' / 'june_manifest.json', 'june_result': june_dir / 'result.json',
        'june_keys': june_dir / 'eligible_keys.parquet', 'june_metadata': june_dir / 'eligible_metadata.parquet',
        'june_complement_keys': june_dir / 'complement_eligible_keys.parquet', 'p11_config': root / 'misc' / 'p11_config.json',
        'p11_preparation': p11 / 'preparation.json', 'p11_analysis_manifest': p11 / 'analysis' / 'manifest.json',
        'p11_calibration_config': root / 'misc' / 'p11_calibration.json', 'p4_preparation': p4 / 'preparation.json',
        'p4_panel': p4 / 'panel.json', 'p10_frequency': p10 / 'parent_baseline_predictions.npz',
        'p10_results': p10 / 'analysis' / 'results.json', 'p10_manifest': p10 / 'analysis' / 'manifest.json',
        'p10_preparation': p10 / 'preparation.json', 'p11_cpanel_june_inputs': p112 / 'june_inputs.npz',
        'p11_predictions': p11 / 'analysis' / 'predictions.npz', 'p4_auxiliary': p4 / 'aux.pkl',
        'p4_model_source': PROJECT / 'pitchmdp' / 'model.py',
        'p10_scalar_summarizer_source': PROJECT / 'scripts' / 'score_ml_matrix.py',
        'p4_frequency_temperature_source': PROJECT / 'scripts' / 'run_sequence_frequency_baselines.py'}
    for seed in mj.SEEDS:
        run = p4 if seed < 3 else p10
        parents[f'cpanel_member_seed{seed}'] = run / 'members' / 'G0-global' / f'seed{seed}' / 'predictions.npz'
        parents[f'dev_member_seed{seed}'] = p11 / 'members' / 'G0-global' / f'seed{seed}' / 'predictions.npz'

    sci = json.loads(SCIENTIFIC.read_text())
    sci['parents'] = {name: {'path': str(path), 'sha256': sha(path)} for name, path in parents.items()}
    sci['contract'] = {'path': str(contract), 'sha256': sha(contract)}
    groups = june_meta.train_volume.to_numpy()
    support = {g: {'pitches': int((groups == g).sum()), 'games': int(june.game_pk[groups == g].nunique())} for g in mj.GROUPS}
    constants = {
        'JUNE': {'rows': len(june), 'games': int(june.game_pk.nunique()), 'pitchers': int(june.pitcher.nunique()),
                 'date_min': '2025-06-01', 'date_max': '2025-06-30',
                 'ordered_key_sha256': hashlib.sha256(june[KEY].to_numpy(np.int64).tobytes()).hexdigest()},
        'CPANEL': {'rows': len(cpanel), 'games': int(cpanel.game_pk.nunique()),
                   'ordered_key_sha256': hashlib.sha256(cpanel[KEY].to_numpy(np.int64).tobytes()).hexdigest()},
        'DEV': {'rows': len(dev), 'games': int(dev.game_pk.nunique()), 'cpanel_rows': len(cpanel_dev),
                'cpanel_games': int(cpanel_dev.game_pk.nunique()), 'complement_rows': len(dev) - len(cpanel_dev),
                'months': ('2025-07', '2025-08', '2025-09')},
        'EXPECTED_SUPPORT': support}
    fit = sci['populations']['fit']
    fit.update(rows=constants['JUNE']['rows'], games=constants['JUNE']['games'], pitchers=constants['JUNE']['pitchers'],
               ordered_key_sha256=constants['JUNE']['ordered_key_sha256'])
    sci['populations']['cpanel_replay'].update(rows=len(cpanel), games=constants['CPANEL']['games'])
    sci['populations']['dev'].update(whole_rows=len(dev), whole_games=constants['DEV']['games'],
                                     cpanel_rows=len(cpanel_dev), complement_rows=len(dev) - len(cpanel_dev))
    sci['variants']['B2']['expected_support'] = support
    sci['components']['draws'] = DRAWS
    sci['budget']['profile_gate'].update(rows=PROFILE_ROWS, draws=DRAWS, cpanel_probe_rows=PROBE_ROWS)
    lock = runs / '.heavy.lock'
    lock.write_bytes(b'')
    sci['budget']['heavy_lock'] = str(lock)
    sci['future_output_root'] = str(runs / 'ML-JUNE-CALIBRATION-v1')
    return {'root': root, 'runs': runs, 'sci': sci, 'constants': constants, 'local': local, 'lock': lock,
            'network': {'kind': 'flatten_mlp', 'n_context': members['0']['network']['n_context'], 'width': WIDTH,
                        'individual_tau': 1000, 'cluster_tau': 10000, 'may_rows': MAY_ROWS},
            'data': {'local_config': local, 'processed_pitches': processed, 'physics_cache': cache,
                     'quality_manifest': root / 'reports' / 'data_quality.json',
                     'cache_manifest': root / 'cache' / 'sequence_physics.json'},
            'full_frame': frame}


def write_configs(family, runner, attempt, *, sci_changes=None, exec_changes=None):
    """Scientific + execution JSON for one fresh attempt directory inside the family root."""
    root = Path(family['root'])
    sci = copy.deepcopy(family['sci'])
    env = runner.runtime_environment()
    sci['B0_provenance']['recorded_c1_versions'] = {'numpy': env['numpy'], 'scipy': env['scipy'],
                                                    'python_executable': env['python']}
    if sci_changes:
        sci_changes(sci)
    sci_path = root / f'scientific-{attempt}.json'
    dump(sci_path, sci)
    output = Path(sci['future_output_root']) / attempt
    config = {'protocol': mj.EXECUTION_PROTOCOL, 'family_id': mj.FAMILY_ID, 'enabled': True, 'code_commit_c': '0' * 40,
              'scientific_config': {'path': str(sci_path), 'sha256': sha(sci_path)},
              'contract': dict(sci['contract']), 'source_hashes': runner.source_closure(), 'environment': env,
              'data': {name: {'path': str(path), 'sha256': sha(path)} for name, path in family['data'].items()},
              'output_dir': str(output), 'heavy_lock': str(family['lock'])}
    if exec_changes:
        exec_changes(config)
    config_path = root / f'execution-{attempt}.json'
    dump(config_path, config)
    return config_path, output

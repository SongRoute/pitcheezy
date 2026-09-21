"""Run no-past-token training and batter validation in one resumable M4 job.

No work runs at import. --dry-run prints the plan without data/model reads or writes.
See docs/HISTORY_BATTER_VALIDATION.md for the frozen comparisons and launch command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = 'temporal-blend-20260921T050400Z'
DEFAULT_REPRESENTATIONS = 'representation-history-20260921T054642Z'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='SSD run directory; same path resumes')
    parser.add_argument('--base', type=Path, help='Completed temporal reference run')
    parser.add_argument('--representations', type=Path, help='Completed representation sweep')
    parser.add_argument('--cpu-threads', type=int, default=min(8, os.cpu_count() or 4),
                        help='Preprocessing/BLAS threads; frozen training uses four')
    parser.add_argument('--inference-batch', type=int, default=8192,
                        help='MPS forward batch size; optimizer batch stays 1024')
    parser.add_argument('--delivery-chunk', type=int, default=128,
                        help='Evaluation rows expanded to 400 candidates at a time')
    parser.add_argument('--allow-cpu', action='store_true', help='Explicitly allow slow CPU fallback')
    parser.add_argument('--dry-run', action='store_true', help='Print plan only; no training or writes')
    args = parser.parse_args(argv)
    for key in ('cpu_threads', 'inference_batch', 'delivery_chunk'):
        if getattr(args, key) < 1:
            parser.error('--'+key.replace('_', '-')+' must be positive')
    if args.delivery_chunk > 512:
        parser.error('--delivery-chunk must be <=512 to bound candidate expansion')
    return args


def make_plan(args, local):
    runs = Path(local['artifact_root']).resolve()/'runs'
    base = (args.base or runs/DEFAULT_BASE).resolve()
    representations = (args.representations or runs/DEFAULT_REPRESENTATIONS).resolve()
    output = (args.output or runs/'history-batter-validation-v1').resolve()
    for path in (base, representations, output):
        if not path.is_relative_to(runs) or path == runs:
            raise ValueError('All run paths must be children of the configured SSD runs directory')
    for a, b in ((base, output), (representations, output), (base, representations)):
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('Use separate, nonoverlapping run directories')
    return {'experiment': 'history_batter_validation_v1', 'base': str(base),
            'representations': str(representations), 'output': str(output),
            'years': [2024, 2025], 'seeds': [42, 43, 44, 45, 46],
            'representations_compared': ['reference', 'continuous', 'clusters_3', 'clusters_5', 'clusters_10', 'clusters_20'],
            'new_history': 0, 'reused_history': 5, 'new_models': 60, 'reused_models': 60,
            'cpu_threads': args.cpu_threads, 'training_cpu_threads': 4,
            'inference_batch': args.inference_batch, 'delivery_chunk': args.delivery_chunk,
            'allow_cpu': args.allow_cpu, 'epochs': 30, 'patience': 5, 'batch_size': 1024,
            'learning_rate': .0005, 'delivery_draws': 400, 'bootstrap_replicates': 2000,
            'raw_years': [2023, 2024, 2025], 'selection': 'No DEV selection of K/window/model',
            'history_scope': 'No past PA physical tokens/masks; game state and prior-date batter profiles retained'}


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def load_archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def validate_references(plan):
    base, representations = Path(plan['base']), Path(plan['representations'])
    for source in (base, representations):
        print('VERIFY_REFERENCE', source, flush=True)
        verify_manifest(source)
        check_sources(source)
        audit = json.loads((source/'final_audit.json').read_text())
        if audit.get('status') != 'passed':
            raise ValueError('Reference independent audit did not pass')
    config = json.loads((representations/'config.json').read_text())
    if (config['experiment'] != 'representation_history' or config['base'] != str(base)
            or config['years'] != list(YEARS) or config['seeds'] != list(SEEDS)
            or config['base_artifact_manifest_sha256'] != hash_file(base/'artifact_hashes.json')):
        raise ValueError('Representation run is not paired with this temporal reference')
    if json.loads((base/'data_identity.json').read_text()) != json.loads((representations/'data_identity.json').read_text()):
        raise ValueError('Reference data identities differ')
    sources = set(json.loads((base/'source_hashes.json').read_text()))
    sources.update(json.loads((representations/'source_hashes.json').read_text()))
    sources.update(['scripts/run_history_batter_validation.py', 'scripts/history_batter_support.py',
                    'docs/HISTORY_BATTER_VALIDATION.md'])
    source_hashes = {rel: hash_file(PROJECT/rel) for rel in sorted(sources)}
    reference_hashes = {str(source/name): hash_file(source/name) for source in (base, representations)
                        for name in ('artifact_hashes.json', 'source_hashes.json', 'data_identity.json', 'final_audit.json')}
    return source_hashes, reference_hashes


def initialize_run(output, plan, source_hashes, reference_hashes):
    contract = {'plan': plan, 'source_hashes': source_hashes, 'reference_hashes': reference_hashes}
    setup = output/'setup'
    if output.exists() and not setup.exists() and any(
            p.name != 'run.log' and not p.name.startswith('.setup-') for p in output.iterdir()):
        raise ValueError('Nonempty output does not belong to this experiment')

    def write(stage):
        dump(stage/'config.json', {**plan, 'created_utc': datetime.now(timezone.utc).isoformat()})
        dump(stage/'source_hashes.json', source_hashes)
        dump(stage/'reference_hashes.json', reference_hashes)
        for rel in source_hashes:
            target = stage/'source'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/rel, target)
    commit_stage(setup, contract, write)
    check_sources(setup)
    return hash_file(setup/'stage.json')


def member_prediction(directory, contract, train, early, ys, parts, delivery, view, context, plan):
    """Interrupted fits restart their seed; completed fit/calibration/prediction stages resume."""
    fitted, calibrated = directory/'fit', directory/'calibration'
    model = None
    try:
        if not read_stage(fitted, contract):
            print('FIT_START', contract, flush=True)
            model = BatchedModel('flatten_mlp', seed=contract['seed'], width=128)
            model.fit(train, ys['train'], early, ys['earlystop'], epochs=30, patience=5,
                      batch_size=1024, learning_rate=.0005)
            torch.set_num_threads(plan['cpu_threads'])

            def write_fit(stage):
                model.save(stage/'model.pt')
                dump(stage/'report.json', model.report)
            commit_stage(fitted, contract, write_fit)
        calibration_contract = {**contract, 'fit_stage_sha256': hash_file(fitted/'stage.json')}
        if not read_stage(calibrated, calibration_contract):
            if model is None:
                model = BatchedModel.load(fitted/'model.pt')

            def write_calibration(stage):
                rows = parts['temperature'].index.to_numpy()
                logits = np.lib.format.open_memmap(stage/'logits.npy', mode='w+', dtype=np.float32,
                                                   shape=(len(rows), 400, 10))
                levels = np.empty(len(rows), dtype=np.int64)
                for begin, chunk, used in integrated_chunks(delivery, model, view, context, rows, plan['delivery_chunk']):
                    logits[begin:begin+len(chunk)] = chunk
                    levels[begin:begin+len(chunk)] = used
                logits.flush()
                model.delivery_temperature, score = fit_temperature_logits(logits, ys['temperature'])
                model.report.update(delivery_temperature=model.delivery_temperature,
                                    calibration_integrated_log_loss=score,
                                    delivery_calibration_rows=len(rows))
                del logits
                np.save(stage/'levels.npy', levels)
                model.save(stage/'model.pt')
                dump(stage/'report.json', model.report)
            commit_stage(calibrated, calibration_contract, write_calibration)
        prediction_contract = {**contract, 'calibration_stage_sha256': hash_file(calibrated/'stage.json')}
        # Always derive predictions from the committed calibrated checkpoint, including on resume.
        model = None
        predictions = {}
        for part in ('blend', 'dev'):
            stage_path = directory/part
            if not read_stage(stage_path, prediction_contract):
                if model is None:
                    model = BatchedModel.load(calibrated/'model.pt')

                def write_predictions(stage):
                    p = integrated_predict(delivery, model, view, context, parts[part].index.to_numpy(), plan['delivery_chunk'])
                    classification_metrics(ys[part], p)
                    np.savez_compressed(stage/'predictions.npz', probabilities=p, y=ys[part],
                                        pitch_keys=parts[part][KEY].to_numpy())
                commit_stage(stage_path, prediction_contract, write_predictions)
            saved = load_archive(stage_path/'predictions.npz')
            np.testing.assert_array_equal(saved['pitch_keys'], parts[part][KEY].to_numpy())
            np.testing.assert_array_equal(saved['y'], ys[part])
            predictions[part] = saved['probabilities']
        print('MEMBER_COMPLETE', contract['year'], contract['representation'], contract['seed'], flush=True)
        return predictions
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def run_fold(raw, year, output, plan, identity):
    base, representations = Path(plan['base'])/str(year), Path(plan['representations'])/str(year)
    dest = output/str(year)
    dest.mkdir(exist_ok=True)
    frame = assign_fold(raw, year)
    ids, cohort = select_cohort(frame)
    full, parts = samples_for(frame, ids)
    ys = {name: outcome_labels(part) for name, part in parts.items()}
    original = json.loads((base/'samples.json').read_text())
    if rows_hash(full) != original['full_train']['rows_sha256']:
        raise ValueError('Full TRAIN sample changed')
    for name, part in parts.items():
        if rows_hash(part) != original[name]['rows_sha256']:
            raise ValueError('Neural sample changed: '+name)
        for source in (base, representations):
            keys = pd.read_parquet(source/(name+'_keys.parquet'))
            np.testing.assert_array_equal(keys[KEY].to_numpy(), part[KEY].to_numpy())
    with (base/'fitted_preprocessors.pkl').open('rb') as stream:
        prep = pickle.load(stream)
    store = HistoryStore.from_frame(frame, normalizer=prep['normalizer'])
    view, delivery = NoPastStore(store), prep['delivery']
    if delivery.draws != 400:
        raise ValueError('Expected exact shared400 TRAIN delivery object')
    olddev, oldcal = load_archive(base/'predictions.npz'), load_archive(base/'calibration_predictions.npz')
    repdev = load_archive(representations/'predictions.npz')
    for archive, part in ((olddev, 'dev'), (repdev, 'dev'), (oldcal, 'blend')):
        np.testing.assert_array_equal(archive['pitch_keys'], parts[part][KEY].to_numpy())
        np.testing.assert_array_equal(archive['y'], ys[part])
    baseline = {'blend': oldcal['baseline'], 'dev': olddev['frequency']}
    seen = set(parts['train'].batter.astype(int))
    available = (store.indices[parts['dev'].index.to_numpy()] >= 0).sum(1)
    subgroup_masks = {
        'known_batter': parts['dev'].batter.isin(seen).to_numpy(),
        'unseen_batter': ~parts['dev'].batter.isin(seen).to_numpy(),
        'past_available': available > 0, 'five_past_available': available >= 5,
        **{f'pitcher_{pid}': parts['dev'].pitcher.eq(pid).to_numpy() for pid in ids}}
    predictions, ensembles, results = {}, {}, {}
    for name in REPRESENTATIONS:
        if name == 'reference':
            context = BatterContext(prep['context'], 'reference').fit(full, id_train=parts['train'])
        else:
            with (representations/name/'context.pkl').open('rb') as stream:
                context = pickle.load(stream)
        train, early = arrays(view, context, parts['train'].index.to_numpy()), arrays(view, context, parts['earlystop'].index.to_numpy())
        members = {0: [], 5: []}
        for seed in SEEDS:
            archive = (base/f'flatten_mlp_{seed}_predictions.npz' if name == 'reference'
                       else representations/name/f'seed{seed}_predictions.npz')
            reused = load_archive(archive)
            for part in ('blend', 'dev'):
                np.testing.assert_array_equal(reused[part+'_keys'], parts[part][KEY].to_numpy())
                classification_metrics(ys[part], reused[part])
            members[5].append({part: reused[part] for part in ('blend', 'dev')})
            contract = {'run_identity': identity, 'year': year, 'representation': name, 'seed': seed, 'past_tokens': 0}
            members[0].append(member_prediction(dest/name/f'seed{seed}', contract, train, early, ys,
                                                parts, delivery, view, context, plan))
        for history in (0, 5):
            key = name+f'_h{history}'
            ensemble = {part: np.stack([member[part] for member in members[history]]).mean(0) for part in ('blend', 'dev')}
            selection = fit_blend(ys['blend'], ensemble['blend'], baseline['blend'], 'log_loss')
            weight = selection['model_weight']
            mixed = weight*ensemble['dev']+(1-weight)*baseline['dev']
            if history == 5:
                expected = olddev['flatten_mlp_blend'] if name == 'reference' else repdev[name]
                np.testing.assert_allclose(mixed, expected, rtol=0, atol=1e-7)
            predictions[key], ensembles[key] = mixed, ensemble['dev']
            results[key] = {'metrics': classification_metrics(ys['dev'], mixed),
                'unblended_metrics': classification_metrics(ys['dev'], ensemble['dev']),
                'selection': selection, 'context': context.report(), 'reused': history == 5,
                'seed_metrics': [classification_metrics(ys['dev'], m['dev']) for m in members[history]],
                'subgroups': {group: {'mixed': classification_metrics(ys['dev'][mask], mixed[mask]),
                                     'unblended': classification_metrics(ys['dev'][mask], ensemble['dev'][mask])}
                              for group, mask in subgroup_masks.items()}}
            np.savez_compressed(dest/(key+'_ensemble.npz'), blend=ensemble['blend'], dev=ensemble['dev'],
                                mixed_dev=mixed, blend_keys=parts['blend'][KEY].to_numpy(),
                                dev_keys=parts['dev'][KEY].to_numpy())
        del train, early, members, context
        gc.collect()
    comparison = compare_families(ys['dev'], predictions, parts['dev'].game_pk.to_numpy())
    # Unblended contrasts are sensitivity analyses, never additional primary selection tests.
    unblended = compare_families(ys['dev'], ensembles, parts['dev'].game_pk.to_numpy())
    np.savez_compressed(dest/'predictions.npz', y=ys['dev'], pitch_keys=parts['dev'][KEY].to_numpy(),
                        game_pk=parts['dev'].game_pk.to_numpy(), pitcher=parts['dev'].pitcher.to_numpy(),
                        batter=parts['dev'].batter.to_numpy(), history_available=available,
                        unknown_batter=subgroup_masks['unseen_batter'], frequency=baseline['dev'], **predictions)
    result = {'year': year, 'cohort': cohort, 'samples': original, 'variants': results,
              'comparisons': comparison, 'unblended_sensitivity': unblended,
              'subgroup_sizes': {name: int(mask.sum()) for name, mask in subgroup_masks.items()}}
    dump(dest/'results.json', result)
    print('FOLD_COMPLETE', year, flush=True)
    return result


def write_report(output, folds):
    lines = ['# 과거 투구 0/5구와 타자 표현 검증', '',
             '주자·아웃·카운트·점수차·홈원정·타자 성향은 모든 조건에 유지했다. '
             '0구는 같은 타석의 과거 물리 토큰과 이력 마스크만 제거한다. 현재 후보 투구는400개 TRAIN 표본으로 적분했다.', '',
             '신규60개 모델과 기존60개 모델. 이미 관측한2024/2025 자료의 탐색 비교이며 최적 K를 선택하지 않는다.', '',
             '| 연도 | 표현 | 과거 구수 | 혼합 LL | 단독 LL | Brier | NN 비중 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for year, fold in folds.items():
        for name, entry in fold['variants'].items():
            representation, history = name.rsplit('_h', 1)
            lines.append(f"| {year} | {representation} | {history} | {entry['metrics']['log_loss']:.6f} | "
                         f"{entry['unblended_metrics']['log_loss']:.6f} | {entry['metrics']['brier_multiclass']:.6f} | "
                         f"{entry['selection']['model_weight']:.1%} |")
    for family, explanation in [('history', '0구−5구. 양수이면 과거5구를 쓰는 방식이 우세.'),
                                ('representation', '해당 표현−연속 성향. 음수이면 해당 표현이 우세.'),
                                ('interaction', '(해당 표현의0구−5구)−(연속 성향의0구−5구). 양수이면 해당 표현에서 이력 이득이 더 큼.')]:
        lines += ['', '## '+family, '', explanation, '', '| 연도 | 비교 | Δ LL | 비교군 보정95%구간 |', '|---|---|---:|---|']
        for year, fold in folds.items():
            for name, entry in fold['comparisons']['families'][family]['log_loss'].items():
                lo, hi = entry['simultaneous95']
                lines.append(f"| {year} | {name} | {entry['estimate']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## 투수별·타자별 기술 통계', '',
              '| 연도 | 구성 | 집단 | n | 혼합 LL | 단독 LL |', '|---|---|---|---:|---:|---:|']
    for year, fold in folds.items():
        for name, entry in fold['variants'].items():
            for group, metrics in entry['subgroups'].items():
                m, u = metrics['mixed'], metrics['unblended']
                if m['n']:
                    lines.append(f"| {year} | {name} | {group} | {m['n']} | {m['log_loss']:.6f} | {u['log_loss']:.6f} |")
                else:
                    lines.append(f'| {year} | {name} | {group} | 0 | 미평가 | 미평가 |')
    lines += ['', '구간은 각 연도·비교군 안에서 보정한 경기 bootstrap이며 학습·보정·모델 선택의 전체 불확실성은 포함하지 않는다. '
              '두 연도의 TRAIN은 중첩된다. 투수/신규 타자 하위집단은 기술 통계이고 군집 seed는42로 고정했다. '
              '표현별 모델 용량이 조금 다르며 혼합은 정보 차이를 완화할 수 있다. 원 표본의 사후 적격 필터를 유지했다. '
              '실제 추천 안정성이나 승률 개선을 평가한 결과가 아니다.', '',
              '세부 Brier 구간·단독 모델 민감도·초기값별 점수는 results.json에 저장했다. '
              '실행 중 해시/표본/확률/기존 결과 재현 검사를 수행했으며 별도 독립 연구 감사와는 구분한다.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))


def execute(plan, local):
    root, output = Path(local['artifact_root']).resolve(), Path(plan['output'])
    if not Path('/Volumes/T7 Shield').is_mount() or not root.is_relative_to(Path('/Volumes/T7 Shield')):
        raise ValueError('Mounted T7 Shield required; no internal-disk fallback')
    if Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise ValueError('Use the configured existing .venv Python')
    if local['raw_allowlist'] != ['statcast_2023.parquet', 'statcast_2024.parquet', 'statcast_2025.parquet']:
        raise ValueError('Only approved2023–2025 raw files are allowed')
    if not torch.backends.mps.is_available() and not plan['allow_cpu']:
        raise ValueError('MPS unavailable; use --allow-cpu only if CPU execution is intended')
    BatchedModel.inference_batch = plan['inference_batch']
    torch.set_num_threads(plan['cpu_threads'])
    with execution_lock(root/'.history_batter_validation.lock'):
        source_hashes, reference_hashes = validate_references(plan)
        identity = initialize_run(output, plan, source_hashes, reference_hashes)
        if (output/'artifact_hashes.json').exists():
            verify_manifest(output)
            print('ALREADY_COMPLETE', output/'RESULTS.md', flush=True)
            return
        with (output/'run.log').open('a', buffering=1) as log:
            oldout, olderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(oldout, log), Tee(olderr, log)
            try:
                print('RUN', output, datetime.now(timezone.utc).isoformat(), flush=True)
                print('DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu', 'PLAN', plan, flush=True)
                raw = add_batter_style_history(prepare_frame(local))
                if raw.attrs.get('sequence_data_identity') != json.loads((Path(plan['base'])/'data_identity.json').read_text()):
                    raise ValueError('Prepared data differs from frozen reference')
                dump(output/'data_identity.json', raw.attrs['sequence_data_identity'])
                folds = {str(year): run_fold(raw, year, output, plan, identity) for year in YEARS}
                check_sources(output/'setup')
                # Recheck immutable reference identities after this potentially long job.
                for path, digest in reference_hashes.items():
                    if hash_file(Path(path)) != digest:
                        raise ValueError('Reference manifest changed during execution')
                dump(output/'results.json', {'folds': folds, 'finished_utc': datetime.now(timezone.utc).isoformat()})
                write_report(output, folds)
                files = {str(p.relative_to(output)): hash_file(p) for p in output.rglob('*')
                         if p.is_file() and p.name not in ('run.log', 'artifact_hashes.json')
                         and not any(part.endswith('.partial') for part in p.relative_to(output).parts)}
                dump(output/'artifact_hashes.json', files)
                verify_manifest(output)
                print('COMPLETE', output/'RESULTS.md', flush=True)
            except BaseException:
                traceback.print_exc()
                print('STOPPED: rerun the identical command to resume committed stages.', flush=True)
                raise
            finally:
                sys.stdout, sys.stderr = oldout, olderr


def main(argv=None):
    args = arguments(argv)
    local = json.loads((PROJECT/'configs/local.json').read_text())
    plan = make_plan(args, local)
    if args.dry_run:
        print(json.dumps({'plan': plan, 'dry_run': True, 'writes': False, 'training': False,
                          'note': 'No raw/model loading or hash verification; actual execution verifies references.'},
                         indent=2, ensure_ascii=False))
        return
    # Set CPU library limits BEFORE loading NumPy/PyTorch; no package installation.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(plan['cpu_threads'])
    sys.path.insert(0, str(PROJECT))
    global np, pd, torch, hash_file, KEY, dump, arrays, prepare_frame, HistoryStore
    global add_batter_style_history, outcome_labels, classification_metrics, BatterContext
    global assign_fold, select_cohort, samples_for, check_sources, rows_hash, fit_blend
    global verify_manifest, fit_temperature_logits, REPRESENTATIONS, YEARS, SEEDS
    global NoPastStore, BatchedModel, integrated_chunks, integrated_predict
    global read_stage, commit_stage, execution_lock, compare_families
    import numpy as np
    import pandas as pd
    import torch
    from pitchmdp.data import KEY, hash_file
    from pitchmdp.archetypes import add_batter_style_history
    from pitchmdp.model import outcome_labels
    from pitchmdp.sequence_data import prepare_frame, HistoryStore
    from pitchmdp.sequence_model import classification_metrics
    from run_sequence_pilot import dump, arrays
    from run_sequence_ablations import rows_hash
    from run_temporal_blend import assign_fold, select_cohort, samples_for, check_sources
    from run_sequence_calibration import fit_blend
    from run_representation_history import verify_manifest, fit_temperature_logits
    from representation_adapters import BatterContext
    from history_batter_support import (REPRESENTATIONS, YEARS, SEEDS, NoPastStore, BatchedModel,
        integrated_chunks, integrated_predict, read_stage, commit_stage, execution_lock, compare_families)
    execute(plan, local)


if __name__ == '__main__':
    main()

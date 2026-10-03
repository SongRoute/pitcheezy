"""DRAFT portable scorer for a COMPLETE CUDA direct-head family (newsrv side, numpy only).

Scores the base 5-arm family (DRAFT-CUDA-DIRECT-20261004), the frontier 9-arm
family (DRAFT-CUDA-FRONTIER-RUN-20261004) or the data/combination 15-arm family
(DRAFT-CUDA-DATA-COMBO-RUN-20261004) where the tasks were written, so results do
not wait for the Mac.  A config with comparison.factorial_contrasts also gets those
fixed candidate-minus-reference pairs (same paired whole-game bootstrap helper,
same-seed and ensemble), validated before any prediction is read; no ranking or
verdict.  Families without the key are scored exactly as before.  Likewise
comparison.regularization_contrasts (architecture x dropout family) must equal
direct_dropout_registry.CONTRASTS (dropout arm minus its same-architecture no-dropout
arm) and is checked before any read.  comparison.workload_dropout_contrasts (periodic
workload x head-dropout family) must equal direct_workload_dropout_registry.CONTRASTS
(12 one-factor pairs; every one of the 50 fits must exist first) and is checked
before any read; no ranking, winner or interaction verdict.  comparison.tabm_contrasts
(TabM-mini family) must equal direct_tabm_registry.CONTRASTS (3 one-factor pairs on
seeds 13-15; all 15 fits must exist first), likewise checked before any read.  Data arms must record the config's subset rule over the
export's own TRAIN rows (fit.json train_subset); every other arm must record none.  No raw data, pandas, sklearn, scipy, torch, G0 or Savant.

Scope: UNBLENDED calibrated direct-head probabilities (predictions.npz 'dev',
mlb_predictions.npz 'mlb_dev') of every arm vs the same-family CUDA control, BEFORE
the June blend / G0 mix.  Not the G15 mix, not a policy.  Descriptive repeated-DEV
only: no decision, ranking or winner (arms are listed in config order).

Fail closed, before any probability is read, unless
  * the config is the pinned file, passes config_check and is REGISTERED/enabled;
  * the export manifest is the pinned one and the dev/mlb_dev keys, y and game_pk
    arrays match their manifest hashes;
  * EVERY config arm x seed is a finished, non-smoke CUDA task whose identity equals
    this config/export/arm/seed and the EXPECTED task source map, with intact
    artifacts and one shared CUDA runtime signature.
The expected task sources are the current runner's own_hashes() by default, or an
explicit --source-manifest file (e.g. the first batch, built on older code without
the frontier modules); an artifact's self-declared sources are never trusted.
Every member's keys, labels and game ids must equal the export rows in order.

Metrics: matrix_metrics.prediction_metrics of the 3-seed probability-mean ensemble;
paired_game_comparison (arm minus control, whole-game bootstrap, 10,000 draws, seed
20260924) for the ensemble and for each seed vs the same-seed control.

  python scripts/score_cuda_direct.py --config <cfg> --config-sha256 <sha> --export <dir> \\
      --export-manifest-sha256 <sha> --family <family dir> --destination <new dir> \\
      [--source-manifest <task sources json>]
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np

import run_cuda_direct as rc  # noqa: E402  numpy only at import
from pitchmdp import direct_dropout_registry as dropout_registry  # noqa: E402  torch-free
from pitchmdp import direct_workload_dropout_registry as workload_dropout_registry  # noqa: E402  torch-free
from pitchmdp import direct_tabm_registry as tabm_registry  # noqa: E402  torch-free
from pitchmdp.matrix_metrics import paired_game_comparison, pitch_losses, prediction_metrics  # noqa: E402

FORMAT = 'cuda_direct_score_v1'
SOURCE_FORMAT = 'cuda_direct_task_sources_v1'
SPLITS = {'dev': 'predictions.npz', 'mlb_dev': 'mlb_predictions.npz'}
EXPORT_ARRAYS = ('keys', 'y', 'game_pk')
DRAWS, BOOTSTRAP_SEED = 10000, 20260924
SCORER_SOURCES = ('scripts/score_cuda_direct.py', 'scripts/run_cuda_direct.py', 'pitchmdp/matrix_metrics.py',
                  'pitchmdp/direct_dropout_registry.py', 'pitchmdp/direct_combo_registry.py',
                  'pitchmdp/direct_frontier_registry.py', 'pitchmdp/direct_workload_dropout_registry.py',
                  'pitchmdp/direct_tabm_registry.py')
RUNTIME_SIGNATURE = ('torch', 'torch_cuda', 'cudnn', 'numpy', 'scipy', 'device_name', 'device_capability',
                     'device_total_memory')
SCOPE = ('unblended calibrated direct-head probabilities vs the same-family CUDA control, before June blend / G0 mix; '
         'not G15 mix, not policy; descriptive repeated-DEV; no decision, ranking or winner')
HEX64 = re.compile(r'[0-9a-f]{64}')


def scorer_sources():
    return {rel: rc.sha256_file(PROJECT / rel) for rel in SCORER_SOURCES}


def task_sources(path=None):
    """Expected task source map: current runner by default, else an explicit pinned file (exact map)."""
    if path is None:
        return rc.own_hashes(), {'kind': 'current_runner_own_hashes'}
    data = rc.read_json(path)
    sources = data.get('source_sha256') if data.get('format') == SOURCE_FORMAT else None
    if not isinstance(sources, dict) or rc.SELF not in sources or rc.MODEL_SOURCES[0] not in sources:
        raise ValueError('Source manifest must be ' + SOURCE_FORMAT + ' with the runner and matrix_models hashes')
    if set(sources) - {rc.SELF, *rc.MODEL_SOURCES} or not all(isinstance(v, str) and HEX64.fullmatch(v)
                                                              for v in sources.values()):
        raise ValueError('Source manifest names unknown files or malformed hashes')
    return dict(sources), {'kind': 'source_manifest', 'path': str(Path(path).resolve()),
                           'sha256': rc.sha256_file(path), 'label': data.get('label')}


def export_rows(export, config, manifest_sha256):
    """dev/mlb_dev keys, labels and game ids from the pinned export (hash-checked files only)."""
    manifest = rc.verify_export(export, config, manifest_sha256, hash_arrays=False)
    rows, hashes = {}, {'train/rows_sha256': manifest['splits']['train']['rows_sha256']}
    for split in SPLITS:
        record = manifest['splits'][split]
        rows[split] = {}
        for name in EXPORT_ARRAYS:
            meta = record['files'][name]
            if rc.sha256_file(Path(export) / meta['path']) != meta['sha256']:
                raise ValueError(f'{split}/{name}: export array hash differs from the manifest')
            rows[split][name] = np.load(Path(export) / meta['path'], allow_pickle=False)
            hashes[f'{split}/{name}'] = meta['sha256']
        r = rows[split]
        if not (len(r['keys']) == len(r['y']) == len(r['game_pk']) == record['n']) \
                or not np.array_equal(r['keys'][:, 0], r['game_pk']):
            raise ValueError(f'{split}: export keys, labels and game ids are misaligned')
    return rows, hashes


def check_train_subset(config, arm, seed, fit, train_rows_sha256):
    """Data arms: the recorded subset is the config rule over the export TRAIN rows; others: full TRAIN."""
    spec, record = rc.arm_spec(config, arm), fit.get('train_subset')
    if 'train_fraction' not in spec:
        if record is not None or fit.get('train_rows_sha256') != train_rows_sha256:
            raise ValueError(f'{arm}/{seed}: full-TRAIN arm did not fit on the export TRAIN rows')
        return None
    expected = {'method': 'nested_key_hash', 'salt': config['train_subset']['salt'], 'fraction': spec['train_fraction'],
                'source_rows_sha256': train_rows_sha256}
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()) \
            or record.get('selected_rows_sha256') != fit.get('train_rows_sha256') \
            or not 0 < int(record.get('n_selected', 0)) < int(record.get('n_source', 0)):
        raise ValueError(f'{arm}/{seed}: TRAIN subset provenance differs from the config rule')
    return {k: record[k] for k in ('fraction', 'n_source', 'n_selected', 'selected_rows_sha256', 'selected_mask_sha256')}


def verify_family(family, config, manifest_sha256, sources, train_rows_sha256=None):
    """Every arm x seed finished with the expected identity; nothing is read from predictions here."""
    family = Path(family)
    if family.name.endswith('-smoke'):
        raise ValueError('Smoke families are never scored')
    arms = [a['name'] for a in config['arms']]
    missing = [f'{a}/{s}' for a in arms for s in config['seeds']
               if not (rc.task_dir(family, a, s, False) / 'state.json').is_file()]
    if missing:
        raise ValueError('Incomplete family; nothing scored. Missing: ' + ', '.join(missing))
    tasks, signature = {}, None
    for arm in arms:
        tasks[arm] = {}
        for seed in config['seeds']:
            dest = rc.task_dir(family, arm, seed, False)
            state = rc.verify_task(dest, rc.expected_identity(config, manifest_sha256, arm, seed, False, 'cuda', sources))
            fit = rc.read_json(dest / 'fit.json')
            runtime = fit.get('runtime') or {}
            seen = {k: runtime.get(k) for k in RUNTIME_SIGNATURE}
            if fit.get('device') != 'cuda' or seen['torch_cuda'] is None or seen['device_name'] is None:
                raise ValueError(f'{arm}/{seed}: not a CUDA fit')
            if signature is None:
                signature = seen
            elif seen != signature:
                raise ValueError(f'{arm}/{seed}: CUDA runtime differs within the family')
            subset = check_train_subset(config, arm, seed, fit, train_rows_sha256) \
                if train_rows_sha256 is not None else None
            tasks[arm][seed] = {'state_sha256': rc.sha256_file(dest / 'state.json'), 'train_subset': subset,
                                'device_uuid': runtime.get('device_uuid'),
                                'delivery_temperature': rc.read_json(dest / 'calibration.json')['delivery_temperature']}
    return tasks, signature


def member(family, arm, seed, split, rows):
    """Calibrated probabilities of one member after exact key/label/game order checks."""
    with np.load(rc.task_dir(family, arm, seed, False) / SPLITS[split], allow_pickle=False) as data:
        for name, ours in (('keys', '_keys'), ('y', '_y'), ('game_pk', '_game_pk')):
            if not np.array_equal(data[split + ours], rows[name]):
                raise ValueError(f'{arm}/{seed}/{split}: {name} differ from the export rows')
        if not (data[split + '_delivery_level'] == rc.NO_PHYSICS_LEVEL).all():
            raise ValueError(f'{arm}/{seed}/{split}: not the no-physics direct head')
        probabilities = np.asarray(data[split], dtype=np.float64)
    pitch_losses(rows['y'], probabilities)  # shape / mass / range contract
    return probabilities


def summary(metrics):
    return {k: metrics[k] for k in ('n', 'log_loss', 'brier_multiclass', 'accuracy', 'top_label_ece10')}


def compare(y, games, candidate, control):
    return paired_game_comparison(y, candidate, control, games, draws=DRAWS, seed=BOOTSTRAP_SEED)


FACTOR_FREE = ('name', 'priority', 'combo', 'family', 'model', 'workload', 'context_continuous')


def check_contrasts(config):
    """Config-owned factorial contrasts (or None): distinct registered arms differing in exactly one factor.

    Validated before any prediction is read.  A contrast pairs a joint arm with its
    same-architecture no-workload arm (isolates workload) or its same-workload flatten
    arm (isolates architecture); data / pitcher-information arms never enter."""
    contrasts = config.get('comparison', {}).get('factorial_contrasts')
    if contrasts is None:
        return None
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError('factorial_contrasts must be a nonempty list')
    names, pairs = set(), set()
    for c in contrasts:
        if not isinstance(c, dict) or set(c) != {'name', 'candidate', 'reference', 'isolates'}:
            raise ValueError('Contrast needs exactly name / candidate / reference / isolates')
        cand, ref = rc.arm_spec(config, c['candidate']), rc.arm_spec(config, c['reference'])
        if c['candidate'] == c['reference'] or c['name'] in names or (c['candidate'], c['reference']) in pairs:
            raise ValueError('Contrast arms must be distinct and contrasts unique: ' + str(c['name']))
        rest = [{k: v for k, v in a.items() if k not in FACTOR_FREE} for a in (cand, ref)]
        if rest[0] != rest[1] or any('train_fraction' in a or 'drop_context' in a for a in (cand, ref)):
            raise ValueError(c['name'] + ': contrast arms differ beyond architecture / workload')
        same_model, same_workload = cand['model'] == ref['model'], cand['workload'] == ref['workload']
        isolates = 'workload' if same_model and not same_workload else \
            'architecture' if same_workload and not same_model and ref['model'] == 'flatten_mlp' else None
        if isolates is None or c['isolates'] != isolates:
            raise ValueError(c['name'] + ': contrast must change exactly the declared factor')
        names.add(c['name'])
        pairs.add((c['candidate'], c['reference']))
    return contrasts


def paired_block(y, games, candidate, reference, seeds):
    return {'ensemble': compare(y, games, np.mean(candidate, axis=0), np.mean(reference, axis=0)),
            'seeds': {str(s): compare(y, games, p, q) for s, p, q in zip(seeds, candidate, reference)}}


def score_family(config, family, rows, seeds, contrasts=None, regularization=None, workload_dropout=None, tabm=None):
    arms = [a['name'] for a in config['arms']]
    control_arm = config.get('comparison', {}).get('control_arm', 'control')
    if control_arm != 'control' or 'control' not in arms:
        raise ValueError('The same-family CUDA control is the only comparator')
    out = {split: {} for split in SPLITS}
    factorial = {split: {} for split in SPLITS} if contrasts else None
    paired = {split: {} for split in SPLITS} if regularization else None
    joint = {split: {} for split in SPLITS} if workload_dropout else None
    head = {split: {} for split in SPLITS} if tabm else None
    for split, r in rows.items():
        y, games = r['y'], r['game_pk']
        load = lambda arm: [member(family, arm, s, split, r) for s in seeds]  # noqa: E731  (bounded memory: reload)
        control = load(control_arm)
        out[split][control_arm] = {'role': 'control', 'ensemble': prediction_metrics(y, np.mean(control, axis=0)),
                                   'seeds': {str(s): summary(prediction_metrics(y, p)) for s, p in zip(seeds, control)}}
        for arm in arms:
            if arm == control_arm:
                continue
            candidate = load(arm)
            out[split][arm] = {
                'role': 'arm', 'ensemble': prediction_metrics(y, np.mean(candidate, axis=0)),
                'paired_vs_control': paired_block(y, games, candidate, control, seeds),
                'seeds': {str(s): summary(prediction_metrics(y, p)) for s, p in zip(seeds, candidate)}}
        for c in contrasts or ():
            factorial[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                           'isolates': c['isolates'],
                                           **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in regularization or ():
            paired[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                        'architecture': c['architecture'], 'isolates': c['isolates'],
                                        **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in workload_dropout or ():
            joint[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                       'isolates': c['isolates'], 'held': c['held'],
                                       **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in tabm or ():
            head[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                      'isolates': c['isolates'], 'held': c['held'],
                                      **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
    if tabm:
        return out, factorial, paired, joint, head
    if workload_dropout:
        return out, factorial, paired, joint
    return out, factorial, paired


def korean_table(result):
    def ci(block, key):
        b = block[key]
        lo, hi = b['ci95'] or (float('nan'), float('nan'))
        return f"{b['delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"
    lines = [f"# {result['family']} 직접 헤드 기술 비교 (보정 확률, 혼합 전)", '',
             f'- 범위: 시드 {len(result.get("seed_block", {}).get("seeds", [0, 1, 2]))}개 평균 보정 확률 vs 같은 가족 CUDA 대조군. 6월 블렌드·G0 혼합·G15·정책 아님.',
             '- 음수 = 대조군보다 손실이 낮음. 경기 단위 부트스트랩 10,000회 95% 구간. 결정·순위·승자 없음(설정 순서).',
             f"- config {result['provenance']['config']['file_sha256'][:12]} · export "
             f"{result['provenance']['export']['manifest_sha256'][:12]} · 과제 소스 {result['provenance']['task_sources']['kind']}",
             '']
    for split, arms in result['splits'].items():
        control = arms['control']['ensemble']
        lines += [f"## {split} (n={control['n']}, 대조군 NLL {control['log_loss']:.5f}, Brier {control['brier_multiclass']:.5f})",
                  '', '| 팔 | ΔNLL [95%] | ΔBrier [95%] | 시드별 ΔNLL |', '|---|---|---|---|']
        for arm, block in arms.items():
            if block['role'] == 'control':
                continue
            paired = block['paired_vs_control']
            seeds = ' / '.join(f"{v['nll']['delta']:+.4f}" for v in paired['seeds'].values())
            lines.append(f"| {arm} | {ci(paired['ensemble'], 'nll')} | {ci(paired['ensemble'], 'brier')} | {seeds} |")
        lines.append('')
    for split, contrasts in (result.get('regularization_contrasts') or {}).items():
        lines += [f'## {split} 헤드 드롭아웃 추가 대비 (추가한 모델 − 기존 구조, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'ResNet 기준 구조에는 블록 드롭아웃 0.1이 이미 있습니다. 이 대비는 출력 헤드에 0.1을 더한 효과입니다.', '',
                  '| 대비 | 구조 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|']
        for name, block in contrasts.items():
            lines.append(f"| {name} | {block['architecture']} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('workload_dropout_contrasts') or {}).items():
        lines += [f'## {split} 작업량×헤드 드롭아웃 대비 (한 요인만 바꾼 짝, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('tabm_contrasts') or {}).items():
        lines += [f'## {split} TabM-mini 헤드·헤드 드롭아웃 대비 (한 요인만 바꾼 짝, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'ensemble_head 행은 TabM-mini 1회 학습(구성원 8)과 단일 신경망 1회 학습의 비교입니다. 연산량은 같지 않습니다. '
                  'head_dropout 행은 단일 신경망끼리의 비교입니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). counts 작업량은 P27 결과를 읽은 뒤 고른 사후 선택입니다.', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('factorial_contrasts') or {}).items():
        lines += [f'## {split} 요인 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  '| 대비 | 분리 요인 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|']
        for name, block in contrasts.items():
            lines.append(f"| {name} | {block['isolates']} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    return '\n'.join(lines)


def score(config_path, config_sha256, export, manifest_sha256, family, destination, source_manifest=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} exists; scores are immutable')
    if rc.sha256_file(config_path) != config_sha256:
        raise ValueError('Config differs from the pinned SHA256')
    config = rc.config_check(rc.read_json(config_path))
    if config.get('status') != 'REGISTERED' or config.get('execution', {}).get('enabled') is not True:
        raise ValueError('Only a REGISTERED, enabled family has real tasks to score')
    contrasts = check_contrasts(config)
    regularization = dropout_registry.check_contrasts(config)
    workload_dropout = workload_dropout_registry.check_contrasts(config)
    tabm = tabm_registry.check_contrasts(config)
    if rc.uses_tabm(config):
        if tabm is None or workload_dropout is not None:
            raise ValueError('The TabM-mini family is scored only with its fixed contrasts')
    elif rc.uses_workload_dropout(config) and workload_dropout is None:
        raise ValueError('The workload-dropout family is scored only with its fixed contrasts')
    sources, source_record = task_sources(source_manifest)
    rows, export_hashes = export_rows(export, config, manifest_sha256)
    tasks, signature = verify_family(family, config, manifest_sha256, sources,
                                     export_hashes['train/rows_sha256'] if config.get('train_subset') else None)
    splits, factorial, paired, *joint = score_family(config, family, rows, config['seeds'], contrasts, regularization,
                                                     workload_dropout, tabm)
    result = {'format': FORMAT, 'family': config['experiment_id'], 'scope': SCOPE, 'decision': None, 'ranking': None,
              'arm_order': 'config order', 'splits': splits,
              **({'factorial_contrasts': factorial} if contrasts else {}),
              **({'regularization_contrasts': paired} if regularization else {}),
              **({'workload_dropout_contrasts': joint[0], 'seed_block': {'seeds': config['seeds'],
                  'repeat_tag': config.get('repeat_tag'),
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test'}} if workload_dropout else {}),
              **({'tabm_contrasts': joint[1], 'seed_block': {'seeds': config['seeds'], 'repeat_tag': None,
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; workload counts chosen after reading P27 '
                          '(post-selection); one TabM-mini fit (k=8) vs one single-network fit is not '
                          'compute-matched'}} if tabm else {}),
              'limits': ['Cpanel DEV and mlb_dev already exposed: repeated-DEV, descriptive only',
                         'Bootstrap conditions on fixed predictions (no refit, calibration or selection uncertainty)',
                         'Prediction only; no 2026 rows; no policy or service claim'],
              'provenance': {
                  'config': {'path': str(Path(config_path).resolve()), 'file_sha256': config_sha256,
                             'digest': rc.config_digest(config), 'experiment_id': config['experiment_id']},
                  'export': {'path': str(Path(export).resolve()), 'manifest_sha256': manifest_sha256,
                             'scored_arrays_sha256': export_hashes},
                  'task_sources': {**source_record, 'source_sha256': sources},
                  'task_runtime_signature': signature, 'tasks': {a: {str(s): v for s, v in t.items()}
                                                                 for a, t in tasks.items()},
                  'scorer_sources_sha256': scorer_sources(),
                  'scorer_runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                                     'hostname': platform.node()},
                  'bootstrap': {'draws': DRAWS, 'seed': BOOTSTRAP_SEED, 'function': 'matrix_metrics.paired_game_comparison'},
                  'created_utc': datetime.now(timezone.utc).isoformat()}}
    if scorer_sources() != result['provenance']['scorer_sources_sha256']:
        raise RuntimeError('Scorer sources changed during scoring')
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
    work.mkdir()
    rc.write_json_atomic(work / 'score.json', result)
    (work / 'table_ko.md').write_text(korean_table(result) + '\n')
    (work / 'SHA256SUMS').write_text(''.join(f'{rc.sha256_file(work / n)}  {n}\n' for n in ('score.json', 'table_ko.md')))
    for name in ('score.json', 'table_ko.md', 'SHA256SUMS'):
        os.chmod(work / name, 0o444)
    if destination.exists():
        raise FileExistsError(f'{destination} appeared during scoring')
    os.rename(work, destination)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--family', type=Path, required=True, help='runner --output directory (arm/seed/...)')
    p.add_argument('--destination', type=Path, required=True, help='new directory; must not exist')
    p.add_argument('--source-manifest', type=Path,
                   help=f'{SOURCE_FORMAT} JSON of the EXACT task source map (default: current runner own_hashes)')
    args = p.parse_args(argv)
    score(args.config, args.config_sha256, args.export, args.export_manifest_sha256, args.family, args.destination,
          args.source_manifest)
    print('CUDA_DIRECT_SCORED', args.destination, flush=True)


if __name__ == '__main__':
    main()

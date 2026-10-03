"""Synthetic tests: frontier arms in the CUDA runner and the portable CUDA scorer (CPU only, no real data)."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import export_cuda_direct as ex  # noqa: E402
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
from pitchmdp import direct_frontier_registry as reg  # noqa: E402

BASE = REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json'
RUN = REPO / 'configs' / 'DRAFT-CUDA-FRONTIER-RUN-20261004.json'
REAL_MANIFEST = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/CUDA-EXPORT-20261004/manifest.json')
REAL_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
SHAPES = {'length': 6, 'n_token': 38, 'n_context': 52, 'n_workload': 4}
COLUMNS = ['prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing']
INFORMATION = ('no_batter', 'no_batter_cluster', 'no_history')
RUNTIME = {'torch': '2.x', 'torch_cuda': '12.x', 'cudnn': 90100, 'numpy': '2.x', 'scipy': '1.x',
           'device_name': 'NVIDIA RTX A5000', 'device_capability': [8, 6], 'device_total_memory': 25438126080}


def load(path=RUN, registered=False, unpin=True):
    cfg = json.loads(Path(path).read_text())
    if unpin:
        cfg['export'].pop('manifest_sha256', None)  # synthetic exports have their own manifest
    if registered:
        cfg['status'], cfg['execution'] = 'REGISTERED', {'enabled': True}
    return rc.config_check(cfg)


def lineage(cfg, **change):
    layout = {**cfg['export']['context_layout'], **change}
    return {'synthetic': True, 'context_layout': {
        'groups': layout['groups'], 'pitcher_profile_columns': layout['pitcher_profile_columns'],
        'stripped_tail': layout['stripped_tail'],
        'base_report': {'features': layout['base_features'], 'archetypes': {
            'style_columns': layout['style_columns'], 'reliability_columns': layout['reliability_columns'],
            'n_clusters': layout['n_clusters']}}}}


def split_arrays(n, seed, game0):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 38)).astype(np.float32)
    tokens[:, -1, :8] = 0.
    tokens[:, -1, -11:] = 0.
    valid = np.ones((n, 6), dtype=bool)
    valid[:, 0] = rng.random(n) > .5
    game = game0 + np.arange(n) // 10
    keys = np.column_stack([game, np.arange(n) % 10 // 4 + 1, np.arange(n) % 4 + 1]).astype(np.int64)
    return {'tokens': tokens, 'valid': valid, 'context': rng.normal(size=(n, 52)).astype(np.float32),
            'workload': rng.random((n, 4)).astype(np.float32), 'keys': keys,
            'y': rng.integers(0, 10, n).astype(np.int64), 'game_pk': game.astype(np.int64),
            'pitcher': rng.integers(1, 5, n).astype(np.int64), 'date': np.full(n, 20240601, dtype=np.int32)}


def make_export(root, lin):
    sizes = {'train': 400, 'earlystop': 120, 'temperature': 80, 'blend': 40, 'dev': 60, 'mlb_dev': 50}
    splits = {s: split_arrays(n, i, 1000 * (i + 1)) for i, (s, n) in enumerate(sizes.items())}
    return root, ex.write_export(root, splits, lin, SHAPES, COLUMNS), splits


@pytest.fixture
def export(tmp_path):
    return make_export(tmp_path / 'export', lineage(load()))


# ------------------------------------------------------------------ config

def test_run_config_is_draft_base_plus_fresh_control_and_eight_frontier_arms():
    raw, base = json.loads(RUN.read_text()), json.loads(BASE.read_text())
    cfg = load(unpin=False)
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and cfg['fits'] == 27
    assert raw['export']['manifest_sha256'] == REAL_SHA
    assert [a['name'] for a in cfg['arms']] == ['control', *reg.ARMS] and cfg['arms'][0] == base['arms'][0]
    for key in ('protocol', 'training', 'calibration', 'device', 'seeds', 'data_rules'):
        assert cfg[key] == base[key]
    assert {k: v for k, v in cfg['export'].items() if k not in ('manifest_sha256', 'context_layout',
                                                                 'context_layout_source', 'context_names')} == base['export']
    names = cfg['export']['context_names']
    assert names == reg.direct_context_names(cfg['export']['context_layout']['pitcher_profile_columns'])
    assert len(names) == 52 and names[:28] == list(reg.BASE_CONTEXT_NAMES) and names[-1] == reg.SUPPORT_NAME
    assert [names.index(c) for c in reg.ARMS['no_batter']['drop_context']] == list(range(11, 28))
    assert [names.index(c) for c in reg.ARMS['no_batter_cluster']['drop_context']] == list(range(23, 28))
    for source in cfg['source_configs'].values():
        assert rc.sha256_file(REPO / source['path']) == source['sha256']
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.require_executable(cfg, smoke=False)


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason='T7 export manifest (metadata only) not mounted')
def test_names_match_the_real_export_manifest_metadata():
    assert rc.sha256_file(REAL_MANIFEST) == REAL_SHA
    manifest = json.loads(REAL_MANIFEST.read_text())
    layout = manifest['lineage']['context_layout']
    assert layout['groups'] == {'base': [0, 28], 'pitcher_profile': [28, 51], 'pitcher_support': [51, 52]}
    assert layout['base_report']['features'][:11] == list(reg.GAME_STATE)
    assert layout['base_report']['features'][11:] == rc.BASE_GROUPED_FEATURES
    cfg = load(unpin=False)
    assert rc.verified_context_names(cfg, manifest) == cfg['export']['context_names']


def test_base_config_unchanged_and_frontier_arms_strict():
    base = rc.config_check(json.loads(BASE.read_text()))
    assert not any('frontier' in a for a in base['arms'])
    cfg = load()

    def broken(edit):
        bad = copy.deepcopy(cfg)
        edit(bad)
        with pytest.raises((ValueError, KeyError)):
            rc.config_check(bad)
    by = lambda c, n: next(a for a in c['arms'] if a['name'] == n)  # noqa: E731
    broken(lambda c: by(c, 'width145').update(width=146))
    broken(lambda c: by(c, 'width203').pop('frontier'))          # base checks reject width 203
    broken(lambda c: by(c, 'dropout').pop('frontier'))           # base checks reject an unknown model
    broken(lambda c: by(c, 'no_history').update(frontier=False))
    broken(lambda c: by(c, 'no_batter').update(extra=1))
    broken(lambda c: by(c, 'weight_decay').update(workload=['prior_records_per100']))
    broken(lambda c: by(c, 'weight_decay').update(weight_decay=1e-3))
    broken(lambda c: by(c, 'no_batter_cluster').update(drop_context=reg.ARMS['no_batter_cluster']['drop_context'][:4]))
    broken(lambda c: c['arms'].append({**by(c, 'width145'), 'name': 'width300'}))
    broken(lambda c: c['arms'].append(copy.deepcopy(by(c, 'dropout'))))
    broken(lambda c: c['arms'][0].update(frontier=True))         # control is not a registry arm
    broken(lambda c: c['export']['context_names'].reverse())
    broken(lambda c: c['export']['context_layout']['groups'].update(base=[0, 27]))
    broken(lambda c: c['export']['context_layout']['base_features'].reverse())
    broken(lambda c: c['export']['context_layout'].update(n_clusters=4))
    broken(lambda c: c['export'].pop('context_names'))
    broken(lambda c: c['arms'].pop(0))                           # no control
    base_bad = json.loads(BASE.read_text())
    base_bad['arms'][1]['width'] = 145
    with pytest.raises(ValueError):
        rc.config_check(base_bad)


def test_frontier_modules_are_pinned_sources_and_config_check_needs_no_torch():
    assert rc.MODEL_SOURCES[-2:] == ['pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py']
    assert set(rc.own_hashes()) == {rc.SELF, *rc.MODEL_SOURCES}
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'rc.config_check(json.load(open({str(RUN)!r}))); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ runner: every frontier arm

@pytest.mark.parametrize('arm', ['control', *reg.ARMS])
def test_frontier_task_train_predict_save_load_cpu(export, tmp_path, arm):
    path, digest, splits = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'f-smoke', arm, 1, 'cpu', {'test': True}, smoke=True)
    good = rc.expected_identity(cfg, digest, arm, 1, True, 'cpu', rc.own_hashes())
    state = rc.verify_task(dest, good)
    assert state['identity']['arm'] == rc.arm_spec(cfg, arm)
    assert set(state['identity']['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    for change in ({'seed': 2}, {'smoke': False}, {'device': 'cuda'}, {'export_manifest_sha256': '0' * 64},
                   {'arm': rc.arm_spec(cfg, 'width145' if arm != 'width145' else 'dropout')},
                   {'source_sha256': {**rc.own_hashes(), 'pitchmdp/direct_frontier_model.py': '0' * 64}}):
        with pytest.raises(ValueError, match='identity'):
            rc.verify_task(dest, {**good, **change})
    fit = json.loads((dest / 'fit.json').read_text())
    spec = rc.arm_spec(cfg, arm)
    assert fit['context_width'] == 52 and fit['context_names_verified'] is (arm in INFORMATION)
    width = spec['width']
    expected_params = reg.flatten_parameter_count(width)
    assert fit['parameter_count'] == expected_params  # dropout/masks keep the flatten parameter count
    names = cfg['export']['context_names'] if arm in INFORMATION else None
    model = rc.load_model(spec, dest / 'model.pt', 'cpu', names)
    manifest = rc.read_json(path / 'manifest.json')
    temperature = json.loads((dest / 'calibration.json').read_text())['delivery_temperature']
    with np.load(dest / 'predictions.npz') as data:
        assert np.array_equal(data['dev_keys'], splits['dev']['keys']) and np.array_equal(data['dev_y'], splits['dev']['y'])
        part = rc.load_split(path, manifest, 'dev')
        again, _ = rc.probabilities(rc.g0_log_probs(model.logits(rc.arm_inputs(part, spec, COLUMNS))), temperature)
        assert np.allclose(again, data['dev'], atol=1e-6)
    if arm != 'control':
        other = rc.arm_spec(cfg, 'no_batter_cluster' if arm != 'no_batter_cluster' else 'dropout')
        with pytest.raises(ValueError):
            rc.load_model(other, dest / 'model.pt', 'cpu', cfg['export']['context_names'])
    if arm in INFORMATION:
        with pytest.raises(ValueError, match='context names'):
            rc.load_model(spec, dest / 'model.pt', 'cpu', list(reversed(cfg['export']['context_names'])))
        with pytest.raises(ValueError, match='context names'):
            rc.build_model(spec, 0, 'cpu', None)
    with pytest.raises(FileExistsError):
        rc.run_task(cfg, path, digest, tmp_path / 'f-smoke', arm, 1, 'cpu', {}, smoke=True)


@pytest.mark.parametrize('change', [
    {'pitcher_profile_columns': None}, {'groups': {'base': [0, 28], 'pitcher_profile': [28, 50], 'pitcher_support': [50, 51]}},
    {'base_features': None}, {'style_columns': None}, {'n_clusters': 6}, 'missing'])
def test_information_arm_refuses_other_export_layout(tmp_path, change):
    cfg = load()
    layout = cfg['export']['context_layout']
    if change == 'missing':
        lin = {'synthetic': True}
    else:
        key, value = next(iter(change.items()))
        if value is None:
            value = list(reversed(layout[key]))
        lin = lineage(cfg, **{key: value})
    path, digest, _ = make_export(tmp_path / 'export', lin)
    with pytest.raises(ValueError, match='context layout'):
        rc.run_task(cfg, path, digest, tmp_path / 'l-smoke', 'no_batter', 0, 'cpu', {}, smoke=True)
    rc.run_task(cfg, path, digest, tmp_path / 'l-smoke', 'width145', 0, 'cpu', {}, smoke=True)  # names unused


def test_config_export_pin_is_enforced(export, tmp_path):
    path, digest, _ = export
    with pytest.raises(ValueError, match='config pin'):
        rc.run_task(load(unpin=False), path, digest, tmp_path / 'p-smoke', 'control', 0, 'cpu', {}, smoke=True)


# ------------------------------------------------------------------ portable scorer

def fake_task(family, cfg, digest, arm, seed, splits, sources, smoke=False, probs=None, runtime=None):
    """A finished task directory exactly as run_task writes it (without a fit)."""
    dest = rc.task_dir(family, arm, seed, smoke)
    dest.mkdir(parents=True)
    identity = rc.expected_identity(cfg, digest, arm, seed, smoke, 'cuda', sources)
    rc.write_json_atomic(dest / 'identity.json', identity)
    (dest / 'model.pt').write_bytes(b'not a model')
    rc.write_json_atomic(dest / 'calibration.json', {'delivery_temperature': 1.1})
    for archive, names in (('predictions.npz', ('blend', 'dev')), ('mlb_predictions.npz', ('mlb_dev',))):
        values = {}
        for split in names:
            n = len(splits[split]['y'])
            p = (probs or {}).get(split)
            if p is None:
                p = np.random.default_rng(zlib.crc32(f'{arm}/{seed}/{split}'.encode())).dirichlet(np.ones(10), n)
            values.update({split: p, split + '_raw': p, split + '_delivery_level': np.full(n, -2, dtype=np.int64),
                           split + '_keys': splits[split]['keys'], split + '_y': splits[split]['y'],
                           split + '_game_pk': splits[split]['game_pk'], split + '_pitcher': splits[split]['pitcher']})
        np.savez_compressed(dest / archive, **values)
    rc.write_json_atomic(dest / 'fit.json', {'device': 'cuda', 'runtime': {**RUNTIME, **(runtime or {}),
                                                                            'device_uuid': 'GPU-x'}})
    reseal(dest, identity)
    return dest


def reseal(dest, identity=None):
    names = ['identity.json', 'model.pt', 'calibration.json', 'predictions.npz', 'mlb_predictions.npz', 'fit.json']
    identity = identity or json.loads((dest / 'identity.json').read_text())
    rc.write_json_atomic(dest / 'state.json', {'identity': identity,
                                               'artifact_hashes': {n: rc.sha256_file(dest / n) for n in names}})


def family(tmp_path, cfg, digest, splits, sources=None, skip=(), same_as_control=('width145',), name='fam'):
    root = tmp_path / name
    sources = sources or rc.own_hashes()
    control = {s: {split: np.random.default_rng(100 + s).dirichlet(np.ones(10), len(splits[split]['y']))
                   for split in ('blend', 'dev', 'mlb_dev')} for s in cfg['seeds']}
    for arm in [a['name'] for a in cfg['arms']]:
        for seed in cfg['seeds']:
            if (arm, seed) not in skip:
                probs = control[seed] if arm == 'control' or arm in same_as_control else None
                fake_task(root, cfg, digest, arm, seed, splits, sources, probs=probs)
    return root


def write_cfg(tmp_path, cfg, name='cfg.json'):
    path = tmp_path / name
    path.write_text(json.dumps(cfg))
    return path, rc.sha256_file(path)


def run_score(tmp_path, cfg, export, fam, dest='score', source_manifest=None):
    path, digest, _ = export
    cpath, csha = write_cfg(tmp_path, cfg)
    return sc.score(cpath, csha, path, digest, fam, tmp_path / dest, source_manifest)


def test_scorer_frontier_family_identical_control_is_zero_and_output_immutable(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = family(tmp_path, cfg, digest, splits)
    result = run_score(tmp_path, cfg, export, fam)
    assert result['decision'] is None and result['ranking'] is None
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == [a['name'] for a in cfg['arms']]  # config order, never sorted by score
        zero = arms['width145']['paired_vs_control']
        for block in (zero['ensemble'], *zero['seeds'].values()):
            assert block['status'] == 'measured' and block['draws'] == 10000
            assert block['nll']['delta'] == 0 and block['nll']['ci95'] == [0, 0] and block['brier']['delta'] == 0
        assert arms['width145']['ensemble']['log_loss'] == arms['control']['ensemble']['log_loss']
        other = arms['dropout']['paired_vs_control']['ensemble']
        assert other['nll']['delta'] != 0 and other['games'] == len(np.unique(splits[split]['game_pk']))
    prov = result['provenance']
    assert prov['task_sources']['kind'] == 'current_runner_own_hashes' and prov['export']['manifest_sha256'] == digest
    assert set(prov['scorer_sources_sha256']) == set(sc.SCORER_SOURCES)
    out = tmp_path / 'score'
    assert sorted(p.name for p in out.iterdir()) == ['SHA256SUMS', 'score.json', 'table_ko.md']
    assert all(not os.access(out / n, os.W_OK) for n in ('score.json', 'table_ko.md'))
    sums = dict(line.split('  ')[::-1] for line in (out / 'SHA256SUMS').read_text().splitlines())
    assert sums == {n: rc.sha256_file(out / n) for n in ('score.json', 'table_ko.md')}
    assert '| width145 | +0.00000 [+0.00000, +0.00000]' in (out / 'table_ko.md').read_text()
    with pytest.raises(FileExistsError):
        run_score(tmp_path, cfg, export, fam)


def test_scorer_refuses_incomplete_family_before_reading(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = family(tmp_path, cfg, digest, splits, skip={('no_history', 2)})
    with pytest.raises(ValueError, match='Incomplete family.*no_history/2'):
        run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


@pytest.mark.parametrize('field', ['keys', 'y', 'game_pk', 'order'])
def test_scorer_refuses_key_label_game_mismatch(export, tmp_path, field):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = family(tmp_path, cfg, digest, splits)
    dest = rc.task_dir(fam, 'no_batter', 1, False)
    with np.load(dest / 'mlb_predictions.npz') as data:
        values = {k: data[k].copy() for k in data.files}
    if field == 'order':
        perm = np.random.default_rng(0).permutation(len(values['mlb_dev']))
        values = {k: (v[perm] if k.startswith('mlb_dev') else v) for k, v in values.items()}
    else:
        values[f'mlb_dev_{field}'][0] += 1
    np.savez_compressed(dest / 'mlb_predictions.npz', **values)
    reseal(dest)  # self-consistent artifact: only the row check can catch it
    with pytest.raises(ValueError, match='differ from the export rows'):
        run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


def test_scorer_refuses_tamper_smoke_draft_runtime_and_wrong_config(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = family(tmp_path, cfg, digest, splits)
    target = rc.task_dir(fam, 'dropout', 0, False) / 'predictions.npz'
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match='artifact hash'):
        run_score(tmp_path, cfg, export, fam, 's1')
    target.write_bytes(original)
    with pytest.raises(ValueError, match='REGISTERED'):
        run_score(tmp_path, load(), export, fam, 's2')
    smoke = family(tmp_path, cfg, digest, splits, name='fam-smoke')
    with pytest.raises(ValueError, match='Smoke'):
        run_score(tmp_path, cfg, export, smoke, 's3')
    smoke_ids = tmp_path / 'fam2'
    for arm in [a['name'] for a in cfg['arms']]:  # smoke identities moved under real task paths
        for seed in cfg['seeds']:
            fake_task(smoke_ids, cfg, digest, arm, seed, splits, rc.own_hashes(), smoke=True)
        os.rename(smoke_ids / (arm + '-smoke'), smoke_ids / arm)
    with pytest.raises(ValueError, match='identity smoke'):
        run_score(tmp_path, cfg, export, smoke_ids, 's4')
    drift = rc.task_dir(fam, 'weight_decay', 2, False)
    fit = json.loads((drift / 'fit.json').read_text())
    fit['runtime']['torch'] = '9.9'
    rc.write_json_atomic(drift / 'fit.json', fit)
    reseal(drift)
    with pytest.raises(ValueError, match='runtime differs'):
        run_score(tmp_path, cfg, export, fam, 's5')
    other = copy.deepcopy(cfg)
    other['queue']['date'] = '2026-10-05'
    with pytest.raises(ValueError, match='identity config_sha256'):
        run_score(tmp_path, other, export, fam, 's6')
    cpath, _ = write_cfg(tmp_path, cfg, 'pin.json')
    with pytest.raises(ValueError, match='pinned'):
        sc.score(cpath, '0' * 64, path, digest, fam, tmp_path / 's7')
    with pytest.raises(ValueError, match='pinned'):
        sc.score(cpath, rc.sha256_file(cpath), path, 'f' * 64, fam, tmp_path / 's8')
    assert not any((tmp_path / f's{i}').exists() for i in range(1, 9))


def old_sources():
    current = rc.own_hashes()
    return {k: (v if k != rc.SELF else '1' * 64) for k, v in current.items()
            if k not in ('pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py')}


def test_scorer_base_family_with_pinned_old_source_manifest(tmp_path):
    base = json.loads(BASE.read_text())
    base['status'], base['execution'] = 'REGISTERED', {'enabled': True}
    rc.config_check(base)
    path, digest, splits = make_export(tmp_path / 'export', {'synthetic': True})  # base arms need no names
    fam = family(tmp_path, base, digest, splits, sources=old_sources(), same_as_control=('counts',))
    with pytest.raises(ValueError, match='identity source_sha256'):  # default = current (new) sources
        run_score(tmp_path, base, (path, digest, splits), fam, 'a')
    manifest = tmp_path / 'first-batch-sources.json'

    def write(sources, fmt=sc.SOURCE_FORMAT):
        manifest.write_text(json.dumps({'format': fmt, 'label': 'first batch edfd4cc', 'source_sha256': sources}))
        return manifest
    wrong = {**old_sources(), 'pitchmdp/matrix_models.py': '2' * 64}
    with pytest.raises(ValueError, match='identity source_sha256'):
        run_score(tmp_path, base, (path, digest, splits), fam, 'b', write(wrong))
    for bad in ({**old_sources(), 'scripts/other.py': '3' * 64}, {**old_sources(), rc.SELF: 'XYZ'},
                {k: v for k, v in old_sources().items() if k != rc.SELF}):
        with pytest.raises(ValueError, match='Source manifest'):
            run_score(tmp_path, base, (path, digest, splits), fam, 'c', write(bad))
    with pytest.raises(ValueError, match='Source manifest'):
        run_score(tmp_path, base, (path, digest, splits), fam, 'c', write(old_sources(), 'other'))
    result = run_score(tmp_path, base, (path, digest, splits), fam, 'd', write(old_sources()))
    assert list(result['splits']['dev']) == ['control', 'counts', 'block', 'residual', 'periodic']
    assert result['splits']['dev']['counts']['paired_vs_control']['ensemble']['nll']['delta'] == 0
    src = result['provenance']['task_sources']
    assert src['kind'] == 'source_manifest' and src['source_sha256'] == old_sources()
    assert src['sha256'] == rc.sha256_file(manifest)


def test_scorer_cli_runs_without_torch_pandas_scipy(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = family(tmp_path, cfg, digest, splits)
    cpath, csha = write_cfg(tmp_path, cfg)
    args = ['--config', str(cpath), '--config-sha256', csha, '--export', str(path), '--export-manifest-sha256', digest,
            '--family', str(fam), '--destination', str(tmp_path / 'cli')]
    code = ('import sys; sys.path.insert(0, "scripts"); import score_cuda_direct as s; s.main(sys.argv[1:]); '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code, *args], cwd=PROJECT, check=True, capture_output=True)
    assert json.loads((tmp_path / 'cli' / 'score.json').read_text())['family'] == cfg['experiment_id']

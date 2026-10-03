"""Synthetic tests for the DRAFT direct-head ensemble search (no real data)."""
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
import search_direct_head_ensemble as ens  # noqa: E402
from pitchmdp.data import hash_file  # noqa: E402
from pitchmdp.matrix_data import canonical_hash  # noqa: E402
from run_ml_matrix import artifact_hashes  # noqa: E402

CONFIG = REPO / 'configs' / 'DRAFT-HEAD-ENSEMBLE-66-20261003.json'


@pytest.fixture(scope='module')
def arrays():
    return ens.synthetic(n_blend=200, n_dev=300, n_mlb=1000, games=20)


@pytest.fixture(scope='module')
def reference(arrays):
    return ens.p19.jsonable(ens.reference_from(*arrays))


@pytest.fixture(scope='module')
def result(arrays, reference):
    return ens.search(*arrays, reference, 1e-9, 129)


# ------------------------------------------------------------------ grid
def test_grid_is_22_subsets_by_3_weights_with_control():
    subs = ens.subsets()
    assert len(subs) == len(set(subs)) == 22
    assert sum(len(s) == 1 for s in subs) == 6 and sum(len(s) == 2 for s in subs) == 15 and subs[-1] == ens.FAMILIES
    grid = ens.candidates()
    assert len(grid) == len(set(grid)) == len({ens.label(*k) for k in grid}) == 66
    assert ens.CONTROL in grid and ens.label(*ens.CONTROL) == 'P15-uniform_w0.50'
    assert ens.G0_KEY not in grid and ens.DIRECT_KEY not in grid


def test_draft_config_matches_runner_and_refuses_run():
    config = json.loads(CONFIG.read_text())
    ens.config_check(config, full=False)
    with pytest.raises(ValueError, match='REGISTERED'):
        ens.config_check(config, full=True)
    registered = {**copy.deepcopy(config), 'status': 'REGISTERED', 'experiment_id': 'EXP-P99-001'}
    registered['execution']['enabled'] = True
    with pytest.raises(ValueError, match='pinned before a run'):
        ens.config_check(registered, full=True)
    for change in ({'temperature': [1]}, {'direct_weight': [.25, .5]}, {'count': 77}):
        bad = copy.deepcopy(config)
        bad['grid'].update(change)
        with pytest.raises(ValueError):
            ens.config_check(bad, full=False)


# ------------------------------------------------------------------ ensemble semantics
def test_average_is_equal_weight_and_same_seed_only(arrays):
    direct = arrays[0]
    pair = ('P20-hl365', 'P22-periodic')
    out = ens.average(direct, pair)
    for i in range(len(ens.SEEDS)):
        for part in ens.PARTS:
            np.testing.assert_allclose(out[i][part], (direct[pair[0]][i][part] + direct[pair[1]][i][part]) / 2, rtol=0, atol=1e-15)
            # never another seed's member
            assert not np.allclose(out[i][part], (direct[pair[0]][i][part] + direct[pair[1]][(i + 1) % 3][part]) / 2)
    six = ens.average(direct, ens.FAMILIES)
    np.testing.assert_allclose(six[1]['dev'], np.mean([direct[f][1]['dev'] for f in ens.FAMILIES], 0), atol=1e-15)
    assert ens.average(direct, ('P15-uniform',)) is direct['P15-uniform']      # singleton: exact, no arithmetic


def test_candidate_member_mix_is_same_seed(arrays):
    direct, g0 = arrays[0], arrays[1]
    subset, w = ('P15-uniform', 'P21-residual'), .25
    members = ens.p19.candidate_members(ens.average(direct, subset), g0, w, 1)
    for i in range(3):
        d = (direct[subset[0]][i]['dev'] + direct[subset[1]][i]['dev']) / 2
        np.testing.assert_allclose(members[i]['dev'], w * d + (1 - w) * g0[i]['dev'], atol=1e-15)


def test_mlb_stream_matches_direct_computation(arrays):
    direct, g0, baseline, direct_mlb, g0_mlb, frequency, _, periods = arrays
    key, ew, sw = (('P20-hl180', 'P21-residual'), .75), .6, [.5, .7, .9]
    stats = ens.mlb_stream(direct_mlb, g0_mlb, frequency['mlb_dev'], [key], {key: (ew, sw)}, periods, 97)[key]
    y, f = g0_mlb[0]['dev_y'], frequency['mlb_dev']
    members = [.75 * (direct_mlb['P20-hl180'][i]['mlb_dev'] + direct_mlb['P21-residual'][i]['mlb_dev']) / 2
               + .25 * g0_mlb[i]['dev'] for i in range(3)]
    expect = ens.pitch_losses(y, ew * np.mean(members, 0) + (1 - ew) * f).mean(0)
    np.testing.assert_allclose(stats['mean'], expect, rtol=1e-12)
    np.testing.assert_allclose(stats['seed_mean'][2], ens.pitch_losses(y, .9 * members[2] + .1 * f).mean(0), rtol=1e-12)


# ------------------------------------------------------------------ baseline reproduction / result
def test_gate_reproduces_frozen_three_seed_baseline(result):
    gate = result['reproduction_gate']
    assert gate['passed'] and len(gate["checks"]) == 21
    assert max(c['max_abs_diff'] for c in gate['checks']) <= 1e-12


def test_gate_fails_on_perturbed_reference(arrays, reference):
    bad = copy.deepcopy(reference)
    bad['mlb_dev_blended']['comparison']['G15-direct-g0-mix']['candidate']['log_loss'] += 1e-6
    out = ens.search(*arrays, bad, 1e-9, 129)
    assert not out['reproduction_gate']['passed'] and 'candidates' not in out


def test_all_66_retained_control_row_is_zero_delta(result):
    rows = result['candidates']
    assert result['candidate_count'] == len(rows) == 66 and len(result['order_by_mean_nll_rank']) == 66
    ctrl = rows['P15-uniform_w0.50']
    assert ctrl['is_control'] and sum(r['is_control'] for r in rows.values()) == 1
    for pop in ('cpanel', 'mlb'):
        assert ctrl[pop]['delta_nll_vs_G15'] == 0 and ctrl[pop]['delta_brier_vs_G15'] == 0
        assert ctrl[pop]['seed_delta_nll_vs_G15'] == [0, 0, 0]
    assert 'G15' in result['controls'] and 'NOT the unblended' in result['mlb_scope']
    text = json.dumps(ens.p19.jsonable(result)).lower()
    assert 'p_less' not in text and 'winner' not in text and 'promot' not in text


# ------------------------------------------------------------------ drift / probability guards
@pytest.mark.parametrize('mutate, message', [
    (lambda a: a[0]['P21-residual'][1].__setitem__('dev_y', np.roll(a[0]['P21-residual'][1]['dev_y'], 1)), 'dev_y'),
    (lambda a: a[0]['P20-hl730'][0].__setitem__('blend_keys', a[0]['P20-hl730'][0]['blend_keys'][::-1].copy()), 'blend_keys'),
    (lambda a: a[3]['P22-periodic'][2].__setitem__('mlb_dev_y', np.roll(a[3]['P22-periodic'][2]['mlb_dev_y'], 1)), 'unpaired whole-MLB'),
    (lambda a: a[4][1].__setitem__('dev_keys', a[4][1]['dev_keys'][::-1].copy()), 'Unpaired G0'),
    (lambda a: a[0]['P15-uniform'][0].__setitem__('dev', a[0]['P15-uniform'][0]['dev'] * 1.01), '(?i)probability'),
    (lambda a: a[3]['P20-hl180'][0].__setitem__('mlb_dev', np.log(a[3]['P20-hl180'][0]['mlb_dev'])), '(?i)probability'),
    (lambda a: a[0]['P21-residual'].pop(), 'three seeds'),
    (lambda a: a[0].pop('P22-periodic'), 'six direct families'),
])
def test_drift_and_nonprobability_refused(arrays, reference, mutate, message):
    a = copy.deepcopy(list(arrays))
    mutate(a)
    with pytest.raises(ValueError, match=message):
        ens.search(*a, reference, 1e-9, 129)


# ------------------------------------------------------------------ family verification (fake state files)
def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_family(root, protocol='ml_direct_residual_v1', device='mps', smoke=False, seeds=(0, 1, 2)):
    """Write a metadata-only family layout like the P21 runner's (tiny dummy files, no predictions)."""
    shared, own = {'scripts/run_ml_direct_head.py': 'shared'}, {'scripts/run_ml_direct_residual.py': 'own'}
    for rel, text in {**shared, **own}.items():
        (root / 'source' / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / 'source' / rel).write_text(text)
    control_sources = {r: hash_file(root / 'source' / r) for r in shared}
    own_hash = {r: hash_file(root / 'source' / r) for r in own}
    arch = {'width': 4}
    registered = {'experiment_id': 'EXP-P21-001', 'protocol': protocol, 'status': 'REGISTERED',
                  'control': {'preparation_sha256': 'CTRL'}, 'candidate_architecture': {**arch, 'other': 1}}
    dump(root / 'registered_config.json', registered)
    files = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
    ident = {'config_sha256': canonical_hash(registered), 'source_hashes': control_sources,
             'residual_source_hashes': own_hash, 'architecture': arch, 'device_option': 'auto', 'resolved_device': device}
    prep = {'identity': ident, 'comparable_device_to_control': device == 'mps', 'artifact_hashes': artifact_hashes(root, files)}
    dump(root / 'preparation.json', prep)
    for seed in seeds:
        d = root / str(seed)
        dump(d / 'fit.json', {'report': {'device': device}})
        (d / 'model.pt').write_bytes(b'm%d' % seed)
        for name in ('predictions.npz', 'calibration.json', 'mlb_predictions.npz'):
            (d / name).write_bytes(name.encode())
        base = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke}
        dump(d / 'fit_state.json', {'identity': base, 'artifact_hashes': artifact_hashes(d, ['model.pt', 'fit.json'])})
        pred = {**base, 'model_sha256': hash_file(d / 'model.pt')}
        dump(d / 'prediction_state.json', {'identity': pred, 'artifact_hashes': artifact_hashes(d, ['predictions.npz', 'calibration.json'])})
        dump(d / 'mlb_prediction_state.json', {'identity': pred, 'artifact_hashes': artifact_hashes(d, ['mlb_predictions.npz'])})
    spec = {'name': 'P21-residual', 'experiment_id': 'EXP-P21-001', 'protocol': 'ml_direct_residual_v1', 'dir': str(root),
            'config_canonical_sha256': canonical_hash(registered), 'own_source_sha256': own_hash}
    return spec


@pytest.fixture
def control():
    return {'prep': {'identity': {'source_hashes': None}}, 'sha256': 'CTRL'}


def make(tmp_path, control, **kw):
    spec = fake_family(tmp_path / 'EXP-P21-001', **kw)
    shared = Path(spec['dir']) / 'source' / 'scripts/run_ml_direct_head.py'
    control['prep']['identity']['source_hashes'] = {'scripts/run_ml_direct_head.py': hash_file(shared)}
    return spec


def test_valid_family_passes_and_pins(tmp_path, control):
    spec = make(tmp_path, control)
    ens.verify_family(spec, control)
    pinned = {**spec, 'preparation_sha256': hash_file(Path(spec['dir']) / 'preparation.json'),
              'state_sha256': {'1': {'prediction_state.json': hash_file(Path(spec['dir']) / '1' / 'prediction_state.json')}}}
    ens.verify_family(pinned, control)


@pytest.mark.parametrize('kw, message', [
    ({'smoke': True}, 'smoke'), ({'device': 'cpu'}, 'MPS'), ({'device': 'cuda'}, 'MPS'),
    ({'seeds': (0, 1)}, 'incomplete'), ({'protocol': 'ml_direct_periodic_v1'}, 'config differs')])
def test_bad_family_refused(tmp_path, control, kw, message):
    spec = make(tmp_path, control, **kw)
    if 'protocol' in kw:
        spec['config_canonical_sha256'] = 'pinned-sibling-config'
    with pytest.raises(ValueError, match=message):
        ens.verify_family(spec, control)


def test_tampered_artifacts_and_sources_refused(tmp_path, control):
    spec = make(tmp_path, control)
    root = Path(spec['dir'])
    (root / '2' / 'mlb_predictions.npz').write_bytes(b'partial')
    with pytest.raises(ValueError):
        ens.verify_family(spec, control)
    spec = make(tmp_path / 'b', control)
    with pytest.raises(ValueError, match='source snapshot'):
        ens.verify_family({**spec, 'own_source_sha256': {'scripts/run_ml_direct_residual.py': '0' * 64}}, control)
    with pytest.raises(ValueError, match='pinned'):
        ens.verify_family({**spec, 'state_sha256': {'0': {'fit_state.json': '0' * 64}}}, control)
    with pytest.raises(ValueError, match='smoke output'):
        ens.verify_family({**spec, 'dir': str(tmp_path / 'X-smoke' / 'EXP-P21-001')}, control)
    control['prep']['identity']['source_hashes'] = {'scripts/run_ml_direct_head.py': '1' * 64}
    with pytest.raises(ValueError, match='shared direct-head source'):
        ens.verify_family(spec, control)


def test_config_pins_match_sibling_metadata():
    config = json.loads(CONFIG.read_text())
    fam = {f['name']: f for f in config['families']}
    assert fam['P15-uniform']['preparation_sha256'] == '2e3f553f3ce86941187d6be21adeda05ebaf948a40d01110ef2326cb8327b0b4'
    assert {f['experiment_id'] for f in config['families']} == {'EXP-P15-001', 'EXP-P20-001', 'EXP-P21-001', 'EXP-P22-001'}
    assert [fam[f'P20-{v}']['dir'].endswith('/EXP-P20-001/' + v) for v in ('hl180', 'hl365', 'hl730')] == [True] * 3
    assert fam['P21-residual']['own_source_sha256']['scripts/run_ml_direct_recency.py'] == \
        fam['P20-hl180']['own_source_sha256']['scripts/run_ml_direct_recency.py']
    assert all(len(f['config_canonical_sha256']) == 64 for f in config['families'][1:])
    assert hash_file(REPO / config['reproduction_gate']['reference']) == config['reproduction_gate']['reference_sha256']

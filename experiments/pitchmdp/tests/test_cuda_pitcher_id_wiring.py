"""Synthetic CPU end-to-end tests: pitcher-ID family (DRAFT-CUDA-PITCHER-ID-20261005, tentative EXP-P38-001 DRAFT).

Config generation / DRAFT identity, negative guards (marker, GPU4, seeds, arms, pitcher_id block, DRAFT real fit, arm
outside the family), earlier P30-P37 configs routing as before, real CPU runner tasks on a synthetic export with
frequent / rare / unseen pitchers and split dates (TRAIN vocabulary, UNK shares, parameters, save/load/temperature
consistency, date-leak refusals), the scorer (vocabulary re-derived from the export, subsets, no verdict, refusals),
the TRAIN-only resource gate, registration helper and pipeline template.  No real data, no GPU, no DEV metric.
author_only; same_model_distinct_review_pending.
"""
import ast
import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
LAUNCH = PROJECT / 'docs' / 'launch'
sys.path[:0] = [str(PROJECT / 'scripts'), str(Path(__file__).resolve().parent), str(LAUNCH)]
import export_cuda_direct as ex  # noqa: E402
import make_p38_registered_config as reg  # noqa: E402
import p38_smoke_gate as gate  # noqa: E402
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic helpers
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_embed_type_registry as et  # noqa: E402
from pitchmdp import direct_pitcher_id_model as pidm  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402
from pitchmdp import direct_robust_clip_registry as rcr  # noqa: E402
from pitchmdp.direct_periodic_residual_model import parameter_count as embed_parameters  # noqa: E402

CONFIGS = REPO / 'configs'
RUN = CONFIGS / 'DRAFT-CUDA-PITCHER-ID-20261005.json'
RUN_SHA256 = '8518f754236447fd054550dee75a5c3da7c510c3dd646f028662f097a02ee537'
MAKE = LAUNCH / 'make_p38_config.py'
TEMPLATE = LAUNCH / 'run-p38-pipeline.TEMPLATE.py'
PREREG = PROJECT / 'docs' / 'PREREG-DRAFT-EXP-P38-001-pitcher-id-2026-10-05.md'
GPU3, GPU4 = pid.RETAINED_GPU, pid.RELEASED_GPUS[0]
FORBIDDEN_KEYS = {'winner', 'promotion', 'promote', 'verdict', 'ranking', 'selected', 'selection', 'decision_rule'}
EARLIER = [CONFIGS / f'EXP-P3{i}-001.json' for i in range(7)] + [CONFIGS / 'DRAFT-CUDA-EMBED-TYPE-20261005.json']
REF, CAND = pid.REFERENCE, pid.CANDIDATE


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def read(path):
    return json.loads(Path(path).read_text())


def load(registered=False):
    cfg = tf.load(RUN, False, True)
    if registered:
        cfg['status'], cfg['execution'] = 'REGISTERED', {'enabled': True, 'executable': True}
    return rc.config_check(cfg)


def keys_deep(value):
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_deep(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_deep(v)}
    return set()


# ------------------------------------------------------------------ synthetic export with pitchers and split dates

SIZES = {'train': 400, 'earlystop': 120, 'temperature': 80, 'blend': 40, 'dev': 60, 'mlb_dev': 50}
DATES = {'train': 20250430, 'earlystop': 20250510, 'temperature': 20250520, 'blend': 20250601, 'dev': 20250701,
         'mlb_dev': 20250801}


def pitchers(split, n, rng):
    if split == 'train':  # 101: 200 rows, 102: 120 (vocab); 103: 50, 104: 30 (rare -> UNK in TRAIN too)
        out = np.repeat(np.array([101, 102, 103, 104]), [200, 120, 50, 30])
        return rng.permutation(out)
    return rng.choice([101, 102, 103, 999], n)  # 999 first appears after TRAIN (cold start -> UNK)


def make_export(root, train_date=None, early_date=None):
    splits = {}
    for i, (s, n) in enumerate(SIZES.items()):
        arrays = tf.split_arrays(n, i, 1000 * (i + 1))
        rng = np.random.default_rng(50 + i)
        arrays['pitcher'] = pitchers(s, n, rng).astype(np.int64)
        date = {'train': train_date, 'earlystop': early_date}.get(s) or DATES[s]
        arrays['date'] = np.full(n, date, dtype=np.int32)
        arrays['date'][: n // 2] = 20240601 if s == 'train' else date  # TRAIN spans earlier seasons too
        splits[s] = arrays
    return root, ex.write_export(root, splits, tf.lineage(load()), tf.SHAPES, tf.COLUMNS), splits


@pytest.fixture
def export(tmp_path):
    return make_export(tmp_path / 'export')


# ------------------------------------------------------------------ config / registry

def test_config_is_the_generated_draft_under_a_draft_name():
    assert hashlib.sha256(RUN.read_bytes()).hexdigest() == RUN_SHA256 == gate.DRAFT_SHA == reg.DRAFT_SHA
    out = subprocess.run([sys.executable, '-c', f'import runpy, json; ns = runpy.run_path({str(MAKE)!r}); '
                          'print(json.dumps(ns["build"](), indent=2, ensure_ascii=False))'],
                         capture_output=True, text=True, check=True).stdout
    assert out == RUN.read_text()  # deterministic generator, nothing hand-edited
    registered = CONFIGS / 'EXP-P38-001.json'  # p38-review-execute-002: registered 2026-10-05 from this DRAFT
    if registered.exists():  # only the registration paths may differ from the DRAFT
        reg_cfg, draft = read(registered), read(RUN)
        assert reg_cfg['status'] == 'REGISTERED' and reg_cfg['registration']['draft_sha256'] == RUN_SHA256
        assert {k for k in set(draft) | set(reg_cfg) if draft.get(k) != reg_cfg.get(k)} == \
            {'status', 'decision', 'execution', 'registration'}
    cfg = read(RUN)
    assert cfg['status'] == 'DRAFT' and cfg['execution']['enabled'] is False and cfg['execution']['executable'] is False
    assert 'registration' not in cfg and cfg['experiment_id'] == 'EXP-P38-001' and cfg[pid.MARKER] is True
    assert cfg['source_configs']['robust_clip_registered']['sha256'] == \
        hashlib.sha256((CONFIGS / 'EXP-P36-001.json').read_bytes()).hexdigest()
    p36 = read(CONFIGS / 'EXP-P36-001.json')
    for key in ('protocol', 'export', 'training', 'calibration', 'data_rules'):
        assert cfg[key] == p36[key], key
    assert cfg['training']['epochs'] == 60 and cfg['seeds'] == [44, 45, 46, 47, 48] and cfg['fits'] == 10
    assert cfg['device'] == {'kind': 'cuda', 'allowed_gpu_uuids': [GPU3], 'max_parallel': 1, 'cpu_fallback': False}
    assert GPU4 not in json.dumps({k: v for k, v in cfg.items() if k != 'queue'})
    assert cfg['comparison'][pid.CONTRAST_KEY] == pid.CONTRASTS and not FORBIDDEN_KEYS & keys_deep(cfg)
    assert 'impossible' not in json.dumps(cfg) and 'not included in this bounded' in json.dumps(cfg)


def test_redating_changes_only_the_queue_clock_and_refuses_after_23():
    ns = __import__('runpy').run_path(str(MAKE))
    base, moved = ns['build'](), ns['build']('2026-10-06', '13:02')
    assert {k for k in base if base[k] != moved[k]} == {'queue'}
    changed = {k for k in base['queue'] if base['queue'][k] != moved['queue'][k]}
    assert changed == {'date', 'launch_by', 'no_new_fit_after', 'kill_own_after', 'aggregate_by'}
    assert moved['queue']['no_new_fit_after'] == '14:35'  # 13:02 + 5,292 s = 14:30:12 -> ceil 14:35
    assert base['queue']['no_new_fit_after'] == '22:15' and base['queue']['aggregate_by'] == '22:50'
    with pytest.raises(SystemExit, match='23:00'):
        ns['build']('2026-10-05', '21:30')


def test_arms_are_the_p36_entry_and_the_fixed_candidate():
    p36 = {a['name']: a for a in read(CONFIGS / 'EXP-P36-001.json')['arms']}
    ref, cand = read(RUN)['arms']
    assert et.same(ref, p36[REF])
    assert {k: v for k, v in cand.items() if k not in ('name', 'family', 'pitcher_id')} == \
        {k: v for k, v in ref.items() if k not in ('name', 'family')}
    assert cand['pitcher_id'] == pid.PITCHER_ID


def test_seed_block_is_disjoint_from_every_earlier_block_including_p37():
    earlier = {s for block in pid.EARLIER_SEEDS.values() for s in block}
    assert earlier == set(range(44)) and not set(pid.SEEDS) & earlier


def test_parameter_counts_rederived_from_the_frozen_networks():
    shapes = read(RUN)['export']['shapes']
    n_context = shapes['n_context'] + 1
    assert embed_parameters('periodic', n_context, shapes['n_token'], shapes['length'], et.CONTINUOUS) == \
        pid.BASE_PARAMETERS == 244898
    net = pidm.PitcherIdPeriodicResidualNetwork(n_context + 1, shapes['n_token'], shapes['length'], et.CONTINUOUS,
                                                n_ids=37, id_seed=1)
    assert sum(p.numel() for p in net.parameters()) == pid.BASE_PARAMETERS + pid.id_parameters(37) \
        == 244898 + 38 * 8 + 1024


def mutate(cfg, edit):
    cfg = copy.deepcopy(cfg)
    edit(cfg)
    return cfg


EDITS = {
    'marker_false': lambda c: c.update({pid.MARKER: False}),
    'seeds_p37': lambda c: c.update(seeds=[39, 40, 41, 42, 43]),
    'repeat_tag': lambda c: c.update(repeat_tag='x'),
    'fits': lambda c: c.update(fits=15),
    'gpu4_added': lambda c: c['device'].update(allowed_gpu_uuids=[GPU3, GPU4]),
    'gpu4_only': lambda c: c['device'].update(allowed_gpu_uuids=[GPU4]),
    'two_parallel': lambda c: c['device'].update(max_parallel=2),
    'cpu_fallback': lambda c: c['device'].update(cpu_fallback=True),
    'cap_30': lambda c: c['training'].update(epochs=30),
    'min_count_50': lambda c: c['arms'][1]['pitcher_id'].update(min_train_pitches=50),
    'cutoff_later': lambda c: c['arms'][1]['pitcher_id'].update(last_train_date=20250515),
    'vocab_dev': lambda c: c['arms'][1]['pitcher_id'].update(vocab_source='dev'),
    'd_id_16': lambda c: c['arms'][1]['pitcher_id'].update(d_id=16),
    'third_arm': lambda c: c['arms'].append(dict(c['arms'][0], name='control')),
    'swapped': lambda c: c['arms'].reverse(),
    'reference_changed': lambda c: c['arms'][0].update(width=256),
}


@pytest.mark.parametrize('edit', EDITS)
def test_config_check_refuses_family_drift(edit):
    with pytest.raises(ValueError):
        rc.config_check(mutate(load(), EDITS[edit]))


def test_pitcher_id_arm_outside_the_marker_family_is_refused():
    p36 = read(CONFIGS / 'EXP-P36-001.json')
    p36['arms'].append(pid.BASELINES[CAND])
    with pytest.raises(ValueError, match='marker family only'):
        rc.config_check(p36)
    with pytest.raises(ValueError, match='fixed pre-registered'):
        pid.check_contrasts(mutate(load(), lambda c: c['comparison'][pid.CONTRAST_KEY][0].update(role='secondary')))


def test_p38_never_reaches_another_family_check(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('P38 config reached another family check')
    for module in (et, rcr, er):
        monkeypatch.setattr(module, 'check_family', boom)
    assert rc.config_check(load())[pid.MARKER] is True
    assert rc.training_budget(load()) == pid.TRAINING and rc.uses_epoch_cap(load())


@pytest.mark.parametrize('path', EARLIER, ids=lambda p: p.stem)
def test_earlier_configs_route_and_budget_exactly_as_before(path):
    cfg = read(path)
    assert not rc.uses_pitcher_id(cfg) and not any(rc.is_pitcher_id_arm(a) for a in cfg['arms'])
    rc.config_check(cfg)
    assert pid.check_contrasts(cfg) is None


def test_constants_and_sources_are_wired():
    assert rc.PITCHER_ID_MARKER == pid.MARKER and rc.PITCHER_ID_FLAG == pid.FLAG
    assert {'pitchmdp/direct_pitcher_id_model.py', 'pitchmdp/direct_pitcher_id_registry.py'} <= set(rc.MODEL_SOURCES)
    assert 'pitchmdp/direct_pitcher_id_registry.py' in sc.SCORER_SOURCES
    out = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, "."); '
                          'import pitchmdp.direct_pitcher_id_registry; print("torch" in sys.modules)'],
                         capture_output=True, text=True, check=True, cwd=PROJECT).stdout.strip()
    assert out == 'False'  # registry (used by config_check and the scorer) is torch-free


def test_prereg_states_the_fixed_design_gate_and_limitation():
    text = PREREG.read_text()
    for needle in ('EXP-P38-001', '44, 45, 46, 47, 48', 'periodic_residual_counts', 'pitcher_id_periodic_residual_counts',
                   '1.25', '5,292', '20250430', GPU3, GPU4, '이번 제한된 2팔 연구에 포함하지 않았다', '인과 귀속 한계',
                   'decision` null', '2265f44', 'DRAFT-CUDA-PITCHER-ID-20261005.json'):
        assert needle in text, needle
    assert '만들 수 없' not in text and 'impossible' not in text


# ------------------------------------------------------------------ runner (real CPU tasks)

def test_draft_refuses_real_fits(export, tmp_path):
    path, digest, _ = export
    for arm in pid.ARM_ORDER:
        with pytest.raises(ValueError, match='REGISTERED'):
            rc.run_task(load(), path, digest, tmp_path / 'real', arm, 44, 'cpu', {})
    assert not (tmp_path / 'real').exists() or not any((tmp_path / 'real').rglob('state.json'))


def test_candidate_smoke_task_records_train_vocab_and_is_consistent_after_load(export, tmp_path):
    path, digest, splits = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', CAND, 44, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    v = fit[pid.FLAG]
    assert (v['n_vocab'], v['min_count'], v['train_rows'], v['train_pitchers'], v['train_unk_rows']) == (2, 100, 400, 4, 80)
    assert v['ids_sha256'] == pidm.vocab_sha256([101, 102]) and v['train_max_date'] == 20250430
    assert v['unk_share']['train'] == pytest.approx(.2)
    for split in ('earlystop', 'temperature', 'blend', 'dev', 'mlb_dev'):
        expected = np.isin(splits[split]['pitcher'][:rc.SMOKE['predict']], [103, 999]).mean() \
            if split in ('blend', 'dev', 'mlb_dev') else np.isin(splits[split]['pitcher'], [103, 999]).mean()
        assert v['unk_share'][split] == pytest.approx(expected), split
    assert v['split_dates']['train'] == [20240601, 20250430] and v['split_dates']['earlystop'][0] > 20250430
    assert fit['parameter_count'] == pid.BASE_PARAMETERS + pid.id_parameters(2)
    assert pid.check_fit(cfg, CAND, fit)['vocab']['n_vocab'] == 2
    # save / load / temperature / prediction inputs consistent: reload and reproduce the saved dev probabilities
    model = rc.load_model(rc.arm_spec(cfg, CAND), dest / 'model.pt', 'cpu')
    assert np.array_equal(model.vocab_ids, [101, 102])
    dev = rc.load_split(path, rc.verify_export(path, cfg, digest), 'dev')
    inputs, _ = pidm.id_inputs(rc.arm_inputs(dev, rc.arm_spec(cfg, CAND), cfg['export']['workload_columns']),
                               dev['pitcher'], model.vocab_ids)
    temperature = read(dest / 'calibration.json')['delivery_temperature']
    calibrated, _ = rc.probabilities(rc.g0_log_probs(model.logits(inputs)), temperature)
    with np.load(dest / 'predictions.npz') as saved:
        np.testing.assert_allclose(saved['dev'], calibrated, rtol=1e-6, atol=1e-7)
        assert np.array_equal(saved['dev_pitcher'], dev['pitcher'])
    # a tampered vocabulary record refuses the load
    fit[pid.FLAG]['ids_sha256'] = '0' * 64
    rc.write_json_atomic(dest / 'fit.json', fit)
    with pytest.raises(ValueError, match='TRAIN vocabulary'):
        rc.load_model(rc.arm_spec(cfg, CAND), dest / 'model.pt', 'cpu')


def test_reference_smoke_task_is_the_unchanged_embedded_path(export, tmp_path):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', REF, 44, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert pid.FLAG not in fit and fit['parameter_count'] == pid.BASE_PARAMETERS
    assert fit['epoch_cap']['cap'] == rc.SMOKE['epochs']
    model = rc.load_model(rc.arm_spec(cfg, REF), dest / 'model.pt', 'cpu')
    assert type(model).__name__ == 'DirectPeriodicResidualModel'
    with pytest.raises(ValueError, match='Only pitcher-ID arms'):
        rc.build_model(rc.arm_spec(cfg, REF), 44, 'cpu', vocab_ids=np.array([1]))
    with pytest.raises(ValueError, match='TRAIN vocabulary'):
        rc.build_model(rc.arm_spec(cfg, CAND), 44, 'cpu')


@pytest.mark.parametrize('train_date,early_date,match', [
    (20250501, None, 'TRAIN max_date'), (None, 20250430, 'TRAIN cutoff')])
def test_date_leak_is_refused_before_any_fit(tmp_path, train_date, early_date, match):
    path, digest, _ = make_export(tmp_path / 'leak', train_date, early_date)
    with pytest.raises(ValueError, match=match):
        rc.run_task(load(), path, digest, tmp_path / 'smoke', CAND, 44, 'cpu', {}, smoke=True)
    assert not any((tmp_path / 'smoke').rglob('state.json'))


def test_vocab_rows_after_cutoff_refused_even_if_manifest_passes():
    with pytest.raises(ValueError, match='after the TRAIN cutoff'):
        pid.build_vocab(np.array([1, 1]), np.array([20250430, 20250501]), pid.LAST_TRAIN_DATE, 1)


# ------------------------------------------------------------------ scorer (real CPU fits relabelled as CUDA tasks)

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)


def run_family(tmp_path, export, cfg, tamper=None):
    """Real non-smoke CPU fits of all 10 tasks, then identity/fit relabelled 'cuda' (scorer contract) and resealed."""
    path, digest, _ = export
    fam = tmp_path / 'fam'
    sources = rc.own_hashes()
    for arm in pid.ARM_ORDER:
        for seed in pid.SEEDS:
            dest = rc.run_task(cfg, path, digest, fam, arm, seed, 'cpu', {})
            identity = rc.expected_identity(cfg, digest, arm, seed, False, 'cuda', sources)
            rc.write_json_atomic(dest / 'identity.json', identity)
            fit = read(dest / 'fit.json')
            fit.update(device='cuda', runtime={**tf.RUNTIME, 'device_uuid': GPU3, 'bound_uuid': GPU3})
            if tamper:
                tamper(arm, seed, fit)
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest, identity)
    return fam


@pytest.fixture(scope='module')
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('scored')
    exp = make_export(tmp / 'export')
    cfg = load(registered=True)
    fam = run_family(tmp, exp, cfg)
    return tmp, exp, cfg, fam


def test_scorer_rederives_vocab_reports_subsets_and_no_verdict(scored, monkeypatch):
    tmp, exp, cfg, fam = scored
    monkeypatch.setattr(sc, 'DRAWS', 200)

    def boom(*args, **kwargs):
        raise AssertionError('P38 reached another family scorer path')
    monkeypatch.setattr(sc, 'score_family', boom)
    monkeypatch.setattr(sc, 'cost_summary', boom)
    result = tf.run_score(tmp, cfg, exp, fam, dest='score')
    assert result['decision'] is None and result['ranking'] is None
    assert not FORBIDDEN_KEYS & keys_deep({k: v for k, v in result.items() if k not in ('decision', 'ranking')})
    v = result['vocabulary']
    assert (v['n_vocab'], v['train_unk_rows'], v['id_parameters']) == (2, 80, pid.id_parameters(2))
    _, _, splits = exp
    for split in ('dev', 'mlb_dev'):
        rows = result['pitcher_id_rows'][split]
        unk = int(np.isin(splits[split]['pitcher'], [103, 999]).sum())
        assert (rows['all']['rows'], rows['unk']['rows'], rows['in_vocab']['rows']) == \
            (SIZES[split], unk, SIZES[split] - unk)
        block = result['pitcher_id_contrasts'][split][pid.PRIMARY]
        assert set(block) == set(pid.SUBSETS) and set(block['all']['seeds']) == {str(s) for s in pid.SEEDS}
        assert block['all']['compute_matched'] is False and block['all']['parameter_matched'] is False
        assert list(result['splits'][split]) == [REF, CAND]
    assert result['seed_block']['same_gpu_pairs'] == {pid.PRIMARY: 5}
    assert 'not included in this bounded' in result['seed_block']['note']
    for arm in pid.ARM_ORDER:
        assert set(result['cost'][arm]) == {str(s) for s in pid.SEEDS}
    table = (tmp / 'score' / 'table_ko.md').read_text()
    assert table.index('## dev') < table.index('## mlb_dev') < table.index('## 비용') and '단정할 수 없습니다' in table


def test_scorer_refuses_tampered_vocab_parameters_and_reference_record(tmp_path, few_draws):
    exp = make_export(tmp_path / 'export')
    cfg = load(registered=True)
    cases = {
        'Recorded vocabulary differs': lambda a, s, f: f[pid.FLAG].update(
            ids_sha256=pidm.vocab_sha256([101])) if a == CAND else None,
        'parameter_count': lambda a, s, f: f.update(parameter_count=f['parameter_count'] + 1) if (a, s) == (CAND, 45) else None,
        'reference fit carries': lambda a, s, f: f.update({pid.FLAG: {'n_vocab': 2}}) if (a, s) == (REF, 46) else None,
        'one vocabulary record': lambda a, s, f: f[pid.FLAG].update(train_rows=399) if (a, s) == (CAND, 47) else None,
    }
    for i, (match, tamper) in enumerate(cases.items()):
        fam = run_family(tmp_path / str(i), exp, cfg, tamper)
        with pytest.raises(ValueError, match=match):
            tf.run_score(tmp_path / str(i), cfg, exp, fam)
        assert not (tmp_path / str(i) / 'score').exists()


def test_scorer_refuses_draft_incomplete_and_foreign_contrast(scored, few_draws):
    tmp, exp, cfg, fam = scored
    with pytest.raises(ValueError, match='REGISTERED'):
        tf.run_score(tmp, load(), exp, fam, dest='draft')
    extra = mutate(cfg, lambda c: c['comparison'].update(embed_type_contrasts=et.CONTRASTS))
    with pytest.raises(ValueError):
        tf.run_score(tmp, extra, exp, fam, dest='foreign')
    p36 = read(CONFIGS / 'EXP-P36-001.json')
    p36['comparison'][pid.CONTRAST_KEY] = pid.CONTRASTS
    with pytest.raises(ValueError):
        sc.score(*tf.write_cfg(tmp, p36, 'p36x.json'), exp[0], exp[1], fam, tmp / 'p36x')


# ------------------------------------------------------------------ TRAIN-only resource gate

def gate_fits(per_epoch=None, uuid=GPU3, vocab=True):
    per_epoch = per_epoch or {REF: 2., CAND: 2.2}
    record = {'n_vocab': 2, 'ids_sha256': 'a' * 64, 'min_count': 100, 'train_rows': 8192, 'train_pitchers': 9,
              'train_unk_rows': 10, 'train_max_date': 20250430, 'last_train_date': 20250430}
    out = {}
    for arm in pid.ARM_ORDER:
        for seed in pid.SEEDS:
            fit = {'report': {'seconds': 2 * per_epoch[arm], 'epochs_run': 2}, 'runtime': {'bound_uuid': uuid},
                   'dev_used_for_fit_or_selection': False}
            if arm == CAND and vocab:
                fit[pid.FLAG] = dict(record)
            out.setdefault(arm, {})[seed] = fit
    return out


def test_gate_passes_at_the_prespecified_ratio_and_refuses_above():
    ok = gate.evaluate(gate_fits())
    assert ok['pass'] and ok['candidate_over_reference'] == pytest.approx(1.1) and ok['ratio_max'] == 1.25
    assert ok['smoke_vocabulary']['n_vocab'] == 2 and ok['tasks'] == 10
    assert not gate.evaluate(gate_fits({REF: 2., CAND: 2.6}))['pass']
    assert gate.evaluate(gate_fits({REF: 2., CAND: 2.5}))['pass']  # boundary 1.25 inclusive


def test_gate_refuses_invalid_records_gpu4_and_vocab_problems():
    assert not gate.evaluate(gate_fits(uuid=GPU4))['pass']
    assert not gate.evaluate(gate_fits(vocab=False))['pass']
    f = gate_fits()
    f[CAND][46][pid.FLAG]['ids_sha256'] = 'b' * 64
    assert 'different vocabulary' in ' '.join(gate.evaluate(f)['errors'])
    f = gate_fits()
    f[REF][44][pid.FLAG] = {'n_vocab': 1}
    assert not gate.evaluate(f)['pass']
    f = gate_fits()
    f[CAND][48]['report']['epochs_run'] = 0
    assert not gate.evaluate(f)['pass']
    f = gate_fits()
    del f[CAND][45]
    assert not gate.evaluate(f)['pass']
    f = gate_fits()
    f[CAND][44][pid.FLAG]['train_max_date'] = 20250501
    assert not gate.evaluate(f)['pass']


def test_all_unk_smoke_vocabulary_runs_and_passes_the_gate(export, tmp_path, monkeypatch):
    """Review-003: smoke reads only the first SMOKE['train'] TRAIN rows; if no pitcher reaches 100 of them the smoke
    vocabulary is empty (every row UNK). That run must still work and must not be refused by the resource gate."""
    path, digest, _ = export
    monkeypatch.setitem(rc.SMOKE, 'train', 60)  # < 100 rows: no pitcher can enter the vocabulary
    dest = rc.run_task(load(), path, digest, tmp_path / 'smoke', CAND, 44, 'cpu', {}, smoke=True)
    v = read(dest / 'fit.json')[pid.FLAG]
    assert (v['n_vocab'], v['train_rows'], v['train_unk_rows']) == (0, 60, 60) and v['unk_share']['train'] == 1.
    assert v['ids_sha256'] == pidm.vocab_sha256([])
    assert rc.load_model(rc.arm_spec(load(), CAND), dest / 'model.pt', 'cpu').net.n_ids == 0
    fits = gate_fits()
    for seed in pid.SEEDS:
        fits[CAND][seed][pid.FLAG] = {k: v[k] for k in pid.VOCAB_KEYS}
    ok = gate.evaluate(fits)
    assert ok['pass'] and ok['smoke_vocabulary']['n_vocab'] == 0
    fits[CAND][45][pid.FLAG]['n_vocab'] = -1
    assert not gate.evaluate(fits)['pass']


def test_gate_main_reads_runner_identities_and_refuses_missing_smoke(tmp_path):
    """Review-003: the gate's IO path (runner.task_dir / expected_identity / verify_task / own_hashes, queue counts,
    new summary file) runs without CUDA and fails closed with exit 3 when the 10 smoke tasks are absent."""
    queue, runs, out = tmp_path / 'queue', tmp_path / 'runs-smoke', tmp_path / 'gate.json'
    queue.mkdir(), runs.mkdir()
    (queue / 'summary.json').write_text(json.dumps({'counts': {'done': 3, 'failed': 7}}))
    assert gate.main([str(RUN), str(queue), str(runs), str(out)]) == 3
    record = read(out)
    assert record['pass'] is False and record['config_sha256'] == gate.DRAFT_SHA
    assert any('smoke queue counts' in e for e in record['errors'])
    assert sum('identity' in e for e in record['errors']) == pid.FITS
    assert record['source_sha256'] == rc.own_hashes()
    with pytest.raises(AssertionError, match='already exists'):
        gate.main([str(RUN), str(queue), str(runs), str(out)])
    with pytest.raises(AssertionError, match='-smoke'):
        gate.main([str(RUN), str(queue), str(tmp_path / 'runs'), str(tmp_path / 'other.json')])


def test_projection_fits_the_default_window():
    q = read(RUN)['queue']
    start = reg.at(q['date'], q['launch_by'])
    assert start + timedelta(seconds=pid.FAMILY_PROJECTION_SECONDS) <= reg.at(q['date'], q['no_new_fit_after'])
    assert pid.FAMILY_PROJECTION_SECONDS == 5 * (2 * 235.20 + 1.25 * 2 * 235.20) == pytest.approx(5292.)
    assert q['aggregate_by'] <= '23:00'


# ------------------------------------------------------------------ registration helper / pipeline template

def summary(tmp_path, **over):
    record = {**gate.evaluate(gate_fits()), 'config_sha256': gate.DRAFT_SHA,
              'export_manifest_sha256': gate.MANIFEST_SHA, 'source_sha256': rc.own_hashes(), **over}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(record, sort_keys=True))
    return path


def test_registration_changes_only_registration_paths_and_refuses_bad_gates(tmp_path):
    from test_p38_resource_v2 import v2_measure  # p38-recovery-001: registration also needs the passing v2 measurement
    s, v2 = summary(tmp_path), v2_measure(tmp_path)
    when = reg.at('2026-10-05', '20:40')
    one = reg.build(s, v2, '20:40', 'abc1234', 'root go ref', when)
    draft = read(RUN)
    assert {k for k in set(one) | set(draft) if one.get(k) != draft.get(k)} == \
        {'status', 'decision', 'execution', 'registration'}
    assert one['execution'] == {'enabled': True, 'executable': True} and one['queue'] == draft['queue']
    rc.config_check(one)
    rc.require_executable(one, False)
    with pytest.raises(AssertionError, match='launch_by'):
        reg.build(s, v2, '20:50', 'abc1234', 'go', reg.at('2026-10-05', '20:50'))
    for over in ({'pass': False}, {'bound_uuids': [GPU3, GPU4]}, {'tasks': 15}, {'config_sha256': '0' * 64},
                 {'candidate_over_reference': 1.0}):
        (tmp_path / 'gate.json').unlink()
        with pytest.raises(AssertionError):
            reg.build(summary(tmp_path, **over), v2, '20:40', 'abc1234', 'go', when)


def test_template_fails_closed_gpu3_only_requires_p37_terminal_and_shares_functions():
    out = subprocess.run([sys.executable, str(TEMPLATE)], capture_output=True, text=True, cwd='/tmp')
    assert out.returncode == 1 and 'configsha not filled' in out.stderr
    text = TEMPLATE.read_text()
    assert GPU4 not in text and text.count("'--gpu'") == 3
    assert "LOCKS='/home/sds733/pitcheezy-autonomy/gpu-locks'" in text and "'--lock-dir',LOCKS" in text
    assert "score-p37-state.json" in text and "'complete','incomplete','blocked'" in text and 'nvidia-smi' in text
    assert 'DRAFT-CUDA-PITCHER-ID-20261005.json' in text and gate.DRAFT_SHA in text and "{'done':10}" in text
    p37, p38 = (ast.parse((LAUNCH / f'run-p3{i}-pipeline.TEMPLATE.py').read_text()) for i in (7, 8))
    for name in ('save', 'interrupted', 'execute', 'queue'):
        get = lambda t: next(ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)  # noqa: E731
        assert get(p37) == get(p38), name
    tree = ast.parse(text)
    env = {'datetime': __import__('datetime'), 'SCORE_SECONDS': 900}
    exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clock'], []),
                 str(TEMPLATE), 'exec'), env)
    q = read(RUN)['queue']
    timeout, score_by = env['clock'](q, reg.at(q['date'], '20:40'), pid.FAMILY_PROJECTION_SECONDS)
    assert score_by == reg.at(q['date'], '22:30') and timeout == 110 * 60 + 300
    with pytest.raises(AssertionError, match='launch_by'):
        env['clock'](q, reg.at(q['date'], '20:50'), pid.FAMILY_PROJECTION_SECONDS)


def test_p38_resource_child_runs_the_frozen_pitcher_id_route_train_only_on_cpu(export, tmp_path, monkeypatch):
    """p38-recovery-001: the v2 measurement child fits both P38 arms through the runner's vocabulary/input/build rule
    with fixed epochs (patience == epochs) and records the TRAIN-only vocabulary; CUDA binding stubbed for CPU."""
    import p38_resource_child as child
    path, digest, _ = export
    cfg = tmp_path / 'draft-test.json'  # the DRAFT as the synthetic-export tests load it (test manifest pin)
    cfg.write_text(json.dumps(load()))
    spec = {'protocol': child.PROTOCOL, 'gpu_uuid': GPU3, 'config': str(cfg), 'export': str(path),
            'pins': {'config_sha256': rc.sha256_file(cfg), 'export_manifest_sha256': digest,
                     'source_sha256': rc.own_hashes()},
            'tasks': [{'arm': REF, 'seed': 902}, {'arm': CAND, 'seed': 902}],
            'workload': {'train_rows': None, 'earlystop_rows': None, 'epochs': 2}}
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU3)
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid} if allowed == [GPU3] else 1 / 0)
    build = rc.build_model
    monkeypatch.setattr(rc, 'build_model', lambda arm, seed, device, *a: build(arm, seed, 'cpu', *a))
    monkeypatch.setattr(rc, 'peak_device_bytes', lambda device: None)
    out = {}
    for arm in (REF, CAND):
        child.main(['--spec', str(spec_path), '--arm', arm, '--seed', '902', '--result', str(tmp_path / f'{arm}.json')])
        out[arm] = read(tmp_path / f'{arm}.json')
    for record in out.values():
        assert record['epochs_run'] == 2 and record['splits_loaded'] == ['train', 'earlystop'] and record['train_only']
        assert record['dev_used_for_fit_or_selection'] is False and record['runtime'] == {'bound_uuid': GPU3}
    assert out[REF]['pitcher_id'] is None and out[REF]['optimizer_updates'] == out[CAND]['optimizer_updates']
    manifest = rc.verify_export(path, load(), digest, hash_arrays=False)
    train, _ = rc.train_subset(load(), rc.arm_spec(load(), CAND), rc.load_split(path, manifest, 'train'))
    vocab = rc.pitcher_vocab(rc.arm_spec(load(), CAND), manifest, train)
    assert {k: out[CAND]['pitcher_id'][k] for k in pid.VOCAB_KEYS} == pid.vocab_record(vocab)
    assert out[CAND]['parameter_count'] == out[REF]['parameter_count'] + (vocab['n_vocab'] + 1) * 8 + 1024
    with pytest.raises(FileExistsError):
        child.main(['--spec', str(spec_path), '--arm', REF, '--seed', '902', '--result', str(tmp_path / f'{REF}.json')])
    with pytest.raises(ValueError, match='predeclared'):
        child.main(['--spec', str(spec_path), '--arm', REF, '--seed', '44', '--result', str(tmp_path / 'x.json')])

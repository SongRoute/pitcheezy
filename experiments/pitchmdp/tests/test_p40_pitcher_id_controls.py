"""Focused CPU tests: EXP-P40-001 DRAFT (pitcher-ID controls: row-shuffled-id control and ID dropout).

Generator / DRAFT identity, design guards, the row-shuffle permutation (information removed, frequencies kept,
seed-independent), the ID-dropout network (eval == candidate path, private generator, archive round trip) and a real
CPU end-to-end family of 20 synthetic tasks through the runner and the scorer.  No real data, no GPU, no DEV metric.
"""
import copy
import hashlib
import json
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
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402
import test_cuda_pitcher_id_wiring as w  # noqa: E402  synthetic export with pitchers and split dates
import test_direct_pitcher_id_model as tm  # noqa: E402
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402
from pitchmdp import direct_pitcher_id_model as pidm  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

RUN = REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json'
RUN_SHA256 = '3dba5445b3a76dbc5f19c73f1cd60f6a69fd18312c775a6f82e06db656ab6944'
MAKE = LAUNCH / 'make_p40_config.py'
REF, CAND, SHUF, DROP = pid.REFERENCE, pid.CANDIDATE, pid.SHUFFLE, pid.DROPOUT


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def load(registered=False):
    cfg = tf.load(RUN, False, True)
    if registered:
        cfg['status'], cfg['execution'] = 'REGISTERED', {'enabled': True, 'executable': True}
    return rc.config_check(cfg)


def test_draft_is_generated_deterministically_and_passes_the_runner_check():
    assert hashlib.sha256(RUN.read_bytes()).hexdigest() == RUN_SHA256
    out = subprocess.run([sys.executable, '-c', f'import runpy, json; ns = runpy.run_path({str(MAKE)!r}); '
                          'print(json.dumps(ns["build"](), indent=2, ensure_ascii=False))'],
                         capture_output=True, text=True, check=True).stdout
    assert out == RUN.read_text()
    raw, p38 = (json.loads(p.read_text()) for p in (RUN, REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-20261005.json'))
    for key in ('protocol', 'export', 'training', 'calibration', 'data_rules', 'device'):
        assert raw[key] == p38[key], key
    cfg = load()
    assert cfg['status'] == 'DRAFT' and cfg['execution']['enabled'] is False and 'registration' not in cfg
    assert [a['name'] for a in cfg['arms']] == [REF, CAND, SHUF, DROP] and cfg['seeds'] == [60, 61, 62, 63, 64]
    assert cfg['fits'] == 20 and cfg['device']['allowed_gpu_uuids'] == [pid.RETAINED_GPU]
    assert pid.RELEASED_GPUS[0] not in json.dumps(raw['device']) + json.dumps(raw['arms'])  # named only as released
    q = cfg['queue']
    assert (q['date'], q['launch_by'], q['no_new_fit_after'], q['kill_own_after'], q['aggregate_by']) == \
        ('2026-10-06', '02:05', '03:50', '04:05', '04:25')
    assert pid.P40_FAMILY_PROJECTION_SECONDS == pytest.approx(20 * 1.25 * 244.70)


def test_design_guards():
    cfg = load()
    for mutate, match in (
            (lambda c: c.update({pid.DESIGN_KEY: 'p40_other'}), 'must be absent'),
            (lambda c: c.update(seeds=[44, 45, 46, 47, 48]), 'seeds must be exactly'),
            (lambda c: c['arms'].pop(), 'must be exactly'),
            (lambda c: c['arms'][2][pid.FLAG].update(shuffle_seed=1), 'differs from its fixed registry entry'),
            (lambda c: c['arms'][3][pid.FLAG].update(id_dropout=0.5), 'differs from its fixed registry entry'),
            (lambda c: c['comparison'][pid.CONTRAST_KEY].pop(), 'differ from the fixed')):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            rc.config_check(bad)
            pid.check_contrasts(bad)
    p38 = tf.load(REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-20261005.json', False, True)
    assert pid.design(rc.config_check(p38)) is pid.DESIGNS['p38']  # P38 routes as before
    p38_with_p40_arm = copy.deepcopy(p38)
    p38_with_p40_arm['arms'][1] = pid.BASELINES[SHUF]
    with pytest.raises(ValueError, match='must be exactly'):
        rc.config_check(p38_with_p40_arm)
    earlier = {s for block in pid.P40_EARLIER_SEEDS.values() for s in block}
    assert set(range(60)) == earlier and not earlier & set(pid.P40_SEEDS)
    c = {x['name']: x for x in pid.P40_CONTRASTS}
    assert pid.P40_PRIMARY == 'pitcher_id_vs_row_shuffle' and (c[pid.P40_PRIMARY]['candidate'],
                                                               c[pid.P40_PRIMARY]['reference']) == (CAND, SHUF)
    assert c[pid.P40_PRIMARY]['parameter_matched'] is True and c[pid.P40_PRIMARY]['compute_matched'] is True


def test_row_shuffle_removes_the_association_and_keeps_frequencies():
    rng = np.random.default_rng(0)
    pitcher = rng.choice([101, 102, 103, 999], 5000)
    out = pidm.shuffled_pitcher(pitcher, 'dev', pid.SHUFFLE_SEED)
    assert np.array_equal(np.sort(out), np.sort(pitcher)) and not np.array_equal(out, pitcher)
    assert np.array_equal(out, pidm.shuffled_pitcher(pitcher.copy(), 'dev', pid.SHUFFLE_SEED))  # fixed
    assert not np.array_equal(out, pidm.shuffled_pitcher(pitcher, 'mlb_dev', pid.SHUFFLE_SEED))  # per split
    assert abs(np.mean(out == pitcher) - np.sum((np.bincount(pitcher) / len(pitcher)) ** 2)) < 0.03  # chance only
    ids = np.array([101, 102])
    inputs = (np.zeros((5000, 2, 3)), np.ones((5000, 2), bool), np.zeros((5000, 4), np.float32))
    (_, _, ident), share = pidm.id_inputs(inputs, pitcher, ids, 'dev')
    (_, _, shuf), share_s = pidm.id_inputs(inputs, pitcher, ids, 'dev', 'row_shuffle', pid.SHUFFLE_SEED)
    assert share == share_s and np.array_equal(np.sort(ident[:, -1]), np.sort(shuf[:, -1]))
    assert not np.array_equal(ident[:, -1], shuf[:, -1]) and np.array_equal(ident[:, :-1], shuf[:, :-1])
    with pytest.raises(ValueError, match='Unknown pitcher-ID index source'):
        pidm.id_inputs(inputs, pitcher, ids, 'dev', 'by_label')


def nets():
    kw = pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic'])
    out = []
    for p in (0., 0.2):
        torch.manual_seed(3)
        out.append(pidm.PitcherIdPeriodicResidualNetwork(8, 24, 3, tm.CONTINUOUS, n_ids=5, id_seed=11, id_dropout=p, **kw))
    for net in out:
        with torch.no_grad():
            net.id_table.normal_(generator=torch.Generator().manual_seed(9))
    return out


def test_id_dropout_is_train_only_private_and_refuses_bad_rates():
    rng = np.random.default_rng(1)
    (tokens, valid, context), _ = tm.arrays(400, rng)
    (_, _, with_id), _ = pidm.id_inputs((tokens, valid, context), rng.integers(1, 6, 400), np.arange(1, 6))
    batch = (torch.as_tensor(tokens), torch.as_tensor(valid), torch.as_tensor(with_id))
    plain, drop = nets()
    assert 'id_dropout' not in plain.config and drop.config['id_dropout'] == 0.2
    plain.eval(), drop.eval()
    assert torch.equal(plain(*batch), drop(*batch))  # eval / calibration / prediction use the true index
    plain.train(), drop.train()
    torch.manual_seed(7)
    out_p = plain(*batch)
    state = torch.get_rng_state()
    torch.manual_seed(7)
    out_d = drop(*batch)
    assert torch.equal(torch.get_rng_state(), state)  # the mask uses a private generator: global stream unchanged
    changed = (out_p != out_d).any(1).float().mean().item()
    assert 0.1 < changed < 0.3, changed  # about 20 % of rows lost their id vector
    with pytest.raises(ValueError, match='id_dropout'):
        pidm.PitcherIdPeriodicResidualNetwork(8, 24, 3, tm.CONTINUOUS, n_ids=5, id_seed=11, id_dropout=1.,
                                              **pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic']))


def test_end_to_end_cpu_family_scores_four_contrasts(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)
    exp = w.make_export(tmp_path / 'export')
    path, digest, splits = exp
    cfg = load(registered=True)
    fam = tmp_path / 'fam'
    sources = rc.own_hashes()
    for arm in pid.P40_ARM_ORDER:
        for seed in pid.P40_SEEDS:
            dest = rc.run_task(cfg, path, digest, fam, arm, seed, 'cpu', {})
            identity = rc.expected_identity(cfg, digest, arm, seed, False, 'cuda', sources)
            rc.write_json_atomic(dest / 'identity.json', identity)
            fit = w.read(dest / 'fit.json')
            fit.update(device='cuda', runtime={**tf.RUNTIME, 'device_uuid': w.GPU3, 'bound_uuid': w.GPU3})
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest, identity)
            record = fit.get(pid.FLAG)
            if arm == REF:
                assert record is None
            else:
                assert fit['parameter_count'] == pid.BASE_PARAMETERS + pid.id_parameters(2)
                extra = {k: record.get(k) for k in ('index_source', 'shuffle_seed', 'id_dropout')}
                assert extra == {CAND: {'index_source': None, 'shuffle_seed': None, 'id_dropout': None},
                                 SHUF: {'index_source': 'row_shuffle', 'shuffle_seed': pid.SHUFFLE_SEED,
                                        'id_dropout': 0.},
                                 DROP: {'index_source': 'identity', 'shuffle_seed': None, 'id_dropout': 0.2}}[arm]
                model = rc.load_model(rc.arm_spec(cfg, arm), dest / 'model.pt', 'cpu')
                assert model.id_dropout == (0.2 if arm == DROP else 0.)
    with pytest.raises(ValueError, match='ID dropout'):  # archive of one arm loaded as another
        rc.load_model(rc.arm_spec(cfg, CAND), rc.task_dir(fam, DROP, 60, False) / 'model.pt', 'cpu')
    result = tf.run_score(tmp_path, cfg, exp, fam, dest='score')
    assert result['decision'] is None and result['seed_block']['primary'] == pid.P40_PRIMARY
    assert set(result['seed_block']['parameters']) == set(pid.P40_ARM_ORDER)
    assert len({result['seed_block']['parameters'][a] for a in (CAND, SHUF, DROP)}) == 1
    for split in ('dev', 'mlb_dev'):
        assert set(result['pitcher_id_contrasts'][split]) == {c['name'] for c in pid.P40_CONTRASTS}
        unk = int(np.isin(splits[split]['pitcher'], [103, 999]).sum())  # TRUE pitcher membership for every arm
        assert result['pitcher_id_rows'][split]['unk']['rows'] == unk
        assert set(result['pitcher_id_rows'][split]['in_vocab']['arms']) == set(pid.P40_ARM_ORDER)
        assert list(result['splits'][split]) == list(pid.P40_ARM_ORDER)
        assert result['splits'][split][SHUF]['role'] == 'control'
        prim = result['pitcher_id_contrasts'][split][pid.P40_PRIMARY]['all']
        assert set(prim['seeds']) == {str(s) for s in pid.P40_SEEDS} and prim['parameter_matched'] is True
    assert result['seed_block']['same_gpu_pairs'] == {c['name']: 5 for c in pid.P40_CONTRASTS}
    assert 'row-shuffled-id control' in result['seed_block']['note']
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert '(주 대비)' in table and '행 섞기 대조군' in table and table.count('## mlb_dev — ') == 4


# ------------------------------------------------------------------ smoke validity, registration, pipeline template

import p40_launch as launch  # noqa: E402
from datetime import datetime  # noqa: E402

TEMPLATE = LAUNCH / 'run-p40-pipeline.TEMPLATE.py'


def smoke_fits():
    fits = {}
    for arm in pid.P40_ARM_ORDER:
        for seed in pid.P40_SEEDS:
            fit = {'runtime': {'bound_uuid': pid.RETAINED_GPU}, 'dev_used_for_fit_or_selection': False,
                   'report': {'seconds': 30. + (arm != REF), 'epochs_run': 2}}
            if arm != REF:
                fit[pid.FLAG] = {**{k: 1 for k in pid.VOCAB_KEYS}, 'n_vocab': 3, 'train_max_date': 20250430,
                                 **{k: v for k, v in launch.MODES[arm].items() if v is not None}}
            fits.setdefault(arm, {})[seed] = fit
    return fits


def test_smoke_validity_refuses_wrong_mode_gpu_and_vocab():
    good = launch.evaluate(smoke_fits())
    assert good['valid'] and good['tasks'] == 20 and good['bound_uuids'] == [pid.RETAINED_GPU]
    assert good['over_reference_descriptive'][CAND] == pytest.approx(31 / 30)
    for mutate, match in ((lambda f: f[SHUF][61][pid.FLAG].pop('index_source'), 'id mode record'),
                          (lambda f: f[DROP][62][pid.FLAG].update(id_dropout=0.5), 'id mode record'),
                          (lambda f: f[CAND][60][pid.FLAG].update(n_vocab=4), 'different vocabulary'),
                          (lambda f: f[REF][63]['runtime'].update(bound_uuid=pid.RELEASED_GPUS[0]), 'released GPU'),
                          (lambda f: f[REF][64].update({pid.FLAG: {}}), 'reference carries'),
                          (lambda f: f[DROP].pop(64), 'fit.json missing')):
        fits = smoke_fits()
        mutate(fits)
        record = launch.evaluate(fits)
        assert not record['valid'] and any(match in e for e in record['errors']), (match, record['errors'])


def test_registration_changes_only_registration_paths_and_checks_the_clock():
    smoke = {**launch.evaluate(smoke_fits()), 'config_sha256': launch.DRAFT_SHA,
             'export_manifest_sha256': launch.MANIFEST_SHA, 'source_sha256': {'a': 'b'}}
    text = RUN.read_text()
    now = datetime(2026, 10, 6, 1, 20, tzinfo=launch.KST)
    cfg = launch.build_registered(text, smoke, 'f' * 64, '01:15', 'abcdef1', 'root go', now)
    draft = json.loads(text)
    assert {k for k in set(cfg) | set(draft) if cfg.get(k) != draft.get(k)} == \
        {'status', 'decision', 'execution', 'registration'}
    assert cfg['status'] == 'REGISTERED' and cfg['execution'] == {'enabled': True, 'executable': True}
    rc.config_check(tf.load(RUN, False, True) | {k: cfg[k] for k in ('status', 'execution')})
    for args, match in (((text, smoke, 'f' * 64, '02:10', 'abcdef1', 'g', now.replace(hour=2, minute=10)), 'launch_by'),
                        ((text, smoke, 'f' * 64, '01:40', 'abcdef1', 'g', now), 'real clock'),
                        ((text, smoke, 'f' * 64, '00:50', 'abcdef1', 'g', now), 'real clock'),
                        ((text, {**smoke, 'valid': False}, 'f' * 64, '01:15', 'abcdef1', 'g', now), 'not valid'),
                        ((text, {**smoke, 'bound_uuids': []}, 'f' * 64, '01:15', 'abcdef1', 'g', now), 'identity'),
                        ((text + ' ', smoke, 'f' * 64, '01:15', 'abcdef1', 'g', now), 'pinned sha')):
        with pytest.raises(ValueError, match=match):
            launch.build_registered(*args)


def test_pipeline_template_fails_closed_and_names_only_gpu3():
    text = TEMPLATE.read_text()
    compile(text, str(TEMPLATE), 'exec')
    assert pid.RELEASED_GPUS[0] not in text and text.count(pid.RETAINED_GPU) == 1
    assert "draftsha='" + RUN_SHA256 + "'" in text and "LOCKS='/home/sds733/pitcheezy-autonomy/gpu-locks'" in text
    assert text.count("'FILL_AT_REGISTRATION'") == 6 and "{'done':20}" in text  # 5 placeholders + 1 guard
    p38 = (LAUNCH / 'run-p38-pipeline.TEMPLATE.py').read_text()
    for fn in ('def save(', 'def interrupted(', 'def execute(', 'def queue('):
        block = lambda t: t[t.index(fn):t.index('\n\n', t.index(fn))]  # noqa: E731
        assert block(text) == block(p38), fn

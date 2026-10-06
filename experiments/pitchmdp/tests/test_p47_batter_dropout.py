"""P47 apparatus tests: generated CPU fixtures only, no GPU, real datasets or experiment registration."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT / 'scripts'), str(PROJECT / 'docs/launch'), str(Path(__file__).resolve().parent)]
import run_cuda_direct as rc
import score_cuda_direct as sc
import test_p43_batter_id as old
import test_cuda_frontier_run as tf
from make_p47_config import build
from p47_plan import plan
from pitchmdp import direct_pitcher_id_registry as pid
from pitchmdp import direct_pitcher_batter_id_model as pbm
from pitchmdp import direct_pitcher_id_model as pm

RUN = REPO / 'configs/DRAFT-CUDA-BATTER-DROPOUT-20261006.json'
REF, BOTH, DROP, ZERO = pid.P47_ARM_ORDER


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)


def model(arm=DROP, seed=80):
    return rc.build_model(pid.BASELINES[arm], seed, 'cpu', None, [np.array([101, 102]), np.array([201, 202])])


def net(arm=DROP, seed=80):
    m = model(arm, seed)
    tokens, valid, context, _ = old.toy_inputs()
    return m.network(context.shape[1] + 2, tokens.shape[2], tokens.shape[1])


def test_draft_and_p43_invariance_and_disabled_plan():
    draft = build()
    assert json.dumps(draft, indent=2, ensure_ascii=False) + '\n' == RUN.read_text()
    rc.config_check(copy.deepcopy(draft))
    assert draft['seeds'] == [80, 81, 82, 83] and draft['fits'] == 16
    assert draft['status'] == 'DRAFT' and draft['execution']['enabled'] is False and 'registration' not in draft
    assert all(draft['queue'][k] is None for k in ('date', 'launch_by', 'no_new_fit_after', 'kill_own_after', 'aggregate_by'))
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.require_executable(draft, smoke=False)
    p43 = json.loads(old.RUN.read_text())
    assert hashlib.sha256(old.RUN.read_bytes()).hexdigest() == old.RUN_SHA256
    assert not pid.is_p43(draft) and pid.is_p43(p43)
    for k in ('export', 'training', 'calibration', 'data_rules', 'device', pid.BATTER_SIDECAR_KEY):
        assert draft[k] == p43[k]
    assert draft['arms'][:2] == [p43['arms'][0], p43['arms'][3]]
    assert plan()['fits'] == 16 and plan()['fit_projection_seconds'] == 4894
    assert plan()['execution_enabled'] is False
    with pytest.raises(ValueError):
        plan('EXP-P43-001')
    import p43_launch
    with pytest.raises(ValueError, match='Not a P43'):
        p43_launch.family(draft)  # P47 is intentionally not wired to the old launch/register path
    for change in (lambda c: c.update(seeds=[70, 71, 72, 73]),
                   lambda c: c['arms'][2][pid.FLAG].update(id_dropout=.1),
                   lambda c: c['comparison'][pid.CONTRAST_KEY][0].update(primary_subset='all'),
                   lambda c: c.pop(pid.BATTER_SIDECAR_KEY)):
        bad = copy.deepcopy(draft)
        change(bad)
        with pytest.raises(ValueError):
            rc.config_check(bad)


def test_dropout_private_independent_train_only_and_reproducible():
    a, b = net(), net()
    indices = torch.ones((4096, 2), dtype=torch.long)
    state = torch.random.get_rng_state().clone()
    changed = a.dropout_indices(indices)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(changed, b.dropout_indices(indices))
    assert not torch.equal(changed[:, 0], changed[:, 1])
    assert .15 < (changed == 0).float().mean().item() < .25
    assert not torch.equal(changed, a.dropout_indices(indices))
    a.eval()
    assert torch.equal(a.dropout_indices(indices), indices)
    assert torch.equal(net(BOTH).dropout_indices(indices), indices)


def test_dropout_eval_logits_equal_no_dropout_for_identical_parameters():
    torch.manual_seed(7)
    a = net(DROP)
    torch.manual_seed(7)
    b = net(BOTH)
    with torch.no_grad():
        for table in a.id_tables:
            table.copy_(torch.arange(table.numel()).reshape_as(table) / 100)
    b.load_state_dict(a.state_dict())
    tokens, valid, context, _ = old.toy_inputs()
    context = pbm.append_indices((tokens, valid, context), [np.ones(64, int), np.ones(64, int)])[2]
    assert np.array_equal(old.logits(a, tokens, valid, context), old.logits(b, tokens, valid, context))


def test_zero_unk_stays_zero_after_adamw_and_gate_at_eval():
    a = net(ZERO)
    tokens, valid, context, rng = old.toy_inputs(256)
    context = pbm.append_indices((tokens, valid, context),
                                 [rng.integers(0, 3, 256), rng.integers(0, 3, 256)])[2]
    args = tuple(torch.as_tensor(v) for v in (tokens, valid, context))
    opt = torch.optim.AdamW(a.parameters(), lr=.01, weight_decay=.01)
    y = torch.as_tensor(rng.integers(0, 10, 256))
    for _ in range(4):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(a(*args), y).backward()
        assert all(torch.count_nonzero(t.grad[0]) == 0 for t in a.id_tables)
        opt.step()
        assert all(torch.count_nonzero(t[0]) == 0 for t in a.id_tables)
    assert all(torch.count_nonzero(t[1:]) > 0 for t in a.id_tables)
    a.eval()
    before = a(*args).detach()
    # Even accidental nonzero row storage has no effective UNK contribution; archive loading also rejects it.
    with torch.no_grad():
        for table in a.id_tables:
            table[0].fill_(50.)
    assert torch.equal(before, a(*args))


@pytest.mark.parametrize('arm', [BOTH, DROP, ZERO])
def test_archive_round_trip_modes_and_corrupt_gate_refused(tmp_path, arm):
    tokens, valid, context, rng = old.toy_inputs(96)
    arrays = pbm.append_indices((tokens, valid, context), [rng.integers(0, 3, 96), rng.integers(0, 3, 96)])
    y = rng.integers(0, 10, 96)
    m = model(arm)
    m.fit(arrays, y, arrays, y, epochs=1, patience=1, batch_size=32, learning_rate=.001)
    path = tmp_path / 'model.pt'
    m.save(path)
    hashes = {c: pm.vocab_sha256(v) for c, v in zip(pbm.COLUMNS, m.vocab_ids)}
    back = rc.load_model(pid.BASELINES[arm], path, 'cpu', ids_sha256=hashes)
    assert np.array_equal(m.logits(arrays), back.logits(arrays))
    if arm == BOTH:
        assert 'id_dropout' not in back.net.config and 'freeze_unk' not in back.net.config
    else:
        with pytest.raises(ValueError, match='dropout / UNK gate'):
            rc.load_model(pid.BASELINES[BOTH], path, 'cpu', ids_sha256=hashes)
    if arm == ZERO:
        payload = torch.load(path, weights_only=True)
        payload['state_dict']['id_tables.0'][0, 0] = 1.
        torch.save(payload, path)
        with pytest.raises(ValueError, match='Frozen UNK'):
            rc.load_model(pid.BASELINES[arm], path, 'cpu', ids_sha256=hashes)


def test_strata_are_true_two_column_membership_with_empty_cells():
    p, b = np.array([1, 1, 0, 0, 2]), np.array([1, 0, 1, 0, 0])
    masks = pid.p47_subsets(p, b)
    assert {k: np.flatnonzero(v).tolist() for k, v in masks.items()} == {
        'all': [0, 1, 2, 3, 4], 'batter_known': [0, 2], 'batter_unknown': [1, 3, 4],
        'pitcher_known_batter_known': [0], 'pitcher_known_batter_unknown': [1, 4],
        'pitcher_unknown_batter_known': [2], 'pitcher_unknown_batter_unknown': [3]}
    assert not pid.p47_subsets(np.ones(4), np.ones(4))['batter_unknown'].any()
    with pytest.raises(ValueError, match='aligned'):
        pid.p47_subsets([1], [1, 0])


def test_synthetic_16_task_runner_scorer_and_mode_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # fixture speed only; production constant remains 10,000
    exp, directory, sha = old.synthetic(tmp_path)
    path, digest, splits = exp
    cfg = tf.load(RUN, registered=True)
    cfg[pid.BATTER_SIDECAR_KEY].update(dirname=old.DIRNAME, sidecar_sha256=sha)
    cfg = rc.config_check(cfg)
    fam, sources = tmp_path / 'synthetic-family', rc.own_hashes()
    last_fit = None
    for arm in pid.P47_ARM_ORDER:
        for seed in pid.P47_SEEDS:
            dest = rc.run_task(cfg, path, digest, fam, arm, seed, 'cpu', {'synthetic_test': True})
            fit = json.loads((dest / 'fit.json').read_text())
            if arm == ZERO:
                assert fit[pid.FLAG]['unk_rows_exact_zero'] == {'pitcher': True, 'batter': True}
            if arm == BOTH:
                assert 'id_dropout' not in fit[pid.FLAG] and 'freeze_unk' not in fit[pid.FLAG]
            # Simulated CUDA identity is confined to generated fixture files, as in the P43 scorer test.
            identity = rc.expected_identity(cfg, digest, arm, seed, False, 'cuda', sources)
            rc.write_json_atomic(dest / 'identity.json', identity)
            fit.update(device='cuda', runtime={**tf.RUNTIME, 'device_uuid': pid.RETAINED_GPU,
                                               'bound_uuid': pid.RETAINED_GPU})
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest, identity)
            last_fit = fit
    result = tf.run_score(tmp_path, cfg, exp, fam, dest='synthetic-score')
    assert result['decision'] is None and result['ranking'] is None
    assert result['seed_block']['primaries'] == pid.P47_PRIMARIES
    for split in ('dev', 'mlb_dev'):
        p = np.isin(splits[split]['pitcher'], [101, 102]).astype(int)
        b = np.isin(splits[split]['batter'], [201, 202]).astype(int)
        masks = pid.p47_subsets(p, b)
        assert all(m.any() for m in masks.values())
        for subset, mask in masks.items():
            r = result['pitcher_id_rows'][split][subset]
            assert r['rows'] == int(mask.sum())
            assert set(r['arm_metrics']) == set(pid.P47_ARM_ORDER)
            assert set(r['arm_metrics'][DROP]['seeds']) == {'80', '81', '82', '83'}
            for c in pid.P47_CONTRASTS:
                pair = result['pitcher_id_contrasts'][split][c['name']][subset]
                expected_primary = (split, subset) == (c.get('primary_split'), c.get('primary_subset'))
                assert (pair['role'] == 'primary') == expected_primary
        assert sum(r['rows'] for k, r in result['pitcher_id_rows'][split].items() if k.startswith('pitcher_')) == len(p)
    assert (tmp_path / 'synthetic-score/table_ko.md').read_text().count('(주 대비)') == 2
    for key, bad in [('id_dropout', .1), ('freeze_unk', False), ('unk_rows_exact_zero', {'pitcher': False, 'batter': True})]:
        changed = copy.deepcopy(last_fit)
        changed[pid.FLAG][key] = bad
        with pytest.raises(ValueError):
            pid.check_fit(cfg, ZERO, changed)

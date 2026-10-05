"""Focused CPU tests: EXP-P43-001 DRAFT (TRAIN-only batter-ID embedding, row-shuffled batter control, pitcher + batter).

Batter sidecar join (row-exact, ids only, 2026 / holdout refusal, tamper detection), DRAFT identity and design guards,
the two-path network (exact reference logits at init, P38 pitcher path verbatim, batter projection shared across the
P43 arms, archive round trip) and a real CPU end-to-end family of 20 synthetic tasks through the runner and the scorer.
No real data, no GPU, no DEV metric.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
import test_direct_pitcher_id_model as tm  # noqa: E402  toy arrays / continuous map
from pitchmdp import batter_sidecar as bs  # noqa: E402
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402
from pitchmdp import direct_pitcher_batter_id_model as pbm  # noqa: E402
from pitchmdp import direct_pitcher_id_model as pidm  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

RUN = REPO / 'configs' / 'DRAFT-CUDA-BATTER-ID-20261006.json'
RUN_SHA256 = '00202862d08725c6898e0d213b8c46c1288d9b7e694343792b1c1a8bcf2cd3c4'
MAKE = LAUNCH / 'make_p43_config.py'
REF, BAT, SHUF, BOTH = pid.REFERENCE, pid.BATTER, pid.BATTER_SHUFFLE, pid.PITCHER_BATTER
DIRNAME = 'batter-sidecar-test'


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def batters(split, n, rng):
    """TRAIN: 201 x 200, 202 x 150 (vocab), 203 x 50 (rare -> UNK); later splits add an unseen 299."""
    if split == 'train':
        return rng.permutation(np.repeat(np.array([201, 202, 203]), [200, 150, 50]))
    return rng.choice(np.array([201, 202, 203, 299]), n)


def write_raw(root, splits, extra=True, name='statcast_2025.parquet', mutate=None):
    cols = {c: [] for c in bs.RAW_COLUMNS}
    for s, arrays in splits.items():
        k = arrays['keys']
        for c, v in zip(('game_pk', 'at_bat_number', 'pitch_number'), k.T):
            cols[c].append(v)
        cols['batter'].append(arrays['batter'])
        cols['pitcher'].append(arrays['pitcher'])
    if extra:  # raw rows that are in no export split (other games) must be ignored
        cols['game_pk'].append(np.array([9999, 9999])), cols['at_bat_number'].append(np.array([1, 1]))
        cols['pitch_number'].append(np.array([1, 2])), cols['batter'].append(np.array([777, 777]))
        cols['pitcher'].append(np.array([5, 5]))
    table = {c: np.concatenate(v).astype(np.int64) for c, v in cols.items()}
    table['description'] = np.array(['secret_outcome'] * len(table['batter']))  # a non-id column that is never read
    if mutate:
        mutate(table)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    pq.write_table(pa.table(table), path)
    return path


def synthetic(tmp_path):
    """(export tuple, splits with batter, sidecar dir, sidecar sha) in tmp_path; the sidecar is a sibling of export."""
    path, digest, splits = w.make_export(tmp_path / 'export')
    for i, (s, arrays) in enumerate(splits.items()):
        arrays['batter'] = batters(s, len(arrays['y']), np.random.default_rng(80 + i)).astype(np.int64)
    raw = write_raw(tmp_path / 'raw', splits)
    _, sha = bs.build(path, digest, [raw], tmp_path / DIRNAME)
    return (path, digest, splits), tmp_path / DIRNAME, sha


def load(sha=None, registered=False):
    cfg = tf.load(RUN, False, True)
    if sha:
        cfg[pid.BATTER_SIDECAR_KEY] = {**cfg[pid.BATTER_SIDECAR_KEY], 'dirname': DIRNAME, 'sidecar_sha256': sha}
    if registered:
        cfg['status'], cfg['execution'] = 'REGISTERED', {'enabled': True, 'executable': True}
    return rc.config_check(cfg)


# ------------------------------------------------------------------ sidecar

def test_join_is_row_exact_and_fails_closed():
    keys = np.array([[10, 1, 1], [10, 1, 2], [11, 2, 1]])
    raw = {'game_pk': np.array([11, 10, 10, 12]), 'at_bat_number': np.array([2, 1, 1, 1]),
           'pitch_number': np.array([1, 2, 1, 1]), 'batter': np.array([3, 2, 1, 9]), 'pitcher': np.array([7, 5, 5, 8])}
    assert bs.join(keys, [5, 5, 7], raw).tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match='raw pitcher differs'):
        bs.join(keys, [5, 6, 7], raw)
    with pytest.raises(ValueError, match='no raw Statcast row'):
        bs.join(np.array([[10, 1, 3]]), [5], raw)
    dup = {k: np.concatenate([v, v[:1]]) for k, v in raw.items()}
    with pytest.raises(ValueError, match='duplicate'):
        bs.join(keys, [5, 5, 7], dup)
    with pytest.raises(ValueError, match='range'):
        bs.combined_key([1], [1000], [1])


def test_raw_file_rule_refuses_2026_holdout_and_unnamed(tmp_path):
    assert bs.check_raw_names(['/x/d20260911-s2325/statcast_2025.parquet'])  # snapshot dir date is not a season
    for bad in ('/x/statcast_2026.parquet', '/x/holdout_2026/statcast_2025.parquet', '/x/statcast.parquet',
                '/x/statcast_2025.csv'):
        with pytest.raises(ValueError, match='Refused'):
            bs.check_raw_names([bad])
    with pytest.raises(ValueError, match='Duplicate'):
        bs.check_raw_names(['/a/statcast_2024.parquet', '/b/statcast_2024.parquet'])


def test_build_reads_ids_only_and_verify_detects_tampering(tmp_path, monkeypatch):
    seen = []
    real = pq.read_table
    monkeypatch.setattr(pq, 'read_table', lambda path, columns=None: seen.append(columns) or real(path, columns=columns))
    (path, digest, splits), directory, sha = synthetic(tmp_path)
    assert seen == [list(bs.RAW_COLUMNS)]  # never the outcome / description column
    manifest = json.loads((path / 'manifest.json').read_text())
    record = bs.verify(directory, manifest, sha)
    for s, arrays in splits.items():
        assert np.array_equal(bs.load(directory, record, s), arrays['batter'])
        assert record['splits'][s]['rows_sha256'] == manifest['splits'][s]['rows_sha256']
    assert np.array_equal(bs.load(directory, record, 'train', 7), splits['train']['batter'][:7])
    with pytest.raises(FileExistsError):
        bs.build(path, digest, [tmp_path / 'raw' / 'statcast_2025.parquet'], directory)
    with pytest.raises(ValueError, match='config pin'):
        bs.verify(directory, manifest, '0' * 64)
    np.save(directory / 'dev' / 'batter.npy', splits['dev']['batter'][::-1].copy())
    with pytest.raises(ValueError, match='array hash'):
        bs.verify(directory, manifest, sha)
    bad = copy.deepcopy(manifest)
    bad['splits']['dev']['rows_sha256'] = '1' * 64
    with pytest.raises(ValueError, match='rows differ'):
        bs.verify(directory, bad, sha, hash_arrays=False)


def test_build_refuses_a_pitcher_mismatch(tmp_path):
    path, digest, splits = w.make_export(tmp_path / 'export')
    for arrays in splits.values():
        arrays['batter'] = np.full(len(arrays['y']), 201, dtype=np.int64)

    def flip(table):
        table['pitcher'][3] += 1
    raw = write_raw(tmp_path / 'raw', splits, mutate=flip)
    with pytest.raises(ValueError, match='raw pitcher differs'):
        bs.build(path, digest, [raw], tmp_path / DIRNAME)
    assert not (tmp_path / DIRNAME).exists()


# ------------------------------------------------------------------ DRAFT and design guards

def test_draft_is_generated_deterministically_and_passes_the_runner_check():
    assert hashlib.sha256(RUN.read_bytes()).hexdigest() == RUN_SHA256
    out = subprocess.run([sys.executable, '-c', f'import runpy, json; ns = runpy.run_path({str(MAKE)!r}); '
                          'print(json.dumps(ns["build"](), indent=2, ensure_ascii=False))'],
                         capture_output=True, text=True, check=True).stdout
    assert out == RUN.read_text()
    raw, p40 = (json.loads(p.read_text()) for p in (RUN, REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json'))
    for key in ('protocol', 'export', 'training', 'calibration', 'data_rules', 'device'):
        assert raw[key] == p40[key], key
    cfg = rc.config_check(copy.deepcopy(raw))
    assert cfg['status'] == 'DRAFT' and cfg['execution']['enabled'] is False and 'registration' not in cfg
    assert cfg['seeds'] == [70, 71, 72, 73, 74] and cfg['fits'] == 20 and cfg['training']['epochs'] == 60
    assert [a['name'] for a in cfg['arms']] == list(pid.P43_ARM_ORDER)
    assert cfg['device'] == {'kind': 'cuda', 'allowed_gpu_uuids': [w.GPU3], 'max_parallel': 1, 'cpu_fallback': False}
    assert cfg['queue']['date'] is None and cfg['queue']['launch_by'] is None  # no window until registration
    assert cfg[pid.BATTER_SIDECAR_KEY]['sidecar_sha256'].startswith('ad2eacc6')
    assert cfg['comparison'][pid.CONTRAST_KEY][0]['name'] == pid.P43_PRIMARY == 'batter_id_vs_row_shuffle'
    assert not w.FORBIDDEN_KEYS & w.keys_deep(cfg) and w.GPU4 not in json.dumps({k: v for k, v in cfg.items() if k != 'queue'})


def test_design_guards():
    base = json.loads(RUN.read_text())
    assert not set(pid.P43_SEEDS) & {s for b in pid.P43_EARLIER_SEEDS.values() for s in b}
    assert {65, 66, 67, 68, 69, 60, 49, 53} <= {s for b in pid.P43_EARLIER_SEEDS.values() for s in b}
    for change, match in ((lambda c: c.pop(pid.BATTER_SIDECAR_KEY), 'batter_sidecar'),
                          (lambda c: c[pid.BATTER_SIDECAR_KEY].update(dirname='../x'), 'batter_sidecar'),
                          (lambda c: c.update(seeds=[60, 61, 62, 63, 64]), 'seeds'),
                          (lambda c: c['arms'][1][pid.FLAG].update(id_columns=['pitcher']), 'differs'),
                          (lambda c: c['arms'][2][pid.FLAG].update(shuffle_seed=40001), 'differs'),
                          (lambda c: c['arms'].pop(3), 'exactly')):
        cfg = copy.deepcopy(base)
        change(cfg)
        with pytest.raises(ValueError, match=match):
            rc.config_check(cfg)
    p40 = json.loads((REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json').read_text())
    p40[pid.BATTER_SIDECAR_KEY] = base[pid.BATTER_SIDECAR_KEY]
    with pytest.raises(ValueError, match='P43 batter-ID design only'):
        rc.config_check(p40)
    assert pid.P43_FAMILY_PROJECTION_SECONDS == 6117.5


# ------------------------------------------------------------------ networks

def toy_inputs(n=64, seed=0):
    """Inputs of the registered arm shape (stripped context of the synthetic export, as the runner feeds them)."""
    rng = np.random.default_rng(seed)
    a = tf.split_arrays(n, seed, 1000)
    cfg = json.loads(RUN.read_text())
    tokens, valid, context = rc.arm_inputs(a, cfg['arms'][0], cfg['export']['workload_columns'])
    return tokens, valid, context, rng


def logits(net, tokens, valid, context):
    net.eval()
    with torch.no_grad():
        return net(torch.as_tensor(tokens), torch.as_tensor(valid), torch.as_tensor(context)).numpy()


def test_two_path_network_starts_at_the_reference_and_shares_paths():
    rng = np.random.default_rng(4)
    (tokens, valid, context), _ = tm.arrays(64, rng)
    cont, kw = tm.CONTINUOUS, pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic'])
    n_context, n_token, length = context.shape[1], tokens.shape[2], tokens.shape[1]
    p_idx, b_idx = rng.integers(0, 4, len(context)), rng.integers(0, 6, len(context))
    torch.manual_seed(70)
    ref = pr.PeriodicResidualNetwork(n_context, n_token, length, cont, **kw)
    torch.manual_seed(70)
    both = pbm.PitcherBatterIdPeriodicResidualNetwork(n_context + 2, n_token, length, cont, [3, 5],
                                                      [70 + pidm.ID_SEED_OFFSET, pbm.batter_id_seed(70)], **kw)
    torch.manual_seed(70)
    p38 = pidm.PitcherIdPeriodicResidualNetwork(n_context + 1, n_token, length, cont, 3, 70 + pidm.ID_SEED_OFFSET, **kw)
    torch.manual_seed(70)
    bat = pidm.PitcherIdPeriodicResidualNetwork(n_context + 1, n_token, length, cont, 5, pbm.batter_id_seed(70), **kw)
    expected = logits(ref, tokens, valid, context)
    ctx2 = pbm.append_indices((tokens, valid, context), [p_idx, b_idx])[2]
    assert np.array_equal(logits(both, tokens, valid, ctx2), expected)  # zero tables: exactly the reference
    assert torch.equal(both.id_projs[0], p38.id_proj) and torch.equal(both.id_projs[1], bat.id_proj)
    assert not torch.equal(both.id_projs[0], both.id_projs[1])
    with torch.no_grad():
        both.id_tables[1][2] += 1.
    moved = logits(both, tokens, valid, ctx2)
    rows = b_idx == 2
    assert np.array_equal(moved[~rows], expected[~rows]) and not np.allclose(moved[rows], expected[rows])
    with pytest.raises(ValueError, match='exact'):
        pbm.append_indices((tokens, valid, context), [p_idx, np.full(len(context), 1 << 24)])


def test_models_round_trip_and_refuse_mismatches(tmp_path):
    tokens, valid, context, rng = toy_inputs(200, 3)
    y = rng.integers(0, 10, len(context))
    cfg = json.loads(RUN.read_text())
    arm = {a['name']: a for a in cfg['arms']}
    ids_p, ids_b = np.array([101, 102, 103]), np.array([201, 202])
    both = rc.build_model(arm[BOTH], 70, 'cpu', None, [ids_p, ids_b])
    ctx = pbm.append_indices((tokens, valid, context), [rng.integers(0, 4, 200), rng.integers(0, 3, 200)])
    both.fit(ctx, y, ctx, y, epochs=1, patience=1, batch_size=64, learning_rate=1e-3)
    both.save(tmp_path / 'both.pt')
    hashes = {'pitcher': pidm.vocab_sha256(ids_p), 'batter': pidm.vocab_sha256(ids_b)}
    back = rc.load_model(arm[BOTH], tmp_path / 'both.pt', 'cpu', ids_sha256=hashes)
    assert np.allclose(back.logits(ctx), both.logits(ctx))
    with pytest.raises(ValueError, match='batter vocabulary'):
        rc.load_model(arm[BOTH], tmp_path / 'both.pt', 'cpu', ids_sha256={**hashes, 'batter': hashes['pitcher']})
    bat = rc.build_model(arm[BAT], 71, 'cpu', None, ids_b)
    ctx1 = pbm.append_indices((tokens, valid, context), [rng.integers(0, 3, 200)])
    bat.fit(ctx1, y, ctx1, y, epochs=1, patience=1, batch_size=64, learning_rate=1e-3)
    assert bat.net.config['id_seed'] == pbm.batter_id_seed(71)
    bat.save(tmp_path / 'bat.pt')
    back = rc.load_model(arm[BAT], tmp_path / 'bat.pt', 'cpu', ids_sha256={'batter': hashes['batter']})
    assert np.allclose(back.logits(ctx1), bat.logits(ctx1))
    with pytest.raises(ValueError):
        rc.build_model(arm[BAT], 71, 'cpu', None, [ids_p, ids_b])  # one-column arm, two vocabularies


# ------------------------------------------------------------------ end to end (runner + scorer, CPU)

def test_end_to_end_cpu_family_scores_four_contrasts(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)
    exp, directory, sha = synthetic(tmp_path)
    path, digest, splits = exp
    cfg = load(sha, registered=True)
    fam = tmp_path / 'fam'
    sources = rc.own_hashes()
    assert {'pitchmdp/batter_sidecar.py', 'pitchmdp/direct_pitcher_batter_id_model.py'} <= set(sources)
    for arm in pid.P43_ARM_ORDER:
        for seed in pid.P43_SEEDS:
            dest = rc.run_task(cfg, path, digest, fam, arm, seed, 'cpu', {})
            identity = rc.expected_identity(cfg, digest, arm, seed, False, 'cuda', sources)
            rc.write_json_atomic(dest / 'identity.json', identity)
            fit = w.read(dest / 'fit.json')
            fit.update(device='cuda', runtime={**tf.RUNTIME, 'device_uuid': w.GPU3, 'bound_uuid': w.GPU3})
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest, identity)
            block = fit.get(pid.FLAG)
            if arm == REF:
                assert block is None and fit['parameter_count'] == pid.BASE_PARAMETERS
                continue
            columns = pid.BASELINES[arm][pid.FLAG]['id_columns']
            assert block['id_columns'] == columns and set(block['vocabs']) == set(columns)
            assert block['vocabs']['batter']['n_vocab'] == 2 and block['vocabs']['batter']['train_unk_rows'] == 50
            assert fit['parameter_count'] == pid.BASE_PARAMETERS + sum(
                pid.id_parameters(block['vocabs'][c]['n_vocab']) for c in columns)
            assert block['index_source'] == ('row_shuffle' if arm == SHUF else 'identity')
            share = block['vocabs']['batter']['unk_share']
            if arm != SHUF:  # identity: UNK share of the TRUE batter column (203 rare, 299 unseen)
                assert share['dev'] == pytest.approx(float(np.isin(splits['dev']['batter'], [203, 299]).mean()))
            model = rc.load_model(rc.arm_spec(cfg, arm), dest / 'model.pt', 'cpu')
            assert model.net.config.get('id_seed', None) in (None, pbm.batter_id_seed(seed))
    result = tf.run_score(tmp_path, cfg, exp, fam, dest='score')
    assert result['decision'] is None and result['seed_block']['primary'] == pid.P43_PRIMARY
    assert result['seed_block']['subset_column'] == 'batter' and set(result['vocabulary']) == {'pitcher', 'batter'}
    params = result['seed_block']['parameters']
    assert params[BAT] == params[SHUF] > params[REF] and params[BOTH] > params[BAT]
    for split in ('dev', 'mlb_dev'):
        assert set(result['pitcher_id_contrasts'][split]) == {c['name'] for c in pid.P43_CONTRASTS}
        unk = int(np.isin(splits[split]['batter'], [203, 299]).sum())  # TRUE batter membership for every arm
        assert result['pitcher_id_rows'][split]['unk']['rows'] == unk
        assert list(result['splits'][split]) == list(pid.P43_ARM_ORDER)
        prim = result['pitcher_id_contrasts'][split][pid.P43_PRIMARY]['all']
        assert set(prim['seeds']) == {str(s) for s in pid.P43_SEEDS} and prim['parameter_matched'] is True
    assert 'row-shuffled batter id' in result['seed_block']['note']
    assert 'batter_sidecar.json' in result['provenance']['export']['scored_arrays_sha256']
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert '타자 ID 임베딩' in table and '(주 대비)' in table and table.count('## mlb_dev — ') == 4
    # a tampered sidecar after the fits is refused by the scorer
    np.save(directory / 'mlb_dev' / 'batter.npy', splits['mlb_dev']['batter'][::-1].copy())
    with pytest.raises(ValueError, match='array hash'):
        tf.run_score(tmp_path, cfg, exp, fam, dest='score2')

"""B2 (LightGBM) checks of the EXP-OPE-RES-02 denominator. Not collected by the default run (file name); executed by
test_ope_a2b_denominator.IsolatedLightGBMTests in a fresh interpreter with DYLD_LIBRARY_PATH=<torch/lib> so torch and
LightGBM share one OpenMP runtime (the A3 launch rule). SYNTHETIC ONLY: the A3 synthetic world and S6-shaped ledger
(behavior_calibration_synthetic), a small LightGBM B2 frozen into a synthetic A3 freeze-primary layout."""
import hashlib
import json

import numpy as np
import pytest

from behavior_calibration_synthetic import make_ledger, make_world
from pitchmdp import ope_denominator_a2b as dn
from pitchmdp import ope_registry_a2b as rb
from pitchmdp import ope_sealed_a2b as sa
from pitchmdp.behavior_calibration import Temperature
from pitchmdp.behavior_calibration_registry import B2_FEATURES, B2_PARAMS
from pitchmdp.behavior_features import split_table
from pitchmdp.behavior_model import Calibrated, LGBMBehavior
from pitchmdp.policy_artifacts import IntegrityError

PARAMS = {**B2_PARAMS, 'min_data_in_leaf': 20}  # synthetic size (as the A3 isolated B2 tests)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope='module')
def world():
    return make_world(0)


@pytest.fixture(scope='module')
def frozen(world, tmp_path_factory):
    train, es = world['tables']['train'], world['tables']['earlystop']
    rows = np.flatnonzero(es.frame.S_in.to_numpy())
    mask = world['b0'].repertoire.mask(es.keys.values['pitcher'][rows])
    model, best, record = LGBMBehavior.train(train.matrix(B2_FEATURES, None), train.frame.label.to_numpy(),
                                             es.matrix(B2_FEATURES, rows), es.frame.label.to_numpy()[rows], mask,
                                             params=PARAMS, rounds=8)
    root = tmp_path_factory.mktemp('a3-attempt')
    files = {'02-fit/b2/model.txt': model.encode(),
             '02-fit/b2/b2.json': json.dumps({'best_round': best, 'record': {'best_round': best}}).encode(),
             '03-calibrate/calibrators/B2.json': json.dumps({'status': 'CALIBRATED',
                                                             'identity': Temperature(0.9).identity()}).encode()}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {'experiment': 'EXP-BC-CAL-01', 'manifest': 'freeze-primary',
                'artifacts': {rel: {'sha256': sha(data), 'bytes': len(data)} for rel, data in files.items()}}
    (root / '04-freeze-primary').mkdir()
    path = root / '04-freeze-primary' / 'manifest.json'
    path.write_bytes(json.dumps(manifest).encode())
    return {'path': path, 'sha256': sha(path.read_bytes()), 'artifacts': {rel: sha(d) for rel, d in files.items()},
            'root': root, 'model': model, 'best': best}


def test_frozen_b2_loads_only_with_its_pins(world, frozen):
    model = dn.load_frozen_b2(frozen['path'], frozen['sha256'], frozen['artifacts'], world['b0'].repertoire)
    assert isinstance(model, Calibrated) and model.model.best_round == frozen['best']
    table = world['tables']['temperature']
    p = model.batch(table, np.arange(min(50, len(table))))
    support = world['b0'].repertoire.mask(table.keys.values['pitcher'][:len(p)])
    assert (p[~support] == 0).all() and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    with pytest.raises(IntegrityError, match='manifest differs'):
        dn.load_frozen_b2(frozen['path'], 'f' * 64, frozen['artifacts'], world['b0'].repertoire)
    wrong = dict(frozen['artifacts'], **{'02-fit/b2/model.txt': 'e' * 64})
    with pytest.raises(IntegrityError, match='differs from its pin'):
        dn.load_frozen_b2(frozen['path'], frozen['sha256'], wrong, world['b0'].repertoire)
    with pytest.raises(IntegrityError, match='three frozen B2 files'):
        dn.load_frozen_b2(frozen['path'], frozen['sha256'], {'02-fit/b2/model.txt': 'e' * 64}, world['b0'].repertoire)
    with pytest.raises(IntegrityError, match='lightgbm'):
        dn.load_frozen_b2(frozen['path'], frozen['sha256'], frozen['artifacts'], world['b0'].repertoire,
                          lightgbm_version='0.0.0')


def test_b2_rows_end_to_end_on_a_synthetic_s6_shaped_ledger(world, frozen, tmp_path):
    ledger = make_ledger(tmp_path / 'ledger', world, games=4)
    decisions = list(ledger['runtime'].ledger.iter_decisions())
    facts = ledger['facts']
    e0 = dn.e0_map(facts)
    e0_ids = {pa for pa, (inside, _) in e0.items() if inside}
    frame = world['frame']
    dev = split_table(frame.loc[frame.split.eq('dev')], 'dev', world['ctx'])[0]
    model = dn.load_frozen_b2(frozen['path'], frozen['sha256'], frozen['artifacts'], world['b0'].repertoire)
    rows = dn.b2_decision_rows(decisions, e0)
    assert rows
    vectors = dn.decision_vectors(model, dev, rows)
    validity = dn.b2_validity(decisions, vectors, e0_ids)
    assert validity['pass'], validity
    weights, b0 = dn.b2_weights(decisions, e0, vectors, large=10.0, mei=0.001, draws=50, seed=rb.ROLE_VALUES['game_bootstrap'])
    again, _ = dn.b2_weights(decisions, e0, vectors, large=10.0, mei=0.001, draws=50, seed=rb.ROLE_VALUES['game_bootstrap'])
    assert dn.gint_r3(again, weights)['result'] == 'PASS'  # deterministic recomputation (G-INT-R3 form)
    assert weights['kl_new_vs_b0']['mean'] > 0
    surrogates = {pa: {'R1_a': (0.5, 'class'), 'R1_b': (0.5, 'bip_quality'), 'spray_missing': False}
                  for pa, info in facts.items() if info.get('reward') is not None}
    public, sealed, _ = sa.evaluate(
        lambda: iter(decisions), facts, mode=rb.MODE_REHEARSAL, gint={'synthetic': 'PASS'},
        decisions_b2=dn.b2_decision_stream(decisions, vectors, e0_ids), b2={'validity': validity, 'weights': weights},
        surrogates=surrogates, train_residuals={}, r1_paths={}, draws=50, minimum_games=2, minimum_pitchers=1,
        minimum_pa_starts=5, placebos=2)
    assert set(public['pairs']) == set(rb.ROWS)
    assert public['B2']['validity']['pass'] is True
    assert 'R5_g' in public['B2'] and 'ess' in public['B2']
    public_bytes, _ = sa.seal(public, sealed)
    assert rb.validate_public(json.loads(public_bytes))

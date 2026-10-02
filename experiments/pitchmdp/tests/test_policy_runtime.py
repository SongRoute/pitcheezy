"""SYNTHETIC-ONLY tests for ML-POLICY-RUNTIME-v1 (COOP-016).

Tiny in-memory fits, fake checkpoints/JSON in temp dirs, constant fake G0 members. No real
data, model weights, cache or 2026 access. Expected values are computed independently of
the runtime (hand arithmetic with scipy softmax), not by calling the code under test.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_matrix_policy import frame, Context  # existing synthetic PolicyInputs fixture pieces
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_policy import PolicyInputs, SupportedBC, context_key
from pitchmdp.matrix_sharing import SharingPredictor
from pitchmdp.rollout_policy import BCRecord, BudgetExceeded, CategoricalBC, PAState, PastPitch, RowBudget
from pitchmdp.sequence_data import PhysicalNormalizer
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_runtime as pr

SHA = lambda b: hashlib.sha256(b).hexdigest()
COMMIT = 'f83caea2d8540992a446b168b498f910111478b0'
PROV = {'train_start': '2023-05-15', 'train_end': '2025-04-30', 'dates': 'declared_unverified',
        'train_rows': 11, 'source_ids': {'synthetic': 'a' * 64}, 'config_sha256': 'b' * 64,
        'code_commit': COMMIT, 'data_version': 'synthetic-v1'}
PHYS = (0.,) * 8


def past(action, balls=0, strikes=0):
    return PastPitch(action, PHYS, 'ball', balls, strikes)


def records():
    """Pitcher 9: FF/SL/CH (CH once); pitcher 11 only KC; histories give several cells."""
    out = []
    for b, s, prev, action in [(0, 0, None, 'FF'), (0, 0, None, 'FF'), (0, 0, None, 'SL'),
                               (1, 0, 'FF', 'SL'), (1, 0, 'FF', 'FF'), (0, 1, 'SL', 'CH'),
                               (2, 2, 'SL', 'FF'), (0, 0, None, 'SL')]:
        out.append(BCRecord(PAState(b, s, '9', 'L', (past(prev),) if prev else (), 'k'), action, 'train'))
    out += [BCRecord(PAState(0, 0, '11', 'R'), 'KC', 'train')] * 3
    return out


def fitted(minimum=1, rows=None):
    return CategoricalBC(20., minimum).fit(rows if rows is not None else records())


def envelope_bytes(schema, payload):
    return pa._canonical_bytes(pa._envelope(schema, payload))


class BCArtifactTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_roundtrip_all_states_and_deterministic_identity(self):
        bc = fitted()
        art = pa.save_train_bc(bc, self.dir / 'bc.json', PROV)
        again = pa.save_train_bc(fitted(rows=list(reversed(records()))), self.dir / 'bc2.json', PROV)
        self.assertEqual(art.file_sha256, again.file_sha256)  # insertion order cannot change identity
        loaded = pa.load_train_bc(self.dir / 'bc.json', art.file_sha256)
        self.assertEqual(loaded.sha256, art.sha256)
        states = [PAState(0, 0, '9', 'L'),                      # seen cell
                  PAState(1, 0, '9', 'L', (past('FF'),)),       # seen cell with previous action
                  PAState(3, 1, '9', 'R', (past('CH'),)),       # unseen cell -> pitcher prior only
                  PAState(0, 0, '11', 'R'),
                  PAState(0, 0, '404', 'L')]                    # unknown pitcher -> league, flagged
        for state in states:
            for frequency in (False, True):
                np.testing.assert_array_equal(loaded.bc.probabilities(state, frequency=frequency),
                                              bc.probabilities(state, frequency=frequency))
            np.testing.assert_array_equal(loaded.bc.support(state), bc.support(state))
            self.assertEqual(loaded.bc.fallback(state), bc.fallback(state))
        self.assertTrue(loaded.bc.fallback(states[-1]))
        self.assertEqual(loaded.vocabulary, ('CH', 'FF', 'KC', 'SL'))
        # Hand check of one smoothed row: pitcher 9 prior (CH1,FF4,SL3)/8, cell (0,0,L,START)=FF2,SL2.
        prior = np.array([1, 4, 0, 3]) / 8
        expect = (np.array([0, 2, 0, 2]) + 20 * prior) / (4 + 20)
        np.testing.assert_allclose(loaded.bc.probabilities(states[0]), expect, atol=1e-15)
        # minimum_action_count is part of the state: CH(1 count) leaves support at 2.
        strict = pa.save_train_bc(fitted(minimum=2), self.dir / 'strict.json', PROV)
        self.assertNotEqual(strict.sha256, art.sha256)
        self.assertEqual(pa.load_train_bc(self.dir / 'strict.json', strict.file_sha256).bc.support(states[0]).tolist(),
                         [False, True, False, True])

    def test_rejections(self):
        art = pa.save_train_bc(fitted(), self.dir / 'bc.json', PROV)
        with self.assertRaises(FileExistsError):
            pa.save_train_bc(fitted(), self.dir / 'bc.json', PROV)
        self.assertEqual(SHA((self.dir / 'bc.json').read_bytes()), art.file_sha256)  # not overwritten
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['bc.json'])     # no temp debris
        with self.assertRaisesRegex(pa.IntegrityError, 'pin is required'):
            pa.load_train_bc(self.dir / 'bc.json', None)
        with self.assertRaisesRegex(pa.IntegrityError, 'differs from pin'):
            pa.load_train_bc(self.dir / 'bc.json', 'c' * 64)
        raw = (self.dir / 'bc.json').read_bytes()
        cut = self.dir / 'cut.json'; cut.write_bytes(raw[:-40])
        with self.assertRaisesRegex(pa.IntegrityError, 'truncated or malformed'):
            pa.load_train_bc(cut, SHA(cut.read_bytes()))

        def rejected(mutate, message, schema=pa.BC_SCHEMA, rehash=True):
            payload = copy.deepcopy(art.payload); mutate(payload)
            env = pa._envelope(schema, payload)
            if not rehash: env['content_sha256'] = canonical_hash(art.payload)
            data = pa._canonical_bytes(env)
            path = self.dir / f'bad{len(list(self.dir.iterdir()))}.json'; path.write_bytes(data)
            with self.assertRaisesRegex(pa.IntegrityError, message):
                pa.load_train_bc(path, SHA(data))  # file pin matches: only content checks can reject

        def bump_cell(p): p['cells'][0]['counts'][0][1] += 1
        rejected(bump_cell, 'content hash mismatch', rehash=False)          # tamper without rehash
        rejected(bump_cell, 'pitcher counts differ from the cell sum')      # rehashed tamper still caught
        rejected(lambda p: None, 'unsupported schema', schema='pitcheezy.train_bc.v2')
        rejected(lambda p: p['vocabulary'].append('ZZ'), 'vocabulary sha256 mismatch')
        rejected(lambda p: p.update(vocabulary=['CH', 'FF', 'KC', 'ZZ'], vocabulary_sha256=pa.vocabulary_sha256(
            ['CH', 'FF', 'KC', 'ZZ'])), 'in-vocabulary action')
        rejected(lambda p: p['league'][0].__setitem__(1, 1.0), 'positive int count')
        rejected(lambda p: p['league'][0].__setitem__(1, True), 'positive int count')
        rejected(lambda p: p['cells'].append(copy.deepcopy(p['cells'][0])), 'duplicate cell key')
        rejected(lambda p: p['cells'][0].__setitem__('balls', 4), 'illegal cell key')
        rejected(lambda p: p['provenance'].__setitem__('train_end', '2026-04-01'), 'TRAIN dates')
        rejected(lambda p: p['provenance'].__setitem__('train_start', '2025-02-30'), 'TRAIN dates')
        rejected(lambda p: p['provenance'].__setitem__('dates', 'verified'), 'observed from rows')
        rejected(lambda p: p['parameters'].__setitem__('prior_strength', -1), 'prior_strength')
        rejected(lambda p: p['provenance'].__setitem__('train_rows', 999), 'train_rows differs')
        rejected(lambda p: p.__setitem__('pitchers', None), 'must be lists')
        # Non-canonical bytes (e.g. pretty-printed) are refused even with a matching pin.
        pretty = self.dir / 'pretty.json'
        pretty.write_text(json.dumps(pa._envelope(pa.BC_SCHEMA, art.payload), indent=1))
        with self.assertRaisesRegex(pa.IntegrityError, 'canonical'):
            pa.load_train_bc(pretty, SHA(pretty.read_bytes()))

    def test_support_table_rejects_coerced_or_outside_bc_masks(self):
        art = pa.save_train_bc(fitted(minimum=2), self.dir / 'bc.json', PROV)
        for mask, message in [([np.nan, 1., 0., 1.], 'bool vector'), ([True, True], 'bool vector'),
                              ([True, True, False, False], 'inside the TRAIN BC support'),   # CH count 1 < 2
                              ([False, False, True, False], 'inside the TRAIN BC support')]:  # KC never thrown by 9
            with self.assertRaisesRegex(pa.IntegrityError, message):
                pa.support_table_payload(art, [('9', 'L', np.array(mask))])
        with self.assertRaisesRegex(pa.IntegrityError, 'known pitcher'):
            pa.support_table_payload(art, [('404', 'L', np.array([False, True, False, False]))])

    def test_export_uses_existing_fit_bc_guard_and_records_observed_dates(self):
        data = frame()
        store = MatrixHistoryStore.from_frame(data, PhysicalNormalizer().fit(data), type_vocabulary=['FF', 'SL'])
        art = pa.export_train_bc(store, data, self.dir / 'bc.json', source_ids={'syn': 'd' * 64},
                                 config_sha256='e' * 64, code_commit=COMMIT, data_version='syn')
        self.assertEqual((art.provenance['train_start'], art.provenance['dates']), ('2025-04-20', 'observed_by_export_train_bc'))
        late = data.assign(game_date='2025-05-01')
        with self.assertRaisesRegex(ValueError, 'frozen TRAIN dates'):
            pa.export_train_bc(store, late, self.dir / 'late.json', source_ids={'syn': 'd' * 64},
                               config_sha256='e' * 64, code_commit=COMMIT, data_version='syn')
        self.assertFalse((self.dir / 'late.json').exists())


# ---------------------------------------------------------------- G0 synthetic fixtures

TYPES = ['CH', 'FF', 'SL']


class FakeNetwork:
    """Stands in for the global MatrixModel inside the real SharingPredictor wrapper."""
    n_classes = 10

    def __init__(self, seed, bad=None):
        self.seed, self.bad, self.context_widths = seed, bad, []

    def logits(self, arrays):
        self.context_widths.append(arrays[2].shape[1])
        n = len(arrays[0])
        z = np.zeros((n, 10)); z[:, 0] = self.seed; z[:, 3] = 4 * arrays[0][:, -1, 0]  # seed & delivery dependent
        if self.bad == 'nan': z[0, 0] = np.nan
        if self.bad == 'shape': z = z[:, :9]
        return z


class FakeFrequency:
    def __init__(self, bad=False): self.bad = bad

    def predict(self, rows):
        raw = np.full((len(rows), 10), .05); raw[:, 1] = .55
        if self.bad: raw[:, 0], raw[:, 2] = -.01, .11
        return raw


def g0_files(root, *, temps=(1.0, 1.1, .9, 1.2, .8), weight=.75, seed_weights=(.7, .71, .66, .67, .68)):
    root.mkdir(parents=True, exist_ok=True)
    files, paths, members = {}, {}, {}
    for seed in range(5):
        ckpt = root / f'seed{seed}.pt'; ckpt.write_bytes(f'fake-checkpoint-{seed}'.encode())
        files[f'seed{seed}_checkpoint'] = {'path': 'registered/elsewhere', 'sha256': SHA(ckpt.read_bytes())}
        paths[f'seed{seed}_checkpoint'] = ckpt
        members[str(seed)] = {'source_arm': 'g' if seed < 3 else 'c1', 'model_sha256': SHA(ckpt.read_bytes()),
                              'delivery_temperature': temps[seed], 'june_model_weight': seed_weights[seed]}
    blobs = {'p11_frozen_calibration': json.dumps({'june_ensemble_model_weight': weight, 'refit_on_whole_mlb': False,
                 'may_delivery_temperatures': {str(s): temps[s] for s in range(5)},
                 'june_seed_model_weights': {str(s): seed_weights[s] for s in range(5)}}),
             'p4_preparation': json.dumps({'baseline_temperature': {'temperature': 1.25},
                                           'features': {'tokens': {'type_vocabulary': TYPES}}}),
             'p4_auxiliary': 'fake-frequency-model'}
    for role, text in blobs.items():
        (root / role).write_text(text)
        files[role] = {'path': 'registered/elsewhere', 'sha256': SHA((root / role).read_bytes())}
        paths[role] = root / role
    bundle = {'protocol': 'g0_research_frozen_v1', 'research_only': True, 'service_promotion': False,
              'members': members, 'files': files}
    return bundle, paths


def g0_inputs(bc):
    data = frame()
    delivery = SimpleNamespace(draws=400, TIERS=[('pitch_type', 'p_throws', 'stand')], fallback=np.ones((400, 8)),
        pools={(0, ('FF', 'R', 'L')): np.full((400, 8), .5), (0, ('SL', 'R', 'L')): np.full((400, 8), -.5)})
    return data, PolicyInputs(data, Context(), delivery, TYPES, 'synthetic-aux', bc)


def FakeMember(seed, bad=None, network_seed=None):
    return SharingPredictor('G0-global', FakeNetwork(seed if network_seed is None else network_seed, bad), {})


def load_g0(bundle, inputs, paths, member=FakeMember, frequency=FakeFrequency()):
    return pa.load_g0_ensemble(bundle, inputs, paths, load_member=lambda seed, record, path: member(seed),
                               load_frequency=lambda path: frequency, frequency_role='p4_auxiliary')


def g0_expected(physics_channel0, balls, temps=(1.0, 1.1, .9, 1.2, .8), weight=.75, bt=1.25):
    members = []
    for seed, t in enumerate(temps):
        z = np.zeros(10); z[0] = seed; z[3] = 4 * physics_channel0
        members.append(softmax(z / t))
    raw = np.full(10, .05); raw[1] = .55
    return weight * np.mean(members, axis=0) + (1 - weight) * softmax(np.log(raw) / bt)


class G0AdapterTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.bc = CategoricalBC().fit([BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'SL', 'CH')])
        self.data, self.inputs = g0_inputs(self.bc)
        self.state = PAState(1, 0, '9', 'L', (), context_key(self.data.iloc[0]))

    def test_five_member_semantics_match_independent_arithmetic(self):
        bundle, paths = g0_files(self.root)
        model = load_g0(bundle, self.inputs, paths)
        p = model([self.state, self.state], ['FF', 'SL'], np.array([[.5] * 8, [-.5] * 8]))
        np.testing.assert_allclose(p[0], g0_expected(.5, 1), atol=1e-12)
        np.testing.assert_allclose(p[1], g0_expected(-.5, 1), atol=1e-12)
        self.assertEqual(len(model.identity['member_model_sha256']), 5)
        # The real wrapper stripped the 7 routing/cluster columns before the global network.
        self.assertEqual({w for m in model.models for w in m.model.global_model.context_widths}, {20 - 7})
        # Per-seed June weights are single-member diagnostics: changing them (consistently)
        # changes the pinned identity but never the ensemble output.
        other, other_paths = g0_files(self.root / 'w', seed_weights=(.1, .2, .3, .4, .5))
        q = load_g0(other, self.inputs, other_paths)([self.state], ['FF'], np.array([[.5] * 8]))
        np.testing.assert_allclose(q[0], p[0], atol=1e-15)
        self.assertNotEqual(load_g0(other, self.inputs, other_paths).identity['sha256'], model.identity['sha256'])
        # A dropped fifth member would be a different (wrong) predictor, and is refused.
        wrong = 0.75 * np.mean([softmax(np.r_[s, 0, 0, 2., np.zeros(6)] / t) for s, t in
                                zip(range(4), (1.0, 1.1, .9, 1.2))], axis=0)  # z[3]=4*.5
        self.assertGreater(np.abs(wrong - (p[0] - .25 * softmax(np.log(np.r_[.05, .55, [.05] * 8]) / 1.25))).max(), 1e-3)

    def test_manifest_and_file_rejections(self):
        bundle, paths = g0_files(self.root)

        def refused(message, mutate=lambda b, p: None, **kw):
            b, p = copy.deepcopy(bundle), dict(paths); mutate(b, p)
            with self.assertRaisesRegex(pa.IntegrityError, message):
                load_g0(b, self.inputs, p, **kw)([self.state], ['FF'], np.array([[.5] * 8]))

        refused('five G0 members', lambda b, p: b['members'].pop('4'))
        refused('seed order', lambda b, p: b.update(members={k: b['members'][k] for k in '10234'}))
        refused('source arm', lambda b, p: b['members']['3'].update(source_arm='g'))
        refused('May temperature', lambda b, p: b['members']['0'].update(delivery_temperature=float('nan')))
        refused('checkpoint pin', lambda b, p: b['members']['1'].update(model_sha256='f' * 64))
        refused('distinct', lambda b, p: (b['members']['1'].update(model_sha256=b['members']['0']['model_sha256']),
                                          b['files']['seed1_checkpoint'].update(sha256=b['members']['0']['model_sha256'])))
        refused('research-only', lambda b, p: b.update(service_promotion=True))
        swapped = self.root / 'swapped.pt'; swapped.write_bytes(b'swapped weights')
        refused('checkpoint changed', lambda b, p: p.update(seed2_checkpoint=swapped))
        refused('explicit path', lambda b, p: p.pop('seed4_checkpoint'))
        refused('May temperatures differ', lambda b, p: b['members']['0'].update(delivery_temperature=1.05)
                or b['files']['seed0_checkpoint'])
        refused('logits not finite', member=lambda seed: FakeMember(seed, 'nan' if seed == 3 else None))
        refused('logits not finite', member=lambda seed: FakeMember(seed, 'shape' if seed == 0 else None))
        refused('frequency baseline rows invalid', frequency=FakeFrequency(bad=True))
        refused('not a G0-global', member=lambda seed: SimpleNamespace(cell='G0-global', logits=None))
        refused('not a G0-global', member=lambda seed: SharingPredictor('G2-feature', FakeNetwork(seed), {}))
        refused('seed/classes differ', member=lambda seed: FakeMember(seed, network_seed={0: 4, 4: 0}.get(seed, seed)))

    def test_pinned_json_tamper_and_vocabulary(self):
        bundle, paths = g0_files(self.root)
        paths['p11_frozen_calibration'].write_text(json.dumps({'june_ensemble_model_weight': .9}))
        with self.assertRaisesRegex(pa.IntegrityError, 'pinned file changed'):
            load_g0(bundle, self.inputs, paths)
        bundle, paths = g0_files(self.root / 'v')
        _, other_inputs = g0_inputs(self.bc)
        other_inputs.types = ('FF', 'SL')
        with self.assertRaisesRegex(pa.IntegrityError, 'token vocabulary'):
            load_g0(bundle, other_inputs, paths)


# ---------------------------------------------------------------- runtime + ledger

def runtime_fixture(root, minimum=1):
    bc = fitted(minimum)
    art = pa.save_train_bc(bc, root / 'bc.json', PROV)
    # CH has no intervention support; pitcher 11 has none. Intersect with BC support so the
    # table stays inside TRAIN BC support for every minimum_action_count used below.
    rows = [('9', 'L', np.array([False, True, False, True]) & bc.support(PAState(0, 0, '9', 'L')))]
    _, support_sha = pa.save_support_table(art, rows, root / 'support.json')
    return art, support_sha


def request(rt, rid, pa_id, i, state, logged, hand='R'):
    return pr.DecisionRequest(rid, pa_id, i, state, logged, rt.sha256, hand)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.art, self.support_sha = runtime_fixture(self.root, minimum=2)

    def runtime(self):
        return pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, self.root / 'support.json',
                                self.support_sha, self.root / 'ledger.jsonl')

    def test_statuses_ratios_denominators_and_replay(self):
        rt = self.runtime()
        s0 = PAState(0, 0, '9', 'L')
        # minimum_action_count=2: CH (1 TRAIN count) has estimated logging mass 0.
        logging = self.art.bc.probabilities(s0)
        self.assertEqual(logging.tolist()[0], 0)
        ff = rt.submit(request(rt, 'r1', 'pa1', 0, s0, 'FF'))
        self.assertEqual(ff['status'], pr.SUPPORTED)
        # Full-vocabulary logging law vs masked reference: rho_ref = 1/pi_b(M|H), here != 1.
        mass = logging[1] + logging[3]
        self.assertAlmostEqual(ff['result']['rho_reference'], 1 / mass, places=12)
        rt.submit(request(rt, 'r2', 'pa1', 1, PAState(1, 0, '9', 'L', (past('FF'),)), 'SL'))
        self.assertEqual(rt.submit(request(rt, 'r3', 'pa2', 0, s0, 'CH'))['status'], pr.LOGGING_POSITIVITY)
        self.assertEqual(rt.submit(request(rt, 'r4', 'pa3', 0, PAState(0, 0, '404', 'L'), 'FF'))['status'], pr.UNKNOWN_PITCHER)
        self.assertEqual(rt.submit(request(rt, 'r5', 'pa4', 0, PAState(0, 0, '11', 'R'), 'KC'))['status'], pr.EMPTY_SUPPORT)
        # Mid-PA pitcher change to an unknown pitcher, then sticky refusal (no continuation filled).
        rt.submit(request(rt, 'r6', 'pa5', 0, s0, 'SL'))
        self.assertEqual(rt.submit(request(rt, 'r7', 'pa5', 1, PAState(1, 0, '404', 'L', (past('SL'),)), 'FF'))['status'],
                         pr.UNKNOWN_PITCHER)
        sticky = rt.submit(request(rt, 'r8', 'pa5', 2, PAState(1, 1, '9', 'L', (past('SL'), past('FF'))), 'FF'))
        self.assertEqual((sticky['status'], sticky['result']), (pr.MID_PA, None))
        # Duplicate retry: same content replays the stored row, no new ledger row.
        before = len(rt.ledger.rows)
        self.assertEqual(rt.submit(request(rt, 'r1', 'pa1', 0, s0, 'FF')), ff)
        self.assertEqual(len(rt.ledger.rows), before)
        expected = {'requests': 8, 'request_status': {pr.SUPPORTED: 3, pr.LOGGING_POSITIVITY: 1, pr.UNKNOWN_PITCHER: 2,
                    pr.EMPTY_SUPPORT: 1, pr.MID_PA: 1}, 'pas': 5,
                    'pa_status': {pr.SUPPORTED: 1, pr.LOGGING_POSITIVITY: 1, pr.UNKNOWN_PITCHER: 1,
                                  pr.EMPTY_SUPPORT: 1, pr.MID_PA: 1},
                    'conflicts': 0, 'run_status': 'OK', 'population_value': None}
        summary = rt.summary()
        self.assertEqual({k: summary[k] for k in expected}, expected)
        # Replay from disk reproduces the same denominators and dedup state.
        again = self.runtime()
        self.assertEqual(again.summary(), summary)
        self.assertEqual(again.submit(request(again, 'r1', 'pa1', 0, s0, 'FF'))['sha256'], ff['sha256'])
        self.assertEqual(again.submit(request(again, 'r9', 'pa5', 3, PAState(2, 1, '9', 'L',
                         (past('SL'), past('FF'), past('FF'))), 'FF'))['status'], pr.MID_PA)
        # Stored rows keep pre-decision context and pins, not physics/outcomes.
        self.assertNotIn('physics', json.dumps(ff))
        self.assertEqual(ff['history_actions'], [])
        # A changed retry of an existing ID is audited, not counted, and halts the run.
        with self.assertRaisesRegex(pa.IntegrityError, 'different content'):
            again.submit(request(again, 'r1', 'pa1', 0, s0, 'SL'))
        with self.assertRaisesRegex(pa.IntegrityError, 'halted'):  # live instance halts too
            again.submit(request(again, 'r10', 'pa6', 0, s0, 'FF'))
        final = self.runtime().summary()
        self.assertEqual((final['requests'], final['conflicts'], final['run_status']), (10, 1, 'HALTED'))

    def test_outside_support_is_valid_zero_ratio(self):
        root = self.root / 'm1'; root.mkdir()
        art, sha = runtime_fixture(root, minimum=1)
        rt = pr.build_runtime(root / 'bc.json', art.file_sha256, root / 'support.json', sha, root / 'l.jsonl')
        row = rt.submit(request(rt, 'a', 'p', 0, PAState(0, 0, '9', 'L', ()), 'CH'))  # PA start is 0-0 (COOP-018)
        self.assertEqual(row['status'], pr.OUTSIDE_POLICY_SUPPORT)
        self.assertEqual(row['result']['rho_reference'], 0.0)
        self.assertGreater(row['result']['logging'][0], 0)
        # Known pitcher whose every TRAIN action is below minimum_action_count: a refusal, not integrity.
        root4 = self.root / 'm4'; root4.mkdir()
        art4, sha4 = runtime_fixture(root4, minimum=4)
        rt4 = pr.build_runtime(root4 / 'bc.json', art4.file_sha256, root4 / 'support.json', sha4, root4 / 'l.jsonl')
        self.assertEqual(rt4.submit(request(rt4, 'k', 'p', 0, PAState(0, 0, '11', 'R'), 'KC'))['status'], pr.EMPTY_SUPPORT)

    def test_integrity_failures_are_recorded_then_halt(self):
        rt = self.runtime()
        s0 = PAState(0, 0, '9', 'L')
        with self.assertRaisesRegex(pa.IntegrityError, 'unknown logged action'):
            rt.submit(request(rt, 'x1', 'pa1', 0, s0, 'XX'))
        with self.assertRaisesRegex(pa.IntegrityError, 'halted'):
            rt.submit(request(rt, 'x2', 'pa2', 0, s0, 'FF'))
        summary = self.runtime().summary()
        self.assertEqual((summary['requests'], summary['request_status'], summary['run_status']),
                         (2, {pr.FAILED_INTEGRITY: 2}, 'HALTED'))

    def test_malformed_request_and_mutated_inputs(self):
        rt = self.runtime()
        with self.assertRaisesRegex(pa.IntegrityError, 'malformed'):
            rt.submit(pr.DecisionRequest([], 'pa', 0, PAState(0, 0, '9', 'L'), 'FF', rt.sha256))
        s = self.runtime().summary()
        self.assertEqual((s['requests'], s['malformed'], s['run_status']), (0, 1, 'HALTED'))
        # Caller mutation after load cannot change the law behind a runtime sha.
        root = self.root / 'mut'; root.mkdir()
        art, sha = runtime_fixture(root)
        support, _ = pa.load_support_table(root / 'support.json', sha, art)
        rt = pr.PolicyRuntime(art, support, pa.canonical_hash(pa.support_table_payload(
            art, [(p, b, m) for (p, b), m in support.items()])), root / 'l.jsonl')
        before = rt.reference.probabilities(PAState(0, 0, '9', 'L'))
        art.bc.cells[('9', 0, 0, 'L', '<START>')]['SL'] += 50
        np.testing.assert_array_equal(rt.reference.probabilities(PAState(0, 0, '9', 'L')), before)
        with self.assertRaisesRegex(pa.IntegrityError, 'BC state differs|differ from the cell sum'):
            pr.PolicyRuntime(art, support, sha, root / 'l2.jsonl')

    def test_runtime_exception_keeps_audit_row(self):
        rt = self.runtime()
        def boom(state): raise BudgetExceeded('budget')
        rt.candidate = boom
        with self.assertRaises(BudgetExceeded):
            rt.submit(request(rt, 'b1', 'pa1', 0, PAState(0, 0, '9', 'L'), 'FF'))
        self.assertEqual(self.runtime().summary()['request_status'], {pr.FAILED_RUNTIME: 1})

    def test_strict_history_and_pins(self):
        for name, req in [('skip', lambda rt: request(rt, 'h2', 'pa', 2, PAState(0, 0, '9', 'L', (past('FF'),) * 2), 'FF')),
                          ('mismatch', lambda rt: request(rt, 'h2', 'pa', 1, PAState(0, 1, '9', 'L', (past('SL'),)), 'FF')),
                          ('pin', lambda rt: pr.DecisionRequest('h2', 'pa', 1, PAState(0, 1, '9', 'L', (past('FF'),)), 'FF', 'e' * 64))]:
            root = self.root / name; root.mkdir()
            art, sha = runtime_fixture(root)
            rt = pr.build_runtime(root / 'bc.json', art.file_sha256, root / 'support.json', sha, root / 'l.jsonl')
            rt.submit(request(rt, 'h1', 'pa', 0, PAState(0, 0, '9', 'L'), 'FF'))
            with self.assertRaises(pa.IntegrityError):
                rt.submit(req(rt))
            self.assertEqual(rt.summary()['request_status'][pr.FAILED_INTEGRITY], 1)

    def test_ledger_corruption_is_refused(self):
        rt = self.runtime()
        rt.submit(request(rt, 'r1', 'pa1', 0, PAState(0, 0, '9', 'L'), 'FF'))
        head = rt.summary()['ledger_head_sha256']
        path = self.root / 'ledger.jsonl'; good = path.read_bytes()
        path.write_bytes(good.replace(b'"SUPPORTED"', b'"UNSUPPORTED_EMPTY_SUPPORT"'))
        with self.assertRaisesRegex(pa.IntegrityError, 'tampered'):
            self.runtime()
        path.write_bytes(good[:-5])
        with self.assertRaisesRegex(pa.IntegrityError, 'truncated'):
            self.runtime()
        lines = good.splitlines(keepends=True)
        path.write_bytes(lines[0])  # dropped a decision row: still a valid prefix, count shrinks visibly
        dropped = self.runtime().summary()  # the chain alone cannot see a row-boundary rollback;
        self.assertEqual(dropped['requests'], 0)  # a saved head pin can
        self.assertNotEqual(dropped['ledger_head_sha256'], head)
        path.write_bytes(good)
        # Another support table = another runtime identity; the old ledger cannot be reused.
        _, other = pa.save_support_table(self.art, [('9', 'L', np.array([False, True, False, False]))], self.root / 's2.json')
        with self.assertRaisesRegex(pa.IntegrityError, 'header pins'):
            pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, self.root / 's2.json', other, path)


class EndToEndTests(unittest.TestCase):
    def test_serialized_bc_support_g0_candidate_and_ledger(self):
        from test_policy_identity import SyntheticPolicy  # pinned synthetic G0 + WE files (COOP-017)
        root = Path(tempfile.mkdtemp())
        bc = CategoricalBC().fit([BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'FF', 'SL', 'CH')])
        art = pa.save_train_bc(bc, root / 'bc.json', {**PROV, 'train_rows': 4})
        loaded, data = pa.load_train_bc(root / 'bc.json', art.file_sha256), frame()
        data = pd.concat([data, data.iloc[[0]].assign(game_pk=2, pitcher=77)], ignore_index=True)  # an unknown pitcher row
        components = SyntheticPolicy(root / 'g0').bind(loaded, data)
        inputs, g0 = components.inputs, components.g0
        key = context_key(data.iloc[0])
        s0 = PAState(0, 0, '9', 'L', (), key)
        # Support table from the bound PolicyInputs support (BC ∩ tokens ∩ 12-count pools).
        mask = inputs.support(s0)
        self.assertEqual(mask.tolist(), [False, True, True])
        _, support_sha = pa.save_support_table(art, [('9', 'L', mask)], root / 'support.json')
        settings = dict(components=components, budget=RowBudget(10 ** 6, seed_count=5), tau=.01,
                        samples=4, pitch_cap=6, seed=701)
        rt = pr.build_runtime(root / 'bc.json', art.file_sha256, root / 'support.json', support_sha,
                              root / 'ledger.jsonl', **settings)
        # Masked reference agrees with the existing SupportedBC on the same inputs.
        np.testing.assert_allclose(rt.reference.probabilities(s0), SupportedBC(inputs).probabilities(s0), atol=1e-15)
        first = rt.submit(request(rt, 'e1', 'pa1', 0, s0, 'FF'))
        c, ref, log = (np.array(first['result'][k]) for k in ('candidate', 'reference', 'logging'))
        self.assertEqual(c[0], 0)                         # candidate support = mask, CH excluded
        self.assertGreater(np.abs(c - ref).max(), 1e-6)   # G0-driven Q moved probability
        self.assertAlmostEqual(first['result']['rho_candidate'], c[1] / log[1], places=12)
        self.assertNotEqual(first['result']['rho_reference'], 1.0)
        h = PastPitch('FF', (.5,) * 8, 'ball', 0, 0)
        second = rt.submit(request(rt, 'e2', 'pa1', 1, PAState(1, 0, '9', 'L', (h,), key), 'CH'))
        self.assertEqual((second['status'], second['result']['rho_candidate']), (pr.OUTSIDE_POLICY_SUPPORT, 0.0))
        rt.submit(request(rt, 'e3', 'pa2', 0, PAState(0, 0, '77', 'L', (), context_key(data.iloc[3])), 'FF'))
        self.assertGreater(g0.actual_neural_network_rows, 0)
        # Determinism + replay: a rebuilt runtime has the same identity and the same candidate row.
        rt2 = pr.build_runtime(root / 'bc.json', art.file_sha256, root / 'support.json', support_sha,
                               root / 'ledger.jsonl', **{**settings, 'budget': RowBudget(10 ** 6, seed_count=5)})
        self.assertEqual(rt2.sha256, rt.sha256)
        np.testing.assert_array_equal(rt2.candidate(s0)[0], c)
        # The candidate is exactly RolloutImprovement's P3 law; its Q is recorded on the mask only (COOP-018).
        np.testing.assert_array_equal(rt2.improvement.policy('P3', tau=.01)(s0, 0)[1], c)
        q, se = first['result']['q_reference'], first['result']['q_mc_se']
        self.assertEqual([v is None for v in q], [True, False, False])
        np.testing.assert_array_equal(np.array(q[1:]), rt2.improvement.q_values(s0)[0][1:])
        self.assertEqual(len(se), 3)
        self.assertEqual(rt2.summary()['request_status'], {pr.SUPPORTED: 1, pr.OUTSIDE_POLICY_SUPPORT: 1,
                                                           pr.UNKNOWN_PITCHER: 1})
        # Perturbed predictor (one May temperature) = new identity: old-pinned requests refused.
        perturbed = SyntheticPolicy(root / 'g0b', temps=(1.0, 1.1, .9, 1.2, .81)).bind(loaded, data)
        rt3 = pr.build_runtime(root / 'bc.json', art.file_sha256, root / 'support.json', support_sha,
                               root / 'ledger3.jsonl', **{**settings, 'components': perturbed})
        self.assertNotEqual(rt3.sha256, rt.sha256)
        with self.assertRaisesRegex(pa.IntegrityError, 'another runtime'):
            rt3.submit(pr.DecisionRequest('z', 'pz', 0, s0, 'FF', rt.sha256))
        with self.assertRaisesRegex(pa.IntegrityError, 'explicit'):
            pr.build_runtime(root / 'bc.json', art.file_sha256, root / 'support.json', support_sha,
                             root / 'ledger4.jsonl', **{**settings, 'tau': None})


if __name__ == '__main__':
    unittest.main()

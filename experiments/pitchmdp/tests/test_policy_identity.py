"""SYNTHETIC-ONLY tests for ML-POLICY-IDENTITY-v1 (COOP-017).

Fake picklable components written as pinned JSON/pickle files in temp dirs, a fake member
loader, and the real evaluation path (JointDelivery, MatrixHistoryStore, SharingContext,
predict_streamed, temperature_predictions) on a tiny synthetic frame. No real data, model
weights, cache or 2026 access. Expected values come from the evaluation code, not the runtime.
"""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts'))
from test_matrix_policy import frame  # existing synthetic safe-context rows (pitcher 9, R vs L)
from pitchmdp.archetypes import RELIABILITY_COLUMNS, STYLE_COLUMNS
from pitchmdp.data import hash_file
from pitchmdp.game import apply_terminal
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_policy import PolicyInputs, context_key, state_from_row
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor
from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, RowBudget
from pitchmdp.sequence_data import PHYSICAL_COLUMNS, PhysicalNormalizer
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_identity as pi
from pitchmdp import policy_runtime as pr
from run_sequence_frequency_baselines import temperature_predictions  # evaluation-path frequency transform

PROV = {'train_start': '2023-05-15', 'train_end': '2025-04-30', 'dates': 'declared_unverified',
        'train_rows': 4, 'source_ids': {'synthetic': 'a' * 64}, 'config_sha256': 'b' * 64,
        'code_commit': '713c7ea0000000000000000000000000000000aa', 'data_version': 'synthetic-v1'}
TYPES = ['CH', 'FF', 'SL']
CLUSTERS = {'columns': ['speed', 'spin'], 'pitcher_cluster': {'9': 0}, 'pitcher_profiles': {'9': [.1, -.2]},
            'pitcher_counts': {'9': 11}, 'cluster_counts': {'0': 1}}
NETWORK = {'kind': 'fake', 'n_context': 9, 'n_token': 8 + len(TYPES) + 1 + 11, 'length': 6, 'width': 1, 'n_classes': 10}


# ---------------------------------------------------------------- picklable fake components

class Encoder:
    """Stands in for SequenceContext: count channels first, as in the frozen layout."""
    def transform(self, rows):
        out = np.zeros((len(rows), 6), dtype=np.float32)
        out[:, 0], out[:, 1], out[:, 2] = rows.balls / 3, rows.strikes / 2, rows.inning / 9
        return out

    def report(self): return {'features': ['balls', 'strikes', 'inning'], 'synthetic': True}


class Delivery:
    draws, TIERS = 400, [('pitch_type', 'p_throws', 'stand')]

    def __init__(self, shift=0.):  # the report does not describe `shift`: only the bytes differ
        self.pools = {(0, ('FF', 'R', 'L')): np.full((400, 8), .5 + shift),
                      (0, ('SL', 'R', 'L')): np.full((400, 8), -.5 + shift)}
        self.fallback = np.ones((400, 8))
        self.report = {'draws': 400, 'seed': 1, 'pool_count': 2, 'tiers': [list(t) for t in self.TIERS]}


class Frequency:
    def __init__(self): self.report = {'synthetic_frequency': True}

    def predict(self, rows):
        raw = np.full((len(rows), 10), .05); raw[:, 1] = .55
        return raw


class Normalizer:
    def report(self): return {'synthetic_normalizer': True}


class Network:
    """Stateless global network (a call must not change the bound content digest)."""
    n_classes, device = 10, 'cpu'

    def __init__(self, seed, parameter_count=7):
        self.seed, self.report = seed, {'network': NETWORK, 'parameter_count': parameter_count}

    def logits(self, arrays):
        z = np.zeros((len(arrays[0]), 10)); z[:, 0] = self.seed; z[:, 3] = 4 * arrays[0][:, -1, 0]
        return z


class WE:
    def __init__(self, base=.5): self.base = base

    def predict_defense(self, state, defender_is_home):
        return self.base + .05 * state.outs - .02 * bin(state.bases).count('1')


class Advancement:
    def distribution(self, state, event): return [(1., apply_terminal(state, event))]


def load_member(config, record, clusters):
    predictor = SharingPredictor('G0-global', Network(record['seed']), clusters)
    predictor.delivery_temperature = record['delivery_temperature']
    return predictor, 'cpu'


def load_member_copy(config, record, clusters):  # same behaviour, another pinned loader
    predictor = SharingPredictor('G0-global', Network(record['seed']), clusters)
    predictor.delivery_temperature = record['delivery_temperature']
    return predictor, 'cpu'


def _json(value):
    return (json.dumps(value, indent=1) + '\n').encode()


class SyntheticPolicy:
    """Pinned synthetic G0 bundle (members, aux pickle, preparation, P11 records/config/calibration)
    and C0 WE bundle (contract, manifest, source lineage, game_values) for ``bind_components``."""
    def __init__(self, root, *, temps=(1.0, 1.1, .9, 1.2, .8), weight=.75, baseline_temperature=1.25,
                 delivery=None, we=None, clusters=CLUSTERS, aux=None, types=TYPES):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.paths, members = root, {}, {}
        self.bundle = {'protocol': 'g0_research_frozen_v1', 'research_only': True, 'service_promotion': False,
                       'members': members, 'files': {}}
        for seed in range(5):
            sha = self.put(f'seed{seed}_checkpoint', f'seed{seed}.pt', f'fake-checkpoint-{seed}'.encode())
            members[str(seed)] = {'source_arm': 'g' if seed < 3 else 'c1', 'model_sha256': sha,
                                  'delivery_temperature': temps[seed], 'june_model_weight': .7}
        self.aux = aux or {'baseline': Frequency(), 'context': Encoder(), 'delivery': delivery or Delivery(),
                           'normalizer': Normalizer()}
        self.put('p4_auxiliary', 'aux.pkl', pickle.dumps(self.aux))
        scope = {key: pi._report(value) for key, value in self.aux.items()}
        self.prep = {'features': {'auxiliary_scope': scope,
                                  'tokens': {'history_length': 5, 'type_vocabulary': list(types),
                                             'token_channels': 8 + len(types) + 1 + 11},
                                  'context': {'base': scope['context'],
                                              'pitcher_representation_sha256': canonical_hash(clusters)}},
                     'clusters': clusters, 'baseline_temperature': {'temperature': baseline_temperature}}
        self.put('p4_preparation', 'preparation.json', _json(self.prep))
        self.p11 = {'clusters': clusters, 'members': {str(s): {
            'seed': s, 'source_arm': members[str(s)]['source_arm'], 'model_sha256': members[str(s)]['model_sha256'],
            'delivery_temperature': temps[s], 'network': NETWORK, 'parameter_count': 7, 'device': 'cpu'} for s in range(5)}}
        self.put('p11_preparation', 'p11_preparation.json', _json(self.p11))
        self.put('p11_registered_config', 'p11_config.json', _json({'kind': 'fake', 'width': 1, 'context_width': 9,
                 'individual_tau': 1000., 'cluster_tau': 10000., 'device': 'cpu'}))
        self.put('p11_frozen_calibration', 'frozen_calibration.json', _json({
            'june_ensemble_model_weight': weight, 'refit_on_whole_mlb': False,
            'may_delivery_temperatures': {str(s): temps[s] for s in range(5)},
            'june_seed_model_weights': {str(s): .7 for s in range(5)}}))
        self.game_values = {'we': we or WE(), 'advancement': Advancement(), 'run_expectancy': {}}
        self.write_we()
        self.classes = {'aux': {key: pi.class_name(value) for key, value in self.aux.items()},
                        'we': {'we': pi.class_name(WE), 'advancement': pi.class_name(Advancement)}}

    def put(self, role, name, data):
        """Write a bundle file and (re)pin it in the bundle, as a consistent re-registration would."""
        self.paths[role] = self.root / name
        self.paths[role].write_bytes(data)
        self.bundle['files'][role] = {'path': 'registered/elsewhere', 'sha256': hashlib.sha256(data).hexdigest()}
        return self.bundle['files'][role]['sha256']

    def write_we(self, *, version='C0-v1', sources=None, game_values=None):
        where = '/synthetic/we-bundle'
        shas = {}
        for role, name, data in [
                ('we_game_values', 'game_values.pkl', game_values or pickle.dumps(self.game_values)),
                ('we_source_hashes', 'source_hashes.json',
                 _json(sources or {name: hash_file(pi.PROJECT / name) for name in pi.WE_SOURCE_FILES}))]:
            self.paths[role] = self.root / name
            self.paths[role].write_bytes(data)
            shas[role] = hashlib.sha256(data).hexdigest()
        manifest = _json({'sha256': {'game_values.pkl': shas['we_game_values']}})
        self.paths['we_bundle_manifest'] = self.root / 'bundle_manifest.json'
        self.paths['we_bundle_manifest'].write_bytes(manifest)
        contract = _json({'contract_version': version, 'value_spec_version': 'defense-we-pa-v1',
                          'training_cutoff': '2025-04-30', 'bundle_path': where,
                          'sha256': {f'{where}/bundle_manifest.json': hashlib.sha256(manifest).hexdigest(),
                                     f'{where}/source_hashes.json': shas['we_source_hashes']},
                          'bundle_files': {'game_values.pkl': shas['we_game_values']}})
        self.paths['we_contract'] = self.root / 'model-v1.json'
        self.paths['we_contract'].write_bytes(contract)
        self.we_sha = hashlib.sha256(contract).hexdigest()

    def bundle_file(self):
        data = _json(self.bundle)  # the bundle itself is read only through its file pin
        (self.root / 'bundle.json').write_bytes(data)
        return self.root / 'bundle.json', hashlib.sha256(data).hexdigest()

    def bind(self, bc_artifact, rows, *, loader=load_member, classes=None, bundle_sha=None):
        path, sha = self.bundle_file()
        return pi.bind_components(path, bundle_sha or sha, self.paths, bc_artifact=bc_artifact, context_rows=rows,
                                  member_loader=loader, classes=classes or self.classes, we_contract_sha256=self.we_sha)


def saved_bc(root):
    bc = CategoricalBC().fit([BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'FF', 'SL', 'CH')])
    return pa.save_train_bc(bc, root / 'bc.json', PROV)


# ---------------------------------------------------------------- binding and identity

class BindTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.art, self.rows = saved_bc(self.root), frame()
        self.syn = SyntheticPolicy(self.root / 'g0')

    def test_complete_identity_and_verification(self):
        bound = self.syn.bind(self.art, self.rows)
        identity = bound.identity
        self.assertEqual(set(identity), {'contract', 'g0_bundle_file_sha256', 'bc', 'predictor', 'members', 'frequency',
                                         'preprocessing', 'delivery_pool', 'we', 'sources', 'environment'})
        self.assertEqual(identity['g0_bundle_file_sha256'], hash_file(self.syn.root / 'bundle.json'))
        for rel in ('experiments/pitchmdp/pitchmdp/policy_identity.py', 'experiments/pitchmdp/pitchmdp/policy_runtime.py',
                    'experiments/pitchmdp/pitchmdp/policy_artifacts.py', 'experiments/pitchmdp/pitchmdp/matrix_policy.py',
                    'experiments/pitchmdp/pitchmdp/matrix_sharing.py', 'experiments/pitchmdp/pitchmdp/rollout_policy.py',
                    'experiments/pitchmdp/pitchmdp/game.py', 'experiments/pitchmdp/tests/test_policy_identity.py'):
            self.assertEqual(identity['sources'][rel], hash_file(pi.REPO / rel))  # fakes are pinned code too
        # Reachable only through module-level imports (matrix_policy -> archetypes; this test module ->
        # test_matrix_policy): the import closure, not just the classes of the connected objects.
        for rel in ('experiments/pitchmdp/pitchmdp/archetypes.py', 'experiments/pitchmdp/tests/test_matrix_policy.py'):
            self.assertIn(rel, identity['sources'])
        self.assertFalse([rel for rel in identity['sources'] if '.venv' in rel or 'site-packages' in rel])
        self.assertEqual(identity['members']['loader'], pi.callable_identity(load_member))
        self.assertEqual((identity['frequency']['key'], identity['frequency']['temperature']), ('baseline', 1.25))
        self.assertEqual(identity['we']['game_values_sha256'], hashlib.sha256(self.syn.paths['we_game_values'].read_bytes()).hexdigest())
        self.assertEqual(identity['delivery_pool']['source_hash'], self.syn.bundle['files']['p4_auxiliary']['sha256'])
        self.assertTrue(bound.verify())
        with self.assertRaises(ValueError):  # pinned pools are frozen in place
            bound.inputs.delivery.pools[(0, ('FF', 'R', 'L'))][0, 0] = 9.
        self.assertEqual(self.syn.bind(self.art, self.rows).sha256, bound.sha256)  # deterministic

    def test_every_component_is_in_the_identity(self):
        base = self.syn.bind(self.art, self.rows).sha256
        variants = {
            'delivery pool bytes (same report)': SyntheticPolicy(self.root / 'd', delivery=Delivery(shift=.1)),
            'WE values': SyntheticPolicy(self.root / 'w', we=WE(base=.45)),
            'frequency temperature': SyntheticPolicy(self.root / 'f', baseline_temperature=1.3),
            'member May temperature': SyntheticPolicy(self.root / 't', temps=(1.0, 1.1, .9, 1.2, .81)),
            'June ensemble weight': SyntheticPolicy(self.root / 'e', weight=.7),
            'context encoder clusters': SyntheticPolicy(self.root / 'c', clusters={**CLUSTERS, 'pitcher_profiles': {'9': [.2, -.2]}}),
        }
        for name, synthetic in variants.items():
            with self.subTest(name):
                self.assertNotEqual(synthetic.bind(self.art, self.rows).sha256, base)
        with self.subTest('member loader function'):
            self.assertNotEqual(self.syn.bind(self.art, self.rows, loader=load_member_copy).sha256, base)

    def test_inconsistent_or_swapped_components_are_refused(self):
        art, rows = self.art, self.rows

        def refused(message, build):
            synthetic = SyntheticPolicy(self.root / f'r{len(list(self.root.iterdir()))}')
            kwargs = build(synthetic) or {}
            with self.assertRaisesRegex(pa.IntegrityError, message):
                synthetic.bind(art, rows, **kwargs)

        def swap_aux(s):  # other bytes behind an unchanged pin
            s.paths['p4_auxiliary'].write_bytes(pickle.dumps({**s.aux, 'delivery': Delivery(shift=.2)}))
        refused('pinned file changed', swap_aux)

        def stale_report(s):  # re-pinned pickle whose object no longer matches the archived report
            other = Delivery(); other.report = {**other.report, 'seed': 2}
            s.put('p4_auxiliary', 'aux.pkl', pickle.dumps({**s.aux, 'delivery': other}))
        refused('differs from the report pinned', stale_report)

        def extra_pool(s):  # report archived before a pool was added: descriptor/object disagree
            other = Delivery(); other.pools[(0, ('CH', 'R', 'L'))] = np.zeros((400, 8))
            s.aux['delivery'] = other
            s.put('p4_auxiliary', 'aux.pkl', pickle.dumps(s.aux))
        refused('delivery pools differ from their pinned descriptor', extra_pool)

        refused('delivery class differs', lambda s: {'classes': {**s.classes, 'aux': {
            **s.classes['aux'], 'delivery': 'pitchmdp.sequence_delivery.JointDelivery'}}})

        def clusters_only(s):  # representation sha no longer describes the clusters
            s.prep['clusters'] = {**CLUSTERS, 'pitcher_counts': {'9': 12}}
            s.put('p4_preparation', 'preparation.json', _json(s.prep))
        refused('context encoder differs', clusters_only)

        def record_temperature(s):
            s.p11['members']['2']['delivery_temperature'] = .95
            s.put('p11_preparation', 'p11_preparation.json', _json(s.p11))
        refused('member record differs from the bundle: 2', record_temperature)

        def wrong_network(config, record, clusters):
            predictor = SharingPredictor('G0-global', Network(record['seed'], parameter_count=8), clusters)
            predictor.delivery_temperature = record['delivery_temperature']
            return predictor, 'cpu'
        refused('loaded network differs', lambda s: {'loader': wrong_network})

        refused('compatible frozen C0', lambda s: s.write_we(version='C0-v2'))

        def swap_values(s):
            s.paths['we_game_values'].write_bytes(pickle.dumps({**s.game_values, 'we': WE(base=.4)}))
        refused('pinned file changed', swap_values)

        refused('WE source lineage differs', lambda s: s.write_we(sources={
            **{name: hash_file(pi.PROJECT / name) for name in pi.WE_SOURCE_FILES}, 'pitchmdp/game.py': 'f' * 64}))

        refused('WE we class differs', lambda s: {'classes': {**s.classes, 'we': {
            **s.classes['we'], 'we': 'pitchmdp.game.WinExpectancy'}}})

        def malformed_records(s):
            s.put('p11_preparation', 'p11_preparation.json', _json({'clusters': CLUSTERS}))
        refused('binding failed: KeyError', malformed_records)

        refused('pinned file changed: .*bundle.json', lambda s: {'bundle_sha': 'e' * 64})  # bundle is file-pinned

        def outside_code(s):  # a registered name that resolves outside the pinned source closure
            s.aux['baseline'] = SimpleNamespace(report={'outside': True})
            s.put('p4_auxiliary', 'aux.pkl', pickle.dumps(s.aux))
            s.prep['features']['auxiliary_scope']['baseline'] = {'outside': True}
            s.put('p4_preparation', 'preparation.json', _json(s.prep))
            return {'classes': {**s.classes, 'aux': {**s.classes['aux'], 'baseline': 'types.SimpleNamespace'}}}
        refused('baseline class is not repository code', outside_code)

        def swap_during_load(config, record, clusters):  # loader reopens a file that changed after the hash
            Path(record['model_path']).write_bytes(b'swapped after the factory hash')
            return load_member(config, record, clusters)
        refused('checkpoint changed during load: 0', lambda s: {'loader': swap_during_load})

        def loader_value_error(config, record, clusters):  # e.g. run_ml_g0_whole device check
            raise ValueError('Device cpu differs from the frozen member device mps')
        refused('binding failed: ValueError: Device cpu', lambda s: {'loader': loader_value_error})

        original = pi.load_g0_ensemble

        def reread_other_bytes(*args, **kwargs):  # factory re-reads JSON after hashing (TOCTOU)
            g0 = original(*args, **kwargs); g0.neural_weight = .5
            return g0
        with mock.patch.object(pi, 'load_g0_ensemble', reread_other_bytes):
            refused('factory calibration differs from the verified', lambda s: None)

        def unpinned_config(s):
            del s.bundle['files']['p11_registered_config']
        refused('G0 pin missing: p11_registered_config', unpinned_config)

        def mutated_bc(s):  # last: mutates the shared artifact
            art.bc.cells[('9', 0, 0, 'L', '<START>')]['SL'] += 1
        refused('BC state differs|differ from the cell sum', mutated_bc)


# ---------------------------------------------------------------- runtime wiring

class RuntimeWiringTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.art = saved_bc(self.root)
        self.bc = pa.load_train_bc(self.root / 'bc.json', self.art.file_sha256)
        self.rows = frame()
        self.syn = SyntheticPolicy(self.root / 'g0')
        self.components = self.syn.bind(self.bc, self.rows)
        self.key = context_key(self.rows.iloc[0])
        self.s0 = PAState(0, 0, '9', 'L', (), self.key)
        _, self.support_sha = pa.save_support_table(self.art, [('9', 'L', self.components.inputs.support(self.s0))],
                                                    self.root / 'support.json')

    def runtime(self, name='l.jsonl', support=None, **changes):
        path, sha = support or (self.root / 'support.json', self.support_sha)
        settings = dict(components=self.components, budget=RowBudget(10 ** 6, seed_count=5), tau=.01, samples=4,
                        pitch_cap=6, seed=701)
        return pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, path, sha, self.root / name,
                                **{**settings, **changes})

    def request(self, rt, rid, state, logged):
        return pr.DecisionRequest(rid, 'pa-' + rid, 0, state, logged, rt.sha256)

    def test_candidate_identity_is_complete_and_pinned(self):
        rt = self.runtime()
        candidate = rt.pins['candidate']
        self.assertTrue({'predictor', 'members', 'frequency', 'preprocessing', 'delivery_pool', 'we', 'sources',
                         'environment', 'search', 'support_table_sha256'} <= set(candidate))
        self.assertEqual({k: candidate['search'][k] for k in ('tau', 'samples', 'pitch_cap', 'seed')},
                         {'tau': .01, 'samples': 4, 'pitch_cap': 6, 'seed': 701})
        self.assertEqual(self.runtime('l2.jsonl', expected_identity_sha256=candidate['sha256']).sha256, rt.sha256)
        with self.assertRaisesRegex(pa.IntegrityError, 'registered pin'):
            self.runtime('l3.jsonl', expected_identity_sha256='0' * 64)
        self.assertNotEqual(self.runtime('l4.jsonl', tau=.02).pins['candidate']['sha256'], candidate['sha256'])
        with self.assertRaises(TypeError):  # loose callables are no longer an entry path
            self.runtime('l5.jsonl', components=None, pool=self.components.inputs.pool)
        with self.assertRaisesRegex(pa.IntegrityError, 'BoundComponents'):
            self.runtime('l6.jsonl', components=SimpleNamespace(**vars(self.components)))
        with self.assertRaisesRegex(pa.IntegrityError, 'identity pin needs'):
            pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, self.root / 'support.json',
                             self.support_sha, self.root / 'l7.jsonl', expected_identity_sha256='a' * 64)

    def test_rewiring_or_mutation_after_binding_is_detected(self):
        rt = self.runtime()
        self.assertEqual(rt.submit(self.request(rt, 'a', self.s0, 'FF'))['status'], pr.SUPPORTED)
        self.assertTrue(rt.verify_components())  # inference itself leaves the bound content unchanged
        other = SyntheticPolicy(self.root / 'other', we=WE(base=.45)).bind(self.bc, self.rows)
        simulator, bound = rt.improvement.simulator, self.components
        rewires = [(simulator, 'pool', other.inputs.pool, 'not wired'), (simulator, 'predictor', other.g0, 'not wired'),
                   (simulator, 'terminal', other.we.terminal, 'not wired'), (rt.improvement, 'cutoff', other.we.cutoff, 'not wired'),
                   # objects inference actually reads (review finding): frequency, pools, inputs, encoder
                   (bound.g0.baseline, 'baseline', Frequency(), 're-linked'), (bound.inputs, 'delivery', Delivery(shift=.4), 're-linked'),
                   (bound.g0, 'inputs', other.inputs, 're-linked'), (bound.inputs, 'context_encoder', other.inputs.context_encoder, 're-linked')]
        for owner, attribute, value, message in rewires:
            with self.subTest(attribute):
                saved = getattr(owner, attribute)
                setattr(owner, attribute, value)
                with self.assertRaisesRegex(pa.IntegrityError, message):
                    rt.verify_components()
                setattr(owner, attribute, saved)
        self.assertTrue(rt.verify_components())
        mutations = [lambda: bound.inputs.contexts[self.key].__setitem__(0, 9.),
                     lambda: bound.inputs.rows[self.key].__setitem__('inning', 9),
                     lambda: type(bound.inputs.delivery).TIERS.append(('pitch_type',)),  # class-level, not pickled
                     lambda: bound.we.terminal_tables[self.key].__setitem__('walk', .1)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                fresh_rt = self.runtime(f'v{index}.jsonl', components=self.syn.bind(self.bc, self.rows))
                bound = fresh_rt.components
                try:
                    mutate()
                    with self.assertRaisesRegex(pa.IntegrityError, 'content changed'):
                        fresh_rt.verify_components()
                finally:
                    Delivery.TIERS = [('pitch_type', 'p_throws', 'stand')]
        fresh = self.syn.bind(self.bc, self.rows)
        real_hash = pi.hash_file
        with mock.patch.object(pi, 'hash_file', lambda path: '0' * 64 if path.name == 'policy_identity.py' else real_hash(path)):
            with self.assertRaisesRegex(pa.IntegrityError, 'source changed'):  # as if the file were edited
                fresh.verify()
        fresh.identity['search'] = {'tau': 9}
        with self.assertRaisesRegex(pa.IntegrityError, 'identity changed'):
            fresh.verify()
        # A directly built candidate runtime cannot claim a verified identity.
        table, content = pa.load_support_table(self.root / 'support.json', self.support_sha, self.bc)
        direct = pr.PolicyRuntime(self.bc, table, content, self.root / 'direct.jsonl', candidate=lambda s: None,
                                  candidate_identity={'name': 'loose'})
        with self.assertRaisesRegex(pa.IntegrityError, 'without bound components'):
            direct.verify_components()

    def test_context_row_values_enter_the_request_fingerprint(self):
        rt = self.runtime('c.jsonl')
        first = rt.submit(self.request(rt, 'a', self.s0, 'FF'))
        self.assertEqual(first['context_sha256'], pi.row_sha256(self.components.inputs.rows[self.key]))
        changed = self.rows.copy(); changed.loc[0, 'inning'] = 7
        rebound = self.syn.bind(self.bc, changed)  # same pins: same component identity...
        self.assertEqual(rebound.sha256, self.components.sha256)
        again = self.runtime('c.jsonl', components=rebound)
        self.assertEqual(again.sha256, rt.sha256)  # ...so the ledger header still matches,
        with self.assertRaisesRegex(pa.IntegrityError, 'different content'):  # but the replay is refused
            again.submit(self.request(again, 'a', self.s0, 'FF'))

    def test_support_table_must_match_the_bound_pools(self):
        # Built from another pool (SL dropped): FAILED_INTEGRITY, recorded, halts.
        _, other = pa.save_support_table(self.art, [('9', 'L', np.array([False, True, False]))], self.root / 's2.json')
        rt = self.runtime('m1.jsonl', support=(self.root / 's2.json', other))
        with self.assertRaisesRegex(pa.IntegrityError, 'differs from the bound delivery-pool support'):
            rt.submit(self.request(rt, 'x', self.s0, 'FF'))
        self.assertEqual((rt.summary()['request_status'], rt.summary()['run_status']), ({pr.FAILED_INTEGRITY: 1}, 'HALTED'))
        # An empty table must not hide the mismatch behind an EMPTY_SUPPORT refusal.
        _, empty = pa.save_support_table(self.art, [], self.root / 's3.json')
        rt = self.runtime('m2.jsonl', support=(self.root / 's3.json', empty))
        with self.assertRaisesRegex(pa.IntegrityError, 'differs from the bound delivery-pool support'):
            rt.submit(self.request(rt, 'y', self.s0, 'FF'))
        # A context that was never bound is an integrity failure, not a runtime error.
        rt = self.runtime('m3.jsonl')
        with self.assertRaisesRegex(pa.IntegrityError, 'not bound'):
            rt.submit(self.request(rt, 'z', PAState(0, 0, '9', 'L', (), '9:9:9'), 'FF'))
        self.assertEqual(rt.summary()['request_status'], {pr.FAILED_INTEGRITY: 1})
        # A state whose batter side disagrees with its bound row is not an EMPTY_SUPPORT refusal
        # and must not poison the shared support cache (review finding).
        rt = self.runtime('m4.jsonl')
        with self.assertRaisesRegex(pa.IntegrityError, 'differ from the bound context row'):
            rt.submit(self.request(rt, 'w', PAState(0, 0, '9', 'R', (), self.key), 'FF'))
        self.assertEqual(rt.summary()['request_status'], {pr.FAILED_INTEGRITY: 1})
        self.assertNotIn(('9', 'R', 'R'), self.components.inputs.support_cache)


# ---------------------------------------------------------------- policy path == evaluation path

class ProbeEncoder:
    def transform(self, rows):
        return np.column_stack([rows.balls / 3, rows.strikes / 2, rows.outs_when_up / 2, rows.inning / 9,
                                rows.bases / 7, rows.stand.eq('L')]).astype(np.float32)


class ProbeNetwork:
    """Dense fake network: any difference in tokens/validity/context changes the logits."""
    n_classes, device = 10, 'cpu'
    report = {'network': NETWORK, 'parameter_count': 7}  # matches the synthetic member records

    def __init__(self, seed, n_token=8 + 3 + 11, n_context=6 + 2 + 1):
        rng = np.random.default_rng(100 + seed)
        self.seed = seed
        self.a, self.b = rng.normal(size=(6 * n_token, 10)) * .2, rng.normal(size=(n_context, 10)) * .5
        self.c = rng.normal(size=(6, 10)) * .3

    def logits(self, arrays):
        tokens, valid, context = arrays
        return (tokens.reshape(len(tokens), -1).astype(float) @ self.a + context.astype(float) @ self.b
                + valid.astype(float) @ self.c)


class ProbeFrequency:
    def __init__(self): self.report = {'synthetic_probe_frequency': True}

    def predict(self, rows):
        z = np.zeros((len(rows), 10))
        z[:, 0], z[:, 1] = rows.balls.to_numpy(float), rows.strikes.to_numpy(float) + rows.outs_when_up.to_numpy(float)
        z[:, 2], z[:, 4] = rows.pitch_type.eq('FF').to_numpy(float), rows.bases.to_numpy(float) / 7
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


class ProbeEncoderWithReport(ProbeEncoder):
    def report(self): return {'synthetic_probe_encoder': True}


def probe_loader(config, record, clusters):
    member = SharingPredictor('G0-global', ProbeNetwork(record['seed']), clusters)
    member.delivery_temperature = record['delivery_temperature']
    return member, 'cpu'


def probe_frame():
    """TRAIN: 80 three-pitch PAs of pitcher 9 (R) vs L batters. DEV: 4 more vs L and one vs R, whose
    (type, R, R) has no TRAIN pool at any tier (the evaluation used the league fallback)."""
    rng, rows = np.random.default_rng(7), []
    for pa_index in range(85):
        train = pa_index < 80
        date = (pd.Timestamp('2024-06-01') + pd.Timedelta(days=pa_index // 10) if train
                else pd.Timestamp('2025-07-01') + pd.Timedelta(days=pa_index - 80)).strftime('%Y-%m-%d')
        for pitch in range(3):
            rows.append(dict(game_pk=5000 + pa_index, at_bat_number=1, pitch_number=pitch + 1, game_date=date,
                             pitcher=9, batter=2 + pa_index % 3, stand='R' if pa_index == 84 else 'L', p_throws='R',
                             balls=pitch, strikes=0, inning=1 + pa_index % 9, inning_topbot='Top' if pa_index % 2 else 'Bot',
                             outs_when_up=pa_index % 3, bases=pa_index % 8, home_score=0, away_score=pa_index % 4,
                             pitch_type=str(rng.choice(['FF', 'SL'])), description='ball', events=None,
                             supported_pa=True, split='train' if train else 'dev',
                             **{c: float(rng.normal()) for c in PHYSICAL_COLUMNS},
                             **{c: .5 for c in STYLE_COLUMNS}, **{c: .3 for c in RELIABILITY_COLUMNS}))
    return pd.DataFrame(rows)


class ConnectionProbeTests(unittest.TestCase):
    def test_bound_policy_path_reproduces_evaluation_path_and_detects_swaps(self):
        root = Path(tempfile.mkdtemp())
        data = probe_frame()
        train = data.loc[data.split.eq('train')]
        normalizer = PhysicalNormalizer().fit(train)
        store = MatrixHistoryStore.from_frame(data, normalizer, 5, type_vocabulary=['FF', 'SL'])
        delivery = JointDelivery().fit(train, normalizer, draws=400, seed=42)
        clusters = {**CLUSTERS, 'columns': ['a', 'b']}
        temps, weight, frequency_t = (1.1, .9, 1.0, 1.2, .95), .7, 1.2
        aux = {'baseline': ProbeFrequency(), 'context': ProbeEncoderWithReport(), 'delivery': delivery,
               'normalizer': normalizer}
        rows = np.flatnonzero(data.split.eq('dev').to_numpy())
        # Evaluation path: exactly how G0 DEV predictions were produced and blended.
        encoder = SharingContext(aux['context'], clusters)
        calibrated, levels = [], None
        for seed, t in enumerate(temps):
            member, _ = probe_loader({}, {'seed': seed, 'delivery_temperature': t}, clusters)
            p, _, levels = predict_streamed(member, delivery, store, encoder, rows)
            calibrated.append(p)
        primary = (weight * np.mean(calibrated, axis=0)
                   + (1 - weight) * temperature_predictions(aux['baseline'].predict(store.frame.iloc[rows]), frequency_t))
        comparable = levels >= 0
        self.assertEqual((int(comparable.sum()), set(levels[comparable]), set(levels[~comparable])), (12, {3}, {-1}))
        art = pa.save_train_bc(CategoricalBC().fit([BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'SL')]),
                               root / 'bc.json', {**PROV, 'train_rows': 2})
        states = [state_from_row(store, int(r)) for r in rows]
        actions = [str(a) for a in store.frame.pitch_type.iloc[rows]]

        def bound(name, **change):  # the policy path goes through bind_components (pinned files)
            options = dict(temps=temps, weight=weight, baseline_temperature=frequency_t, clusters=clusters,
                           aux=aux, types=['FF', 'SL'])
            return SyntheticPolicy(root / name, **{**options, **change}).bind(
                art, store.frame.iloc[rows], loader=probe_loader).g0

        g0 = bound('base')
        keep = [i for i in range(len(rows)) if comparable[i]]
        match = pi.compare_probe(pi.integrated_predictions(g0, [states[i] for i in keep], [actions[i] for i in keep]),
                                 primary[comparable], 1e-10)
        self.assertTrue(match['pass'], match)
        for i in np.flatnonzero(~comparable):  # evaluation used the fallback pool: the policy path refuses
            with self.assertRaisesRegex(ValueError, 'No frozen action-specific TRAIN delivery support'):
                pi.integrated_predictions(g0, [states[i]], [actions[i]])
        swaps = {'delivery pools (other seed)': dict(aux={**aux, 'delivery': JointDelivery().fit(train, normalizer, draws=400, seed=43)}),
                 'context encoder clusters': dict(clusters={**clusters, 'pitcher_profiles': {'9': [.3, -.2]}}),
                 'member temperatures reordered': dict(temps=temps[::-1]),
                 'frequency temperature': dict(baseline_temperature=1.3)}
        for name, change in swaps.items():
            with self.subTest(name):
                other = bound(name.split()[0] + str(len(name)), **change)
                report = pi.compare_probe(pi.integrated_predictions(other, [states[i] for i in keep], [actions[i] for i in keep]),
                                          primary[comparable], 1e-6)
                self.assertFalse(report['pass'], report)


if __name__ == '__main__':
    unittest.main()

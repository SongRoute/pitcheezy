"""SYNTHETIC-ONLY: the low-memory ledger/estimator (D123) changes no output byte and keeps memory
compact. The golden digests below were produced by the in-memory code at commit 25bcdb7 (the
sealed <=2025 S6-a4 code path) running ``stage_digests`` on the same synthetic two-game stage.

The candidate identity pins the sha256 of its source closure, which includes policy_runtime.py, so
the edited file necessarily changes the identity, the runtime sha and hence the ledger header. For
the byte comparison only, both the old and the new run see that one file hash as a fixed label."""
import hashlib
import json
import sys
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts'))
from test_policy_validation import MIN, NO_PITCH, RunnerFixture, blocks_of
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_estimator as est
from pitchmdp import policy_identity as pi
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as pr
from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, PastPitch, RowBudget
from test_policy_identity import PROV
import run_policy_validation as rpv

GOLDEN = {  # from commit 25bcdb7 (in-memory Ledger.rows and by_pa grouping): `python tests/test_policy_lowmem.py`,
    # regenerated 2026-10-01 on 25bcdb7 code with the current test files (D126: test harness files are labelled)
    'result': '6af7db5830dcfb3e8c49af2f85553f6376934ec86d24de06887b0ba0ee27eb59',
    'rows': '63dd9ea101c5eb940e9a6350a9082b8f471084b1ec0c81a5b1a16b20ddac1f29',
    'table': '49f60df746df37572f4e254f8551d0af3acad488850905d2318828b1f0329ce0',
    'ledger': 'c6c500034c9025d4e937fa5731df57cbfc4735b89d1ca7d872b23e793b4f584c',
    'pair': 'a7281abed5a8154f4a974e8fb575ab5b4bd5f7da6414e59bd6a39125a3ec8bbc'}


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
                          json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# Files edited by D123. The registered runner's closure (S6-a4 ledger header) holds policy_runtime.py
# only; this synthetic fixture's closure also reaches the estimator and the runner script.
CHANGED = ('policy_runtime.py', 'policy_estimator.py', 'run_policy_validation.py',
           'test_policy_lowmem.py', 'test_policy_validation.py')  # test harness files (they join the fixture closure)


def stage_digests(fixture, root):
    """The ope-2026 shape of ``run_dr`` (runtime v3, L1-R, S-v1/S-NP/S-B/S-C, same-ledger variants,
    strata, the M-10 paired run) on the two synthetic dev games; every output hashed."""
    real = pi.hash_file
    label = lambda path: 'code-under-test' if Path(path).name in CHANGED else real(path)
    with mock.patch.object(pi, 'hash_file', label):
        return _stage_digests(fixture, root)


def _stage_digests(fixture, root):
    out, report = fixture.materialize()
    pins = fixture.pins(out, report)
    dev = blocks_of(fixture.frame, 'dev')
    snapshot = preq.style_snapshot(fixture.frame, '2025-07-01')
    contexts, _ = rpv.pa_contexts(fixture.store, dev, snapshot, '2025-07-01')
    components = fixture.syn.bind(pa.load_train_bc(*pins['bc']), contexts)
    runtime = pr.build_runtime(*pins['bc'], *pins['support'], root / 'ledger.jsonl', components=components,
                               budget=RowBudget(10 ** 8, seed_count=5), tau=.01, samples=2, pitch_cap=4, seed=701,
                               evaluation_seed=1701, hand_registry=pins['hands'], positivity_record=True)
    pair = lambda: pr.build_reference_pair_runtime(*pins['bc'], *pins['support'], root / 'pair.jsonl',
                                                    hand_registry=pins['hands'], positivity_record=True)
    result, rows = rpv.run_dr(runtime, fixture.store, components, dev, rpv.Deadline(), no_pitch=NO_PITCH,
                              bootstrap={'draws': 200, 'seed': 5, 'invalid_share_max': .5, 'minimum': MIN},
                              ess_gate={'pa': 100., 'game': 30.},
                              sensitivities=['r5-events-v1', 'flags_to_bounds', 'post_pitch_scores'],
                              strata=rpv.strata_function(fixture.frame, [1., 2.]), expected_pas=16, pair=pair,
                              estimator={'c_deltas': [0.01, 0.05]})
    result.pop('seconds')  # wall clock
    table = pd.DataFrame([{**{k: v for k, v in r.items() if k != 'strata'},
                           **{f'stratum_{k}': v for k, v in r['strata'].items()}} for r in rows])
    return {'result': _sha(result), 'rows': _sha(rows), 'table': _sha(table.to_json().encode()),
            'ledger': _sha((root / 'ledger.jsonl').read_bytes()), 'pair': _sha((root / 'pair.jsonl').read_bytes())}


class StageBytesTests(RunnerFixture):
    def test_run_dr_outputs_equal_the_in_memory_code(self):
        root = self.root / 'stage'
        root.mkdir()
        self.assertEqual(stage_digests(self, root), GOLDEN)


# ---------------------------------------------------------------- memory over many PAs

PINS = {}  # root -> pinned artifact arguments


def pair_runtime(root):
    """Cheap candidate runtime (cand = independent reference, Q recorded) over a tiny TRAIN BC;
    a second call reopens the same ledger."""
    if root not in PINS:
        records = [BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'FF', 'SL', 'CH')]
        art = pa.save_train_bc(CategoricalBC().fit(records), root / 'bc.json', {**PROV, 'train_rows': 4})
        _, support = pa.save_support_table(art, [('9', 'L', np.array([True, True, True]))], root / 's.json')
        _, hands = pa.save_hand_registry(art, {'9': 'R'}, 4, root / 'h.json')
        PINS[root] = (root / 'bc.json', art.file_sha256, root / 's.json', support, root / 'h.json', hands)
    bc, bc_sha, s, support, h, hands = PINS[root]
    return pr.build_reference_pair_runtime(bc, bc_sha, s, support, root / 'ledger.jsonl', hand_registry=(h, hands),
                                           positivity_record=True)


def submit_pas(rt, first, n, facts):
    """PAs of three decisions (FF ball, SL strike, FF in play), ten PAs per game."""
    past = lambda a, o, b, s: PastPitch(a, (0.,) * 8, o, b, s)
    for i in range(first, first + n):
        pa_id, h = f'{i // 10}:{i}', (past('FF', 'ball', 0, 0), past('SL', 'strike', 1, 0))
        for k, state in enumerate((PAState(0, 0, '9', 'L'), PAState(1, 0, '9', 'L', h[:1]), PAState(1, 1, '9', 'L', h))):
            rt.submit(pr.DecisionRequest(f'{pa_id}:{k}', pa_id, k, state, ('FF', 'SL', 'FF')[k], rt.sha256, 'R'))
        facts[pa_id] = {'game': i // 10, 'in_population': True, 'problem': None, 'problem_index': None,
                        'reward': (i % 7) / 7, 'kind': None, 'reason': None}


def growth(root, pas):
    """Traced bytes kept per request by the runtime after ``pas`` more PAs (facts excluded)."""
    rt, facts = pair_runtime(root), {}
    submit_pas(rt, 0, 50, facts)
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    submit_pas(rt, 50, pas, {})
    kept = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    return rt, kept / (3 * pas)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.root = Path(tempfile.mkdtemp())

    def test_runtime_keeps_a_compact_index_and_streams_rows(self):
        rt, per_request = growth(self.root, 1500)
        # 25bcdb7 kept every hash-chained row in memory (measured 2,520 B/request on this fixture vs 195 now);
        # the index keeps one byte offset per request ID and one small PA record.
        self.assertLess(per_request, 400)
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        self.assertEqual(sum(1 for _ in rt.ledger.iter_decisions()), 3 * 1550)
        self.assertLess(tracemalloc.get_traced_memory()[1] - before, 256 * 1024)  # flat: one row at a time
        tracemalloc.stop()

    def test_streaming_estimate_equals_the_list_estimate_and_holds_one_pa(self):
        rt, facts = pair_runtime(self.root), {}
        submit_pas(rt, 0, 600, facts)
        kwargs = dict(draws=50, seed=3, invalid_share_max=.5, minimum={'games': 1, 'pa_starts': 1}, censoring='l1r')
        listed = est.estimate(rt.ledger.decisions(), facts, **kwargs)
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        streamed = est.estimate(rt.ledger.iter_decisions(), facts, **kwargs)
        peak = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        self.assertEqual(json.dumps(streamed, sort_keys=True), json.dumps(listed, sort_keys=True))
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        rows = rt.ledger.decisions()
        materialized = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        del rows
        self.assertLess(peak, materialized)  # the per-PA values cost less than the decision rows alone

    def test_stream_verifies_the_chain_up_to_the_head(self):
        rt, facts = pair_runtime(self.root), {}
        submit_pas(rt, 0, 3, facts)
        head = rt.summary()
        again = pair_runtime(self.root)  # reopen: every row replayed and verified
        self.assertEqual(again.summary(), head)
        self.assertEqual(again.submit(pr.DecisionRequest('0:0:0', '0:0', 0, PAState(0, 0, '9', 'L'), 'FF',
                                                         again.sha256, 'R'))['seq'], 1)  # replayed from its offset
        path = rt.ledger.path
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"SL"', b'"CH"', 1))  # tamper after the rows were written
        with self.assertRaisesRegex(pa.IntegrityError, 'tampered'):
            list(rt.ledger.iter_decisions())
        path.write_bytes(raw + raw.splitlines(keepends=True)[-1])  # a row beyond the in-memory head
        with self.assertRaisesRegex(pa.IntegrityError, 'beyond the in-memory chain head'):
            rt.summary()
        path.write_bytes(raw[:-len(raw.splitlines(keepends=True)[-1])])  # the last row removed
        with self.assertRaisesRegex(pa.IntegrityError, 'differs from the in-memory chain head'):
            rt.summary()
        path.write_bytes(raw)
        order = dict(reversed(list(facts.items())))  # facts order no longer the ledger order
        with self.assertRaises(pa.IntegrityError):
            est.estimate(rt.ledger.iter_decisions(), order, draws=5, seed=1, invalid_share_max=1.,
                         minimum={'games': 1, 'pa_starts': 1})
        with self.assertRaisesRegex(pa.IntegrityError, 'needs its facts'):
            est.estimate(rt.ledger.iter_decisions(), {'0:0': facts['0:0']}, draws=5, seed=1, invalid_share_max=1.,
                         minimum={'games': 1, 'pa_starts': 1})


if __name__ == '__main__':  # print the digests (used once on commit 25bcdb7 to fill GOLDEN)
    fixture = RunnerFixture()
    fixture.setUp()
    (fixture.root / 'stage').mkdir()
    print(json.dumps(stage_digests(fixture, fixture.root / 'stage'), indent=1))

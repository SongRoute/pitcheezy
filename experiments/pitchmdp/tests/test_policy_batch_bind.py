"""SYNTHETIC-ONLY (D125): binding the candidate to one game batch of context rows at a time
(``run_dr(rebind=..., batch_games=...)``, used by ope-2026 when ``bind_batch_games`` is registered)
writes the same ledger bytes and the same outputs as one unbatched pass, and the memory it keeps
no longer grows with the number of games."""
import copy
import hashlib
import json
import subprocess
import sys
import tracemalloc
import weakref
from pathlib import Path
from unittest import mock

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts'))
sys.path.insert(0, str(HERE.parent))
from test_policy_validation import MIN, NO_PITCH, RunnerFixture, blocks_of, build_store
import test_policy_validation as tpv
from pitchmdp.data import KEY
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as pr
from pitchmdp.rollout_policy import RowBudget
import run_policy_validation as rpv

AS_OF = '2025-07-01'


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
                          json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def many_games(frame, games):
    """The fixture frame with its two DEV games copied to ``games`` DEV games (new game_pk, July dates)."""
    dev = frame.loc[frame.split.eq('dev')]
    copies = []
    for g in range(games):
        source = dev.loc[dev.game_pk.eq(200 + g % 2)]
        date = (pd.Timestamp(AS_OF) + pd.Timedelta(days=g // 2)).strftime('%Y-%m-%d')
        copies.append(source.assign(game_pk=300 + g, game_date=date))
    out = pd.concat([frame.loc[~frame.split.eq('dev')], *copies])
    return out.sort_values(['game_date', *KEY], kind='stable').reset_index(drop=True)


class BatchBindTests(RunnerFixture):
    def stage(self, root, store, blocks, batch_games):
        """The ope-2026 shape of ``run_dr`` (frozen style snapshot, runtime v3, L1-R, every sensitivity,
        strata and the paired run); ``batch_games`` None = today's single binding."""
        root.mkdir()
        snapshot = preq.style_snapshot(self.frame, AS_OF)
        bc = lambda: pa.load_train_bc(*self.pins_['bc'])

        def build(part, budget, style):
            contexts, own = rpv.pa_contexts(store, part, snapshot, AS_OF)
            components = self.syn.bind(bc(), contexts)
            runtime = pr.build_runtime(*self.pins_['bc'], *self.pins_['support'], root / 'ledger.jsonl',
                                       components=components, budget=budget, tau=.01, samples=2, pitch_cap=4, seed=701,
                                       evaluation_seed=1701, hand_registry=self.pins_['hands'], positivity_record=True,
                                       provenance={'style_report': own if style is None else style})
            return runtime, components
        if batch_games is None:
            runtime, components = build(blocks, RowBudget(10 ** 8, seed_count=5), None)
            rebind, provenance = None, runtime.pins['provenance']
        else:
            style = rpv.style_report(store, blocks, snapshot, AS_OF)
            budget, runtime, components = RowBudget(10 ** 8, seed_count=5), None, None
            rebind, provenance = (lambda part: build(part, budget, style)), {'style_report': style}
        pair = lambda: pr.build_reference_pair_runtime(*self.pins_['bc'], *self.pins_['support'], root / 'pair.jsonl',
                                                        hand_registry=self.pins_['hands'], provenance=provenance,
                                                        positivity_record=True)
        result, rows = rpv.run_dr(runtime, store, components, blocks, rpv.Deadline(), no_pitch=NO_PITCH,
                                  bootstrap={'draws': 200, 'seed': 5, 'invalid_share_max': .5, 'minimum': MIN},
                                  ess_gate={'pa': 100., 'game': 30.},
                                  sensitivities=['r5-events-v1', 'flags_to_bounds', 'post_pitch_scores'],
                                  strata=rpv.strata_function(store.frame, [1., 2.]), expected_pas=len(blocks), pair=pair,
                                  estimator={'c_deltas': [0.01, 0.05]}, rebind=rebind, batch_games=batch_games)
        result.pop('seconds')  # wall clock
        table = pd.DataFrame([{**{k: v for k, v in r.items() if k != 'strata'},
                               **{f'stratum_{k}': v for k, v in r['strata'].items()}} for r in rows])
        return {'result': _sha(result), 'rows': _sha(rows), 'table': _sha(table.to_json().encode()),
                'ledger': _sha((root / 'ledger.jsonl').read_bytes()), 'pair': _sha((root / 'pair.jsonl').read_bytes())}

    def setUp(self):
        super().setUp()
        out, report = self.materialize()
        self.pins_ = self.pins(out, report)

    def test_batched_binding_writes_the_same_bytes(self):
        frame = many_games(self.frame, 5)
        store = build_store(frame)
        blocks = blocks_of(frame, 'dev')
        self.assertEqual([len({b[0].split(':')[0] for b in part}) for part in rpv.game_batches(frame, blocks, 2)],
                         [2, 2, 1])
        self.assertEqual(sum(rpv.game_batches(frame, blocks, 2), []), blocks)  # same blocks, same order
        snapshot = preq.style_snapshot(self.frame, AS_OF)
        self.assertEqual(rpv.style_report(store, blocks, snapshot, AS_OF),
                         rpv.pa_contexts(store, blocks, snapshot, AS_OF)[1])
        once = self.stage(self.root / 'once', store, blocks, None)
        for batch in (1, 2, 5):
            with self.subTest(batch=batch):
                self.assertEqual(self.stage(self.root / f'batched-{batch}', store, blocks, batch), once)

    def test_every_batch_is_rebound_and_checked(self):
        frame = many_games(self.frame, 3)
        store = build_store(frame)
        blocks, seen = blocks_of(frame, 'dev'), []
        snapshot = preq.style_snapshot(self.frame, AS_OF)
        budget = RowBudget(10 ** 8, seed_count=5)

        def rebind(part):
            components = self.syn.bind(pa.load_train_bc(*self.pins_['bc']), rpv.pa_contexts(store, part, snapshot, AS_OF)[0])
            seen.append((len(components.inputs.rows), components.sha256))
            return pr.build_runtime(*self.pins_['bc'], *self.pins_['support'], self.root / 'l.jsonl', components=components,
                                    budget=budget, tau=.01, samples=2, pitch_cap=4, seed=701,
                                    hand_registry=self.pins_['hands']), components
        games = preq.game_table(frame, {300, 301, 302})
        runtime, _, facts = rpv.submit_batches(rebind, store, blocks, rpv.Deadline(), 1, no_pitch=NO_PITCH, games=games)
        per_game = [sum(len(p) for _, p in part) for part in rpv.game_batches(frame, blocks, 1)]
        self.assertEqual([n for n, _ in seen], per_game)  # each binding holds one game's context rows only
        self.assertEqual(len(per_game), 3)
        self.assertEqual(len({s for _, s in seen}), 1)  # the component identity does not depend on the rows bound
        self.assertEqual((len(facts), runtime.ledger.kinds['decision']), (len(blocks), sum(f['submitted'] for f in facts.values())))
        with self.assertRaisesRegex(pa.IntegrityError, 'no pre-bound runtime'):
            rpv.run_dr(runtime, store, None, blocks, rpv.Deadline(), no_pitch=NO_PITCH, bootstrap={}, ess_gate={},
                       sensitivities=[], rebind=rebind, batch_games=1)

    def rebinder(self, frame, games, batch_games, released=None):
        store = build_store(frame)
        snapshot = preq.style_snapshot(self.frame, AS_OF)
        root = self.root / f'mem-{games}-{batch_games}'
        root.mkdir()
        budget = RowBudget(10 ** 8, seed_count=5)

        def rebind(part):
            if released is not None:  # every earlier batch's components are already freed (after gc)
                released.append(all(ref() is None for ref in released[0]))
            components = self.syn.bind(pa.load_train_bc(*self.pins_['bc']), rpv.pa_contexts(store, part, snapshot, AS_OF)[0])
            if released is not None:
                released[0] += [weakref.ref(components), weakref.ref(components.inputs), weakref.ref(components.g0)]
            return pr.build_runtime(*self.pins_['bc'], *self.pins_['support'], root / 'l.jsonl', components=components,
                                    budget=budget, tau=.01, samples=2, pitch_cap=4, seed=701,
                                    hand_registry=self.pins_['hands']), components
        return store, rebind

    def kept(self, games, batch_games):
        """Traced peak while submitting ``games`` DEV games in batches of ``batch_games``."""
        frame = many_games(self.frame, games)
        store, rebind = self.rebinder(frame, games, batch_games)
        table = preq.game_table(frame, set(frame.loc[frame.split.eq('dev')].game_pk.astype(int)))
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        result = rpv.submit_batches(rebind, store, blocks_of(frame, 'dev'), rpv.Deadline(), batch_games,
                                    no_pitch=NO_PITCH, games=table)
        peak = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        del result
        return peak

    def test_each_batch_releases_the_previous_binding(self):
        frame = many_games(self.frame, 4)
        released = [[]]
        store, rebind = self.rebinder(frame, 4, 1, released)
        rpv.submit_batches(rebind, store, blocks_of(frame, 'dev'), rpv.Deadline(), 1, no_pitch=NO_PITCH,
                           games=preq.game_table(frame, {300, 301, 302, 303}))
        self.assertEqual(released[1:], [True] * 4)

    def test_memory_is_flat_in_the_number_of_batches(self):
        """Traced peak while submitting: unbatched grows with every game's bound rows and caches; with
        one-game batches only the per-PA facts and ledger index grow. Measured in a fresh interpreter:
        inside the full suite a one-off interpreter table resize (sys.intern, ~7 MiB) can land in the
        window and swamp this small synthetic signal."""
        out = subprocess.run([sys.executable, __file__, 'memory'], capture_output=True, text=True, check=True)
        growth = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"\ntraced peak growth 2->12 games: unbatched {growth['unbatched'] / 1024:.0f} KiB, "
              f"one-game batches {growth['batched'] / 1024:.0f} KiB")
        self.assertGreater(growth['unbatched'], 0)
        self.assertLess(growth['batched'], growth['unbatched'] / 3)


class BatchedOPEDispatchTests(RunnerFixture):
    """The registered chain of ``DispatchTests`` through ``dispatch``; right after its sealed ope-2026
    (OPE-a1) the same registration with ``bind_batch_games: 1`` (one game per binding) runs under its
    own root and must write the same ledgers, PA table and record (only the batch size is added)."""
    config = tpv.DispatchTests.config

    def test_batched_ope_through_dispatch_writes_the_same_bytes(self):
        real, compared = rpv.dispatch, []

        def dispatch(command, reg, local, output, load, **kwargs):
            real(command, reg, local, output, load, **kwargs)
            if command != rpv.OPE or output.name != 'OPE-a1':
                return
            root = output.parent.with_name('OPE-2026-batched')
            root.mkdir()
            other = copy.deepcopy(reg)
            other['config']['mlb2026_ope'].update(output_root=str(root), bind_batch_games=1)
            real(command, other, local, root / 'OPE-a1', load, **kwargs)
            once, batched = (json.loads((d / 'ope2026.json').read_text()) for d in (output, root / 'OPE-a1'))
            self.assertEqual(batched.pop('bind_batch_games'), 1)
            for key in ('seconds', 'prior_attempts'):
                once.pop(key), batched.pop(key)
            self.assertEqual(batched, once)
            for name in ('ledger-eval2026-0.jsonl', 'ledger-2026-paired-identity.jsonl', 'pa_values.parquet'):
                self.assertEqual((root / 'OPE-a1' / name).read_bytes(), (output / name).read_bytes(), name)
            other['config']['mlb2026_ope']['bind_batch_games'] = 0
            with self.assertRaisesRegex(pa.IntegrityError, 'bind_batch_games'):
                rpv.registration(other['config'], rpv.OPE)
            compared.append(once['games'])
        with mock.patch.object(rpv, 'dispatch', dispatch):
            tpv.DispatchTests.test_registered_chain_through_dispatch(self)
        self.assertEqual(compared, [2])  # two synthetic 2026 games: two bindings


def memory_growth(small=2, large=12):
    fixture = BatchBindTests()
    fixture.setUp()
    return {'unbatched': fixture.kept(large, large) - fixture.kept(small, small),
            'batched': fixture.kept(large, 1) - fixture.kept(small, 1)}


if __name__ == '__main__':
    if sys.argv[1:] == ['memory']:
        print(json.dumps(memory_growth()))
    else:
        import unittest
        unittest.main()

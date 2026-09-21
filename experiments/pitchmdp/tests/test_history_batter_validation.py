"""Synthetic CPU-only tests: never fit a real model or open research datasets."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.special import softmax
import history_batter_support as support
import run_history_batter_validation as runner
from pitchmdp.data import KEY
from pitchmdp.sequence_model import classification_metrics
from pitchmdp.sequence_model import SequenceModel


class FakeStore:
    frame = pd.DataFrame({'x': [0, 1, 2]})
    normalizer = object()

    def __init__(self):
        self.tokens = np.arange(144, dtype=np.float32).reshape(3, 6, 8)
        self.valid = np.array([[False]*5+[True], [False]*3+[True]*3, [True]*6])

    def gather(self, rows, current=None):
        tokens, valid = self.tokens[rows].copy(), self.valid[rows].copy()
        if current is not None:
            tokens[:, -1] = current
        return tokens, valid


class FakeDelivery:
    def logits(self, model, store, context, rows, chunk_size):
        logits = (np.asarray(rows)[:, None, None]*np.linspace(-.2, .3, 10)[None, None, :]
                  + np.linspace(-1, 1, 400)[None, :, None]*np.linspace(.1, 1, 10)[None, None, :])
        return logits.astype(np.float32), np.zeros(len(rows), dtype=int)


class FakeModel:
    fits = 0

    def __init__(self, *args, **kwargs):
        self.temperature = 1.
        self.delivery_temperature = 1.
        self.report = {'seconds': 0.}

    def fit(self, *args, **kwargs):
        type(self).fits += 1

    def save(self, path):
        path.write_text(json.dumps({'temperature': self.delivery_temperature, 'report': self.report}))

    @classmethod
    def load(cls, path):
        saved = json.loads(path.read_text())
        model = cls()
        model.delivery_temperature, model.report = saved['temperature'], saved['report']
        return model


class HistoryBatterTests(unittest.TestCase):
    def test_inference_tuning_does_not_change_earlystop_batches(self):
        model = object.__new__(support.BatchedModel)
        model.inference_batch = 8192

        def fake_fit(self):
            self.logits(None)
            return self
        with patch.object(SequenceModel, 'fit', fake_fit), patch.object(SequenceModel, 'logits') as logits:
            model.fit()
            logits.assert_called_with(None, 4096)
            model.logits(None)
            logits.assert_called_with(None, 8192)

    def test_zero_window_preserves_candidate_and_does_not_mutate_original(self):
        store = FakeStore()
        before, masks = store.tokens.copy(), store.valid.copy()
        rows = np.array([2, 0, 2])
        candidate = np.full((3, 8), -123, dtype=np.float32)
        tokens, valid = support.NoPastStore(store).gather(rows, candidate)
        np.testing.assert_array_equal(tokens[:, :5], np.zeros((3, 5, 8)))
        self.assertFalse(valid[:, :5].any())
        self.assertTrue(valid[:, -1].all())
        np.testing.assert_array_equal(tokens[:, -1], candidate)
        np.testing.assert_array_equal(store.tokens, before)
        np.testing.assert_array_equal(store.valid, masks)

    def test_streaming_matches_full_softmax_then_average(self):
        rows = np.array([0, 7, 2, 19, 3, 5, 6])
        delivery = FakeDelivery()
        model = FakeModel()
        model.delivery_temperature = 1.37
        logits, _ = delivery.logits(model, None, None, rows, len(rows))
        expected = softmax(logits/model.delivery_temperature, axis=-1).mean(1)
        for chunk in (1, 2, 4, 20):
            p = support.integrated_predict(delivery, model, None, None, rows, chunk)
            np.testing.assert_array_equal(p, expected)
        self.assertGreater(np.abs(expected-softmax(logits.mean(1)/model.delivery_temperature, axis=-1)).max(), 1e-5)

    def test_stage_interrupt_is_recoverable_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fit'
            contract = {'seed': 42}

            def fail(stage):
                (stage/'model').write_text('partial')
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                support.commit_stage(path, contract, fail)
            self.assertFalse(path.exists())
            writer = Mock(side_effect=lambda stage: (stage/'model').write_text('complete'))
            self.assertTrue(support.commit_stage(path, contract, writer))
            self.assertFalse(support.commit_stage(path, contract, writer))
            self.assertEqual(writer.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                support.read_stage(path, {'seed': 43})
            (path/'model').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                support.read_stage(path, contract)

    def test_shared_lock_excludes_concurrent_runs_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'lock'
            with support.execution_lock(path):
                with self.assertRaises(RuntimeError):
                    with support.execution_lock(path):
                        pass
            with support.execution_lock(path):
                pass

    def test_bootstrap_linear_interaction_sign_and_paired_games(self):
        y = np.zeros(9, int)
        games = np.array([1, 1, 1, 1, 1, 2, 3, 3, 4])
        predictions = {}
        for name in support.REPRESENTATIONS:
            for h in (0, 5):
                correct = np.full(9, .5)
                if h == 0:
                    correct *= np.exp(-np.linspace(.1, .9, 9))
                    if name == 'reference':
                        correct *= np.exp(-.2)
                p = np.repeat(((1-correct)/9)[:, None], 10, axis=1)
                p[:, 0] = correct
                predictions[name+f'_h{h}'] = p
        result = support.compare_families(y, predictions, games, replicates=300)
        self.assertEqual([len(v) for v in result['contrasts'].values()], [6, 10, 5])
        interaction = result['families']['interaction']['log_loss']['reference']
        np.testing.assert_allclose(interaction['estimate'], .2, atol=1e-14)
        np.testing.assert_allclose(interaction['simultaneous95'], [.2, .2], atol=1e-14)
        history = result['families']['history']['log_loss']['continuous']
        unique, ix = np.unique(games, return_inverse=True)
        draws = np.random.default_rng(42).integers(0, len(unique), size=(300, len(unique)))
        expected = np.bincount(ix, weights=np.linspace(.1, .9, 9))[draws].sum(1)/np.bincount(ix)[draws].sum(1)
        np.testing.assert_allclose(history['pointwise95'], np.quantile(expected, [.025, .975]))

    def test_dry_run_does_not_execute_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)/'project'
            (project/'configs').mkdir(parents=True)
            root = Path(tmp)/'ssd'
            (project/'configs/local.json').write_text(json.dumps({'artifact_root': str(root)}))
            with patch.object(runner, 'PROJECT', project), patch.object(runner, 'execute') as execute:
                with redirect_stdout(io.StringIO()) as out:
                    runner.main(['--dry-run'])
                result = json.loads(out.getvalue())
                self.assertEqual(result['plan']['new_models'], 60)
                self.assertFalse(result['training'])
                execute.assert_not_called()
                self.assertFalse(root.exists())

    def test_plan_rejects_reference_overwrite_and_outside_ssd(self):
        local = {'artifact_root': '/tmp/synthetic-ssd'}
        for output in ('/tmp/outside', '/tmp/synthetic-ssd/runs',
                       '/tmp/synthetic-ssd/runs/'+runner.DEFAULT_BASE,
                       '/tmp/synthetic-ssd/runs/'+runner.DEFAULT_REPRESENTATIONS+'/nested'):
            with self.assertRaises(ValueError):
                runner.make_plan(runner.arguments(['--output', output]), local)

    def test_resume_after_calibration_interrupt_does_not_refit(self):
        FakeModel.fits = 0
        parts = {name: pd.DataFrame({'game_pk': [1, 2], 'at_bat_number': [1, 1], 'pitch_number': [1, 2]})
                 for name in ('temperature', 'blend', 'dev')}
        ys = {name: np.array([0, 1]) for name in ('train', 'earlystop', *parts)}
        torch = Mock()
        torch.backends.mps.is_available.return_value = False
        dependencies = {'np': np, 'torch': torch, 'KEY': KEY, 'BatchedModel': FakeModel, 'hash_file': support.hash_file,
            'read_stage': support.read_stage, 'commit_stage': support.commit_stage,
            'dump': support.dump, 'classification_metrics': classification_metrics,
            'integrated_chunks': support.integrated_chunks, 'integrated_predict': support.integrated_predict,
            'fit_temperature_logits': lambda logits, y: (1.37, .5)}
        contract = {'year': 2024, 'representation': 'reference', 'seed': 42, 'past_tokens': 0}
        plan = {'cpu_threads': 4, 'delivery_chunk': 1}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(runner.__dict__, dependencies), redirect_stdout(io.StringIO()):
            path = Path(tmp)/'member'
            with patch.object(runner, 'integrated_chunks', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner.member_prediction(path, contract, None, None, ys, parts, FakeDelivery(), None, None, plan)
            self.assertEqual(FakeModel.fits, 1)
            self.assertTrue((path/'fit/stage.json').exists())
            self.assertFalse((path/'calibration').exists())
            first = runner.member_prediction(path, contract, None, None, ys, parts, FakeDelivery(), None, None, plan)
            second = runner.member_prediction(path, contract, None, None, ys, parts, FakeDelivery(), None, None, plan)
            self.assertEqual(FakeModel.fits, 1)
            for part in ('blend', 'dev'):
                np.testing.assert_array_equal(first[part], second[part])
                logits, _ = FakeDelivery().logits(None, None, None, np.array([0, 1]), 1)
                np.testing.assert_array_equal(first[part], softmax(logits/1.37, axis=-1).mean(1))
            calibration = json.loads((path/'calibration/stage.json').read_text())
            self.assertEqual(calibration['contract']['fit_stage_sha256'], support.hash_file(path/'fit/stage.json'))
            prediction = json.loads((path/'dev/stage.json').read_text())
            self.assertEqual(prediction['contract']['calibration_stage_sha256'], support.hash_file(path/'calibration/stage.json'))


if __name__ == '__main__':
    unittest.main()

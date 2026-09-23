"""Tiny CPU tests; no experimental data or accelerator training."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.matrix_models import MatrixModel, MatrixNetwork
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.sequence_delivery import JointDelivery
from test_matrix_features import frame


def arrays(length=6):
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(12, length, 23)).astype(np.float32)
    tokens[:, -1, -11:] = 0.
    mask = np.ones((12, length), dtype=bool)
    if length > 1:
        mask[:, :2] = False
    context = rng.normal(size=(12, 4)).astype(np.float32)
    return tokens, mask, context


class MatrixModelsTests(unittest.TestCase):
    def test_every_torch_family_fit_roundtrip_and_padding_neutrality(self):
        a, y = arrays(), np.arange(12) % 10
        for kind in ('linear', 'flatten_mlp', 'transformer', 'lstm', 'gru', 'melville'):
            with self.subTest(kind=kind):
                model = MatrixModel(kind, seed=3, width=8, device='cpu').fit(
                    a, y, a, y, epochs=2, patience=2, batch_size=6,
                    sample_weight=np.arange(12) + 1.)
                logits = model.logits(a)
                changed = tuple(value.copy() for value in a)
                changed[0][~changed[1]] = 10000.
                np.testing.assert_allclose(logits, model.logits(changed), rtol=0, atol=0)
                np.testing.assert_allclose(model.predict(a).sum(1), 1, atol=1e-6)
                model.delivery_temperature = 1.37
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'model.pt'
                    model.save(path)
                    restored = MatrixModel.load(path, device='cpu')
                    np.testing.assert_array_equal(logits, restored.logits(a))
                    self.assertEqual(restored.delivery_temperature, 1.37)
                    self.assertEqual(restored.report, model.report)

    def test_zero_history_all_families_and_current_label_rejection(self):
        for kind in ('linear', 'flatten_mlp', 'transformer', 'lstm', 'gru', 'melville'):
            model = MatrixModel(kind, width=8, device='cpu')
            a = arrays(1)
            model.net = MatrixNetwork(kind, 4, 23, 1, width=8)
            self.assertEqual(model.logits(a).shape, (12, 10))
            a[0][0, -1, -1] = 1
            with self.assertRaisesRegex(ValueError, 'current-outcome'):
                model.logits(a)

    def test_same_seed_reproducible_cpu(self):
        a, y = arrays(), np.arange(12) % 10
        m1 = MatrixModel('flatten_mlp', seed=19, width=8, device='cpu').fit(a, y, a, y, epochs=2)
        m2 = MatrixModel('flatten_mlp', seed=19, width=8, device='cpu').fit(a, y, a, y, epochs=2)
        np.testing.assert_array_equal(m1.logits(a), m2.logits(a))

    def test_joint_delivery_integration_overwrites_current_physics(self):
        store = MatrixHistoryStore.from_frame(frame())
        class Context:
            def transform(self, df):
                return np.zeros((len(df), 4), dtype=np.float32)
        context = Context()
        tokens, mask = store.gather([2, 5])
        a = tokens, mask, context.transform(store.frame.iloc[[2, 5]])
        model = MatrixModel('linear', width=8, device='cpu').fit(a, [0, 2], a, [0, 2], epochs=1)
        delivery = JointDelivery().fit(store.frame.loc[store.frame.split.eq('train')], store.normalizer, draws=4)
        before = delivery.predict(model, store, context, np.array([2]))
        # Row 2 is current, never historical for this requested prediction.
        store.physical[2] = 99999.
        after = delivery.predict(model, store, context, np.array([2]))
        np.testing.assert_array_equal(before, after)
        delivery.calibrate(model, store, context, np.array([2]), np.array([0]))
        self.assertTrue(.5 <= model.delivery_temperature <= 2.5)

    def test_melville_current_outcome_not_connected_to_network(self):
        net = MatrixNetwork('melville', 4, 23, 6, width=8)
        t, valid, c = [torch.as_tensor(value) for value in arrays()]
        before = net(t, valid, c).detach().numpy()
        t[:, -1, -11:] = 999.
        np.testing.assert_array_equal(before, net(t, valid, c).detach().numpy())


if __name__ == '__main__':
    unittest.main()

"""CPU checks that fixed-checkpoint audit distinguishes calibration from retraining."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_sequence_integration import unchanged_checkpoint


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.original = {'network_config': {'kind': 'transformer', 'width': 128}, 'seed': 42,
                         'temperature': 1.1, 'delivery_temperature': 1.2,
                         'state_dict': {'weight': torch.tensor([[1., 2.], [3., 4.]])}}

    def test_delivery_temperature_update_preserves_neural_identity(self):
        updated = copy.deepcopy(self.original)
        updated['delivery_temperature'] = .9
        self.assertEqual(unchanged_checkpoint(self.original, updated), unchanged_checkpoint(self.original, self.original))

    def test_rejects_tensor_seed_architecture_or_conditional_temperature_change(self):
        for field, value in [('seed', 43), ('network_config', {'kind': 'flatten_mlp', 'width': 128}), ('temperature', 1.3)]:
            with self.subTest(field=field):
                updated = copy.deepcopy(self.original)
                updated[field] = value
                with self.assertRaises(AssertionError):
                    unchanged_checkpoint(self.original, updated)
        updated = copy.deepcopy(self.original)
        updated['state_dict']['weight'][0, 0] += .001
        with self.assertRaises(AssertionError):
            unchanged_checkpoint(self.original, updated)


if __name__ == '__main__':
    unittest.main()

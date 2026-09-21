"""CPU-only exact partition guard for the saved-probability 100-draw reblend."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_sequence_context_blend100 import key_hash, verify_partition_keys


class ReblendPartitionContracts(unittest.TestCase):
    @staticmethod
    def keys(n, game):
        return np.column_stack((np.full(n, game), np.arange(n), np.ones(n))).astype(np.int64)

    def test_exact_disjoint_partition_and_ordered_hash(self):
        temperature, blend, dev = self.keys(4000, 1), self.keys(12000, 2), self.keys(7276, 3)
        verify_partition_keys(temperature, blend, dev)
        self.assertEqual(key_hash(dev), key_hash(dev.astype(np.float64)))
        self.assertNotEqual(key_hash(dev), key_hash(dev[::-1]))

    def test_wrong_sizes_duplicates_and_cross_partition_reuse_rejected(self):
        temperature, blend, dev = self.keys(4000, 1), self.keys(12000, 2), self.keys(7276, 3)
        with self.assertRaises(ValueError):
            verify_partition_keys(temperature[:-1], blend, dev)
        duplicated = blend.copy(); duplicated[0] = duplicated[1]
        with self.assertRaises(ValueError):
            verify_partition_keys(temperature, duplicated, dev)
        overlap = blend.copy(); overlap[0] = temperature[0]
        with self.assertRaises(ValueError):
            verify_partition_keys(temperature, overlap, dev)


if __name__ == "__main__":
    unittest.main()

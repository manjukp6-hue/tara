"""
tests/test_fast_weight_plasticity.py

Verification suite for Real-Time Synaptic Plasticity & Fast-Weight Cognitive Adapter.
"""

import unittest
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.fast_weight_plasticity import FastWeightPlasticityEngine


class TestFastWeightPlasticity(unittest.TestCase):

    def setUp(self):
        self.engine = FastWeightPlasticityEngine(dimension=16, decay_rate=0.90, learning_rate=0.80)

    def test_01_associative_update_and_retrieval(self):
        # Key vector representing a concept: [1, 0, 0, ...]
        key = [1.0 if i == 0 else 0.0 for i in range(16)]
        # Value vector representing target reaction: [0, 1, 0, ...]
        val = [1.0 if i == 1 else 0.0 for i in range(16)]

        # Before update, retrieval should be zero
        initial_out = self.engine.retrieve_association(key)
        self.assertAlmostEqual(sum(abs(x) for x in initial_out), 0.0, places=5)

        # Apply Hebbian update
        self.engine.update_synapses(key, val)
        self.assertEqual(self.engine.total_updates, 1)

        # Now query with key, should retrieve value
        retrieved = self.engine.retrieve_association(key)
        self.assertGreater(retrieved[1], 0.5)

    def test_02_linear_projection_adaptation(self):
        base_proj = [0.5] * 16
        input_x = [1.0] * 16
        val = [2.0] * 16

        self.engine.update_synapses(input_x, val)
        adapted = self.engine.adapt_linear_projection(base_proj, input_x, adaptation_scale=0.25)

        self.assertEqual(len(adapted), 16)
        # Verify adaptation delta took effect
        self.assertNotEqual(adapted, base_proj)
        self.assertGreater(adapted[0], base_proj[0])

    def test_03_decay_and_reset(self):
        key = [1.0 if i == 2 else 0.0 for i in range(16)]
        val = [1.0 if i == 3 else 0.0 for i in range(16)]

        self.engine.update_synapses(key, val)
        act1 = self.engine.retrieve_association(key)[3]

        # Multiple dummy updates without (key, val) to test decay
        dummy_k = [0.1] * 16
        dummy_v = [0.1] * 16
        for _ in range(5):
            self.engine.update_synapses(dummy_k, dummy_v)

        act2 = self.engine.retrieve_association(key)[3]
        self.assertLess(act2, act1)

        # Full reset
        self.engine.reset_synapses()
        self.assertEqual(self.engine.total_updates, 0)
        self.assertAlmostEqual(self.engine.retrieve_association(key)[3], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

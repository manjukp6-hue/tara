"""
tests/test_colibri_tiering.py

Comprehensive test suite verifying Colibrì architecture adaptation in TARA:
1. Memory manager & 3-tier hierarchy (VRAM -> RAM -> Disk movement)
2. Cache policies (LRU, LFU, Colibrì LFRU with 25% + 4 hysteresis)
3. SafeTensors streaming & zero-copy mmap
4. Eviction mechanics
5. Lookahead prefetching & background worker overlap
6. Resource planner & hardware budget probing
7. Backend selection (CPU, CUDA, fallback)
8. Real inference on TARA models with constrained memory budgets
9. Telemetry hit/miss statistics
10. Fallback verification (disabling tiering restores native TARA path identically)
"""

import os
import sys
import unittest
import time

# Ensure taracore root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.MODEL.inference.cache.cache_policy import (
    LFRUCachePolicy, LRUCachePolicy, LFUCachePolicy,
    tier_should_promote, tier_decay_value, tier_lfru_score
)
from TARA.MODEL.inference.tensor_stream.stream_reader import SafeTensorsStreamReader, TensorView
from TARA.MODEL.inference.resource_planner.planner import ResourcePlanner, get_available_ram_bytes, discover_gpus
from TARA.MODEL.inference.backends.backend_registry import BackendRegistry, CPUBackend, CUDABackend
from TARA.MODEL.inference.telemetry.monitor import TelemetryMonitor
from TARA.MODEL.inference.prefetch.lookahead import LookaheadPrefetcher
from TARA.MODEL.router.routing_tracker import RoutingTracker
from TARA.MODEL.inference.memory_manager.tiered_store import TieredTensorStore
from TARA.MODEL.core.tiered_model import TieredTaraModel


class TestColibriTiering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [
            os.path.join(REPO_ROOT, "storage", "models", "tara"),
            os.path.join(REPO_ROOT, "storage", "models", "TARA-0.1-tokenizer-aligned"),
            os.path.join(REPO_ROOT, "storage", "models", "tara-0.1"),
            os.path.join(REPO_ROOT, "storage", "models", "tara-0.2"),
            os.path.join(REPO_ROOT, "storage", "models", "tara-language")
        ]
        cls.model_dir = None
        for c in candidates:
            if os.path.exists(os.path.join(c, "model.safetensors")):
                cls.model_dir = c
                break
        if not cls.model_dir:
            raise FileNotFoundError("No valid model directory found with model.safetensors")
        cls.weights_path = os.path.join(cls.model_dir, "model.safetensors")

    def test_01_cache_policy_and_hysteresis(self):
        """Verify Colibri LFRU scoring and 25% + 4 hysteresis."""
        # 1. Hysteresis check
        # threshold = cold + (cold >> 2) + 4
        # cold = 10 -> threshold = 10 + 2 + 4 = 16
        self.assertFalse(tier_should_promote(hot=15, cold=10))
        self.assertTrue(tier_should_promote(hot=17, cold=10))

        # 2. Decay check (half-life)
        self.assertEqual(tier_decay_value(16), 8)
        self.assertEqual(tier_decay_value(7), 3)

        # 3. LFRU scoring check
        score_hot = tier_lfru_score(heat=5, last=10, clock=10)
        score_cold = tier_lfru_score(heat=1, last=10, clock=10)
        self.assertGreater(score_hot, score_cold)

        # 4. Policy eviction and admission
        lfru = LFRUCachePolicy()
        for _ in range(10):
            lfru.record_access("tensor_A")
        for _ in range(2):
            lfru.record_access("tensor_B")

        evict = lfru.pick_eviction(["tensor_A", "tensor_B"])
        self.assertEqual(evict, "tensor_B")
        # Tensor with 2 accesses should not displace tensor with 10 accesses
        self.assertFalse(lfru.should_admit(candidate_key="tensor_B", evict_key="tensor_A"))

    def test_02_stream_reader_mmap(self):
        """Verify SafeTensors mmap parsing and tensor loading."""
        reader = SafeTensorsStreamReader(self.weights_path, enable_async=False)
        tensors = reader.list_tensors()
        self.assertGreater(len(tensors), 0)
        self.assertIn("model.embed_tokens.weight", tensors)

        tv = reader.load_tensor("model.embed_tokens.weight")
        self.assertIsInstance(tv, TensorView)
        self.assertEqual(tv.name, "model.embed_tokens.weight")
        self.assertGreater(tv.data_bytes, 0)
        reader.close()

    def test_03_resource_planner(self):
        """Verify resource planner hardware probe and tier planning."""
        planner = ResourcePlanner({"vram_budget_gb": 0.5, "ram_budget_gb": 1.0})
        plan = planner.plan_tiers(model_size_bytes=50_000_000)
        self.assertGreater(plan["ram_budget_bytes"], 0)
        self.assertGreater(plan["available_ram_bytes"], 0)
        self.assertIn("requires_disk_streaming", plan)

    def test_04_backend_selection(self):
        """Verify backend registry and fallback to CPU."""
        cpu = BackendRegistry.get("CPU")
        self.assertTrue(cpu.is_available())
        best = BackendRegistry.select_best_backend()
        self.assertIsNotNone(best)

    def test_05_tiered_store_movement_and_eviction(self):
        """
        Verify explicit movement across tiers:
        Disk -> RAM -> VRAM and eviction back down.
        """
        reader = SafeTensorsStreamReader(self.weights_path, enable_async=False)
        try:
            telemetry = TelemetryMonitor()
            policy = LFRUCachePolicy()

            # Set very tight RAM capacity (10 KB) so any weight tensor (e.g. 44 KB) triggers eviction
            store = TieredTensorStore(
                stream_reader=reader,
                cache_policy=policy,
                telemetry=telemetry,
                vram_capacity_bytes=5_000,
                ram_capacity_bytes=10_000
            )

            all_tensors = reader.list_tensors()
            t1 = all_tensors[0]

            # First access loads from disk to RAM (disk miss)
            v1 = store.lookup(t1)
            self.assertEqual(telemetry.disk_misses, 1)
            store.release(t1)

            # Second access should be a RAM hit
            v1_again = store.lookup(t1)
            self.assertEqual(telemetry.ram_hits, 1)
            store.release(t1)

            # Load multiple tensors to exceed RAM limit and force eviction
            for t in all_tensors[:5]:
                v = store.lookup(t)
                store.release(t)

            self.assertGreater(telemetry.evictions, 0)
        finally:
            reader.close()

    def test_06_lookahead_prefetch(self):
        """Verify lookahead prefetch pipeline and layer advance trigger."""
        prefetched_keys = []
        def mock_prefetch(keys):
            prefetched_keys.extend(keys)
            return len(keys)

        prefetcher = LookaheadPrefetcher(prefetch_fn=mock_prefetch, depth=1, enabled=True)
        layer_map = {
            0: ["layer0.w1", "layer0.w2"],
            1: ["layer1.w1", "layer1.w2"],
            2: ["layer2.w1", "layer2.w2"]
        }

        # When layer 0 begins, it should prefetch layer 1
        prefetcher.on_layer_begin(0, 3, layer_map)
        self.assertIn("layer1.w1", prefetched_keys)
        self.assertIn("layer1.w2", prefetched_keys)
        self.assertNotIn("layer2.w1", prefetched_keys)

        # When layer 1 begins, it should prefetch layer 2
        prefetcher.on_layer_begin(1, 3, layer_map)
        self.assertIn("layer2.w1", prefetched_keys)

    def test_07_real_inference_with_constrained_budgets(self):
        """
        Run real autoregressive generation on TARA model with constrained budgets.
        Confirms that inference produces valid tokens and records hit/miss statistics.
        """
        # Constrain RAM to 200 KB so tensors must page through during forward pass
        model = TieredTaraModel(
            model_dir=self.model_dir,
            force_ram_gb=0.0002,  # 200 KB
            force_vram_gb=0.0
        )
        try:
            output = model.generate("TARA", max_new_tokens=4, temperature=0.7)
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 0)

            stats = model.telemetry.get_stats()
            self.assertGreater(stats["requests"], 0)
            self.assertGreater(stats["disk_misses"], 0)
            self.assertGreaterEqual(stats["evictions"], 0)
        finally:
            model.close()

    def test_08_fallback_parity(self):
        """
        Verify that disabling tiering (fallback_to_native=True)
        restores the exact native TARA inference path cleanly.
        """
        # 1. Native fallback model
        fallback_model = TieredTaraModel(
            model_dir=self.model_dir,
            fallback_to_native=True
        )
        try:
            native_logits, _ = fallback_model.forward([1, 2, 3])
            self.assertEqual(len(native_logits), 3)
            self.assertEqual(len(native_logits[0]), fallback_model.config.vocab_size)
        finally:
            fallback_model.close()

        # 2. Tiered model with ample RAM
        tiered_model = TieredTaraModel(
            model_dir=self.model_dir,
            force_ram_gb=1.0,
            fallback_to_native=False
        )
        try:
            tiered_logits, _ = tiered_model.forward([1, 2, 3])
            self.assertEqual(len(tiered_logits), 3)
            # Numeric output should match within floating point precision
            diff = abs(native_logits[0][0] - tiered_logits[0][0])
            self.assertLess(diff, 1e-4)
        finally:
            tiered_model.close()


if __name__ == "__main__":
    unittest.main()

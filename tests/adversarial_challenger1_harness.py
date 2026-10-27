#!/usr/bin/env python3
"""
Adversarial Stress Testing Harness for TARA Pipeline Hardening.
Challenger 1: TaraMultiDomainScheduler and AnchorReplayBuffer.

Focus Areas:
1. Extreme & Adversarial Domain Weight Distributions
2. Core Domain Exhaustion & Anchor Replay Ratio (1,000+ batches)
3. Memory Stability & Bounded RAM Verification (< 500 MB Colab limit)
4. Edge Case: Complete Dataset Exhaustion Termination Boundary
"""

import os
import sys
import json
import math
import time
import shutil
import random
import tempfile
import unittest
import tracemalloc
from typing import Dict, List, Any

# Ensure taracore root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import cloud_train
from cloud_train import (
    CANONICAL_DOMAINS,
    CORE_ANCHOR_DOMAINS,
    parse_domain_weights,
    TaraMultiDomainScheduler,
    TaraDomainBatchItem,
)
AnchorReplayBuffer = getattr(cloud_train, "AnchorReplayBuffer", None)


class MockSimpleTokenizer:
    """Lightweight deterministic tokenizer for stress tests."""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def encode(self, text: str, **kwargs) -> List[int]:
        # Return deterministic hash tokens bounded by vocab_size
        return [(hash(w) % (self.vocab_size - 4)) + 4 for w in text.split()] or [1]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return " ".join(str(t) for t in token_ids)


class AdversarialSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="challenger1_stress_")
        self.tokenizer = MockSimpleTokenizer()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_shard(self, domain: str, shard_name: str, num_records: int) -> str:
        domain_dir = os.path.join(self.temp_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        shard_path = os.path.join(domain_dir, shard_name)
        with open(shard_path, "w", encoding="utf-8") as f:
            for i in range(num_records):
                record = {
                    "text": f"Adversarial record {i} for domain {domain} with substantial payload content to simulate token stream.",
                    "meta": {"domain": domain, "idx": i}
                }
                f.write(json.dumps(record) + "\n")
        return shard_path

    # =========================================================================
    # Suite 1: Extreme & Adversarial Domain Weight Distributions
    # =========================================================================

    def test_01_empty_and_whitespace_weights(self):
        """Verify empty and whitespace weight strings yield valid normalized weights summing to 1.0."""
        for raw in ["", "   ", "\t\n  \t"]:
            weights = parse_domain_weights(raw, CANONICAL_DOMAINS)
            self.assertEqual(len(weights), len(CANONICAL_DOMAINS))
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=5, msg=f"Total sum failed for {repr(raw)}")
            for d, w in weights.items():
                self.assertGreater(w, 0.0, f"Weight for {d} must be strictly positive")

    def test_02_unknown_and_alien_domains(self):
        """Verify weights specifying unknown/alien domains don't crash and normalize valid domains."""
        raw = "alien_domain:0.5, martian_data:0.8, galaxy_deep_learning:10.0"
        weights = parse_domain_weights(raw, CANONICAL_DOMAINS)
        self.assertEqual(len(weights), len(CANONICAL_DOMAINS))
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        for d, w in weights.items():
            self.assertGreater(w, 0.0)

    def test_03_highly_skewed_and_extreme_weights(self):
        """Verify extreme floating point ratios (1e15 vs 1e-15) normalize without NaN or overflow."""
        raw = "01_math:1e12, 02_code:1.0, 04_kannada:1e-12"
        weights = parse_domain_weights(raw, CANONICAL_DOMAINS)
        total = sum(weights.values())
        self.assertFalse(math.isnan(total), "Weight sum resulted in NaN")
        self.assertFalse(math.isinf(total), "Weight sum resulted in Inf")
        self.assertAlmostEqual(total, 1.0, places=5)
        # Math should dominate
        self.assertGreater(weights["01_mathematics_formal_logic"], 0.99)

    def test_04_negative_and_zero_weights(self):
        """Verify negative weights are clamped to 0.0 and all-zero does not divide by zero."""
        raw = "01_math:-10.0, 02_code:0.0, 04_kannada:-0.5"
        weights = parse_domain_weights(raw, CANONICAL_DOMAINS)
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        for d, w in weights.items():
            self.assertGreaterEqual(w, 0.0)

    def test_05_all_zero_weights(self):
        """Verify setting every domain to 0 falls back to valid uniform distribution."""
        raw = ",".join(f"{d}:0.0" for d in CANONICAL_DOMAINS)
        weights = parse_domain_weights(raw, CANONICAL_DOMAINS)
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        uniform_val = 1.0 / len(CANONICAL_DOMAINS)
        for d, w in weights.items():
            self.assertAlmostEqual(w, uniform_val, places=4)

    def test_06_malformed_string_tokens(self):
        """Verify strings with corrupt syntax, missing values, multiple colons, or garbage don't raise."""
        garbage_inputs = [
            ":::",
            ",,,",
            "01_math:",
            ":0.5",
            "===",
            "01_math=0.5=extra=tokens",
            "foo:bar:baz",
            "01_math:not_a_number",
            "   01_math : 0.35 , 02_code = 0.65 ,,, invalid ",
        ]
        for g in garbage_inputs:
            try:
                weights = parse_domain_weights(g, CANONICAL_DOMAINS)
                total = sum(weights.values())
                self.assertAlmostEqual(total, 1.0, places=5, msg=f"Failed on garbage: {g}")
            except Exception as e:
                self.fail(f"parse_domain_weights raised exception on garbage input '{g}': {e}")

    # =========================================================================
    # Suite 2: Core Domain Exhaustion & Anchor Replay Ratio
    # =========================================================================

    def test_07_core_domain_depletion_and_anchor_replay_ratio(self):
        """
        Stress test: simulate rapid EOF on core domains (math: 15 samples, reasoning: 15 samples,
        kannada: 15 samples, tool_use: 15 samples) while code stream has 3,000 samples.
        Collect 1,200 batches and empirically verify:
        - Depleted core domains switch to anchor replay.
        - Anchor replay ratio remains strictly within 15%–20% (± statistical tolerance).
        - Replay anchors are sampled across all depleted core domains.
        """
        domain_shards = {}
        # Core domains with 15 records each (rapid EOF)
        core_records = 15
        for d in CORE_ANCHOR_DOMAINS:
            sp = self._create_mock_shard(d, "shard_000.jsonl", core_records)
            domain_shards[d] = [sp]

        # Non-core domain (code) with 3,000 records (long-running stream)
        code_domain = "02_computer_science_systems"
        code_sp = self._create_mock_shard(code_domain, "shard_000.jsonl", 3000)
        domain_shards[code_domain] = [code_sp]

        # Target anchor ratio: 0.175 (17.5%, centered between 15% and 20%)
        target_anchor_ratio = 0.175
        scheduler = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            domain_weights={d: 0.2 for d in domain_shards},
            tokenizer=self.tokenizer,
            max_seq_len=64,
            max_samples=1200,
            anchor_ratio=target_anchor_ratio,
        )

        domain_counts = {d: 0 for d in domain_shards}
        anchor_counts = {d: 0 for d in domain_shards}
        total_batches = 0
        total_anchor_batches = 0

        for item in scheduler:
            total_batches += 1
            d = item.domain
            domain_counts[d] = domain_counts.get(d, 0) + 1
            if getattr(item, "is_anchor", False):
                total_anchor_batches += 1
                anchor_counts[d] = anchor_counts.get(d, 0) + 1

        print(f"\n[Stress Test Results] Total Batches: {total_batches}")
        print(f"[Stress Test Results] Total Anchor Batches: {total_anchor_batches}")
        empirical_ratio = total_anchor_batches / total_batches if total_batches > 0 else 0
        print(f"[Stress Test Results] Empirical Anchor Replay Ratio: {empirical_ratio:.4f} (Target: {target_anchor_ratio})")
        print(f"[Stress Test Results] Domain Counts: {domain_counts}")
        print(f"[Stress Test Results] Anchor Counts per Domain: {anchor_counts}")

        self.assertEqual(total_batches, 1200)

        # Statistical tolerance: For N=1200, p=0.175, standard deviation is sqrt(1200*0.175*0.825) = 13.16 batches (~1.1%)
        # 3-sigma range for ratio: [0.175 - 0.035, 0.175 + 0.035] = [0.140, 0.210]
        # Required window: 15%–20% (0.15 to 0.20), allowing ±0.025 statistical variation for Bernoulli sampling
        self.assertGreaterEqual(empirical_ratio, 0.13, f"Empirical anchor ratio {empirical_ratio:.4f} too low (< 13%)")
        self.assertLessEqual(empirical_ratio, 0.22, f"Empirical anchor ratio {empirical_ratio:.4f} too high (> 22%)")

        # Verify that all 4 depleted core domains provided anchors
        for cd in CORE_ANCHOR_DOMAINS:
            self.assertIn(cd, scheduler.anchor_buffer.depleted_domains, f"Domain {cd} was not marked depleted")
            self.assertGreater(anchor_counts[cd], 0, f"Depleted core domain {cd} had 0 anchor replay batches!")

    # =========================================================================
    # Suite 3: Memory Stability & RAM Leak Detection
    # =========================================================================

    def test_08_memory_stability_over_thousands_of_batches(self):
        """
        Verify memory stability: assert that CPU RAM does not grow monotonically
        during 3,000 streaming iterations (< 500 MB budget, ideally < 30 MB).
        """
        domain_shards = {}
        for d in ["01_mathematics_formal_logic", "02_computer_science_systems", "04_linguistics_multilingual"]:
            sp = self._create_mock_shard(d, "shard_000.jsonl", 1500)
            domain_shards[d] = [sp]

        scheduler = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            tokenizer=self.tokenizer,
            max_seq_len=64,
            max_samples=3000,
            anchor_ratio=0.15,
        )

        tracemalloc.start()
        memory_checkpoints = []

        batch_idx = 0
        for item in scheduler:
            batch_idx += 1
            if batch_idx % 500 == 0:
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                memory_checkpoints.append((batch_idx, current_mem / (1024 * 1024), peak_mem / (1024 * 1024)))

        current_final, peak_final = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\n[Memory Stability Checkpoints (Batch, Current MB, Peak MB)]:")
        for cp in memory_checkpoints:
            print(f"  Step {cp[0]:4d}: Current {cp[1]:.2f} MB, Peak {cp[2]:.2f} MB")
        print(f"Final Peak: {peak_final / (1024*1024):.2f} MB")

        # Verify Peak Memory is well within 500 MB Free Colab budget
        peak_mb = peak_final / (1024 * 1024)
        self.assertLess(peak_mb, 150.0, f"Peak memory {peak_mb:.2f} MB exceeded 150 MB safety threshold (budget is 500 MB)")

        # Verify no runaway slope between step 1000 and 3000
        mem_at_1000 = memory_checkpoints[1][1]  # 1000
        mem_at_3000 = memory_checkpoints[-1][1]  # 3000
        growth_mb = mem_at_3000 - mem_at_1000
        print(f"Memory growth between step 1000 and 3000: {growth_mb:.2f} MB")
        self.assertLess(growth_mb, 20.0, f"Memory growth {growth_mb:.2f} MB between step 1000 and 3000 indicates leak")

    def test_09_anchor_buffer_reservoir_capacity_bound(self):
        """
        Verify AnchorReplayBuffer strictly bounds capacity per domain to capacity_per_domain (1500)
        even when bombarded with 50,000 samples.
        """
        buf = AnchorReplayBuffer(capacity_per_domain=1500)
        target_domain = "12_planning_reasoning_traces"

        for i in range(50000):
            mock_item = TaraDomainBatchItem(
                torch.tensor([i, i + 1, i + 2]), torch.tensor([i, i + 1, i + 2]),
                "shard_001.jsonl", i, i * 100, 0, False, domain=target_domain
            )
            buf.add_sample(target_domain, mock_item)

        self.assertEqual(len(buf.buffers[target_domain]), 1500)
        self.assertEqual(buf.counts[target_domain], 50000)

        # Sample 1000 anchors and verify all are from valid domain
        sampled = [buf.sample_anchor() for _ in range(1000)]
        self.assertEqual(len(sampled), 1000)
        for s in sampled:
            self.assertEqual(s.domain, target_domain)

    # =========================================================================
    # Suite 4: Edge Case & Termination Boundary Stress
    # =========================================================================

    def test_10_complete_dataset_exhaustion_behavior(self):
        """
        Test what happens when ALL domains (both core and non-core) completely exhaust.
        Does iteration terminate cleanly, or does it hang/loop?
        """
        domain_shards = {}
        # 2 domains with only 5 records each, max_samples=0 (natural exhaustion)
        for d in ["01_mathematics_formal_logic", "02_computer_science_systems"]:
            sp = self._create_mock_shard(d, "shard_000.jsonl", 5)
            domain_shards[d] = [sp]

        scheduler = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            tokenizer=self.tokenizer,
            max_seq_len=64,
            max_samples=0,  # Run until natural exhaustion
            anchor_ratio=0.15,
        )

        # We set an explicit safety ceiling to detect infinite loop
        count = 0
        max_safety_iterations = 200
        start_t = time.time()

        for item in scheduler:
            count += 1
            if count >= max_safety_iterations:
                break
            if time.time() - start_t > 5.0:
                break

        print(f"\n[Exhaustion Test] Total items yielded from 10 total records (max_samples=0): {count}")
        # Note: If count reached max_safety_iterations, scheduler did NOT terminate when active_candidates depleted!
        # It kept yielding anchor items infinitely because anchor_buffer.has_anchors() is True!
        print(f"[Exhaustion Test] Terminated naturally? {count < max_safety_iterations}")
        self.assertLess(count, max_safety_iterations, f"Scheduler failed to terminate on total exhaustion: yielded {count} items")
        self.assertGreaterEqual(count, 10, "Scheduler should have yielded at least the 10 available records before termination")

    def test_11_handover_resumption_anchor_loss(self):
        """
        Adversarial test: verify whether anchor exemplars survive get_state() -> load_state() handover.
        When Colab Account 1 saves state and Account 2 resumes, are anchor buffers preserved,
        or are they silently lost, causing has_anchors() to be False and anchor replay to drop to 0%?
        """
        domain_shards = {}
        math_domain = "01_mathematics_formal_logic"
        code_domain = "02_computer_science_systems"
        domain_shards[math_domain] = [self._create_mock_shard(math_domain, "shard_000.jsonl", 10)]
        domain_shards[code_domain] = [self._create_mock_shard(code_domain, "shard_000.jsonl", 500)]

        scheduler1 = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            tokenizer=self.tokenizer,
            max_seq_len=64,
            max_samples=50,
            anchor_ratio=0.15,
        )

        # Stream 50 batches to ensure math is depleted and cached into anchor_buffer
        batches_s1 = list(scheduler1)
        self.assertIn(math_domain, scheduler1.anchor_buffer.depleted_domains)
        cached_count_s1 = len(scheduler1.anchor_buffer.buffers.get(math_domain, []))
        self.assertGreater(cached_count_s1, 0, "Scheduler 1 should have cached anchor exemplars for math")
        self.assertTrue(scheduler1.anchor_buffer.has_anchors(), "Scheduler 1 has_anchors() must be True")

        # Snapshot state (simulating GDrive checkpoint save)
        state = scheduler1.get_state()
        print(f"\n[Handover Test] Snapshot state keys: {list(state.keys())}")
        print(f"[Handover Test] Depleted domains in state: {state.get('depleted_domains')}")

        # Instantiate Scheduler 2 (simulating Account 2 resume from GDrive)
        scheduler2 = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            tokenizer=self.tokenizer,
            max_seq_len=64,
            max_samples=100,
            anchor_ratio=0.15,
        )
        scheduler2.load_state(state)

        # Empirically check anchor buffer status in Scheduler 2
        cached_count_s2 = len(scheduler2.anchor_buffer.buffers.get(math_domain, []))
        has_anchors_s2 = scheduler2.anchor_buffer.has_anchors()
        print(f"[Handover Test] Scheduler 2 cached anchors for '{math_domain}': {cached_count_s2}")
        print(f"[Handover Test] Scheduler 2 has_anchors(): {has_anchors_s2}")

        # Stream from Scheduler 2 and count anchor batches
        replayed_anchors_s2 = 0
        for item in scheduler2:
            if getattr(item, "is_anchor", False):
                replayed_anchors_s2 += 1

        print(f"[Handover Test] Replayed anchors yielded in Scheduler 2: {replayed_anchors_s2}")
        # Explicit verification assertions
        self.assertGreater(cached_count_s2, 0, f"Scheduler 2 has 0 cached anchors for {math_domain} after load_state")
        self.assertTrue(has_anchors_s2, "Scheduler 2 has_anchors() must be True after load_state")
        self.assertGreater(replayed_anchors_s2, 0, "Scheduler 2 must yield replayed anchor batches during resumed stream")

    def test_12_anchor_replay_statistical_ratios(self):
        """
        Adversarial test: evaluate anchor replay ratios at 15% and 20% over 1,500 batches each.
        Empirically verify adherence to the 15%–20% specification.
        """
        for target_ratio in [0.15, 0.20]:
            domain_shards = {}
            for cd in CORE_ANCHOR_DOMAINS:
                domain_shards[cd] = [self._create_mock_shard(cd, f"shard_{target_ratio}.jsonl", 10)]
            code_d = "02_computer_science_systems"
            domain_shards[code_d] = [self._create_mock_shard(code_d, f"shard_{target_ratio}.jsonl", 3000)]

            scheduler = TaraMultiDomainScheduler(
                domain_shards=domain_shards,
                tokenizer=self.tokenizer,
                max_seq_len=64,
                max_samples=1500,
                anchor_ratio=target_ratio,
            )

            anchor_count = 0
            total_count = 0
            for item in scheduler:
                total_count += 1
                if getattr(item, "is_anchor", False):
                    anchor_count += 1

            emp_ratio = anchor_count / total_count
            print(f"\n[Ratio Test] Target: {target_ratio:.2f} -> Observed: {emp_ratio:.4f} ({anchor_count}/{total_count})")
            # Tolerance: ±0.035 for 1500 samples
            self.assertGreaterEqual(emp_ratio, target_ratio - 0.035)
            self.assertLessEqual(emp_ratio, target_ratio + 0.035)


if __name__ == "__main__":
    unittest.main()

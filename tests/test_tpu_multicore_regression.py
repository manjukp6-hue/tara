#!/usr/bin/env python3
"""
=============================================================================
TARA TPU Multi-Core Collective Skip Synchronization Regression Test Suite
=============================================================================
Simulates multiple TPU workers (Worker 0 & Worker 1) executing distributed
optimizer steps to verify:
  1. If Worker 1 detects a loss shock (>10x EMA), ALL workers receive skip=True.
  2. If Worker 1 detects a non-finite loss (NaN / Inf), ALL workers receive skip=True.
  3. No worker updates model parameters or optimizer state on a skipped update.
  4. Both workers resume valid parameter updates together on the next valid step.
  5. Both workers perform normal updates when all report valid losses.
  6. Workers remain strictly synchronized with zero weight divergence.

Execution Mode:
  - CPU Emulation with Simulated Collective Coordinator (Thread-Safe Rendezvous)
  - Distinguishes CPU mocked collective testing from physical TPU hardware.
=============================================================================
"""

import os
import sys
import math
import copy
import threading
from typing import Dict, List, Tuple, Any, Callable

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from tpu_config import (
    tpu_optimizer_step,
    sync_tpu_decision,
    set_collective_mock,
    HAS_XLA,
    build_tpu_optimizer_groups,
)
from model_dataset import TaraForCausalLM


class SimulatedTPUCollectiveCoordinator:
    """
    Simulates the PyTorch/XLA xm.mesh_reduce collective reduction and xm.reduce_gradients
    across N simulated TPU workers. Ensures thread-safe, synchronous rendezvous across workers.
    """
    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._payloads: Dict[str, List[int]] = {}
        self._results: Dict[str, int] = {}
        self._grads: Dict[str, List[List[Optional[torch.Tensor]]]] = {}
        self._reduced_grads: Dict[str, List[Optional[torch.Tensor]]] = {}

    def mesh_reduce(self, value: int, tag: str) -> int:
        with self._lock:
            if tag not in self._payloads:
                self._payloads[tag] = []

            self._payloads[tag].append(value)

            if len(self._payloads[tag]) == self.world_size:
                reduced = max(self._payloads[tag])
                self._results[tag] = int(reduced)
                self._cond.notify_all()
            else:
                while tag not in self._results:
                    self._cond.wait(timeout=5.0)

            return self._results.get(tag, value)

    def reduce_gradients(self, model: nn.Module, step: int):
        tag = f"grad_reduce_{step}"
        with self._lock:
            if tag not in self._grads:
                self._grads[tag] = []
            grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            self._grads[tag].append(grads)

            if len(self._grads[tag]) == self.world_size:
                avg_grads = []
                for p_idx in range(len(grads)):
                    if grads[p_idx] is not None:
                        total = sum(self._grads[tag][w_idx][p_idx] for w_idx in range(self.world_size))
                        avg_grads.append(total / float(self.world_size))
                    else:
                        avg_grads.append(None)
                self._reduced_grads[tag] = avg_grads
                self._cond.notify_all()
            else:
                while tag not in self._reduced_grads:
                    self._cond.wait(timeout=5.0)

            for p, g in zip(model.parameters(), self._reduced_grads[tag]):
                if g is not None:
                    p.grad.copy_(g)

    def reset(self):
        with self._lock:
            self._payloads.clear()
            self._results.clear()
            self._grads.clear()
            self._reduced_grads.clear()


def create_micro_test_model():
    """Instantiates a lightweight 2-layer micro TaraForCausalLM for unit testing."""
    config = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }
    model = TaraForCausalLM(config)
    return model, config


def run_multi_worker_step(
    coordinator: SimulatedTPUCollectiveCoordinator,
    worker_inputs: List[Dict[str, Any]],
    step: int,
) -> List[Dict[str, Any]]:
    """
    Executes an optimizer step concurrently across simulated TPU workers.
    Each worker runs in its own thread to verify parallel collective synchronization.
    """
    results: List[Dict[str, Any]] = [{} for _ in worker_inputs]

    def _worker_fn(rank: int, worker_data: Dict[str, Any]):
        model = worker_data["model"]
        optimizer = worker_data["optimizer"]
        cur_loss = worker_data["cur_loss"]
        ema_loss = worker_data["ema_loss"]

        # Capture weight snapshot before step
        weights_before = [p.clone().detach() for p in model.parameters()]

        # Generate fake synthetic gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.01

        # Execute optimizer step
        is_skipped, grad_norm = tpu_optimizer_step(
            optimizer=optimizer,
            model=model,
            max_norm=1.0,
            cur_loss=cur_loss,
            ema_loss=ema_loss,
            step=step,
        )

        # Capture weight snapshot after step
        weights_after = [p.clone().detach() for p in model.parameters()]

        # Check if parameters actually changed
        weights_changed = any(not torch.equal(b, a) for b, a in zip(weights_before, weights_after))

        results[rank] = {
            "rank": rank,
            "is_skipped": is_skipped,
            "grad_norm": grad_norm,
            "weights_changed": weights_changed,
            "weights_after": weights_after,
        }

    threads = []
    for rank, w_data in enumerate(worker_inputs):
        t = threading.Thread(target=_worker_fn, args=(rank, w_data))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    return results


def run_all_multicore_regression_tests() -> bool:
    print("=" * 80)
    print("  TARA TPU MULTI-CORE SKIP SYNCHRONIZATION REGRESSION TEST SUITE")
    print("=" * 80)
    print(f"  Physical PyTorch/XLA Installed : {HAS_XLA}")
    print("  Execution Mode                 : Simulated Multi-Worker Collective Coordinator (2 Workers)")
    print("  Hardware Environment           : CPU Emulation (Physical TPU Chips NOT Attached)")
    print("  Collective Synchronization Hook: sync_tpu_decision -> Coordinator.mesh_reduce")
    print("=" * 80 + "\n")

    coordinator = SimulatedTPUCollectiveCoordinator(world_size=2)
    set_collective_mock(
        hook=lambda val, tag: coordinator.mesh_reduce(val, tag),
        grad_hook=lambda model, step: coordinator.reduce_gradients(model, step),
    )

    all_passed = True

    # Build 2 identical models with exact same initial weights
    base_model, cfg = create_micro_test_model()
    model_w0 = copy.deepcopy(base_model)
    model_w1 = copy.deepcopy(base_model)

    opt_w0 = torch.optim.AdamW(build_tpu_optimizer_groups(model_w0), lr=1e-3)
    opt_w1 = torch.optim.AdamW(build_tpu_optimizer_groups(model_w1), lr=1e-3)

    # -------------------------------------------------------------------------
    # TEST 1: Worker 0 Normal Loss vs Worker 1 Shock Loss (>10x EMA)
    # -------------------------------------------------------------------------
    print("[TEST 1/5] Testing Multi-Worker Loss Shock Divergence Prevention...")
    coordinator.reset()
    w_inputs = [
        {"model": model_w0, "optimizer": opt_w0, "cur_loss": 1.5, "ema_loss": 1.5},
        {"model": model_w1, "optimizer": opt_w1, "cur_loss": 25.0, "ema_loss": 1.5},  # Shock loss: 25.0 > 15.0 and > 10*1.5
    ]
    res = run_multi_worker_step(coordinator, w_inputs, step=1)

    try:
        assert res[0]["is_skipped"] is True, "Worker 0 failed to skip step despite Worker 1 experiencing shock loss!"
        assert res[1]["is_skipped"] is True, "Worker 1 failed to report skip!"
        assert not res[0]["weights_changed"], "Worker 0 mutated weights on a skipped step!"
        assert not res[1]["weights_changed"], "Worker 1 mutated weights on a skipped step!"

        # Verify weights remain identical across workers
        for p0, p1 in zip(model_w0.parameters(), model_w1.parameters()):
            assert torch.equal(p0, p1), "Worker 0 and Worker 1 weights diverged after shock skip!"

        print("  -> Worker 0: Loss=1.5 (Normal) -> Decision: SKIPPED (Weights Unchanged)")
        print("  -> Worker 1: Loss=25.0 (Shock) -> Decision: SKIPPED (Weights Unchanged)")
        print("  -> Collective sync verified: Both workers reached identical skip decision.")
        print("  [PASS] Test 1: Multi-worker loss-shock divergence prevented.")
    except Exception as e:
        print(f"  [FAIL] Test 1: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 2: Worker 0 Normal Loss vs Worker 1 Non-Finite Loss (NaN)
    # -------------------------------------------------------------------------
    print("\n[TEST 2/5] Testing Multi-Worker Non-Finite (NaN) Loss Divergence Prevention...")
    coordinator.reset()
    w_inputs = [
        {"model": model_w0, "optimizer": opt_w0, "cur_loss": 1.5, "ema_loss": 1.5},
        {"model": model_w1, "optimizer": opt_w1, "cur_loss": float("nan"), "ema_loss": 1.5},
    ]
    res = run_multi_worker_step(coordinator, w_inputs, step=2)

    try:
        assert res[0]["is_skipped"] is True, "Worker 0 did not skip on Worker 1 NaN loss!"
        assert res[1]["is_skipped"] is True, "Worker 1 did not skip on NaN loss!"
        assert not res[0]["weights_changed"], "Worker 0 updated weights on NaN step!"
        assert not res[1]["weights_changed"], "Worker 1 updated weights on NaN step!"

        for p0, p1 in zip(model_w0.parameters(), model_w1.parameters()):
            assert torch.equal(p0, p1), "Weights diverged on NaN skip!"

        print("  -> Worker 0: Loss=1.5 (Normal) -> Decision: SKIPPED")
        print("  -> Worker 1: Loss=NaN (Corrupted) -> Decision: SKIPPED")
        print("  -> Collective sync verified: NaN safely caught without averaging NaN values.")
        print("  [PASS] Test 2: Multi-worker NaN loss divergence prevented.")
    except Exception as e:
        print(f"  [FAIL] Test 2: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 3: Worker 0 Normal Loss vs Worker 1 Non-Finite Loss (Inf)
    # -------------------------------------------------------------------------
    print("\n[TEST 3/5] Testing Multi-Worker Non-Finite (Inf) Loss Divergence Prevention...")
    coordinator.reset()
    w_inputs = [
        {"model": model_w0, "optimizer": opt_w0, "cur_loss": 1.5, "ema_loss": 1.5},
        {"model": model_w1, "optimizer": opt_w1, "cur_loss": float("inf"), "ema_loss": 1.5},
    ]
    res = run_multi_worker_step(coordinator, w_inputs, step=3)

    try:
        assert res[0]["is_skipped"] is True, "Worker 0 did not skip on Worker 1 Inf loss!"
        assert res[1]["is_skipped"] is True, "Worker 1 did not skip on Inf loss!"
        assert not res[0]["weights_changed"], "Worker 0 updated weights on Inf step!"
        assert not res[1]["weights_changed"], "Worker 1 updated weights on Inf step!"

        for p0, p1 in zip(model_w0.parameters(), model_w1.parameters()):
            assert torch.equal(p0, p1), "Weights diverged on Inf skip!"

        print("  -> Worker 0: Loss=1.5 (Normal) -> Decision: SKIPPED")
        print("  -> Worker 1: Loss=Inf (Corrupted) -> Decision: SKIPPED")
        print("  [PASS] Test 3: Multi-worker Inf loss divergence prevented.")
    except Exception as e:
        print(f"  [FAIL] Test 3: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 4: Immediate Step Recovery (All Workers Normal Loss After Skip)
    # -------------------------------------------------------------------------
    print("\n[TEST 4/5] Testing Immediate Step Recovery on Next Valid Batch...")
    coordinator.reset()
    w_inputs = [
        {"model": model_w0, "optimizer": opt_w0, "cur_loss": 1.45, "ema_loss": 1.5},
        {"model": model_w1, "optimizer": opt_w1, "cur_loss": 1.48, "ema_loss": 1.5},
    ]
    res = run_multi_worker_step(coordinator, w_inputs, step=4)

    try:
        assert res[0]["is_skipped"] is False, "Worker 0 skipped valid recovery step!"
        assert res[1]["is_skipped"] is False, "Worker 1 skipped valid recovery step!"
        assert res[0]["weights_changed"] is True, "Worker 0 failed to update weights on recovery step!"
        assert res[1]["weights_changed"] is True, "Worker 1 failed to update weights on recovery step!"

        for p0, p1 in zip(model_w0.parameters(), model_w1.parameters()):
            assert torch.equal(p0, p1), "Worker 0 and Worker 1 weights diverged after recovery step!"

        print("  -> Worker 0: Loss=1.45 (Valid) -> Decision: UPDATED (Weights Mutated)")
        print("  -> Worker 1: Loss=1.48 (Valid) -> Decision: UPDATED (Weights Mutated)")
        print("  -> Confirmed: Both workers resumed updates together with 100% synchronized weights.")
        print("  [PASS] Test 4: Immediate step recovery verified.")
    except Exception as e:
        print(f"  [FAIL] Test 4: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 5: Clean Multi-Worker Normal Steps
    # -------------------------------------------------------------------------
    print("\n[TEST 5/5] Testing Clean Normal Updates Across Workers...")
    coordinator.reset()
    w_inputs = [
        {"model": model_w0, "optimizer": opt_w0, "cur_loss": 1.40, "ema_loss": 1.45},
        {"model": model_w1, "optimizer": opt_w1, "cur_loss": 1.42, "ema_loss": 1.45},
    ]
    res = run_multi_worker_step(coordinator, w_inputs, step=5)

    try:
        assert res[0]["is_skipped"] is False and res[1]["is_skipped"] is False
        assert res[0]["weights_changed"] is True and res[1]["weights_changed"] is True
        for p0, p1 in zip(model_w0.parameters(), model_w1.parameters()):
            assert torch.equal(p0, p1)

        print("  -> Both workers performed clean updates and remain 100% in lockstep.")
        print("  [PASS] Test 5: Clean multi-worker updates verified.")
    except Exception as e:
        print(f"  [FAIL] Test 5: {e}")
        all_passed = False

    # Clean up collective mock
    set_collective_mock(None)

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL 5 MULTI-CORE REGRESSION TESTS PASSED! (COLLECTIVE SYNC VERIFIED)")
    else:
        print("  MULTI-CORE REGRESSION TESTS ENCOUNTERED FAILURES.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_multicore_regression_tests()
    sys.exit(0 if success else 1)

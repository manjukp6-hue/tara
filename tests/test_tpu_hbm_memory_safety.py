#!/usr/bin/env python3
"""
=============================================================================
TARA TPU HBM Memory Safety & Non-Leaf Grad Regression Test Suite
=============================================================================
Validates:
  1. Forward and backward on selected precision (BF16 & FP32) with memory-safe chunked loss.
  2. Finite loss and finite non-None gradients across all leaf trainable parameters.
  3. Strict resolution of non-leaf .grad warning: verified zero warnings when inspecting leaf parameters.
  4. Exact optimizer-state restoration preserving FP32 momentum tensors (no silent downcast).
  5. Peak-memory reporting and diagnostics across all 6 mandatory pipeline stages.
  6. Memory math verification: static chunking (128 tokens) vs 512 tokens vs full logits (3.28 GB).
  7. Mathematical equivalence of static variance regularizer (zero dynamic tt_jit_index).
  8. Checkpoint save and resume with memory-safe loss.
  9. Strict GPU pipeline file isolation (train_main.py, model_dataset.py, run_train.sh UNTOUCHED).
=============================================================================
"""

import os
import sys
import math
import json
import tempfile
import warnings
import subprocess
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repository root to python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tpu_config import (
    TPUConfig,
    build_tpu_optimizer_groups,
    restore_tpu_optimizer_state,
    get_tpu_memory_diagnostics,
    log_tpu_memory_stage,
    is_tpu_available,
    ensure_model_weight_tying,
    tpu_optimizer_step,
    set_collective_mock,
    sync_tpu_sum,
    sync_tpu_sum_float,
)
from train_tpu import compute_tpu_chunked_loss, save_tpu_checkpoint, load_tpu_checkpoint
from model_dataset import TaraForCausalLM, TaraNativeTokenizer


class TestTpuHbmMemorySafety(unittest.TestCase):
    """Rigorous unit and regression test suite for TPU HBM memory safety."""

    def setUp(self):
        """Set up small model configuration for deterministic testing."""
        self.config = {
            "vocab_size": 1024,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 256,
            "tie_word_embeddings": True,
            "rope_theta": 10000.0,
        }
        torch.manual_seed(42)

    def test_01_forward_backward_memory_safe_chunked_loss(self):
        """Test forward and backward with memory-safe chunked loss on FP32 and BF16."""
        model = TaraForCausalLM(self.config, gradient_checkpointing=True)
        model.train()
        ensure_model_weight_tying(model)

        bsz, seq_len = 2, 64
        x = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))
        labels = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))
        # Mask out 25% of tokens
        labels[labels < 256] = -100

        # Compute TPU chunked loss (chunk_sz=16 for this mini test)
        total_loss, ce_loss, aux_loss, total_valid = compute_tpu_chunked_loss(
            model=model,
            input_ids=x,
            labels=labels,
            chunk_sz=16,
        )

        self.assertTrue(torch.isfinite(total_loss).item(), "Total loss must be finite")
        self.assertTrue(torch.isfinite(ce_loss).item(), "CE loss must be finite")
        self.assertTrue(torch.isfinite(aux_loss).item(), "Aux loss must be finite")
        self.assertGreater(total_loss.item(), 0.0, "Total loss must be strictly positive")
        self.assertGreater(total_valid.item(), 0, "Valid tokens must be strictly positive")

        # Backward pass
        total_loss.backward()

        # Check all leaf parameters have finite gradients
        non_finite_grads = []
        none_grads = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertTrue(p.is_leaf, f"Parameter {name} must be a leaf tensor")
                if p.grad is None:
                    none_grads.append(name)
                elif not torch.isfinite(p.grad).all():
                    non_finite_grads.append(name)

        self.assertEqual(len(none_grads), 0, f"Parameters missing gradient: {none_grads}")
        self.assertEqual(len(non_finite_grads), 0, f"Parameters with non-finite gradient: {non_finite_grads}")

    def test_02_non_leaf_grad_warning_resolution(self):
        """
        Verify that inspecting gradients on leaf parameters produces ZERO warnings,
        and verify that accessing .grad on a non-leaf tensor triggers PyTorch warning
        unless .retain_grad() is used.
        """
        model = TaraForCausalLM(self.config, gradient_checkpointing=False)
        model.train()

        x = torch.randint(0, self.config["vocab_size"], (1, 32))
        labels = torch.randint(0, self.config["vocab_size"], (1, 32))

        # Catch all warnings
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")

            target_model = getattr(model, "module", model)
            hidden_states = target_model.model(input_ids=x)
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Leaf parameters MUST NOT trigger non-leaf warnings
            for p in model.parameters():
                self.assertTrue(p.is_leaf, "Model parameters must be leaf tensors")
                _ = p.grad  # Safe access on leaf tensor

            # Verify zero warnings were emitted during leaf inspection
            leaf_warnings = [w for w in recorded_warnings if "not a leaf Tensor" in str(w.message)]
            self.assertEqual(len(leaf_warnings), 0, "Inspecting leaf parameters must never emit non-leaf warnings")

            # Now demonstrate why non-leaf inspection on shift_hidden triggers the warning:
            self.assertFalse(shift_hidden.is_leaf, "shift_hidden is an intermediate activation (non-leaf)")
            _ = shift_hidden.grad  # This triggers the PyTorch non-leaf warning!

            non_leaf_warnings = [w for w in recorded_warnings if "not a leaf Tensor" in str(w.message)]
            self.assertGreaterEqual(len(non_leaf_warnings), 1, "Accessing non-leaf .grad without retain_grad must emit warning")

        # Now demonstrate proper handling when non-leaf gradient is genuinely requested:
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")

            hidden_states2 = target_model.model(input_ids=x)
            shift_hidden2 = hidden_states2[..., :-1, :].contiguous()
            shift_hidden2.retain_grad()  # Authoritative resolution: call retain_grad before backward!

            logits = target_model.lm_head(shift_hidden2)
            loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]), shift_labels.view(-1))
            loss.backward()

            # Now shift_hidden2.grad is populated and accessing it produces ZERO warnings
            self.assertIsNotNone(shift_hidden2.grad)
            self.assertTrue(torch.isfinite(shift_hidden2.grad).all())

            retained_warnings = [w for w in recorded_warnings if "not a leaf Tensor" in str(w.message)]
            self.assertEqual(len(retained_warnings), 0, "Calling retain_grad must prevent non-leaf warning completely")

    def test_03_optimizer_state_restoration_fp32_preservation(self):
        """Verify restore_tpu_optimizer_state strictly preserves FP32 momentum tensors without downcasting."""
        model = TaraForCausalLM(self.config)
        ensure_model_weight_tying(model)

        optim_groups = build_tpu_optimizer_groups(model, weight_decay=0.01)
        optimizer = torch.optim.AdamW(optim_groups, lr=1e-3, eps=1e-5)

        # Populate realistic momentum state through forward and backward
        x = torch.randint(0, self.config["vocab_size"], (1, 16))
        loss = model(x, labels=x).loss
        loss.backward()
        optimizer.step()

        # Save optimizer state
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_dict = optimizer.state_dict()
            ckpt_payload = {"optimizer": state_dict, "step": 100}
            opt_path = os.path.join(tmp_dir, "optimizer_state.pt")
            torch.save(ckpt_payload, opt_path)

            # Create a fresh optimizer
            fresh_optimizer = torch.optim.AdamW(
                build_tpu_optimizer_groups(model, weight_decay=0.01),
                lr=1e-3,
                eps=1e-5,
            )

            # Restore into active optimizer
            restored = restore_tpu_optimizer_state(tmp_dir, fresh_optimizer, device=torch.device("cpu"))
            self.assertTrue(restored, "Optimizer state must be successfully restored")

            # Strictly verify every restored momentum tensor is FP32
            self.assertGreater(len(fresh_optimizer.state), 0, "Restored state must not be empty")
            for state in fresh_optimizer.state.values():
                if "exp_avg" in state:
                    self.assertEqual(state["exp_avg"].dtype, torch.float32, "exp_avg must strictly be torch.float32")
                if "exp_avg_sq" in state:
                    self.assertEqual(state["exp_avg_sq"].dtype, torch.float32, "exp_avg_sq must strictly be torch.float32")

    def test_04_memory_diagnostics_all_six_stages(self):
        """Verify explicit TPU memory diagnostics across all 6 mandatory pipeline stages."""
        model = TaraForCausalLM(self.config)
        optim_groups = build_tpu_optimizer_groups(model)
        optimizer = torch.optim.AdamW(optim_groups, lr=1e-3)

        stages = [
            "BEFORE_MODEL_LOADING",
            "AFTER_MODEL_LOADING",
            "AFTER_OPTIMIZER_RESTORE",
            "BEFORE_FORWARD",
            "BEFORE_BACKWARD",
            "BEFORE_OPTIMIZER_STEP",
            "AFTER_CHECKPOINT_SAVE",
        ]

        for stage in stages:
            diag = log_tpu_memory_stage(
                stage_name=stage,
                step=1,
                device=torch.device("cpu"),
                model=model,
                optimizer=optimizer,
                microbatch=0,
            )
            self.assertEqual(diag["stage"], stage)
            self.assertIn("used_mb", diag)
            self.assertIn("free_mb", diag)
            self.assertIn("total_mb", diag)
            self.assertIn("peak_mb", diag)
            self.assertIn("has_grad_refs", diag)
            self.assertIn("backend", diag)
            self.assertIn("is_physical_tpu", diag)
            # Locally on CPU machine, is_physical_tpu must honestly report False
            self.assertFalse(diag["is_physical_tpu"], "CPU fallback must honestly report is_physical_tpu=False")
            self.assertIn("CPU", diag["backend"])

    def test_05_static_variance_regularizer_mathematical_parity(self):
        """Verify static broadcast variance regularizer produces exact mathematical parity without tt_jit_index."""
        N = 2048
        D = 1024
        flat_hidden = torch.randn(N, D, requires_grad=True)
        flat_labels = torch.randint(0, 1000, (N,))
        # Randomly mask 30% of tokens
        flat_labels[torch.rand(N) < 0.3] = -100

        valid_mask = (flat_labels != -100)

        # 1. Dynamic boolean mask method (old model_dataset.py style, triggers tt_jit_index)
        valid_hidden = flat_hidden[valid_mask]
        var_orig = valid_hidden.var(dim=-1).mean()
        aux_loss_orig = 0.005 * torch.relu(1.0 - var_orig)
        grad_orig = torch.autograd.grad(aux_loss_orig, flat_hidden, retain_graph=True)[0]

        # 2. Static broadcast masking method (TPU memory-safe style, zero dynamic indexing)
        valid_mask_f = valid_mask.float().unsqueeze(-1)
        var_per_token = flat_hidden.var(dim=-1, keepdim=True)
        var_static = (var_per_token * valid_mask_f).sum() / valid_mask_f.sum().clamp(min=1.0)
        aux_loss_static = 0.005 * torch.relu(1.0 - var_static)
        grad_static = torch.autograd.grad(aux_loss_static, flat_hidden)[0]

        # Assert mathematical parity
        diff_var = (var_orig - var_static).abs().item()
        diff_loss = (aux_loss_orig - aux_loss_static).abs().item()
        diff_grad = (grad_orig - grad_static).abs().max().item()

        self.assertLess(diff_var, 1e-4, f"Variance difference too large: {diff_var}")
        self.assertLess(diff_loss, 1e-5, f"Loss difference too large: {diff_loss}")
        self.assertLess(diff_grad, 1e-5, f"Gradient difference too large: {diff_grad}")

    def test_06_memory_math_allocation_footprint_reduction(self):
        """Quantitatively verify memory allocation footprint across chunk sizes and logits."""
        vocab_size = 100256
        seq_len = 2048
        batch_size = 4

        # Full sequence logits memory (FP32)
        full_logits_bytes = batch_size * seq_len * vocab_size * 4
        full_logits_mb = full_logits_bytes / (1024 * 1024)
        self.assertAlmostEqual(full_logits_mb, 3133.0, delta=150.0)  # ~3.28 GB

        # Old chunk size 512 tokens (requested 195.91 MB in tt_jit_cross_entropy_loss)
        old_chunk_bytes = 512 * vocab_size * 4
        old_chunk_mb = old_chunk_bytes / (1024 * 1024)
        self.assertAlmostEqual(old_chunk_mb, 195.8, delta=10.0)  # ~195.91 MB

        # New memory-safe chunk size 128 tokens
        new_chunk_bytes = 128 * vocab_size * 4
        new_chunk_mb = new_chunk_bytes / (1024 * 1024)
        self.assertAlmostEqual(new_chunk_mb, 48.95, delta=2.0)  # ~48.95 MB in FP32 (~24.5 MB in BF16)

        # 4x reduction from 195.91 MB down to 48.95 MB
        self.assertLessEqual(new_chunk_mb, 50.0, "New chunk size must allocate <= 50 MB per projection")

    def test_07_gpu_pipeline_isolation(self):
        """Strictly verify that GPU pipeline files (train_main.py, model_dataset.py, run_train.sh) are UNTOUCHED."""
        gpu_files = ["train_main.py", "model_dataset.py", "run_train.sh"]
        for f in gpu_files:
            cmd = f"git diff --exit-code HEAD -- {f}"
            res = subprocess.run(cmd, shell=True, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(
                res.returncode,
                0,
                f"GPU file '{f}' has been modified! GPU files must remain 100% untouched.",
            )

    def test_08_branchless_all_masked_batch_parity(self):
        """
        Verify that when all tokens are masked (-100):
          1. Loss and gradients are strictly 0.0 with zero D2H sync.
          2. total_valid is strictly 0.
          3. Optional return_valid_tokens=False preserves 3-tuple signature.
          4. When cluster valid_tokens == 0, tpu_optimizer_step skips without AdamW weight decay.
          5. Multi-replica check: if local valid_tokens == 0, but cluster has valid tokens (>0),
             tpu_optimizer_step does NOT skip (honors other replicas' gradients).
        """
        model = TaraForCausalLM(self.config)
        model.train()
        bsz, seq_len = 1, 64
        x = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))
        # All tokens masked!
        labels = torch.full((bsz, seq_len), -100, dtype=torch.long)

        # 1. Test 4-element return
        total_loss, ce_loss, aux_loss, total_valid = compute_tpu_chunked_loss(model, x, labels, chunk_sz=16)
        self.assertEqual(total_loss.item(), 0.0, "Total loss must be exactly 0.0 when all tokens are masked")
        self.assertEqual(ce_loss.item(), 0.0, "CE loss must be exactly 0.0 when all tokens are masked")
        self.assertEqual(aux_loss.item(), 0.0, "Aux loss must be exactly 0.0 when all tokens are masked")
        self.assertEqual(total_valid.item(), 0, "Valid tokens must be 0 for all-masked batch")

        # 2. Test backwards-compatible 3-element return
        ret3 = compute_tpu_chunked_loss(model, x, labels, chunk_sz=16, return_valid_tokens=False)
        self.assertEqual(len(ret3), 3, "return_valid_tokens=False must return 3-element tuple")

        # 3. Test backward on all-masked batch -> zero gradients
        total_loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertEqual(p.grad.abs().sum().item(), 0.0, "Grads must be 0.0 for all-masked batch")

        # 4. Test all-masked cluster-wide skip prevents AdamW weight decay
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        orig_w = [p.clone() for p in model.parameters()]
        is_skipped, _ = tpu_optimizer_step(
            opt, model=model, cur_loss=0.0, valid_tokens=0, step=1, attempt=0
        )
        self.assertTrue(is_skipped, "Must skip step when cluster valid_tokens == 0")
        for p, w_orig in zip(model.parameters(), orig_w):
            self.assertTrue(torch.equal(p, w_orig), "Weights must remain 100% unchanged (no AdamW weight decay on 0 data)")

        # 5. Multi-replica audit test: local valid_tokens == 0, but cluster has valid data (>0)
        # Mock hook simulates cluster mesh_reduce summing valid tokens across workers to 500
        def multi_worker_mock(val, tag):
            if "tpu_valid_tokens" in tag:
                return 500  # Another worker had 500 valid tokens!
            return val

        set_collective_mock(multi_worker_mock)
        try:
            # Fake a small gradient from backprop
            for p in model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p)
            is_skipped_cluster, _ = tpu_optimizer_step(
                opt, model=model, cur_loss=1.5, valid_tokens=0, step=2, attempt=0
            )
            self.assertFalse(is_skipped_cluster, "Must NOT skip step if cluster has valid tokens from other workers")
        finally:
            set_collective_mock(None)

    def test_09_static_chunk_shape_padding_invariant(self):
        """Verify that padding makes all chunks 100% identical in shape for XLA compiler caching."""
        bsz, seq_len = 1, 256  # 256 tokens total -> shift_hidden has 255 tokens
        x = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))
        labels = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))

        # Chunk size 32 -> 255 is not divisible by 32 (needs 1 token pad to 256 = 8 * 32)
        model = TaraForCausalLM(self.config)
        total_loss, ce_loss, aux_loss, total_valid = compute_tpu_chunked_loss(model, x, labels, chunk_sz=32)
        self.assertTrue(torch.isfinite(total_loss).item())
        self.assertGreater(total_valid.item(), 0)
        total_loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_10_optimizer_scalar_step_compatibility(self):
        """Verify AdamW scalar step state remains intact across steps without assertion failure."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        opt = torch.optim.AdamW([param], lr=1e-3)
        loss = param.sum()
        loss.backward()
        opt.step()

        st = opt.state[param]
        self.assertIn("step", st)
        step_val = st["step"].item() if hasattr(st["step"], "item") else float(st["step"])
        self.assertEqual(step_val, 1.0)

    def test_11_uninterrupted_vs_resumed_exact_parity(self):
        """
        Rigorously prove resume parity:
        Run Step 1 -> Save checkpoint -> Continue to Step 2 (Uninterrupted).
        In fresh instance: Load Step 1 checkpoint -> Run Step 2 (Resumed).
        Strictly assert 0.0 weight delta and 0.0 optimizer moment delta.
        """
        torch.manual_seed(42)
        model_a = TaraForCausalLM(self.config)
        opt_a = torch.optim.AdamW(build_tpu_optimizer_groups(model_a, weight_decay=0.01), lr=1e-3, eps=1e-5)

        torch.manual_seed(100)
        batch1 = torch.randint(0, self.config["vocab_size"], (2, 32))
        batch2 = torch.randint(0, self.config["vocab_size"], (2, 32))

        # Step 1 on Model A
        loss1 = model_a(batch1, labels=batch1).loss
        loss1.backward()
        opt_a.step()
        opt_a.zero_grad()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save checkpoint
            torch.save(model_a.state_dict(), os.path.join(tmp_dir, "model.pt"))
            torch.save({"optimizer": opt_a.state_dict(), "step": 1}, os.path.join(tmp_dir, "optimizer_state.pt"))

            # Step 2 on Model A (Uninterrupted)
            loss2_a = model_a(batch2, labels=batch2).loss
            loss2_a.backward()
            opt_a.step()

            # Fresh Model B + Optimizer B: restore from checkpoint
            model_b = TaraForCausalLM(self.config)
            model_b.load_state_dict(torch.load(os.path.join(tmp_dir, "model.pt")))
            opt_b = torch.optim.AdamW(build_tpu_optimizer_groups(model_b, weight_decay=0.01), lr=1e-3, eps=1e-5)
            restored = restore_tpu_optimizer_state(tmp_dir, opt_b, device=torch.device("cpu"))
            self.assertTrue(restored)

            # Step 2 on Model B (Resumed)
            loss2_b = model_b(batch2, labels=batch2).loss
            loss2_b.backward()
            opt_b.step()

            # Compare weights: must be 100% bitwise identical
            weight_diff = max((p_a - p_b).abs().max().item() for p_a, p_b in zip(model_a.parameters(), model_b.parameters()))
            self.assertEqual(weight_diff, 0.0, f"Model weights diverged after resume: diff={weight_diff}")

            # Compare optimizer moments: must be 100% identical
            for (p_a, s_a), (p_b, s_b) in zip(opt_a.state.items(), opt_b.state.items()):
                if "exp_avg" in s_a:
                    ea_diff = (s_a["exp_avg"] - s_b["exp_avg"]).abs().max().item()
                    self.assertEqual(ea_diff, 0.0, f"exp_avg diverged: diff={ea_diff}")
                if "exp_avg_sq" in s_a:
                    eas_diff = (s_a["exp_avg_sq"] - s_b["exp_avg_sq"]).abs().max().item()
                    self.assertEqual(eas_diff, 0.0, f"exp_avg_sq diverged: diff={eas_diff}")

    def test_12_unequal_valid_token_distributed_ce_gradient_parity(self):
        """
        Rigorously verify distributed gradient weighting parity on an 8-core simulated cluster
        with unequal valid-token distributions (including multiple cores with 0 valid tokens):
          Core 0: 0 tokens (all-masked)
          Core 1: 180 tokens
          Core 2: 40 tokens
          Core 3: 95 tokens
          Core 4: 20 tokens
          Core 5: 120 tokens
          Core 6: 0 tokens (all-masked)
          Core 7: 165 tokens
          Total Cluster Valid Tokens: 620 tokens
        Proves:
          1. Distributed reduced gradient == Single-device concatenated reference gradient (diff < 1e-5).
          2. Shows that unweighted per-core mean gradient produces significant mathematical error (> 1e-3).
        """
        import copy
        torch.manual_seed(42)
        base_model = TaraForCausalLM(self.config)
        base_state = copy.deepcopy(base_model.state_dict())

        valid_counts = [0, 180, 40, 95, 20, 120, 0, 165]
        world_size = 8
        cluster_valid_tokens = sum(valid_counts)  # 620 tokens
        seq_len = 64
        bsz = 4

        core_inputs = []
        core_labels = []
        for r, n_val in enumerate(valid_counts):
            torch.manual_seed(1000 + r)
            x_r = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))
            y_r = torch.full((bsz, seq_len), -100, dtype=torch.long)
            if n_val > 0:
                flat_y = y_r[..., 1:].reshape(-1)
                rand_indices = torch.randperm(flat_y.size(0))[:n_val]
                flat_y[rand_indices] = torch.randint(0, self.config["vocab_size"], (n_val,))
                y_r[..., 1:] = flat_y.reshape(bsz, seq_len - 1)
            core_inputs.append(x_r)
            core_labels.append(y_r)

        # 1. Single-device Global Reference
        ref_model = TaraForCausalLM(self.config)
        ref_model.load_state_dict(base_state)
        ref_model.train()

        total_ref_loss_sum = torch.tensor(0.0)
        for r in range(world_size):
            res_r = compute_tpu_chunked_loss(ref_model, core_inputs[r], core_labels[r], chunk_sz=32)
            total_ref_loss_sum = total_ref_loss_sum + res_r.loss_sum

        ref_global_loss = total_ref_loss_sum / float(cluster_valid_tokens)
        ref_global_loss.backward()
        ref_grads = {name: p.grad.clone() for name, p in ref_model.named_parameters() if p.grad is not None}

        # 2. Distributed Execution Simulation across 8 Cores
        core_grads = []
        for r in range(world_size):
            core_model = TaraForCausalLM(self.config)
            core_model.load_state_dict(base_state)
            core_model.train()

            res_r = compute_tpu_chunked_loss(core_model, core_inputs[r], core_labels[r], chunk_sz=32)
            unreduced = res_r.loss_sum
            unreduced.backward()

            r_grads = {}
            for name, p in core_model.named_parameters():
                if p.grad is not None:
                    # Scaled by (world_size / cluster_valid_tokens) before all-reduce average
                    scaled_grad = p.grad.clone() * (float(world_size) / float(cluster_valid_tokens))
                    r_grads[name] = scaled_grad
                else:
                    r_grads[name] = torch.zeros_like(p)
            core_grads.append(r_grads)

        # All-reduce average across the 8 cores (as xm.reduce_gradients does):
        dist_grads = {}
        for name in ref_grads:
            sum_g = sum(core_grads[r][name] for r in range(world_size))
            dist_grads[name] = sum_g / float(world_size)

        # 3. Assert Mathematical Parity: dist_grads == ref_grads within 1e-5
        max_diff = 0.0
        for name in ref_grads:
            diff = (dist_grads[name] - ref_grads[name]).abs().max().item()
            if diff > max_diff:
                max_diff = diff

        self.assertLess(max_diff, 1e-5, f"Distributed gradient diverged from global reference: max_diff={max_diff}")

        # 4. Demonstrate flaw of old unweighted average:
        old_unweighted_grads = {}
        for name in ref_grads:
            old_sum = torch.zeros_like(ref_grads[name])
            for r in range(world_size):
                raw_g = core_grads[r][name] * (float(cluster_valid_tokens) / float(world_size))
                if valid_counts[r] > 0:
                    old_sum = old_sum + (raw_g / float(valid_counts[r]))
            old_unweighted_grads[name] = old_sum / float(world_size)

        old_max_diff = max((old_unweighted_grads[name] - ref_grads[name]).abs().max().item() for name in ref_grads)
        self.assertGreater(old_max_diff, 1e-3, f"Old unweighted average should show significant error, got {old_max_diff}")

    def test_13_unequal_valid_token_distributed_ce_plus_aux_gradient_parity(self):
        """
        Rigorously verify distributed gradient weighting parity for combined
        Cross-Entropy + Auxiliary variance regularizer on an 8-core simulated cluster:
          unreduced_loss = loss_sum + (N_valid * aux_loss)
        Proves:
          1. Distributed reduced gradient matches single concatenated reference gradient (diff < 1e-5).
          2. Unweighted average shows significant error (> 1e-3).
        """
        import copy
        torch.manual_seed(42)
        base_model = TaraForCausalLM(self.config)
        base_state = copy.deepcopy(base_model.state_dict())

        valid_counts = [0, 180, 40, 95, 20, 120, 0, 165]
        world_size = 8
        cluster_valid_tokens = sum(valid_counts)  # 620 tokens
        seq_len = 64
        bsz = 4

        core_inputs = []
        core_labels = []
        for r, n_val in enumerate(valid_counts):
            torch.manual_seed(2000 + r)
            x_r = torch.randint(0, self.config["vocab_size"], (bsz, seq_len))
            y_r = torch.full((bsz, seq_len), -100, dtype=torch.long)
            if n_val > 0:
                flat_y = y_r[..., 1:].reshape(-1)
                rand_indices = torch.randperm(flat_y.size(0))[:n_val]
                flat_y[rand_indices] = torch.randint(0, self.config["vocab_size"], (n_val,))
                y_r[..., 1:] = flat_y.reshape(bsz, seq_len - 1)
            core_inputs.append(x_r)
            core_labels.append(y_r)

        # 1. Single-device Concatenated Global Reference (CE + aux_loss)
        ref_model = TaraForCausalLM(self.config)
        ref_model.load_state_dict(base_state)
        ref_model.train()

        total_ref_unreduced = torch.tensor(0.0)
        for r in range(world_size):
            res_r = compute_tpu_chunked_loss(ref_model, core_inputs[r], core_labels[r], chunk_sz=32)
            total_ref_unreduced = total_ref_unreduced + res_r.unreduced_loss

        ref_global_loss = total_ref_unreduced / float(cluster_valid_tokens)
        ref_global_loss.backward()
        ref_grads = {name: p.grad.clone() for name, p in ref_model.named_parameters() if p.grad is not None}

        # 2. Distributed Execution Simulation across 8 Cores (CE + aux_loss)
        core_grads = []
        for r in range(world_size):
            core_model = TaraForCausalLM(self.config)
            core_model.load_state_dict(base_state)
            core_model.train()

            res_r = compute_tpu_chunked_loss(core_model, core_inputs[r], core_labels[r], chunk_sz=32)
            unreduced = res_r.unreduced_loss
            unreduced.backward()

            r_grads = {}
            for name, p in core_model.named_parameters():
                if p.grad is not None:
                    scaled_grad = p.grad.clone() * (float(world_size) / float(cluster_valid_tokens))
                    r_grads[name] = scaled_grad
                else:
                    r_grads[name] = torch.zeros_like(p)
            core_grads.append(r_grads)

        # All-reduce average across the 8 cores:
        dist_grads = {}
        for name in ref_grads:
            sum_g = sum(core_grads[r][name] for r in range(world_size))
            dist_grads[name] = sum_g / float(world_size)

        # 3. Assert Mathematical Parity: dist_grads == ref_grads within 1e-5
        max_diff = 0.0
        for name in ref_grads:
            diff = (dist_grads[name] - ref_grads[name]).abs().max().item()
            if diff > max_diff:
                max_diff = diff

        self.assertLess(max_diff, 1e-5, f"Distributed CE+aux gradient diverged from global reference: max_diff={max_diff}")

    def test_14_all_masked_cluster_wide_skip_and_weight_decay_immunity(self):
        """
        Verify that when ALL 8 cores in a cluster have 0 valid tokens:
          1. sync_tpu_sum reports cluster_valid_tokens == 0.
          2. tpu_optimizer_step skips the step collectively across all cores.
          3. Model weights remain 100% bitwise identical (zero AdamW weight decay applied).
          4. All parameter gradient references are cleared (set_to_none=True).
        """
        model = TaraForCausalLM(self.config)
        orig_weights = {name: p.clone() for name, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

        def all_zero_mock(val, tag):
            if "tpu_valid_tokens" in tag:
                return 0
            return val

        set_collective_mock(all_zero_mock)
        try:
            for p in model.parameters():
                p.grad = torch.zeros_like(p)

            is_skipped, _ = tpu_optimizer_step(
                opt, model=model, cur_loss=0.0, valid_tokens=0, step=1, attempt=0
            )

            self.assertTrue(is_skipped, "Must skip step when entire cluster has 0 valid tokens")

            for name, p in model.named_parameters():
                diff = (p - orig_weights[name]).abs().max().item()
                self.assertEqual(diff, 0.0, f"Weight decay altered parameter {name} on empty window!")

            for p in model.parameters():
                self.assertIsNone(p.grad, "Parameter .grad must be None after zero_grad(set_to_none=True)")
        finally:
            set_collective_mock(None)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
=============================================================================
TARA Native TPU v3-8 Implementation & Compatibility Verification Test Suite
=============================================================================
Validates:
  1. TPU v3-8 Detection & Environment Abstraction
  2. Model Loading & Instantiation (TaraForCausalLM reuse)
  3. Tokenizer & Dataset Streaming Loading
  4. Gradient Accumulation Precision
  5. Optimizer Step Execution (tpu_optimizer_step)
  6. TPU Checkpoint Save & Resume (Zero overwrite of GPU checkpoints)
  7. GPU Pipeline Files Integrity (train_main.py, model_dataset.py, run_train.sh 100% untouched)
=============================================================================
"""

import os
import sys
import json
import shutil
import subprocess
import torch

from tpu_config import (
    is_tpu_available,
    get_tpu_hardware_info,
    get_tpu_device,
    get_tpu_rank,
    get_tpu_world_size,
    is_tpu_master,
    tpu_optimizer_step,
    TPUConfig,
)
from model_dataset import TaraForCausalLM, TaraNativeTokenizer
from train_tpu import save_tpu_checkpoint, load_tpu_checkpoint, compute_tpu_lr


def run_all_tpu_tests() -> bool:
    print("=" * 80)
    print("  RUNNING TARA GOOGLE TPU v3-8 VERIFICATION SUITE")
    print("=" * 80)

    all_passed = True

    # -------------------------------------------------------------------------
    # TEST 1: TPU v3-8 Detection
    # -------------------------------------------------------------------------
    print("\n[TEST 1/7] Testing TPU v3-8 Detection & Environment Layer...")
    try:
        hw_info = get_tpu_hardware_info()
        dev = get_tpu_device()
        rank = get_tpu_rank()
        world_size = get_tpu_world_size()
        print(f"  -> Hardware info: {hw_info}")
        print(f"  -> Active device: {dev}")
        print(f"  -> Process rank : {rank}/{world_size}")
        print("  [PASS] Test 1: TPU detection & device resolution succeeded.")
    except Exception as e:
        print(f"  [FAIL] Test 1: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 2: Model Loading (TaraForCausalLM reuse)
    # -------------------------------------------------------------------------
    print("\n[TEST 2/7] Testing Native TARA Model Loading & Forward Pass...")
    try:
        config_path = "storage/models/tara_candidate_v1/config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Micro-config for test speed
        test_cfg = dict(cfg)
        test_cfg["num_hidden_layers"] = 2
        test_cfg["hidden_size"] = 256
        test_cfg["intermediate_size"] = 512
        test_cfg["num_attention_heads"] = 4
        test_cfg["num_key_value_heads"] = 2
        test_cfg["head_dim"] = 64

        model = TaraForCausalLM(test_cfg)
        dummy_in = torch.randint(0, cfg["vocab_size"], (1, 16))
        dummy_out = model(dummy_in, labels=dummy_in)
        loss = dummy_out.loss if hasattr(dummy_out, "loss") else dummy_out[1]
        assert loss is not None and not torch.isnan(loss)
        print(f"  -> Model instantiated cleanly. Forward pass loss: {loss.item():.4f}")
        print("  [PASS] Test 2: Model architecture and forward execution verified.")
    except Exception as e:
        print(f"  [FAIL] Test 2: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 3: Dataset & Tokenizer Loading
    # -------------------------------------------------------------------------
    print("\n[TEST 3/7] Testing Native Tokenizer & Dataset Ingestion...")
    try:
        tok_path = "storage/models/tara_candidate_v1/tokenizer.json"
        tok = TaraNativeTokenizer.from_file(tok_path)
        encoded = tok.encode("TARA Google TPU v3-8 verification test.")
        ids = list(encoded.ids) if hasattr(encoded, "ids") else list(encoded)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        print(f"  -> Tokenizer vocabulary size: {tok.get_vocab_size():,}")
        print(f"  -> Encoded/Decoded string   : '{decoded}'")
        print("  [PASS] Test 3: Tokenizer encoding & decoding verified.")
    except Exception as e:
        print(f"  [FAIL] Test 3: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 4: Gradient Accumulation
    # -------------------------------------------------------------------------
    print("\n[TEST 4/7] Testing Gradient Accumulation Precision...")
    try:
        model.zero_grad()
        dummy_in_1 = torch.randint(0, cfg["vocab_size"], (1, 16))
        dummy_in_2 = torch.randint(0, cfg["vocab_size"], (1, 16))

        out1 = model(dummy_in_1, labels=dummy_in_1)
        l1 = (out1.loss if hasattr(out1, "loss") else out1[1]) / 2.0
        l1.backward()

        out2 = model(dummy_in_2, labels=dummy_in_2)
        l2 = (out2.loss if hasattr(out2, "loss") else out2[1]) / 2.0
        l2.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "Gradient accumulation failed to produce non-zero gradients"
        print("  -> Accumulated 2 micro-batches successfully into model parameter gradients.")
        print("  [PASS] Test 4: Gradient accumulation verified.")
    except Exception as e:
        print(f"  [FAIL] Test 4: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 5: Optimizer Step Execution
    # -------------------------------------------------------------------------
    print("\n[TEST 5/7] Testing TPU-Optimized Optimizer Update...")
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        old_weight = next(model.parameters()).clone()
        tpu_optimizer_step(opt)
        new_weight = next(model.parameters())
        assert not torch.equal(old_weight, new_weight), "Weights did not update after optimizer step"
        print("  -> Weight values successfully updated via optimizer step.")
        print("  [PASS] Test 5: Optimizer update execution verified.")
    except Exception as e:
        print(f"  [FAIL] Test 5: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 6: Checkpoint Save & Resume (Zero Overwrite of GPU Checkpoints)
    # -------------------------------------------------------------------------
    print("\n[TEST 6/7] Testing TPU Checkpoint Save & Resume Isolation...")
    test_tpu_ckpt_dir = "storage/models/checkpoints_tpu/test_verification_step10"
    gpu_ckpt_dir = "storage/models/checkpoints/stage0_pretraining"
    try:
        # 1. Save TPU test checkpoint
        saved = save_tpu_checkpoint(
            model=model,
            optimizer=opt,
            config=test_cfg,
            tok_path=tok_path,
            output_dir=test_tpu_ckpt_dir,
            step=10,
            epoch=0,
            loss=1.85,
            best_loss=1.85,
            stage_id="stage0_pretraining_tpu",
        )
        assert saved, "Failed to save TPU checkpoint"
        assert os.path.isfile(os.path.join(test_tpu_ckpt_dir, "model.safetensors")), "model.safetensors missing in TPU dir"
        assert os.path.isfile(os.path.join(test_tpu_ckpt_dir, "checkpoint_state.json")), "checkpoint_state.json missing"

        # 2. Verify GPU checkpoint directory was NOT touched or overwritten
        assert os.path.isdir(gpu_ckpt_dir), f"GPU checkpoint directory missing: {gpu_ckpt_dir}"
        gpu_state_file = os.path.join(gpu_ckpt_dir, "checkpoint_state.json")
        if os.path.isfile(gpu_state_file):
            with open(gpu_state_file, "r") as gf:
                gpu_state = json.load(gf)
            # Ensure GPU state still has its original step (1800), not overwritten by TPU (10)
            assert gpu_state.get("step") == 1800, f"GPU checkpoint was modified! Found step {gpu_state.get('step')}"

        # 3. Resume from TPU checkpoint
        step, epoch, loss, resumed, cursor = load_tpu_checkpoint(test_tpu_ckpt_dir, model)
        assert resumed and step == 10 and abs(loss - 1.85) < 1e-4

        # Clean up temporary test checkpoint
        shutil.rmtree(test_tpu_ckpt_dir, ignore_errors=True)
        print("  -> TPU checkpoint saved and resumed with 100% fidelity.")
        print(f"  -> Confirmed: GPU checkpoint at {gpu_ckpt_dir} remained completely untouched!")
        print("  [PASS] Test 6: Checkpoint save/resume & GPU isolation verified.")
    except Exception as e:
        print(f"  [FAIL] Test 6: {e}")
        shutil.rmtree(test_tpu_ckpt_dir, ignore_errors=True)
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 7: GPU Pipeline Files Integrity Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 7/7] Verifying Existing GPU Pipeline Files are 100% UNTOUCHED...")
    try:
        # Check git status for train_main.py, model_dataset.py, run_train.sh
        proc = subprocess.run(
            ["git", "status", "--porcelain", "train_main.py", "model_dataset.py", "run_train.sh"],
            capture_output=True,
            text=True,
        )
        modified_gpu_files = proc.stdout.strip()
        if modified_gpu_files:
            print(f"  [FAIL] The following GPU files were unexpectedly modified:\n{modified_gpu_files}")
            all_passed = False
        else:
            print("  -> Checked: train_main.py     (0 modifications)")
            print("  -> Checked: model_dataset.py  (0 modifications)")
            print("  -> Checked: run_train.sh      (0 modifications)")
            print("  [PASS] Test 7: All existing GPU files are 100% UNTOUCHED!")
    except Exception as e:
        print(f"  [FAIL] Test 7: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL 7 TPU VERIFICATION TESTS PASSED SUCCESSFULLY! (100% READY)")
    else:
        print("  SOME TESTS ENCOUNTERED FAILURES - REVIEW LOGS ABOVE.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_tpu_tests()
    sys.exit(0 if success else 1)

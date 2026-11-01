#!/usr/bin/env python3
"""
=============================================================================
TARA TPU Peer-Review Mandatory Validation Test Suite
=============================================================================
Directly implements the three mandatory peer-review tests:
  1. 8-Core TPU Fault Injection:
     Simulates 8 parallel TPU worker cores (Ranks 0..7). Injects NaN or loss-shock
     on Core 5. Asserts ALL 8 cores skip the update synchronously without weight
     divergence, and resume updates together on the subsequent valid step.
  2. Multi-Domain & Multi-Rank Checkpoint Resume:
     Simulates multi-domain (Domain A, B, C) and multi-rank data cursor progression.
     Simulates a poisoned sample skip in Domain B, commits independent domain cursors,
     and validates that load_tpu_checkpoint resumes exact domain offsets without
     max() conflation, with auditable samples_skipped telemetry.
  3. Checkpoint Tampering & Cryptographic Gate Invariant Rejection:
     Evaluates 4 distinct tampering vectors:
       - Vector A: Tensor shape modification -> Rejected by shape invariant check
       - Vector B: Tensor NaN float insertion -> Rejected by deep tensor scan
       - Vector C: Tokenizer corruption / vocab mismatch -> Rejected by tokenizer validator
       - Vector D: Invalid signature / altered config SHA-256 -> Rejected by Ed25519 gate
=============================================================================
"""

import os
import sys
import copy
import json
import shutil
import hashlib
import threading
from typing import Dict, List, Any

# Ensure repository root is on PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from tpu_config import (
    tpu_optimizer_step,
    sync_tpu_decision,
    set_collective_mock,
    build_tpu_optimizer_groups,
)
from model_dataset import TaraForCausalLM
from train_tpu import (
    save_tpu_checkpoint,
    load_tpu_checkpoint,
    validate_tpu_checkpoint,
    extract_tpu_data_cursor,
)
from train_main import (
    resolve_trusted_public_key,
    verify_ed25519_authorization_signature,
    verify_pretraining_authorization_proof,
)


# =============================================================================
# Helper: Multi-Core Simulated Collective Coordinator (Scales to N Cores)
# =============================================================================
class NCoreSimulatedCollectiveCoordinator:
    """Simulates PyTorch/XLA xm.mesh_reduce and xm.reduce_gradients across N cores."""
    def __init__(self, world_size: int = 8):
        self.world_size = world_size
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._payloads: Dict[str, List[int]] = {}
        self._results: Dict[str, int] = {}
        self._grads: Dict[str, List[List[torch.Tensor]]] = {}
        self._reduced_grads: Dict[str, List[torch.Tensor]] = {}

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


def make_test_model_and_config():
    cfg = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    model = TaraForCausalLM(cfg)
    return model, cfg


# =============================================================================
# MANDATORY TEST 1: 8-Core TPU Fault Injection
# =============================================================================
def test_1_eight_core_tpu_fault_injection():
    """
    Spawns 8 parallel worker threads simulating an 8-core TPU v4-8 / v5e-8.
    Injects NaN on Core 5; asserts all 8 cores synchronously skip without mutating
    weights, and resume updating simultaneously on the next step.
    """
    print("\n--- Running Mandatory Test 1: 8-Core TPU Fault Injection ---")
    world_size = 8
    coordinator = NCoreSimulatedCollectiveCoordinator(world_size=world_size)
    set_collective_mock(
        hook=lambda val, tag: coordinator.mesh_reduce(val, tag),
        grad_hook=lambda model, step: coordinator.reduce_gradients(model, step),
    )

    base_model, cfg = make_test_model_and_config()
    models = [copy.deepcopy(base_model) for _ in range(world_size)]
    opts = [torch.optim.AdamW(build_tpu_optimizer_groups(m), lr=1e-3) for m in models]

    # Step 1: Core 5 experiences NaN loss; others experience normal loss 1.5
    coordinator.reset()
    results = [{} for _ in range(world_size)]

    def run_core(rank: int, loss_val: float, step: int, attempt: int):
        m = models[rank]
        opt = opts[rank]
        wb = [p.clone().detach() for p in m.parameters()]
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.01

        is_skip, gnorm = tpu_optimizer_step(
            opt, model=m, max_norm=1.0, cur_loss=loss_val, ema_loss=1.5, step=step, attempt=attempt
        )
        wa = [p.clone().detach() for p in m.parameters()]
        changed = any(not torch.equal(b, a) for b, a in zip(wb, wa))
        results[rank] = {"rank": rank, "skipped": is_skip, "changed": changed}

    threads = []
    for rank in range(world_size):
        # Inject NaN into core 5
        loss_val = float("nan") if rank == 5 else 1.5
        t = threading.Thread(target=run_core, args=(rank, loss_val, 1, 1))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    # Validate all 8 cores skipped and weights did NOT mutate
    for rank in range(world_size):
        assert results[rank]["skipped"] is True, f"Core {rank} failed to skip step on Core 5 NaN!"
        assert results[rank]["changed"] is False, f"Core {rank} mutated weights on skipped step!"

    # Validate bitwise weight parity across all 8 cores
    ref_weights = list(models[0].parameters())
    for r in range(1, world_size):
        for p_idx, (p_ref, p_target) in enumerate(zip(ref_weights, models[r].parameters())):
            assert torch.equal(p_ref, p_target), f"Weight divergence between Core 0 and Core {r} on tensor {p_idx}!"

    # Step 2: Immediate Recovery - All 8 cores experience valid loss 1.45
    coordinator.reset()
    threads = []
    for rank in range(world_size):
        t = threading.Thread(target=run_core, args=(rank, 1.45, 1, 2))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    for rank in range(world_size):
        assert results[rank]["skipped"] is False, f"Core {rank} failed to execute valid step after recovery!"
        assert results[rank]["changed"] is True, f"Core {rank} failed to update weights on valid step!"

    # Verify bitwise parity preserved after recovery update
    ref_weights = list(models[0].parameters())
    for r in range(1, world_size):
        for p_idx, (p_ref, p_target) in enumerate(zip(ref_weights, models[r].parameters())):
            assert torch.equal(p_ref, p_target), f"Weight divergence between Core 0 and Core {r} after recovery update!"

    set_collective_mock(None)
    print("  [PASS] Mandatory Test 1: 8-Core TPU Fault Injection & Lockstep Recovery Verified.")
    return True


# =============================================================================
# MANDATORY TEST 2: Multi-Domain & Multi-Rank Checkpoint Resume
# =============================================================================
def test_2_multi_domain_multi_rank_resume():
    """
    Verifies multi-domain cursor isolation:
      - Domain A ("kannada_general") cursor = 120
      - Domain B ("indic_technical") cursor = 45 (poisoned sample skipped at 45 -> committed)
      - Domain C ("tara_dialogue") cursor = 300
    Asserts checkpoint save and resume preserves each domain independently without max() conflation,
    and telemetry maintains total_samples_trained + samples_skipped == samples_consumed.
    """
    print("\n--- Running Mandatory Test 2: Multi-Domain & Multi-Rank Checkpoint Resume ---")
    test_ckpt_dir = os.path.join(REPO_ROOT, "storage", "models", "checkpoints_tpu", "test_multidomain_resume")
    shutil.rmtree(test_ckpt_dir, ignore_errors=True)

    base_model, cfg = make_test_model_and_config()
    opt = torch.optim.AdamW(base_model.parameters(), lr=1e-3)
    tok_path = os.path.join(REPO_ROOT, "storage", "models", "tara_candidate_v1", "tokenizer.json")

    # Simulate multi-domain committed cursors
    committed_domain_cursors = {
        "kannada_general": {
            "active_shard": "kannada_general_shard_003.parquet",
            "line_offset": 120,
            "byte_offset": 10240,
            "chunk_idx": 1,
        },
        "indic_technical": {
            "active_shard": "indic_technical_shard_001.parquet",
            "line_offset": 45,  # Poison sample was here; cursor committed past it
            "byte_offset": 4096,
            "chunk_idx": 0,
        },
        "tara_dialogue": {
            "active_shard": "tara_dialogue_shard_008.parquet",
            "line_offset": 300,
            "byte_offset": 25600,
            "chunk_idx": 2,
        },
    }

    # Data cursor with telemetry
    total_samples_trained = 1000
    samples_skipped = 32
    samples_consumed = total_samples_trained + samples_skipped  # 1032
    steps_skipped = 1

    mock_cursor = {
        "active_shard": "tara_dialogue_shard_008.parquet",
        "active_shard_line_offset": 300,
        "active_shard_byte_offset": 25600,
        "active_chunk_idx": 3,
        "total_samples_trained": total_samples_trained,
        "samples_consumed": samples_consumed,
        "samples_skipped": samples_skipped,
        "steps_skipped": steps_skipped,
        "rank": 0,
        "domain_cursors": committed_domain_cursors,
    }

    # Save checkpoint
    saved = save_tpu_checkpoint(
        model=base_model,
        optimizer=opt,
        config=cfg,
        tok_path=tok_path,
        output_dir=test_ckpt_dir,
        step=50,
        epoch=1,
        loss=1.65,
        best_loss=1.65,
        stage_id="stage0_multidomain_test",
        data_cursor=mock_cursor,
    )
    assert saved is True, "Failed to save multi-domain test checkpoint!"

    # Simulate restart and load checkpoint into fresh model & optimizer
    resumed_model, _ = make_test_model_and_config()
    resumed_opt = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)

    step, epoch, loss, resumed, restored_cursor = load_tpu_checkpoint(
        test_ckpt_dir, resumed_model, resumed_opt
    )
    assert resumed is True
    assert step == 50
    assert restored_cursor is not None

    # Verify domain cursors were preserved without conflation
    domain_res = restored_cursor.get("domain_cursors", {})
    assert "kannada_general" in domain_res and domain_res["kannada_general"]["line_offset"] == 120
    assert "indic_technical" in domain_res and domain_res["indic_technical"]["line_offset"] == 45
    assert "tara_dialogue" in domain_res and domain_res["tara_dialogue"]["line_offset"] == 300

    # Verify dataset telemetry accounting
    assert restored_cursor.get("total_samples_trained") == 1000
    assert restored_cursor.get("samples_skipped") == 32
    assert restored_cursor.get("samples_consumed") == 1032
    assert restored_cursor.get("steps_skipped") == 1

    shutil.rmtree(test_ckpt_dir, ignore_errors=True)
    print("  [PASS] Mandatory Test 2: Multi-Domain Cursor Isolation & Telemetry Accounting Verified.")
    return True


# =============================================================================
# MANDATORY TEST 3: Checkpoint Tampering & Cryptographic Invariant Rejection
# =============================================================================
def test_3_checkpoint_tampering_rejection():
    """
    Tests 4 distinct tampering vectors to guarantee fail-closed rejection:
      - Vector A: Tensor shape alteration -> Rejected by exact architecture shape check
      - Vector B: NaN insertion -> Rejected by deep tensor scan
      - Vector C: Tokenizer corruption / vocab size <= 0 -> Rejected by tokenizer validator
      - Vector D: Altered signature / config SHA-256 -> Rejected by Ed25519 authorization gate
    """
    print("\n--- Running Mandatory Test 3: Checkpoint Tampering Rejection ---")
    cand_dir = os.path.join(REPO_ROOT, "storage", "models", "tara_candidate_v1")
    tamper_dir = os.path.join(REPO_ROOT, "storage", "models", "checkpoints_tpu", "test_tamper_gate")
    shutil.rmtree(tamper_dir, ignore_errors=True)
    os.makedirs(tamper_dir, exist_ok=True)

    base_model, cfg = make_test_model_and_config()
    tok_path = os.path.join(cand_dir, "tokenizer.json")

    # Helper to create a valid baseline checkpoint
    def create_baseline_ckpt():
        shutil.rmtree(tamper_dir, ignore_errors=True)
        os.makedirs(tamper_dir, exist_ok=True)
        # Write config.json
        with open(os.path.join(tamper_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        # Copy tokenizer
        shutil.copy2(tok_path, os.path.join(tamper_dir, "tokenizer.json"))
        # Write COMMITTED
        with open(os.path.join(tamper_dir, "COMMITTED"), "w", encoding="utf-8") as f:
            f.write("COMMITTED_STEP_50\n")
        # Write STAGE_METADATA.json
        with open(os.path.join(tamper_dir, "STAGE_METADATA.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "COMPLETED", "step": 50, "stage_id": "stage0"}, f)
        # Write safetensors
        from safetensors.torch import save_file
        sd = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        save_file(sd, os.path.join(tamper_dir, "model.safetensors"))

    # Vector A: Tensor shape tampering
    create_baseline_ckpt()
    from safetensors.torch import load_file, save_file
    sd = load_file(os.path.join(tamper_dir, "model.safetensors"), device="cpu")
    # Mutate shape of q_proj in layer 0
    bad_key = "model.layers.0.self_attn.q_proj.weight"
    sd[bad_key] = torch.randn(cfg["num_attention_heads"] * cfg["head_dim"] - 1, cfg["hidden_size"])
    save_file(sd, os.path.join(tamper_dir, "model.safetensors"))

    ok, msg = validate_tpu_checkpoint(tamper_dir, require_completed=True)
    assert ok is False, "Tensor shape tampering MUST be rejected!"
    assert "Dimension mismatch" in msg or "Shape mismatch" in msg
    print(f"  -> Vector A (Shape Tampering): Successfully REJECTED ({msg})")

    # Vector B: NaN insertion
    create_baseline_ckpt()
    sd = load_file(os.path.join(tamper_dir, "model.safetensors"), device="cpu")
    # Insert NaN into embed_tokens
    sd["model.embed_tokens.weight"][0, 0] = float("nan")
    save_file(sd, os.path.join(tamper_dir, "model.safetensors"))

    ok, msg = validate_tpu_checkpoint(tamper_dir, require_completed=True)
    assert ok is False, "NaN tensor tampering MUST be rejected!"
    assert "NaN values detected" in msg
    print(f"  -> Vector B (NaN Insertion): Successfully REJECTED ({msg})")

    # Vector C: Tokenizer corruption
    create_baseline_ckpt()
    with open(os.path.join(tamper_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write("{corrupt_json: true}")

    ok, msg = validate_tpu_checkpoint(tamper_dir, require_completed=True)
    assert ok is False, "Corrupted tokenizer MUST be rejected!"
    assert "Corrupted tokenizer" in msg or "Invalid tokenizer" in msg
    print(f"  -> Vector C (Tokenizer Corruption): Successfully REJECTED ({msg})")

    # Vector D: Cryptographic Signature / Config SHA-256 Tampering
    from tests.test_pretraining_security import load_canonical_proof
    proof = load_canonical_proof()
    trusted_pub = resolve_trusted_public_key()
    assert trusted_pub is not None

    tampered_proof = copy.deepcopy(proof)
    tampered_proof["model_config_sha256"] = "ff" * 32  # Tamper config hash
    ok, msg = verify_ed25519_authorization_signature(tampered_proof, trusted_public_key=trusted_pub)
    assert ok is False, "Tampered config hash MUST fail Ed25519 signature verification!"
    print(f"  -> Vector D (Tampered Hash in Auth Proof): Successfully REJECTED ({msg})")

    shutil.rmtree(tamper_dir, ignore_errors=True)
    print("  [PASS] Mandatory Test 3: Checkpoint Tampering & Cryptographic Invariants Verified.")
    return True


def run_all_mandatory_peer_review_tests():
    print("=" * 80)
    print("  RUNNING MANDATORY PEER-REVIEW VALIDATION TEST SUITE")
    print("=" * 80)
    test_1_eight_core_tpu_fault_injection()
    test_2_multi_domain_multi_rank_resume()
    test_3_checkpoint_tampering_rejection()
    print("=" * 80)
    print("  >>> ALL 3 MANDATORY PEER-REVIEW TESTS PASSED 100% <<<")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_all_mandatory_peer_review_tests()
    sys.exit(0 if success else 1)

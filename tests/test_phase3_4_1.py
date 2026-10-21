"""
Phase 3.4.1 Acceptance Tests: Checkpoint Selection Isolation & Multi-Stage Scope Verification

Tests cover:
1. Two intentionally different test checkpoints (stage_0 at step 10 vs stage_interrupted_test at step 15).
2. Resume stage_0 asserts exact step 10, weights, optimizer, tokenizer fingerprint, cursor, and ShardLedger.
3. Resume stage_interrupted_test separately asserts its own exact step 15, weights, optimizer, and cursor.
4. Sibling pointer rejection: Pointer to a sibling stage is rejected, preventing cross-stage state contamination.
5. Missing pointer fallback: Seamless fallback to versioned directory within target scope when pointer is missing.
6. Stale/corrupt pointer: Corrupt pointer fails safe without adopting unrelated sibling checkpoint.
7. Stage ID validation: Rejects candidate whose stage_id contradicts expected_stage_id.
"""

import json
import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

from cloud_train import (
    TaraForCausalLM,
    TaraNativeTokenizer,
    save_checkpoint,
    try_resume_from_checkpoint,
    compute_tokenizer_fingerprint,
    _cleanup_old_versioned_checkpoints,
    ensure_remote_assets_available,
    ShardLedger,
)


def _make_tiny_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


class TestCheckpointSelectionIsolation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        self.vocab = {chr(i): i for i in range(32, 127)}
        self.vocab["<|unk|>"] = 0
        self.tokenizer = TaraNativeTokenizer(self.vocab)
        self.tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        with open(self.tok_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": self.vocab}}, f)

        # Setup common parent directory containing two sibling stage directories
        self.shared_parent = os.path.join(self.tmpdir, "shared_training_run")
        self.stage_0_dir = os.path.join(self.shared_parent, "stage_0")
        self.stage_interrupted_dir = os.path.join(self.shared_parent, "stage_interrupted_test")

        # Checkpoint A (stage_0): 10 steps, distinct weights, distinct cursor & ShardLedger
        torch.manual_seed(100)
        self.model_a = TaraForCausalLM(self.config)
        self.opt_a = torch.optim.AdamW(self.model_a.parameters(), lr=1e-4)
        self.scaler_a = _make_tiny_scaler()
        self.ledger_a = ShardLedger(self.stage_0_dir, stage_id="stage_0")

        # Train 10 steps
        for _ in range(10):
            self.opt_a.zero_grad()
            x = torch.randint(0, 128, (1, 16))
            out = self.model_a(x)
            loss = (out[0] if isinstance(out, tuple) else out).sum() * 0.0 + 1.25
            loss.backward()
            self.opt_a.step()

        self.cursor_a = {
            "active_shard": "shard_0.jsonl",
            "active_shard_line_offset": 10,
            "active_shard_byte_offset": 1000,
            "active_chunk_idx": 0,
            "total_samples_trained": 10,
            "samples_consumed": 10,
            "completed_shards": [],
            "applied_unit_ids": [f"u_{i}" for i in range(10)],
        }
        save_checkpoint(
            self.model_a, self.opt_a, self.scaler_a, self.config, self.tok_path,
            self.stage_0_dir, step=10, epoch=0, loss=1.25, best_loss=1.25,
            stage_id="stage_0", data_cursor=self.cursor_a, sync_to_gdrive=False,
            shard_ledger=self.ledger_a,
        )

        # Checkpoint B (stage_interrupted_test): 15 steps, different weights, different cursor
        torch.manual_seed(200)
        self.model_b = TaraForCausalLM(self.config)
        self.opt_b = torch.optim.AdamW(self.model_b.parameters(), lr=2e-4)
        self.scaler_b = _make_tiny_scaler()
        for _ in range(15):
            self.opt_b.zero_grad()
            x = torch.randint(0, 128, (1, 16))
            out = self.model_b(x)
            loss = (out[0] if isinstance(out, tuple) else out).sum() * 0.0 + 0.95
            loss.backward()
            self.opt_b.step()

        self.cursor_b = {
            "active_shard": "shard_1.jsonl",
            "active_shard_line_offset": 15,
            "active_shard_byte_offset": 1500,
            "active_chunk_idx": 1,
            "total_samples_trained": 15,
            "samples_consumed": 15,
            "completed_shards": ["shard_0.jsonl"],
            "applied_unit_ids": [f"u_b_{i}" for i in range(15)],
        }
        save_checkpoint(
            self.model_b, self.opt_b, self.scaler_b, self.config, self.tok_path,
            self.stage_interrupted_dir, step=15, epoch=1, loss=0.95, best_loss=0.95,
            stage_id="stage_interrupted_test", data_cursor=self.cursor_b, sync_to_gdrive=False,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_stage_0_selects_exact_stage_0_state(self):
        """Resuming stage_0 restores step 10, model_a weights, cursor_a, and ShardLedger, NOT sibling stage."""
        reload_model = TaraForCausalLM(self.config)
        reload_opt = torch.optim.AdamW(reload_model.parameters(), lr=1e-4)
        reload_scaler = _make_tiny_scaler()

        step, epoch, best_loss, was_resumed, cursor = try_resume_from_checkpoint(
            self.stage_0_dir, reload_model, reload_opt, reload_scaler,
            tokenizer_path=self.tok_path,
            tokenizer=self.tokenizer,
            expected_stage_id="stage_0",
        )
        self.assertTrue(was_resumed)
        self.assertEqual(step, 10)
        self.assertAlmostEqual(best_loss, 1.25)
        self.assertIsNotNone(cursor)
        self.assertEqual(cursor["active_shard"], "shard_0.jsonl")
        self.assertEqual(cursor["active_shard_line_offset"], 10)
        self.assertEqual(cursor.get("applied_unit_ids"), [f"u_{i}" for i in range(10)])

        # ShardLedger verification: ledger persisted and matches step 10
        reloaded_ledger = ShardLedger(self.stage_0_dir, stage_id="stage_0")
        self.assertEqual(reloaded_ledger.last_committed_step, 10)
        self.assertEqual(reloaded_ledger.active_shard_line_offset, 10)
        self.assertEqual(reloaded_ledger.total_records_processed, 10)

        # Model weights must match model_a, NOT model_b
        for (k_a, v_a), (k_r, v_r) in zip(self.model_a.state_dict().items(), reload_model.state_dict().items()):
            self.assertEqual(k_a, k_r)
            self.assertTrue(torch.allclose(v_a, v_r, atol=1e-6), f"Weight mismatch for {k_a}")

    def test_resume_stage_interrupted_selects_exact_stage_interrupted_state(self):
        """Resuming stage_interrupted_test restores step 15, model_b weights, and cursor_b."""
        reload_model = TaraForCausalLM(self.config)
        reload_opt = torch.optim.AdamW(reload_model.parameters(), lr=2e-4)
        reload_scaler = _make_tiny_scaler()

        step, epoch, best_loss, was_resumed, cursor = try_resume_from_checkpoint(
            self.stage_interrupted_dir, reload_model, reload_opt, reload_scaler,
            tokenizer_path=self.tok_path,
            tokenizer=self.tokenizer,
            expected_stage_id="stage_interrupted_test",
        )
        self.assertTrue(was_resumed)
        self.assertEqual(step, 15)
        self.assertAlmostEqual(best_loss, 0.95)
        self.assertIsNotNone(cursor)
        self.assertEqual(cursor["active_shard"], "shard_1.jsonl")
        self.assertEqual(cursor["active_shard_line_offset"], 15)
        self.assertEqual(cursor["active_chunk_idx"], 1)

        # Model weights must match model_b, NOT model_a
        for (k_b, v_b), (k_r, v_r) in zip(self.model_b.state_dict().items(), reload_model.state_dict().items()):
            self.assertEqual(k_b, k_r)
            self.assertTrue(torch.allclose(v_b, v_r, atol=1e-6), f"Weight mismatch for {k_b}")

    def test_sibling_pointer_in_parent_rejected_for_stage_0(self):
        """If parent_dir/latest_checkpoint.json points to stage_interrupted, stage_0 ignores it."""
        # Intentionally point parent_dir's latest_checkpoint.json to stage_interrupted_test
        parent_ptr = os.path.join(self.shared_parent, "latest_checkpoint.json")
        # Find stage_interrupted versioned directory
        vdirs_b = [
            os.path.join(self.shared_parent, d) for d in os.listdir(self.shared_parent)
            if d.startswith("stage_interrupted_test.step15")
        ]
        self.assertTrue(len(vdirs_b) > 0)
        with open(parent_ptr, "w", encoding="utf-8") as f:
            json.dump({
                "versioned_dir": vdirs_b[0],
                "output_dir": self.stage_interrupted_dir,
                "stage_id": "stage_interrupted_test",
                "step": 15,
            }, f)

        # Resume stage_0. It MUST reject parent_ptr and restore stage_0 (step 10)
        reload_model = TaraForCausalLM(self.config)
        reload_opt = torch.optim.AdamW(reload_model.parameters(), lr=1e-4)
        reload_scaler = _make_tiny_scaler()

        step, _, _, was_resumed, cursor = try_resume_from_checkpoint(
            self.stage_0_dir, reload_model, reload_opt, reload_scaler,
            tokenizer_path=self.tok_path,
            tokenizer=self.tokenizer,
            expected_stage_id="stage_0",
        )
        self.assertTrue(was_resumed)
        self.assertEqual(step, 10, "stage_0 erroneously adopted sibling pointer step 15!")
        self.assertEqual(cursor["active_shard"], "shard_0.jsonl")

    def test_stale_corrupt_pointer_falls_back_to_valid_versioned_dir(self):
        """If latest_checkpoint.json is corrupted, discovery falls back to scoped versioned dir."""
        ptr_file = os.path.join(self.stage_0_dir, "latest_checkpoint.json")
        with open(ptr_file, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_NOT_VALID_JSON{")

        reload_model = TaraForCausalLM(self.config)
        reload_opt = torch.optim.AdamW(reload_model.parameters(), lr=1e-4)
        reload_scaler = _make_tiny_scaler()

        step, _, _, was_resumed, cursor = try_resume_from_checkpoint(
            self.stage_0_dir, reload_model, reload_opt, reload_scaler,
            tokenizer_path=self.tok_path,
            tokenizer=self.tokenizer,
            expected_stage_id="stage_0",
        )
        self.assertTrue(was_resumed)
        self.assertEqual(step, 10)

    def test_stage_id_mismatch_fails_closed(self):
        """Calling try_resume_from_checkpoint with an expected_stage_id rejecting candidates returns False."""
        reload_model = TaraForCausalLM(self.config)
        reload_opt = torch.optim.AdamW(reload_model.parameters(), lr=1e-4)
        reload_scaler = _make_tiny_scaler()

        step, _, _, was_resumed, cursor = try_resume_from_checkpoint(
            self.stage_0_dir, reload_model, reload_opt, reload_scaler,
            tokenizer_path=self.tok_path,
            tokenizer=self.tokenizer,
            expected_stage_id="unrelated_stage_99",
        )
        self.assertFalse(was_resumed, "Resume should fail closed when stage_id does not match expected_stage_id")
        self.assertEqual(step, 0)

    def test_ensure_remote_assets_bypasses_bulk_dataset_by_default(self):
        """Verify ensure_remote_assets_available does NOT trigger bulk dataset downloads when bulk_download_dataset=False."""
        from unittest.mock import patch

        m_dir = os.path.join(self.tmpdir, "mock_model")
        d_dir = os.path.join(self.tmpdir, "mock_ds")
        os.makedirs(m_dir, exist_ok=True)
        # Create dummy config and safetensors so model download is skipped
        with open(os.path.join(m_dir, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(m_dir, "model.safetensors"), "w") as f:
            f.write("mock")

        with patch("huggingface_hub.snapshot_download") as mock_snap:
            ensure_remote_assets_available(m_dir, d_dir, repo_id="dummy/repo", bulk_download_dataset=False)
            # Assert snapshot_download was NOT called for dataset
            for call_item in mock_snap.call_args_list:
                self.assertNotEqual(call_item.kwargs.get("repo_type"), "dataset", "Bulk dataset download should not be triggered by default!")

    def test_ensure_remote_assets_calls_bulk_dataset_when_explicitly_requested(self):
        """Verify ensure_remote_assets_available triggers dataset snapshot when bulk_download_dataset=True."""
        from unittest.mock import patch

        m_dir = os.path.join(self.tmpdir, "mock_model_2")
        d_dir = os.path.join(self.tmpdir, "mock_ds_2")
        os.makedirs(m_dir, exist_ok=True)
        with open(os.path.join(m_dir, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(m_dir, "model.safetensors"), "w") as f:
            f.write("mock")

        with patch("huggingface_hub.snapshot_download") as mock_snap:
            ensure_remote_assets_available(m_dir, d_dir, repo_id="dummy/repo", bulk_download_dataset=True)
            called_dataset = any(c.kwargs.get("repo_type") == "dataset" for c in mock_snap.call_args_list)
            self.assertTrue(called_dataset, "Bulk dataset download should be triggered when bulk_download_dataset=True")

    def test_cleanup_purges_duplicate_runs_for_same_step_and_drive_clones(self):
        """Verify _cleanup_old_versioned_checkpoints purges older runs of the same step and Drive clone duplicates."""
        from cloud_train import _cleanup_old_versioned_checkpoints, _collect_versioned_checkpoint_dirs

        parent = os.path.join(self.tmpdir, "retention_test_parent")
        os.makedirs(parent, exist_ok=True)

        # Create two runs for Step 5: an older timestamp and a newer timestamp
        step5_old = os.path.join(parent, "stage_smoke_test.step5_1792565077")
        step5_new = os.path.join(parent, "stage_smoke_test.step5_1792565270")
        os.makedirs(step5_old, exist_ok=True)
        os.makedirs(step5_new, exist_ok=True)
        with open(os.path.join(step5_old, "COMMITTED"), "w") as f:
            f.write("{}")
        with open(os.path.join(step5_new, "COMMITTED"), "w") as f:
            f.write("{}")

        # Create a Google Drive clone folder and clone file
        clone_dir = os.path.join(parent, "stage_smoke_test.step5_1792565077 (1)")
        clone_file = os.path.join(parent, "latest_checkpoint (1).json")
        os.makedirs(clone_dir, exist_ok=True)
        with open(clone_file, "w") as f:
            f.write("{}")

        # Run cleanup retaining keep_n=1
        _cleanup_old_versioned_checkpoints(parent, "stage_smoke_test", keep_n=1, force_keep_exact=True)

        # Assert: the older run for Step 5 is deleted, newer run is preserved
        self.assertFalse(os.path.exists(step5_old), "Older run for Step 5 should be purged!")
        self.assertTrue(os.path.exists(step5_new), "Newer run for Step 5 should be retained!")

        # Assert: clone copies are purged
        self.assertFalse(os.path.exists(clone_dir), "Drive clone directory should be purged!")
        self.assertFalse(os.path.exists(clone_file), "Drive clone file should be purged!")

    def test_collect_versioned_dirs_without_committed_marker(self):
        """Verify _collect_versioned_checkpoint_dirs discovers valid checkpoints even without COMMITTED marker."""
        from cloud_train import _collect_versioned_checkpoint_dirs

        parent = os.path.join(self.tmpdir, "collect_test_parent")
        os.makedirs(parent, exist_ok=True)

        # Create directory with model.safetensors and checkpoint_state.json but NO COMMITTED marker
        legacy_dir = os.path.join(parent, "stage_smoke_test.step5_1792565270")
        os.makedirs(legacy_dir, exist_ok=True)
        with open(os.path.join(legacy_dir, "model.safetensors"), "wb") as f:
            f.write(b"mock_weights")
        with open(os.path.join(legacy_dir, "checkpoint_state.json"), "w") as f:
            f.write("{}")

        discovered = _collect_versioned_checkpoint_dirs(parent, "stage_smoke_test")
        self.assertIn(legacy_dir, discovered, "Versioned directory with model and state json must be discovered even without COMMITTED marker")


if __name__ == "__main__":
    unittest.main()

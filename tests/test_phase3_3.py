#!/usr/bin/env python3
"""
Phase 3.3 — Final Blocker Remediation Acceptance Tests.

Covers:
- Finding 1: DatasetSnapshot Diff Logging (added, removed, renamed, modified, empty discovery protection)
- Finding 2: Safe Cleanup of Abandoned Staging Directories (active protected, recovery checkpoints retained, pointer protection, age/PID checks)
- Finding 3: Resume at Completed Shard Boundary (zero redundant download on completed shard, safe reconciliation)
- Finding 4: Tokenizer Fingerprint Verification (fail-closed on mismatch, config/special-token changes, legacy compatibility)
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from cloud_train import (
    TaraForCausalLM,
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    ShardLedger,
    DatasetSnapshot,
    save_checkpoint,
    try_resume_from_checkpoint,
    compute_file_sha256,
    compute_tokenizer_fingerprint,
    _cleanup_old_versioned_checkpoints,
)


def _make_tokenizer(vocab_size: int = 64):
    vocab = {chr(i): i for i in range(32, 32 + min(vocab_size, 95))}
    vocab["<|unk|>"] = 0
    return TaraNativeTokenizer(vocab)


def _write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


# ===========================================================================
# Finding 1: DatasetSnapshot Diff Logging & Overwrite Protection
# ===========================================================================

class TestFinding1_DatasetSnapshot(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_f1_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_added_shards_detected(self):
        """Snapshot with [s0] compared against live [s0, s1] detects s1 as added."""
        snap = DatasetSnapshot(self.tmpdir)
        snap.initialize_from_discovery("repo", "v2", ["s0.jsonl"])

        # Reload fresh instance
        snap2 = DatasetSnapshot(self.tmpdir)
        diff = snap2.diff_against_live(["s0.jsonl", "s1.jsonl"])
        self.assertEqual(diff["added"], ["s1.jsonl"])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["common"], ["s0.jsonl"])

    def test_removed_shards_detected(self):
        """Snapshot with [s0, s1] compared against live [s0] detects s1 as removed."""
        snap = DatasetSnapshot(self.tmpdir)
        snap.initialize_from_discovery("repo", "v2", ["s0.jsonl", "s1.jsonl"])

        snap2 = DatasetSnapshot(self.tmpdir)
        diff = snap2.diff_against_live(["s0.jsonl"])
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], ["s1.jsonl"])
        self.assertEqual(diff["common"], ["s0.jsonl"])

    def test_renamed_shards_detected(self):
        """Renaming a shard manifests as added new name + removed old name."""
        snap = DatasetSnapshot(self.tmpdir)
        snap.initialize_from_discovery("repo", "v2", ["old_name.jsonl"])

        snap2 = DatasetSnapshot(self.tmpdir)
        diff = snap2.diff_against_live(["new_name.jsonl"])
        self.assertIn("new_name.jsonl", diff["added"])
        self.assertIn("old_name.jsonl", diff["removed"])

    def test_modified_shard_content_detected(self):
        """Content change under the same shard name is detected by is_shard_content_changed()."""
        snap = DatasetSnapshot(self.tmpdir)
        path = os.path.join(self.tmpdir, "shard.jsonl")
        _write_jsonl(path, [{"text": "AAA"}])
        sha1 = compute_file_sha256(path)
        snap.record_shard_hash("shard.jsonl", sha1)

        # Modify content
        _write_jsonl(path, [{"text": "BBB"}])
        sha2 = compute_file_sha256(path)
        self.assertTrue(snap.is_shard_content_changed("shard.jsonl", sha2))

    def test_failed_empty_discovery_cannot_overwrite_valid_prior_snapshot(self):
        """An empty discovery list cannot overwrite a valid non-empty prior snapshot."""
        snap = DatasetSnapshot(self.tmpdir)
        snap.initialize_from_discovery("repo", "v2", ["s0.jsonl", "s1.jsonl"])
        self.assertEqual(len(snap.ordered_shards), 2)

        # Attempt empty discovery
        snap.initialize_from_discovery("repo", "v2", [])

        # Verify prior catalog is preserved
        snap2 = DatasetSnapshot(self.tmpdir)
        self.assertEqual(snap2.ordered_shards, ["s0.jsonl", "s1.jsonl"],
                         "Empty discovery must not erase valid prior snapshot catalog")


# ===========================================================================
# Finding 2: Safe Cleanup of Abandoned Staging Directories
# ===========================================================================

class TestFinding2_OrphanedStagingDirectories(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_f2_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mock_checkpoint(self, name: str, committed: bool = True) -> str:
        d = os.path.join(self.tmpdir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{}")
        if committed:
            with open(os.path.join(d, "COMMITTED"), "w") as f:
                f.write(json.dumps({"committed_utc": "2026-10-20T12:00:00Z"}))
        return d

    def test_active_staging_directory_never_deleted(self):
        """An active staging directory passed as active_staging_dir is never deleted."""
        active_staging = os.path.join(self.tmpdir, ".ckpt_staging_ckpt_step10_12345_99999")
        os.makedirs(active_staging, exist_ok=True)

        _cleanup_old_versioned_checkpoints(
            parent_dir=self.tmpdir,
            output_dir_basename="ckpt",
            keep_n=2,
            active_staging_dir=active_staging,
            min_abandoned_age_seconds=0,
        )
        self.assertTrue(os.path.exists(active_staging),
                        "Active staging directory must never be deleted")

    def test_committed_checkpoint_never_deleted_within_retention(self):
        """At least keep_n committed checkpoints are always preserved."""
        c1 = self._create_mock_checkpoint("ckpt.step1_1000", committed=True)
        c2 = self._create_mock_checkpoint("ckpt.step2_2000", committed=True)
        c3 = self._create_mock_checkpoint("ckpt.step3_3000", committed=True)

        _cleanup_old_versioned_checkpoints(
            parent_dir=self.tmpdir,
            output_dir_basename="ckpt",
            keep_n=2,
        )
        # Step 2 and Step 3 must be kept (newest 2)
        self.assertTrue(os.path.exists(c3), "Newest checkpoint must be kept")
        self.assertTrue(os.path.exists(c2), "Second newest checkpoint must be kept")
        self.assertFalse(os.path.exists(c1), "Checkpoint beyond keep_n must be purged")

    def test_latest_pointer_target_never_deleted(self):
        """Directory pointed to by latest_checkpoint.json is preserved even if step sort differs."""
        c1 = self._create_mock_checkpoint("ckpt.step1_1000", committed=True)
        c2 = self._create_mock_checkpoint("ckpt.step2_2000", committed=True)
        c3 = self._create_mock_checkpoint("ckpt.step3_3000", committed=True)

        # Pointer points to c1 (e.g. pinned or roll-back checkpoint)
        ptr_file = os.path.join(self.tmpdir, "latest_checkpoint.json")
        with open(ptr_file, "w") as f:
            json.dump({"versioned_dir": c1}, f)

        _cleanup_old_versioned_checkpoints(
            parent_dir=self.tmpdir,
            output_dir_basename="ckpt",
            keep_n=2,
        )
        self.assertTrue(os.path.exists(c1),
                        "Checkpoint referenced by latest pointer must be preserved")

    def test_abandoned_staging_directory_purged(self):
        """Abandoned staging directory from dead PID older than threshold is removed."""
        abandoned = os.path.join(self.tmpdir, ".ckpt_staging_ckpt_step5_1000_99999999")
        os.makedirs(abandoned, exist_ok=True)
        with open(os.path.join(abandoned, "partial.bin"), "w") as f:
            f.write("partial")

        # Set mtime back by 600s
        past_time = time.time() - 600
        os.utime(abandoned, (past_time, past_time))

        _cleanup_old_versioned_checkpoints(
            parent_dir=self.tmpdir,
            output_dir_basename="ckpt",
            keep_n=2,
            min_abandoned_age_seconds=300,
        )
        self.assertFalse(os.path.exists(abandoned),
                         "Abandoned staging directory older than threshold must be purged")

    def test_recent_staging_directory_not_purged(self):
        """Staging directory younger than age threshold is kept for write safety."""
        recent = os.path.join(self.tmpdir, ".ckpt_staging_ckpt_step5_1000_99999999")
        os.makedirs(recent, exist_ok=True)

        _cleanup_old_versioned_checkpoints(
            parent_dir=self.tmpdir,
            output_dir_basename="ckpt",
            keep_n=2,
            min_abandoned_age_seconds=300,
        )
        self.assertTrue(os.path.exists(recent),
                        "Recent staging directory must not be deleted")

    def test_stale_pointer_does_not_purge_valid_checkpoints(self):
        """A corrupt or invalid pointer file does not prevent valid recovery checkpoints from being kept."""
        c1 = self._create_mock_checkpoint("ckpt.step10_1000", committed=True)
        c2 = self._create_mock_checkpoint("ckpt.step20_2000", committed=True)

        ptr_file = os.path.join(self.tmpdir, "latest_checkpoint.json")
        with open(ptr_file, "w") as f:
            f.write("CORRUPTED_JSON_NOT_VALID")

        _cleanup_old_versioned_checkpoints(
            parent_dir=self.tmpdir,
            output_dir_basename="ckpt",
            keep_n=2,
        )
        self.assertTrue(os.path.exists(c1))
        self.assertTrue(os.path.exists(c2))


# ===========================================================================
# Finding 3: Resume at a Completed Shard Boundary
# ===========================================================================

class TestFinding3_ResumeCompletedShardBoundary(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_f3_")
        self.tokenizer = _make_tokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_completed_shard_skips_download_on_resume(self):
        """
        When a shard is verified COMPLETED in the ledger and the resume cursor
        is at or past the final line, TaraStreamingJsonlDataset skips downloading
        that shard and advances directly to the subsequent shard.
        """
        ledger = ShardLedger(self.tmpdir)
        # Register shard_0 as completed with 3 records (lines 0, 1, 2)
        ledger.mark_completed("shard_0.jsonl", records_count=3, file_sha256="0" * 64)

        # Create shard_1 locally
        shard1_path = os.path.join(self.tmpdir, "shard_1.jsonl")
        _write_jsonl(shard1_path, [{"text": "SHARD_ONE_RECORD"}])

        # Download tracker mock
        download_calls = []

        def mock_downloader(repo_id, prefix, shard_name, dataset_dir, **kwargs):
            download_calls.append(shard_name)
            p = os.path.join(dataset_dir, shard_name)
            _write_jsonl(p, [{"text": "DOWNLOADED"}])
            return p

        with unittest.mock.patch("cloud_train.download_single_shard_jit", side_effect=mock_downloader):
            ds = TaraStreamingJsonlDataset(
                file_paths=["shard_0.jsonl", "shard_1.jsonl"],
                tokenizer=self.tokenizer,
                max_seq_len=8,
                resume_shard="shard_0.jsonl",
                resume_line_offset=2,  # final line of 3-record shard (line 2 == rec_count - 1)
                shard_ledger=ledger,
                dataset_dir=self.tmpdir,
                repo_id="dummy_repo",
            )
            items = list(ds)

        # shard_0 must NOT have been downloaded
        self.assertNotIn("shard_0.jsonl", download_calls,
                         "Completed shard_0 must not be downloaded when cursor is at final line")
        # Items should come from shard_1
        self.assertTrue(len(items) > 0)
        self.assertEqual(items[0][2], "shard_1.jsonl")

    def test_incomplete_shard_does_not_skip(self):
        """
        When cursor is in the middle of a shard (not at EOF), even if ledger has
        metadata, the shard must NOT be skipped (safe reconciliation).
        """
        ledger = ShardLedger(self.tmpdir)
        ledger.mark_completed("shard_0.jsonl", records_count=10, file_sha256="0" * 64)

        download_calls = []

        def mock_downloader(repo_id, prefix, shard_name, dataset_dir, **kwargs):
            download_calls.append(shard_name)
            p = os.path.join(dataset_dir, shard_name)
            _write_jsonl(p, [{"text": "A" * 10} for _ in range(10)])
            return p

        with unittest.mock.patch("cloud_train.download_single_shard_jit", side_effect=mock_downloader):
            ds = TaraStreamingJsonlDataset(
                file_paths=["shard_0.jsonl", "shard_1.jsonl"],
                tokenizer=self.tokenizer,
                max_seq_len=8,
                resume_shard="shard_0.jsonl",
                resume_line_offset=2,  # Middle of 10-record shard
                shard_ledger=ledger,
                dataset_dir=self.tmpdir,
                repo_id="dummy_repo",
            )
            _ = list(ds)

        self.assertIn("shard_0.jsonl", download_calls,
                      "Incomplete shard must be downloaded to continue remaining records")


# ===========================================================================
# Finding 4: Tokenizer Fingerprint Verification & Fail-Closed Policy
# ===========================================================================

class TestFinding4_TokenizerFingerprintVerification(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_f4_")
        self.tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        with open(self.tok_path, "w") as f:
            json.dump({"version": "1.0", "model": {"vocab": {"A": 0, "B": 1, "<|unk|>": 2}}}, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save_test_ckpt(self, out_dir: str, tok_path: str) -> None:
        cfg = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False
        }
        m = TaraForCausalLM(cfg).cpu()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = _make_scaler()
        os.makedirs(out_dir, exist_ok=True)
        save_checkpoint(
            m, opt, scaler, cfg, tok_path, out_dir,
            step=1, epoch=0, loss=1.0, best_loss=1.0, stage_id="stage_0",
            sync_to_gdrive=False,
        )

    def test_matching_tokenizer_accepted(self):
        """Resuming with matching tokenizer succeeds and restores step."""
        out_dir = os.path.join(self.tmpdir, "ckpt_match")
        self._save_test_ckpt(out_dir, self.tok_path)

        cfg = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False
        }
        m = TaraForCausalLM(cfg).cpu()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = _make_scaler()

        step, _, _, was_resumed, _ = try_resume_from_checkpoint(
            out_dir, m, opt, scaler,
            tokenizer_path=self.tok_path,
        )
        self.assertTrue(was_resumed, "Matching tokenizer must resume successfully")
        self.assertEqual(step, 1)

    def test_modified_tokenizer_file_fails_closed(self):
        """Resuming with modified tokenizer.json fails closed (candidate rejected)."""
        out_dir = os.path.join(self.tmpdir, "ckpt_mod")
        self._save_test_ckpt(out_dir, self.tok_path)

        # Create altered tokenizer
        alt_tok_path = os.path.join(self.tmpdir, "alt_tokenizer.json")
        with open(alt_tok_path, "w") as f:
            json.dump({"version": "2.0_ALTERED", "model": {"vocab": {"X": 0, "Y": 1}}}, f)

        cfg = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False
        }
        m = TaraForCausalLM(cfg).cpu()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = _make_scaler()

        step, _, _, was_resumed, _ = try_resume_from_checkpoint(
            out_dir, m, opt, scaler,
            tokenizer_path=alt_tok_path,
        )
        self.assertFalse(was_resumed,
                         "Mismatched tokenizer must fail closed (candidate rejected)")
        self.assertEqual(step, 0)

    def test_special_tokens_change_fails_closed(self):
        """Altering special_tokens_map.json changes fingerprint and fails closed."""
        tok_dir = os.path.join(self.tmpdir, "tok_with_sp")
        os.makedirs(tok_dir, exist_ok=True)
        tok_file = os.path.join(tok_dir, "tokenizer.json")
        shutil.copy2(self.tok_path, tok_file)
        sp_file = os.path.join(tok_dir, "special_tokens_map.json")
        with open(sp_file, "w") as f:
            json.dump({"unk_token": "<|unk|>"}, f)

        fp1 = compute_tokenizer_fingerprint(tok_file)

        # Modify special tokens
        with open(sp_file, "w") as f:
            json.dump({"unk_token": "<|diff_unk|>"}, f)
        fp2 = compute_tokenizer_fingerprint(tok_file)

        self.assertNotEqual(fp1["canonical_fingerprint"], fp2["canonical_fingerprint"],
                            "Altering special_tokens_map.json must alter canonical fingerprint")

    def test_missing_fingerprint_legacy_checkpoint_backward_compatible(self):
        """A legacy checkpoint without tokenizer_fingerprint metadata is allowed with matching file."""
        out_dir = os.path.join(self.tmpdir, "legacy_ckpt")
        self._save_test_ckpt(out_dir, self.tok_path)

        # Manually strip tokenizer_fingerprint from checkpoint_state.json to simulate legacy checkpoint
        import glob as _glob
        parent = os.path.dirname(os.path.abspath(out_dir))
        versioned_dirs = _glob.glob(os.path.join(parent, f"{os.path.basename(out_dir)}.step*"))
        self.assertTrue(versioned_dirs)
        vdir = sorted(versioned_dirs)[-1]
        state_file = os.path.join(vdir, "checkpoint_state.json")
        with open(state_file, "r") as f:
            st = json.load(f)
        st.pop("tokenizer_fingerprint", None)
        with open(state_file, "w") as f:
            json.dump(st, f, indent=2)

        cfg = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False
        }
        m = TaraForCausalLM(cfg).cpu()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = _make_scaler()

        step, _, _, was_resumed, _ = try_resume_from_checkpoint(
            out_dir, m, opt, scaler,
            tokenizer_path=self.tok_path,
        )
        self.assertTrue(was_resumed,
                        "Legacy checkpoint with matching tokenizer.json must resume successfully")
        self.assertEqual(step, 1)


if __name__ == "__main__":
    unittest.main()

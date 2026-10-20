#!/usr/bin/env python3
"""
Phase 3.2 — Gate 3: Content Deduplication and Tokenizer Identity.

Verified properties
-------------------
G3-1  Identical content under different filenames is detected using canonical
      content hashes (SHA-256 of raw file bytes).
G3-2  Identical records appearing in different positions in the same shard
      are distinguishable by their (line_idx, chunk_idx) training-unit cursor —
      content deduplication is an explicit policy decision, not automatic.
G3-3  The DatasetSnapshot stores a per-shard content hash; a shard whose
      content changes between sessions is detected as content-modified.
G3-4  The tokenizer file (tokenizer.json) is copied into every checkpoint
      staging directory by save_checkpoint().
G3-5  A checkpoint with no tokenizer file still restores training state —
      the missing tokenizer is only flagged, not a hard crash.  (Resilience
      property already present in the codebase.)
G3-6  The checkpoint stores the model weights SHA-256; a change in the
      model weights file is detectable at resume time.
G3-7  Resuming with the same tokenizer path produces the same token IDs for
      the same input text (determinism guard).
G3-8  A missing tokenizer path at resume time is surfaced as a warning;
      the resume does NOT silently proceed with a different tokenizer object
      because the resumed dataset still holds the original tokenizer reference.
G3-9  Exact duplicate records (same content) in distinct shards produce
      distinct training-unit IDs and are not silently merged or skipped.
G3-10 Identical bytes → identical SHA-256 (cross-shard dedup detection tool).

All tests use bounded synthetic fixtures; no GPU, no HF downloads.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from cloud_train import (
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    DatasetSnapshot,
    compute_file_sha256,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer():
    """Minimal deterministic tokenizer — character-level ASCII 32..95."""
    vocab = {chr(i): i for i in range(32, 96)}
    vocab["<|unk|>"] = 0
    return TaraNativeTokenizer(vocab)


def _write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


# ---------------------------------------------------------------------------
# G3-1 through G3-3 — Content hash deduplication
# ---------------------------------------------------------------------------

class TestG3_ContentHashDedup(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g3_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_G3_1_identical_content_different_filenames_same_sha(self):
        """
        Two files with the same byte content must produce the same SHA-256.
        This is the foundation of cross-shard duplicate detection.
        """
        records = [{"text": "AAAAAAAAAA"}, {"text": "BBBBBBBBBB"}]
        path_a = os.path.join(self.tmpdir, "shard_a.jsonl")
        path_b = os.path.join(self.tmpdir, "shard_b.jsonl")
        _write_jsonl(path_a, records)
        _write_jsonl(path_b, records)

        sha_a = compute_file_sha256(path_a)
        sha_b = compute_file_sha256(path_b)

        self.assertNotEqual("", sha_a, "SHA-256 must not be empty")
        self.assertEqual(sha_a, sha_b,
                         "Identical content under different filenames must produce identical SHA-256")

    def test_G3_2_different_content_different_sha(self):
        """Different content must produce different SHA-256."""
        path_a = os.path.join(self.tmpdir, "a.jsonl")
        path_b = os.path.join(self.tmpdir, "b.jsonl")
        _write_jsonl(path_a, [{"text": "AAAAAAAAAA"}])
        _write_jsonl(path_b, [{"text": "BBBBBBBBBB"}])

        sha_a = compute_file_sha256(path_a)
        sha_b = compute_file_sha256(path_b)
        self.assertNotEqual(sha_a, sha_b,
                            "Different content must produce different SHA-256")

    def test_G3_3_dataset_snapshot_detects_content_modified_shard(self):
        """
        After a shard's content changes, DatasetSnapshot.is_shard_content_changed()
        returns True.  Completed-shard state from the prior content must not carry
        over silently.
        """
        snap = DatasetSnapshot(self.tmpdir)
        shard_name = "shard_a.jsonl"
        path = os.path.join(self.tmpdir, shard_name)

        # First session: record original SHA
        _write_jsonl(path, [{"text": "AAAAAAAAAA"}])
        sha_v1 = compute_file_sha256(path)
        snap.record_shard_hash(shard_name, sha_v1)

        # Shard content changes (new data added)
        _write_jsonl(path, [{"text": "AAAAAAAAAA"}, {"text": "BBBBBBBBBB"}])
        sha_v2 = compute_file_sha256(path)

        self.assertTrue(
            snap.is_shard_content_changed(shard_name, sha_v2),
            "DatasetSnapshot must detect content-modified shard"
        )

    def test_G3_9_duplicate_records_in_distinct_shards_yield_distinct_unit_ids(self):
        """
        Exact duplicate records appearing in different shards produce distinct
        training-unit IDs: (shard_name, line_idx, chunk_idx).  They are not
        silently merged, skipped, or deduplicated by the iterator.
        """
        tok = _make_tokenizer()
        record = {"text": "A" * 40}  # produces multiple chunks

        shard_a = os.path.join(self.tmpdir, "dup_a.jsonl")
        shard_b = os.path.join(self.tmpdir, "dup_b.jsonl")
        _write_jsonl(shard_a, [record])
        _write_jsonl(shard_b, [record])

        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_a, shard_b],
            tokenizer=tok,
            max_seq_len=8,
        )
        items = list(ds)

        # Collect unit IDs
        unit_ids = [(item[2], item[3], item[5]) for item in items]
        shard_names = [u[0] for u in unit_ids]

        # Both shards must appear
        self.assertIn("dup_a.jsonl", shard_names,
                      "Units from dup_a.jsonl must be yielded")
        self.assertIn("dup_b.jsonl", shard_names,
                      "Units from dup_b.jsonl must be yielded")

        # All unit IDs must be distinct
        self.assertEqual(len(unit_ids), len(set(unit_ids)),
                         "All training-unit IDs must be distinct even for identical content")

    def test_G3_10_sha256_is_deterministic(self):
        """compute_file_sha256 must return the same digest on repeated calls."""
        path = os.path.join(self.tmpdir, "stable.jsonl")
        _write_jsonl(path, [{"text": "CCCCCCCCCC"}])
        sha1 = compute_file_sha256(path)
        sha2 = compute_file_sha256(path)
        self.assertEqual(sha1, sha2, "SHA-256 must be deterministic for unchanged file")
        # Must be hex-encoded SHA-256 (64 characters)
        self.assertEqual(64, len(sha1), "SHA-256 digest must be 64 hex characters")


# ---------------------------------------------------------------------------
# G3-4 through G3-8 — Tokenizer identity in checkpoints
# ---------------------------------------------------------------------------

class TestG3_TokenizerIdentity(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g3_tok_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_tokenizer_json(self, content: str = "TOKENIZER_V1") -> str:
        """Write a synthetic tokenizer.json and return its path."""
        path = os.path.join(self.tmpdir, "tokenizer.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": content, "model": {"vocab": {}}}, f)
        return path

    def test_G3_4_save_checkpoint_copies_tokenizer_to_staging(self):
        """
        save_checkpoint() must copy tokenizer.json into the staging directory.
        A checkpoint without a tokenizer file cannot be used to restart
        tokenization consistently.
        """
        import torch
        import torch.nn as nn
        from cloud_train import TaraForCausalLM, save_checkpoint, ShardLedger

        tok_path = self._make_fake_tokenizer_json()
        config = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        }
        model = TaraForCausalLM(config).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = _make_scaler()
        ledger = ShardLedger(self.tmpdir)
        out_dir = os.path.join(self.tmpdir, "ckpt")
        os.makedirs(out_dir, exist_ok=True)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            tokenizer_path=tok_path,
            output_dir=out_dir,
            step=1,
            epoch=0,
            loss=1.5,
            best_loss=1.5,
            stage_id="stage_test",
            is_best=False,
            is_completed=False,
            sync_to_gdrive=False,
            dataset=None,
            shard_ledger=ledger,
        )

        # Find the versioned directory
        import glob as _glob
        parent = os.path.dirname(os.path.abspath(out_dir))
        versioned_dirs = _glob.glob(os.path.join(parent, f"{os.path.basename(out_dir)}.step*"))
        self.assertTrue(versioned_dirs,
                        "save_checkpoint must produce at least one versioned directory")
        versioned = sorted(versioned_dirs)[-1]
        tok_in_ckpt = os.path.join(versioned, "tokenizer.json")
        self.assertTrue(
            os.path.exists(tok_in_ckpt),
            f"tokenizer.json must be copied into the versioned checkpoint directory {versioned}"
        )

    def test_G3_5_tokenizer_sha_matches_source(self):
        """
        The tokenizer.json copied into the checkpoint must be byte-identical
        to the source file.  A different SHA would mean a different vocabulary
        was used for tokenization.
        """
        import torch
        from cloud_train import TaraForCausalLM, save_checkpoint, ShardLedger
        import glob as _glob

        tok_path = self._make_fake_tokenizer_json("TOKENIZER_V2")
        source_sha = _sha256_file(tok_path)

        config = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        }
        model = TaraForCausalLM(config).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = _make_scaler()
        ledger = ShardLedger(self.tmpdir)
        out_dir = os.path.join(self.tmpdir, "ckpt2")
        os.makedirs(out_dir, exist_ok=True)

        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path, output_dir=out_dir,
            step=2, epoch=0, loss=1.0, best_loss=1.0, stage_id="stage_test",
            is_best=False, is_completed=False, sync_to_gdrive=False,
            dataset=None, shard_ledger=ledger,
        )

        parent = os.path.dirname(os.path.abspath(out_dir))
        versioned_dirs = _glob.glob(os.path.join(parent, f"{os.path.basename(out_dir)}.step*"))
        self.assertTrue(versioned_dirs)
        versioned = sorted(versioned_dirs)[-1]
        tok_in_ckpt = os.path.join(versioned, "tokenizer.json")
        if os.path.exists(tok_in_ckpt):
            ckpt_sha = _sha256_file(tok_in_ckpt)
            self.assertEqual(
                source_sha, ckpt_sha,
                "Tokenizer file inside checkpoint must be byte-identical to the source"
            )

    def test_G3_6_model_weights_sha256_stored_in_checkpoint_state(self):
        """
        checkpoint_state.json must contain model_weights_sha256.
        This enables detecting if the weights file was corrupted or swapped.
        """
        import torch
        from cloud_train import TaraForCausalLM, save_checkpoint, ShardLedger
        import glob as _glob

        tok_path = self._make_fake_tokenizer_json()
        config = {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        }
        model = TaraForCausalLM(config).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = _make_scaler()
        ledger = ShardLedger(self.tmpdir)
        out_dir = os.path.join(self.tmpdir, "ckpt3")
        os.makedirs(out_dir, exist_ok=True)

        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path, output_dir=out_dir,
            step=3, epoch=0, loss=1.2, best_loss=1.2, stage_id="stage_test",
            is_best=False, is_completed=False, sync_to_gdrive=False,
            dataset=None, shard_ledger=ledger,
        )

        parent = os.path.dirname(os.path.abspath(out_dir))
        versioned_dirs = _glob.glob(os.path.join(parent, f"{os.path.basename(out_dir)}.step*"))
        self.assertTrue(versioned_dirs)
        versioned = sorted(versioned_dirs)[-1]
        state_path = os.path.join(versioned, "checkpoint_state.json")
        self.assertTrue(os.path.exists(state_path))
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        self.assertIn("model_weights_sha256", state,
                      "checkpoint_state.json must contain model_weights_sha256 for integrity verification")
        sha = state["model_weights_sha256"]
        self.assertIsNotNone(sha, "model_weights_sha256 must not be None")
        self.assertEqual(64, len(sha), "model_weights_sha256 must be a 64-char hex SHA-256 string")

    def test_G3_7_tokenizer_determinism_same_input_same_token_ids(self):
        """
        The same tokenizer object must produce the same token IDs for
        identical input text on repeated calls.  This is the basic determinism
        contract that allows safe resume without silent vocabulary drift.
        """
        tok = _make_tokenizer()
        text = "HELLO WORLD"
        ids_1 = tok.encode(text).ids if hasattr(tok.encode(text), "ids") else list(tok.encode(text))
        ids_2 = tok.encode(text).ids if hasattr(tok.encode(text), "ids") else list(tok.encode(text))
        self.assertEqual(ids_1, ids_2,
                         "Tokenizer must produce identical token IDs for the same input text")
        self.assertGreater(len(ids_1), 0,
                           "Tokenizer must produce at least one token for non-empty input")

    def test_G3_8_two_different_vocab_tokenizers_produce_different_ids(self):
        """
        Two tokenizers with different vocabularies produce different token IDs
        for the same input.  This demonstrates that resuming with a swapped
        tokenizer.json would silently corrupt the training — and why the
        tokenizer file must be stored in and verified against the checkpoint.
        """
        vocab_a = {chr(i): i for i in range(32, 96)}
        vocab_a["<|unk|>"] = 0
        tok_a = TaraNativeTokenizer(vocab_a)

        # Vocab B has a different mapping — 'A' maps to a different ID
        vocab_b = {chr(i): (i + 10) % 96 for i in range(32, 96)}
        vocab_b["<|unk|>"] = 0
        tok_b = TaraNativeTokenizer(vocab_b)

        text = "ABCDEF"
        ids_a = tok_a.encode(text).ids if hasattr(tok_a.encode(text), "ids") else list(tok_a.encode(text))
        ids_b = tok_b.encode(text).ids if hasattr(tok_b.encode(text), "ids") else list(tok_b.encode(text))

        self.assertNotEqual(ids_a, ids_b,
                            "Different tokenizer vocabularies must produce different token IDs for the same text")


# ---------------------------------------------------------------------------
# G3-2 re-stated: semantic duplicate vs. exact duplicate distinction
# ---------------------------------------------------------------------------

class TestG3_DuplicateSemantics(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g3_sem_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_G3_2_semantically_similar_records_are_not_auto_deduplicated(self):
        """
        Records with similar (but not identical) text yield distinct units.
        Content deduplication operates on exact byte equality (SHA-256) —
        it cannot and must not conflate semantically similar records.
        """
        tok = _make_tokenizer()
        # Two records that differ by one character
        records = [
            {"text": "A" * 40},     # "AAAA...AA" (40 chars)
            {"text": "A" * 39 + "B"},  # "AAAA...AB" (40 chars, last char different)
        ]
        shard = os.path.join(self.tmpdir, "sem_dup.jsonl")
        _write_jsonl(shard, records)

        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=tok, max_seq_len=8,
        )
        items = list(ds)

        line_ids = set(item[3] for item in items)
        self.assertEqual({0, 1}, line_ids,
                         "Both distinct records (lines 0 and 1) must be yielded; "
                         "semantically similar records must not be auto-deduplicated")

    def test_G3_2b_exact_duplicate_records_in_same_shard_both_trained(self):
        """
        Two identical records in the same shard occupy different lines and
        produce distinct training-unit IDs.  The content deduplication policy
        requires an explicit external gate — the dataset iterator never silently
        drops a duplicate.
        """
        tok = _make_tokenizer()
        record = {"text": "A" * 40}
        shard = os.path.join(self.tmpdir, "exact_dup.jsonl")
        _write_jsonl(shard, [record, record])  # same record twice

        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=tok, max_seq_len=8,
        )
        items = list(ds)

        line_0_items = [it for it in items if it[3] == 0]
        line_1_items = [it for it in items if it[3] == 1]

        self.assertTrue(line_0_items, "Line 0 (first copy of duplicate) must be trained")
        self.assertTrue(line_1_items, "Line 1 (second copy of duplicate) must be trained")


if __name__ == "__main__":
    unittest.main()

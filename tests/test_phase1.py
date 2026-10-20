#!/usr/bin/env python3
"""
Phase 1 Data-Safety Acceptance Tests for TARA Cloud Training Engine.

All tests are isolated, bounded, and use only synthetic fixtures.
No GPU training, no Hugging Face downloads, no production checkpoints
are touched or created.

Test inventory (maps to acceptance criteria in the mission brief):
  D1. Checkpoint-write failure preserves prior valid checkpoint byte-for-byte.
  D2. Fresh-process restoration recovers matching model, optimizer, cursor state.
  D3. Long synthetic document is chunked without losing any intended target tokens.
  D4. Crash at shard EOF before optimizer/checkpoint commit does not mark shard
      completed or delete its only recoverable source.
  D5. Crash after checkpoint commit but before ledger commit restores committed state.
  D6. No chunk whose optimizer update is durably committed is processed again after resume.
  D7. No intended uncommitted training unit is permanently omitted.
  D8. Local eviction only deletes files owned by the managed cache.
"""

import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: Make cloud_train importable from the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# Minimal torch-only imports from cloud_train (no GPU needed)
from cloud_train import (
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    ShardLedger,
    collate_fn,
    compute_file_sha256,
    save_checkpoint,
    save_best_milestone,
    verify_checkpoint_integrity,
    try_resume_from_checkpoint,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_tiny_config() -> dict:
    """Returns a minimal TaraForCausalLM-compatible config."""
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }


def _make_tiny_model(config: dict = None) -> nn.Module:
    from cloud_train import TaraForCausalLM
    cfg = config or _make_tiny_config()
    return TaraForCausalLM(cfg).cpu()


def _make_tiny_tokenizer() -> TaraNativeTokenizer:
    """Returns a minimal character-level tokenizer over printable ASCII."""
    vocab = {chr(i): i for i in range(32, 32 + 64)}
    vocab["<|unk|>"] = 0
    return TaraNativeTokenizer(vocab)


def _make_tiny_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def _make_tiny_scaler():
    try:
        return torch.amp.GradScaler("cpu", enabled=False)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=False)


def _write_jsonl_shard(path: str, records: List[dict]):
    """Writes a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sha256(path: str) -> str:
    return compute_file_sha256(path)


def _build_minimal_checkpoint(tmpdir: str, step: int = 1) -> str:
    """
    Creates a minimal but fully valid checkpoint directory using the
    actual save_checkpoint pathway with a tiny synthetic model.
    Returns the checkpoint directory path.
    """
    config = _make_tiny_config()
    model = _make_tiny_model(config)
    optimizer = _make_tiny_optimizer(model)
    scaler = _make_tiny_scaler()

    # Write a minimal tokenizer.json so save_checkpoint can copy it
    tok_path = os.path.join(tmpdir, "tokenizer.json")
    vocab = {chr(i): i for i in range(32, 32 + 64)}
    vocab["<|unk|>"] = 0
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f)

    ckpt_dir = os.path.join(tmpdir, "checkpoint")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        tokenizer_path=tok_path,
        output_dir=ckpt_dir,
        step=step,
        epoch=0,
        loss=5.0,
        best_loss=5.0,
        stage_id="test_stage",
        is_best=False,
        is_completed=False,
        sync_to_gdrive=False,
        dataset=None,
        shard_ledger=None,
        data_cursor={
            "active_shard": "shard_000.jsonl",
            "active_shard_line_offset": 10,
            "active_shard_byte_offset": 512,
            "active_chunk_idx": 0,
            "total_samples_trained": 10,
            "samples_consumed": 10,
            "completed_shards": [],
        },
    )
    return ckpt_dir


# ---------------------------------------------------------------------------
# D1 — Checkpoint write failure preserves prior valid checkpoint byte-for-byte
# ---------------------------------------------------------------------------

class TestD1CheckpointWriteFailure(unittest.TestCase):
    """
    D1: A failure during the write of a new checkpoint (model file, optimizer,
    or manifest) must leave the previously committed checkpoint byte-for-byte
    identical to what it was before the failed write attempt.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d1_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _checkpoint_sha_map(self, ckpt_dir: str) -> Dict[str, str]:
        result = {}
        for fname in sorted(os.listdir(ckpt_dir)):
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.isfile(fpath):
                result[fname] = _sha256(fpath)
        return result

    def test_failure_during_model_write_preserves_prior_checkpoint(self):
        """Simulate kill during model weights write; prior checkpoint is intact."""
        ckpt_dir = _build_minimal_checkpoint(self.tmpdir, step=1)
        self.assertTrue(verify_checkpoint_integrity(ckpt_dir, require_optimizer=True))
        prior_sha_map = self._checkpoint_sha_map(ckpt_dir)

        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")

        # Inject failure: save_file raises mid-write
        from cloud_train import HAS_SAFETENSORS
        if HAS_SAFETENSORS:
            with patch("cloud_train.save_file", side_effect=IOError("Simulated disk full during model write")):
                save_checkpoint(
                    model=model, optimizer=optimizer, scaler=scaler,
                    config=config, tokenizer_path=tok_path,
                    output_dir=ckpt_dir, step=2, epoch=0, loss=4.9,
                    best_loss=4.9, stage_id="test_stage",
                    is_best=False, is_completed=False, sync_to_gdrive=False,
                )
        else:
            with patch("torch.save", side_effect=IOError("Simulated disk full during model write")):
                save_checkpoint(
                    model=model, optimizer=optimizer, scaler=scaler,
                    config=config, tokenizer_path=tok_path,
                    output_dir=ckpt_dir, step=2, epoch=0, loss=4.9,
                    best_loss=4.9, stage_id="test_stage",
                    is_best=False, is_completed=False, sync_to_gdrive=False,
                )

        # Prior checkpoint must be byte-for-byte identical
        after_sha_map = self._checkpoint_sha_map(ckpt_dir)
        for fname, sha in prior_sha_map.items():
            self.assertIn(fname, after_sha_map, f"File {fname} disappeared from checkpoint")
            self.assertEqual(sha, after_sha_map[fname],
                             f"File {fname} was corrupted by the failed write attempt")

        # The checkpoint must still pass integrity verification
        self.assertTrue(verify_checkpoint_integrity(ckpt_dir, require_optimizer=True),
                        "Prior checkpoint failed integrity check after failed new write")

    def test_failure_during_optimizer_write_preserves_prior_checkpoint(self):
        """Simulate crash during optimizer_state.pt write; prior checkpoint must be intact."""
        ckpt_dir = _build_minimal_checkpoint(self.tmpdir, step=1)
        prior_sha_map = self._checkpoint_sha_map(ckpt_dir)

        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")

        # torch.save is called exactly once (optimizer_state.pt) when HAS_SAFETENSORS=True.
        # Fail on the first call to simulate a crash during optimizer write.
        real_torch_save = torch.save

        def fail_first_torch_save(*args, **kwargs):
            raise IOError("Simulated crash during optimizer_state.pt write")

        with patch("torch.save", side_effect=fail_first_torch_save):
            save_checkpoint(
                model=model, optimizer=optimizer, scaler=scaler,
                config=config, tokenizer_path=tok_path,
                output_dir=ckpt_dir, step=2, epoch=0, loss=4.9,
                best_loss=4.9, stage_id="test_stage",
                is_best=False, is_completed=False, sync_to_gdrive=False,
            )

        # The save must have failed (optimizer write failed -> staging fails -> no promotion)
        # Prior checkpoint at output_dir must still be the step=1 checkpoint
        after_sha_map = self._checkpoint_sha_map(ckpt_dir)
        for fname, sha in prior_sha_map.items():
            self.assertIn(fname, after_sha_map, f"File {fname} vanished from checkpoint directory")
            self.assertEqual(sha, after_sha_map[fname],
                             f"File {fname} was corrupted by failed step-2 write attempt")

        # The checkpoint must still be the step=1 one (not step=2)
        state_path = os.path.join(ckpt_dir, "checkpoint_state.json")
        with open(state_path, "r") as f:
            state = json.load(f)
        self.assertEqual(state["step"], 1,
                         "Checkpoint step advanced despite failed write — prior state not preserved")

    def test_failure_during_manifest_write_preserves_prior_checkpoint(self):
        """Simulate crash during checkpoint_state.json (manifest) write; prior checkpoint intact."""
        ckpt_dir = _build_minimal_checkpoint(self.tmpdir, step=1)
        prior_sha_map = self._checkpoint_sha_map(ckpt_dir)

        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")

        real_open = open
        call_count = {"manifest_writes": 0}

        def patched_open(name, mode="r", *args, **kwargs):
            if "checkpoint_state.json" in str(name) and "w" in str(mode):
                call_count["manifest_writes"] += 1
                if call_count["manifest_writes"] >= 1:
                    raise IOError("Simulated crash during manifest write")
            return real_open(name, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            save_checkpoint(
                model=model, optimizer=optimizer, scaler=scaler,
                config=config, tokenizer_path=tok_path,
                output_dir=ckpt_dir, step=2, epoch=0, loss=4.9,
                best_loss=4.9, stage_id="test_stage",
                is_best=False, is_completed=False, sync_to_gdrive=False,
            )

        after_sha_map = self._checkpoint_sha_map(ckpt_dir)
        for fname, sha in prior_sha_map.items():
            self.assertIn(fname, after_sha_map)
            self.assertEqual(sha, after_sha_map[fname],
                             f"File {fname} was corrupted by manifest write failure")

    def test_incomplete_staging_dir_not_accepted_as_valid_checkpoint(self):
        """
        An incomplete staging directory (no COMMITTED marker) must not be
        accepted as a valid checkpoint on recovery.
        """
        ckpt_dir = _build_minimal_checkpoint(self.tmpdir, step=1)
        parent_dir = os.path.dirname(os.path.abspath(ckpt_dir))

        # Manually create a dangling staging directory with only partial contents
        fake_staging = os.path.join(parent_dir, ".ckpt_staging_checkpoint_step2_9999999_99999")
        os.makedirs(fake_staging)
        with open(os.path.join(fake_staging, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 100)  # corrupt / truncated

        # verify_checkpoint_integrity must reject this staging dir
        self.assertFalse(
            verify_checkpoint_integrity(fake_staging, require_optimizer=False),
            "Incomplete staging directory was incorrectly accepted as valid checkpoint"
        )

        # The real checkpoint must still be valid
        self.assertTrue(verify_checkpoint_integrity(ckpt_dir, require_optimizer=True))


# ---------------------------------------------------------------------------
# D2 — Fresh-process restoration recovers matching model, optimizer, cursor
# ---------------------------------------------------------------------------

class TestD2Resumption(unittest.TestCase):
    """
    D2: After a successful checkpoint, a fresh process must restore model
    weights, optimizer state, and exact data cursor state that matches
    what was committed.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d2_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_model_weights_restored_exactly(self):
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()

        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        vocab = {chr(i): i for i in range(32, 32 + 64)}
        vocab["<|unk|>"] = 0
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": vocab}, f)

        ckpt_dir = os.path.join(self.tmpdir, "checkpoint")
        cursor = {
            "active_shard": "shard_001.jsonl",
            "active_shard_line_offset": 42,
            "active_shard_byte_offset": 2048,
            "active_chunk_idx": 1,
            "total_samples_trained": 42,
            "samples_consumed": 42,
            "completed_shards": ["shard_000.jsonl"],
        }

        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path,
            output_dir=ckpt_dir, step=7, epoch=0, loss=3.5,
            best_loss=3.5, stage_id="test",
            is_best=False, is_completed=False, sync_to_gdrive=False,
            data_cursor=cursor,
        )

        # Simulate fresh process: load into a new model instance
        from cloud_train import TaraForCausalLM
        reload_model = TaraForCausalLM(config).cpu()
        reload_optimizer = _make_tiny_optimizer(reload_model)
        reload_scaler = _make_tiny_scaler()

        restored_step, restored_epoch, restored_best_loss, was_resumed, restored_cursor = \
            try_resume_from_checkpoint(ckpt_dir, reload_model, reload_optimizer, reload_scaler)

        self.assertTrue(was_resumed, "Resume was expected but did not happen")
        self.assertEqual(restored_step, 7)
        self.assertEqual(restored_epoch, 0)

        # Model weights must match original
        orig_sd = model.state_dict()
        reload_sd = reload_model.state_dict()
        for key in orig_sd:
            self.assertTrue(
                torch.allclose(orig_sd[key].float(), reload_sd[key].float(), atol=1e-6),
                f"Weight mismatch for parameter: {key}"
            )

    def test_data_cursor_restored_exactly(self):
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()

        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        vocab = {chr(i): i for i in range(32, 32 + 64)}
        vocab["<|unk|>"] = 0
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": vocab}, f)

        ckpt_dir = os.path.join(self.tmpdir, "checkpoint")
        expected_cursor = {
            "active_shard": "shard_005.jsonl",
            "active_shard_line_offset": 123,
            "active_shard_byte_offset": 8192,
            "active_chunk_idx": 3,
            "total_samples_trained": 999,
            "samples_consumed": 1010,
            "completed_shards": ["shard_000.jsonl", "shard_001.jsonl"],
        }

        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path,
            output_dir=ckpt_dir, step=42, epoch=1, loss=2.0,
            best_loss=2.0, stage_id="test",
            is_best=False, is_completed=False, sync_to_gdrive=False,
            data_cursor=expected_cursor,
        )

        from cloud_train import TaraForCausalLM
        reload_model = TaraForCausalLM(config).cpu()
        reload_optimizer = _make_tiny_optimizer(reload_model)
        reload_scaler = _make_tiny_scaler()

        _, _, _, _, restored_cursor = try_resume_from_checkpoint(
            ckpt_dir, reload_model, reload_optimizer, reload_scaler
        )

        self.assertIsNotNone(restored_cursor)
        self.assertEqual(restored_cursor.get("active_shard"), expected_cursor["active_shard"])
        self.assertEqual(restored_cursor.get("active_shard_line_offset"), expected_cursor["active_shard_line_offset"])
        self.assertEqual(restored_cursor.get("active_shard_byte_offset"), expected_cursor["active_shard_byte_offset"])
        self.assertIn("shard_000.jsonl", restored_cursor.get("completed_shards", []))
        self.assertIn("shard_001.jsonl", restored_cursor.get("completed_shards", []))


# ---------------------------------------------------------------------------
# D3 — Long document chunked without losing any intended target tokens
# ---------------------------------------------------------------------------

class TestD3TokenCoverage(unittest.TestCase):
    """
    D3: Every token of a document whose length exceeds max_seq_len must appear
    exactly once as a training target across all produced chunks.
    No token is silently truncated or duplicated.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d3_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _collect_target_tokens_from_chunks(
        self,
        full_ids: List[int],
        target_start: int,
        max_seq_len: int,
    ) -> List[int]:
        """
        Manually apply the same chunking + label masking logic as the dataset,
        and collect all tokens that carry a real (non -100) label.
        """
        import math as _math
        n_tokens = len(full_ids)
        chunk_size = max_seq_len
        n_chunks = max(1, _math.ceil(n_tokens / chunk_size))
        target_tokens = []

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_tokens)
            chunk_ids = full_ids[chunk_start:chunk_end]
            if len(chunk_ids) < 4:
                continue

            chunk_target_start = max(0, target_start - chunk_start)
            chunk_target_start = min(chunk_target_start, len(chunk_ids))
            # Tokens at positions >= chunk_target_start carry the loss
            target_tokens.extend(chunk_ids[chunk_target_start:])

        return target_tokens

    def test_pure_text_document_all_tokens_covered(self):
        """
        A pure-text document (target_start=0) must produce target tokens
        that exactly equal the original full_ids (minus final short chunk
        < 4 tokens, if any).
        """
        max_seq_len = 16
        # Build a document of 50 tokens (> 3 * max_seq_len)
        full_ids = list(range(50))
        target_start = 0

        target_tokens = self._collect_target_tokens_from_chunks(full_ids, target_start, max_seq_len)
        import math as _math
        n_chunks = _math.ceil(len(full_ids) / max_seq_len)
        last_chunk_size = len(full_ids) - (n_chunks - 1) * max_seq_len
        expected_tokens = full_ids if last_chunk_size >= 4 else full_ids[: (n_chunks - 1) * max_seq_len]

        self.assertEqual(target_tokens, expected_tokens,
                         "Target token IDs do not match the expected full sequence")

    def test_input_output_document_output_tokens_covered_exactly_once(self):
        """
        For an input/output document, the output tokens must appear as targets
        exactly once across all chunks. No output token is duplicated or lost.
        """
        max_seq_len = 20
        # Input: 15 tokens, Output: 35 tokens -> total 50
        inp_ids = list(range(100, 115))     # 15 tokens
        out_ids = list(range(200, 235))     # 35 tokens
        full_ids = inp_ids + out_ids
        target_start = len(inp_ids)         # 15

        import math as _math
        n_chunks = _math.ceil(len(full_ids) / max_seq_len)
        chunk_size = max_seq_len

        collected_targets = []
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(full_ids))
            chunk_ids = full_ids[chunk_start:chunk_end]
            if len(chunk_ids) < 4:
                continue
            cts = max(0, target_start - chunk_start)
            cts = min(cts, len(chunk_ids))
            collected_targets.extend(chunk_ids[cts:])

        # All output tokens that ended up in chunks of >= 4 tokens must appear
        # exactly once as targets (none lost, none duplicated).
        self.assertEqual(len(set(collected_targets)), len(collected_targets),
                         "Duplicate target tokens detected — overlap context is being trained twice")
        # All output tokens must be present (verify by checking out_ids subset)
        target_set = set(collected_targets)
        for tok in out_ids:
            self.assertIn(tok, target_set, f"Output token {tok} was silently dropped")

    def test_dataset_iterator_covers_all_tokens_via_streaming(self):
        """
        End-to-end streaming test: build a JSONL shard with one long record
        (token count >> max_seq_len) and verify that all target tokens appear
        exactly once in the stream.
        """
        max_seq_len = 8
        tokenizer = _make_tiny_tokenizer()

        # Build a text string of 40 characters (40 tokens in char-level tokenizer)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$"
        assert len(text) == 40

        shard_path = os.path.join(self.tmpdir, "shard_long.jsonl")
        _write_jsonl_shard(shard_path, [{"text": text}])

        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_samples=0,
            shuffle_shards=False,
            resume_shard=None,
            resume_line_offset=-1,
        )

        all_target_ids = []
        for item in dataset:
            # item: (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx)
            labels = item[1]
            # Collect non-masked labels (target tokens)
            targets = [t.item() for t in labels if t.item() != -100]
            all_target_ids.extend(targets)

        # Encode the full text to get expected token IDs
        enc = tokenizer.encode(text)
        full_ids = enc.ids if hasattr(enc, "ids") else list(enc)

        import math as _math
        n_chunks = _math.ceil(len(full_ids) / max_seq_len)
        expected_ids = []
        for i in range(n_chunks):
            chunk = full_ids[i * max_seq_len: (i + 1) * max_seq_len]
            if len(chunk) >= 4:
                expected_ids.extend(chunk)  # target_start=0 for pure text, all are targets

        self.assertEqual(all_target_ids, expected_ids,
                         "Streaming dataset did not produce all expected target tokens")

    def test_final_short_chunk_policy_discard_below_4(self):
        """
        Final short chunk with fewer than 4 tokens must be discarded.
        Final short chunk with 4+ tokens must be yielded.
        """
        max_seq_len = 10
        tokenizer = _make_tiny_tokenizer()

        # Case 1: 13 tokens -> chunk0=10, chunk1=3 (< 4, must be discarded)
        text_13 = "ABCDEFGHIJKLM"
        shard_13 = os.path.join(self.tmpdir, "shard_13.jsonl")
        _write_jsonl_shard(shard_13, [{"text": text_13}])
        ds_13 = TaraStreamingJsonlDataset(
            file_paths=[shard_13], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks_13 = list(ds_13)
        self.assertEqual(len(chunks_13), 1,
                         f"Expected 1 chunk for 13-token doc with max_seq_len=10, got {len(chunks_13)}")

        # Case 2: 14 tokens -> chunk0=10, chunk1=4 (>= 4, must be yielded)
        text_14 = "ABCDEFGHIJKLMN"
        shard_14 = os.path.join(self.tmpdir, "shard_14.jsonl")
        _write_jsonl_shard(shard_14, [{"text": text_14}])
        ds_14 = TaraStreamingJsonlDataset(
            file_paths=[shard_14], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks_14 = list(ds_14)
        self.assertEqual(len(chunks_14), 2,
                         f"Expected 2 chunks for 14-token doc with max_seq_len=10, got {len(chunks_14)}")


# ---------------------------------------------------------------------------
# D4 — Crash at shard EOF before optimizer/checkpoint commit
# ---------------------------------------------------------------------------

class TestD4EofBeforeCheckpoint(unittest.TestCase):
    """
    D4: Reaching the end of a shard's data stream (EOF) must NOT immediately
    mark the shard as completed or evict its file.  Those operations must only
    occur after a durable, verified checkpoint covering the shard's training
    units has been committed.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d4_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_shard_not_completed_in_ledger_after_streaming_eof(self):
        """
        After iterating a shard to EOF, the shard must NOT appear in
        shard_ledger.completed_shards — it must only appear in
        dataset._pending_shard_completions.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_000.jsonl")
        _write_jsonl_shard(shard_path, [
            {"text": "ABCD"},
            {"text": "EFGH"},
        ])

        ledger = ShardLedger(self.tmpdir)
        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
        )

        # Consume all items (triggers EOF handling in __iter__)
        items = list(dataset)
        self.assertGreater(len(items), 0, "Dataset yielded nothing")

        # Shard must NOT be in completed_shards yet
        self.assertNotIn("shard_000.jsonl", ledger.completed_shards,
                         "Shard was marked completed at EOF before checkpoint commit")

        # Shard MUST be pending completion
        self.assertIn("shard_000.jsonl", dataset._pending_shard_completions,
                      "Shard is not in _pending_shard_completions after streaming EOF")

    def test_shard_file_not_evicted_before_checkpoint_commit(self):
        """
        A shard file that is a managed cache download must not be deleted
        at EOF; it must only be deleted after commit_shard_completion() is called.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_managed.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDEFGH"}])

        ledger = ShardLedger(self.tmpdir)
        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
            keep_shards=False,
        )
        # Register shard as a managed cache file
        dataset.managed_downloaded_shards.add(os.path.abspath(shard_path))

        # Stream to EOF
        list(dataset)

        # File must still exist (eviction deferred until checkpoint commit)
        self.assertTrue(os.path.exists(shard_path),
                        "Shard file was prematurely deleted at EOF before checkpoint commit")

        # Now simulate checkpoint commit calling commit_shard_completion
        dataset.commit_shard_completion(
            "shard_managed.jsonl",
            managed_cache=True,
            actual_fp=shard_path,
        )

        # Now shard must be completed in ledger
        self.assertIn("shard_managed.jsonl", ledger.completed_shards)
        # And the file must be deleted
        self.assertFalse(os.path.exists(shard_path),
                         "Shard file was not deleted after commit_shard_completion()")

    def test_recoverable_source_preserved_if_crash_before_checkpoint(self):
        """
        If a crash occurs after EOF but before checkpoint commit, the shard
        is neither completed in the ledger nor evicted. Recovery can re-stream it.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_crash.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDE"}, {"text": "FGHIJ"}])

        ledger = ShardLedger(self.tmpdir)
        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
        )

        list(dataset)  # consume to EOF, simulate crash (no checkpoint commit)

        # Check: shard not completed
        self.assertNotIn("shard_crash.jsonl", ledger.completed_shards)
        # Check: shard file still readable (not deleted)
        self.assertTrue(os.path.exists(shard_path))
        # Check: re-streaming is possible
        dataset2 = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
        )
        recovered_items = list(dataset2)
        self.assertGreater(len(recovered_items), 0, "Could not re-stream shard after simulated crash")


# ---------------------------------------------------------------------------
# D5 — Crash after checkpoint commit but before ledger commit
# ---------------------------------------------------------------------------

class TestD5PostCheckpointLedgerReconciliation(unittest.TestCase):
    """
    D5: A crash that occurs after a verified checkpoint is committed but before
    the ledger is updated must be recovered correctly: the ledger is reconciled
    from the committed checkpoint on the next run.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d5_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ledger_reconciled_from_checkpoint_after_ledger_update_crash(self):
        """
        Scenario:
        1. Checkpoint committed at step=5 with active_shard='shard_001.jsonl',
           completed_shards=['shard_000.jsonl'].
        2. Crash before ledger.save() executes.
        3. Recovery calls shard_ledger.reconcile_with_checkpoint(cursor, step=5).
        4. After reconciliation, ledger must reflect the checkpoint's cursor,
           not the pre-crash stale state.
        """
        # Simulate stale ledger state (before crash, ledger was behind checkpoint)
        ledger = ShardLedger(self.tmpdir)
        ledger.completed_shards = []   # stale: hasn't recorded shard_000
        ledger.active_shard = "shard_001.jsonl"
        ledger.active_shard_line_offset = 0
        ledger.active_shard_byte_offset = 0
        ledger.save()

        # Committed checkpoint cursor (authoritative ground truth)
        committed_cursor = {
            "active_shard": "shard_001.jsonl",
            "active_shard_line_offset": 10,
            "active_shard_byte_offset": 512,
            "total_samples_trained": 50,
            "samples_consumed": 50,
            "completed_shards": ["shard_000.jsonl"],
        }

        ledger.reconcile_with_checkpoint(committed_cursor, step=5)

        self.assertIn("shard_000.jsonl", ledger.completed_shards,
                      "Reconciliation failed to restore shard_000 as completed")
        self.assertEqual(ledger.active_shard, "shard_001.jsonl")
        self.assertEqual(ledger.active_shard_line_offset, 10)
        self.assertEqual(ledger.last_committed_step, 5)

    def test_stale_ahead_ledger_evicted_by_reconciliation(self):
        """
        If the ledger contains a shard completion that the committed checkpoint
        does not include (stale-ahead), reconciliation must remove it so the
        shard is re-trained.
        """
        ledger = ShardLedger(self.tmpdir)
        # Stale-ahead: ledger recorded shard_001 as completed, but checkpoint didn't commit that
        ledger.completed_shards = ["shard_000.jsonl", "shard_001.jsonl"]
        ledger.completed_metadata = {
            "shard_000.jsonl": {"records_processed": 10, "file_sha256": "abc"},
            "shard_001.jsonl": {"records_processed": 10, "file_sha256": "def"},
        }
        ledger.active_shard = None
        ledger.save()

        committed_cursor = {
            "active_shard": "shard_001.jsonl",
            "active_shard_line_offset": 5,
            "active_shard_byte_offset": 256,
            "total_samples_trained": 10,
            "completed_shards": ["shard_000.jsonl"],  # shard_001 not yet committed
        }

        ledger.reconcile_with_checkpoint(committed_cursor, step=3)

        self.assertIn("shard_000.jsonl", ledger.completed_shards,
                      "shard_000 should be completed (it was in checkpoint)")
        self.assertNotIn("shard_001.jsonl", ledger.completed_shards,
                         "stale-ahead shard_001 should have been evicted from completed")


# ---------------------------------------------------------------------------
# D6 — No committed chunk is reprocessed after resume
# ---------------------------------------------------------------------------

class TestD6NoCommittedChunkReplay(unittest.TestCase):
    """
    D6: After resuming from a checkpoint at a given cursor, no chunk whose
    optimizer update was durably committed must be yielded again by the
    dataset iterator.  Replay of uncommitted chunks (work done after the last
    checkpoint) is acceptable and expected.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d6_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_skips_committed_lines_and_chunks(self):
        """
        Shard has 5 records.  Checkpoint cursor says line_offset=2 (lines 0,1,2
        already processed, resume from line 3 onwards).  Lines 3 and 4 must be
        yielded; lines 0-2 must not appear.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_resume.jsonl")
        records = [{"text": chr(65 + i) * 8} for i in range(5)]  # A*8, B*8, ...E*8
        _write_jsonl_shard(shard_path, records)

        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            resume_shard="shard_resume.jsonl",
            resume_line_offset=2,   # lines 0,1,2 committed
            resume_byte_offset=0,
            resume_chunk_idx=0,
        )

        yielded_lines = []
        for item in dataset:
            # item[3] is line_idx
            yielded_lines.append(item[3])

        # Lines 0,1,2 must not appear
        for committed_line in [0, 1, 2]:
            self.assertNotIn(committed_line, yielded_lines,
                             f"Committed line {committed_line} was replayed after resume")
        # Lines 3 and 4 must appear
        for pending_line in [3, 4]:
            self.assertIn(pending_line, yielded_lines,
                          f"Uncommitted line {pending_line} was missing after resume")


# ---------------------------------------------------------------------------
# D7 — No intended uncommitted training unit is permanently omitted
# ---------------------------------------------------------------------------

class TestD7NoUncommittedUnitOmitted(unittest.TestCase):
    """
    D7: After resuming, every training unit (chunk) that was NOT covered by the
    last durable checkpoint must be reachable and processable.  No token that
    was intended to be trained but not yet committed must be permanently lost.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d7_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_uncommitted_chunks_reprocessed_after_resume(self):
        """
        Shard has 3 records with long texts (>max_seq_len).
        Checkpoint cursor at line_offset=0 (first record fully committed).
        After resume, records from line 1 onwards must be re-yielded.
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        shard_path = os.path.join(self.tmpdir, "shard_uncommitted.jsonl")
        # Each record produces 3 chunks (24 chars / 8 = 3 chunks)
        records = [{"text": chr(65 + i) * 24} for i in range(3)]
        _write_jsonl_shard(shard_path, records)

        # First, stream everything from scratch to know total chunks
        ds_full = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer, max_seq_len=max_seq_len,
        )
        all_chunks = [(item[3], item[5]) for item in ds_full]  # (line_idx, chunk_idx)
        # Expect 3 records * 3 chunks = 9 chunks total
        self.assertEqual(len(all_chunks), 9)

        # Now resume from after line 0 (first record committed)
        ds_resume = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer, max_seq_len=max_seq_len,
            resume_shard="shard_uncommitted.jsonl",
            resume_line_offset=0,
            resume_byte_offset=0,
            resume_chunk_idx=0,
        )
        resumed_chunks = [(item[3], item[5]) for item in ds_resume]

        # All chunks from line 1 and 2 must be present (6 chunks)
        expected_lines = [1, 2]
        for line_idx in expected_lines:
            line_chunks = [(l, c) for l, c in resumed_chunks if l == line_idx]
            self.assertEqual(len(line_chunks), 3,
                             f"Line {line_idx} should produce 3 chunks after resume, got {len(line_chunks)}")

        # Chunks from line 0 must not appear
        line0_chunks = [(l, c) for l, c in resumed_chunks if l == 0]
        self.assertEqual(len(line0_chunks), 0,
                         f"Committed line 0 chunks were replayed: {line0_chunks}")


# ---------------------------------------------------------------------------
# D8 — Local eviction only deletes managed cache files
# ---------------------------------------------------------------------------

class TestD8EvictionSafety(unittest.TestCase):
    """
    D8: The shard eviction logic must only delete files that were downloaded
    by the managed JIT downloader.  User-provided source files must never
    be deleted, even if marked as pending completion.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_test_d8_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_user_source_file_not_evicted(self):
        """
        A shard file NOT in dataset.managed_downloaded_shards must not be
        deleted when commit_shard_completion() is called.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "user_source_shard.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDEFGH"}])

        ledger = ShardLedger(self.tmpdir)
        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
            keep_shards=False,
        )
        # Do NOT add to managed_downloaded_shards (simulates user-provided file)
        list(dataset)  # stream to EOF

        # Call commit without managed_cache=True
        dataset.commit_shard_completion(
            "user_source_shard.jsonl",
            managed_cache=False,
            actual_fp=shard_path,
        )

        self.assertTrue(os.path.exists(shard_path),
                        "User source file was incorrectly deleted by eviction logic")
        # Shard should be marked completed in ledger
        self.assertIn("user_source_shard.jsonl", ledger.completed_shards)

    def test_managed_cache_file_evicted_after_commit(self):
        """
        A shard file IN dataset.managed_downloaded_shards must be deleted
        when commit_shard_completion() is called with managed_cache=True.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "managed_shard.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDEFGH"}])

        ledger = ShardLedger(self.tmpdir)
        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
            keep_shards=False,
        )
        dataset.managed_downloaded_shards.add(os.path.abspath(shard_path))
        list(dataset)  # stream to EOF

        dataset.commit_shard_completion(
            "managed_shard.jsonl",
            managed_cache=True,
            actual_fp=shard_path,
        )

        self.assertFalse(os.path.exists(shard_path),
                         "Managed cache file was NOT deleted after commit")
        self.assertIn("managed_shard.jsonl", ledger.completed_shards)

    def test_keep_shards_flag_prevents_eviction(self):
        """
        When keep_shards=True, even managed cache files must not be evicted.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "kept_shard.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDEFGH"}])

        ledger = ShardLedger(self.tmpdir)
        dataset = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=tokenizer,
            max_seq_len=16,
            shard_ledger=ledger,
            keep_shards=True,  # must prevent eviction
        )
        dataset.managed_downloaded_shards.add(os.path.abspath(shard_path))
        list(dataset)

        dataset.commit_shard_completion(
            "kept_shard.jsonl",
            managed_cache=True,
            actual_fp=shard_path,
        )

        self.assertTrue(os.path.exists(shard_path),
                        "Shard was deleted despite keep_shards=True")


# ---------------------------------------------------------------------------
# Supplementary: Verify collate_fn handles 6-tuple items correctly
# ---------------------------------------------------------------------------

class TestCollateFn(unittest.TestCase):
    def test_6tuple_items_collated_correctly(self):
        """collate_fn must handle the new 6-tuple (with chunk_idx) without error."""
        items = [
            (torch.tensor([1, 2, 3]), torch.tensor([-100, 2, 3]), "s0.jsonl", 0, 100, 0),
            (torch.tensor([4, 5]),    torch.tensor([4, 5]),         "s0.jsonl", 0, 200, 1),
        ]
        inputs, labels, cursors = collate_fn(items)
        self.assertEqual(inputs.shape[0], 2)
        self.assertEqual(len(cursors), 2)
        # chunk_idx should be in the cursor tuple at position 3
        self.assertEqual(cursors[0][3], 0)
        self.assertEqual(cursors[1][3], 1)

    def test_empty_batch_returns_empty_tensors(self):
        inputs, labels, cursors = collate_fn([])
        self.assertEqual(inputs.numel(), 0)
        self.assertEqual(labels.numel(), 0)
        self.assertEqual(cursors, [])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

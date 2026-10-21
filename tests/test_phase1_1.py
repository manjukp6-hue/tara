#!/usr/bin/env python3
"""
Phase 1.1 Acceptance Gate — Regression Tests for TARA Cloud Training Engine.

These tests address the six blocking issues identified in the Phase 1.1
acceptance review and prove that every gate has a genuine passing test.

Gate inventory:
  G1 — Token/chunk completeness (short docs, exact-length, long, chunk label masking,
        short-doc total discard, all-input chunk all-masked labels).
  G2 — Ordering and cursor correctness (buffer_size > 1 + resume warning, cursor
        cannot advance past uncommitted chunk, deterministic sequential order at buffer=1).
  G3 — Checkpoint durability (resume prefers versioned dir via pointer over output_dir;
        crash mid-copy into output_dir leaves prior versioned dir recoverable).
  G4 — Checkpoint/ledger consistency (in-memory _pending_shard_completions lost on
        restart causes shard re-stream not loss; stale-behind and stale-ahead ledgers).
  G5 — Best Model safety (incomplete best_checkpoint rejected; optimizer resume
        never selects a best_checkpoint without optimizer_state.pt;
        committed best checkpoint has COMMITTED marker).
  G6 — Storage retention (retention policy keeps keep_n, deletes older,
        never deletes before new is committed, uncommitted staging dirs removed).

All tests are isolated, use only synthetic fixtures (temp dirs), and never
touch production checkpoints, datasets, or Google Drive artifacts.
No GPU training, no Hugging Face downloads.
"""

import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

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
    _collect_versioned_checkpoint_dirs,
    _cleanup_old_versioned_checkpoints,
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_phase1.py helpers — no code duplication)
# ---------------------------------------------------------------------------

def _make_tiny_config() -> dict:
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
    return TaraForCausalLM(config or _make_tiny_config()).cpu()


def _make_tiny_tokenizer() -> TaraNativeTokenizer:
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
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_tokenizer(path: str):
    vocab = {chr(i): i for i in range(32, 32 + 64)}
    vocab["<|unk|>"] = 0
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f)


def _build_checkpoint(tmpdir: str, step: int, cursor: dict = None) -> str:
    """Build a fully valid checkpoint at output_dir=<tmpdir>/ckpt using save_checkpoint."""
    config = _make_tiny_config()
    model = _make_tiny_model(config)
    optimizer = _make_tiny_optimizer(model)
    scaler = _make_tiny_scaler()
    tok_path = os.path.join(tmpdir, "tokenizer.json")
    if not os.path.exists(tok_path):
        _write_tokenizer(tok_path)
    ckpt_dir = os.path.join(tmpdir, "ckpt")
    if cursor is None:
        cursor = {
            "active_shard": "shard_000.jsonl",
            "active_shard_line_offset": step,
            "active_shard_byte_offset": step * 64,
            "active_chunk_idx": 0,
            "total_samples_trained": step,
            "samples_consumed": step,
            "completed_shards": [],
        }
    save_checkpoint(
        model=model, optimizer=optimizer, scaler=scaler,
        config=config, tokenizer_path=tok_path,
        output_dir=ckpt_dir, step=step, epoch=0,
        loss=5.0 - step * 0.1, best_loss=5.0 - step * 0.1,
        stage_id="test_stage",
        is_best=False, is_completed=False, sync_to_gdrive=False,
        data_cursor=cursor,
    )
    return ckpt_dir


# =============================================================================
# G1 — Token / chunk completeness
# =============================================================================

class TestG1TokenChunkCompleteness(unittest.TestCase):
    """
    Gate 1: Verify token coverage, label masking, short-document handling,
    exact-length documents, and the all-input-chunk all-masked-labels scenario.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g1_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_short_doc_under_4_tokens_entirely_discarded(self):
        """
        A document with fewer than 4 tokens must yield zero chunks.
        The entire document is silently discarded by the < 4 tokens policy.
        """
        tokenizer = _make_tiny_tokenizer()
        # "ABC" = 3 tokens — must be discarded entirely
        shard_path = os.path.join(self.tmpdir, "shard_short.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABC"}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=16, max_samples=0,
        )
        chunks = list(ds)
        self.assertEqual(len(chunks), 0,
                         f"Expected 0 chunks for 3-token doc, got {len(chunks)}")

    def test_exact_length_document_produces_one_full_chunk(self):
        """
        A document with token count == max_seq_len must yield exactly one chunk
        with all tokens as targets (for pure-text records).
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        text = "ABCDEFGH"  # exactly 8 chars = 8 tokens
        shard_path = os.path.join(self.tmpdir, "shard_exact.jsonl")
        _write_jsonl_shard(shard_path, [{"text": text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks = list(ds)
        self.assertEqual(len(chunks), 1,
                         f"Expected 1 chunk for exact-length doc, got {len(chunks)}")
        input_ids, labels, _, _, _, chunk_idx, _ = chunks[0]
        self.assertEqual(len(input_ids), max_seq_len)
        self.assertEqual(chunk_idx, 0)
        # All labels must be real (no masking for pure text: target_start=0)
        self.assertTrue(all(t.item() != -100 for t in labels),
                        "Pure-text exact-length chunk should have no masked labels")

    def test_long_document_all_chunks_covered_no_duplication(self):
        """
        A document with 3 * max_seq_len tokens must yield 3 chunks,
        and no token must appear in more than one chunk (non-overlapping).
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        text = "ABCDEFGHIJKLMNOPQRSTUVWX"  # 24 chars = 24 tokens
        shard_path = os.path.join(self.tmpdir, "shard_long.jsonl")
        _write_jsonl_shard(shard_path, [{"text": text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks = list(ds)
        self.assertEqual(len(chunks), 3, f"Expected 3 chunks, got {len(chunks)}")

        all_input_ids = []
        for item in chunks:
            all_input_ids.extend(item[0].tolist())

        # Token IDs must be unique across chunks (non-overlapping = no duplicates)
        self.assertEqual(len(all_input_ids), len(set(all_input_ids)),
                         "Duplicate token IDs found across chunks — chunks overlap")
        self.assertEqual(len(all_input_ids), 24,
                         "Total tokens across chunks does not equal document length")

    def test_all_input_span_chunk_has_all_labels_masked(self):
        """
        When a chunk falls entirely within the input span of a record
        (e.g., long input, short output), ALL labels in that chunk must be -100.
        The chunk is still yielded but contributes zero loss — this is correct.
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        # Input: 20 tokens, Output: 4 tokens => full_ids = 24 tokens
        # Chunk 0 (tokens 0-7) falls entirely in input span (target_start=20)
        # => chunk_target_start = max(0, 20-0) = 20, capped to len(chunk)=8 => all masked
        inp_text = "ABCDEFGHIJKLMNOPQRST"  # 20 chars
        out_text = "UVWX"  # 4 chars
        shard_path = os.path.join(self.tmpdir, "shard_allinput.jsonl")
        _write_jsonl_shard(shard_path, [{"input": inp_text, "output": out_text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks = list(ds)
        # chunk 0: tokens 0-7 (all input span) — all labels = -100
        chunk0_labels = chunks[0][1].tolist()
        self.assertTrue(all(l == -100 for l in chunk0_labels),
                        f"Chunk 0 should have all-masked labels but got: {chunk0_labels}")

    def test_output_tokens_appear_as_targets_exactly_once(self):
        """
        For a multi-chunk input/output record, every output token must appear
        as a non-masked label in exactly one chunk.  Zero duplication.
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        # Input: 4 tokens, Output: 16 tokens => 20 total, 3 chunks
        # chunk0: tokens 0-7 (4 input + 4 output), chunk1: tokens 8-15 (8 output),
        # chunk2: tokens 16-19 (4 output) — but 4 >= 4, so yielded
        inp_text = "ABCD"   # 4 tokens
        out_text = "EFGHIJKLMNOPQRST"  # 16 tokens
        shard_path = os.path.join(self.tmpdir, "shard_inpout.jsonl")
        _write_jsonl_shard(shard_path, [{"input": inp_text, "output": out_text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks = list(ds)

        target_label_ids = []
        for item in chunks:
            labels = item[1].tolist()
            target_label_ids.extend(t for t in labels if t != -100)

        # No output token must appear more than once
        self.assertEqual(len(target_label_ids), len(set(target_label_ids)),
                         "Duplicate output tokens as targets — overlap in label coverage")

        # Encode expected output tokens and verify coverage
        enc_out = tokenizer.encode(out_text)
        expected_out_ids = enc_out.ids if hasattr(enc_out, "ids") else list(enc_out)
        target_set = set(target_label_ids)
        for tok in expected_out_ids:
            self.assertIn(tok, target_set,
                          f"Output token {tok} was not covered as a training target")

    def test_final_short_chunk_exactly_4_tokens_is_yielded(self):
        """Boundary: final chunk of exactly 4 tokens must be yielded (>= 4 policy)."""
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 10
        # 14 tokens: chunk0=10 tokens, chunk1=4 tokens (exactly 4 — must be yielded)
        text = "ABCDEFGHIJKLMN"  # 14 chars
        shard_path = os.path.join(self.tmpdir, "shard_4tok.jsonl")
        _write_jsonl_shard(shard_path, [{"text": text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks = list(ds)
        self.assertEqual(len(chunks), 2,
                         f"Final chunk of exactly 4 tokens should be yielded, got {len(chunks)} chunks")
        chunk_idx_values = [item[5] for item in chunks]
        self.assertIn(1, chunk_idx_values, "chunk_idx=1 missing for second chunk")

    def test_final_short_chunk_3_tokens_is_discarded(self):
        """Boundary: final chunk of exactly 3 tokens must be discarded (< 4 policy)."""
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 10
        # 13 tokens: chunk0=10, chunk1=3 (< 4 — must be discarded)
        text = "ABCDEFGHIJKLM"
        shard_path = os.path.join(self.tmpdir, "shard_3tok.jsonl")
        _write_jsonl_shard(shard_path, [{"text": text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_samples=0,
        )
        chunks = list(ds)
        self.assertEqual(len(chunks), 1,
                         f"Final chunk of 3 tokens should be discarded, got {len(chunks)} chunks")


# =============================================================================
# G2 — Ordering and cursor correctness
# =============================================================================

class TestG2OrderingAndCursorCorrectness(unittest.TestCase):
    """
    Gate 2: Cursor semantics, buffer_size > 1 + resume warning, deterministic
    sequential ordering at buffer_size=1, chunk_idx in cursor.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g2_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_buffer_size_1_produces_deterministic_order(self):
        """
        With buffer_size=1, two independent iterations of the same shard must
        produce items in the same order (deterministic, no shuffling).
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_order.jsonl")
        records = [{"text": chr(65 + i) * 8} for i in range(5)]
        _write_jsonl_shard(shard_path, records)

        def _collect_line_order(buf_sz):
            ds = TaraStreamingJsonlDataset(
                file_paths=[shard_path], tokenizer=tokenizer,
                max_seq_len=16, buffer_size=buf_sz,
            )
            return [item[3] for item in ds]  # line_idx

        order1 = _collect_line_order(1)
        order2 = _collect_line_order(1)
        self.assertEqual(order1, order2,
                         "buffer_size=1 must produce identical order on two runs")
        # Must be sequential
        self.assertEqual(order1, sorted(order1),
                         "buffer_size=1 must yield items in sequential line order")

    def test_buffer_size_gt1_with_resume_emits_warning(self):
        """
        Instantiating TaraStreamingJsonlDataset with buffer_size > 1 and a
        non-None resume_shard must print a determinism warning.
        Any output that contains the word 'Warning' satisfies the check.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_warn.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDEFGH"}])

        import io, sys as _sys
        captured = io.StringIO()
        old_stdout = _sys.stdout
        _sys.stdout = captured
        try:
            TaraStreamingJsonlDataset(
                file_paths=[shard_path], tokenizer=tokenizer,
                max_seq_len=16, buffer_size=4,
                resume_shard="shard_warn.jsonl",
                resume_line_offset=0,
            )
        finally:
            _sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("Warning", output,
                      "Expected a determinism warning when buffer_size > 1 and resume_shard is set")

    def test_chunk_idx_in_yielded_cursor_is_correct(self):
        """
        For a document producing multiple chunks, the chunk_idx field in each
        yielded item must increase monotonically from 0.
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        # 24 tokens -> 3 chunks
        text = "ABCDEFGHIJKLMNOPQRSTUVWX"
        shard_path = os.path.join(self.tmpdir, "shard_chunkidx.jsonl")
        _write_jsonl_shard(shard_path, [{"text": text}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        chunks = list(ds)
        chunk_indices = [item[5] for item in chunks]
        self.assertEqual(chunk_indices, [0, 1, 2],
                         f"chunk_idx values should be [0,1,2], got {chunk_indices}")

    def test_resume_with_chunk_idx_skips_already_committed_chunks(self):
        """
        Resuming with resume_line_offset=1, resume_chunk_idx=1 must skip:
        - All of line 0 (committed)
        - chunk_idx 0 of line 1 (committed)
        And must yield chunk_idx >= 1 of line 1, and all chunks of lines >= 2.
        """
        tokenizer = _make_tiny_tokenizer()
        max_seq_len = 8
        # 3 records, each producing 3 chunks (24 tokens each)
        records = [{"text": chr(65 + i) * 24} for i in range(3)]
        shard_path = os.path.join(self.tmpdir, "shard_chunkresume.jsonl")
        _write_jsonl_shard(shard_path, records)

        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            resume_shard="shard_chunkresume.jsonl",
            resume_line_offset=1,
            resume_byte_offset=0,
            resume_chunk_idx=1,
        )
        items = list(ds)
        yielded = [(item[3], item[5]) for item in items]  # (line_idx, chunk_idx)

        # Line 0 must not appear at all
        line0 = [(l, c) for l, c in yielded if l == 0]
        self.assertEqual(line0, [], f"Committed line 0 appeared after resume: {line0}")

        # Line 1, chunk_idx 0 must not appear
        line1_chunk0 = [(l, c) for l, c in yielded if l == 1 and c == 0]
        self.assertEqual(line1_chunk0, [],
                         f"Committed chunk (line=1, chunk_idx=0) appeared after resume")

        # Line 1, chunk_idx >= 1 must appear
        line1_rest = [(l, c) for l, c in yielded if l == 1 and c >= 1]
        self.assertGreater(len(line1_rest), 0,
                           "Expected uncommitted chunks of line 1 (chunk_idx>=1) after resume")

        # All 3 chunks of line 2 must appear
        line2_chunks = [(l, c) for l, c in yielded if l == 2]
        self.assertEqual(len(line2_chunks), 3,
                         f"Line 2 should produce 3 chunks after resume, got {len(line2_chunks)}")


# =============================================================================
# G3 — Checkpoint durability
# =============================================================================

class TestG3CheckpointDurability(unittest.TestCase):
    """
    Gate 3: verify_checkpoint_integrity rejects incomplete versioned dirs;
    try_resume_from_checkpoint prefers versioned dir via pointer over output_dir;
    crash mid-copy into output_dir leaves prior versioned dir fully recoverable.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g3_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_uses_versioned_dir_via_pointer_not_output_dir(self):
        """
        After save_checkpoint completes, try_resume_from_checkpoint must find
        the checkpoint via latest_checkpoint.json -> versioned_dir, even when
        output_dir contains a partially-corrupted copy (simulate crash mid-copy
        by corrupting model.safetensors in output_dir after the save).
        The versioned dir must always be intact and take precedence.
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")

        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path,
            output_dir=ckpt_dir, step=5, epoch=0, loss=4.0, best_loss=4.0,
            stage_id="test", is_best=False, is_completed=False, sync_to_gdrive=False,
            data_cursor={
                "active_shard": "shard_000.jsonl",
                "active_shard_line_offset": 5,
                "active_shard_byte_offset": 320,
                "active_chunk_idx": 0,
                "total_samples_trained": 5,
                "samples_consumed": 5,
                "completed_shards": [],
            },
        )

        # Confirm versioned dir exists and pointer is present
        parent_dir = os.path.dirname(os.path.abspath(ckpt_dir))
        pointer_path = os.path.join(parent_dir, "latest_checkpoint.json")
        self.assertTrue(os.path.isfile(pointer_path),
                        "latest_checkpoint.json pointer was not written")
        with open(pointer_path) as f:
            ptr = json.load(f)
        versioned_dir = ptr["versioned_dir"]
        self.assertTrue(os.path.isdir(versioned_dir),
                        "Versioned dir referenced by pointer does not exist")
        committed = os.path.join(versioned_dir, "COMMITTED")
        self.assertTrue(os.path.isfile(committed),
                        "Versioned dir has no COMMITTED marker")

        # Simulate crash mid-copy: corrupt model.safetensors in output_dir
        model_in_output = os.path.join(ckpt_dir, "model.safetensors")
        with open(model_in_output, "wb") as f:
            f.write(b"\x00" * 16)  # truncated / corrupt

        # Now try_resume_from_checkpoint must still recover from versioned dir
        reload_model = _make_tiny_model(config)
        reload_optimizer = _make_tiny_optimizer(reload_model)
        reload_scaler = _make_tiny_scaler()
        step, epoch, best_loss, was_resumed, cursor = try_resume_from_checkpoint(
            ckpt_dir, reload_model, reload_optimizer, reload_scaler
        )
        self.assertTrue(was_resumed,
                        "Resume failed even though a valid versioned checkpoint exists")
        self.assertEqual(step, 5,
                         "Resumed step does not match committed checkpoint step")
        self.assertEqual(cursor.get("active_shard_line_offset"), 5)

    def test_versioned_dir_without_committed_marker_rejected(self):
        """
        A versioned directory that exists but has no COMMITTED marker (incomplete
        promotion) must be rejected by _collect_versioned_checkpoint_dirs and
        must not be returned as a valid candidate.
        """
        parent_dir = self.tmpdir
        output_basename = "ckpt"
        # Create a fake versioned dir without COMMITTED marker
        fake_vdir = os.path.join(parent_dir, f"{output_basename}.step10_9999999")
        os.makedirs(fake_vdir)
        # Write some artifacts but no COMMITTED marker
        with open(os.path.join(fake_vdir, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 64)

        candidates = _collect_versioned_checkpoint_dirs(parent_dir, output_basename)
        self.assertNotIn(fake_vdir, candidates,
                         "Versioned dir without COMMITTED marker was incorrectly accepted")

    def test_committed_versioned_dirs_discovered_newest_first(self):
        """
        _collect_versioned_checkpoint_dirs must return committed dirs sorted
        newest-first (by step number extracted from dir name).
        """
        parent_dir = self.tmpdir
        basename = "ckpt"
        for step in [3, 7, 1, 15, 5]:
            vdir = os.path.join(parent_dir, f"{basename}.step{step}_100000{step}")
            os.makedirs(vdir)
            with open(os.path.join(vdir, "COMMITTED"), "w") as f:
                json.dump({"step": step}, f)

        candidates = _collect_versioned_checkpoint_dirs(parent_dir, basename)
        self.assertEqual(len(candidates), 5)
        # Extract step numbers from paths and verify descending order
        steps = []
        for p in candidates:
            name = os.path.basename(p)
            step_str = name[len(basename) + len(".step"):].split("_")[0]
            steps.append(int(step_str))
        self.assertEqual(steps, sorted(steps, reverse=True),
                         f"Versioned dirs not sorted newest-first: {steps}")

    def test_crash_before_committed_marker_leaves_prior_checkpoint_recoverable(self):
        """
        If the process crashes after staging -> versioned rename but before
        writing the COMMITTED marker, the prior checkpoint at output_dir must
        still be recoverable (the new versioned dir is not committed so is ignored).
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")

        # Save a valid first checkpoint (step=1)
        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path,
            output_dir=ckpt_dir, step=1, epoch=0, loss=5.0, best_loss=5.0,
            stage_id="test", sync_to_gdrive=False,
            data_cursor={
                "active_shard": "s0.jsonl", "active_shard_line_offset": 1,
                "active_shard_byte_offset": 64, "active_chunk_idx": 0,
                "total_samples_trained": 1, "samples_consumed": 1, "completed_shards": [],
            },
        )

        parent_dir = os.path.dirname(os.path.abspath(ckpt_dir))
        versioned_step1_dirs = _collect_versioned_checkpoint_dirs(
            parent_dir, os.path.basename(ckpt_dir)
        )
        self.assertEqual(len(versioned_step1_dirs), 1,
                         "Expected exactly 1 committed versioned dir after step 1")

        # Simulate crash during step-2 COMMITTED marker write:
        # manually create a versioned dir for step 2 WITHOUT a COMMITTED marker
        fake_v2 = os.path.join(parent_dir, f"{os.path.basename(ckpt_dir)}.step2_99999999")
        os.makedirs(fake_v2, exist_ok=True)
        # copy artifacts but no COMMITTED
        for fname in ["model.safetensors", "optimizer_state.pt", "checkpoint_state.json",
                      "config.json", "tokenizer.json", "STAGE_METADATA.json"]:
            src = os.path.join(ckpt_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(fake_v2, fname))

        # Recovery must find step-1 versioned dir (step-2 has no COMMITTED marker)
        reload_model = _make_tiny_model(config)
        reload_optimizer = _make_tiny_optimizer(reload_model)
        reload_scaler = _make_tiny_scaler()
        step, _, _, was_resumed, cursor = try_resume_from_checkpoint(
            ckpt_dir, reload_model, reload_optimizer, reload_scaler
        )
        self.assertTrue(was_resumed, "Expected recovery from step-1 checkpoint")
        self.assertEqual(step, 1,
                         f"Should have resumed at step=1 (step-2 is incomplete), got step={step}")


# =============================================================================
# G4 — Checkpoint / ledger consistency
# =============================================================================

class TestG4CheckpointLedgerConsistency(unittest.TestCase):
    """
    Gate 4: in-memory _pending_shard_completions lost on restart causes shard
    to be re-streamed (not permanently lost or incorrectly skipped);
    stale-behind and stale-ahead ledger reconciliation.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g4_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pending_completions_lost_on_restart_causes_resream_not_loss(self):
        """
        Scenario: shard reaches EOF and is put in _pending_shard_completions.
        Process crashes before checkpoint commit (so shard is never in
        ledger.completed_shards AND the in-memory pending set is lost).
        On restart the shard must be re-streamable (not marked completed,
        not evicted).
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_restream.jsonl")
        _write_jsonl_shard(shard_path, [
            {"text": "ABCDEFGH"},
            {"text": "IJKLMNOP"},
        ])

        # First run: stream to EOF, simulate crash (never call commit_shard_completion)
        ledger = ShardLedger(self.tmpdir)
        ds1 = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=16, shard_ledger=ledger,
        )
        items1 = list(ds1)
        self.assertGreater(len(items1), 0)
        self.assertIn("shard_restream.jsonl", ds1._pending_shard_completions)
        self.assertNotIn("shard_restream.jsonl", ledger.completed_shards)

        # Simulate process restart: ledger reloaded from disk, pending set is empty
        ledger2 = ShardLedger(self.tmpdir)
        # Shard must NOT be marked completed in the reloaded ledger
        self.assertNotIn("shard_restream.jsonl", ledger2.completed_shards,
                         "Shard appeared in ledger.completed_shards before checkpoint commit")

        # Second run (after simulated restart) must re-stream the shard
        ds2 = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=16, shard_ledger=ledger2,
        )
        items2 = list(ds2)
        self.assertGreater(len(items2), 0,
                           "Shard could not be re-streamed after simulated restart")
        self.assertEqual(len(items1), len(items2),
                         "Re-streamed item count differs from original — data was lost or duplicated")

    def test_stale_behind_ledger_reconciled_from_checkpoint(self):
        """
        Ledger is behind the committed checkpoint (crash after checkpoint commit
        but before ledger.save()).  reconcile_with_checkpoint() must bring the
        ledger up to the checkpoint's authoritative state.
        """
        ledger = ShardLedger(self.tmpdir)
        # Stale: ledger has no completed shards
        ledger.completed_shards = []
        ledger.save()

        committed_cursor = {
            "active_shard": "shard_001.jsonl",
            "active_shard_line_offset": 20,
            "active_shard_byte_offset": 1024,
            "total_samples_trained": 100,
            "completed_shards": ["shard_000.jsonl"],
        }
        ledger.reconcile_with_checkpoint(committed_cursor, step=10)

        self.assertIn("shard_000.jsonl", ledger.completed_shards,
                      "Stale-behind reconciliation failed: shard_000 not restored")
        self.assertEqual(ledger.active_shard, "shard_001.jsonl")
        self.assertEqual(ledger.last_committed_step, 10)

    def test_stale_ahead_ledger_reconciled_removes_premature_completions(self):
        """
        Ledger is ahead of the committed checkpoint (shard marked completed in
        ledger but not in checkpoint — e.g., pre-Phase 1 bug replay).
        reconcile_with_checkpoint() must remove the premature completion.
        """
        ledger = ShardLedger(self.tmpdir)
        ledger.completed_shards = ["shard_000.jsonl", "shard_001.jsonl"]
        ledger.completed_metadata = {
            "shard_000.jsonl": {"records_processed": 10, "file_sha256": "abc"},
            "shard_001.jsonl": {"records_processed": 10, "file_sha256": "def"},
        }
        ledger.save()

        committed_cursor = {
            "active_shard": "shard_001.jsonl",
            "active_shard_line_offset": 5,
            "active_shard_byte_offset": 256,
            "total_samples_trained": 10,
            "completed_shards": ["shard_000.jsonl"],  # shard_001 NOT in checkpoint
        }
        ledger.reconcile_with_checkpoint(committed_cursor, step=5)

        self.assertIn("shard_000.jsonl", ledger.completed_shards)
        self.assertNotIn("shard_001.jsonl", ledger.completed_shards,
                         "Stale-ahead shard_001 was not removed by reconciliation")

    def test_shard_completion_only_after_checkpoint_commit(self):
        """
        After a shard reaches EOF, calling save_checkpoint (which calls
        commit_shard_completion internally) must move the shard from
        _pending_shard_completions into ledger.completed_shards.
        """
        tokenizer = _make_tiny_tokenizer()
        shard_path = os.path.join(self.tmpdir, "shard_pending.jsonl")
        _write_jsonl_shard(shard_path, [{"text": "ABCDEFGH"}])

        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)

        ledger = ShardLedger(self.tmpdir)
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_path], tokenizer=tokenizer,
            max_seq_len=16, shard_ledger=ledger,
            dataset_dir=self.tmpdir,
        )
        # Stream to EOF
        list(ds)

        self.assertIn("shard_pending.jsonl", ds._pending_shard_completions,
                      "Shard should be in pending after EOF")
        self.assertNotIn("shard_pending.jsonl", ledger.completed_shards,
                         "Shard should NOT be in ledger before checkpoint commit")

        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path,
            output_dir=ckpt_dir, step=1, epoch=0, loss=4.5, best_loss=4.5,
            stage_id="test", sync_to_gdrive=False,
            dataset=ds, shard_ledger=ledger,
            data_cursor={
                "active_shard": "shard_pending.jsonl",
                "active_shard_line_offset": 1,
                "active_shard_byte_offset": 64,
                "active_chunk_idx": 0,
                "total_samples_trained": 1,
                "samples_consumed": 1,
                "completed_shards": [],
            },
        )

        self.assertIn("shard_pending.jsonl", ledger.completed_shards,
                      "Shard should be in ledger.completed_shards after checkpoint commit")
        self.assertNotIn("shard_pending.jsonl", ds._pending_shard_completions,
                         "Shard should be removed from _pending after commit")


# =============================================================================
# G5 — Best Model safety
# =============================================================================

class TestG5BestModelSafety(unittest.TestCase):
    """
    Gate 5: incomplete best_checkpoint (no COMMITTED marker) is rejected;
    optimizer resume never selects a best_checkpoint directory;
    completed best_checkpoint has a COMMITTED marker.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g5_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_committed_best_checkpoint_has_committed_marker(self):
        """save_best_milestone must write a COMMITTED marker after successful commit."""
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)
        best_dir = os.path.join(self.tmpdir, "best_checkpoint")

        result = save_best_milestone(
            model=model, config=config, tokenizer_path=tok_path,
            best_dir=best_dir, step=10, epoch=0, loss=3.0, best_loss=3.0,
            stage_id="test",
        )
        self.assertTrue(result, "save_best_milestone returned False unexpectedly")
        committed_path = os.path.join(best_dir, "COMMITTED")
        self.assertTrue(os.path.isfile(committed_path),
                        "COMMITTED marker is missing from committed best_checkpoint")
        with open(committed_path) as f:
            marker = json.load(f)
        self.assertTrue(marker.get("is_best"), "COMMITTED marker does not have is_best=True")

    def test_best_checkpoint_without_optimizer_excluded_from_resume(self):
        """
        A best_checkpoint directory (no optimizer_state.pt) must be skipped
        by try_resume_from_checkpoint — it can never be used for optimizer resume.
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)
        best_dir = os.path.join(self.tmpdir, "best_checkpoint")

        save_best_milestone(
            model=model, config=config, tokenizer_path=tok_path,
            best_dir=best_dir, step=99, epoch=0, loss=2.5, best_loss=2.5,
            stage_id="test",
        )

        # best_dir has model + state but no optimizer_state.pt
        self.assertFalse(
            os.path.exists(os.path.join(best_dir, "optimizer_state.pt")),
            "best_checkpoint unexpectedly has optimizer_state.pt"
        )

        reload_model = _make_tiny_model(config)
        reload_optimizer = _make_tiny_optimizer(reload_model)
        reload_scaler = _make_tiny_scaler()
        # Pass best_dir as checkpoint_dir — must NOT resume (missing optimizer)
        step, _, _, was_resumed, _ = try_resume_from_checkpoint(
            best_dir, reload_model, reload_optimizer, reload_scaler
        )
        self.assertFalse(was_resumed,
                         "try_resume_from_checkpoint accepted a best_checkpoint without optimizer state")

    def test_incomplete_best_checkpoint_rejected_by_integrity_check(self):
        """
        A partially-written best_checkpoint (no COMMITTED marker, truncated model)
        must be rejected by verify_checkpoint_integrity.
        """
        incomplete_dir = os.path.join(self.tmpdir, "incomplete_best")
        os.makedirs(incomplete_dir)
        # Write only a truncated model file (no COMMITTED, no state json)
        with open(os.path.join(incomplete_dir, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 32)

        self.assertFalse(
            verify_checkpoint_integrity(incomplete_dir, require_optimizer=False),
            "Incomplete best_checkpoint should fail integrity check"
        )

    def test_best_checkpoint_not_selected_over_training_checkpoint(self):
        """
        When both a training checkpoint and a best_checkpoint exist in the same
        parent directory, try_resume_from_checkpoint must select the training
        checkpoint (which has optimizer_state.pt) and ignore the best_checkpoint.
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)

        # Save training checkpoint at step=5
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        save_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler,
            config=config, tokenizer_path=tok_path,
            output_dir=ckpt_dir, step=5, epoch=0, loss=4.0, best_loss=4.0,
            stage_id="test", sync_to_gdrive=False,
            data_cursor={
                "active_shard": "s0.jsonl", "active_shard_line_offset": 5,
                "active_shard_byte_offset": 320, "active_chunk_idx": 0,
                "total_samples_trained": 5, "samples_consumed": 5,
                "completed_shards": [],
            },
        )

        # Save best_checkpoint (model only, step=99 but no optimizer)
        best_dir = os.path.join(self.tmpdir, "best_checkpoint")
        save_best_milestone(
            model=model, config=config, tokenizer_path=tok_path,
            best_dir=best_dir, step=99, epoch=0, loss=2.0, best_loss=2.0,
            stage_id="test",
        )

        # Resume from ckpt_dir: must pick training ckpt (step=5), not best (step=99)
        reload_model = _make_tiny_model(config)
        reload_optimizer = _make_tiny_optimizer(reload_model)
        reload_scaler = _make_tiny_scaler()
        step, _, _, was_resumed, _ = try_resume_from_checkpoint(
            ckpt_dir, reload_model, reload_optimizer, reload_scaler
        )
        self.assertTrue(was_resumed)
        self.assertEqual(step, 5,
                         f"Should have resumed at step=5 (training ckpt), got step={step}")


# =============================================================================
# G6 — Storage retention
# =============================================================================

class TestG6StorageRetention(unittest.TestCase):
    """
    Gate 6: retention policy keeps the newest keep_n committed versioned dirs
    and removes older committed dirs and all uncommitted staging remnants.
    Never deletes before new checkpoint is committed.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g6_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_versioned_dir(self, parent: str, basename: str, step: int,
                             committed: bool = True) -> str:
        vdir = os.path.join(parent, f"{basename}.step{step}_{1000000 + step}")
        os.makedirs(vdir)
        if committed:
            with open(os.path.join(vdir, "COMMITTED"), "w") as f:
                json.dump({"step": step}, f)
        return vdir

    def test_retention_keeps_newest_two_deletes_older(self):
        """
        With keep_n=2 and 4 committed versioned dirs (steps 1,2,3,4),
        _cleanup_old_versioned_checkpoints must retain steps 3 and 4,
        and delete steps 1 and 2.
        """
        basename = "ckpt"
        dirs = {}
        for step in [1, 2, 3, 4]:
            dirs[step] = self._make_versioned_dir(self.tmpdir, basename, step)

        _cleanup_old_versioned_checkpoints(self.tmpdir, basename, keep_n=2)

        self.assertTrue(os.path.isdir(dirs[4]), "Newest checkpoint (step=4) was deleted")
        self.assertTrue(os.path.isdir(dirs[3]), "Second-newest checkpoint (step=3) was deleted")
        self.assertFalse(os.path.isdir(dirs[2]),
                         "Old checkpoint (step=2) should have been deleted")
        self.assertFalse(os.path.isdir(dirs[1]),
                         "Oldest checkpoint (step=1) should have been deleted")

    def test_retention_removes_uncommitted_staging_remnants(self):
        """
        Uncommitted versioned directories (no COMMITTED marker) must be removed
        by the retention cleanup regardless of keep_n.
        """
        basename = "ckpt"
        # One committed dir
        committed_dir = self._make_versioned_dir(self.tmpdir, basename, step=10, committed=True)
        # One uncommitted dir (staging remnant)
        uncommitted_dir = self._make_versioned_dir(self.tmpdir, basename, step=11, committed=False)

        _cleanup_old_versioned_checkpoints(self.tmpdir, basename, keep_n=2)

        self.assertTrue(os.path.isdir(committed_dir),
                        "Committed dir was incorrectly deleted")
        self.assertFalse(os.path.isdir(uncommitted_dir),
                         "Uncommitted staging remnant was not removed by retention")

    def test_retention_never_keeps_fewer_than_2(self):
        """
        Even if keep_n=1 is passed, the function must enforce a minimum of 2.
        With 3 committed dirs (steps 1,2,3), passing keep_n=1 must still keep
        steps 2 and 3 (enforced minimum of 2).
        """
        basename = "ckpt"
        dirs = {}
        for step in [1, 2, 3]:
            dirs[step] = self._make_versioned_dir(self.tmpdir, basename, step)

        _cleanup_old_versioned_checkpoints(self.tmpdir, basename, keep_n=1)

        self.assertTrue(os.path.isdir(dirs[3]), "Step=3 (newest) must be kept")
        self.assertTrue(os.path.isdir(dirs[2]), "Step=2 must be kept (minimum keep=2)")
        self.assertFalse(os.path.isdir(dirs[1]), "Step=1 (oldest) should be deleted")

    def test_retention_called_after_save_checkpoint_keeps_two(self):
        """
        End-to-end: after multiple sequential save_checkpoint calls, only the
        two most recent versioned checkpoint directories must remain.
        The third (oldest) must be automatically deleted by the retention policy.
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        _write_tokenizer(tok_path)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        parent_dir = self.tmpdir
        basename = "ckpt"

        for step in [1, 2, 3]:
            # Small sleep to ensure unique timestamp in versioned dir name
            time.sleep(0.05)
            save_checkpoint(
                model=model, optimizer=optimizer, scaler=scaler,
                config=config, tokenizer_path=tok_path,
                output_dir=ckpt_dir, step=step, epoch=0,
                loss=5.0 - step * 0.5, best_loss=5.0 - step * 0.5,
                stage_id="test", sync_to_gdrive=False,
                data_cursor={
                    "active_shard": "s0.jsonl",
                    "active_shard_line_offset": step,
                    "active_shard_byte_offset": step * 64,
                    "active_chunk_idx": 0,
                    "total_samples_trained": step,
                    "samples_consumed": step,
                    "completed_shards": [],
                },
            )

        # After 3 saves with keep_n=2 retention, only 2 versioned dirs should exist
        remaining = _collect_versioned_checkpoint_dirs(parent_dir, basename)
        self.assertLessEqual(len(remaining), 2,
                             f"Expected at most 2 versioned dirs after retention, found {len(remaining)}: "
                             f"{[os.path.basename(d) for d in remaining]}")
        # The most recent (step=3) must be among them
        most_recent = remaining[0] if remaining else ""
        step_in_name = os.path.basename(most_recent)[len(basename) + len(".step"):].split("_")[0]
        self.assertEqual(int(step_in_name), 3,
                         "The most recent versioned dir (step=3) must be retained")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

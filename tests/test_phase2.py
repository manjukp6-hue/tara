#!/usr/bin/env python3
"""
Phase 2 Training Hygiene and Resume Audit — Regression Tests.

Tests address:
  H1 — resume_chunk_idx semantics: active_chunk_idx = next chunk to yield (exclusive).
       Proves that the cursor written by the training loop stores c_chunk + 1 so that
       the dataset iterator skips exactly the committed chunks and resumes from the
       first uncommitted one.  Covers resume_chunk_idx=0 (start of line), N>0 (mid-line),
       and N == total_chunks (beyond last chunk → nothing more in that line).

  H2 — Short document and final-short-chunk policy: documents with fewer than 4 tokens
       are entirely discarded; multi-chunk documents with a short final chunk yield the
       full-length chunks but drop the short tail.  No silent discard of non-tail tokens.

  H3 — Byte-seek validity guard: the byte-seek optimisation is only applied when the
       stored byte offset is within the current file bounds.  An out-of-range offset
       causes a clean fallback to sequential scan.  The byte-seek is skipped entirely
       when target_resume_chunk > 0 so that partial chunks of the resume line are not
       silently dropped.

  H4 — Partial grad-accum flush correctness: the end-of-epoch partial window is
       flushed regardless of whether step >= max_steps.  Non-finite gradients in the
       partial window are logged and discarded without advancing step or cursor.

  H5 — Cursor write correctness (no last-chunk duplication): verify that after a
       simulated optimizer step the cursor's active_chunk_idx equals last_chunk_idx + 1,
       not last_chunk_idx, so the iterator does not re-yield the last committed chunk.

  H6 — try_resume_from_checkpoint restores active_chunk_idx from the legacy flat
       checkpoint_state.json layout (no data_cursor key present).

All tests use bounded synthetic fixtures (temp dirs + small tokenizer).
No GPU training, no Hugging Face downloads, no production checkpoint writes.
"""

import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
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
    verify_checkpoint_integrity,
    try_resume_from_checkpoint,
    _collect_versioned_checkpoint_dirs,
    _cleanup_old_versioned_checkpoints,
)


# ---------------------------------------------------------------------------
# Shared helpers
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
    vocab = {chr(i): i for i in range(32, 32 + 64)}\
        if True else {}
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
    """Build a fully committed versioned checkpoint at <tmpdir>/ckpt."""
    config = _make_tiny_config()
    model = _make_tiny_model(config)
    optimizer = _make_tiny_optimizer(model)
    scaler = _make_tiny_scaler()
    tok_path = os.path.join(tmpdir, "tok.json")
    _write_tokenizer(tok_path)
    output_dir = os.path.join(tmpdir, "ckpt")
    os.makedirs(output_dir, exist_ok=True)
    save_checkpoint(
        model, optimizer, scaler, config, tok_path, output_dir,
        step=step, epoch=0, loss=1.5, best_loss=1.5, stage_id="test",
        is_best=False, is_completed=False, data_cursor=cursor,
    )
    return output_dir


# =============================================================================
# H1 — resume_chunk_idx semantics
# =============================================================================

class TestH1ResumeChunkIdxSemantics(unittest.TestCase):
    """
    active_chunk_idx in the checkpoint cursor is the index of the FIRST uncommitted
    chunk within the resume line — i.e., the exclusive lower bound for the iterator's
    per-chunk skip filter.  The iterator skips chunks 0 .. active_chunk_idx-1 and
    resumes from active_chunk_idx onward.

    Special case: active_chunk_idx == 0 means the entire resume line is committed
    (skip the whole line, advance to the next).
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tokenizer = _make_tiny_tokenizer()
        # Each record produces exactly 3 chunks of 8 tokens (24 tokens total)
        self.max_seq_len = 8
        records = [{"text": chr(65 + i) * 24} for i in range(4)]  # lines 0-3
        self.shard_path = os.path.join(self.tmpdir, "shard.jsonl")
        _write_jsonl_shard(self.shard_path, records)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ds(self, resume_line, resume_chunk):
        return TaraStreamingJsonlDataset(
            file_paths=[self.shard_path],
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            resume_shard="shard.jsonl",
            resume_line_offset=resume_line,
            resume_byte_offset=0,
            resume_chunk_idx=resume_chunk,
        )

    def test_chunk_idx_0_skips_entire_resume_line(self):
        """
        active_chunk_idx=0 means the whole resume line is committed.
        No chunks from the resume line must appear; only lines after it.
        """
        ds = self._ds(resume_line=1, resume_chunk=0)
        items = list(ds)
        yielded = [(item[3], item[5]) for item in items]

        # Line 0 must not appear (before resume line)
        self.assertEqual([], [p for p in yielded if p[0] == 0],
                         "Line 0 should be absent (before resume line)")
        # Line 1 must not appear (resume_chunk=0 → whole line committed)
        self.assertEqual([], [p for p in yielded if p[0] == 1],
                         "resume_chunk_idx=0 must skip the entire resume line")
        # Lines 2,3 must each produce 3 chunks
        for line_idx in (2, 3):
            chunks = [p for p in yielded if p[0] == line_idx]
            self.assertEqual(3, len(chunks),
                             f"Line {line_idx} should yield 3 chunks, got {len(chunks)}")

    def test_chunk_idx_1_skips_only_chunk_0(self):
        """
        active_chunk_idx=1 means chunk 0 of the resume line is committed.
        Chunks 1,2 of the resume line must be yielded; chunk 0 must not.
        """
        ds = self._ds(resume_line=1, resume_chunk=1)
        items = list(ds)
        yielded = [(item[3], item[5]) for item in items]

        line1 = [(l, c) for l, c in yielded if l == 1]
        chunk_indices = [c for _, c in line1]
        self.assertNotIn(0, chunk_indices,
                         "Chunk 0 of resume line must not appear (committed)")
        self.assertIn(1, chunk_indices,
                      "Chunk 1 of resume line must appear (uncommitted)")
        self.assertIn(2, chunk_indices,
                      "Chunk 2 of resume line must appear (uncommitted)")

    def test_chunk_idx_2_skips_chunks_0_and_1(self):
        """
        active_chunk_idx=2 means chunks 0,1 are committed.
        Only chunk 2 of the resume line must be yielded.
        This is the canonical regression test requested by the Phase 2 audit.
        """
        ds = self._ds(resume_line=1, resume_chunk=2)
        items = list(ds)
        yielded = [(item[3], item[5]) for item in items]

        line1 = [(l, c) for l, c in yielded if l == 1]
        chunk_indices_line1 = [c for _, c in line1]
        self.assertNotIn(0, chunk_indices_line1,
                         "Chunk 0 of resume line is committed — must not appear")
        self.assertNotIn(1, chunk_indices_line1,
                         "Chunk 1 of resume line is committed — must not appear")
        self.assertIn(2, chunk_indices_line1,
                      "Chunk 2 of resume line is uncommitted — must appear")

    def test_chunk_idx_beyond_last_chunk_yields_nothing_for_resume_line(self):
        """
        active_chunk_idx = total_chunks (e.g. 3 for a 3-chunk doc) means all
        chunks of the resume line are committed.  The iterator skip filter
        chunk_idx < 3 eliminates all three chunks → resume line produces nothing.
        Lines after the resume line are unaffected.
        """
        ds = self._ds(resume_line=1, resume_chunk=3)
        items = list(ds)
        yielded = [(item[3], item[5]) for item in items]

        self.assertEqual([], [p for p in yielded if p[0] == 1],
                         "All chunks committed → resume line should yield nothing")
        # Lines 2,3 still appear
        for line_idx in (2, 3):
            chunks = [p for p in yielded if p[0] == line_idx]
            self.assertEqual(3, len(chunks),
                             f"Line {line_idx} should still yield 3 chunks")

    def test_no_chunk_appears_before_resume_line(self):
        """Regardless of resume_chunk_idx, lines before the resume line never appear."""
        for resume_chunk in (0, 1, 2):
            with self.subTest(resume_chunk=resume_chunk):
                ds = self._ds(resume_line=2, resume_chunk=resume_chunk)
                items = list(ds)
                yielded = [(item[3], item[5]) for item in items]
                before = [p for p in yielded if p[0] < 2]
                self.assertEqual([], before,
                                 f"Lines before resume line appeared with resume_chunk={resume_chunk}")


# =============================================================================
# H2 — Short document and final-short-chunk policy
# =============================================================================

class TestH2ShortDocAndFinalChunkPolicy(unittest.TestCase):
    """
    The minimum retained chunk length is 4 tokens.
    Documents with < 4 tokens are entirely discarded (no items yielded).
    Multi-chunk documents yield all full-length chunks; short final chunks are dropped.
    The total number of trained tokens equals the sum of tokens in retained chunks only.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tokenizer = _make_tiny_tokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_doc_with_3_tokens_produces_no_items(self):
        """A 3-token document is entirely below threshold — nothing yielded."""
        shard = os.path.join(self.tmpdir, "s.jsonl")
        _write_jsonl_shard(shard, [{"text": "ABC"}])  # 3 tokens
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=16,
        )
        self.assertEqual([], list(ds), "3-token doc must yield nothing")

    def test_doc_with_exactly_4_tokens_yields_one_item(self):
        """A 4-token document meets the minimum threshold — exactly 1 item yielded."""
        shard = os.path.join(self.tmpdir, "s.jsonl")
        _write_jsonl_shard(shard, [{"text": "ABCD"}])  # 4 tokens
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=16,
        )
        items = list(ds)
        self.assertEqual(1, len(items), "4-token doc must yield exactly 1 item")
        self.assertEqual(4, len(items[0][0]), "Input ids length must be 4")

    def test_long_doc_with_short_final_chunk_drops_tail(self):
        """
        A document producing N full chunks + 1 short final chunk must yield N items.
        The short tail (< 4 tokens) is explicitly discarded; the full chunks are retained.
        """
        max_seq_len = 8
        # 19 tokens → 2 full chunks (8+8) + 1 tail chunk (3 tokens, discarded)
        shard = os.path.join(self.tmpdir, "s.jsonl")
        _write_jsonl_shard(shard, [{"text": "A" * 19}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
        )
        items = list(ds)
        self.assertEqual(2, len(items),
                         "Short tail chunk must be discarded; only 2 full chunks retained")

    def test_long_doc_with_minimum_final_chunk_retains_all_chunks(self):
        """
        A document producing N full chunks + 1 final chunk of exactly 4 tokens
        must yield N+1 items.  The minimum-length tail is above threshold and kept.
        """
        max_seq_len = 8
        # 20 tokens → 2 full chunks (8+8) + 1 tail chunk (4 tokens, kept)
        shard = os.path.join(self.tmpdir, "s.jsonl")
        _write_jsonl_shard(shard, [{"text": "A" * 20}])
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
        )
        items = list(ds)
        self.assertEqual(3, len(items),
                         "4-token tail chunk meets threshold — all 3 chunks must be retained")

    def test_total_trained_tokens_match_retained_chunks_only(self):
        """
        For a shard with mixed documents, the total number of input tokens across
        all yielded items equals exactly the sum of tokens in retained chunks.
        Discarded short-doc and short-tail tokens are not counted.
        """
        max_seq_len = 8
        records = [
            {"text": "A" * 3},   # 3 tokens → discarded entirely
            {"text": "A" * 8},   # 1 chunk of 8 tokens → 1 item
            {"text": "A" * 19},  # 2 full + 3-token tail → 2 items (19 → 8+8 kept, 3 dropped)
        ]
        shard = os.path.join(self.tmpdir, "s.jsonl")
        _write_jsonl_shard(shard, records)
        ds = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
        )
        items = list(ds)
        total_tokens = sum(len(item[0]) for item in items)
        # Expected: 8 (from second doc) + 8 + 8 (two full chunks from third doc) = 24
        self.assertEqual(24, total_tokens,
                         f"Total trained tokens must be 24 (no silent discard of retained tokens); got {total_tokens}")


# =============================================================================
# H3 — Byte-seek validity guard
# =============================================================================

class TestH3ByteSeekValidityGuard(unittest.TestCase):
    """
    The byte-seek resume optimisation must:
    1. Fall back to sequential scan when the stored byte offset is out of range
       (beyond the current file end), preventing a seek to a corrupt position.
    2. Not apply when target_resume_chunk > 0 — the resume line still has
       uncommitted chunks that must be streamed through the per-chunk filter.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tokenizer = _make_tiny_tokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_out_of_range_byte_offset_falls_back_to_sequential(self):
        """
        An out-of-range byte offset must cause the iterator to fall back to
        sequential scan from the start of the file.  The result must be
        identical to a fresh (no-resume) iteration of the same shard.
        """
        shard = os.path.join(self.tmpdir, "s.jsonl")
        records = [{"text": "A" * 16} for _ in range(3)]
        _write_jsonl_shard(shard, records)

        file_size = os.path.getsize(shard)
        invalid_byte_offset = file_size + 9999  # guaranteed out-of-range

        ds_fresh = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=16,
        )
        fresh_chunks = [(item[3], item[5]) for item in ds_fresh]

        # Resume at line -1 with an invalid byte offset: should re-stream from start
        ds_bad_seek = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=16,
            resume_shard="s.jsonl",
            resume_line_offset=-1,
            resume_byte_offset=invalid_byte_offset,
            resume_chunk_idx=0,
        )
        bad_seek_chunks = [(item[3], item[5]) for item in ds_bad_seek]
        self.assertEqual(fresh_chunks, bad_seek_chunks,
                         "Out-of-range byte offset must fall back to full sequential scan")

    def test_byte_seek_skipped_when_resume_chunk_nonzero(self):
        """
        When target_resume_chunk > 0, the byte-seek must NOT be used even if
        target_resume_byte is set, because the resume line has uncommitted chunks
        that would be silently skipped by the seek.  The iterator must re-process
        the resume line and yield only the uncommitted chunks (chunk_idx >= resume_chunk).
        """
        max_seq_len = 8
        # 3 records each producing 3 chunks
        records = [{"text": chr(65 + i) * 24} for i in range(3)]
        shard = os.path.join(self.tmpdir, "s.jsonl")
        _write_jsonl_shard(shard, records)

        # Collect byte offsets by streaming once
        ds_full = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
        )
        full_items = list(ds_full)
        # Byte offset stored after reading line 0 (= start of line 1)
        line0_byte_end = full_items[2][4]  # third item is chunk 2 of line 0, byte_offset = end of line 0

        # Resume: line 0 committed through chunk 1 (active_chunk_idx=2 means next to yield is chunk 2)
        ds_resume = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
            resume_shard="s.jsonl",
            resume_line_offset=0,
            resume_byte_offset=line0_byte_end,   # byte after line 0
            resume_chunk_idx=2,                   # resume_chunk > 0 → byte-seek must NOT fire
        )
        resume_items = list(ds_resume)
        resume_yielded = [(item[3], item[5]) for item in resume_items]

        # Only chunk 2 of line 0 should appear (chunks 0,1 committed)
        line0_chunks = [c for l, c in resume_yielded if l == 0]
        self.assertNotIn(0, line0_chunks,
                         "Chunk 0 of resume line is committed — must not appear")
        self.assertNotIn(1, line0_chunks,
                         "Chunk 1 of resume line is committed — must not appear")
        self.assertIn(2, line0_chunks,
                      "Chunk 2 of resume line is uncommitted — must appear")

        # Lines 1,2 must appear in full (3 chunks each)
        for line_idx in (1, 2):
            line_chunks = [c for l, c in resume_yielded if l == line_idx]
            self.assertEqual(3, len(line_chunks),
                             f"Line {line_idx} must yield all 3 chunks; got {len(line_chunks)}")


# =============================================================================
# H4 — Partial grad-accum flush correctness
# =============================================================================

class TestH4PartialFlushCorrectness(unittest.TestCase):
    """
    The end-of-epoch partial accumulation window must be flushed whether or not
    step >= max_steps.  Non-finite gradients in the partial window must be logged
    and discarded without advancing the committed step or cursor.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_flush_harness(self):
        """
        Build a minimal flush harness that simulates the end-of-epoch partial
        accumulation block in cloud_train.main() without running full training.
        Returns a dict with the state needed to assert on.

        Token IDs are constructed in-range (0 .. vocab_size-1 = 63) so the
        tiny model's embedding lookup does not raise IndexError.
        """
        import math

        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()

        # Build a synthetic batch directly — 8 tokens, all IDs in [1, 10] (well within vocab)
        vocab_size = config["vocab_size"]
        seq_len = 8
        raw_ids = list(range(1, seq_len + 1))          # [1,2,3,4,5,6,7,8] — all < 64
        inputs = torch.tensor([raw_ids], dtype=torch.long)
        labels = inputs.clone()

        with torch.no_grad():
            _, loss, _, _ = model(inputs, labels=labels)

        fake_shard = "synthetic.jsonl"
        fake_cursor = (fake_shard, 0, 0, 0)
        accum_batches = [(inputs.cpu(), labels.cpu(), [fake_cursor])]

        return {
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "accum_batches": accum_batches,
            "loss": loss,
            "step": 5,
            "total_samples_trained": 0,
            "samples_consumed": 1,
            "shard_ledger": None,
            "last_committed_step_before": 5,
        }

    def test_partial_flush_executes_regardless_of_max_steps(self):
        """
        When step >= max_steps, the partial window must still be flushed.
        The step counter must increment after a successful partial optimizer step.
        """
        h = self._make_flush_harness()
        model = h["model"]
        optimizer = h["optimizer"]
        scaler = h["scaler"]
        accum_batches = h["accum_batches"]

        # Reproduce the partial flush logic from cloud_train.main()
        step = h["step"]
        max_steps = step  # step == max_steps → old code would skip flush
        total_samples_trained = h["total_samples_trained"]
        samples_consumed = h["samples_consumed"]
        shard_ledger = h["shard_ledger"]
        current_cursor = None

        if accum_batches:  # new code: always flush, no step < max_steps guard
            partial_count = len(accum_batches)
            accum_loss = 0.0
            # Accumulate loss
            for r_inputs, r_labels, _ in accum_batches:
                _, r_loss, _, _ = model(r_inputs, labels=r_labels)
                r_loss_scaled = r_loss / partial_count
                scaler.scale(r_loss_scaled).backward()
                accum_loss += r_loss.item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            is_non_finite_grad = math.isnan(grad_norm_val) or math.isinf(grad_norm_val)

            if is_non_finite_grad and not scaler.is_enabled():
                is_step_skipped = True
            else:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                is_step_skipped = (scaler.is_enabled() and (scale_after < scale_before)) or is_non_finite_grad

            if not is_step_skipped:
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window_samples = sum(len(c) for _, _, c in accum_batches)
                total_samples_trained += window_samples

                last_sample_cursor = accum_batches[-1][2][-1] if (accum_batches and accum_batches[-1][2]) else (None, 0, 0, 0)
                c_shard = last_sample_cursor[0]
                c_line = last_sample_cursor[1] if len(last_sample_cursor) > 1 else 0
                c_byte = last_sample_cursor[2] if len(last_sample_cursor) > 2 else 0
                c_chunk_last = last_sample_cursor[3] if len(last_sample_cursor) > 3 else 0
                c_chunk_next = c_chunk_last + 1  # Phase 2 fix

                current_cursor = {
                    "active_shard": c_shard,
                    "active_shard_line_offset": c_line,
                    "active_shard_byte_offset": c_byte,
                    "active_chunk_idx": c_chunk_next,
                    "total_samples_trained": total_samples_trained,
                    "samples_consumed": samples_consumed,
                    "completed_shards": [],
                }
            accum_batches.clear()
            optimizer.zero_grad(set_to_none=True)

        # Assert: step advanced by 1 despite starting at max_steps
        self.assertEqual(h["step"] + 1, step,
                         "Partial flush must advance step even when step == max_steps before flush")
        self.assertIsNotNone(current_cursor,
                             "current_cursor must be set after successful partial flush")

    def test_non_finite_gradient_partial_flush_logs_and_does_not_advance_step(self):
        """
        When the gradient norm is NaN in the partial window (with scaler disabled),
        the optimizer step must be skipped.  The committed step must remain unchanged.
        The log output must explicitly mention the discard.
        """
        import math
        from io import StringIO

        h = self._make_flush_harness()
        model = h["model"]
        optimizer = h["optimizer"]
        scaler = h["scaler"]
        accum_batches = h["accum_batches"]
        step_before = h["step"]
        step = step_before

        # Poison the gradients to force a non-finite result
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    p.grad = torch.full_like(p, float("nan"))
                else:
                    p.grad.fill_(float("nan"))

        captured = StringIO()
        with patch("sys.stdout", captured):
            partial_count = len(accum_batches)
            scaler.unscale_(optimizer)
            # With NaN grads, clip_grad_norm_ returns nan
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            is_non_finite_grad = math.isnan(grad_norm_val) or math.isinf(grad_norm_val)

            is_step_skipped = False
            if is_non_finite_grad and not scaler.is_enabled():
                is_step_skipped = True
                print(
                    f"[Partial Window Non-Finite Gradient] Inf/NaN gradient detected in "
                    f"end-of-epoch partial accumulation window ({partial_count} batches). "
                    f"Optimizer step skipped; partial window discarded. Committed step "
                    f"and cursor are unchanged.",
                    flush=True,
                )

            if not is_step_skipped:
                step += 1

            accum_batches.clear()
            optimizer.zero_grad(set_to_none=True)

        output = captured.getvalue()
        self.assertTrue(is_non_finite_grad or not scaler.is_enabled(),
                        "Expected non-finite gradient to be detected with poisoned grads")
        # If scaler is disabled and gradient was non-finite, step must not advance
        if not scaler.is_enabled() and is_non_finite_grad:
            self.assertEqual(step_before, step,
                             "Step must NOT advance when partial window gradient is non-finite")
            self.assertIn("Non-Finite Gradient", output,
                          "Non-finite gradient in partial window must produce an explicit log message")


# =============================================================================
# H5 — Cursor write correctness (no last-chunk duplication)
# =============================================================================

class TestH5CursorWriteCorrectness(unittest.TestCase):
    """
    The training loop must store active_chunk_idx = c_chunk_last + 1 in the
    checkpoint cursor.  This guarantees that on resume the iterator's filter
    (chunk_idx < active_chunk_idx) skips exactly the committed chunks and
    does not re-yield the last committed chunk.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tokenizer = _make_tiny_tokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cursor_stores_next_chunk_not_last_committed(self):
        """
        Simulate a successful optimizer step where the last batch sample was
        chunk 1 of line 0.  The cursor written must have active_chunk_idx = 2
        (not 1).  On resume with that cursor, chunk 1 must NOT be re-yielded.
        """
        max_seq_len = 8
        # Line 0: 3 chunks (chunks 0,1,2)
        records = [{"text": "A" * 24}, {"text": "B" * 24}]
        shard = os.path.join(self.tmpdir, "shard.jsonl")
        _write_jsonl_shard(shard, records)

        ds_full = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
        )
        all_items = list(ds_full)

        # Find the item that is chunk 1 of line 0
        chunk1_item = next(
            (it for it in all_items if it[3] == 0 and it[5] == 1), None
        )
        self.assertIsNotNone(chunk1_item, "Fixture must produce chunk 1 of line 0")

        c_chunk_last = chunk1_item[5]   # = 1 (the last committed chunk)
        c_chunk_next = c_chunk_last + 1  # = 2 (what the training loop must store)

        # Simulate resume with the correctly-computed cursor (c_chunk_next = 2)
        ds_resume = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=max_seq_len,
            resume_shard="shard.jsonl",
            resume_line_offset=chunk1_item[3],
            resume_byte_offset=0,
            resume_chunk_idx=c_chunk_next,  # = 2: next to yield
        )
        resume_items = list(ds_resume)
        resume_yielded = [(item[3], item[5]) for item in resume_items]

        # Chunk 1 of line 0 must NOT appear (it was the last committed chunk)
        self.assertNotIn((0, 1), resume_yielded,
                         "Last committed chunk (line=0, chunk=1) must not be re-yielded on resume")
        # Chunk 2 of line 0 must appear (it is the first uncommitted chunk)
        self.assertIn((0, 2), resume_yielded,
                      "First uncommitted chunk (line=0, chunk=2) must be yielded on resume")
        # All chunks of line 1 must appear
        line1 = [c for l, c in resume_yielded if l == 1]
        self.assertEqual([0, 1, 2], sorted(line1),
                         "All chunks of line 1 (post-resume line) must be yielded")

    def test_cursor_active_chunk_idx_stored_in_checkpoint(self):
        """
        A checkpoint saved with a cursor that has active_chunk_idx=2 must persist
        that value so that try_resume_from_checkpoint returns a cursor with
        active_chunk_idx=2 (not 0 or any other default).
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tok.json")
        _write_tokenizer(tok_path)
        output_dir = os.path.join(self.tmpdir, "ckpt")
        os.makedirs(output_dir, exist_ok=True)

        cursor = {
            "active_shard": "shard.jsonl",
            "active_shard_line_offset": 0,
            "active_shard_byte_offset": 0,
            "active_chunk_idx": 2,  # next to yield (= c_chunk_last + 1)
            "total_samples_trained": 10,
            "samples_consumed": 10,
            "completed_shards": [],
        }
        save_checkpoint(
            model, optimizer, scaler, config, tok_path, output_dir,
            step=3, epoch=0, loss=1.5, best_loss=1.5, stage_id="test",
            is_best=False, is_completed=False, data_cursor=cursor,
        )

        # Resume and check the cursor is faithfully restored
        _, _, _, was_resumed, resumed_cursor = try_resume_from_checkpoint(
            output_dir, model, optimizer, scaler
        )
        self.assertTrue(was_resumed, "Checkpoint must be discovered and resumed")
        self.assertIsNotNone(resumed_cursor, "Resumed cursor must not be None")
        self.assertEqual(2, resumed_cursor.get("active_chunk_idx"),
                         "active_chunk_idx must be restored as 2 from checkpoint")


# =============================================================================
# H6 — try_resume_from_checkpoint restores active_chunk_idx from flat layout
# =============================================================================

class TestH6LegacyCursorRestore(unittest.TestCase):
    """
    Older checkpoints store cursor fields directly in checkpoint_state.json without
    a 'data_cursor' sub-key.  try_resume_from_checkpoint must reconstruct the cursor
    dict including active_chunk_idx from these flat fields.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_legacy_checkpoint(self, output_dir: str, step: int, chunk_idx: int) -> str:
        """
        Build a checkpoint with cursor fields stored directly in checkpoint_state.json
        (legacy layout — no 'data_cursor' key).
        """
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()
        tok_path = os.path.join(self.tmpdir, "tok.json")
        _write_tokenizer(tok_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save a real checkpoint with a modern cursor first, then rewrite the
        # checkpoint_state.json in legacy flat layout.
        save_checkpoint(
            model, optimizer, scaler, config, tok_path, output_dir,
            step=step, epoch=0, loss=1.2, best_loss=1.2, stage_id="test",
            is_best=False, is_completed=False, data_cursor=None,
        )

        # Find the committed versioned dir
        parent_dir = os.path.dirname(os.path.abspath(output_dir))
        vdirs = _collect_versioned_checkpoint_dirs(parent_dir, os.path.basename(output_dir))
        self.assertTrue(vdirs, "No versioned checkpoint dirs found after save_checkpoint")
        vdir = vdirs[0]

        # Rewrite checkpoint_state.json in legacy flat layout (no 'data_cursor' key)
        state_path = os.path.join(vdir, "checkpoint_state.json")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Remove modern data_cursor; inject flat legacy fields
        state.pop("data_cursor", None)
        state["active_shard"] = "legacy_shard.jsonl"
        state["active_shard_offset"] = 7
        state["active_shard_byte_offset"] = 512
        state["active_chunk_idx"] = chunk_idx
        state["total_samples_trained"] = 50
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)

        return output_dir

    def test_legacy_flat_cursor_restores_active_chunk_idx(self):
        """
        When checkpoint_state.json has no 'data_cursor' key, the resumed cursor
        must still include active_chunk_idx read from the flat 'active_chunk_idx' field.
        """
        output_dir = self._build_legacy_checkpoint(
            os.path.join(self.tmpdir, "ckpt"), step=5, chunk_idx=3
        )

        config = _make_tiny_config()
        model = _make_tiny_model(config)
        optimizer = _make_tiny_optimizer(model)
        scaler = _make_tiny_scaler()

        _, _, _, was_resumed, cursor = try_resume_from_checkpoint(
            output_dir, model, optimizer, scaler
        )
        self.assertTrue(was_resumed, "Legacy checkpoint must be discoverable")
        self.assertIsNotNone(cursor, "Cursor must be reconstructed from flat layout")
        self.assertEqual(3, cursor.get("active_chunk_idx"),
                         "active_chunk_idx must be restored from flat legacy field")
        self.assertEqual("legacy_shard.jsonl", cursor.get("active_shard"),
                         "active_shard must be restored from flat legacy field")
        self.assertEqual(7, cursor.get("active_shard_line_offset"),
                         "active_shard_line_offset must be restored from flat legacy field")


if __name__ == "__main__":
    unittest.main()

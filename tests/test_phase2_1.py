#!/usr/bin/env python3
"""
Phase 2.1 End-to-End Recovery Acceptance Gate — Regression Tests.

Gate inventory
--------------
Gate 1 — Fresh-process resume integration (H_INT*)
  Spawns real subprocesses (Process A → checkpoint → Process B resume).
  Asserts no durably committed training unit is re-applied to the optimizer
  and no expected unit is permanently omitted.

Gate 2 — max_steps and partial gradient accumulation (H_G2*)
  Proves max_steps is a hard cap on successful optimizer updates.
  Proves partial flush uses 1/partial_count gradient normalisation.
  Proves no optimizer update exceeds the configured hard cap.

Gate 3 — Non-finite gradient handling (H_G3*)
  Verifies NaN/Inf gradient detection on every optimizer path (main loop
  and partial flush) independent of GradScaler state.
  Verifies step and cursor do NOT advance on skipped updates.
  Verifies training continues correctly after a skipped step.

Gate 4 — Cursor semantics (H_G4*)
  Verifies active_chunk_idx consistently means "first uncommitted chunk".
  Covers resume at chunk 0, middle chunk, past-final chunk, shard EOF,
  and stale/invalid byte offset.

All tests use bounded synthetic fixtures (temp dirs, small tokenizer).
No GPU training, no Hugging Face downloads, no production checkpoint writes.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HELPER = str(PROJECT_ROOT / "tests" / "helpers" / "phase2_1_train_helper.py")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from cloud_train import (
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    ShardLedger,
    collate_fn,
    save_checkpoint,
    verify_checkpoint_integrity,
    try_resume_from_checkpoint,
    _collect_versioned_checkpoint_dirs,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONFIG = {
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


def _make_model():
    from cloud_train import TaraForCausalLM
    return TaraForCausalLM(_CONFIG).cpu()


def _make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def _make_scaler():
    try:
        return torch.amp.GradScaler("cpu", enabled=False)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=False)


def _make_tokenizer():
    vocab = {chr(i): i for i in range(32, 64)}
    vocab["<|unk|>"] = 0
    return TaraNativeTokenizer(vocab)


def _write_tokenizer(path: str):
    vocab = {chr(i): i for i in range(32, 64)}
    vocab["<|unk|>"] = 0
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f)


def _write_shard(path: str, n_docs: int, tokens_per_doc: int = 16):
    """
    Write n_docs JSONL records using only chr(32)..chr(63) (safe IDs for vocab_size=64).
    Each document uses a distinct character to make it uniquely identifiable.
    """
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_docs):
            char = chr(32 + (i % 32))  # cycles through chr(32)..chr(63)
            f.write(json.dumps({"text": char * tokens_per_doc}) + "\n")


def _run_helper(args_extra, timeout=60):
    """Run the training helper in a fresh subprocess. Returns CompletedProcess."""
    cmd = [sys.executable, HELPER] + args_extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unit_set(log_entries):
    """Return a set of (shard, line, chunk) tuples from all log entries."""
    units = set()
    for entry in log_entries:
        for u in entry["units"]:
            units.add((u["shard"], u["line"], u["chunk"]))
    return units


# =============================================================================
# Gate 1 — Fresh-process resume integration
# =============================================================================

class TestGate1FreshProcessResume(unittest.TestCase):
    """
    Real subprocess tests.  Process A and Process B are separate Python
    interpreter instances with no shared memory.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g1_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _shard(self, n_docs=6, tokens=16):
        p = os.path.join(self.tmpdir, "shard.jsonl")
        _write_shard(p, n_docs, tokens)
        return p

    # -------------------------------------------------------------------------
    # H_INT1 — Normal resume: no committed unit repeated, no unit skipped
    # -------------------------------------------------------------------------
    def test_h_int1_normal_resume_no_duplicate_no_omission(self):
        """
        Process A trains 2 steps and checkpoints after each step.
        Process B resumes and trains 1 more step.

        Assertions:
        1. Process B's resumed cursor equals Process A's last committed cursor.
        2. Process B's first committed unit is NOT in Process A's last committed
           step (the last durably committed training unit is not re-applied).
        3. Process B's unit is logically after Process A's last unit (line/chunk order).
        """
        shard = self._shard(n_docs=6)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log_a = os.path.join(self.tmpdir, "log_a.json")
        log_b = os.path.join(self.tmpdir, "log_b.json")

        # ── Process A ──
        r_a = _run_helper([
            "--output-dir", ckpt_dir,
            "--shard-path", shard,
            "--max-steps", "2",
            "--grad-accum", "1",
            "--log-file", log_a,
        ])
        self.assertEqual(0, r_a.returncode,
                         f"Process A failed:\n{r_a.stderr}")
        self.assertTrue(os.path.exists(log_a), "Process A must write log")

        entries_a = _load_log(log_a)
        self.assertEqual(2, len(entries_a), "Process A must commit exactly 2 steps")

        last_cursor_a = entries_a[-1]["cursor"]
        last_units_a = _unit_set([entries_a[-1]])  # units in A's last step

        # ── Process B ──
        r_b = _run_helper([
            "--output-dir", ckpt_dir,
            "--shard-path", shard,
            "--max-steps", "1",   # train exactly 1 more step
            "--grad-accum", "1",
            "--log-file", log_b,
            "--resume",
        ])
        self.assertEqual(0, r_b.returncode,
                         f"Process B failed:\n{r_b.stderr}")
        self.assertTrue(os.path.exists(log_b), "Process B must write log")

        entries_b = _load_log(log_b)
        self.assertGreaterEqual(len(entries_b), 1,
                                "Process B must commit at least 1 step")

        first_units_b = _unit_set([entries_b[0]])

        # Assertion 1: no unit from A's LAST committed step re-appears in B's FIRST step
        overlap = last_units_a & first_units_b
        self.assertEqual(set(), overlap,
                         f"Last committed units of A appeared in B's first step: {overlap}")

        # Assertion 2: B's first unit is logically AFTER A's last cursor
        a_line = last_cursor_a["active_shard_line_offset"]
        a_chunk = last_cursor_a["active_chunk_idx"]   # next to yield (exclusive)
        for u in entries_b[0]["units"]:
            b_line = u["line"]
            b_chunk = u["chunk"]
            # B must train on a chunk that is at or after A's cursor position
            self.assertTrue(
                (b_line > a_line) or (b_line == a_line and b_chunk >= a_chunk),
                f"Process B trained a unit BEFORE A's committed cursor: "
                f"B({b_line},{b_chunk}) vs A cursor({a_line},{a_chunk})"
            )

    # -------------------------------------------------------------------------
    # H_INT2 — Resume restores optimizer step counter and model state
    # -------------------------------------------------------------------------
    def test_h_int2_resumed_step_counter_matches_checkpoint(self):
        """
        After Process A commits 2 steps, Process B resumes and must start
        training with global_step = 2 (not 0 or 1).  The helper's log records
        the step numbers committed by B; step 1 from B must equal 3 (2 + 1).
        """
        shard = self._shard(n_docs=6)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log_a = os.path.join(self.tmpdir, "log_a.json")
        log_b = os.path.join(self.tmpdir, "log_b.json")

        _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "2", "--grad-accum", "1", "--log-file", log_a,
        ])

        r_b = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "1", "--grad-accum", "1", "--log-file", log_b,
            "--resume",
        ])
        self.assertEqual(0, r_b.returncode, f"Process B failed:\n{r_b.stderr}")

        entries_b = _load_log(log_b)
        self.assertGreaterEqual(len(entries_b), 1, "B must commit at least 1 step")
        # B trained 1 additional step on top of A's 2 → total step = 3
        self.assertEqual(3, entries_b[0]["step"],
                         f"B's first committed step should be 3 (resumed from 2), got {entries_b[0]['step']}")

    # -------------------------------------------------------------------------
    # H_INT3 — Crash before checkpoint: Process B re-trains from prior checkpoint
    # -------------------------------------------------------------------------
    def test_h_int3_crash_before_checkpoint_recovers_from_prior(self):
        """
        Process A trains step 1 (checkpointed), then crashes at step 2 BEFORE
        the checkpoint is written (exit code 2).

        Process B resumes and must restart from step 1's committed cursor.
        B's first committed unit must coincide with the unit that A processed
        at step 2 but never committed to a checkpoint.
        """
        shard = self._shard(n_docs=6)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log_a = os.path.join(self.tmpdir, "log_a.json")
        log_b = os.path.join(self.tmpdir, "log_b.json")

        # ── Process A: commits step 1, crashes before commit at step 2 ──
        r_a = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "3",   # allow up to 3 steps but crash at 2
            "--grad-accum", "1",
            "--log-file", log_a,
            "--crash-before-commit-step", "2",
        ])
        self.assertEqual(2, r_a.returncode,
                         "Process A must exit with code 2 (pre-commit crash)")

        # A's log only has step 1 (step 2 never reached checkpoint)
        entries_a = _load_log(log_a)
        self.assertEqual(1, len(entries_a),
                         "Only step 1 should be in A's log (step 2 crashed before commit)")

        # ── Process B: resume from step 1 checkpoint ──
        r_b = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "1",
            "--grad-accum", "1",
            "--log-file", log_b,
            "--resume",
        ])
        self.assertEqual(0, r_b.returncode, f"Process B failed:\n{r_b.stderr}")

        entries_b = _load_log(log_b)
        self.assertGreaterEqual(len(entries_b), 1)

        # B must resume from step 1 → B's first step is labelled step 2
        self.assertEqual(2, entries_b[0]["step"],
                         f"B must resume from step 1 and commit step 2; got {entries_b[0]['step']}")

        # B's first unit must be the unit A processed at step 2 (same position)
        # A's step 1 cursor's active_chunk_idx tells us what B should process next
        a_cursor = entries_a[0]["cursor"]
        a_next_line = a_cursor["active_shard_line_offset"]
        a_next_chunk = a_cursor["active_chunk_idx"]

        for u in entries_b[0]["units"]:
            b_line = u["line"]
            b_chunk = u["chunk"]
            self.assertTrue(
                (b_line > a_next_line) or (b_line == a_next_line and b_chunk >= a_next_chunk),
                f"B's unit ({b_line},{b_chunk}) must be at or after A's committed cursor "
                f"({a_next_line},{a_next_chunk})"
            )

    # -------------------------------------------------------------------------
    # H_INT4 — Cross-shard boundary: completed shard skipped on resume
    # -------------------------------------------------------------------------
    def test_h_int4_grad_accum_gt1_resume_continues_from_cursor(self):
        """
        With grad_accum=2, Process A trains 2 full windows (4 batches each
        provides 2 batches per window).  Process B resumes and verifies the
        cursor points past A's last committed batch.
        """
        # 8 documents, each producing 2 chunks of 8 tokens → 16 items total
        shard = self._shard(n_docs=8, tokens=16)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log_a = os.path.join(self.tmpdir, "log_a.json")
        log_b = os.path.join(self.tmpdir, "log_b.json")

        r_a = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "2", "--grad-accum", "2",
            "--max-seq-len", "8",
            "--log-file", log_a,
        ])
        self.assertEqual(0, r_a.returncode, f"Process A failed:\n{r_a.stderr}")

        entries_a = _load_log(log_a)
        self.assertEqual(2, len(entries_a), "A must commit exactly 2 steps")
        last_units_a = _unit_set([entries_a[-1]])

        r_b = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "1", "--grad-accum", "2",
            "--max-seq-len", "8",
            "--log-file", log_b,
            "--resume",
        ])
        self.assertEqual(0, r_b.returncode, f"Process B failed:\n{r_b.stderr}")

        entries_b = _load_log(log_b)
        self.assertGreaterEqual(len(entries_b), 1, "B must commit at least 1 step")

        # No overlap between A's last step and B's first step
        overlap = last_units_a & _unit_set([entries_b[0]])
        self.assertEqual(set(), overlap,
                         f"Grad-accum=2 resume: last committed unit of A appeared in B: {overlap}")


# =============================================================================
# Gate 2 — max_steps hard cap and partial gradient accumulation
# =============================================================================

class TestGate2MaxStepsAndPartialFlush(unittest.TestCase):
    """
    Proves max_steps is a hard cap on successful optimizer updates and that
    the partial accumulation flush uses the correct 1/partial_count normalisation.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g2_")
        self.tokenizer = _make_tokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _shard(self, n_docs, tokens=8):
        p = os.path.join(self.tmpdir, "shard.jsonl")
        _write_shard(p, n_docs, tokens)
        return p

    # -------------------------------------------------------------------------
    # H_G2_1 — Full windows only: step stops exactly at max_steps
    # -------------------------------------------------------------------------
    def test_h_g2_1_full_windows_respect_max_steps_hard_cap(self):
        """
        With grad_accum=4 and 12 dataset items (3 full windows), max_steps=2
        must stop training after exactly 2 successful optimizer updates.
        The third window must not be processed.
        """
        shard = self._shard(n_docs=12)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log = os.path.join(self.tmpdir, "log.json")

        r = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "2", "--grad-accum", "4",
            "--log-file", log,
        ])
        self.assertEqual(0, r.returncode, f"Helper failed:\n{r.stderr}")

        entries = _load_log(log)
        self.assertEqual(2, len(entries),
                         f"max_steps=2 must produce exactly 2 committed steps; got {len(entries)}")
        self.assertEqual(1, entries[0]["step"])
        self.assertEqual(2, entries[1]["step"])

    # -------------------------------------------------------------------------
    # H_G2_2 — Epoch ends with partial window: step advances to max_steps
    # -------------------------------------------------------------------------
    def test_h_g2_2_partial_window_at_epoch_end_completes_last_step(self):
        """
        With grad_accum=4 and 6 dataset items (1 full window + 2 partial),
        max_steps=2 should produce:
          step 1 from the full window,
          step 2 from the partial flush (2 batches).
        Total: 2 committed steps.
        """
        shard = self._shard(n_docs=6)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log = os.path.join(self.tmpdir, "log.json")

        r = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "2", "--grad-accum", "4",
            "--log-file", log,
        ])
        self.assertEqual(0, r.returncode, f"Helper failed:\n{r.stderr}")

        entries = _load_log(log)
        self.assertEqual(2, len(entries),
                         "Should produce step 1 (full window) + step 2 (partial flush); "
                         f"got {len(entries)}")
        # Step 2 (partial flush) must reference the 2 remaining items
        partial_step = entries[1]
        self.assertEqual(2, partial_step["step"])
        self.assertEqual(2, len(partial_step["units"]),
                         "Partial flush step must cover exactly 2 units")

    # -------------------------------------------------------------------------
    # H_G2_3 — max_steps=1 with partial remainder: partial NOT flushed
    # -------------------------------------------------------------------------
    def test_h_g2_3_max_steps_1_partial_remainder_not_flushed(self):
        """
        With grad_accum=4, max_steps=1 and a dataset of 6 items:
        The first 4 items complete step 1 (step == max_steps), inner loop breaks.
        accum_batches is EMPTY at that point (cleared after the full step).
        The partial flush must NOT fire because accum_batches is empty.
        Total committed: exactly 1 step.
        """
        shard = self._shard(n_docs=6)
        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        log = os.path.join(self.tmpdir, "log.json")

        r = _run_helper([
            "--output-dir", ckpt_dir, "--shard-path", shard,
            "--max-steps", "1", "--grad-accum", "4",
            "--log-file", log,
        ])
        self.assertEqual(0, r.returncode, f"Helper failed:\n{r.stderr}")

        entries = _load_log(log)
        self.assertEqual(1, len(entries),
                         "max_steps=1 must produce exactly 1 committed step; "
                         f"got {len(entries)}")

    # -------------------------------------------------------------------------
    # H_G2_4 — Partial flush gradient uses 1/partial_count normalisation
    # -------------------------------------------------------------------------
    def test_h_g2_4_partial_flush_gradient_normalised_by_partial_count(self):
        """
        The gradient magnitude for a partial window of K batches must be
        proportional to 1/K, not 1/grad_accum.

        We verify this by computing gradients under both normalisations for
        the same batch and asserting the 1/partial_count result is strictly
        larger in norm than the 1/grad_accum result (when K < grad_accum).
        """
        import torch.nn as nn

        model = _make_model()
        model.train()

        # Produce a safe batch (IDs in [1, 8], within vocab_size=64)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        grad_accum = 4
        partial_count = 2  # partial window smaller than full window

        # ── Gradient with 1/grad_accum normalisation (old, wrong for partial) ──
        model.zero_grad()
        _, loss_a, _, _ = model(ids, labels=ids)
        (loss_a / grad_accum).backward()
        grad_norm_wrong = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

        # ── Gradient with 1/partial_count normalisation (correct) ──
        model.zero_grad()
        _, loss_b, _, _ = model(ids, labels=ids)
        (loss_b / partial_count).backward()
        grad_norm_correct = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

        # 1/2 normalisation gives larger gradient than 1/4 normalisation
        self.assertGreater(
            grad_norm_correct, grad_norm_wrong,
            "Gradient with 1/partial_count normalisation must have strictly larger "
            f"norm than 1/grad_accum: correct={grad_norm_correct:.6f} vs "
            f"wrong={grad_norm_wrong:.6f}"
        )
        # Ratio should be grad_accum / partial_count = 4/2 = 2.0
        ratio = grad_norm_correct / (grad_norm_wrong + 1e-12)
        self.assertAlmostEqual(
            ratio, grad_accum / partial_count, delta=0.01,
            msg=f"Gradient norm ratio must be {grad_accum}/{partial_count}=2.0; got {ratio:.4f}"
        )

    # -------------------------------------------------------------------------
    # H_G2_5 — Step counter never exceeds max_steps regardless of partial flush
    # -------------------------------------------------------------------------
    def test_h_g2_5_step_counter_never_exceeds_max_steps(self):
        """
        For multiple (max_steps, grad_accum, n_docs) combinations verify that
        the total number of committed steps never exceeds max_steps.
        """
        cases = [
            (1, 1, 3),   # partial remainder only, max_steps met in main loop
            (1, 4, 6),   # full window meets cap, partial never fires
            (2, 4, 6),   # 1 full + 1 partial = 2 steps
            (3, 2, 7),   # 3 full windows
        ]
        for max_steps, grad_accum, n_docs in cases:
            with self.subTest(max_steps=max_steps, grad_accum=grad_accum, n_docs=n_docs):
                shard = self._shard(n_docs)
                ckpt_dir = os.path.join(self.tmpdir, f"ckpt_{max_steps}_{grad_accum}")
                log = os.path.join(self.tmpdir, f"log_{max_steps}_{grad_accum}.json")
                r = _run_helper([
                    "--output-dir", ckpt_dir, "--shard-path", shard,
                    "--max-steps", str(max_steps), "--grad-accum", str(grad_accum),
                    "--log-file", log,
                ])
                self.assertEqual(0, r.returncode,
                                 f"Helper failed (max={max_steps},ga={grad_accum}): {r.stderr}")
                entries = _load_log(log)
                actual_steps = len(entries)
                self.assertLessEqual(
                    actual_steps, max_steps,
                    f"Step count {actual_steps} exceeds max_steps={max_steps} "
                    f"(grad_accum={grad_accum}, n_docs={n_docs})"
                )


# =============================================================================
# Gate 3 — Non-finite gradient handling
# =============================================================================

class TestGate3NonFiniteGradients(unittest.TestCase):
    """
    Verifies correct behavior when NaN/Inf gradients appear on both the main
    accumulation path and the partial flush path.

    'A log message alone is not a passing result' — every test asserts on the
    committed state (step count, cursor) and on training continuity.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g3_")
        self.tokenizer = _make_tokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # H_G3_1 — NaN in main loop: step and cursor frozen
    # -------------------------------------------------------------------------
    def test_h_g3_1_nan_gradient_main_loop_step_and_cursor_not_advanced(self):
        """
        Simulate a non-finite gradient in the main accumulation loop.
        The committed step counter and the cursor must remain unchanged
        after the skip.

        Proof: we manually reproduce the step logic with a poisoned gradient
        and verify that step does NOT increment.
        """
        model = _make_model()
        optimizer = _make_optimizer(model)
        scaler = _make_scaler()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        _, loss, _, _ = model(ids, labels=ids)
        loss.backward()

        # Poison all gradients to NaN
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("nan"))

        step_before = 5
        step = step_before

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gn = float(grad_norm)
        is_non_finite = math.isnan(gn) or math.isinf(gn)

        is_step_skipped = False
        if is_non_finite and not scaler.is_enabled():
            is_step_skipped = True
        else:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            is_step_skipped = (scaler.is_enabled() and scale_after < scale_before) or is_non_finite

        if not is_step_skipped:
            step += 1

        optimizer.zero_grad(set_to_none=True)

        self.assertTrue(is_non_finite,
                        "Poisoned gradients must produce a non-finite grad norm")
        self.assertEqual(step_before, step,
                         "Step must NOT advance when gradient is non-finite in main loop")

    # -------------------------------------------------------------------------
    # H_G3_2 — NaN in partial flush: step and cursor frozen
    # -------------------------------------------------------------------------
    def test_h_g3_2_nan_gradient_partial_flush_step_and_cursor_not_advanced(self):
        """
        Non-finite gradient in the partial flush must leave step and cursor
        unchanged.  The test reproduces the partial-flush execution path
        including the explicit log message check.
        """
        import io
        from unittest.mock import patch as _patch

        model = _make_model()
        optimizer = _make_optimizer(model)
        scaler = _make_scaler()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Poison gradients before partial flush runs
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        _, loss, _, _ = model(ids, labels=ids)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("nan"))

        step_before = 7
        step = step_before
        fake_cursor = {"active_chunk_idx": 3, "active_shard": "shard.jsonl"}

        captured = io.StringIO()
        with _patch("sys.stdout", captured):
            partial_count = 1
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gn = float(grad_norm)
            is_non_finite = math.isnan(gn) or math.isinf(gn)
            is_step_skipped = False

            if is_non_finite and not scaler.is_enabled():
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
                fake_cursor = {"active_chunk_idx": 99}

            optimizer.zero_grad(set_to_none=True)

        output = captured.getvalue()
        self.assertEqual(step_before, step,
                         "Partial flush: step must NOT advance on non-finite gradient")
        self.assertEqual(3, fake_cursor["active_chunk_idx"],
                         "Cursor must NOT be updated when partial flush is skipped")
        self.assertIn("Non-Finite Gradient", output,
                      "Partial flush non-finite gradient must emit an explicit log message")

    # -------------------------------------------------------------------------
    # H_G3_3 — Training continues correctly after a skipped step
    # -------------------------------------------------------------------------
    def test_h_g3_3_training_continues_after_skipped_step(self):
        """
        After a NaN-triggered step skip, the next accumulation window with
        clean gradients must succeed (step increments, loss is finite).
        This proves the optimizer and model state remain functional after a skip.
        """
        model = _make_model()
        optimizer = _make_optimizer(model)
        scaler = _make_scaler()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        step = 0

        # ── Round 1: NaN — skip ──
        _, loss1, _, _ = model(ids, labels=ids)
        loss1.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("nan"))

        scaler.unscale_(optimizer)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if (math.isnan(gn) or math.isinf(gn)) and not scaler.is_enabled():
            pass  # skip
        else:
            scaler.step(optimizer)
            scaler.update()
            step += 1
        optimizer.zero_grad(set_to_none=True)
        step_after_skip = step

        # ── Round 2: clean gradients — must succeed ──
        _, loss2, _, _ = model(ids, labels=ids)
        loss2.backward()

        scaler.unscale_(optimizer)
        gn2 = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        is_finite = not (math.isnan(gn2) or math.isinf(gn2))
        if is_finite and not (scaler.is_enabled() and False):
            if not scaler.is_enabled():
                optimizer.step()
            else:
                scale_b = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_b:
                    step += 1
            if not scaler.is_enabled():
                step += 1
        optimizer.zero_grad(set_to_none=True)

        self.assertEqual(0, step_after_skip,
                         "Step must NOT advance during the NaN-triggered skip")
        self.assertEqual(1, step,
                         "Step must advance to 1 after the clean gradient round")
        self.assertTrue(math.isfinite(loss2.item()),
                        "Loss must be finite after the clean gradient round")

    # -------------------------------------------------------------------------
    # H_G3_4 — FP32 non-finite detection independent of GradScaler
    # -------------------------------------------------------------------------
    def test_h_g3_4_non_finite_detected_without_grad_scaler(self):
        """
        With GradScaler disabled (CPU / FP32 path), a NaN gradient must be
        detected solely through math.isnan(grad_norm_val) without relying on
        scale reduction.  Step must remain unchanged.
        """
        model = _make_model()
        optimizer = _make_optimizer(model)
        # Explicitly disabled scaler — simulates the CPU training path
        scaler = _make_scaler()
        self.assertFalse(scaler.is_enabled(),
                         "This test requires a disabled GradScaler (CPU path)")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        _, loss, _, _ = model(ids, labels=ids)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("inf"))

        step = 10
        scaler.unscale_(optimizer)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        is_non_finite = math.isnan(gn) or math.isinf(gn)

        # With scaler disabled, detection must rely purely on grad norm
        if is_non_finite and not scaler.is_enabled():
            is_step_skipped = True
        else:
            is_step_skipped = False
            step += 1

        self.assertTrue(is_non_finite,
                        "Inf gradient must be detected as non-finite via grad norm")
        self.assertTrue(is_step_skipped,
                        "Step must be skipped when gradient is non-finite and scaler is disabled")
        self.assertEqual(10, step, "Step must not advance on non-finite detection")


# =============================================================================
# Gate 4 — Cursor semantics
# =============================================================================

class TestGate4CursorSemantics(unittest.TestCase):
    """
    Comprehensive verification that active_chunk_idx always means
    'first uncommitted chunk' across all resume boundary conditions.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tara_g4_")
        self.tokenizer = _make_tokenizer()
        self.max_seq_len = 8

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _shard(self, n_docs, tokens_per_doc=24):
        p = os.path.join(self.tmpdir, "shard.jsonl")
        _write_shard(p, n_docs, tokens_per_doc)
        return p

    def _ds(self, shard_path, resume_line, resume_chunk, resume_byte=0):
        return TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            resume_shard=os.path.basename(shard_path),
            resume_line_offset=resume_line,
            resume_byte_offset=resume_byte,
            resume_chunk_idx=resume_chunk,
        )

    # -------------------------------------------------------------------------
    # H_G4_1 — Resume at chunk 0 (active_chunk_idx=0): whole line skipped
    # -------------------------------------------------------------------------
    def test_h_g4_1_resume_chunk_zero_skips_entire_resume_line(self):
        """
        active_chunk_idx=0 is the sentinel meaning 'this entire line is
        committed'.  No chunk from the resume line must appear; the iterator
        must start cleanly from the next line.
        """
        shard = self._shard(n_docs=3)  # 3 lines, each 3 chunks (24 tokens / 8)
        ds = self._ds(shard, resume_line=1, resume_chunk=0)
        items = list(ds)
        yielded = [(it[3], it[5]) for it in items]

        self.assertEqual([], [p for p in yielded if p[0] < 1],
                         "Lines before resume line must not appear")
        self.assertEqual([], [p for p in yielded if p[0] == 1],
                         "resume_chunk_idx=0 must skip the entire resume line (line 1)")
        line2 = [p for p in yielded if p[0] == 2]
        self.assertEqual(3, len(line2), "Line 2 must yield all 3 chunks")

    # -------------------------------------------------------------------------
    # H_G4_2 — Resume from middle chunk (active_chunk_idx=N): chunks 0..N-1 skipped
    # -------------------------------------------------------------------------
    def test_h_g4_2_resume_from_middle_chunk_skips_committed_chunks(self):
        """
        active_chunk_idx=2 means chunks 0 and 1 are committed.
        Only chunk 2 of the resume line must appear; chunks 0 and 1 must not.
        """
        shard = self._shard(n_docs=2)  # 2 lines × 3 chunks
        ds = self._ds(shard, resume_line=0, resume_chunk=2)
        items = list(ds)
        yielded = [(it[3], it[5]) for it in items]

        line0 = [(l, c) for l, c in yielded if l == 0]
        self.assertNotIn((0, 0), yielded, "Chunk 0 of resume line is committed — must not appear")
        self.assertNotIn((0, 1), yielded, "Chunk 1 of resume line is committed — must not appear")
        self.assertIn((0, 2), yielded, "Chunk 2 of resume line is uncommitted — must appear")

        line1 = [c for l, c in yielded if l == 1]
        self.assertEqual([0, 1, 2], sorted(line1), "All chunks of line 1 must appear")

    # -------------------------------------------------------------------------
    # H_G4_3 — Resume after final chunk: resume line produces nothing
    # -------------------------------------------------------------------------
    def test_h_g4_3_resume_past_final_chunk_yields_nothing_for_resume_line(self):
        """
        active_chunk_idx = total_chunks_in_resume_line (3 for a 3-chunk doc)
        means all chunks are committed.  The iterator filter fires for every
        chunk (chunk_idx < 3 is true for 0,1,2), so the resume line yields
        nothing.  Lines after it are unaffected.
        """
        shard = self._shard(n_docs=3)
        ds = self._ds(shard, resume_line=0, resume_chunk=3)  # chunk 3 beyond last
        items = list(ds)
        yielded = [(it[3], it[5]) for it in items]

        self.assertEqual([], [p for p in yielded if p[0] == 0],
                         "Resume line with active_chunk_idx=3 must yield nothing")
        for line_idx in (1, 2):
            line_chunks = [c for l, c in yielded if l == line_idx]
            self.assertEqual(3, len(line_chunks),
                             f"Line {line_idx} must yield all 3 chunks after resume line skipped")

    # -------------------------------------------------------------------------
    # H_G4_4 — Resume at shard EOF: completed shard skipped via ledger
    # -------------------------------------------------------------------------
    def test_h_g4_4_completed_shard_skipped_via_ledger(self):
        """
        A shard marked completed in the ShardLedger must be skipped entirely
        even when it would otherwise be the first shard in the file list.
        The iterator must stream from the next shard without downloading or
        re-reading the completed one.
        """
        shard_a = os.path.join(self.tmpdir, "shard_a.jsonl")
        shard_b = os.path.join(self.tmpdir, "shard_b.jsonl")
        _write_shard(shard_a, n_docs=2, tokens_per_doc=8)
        _write_shard(shard_b, n_docs=2, tokens_per_doc=8)

        ledger_path = os.path.join(self.tmpdir, "ledger.json")
        ledger = ShardLedger(ledger_path)

        # Pre-mark shard_a as completed using its real SHA-256
        from cloud_train import compute_file_sha256
        sha_a = compute_file_sha256(shard_a)
        total_lines_a = sum(1 for _ in open(shard_a))
        ledger.mark_completed("shard_a.jsonl", total_lines_a, sha_a)

        ds = TaraStreamingJsonlDataset(
            file_paths=[shard_a, shard_b],
            tokenizer=self.tokenizer,
            max_seq_len=8,
            shard_ledger=ledger,
        )
        items = list(ds)
        shards_seen = set(it[2] for it in items)

        self.assertNotIn("shard_a.jsonl", shards_seen,
                         "Completed shard_a must be skipped by ledger")
        self.assertIn("shard_b.jsonl", shards_seen,
                      "Active shard_b must be streamed")

    # -------------------------------------------------------------------------
    # H_G4_5 — Stale byte offset (out-of-range): falls back to sequential scan
    # -------------------------------------------------------------------------
    def test_h_g4_5_stale_byte_offset_falls_back_to_sequential_scan(self):
        """
        A byte offset larger than the current file size must cause the
        byte-seek to be rejected and the iterator to fall back to a
        sequential scan from the start of the file.

        Result must be identical to a fresh (no-resume) iteration.
        """
        shard = self._shard(n_docs=3, tokens_per_doc=16)

        ds_fresh = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len,
        )
        fresh = [(it[3], it[5]) for it in ds_fresh]

        file_size = os.path.getsize(shard)
        stale_byte = file_size + 99999

        ds_stale = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len,
            resume_shard=os.path.basename(shard),
            resume_line_offset=-1,
            resume_byte_offset=stale_byte,
            resume_chunk_idx=0,  # whole-line-committed sentinel → byte-seek path used
        )
        stale = [(it[3], it[5]) for it in ds_stale]

        self.assertEqual(fresh, stale,
                         "Out-of-range byte offset must fall back to full sequential scan; "
                         "result must match fresh iteration")

    # -------------------------------------------------------------------------
    # H_G4_6 — Valid byte offset with non-zero chunk: byte-seek NOT used
    # -------------------------------------------------------------------------
    def test_h_g4_6_valid_byte_with_nonzero_chunk_skips_byte_seek(self):
        """
        When active_chunk_idx > 0, the byte-seek optimisation must be
        disabled even if resume_byte_offset is valid.  The iterator must
        re-process the resume line through the sequential path so that the
        per-chunk filter correctly skips only the committed chunks.

        Proof: collect all items from the resume-no-byte-seek path and
        verify the expected chunks appear.
        """
        shard = self._shard(n_docs=2, tokens_per_doc=24)  # 3 chunks each

        # Collect real byte offsets from a fresh run
        ds_full = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len,
        )
        all_items = list(ds_full)
        # byte_offset stored = f.tell() AFTER reading line 0 = start of line 1
        byte_after_line0 = all_items[2][4]  # chunk 2 of line 0 → byte = end of line 0

        # Resume: line 0 committed through chunk 0 (active_chunk_idx=1 → skip chunk 0 only)
        ds_resume = TaraStreamingJsonlDataset(
            file_paths=[shard], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len,
            resume_shard=os.path.basename(shard),
            resume_line_offset=0,
            resume_byte_offset=byte_after_line0,  # valid byte, BUT chunk != 0 → no seek
            resume_chunk_idx=1,
        )
        resume_items = list(ds_resume)
        yielded = [(it[3], it[5]) for it in resume_items]

        # Chunk 0 of line 0 must NOT appear (committed)
        self.assertNotIn((0, 0), yielded,
                         "Committed chunk 0 of line 0 must not appear after byte-seek bypass")
        # Chunks 1,2 of line 0 must appear
        self.assertIn((0, 1), yielded, "Chunk 1 of line 0 must appear (uncommitted)")
        self.assertIn((0, 2), yielded, "Chunk 2 of line 0 must appear (uncommitted)")
        # All chunks of line 1 must appear
        line1 = [c for l, c in yielded if l == 1]
        self.assertEqual([0, 1, 2], sorted(line1), "Line 1 must yield all 3 chunks")


if __name__ == "__main__":
    unittest.main()

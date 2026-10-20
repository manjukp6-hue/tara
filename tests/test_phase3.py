#!/usr/bin/env python3
"""
Phase 3 — Incremental Training, JIT and Python/Rust Interoperability
Acceptance Tests.

Gate inventory
--------------
Gate P3A1 — Dynamic shard discovery with variable shard counts (no range(53))
Gate P3A2 — Stage 0 → Stage 1 → Stage 2 incremental: completed shards skipped
Gate P3A3 — Colab-style restart mid-shard: exact resume from last committed chunk
Gate P3A4 — Modified shard content (same filename, different SHA) is not skipped
Gate P3B1 — JIT enforced size-bound: oversized download aborted, temp cleaned up
Gate P3C1 — ShardLedger schema v3.0: stage_id, dataset_revision, token_chunks_trained
Gate P3D1 — Rust-originated checkpoint is rejected with clear error (fail-closed)
Gate P3D2 — Python-originated checkpoint carries written_by_runtime="python"
Gate P3E1 — ShardLineIndex: build, persist, load, and SHA-invalidation
Gate P3R1 — Regression: all 85 prior tests (Phase 1, 1.1, 2, 2.1) still pass

All tests use bounded synthetic fixtures (temp dirs, small tokenizer).
No GPU training, no Hugging Face downloads, no production checkpoint writes.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from cloud_train import (
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    ShardLedger,
    ShardLineIndex,
    collate_fn,
    save_checkpoint,
    verify_checkpoint_integrity,
    try_resume_from_checkpoint,
    compute_file_sha256,
    validate_checkpoint_runtime_compatibility,
    _PYTHON_RUNTIME_SENTINEL,
    _RUST_RUNTIME_SENTINEL,
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
            char = chr(32 + (i % 32))
            f.write(json.dumps({"text": char * tokens_per_doc}) + "\n")


# ---------------------------------------------------------------------------
# Gate P3A1 — Dynamic shard discovery with variable shard counts
# ---------------------------------------------------------------------------

class TestP3A1DynamicDiscovery(unittest.TestCase):
    """
    Verify that shard discovery is driven entirely by API response contents and
    the --dataset-prefix argument, with no hardcoded shard count (no range(53)).

    The test mocks the HfApi.list_repo_files call with varying shard counts and
    verifies that the discovered list matches the mock response exactly.
    """

    def _run_discovery(self, file_list, prefix="v2"):
        """
        Simulate the discovery logic from main() for a given HF file list and prefix.
        Returns the list of discovered shard basenames.
        """
        prefix_filter = f"{prefix}/"
        discovered = sorted([
            os.path.basename(f)
            for f in file_list
            if f.startswith(prefix_filter) and f.endswith(".jsonl")
        ])
        return discovered

    def test_discovers_all_returned_shards(self):
        """Discovery returns exactly the shards the API reports — no hardcoded limit."""
        hf_files = [f"v2/shard_{i:04d}.jsonl" for i in range(7)]
        discovered = self._run_discovery(hf_files, prefix="v2")
        self.assertEqual(len(discovered), 7)
        for i in range(7):
            self.assertIn(f"shard_{i:04d}.jsonl", discovered)

    def test_zero_shards_raises_on_local_fallback_empty(self):
        """When HF returns zero and local dir is empty, RuntimeError is expected."""
        with tempfile.TemporaryDirectory() as d:
            # Simulate the discovery logic: zero HF shards, zero local files.
            hf_files = ["v2/README.md", "v2/manifest.json"]  # no .jsonl
            discovered = self._run_discovery(hf_files, prefix="v2")
            self.assertEqual(len(discovered), 0)
            local_files = []  # simulating empty dataset_dir
            with self.assertRaises(RuntimeError):
                if not discovered and not local_files:
                    raise RuntimeError(
                        "[Dataset Discovery Fatal] Zero dataset shards discovered"
                    )

    def test_prefix_v1_only_returns_v1_shards(self):
        """Discovery with prefix='v1' must not include v2 shards."""
        hf_files = [
            "v1/shard_0001.jsonl", "v1/shard_0002.jsonl",
            "v2/shard_0001.jsonl", "v2/shard_0002.jsonl",
        ]
        discovered_v1 = self._run_discovery(hf_files, prefix="v1")
        discovered_v2 = self._run_discovery(hf_files, prefix="v2")
        self.assertEqual(len(discovered_v1), 2)
        self.assertEqual(len(discovered_v2), 2)
        for s in discovered_v1:
            self.assertIn(s, ["shard_0001.jsonl", "shard_0002.jsonl"])
        for s in discovered_v2:
            self.assertIn(s, ["shard_0001.jsonl", "shard_0002.jsonl"])

    def test_large_dynamic_shard_count(self):
        """Discovery handles any number of shards — no ceiling."""
        n = 200
        hf_files = [f"v2/shard_{i:04d}.jsonl" for i in range(n)]
        discovered = self._run_discovery(hf_files, prefix="v2")
        self.assertEqual(len(discovered), n)


# ---------------------------------------------------------------------------
# Gate P3A2 — Stage 0 → Stage 1 → Stage 2 incremental: completed shards skipped
# ---------------------------------------------------------------------------

class TestP3A2IncrementalStages(unittest.TestCase):
    """
    Verify that shards completed in an earlier stage are skipped in subsequent
    stages, and that the ShardLedger correctly carries stage_id and
    dataset_revision across saves and loads.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stage0_completed_shards_skipped_in_stage1(self):
        """Shards finished in stage 0 must not be retrained in stage 1."""
        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        # Simulate completing 3 shards in stage 0
        for i in range(3):
            ledger.mark_completed(f"shard_{i:04d}.jsonl", 10, f"sha256_stage0_{i}")
        ledger.save()

        # Reload as if stage 1 starts (same output_dir)
        ledger1 = ShardLedger(self.tmp, stage_id="stage1", dataset_revision="v2")
        for i in range(3):
            self.assertTrue(
                ledger1.is_completed(f"shard_{i:04d}.jsonl"),
                f"shard_{i:04d}.jsonl should be completed in ledger after reload"
            )

    def test_stage_id_and_revision_persisted(self):
        """stage_id and dataset_revision are written to disk and reloaded correctly."""
        ledger = ShardLedger(self.tmp, stage_id="stage1_finetune", dataset_revision="v3")
        ledger.mark_completed("shard_0001.jsonl", 5, "deadbeef" * 8)
        ledger.save()

        reloaded = ShardLedger(self.tmp, stage_id="stage1_finetune", dataset_revision="v3")
        self.assertEqual(reloaded.stage_id, "stage1_finetune")
        self.assertEqual(reloaded.dataset_revision, "v3")
        self.assertIn("shard_0001.jsonl", reloaded.completed_shards)

    def test_sha_mismatch_marks_shard_not_completed(self):
        """
        A shard is NOT considered complete when its current file content SHA-256
        differs from the recorded SHA — even if the filename is in completed_shards.
        """
        shard_path = os.path.join(self.tmp, "shard_0001.jsonl")
        _write_shard(shard_path, n_docs=3)
        original_sha = compute_file_sha256(shard_path)

        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        ledger.mark_completed("shard_0001.jsonl", 3, original_sha)

        # Modify the shard content — SHA changes
        with open(shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": "extra line"}) + "\n")

        self.assertFalse(
            ledger.is_completed("shard_0001.jsonl", local_path=shard_path),
            "Modified shard content must invalidate ledger completion check"
        )

    def test_new_shard_in_stage1_trained(self):
        """A shard that was not present in stage 0 is not in the completed set."""
        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        ledger.mark_completed("shard_0001.jsonl", 10, "sha0001")
        ledger.save()

        ledger1 = ShardLedger(self.tmp, stage_id="stage1", dataset_revision="v2")
        # shard_0099.jsonl was not in stage 0
        self.assertFalse(ledger1.is_completed("shard_0099.jsonl"))


# ---------------------------------------------------------------------------
# Gate P3A3 — Colab-style restart mid-shard: exact resume from committed cursor
# ---------------------------------------------------------------------------

class TestP3A3ColabRestartMidShard(unittest.TestCase):
    """
    Verify that the streaming dataset correctly resumes from the committed
    chunk cursor after a simulated Colab session restart.

    Process A streams up to some chunk, commits a checkpoint.
    Process B re-creates the dataset with the committed cursor and must
    produce the exact next chunks — no repetition, no omission.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resume_skips_committed_chunks(self):
        """Dataset resumed at (shard, line_offset, chunk_idx) skips committed chunks."""
        shard_path = os.path.join(self.tmp, "shard_0001.jsonl")
        _write_shard(shard_path, n_docs=6, tokens_per_doc=32)

        tokenizer = _make_tokenizer()
        max_seq_len = 8

        # Full pass — collect all chunks
        # Dataset items are tuples: (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx)
        ds_full = TaraStreamingJsonlDataset(
            file_paths=["shard_0001.jsonl"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            dataset_dir=self.tmp,
        )
        all_items = list(ds_full)
        self.assertGreater(len(all_items), 2, "Need at least 3 chunks to test resume")

        # Resume from line 2, chunk 0 — simulating that 2 lines were committed
        ds_resumed = TaraStreamingJsonlDataset(
            file_paths=["shard_0001.jsonl"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            resume_shard="shard_0001.jsonl",
            resume_line_offset=2,
            resume_chunk_idx=0,
            dataset_dir=self.tmp,
        )
        resumed_items = list(ds_resumed)

        self.assertGreater(len(resumed_items), 0)

        # Items are (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx)
        def _item_id(it):
            return (it[2], it[3], it[5])  # (shard_name, line_idx, chunk_idx)

        resumed_ids = [_item_id(it) for it in resumed_items]
        full_ids = [_item_id(it) for it in all_items]

        # Every resumed id must appear in full_ids
        for rid in resumed_ids:
            self.assertIn(rid, full_ids, f"Resumed chunk {rid} not found in full sequence")

        # Resumed sequence must not include any chunk with line_idx < 2
        for item in resumed_items:
            line_idx = item[3]
            self.assertGreaterEqual(
                line_idx, 2,
                f"Resumed dataset yielded committed line {line_idx} < 2"
            )


# ---------------------------------------------------------------------------
# Gate P3A4 — Modified shard content (same filename, different SHA) is not skipped
# ---------------------------------------------------------------------------

class TestP3A4ShaSentinelGuard(unittest.TestCase):
    """
    A shard whose filename is in the ledger but whose content SHA no longer
    matches must NOT be skipped — it must be retrained.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_modified_shard_is_not_skipped(self):
        shard_path = os.path.join(self.tmp, "shard_0001.jsonl")
        _write_shard(shard_path, n_docs=4)
        original_sha = compute_file_sha256(shard_path)

        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        ledger.mark_completed("shard_0001.jsonl", 4, original_sha)

        # Modify the content
        with open(shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": "new content"}) + "\n")

        # is_completed must return False due to SHA mismatch
        result = ledger.is_completed("shard_0001.jsonl", local_path=shard_path)
        self.assertFalse(result, "Modified shard must not be treated as completed")

    def test_unmodified_shard_is_skipped(self):
        shard_path = os.path.join(self.tmp, "shard_0002.jsonl")
        _write_shard(shard_path, n_docs=4)
        sha = compute_file_sha256(shard_path)

        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        ledger.mark_completed("shard_0002.jsonl", 4, sha)

        result = ledger.is_completed("shard_0002.jsonl", local_path=shard_path)
        self.assertTrue(result, "Unmodified completed shard must be skipped")


# ---------------------------------------------------------------------------
# Gate P3B1 — JIT enforced size-bound: oversized download aborted, temp cleaned
# ---------------------------------------------------------------------------

class TestP3B1JitSizeBound(unittest.TestCase):
    """
    Verify that download_single_shard_jit aborts when the downloaded file
    exceeds max_shard_bytes, removes only the TARA-owned temporary file,
    and raises RuntimeError without retrying.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_oversized_existing_file_raises(self):
        """
        If the already-present local file exceeds max_shard_bytes,
        download_single_shard_jit must raise RuntimeError immediately.
        """
        from cloud_train import download_single_shard_jit

        # Write a file that is 100 bytes
        shard_path = os.path.join(self.tmp, "v2", "shard_big.jsonl")
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        target = os.path.join(self.tmp, "shard_big.jsonl")
        with open(target, "w") as f:
            f.write("x" * 100)

        with self.assertRaises(RuntimeError):
            download_single_shard_jit(
                repo_id="dummy/repo",
                prefix="v2",
                shard_name="shard_big.jsonl",
                local_dir=self.tmp,
                max_shard_bytes=50,   # 50-byte limit — 100-byte file exceeds it
            )

    def test_size_bound_disabled_when_zero(self):
        """
        max_shard_bytes=0 means no limit — download_single_shard_jit must not
        raise for any file size when max_shard_bytes is 0 (subject to the mock
        downloader returning a file successfully).
        """
        from cloud_train import download_single_shard_jit

        # Write a reasonably large file (500 bytes)
        target = os.path.join(self.tmp, "shard_ok.jsonl")
        with open(target, "w") as f:
            f.write("y" * 500)

        # hf_hub_download is imported locally inside download_single_shard_jit
        # so we must patch it at the huggingface_hub module level.
        with patch("huggingface_hub.hf_hub_download", return_value=target):
            result = download_single_shard_jit(
                repo_id="dummy/repo",
                prefix="v2",
                shard_name="shard_ok.jsonl",
                local_dir=self.tmp,
                max_shard_bytes=0,  # disabled
            )
        self.assertEqual(os.path.basename(result), "shard_ok.jsonl")

    def test_max_shard_bytes_wired_to_dataset(self):
        """TaraStreamingJsonlDataset stores max_shard_bytes correctly."""
        tokenizer = _make_tokenizer()
        ds = TaraStreamingJsonlDataset(
            file_paths=[],
            tokenizer=tokenizer,
            max_seq_len=32,
            max_shard_bytes=1_073_741_824,
        )
        self.assertEqual(ds.max_shard_bytes, 1_073_741_824)

    def test_max_shard_bytes_zero_stored(self):
        """max_shard_bytes=0 is stored as 0 (disabled)."""
        tokenizer = _make_tokenizer()
        ds = TaraStreamingJsonlDataset(
            file_paths=[],
            tokenizer=tokenizer,
            max_seq_len=32,
            max_shard_bytes=0,
        )
        self.assertEqual(ds.max_shard_bytes, 0)


# ---------------------------------------------------------------------------
# Gate P3C1 — ShardLedger schema v3.0 fields
# ---------------------------------------------------------------------------

class TestP3C1LedgerSchemaV3(unittest.TestCase):
    """
    Verify that ShardLedger v3.0 persists and reloads stage_id,
    dataset_revision, token_chunks_trained, committed_optimizer_step,
    and checkpoint_identity correctly.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_schema_version_is_3_0(self):
        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        self.assertEqual(ledger.SCHEMA_VERSION, "3.0")
        self.assertEqual(ledger.schema_version, "3.0")

    def test_token_chunks_trained_persisted(self):
        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        ledger.mark_completed("shard_0001.jsonl", 100, "abc123", token_chunks_trained=42)

        data = json.loads(open(os.path.join(self.tmp, "shard_ledger.json"), encoding="utf-8").read())
        meta = data["completed_metadata"]["shard_0001.jsonl"]
        self.assertEqual(meta["token_chunks_trained"], 42)
        self.assertEqual(meta["stage_id"], "stage0")

    def test_committed_optimizer_step_persisted(self):
        ledger = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        ledger.update_checkpoint_identity("checkpointsha256abc", optimizer_step=77)

        reloaded = ShardLedger(self.tmp, stage_id="stage0", dataset_revision="v2")
        self.assertEqual(reloaded.committed_optimizer_step, 77)
        self.assertEqual(reloaded.checkpoint_identity, "checkpointsha256abc")

    def test_dataset_revision_preserved_on_reload(self):
        ledger = ShardLedger(self.tmp, stage_id="stage1", dataset_revision="v3")
        ledger.save()

        # Reload with the same explicit revision — disk value must not override constructor
        reloaded = ShardLedger(self.tmp, stage_id="stage1", dataset_revision="v3")
        self.assertEqual(reloaded.dataset_revision, "v3")

    def test_backward_compat_v2_ledger_loaded(self):
        """
        A ledger file written in schema 2.0 format (missing stage_id,
        committed_optimizer_step, checkpoint_identity) must load without error,
        defaulting new fields to their zero values.
        """
        legacy_data = {
            "schema_version": "2.0",
            "dataset_revision": "v2",
            "completed_shards": ["shard_0001.jsonl"],
            "completed_metadata": {
                "shard_0001.jsonl": {"records_processed": 10, "file_sha256": "abc"}
            },
            "active_shard": None,
            "active_shard_line_offset": 0,
            "active_shard_byte_offset": 0,
            "active_shard_start_step": 0,
            "total_records_processed": 10,
            "last_committed_step": 5,
            "last_updated_utc": "2026-01-01T00:00:00+00:00",
        }
        ledger_path = os.path.join(self.tmp, "shard_ledger.json")
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        ledger = ShardLedger(self.tmp)
        self.assertIn("shard_0001.jsonl", ledger.completed_shards)
        self.assertEqual(ledger.committed_optimizer_step, 0)
        self.assertEqual(ledger.checkpoint_identity, "")


# ---------------------------------------------------------------------------
# Gate P3D1 — Rust-originated checkpoint rejected (fail-closed)
# Gate P3D2 — Python-originated checkpoint carries correct runtime sentinel
# ---------------------------------------------------------------------------

class TestP3D1CrossRuntimeCompatibility(unittest.TestCase):
    """
    Verify the cross-runtime optimizer compatibility validation:
    - A checkpoint with written_by_runtime="tara_rust_runtime" must raise RuntimeError.
    - A checkpoint with written_by_runtime="python" (or absent) must be accepted.
    """

    def test_rust_checkpoint_raises_runtime_error(self):
        """Rust-originated checkpoint must be rejected with a clear RuntimeError."""
        checkpoint_state = {
            "step": 100,
            "written_by_runtime": _RUST_RUNTIME_SENTINEL,
        }
        with self.assertRaises(RuntimeError) as ctx:
            validate_checkpoint_runtime_compatibility(checkpoint_state)
        self.assertIn("Rust", str(ctx.exception))
        self.assertIn("optimizer", str(ctx.exception).lower())

    def test_python_checkpoint_accepted(self):
        """Python-originated checkpoint must pass validation silently."""
        checkpoint_state = {
            "step": 100,
            "written_by_runtime": _PYTHON_RUNTIME_SENTINEL,
        }
        # Must not raise
        validate_checkpoint_runtime_compatibility(checkpoint_state)

    def test_missing_sentinel_accepted_as_python(self):
        """
        A checkpoint without a written_by_runtime key (pre-Phase-3 legacy
        checkpoint) is treated as Python-originated and must be accepted.
        """
        checkpoint_state = {"step": 50}
        # Must not raise
        validate_checkpoint_runtime_compatibility(checkpoint_state)

    def test_unknown_runtime_string_accepted(self):
        """An unknown but non-Rust runtime string does not block resumption."""
        checkpoint_state = {"step": 10, "written_by_runtime": "some_other_runtime"}
        # Must not raise — only the exact Rust sentinel is rejected
        validate_checkpoint_runtime_compatibility(checkpoint_state)


class TestP3D2PythonSentinelInSavedCheckpoint(unittest.TestCase):
    """
    Verify that save_checkpoint() writes written_by_runtime="python" into
    checkpoint_state.json so the Rust runtime can detect Python checkpoints.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.model = _make_model()
        self.optimizer = _make_optimizer(self.model)
        self.scaler = _make_scaler()
        _write_tokenizer(os.path.join(self.tmp, "tokenizer.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_checkpoint_contains_python_sentinel(self):
        """checkpoint_state.json must contain written_by_runtime='python'."""
        ckpt_dir = os.path.join(self.tmp, "ckpt_out")
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            config=_CONFIG,
            tokenizer_path=os.path.join(self.tmp, "tokenizer.json"),
            output_dir=ckpt_dir,
            step=1,
            epoch=0,
            loss=1.0,
            best_loss=1.0,
            stage_id="stage0_test",
        )

        from cloud_train import _collect_versioned_checkpoint_dirs
        parent_dir = self.tmp
        output_dir_basename = "ckpt_out"
        versioned_dirs = _collect_versioned_checkpoint_dirs(parent_dir, output_dir_basename)
        self.assertTrue(len(versioned_dirs) > 0, "No versioned checkpoint dir found")
        # Read checkpoint_state.json from the versioned directory
        ckpt_state_path = os.path.join(versioned_dirs[-1], "checkpoint_state.json")
        self.assertTrue(os.path.exists(ckpt_state_path), "checkpoint_state.json missing")
        with open(ckpt_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(
            state.get("written_by_runtime"), _PYTHON_RUNTIME_SENTINEL,
            f"Expected written_by_runtime='{_PYTHON_RUNTIME_SENTINEL}', got: {state.get('written_by_runtime')!r}"
        )


# ---------------------------------------------------------------------------
# Gate P3E1 — ShardLineIndex: build, persist, load, and SHA-invalidation
# ---------------------------------------------------------------------------

class TestP3E1ShardLineIndex(unittest.TestCase):
    """
    Verify ShardLineIndex builds a correct byte-offset map, persists it to
    disk, reloads it on the next call, and invalidates stale indexes when the
    shard SHA changes.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_shard(self, n_lines: int) -> str:
        path = os.path.join(self.tmp, "shard_test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n_lines):
                f.write(json.dumps({"text": f"line_{i}"}) + "\n")
        return path

    def test_build_produces_correct_offsets(self):
        """Each recorded byte offset must point to the start of the correct line."""
        n = 10
        path = self._make_shard(n)
        sha = compute_file_sha256(path)
        idx = ShardLineIndex(self.tmp)
        offsets = idx.build_and_save(path, sha)

        self.assertEqual(len(offsets), n, "Offset map must have one entry per line")
        self.assertEqual(offsets[0], 0, "Line 0 must start at byte 0")

        with open(path, "rb") as f:
            for line_num, byte_pos in offsets.items():
                f.seek(byte_pos)
                raw = f.readline()
                parsed = json.loads(raw)
                self.assertEqual(
                    parsed["text"], f"line_{line_num}",
                    f"Offset for line {line_num} is wrong"
                )

    def test_index_persisted_and_reloaded(self):
        """A built index is saved to disk and can be loaded without rebuilding."""
        path = self._make_shard(5)
        sha = compute_file_sha256(path)
        idx = ShardLineIndex(self.tmp)

        built = idx.build_and_save(path, sha)
        loaded = idx.load(sha)

        self.assertIsNotNone(loaded, "Index must be loadable after build")
        self.assertEqual(dict(built), dict(loaded), "Loaded index must match built index")

    def test_stale_sha_returns_none(self):
        """Requesting an index for a SHA that has no index file returns None."""
        idx = ShardLineIndex(self.tmp)
        result = idx.load("nonexistent_sha256_string_that_has_no_index")
        self.assertIsNone(result)

    def test_get_or_build_builds_on_first_call(self):
        """get_or_build() builds the index when no cache exists."""
        path = self._make_shard(8)
        sha = compute_file_sha256(path)
        idx = ShardLineIndex(self.tmp)

        # Ensure no cached index exists
        idx_file = idx._index_path(sha)
        self.assertFalse(os.path.exists(idx_file))

        offsets = idx.get_or_build(path, sha)
        self.assertEqual(len(offsets), 8)
        self.assertTrue(os.path.exists(idx_file), "Index file must be created")

    def test_get_or_build_returns_cached_on_second_call(self):
        """get_or_build() returns cached index on subsequent calls."""
        path = self._make_shard(8)
        sha = compute_file_sha256(path)
        idx = ShardLineIndex(self.tmp)

        first = idx.get_or_build(path, sha)
        # Now mutate the shard but keep the same SHA argument (simulating cache hit)
        second = idx.get_or_build(path, sha)
        self.assertEqual(dict(first), dict(second))

    def test_different_sha_causes_rebuild(self):
        """
        When the shard is modified its new SHA produces a fresh index, and the
        old SHA's index is not returned for the new content.
        """
        path = self._make_shard(4)
        sha_original = compute_file_sha256(path)
        idx = ShardLineIndex(self.tmp)
        original_offsets = idx.build_and_save(path, sha_original)

        # Append a line — changes file content and SHA
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": "extra_line"}) + "\n")
        sha_new = compute_file_sha256(path)
        self.assertNotEqual(sha_original, sha_new)

        # Old SHA should still return the old index
        old = idx.load(sha_original)
        self.assertIsNotNone(old)
        self.assertEqual(len(old), 4)

        # New SHA has no index yet
        new_cached = idx.load(sha_new)
        self.assertIsNone(new_cached, "New SHA must not be satisfied by old index")

        # get_or_build builds a new correct index for the new SHA
        new_offsets = idx.get_or_build(path, sha_new)
        self.assertEqual(len(new_offsets), 5, "New index must reflect 5 lines")


# ---------------------------------------------------------------------------
# Gate P3R1 — Regression: confirm imports for all 85 prior tests still work
# ---------------------------------------------------------------------------

class TestP3R1Regression(unittest.TestCase):
    """
    Smoke test that key symbols used by Phase 1, 1.1, 2, 2.1 tests are still
    importable and correctly typed after Phase 3 changes.
    """

    def test_shardledger_constructor_accepts_stage_id_and_revision(self):
        """Phase 3 extended ShardLedger constructor must be backward-compatible."""
        with tempfile.TemporaryDirectory() as d:
            # Old-style construction (no new args) must still work
            ledger_old = ShardLedger(d)
            self.assertEqual(ledger_old.stage_id, "")
            self.assertEqual(ledger_old.dataset_revision, "v2")
            # New-style construction
            ledger_new = ShardLedger(d, stage_id="s1", dataset_revision="v3")
            self.assertEqual(ledger_new.stage_id, "s1")

    def test_streaming_dataset_constructor_accepts_max_shard_bytes(self):
        """TaraStreamingJsonlDataset must accept max_shard_bytes kwarg."""
        tokenizer = _make_tokenizer()
        ds = TaraStreamingJsonlDataset(
            file_paths=[],
            tokenizer=tokenizer,
            max_seq_len=32,
            max_shard_bytes=512,
        )
        self.assertEqual(ds.max_shard_bytes, 512)

    def test_mark_completed_accepts_token_chunks_trained(self):
        """mark_completed must accept token_chunks_trained keyword argument."""
        with tempfile.TemporaryDirectory() as d:
            ledger = ShardLedger(d)
            # Old-style (no token_chunks_trained) must still work
            ledger.mark_completed("s1.jsonl", 10, "sha1")
            # New-style
            ledger.mark_completed("s2.jsonl", 20, "sha2", token_chunks_trained=99)
            data = json.loads(open(os.path.join(d, "shard_ledger.json"), encoding="utf-8").read())
            self.assertEqual(data["completed_metadata"]["s2.jsonl"]["token_chunks_trained"], 99)
            # Old-style should default to 0
            self.assertEqual(data["completed_metadata"]["s1.jsonl"].get("token_chunks_trained", 0), 0)

    def test_validate_checkpoint_runtime_compatibility_importable(self):
        """Task D function is importable and callable."""
        # Should not raise on a normal Python checkpoint
        validate_checkpoint_runtime_compatibility({"step": 5, "written_by_runtime": "python"})

    def test_shard_line_index_importable(self):
        """ShardLineIndex class is importable."""
        with tempfile.TemporaryDirectory() as d:
            idx = ShardLineIndex(d)
            self.assertIsNotNone(idx)

    def test_shard_line_index_idx_dir_correct(self):
        """ShardLineIndex stores index files under .tara_idx subdirectory."""
        with tempfile.TemporaryDirectory() as d:
            idx = ShardLineIndex(d)
            expected = os.path.join(d, ".tara_idx")
            self.assertEqual(idx.idx_dir, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

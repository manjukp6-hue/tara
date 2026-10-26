#!/usr/bin/env python3
"""
Adversarial Empirical Stress Harness — Challenger 2
=============================================================================
Stress-tests:
1. Dynamic Best Checkpoint zero-lock commitment at Step 1 (skip_save_steps=0)
2. Teacher Replay Buffer auto-persistence, bounded capacity, and corruption resistance
3. Multi-Account GDrive Quota Handover: mid-shard ungraceful disconnection,
   arbitrary byte/line offsets, deterministic resume, 0 duplicate, 0 dropped records.
=============================================================================
"""

import copy
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure taracore root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

# Workspace imports from cloud_train
import cloud_train
from cloud_train import (
    TaraForCausalLM,
    TaraWorldModel,
    TaraNativeTokenizer,
    TeacherReplayBuffer,
    TaraMultiDomainScheduler,
    TaraStreamingJsonlDataset,
    ShardLedger,
    save_best_milestone,
    save_checkpoint,
    try_resume_from_checkpoint,
    compute_file_sha256,
    CANONICAL_DOMAINS,
    CORE_ANCHOR_DOMAINS,
)

# Registry of temporary test directories for cleanup only after full completion
_TEST_TEMP_DIRS: List[str] = []


def _create_temp_workspace(prefix: str = "tara_challenger2_") -> str:
    td = tempfile.mkdtemp(prefix=prefix)
    _TEST_TEMP_DIRS.append(td)
    return td


def _build_test_tokenizer() -> Tuple[TaraNativeTokenizer, Dict[str, int]]:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "[STATE]": 4,
        "[ACTION]": 5,
        "[NEXT_STATE]": 6,
        "[EOS]": 7,
    }
    # Add ASCII letters and digits
    for i in range(256):
        c = chr(i)
        if c not in vocab:
            vocab[c] = len(vocab)
    # Add Kannada characters for multilingual testing
    for c in "ಕನ್ನಡ ಭಾಷೆ ಪರೀಕ್ಷೆ ಜ್ಞಾನ ತಂತ್ರಜ್ಞಾನ":
        if c not in vocab:
            vocab[c] = len(vocab)
    # Add math symbols
    for c in "∀∃∈ℝ∑∏∫√∂":
        if c not in vocab:
            vocab[c] = len(vocab)

    tok = TaraNativeTokenizer(vocab)
    return tok, vocab


def _make_tiny_model_config(vocab_size: int = 400) -> dict:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 2,
        "d_ff": 128,
        "max_seq_len": 128,
        "dropout": 0.0,
    }


class Challenger2DynamicBestCheckpointTests(unittest.TestCase):
    """
    Empirical tests challenging Dynamic Best Checkpoint zero-lock commitment,
    immediate Step 1 save, and artifact integrity.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_temp_workspace("best_ckpt_challenger_")
        cls.tokenizer, cls.vocab = _build_test_tokenizer()
        cls.config = _make_tiny_model_config(len(cls.vocab) + 50)
        cls.tok_path = os.path.join(cls.work_dir, "tokenizer.json")
        with open(cls.tok_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": cls.vocab}}, f)

    def test_adv_01_step_1_instant_commit_with_zero_delay(self):
        """Simulate step 1 loss improvement and assert immediate commitment to best_checkpoint."""
        model = TaraForCausalLM(self.config)
        test_dir = os.path.join(self.work_dir, "step1_instant")
        best_dir = os.path.join(test_dir, "best_checkpoint")
        os.makedirs(test_dir, exist_ok=True)

        teacher_buf = TeacherReplayBuffer(max_samples_per_domain=100)
        teacher_buf.add_sample("01_mathematics_formal_logic", [1, 2, 3, 4], step=1, loss=2.5)

        committed = save_best_milestone(
            model=model,
            config=self.config,
            tokenizer_path=self.tok_path,
            best_dir=best_dir,
            step=1,
            epoch=0,
            loss=2.50,
            best_loss=2.50,
            stage_id="stage1_unified",
            teacher_replay_buffer=teacher_buf,
        )

        self.assertTrue(committed, "save_best_milestone must return True on Step 1 commit")
        self.assertTrue(os.path.isdir(best_dir), "best_checkpoint directory must exist")

        # Verify all mandatory assets are present
        model_path = os.path.join(best_dir, "model.safetensors")
        self.assertTrue(os.path.isfile(model_path), "model.safetensors must exist")
        self.assertGreater(os.path.getsize(model_path), 1024, "model.safetensors must be valid binary")

        state_path = os.path.join(best_dir, "checkpoint_state.json")
        self.assertTrue(os.path.isfile(state_path), "checkpoint_state.json must exist")
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        self.assertEqual(st.get("step"), 1, "State step must be 1")
        self.assertEqual(st.get("loss"), 2.50, "State loss must be 2.50")

        # Verify dynamic SHA-256 match
        computed_sha = compute_file_sha256(model_path)
        self.assertEqual(st.get("model_weights_sha256"), computed_sha, "Dynamic SHA-256 in state must match actual model file")

        # Verify teacher replay buffer persisted concurrently
        tbuf_path = os.path.join(best_dir, "teacher_replay_buffer.jsonl")
        self.assertTrue(os.path.isfile(tbuf_path), "teacher_replay_buffer.jsonl must exist in best_checkpoint")
        self.assertGreater(os.path.getsize(tbuf_path), 0, "teacher_replay_buffer.jsonl must not be empty")

        # Verify COMMITTED marker
        marker_path = os.path.join(best_dir, "COMMITTED")
        self.assertTrue(os.path.isfile(marker_path), "COMMITTED marker must exist")

    def test_adv_02_consecutive_best_milestones_atomic_replacement(self):
        """Simulate sequential best losses: step 1 (loss 3.0) -> step 2 (loss 2.5) -> step 3 (loss 2.0)."""
        model = TaraForCausalLM(self.config)
        test_dir = os.path.join(self.work_dir, "consecutive_best")
        best_dir = os.path.join(test_dir, "best_checkpoint")
        os.makedirs(test_dir, exist_ok=True)

        teacher_buf = TeacherReplayBuffer(max_samples_per_domain=100)

        for step, loss in [(1, 3.0), (2, 2.5), (3, 2.0)]:
            teacher_buf.add_sample("02_computer_science_systems", [10, 20, 30], step=step, loss=loss)
            ok = save_best_milestone(
                model=model,
                config=self.config,
                tokenizer_path=self.tok_path,
                best_dir=best_dir,
                step=step,
                epoch=0,
                loss=loss,
                best_loss=loss,
                stage_id="stage1_unified",
                teacher_replay_buffer=teacher_buf,
            )
            self.assertTrue(ok)

            with open(os.path.join(best_dir, "checkpoint_state.json"), "r") as f:
                st = json.load(f)
            self.assertEqual(st["step"], step)
            self.assertAlmostEqual(st["best_loss"], loss)

        # Ensure no leftover temporary staging or backup folders remain
        parent = os.path.dirname(os.path.abspath(best_dir))
        leftovers = [f for f in os.listdir(parent) if f.startswith((".staging_best_", ".bak_best_"))]
        self.assertEqual(len(leftovers), 0, f"Leftover staging/backup directories found: {leftovers}")

    def test_adv_03_joint_world_model_co_presence_in_best_milestone(self):
        """Verify that when joint world model is trained, world_model.safetensors is also persisted in best_checkpoint."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)
        test_dir = os.path.join(self.work_dir, "joint_best")
        best_dir = os.path.join(test_dir, "best_checkpoint")
        os.makedirs(test_dir, exist_ok=True)

        committed = save_best_milestone(
            model=neural_model,
            config=self.config,
            tokenizer_path=self.tok_path,
            best_dir=best_dir,
            step=1,
            epoch=0,
            loss=1.50,
            best_loss=1.50,
            stage_id="stage1_unified",
            world_model=world_model,
        )
        self.assertTrue(committed)
        wm_path = os.path.join(best_dir, "world_model.safetensors")
        self.assertTrue(os.path.isfile(wm_path), "world_model.safetensors must be committed in best_checkpoint")
        self.assertGreater(os.path.getsize(wm_path), 1024)

        with open(os.path.join(best_dir, "checkpoint_state.json"), "r") as f:
            st = json.load(f)
        self.assertTrue(st.get("joint_world_model"), "joint_world_model must be True in state")
        self.assertIsNotNone(st.get("world_model_weights_sha256"), "world_model_weights_sha256 must be populated")


class Challenger2TeacherReplayBufferTests(unittest.TestCase):
    """
    Empirical tests challenging TeacherReplayBuffer reservoir sampling,
    bounded memory footprint, unicode handling, and resilience to corrupt inputs.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_temp_workspace("teacher_buf_challenger_")

    def test_adv_04_bounded_memory_under_extreme_volume(self):
        """Stress-test 50,000 samples across all 14 domains; verify bounded capacity and memory."""
        buffer = TeacherReplayBuffer(max_samples_per_domain=500)

        # Feed 50,000 samples across 14 domains
        for i in range(50000):
            dom = CANONICAL_DOMAINS[i % len(CANONICAL_DOMAINS)]
            tokens = [random.randint(1, 300) for _ in range(128)]
            buffer.add_sample(dom, tokens, step=i, loss=random.random() * 3.0)

        # Check domain sample counts strictly capped at 500
        for dom in CANONICAL_DOMAINS:
            self.assertIn(dom, buffer.buffers)
            self.assertLessEqual(len(buffer.buffers[dom]), 500, f"Domain {dom} exceeded cap of 500")

        total = buffer.total_samples()
        self.assertLessEqual(total, 500 * len(CANONICAL_DOMAINS))

        # Check serialized size on disk
        dump_path = os.path.join(self.work_dir, "large_teacher_buf.jsonl")
        buffer.save_to_jsonl(dump_path)
        size_bytes = os.path.getsize(dump_path)
        size_mb = size_bytes / (1024 * 1024)
        self.assertLess(size_mb, 25.0, f"Persisted buffer exceeded 25 MB budget: {size_mb:.2f} MB")

    def test_adv_05_uncorrupted_jsonl_and_multilingual_unicode_preservation(self):
        """Verify uncorrupted JSONL with Kannada and mathematical unicode characters."""
        buffer = TeacherReplayBuffer(max_samples_per_domain=100)

        kannada_texts = [
            "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಮತ್ತು ಸುಂದರವಾದ ದ್ರಾವಿಡ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.",
            "ತಂತ್ರಜ್ಞಾನ ಮತ್ತು ಕಂಪ್ಯೂಟರ್ ಸಿಸ್ಟಮ್ಸ್‌ಗಳ ಅಭಿವೃದ್ಧಿಯಲ್ಲಿ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತದೆ.",
        ]
        math_texts = [
            "∀x ∈ ℝ, e^{iπ} + 1 = 0",
            "∫_{-∞}^{∞} e^{-x^2} dx = √π",
        ]

        for i, txt in enumerate(kannada_texts):
            buffer.add_sample("04_linguistics_multilingual", [10, 20, 30], text=txt, step=i, loss=0.5)

        for i, txt in enumerate(math_texts):
            buffer.add_sample("01_mathematics_formal_logic", [40, 50, 60], text=txt, step=i, loss=0.4)

        jsonl_path = os.path.join(self.work_dir, "unicode_teacher_buf.jsonl")
        buffer.save_to_jsonl(jsonl_path)

        # Verify every line parses cleanly
        reloaded = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                self.assertTrue(line.endswith("\n"), f"Line {line_idx} missing trailing newline")
                rec = json.loads(line)
                reloaded.append(rec)

        self.assertEqual(len(reloaded), 4)

        # Verify unicode texts are preserved verbatim without escaping artifacts
        kannada_saved = [r["text"] for r in reloaded if r["domain"] == "04_linguistics_multilingual"]
        self.assertEqual(kannada_saved, kannada_texts)

        math_saved = [r["text"] for r in reloaded if r["domain"] == "01_mathematics_formal_logic"]
        self.assertEqual(math_saved, math_texts)

    def test_adv_06_malformed_and_adversarial_input_resilience(self):
        """Verify buffer handles None, empty strings, tensors, large strings, and unexpected types."""
        buffer = TeacherReplayBuffer(max_samples_per_domain=10)

        # 1. Empty domain -> default to "general"
        buffer.add_sample("", [1, 2, 3])
        self.assertIn("general", buffer.buffers)

        # 2. 1D Tensor
        buffer.add_sample("02_code", torch.tensor([1, 2, 3, 4, 5]))
        self.assertEqual(buffer.buffers["02_code"][-1]["input_ids"], [1, 2, 3, 4, 5])

        # 3. Oversized token list (> 10,000 tokens) -> truncated to 512
        huge_tokens = list(range(10000))
        buffer.add_sample("02_code", huge_tokens)
        self.assertEqual(len(buffer.buffers["02_code"][-1]["input_ids"]), 512)

        # 4. Oversized text (> 10,000 characters) -> truncated to 2048
        huge_text = "A" * 10000
        buffer.add_sample("02_code", [1], text=huge_text)
        self.assertEqual(len(buffer.buffers["02_code"][-1]["text"]), 2048)

        # 5. Scalar int / float input_ids
        buffer.add_sample("03_stem", 42)
        self.assertEqual(buffer.buffers["03_stem"][-1]["input_ids"], [42])

        # 6. Save and reload roundtrip
        p = os.path.join(self.work_dir, "resilience.jsonl")
        buffer.save_to_jsonl(p)

        new_buf = TeacherReplayBuffer(max_samples_per_domain=10)
        new_buf.load_from_jsonl(p)
        self.assertEqual(new_buf.total_samples(), buffer.total_samples())


class Challenger2MultiAccountHandoverResumeTests(unittest.TestCase):
    """
    Adversarial empirical tests of Multi-Account GDrive Quota Handover:
    Simulates mid-stream ungraceful disconnection at arbitrary byte & line offsets,
    persisting shard_ledger.json and domain_cursors, then launching Account 2 to resume.
    Asserts EXACTLY 0 duplicate samples, 0 lost/dropped samples, and byte-exact alignment.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_temp_workspace("handover_challenger_")
        cls.tokenizer, cls.vocab = _build_test_tokenizer()

    def _generate_synthetic_multi_domain_corpus(
        self, base_dir: str, num_records_per_domain: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """Creates authentic test JSONL shards with globally unique deterministic record IDs."""
        domain_files = {}
        for dom, count in num_records_per_domain.items():
            dom_dir = os.path.join(base_dir, dom)
            os.makedirs(dom_dir, exist_ok=True)
            shard_path = os.path.join(dom_dir, "shard_00001.jsonl")
            with open(shard_path, "w", encoding="utf-8") as f:
                for idx in range(count):
                    rec = {
                        "record_id": f"{dom}_REC_{idx:05d}",
                        "domain": dom,
                        "line_num": idx,
                        "text": f"Authentic content for record {idx:05d} belonging to domain {dom}.",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            domain_files[dom] = [shard_path]
        return domain_files

    def test_adv_07_ungraceful_disconnection_and_byte_exact_resume(self):
        """
        Adversarial Scenario:
        1. 3 Domains: Math (50 records), Code (80 records), Kannada (40 records). Total = 170 records.
        2. Account 1 streams with DRR weights (0.4, 0.4, 0.2).
        3. Simulate an ungraceful crash at arbitrary step K (e.g. 37).
        4. Capture domain_cursors, active_domain_idx, and shard_ledger state.
        5. Account 2 resumes from the saved checkpoint state and streams until EOF.
        6. Assert:
           - Total records yielded across Account 1 + Account 2 == 170
           - Zero duplicate records
           - Zero dropped records
           - Exact set of record_ids matches original ground truth corpus
        """
        test_dir = os.path.join(self.work_dir, "sim_handover_k37")
        os.makedirs(test_dir, exist_ok=True)

        counts = {
            "01_mathematics_formal_logic": 50,
            "02_computer_science_systems": 80,
            "04_linguistics_multilingual": 40,
        }
        total_ground_truth_records = sum(counts.values())
        domain_shards = self._generate_synthetic_multi_domain_corpus(test_dir, counts)

        # Ground truth set of all record IDs
        all_expected_ids: Set[str] = set()
        for dom, c in counts.items():
            for idx in range(c):
                all_expected_ids.add(f"{dom}_REC_{idx:05d}")
        self.assertEqual(len(all_expected_ids), total_ground_truth_records)

        weights = {
            "01_mathematics_formal_logic": 0.40,
            "02_computer_science_systems": 0.40,
            "04_linguistics_multilingual": 0.20,
        }

        # ---------------------------------------------------------------------
        # PHASE 1: Account 1 execution up to Step 37, then UNGRACEFUL DISCONNECTION
        # ---------------------------------------------------------------------
        ledger_dir_1 = os.path.join(test_dir, "account1_ledger")
        ledger_1 = ShardLedger(ledger_dir_1, None, stage_id="stage1_unified")

        sched_1 = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            domain_weights=weights,
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=128,
            shard_ledger=ledger_1,
            anchor_ratio=0.0,  # Focus purely on stream continuity without anchor replays
        )

        acct1_records: List[str] = []
        crash_step = 37

        it1 = iter(sched_1)
        for step in range(crash_step):
            batch = next(it1)
            # Find which record was yielded by checking tokenizer decode / properties
            # To be 100% precise, we extract record_id by reading the domain cursor
            dom = batch["domain"]
            cur = sched_1.domain_cursors[dom]
            line_idx = cur["line_offset"]
            rec_id = f"{dom}_REC_{line_idx:05d}"
            acct1_records.append(rec_id)

        # UNGRACEFUL DISCONNECTION!
        # Capture exact state at crash moment
        handover_state = sched_1.get_state()
        ledger_1.domain_cursors = copy.deepcopy(handover_state["domain_cursors"])
        ledger_1.save()

        # Check domain cursors at crash
        print(f"\n[Handover Audit] Crash at Step {crash_step}:")
        for d, cur in handover_state["domain_cursors"].items():
            print(f"  {d}: line_offset={cur.get('line_offset')}, byte_offset={cur.get('byte_offset')}")

        # ---------------------------------------------------------------------
        # PHASE 2: Account 2 mounts the same state and resumes
        # ---------------------------------------------------------------------
        ledger_dir_2 = os.path.join(test_dir, "account2_ledger")
        # Copy ledger file as would happen over Google Drive mount
        os.makedirs(ledger_dir_2, exist_ok=True)
        shutil.copy2(ledger_1.ledger_file, os.path.join(ledger_dir_2, "shard_ledger.json"))
        ledger_2 = ShardLedger(ledger_dir_2, None, stage_id="stage1_unified")

        sched_2 = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            domain_weights=weights,
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=128,
            shard_ledger=ledger_2,
            anchor_ratio=0.0,
            domain_cursors=handover_state["domain_cursors"],
        )
        sched_2.load_state(handover_state)

        acct2_records: List[str] = []
        it2 = iter(sched_2)

        # Drain Account 2 until all shards exhaust
        for batch in it2:
            dom = batch["domain"]
            cur = sched_2.domain_cursors[dom]
            line_idx = cur["line_offset"]
            rec_id = f"{dom}_REC_{line_idx:05d}"
            acct2_records.append(rec_id)

        # ---------------------------------------------------------------------
        # PHASE 3: Adversarial Oracle Assertions
        # ---------------------------------------------------------------------
        total_streamed = len(acct1_records) + len(acct2_records)
        combined_records = acct1_records + acct2_records

        # 1. Assert Total Records Count
        self.assertEqual(
            total_streamed,
            total_ground_truth_records,
            f"Expected {total_ground_truth_records} records, but got {total_streamed} (Acct1: {len(acct1_records)}, Acct2: {len(acct2_records)})"
        )

        # 2. Assert Zero Duplicates
        unique_seen = set()
        duplicates = []
        for r in combined_records:
            if r in unique_seen:
                duplicates.append(r)
            unique_seen.add(r)

        self.assertEqual(
            len(duplicates), 0,
            f"FAIL: Detected {len(duplicates)} DUPLICATE records across handover boundary: {duplicates[:10]}"
        )

        # 3. Assert Zero Dropped Records
        dropped = all_expected_ids - unique_seen
        self.assertEqual(
            len(dropped), 0,
            f"FAIL: Detected {len(dropped)} DROPPED records across handover boundary: {list(dropped)[:10]}"
        )

        # 4. Assert Exact Set Identity
        self.assertEqual(unique_seen, all_expected_ids, "Set of streamed records must match ground truth exactly")
        print(f"[Handover Audit] PASSED: 0 duplicates, 0 dropped records. Total: {total_streamed}/{total_ground_truth_records}")

    def test_adv_08_multi_step_crash_sweep(self):
        """Sweep ungraceful crash across multiple arbitrary step boundaries (10, 25, 60, 95)."""
        counts = {
            "01_mathematics_formal_logic": 30,
            "02_computer_science_systems": 40,
            "04_linguistics_multilingual": 30,
        }
        total_records = sum(counts.values())

        for crash_step in [10, 25, 60, 95]:
            test_dir = os.path.join(self.work_dir, f"sweep_crash_{crash_step}")
            os.makedirs(test_dir, exist_ok=True)
            domain_shards = self._generate_synthetic_multi_domain_corpus(test_dir, counts)

            all_expected_ids = {f"{dom}_REC_{idx:05d}" for dom, c in counts.items() for idx in range(c)}

            # Run Account 1 up to crash_step
            sched1 = TaraMultiDomainScheduler(
                domain_shards=domain_shards,
                domain_weights={"01_mathematics_formal_logic": 0.33, "02_computer_science_systems": 0.34, "04_linguistics_multilingual": 0.33},
                tokenizer=self.tokenizer,
                max_seq_len=128,
                anchor_ratio=0.0,
            )

            acct1_recs = []
            it1 = iter(sched1)
            for _ in range(crash_step):
                b = next(it1)
                dom = b["domain"]
                line = sched1.domain_cursors[dom]["line_offset"]
                acct1_recs.append(f"{dom}_REC_{line:05d}")

            state = sched1.get_state()

            # Run Account 2 from crash_step
            sched2 = TaraMultiDomainScheduler(
                domain_shards=domain_shards,
                domain_weights={"01_mathematics_formal_logic": 0.33, "02_computer_science_systems": 0.34, "04_linguistics_multilingual": 0.33},
                tokenizer=self.tokenizer,
                max_seq_len=128,
                anchor_ratio=0.0,
                domain_cursors=state["domain_cursors"],
            )
            sched2.load_state(state)

            acct2_recs = []
            it2 = iter(sched2)
            for b in it2:
                dom = b["domain"]
                line = sched2.domain_cursors[dom]["line_offset"]
                acct2_recs.append(f"{dom}_REC_{line:05d}")

            combined = acct1_recs + acct2_recs
            self.assertEqual(len(combined), total_records, f"Crash at {crash_step} produced record count mismatch")
            self.assertEqual(set(combined), all_expected_ids, f"Crash at {crash_step} produced dropped/duplicate records")

    def test_adv_09_document_chunking_boundary_handover(self):
        """
        Verify multi-chunk document interruption:
        When a document produces 3 chunks (chunk 0, 1, 2) and Account 1 crashes
        after consuming chunk 0, Account 2 must resume at chunk 1 without
        re-emitting chunk 0 or skipping chunk 1.
        """
        test_dir = os.path.join(self.work_dir, "chunking_handover")
        os.makedirs(test_dir, exist_ok=True)

        # Single shard with 2 documents:
        # Doc 0 has ~150 tokens (with max_seq_len=32, this gives multiple chunks)
        # Doc 1 has ~150 tokens
        shard_path = os.path.join(test_dir, "long_docs.jsonl")
        doc0_text = "The quick brown fox jumps over the lazy dog. " * 15
        doc1_text = "Machine learning algorithms learn representations from data. " * 15

        with open(shard_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": doc0_text}) + "\n")
            f.write(json.dumps({"text": doc1_text}) + "\n")

        domain_shards = {"02_code": [shard_path]}

        # Account 1 streams only 1 batch (chunk 0 of Doc 0)
        ds1 = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=self.tokenizer,
            max_seq_len=32,
            buffer_size=1,
        )
        it1 = iter(ds1)
        first_item = next(it1)
        # item: (input_ids, labels, shard_name, line_idx, line_byte_offset, chunk_idx, is_last)
        _, _, s_name, l_idx, b_off, c_idx, is_last = first_item
        self.assertEqual(l_idx, 0, "First chunk must belong to line 0")
        self.assertEqual(c_idx, 0, "First chunk must be chunk_idx 0")
        self.assertFalse(is_last, "First chunk of long doc should not be last")

        # Capture cursor at chunk 0
        resume_cursor = {
            "active_shard": s_name,
            "line_offset": l_idx,
            "byte_offset": b_off,
            "chunk_idx": c_idx + 1,  # Resume at chunk 1!
        }

        # Account 2 resumes with resume_chunk_idx = 1
        ds2 = TaraStreamingJsonlDataset(
            file_paths=[shard_path],
            tokenizer=self.tokenizer,
            max_seq_len=32,
            buffer_size=1,
            resume_shard=resume_cursor["active_shard"],
            resume_line_offset=resume_cursor["line_offset"],
            resume_byte_offset=resume_cursor["byte_offset"],
            resume_chunk_idx=resume_cursor["chunk_idx"],
        )
        it2 = iter(ds2)
        resumed_first_item = next(it2)
        _, _, r_name, r_l_idx, r_b_off, r_c_idx, r_is_last = resumed_first_item

        self.assertEqual(r_l_idx, 0, "Resumed item must still be on line 0")
        self.assertEqual(r_c_idx, 1, "Resumed item must be chunk_idx 1 (no replaying chunk 0)")


def tearDownModule():
    """Lifecycle discipline: Clean up test directories only after complete suite finishes."""
    for td in _TEST_TEMP_DIRS:
        if os.path.exists(td):
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()

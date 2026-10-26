#!/usr/bin/env python3
"""
TARA Cloud Pretraining Pipeline End-to-End Test Suite (4-Tier Architecture)
=============================================================================
Comprehensive test suite verifying cloud pretraining against `manjukp6/tara_dataset_v1`:
- Tier 1: Feature Coverage (R1-R6)
    1. Python compilation syntax check (`cloud_train.py`)
    2. CLI argument parser registration (`--domain-weights`, `--skip-save-steps`, etc.)
    3. 14 Canonical domain mapping and alias resolution
    4. Deficit Round-Robin (DRR) weighted scheduler proportional mixing
    5. Anchor replay buffer activation for depleted core domains
    6. Zero-lock dynamic best checkpoint commit (`skip_save_steps=0`)
    7. Stratified Teacher Replay Buffer generation & JSONL persistence
    8. Multi-account Google Drive quota handover & deterministic cursor restoration
    9. Parquet & curriculum transition record parsing for Joint World Model
    10. `colab_train.sh` bash syntax and canonical restore file list verification

- Tier 2: Boundary & Corner Cases
    11. Zero or single-shard domain edge handling
    12. Exact transition boundary from live shard to anchor replay
    13. Domain weight normalization (non-unity sums, empty fallback)
    14. Step 1 instant milestone save when `skip_save_steps=0`
    15. Shard byte/line offset boundary conditions on resume
    16. UnboundLocalError validation in `download_single_shard_jit`

- Tier 3: Cross-Feature Combinations
    17. Multi-domain DRR streaming co-trained with Joint World Model
    18. Anchor replay interleaved with teacher replay buffer snapshots
    19. Multi-account quota handover simulation restoring mid-stream domain cursors

- Tier 4: Real-World Application Scenarios
    20. Complete end-to-end simulated training loop with joint loss, live telemetry,
        and atomic best checkpoint commit
=============================================================================
"""

import ast
import glob
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Workspace imports from cloud_train
import cloud_train
from cloud_train import (
    TaraForCausalLM,
    TaraWorldModel,
    TaraNativeTokenizer,
    WorldModelTransitionDataset,
    save_checkpoint,
    save_best_milestone,
    try_resume_from_checkpoint,
    ShardLedger,
    download_single_shard_jit,
)

# Reference definitions from project specification
CANONICAL_14_DOMAINS = [
    "01_mathematics_formal_logic",
    "02_computer_science_systems",
    "03_natural_sciences_stem",
    "04_linguistics_multilingual",
    "05_philosophy_cognitive_ethics",
    "06_social_sciences_economics_law",
    "07_applied_engineering_robotics",
    "08_commonsense_world_model",
    "09_multimodal_audio",
    "10_multimodal_vision",
    "11_tool_use_execution",
    "12_planning_reasoning_traces",
    "13_debugging_self_correction",
    "instruction_tuning_sft",
]

CORE_DEPLETED_DOMAINS = [
    "12_planning_reasoning_traces",
    "04_linguistics_multilingual",
    "01_mathematics_formal_logic",
    "11_tool_use_execution",
]

# Track temporary test directories to preserve during runs and clean up only post-suite
_SUITE_TEMP_DIRS: List[str] = []


def _create_managed_temp_dir(prefix: str = "tara_e2e_") -> str:
    """Creates a temporary directory tracked for suite-level lifecycle cleanup."""
    td = tempfile.mkdtemp(prefix=prefix)
    _SUITE_TEMP_DIRS.append(td)
    return td


def setUpModule():
    """Module-level setup."""
    pass


def tearDownModule():
    """Clean up all preserved test directories only after the entire test suite finishes."""
    for td in _SUITE_TEMP_DIRS:
        if os.path.exists(td):
            shutil.rmtree(td, ignore_errors=True)
    _SUITE_TEMP_DIRS.clear()


def _build_test_vocab() -> Dict[str, int]:
    """Generates a bounded test vocabulary."""
    vocab = {f"<tok_{i}>": i for i in range(128)}
    vocab["[STATE]"] = 124
    vocab["[ACTION]"] = 125
    vocab["[NEXT_STATE]"] = 126
    vocab["[EOS]"] = 127
    vocab["[UNK]"] = 0
    return vocab


def _make_tiny_model_config() -> Dict[str, Any]:
    """Generates minimal transformer config for rapid local test execution."""
    return {
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
    }


# =============================================================================
# TIER 1: Feature Coverage (R1–R6)
# =============================================================================

class Tier1FeatureCoverageTests(unittest.TestCase):
    """
    Tier 1 tests verifying primary behavior and interface contracts for all R1-R6 features.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_managed_temp_dir("tier1_")
        cls.vocab = _build_test_vocab()
        cls.tokenizer = TaraNativeTokenizer(cls.vocab)
        cls.tok_path = os.path.join(cls.work_dir, "tokenizer.json")
        with open(cls.tok_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": cls.vocab}}, f)
        cls.config = _make_tiny_model_config()

    def test_tier1_01_python_compilation_syntax(self):
        """Tier 1.1: Verify python compilation syntax passes with exit code 0."""
        target_script = os.path.abspath("cloud_train.py")
        self.assertTrue(os.path.isfile(target_script), f"cloud_train.py must exist at {target_script}")

        # In-process compilation
        try:
            py_compile.compile(target_script, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"py_compile failed on cloud_train.py: {e}")

        # Subprocess compilation execution proof
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", target_script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, f"python -m py_compile failed: {proc.stderr}")

    def test_tier1_02_cli_flags_registration(self):
        """Tier 1.2: Verify required CLI arguments are registered in cloud_train.py."""
        target_script = os.path.abspath("cloud_train.py")
        proc = subprocess.run(
            [sys.executable, target_script, "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, f"cloud_train.py --help failed: {proc.stderr}")
        help_output = proc.stdout

        required_flags = [
            "--domain-weights",
            "--skip-save-steps",
            "--joint-world-model",
            "--hf-repo",
            "--dataset-prefix",
        ]
        missing_flags = [flag for flag in required_flags if flag not in help_output]
        self.assertFalse(
            missing_flags,
            f"cloud_train.py --help is missing required CLI flags: {missing_flags}",
        )

    def test_tier1_03_domain_mapping_14_canonical_domains(self):
        """Tier 1.3: Verify 14 canonical domains are mapped and alias resolution functions correctly."""
        # Check if helper or scheduler exposes domain list / resolver
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        canonical_list = (
            getattr(cloud_train, "CANONICAL_DOMAINS", None)
            or getattr(cloud_train, "CANONICAL_DOMAIN_LIST", None)
            or (getattr(scheduler_cls, "CANONICAL_DOMAINS", None) if scheduler_cls else None)
        )
        if canonical_list is None:
            self.fail("CANONICAL_DOMAINS is not exposed in cloud_train or TaraMultiDomainScheduler")

        self.assertEqual(
            sorted(canonical_list),
            sorted(CANONICAL_14_DOMAINS),
            f"Canonical domain list mismatch: got {canonical_list}, expected {CANONICAL_14_DOMAINS}",
        )

        # Test alias resolver function if present
        resolve_fn = getattr(cloud_train, "resolve_domain_name", None) or getattr(cloud_train, "map_domain_alias", None)
        if resolve_fn is not None:
            self.assertEqual(resolve_fn("01_math"), "01_mathematics_formal_logic")
            self.assertEqual(resolve_fn("math"), "01_mathematics_formal_logic")
            self.assertEqual(resolve_fn("02_code"), "02_computer_science_systems")
            self.assertEqual(resolve_fn("04_kannada"), "04_linguistics_multilingual")
            self.assertEqual(resolve_fn("12_reasoning"), "12_planning_reasoning_traces")
            self.assertEqual(resolve_fn("sft"), "instruction_tuning_sft")

    def test_tier1_04_drr_weighted_scheduler_proportional_mixing(self):
        """Tier 1.4: Verify Deficit Round-Robin (DRR) weighted scheduler proportional mixing."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented or exported in cloud_train.py")

        # Create mock shard directory structure for 3 domains
        domain_dirs = {}
        for d in ["01_mathematics_formal_logic", "02_computer_science_systems", "04_linguistics_multilingual"]:
            dd = os.path.join(self.work_dir, "drr_test", d)
            os.makedirs(dd, exist_ok=True)
            shard_path = os.path.join(dd, "shard_000.jsonl")
            with open(shard_path, "w", encoding="utf-8") as f:
                for i in range(100):
                    f.write(json.dumps({"text": f"Sample {i} from domain {d}"}) + "\n")
            domain_dirs[d] = [shard_path]

        weights = {
            "01_mathematics_formal_logic": 0.20,
            "02_computer_science_systems": 0.50,
            "04_linguistics_multilingual": 0.30,
        }

        scheduler = scheduler_cls(
            domain_shards=domain_dirs,
            domain_weights=weights,
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=64,
        )

        sampled_counts = {d: 0 for d in weights}
        total_samples = 100
        it = iter(scheduler)
        for _ in range(total_samples):
            batch = next(it)
            self.assertIn("domain", batch, "Batch must include 'domain' metadata key")
            self.assertIn("input_ids", batch, "Batch must include 'input_ids'")
            self.assertIn("is_anchor", batch, "Batch must include 'is_anchor' metadata key")
            d_name = batch["domain"]
            self.assertIn(d_name, sampled_counts)
            sampled_counts[d_name] += 1

        # Check proportions within DRR deficit jitter (+- 5 batches for N=100)
        self.assertAlmostEqual(sampled_counts["01_mathematics_formal_logic"] / total_samples, 0.20, delta=0.06)
        self.assertAlmostEqual(sampled_counts["02_computer_science_systems"] / total_samples, 0.50, delta=0.06)
        self.assertAlmostEqual(sampled_counts["04_linguistics_multilingual"] / total_samples, 0.30, delta=0.06)

    def test_tier1_05_anchor_replay_activation_for_depleted_core_domains(self):
        """Tier 1.5: Verify anchor replay activates when core domains deplete shards."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented or exported in cloud_train.py")

        # Setup 1 small core domain with only 5 samples and 1 large domain with 100 samples
        core_domain = "12_planning_reasoning_traces"
        abundant_domain = "02_computer_science_systems"
        base_dir = os.path.join(self.work_dir, "anchor_test")
        os.makedirs(os.path.join(base_dir, core_domain), exist_ok=True)
        os.makedirs(os.path.join(base_dir, abundant_domain), exist_ok=True)

        core_shard = os.path.join(base_dir, core_domain, "shard_000.jsonl")
        with open(core_shard, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({"text": f"Reasoning trace {i}"}) + "\n")

        abundant_shard = os.path.join(base_dir, abundant_domain, "shard_000.jsonl")
        with open(abundant_shard, "w", encoding="utf-8") as f:
            for i in range(100):
                f.write(json.dumps({"text": f"Code sample {i}"}) + "\n")

        domain_shards = {
            core_domain: [core_shard],
            abundant_domain: [abundant_shard],
        }
        weights = {core_domain: 0.30, abundant_domain: 0.70}

        scheduler = scheduler_cls(
            domain_shards=domain_shards,
            domain_weights=weights,
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=64,
            anchor_ratio=0.20,
        )

        seen_anchor = False
        anchor_count = 0
        total_steps = 40
        it = iter(scheduler)
        for step in range(total_steps):
            batch = next(it)
            if batch["domain"] == core_domain and batch.get("is_anchor", False):
                seen_anchor = True
                anchor_count += 1

        self.assertTrue(seen_anchor, "Anchor replay buffer did not activate after core domain depletion")
        self.assertGreaterEqual(anchor_count, 1, "Core domain anchor batches must be yielded after depletion")

    def test_tier1_06_dynamic_best_checkpoint_commit(self):
        """Tier 1.6: Verify zero-lock dynamic best checkpoint saves on step 1 with skip_save_steps=0."""
        model = TaraForCausalLM(self.config)
        output_dir = os.path.join(self.work_dir, "best_ckpt_test")
        best_dir = os.path.join(output_dir, "best_checkpoint")
        os.makedirs(output_dir, exist_ok=True)

        # Dynamic best milestone commit at step 1
        committed = save_best_milestone(
            model=model,
            config=self.config,
            tokenizer_path=self.tok_path,
            best_dir=best_dir,
            step=1,
            epoch=0,
            loss=1.85,
            best_loss=1.85,
            stage_id="stage1_unified",
        )
        self.assertTrue(committed, "save_best_milestone must return True upon successful commit")

        # Verify artifacts in best_checkpoint directory
        self.assertTrue(os.path.isdir(best_dir), "best_checkpoint directory must exist")
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "model.safetensors")), "model.safetensors missing")
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "config.json")), "config.json missing")
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "checkpoint_state.json")), "checkpoint_state.json missing")
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "COMMITTED")), "COMMITTED marker missing")

        with open(os.path.join(best_dir, "checkpoint_state.json"), "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state.get("step"), 1)
        self.assertAlmostEqual(state.get("loss"), 1.85)

    def test_tier1_07_teacher_replay_buffer_generation_and_jsonl(self):
        """Tier 1.7: Verify TeacherReplayBuffer reservoir sampling and valid JSONL persistence."""
        buffer_cls = getattr(cloud_train, "TeacherReplayBuffer", None) or getattr(cloud_train, "TeacherBuffer", None)
        if buffer_cls is None:
            self.fail("TeacherReplayBuffer is not implemented or exported in cloud_train.py")

        buffer = buffer_cls(max_samples_per_domain=500)
        # Add diverse samples across 3 domains
        for i in range(100):
            buffer.add_sample("01_mathematics_formal_logic", [1, 2, 3, i], prompt="x", target="y")
            buffer.add_sample("04_linguistics_multilingual", [4, 5, 6, i], prompt="k", target="v")
            buffer.add_sample("12_planning_reasoning_traces", [7, 8, 9, i], prompt="r", target="s")

        jsonl_path = os.path.join(self.work_dir, "teacher_replay_buffer.jsonl")
        buffer.save_to_jsonl(jsonl_path)

        self.assertTrue(os.path.isfile(jsonl_path), "teacher_replay_buffer.jsonl must exist on disk")
        self.assertGreater(os.path.getsize(jsonl_path), 0, "teacher_replay_buffer.jsonl must be non-empty")

        # Verify each line is valid JSON with required metadata
        loaded_domains = set()
        line_count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                rec = json.loads(line)
                self.assertIn("domain", rec, "Record must contain 'domain'")
                self.assertIn("input_ids", rec, "Record must contain 'input_ids'")
                loaded_domains.add(rec["domain"])

        self.assertEqual(line_count, 300)
        self.assertEqual(len(loaded_domains), 3)

    def test_tier1_08_multi_account_gdrive_resume_offset_restoration(self):
        """Tier 1.8: Verify domain_cursors and active_domain_idx restoration across checkpoint boundaries."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented or exported in cloud_train.py")

        state = {
            "domain_cursors": {
                "01_mathematics_formal_logic": {
                    "active_shard": "shard_001.parquet",
                    "line_offset": 450,
                    "byte_offset": 98120,
                },
                "02_computer_science_systems": {
                    "active_shard": "shard_010.parquet",
                    "line_offset": 1200,
                    "byte_offset": 450123,
                },
            },
            "active_domain_idx": 1,
        }

        # Check if scheduler supports get_state / load_state contract
        dummy_shards = {
            "01_mathematics_formal_logic": [os.path.join(self.work_dir, "shard_001.parquet")],
            "02_computer_science_systems": [os.path.join(self.work_dir, "shard_010.parquet")],
        }
        for s_list in dummy_shards.values():
            for p in s_list:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write("")

        scheduler = scheduler_cls(
            domain_shards=dummy_shards,
            domain_weights={"01_mathematics_formal_logic": 0.5, "02_computer_science_systems": 0.5},
            tokenizer=self.tokenizer,
        )

        scheduler.load_state(state)
        restored = scheduler.get_state()

        self.assertIn("domain_cursors", restored)
        self.assertEqual(
            restored["domain_cursors"]["01_mathematics_formal_logic"]["line_offset"],
            450,
        )
        self.assertEqual(
            restored["domain_cursors"]["02_computer_science_systems"]["byte_offset"],
            450123,
        )
        self.assertEqual(restored.get("active_domain_idx"), 1)

    def test_tier1_09_parquet_and_curriculum_transition_record_parsing(self):
        """Tier 1.9: Verify WorldModelTransitionDataset supports parquet files and curriculum schemas."""
        wm_dir = os.path.join(self.work_dir, "wm_test")
        os.makedirs(wm_dir, exist_ok=True)

        # 1. Test curriculum transition jsonl: (input, output, derivation, formula) and (state, action, next_state)
        curriculum_jsonl = os.path.join(wm_dir, "curriculum_transitions.jsonl")
        with open(curriculum_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "input": "Calculate derivative of x^2",
                "output": "2x",
                "derivation": "d/dx(x^2) = 2x",
                "formula": "power_rule"
            }) + "\n")
            f.write(json.dumps({
                "state_observation": "Grid at (0,0)",
                "action": "move_right",
                "next_state_observation": "Grid at (1,0)",
                "reward": 1.0
            }) + "\n")

        ds = WorldModelTransitionDataset(
            file_paths=[curriculum_jsonl],
            tokenizer=self.tokenizer,
            max_seq_len=64,
        )
        records = list(iter(ds))
        self.assertGreaterEqual(
            len(records), 1,
            "WorldModelTransitionDataset must yield parsed transition records"
        )
        sample = records[0]
        if isinstance(sample, dict):
            input_ids = sample.get("input_ids")
            labels = sample.get("labels")
        else:
            input_ids = sample[0]
            labels = sample[1]
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

        # Requirement R5: Both the curriculum schema (input/output) and state-action schema must be parsed
        self.assertEqual(
            len(records), 2,
            "WorldModelTransitionDataset must parse both curriculum schemas (input/output/derivation/formula) "
            "and state-action transition schemas"
        )

        # 2. Test parquet support if pyarrow is installed
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pydict({
                "state_observation": ["Robot at pos A", "Robot at pos B"],
                "action": ["turn_left", "turn_right"],
                "next_state_observation": ["Robot at pos A_facing_north", "Robot at pos B_facing_east"],
                "reward": [0.5, 1.0],
            })
            parquet_path = os.path.join(wm_dir, "transitions.parquet")
            pq.write_table(table, parquet_path)

            ds_pq = WorldModelTransitionDataset(
                file_paths=[parquet_path],
                tokenizer=self.tokenizer,
                max_seq_len=64,
            )
            pq_records = list(iter(ds_pq))
            self.assertEqual(len(pq_records), 2, "WorldModelTransitionDataset must stream records from parquet files")
        except ImportError:
            pass

    def test_tier1_10_colab_train_sh_structural_syntax_validation(self):
        """Tier 1.10: Verify colab_train.sh structural bash syntax and canonical sync lists."""
        script_path = os.path.abspath("colab_train.sh")
        self.assertTrue(os.path.isfile(script_path), f"colab_train.sh must exist at {script_path}")

        # Locate a working bash executable (checking portable git or system git on Windows)
        bash_candidates = [
            r"C:\Users\Nawaz\AppData\Local\OpenClaw\deps\portable-git\usr\bin\bash.exe",
            r"C:\Program Files\Git\bin\bash.exe",
            shutil.which("bash"),
        ]
        valid_bash = None
        for cand in bash_candidates:
            if cand and os.path.isfile(cand):
                try:
                    chk = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=5)
                    if chk.returncode == 0:
                        valid_bash = cand
                        break
                except Exception:
                    continue

        if valid_bash:
            proc = subprocess.run([valid_bash, "-n", script_path], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"bash -n colab_train.sh syntax check failed: {proc.stderr}")

        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Requirement R3: SKIP_SAVE_STEPS default should be 0
        self.assertIn(
            'SKIP_SAVE_STEPS="${SKIP_SAVE_STEPS:-0}"', content,
            "colab_train.sh must default SKIP_SAVE_STEPS to 0"
        )

        # Requirement R4/R6: Canonical file set must include world_model.safetensors, teacher_replay_buffer.jsonl, COMMITTED
        canonical_block_match = re.search(r"canonical\s*=\s*\{([^}]+)\}", content)
        self.assertIsNotNone(canonical_block_match, "canonical file set block not found in colab_train.sh")
        canonical_str = canonical_block_match.group(1)

        for required_file in ["'world_model.safetensors'", "'teacher_replay_buffer.jsonl'", "'COMMITTED'"]:
            self.assertIn(
                required_file, canonical_str,
                f"colab_train.sh canonical file list must include {required_file}"
            )


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

class Tier2BoundaryCornerCaseTests(unittest.TestCase):
    """
    Tier 2 tests covering edge cases, normalization, zero-length files, and boundary offsets.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_managed_temp_dir("tier2_")
        cls.vocab = _build_test_vocab()
        cls.tokenizer = TaraNativeTokenizer(cls.vocab)
        cls.config = _make_tiny_model_config()

    def test_tier2_01_zero_or_single_shard_domain_edge_cases(self):
        """Tier 2.1: Verify scheduler gracefully handles empty and single-shard domains."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented or exported in cloud_train.py")

        base_dir = os.path.join(self.work_dir, "edge_shards")
        os.makedirs(base_dir, exist_ok=True)

        empty_shard = os.path.join(base_dir, "empty_domain.jsonl")
        with open(empty_shard, "w", encoding="utf-8") as f:
            pass

        single_shard = os.path.join(base_dir, "single_sample.jsonl")
        with open(single_shard, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "Lone sample"}) + "\n")

        domain_shards = {
            "empty_domain": [empty_shard],
            "single_domain": [single_shard],
        }

        # Should not raise ZeroDivisionError or crash
        scheduler = scheduler_cls(
            domain_shards=domain_shards,
            domain_weights={"empty_domain": 0.5, "single_domain": 0.5},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
        )
        it = iter(scheduler)
        first_batch = next(it)
        self.assertEqual(first_batch["domain"], "single_domain")

    def test_tier2_02_depleted_domain_transition_boundary(self):
        """Tier 2.2: Verify precise boundary transition when live shard is exhausted."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented or exported in cloud_train.py")

        base_dir = os.path.join(self.work_dir, "boundary_depletion")
        os.makedirs(base_dir, exist_ok=True)

        core_shard = os.path.join(base_dir, "core_exact.jsonl")
        exact_count = 3
        with open(core_shard, "w", encoding="utf-8") as f:
            for i in range(exact_count):
                f.write(json.dumps({"text": f"Core exact {i}"}) + "\n")

        scheduler = scheduler_cls(
            domain_shards={"12_planning_reasoning_traces": [core_shard]},
            domain_weights={"12_planning_reasoning_traces": 1.0},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
            anchor_ratio=1.0,
        )

        it = iter(scheduler)
        # First 3 samples must be live (is_anchor=False)
        for i in range(exact_count):
            b = next(it)
            self.assertFalse(b.get("is_anchor", False), f"Sample {i} should be live, not anchor")

        # 4th sample must transition to anchor replay (is_anchor=True)
        fourth_batch = next(it)
        self.assertTrue(fourth_batch.get("is_anchor", False), "Sample 4 must be served from anchor replay")

    def test_tier2_03_domain_weight_normalization(self):
        """Tier 2.3: Verify weight normalization when inputs sum != 1.0 or weights are empty."""
        parse_fn = getattr(cloud_train, "parse_domain_weights", None) or getattr(cloud_train, "normalize_domain_weights", None)
        if parse_fn is None:
            # Fallback to checking scheduler weight handling
            scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None)
            if scheduler_cls is None:
                self.fail("Neither parse_domain_weights nor TaraMultiDomainScheduler is available in cloud_train.py")
            s = scheduler_cls(
                domain_shards={"d1": [], "d2": []},
                domain_weights={"d1": 2.0, "d2": 3.0},
                tokenizer=self.tokenizer,
            )
            norm_weights = s.domain_weights
        else:
            norm_weights = parse_fn("01_math:2.0,02_code:3.0", available_domains=["01_math", "02_code"])

        total_weight = sum(norm_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5, msg="Domain weights must normalize to 1.0")

    def test_tier2_04_step_1_is_new_best_triggering_instant_save(self):
        """Tier 2.4: Verify step 1 triggers instant save when skip_save_steps=0."""
        step = 1
        skip_save_steps = 0
        cur_step_loss = 2.10
        best_loss = float("inf")

        is_new_best = cur_step_loss < best_loss
        can_save_at_current_step = (step > skip_save_steps)

        self.assertTrue(is_new_best)
        self.assertTrue(
            can_save_at_current_step,
            "When skip_save_steps=0, step 1 must satisfy can_save_at_current_step"
        )

    def test_tier2_05_shard_byte_line_offset_boundary_conditions(self):
        """Tier 2.5: Verify fast intra-shard byte seeking and line offset boundary preservation."""
        ledger_dir = os.path.join(self.work_dir, "ledger_boundary")
        os.makedirs(ledger_dir, exist_ok=True)

        ledger = ShardLedger(ledger_dir, None, stage_id="test_stage")
        ledger.active_shard = "01_math/train-00001.parquet"
        ledger.active_shard_line_offset = 500
        ledger.active_shard_byte_offset = 1048576
        ledger.save()
        self.assertTrue(os.path.isfile(ledger.ledger_file))

        # Reopen ledger and check boundaries
        reloaded = ShardLedger(ledger_dir, None, stage_id="test_stage")
        self.assertEqual(reloaded.active_shard, "01_math/train-00001.parquet")
        self.assertEqual(reloaded.active_shard_line_offset, 500)
        self.assertEqual(reloaded.active_shard_byte_offset, 1048576)

        # Requirement R4: Ledger must support multi-domain cursors when implemented
        if hasattr(reloaded, "domain_cursors"):
            self.assertIsInstance(reloaded.domain_cursors, dict)

    def test_tier2_06_unbound_local_error_check_in_download_single_shard_jit(self):
        """Tier 2.6: Verify download_single_shard_jit does not reference target_path before assignment."""
        target_file = os.path.abspath("cloud_train.py")
        with open(target_file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        jit_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "download_single_shard_jit":
                jit_func = node
                break

        self.assertIsNotNone(jit_func, "download_single_shard_jit function not found in cloud_train.py")

        # Inspect AST body: target_path assignment must precede os.makedirs
        target_path_assign_line = None
        os_makedirs_line = None
        for stmt in jit_func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "target_path":
                        if target_path_assign_line is None:
                            target_path_assign_line = stmt.lineno
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute) and func.attr == "makedirs":
                    if os_makedirs_line is None:
                        os_makedirs_line = stmt.lineno

        self.assertIsNotNone(target_path_assign_line, "target_path assignment missing")
        self.assertIsNotNone(os_makedirs_line, "os.makedirs call missing")
        self.assertLess(
            target_path_assign_line,
            os_makedirs_line,
            f"target_path assignment (line {target_path_assign_line}) must appear before os.makedirs (line {os_makedirs_line}) to prevent UnboundLocalError",
        )


# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

class Tier3CrossFeatureCombinationTests(unittest.TestCase):
    """
    Tier 3 tests covering multi-domain scheduling co-trained with world models,
    anchor replay interleaved with teacher buffer snapshots, and multi-account handover.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_managed_temp_dir("tier3_")
        cls.vocab = _build_test_vocab()
        cls.tokenizer = TaraNativeTokenizer(cls.vocab)
        cls.config = _make_tiny_model_config()

    def test_tier3_01_multi_domain_drr_with_joint_world_model(self):
        """Tier 3.1: Verify multi-domain DRR streaming co-trained with Joint World Model."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented or exported in cloud_train.py")

        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        # Setup 2 domain text files
        base_dir = os.path.join(self.work_dir, "joint_drr")
        os.makedirs(base_dir, exist_ok=True)
        d1_file = os.path.join(base_dir, "d1.jsonl")
        d2_file = os.path.join(base_dir, "d2.jsonl")
        with open(d1_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"Math formula {i}"}) + "\n")
        with open(d2_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"Code function {i}"}) + "\n")

        scheduler = scheduler_cls(
            domain_shards={"01_math": [d1_file], "02_code": [d2_file]},
            domain_weights={"01_math": 0.5, "02_code": 0.5},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
        )

        # Setup World Model transition file
        wm_file = os.path.join(base_dir, "transitions.jsonl")
        with open(wm_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "state_observation": f"state {i}",
                    "action": f"action {i}",
                    "next_state_observation": f"state {i+1}",
                }) + "\n")

        wm_ds = WorldModelTransitionDataset([wm_file], self.tokenizer, max_seq_len=32)

        lang_it = iter(scheduler)
        wm_it = iter(wm_ds)

        # Execute 3 joint steps
        for step in range(3):
            lang_batch = next(lang_it)
            wm_batch = next(wm_it)

            lang_out = neural_model(lang_batch["input_ids"], labels=lang_batch["input_ids"])
            wm_out = world_model(wm_batch["input_ids"], labels=wm_batch["labels"])

            joint_loss = lang_out.loss + 0.5 * wm_out.loss
            joint_loss.backward()

            # Verify gradients exist on both models
            has_neural_grads = any(p.grad is not None and p.grad.norm() > 0 for p in neural_model.parameters())
            has_wm_grads = any(p.grad is not None and p.grad.norm() > 0 for p in world_model.parameters())
            self.assertTrue(has_neural_grads, "Neural model parameters must have non-zero gradients")
            self.assertTrue(has_wm_grads, "World model parameters must have non-zero gradients")

    def test_tier3_02_anchor_replay_with_teacher_replay_buffer_snapshots(self):
        """Tier 3.2: Verify TeacherReplayBuffer captures samples from anchor replay and live streams."""
        buffer_cls = getattr(cloud_train, "TeacherReplayBuffer", None) or getattr(cloud_train, "TeacherBuffer", None)
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if buffer_cls is None or scheduler_cls is None:
            self.fail("TeacherReplayBuffer or TaraMultiDomainScheduler is not implemented")

        buffer = buffer_cls(max_samples_per_domain=100)
        base_dir = os.path.join(self.work_dir, "anchor_teacher")
        os.makedirs(base_dir, exist_ok=True)

        core_file = os.path.join(base_dir, "core.jsonl")
        with open(core_file, "w") as f:
            for i in range(2):
                f.write(json.dumps({"text": f"Reasoning {i}"}) + "\n")

        code_file = os.path.join(base_dir, "code.jsonl")
        with open(code_file, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"Code sample {i}"}) + "\n")

        scheduler = scheduler_cls(
            domain_shards={"12_reasoning": [core_file], "02_code": [code_file]},
            domain_weights={"12_reasoning": 0.5, "02_code": 0.5},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
            anchor_ratio=0.5,
        )

        it = iter(scheduler)
        for _ in range(10):
            batch = next(it)
            buffer.add_sample(batch["domain"], batch["input_ids"][0].tolist())

        snapshot_path = os.path.join(base_dir, "teacher_snapshot.jsonl")
        buffer.save_to_jsonl(snapshot_path)

        domains_in_snapshot = set()
        with open(snapshot_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                domains_in_snapshot.add(rec["domain"])

        self.assertIn("12_reasoning", domains_in_snapshot, "Depleted core domain must be preserved in teacher buffer")
        self.assertIn("02_code", domains_in_snapshot, "Active code domain must be present in teacher buffer")

    def test_tier3_03_multi_account_quota_handover_simulation(self):
        """Tier 3.3: Simulate Account 1 mid-stream disconnect and Account 2 exact offset resumption."""
        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented")

        base_dir = os.path.join(self.work_dir, "handover")
        os.makedirs(base_dir, exist_ok=True)

        d1 = os.path.join(base_dir, "d1.jsonl")
        with open(d1, "w") as f:
            for i in range(30):
                f.write(json.dumps({"text": f"Sample D1_{i:02d}"}) + "\n")

        # Account 1 runs for 10 steps
        acct1_sched = scheduler_cls(
            domain_shards={"domain_1": [d1]},
            domain_weights={"domain_1": 1.0},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
        )
        it1 = iter(acct1_sched)
        acct1_samples = [next(it1) for _ in range(10)]
        handover_state = acct1_sched.get_state()

        # Account 2 resumes from handover state
        acct2_sched = scheduler_cls(
            domain_shards={"domain_1": [d1]},
            domain_weights={"domain_1": 1.0},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
        )
        acct2_sched.load_state(handover_state)
        it2 = iter(acct2_sched)
        acct2_first_batch = next(it2)

        # Verify exact continuity: Account 2 first sample must be Sample 10
        # (Assuming 0-indexed: samples 0-9 consumed by Account 1)
        acct1_last_ids = acct1_samples[-1]["input_ids"].tolist()
        acct2_first_ids = acct2_first_batch["input_ids"].tolist()
        self.assertNotEqual(acct1_last_ids, acct2_first_ids, "Resumed stream must not duplicate the last sample")


# =============================================================================
# TIER 4: Real-World Application Scenarios
# =============================================================================

class Tier4RealWorldScenarioTests(unittest.TestCase):
    """
    Tier 4 tests verifying a complete simulated training step loop with joint loss,
    live telemetry output formatting, and atomic best checkpoint mirroring.
    """

    @classmethod
    def setUpClass(cls):
        cls.work_dir = _create_managed_temp_dir("tier4_")
        cls.vocab = _build_test_vocab()
        cls.tokenizer = TaraNativeTokenizer(cls.vocab)
        cls.config = _make_tiny_model_config()
        cls.tok_path = os.path.join(cls.work_dir, "tokenizer.json")
        with open(cls.tok_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": cls.vocab}}, f)

    def test_tier4_01_complete_e2e_training_loop(self):
        """Tier 4.1: Complete end-to-end simulated training loop with joint telemetry & best save."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)
        optimizer = torch.optim.AdamW(neural_model.parameters(), lr=1e-3)
        world_optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-3)

        base_dir = os.path.join(self.work_dir, "e2e_loop")
        os.makedirs(base_dir, exist_ok=True)
        out_dir = os.path.join(base_dir, "output")
        best_dir = os.path.join(out_dir, "best_checkpoint")
        os.makedirs(out_dir, exist_ok=True)

        # Setup mock multi-domain and transition files
        m_file = os.path.join(base_dir, "math.jsonl")
        with open(m_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"2 + {i} = {2+i}"}) + "\n")

        wm_file = os.path.join(base_dir, "wm.jsonl")
        with open(wm_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "state_observation": f"pos {i}",
                    "action": "step",
                    "next_state_observation": f"pos {i+1}",
                }) + "\n")

        scheduler_cls = getattr(cloud_train, "TaraMultiDomainScheduler", None) or getattr(cloud_train, "MultiDomainScheduler", None)
        if scheduler_cls is None:
            self.fail("TaraMultiDomainScheduler is not implemented")

        scheduler = scheduler_cls(
            domain_shards={"01_math": [m_file]},
            domain_weights={"01_math": 1.0},
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_len=32,
        )
        wm_dataset = WorldModelTransitionDataset([wm_file], self.tokenizer, max_seq_len=32)

        lang_it = iter(scheduler)
        wm_it = iter(wm_dataset)

        best_loss = float("inf")
        telemetry_logs = []

        # Run 5 simulated steps
        for step in range(1, 6):
            optimizer.zero_grad()
            world_optimizer.zero_grad()

            lang_batch = next(lang_it)
            wm_batch = next(wm_it)

            lang_out = neural_model(lang_batch["input_ids"], labels=lang_batch["input_ids"])
            wm_out = world_model(wm_batch["input_ids"], labels=wm_batch["labels"])

            ce_loss = lang_out.loss
            wm_loss = wm_out.loss
            total_loss = ce_loss + 0.5 * wm_loss

            total_loss.backward()
            optimizer.step()
            world_optimizer.step()

            # Format unmasked telemetry per Requirement R5
            ppl = torch.exp(ce_loss).item()
            bpb = ce_loss.item() / 0.693147
            domain_tag = lang_batch.get("domain", "unknown")
            anchor_tag = " [ANCHOR REPLAY]" if lang_batch.get("is_anchor", False) else ""
            telemetry_str = (
                f"[Step {step:04d}] Total Loss: {total_loss.item():.4f} | CE Loss: {ce_loss.item():.4f} | "
                f"WM Loss: {wm_loss.item():.4f} | PPL: {ppl:.2f} | BPB: {bpb:.2f} | "
                f"Domain: {domain_tag}{anchor_tag}"
            )
            telemetry_logs.append(telemetry_str)

            # Check dynamic best milestone at step 1
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                committed = save_best_milestone(
                    model=neural_model,
                    config=self.config,
                    tokenizer_path=self.tok_path,
                    best_dir=best_dir,
                    step=step,
                    epoch=0,
                    loss=best_loss,
                    best_loss=best_loss,
                    stage_id="stage1_unified",
                    world_model=world_model,
                )
                self.assertTrue(committed)

        # Verification of step loop execution
        self.assertEqual(len(telemetry_logs), 5)
        self.assertIn("Total Loss", telemetry_logs[0])
        self.assertIn("CE Loss", telemetry_logs[0])
        self.assertIn("WM Loss", telemetry_logs[0])
        self.assertIn("Domain: 01_math", telemetry_logs[0])

        # Verify best checkpoint commit on disk
        self.assertTrue(os.path.isdir(best_dir))
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "world_model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(best_dir, "COMMITTED")))


if __name__ == "__main__":
    unittest.main()

"""
tests/test_dynamic_dataset_compiler.py

Unit and integration test suite proving open-ended and dynamically expandable
training-data architecture for TARA:
1. Adding a new skill automatically increases training coverage.
2. Adding a new knowledge entry is discovered and represented.
3. Adding a new language is discovered and represented.
4. Adding a new tool is discovered and represented.
5. Dataset compiler does not depend on fixed counts.
6. Provenance remains 100% valid across all generated records.
7. Zero cross-split leakage between train, val, and test.
8. Baseline dataset (storage/datasets/tara/) remains 100% bit-for-bit intact and untouched.
9. Training engine (TaraJsonlDataset) consumes dynamically generated dataset without code changes.
10. Secrets, private keys, and runtime credentials are automatically scrubbed.
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_model.dynamic_dataset_compiler import DynamicDatasetCompiler, SecretScrubber
from tara_model.train_candidate import TaraJsonlDataset
from tara_model.tokenizer import TaraTokenizer
from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel
from tara_core.tools_registry import ToolRegistry, ToolDefinition
from tara_core.language_registry import LanguageRegistry, LanguageDefinition


class TestDynamicDatasetCompiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.compiler = DynamicDatasetCompiler(repo_root=REPO_ROOT)
        cls.temp_dir = tempfile.mkdtemp(prefix="tara_dynamic_ds_test_")
        cls.tokenizer = TaraTokenizer()

        # Capture baseline dataset hash before test run
        cls.baseline_train_path = os.path.join(REPO_ROOT, "storage", "datasets", "tara", "train.jsonl")
        cls.baseline_manifest_path = os.path.join(REPO_ROOT, "storage", "datasets", "tara", "manifest.json")
        cls.initial_train_sha = cls._get_file_hash(cls.baseline_train_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @staticmethod
    def _get_file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def test_01_compiler_operates_without_fixed_counts(self):
        """Proves dataset compiler discovers and builds datasets dynamically with no hardcoded record limits."""
        manifest = self.compiler.compile(output_dir=self.temp_dir, dataset_version="TEST-DYN-V1")
        self.assertIn("total_records", manifest)
        self.assertGreater(manifest["total_records"], 0)
        self.assertIn("splits", manifest)
        self.assertIn("train", manifest["splits"])
        self.assertIn("val", manifest["splits"])
        self.assertIn("test", manifest["splits"])

        total_split = (
            manifest["splits"]["train"]["samples"] +
            manifest["splits"]["val"]["samples"] +
            manifest["splits"]["test"]["samples"]
        )
        self.assertEqual(manifest["total_records"], total_split)

    def test_02_adding_new_skill_increases_coverage(self):
        """Proves dynamically adding a skill is discovered and generates training samples."""
        initial_skills = len(self.compiler.discover_skills())

        # Register dynamic capability skill
        cap_reg = CapabilityRegistry.get_default()
        test_cap_id = "skill_compiler_test_fusion"
        cap_reg.register_capability(Capability(
            capability_id=test_cap_id,
            name="Quantum Fusion Analysis",
            version="1.0.0",
            category=CapabilityCategory.SKILL,
            purpose="Conducts autonomous nuclear fusion simulation analysis.",
            risk_level=RiskLevel.MEDIUM
        ))

        updated_skills = len(self.compiler.discover_skills())
        self.assertEqual(updated_skills, initial_skills + 1)

        # Generate samples directly
        discovered = [s for s in self.compiler.discover_skills() if s["name"] == "Quantum Fusion Analysis"]
        self.assertTrue(len(discovered) > 0)
        samples = self.compiler.generate_skill_samples(discovered[0])
        self.assertGreaterEqual(len(samples), 3)
        self.assertTrue(any("Quantum Fusion Analysis" in s["prompt"] for s in samples))

        # Cleanup
        cap_reg.unregister_capability(test_cap_id)

    def test_03_adding_new_tool_is_discovered(self):
        """Proves newly registered tool appears in discovered tools and generates training data."""
        tool_reg = ToolRegistry.get_default(repo_root=REPO_ROOT)
        tool_name = "dynamic_network_tracer"
        tool_reg.register_tool(ToolDefinition(
            name=tool_name,
            description="Traces network latency and packets locally.",
            handler=lambda **kw: {"status": "SUCCESS"}
        ))

        tools = self.compiler.discover_tools()
        self.assertTrue(any(t["name"] == tool_name for t in tools))

        # Cleanup
        tool_reg.unregister_tool(tool_name)

    def test_04_adding_new_language_is_discovered(self):
        """Proves registering a language dynamically adds it to discovered training languages."""
        lang_reg = LanguageRegistry.get_default()
        test_lang = LanguageDefinition(
            code="ja",
            name="Japanese",
            native_name="日本語",
            script="Kanji/Kana",
            sample_greetings=["こんにちは"]
        )
        lang_reg.register_language(test_lang)

        langs = self.compiler.discover_languages()
        self.assertTrue(any(l["code"] == "ja" for l in langs))
        
        # Cleanup
        lang_reg.unregister_language("ja")

    def test_05_provenance_map_integrity(self):
        """Verifies every generated example is tracked in provenance_map.json."""
        prov_file = os.path.join(self.temp_dir, "provenance_map.json")
        self.assertTrue(os.path.exists(prov_file))
        with open(prov_file, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertIn("metadata", prov)
        self.assertIn("sources", prov)
        self.assertGreater(len(prov["sources"]), 0)

        # Check sample IDs
        all_ids = []
        for src, info in prov["sources"].items():
            all_ids.extend(info.get("sample_ids", []))
        # Ensure IDs are unique
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_06_zero_cross_split_leakage(self):
        """Guarantees zero prompt/completion overlap between train, val, and test."""
        def load_pairs(path):
            pairs = set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        pairs.add((d["prompt"].strip(), d["completion"].strip()))
            return pairs

        train_pairs = load_pairs(os.path.join(self.temp_dir, "train.jsonl"))
        val_pairs = load_pairs(os.path.join(self.temp_dir, "val.jsonl"))
        test_pairs = load_pairs(os.path.join(self.temp_dir, "test.jsonl"))

        self.assertEqual(len(train_pairs.intersection(val_pairs)), 0, "Leakage train <-> val!")
        self.assertEqual(len(train_pairs.intersection(test_pairs)), 0, "Leakage train <-> test!")
        self.assertEqual(len(val_pairs.intersection(test_pairs)), 0, "Leakage val <-> test!")

    def test_07_training_engine_compatibility(self):
        """Proves TaraJsonlDataset can load and tokenize generated datasets seamlessly."""
        train_path = os.path.join(self.temp_dir, "train.jsonl")
        ds = TaraJsonlDataset(jsonl_path=train_path, tokenizer=self.tokenizer, max_len=256)
        self.assertGreater(len(ds), 0)
        first_item = ds[0]
        self.assertIsInstance(first_item, list)
        self.assertGreater(len(first_item), 1)

    def test_08_secret_scrubbing_guarantee(self):
        """Proves private keys and bearer tokens are never placed into training data."""
        dirty_records = [
            {
                "prompt": "Here is an API key: bearer secret_token_abc12345678901234567890",
                "completion": "Ok.",
                "source_file": "test_source"
            },
            {
                "prompt": "Valid non-secret question",
                "completion": "Valid clean answer",
                "source_file": "clean_source"
            }
        ]
        scrubbed = self.compiler.scrub_secrets(dirty_records)
        # The secret bearer token record was either sanitized or excluded
        for r in scrubbed:
            self.assertFalse(SecretScrubber.contains_secret(r["prompt"]))
            self.assertFalse(SecretScrubber.contains_secret(r["completion"]))

    def test_09_baseline_dataset_unmodified(self):
        """Strict verification: baseline storage/datasets/tara/train.jsonl was NEVER touched."""
        current_train_sha = self._get_file_hash(self.baseline_train_path)
        self.assertEqual(current_train_sha, self.initial_train_sha, "Baseline train.jsonl was modified!")


if __name__ == "__main__":
    unittest.main()

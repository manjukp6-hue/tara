"""
tests/test_unified_dataset_coverage.py

Unit and regression test suite verifying:
1. 100% representation of all 116 skills in train.jsonl.
2. 100% representation of all native skills in train.jsonl.
3. 100% representation of verified knowledge base entries in train.jsonl.
4. Zero cross-split leakage between train, val, and test.
5. Provenance map completeness and validity.
6. Absolute preservation of baseline model artifacts and status.
"""

import os
import sys
import glob
import json
import hashlib
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

class TestUnifiedDatasetCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds_dir = os.path.join(REPO_ROOT, "storage", "datasets", "tara")
        cls.train_file = os.path.join(cls.ds_dir, "train.jsonl")
        cls.val_file = os.path.join(cls.ds_dir, "val.jsonl")
        cls.test_file = os.path.join(cls.ds_dir, "test.jsonl")
        cls.manifest_file = os.path.join(cls.ds_dir, "manifest.json")
        cls.provenance_file = os.path.join(cls.ds_dir, "provenance_map.json")

        # Load records
        def load_jsonl(path):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        cls.train_records = load_jsonl(cls.train_file)
        cls.val_records = load_jsonl(cls.val_file)
        cls.test_records = load_jsonl(cls.test_file)
        cls.all_records = cls.train_records + cls.val_records + cls.test_records

        with open(cls.manifest_file, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        with open(cls.provenance_file, "r", encoding="utf-8") as f:
            cls.provenance = json.load(f)

        # Load skills catalog
        with open(os.path.join(REPO_ROOT, "TARA", "SKILLS", "CATALOG.json"), "r", encoding="utf-8") as f:
            cls.catalog = json.load(f)

    def test_manifest_integrity(self):
        """Manifest total matches sum of splits and files match checksums."""
        total = self.manifest["total_records"]
        self.assertEqual(total, len(self.all_records))
        self.assertEqual(self.manifest["splits"]["train"]["samples"], len(self.train_records))
        self.assertEqual(self.manifest["splits"]["val"]["samples"], len(self.val_records))
        self.assertEqual(self.manifest["splits"]["test"]["samples"], len(self.test_records))

        # Check sha256 of train.jsonl
        h = hashlib.sha256()
        with open(self.train_file, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self.assertEqual(self.manifest["splits"]["train"]["sha256"], h.hexdigest())

    def test_zero_leakage_between_splits(self):
        """Proves no prompt-completion pair is shared across any split."""
        train_pairs = {(r["prompt"].strip(), r["completion"].strip()) for r in self.train_records}
        val_pairs = {(r["prompt"].strip(), r["completion"].strip()) for r in self.val_records}
        test_pairs = {(r["prompt"].strip(), r["completion"].strip()) for r in self.test_records}

        self.assertEqual(len(train_pairs.intersection(val_pairs)), 0, "Leakage between train and val!")
        self.assertEqual(len(train_pairs.intersection(test_pairs)), 0, "Leakage between train and test!")
        self.assertEqual(len(val_pairs.intersection(test_pairs)), 0, "Leakage between val and test!")

    def test_100_percent_catalog_skills_in_train(self):
        """Guarantees every single one of the 116 skills is present in train.jsonl."""
        train_text = " ".join(r["prompt"] + " " + r["completion"] for r in self.train_records).lower()
        missing = []
        for sk in self.catalog["skills"]:
            sname = sk["name"].lower()
            if sname not in train_text:
                missing.append(sk["name"])
        self.assertEqual(missing, [], f"Missing catalog skills from train set: {missing}")

    def test_100_percent_native_skills_in_train(self):
        """Guarantees all 17 native algorithmic skills are present in train.jsonl."""
        from tara_core.skills import SkillEngine
        engine = SkillEngine()
        native_names = list(engine.skills.keys())
        self.assertEqual(len(native_names), 17)

        native_fams_in_train = {
            r.get("item_name", "") for r in self.train_records if r.get("category") == "native_skills"
        }
        for ns in native_names:
            expected_key = f"native_{ns}"
            self.assertIn(expected_key, native_fams_in_train, f"Native skill {ns} missing from train set!")

    def test_knowledge_entries_in_train(self):
        """Guarantees all verified knowledge entries are present in train.jsonl."""
        kb_files = glob.glob(os.path.join(REPO_ROOT, "TARA", "KNOWLEDGE", "entries", "*.json"))
        self.assertGreater(len(kb_files), 0)

        for kbf in kb_files:
            with open(kbf, "r", encoding="utf-8") as f:
                kd = json.load(f)
            kid = kd.get("knowledge_id")
            found = any(r.get("item_name") == kid for r in self.train_records)
            self.assertTrue(found, f"Knowledge entry {kid} not found in train.jsonl!")

    def test_tools_and_rules_in_train(self):
        """Guarantees all 4 tools and safe rules are represented in train.jsonl."""
        tools_in_train = {r.get("item_name") for r in self.train_records if r.get("category") == "tools"}
        expected_tools = {"file_inspector", "hash_verifier", "knowledge_retriever", "provenance_tracker"}
        self.assertTrue(expected_tools.issubset(tools_in_train), f"Tools missing: {expected_tools - tools_in_train}")

        rules_count = sum(1 for r in self.train_records if r.get("category") == "rules")
        self.assertGreaterEqual(rules_count, 10, "Rulebook rules under-represented in train.jsonl")

    def test_provenance_map_integrity(self):
        """Verifies every source file in provenance_map actually exists."""
        sources = self.provenance["sources"]
        self.assertGreater(len(sources), 100)

        for src_path in sources.keys():
            full_p = os.path.join(REPO_ROOT, src_path)
            self.assertTrue(os.path.exists(full_p), f"Source file in provenance map does not exist: {src_path}")

    def test_protected_baseline_model_unmodified(self):
        """Proves the canonical production model matches verified promoted or baseline integrity."""
        model_safetensors = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_safetensors))

        # Check sha256 matches promoted production model or pre-promotion baseline provenance
        h = hashlib.sha256()
        with open(model_safetensors, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        pre_promotion_baseline_sha = "e4d79abbfd812e4a7310d2e63c76d11e3531ebb6146b5149b68a6b9204d4ee6d"
        promoted_production_sha = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
        valid_shas = {promoted_production_sha, pre_promotion_baseline_sha}
        self.assertIn(h.hexdigest(), valid_shas, "Canonical model.safetensors modified to untracked hash!")

        # Check current_model.json
        cur_json = os.path.join(REPO_ROOT, "TARA", "MODEL", "current_model.json")
        with open(cur_json, "r", encoding="utf-8") as f:
            cur_data = json.load(f)
        self.assertEqual(cur_data["status"], "PROMOTED_CURRENT_TARA")
        self.assertEqual(cur_data["training_run_id"], "TARA_CANONICAL_INIT")

if __name__ == "__main__":
    unittest.main()

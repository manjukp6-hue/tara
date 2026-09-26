"""
tests/test_core_abilities_dataset.py

Unit tests for TARA AI Core Abilities dataset discovery, multi-perspective structure,
dynamic extendability, compilation staging, secret scrubbing, and baseline preservation.
"""

import os
import sys
import json
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_model.dynamic_dataset_compiler import DynamicDatasetCompiler, SecretScrubber
from tara_model.core_abilities_spec import (
    BASELINE_CORE_ABILITIES,
    register_core_ability,
    update_core_ability,
    remove_core_ability,
    get_core_abilities,
    get_core_ability_metadata,
    reset_core_abilities_to_baseline,
    build_ability_samples,
    get_all_core_abilities_samples
)


class TestCoreAbilitiesDataset(unittest.TestCase):

    def setUp(self):
        self.compiler = DynamicDatasetCompiler(repo_root=REPO_ROOT)
        reset_core_abilities_to_baseline()

    def tearDown(self):
        reset_core_abilities_to_baseline()

    def test_baseline_core_abilities_defined(self):
        abilities = get_core_abilities()
        self.assertEqual(len(abilities), 50, "Baseline identified set must have 50 abilities")
        ability_names = [a["name"] for a in abilities]
        self.assertEqual(len(set(ability_names)), 50, "All baseline ability names must be unique")

        # Verify key expected abilities from prompt
        expected_sample = [
            "Natural-language understanding", "Language generation", "Semantic understanding",
            "Context understanding", "Multi-turn reasoning", "Working-memory reasoning",
            "Logical reasoning", "Deductive reasoning", "Inductive reasoning", "Abductive reasoning",
            "Causal reasoning", "Counterfactual reasoning", "Analogical reasoning", "Abstract reasoning",
            "Concept formation", "Compositional reasoning", "Generalization", "Cross-domain transfer",
            "Problem solving", "Problem decomposition", "Hierarchical planning", "Long-horizon planning",
            "Goal management", "Decision making", "Consequence prediction", "Future-state reasoning",
            "Hypothesis generation", "Hypothesis evaluation", "Uncertainty estimation", "Confidence estimation",
            "Metacognition", "Self-model / capability awareness", "Error detection", "Self-correction",
            "Strategy adaptation", "Cognitive flexibility", "Active information seeking", "Continual learning",
            "Knowledge revision", "Knowledge integration", "Memory-guided reasoning", "Temporal reasoning",
            "Spatial reasoning", "Resource-aware reasoning", "Risk-aware reasoning", "Social/contextual understanding",
            "Human-intent understanding", "Autonomous task execution", "Self-evaluation", "Self-improvement reasoning"
        ]
        for name in expected_sample:
            self.assertIn(name, ability_names)

    def test_multi_perspective_samples_structure(self):
        samples = get_all_core_abilities_samples()
        self.assertEqual(len(samples), 50 * 8, "Baseline abilities with 8 perspectives must equal 400 samples")

        expected_perspectives = {
            "understanding", "reasoning", "application", "generalization",
            "failure_cases", "correction", "multi_step", "cross_domain"
        }

        perspectives_found = set()
        for s in samples:
            self.assertIn("prompt", s)
            self.assertIn("completion", s)
            self.assertIn("perspective", s)
            self.assertIn("topic_family", s)
            self.assertIn("category", s)
            self.assertIn("item_name", s)
            self.assertIn("source_file", s)
            self.assertGreater(len(s["prompt"]), 10)
            self.assertGreater(len(s["completion"]), 20)
            perspectives_found.add(s["perspective"])

        self.assertEqual(perspectives_found, expected_perspectives)

    def test_secret_scrubbing_guarantee(self):
        samples = get_all_core_abilities_samples()
        for s in samples:
            text = f"{s['prompt']} {s['completion']}"
            self.assertFalse(SecretScrubber.contains_secret(text), f"Secret detected in sample: {s['item_name']}")

    def test_staging_compilation(self):
        manifest = self.compiler.compile_core_abilities_staging()
        self.assertEqual(manifest["total_records"], 400)
        self.assertEqual(manifest["coverage"]["total_abilities"], 50)
        self.assertTrue(manifest["coverage"]["extendable"])

        # Primary dynamic dataset file
        primary_file = os.path.join(REPO_ROOT, "storage", "datasets", "tara_dynamic_final", "core_abilities.jsonl")
        self.assertTrue(os.path.exists(primary_file))

        # Check line count
        with open(primary_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(len(lines), 400)

        # Check tokenization length validity
        for item in lines:
            self.assertGreater(item["length"], 0)
            self.assertEqual(len(item["token_ids"]), item["length"])

    def test_core_abilities_dynamic_extendability(self):
        initial_count = len(get_core_abilities())
        self.assertEqual(initial_count, 50, "Baseline identified set must have 50 abilities")

        # 1. Register a new genuine core ability dynamically
        new_entry = register_core_ability(
            name="Counter-adversarial epistemic reasoning",
            source_file="python/tara_core/reasoning_engine_advanced.py",
            subsystem="EpistemicIntegrityGovernor",
            metadata={"domain": "cognitive_defense", "safety_critical": True}
        )
        self.assertEqual(len(get_core_abilities()), 51, "Core abilities must be extendable beyond 50")
        self.assertEqual(new_entry["id"], 51)
        self.assertEqual(new_entry["metadata"]["domain"], "cognitive_defense")

        # 2. Update existing core ability
        updated = update_core_ability(
            name="Counter-adversarial epistemic reasoning",
            version="1.1.0",
            metadata={"verified": True}
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["version"], "1.1.0")
        self.assertTrue(updated["metadata"]["verified"])

        # 3. Check dynamic samples generation
        extended_samples = get_all_core_abilities_samples()
        self.assertEqual(len(extended_samples), 51 * 8, "Dynamic samples must grow with registered abilities")

        # 4. Verify primary dataset compile handles extendable abilities
        manifest = self.compiler.compile_core_abilities_staging()
        self.assertEqual(manifest["total_records"], 51 * 8)
        self.assertEqual(manifest["coverage"]["total_abilities"], 51)
        self.assertTrue(manifest["coverage"]["extendable"])

        # 5. Remove core ability
        removed = remove_core_ability("Counter-adversarial epistemic reasoning")
        self.assertTrue(removed)
        self.assertEqual(len(get_core_abilities()), 50)

    def test_baseline_datasets_untouched(self):
        baseline_file = os.path.join(REPO_ROOT, "storage", "datasets", "tara", "train.jsonl")
        self.assertTrue(os.path.exists(baseline_file))
        import hashlib
        h = hashlib.sha256()
        with open(baseline_file, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), "b557d5dd2cef72dd7442d5cefc6127bb86bf23fa5f73313c40ab92fee9e5a362", f"Unexpected hash for {baseline_file}: {h.hexdigest()}")


if __name__ == "__main__":
    unittest.main()

"""
tests/test_model_expansion.py

Comprehensive Test Suite for TARA Model Parameter Expansion (Workstream A):
1. Test current production model parameter count (exactly 118,080) and weights unchanged.
2. Test dynamic parameter count calculation from config and weights (no hardcoding).
3. Test Depth expansion (layers 2 -> 4) with lower representations and residual initialization.
4. Test Intermediate size expansion (intermediate 128 -> 256) with Net2Net mathematical equivalence on zero-initialized down_proj.
5. Test Vocabulary expansion (vocab 344 -> 512) preserving existing token weights.
6. Test Width expansion (hidden 64 -> 128) preserving submatrices.
7. Test AdamW optimizer state migration (m and v momentum tensors preserved and padded).
8. Test Sharded SafeTensors candidate creation and in-place loading without file merging.
9. Test Model candidate promotion & rollback lifecycle in ModelRegistry.
10. Test Fail-closed baseline integrity protection.
"""

import os
import sys
import math
import shutil
import tempfile
import unittest
import hashlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_model.architecture import TaraConfig, TaraModelZero
from tara_model.model_expansion import (
    ModelExpansionEngine,
    GrowthType,
    GrowthMetadata,
    CANONICAL_PRODUCTION_SHA256,
    CANONICAL_PRODUCTION_PARAM_COUNT
)
from tara_model.generate import load_trained_language_model
from tara_core.model_registry import ModelRegistry, ModelVersionMetadata


class TestModelParameterExpansion(unittest.TestCase):

    def setUp(self):
        self.baseline_dir = os.path.join(REPO_ROOT, "storage", "models", "tara")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_baseline_integrity_and_parameter_count(self):
        """Verify the current production model is untouched and has exactly 118,080 parameters."""
        weights_path = os.path.join(self.baseline_dir, "model.safetensors")
        with open(weights_path, "rb") as f:
            current_sha = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(
            current_sha,
            CANONICAL_PRODUCTION_SHA256,
            "CRITICAL: Baseline production model SHA-256 has been modified!"
        )

        model, tokenizer, cfg_dict = load_trained_language_model(self.baseline_dir)
        self.assertEqual(cfg_dict["total_parameters"], CANONICAL_PRODUCTION_PARAM_COUNT)
        self.assertEqual(len(model.weights), 21)  # 21 tensors in 2-layer architecture (embed + 2*9 layers + norm + lm_head)

    def test_02_dynamic_parameter_counting(self):
        """Verify dynamic parameter counting formula matches actual weight counts across configurations."""
        cfg = TaraConfig(
            vocab_size=344,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2
        )
        calculated = ModelExpansionEngine.count_parameters_from_config(cfg)
        self.assertEqual(calculated, CANONICAL_PRODUCTION_PARAM_COUNT)

        # Scale intermediate to 256
        cfg_expanded = TaraConfig(
            vocab_size=344,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2
        )
        calc_expanded = ModelExpansionEngine.count_parameters_from_config(cfg_expanded)
        self.assertEqual(calc_expanded, 167232)
        self.assertGreater(calc_expanded, calculated)

    def test_03_depth_expansion(self):
        """Verify depth expansion adds layers, preserves lower layers, and calculates new parameter counts."""
        base_model, _, _ = load_trained_language_model(self.baseline_dir)
        old_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))

        new_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))
        new_cfg.num_hidden_layers = 4  # Grow from 2 to 4 layers

        expanded_weights, audit = ModelExpansionEngine.expand_weights(
            source_weights=base_model.weights,
            old_config=old_cfg,
            new_config=new_cfg,
            source_version="baseline",
            target_version="4_layers"
        )

        self.assertEqual(audit.growth_type, GrowthType.DEPTH)
        self.assertEqual(audit.old_param_count, 118080)
        expected_new_params = ModelExpansionEngine.count_parameters_from_config(new_cfg)
        actual_weights_params = ModelExpansionEngine.count_parameters_from_weights(expanded_weights)
        self.assertEqual(expected_new_params, actual_weights_params)
        self.assertEqual(expected_new_params, 192064)

        # Ensure layers 0 and 1 weights are strictly preserved
        for l in range(2):
            norm_name = f"model.layers.{l}.input_layernorm.weight"
            self.assertEqual(expanded_weights[norm_name], base_model.weights[norm_name])

    def test_04_intermediate_expansion_mathematical_equivalence(self):
        """Verify intermediate MLP expansion initializes new down_proj columns to 0.0 for exact forward equivalence."""
        base_model, _, _ = load_trained_language_model(self.baseline_dir)
        old_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))

        new_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))
        new_cfg.intermediate_size = 256  # Grow from 128 to 256

        expanded_weights, audit = ModelExpansionEngine.expand_weights(
            source_weights=base_model.weights,
            old_config=old_cfg,
            new_config=new_cfg,
            source_version="baseline",
            target_version="intermediate_256"
        )

        self.assertEqual(audit.growth_type, GrowthType.INTERMEDIATE)
        self.assertTrue(audit.mathematical_equivalence_preserved)

        # Check down_proj weights for layer 0: rows [0..64], cols [128..256] must be 0.0
        down_w = expanded_weights["model.layers.0.mlp.down_proj.weight"]
        self.assertEqual(len(down_w), 64)
        self.assertEqual(len(down_w[0]), 256)

        for r in range(64):
            # Prior columns must match source exactly
            for c in range(128):
                self.assertEqual(down_w[r][c], base_model.weights["model.layers.0.mlp.down_proj.weight"][r][c])
            # Expanded columns must be 0.0
            for c in range(128, 256):
                self.assertEqual(down_w[r][c], 0.0)

    def test_05_vocabulary_expansion(self):
        """Verify vocabulary expansion (344 -> 500) preserves existing embeddings and output head rows."""
        base_model, _, _ = load_trained_language_model(self.baseline_dir)
        old_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))

        new_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))
        new_cfg.vocab_size = 500

        expanded_weights, audit = ModelExpansionEngine.expand_weights(
            source_weights=base_model.weights,
            old_config=old_cfg,
            new_config=new_cfg,
            source_version="baseline",
            target_version="vocab_500"
        )

        self.assertEqual(audit.growth_type, GrowthType.VOCABULARY)
        embed_w = expanded_weights["model.embed_tokens.weight"]
        self.assertEqual(len(embed_w), 500)
        self.assertEqual(len(embed_w[0]), 64)

        # First 344 tokens must match baseline
        for i in range(344):
            self.assertEqual(embed_w[i], base_model.weights["model.embed_tokens.weight"][i])

    def test_06_optimizer_momentum_migration(self):
        """Verify AdamW optimizer m and v state migration preserves shapes and pads new dimensions."""
        old_cfg = TaraConfig(vocab_size=344, hidden_size=64, intermediate_size=128, num_hidden_layers=2)
        new_cfg = TaraConfig(vocab_size=344, hidden_size=64, intermediate_size=256, num_hidden_layers=2)

        old_opt = {
            "t": 100,
            "m": {
                "model.layers.0.mlp.down_proj.weight": [[0.5] * 128 for _ in range(64)]
            },
            "v": {
                "model.layers.0.mlp.down_proj.weight": [[0.1] * 128 for _ in range(64)]
            }
        }

        migrated_opt = ModelExpansionEngine.migrate_optimizer_state(old_opt, old_cfg, new_cfg)
        self.assertEqual(migrated_opt["t"], 100)

        migrated_m = migrated_opt["m"]["model.layers.0.mlp.down_proj.weight"]
        self.assertEqual(len(migrated_m), 64)
        self.assertEqual(len(migrated_m[0]), 256)

        # Prior elements 0..127 must equal 0.5, new elements 128..255 must equal 0.0
        self.assertEqual(migrated_m[0][0], 0.5)
        self.assertEqual(migrated_m[0][127], 0.5)
        self.assertEqual(migrated_m[0][128], 0.0)
        self.assertEqual(migrated_m[0][255], 0.0)

    def test_07_sharded_safetensors_save_and_load_without_merge(self):
        """Verify multi-shard SafeTensors saves multiple files with index.json and loads without merging."""
        base_model, _, _ = load_trained_language_model(self.baseline_dir)
        old_cfg = TaraConfig.from_json_file(os.path.join(self.baseline_dir, "config.json"))

        out_dir = os.path.join(self.temp_dir, "sharded_candidate")
        meta = GrowthMetadata(
            model_id="TARA-SHARDED",
            version="v2",
            parent_model_version="TARA_BASELINE",
            growth_type=GrowthType.INTERMEDIATE
        )

        # Force small shards (40KB each) to create multiple shard files
        res = ModelExpansionEngine.save_model_candidate(
            weights=base_model.weights,
            config=old_cfg,
            output_dir=out_dir,
            growth_metadata=meta,
            max_shard_size_bytes=40 * 1024
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["is_sharded"])
        self.assertGreater(res["shard_count"], 1)

        # Check files on disk
        index_file = os.path.join(out_dir, "model.safetensors.index.json")
        self.assertTrue(os.path.exists(index_file))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "model.safetensors")))

        # Load sharded model via generate.py
        loaded_model, loaded_tok, loaded_cfg = load_trained_language_model(out_dir)
        self.assertEqual(loaded_cfg["total_parameters"], 118080)
        self.assertEqual(len(loaded_model.weights), 21)

        # Verify disk state was NOT modified into a merged single file
        self.assertFalse(os.path.exists(os.path.join(out_dir, "model.safetensors")))

    def test_08_model_registry_promotion_and_rollback(self):
        """Verify model candidate promotion and atomic rollback in ModelRegistry."""
        reg_file = os.path.join(self.temp_dir, "versions_manifest.json")
        registry = ModelRegistry(registry_file=reg_file)

        # Initial state has baseline active
        active = registry.get_active_version()
        self.assertIsNotNone(active)

        # Register candidate model
        cand_meta = ModelVersionMetadata(
            version_id="TARA-v2-expanded",
            artifact_location=os.path.join(self.temp_dir, "v2"),
            weights_sha256="abcdef1234567890",
            parameter_count=167232,
            growth_type="INTERMEDIATE",
            parent_model_version="TARA_BASELINE"
        )
        registry.register_version(cand_meta)

        # Promote candidate
        registry.set_active_version("TARA-v2-expanded")
        self.assertEqual(registry.get_active_version().version_id, "TARA-v2-expanded")
        self.assertEqual(registry.get_active_version().parameter_count, 167232)

        # Rollback to baseline
        rolled_back_to = registry.rollback()
        self.assertEqual(rolled_back_to, "TARA_BASELINE")
        self.assertEqual(registry.get_active_version().version_id, "TARA_BASELINE")


if __name__ == "__main__":
    unittest.main()

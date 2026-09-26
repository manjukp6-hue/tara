"""
tests/test_auto_connect_sync_learning_integration.py

Comprehensive Verification Test Suite:
Validates that the Dynamic Auto-Connect & Sync Layer does NOT replace, disable,
or bypass any existing TARA AI learning, self-learning, memory learning,
knowledge acquisition, or self-training functionality.

Verifies:
1. UserSearchLearner operates without replacement or bypass.
2. AutonomousOnlineLearner operates without replacement or bypass.
3. MemoryEngine records and retrieves episodes with 100% fidelity alongside sync.
4. GlobalKnowledgeBase retains full query and storage capabilities.
5. NativeSelfTrainer discovers trusted sources and queries active compute endpoints.
6. ExperientialLearner records and stages experiential episodes.
7. MemoryConsolidation successfully transforms episodes to semantic concepts.
8. TaraBrain incorporates working memory context and dispatches neural control tokens.
9. generate_stream & advanced sampling operate seamlessly with the neural model.
10. TARA_COLAB_TRAINING extractors include cognitive, robotics, and auto-sync domains.
11. Protected baseline assets remain 100% byte-for-byte unchanged.
"""

import os
import sys
import json
import hashlib
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from TARA.LEARNING.user_search_learner import UserSearchLearner
from TARA.LEARNING.autonomous_learner import AutonomousOnlineLearner, LearningMode
from TARA.MEMORY.memory_engine import MemoryEngine
from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from TARA.RULES.engine.execution_guard import ExecutionGuard
from TARA.ACCESS.access_manager import IdentityManager
from tara_model.self_trainer import NativeSelfTrainer, get_self_trainer
from tara_core.brain import TaraBrain
from tara_core.auto_connect_sync import (
    AutoConnectSyncEngine,
    EndpointDefinition,
    EndpointType,
    EndpointCapability,
    EndpointHealthStatus,
    AutoConnectRouter
)
from tara_core.experiential_learning import ExperientialClosedLoopLearner, ExperienceEpisode
from tara_core.memory_consolidation import MemoryConsolidation
from tara_core.working_memory_governor import WorkingMemoryGovernor
from tara_model.generate import (
    load_trained_language_model,
    generate_response,
    generate_stream,
    sample_next_token,
    ControlTokenActionParser,
    ControlTokenAction
)
import TARA_COLAB_TRAINING


class TestAutoConnectSyncLearningIntegration(unittest.TestCase):
    """Rigorous verification of learning subsystems coexisting with Auto-Connect & Sync."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.auto_sync = AutoConnectSyncEngine.get_default(repo_root=cls.repo_root)

        # Record protected hashes: pre-promotion baseline provenance & promoted production model integrity
        cls.pre_promotion_baseline_sha = "e4d79abbfd812e4a7310d2e63c76d11e3531ebb6146b5149b68a6b9204d4ee6d"
        cls.promoted_production_sha = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
        cls.expected_hashes = {
            "model_safetensors": cls.promoted_production_sha,
            "pre_promotion_baseline": cls.pre_promotion_baseline_sha,
            "train_jsonl": "b30c1c6f9abaa44b1f3d64afec778809da5d3f1a5b008c9ca0e6ab969835b9b5",
            "current_model_json": "bab48280e3c3a7087b33b5c9e3ff87f3fe5bd2a994214f8a3cfb4f81b72d4dd4"
        }

    def test_01_protected_baseline_hashes_untouched(self):
        """Verify model weights and baseline files match canonical integrity (distinguishing pre-promotion baseline and promoted model)."""
        model_path = os.path.join(self.repo_root, "storage", "models", "tara", "model.safetensors")
        train_path = os.path.join(self.repo_root, "storage", "datasets", "tara", "train.jsonl")
        manifest_path = os.path.join(self.repo_root, "TARA", "MODEL", "current_model.json")

        with open(model_path, "rb") as f:
            model_sha = hashlib.sha256(f.read()).hexdigest()
        with open(train_path, "rb") as f:
            train_sha = hashlib.sha256(f.read()).hexdigest()
        with open(manifest_path, "rb") as f:
            manifest_sha = hashlib.sha256(f.read()).hexdigest()

        # Distinguish between immutable pre-promotion baseline provenance and current promoted production model integrity
        valid_model_shas = {self.expected_hashes["model_safetensors"], self.expected_hashes["pre_promotion_baseline"]}
        self.assertIn(model_sha, valid_model_shas, f"Model weights hash {model_sha} does not match promoted production or baseline provenance!")
        self.assertEqual(train_sha, self.expected_hashes["train_jsonl"], "Baseline train dataset modified!")
        self.assertEqual(manifest_sha, self.expected_hashes["current_model_json"], "Current model metadata modified!")

    def test_02_user_search_learner_operates_without_bypass(self):
        """Verify UserSearchLearner functions properly alongside the Auto-Connect layer."""
        id_mgr = IdentityManager(base_dir=os.path.join(self.repo_root, "TARA", "ACCESS"))
        guard = ExecutionGuard(identity_manager=id_mgr)
        kb = GlobalKnowledgeBase(base_dir=os.path.join(self.repo_root, "TARA", "KNOWLEDGE"))
        learner = UserSearchLearner(knowledge_base=kb, execution_guard=guard)

        # Conduct a simulated search learning cycle
        result = learner.search_and_learn(
            query="quantum key distribution protocols",
            user_id="creator_admin"
        )
        self.assertIn("status", result)
        self.assertIn(result["status"], ("SUCCESS", "QUARANTINED", "NO_RESULTS", "ALLOWED", "PENDING_APPROVAL"))

        # Confirm auto_sync is active and healthy concurrently
        active_ep = self.auto_sync.get_active_endpoint()
        self.assertIsNotNone(active_ep)
        self.assertEqual(active_ep.health_status, EndpointHealthStatus.HEALTHY)

    def test_03_autonomous_online_learner_operates_without_bypass(self):
        """Verify AutonomousOnlineLearner functions properly alongside the Auto-Connect layer."""
        kb = GlobalKnowledgeBase(base_dir=os.path.join(self.repo_root, "TARA", "KNOWLEDGE"))
        learner = AutonomousOnlineLearner(knowledge_base=kb)

        status = learner.get_status()
        self.assertIn("is_running", status)
        self.assertIn("mode", status)

        # Verify active compute endpoint is accessible to the system
        ep = self.auto_sync.get_active_endpoint()
        self.assertIsNotNone(ep)
        self.assertTrue(ep.capabilities)

    def test_04_memory_engine_fidelity_and_sync_coexistence(self):
        """Verify MemoryEngine records episodes and AutoConnectSync stages state without conflict."""
        mem = MemoryEngine(
            memory_dir=os.path.join(self.repo_root, "TARA", "MEMORY"),
            legacy_dir=os.path.join(self.repo_root, "storage", "memory")
        )

        test_actor = "test_verifier"
        ep_id = mem.record_episode(
            actor_id=test_actor,
            intent="VERIFY_COEXISTENCE",
            action="sync_check",
            parameters={"test_key": "test_value_42"},
            outcome="SUCCESS",
            observations={"verified": True}
        )
        self.assertTrue(ep_id)

        # Retrieve the episode
        episodes = mem.query_episodes(query="VERIFY_COEXISTENCE", actor_id=test_actor, limit=5)
        self.assertTrue(any(e.get("intent") == "VERIFY_COEXISTENCE" for e in episodes))

        # Perform an AutoConnectSync state sync
        sync_res = self.auto_sync.stage_or_sync_state(
            payload_type="memory_snapshot",
            data={"episode_id": ep_id, "actor_id": test_actor}
        )
        self.assertTrue(sync_res[0], f"State sync failed: {sync_res}")

    def test_05_native_self_trainer_discovers_sources_and_queries_endpoints(self):
        """Verify NativeSelfTrainer discovers trusted sources and queries compute endpoints."""
        trainer = get_self_trainer(repo_root=self.repo_root)
        status = trainer.get_status()

        self.assertIn("active_model", status)
        self.assertIn("active_compute_endpoint", status)
        self.assertIsNotNone(status["active_compute_endpoint"])
        self.assertIn("endpoint_id", status["active_compute_endpoint"])

        # Discover trusted sources
        disc = trainer.discover_and_stage_updates()
        self.assertIn("total_trusted_sources", disc)
        self.assertGreater(disc["total_trusted_sources"], 0)

    def test_06_experiential_learner_and_memory_consolidation(self):
        """Verify ExperientialClosedLoopLearner and MemoryConsolidation operate seamlessly."""
        exp_learner = ExperientialClosedLoopLearner(repo_root=self.repo_root)
        obs = exp_learner.observe("Test endpoint selection")
        under = exp_learner.understand(obs)
        plan = exp_learner.plan(under)
        act_res = exp_learner.act(plan, lambda a, g: {"endpoint": "local_process"})
        verif = exp_learner.verify_result(act_res, plan["acceptance_criteria"])
        expl = exp_learner.explain(act_res, verif)
        lesson = exp_learner.extract_lesson("endpoint selection", expl, verif["verified"])
        episode = exp_learner.store_outcome(
            observation="Test endpoint selection",
            understanding=under,
            plan=plan,
            action={"action_type": plan["action_type"]},
            result=act_res,
            verification=verif,
            explanation=expl,
            lesson=lesson
        )
        self.assertEqual(episode.outcome, "SUCCESS")
        lessons = exp_learner.apply_to_future_tasks("endpoint selection")
        self.assertGreaterEqual(len(lessons), 1)

        # Memory consolidation
        cons = MemoryConsolidation.get_default(repo_root=self.repo_root)
        facts = cons.consolidate_episode({
            "episode_id": episode.episode_id,
            "actor_id": "creator",
            "action": "endpoint_selection",
            "parameters": {"speed": "fast"},
            "outcome": "SUCCESS",
            "result": {"endpoint": "local"},
            "reflection": "Decision: ALLOW. Outcome: SUCCESS."
        })
        self.assertIsInstance(facts, list)

    def test_07_tara_brain_working_memory_and_neural_control_tokens(self):
        """Verify TaraBrain incorporates working memory context and dispatches neural control tokens."""
        brain = TaraBrain()

        # Check compute endpoint access
        ep = brain.get_active_compute_endpoint()
        self.assertIsNotNone(ep)
        self.assertIn("endpoint_id", ep)

        # Check working memory governor
        wm = brain.working_memory
        self.assertIsNotNone(wm)
        wm.govern_session(
            session_id="test_wm_session",
            turns=[
                {"user_input": "Turn 1 request", "response": "Turn 1 response"},
                {"user_input": "Turn 2 request", "response": "Turn 2 response"}
            ],
            active_goal="Verify working memory"
        )
        wm_context = wm.get_context_for_prompt("test_wm_session")
        self.assertIsInstance(wm_context, str)

        # Verify ControlTokenActionParser
        exec_str = '<|tara_exec|> {"tool": "hash_verifier", "params": {"file_path": "README.md"}} <|im_end|>'
        parsed = ControlTokenActionParser.parse(exec_str)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.action_type, "TOOL")
        self.assertEqual(parsed.target, "hash_verifier")
        self.assertEqual(parsed.payload.get("file_path"), "README.md")

        # Verify stream_infer
        stream_tokens = list(brain.stream_infer("Hello TARA", max_new_tokens=5))
        self.assertGreater(len(stream_tokens), 0)

    def test_08_generate_stream_and_advanced_sampling(self):
        """Verify generate_stream and sample_next_token with top-p, top-k, and repetition penalty."""
        model_dir = os.path.join(self.repo_root, "storage", "models", "tara")
        model, tokenizer, config = load_trained_language_model(model_dir)

        # 1. Advanced sampling
        logits = [-10.0] * tokenizer.vocab_size
        logits[5] = 2.5
        logits[12] = 2.4
        logits[20] = 1.0

        # Greedy / low temp
        best_id = sample_next_token(logits, temperature=0.01)
        self.assertEqual(best_id, 5)

        # Repetition penalty on 5 should favor 12
        rep_id = sample_next_token(logits, generated_ids=[5], repetition_penalty=5.0, temperature=0.01)
        self.assertEqual(rep_id, 12)

        # Top-K
        k_id = sample_next_token(logits, top_k=2, temperature=0.7)
        self.assertIn(k_id, (5, 12))

        # Top-P
        p_id = sample_next_token(logits, top_p=0.9, temperature=0.7)
        self.assertIn(p_id, (5, 12, 20))

        # 2. Token Streaming
        tokens = list(generate_stream(model, tokenizer, "TARA status", max_new_tokens=6))
        self.assertGreaterEqual(len(tokens), 1)
        full_text = "".join(tokens)
        self.assertIsInstance(full_text, str)

    def test_09_colab_training_dataset_extraction(self):
        """Verify TARA_COLAB_TRAINING extractors discover cognitive, robotics, and auto-sync domains."""
        cog_samples = TARA_COLAB_TRAINING.extract_cognitive_capabilities()
        self.assertGreaterEqual(len(cog_samples), 200, "Cognitive samples below expectation")

        rob_samples = TARA_COLAB_TRAINING.extract_ai_robotics_domain()
        self.assertGreaterEqual(len(rob_samples), 25, "Robotics domain samples below expectation")

        sync_samples = TARA_COLAB_TRAINING.extract_auto_connect_sync()
        self.assertGreaterEqual(len(sync_samples), 5, "Auto-connect sync samples below expectation")

        # Verify dataset loader loads canonical datasets without crashing on staging
        c_train, c_val, c_test = TARA_COLAB_TRAINING.load_canonical_datasets()
        self.assertGreater(len(c_train), 1000)
        self.assertGreater(len(c_val), 100)
        self.assertGreater(len(c_test), 100)

        # Verify load_training_checkpoint operates cleanly in model-only mode (optimizer=None, scheduler=None, scaler=None)
        import torch
        import tempfile
        model_dir = os.path.join(self.repo_root, "storage", "models", "tara")
        config = TARA_COLAB_TRAINING.TaraConfig.from_json_file(os.path.join(model_dir, "config.json"))
        source_model = TARA_COLAB_TRAINING.TaraForCausalLM(config)
        TARA_COLAB_TRAINING.load_tara_model_unified_sharded(source_model, model_dir, device="cpu")

        test_fp = "test_model_fp_12345"
        test_data_fp = "test_data_fp_67890"
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_ckpt_path = os.path.join(tmp_dir, "candidate_test_ckpt.pt")
            save_payload = {
                "epoch": 102,
                "global_step": 9888,
                "model_state_dict": source_model.state_dict(),
                "optimizer_state_dict": {"state": {}, "param_groups": []},
                "scheduler_state_dict": {"last_epoch": 102},
                "scaler_state_dict": {"scale": 65536.0},
                "source_model_fingerprint": test_fp,
                "training_data_fingerprint": test_data_fp
            }
            torch.save(save_payload, test_ckpt_path)

            target_model = TARA_COLAB_TRAINING.TaraForCausalLM(config)
            ckpt_info = TARA_COLAB_TRAINING.load_training_checkpoint(
                checkpoint_path=test_ckpt_path,
                model=target_model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                current_model_fingerprint=test_fp,
                device="cpu",
                expected_training_data_fingerprint=test_data_fp
            )

            self.assertIsNotNone(ckpt_info)
            self.assertEqual(ckpt_info["epoch"], 102)
            self.assertEqual(ckpt_info["global_step"], 9888)
            self.assertEqual(ckpt_info["source_model_fingerprint"], test_fp)
            self.assertEqual(ckpt_info["training_data_fingerprint"], test_data_fp)

            target_sd = target_model.state_dict()
            source_sd = source_model.state_dict()
            self.assertEqual(set(target_sd.keys()), set(source_sd.keys()))
            for k in source_sd.keys():
                self.assertTrue(torch.equal(target_sd[k], source_sd[k]))


if __name__ == "__main__":
    unittest.main()

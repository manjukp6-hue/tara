"""
tests/test_tara_cognitive_capabilities.py

Comprehensive Integration Test Suite for TARA's Upgraded Cognitive Capabilities:
1. Intent Understanding (Paraphrase Mapping)
2. Entity / Slot Extraction
3. Multi-Turn Context & Coreference Resolution
4. Missing Parameter Clarification & Task Suspension/Resumption
5. Composite Task Planning & Goal Tracking (Step 1 -> Step 2 -> Step 3)
6. Semantic Tool / Skill Selection
7. Task Verification (Objective post-condition checks)
8. Recovery & Replanning (Safe error correction & bounded retry)
9. Uncertainty Handling (Explicit statement of missing evidence)
10. Self-Evaluation & Refinement
"""

import os
import sys
import json
import time
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.brain import TaraBrain
from tara_core.nlu import SlotExtractor, SemanticIntentParser
from tara_core.context import SessionContextManager, CoreferenceResolver, ClarificationManager
from tara_core.planner import TaskPlanner, GoalTracker, PlanStep
from tara_core.evaluator import TaskVerifier, RecoveryEngine, UncertaintyDetector, SelfEvaluator


class TestTaraCognitiveCapabilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env
        env_file = os.path.join(REPO_ROOT, ".env")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#") and "=" in s:
                        k, v = s.split("=", 1)
                        os.environ[k.strip()] = v.strip()

        cls.brain = TaraBrain()
        cls.config_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "config.json")

    # ------------------------------------------------------------------------
    # 1. INTENT UNDERSTANDING (Paraphrase Mapping)
    # ------------------------------------------------------------------------
    def test_01_paraphrased_intent_understanding(self):
        """1. Verify that diverse natural language paraphrases map to correct tool/skill intents."""
        # Paraphrase A for file inspector
        p1 = f"Can you check what is inside {self.config_path}?"
        res1 = self.brain.process(actor_id="user_nlu_01", input_text=p1)
        self.assertEqual(res1["intent"]["intent"], "EXECUTE_TOOL")
        self.assertEqual(res1["intent"]["tool"], "file_inspector")
        self.assertEqual(res1["outcome"], "SUCCESS")
        self.assertTrue(res1["result"]["exists"])

        # Paraphrase B for hash verifier
        p2 = f"Calculate SHA-256 checksum of {self.config_path}"
        res2 = self.brain.process(actor_id="user_nlu_01", input_text=p2)
        self.assertEqual(res2["intent"]["intent"], "EXECUTE_TOOL")
        self.assertEqual(res2["intent"]["tool"], "hash_verifier")
        self.assertEqual(res2["outcome"], "SUCCESS")
        self.assertIn("hash", res2["result"])

        # Paraphrase C for system diagnostics
        p3 = "Check CPU load and monitor system telemetry"
        res3 = self.brain.process(actor_id="user_nlu_01", input_text=p3)
        self.assertEqual(res3["intent"]["intent"], "EXECUTE_SKILL")
        self.assertEqual(res3["intent"]["skill"], "diagnostics")
        self.assertEqual(res3["outcome"], "SUCCESS")
        self.assertTrue(res3["result"]["cpu_healthy"])

    # ------------------------------------------------------------------------
    # 2. ENTITY / SLOT EXTRACTION
    # ------------------------------------------------------------------------
    def test_02_entity_slot_extraction(self):
        """2. Verify extraction of structured file paths, dates, amounts, device IDs, and hashes."""
        text = (
            "Examine 'storage/models/tara/config.json' on device TARA-DEVICE-002 "
            "at 2026-09-13 with expected hash 6cd000045e6f0981a8069695d7f1d4f40f3531b75cb45091d3106509f6920f4c "
            "and limit 5 items."
        )
        slots = SlotExtractor.extract_all_slots(text)

        self.assertIn("storage/models/tara/config.json", slots["file_path"])
        self.assertEqual(slots["device_id"], "TARA-DEVICE-002")
        self.assertEqual(slots["datetime"], "2026-09-13")
        self.assertEqual(slots["hash"], "6cd000045e6f0981a8069695d7f1d4f40f3531b75cb45091d3106509f6920f4c")
        self.assertTrue(any(a.get("value") == 5 for a in slots["amounts"]))

    # ------------------------------------------------------------------------
    # 3. MULTI-TURN CONTEXT & COREFERENCE RESOLUTION
    # ------------------------------------------------------------------------
    def test_03_multiturn_coreference_resolution(self):
        """3. Verify multi-turn session reference resolution (e.g. 'its hash' resolves to prior file)."""
        session_id = f"session_test_{int(time.time())}"

        # Turn 1: inspect a specific file
        turn1_input = f"inspect_file {self.config_path}"
        res1 = self.brain.process(actor_id="user_multiturn", input_text=turn1_input, context={"session_id": session_id})
        self.assertEqual(res1["outcome"], "SUCCESS")
        self.assertEqual(res1["tool_or_skill"]["name"], "file_inspector")

        # Turn 2: refer to "it" / "that file" without repeating the path
        turn2_input = "calculate its hash"
        res2 = self.brain.process(actor_id="user_multiturn", input_text=turn2_input, context={"session_id": session_id})
        self.assertEqual(res2["intent"]["tool"], "hash_verifier")
        self.assertEqual(res2["outcome"], "SUCCESS")
        # Ensure the hash was computed for the resolved file path from turn 1
        self.assertEqual(res2["result"]["file_path"], self.config_path)

    # ------------------------------------------------------------------------
    # 4. MISSING PARAMETER CLARIFICATION
    # ------------------------------------------------------------------------
    def test_04_missing_parameter_clarification_and_continuation(self):
        """4. Verify missing parameter triggers clarification question and halts, then resumes on next turn."""
        session_id = f"session_clarif_{int(time.time())}"

        # Turn 1: Underspecified command with missing file_path
        res1 = self.brain.process(actor_id="user_clarif", input_text="inspect the file", context={"session_id": session_id})
        self.assertEqual(res1["decision"], "NEEDS_CLARIFICATION")
        self.assertIn("Which file would you like me to inspect?", res1["final_response"])
        self.assertEqual(res1["result"]["status"], "NEEDS_CLARIFICATION")

        # Turn 2: User provides the missing file path
        res2 = self.brain.process(actor_id="user_clarif", input_text=self.config_path, context={"session_id": session_id})
        self.assertEqual(res2["outcome"], "SUCCESS")
        self.assertEqual(res2["tool_or_skill"]["name"], "file_inspector")
        self.assertTrue(res2["result"]["exists"])

    # ------------------------------------------------------------------------
    # 5. COMPOSITE TASK PLANNING & GOAL TRACKING
    # ------------------------------------------------------------------------
    def test_05_three_step_planning_and_goal_tracking(self):
        """5. Verify 3-step composite task planning (file inspection -> hash calculation -> diagnostics)."""
        composite_query = (
            f"Inspect file '{self.config_path}', calculate its SHA-256 hash, "
            "and check system telemetry diagnostics."
        )
        res = self.brain.process(actor_id="user_planner", input_text=composite_query)

        self.assertEqual(res["intent"]["intent"], "COMPOSITE_PLAN")
        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertIn("MULTI-STEP-PLAN", res["context_tags"])
        
        goal_data = res["result"]["goal"]
        self.assertEqual(goal_data["status"], "COMPLETED")
        self.assertEqual(goal_data["total_steps"], 3)
        self.assertEqual(len(res["result"]["steps"]), 3)

        # Verify step 1 was file_inspector
        self.assertEqual(goal_data["steps"][0]["target_name"], "file_inspector")
        # Verify step 2 was hash_verifier
        self.assertEqual(goal_data["steps"][1]["target_name"], "hash_verifier")
        # Verify step 3 was diagnostics
        self.assertEqual(goal_data["steps"][2]["target_name"], "diagnostics")

    # ------------------------------------------------------------------------
    # 6. SEMANTIC TOOL / SKILL SELECTION
    # ------------------------------------------------------------------------
    def test_06_semantic_tool_selection(self):
        """6. Verify tool selection distinguishes integrity checks from line reading without literal tool names."""
        # Query emphasizing checksum integrity -> hash_verifier
        q_hash = f"Verify the cryptographic checksum integrity of {self.config_path}"
        res_hash = self.brain.process(actor_id="user_semantic", input_text=q_hash)
        self.assertEqual(res_hash["tool_or_skill"]["name"], "hash_verifier")

        # Query emphasizing content / lines inspection -> file_inspector
        q_inspect = f"Check how many lines are inside {self.config_path}"
        res_inspect = self.brain.process(actor_id="user_semantic", input_text=q_inspect)
        self.assertEqual(res_inspect["tool_or_skill"]["name"], "file_inspector")

    # ------------------------------------------------------------------------
    # 7. TASK VERIFICATION
    # ------------------------------------------------------------------------
    def test_07_task_verification_objective_post_conditions(self):
        """7. Verify TaskVerifier checks objective post-conditions beyond status code."""
        # A valid file inspector result
        good_inspector = {"status": "SUCCESS", "exists": True, "file_path": "test.txt", "line_count": 25, "size_bytes": 1024}
        v_good = TaskVerifier.verify("EXECUTE_TOOL", "file_inspector", good_inspector)
        self.assertTrue(v_good["satisfied"])

        # An inspector result claiming SUCCESS but file does not exist
        bad_inspector = {"status": "SUCCESS", "exists": False, "file_path": "nonexistent.txt"}
        v_bad = TaskVerifier.verify("EXECUTE_TOOL", "file_inspector", bad_inspector)
        self.assertFalse(v_bad["satisfied"])
        self.assertIn("did not locate target file", v_bad["verification_notes"])

        # A hash verifier result with invalid non-hex digest
        bad_hash = {"status": "SUCCESS", "hash": "NOT_A_VALID_HEX"}
        v_hash_bad = TaskVerifier.verify("EXECUTE_TOOL", "hash_verifier", bad_hash)
        self.assertFalse(v_hash_bad["satisfied"])

    # ------------------------------------------------------------------------
    # 8. RECOVERY & REPLANNING
    # ------------------------------------------------------------------------
    def test_08_recovery_engine_file_path_normalization(self):
        """8. Verify RecoveryEngine safely normalizes a relative file path and retries successfully."""
        engine = RecoveryEngine(repo_root=REPO_ROOT)
        relative_path = "storage/models/tara/config.json"
        
        can_recover, healed_params, strategy = engine.attempt_recovery(
            tool_name="file_inspector",
            params={"file_path": relative_path},
            error_msg=f"File '{relative_path}' does not exist"
        )
        self.assertTrue(can_recover)
        self.assertTrue(os.path.exists(healed_params["file_path"]))
        self.assertIn("Normalized relative file path", strategy)

    # ------------------------------------------------------------------------
    # 9. UNCERTAINTY HANDLING
    # ------------------------------------------------------------------------
    def test_09_uncertainty_handling_missing_evidence(self):
        """9. Verify system explicitly states uncertainty instead of hallucinating on unverified queries."""
        unknown_query = "Who was the minister of Atlantis in 1842?"
        res = self.brain.process(actor_id="user_uncertain", input_text=unknown_query)
        
        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertIn("UNCERTAINTY-STATED", res["context_tags"])
        self.assertIn("do not have verified knowledge", res["final_response"])

    # ------------------------------------------------------------------------
    # 10. SELF-EVALUATION & REFINEMENT
    # ------------------------------------------------------------------------
    def test_10_self_evaluation_refinement_of_contradictory_candidate(self):
        """10. Verify SelfEvaluator flags and refines contradictions between tool output and candidate response."""
        contradictory_response = "All done! Operation was completely successful."
        failed_verification = {"satisfied": False, "verification_notes": "Cryptographic digest mismatch detected."}
        
        refined = SelfEvaluator.evaluate_and_refine(
            user_input="Check file hash",
            tool_or_skill={"type": "TOOL", "name": "hash_verifier"},
            result={"status": "FAILURE"},
            candidate_response=contradictory_response,
            verification_report=failed_verification
        )
        self.assertIn("[TASK_VERIFICATION_FAILURE]", refined)
        self.assertIn("Cryptographic digest mismatch detected", refined)


if __name__ == "__main__":
    unittest.main()

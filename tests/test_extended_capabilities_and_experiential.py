"""
tests/test_extended_capabilities_and_experiential.py

Comprehensive tests for:
1. Extended Cognitive Capabilities (22 - 46)
2. Experiential Closed-Loop Learning
3. Integration with CapabilityRegistry and TaraBrain
"""

import os
import sys
import unittest
import tempfile
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.extended_capabilities import (
    ModalityType,
    MultimodalPerceptionEngine,
    ToolLearningCreationEngine,
    ActionPlanningExecutionMonitor,
    PlanAction,
    WorldStateModel,
    SpatialReasoningEngine,
    EventCausalMemory,
    PredictiveReasoningEngine,
    DecisionTradeoffEngine,
    AttentionPriorityManager,
    CuriosityKnowledgeGapDetector,
    SourceCredibilityReasoning,
    FactVerificationCrossValidator,
    HallucinationPreventionGrounding,
    PersonalizationEngine,
    ConversationalSocialState,
    MetaReasoningEngine,
    SelfImprovementEngine,
    ExperimentationABEngine,
    DependencyCompatibilityManager,
    ConfigVersionGovernance,
    DistributedMultiDeviceCoordinator,
    HumanInTheLoopEscalator,
    DigitalTwinSimulationReasoning,
    ResourceAwareIntelligence,
    LongHorizonPlanningEngine,
    Milestone,
    ExtendedCapabilitiesHub,
    get_extended_capabilities_hub
)
from tara_core.experiential_learning import (
    ExperientialClosedLoopLearner,
    ClosedLoopStage,
    ExperienceEpisode,
    get_experiential_learner
)
from tara_core.registry import CapabilityRegistry
from tara_core.cognitive_capabilities import get_cognitive_capabilities_hub, register_cognitive_capabilities
from tara_core.brain import TaraBrain


class TestExtendedCapabilities(unittest.TestCase):
    """Verifies all 25 extended capabilities (22 through 46)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_ext_test_")
        self.hub = ExtendedCapabilitiesHub()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_22_multimodal_perception(self):
        engine = self.hub.multimodal
        sample = engine.process_perception(
            ModalityType.AUDIO,
            content_uri="sensor://mic0",
            metadata={"sample_rate_hz": 16000, "duration_seconds": 2.5}
        )
        self.assertEqual(sample.modality, ModalityType.AUDIO)
        self.assertIn("speech_detected", sample.extracted_features)

    def test_23_tool_learning_creation(self):
        engine = self.hub.tool_learning
        spec = engine.parse_and_validate_spec({
            "name": "custom_calc",
            "description": "Sums two integers",
            "parameters": {"a": "int", "b": "int"},
            "returns": "int"
        })
        self.assertEqual(spec.tool_name, "custom_calc")
        eval_res = engine.test_in_sandbox(spec, {"a": 1, "b": 2})
        self.assertTrue(eval_res["passed"])

    def test_24_action_planning_monitor(self):
        engine = self.hub.action_planning
        actions = [
            PlanAction(action_id="step_1", name="fetch_data", expected_duration_s=1.0),
            PlanAction(action_id="step_2", name="process_data", expected_duration_s=2.0)
        ]
        engine.create_plan("plan_1", actions)
        mon = engine.monitor_step(
            plan_id="plan_1",
            action_id="step_1",
            actual_duration_s=0.5,
            reported_status="COMPLETED"
        )
        self.assertFalse(mon["deviation_detected"])

    def test_25_world_state_model(self):
        engine = self.hub.world_model
        engine.register_entity("robot_arm_1", "DEVICE", "Arm 1", {"position": [0.5, 0.2, 0.8], "gripper": "closed"})
        entities = engine.query_state("DEVICE")
        self.assertGreaterEqual(len(entities), 1)
        self.assertEqual(entities[0]["properties"]["gripper"], "closed")

    def test_26_spatial_reasoning(self):
        engine = self.hub.spatial_reasoning
        dist = engine.euclidean_distance_3d((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        self.assertEqual(dist, 5.0)
        env = engine.verify_workspace_envelope((0.5, 0.5, 1.0))
        self.assertTrue(env["within_bounds"])
        self.assertEqual(env["safety_verdict"], "SAFE")

    def test_27_event_causal_memory(self):
        engine = self.hub.event_causal_memory
        rec = engine.record_event(
            what="power_surge",
            why="grid instability",
            action="isolate_circuit",
            result="safe",
            success=True,
            lesson="Install surge suppressor"
        )
        self.assertIsNotNone(rec.event_id)
        lessons = engine.query_lessons("surge")
        self.assertIn("Install surge suppressor", lessons)

    def test_28_predictive_reasoning(self):
        engine = self.hub.predictive_reasoning
        pred = engine.predict_action_outcome("train_model", params={}, historical_success_rate=0.95)
        self.assertEqual(pred["estimated_resource_cost"], "HIGH")
        self.assertTrue(pred["likely_success"])

    def test_29_decision_tradeoff(self):
        engine = self.hub.decision_tradeoff
        options = [
            {"name": "deep_reasoning", "safety": 1.0, "quality": 0.95, "cost": 0.2, "latency": 0.3},
            {"name": "fast_heuristic", "safety": 0.7, "quality": 0.60, "cost": 0.1, "latency": 0.1}
        ]
        best = engine.evaluate_tradeoffs(options)
        self.assertEqual(best["best_option"], "deep_reasoning")

    def test_30_attention_priority(self):
        engine = self.hub.attention_priority
        tasks = [
            {"id": "t1", "urgency": 2, "importance": 5},
            {"id": "t2", "urgency": 9, "importance": 9},
            {"id": "t3", "urgency": 6, "importance": 4}
        ]
        ranked = engine.prioritize_tasks(tasks)
        self.assertEqual(ranked[0]["id"], "t2")

    def test_31_curiosity_knowledge_gap(self):
        engine = self.hub.curiosity
        res = engine.inspect_query_knowledge("topological_qubits", [])
        self.assertTrue(res["knowledge_gap_detected"])
        self.assertEqual(res["total_recorded_gaps"], 1)

    def test_32_source_credibility(self):
        engine = self.hub.source_credibility
        s1 = engine.evaluate_source("creator")
        s2 = engine.evaluate_source("unverified_web")
        self.assertGreater(s1["credibility_score"], s2["credibility_score"])
        self.assertTrue(s1["is_trusted"])
        self.assertTrue(s2["quarantine_required"])

    def test_33_fact_verification(self):
        engine = self.hub.fact_verification
        res = engine.cross_validate(
            claim="Speed of light is invariant",
            evidence_sources=[{"source": "textbook", "supports": True}, {"source": "paper", "supports": True}]
        )
        self.assertTrue(res["verified"])
        self.assertEqual(res["consensus"], "SUPPORTED")

    def test_34_hallucination_prevention(self):
        engine = self.hub.hallucination_grounding
        res = engine.ground_response(
            assertion="Operating limits: max temperature is 45C.",
            verified_evidence=["Operating limits: max temperature is 45C."]
        )
        self.assertTrue(res["is_grounded"])
        self.assertEqual(res["epistemic_status"], "GROUNDED_VERIFIED")

    def test_35_personalization(self):
        engine = self.hub.personalization
        engine.set_user_preference(user_id="user1", key="verbosity", value="concise")
        val = engine.get_user_preference(user_id="user1", key="verbosity")
        self.assertEqual(val, "concise")

    def test_36_conversational_social_state(self):
        engine = self.hub.conversational_state
        engine.update_turn(session_id="s1", intent="SEARCH", user_goal="find file", needs_clarification=False)
        state = engine.get_session_state("s1")
        self.assertEqual(state["user_goal"], "find file")

    def test_37_meta_reasoning(self):
        engine = self.hub.meta_reasoning
        strat_simple = engine.select_strategy(task_complexity=1, confidence=0.95, requires_tools=False)
        self.assertEqual(strat_simple["strategy"], "DIRECT_COGNITIVE_RESPONSE")
        strat_complex = engine.select_strategy(task_complexity=8, confidence=0.9, requires_tools=True)
        self.assertEqual(strat_complex["strategy"], "DECOMPOSE_GOAL_DAG")

    def test_38_self_improvement(self):
        engine = self.hub.self_improvement
        rec = engine.record_failure_pattern("timeout_in_large_payload")
        self.assertEqual(rec["occurrence_count"], 1)

    def test_39_experimentation_ab(self):
        engine = self.hub.experimentation
        res = engine.compare_variants(
            variant_a={"name": "candidate_a", "val_loss": 1.25},
            variant_b={"name": "candidate_b", "val_loss": 1.45},
            metric_key="val_loss"
        )
        self.assertEqual(res["winner"], "candidate_a")

    def test_40_dependency_compatibility(self):
        engine = self.hub.dependency_manager
        res = engine.check_compatibility(
            required_dependencies=["numpy", "torch"],
            available_features={"numpy", "torch", "safetensors"}
        )
        self.assertTrue(res["compatible"])

    def test_41_config_governance(self):
        engine = self.hub.config_governance
        v1 = engine.snapshot_config("v1.0", {"max_tokens": 100, "temperature": 0.7})
        restored = engine.restore_config(v1)
        self.assertEqual(restored["max_tokens"], 100)

    def test_42_distributed_multi_device(self):
        engine = self.hub.multi_device
        node = engine.register_node("edge_pi_1", "ROBOT", ["temperature", "lidar"])
        nodes = engine.list_nodes()
        self.assertEqual(nodes[0]["id"], "edge_pi_1")

    def test_43_human_in_the_loop(self):
        engine = self.hub.human_in_the_loop
        eval_low = engine.evaluate_escalation(risk_level="LOW", confidence=0.9, is_critical=False)
        self.assertFalse(eval_low["escalation_required"])
        eval_high = engine.evaluate_escalation(risk_level="HIGH", confidence=0.5, is_critical=True)
        self.assertTrue(eval_high["escalation_required"])

    def test_44_digital_twin_simulation(self):
        engine = self.hub.digital_twin
        traj_check = engine.validate_trajectory([(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
        self.assertTrue(traj_check["trajectory_valid"])

    def test_45_resource_aware_intelligence(self):
        engine = self.hub.resource_aware
        strat_high = engine.compute_strategy_budget(cpu_percent=20.0, ram_available_mb=4096.0)
        self.assertEqual(strat_high["mode"], "PERFORMANCE")
        strat_low = engine.compute_strategy_budget(cpu_percent=95.0, ram_available_mb=256.0)
        self.assertEqual(strat_low["mode"], "LOW_RESOURCE")

    def test_46_long_horizon_planning(self):
        engine = self.hub.long_horizon
        m1 = Milestone("m1", "Init database", "2026-10-01", completed=True)
        m2 = Milestone("m2", "Deploy model", "2026-10-05", completed=False)
        engine.set_project_milestones("proj_1", [m1, m2])
        progress = engine.get_progress("proj_1")
        self.assertEqual(progress["progress_pct"], 50.0)


class TestExperientialLearning(unittest.TestCase):
    """Verifies Experiential Closed-Loop Learning end-to-end."""

    def setUp(self):
        self.learner = ExperientialClosedLoopLearner()

    def test_full_closed_loop_execution(self):
        def dummy_action_executor(action_type: str, goal: str):
            return {
                "status": "SUCCESS",
                "tool_used": "hash_verifier",
                "verified": True,
                "output": "SHA-256 verified successfully"
            }

        result = self.learner.run_full_closed_loop(
            input_text="Verify system integrity using cryptographic hash verification",
            action_executor=dummy_action_executor
        )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("episode_id", result)
        self.assertEqual(result["verification_result"]["verified"], True)
        self.assertIn("lesson", result)
        self.assertIn("strategy_adapted", result)

        # Check lessons and episodes query
        relevant = self.learner.query_past_lessons("cryptographic hash verification")
        self.assertGreaterEqual(len(relevant), 1)

    def test_experiential_cycle_stages(self):
        expected_stages = [
            "OBSERVE", "UNDERSTAND", "PLAN", "ACT", "OBSERVE_RESULT",
            "VERIFY_RESULT", "EXPLAIN_OUTCOME", "STORE_OUTCOME",
            "EXTRACT_LESSON", "UPDATE_KNOWLEDGE_MEMORY",
            "IMPROVE_STRATEGY_SKILL", "APPLY_TO_FUTURE_TASKS"
        ]
        enum_names = [e.name for e in ClosedLoopStage]
        for stage in expected_stages:
            self.assertIn(stage, enum_names)


class TestCognitiveHubAndBrainIntegration(unittest.TestCase):
    """Verifies all 48 capabilities registered in CapabilityRegistry and accessible via TaraBrain."""

    def test_48_capabilities_in_manifest(self):
        hub = get_cognitive_capabilities_hub()
        manifest = hub.get_capabilities_manifest()
        self.assertGreaterEqual(len(manifest), 48)
        ids = [m["id"] for m in manifest]
        self.assertIn("long_term_reasoning", ids)
        self.assertIn("self_model", ids)
        self.assertIn("multimodal_perception", ids)
        self.assertIn("long_horizon_planning", ids)
        self.assertIn("experiential_learning", ids)
        self.assertIn("robotics_hal", ids)

    def test_registry_registration(self):
        reg = CapabilityRegistry()
        count = register_cognitive_capabilities(registry=reg)
        self.assertGreaterEqual(count, 48)
        self.assertIsNotNone(reg.get_capability("multimodal_perception"))
        self.assertIsNotNone(reg.get_capability("experiential_learning"))
        self.assertIsNotNone(reg.get_capability("robotics_hal"))

    def test_brain_runtime_access(self):
        brain = TaraBrain()
        self.assertIsNotNone(brain.cognitive_capabilities)
        self.assertIsNotNone(brain.extended_capabilities)
        self.assertIsNotNone(brain.experiential_learner)
        self.assertIsNotNone(brain.event_bus)
        self.assertIsNotNone(brain.robotics_hal)
        self.assertIsNotNone(brain.working_memory)
        self.assertIsNotNone(brain.experiential_bridge)
        self.assertIsNotNone(brain.capability_registry.get_capability("experiential_learning"))
        self.assertIsNotNone(brain.capability_registry.get_capability("robotics_hal"))

        # Test experiential cycle via brain
        exp_res = brain.run_experiential_cycle("Inspect file provenance and record experience")
        self.assertEqual(exp_res["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()

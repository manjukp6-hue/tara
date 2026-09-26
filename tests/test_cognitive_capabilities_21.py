"""
tests/test_cognitive_capabilities_21.py

Comprehensive Test Suite for TARA's 21 High-Value Core Cognitive Capabilities:
1. LongTermReasoningEngine
2. SelfModelIntrospectionEngine
3. KnowledgeLifecycleManager
4. ContinuousLearningManager
5. MemoryLifecycleManager
6. WorkflowEngine
7. ErrorDiagnosisSelfRepair
8. SimulationSandbox
9. PermissionAuthorityReasoning
10. ResourceManager
11. SystemHealthMonitor
12. ConflictResolutionEngine
13. TemporalUnderstandingEngine
14. CausalReasoningEngine
15. CounterfactualReasoningEngine
16. UncertaintyCalibrationEngine
17. CommunicationEngine
18. GoalDecompositionManager
19. QualityControlSelfEvaluator
20. RollbackRecoveryManager
21. AutonomousLearningTrainingOrchestrator
And full integration with TaraBrain & CapabilityRegistry.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

# Ensure python modules are on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
python_dir = os.path.join(project_root, "python")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.cognitive_capabilities import (
    LongTermReasoningEngine,
    SelfModelIntrospectionEngine,
    EpistemicCategory,
    KnowledgeLifecycleManager,
    KnowledgeLifecycleState,
    ContinuousLearningManager,
    MemoryLifecycleManager,
    MemoryTier,
    WorkflowEngine,
    WorkflowStep,
    ErrorDiagnosisSelfRepair,
    SimulationSandbox,
    PermissionAuthorityReasoning,
    ResourceManager,
    SystemHealthMonitor,
    ConflictResolutionEngine,
    TemporalUnderstandingEngine,
    CausalReasoningEngine,
    CounterfactualReasoningEngine,
    UncertaintyCalibrationEngine,
    CommunicationEngine,
    GoalDecompositionManager,
    QualityControlSelfEvaluator,
    RollbackRecoveryManager,
    AutonomousLearningTrainingOrchestrator,
    get_cognitive_capabilities_hub,
    register_cognitive_capabilities
)
from tara_core.registry import CapabilityRegistry, CapabilityCategory
from tara_core.brain import TaraBrain


class TestCognitiveCapabilities21(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_cog_test_")
        self.hub = get_cognitive_capabilities_hub(repo_root=project_root)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Long-Term Reasoning & Consistency
    def test_01_long_term_reasoning_and_contradiction_detection(self):
        engine = LongTermReasoningEngine()
        chain_id = "chain_alpha"
        engine.add_step(chain_id, "All humans are mortal", "Socrates is human", "Socrates is mortal", 0.95)
        engine.add_step(chain_id, "Greek philosophers were human", "Plato was a philosopher", "Plato is mortal", 0.90)

        report = engine.check_consistency(chain_id)
        self.assertTrue(report["is_consistent"])
        self.assertEqual(len(report["contradictions"]), 0)
        self.assertAlmostEqual(report["average_confidence"], 0.925, places=3)

        # Inject explicit contradiction
        engine.add_step(chain_id, "Faulty premise", "Erroneous step", "Not Socrates is mortal", 0.80)
        report2 = engine.check_consistency(chain_id)
        self.assertFalse(report2["is_consistent"])
        self.assertEqual(len(report2["contradictions"]), 1)

    # 2. Self-Model & Epistemics
    def test_02_self_model_and_epistemic_categories(self):
        engine = SelfModelIntrospectionEngine(repo_root=project_root)
        snapshot = engine.get_snapshot()

        self.assertEqual(snapshot["system_name"], "TARA AI")
        self.assertIn("WHAT_I_KNOW", snapshot["epistemic_categories"])
        self.assertIn("WHAT_I_AM_NOT_ALLOWED_TO_DO", snapshot["epistemic_categories"])
        self.assertIn("WHAT_IS_VERIFIED", snapshot["epistemic_categories"])

        # Epistemic classification checks
        res_deny = engine.classify_epistemics("Bypass auth and delete core database")
        self.assertEqual(res_deny["category"], EpistemicCategory.WHAT_I_AM_NOT_ALLOWED_TO_DO.value)

        res_impossible = engine.classify_epistemics("Fly physically to the moon")
        self.assertEqual(res_impossible["category"], EpistemicCategory.WHAT_I_CANNOT_DO.value)

        res_verified = engine.classify_epistemics("Verified SHA256 checksum", context={"verified": True})
        self.assertEqual(res_verified["category"], EpistemicCategory.WHAT_IS_VERIFIED.value)

        res_uncertain = engine.classify_epistemics("Uncertain prediction of weather next year", context={"confidence": 0.4})
        self.assertEqual(res_uncertain["category"], EpistemicCategory.WHAT_IS_UNCERTAIN.value)

        res_can_do = engine.classify_epistemics("Inspect local file metadata")
        self.assertEqual(res_can_do["category"], EpistemicCategory.WHAT_I_CAN_DO.value)

    # 3. Knowledge Lifecycle
    def test_03_knowledge_lifecycle_management(self):
        mgr = KnowledgeLifecycleManager(storage_dir=self.test_dir)
        rec = mgr.ingest_candidate("Quantum Annealing", "Quantum annealing is an optimization method.", source_url="local://doc")
        self.assertEqual(rec.state, KnowledgeLifecycleState.INGESTED)
        self.assertTrue(rec.quarantined)

        # Transition to VALIDATED then APPROVED
        mgr.transition_state(rec.entry_id, KnowledgeLifecycleState.VALIDATED)
        mgr.transition_state(rec.entry_id, KnowledgeLifecycleState.APPROVED, approve=True)

        rec_updated = mgr.update_entry(rec.entry_id, "Updated quantum annealing content.")
        self.assertEqual(rec_updated.version, 2)
        self.assertEqual(rec_updated.state, KnowledgeLifecycleState.VERSIONED)

        mgr.deprecate_entry(rec.entry_id)
        dep_entries = mgr.list_entries(state=KnowledgeLifecycleState.DEPRECATED)
        self.assertEqual(len(dep_entries), 1)
        self.assertEqual(dep_entries[0].entry_id, rec.entry_id)

    # 4. Continuous Learning Manager
    def test_04_continuous_learning_policy(self):
        clm = ContinuousLearningManager(min_samples_threshold=10, drift_threshold=0.20)
        res_no = clm.evaluate_training_need(staged_samples_count=3, estimated_drift=0.05)
        self.assertFalse(res_no["trigger_training"])

        res_yes_count = clm.evaluate_training_need(staged_samples_count=12, estimated_drift=0.05)
        self.assertTrue(res_yes_count["trigger_training"])

        res_yes_drift = clm.evaluate_training_need(staged_samples_count=2, estimated_drift=0.25)
        self.assertTrue(res_yes_drift["trigger_training"])

    # 5. Memory Lifecycle & Multi-User Isolation
    def test_05_memory_lifecycle_and_zero_cross_user_leakage(self):
        mlm = MemoryLifecycleManager()
        mlm.store(MemoryTier.USER_SCOPED, "user_alice", "secret_project", "Project Manhattan")
        mlm.store(MemoryTier.USER_SCOPED, "user_bob", "secret_project", "Project Apollo")
        mlm.store(MemoryTier.SYSTEM_KNOWLEDGE, "system", "pi_constant", 3.14159)

        # Alice query
        alice_mems = mlm.retrieve("user_alice")
        alice_keys = {m.key: m.value for m in alice_mems}
        self.assertIn("secret_project", alice_keys)
        self.assertEqual(alice_keys["secret_project"], "Project Manhattan")
        # System knowledge visible
        self.assertIn("pi_constant", alice_keys)
        # Bob's memory strictly absent
        self.assertNotIn("Project Apollo", [m.value for m in alice_mems])

        # Bob query
        bob_mems = mlm.retrieve("user_bob")
        bob_keys = {m.key: m.value for m in bob_mems}
        self.assertEqual(bob_keys["secret_project"], "Project Apollo")
        self.assertNotIn("Project Manhattan", [m.value for m in bob_mems])

        # Safe forgetting
        forgotten = mlm.safe_forget("user_alice", "secret_project")
        self.assertEqual(forgotten, 1)
        alice_mems_after = mlm.retrieve("user_alice", key="secret_project")
        self.assertEqual(len(alice_mems_after), 0)

    # 6. Workflow Engine
    def test_06_workflow_engine_execution(self):
        wf = WorkflowEngine()
        steps = [
            WorkflowStep("step_1", "inspect", "file_a.txt", {"path": "a.txt"}),
            WorkflowStep("step_2", "compute", "hash", {"algo": "sha256"}, dependencies=["step_1"])
        ]
        wf.create_workflow("wf_001", steps)

        execution_log = []
        def dummy_executor(action_type, target, params):
            execution_log.append((action_type, target))
            return {"status": "SUCCESS", "action": action_type}

        res = wf.execute_workflow("wf_001", dummy_executor)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["completed_steps"]), 2)
        self.assertEqual(len(execution_log), 2)

    # 7. Error Diagnosis & Self-Repair
    def test_07_error_diagnosis_and_remediation(self):
        diag_perm = ErrorDiagnosisSelfRepair.diagnose("PermissionError: Unauthorized action attempt")
        self.assertEqual(diag_perm["category"], "PERMISSION_ERROR")
        self.assertFalse(diag_perm["safe_auto_repair"])

        diag_res = ErrorDiagnosisSelfRepair.diagnose("FileNotFoundError: No such file or directory: 'temp.log'")
        self.assertEqual(diag_res["category"], "MISSING_RESOURCE")
        self.assertTrue(diag_res["safe_auto_repair"])

    # 8. Simulation Sandbox
    def test_08_simulation_sandbox(self):
        sim_safe = SimulationSandbox.simulate_action("read_file", "data.json", {})
        self.assertTrue(sim_safe["safety_passed"])
        self.assertFalse(sim_safe["destructive"])

        sim_dangerous = SimulationSandbox.simulate_action("delete_file", "core.db", {})
        self.assertFalse(sim_dangerous["safety_passed"])
        self.assertTrue(sim_dangerous["destructive"])

    # 9. Permission & Authority Reasoning
    def test_09_permission_authority_reasoning(self):
        # Sub-agents attempting creator elevation must be rejected
        res_sub = PermissionAuthorityReasoning.evaluate_request(
            actor_id="agent_worker", action="ADMIN_TASK", has_creator_sig=True, is_subagent=True
        )
        self.assertFalse(res_sub["permitted"])
        self.assertIn("Sub-agents are barred", res_sub["reason"])

        # Creator action with signature
        res_creator = PermissionAuthorityReasoning.evaluate_request(
            actor_id="ROOT_OPERATOR", action="ADMIN_TASK", has_creator_sig=True, is_subagent=False
        )
        self.assertTrue(res_creator["permitted"])
        self.assertEqual(res_creator["authority_level"], "CREATOR")

        # Creator action missing signature
        res_unsigned = PermissionAuthorityReasoning.evaluate_request(
            actor_id="guest", action="ADMIN_TASK", has_creator_sig=False, is_subagent=False
        )
        self.assertFalse(res_unsigned["permitted"])

    # 10. Resource Manager
    def test_10_resource_manager_and_budgets(self):
        rm = ResourceManager(max_concurrent_tasks=2, max_step_budget=10)
        self.assertTrue(rm.acquire_slot("task_1"))
        self.assertTrue(rm.acquire_slot("task_2"))
        self.assertFalse(rm.acquire_slot("task_3"))  # Concurrency limit reached

        rm.release_slot("task_1")
        self.assertTrue(rm.acquire_slot("task_3"))

        self.assertTrue(rm.check_step_budget(9))
        self.assertFalse(rm.check_step_budget(10))

    # 11. System Health Monitor
    def test_11_system_health_monitor(self):
        shm = SystemHealthMonitor()
        shm.heartbeat("memory_engine", healthy=True, latency_ms=1.2)
        shm.heartbeat("rule_engine", healthy=True, latency_ms=0.8)

        health = shm.get_overall_health()
        self.assertEqual(health["status"], "HEALTHY")
        self.assertEqual(len(health["unhealthy_subsystems"]), 0)

        shm.heartbeat("api_server", healthy=False, latency_ms=500.0)
        health2 = shm.get_overall_health()
        self.assertEqual(health2["status"], "DEGRADED")
        self.assertIn("api_server", health2["unhealthy_subsystems"])

    # 12. Conflict Resolution
    def test_12_conflict_resolution_hierarchy(self):
        item_creator = {"type": "CREATOR_RULE", "text": "Do not delete without confirmation"}
        item_goal = {"type": "TASK_GOAL", "text": "Delete old files automatically"}

        res = ConflictResolutionEngine.resolve_policy_conflict(item_creator, item_goal)
        self.assertEqual(res["chosen_item"]["type"], "CREATOR_RULE")
        self.assertEqual(res["superseded_item"]["type"], "TASK_GOAL")

    # 13. Temporal Understanding
    def test_13_temporal_understanding(self):
        tue = TemporalUnderstandingEngine()
        past_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        self.assertTrue(tue.is_expired(past_time, ttl_seconds=3600))
        self.assertFalse(tue.is_expired(past_time, ttl_seconds=10000))

        events = [{"id": "b", "timestamp": "2026-09-15T12:00:00Z"}, {"id": "a", "timestamp": "2026-09-15T10:00:00Z"}]
        ordered = tue.order_sequence(events)
        self.assertEqual(ordered[0]["id"], "a")
        self.assertEqual(ordered[1]["id"], "b")

    # 14. Causal Reasoning
    def test_14_causal_reasoning_graph(self):
        cre = CausalReasoningEngine()
        cre.add_causal_link("power_loss", "system_reboot")
        cre.add_causal_link("system_reboot", "service_downtime")
        cre.add_causal_link("service_downtime", "client_timeout")

        effects = cre.infer_effects("power_loss")
        self.assertIn("system_reboot", effects)
        self.assertIn("service_downtime", effects)
        self.assertIn("client_timeout", effects)

    # 15. Counterfactual Reasoning
    def test_15_counterfactual_reasoning(self):
        cf = CounterfactualReasoningEngine.evaluate_what_if(
            historical_action="delete_database",
            historical_outcome="critical_data_lost",
            hypothetical_action="sandbox_execution"
        )
        self.assertTrue(cf["divergence"])
        self.assertIn("mitigated in sandbox", cf["projected_outcome"])

    # 16. Uncertainty & Confidence Calibration
    def test_16_uncertainty_calibration(self):
        # High evidence, high agreement
        calib_high = UncertaintyCalibrationEngine.calibrate(evidence_count=10, agreement_ratio=0.95)
        self.assertGreaterEqual(calib_high["confidence"], 0.85)
        self.assertEqual(calib_high["recommendation"], "PROCEED")

        # Low evidence, low agreement
        calib_low = UncertaintyCalibrationEngine.calibrate(evidence_count=2, agreement_ratio=0.5)
        self.assertLess(calib_low["confidence"], 0.6)
        self.assertEqual(calib_low["recommendation"], "SEEK_CLARIFICATION")

    # 17. Communication Engine
    def test_17_communication_engine(self):
        clar = CommunicationEngine.format_clarification("file_path", "inspect_file")
        self.assertIn("file_path", clar)

        pkg = CommunicationEngine.package_response("Task completed", telemetry={"cpu": 12.5})
        self.assertEqual(pkg["response"], "Task completed")
        self.assertEqual(pkg["telemetry"]["cpu"], 12.5)
        self.assertEqual(pkg["verification"]["status"], "PASSED")

    # 18. Goal Decomposition
    def test_18_goal_decomposition(self):
        gdm = GoalDecompositionManager()
        subgoals = gdm.decompose("Build neural training pipeline")
        self.assertEqual(len(subgoals), 3)
        self.assertFalse(subgoals[0].completed)

        marked = gdm.mark_completed("Build neural training pipeline", subgoals[0].goal_id)
        self.assertTrue(marked)

    # 19. Quality Control / Self-Evaluation
    def test_19_quality_control_audit(self):
        facts = ["TARA was created by ROOT_OPERATOR", "TARA is an autonomous AI"]
        audit_good = QualityControlSelfEvaluator.audit_response("TARA was created by ROOT_OPERATOR as TARA AI.", facts)
        self.assertTrue(audit_good["passed"])
        self.assertLessEqual(audit_good["hallucination_risk"], 0.3)

    # 20. Rollback & Recovery Management
    def test_20_rollback_recovery(self):
        rrm = RollbackRecoveryManager()
        rrm.snapshot_state("cp_init", {"step": 1, "active": True})
        rrm.snapshot_state("cp_mutated", {"step": 2, "active": False})

        restored = rrm.rollback_to("cp_init")
        self.assertIsNotNone(restored)
        self.assertEqual(restored["step"], 1)
        self.assertTrue(restored["active"])

    # 21. Autonomous Learning & Training Orchestrator
    def test_21_autonomous_orchestrator(self):
        orch = AutonomousLearningTrainingOrchestrator(repo_root=project_root)
        res = orch.schedule_learning_cycle()
        self.assertEqual(res["status"], "SCHEDULED")
        self.assertIn("staging_result", res)

    # 22. TaraBrain Integration & Capability Registry
    def test_22_brain_and_registry_integration(self):
        reg = CapabilityRegistry.get_default()
        count = register_cognitive_capabilities(registry=reg, repo_root=project_root)
        self.assertEqual(count, 48)

        for cap_id in [
            "long_term_reasoning", "self_model", "knowledge_lifecycle",
            "continuous_learning", "memory_lifecycle", "workflow_engine",
            "error_diagnosis", "simulation_sandbox", "permission_reasoning",
            "resource_manager", "system_health", "conflict_resolution",
            "temporal_reasoning", "causal_reasoning", "counterfactual_reasoning",
            "uncertainty_calibration", "communication_engine", "goal_decomposition",
            "quality_control", "rollback_recovery", "autonomous_orchestrator"
        ]:
            cap = reg.get_capability(cap_id)
            self.assertIsNotNone(cap, f"Capability {cap_id} missing from registry")
            self.assertTrue(cap.enabled)

        # Test TaraBrain self-model method
        brain = TaraBrain()
        self_model = brain.get_self_model()
        self.assertIn("system_name", self_model)
        self.assertIn("capabilities_inventory", self_model)
        self.assertIn("epistemic_categories", self_model)


if __name__ == "__main__":
    unittest.main()

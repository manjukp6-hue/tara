"""
tests/test_e2e_36_capabilities_flows.py

Comprehensive End-to-End Verification Test Suite for TARA AI:
Validates all 20 real runtime workflows (Flows A through T) through TaraBrain and core engines:
- Flow A: Structured Knowledge Ingestion & Ontology Linking
- Flow B: Multi-Factor Knowledge Retrieval & Prompt Injection
- Flow C: Multi-Objective Plan Optimization & Constraint Satisfaction
- Flow D: Uncertainty-Aware Planning & Dynamic Re-planning
- Flow E: Sensor Fusion & Real-Time Environment State Estimation
- Flow F: Safe Hardware Command Execution & Verification (Robot / CNC / LFAM / 3D Printer)
- Flow G: GUI Interaction & Screen Analysis
- Flow H: Software Engineering & AST Modification with Snapshot Rollback
- Flow I: Data Engineering & Normalization Pipeline
- Flow J: Experiment Design & Uplift Significance Calculation
- Flow K: Incident Learning & Prevention Lesson Extraction
- Flow L: Goal Persistence Across Restart & Subgoal Resumption
- Flow M: Dynamic Skill Acquisition from Spec & 5-Stage Certification
- Flow N: Demonstration Learning & Procedure Generalization
- Flow O: Capability Composition into Higher-Order Workflow
- Flow P: Continual Learning Catastrophic-Forgetting Gating & Rejection
- Flow Q: Curriculum Priority Calculation & Learning Item Ranking
- Flow R: Model Routing Based on Task Complexity & Security Tier
- Flow S: Decision Audit Trail with Secret Scrubbing
- Flow T: Full 10-Stage World Model + Prediction Error Learning Loop & Training Staging
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.brain import TaraBrain
from tara_core.knowledge_ontology import (
    KnowledgeOntologyEngine, RelationType, ProvenanceRecord, ProvenanceSourceType,
    MultiFactorKnowledgeRanker, DataGovernancePolicy
)
from tara_core.robotics_hal import (
    RoboticsHAL, DeviceType, DeviceInterfaceType, DeviceProfile
)
from tara_core.reasoning_engine_advanced import (
    ConstraintSolver, MultiObjectiveOptimizer, ProbabilisticReasoningEngine,
    UncertaintyAwarePlanner
)
from tara_core.environment_sensor_fusion import (
    DynamicEnvironmentModel, SensorFusionEngine, RealWorldActionGrounder
)
from tara_core.engineering_capabilities import (
    SoftwareEngineeringEngine, DataEngineeringEngine, ExperimentDesignEngine
)
from tara_core.skill_acquisition_certification import (
    SkillAcquisitionEngine, DemonstrationLearningEngine, CapabilityComposer,
    SkillCertifier, CertificationState
)
from tara_core.continual_learning_governor import (
    ContinualLearningGovernor, CurriculumPriorityEngine, LearningItem,
    ModelAdaptationLayer, ModelRouter
)
from tara_core.resilience_maintenance import (
    GoalPersistenceManager, IncidentLearningEngine, PolicyStrategyLearner,
    AuditExplainabilityEngine, CorruptedLearningRecovery, SelfMaintenanceEngine
)
from tara_core.prediction_error_loop import (
    PredictionErrorEngine, PredictionExpectation, PredictionErrorRecord
)


class TestE2E36CapabilitiesFlows(unittest.TestCase):
    """Verifies all 20 runtime flows (A through T) through TaraBrain and connected subsystems."""

    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Flow A: Structured Knowledge Ingestion & Ontology Linking
    def test_flow_a_ontology_linking(self):
        prov = ProvenanceRecord(
            provenance_id="prov_flow_a",
            source_type=ProvenanceSourceType.CREATOR_DIRECTIVE,
            source_uri="TARA/FLOW_A",
            author="ROOT_OPERATOR",
            trust_score=1.0
        )
        self.brain.ontology.register_concept(
            concept_id="adaptive_control",
            name="Adaptive Feedback Control",
            category="ROBOTICS",
            properties={"type": "feedback"},
            rules=["RULE-CTRL-01"],
            provenance=prov
        )
        self.brain.ontology.add_relation("adaptive_control", RelationType.DEPENDS_ON, "robotics_hardware")
        parents = self.brain.ontology.get_parents("adaptive_control")
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["target_id"], "robotics_hardware")

    # Flow B: Multi-Factor Knowledge Retrieval & Prompt Injection
    def test_flow_b_multi_factor_retrieval(self):
        docs = [
            {"name": "Kinematics V1", "relevance": 0.6, "recency": 0.4, "confidence": 0.8, "authority": 0.7, "utility": 0.5},
            {"name": "Kinematics V2 Autonomous", "relevance": 0.95, "recency": 0.98, "confidence": 0.99, "authority": 1.0, "utility": 0.95},
            {"name": "General Chat", "relevance": 0.2, "recency": 0.5, "confidence": 0.3, "authority": 0.3, "utility": 0.2}
        ]
        ranked = MultiFactorKnowledgeRanker.rank_candidates(docs)
        self.assertEqual(ranked[0]["name"], "Kinematics V2 Autonomous")
        self.assertGreater(ranked[0]["_composite_rank_score"], 0.90)

    # Flow C: Multi-Objective Plan Optimization & Constraint Satisfaction
    def test_flow_c_plan_optimization_and_constraints(self):
        solver = ConstraintSolver()
        solver.add_hard_constraint("speed_limit", lambda a: a.get("speed", 0) <= 2.0)
        solver.add_soft_constraint("energy_efficiency", lambda a: a.get("power_w", 0) <= 50, penalty=0.2)

        opt = MultiObjectiveOptimizer()
        plans = [
            {"id": "plan_fast", "speed": 1.8, "power_w": 45, "quality": 0.95, "reliability": 0.95, "time_s": 10.0, "cost": 15.0, "risk": 0.05},
            {"id": "plan_violates", "speed": 2.5, "power_w": 60, "quality": 0.80, "reliability": 0.80, "time_s": 5.0, "cost": 30.0, "risk": 0.30},
            {"id": "plan_slow", "speed": 0.8, "power_w": 20, "quality": 0.85, "reliability": 0.90, "time_s": 35.0, "cost": 10.0, "risk": 0.10}
        ]

        feasible_plans = []
        for p in plans:
            feas = solver.check_feasibility(p)
            if feas["feasible"]:
                feasible_plans.append(p)

        self.assertEqual(len(feasible_plans), 2)
        ranked = opt.rank_candidates(feasible_plans)
        self.assertEqual(ranked[0]["id"], "plan_fast")

    # Flow D: Uncertainty-Aware Planning & Dynamic Re-planning
    def test_flow_d_uncertainty_re_planning(self):
        planner = UncertaintyAwarePlanner()
        planner.register_plan(
            plan_id="plan_circuit_assembly",
            primary=["place_esp32", "route_traces", "solder_joints"],
            branches={"solder_cold_joint": ["desolder_component", "clean_pads", "resolder_joints"]},
            tripwires=["solder_cold_joint"]
        )

        step_res = {"status": "FAILED", "error": "Inspection detected: solder_cold_joint on pin 4"}
        branch = planner.evaluate_execution_step("plan_circuit_assembly", step_res)
        self.assertEqual(branch["status"], "FALLBACK_TRIGGERED")
        self.assertEqual(branch["fallback_steps"], ["desolder_component", "clean_pads", "resolder_joints"])

    # Flow E: Sensor Fusion & Real-Time Environment State Estimation
    def test_flow_e_sensor_fusion_environment_state(self):
        fused_state = self.brain.sensor_fusion.fuse_readings(
            vision_input={"confidence": 0.94, "detected_objects": [{"label": "obstacle", "position": [0.6, 0.6, 0.0]}]},
            telemetry_input={"confidence": 0.98, "temperature_c": 22.8, "light_lux": 580.0},
            gpio_input={"confidence": 0.99, "e_stop_pin_high": False}
        )
        self.assertAlmostEqual(fused_state.ambient_temperature_c, 22.8, places=1)
        self.assertGreater(fused_state.overall_confidence, 0.85)
        self.assertEqual(len(fused_state.detected_obstacles), 1)

    # Flow F: Safe Hardware Command Execution & Verification (Robot/CNC/LFAM/3D Printer)
    def test_flow_f_hardware_command_execution(self):
        # Register CNC and 3D printer
        self.brain.robotics_hal.register_device(
            device_id="cnc_mill_01",
            name="Precision 5-Axis CNC",
            device_type=DeviceType.CNC,
            interface=DeviceInterfaceType.SERIAL
        )
        self.brain.robotics_hal.register_device(
            device_id="lfam_printer_01",
            name="Large Format Additive Machine",
            device_type=DeviceType.LFAM,
            interface=DeviceInterfaceType.ETHERNET_IP
        )

        # Ground and execute safe motion
        res_motion = self.brain.action_grounder.ground_and_execute(
            intent="Position CNC spindle",
            device_id="cnc_mill_01",
            command="motion",
            params={"target_coords": (0.2, 0.3, 0.1), "velocity": 0.4},
            is_authorized=True
        )
        self.assertTrue(res_motion["grounded"])
        self.assertEqual(res_motion["status"], "SUCCESS")

        # Telemetry read on LFAM
        res_telem = self.brain.action_grounder.ground_and_execute(
            intent="Read LFAM nozzle temperature",
            device_id="lfam_printer_01",
            command="read_telemetry",
            params={},
            is_authorized=True
        )
        self.assertTrue(res_telem["grounded"])
        self.assertEqual(res_telem["status"], "SUCCESS")

    # Flow G: GUI Interaction & Screen Analysis
    def test_flow_g_gui_interaction(self):
        screen = self.brain.gui_interaction.capture_screen_state(display_id=0)
        self.assertEqual(screen["status"], "SUCCESS")
        self.assertGreater(len(screen["ui_elements"]), 0)

        # Dispatch click
        dispatch_res = self.brain.gui_interaction.dispatch_input_action(
            action_type="mouse_click",
            target_coords=(120, 210),
            params={"button": "left"},
            is_authorized=True
        )
        self.assertEqual(dispatch_res["status"], "SUCCESS")

    # Flow H: Software Engineering & AST Modification with Snapshot Rollback
    def test_flow_h_software_engineering_rollback(self):
        swe = SoftwareEngineeringEngine(backup_dir=os.path.join(self.temp_dir, "backups"))
        target_py = os.path.join(self.temp_dir, "controller.py")
        with open(target_py, "w", encoding="utf-8") as f:
            f.write("def state(): return 'STABLE'\n")

        # Create versioned modification
        swe.modify_file_with_safety_rollback(target_py, "def state(): return 'EXPERIMENTAL'\n")
        with open(target_py, "r", encoding="utf-8") as f:
            self.assertIn("EXPERIMENTAL", f.read())

        # Rollback
        swe.rollback_file(target_py)
        with open(target_py, "r", encoding="utf-8") as f:
            self.assertIn("STABLE", f.read())

    # Flow I: Data Engineering & Normalization Pipeline
    def test_flow_i_data_engineering_pipeline(self):
        csv_input = "axis,speed,error\nx,10.5,0.01\ny,12.2,0.02\nz,5.0,0.005\n"
        validated = DataEngineeringEngine.parse_and_validate_csv(csv_input, required_columns=["axis", "speed", "error"])
        self.assertTrue(validated["valid"])
        self.assertEqual(validated["row_count"], 3)

        normalized = DataEngineeringEngine.normalize_records(
            validated["sample_rows"],
            {"axis": str, "speed": float, "error": float}
        )
        self.assertEqual(normalized[0]["axis"], "x")
        self.assertEqual(normalized[0]["speed"], 10.5)

    # Flow J: Experiment Design & Uplift Significance Calculation
    def test_flow_j_experiment_design(self):
        exp = ExperimentDesignEngine()
        control_feed = [100.0, 101.2, 99.5, 100.8, 100.2]
        treatment_feed = [118.5, 119.0, 117.8, 118.9, 118.2]
        res = exp.run_ab_experiment(
            experiment_id="exp_cnc_feed_rate",
            hypothesis="Higher feed rate optimizes machining duration",
            control_samples=control_feed,
            variant_samples=treatment_feed,
            min_uplift_threshold=0.10
        )
        self.assertTrue(res.statistically_significant)
        self.assertGreater(res.uplift_percent, 15.0)

    # Flow K: Incident Learning & Prevention Lesson Extraction
    def test_flow_k_incident_learning(self):
        rep = self.brain.incident_learning.record_failure(
            "Hardware communication timeout with robotic arm over CAN_BUS"
        )
        self.assertEqual(rep.remediation_policy, "RETRY_WITH_BACKOFF")
        lesson = self.brain.incident_learning.get_lesson_for_error("CAN_BUS timeout")
        self.assertIsNotNone(lesson)
        self.assertIn("timeout", lesson)

    # Flow L: Goal Persistence Across Restart & Subgoal Resumption
    def test_flow_l_goal_persistence_restart(self):
        goals_path = os.path.join(self.temp_dir, "persisted_goals")
        gpm = GoalPersistenceManager(storage_dir=goals_path)
        gpm.persist_goal({
            "goal_id": "goal_autonomous_assembly",
            "name": "Full Autonomous Manufacturing DAG",
            "status": "STAGE_2_ACTIVE",
            "active_subgoal": "solder_microcontroller"
        })

        # Emulate reboot with clean manager
        gpm_reboot = GoalPersistenceManager(storage_dir=goals_path)
        restored = gpm_reboot.restore_goal("goal_autonomous_assembly")
        self.assertIsNotNone(restored)
        self.assertEqual(restored["status"], "STAGE_2_ACTIVE")
        self.assertEqual(restored["active_subgoal"], "solder_microcontroller")

    # Flow M: Dynamic Skill Acquisition from Spec & 5-Stage Certification
    def test_flow_m_skill_acquisition_and_certification(self):
        engine = SkillAcquisitionEngine()
        spec = {
            "skill_id": "arc_welding_pass",
            "name": "Single-Pass Arc Weld",
            "category": "manufacturing",
            "procedure_steps": [{"step": 1, "action": "ignite_arc"}, {"step": 2, "action": "linear_bead"}],
            "constraints": ["voltage_nominal <= 24.0V"]
        }
        acquired_skill = engine.acquire_skill_from_spec(spec)
        cert_report = SkillCertifier.certify_skill(acquired_skill)
        self.assertTrue(cert_report["certified"])
        self.assertEqual(acquired_skill.certification_state, CertificationState.CERTIFIED)

    # Flow N: Demonstration Learning & Procedure Generalization
    def test_flow_n_demonstration_learning(self):
        execution_trace = [
            {"action": "capture_image", "tool": "camera_driver"},
            {"action": "localize_fiducial", "tool": "vision_detector"},
            {"action": "dispatch_actuation", "tool": "robotics_hal"}
        ]
        generalized_skill = DemonstrationLearningEngine.generalize_trace_to_skill(
            "fiducial_pick_and_place",
            execution_trace
        )
        self.assertEqual(generalized_skill.name, "fiducial_pick_and_place")
        self.assertEqual(len(generalized_skill.procedure_steps), 3)
        self.assertIn("robotics_hal", generalized_skill.required_tools)

    # Flow O: Capability Composition into Higher-Order Workflow
    def test_flow_o_capability_composition(self):
        composer = CapabilityComposer()
        composite = composer.compose_capabilities(
            composite_id="cap_autonomous_full_inspection",
            name="Autonomous Multimodal Quality Inspection",
            sub_capabilities=["camera_capture", "sensor_fusion", "data_engineering"],
            execution_graph=[{"stage": 1, "execute": "capture"}, {"stage": 2, "execute": "fuse"}]
        )
        self.assertEqual(composite["composite_id"], "cap_autonomous_full_inspection")
        self.assertEqual(len(composite["sub_capabilities"]), 3)

    # Flow P: Continual Learning Catastrophic-Forgetting Gating & Rejection
    def test_flow_p_continual_learning_gating(self):
        baseline = {"val_loss": 0.45, "skills_pass_rate": 1.0, "tools_pass_rate": 1.0}
        candidate_degraded = {"val_loss": 0.45, "skills_pass_rate": 0.88, "tools_pass_rate": 1.0}
        gate_res = self.brain.continual_learning.evaluate_retention(baseline, candidate_degraded)
        self.assertFalse(gate_res["safe_to_promote"])
        self.assertTrue(any("Skills retention dropped" in r for r in gate_res["reasons"]))

    # Flow Q: Curriculum Priority Calculation & Learning Item Ranking
    def test_flow_q_curriculum_priority_ranking(self):
        items = [
            LearningItem("it_adv_ai", "Deep Reinforcement Learning", "ai", "PPO algorithm", prerequisites=["math_prob"], utility_score=0.9, gap_urgency=0.9, risk_score=0.1),
            LearningItem("it_math_prob", "Probability Theory", "math", "Bayes rule", prerequisites=[], utility_score=0.7, gap_urgency=0.6, risk_score=0.1)
        ]
        ranked_before = self.brain.curriculum_priority.rank_curriculum(items, completed_prereqs=set())
        self.assertEqual(ranked_before[0].item_id, "it_math_prob")

        ranked_after = self.brain.curriculum_priority.rank_curriculum(items, completed_prereqs={"math_prob"})
        self.assertEqual(ranked_after[0].item_id, "it_adv_ai")

    # Flow R: Model Routing Based on Task Complexity & Security Tier
    def test_flow_r_model_routing(self):
        route_complex = self.brain.model_router.route_task("autonomous_system_refactoring", complexity_score=0.9)
        self.assertEqual(route_complex["selected_strategy"], "DECOMPOSED_AGENT_REASONING")
        self.assertEqual(route_complex["model_tier"], "PRIMARY_TARA_AI")

        route_fast = self.brain.model_router.route_task("sensor_heartbeat_check", complexity_score=0.05)
        self.assertEqual(route_fast["selected_strategy"], "DETERMINISTIC_DIRECT")
        self.assertEqual(route_fast["model_tier"], "FAST_INFERENCE")

    # Flow S: Decision Audit Trail with Secret Scrubbing
    def test_flow_s_audit_trail_secret_scrubbing(self):
        audit_entry = self.brain.audit_explainability.record_decision(
            actor_id="ROOT_OPERATOR",
            intent="Authorize LFAM firmware upgrade",
            evidence=["Firmware checksum verified", "Creator Ed25519 signature verified"],
            selected_strategy="DECOMPOSED_AGENT_REASONING",
            action="flash_firmware with auth_token: TOKEN-SECRET-1234567890abcdef",
            outcome="SUCCESS",
            rationale="Approved by Creator with api_key: KEY-1234567890123456."
        )
        self.assertNotIn("TOKEN-SECRET-1234567890abcdef", audit_entry["action"])
        self.assertNotIn("KEY-1234567890123456", audit_entry["rationale"])

    # Flow T: Full 10-Stage World Model + Prediction Error Learning Loop & Training Staging
    def test_flow_t_full_prediction_error_learning_loop(self):
        pred_expectation = PredictionExpectation(
            expected_duration_s=1.5,
            expected_state={"status": "SUCCESS", "spindle_rpm": 12000},
            expected_metrics={"latency_s": 1.5}
        )

        def mock_spindle_action():
            return {"status": "SUCCESS", "simulated_duration_s": 1.9, "spindle_rpm": 11950}

        record = self.brain.run_prediction_error_cycle(
            task_name="high_speed_milling_cycle",
            action_fn=mock_spindle_action,
            expectation=pred_expectation
        )
        self.assertEqual(record["task_name"], "high_speed_milling_cycle")
        self.assertAlmostEqual(record["error_duration_s"], 0.4, places=2)
        self.assertTrue(record["world_model_updated"])
        self.assertTrue(record["strategy_updated"])
        self.assertTrue(record["staged_for_training"])


if __name__ == "__main__":
    unittest.main()

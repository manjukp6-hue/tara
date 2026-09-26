"""
tests/test_advanced_capabilities_36.py

Comprehensive, Production-Grade Test Suite for the 36 Advanced Core Capabilities:
1. Knowledge Representation / Ontology (OntologyConcept, OntologyRelation, KnowledgeOntologyEngine)
2. Semantic Memory <-> Episodic Memory Consolidation (MemoryConsolidationEngine, ConsolidatedFact)
3. Continual Learning / Catastrophic-Forgetting Control (ContinualLearningGovernor)
4. Skill Acquisition Engine (SkillAcquisitionEngine, StructuredSkill)
5. Demonstration / Example Learning (DemonstrationLearningEngine)
6. Curriculum / Learning-Priority Engine (CurriculumPriorityEngine, LearningItem)
7. Model Adaptation Layer (ModelAdaptationLayer)
8. Model Routing (ModelRouter)
9. Knowledge Retrieval / Ranking (MultiFactorKnowledgeRanker)
10. Context Compression / Reconstruction (ContextCompressor)
11. Constraint Solving (ConstraintSolver, hard/soft constraints)
12. Optimization Engine (MultiObjectiveOptimizer, Pareto/composite scoring)
13. Probabilistic Reasoning (ProbabilisticReasoningEngine, Bayesian belief updates)
14. Planning Under Uncertainty (UncertaintyAwarePlanner, contingency branches)
15. Real-World Action Grounding (RealWorldActionGrounder, intent->execution->verification)
16. Environment Modeling (DynamicEnvironmentModel, ambient & obstacle collision)
17. Sensor Fusion (SensorFusionEngine, multimodal confidence weighting)
18. Device / Hardware Abstraction (RoboticsHAL, Phone, PC, Camera, CNC, LFAM, 3D Printer, Robot, ESP32)
19. Computer / GUI Interaction (GUIComputerInteraction, screen capture, gated action dispatch)
20. Software Engineering Capability (SoftwareEngineeringEngine, AST analysis, sandboxed testing, rollback)
21. Data Engineering Capability (DataEngineeringEngine, CSV validation, schema normalization, querying)
22. Experiment Design (ExperimentDesignEngine, A/B testing, statistical significance)
23. Failure Knowledge / Incident Learning (IncidentLearningEngine, root cause, prevention lessons)
24. Goal Persistence / Interruption Recovery (GoalPersistenceManager, serialization and restart recovery)
25. Self-Architecture Awareness (SelfArchitectureAwareness, introspective subsystem model)
26. Architecture Evolution Engine (ArchitectureEvolutionEngine, expansion evaluation)
27. Capability Composition (CapabilityComposer, higher-order workflow registration)
28. Skill Validation / Certification (SkillCertifier, structural & safety certification)
29. Policy / Strategy Learning (PolicyStrategyLearner, success-rate tracking)
30. Value / Utility Model (MultiObjectiveOptimizer candidate valuation)
31. Auditability / Explainability (AuditExplainabilityEngine, secret-scrubbed decision trails)
32. Provenance Propagation (ProvenanceRecord, lineage parents, trust scoring)
33. Privacy / Data Governance (DataGovernancePolicy, SecretScrubber, credential protection)
34. Recovery from Corrupted Learning (CorruptedLearningRecovery, automated quarantine)
35. Self-Maintenance (SelfMaintenanceEngine, health sweep of stale knowledge and broken tools)
36. Major Core Loop: World Model + Prediction Error Learning (PredictionErrorEngine, 10-stage loop)
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

from tara_core.knowledge_ontology import (
    KnowledgeOntologyEngine, OntologyConcept, OntologyRelation, RelationType,
    ProvenanceRecord, ProvenanceSourceType, MultiFactorKnowledgeRanker, DataGovernancePolicy
)
from tara_core.memory_consolidation import MemoryConsolidationEngine, ConsolidatedFact
from tara_core.reasoning_engine_advanced import (
    ConstraintSolver, MultiObjectiveOptimizer, ProbabilisticReasoningEngine,
    UncertaintyAwarePlanner
)
from tara_core.robotics_hal import (
    RoboticsHAL, SafetyInterlock, DeviceType, DeviceInterfaceType, DeviceProfile,
    GUIComputerInteraction
)
from tara_core.environment_sensor_fusion import (
    DynamicEnvironmentModel, SensorFusionEngine, RealWorldActionGrounder
)
from tara_core.engineering_capabilities import (
    SoftwareEngineeringEngine, DataEngineeringEngine, ExperimentDesignEngine
)
from tara_core.skill_acquisition_certification import (
    SkillAcquisitionEngine, DemonstrationLearningEngine, CapabilityComposer,
    SkillCertifier, SelfArchitectureAwareness, ArchitectureEvolutionEngine,
    StructuredSkill, CertificationState
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
from tara_core.cognitive_capabilities import CognitiveCapabilitiesHub
from tara_core.compressor import ContextCompressor
from tara_model.dynamic_dataset_compiler import SecretScrubber


class TestAdvancedCapabilities36(unittest.TestCase):
    """Exhaustive tests covering all 36 capabilities."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        KnowledgeOntologyEngine.reset_instance()
        MemoryConsolidationEngine.reset_instance()
        PredictionErrorEngine.reset_instance()
        self.ontology = KnowledgeOntologyEngine.get_default()

    def tearDown(self):
        KnowledgeOntologyEngine.reset_instance()
        MemoryConsolidationEngine.reset_instance()
        PredictionErrorEngine.reset_instance()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Knowledge Representation / Ontology
    def test_01_knowledge_ontology(self):
        prov = ProvenanceRecord(
            provenance_id="prov_test_01",
            source_type=ProvenanceSourceType.CREATOR_DIRECTIVE,
            source_uri="TARA/TEST",
            author="ROOT_OPERATOR",
            trust_score=1.0
        )
        self.ontology.register_concept(
            concept_id="robotics_planning",
            name="Robotics Motion Planning",
            category="ROBOTICS",
            properties={"motion_types": ["cartesian", "joint"]},
            rules=["RULE-ROB-01"],
            provenance=prov
        )
        self.ontology.add_relation("robotics_planning", RelationType.DEPENDS_ON, "robotics_hardware", weight=0.95)

        parents = self.ontology.get_parents("robotics_planning")
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["relation"], RelationType.DEPENDS_ON.value)

        path = self.ontology.find_path("tara_core", "robotics_hardware")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "tara_core")
        self.assertEqual(path[-1], "robotics_hardware")

    # 2. Semantic Memory <-> Episodic Memory Consolidation
    def test_02_memory_consolidation(self):
        consolidator = MemoryConsolidationEngine(repo_root=self.temp_dir)
        episode = {
            "episode_id": "ep_test_99",
            "actor_id": "creator_admin",
            "action": "execute_tool",
            "parameters": {"tool_name": "ast_analyzer", "secret_key": "SEC-KEY-12345678901234567890"},
            "result": {"status": "SUCCESS", "analysis": "clean AST"},
            "outcome": "SUCCESS"
        }
        facts = consolidator.consolidate_episode(episode)
        self.assertGreater(len(facts), 0)
        # Verify secret scrubbing in consolidated knowledge
        for f in facts:
            self.assertNotIn("SEC-KEY-12345678901234567890", str(f))

    # 3. Continual Learning / Catastrophic-Forgetting Control
    def test_03_continual_learning_governor(self):
        gov = ContinualLearningGovernor(max_allowed_degradation_ratio=0.10)
        baseline = {"val_loss": 0.50, "skills_pass_rate": 1.0, "tools_pass_rate": 1.0}

        # Acceptable minor change
        candidate_good = {"val_loss": 0.52, "skills_pass_rate": 1.0, "tools_pass_rate": 1.0}
        rep_good = gov.evaluate_retention(baseline, candidate_good)
        self.assertTrue(rep_good["safe_to_promote"])

        # Catastrophic forgetting in skills
        candidate_bad = {"val_loss": 0.50, "skills_pass_rate": 0.85, "tools_pass_rate": 1.0}
        rep_bad = gov.evaluate_retention(baseline, candidate_bad)
        self.assertFalse(rep_bad["safe_to_promote"])
        self.assertTrue(any("Skills retention dropped" in r for r in rep_bad["reasons"]))

    # 4. Skill Acquisition Engine
    def test_04_skill_acquisition_engine(self):
        engine = SkillAcquisitionEngine()
        spec = {
            "skill_id": "linear_interpolation",
            "name": "Linear Interpolation Path",
            "category": "robotics_kinematics",
            "prerequisites": ["cartesian_math"],
            "procedure_steps": [
                {"step": 1, "action": "compute_waypoints"},
                {"step": 2, "action": "check_velocity_bounds"}
            ],
            "constraints": ["max_velocity_mps <= 1.0"]
        }
        skill = engine.acquire_skill_from_spec(spec)
        self.assertEqual(skill.skill_id, "linear_interpolation")
        self.assertEqual(len(skill.procedure_steps), 2)
        self.assertEqual(skill.certification_state, CertificationState.EXPERIMENTAL)

    # 5. Demonstration / Example Learning
    def test_05_demonstration_learning(self):
        trace = [
            {"action": "read_sensor", "tool": "sensor_reader", "target": "camera_01"},
            {"action": "detect_feature", "tool": "vision_analyzer", "target": "bounding_box"},
            {"action": "move_actuator", "tool": "motion_controller", "target": "x_axis"}
        ]
        skill = DemonstrationLearningEngine.generalize_trace_to_skill("vision_guided_pick", trace)
        self.assertEqual(skill.name, "vision_guided_pick")
        self.assertEqual(len(skill.procedure_steps), 3)
        self.assertIn("vision_analyzer", skill.required_tools)
        self.assertIn("motion_controller", skill.required_tools)

    # 6. Curriculum / Learning-Priority Engine
    def test_06_curriculum_priority_engine(self):
        engine = CurriculumPriorityEngine()
        item1 = LearningItem(
            item_id="item_advanced_slam",
            topic="Advanced SLAM",
            category="robotics",
            content="Simultaneous Localization and Mapping",
            prerequisites=["basic_geometry"],
            utility_score=0.9,
            gap_urgency=0.8,
            risk_score=0.1
        )
        item2 = LearningItem(
            item_id="item_basic_geometry",
            topic="Basic Geometry",
            category="math",
            content="Euclidean space",
            prerequisites=[],
            utility_score=0.7,
            gap_urgency=0.5,
            risk_score=0.1
        )

        # Before basic_geometry is completed: basic_geometry ranks higher because advanced has unmet prereq
        ranked = engine.rank_curriculum([item1, item2], completed_prereqs=set())
        self.assertEqual(ranked[0].item_id, "item_basic_geometry")

        # After basic_geometry is completed: advanced_slam ranks higher due to higher utility & urgency
        ranked2 = engine.rank_curriculum([item1, item2], completed_prereqs={"basic_geometry"})
        self.assertEqual(ranked2[0].item_id, "item_advanced_slam")

    # 7. Model Adaptation Layer
    def test_07_model_adaptation_layer(self):
        layer = ModelAdaptationLayer()
        adapter = layer.register_adapter(
            adapter_id="lora_kinematics_v1",
            base_model="tara",
            rank=16,
            target_modules=["q_proj", "v_proj"]
        )
        self.assertEqual(adapter["adapter_id"], "lora_kinematics_v1")
        self.assertEqual(adapter["lora_rank"], 16)
        retrieved = layer.get_adapter("lora_kinematics_v1")
        self.assertIsNotNone(retrieved)

    # 8. Model Routing
    def test_08_model_routing(self):
        router = ModelRouter()
        # High complexity task
        r_high = router.route_task("multi_robot_scheduling", complexity_score=0.85)
        self.assertEqual(r_high["selected_strategy"], "DECOMPOSED_AGENT_REASONING")
        self.assertEqual(r_high["model_tier"], "PRIMARY_TARA_AI")

        # Simple fast task
        r_low = router.route_task("status_ping", complexity_score=0.1)
        self.assertEqual(r_low["selected_strategy"], "DETERMINISTIC_DIRECT")
        self.assertEqual(r_low["model_tier"], "FAST_INFERENCE")

    # 9. Knowledge Retrieval / Ranking
    def test_09_knowledge_retrieval_ranking(self):
        ranker = MultiFactorKnowledgeRanker()
        cands = [
            {"name": "Old Spec", "relevance": 0.8, "recency": 0.2, "confidence": 0.7, "authority": 0.5, "utility": 0.6},
            {"name": "Creator Directive", "relevance": 0.9, "recency": 0.95, "confidence": 1.0, "authority": 1.0, "utility": 1.0},
            {"name": "Low Match", "relevance": 0.1, "recency": 0.8, "confidence": 0.4, "authority": 0.3, "utility": 0.2}
        ]
        ranked = ranker.rank_candidates(cands)
        self.assertEqual(ranked[0]["name"], "Creator Directive")
        self.assertGreater(ranked[0]["_composite_rank_score"], ranked[1]["_composite_rank_score"])

    # 10. Context Compression / Reconstruction
    def test_10_context_compression(self):
        compressor = ContextCompressor()
        text = "TARA cognitive brain execution cycle. " * 30
        res = compressor.compress(text, max_tokens=25)
        self.assertLess(res.compressed_tokens, res.original_tokens)
        self.assertGreater(len(res.compressed_text), 0)

    # 11. Constraint Solving
    def test_11_constraint_solving(self):
        solver = ConstraintSolver()
        # Hard constraint: target velocity <= 1.5 m/s
        solver.add_hard_constraint("max_vel", lambda a: a.get("velocity", 0.0) <= 1.5, "Velocity limit")
        # Soft constraint: prefer velocity <= 1.0 m/s
        solver.add_soft_constraint("preferred_vel", lambda a: a.get("velocity", 0.0) <= 1.0, penalty=0.5, description="Preferred velocity")

        # Feasible nominal
        feas1 = solver.check_feasibility({"velocity": 0.8})
        self.assertTrue(feas1["feasible"])
        self.assertEqual(feas1["total_penalty"], 0.0)

        # Soft penalty
        feas2 = solver.check_feasibility({"velocity": 1.2})
        self.assertTrue(feas2["feasible"])
        self.assertEqual(feas2["total_penalty"], 0.5)

        # Hard violation
        feas3 = solver.check_feasibility({"velocity": 2.0})
        self.assertFalse(feas3["feasible"])
        self.assertEqual(feas3["hard_violation"], "max_vel")

    # 12. Optimization Engine
    def test_12_optimization_engine(self):
        opt = MultiObjectiveOptimizer()
        cands = [
            {"id": "plan_fast", "quality": 0.9, "reliability": 0.95, "time_s": 5.0, "cost": 10.0, "risk": 0.05},
            {"id": "plan_slow", "quality": 0.7, "reliability": 0.70, "time_s": 50.0, "cost": 80.0, "risk": 0.40}
        ]
        ranked = opt.rank_candidates(cands)
        self.assertEqual(ranked[0]["id"], "plan_fast")
        self.assertGreater(ranked[0]["composite_score"], ranked[1]["composite_score"])

    # 13. Probabilistic Reasoning
    def test_13_probabilistic_reasoning(self):
        engine = ProbabilisticReasoningEngine()
        engine.register_hypothesis(
            name="sensor_malfunction",
            prior=0.10,
            likelihoods={"erratic_telemetry": 0.90}
        )
        engine.register_hypothesis(
            name="normal_operation",
            prior=0.90,
            likelihoods={"erratic_telemetry": 0.05}
        )

        # Observe erratic telemetry
        post = engine.update_with_evidence("erratic_telemetry", observed=True)
        # Prior P(malfunction) was 0.10, but under erratic telemetry it should surge significantly
        self.assertGreater(post["sensor_malfunction"], 0.60)
        self.assertLess(post["normal_operation"], 0.40)

    # 14. Planning Under Uncertainty
    def test_14_planning_under_uncertainty(self):
        planner = UncertaintyAwarePlanner()
        planner.register_plan(
            plan_id="plan_weld_seam",
            primary=["pre_heat", "weld_pass_1", "inspect"],
            branches={"temperature_underrun": ["abort_pass", "reheat_zone", "retry_pass_1"]},
            tripwires=["temperature_underrun"]
        )

        nominal = planner.evaluate_execution_step("plan_weld_seam", {"status": "SUCCESS"})
        self.assertEqual(nominal["status"], "CONTINUE_PRIMARY")

        fail_tripwire = planner.evaluate_execution_step("plan_weld_seam", {
            "status": "FAILED",
            "error": "Sensor triggered: temperature_underrun below 180C"
        })
        self.assertEqual(fail_tripwire["status"], "FALLBACK_TRIGGERED")
        self.assertEqual(fail_tripwire["fallback_steps"], ["abort_pass", "reheat_zone", "retry_pass_1"])

    # 15. Real-World Action Grounding
    def test_15_action_grounding(self):
        grounder = RealWorldActionGrounder()
        # Authorized nominal execution
        res = grounder.ground_and_execute(
            intent="Measure workcell telemetry",
            device_id="mock_arm_01",
            command="read_telemetry",
            params={},
            is_authorized=True
        )
        self.assertTrue(res["grounded"])
        self.assertEqual(res["status"], "SUCCESS")

        # Unauthorized attempt
        unauth = grounder.ground_and_execute(
            intent="Unauthorized move",
            device_id="mock_arm_01",
            command="motion",
            params={"target_coords": (0.5, 0.5, 0.5)},
            is_authorized=False
        )
        self.assertEqual(unauth["status"], "BLOCKED")
        self.assertEqual(unauth["stage"], "PERMISSION_CHECK")

    # 16. Environment Modeling
    def test_16_environment_modeling(self):
        env = DynamicEnvironmentModel()
        env.update_ambient(temp_c=24.0, light_lux=650.0)
        self.assertEqual(env.ambient_temp_c, 24.0)
        self.assertEqual(env.ambient_light_lux, 650.0)

        env.add_or_update_obstacle("pillar_01", (0.5, 0.5, 0.5), radius_m=0.3)
        self.assertIn("pillar_01", env.obstacles)

        # Path directly intersecting obstacle midpoint
        collides = env.check_path_collision((0.0, 0.0, 0.5), (1.0, 1.0, 0.5))
        self.assertTrue(collides)

        # Path far away
        clear = env.check_path_collision((-1.0, -1.0, 0.0), (-0.8, -0.8, 0.0))
        self.assertFalse(clear)

    # 17. Sensor Fusion
    def test_17_sensor_fusion(self):
        env = DynamicEnvironmentModel()
        fusion = SensorFusionEngine(env_model=env)
        fused = fusion.fuse_readings(
            vision_input={"confidence": 0.92, "detected_objects": [{"label": "obstacle", "position": [0.4, 0.4, 0.2]}]},
            telemetry_input={"confidence": 0.95, "temperature_c": 23.5, "light_lux": 520.0},
            audio_input={"confidence": 0.80},
            gpio_input={"confidence": 0.99, "e_stop_pin_high": False}
        )
        self.assertEqual(fused.ambient_temperature_c, 23.5)
        self.assertEqual(len(fused.detected_obstacles), 1)
        self.assertGreater(fused.overall_confidence, 0.85)

    # 18. Device / Hardware Abstraction (8 Profiles)
    def test_18_device_hardware_abstraction(self):
        hal = RoboticsHAL.get_default()
        device_types = [
            DeviceType.PHONE, DeviceType.PC, DeviceType.CAMERA, DeviceType.CNC,
            DeviceType.LFAM, DeviceType.PRINTER_3D, DeviceType.ROBOT, DeviceType.ESP32
        ]
        for dt in device_types:
            prof = DeviceProfile(
                device_id=f"device_{dt.value.lower()}",
                name=f"Device {dt.value}",
                device_type=dt,
                interface=DeviceInterfaceType.SERIAL
            )
            self.assertTrue(hal.register_device(prof))
            self.assertIn(f"device_{dt.value.lower()}", hal.device_profiles)

    # 19. Computer / GUI Interaction
    def test_19_gui_interaction(self):
        gui = GUIComputerInteraction()
        screen = gui.capture_screen_state(display_id=0)
        self.assertEqual(screen["status"], "SUCCESS")
        self.assertIn("ui_elements", screen)
        self.assertGreater(len(screen["ui_elements"]), 0)

        # Authorized dispatch
        dispatched = gui.dispatch_input_action("click", (150, 220), params={"button": "left"}, is_authorized=True)
        self.assertEqual(dispatched["status"], "SUCCESS")

        # Unauthorized dispatch
        blocked = gui.dispatch_input_action("click", (150, 220), params={"button": "left"}, is_authorized=False)
        self.assertEqual(blocked["status"], "BLOCKED")

    # 20. Software Engineering Capability
    def test_20_software_engineering(self):
        backup_dir = os.path.join(self.temp_dir, "code_backups")
        swe = SoftwareEngineeringEngine(backup_dir=backup_dir)
        code = "class SensorNode:\n    def read(self):\n        return 42\n"
        analysis = swe.understand_code(code)
        self.assertTrue(analysis.is_valid_syntax)
        self.assertIn("SensorNode", analysis.classes_defined)
        self.assertIn("read", analysis.functions_defined)

        # Restricted execution test
        test_res = swe.test_code_snippet("x = sum([1, 2, 3, 4])", expected_variable="x", expected_value=10)
        self.assertEqual(test_res["status"], "SUCCESS")
        self.assertTrue(test_res["passed"])

        # Rollback verification
        target_file = os.path.join(self.temp_dir, "test_target.py")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("original_code = True\n")
        swe.modify_file_with_safety_rollback(target_file, "original_code = False\n")
        swe.rollback_file(target_file)
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("original_code = True", content)

    # 21. Data Engineering Capability
    def test_21_data_engineering(self):
        csv_data = "device_id,temperature,status\narm_01,24.5,active\narm_02,26.1,idle\n"
        val = DataEngineeringEngine.parse_and_validate_csv(csv_data, required_columns=["device_id", "temperature"])
        self.assertTrue(val["valid"])
        self.assertEqual(val["row_count"], 2)

        records = [
            {"id": "1", "temp": "24.5"},
            {"id": "2", "temp": "26.1"}
        ]
        normalized = DataEngineeringEngine.normalize_records(records, {"id": int, "temp": float})
        self.assertEqual(normalized[0]["id"], 1)
        self.assertEqual(normalized[0]["temp"], 24.5)

        queried = DataEngineeringEngine.query_dataset(normalized, {"id": 1})
        self.assertEqual(len(queried), 1)

    # 22. Experiment Design
    def test_22_experiment_design(self):
        exp = ExperimentDesignEngine()
        control = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9]
        variant = [12.5, 12.6, 12.4, 12.7, 12.5, 12.4, 12.6, 12.5]
        res = exp.run_ab_experiment(
            experiment_id="exp_accel_tuning",
            hypothesis="Higher feed rate increases throughput",
            control_samples=control,
            variant_samples=variant,
            min_uplift_threshold=0.10
        )
        self.assertTrue(res.statistically_significant)
        self.assertGreater(res.uplift_percent, 20.0)
        self.assertIn("confirmed", res.conclusion)

    # 23. Failure Knowledge / Incident Learning
    def test_23_incident_learning(self):
        engine = IncidentLearningEngine()
        rep = engine.record_failure("Communication timeout exceeded watchdog threshold 3.0s")
        self.assertEqual(rep.remediation_policy, "RETRY_WITH_BACKOFF")
        lesson = engine.get_lesson_for_error("watchdog threshold timeout")
        self.assertIsNotNone(lesson)
        self.assertIn("watchdog timeout", lesson)

    # 24. Goal Persistence / Interruption Recovery
    def test_24_goal_persistence(self):
        goals_dir = os.path.join(self.temp_dir, "saved_goals")
        gpm = GoalPersistenceManager(storage_dir=goals_dir)
        gdata = {
            "goal_id": "goal_autonomy_01",
            "name": "Complete Autonomous Loop",
            "status": "IN_PROGRESS",
            "dag_steps": ["step_audit", "step_verify", "step_promote"]
        }
        gpm.persist_goal(gdata)

        # Restore from simulated reboot
        gpm2 = GoalPersistenceManager(storage_dir=goals_dir)
        restored = gpm2.restore_goal("goal_autonomy_01")
        self.assertIsNotNone(restored)
        self.assertEqual(restored["name"], "Complete Autonomous Loop")
        self.assertEqual(len(restored["dag_steps"]), 3)

    # 25. Self-Architecture Awareness
    def test_25_self_architecture_awareness(self):
        snap = SelfArchitectureAwareness.get_architecture_snapshot()
        self.assertEqual(snap["system_name"], "TARA AI")
        self.assertEqual(snap["security_status"], "ENFORCED_FAIL_CLOSED")
        self.assertEqual(snap["creator_identity"], "ROOT_OPERATOR")
        self.assertIn("WorkingMemory", snap["memory_subsystems"])

    # 26. Architecture Evolution Engine
    def test_26_architecture_evolution(self):
        # Request hardware sensor extension
        rec_hw = ArchitectureEvolutionEngine.evaluate_evolution_need(
            "lidar_depth_sensor_hardware",
            existing_capabilities=["basic_vision", "cartesian_planning"]
        )
        self.assertEqual(rec_hw["recommendation"], "PLUGIN_OR_HAL_EXTENSION")
        self.assertFalse(rec_hw["requires_core_modification"])

        # Request new cognitive domain skill
        rec_skill = ArchitectureEvolutionEngine.evaluate_evolution_need(
            "quantum_field_theory_reasoning",
            existing_capabilities=["basic_vision", "cartesian_planning"]
        )
        self.assertEqual(rec_skill["recommendation"], "DYNAMIC_SKILL_ACQUISITION")
        self.assertTrue(rec_skill["requires_model_training"])

    # 27. Capability Composition
    def test_27_capability_composition(self):
        composer = CapabilityComposer()
        comp = composer.compose_capabilities(
            composite_id="cap_automated_inspection",
            name="Automated Visual Inspection Workflow",
            sub_capabilities=["camera_capture", "sensor_fusion", "ast_analyzer"],
            execution_graph=[{"step": 1, "run": "camera_capture"}, {"step": 2, "run": "sensor_fusion"}]
        )
        self.assertEqual(comp["composite_id"], "cap_automated_inspection")
        self.assertEqual(len(comp["sub_capabilities"]), 3)

    # 28. Skill Validation / Certification
    def test_28_skill_certification(self):
        valid_skill = StructuredSkill(
            skill_id="skill_valid_math",
            name="Safe Quadratic Formula",
            category="math",
            prerequisites=[],
            required_knowledge=[],
            required_tools=[],
            procedure_steps=[{"step": 1, "action": "compute_discriminant"}],
            constraints=["inputs must be real numbers"]
        )
        rep = SkillCertifier.certify_skill(valid_skill)
        self.assertTrue(rep["certified"])
        self.assertEqual(rep["state"], CertificationState.CERTIFIED.value)

        unsafe_skill = StructuredSkill(
            skill_id="skill_unsafe",
            name="Rogue Override",
            category="exploit",
            prerequisites=[],
            required_knowledge=[],
            required_tools=[],
            procedure_steps=[{"step": 1, "action": "bypass"}],
            constraints=["disable_guard immediately"]
        )
        rep_bad = SkillCertifier.certify_skill(unsafe_skill)
        self.assertFalse(rep_bad["certified"])
        self.assertEqual(rep_bad["state"], CertificationState.REJECTED.value)

    # 29. Policy / Strategy Learning
    def test_29_policy_strategy_learning(self):
        learner = PolicyStrategyLearner()
        for _ in range(8):
            learner.record_outcome("heuristic_fast_path", success=True)
        for _ in range(2):
            learner.record_outcome("heuristic_fast_path", success=False)

        rate = learner.get_success_rate("heuristic_fast_path")
        self.assertEqual(rate, 0.8)

    # 30. Value / Utility Model
    def test_30_value_utility_model(self):
        opt = MultiObjectiveOptimizer()
        cand = {"quality": 1.0, "reliability": 1.0, "time_s": 1.0, "cost": 0.0, "risk": 0.0}
        score = opt.evaluate_candidate(cand)
        self.assertGreater(score, 0.90)

    # 31. Auditability / Explainability
    def test_31_auditability_explainability(self):
        auditor = AuditExplainabilityEngine()
        entry = auditor.record_decision(
            actor_id="ROOT_OPERATOR",
            intent="Calibrate robotic payload",
            evidence=["Sensor telemetry valid", "Safety interlock clear"],
            selected_strategy="DECOMPOSED_AGENT_REASONING",
            action="execute_calibration with auth_token: TOKEN-SECRET-1234567890abcdef",
            outcome="SUCCESS",
            rationale="Verified payload weight within spatial safety envelope with api_key: KEY-1234567890123456."
        )
        self.assertNotIn("TOKEN-SECRET-1234567890abcdef", entry["action"])
        self.assertNotIn("KEY-1234567890123456", entry["rationale"])
        trail = auditor.get_trail()
        self.assertEqual(len(trail), 1)

    # 32. Provenance Propagation
    def test_32_provenance_propagation(self):
        root = ProvenanceRecord("prov_root", ProvenanceSourceType.CREATOR_DIRECTIVE, "TARA/CREATOR", "ROOT_OPERATOR", 1.0)
        derived = ProvenanceRecord(
            provenance_id="prov_child",
            source_type=ProvenanceSourceType.VERIFIED_EXPERIENCE,
            source_uri="TARA/EXPERIENCE/01",
            author="SYSTEM",
            trust_score=0.95,
            lineage_parent_ids=[root.provenance_id]
        )
        self.assertIn("prov_root", derived.lineage_parent_ids)
        self.assertEqual(derived.trust_score, 0.95)

    # 33. Privacy / Data Governance
    def test_33_privacy_data_governance(self):
        gov = DataGovernancePolicy()
        self.assertFalse(gov.can_remember("credentials", {"api_key": "12345"}))
        self.assertTrue(gov.can_remember("kinematics", {"axis_count": 6}))

        # Prohibit training on specific topic
        gov.prohibit_training_on("confidential_facility_schematic")
        self.assertFalse(gov.can_train_on("confidential_facility_schematic"))
        self.assertTrue(gov.can_train_on("standard_inverse_kinematics"))

    # 34. Recovery from Corrupted Learning
    def test_34_corrupted_learning_recovery(self):
        corrupt_file = os.path.join(self.temp_dir, "bad_dataset.jsonl")
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("corrupted data byte syntax error")

        res = CorruptedLearningRecovery.detect_and_quarantine(corrupt_file, "JSON syntax validation failure")
        self.assertEqual(res["status"], "QUARANTINED")
        self.assertFalse(os.path.exists(corrupt_file))
        self.assertTrue(os.path.exists(res["quarantined_file"]))

    # 35. Self-Maintenance
    def test_35_self_maintenance(self):
        knowledge_entries = [
            {"id": "k1", "title": "Old knowledge", "created_at": "2020-01-01T00:00:00Z"},
            {"id": "k2", "title": "Fresh knowledge", "created_at": "2026-09-15T00:00:00Z"}
        ]
        tools_list = [
            {"name": "tool_online", "enabled": True},
            {"name": "tool_deprecated", "enabled": False}
        ]
        sweep = SelfMaintenanceEngine.run_health_sweep(knowledge_entries, tools_list, max_stale_days=365)
        self.assertEqual(sweep["status"], "SWEEP_COMPLETED")
        self.assertIn("Old knowledge", sweep["stale_knowledge_items"])
        self.assertIn("tool_deprecated", sweep["broken_tools"])

    # 36. Major Core Loop: World Model + Prediction Error Learning
    def test_36_prediction_error_loop(self):
        loop_engine = PredictionErrorEngine.get_default(repo_root=self.temp_dir)
        expectation = PredictionExpectation(
            expected_duration_s=2.0,
            expected_state={"status": "SUCCESS"},
            expected_metrics={"latency_s": 2.0}
        )

        def mock_action():
            return {"status": "SUCCESS", "simulated_duration_s": 2.4, "telemetry": "nominal"}

        record = loop_engine.execute_and_learn_cycle("test_drill_cycle", mock_action, expectation)
        self.assertEqual(record.task_name, "test_drill_cycle")
        self.assertAlmostEqual(record.error_duration_s, 0.4, places=2)
        self.assertEqual(record.classified_cause, "NOMINAL_EXECUTION")
        self.assertTrue(record.world_model_updated)
        self.assertTrue(record.strategy_updated)
        self.assertEqual(loop_engine.get_calibrated_latency("test_drill_cycle"), 2.4)

    # Verification of CognitiveCapabilitiesHub Integration
    def test_37_cognitive_capabilities_hub_full_integration(self):
        hub = CognitiveCapabilitiesHub.get_default(repo_root=self.temp_dir)
        all_caps = hub.list_capabilities()
        # Verify that all 36 capabilities are accounted for
        self.assertGreaterEqual(len(all_caps), 36)
        self.assertIsNotNone(hub.ontology)
        self.assertIsNotNone(hub.memory_consolidation)
        self.assertIsNotNone(hub.continual_learning)
        self.assertIsNotNone(hub.skill_acquisition)
        self.assertIsNotNone(hub.prediction_error_loop)


if __name__ == "__main__":
    unittest.main()

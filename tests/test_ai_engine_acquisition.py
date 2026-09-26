"""
tests/test_ai_engine_acquisition.py

Comprehensive test suite verifying AI-Driven Engine Acquisition, Generation,
Sandbox Execution, Certification, Outcome Verification, and Learning Lifecycle.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, 'python')
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.dynamic_engine_system import (
    DynamicEngineSystem,
    AcquisitionPath,
    EngineCandidate,
    EngineSandboxRunner,
    OutcomeVerifier,
    EngineAcquisitionLearner,
    EngineCreationGenerator,
    EngineAcquisitionPipeline,
    EngineManifest,
    EngineStatus
)
from tara_core.registry import CapabilityRegistry, CapabilityCategory
from tara_core.extended_capabilities import WorldStateModel
from tara_core.resilience_maintenance import GoalPersistenceManager, GoalStatus, SelfMaintenanceEngine


class TestAIEngineAcquisition(unittest.TestCase):

    def setUp(self):
        self.engine_system = DynamicEngineSystem.get_default(repo_root=REPO_ROOT)
        self.pipeline = self.engine_system.acquisition_pipeline

    def test_01_missing_capability_detection(self):
        gap = self.pipeline.detect_capability_gap(
            task_name="unseen_tensor_fft_processor",
            payload={"dimensions": [1024, 1024]},
            context={"category": "QUANTUM"}
        )
        self.assertTrue(gap["gap_detected"])
        self.assertEqual(gap["missing_capability"], "unseen_tensor_fft_processor")
        self.assertEqual(gap["recommended_path"], AcquisitionPath.NEW_REUSABLE_ENGINE_MODULE.value)
        self.assertEqual(gap["category"], "QUANTUM")

    def test_02_existing_engine_discovery(self):
        dummy_manifest = EngineManifest(
            engine_id="test_dummy_existing_eng",
            name="Existing Audio Filter",
            category="AUDIO",
            capabilities=["denoise_audio"],
            supported_tasks=["denoise_audio"]
        )
        self.engine_system.evaluate_and_register_engine(
            manifest_data=dummy_manifest,
            handler=lambda p: {"status": "SUCCESS", "audio_cleaned": True}
        )

        gap = self.pipeline.detect_capability_gap(
            task_name="denoise_audio",
            payload={}
        )
        self.assertFalse(gap["gap_detected"])
        self.assertEqual(gap["recommendation"], "USE_EXISTING_ENGINE")

    def test_03_candidate_engine_evaluation(self):
        candidate = EngineCreationGenerator.generate_engine_candidate(
            task_name="analyze_spectrogram",
            category="AUDIO"
        )
        self.assertIsInstance(candidate, EngineCandidate)
        self.assertEqual(candidate.category, "AUDIO")
        self.assertIn("analyze_spectrogram", candidate.manifest.supported_tasks)
        self.assertTrue(candidate.source_code.startswith("def execute_analyze_spectrogram"))

    def test_04_new_engine_creation_pipeline(self):
        res = self.engine_system.acquire_or_create_engine(
            task_name="cluster_hyperspectral_bands",
            category="REMOTE_SENSING",
            acceptance_criteria={"required_keys": ["status", "output"]}
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "CERTIFIED_AND_REGISTERED")
        self.assertEqual(res["category"], "REMOTE_SENSING")
        self.assertTrue(self.engine_system.registry.has_engine(res["engine_id"]))

    def test_05_security_rejection_ast(self):
        unsafe_code = """
import os
def execute_malicious_code(params):
    os.system("rm -rf /")
    return {"status": "injected"}
"""
        res = self.engine_system.acquire_or_create_engine(
            task_name="malicious_eval_probe",
            category="EXPLOIT",
            candidate_code=unsafe_code
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "SECURITY_REJECTED")
        self.assertTrue(any("unauthorized module import" in e or "system" in e for e in res["errors"]))

    def test_06_sandbox_execution(self):
        sandbox = EngineSandboxRunner()
        handler = lambda p: {"status": "SUCCESS", "energy_level": p.get("voltage", 0) * 1.5}
        run_res = sandbox.run_in_sandbox(
            handler=handler,
            test_payload={"voltage": 12.0},
            acceptance_criteria={"required_output_keys": ["status", "energy_level"]}
        )
        self.assertTrue(run_res["passed"])
        self.assertEqual(run_res["output"]["energy_level"], 18.0)

    def test_07_dependency_validation(self):
        spec = {"dependencies": ["completely_non_existent_fake_module_xyz_12345"]}
        res = self.engine_system.acquire_or_create_engine(
            task_name="dependency_check_task",
            category="COMPUTATION",
            spec=spec
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "MISSING_DEPENDENCIES")

    def test_08_resource_validation(self):
        spec = {"min_ram_mb": 10_000_000_000}
        res = self.engine_system.acquire_or_create_engine(
            task_name="huge_ram_task",
            category="SIMULATION",
            spec=spec
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "RESOURCE_INCOMPATIBLE")

    def test_09_functional_certification(self):
        res = self.engine_system.acquire_or_create_engine(
            task_name="solve_navier_stokes_grid",
            category="FLUID_DYNAMICS"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "CERTIFIED_AND_REGISTERED")

        manifest = self.engine_system.registry.get_manifest(res["engine_id"])
        self.assertEqual(manifest.status, EngineStatus.ACTIVE)

    def test_10_dynamic_registration(self):
        res = self.engine_system.acquire_or_create_engine(
            task_name="optimize_orbital_insertion",
            category="ASTRODYNAMICS"
        )
        self.assertTrue(res["success"])
        eng_id = res["engine_id"]

        self.assertTrue(self.engine_system.registry.has_engine(eng_id))
        cap = CapabilityRegistry.get_default().get_capability(f"engine_{eng_id}")
        self.assertIsNotNone(cap)
        self.assertEqual(cap.category, CapabilityCategory.ENGINE)

    def test_11_failed_engine_quarantine(self):
        bad_manifest = EngineManifest(
            engine_id="bad_flaky_engine_quarantine_test",
            name="Flaky Engine",
            category="CUSTOM",
            supported_tasks=["flaky_task"]
        )

        def failing_handler(p):
            raise RuntimeError("Hardware bus desynchronization")

        self.engine_system.evaluate_and_register_engine(
            manifest_data=bad_manifest,
            handler=failing_handler
        )

        for _ in range(3):
            self.engine_system.execute_task("flaky_task", {})

        manifest = self.engine_system.registry.get_manifest("bad_flaky_engine_quarantine_test")
        self.assertEqual(manifest.status, EngineStatus.QUARANTINED)

    def test_12_outcome_verification(self):
        acq = self.engine_system.acquire_or_create_engine(
            task_name="render_neural_sdf",
            category="3D_VISION"
        )
        self.assertTrue(acq["success"])

        res = self.engine_system.execute_with_outcome_verification(
            task_type="render_neural_sdf",
            payload={"resolution": 512, "data": "mesh_data"},
            acceptance_criteria={"required_keys": ["status", "output"]}
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["outcome_report"]["verified"])

    def test_13_learning_from_acquisition_result(self):
        history = self.engine_system.get_acquisition_history()
        self.assertGreater(len(history), 0)

        last_rec = history[-1]
        self.assertIn("provenance_id", last_rec)
        self.assertIn("task_name", last_rec)
        self.assertIn("category", last_rec)
        self.assertIn("status", last_rec)
        self.assertIn("timestamp", last_rec)
        self.assertIn("digest", last_rec)

    def test_14_novel_engine_category_unlimited(self):
        res = self.engine_system.acquire_or_create_engine(
            task_name="simulate_gravitational_waves",
            category="QUANTUM_ASTROPHYSICS"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["category"], "QUANTUM_ASTROPHYSICS")

        exec_res = self.engine_system.execute(
            task_type="simulate_gravitational_waves",
            payload={"mass_solar": 30.5}
        )
        self.assertTrue(exec_res["success"])
        self.assertEqual(exec_res["category"], "QUANTUM_ASTROPHYSICS")

    def test_15_existing_tara_functionality_intact(self):
        from tara_core.brain import TaraBrain
        brain = TaraBrain()
        self.assertTrue(hasattr(brain, "process"))
        self.assertTrue(hasattr(brain, "execute_engine"))
        self.assertTrue(hasattr(brain, "resolve_missing_capability"))

        wm = WorldStateModel()
        wm.register_entity("camera_01", "DEVICE", "HD Camera", {"fps": 30})
        delta_rec = wm.track_temporal_state("camera_01", {"fps": 60})
        self.assertEqual(delta_rec["delta"], {"fps": 60})

        gp = GoalPersistenceManager()
        gid = "test_autonomy_goal_99"
        gp.persist_goal({"goal_id": gid, "objective": "deploy_orbit"})
        gp.pause_goal(gid, reason="operator_hold")
        restored = gp.restore_goal(gid)
        self.assertEqual(restored["status"], GoalStatus.PAUSED.value)
        gp.resume_goal(gid)
        restored2 = gp.restore_goal(gid)
        self.assertEqual(restored2["status"], GoalStatus.RUNNING.value)

        diag = SelfMaintenanceEngine.run_comprehensive_diagnostics()
        self.assertEqual(diag["total_subsystems"], 11)


if __name__ == "__main__":
    unittest.main()

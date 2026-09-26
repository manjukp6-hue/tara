"""
tests/test_advanced_reasoning_and_analogy.py

Unit and integration tests for:
1. ConceptAbstractionEngine (Capability 3: Abstraction & Concept Formation)
2. AnalogyReasoningEngine (Capability 4: Analogy & Structural Transfer)
3. CrossModalReasoningEngine (Capability 60: Cross-Modal Reasoning & Joint Alignment)
4. DynamicCapabilityAcquisitionEngine (12-Step Dynamic Capability Acquisition Pipeline)
5. TaraBrain integration for missing capability resolution
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.reasoning_engine_advanced import (
    ConceptAbstractionEngine,
    AbstractConcept,
    AnalogyReasoningEngine,
    RelationalDomain,
    AnalogyMapping
)
from tara_core.extended_capabilities import (
    CrossModalReasoningEngine,
    MultimodalInput,
    ModalityType
)
from tara_core.skill_acquisition_certification import (
    DynamicCapabilityAcquisitionEngine,
    SecurityASTAuditor
)
from tara_core.registry import CapabilityRegistry
from TARA.TOOLS.provenance_tracker import verify_provenance


class TestAdvancedReasoningAndAnalogy(unittest.TestCase):

    def test_concept_abstraction_formation(self):
        engine = ConceptAbstractionEngine()
        instances = [
            {"id": "sparrow", "has_wings": True, "can_fly": True, "warm_blooded": True, "color": "brown"},
            {"id": "eagle", "has_wings": True, "can_fly": True, "warm_blooded": True, "color": "golden"},
            {"id": "robin", "has_wings": True, "can_fly": True, "warm_blooded": True, "color": "red"}
        ]

        concept = engine.form_concept_from_instances(
            concept_name="flying_bird",
            instances=instances,
            invariance_threshold=0.8
        )

        self.assertEqual(concept.name, "flying_bird")
        self.assertIn("has_wings", concept.common_attributes)
        self.assertIn("can_fly", concept.common_attributes)
        self.assertIn("warm_blooded", concept.common_attributes)
        self.assertNotIn("color", concept.common_attributes)
        self.assertIn("has_wings == true", [inv.lower() for inv in concept.invariants])

        # Test hierarchical generalization
        water_bird_instances = [
            {"id": "duck", "has_wings": True, "can_fly": True, "warm_blooded": True, "swims": True},
            {"id": "swan", "has_wings": True, "can_fly": True, "warm_blooded": True, "swims": True}
        ]
        water_concept = engine.form_concept_from_instances(
            concept_name="water_bird",
            instances=water_bird_instances,
            invariance_threshold=0.8
        )
        generalized = engine.generalize_concept([concept.concept_id, water_concept.concept_id], "avian")
        self.assertEqual(generalized.name, "avian")
        self.assertIn("has_wings", generalized.common_attributes)
        self.assertNotIn("swims", generalized.common_attributes)

    def test_analogy_structural_transfer(self):
        engine = AnalogyReasoningEngine()

        # Solar system domain
        engine.register_domain(
            name="solar_system",
            entities=["sun", "planet"],
            relations=[
                ("sun", "attracts", "planet"),
                ("sun", "heavier_than", "planet"),
                ("planet", "orbits", "sun")
            ]
        )

        # Rutherford atom model domain
        engine.register_domain(
            name="rutherford_atom",
            entities=["nucleus", "electron"],
            relations=[
                ("nucleus", "attracts", "electron"),
                ("nucleus", "heavier_than", "electron")
            ]
        )

        mapping = engine.map_analogy(
            source_domain_name="solar_system",
            target_domain_name="rutherford_atom"
        )

        self.assertEqual(mapping.source_domain, "solar_system")
        self.assertEqual(mapping.target_domain, "rutherford_atom")
        self.assertGreater(mapping.structural_similarity_score, 0.5)
        # Verify inferred relation: ('electron', 'orbits', 'nucleus') transferred to target
        self.assertIn(("electron", "orbits", "nucleus"), mapping.inferred_target_relations)

    def test_cross_modal_reasoning(self):
        engine = CrossModalReasoningEngine()

        inputs = [
            MultimodalInput(
                modality=ModalityType.TEXT,
                content_uri="text://sample",
                metadata={"text": "A red car moving rapidly north"}
            ),
            MultimodalInput(
                modality=ModalityType.IMAGE,
                content_uri="image://frame01.jpg",
                extracted_features={"detected_objects": ["car", "vehicle"]}
            ),
            MultimodalInput(
                modality=ModalityType.SENSOR,
                content_uri="sensor://telemetry",
                extracted_features={"sensor_type": "odometry", "values_in_nominal_range": True}
            )
        ]

        alignment = engine.align_and_reason(inputs)
        self.assertEqual(len(alignment.modalities_involved), 3)
        self.assertGreater(alignment.coherence_score, 0.7)
        self.assertFalse(alignment.discrepancies)

        # Test with sensor anomaly
        anomaly_inputs = [
            MultimodalInput(
                modality=ModalityType.SENSOR,
                content_uri="sensor://imu",
                extracted_features={"sensor_type": "imu", "values_in_nominal_range": False}
            )
        ]
        anomaly_res = engine.align_and_reason(anomaly_inputs)
        self.assertTrue(anomaly_res.discrepancies)
        self.assertLess(anomaly_res.coherence_score, 1.0)

    def test_dynamic_capability_acquisition_12_step_pipeline(self):
        engine = DynamicCapabilityAcquisitionEngine()

        task_payload = {"input": "telemetry_checksum_test", "multiplier": 3}
        res = engine.resolve_and_execute(
            task_name="dynamic_telemetry_analyzer",
            task_payload=task_payload,
            actor_id="TARA_TEST_ACTOR"
        )

        self.assertEqual(res["status"], "RESOLVED")
        self.assertEqual(res["resolution_method"], "DYNAMIC_SYNTHESIS_AND_CERTIFICATION")
        self.assertTrue(res["ast_audit_passed"])
        self.assertTrue(res["sandbox_verification_passed"])
        self.assertIn("STEP_1_GAP_DETECTION", res["steps_trace"])
        self.assertIn("STEP_8_AST_AUDIT", res["steps_trace"])
        self.assertIn("STEP_9_SANDBOX_TEST", res["steps_trace"])
        self.assertIn("STEP_10_REGISTER_CAPABILITY", res["steps_trace"])
        self.assertIn("STEP_11_RETRY_TASK", res["steps_trace"])
        self.assertIn("STEP_12_RECORD_PROVENANCE", res["steps_trace"])

        # Check cryptographic provenance
        prov = res["provenance_record"]
        self.assertIsNotNone(prov)
        verification = verify_provenance(prov)
        self.assertEqual(verification["status"], "SUCCESS")
        self.assertTrue(verification["verified"])

    def test_dynamic_capability_acquisition_rejects_unsafe_code(self):
        engine = DynamicCapabilityAcquisitionEngine()

        malicious_code = """
import os
def execute_unsafe_task(payload):
    os.system("echo compromised")
    return {"status": "injected"}
"""
        res = engine.resolve_and_execute(
            task_name="malicious_exploit_task",
            task_payload={},
            candidate_code=malicious_code,
            candidate_entry_point="execute_unsafe_task"
        )

        self.assertEqual(res["status"], "SECURITY_VIOLATION")
        self.assertFalse(res["ast_audit_passed"])
        self.assertIn("AST_AUDIT_FAILED", res["steps_trace"])
        self.assertTrue(any("unauthorized module import" in v or "disallowed method" in v for v in res["violations"]))

    def test_tara_brain_resolve_missing_capability(self):
        from tara_core.brain import TaraBrain
        brain = TaraBrain()

        self.assertTrue(hasattr(brain, "resolve_missing_capability"))
        res = brain.resolve_missing_capability(
            task_name="unseen_autonomous_scheduler",
            task_payload={"schedule_interval": 60}
        )

        self.assertEqual(res["status"], "RESOLVED")
        self.assertEqual(res["resolution_method"], "DYNAMIC_SYNTHESIS_AND_CERTIFICATION")
        self.assertTrue(res["ast_audit_passed"])


if __name__ == "__main__":
    unittest.main()

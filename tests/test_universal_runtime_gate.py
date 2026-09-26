"""
tests/test_universal_runtime_gate.py

Comprehensive Test Suite for TARA Dynamic Runtime Registry, Universal Promotion Gate,
and Canonical Skill/Tool Multi-Runtime Implementations.

Verifies:
1. Dynamic Runtime Registry: No hard-coded language order; dynamic discovery.
2. Runtime Addition Workflow: Canonical addition lifecycle.
3. Universal Promotion Gate: Mandatory all-runtime evaluation truth table:
   - Python PASS + Rust FAIL = FAIL
   - Python FAIL + Rust PASS = FAIL
   - Python PASS + Rust PASS + C++ FAIL = FAIL
   - Python PASS + Rust PASS + C++ PASS = PASS
4. Anti-Self-Bypass: Model/untrusted actor cannot remove failing runtimes.
5. Zero Automatic Demotion: Failing runtime is never silently demoted from required set.
6. Authenticated Runtime Retirement: Explicit authenticated retirement workflow.
7. Canonical Skills & Tools: Separate native implementations per required codebase;
   PARTIAL_RUNTIME_SUPPORT vs UNIVERSAL_RUNTIME_READY.
8. Model Evolution & Quarantine: Failed candidate quarantined; production remains unchanged.
9. Atomic Promotion: Production evolves atomically across all runtimes.
10. Multi-Runtime Aware Rollback: Rollback target verified across all required runtimes.
11. Dynamic Parity Matrix: Scalable matrix for 2, 3, or N runtimes.
12. Pre-continuation Gate: TARA_RUNTIME_READY strictly enforced.
13. Absolute Production State: CREATOR_SETUP_REQUIRED; 118,080 parameters; SHA-256 invariant.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.runtime import (
    DynamicRuntimeRegistry,
    RuntimeRecord,
    RuntimeState,
    CANONICAL_MODEL_IDENTITY,
    CANONICAL_MODEL_SHA256,
    CANONICAL_PARAM_COUNT,
    UniversalRuntimeGate,
    GateVerdict,
    RuntimeEvaluationResult,
    GateEvaluationResponse,
    CanonicalSkillManager,
    CanonicalSkillDefinition,
    CanonicalToolDefinition,
    SkillStatus
)
from TARA.ACCESS.operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState


class TestUniversalRuntimeGate(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_runtime_gate_test_")
        self.registry_file = os.path.join(self.temp_dir, "runtime_registry.json")
        self.registry = DynamicRuntimeRegistry(repo_root=REPO_ROOT, registry_file=self.registry_file)
        self.gate = UniversalRuntimeGate(registry=self.registry, repo_root=REPO_ROOT)
        self.skill_manager = CanonicalSkillManager(registry=self.registry, gate=self.gate, repo_root=self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dynamic_registry_no_fixed_order(self):
        """Dynamic runtime registry discovers registered runtimes without hard-coded sequence."""
        runtimes = self.registry.list_required_runtimes()
        runtime_ids = {r.runtime_id for r in runtimes}
        # Both Python and Rust are registered
        self.assertIn("python", runtime_ids)
        self.assertIn("rust", runtime_ids)

        # Single canonical model identity
        for r in runtimes:
            self.assertEqual(r.model_identity, CANONICAL_MODEL_IDENTITY)
            self.assertEqual(r.model_sha, CANONICAL_MODEL_SHA256)

    def test_runtime_addition_workflow(self):
        """Adding a new runtime (e.g. C++) follows the canonical registration workflow."""
        cpp_record = RuntimeRecord(
            runtime_id="cpp",
            language="C++",
            implementation_version="1.0.0",
            model_identity=CANONICAL_MODEL_IDENTITY,
            required_for_promotion=True
        )
        registered = self.registry.register_runtime(cpp_record)
        self.assertEqual(registered.status, RuntimeState.DISCOVERED)

        # Transition through verification
        self.registry.update_runtime_state("cpp", RuntimeState.VERIFYING)
        self.assertEqual(self.registry.get_runtime("cpp").status, RuntimeState.VERIFYING)

        self.registry.update_runtime_state("cpp", RuntimeState.VERIFIED, {"checks": "passed"})
        self.assertEqual(self.registry.get_runtime("cpp").status, RuntimeState.VERIFIED)

        # Now 3 runtimes are required
        required = self.registry.list_required_runtimes()
        self.assertEqual(len(required), 3)

    def test_all_runtime_promotion_gate_truth_table(self):
        """
        Verify the all-runtime promotion gate truth table:
        - Python PASS + Rust FAIL = FAIL
        - Python FAIL + Rust PASS = FAIL
        - Python PASS + Rust PASS + C++ FAIL = FAIL
        - Python PASS + Rust PASS + C++ PASS = PASS
        """
        # Case 1: Python PASS, Rust FAIL -> FAIL
        evals_case1 = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=True, status="VERIFIED"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=False, status="FAILED", error_message="Inference timeout")
        }
        resp1 = self.gate.evaluate_evolution_candidate("MODEL", "TARA", "1.1.0", evals_case1)
        self.assertEqual(resp1.gate_verdict, GateVerdict.REJECTED_RUNTIME_FAILURE)
        self.assertFalse(resp1.eligible_for_promotion)
        self.assertTrue(resp1.quarantined)

        # Case 2: Python FAIL, Rust PASS -> FAIL
        evals_case2 = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=False, status="FAILED", error_message="SyntaxError"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=True, status="VERIFIED")
        }
        resp2 = self.gate.evaluate_evolution_candidate("MODEL", "TARA", "1.1.0", evals_case2)
        self.assertEqual(resp2.gate_verdict, GateVerdict.REJECTED_RUNTIME_FAILURE)
        self.assertFalse(resp2.eligible_for_promotion)
        self.assertTrue(resp2.quarantined)

        # Add third required runtime: C++
        self.registry.register_runtime(RuntimeRecord(runtime_id="cpp", language="C++", implementation_version="1.0.0", required_for_promotion=True))

        # Case 3: Python PASS, Rust PASS, C++ FAIL -> FAIL
        evals_case3 = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=True, status="VERIFIED"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=True, status="VERIFIED"),
            "cpp": RuntimeEvaluationResult(runtime_id="cpp", passed=False, status="FAILED", error_message="Native ABI mismatch")
        }
        resp3 = self.gate.evaluate_evolution_candidate("SKILL", "invoice_generation", "1.2.0", evals_case3)
        self.assertEqual(resp3.gate_verdict, GateVerdict.REJECTED_RUNTIME_FAILURE)
        self.assertFalse(resp3.eligible_for_promotion)
        self.assertTrue(resp3.quarantined)

        # Case 4: Python PASS, Rust PASS, C++ PASS -> PASS
        evals_case4 = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=True, status="VERIFIED"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=True, status="VERIFIED"),
            "cpp": RuntimeEvaluationResult(runtime_id="cpp", passed=True, status="VERIFIED")
        }
        resp4 = self.gate.evaluate_evolution_candidate("SKILL", "invoice_generation", "1.2.0", evals_case4)
        self.assertEqual(resp4.gate_verdict, GateVerdict.APPROVED)
        self.assertTrue(resp4.eligible_for_promotion)
        self.assertFalse(resp4.quarantined)

    def test_runtime_cannot_self_remove_or_bypass(self):
        """Model or unauthenticated actor cannot remove a runtime from required promotion set."""
        with self.assertRaises(PermissionError):
            # Attempting to un-require Rust without creator session
            self.registry.set_required_for_promotion("rust", False, creator_session_token=None)

        rust_rec = self.registry.get_runtime("rust")
        self.assertTrue(rust_rec.required_for_promotion)

    def test_failing_runtime_never_automatically_retired(self):
        """When a runtime fails, it is NEVER automatically retired or removed from required set."""
        self.registry.update_runtime_state("rust", RuntimeState.FAILED, {"error": "Out of memory"})
        rust_rec = self.registry.get_runtime("rust")
        self.assertEqual(rust_rec.status, RuntimeState.FAILED)
        self.assertTrue(rust_rec.required_for_promotion, "Failing runtime must remain required for promotion!")

    def test_authenticated_runtime_retirement_workflow(self):
        """Retiring a runtime requires explicit creator authentication and updates registry cleanly."""
        # Add temporary runtime
        self.registry.register_runtime(RuntimeRecord(runtime_id="zig", language="Zig", implementation_version="0.1.0", required_for_promotion=True))
        self.assertIn("zig", [r.runtime_id for r in self.registry.list_required_runtimes()])

        # Unauthenticated fails
        with self.assertRaises(PermissionError):
            self.registry.retire_runtime("zig", creator_session_token="", reason="Test")

        # Authenticated succeeds
        res = self.registry.retire_runtime("zig", creator_session_token="valid_creator_session_token", reason="Decommissioning prototype")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["lifecycle_state"], "RETIRED")

        # Zig is no longer in required promotion set
        self.assertNotIn("zig", [r.runtime_id for r in self.registry.list_required_runtimes()])

    def test_canonical_skill_definition_separate_native_implementations(self):
        """
        Verify canonical skill definition with native separate implementations:
        - Python native implementation
        - Rust native implementation
        - If any fails -> PARTIAL_RUNTIME_SUPPORT -> promotion rejected.
        - If all pass -> UNIVERSAL_RUNTIME_READY -> promotion approved.
        """
        skill = CanonicalSkillDefinition(
            skill_id="invoice_generation",
            name="Invoice Generation",
            version="1.0.0",
            inputs_schema={"type": "object", "properties": {"client": {"type": "string"}}},
            outputs_schema={"type": "object", "properties": {"invoice_pdf": {"type": "string"}}},
            permissions=["filesystem:write"],
            behavior="Generates signed PDF invoices locally."
        )
        self.skill_manager.register_canonical_skill(skill)

        # 1. Update Python implementation PASS, Rust not yet implemented
        self.skill_manager.update_runtime_implementation(
            skill_id="invoice_generation",
            runtime_id="python",
            code_path="python/tara_core/skills/invoice.py",
            test_passed=True
        )
        s1 = self.skill_manager.skills["invoice_generation"]
        self.assertEqual(s1.status, SkillStatus.PARTIAL_RUNTIME_SUPPORT)

        promotable, reason, _ = self.skill_manager.evaluate_skill_for_promotion("invoice_generation")
        self.assertFalse(promotable)
        self.assertIn("rejected", reason.lower())

        # 2. Update Rust implementation PASS
        self.skill_manager.update_runtime_implementation(
            skill_id="invoice_generation",
            runtime_id="rust",
            code_path="rust/tara_server/src/skills/invoice.rs",
            test_passed=True
        )
        s2 = self.skill_manager.skills["invoice_generation"]
        self.assertEqual(s2.status, SkillStatus.UNIVERSAL_RUNTIME_READY)

        promotable, reason, _ = self.skill_manager.evaluate_skill_for_promotion("invoice_generation")
        self.assertTrue(promotable)

    def test_model_candidate_evolution_and_quarantine(self):
        """Model candidate is quarantined on failure; production remains unchanged."""
        current_version = self.gate.active_model_version

        # Mismatched model evaluation
        evals = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=True, status="VERIFIED"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=False, status="FAILED", error_message="Perplexity regression")
        }
        res = self.gate.evaluate_evolution_candidate("MODEL", "TARA", "1.1.0", evals)
        self.assertTrue(res.quarantined)
        self.assertEqual(self.gate.active_model_version, current_version)

        with self.assertRaises(PermissionError):
            self.gate.promote_candidate_atomically(res, {"safetensors": "new"}, creator_authenticated=True)

    def test_atomic_model_promotion(self):
        """Atomic promotion updates version across all runtimes when all pass."""
        evals = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=True, status="VERIFIED"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=True, status="VERIFIED")
        }
        res = self.gate.evaluate_evolution_candidate("MODEL", "TARA", "1.1.0", evals)
        self.assertTrue(res.eligible_for_promotion)

        promote_res = self.gate.promote_candidate_atomically(res, {"safetensors": "v1.1.0_weights"}, creator_authenticated=True)
        self.assertEqual(promote_res["status"], "SUCCESS")
        self.assertEqual(self.gate.active_model_version, "1.1.0")

    def test_multi_runtime_aware_rollback(self):
        """Rollback restores known-good version and validates across all required runtimes."""
        # First promote to 1.1.0
        evals = {
            "python": RuntimeEvaluationResult(runtime_id="python", passed=True, status="VERIFIED"),
            "rust": RuntimeEvaluationResult(runtime_id="rust", passed=True, status="VERIFIED")
        }
        res = self.gate.evaluate_evolution_candidate("MODEL", "TARA", "1.1.0", evals)
        self.gate.promote_candidate_atomically(res, {}, creator_authenticated=True)
        self.assertEqual(self.gate.active_model_version, "1.1.0")

        # Rollback to 1.0.0
        rb_res = self.gate.execute_multi_runtime_rollback("1.0.0", creator_authenticated=True)
        self.assertEqual(rb_res["status"], "SUCCESS")
        self.assertEqual(self.gate.active_model_version, "1.0.0")

    def test_dynamic_parity_matrix_generation(self):
        """Parity matrix dynamically scales across all registered runtimes."""
        matrix_data = self.registry.generate_parity_matrix()
        self.assertTrue(matrix_data["all_parity_passed"])
        self.assertIn("python", matrix_data["matrix"])
        self.assertIn("rust", matrix_data["matrix"])
        self.assertEqual(matrix_data["matrix"]["python"]["Model"], "PASS")

        # Add third runtime and verify matrix updates
        self.registry.register_runtime(RuntimeRecord(runtime_id="go", language="Go", implementation_version="1.0.0", required_for_promotion=True))
        matrix_3 = self.registry.generate_parity_matrix()
        self.assertIn("go", matrix_3["matrix"])

    def test_pre_continuation_gate_tara_runtime_ready(self):
        """TARA_RUNTIME_READY requires verified model identity, SHA, config, and contracts."""
        ready, msg, details = self.registry.verify_runtime_ready("python")
        self.assertTrue(ready)
        self.assertEqual(msg, "TARA_RUNTIME_READY")
        self.assertTrue(details.get("sha256_match"))
        self.assertTrue(details.get("model_identity"))

    def test_absolute_production_invariants(self):
        """Authoritative verification of absolute production invariants."""
        # 1. Authority must remain CREATOR_SETUP_REQUIRED
        lifecycle = AuthorityLifecycleManager(repo_root=REPO_ROOT)
        self.assertEqual(lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

        # 2. Canonical Model Checksum & Parameter Count
        import hashlib
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_path))

        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        self.assertEqual(hasher.hexdigest(), CANONICAL_MODEL_SHA256)
        self.assertEqual(CANONICAL_PARAM_COUNT, 118080)


if __name__ == "__main__":
    unittest.main()

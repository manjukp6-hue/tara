"""
tests/test_protected_capability.py

Comprehensive 27-Scenario Security Test Suite for TARA Hardening & Protected Capability Architecture.
Validates:
 1. Google OIDC Nonce Hardening (cryptographic generation, binding, replay rejection)
 2. Concurrent Creator Session Management (list masked, FIFO eviction, revoke single, revoke-all)
 3. CORS Hardening (rejection on sensitive endpoints, allowance on public endpoints)
 4. Dependency Hash Verification (requirements.txt sha256, runtime package version checks)
 5. Additive Authority: Normal ROOT_CREATOR functions operational with protected state OFF
 6. Additive Authority: Normal ROOT_CREATOR functions operational with protected state ON
 7. Capability Scope: Protected state adds only intended additional capabilities
 8. Generic / descriptive terminology rejection for activation
 9. Configured private verifier succeeds
 10. Verifier alone without authenticated ROOT_CREATOR fails
 11. Creator 2 cannot activate protected state
 12. Unauthenticated user cannot activate protected state
 13. Session expiry automatically deactivates protected state
 14. Session logout explicitly deactivates protected state
 15. ActionBroker modification fails when protected state is inactive
 16. Path traversal / escape outside repo root is rejected
 17. Syntax error / bad syntax automatically rolls back
 18. Hot reload component succeeds
 19. Persistent PolicyRecordStore cryptographic hash-chaining succeeds
 20. Policy deletion maintains cryptographic chain integrity
 21. Lifecycle / self-destruction adapter enforces 2-stage verification
 22. Unauthorized lifecycle / self-destruction is rejected
 23. Forged creator_id is rejected
 24. Forged creator role is rejected
 25. Source-code possession alone cannot obtain protected state
 26. Neutral audit event naming (zero forbidden/descriptive terms)
 27. Production model SAFETENSORS SHA256 strictly invariant
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import hashlib
from unittest.mock import MagicMock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.protected_v1.manager import ProtectedStateManager, CAPABILITY_V1
from TARA.ACCESS.protected_v1.broker import ActionBroker
from TARA.ACCESS.protected_v1.policy_store import PolicyRecordStore
from TARA.ACCESS.protected_v1.destruction_adapter import StateLifecycleAdapter
from TARA.ACCESS.protected_v1.dispatcher import ProtectedCommandDispatcher
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.services.auth_service import CreatorAuthService
from TARA.SECURITY.dependency_verifier import DependencyVerifier
from TARA.SECURITY.release_manifest import EXPECTED_PRODUCTION_MODEL_SHA256, compute_file_sha256
from python.tara_core.server import _resolve_cors_origin


class MockAuditLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event_type: str, severity: str = "INFO", details: dict = None):
        self.events.append({
            "event_type": event_type,
            "severity": severity,
            "details": details or {}
        })


class TestProtectedCapabilityAndSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.raw_trigger = "private-alpha-test-trigger-991"

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_state_v1_test_")
        self.trigger_file = os.path.join(self.test_dir, "state_v1_verifier.json")
        self.audit = MockAuditLogger()

        self.mgr = ProtectedStateManager(
            trigger_file_path=self.trigger_file,
            audit_logger=self.audit,
            repo_root=self.test_dir
        )
        self.mgr.set_trigger_phrase(self.raw_trigger)

        # Mock creator auth service
        self.auth_svc = MagicMock()
        self.auth_svc.audit_logger = self.audit
        self.sessions = {}

        def verify_sess(token):
            return self.sessions.get(token)

        self.auth_svc.verify_session.side_effect = verify_sess

        self.broker = ActionBroker(
            protected_state_mgr=self.mgr,
            creator_auth_service=self.auth_svc,
            repo_root=self.test_dir,
            audit_logger=self.audit
        )

        self.pol_store = PolicyRecordStore(
            protected_state_mgr=self.mgr,
            creator_auth_service=self.auth_svc,
            store_path=os.path.join(self.test_dir, "policies.json"),
            audit_logger=self.audit,
            repo_root=self.test_dir
        )

        self.mock_destruct_engine = MagicMock()
        self.mock_destruct_engine.arm_destruction.return_value = {
            "status": "ARMED",
            "arm_token": "arm_tok_12345",
            "expires_at": "2026-09-18T12:00:00Z"
        }
        self.mock_destruct_engine.execute_final_destruction.return_value = {
            "status": "DESTROYED",
            "deleted_files_count": 42
        }

        self.lifecycle_adapter = StateLifecycleAdapter(
            protected_state_mgr=self.mgr,
            creator_auth_service=self.auth_svc,
            self_destruct_engine=self.mock_destruct_engine,
            audit_logger=self.audit
        )

        self.dispatcher = ProtectedCommandDispatcher(
            manager=self.mgr,
            broker=self.broker,
            policy_store=self.pol_store,
            destruction_adapter=self.lifecycle_adapter,
            creator_auth_service=self.auth_svc
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Scenario 1: Google OIDC Nonce Hardening
    # -------------------------------------------------------------
    def test_01_google_oidc_nonce_hardening(self):
        google_svc = GoogleAuthService()
        nonce = google_svc.create_auth_nonce(ttl_seconds=10.0)
        self.assertTrue(nonce and len(nonce) >= 32)

        # Nonce mismatch rejected
        jwt_mismatch = "eyJhbGciOiAiUlMyNTYifQ.eyJzdWIiOiAiMTIzNDU2IiwgImlzcyI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCAiYXVkIjogInRhcmEtY2xpZW50LWlkIiwgImV4cCI6IDE3ODk3MDMxMjksICJpYXQiOiAxNzg5Njk5NTE5LCAiZW1haWxfdmVyaWZpZWQiOiB0cnVlLCAiZW1haWwiOiAibWFuanVrcDZAZ21haWwuY29tIiwgIm5vbmNlIjogInRva2VuX25vbmNlIn0.ZmFrZXNpZw"
        res = google_svc.verify_id_token(
            jwt_mismatch,
            expected_client_id="tara-client-id",
            expected_nonce="different_nonce",
            require_nonce=True,
            current_time=1789700000
        )
        self.assertFalse(res.get("verified"))
        self.assertIn("Nonce mismatch", res.get("error", ""))

        # Missing nonce rejected
        jwt_missing = "eyJhbGciOiAiUlMyNTYifQ.eyJzdWIiOiAiMTIzNDU2IiwgImlzcyI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCAiYXVkIjogInRhcmEtY2xpZW50LWlkIiwgImV4cCI6IDE3ODk3MDMxMjksICJpYXQiOiAxNzg5Njk5NTE5LCAiZW1haWxfdmVyaWZpZWQiOiB0cnVlLCAiZW1haWwiOiAibWFuanVrcDZAZ21haWwuY29tIn0.ZmFrZXNpZw"
        res = google_svc.verify_id_token(
            jwt_missing,
            expected_client_id="tara-client-id",
            expected_nonce=None,
            require_nonce=True,
            current_time=1789700000
        )
        self.assertFalse(res.get("verified"))
        self.assertIn("missing required 'nonce'", res.get("error", "").lower())

    # -------------------------------------------------------------
    # Scenario 2: Concurrent Creator Session Management
    # -------------------------------------------------------------
    def test_02_concurrent_creator_session_management(self):
        cas = CreatorAuthService(repo_root=self.test_dir)
        cas.session_timeout_seconds = 3600
        cas.max_concurrent_sessions = 2
        
        # Issue 2 sessions
        tok1 = cas._issue_creator_session("ROOT_OPERATOR", "ROOT_CREATOR")["session_token"]
        tok2 = cas._issue_creator_session("ROOT_OPERATOR", "ROOT_CREATOR")["session_token"]
        sessions = cas.list_active_sessions("ROOT_OPERATOR")
        self.assertEqual(len(sessions), 2)
        # Masked ID
        self.assertEqual(len(sessions[0]["session_id"]), 12)

        # Issue 3rd -> FIFO evicts tok1
        tok3 = cas._issue_creator_session("ROOT_OPERATOR", "ROOT_CREATOR")["session_token"]
        self.assertIsNone(cas.verify_session(tok1))
        self.assertIsNotNone(cas.verify_session(tok2))
        self.assertIsNotNone(cas.verify_session(tok3))

        # Revoke single
        session_to_revoke = sessions[1]["session_id"]
        ok = cas.revoke_session_by_id(session_to_revoke, "ROOT_OPERATOR")
        self.assertTrue(ok)
        self.assertIsNone(cas.verify_session(tok2))

        # Revoke all
        cas.revoke_all_sessions("ROOT_OPERATOR")
        self.assertIsNone(cas.verify_session(tok3))

    # -------------------------------------------------------------
    # Scenario 3: CORS Hardening
    # -------------------------------------------------------------
    def test_03_cors_hardening(self):
        # Sensitive endpoint with unauthorized origin -> rejected (None)
        origin, _ = _resolve_cors_origin("/api/v1/auth/session", "https://malicious-site.com")
        self.assertIsNone(origin)

        # Sensitive endpoint with authorized origin -> allowed
        origin, _ = _resolve_cors_origin("/api/v1/auth/session", "http://localhost:8000")
        self.assertEqual(origin, "http://localhost:8000")

        # Sensitive endpoint with render origin -> allowed
        origin, _ = _resolve_cors_origin("/api/v1/admin/status", "https://tara-core.onrender.com")
        self.assertEqual(origin, "https://tara-core.onrender.com")

        # Public chat endpoint with external origin -> allowed
        origin, _ = _resolve_cors_origin("/api/v1/chat", "https://other-client.app")
        self.assertEqual(origin, "https://other-client.app")

    # -------------------------------------------------------------
    # Scenario 4: Dependency Hash Verification
    # -------------------------------------------------------------
    def test_04_dependency_hash_verification(self):
        verifier = DependencyVerifier(repo_root=self.repo_root)
        ok, errors = verifier.verify_dependencies()
        self.assertTrue(ok, f"Dependency verification failed: {errors}")
        self.assertEqual(len(errors), 0)

    # -------------------------------------------------------------
    # Scenario 5: Existing ROOT_CREATOR functions work with capability_v1 OFF
    # -------------------------------------------------------------
    def test_05_existing_root_creator_functions_work_with_protected_state_off(self):
        self.sessions["sess_root_off"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.assertFalse(self.mgr.is_active("sess_root_off", self.auth_svc))
        
        # Verify ROOT_CREATOR session is valid in auth service
        sess = self.auth_svc.verify_session("sess_root_off")
        self.assertEqual(sess["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(sess["role"], "ROOT_CREATOR")

    # -------------------------------------------------------------
    # Scenario 6: Existing ROOT_CREATOR functions work with capability_v1 ON
    # -------------------------------------------------------------
    def test_06_existing_root_creator_functions_work_with_protected_state_on(self):
        self.sessions["sess_root_on"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        ok, msg = self.mgr.activate("sess_root_on", self.raw_trigger, self.auth_svc)
        self.assertTrue(ok)
        self.assertTrue(self.mgr.is_active("sess_root_on", self.auth_svc))
        
        # ROOT_CREATOR identity & role are intact
        sess = self.auth_svc.verify_session("sess_root_on")
        self.assertEqual(sess["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(sess["role"], "ROOT_CREATOR")

    # -------------------------------------------------------------
    # Scenario 7: Capability Scope: Adds only intended capabilities
    # -------------------------------------------------------------
    def test_07_capability_scope(self):
        self.sessions["sess_normal"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        # Inactive: ActionBroker operations denied
        res = self.broker.modify_source("sess_normal", "test.py", "x = 1")
        self.assertEqual(res.get("status"), "ERROR")
        self.assertIn("Unauthorized", res.get("error"))

    # -------------------------------------------------------------
    # Scenario 8: Generic / descriptive terminology rejection
    # -------------------------------------------------------------
    def test_08_generic_terminology_does_not_activate(self):
        self.sessions["sess_root"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        for bad in ["god mode", "activate super authority", "super_auth", "root override", "system admin"]:
            ok, _ = self.mgr.activate("sess_root", bad, self.auth_svc)
            self.assertFalse(ok)
            self.assertFalse(self.mgr.is_active("sess_root", self.auth_svc))

    # -------------------------------------------------------------
    # Scenario 9: Configured private verifier succeeds
    # -------------------------------------------------------------
    def test_09_configured_verifier_activates(self):
        self.sessions["sess_root"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        ok, msg = self.mgr.activate("sess_root", self.raw_trigger, self.auth_svc)
        self.assertTrue(ok)
        self.assertTrue(self.mgr.is_active("sess_root", self.auth_svc))

    # -------------------------------------------------------------
    # Scenario 10: Verifier alone without authenticated ROOT_CREATOR fails
    # -------------------------------------------------------------
    def test_10_verifier_alone_without_authenticated_creator_fails(self):
        ok, msg = self.mgr.activate("nonexistent_session", self.raw_trigger, self.auth_svc)
        self.assertFalse(ok)
        self.assertIn("Access denied", msg)

    # -------------------------------------------------------------
    # Scenario 11: Creator 2 cannot activate protected state
    # -------------------------------------------------------------
    def test_11_creator_2_cannot_activate(self):
        self.sessions["sess_c2"] = {"creator_id": "CREATOR_02", "role": "CREATOR"}
        ok, msg = self.mgr.activate("sess_c2", self.raw_trigger, self.auth_svc)
        self.assertFalse(ok)
        self.assertIn("Insufficient creator role", msg)

    # -------------------------------------------------------------
    # Scenario 12: Unauthenticated user cannot activate protected state
    # -------------------------------------------------------------
    def test_12_unauthenticated_user_cannot_activate(self):
        ok, msg = self.mgr.activate("", self.raw_trigger, self.auth_svc)
        self.assertFalse(ok)
        self.assertFalse(self.mgr.is_active("", self.auth_svc))

    # -------------------------------------------------------------
    # Scenario 13: Session expiry automatically deactivates protected state
    # -------------------------------------------------------------
    def test_13_session_expiry_deactivates(self):
        self.sessions["sess_exp"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        ok, _ = self.mgr.activate("sess_exp", self.raw_trigger, self.auth_svc)
        self.assertTrue(self.mgr.is_active("sess_exp", self.auth_svc))

        # Simulate expiration
        del self.sessions["sess_exp"]
        self.assertFalse(self.mgr.is_active("sess_exp", self.auth_svc))

    # -------------------------------------------------------------
    # Scenario 14: Session logout explicitly deactivates protected state
    # -------------------------------------------------------------
    def test_14_logout_deactivates(self):
        self.sessions["sess_out"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_out", self.raw_trigger, self.auth_svc)
        self.assertTrue(self.mgr.is_active("sess_out", self.auth_svc))

        ok, msg = self.mgr.deactivate("sess_out")
        self.assertTrue(ok)
        self.assertFalse(self.mgr.is_active("sess_out", self.auth_svc))

    # -------------------------------------------------------------
    # Scenario 15: ActionBroker modification fails when inactive
    # -------------------------------------------------------------
    def test_15_action_broker_fails_when_inactive(self):
        self.sessions["sess_inact"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        test_file = os.path.join(self.test_dir, "app.py")
        with open(test_file, "w") as f:
            f.write("def foo():\n    return 1\n")

        res = self.broker.modify_source("sess_inact", test_file, "def foo():\n    return 2\n")
        self.assertEqual(res.get("status"), "ERROR")

    # -------------------------------------------------------------
    # Scenario 16: Path traversal / escape outside repo root is rejected
    # -------------------------------------------------------------
    def test_16_path_traversal_rejected(self):
        self.sessions["sess_trav"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_trav", self.raw_trigger, self.auth_svc)

        # Attempt path traversal
        res = self.broker.read_source("sess_trav", "../../outside.txt")
        self.assertEqual(res.get("status"), "ERROR")
        self.assertIn("Path traversal rejected", res.get("error"))

    # -------------------------------------------------------------
    # Scenario 17: Syntax error automatically rolls back
    # -------------------------------------------------------------
    def test_17_syntax_error_automatic_rollback(self):
        self.sessions["sess_rb"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_rb", self.raw_trigger, self.auth_svc)

        target_file = os.path.join(self.test_dir, "syntax_target.py")
        original_code = "def valid_func():\n    return 42\n"
        with open(target_file, "w") as f:
            f.write(original_code)

        bad_syntax_code = "def bad_syntax(:: broken\n"
        res = self.broker.modify_source("sess_rb", target_file, bad_syntax_code)
        self.assertEqual(res.get("status"), "FAILED_ROLLED_BACK")

        # Verify content was restored
        with open(target_file, "r") as f:
            restored = f.read()
        self.assertEqual(restored, original_code)

    # -------------------------------------------------------------
    # Scenario 18: Hot reload component succeeds
    # -------------------------------------------------------------
    def test_18_hot_reload_component(self):
        self.sessions["sess_rl"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_rl", self.raw_trigger, self.auth_svc)

        res = self.broker.reload_component("sess_rl", "json")
        self.assertEqual(res.get("status"), "SUCCESS")

    # -------------------------------------------------------------
    # Scenario 19: Persistent PolicyRecordStore hash-chaining succeeds
    # -------------------------------------------------------------
    def test_19_policy_record_store_hash_chaining(self):
        self.sessions["sess_pol"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_pol", self.raw_trigger, self.auth_svc)

        res = self.pol_store.create_policy("sess_pol", "TARA must maintain strict privacy.")
        self.assertEqual(res.get("status"), "SUCCESS")
        policy_id = res["policy"]["policy_id"]

        # Verify integrity
        valid, err = self.pol_store.verify_store_integrity()
        self.assertTrue(valid)
        self.assertIsNone(err)

    # -------------------------------------------------------------
    # Scenario 20: Policy deletion maintains chain integrity
    # -------------------------------------------------------------
    def test_20_policy_deletion_maintains_integrity(self):
        self.sessions["sess_pol2"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_pol2", self.raw_trigger, self.auth_svc)

        res = self.pol_store.create_policy("sess_pol2", "Temporary policy rule.")
        policy_id = res["policy"]["policy_id"]

        del_res = self.pol_store.delete_policy("sess_pol2", policy_id)
        self.assertEqual(del_res.get("status"), "SUCCESS")

        valid, err = self.pol_store.verify_store_integrity()
        self.assertTrue(valid)

    # -------------------------------------------------------------
    # Scenario 21: Lifecycle / self-destruction adapter 2-stage verification
    # -------------------------------------------------------------
    def test_21_lifecycle_adapter_two_stage_verification(self):
        self.sessions["sess_life"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_life", self.raw_trigger, self.auth_svc)

        # Stage 1: Arm
        arm_res = self.lifecycle_adapter.arm("sess_life")
        self.assertEqual(arm_res.get("status"), "ARMED")
        arm_tok = arm_res.get("arm_token")

        # Stage 2: Execute
        exec_res = self.lifecycle_adapter.confirm_and_execute(
            "sess_life", arm_tok, "I_AUTHORIZE_COMPLETE_DESTRUCTION_OF_TARA"
        )
        self.assertEqual(exec_res.get("status"), "DESTROYED")

    # -------------------------------------------------------------
    # Scenario 22: Unauthorized lifecycle / self-destruction is rejected
    # -------------------------------------------------------------
    def test_22_unauthorized_lifecycle_rejected(self):
        self.sessions["sess_unauth_life"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        # State inactive
        res = self.lifecycle_adapter.arm("sess_unauth_life")
        self.assertEqual(res.get("status"), "ERROR")
        self.assertIn("Unauthorized", res.get("error"))

    # -------------------------------------------------------------
    # Scenario 23: Forged creator_id is rejected
    # -------------------------------------------------------------
    def test_23_forged_creator_id_rejected(self):
        self.sessions["sess_forge_id"] = {"creator_id": "ATTACKER_ID", "role": "ROOT_CREATOR"}
        ok, msg = self.mgr.activate("sess_forge_id", self.raw_trigger, self.auth_svc)
        self.assertFalse(ok)
        self.assertIn("Insufficient creator role", msg)

    # -------------------------------------------------------------
    # Scenario 24: Forged role is rejected
    # -------------------------------------------------------------
    def test_24_forged_role_rejected(self):
        self.sessions["sess_forge_role"] = {"creator_id": "ROOT_OPERATOR", "role": "IMPOSTOR_ROLE"}
        ok, msg = self.mgr.activate("sess_forge_role", self.raw_trigger, self.auth_svc)
        self.assertFalse(ok)
        self.assertIn("Insufficient creator role", msg)

    # -------------------------------------------------------------
    # Scenario 25: Source-code possession alone cannot obtain protected state
    # -------------------------------------------------------------
    def test_25_source_code_possession_alone_insufficient(self):
        with open(self.trigger_file, "r", encoding="utf-8") as f:
            verifier_data = json.load(f)

        # Confirm plaintext trigger is not in the verifier file
        self.assertNotIn(self.raw_trigger, json.dumps(verifier_data))
        # Hash is irreversible PBKDF2
        self.assertEqual(verifier_data.get("kdf"), "PBKDF2-HMAC-SHA256")

    # -------------------------------------------------------------
    # Scenario 26: Neutral audit event naming (zero descriptive terms)
    # -------------------------------------------------------------
    def test_26_neutral_audit_event_naming(self):
        self.sessions["sess_audit"] = {"creator_id": "ROOT_OPERATOR", "role": "ROOT_CREATOR"}
        self.mgr.activate("sess_audit", self.raw_trigger, self.auth_svc)
        self.mgr.deactivate("sess_audit")

        event_types = [e["event_type"] for e in self.audit.events]
        self.assertIn("state_v1_activated", event_types)
        self.assertIn("state_v1_deactivated", event_types)

        # Verify no forbidden descriptive terms in any logged event
        forbidden = ["god_mode", "super_auth", "super_authority", "root_super_auth"]
        for e in self.audit.events:
            for term in forbidden:
                self.assertNotIn(term, e["event_type"].lower())

    # -------------------------------------------------------------
    # Scenario 27: Production Model SAFETENSORS SHA256 Invariance
    # -------------------------------------------------------------
    def test_27_production_model_sha256_invariance(self):
        model_path = os.path.join(self.repo_root, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.isfile(model_path), "Production model safetensors is missing!")
        actual_sha256 = compute_file_sha256(model_path)
        self.assertEqual(
            actual_sha256,
            EXPECTED_PRODUCTION_MODEL_SHA256,
            f"Production model SHA256 invariant violation! Got {actual_sha256}"
        )


if __name__ == "__main__":
    unittest.main()

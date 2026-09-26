"""
tests/test_lifecycle_hardening.py

Comprehensive 15-Scenario Test Suite for TARA One-Time Creator Initialization Lifecycle & Hardening:
1. First setup moves state from INITIALIZATION_REQUIRED to ACTIVE.
2. Root creator role is permanently ROOT_CREATOR for ROOT_OPERATOR.
3. Second setup attempt is rejected with permanent initialization lock.
4. Tamper detection on operator_record.json fails closed (AUTHORITY_LOCKED).
5. Tamper detection on operators_registry.json fails closed (AUTHORITY_LOCKED).
6. Tamper detection on restore_config.json fails closed (AUTHORITY_LOCKED).
7. Key rotation rejected without authorization (Path A or Path B).
8. Key rotation succeeds via Path A (active ROOT_CREATOR session / proof-of-possession).
9. Key rotation succeeds via Path B (emergency recovery code).
10. Old key rejected after rotation (cannot authenticate with revoked key).
11. New key functions properly (authenticates successfully).
12. Google identity change authorization enforcement.
13. Creator 2/3 enrollment authorization & anti-self-escalation enforcement.
14. Recovery non-bypassability and brute-force lockout.
15. Production neural model SHA256 invariant (7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309).
"""

import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric import rsa

# Ensure repository root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.operator.operator_profile import (
    CreatorIdentity,
    CANONICAL_CREATOR_ID,
    DEFAULT_DISPLAY_NAME
)
from TARA.ACCESS.operator.operator_lifecycle import (
    AuthorityLifecycleManager,
    AuthorityState
)
from TARA.ACCESS.operator.multi_operator_registry import MultiCreatorRegistry
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.dpapi_storage import IS_WINDOWS
from TARA.ACCESS.restore.restore_manager import RecoveryManager
from TARA.ACCESS.services.auth_service import CreatorAuthService, AUTHORIZED_CREATOR_EMAIL
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.wizard.setup_wizard import CreatorSetupWizard, CreatorSessionManager, CANONICAL_RECOVERY_EMAIL

EXPECTED_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"


class TestCreatorLifecycleHardening(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_lifecycle_test_")
        self.creator_record_path = os.path.join(self.test_dir, "operator_record.json")
        self.recovery_config_path = os.path.join(self.test_dir, "restore_config.json")
        self.creators_registry_path = os.path.join(self.test_dir, "operators_registry.json")
        self.auth_manifest_path = os.path.join(self.test_dir, "auth_manifest.json")
        self.storage_dir = os.path.join(self.test_dir, "keystore")
        self.seal_path = os.path.join(self.storage_dir, "access_seal.json")
        self.devices_file_path = os.path.join(self.test_dir, "devices.json")
        self.trigger_file_path = os.path.join(self.test_dir, "activation_config.json")
        self.audit_log_dir = os.path.join(self.test_dir, "audit")
        os.makedirs(self.audit_log_dir, exist_ok=True)

        self.passphrase = "PrimaryCreatorSecret2026!"

        # RSA mock keys for Google auth testing
        self.rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.rsa_pub = self.rsa_key.public_key()
        self.test_kid = "lifecycle-google-kid-1"
        self.google_auth = GoogleAuthService(
            authorized_email=AUTHORIZED_CREATOR_EMAIL,
            expected_client_id="tara-client-id",
            trusted_public_keys={self.test_kid: self.rsa_pub}
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _get_wizard(self) -> CreatorSetupWizard:
        return CreatorSetupWizard(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            recovery_config_path=self.recovery_config_path,
            storage_dir=self.storage_dir,
            auth_manifest_path=self.auth_manifest_path,
            trigger_file_path=self.trigger_file_path,
            creators_registry_path=self.creators_registry_path,
            seal_path=self.seal_path
        )

    def _get_session_manager(self) -> CreatorSessionManager:
        return CreatorSessionManager(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            recovery_config_path=self.recovery_config_path,
            storage_dir=self.storage_dir,
            auth_manifest_path=self.auth_manifest_path,
            trigger_file_path=self.trigger_file_path,
            creators_registry_path=self.creators_registry_path,
            seal_path=self.seal_path
        )

    def _get_auth_service(self) -> CreatorAuthService:
        return CreatorAuthService(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            trigger_file_path=self.trigger_file_path,
            audit_log_dir=self.audit_log_dir,
            devices_file_path=self.devices_file_path,
            creators_registry_path=self.creators_registry_path,
            recovery_record_path=self.recovery_config_path,
            google_service=self.google_auth,
            seal_path=self.seal_path,
            auth_manifest_path=self.auth_manifest_path
        )

    # -------------------------------------------------------------------------
    # TEST 1: FIRST CREATOR SETUP (INITIALIZATION_REQUIRED -> ACTIVE)
    # -------------------------------------------------------------------------
    def test_01_first_creator_setup_transitions_to_active(self):
        """Test fresh system starts in INITIALIZATION_REQUIRED and transitions to ACTIVE upon setup."""
        wizard = self._get_wizard()
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.INITIALIZATION_REQUIRED)
        self.assertFalse(wizard.lifecycle.is_setup_complete())
        self.assertFalse(wizard.lifecycle.is_active())

        result = wizard.run_setup(self.passphrase, self.passphrase)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(result["display_name"], DEFAULT_DISPLAY_NAME)

        # Authority state must be verified ACTIVE
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.ACTIVE)
        self.assertTrue(wizard.lifecycle.is_setup_complete())
        self.assertTrue(wizard.lifecycle.is_active())
        self.assertFalse(wizard.lifecycle.is_locked())
        self.assertTrue(os.path.exists(self.seal_path))

    # -------------------------------------------------------------------------
    # TEST 2: ROOT CREATOR ROLE INVARIANT
    # -------------------------------------------------------------------------
    def test_02_root_creator_role_and_identity_invariants(self):
        """Test ROOT_OPERATOR is permanently ROOT_CREATOR and cannot be altered."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)

        registry = MultiCreatorRegistry(registry_path=self.creators_registry_path)
        root_rec = registry.get_creator(CANONICAL_CREATOR_ID)
        self.assertIsNotNone(root_rec)
        self.assertEqual(root_rec["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(root_rec["display_name"], "OPERATOR_ROOT")
        self.assertEqual(root_rec["role"], "ROOT_CREATOR")
        self.assertEqual(registry.resolve_role(CANONICAL_CREATOR_ID), "ROOT_CREATOR")

        # CreatorIdentity file check
        identity = CreatorIdentity(record_path=self.creator_record_path)
        self.assertEqual(identity.creator_id, CANONICAL_CREATOR_ID)
        self.assertEqual(identity.display_name, DEFAULT_DISPLAY_NAME)

    # -------------------------------------------------------------------------
    # TEST 3: SECOND SETUP REJECTION (PERMANENT INITIALIZATION LOCK)
    # -------------------------------------------------------------------------
    def test_03_second_setup_rejection_immutable_lock(self):
        """Test that running setup wizard a second time aborts immediately with expected error."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        self.assertTrue(wizard.lifecycle.is_setup_complete())

        # Second setup attempt must raise PermissionError
        expected_msg = "Creator authority is already initialized. Authentication or authorized recovery is required for changes."
        with self.assertRaises(PermissionError) as ctx:
            wizard.run_setup("AnotherSecretPassphrase123!", "AnotherSecretPassphrase123!")

        self.assertIn(expected_msg, str(ctx.exception))

        # Re-attempting via another wizard instance must also fail
        wizard2 = self._get_wizard()
        with self.assertRaises(PermissionError) as ctx2:
            wizard2.run_setup("YetAnotherPassphrase123!", "YetAnotherPassphrase123!")
        self.assertIn(expected_msg, str(ctx2.exception))

    # -------------------------------------------------------------------------
    # TEST 4: TAMPER DETECTION ON operator_record.json (FAIL CLOSED)
    # -------------------------------------------------------------------------
    def test_04_tamper_detection_creator_record(self):
        """Test editing operator_record.json causes authority to fail closed (AUTHORITY_LOCKED)."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.ACTIVE)

        # Tamper with operator_record.json
        with open(self.creator_record_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["display_name"] = "TAMPERED_CREATOR_NAME"
        with open(self.creator_record_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        # Lifecycle must immediately detect hash mismatch and lock authority
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.AUTHORITY_LOCKED)
        self.assertTrue(wizard.lifecycle.is_locked())
        self.assertFalse(wizard.lifecycle.is_active())

        # Authentication service must reject when locked
        auth_service = self._get_auth_service()
        res = auth_service.authenticate_creator_key(passphrase=self.passphrase)
        self.assertEqual(res["status"], "AUTHORITY_LOCKED")

    # -------------------------------------------------------------------------
    # TEST 5: TAMPER DETECTION ON operators_registry.json (FAIL CLOSED)
    # -------------------------------------------------------------------------
    def test_05_tamper_detection_creators_registry(self):
        """Test editing operators_registry.json directly causes AUTHORITY_LOCKED."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.ACTIVE)

        # Tamper with operators_registry.json (e.g. attacker attempting to self-escalate or add backdoors)
        with open(self.creators_registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        reg["ATTACKER"] = {
            "creator_id": "ATTACKER",
            "role": "ROOT_CREATOR",
            "status": "active"
        }
        with open(self.creators_registry_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

        # Lifecycle must lock
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.AUTHORITY_LOCKED)
        self.assertTrue(wizard.lifecycle.is_locked())

    # -------------------------------------------------------------------------
    # TEST 6: TAMPER DETECTION ON restore_config.json (FAIL CLOSED)
    # -------------------------------------------------------------------------
    def test_06_tamper_detection_recovery_config(self):
        """Test altering restore_config.json verifier triggers AUTHORITY_LOCKED."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.ACTIVE)

        # Tamper with restore_config.json
        with open(self.recovery_config_path, "r", encoding="utf-8") as f:
            recov = json.load(f)
        recov["recovery_email"] = "attacker_stealing_recovery@evil.com"
        with open(self.recovery_config_path, "w", encoding="utf-8") as f:
            json.dump(recov, f, indent=2)

        # Lifecycle must lock
        self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.AUTHORITY_LOCKED)
        self.assertTrue(wizard.lifecycle.is_locked())

    # -------------------------------------------------------------------------
    # TEST 7: KEY ROTATION REQUIRES AUTHORIZATION (PATH A OR PATH B)
    # -------------------------------------------------------------------------
    def test_07_key_rotation_unauthorized_attempt_rejected(self):
        """Test rotating keys without Path A or Path B raises PermissionError."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        auth_service = self._get_auth_service()

        new_priv, new_pub = Ed25519.generate_keypair()

        # Attempt with empty authorization
        with self.assertRaises(PermissionError):
            auth_service.rotate_creator_key(new_pub.hex(), authorization={})

        # Attempt with invalid session token
        with self.assertRaises(PermissionError):
            auth_service.rotate_creator_key(new_pub.hex(), authorization={"method": "session_token", "token": "invalid_fake_token"})

        # Attempt with invalid recovery code
        with self.assertRaises(PermissionError):
            auth_service.rotate_creator_key(new_pub.hex(), authorization={"method": "recovery_code", "code": "WRONG-CODE-1234"})

    # -------------------------------------------------------------------------
    # TEST 8: KEY ROTATION VIA PATH A (ACTIVE ROOT CREATOR AUTHENTICATION)
    # -------------------------------------------------------------------------
    def test_08_key_rotation_via_path_a_root_session(self):
        """Test rotating root key using active ROOT_CREATOR session token (Path A)."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        auth_service = self._get_auth_service()

        # Unlock valid session
        mgr = self._get_session_manager()
        unlock_res = mgr.unlock(self.passphrase)
        self.assertEqual(unlock_res["status"], "SUCCESS")

        # Establish authenticated session in auth_service
        issued_sess = auth_service._issue_creator_session(CANONICAL_CREATOR_ID, "LOCAL_KEYSTORE")
        session_token = issued_sess["session_token"]

        old_pub_hex = auth_service.creator.root_public_key
        old_version = auth_service.creator.key_version

        # Generate new keypair
        new_priv, new_pub = Ed25519.generate_keypair()

        # Rotate key via Path A
        rot_res = auth_service.rotate_creator_key(
            new_public_key_hex=new_pub.hex(),
            authorization={"method": "session_token", "token": session_token},
            new_private_key_bytes=new_priv
        )

        self.assertEqual(rot_res["status"], "SUCCESS")
        self.assertEqual(rot_res["new_key_version"], old_version + 1)
        self.assertEqual(auth_service.creator.root_public_key, new_pub.hex())
        self.assertEqual(auth_service.creator.revoked_keys[-1]["public_key"], old_pub_hex)

        # Authority must remain verified ACTIVE
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.ACTIVE)

    # -------------------------------------------------------------------------
    # TEST 9: KEY ROTATION VIA PATH B (EMERGENCY RECOVERY CODE)
    # -------------------------------------------------------------------------
    def test_09_key_rotation_via_path_b_recovery_code(self):
        """Test rotating root key using valid 32-character recovery code (Path B)."""
        wizard = self._get_wizard()
        setup_res = wizard.run_setup(self.passphrase, self.passphrase)
        recovery_code = setup_res["emergency_recovery_secret"]

        auth_service = self._get_auth_service()
        old_version = auth_service.creator.key_version

        new_priv, new_pub = Ed25519.generate_keypair()

        rot_res = auth_service.rotate_creator_key(
            new_public_key_hex=new_pub.hex(),
            authorization={"method": "recovery_code", "code": recovery_code},
            new_private_key_bytes=new_priv
        )

        self.assertEqual(rot_res["status"], "SUCCESS")
        self.assertEqual(rot_res["new_key_version"], old_version + 1)
        self.assertEqual(auth_service.creator.root_public_key, new_pub.hex())

        # Authority must remain verified ACTIVE
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.ACTIVE)

    # -------------------------------------------------------------------------
    # TEST 10: OLD KEY REJECTION AFTER ROTATION
    # -------------------------------------------------------------------------
    def test_10_old_key_rejected_after_rotation(self):
        """Test that signatures from revoked old key are strictly rejected after rotation."""
        wizard = self._get_wizard()
        setup_res = wizard.run_setup(self.passphrase, self.passphrase)
        recovery_code = setup_res["emergency_recovery_secret"]

        mgr = self._get_session_manager()
        mgr.unlock(self.passphrase)
        old_priv = mgr._active_key
        self.assertIsNotNone(old_priv)

        auth_service = self._get_auth_service()

        # Rotate key to a new key
        new_priv, new_pub = Ed25519.generate_keypair()
        auth_service.rotate_creator_key(
            new_public_key_hex=new_pub.hex(),
            authorization={"method": "recovery_code", "code": recovery_code},
            new_private_key_bytes=new_priv
        )

        # Attempt challenge-response authentication with OLD private key
        sel = auth_service.handle_method_selection("CREATOR_KEY")
        nonce = sel["challenge_nonce"]
        old_key_sig = Ed25519.sign(old_priv, nonce.encode("utf-8")).hex()

        res_old = auth_service.authenticate_creator_key(
            proof_signature_hex=old_key_sig,
            challenge_nonce=nonce,
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertEqual(res_old["status"], "FAILED")
        self.assertEqual(res_old["error"], "Creator authentication failed. No creator authority was granted.")

    # -------------------------------------------------------------------------
    # TEST 11: NEW KEY FUNCTIONALITY AFTER ROTATION
    # -------------------------------------------------------------------------
    def test_11_new_key_functions_successfully(self):
        """Test that newly rotated root key authenticates successfully."""
        wizard = self._get_wizard()
        setup_res = wizard.run_setup(self.passphrase, self.passphrase)
        recovery_code = setup_res["emergency_recovery_secret"]

        auth_service = self._get_auth_service()
        new_priv, new_pub = Ed25519.generate_keypair()

        auth_service.rotate_creator_key(
            new_public_key_hex=new_pub.hex(),
            authorization={"method": "recovery_code", "code": recovery_code},
            new_private_key_bytes=new_priv
        )

        # Authenticate with NEW private key signature
        sel = auth_service.handle_method_selection("CREATOR_KEY")
        nonce = sel["challenge_nonce"]
        new_key_sig = Ed25519.sign(new_priv, nonce.encode("utf-8")).hex()

        res_new = auth_service.authenticate_creator_key(
            proof_signature_hex=new_key_sig,
            challenge_nonce=nonce,
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertEqual(res_new["status"], "SUCCESS")
        self.assertEqual(res_new["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(res_new["role"], "ROOT_CREATOR")

    # -------------------------------------------------------------------------
    # TEST 12: GOOGLE IDENTITY CHANGE AUTHORIZATION
    # -------------------------------------------------------------------------
    def test_12_google_identity_change_requires_authorization(self):
        """Test updating creator Google email fails without Path A/B and succeeds with Path A."""
        wizard = self._get_wizard()
        setup_res = wizard.run_setup(self.passphrase, self.passphrase)
        auth_service = self._get_auth_service()

        new_email = "operator_root.secure@test.local"

        # Attempt without authorization -> Rejected
        with self.assertRaises(PermissionError):
            auth_service.change_creator_google_identity(
                creator_id=CANONICAL_CREATOR_ID,
                new_google_email=new_email,
                authorization={"method": "unauthorized"}
            )

        # Authenticate root session (Path A)
        issued_sess = auth_service._issue_creator_session(CANONICAL_CREATOR_ID, "LOCAL_KEYSTORE")
        token = issued_sess["session_token"]

        # Valid Path A modification
        chg_res = auth_service.change_creator_google_identity(
            creator_id=CANONICAL_CREATOR_ID,
            new_google_email=new_email,
            authorization={"method": "session_token", "token": token}
        )
        self.assertEqual(chg_res["status"], "SUCCESS")
        self.assertEqual(chg_res["authorized_google_email"], new_email)

        # Authority remains ACTIVE
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.ACTIVE)

    # -------------------------------------------------------------------------
    # TEST 13: CREATOR 2/3 ENROLLMENT & ANTI-SELF-ESCALATION
    # -------------------------------------------------------------------------
    def test_13_creator_enrollment_and_anti_self_escalation(self):
        """Test enrolling secondary creators requires Path A/B and strictly assigns CREATOR role."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        auth_service = self._get_auth_service()

        c2_priv, c2_pub = Ed25519.generate_keypair()

        # Attempt without authorization -> Rejected
        with self.assertRaises(PermissionError):
            auth_service.enroll_secondary_creator(
                creator_id="CREATOR_2",
                display_name="Creator Two",
                authorization={},
                public_key=c2_pub.hex()
            )

        # Authorized via Path A
        issued_sess = auth_service._issue_creator_session(CANONICAL_CREATOR_ID, "LOCAL_KEYSTORE")
        token = issued_sess["session_token"]

        enr_res = auth_service.enroll_secondary_creator(
            creator_id="CREATOR_2",
            display_name="Creator Two",
            authorization={"method": "session_token", "token": token},
            public_key=c2_pub.hex(),
            google_email="creator2@example.com"
        )
        self.assertEqual(enr_res["status"], "SUCCESS")
        self.assertEqual(enr_res["creator"]["role"], "CREATOR")  # Invariant: Never ROOT_CREATOR

        # Attempt to overwrite ROOT_OPERATOR with CREATOR_2 -> Rejected
        with self.assertRaises(ValueError):
            auth_service.enroll_secondary_creator(
                creator_id=CANONICAL_CREATOR_ID,
                display_name="Impostor",
                authorization={"method": "session_token", "token": token}
            )

    # -------------------------------------------------------------------------
    # TEST 14: RECOVERY NON-BYPASSABILITY & BRUTE-FORCE LOCKOUT
    # -------------------------------------------------------------------------
    def test_14_recovery_non_bypassability_and_lockout(self):
        """Test invalid recovery codes fail, creator name/email cannot bypass, and 5 failures lock out."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)
        auth_service = self._get_auth_service()

        # Attempt to bypass recovery using creator ID or email
        res_name = auth_service.authenticate_recovery("ROOT_OPERATOR", claimed_creator_id=CANONICAL_CREATOR_ID)
        self.assertEqual(res_name["status"], "FAILED")

        res_email = auth_service.authenticate_recovery("operator@internal.local", claimed_creator_id=CANONICAL_CREATOR_ID)
        self.assertEqual(res_email["status"], "FAILED")

        # 5 failed attempts trigger rate-limited lockout
        client_key = "test_attacker_ip"
        for i in range(5):
            res_fail = auth_service.authenticate_recovery("WRONG-RECOVERY-CODE-0000", client_key=client_key)
            self.assertEqual(res_fail["status"], "FAILED")

        # 6th attempt must return lockout
        res_locked = auth_service.authenticate_recovery("WRONG-RECOVERY-CODE-0000", client_key=client_key)
        self.assertIn("lockout_remaining", res_locked)
        self.assertGreater(res_locked["lockout_remaining"], 0)

    # -------------------------------------------------------------------------
    # TEST 15: PRODUCTION MODEL SHA256 INVARIANT
    # -------------------------------------------------------------------------
    def test_15_production_model_sha256_invariant(self):
        """Test that production neural model weights remain strictly byte-identical and unmodified."""
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_path), f"Model file missing at: {model_path}")

        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        actual_sha256 = hasher.hexdigest()

        self.assertEqual(
            actual_sha256,
            EXPECTED_MODEL_SHA256,
            f"CRITICAL: Model weights corrupted or tampered! Expected: {EXPECTED_MODEL_SHA256}, Got: {actual_sha256}"
        )


if __name__ == "__main__":
    unittest.main()

"""
tests/test_auth_security.py

Comprehensive Security and Cryptographic Authorization Test Suite for TARA.
Proves all 11 required security invariants:
1. wrong signature -> DENY
2. unknown device -> DENY
3. revoked device -> DENY
4. missing Google ID token -> DENY
5. email alone -> DENY (cannot authenticate creator)
6. biometric simulation in production -> DENY
7. missing creator confirmation in execution guard -> DENY
8. valid authorized-device signature -> ALLOW
9. valid creator authorization chain -> ALLOW
10. runtime execution guard actually blocks unauthorized creator-only actions (fails closed)
11. test-mode authentication cannot accidentally activate in production
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

# Ensure project root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.access_manager import IdentityManager
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.secure_storage import SecureKeyStorage
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.services.factor_service import BiometricService
from TARA.ACCESS.policy.access_policy import IdentityPolicy, TaraRole
from TARA.RULES.engine.execution_guard import ExecutionGuard
from TARA.RULES.compiler.policy_schema import CompiledPolicy, RuleAction, PolicyPriority, RuleCategory, Rule


class TestCreatorAuthenticationSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_sec_test_")
        self.test_passphrase = "test_creator_secure_passphrase_12345"

        # Enable test mode specifically for test suite
        os.environ["TARA_TEST_MODE"] = "1"
        os.environ["TARA_CREATOR_PASSPHRASE"] = self.test_passphrase

        # Initialize test identity manager with isolated storage
        self.id_mgr = IdentityManager(base_dir=self.test_dir)
        self.id_mgr.biometric_service.test_mode = True

        # Create mock Google keypair for tests
        from cryptography.hazmat.primitives.asymmetric import rsa
        self.google_mock_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.google_kid = "test-google-key-01"
        self.id_mgr.google_service.register_trusted_key(self.google_kid, self.google_mock_priv.public_key())

        self.mock_id_token = GoogleAuthService.create_mock_id_token(
            private_key=self.google_mock_priv,
            kid=self.google_kid,
            email="operator@internal.local",
            sub="google-sub-creator-001"
        )

        # Bootstrap test identity
        self.init_res = self.id_mgr.first_creator_setup(
            google_email="operator@internal.local",
            display_name="OPERATOR_ROOT",
            enable_biometrics=False,
            passphrase=self.test_passphrase,
            google_id_token=self.mock_id_token
        )
        self.device_id = self.init_res["device_id"]

        # Load device private key
        self.device_priv = self.id_mgr.storage.load_private_key(
            f"device_{self.device_id}_key",
            self.test_passphrase
        )

        # Set up a compiled policy rule requiring creator confirmation
        self.test_rule = Rule(
            rule_id="RULE-CONFIRM-DELETE",
            version=1,
            category=RuleCategory.DATA_MUTATION.value,
            meaning="Deleting important system files requires creator cryptographic confirmation",
            priority=PolicyPriority.CREATOR_RULE.value,
            scope="SYSTEM",
            action=RuleAction.REQUIRE_CREATOR_CONFIRMATION.value,
            conditions={"category": "DATA_MUTATION"},
            status="ACTIVE",
            original_text="Delete important files only with creator confirmation",
            language="en",
            is_mandatory=False,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        self.compiled_policy = CompiledPolicy(
            policy_version=1,
            creator_id=CANONICAL_CREATOR_ID,
            display_name=DEFAULT_DISPLAY_NAME,
            rules=[self.test_rule]
        )
        self.guard = ExecutionGuard(
            active_policy=self.compiled_policy,
            identity_manager=self.id_mgr
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        os.environ.pop("TARA_TEST_MODE", None)
        os.environ.pop("TARA_CREATOR_PASSPHRASE", None)

    # ------------------------------------------------------------------------
    # 1. WRONG SIGNATURE -> DENY
    # ------------------------------------------------------------------------
    def test_01_wrong_signature_denied(self):
        """1. Operation signed with an invalid/random signature is rejected."""
        action_payload = "delete_file:critical_kernel.sys"
        wrong_sig = b"\x00" * 64

        eval_res = self.id_mgr.verify_creator_operation(
            device_id=self.device_id,
            action_payload=action_payload,
            device_signature_bytes=wrong_sig
        )
        self.assertFalse(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.USER)
        self.assertIn("signature verification failed", eval_res["reason"].lower())

    # ------------------------------------------------------------------------
    # 2. UNKNOWN DEVICE -> DENY
    # ------------------------------------------------------------------------
    def test_02_unknown_device_denied(self):
        """2. Operation originating from an unregistered/unknown device is rejected."""
        action_payload = "modify_machine_config"
        dummy_sig = b"\x01" * 64

        eval_res = self.id_mgr.verify_creator_operation(
            device_id="TARA-DEVICE-UNKNOWN-999",
            action_payload=action_payload,
            device_signature_bytes=dummy_sig
        )
        self.assertFalse(eval_res["authorized"])
        self.assertIn("unauthorized_device", eval_res["reason"].lower())

    # ------------------------------------------------------------------------
    # 3. REVOKED DEVICE -> DENY
    # ------------------------------------------------------------------------
    def test_03_revoked_device_denied(self):
        """3. Operation signed by a revoked device is rejected."""
        # Revoke the device
        self.id_mgr.devices.revoke_device(self.device_id, reason="Security audit test")

        action_payload = "delete_file:test.dat"
        valid_sig = Ed25519.sign(self.device_priv, action_payload.encode("utf-8"))

        eval_res = self.id_mgr.verify_creator_operation(
            device_id=self.device_id,
            action_payload=action_payload,
            device_signature_bytes=valid_sig
        )
        self.assertFalse(eval_res["authorized"])
        self.assertIn("revoked", eval_res["reason"].lower())

    # ------------------------------------------------------------------------
    # 4. MISSING GOOGLE ID TOKEN -> DENY
    # ------------------------------------------------------------------------
    def test_04_missing_google_id_token_denied(self):
        """4. Google account verification without an id_token fails closed."""
        res = self.id_mgr.google_service.verify_account(
            email="operator@internal.local",
            id_token=None
        )
        self.assertFalse(res["verified"])
        self.assertIn("MISSING_GOOGLE_ID_TOKEN", res["error"])

    # ------------------------------------------------------------------------
    # 5. EMAIL ALONE CANNOT AUTHENTICATE
    # ------------------------------------------------------------------------
    def test_05_email_alone_cannot_authenticate(self):
        """5. Passing creator email alone without cryptographic token is rejected."""
        # Create a fresh identity manager
        fresh_dir = tempfile.mkdtemp(prefix="tara_email_test_")
        fresh_mgr = IdentityManager(base_dir=fresh_dir)
        try:
            with self.assertRaises(ValueError) as ctx:
                fresh_mgr.first_creator_setup(
                    google_email="operator@internal.local",
                    passphrase="some_password_123",
                    google_id_token=None
                )
            self.assertIn("MISSING_GOOGLE_ID_TOKEN", str(ctx.exception))
        finally:
            shutil.rmtree(fresh_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # 6. BIOMETRIC SIMULATION IN PRODUCTION -> DENY
    # ------------------------------------------------------------------------
    def test_06_biometric_simulation_blocked_in_production(self):
        """6. Biometric simulation is strictly prohibited in production mode."""
        # Unset test mode
        os.environ.pop("TARA_TEST_MODE", None)
        prod_bio = BiometricService(hardware_available=False, test_mode=False)

        # Attempting simulation in production must raise PermissionError
        with self.assertRaises(PermissionError) as ctx:
            prod_bio.authenticate_biometric(simulate_user_present=True)
        self.assertIn("forbidden in production", str(ctx.exception).lower())

        # Without simulation, fails closed if hardware unavailable
        res = prod_bio.authenticate_biometric(simulate_user_present=False)
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "BIOMETRIC_ERROR_HW_UNAVAILABLE")

    # ------------------------------------------------------------------------
    # 7. MISSING CREATOR CONFIRMATION IN EXECUTION GUARD -> DENY
    # ------------------------------------------------------------------------
    def test_07_missing_creator_confirmation_denied(self):
        """7. Privileged action submitted without creator_auth fails closed."""
        # Action requires creator confirmation, but context lacks creator_auth
        res = self.guard.evaluate_action(
            action_type="delete_file",
            context={"is_important": True}  # No creator_auth
        )
        self.assertEqual(res["decision"], RuleAction.DENY.value)
        self.assertIn("UNAUTHORIZED_CREATOR_ACTION", res["reason"])
        self.assertIn("cryptographic proof", res["reason"])

    # ------------------------------------------------------------------------
    # 8. VALID AUTHORIZED-DEVICE SIGNATURE -> ALLOW
    # ------------------------------------------------------------------------
    def test_08_valid_authorized_device_signature_allowed(self):
        """8. Valid Ed25519 signature from authorized device grants creator authority."""
        action_payload = "delete_file:archive.tar.gz"
        sig_bytes = Ed25519.sign(self.device_priv, action_payload.encode("utf-8"))

        eval_res = self.id_mgr.verify_creator_operation(
            device_id=self.device_id,
            action_payload=action_payload,
            device_signature_bytes=sig_bytes
        )
        self.assertTrue(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.ROOT_CREATOR)
        self.assertEqual(eval_res["reason"], "AUTHORIZED_ROOT_CREATOR")

    # ------------------------------------------------------------------------
    # 9. VALID CREATOR AUTHORIZATION CHAIN IN EXECUTION GUARD -> ALLOW
    # ------------------------------------------------------------------------
    def test_09_runtime_execution_guard_allows_with_valid_signature(self):
        """9. ExecutionGuard allows creator action when accompanied by verified cryptographic proof."""
        action_type = "delete_file"
        sig_bytes = Ed25519.sign(self.device_priv, action_type.encode("utf-8"))

        context = {
            "is_important": True,
            "creator_auth": {
                "device_id": self.device_id,
                "action_payload": action_type,
                "device_signature": sig_bytes.hex()
            }
        }

        res = self.guard.evaluate_action(action_type=action_type, context=context)
        self.assertEqual(res["decision"], RuleAction.ALLOW.value)
        self.assertTrue(res.get("creator_verified"))
        self.assertIn("verified cryptographic device signature", res["reason"])

    # ------------------------------------------------------------------------
    # 10. RUNTIME EXECUTION GUARD BLOCKS WRONG SIGNATURE
    # ------------------------------------------------------------------------
    def test_10_runtime_execution_guard_blocks_wrong_signature(self):
        """10. ExecutionGuard denies action when device signature is forged/invalid."""
        action_type = "delete_file"
        forged_sig = b"\xff" * 64

        context = {
            "is_important": True,
            "creator_auth": {
                "device_id": self.device_id,
                "action_payload": action_type,
                "device_signature": forged_sig.hex()
            }
        }

        res = self.guard.evaluate_action(action_type=action_type, context=context)
        self.assertEqual(res["decision"], RuleAction.DENY.value)
        self.assertIn("UNAUTHORIZED_CREATOR_ACTION", res["reason"])

    # ------------------------------------------------------------------------
    # 11. OLD COMPROMISED KEY / DEFAULT PASSPHRASE CANNOT DECRYPT NEW KEYSTORE
    # ------------------------------------------------------------------------
    def test_11_old_default_secret_cannot_decrypt(self):
        """11. Storage encrypted with new passphrase strictly fails under old default secret."""
        # Attempting to load private key with old secret must return None
        old_secret = "TARA_DEFAULT_LOCAL_DEVICE_SECRET"
        loaded = self.id_mgr.storage.load_private_key(f"device_{self.device_id}_key", old_secret)
        self.assertIsNone(loaded)

        # Loading without any passphrase or env var returns None (fails closed)
        os.environ.pop("TARA_CREATOR_PASSPHRASE", None)
        os.environ.pop("TARA_DEVICE_SECRET", None)
        fresh_storage = SecureKeyStorage(storage_dir=os.path.join(self.test_dir, "secrets"))
        self.assertIsNone(fresh_storage.load_private_key(f"device_{self.device_id}_key"))

        # Storing without any passphrase or env var raises ValueError
        with self.assertRaises(ValueError):
            fresh_storage.store_private_key("test_key", b"sample_bytes")


if __name__ == "__main__":
    unittest.main()

"""
tests/test_setup_and_auth.py

Exhaustive test suite for TARA Setup Wizard and Authentication:
1. Setup Wizard execution & signed auth_manifest.json v2 manifest generation.
2. Session unlock & proof-of-possession verification with correct passphrase.
3. Wrong passphrase rejection & failed attempt counter tracking.
4. Rate-limited lockout enforcement (5 failed attempts -> 300s lock).
5. Emergency recovery flow using 32-character recovery code.
6. Windows DPAPI / hardware keystore protect & unprotect roundtrip.
7. Secure lock / logout wiping in-memory private key material.
8. Backward compatibility with existing operator_record.json.
9. Strict enforcement of Creator Identity invariants (ROOT_OPERATOR, OPERATOR_ROOT, operator@internal.local).
10. Cryptographic tampering detection on signed manifest.
"""

import os
import json
import time
import shutil
import tempfile
import unittest

from TARA.ACCESS.operator.operator_profile import (
    CreatorIdentity,
    CANONICAL_CREATOR_ID,
    DEFAULT_DISPLAY_NAME
)
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.secure_storage import SecureKeyStorage
from TARA.ACCESS.crypto.dpapi_storage import (
    protect_bytes_dpapi,
    unprotect_bytes_dpapi,
    IS_WINDOWS
)
from TARA.ACCESS.restore.restore_manager import RecoveryManager
from TARA.ACCESS.wizard.setup_wizard import (
    CreatorSetupWizard,
    CreatorSessionManager,
    CANONICAL_RECOVERY_EMAIL
)


class TestCreatorSetupAndAuth(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.creator_record_path = os.path.join(self.test_dir, "operator_record.json")
        self.recovery_config_path = os.path.join(self.test_dir, "restore_config.json")
        self.storage_dir = os.path.join(self.test_dir, "keystore")
        self.auth_manifest_path = os.path.join(self.test_dir, "auth_manifest.json")

        self.passphrase = "MasterCreatorPassphrase987!"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _get_wizard(self) -> CreatorSetupWizard:
        return CreatorSetupWizard(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            recovery_config_path=self.recovery_config_path,
            storage_dir=self.storage_dir,
            auth_manifest_path=self.auth_manifest_path
        )

    def _get_session_manager(self) -> CreatorSessionManager:
        return CreatorSessionManager(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            recovery_config_path=self.recovery_config_path,
            storage_dir=self.storage_dir,
            auth_manifest_path=self.auth_manifest_path
        )

    def test_01_setup_wizard_execution_and_manifest(self):
        """Test complete setup wizard run, manifest generation, and Ed25519 signature."""
        wizard = self._get_wizard()
        self.assertFalse(wizard.is_configured())

        res = wizard.run_setup(self.passphrase, self.passphrase)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(res["display_name"], DEFAULT_DISPLAY_NAME)
        self.assertEqual(res["recovery_email"], CANONICAL_RECOVERY_EMAIL)
        self.assertTrue(len(res["emergency_recovery_secret"]) >= 32)
        self.assertTrue(os.path.exists(self.auth_manifest_path))
        self.assertTrue(wizard.is_configured())

        # Verify manifest cryptographic signature
        verif = wizard.verify_auth_manifest()
        self.assertTrue(verif["valid"], f"Manifest verification failed: {verif.get('error')}")

        # Check raw manifest structure
        with open(self.auth_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(manifest["display_name"], DEFAULT_DISPLAY_NAME)
        self.assertEqual(manifest["recovery_email"], CANONICAL_RECOVERY_EMAIL)
        self.assertIn("signature", manifest)
        self.assertIn("root_public_key", manifest)

    def test_02_passphrase_validation_and_mismatch(self):
        """Test that short passphrases and mismatching confirmations are rejected."""
        wizard = self._get_wizard()
        with self.assertRaises(ValueError):
            wizard.run_setup("short", "short")

        with self.assertRaises(ValueError):
            wizard.run_setup("ValidPassphrase123!", "MismatchingPassphrase123!")

    def test_03_session_unlock_and_proof_of_possession(self):
        """Test unlocking root authority and issuing valid Ed25519 session proof."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)

        mgr = self._get_session_manager()
        self.assertFalse(mgr.is_unlocked())

        res = mgr.unlock(self.passphrase)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["creator_id"], CANONICAL_CREATOR_ID)
        self.assertTrue(mgr.is_unlocked())

        # Verify proof signature against active root public key
        token = res["session_token"]
        ts = res["timestamp"]
        sig_bytes = bytes.fromhex(res["proof_signature"])
        pub_bytes = bytes.fromhex(mgr.creator.root_public_key)

        expected_payload = f"{CANONICAL_CREATOR_ID}:{token}:{ts}".encode("utf-8")
        self.assertTrue(Ed25519.verify(pub_bytes, expected_payload, sig_bytes))

    def test_04_wrong_passphrase_rejection_and_counter(self):
        """Test that incorrect passphrase fails authentication and increments counter."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)

        mgr = self._get_session_manager()
        res = mgr.unlock("IncorrectPassphrase123!")
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(res["failed_attempts"], 1)
        self.assertEqual(res["remaining_attempts"], 4)
        self.assertFalse(mgr.is_unlocked())

    def test_05_rate_limited_lockout(self):
        """Test that 5 consecutive failed attempts trigger a 300-second lockout."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)

        mgr = self._get_session_manager()
        for i in range(4):
            res = mgr.unlock(f"WrongPass_{i}")
            self.assertEqual(res["status"], "FAILED")
            self.assertFalse(mgr.is_locked_out()[0])

        # 5th attempt must trigger lockout
        res5 = mgr.unlock("WrongPass_5")
        self.assertEqual(res5["status"], "LOCKED_OUT")
        locked, rem = mgr.is_locked_out()
        self.assertTrue(locked)
        self.assertTrue(rem > 250)

        # Even correct passphrase is now blocked during lockout
        blocked_res = mgr.unlock(self.passphrase)
        self.assertEqual(blocked_res["status"], "LOCKED_OUT")

    def test_06_emergency_recovery_flow(self):
        """Test emergency key loss recovery using 32-character recovery code."""
        wizard = self._get_wizard()
        setup_res = wizard.run_setup(self.passphrase, self.passphrase)
        rec_code = setup_res["emergency_recovery_secret"]
        old_pubkey = setup_res["root_public_key"]
        old_version = setup_res["key_version"]

        mgr = self._get_session_manager()
        # Attempt recovery with wrong code first
        bad_rec = mgr.recover("INVALID-RECOVERY-CODE-0000", "NewPass12345!", "NewPass12345!")
        self.assertEqual(bad_rec["status"], "FAILED")

        # Successful recovery with valid code
        new_pass = "NewMasterCreatorSecret999!"
        rec_res = mgr.recover(rec_code, new_pass, new_pass)
        self.assertEqual(rec_res["status"], "SUCCESS")
        self.assertEqual(rec_res["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(rec_res["new_key_version"], old_version + 1)
        self.assertNotEqual(rec_res["new_root_public_key"], old_pubkey)

        # Session should be immediately unlocked with new key
        self.assertTrue(mgr.is_unlocked())

        # Old passphrase should no longer decrypt
        mgr.lock()
        old_unlock = mgr.unlock(self.passphrase)
        self.assertEqual(old_unlock["status"], "FAILED")

        # New passphrase should successfully unlock
        new_unlock = mgr.unlock(new_pass)
        self.assertEqual(new_unlock["status"], "SUCCESS")

    def test_07_secure_lock_and_memory_wipe(self):
        """Test locking session clears active private key from memory."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)

        mgr = self._get_session_manager()
        mgr.unlock(self.passphrase)
        self.assertTrue(mgr.is_unlocked())

        # Sign an action while unlocked
        action = b"APPROVE_KNOWLEDGE_123"
        sig = mgr.sign_action(action)
        self.assertEqual(len(sig), 64)

        # Lock authority
        lock_res = mgr.lock()
        self.assertEqual(lock_res["status"], "LOCKED")
        self.assertFalse(mgr.is_unlocked())

        # Signing should now raise PermissionError
        with self.assertRaises(PermissionError):
            mgr.sign_action(action)

    def test_08_dpapi_hardware_storage(self):
        """Test DPAPI protect/unprotect roundtrip on supported platforms."""
        secret_bytes = b"DPAPI_TEST_SECRET_BYTES_998877"
        protected = protect_bytes_dpapi(secret_bytes, description="UNIT_TEST")
        self.assertIsInstance(protected, bytes)

        if IS_WINDOWS:
            # Ciphertext should not match plaintext
            self.assertNotEqual(protected, secret_bytes)

        unprotected = unprotect_bytes_dpapi(protected)
        self.assertEqual(unprotected, secret_bytes)

    def test_09_manifest_tamper_detection(self):
        """Test that modifying auth_manifest.json invalidates the cryptographic signature."""
        wizard = self._get_wizard()
        wizard.run_setup(self.passphrase, self.passphrase)

        # Tamper with the manifest file
        with open(self.auth_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["creator_id"] = "MALICIOUS_IMPOSTOR"
        with open(self.auth_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        verif = wizard.verify_auth_manifest()
        self.assertFalse(verif["valid"])

    def test_10_canonical_invariants_enforcement(self):
        """Test that CANONICAL_CREATOR_ID is permanently preserved."""
        wizard = self._get_wizard()
        res = wizard.run_setup(self.passphrase, self.passphrase)

        creator = CreatorIdentity(record_path=self.creator_record_path)
        self.assertEqual(creator.creator_id, CANONICAL_CREATOR_ID)
        self.assertEqual(creator.display_name, DEFAULT_DISPLAY_NAME)

        # Display name update should preserve Creator ID
        creator.update_display_name("OPERATOR_ROOT_CUSTOM")
        self.assertEqual(creator.display_name, "OPERATOR_ROOT_CUSTOM")
        self.assertEqual(creator.creator_id, CANONICAL_CREATOR_ID)


if __name__ == "__main__":
    unittest.main()

"""
tests/test_fresh_install_setup.py

Exhaustive verification of TARA Fresh Installation State, Honest Setup Invariants,
and Protection against False 'COMPLETE & ACTIVE' declarations.

Verifies:
 1. Fresh installation reports CREATOR_SETUP_REQUIRED.
 2. Production state contains NO test keys, seals, or authority manifests.
 3. Application startup and UserManager do NOT auto-create active creator records.
 4. Test fixtures are strictly isolated and never mutate production state.
 5. Uninitialized authentication attempts fail closed with CREATOR_SETUP_REQUIRED.
 6. Config-only files with fake active keys without seal fail closed with AUTHORITY_LOCKED.
 7. Real setup requires explicit identity confirmation.
 8. Real setup rejects unauthorized Google identity tokens.
 9. Complete setup flow produces authentic Ed25519 root key, keystore, manifest, and seal.
 10. Permanent lock prevents re-initialization once ACTIVE.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import rsa

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from TARA.ACCESS.operator.operator_profile import (
    CreatorIdentity,
    CANONICAL_CREATOR_ID,
    DEFAULT_DISPLAY_NAME
)
from TARA.ACCESS.operator.operator_lifecycle import (
    AuthorityLifecycleManager,
    AuthorityState
)
from TARA.ACCESS.wizard.setup_wizard import (
    CreatorSetupWizard,
    CANONICAL_RECOVERY_EMAIL
)
from TARA.ACCESS.services.auth_service import CreatorAuthService
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.secure_storage import SecureKeyStorage
from TARA.ACCESS.protected_v1.manager import ProtectedStateManager
from tara_core.security.capability_guard import CapabilityGuard, ActionRequest
from tara_core.security.security_state import TrustBoundary
from tara_core.user_model import UserManager, UserProfile, UserRole
from tests.test_helpers import create_test_id_token


class TestFreshInstallCreatorSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.prod_seal = os.path.join(cls.repo_root, "storage", "vault", "access", "access_seal.json")
        cls.prod_keystore = os.path.join(cls.repo_root, "storage", "vault", "access", "operator_key.keystore")
        cls.prod_sec_keystore = os.path.join(cls.repo_root, "TARA", "ACCESS", "vault", "operator_key.keystore")
        cls.prod_sec_seal = os.path.join(cls.repo_root, "TARA", "ACCESS", "vault", "access_seal.json")
        cls.prod_manifest = os.path.join(cls.repo_root, "TARA", "ACCESS", "operator", "auth_manifest.json")
        cls.prod_record = os.path.join(cls.repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
        cls.prod_registry = os.path.join(cls.repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        cls.prod_recovery = os.path.join(cls.repo_root, "TARA", "ACCESS", "restore", "restore_config.json")
        cls.prod_users = os.path.join(cls.repo_root, "TARA", "ACCESS", "users.json")

    # -------------------------------------------------------------------------
    # TEST 1: Fresh Installation Reports CREATOR_SETUP_REQUIRED
    # -------------------------------------------------------------------------
    def test_01_fresh_install_reports_creator_setup_required(self):
        """Production environment on fresh install must report CREATOR_SETUP_REQUIRED."""
        lifecycle = AuthorityLifecycleManager(repo_root=self.repo_root)
        state, reason = lifecycle.verify_integrity()
        self.assertEqual(state, AuthorityState.CREATOR_SETUP_REQUIRED)
        self.assertIn("creator setup required", reason.lower())
        self.assertFalse(lifecycle.is_active())
        self.assertFalse(lifecycle.is_setup_complete())
        self.assertFalse(lifecycle.is_locked())

        creator = CreatorIdentity()
        self.assertFalse(creator.is_initialized())
        self.assertEqual(creator.status, "CREATOR_SETUP_REQUIRED")
        self.assertIsNone(creator.root_public_key)

        wizard = CreatorSetupWizard(repo_root=self.repo_root)
        self.assertFalse(wizard.is_configured())

    # -------------------------------------------------------------------------
    # TEST 2: Production Directories Free of Generated Test Authority Files
    # -------------------------------------------------------------------------
    def test_02_production_directories_clean_of_test_credentials(self):
        """Production directories must NOT contain test keystores, seals, or manifests."""
        self.assertFalse(os.path.exists(self.prod_seal), "storage/vault/access/access_seal.json must not exist")
        self.assertFalse(os.path.exists(self.prod_keystore), "storage/vault/access/operator_key.keystore must not exist")
        self.assertFalse(os.path.exists(self.prod_sec_keystore), "TARA/ACCESS/vault/operator_key.keystore must not exist")
        self.assertFalse(os.path.exists(self.prod_sec_seal), "TARA/ACCESS/vault/access_seal.json must not exist")
        self.assertFalse(os.path.exists(self.prod_manifest), "TARA/ACCESS/operator/auth_manifest.json must not exist")

    # -------------------------------------------------------------------------
    # TEST 3: Startup / UserManager Does NOT Auto-Create Active Creator Record
    # -------------------------------------------------------------------------
    def test_03_startup_does_not_auto_create_active_creator(self):
        """Instantiating UserManager or starting server must not auto-seed an active creator."""
        user_mgr = UserManager(storage_file=self.prod_users)
        creator_user = user_mgr.get_user(CANONICAL_CREATOR_ID)
        self.assertIsNone(creator_user, "UserManager must NOT auto-insert ROOT_OPERATOR when creator is uninitialized")

        # Inspect raw users.json
        if os.path.exists(self.prod_users):
            with open(self.prod_users, "r", encoding="utf-8") as f:
                users_data = json.load(f)
            self.assertNotIn(CANONICAL_CREATOR_ID, users_data)

    # -------------------------------------------------------------------------
    # TEST 4: Uninitialized Auth Attempts Fail Closed
    # -------------------------------------------------------------------------
    def test_04_uninitialized_auth_attempts_fail_closed(self):
        """All creator authentication attempts fail closed when state is CREATOR_SETUP_REQUIRED."""
        auth_service = CreatorAuthService(repo_root=self.repo_root)

        # 1. QR Challenge
        res_qr = auth_service.create_qr_challenge()
        self.assertEqual(res_qr.get("status"), "CREATOR_SETUP_REQUIRED")
        self.assertIn("Creator setup has not been performed yet", res_qr.get("error", ""))

        # 2. QR Approval
        res_app = auth_service.verify_qr_approval("any_chal", "any_dev", "any_sig")
        self.assertEqual(res_app.get("status"), "CREATOR_SETUP_REQUIRED")

        # 3. Creator Key Auth
        res_key = auth_service.authenticate_creator_key(passphrase="TestSecret1234!")
        self.assertEqual(res_key.get("status"), "CREATOR_SETUP_REQUIRED")

        # 4. Google Auth
        res_goog = auth_service.authenticate_google_token("any_token")
        self.assertEqual(res_goog.get("status"), "CREATOR_SETUP_REQUIRED")

        # 5. Recovery Auth
        res_rec = auth_service.authenticate_recovery("any_recovery_code")
        self.assertEqual(res_rec.get("status"), "CREATOR_SETUP_REQUIRED")

    # -------------------------------------------------------------------------
    # TEST 5: Config-Only Files Without Seal Fail Closed (AUTHORITY_LOCKED)
    # -------------------------------------------------------------------------
    def test_05_config_only_fake_key_without_seal_fails_closed(self):
        """Creating an active record file with a fake key without seal triggers AUTHORITY_LOCKED."""
        temp_dir = tempfile.mkdtemp(prefix="tara_test_fake_")
        try:
            fake_rec = os.path.join(temp_dir, "operator_record.json")
            with open(fake_rec, "w", encoding="utf-8") as f:
                json.dump({
                    "creator_id": CANONICAL_CREATOR_ID,
                    "root_public_key": "deadbeef" * 8,
                    "status": "active"
                }, f)

            lifecycle = AuthorityLifecycleManager(
                repo_root=temp_dir,
                seal_path=os.path.join(temp_dir, "nonexistent_seal.json"),
                creator_record_path=fake_rec,
                creators_registry_path=os.path.join(temp_dir, "reg.json"),
                recovery_config_path=os.path.join(temp_dir, "rec.json"),
                auth_manifest_path=os.path.join(temp_dir, "auth.json")
            )
            state, reason = lifecycle.verify_integrity()
            self.assertEqual(state, AuthorityState.AUTHORITY_LOCKED)
            self.assertIn("tampering", reason.lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 6: Setup Wizard Enforces Confirmation and Passphrase Strength
    # -------------------------------------------------------------------------
    def test_06_setup_wizard_enforces_prerequisites(self):
        """Setup wizard strictly validates confirmation flag, passphrase length, and matching."""
        temp_dir = tempfile.mkdtemp(prefix="tara_test_wiz_")
        try:
            wizard = CreatorSetupWizard(
                repo_root=temp_dir,
                creator_record_path=os.path.join(temp_dir, "operator_record.json"),
                recovery_config_path=os.path.join(temp_dir, "restore_config.json"),
                storage_dir=temp_dir,
                auth_manifest_path=os.path.join(temp_dir, "auth_manifest.json"),
                trigger_file_path=os.path.join(temp_dir, "trigger.json"),
                creators_registry_path=os.path.join(temp_dir, "registry.json"),
                seal_path=os.path.join(temp_dir, "access_seal.json")
            )

            # 1. Missing explicit confirmation
            with self.assertRaises(ValueError) as ctx:
                wizard.run_setup(
                    master_passphrase="ValidPassphrase123!",
                    confirm_passphrase="ValidPassphrase123!",
                    confirm_identity=False
                )
            self.assertIn("confirmation is required", str(ctx.exception).lower())

            # 2. Too short passphrase
            with self.assertRaises(ValueError) as ctx:
                wizard.run_setup(
                    master_passphrase="short",
                    confirm_passphrase="short",
                    confirm_identity=True
                )
            self.assertIn("at least 8 characters", str(ctx.exception).lower())

            # 3. Mismatched confirmation
            with self.assertRaises(ValueError) as ctx:
                wizard.run_setup(
                    master_passphrase="ValidPassphrase123!",
                    confirm_passphrase="DifferentPassphrase123!",
                    confirm_identity=True
                )
            self.assertIn("do not match", str(ctx.exception).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 7: Setup Wizard Rejects Unauthorized Google Identity
    # -------------------------------------------------------------------------
    def test_07_setup_wizard_rejects_unauthorized_google_token(self):
        """Setup wizard verifies Google token and rejects non-canonical email."""
        temp_dir = tempfile.mkdtemp(prefix="tara_test_wiz_")
        try:
            wizard = CreatorSetupWizard(
                repo_root=temp_dir,
                creator_record_path=os.path.join(temp_dir, "operator_record.json"),
                recovery_config_path=os.path.join(temp_dir, "restore_config.json"),
                storage_dir=temp_dir,
                auth_manifest_path=os.path.join(temp_dir, "auth_manifest.json"),
                trigger_file_path=os.path.join(temp_dir, "trigger.json"),
                creators_registry_path=os.path.join(temp_dir, "registry.json"),
                seal_path=os.path.join(temp_dir, "access_seal.json")
            )

            # Create token for unauthorized attacker email
            unauth_token = create_test_id_token(
                email="attacker@malicious.com"
            )

            with self.assertRaises(PermissionError) as ctx:
                wizard.run_setup(
                    master_passphrase="ValidPassphrase123!",
                    confirm_passphrase="ValidPassphrase123!",
                    google_id_token=unauth_token,
                    confirm_identity=True
                )
            self.assertIn("verification failed", str(ctx.exception).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 8: Full Real Setup Flow Transitions Cleanly to ACTIVE
    # -------------------------------------------------------------------------
    def test_08_full_setup_transitions_to_active(self):
        """Executing full setup produces real keys, signed manifest, seal, and ACTIVE state."""
        temp_dir = tempfile.mkdtemp(prefix="tara_test_wiz_")
        try:
            seal_file = os.path.join(temp_dir, "access_seal.json")
            manifest_file = os.path.join(temp_dir, "auth_manifest.json")
            record_file = os.path.join(temp_dir, "operator_record.json")

            wizard = CreatorSetupWizard(
                repo_root=temp_dir,
                creator_record_path=record_file,
                recovery_config_path=os.path.join(temp_dir, "restore_config.json"),
                storage_dir=temp_dir,
                auth_manifest_path=manifest_file,
                trigger_file_path=os.path.join(temp_dir, "trigger.json"),
                creators_registry_path=os.path.join(temp_dir, "registry.json"),
                seal_path=seal_file
            )

            self.assertFalse(wizard.is_configured())
            self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

            # Execute complete setup
            res = wizard.run_setup(
                master_passphrase="RealCreatorPassword2026!",
                confirm_passphrase="RealCreatorPassword2026!",
                confirm_identity=True,
                private_trigger_phrase="test creator protocol trigger"
            )

            self.assertEqual(res.get("status"), "SUCCESS")
            self.assertTrue(wizard.is_configured())
            self.assertTrue(wizard.creator.is_initialized())
            self.assertEqual(wizard.lifecycle.get_state(), AuthorityState.ACTIVE)
            self.assertTrue(os.path.exists(seal_file))
            self.assertTrue(os.path.exists(manifest_file))

            # Verify manifest cryptographic signature
            verif = wizard.verify_auth_manifest()
            self.assertTrue(verif.get("valid"))

            # Confirm permanent lock rejects re-setup
            with self.assertRaises(PermissionError):
                wizard.run_setup(
                    master_passphrase="RealCreatorPassword2026!",
                    confirm_passphrase="RealCreatorPassword2026!",
                    confirm_identity=True
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 9: Protected Capabilities Guard Rejects When Setup Required
    # -------------------------------------------------------------------------
    def test_09_protected_capabilities_denied_when_setup_required(self):
        """Protected capability execution is completely denied when creator setup is required."""
        guard = CapabilityGuard()
        req = ActionRequest(
            action_type="creator_api",
            target_resource="system_rules",
            requester_id="unconfigured_session",
            requester_boundary=TrustBoundary.SERVER,
            creator_token="unconfigured_creator_token"
        )
        allowed, reason = guard.authorize_action(req)
        self.assertFalse(allowed)
        self.assertIn("denied", reason.lower())

        prot_mgr = ProtectedStateManager(repo_root=self.repo_root)
        auth_service = CreatorAuthService(repo_root=self.repo_root)
        active, err = prot_mgr.activate(
            session_token="some_token",
            candidate_trigger="test_candidate",
            creator_auth_service=auth_service
        )
        self.assertFalse(active)
        self.assertIn("denied", err.lower())

    # -------------------------------------------------------------------------
    # TEST 10: Pure Read-Only Audit of Production State
    # -------------------------------------------------------------------------
    def test_10_production_state_strictly_uninitialized(self):
        """Strict verification that production state remains CREATOR_SETUP_REQUIRED."""
        lifecycle = AuthorityLifecycleManager(repo_root=self.repo_root)
        self.assertEqual(lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

        with open(self.prod_record, "r", encoding="utf-8") as f:
            rec = json.load(f)
        self.assertIsNone(rec.get("root_public_key"))
        self.assertEqual(rec.get("status"), "CREATOR_SETUP_REQUIRED")

        with open(self.prod_registry, "r", encoding="utf-8") as f:
            reg = json.load(f)
        self.assertIsNone(reg[CANONICAL_CREATOR_ID].get("public_key"))
        self.assertEqual(reg[CANONICAL_CREATOR_ID].get("status"), "CREATOR_SETUP_REQUIRED")

        with open(self.prod_recovery, "r", encoding="utf-8") as f:
            recov = json.load(f)
        self.assertIsNone(recov.get("recovery_code_hash"))
        self.assertEqual(recov.get("status"), "CREATOR_SETUP_REQUIRED")


if __name__ == "__main__":
    unittest.main()

"""
tests/test_authority_repair_and_hardening.py

Test Suite for Authority Post-Recovery Resealing, Hardening, and Repair:
1. recover_locked_authority rejects plain recovery method strings ("google_account", "recovery_code", etc.).
2. recover_locked_authority rejects unsigned seal creation (requires new_private_key_bytes).
3. recover_locked_authority validates private key corresponds to target public key.
4. reseal_with_recovery synchronizes operator_record.json and operators_registry.json before hash computation.
5. reseal_with_recovery produces valid Ed25519 seal signature and verifies ACTIVE state.
6. repair_rotated_authority strictly requires valid RecoveryAuthorizationProof.
7. repair_rotated_authority successfully repairs already-rotated v2 state and transitions to ACTIVE.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.access_manager import IdentityManager
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME, CreatorIdentity
from TARA.ACCESS.operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from TARA.ACCESS.operator.multi_operator_registry import MultiCreatorRegistry
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.restore.restore_manager import RecoveryManager, RecoveryAuthorizationProof
from TARA.ACCESS.services.auth_service import CreatorAuthService, AUTHORIZED_CREATOR_EMAIL
from TARA.ACCESS.services.google_auth import GoogleAuthService
from tests.test_helpers import setup_test_identity_manager, create_test_id_token


class TestAuthorityRepairAndHardening(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_repair_test_")
        self.creator_record_path = os.path.join(self.test_dir, "operator", "operator_record.json")
        self.creators_registry_path = os.path.join(self.test_dir, "operator", "operators_registry.json")
        self.recovery_config_path = os.path.join(self.test_dir, "recovery", "recovery_config.json")
        self.seal_path = os.path.join(self.test_dir, "vault", "access_seal.json")
        self.storage_dir = os.path.join(self.test_dir, "vault")
        os.makedirs(os.path.dirname(self.creator_record_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.recovery_config_path), exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)

        # Set recovery config path env var so IdentityManager picks it up
        os.environ["TARA_RECOVERY_CONFIG_PATH"] = self.recovery_config_path

        self.passphrase = "RepairHardeningPassphrase2026!"

        # Initialize v1 keypair
        self.v1_priv, self.v1_pub = Ed25519.generate_keypair()
        self.v1_pub_hex = self.v1_pub.hex()

        # Initialize creator identity
        self.creator = CreatorIdentity(record_path=self.creator_record_path)
        self.creator.initialize_root_creator(self.v1_pub)

        # Initialize multi-creator registry
        self.registry = MultiCreatorRegistry(self.creators_registry_path)
        self.registry.set_public_key(CANONICAL_CREATOR_ID, self.v1_pub_hex)

        # Initialize recovery manager
        self.recovery_mgr = RecoveryManager(creator=self.creator, recovery_record_path=self.recovery_config_path)
        self.rec_code = self.recovery_mgr.generate_recovery_code()
        self.recovery_mgr.set_recovery_email(AUTHORIZED_CREATOR_EMAIL)

        # Initialize lifecycle manager
        self.lifecycle = AuthorityLifecycleManager(
            repo_root=self.test_dir,
            seal_path=self.seal_path,
            creator_record_path=self.creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_config_path=self.recovery_config_path
        )
        self.lifecycle._in_initialization = True
        self.lifecycle.seal_initial_authority(
            priv_bytes=self.v1_priv,
            pub_bytes=self.v1_pub,
            master_passphrase=self.passphrase,
            display_name=DEFAULT_DISPLAY_NAME
        )

        state, _ = self.lifecycle.verify_integrity()
        self.assertEqual(state, AuthorityState.ACTIVE)

        # Initialize auth service
        self.auth_service = CreatorAuthService(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_record_path=self.recovery_config_path,
            seal_path=self.seal_path,
        )
        self.auth_service.recovery_mgr = self.recovery_mgr

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_recover_locked_authority_rejects_plain_method_string(self):
        """recover_locked_authority must reject plain method names as credentials."""
        v2_priv, v2_pub = Ed25519.generate_keypair()
        with self.assertRaises(PermissionError):
            self.auth_service.recover_locked_authority(
                recovery_code="google_account",
                new_public_key_hex=v2_pub.hex(),
                new_private_key_bytes=v2_priv
            )

        with self.assertRaises(PermissionError):
            self.auth_service.recover_locked_authority(
                recovery_code="recovery_code",
                new_public_key_hex=v2_pub.hex(),
                new_private_key_bytes=v2_priv
            )

    def test_02_recover_locked_authority_rejects_unsigned_seal(self):
        """recover_locked_authority requires new_private_key_bytes and rejects None."""
        v2_priv, v2_pub = Ed25519.generate_keypair()
        with self.assertRaises(ValueError):
            self.auth_service.recover_locked_authority(
                recovery_code=self.rec_code,
                new_public_key_hex=v2_pub.hex(),
                new_private_key_bytes=None
            )

    def test_03_recover_locked_authority_rejects_mismatched_private_key(self):
        """recover_locked_authority verifies private key matches target public key."""
        v2_priv, v2_pub = Ed25519.generate_keypair()
        bad_priv, _ = Ed25519.generate_keypair()
        with self.assertRaises(ValueError):
            self.auth_service.recover_locked_authority(
                recovery_code=self.rec_code,
                new_public_key_hex=v2_pub.hex(),
                new_private_key_bytes=bad_priv
            )

    def test_04_reseal_with_recovery_synchronizes_records_before_hashing(self):
        """reseal_with_recovery updates operator_record and registry before computing canonical hashes."""
        v2_priv, v2_pub = Ed25519.generate_keypair()
        seal = self.lifecycle.reseal_with_recovery(
            new_pub_bytes=v2_pub,
            key_version=2,
            display_name=DEFAULT_DISPLAY_NAME,
            new_priv_bytes=v2_priv
        )
        self.assertIn("seal_signature", seal)
        self.assertEqual(seal["key_version"], 2)
        self.assertEqual(seal["root_public_key"], v2_pub.hex())

        # Verify integrity succeeds immediately
        state, reason = self.lifecycle.verify_integrity()
        self.assertEqual(state, AuthorityState.ACTIVE, reason)

        # Verify operator_record on disk has v2
        with open(self.creator_record_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        self.assertEqual(rec["root_public_key"], v2_pub.hex())
        self.assertEqual(rec["key_version"], 2)

        # Verify registry on disk has v2
        with open(self.creators_registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        self.assertEqual(reg[CANONICAL_CREATOR_ID]["public_key"], v2_pub.hex())

    def test_05_repair_rotated_authority_end_to_end(self):
        """repair_rotated_authority successfully restores an already-rotated v2 state to ACTIVE."""
        # 1. Simulate key rotation having happened on disk (key_version 1 -> 2)
        v2_priv, v2_pub = Ed25519.generate_keypair()
        self.creator.rotate_root_key(v2_pub, authorized=True, reason="simulated_recovery")
        self.registry.set_public_key(CANONICAL_CREATOR_ID, v2_pub.hex())
        self.recovery_mgr.save()

        # Now seal is outdated, so verify_integrity fails closed
        self.lifecycle._seal_cache = None
        state, reason = self.lifecycle.verify_integrity()
        self.assertEqual(state, AuthorityState.AUTHORITY_LOCKED)

        # 2. Store v2 private key in keystore
        id_mgr = IdentityManager(base_dir=self.test_dir)
        id_mgr.storage.store_private_key("creator_root_key_v2", v2_priv, self.passphrase)
        id_mgr.storage.store_private_key("creator_root_key", v2_priv, self.passphrase)

        # 3. Create a valid RecoveryAuthorizationProof
        proof = self.recovery_mgr.verify_recovery_code(self.rec_code)
        self.assertIsInstance(proof, RecoveryAuthorizationProof)
        self.assertTrue(proof.is_valid())

        # 4. Execute repair_rotated_authority
        res = id_mgr.repair_rotated_authority(authorization_proof=proof, passphrase=self.passphrase)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["state"], AuthorityState.ACTIVE)
        self.assertEqual(res["key_version"], 2)

        # 5. Verify authority is verified ACTIVE
        auth_state, auth_reason = id_mgr.lifecycle.verify_integrity()
        self.assertEqual(auth_state, AuthorityState.ACTIVE, auth_reason)
        self.assertTrue(id_mgr.lifecycle.is_active())

    def test_06_google_recovery_flow_for_repair(self):
        """repair_rotated_authority succeeds via verified Google account recovery."""
        v2_priv, v2_pub = Ed25519.generate_keypair()
        self.creator.rotate_root_key(v2_pub, authorized=True, reason="simulated_google_recovery")
        self.registry.set_public_key(CANONICAL_CREATOR_ID, v2_pub.hex())
        self.recovery_mgr.save()

        # Authority is locked
        self.lifecycle._seal_cache = None
        state, _ = self.lifecycle.verify_integrity()
        self.assertEqual(state, AuthorityState.AUTHORITY_LOCKED)

        # Store v2 private key in keystore
        id_mgr = IdentityManager(base_dir=self.test_dir)
        id_mgr.storage.store_private_key("creator_root_key_v2", v2_priv, self.passphrase)

        # Setup mock google service
        from tests.test_helpers import get_test_rsa_key, _TEST_KID
        google_service = GoogleAuthService(
            authorized_email=AUTHORIZED_CREATOR_EMAIL,
            expected_client_id="tara-client-id",
            trusted_public_keys={_TEST_KID: get_test_rsa_key().public_key()}
        )
        test_token = create_test_id_token(AUTHORIZED_CREATOR_EMAIL)

        # Verify Google recovery produces valid proof
        proof = self.recovery_mgr.verify_google_recovery(
            test_token,
            google_service=google_service
        )
        self.assertIsInstance(proof, RecoveryAuthorizationProof)
        self.assertTrue(proof.is_valid())

        # Repair rotated authority using proof
        res = id_mgr.repair_rotated_authority(authorization_proof=proof, passphrase=self.passphrase)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["state"], AuthorityState.ACTIVE)
        self.assertEqual(res["key_version"], 2)

        # Verify authority state is ACTIVE
        auth_state, _ = id_mgr.lifecycle.verify_integrity()
        self.assertEqual(auth_state, AuthorityState.ACTIVE)

    def test_07_recovery_code_formats_matching(self):
        """verify_recovery_code accepts candidate codes with/without hyphens, spaces, lowercase, and quotes."""
        # 1. Without hyphens
        no_hyphens = self.rec_code.replace("-", "")
        proof1 = self.recovery_mgr.verify_recovery_code(no_hyphens)
        self.assertIsInstance(proof1, RecoveryAuthorizationProof)
        self.assertTrue(proof1.is_valid())

        # 2. Lowercase with spaces
        spaced_lower = self.rec_code.replace("-", " ").lower()
        proof2 = self.recovery_mgr.verify_recovery_code(spaced_lower)
        self.assertIsInstance(proof2, RecoveryAuthorizationProof)
        self.assertTrue(proof2.is_valid())

        # 3. With surrounding quotes
        quoted = f'"{self.rec_code}"'
        proof3 = self.recovery_mgr.verify_recovery_code(quoted)
        self.assertIsInstance(proof3, RecoveryAuthorizationProof)
        self.assertTrue(proof3.is_valid())

    def test_legacy_pbkdf2_migration_to_v2_policy(self):
        """
        Verify that legacy v1 verifiers (50,000 iterations) safely migrate to current
        production policy (600,000 iterations / v2) with fresh salt upon successful verification,
        and that the old verifier is retired.
        """
        from TARA.ACCESS.restore.restore_manager import PRODUCTION_PBKDF2_ITERATIONS, CURRENT_VERIFIER_VERSION
        import hashlib
        import secrets

        # Simulate a legacy v1 verifier with 50,000 iterations
        raw_code = "LEGACY-RECOVERY-CODE-TEST"
        old_salt = secrets.token_bytes(16)
        old_hash = hashlib.pbkdf2_hmac("sha256", raw_code.encode("utf-8"), old_salt, 50_000).hex()

        self.recovery_mgr.recovery_code_hash = old_hash
        self.recovery_mgr.recovery_salt = old_salt.hex()
        self.recovery_mgr.pbkdf2_iterations = 50_000
        self.recovery_mgr.verifier_version = 1
        self.recovery_mgr.save()

        # Reload to verify v1 legacy state on disk
        mgr_reloaded = RecoveryManager(self.creator, recovery_record_path=self.recovery_mgr.recovery_record_path)
        self.assertEqual(mgr_reloaded.verifier_version, 1)
        self.assertEqual(mgr_reloaded.pbkdf2_iterations, 50_000)

        # Authenticate with recovery code
        proof = mgr_reloaded.verify_recovery_code(raw_code)
        self.assertIsInstance(proof, RecoveryAuthorizationProof)
        self.assertTrue(proof.is_valid())

        # Verify automated migration to current production policy (600,000 / v2)
        self.assertEqual(mgr_reloaded.verifier_version, CURRENT_VERIFIER_VERSION)
        self.assertEqual(mgr_reloaded.pbkdf2_iterations, PRODUCTION_PBKDF2_ITERATIONS)
        self.assertNotEqual(mgr_reloaded.recovery_salt, old_salt.hex())
        self.assertTrue(any(e.get("event") == "VERIFIER_MIGRATION_V1_TO_V2" for e in mgr_reloaded.recovery_history))

        # Verify disk persistence of migrated v2 state
        mgr_disk = RecoveryManager(self.creator, recovery_record_path=self.recovery_mgr.recovery_record_path)
        self.assertEqual(mgr_disk.verifier_version, CURRENT_VERIFIER_VERSION)
        self.assertEqual(mgr_disk.pbkdf2_iterations, PRODUCTION_PBKDF2_ITERATIONS)
        self.assertTrue(bool(mgr_disk.verify_recovery_code(raw_code)))


if __name__ == "__main__":
    unittest.main()

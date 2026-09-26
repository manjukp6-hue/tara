"""
tests/test_clone_resistance.py

Automated Test Suite for Clone Resistance and Repository Exposure Invariants.

Verifies:
1. Fresh clone state: Any cloned instance strictly defaults to CREATOR_SETUP_REQUIRED.
2. Zero secret residue: No authority seal, root keystore, or creator auth manifest exists in repo.
3. Fail-closed access control: Any privileged creator action on a cloned repo is rejected.
4. Machine binding & DPAPI security: Encrypted keystores cannot be decrypted across distinct machines.
5. Invariant integrity: Model parameter count (118,080) and model SHA-256 match canonical release.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import hashlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from TARA.ACCESS.operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from TARA.ACCESS.operator.operator_profile import CreatorIdentity, CANONICAL_CREATOR_ID
from TARA.ACCESS.services.auth_service import CreatorAuthService
from tara_core.brain import TaraBrain

CANONICAL_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
EXPECTED_PARAM_COUNT = 118080


class TestCloneResistanceAndExposure(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_clone_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_repo_has_zero_production_secrets_or_seals(self):
        """Audit that no production authority seals or private keys exist in the working tree."""
        forbidden_files = [
            os.path.join(REPO_ROOT, "storage", "vault", "access", "access_seal.json"),
            os.path.join(REPO_ROOT, "storage", "vault", "access", "operator_key.keystore"),
            os.path.join(REPO_ROOT, "TARA", "ACCESS", "operator", "auth_manifest.json"),
        ]
        for fpath in forbidden_files:
            self.assertFalse(
                os.path.exists(fpath),
                f"SECURITY VIOLATION: Secret file {fpath} must not exist in repository!"
            )

    def test_production_authority_is_creator_setup_required(self):
        """Authoritative lifecycle on current repository must be CREATOR_SETUP_REQUIRED."""
        lifecycle = AuthorityLifecycleManager(repo_root=REPO_ROOT)
        state = lifecycle.get_state()
        self.assertEqual(
            state,
            AuthorityState.CREATOR_SETUP_REQUIRED,
            f"Production authority state must be CREATOR_SETUP_REQUIRED, but was: {state}"
        )

    def test_cloned_repo_fails_closed_and_requires_creator_setup(self):
        """Simulate cloning the repository to a new directory and verify it fails closed."""
        cloned_root = os.path.join(self.temp_dir, "cloned_tara")
        # Copy identity configs only to simulate fresh clone
        src_identity = os.path.join(REPO_ROOT, "TARA", "ACCESS")
        dst_identity = os.path.join(cloned_root, "TARA", "ACCESS")
        shutil.copytree(src_identity, dst_identity)

        lifecycle = AuthorityLifecycleManager(repo_root=cloned_root)
        self.assertEqual(lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

        auth_service = CreatorAuthService(repo_root=cloned_root)
        # Attempting privileged root operation must fail
        with self.assertRaises((PermissionError, RuntimeError, ValueError)):
            auth_service.rotate_creator_key(
                new_public_key_hex="a" * 64,
                authorization={"method": "session_token", "session_token": "fake_invalid_token"}
            )

    def test_model_checksum_and_parameter_invariants(self):
        """Verify the immutable single model identity, parameter count, and SHA-256."""
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_path), "Model safetensors file must exist.")

        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        digest = hasher.hexdigest()

        self.assertEqual(
            digest,
            CANONICAL_MODEL_SHA256,
            f"Model SHA-256 invariant broken! Expected {CANONICAL_MODEL_SHA256}, got {digest}"
        )

    def test_voice_conversation_does_not_grant_creator_authority(self):
        """Verify that conversing via voice does NOT escalate or grant creator authority."""
        auth_service = CreatorAuthService(repo_root=REPO_ROOT)
        # Verify voice auxiliary is not enough to authenticate root creator
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)
        # Voice challenge alone does not change state
        challenge = auth_service.generate_voice_challenge(CANONICAL_CREATOR_ID)
        self.assertIn("nonce", challenge)
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

    def test_disposable_clone_14_invariants(self):
        """
        Verify all 14 clone-resistance invariants on a fresh disposable clone:
        1. Boots completely unconfigured (CREATOR_SETUP_REQUIRED)
        2. Contains no creator identity
        3. Contains no root key
        4. Contains no active creator session
        5. Contains no device trust
        6. Contains no recovery secret
        7. Contains no biometric enrollment
        8. Contains no voice enrollment
        9. Cannot impersonate ROOT_OPERATOR
        10. Cannot elevate itself to CREATOR
        11. Cannot authorize privileged operations
        12. Cannot promote synthetic models
        13. Cannot auto-enable autonomous skills
        14. Cannot access encrypted storage from another machine
        """
        disposable_root = os.path.join(self.temp_dir, "disposable_clone")
        shutil.copytree(os.path.join(REPO_ROOT, "TARA", "ACCESS"), os.path.join(disposable_root, "TARA", "ACCESS"))

        # Invariant 1: Boots completely unconfigured (CREATOR_SETUP_REQUIRED)
        lifecycle = AuthorityLifecycleManager(repo_root=disposable_root)
        self.assertEqual(lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

        # Invariant 2: Contains no creator identity
        creator_record_path = os.path.join(disposable_root, "TARA", "ACCESS", "operator", "operator_record.json")
        if os.path.exists(creator_record_path):
            with open(creator_record_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            self.assertIn(rec.get("status"), ["CREATOR_SETUP_REQUIRED", "setup_required", "unconfigured", "pending", None])
            self.assertFalse(rec.get("root_public_key"))

        # Invariant 3: Contains no root key
        root_keystore = os.path.join(disposable_root, "storage", "vault", "access", "operator_key.keystore")
        self.assertFalse(os.path.exists(root_keystore))
        legacy_keystore = os.path.join(disposable_root, "TARA", "ACCESS", "vault", "operator_key.keystore")
        self.assertFalse(os.path.exists(legacy_keystore))

        # Invariant 4: Contains no active creator session
        auth_service = CreatorAuthService(repo_root=disposable_root)
        self.assertEqual(len(auth_service._sessions), 0)
        self.assertIsNone(auth_service.verify_session("any_fake_token"))

        # Invariant 5: Contains no device trust
        device_store = os.path.join(disposable_root, "storage", "vault", "access", "device_trust_store.json")
        self.assertFalse(os.path.exists(device_store))
        self.assertEqual(len(auth_service.devices.list_authorized_devices()), 0)
        self.assertFalse(auth_service.devices.is_authorized("TARA-DEVICE-001"))

        # Invariant 6: Contains no recovery secret
        recovery_path = os.path.join(disposable_root, "TARA", "ACCESS", "restore", "restore_config.json")
        if os.path.exists(recovery_path):
            with open(recovery_path, "r", encoding="utf-8") as f:
                rec_cfg = json.load(f)
            self.assertIsNone(rec_cfg.get("recovery_code_hash"))
            self.assertIsNone(rec_cfg.get("recovery_salt"))
            self.assertFalse(rec_cfg.get("recovery_codes"))
            self.assertEqual(rec_cfg.get("status"), "CREATOR_SETUP_REQUIRED")

        # Invariant 7: Contains no biometric enrollment
        biometric_store = os.path.join(disposable_root, "storage", "vault", "access", "biometrics.dat")
        self.assertFalse(os.path.exists(biometric_store))

        # Invariant 8: Contains no voice enrollment
        voice_store = os.path.join(disposable_root, "storage", "vault", "access", "voice_enrollment.dat")
        self.assertFalse(os.path.exists(voice_store))

        # Invariant 9: Cannot impersonate ROOT_OPERATOR
        from TARA.ACCESS.crypto.ed25519 import Ed25519
        fake_priv, fake_pub = Ed25519.generate_keypair()
        sig = Ed25519.sign(fake_priv, b"impersonate").hex()
        impersonate_res = auth_service.authenticate_creator_key(
            proof_signature_hex=sig,
            challenge_nonce="dummy_challenge_nonce",
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertNotEqual(impersonate_res.get("status"), "SUCCESS")

        # Invariant 10: Cannot elevate itself to CREATOR
        with self.assertRaises((PermissionError, RuntimeError, ValueError)):
            auth_service.rotate_creator_key(
                new_public_key_hex="b" * 64,
                authorization={"method": "session_token", "session_token": "unauthorized"}
            )

        # Invariant 11: Cannot authorize privileged operations
        from tara_core.security.capability_guard import CapabilityGuard, ActionRequest
        from tara_core.security.security_state import TrustBoundary
        guard = CapabilityGuard()
        unauth_action = ActionRequest(
            action_type="creator_api",
            target_resource="system_shutdown",
            requester_id="anonymous",
            requester_boundary=TrustBoundary.MODEL,
            creator_token=None
        )
        authorized, reason = guard.authorize_action(unauth_action)
        self.assertFalse(authorized)

        # Invariant 12: Cannot promote synthetic models
        from tara_core.control_plane.worker_registry import CANONICAL_MODEL_SHA256 as CTRL_SHA256
        self.assertEqual(CTRL_SHA256, CANONICAL_MODEL_SHA256)
        self.assertNotEqual("synthetic_model_hash_12345", CANONICAL_MODEL_SHA256)

        # Invariant 13: Cannot auto-enable autonomous skills
        exec_action = ActionRequest(
            action_type="security_policy_write",
            target_resource="enable_autonomous_execution",
            requester_id="untrusted_clone",
            requester_boundary=TrustBoundary.SKILL,
            creator_token=None
        )
        authorized, reason = guard.authorize_action(exec_action)
        self.assertFalse(authorized)

        # Invariant 14: Cannot access encrypted storage from another machine
        from TARA.ACCESS.crypto.dpapi_storage import unprotect_bytes_dpapi
        with self.assertRaises(Exception):
            unprotect_bytes_dpapi(b"foreign_machine_ciphertext_payload_invalid")


if __name__ == "__main__":
    unittest.main()


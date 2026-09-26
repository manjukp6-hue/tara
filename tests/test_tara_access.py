"""
tests/test_tara_access.py

Comprehensive test suite verifying the complete TARA Root Creator Identity and Recovery System.
Implements the 22 required test scenarios:
1. First creator registration
2. Creator ID persistence (ROOT_OPERATOR)
3. Display name persistence (OPERATOR_ROOT)
4. Creator key generation (Ed25519)
5. Device key generation (per-device keys)
6. Google/Gmail verification
7. Firebase authentication support
8. Biometric success
9. Biometric failure & fallback
10. New device authorization
11. Existing device recognition
12. Offline creator verification
13. Creator key deletion
14. Recovery workflow
15. Key rotation (key_version 1 -> 2)
16. Old-key revocation
17. Multiple devices handling
18. Unauthorized device rejection
19. Normal user cannot obtain creator authority
20. Creator ID remains ROOT_OPERATOR after key rotation
21. Creator ID remains ROOT_OPERATOR after device migration
22. Creator ID remains ROOT_OPERATOR after Firebase migration/failure scenario
"""

import os
import sys
import shutil
import tempfile
import unittest
import json
import time
import secrets

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.access_manager import IdentityManager
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.secure_storage import SecureKeyStorage
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.policy.access_policy import IdentityPolicy, TaraRole
from cryptography.hazmat.primitives.asymmetric import rsa
from tests.test_helpers import setup_test_identity_manager, create_test_id_token


class TestTaraRootCreatorIdentity(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary test directory for identity state
        self.test_dir = tempfile.mkdtemp(prefix="tara_id_test_")
        self.id_mgr = IdentityManager(base_dir=self.test_dir)
        setup_test_identity_manager(self.id_mgr)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_first_creator_registration(self):
        """1. First creator registration creates ROOT_OPERATOR identity with OPERATOR_ROOT display name."""
        res = self.id_mgr.first_creator_setup(google_email="creator@test.local", display_name="OPERATOR_ROOT")
        self.assertEqual(res["status"], "CREATOR_MODE_ACTIVATED")
        self.assertEqual(res["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(res["display_name"], "OPERATOR_ROOT")
        self.assertEqual(res["role"], "ROOT_CREATOR")
        self.assertTrue(self.id_mgr.creator.is_initialized())

        # Enforce that attempting to re-initialize fails
        with self.assertRaises(RuntimeError):
            self.id_mgr.first_creator_setup(google_email="another@test.local")

    def test_02_creator_id_persistence(self):
        """2. Creator ID persistence across restarts and reloads."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        # Reload from disk into a fresh instance
        reloaded_mgr = IdentityManager(base_dir=self.test_dir)
        self.assertEqual(reloaded_mgr.creator.creator_id, "ROOT_OPERATOR")
        self.assertEqual(reloaded_mgr.creator.creator_id, CANONICAL_CREATOR_ID)

    def test_03_display_name_persistence(self):
        """3. Display name persistence and safe update without changing Creator ID."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local", display_name="OPERATOR_ROOT")
        self.assertEqual(self.id_mgr.creator.display_name, "OPERATOR_ROOT")

        # Update display name
        self.id_mgr.creator.update_display_name("OPERATOR_UPDATED")
        reloaded = IdentityManager(base_dir=self.test_dir)
        self.assertEqual(reloaded.creator.display_name, "OPERATOR_UPDATED")
        # Creator ID must remain strictly ROOT_OPERATOR
        self.assertEqual(reloaded.creator.creator_id, "ROOT_OPERATOR")

    def test_04_creator_key_generation(self):
        """4. Creator key generation produces valid Ed25519 keypair and saves encrypted."""
        res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        self.assertIsNotNone(self.id_mgr.creator.root_public_key)
        self.assertTrue(self.id_mgr.storage.has_key("creator_root_key"))

        priv = self.id_mgr.storage.load_private_key("creator_root_key")
        self.assertIsNotNone(priv)
        self.assertEqual(len(priv), 32)
        derived_pub = Ed25519.public_key_from_private(priv)
        self.assertEqual(derived_pub.hex(), self.id_mgr.creator.root_public_key)

    def test_05_device_key_generation(self):
        """5. Device key generation generates per-device keys without sharing creator private key."""
        res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = res["device_id"]
        self.assertEqual(dev_id, "TARA-DEVICE-001")

        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        creator_priv = self.id_mgr.storage.load_private_key("creator_root_key")
        self.assertIsNotNone(dev_priv)
        self.assertNotEqual(dev_priv, creator_priv)

    def test_06_google_gmail_verification(self):
        """6. Google/Gmail verification handles verification and account binding."""
        # Missing token fails closed
        unauth = self.id_mgr.google_service.verify_account("creator@test.local")
        self.assertFalse(unauth["verified"])
        self.assertIn("MISSING_GOOGLE_ID_TOKEN", unauth["error"])

        # Valid cryptographic test token succeeds
        token = create_test_id_token("creator@test.local")
        res = self.id_mgr.google_service.verify_account("creator@test.local", id_token=token)
        self.assertTrue(res["verified"])
        self.assertEqual(res["provider"], "google.com")

        invalid = self.id_mgr.google_service.verify_account("invalid-email", id_token=token)
        self.assertFalse(invalid["verified"])

    def test_07_firebase_authentication(self):
        """7. Firebase authentication manages supporting metadata without storing secrets."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        metadata = self.id_mgr.firebase_service.get_creator_metadata("ROOT_OPERATOR")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["creator_id"], "ROOT_OPERATOR")
        # Ensure private key is NOT in firebase metadata
        self.assertNotIn("private_key", metadata)
        self.assertNotIn("recovery_code", metadata)

    def test_08_biometric_success(self):
        """8. Biometric success permits confirmation without exposing templates."""
        bio_res = self.id_mgr.biometric_service.authenticate_biometric(simulate_user_present=True)
        self.assertTrue(bio_res["success"])
        self.assertNotIn("template", bio_res)

    def test_09_biometric_failure_and_fallback(self):
        """9. Biometric failure correctly activates device fallback credential (PIN/passkey)."""
        bio_res = self.id_mgr.biometric_service.authenticate_biometric(simulate_user_present=False)
        self.assertFalse(bio_res["success"])
        self.assertTrue(bio_res["fallback_available"])

        fallback_res = self.id_mgr.biometric_service.authenticate_fallback_credential(pin_or_passkey_correct=True)
        self.assertTrue(fallback_res["success"])

    def test_10_new_device_authorization(self):
        """10. New device authorization registers TARA-DEVICE-002 and approves via creator."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        new_dev_res = self.id_mgr.register_new_device(google_email="creator@test.local", device_name="Laptop")
        dev2_id = new_dev_res["device_id"]
        self.assertEqual(dev2_id, "TARA-DEVICE-002")
        self.assertEqual(new_dev_res["status"], "DEVICE_PENDING_APPROVAL")
        self.assertEqual(new_dev_res["creator_id"], "ROOT_OPERATOR")

        # Authorize device
        ok = self.id_mgr.authorize_new_device_via_creator(dev2_id, authorization_proof_type="biometric")
        self.assertTrue(ok)
        self.assertTrue(self.id_mgr.devices.is_authorized(dev2_id))

    def test_11_existing_device_recognition(self):
        """11. Existing device recognition recognizes authorized devices."""
        res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev1_id = res["device_id"]
        self.assertTrue(self.id_mgr.devices.is_authorized(dev1_id))
        dev_info = self.id_mgr.devices.get_device(dev1_id)
        self.assertEqual(dev_info["status"], "AUTHORIZED")

    def test_12_offline_creator_verification(self):
        """12. Offline creator verification succeeds when Firebase is completely disconnected."""
        res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = res["device_id"]

        # Simulate offline
        self.id_mgr.firebase_service.set_online_status(False)

        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        payload = "DEPLOY_CORE_MODEL_OFFLINE"
        sig = Ed25519.sign(dev_priv, payload.encode("utf-8"))

        eval_res = self.id_mgr.verify_creator_operation(
            device_id=dev_id,
            action_payload=payload,
            device_signature_bytes=sig,
            is_offline=True
        )
        self.assertTrue(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.ROOT_CREATOR)
        self.assertTrue(eval_res["offline_execution"])

    def test_13_creator_key_deletion_scenario(self):
        """13. Creator key deletion does NOT destroy the Creator ID ROOT_OPERATOR."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        # Simulate local key deletion
        self.id_mgr.storage.delete_private_key("creator_root_key")
        self.assertFalse(self.id_mgr.storage.has_key("creator_root_key"))

        # Creator ID remains intact
        self.assertEqual(self.id_mgr.creator.creator_id, "ROOT_OPERATOR")
        self.assertEqual(self.id_mgr.creator.display_name, "OPERATOR_ROOT")

    def test_14_recovery_workflow(self):
        """14. Recovery restores authority after key loss using recovery code."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        recovery_code = setup_res["recovery_code"]

        # Delete key
        self.id_mgr.storage.delete_private_key("creator_root_key")

        # Recover
        event = self.id_mgr.recover_after_key_loss(
            recovery_method="recovery_code",
            recovery_credential=recovery_code
        )
        self.assertEqual(event["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(event["new_key_version"], 2)
        self.assertTrue(self.id_mgr.storage.has_key("creator_root_key"))

    def test_15_key_rotation(self):
        """15. Key rotation increments key_version (1 -> 2)."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        self.assertEqual(self.id_mgr.creator.key_version, 1)

        event = self.id_mgr.rotate_creator_key()
        self.assertEqual(event["old_key_version"], 1)
        self.assertEqual(event["new_key_version"], 2)
        self.assertEqual(self.id_mgr.creator.key_version, 2)
        self.assertEqual(self.id_mgr.creator.creator_id, "ROOT_OPERATOR")

    def test_16_old_key_revocation(self):
        """16. Old-key revocation archives old key in revoked_keys and rejects old signatures."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        old_priv = self.id_mgr.storage.load_private_key("creator_root_key")
        old_pub = self.id_mgr.creator.root_public_key

        msg = b"AUTHORITY_VERIFICATION"
        old_sig = Ed25519.sign(old_priv, msg)

        # Rotate key
        self.id_mgr.rotate_creator_key(reason="security_upgrade")
        self.assertEqual(len(self.id_mgr.creator.revoked_keys), 1)
        self.assertEqual(self.id_mgr.creator.revoked_keys[0]["public_key"], old_pub)

        # Old key signature must be rejected against new active root
        self.assertFalse(self.id_mgr.creator.verify_authority(msg, old_sig))

    def test_17_multiple_devices(self):
        """17. Multiple devices can be registered and authorized concurrently."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev2 = self.id_mgr.register_new_device("creator@test.local", "Tablet")
        dev3 = self.id_mgr.register_new_device("creator@test.local", "Desktop")

        self.id_mgr.authorize_new_device_via_creator(dev2["device_id"], "trusted_device")
        self.id_mgr.authorize_new_device_via_creator(dev3["device_id"], "trusted_device")

        auth_devs = self.id_mgr.devices.list_authorized_devices()
        self.assertEqual(len(auth_devs), 3)

    def test_18_unauthorized_device_rejection(self):
        """18. Unauthorized device rejection blocks pending or revoked devices."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev2 = self.id_mgr.register_new_device("creator@test.local", "Unknown Phone")
        dev2_id = dev2["device_id"]

        dev2_priv = self.id_mgr.storage.load_private_key(f"device_{dev2_id}_key")
        sig = Ed25519.sign(dev2_priv, b"PAYLOAD")

        # Must be rejected because status is PENDING
        eval_res = self.id_mgr.verify_creator_operation(dev2_id, "PAYLOAD", sig)
        self.assertFalse(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.USER)

        # Authorize then revoke
        self.id_mgr.authorize_new_device_via_creator(dev2_id, "biometric")
        self.id_mgr.devices.revoke_device(dev2_id, reason="device_lost")

        eval_res2 = self.id_mgr.verify_creator_operation(dev2_id, "PAYLOAD", sig)
        self.assertFalse(eval_res2["authorized"])

    def test_19_normal_user_cannot_obtain_creator_authority(self):
        """19. Normal Firebase user or invalid Creator ID cannot obtain creator authority."""
        # 1. Normal user in Firebase
        user_auth = self.id_mgr.firebase_service.authenticate_user(uid="user_123", email="user@test.local", role="USER")
        self.assertEqual(user_auth["role"], "USER")

        # 2. Evaluation with non-creator ID
        eval_res = IdentityPolicy.evaluate_creator_permission(
            claimed_creator_id="SOME_OTHER_USER",
            is_device_authorized=True,
            cryptographic_proof_valid=True
        )
        self.assertFalse(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.USER)

    def test_20_creator_id_remains_root_operator_after_key_rotation(self):
        """20. Creator ID remains ROOT_OPERATOR after key rotation."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        self.id_mgr.rotate_creator_key()
        self.id_mgr.rotate_creator_key()
        self.assertEqual(self.id_mgr.creator.creator_id, "ROOT_OPERATOR")
        self.assertEqual(self.id_mgr.creator.display_name, "OPERATOR_ROOT")
        self.assertEqual(self.id_mgr.creator.key_version, 3)

    def test_21_creator_id_remains_root_operator_after_device_migration(self):
        """21. Creator ID remains ROOT_OPERATOR after device migration."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev2 = self.id_mgr.register_new_device("creator@test.local", "New Phone")
        self.id_mgr.authorize_new_device_via_creator(dev2["device_id"], "biometric")
        # Revoke old device
        self.id_mgr.devices.revoke_device("TARA-DEVICE-001", "migrated_to_new_phone")

        display = self.id_mgr.get_creator_display()
        self.assertEqual(display["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(display["display_name"], "OPERATOR_ROOT")

    def test_22_creator_id_remains_root_operator_after_firebase_migration(self):
        """22. Creator ID remains ROOT_OPERATOR after Firebase migration or complete offline failure."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        # Complete destruction of Firebase remote data / migration to new instance
        self.id_mgr.firebase_service = None

        reloaded = IdentityManager(base_dir=self.test_dir)
        self.assertEqual(reloaded.creator.creator_id, "ROOT_OPERATOR")
        self.assertEqual(reloaded.creator.display_name, "OPERATOR_ROOT")
        self.assertEqual(reloaded.creator.creator_id, CANONICAL_CREATOR_ID)

    def test_23_recovery_code_brute_force_rate_limiting(self):
        """23. Recovery code verification enforces brute-force lockout after 5 failed attempts."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        valid_code = setup_res["recovery_code"]

        # Attempt 5 incorrect codes
        for _ in range(5):
            self.assertFalse(self.id_mgr.recovery.verify_recovery_code("WRONG-CODE-0000-1111"))

        # 6th attempt: System must be locked out
        self.assertTrue(self.id_mgr.recovery.is_locked_out())
        # Even the valid code must be rejected while locked out
        self.assertFalse(self.id_mgr.recovery.verify_recovery_code(valid_code))

    def test_24_firebase_client_privilege_escalation_blocked(self):
        """24. Firestore rules block clients from setting role=creator or creator_id=ROOT_OPERATOR."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        user = self.id_mgr.firebase_service.authenticate_user(uid="attacker_uid", email="attacker@test.local", requested_role="CREATOR")
        # Role must be clamped to USER
        self.assertEqual(user["role"], "USER")
        self.assertIsNone(user["creator_id"])

        # Attempt client profile update
        ok, msg = self.id_mgr.firebase_service.update_user_profile("attacker_uid", {"role": "CREATOR"})
        self.assertFalse(ok)
        self.assertIn("PERMISSION_DENIED", msg)

        ok2, msg2 = self.id_mgr.firebase_service.update_user_profile("attacker_uid", {"creator_id": "ROOT_OPERATOR"})
        self.assertFalse(ok2)
        self.assertIn("PERMISSION_DENIED", msg2)

    def test_25_google_auth_alone_cannot_grant_creator_authority(self):
        """25. Google authentication alone fails without authorized device signature."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        token = create_test_id_token("creator@test.local")
        google_res = self.id_mgr.google_service.verify_account("creator@test.local", id_token=token)
        self.assertTrue(google_res["verified"])

        # Attempt privileged action with only Google verification (no device signature)
        eval_res = IdentityPolicy.evaluate_creator_permission(
            claimed_creator_id="ROOT_OPERATOR",
            is_device_authorized=False,
            cryptographic_proof_valid=False
        )
        self.assertFalse(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.USER)

    def test_26_device_2_cannot_bypass_authorization_with_known_metadata(self):
        """26. Device 2 knowing Creator ID, display name, email, and UID is still rejected."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev2 = self.id_mgr.register_new_device("creator@test.local", "Attacker Device")
        dev2_id = dev2["device_id"]

        # Attacker knows metadata: Creator ID = ROOT_OPERATOR, Display Name = OPERATOR_ROOT, email = creator@test.local
        dev2_priv = self.id_mgr.storage.load_private_key(f"device_{dev2_id}_key")
        sig = Ed25519.sign(dev2_priv, b"PRIVILEGED_ACTION")

        # Without creator approval, verification fails
        eval_res = self.id_mgr.verify_creator_operation(dev2_id, "PRIVILEGED_ACTION", sig)
        self.assertFalse(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.USER)

    def test_27_new_device_authorized_after_creator_key_loss_and_recovery(self):
        """27. Device 1 key lost -> recovery -> new key -> Device 2 authorized only via recovery."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        rec_code = setup_res["recovery_code"]
        dev1_id = setup_res["device_id"]

        # Key loss
        self.id_mgr.storage.delete_private_key("creator_root_key")

        # Recover on new installation / setup
        rec_event = self.id_mgr.recover_after_key_loss("recovery_code", rec_code)
        self.assertEqual(rec_event["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(rec_event["new_key_version"], 2)

        # Register Device 2
        dev2 = self.id_mgr.register_new_device("creator@test.local", "New Device 2")
        dev2_id = dev2["device_id"]
        # Device 2 is authorized via the restored creator authority
        ok = self.id_mgr.authorize_new_device_via_creator(dev2_id, "biometric")
        self.assertTrue(ok)
        self.assertTrue(self.id_mgr.devices.is_authorized(dev2_id))

    # ------------------------------------------------------------------------
    # FINAL CRYPTOGRAPHIC HARDENING TESTS (Tests 28-39)
    # ------------------------------------------------------------------------
    def test_28_aes_gcm_aead_private_key_storage_and_zero_plaintext(self):
        """28. AES-256-GCM AEAD storage verifies confidentiality, integrity, and zero plaintext on disk."""
        secret_bytes = secrets.token_bytes(32)
        storage = SecureKeyStorage(self.test_dir)
        key_path = storage.store_private_key("test_hardened_key", secret_bytes, "SuperSecretPassphrase123!")

        self.assertTrue(os.path.exists(key_path))
        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Standard AEAD specifications
        self.assertEqual(data.get("aead"), "AES-256-GCM")
        self.assertEqual(data.get("version"), 2)
        self.assertEqual(data.get("iterations"), 100_000)
        self.assertEqual(data.get("key_id"), "test_hardened_key")
        # 12-byte nonce hex = 24 hex characters
        self.assertEqual(len(data.get("nonce")), 24)
        # 16-byte salt hex = 32 hex characters
        self.assertEqual(len(data.get("salt")), 32)
        # Verify raw secret is NEVER stored in plaintext
        with open(key_path, "rb") as f:
            raw_file = f.read()
        self.assertNotIn(secret_bytes, raw_file)

        # Decryption succeeds with correct passphrase
        decrypted = storage.load_private_key("test_hardened_key", "SuperSecretPassphrase123!")
        self.assertEqual(decrypted, secret_bytes)

    def test_29_corrupted_ciphertext_detected_and_rejected(self):
        """29. Corrupted ciphertext is caught by AES-256-GCM authentication tag and rejected."""
        secret_bytes = secrets.token_bytes(32)
        storage = SecureKeyStorage(self.test_dir)
        key_path = storage.store_private_key("corrupt_test_key", secret_bytes, "Passphrase123")

        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Tamper with ciphertext bytes
        ct = bytearray(bytes.fromhex(data["ciphertext"]))
        ct[0] ^= 0x55
        data["ciphertext"] = ct.hex()

        with open(key_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Load must detect authentication tag failure and return None
        decrypted = storage.load_private_key("corrupt_test_key", "Passphrase123")
        self.assertIsNone(decrypted)

    def test_30_wrong_passphrase_rejected(self):
        """30. Wrong passphrase fails AES-256-GCM decryption due to authentication tag mismatch."""
        secret_bytes = secrets.token_bytes(32)
        storage = SecureKeyStorage(self.test_dir)
        storage.store_private_key("wrong_pass_key", secret_bytes, "CorrectPassphrase")

        decrypted = storage.load_private_key("wrong_pass_key", "WRONGPassphrase")
        self.assertIsNone(decrypted)

    def test_31_nonce_uniqueness_across_encryptions(self):
        """31. Every encryption generates a cryptographically unique 96-bit nonce (never reused)."""
        secret_bytes = secrets.token_bytes(32)
        storage = SecureKeyStorage(self.test_dir)
        nonces = set()
        count = 40
        for i in range(count):
            storage.store_private_key(f"nonce_key_{i}", secret_bytes, "pass")
            meta = storage.get_key_metadata(f"nonce_key_{i}")
            self.assertIsNotNone(meta)
            nonces.add(meta["nonce_hex"])
        # All 40 nonces must be strictly distinct
        self.assertEqual(len(nonces), count)

    def test_32_ed25519_production_crypto_valid_tampered_wrong_and_revoked(self):
        """32. Ed25519 validates legitimate signatures and rejects tampered, wrong, or revoked keys."""
        priv_bytes, pub_bytes = Ed25519.generate_keypair()
        message = b"ROOT_OPERATOR_ROOT_AUTHORIZATION_ACTION"
        signature = Ed25519.sign(priv_bytes, message)

        # 1. Valid signature
        self.assertTrue(Ed25519.verify(pub_bytes, message, signature))

        # 2. Tampered message rejected
        self.assertFalse(Ed25519.verify(pub_bytes, b"TAMPERED_ACTION", signature))

        # 3. Tampered signature rejected
        tampered_sig = bytearray(signature)
        tampered_sig[10] ^= 0xFF
        self.assertFalse(Ed25519.verify(pub_bytes, message, bytes(tampered_sig)))

        # 4. Wrong public key rejected
        _, other_pub = Ed25519.generate_keypair()
        self.assertFalse(Ed25519.verify(other_pub, message, signature))

        # 5. Revoked key rejected in identity policy
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        old_pub = self.id_mgr.creator.root_public_key
        # Rotate key
        self.id_mgr.rotate_creator_key()
        revoked_pubkeys = [r["public_key"] for r in self.id_mgr.creator.revoked_keys]
        self.assertIn(old_pub, revoked_pubkeys)
        # Attempting operation with old key fails revoked check
        self.assertFalse(self.id_mgr.creator.is_key_active(old_pub))

    def test_33_google_id_token_valid_cryptographic_verification(self):
        """33. Google ID token with valid RS256 signature and matching claims succeeds."""
        # Generate RSA test key pair for Google
        rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rsa_pub = rsa_priv.public_key()

        google_service = GoogleAuthService(expected_client_id="tara-client-prod")
        google_service.register_trusted_key("google-key-001", rsa_pub)

        token = GoogleAuthService.create_mock_id_token(
            private_key=rsa_priv,
            kid="google-key-001",
            email="creator@test.local",
            aud="tara-client-prod",
            iss="https://accounts.google.com",
            email_verified=True
        )

        res = google_service.verify_id_token(token)
        self.assertTrue(res["verified"])
        self.assertEqual(res["email"], "creator@test.local")
        self.assertEqual(res["sub"], "google-sub-1001")

        # Full account verification
        acc_res = google_service.verify_account("creator@test.local", id_token=token)
        self.assertTrue(acc_res["verified"])
        self.assertEqual(acc_res["email"], "creator@test.local")

    def test_34_google_id_token_expired_rejected(self):
        """34. Expired Google ID token is rejected."""
        rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        google_service = GoogleAuthService(expected_client_id="tara-client")
        google_service.register_trusted_key("k1", rsa_priv.public_key())

        # Expired 200 seconds ago
        past_time = int(time.time()) - 200
        token = GoogleAuthService.create_mock_id_token(
            private_key=rsa_priv,
            kid="k1",
            exp=past_time,
            iat=past_time - 3600
        )
        res = google_service.verify_id_token(token)
        self.assertFalse(res["verified"])
        self.assertIn("expired", res["error"].lower())

    def test_35_google_id_token_wrong_audience_rejected(self):
        """35. Google ID token with wrong audience is rejected."""
        rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        google_service = GoogleAuthService(expected_client_id="tara-expected-client")
        google_service.register_trusted_key("k1", rsa_priv.public_key())

        token = GoogleAuthService.create_mock_id_token(
            private_key=rsa_priv,
            kid="k1",
            aud="unauthorized-malicious-app"
        )
        res = google_service.verify_id_token(token)
        self.assertFalse(res["verified"])
        self.assertIn("audience mismatch", res["error"].lower())

    def test_36_google_id_token_wrong_issuer_rejected(self):
        """36. Google ID token with untrusted/spoofed issuer is rejected."""
        rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        google_service = GoogleAuthService()
        google_service.register_trusted_key("k1", rsa_priv.public_key())

        token = GoogleAuthService.create_mock_id_token(
            private_key=rsa_priv,
            kid="k1",
            iss="https://fake-accounts.rogue-domain.com"
        )
        res = google_service.verify_id_token(token)
        self.assertFalse(res["verified"])
        self.assertIn("invalid issuer", res["error"].lower())

    def test_37_google_id_token_invalid_signature_rejected(self):
        """37. Google ID token with tampered signature is cryptographically rejected."""
        rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        google_service = GoogleAuthService()
        google_service.register_trusted_key("k1", rsa_priv.public_key())

        token = GoogleAuthService.create_mock_id_token(
            private_key=rsa_priv,
            kid="k1",
            tamper_signature=True
        )
        res = google_service.verify_id_token(token)
        self.assertFalse(res["verified"])
        self.assertIn("signature verification failed", res["error"].lower())

    def test_38_google_id_token_malformed_rejected(self):
        """38. Malformed token strings are safely rejected without crashing."""
        google_service = GoogleAuthService()
        for bad_token in ["", "not-a-token", "a.b", "a.b.c.d", "badheader.badpayload.badsig"]:
            res = google_service.verify_id_token(bad_token)
            self.assertFalse(res["verified"])
            self.assertIn("error", res)

    def test_39_offline_authorization_retains_creator_power_without_network(self):
        """39. Already-authorized device operates with full creator authority completely offline."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")

        action = "OFFLINE_SENSITIVE_DEPLOYMENT"
        sig = Ed25519.sign(dev_priv, action.encode("utf-8"))

        # Completely offline mode (no network, no Firebase, no Google)
        eval_res = self.id_mgr.verify_creator_operation(
            device_id=dev_id,
            action_payload=action,
            device_signature_bytes=sig,
            is_offline=True
        )
        self.assertTrue(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.ROOT_CREATOR)
        self.assertTrue(eval_res["offline_execution"])


if __name__ == "__main__":
    unittest.main()


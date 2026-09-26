"""
tests/test_tara_integrated_security.py

Comprehensive 50-Scenario Automated Verification Suite for the Final TARA Security System.
Implements the exact 50 scenarios required by Section 26 of the Security Specification:

CRYPTO:
1. Valid Ed25519 signature
2. Wrong key
3. Invalid signature
4. Tampered message
5. Revoked key
6. Old key version
7. Key rotation
8. Secure private-key storage
9. Corrupted encrypted key detection

GOOGLE:
10. Valid ID token
11. Invalid token signature
12. Expired token
13. Wrong audience
14. Wrong issuer
15. Gmail-only creator escalation attempt

FIREBASE:
16. Firebase UID creator escalation attempt
17. Unauthorized Firestore role modification
18. Firebase offline operation

DEVICE:
19. Unknown device
20. Pending device
21. Authorized device
22. Trusted-device recovery

BIOMETRIC:
23. Biometric success
24. Biometric failure (accidental != malicious attack)

LOCKDOWN:
25. Progressive lock (rate limiting)
26. Full lockdown (5 failures)
27. Privileged command blocked during lockdown
28. Firebase/Google cannot bypass lockdown

SELF-DESTRUCT:
29. Explicit destruction authorization
30. Destruction confirmation phrase verification
31. Local TARA deletion
32. Model deletion
33. Skill deletion
34. Knowledge deletion
35. Memory deletion
36. Cache deletion
37. Database deletion
38. External registered storage deletion
39. Cloud registered storage deletion
40. Worker shutdown
41. Background-write prevention (write-block)
42. Storage registry enumeration
43. Key destruction
44. Recovery-secret destruction
45. Post-destruction old-key rejection
46. No automatic recovery after destruction
47. No automatic Creator ID recreation
48. Unrelated user-data protection
49. Unknown storage is not blindly deleted (DELETION_REQUIRES_REVIEW)
50. Deletion verification failure reported correctly (DELETION_NOT_VERIFIABLE)
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
from TARA.ACCESS.lockdown.lockdown_manager import LockdownManager, SecurityState
from TARA.ACCESS.storage.storage_registry import (
    TaraStorageRegistry, LocalStorageProvider, ExternalStorageProvider,
    CloudStorageProvider, StorageType, OwnershipTag, DeletionStatus
)
from TARA.ACCESS.destruction.self_destruct import (
    SelfDestructEngine, SystemDestroyedError, SystemDestroyingError, FINAL_CONFIRMATION_PHRASE
)
from cryptography.hazmat.primitives.asymmetric import rsa
from tests.test_helpers import setup_test_identity_manager, create_test_id_token


class TestTaraIntegratedSecurity50Scenarios(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_sec_test_")
        self.id_mgr = IdentityManager(base_dir=self.test_dir)
        setup_test_identity_manager(self.id_mgr)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # ========================================================================
    # 1. CRYPTO TESTS (1 - 9)
    # ========================================================================
    def test_01_crypto_valid_ed25519_signature(self):
        """1. Valid Ed25519 signature -> ACCEPT."""
        priv, pub = Ed25519.generate_keypair()
        msg = b"TARA_CRITICAL_PAYLOAD"
        sig = Ed25519.sign(priv, msg)
        self.assertTrue(Ed25519.verify(pub, msg, sig))

    def test_02_crypto_wrong_key(self):
        """2. Wrong key -> DENY."""
        priv1, pub1 = Ed25519.generate_keypair()
        priv2, pub2 = Ed25519.generate_keypair()
        msg = b"TARA_CRITICAL_PAYLOAD"
        sig = Ed25519.sign(priv1, msg)
        self.assertFalse(Ed25519.verify(pub2, msg, sig))

    def test_03_crypto_invalid_signature(self):
        """3. Invalid signature -> DENY."""
        priv, pub = Ed25519.generate_keypair()
        msg = b"TARA_CRITICAL_PAYLOAD"
        sig = Ed25519.sign(priv, msg)
        bad_sig = bytearray(sig)
        bad_sig[5] ^= 0xAA
        self.assertFalse(Ed25519.verify(pub, msg, bytes(bad_sig)))

    def test_04_crypto_tampered_message(self):
        """4. Tampered message -> DENY."""
        priv, pub = Ed25519.generate_keypair()
        msg = b"ORIGINAL_TRANSACTION"
        sig = Ed25519.sign(priv, msg)
        self.assertFalse(Ed25519.verify(pub, b"MODIFIED_TRANSACTION", sig))

    def test_05_crypto_revoked_key(self):
        """5. Revoked key -> DENY."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        old_pub = self.id_mgr.creator.root_public_key
        self.id_mgr.rotate_creator_key()
        revoked_keys = [r["public_key"] for r in self.id_mgr.creator.revoked_keys]
        self.assertIn(old_pub, revoked_keys)
        self.assertFalse(self.id_mgr.creator.is_key_active(old_pub))

    def test_06_crypto_old_key_version(self):
        """6. Old key version -> DENY."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        v1_key = self.id_mgr.creator.root_public_key
        self.id_mgr.rotate_creator_key()  # now v2
        self.id_mgr.rotate_creator_key()  # now v3
        self.assertEqual(self.id_mgr.creator.key_version, 3)
        self.assertFalse(self.id_mgr.creator.is_key_active(v1_key))

    def test_07_crypto_key_rotation(self):
        """7. Key rotation increments version and preserves ROOT_OPERATOR."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        event = self.id_mgr.rotate_creator_key()
        self.assertEqual(event["old_key_version"], 1)
        self.assertEqual(event["new_key_version"], 2)
        self.assertEqual(event["creator_id"], CANONICAL_CREATOR_ID)

    def test_08_crypto_secure_private_key_storage(self):
        """8. Secure private-key storage uses AES-256-GCM and zero plaintext on disk."""
        storage = SecureKeyStorage(self.test_dir)
        secret = secrets.token_bytes(32)
        key_path = storage.store_private_key("k_sec", secret, "pass123")
        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("aead"), "AES-256-GCM")
        self.assertEqual(data.get("version"), 2)
        with open(key_path, "rb") as f:
            raw = f.read()
        self.assertNotIn(secret, raw)
        decrypted = storage.load_private_key("k_sec", "pass123")
        self.assertEqual(decrypted, secret)

    def test_09_crypto_corrupted_encrypted_key_detection(self):
        """9. Corrupted encrypted key detection fails gracefully via AEAD tag check."""
        storage = SecureKeyStorage(self.test_dir)
        secret = secrets.token_bytes(32)
        key_path = storage.store_private_key("k_corrupt", secret, "pass123")
        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ct = bytearray(bytes.fromhex(data["ciphertext"]))
        ct[2] ^= 0xFF
        data["ciphertext"] = ct.hex()
        with open(key_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self.assertIsNone(storage.load_private_key("k_corrupt", "pass123"))

    # ========================================================================
    # 2. GOOGLE TESTS (10 - 15)
    # ========================================================================
    def test_10_google_valid_id_token(self):
        """10. Valid Google ID token cryptographically verified."""
        rsa_priv = rsa.generate_private_key(65537, 2048)
        service = GoogleAuthService(expected_client_id="tara-app")
        service.register_trusted_key("k1", rsa_priv.public_key())
        tok = GoogleAuthService.create_mock_id_token(rsa_priv, "k1", email="creator@test.local", aud="tara-app")
        res = service.verify_id_token(tok)
        self.assertTrue(res["verified"])
        self.assertEqual(res["email"], "creator@test.local")

    def test_11_google_invalid_token_signature(self):
        """11. Invalid token signature -> rejected."""
        rsa_priv = rsa.generate_private_key(65537, 2048)
        service = GoogleAuthService()
        service.register_trusted_key("k1", rsa_priv.public_key())
        tok = GoogleAuthService.create_mock_id_token(rsa_priv, "k1", tamper_signature=True)
        res = service.verify_id_token(tok)
        self.assertFalse(res["verified"])

    def test_12_google_expired_token(self):
        """12. Expired Google token -> rejected."""
        rsa_priv = rsa.generate_private_key(65537, 2048)
        service = GoogleAuthService()
        service.register_trusted_key("k1", rsa_priv.public_key())
        tok = GoogleAuthService.create_mock_id_token(rsa_priv, "k1", exp=int(time.time()) - 100)
        res = service.verify_id_token(tok)
        self.assertFalse(res["verified"])
        self.assertIn("expired", res["error"].lower())

    def test_13_google_wrong_audience(self):
        """13. Wrong audience -> rejected."""
        rsa_priv = rsa.generate_private_key(65537, 2048)
        service = GoogleAuthService(expected_client_id="expected-client")
        service.register_trusted_key("k1", rsa_priv.public_key())
        tok = GoogleAuthService.create_mock_id_token(rsa_priv, "k1", aud="wrong-client")
        res = service.verify_id_token(tok)
        self.assertFalse(res["verified"])
        self.assertIn("audience mismatch", res["error"].lower())

    def test_14_google_wrong_issuer(self):
        """14. Wrong issuer -> rejected."""
        rsa_priv = rsa.generate_private_key(65537, 2048)
        service = GoogleAuthService()
        service.register_trusted_key("k1", rsa_priv.public_key())
        tok = GoogleAuthService.create_mock_id_token(rsa_priv, "k1", iss="https://attacker-idp.com")
        res = service.verify_id_token(tok)
        self.assertFalse(res["verified"])
        self.assertIn("invalid issuer", res["error"].lower())

    def test_15_google_gmail_only_creator_escalation_attempt(self):
        """15. Gmail account alone cannot grant creator authority without device signature."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        tok = create_test_id_token("creator@test.local")
        acc = self.id_mgr.google_service.verify_account("creator@test.local", id_token=tok)
        self.assertTrue(acc["verified"])
        eval_res = IdentityPolicy.evaluate_creator_permission(
            claimed_creator_id=CANONICAL_CREATOR_ID,
            is_device_authorized=False,
            cryptographic_proof_valid=False
        )
        self.assertFalse(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.USER)

    # ========================================================================
    # 3. FIREBASE TESTS (16 - 18)
    # ========================================================================
    def test_16_firebase_uid_creator_escalation_attempt(self):
        """16. Firebase UID alone cannot obtain creator authority."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        normal_auth = self.id_mgr.firebase_service.authenticate_user(
            uid="attacker-uid-999",
            email="attacker@test.local",
            requested_role="creator"
        )
        self.assertEqual(normal_auth["role"], "USER")
        self.assertIsNone(normal_auth.get("creator_id"))

    def test_17_firebase_unauthorized_firestore_role_modification(self):
        """17. Firestore rules block client from setting role=creator or creator_id=ROOT_OPERATOR."""
        rules_path = os.path.join(REPO_ROOT, "TARA", "ACCESS", "services", "firestore.rules")
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_content = f.read()
        self.assertIn("request.resource.data.role != 'creator'", rules_content)
        self.assertIn("creator_id", rules_content)
        self.assertIn("allow write: if false;", rules_content)

    def test_18_firebase_offline_operation(self):
        """18. Firebase offline operation preserves local creator verification."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"OFFLINE_ACTION")
        eval_res = self.id_mgr.verify_creator_operation(dev_id, "OFFLINE_ACTION", sig, is_offline=True)
        self.assertTrue(eval_res["authorized"])
        self.assertEqual(eval_res["role"], TaraRole.ROOT_CREATOR)

    # ========================================================================
    # 4. DEVICE TESTS (19 - 22)
    # ========================================================================
    def test_19_device_unknown_device(self):
        """19. Unknown device -> DENY."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        eval_res = self.id_mgr.verify_creator_operation("UNKNOWN-DEVICE", "ACTION", b"dummy_sig")
        self.assertFalse(eval_res["authorized"])

    def test_20_device_pending_device(self):
        """20. Pending device -> DENY until authorized."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        new_dev = self.id_mgr.register_new_device("creator@test.local", "Pending Dev")
        dev_id = new_dev["device_id"]
        self.assertEqual(new_dev["status"], "DEVICE_PENDING_APPROVAL")
        self.assertFalse(self.id_mgr.devices.is_authorized(dev_id))

    def test_21_device_authorized_device(self):
        """21. Authorized device operates with verified creator authority."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        new_dev = self.id_mgr.register_new_device("creator@test.local", "Dev 2")
        dev2_id = new_dev["device_id"]
        self.id_mgr.authorize_new_device_via_creator(dev2_id, "biometric")
        self.assertTrue(self.id_mgr.devices.is_authorized(dev2_id))

    def test_22_device_trusted_device_recovery(self):
        """22. Trusted device can authorize key restoration."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev1_id = setup_res["device_id"]
        dev1 = self.id_mgr.devices.get_device(dev1_id)
        dev1_priv = self.id_mgr.storage.load_private_key(f"device_{dev1_id}_key")
        challenge = b"RECOVER_KEY_CHALLENGE"
        sig = Ed25519.sign(dev1_priv, challenge)
        verified = self.id_mgr.recovery.verify_trusted_device_recovery(
            dev1_id, dev1["device_public_key"], challenge, sig
        )
        self.assertTrue(verified)

    # ========================================================================
    # 5. BIOMETRIC TESTS (23 - 24)
    # ========================================================================
    def test_23_biometric_success(self):
        """23. Biometric success allows confirmation without exposing templates."""
        res = self.id_mgr.biometric_service.authenticate_biometric(simulate_user_present=True)
        self.assertTrue(res["success"])
        self.assertNotIn("template", res)
        self.assertNotIn("raw_data", res)

    def test_24_biometric_failure(self):
        """24. Accidental biometric failure activates PIN fallback and does NOT trigger lockdown."""
        res = self.id_mgr.biometric_service.authenticate_biometric(simulate_user_present=False)
        self.assertFalse(res["success"])
        # Report failure to lockdown manager
        lock_res = self.id_mgr.lockdown.record_biometric_failure()
        self.assertEqual(lock_res["action"], "DEVICE_PIN_FALLBACK_REQUIRED")
        self.assertFalse(self.id_mgr.lockdown.is_in_lockdown())

    # ========================================================================
    # 6. LOCKDOWN TESTS (25 - 28)
    # ========================================================================
    def test_25_lockdown_progressive_lock(self):
        """25. Progressive lock activates temporary lockout at 3 failures."""
        for i in range(3):
            self.id_mgr.lockdown.record_crypto_failure({"attempt": i})
        self.assertTrue(self.id_mgr.lockdown.is_temporarily_locked())
        self.assertFalse(self.id_mgr.lockdown.is_in_lockdown())

    def test_26_lockdown_full_lockdown(self):
        """26. Full lockdown engages after 5 consecutive malicious cryptographic failures."""
        for i in range(5):
            self.id_mgr.lockdown.record_crypto_failure({"attempt": i})
        self.assertTrue(self.id_mgr.lockdown.is_in_lockdown())
        self.assertEqual(self.id_mgr.lockdown.current_state, SecurityState.LOCKDOWN)

    def test_27_lockdown_privileged_command_blocked_during_lockdown(self):
        """27. Privileged operations are blocked during lockdown."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        self.id_mgr.lockdown.engage_manual_lockdown("Security test")
        # Key rotation blocked
        with self.assertRaises(PermissionError):
            self.id_mgr.rotate_creator_key()
        # Device registration blocked
        with self.assertRaises(PermissionError):
            self.id_mgr.register_new_device("creator@test.local", "Blocked dev")

    def test_28_lockdown_firebase_google_cannot_bypass_lockdown(self):
        """28. Valid Firebase or Google accounts cannot bypass lockdown."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        self.id_mgr.lockdown.engage_manual_lockdown("Threat detected")
        tok = create_test_id_token("creator@test.local")
        google_res = self.id_mgr.google_service.verify_account("creator@test.local", id_token=tok)
        self.assertTrue(google_res["verified"])
        # Even with verified Google account, privileged operation is blocked by lockdown
        can_exec = self.id_mgr.lockdown.can_execute_privileged_operation("rotate_creator_key")
        self.assertFalse(can_exec)

    # ========================================================================
    # 7. SELF-DESTRUCT TESTS (29 - 50)
    # ========================================================================
    def test_29_self_destruct_explicit_authorization_required(self):
        """29. Self-destruct arming fails without valid creator cryptographic authority."""
        self.id_mgr.first_creator_setup(google_email="creator@test.local")
        # Unauthorized device or wrong signature
        with self.assertRaises(PermissionError):
            self.id_mgr.arm_self_destruct(
                claimed_creator_id=CANONICAL_CREATOR_ID,
                device_id="UNAUTHORIZED_DEV",
                device_signature=b"dummy_signature",
                arm_challenge=b"challenge"
            )

    def test_30_self_destruct_confirmation_phrase_verification(self):
        """30. Execution fails with incorrect confirmation phrase."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        challenge = b"ARM_DESTRUCTION_CHALLENGE"
        sig = Ed25519.sign(dev_priv, challenge)
        arm_res = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, challenge)
        arm_token = arm_res["arm_token"]
        with self.assertRaises(ValueError):
            self.id_mgr.execute_self_destruct(arm_token, "WRONG_CONFIRMATION_PHRASE")

    def test_31_self_destruct_local_tara_deletion(self):
        """31. Self-destruct deletes registered local TARA data directories."""
        local_dir = os.path.join(self.test_dir, "TARA_LOCAL_MOCK")
        os.makedirs(local_dir, exist_ok=True)
        file_path = os.path.join(local_dir, "tara_system.dat")
        with open(file_path, "wb") as f:
            f.write(b"tara_data")
        provider = LocalStorageProvider("local_test", local_dir)
        provider.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(file_path))

    def test_32_self_destruct_model_deletion(self):
        """32. Self-destruct targets and deletes MODEL assets."""
        model_dir = os.path.join(self.test_dir, "MODEL")
        os.makedirs(model_dir, exist_ok=True)
        m_file = os.path.join(model_dir, "weights.safetensors")
        with open(m_file, "wb") as f:
            f.write(b"weights")
        prov = LocalStorageProvider("model_test", model_dir)
        prov.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(m_file))

    def test_33_self_destruct_skill_deletion(self):
        """33. Self-destruct targets and deletes SKILLS assets."""
        skills_dir = os.path.join(self.test_dir, "SKILLS")
        os.makedirs(skills_dir, exist_ok=True)
        s_file = os.path.join(skills_dir, "skill.json")
        with open(s_file, "w") as f:
            f.write("{}")
        prov = LocalStorageProvider("skills_test", skills_dir)
        prov.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(s_file))

    def test_34_self_destruct_knowledge_deletion(self):
        """34. Self-destruct targets and deletes KNOWLEDGE assets."""
        k_dir = os.path.join(self.test_dir, "KNOWLEDGE")
        os.makedirs(k_dir, exist_ok=True)
        k_file = os.path.join(k_dir, "facts.db")
        with open(k_file, "wb") as f:
            f.write(b"db")
        prov = LocalStorageProvider("k_test", k_dir)
        prov.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(k_file))

    def test_35_self_destruct_memory_deletion(self):
        """35. Self-destruct targets and deletes MEMORY assets."""
        mem_dir = os.path.join(self.test_dir, "MEMORY")
        os.makedirs(mem_dir, exist_ok=True)
        m_file = os.path.join(mem_dir, "episodes.jsonl")
        with open(m_file, "w") as f:
            f.write("episodes")
        prov = LocalStorageProvider("mem_test", mem_dir)
        prov.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(m_file))

    def test_36_self_destruct_cache_deletion(self):
        """36. Self-destruct targets and deletes CACHE assets."""
        cache_dir = os.path.join(self.test_dir, "CACHE")
        os.makedirs(cache_dir, exist_ok=True)
        c_file = os.path.join(cache_dir, "cache.bin")
        with open(c_file, "wb") as f:
            f.write(b"cache")
        prov = LocalStorageProvider("cache_test", cache_dir)
        prov.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(c_file))

    def test_37_self_destruct_database_deletion(self):
        """37. Self-destruct targets and deletes DATABASE files."""
        db_dir = os.path.join(self.test_dir, "DATABASE")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, "tara_records.sqlite")
        with open(db_file, "wb") as f:
            f.write(b"sqlite")
        prov = LocalStorageProvider("db_test", db_dir)
        prov.delete_tara_objects(verify=True)
        self.assertFalse(os.path.exists(db_file))

    def test_38_self_destruct_external_registered_storage_deletion(self):
        """38. Registered external SSD / USB storage is enumerated and deleted."""
        ext_dir = os.path.join(self.test_dir, "EXTERNAL_SSD")
        os.makedirs(ext_dir, exist_ok=True)
        ext_file = os.path.join(ext_dir, "tara_backup.tar")
        with open(ext_file, "wb") as f:
            f.write(b"backup")
        prov = ExternalStorageProvider("ext_test", ext_dir)
        self.id_mgr.storage_registry.register_provider("ext_test", prov)
        rep = prov.delete_tara_objects(verify=True)
        self.assertEqual(rep.status, DeletionStatus.SUCCESS)
        self.assertFalse(os.path.exists(ext_file))

    def test_39_self_destruct_cloud_registered_storage_deletion(self):
        """39. Registered cloud storage objects are deleted."""
        cloud_prov = CloudStorageProvider("cloud_drive", "gdrive://tara_folder", supports_remote_verification=True)
        cloud_prov.register_cloud_object("model_snapshot.bin", b"weights")
        self.id_mgr.storage_registry.register_provider("cloud_drive", cloud_prov)
        rep = cloud_prov.delete_tara_objects(verify=True)
        self.assertEqual(rep.status, DeletionStatus.SUCCESS)
        self.assertTrue(cloud_prov.verify_deletion())

    def test_40_self_destruct_worker_shutdown(self):
        """40. Worker shutdown hooks are invoked during self-destruct."""
        worker_stopped = False
        def mock_stop():
            nonlocal worker_stopped
            worker_stopped = True
        self.id_mgr.self_destruct.register_worker_shutdown_hook(mock_stop)
        # Execute arm + final destruct
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"ARM")
        arm = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, b"ARM")
        self.id_mgr.execute_self_destruct(arm["arm_token"], FINAL_CONFIRMATION_PHRASE)
        self.assertTrue(worker_stopped)

    def test_41_self_destruct_background_write_prevention(self):
        """41. Persistent writes are frozen immediately upon entering DESTROYING state."""
        self.id_mgr.lockdown.set_destroying_state()
        self.assertTrue(self.id_mgr.lockdown.is_write_blocked())
        with self.assertRaises(SystemDestroyingError):
            self.id_mgr._assert_not_write_blocked()

    def test_42_self_destruct_storage_registry_enumeration(self):
        """42. Storage registry enumerates all registered providers and objects."""
        objs = self.id_mgr.storage_registry.enumerate_all_objects()
        self.assertIn("local_model", objs)
        self.assertIn("local_skills", objs)
        self.assertIn("local_identity", objs)

    def test_43_self_destruct_key_destruction(self):
        """43. Creator and device private keys are shredded and unlinked."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"ARM")
        arm = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, b"ARM")
        self.id_mgr.execute_self_destruct(arm["arm_token"], FINAL_CONFIRMATION_PHRASE)
        self.assertIsNone(self.id_mgr.storage.load_private_key("creator_root_key"))
        self.assertIsNone(self.id_mgr.storage.load_private_key(f"device_{dev_id}_key"))

    def test_44_self_destruct_recovery_secret_destruction(self):
        """44. Recovery codes and recovery records are completely destroyed."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"ARM")
        arm = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, b"ARM")
        self.id_mgr.execute_self_destruct(arm["arm_token"], FINAL_CONFIRMATION_PHRASE)
        self.assertIsNone(self.id_mgr.recovery.recovery_code_hash)
        self.assertFalse(os.path.exists(self.id_mgr.recovery.recovery_record_path))

    def test_45_self_destruct_post_destruction_old_key_rejection(self):
        """45. Old keys are permanently rejected after destruction."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"ARM")
        arm = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, b"ARM")
        self.id_mgr.execute_self_destruct(arm["arm_token"], FINAL_CONFIRMATION_PHRASE)
        # Attempting verify operation on destroyed system raises SystemDestroyedError
        with self.assertRaises(SystemDestroyedError):
            self.id_mgr.verify_creator_operation(dev_id, "ACTION", sig)

    def test_46_self_destruct_no_automatic_recovery(self):
        """46. No recovery of any kind is permitted after final destruction."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        rec_code = setup_res["recovery_code"]
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"ARM")
        arm = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, b"ARM")
        self.id_mgr.execute_self_destruct(arm["arm_token"], FINAL_CONFIRMATION_PHRASE)
        with self.assertRaises(SystemDestroyedError):
            self.id_mgr.recover_after_key_loss("recovery_code", rec_code)

    def test_47_self_destruct_no_automatic_creator_id_recreation(self):
        """47. System will not allow recreation or re-initialization after destruction."""
        setup_res = self.id_mgr.first_creator_setup(google_email="creator@test.local")
        dev_id = setup_res["device_id"]
        dev_priv = self.id_mgr.storage.load_private_key(f"device_{dev_id}_key")
        sig = Ed25519.sign(dev_priv, b"ARM")
        arm = self.id_mgr.arm_self_destruct(CANONICAL_CREATOR_ID, dev_id, sig, b"ARM")
        self.id_mgr.execute_self_destruct(arm["arm_token"], FINAL_CONFIRMATION_PHRASE)
        # Any attempt to re-init first_creator_setup fails permanently
        with self.assertRaises(SystemDestroyedError):
            self.id_mgr.first_creator_setup(google_email="creator@test.local")

    def test_48_self_destruct_unrelated_user_data_protection(self):
        """48. Personal user documents (.docx, .jpg) are NEVER deleted."""
        user_dir = os.path.join(self.test_dir, "user_folder")
        os.makedirs(user_dir, exist_ok=True)
        personal_doc = os.path.join(user_dir, "resume.docx")
        personal_photo = os.path.join(user_dir, "family.jpg")
        with open(personal_doc, "wb") as f:
            f.write(b"personal_doc")
        with open(personal_photo, "wb") as f:
            f.write(b"personal_photo")
        prov = LocalStorageProvider("user_dir_test", user_dir)
        rep = prov.delete_tara_objects(verify=True)
        self.assertTrue(os.path.exists(personal_doc))
        self.assertTrue(os.path.exists(personal_photo))
        self.assertEqual(rep.status, DeletionStatus.DELETION_REQUIRES_REVIEW)

    def test_49_self_destruct_unknown_storage_not_blindly_deleted(self):
        """49. Ambiguous storage is tagged DELETION_REQUIRES_REVIEW and spared."""
        ambig_dir = os.path.join(self.test_dir, "ambiguous_data")
        os.makedirs(ambig_dir, exist_ok=True)
        ambig_file = os.path.join(ambig_dir, "unknown_archive.zip")
        with open(ambig_file, "wb") as f:
            f.write(b"unknown")
        prov = LocalStorageProvider("ambig_test", ambig_dir)
        rep = prov.delete_tara_objects(verify=True)
        self.assertEqual(rep.status, DeletionStatus.DELETION_REQUIRES_REVIEW)
        self.assertEqual(rep.ambiguous_count, 1)

    def test_50_self_destruct_deletion_verification_failure_reported(self):
        """50. When cloud deletion cannot be verified, reports DELETION_NOT_VERIFIABLE."""
        unverifiable_prov = CloudStorageProvider(
            "immutable_cloud",
            "s3://immutable-snapshot-bucket",
            supports_remote_verification=False
        )
        unverifiable_prov.register_cloud_object("snapshot.bak", b"data")
        rep = unverifiable_prov.delete_tara_objects(verify=True)
        self.assertEqual(rep.status, DeletionStatus.DELETION_NOT_VERIFIABLE)
        self.assertFalse(unverifiable_prov.verify_deletion())


if __name__ == "__main__":
    unittest.main()

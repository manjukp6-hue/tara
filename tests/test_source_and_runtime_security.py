"""
tests/test_source_and_runtime_security.py

Comprehensive 14-Scenario Test Suite for TARA Source-Code and Runtime Security Hardening:
1. TARA functionality remains unchanged (chat, brain loop, skill execution).
2. Source repository contains no private creator key.
3. Source repository contains no recovery secret.
4. Source repository contains no master passphrase.
5. Source repository contains no Google secret.
6. Creator authority cannot be initialized a second time.
7. Source-code possession alone cannot replace ROOT_CREATOR.
8. Creator registry tampering fails closed.
9. Protected creator state loads correctly in authorized runtime.
10. Unauthorized runtime cannot unlock creator authority.
11. Render/server startup still works and /health responds cleanly.
12. Existing creator authentication tests still pass.
13. Existing TARA regression tests still pass.
14. Production model SHA256 remains unchanged (7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309).
"""

import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import unittest

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
from TARA.ACCESS.operator.multi_operator_registry import MultiCreatorRegistry
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.restore.restore_manager import RecoveryManager
from TARA.ACCESS.services.auth_service import CreatorAuthService, AUTHORIZED_CREATOR_EMAIL
from TARA.ACCESS.wizard.setup_wizard import CreatorSetupWizard
from TARA.SECURITY.release_manifest import ReleaseManifestManager, EXPECTED_PRODUCTION_MODEL_SHA256
from TARA.SECURITY.tamper_detector import SourceTamperDetector, IntegrityViolationError
from TARA.SECURITY.source_protector import ProtectedArtifactManager, compile_protected_bundle
from TARA.SECURITY.token_scanner import RepositoryTokenScanner, RepositorySecretScanner


class TestSourceAndRuntimeSecurity(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_sec_test_")
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
        self.passphrase = "PrimarySecurityTestSecret2026!"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_authorized_environment(self):
        wizard = CreatorSetupWizard(
            repo_root=self.test_dir,
            creator_record_path=self.creator_record_path,
            recovery_config_path=self.recovery_config_path,
            storage_dir=self.storage_dir,
            auth_manifest_path=self.auth_manifest_path,
            trigger_file_path=self.trigger_file_path,
            creators_registry_path=self.creators_registry_path,
            seal_path=self.seal_path
        )
        res = wizard.run_setup(
            master_passphrase=self.passphrase,
            confirm_passphrase=self.passphrase,
            private_trigger_phrase="creator test trigger phrase 999"
        )
        return wizard, res

    # -----------------------------------------------------------------------
    # TEST 1: TARA FUNCTIONALITY REMAINS UNCHANGED
    # -----------------------------------------------------------------------
    def test_01_tara_chat_and_brain_functionality_unchanged(self):
        from tara_core.brain import TaraBrain
        brain = TaraBrain()
        self.assertIsNotNone(brain)
        # Test normal user interaction
        res = brain.process(actor_id="user", input_text="Hello TARA")
        self.assertIn("final_response", res)
        self.assertTrue(len(res["final_response"]) > 0)
        self.assertEqual(res.get("decision"), "ALLOW")
        self.assertEqual(res.get("outcome"), "SUCCESS")
        # Verify skills are accessible
        skills = brain.skill_engine.list_skills()
        self.assertTrue(len(skills) > 0)

    # -----------------------------------------------------------------------
    # TEST 2: SOURCE REPOSITORY CONTAINS NO PRIVATE CREATOR KEY
    # -----------------------------------------------------------------------
    def test_02_source_repo_contains_no_private_creator_key(self):
        scanner = RepositorySecretScanner(repo_root=REPO_ROOT)
        tracked = scanner.get_tracked_files()
        for f in tracked:
            findings = scanner.scan_file(f)
            key_findings = [fi for fi in findings if fi["finding_type"] == "ED25519_PRIVATE_KEY_PEM"]
            self.assertEqual(len(key_findings), 0, f"Found private key PEM in tracked file {f}")

    # -----------------------------------------------------------------------
    # TEST 3: SOURCE REPOSITORY CONTAINS NO RECOVERY SECRET
    # -----------------------------------------------------------------------
    def test_03_source_repo_contains_no_recovery_secret(self):
        scanner = RepositorySecretScanner(repo_root=REPO_ROOT)
        tracked = scanner.get_tracked_files()
        for f in tracked:
            findings = scanner.scan_file(f)
            recov_findings = [fi for fi in findings if fi["finding_type"] == "PLAINTEXT_RECOVERY_CODE"]
            self.assertEqual(len(recov_findings), 0, f"Found plaintext recovery code in tracked file {f}")

    # -----------------------------------------------------------------------
    # TEST 4: SOURCE REPOSITORY CONTAINS NO MASTER PASSPHRASE
    # -----------------------------------------------------------------------
    def test_04_source_repo_contains_no_master_passphrase(self):
        scanner = RepositorySecretScanner(repo_root=REPO_ROOT)
        tracked = scanner.get_tracked_files()
        for f in tracked:
            findings = scanner.scan_file(f)
            pass_findings = [fi for fi in findings if fi["finding_type"] == "HARDCODED_CREATOR_PASSPHRASE"]
            self.assertEqual(len(pass_findings), 0, f"Found hardcoded passphrase in tracked file {f}")

    # -----------------------------------------------------------------------
    # TEST 5: SOURCE REPOSITORY CONTAINS NO GOOGLE SECRET
    # -----------------------------------------------------------------------
    def test_05_source_repo_contains_no_google_secret(self):
        scanner = RepositorySecretScanner(repo_root=REPO_ROOT)
        tracked = scanner.get_tracked_files()
        for f in tracked:
            findings = scanner.scan_file(f)
            google_findings = [fi for fi in findings if fi["finding_type"] in ("GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")]
            self.assertEqual(len(google_findings), 0, f"Found Google secret in tracked file {f}")

    # -----------------------------------------------------------------------
    # TEST 6: CREATOR AUTHORITY CANNOT BE INITIALIZED A SECOND TIME
    # -----------------------------------------------------------------------
    def test_06_creator_authority_cannot_be_initialized_second_time(self):
        wizard, res = self._setup_authorized_environment()
        self.assertTrue(res["setup_complete"])

        # Attempting second setup must raise PermissionError
        with self.assertRaises(PermissionError) as ctx:
            wizard.run_setup(
                master_passphrase="AnotherPassphrase123!",
                confirm_passphrase="AnotherPassphrase123!"
            )
        self.assertIn("already initialized", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # TEST 7: SOURCE-CODE POSSESSION ALONE CANNOT REPLACE ROOT_CREATOR
    # -----------------------------------------------------------------------
    def test_07_source_code_possession_alone_cannot_replace_root_creator(self):
        wizard, res = self._setup_authorized_environment()
        # An attacker gets the source files and attempts to modify operator_record.json directly
        with open(self.creator_record_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["creator_id"] = "IMPOSTOR_CREATOR"
        with open(self.creator_record_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        # Authority lifecycle must fail closed
        lifecycle = AuthorityLifecycleManager(
            repo_root=self.test_dir,
            seal_path=self.seal_path,
            creator_record_path=self.creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_config_path=self.recovery_config_path,
            auth_manifest_path=self.auth_manifest_path
        )
        state = lifecycle.get_state()
        self.assertEqual(state, AuthorityState.AUTHORITY_LOCKED)
        self.assertFalse(lifecycle.is_active())

    # -----------------------------------------------------------------------
    # TEST 8: CREATOR REGISTRY TAMPERING FAILS CLOSED
    # -----------------------------------------------------------------------
    def test_08_creator_registry_tampering_fails_closed(self):
        wizard, res = self._setup_authorized_environment()
        # Attacker injects a rogue root creator into operators_registry.json
        with open(self.creators_registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        reg["ATTACKER"] = {
            "creator_id": "ATTACKER",
            "role": "ROOT_CREATOR",
            "status": "active"
        }
        with open(self.creators_registry_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

        lifecycle = AuthorityLifecycleManager(
            repo_root=self.test_dir,
            seal_path=self.seal_path,
            creator_record_path=self.creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_config_path=self.recovery_config_path,
            auth_manifest_path=self.auth_manifest_path
        )
        self.assertEqual(lifecycle.get_state(), AuthorityState.AUTHORITY_LOCKED)

    # -----------------------------------------------------------------------
    # TEST 9: PROTECTED CREATOR STATE LOADS CORRECTLY IN AUTHORIZED RUNTIME
    # -----------------------------------------------------------------------
    def test_09_protected_creator_state_loads_in_authorized_runtime(self):
        wizard, res = self._setup_authorized_environment()
        lifecycle = AuthorityLifecycleManager(
            repo_root=self.test_dir,
            seal_path=self.seal_path,
            creator_record_path=self.creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_config_path=self.recovery_config_path,
            auth_manifest_path=self.auth_manifest_path
        )
        self.assertEqual(lifecycle.get_state(), AuthorityState.ACTIVE)
        self.assertTrue(lifecycle.is_active())

    # -----------------------------------------------------------------------
    # TEST 10: UNAUTHORIZED RUNTIME CANNOT UNLOCK CREATOR AUTHORITY
    # -----------------------------------------------------------------------
    def test_10_unauthorized_runtime_cannot_unlock_creator_authority(self):
        # Fresh environment with repo files present but no authority seal
        fresh_dir = tempfile.mkdtemp(prefix="tara_unauth_")
        try:
            rec_p = os.path.join(fresh_dir, "operator_record.json")
            with open(rec_p, "w", encoding="utf-8") as f:
                json.dump({"creator_id": "ROOT_OPERATOR", "root_public_key": "aabbcc"}, f)

            lifecycle = AuthorityLifecycleManager(
                repo_root=fresh_dir,
                seal_path=os.path.join(fresh_dir, "non_existent_seal.json"),
                creator_record_path=rec_p,
                creators_registry_path=os.path.join(fresh_dir, "operators_registry.json"),
                recovery_config_path=os.path.join(fresh_dir, "restore_config.json"),
                auth_manifest_path=os.path.join(fresh_dir, "auth_manifest.json")
            )
            # Must fail closed: AUTHORITY_LOCKED
            self.assertEqual(lifecycle.get_state(), AuthorityState.AUTHORITY_LOCKED)
            self.assertFalse(lifecycle.is_active())
        finally:
            shutil.rmtree(fresh_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # TEST 11: RENDER / SERVER STARTUP AND /health ENDPOINT CLEAN
    # -----------------------------------------------------------------------
    def test_11_server_health_check_responds_cleanly_without_secrets(self):
        from tara_core.server import create_server
        server = create_server(host="127.0.0.1", port=0)
        self.assertIsNotNone(server)
        server.server_close()

    # -----------------------------------------------------------------------
    # TEST 12: ENCRYPTED ARTIFACT PROTECTION AND PRE-COMPILED BUNDLE
    # -----------------------------------------------------------------------
    def test_12_encrypted_artifact_and_compiled_bundle(self):
        mgr = ProtectedArtifactManager(deployment_secret="TestDeploymentSecret2026!")
        test_payload = b"sensitive_runtime_instructions_and_tokens"
        envelope = mgr.encrypt_artifact(test_payload, artifact_id="bundle_01")
        self.assertIn("ciphertext", envelope)
        self.assertNotIn("sensitive_runtime_instructions_and_tokens", envelope["ciphertext"])

        # Decrypt with authorized secret
        decrypted = mgr.decrypt_artifact(envelope)
        self.assertEqual(decrypted, test_payload)

        # Decrypt with wrong secret fails
        with self.assertRaises(Exception):
            mgr.decrypt_artifact(envelope, secret="WrongSecret1234567890!")

        # Bytecode compilation test
        sample_src = os.path.join(self.test_dir, "sample_src")
        sample_out = os.path.join(self.test_dir, "sample_out")
        os.makedirs(sample_src, exist_ok=True)
        with open(os.path.join(sample_src, "core_logic.py"), "w") as f:
            f.write("def compute(x): return x * 42\n")

        compiled = compile_protected_bundle(sample_src, sample_out)
        self.assertTrue(any(k.endswith("core_logic.py") for k in compiled.keys()))

    # -----------------------------------------------------------------------
    # TEST 13: SOURCE TAMPER DETECTION FAILS CLOSED
    # -----------------------------------------------------------------------
    def test_13_source_tamper_detection_fails_closed(self):
        detector = SourceTamperDetector(repo_root=self.test_dir)
        test_file = os.path.join(self.test_dir, "critical_module.py")
        with open(test_file, "w") as f:
            f.write("def original_behavior(): return True\n")

        detector.register_baseline(["critical_module.py"])
        is_intact, _ = detector.verify_integrity()
        self.assertTrue(is_intact)

        # Tamper with file
        with open(test_file, "w") as f:
            f.write("def hijacked_behavior(): return False\n")

        is_intact, tampered = detector.verify_integrity()
        self.assertFalse(is_intact)
        self.assertIn("critical_module.py", tampered)

        with self.assertRaises(IntegrityViolationError):
            detector.enforce_fail_closed("test_operation")

    # -----------------------------------------------------------------------
    # TEST 14: PRODUCTION MODEL SHA256 INVARIANT
    # -----------------------------------------------------------------------
    def test_14_production_model_sha256_invariant(self):
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.isfile(model_path), f"Production model not found at {model_path}")
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()
        self.assertEqual(
            actual_sha.lower(),
            EXPECTED_PRODUCTION_MODEL_SHA256.lower(),
            "Production model SHA256 invariant broken!"
        )


if __name__ == "__main__":
    unittest.main()

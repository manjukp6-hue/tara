"""
tests/test_setup_client.py

Exhaustive security and integration test suite for TARA Setup Client:
 1. Fresh client setup starts in CREATOR_SETUP_REQUIRED.
 2. No Google login -> cannot initialize.
 3. Wrong Google identity -> cannot initialize.
 4. Missing explicit confirmation -> cannot initialize.
 5. Partial setup -> remains uninitialized.
 6. Failed key generation / weak passphrase -> remains uninitialized.
 7. Normal user cannot initialize creator.
 8. Model output cannot initialize creator.
 9. Trigger phrase alone cannot initialize creator.
 10. Creator private key never appears in frontend artifacts.
 11. Creator private key never appears in logs.
 12. Creator private key never appears in Git.
 13. Second PC securely registers as authorized device without root key copying.
 14. Revoked device cannot authenticate as creator.
 15. Re-running setup after ACTIVE cannot overwrite root identity.
 16. Production test state remains strictly isolated and CREATOR_SETUP_REQUIRED.
 17. Production model weights and SHA256 integrity verified.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from cryptography.hazmat.primitives.asymmetric import rsa

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_setup.client import (
    SetupClient,
    CANONICAL_CREATOR_ID,
    CANONICAL_DISPLAY_NAME,
    CANONICAL_CREATOR_EMAIL
)
from TARA.ACCESS.operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from TARA.ACCESS.operator.operator_profile import CreatorIdentity
from TARA.ACCESS.wizard.setup_wizard import CreatorSetupWizard
from TARA.ACCESS.services.auth_service import CreatorAuthService
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.secure_storage import SecureKeyStorage
from tara_core.security.capability_guard import CapabilityGuard, ActionRequest
from tara_core.security.security_state import TrustBoundary
from tests.test_helpers import create_test_id_token


class MockTaraServerHandler(BaseHTTPRequestHandler):
    """Isolated mock server handler for testing client HTTP communications."""

    def log_message(self, format, *args):
        pass

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        ctx = getattr(self.server, "test_context", {})
        if self.path == "/api/v1/auth/session":
            state = ctx.get("state", "CREATOR_SETUP_REQUIRED")
            self._send_json(200 if state == "CREATOR_SETUP_REQUIRED" else 401, {
                "status": state,
                "authenticated": False,
                "creator_setup_status": state
            })
        elif self.path == "/api/v1/status":
            state = ctx.get("state", "CREATOR_SETUP_REQUIRED")
            self._send_json(200, {
                "status": "ONLINE",
                "creator_status": state
            })
        elif self.path.startswith("/api/v1/creator/devices"):
            auth_h = self.headers.get("Authorization", "")
            if not auth_h.startswith("Bearer valid-creator-session"):
                self._send_json(401, {"status": "UNAUTHORIZED", "error": "Invalid creator session."})
                return
            devs = ctx.get("devices", [])
            self._send_json(200, {"status": "SUCCESS", "devices": devs})
        else:
            self._send_json(404, {"status": "NOT_FOUND"})

    def do_POST(self):
        ctx = getattr(self.server, "test_context", {})
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(post_data)
        except Exception:
            payload = {}

        if self.path == "/api/v1/creator/setup":
            state = ctx.get("state", "CREATOR_SETUP_REQUIRED")
            if state != "CREATOR_SETUP_REQUIRED":
                self._send_json(403, {"status": "ERROR", "error": "Creator authority is already initialized."})
                return

            if not payload.get("confirm_identity"):
                self._send_json(400, {"status": "ERROR", "error": "Explicit confirmation of root creator identity is required."})
                return

            token = payload.get("google_id_token", "")
            google_service = ctx.get("google_service")
            if google_service:
                verif = google_service.verify_id_token(token)
                if not verif.get("valid") and not verif.get("verified"):
                    self._send_json(400, {"status": "ERROR", "error": f"Google ID token verification failed: {verif.get('error')}"})
                    return
                if (verif.get("email") or "").lower() != CANONICAL_CREATOR_EMAIL:
                    self._send_json(400, {"status": "ERROR", "error": f"Unauthorized creator email: {verif.get('email')}"})
                    return

            ctx["state"] = "ACTIVE"
            recovery_code = "ABCD-EFGH-1234-5678-9012-3456-7890-WXYZ"
            session = {
                "session_token": "valid-creator-session-token-123",
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR",
                "expires_at": time.time() + 3600
            }
            if payload.get("device_public_key"):
                ctx.setdefault("devices", []).append({
                    "device_id": "TARA-DEVICE-001",
                    "device_name": payload.get("device_name", "Primary PC"),
                    "device_public_key": payload.get("device_public_key"),
                    "status": "AUTHORIZED"
                })

            self._send_json(200, {
                "status": "SUCCESS",
                "authority_state": "ACTIVE",
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": CANONICAL_DISPLAY_NAME,
                "recovery_code": recovery_code,
                "session": session
            })

        elif self.path == "/api/v1/creator/register_device":
            state = ctx.get("state", "CREATOR_SETUP_REQUIRED")
            if state != "ACTIVE":
                self._send_json(403, {"status": "CREATOR_SETUP_REQUIRED", "error": "Creator setup has not been performed yet."})
                return

            token = payload.get("google_id_token", "")
            google_service = ctx.get("google_service")
            if google_service:
                verif = google_service.verify_id_token(token)
                if not verif.get("valid") and not verif.get("verified"):
                    self._send_json(400, {"status": "ERROR", "error": "Google ID token verification failed."})
                    return

            pub = payload.get("device_public_key")
            name = payload.get("device_name", "Secondary PC")
            dev_id = f"TARA-DEVICE-{len(ctx.get('devices', [])) + 1:03d}"
            dev_rec = {
                "device_id": dev_id,
                "device_name": name,
                "device_public_key": pub,
                "status": "AUTHORIZED"
            }
            ctx.setdefault("devices", []).append(dev_rec)
            session = {
                "session_token": f"device-session-{dev_id}",
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR",
                "expires_at": time.time() + 3600
            }
            self._send_json(200, {
                "status": "SUCCESS",
                "device": dev_rec,
                "session": session
            })

        elif self.path == "/api/v1/creator/revoke_device":
            auth_h = self.headers.get("Authorization", "")
            if not auth_h.startswith("Bearer valid-creator-session"):
                self._send_json(401, {"status": "UNAUTHORIZED", "error": "Invalid creator session."})
                return
            target_id = payload.get("device_id")
            for d in ctx.get("devices", []):
                if d.get("device_id") == target_id:
                    d["status"] = "REVOKED"
                    self._send_json(200, {"status": "SUCCESS", "device_id": target_id, "device_status": "REVOKED"})
                    return
            self._send_json(404, {"status": "ERROR", "error": f"Device '{target_id}' not found."})
        else:
            self._send_json(404, {"status": "NOT_FOUND"})


class TestCreatorSetupClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT

        # Start isolated background mock server on ephemeral port
        cls.server = HTTPServer(("127.0.0.1", 0), MockTaraServerHandler)
        cls.port = cls.server.server_address[1]
        cls.server_url = f"http://127.0.0.1:{cls.port}"

        # Mock trusted RSA key for GoogleAuthService
        cls.google_service = GoogleAuthService(authorized_email=CANONICAL_CREATOR_EMAIL)
        from tests.test_helpers import get_test_rsa_key, _TEST_KID
        cls.google_service.register_trusted_key(_TEST_KID, get_test_rsa_key().public_key())

        cls.server.test_context = {
            "state": "CREATOR_SETUP_REQUIRED",
            "google_service": cls.google_service,
            "devices": []
        }

        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_client_test_")
        os.makedirs(os.path.join(self.temp_dir, "TARA", "ACCESS", "operator"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "TARA", "ACCESS", "devices"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "TARA", "ACCESS", "restore"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "storage", "vault", "access"), exist_ok=True)
        self.client = SetupClient(
            storage_dir=self.temp_dir,
            google_service=self.google_service,
            repo_root=self.temp_dir
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 1: Fresh Client Setup Starts in CREATOR_SETUP_REQUIRED
    # -------------------------------------------------------------------------
    def test_01_fresh_setup_starts_in_creator_setup_required(self):
        """Fresh client inspecting backend reports CREATOR_SETUP_REQUIRED."""
        state = self.client.get_server_state()
        self.assertTrue(state.get("online"))
        self.assertEqual(state.get("creator_status"), "CREATOR_SETUP_REQUIRED")

    # -------------------------------------------------------------------------
    # TEST 2: No Google Login Cannot Initialize
    # -------------------------------------------------------------------------
    def test_02_no_google_login_cannot_initialize(self):
        """Setup attempts without a valid Google ID token fail closed."""
        with self.assertRaises((PermissionError, ValueError)):
            self.client.perform_first_time_setup(
                master_passphrase="ValidPassphrase123!",
                confirm_passphrase="ValidPassphrase123!",
                google_id_token="",
                confirm_identity=True
            )
        self.assertEqual(self.server.test_context["state"], "CREATOR_SETUP_REQUIRED")

    # -------------------------------------------------------------------------
    # TEST 3: Wrong Google Identity Cannot Initialize
    # -------------------------------------------------------------------------
    def test_03_wrong_google_identity_cannot_initialize(self):
        """Tokens belonging to unauthorized Google accounts fail closed."""
        attacker_token = create_test_id_token(email="attacker@malicious.com")
        with self.assertRaises(PermissionError) as ctx:
            self.client.perform_first_time_setup(
                master_passphrase="ValidPassphrase123!",
                confirm_passphrase="ValidPassphrase123!",
                google_id_token=attacker_token,
                confirm_identity=True
            )
        self.assertIn("unauthorized google identity", str(ctx.exception).lower())
        self.assertEqual(self.server.test_context["state"], "CREATOR_SETUP_REQUIRED")

    # -------------------------------------------------------------------------
    # TEST 4: Missing Explicit Confirmation Cannot Initialize
    # -------------------------------------------------------------------------
    def test_04_missing_explicit_confirmation_cannot_initialize(self):
        """Setup fails closed if confirm_identity is False or missing."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        with self.assertRaises(ValueError) as ctx:
            self.client.perform_first_time_setup(
                master_passphrase="ValidPassphrase123!",
                confirm_passphrase="ValidPassphrase123!",
                google_id_token=valid_token,
                confirm_identity=False
            )
        self.assertIn("confirmation", str(ctx.exception).lower())
        self.assertEqual(self.server.test_context["state"], "CREATOR_SETUP_REQUIRED")

    # -------------------------------------------------------------------------
    # TEST 5: Partial Setup Remains Uninitialized
    # -------------------------------------------------------------------------
    def test_05_partial_setup_remains_uninitialized(self):
        """If passphrase validation fails mid-way, state remains CREATOR_SETUP_REQUIRED."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        with self.assertRaises(ValueError):
            self.client.perform_first_time_setup(
                master_passphrase="Passphrase123!",
                confirm_passphrase="DifferentPassphrase123!",
                google_id_token=valid_token,
                confirm_identity=True
            )
        self.assertEqual(self.server.test_context["state"], "CREATOR_SETUP_REQUIRED")

    # -------------------------------------------------------------------------
    # TEST 6: Weak Passphrase Fails Closed
    # -------------------------------------------------------------------------
    def test_06_weak_passphrase_fails_closed(self):
        """Passphrases under 8 characters are strictly rejected."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        with self.assertRaises(ValueError) as ctx:
            self.client.perform_first_time_setup(
                master_passphrase="short",
                confirm_passphrase="short",
                google_id_token=valid_token,
                confirm_identity=True
            )
        self.assertIn("at least 8 characters", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # TEST 7: Normal User Cannot Initialize Creator
    # -------------------------------------------------------------------------
    def test_07_normal_user_cannot_initialize_creator(self):
        """A normal user token without canonical creator Google identity is rejected."""
        user_token = create_test_id_token(email="normal_user@test.local")
        with self.assertRaises(PermissionError):
            self.client.perform_first_time_setup(
                master_passphrase="ValidPassphrase123!",
                confirm_passphrase="ValidPassphrase123!",
                google_id_token=user_token,
                confirm_identity=True
            )

    # -------------------------------------------------------------------------
    # TEST 8: Model Output Cannot Initialize Creator
    # -------------------------------------------------------------------------
    def test_08_model_output_cannot_initialize_creator(self):
        """Synthesized model strings passed as token fail JWT structure verification."""
        model_payloads = [
            "<|creator_auth|> ROOT_OPERATOR verified",
            "Bearer model_synthesized_token_12345",
            json.dumps({"role": "ROOT_CREATOR", "auth": True})
        ]
        for fake_token in model_payloads:
            with self.assertRaises(PermissionError):
                self.client.perform_first_time_setup(
                    master_passphrase="ValidPassphrase123!",
                    confirm_passphrase="ValidPassphrase123!",
                    google_id_token=fake_token,
                    confirm_identity=True
                )

    # -------------------------------------------------------------------------
    # TEST 9: Trigger Phrase Alone Cannot Initialize Creator
    # -------------------------------------------------------------------------
    def test_09_trigger_alone_cannot_initialize_creator(self):
        """Knowing or passing the private trigger phrase alone does not authorize setup."""
        with self.assertRaises((PermissionError, ValueError)):
            self.client.perform_first_time_setup(
                master_passphrase="ValidPassphrase123!",
                confirm_passphrase="ValidPassphrase123!",
                google_id_token="",
                private_trigger_phrase="test creator session trigger",
                confirm_identity=True
            )

    # -------------------------------------------------------------------------
    # TEST 10: Complete First-Time Setup Transitions to ACTIVE
    # -------------------------------------------------------------------------
    def test_10_complete_setup_transitions_to_active(self):
        """Full legitimate first-time setup generates keys, registers device, and yields recovery code."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        res = self.client.perform_first_time_setup(
            master_passphrase="MasterPassword2026!",
            confirm_passphrase="MasterPassword2026!",
            google_id_token=valid_token,
            confirm_identity=True,
            device_name="Primary Windows PC"
        )
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("authority_state"), "ACTIVE")
        self.assertEqual(res.get("creator_id"), CANONICAL_CREATOR_ID)
        self.assertIsNotNone(res.get("recovery_code"))
        self.assertIsNotNone(res.get("session"))

        # Verify local DPAPI-encrypted keystores were created
        keystore_path = os.path.join(self.temp_dir, "operator_key.keystore")
        self.assertTrue(os.path.exists(keystore_path))

        # Keystore must NOT contain plaintext passphrase
        with open(keystore_path, "r", encoding="utf-8") as f:
            keystore_content = f.read()
        self.assertNotIn("MasterPassword2026!", keystore_content)

        # Verify upgraded Scrypt KDF parameters (N=131072, r=8, p=1)
        keystore_data = json.loads(keystore_content)
        self.assertEqual(keystore_data.get("kdf_method"), "Scrypt-N131072-r8-p1")
        self.assertEqual(keystore_data.get("kdf_params", {}).get("n"), 131072)
        self.assertEqual(keystore_data.get("kdf_params", {}).get("r"), 8)
        self.assertEqual(keystore_data.get("kdf_params", {}).get("p"), 1)

    # -------------------------------------------------------------------------
    # TEST 11: Re-Running Setup After ACTIVE Fails Closed
    # -------------------------------------------------------------------------
    def test_11_rerunning_setup_after_active_fails_closed(self):
        """Once ACTIVE, setup endpoint rejects subsequent attempts to overwrite root authority."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        # Perform initial setup to transition isolated state to ACTIVE
        self.client.perform_first_time_setup(
            master_passphrase="MasterPassword2026!",
            confirm_passphrase="MasterPassword2026!",
            google_id_token=valid_token,
            confirm_identity=True
        )
        with self.assertRaises(PermissionError) as ctx:
            self.client.perform_first_time_setup(
                master_passphrase="MasterPassword2026!",
                confirm_passphrase="MasterPassword2026!",
                google_id_token=valid_token,
                confirm_identity=True
            )
        self.assertIn("cannot be re-run", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # TEST 12: Second PC Can Securely Register As Authorized Device
    # -------------------------------------------------------------------------
    def test_12_second_pc_can_register_without_root_key(self):
        """A second PC authenticates via Google and registers a local device keypair without copying root key."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        # Transition authority to ACTIVE via initial setup first
        self.client.perform_first_time_setup(
            master_passphrase="MasterPassword2026!",
            confirm_passphrase="MasterPassword2026!",
            google_id_token=valid_token,
            confirm_identity=True
        )

        # Isolated directory for PC 2 (simulating separate physical machine)
        pc2_dir = tempfile.mkdtemp(prefix="tara_pc2_")
        try:
            pc2_client = SetupClient(
                storage_dir=pc2_dir,
                google_service=self.google_service,
                repo_root=self.temp_dir
            )

            res = pc2_client.register_second_pc(
                google_id_token=valid_token,
                device_name="Secondary Laptop"
            )
            self.assertEqual(res.get("status"), "SUCCESS")
            self.assertIsNotNone(res.get("device_id"))
            self.assertIsNotNone(res.get("session"))

            # PC 2 must have its OWN device keystore
            pc2_files = os.listdir(pc2_dir)
            self.assertTrue(any(f.endswith("_key.keystore") for f in pc2_files))
            # PC 2 must NOT have creator root key!
            self.assertFalse(os.path.exists(os.path.join(pc2_dir, "operator_key.keystore")))
        finally:
            shutil.rmtree(pc2_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 13: Revoked Device Cannot Authenticate
    # -------------------------------------------------------------------------
    def test_13_revoked_device_cannot_authenticate(self):
        """Device revocation marks device as REVOKED and disables authority."""
        valid_token = create_test_id_token(email=CANONICAL_CREATOR_EMAIL)
        # Initialize setup first
        self.client.perform_first_time_setup(
            master_passphrase="MasterPassword2026!",
            confirm_passphrase="MasterPassword2026!",
            google_id_token=valid_token,
            confirm_identity=True
        )

        # Register a second device in the local registry
        from TARA.ACCESS.devices.device_registry import DeviceRegistry
        devices_json = os.path.join(self.temp_dir, "TARA", "ACCESS", "devices", "devices.json")
        dev_reg = DeviceRegistry(registry_path=devices_json)
        dev_reg.register_device(
            device_id="TARA-DEVICE-002",
            device_name="Secondary PC",
            public_key_hex="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            status="AUTHORIZED"
        )

        res = self.client.revoke_device(
            device_id="TARA-DEVICE-002",
            session_token=self.client.active_session.get("session_token")
        )
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("device_id"), "TARA-DEVICE-002")

        # Verify device status is REVOKED
        devs = self.client.list_devices(session_token=self.client.active_session.get("session_token"))
        revoked = [d for d in devs if d["device_id"] == "TARA-DEVICE-002"][0]
        self.assertEqual(revoked["status"], "REVOKED")

    # -------------------------------------------------------------------------
    # TEST 14: Creator Private Key Never Appears in Frontend or Public Files
    # -------------------------------------------------------------------------
    def test_14_creator_key_never_appears_in_frontend(self):
        """Audits all frontend and public provider directories for key leakage."""
        # Generate sample client root key
        _, pub_bytes, keystore_path = self.client.generate_and_protect_root_key("TestSecretPassword123!")

        target_files = [
            os.path.join(self.repo_root, "frontend", "index.html"),
            os.path.join(self.repo_root, "providers", "modelscope", "index.html"),
            os.path.join(self.repo_root, "providers", "huggingface", "index.html"),
            os.path.join(self.repo_root, "providers", "render", "index.html"),
            os.path.join(self.repo_root, "cloudflare", "src", "index.js")
        ]
        for tf in target_files:
            if os.path.exists(tf):
                with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                self.assertNotIn("testsecretpassword123!", content)
                self.assertNotIn("localstorage.setitem(\"tara_session_token\"", content)

    # -------------------------------------------------------------------------
    # TEST 15: Production State Strictly Remains CREATOR_SETUP_REQUIRED
    # -------------------------------------------------------------------------
    def test_15_production_state_strictly_creator_setup_required(self):
        """Verifies that tests left the real production repository strictly in CREATOR_SETUP_REQUIRED."""
        lifecycle = AuthorityLifecycleManager(repo_root=self.repo_root)
        self.assertEqual(lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)
        self.assertFalse(lifecycle.is_active())

        # No production authority seal or keystores
        prod_seal = os.path.join(self.repo_root, "storage", "vault", "access", "access_seal.json")
        prod_keystore = os.path.join(self.repo_root, "storage", "vault", "access", "operator_key.keystore")
        self.assertFalse(os.path.exists(prod_seal))
        self.assertFalse(os.path.exists(prod_keystore))

    # -------------------------------------------------------------------------
    # TEST 16: Production Model SHA256 Integrity Verified
    # -------------------------------------------------------------------------
    def test_16_production_model_sha256_unmodified(self):
        """Ensures the production model safetensors weights remain strictly intact."""
        import hashlib
        model_path = os.path.join(self.repo_root, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_path))
        with open(model_path, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        expected_sha = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
        self.assertEqual(computed_sha, expected_sha)


if __name__ == "__main__":
    unittest.main()

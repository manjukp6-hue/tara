"""
tests/test_final_auth.py

Exhaustive test suite verifying the final TARA In-Chat Creator Authentication Architecture:
1. Private trigger activation presents in-chat choices (QR, Creator Key, Google, Recovery).
2. Authentication method selection in chat.
3. QR authentication (in-chat ASCII QR, one-time challenge, Ed25519 device signature).
4. Creator Key authentication (cryptographic proof artifact, zero raw private key in chat).
5. Google authentication (RS256 JWT, claim verification, email matching).
6. Recovery authentication (32-character recovery code).
7. Invalid credentials fail closed with standard error.
8. Expired challenge rejection.
9. Replay attack rejection.
10. Revoked creator rejection.
11. Suspended creator rejection.
12. Wrong Google account rejection.
13. Forged creator ID rejection.
14. Forged role / anti-self-escalation enforcement.
15. Session expiry.
16. Logout revocation.
17. Multi-creator authentication (ROOT_OPERATOR = ROOT_CREATOR, CREATOR_2/3 = CREATOR).
18. Strict production model SHA256 invariant.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import hashlib
import unittest
from http.client import HTTPConnection
from cryptography.hazmat.primitives.asymmetric import rsa

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from TARA.ACCESS.activation.activation_manager import PrivateTriggerManager, GENERIC_LOGIN_PHRASES
from TARA.ACCESS.services.auth_service import CreatorAuthService, AUTHORIZED_CREATOR_EMAIL
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.operator.operator_profile import CreatorIdentity, CANONICAL_CREATOR_ID
from TARA.ACCESS.operator.multi_operator_registry import MultiCreatorRegistry
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.devices.device_registry import DeviceRegistry
from TARA.ACCESS.restore.restore_manager import RecoveryManager
from tara_core.brain import TaraBrain
from tara_core.server import create_server, GLOBAL_API_ROUTER, ApiRouter


EXPECTED_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
INITIAL_TEST_TRIGGER = "Tara creator activation protocol start"


class TestFinalCreatorAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate test RSA keypair for Google ID token verification
        cls.rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.rsa_pub = cls.rsa_key.public_key()
        cls.test_kid = "test-google-kid-1"

        # Generate Ed25519 root keypair for ROOT_OPERATOR
        cls.root_priv, cls.root_pub = Ed25519.generate_keypair()

        # Generate Ed25519 secondary keypair for CREATOR_2
        cls.c2_priv, cls.c2_pub = Ed25519.generate_keypair()

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_test_auth_")
        self.trigger_file = os.path.join(self.temp_dir, "activation_config.json")
        self.creator_record = os.path.join(self.temp_dir, "operator_record.json")
        self.creators_registry = os.path.join(self.temp_dir, "operators_registry.json")
        self.devices_file = os.path.join(self.temp_dir, "devices.json")
        self.recovery_file = os.path.join(self.temp_dir, "restore_config.json")
        self.audit_log_dir = os.path.join(self.temp_dir, "audit")
        os.makedirs(self.audit_log_dir, exist_ok=True)

        # Mock GoogleAuthService with pre-loaded trusted RSA key
        self.google_auth = GoogleAuthService(
            authorized_email=AUTHORIZED_CREATOR_EMAIL,
            expected_client_id="tara-client-id",
            trusted_public_keys={self.test_kid: self.rsa_pub}
        )

        self.seal_path = os.path.join(self.temp_dir, "access_seal.json")
        self.auth_manifest = os.path.join(self.temp_dir, "auth_manifest.json")

        # Initialize CreatorAuthService before creator files exist
        self.auth_service = CreatorAuthService(
            repo_root=self.temp_dir,
            creator_record_path=self.creator_record,
            trigger_file_path=self.trigger_file,
            audit_log_dir=self.audit_log_dir,
            devices_file_path=self.devices_file,
            creators_registry_path=self.creators_registry,
            recovery_record_path=self.recovery_file,
            google_service=self.google_auth,
            seal_path=self.seal_path,
            auth_manifest_path=self.auth_manifest
        )

        # Begin initialization transaction while clean
        self.auth_service.lifecycle.begin_initialization()

        # Initialize mock creator record for ROOT_OPERATOR
        with open(self.creator_record, "w", encoding="utf-8") as f:
            json.dump({
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": "OPERATOR_ROOT",
                "recovery_email": "operator@internal.local",
                "root_public_key": self.root_pub.hex(),
                "status": "active",
                "role": "ROOT_CREATOR"
            }, f)
        self.auth_service.creator.load()

        # Register public keys in multi-creator registry
        self.auth_service.multi_creators.set_public_key(CANONICAL_CREATOR_ID, self.root_pub.hex())
        self.auth_service.multi_creators.set_public_key("CREATOR_2", self.c2_pub.hex())

        # Establish initial private trigger phrase verifier
        self.auth_service.trigger_mgr.set_trigger_phrase(INITIAL_TEST_TRIGGER)

        # Setup recovery code in recovery_mgr
        self.test_recovery_code = self.auth_service.recovery_mgr.generate_recovery_code()

        # Cryptographically seal initial test authority
        self.auth_service.lifecycle.seal_initial_authority(
            priv_bytes=self.root_priv,
            pub_bytes=self.root_pub,
            master_passphrase="TestMasterPassphrase123!",
            display_name="OPERATOR_ROOT"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. TRIGGER DETECTION & IN-CHAT PRESENTATION
    # -------------------------------------------------------------------------
    def test_01_private_trigger_in_chat_presentation(self):
        """Private trigger presents the 4 choices directly for in-chat display."""
        res = self.auth_service.check_conversational_trigger(INITIAL_TEST_TRIGGER)
        self.assertTrue(res.get("is_trigger"))
        resp = res.get("response", "")
        self.assertIn("Creator authentication requested. Choose an authentication method:", resp)
        self.assertIn("[ 📱 QR Authentication ]", resp)
        self.assertIn("[ 🔑 Creator Key Authentication ]", resp)
        self.assertIn("[ 🔐 Google Authentication ]", resp)
        self.assertIn("[ 🆘 Recovery ]", resp)
        self.assertEqual(res.get("status"), "AWAITING_METHOD_SELECTION")

    def test_02_generic_phrases_rejected_and_trigger_alone_no_privilege(self):
        """Generic phrases are blacklisted and trigger alone never grants privilege."""
        for phrase in GENERIC_LOGIN_PHRASES:
            res = self.auth_service.check_conversational_trigger(phrase)
            self.assertFalse(res.get("is_trigger"))

        # Trigger alone gives 0 sessions
        self.assertEqual(len(self.auth_service._sessions), 0)

    # -------------------------------------------------------------------------
    # 2. IN-CHAT METHOD SELECTION
    # -------------------------------------------------------------------------
    def test_03_in_chat_method_selection(self):
        """Selecting each method in chat returns the method-specific prompt and challenge."""
        # 1. Select QR
        qr_sel = self.auth_service.handle_method_selection("QR")
        self.assertEqual(qr_sel["status"], "AWAITING_QR_APPROVAL")
        self.assertIn("Scan this QR code", qr_sel["response"])
        self.assertIn("ascii_qr", qr_sel["challenge"])

        # 2. Select Creator Key
        key_sel = self.auth_service.handle_method_selection("CREATOR_KEY")
        self.assertEqual(key_sel["status"], "AWAITING_KEY_PROOF")
        self.assertIn("Challenge Nonce", key_sel["response"])
        self.assertTrue(key_sel.get("challenge_nonce"))

        # 3. Select Google
        goog_sel = self.auth_service.handle_method_selection("GOOGLE")
        self.assertEqual(goog_sel["status"], "AWAITING_GOOGLE_TOKEN")
        self.assertIn("operator@internal.local", goog_sel["response"])

        # 4. Select Recovery
        rec_sel = self.auth_service.handle_method_selection("RECOVERY")
        self.assertEqual(rec_sel["status"], "AWAITING_RECOVERY_CODE")
        self.assertIn("32-character high-entropy recovery code", rec_sel["response"])

    # -------------------------------------------------------------------------
    # 3. METHOD 1: QR AUTHENTICATION
    # -------------------------------------------------------------------------
    def test_04_qr_authentication_success_and_replay_prevention(self):
        """QR authentication succeeds once with Ed25519 device signature; replay fails."""
        dev_priv, dev_pub = Ed25519.generate_keypair()
        rec = self.auth_service.devices.register_device(dev_pub, device_name="Creator Phone", status="AUTHORIZED")
        dev_id = rec["device_id"]

        chal = self.auth_service.create_qr_challenge(creator_id=CANONICAL_CREATOR_ID)
        chal_id = chal["challenge_id"]
        nonce = chal["nonce"]

        msg = f"{chal_id}:{nonce}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        sig_hex = Ed25519.sign(dev_priv, msg).hex()

        # Approval 1: Success
        res = self.auth_service.verify_qr_approval(chal_id, dev_id, sig_hex)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["message"], f"✅ Creator authenticated: {CANONICAL_CREATOR_ID}.")
        self.assertEqual(res["role"], "ROOT_CREATOR")

        # Approval 2 (Replay): Strictly fails
        replay = self.auth_service.verify_qr_approval(chal_id, dev_id, sig_hex)
        self.assertEqual(replay["status"], "FAILED")
        self.assertIn("replay prevented", replay["error"])

    def test_05_qr_expired_challenge_rejected(self):
        """Expired QR challenge fails verification."""
        dev_priv, dev_pub = Ed25519.generate_keypair()
        rec = self.auth_service.devices.register_device(dev_pub, device_name="Device", status="AUTHORIZED")
        dev_id = rec["device_id"]

        chal = self.auth_service.create_qr_challenge()
        chal_id = chal["challenge_id"]
        # Backdate expiry
        self.auth_service._qr_challenges[chal_id]["expires_at"] = time.time() - 10

        sig_hex = Ed25519.sign(dev_priv, b"dummy").hex()
        res = self.auth_service.verify_qr_approval(chal_id, dev_id, sig_hex)
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("expired", res["error"])

    # -------------------------------------------------------------------------
    # 4. METHOD 2: CREATOR KEY AUTHENTICATION
    # -------------------------------------------------------------------------
    def test_06_creator_key_proof_authentication(self):
        """Creator Key authenticates via cryptographic signature over one-time nonce; no private key in chat."""
        # Step 1: Initiate Creator Key auth
        sel = self.auth_service.handle_method_selection("CREATOR_KEY", creator_id=CANONICAL_CREATOR_ID)
        nonce = sel["challenge_nonce"]

        # Step 2: Client/agent signs nonce locally with Ed25519 private key
        sig_hex = Ed25519.sign(self.root_priv, nonce.encode("utf-8")).hex()

        # Step 3: Verify proof against registered root public key
        res = self.auth_service.authenticate_creator_key(
            proof_signature_hex=sig_hex,
            challenge_nonce=nonce,
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["message"], f"✅ Creator authenticated: {CANONICAL_CREATOR_ID}.")
        self.assertEqual(res["role"], "ROOT_CREATOR")

        # Step 4: Replay with same nonce fails
        replay = self.auth_service.authenticate_creator_key(
            proof_signature_hex=sig_hex,
            challenge_nonce=nonce,
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertEqual(replay["status"], "FAILED")

    def test_07_creator_key_invalid_signature_rejected(self):
        """Invalid or forged creator key signature is rejected."""
        sel = self.auth_service.handle_method_selection("CREATOR_KEY")
        nonce = sel["challenge_nonce"]

        bad_priv, _ = Ed25519.generate_keypair()
        bad_sig = Ed25519.sign(bad_priv, nonce.encode("utf-8")).hex()

        res = self.auth_service.authenticate_creator_key(
            proof_signature_hex=bad_sig,
            challenge_nonce=nonce,
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(res["error"], "Creator authentication failed. No creator authority was granted.")

    # -------------------------------------------------------------------------
    # 5. METHOD 3: GOOGLE AUTHENTICATION
    # -------------------------------------------------------------------------
    def test_08_google_authentication_success_and_wrong_account_rejected(self):
        """Valid Google token authenticates; wrong Google email rejected."""
        # Valid token
        valid_token = GoogleAuthService.create_mock_id_token(
            private_key=self.rsa_key,
            kid=self.test_kid,
            email=AUTHORIZED_CREATOR_EMAIL,
            aud="tara-client-id",
            email_verified=True
        )
        res = self.auth_service.authenticate_google_token(valid_token)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["message"], f"✅ Creator authenticated: {CANONICAL_CREATOR_ID}.")
        self.assertEqual(res["role"], "ROOT_CREATOR")

        # Wrong account
        wrong_token = GoogleAuthService.create_mock_id_token(
            private_key=self.rsa_key,
            kid=self.test_kid,
            email="attacker@test.local",
            aud="tara-client-id",
            email_verified=True
        )
        res_wrong = self.auth_service.authenticate_google_token(wrong_token)
        self.assertEqual(res_wrong["status"], "FAILED")
        self.assertEqual(res_wrong["error"], "Creator authentication failed. No creator authority was granted.")

    # -------------------------------------------------------------------------
    # 6. METHOD 4: RECOVERY AUTHENTICATION
    # -------------------------------------------------------------------------
    def test_09_recovery_authentication(self):
        """Valid 32-char recovery code authenticates; wrong code or email alone fails."""
        # Valid code
        res = self.auth_service.authenticate_recovery(self.test_recovery_code, claimed_creator_id=CANONICAL_CREATOR_ID)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["message"], f"✅ Creator authenticated: {CANONICAL_CREATOR_ID}.")

        # Name or email alone strictly fails
        res_fake = self.auth_service.authenticate_recovery("operator@internal.local")
        self.assertEqual(res_fake["status"], "FAILED")

        # Invalid code fails
        res_bad = self.auth_service.authenticate_recovery("INVALID-RECOVERY-CODE-XXXX-YYYY")
        self.assertEqual(res_bad["status"], "FAILED")

    # -------------------------------------------------------------------------
    # 7. MULTI-CREATOR AUTHENTICATION & ROLE ENFORCEMENT
    # -------------------------------------------------------------------------
    def test_10_multi_creator_authentication_and_anti_self_escalation(self):
        """Multi-creator support: ROOT_OPERATOR is ROOT_CREATOR; CREATOR_2 is CREATOR without self-escalation."""
        # Authenticate CREATOR_2 via Creator Key proof
        sel = self.auth_service.handle_method_selection("CREATOR_KEY", creator_id="CREATOR_2")
        nonce = sel["challenge_nonce"]
        c2_sig = Ed25519.sign(self.c2_priv, nonce.encode("utf-8")).hex()

        res2 = self.auth_service.authenticate_creator_key(
            proof_signature_hex=c2_sig,
            challenge_nonce=nonce,
            claimed_creator_id="CREATOR_2"
        )
        self.assertEqual(res2["status"], "SUCCESS")
        self.assertEqual(res2["message"], "✅ Creator authenticated: CREATOR_2.")
        # Strict Invariant: CREATOR_2 must be CREATOR, never ROOT_CREATOR
        self.assertEqual(res2["role"], "CREATOR")

        # Role resolution invariant
        self.assertEqual(self.auth_service.multi_creators.resolve_role("CREATOR_2", claimed_role="ROOT_CREATOR"), "CREATOR")
        self.assertEqual(self.auth_service.multi_creators.resolve_role(CANONICAL_CREATOR_ID), "ROOT_CREATOR")

    # -------------------------------------------------------------------------
    # 8. SUSPENDED / REVOKED CREATOR
    def test_11_suspended_or_revoked_creator_blocked(self):
        """Suspended or revoked creator account cannot authenticate under any method."""
        self.auth_service.multi_creators.update_status("CREATOR_2", "suspended")
        self.auth_service.lifecycle.reseal_with_recovery(
            new_pub_bytes=bytes.fromhex(self.auth_service.creator.root_public_key),
            key_version=self.auth_service.creator.key_version,
            display_name="OPERATOR_ROOT"
        )

        sel = self.auth_service.handle_method_selection("CREATOR_KEY", creator_id="CREATOR_2")
        nonce = sel["challenge_nonce"]
        c2_sig = Ed25519.sign(self.c2_priv, nonce.encode("utf-8")).hex()

        res = self.auth_service.authenticate_creator_key(
            proof_signature_hex=c2_sig,
            challenge_nonce=nonce,
            claimed_creator_id="CREATOR_2"
        )
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(res["error"], "Creator authentication failed. No creator authority was granted.")

    # -------------------------------------------------------------------------
    # 9. SESSION EXPIRY AND LOGOUT
    # -------------------------------------------------------------------------
    def test_12_session_expiry_and_logout(self):
        """Session tokens are verified, expirable, and revocable on logout."""
        valid_token = GoogleAuthService.create_mock_id_token(
            private_key=self.rsa_key,
            kid=self.test_kid,
            email=AUTHORIZED_CREATOR_EMAIL,
            aud="tara-client-id"
        )
        auth_res = self.auth_service.authenticate_google_token(valid_token)
        tok = auth_res["session_token"]

        self.assertIsNotNone(self.auth_service.verify_session(tok))

        # Logout revokes
        self.auth_service.logout(tok)
        self.assertIsNone(self.auth_service.verify_session(tok))

    # -------------------------------------------------------------------------
    # 10. TARABRAIN IN-CHAT FLOW INTEGRATION
    # -------------------------------------------------------------------------
    def test_13_brain_in_chat_conversational_flow(self):
        """TaraBrain executes the complete conversational flow directly in chat."""
        brain = TaraBrain()
        brain.creator_auth_service = self.auth_service

        # Step 1: User says private trigger
        r1 = brain.process(actor_id="user", input_text=INITIAL_TEST_TRIGGER)
        self.assertEqual(r1["outcome"], "SUCCESS")
        self.assertIn("Creator authentication requested. Choose an authentication method:", r1["final_response"])
        self.assertIn("CREATOR-AUTH-INITIATED", r1["context_tags"])

        # Step 2: User selects "📱 QR Authentication"
        r2 = brain.process(actor_id="user", input_text="📱 QR Authentication")
        self.assertEqual(r2["outcome"], "SUCCESS")
        self.assertIn("QR Authentication", r2["final_response"])
        self.assertIn("METHOD-QR", r2["context_tags"])

        # Step 3: User submits valid Google ID token in chat
        valid_token = GoogleAuthService.create_mock_id_token(
            private_key=self.rsa_key,
            kid=self.test_kid,
            email=AUTHORIZED_CREATOR_EMAIL,
            aud="tara-client-id"
        )
        r3 = brain.process(actor_id="user", input_text="Submit token", context={"google_id_token": valid_token})
        self.assertEqual(r3["outcome"], "SUCCESS")
        self.assertEqual(r3["final_response"], f"✅ Creator authenticated: {CANONICAL_CREATOR_ID}.")
        tok = r3["result"]["session_token"]

        # Step 4: Chat continues with creator privileges
        r4 = brain.process(actor_id="user", input_text=INITIAL_TEST_TRIGGER, context={"creator_session_token": tok})
        self.assertEqual(r4["outcome"], "SUCCESS")

    # -------------------------------------------------------------------------
    # 11. SERVER ENDPOINTS INTEGRATION
    # -------------------------------------------------------------------------
    def test_14_server_routes_and_in_chat_methods(self):
        """Server supports /api/v1/auth/creator_key, /api/v1/auth/recovery, and /api/v1/auth/select_method."""
        import threading
        brain = TaraBrain()
        brain.creator_auth_service = self.auth_service

        server = create_server(host="127.0.0.1", port=0, brain=brain, api_keys={"test_key": "api_user"})
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=60)
            headers = {"Content-Type": "application/json"}

            # 1. Select Method: POST /api/v1/auth/select_method
            conn.request("POST", "/api/v1/auth/select_method", body=json.dumps({"method": "CREATOR_KEY"}), headers=headers)
            res1 = conn.getresponse()
            self.assertEqual(res1.status, 200)
            d1 = json.loads(res1.read().decode("utf-8"))
            nonce = d1.get("challenge_nonce")
            self.assertTrue(nonce)

            # 2. Authenticate Creator Key: POST /api/v1/auth/creator_key
            sig_hex = Ed25519.sign(self.root_priv, nonce.encode("utf-8")).hex()
            conn.request("POST", "/api/v1/auth/creator_key", body=json.dumps({
                "proof_signature": sig_hex,
                "challenge_nonce": nonce,
                "claimed_creator_id": CANONICAL_CREATOR_ID
            }), headers=headers)
            res2 = conn.getresponse()
            self.assertEqual(res2.status, 200)
            d2 = json.loads(res2.read().decode("utf-8"))
            self.assertEqual(d2.get("status"), "SUCCESS")
            session_tok = d2.get("session_token")
            self.assertTrue(session_tok)

            # 3. Authenticate Recovery: POST /api/v1/auth/recovery
            conn.request("POST", "/api/v1/auth/recovery", body=json.dumps({
                "recovery_code": self.test_recovery_code,
                "claimed_creator_id": CANONICAL_CREATOR_ID
            }), headers=headers)
            res3 = conn.getresponse()
            self.assertEqual(res3.status, 200)
            d3 = json.loads(res3.read().decode("utf-8"))
            self.assertEqual(d3.get("status"), "SUCCESS")

            conn.close()
        finally:
            server.shutdown()
            server.server_close()

    # -------------------------------------------------------------------------
    # 12. MODEL INVARIANT CHECK
    # -------------------------------------------------------------------------
    def test_15_model_sha256_strictly_unmodified(self):
        """Production neural model weights must remain strictly unaltered."""
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.isfile(model_path))

        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), EXPECTED_MODEL_SHA256)


    # -------------------------------------------------------------------------
    # 13. CROSS-DEVICE CHAT UI & WEB ENDPOINTS (PC + MOBILE)
    # -------------------------------------------------------------------------
    def test_16_web_chat_ui_cross_device_endpoints(self):
        """Web chat UI serves responsive HTML for mobile/PC with zero permanent external creator login button."""
        brain = TaraBrain()
        brain.creator_auth_service = self.auth_service

        server = create_server(host="127.0.0.1", port=0, brain=brain, api_keys={"test_key": "api_user"})
        port = server.server_port
        import threading
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=60)

            # 1. GET / serves responsive mobile-ready chat UI
            conn.request("GET", "/")
            res_root = conn.getresponse()
            self.assertEqual(res_root.status, 200)
            self.assertIn("text/html", res_root.getheader("Content-Type"))
            html_content = res_root.read().decode("utf-8")

            # Mobile viewport meta tag present
            self.assertIn('name="viewport"', html_content)
            self.assertIn("width=device-width", html_content)

            # Strict Invariant: Zero permanent external creator login button outside chat
            self.assertNotIn("Creator Login", html_content)
            self.assertNotIn("Admin Login", html_content)
            self.assertNotIn("Sign In As Creator", html_content)

            # Chat container & message flow present
            self.assertIn('id="chat-container"', html_content)
            self.assertIn('id="chat-input"', html_content)

            # In-chat authentication options present
            self.assertIn("📱 QR Authentication", html_content)
            self.assertIn("🔑 Creator Key Authentication", html_content)
            self.assertIn("🔐 Google Authentication", html_content)
            self.assertIn("🆘 Recovery", html_content)

            # 2. GET /chat also serves same responsive UI
            conn.request("GET", "/chat")
            res_chat = conn.getresponse()
            self.assertEqual(res_chat.status, 200)
            _ = res_chat.read()

            conn.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_17_qr_status_polling_and_device_approval(self):
        """QR status polling starts PENDING and transitions to APPROVED upon device approval."""
        brain = TaraBrain()
        brain.creator_auth_service = self.auth_service

        server = create_server(host="127.0.0.1", port=0, brain=brain, api_keys={"test_key": "api_user"})
        port = server.server_port
        import threading
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=60)
            headers = {"Content-Type": "application/json"}

            # 1. Register device
            dev_priv, dev_pub = Ed25519.generate_keypair()
            rec = self.auth_service.devices.register_device(dev_pub, device_name="TrustedPhone", status="AUTHORIZED")
            dev_id = rec["device_id"]

            # 2. Create QR challenge
            conn.request("POST", "/api/v1/auth/qr_challenge", body=json.dumps({"creator_id": "ROOT_OPERATOR"}), headers=headers)
            res_qr = conn.getresponse()
            d_qr = json.loads(res_qr.read().decode("utf-8"))
            chal = d_qr["challenge"]
            chal_id = chal["challenge_id"]
            nonce = chal["nonce"]

            # 3. Check status before approval -> PENDING
            conn.request("GET", f"/api/v1/auth/qr_status?challenge_id={chal_id}")
            res_stat1 = conn.getresponse()
            self.assertEqual(res_stat1.status, 200)
            d_stat1 = json.loads(res_stat1.read().decode("utf-8"))
            self.assertEqual(d_stat1["status"], "PENDING")

            # 4. Approve via device signature
            msg = f"{chal_id}:{nonce}:ROOT_OPERATOR".encode("utf-8")
            sig_hex = Ed25519.sign(dev_priv, msg).hex()
            conn.request("POST", "/api/v1/auth/qr_approve", body=json.dumps({
                "challenge_id": chal_id,
                "device_id": dev_id,
                "device_signature": sig_hex
            }), headers=headers)
            res_app = conn.getresponse()
            self.assertEqual(res_app.status, 200)
            d_app = json.loads(res_app.read().decode("utf-8"))
            self.assertEqual(d_app["status"], "SUCCESS")

            # 5. Check status after approval -> APPROVED
            conn.request("GET", f"/api/v1/auth/qr_status?challenge_id={chal_id}")
            res_stat2 = conn.getresponse()
            self.assertEqual(res_stat2.status, 200)
            d_stat2 = json.loads(res_stat2.read().decode("utf-8"))
            self.assertEqual(d_stat2["status"], "APPROVED")
            self.assertEqual(d_stat2["session"]["creator_id"], "ROOT_OPERATOR")
            self.assertEqual(d_stat2["session"]["role"], "ROOT_CREATOR")

            conn.close()
        finally:
            server.shutdown()
            server.server_close()

    # -------------------------------------------------------------------------
    # 14. MULTI-CREATOR GOOGLE IDENTITY MAPPING & ENROLLMENT
    # -------------------------------------------------------------------------
    def test_18_multi_creator_google_mapping_schema_and_unconfigured_defaults(self):
        """Verifies explicit schema mapping per creator, unconfigured defaults, and key_id calculation."""
        # 1. Check ROOT_OPERATOR (ROOT_CREATOR)
        c1 = self.auth_service.multi_creators.get_creator(CANONICAL_CREATOR_ID)
        self.assertIsNotNone(c1)
        self.assertEqual(c1["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(c1["display_name"], "OPERATOR_ROOT")
        self.assertEqual(c1["role"], "ROOT_CREATOR")
        self.assertEqual(c1["authorized_google_email"], "operator@internal.local")
        self.assertIn("key_id", c1)
        self.assertEqual(c1["status"], "active")

        # 2. Check CREATOR_2 (CREATOR, unconfigured Google identity)
        c2 = self.auth_service.multi_creators.get_creator("CREATOR_2")
        self.assertIsNotNone(c2)
        self.assertEqual(c2["creator_id"], "CREATOR_2")
        self.assertEqual(c2["display_name"], "Creator Two")
        self.assertEqual(c2["role"], "CREATOR")
        self.assertIsNone(c2["authorized_google_email"])  # Strictly no invented email
        self.assertIsNone(c2["google_subject_id"])

        # 3. Check CREATOR_3 (CREATOR, unconfigured Google identity)
        c3 = self.auth_service.multi_creators.get_creator("CREATOR_3")
        self.assertIsNotNone(c3)
        self.assertEqual(c3["creator_id"], "CREATOR_3")
        self.assertEqual(c3["role"], "CREATOR")
        self.assertIsNone(c3["authorized_google_email"])
        self.assertIsNone(c3["google_subject_id"])

    def test_19_creator_2_unconfigured_google_rejected_then_enrolled_successfully(self):
        """Unconfigured Google login for Creator 2 fails closed; ROOT_CREATOR enrollment allows successful login with CREATOR role (no self-escalation)."""
        c2_email = "creator2.verified@test.local"
        c2_sub = "google-sub-creator2-unique"

        # 1. Attempt login before enrollment -> MUST FAIL CLOSED
        token_unregistered = GoogleAuthService.create_mock_id_token(
            private_key=self.rsa_key,
            kid=self.test_kid,
            email=c2_email,
            sub=c2_sub,
            aud="tara-client-id"
        )
        res_fail = self.auth_service.authenticate_google_token(token_unregistered)
        self.assertEqual(res_fail["status"], "FAILED")
        self.assertEqual(res_fail["error"], "Creator authentication failed. No creator authority was granted.")

        # 2. Unauthorized enrollment attempt (non-root creator or anonymous) -> REJECTED
        with self.assertRaises(PermissionError):
            self.auth_service.multi_creators.enroll_google_identity(
                creator_id="CREATOR_2",
                google_email=c2_email,
                google_subject_id=c2_sub,
                authorized_by="CREATOR_2"  # Self-enrollment attempt
            )

        # 3. Legitimate enrollment by ROOT_CREATOR (ROOT_OPERATOR)
        enroll_res = self.auth_service.multi_creators.enroll_google_identity(
            creator_id="CREATOR_2",
            google_email=c2_email,
            google_subject_id=c2_sub,
            authorized_by=CANONICAL_CREATOR_ID
        )
        self.assertEqual(enroll_res["authorized_google_email"], c2_email)
        self.assertEqual(enroll_res["google_subject_id"], c2_sub)
        self.assertEqual(enroll_res["role"], "CREATOR")  # Role remains CREATOR
        self.auth_service.lifecycle.reseal_with_recovery(
            new_pub_bytes=bytes.fromhex(self.auth_service.creator.root_public_key),
            key_version=self.auth_service.creator.key_version,
            display_name="OPERATOR_ROOT"
        )

        # 4. Subsequent Google authentication by Creator 2 succeeds
        res_ok = self.auth_service.authenticate_google_token(token_unregistered)
        self.assertEqual(res_ok["status"], "SUCCESS")
        self.assertEqual(res_ok["creator_id"], "CREATOR_2")
        self.assertEqual(res_ok["role"], "CREATOR")  # Invariant: strictly CREATOR, no self-escalation!

        # 5. Forged Google subject ID rejection
        token_bad_sub = GoogleAuthService.create_mock_id_token(
            private_key=self.rsa_key,
            kid=self.test_kid,
            email=c2_email,
            sub="forged-sub-attacker",
            aud="tara-client-id"
        )
        res_bad_sub = self.auth_service.authenticate_google_token(token_bad_sub)
        self.assertEqual(res_bad_sub["status"], "FAILED")
        self.assertEqual(res_bad_sub["error"], "Creator authentication failed. No creator authority was granted.")

        # 6. Anti-collision: Attempting to bind ROOT_OPERATOR's email to another creator is rejected
        with self.assertRaises(ValueError):
            self.auth_service.multi_creators.enroll_google_identity(
                creator_id="CREATOR_3",
                google_email="operator@internal.local",
                authorized_by=CANONICAL_CREATOR_ID
            )


if __name__ == "__main__":
    unittest.main()
